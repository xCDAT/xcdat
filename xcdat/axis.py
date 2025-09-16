"""
Axis module for utilities related to axes, including functions to manipulate
coordinates.
"""

from typing import Literal

import numpy as np
import xarray as xr

from xcdat.utils import _if_multidim_dask_array_then_load

# https://cf-xarray.readthedocs.io/en/latest/coord_axes.html#axis-names
CFAxisKey = Literal["X", "Y", "T", "Z"]
# https://cf-xarray.readthedocs.io/en/latest/coord_axes.html#coordinate-names
CFStandardNameKey = Literal[
    "latitude", "longitude", "time", "vertical", "height", "pressure"
]

# A dictionary that maps the xCDAT `axis` arguments to keys used for `cf_xarray`
# accessor class indexing. For example, if we pass `axis="X"` to a function,
# we can fetch specific `cf_xarray` mapping tables such as `ds.cf.axes["X"]`
# or `ds.cf.coordinates["longitude"]`.
# More information: https://cf-xarray.readthedocs.io/en/latest/coord_axes.html
CF_ATTR_MAP: dict[CFAxisKey, dict[str, CFAxisKey | CFStandardNameKey]] = {
    "X": {"axis": "X", "coordinate": "longitude"},
    "Y": {"axis": "Y", "coordinate": "latitude"},
    "T": {"axis": "T", "coordinate": "time"},
    "Z": {"axis": "Z", "coordinate": "vertical"},
}

COORD_DEFAULT_ATTRS: dict[CFAxisKey, dict[str, str | CFAxisKey | CFStandardNameKey]] = {
    "X": dict(units="degrees_east", **CF_ATTR_MAP["X"]),
    "Y": dict(units="degrees_north", **CF_ATTR_MAP["Y"]),
    "T": dict(calendar="standard", **CF_ATTR_MAP["T"]),
    "Z": dict(**CF_ATTR_MAP["Z"]),
}

# A dictionary that maps common variable names to coordinate variables. This
# map is used as fall-back when coordinate variables don't have CF attributes
# set for ``cf_xarray`` to interpret using `CF_ATTR_MAP`.
VAR_NAME_MAP: dict[CFAxisKey, list[str]] = {
    "X": ["longitude", "lon"],
    "Y": ["latitude", "lat"],
    "T": ["time"],
    "Z": ["vertical", "height", "pressure", "lev", "plev"],
}


def get_dim_keys(obj: xr.Dataset | xr.DataArray, axis: CFAxisKey) -> str | list[str]:
    """Gets the dimension key(s) for an axis.

    Each dimension should have a corresponding dimension coordinate variable,
    which has a 1:1 map in keys and is denoted by the * symbol when printing out
    the xarray object.


    Parameters
    ----------
    obj : xr.Dataset | xr.DataArray
        The Dataset or DataArray object.
    axis : CFAxisKey
        The CF axis key ("X", "Y", "T", or "Z")

    Returns
    -------
    str | list[str]
        The dimension string or a list of dimensions strings for an axis.
    """
    dims = sorted([str(dim) for dim in get_dim_coords(obj, axis).dims])

    return dims[0] if len(dims) == 1 else dims


def get_dim_coords(
    obj: xr.Dataset | xr.DataArray, axis: CFAxisKey
) -> xr.Dataset | xr.DataArray:
    """Gets the dimension coordinates for an axis.

    This function uses ``cf_xarray`` to attempt to map the axis to its
    dimension coordinates by interpreting the CF axis and coordinate names
    found in the coordinate attributes. Refer to [1]_ for a list of CF axis and
    coordinate names that can be interpreted by ``cf_xarray``.

    If ``obj`` is an ``xr.Dataset,``, this function can return a single
    dimension coordinate variable as an ``xr.DataArray`` or multiple dimension
    coordinate variables in an ``xr Dataset``. If ``obj`` is an ``xr.DataArray``,
    this function should return a single dimension coordinate variable as an
    ``xr.DataArray``.

    Parameters
    ----------
    obj : xr.Dataset | xr.DataArray
        The Dataset or DataArray object.
    axis : CFAxisKey
        The CF axis key ("X", "Y", "T", "Z").

    Returns
    -------
    xr.Dataset | xr.DataArray
        A Dataset of dimension coordinate variables or a DataArray for
        the single dimension coordinate variable.

    Raises
    ------
    ValueError
        If the ``obj`` is an ``xr.DataArray`` and more than one dimension is
        mapped to the same axis.
    KeyError
        If no dimension coordinate variables were found for the ``axis``.

    Notes
    -----
    Multidimensional coordinates are ignored.

    References
    ----------
    .. [1] https://cf-xarray.readthedocs.io/en/latest/coord_axes.html#axes-and-coordinates
    """
    # Get the object's index keys, with each being a dimension.
    # NOTE: xarray does not include multidimensional coordinates as index keys.
    # Example: ["lat", "lon", "time"]
    index_keys = obj.indexes.keys()

    # Attempt to map the axis it all of its coordinate variable(s) using the
    # axis and coordinate names in the object attributes (if they are set).
    # Example: Returns ["time", "time_centered"] with `axis="T"`
    coord_keys = _get_all_coord_keys(obj, axis)
    # Filter the index keys to just the dimension coordinate keys.
    # Example: Returns ["time"], since "time_centered" is not in `index_keys`
    dim_coord_keys = list(set(index_keys) & set(coord_keys))

    if isinstance(obj, xr.DataArray) and len(dim_coord_keys) > 1:
        raise ValueError(
            f"This DataArray has more than one dimension {dim_coord_keys} mapped to the "
            f"'{axis}' axis, which is an unexpected behavior. Try dropping extraneous "
            "dimensions from the DataArray first (might affect data shape)."
        )

    if len(dim_coord_keys) == 0:
        raise KeyError(
            f"No '{axis}' axis dimension coordinate variables were found in the "
            f"xarray object. Make sure dimension coordinate variables exist, they are "
            "one dimensional, and their CF 'axis' or 'standard_name' attrs are "
            "correctly set."
        )

    dim_coords = obj[
        dim_coord_keys if len(dim_coord_keys) > 1 else dim_coord_keys[0]
    ].copy()

    return dim_coords


def get_coords_by_name(obj: xr.Dataset | xr.DataArray, axis: CFAxisKey) -> xr.DataArray:
    """Retrieve the coordinate variable based on its name.

    This method is useful for returning the desired coordinate in the following
    cases:

    - Coordinates that are not CF-compliant (e.g., missing CF attributes like
      "axis", "standard_name", or "long_name") but use common names.
    - Axes with multiple sets of coordinates. For example, curvilinear grids may
      have multiple coordinates for the same axis (e.g., (nlat, lat) for X and
      (nlon, lon) for Y). In most cases, "lat" and "lon" are the desired
      coordinates, which this function will return.

    Common variable names for each axis (from ``VAR_NAME_MAP``):

    - "X" axis: ["longitude", "lon"]
    - "Y" axis: ["latitude", "lat"]
    - "T" axis: ["time"]
    - "Z" axis: ["vertical", "height", "pressure", "lev", "plev"]

    Parameters
    ----------
    axis : CFAxisKey
        The CF axis key ("X", "Y", "T", or "Z").

    Returns
    -------
    xr.DataArray
        The coordinate variable.

    Raises
    ------
    KeyError
        If the coordinate variable is not found in the dataset.
    """
    coord_names = VAR_NAME_MAP[axis]

    for coord_name in coord_names:
        if coord_name in obj.coords:
            coord = obj.coords[coord_name]

            return coord

    raise KeyError(f"Coordinate with name '{axis}' not found in the dataset.")


def center_times(dataset: xr.Dataset) -> xr.Dataset:
    """Centers time coordinates using the midpoint between time bounds.

    Time coordinates can be recorded using different intervals, including
    the beginning, middle, or end of the interval. Centering time
    coordinates, ensures calculations using these values are performed
    reliably regardless of the recorded interval.

    This method attempts to get bounds for each time variable using the
    CF "bounds" attribute. Coordinate variables that cannot be mapped to
    bounds will be skipped.

    Parameters
    ----------
    dataset : xr.Dataset
        The Dataset with original time coordinates.

    Returns
    -------
    xr.Dataset
        The Dataset with centered time coordinates.
    """
    ds = dataset.copy()
    coords = get_dim_coords(ds, "T")

    for coord in coords.coords.values():
        try:
            coord_bounds = ds.bounds.get_bounds("T", str(coord.name))
        except KeyError:
            coord_bounds = None

        if coord_bounds is not None:
            lower_bounds, upper_bounds = (
                coord_bounds[:, 0].data,
                coord_bounds[:, 1].data,
            )
            bounds_diffs: np.timedelta64 = (upper_bounds - lower_bounds) / 2
            bounds_mids: np.ndarray = lower_bounds + bounds_diffs

            coord_centered = xr.DataArray(
                name=coord.name,
                data=bounds_mids,
                dims=coord.dims,
                attrs=coord.attrs,
            )
            coord_centered.encoding = coord.encoding
            ds = ds.assign_coords({coord.name: coord_centered})

            # Update time bounds with centered time coordinates.
            coord_bounds[coord_centered.name] = coord_centered
            ds[coord_bounds.name] = coord_bounds

    return ds


def swap_lon_axis(
    dataset: xr.Dataset, to: tuple[float, float], sort_ascending: bool = True
) -> xr.Dataset:
    """Swaps the orientation of a dataset's longitude axis.

    This method also swaps the axis orientation of the longitude bounds if it
    exists. Afterwards, it sorts longitude and longitude bounds values in
    ascending order.

    Note, based on how datasets are chunked, swapping the longitude dimension
    and sorting might raise ``PerformanceWarning: Slicing is producing a
    large chunk. To accept the large chunk and silence this warning, set the
    option...``. This function uses xarray's arithmetic to swap orientations,
    so this warning seems potentially unavoidable.

    Parameters
    ----------
    dataset : xr.Dataset
         The Dataset containing a longitude axis.
    to : tuple[float, float]
        The orientation to swap the Dataset's longitude axis to. Supported
        orientations include:

        * (-180, 180): represents [-180, 180) in math notation
        * (0, 360): represents [0, 360) in math notation
    sort_ascending : bool
        After swapping, sort in ascending order (True), or keep existing order
        (False).

    Returns
    -------
    xr.Dataset
        The Dataset with swapped lon axes orientation.
    """
    ds = dataset.copy()
    coords = get_dim_coords(ds, "X").coords
    coord_keys = list(coords.keys())

    # Attempt to swap the orientation for longitude coordinates.
    for key in coord_keys:
        new_coord = _swap_lon_axis(ds.coords[key], to)

        if ds.coords[key].identical(new_coord):
            continue

        ds.coords[key] = new_coord

    try:
        bounds = ds.bounds.get_bounds("X")
    except KeyError:
        bounds = None

    if isinstance(bounds, xr.DataArray):
        ds = _swap_lon_bounds(ds, str(bounds.name), to)
    elif isinstance(bounds, xr.Dataset):
        for key in bounds.data_vars.keys():
            ds = _swap_lon_bounds(ds, str(key), to)

    if sort_ascending:
        ds = ds.sortby(list(coords.dims), ascending=True)

    return ds


def _get_all_coord_keys(obj: xr.Dataset | xr.DataArray, axis: CFAxisKey) -> list[str]:
    """Gets all dimension and non-dimension coordinate keys for an axis.

    This function uses ``cf_xarray`` to interpret CF axis and coordinate name
    metadata to map an ``axis`` to its coordinate keys. Refer to [2]_ for more
    information on the ``cf_xarray`` mapping tables.

    It also loops over a list of statically defined coordinate variable names to
    see if they exist in the object, and appends keys that do exist.

    Parameters
    ----------
    obj : xr.Dataset | xr.DataArray
        The Dataset or DataArray object.
    axis : CFAxisKey
        The CF axis key ("X", "Y", "T", or "Z").

    Returns
    -------
    list[str]
        The axis coordinate variable keys.

    References
    ----------
    .. [2] https://cf-xarray.readthedocs.io/en/latest/coord_axes.html#axes-and-coordinates
    """
    cf_attrs = CF_ATTR_MAP[axis]
    var_names = VAR_NAME_MAP[axis]

    keys: list[str] = []

    try:
        keys = keys + obj.cf.axes[cf_attrs["axis"]]
    except KeyError:
        pass

    try:
        keys = keys + obj.cf.coordinates[cf_attrs["coordinate"]]
    except KeyError:
        pass

    for name in var_names:
        if name in obj.coords.keys():
            keys.append(name)

    return list(set(keys))


def _swap_lon_bounds(ds: xr.Dataset, key: str, to: tuple[float, float]):
    bounds = ds[key].copy()
    new_bounds = _swap_lon_axis(bounds, to)

    if not ds[key].identical(new_bounds):
        ds_new = ds.copy()
        ds_new[key] = new_bounds

        # Handle cases where a prime meridian cell exists after converting
        # longitude bounds to the range (0, 360). This involves extending
        # longitude bounds and data variables by one cell to account for the
        # prime meridian, updating bounds interval for the prime meridian cell,
        # and then trimming the extra cell to maintain the original shape.
        if to == (0, 360):
            p_meridian_index = _get_prime_meridian_index(ds_new[key])

            if p_meridian_index is not None:
                ds_new = _adjust_bounds_for_prime_meridian(
                    ds_new, key, p_meridian_index
                )

        return ds_new

    return ds


def _swap_lon_axis(coords: xr.DataArray, to: tuple[float, float]) -> xr.DataArray:
    """Swaps the axis orientation for longitude coordinates.

    Parameters
    ----------
    coords : xr.DataArray
        Coordinates on a longitude axis.
    to : tuple[float, float]
        The new longitude axis orientation.

    Returns
    -------
    xr.DataArray
        The longitude coordinates the opposite axis orientation If the
        coordinates are already on the specified axis orientation, the same
        coordinates are returned.
    """
    with xr.set_options(keep_attrs=True):
        if to == (-180, 180):
            # FIXME: Performance warning produced after swapping and then sorting
            # based on how datasets are chunked.
            new_coords = ((coords + 180) % 360) - 180
        elif to == (0, 360):
            # Example with 180 coords: [-180, -0, 179] -> [0, 180, 360]
            # Example with 360 coords: [60, 150, 360] -> [60, 150, 0]
            # FIXME: Performance warning produced after swapping and then sorting
            # based on how datasets are chunked.
            new_coords = coords % 360

            # Check if the original coordinates contain an element with a value
            # of 360. If this element exists, use its index to revert its
            # swapped value of 0 (360 % 360 is 0) back to 360. This case usually
            # happens if the coordinate are already on the (0, 360) axis
            # orientation.
            # Example with 360 coords: [60, 150, 0] -> [60, 150, 360]
            index_with_360 = np.where(coords == 360)

            if index_with_360[0].size > 0:
                _if_multidim_dask_array_then_load(new_coords)

                new_coords[index_with_360] = 360
        else:
            raise ValueError(
                "Currently, only (-180, 180) and (0, 360) are supported longitude axis "
                "orientations."
            )

    new_coords.encoding = coords.encoding

    return new_coords


def _get_prime_meridian_index(lon_bounds: xr.DataArray) -> np.ndarray | None:
    """Gets the index of the prime meridian cell in the longitude bounds.

    A prime meridian cell can exist when converting the axis orientation
    from [-180, 180) to [0, 360).

    Parameters
    ----------
    lon_bounds : xr.DataArray
        The longitude bounds.

    Returns
    -------
    np.ndarray | None
        An array with a single element representing the index of the prime
        meridian index if it exists. Otherwise, None if the cell does not exist.

    Raises
    ------
    ValueError
        If more than one grid cell spans the prime meridian.
    """
    p_meridian_index = np.where(lon_bounds[:, 1] - lon_bounds[:, 0] < 0)[0]

    if p_meridian_index.size == 0:  # pragma:no cover
        return None
    elif p_meridian_index.size > 1:
        raise ValueError("More than one grid cell spans prime meridian.")

    return p_meridian_index


def _adjust_bounds_for_prime_meridian(
    ds: xr.Dataset, key: str, p_meridian_index: np.ndarray
) -> xr.Dataset:
    """Adjusts bounds for the prime meridian cell in longitude coordinates.

    This function modifies the longitude bounds to handle cases where a grid
    cell spans the prime meridian. It ensures that the longitude bounds are
    aligned to the (0, 360) range by splitting the prime meridian cell into
    two parts and adjusting the bounds accordingly. The extra cell created
    during this process is removed to restore the original shape of the dataset.

    Parameters
    ----------
    ds : xr.Dataset
        The Dataset containing longitude bounds and coordinates.
    key : str
        The name of the longitude bounds variable in the dataset.
    p_meridian_index : np.ndarray
        An array with a single element representing the index of the prime
        meridian cell.

    Returns
    -------
    xr.Dataset
        The Dataset with adjusted longitude bounds for the prime meridian cell.
    """
    ds_new = ds.copy()

    ds_new = _align_lon_to_360(ds_new, ds_new[key], p_meridian_index)
    dim = get_dim_keys(ds_new[key], "X")

    # NOTE: The logic for calculating new bounds is similar to the
    # implementation in xcdat.bounds.create_bounds.
    # Calculate the bounds interval for the prime meridian cell using
    # the cell as the midpoint and the difference of the bounds.
    bounds = ds_new[key].isel({dim: 0})
    bounds_dim = _get_bounds_dim(ds_new[dim], ds_new[key])
    bounds_delta = abs(bounds.diff(bounds_dim).item()) / 2

    # Create new bounds for the prime meridian cell.
    midpoint = ds_new[dim][p_meridian_index].item()
    new_lower = midpoint - bounds_delta
    new_upper = midpoint + bounds_delta

    ds_new[key][p_meridian_index, :] = [new_lower, new_upper]

    # Remove the extra cell to restore the original shape.
    ds_new = ds_new.isel({dim: slice(0, -1)})

    return ds_new


def _align_lon_to_360(
    ds: xr.Dataset,
    lon_bounds: xr.DataArray,
    p_meridian_index: np.ndarray,
) -> xr.Dataset:
    """Handles a prime meridian cell to align longitude axis to (0, 360).

    This method ensures the domain bounds are within 0 to 360 by handling
    the grid cell that encompasses the prime meridian (e.g., [359, 1]).

    First, it handles the prime meridian cell within the longitude axis bounds
    by splitting the cell into two parts (one east and one west of the prime
    meridian, refer to `_align_lon_bounds_to_360()` for more information). Then
    it concatenates the 360 coordinate point to the longitude coordinates to
    handle the addition of the extra grid cell from the previous step. Finally,
    for each data variable associated with the longitude axis, the value of the
    data variable at the prime meridian cell is concatenated to the data
    variable.

    Parameters
    ----------
    dataset : xr.Dataset
        The Dataset.
    p_meridian_index : np.ndarray
        An array with a single element representing the index of the prime
        meridian cell.

    Returns
    -------
    xr.Dataset
        The Dataset.
    """
    dim = get_dim_keys(lon_bounds, "X")

    # Create a dataset to store updated longitude variables.
    ds_lon = xr.Dataset()

    # Align the the longitude bounds to the 360 orientation using the prime
    # meridian index. This function splits the grid cell into two parts (east
    # and west), which appends an extra set of bounds for the 360 coordinate.
    ds_lon[lon_bounds.name] = _align_lon_bounds_to_360(lon_bounds, p_meridian_index)

    # After appending the extra set of bounds, update the last coordinate from
    # 0 to 360.
    for key, coord in ds_lon.coords.items():
        coord.values[-1] = 360
        ds_lon[key] = coord

    # Get the data variables related to the longitude axis and concatenate each
    # with the value at the prime meridian.
    for key, var in ds.cf.data_vars.items():
        if key != lon_bounds.name and dim in var.dims:
            # Concatenate the prime meridian cell to the variable
            p_meridian_val = var.isel({dim: p_meridian_index}).copy()
            new_var = xr.concat((var, p_meridian_val), dim=dim)

            # Update the longitude coordinates for the variable.
            new_var[dim] = ds_lon[dim]
            ds_lon[var.name] = new_var

    # Create a new dataset of non-longitude vars and updated longitude vars.
    ds_no_lon = ds.get([v for v in ds.data_vars if dim not in ds[v].dims])  # type: ignore
    ds_final = xr.merge((ds_no_lon, ds_lon))

    return ds_final


def _align_lon_bounds_to_360(
    bounds: xr.DataArray, p_meridian_index: np.ndarray
) -> xr.DataArray:
    """Handles a prime meridian cell to align longitude bounds axis to (0, 360).

    This method ensures the domain bounds are within 0 to 360 by handling
    the grid cell that encompasses the prime meridian (e.g., [359, 1]). In
    this case, calculating longitudinal weights is complicated because the
    weights are determined by the difference of the bounds.

    If this situation exists, the method will split this grid cell into
    two parts (one east and west of the prime meridian). The original
    grid cell will have domain bounds extending east of the prime meridian
    and an extra set of bounds will be concatenated to ``bounds``
    corresponding to the domain bounds west of the prime meridian. For
    instance, a grid cell spanning -1 to 1, will be broken into a cell
    from 0 to 1 and 359 to 360 (or -1 to 0).

    Parameters
    ----------
    bounds : xr.DataArray
        The longitude domain bounds with prime meridian cell.
    p_meridian_index : np.ndarray
        The index of the prime meridian cell.

    Returns
    -------
    xr.DataArray
        The longitude domain bounds with split prime meridian cell.

    Raises
    ------
    ValueError
        If longitude bounds are inclusively between 0 and 360.
    """
    _if_multidim_dask_array_then_load(bounds)

    # Example array: [[359, 1], [1, 90], [90, 180], [180, 359]]
    # Reorient bound to span across zero (i.e., [359, 1] -> [-1, 1]).
    # Result: [[-1, 1], [1, 90], [90, 180], [180, 359]]
    bounds[p_meridian_index, 0] = bounds[p_meridian_index, 0] - 360.0

    # Extend the array to nlon+1 by concatenating the grid cell that
    # spans the prime meridian to the end.
    # Result: [[-1, 1], [1, 90], [90, 180], [180, 359], [-1, 1]]
    dim = get_dim_keys(bounds, "X")
    bounds = xr.concat((bounds, bounds[p_meridian_index, :]), dim=dim)

    # Add an equivalent bound that spans 360
    # (i.e., [-1, 1] -> [359, 361]) to the end of the array.
    # Result: [[-1, 1], [1, 90], [90, 180], [180, 359], [359, 361]]
    repeat_bound = bounds[p_meridian_index, :][0] + 360.0
    bounds[-1, :] = repeat_bound

    # Update the lower-most min and upper-most max bounds to [0, 360].
    # Result: [[0, 1], [1, 90], [90, 180], [180, 359], [359, 360]]
    bounds[p_meridian_index, 0], bounds[-1, 1] = (0.0, 360.0)

    return bounds


def _get_bounds_dim(da_coords: xr.DataArray, da_bounds: xr.DataArray) -> str:
    """Identify the bounds dimension in the given bounds DataArray.

    This function determines which dimension in the bounds DataArray
    represents the bounds (e.g., lower and upper limits) for each value
    in the coordinate DataArray. According to the CF Conventions, a boundary
    variable will have one more dimension than its associated coordinate or
    auxiliary coordinate variable. Refer to [6] for more details.

    Parameters
    ----------
    da_coords : xr.DataArray
        The coordinate DataArray.
    da_bounds : xr.DataArray
        The bounds DataArray.

    Returns
    -------
    str
        The name of the bounds dimension.

    Raises
    ------
    ValueError
        If the bounds variable does not have exactly one more dimension than
        the coordinate variable, or if no valid bounds dimension is found.

    References
    ----------
    .. [6] https://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/cf-conventions.html#cell-boundaries
    """
    bounds_dim = [dim for dim in da_bounds.dims if dim not in da_coords.dims]

    if len(bounds_dim) == 1:
        return str(bounds_dim[0])
    elif len(bounds_dim) == 0:
        raise ValueError(
            "No extra dimension found in bounds variable compared to coordinate "
            f"variable. Coordinate dims: {da_coords.dims}, bounds dims: "
            f"{da_bounds.dims}"
        )
    else:
        raise ValueError(
            f"Bounds variable must have exactly one more dimension than the coordinate "
            f"variable. Coordinate dims: {da_coords.dims}, bounds dims: {da_bounds.dims}"
        )
