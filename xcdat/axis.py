"""
Axis module for utilities related to axes, including functions to manipulate
coordinates.
"""

from typing import Literal

import numpy as np
import xarray as xr

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
    Raises
    ------
    ValueError
        If the ``to`` argument is not one of the supported orientations,
        including (-180, 180) or (0, 360).
    """
    ds = dataset.copy()
    coords = get_dim_coords(ds, "X").coords
    coord_keys = list(coords.keys())

    if to not in [(-180, 180), (0, 360)]:
        raise ValueError(
            "Only (-180, 180) and (0, 360) longitude ranges are supported."
        )

    # Attempt to swap the orientation for longitude coordinates and bounds (if
    # they exist).
    for key in coord_keys:
        ds[key] = _orient_lon_coords(ds.coords[key], to)

        try:
            bounds = ds.bounds.get_bounds("X", ds[key].name)
        except KeyError:
            bounds = None

        if isinstance(bounds, xr.DataArray):
            ds[bounds.name] = _orient_lon_bounds(ds[key], bounds, to)

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


def _orient_lon_coords(coords: xr.DataArray, to: tuple[float, float]) -> xr.DataArray:
    """
    Adjust longitude coordinates to match the specified orientation range.

    This function ensures that longitude centers are mapped correctly to the
    target orientation range and avoids introducing a literal 360 center. If
    the coordinates are already in the desired orientation, they are returned
    as-is (idempotent). Longitude centers are treated as half-open in the range
    [0, 360), ensuring no literal 360 center is introduced.

    Parameters
    ----------
    coords : xr.DataArray
        The longitude coordinates to be reoriented.
    to : tuple[float, float]
        The orientation to swap the Dataset's longitude axis to. Supported
        orientations include:

        * (-180, 180): represents [-180, 180) in math notation
        * (0, 360): represents [0, 360) in math notation

    Returns
    -------
    xr.DataArray
        The longitude coordinates reoriented to the target range, with the
        original encoding preserved.
    """
    with xr.set_options(keep_attrs=True):
        if _is_in_orientation(coords, to):
            out = coords
        else:
            out = _map_to_orientation(coords, to)

    out.encoding = coords.encoding
    return out


def _orient_lon_bounds(
    da_coords: xr.DataArray, da_bounds: xr.DataArray, to: tuple[float, float]
) -> xr.DataArray:
    """Map longitude bounds to the target orientation.

    This function adjusts the longitude bounds of a dataset to match the specified
    orientation. It ensures idempotency, meaning if the bounds are already in the
    target orientation, they are returned as-is. For the (0, 360) orientation, it
    optionally normalizes seam wraps (e.g., [359, 0] → [359, 360]).

    Parameters
    ----------
    da_coords : xr.DataArray
        The longitude coordinate values of the dataset.
    da_bounds : xr.DataArray
        The longitude bounds of the dataset.
    to : tuple[float, float]
        The orientation to swap the Dataset's longitude axis to. Supported
        orientations include:

        * (-180, 180): represents [-180, 180) in math notation
        * (0, 360): represents [0, 360) in math notation

    Returns
    -------
    xr.DataArray
        The longitude bounds adjusted to the target orientation.
    """
    with xr.set_options(keep_attrs=True):
        if _is_in_orientation(da_bounds, to):
            out = da_bounds
        else:
            out = _map_to_orientation(da_bounds, to)

            # Adjusts wrap-around bounds in a DataArray from [0, 360) longitude
            # range. Example: [359, 0] -> [359, 360].
            if to == (0, 360):
                bounds_dim = _get_bounds_dim(da_coords, out)
                out = _normalize_wrap_bounds_0_to_360(out, bounds_dim)

    out.encoding = da_bounds.encoding

    return out


def _is_in_orientation(da: xr.DataArray, to: tuple[float, float]) -> bool:
    """
    Check if the values in a DataArray conform to a specified orientation range.

    Parameters
    ----------
    da : xr.DataArray
        The input data array to check.
    to : tuple[float, float]
        The orientation to swap the Dataset's longitude axis to. Supported
        orientations include:

        * (-180, 180): represents [-180, 180) in math notation
        * (0, 360): represents [0, 360) in math notation

    Returns
    -------
    bool
        True if all values in `da` fall within the specified range, False otherwise.
    """
    if to == (-180, 180):
        return bool(np.all(da >= -180) and np.all(da <= 180))

    return bool(np.all(da >= 0) and np.all(da <= 360))


def _map_to_orientation(da: xr.DataArray, to: tuple[float, float]) -> xr.DataArray:
    """
    Map the values of a DataArray to a specified longitude orientation.

    Parameters
    ----------
    da : xr.DataArray
        The input DataArray containing longitude values.
    to : tuple[float, float]
        The orientation to swap the Dataset's longitude axis to. Supported
        orientations include:

        * (-180, 180): represents [-180, 180) in math notation
        * (0, 360): represents [0, 360) in math notation

    Returns
    -------
    xr.DataArray
        The DataArray with longitude values mapped to the specified range.

    Notes
    -----
    If a dataset includes both -180 and +180, this will create a duplicate 180
    when converting to [0, 360). We may need to drop or remap one of them in the
    future if this becomes an issue. Datasets that follow CF convetion typically
    do not run into this issue, as the longitude centers are within the
    half-open interval [0, 360).
    """
    if to == (-180, 180):
        return ((da + 180) % 360) - 180

    return da % 360


def _normalize_wrap_bounds_0_to_360(
    da_bounds: xr.DataArray, bounds_dim: str
) -> xr.DataArray:
    """
    Adjusts wrap-around bounds in a DataArray from [0, 360) longitude range.

    This function modifies rows in the input DataArray where the bounds wrap
    around the seam (e.g., [359, 0]) to ensure continuity by converting them
    to [359, 360]. Rows that do not wrap are left unchanged.

    Parameters
    ----------
    da_bounds : xr.DataArray
        The input DataArray containing bounds to normalize. It is expected
        to have a dimension specified by `bounds_dim` with two elements
        representing the lower and upper bounds.
    bounds_dim : str
        The name of the dimension in `da_bounds` that represents the bounds.

    Returns
    -------
    xr.DataArray
        A copy of the input DataArray with normalized wrap-around bounds.
    """
    lo = da_bounds.sel({bounds_dim: 0})
    hi = da_bounds.sel({bounds_dim: 1})

    wrap = lo > hi

    out = da_bounds.copy()
    out.loc[{bounds_dim: 1}] = xr.where(wrap, 360.0, hi)

    return out


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
