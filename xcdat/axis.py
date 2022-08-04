"""
Axis module for utilities related to axes, including functions to manipulate
coordinates.
"""
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import xarray as xr
from dask.array.core import Array

# https://cf-xarray.readthedocs.io/en/latest/coord_axes.html#axis-names
CFAxisName = Literal["X", "Y", "T", "Z"]
# https://cf-xarray.readthedocs.io/en/latest/coord_axes.html#coordinate-names
CFStandardName = Literal["latitude", "longitude", "time", "height", "pressure"]
ShortName = Literal["lat", "lon"]

# The key is the accepted value for method and function arguments, and the
# values are the CF-compliant axis and standard names that are interpreted in
# the dataset.
CF_NAME_MAP: Dict[CFAxisName, List[Union[CFAxisName, CFStandardName, ShortName]]] = {
    "X": ["X", "longitude", "lon"],
    "Y": ["Y", "latitude", "lat"],
    "T": ["T", "time"],
    "Z": ["Z", "height", "pressure"],
}


def get_axis_coord(
    obj: Union[xr.Dataset, xr.DataArray], axis: CFAxisName
) -> xr.DataArray:
    """Gets the coordinate variable for an axis.

    This function uses ``cf_xarray`` to try to find the matching coordinate
    variable by checking the following attributes in order:

    - ``"axis"``
    - ``"standard_name"``
    - Dimension name

      - Must follow the valid short-hand convention
      - For example, ``"lat"`` for latitude and ``"lon"`` for longitude

    Parameters
    ----------
    obj : Union[xr.Dataset, xr.DataArray]
        The Dataset or DataArray object.
    axis : CFAxisName
        The CF-compliant axis name ("X", "Y", "T", "Z").

    Returns
    -------
    xr.DataArray
        The coordinate variable.

    Raises
    ------
    KeyError
        If the coordinate variable was not found.

    Notes
    -----
    Refer to [1]_ for a list of CF-compliant ``"axis"`` and ``"standard_name"``
    attr names that can be interpreted by ``cf_xarray``.

    References
    ----------

    .. [1] https://cf-xarray.readthedocs.io/en/latest/coord_axes.html#axes-and-coordinates
    """
    keys = CF_NAME_MAP[axis]
    coord_var = None

    for key in keys:
        try:
            coord_var = obj.cf[key]
            break
        except KeyError:
            pass

    if coord_var is None:
        raise KeyError(
            f"A coordinate variable for the {axis} axis was not found. Make sure "
            "the coordinate variable exists and either the (1) 'axis' attr or (2) "
            "'standard_name' attr is set, or (3) the dimension name follows the "
            "short-hand convention (e.g.,'lat')."
        )
    return coord_var


def get_axis_dim(obj: Union[xr.Dataset, xr.DataArray], axis: CFAxisName) -> str:
    """Gets the dimension for an axis.

    The coordinate name should be identical to the dimension name, so this
    function simply returns the coordinate name.

    Parameters
    ----------
    obj : Union[xr.Dataset, xr.DataArray]
        The Dataset or DataArray object.
    axis : CFAxisName
        The CF-compliant axis name ("X", "Y", "T", "Z")

    Returns
    -------
    str
        The dimension for an axis.
    """
    return str(get_axis_coord(obj, axis).name)


def center_times(dataset: xr.Dataset) -> xr.Dataset:
    """Centers time coordinates using the midpoint between time bounds.

    Time coordinates can be recorded using different intervals, including
    the beginning, middle, or end of the interval. Centering time
    coordinates, ensures calculations using these values are performed
    reliably regardless of the recorded interval.

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
    time: xr.DataArray = get_axis_coord(ds, "T")
    time_bounds = ds.bounds.get_bounds("T")

    lower_bounds, upper_bounds = (time_bounds[:, 0].data, time_bounds[:, 1].data)
    bounds_diffs: np.timedelta64 = (upper_bounds - lower_bounds) / 2
    bounds_mids: np.ndarray = lower_bounds + bounds_diffs

    time_centered = xr.DataArray(
        name=time.name,
        data=bounds_mids,
        coords={"time": bounds_mids},
        attrs=time.attrs,
    )
    time_centered.encoding = time.encoding
    ds = ds.assign_coords({"time": time_centered})

    # Update time bounds with centered time coordinates.
    time_bounds[time_centered.name] = time_centered
    ds[time_bounds.name] = time_bounds
    return ds


def swap_lon_axis(
    dataset: xr.Dataset, to: Tuple[float, float], sort_ascending: bool = True
) -> xr.Dataset:
    """Swaps the orientation of a dataset's longitude axis.

    This method also swaps the axis orientation of the longitude bounds if it
    exists. Afterwards, it sorts longitude and longitude bounds values in
    ascending order.

    Parameters
    ----------
    dataset : xr.Dataset
         The Dataset containing a longitude axis.
    to : Tuple[float, float]
        The orientation to swap the Dataset's longitude axis to.

        Supported orientations:

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
    lon: xr.DataArray = get_axis_coord(ds, "X").copy()
    lon_bounds: xr.DataArray = dataset.bounds.get_bounds("X").copy()

    with xr.set_options(keep_attrs=True):
        if to == (-180, 180):
            new_lon = ((lon + 180) % 360) - 180
            new_lon_bounds = ((lon_bounds + 180) % 360) - 180
            ds = _reassign_lon(ds, new_lon, new_lon_bounds)
        elif to == (0, 360):
            new_lon = lon % 360
            new_lon_bounds = lon_bounds % 360
            ds = _reassign_lon(ds, new_lon, new_lon_bounds)

            # Handle cases where a prime meridian cell exists, which can occur
            # after swapping to (0, 360).
            p_meridian_index = _get_prime_meridian_index(new_lon_bounds)
            if p_meridian_index is not None:
                ds = _align_lon_to_360(ds, p_meridian_index)
        else:
            raise ValueError(
                "Currently, only (-180, 180) and (0, 360) are supported longitude axis "
                "orientations."
            )

    # If the swapped axis orientation is the same as the existing axis
    # orientation, return the original Dataset.
    if new_lon.identical(lon):
        return dataset

    if sort_ascending:
        ds = ds.sortby(new_lon.name, ascending=True)

    return ds


def _reassign_lon(dataset: xr.Dataset, lon: xr.DataArray, lon_bounds: xr.DataArray):
    """
    Reassign longitude coordinates and bounds to the Dataset after swapping the
    orientation.

    Parameters
    ----------
    dataset : xr.Dataset
        The Dataset.
    lon : xr.DataArray
        The swapped longitude coordinates.
    lon_bounds : xr.DataArray
        The swapped longitude bounds.

    Returns
    -------
    xr.Dataset
        The Dataset with swapped longitude coordinates and bounds.
    """
    lon[lon.name] = lon_bounds[lon.name] = lon

    dataset[lon.name] = lon
    dataset[lon_bounds.name] = lon_bounds
    return dataset


def _align_lon_to_360(dataset: xr.Dataset, p_meridian_index: np.ndarray) -> xr.Dataset:
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
    ds = dataset.copy()
    lon: xr.DataArray = get_axis_coord(ds, "X")
    lon_bounds: xr.DataArray = dataset.bounds.get_bounds("X")

    # If chunking, must convert the xarray data structure from lazy
    # Dask arrays into eager, in-memory NumPy arrays before performing
    # manipulations on the data. Otherwise, it raises `NotImplementedError
    # xarray can't set arrays with multiple array indices to dask yet`.
    if isinstance(lon_bounds.data, Array):
        lon_bounds.load()

    # Align the the longitude bounds using the prime meridian index.
    lon_bounds = _align_lon_bounds_to_360(lon_bounds, p_meridian_index)

    # Concatenate the longitude coordinates with 360 to handle the prime
    # meridian cell and update the coordinates for the longitude bounds.
    p_meridian_cell = xr.DataArray([360.0], coords={lon.name: [360.0]}, dims=[lon.name])
    lon = xr.concat((lon, p_meridian_cell), dim=lon.name)
    lon_bounds[lon.name] = lon

    # Get the data variables related to the longitude axis and concatenate each
    # with the value at the prime meridian.
    lon_vars = {}
    for key, value in ds.cf.data_vars.items():
        if key != lon_bounds.name and lon.name in value.dims:
            lon_vars[key] = value

    for name, var in lon_vars.items():
        p_meridian_val = var.isel({lon.name: p_meridian_index})
        new_var = xr.concat((var, p_meridian_val), dim=lon.name)
        new_var[lon.name] = lon
        lon_vars[name] = new_var

    # Create a Dataset with longitude data vars and merge it to the Dataset
    # without longitude data vars.
    ds_lon = xr.Dataset(data_vars={**lon_vars, lon_bounds.name: lon_bounds})
    ds_no_lon = ds.get([v for v in ds.data_vars if lon.name not in ds[v].dims])
    ds = xr.merge((ds_no_lon, ds_lon))
    return ds


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
    # Example array: [[359, 1], [1, 90], [90, 180], [180, 359]]
    # Reorient bound to span across zero (i.e., [359, 1] -> [-1, 1]).
    # Result: [[-1, 1], [1, 90], [90, 180], [180, 359]]
    bounds[p_meridian_index, 0] = bounds[p_meridian_index, 0] - 360.0

    # Extend the array to nlon+1 by concatenating the grid cell that
    # spans the prime meridian to the end.
    # Result: [[-1, 1], [1, 90], [90, 180], [180, 359], [-1, 1]]
    dim = get_axis_dim(bounds, "X")
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


def _get_prime_meridian_index(lon_bounds: xr.DataArray) -> Optional[np.ndarray]:
    """Gets the index of the prime meridian cell in the longitude bounds.

    A prime meridian cell can exist when converting the axis orientation
    from [-180, 180) to [0, 360).

    Parameters
    ----------
    lon_bounds : xr.DataArray
        The longitude bounds.

    Returns
    -------
    Optional[np.ndarray]
        An array with a single elementing representing the index of the prime
        meridian index if it exists. Otherwise, None if the cell does not exist.

    Raises
    ------
    ValueError
        If more than one grid cell spans the prime meridian.
    """
    p_meridian_index = np.where(lon_bounds[:, 1] - lon_bounds[:, 0] < 0)[0]

    # FIXME: When does this conditional return true? It seems like swapping from
    # (-180, to 180) to (0, 360) always produces a prime meridian cell?
    if p_meridian_index.size == 0:  # pragma:no cover
        return None
    elif p_meridian_index.size > 1:
        raise ValueError("More than one grid cell spans prime meridian.")
    return p_meridian_index
