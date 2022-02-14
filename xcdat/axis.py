"""Axis module for utilities related to axes."""

from typing import Dict, Literal, Optional, Tuple

import numpy as np
import xarray as xr
from dask.array.core import Array

# Mapping of CF compliant long and short axis keys to their generic
# representations. This map is useful for indexing a Dataset/DataArray on
# a key by falling back on the generic version. Attempting to index on the short
# key when the long key is used will fail, but using the generic key should
# work.
CFAxis = Literal["lat", "latitude", "Y", "lon", "longitude", "X", "time", "T", "height", "pressure", "Z"]
GenericAxis = Literal["X", "Y", "T", "Z"]
GENERIC_AXIS_MAP: Dict[CFAxis, GenericAxis] = {
    "lat": "Y",
    "latitude": "Y",
    "Y": "Y",
    "lon": "X",
    "longitude": "X",
    "X": "X",
    "time": "T",
    "T": "T",
    "height": "Z",
    "pressure": "Z",
    "Z": "Z",
}


def swap_lon_axis(
    dataset: xr.Dataset, to: Tuple[float, float], sort_ascending: bool = True
) -> xr.Dataset:
    """Swap the longitude axis orientation for a dataset.

    This method also swaps the longitude bounds axis orientation if it exists,
    and sorts values in ascending order.

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
    lon: xr.DataArray = dataset.bounds._get_coords("lon").copy()
    lon_bounds: xr.DataArray = dataset.bounds.get_bounds("lon").copy()

    with xr.set_options(keep_attrs=True):
        if to == (-180, 180):
            lon = ((lon + 180) % 360) - 180
            lon_bounds = ((lon_bounds + 180) % 360) - 180
            ds = _reassign_lon(ds, lon, lon_bounds)
        elif to == (0, 360):
            lon = lon % 360
            lon_bounds = lon_bounds % 360
            ds = _reassign_lon(ds, lon, lon_bounds)

            # Handle cases where a prime meridian cell exists, which can occur
            # after swapping to (0, 360).
            p_meridian_index = _get_prime_meridian_index(lon_bounds)
            if p_meridian_index is not None:
                ds = _align_lon_to_360(ds, p_meridian_index)
        else:
            raise ValueError(
                "Currently, only (-180, 180) and (0, 360) are supported longitude axis "
                "orientations."
            )

    if sort_ascending:
        ds = ds.sortby(lon.name, ascending=True)

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
    lon_name = lon.name
    lon[lon_name] = lon_bounds[lon_name] = lon

    dataset[lon_name] = lon
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
    lon: xr.DataArray = dataset.bounds._get_coords("lon").copy()
    lon_name = lon.name
    lon_bounds: xr.DataArray = dataset.bounds.get_bounds("lon").copy()

    # If chunking, must convert convert the xarray data structure from lazy
    # Dask arrays into eager, in-memory NumPy arrays before performing
    # manipulations on the data. Otherwise, it raises `NotImplementedError
    # xarray can't set arrays with multiple array indices to dask yet`.
    if isinstance(lon_bounds.data, Array):
        lon_bounds.load()

    # Align the the longitude bounds using the prime meridian index.
    lon_bounds = _align_lon_bounds_to_360(lon_bounds, p_meridian_index)

    # Concatenate the longitude coordinates with 360 to handle the prime
    # meridian cell and update the coordinates for the longitude bounds.
    p_meridian_cell = xr.DataArray([360.0], coords={lon_name: [360.0]}, dims=[lon_name])
    lon = xr.concat((lon, p_meridian_cell), dim=lon_name)
    lon_bounds[lon_name] = lon

    # Get the data variables related to the longitude axis and concatenate each
    # with the value at the prime meridian.
    lon_vars = {}
    for key, value in ds.cf.data_vars.items():
        if key != lon_bounds.name and lon_name in value.dims:
            lon_vars[key] = value

    for name, var in lon_vars.items():
        p_meridian_val = var.isel({lon_name: p_meridian_index})
        new_var = xr.concat((var, p_meridian_val), dim=lon_name)
        new_var[lon_name] = lon
        lon_vars[name] = new_var

    # Create a Dataset with longitude data vars and merge it to the Dataset
    # without longitude data vars.
    ds_lon = xr.Dataset(data_vars={**lon_vars, lon_bounds.name: lon_bounds})
    ds_no_lon = ds.get([v for v in ds.data_vars if lon_name not in ds[v].dims])
    ds = xr.merge((ds_no_lon, ds_lon))  # type: ignore
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
    if (bounds.values.min() < 0) | (bounds.values.max() > 360):
        raise ValueError(
            "Longitude bounds aren't inclusively between 0 and 360. "
            "Use `_swap_lon_axis()` before calling `_align_longitude_to_360_axis()`."
        )

    # Example array: [[359, 1], [1, 90], [90, 180], [180, 359]]
    # Reorient bound to span across zero (i.e., [359, 1] -> [-1, 1]).
    # Result: [[-1, 1], [1, 90], [90, 180], [180, 359]]
    bounds[p_meridian_index, 0] = bounds[p_meridian_index, 0] - 360.0

    # Extend the array to nlon+1 by concatenating the grid cell that
    # spans the prime meridian to the end.
    # Result: [[-1, 1], [1, 90], [90, 180], [180, 359], [-1, 1]]
    bounds = xr.concat(
        (bounds, bounds[p_meridian_index, :]), dim=bounds.cf.axes["X"][0]
    )

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

    A prime meridian cell can exist when converting from converting the axis
    orientation from [-180, 180) to [0, 360).

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

    if p_meridian_index.size == 0:
        return None
    elif p_meridian_index.size > 1:
        raise ValueError("More than one grid cell spans prime meridian.")
    return p_meridian_index
