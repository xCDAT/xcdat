"""Variable module for functions related to an xarray.Dataset data variable"""
import xarray as xr

from xcdat.bounds import DataArrayBoundsAccessor  # noqa: F401


def open_variable(dataset: xr.Dataset, name: str) -> xr.DataArray:
    """Opens a Dataset data variable and applies additional operations.

    Operations include:

    - Propagate coordinate bounds from the parent ``Dataset`` to the
      ``DataArray`` data variable.

    Parameters
    ----------
    dataset : xr.Dataset
        The parent Dataset.
    name : str
        The name of the data variable to be opened.

    Returns
    -------
    xr.DataArray
        The data variable.

    Notes
    -----
    If you are familiar with CDAT, the ``DataArray`` data variable output is
    similar to a ``TransientVariable``, which stores coordinate bounds as object
    attributes.

    Examples
    --------
    Import module:

    >>> from xcdat.dataset import open_dataset
    >>> from xcdat.variable import open_variable

    Open a variable from a Dataset:

    >>> ds = open_dataset("file_path") # Auto-generate bounds if missing
    >>> ts = open_variable(ds, "ts")

    List coordinate bounds:

    >>> ts.coords
    * bnds       (bnds) int64 0 1
    * time       (time) datetime64[ns] 1850-01-16T12:00:00 ... 2005-12-16T12:00:00
    * lat        (lat) float64 -90.0 -88.75 -87.5 -86.25 ... 86.25 87.5 88.75 90.0
    * lon        (lon) float64 0.0 1.875 3.75 5.625 ... 352.5 354.4 356.2 358.1
    lon_bnds   (lon, bnds) float64 ...
    lat_bnds   (lat, bnds) float64 ...
    time_bnds  (time, bnds) datetime64[ns] ...

    Return coordinate bounds:

    >>> ts.lon_bnds
    >>> ts.lat_bnds
    >>> ts.time_bnds
    """
    data_var = dataset[name].copy()
    data_var = data_var.bounds.copy_from_parent(dataset)

    return data_var


def copy_variable(var: xr.DataArray) -> xr.DataArray:
    """Returns a copy of a variable with all accessor class attributes.

    If a copy of a variable is created by invoking the native
    `xarray.DataArray.copy()`, all cached accessor attributes will be dropped.
    This is an unfortunate issue that can compromise data integrity. Refer to
    https://github.com/pydata/xarray/issues/3268 for more information.

    This function works around this issue by invoking each accessor class'
    custom `.copy()` method, which just returns the object within the scope of
    the accessor class to retain attributes.

    Parameters
    ----------
    var : xr.DataArray
        The variable being copied.

    Returns
    -------
    xr.DataArray
        The copied variable with accessor attributes.

    Examples
    --------
    Import:

    >>> from xcdat.dataset import open_dataset
    >>> from xcdat.variable import copy_variable, open_variable

    Open a variable:

    >>> ds = open_dataset("file_path")
    >>> ts = open_variable(ds, "ts")

    Copying a variable:

    >>> ts_copy = copy_variable(ts)
    """
    var_copy = var.bounds.copy()

    return var_copy
