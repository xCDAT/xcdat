"""Variable module for functions related to an xarray.Dataset data variable"""
import xarray as xr

from xcdat.bounds import DataArrayBoundsAccessor  # noqa: F401


def open_variable(dataset: xr.Dataset, var_name: str) -> xr.DataArray:
    """Opens a Dataset data variable with additional attributes.

    This function calls DataArray accessors on the data variable to store
    additional attributes, which are cached in the DataArray object.

    Attributes include:

    - Coordinate bounds ("lat_bnds", "lon_bnds", "time_bnds)

    Parameters
    ----------
    dataset : xr.Dataset
        The parent Dataset.
    var_name : str
        The name of the variable to be opened.

    Returns
    -------
    xr.DataArray
        The variable with additional attributes.

    Notes
    -----
    If you are coming from CDAT, the DataArray output is similar to a
    TransientVariable, which contains references to bounds and other
    attributes.

    Examples
    --------
    Import module:

    >>> from xcdat.dataset import open_dataset
    >>> from xcdat.variable import open_variable

    Open a variable from a Dataset:

    >>> ds = open_dataset("file_path") # Auto-generate bounds if missing
    >>> tas = open_variable(ds, "tas")

    Return dictionary of bounds accessor attributes:

    >>> tas.bounds.__dict__

    Return coordinate bounds attributes:

    >>> tas.bounds.lat
    >>> tas.bounds.lon
    >>> tas.bounds.time
    """
    data_var = dataset[var_name].copy()
    data_var.bounds._copy_from_dataset(dataset)

    return data_var
