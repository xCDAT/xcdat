import importlib
import json
from typing import Hashable, cast

import xarray as xr
from dask.array.core import Array


def compare_datasets(ds1: xr.Dataset, ds2: xr.Dataset) -> dict[str, list[Hashable]]:
    """Compares the keys and values of two datasets.

    This utility function is especially useful for debugging tests that
    involve comparing two Dataset objects for being identical or equal.

    Checks include:

    - Unique keys - keys that exist only in one of the two datasets.
    - Non-identical - keys whose values have the same dimension, coordinates,
      values, name, attributes, and attributes on all coordinates.
    - Non-equal keys - keys whose values have the same dimension, coordinates,
      and values, but not necessarily the same attributes. Key values that are
      non-equal will also be non-identical.

    Parameters
    ----------
    ds1 : xr.Dataset
        The first Dataset.
    ds2 : xr.Dataset
        The second Dataset.

    Returns
    -------
    dict[str, list[Hashable]]
        A dictionary mapping unique, non-identical, and non-equal keys in
        both Datasets.
    """
    results = {
        "unique_coords": list(ds1.coords.keys() ^ ds2.coords.keys()),
        "unique_data_vars": list(ds1.data_vars.keys() ^ ds2.data_vars.keys()),
        "nonidentical_coords": [],
        "nonidentical_data_vars": [],
        "nonequal_coords": [],
        "nonequal_data_vars": [],
    }

    ds_keys = {
        "coords": ds1.coords.keys() & ds2.coords.keys(),
        "data_vars": ds1.data_vars.keys() & ds2.data_vars.keys(),
    }
    for key_type, keys in ds_keys.items():
        for key in keys:
            identical = ds1[key].identical(ds2[key])
            equals = ds1[key].equals(ds2[key])

            if not identical:
                results[f"nonidentical_{key_type}"].append(key)
            if not equals:
                results[f"nonequal_{key_type}"].append(key)

    return results


def str_to_bool(attr: str) -> bool:
    """Converts bool string to bool.

    netCDF files can only store attributes with a type of str, Number, ndarray,
    number, list, or tuple.

    xCDAT methods store boolean attributes as strings. This function will
    convert such attributes back to booleans.

    Parameters
    ----------
    attr : str
        The boolean attribute as type str.

    Returns
    -------
    bool
        The boolean attribute as type bool.
    """
    if attr != "True" and attr != "False":
        raise ValueError(
            "The attribute is not a string representation of a Python"
            "bool ('True' or 'False')"
        )

    bool_attr = json.loads(attr.lower())
    return bool_attr


def _has_module(modname: str) -> bool:  # pragma: no cover
    """Checks if the specified module is installed in the Python environment.

    Parameters
    ----------
    modname : str
        The name of the module.

    Returns
    -------
    bool
    """
    try:
        importlib.import_module(modname)
        has = True
    except ImportError:
        has = False

    return has


def _if_multidim_dask_array_then_load(
    obj: xr.DataArray | xr.Dataset,
) -> xr.DataArray | xr.Dataset | None:
    """
    If the underlying array for an xr.DataArray or xr.Dataset is a
    multidimensional, lazy Dask Array, load it into an in-memory NumPy array.

    This function must be called before manipulating values in a
    multidimensional Dask Array, which xarray does not support directly.
    Otherwise, it raises `NotImplementedError xarray can't set arrays with
    multiple array indices to dask yet`.

    Parameters
    ----------
    obj : xr.DataArray | xr.Dataset | None
        The xr.DataArray or xr.Dataset. If the xarray object is chunked,
        the underlying array will be a Dask Array. Otherwise, return None.
    """
    if isinstance(obj.data, Array) and obj.ndim > 1:
        return obj.load()

    return None


def _get_masked_weights(dv: xr.DataArray, weights: xr.DataArray) -> xr.DataArray:
    """Get weights with missing data (`np.nan`) receiving no weight (zero).

    Parameters
    ----------
    dv : xr.DataArray
        The variable.
    weights : xr.DataArray
        A DataArray containing either the regional or temporal weights used for
        weighted averaging. ``weights`` must include the same axis dimensions
        and dimensional sizes as the data variable.

    Returns
    -------
    xr.DataArray
        The masked weights.
    """
    masked_weights = xr.where(dv.copy().isnull(), 0.0, weights)

    return masked_weights


def _validate_min_weight(min_weight: float | None) -> float:
    """Validate the ``min_weight`` value.

    Parameters
    ----------
    min_weight : float | None
        Fraction of data coverage (i..e, weight) needed to return a
        spatial average value. Value must range from 0 to 1.

    Returns
    -------
    float
        The required weight percentage.

    Raises
    ------
    ValueError
        If the `min_weight` argument is less than 0.
    ValueError
        If the `min_weight` argument is greater than 1.
    """
    if min_weight is None:
        return 0.0
    elif min_weight < 0.0:
        raise ValueError(
            "min_weight argument is less than 0. min_weight must be between 0 and 1.",
        )
    elif min_weight > 1.0:
        raise ValueError(
            "min_weight argument is greater than 1. min_weight must be between 0 and 1.",
        )

    return min_weight


def _as_dataarray(x) -> xr.DataArray:
    """Assert (for typing purposes) that `x` is an xarray.DataArray.

    This function is useful for type checkers like mypy to understand that
    the input `x` should be treated as an xarray.DataArray. For example, calling
    NumPy functions directly on Xarray objects can cause mypy to think a
    NumPy ndarray is being used, leading to type errors.

    It relies on xarray's __array_ufunc__ behavior when applying NumPy ufuncs
    to DataArray inputs. No runtime conversion is performed.

    Parameters
    ----------
    x : Any
        The input to be treated as an xarray.DataArray.

    Returns
    -------
    xr.DataArray
        The input `x` cast as an xarray.DataArray.
    """
    return cast(xr.DataArray, x)
