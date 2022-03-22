import json
from typing import Dict, List

import xarray as xr


def is_documented_by(original):
    """A decorator for reusing API docstrings.

    Parameters
    ----------
    original
        The original function or method source for the API docstring.
    """

    def wrapper(target):
        target.__doc__ = original.__doc__
        return target

    return wrapper


def compare_datasets(ds1: xr.Dataset, ds2: xr.Dataset) -> Dict[str, List[str]]:
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
    Dict[str, Union[List[str]]]
        A dictionary mapping unique, non-identical, and
        non-equal keys in both Datasets.
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

    netCDF4 files cannot store bools in Dataset/DataArray attributes. XCDAT
    works around this by storing boolean operation metadata as strings.
    For example, True gets stored as "True" in the attributes of the xarray
    object. When using these string attributes, they must be converted back to a
    bool.

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
