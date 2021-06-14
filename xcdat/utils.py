"""Utility functions that might be helpful to use."""
import glob
import os
from typing import Dict, Tuple, get_args

import xarray as xr
from typing_extensions import Literal

# Add supported extensions on as-need basis
# https://xarray.pydata.org/en/stable/io.html#
SupportedExtensions = Literal["nc"]

SUPPORTED_EXTENSIONS: Tuple[SupportedExtensions, ...] = get_args(SupportedExtensions)


def open_datasets(
    path: str,
    extension: SupportedExtensions = None,
) -> Dict[str, xr.Dataset]:
    """Lazily loads datasets from a specified path.

    Parameters
    ----------
    path : str
        The relative or absolute path of the input files.
    extension : SupportedExtensions, optional
        The file extension to look for. Refer to ``SupportedExtensions``, by
        default None.

    Returns
    -------
    Dict[str, xr.Dataset]
        A dictionary of datasets, key is file name and value is dataset object.
    """
    datasets: Dict[str, xr.Dataset] = dict()
    files_grabbed = []

    if extension:
        files_grabbed.extend(glob.glob(os.path.join(path, f"*.{extension}")))
    else:
        for extension in SUPPORTED_EXTENSIONS:
            files_grabbed.extend(glob.glob(os.path.join(path, f"*.{extension}")))

    for file in files_grabbed:
        key = file.replace(f"{path}/", "")
        datasets[key] = xr.open_dataset(file)

    return datasets
