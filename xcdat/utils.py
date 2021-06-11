"""Utility functions that might be helpful to use."""
import glob
import os
from typing import Dict, Tuple, get_args

import xarray as xr
from typing_extensions import Literal

# Add supported extensions on as-need basis
# https://xarray.pydata.org/en/stable/io.html#
Extension = Literal["nc"]

SUPPORTED_EXTENSIONS: Tuple[Extension, ...] = get_args(Extension)


def open_datasets(
    path: str,
    extension: Extension = None,
) -> Dict[str, xr.Dataset]:
    """Lazily loads datasets from a specified path

    :param path: The path of the input files (e.g., "../input_data")
    :type path: str
    :param extension: [description], defaults to None
    :type extension: extension, optional
    :return: The extension of the input files
    :rtype: Dict[str, xr.Dataset]
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
