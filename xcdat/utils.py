import glob
import os
from typing import Dict, Tuple, get_args

import xarray as xr
from typing_extensions import Literal

# Add supported extensions on as-need basis
# https://xarray.pydata.org/en/stable/io.html#
extension = Literal["nc"]
SUPPORTED_EXTENSIONS: Tuple[extension, ...] = get_args(extension)


def open_datasets(
    path: str,
    extension: extension = None,
) -> Dict[str, xr.Dataset]:
    """Lazily loads datasets from a specified path

    Args:
        path (str): The path of the input files (e.g., "../input_data")
        extension (Literal[, optional): [description]. Defaults to "None".

    Returns:
        Dict[str, xr.Dataset]: [description]
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
