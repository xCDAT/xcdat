from __future__ import annotations

import os
import pathlib
import sys
from typing import Dict, List, Tuple

import xarray as xr
from xarray.tutorial import _construct_cache_dir, file_formats

import xcdat.bounds  # noqa: F401
from xcdat.axis import CFAxisKey

DEFAULT_CACHE_DIR_NAME = "xcdat_tutorial_data"
base_url = "https://github.com/xCDAT/xcdat-data"
version = "main"

XARRAY_DATASETS = list(file_formats.keys()) + ["era5-2mt-2019-03-uk.grib"]
XCDAT_DATASETS: Dict[str, str] = {
    # Monthly precipitation data from the ACCESS-ESM1-5 model.
    "pr_amon_access": "pr_Amon_ACCESS-ESM1-5_historical_r10i1p1f1_gn_185001-201412_subset.nc",
    # Monthly ocean salinity data from the CESM2 model.
    "so_omon_cesm2": "so_Omon_CESM2_historical_r1i1p1f1_gn_185001-201412_subset.nc",
    # Monthly near-surface air temperature from the ACCESS-ESM1-5 model.
    "tas_amon_access": "tas_Amon_ACCESS-ESM1-5_historical_r10i1p1f1_gn_185001-201412_subset.nc",
    # 3-hourly near-surface air temperature from the ACCESS-ESM1-5 model.
    "tas_3hr_access": "tas_3hr_ACCESS-ESM1-5_historical_r10i1p1f1_gn_201001010300-201501010000_subset.nc",
    # Monthly near-surface air temperature from the CanESM5 model.
    "tas_amon_canesm5": "tas_Amon_CanESM5_historical_r13i1p1f1_gn_185001-201412_subset.nc",
    # Monthly ocean potential temperature from the CESM2 model.
    "thetao_omon_cesm2": "thetao_Omon_CESM2_historical_r1i1p1f1_gn_185001-201412_subset.nc",
    # Monthly cloud fraction data from the E3SM-2-0 model.
    "cl_amon_e3sm2": "cl_Amon_E3SM-2-0_historical_r1i1p1f1_gr_185001-189912_subset.nc",
    # Monthly air temperature data from the E3SM-2-0 model.
    "ta_amon_e3sm2": "ta_Amon_E3SM-2-0_historical_r1i1p1f1_gr_185001-189912_subset.nc",
}


def open_dataset(
    name: str,
    cache: bool = True,
    cache_dir: None | str | os.PathLike = DEFAULT_CACHE_DIR_NAME,
    add_bounds: List[CFAxisKey] | Tuple[CFAxisKey, ...] | None = ("X", "Y"),
    **kargs,
) -> xr.Dataset:
    """
     Open a dataset from the online repository (requires internet).

    This function is mostly based on ``xarray.tutorial.open_dataset()`` with
    some modifications, including adding missing bounds to the dataset.

    If a local copy is found then always use that to avoid network traffic.

     Available xCDAT datasets:

     * ``"pr_amon_access"``: Monthly precipitation data from the ACCESS-ESM1-5 model.
     * ``"so_omon_cesm2"``: Monthly ocean salinity data from the CESM2 model.
     * ``"tas_amon_access"``: Monthly near-surface air temperature from the ACCESS-ESM1-5 model.
     * ``"tas_3hr_access"``: 3-hourly near-surface air temperature from the ACCESS-ESM1-5 model.
     * ``"tas_amon_canesm5"``: Monthly near-surface air temperature from the CanESM5 model.
     * ``"thetao_omon_cesm2"``: Monthly ocean potential temperature from the CESM2 model.
     * ``"cl_amon_e3sm2"``: Monthly cloud fraction data from the E3SM-2-0 model.
     * ``"ta_amon_e3sm2"``: Monthly air temperature data from the E3SM-2-0 model.

     Parameters
     ----------
     name : str
         Name of the file containing the dataset.
         e.g. 'air_temperature'
     cache_dir : path-like, optional
         The directory in which to search for and write cached data.
     cache : bool, optional
         If True, then cache data locally for use on subsequent calls
     add_bounds : List[CFAxisKey] | Tuple[CFAxisKey] | None, optional
         List or tuple of axis keys for which to add bounds, by default
         ("X", "Y").
     **kargs : dict, optional
         Passed to ``xcdat.open_dataset``.
    """
    try:
        import pooch
    except ImportError as e:
        raise ImportError(
            "tutorial.open_dataset depends on pooch to download and manage datasets."
            " To proceed please install pooch."
        ) from e

    # Avoid circular import in __init__.py
    from xcdat.dataset import open_dataset

    logger = pooch.get_logger()
    logger.setLevel("WARNING")

    cache_dir = _construct_cache_dir(cache_dir)

    filename = XCDAT_DATASETS.get(name)
    if filename is None:
        raise ValueError(
            f"Dataset {name} not found. Available xcdat datasets are: {XCDAT_DATASETS.keys()}"
        )

    path = pathlib.Path(filename)
    url = f"{base_url}/raw/{version}/{path.name}"

    headers = {"User-Agent": f"xcdat {sys.modules['xcdat'].__version__}"}
    downloader = pooch.HTTPDownloader(headers=headers)

    filepath = pooch.retrieve(
        url=url, known_hash=None, path=cache_dir, downloader=downloader
    )
    ds = open_dataset(filepath, **kargs, add_bounds=add_bounds)

    if not cache:
        ds = ds.load()
        pathlib.Path(filepath).unlink()

    return ds
