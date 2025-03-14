from __future__ import annotations

import os
import pathlib
import sys

from typing import TYPE_CHECKING, Dict

import xarray as xr
from xarray import open_dataset as xr_open_dataset
from xarray.tutorial import (
    _construct_cache_dir,
    open_dataset as xr_tut_open_dataset,
    file_formats,
)
import xcdat.bounds  # noqa: F401

if TYPE_CHECKING:
    import os

    from xarray.backends.api import T_Engine

try:
    import pooch
except ImportError as e:
    raise ImportError(
        "tutorial.open_dataset depends on pooch to download and manage datasets."
        " To proceed please install pooch."
    ) from e


_default_cache_dir_name = "xcdat_tutorial_data"
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
    # Monthly near-surface air temperature from the CanESM5 model.
    "tas_amon_canesm5": "tas_Amon_CanESM5_historical_r13i1p1f1_gn_185001-201412_subset.nc",
    # Monthly ocean potential temperature from the CESM2 model.
    "thetao_omon_cesm2": "thetao_Omon_CESM2_historical_r1i1p1f1_gn_185001-201412_subset.nc",
}


def open_dataset(
    name: str,
    cache: bool = True,
    cache_dir: None | str | os.PathLike = None,
    *,
    engine: T_Engine = None,
    **kws,
) -> xr.Dataset:
    """
    Open a dataset from the online repository (requires internet).

    If an Xarray tutorial dataset is specified, this function will use the
    ``xarray.tutorial.open_dataset()`` function. This function is mostly based
    on ``xarray.tutorial.open_dataset()`` with some modifications such as adding
    bounds to the dataset.

    If a local copy is found then always use that to avoid network traffic.

    Available Xarray datasets:

    * ``"air_temperature"``: NCEP reanalysis subset
    * ``"air_temperature_gradient"``: NCEP reanalysis subset with approximate x,y gradients
    * ``"basin_mask"``: Dataset with ocean basins marked using integers
    * ``"ASE_ice_velocity"``: MEaSUREs InSAR-Based Ice Velocity of the Amundsen Sea Embayment, Antarctica, Version 1
    * ``"rasm"``: Output of the Regional Arctic System Model (RASM)
    * ``"ROMS_example"``: Regional Ocean Model System (ROMS) output
    * ``"tiny"``: small synthetic dataset with a 1D data variable
    * ``"era5-2mt-2019-03-uk.grib"``: ERA5 temperature data over the UK
    * ``"eraint_uvz"``: data from ERA-Interim reanalysis, monthly averages of upper level data
    * ``"ersstv5"``: NOAA's Extended Reconstructed Sea Surface Temperature monthly averages

    Available xCDAT datasets:
    * ``"pr_amon_access"``: Monthly precipitation data from the ACCESS-ESM1-5 model.
    * ``"so_omon_cesm2"``: Monthly ocean salinity data from the CESM2 model.
    * ``"tas_amon_access"``: Monthly near-surface air temperature from the ACCESS-ESM1-5 model.
    * ``"tas_amon_canesm5"``: Monthly near-surface air temperature from the CanESM5 model.
    * ``"thetao_omon_cesm2"``: Monthly ocean potential temperature from the CESM2 model.

    Parameters
    ----------
    name : str
        Name of the file containing the dataset.
        e.g. 'air_temperature'
    cache_dir : path-like, optional
        The directory in which to search for and write cached data.
    cache : bool, optional
        If True, then cache data locally for use on subsequent calls
    **kws : dict, optional
        Passed to xarray.open_dataset
    """
    if name in XARRAY_DATASETS:
        ds = xr_tut_open_dataset(
            name=name, cache=cache, cache_dir=cache_dir, engine=engine, **kws
        )
    else:
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
        ds = xr_open_dataset(filepath, **kws)

        if not cache:
            ds = ds.load()
            pathlib.Path(filepath).unlink()

    ds = ds.bounds.add_missing_bounds(axes=["X", "Y", "T"])

    return ds
