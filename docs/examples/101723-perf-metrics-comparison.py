"""
A script that compares the API runtimes of xCDAT against CDAT using
multi-file time series datasets of varying sizes.

For xCDAT's parallel configuration, datasets are chunked using the "time"
axis with Dask's auto chunking option. Datasets are also opened in parallel
using the `parallel=True` (uses `dask.delayed`). Runtimes only include
API computation and not I/O. The `flox` package is used for map-reduce grouping
instead of the native Xarray serial grouping logic for temporal averaging APIs
that use Xarray's groupby() under the hood (e.g., `group_average()`,
`climatology()`, and `departures()`)
"""
from __future__ import annotations

import time
import timeit
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xcdat._logger import _setup_custom_logger

# FIXME: I can't get the logger to not print out two messages.
# I already tried logger.propagate=False and using the root logger.
logger = _setup_custom_logger(__name__, propagate=True)

TIME_STR = time.strftime("%Y%m%d-%H%M%S")
XC_FILENAME = f"{TIME_STR}-xcdat-perf-metrics.csv"
CD_FILENAME = f"{TIME_STR}-cdat-perf-metrics.csv"

PLOT_FILENAME = f"{TIME_STR}-xcdat-cdat-perf-metrics.png"
FILES_DICT: Dict[str, Dict[str, str | None]] = {
    "7_gb": {
        "var_key": "tas",
        "dir_path": "/p/css03/esgf_publish/CMIP6/CMIP/NCAR/CESM2/historical/r1i1p1f1/day/tas/gn/v20190308/",
        "xml_path": "/p/user_pub/e3sm/vo13/xclim/CMIP6/CMIP/historical/atmos/day/tas/CMIP6.CMIP.historical.NCAR.CESM2.r1i1p1f1.day.tas.atmos.glb-p8-gn.v20190308.0000000.0.xml",
    },
    # "12_gb": {
    #     "var_key": "tas",
    #     "dir_path": "/p/css03/esgf_publish/CMIP6/CMIP/MRI/MRI-ESM2-0/amip/r1i1p1f1/3hr/tas/gn/v20190829/",
    #     "xml_path": "/p/user_pub/e3sm/vo13/xclim/CMIP6/CMIP/historical/atmos/3hr/tas/CMIP6.CMIP.historical.MRI.MRI-ESM2-0.r1i1p1f1.3hr.tas.gn.v20190829.0000000.0.xml",
    # },
    # "22_gb": {
    #     "var_key": "ta",
    #     "dir_path": "/p/css03/esgf_publish/CMIP6/CMIP/MOHC/UKESM1-0-LL/historical/r5i1p1f3/day/ta/gn/v20191115/",
    #     "xml_path": "/p/user_pub/xclim/CMIP6/CMIP/historical/atmos/day/ta/CMIP6.CMIP.historical.MOHC.UKESM1-0-LLr5i1p1f3.day.ta.atmos.glb-p8-gn.v20191115.0000000.0.xml",
    # },
    # "50_gb": {
    #     "var_key": "ta",
    #     "dir_path": "/p/css03/esgf_publish/CMIP6/CMIP/NCAR/CESM2/historical/r1i1p1f1/day/ta/gn/v20190308/",
    #     "xml_path": "/p/user_pub/xclim/CMIP6/CMIP/historical/atmos/day/ta/CMIP6.CMIP.historical.NCAR.CESM2.r1i1p1f1.day.ta.atmos.glb-p8-gn.v20190308.0000000.0.xml",
    # },
    # "74_gb": {
    #     "var_key": "ta",
    #     "dir_path": "/p/css03/esgf_publish/CMIP6/CMIP/CCCma/CanESM5/historical/r1i1p2f1/CFday/ta/gn/v20190429/",
    #     "xml_path": "/p/user_pub/e3sm/vo13/xclim/CMIP6/CMIP/historical/atmos/day/ta/CMIP6.CMIP.historical.CCCma.CanESM5.r1i1p2f1.CFday.ta.atmos.glb-p80-gn.v20190429.0000000.0.xml",
    # },
    # "105_gb": {
    #     "var_key": "ta",
    #     "dir_path": "/p/css03/esgf_publish/CMIP6/CMIP/MOHC/HadGEM3-GC31-MM/historical/r2i1p1f3/day/ta/gn/v20191218",
    #     "xml_path": "/p/user_pub/xclim/CMIP6/CMIP/historical/atmos/day/ta/CMIP6.CMIP.historical.MOHC.HadGEM3-GC31-MM.r2i1p1f3.day.ta.atmos.glb-p8-gn.v20191218.0000000.0.xml",
    # },
}


def get_runtimes():
    """Get the API runtimes for xCDAT and CDAT.

    APIs tested include:
      - Spatial averaging
      - Time averaging (single snap-shot)
      - Climatology
      - Depatures
    """
    df_xc_serial = _get_xcdat_runtimes(parallel=False, repeat=1)
    df_xc_parallel = _get_xcdat_runtimes(parallel=True, repeat=1)

    df_xc_final = pd.concat([df_xc_serial, df_xc_parallel]).reset_index()
    df_xc_final.to_csv(XC_FILENAME)

    df_cdat_times = _get_cdat_runtimes(repeat=1)
    df_cdat_times.to_csv(CD_FILENAME)

    # plot_metrics(df_xc_serial, df_xc_parallel)


def _get_xcdat_runtimes(
    parallel: bool,
    repeat: int,
) -> pd.DataFrame:
    """Get the xCDAT API runtimes for spatial and temporal averaging.

    Parameters
    ----------
    parallel : bool
        Whether to run the APIs using Dask parallelism (True) or in serial
        (False). If in parallel, datasets are chunked on the time axis using
        Dask's auto chunking, and `flox` is used for temporal averaging.
    repeat : int
        Number of samples to take for each API call.

    Returns
    -------
    pd.DataFrame
        A DataFrame of API runtimes.
    """
    process_type = "serial" if parallel is False else "parallel"
    logger.info(f"Getting xCDAT {process_type} runtimes...")

    chunks, parallel, use_flox = _get_xr_config(parallel)

    runtimes = []
    for fsize, finfo in FILES_DICT.items():
        dir_path = finfo["dir_path"]  # type: ignore
        var_key = finfo["var_key"]  # type: ignore

        logger.info(f"Opening '{var_key}' dataset ({fsize}, `{dir_path}`.")
        setup = (
            "import xarray as xr\n"
            "import xcdat as xc\n"
            f"ds = xc.open_mfdataset(f'{dir_path}', chunks={chunks}, parallel={parallel})\n"
        )
        api_to_call = {
            "spatial_avg": f"ds.spatial.average('{var_key}', axis=['X', 'Y'])",
            # "temporal_avg": f"ds.temporal.average('{var_key}', weighted=True)",
            # "climatology": f"ds.temporal.climatology('{var_key}', freq='month', weighted=True)",
            # "departures": f"ds.temporal.departures('{var_key}', freq='month', weighted=True)",
        }

        for api, call in api_to_call.items():
            logger.info(f"Getting runtime for `{api}()`.")
            entry: Dict[str, str | float | None] = {
                "pkg": "xcdat",
                "gb": fsize.split("_")[0],
                "process": process_type,
                "api": api,
            }

            try:
                entry["runtime"] = _get_runtime(
                    setup=setup, stmt=use_flox + call, repeat=repeat
                )
            except RuntimeError as e:
                logger.error(e)
                entry["runtime"] = None

            runtimes.append(entry)

    df_runtimes = pd.DataFrame(runtimes)

    return df_runtimes


def _get_xr_config(parallel: bool) -> Tuple[None | Dict[str, str], bool, str]:
    if not parallel:
        chunks = None
        parallel = False
        use_flox = "with xr.set_options(use_flox=False): \n    "
    elif parallel:
        chunks = {"time": "auto"}
        parallel = True
        use_flox = "with xr.set_options(use_flox=True): \n    "

    return chunks, parallel, use_flox


def _get_cdat_runtimes(repeat: int) -> pd.DataFrame:
    """Get the CDAT API runtimes (only supports serial).

    Parameters
    ----------
    repeat : int
        Number of samples to take for each API call.

    Returns
    -------
    pd.DataFrame
        A DataFrame of runtimes for CDAT APIs.
    """
    logger.info(f"Getting CDAT serial runtimes...")

    runtimes = []

    for fsize, finfo in FILES_DICT.items():
        xml_path = finfo["xml_path"]
        var_key = finfo["var_key"]
        logger.info(f"Opening '{var_key}' dataset ({fsize}, `{xml_path}`.")

        setup = (
            "import cdms2\n"
            "import cdutil\n"
            f"ds = cdms2.open('{xml_path}')\n"
            f"t_var = ds['{var_key}']"
        )
        api_calls = {
            "spatial_avg": "cdutil.averager(t_var, axis='xy')",
            # "temporal_avg": "cdutil.averager(t_var, axis='t')",
            # "climatology": "cdutil.ANNUALCYCLE.climatology(t_var)",
            # "departures": "cdutil.ANNUALCYCLE.departures(t_var)",
        }

        for api, call in api_calls.items():
            logger.info(f"Getting runtime for `{api}()`.")
            entry: Dict[str, str | float] = {
                "pkg": "xcdat",
                "gb": fsize.split("_")[0],
                "process": "serial",
                "api": api,
            }
            entry["runtime"] = _get_runtime(setup=setup, stmt=call, repeat=repeat)

        runtimes.append(entry)

    df_runtimes = pd.DataFrame(runtimes)

    return df_runtimes


def _get_runtime(setup: str, stmt: str, repeat: int = 5, number: int = 1) -> float:
    """Get the runtime for a code statement using timeit.

    This function takes the lowest performance value for each sample.

    Parameters
    ----------
    setup : str
        The setup code (e.g,. imports).
    stmt : str
        The statement to measure performance on (e.g., API calls).
    repeat : int, optional
        Number of samples to take, by default 5.
    number : int, optional
        Number of times to repeat the statement for each sample, by default 1.

    Returns
    -------
    float
        The average minimum runtime out of all of the samples.

    Example
    -------

    ``setup`` example:

    >>> DIR = (
    >>>    "/p/css03/esgf_publish/CMIP6/CMIP/MOHC/HadGEM3-GC31-MM/historical/r2i1p1f3"
    >>>    "/day/ta/gn/v20191218"
    >>> )
    >>> setup = (
    >>>    "import xcdat as xc\n"
    >>>    f"ds = xc.open_mfdataset(f'{DIR}/*.nc', chunks='auto')\n"
    >>> )

    ``stmt` example:

    >>> ds.spatial.average('ta')
    """
    runtimes: List[float] = timeit.repeat(
        setup=setup,
        stmt=stmt,
        repeat=repeat,
        number=number,
    )

    min = np.around(np.min(runtimes), decimals=6)

    return min


def plot_metrics(df_serial, df_parallel):
    # Base plot configuraiton.
    base_config: pd.DataFrame.plot.__init__ = {
        "kind": "line",
        "legend": True,
        "x": "gb",
        "xlabel": "File Size (GB)",
        "y": "runtime",
        "ylabel": "Runtime (secs)",
    }

    apis = df_serial.api.unique()

    for api in apis:
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.suptitle(f"xCDAT {api}() Runtime")

        # Add both plots to the subplots Axes.
        df_serial_sub = df_serial.loc[df_serial.api == api]
        df_serial_sub.plot(**base_config, ax=ax)

        df_parallel_sub = df_parallel[df_parallel.api == api]
        df_parallel_sub.plot(**base_config, ax=ax)

        # Update legend and set Y ticks.
        ax.legend(["Parallel", "Serial"])
        # runtimes = pd.concat([df_serial_sub["runtime"], df_parallel_sub["runtime"]])
        # # ax.set_yticks(np.arange(0, runtimes.max() + 0.25, 0.25))

        # Set tight layout and save.
        fig.tight_layout()
        fig.savefig(PLOT_FILENAME)


if __name__ == "__main__":
    get_runtimes()

# %%
