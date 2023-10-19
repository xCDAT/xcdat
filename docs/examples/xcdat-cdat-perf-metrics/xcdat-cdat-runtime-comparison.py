"""
A script that compares the API runtimes of xCDAT against CDAT using multi-file
time series datasets with varying sizes. The default number of samples taken
for each API runtime is 5, and the minimum value is recorded. Runtimes only
include computation and exclude I/O.  xCDAT can operate in serial or parallel,
while CDAT can only operate in serial.

xCDAT's parallel configuration:
  - datasets are chunked using the "time" axis with Dask's auto chunking option.
  - datasets are also opened in parallel using the `parallel=True`
    (uses `dask.delayed`).
  - The `flox` package is used for map-reduce grouping instead of the native
    Xarray serial grouping logic for temporal averaging APIs that use Xarray's
    groupby() under the hood. This includes `group_average()`, `climatology()`,
    and `departures()`)

How to use:
   1. Must have direct access to LLNL Climate Program filesystem with CMIP data.
   2. Create the conda/mamba environment:
      - `mamba create -n xcdat-cdat-runtime -c conda-forge python<3.12 numpy pandas xcdat=0.6.0 xesmf cdms2 cdutil
      - `mamba activate xcdat-cdat-runtime`
   3. Run the script
      - `python xcdat-cdat-runtime-comparison.py`

TODO: Record runtimes for temporal averaging APIs, only spatial averaging is
being recorded.
"""
from __future__ import annotations

import time
import timeit
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from xcdat._logger import _setup_custom_logger

# FIXME: I can't get the logger to not print out two messages.
# I already tried logger.propagate=False and using the root logger.
logger = _setup_custom_logger(__name__, propagate=True)

# Output file configurations
# --------------------------
TIME_STR = time.strftime("%Y%m%d-%H%M%S")
XC_FILENAME = f"{TIME_STR}-xcdat-runtimes"
CD_FILENAME = f"{TIME_STR}-cdat-runtimes"

# Plot Configurations
# -------------------
# The base plot configuration passed to Panda's DataFrame plotting API.
PLOT_CONFIG: pd.DataFrame.plot.__init__ = {
    "kind": "bar",
    "legend": True,
    "rot": 0,
    "x": "gb",
    "xlabel": "File Size (GB)",
    "ylabel": "Runtime (secs)",
    "figsize": (6, 4),
}
# The base bar label configuration passed to axis containers to add
# the floating point labels above the bars.
BAR_LABEL_CONFIG = {"fmt": "{:10.2f}", "label_type": "edge", "padding": 3}


# Input data configurations
# -------------------------
FILES_DICT: Dict[str, Dict[str, str]] = {
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
    #     "xml_path": "/p/user_pub/xclim/CMIP6/CMIP/historical/atmos/day/ta/CMIP6.CMIP.historical.MOHC.UKESM1-0-LL.r5i1p1f3.day.ta.atmos.glb-p8-gn.v20191115.0000000.0.xml",
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


def main(repeat: int = 5):
    """Get the API runtimes for xCDAT and CDAT.

    APIs tested include:
      - Spatial averaging
      - Time averaging (single snap-shot)
      - Climatology
      - Depatures

    Parameters
    ----------
    repeat : int
        Number of samples to take for each API call, by default 5. The minimum
        runtime is taken as the final runtime (refer to Notes).

    Notes
    -----
    According to `Timer.repeat()`:

        "In a typical case, the lowest value gives a lower bound for how fast
        your machine can run the given code snippet; higher values in the result
        vector are typically not caused by variability in Python's speed, but by
        other processes interfering with your timing accuracy. So the min() of
        the result is probably the only number you should be interested in."

    Source: https://github.com/python/cpython/blob/2587b9f64eefde803a5e0b050171ad5f6654f31b/Lib/timeit.py#L193-L203
    """
    df_xc_serial = _get_xcdat_runtimes(parallel=False, repeat=repeat)
    df_xc_parallel = _get_xcdat_runtimes(parallel=True, repeat=repeat)
    df_xc_times = pd.merge(df_xc_serial, df_xc_parallel, on=["pkg", "gb", "api"])
    df_xc_times.to_csv(f"{XC_FILENAME}.csv", index=False)
    plot_xcdat_runtimes(df_xc_times)

    df_cdat_times = _get_cdat_runtimes(repeat=repeat)
    df_cdat_times.to_csv(f"{CD_FILENAME}.csv", index=False)
    plot_cdat_runtimes(df_cdat_times)


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
        Number of samples to take for each API call. The minimum runtime is
        taken as the final runtime.

    Returns
    -------
    pd.DataFrame
        A DataFrame of API runtimes.
    """
    process_type = "serial" if parallel is False else "parallel"
    logger.info(f"Getting xCDAT {process_type} runtimes.")

    chunks, parallel, use_flox = _get_xr_config(parallel)

    api_runtimes = []

    for fsize, finfo in FILES_DICT.items():
        dir_path = finfo["dir_path"]
        var_key = finfo["var_key"]

        setup = _get_xr_setup(dir_path, chunks, parallel)
        api_map = _get_xr_api_map(var_key)

        logger.info(f"Opening '{var_key}' dataset ({fsize}, `{dir_path}`.")
        for api, call in api_map.items():
            logger.info(f"Getting runtime for `{api}()`.")
            entry: Dict[str, str | float | None] = {
                "pkg": "xcdat",
                "gb": fsize.split("_")[0],
                "api": api,
            }

            try:
                runtime = _get_runtime(setup=setup, stmt=use_flox + call, repeat=repeat)
            except Exception as e:
                print(e)
                runtime = None

            entry[f"runtime_{process_type}"] = runtime
            logger.info(f"`{api}()` runtime: {runtime}")

            api_runtimes.append(entry)

    df_runtimes = pd.DataFrame(api_runtimes)

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


def _get_xr_setup(dir_path: str, chunks: None | Dict[str, str], parallel: bool):
    return (
        "import xarray as xr\n"
        "import xcdat as xc\n"
        f"ds = xc.open_mfdataset('{dir_path}', chunks={chunks}, parallel={parallel})\n"
    )


def _get_xr_api_map(var_key: str):
    return {
        "spatial_avg": f"ds.spatial.average('{var_key}', axis=['X', 'Y'])",
        # "temporal_avg": f"ds.temporal.average('{var_key}', weighted=True)",
        # "climatology": f"ds.temporal.climatology('{var_key}', freq='month', weighted=True)",
        # "departures": f"ds.temporal.departures('{var_key}', freq='month', weighted=True)",
    }


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
    logger.info("Getting CDAT runtimes (serial-only).")

    runtimes = []

    for fsize, finfo in FILES_DICT.items():
        xml_path = finfo["xml_path"]
        var_key = finfo["var_key"]

        setup = _get_cdat_setup(var_key, xml_path)
        api_map = _get_cdat_api_map()

        logger.info(f"Opening '{var_key}' dataset ({fsize}, `{xml_path}`.")
        for api, call in api_map.items():
            logger.info(f"Getting runtime for `{api}()`.")
            entry: Dict[str, str | float | None] = {
                "pkg": "cdat",
                "gb": fsize.split("_")[0],
                "api": api,
            }
            try:
                runtime = _get_runtime(setup=setup, stmt=call, repeat=repeat)
            except Exception as e:
                logger.error(e)
                runtime = None

            entry["runtime_serial"] = runtime
            logger.info(f"`{api}()` runtime: {runtime}")

        runtimes.append(entry)

    df_runtimes = pd.DataFrame(runtimes)

    return df_runtimes


def _get_cdat_setup(var_key: str, xml_path: str):
    setup = (
        "import cdms2\n"
        "import cdutil\n"
        f"ds = cdms2.open('{xml_path}')\n"
        f"t_var = ds['{var_key}']"
    )

    return setup


def _get_cdat_api_map():
    api_calls = {
        "spatial_avg": "cdutil.averager(t_var, axis='xy')",
        # "temporal_avg": "cdutil.averager(t_var, axis='t')",
        # "climatology": "cdutil.ANNUALCYCLE.climatology(t_var)",
        # "departures": "cdutil.ANNUALCYCLE.departures(t_var)",
    }

    return api_calls


def _get_runtime(setup: str, stmt: str, repeat: int = 5, number: int = 1) -> float:
    """Get the minimum runtime for a code statement using timeit.

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
    """
    runtimes: List[float] = timeit.repeat(
        setup=setup,
        stmt=stmt,
        repeat=repeat,
        number=number,
    )

    min = np.around(np.min(runtimes), decimals=6)

    return min


def plot_xcdat_runtimes(df_xcdat: pd.DataFrame):
    apis = df_xcdat.api.unique()

    for api in apis:
        ax = df_xcdat.plot(**PLOT_CONFIG)

        for cont in ax.containers:
            ax.bar_label(cont, **BAR_LABEL_CONFIG)

        ax.margins(y=0.1)
        ax.legend(["Serial", "Parallel"], fontsize="medium", loc="upper center", ncol=2)

        fig = ax.get_figure()

        api_title = api.title().replace("_", " ")
        fig.suptitle(f"xCDAT {api_title} Runtime")
        fig.tight_layout()
        fig.savefig(f"{XC_FILENAME}-{api}.png")


def plot_cdat_runtimes(df_cdat: pd.DataFrame):
    apis = df_cdat.api.unique()

    for api in apis:
        ax = df_cdat.plot(**PLOT_CONFIG)

        for cont in ax.containers:
            ax.bar_label(cont, **BAR_LABEL_CONFIG)

        ax.margins(y=0.1)
        ax.legend(["Serial"], fontsize="medium", loc="upper center", ncol=1)

        fig = ax.get_figure()

        api_title = api.title().replace("_", " ")
        fig.suptitle(f"CDAT {api_title} Runtime")
        fig.tight_layout()

        fig.savefig(f"{CD_FILENAME}-{api}.png")


if __name__ == "__main__":
    main()
