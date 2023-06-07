# %%
import timeit
from typing import List, Tuple

import cdms2
import cdutil
import numpy as np
import xarray as xr
import xcdat as xc


# %%
def get_runtime(
    setup: str, stmt: str, repeat: int = 5, number: int = 1
) -> Tuple[float, float]:
    """Get the runtime for a code statement using timeit.

    This function takes the lowest performance value for each sample.

    According to `Timer.repeat()`:

        "In a typical case, the lowest value gives a lower bound for how fast
        your machine can run the given code snippet; higher values in the result
        vector are typically not caused by variability in Python's speed, but by
        other processes interfering with your timing accuracy. So the min() of
        the result is probably the only number you should be interested in."

    Source: https://github.com/python/cpython/blob/2587b9f64eefde803a5e0b050171ad5f6654f31b/Lib/timeit.py#L193-L203

    `timeit` is more accurate than `time.time` because:

      - it repeats the tests many times to eliminate the influence of other
        tasks on your machine, such as disk flushing and OS scheduling.
      - it disables the garbage collector to prevent that process from skewing
        the results by scheduling a collection run at an inopportune moment.
      - it picks the most accurate timer for your OS, ``time.time`` or
        ``time.clock`` in Python 2 and ``time.perf_counter()`` on Python 3.
        See ``timeit.default_timer``.

    Source: https://stackoverflow.com/questions/17579357/time-time-vs-timeit-timeit

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


# %% STATIC VARIABLES
# -------------------
DIR = (
    "/p/css03/esgf_publish/CMIP6/CMIP/MOHC/HadGEM3-GC31-MM/historical/r2i1p1f3"
    "/day/ta/gn/v20191218"
)
VAR = "ta"

# xCDAT API Calls for `get_runtime`
XC_SPATIAL_AVG_STMT = f"ds.spatial.average('{VAR}')"
XC_TEMP_AVG_STMT = f"ds.temporal.average('{VAR}', weighted=True)"
XC_CLIMATOLOGY_STMT = f"ds.temporal.climatology('{VAR}', freq='month', weighted=True)"
XC_DEPARTURES_STMT = f"ds.temporal.depatures('{VAR}', freq='month', weighted=True)"


# %% xCDAT SERIAL
# -------------------
# Dataset object stored here for inspection. It is not actually used in the
# performance benchmarking.
# ds_serial = xc.open_mfdataset(f"{DIR}/*.nc", chunks="auto")

XC_SERIAL_SETUP = (
    "import xcdat as xc\n" f"ds = xc.open_mfdataset(f'{DIR}/*.nc', chunks=None)\n"
)
serial_results = {
    "spatial_avg": get_runtime(XC_SERIAL_SETUP, XC_SPATIAL_AVG_STMT, repeat=5),
    "temporal_avg": get_runtime(XC_SERIAL_SETUP, XC_TEMP_AVG_STMT, repeat=5),
    "climatology": get_runtime(XC_SERIAL_SETUP, XC_CLIMATOLOGY_STMT, repeat=5),
    "departures": get_runtime(XC_SERIAL_SETUP, XC_DEPARTURES_STMT, repeat=5),
}

# %% xCDAT Parallel
# ---------------------
# 245 GB -> 125MB chunk sizes
# Dataset object stored here for inspection. It is not actually used in the
# performance benchmarking.
# ds_parallel = xc.open_mfdataset(f"{DIR}/*.nc", chunks="auto")
XC_PARALLEL_SETUP = (
    "import xcdat as xc\n" f"ds = xc.open_mfdataset(f'{DIR}/*.nc', chunks='auto')\n"
)
parallel_results = {
    "spatial_avg": get_runtime(XC_PARALLEL_SETUP, XC_SPATIAL_AVG_STMT, repeat=5),
    "temporal_avg": get_runtime(XC_PARALLEL_SETUP, XC_TEMP_AVG_STMT, repeat=5),
    "climatology": get_runtime(XC_PARALLEL_SETUP, XC_CLIMATOLOGY_STMT, repeat=5),
    "departures": get_runtime(XC_PARALLEL_SETUP, XC_DEPARTURES_STMT, repeat=5),
}

# %% CDMS2 (serial)
# ---------------------
C_SPATIAL_AVG_STMT = f"cdutil.averager(t_var, axis='xy')"
C_TEMP_AVG_STMT = f"cdutil.averager(t_var, axis='t')"
C_CLIMATOLOGY_STMT = f"cdutil.ANNUALCYCLE.climatoloy(t_var)"
C_DEPARTURES_STMT = f"cdutil.ANNUALCYCLE.depatures(t_var)"

ds_cdms = cdms2(DIR)
t_var = ds_cdms(VAR)

# %%
