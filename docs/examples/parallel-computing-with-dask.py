# %%
import collections
import timeit
from typing import DefaultDict, Dict, List, Literal

import numpy as np


# %%
def get_runtime(setup: str, stmt: str, repeat: int = 5, number: int = 1) -> float:
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


# %% xCDAT
# ----------
def get_xcdat_runtimes(
    dir_path: str,
    var: str,
    type: Literal["serial", "parallel"],
) -> Dict[str, float]:
    """Get the cdms2 runtimes for spatial and temporal averaging.

    Parameters
    ----------
    dir_path : str
        The path to the directory containing `.nc` datasets.
    var : str
        The variable to operate on.
    type : Literal[&quot;serial&quot;, &quot;parallel&quot;]
        Whether to run the API serially or in parallel.

    Returns
    -------
    Dict[str, float]
        A dictionary mapping the API to the runtime.
    """
    if type == "serial":
        chunks = None
        use_flox = "with xr.set_options(use_flox=False): \n    "
    elif type == "parallel":
        chunks = "auto"
        use_flox = "with xr.set_options(use_flox=True): \n    "

    setup = (
        "import xcdat as xc\n"
        "import xarray as xr\n"
        f"ds = xc.open_mfdataset(f'{dir_path}/*.nc', chunks={chunks})\n"
    )
    api_calls = {
        "spatial_avg": f"ds.spatial.average('{var}')",
        "temporal_avg": f"ds.temporal.average('{var}', weighted=True)",
        "climatology": f"ds.temporal.climatology('{var}', freq='month', weighted=True)",
        # "departures": f"ds.temporal.departures('{var}', freq='month', weighted=True)",
    }

    runtimes = {}
    for api, stmt in api_calls.items():
        stmt = use_flox + stmt
        runtimes[api] = get_runtime(setup, stmt, repeat=1)

    return runtimes


# %%
FILEPATHS = {
    "7 GB": "/p/css03/esgf_publish/CMIP6/CMIP/NCAR/CESM2/historical/r1i1p1f1/day/tas/gn/v20190308/",
    # "17 GB": "/p/css03/cmip5_css01/data/cmip5/output1/CNRM-CERFACS/CNRM-CM5/historical/day/atmos/day/r1i1p1/v20120530/ta/",
    # "12 GB": "/p/css03/esgf_publish/CMIP6/CMIP/MRI/MRI-ESM2-0/amip/r1i1p1f1/3hr/tas/gn/v20190829/",
    # "22 GB": "/p/css03/esgf_publish/CMIP6/CMIP/MOHC/UKESM1-0-LL/historical/r5i1p1f3/day/ta/gn/v20191115/",
    # "50 GB": "/p/css03/esgf_publish/CMIP6/CMIP/NCAR/CESM2/historical/r1i1p1f1/day/ta/gn/v20190308/",
    # "74 GB": "/p/css03/esgf_publish/CMIP6/CMIP/CCCma/CanESM5/historical/r1i1p2f1/CFday/ta/gn/v20190429/",
    # "105 GB": "/p/css03/esgf_publish/CMIP6/CMIP/MOHC/HadGEM3-GC31-MM/historical/r2i1p1f3/day/ta/gn/v20191218",
}

# %%
xcdat_serial_runtimes: DefaultDict[str, Dict[str, float]] = collections.defaultdict(
    dict
)
xcdat_parallel_runtimes: DefaultDict[str, Dict[str, float]] = collections.defaultdict(
    dict
)

for filesize, path in FILEPATHS.items():
    xcdat_serial_runtimes[filesize] = get_xcdat_runtimes(path, "tas", "serial")
    # xcdat_runtimes[filesize]["parallel"] = get_xcdat_runtimes(path, "tas", "parallel")


# %% CDMS2 (serial)
# ---------------------
def get_cdms2_runtimes(
    cdml_filepath: str, var: str, repeat: int = 1
) -> Dict[str, float]:
    """Get the cdms2 runtimes for spatial and temporal averaging.

    Parameters
    ----------
    xml_path : str
        The path to the CDML file that maps to a multi-file dataset.
    var : str
        The variable to operate on.
    repeat : int
        Number of samples to take for each API call, by default 1.


    Returns
    -------
    Dict[str, float]
        A dictionary mapping the API to the runtime.
    """
    setup = (
        "import cdms2\n"
        "import cdutil\n"
        f"ds = cdms2.open('{cdml_filepath}')\n"
        f"t_var = ds['{var}']"
    )
    api_calls = {
        "spatial_avg": "cdutil.averager(t_var, axis='xy')",
        "temporal_avg": "cdutil.averager(t_var, axis='t')",
        "climatology": "cdutil.ANNUALCYCLE.climatology(t_var)",
        # "departures": "cdutil.ANNUALCYCLE.departures(t_var)",
    }

    runtimes = {}
    for api, stmt in api_calls.items():
        runtimes[api] = get_runtime(setup, stmt, repeat=repeat)

    return runtimes


# %%
# They are stored here for data on climate machines: /p/user_pub/xclim
# You can also generate them from the command line cdscan -x myxml.xml /full/path/to/file/*nc
# /p/user_pub/xclim/$MIP_ERA/$ACTIVITY/$EXPERIMENT/$REALM/$FREQUENCY/$VARIABLE/
# filename: MIP_ERA.ACTIVITY.EXPERIMENT.INSTITUTION.MODEL.MEMBER.FREQUENCY.VARIABLE.REALM.GRID.VERSION.FLAGS.LATEST.xml
XML_FILEPATHS = {
    "7 GB": "/home/vo13/xCDAT/xcdat/input/485-xml/CMIP6.CMIP.historical.NCAR.CESM2.r1i1p1f1.day.tas.gn.v20190308.0000000.0.xml",
    # "105 GB": "/p/user_pub/xclim/CMIP6/CMIP/historical/atmos/day/ta/"
    # "CMIP6.CMIP.historical.MOHC.HadGEM3-GC31-MM.r2i1p1f3.day.ta.atmos.glb-p8-gn.v20191218.0000000.0.xml",
}


# %%
cdms2_serial_runtimes: DefaultDict[str, Dict[str, float]] = collections.defaultdict(
    dict
)

for filesize, path in XML_FILEPATHS.items():
    cdms2_serial_runtimes[filesize] = get_cdms2_runtimes(path, "tas")


# %%
