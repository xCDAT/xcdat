# %%
import time
import warnings
from typing import Dict, Tuple

import cdms2
import cdutil
import numpy as np
import xarray as xr
import xcdat as xc

# Silence Xarray warning: `SerializationWarning: variable 'ta' has multiple fill
# values {1e+20, 1e+20}, decoding all values to NaN.`
warnings.filterwarnings(
    action="ignore", category=xr.SerializationWarning, module=".*conventions"
)

# Input data configurations
# -------------------------
# Only test 7 GB and 12 GB because 22 GB + crashes CDAT (memory allocation)
FILES_DICT: Dict[str, Dict[str, str]] = {
    "7_gb": {
        "var_key": "tas",
        "dir_path": "/p/css03/esgf_publish/CMIP6/CMIP/NCAR/CESM2/historical/r1i1p1f1/day/tas/gn/v20190308/",
        "xml_path": "/p/user_pub/e3sm/vo13/xclim/CMIP6/CMIP/historical/atmos/day/tas/CMIP6.CMIP.historical.NCAR.CESM2.r1i1p1f1.day.tas.atmos.glb-p8-gn.v20190308.0000000.0.xml",
    },
    "12_gb": {
        "var_key": "tas",
        "dir_path": "/p/css03/esgf_publish/CMIP6/CMIP/MRI/MRI-ESM2-0/amip/r1i1p1f1/3hr/tas/gn/v20190829/",
        "xml_path": "/p/user_pub/e3sm/vo13/xclim/CMIP6/CMIP/historical/atmos/3hr/tas/CMIP6.CMIP.historical.MRI.MRI-ESM2-0.r1i1p1f1.3hr.tas.gn.v20190829.0000000.0.xml",
    },
}


# %%
def main(
    fsize: str, finfo: Dict[str, str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    var_key = finfo["var_key"]
    dir_path = finfo["dir_path"]
    xml_path = finfo["xml_path"]

    print(
        f"Variable: '{var_key}', File size: {fsize}\n"
        f"Dir Path: {dir_path}, XML Path: {xml_path} "
    )

    print("1. xCDAT Serial Spatial Average")
    xc_sa_ser = _get_xc_spatial_avg(var_key, dir_path, chunks=None, parallel=False)
    xc_sa_ser_arr = xc_sa_ser[var_key].values

    print("2. xCDAT Parallel Spatial Average")
    xc_sa_par = _get_xc_spatial_avg(
        var_key, dir_path, chunks={"time": "auto"}, parallel=True
    )
    # Make sure to load the data into memory before doing floating point
    # comparison. Otherwise it will be loaded during that operation instead.
    xc_sa_par_arr = xc_sa_par[var_key].values

    print("3. CDAT Spatial Average (Serial-Only)")
    cdat_sa = _get_cdat_spatial_avg(var_key, xml_path)

    return xc_sa_ser_arr, xc_sa_par_arr, cdat_sa


def _get_xc_spatial_avg(
    var_key: str, dir_path: str, chunks: None | Dict[str, str], parallel: bool
):
    time_start_io = time.perf_counter(), time.process_time()
    ds = xc.open_mfdataset(dir_path, chunks=chunks, parallel=parallel)  # type: ignore
    time_end_io = time.perf_counter(), time.process_time()

    print(f"  * Real I/O time: {time_end_io[0] - time_start_io[0]:.4f} seconds")
    print(f"  * CPU I/O time: {time_end_io[1] - time_start_io[1]:.4f} seconds")

    time_start = time.perf_counter(), time.process_time()
    result = ds.spatial.average(var_key, axis=["X", "Y"])
    time_end = time.perf_counter(), time.process_time()

    print(f"  * Real compute time: {time_end[0] - time_start[0]:.4f} seconds")
    print(f"  * CPU compute time: {time_end[1] - time_start[1]:.4f} seconds")

    # Just making sure the dataset is for sure closed.
    ds.close()

    return result


def _get_cdat_spatial_avg(var_key: str, xml_path: str):
    time_start_io = time.perf_counter(), time.process_time()
    ds = cdms2.open(xml_path)
    tvar = ds(var_key)
    time_end_io = time.perf_counter(), time.process_time()

    print(f"  * Real I/O time: {time_end_io[0] - time_start_io[0]:.4f} seconds")
    print(f"  * CPU I/O time: {time_end_io[1] - time_start_io[1]:.4f} seconds")

    time_start = time.perf_counter(), time.process_time()
    result = cdutil.averager(tvar, axis="xy")
    time_end = time.perf_counter(), time.process_time()

    print(f"  * Real compute time: {time_end[0] - time_start[0]:.4f} seconds")
    print(f"  * CPU compute time: {time_end[1] - time_start[1]:.4f} seconds")

    # # Just making sure the dataset is for sure closed.
    ds.close()

    return result


# %%
# Get spatial averaging outputs.
xc_7gb_s, xc_7gb_p, cdat_7gb = main("7 GB", FILES_DICT["7_gb"])
xc_12gb_s, xc_12gb_p, cdat_12gb = main("12 GB", FILES_DICT["12_gb"])


# %%
def _compare_outputs(arr_a: np.ndarray, arr_b: np.ndarray):
    np.testing.assert_allclose(arr_a, arr_b, rtol=0, atol=0)


# %% Test case 1: xCDAT serial vs. xCDAT Parallel
# Both are identical.
_compare_outputs(xc_7gb_s, xc_7gb_p)
_compare_outputs(xc_12gb_s, xc_12gb_p)

# Test Case 2: 7 GB dataset
# --------------------------------------
# NOTE: For some reason the first value of CDAT's spatial averager is missing (inf).
# We skip this value to make sure it doesn't influence the results.
_compare_outputs(xc_7gb_s[1:], cdat_7gb.data[1:])  # type: ignore
_compare_outputs(xc_7gb_p[1:], cdat_7gb.data[1:])  # type: ignore

"""
AssertionError:
Not equal to tolerance rtol=0, atol=0

Mismatched elements: 60225 / 60225 (100%)
Max absolute difference: 0.00642914
Max relative difference: 2.2326587e-05
 x: array([285.212858, 285.194082, 285.187531, ..., 286.16203 , 286.127491,
       286.190507])
 y: array([285.21893, 285.2002 , 285.1936 , ..., 286.16824, 286.13358,
       286.19662], dtype=float32)
"""
# %%
# Test Case 2: 12 GB dataset
# --------------------------------------
_compare_outputs(xc_12gb_s, cdat_12gb.data)  # type: ignore
_compare_outputs(xc_12gb_p, cdat_12gb.data)  # type: ignore

# AssertionError:
# Not equal to tolerance rtol=0, atol=0

# Mismatched elements: 105192 / 105192 (100%)
# Max absolute difference: 7.44648787e-12
# Max relative difference: 2.57343965e-14
#  x: array([285.027095, 285.062351, 285.270414, ..., 287.040131, 286.85913 ,
#        286.67638 ])
#  y: array([285.027095, 285.062351, 285.270414, ..., 287.040131, 286.85913 ,
#        286.67638 ])


"""
Variable: 'tas', File size: 7 GB
Dir Path: /p/css03/esgf_publish/CMIP6/CMIP/NCAR/CESM2/historical/r1i1p1f1/day/tas/gn/v20190308/, XML Path: /p/user_pub/e3sm/vo13/xclim/CMIP6/CMIP/historical/atmos/day/tas/CMIP6.CMIP.historical.NCAR.CESM2.r1i1p1f1.day.tas.atmos.glb-p8-gn.v20190308.0000000.0.xml
1. xCDAT Serial Spatial Average
  * Real I/O time: 1.5270 seconds
  * CPU I/O time: 1.5098 seconds
  * Real compute time: 1.4385 seconds
  * CPU compute time: 1.4395 seconds
2. xCDAT Parallel Spatial Average
  * Real I/O time: 2.4733 seconds
  * CPU I/O time: 3.3816 seconds
  * Real compute time: 0.2422 seconds
  * CPU compute time: 0.2434 seconds
3. CDAT Spatial Average (Serial-Only)
  * Real I/O time: 73.6501 seconds
  * CPU I/O time: 73.6546 seconds
  * Real compute time: 413.0872 seconds
  * CPU compute time: 413.1232 seconds
Variable: 'tas', File size: 12 GB
Dir Path: /p/css03/esgf_publish/CMIP6/CMIP/MRI/MRI-ESM2-0/amip/r1i1p1f1/3hr/tas/gn/v20190829/, XML Path: /p/user_pub/e3sm/vo13/xclim/CMIP6/CMIP/historical/atmos/3hr/tas/CMIP6.CMIP.historical.MRI.MRI-ESM2-0.r1i1p1f1.3hr.tas.gn.v20190829.0000000.0.xml
1. xCDAT Serial Spatial Average
  * Real I/O time: 16.6844 seconds
  * CPU I/O time: 3.9275 seconds
  * Real compute time: 2.5575 seconds
  * CPU compute time: 2.5589 seconds
2. xCDAT Parallel Spatial Average
  * Real I/O time: 16.3017 seconds
  * CPU I/O time: 4.6564 seconds
  * Real compute time: 0.3206 seconds
  * CPU compute time: 0.3224 seconds
3. CDAT Spatial Average (Serial-Only)
  * Real I/O time: 95.0600 seconds
  * CPU I/O time: 94.8878 seconds
  * Real compute time: 664.6169 seconds
  * CPU compute time: 664.6510 seconds
"""


# %%
"""
Info on CDML files:
  - They are stored here for data on climate machines: /p/user_pub/xclim
  - You can also generate them from the command line cdscan -x myxml.xml /full/path/to/file/*nc
  - `/p/user_pub/xclim/$MIP_ERA/$ACTIVITY/$EXPERIMENT/$REALM/$FREQUENCY/$VARIABLE/`
  - `filename: MIP_ERA.ACTIVITY.EXPERIMENT.INSTITUTION.MODEL.MEMBER.FREQUENCY.VARIABLE.REALM.GRID.VERSION.FLAGS.LATEST.xml`

cdscan -x /p/user_pub/e3sm/vo13/xclim/CMIP6/CMIP/historical/atmos/day/tas/CMIP6.CMIP.historical.NCAR.CESM2.r1i1p1f1.day.tas.atmos.glb-p8-gn.v20190308.0000000.0.xml /p/css03/esgf_publish/CMIP6/CMIP/NCAR/CESM2/historical/r1i1p1f1/day/tas/gn/v20190308/*.nc && cdscan -x /p/user_pub/e3sm/vo13/xclim/CMIP6/CMIP/historical/atmos/3hr/tas/CMIP6.CMIP.historical.MRI.MRI-ESM2-0.r1i1p1f1.3hr.tas.gn.v20190829.0000000.0.xml /p/css03/esgf_publish/CMIP6/CMIP/MRI/MRI-ESM2-0/amip/r1i1p1f1/3hr/tas/gn/v20190829/*.nc && cdscan -x /p/user_pub/e3sm/vo13/xclim/CMIP6/CMIP/historical/atmos/day/ta/CMIP6.CMIP.historical.CCCma.CanESM5.r1i1p2f1.CFday.ta.atmos.glb-p80-gn.v20190429.0000000.0.xml /p/css03/esgf_publish/CMIP6/CMIP/CCCma/CanESM5/historical/r1i1p2f1/CFday/ta/gn/v20190429/*.nc
"""
