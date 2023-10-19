# %%
"""
Info on CDML files:
  - They are stored here for data on climate machines: /p/user_pub/xclim
  - You can also generate them from the command line cdscan -x myxml.xml /full/path/to/file/*nc
  - `/p/user_pub/xclim/$MIP_ERA/$ACTIVITY/$EXPERIMENT/$REALM/$FREQUENCY/$VARIABLE/`
  - `filename: MIP_ERA.ACTIVITY.EXPERIMENT.INSTITUTION.MODEL.MEMBER.FREQUENCY.VARIABLE.REALM.GRID.VERSION.FLAGS.LATEST.xml`

cdscan -x /p/user_pub/e3sm/vo13/xclim/CMIP6/CMIP/historical/atmos/day/tas/CMIP6.CMIP.historical.NCAR.CESM2.r1i1p1f1.day.tas.atmos.glb-p8-gn.v20190308.0000000.0.xml /p/css03/esgf_publish/CMIP6/CMIP/NCAR/CESM2/historical/r1i1p1f1/day/tas/gn/v20190308/*.nc && cdscan -x /p/user_pub/e3sm/vo13/xclim/CMIP6/CMIP/historical/atmos/3hr/tas/CMIP6.CMIP.historical.MRI.MRI-ESM2-0.r1i1p1f1.3hr.tas.gn.v20190829.0000000.0.xml /p/css03/esgf_publish/CMIP6/CMIP/MRI/MRI-ESM2-0/amip/r1i1p1f1/3hr/tas/gn/v20190829/*.nc && cdscan -x /p/user_pub/e3sm/vo13/xclim/CMIP6/CMIP/historical/atmos/day/ta/CMIP6.CMIP.historical.CCCma.CanESM5.r1i1p2f1.CFday.ta.atmos.glb-p80-gn.v20190429.0000000.0.xml /p/css03/esgf_publish/CMIP6/CMIP/CCCma/CanESM5/historical/r1i1p2f1/CFday/ta/gn/v20190429/*.nc
"""
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
    "22_gb": {
        "var_key": "ta",
        "dir_path": "/p/css03/esgf_publish/CMIP6/CMIP/MOHC/UKESM1-0-LL/historical/r5i1p1f3/day/ta/gn/v20191115/",
        "xml_path": "/p/user_pub/xclim/CMIP6/CMIP/historical/atmos/day/ta/CMIP6.CMIP.historical.MOHC.UKESM1-0-LL.r5i1p1f3.day.ta.atmos.glb-p8-gn.v20191115.0000000.0.xml",
    },
    "50_gb": {
        "var_key": "ta",
        "dir_path": "/p/css03/esgf_publish/CMIP6/CMIP/NCAR/CESM2/historical/r1i1p1f1/day/ta/gn/v20190308/",
        "xml_path": "/p/user_pub/xclim/CMIP6/CMIP/historical/atmos/day/ta/CMIP6.CMIP.historical.NCAR.CESM2.r1i1p1f1.day.ta.atmos.glb-p8-gn.v20190308.0000000.0.xml",
    },
    "74_gb": {
        "var_key": "ta",
        "dir_path": "/p/css03/esgf_publish/CMIP6/CMIP/CCCma/CanESM5/historical/r1i1p2f1/CFday/ta/gn/v20190429/",
        "xml_path": "/p/user_pub/e3sm/vo13/xclim/CMIP6/CMIP/historical/atmos/day/ta/CMIP6.CMIP.historical.CCCma.CanESM5.r1i1p2f1.CFday.ta.atmos.glb-p80-gn.v20190429.0000000.0.xml",
    },
    "105_gb": {
        "var_key": "ta",
        "dir_path": "/p/css03/esgf_publish/CMIP6/CMIP/MOHC/HadGEM3-GC31-MM/historical/r2i1p1f3/day/ta/gn/v20191218",
        "xml_path": "/p/user_pub/xclim/CMIP6/CMIP/historical/atmos/day/ta/CMIP6.CMIP.historical.MOHC.HadGEM3-GC31-MM.r2i1p1f3.day.ta.atmos.glb-p8-gn.v20191218.0000000.0.xml",
    },
}


# CDML for "/p/css03/esgf_publish/CMIP6/CMIP/NCAR/CESM2/historical/r1i1p1f1/day/tas/gn/v20190308/"


def get_spatial_averages(
    fsize: str, finfo: Dict[str, str]
) -> Tuple[xr.DataArray, xr.DataArray, np.ndarray]:
    var_key, dir_path, xml_path = (
        finfo["var_key"],
        finfo["dir_path"],
        finfo["xml_path"],
    )

    print(
        f"Variable: '{var_key}', File size: {fsize}, Dir Path: {dir_path}, XML Path: {xml_path} "
    )

    print("1. xCDAT Serial Spatial Average")
    result_xc_serial = _get_xc_spatial_avg(
        var_key, dir_path, chunks=None, parallel=False
    )

    print("2. xCDAT Parallel Spatial Average")
    result_xc_parallel = _get_xc_spatial_avg(
        var_key, dir_path, chunks={"time": "auto"}, parallel=True
    )

    print("3. CDAT Spatial Average (Serial-Only)")
    result_cdat = _get_cdat_spatial_avg(var_key, xml_path)

    return result_xc_serial[var_key], result_xc_parallel[var_key], result_cdat


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

    # Just making sure the dataset is for sure closed.
    ds.close()

    return result


# %%
# Get spatial averaging outputs.
xc_7gb_s, xc_7gb_p, cdat_7gb = get_spatial_averages("7 GB", FILES_DICT["7_gb"])
xc_12gb_s, xc_12_gb_p, cdat_12gb = get_spatial_averages("12 GB", FILES_DICT["12_gb"])


# %%
# Test case 1: xCDAT serial vs. xCDAT Parallel
np.testing.assert_allclose(xc_7gb_s.values, xc_7gb_p.values, rtol=0, atol=0)

# %%
# Test case 2: xCDAT serial vs. CDAT (serial-only)
np.testing.assert_allclose(xc_7gb_s.values, cdat_7gb, rtol=0, atol=0)

# AssertionError:
# Not equal to tolerance rtol=1e-07, atol=0

# Mismatched elements: 60225 / 60225 (100%)
# Max absolute difference: 0.00642914
# Max relative difference: 2.2326587e-05
#  x: array([285.212858, 285.194082, 285.187531, ..., 286.16203 , 286.127491,
#        286.190507])
#  y: array([285.21893, 285.2002 , 285.1936 , ..., 286.16824, 286.13358,
#        286.19662], dtype=float32)

# %%
# Test case 3: xCDAT serial vs. CDAT (serial-only)
np.testing.assert_allclose(xc_7gb_p.tas.data, cdat_7gb)
