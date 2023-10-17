"""According to `Timer.repeat()`:

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

Info on CDML files:
  - They are stored here for data on climate machines: /p/user_pub/xclim
  - You can also generate them from the command line cdscan -x myxml.xml /full/path/to/file/*nc
  - `/p/user_pub/xclim/$MIP_ERA/$ACTIVITY/$EXPERIMENT/$REALM/$FREQUENCY/$VARIABLE/`
  - `filename: MIP_ERA.ACTIVITY.EXPERIMENT.INSTITUTION.MODEL.MEMBER.FREQUENCY.VARIABLE.REALM.GRID.VERSION.FLAGS.LATEST.xml`

cdscan -x /p/user_pub/e3sm/vo13/xclim/CMIP6/CMIP/historical/atmos/day/tas/CMIP6.CMIP.historical.NCAR.CESM2.r1i1p1f1.day.tas.atmos.glb-p8-gn.v20190308.0000000.0.xml /p/css03/esgf_publish/CMIP6/CMIP/NCAR/CESM2/historical/r1i1p1f1/day/tas/gn/v20190308/*.nc && cdscan -x /p/user_pub/e3sm/vo13/xclim/CMIP6/CMIP/historical/atmos/3hr/tas/CMIP6.CMIP.historical.MRI.MRI-ESM2-0.r1i1p1f1.3hr.tas.gn.v20190829.0000000.0.xml /p/css03/esgf_publish/CMIP6/CMIP/MRI/MRI-ESM2-0/amip/r1i1p1f1/3hr/tas/gn/v20190829/*.nc && cdscan -x /p/user_pub/e3sm/vo13/xclim/CMIP6/CMIP/historical/atmos/day/ta/CMIP6.CMIP.historical.CCCma.CanESM5.r1i1p2f1.CFday.ta.atmos.glb-p80-gn.v20190429.0000000.0.xml /p/css03/esgf_publish/CMIP6/CMIP/CCCma/CanESM5/historical/r1i1p2f1/CFday/ta/gn/v20190429/*.nc

"""

import time

import cdutil
import cdms2
import numpy as np
import xcdat as xc

# CDML for "/p/css03/esgf_publish/CMIP6/CMIP/NCAR/CESM2/historical/r1i1p1f1/day/tas/gn/v20190308/"
xml_path = "/p/user_pub/e3sm/vo13/xclim/CMIP6/CMIP/historical/atmos/day/tas/CMIP6.CMIP.historical.NCAR.CESM2.r1i1p1f1.day.tas.atmos.glb-p8-gn.v20190308.0000000.0.xml"

ds = cdms2.open(xml_path)
tvar = ds("tas")

t1 = time.perf_counter(), time.process_time()
cd_sa = cdutil.averager(tvar, axis="xy")
t2 = time.perf_counter(), time.process_time()

print(f" Real time: {t2[0] - t1[0]:.2f} seconds")
print(f" CPU time: {t2[1] - t1[1]:.2f} seconds")
#  Real time: 417.52 seconds
#  CPU time: 417.55 seconds


# %%
dir_path = "/p/css03/esgf_publish/CMIP6/CMIP/NCAR/CESM2/historical/r1i1p1f1/day/tas/gn/v20190308/*.nc"
ds = xc.open_mfdataset(f"{dir_path}", chunks={"time": "auto"}, parallel=True)

t1_xc = time.perf_counter(), time.process_time()
xc_sa = ds.spatial.average("tas", axis=["X", "Y"])
t2_xc = time.perf_counter(), time.process_time()

print(f" Real time: {t2_xc[0] - t1_xc[0]:.2f} seconds")
print(f" CPU time: {t2_xc[1] - t1_xc[1]:.2f} seconds")

# %%
# cd_sa = cd_sa.filled(np.nan)
np.testing.assert_allclose(cd_sa[1:], xc_sa.tas.values[1:])

# AssertionError:
# Not equal to tolerance rtol=1e-07, atol=0

# Mismatched elements: 60225 / 60225 (100%)
# Max absolute difference: 0.00642914
# Max relative difference: 2.2326587e-05
#  x: array([285.212858, 285.194082, 285.187531, ..., 286.16203 , 286.127491,
#        286.190507])
#  y: array([285.21893, 285.2002 , 285.1936 , ..., 286.16824, 286.13358,
#        286.19662], dtype=float32)
