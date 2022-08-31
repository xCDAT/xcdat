from xcdat.regridder.accessor import RegridderAccessor
from xcdat.regridder.regrid2 import Regrid2Regridder
# esmpy, xesmf dependency not solved for osx-arm64 https://github.com/conda-forge/esmpy-feedstock/issues/55
# _importorskip is another option if these accumulate https://github.com/pydata/xarray/blob/main/xarray/tests/__init__.py#L29-L60
# discussion: https://github.com/xCDAT/xcdat/issues/315
try:
    from xcdat.regridder.xesmf import XESMFRegridder
except ImportError:
    print("xesmf module not available")
    pass
