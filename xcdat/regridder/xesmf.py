from typing import Any

import xarray as xr
import xesmf as xe

from xcdat import dataset
from xcdat.regridder.base import BaseRegridder


class XESMFRegridder(BaseRegridder):
    def __init__(
        self, src_grid: xr.Dataset, dst_grid: xr.Dataset, method: str, **options: Any
    ):
        self._src_grid = src_grid
        self._dst_grid = dst_grid

        src_var = dataset.get_inferred_var(src_grid)

        self._regridder = xe.Regridder(src_var, self._dst_grid, method)

    def regrid(self, ds: xr.Dataset) -> xr.Dataset:
        return self._regridder(ds)
