from typing import Any

import xarray as xr
import xesmf as xe

from xcdat import dataset
from xcdat.regridder.base import BaseRegridder


class XESMFRegridder(BaseRegridder):
    def __init__(
        self,
        src_grid: xr.Dataset,
        dst_grid: xr.Dataset,
        method: str,
        data_var: str = None,
        **options: Any,
    ):
        self._src_grid = src_grid
        self._dst_grid = dst_grid

        if data_var is None:
            src_var = dataset.get_inferred_var(src_grid)
        else:
            src_var = src_grid.get(data_var)

            if src_var is None:
                raise KeyError(
                    f"The data variable {data_var!r} does not exist in the dataset."
                )

        self._regridder = xe.Regridder(src_var, self._dst_grid, method)

    def regrid(self, ds: xr.Dataset) -> xr.Dataset:
        return self._regridder(ds)
