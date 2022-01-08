import abc
from typing import Any, Optional

import xarray as xr


class BaseRegridder(abc.ABC):
    def __init__(self, src_grid: xr.Dataset, dst_grid: xr.Dataset, **options: Any):
        self._src_grid = src_grid
        self._dst_grid = dst_grid
        self._options = options

    @abc.abstractmethod
    def regrid(
        self, ds: xr.Dataset, data_var: Optional[str] = None
    ) -> xr.Dataset:  # pragma: no cover
        pass
