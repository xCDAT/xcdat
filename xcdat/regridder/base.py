import abc
from typing import Any

import xarray as xr


class BaseRegridder(abc.ABC):
    """BaseRegridder."""

    def __init__(self, src_grid: xr.Dataset, dst_grid: xr.Dataset, **options: Any):
        self._src_grid = src_grid
        self._dst_grid = dst_grid
        self._options = options

    @abc.abstractmethod
    def horizontal(
        self, data_var: str, ds: xr.Dataset
    ) -> xr.Dataset:  # pragma: no cover
        pass
