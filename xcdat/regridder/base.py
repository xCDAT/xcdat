import abc
from typing import Any

import xarray as xr


class BaseRegridder(abc.ABC):
    def __init__(self, src_grid: xr.Dataset, dst_grid: xr.Dataset, **options: Any):
        pass

    def regrid(self, ds: xr.Dataset) -> xr.Dataset:
        pass
