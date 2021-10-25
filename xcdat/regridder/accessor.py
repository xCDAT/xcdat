from typing import Any

import xarray as xr

from xcdat.regridder import xesmf

VALID_TOOLS = {
    "xesmf": xesmf.XESMFRegridder,
}


@xr.register_dataset_accessor(name="regridder")
class DatasetRegridderAccessor:
    def __init__(self, ds: xr.Dataset):
        self._ds = ds

    def regrid(
        self, dst_grid: xr.Dataset, tool: str, method: str, **options: Any
    ) -> xr.Dataset:
        if tool not in VALID_TOOLS:
            raise ValueError(
                f"{tool!r} is not a valid tool, choices {list(VALID_TOOLS)}"
            )

        cls = VALID_TOOLS[tool]

        regridder = cls(self._ds, dst_grid, method=method, **options)

        return regridder.regrid(self._ds)
