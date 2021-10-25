"""Main xcdat module."""
from typing import Any, Dict, List, Optional, Union

import xarray as xr

from xcdat.bounds import Coord, DatasetBoundsAccessor  # noqa: F401
from xcdat.regridder.accessor import DatasetRegridderAccessor  # noqa: F401
from xcdat.spatial_avg import DatasetSpatialAverageAccessor  # noqa: F401
from xcdat.spatial_avg import RegionAxisBounds, SupportedAxes
from xcdat.utils import is_documented_by


@xr.register_dataset_accessor("xcdat")
class XCDATAccessor:
    """
    A class representing XCDATAccessor, which combines all of the public methods
    of XCDAT accessors into a centralized namespace ("xcdat").

    Examples
    ========
    Import xcdat module:

    >>> import xcdat

    Access ``XCDATAccessor`` methods:

    >>> ds.xcdat.<name_of_method>
    >>>
    >>> # Spatial averaging
    >>> ds.xcdat.spatial_avg(...)
    """

    def __init__(self, dataset: xr.Dataset):
        self._dataset: xr.Dataset = dataset

    @is_documented_by(DatasetSpatialAverageAccessor.avg)
    def spatial_avg(
        self,
        data_var: Optional[str] = None,
        axis: Union[List[SupportedAxes], SupportedAxes] = ["lat", "lon"],
        weights: xr.DataArray = None,
        lat_bounds: Optional[RegionAxisBounds] = None,
        lon_bounds: Optional[RegionAxisBounds] = None,
    ) -> xr.Dataset:
        obj = DatasetSpatialAverageAccessor(self._dataset)
        return obj.avg(data_var, axis, weights, lat_bounds, lon_bounds)

    @is_documented_by(DatasetRegridderAccessor.regrid)
    def regrid(
        self, dst_grid: xr.Dataset, tool: str, method: str, **options: Any
    ) -> xr.Dataset:
        obj = DatasetRegridderAccessor(self._dataset)
        return obj.regrid(dst_grid, tool, method, **options)

    @property  # type: ignore
    @is_documented_by(DatasetBoundsAccessor.bounds)
    def bounds(self) -> Dict[str, Optional[xr.DataArray]]:
        obj = DatasetBoundsAccessor(self._dataset)
        return obj.bounds

    @is_documented_by(DatasetBoundsAccessor.fill_missing)
    def fill_missing_bounds(self) -> xr.Dataset:
        obj = DatasetBoundsAccessor(self._dataset)
        return obj.fill_missing()

    @is_documented_by(DatasetBoundsAccessor.get_bounds)
    def get_bounds(self, coord: Coord) -> xr.DataArray:
        obj = DatasetBoundsAccessor(self._dataset)
        return obj.get_bounds(coord)

    @is_documented_by(DatasetBoundsAccessor.add_bounds)
    def add_bounds(self, coord: Coord, width: float = 0.5) -> xr.Dataset:
        obj = DatasetBoundsAccessor(self._dataset)
        return obj.add_bounds(coord, width)
