"""Main xcdat module."""
from typing import Any, Dict, List, Optional, Union

import xarray as xr

from xcdat.bounds import BoundsAccessor, Coord
from xcdat.regridder.accessor import DatasetRegridderAccessor  # noqa: F401
from xcdat.spatial_avg import RegionAxisBounds, SpatialAverageAccessor, SupportedAxes
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

    @is_documented_by(SpatialAverageAccessor.spatial_avg)
    def spatial_avg(
        self,
        data_var: Optional[str] = None,
        axis: Union[List[SupportedAxes], SupportedAxes] = ["lat", "lon"],
        weights: xr.DataArray = None,
        lat_bounds: Optional[RegionAxisBounds] = None,
        lon_bounds: Optional[RegionAxisBounds] = None,
    ) -> xr.Dataset:
        obj = SpatialAverageAccessor(self._dataset)
        return obj.spatial_avg(data_var, axis, weights, lat_bounds, lon_bounds)

    @is_documented_by(DatasetRegridderAccessor.regrid)
    def regrid(
        self,
        dst_grid: xr.Dataset,
        tool: str,
        method: str,
        data_var: str = None,
        **options: Any,
    ) -> xr.Dataset:
        obj = DatasetRegridderAccessor(self._dataset)
        return obj.regrid(dst_grid, tool, method, data_var, **options)

    @property  # type: ignore
    @is_documented_by(BoundsAccessor.bounds)
    def bounds(self) -> Dict[str, Optional[xr.DataArray]]:
        obj = BoundsAccessor(self._dataset)
        return obj.bounds

    @is_documented_by(BoundsAccessor.fill_missing_bounds)
    def fill_missing_bounds(self) -> xr.Dataset:
        obj = BoundsAccessor(self._dataset)
        return obj.fill_missing_bounds()

    @is_documented_by(BoundsAccessor.get_bounds)
    def get_bounds(self, coord: Coord) -> xr.DataArray:
        obj = BoundsAccessor(self._dataset)
        return obj.get_bounds(coord)

    @is_documented_by(BoundsAccessor.add_bounds)
    def add_bounds(self, coord: Coord, width: float = 0.5) -> xr.Dataset:
        obj = BoundsAccessor(self._dataset)
        return obj.add_bounds(coord, width)
