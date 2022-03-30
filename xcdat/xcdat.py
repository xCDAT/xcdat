"""Main xcdat module."""
from typing import Any, Dict, List, Literal, Optional, Union

import xarray as xr

from xcdat.bounds import BoundsAccessor, BoundsAxis
from xcdat.regridder.accessor import DatasetRegridderAccessor, RegridTool  # noqa: F401
from xcdat.spatial import RegionAxisBounds, SpatialAccessor, SpatialAxis
from xcdat.temporal import Frequency, Mode, SeasonConfig, TemporalAccessor
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

    @is_documented_by(DatasetRegridderAccessor.regrid)
    def regrid(
        self,
        data_var: str,
        dst_grid: xr.Dataset,
        tool: RegridTool,
        **options: Any,
    ) -> xr.Dataset:
        obj = DatasetRegridderAccessor(self._dataset)
        return obj.regrid(data_var, dst_grid, tool, **options)

    @is_documented_by(SpatialAccessor.spatial_avg)
    def spatial_avg(
        self,
        data_var: str,
        axis: Union[List[SpatialAxis], SpatialAxis] = ["lat", "lon"],
        weights: Union[Literal["generate"], xr.DataArray] = "generate",
        lat_bounds: Optional[RegionAxisBounds] = None,
        lon_bounds: Optional[RegionAxisBounds] = None,
    ) -> xr.Dataset:
        obj = SpatialAccessor(self._dataset)
        return obj.spatial_avg(data_var, axis, weights, lat_bounds, lon_bounds)

    @is_documented_by(TemporalAccessor.temporal_avg)
    def temporal_avg(
        self,
        data_var: str,
        mode: Mode,
        freq: Frequency,
        weighted: bool = True,
        center_times: bool = False,
        season_config: SeasonConfig = {
            "dec_mode": "DJF",
            "drop_incomplete_djf": False,
            "custom_seasons": None,
        },
    ) -> xr.Dataset:
        obj = TemporalAccessor(self._dataset)
        return obj.temporal_avg(
            data_var, mode, freq, weighted, center_times, season_config
        )

    @is_documented_by(TemporalAccessor.departures)
    def departures(self, data_var: str) -> xr.Dataset:
        obj = TemporalAccessor(self._dataset)
        return obj.departures(data_var)

    @is_documented_by(TemporalAccessor.center_times)
    def center_times(self):
        obj = TemporalAccessor(self._dataset)
        return obj.center_times(self._dataset)

    @property  # type: ignore
    @is_documented_by(BoundsAccessor.bounds)
    def bounds(self) -> Dict[str, Optional[xr.DataArray]]:
        obj = BoundsAccessor(self._dataset)
        return obj.bounds

    @is_documented_by(BoundsAccessor.add_missing_bounds)
    def add_missing_bounds(self) -> xr.Dataset:
        obj = BoundsAccessor(self._dataset)
        return obj.add_missing_bounds()

    @is_documented_by(BoundsAccessor.get_bounds)
    def get_bounds(self, axis: BoundsAxis) -> xr.DataArray:
        obj = BoundsAccessor(self._dataset)
        return obj.get_bounds(axis)

    @is_documented_by(BoundsAccessor.add_bounds)
    def add_bounds(self, axis: BoundsAxis, width: float = 0.5) -> xr.Dataset:
        obj = BoundsAccessor(self._dataset)
        return obj.add_bounds(axis, width)
