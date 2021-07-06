"""Main xcdat module."""
from typing import Dict, List, Optional, Union

import xarray as xr

from xcdat.bounds import Coord, DatasetBoundsAccessor  # noqa: F401
from xcdat.spatial_avg import DatasetSpatialAverageAccessor  # noqa: F401
from xcdat.spatial_avg import RegionAxisBounds, SupportedAxes


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

    def spatial_avg(
        self,
        data_var_name: str,
        axis: Union[List[SupportedAxes], SupportedAxes],
        weights: xr.DataArray = None,
        lat_bounds: Optional[RegionAxisBounds] = None,
        lon_bounds: Optional[RegionAxisBounds] = None,
    ) -> xr.Dataset:
        """
        Calculate the spatial average for a rectilinear grid over a (optional)
        specified regional domain.

        This method is simply a wrapper for the parent method,
        ``xcdat.spatial_avg.DatasetSpatialAverageAccessor.avg()`` or
        ``ds.spatial.avg()`` (the decorator variant).

        Operations include:

        - If a regional boundary is specified, check to ensure it is within the
          data variable's domain boundary.
        - If axis weights are not provided, get axis weights for standard axes
          domains specified in ``axis``.
        - Adjust weights to conform to the specified regional boundary.
        - Compute spatial weighted average.

        Parameters
        ----------
        data_var_name: str
            The name of the data variable inside the dataset to spatially
            average.
        axis : Union[List[SupportedAxes], SupportedAxes]
            List of axis dimensions or single axes dimension to average over.
            For example, ["lat", "lon"]  or "lat", by default ["lat", "lon"].
        weights : Optional[xr.DataArray], optional
            A DataArray containing the regional weights used for weighted
            averaging. ``weights`` must include the same spatial axis dimensions
            and have the same dimensional sizes as the data variable. If None,
            then weights are generated; by default None.
        lat_bounds : Optional[RegionAxisBounds], optional
            A tuple of floats/ints for the regional latitude lower and upper
            boundaries. This arg is used when calculating axis weights, but is
            ignored if ``weights`` are supplied. The lower bound cannot be
            larger than the upper bound, by default None.
        lon_bounds : Optional[RegionAxisBounds], optional
            A tuple of floats/ints for the regional longitude lower and upper
            boundaries. This arg is used when calculating axis weights, but is
            ignored if ``weights`` are supplied. The lower bound can be larger
            than the upper bound (e.g., across the prime meridian, dateline), by
            default None.

        Returns
        -------
        xr.Dataset
            Dataset with the spatially averaged variable.

        Raises
        ------
        KeyError
            If data variable does not exist in the Dataset.
        KeyError
            If data variable does not contain an "Y" axes (latitude dimension).
        KeyError
            If data variable does not contain an "X" axes (longitude dimension).
        ValueError
            If an incorrect axes is specified in ``axis``.

        Examples
        --------
        Import:

        >>> import xcdat

        Open a dataset and limit to a single variable:

        >>> ds = xcdat.open_dataset("path/to/file.nc", var="tas")

        Get global average time series:

        >>> ts_global = ds.xcdat.spatial_avg("tas", axis=["lat", "lon"])["tas"]

        Get time series in Nino 3.4 domain:

        >>> ts_n34 = ds.xcdat.spatial_avg("tas", axis=["lat", "lon"],
        >>>     lat_bounds=(-5, 5),
        >>>     lon_bounds=(-170, -120))["tas"]

        Get zonal mean time series:

        >>> ts_zonal = ds.xcdat.spatial_avg("tas", axis=['lon'])["tas"]

        Using custom weights for averaging:

        >>> # The shape of the weights must align with the data var.
        >>> self.weights = xr.DataArray(
        >>>     data=np.ones((4, 4)),
        >>>     coords={"lat": self.ds.lat, "lon": self.ds.lon},
        >>>     dims=["lat", "lon"],
        >>> )
        >>>
        >>> ts_global = ds.xcdat.spatial_avg("tas", axis=["lat","lon"],
        >>>     weights=weights)["tas"]
        """
        obj = DatasetSpatialAverageAccessor(self._dataset)
        return obj.avg(data_var_name, axis, weights, lat_bounds, lon_bounds)

    @property
    def bounds(self) -> Dict[str, Optional[xr.DataArray]]:
        """Returns a mapping of coordinate and axis keys to their bounds.

        The dictionary provides all valid CF compliant keys for a coordinate.
        For example, latitude will includes keys for "lat", "latitude", and "Y".

        Returns
        -------
        Dict[str, Optional[xr.DataArray]]
            Dictionary mapping coordinate keys to their bounds.
        """
        obj = DatasetBoundsAccessor(self._dataset)
        return obj.bounds

    def fill_missing_bounds(self) -> xr.Dataset:
        """Fills any missing bounds for supported coordinates in the Dataset.

        Returns
        -------
        xr.Dataset
        """
        obj = DatasetBoundsAccessor(self._dataset)
        return obj.fill_missing()

    def get_bounds(self, coord: Coord) -> xr.DataArray:
        """Get bounds for a coordinate.

        Parameters
        ----------
        coord : Coord
            The coordinate key.

        Returns
        -------
        xr.DataArray
            The coordinate bounds.

        Raises
        ------
        ValueError
            If an incorrect ``coord`` argument is passed.

        ValueError
            If bounds were not found. They must be added.
        """
        obj = DatasetBoundsAccessor(self._dataset)
        return obj.get_bounds(coord)

    def add_bounds(self, coord: Coord, width: float = 0.5) -> xr.Dataset:
        """Add bounds for a coordinate using its data points.

        If bounds already exist, they must be dropped first.

        Parameters
        ----------
        coord : Coord
            The coordinate key.
        width : float, optional
            Width of the bounds relative to the position of the nearest points,
            by default 0.5.

        Returns
        -------
        xr.Dataset
            The dataset with bounds added.

        Raises
        ------
        ValueError
            If bounds already exist. They must be dropped first.
        """
        obj = DatasetBoundsAccessor(self._dataset)
        return obj.add_bounds(coord, width)
