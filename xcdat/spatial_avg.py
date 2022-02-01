"""Averager module that contains functions related to geospatial averaging."""
from functools import reduce
from typing import (
    Dict,
    Hashable,
    List,
    Literal,
    Optional,
    Tuple,
    TypedDict,
    Union,
    get_args,
)

import cf_xarray  # noqa: F401
import numpy as np
import xarray as xr
from dask.array.core import Array

from xcdat.axis import (
    GENERIC_AXIS_MAP,
    GenericAxis,
    _align_lon_bounds_to_360,
    _get_prime_meridian_index,
)

#: Type alias for a dictionary of axis keys mapped to their bounds.
AxisWeights = Dict[Hashable, xr.DataArray]
#: Type alias for supported axis keys for spatial averaging.
SpatialAxis = Literal["lat", "lon"]
SPATIAL_AXES: Tuple[SpatialAxis, ...] = get_args(SpatialAxis)
#: Type alias for a tuple of floats/ints for the regional selection bounds.
RegionAxisBounds = Tuple[float, float]


@xr.register_dataset_accessor("spatial")
class SpatialAverageAccessor:
    """A class to represent the SpatialAverageAccessor."""

    def __init__(self, dataset: xr.Dataset):
        self._dataset: xr.Dataset = dataset

    def spatial_avg(
        self,
        data_var: str,
        axis: Union[List[SpatialAxis], SpatialAxis] = ["lat", "lon"],
        weights: Union[Literal["generate"], xr.DataArray] = "generate",
        lat_bounds: Optional[RegionAxisBounds] = None,
        lon_bounds: Optional[RegionAxisBounds] = None,
    ) -> xr.Dataset:
        """
        Calculate the spatial average for a rectilinear grid over a (optional)
        specified regional domain.

        Operations include:

        - If a regional boundary is specified, check to ensure it is within the
          data variable's domain boundary.
        - If axis weights are not provided, get axis weights for standard axis
          domains specified in ``axis``.
        - Adjust weights to conform to the specified regional boundary.
        - Compute spatial weighted average.

        Parameters
        ----------
        data_var: str
            The name of the data variable inside the dataset to spatially
            average.
        axis : Union[List[SpatialAxis], SpatialAxis]
            List of axis dimensions or single axis dimension to average over.
            For example, ["lat", "lon"]  or "lat", by default ["lat", "lon"].
        weights : Union[Literal["generate"], xr.DataArray], optional
            If "generate", then weights are generated. Otherwise, pass a
            DataArray containing the regional weights used for weighted
            averaging. ``weights`` must include the same spatial axis dimensions
            and have the same dimensional sizes as the data variable, by default
            "generate".
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

        Examples
        --------
        Import:

        >>> import xcdat

        Open a dataset and limit to a single variable:

        >>> ds = xcdat.open_dataset("path/to/file.nc", var="tas")

        Get global average time series:

        >>> ts_global = ds.spatial.spatial_avg("tas", axis=["lat", "lon"])["tas"]

        Get time series in Nino 3.4 domain:

        >>> ts_n34 = ds.spatial.spatial_avg("ts", axis=["lat", "lon"],
        >>>     lat_bounds=(-5, 5),
        >>>     lon_bounds=(-170, -120))["ts"]

        Get zonal mean time series:

        >>> ts_zonal = ds.spatial.spatial_avg("tas", axis=['lon'])["tas"]

        Using custom weights for averaging:

        >>> # The shape of the weights must align with the data var.
        >>> self.weights = xr.DataArray(
        >>>     data=np.ones((4, 4)),
        >>>     coords={"lat": self.ds.lat, "lon": self.ds.lon},
        >>>     dims=["lat", "lon"],
        >>> )
        >>>
        >>> ts_global = ds.spatial.spatial_avg("tas", axis=["lat","lon"],
        >>>     weights=weights)["tas"]
        """
        dataset = self._dataset.copy()
        dv = dataset.get(data_var, None)
        if dv is None:
            raise KeyError(
                f"The data variable '{data_var}' does not exist in the dataset."
            )

        axis = self._validate_axis(dv, axis)

        if isinstance(weights, str) and weights == "generate":
            if lat_bounds is not None:
                self._validate_region_bounds("lat", lat_bounds)
            if lon_bounds is not None:
                self._validate_region_bounds("lon", lon_bounds)
            dv_weights = self._get_weights(axis, lat_bounds, lon_bounds)
        elif isinstance(weights, xr.DataArray):
            dv_weights = weights

        self._validate_weights(dv, axis, dv_weights)
        dataset[dv.name] = self._averager(dv, axis, dv_weights)
        return dataset

    def _validate_axis(
        self, data_var: xr.DataArray, axis: Union[List[SpatialAxis], SpatialAxis]
    ) -> List[SpatialAxis]:
        """Validates if ``axis`` arg is supported and exists in the data var.

        Parameters
        ----------
        data_var : xr.DataArray
            The data variable.
        axis : Union[List[SpatialAxis], SpatialAxis]
            List of axis dimensions or single axis dimension to average over.

        Returns
        -------
        List[SpatialAxis]
            List of axis dimensions or single axis dimension to average over.

        Raises
        ------
        ValueError
            If any key in ``axis`` is not supported for spatial averaging.
        KeyError
            If any key in ``axis`` does not exist in the ``data_var``.
        """
        if isinstance(axis, str):
            axis = [axis]

        for key in axis:
            if key not in SPATIAL_AXES:
                raise ValueError(
                    "Incorrect `axis` argument. Supported axes include: "
                    f"{', '.join(SPATIAL_AXES)}."
                )

            generic_axis_key = GENERIC_AXIS_MAP[key]
            try:
                data_var.cf.axes[generic_axis_key]
            except KeyError:
                raise KeyError(
                    f"The data variable '{data_var.name}' is missing the '{axis}' "
                    "dimension, which is required for spatial averaging."
                )

        return axis

    def _validate_domain_bounds(self, domain_bounds: xr.DataArray):
        """Validates the ``domain_bounds`` arg based on a set of criteria.

        Parameters
        ----------
        domain_bounds: xr.DataArray
            The bounds of an axis

        Raises
        ------
        TypeError
            If the ``domain_bounds`` of a grid cell are not ordered low-to-high.
        """
        index_bad_cells = np.where(domain_bounds[:, 1] - domain_bounds[:, 0] < 0)[0]
        if len(index_bad_cells) > 0:
            raise ValueError(
                "The bounds have unexpected ordering. A lower bound has a "
                "greater value than an upper bound."
            )

    def _validate_region_bounds(self, axis: SpatialAxis, bounds: RegionAxisBounds):
        """Validates the ``bounds`` arg based on a set of criteria.

        Parameters
        ----------
        axis : SpatialAxis
            The axis related to the bounds.
        bounds : RegionAxisBounds
            The axis bounds.

        Raises
        ------
        TypeError
            If ``bounds`` is not a tuple.
        ValueError
            If the ``bounds`` has 0 elements or greater than 2 elements.
        TypeError
            If the ``bounds`` lower bound is not a float or integer.
        TypeError
            If the ``bounds`` upper bound is not a float or integer.
        ValueError
            If the ``axis`` is "lat" and the ``bounds`` lower value is larger
            than the upper value.
        """
        if not isinstance(bounds, tuple):
            raise TypeError(
                f"The {axis} regional bounds is not a tuple representing the lower and "
                "upper bounds, (lower, upper)."
            )

        if len(bounds) <= 0 or len(bounds) > 2:
            raise ValueError(
                f"The {axis} regional bounds is not a length of 2 (lower, upper)."
            )

        lower, upper = bounds
        if not isinstance(lower, float) and not isinstance(lower, int):
            raise TypeError(
                f"The regional {axis} lower bound is not a float or an integer."
            )

        if not isinstance(upper, float) and not isinstance(upper, int):
            raise TypeError(
                f"The regional {axis} upper bound is not a float or an integer."
            )

        # For latitude, require that the upper bound be larger than the lower
        # bound. Note that this does not apply to longitude (since it is
        # a circular axis).
        if axis == "lat" and lower >= upper:
            raise ValueError(
                f"The regional {axis} lower bound is greater than the upper. "
                "Pass a tuple with the format (lower, upper)."
            )

    def _get_weights(
        self,
        axis: List[SpatialAxis],
        lat_bounds: Optional[RegionAxisBounds],
        lon_bounds: Optional[RegionAxisBounds],
    ) -> xr.DataArray:
        """
        Get area weights for specified axis keys and an optional target domain.

        This method first determines the weights for an individual axis based on
        the difference between the upper and lower bound. For latitude the
        weight is determined by the difference of sine(latitude). All axis
        weights are then combined to form a DataArray of weights that can be
        used to perform a weighted (spatial) average.

        If ``lat_bounds`` or ``lon_bounds`` are supplied, then grid cells
        outside this selected regional domain are given zero weight. Grid cells
        that are partially in this domain are given partial weight.

        Parameters
        ----------
        axis : Union[List[SpatialAxis], SpatialAxis]
            List of axis dimensions or single axis dimension to average over.
            For example, ["lat", "lon"]  or "lat".
        lat_bounds : Optional[RegionAxisBounds]
            Tuple of latitude boundaries for regional selection.
        lon_bounds : Optional[RegionAxisBounds]
            Tuple of longitude boundaries for regional selection.

        Returns
        -------
        xr.DataArray
           A DataArray containing the region weights to use during averaging.
           ``weights`` are 1-D and correspond to the specified axes (``axis``)
           in the region.

        Notes
        -----
        This method was developed for rectilinear grids only. ``_get_weights()``
        recognizes and operate on latitude and longitude, but could be extended
        to work with other standard geophysical dimensions (e.g., time, depth,
        and pressure).
        """
        Bounds = TypedDict(
            "Bounds",
            {"domain": xr.DataArray, "region": Optional[RegionAxisBounds]},
        )
        axis_bounds: Dict[SpatialAxis, Bounds] = {
            "lat": {
                "domain": self._dataset.bounds.get_bounds("lat").copy(),
                "region": lat_bounds,
            },
            "lon": {
                "domain": self._dataset.bounds.get_bounds("lon").copy(),
                "region": lon_bounds,
            },
        }

        axis_weights: AxisWeights = {}
        for key in axis:
            d_bounds = axis_bounds[key]["domain"]
            self._validate_domain_bounds(d_bounds)

            r_bounds: Union[Optional[RegionAxisBounds], np.ndarray] = axis_bounds[key][
                "region"
            ]
            if r_bounds is not None:
                r_bounds = np.array(r_bounds, dtype="float")

            if key == "lon":
                weights = self._get_longitude_weights(d_bounds, r_bounds)
            elif key == "lat":
                weights = self._get_latitude_weights(d_bounds, r_bounds)

            weights.attrs = d_bounds.attrs
            weights.name = key + "_wts"
            axis_weights[key] = weights

        weights = self._combine_weights(axis_weights)
        return weights

    def _get_longitude_weights(
        self, domain_bounds: xr.DataArray, region_bounds: Optional[np.ndarray]
    ) -> xr.DataArray:
        """Gets weights for the longitude axis.

        This method performs longitudinal processing including (in order):

        1. Align the axis orientations of the domain and region bounds to
           (0, 360) to ensure compatibility in the proceeding steps.
        2. Handle grid cells that cross the prime meridian (e.g., [-1, 1]) by
           breaking such grid cells into two (e.g., [0, 1] and [359, 360]) to
           ensure alignment with the (0, 360) axis orientation. This results in
           a bounds axis of length(nlon)+1. The index of the grid cell that
           crosses the prime meridian is returned in order to reduce the length
           of weights to nlon.
        3. Scale the domain down to a region (if selected).
        4. Calculate weights using the domain bounds.
        5. If the prime meridian grid cell exists, use this cell's index to
           handle the weights vector's increased length as a result of the two
           additional grid cells. The extra weights are added to the prime
           meridian grid cell and removed from the weights vector to ensure the
           lengths of the weights and its corresponding domain remain in
           alignment.

        Parameters
        ----------
        domain_bounds : xr.DataArray
            The array of bounds for the latitude domain.
        region_bounds : Optional[np.ndarray]
            The array of bounds for latitude regional selection.

        Returns
        -------
        xr.DataArray
            The longitude axis weights.
        """
        p_meridian_index: Optional[np.ndarray] = None
        d_bounds = domain_bounds.copy()

        if region_bounds is not None:
            d_bounds: xr.DataArray = self._swap_lon_axis(d_bounds, to=360)  # type: ignore
            r_bounds: np.ndarray = self._swap_lon_axis(
                region_bounds, to=360
            )  # type:ignore

            is_region_circular = r_bounds[1] - r_bounds[0] == 0
            if is_region_circular:
                r_bounds = np.array([0.0, 360.0])

            p_meridian_index = _get_prime_meridian_index(d_bounds)
            if p_meridian_index is not None:
                d_bounds = _align_lon_bounds_to_360(d_bounds, p_meridian_index)

            d_bounds = self._scale_domain_to_region(d_bounds, r_bounds)

        weights = self._calculate_weights(d_bounds)
        if p_meridian_index is not None:
            weights[p_meridian_index] = weights[p_meridian_index] + weights[-1]
            weights = weights[0:-1]

        return weights

    def _get_latitude_weights(
        self, domain_bounds: xr.DataArray, region_bounds: Optional[np.ndarray]
    ) -> xr.DataArray:
        """Gets weights for the latitude axis.

        This method scales the domain to a region (if selected). It also scales
        the area between two lines of latitude as the difference of the sine of
        latitude bounds.

        Parameters
        ----------
        domain_bounds : xr.DataArray
            The array of bounds for the latitude domain.
        region_bounds : Optional[np.ndarray]
            The array of bounds for latitude regional selection.

        Returns
        -------
        xr.DataArray
            The latitude axis weights.
        """
        if region_bounds is not None:
            domain_bounds = self._scale_domain_to_region(domain_bounds, region_bounds)

        d_bounds = np.sin(np.radians(domain_bounds))
        weights = self._calculate_weights(d_bounds)  # type: ignore
        return weights

    def _calculate_weights(self, domain_bounds: xr.DataArray):
        """Calculate weights for the domain.

        This method takes the absolute difference between the upper and lower
        bound values to calculate weights.

        Parameters
        ----------
        domain_bounds : xr.DataArray
            The array of bounds for a domain.

        Returns
        -------
        xr.DataArray
            The weights for an axes.
        """
        return np.abs(domain_bounds[:, 1] - domain_bounds[:, 0])

    def _swap_lon_axis(
        self, lon: Union[xr.DataArray, np.ndarray], to: Literal[180, 360]
    ) -> Union[xr.DataArray, np.ndarray]:
        """Swap the longitude axis orientation.

        Parameters
        ----------
        lon : Union[xr.DataArray, np.ndarray]
             Longitude values to convert.
        to : Literal[180, 360]
            Axis orientation to convert to, either 180 [-180, 180) or 360
            [0, 360).

        Returns
        -------
        Union[xr.DataArray, np.ndarray]
            Converted longitude values.

        Notes
        -----
        This does not reorder the values in any way; it only converts the values
        in-place between longitude conventions [-180, 180) or [0, 360).
        """
        lon_swap = lon.copy()

        # If chunking, must convert convert the xarray data structure from lazy
        # Dask arrays into eager, in-memory NumPy arrays before performing
        # manipulations on the data. Otherwise, it raises `NotImplementedError
        # xarray can't set arrays with multiple array indices to dask yet`.
        if type(lon_swap.data) == Array:
            lon_swap.load()  # type: ignore

        # Must set keep_attrs=True or the xarray DataArray attrs will get
        # dropped. This has no affect on NumPy arrays.
        with xr.set_options(keep_attrs=True):
            if to == 180:
                lon_swap = ((lon_swap + 180) % 360) - 180
            elif to == 360:
                lon_swap = lon_swap % 360
            else:
                raise ValueError(
                    "Only longitude axis orientation 180 or 360 is supported."
                )

        return lon_swap

    def _scale_domain_to_region(
        self, domain_bounds: xr.DataArray, region_bounds: np.ndarray
    ) -> xr.DataArray:
        """
        Scale domain bounds to conform to a regional selection in order to
        calculate spatial weights.

        Axis weights are determined by the difference between the upper
        and lower boundary. If a region is selected, the grid cell
        bounds outside the selected region are adjusted using this method
        so that the grid cell bounds match the selected region bounds. The
        effect of this adjustment is to give partial weight to grid cells
        that are partially in the selected regional domain and zero weight
        to grid cells outside the selected domain.

        Parameters
        ----------
        domain_bounds : xr.DataArray
            The domain's bounds.
        region_bounds : np.ndarray
            The region bounds that the domain bounds are scaled down to.

        Returns
        -------
        xr.DataArray
            Scaled dimension bounds based on regional selection.

        Notes
        -----
        If a lower regional selection bound exceeds the upper selection bound,
        this algorithm assumes that the axis is longitude and the user is
        specifying a region that includes the prime meridian. The lower
        selection bound should not exceed the upper bound for latitude.
        """
        d_bounds = domain_bounds.copy()
        r_bounds = region_bounds.copy()

        if type(d_bounds.data) == Array:
            d_bounds.load()

        # Since longitude is circular, the logic depends on whether the region
        # spans across the prime meridian or not. If a region does not include
        # the prime meridian, then grid cells between the upper/lower region
        # domain values are given weight. If the prime meridian is included in
        # the domain (e.g., for a left bound of 300 degrees and a right bound
        # of 20, then the grid cells in between the region bounds (20 and 300)
        # are given zero weight (or partial weight if the grid bounds overlap
        # with the region bounds).
        if r_bounds[1] >= r_bounds[0]:
            # Case 1 (simple case): not wrapping around prime meridian (or
            # latitude axis).
            # Adjustments for above / right of region.
            d_bounds[d_bounds[:, 0] > r_bounds[1], 0] = r_bounds[1]
            d_bounds[d_bounds[:, 1] > r_bounds[1], 1] = r_bounds[1]
            # Adjustments for below / left of region.
            d_bounds[d_bounds[:, 0] < r_bounds[0], 0] = r_bounds[0]
            d_bounds[d_bounds[:, 1] < r_bounds[0], 1] = r_bounds[0]

        else:
            # Case 2: wrapping around prime meridian [for longitude only]
            domain_lowers = d_bounds[:, 0]
            domain_uppers = d_bounds[:, 1]
            region_lower, region_upper = r_bounds

            # Grid cell stradling lower boundary.
            inds = np.where(
                (domain_lowers < region_lower) & (domain_uppers > region_lower)
            )[0]
            d_bounds[inds, 0] = region_lower

            # Grid cells in between boundaries (i.e., outside selection domain).
            inds = np.where(
                (domain_lowers >= region_upper) & (domain_uppers <= region_lower)
            )[0]
            # Set upper and lower grid cell boundaries to upper edge of
            # regional domain. This will mean the grid cell upper and lower
            # boundary are equal. Therefore their difference will be zero
            # and their weight will also be zero.
            d_bounds[inds, :] = region_upper

            # Grid cell stradling upper boundary.
            inds = np.where(
                (domain_lowers < region_upper) & (domain_uppers > region_upper)
            )[0]
            d_bounds[inds, 1] = r_bounds[1]

        return d_bounds

    def _combine_weights(self, axis_weights: AxisWeights) -> xr.DataArray:
        """Generically rescales axis weights for a given region.

        This method creates an n-dimensional weighting array by performing
        matrix multiplication for a list of specified axis keys using a
        dictionary of axis weights.

        Parameters
        ----------
        axis_weights : AxisWeights
            Dictionary of axis weights, where key is axis and value is the
            corresponding DataArray of weights.

        Returns
        -------
        xr.DataArray
            A DataArray containing the region weights to use during averaging.
            ``weights`` are 1-D and correspond to the specified axis keys
            (``axis``) in the region.
        """
        region_weights = reduce((lambda x, y: x * y), axis_weights.values())
        return region_weights

    def _validate_weights(
        self, data_var: xr.DataArray, axis: List[SpatialAxis], weights: xr.DataArray
    ):
        """Validates the ``weights`` arg based on a set of criteria.

        This methods checks for the dimensional alignment between the
        ``weights`` and ``data_var``. It assumes that ``data_var`` has the same
        keys that are specified  in ``axis``, which has already been validated
        using ``self._validate_axis()`` in ``self.spatial_avg()``.

        Parameters
        ----------
        data_var : xr.DataArray
            The data variable used for validation with user supplied weights.
        axis : List[SpatialAxis]
            List of axes dimension(s) average over.
        weights : xr.DataArray
            A DataArray containing the region area weights for averaging.
            ``weights`` must include the same spatial axis dimensions found in
            ``axis`` and ``data_var``, and the same axis dims sizes as
            ``data_var``.
        Raises
        ------
        KeyError
            If ``weights`` does not include the latitude dimension.
        KeyError
            If ``weights`` does not include the longitude dimension.
        ValueError
            If the axis dimension sizes between ``weights`` and ``data_var``
            are misaligned.
        """
        # Check that the supplied weights include lat and lon dimensions.
        lat_key = data_var.cf.axes["Y"][0]
        lon_key = data_var.cf.axes["X"][0]

        if "lat" in axis and lat_key not in weights.dims:
            raise KeyError(f"Check weights DataArray includes {lat_key} dimension.")
        if "lon" in axis and lon_key not in weights.dims:
            raise KeyError(f"Check weights DataArray includes {lon_key} dimension.")

        # Check the weight dim sizes equal data var dim sizes.
        dim_sizes = {key: data_var.sizes[key] for key in weights.sizes.keys()}
        for dim, size in weights.sizes.items():
            if size != dim_sizes[dim]:
                raise ValueError(
                    f"The axis dimension sizes between supplied `weights` {dict(weights.sizes)} "
                    f"and the data variable {dim_sizes} are misaligned."
                )

    def _averager(
        self, data_var: xr.DataArray, axis: List[SpatialAxis], weights: xr.DataArray
    ):
        """Perform a weighted average of a data variable.

        This method assumes all specified keys in ``axis`` exists in the data
        variable. Validation for this criteria is performed in
        ``_validate_weights()``.

        Operations include:

        - Masked (missing) data receives zero weight.
        - Perform weighted average over user-specified axes/axis.

        Parameters
        ----------
        data_var : xr.DataArray
            Data variable inside a Dataset.
        axis : List[SpatialAxis]
            List of axis dimensions or single axis dimension to average over.
            For example, ["lat", "lon"]  or "lat".
        weights : xr.DataArray
            A DataArray containing the region area weights for averaging.
            ``weights`` must include the same spatial axis dimensions and have
            the same sizes as the data variable.

        Returns
        -------
        xr.DataArray
            Variable that has been reduced via a weighted average.

        Notes
        -----
        ``weights`` must be a DataArray and cannot contain missing values.
        Missing values are replaced with 0 using ``weights.fillna(0)``.
        """
        weights = weights.fillna(0)
        with xr.set_options(keep_attrs=True):
            weighted_mean = data_var.cf.weighted(weights).mean(
                self._get_generic_axis_keys(axis)
            )
            return weighted_mean

    def _get_generic_axis_keys(self, axis: List[SpatialAxis]) -> List[GenericAxis]:
        """Converts supported axis keys to their generic CF representations.

        Since XCDAT's spatial averaging accepts the CF short version of axes
        keys, attempting to index a Dataset/DataArray on the short key through
        cf_xarray might fail for cases where the long key is used instead (e.g.,
        "latitude" instead of "lat"). This method handles this edge case by
        converting the list of axis keys to their generic representations (e.g.,
        "Y" instead of "lat") for indexing operations.

        Parameters
        ----------
        axis_keys : List[SpatialAxis]
            List of axis dimension(s) to average over.

        Returns
        -------
        List[GenericAxis]
            List of axis dimension(s) to average over.
        """
        generic_axis_keys = []
        for key in axis:
            generic_axis_keys.append(GENERIC_AXIS_MAP[key])

        return generic_axis_keys
