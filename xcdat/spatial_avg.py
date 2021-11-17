"""Averager module that contains functions related to geospatial averaging."""
from functools import reduce
from typing import Dict, Hashable, List, Optional, Tuple, Union

import cf_xarray  # noqa: F401
import numpy as np
import xarray as xr
from dask.array.core import Array
from typing_extensions import Literal, TypedDict, get_args

from xcdat.dataset import get_inferred_var

#: Type alias for a dictionary of axes keys mapped to their bounds.
AxisWeights = Dict[Hashable, xr.DataArray]
#: Type alias of supported axes strings for spatial averaging.
SupportedAxes = Literal["lat", "lon"]
SUPPORTED_AXES: Tuple[SupportedAxes, ...] = get_args(SupportedAxes)

#: Type alias for a tuple of floats/ints for the regional selection bounds.
RegionAxisBounds = Tuple[Union[float, int], Union[float, int]]


@xr.register_dataset_accessor("spatial")
class SpatialAverageAccessor:
    """A class to represent the SpatialAverageAccessor."""

    def __init__(self, dataset: xr.Dataset):
        self._dataset: xr.Dataset = dataset

    def spatial_avg(
        self,
        data_var: Optional[str] = None,
        axis: Union[List[SupportedAxes], SupportedAxes] = ["lat", "lon"],
        weights: xr.DataArray = None,
        lat_bounds: Optional[RegionAxisBounds] = None,
        lon_bounds: Optional[RegionAxisBounds] = None,
    ) -> xr.Dataset:
        """
        Calculate the spatial average for a rectilinear grid over a (optional)
        specified regional domain.

        Operations include:

        - If a regional boundary is specified, check to ensure it is within the
          data variable's domain boundary.
        - If axis weights are not provided, get axis weights for standard axes
          domains specified in ``axis``.
        - Adjust weights to conform to the specified regional boundary.
        - Compute spatial weighted average.

        Parameters
        ----------
        data_var: Optional[str], optional
            The name of the data variable inside the dataset to spatially
            average. If None, an inference to the desired data variable is
            attempted with the Dataset's "xcdat_infer" attr and
            ``get_inferred_var()``, by default None.
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

        if data_var is None:
            da_data_var = get_inferred_var(dataset)
        else:
            da_data_var = dataset.get(data_var, None)
            if da_data_var is None:
                raise KeyError(
                    f"The data variable '{data_var}' does not exist in the dataset."
                )

        axis = self._validate_axis(da_data_var, axis)

        if weights is None:
            if lat_bounds is not None:
                self._validate_region_bounds("lat", lat_bounds)
            if lon_bounds is not None:
                self._validate_region_bounds("lon", lon_bounds)
            weights = self._get_weights(axis, lat_bounds, lon_bounds)

        self._validate_weights(da_data_var, axis, weights)
        dataset[da_data_var.name] = self._averager(da_data_var, axis, weights)
        return dataset

    def _validate_axis(
        self, data_var: xr.DataArray, axis: Union[List[SupportedAxes], SupportedAxes]
    ) -> List[SupportedAxes]:
        """Validates the ``axis`` arg based on a set of criteria.

        This method checks if ``axis`` values are supported strings and if they
        exist in ``data_var``.

        Parameters
        ----------
        data_var : xr.DataArray
            The data variable.
        axis : Union[List[SupportedAxes], SupportedAxes]
            List of axis dimensions or single axes dimension to average over.
            For example, ["lat", "lon"]  or "lat".

        Returns
        -------
        List[SupportedAxes]
            List of axis dimensions or single axes dimension to average over.

        Raises
        ------
        ValueError
            If any value inside ``axis`` is not supported.
        KeyError
            If any value inside ``axis`` does not exist in the ``data_var``.
        """
        if isinstance(axis, str):
            axis = [axis]

        for axes in axis:
            if axes not in SUPPORTED_AXES:
                raise ValueError(
                    "Incorrect `axis` argument. Supported axes include: "
                    f"{', '.join(SUPPORTED_AXES)}."
                )

            # Must use a try and except because native xarray is not name
            # agnostic with key access using axes names (e.g., "lat" is not
            # linked to a "latitude" key ), and cf_xarray does not support
            # `data_var.cf.get(axes, None)`.
            try:
                data_var.cf[axes]
            except KeyError:
                raise KeyError(
                    f"The data variable '{data_var.name}' is missing a '{axes}' "
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
                "The bounds have unexpected ordering: "
                "A lower bound has a greater value than an "
                "upper bound"
            )

    def _validate_region_bounds(self, axis: SupportedAxes, bounds: RegionAxisBounds):
        """Validates the ``bounds`` arg based on a set of criteria.

        Parameters
        ----------
        axis : SupportedAxes
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
                "The {axis} regional bounds is not a tuple representing the lower and "
                "upper bounds, (lower, upper)."
            )

        if len(bounds) <= 0 or len(bounds) > 2:
            raise ValueError(
                "The {axis} regional bounds is not a length of 2 (lower, upper)."
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
        axis: List[SupportedAxes],
        lat_bounds: Optional[RegionAxisBounds],
        lon_bounds: Optional[RegionAxisBounds],
    ) -> xr.DataArray:
        """
        Get area weights for specified axes and an optional target domain.

        This method first determines the weights for individual axes based on
        the difference between the upper and lower bound. For latitude the
        weight is determined by the difference of sine(latitude). The individual
        axis weights are then combined to form a DataArray of weights that can
        be used to perform a weighted (spatial) average.

        If ``lat_bounds`` or ``lon_bounds`` are supplied, then grid cells
        outside this selected regional domain are given zero weight. Grid cells
        that are partially in this domain are given partial weight.

        Parameters
        ----------
        axis : Union[List[SupportedAxes], SupportedAxes]
            List of axis dimensions or single axes dimension to average over.
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
        BoundsByType = TypedDict(
            "BoundsByType",
            {"domain": xr.DataArray, "region": Optional[RegionAxisBounds]},
        )

        bounds: Dict[str, BoundsByType] = {
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

        for axes, bounds_by_type in bounds.items():
            d_bounds = bounds_by_type["domain"]
            self._validate_domain_bounds(d_bounds)

            r_bounds = bounds_by_type["region"]
            if r_bounds is not None:
                r_bounds = np.array(r_bounds, dtype="float")

            if axes == "lon":
                weights = self._get_longitude_weights(d_bounds, r_bounds)
            elif axes == "lat":
                weights = self._get_latitude_weights(d_bounds, r_bounds)

            weights.attrs = d_bounds.attrs
            weights.name = axes + "_wts"
            axis_weights[axes] = weights

        weights = self._combine_weights(axis, axis_weights)
        return weights

    def _get_longitude_weights(
        self, domain_bounds: xr.DataArray, region_bounds: np.array
    ) -> xr.DataArray:
        """Gets weights for the longitude axes.

        This method performs longitudinal processing including (in order):

        1. Align the axes orientations of the domain and region bounds to
           (0, 360) to ensure compatibility in the proceeding steps.
        2. Handle grid cells that cross the prime meridian (e.g., [-1, 1])
           by recreating the axes with two additional grid cells ([0, 1] and
           [359, 360]) to ensure alignment with the (0, 360) axes orientation.
           The prime meridian grid cell is returned as a variable to handle
           the length of weights in a proceeding step.
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
        region_bounds : np.array
            The array of bounds for latitude regional selection.

        Returns
        -------
        xr.DataArray
            The longitude axes weights.
        """
        p_meridian_index: Optional[np.array] = None

        if region_bounds is not None:
            domain_bounds = self._swap_lon_axes(domain_bounds, to=360)
            region_bounds = self._swap_lon_axes(region_bounds, to=360)

            is_region_circular = region_bounds[1] - region_bounds[0] == 0
            if is_region_circular:
                region_bounds = np.array([0.0, 360.0])

            (
                domain_bounds,
                p_meridian_index,
            ) = self._align_longitude_to_360_axis(domain_bounds)
            domain_bounds = self._scale_domain_to_region(domain_bounds, region_bounds)

        weights = self._calculate_weights(domain_bounds)
        if p_meridian_index is not None:
            weights[p_meridian_index] = weights[p_meridian_index] + weights[-1]
            weights = weights[0:-1]

        return weights

    def _get_latitude_weights(
        self, domain_bounds: xr.DataArray, region_bounds: xr.DataArray
    ) -> xr.DataArray:
        """Gets weights for the latitude axes.

        This method scales the domain to a region (if selected). It also scales
        the area between two lines of latitude scales as the difference of the
        sine of latitude bounds.

        Parameters
        ----------
        domain_bounds : xr.DataArray
            The array of bounds for the latitude domain.
        region_bounds : np.array
            The array of bounds for latitude regional selection.

        Returns
        -------
        xr.DataArray
            The latitude axes weights.
        """
        if region_bounds is not None:
            domain_bounds = self._scale_domain_to_region(domain_bounds, region_bounds)

        domain_bounds = np.sin(np.radians(domain_bounds))
        weights = self._calculate_weights(domain_bounds)

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

    def _align_longitude_to_360_axis(
        self, domain_bounds: xr.DataArray
    ) -> Tuple[xr.DataArray, np.array]:
        """Handles a prime meridian cell to align longitude axes to (0, 360).

        This method ensures the domain bounds are within 0 to 360 by handling
        the grid cell that encompasses the prime meridian (e.g., [359, 1]). In
        this case, calculating longitudinal weights is complicated because the
        weights are determined by the difference of the bounds.

        If this situation exists, the method will split this grid cell into
        two parts (one east and west of the prime meridian). The original
        grid cell will have domain bounds extending east of the prime meridian
        and an extra set of bounds will be concatenated to ``domain_bounds``
        corresponding to the domain bounds west of the prime meridian. For
        instance, a grid cell spanning -1 to 1, will be broken into a cell
        from 0 to 1 and 359 to 360 (or -1 to 0). The index of the original
        prime meridian grid cell is returned as ``p_meridian_index`` along with
        the updated ``domain_bounds``.

        If no domain grid bounds span across the prime meridian, the original
        ``domain_bounds`` are returned and `p_meridian_index` returns None.

        Parameters
        ----------
        domain_bounds : xr.DataArray
            The domain's bounds. The bounds should have values between 0 and
            360.

        Returns
        -------
        Tuple[xr.DataArray, Optional[np.array]]
           A tuple, with the first element being the domain bounds DataArray,
           and the second being an np.array with a single element representing
           the index of the prime meridian grid cell.

        Notes
        -----
        This method returns ``domain_bounds`` that are intended for calculating
        spatial weights only.
        """
        if (domain_bounds.values.min() < 0) | (domain_bounds.values.max() > 360):
            raise ValueError(
                "Longitude bounds aren't inclusively between 0 and 360. "
                "Use `_swap_lon_axes()` before calling `_align_longitude_to_360_axis()`."
            )

        p_meridian_index = np.where(domain_bounds[:, 1] - domain_bounds[:, 0] < 0)[0]
        if len(p_meridian_index) == 0:
            p_meridian_index = None
        elif len(p_meridian_index) > 1:
            raise ValueError("More than one grid cell spans prime meridian.")
        elif len(p_meridian_index) == 1:
            # Example array: [[359, 1], [1, 90], [90, 180], [180, 359]]
            # Reorient bound to span across zero (i.e., [359, 1] -> [-1, 1]).
            # Result: [[-1, 1], [1, 90], [90, 180], [180, 359]]
            domain_bounds[p_meridian_index, 0] = (
                domain_bounds[p_meridian_index, 0] - 360.0
            )
            # Extend the array to nlon+1 by concatenating the grid cell that
            # spans the prime meridian to the end.
            # Result: [[-1, 1], [1, 90], [90, 180], [180, 359], [-1, 1]]
            domain_bounds = xr.concat(
                (domain_bounds, domain_bounds[p_meridian_index, :]), dim="lon"
            )
            # Add an equivalent bound that spans 360
            # (i.e., [-1, 1] -> [359, 361]) to the end of the array.
            # Result: [[-1, 1], [1, 90], [90, 180], [180, 359], [359, 361]]
            repeat_bound = domain_bounds[p_meridian_index, :][0] + 360.0
            domain_bounds[-1, :] = repeat_bound

            # Update the lower-most min and upper-most max bounds to [0, 360].
            # Result: [[0, 1], [1, 90], [90, 180], [180, 359], [359, 360]]
            domain_bounds[p_meridian_index, 0], domain_bounds[-1, 1] = (0.0, 360.0)

        return domain_bounds, p_meridian_index

    def _swap_lon_axes(
        self, lon: Union[xr.DataArray, np.ndarray], to: Literal[180, 360]
    ) -> Union[xr.DataArray, np.ndarray]:
        """Swap the longitude axes orientation.

        Parameters
        ----------
        lon : Union[xr.DataArray, np.ndarray]
             Longitude values to convert.
        to : Literal[180, 360]
            Axis orientation to convert to, either 180 (-180 to 180) or 360
            (0 to 360).

        Returns
        -------
        Union[xr.DataArray, np.ndarray]
            Converted longitude values.

        Notes
        -----
        This does not reorder the values in any way; it only converts the values
        in-place between longitude conventions (-180 to 180) or (0 to 360).
        """
        lon_swap = lon.copy()

        # If chunking, must convert convert the xarray data structure from lazy
        # Dask arrays into eager, in-memory NumPy arrays before performing
        # manipulations on the data. Otherwise, it raises `NotImplementedError
        # xarray can't set arrays with multiple array indices to dask yet`.
        if type(lon_swap.data) == Array:
            lon_swap.load()

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

    def _combine_weights(
        self, axis: List[SupportedAxes], axis_weights: AxisWeights
    ) -> xr.DataArray:
        """Generically rescales axis weights for a given region.

        This method creates an n-dimensional weighting array by performing
        matrix multiplication for a list of specified axes using a dictionary of
        axis weights.

        Parameters
        ----------
        axis : List[SupportedAxes]
            List of axes that should be weighted.
        axis_weights : AxisWeights
            Dictionary of axis weights, where key is axes and value is the
            corresponding DataArray of weights.

        Returns
        -------
        xr.DataArray
            A DataArray containing the region weights to use during averaging.
            ``weights`` are 1-D and correspond to the specified axes (``axis``)
            in the region.
        """
        weights = {
            axes: weights for axes, weights in axis_weights.items() if axes in axis
        }
        region_weights = reduce((lambda x, y: x * y), weights.values())
        return region_weights

    def _validate_weights(
        self, data_var: xr.DataArray, axis: List[SupportedAxes], weights: xr.DataArray
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
        axis : List[SupportedAxes]
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
        self, data_var: xr.DataArray, axis: List[SupportedAxes], weights: xr.DataArray
    ):
        """Perform a weighted average of a data variable.

        This method assumes all specified axes in ``axis`` exists in the data
        variable. Validation for this criteria is performed in
        ``_validate_weights()``.

        Operations include:

        - Masked (missing) data receives zero weight.
        - Perform weighted average over user-specified axes/axis.

        Parameters
        ----------
        data_var : xr.DataArray
            Data variable inside a Dataset.
        axis : List[SupportedAxes]
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
            weighted_mean = data_var.weighted(weights).mean(tuple(axis))
            return weighted_mean
