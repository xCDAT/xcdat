"""Module containing geospatial averaging functions."""

from collections.abc import Callable, Hashable
from functools import reduce
from typing import Any, Literal, TypedDict, get_args

import cf_xarray  # noqa: F401
import numpy as np
import xarray as xr

from xcdat.axis import (
    _align_lon_bounds_to_360,
    _get_prime_meridian_index,
    get_dim_coords,
    get_dim_keys,
)
from xcdat.dataset import _get_data_var
from xcdat.mask import (
    generate_and_apply_land_sea_mask,
    generate_land_sea_mask,
)
from xcdat.utils import (
    _get_masked_weights,
    _if_multidim_dask_array_then_load,
    _validate_min_weight,
)

#: Type alias for a dictionary of axis keys mapped to their bounds.
AxisWeights = dict[Hashable, xr.DataArray]
#: Type alias for supported spatial axis keys.
SpatialAxis = Literal["X", "Y"]
SPATIAL_AXES: tuple[SpatialAxis, ...] = get_args(SpatialAxis)
#: Type alias for a tuple of floats/ints for the regional selection bounds.
RegionAxisBounds = tuple[float, float]


@xr.register_dataset_accessor("spatial")
class SpatialAccessor:
    """
    An accessor class that provides spatial attributes and methods on xarray
    Datasets through the ``.spatial`` attribute.

    Examples
    --------

    Import SpatialAccessor class:

    >>> import xcdat  # or from xcdat import spatial

    Use SpatialAccessor class:

    >>> ds = xcdat.open_dataset("/path/to/file")
    >>>
    >>> ds.spatial.<attribute>
    >>> ds.spatial.<method>
    >>> ds.spatial.<property>

    Parameters
    ----------
    dataset : xr.Dataset
        A Dataset object.
    """

    def __init__(self, dataset: xr.Dataset):
        self._dataset: xr.Dataset = dataset

    def mask_land(
        self,
        data_var: str,
        method: str = "regionmask",
        criteria: float | None = None,
        mask: xr.DataArray | None = None,
        output_mask: bool | str = False,
        **options: Any,
    ):
        """
        Masks a data variable by land.

        Parameters
        ----------
        data_var : str
            The key of the data variable to mask.
        method : str, optional
            The masking method, by default "regionmask".
            Supported methods: "regionmask", "pcmdi".
        criteria : float | None, optional
            The value to use as the criteria for cell values that are considered
            land, by default 0.2.
        mask : xr.DataArray | None, optional
            A custom mask to apply, by default None. If None, a mask is
            generated using the specified ``method``.
        output_mask : bool | str, optional
            If True, returns the mask as a DataArray along with the masked
            dataset. If a string, the name of the mask variable to add to the
            dataset. By default False.
        **options : Any
            These options are passed directly to the ``method``. See specific
            method documentation for available options:
            :func:`xcdat.mask.pcmdi_land_sea_mask` for PCMDI options.

        Returns
        -------
        xr.Dataset
            The dataset with the data variable masked by land.

        Examples
        --------

        Mask a data variable by land using the default method (regionmask):

        >>> ds_masked = ds.spatial.mask_land("tas")

        Mask a data variable by land using the PCMDI method with custom criteria:

        >>> ds_masked = ds.spatial.mask_land("tas", method="pcmdi", criteria=0.3)

        Mask a data variable by land using a custom mask and output the mask:

        >>> custom_mask = xr.DataArray(...)  # Define your custom mask here
        >>> ds_masked = ds.spatial.mask_land("tas", mask=custom_mask, output_mask=True)

        Mask a data variable by land and add the mask to the dataset with a custom name:

        >>> ds_masked = ds.spatial.mask_land("tas", output_mask="land_mask")
        """
        return generate_and_apply_land_sea_mask(
            self._dataset,
            data_var,
            method,
            keep="sea",
            criteria=criteria,
            mask=mask,
            output_mask=output_mask,
            **options,
        )

    def mask_sea(
        self,
        data_var: str,
        method: str = "regionmask",
        criteria: float | None = None,
        mask: xr.DataArray | None = None,
        output_mask: bool = False,
        **options: Any,
    ):
        """
        Masks a data variable by sea.

        Parameters
        ----------
        data_var : str
            The key of the data variable to mask.
        method : str, optional
            The masking method, by default "regionmask".
            Supported methods: "regionmask", "pcmdi".
        criteria : float | None, optional
            The value to use as the criteria for cell values that are considered
            sea, by default 0.8.
        mask : xr.DataArray | None, optional
            A custom mask to apply, by default None. If None, a mask is
            generated using the specified ``method``.
        output_mask : bool | str, optional
            If True, returns the mask as a DataArray along with the masked
            dataset. If a string, the name of the mask variable to add to the
            dataset. By default False.
        **options : Any
            These options are passed directly to the ``method``. See specific
            method documentation for available options:
            :func:`xcdat.mask.pcmdi_land_sea_mask` for PCMDI options

        Returns
        -------
        xr.Dataset
            The dataset with the data variable masked by sea.

        Examples
        --------

        Mask a data variable by sea using the default method (regionmask):

        >>> ds_masked = ds.spatial.mask_sea("tas")

        Mask a data variable by sea using the PCMDI method with custom criteria:

        >>> ds_masked = ds.spatial.mask_sea("tas", method="pcmdi", criteria=0.7)

        Mask a data variable by sea using a custom mask and output the mask:

        >>> custom_mask = xr.DataArray(...)  # Define your custom mask here
        >>> ds_masked = ds.spatial.mask_sea("tas", mask=custom_mask, output_mask=True)

        Mask a data variable by sea and add the mask to the dataset with a custom name:

        >>> ds_masked = ds.spatial.mask_sea("tas", output_mask="sea_mask")
        """
        return generate_and_apply_land_sea_mask(
            self._dataset,
            data_var,
            method,
            keep="land",
            criteria=criteria,
            mask=mask,
            output_mask=output_mask,
            **options,
        )

    def generate_land_sea_mask(
        self,
        data_var: str | None = None,
        method: str = "regionmask",
        **options: Any,
    ) -> xr.DataArray:
        """
        Generate a land-sea mask.

        Parameters
        ----------
        data_var : str, optional
            Name of the variable whose lat/lon coordinates will be used to
            generate the land/sea mask. If omitted then a `mask` variable will
            be generated using the lat/lon coordinates in the dataset.
        method : str, optional
            The method to use for generating the mask, by default "regionmask".
            Supported methods: "regionmask", "pcmdi".
        **options : Any
            These options are passed directly to the ``method``. See specific
            method documentation for available options:
            :func:`xcdat.mask.pcmdi_land_sea_mask` for PCMDI options

        Returns
        -------
        xr.DataArray
            The land/sea mask.

        Examples
        --------

        Generate a mask using the default method (regionmask):

        >>> mask = ds.spatial.generate_land_sea_mask("tas")

        Generate a mask using the "pcmdi" method:

        >>> mask = ds.spatial.generate_land_sea_mask("tas", method="pcmdi")

        Generate a mask using the "pcmdi" method, with customization:

        >>> mask = ds.spatial.generate_land_sea_mask("tas", method="pcmdi", source=high_res_ds, source_data_var="highres")

        Generating a mask from a new grid:

        >>> grid = xc.create_uniform_grid(-90, 90, 1, 0, 359, 1)

        >>> mask = grid.spatial.generate_land_sea_mask()
        """
        if data_var is None:
            try:
                da_shape = list(self._dataset.cf[x].shape[0] for x in ("X", "Y"))

                da_dims = list(self._dataset.cf[x].name for x in ("X", "Y"))

                da_coords = {x: self._dataset[x].copy() for x in da_dims}
            except KeyError:
                raise KeyError(
                    "Dataset is missing a required coordinate, ensure a lat and lon coordinate exist"
                ) from None

            da = xr.DataArray(np.ones(da_shape), dims=da_dims, coords=da_coords)
        else:
            da = self._dataset[data_var]

        return generate_land_sea_mask(da, method, **options)

    def average(
        self,
        data_var: str,
        axis: list[SpatialAxis] | tuple[SpatialAxis, ...] = ("X", "Y"),
        weights: Literal["generate"] | xr.DataArray = "generate",
        keep_weights: bool = False,
        lat_bounds: RegionAxisBounds | None = None,
        lon_bounds: RegionAxisBounds | None = None,
        skipna: bool | None = None,
        min_weight: float | None = None,
    ) -> xr.Dataset:
        """
        Calculates the spatial average for a rectilinear grid over an optionally
        specified regional domain.

        Operations include:

        - If a regional boundary is specified, check to ensure it is within the
          data variable's domain boundary.
        - If axis weights are not provided, get axis weights for standard axis
          domains specified in ``axis``.
        - Adjust weights to conform to the specified regional boundary.
        - Compute spatial weighted average.

        This method requires that the dataset's coordinates have the 'axis'
        attribute set to the keys in ``axis``. For example, the latitude
        coordinates should have its 'axis' attribute set to 'Y' (which is also
        CF-compliant). This 'axis' attribute is used to retrieve the related
        coordinates via `cf_xarray`. Refer to this method's examples for more
        information.

        Parameters
        ----------
        data_var: str
            The name of the data variable inside the dataset to spatially
            average.
        axis : list[SpatialAxis]
            List of axis dimensions to average over, by default ("X", "Y").
            Valid axis keys include "X" and "Y".
        weights : {"generate", xr.DataArray}, optional
            If "generate", then weights are generated. Otherwise, pass a
            DataArray containing the regional weights used for weighted
            averaging. ``weights`` must include the same spatial axis dimensions
            and have the same dimensional sizes as the data variable, by default
            "generate".
        keep_weights : bool, optional
            If calculating averages using weights, keep the weights in the
            final dataset output, by default False.
        lat_bounds : RegionAxisBounds | None, optional
            A tuple of floats/ints for the regional latitude lower and upper
            boundaries. This arg is used when calculating axis weights, but is
            ignored if ``weights`` are supplied. The lower bound cannot be
            larger than the upper bound, by default None.
        lon_bounds : RegionAxisBounds | None, optional
            A tuple of floats/ints for the regional longitude lower and upper
            boundaries. This arg is used when calculating axis weights, but is
            ignored if ``weights`` are supplied. The lower bound can be larger
            than the upper bound (e.g., across the prime meridian, dateline), by
            default None.
        skipna : bool | None, optional
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or ``skipna=True`` has not been
            implemented (object, datetime64 or timedelta64).
        min_weight : float | None, optional
            Minimum threshold of data coverage (weight) required to compute
            a spatial average for a grouping window. Must be between 0 and 1.
            Useful for ensuring accurate averages in regions with missing data,
            by default None (equivalent to 0.0).

            The value must be between 0 and 1, where:
                - 0/``None`` means no minimum threshold (all data is considered,
                  even if coverage is minimal).
                - 1 means full data coverage is required (no missing data is
                  allowed).

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

        Check the 'axis' attribute is set on the required coordinates:

        >>> ds.lat.attrs["axis"]
        >>> Y
        >>>
        >>> ds.lon.attrs["axis"]
        >>> X

        Set the 'axis' attribute for the required coordinates if it isn't:

        >>> ds.lat.attrs["axis"] = "Y"
        >>> ds.lon.attrs["axis"] = "X"

        Call spatial averaging method:

        >>> ds.spatial.average(...)

        Get global average time series:

        >>> ts_global = ds.spatial.average("tas", axis=["X", "Y"])["tas"]

        Get time series in Nino 3.4 domain:

        >>> ts_n34 = ds.spatial.average("ts", axis=["X", "Y"],
        >>>     lat_bounds=(-5, 5),
        >>>     lon_bounds=(-170, -120))["ts"]

        Get zonal mean time series:

        >>> ts_zonal = ds.spatial.average("tas", axis=["X"])["tas"]

        Using custom weights for averaging:

        >>> # The shape of the weights must align with the data var.
        >>> self.weights = xr.DataArray(
        >>>     data=np.ones((4, 4)),
        >>>     coords={"lat": self.ds.lat, "lon": self.ds.lon},
        >>>     dims=["lat", "lon"],
        >>> )
        >>>
        >>> ts_global = ds.spatial.average("tas", axis=["X", "Y"],
        >>>     weights=weights)["tas"]
        """
        ds = self._dataset.copy()
        dv = _get_data_var(ds, data_var)

        self._validate_axis_arg(axis)
        min_weight = _validate_min_weight(min_weight)

        if isinstance(weights, str) and weights == "generate":
            if lat_bounds is not None:
                self._validate_region_bounds("Y", lat_bounds)
            if lon_bounds is not None:
                self._validate_region_bounds("X", lon_bounds)
            self._weights = self.get_weights(axis, lat_bounds, lon_bounds, data_var)
        elif isinstance(weights, xr.DataArray):
            self._weights = weights

        self._validate_weights(dv, axis)
        ds[dv.name] = self._averager(dv, axis, skipna=skipna, min_weight=min_weight)

        if keep_weights:
            ds[self._weights.name] = self._weights

        return ds

    def get_weights(
        self,
        axis: list[SpatialAxis] | tuple[SpatialAxis, ...],
        lat_bounds: RegionAxisBounds | None = None,
        lon_bounds: RegionAxisBounds | None = None,
        data_var: str | None = None,
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
        axis : list[SpatialAxis] | tuple[SpatialAxis, ...]
            List of axis dimensions to average over.
        lat_bounds : RegionAxisBounds | None
            Tuple of latitude boundaries for regional selection, by default
            None.
        lon_bounds : RegionAxisBounds | None
            Tuple of longitude boundaries for regional selection, by default
            None.
        data_var: str | None
            The key of the data variable, by default None. Pass this argument
            when the dataset has more than one bounds per axis (e.g., "lon"
            and "zlon_bnds" for the "X" axis), or you want weights for a
            specific data variable.

        Returns
        -------
        xr.DataArray
           A DataArray containing the region weights to use during averaging.
           ``weights`` are 1-D and correspond to the specified axes (``axis``)
           in the region.

        Notes
        -----
        This method was developed for rectilinear grids only. ``get_weights()``
        recognizes and operate on latitude and longitude, but could be extended
        to work with other standard geophysical dimensions (e.g., time, depth,
        and pressure).
        """
        Bounds = TypedDict(
            "Bounds", {"weights_method": Callable, "region": np.ndarray | None}
        )

        axis_bounds: dict[SpatialAxis, Bounds] = {
            "X": {
                "weights_method": self._get_longitude_weights,
                "region": np.array(lon_bounds, dtype="float")
                if lon_bounds is not None
                else None,
            },
            "Y": {
                "weights_method": self._get_latitude_weights,
                "region": np.array(lat_bounds, dtype="float")
                if lat_bounds is not None
                else None,
            },
        }

        axis_weights: AxisWeights = {}
        for key in axis:
            d_bounds = self._dataset.bounds.get_bounds(axis=key, var_key=data_var)

            if isinstance(d_bounds, xr.Dataset):
                raise TypeError(
                    "Generating area weights requires a single bounds per "
                    f"axis, but the dataset has multiple bounds for the '{key}' axis "
                    f"{list(d_bounds.data_vars)}. Pass a `data_var` key "
                    "to reference a specific data variable's axis bounds."
                )

            r_bounds = axis_bounds[key]["region"]

            weights = axis_bounds[key]["weights_method"](d_bounds, r_bounds)
            weights.attrs = d_bounds.attrs
            axis_weights[key] = weights

        weights = self._combine_weights(axis_weights)

        return weights

    def _validate_axis_arg(self, axis: list[SpatialAxis] | tuple[SpatialAxis, ...]):
        """
        Validates that the ``axis`` dimension(s) exists in the dataset.

        Parameters
        ----------
        axis : list[SpatialAxis] | tuple[SpatialAxis, ...]
            List of axis dimensions to average over.

        Raises
        ------
        ValueError
            If a key in ``axis`` is not a supported value.
        KeyError
            If the dataset does not have coordinates for the ``axis`` dimension,
            or the `axis` attribute is not set for those coordinates.
        """
        for key in axis:
            if key not in SPATIAL_AXES:
                raise ValueError(
                    "Incorrect `axis` argument value. Supported values include: "
                    f"{', '.join(SPATIAL_AXES)}."
                )

            # Check the axis coordinate variable exists in the Dataset.
            get_dim_coords(self._dataset, key)

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
            If the ``axis`` is "Y" and the ``bounds`` lower value is larger
            than the upper value.
        """
        if not isinstance(bounds, tuple):
            raise TypeError(
                f"The {axis} regional bounds is not a tuple representing the lower and "
                "upper bounds, (lower, upper)."
            )

        if len(bounds) != 2:
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

        # For the "Y" axis (latitude), require that the upper bound be larger
        # than the lower bound. Note that this does not apply to the "X" axis
        # (longitude) since it is circular.
        if axis == "Y" and lower >= upper:
            raise ValueError(
                "The regional latitude lower bound is greater than the upper. "
                "Pass a tuple with the format (lower, upper)."
            )

    def _get_longitude_weights(
        self, domain_bounds: xr.DataArray, region_bounds: np.ndarray | None
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
            The array of bounds for the longitude domain.
        region_bounds : np.ndarray | None
            The array of bounds for longitude regional selection.

        Returns
        -------
        xr.DataArray
            The longitude axis weights.

        Raises
        ------
        ValueError
            If the there are multiple instances in which the
            domain_bounds[:, 0] > domain_bounds[:, 1]
        """
        p_meridian_index: np.ndarray | None = None
        d_bounds = domain_bounds.copy()

        pm_cells = np.where(domain_bounds[:, 1] - domain_bounds[:, 0] < 0)[0]
        if len(pm_cells) > 1:
            raise ValueError(
                "More than one longitude bound is out of order. Only one bound "
                "value spanning the prime meridian is permitted in data on "
                "a rectilinear grid."
            )
        d_bounds: xr.DataArray = self._swap_lon_axis(d_bounds, to=360)  # type: ignore
        p_meridian_index = _get_prime_meridian_index(d_bounds)
        if p_meridian_index is not None:
            d_bounds = _align_lon_bounds_to_360(d_bounds, p_meridian_index)

        if region_bounds is not None:
            r_bounds: np.ndarray = self._swap_lon_axis(region_bounds, to=360)  # type:ignore

            is_region_circular = r_bounds[1] - r_bounds[0] == 0
            if is_region_circular:
                r_bounds = np.array([0.0, 360.0])

            d_bounds = self._scale_domain_to_region(d_bounds, r_bounds)

        weights = self._calculate_weights(d_bounds)
        if p_meridian_index is not None:
            weights[p_meridian_index] = weights[p_meridian_index] + weights[-1]
            weights = weights[0:-1]

        return weights

    def _get_latitude_weights(
        self, domain_bounds: xr.DataArray, region_bounds: np.ndarray | None
    ) -> xr.DataArray:
        """Gets weights for the latitude axis.

        This method scales the domain to a region (if selected). It also scales
        the area between two lines of latitude as the difference of the sine of
        latitude bounds.

        Parameters
        ----------
        domain_bounds : xr.DataArray
            The array of bounds for the latitude domain.
        region_bounds : np.ndarray | None
            The array of bounds for latitude regional selection.

        Returns
        -------
        xr.DataArray
            The latitude axis weights.
        """
        if region_bounds is not None:
            domain_bounds = self._scale_domain_to_region(domain_bounds, region_bounds)

        d_bounds = np.sin(np.radians(domain_bounds))
        weights = self._calculate_weights(d_bounds)
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
        self, lon: xr.DataArray | np.ndarray, to: Literal[180, 360]
    ) -> xr.DataArray | np.ndarray:
        """Swap the longitude axis orientation.

        Parameters
        ----------
        lon : xr.DataArray | np.ndarray
             Longitude values to convert.
        to : Literal[180, 360]
            Axis orientation to convert to, either 180 [-180, 180) or 360
            [0, 360).

        Returns
        -------
        xr.DataArray | np.ndarray
            Converted longitude values.

        Notes
        -----
        This does not reorder the values in any way; it only converts the values
        in-place between longitude conventions [-180, 180) or [0, 360).
        """
        lon_swap = lon.copy()

        if isinstance(lon_swap, xr.DataArray):
            _if_multidim_dask_array_then_load(lon_swap)

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

        _if_multidim_dask_array_then_load(d_bounds)

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

            # Grid cell straddling lower boundary.
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

            # Grid cell straddling upper boundary.
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

        coord_keys = sorted(region_weights.dims)  # type: ignore
        region_weights.name = "_".join(coord_keys) + "_wts"  # type: ignore

        return region_weights

    def _validate_weights(
        self, data_var: xr.DataArray, axis: list[SpatialAxis] | tuple[SpatialAxis, ...]
    ):
        """Validates the ``weights`` arg based on a set of criteria.

        This methods checks for the dimensional alignment between the
        ``weights`` and ``data_var``. It assumes that ``data_var`` has the same
        keys that are specified  in ``axis``, which has already been validated
        using ``self._validate_axis()`` in ``self.average()``.

        Parameters
        ----------
        data_var : xr.DataArray
            The data variable used for validation with user supplied weights.
        axis : list[SpatialAxis] | tuple[SpatialAxis, ...]
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

        # Check the weights includes the same axis as the data variable.
        for key in axis:
            dim_name = get_dim_keys(data_var, key)
            if dim_name not in self._weights.dims:
                raise KeyError(
                    f"The weights DataArray does not include an {key} axis, or the "
                    "dimension names are not the same."
                )

        # Check the weight dim sizes equal data var dim sizes.
        dim_sizes = {key: data_var.sizes[key] for key in self._weights.sizes.keys()}
        for dim, size in self._weights.sizes.items():
            if size != dim_sizes[dim]:
                raise ValueError(
                    f"The axis dimension sizes between supplied `weights` {dict(self._weights.sizes)} "
                    f"and the data variable {dim_sizes} are misaligned."
                )

    def _averager(
        self,
        data_var: xr.DataArray,
        axis: list[SpatialAxis] | tuple[SpatialAxis, ...],
        skipna: bool | None = None,
        min_weight: float = 0.0,
    ) -> xr.DataArray:
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
        axis : list[SpatialAxis] | tuple[SpatialAxis, ...]
            List of axis dimensions to average over.
        skipna : bool | None, optional
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or ``skipna=True`` has not been
            implemented (object, datetime64 or timedelta64).
        min_weight : float, optional
            Minimum threshold of data coverage (weight) required to compute
            a spatial average for a grouping window. Must be between 0 and 1.
            Useful for ensuring accurate averages in regions with missing data,
            by default 0.0.

            The value must be between 0 and 1, where:
                - 0 means no minimum threshold (all data is considered, even if
                  coverage is minimal).
                - 1 means full data coverage is required (no missing data is
                  allowed).

        Returns
        -------
        xr.DataArray
            Variable that has been reduced via a weighted average.

        Notes
        -----
        ``weights`` must be a DataArray and cannot contain missing values.
        Missing values are replaced with 0 using ``weights.fillna(0)``.
        """
        dv = data_var.copy()
        weights = self._weights.fillna(0)

        dim: list[str] = []
        for key in axis:
            dim.append(get_dim_keys(dv, key))  # type: ignore

        with xr.set_options(keep_attrs=True):
            dv_mean = dv.cf.weighted(weights).mean(dim=dim, skipna=skipna)

        if min_weight > 0.0:
            dv_mean = self._mask_var_with_weight_threshold(
                dv, dv_mean, dim, weights, min_weight
            )

        return dv_mean

    def _mask_var_with_weight_threshold(
        self,
        dv: xr.DataArray,
        dv_mean: xr.DataArray,
        dim: list[str],
        weights: xr.DataArray,
        min_weight: float,
    ) -> xr.DataArray:
        """Mask values that do not meet the minimum weight threshold with np.nan.

        This function is useful for cases where the weighting of data might be
        skewed based on the availability of data. For example, if a portion of
        cells in a region has significantly more missing data than other other
        regions, it can result in inaccurate spatial average calculations.
        Masking values that do not meet the minimum weight threshold ensures
        more accurate calculations.

        Parameters
        ----------
        dv : xr.DataArray
            The weighted variable used for getting masked weights.
        dv_mean : xr.DataArray
            The average of the weighted variable.
        dim: list[str]:
            List of axis dimensions to average over.
        weights : xr.DataArray
            A DataArray containing either the regional weights used for weighted
            averaging. ``weights`` must include the same axis dimensions and
            dimensional sizes as the data variable.
        min_weight : float, optional
            Minimum threshold of data coverage (weight) required to compute
            a spatial average for a grouping window. Must be between 0 and 1.
            Useful for ensuring accurate averages in regions with missing data,
            by default None (equivalent to 0.0).

            The value must be between 0 and 1, where:
                - 0/``None`` means no minimum threshold (all data is considered,
                  even if coverage is minimal).
                - 1 means full data coverage is required (no missing data is
                  allowed).

        Returns
        -------
        xr.DataArray
            The average of the weighted variable with the minimum weight
            threshold applied.
        """
        # Sum all weights, including zero for missing values.
        weight_sum_all = weights.sum(dim=dim)

        masked_weights = _get_masked_weights(dv, weights)
        weight_sum_masked = masked_weights.sum(dim=dim)

        # Get fraction of the available weight.
        frac = weight_sum_masked / weight_sum_all

        # Nan out values that don't meet specified weight threshold.
        dv_new = xr.where(frac >= min_weight, dv_mean, np.nan, keep_attrs=True)
        dv_new.name = dv_mean.name

        return dv_new
