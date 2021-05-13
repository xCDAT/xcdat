from typing import Optional

import numpy as np
import xarray as xr
from typing_extensions import Literal

from xcdat.xcdat import logger


@xr.register_dataset_accessor("axis")
class AxisAccessor:
    """A class used to represent an AxisAccessor (xarray dataset accessor).

    :param xarray_obj: The Dataset object to be extended
    :type xarray_obj: xr.Dataset

    Examples

    >>> import xarray as xr
    >>> from xcdat import axis

    Get latitude and longitude bounds (cached properties):

    >>> ds = xr.open_dataset("file_path")
    >>> lat_bnds = ds.axis.lat_bnds
    >>> lon_bnds = ds.axis.lon_bnds

    Get axis bounds explicitly:

    >>> ds = xr.open_dataset("file_path")
    >>> lat_bnds = ds.axis.get_bounds('lat')
    >>> lon_bnds = ds.axis.get_bounds('lon')

    Get axis bounds explicitly and do not generate bounds if it doesn't exist:

    >>> ds = xr.open_dataset("file_path")
    >>> lat_bnds = ds.axis.get_bounds('lat', allow_generating=False)
    """

    # Type annotation definitions
    Axis = Literal["lat", "lon"]

    def __init__(self, xarray_obj: xr.Dataset):
        self._dataset: xr.Dataset = xarray_obj

    @property
    def lat_bnds(self) -> Optional[xr.DataArray]:
        """Get latitude boundaries and caches value.

        :return: Latitude boundaries
        :rtype: Optional[xr.DataArray]
        """
        return self.get_bounds("lat")

    @property
    def lon_bnds(self) -> Optional[xr.DataArray]:
        """Get longitude boundaries and caches value.

        :return: Longitude boundaries
        :rtype: Optional[xr.DataArray]
        """
        return self.get_bounds("lon")

    def get_bounds(
        self, axis: Axis, allow_generating: bool = True
    ) -> Optional[xr.DataArray]:
        """Get bounds for an axis.

        It will return the existing axis bounds in the Dataset if a match is found,
        otherwise automatically generate the axis bounds.

        :param axis: "lat" or "lon" axis
        :type axis: Axis
        :param allow_generating: If True, generate the bounds if they don't exist.
            If False, return only existing bounds or throw error if it doesn't exist (useful
            for explicit method behavior or debugging), defaults to True
        :type allow_generating: bool, optional
        :raises ValueError: [description]
        :return: Axis bounds or None
        :rtype: Optional[xr.DataArray]
        """
        if axis not in ["lat", "lon"]:
            raise ValueError("Axis must be 'lat' or 'lon")

        matching_bounds = self._dataset.data_vars.get(f"{axis}_bnds")
        if matching_bounds is not None:
            logger.info(f"{axis} bounds were found in the dataset.")
            return matching_bounds

        if matching_bounds is None and allow_generating:
            logger.warning(
                f"{axis} bounds were not found in the dataset, generating bounds."
            )
            return self._gen_bounds(axis)
        elif matching_bounds is None and not allow_generating:
            raise ValueError(
                f"{axis} bounds were not found in the dataset, bounds must be generated"
            )

        # Need a default return, but will never be reached based on conditionals.
        return None  # pragma: no cover

    def _gen_bounds(self, axis: Axis, width: float = 1) -> xr.DataArray:
        """Generates the bounds for an axis and adds it to the Dataset's data variables.

        :param axis: "lat" or "lon" axis
        :type axis: Axis
        :param width: Width of the bounds when axis length is 1, defaults to 1
        :type width: float, optional
        :raises TypeError: If Axis coordinates has no "units" attribute
        :raises ValueError: If Axis coordinates' "units" attribute does not contain
            "degree" substring
        :return: The generated axis bounds DataArray
        :rtype: xr.DataArray
        """
        axis_coords: xr.DataArray = self._get_coords(axis)

        # The units for axis coordinates must be in degree(s), or the bounds
        # won't be generated correctly.
        units_attr = axis_coords.attrs.get("units")
        if units_attr is None:
            raise TypeError(f"Dataset {axis} coordinates has no units attribute.")
        if "degree" not in units_attr:
            raise ValueError(
                f"Dataset {axis} units attribute ('{units_attr}') does not contain 'degree'."
            )

        # Bounds are generated and adjusted to ensure circularity or within an
        # allowed range.
        bounds = self._gen_base_bounds(axis_coords.data, width)
        if axis == "lat":
            bounds = self._adjust_lat_bounds(bounds)
        if axis == "lon" and len(bounds.shape) == 2:
            bounds = self._adjust_lon_bounds(bounds)

        var_name = f"{axis}_bnds"
        self._dataset[var_name] = xr.DataArray(
            name=var_name,
            data=bounds,
            coords={axis: axis_coords.data},
            dims=[axis, "bnds"],
            attrs={"units": axis_coords.units, "is_generated": True},
        )
        return self._dataset[var_name]

    def _get_coords(self, axis: Axis) -> xr.DataArray:
        """Get the coordinates for an axis.

        :param axis: "lat" or "lon" axis
        :type axis: Axis
        :raises KeyError: If an incorrect `axis` argument is passed
        :return: Matching coordinates
        :rtype: xr.DataArray
        """
        matching_coords = self._dataset.coords.get(axis)
        if matching_coords is not None:
            return matching_coords

        raise TypeError(f"No matching coordinates for axis: {axis}")

    def _gen_base_bounds(
        self,
        data: np.ndarray,
        width: float,
    ) -> np.ndarray:
        """Generates the base 2D bounds for an axis.

        :param data: Axis coordinate data
        :type data: np.ndarray
        :param width: Width of the bounds
        :type width: float
        :return: Base 2D bounds for an axis coordinates variable
        :rtype: np.ndarray
        """
        if len(data) > 1:
            left = np.array([1.5 * data[0] - 0.5 * data[1]])
            center = (data[0:-1] + data[1:]) / 2.0
            right = np.array([1.5 * data[-1] - 0.5 * data[-2]])
            bounds = np.concatenate((left, center, right))
        else:
            delta = width / 2.0
            bounds = np.array([data[0] - delta, data[0] + delta])

        bounds_2d = np.array(list(zip(*(bounds[i:] for i in range(2)))))
        return bounds_2d

    def _adjust_lat_bounds(self, bounds: np.ndarray) -> np.ndarray:
        """If necessary, adjusts latitude boundaries to avoid floating point errors.

        The endpoints are also capped at (-90, 90).

        :param bounds: Base latitude boundaries
        :type bounds: np.ndarray
        :return: Adjusted latitude boundaries
        :rtype: np.ndarray
        """
        bounds[0, ...] = np.maximum(-90.0, np.minimum(90.0, bounds[0, ...]))
        bounds[-1, ...] = np.maximum(-90.0, np.minimum(90.0, bounds[-1, ...]))

        return bounds

    def _adjust_lon_bounds(self, bounds: np.ndarray) -> np.ndarray:
        """If necessary, adjusts longitude boundaries to avoid floating point errors.

        The endpoints may also be adjusted to ensure circularity.

        :param bounds: Base longitude boundaries
        :type bounds: np.ndarray
        :return: Adjusted longitude boundaries
        :rtype: np.ndarray
        """
        # Example: [[-1, -0.5], [0.5, 1]]
        # [[min_bound_left, min_bound_right][max_bound_left, max_bound_right]]
        min_bound_left = bounds[0, 0]
        min_right_right = bounds[0, 1]
        max_bound_left = bounds[-1, 0]
        max_bound_right = bounds[-1, 1]

        # Variables are used to ensure circularity
        max_degree_interval = 360.0
        max_degree_threshold = np.minimum(
            0.01, abs(min_right_right - min_bound_left) * 0.1
        )

        # Check if the bounds are within the max degree limit to determine if rounding
        # might be necessary
        bounds_diff = abs(max_bound_right - min_bound_left)
        bounds_diff_with_max_degree = abs(bounds_diff - max_degree_interval)

        # For example, for 0.001 < 0.01, the bounds are below the threshold and considered
        # near the max degree interval, so adjustment is necessary
        bounds_near_max_degree: bool = (
            bounds_diff_with_max_degree < max_degree_threshold
        )

        if bounds_near_max_degree:
            min_bound_int_limit = abs(min_right_right - min_bound_left) * 0.01
            min_left_val_near_int: bool = (
                abs(min_bound_left - np.floor(min_bound_left + 0.5))
                < min_bound_int_limit
            )

            max_bound_int_limit = abs(max_bound_right - max_bound_left) * 0.01
            max_right_val_near_int: bool = (
                abs(max_bound_right - np.floor(max_bound_right + 0.5))
                < max_bound_int_limit
            )

            # For (-180, 180), if either bound is near an integer value, round both integers
            # Otherwise it is not needed if all values are positive (0, 360)
            if (
                min_left_val_near_int or max_right_val_near_int
            ) and min_bound_left * max_bound_right < 0:
                bounds[0, 0] = np.floor(min_bound_left + 0.5)
                bounds[0, 1] = np.floor(max_bound_right + 0.5)
            else:
                if max_bound_right > min_bound_left:
                    bounds[-1, 1] = min_bound_left + max_degree_interval
                else:
                    bounds[0, 0] = max_bound_right + max_degree_interval

        return bounds
