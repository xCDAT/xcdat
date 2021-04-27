from typing import List, Literal, TypedDict

import numpy as np
import xarray as xr

from xcdat.xcdat import logger


@xr.register_dataset_accessor("axis")
class AxisAccessor:
    """A class used to represent an AxisAccessor (xarray dataset accessor)

    :param xarray_obj: The Dataset object to be extended
    :type xarray_obj: xr.Dataset

    Examples
    --------
    Get bounds:

    >>> from xcdat import axis
    >>> ds = xr.open_dataset("file_path")
    >>> # Get generated bounds
    >>> lat_bnds = ds.axis.get_bounds('lat', generate=True)
    >>> lon_bnds = ds.axis.get_bounds('lon', generate=True)
    >>> # Get existing bounds
    >>> lat_bnds = ds.axis.get_bounds('lat')
    >>> lon_bnds = ds.axis.get_bounds('lon')
    """

    # Type annotation definitions
    Axis = Literal["lat", "lon"]
    AxesMapValue = TypedDict(
        "AxesMapValue", {"coords": List[str], "bounds_vars": List[str]}
    )
    AxesMap = TypedDict("AxesMap", {"lat": AxesMapValue, "lon": AxesMapValue})

    # Mapping of generic axes names to coordinates and bounds variables in the Dataset
    axes_map: AxesMap = {
        "lat": {
            "coords": ["lat", "latitude"],
            "bounds_vars": ["lat_bnds", "latitude_bnds"],
        },
        "lon": {
            "coords": ["lon", "longitude"],
            "bounds_vars": ["lon_bnds", "longitude_bnds"],
        },
    }

    def __init__(self, xarray_obj: xr.Dataset):
        self._dataset = xarray_obj

    def get_bounds(self, axis: Axis, generate: bool = False) -> xr.DataArray:
        """Get bounds for an axis.

        It will either generate the bounds or return the existing bounds.

        :param axis: "lat" or "lon" axis
        :type axis: Axis
        :param generate: If True, generate the bounds regardless if it exists in the dataset or not. If False and the bounds already exist, return the existing bounds, defaults to False
        :type generate: bool, optional
        :return: Axis bounds
        :rtype: xr.DataArray
        """
        if generate:
            logger.info(f"Generating bounds for {axis} axis")
            return self._gen_bounds(axis)

        bounds_vars = AxisAccessor.axes_map[axis]["bounds_vars"]
        matching_bounds_var = None

        for var in bounds_vars:
            matching_bounds_var = self._dataset.data_vars.get(var)
            if matching_bounds_var is not None:
                logger.info(f"Returning existing {axis} bounds from the dataset.")
                return matching_bounds_var

        raise ValueError(
            f"{axis} bounds not found in the dataset. Pass generate=True to generate bounds."
        )

    def _gen_bounds(self, axis: Axis, width: float = 1) -> xr.DataArray:
        """Generates the bounds for an axis and adds it to the Dataset's data variables.

        :param axis: "lat" or "lon" axis
        :type axis: Axis
        :param width: Width of the bounds when axis length is 1, defaults to 1
        :type width: float, optional
        :raises TypeError: If Axis coordinates has no "units" attribute
        :raises ValueError: If Axis coordinates' "units" attribute does not contain "degree" substring
        :return: The generated axis bounds DataArray
        :rtype: xr.DataArray
        """
        axis_coords = self._extract_axis_coords(axis)
        bounds = self._gen_base_bounds(axis_coords.data, width)

        units_attr = axis_coords.attrs.get("units")
        if units_attr is None:
            raise TypeError(f"Dataset {axis} coordinates has no attribute 'units'.")
        if "degree" not in units_attr:
            raise ValueError(
                f"Dataset {axis} coordinates value for 'units' ('{units_attr}'), does not contain 'degree' substring."
            )

        # Perform adjustments based on boundary values as needed
        if axis == "lat":
            bounds = self._adjust_lat_bounds(bounds)
        if axis == "lon" and len(bounds.shape) == 2:
            bounds = self._adjust_lon_bounds(bounds)

        # Compose final DataArray to set as a variable
        axis_bnds_var = xr.DataArray(
            data=bounds,
            coords={axis: axis_coords.data},
            dims=[axis, "bnds"],
            attrs={"units": axis_coords.units, "is_generated": True},
        )
        self._dataset[f"{axis}_bnds"] = axis_bnds_var
        return axis_bnds_var

    def _extract_axis_coords(self, axis: Axis) -> xr.DataArray:
        """Extracts the axis' coordinates variable for the specified axis.

        :param axis: "lat" or "lon" axis
        :type axis: Axis
        :raises KeyError: If an incorrect `axis` argument is passed
        :return: Matching coordinates
        :rtype: xr.DataArray
        """
        axis_coord_names = AxisAccessor.axes_map[axis]["coords"]
        matching_coords = None

        for name in axis_coord_names:
            matching_coords = self._dataset.coords.get(name)
            if matching_coords is not None:
                return matching_coords

        raise TypeError(f"No matching coordinates for axis: {axis}")

    def _gen_base_bounds(
        self,
        data: np.ndarray,
        width: float,
    ) -> np.ndarray:
        """Generates an axis coordinates variable's base 2D bounds.

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

        The endpoints are also are capped at (-90, 90).

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

        # [[min_bound_left, min_bound_right][max_bound_left, max_bound_right]]
        # e.g [[-1, -0.5], [0.5, 1]]
        min_bound_left = bounds[0, 0]  # = -1
        min_right_right = bounds[0, 1]  # = -0.5
        max_bound_left = bounds[-1, 0]  # = 0/5
        max_bound_right = bounds[-1, 1]  # = 1

        # Variables are used to ensure circularity
        max_degree_interval = 360.0
        max_degree_threshold = np.minimum(
            0.01, abs(min_right_right - min_bound_left) * 0.1
        )

        # Check if the bounds are within the max degree limit to determine if rounding
        # might be necessary
        bounds_diff = abs(max_bound_right - min_bound_left)
        bounds_diff_with_max_degree = abs(bounds_diff - max_degree_interval)

        # For example, for 0.001 < 0.01, the bounds are above the threshold and considered
        # near the max degree interval so adjustment is necessary
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
                    bounds[-1, 1] = round(min_bound_left + max_degree_interval, 4)
                else:
                    bounds[0, 0] = round(max_bound_right + max_degree_interval, 4)

        return bounds
