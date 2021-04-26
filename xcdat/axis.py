from typing import List, Literal, TypedDict

import numpy as np
import xarray as xr


@xr.register_dataset_accessor("axis")
class AxisAccessor:
    """A class used to represent an AxisAccessor (xarray dataset accessor)

    :param xarray_obj: The Dataset object to be extended
    :type xarray_obj: xr.DataArray
    """

    Axis = Literal["lat", "lon"]
    AxesMapValue = TypedDict(
        "AxesMapValue", {"coords": List[str], "bounds_vars": List[str]}
    )
    AxesMap = TypedDict("AxesMap", {"lat": AxesMapValue, "lon": AxesMapValue})

    # Mapping of generic axes names to the possible names for Dataset
    # coordinates and boundary variables
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
        self._obj = xarray_obj

    def get_bounds(self, axis: Axis) -> xr.DataArray:
        """Get bounds for an axis.

        :param axis: "lat" or "lon" axis
        :type axis: Axis
        :return: The axis bounds, either existing bounds or calculated bounds
        :rtype: xr.DataArray
        """
        bounds_vars = AxisAccessor.axes_map[axis]["bounds_vars"]
        matching_bounds_var = None

        for var in bounds_vars:
            matching_bounds_var = self._obj.data_vars.get(var)
            if matching_bounds_var is not None:
                break

        if matching_bounds_var is None:
            return self._gen_bounds(axis)

        return matching_bounds_var

    def _gen_bounds(self, axis: Axis, width: float = 1) -> xr.DataArray:
        """Generates the bounds variable for an axis and adds it to the Dataset.

        :param axis: [description]
        :type axis: Axis
        :param width: Width of the bounds when axis length is 1, defaults to 1
        :type width: float, optional
        :raises TypeError: If the Dataset has coordinates without "units" attribute
        :return: The generated axis bounds DataArray
        :rtype: xr.DataArray
        """
        axis_coords = self._extract_axis_coords(axis)
        bounds = self._gen_base_bounds(axis_coords, width)

        units_attr = axis_coords.attrs.get("units")
        if units_attr is None:
            raise TypeError(f"Dataset {axis} coordinates has no attribute 'units'")
        if "degree" not in units_attr:
            raise ValueError(
                f"Dataset {axis} coordinates 'units' attr, {units_attr} does not contain 'degree'"
            )

        if axis == "lon" and len(bounds.shape) == 2:
            bounds = self._calc_lat_bounds(bounds)
        if axis == "lat":
            bounds = self._calc_lon_bounds(bounds)

        # Compose final DataArray to set as a data variable
        axis_bnds_var = xr.DataArray(
            data=bounds,
            coords={axis: axis_coords.data},
            dims=[axis, "bnds"],
            attrs={"units": axis_coords.units, "is_calculated": True},
        )
        self._obj[f"{axis}_bnds"] = axis_bnds_var
        return axis_bnds_var

    def _extract_axis_coords(self, axis: Axis) -> xr.DataArray:
        """Extracts the coordinates for the specified axis.

        :param axis: "lat" or "lon" axis
        :type axis: Axis
        :raises KeyError: If an incorrect "axis" argument is passed
        :return: Matching coordinates
        :rtype: xr.DataArray
        """
        axis_coord_names = AxisAccessor.axes_map[axis]["coords"]
        matching_coords = None

        for name in axis_coord_names:
            matching_coords = self._obj.coords.get(name)
            if matching_coords is not None:
                break

        if matching_coords is None:
            raise TypeError(f"No matching coordinates for axis: {axis}")

        return matching_coords

    def _gen_base_bounds(
        self,
        axis_coords: xr.DataArray,
        width: float,
    ) -> np.ndarray:
        """Generates the base 2D bounds array based on the axis coordinates variable.

        :param axis_coords: The axis coordinates variable
        :type axis_coords: xr.DataArray
        :param width: Width of the bounds
        :type width: float
        :return: [description]
        :rtype: np.ndarray
        """
        axis_coords_data = axis_coords.data

        if len(axis_coords_data) > 1:
            left = np.array([1.5 * axis_coords_data[0] - 0.5 * axis_coords_data[1]])
            center = (axis_coords_data[0:-1] + axis_coords_data[1:]) / 2.0
            right = np.array([1.5 * axis_coords_data[-1] - 0.5 * axis_coords_data[-2]])

            bounds = np.concatenate((left, center, right))
        else:
            delta = width / 2.0
            bounds = np.array([axis_coords[0] - delta, axis_coords[0] + delta])

        bounds_2d = np.array(list(zip(*(bounds[i:] for i in range(2)))))
        return bounds_2d

    def _calc_lat_bounds(self, bounds_2d: np.ndarray) -> np.ndarray:
        """Calculate latitude boundaries and avoids floating point errors.

        :param bounds_2d: [description]
        :type bounds_2d: np.ndarray
        :return: [description]
        :rtype: np.ndarray

        # TODO: If possible, refactor this method to simplify conditionals
        """
        min_neg_val = bounds_2d[0, 0]
        min_pos_val = bounds_2d[0, 1]
        second_max_pos_val = bounds_2d[-1, 0]
        max_pos_val = bounds_2d[-1, 1]
        max_degree_interval = 360.0

        # Check if bounds are close to the max degree interval (360)
        near_max_degree_interval: bool = abs(
            abs(max_pos_val - min_neg_val) - max_degree_interval
        ) < np.minimum(0.01, abs(min_pos_val - min_neg_val) * 0.1)

        if near_max_degree_interval:
            # For (-180, 180), if either bound is near an integer value, round both integers
            # Otherwise it is not needed if all values are positive (0, 360)
            start_bound_near_int: bool = (
                abs(min_neg_val - np.floor(min_neg_val + 0.5))
                < abs(min_pos_val - min_neg_val) * 0.01
            )
            end_bound_near_int: bool = (
                abs(max_pos_val - np.floor(max_pos_val + 0.5))
                < abs(max_pos_val - second_max_pos_val) * 0.01
            )
            if (
                start_bound_near_int or end_bound_near_int
            ) and min_neg_val * max_pos_val < 0:
                bounds_2d[0, 0] = np.floor(min_neg_val + 0.5)
                bounds_2d[0, 1] = np.floor(max_pos_val + 0.5)
            else:
                if max_pos_val > min_neg_val:
                    bounds_2d[-1, 1] = min_neg_val + max_degree_interval
                else:
                    bounds_2d[0, 0] = max_pos_val + max_degree_interval

        return bounds_2d

    def _calc_lon_bounds(self, bounds_2d: np.ndarray) -> np.ndarray:
        """Calculates longitude boundaries and avoids floating point errors.

        :param bounds_2d: [description]
        :type bounds_2d: np.ndarray
        :return: [description]
        :rtype: np.ndarray
        """
        bounds_2d[0, ...] = np.maximum(-90.0, np.minimum(90.0, bounds_2d[0, ...]))
        bounds_2d[-1, ...] = np.maximum(-90.0, np.minimum(90.0, bounds_2d[-1, ...]))
        return bounds_2d
