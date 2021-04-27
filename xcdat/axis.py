from typing import List, Literal, TypedDict

import numpy as np
import xarray as xr

from xcdat.xcdat import logger


@xr.register_dataset_accessor("axis")
class AxisAccessor:
    """A class used to represent an AxisAccessor (xarray dataset accessor)

    :param xarray_obj: The Dataset object to be extended
    :type xarray_obj: xr.Dataset
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

    def get_bounds(self, axis: Axis, generate: bool = False) -> xr.DataArray:
        """Get bounds for an axis.

        If generate=True, generate the bounds regardless if the exist in the dataset or not.
        If generate=False and the bounds already exist, return the existing bounds.

        :param axis: "lat" or "lon" axis
        :type axis: Axis
        :return: Axis bounds
        :rtype: xr.DataArray
        """
        if generate:
            logger.info(f"Generating bounds for {axis} axis")
            return self._gen_bounds(axis)

        bounds_vars = AxisAccessor.axes_map[axis]["bounds_vars"]
        matching_bounds_var = None
        for var in bounds_vars:
            matching_bounds_var = self._obj.data_vars.get(var)
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
        :raises ValueError: If Axis coordinates' "units" attribute does not contain "degree"
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
                f"Dataset {axis} coordinates value for 'units' ('{units_attr}'), does not contain 'degree'."
            )

        # Adjust bounds to avoid floating point errors
        if axis == "lat":
            bounds = self._adjust_lat_bounds(bounds)
        if axis == "lon" and len(bounds.shape) == 2:
            bounds = self._adjust_lon_bounds(bounds)

        # Compose final DataArray to set as a data variable
        axis_bnds_var = xr.DataArray(
            data=bounds,
            coords={axis: axis_coords.data},
            dims=[axis, "bnds"],
            attrs={"units": axis_coords.units, "is_generated": True},
        )
        self._obj[f"{axis}_bnds"] = axis_bnds_var
        return axis_bnds_var

    def _extract_axis_coords(self, axis: Axis) -> xr.DataArray:
        """Extracts the coordinates data variable for the specified axis.

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
        data: np.ndarray,
        width: float,
    ) -> np.ndarray:
        """Generates the base 2D bounds array for an axis coordinates variable.

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

    def _adjust_lat_bounds(self, base_lat_bnds: np.ndarray) -> np.ndarray:
        """Adjusts latitude boundaries to avoid floating point errors.

        The endpoints are also are capped at (-90, 90).

        :param base_lat_bnds: Base latitude boundaries
        :type base_lat_bnds: np.ndarray
        :return: Adjusted latitude boundaries
        :rtype: np.ndarray
        """
        lat_bnds = base_lat_bnds
        lat_bnds[0, ...] = np.maximum(-90.0, np.minimum(90.0, lat_bnds[0, ...]))
        lat_bnds[-1, ...] = np.maximum(-90.0, np.minimum(90.0, lat_bnds[-1, ...]))

        return lat_bnds

    def _adjust_lon_bounds(self, base_lon_bounds: np.ndarray) -> np.ndarray:
        """Adjusts longitude boundaries to avoid floating point errors.

        The endpoints are also adjusted to ensure circularity.

        :param base_lon_bounds: Base longitude boundaries
        :type base_lon_bounds: np.ndarray
        :return: Adjusted longitude boundaries
        :rtype: np.ndarray

        """
        lon_bnds = base_lon_bounds

        min_left_val = lon_bnds[0, 0]
        min_right_val = lon_bnds[0, 1]
        max_left_val = lon_bnds[-1, 0]
        max_right_val = lon_bnds[-1, 1]
        max_degree = 360.0

        bounds_near_max_degree: bool = abs(
            abs(max_right_val - min_left_val) - max_degree
        ) < np.minimum(0.01, abs(min_right_val - min_left_val) * 0.1)
        min_left_val_near_int: bool = (
            abs(min_left_val - np.floor(min_left_val + 0.5))
            < abs(min_right_val - min_left_val) * 0.01
        )
        max_right_val_near_int: bool = (
            abs(max_right_val - np.floor(max_right_val + 0.5))
            < abs(max_right_val - max_left_val) * 0.01
        )

        if bounds_near_max_degree:
            # For (-180, 180), if either bound is near an integer value, round
            # both integers.
            if (
                min_left_val_near_int or max_right_val_near_int
            ) and min_left_val * max_right_val < 0:
                lon_bnds[0, 0] = np.floor(min_left_val + 0.5)
                lon_bnds[0, 1] = np.floor(max_right_val + 0.5)
            # Otherwise it is not needed if all values are positive (0, 360)
            else:
                if max_right_val > min_left_val:
                    lon_bnds[-1, 1] = min_left_val + max_degree
                else:
                    lon_bnds[0, 0] = max_right_val + max_degree

        return lon_bnds
