from typing import Literal

import numpy as np
import xarray as xr

Axis = Literal["lat", "lon"]


@xr.register_dataset_accessor("axis")
class AxisAccessor:
    axes_map = {
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
        self._obj["calculated_bounds"] = False
        bounds_vars = AxisAccessor.axes_map[axis]["bounds_vars"]
        matching_bounds_var = None

        for var in bounds_vars:
            matching_bounds_var = self._obj.data_vars.get(var)
            if matching_bounds_var is not None:
                break

        if matching_bounds_var is None:
            return self._calc_bounds(axis)

        return matching_bounds_var

    def _calc_bounds(self, axis: Axis) -> xr.DataArray:
        """Calculates axis bounds if it doesn't exist in the DataSet.

        :param axis: "lat" or "lon" bounds
        :type axis: Axis
        :return: Axis bounds
        :rtype: xr.DataArray
        """
        axis_coords = self._extract_axis_coords(axis)
        print(axis_coords)

        axis_mid = np.array(
            (np.array(axis_coords[1:]) + np.array(axis_coords[0:-1])) / 2.0
        )
        axis_bnds = np.zeros((len(axis_coords), 2))

        if np.sign(axis_coords[0]) < 0:
            axis_bnds[:, 0] = np.concatenate(([-90], axis_mid))
            axis_bnds[:, 1] = np.concatenate((axis_mid, [90]))
        else:
            axis_bnds[:, 0] = np.concatenate(([90], axis_mid))
            axis_bnds[:, 1] = np.concatenate((axis_mid, [-90]))

        # TODO: Determine the coords value for "bnds"
        axis_bnds_var = xr.DataArray(
            data=axis_bnds,
            coords={axis: np.array(axis_coords), "bnds": [0, 1]},
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
