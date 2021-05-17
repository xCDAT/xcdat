"""This file stores reusable test fixtures."""
from datetime import datetime

import numpy as np
import xarray as xr


def generate_dataset(bounds: bool = True) -> xr.Dataset:
    """Generates a dataset using fixtures for coordinates and data variables.

    This function creates test datasets without running .copy() to create a
    shallow copy, which can easily be forgotten.

    NOTE: Using .assign() to add data variables to an existing dataset will
    remove attributes from existing coordinates. The workaround is to update a
    data_vars dict then create the dataset.
    https://github.com/pydata/xarray/issues/2245

    :param bounds: Include bounds for coordinates, defaults to False
    :type bounds: bool, optional
    :return: Test dataset with or without bounds
    :rtype: xr.Dataset
    """
    coords = {"time": time.copy(), "lat": lat.copy(), "lon": lon.copy()}
    data_vars = {"ts": ts.copy()}

    if bounds:
        data_vars.update(
            {
                "time_bnds": time_bnds.copy(),
                "lat_bnds": lat_bnds.copy(),
                "lon_bnds": lon_bnds.copy(),
            }
        )

    return xr.Dataset(coords=coords, data_vars=data_vars)


# If the fixture is an xarray object, make sure to use .copy() to create a
# shallow copy of the object. Otherwise, you might run into unintentional
# side-effects caused by reference assignment.
# https://xarray.pydata.org/en/stable/generated/xarray.DataArray.copy.html

# Coordinates
time = xr.DataArray(
    data=[
        datetime(2000, 1, 1),
        datetime(2000, 2, 1),
        datetime(2000, 3, 1),
        datetime(2000, 4, 1),
        datetime(2000, 5, 1),
        datetime(2000, 6, 1),
        datetime(2000, 7, 1),
        datetime(2000, 8, 1),
        datetime(2000, 9, 1),
        datetime(2000, 10, 1),
        datetime(2000, 11, 1),
        datetime(2000, 12, 1),
    ],
    dims=["time"],
)
lat = xr.DataArray(
    data=np.array([-90, -88.75, 88.75, 90]),
    dims=["lat"],
    attrs={"units": "degrees_north", "axis": "Y"},
)
lon = xr.DataArray(
    data=np.array([0, 1.875, 356.25, 358.125]),
    dims=["lon"],
    attrs={"units": "degrees_east", "axis": "x"},
)

# Data Variables
time_bnds = xr.DataArray(
    name="time_bnds",
    data=[
        [datetime(2000, 1, 1), datetime(2000, 2, 1)],
        [datetime(2000, 2, 1), datetime(2000, 3, 1)],
        [datetime(2000, 3, 1), datetime(2000, 4, 1)],
        [datetime(2000, 4, 1), datetime(2000, 5, 1)],
        [datetime(2000, 5, 1), datetime(2000, 6, 1)],
        [datetime(2000, 6, 1), datetime(2000, 7, 1)],
        [datetime(2000, 7, 1), datetime(2000, 8, 1)],
        [datetime(2000, 8, 1), datetime(2000, 9, 1)],
        [datetime(2000, 9, 1), datetime(2000, 10, 1)],
        [datetime(2000, 10, 1), datetime(2000, 11, 1)],
        [datetime(2000, 11, 1), datetime(2000, 12, 1)],
        [datetime(2000, 12, 1), datetime(2001, 1, 1)],
    ],
    coords={"time": time.data},
    dims=["time", "bnds"],
)
lat_bnds = xr.DataArray(
    name="lat_bnds",
    data=np.array([[-90, -89.375], [-89.375, 0.0], [0.0, 89.375], [89.375, 90]]),
    coords={"lat": lat.data},
    dims=["lat", "bnds"],
    attrs={"units": "degrees_north", "is_generated": True},
)
lon_bnds = xr.DataArray(
    name="lon_bnds",
    data=np.array(
        [
            [-0.9375, 0.9375],
            [0.9375, 179.0625],
            [179.0625, 357.1875],
            [357.1875, 359.0625],
        ]
    ),
    coords={"lon": lon.data},
    dims=["lon", "bnds"],
    attrs={"units": "degrees_east", "is_generated": True},
)

ts = xr.DataArray(
    name="ts",
    data=np.ones((12, 4, 4)),
    coords={"time": time, "lat": lat, "lon": lon},
    dims=["time", "lat", "lon"],
)
