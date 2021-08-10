"""This module stores reusable test fixtures."""
from datetime import datetime

import numpy as np
import xarray as xr

# If the fixture is an xarray object, make sure to use .copy() to create a
# shallow copy of the object. Otherwise, you might run into unintentional
# side-effects caused by reference assignment.
# https://xarray.pydata.org/en/stable/generated/xarray.DataArray.copy.html

# Dataset coordinates
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
time_non_cf_compliant = xr.DataArray(
    data=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    dims=["time"],
    attrs={"units": "months since 2000-01-01"},
)

lat = xr.DataArray(
    data=np.array([-90, -88.75, 88.75, 90]),
    dims=["lat"],
    attrs={"units": "degrees_north", "axis": "Y"},
)
lon = xr.DataArray(
    data=np.array([0, 1.875, 356.25, 358.125]),
    dims=["lon"],
    attrs={"units": "degrees_east", "axis": "X"},
)

# Dataset data variables (bounds)
time_bnds = xr.DataArray(
    name="time_bnds",
    data=[
        [datetime(1999, 12, 16, 12), datetime(2000, 1, 16, 12)],
        [datetime(2000, 1, 16, 12), datetime(2000, 2, 15, 12)],
        [datetime(2000, 2, 15, 12), datetime(2000, 3, 16, 12)],
        [datetime(2000, 3, 16, 12), datetime(2000, 4, 16)],
        [datetime(2000, 4, 16), datetime(2000, 5, 16, 12)],
        [datetime(2000, 5, 16, 12), datetime(2000, 6, 16)],
        [datetime(2000, 6, 16), datetime(2000, 7, 16, 12)],
        [datetime(2000, 7, 16, 12), datetime(2000, 8, 16, 12)],
        [datetime(2000, 8, 16, 12), datetime(2000, 9, 16)],
        [datetime(2000, 9, 16), datetime(2000, 10, 16, 12)],
        [datetime(2000, 10, 16, 12), datetime(2000, 11, 16)],
        [datetime(2000, 11, 16), datetime(2000, 12, 16)],
    ],
    coords={"time": time},
    dims=["time", "bnds"],
    attrs={"is_generated": "True"},
)

time_bnds_non_cf_compliant = xr.DataArray(
    name="time_bnds",
    data=[
        [datetime(1999, 12, 16, 12), datetime(2000, 1, 16, 12)],
        [datetime(2000, 1, 16, 12), datetime(2000, 2, 15, 12)],
        [datetime(2000, 2, 15, 12), datetime(2000, 3, 16, 12)],
        [datetime(2000, 3, 16, 12), datetime(2000, 4, 16)],
        [datetime(2000, 4, 16), datetime(2000, 5, 16, 12)],
        [datetime(2000, 5, 16, 12), datetime(2000, 6, 16)],
        [datetime(2000, 6, 16), datetime(2000, 7, 16, 12)],
        [datetime(2000, 7, 16, 12), datetime(2000, 8, 16, 12)],
        [datetime(2000, 8, 16, 12), datetime(2000, 9, 16)],
        [datetime(2000, 9, 16), datetime(2000, 10, 16, 12)],
        [datetime(2000, 10, 16, 12), datetime(2000, 11, 16)],
        [datetime(2000, 11, 16), datetime(2000, 12, 16)],
    ],
    coords={"time": time.data},
    dims=["time", "bnds"],
    attrs={"is_generated": "True"},
)
lat_bnds = xr.DataArray(
    name="lat_bnds",
    data=np.array([[-90, -89.375], [-89.375, 0.0], [0.0, 89.375], [89.375, 90]]),
    coords={"lat": lat.data},
    dims=["lat", "bnds"],
    attrs={"units": "degrees_north", "axis": "Y", "is_generated": "True"},
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
    attrs={"units": "degrees_east", "axis": "X", "is_generated": "True"},
)

# Dataset data variables (variables)
ts = xr.DataArray(
    name="ts",
    data=np.ones((12, 4, 4)),
    coords={"time": time, "lat": lat, "lon": lon},
    dims=["time", "lat", "lon"],
)
ts_non_cf_compliant = xr.DataArray(
    name="ts",
    data=np.ones((12, 4, 4)),
    coords={"time": time_non_cf_compliant, "lat": lat, "lon": lon},
    dims=["time", "lat", "lon"],
)

ts_with_bnds = xr.DataArray(
    name="ts",
    data=np.ones((2, 12, 4, 4)),
    coords={
        "bnds": np.array([0, 1]),
        "time": time.assign_attrs(bounds="time_bnds"),
        "lat": lat.assign_attrs(bounds="lat_bnds"),
        "lon": lon.assign_attrs(bounds="lon_bnds"),
        "lat_bnds": lat_bnds,
        "lon_bnds": lon_bnds,
        "time_bnds": time_bnds,
    },
    dims=[
        "bnds",
        "time",
        "lat",
        "lon",
    ],
)


def generate_dataset(cf_compliant=True, has_bounds: bool = True) -> xr.Dataset:
    """Generates a dataset using coordinate and data variable fixtures.

    NOTE: Using ``.assign()`` to add data variables to an existing dataset will
    remove attributes from existing coordinates. The workaround is to update a
    data_vars dict then create the dataset. https://github.com/pydata/xarray/issues/2245

    Parameters
    ----------
    cf_compliant : bool, optional
        CF compliant time units, by default True
    has_bounds : bool, optional
        Include bounds for coordinates, by default True

    Returns
    -------
    xr.Dataset
        Test dataset.
    """
    data_vars = {}
    coords = {
        "lat": lat.copy(),
        "lon": lon.copy(),
    }

    if cf_compliant:
        coords.update({"time": time.copy()})
        data_vars.update({"ts": ts.copy()})
    else:
        coords.update({"time": time_non_cf_compliant.copy()})
        data_vars.update({"ts": ts_non_cf_compliant.copy()})

    if has_bounds:
        data_vars.update(
            {
                "time_bnds": time_bnds.copy(),
                "lat_bnds": lat_bnds.copy(),
                "lon_bnds": lon_bnds.copy(),
            }
        )

    ds = xr.Dataset(data_vars=data_vars, coords=coords)
    return ds
