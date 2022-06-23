"""This module stores reusable test fixtures."""
import numpy as np
import xarray as xr

# If the fixture is an xarray object, make sure to use .copy() to create a
# shallow copy of the object. Otherwise, you might run into unintentional
# side-effects caused by reference assignment.
# https://xarray.pydata.org/en/stable/generated/xarray.DataArray.copy.html

# NOTE:
# - Non-CF time includes "units" attr
# - Coordinates with bounds includes "bounds" attr and vice versa

# TIME
# ====
time_cf = xr.DataArray(
    data=np.array(
        [
            "2000-01-16T12:00:00.000000000",
            "2000-02-15T12:00:00.000000000",
            "2000-03-16T12:00:00.000000000",
            "2000-04-16T00:00:00.000000000",
            "2000-05-16T12:00:00.000000000",
            "2000-06-16T00:00:00.000000000",
            "2000-07-16T12:00:00.000000000",
            "2000-08-16T12:00:00.000000000",
            "2000-09-16T00:00:00.000000000",
            "2000-10-16T12:00:00.000000000",
            "2000-11-16T00:00:00.000000000",
            "2000-12-16T12:00:00.000000000",
            "2001-01-16T12:00:00.000000000",
            "2001-02-15T00:00:00.000000000",
            "2001-12-16T12:00:00.000000000",
        ],
        dtype="datetime64[ns]",
    ),
    dims=["time"],
    attrs={
        "axis": "T",
        "long_name": "time",
        "standard_name": "time",
    },
)
time_non_cf = xr.DataArray(
    data=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    dims=["time"],
    attrs={
        "units": "months since 2000-01-01",
        "calendar": "standard",
        "axis": "T",
        "long_name": "time",
        "standard_name": "time",
    },
)
time_non_cf_unsupported = xr.DataArray(
    data=np.arange(1850 + 1 / 24.0, 1851 + 3 / 12.0, 1 / 12.0),
    dims=["time"],
    attrs={
        "units": "year A.D.",
        "long_name": "time",
        "standard_name": "time",
    },
)
time_bnds = xr.DataArray(
    name="time_bnds",
    data=np.array(
        [
            ["2000-01-01T00:00:00.000000000", "2000-02-01T00:00:00.000000000"],
            ["2000-02-01T00:00:00.000000000", "2000-03-01T00:00:00.000000000"],
            ["2000-03-01T00:00:00.000000000", "2000-04-01T00:00:00.000000000"],
            ["2000-04-01T00:00:00.000000000", "2000-05-01T00:00:00.000000000"],
            ["2000-05-01T00:00:00.000000000", "2000-06-01T00:00:00.000000000"],
            ["2000-06-01T00:00:00.000000000", "2000-07-01T00:00:00.000000000"],
            ["2000-07-01T00:00:00.000000000", "2000-08-01T00:00:00.000000000"],
            ["2000-08-01T00:00:00.000000000", "2000-09-01T00:00:00.000000000"],
            ["2000-09-01T00:00:00.000000000", "2000-10-01T00:00:00.000000000"],
            ["2000-10-01T00:00:00.000000000", "2000-11-01T00:00:00.000000000"],
            ["2000-11-01T00:00:00.000000000", "2000-12-01T00:00:00.000000000"],
            ["2000-12-01T00:00:00.000000000", "2001-01-01T00:00:00.000000000"],
            ["2001-01-01T00:00:00.000000000", "2001-02-01T00:00:00.000000000"],
            ["2001-02-01T00:00:00.000000000", "2001-03-01T00:00:00.000000000"],
            ["2001-12-01T00:00:00.000000000", "2002-01-01T00:00:00.000000000"],
        ],
        dtype="datetime64[ns]",
    ),
    coords={"time": time_cf},
    dims=["time", "bnds"],
    attrs={
        "xcdat_bounds": "True",
    },
)
time_bnds_non_cf = xr.DataArray(
    name="time_bnds",
    data=[
        [-1, 0],
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 8],
        [8, 9],
        [9, 10],
        [10, 11],
        [11, 12],
        [12, 13],
        [13, 14],
    ],
    coords={"time": time_non_cf},
    dims=["time", "bnds"],
    attrs={"xcdat_bounds": "True"},
)
tb = []
for t in time_non_cf_unsupported:
    tb.append([t - 1 / 24.0, t + 1 / 24.0])
time_bnds_non_cf_unsupported = xr.DataArray(
    name="time_bnds",
    data=tb,
    coords={"time": time_non_cf_unsupported},
    dims=["time", "bnds"],
    attrs={"is_generated": "True"},
)

# LATITUDE
# ========
lat = xr.DataArray(
    data=np.array([-90, -88.75, 88.75, 90]),
    dims=["lat"],
    attrs={"units": "degrees_north", "axis": "Y", "standard_name": "latitude"},
)
lat_bnds = xr.DataArray(
    name="lat_bnds",
    data=np.array([[-90, -89.375], [-89.375, 0.0], [0.0, 89.375], [89.375, 90]]),
    coords={"lat": lat},
    dims=["lat", "bnds"],
    attrs={"xcdat_bounds": "True"},
)

# LONGITUDE
# =========
lon = xr.DataArray(
    data=np.array([0, 1.875, 356.25, 358.125]),
    dims=["lon"],
    attrs={"units": "degrees_east", "axis": "X", "standard_name": "longitude"},
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
    coords={"lon": lon},
    dims=["lon", "bnds"],
    attrs={"xcdat_bounds": "True"},
)

# VARIABLES
# =========
ts_cf = xr.DataArray(
    name="ts",
    data=np.ones((15, 4, 4)),
    coords={"time": time_cf, "lat": lat, "lon": lon},
    dims=["time", "lat", "lon"],
)

ts_non_cf = xr.DataArray(
    name="ts",
    data=np.ones((15, 4, 4)),
    coords={"time": time_non_cf, "lat": lat, "lon": lon},
    dims=["time", "lat", "lon"],
)


def generate_dataset(
    cf_compliant: bool, has_bounds: bool, unsupported=False
) -> xr.Dataset:
    """Generates a dataset using coordinate and data variable fixtures.

    Parameters
    ----------
    cf_compliant : bool, optional
        CF compliant time units.
    has_bounds : bool, optional
        Include bounds for coordinates. This also adds the "bounds" attribute
        to existing coordinates to link them to their respective bounds.

    Returns
    -------
    xr.Dataset
        Test dataset.
    """
    if has_bounds:
        ds = xr.Dataset(
            data_vars={
                "ts": ts_cf.copy(),
                "lat_bnds": lat_bnds.copy(),
                "lon_bnds": lon_bnds.copy(),
            },
            coords={"lat": lat.copy(), "lon": lon.copy()},
        )

        if cf_compliant:
            ds.coords["time"] = time_cf.copy()
            ds["time_bnds"] = time_bnds.copy()
        elif not cf_compliant:
            if unsupported:
                ds.coords["time"] = time_non_cf_unsupported.copy()
                ds["time_bnds"] = time_bnds_non_cf_unsupported.copy()
            else:
                ds.coords["time"] = time_non_cf.copy()
                ds["time_bnds"] = time_bnds_non_cf.copy()

        # If the "bounds" attribute is included in an existing DataArray and
        # added to a new Dataset, it will get dropped. Therefore, it needs to be
        # assigned to the DataArrays after they are added to Dataset.
        ds["lat"].attrs["bounds"] = "lat_bnds"
        ds["lon"].attrs["bounds"] = "lon_bnds"
        ds["time"].attrs["bounds"] = "time_bnds"

    elif not has_bounds:
        ds = xr.Dataset(
            data_vars={"ts": ts_cf.copy()},
            coords={"lat": lat.copy(), "lon": lon.copy()},
        )

        if cf_compliant:
            ds.coords["time"] = time_cf.copy()
        elif not cf_compliant:
            ds.coords["time"] = time_non_cf.copy()

    return ds
