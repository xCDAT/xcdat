"""This module stores reusable test fixtures."""
import cftime
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
time_decoded = xr.DataArray(
    data=np.array(
        [
            cftime.DatetimeGregorian(2000, 1, 16, 12, 0, 0, 0, has_year_zero=False),
            cftime.DatetimeGregorian(2000, 2, 15, 12, 0, 0, 0, has_year_zero=False),
            cftime.DatetimeGregorian(2000, 3, 16, 12, 0, 0, 0, has_year_zero=False),
            cftime.DatetimeGregorian(2000, 4, 16, 0, 0, 0, 0, has_year_zero=False),
            cftime.DatetimeGregorian(2000, 5, 16, 12, 0, 0, 0, has_year_zero=False),
            cftime.DatetimeGregorian(2000, 6, 16, 0, 0, 0, 0, has_year_zero=False),
            cftime.DatetimeGregorian(2000, 7, 16, 12, 0, 0, 0, has_year_zero=False),
            cftime.DatetimeGregorian(2000, 8, 16, 12, 0, 0, 0, has_year_zero=False),
            cftime.DatetimeGregorian(2000, 9, 16, 0, 0, 0, 0, has_year_zero=False),
            cftime.DatetimeGregorian(2000, 10, 16, 12, 0, 0, 0, has_year_zero=False),
            cftime.DatetimeGregorian(2000, 11, 16, 0, 0, 0, 0, has_year_zero=False),
            cftime.DatetimeGregorian(2000, 12, 16, 12, 0, 0, 0, has_year_zero=False),
            cftime.DatetimeGregorian(2001, 1, 16, 12, 0, 0, 0, has_year_zero=False),
            cftime.DatetimeGregorian(2001, 2, 15, 0, 0, 0, 0, has_year_zero=False),
            cftime.DatetimeGregorian(2001, 12, 16, 12, 0, 0, 0, has_year_zero=False),
        ],
        dtype=object,
    ),
    dims=["time"],
    attrs={
        "axis": "T",
        "long_name": "time",
        "standard_name": "time",
    },
)
time_encoded = xr.DataArray(
    data=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    dims=["time"],
    attrs={
        "axis": "T",
        "long_name": "time",
        "standard_name": "time",
    },
)

time_bnds_decoded = xr.DataArray(
    name="time_bnds",
    data=np.array(
        [
            [
                cftime.DatetimeGregorian(2000, 1, 1, 0, 0, 0, 0, has_year_zero=False),
                cftime.DatetimeGregorian(2000, 2, 1, 0, 0, 0, 0, has_year_zero=False),
            ],
            [
                cftime.DatetimeGregorian(2000, 2, 1, 0, 0, 0, 0, has_year_zero=False),
                cftime.DatetimeGregorian(2000, 3, 1, 0, 0, 0, 0, has_year_zero=False),
            ],
            [
                cftime.DatetimeGregorian(2000, 3, 1, 0, 0, 0, 0, has_year_zero=False),
                cftime.DatetimeGregorian(2000, 4, 1, 0, 0, 0, 0, has_year_zero=False),
            ],
            [
                cftime.DatetimeGregorian(2000, 4, 1, 0, 0, 0, 0, has_year_zero=False),
                cftime.DatetimeGregorian(2000, 5, 1, 0, 0, 0, 0, has_year_zero=False),
            ],
            [
                cftime.DatetimeGregorian(2000, 5, 1, 0, 0, 0, 0, has_year_zero=False),
                cftime.DatetimeGregorian(2000, 6, 1, 0, 0, 0, 0, has_year_zero=False),
            ],
            [
                cftime.DatetimeGregorian(2000, 6, 1, 0, 0, 0, 0, has_year_zero=False),
                cftime.DatetimeGregorian(2000, 7, 1, 0, 0, 0, 0, has_year_zero=False),
            ],
            [
                cftime.DatetimeGregorian(2000, 7, 1, 0, 0, 0, 0, has_year_zero=False),
                cftime.DatetimeGregorian(2000, 8, 1, 0, 0, 0, 0, has_year_zero=False),
            ],
            [
                cftime.DatetimeGregorian(2000, 8, 1, 0, 0, 0, 0, has_year_zero=False),
                cftime.DatetimeGregorian(2000, 9, 1, 0, 0, 0, 0, has_year_zero=False),
            ],
            [
                cftime.DatetimeGregorian(2000, 9, 1, 0, 0, 0, 0, has_year_zero=False),
                cftime.DatetimeGregorian(2000, 10, 1, 0, 0, 0, 0, has_year_zero=False),
            ],
            [
                cftime.DatetimeGregorian(2000, 10, 1, 0, 0, 0, 0, has_year_zero=False),
                cftime.DatetimeGregorian(2000, 11, 1, 0, 0, 0, 0, has_year_zero=False),
            ],
            [
                cftime.DatetimeGregorian(2000, 11, 1, 0, 0, 0, 0, has_year_zero=False),
                cftime.DatetimeGregorian(2000, 12, 1, 0, 0, 0, 0, has_year_zero=False),
            ],
            [
                cftime.DatetimeGregorian(2000, 12, 1, 0, 0, 0, 0, has_year_zero=False),
                cftime.DatetimeGregorian(2001, 1, 1, 0, 0, 0, 0, has_year_zero=False),
            ],
            [
                cftime.DatetimeGregorian(2001, 1, 1, 0, 0, 0, 0, has_year_zero=False),
                cftime.DatetimeGregorian(2001, 2, 1, 0, 0, 0, 0, has_year_zero=False),
            ],
            [
                cftime.DatetimeGregorian(2001, 2, 1, 0, 0, 0, 0, has_year_zero=False),
                cftime.DatetimeGregorian(2001, 3, 1, 0, 0, 0, 0, has_year_zero=False),
            ],
            [
                cftime.DatetimeGregorian(2001, 12, 1, 0, 0, 0, 0, has_year_zero=False),
                cftime.DatetimeGregorian(2002, 1, 1, 0, 0, 0, 0, has_year_zero=False),
            ],
        ],
        dtype=object,
    ),
    dims=["time", "bnds"],
    attrs={
        "xcdat_bounds": "True",
    },
)
time_bnds_encoded = xr.DataArray(
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
    dims=["time", "bnds"],
    attrs={"xcdat_bounds": "True"},
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
ts_decoded = xr.DataArray(
    name="ts",
    data=np.ones((15, 4, 4)),
    coords={"time": time_decoded, "lat": lat, "lon": lon},
    dims=["time", "lat", "lon"],
)

ts_encoded = xr.DataArray(
    name="ts",
    data=np.ones((15, 4, 4)),
    coords={"time": time_encoded, "lat": lat, "lon": lon},
    dims=["time", "lat", "lon"],
)


def generate_dataset(
    decode_times: bool,
    cf_compliant: bool,
    has_bounds: bool,
) -> xr.Dataset:
    """Generates a dataset using coordinate and data variable fixtures.

    Parameters
    ----------
    decode_times : bool
        If True, represent time coordinates `cftime` objects. If False,
        represent time coordinates as numbers.
    cf_compliant : bool
        If True, use CF compliant time units ("days since ..."). If False,
        use non-CF compliant time units ("months since ...").
    has_bounds : bool
        Include bounds for coordinates. This also adds the "bounds" attribute
        to existing coordinates to link them to their respective bounds.

    Returns
    -------
    xr.Dataset
        Test dataset.
    """
    # First, create a dataset with either encoded or decoded coordinates.
    if decode_times:
        ds = xr.Dataset(
            data_vars={
                "ts": ts_decoded.copy(),
            },
            coords={"lat": lat.copy(), "lon": lon.copy(), "time": time_decoded.copy()},
        )

        # Add the calendar and units attr to the encoding dict.
        ds["time"].encoding["calendar"] = "standard"
        if cf_compliant:
            ds["time"].encoding["units"] = "days since 2000-01-01"
        else:
            ds["time"].encoding["units"] = "months since 2000-01-01"

    else:
        ds = xr.Dataset(
            data_vars={
                "ts": ts_encoded.copy(),
            },
            coords={"lat": lat.copy(), "lon": lon.copy(), "time": time_encoded.copy()},
        )

        # Add the calendar and units attr to the attrs dict.
        ds["time"].attrs["calendar"] = "standard"
        if cf_compliant:
            ds["time"].attrs["units"] = "days since 2000-01-01"
        else:
            ds["time"].attrs["units"] = "months since 2000-01-01"

    if has_bounds:
        ds["lat_bnds"] = lat_bnds.copy()
        ds["lon_bnds"] = lon_bnds.copy()

        if decode_times:
            ds["time_bnds"] = time_bnds_decoded.copy()
        else:
            ds["time_bnds"] = time_bnds_encoded.copy()

        # If the "bounds" attribute is included in an existing DataArray and
        # added to a new Dataset, it will get dropped. Therefore, it needs to be
        # assigned to the DataArrays after they are added to Dataset.
        ds["lat"].attrs["bounds"] = "lat_bnds"
        ds["lon"].attrs["bounds"] = "lon_bnds"
        ds["time"].attrs["bounds"] = "time_bnds"

    return ds
