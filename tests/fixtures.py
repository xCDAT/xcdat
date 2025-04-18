"""This module stores reusable test fixtures."""

from typing import Literal

import cftime
import numpy as np
import pandas as pd
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
time_yearly = xr.DataArray(
    data=np.array(
        [
            cftime.DatetimeGregorian(2000, 7, 1, 12, 0, 0, 0, has_year_zero=False),
            cftime.DatetimeGregorian(2001, 7, 1, 12, 0, 0, 0, has_year_zero=False),
            cftime.DatetimeGregorian(2002, 7, 1, 12, 0, 0, 0, has_year_zero=False),
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
time_daily = xr.DataArray(
    data=np.array(
        [
            cftime.DatetimeGregorian(2000, 1, 1, 12, 0, 0, 0, has_year_zero=False),
            cftime.DatetimeGregorian(2000, 1, 2, 12, 0, 0, 0, has_year_zero=False),
            cftime.DatetimeGregorian(2000, 1, 3, 12, 0, 0, 0, has_year_zero=False),
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
time_hourly = xr.DataArray(
    data=np.array(
        [
            cftime.DatetimeGregorian(2000, 1, 1, 0, 0, 0, 0, has_year_zero=False),
            cftime.DatetimeGregorian(2000, 1, 1, 1, 0, 0, 0, has_year_zero=False),
            cftime.DatetimeGregorian(2000, 1, 1, 2, 0, 0, 0, has_year_zero=False),
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
time_hourly_dt = xr.DataArray(
    data=np.array(
        [
            "2000-01-01T00:00:00.000000000",
            "2000-01-01T01:00:00.000000000",
            "2000-01-01T02:00:00.000000000",
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
time_subhourly = xr.DataArray(
    data=np.array(
        [
            cftime.DatetimeGregorian(2000, 1, 1, 0, 15, 0, 0, has_year_zero=False),
            cftime.DatetimeGregorian(2000, 1, 1, 0, 45, 0, 0, has_year_zero=False),
            cftime.DatetimeGregorian(2000, 1, 2, 1, 15, 0, 0, has_year_zero=False),
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
time_bnds_yearly = xr.DataArray(
    name="time_bnds",
    data=np.array(
        [
            [
                cftime.DatetimeGregorian(2000, 1, 1, 0, 0, 0, 0, has_year_zero=False),
                cftime.DatetimeGregorian(2001, 1, 1, 0, 0, 0, 0, has_year_zero=False),
            ],
            [
                cftime.DatetimeGregorian(2001, 1, 1, 0, 0, 0, 0, has_year_zero=False),
                cftime.DatetimeGregorian(2002, 1, 1, 0, 0, 0, 0, has_year_zero=False),
            ],
            [
                cftime.DatetimeGregorian(2002, 1, 1, 0, 0, 0, 0, has_year_zero=False),
                cftime.DatetimeGregorian(2003, 1, 1, 0, 0, 0, 0, has_year_zero=False),
            ],
        ],
        dtype=object,
    ),
    dims=["time", "bnds"],
    attrs={
        "xcdat_bounds": "True",
    },
)
time_bnds_daily = xr.DataArray(
    name="time_bnds",
    data=np.array(
        [
            [
                cftime.DatetimeGregorian(2000, 1, 1, 0, 0, 0, 0, has_year_zero=False),
                cftime.DatetimeGregorian(2000, 1, 2, 0, 0, 0, 0, has_year_zero=False),
            ],
            [
                cftime.DatetimeGregorian(2000, 1, 2, 0, 0, 0, 0, has_year_zero=False),
                cftime.DatetimeGregorian(2000, 1, 3, 0, 0, 0, 0, has_year_zero=False),
            ],
            [
                cftime.DatetimeGregorian(2000, 1, 3, 0, 0, 0, 0, has_year_zero=False),
                cftime.DatetimeGregorian(2000, 1, 4, 0, 0, 0, 0, has_year_zero=False),
            ],
        ],
        dtype=object,
    ),
    dims=["time", "bnds"],
    attrs={
        "xcdat_bounds": "True",
    },
)
time_bnds_hourly = xr.DataArray(
    name="time_bnds",
    data=np.array(
        [
            [
                cftime.DatetimeGregorian(2000, 1, 1, 0, 0, 0, 0, has_year_zero=False),
                cftime.DatetimeGregorian(2000, 1, 1, 1, 0, 0, 0, has_year_zero=False),
            ],
            [
                cftime.DatetimeGregorian(2000, 1, 1, 1, 0, 0, 0, has_year_zero=False),
                cftime.DatetimeGregorian(2000, 1, 1, 2, 0, 0, 0, has_year_zero=False),
            ],
            [
                cftime.DatetimeGregorian(2000, 1, 1, 2, 0, 0, 0, has_year_zero=False),
                cftime.DatetimeGregorian(2000, 1, 1, 3, 0, 0, 0, has_year_zero=False),
            ],
        ],
        dtype=object,
    ),
    dims=["time", "bnds"],
    attrs={
        "xcdat_bounds": "True",
    },
)
time_bnds_hourly_dt = xr.DataArray(
    name="time_bnds",
    data=np.array(
        [
            [
                "2000-01-01T00:00:00.000000000",
                "2000-01-01T01:00:00.000000000",
            ],
            [
                "2000-01-01T01:00:00.000000000",
                "2000-01-01T02:00:00.000000000",
            ],
            [
                "2000-01-01T02:00:00.000000000",
                "2000-01-01T03:00:00.000000000",
            ],
        ],
        dtype="datetime64[ns]",
    ),
    dims=["time", "bnds"],
    attrs={
        "xcdat_bounds": "True",
    },
)
time_bnds_subhourly = xr.DataArray(
    name="time_bnds",
    data=np.array(
        [
            [
                cftime.DatetimeGregorian(2000, 1, 1, 0, 0, 0, 0, has_year_zero=False),
                cftime.DatetimeGregorian(2000, 1, 1, 0, 30, 0, 0, has_year_zero=False),
            ],
            [
                cftime.DatetimeGregorian(2000, 1, 1, 0, 30, 0, 0, has_year_zero=False),
                cftime.DatetimeGregorian(2000, 1, 1, 1, 0, 0, 0, has_year_zero=False),
            ],
            [
                cftime.DatetimeGregorian(2000, 1, 1, 1, 0, 0, 0, has_year_zero=False),
                cftime.DatetimeGregorian(2000, 1, 1, 1, 30, 0, 0, has_year_zero=False),
            ],
        ],
        dtype=object,
    ),
    dims=["time", "bnds"],
    attrs={
        "xcdat_bounds": "True",
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

# LEVEL
# =====
lev = xr.DataArray(
    data=np.flip(np.arange(2000, 10000, 2000)),
    dims=["lev"],
    attrs={"units": "m", "positive": "down", "axis": "Z"},
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


# TODO merge with generate_dataset to allow 4th dimension
def generate_lev_dataset(position="center") -> xr.Dataset:
    ds = xr.Dataset(
        data_vars={
            "so": xr.DataArray(
                name="so",
                data=np.ones((15, 4, 4, 4)),
                coords={"time": time_decoded, "lev": lev, "lat": lat, "lon": lon},
            ),
        },
        coords={
            "lat": lat.copy(),
            "lon": lon.copy(),
            "time": time_decoded.copy(),
            "lev": lev.copy(),
        },
    )

    ds["time"].encoding["calendar"] = "standard"

    ds = ds.bounds.add_missing_bounds(axes=["X", "Y", "Z", "T"])

    if position == "left":
        ds["lev"] = ds["lev_bnds"][:, 0]
    elif position == "right":
        ds["lev"] = ds["lev_bnds"][:, 1]
    elif position == "malformed":
        ds["lev"] = np.random.random(ds["lev"].shape)

    ds["lev"].attrs["axis"] = "Z"
    ds["lev"].attrs["bounds"] = "lev_bnds"

    return ds


def generate_multiple_variable_dataset(
    copies: int, separate_dims: bool = False, **kwargs
) -> xr.Dataset:
    ds_base = generate_dataset(**kwargs)

    datasets = [ds_base]

    for idx in range(copies):
        ds_copy = ds_base.copy(deep=True)

        var_names = list(["ts"])

        if separate_dims:
            var_names += list(ds_base.sizes.keys())  # type: ignore[arg-type]

        ds_copy = ds_copy.rename({x: f"{x}{idx + 1}" for x in var_names})

        datasets.append(ds_copy)

    return xr.merge(datasets, compat="override")


def generate_dataset(
    decode_times: bool, cf_compliant: bool, has_bounds: bool
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
    # Get the data_var, time axis, and time_bnds based on if time is decoded.
    if decode_times:
        ts = ts_decoded.copy()
        time = time_decoded.copy()
        time_bnds = time_bnds_decoded.copy()
    else:
        ts = ts_encoded.copy()
        time = time_encoded.copy()
        time_bnds = time_bnds_encoded.copy()

    # Create the base dataset.
    ds = xr.Dataset(
        data_vars={"ts": ts},
        coords={"lat": lat.copy(), "lon": lon.copy(), "time": time},
    )

    # Add the calendar and units attr to the encoding dict.
    if decode_times:
        ds["time"].encoding["calendar"] = "standard"
        if cf_compliant:
            ds["time"].encoding["units"] = "days since 2000-01-01"
        else:
            ds["time"].encoding["units"] = "months since 2000-01-01"
    else:
        ds["time"].attrs["calendar"] = "standard"
        if cf_compliant:
            ds["time"].attrs["units"] = "days since 2000-01-01"
        else:
            ds["time"].attrs["units"] = "months since 2000-01-01"

    if has_bounds:
        ds["lat_bnds"] = lat_bnds.copy()
        ds["lon_bnds"] = lon_bnds.copy()
        ds["time_bnds"] = time_bnds

        # If the "bounds" attribute is included in an existing DataArray and
        # added to a new Dataset, it will get dropped. Therefore, it needs to be
        # assigned to the DataArrays after they are added to Dataset.
        ds["lat"].attrs["bounds"] = "lat_bnds"
        ds["lon"].attrs["bounds"] = "lon_bnds"
        ds["time"].attrs["bounds"] = "time_bnds"

    return ds


def generate_dataset_by_frequency(
    freq: Literal["subhour", "hour", "day", "month", "year"] = "month",
    obj_type: Literal["cftime", "datetime"] = "cftime",
) -> xr.Dataset:
    """Generates a dataset for a given temporal frequency.

    This function uses the coordinate and data variable fixtures to generate a
    dataset that is decoded, cf-compliant, and includes bounds.

    Parameters
    ----------
    freq : Literal["subhour", "hour", "day", "month", "year"], optional
        Frequency of time step (and bounds), by default 'month'.

    Returns
    -------
    xr.Dataset
        Test dataset.
    """
    # get correct time axis and time_bnds
    if freq == "month":
        time = time_decoded.copy()
        time_bnds = time_bnds_decoded.copy()
    elif freq == "year":
        time = time_yearly.copy()
        time_bnds = time_bnds_yearly.copy()
    elif freq == "day":
        time = time_daily.copy()
        time_bnds = time_bnds_daily.copy()
    elif freq == "hour":
        # Test cftime and datetime. datetime subtraction results in
        # dtype=timedelta64[ns] objects, which need to be converted to Pandas
        # TimeDelta objects to use the `.seconds` time component.
        if obj_type == "cftime":
            time = time_hourly.copy()
            time_bnds = time_bnds_hourly.copy()
        else:
            time = time_hourly_dt.copy()
            time_bnds = time_bnds_hourly_dt.copy()
    elif freq == "subhour":
        time = time_subhourly.copy()
        time_bnds = time_bnds_subhourly.copy()

    # Create the base dataset.
    ds = xr.Dataset(
        data_vars={"ts": ts_decoded},
        coords={"lat": lat.copy(), "lon": lon.copy(), "time": time},
    )

    # Add the calendar and units attr to the encoding dict.
    ds["time"].encoding["calendar"] = "standard"
    ds["time"].encoding["units"] = "days since 2000-01-01"

    ds["lat_bnds"] = lat_bnds.copy()
    ds["lon_bnds"] = lon_bnds.copy()
    ds["time_bnds"] = time_bnds

    # If the "bounds" attribute is included in an existing DataArray and
    # added to a new Dataset, it will get dropped. Therefore, it needs to be
    # assigned to the DataArrays after they are added to Dataset.
    ds["lat"].attrs["bounds"] = "lat_bnds"
    ds["lon"].attrs["bounds"] = "lon_bnds"
    ds["time"].attrs["bounds"] = "time_bnds"

    return ds


def generate_curvilinear_dataset() -> xr.Dataset:
    """Generate a curvilinear Dataset with CF-compliant metadata.

    The dataset includes variables for time, latitude, longitude, and their
    respective bounds. It also contains a synthetic data variable (``test_var``)
    and additional coordinate information.

    Returns
    -------
    xr.Dataset
        A curvilinear xarray Dataset with the following structure:
        - Dimensions: time, nlat, nlon, vertices, d2, bnds
        - Coordinates:
            - lat (nlat, nlon): 2D latitude values
            - lon (nlat, nlon): 2D longitude values
            - nlat (nlat): Cell indices along the second dimension
            - nlon (nlon): Cell indices along the first dimension
            - time (time): Time values
        - Data variables:
            - test_var (time, nlat, nlon): Synthetic data array
            - time_bnds (time, d2): Time bounds
            - lat_bnds (nlat, nlon, vertices): Latitude bounds for each grid cell
            - lon_bnds (nlat, nlon, vertices): Longitude bounds for each grid cell
            - nlat_bnds (nlat, bnds): Bounds for nlat indices
            - nlon_bnds (nlon, bnds): Bounds for nlon indices

    Notes
    -----
    - The latitude and longitude bounds are calculated assuming uniform spacing.
    - The time bounds are generated with a fixed 30-day offset.
    - Metadata attributes are added to ensure CF-compliance.
    """
    # Define the dimensions
    n_time = 4
    n_lat = 4
    n_lon = 4
    n_vertices = 4

    # Create a time range with 12 monthly points
    time_vals = pd.date_range("2015-01-15", periods=n_time, freq="MS")

    # Create a simple 1D lat/lon array and meshgrid for 2D lat/lon
    lat_1d = np.linspace(-90, 90, n_lat)
    lon_1d = np.linspace(-180, 180, n_lon)
    lat_2d, lon_2d = np.meshgrid(lat_1d, lon_1d, indexing="ij")  # shape: (n_lat, n_lon)

    # Create random data array for var.
    var_data = np.zeros((n_time, n_lat, n_lon))

    # Create time bounds (shape: (time, d2))
    time_bnds_vals = np.stack(
        [time_vals.values, (time_vals + pd.DateOffset(days=30)).values], axis=1
    )

    # Add bounds for lat_2d and lon_2d
    lat_bnds_data = np.zeros((n_lat, n_lon, n_vertices))
    lon_bnds_data = np.zeros((n_lat, n_lon, n_vertices))

    # Calculate bounds for each grid cell
    for i in range(n_lat):
        for j in range(n_lon):
            lat_bnds_data[i, j] = [
                lat_2d[i, j] - (90 / n_lat),
                lat_2d[i, j] - (90 / n_lat),
                lat_2d[i, j] + (90 / n_lat),
                lat_2d[i, j] + (90 / n_lat),
            ]
            lon_bnds_data[i, j] = [
                lon_2d[i, j] - (180 / n_lon),
                lon_2d[i, j] + (180 / n_lon),
                lon_2d[i, j] + (180 / n_lon),
                lon_2d[i, j] - (180 / n_lon),
            ]

    # Build the Dataset
    ds = xr.Dataset(
        data_vars={
            "test_var": (("time", "nlat", "nlon"), var_data),
            "time_bnds": (("time", "d2"), time_bnds_vals),
            "lat_bnds": (("nlat", "nlon", "vertices"), lat_bnds_data),
            "lon_bnds": (("nlat", "nlon", "vertices"), lon_bnds_data),
            "nlat_bnds": (
                ("nlat", "bnds"),
                np.stack([np.arange(n_lat) - 0.5, np.arange(n_lat) + 0.5], axis=1),
            ),
            "nlon_bnds": (
                ("nlon", "bnds"),
                np.stack([np.arange(n_lon) - 0.5, np.arange(n_lon) + 0.5], axis=1),
            ),
        },
        coords={
            # 2D lat/lon fields
            "lat": (("nlat", "nlon"), lat_2d),
            "lon": (("nlat", "nlon"), lon_2d),
            # Main coordinates
            "nlat": ("nlat", np.arange(n_lat)),
            "nlon": ("nlon", np.arange(n_lon)),
            "time": ("time", time_vals),
        },
    )

    # Add CF metadata to time, lat, and lon
    ds["time"].attrs = {
        "standard_name": "time",
        "long_name": "time",
        "bounds": "time_bnds",
        "units": "days since 2015-01-01",
        "calendar": "gregorian",
    }

    ds["lat"].attrs = {
        "standard_name": "latitude",
        "long_name": "latitude",
        "units": "degrees_north",
        "bounds": "lat_bnds",
    }

    ds["lon"].attrs = {
        "standard_name": "longitude",
        "long_name": "longitude",
        "units": "degrees_east",
        "bounds": "lon_bnds",
    }

    ds["nlat"].attrs = {"long_name": "cell index along second dimension", "units": "1"}
    ds["nlon"].attrs = {"long_name": "cell index along first dimension", "units": "1"}

    return ds
