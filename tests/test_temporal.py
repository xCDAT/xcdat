import cftime
import numpy as np
import pytest
import xarray as xr
from xarray.tests import requires_dask

from tests.fixtures import generate_dataset
from xcdat.temporal import TemporalAccessor


class TestTemporalAccessor:
    def test__init__(self):
        ds: xr.Dataset = generate_dataset(cf_compliant=True, has_bounds=True)
        obj = TemporalAccessor(ds)
        assert obj._dataset.identical(ds)

    def test_decorator(self):
        ds: xr.Dataset = generate_dataset(cf_compliant=True, has_bounds=True)
        obj = ds.temporal
        assert obj._dataset.identical(ds)


class TestAverage:
    @pytest.fixture(autouse=True)
    def setup(self):
        time = xr.DataArray(
            data=np.array(
                [
                    "2000-01-16T12:00:00.000000000",
                    "2000-03-16T12:00:00.000000000",
                    "2000-06-16T00:00:00.000000000",
                    "2000-09-16T00:00:00.000000000",
                    "2001-02-15T12:00:00.000000000",
                ],
                dtype="datetime64[ns]",
            ),
            dims=["time"],
            attrs={"axis": "T", "long_name": "time", "standard_name": "time"},
        )

        time_bnds = xr.DataArray(
            name="time_bnds",
            data=np.array(
                [
                    ["2000-01-01T00:00:00.000000000", "2000-02-01T00:00:00.000000000"],
                    ["2000-03-01T00:00:00.000000000", "2000-04-01T00:00:00.000000000"],
                    ["2000-06-01T00:00:00.000000000", "2000-07-01T00:00:00.000000000"],
                    ["2000-09-01T00:00:00.000000000", "2000-10-01T00:00:00.000000000"],
                    ["2001-02-01T00:00:00.000000000", "2001-03-01T00:00:00.000000000"],
                ],
                dtype="datetime64[ns]",
            ),
            coords={"time": time},
            dims=["time", "bnds"],
            attrs={"is_generated": "True"},
        )

        self.ds = xr.Dataset(
            data_vars={"time_bnds": time_bnds},
            coords={"lat": [-90], "lon": [0], "time": time},
        )
        self.ds.time.attrs["bounds"] = "time_bnds"

        self.ds["ts"] = xr.DataArray(
            data=np.array([[[2]], [[1]], [[1]], [[1]], [[2]]]),
            coords={"time": self.ds.time, "lat": self.ds.lat, "lon": self.ds.lon},
            dims=["time", "lat", "lon"],
        )

    def test_averages_weighted_by_year(self):
        ds = self.ds.copy()

        result = ds.temporal.average("ts", freq="year")
        expected = ds.copy()
        expected["ts"] = xr.DataArray(
            data=np.array([[1.62704981]]),
            coords={"lat": expected.lat, "lon": expected.lon},
            dims=["lat", "lon"],
        )

        xr.testing.assert_allclose(result.ts, expected.ts)

    def test_averages_weighted_by_season_with_DJF_and_drop_incomplete_DJF_seasons(self):
        ds = self.ds.copy()

        result = ds.temporal.average(
            "ts",
            freq="season",
            season_config={"dec_mode": "DJF", "drop_incomplete_djf": True},
        )
        expected = ds.copy()
        expected["ts"] = xr.DataArray(
            data=np.array([[1.25]]),
            coords={"lat": expected.lat, "lon": expected.lon},
            dims=["lat", "lon"],
        )

        xr.testing.assert_allclose(result.ts, expected.ts)

    def test_averages_weighted_by_season_with_DJF_without_dropping_incomplete_DJF_seasons(
        self,
    ):
        ds = self.ds.copy()

        result = ds.temporal.average(
            "ts",
            freq="season",
            season_config={"dec_mode": "DJF", "drop_incomplete_djf": False},
        )
        expected = ds.copy()
        expected["ts"] = xr.DataArray(
            data=np.array([[1.25]]),
            coords={"lat": expected.lat, "lon": expected.lon},
            dims=["lat", "lon"],
        )

        xr.testing.assert_allclose(result.ts, expected.ts)

    def test_averages_weighted_by_season_with_JFD(self):
        ds = self.ds.copy()

        result = ds.temporal.average(
            "ts",
            freq="season",
            season_config={"dec_mode": "JFD"},
        )
        expected = ds.copy()
        expected["ts"] = xr.DataArray(
            data=np.array([[1.25]]),
            coords={"lat": expected.lat, "lon": expected.lon},
            dims=["lat", "lon"],
        )

        xr.testing.assert_allclose(result.ts, expected.ts)

    def test_averages_weighted_by_custom_season(self):
        ds = self.ds.copy()

        custom_seasons = [
            ["Jan", "Feb", "Mar"],
            ["Apr", "May", "Jun"],
            ["Jul", "Aug", "Sep"],
            ["Oct", "Nov", "Dec"],
        ]
        result = ds.temporal.average(
            "ts", "season", season_config={"custom_seasons": custom_seasons}
        )
        expected = ds.copy()
        expected["ts"] = xr.DataArray(
            data=np.array([[1.25]]),
            coords={"lat": expected.lat, "lon": expected.lon},
            dims=["lat", "lon"],
        )

        assert result.identical(expected)

    def test_averages_weighted_by_month(self):
        ds = self.ds.copy()

        result = ds.temporal.average("ts", freq="month")
        expected = ds.copy()
        expected["ts"] = xr.DataArray(
            data=np.array([[1.4]]),
            coords={"lat": expected.lat, "lon": expected.lon},
            dims=["lat", "lon"],
        )

        xr.testing.assert_allclose(result.ts, expected.ts)

    def test_averages_weighted_by_day(self):
        ds = self.ds.copy()

        result = ds.temporal.average("ts", freq="day")
        expected = ds.copy()
        expected["ts"] = xr.DataArray(
            data=np.array([[1.627049]]),
            coords={"lat": expected.lat, "lon": expected.lon},
            dims=["lat", "lon"],
        )

        xr.testing.assert_allclose(result.ts, expected.ts)

    def test_averages_weighted_by_hour(self):
        ds = self.ds.copy()

        result = ds.temporal.average("ts", freq="hour")
        expected = ds.copy()
        expected["ts"] = xr.DataArray(
            data=np.array([[1.327778]]),
            coords={"lat": expected.lat, "lon": expected.lon},
            dims=["lat", "lon"],
        )

        xr.testing.assert_allclose(result.ts, expected.ts)


class TestGroupAverage:
    @pytest.fixture(autouse=True)
    def setup(self):
        time = xr.DataArray(
            data=np.array(
                [
                    "2000-01-16T12:00:00.000000000",
                    "2000-03-16T12:00:00.000000000",
                    "2000-06-16T00:00:00.000000000",
                    "2000-09-16T00:00:00.000000000",
                    "2001-02-15T12:00:00.000000000",
                ],
                dtype="datetime64[ns]",
            ),
            dims=["time"],
            attrs={"axis": "T", "long_name": "time", "standard_name": "time"},
        )
        time_bnds = xr.DataArray(
            name="time_bnds",
            data=np.array(
                [
                    ["2000-01-01T00:00:00.000000000", "2000-02-01T00:00:00.000000000"],
                    ["2000-03-01T00:00:00.000000000", "2000-04-01T00:00:00.000000000"],
                    ["2000-06-01T00:00:00.000000000", "2000-07-01T00:00:00.000000000"],
                    ["2000-09-01T00:00:00.000000000", "2000-10-01T00:00:00.000000000"],
                    ["2001-02-01T00:00:00.000000000", "2001-03-01T00:00:00.000000000"],
                ],
                dtype="datetime64[ns]",
            ),
            coords={"time": time},
            dims=["time", "bnds"],
            attrs={"is_generated": "True"},
        )

        self.ds = xr.Dataset(
            data_vars={"time_bnds": time_bnds},
            coords={"lat": [-90], "lon": [0], "time": time},
        )
        self.ds.time.attrs["bounds"] = "time_bnds"

        self.ds["ts"] = xr.DataArray(
            data=np.array(
                [[[2.0]], [[1.0]], [[1.0]], [[1.0]], [[2.0]]], dtype="float64"
            ),
            coords={"time": self.ds.time, "lat": self.ds.lat, "lon": self.ds.lon},
            dims=["time", "lat", "lon"],
        )

    def test_weighted_annual_averages(self):
        ds = self.ds.copy()

        result = ds.temporal.group_average("ts", "year")
        expected = ds.copy()
        expected = expected.drop_dims("time")
        expected["ts"] = xr.DataArray(
            name="ts",
            data=np.array([[[1.25409836]], [[2.0]]]),
            coords={
                "lat": expected.lat,
                "lon": expected.lon,
                "time": xr.DataArray(
                    data=np.array(
                        [
                            "2000-01-01T00:00:00.000000000",
                            "2001-01-01T00:00:00.000000000",
                        ],
                        dtype="datetime64[ns]",
                    ),
                    coords={
                        "time": np.array(
                            [
                                "2000-01-01T00:00:00.000000000",
                                "2001-01-01T00:00:00.000000000",
                            ],
                            dtype="datetime64[ns]",
                        )
                    },
                    dims=["time"],
                    attrs={
                        "axis": "T",
                        "long_name": "time",
                        "standard_name": "time",
                        "bounds": "time_bnds",
                    },
                ),
            },
            dims=["time", "lat", "lon"],
            attrs={
                "operation": "temporal_avg",
                "mode": "group_average",
                "freq": "year",
                "weighted": "True",
                "center_times": "False",
            },
        )

        xr.testing.assert_allclose(result, expected)
        assert result.ts.attrs == expected.ts.attrs

    @requires_dask
    def test_weighted_annual_averages_with_chunking(self):
        ds = self.ds.copy().chunk({"time": 2})

        result = ds.temporal.group_average("ts", "year")
        expected = ds.copy()
        expected = expected.drop_dims("time")
        expected["ts"] = xr.DataArray(
            name="ts",
            data=np.array([[[1.25409836]], [[2.0]]]),
            coords={
                "lat": expected.lat,
                "lon": expected.lon,
                "time": xr.DataArray(
                    data=np.array(
                        [
                            "2000-01-01T00:00:00.000000000",
                            "2001-01-01T00:00:00.000000000",
                        ],
                        dtype="datetime64[ns]",
                    ),
                    coords={
                        "time": np.array(
                            [
                                "2000-01-01T00:00:00.000000000",
                                "2001-01-01T00:00:00.000000000",
                            ],
                            dtype="datetime64[ns]",
                        )
                    },
                    dims=["time"],
                    attrs={
                        "axis": "T",
                        "long_name": "time",
                        "standard_name": "time",
                        "bounds": "time_bnds",
                    },
                ),
            },
            dims=["time", "lat", "lon"],
            attrs={
                "operation": "temporal_avg",
                "mode": "group_average",
                "freq": "year",
                "weighted": "True",
                "center_times": "False",
            },
        )

        xr.testing.assert_allclose(result, expected)
        assert result.ts.attrs == expected.ts.attrs

    def test_weighted_seasonal_averages_with_DJF_and_drop_incomplete_seasons(self):
        ds = self.ds.copy()

        result = ds.temporal.group_average(
            "ts",
            "season",
            season_config={"dec_mode": "DJF", "drop_incomplete_djf": True},
        )
        expected = ds.copy()
        # Drop the incomplete DJF seasons
        expected = expected.isel(time=slice(2, -1))
        expected = expected.drop_dims("time")
        expected["ts"] = xr.DataArray(
            name="ts",
            data=np.array([[[1]], [[1]], [[1]], [[2.0]]]),
            coords={
                "lat": expected.lat,
                "lon": expected.lon,
                "time": xr.DataArray(
                    data=np.array(
                        [
                            "2000-04-01T00:00:00.000000000",
                            "2000-07-01T00:00:00.000000000",
                            "2000-10-01T00:00:00.000000000",
                            "2001-01-01T00:00:00.000000000",
                        ],
                        dtype="datetime64[ns]",
                    ),
                    dims=["time"],
                    attrs={
                        "axis": "T",
                        "long_name": "time",
                        "standard_name": "time",
                        "bounds": "time_bnds",
                    },
                ),
            },
            dims=["time", "lat", "lon"],
            attrs={
                "operation": "temporal_avg",
                "mode": "group_average",
                "freq": "season",
                "weighted": "True",
                "center_times": "False",
                "dec_mode": "DJF",
                "drop_incomplete_djf": "True",
            },
        )

        assert result.identical(expected)

    def test_weighted_seasonal_averages_with_DJF_without_dropping_incomplete_seasons(
        self,
    ):
        ds = self.ds.copy()

        result = ds.temporal.group_average(
            "ts",
            "season",
            season_config={"dec_mode": "DJF", "drop_incomplete_djf": False},
        )
        expected = ds.copy()
        expected = expected.drop_dims("time")
        expected["ts"] = xr.DataArray(
            name="ts",
            data=np.array([[[2.0]], [[1.0]], [[1.0]], [[1.0]], [[2.0]]]),
            coords={
                "lat": expected.lat,
                "lon": expected.lon,
                "time": xr.DataArray(
                    data=np.array(
                        [
                            "2000-01-01T00:00:00.000000000",
                            "2000-04-01T00:00:00.000000000",
                            "2000-07-01T00:00:00.000000000",
                            "2000-10-01T00:00:00.000000000",
                            "2001-01-01T00:00:00.000000000",
                        ],
                        dtype="datetime64[ns]",
                    ),
                    dims=["time"],
                    attrs={
                        "axis": "T",
                        "long_name": "time",
                        "standard_name": "time",
                        "bounds": "time_bnds",
                    },
                ),
            },
            dims=["time", "lat", "lon"],
            attrs={
                "operation": "temporal_avg",
                "mode": "group_average",
                "freq": "season",
                "weighted": "True",
                "center_times": "False",
                "dec_mode": "DJF",
                "drop_incomplete_djf": "False",
            },
        )

        assert result.identical(expected)

    def test_weighted_seasonal_averages_with_JFD(self):
        ds = self.ds.copy()

        result = ds.temporal.group_average(
            "ts",
            "season",
            season_config={"dec_mode": "JFD"},
        )
        expected = ds.copy()
        expected = expected.drop_dims("time")
        expected["ts"] = xr.DataArray(
            name="ts",
            data=np.array([[[2.0]], [[1.0]], [[1.0]], [[1.0]], [[2.0]]]),
            coords={
                "lat": expected.lat,
                "lon": expected.lon,
                "time": xr.DataArray(
                    data=np.array(
                        [
                            "2000-01-01T00:00:00.000000000",
                            "2000-04-01T00:00:00.000000000",
                            "2000-07-01T00:00:00.000000000",
                            "2000-10-01T00:00:00.000000000",
                            "2001-01-01T00:00:00.000000000",
                        ],
                        dtype="datetime64[ns]",
                    ),
                    coords={
                        "time": np.array(
                            [
                                "2000-01-01T00:00:00.000000000",
                                "2000-04-01T00:00:00.000000000",
                                "2000-07-01T00:00:00.000000000",
                                "2000-10-01T00:00:00.000000000",
                                "2001-01-01T00:00:00.000000000",
                            ],
                            dtype="datetime64[ns]",
                        )
                    },
                    dims=["time"],
                    attrs={
                        "axis": "T",
                        "long_name": "time",
                        "standard_name": "time",
                        "bounds": "time_bnds",
                    },
                ),
            },
            dims=["time", "lat", "lon"],
            attrs={
                "operation": "temporal_avg",
                "mode": "group_average",
                "freq": "season",
                "weighted": "True",
                "center_times": "False",
                "dec_mode": "JFD",
            },
        )

        assert result.identical(expected)

    def test_weighted_custom_seasonal_averages(self):
        ds = self.ds.copy()

        custom_seasons = [
            ["Jan", "Feb", "Mar"],
            ["Apr", "May", "Jun"],
            ["Jul", "Aug", "Sep"],
            ["Oct", "Nov", "Dec"],
        ]
        result = ds.temporal.group_average(
            "ts",
            "season",
            season_config={"custom_seasons": custom_seasons},
        )
        expected = ds.copy()
        expected = expected.drop_dims("time")
        expected["ts"] = xr.DataArray(
            name="ts",
            data=np.array([[[1.5]], [[1.0]], [[1.0]], [[2.0]]]),
            coords={
                "lat": expected.lat,
                "lon": expected.lon,
                "time": xr.DataArray(
                    data=np.array(
                        [
                            "2000-02-01T00:00:00.000000000",
                            "2000-05-01T00:00:00.000000000",
                            "2000-08-01T00:00:00.000000000",
                            "2001-02-01T00:00:00.000000000",
                        ],
                        dtype="datetime64[ns]",
                    ),
                    dims=["time"],
                    attrs={
                        "axis": "T",
                        "long_name": "time",
                        "standard_name": "time",
                        "bounds": "time_bnds",
                    },
                ),
            },
            dims=["time", "lat", "lon"],
            attrs={
                "operation": "temporal_avg",
                "mode": "group_average",
                "freq": "season",
                "custom_seasons": [
                    "JanFebMar",
                    "AprMayJun",
                    "JulAugSep",
                    "OctNovDec",
                ],
                "weighted": "True",
                "center_times": "False",
            },
        )

        assert result.identical(expected)

    def test_raises_error_with_incorrect_custom_seasons_argument(self):
        # Test raises error with non-3 letter strings
        with pytest.raises(ValueError):
            custom_seasons = [
                ["J", "Feb", "Mar"],
                ["Apr", "May", "Jun"],
                ["Jul", "Aug", "Sep"],
                ["Oct", "Nov", "Dec"],
            ]
            self.ds.temporal.group_average(
                "ts",
                "season",
                season_config={"custom_seasons": custom_seasons},
            )

        # Test raises error with missing month(s)
        with pytest.raises(ValueError):
            custom_seasons = [
                ["Feb", "Mar"],
                ["Apr", "May", "Jun"],
                ["Jul", "Aug", "Sep"],
                ["Oct", "Nov", "Dec"],
            ]
            self.ds.temporal.group_average(
                "ts",
                "season",
                season_config={"custom_seasons": custom_seasons},
            )

        # Test raises error if duplicate month(s) were found
        with pytest.raises(ValueError):
            custom_seasons = [
                ["Jan", "Jan", "Mar"],
                ["Apr", "May", "Jun"],
                ["Jul", "Aug", "Sep"],
                ["Oct", "Nov", "Dec"],
            ]
            self.ds.temporal.group_average(
                "ts",
                "season",
                season_config={"custom_seasons": custom_seasons},
            )

    def test_weighted_monthly_averages(self):
        ds = self.ds.copy()

        result = ds.temporal.group_average("ts", "month")
        expected = ds.copy()
        expected = expected.drop_dims("time")
        expected["ts"] = xr.DataArray(
            name="ts",
            data=np.array([[[2.0]], [[1.0]], [[1.0]], [[1.0]], [[2.0]]]),
            coords={
                "lat": expected.lat,
                "lon": expected.lon,
                "time": xr.DataArray(
                    data=np.array(
                        [
                            "2000-01-01T00:00:00.000000000",
                            "2000-03-01T00:00:00.000000000",
                            "2000-06-01T00:00:00.000000000",
                            "2000-09-01T00:00:00.000000000",
                            "2001-02-01T00:00:00.000000000",
                        ],
                        dtype="datetime64[ns]",
                    ),
                    dims=["time"],
                    attrs={
                        "axis": "T",
                        "long_name": "time",
                        "standard_name": "time",
                        "bounds": "time_bnds",
                    },
                ),
            },
            dims=["time", "lat", "lon"],
            attrs={
                "operation": "temporal_avg",
                "mode": "group_average",
                "freq": "month",
                "weighted": "True",
                "center_times": "False",
            },
        )

        assert result.identical(expected)

    def test_weighted_daily_averages(self):
        ds = self.ds.copy()

        result = ds.temporal.group_average("ts", "day")
        expected = ds.copy()
        expected = expected.drop_dims("time")
        expected["ts"] = xr.DataArray(
            name="ts",
            data=np.array([[[2.0]], [[1.0]], [[1.0]], [[1.0]], [[2.0]]]),
            coords={
                "lat": expected.lat,
                "lon": expected.lon,
                "time": xr.DataArray(
                    data=np.array(
                        [
                            "2000-01-16T00:00:00.000000000",
                            "2000-03-16T00:00:00.000000000",
                            "2000-06-16T00:00:00.000000000",
                            "2000-09-16T00:00:00.000000000",
                            "2001-02-15T00:00:00.000000000",
                        ],
                        dtype="datetime64[ns]",
                    ),
                    dims=["time"],
                    attrs={
                        "axis": "T",
                        "long_name": "time",
                        "standard_name": "time",
                        "bounds": "time_bnds",
                    },
                ),
            },
            dims=["time", "lat", "lon"],
            attrs={
                "operation": "temporal_avg",
                "mode": "group_average",
                "freq": "day",
                "weighted": "True",
                "center_times": "False",
            },
        )

        assert result.identical(expected)

    def test_weighted_daily_averages_and_center_times(self):
        ds = self.ds.copy()
        ds["time"] = xr.DataArray(
            data=np.array(
                [
                    "2000-01-01T12:00:00.000000000",
                    "2000-03-01T12:00:00.000000000",
                    "2000-06-01T00:00:00.000000000",
                    "2000-09-01T00:00:00.000000000",
                    "2001-02-01T12:00:00.000000000",
                ],
                dtype="datetime64[ns]",
            ),
            dims=["time"],
            attrs={
                "axis": "T",
                "long_name": "time",
                "standard_name": "time",
                "bounds": "time_bnds",
            },
        )
        ds["time_bnds"] = xr.DataArray(
            name="time_bnds",
            data=np.array(
                [
                    ["2000-01-01T00:00:00.000000000", "2000-02-01T00:00:00.000000000"],
                    ["2000-03-01T00:00:00.000000000", "2000-04-01T00:00:00.000000000"],
                    ["2000-06-01T00:00:00.000000000", "2000-07-01T00:00:00.000000000"],
                    ["2000-09-01T00:00:00.000000000", "2000-10-01T00:00:00.000000000"],
                    ["2001-02-01T00:00:00.000000000", "2001-03-01T00:00:00.000000000"],
                ],
                dtype="datetime64[ns]",
            ),
            coords={"time": ds.time},
            dims=["time", "bnds"],
            attrs={"is_generated": "True"},
        )

        result = ds.temporal.group_average("ts", "day", center_times=True)
        expected = ds.copy()
        expected = expected.drop_dims("time")
        expected["ts"] = xr.DataArray(
            name="ts",
            data=np.array([[[2]], [[1]], [[1]], [[1]], [[2]]]),
            coords={
                "lat": expected.lat,
                "lon": expected.lon,
                "time": xr.DataArray(
                    data=np.array(
                        [
                            "2000-01-16T00:00:00.000000000",
                            "2000-03-16T00:00:00.000000000",
                            "2000-06-16T00:00:00.000000000",
                            "2000-09-16T00:00:00.000000000",
                            "2001-02-15T00:00:00.000000000",
                        ],
                        dtype="datetime64[ns]",
                    ),
                    dims=["time"],
                    attrs={
                        "axis": "T",
                        "long_name": "time",
                        "standard_name": "time",
                        "bounds": "time_bnds",
                    },
                ),
            },
            dims=["time", "lat", "lon"],
            attrs={
                "operation": "temporal_avg",
                "mode": "group_average",
                "freq": "day",
                "weighted": "True",
                "center_times": "True",
            },
        )

        assert result.identical(expected)

    def test_weighted_hourly_averages(self):
        ds = self.ds.copy()
        ds.coords["time"].attrs["bounds"] = "time_bnds"

        result = ds.temporal.group_average("ts", "hour")
        expected = ds.copy()
        expected = expected.drop_dims("time")
        expected["ts"] = xr.DataArray(
            name="ts",
            data=np.array([[[2]], [[1]], [[1]], [[1]], [[2]]]),
            coords={
                "lat": expected.lat,
                "lon": expected.lon,
                "time": ds.time,
            },
            dims=["time", "lat", "lon"],
            attrs={
                "operation": "temporal_avg",
                "mode": "group_average",
                "freq": "hour",
                "weighted": "True",
                "center_times": "False",
            },
        )

        assert result.identical(expected)


class TestClimatology:
    # TODO: Update TestClimatology tests to use other numbers rather than 1's
    # for better test reliability and accuracy. This may require subsetting.
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds: xr.Dataset = generate_dataset(cf_compliant=True, has_bounds=True)

    def test_weighted_seasonal_climatology_with_DJF(self):
        ds = self.ds.copy()

        result = ds.temporal.climatology(
            "ts",
            "season",
            season_config={"dec_mode": "DJF", "drop_incomplete_djf": True},
        )

        expected = ds.copy()
        expected = expected.drop_dims("time")
        expected_time = xr.DataArray(
            data=np.array(
                [
                    cftime.datetime(1, 1, 1),
                    cftime.datetime(1, 4, 1),
                    cftime.datetime(1, 7, 1),
                    cftime.datetime(1, 10, 1),
                ],
            ),
            coords={
                "time": np.array(
                    [
                        cftime.datetime(1, 1, 1),
                        cftime.datetime(1, 4, 1),
                        cftime.datetime(1, 7, 1),
                        cftime.datetime(1, 10, 1),
                    ],
                ),
            },
            attrs={
                "axis": "T",
                "long_name": "time",
                "standard_name": "time",
                "bounds": "time_bnds",
            },
        )
        expected["ts"] = xr.DataArray(
            name="ts",
            data=np.ones((4, 4, 4)),
            coords={"lat": expected.lat, "lon": expected.lon, "time": expected_time},
            dims=["time", "lat", "lon"],
            attrs={
                "operation": "temporal_avg",
                "mode": "climatology",
                "freq": "season",
                "weighted": "True",
                "center_times": "False",
                "dec_mode": "DJF",
                "drop_incomplete_djf": "True",
            },
        )

        assert result.identical(expected)

    @requires_dask
    def test_chunked_weighted_seasonal_climatology_with_DJF(self):
        ds = self.ds.copy().chunk({"time": 2})

        result = ds.temporal.climatology(
            "ts",
            "season",
            season_config={"dec_mode": "DJF", "drop_incomplete_djf": True},
        )

        expected = ds.copy()
        expected = expected.drop_dims("time")
        expected_time = xr.DataArray(
            data=np.array(
                [
                    cftime.datetime(1, 1, 1),
                    cftime.datetime(1, 4, 1),
                    cftime.datetime(1, 7, 1),
                    cftime.datetime(1, 10, 1),
                ],
            ),
            coords={
                "time": np.array(
                    [
                        cftime.datetime(1, 1, 1),
                        cftime.datetime(1, 4, 1),
                        cftime.datetime(1, 7, 1),
                        cftime.datetime(1, 10, 1),
                    ],
                ),
            },
            attrs={
                "axis": "T",
                "long_name": "time",
                "standard_name": "time",
                "bounds": "time_bnds",
            },
        )
        expected["ts"] = xr.DataArray(
            name="ts",
            data=np.ones((4, 4, 4)),
            coords={"lat": expected.lat, "lon": expected.lon, "time": expected_time},
            dims=["time", "lat", "lon"],
            attrs={
                "operation": "temporal_avg",
                "mode": "climatology",
                "freq": "season",
                "weighted": "True",
                "center_times": "False",
                "dec_mode": "DJF",
                "drop_incomplete_djf": "True",
            },
        )

        assert result.identical(expected)

    def test_weighted_seasonal_climatology_with_JFD(self):
        ds = self.ds.copy()

        result = ds.temporal.climatology(
            "ts", "season", season_config={"dec_mode": "JFD"}
        )

        expected = ds.copy()
        expected = expected.drop_dims("time")
        expected_time = xr.DataArray(
            data=np.array(
                [
                    cftime.datetime(1, 1, 1),
                    cftime.datetime(1, 4, 1),
                    cftime.datetime(1, 7, 1),
                    cftime.datetime(1, 10, 1),
                ],
            ),
            coords={
                "time": np.array(
                    [
                        cftime.datetime(1, 1, 1),
                        cftime.datetime(1, 4, 1),
                        cftime.datetime(1, 7, 1),
                        cftime.datetime(1, 10, 1),
                    ],
                ),
            },
            attrs={
                "axis": "T",
                "long_name": "time",
                "standard_name": "time",
                "bounds": "time_bnds",
            },
        )
        expected["ts"] = xr.DataArray(
            name="ts",
            data=np.ones((4, 4, 4)),
            coords={"lat": expected.lat, "lon": expected.lon, "time": expected_time},
            dims=["time", "lat", "lon"],
            attrs={
                "operation": "temporal_avg",
                "mode": "climatology",
                "freq": "season",
                "weighted": "True",
                "center_times": "False",
                "dec_mode": "JFD",
            },
        )

        assert result.identical(expected)

    def test_weighted_custom_seasonal_climatology(self):
        ds = self.ds.copy()

        custom_seasons = [
            ["Jan", "Feb", "Mar"],
            ["Apr", "May", "Jun"],
            ["Jul", "Aug", "Sep"],
            ["Oct", "Nov", "Dec"],
        ]
        result = ds.temporal.climatology(
            "ts", "season", season_config={"custom_seasons": custom_seasons}
        )

        expected = ds.copy()
        expected = expected.drop_dims("time")
        expected_time = xr.DataArray(
            data=np.array(
                [
                    cftime.datetime(1, 2, 1),
                    cftime.datetime(1, 5, 1),
                    cftime.datetime(1, 8, 1),
                    cftime.datetime(1, 11, 1),
                ],
            ),
            coords={
                "time": np.array(
                    [
                        cftime.datetime(1, 2, 1),
                        cftime.datetime(1, 5, 1),
                        cftime.datetime(1, 8, 1),
                        cftime.datetime(1, 11, 1),
                    ],
                ),
            },
            attrs={
                "axis": "T",
                "long_name": "time",
                "standard_name": "time",
                "bounds": "time_bnds",
            },
        )

        expected["ts"] = xr.DataArray(
            name="ts",
            data=np.ones((4, 4, 4)),
            coords={"lat": expected.lat, "lon": expected.lon, "time": expected_time},
            dims=["time", "lat", "lon"],
            attrs={
                "operation": "temporal_avg",
                "mode": "climatology",
                "freq": "season",
                "weighted": "True",
                "center_times": "False",
                "custom_seasons": [
                    "JanFebMar",
                    "AprMayJun",
                    "JulAugSep",
                    "OctNovDec",
                ],
            },
        )

        assert result.identical(expected)

    def test_weighted_monthly_climatology(self):
        result = self.ds.temporal.climatology("ts", "month")

        expected = self.ds.copy()
        expected = expected.drop_dims("time")
        expected_time = xr.DataArray(
            data=np.array(
                [
                    cftime.datetime(1, 1, 1),
                    cftime.datetime(1, 2, 1),
                    cftime.datetime(1, 3, 1),
                    cftime.datetime(1, 4, 1),
                    cftime.datetime(1, 5, 1),
                    cftime.datetime(1, 6, 1),
                    cftime.datetime(1, 7, 1),
                    cftime.datetime(1, 8, 1),
                    cftime.datetime(1, 9, 1),
                    cftime.datetime(1, 10, 1),
                    cftime.datetime(1, 11, 1),
                    cftime.datetime(1, 12, 1),
                ],
            ),
            coords={
                "time": np.array(
                    [
                        cftime.datetime(1, 1, 1),
                        cftime.datetime(1, 2, 1),
                        cftime.datetime(1, 3, 1),
                        cftime.datetime(1, 4, 1),
                        cftime.datetime(1, 5, 1),
                        cftime.datetime(1, 6, 1),
                        cftime.datetime(1, 7, 1),
                        cftime.datetime(1, 8, 1),
                        cftime.datetime(1, 9, 1),
                        cftime.datetime(1, 10, 1),
                        cftime.datetime(1, 11, 1),
                        cftime.datetime(1, 12, 1),
                    ],
                ),
            },
            attrs={
                "axis": "T",
                "long_name": "time",
                "standard_name": "time",
                "bounds": "time_bnds",
            },
        )

        expected["ts"] = xr.DataArray(
            name="ts",
            data=np.ones((12, 4, 4)),
            coords={"lat": expected.lat, "lon": expected.lon, "time": expected_time},
            dims=["time", "lat", "lon"],
            attrs={
                "operation": "temporal_avg",
                "mode": "climatology",
                "freq": "month",
                "weighted": "True",
                "center_times": "False",
            },
        )

        assert result.identical(expected)

    def test_unweighted_monthly_climatology(self):
        result = self.ds.temporal.climatology("ts", "month", weighted=False)

        expected = self.ds.copy()
        expected = expected.drop_dims("time")
        expected_time = xr.DataArray(
            data=np.array(
                [
                    cftime.datetime(1, 1, 1),
                    cftime.datetime(1, 2, 1),
                    cftime.datetime(1, 3, 1),
                    cftime.datetime(1, 4, 1),
                    cftime.datetime(1, 5, 1),
                    cftime.datetime(1, 6, 1),
                    cftime.datetime(1, 7, 1),
                    cftime.datetime(1, 8, 1),
                    cftime.datetime(1, 9, 1),
                    cftime.datetime(1, 10, 1),
                    cftime.datetime(1, 11, 1),
                    cftime.datetime(1, 12, 1),
                ],
            ),
            coords={
                "time": np.array(
                    [
                        cftime.datetime(1, 1, 1),
                        cftime.datetime(1, 2, 1),
                        cftime.datetime(1, 3, 1),
                        cftime.datetime(1, 4, 1),
                        cftime.datetime(1, 5, 1),
                        cftime.datetime(1, 6, 1),
                        cftime.datetime(1, 7, 1),
                        cftime.datetime(1, 8, 1),
                        cftime.datetime(1, 9, 1),
                        cftime.datetime(1, 10, 1),
                        cftime.datetime(1, 11, 1),
                        cftime.datetime(1, 12, 1),
                    ],
                ),
            },
            attrs={
                "axis": "T",
                "long_name": "time",
                "standard_name": "time",
                "bounds": "time_bnds",
            },
        )
        expected["ts"] = xr.DataArray(
            name="ts",
            data=np.ones((12, 4, 4)),
            coords={"lat": expected.lat, "lon": expected.lon, "time": expected_time},
            dims=["time", "lat", "lon"],
            attrs={
                "operation": "temporal_avg",
                "mode": "climatology",
                "freq": "month",
                "weighted": "False",
                "center_times": "False",
            },
        )

        assert result.identical(expected)

    def test_weighted_daily_climatology(self):
        result = self.ds.temporal.climatology("ts", "day", weighted=True)

        expected = self.ds.copy()
        expected = expected.drop_dims("time")
        expected_time = xr.DataArray(
            data=np.array(
                [
                    cftime.datetime(1, 1, 16),
                    cftime.datetime(1, 2, 15),
                    cftime.datetime(1, 3, 16),
                    cftime.datetime(1, 4, 16),
                    cftime.datetime(1, 5, 16),
                    cftime.datetime(1, 6, 16),
                    cftime.datetime(1, 7, 16),
                    cftime.datetime(1, 8, 16),
                    cftime.datetime(1, 9, 16),
                    cftime.datetime(1, 10, 16),
                    cftime.datetime(1, 11, 16),
                    cftime.datetime(1, 12, 16),
                ],
            ),
            coords={
                "time": np.array(
                    [
                        cftime.datetime(1, 1, 16),
                        cftime.datetime(1, 2, 15),
                        cftime.datetime(1, 3, 16),
                        cftime.datetime(1, 4, 16),
                        cftime.datetime(1, 5, 16),
                        cftime.datetime(1, 6, 16),
                        cftime.datetime(1, 7, 16),
                        cftime.datetime(1, 8, 16),
                        cftime.datetime(1, 9, 16),
                        cftime.datetime(1, 10, 16),
                        cftime.datetime(1, 11, 16),
                        cftime.datetime(1, 12, 16),
                    ],
                ),
            },
            attrs={
                "axis": "T",
                "long_name": "time",
                "standard_name": "time",
                "bounds": "time_bnds",
            },
        )
        expected["ts"] = xr.DataArray(
            name="ts",
            data=np.ones((12, 4, 4)),
            coords={"lat": expected.lat, "lon": expected.lon, "time": expected_time},
            dims=["time", "lat", "lon"],
            attrs={
                "operation": "temporal_avg",
                "mode": "climatology",
                "freq": "day",
                "weighted": "True",
                "center_times": "False",
            },
        )

        assert result.identical(expected)

    def test_unweighted_daily_climatology(self):
        result = self.ds.temporal.climatology("ts", "day", weighted=False)

        expected = self.ds.copy()
        expected = expected.drop_dims("time")
        expected_time = xr.DataArray(
            data=np.array(
                [
                    cftime.datetime(1, 1, 16),
                    cftime.datetime(1, 2, 15),
                    cftime.datetime(1, 3, 16),
                    cftime.datetime(1, 4, 16),
                    cftime.datetime(1, 5, 16),
                    cftime.datetime(1, 6, 16),
                    cftime.datetime(1, 7, 16),
                    cftime.datetime(1, 8, 16),
                    cftime.datetime(1, 9, 16),
                    cftime.datetime(1, 10, 16),
                    cftime.datetime(1, 11, 16),
                    cftime.datetime(1, 12, 16),
                ],
            ),
            coords={
                "time": np.array(
                    [
                        cftime.datetime(1, 1, 16),
                        cftime.datetime(1, 2, 15),
                        cftime.datetime(1, 3, 16),
                        cftime.datetime(1, 4, 16),
                        cftime.datetime(1, 5, 16),
                        cftime.datetime(1, 6, 16),
                        cftime.datetime(1, 7, 16),
                        cftime.datetime(1, 8, 16),
                        cftime.datetime(1, 9, 16),
                        cftime.datetime(1, 10, 16),
                        cftime.datetime(1, 11, 16),
                        cftime.datetime(1, 12, 16),
                    ],
                ),
            },
            attrs={
                "axis": "T",
                "long_name": "time",
                "standard_name": "time",
                "bounds": "time_bnds",
            },
        )
        expected["ts"] = xr.DataArray(
            name="ts",
            data=np.ones((12, 4, 4)),
            coords={"lat": expected.lat, "lon": expected.lon, "time": expected_time},
            dims=["time", "lat", "lon"],
            attrs={
                "operation": "temporal_avg",
                "mode": "climatology",
                "freq": "day",
                "weighted": "False",
                "center_times": "False",
            },
        )

        assert result.identical(expected)


class TestDepartures:
    # TODO: Update TestDepartures tests to use other numbers rather than 1's for
    # better test reliability and accuracy. This may require subsetting.
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds: xr.Dataset = generate_dataset(cf_compliant=True, has_bounds=True)

        self.seasons = ["JJA", "MAM", "SON", "DJF"]

    def test_weighted_seasonal_departures_with_DJF(self):
        # Create a post-climatology dataset.
        ds = self.ds.copy()
        # Drop incomplete DJF seasons
        ds = ds.isel(time=slice(2, -1))

        # Compare result of the method against the expected.
        result = ds.temporal.departures(
            "ts",
            "season",
            season_config={"dec_mode": "DJF", "drop_incomplete_djf": True},
        )
        expected = ds.copy()
        expected["ts"] = xr.DataArray(
            data=np.zeros((12, 4, 4)),
            coords={
                "lat": expected.lat,
                "lon": expected.lon,
                "time": ds.time,
            },
            dims=["time", "lat", "lon"],
            attrs={
                "operation": "temporal_avg",
                "mode": "departures",
                "freq": "season",
                "weighted": "True",
                "center_times": "False",
                "dec_mode": "DJF",
                "drop_incomplete_djf": "True",
            },
        )

        assert result.identical(expected)

    def test_unweighted_seasonal_departures_with_DJF(self):
        ds = self.ds.copy()
        # Drop incomplete DJF seasons
        ds = ds.isel(time=slice(2, -1))

        # Compare result of the method against the expected.
        result = ds.temporal.departures(
            "ts",
            "season",
            weighted=False,
            season_config={"dec_mode": "DJF", "drop_incomplete_djf": True},
        )
        expected = ds.copy()
        expected["ts"] = xr.DataArray(
            data=np.zeros((12, 4, 4)),
            coords={
                "lat": expected.lat,
                "lon": expected.lon,
                "time": ds.time,
            },
            dims=["time", "lat", "lon"],
            attrs={
                "operation": "temporal_avg",
                "mode": "departures",
                "freq": "season",
                "weighted": "False",
                "center_times": "False",
                "dec_mode": "DJF",
                "drop_incomplete_djf": "True",
            },
        )

        assert result.identical(expected)

    def test_unweighted_seasonal_departures_with_JFD(self):
        ds = self.ds.copy()

        # Compare result of the method against the expected.
        result = ds.temporal.departures(
            "ts",
            "season",
            weighted=False,
            season_config={"dec_mode": "JFD"},
        )
        expected = ds.copy()
        expected["ts"] = xr.DataArray(
            data=np.zeros((15, 4, 4)),
            coords={
                "lat": expected.lat,
                "lon": expected.lon,
                "time": ds.time,
            },
            dims=["time", "lat", "lon"],
            attrs={
                "operation": "temporal_avg",
                "mode": "departures",
                "freq": "season",
                "weighted": "False",
                "center_times": "False",
                "dec_mode": "JFD",
            },
        )

        assert result.identical(expected)


class TestCenterTimes:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = generate_dataset(cf_compliant=True, has_bounds=True)

    def test_raises_error_if_time_dimension_does_not_exist_in_dataset(self):
        ds = self.ds.copy()
        ds = ds.drop_dims("time")

        with pytest.raises(KeyError):
            ds.temporal.center_times(ds)

    def test_gets_time_as_the_midpoint_between_time_bounds(self):
        ds = self.ds.copy()

        # Make the time coordinates uncentered.
        uncentered_time = np.array(
            [
                "2000-01-31T12:00:00.000000000",
                "2000-02-29T12:00:00.000000000",
                "2000-03-31T12:00:00.000000000",
                "2000-04-30T00:00:00.000000000",
                "2000-05-31T12:00:00.000000000",
                "2000-06-30T00:00:00.000000000",
                "2000-07-31T12:00:00.000000000",
                "2000-08-31T12:00:00.000000000",
                "2000-09-30T00:00:00.000000000",
                "2000-10-16T12:00:00.000000000",
                "2000-11-30T00:00:00.000000000",
                "2000-12-31T12:00:00.000000000",
                "2001-01-31T12:00:00.000000000",
                "2001-02-28T00:00:00.000000000",
                "2001-12-31T12:00:00.000000000",
            ],
            dtype="datetime64[ns]",
        )
        ds.time.data[:] = uncentered_time

        # Set object attrs required to test the method.
        ds.temporal._time_bounds = ds.time_bnds.copy()

        # Compare result of the method against the expected.
        expected = ds.copy()
        expected_time_data = np.array(
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
        )
        expected = expected.assign_coords(
            {
                "time": xr.DataArray(
                    name="time",
                    data=expected_time_data,
                    coords={"time": expected_time_data},
                    dims="time",
                    attrs={
                        "long_name": "time",
                        "standard_name": "time",
                        "axis": "T",
                        "bounds": "time_bnds",
                    },
                )
            }
        )
        # Update time bounds with centered time coordinates.
        time_bounds = ds.time_bnds.copy()
        time_bounds["time"] = expected.time
        expected["time_bnds"] = time_bounds

        result = ds.temporal.center_times(ds)
        assert result.identical(expected)


class Test_SetObjAttrs:
    # NOTE: Testing this private method directly instead of through the public
    # methods because it eliminates redundancy.
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = generate_dataset(cf_compliant=True, has_bounds=True)

    def test_raises_error_if_operation_is_not_supported(self):
        with pytest.raises(ValueError):
            self.ds.temporal._set_obj_attrs(
                "unsupported",
                freq="season",
                weighted=True,
                center_times=True,
                season_config={
                    "dec_mode": "DJF",
                    "drop_incomplete_djf": False,
                    "custom_seasons": None,
                },
            )

    def test_raises_error_if_freq_arg_is_not_supported_by_operation(self):
        ds = self.ds.copy()

        with pytest.raises(ValueError):
            ds.temporal._set_obj_attrs(
                "group_average",
                freq="unsupported",
                weighted=True,
                center_times=True,
                season_config={
                    "dec_mode": "DJF",
                    "drop_incomplete_djf": False,
                    "custom_seasons": None,
                },
            )
        with pytest.raises(ValueError):
            ds.temporal._set_obj_attrs(
                "climatology",
                freq="unsupported",
                weighted=True,
                center_times=True,
                season_config={
                    "dec_mode": "DJF",
                    "drop_incomplete_djf": False,
                    "custom_seasons": None,
                },
            )
        with pytest.raises(ValueError):
            ds.temporal._set_obj_attrs(
                "departures",
                freq="unsupported",
                weighted=True,
                center_times=True,
                season_config={
                    "dec_mode": "DJF",
                    "drop_incomplete_djf": False,
                    "custom_seasons": None,
                },
            )

    def test_does_not_raise_error_if_freq_arg_is_supported_by_operation(self):
        ds = self.ds.copy()
        climatology_freqs = ["season", "month", "day"]
        departure_freqs = ["season", "month", "day"]
        time_series_freqs = ["year", "season", "month", "day", "hour"]

        for freq in time_series_freqs:
            ds.temporal._set_obj_attrs(
                "group_average",
                freq=freq,
                weighted=True,
                center_times=True,
                season_config={
                    "dec_mode": "DJF",
                    "drop_incomplete_djf": False,
                    "custom_seasons": None,
                },
            )

        for freq in climatology_freqs:
            ds.temporal._set_obj_attrs(
                "climatology",
                freq=freq,
                weighted=True,
                center_times=True,
                season_config={
                    "dec_mode": "DJF",
                    "drop_incomplete_djf": False,
                    "custom_seasons": None,
                },
            )

        for freq in departure_freqs:
            ds.temporal._set_obj_attrs(
                "departures",
                freq=freq,
                weighted=True,
                center_times=True,
                season_config={
                    "dec_mode": "DJF",
                    "drop_incomplete_djf": False,
                    "custom_seasons": None,
                },
            )

    def test_raises_error_if_season_config_key_is_not_supported(self):
        with pytest.raises(KeyError):
            self.ds.temporal._set_obj_attrs(
                "climatology",
                freq="season",
                weighted=True,
                center_times=True,
                season_config={
                    "not_supported": "invalid",
                },
            )

    def test_raises_error_if_december_mode_is_not_supported(self):
        with pytest.raises(ValueError):
            self.ds.temporal._set_obj_attrs(
                "climatology",
                freq="season",
                weighted=True,
                center_times=True,
                season_config={
                    "dec_mode": "unsupported",
                    "drop_incomplete_djf": False,
                    "custom_seasons": None,
                },
            )


class Test_GetWeights:
    # NOTE: Testing this private method directly instead of through the public
    # methods because there is potential for this method to become public.
    class TestWeightsForAverageMode:
        @pytest.fixture(autouse=True)
        def setup(self):
            self.ds: xr.Dataset = generate_dataset(cf_compliant=True, has_bounds=True)

        def test_weights_for_yearly_averages(self):
            ds = self.ds.copy()

            # Set object attrs required to test the method.
            ds.temporal._time_bounds = ds.time_bnds.copy()
            ds.temporal._mode = "average"
            ds.temporal._freq = "year"
            ds.temporal._weighted = "True"
            ds.temporal._labeled_time = xr.DataArray(
                name="year",
                data=np.array(
                    [
                        "2000-01-01T00:00:00.000000000",
                        "2000-01-01T00:00:00.000000000",
                        "2000-01-01T00:00:00.000000000",
                        "2000-01-01T00:00:00.000000000",
                        "2000-01-01T00:00:00.000000000",
                        "2000-01-01T00:00:00.000000000",
                        "2000-01-01T00:00:00.000000000",
                        "2000-01-01T00:00:00.000000000",
                        "2000-01-01T00:00:00.000000000",
                        "2000-01-01T00:00:00.000000000",
                        "2000-01-01T00:00:00.000000000",
                        "2000-01-01T00:00:00.000000000",
                        "2001-01-01T00:00:00.000000000",
                        "2001-01-01T00:00:00.000000000",
                        "2001-01-01T00:00:00.000000000",
                    ],
                    dtype="datetime64[ns]",
                ),
                coords={"time": ds.time},
                dims=["time"],
                attrs={
                    "axis": "T",
                    "long_name": "time",
                    "standard_name": "time",
                    "bounds": "time_bnds",
                },
            )

            # Compare result of the method against the expected.
            result = ds.temporal._get_weights()
            expected = np.array(
                [
                    0.08469945,
                    0.07923497,
                    0.08469945,
                    0.08196721,
                    0.08469945,
                    0.08196721,
                    0.08469945,
                    0.08469945,
                    0.08196721,
                    0.08469945,
                    0.08196721,
                    0.08469945,
                    0.34444444,
                    0.31111111,
                    0.34444444,
                ]
            )
            assert np.allclose(result, expected)

        def test_weights_for_monthly_averages(self):
            ds = self.ds.copy()

            # Set object attrs required to test the method.
            ds.temporal._time_bounds = ds.time_bnds.copy()
            ds.temporal._mode = "average"
            ds.temporal._freq = "month"
            ds.temporal._weighted = "True"
            ds.temporal._labeled_time = xr.DataArray(
                name="month",
                data=np.array(
                    [
                        cftime.datetime(1, 1, 1),
                        cftime.datetime(1, 2, 1),
                        cftime.datetime(1, 3, 1),
                        cftime.datetime(1, 4, 1),
                        cftime.datetime(1, 5, 1),
                        cftime.datetime(1, 6, 1),
                        cftime.datetime(1, 7, 1),
                        cftime.datetime(1, 8, 1),
                        cftime.datetime(1, 9, 1),
                        cftime.datetime(1, 10, 1),
                        cftime.datetime(1, 11, 1),
                        cftime.datetime(1, 12, 1),
                        cftime.datetime(1, 1, 1),
                        cftime.datetime(1, 2, 1),
                        cftime.datetime(1, 12, 1),
                    ],
                ),
                coords={"time": ds.time},
                attrs={
                    "axis": "T",
                    "long_name": "time",
                    "standard_name": "time",
                    "bounds": "time_bnds",
                },
            )

            # Compare result of the method against the expected.
            result = ds.temporal._get_weights()
            expected = np.array(
                [
                    0.5,
                    0.50877193,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    0.5,
                    0.5,
                    # This is a leap year month, so the weight is less.
                    0.49122807,
                    0.5,
                ]
            )
            assert np.allclose(result, expected)

    class TestWeightsForGroupAverageMode:
        @pytest.fixture(autouse=True)
        def setup(self):
            self.ds: xr.Dataset = generate_dataset(cf_compliant=True, has_bounds=True)

        def test_weights_for_yearly_averages(self):
            ds = self.ds.copy()

            # Set object attrs required to test the method.
            ds.temporal._time_bounds = ds.time_bnds.copy()
            ds.temporal._mode = "group_average"
            ds.temporal._freq = "year"
            ds.temporal._weighted = "True"
            ds.temporal._labeled_time = xr.DataArray(
                name="year",
                data=np.array(
                    [
                        "2000-01-01T00:00:00.000000000",
                        "2000-01-01T00:00:00.000000000",
                        "2000-01-01T00:00:00.000000000",
                        "2000-01-01T00:00:00.000000000",
                        "2000-01-01T00:00:00.000000000",
                        "2000-01-01T00:00:00.000000000",
                        "2000-01-01T00:00:00.000000000",
                        "2000-01-01T00:00:00.000000000",
                        "2000-01-01T00:00:00.000000000",
                        "2000-01-01T00:00:00.000000000",
                        "2000-01-01T00:00:00.000000000",
                        "2000-01-01T00:00:00.000000000",
                        "2001-01-01T00:00:00.000000000",
                        "2001-01-01T00:00:00.000000000",
                        "2001-01-01T00:00:00.000000000",
                    ],
                    dtype="datetime64[ns]",
                ),
                coords={"time": ds.time},
                dims=["time"],
                attrs={
                    "axis": "T",
                    "long_name": "time",
                    "standard_name": "time",
                    "bounds": "time_bnds",
                },
            )

            # Compare result of the method against the expected.
            result = ds.temporal._get_weights()
            expected = np.array(
                [
                    0.08469945,
                    0.07923497,
                    0.08469945,
                    0.08196721,
                    0.08469945,
                    0.08196721,
                    0.08469945,
                    0.08469945,
                    0.08196721,
                    0.08469945,
                    0.08196721,
                    0.08469945,
                    0.34444444,
                    0.31111111,
                    0.34444444,
                ]
            )
            assert np.allclose(result, expected)

        def test_weights_for_monthly_averages(self):
            ds = self.ds.copy()

            # Set object attrs required to test the method.
            ds.temporal._time_bounds = ds.time_bnds.copy()
            ds.temporal._mode = "group_average"
            ds.temporal._freq = "month"
            ds.temporal._weighted = "True"
            ds.temporal._labeled_time = xr.DataArray(
                name="year_month",
                data=np.array(
                    [
                        "2000-01-01T00:00:00.000000000",
                        "2000-02-01T00:00:00.000000000",
                        "2000-03-01T00:00:00.000000000",
                        "2000-04-01T00:00:00.000000000",
                        "2000-05-01T00:00:00.000000000",
                        "2000-06-01T00:00:00.000000000",
                        "2000-07-01T00:00:00.000000000",
                        "2000-08-01T00:00:00.000000000",
                        "2000-09-01T00:00:00.000000000",
                        "2000-10-01T00:00:00.000000000",
                        "2000-11-01T00:00:00.000000000",
                        "2000-12-01T00:00:00.000000000",
                        "2001-01-01T00:00:00.000000000",
                        "2001-02-01T00:00:00.000000000",
                        "2001-12-01T00:00:00.000000000",
                    ],
                    dtype="datetime64[ns]",
                ),
                coords={"time": ds.time},
                dims=["time"],
                attrs={
                    "axis": "T",
                    "long_name": "time",
                    "standard_name": "time",
                    "bounds": "time_bnds",
                },
            )

            # Compare result of the method against the expected.
            result = ds.temporal._get_weights()
            expected = np.ones(15)
            assert np.allclose(result, expected)

        def test_weights_for_seasonal_averages_with_DJF_and_drop_incomplete_seasons(
            self,
        ):
            ds = self.ds.copy()

            # Replace time and time bounds with incomplete seasons removed
            ds = ds.drop_dims("time")
            ds.coords["time"] = xr.DataArray(
                data=np.array(
                    [
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
            ds["ts"] = xr.DataArray(
                name="ts",
                data=np.ones((12, 4, 4)),
                coords={"time": ds.time, "lat": ds.lat, "lon": ds.lon},
                dims=["time", "lat", "lon"],
            )
            ds["time_bnds"] = xr.DataArray(
                name="time_bnds",
                data=np.array(
                    [
                        [
                            "2000-03-01T00:00:00.000000000",
                            "2000-04-01T00:00:00.000000000",
                        ],
                        [
                            "2000-04-01T00:00:00.000000000",
                            "2000-05-01T00:00:00.000000000",
                        ],
                        [
                            "2000-05-01T00:00:00.000000000",
                            "2000-06-01T00:00:00.000000000",
                        ],
                        [
                            "2000-06-01T00:00:00.000000000",
                            "2000-07-01T00:00:00.000000000",
                        ],
                        [
                            "2000-07-01T00:00:00.000000000",
                            "2000-08-01T00:00:00.000000000",
                        ],
                        [
                            "2000-08-01T00:00:00.000000000",
                            "2000-09-01T00:00:00.000000000",
                        ],
                        [
                            "2000-09-01T00:00:00.000000000",
                            "2000-10-01T00:00:00.000000000",
                        ],
                        [
                            "2000-10-01T00:00:00.000000000",
                            "2000-11-01T00:00:00.000000000",
                        ],
                        [
                            "2000-11-01T00:00:00.000000000",
                            "2000-12-01T00:00:00.000000000",
                        ],
                        [
                            "2000-12-01T00:00:00.000000000",
                            "2001-01-01T00:00:00.000000000",
                        ],
                        [
                            "2001-01-01T00:00:00.000000000",
                            "2001-02-01T00:00:00.000000000",
                        ],
                        [
                            "2001-02-01T00:00:00.000000000",
                            "2001-03-01T00:00:00.000000000",
                        ],
                    ],
                    dtype="datetime64[ns]",
                ),
                coords={"time": ds.time},
                dims=["time", "bnds"],
                attrs={
                    "is_generated": "True",
                },
            )

            # Set object attrs required to test the method.
            ds.temporal._time_bounds = ds.time_bnds.copy()
            ds.temporal._mode = "group_average"
            ds.temporal._freq = "season"
            ds.temporal._weighted = "True"
            ds.temporal._season_config = {"dec_mode": "DJF"}
            ds.temporal._labeled_time = xr.DataArray(
                name="year_season",
                data=np.array(
                    [
                        # 2000 MAM
                        "2000-04-01T00:00:00.000000000",
                        "2000-04-01T00:00:00.000000000",
                        "2000-04-01T00:00:00.000000000",
                        # 2000 JJA
                        "2000-07-01T00:00:00.000000000",
                        "2000-07-01T00:00:00.000000000",
                        "2000-07-01T00:00:00.000000000",
                        # 2000 SON
                        "2000-10-01T00:00:00.000000000",
                        "2000-10-01T00:00:00.000000000",
                        "2000-10-01T00:00:00.000000000",
                        # 2001 DJF
                        "2001-01-01T00:00:00.000000000",
                        "2001-01-01T00:00:00.000000000",
                        "2001-01-01T00:00:00.000000000",
                    ],
                    dtype="datetime64[ns]",
                ),
                coords={"time": ds.time},
                dims=["time"],
                attrs={
                    "axis": "T",
                    "long_name": "time",
                    "standard_name": "time",
                    "bounds": "time_bnds",
                },
            )

            # Compare result of the method against the expected.
            result = ds.temporal._get_weights()
            expected = np.array(
                [
                    0.33695652,
                    0.32608696,
                    0.33695652,
                    0.32608696,
                    0.33695652,
                    0.33695652,
                    0.32967033,
                    0.34065934,
                    0.32967033,
                    0.34444444,
                    0.34444444,
                    0.31111111,
                ]
            )
            assert np.allclose(result, expected, equal_nan=True)

        def test_weights_for_seasonal_averages_with_JFD(self):
            ds = self.ds.copy()

            # Set object attrs required to test the method.
            ds.temporal._time_bounds = ds.time_bnds.copy()
            ds.temporal._mode = "group_average"
            ds.temporal._freq = "season"
            ds.temporal._weighted = "True"
            ds.temporal._season_config = {"dec_mode": "JDF"}
            ds.temporal._labeled_time = xr.DataArray(
                name="year_season",
                data=np.array(
                    [
                        # 2000 JFD
                        "2000-01-01T00:00:00.000000000",
                        "2000-01-01T00:00:00.000000000",
                        # 2000 MAM
                        "2000-04-01T00:00:00.000000000",
                        "2000-04-01T00:00:00.000000000",
                        "2000-04-01T00:00:00.000000000",
                        # 2000 JJA
                        "2000-07-01T00:00:00.000000000",
                        "2000-07-01T00:00:00.000000000",
                        "2000-07-01T00:00:00.000000000",
                        # 2000 SON
                        "2000-10-01T00:00:00.000000000",
                        "2000-10-01T00:00:00.000000000",
                        "2000-10-01T00:00:00.000000000",
                        # 2000 JFD
                        "2000-01-01T00:00:00.000000000",
                        # 2001 JFD
                        "2001-01-01T00:00:00.000000000",
                        "2001-01-01T00:00:00.000000000",
                        "2001-01-01T00:00:00.000000000",
                    ],
                    dtype="datetime64[ns]",
                ),
                coords={"time": ds.time},
                dims=["time"],
                attrs={
                    "axis": "T",
                    "long_name": "time",
                    "standard_name": "time",
                    "bounds": "time_bnds",
                },
            )

            # Compare result of the method against the expected.
            result = ds.temporal._get_weights()
            expected = np.array(
                [
                    0.34065934,
                    0.31868132,
                    0.33695652,
                    0.32608696,
                    0.33695652,
                    0.32608696,
                    0.33695652,
                    0.33695652,
                    0.32967033,
                    0.34065934,
                    0.32967033,
                    0.34065934,
                    0.34444444,
                    0.31111111,
                    0.34444444,
                ]
            )
            assert np.allclose(result, expected)

        def test_custom_season_time_series_weights(self):
            ds = self.ds.copy()

            # Set object attrs required to test the method.
            ds.temporal._time_bounds = ds.time_bnds.copy()
            ds.temporal._mode = "group_average"
            ds.temporal._freq = "season"
            ds.temporal._weighted = "True"
            ds.temporal._season_config = {
                "custom_seasons": {
                    "JanFebMar": ["Jan", "Feb", "Mar"],
                    "AprMayJun": ["Apr", "May", "Jun"],
                    "JulAugSep": ["Jul", "Aug", "Sep"],
                    "OctNovDec": ["Oct", "Nov", "Dec"],
                }
            }

            ds.temporal._labeled_time = xr.DataArray(
                name="year_season",
                data=np.array(
                    [
                        # 2000 JanFebMar
                        "2000-02-01T00:00:00.000000000",
                        "2000-02-01T00:00:00.000000000",
                        "2000-02-01T00:00:00.000000000",
                        # 2000 AprMayJun
                        "2000-05-01T00:00:00.000000000",
                        "2000-05-01T00:00:00.000000000",
                        "2000-05-01T00:00:00.000000000",
                        # 2000 JunAugSep
                        "2000-08-01T00:00:00.000000000",
                        "2000-08-01T00:00:00.000000000",
                        "2000-08-01T00:00:00.000000000",
                        # 2000 OctNovDec
                        "2000-11-01T00:00:00.000000000",
                        "2000-11-01T00:00:00.000000000",
                        "2000-11-01T00:00:00.000000000",
                        # 2001 JanFebMar
                        "2001-02-01T00:00:00.000000000",
                        "2001-02-01T00:00:00.000000000",
                        "2002-02-01T00:00:00.000000000",
                    ],
                    dtype="datetime64[ns]",
                ),
                coords={"time": ds.time},
                dims=["time"],
                attrs={
                    "axis": "T",
                    "long_name": "time",
                    "standard_name": "time",
                    "bounds": "time_bnds",
                },
            )

            # Compare result of the method against the expected.
            result = ds.temporal._get_weights()
            expected = np.array(
                [
                    0.34065934,
                    0.31868132,
                    0.34065934,
                    0.32967033,
                    0.34065934,
                    0.32967033,
                    0.33695652,
                    0.33695652,
                    0.32608696,
                    0.33695652,
                    0.32608696,
                    0.33695652,
                    0.52542373,
                    0.47457627,
                    1.0,
                ]
            )
            assert np.allclose(result, expected)

        def test_weights_for_daily_averages(self):
            ds = self.ds.copy()

            # Set object attrs required to test the method.
            ds.temporal._time_bounds = ds.time_bnds.copy()
            ds.temporal._mode = "group_average"
            ds.temporal._freq = "day"
            ds.temporal._weighted = "True"
            ds.temporal._labeled_time = xr.DataArray(
                name="year_month_day",
                data=np.array(
                    [
                        "2000-01-16T00:00:00.000000000",
                        "2000-02-15T00:00:00.000000000",
                        "2000-03-16T00:00:00.000000000",
                        "2000-04-16T00:00:00.000000000",
                        "2000-05-16T00:00:00.000000000",
                        "2000-06-16T00:00:00.000000000",
                        "2000-07-16T00:00:00.000000000",
                        "2000-08-16T00:00:00.000000000",
                        "2000-09-16T00:00:00.000000000",
                        "2000-10-16T00:00:00.000000000",
                        "2000-11-16T00:00:00.000000000",
                        "2000-12-16T00:00:00.000000000",
                        "2001-01-16T00:00:00.000000000",
                        "2001-02-15T00:00:00.000000000",
                        "2001-12-16T00:00:00.000000000",
                    ],
                    dtype="datetime64[ns]",
                ),
                coords={"time": ds.time},
                dims=["time"],
                attrs={
                    "axis": "T",
                    "long_name": "time",
                    "standard_name": "time",
                    "bounds": "time_bnds",
                },
            )

            # Compare result of the method against the expected.
            result = ds.temporal._get_weights()
            expected = np.ones(15)
            assert np.allclose(result, expected)

        def test_weights_for_hourly_averages(self):
            ds = self.ds.copy()

            # Set object attrs required to test the method.
            ds.temporal._time_bounds = ds.time_bnds.copy()
            ds.temporal._mode = "group_average"
            ds.temporal._freq = "hour"
            ds.temporal._weighted = "True"
            ds.temporal._season_config = {"dec_mode": "JDF"}
            ds.temporal._labeled_time = xr.DataArray(
                name="year_month_day_hour",
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
                coords={"time": ds.time},
                dims=["time"],
                attrs={
                    "axis": "T",
                    "long_name": "time",
                    "standard_name": "time",
                    "bounds": "time_bnds",
                },
            )

            # Compare result of the method against the expected.
            result = ds.temporal._get_weights()
            expected = np.ones(15)
            assert np.allclose(result, expected)

    class TestWeightsForClimatologyMode:
        @pytest.fixture(autouse=True)
        def setup(self):
            self.ds: xr.Dataset = generate_dataset(cf_compliant=True, has_bounds=True)

        def test_weights_for_seasonal_climatology_with_DJF(self):
            ds = self.ds.copy()

            # Replace time and time bounds with incomplete seasons removed
            ds = ds.drop_dims("time")
            ds.coords["time"] = xr.DataArray(
                data=np.array(
                    [
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
            ds["ts"] = xr.DataArray(
                name="ts",
                data=np.ones((12, 4, 4)),
                coords={"time": ds.time, "lat": ds.lat, "lon": ds.lon},
                dims=["time", "lat", "lon"],
            )
            ds["time_bnds"] = xr.DataArray(
                name="time_bnds",
                data=np.array(
                    [
                        [
                            "2000-03-01T00:00:00.000000000",
                            "2000-04-01T00:00:00.000000000",
                        ],
                        [
                            "2000-04-01T00:00:00.000000000",
                            "2000-05-01T00:00:00.000000000",
                        ],
                        [
                            "2000-05-01T00:00:00.000000000",
                            "2000-06-01T00:00:00.000000000",
                        ],
                        [
                            "2000-06-01T00:00:00.000000000",
                            "2000-07-01T00:00:00.000000000",
                        ],
                        [
                            "2000-07-01T00:00:00.000000000",
                            "2000-08-01T00:00:00.000000000",
                        ],
                        [
                            "2000-08-01T00:00:00.000000000",
                            "2000-09-01T00:00:00.000000000",
                        ],
                        [
                            "2000-09-01T00:00:00.000000000",
                            "2000-10-01T00:00:00.000000000",
                        ],
                        [
                            "2000-10-01T00:00:00.000000000",
                            "2000-11-01T00:00:00.000000000",
                        ],
                        [
                            "2000-11-01T00:00:00.000000000",
                            "2000-12-01T00:00:00.000000000",
                        ],
                        [
                            "2000-12-01T00:00:00.000000000",
                            "2001-01-01T00:00:00.000000000",
                        ],
                        [
                            "2001-01-01T00:00:00.000000000",
                            "2001-02-01T00:00:00.000000000",
                        ],
                        [
                            "2001-02-01T00:00:00.000000000",
                            "2001-03-01T00:00:00.000000000",
                        ],
                    ],
                    dtype="datetime64[ns]",
                ),
                coords={"time": ds.time},
                dims=["time", "bnds"],
                attrs={
                    "is_generated": "True",
                },
            )

            # Set object attrs required to test the method.
            ds.temporal._time_bounds = ds.time_bnds.copy()
            ds.temporal._mode = "climatology"
            ds.temporal._freq = "season"
            ds.temporal._weighted = "True"
            ds.temporal._season_config = {"dec_mode": "DJF"}
            ds.temporal._labeled_time = xr.DataArray(
                name="season",
                data=np.array(
                    [
                        cftime.datetime(1, 4, 1),
                        cftime.datetime(1, 4, 1),
                        cftime.datetime(1, 4, 1),
                        cftime.datetime(1, 7, 1),
                        cftime.datetime(1, 7, 1),
                        cftime.datetime(1, 7, 1),
                        cftime.datetime(1, 10, 1),
                        cftime.datetime(1, 10, 1),
                        cftime.datetime(1, 10, 1),
                        cftime.datetime(1, 1, 1),
                        cftime.datetime(1, 1, 1),
                        cftime.datetime(1, 1, 1),
                    ],
                ),
                coords={"time": ds.time},
                dims=["time"],
                attrs={
                    "axis": "T",
                    "long_name": "time",
                    "standard_name": "time",
                    "bounds": "time_bnds",
                },
            )
            # Compare result of the method against the expected.
            result = ds.temporal._get_weights()
            expected = np.array(
                [
                    0.33695652,
                    0.32608696,
                    0.33695652,
                    0.32608696,
                    0.33695652,
                    0.33695652,
                    0.32967033,
                    0.34065934,
                    0.32967033,
                    0.34444444,
                    0.34444444,
                    0.31111111,
                ]
            )

            assert np.allclose(result, expected, equal_nan=True)

        def test_weights_for_seasonal_climatology_with_JFD(self):
            ds = self.ds.copy()

            # Set object attrs required to test the method.
            ds.temporal._time_bounds = ds.time_bnds.copy()
            ds.temporal._mode = "climatology"
            ds.temporal._freq = "season"
            ds.temporal._weighted = "True"
            ds.temporal._season_config = {"dec_mode": "JDF"}
            ds.temporal._labeled_time = xr.DataArray(
                name="season",
                data=np.array(
                    [
                        cftime.datetime(1, 1, 1),
                        cftime.datetime(1, 1, 1),
                        cftime.datetime(1, 4, 1),
                        cftime.datetime(1, 4, 1),
                        cftime.datetime(1, 4, 1),
                        cftime.datetime(1, 7, 1),
                        cftime.datetime(1, 7, 1),
                        cftime.datetime(1, 7, 1),
                        cftime.datetime(1, 10, 1),
                        cftime.datetime(1, 10, 1),
                        cftime.datetime(1, 10, 1),
                        cftime.datetime(1, 1, 1),
                        cftime.datetime(1, 1, 1),
                        cftime.datetime(1, 1, 1),
                        cftime.datetime(1, 1, 1),
                    ],
                ),
                coords={"time": ds.time},
                dims=["time"],
                attrs={
                    "axis": "T",
                    "long_name": "time",
                    "standard_name": "time",
                    "bounds": "time_bnds",
                },
            )

            # Compare result of the method against the expected.
            result = ds.temporal._get_weights()
            expected = np.array(
                [
                    [
                        0.17127072,
                        0.16022099,
                        0.33695652,
                        0.32608696,
                        0.33695652,
                        0.32608696,
                        0.33695652,
                        0.33695652,
                        0.32967033,
                        0.34065934,
                        0.32967033,
                        0.17127072,
                        0.17127072,
                        0.15469613,
                        0.17127072,
                    ]
                ]
            )
            assert np.allclose(result, expected, equal_nan=True)

        def test_weights_for_annual_climatology(self):
            ds = self.ds.copy()

            # Set object attrs required to test the method.
            ds.temporal._time_bounds = ds.time_bnds.copy()
            ds.temporal._mode = "climatology"
            ds.temporal._freq = "month"
            ds.temporal._weighted = "True"
            ds.temporal._labeled_time = xr.DataArray(
                name="month",
                data=np.array(
                    [
                        cftime.datetime(1, 1, 1),
                        cftime.datetime(1, 2, 1),
                        cftime.datetime(1, 3, 1),
                        cftime.datetime(1, 4, 1),
                        cftime.datetime(1, 5, 1),
                        cftime.datetime(1, 6, 1),
                        cftime.datetime(1, 7, 1),
                        cftime.datetime(1, 8, 1),
                        cftime.datetime(1, 9, 1),
                        cftime.datetime(1, 10, 1),
                        cftime.datetime(1, 11, 1),
                        cftime.datetime(1, 12, 1),
                        cftime.datetime(1, 1, 1),
                        cftime.datetime(1, 2, 1),
                        cftime.datetime(1, 12, 1),
                    ],
                ),
                coords={"time": ds.time},
                attrs={
                    "axis": "T",
                    "long_name": "time",
                    "standard_name": "time",
                    "bounds": "time_bnds",
                },
            )

            # Compare result of the method against the expected.
            result = ds.temporal._get_weights()
            expected = np.array(
                [
                    [
                        0.5,
                        0.50877193,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        0.5,
                        0.5,
                        0.49122807,
                        0.5,
                    ]
                ]
            )
            assert np.allclose(result, expected)

        def test_weights_for_daily_climatology(self):
            ds = self.ds.copy()

            # Set object attrs required to test the method.
            ds.temporal._time_bounds = ds.time_bnds.copy()
            ds.temporal._mode = "climatology"
            ds.temporal._freq = "day"
            ds.temporal._weighted = "True"
            ds.temporal._labeled_time = xr.DataArray(
                name="month_day",
                data=np.array(
                    [
                        cftime.datetime(1, 1, 16),
                        cftime.datetime(1, 2, 15),
                        cftime.datetime(1, 3, 16),
                        cftime.datetime(1, 4, 16),
                        cftime.datetime(1, 5, 6),
                        cftime.datetime(1, 6, 16),
                        cftime.datetime(1, 7, 16),
                        cftime.datetime(1, 8, 16),
                        cftime.datetime(1, 9, 16),
                        cftime.datetime(1, 10, 16),
                        cftime.datetime(1, 11, 16),
                        cftime.datetime(1, 12, 16),
                        cftime.datetime(1, 1, 16),
                        cftime.datetime(1, 2, 15),
                        cftime.datetime(1, 12, 16),
                    ],
                ),
                coords={"time": ds.time},
                attrs={
                    "axis": "T",
                    "long_name": "time",
                    "standard_name": "time",
                    "bounds": "time_bnds",
                },
            )

            # Compare result of the method against the expected.
            result = ds.temporal._get_weights()
            expected = np.array(
                [
                    0.5,
                    0.50877193,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    0.5,
                    0.5,
                    0.49122807,
                    0.5,
                ]
            )
            assert np.allclose(result, expected)
