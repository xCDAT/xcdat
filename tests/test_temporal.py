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
    def test_averages_for_yearly_time_series(self):
        ds = xr.Dataset(
            coords={
                "lat": [-90],
                "lon": [0],
                "time": xr.DataArray(
                    data=np.array(
                        [
                            "2000-01-01T00:00:00.000000000",
                            "2001-01-01T00:00:00.000000000",
                            "2002-01-01T00:00:00.000000000",
                            "2003-01-01T00:00:00.000000000",
                            "2004-01-01T00:00:00.000000000",
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
            }
        )
        ds["time_bnds"] = xr.DataArray(
            name="time_bnds",
            data=np.array(
                [
                    ["2000-01-01T00:00:00.000000000", "2001-01-01T00:00:00.000000000"],
                    ["2001-01-01T00:00:00.000000000", "2002-01-01T00:00:00.000000000"],
                    ["2002-01-01T00:00:00.000000000", "2003-01-01T00:00:00.000000000"],
                    ["2003-01-01T00:00:00.000000000", "2004-01-01T00:00:00.000000000"],
                    ["2004-01-01T00:00:00.000000000", "2005-01-01T00:00:00.000000000"],
                ],
                dtype="datetime64[ns]",
            ),
            coords={"time": ds.time},
            dims=["time", "bnds"],
            attrs={"xcdat_bounds": "True"},
        )
        ds["ts"] = xr.DataArray(
            data=np.array([[[2]], [[np.nan]], [[1]], [[1]], [[2]]]),
            coords={"lat": ds.lat, "lon": ds.lon, "time": ds.time},
            dims=["time", "lat", "lon"],
        )

        # Test averages weighted by year
        result = ds.temporal.average("ts")
        expected = ds.copy()
        expected = expected.drop_dims("time")
        expected["ts"] = xr.DataArray(
            data=np.array([[1.5]]),
            coords={"lat": expected.lat, "lon": expected.lon},
            dims=["lat", "lon"],
            attrs={
                "operation": "temporal_avg",
                "mode": "average",
                "freq": "year",
                "weighted": "True",
            },
        )

        assert result.identical(expected)

        # Test unweighted averages
        result = ds.temporal.average("ts", weighted=False)
        expected = ds.copy()
        expected = expected.drop_dims("time")
        expected["ts"] = xr.DataArray(
            data=np.array([[1.5]]),
            coords={"lat": expected.lat, "lon": expected.lon},
            dims=["lat", "lon"],
            attrs={
                "operation": "temporal_avg",
                "mode": "average",
                "freq": "year",
                "weighted": "False",
            },
        )

        assert result.identical(expected)

    def test_averages_for_monthly_time_series(self):
        # Set up dataset
        ds = xr.Dataset(
            coords={
                "lat": [-90],
                "lon": [0],
                "time": xr.DataArray(
                    data=np.array(
                        [
                            "2000-01-01T00:00:00.000000000",
                            "2000-02-01T00:00:00.000000000",
                            "2000-03-01T00:00:00.000000000",
                            "2000-04-01T00:00:00.000000000",
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
            }
        )
        ds["time_bnds"] = xr.DataArray(
            name="time_bnds",
            data=np.array(
                [
                    ["2000-01-01T00:00:00.000000000", "2000-02-01T00:00:00.000000000"],
                    ["2000-02-01T00:00:00.000000000", "2000-03-01T00:00:00.000000000"],
                    ["2000-03-01T00:00:00.000000000", "2000-04-01T00:00:00.000000000"],
                    ["2000-04-01T00:00:00.000000000", "2000-05-01T00:00:00.000000000"],
                    ["2001-01-01T00:00:00.000000000", "2000-03-01T00:00:00.000000000"],
                ],
                dtype="datetime64[ns]",
            ),
            coords={"time": ds.time},
            dims=["time", "bnds"],
            attrs={"xcdat_bounds": "True"},
        )
        ds["ts"] = xr.DataArray(
            data=np.array([[[2]], [[np.nan]], [[1]], [[1]], [[1]]]),
            coords={"lat": ds.lat, "lon": ds.lon, "time": ds.time},
            dims=["time", "lat", "lon"],
        )

        # Test averages weighted by month
        result = ds.temporal.average("ts")
        expected = ds.copy()
        expected = expected.drop_dims("time")
        expected["ts"] = xr.DataArray(
            data=np.array([[1.24362357]]),
            coords={"lat": expected.lat, "lon": expected.lon},
            dims=["lat", "lon"],
            attrs={
                "operation": "temporal_avg",
                "mode": "average",
                "freq": "month",
                "weighted": "True",
            },
        )

        xr.testing.assert_allclose(result, expected)

        # Test unweighted averages
        result = ds.temporal.average("ts", weighted=False)
        expected = ds.copy()
        expected = expected.drop_dims("time")
        expected["ts"] = xr.DataArray(
            data=np.array([[1.25]]),
            coords={"lat": expected.lat, "lon": expected.lon},
            dims=["lat", "lon"],
            attrs={
                "operation": "temporal_avg",
                "mode": "average",
                "freq": "month",
                "weighted": "False",
            },
        )
        xr.testing.assert_allclose(result, expected)

    def test_averages_for_daily_time_series(self):
        ds = xr.Dataset(
            coords={
                "lat": [-90],
                "lon": [0],
                "time": xr.DataArray(
                    data=np.array(
                        [
                            "2000-01-01T00:00:00.000000000",
                            "2000-01-02T00:00:00.000000000",
                            "2000-01-03T00:00:00.000000000",
                            "2000-01-04T00:00:00.000000000",
                            "2000-01-05T00:00:00.000000000",
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
            }
        )
        ds["time_bnds"] = xr.DataArray(
            name="time_bnds",
            data=np.array(
                [
                    ["2000-01-01T00:00:00.000000000", "2000-01-02T00:00:00.000000000"],
                    ["2000-01-02T00:00:00.000000000", "2000-01-03T00:00:00.000000000"],
                    ["2000-01-03T00:00:00.000000000", "2000-01-04T00:00:00.000000000"],
                    ["2000-01-04T00:00:00.000000000", "2000-01-05T00:00:00.000000000"],
                    ["2000-01-05T00:00:00.000000000", "2000-01-06T00:00:00.000000000"],
                ],
                dtype="datetime64[ns]",
            ),
            coords={"time": ds.time},
            dims=["time", "bnds"],
            attrs={"xcdat_bounds": "True"},
        )
        ds["ts"] = xr.DataArray(
            data=np.array([[[2]], [[1]], [[1]], [[1]], [[1]]]),
            coords={"lat": ds.lat, "lon": ds.lon, "time": ds.time},
            dims=["time", "lat", "lon"],
        )

        # Test averages weighted by day
        result = ds.temporal.average("ts")
        expected = ds.copy()
        expected = expected.drop_dims("time")
        expected["ts"] = xr.DataArray(
            data=np.array([[1.2]]),
            coords={"lat": expected.lat, "lon": expected.lon},
            dims=["lat", "lon"],
            attrs={
                "operation": "temporal_avg",
                "mode": "average",
                "freq": "day",
                "weighted": "True",
            },
        )

        assert result.identical(expected)

        # Test unweighted averages
        result = ds.temporal.average("ts", weighted=False)
        expected = ds.copy()
        expected = expected.drop_dims("time")
        expected["ts"] = xr.DataArray(
            data=np.array([[1.2]]),
            coords={"lat": expected.lat, "lon": expected.lon},
            dims=["lat", "lon"],
            attrs={
                "operation": "temporal_avg",
                "mode": "average",
                "freq": "day",
                "weighted": "False",
            },
        )
        assert result.identical(expected)

    def test_averages_for_hourly_time_series(self):
        ds = xr.Dataset(
            coords={
                "lat": [-90],
                "lon": [0],
                "time": xr.DataArray(
                    data=np.array(
                        [
                            "2000-01-01T01:00:00.000000000",
                            "2000-01-01T02:00:00.000000000",
                            "2000-01-01T03:00:00.000000000",
                            "2000-01-01T04:00:00.000000000",
                            "2000-01-01T05:00:00.000000000",
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
            }
        )
        ds["time_bnds"] = xr.DataArray(
            name="time_bnds",
            data=np.array(
                [
                    ["2000-01-01T01:00:00.000000000", "2000-01-01T02:00:00.000000000"],
                    ["2000-01-01T02:00:00.000000000", "2000-01-01T03:00:00.000000000"],
                    ["2000-01-01T03:00:00.000000000", "2000-01-01T04:00:00.000000000"],
                    ["2000-01-01T04:00:00.000000000", "2000-01-01T05:00:00.000000000"],
                    ["2000-01-01T05:00:00.000000000", "2000-01-01T06:00:00.000000000"],
                ],
                dtype="datetime64[ns]",
            ),
            coords={"time": ds.time},
            dims=["time", "bnds"],
            attrs={"xcdat_bounds": "True"},
        )
        ds["ts"] = xr.DataArray(
            data=np.array([[[2]], [[1]], [[1]], [[1]], [[1]]]),
            coords={"lat": ds.lat, "lon": ds.lon, "time": ds.time},
            dims=["time", "lat", "lon"],
        )

        # Test averages weighted by hour
        result = ds.temporal.average("ts")
        expected = ds.copy()
        expected = expected.drop_dims("time")
        expected["ts"] = xr.DataArray(
            data=np.array([[1.2]]),
            coords={"lat": expected.lat, "lon": expected.lon},
            dims=["lat", "lon"],
            attrs={
                "operation": "temporal_avg",
                "mode": "average",
                "freq": "hour",
                "weighted": "True",
            },
        )

        assert result.identical(expected)

        # Test unweighted averages
        result = ds.temporal.average("ts", weighted=False)
        expected = ds.copy()
        expected = expected.drop_dims("time")
        expected["ts"] = xr.DataArray(
            data=np.array([[1.2]]),
            coords={"lat": expected.lat, "lon": expected.lon},
            dims=["lat", "lon"],
            attrs={
                "operation": "temporal_avg",
                "mode": "average",
                "freq": "hour",
                "weighted": "False",
            },
        )

        assert result.identical(expected)


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
            attrs={"xcdat_bounds": "True"},
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
            },
        )

        assert result.identical(expected)

    def test_weighted_monthly_averages_with_masked_data(self):
        ds = self.ds.copy()
        ds["ts"] = xr.DataArray(
            data=np.array(
                [[[2.0]], [[np.nan]], [[1.0]], [[1.0]], [[2.0]]], dtype="float64"
            ),
            coords={"time": self.ds.time, "lat": self.ds.lat, "lon": self.ds.lon},
            dims=["time", "lat", "lon"],
        )

        result = ds.temporal.group_average("ts", "month")
        expected = ds.copy()
        expected = expected.drop_dims("time")
        expected["ts"] = xr.DataArray(
            name="ts",
            data=np.array([[[2.0]], [[0.0]], [[1.0]], [[1.0]], [[2.0]]]),
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
                "dec_mode": "DJF",
                "drop_incomplete_djf": "True",
            },
        )

        assert result.identical(expected)

    def test_weighted_seasonal_departures_with_DJF_and_keep_weights(self):
        ds = self.ds.copy()

        # Drop incomplete DJF seasons
        ds = ds.isel(time=slice(2, -1))

        # Compare result of the method against the expected.
        result = ds.temporal.departures(
            "ts",
            "season",
            weighted=True,
            keep_weights=True,
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
                "dec_mode": "DJF",
                "drop_incomplete_djf": "True",
            },
        )
        expected["time_wts"] = xr.DataArray(
            data=np.array(
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
            ),
            coords={"time": ds.time},
            dims=["time"],
        )

        xr.testing.assert_allclose(result, expected)
        assert result.ts.attrs == expected.ts.attrs

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
                "dec_mode": "JFD",
            },
        )

        assert result.identical(expected)


class Test_GetWeights:
    class TestWeightsForAverageMode:
        @pytest.fixture(autouse=True)
        def setup(self):
            self.ds: xr.Dataset = generate_dataset(cf_compliant=True, has_bounds=True)

        def test_weights_for_yearly_averages(self):
            ds = self.ds.copy()

            # Set object attrs required to test the method.
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
                    "bounds": "time_bnds",
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
                    "xcdat_bounds": "True",
                },
            )

            # Set object attrs required to test the method.
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
                    "bounds": "time_bnds",
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
                    "xcdat_bounds": "True",
                },
            )

            # Set object attrs required to test the method.
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


class Test_Averager:
    # NOTE: This private method is tested because it is more redundant to
    # test these cases for the public methods that call this private method.
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = generate_dataset(cf_compliant=True, has_bounds=True)

    def test_raises_error_with_incorrect_mode_arg(self):
        with pytest.raises(ValueError):
            self.ds.temporal._averager(
                "ts",
                "unsupported",
                freq="season",
                weighted=True,
                season_config={
                    "dec_mode": "DJF",
                    "drop_incomplete_djf": False,
                    "custom_seasons": None,
                },
            )

    def test_raises_error_if_freq_arg_is_not_supported_by_operation(self):
        ds = self.ds.copy()

        with pytest.raises(ValueError):
            ds.temporal._averager(
                "ts",
                "group_average",
                freq="unsupported",
                weighted=True,
                season_config={
                    "dec_mode": "DJF",
                    "drop_incomplete_djf": False,
                    "custom_seasons": None,
                },
            )
        with pytest.raises(ValueError):
            ds.temporal._averager(
                "ts",
                "climatology",
                freq="unsupported",
                weighted=True,
                season_config={
                    "dec_mode": "DJF",
                    "drop_incomplete_djf": False,
                    "custom_seasons": None,
                },
            )
        with pytest.raises(ValueError):
            ds.temporal._averager(
                "ts",
                "departures",
                freq="unsupported",
                weighted=True,
                season_config={
                    "dec_mode": "DJF",
                    "drop_incomplete_djf": False,
                    "custom_seasons": None,
                },
            )

    def test_raises_error_if_season_config_key_is_not_supported(self):
        with pytest.raises(KeyError):
            self.ds.temporal._averager(
                "ts",
                "climatology",
                freq="season",
                weighted=True,
                season_config={
                    "not_supported": "invalid",
                },
            )

    def test_raises_error_if_december_mode_is_not_supported(self):
        with pytest.raises(ValueError):
            self.ds.temporal._averager(
                "ts",
                "climatology",
                freq="season",
                weighted=True,
                season_config={
                    "dec_mode": "unsupported",
                    "drop_incomplete_djf": False,
                    "custom_seasons": None,
                },
            )
