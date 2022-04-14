from datetime import datetime

import cftime
import numpy as np
import pandas as pd
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
    def test_weighted_annual_avg(self):
        ds: xr.Dataset = generate_dataset(cf_compliant=True, has_bounds=True)

        result = ds.temporal.average("ts", "year")
        expected = ds.copy()
        expected = expected.drop_dims("time")
        time_new = xr.DataArray(
            data=np.array(
                ["2000-01-01T00:00:00.000000000", "2001-01-01T00:00:00.000000000"],
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
        )
        expected["ts"] = xr.DataArray(
            name="ts",
            data=np.ones((2, 4, 4)),
            coords={
                "lat": ds.lat,
                "lon": ds.lon,
                "time": time_new,
            },
            dims=["time", "lat", "lon"],
            attrs={
                "operation": "temporal_avg",
                "mode": "time_series",
                "freq": "year",
                "groupby": "year",
                "weighted": "True",
                "center_times": "False",
            },
        )

        # For some reason, there is a very small floating point difference
        # between both for ts so we have to use floating point comparison
        xr.testing.assert_allclose(result, expected)
        assert result.ts.attrs == expected.ts.attrs


class TestClimatology:
    def test_weighted_seasonal_climatology_with_DJF(self):
        ds: xr.Dataset = generate_dataset(cf_compliant=True, has_bounds=True)

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
            coords={"lat": ds.lat, "lon": ds.lon, "time": expected_time},
            dims=["time", "lat", "lon"],
            attrs={
                "operation": "temporal_avg",
                "mode": "climatology",
                "freq": "season",
                "groupby": "season",
                "weighted": "True",
                "center_times": "False",
                "dec_mode": "DJF",
                "drop_incomplete_djf": "True",
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
                "lat": ds.lat,
                "lon": ds.lon,
                "time": ds.time,
            },
            dims=["time", "lat", "lon"],
            attrs={
                "operation": "temporal_avg",
                "mode": "departures",
                "freq": "season",
                "weighted": "True",
                "center_times": "False",
                "groupby": "season",
                "dec_mode": "DJF",
                "drop_incomplete_djf": "True",
            },
        )

        assert result.identical(expected)

    def test_unweighted_seasonal_departures_with_DJF(self):
        # Create a post-climatology dataset.
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
                "lat": ds.lat,
                "lon": ds.lon,
                "time": ds.time,
            },
            dims=["time", "lat", "lon"],
            attrs={
                "operation": "temporal_avg",
                "mode": "departures",
                "freq": "season",
                "weighted": "False",
                "center_times": "False",
                "groupby": "season",
                "dec_mode": "DJF",
                "drop_incomplete_djf": "True",
            },
        )

        assert result.identical(expected)

    def test_unweighted_seasonal_departures_with_JFD(self):
        # Create a post-climatology dataset.
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
                "lat": ds.lat,
                "lon": ds.lon,
                "time": ds.time,
            },
            dims=["time", "lat", "lon"],
            attrs={
                "operation": "temporal_avg",
                "mode": "departures",
                "freq": "season",
                "weighted": "False",
                "center_times": "False",
                "groupby": "season",
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

        # Compare result of the method against the expected.
        result = ds.temporal.center_times(ds)
        assert result.identical(expected)


class TestTemporalAvg:
    # TODO: Update TestTimeSeries tests to use other numbers rather than 1's
    # for better test reliability and accuracy. This may require subsetting.
    class TestTimeSeries:
        @pytest.fixture(autouse=True)
        def setup(self):
            # FIXME: Update test this so that it is accurate, rather than 1's
            # for averages
            # May involve subsetting
            self.ds: xr.Dataset = generate_dataset(cf_compliant=True, has_bounds=True)

        def test_weighted_annual_avg(self):
            ds = self.ds.copy()

            result = ds.temporal._temporal_avg("ts", "time_series", "year")
            expected = ds.copy()
            expected = expected.drop_dims("time")
            time_new = xr.DataArray(
                data=np.array(
                    ["2000-01-01T00:00:00.000000000", "2001-01-01T00:00:00.000000000"],
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
            )
            expected["ts"] = xr.DataArray(
                name="ts",
                data=np.ones((2, 4, 4)),
                coords={
                    "lat": self.ds.lat,
                    "lon": self.ds.lon,
                    "time": time_new,
                },
                dims=["time", "lat", "lon"],
                attrs={
                    "operation": "temporal_avg",
                    "mode": "time_series",
                    "freq": "year",
                    "groupby": "year",
                    "weighted": "True",
                    "center_times": "False",
                },
            )

            # For some reason, there is a very small floating point difference
            # between both for ts so we have to use floating point comparison
            xr.testing.assert_allclose(result, expected)
            assert result.ts.attrs == expected.ts.attrs

        @requires_dask
        def test_weighted_annual_avg_with_chunking(self):
            ds = self.ds.copy().chunk({"time": 2})

            result = ds.temporal._temporal_avg("ts", "time_series", "year")
            expected = ds.copy()
            expected = expected.drop_dims("time")
            time_new = xr.DataArray(
                data=np.array(
                    ["2000-01-01T00:00:00.000000000", "2001-01-01T00:00:00.000000000"],
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
            )
            expected["ts"] = xr.DataArray(
                name="ts",
                data=np.ones((2, 4, 4)),
                coords={
                    "lat": self.ds.lat,
                    "lon": self.ds.lon,
                    "time": time_new,
                },
                dims=["time", "lat", "lon"],
                attrs={
                    "operation": "temporal_avg",
                    "mode": "time_series",
                    "freq": "year",
                    "groupby": "year",
                    "weighted": "True",
                    "center_times": "False",
                },
            )

            # For some reason, there is a very small floating point difference
            # between both for ts so we have to use floating point comparison
            xr.testing.assert_allclose(result, expected)
            assert result.ts.attrs == expected.ts.attrs

        def test_weighted_annual_avg_with_centering_time(self):
            ds = self.ds.copy()

            result = ds.temporal._temporal_avg(
                "ts", "time_series", "year", center_times=True
            )
            expected = ds.copy()
            expected = expected.drop_dims("time")
            time_new = xr.DataArray(
                data=np.array(
                    ["2000-01-01T00:00:00.000000000", "2001-01-01T00:00:00.000000000"],
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
            )
            expected["ts"] = xr.DataArray(
                name="ts",
                data=np.ones((2, 4, 4)),
                coords={
                    "lat": self.ds.lat,
                    "lon": self.ds.lon,
                    "time": time_new,
                },
                dims=["time", "lat", "lon"],
                attrs={
                    "operation": "temporal_avg",
                    "mode": "time_series",
                    "freq": "year",
                    "groupby": "year",
                    "weighted": "True",
                    "center_times": "True",
                },
            )

            # For some reason, there is a floating point difference between both
            # for ts so we have to use floating point comparison
            xr.testing.assert_allclose(result, expected)
            assert result.ts.attrs == expected.ts.attrs

        def test_weighted_seasonal_avg_with_DJF(self):
            ds = self.ds.copy()

            result = ds.temporal._temporal_avg(
                "ts",
                "time_series",
                "season",
                season_config={"dec_mode": "DJF", "drop_incomplete_djf": True},
            )
            expected = ds.copy()
            # Drop the incomplete DJF seasons
            expected = expected.isel(time=slice(2, -1))

            expected = expected.drop_dims("time")
            time_new = xr.DataArray(
                data=np.array(
                    [
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
            )
            expected["ts"] = xr.DataArray(
                name="ts",
                data=np.ones((4, 4, 4)),
                coords={
                    "lat": self.ds.lat,
                    "lon": self.ds.lon,
                    "time": time_new,
                },
                dims=["time", "lat", "lon"],
                attrs={
                    "operation": "temporal_avg",
                    "mode": "time_series",
                    "freq": "season",
                    "groupby": "year_season",
                    "weighted": "True",
                    "center_times": "False",
                    "dec_mode": "DJF",
                    "drop_incomplete_djf": "True",
                },
            )

            assert result.identical(expected)

        def test_weighted_seasonal_avg_with_DJF_without_dropping_incomplete_seasons(
            self,
        ):
            ds = self.ds.copy()

            result = ds.temporal._temporal_avg(
                "ts",
                "time_series",
                "season",
                season_config={"dec_mode": "DJF", "drop_incomplete_djf": False},
            )
            expected = ds.copy()
            expected = expected.drop_dims("time")
            time_new = xr.DataArray(
                data=np.array(
                    [
                        "2000-01-01T00:00:00.000000000",
                        "2000-04-01T00:00:00.000000000",
                        "2000-07-01T00:00:00.000000000",
                        "2000-10-01T00:00:00.000000000",
                        "2001-01-01T00:00:00.000000000",
                        "2002-01-01T00:00:00.000000000",
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
                            "2002-01-01T00:00:00.000000000",
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
            )
            expected["ts"] = xr.DataArray(
                name="ts",
                data=np.ones((6, 4, 4)),
                coords={
                    "lat": self.ds.lat,
                    "lon": self.ds.lon,
                    "time": time_new,
                },
                dims=["time", "lat", "lon"],
                attrs={
                    "operation": "temporal_avg",
                    "mode": "time_series",
                    "freq": "season",
                    "groupby": "year_season",
                    "weighted": "True",
                    "center_times": "False",
                    "dec_mode": "DJF",
                    "drop_incomplete_djf": "False",
                },
            )

            assert result.identical(expected)

        def test_weighted_seasonal_avg_with_JFD(self):
            ds = self.ds.copy()

            ds = self.ds.copy()

            result = ds.temporal._temporal_avg(
                "ts",
                "time_series",
                "season",
                season_config={"dec_mode": "JFD"},
            )
            expected = ds.copy()
            expected = expected.drop_dims("time")
            time_new = xr.DataArray(
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
            )
            expected["ts"] = xr.DataArray(
                name="ts",
                data=np.ones((5, 4, 4)),
                coords={
                    "lat": self.ds.lat,
                    "lon": self.ds.lon,
                    "time": time_new,
                },
                dims=["time", "lat", "lon"],
                attrs={
                    "operation": "temporal_avg",
                    "mode": "time_series",
                    "freq": "season",
                    "groupby": "year_season",
                    "weighted": "True",
                    "center_times": "False",
                    "dec_mode": "JFD",
                },
            )

            assert result.identical(expected)

        @pytest.mark.xfail
        def test_weighted_custom_season_avg(self):
            ds = self.ds.copy()

            custom_seasons = [
                ["Jan", "Feb", "Mar"],
                ["Apr", "May", "Jun"],
                ["Jul", "Aug", "Sep"],
                ["Oct", "Nov", "Dec"],
            ]
            result = ds.temporal._temporal_avg(
                "ts",
                "time_series",
                "season",
                season_config={"custom_seasons": custom_seasons},
            )
            expected = ds.copy()
            expected["ts_original"] = ds.ts.copy()
            expected["ts"] = xr.DataArray(
                name="ts",
                data=np.ones((6, 4, 4)),
                coords={
                    "lat": self.ds.lat,
                    "lon": self.ds.lon,
                    "year_season": pd.MultiIndex.from_tuples(
                        [
                            (2000, "JanFebMar"),
                            (2000, "AprMayJun"),
                            (2000, "JulAugSep"),
                            (2000, "OctNovDec"),
                            (2001, "JanFebMar"),
                            (2001, "OctNovDec"),
                        ],
                    ),
                },
                dims=["year_season", "lat", "lon"],
                attrs={
                    "operation": "temporal_avg",
                    "mode": "time_series",
                    "freq": "season",
                    "groupby": "year_season",
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

        def test_weighted_monthly_avg(self):
            ds = self.ds.copy()

            result = ds.temporal._temporal_avg("ts", "time_series", "month")
            expected = ds.copy()
            expected = expected.drop_dims("time")
            time_new = xr.DataArray(
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
                coords={
                    "time": np.array(
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
                },
                dims=["time"],
                attrs={
                    "axis": "T",
                    "long_name": "time",
                    "standard_name": "time",
                    "bounds": "time_bnds",
                },
            )
            expected["ts"] = xr.DataArray(
                name="ts",
                data=np.ones((15, 4, 4)),
                coords={
                    "lat": self.ds.lat,
                    "lon": self.ds.lon,
                    "time": time_new,
                },
                dims=["time", "lat", "lon"],
                attrs={
                    "operation": "temporal_avg",
                    "mode": "time_series",
                    "freq": "month",
                    "groupby": "year_month",
                    "weighted": "True",
                    "center_times": "False",
                },
            )

            assert result.identical(expected)

        def test_weighted_daily_avg(self):
            ds = self.ds.copy()

            result = ds.temporal._temporal_avg("ts", "time_series", "day")
            expected = ds.copy()
            expected = expected.drop_dims("time")
            time_new = xr.DataArray(
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
                coords={
                    "time": np.array(
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
                },
                dims=["time"],
                attrs={
                    "axis": "T",
                    "long_name": "time",
                    "standard_name": "time",
                    "bounds": "time_bnds",
                },
            )
            expected["ts"] = xr.DataArray(
                name="ts",
                data=np.ones((15, 4, 4)),
                coords={
                    "lat": self.ds.lat,
                    "lon": self.ds.lon,
                    "time": time_new,
                },
                dims=["time", "lat", "lon"],
                attrs={
                    "operation": "temporal_avg",
                    "mode": "time_series",
                    "freq": "day",
                    "groupby": "year_month_day",
                    "weighted": "True",
                    "center_times": "False",
                },
            )

            assert result.identical(expected)

        def test_weighted_hourly_avg(self):
            ds = self.ds.copy()
            ds.coords["time"].attrs["bounds"] = "time_bnds"

            result = ds.temporal._temporal_avg("ts", "time_series", "hour")
            expected = ds.copy()
            expected["ts_original"] = ds.ts.copy()
            expected["ts"] = xr.DataArray(
                name="ts",
                data=np.ones((15, 4, 4)),
                coords={
                    "lat": self.ds.lat,
                    "lon": self.ds.lon,
                    "year_month_day_hour": pd.MultiIndex.from_tuples(
                        [
                            (2000, 1, 16, 12),
                            (2000, 2, 15, 12),
                            (2000, 3, 16, 12),
                            (2000, 4, 16, 0),
                            (2000, 5, 16, 12),
                            (2000, 6, 16, 0),
                            (2000, 7, 16, 12),
                            (2000, 8, 16, 12),
                            (2000, 9, 16, 0),
                            (2000, 10, 16, 12),
                            (2000, 11, 16, 0),
                            (2000, 12, 16, 12),
                            (2001, 1, 16, 12),
                            (2001, 2, 15, 0),
                            (2001, 12, 16, 12),
                        ]
                    ),
                },
                dims=["year_month_day_hour", "lat", "lon"],
                attrs={
                    "operation": "temporal_avg",
                    "mode": "time_series",
                    "freq": "hour",
                    "groupby": "year_month_day_hour",
                    "weighted": "True",
                    "center_times": "False",
                },
            )

            expected = ds.copy()
            expected = expected.drop_dims("time")
            time_new = xr.DataArray(
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
                coords={
                    "time": np.array(
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
                },
                dims=["time"],
                attrs={
                    "axis": "T",
                    "long_name": "time",
                    "standard_name": "time",
                    "bounds": "time_bnds",
                },
            )
            expected["ts"] = xr.DataArray(
                name="ts",
                data=np.ones((15, 4, 4)),
                coords={
                    "lat": self.ds.lat,
                    "lon": self.ds.lon,
                    "time": time_new,
                },
                dims=["time", "lat", "lon"],
                attrs={
                    "operation": "temporal_avg",
                    "mode": "time_series",
                    "freq": "hour",
                    "groupby": "year_month_day_hour",
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

        def test_raises_error_without_time_dimension(self):
            ds = self.ds.copy()
            ds = ds.drop_dims("time")

            with pytest.raises(KeyError):
                ds.temporal._temporal_avg("climatology", "season", "ts")

        def test_raises_error_with_incorrect_freq_arg(self):
            with pytest.raises(ValueError):
                self.ds.temporal._temporal_avg(
                    "ts",
                    "climatology",
                    "incorrect_freq",
                )

        def test_raises_error_with_incorrect_dec_mode_arg(self):
            with pytest.raises(ValueError):
                self.ds.temporal._temporal_avg(
                    "ts",
                    "climatology",
                    freq="season",
                    season_config={"dec_mode": "incorrect"},
                )

        def test_raises_error_if_data_var_does_not_exist_in_dataset(self):
            with pytest.raises(KeyError):
                self.ds.temporal._temporal_avg(
                    "nonexistent_var", "climatology", freq="season"
                )

        def test_weighted_seasonal_climatology_with_DJF(self):
            ds = self.ds.copy()

            result = ds.temporal._temporal_avg(
                "ts",
                "climatology",
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
                coords={"lat": self.ds.lat, "lon": self.ds.lon, "time": expected_time},
                dims=["time", "lat", "lon"],
                attrs={
                    "operation": "temporal_avg",
                    "mode": "climatology",
                    "freq": "season",
                    "groupby": "season",
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

            result = ds.temporal._temporal_avg(
                "ts",
                "climatology",
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
                coords={"lat": self.ds.lat, "lon": self.ds.lon, "time": expected_time},
                dims=["time", "lat", "lon"],
                attrs={
                    "operation": "temporal_avg",
                    "mode": "climatology",
                    "freq": "season",
                    "groupby": "season",
                    "weighted": "True",
                    "center_times": "False",
                    "dec_mode": "DJF",
                    "drop_incomplete_djf": "True",
                },
            )

            assert result.identical(expected)

        def test_weighted_seasonal_climatology_with_JFD(self):
            ds = self.ds.copy()

            result = ds.temporal._temporal_avg(
                "ts",
                "climatology",
                "season",
                season_config={"dec_mode": "JFD"},
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
                coords={"lat": self.ds.lat, "lon": self.ds.lon, "time": expected_time},
                dims=["time", "lat", "lon"],
                attrs={
                    "operation": "temporal_avg",
                    "mode": "climatology",
                    "freq": "season",
                    "groupby": "season",
                    "weighted": "True",
                    "center_times": "False",
                    "dec_mode": "JFD",
                },
            )

            assert result.identical(expected)

        @pytest.mark.xfail
        def test_weighted_custom_seasonal_climatology(self):
            # FIXME: Fix this test
            ds = self.ds.copy()

            custom_seasons = [
                ["Jan", "Feb", "Mar"],
                ["Apr", "May", "Jun"],
                ["Jul", "Aug", "Sep"],
                ["Oct", "Nov", "Dec"],
            ]
            result = ds.temporal._temporal_avg(
                "ts",
                "climatology",
                "season",
                season_config={"custom_seasons": custom_seasons},
            )
            expected = ds.copy()
            expected["ts_original"] = ds.ts.copy()
            expected["ts"] = xr.DataArray(
                name="ts",
                data=np.ones((4, 4, 4)),
                coords={
                    "lat": self.ds.lat,
                    "lon": self.ds.lon,
                    "season": pd.MultiIndex.from_tuples(
                        [
                            ("JanFebMar",),
                            ("AprMayJun",),
                            ("JulAugSep",),
                            ("OctNovDec",),
                        ],
                    ),
                },
                dims=["season", "lat", "lon"],
                attrs={
                    "operation": "temporal_avg",
                    "mode": "climatology",
                    "freq": "season",
                    "groupby": "season",
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
            result = self.ds.temporal._temporal_avg("ts", "climatology", "month")

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
                coords={"lat": self.ds.lat, "lon": self.ds.lon, "time": expected_time},
                dims=["time", "lat", "lon"],
                attrs={
                    "operation": "temporal_avg",
                    "mode": "climatology",
                    "freq": "month",
                    "groupby": "month",
                    "weighted": "True",
                    "center_times": "False",
                },
            )

            assert result.identical(expected)

        def test_unweighted_monthly_climatology(self):
            result = self.ds.temporal._temporal_avg(
                "ts", "climatology", "month", weighted=False
            )

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
                coords={"lat": self.ds.lat, "lon": self.ds.lon, "time": expected_time},
                dims=["time", "lat", "lon"],
                attrs={
                    "operation": "temporal_avg",
                    "mode": "climatology",
                    "freq": "month",
                    "groupby": "month",
                    "weighted": "False",
                    "center_times": "False",
                },
            )

            assert result.identical(expected)

        def test_weighted_daily_climatology(self):
            result = self.ds.temporal._temporal_avg(
                "ts", "climatology", "day", weighted=True
            )

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
                coords={"lat": self.ds.lat, "lon": self.ds.lon, "time": expected_time},
                dims=["time", "lat", "lon"],
                attrs={
                    "operation": "temporal_avg",
                    "mode": "climatology",
                    "freq": "day",
                    "groupby": "month_day",
                    "weighted": "True",
                    "center_times": "False",
                },
            )

            assert result.identical(expected)

        def test_unweighted_daily_climatology(self):
            result = self.ds.temporal._temporal_avg(
                "ts", "climatology", "day", weighted=False
            )

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
                coords={"lat": self.ds.lat, "lon": self.ds.lon, "time": expected_time},
                dims=["time", "lat", "lon"],
                attrs={
                    "operation": "temporal_avg",
                    "mode": "climatology",
                    "freq": "day",
                    "groupby": "month_day",
                    "weighted": "False",
                    "center_times": "False",
                },
            )

            assert result.identical(expected)


class TestSetObjAttrs:
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
                "time_series",
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
                "time_series",
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

    def test_sets_object_attributes(self):
        ds = self.ds.copy()
        ds.temporal._set_obj_attrs(
            "climatology",
            freq="season",
            weighted=True,
            center_times=True,
            season_config={"dec_mode": "JFD"},
        )
        assert ds.temporal._mode == "climatology"
        assert ds.temporal._freq == "season"
        assert ds.temporal._center_times
        assert ds.temporal._weighted
        assert ds.temporal._season_config == {
            "dec_mode": "JFD",
        }

        ds.temporal._set_obj_attrs(
            "climatology",
            freq="season",
            weighted=True,
            center_times=True,
            season_config={
                "custom_seasons": [
                    ["Jan", "Feb", "Mar"],
                    ["Apr", "May", "Jun"],
                    ["Jul", "Aug", "Sep"],
                    ["Oct", "Nov", "Dec"],
                ],
            },
        )
        assert ds.temporal._season_config == {
            "custom_seasons": [
                "JanFebMar",
                "AprMayJun",
                "JulAugSep",
                "OctNovDec",
            ],
        }


class TestCustomSeasons:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = generate_dataset(cf_compliant=True, has_bounds=True)
        self.expected = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]

    def test_raises_error_if_month_str_not_supported(
        self,
    ):
        # Incorrect str "J".
        with pytest.raises(ValueError):
            self.ds.temporal._form_seasons(
                custom_seasons=[
                    ["J", "Feb", "Mar"],
                    ["Apr", "May", "Jun"],
                    ["Jul", "Aug", "Sep"],
                    ["Oct", "Nov", "Dec"],
                ]
            )

        # Incorrect str "January".
        with pytest.raises(ValueError):
            self.ds.temporal._form_seasons(
                custom_seasons=[
                    ["January", "Feb", "Mar"],
                    ["Apr", "May", "Jun"],
                    ["Jul", "Aug", "Sep"],
                    ["Oct", "Nov", "Dec"],
                ]
            )

    def test_raises_error_if_missing_months(self):
        with pytest.raises(ValueError):
            # "Jan" is missing.
            self.ds.temporal._form_seasons(
                custom_seasons=[
                    ["Feb", "Mar"],
                    ["Apr", "May", "Jun"],
                    ["Jul", "Aug", "Sep"],
                    ["Oct", "Nov", "Dec"],
                ]
            )

    def test_raises_error_if_duplicate_months_were_found(self):
        with pytest.raises(ValueError):
            # "Jan" is duplicated.
            self.ds.temporal._form_seasons(
                custom_seasons=[
                    ["Jan", "Jan", "Feb"],
                    ["Apr", "May", "Jun"],
                    ["Jul", "Aug", "Sep"],
                    ["Oct", "Nov", "Dec"],
                ]
            )

    def test_does_not_raise_error(self):
        result = self.ds.temporal._form_seasons(
            custom_seasons=[
                ["Jan", "Feb", "Mar"],
                ["Apr", "May", "Jun"],
                ["Jul", "Aug", "Sep"],
                ["Oct", "Nov", "Dec"],
            ]
        )
        expected = ["JanFebMar", "AprMayJun", "JulAugSep", "OctNovDec"]
        assert result == expected

        result = self.ds.temporal._form_seasons(
            custom_seasons=[
                ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
                ["Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
            ]
        )
        expected = ["JanFebMarAprMayJun", "JulAugSepOctNovDec"]
        assert result == expected

        result = self.ds.temporal._form_seasons(
            custom_seasons=[
                ["Jan", "Feb", "Mar"],
                ["Apr", "May", "Jun", "Jul"],
                ["Aug", "Sep", "Oct", "Nov", "Dec"],
            ]
        )
        expected = ["JanFebMar", "AprMayJunJul", "AugSepOctNovDec"]
        assert result == expected


class TestAverager:
    # FIXME: Update test this so that it is accurate, rather than 1's
    # for averages
    # May involve subsetting
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds: xr.Dataset = generate_dataset(cf_compliant=True, has_bounds=True)
        self.ds.attrs.update({"operation_type": "climatology"})

    def test_weighted_by_month_day(self):
        ds = self.ds.copy()

        # Set object attrs required to test the method
        ds.temporal._time_bounds = ds.time_bnds.copy()
        ds.temporal._mode = "climatology"
        ds.temporal._freq = "day"
        ds.temporal._weighted = True
        ds.temporal._center_times = True
        ds.temporal._time_grouped = xr.DataArray(
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
        ts_result = ds.temporal._averager(ds["ts"])
        ts_expected = np.ones((12, 4, 4))
        assert np.allclose(ts_result, ts_expected)

    def test_unweighted_daily_climatology(self):
        ds = self.ds.copy()

        # Set object attrs required to test the method.
        ds.temporal._time_bounds = ds.time_bnds.copy()
        ds.temporal._mode = "climatology"
        ds.temporal._freq = "day"
        ds.temporal._weighted = False
        ds.temporal._center_times = True
        ds.temporal._time_grouped = xr.DataArray(
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
        ts_result = ds.temporal._averager(ds["ts"])
        ts_expected = np.ones((12, 4, 4))
        assert np.allclose(ts_result, ts_expected)

    def test_weighted_annual_climatology(self):
        ds = self.ds.copy()

        # Set object attrs required to test the method.
        ds.temporal._time_bounds = ds.time_bnds.copy()
        ds.temporal._mode = "climatology"
        ds.temporal._freq = "month"
        ds.temporal._weighted = True
        ds.temporal._center_times = True
        ds.temporal._time_grouped = xr.DataArray(
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
        # Check non-bounds variables were properly grouped and averaged
        ts_result = ds.temporal._averager(ds["ts"])
        ts_expected = np.ones((12, 4, 4))
        assert np.allclose(ts_result, ts_expected)

    def test_weighted_seasonal_climatology_with_DJF_and_drop_incomplete_DJF(self):
        ds = self.ds.copy()
        # Drop the incomplete seasons
        ds = ds.isel(time=slice(2, -1))

        # Set object attrs required to test the method.
        ds.temporal._time_bounds = ds.time_bnds.copy()
        ds.temporal._mode = "climatology"
        ds.temporal._freq = "season"
        ds.temporal._weighted = True
        ds.temporal._center_times = True
        ds.temporal._season_config = {
            "dec_mode": "DJF",
            "drop_incomplete_djf": True,
        }
        ds.temporal._time_grouped = xr.DataArray(
            name="season",
            data=np.array(
                [
                    # MAM
                    cftime.datetime(1, 3, 1),
                    cftime.datetime(1, 3, 1),
                    cftime.datetime(1, 3, 1),
                    # JJA
                    cftime.datetime(1, 7, 1),
                    cftime.datetime(1, 7, 1),
                    cftime.datetime(1, 7, 1),
                    # SON
                    cftime.datetime(1, 10, 1),
                    cftime.datetime(1, 10, 1),
                    cftime.datetime(1, 10, 1),
                    # DJF
                    cftime.datetime(1, 1, 1),
                    cftime.datetime(1, 1, 1),
                    cftime.datetime(1, 1, 1),
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
        # Check non-bounds variables were properly grouped and averaged
        ts_result = ds.temporal._averager(ds["ts"])
        ts_expected = np.ones((4, 4, 4))
        assert np.allclose(ts_result, ts_expected)

    def test_weighted_seasonal_climatology_with_JFD(self):
        ds = self.ds.copy()

        # Set object attrs required to test the method.
        ds.temporal._time_bounds = ds.time_bnds.copy()
        ds.temporal._mode = "climatology"
        ds.temporal._freq = "season"
        ds.temporal._weighted = True
        ds.temporal._center_times = True
        ds.temporal._season_config = {"dec_mode": "JFD"}
        ds.temporal._time_grouped = xr.DataArray(
            name="season",
            data=np.array(
                [
                    # JFD
                    cftime.datetime(1, 1, 1),
                    cftime.datetime(1, 1, 1),
                    cftime.datetime(1, 1, 1),
                    # MAM
                    cftime.datetime(1, 3, 1),
                    cftime.datetime(1, 3, 1),
                    cftime.datetime(1, 3, 1),
                    # JJA
                    cftime.datetime(1, 7, 1),
                    cftime.datetime(1, 7, 1),
                    cftime.datetime(1, 7, 1),
                    # SON
                    cftime.datetime(1, 10, 1),
                    cftime.datetime(1, 10, 1),
                    cftime.datetime(1, 10, 1),
                    # JFD
                    cftime.datetime(1, 1, 1),
                    cftime.datetime(1, 1, 1),
                    cftime.datetime(1, 1, 1),
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
        ts_result = ds.temporal._averager(ds["ts"])
        ts_expected = np.ones((4, 4, 4))
        assert np.allclose(ts_result, ts_expected)


class TestDropIncompleteDJF:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds: xr.Dataset = generate_dataset(cf_compliant=True, has_bounds=True)

    def test_incomplete_DJF_seasons_are_dropped(self):
        ds = self.ds.copy()

        # Set object attrs required to test the method.
        ds.temporal._time_bounds = ds.time_bnds.copy()

        # Define method inputs.
        ds["ts"] = xr.DataArray(
            data=np.ones(5),
            coords={
                "time": [
                    datetime(2000, 1, 1),
                    datetime(2000, 2, 1),
                    datetime(2000, 3, 1),
                    datetime(2000, 4, 1),
                    datetime(2001, 12, 1),
                ]
            },
            dims=["time"],
        )

        # Compare result of the method against the expected.
        result = ds.temporal._drop_incomplete_djf(ds)
        expected = ds.copy()
        # Drop the incomplete DJF seasons
        expected = expected.isel(time=slice(2, -1))
        expected["ts"] = xr.DataArray(
            data=np.ones(2),
            coords={"time": [datetime(2000, 3, 1), datetime(2000, 4, 1)]},
            dims=["time"],
        )
        assert result.identical(expected)

    def test_does_not_drop_incomplete_DJF_seasons_since_if_dont_exist(self):
        ds = self.ds.copy()

        # Set object attrs required to test the method.
        ds.temporal._time_bounds = ds.time_bnds.copy()

        # Update time coordinate points so that the months don't fall in
        # incomplete seasons.
        ds.time.values[0] = datetime(1999, 3, 1)
        ds.time.values[1] = datetime(1999, 4, 1)
        ds.time.values[-1] = datetime(1999, 5, 1)

        # Compare result of the method against the expected.
        result = ds.temporal._drop_incomplete_djf(ds)
        expected = ds
        assert result.identical(expected)


class TestGroupTimeCoords:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = generate_dataset(cf_compliant=True, has_bounds=True)

    def test_groups_time_coords_for_time_series_season_freq(self):
        ds = self.ds.copy()

        # Set object attrs required to test the method.
        ds.temporal._mode = "time_series"
        ds.temporal._freq = "season"
        ds.temporal._season_config = {"dec_mode": "DJF", "drop_incomplete_djf": False}

        # Compare result of the method against the expected.
        result = ds.temporal._group_time_coords(ds.ts)
        expected = xr.DataArray(
            name="year_season",
            data=np.array(
                [
                    "2000-01-01T00:00:00.000000000",
                    "2000-01-01T00:00:00.000000000",
                    "2000-04-01T00:00:00.000000000",
                    "2000-04-01T00:00:00.000000000",
                    "2000-04-01T00:00:00.000000000",
                    "2000-07-01T00:00:00.000000000",
                    "2000-07-01T00:00:00.000000000",
                    "2000-07-01T00:00:00.000000000",
                    "2000-10-01T00:00:00.000000000",
                    "2000-10-01T00:00:00.000000000",
                    "2000-10-01T00:00:00.000000000",
                    "2001-01-01T00:00:00.000000000",
                    "2001-01-01T00:00:00.000000000",
                    "2001-01-01T00:00:00.000000000",
                    "2002-01-01T00:00:00.000000000",
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
        assert result.identical(expected)

    def test_groups_time_coords_for_climatology_season_freq(self):
        ds = self.ds.copy()

        # Set object attrs required to test the method.
        ds.temporal._mode = "climatology"
        ds.temporal._freq = "season"
        ds.temporal._time_bounds = ds.time_bnds.copy()
        ds.temporal._season_config = {"dec_mode": "DJF", "drop_incomplete_djf": False}

        # Compare result of the method against the expected.
        result = ds.temporal._group_time_coords(ds.ts)
        expected = xr.DataArray(
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

        assert result.identical(expected)


class TestProcessSeasonDataFrame:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = generate_dataset(cf_compliant=True, has_bounds=True)
        self.df = pd.DataFrame(
            data=np.array(
                [
                    (2000, "DJF", 1),
                    (2000, "DJF", 2),
                    (2000, "MAM", 3),
                    (2000, "MAM", 4),
                    (2000, "MAM", 5),
                    (2000, "JJA", 6),
                    (2000, "JJA", 7),
                    (2000, "JJA", 8),
                    (2000, "SON", 9),
                    (2000, "SON", 10),
                    (2000, "SON", 11),
                    (2000, "DJF", 12),
                ],
                dtype=object,
            ),
            columns=["year", "season", "month"],
        )

    def test_maps_custom_seasons_if_custom_seasons_specified_and_drops_columns(self):
        ds = self.ds.copy()
        df = self.df.copy()

        # Set object attrs required to test the method.
        ds.temporal._mode = "time_series"
        ds.temporal._season_config = {
            "custom_seasons": [
                "JanFebMar",
                "AprMayJun",
                "JulAugSep",
                "OctNovDec",
            ]
        }

        # Compare result of the method against the expected.
        result = ds.temporal._process_season_dataframe(df)
        expected = pd.DataFrame(
            data=np.array(
                [
                    (2000, "JanFebMar"),
                    (2000, "JanFebMar"),
                    (2000, "JanFebMar"),
                    (2000, "AprMayJun"),
                    (2000, "AprMayJun"),
                    (2000, "AprMayJun"),
                    (2000, "JulAugSep"),
                    (2000, "JulAugSep"),
                    (2000, "JulAugSep"),
                    (2000, "OctNovDec"),
                    (2000, "OctNovDec"),
                    (2000, "OctNovDec"),
                ],
                dtype=object,
            ),
            columns=["year", "season"],
        )
        assert result.equals(expected)

    def test_shifts_decembers_for_DJF_if_DJF_is_specified(self):
        ds = self.ds.copy()
        df = self.df.copy()

        # Set object attrs required to test the method.
        ds.temporal._mode = "climatology"
        ds.temporal._season_config = {
            "dec_mode": "DJF",
            "drop_incomplete_djf": True,
        }

        # Compare result of the method against the expected.
        result = ds.temporal._process_season_dataframe(df)
        expected = pd.DataFrame(
            data=np.array(
                [
                    ("DJF"),
                    ("DJF"),
                    ("MAM"),
                    ("MAM"),
                    ("MAM"),
                    ("JJA"),
                    ("JJA"),
                    ("JJA"),
                    ("SON"),
                    ("SON"),
                    ("SON"),
                    ("DJF"),
                ],
                dtype=object,
            ),
            columns=["season"],
        )
        assert result.equals(expected)


class TestConvertDFtoDTObjects:
    @pytest.mark.xfail
    def test_placeholder(self):
        assert 0


class TestMapCustomSeasons:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = generate_dataset(cf_compliant=True, has_bounds=True)

    def test_raises_error_if_custom_seasons_are_not_mapped(self):
        ds = self.ds.copy()
        ds.temporal._season_config = {"custom_seasons": None}
        df = pd.DataFrame(
            data=np.array(
                [
                    (2000, "DJF", 1),
                    (2000, "DJF", 2),
                    (2000, "MAM", 3),
                    (2000, "MAM", 4),
                    (2000, "MAM", 5),
                    (2000, "JJA", 6),
                    (2000, "JJA", 7),
                    (2000, "JJA", 8),
                    (2000, "SON", 9),
                    (2000, "SON", 10),
                    (2000, "SON", 11),
                    (2000, "DJF", 12),
                ],
                dtype=object,
            ),
            columns=["year", "season", "month"],
        )

        with pytest.raises(ValueError):
            ds.temporal._map_custom_seasons(df)

    def test_maps_three_month_custom_seasons(self):
        ds = self.ds.copy()

        # Set object attrs required to test the method.
        ds.temporal._season_config = {
            "custom_seasons": [
                "JanFebMar",
                "AprMayJun",
                "JulAugSep",
                "OctNovDec",
            ]
        }

        # Define method inputs.
        # Includes default seasons.
        df = pd.DataFrame(
            data=np.array(
                [
                    (2000, "DJF", 1),
                    (2000, "DJF", 2),
                    (2000, "MAM", 3),
                    (2000, "MAM", 4),
                    (2000, "MAM", 5),
                    (2000, "JJA", 6),
                    (2000, "JJA", 7),
                    (2000, "JJA", 8),
                    (2000, "SON", 9),
                    (2000, "SON", 10),
                    (2000, "SON", 11),
                    (2000, "DJF", 12),
                ],
                dtype=object,
            ),
            columns=["year", "season", "month"],
        )

        # Compare result of the method against the expected.
        result = ds.temporal._map_custom_seasons(df)
        expected = pd.DataFrame(
            data=np.array(
                [
                    (2000, "JanFebMar", 1),
                    (2000, "JanFebMar", 2),
                    (2000, "JanFebMar", 3),
                    (2000, "AprMayJun", 4),
                    (2000, "AprMayJun", 5),
                    (2000, "AprMayJun", 6),
                    (2000, "JulAugSep", 7),
                    (2000, "JulAugSep", 8),
                    (2000, "JulAugSep", 9),
                    (2000, "OctNovDec", 10),
                    (2000, "OctNovDec", 11),
                    (2000, "OctNovDec", 12),
                ],
                dtype=object,
            ),
            columns=["year", "season", "month"],
        )
        assert result.equals(expected)

    def test_maps_six_month_custom_seasons(self):
        ds = self.ds.copy()

        # Set object attrs required to test the method.
        ds.temporal._season_config = {
            "custom_seasons": [
                "JanFebMarAprMayJun",
                "JulAugSepOctNovDec",
            ]
        }

        # Define method inputs.
        # Includes default seasons.
        df = pd.DataFrame(
            data=np.array(
                [
                    (2000, "DJF", 1),
                    (2000, "DJF", 2),
                    (2000, "MAM", 3),
                    (2000, "MAM", 4),
                    (2000, "MAM", 5),
                    (2000, "JJA", 6),
                    (2000, "JJA", 7),
                    (2000, "JJA", 8),
                    (2000, "SON", 9),
                    (2000, "SON", 10),
                    (2000, "SON", 11),
                    (2000, "DJF", 12),
                ],
                dtype=object,
            ),
            columns=["year", "season", "month"],
        )

        # Compare result of the method against the expected.
        result = ds.temporal._map_custom_seasons(df)
        expected = pd.DataFrame(
            data=np.array(
                [
                    (2000, "JanFebMarAprMayJun", 1),
                    (2000, "JanFebMarAprMayJun", 2),
                    (2000, "JanFebMarAprMayJun", 3),
                    (2000, "JanFebMarAprMayJun", 4),
                    (2000, "JanFebMarAprMayJun", 5),
                    (2000, "JanFebMarAprMayJun", 6),
                    (2000, "JulAugSepOctNovDec", 7),
                    (2000, "JulAugSepOctNovDec", 8),
                    (2000, "JulAugSepOctNovDec", 9),
                    (2000, "JulAugSepOctNovDec", 10),
                    (2000, "JulAugSepOctNovDec", 11),
                    (2000, "JulAugSepOctNovDec", 12),
                ],
                dtype=object,
            ),
            columns=["year", "season", "month"],
        )
        assert result.equals(expected)

    def test_maps_three_month_custom_seasons_random_order(self):
        ds = self.ds.copy()

        # Set object attrs required to test the method.
        ds.temporal._season_config = {
            "custom_seasons": [
                # Swapped Jan and Dec
                "DecFebMar",
                "AprMayJun",
                "JulAugSep",
                "OctNovJan",
            ]
        }

        # Define method inputs.
        # Includes default seasons.
        df = pd.DataFrame(
            data=np.array(
                [
                    (2000, "DJF", 1),
                    (2000, "DJF", 2),
                    (2000, "MAM", 3),
                    (2000, "MAM", 4),
                    (2000, "MAM", 5),
                    (2000, "JJA", 6),
                    (2000, "JJA", 7),
                    (2000, "JJA", 8),
                    (2000, "SON", 9),
                    (2000, "SON", 10),
                    (2000, "SON", 11),
                    (2000, "DJF", 12),
                ],
                dtype=object,
            ),
            columns=["year", "season", "month"],
        )

        # Compare result of the method against the expected.
        result = ds.temporal._map_custom_seasons(df)
        expected = pd.DataFrame(
            data=np.array(
                [
                    (2000, "OctNovJan", 1),
                    (2000, "DecFebMar", 2),
                    (2000, "DecFebMar", 3),
                    (2000, "AprMayJun", 4),
                    (2000, "AprMayJun", 5),
                    (2000, "AprMayJun", 6),
                    (2000, "JulAugSep", 7),
                    (2000, "JulAugSep", 8),
                    (2000, "JulAugSep", 9),
                    (2000, "OctNovJan", 10),
                    (2000, "OctNovJan", 11),
                    (2000, "DecFebMar", 12),
                ],
                dtype=object,
            ),
            columns=["year", "season", "month"],
        )
        assert result.equals(expected)


class TestShiftDecembers:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = generate_dataset(cf_compliant=True, has_bounds=True)

    def test_decembers_shift_for_all_years(self):
        ds = self.ds.copy()

        # Define method inputs.
        df = pd.DataFrame(
            data=np.array(
                [
                    (2000, "DJF", 1),
                    (2000, "DJF", 2),
                    (2000, "DJF", 12),
                    (2001, "DJF", 1),
                    (2001, "DJF", 2),
                    (2001, "DJF", 12),
                ],
                dtype=object,
            ),
            columns=["year", "season", "month"],
        )

        # Compare result of the method against the expected.
        result = ds.temporal._shift_decembers(df)
        expected = pd.DataFrame(
            data=np.array(
                [
                    (2000, "DJF", 1),
                    (2000, "DJF", 2),
                    (2001, "DJF", 12),
                    (2001, "DJF", 1),
                    (2001, "DJF", 2),
                    (2002, "DJF", 12),
                ],
                dtype=object,
            ),
            columns=["year", "season", "month"],
        )

        assert result.equals(expected)


class TestDropObsoleteColumns:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = generate_dataset(cf_compliant=True, has_bounds=True)

    def test_drops_month_col_for_time_series_operations(self):
        ds = self.ds.copy()

        # Set object attrs required to test the method.
        ds.temporal._mode = "time_series"

        # Define method inputs.
        df = pd.DataFrame(columns=["year", "season", "month"])

        # Compare result of the method against the expected.
        result = ds.temporal._drop_obsolete_columns(df)
        expected = pd.DataFrame(columns=["year", "season"])

        assert result.equals(expected)

    def test_drops_year_and_month_cols_for_climatology_and_departure_operations(self):
        ds = self.ds.copy()

        # Set object attrs required to test the method.
        ds.temporal._mode = "climatology"

        # Define method inputs.
        df = pd.DataFrame(columns=["year", "season", "month"])

        # Compare result of the method against the expected.
        result = ds.temporal._drop_obsolete_columns(df)
        expected = pd.DataFrame(columns=["season"])

        assert result.equals(expected)

    def test_raises_error_with_unsupported_operation(self):
        ds = self.ds.copy()

        # Set object attrs required to test the method.
        ds.temporal._mode = "unsupported_operation"

        df = pd.DataFrame(columns=["year", "season", "month"])
        with pytest.raises(ValueError):
            ds.temporal._drop_obsolete_columns(df)


class TestCalculateWeights:
    class TestClimatology:
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
            ds.temporal.season_config = {"dec_mode": "DJF"}
            ds.temporal._time_grouped = xr.DataArray(
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
            result = ds.temporal._get_weights(ds["ts"])
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
            ds.temporal.season_config = {"dec_mode": "JDF"}
            ds.temporal._time_grouped = xr.DataArray(
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
            result = ds.temporal._get_weights(ds["ts"])
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
            ds.temporal.season_config = {"dec_mode": "DJF"}
            ds.temporal._time_grouped = xr.DataArray(
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
            result = ds.temporal._get_weights(self.ds["ts"])
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
            ds.temporal._season_config = {
                "dec_mode": "DJF",
                "drop_incomplete_djf": True,
            }
            ds.temporal._time_grouped = xr.DataArray(
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
            result = ds.temporal._get_weights(self.ds["ts"])
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

    class TestTimeSeries:
        @pytest.fixture(autouse=True)
        def setup(self):
            self.ds: xr.Dataset = generate_dataset(cf_compliant=True, has_bounds=True)

        def test_weights_for_yearly_averages(self):
            ds = self.ds.copy()

            # Set object attrs required to test the method.
            ds.temporal._time_bounds = ds.time_bnds.copy()
            ds.temporal._mode = "time_series"
            ds.temporal._freq = "year"
            ds.temporal._weighted = "True"
            ds.temporal._time_grouped = xr.DataArray(
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
            result = ds.temporal._get_weights(self.ds["ts"])
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
            ds.temporal._mode = "time_series"
            ds.temporal._freq = "month"
            ds.temporal._weighted = "True"
            ds.temporal._time_grouped = xr.DataArray(
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
                        "20012-12-01T00:00:00.000000000",
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
            result = ds.temporal._get_weights(self.ds["ts"])
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
            ds.temporal._mode = "time_series"
            ds.temporal._freq = "season"
            ds.temporal._weighted = "True"
            ds.temporal.season_config = {"dec_mode": "DJF"}
            ds.temporal._time_grouped = xr.DataArray(
                name="year_season",
                data=np.array(
                    [
                        "2000-04-01T00:00:00.000000000",
                        "2000-04-01T00:00:00.000000000",
                        "2000-04-01T00:00:00.000000000",
                        "2000-07-01T00:00:00.000000000",
                        "2000-07-01T00:00:00.000000000",
                        "2000-07-01T00:00:00.000000000",
                        "2000-10-01T00:00:00.000000000",
                        "2000-10-01T00:00:00.000000000",
                        "2000-10-01T00:00:00.000000000",
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
            result = ds.temporal._get_weights(ds["ts"])
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

        def test_weighs_for_seasonal_averages_with_JFD(self):
            ds = self.ds.copy()

            # Set object attrs required to test the method.
            ds.temporal._time_bounds = ds.time_bnds.copy()
            ds.temporal._mode = "time_series"
            ds.temporal._freq = "season"
            ds.temporal._weighted = "True"
            ds.temporal.season_config = {"dec_mode": "JDF"}
            ds.temporal._time_grouped = xr.DataArray(
                name="year_season",
                data=np.array(
                    [
                        "2000-01-01T00:00:00.000000000",
                        "2000-01-01T00:00:00.000000000",
                        "2000-04-01T00:00:00.000000000",
                        "2000-04-01T00:00:00.000000000",
                        "2000-04-01T00:00:00.000000000",
                        "2000-07-01T00:00:00.000000000",
                        "2000-07-01T00:00:00.000000000",
                        "2000-07-01T00:00:00.000000000",
                        "2000-10-01T00:00:00.000000000",
                        "2000-10-01T00:00:00.000000000",
                        "2000-10-01T00:00:00.000000000",
                        # This month is included in the JFD season
                        "2000-01-01T00:00:00.000000000",
                        "2001-01-01T00:00:00.000000000",
                        "2001-01-01T00:00:00.000000000",
                        "2002-01-01T00:00:00.000000000",
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
            result = ds.temporal._get_weights(self.ds["ts"])
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
                    0.52542373,
                    0.47457627,
                    1.0,
                ]
            )
            assert np.allclose(result, expected)

        @pytest.mark.xfail
        def test_custom_season_time_series_weights(self):
            ds = self.ds.copy()

            # Set object attrs required to test the method.
            ds.temporal._time_bounds = ds.time_bnds.copy()
            ds.temporal._mode = "time_series"
            ds.temporal._freq = "season"
            ds.temporal._weighted = "True"
            ds.temporal._season_config = {
                "custom_seasons": [
                    "JanFebMar",
                    "AprMayJun",
                    "JulAugSep",
                    "OctNovDec",
                ]
            }

            ds.temporal._multiindex = pd.MultiIndex.from_tuples(
                [
                    (2000, "JanFebMar"),
                    (2000, "JanFebMar"),
                    (2000, "JanFebMar"),
                    (2000, "AprMayJun"),
                    (2000, "AprMayJun"),
                    (2000, "AprMayJun"),
                    (2000, "JulAugSep"),
                    (2000, "JulAugSep"),
                    (2000, "JulAugSep"),
                    (2000, "OctNovDec"),
                    (2000, "OctNovDec"),
                    (2000, "OctNovDec"),
                    (2001, "JanFebMar"),
                    (2001, "JanFebMar"),
                    (2002, "JanFebMar"),
                ],
                names=[(0, "year"), (1, "season")],
            )
            ds.temporal._multiindex_name = "year_season"

            # Compare result of the method against the expected.
            result = ds.temporal._get_weights(self.ds["ts"])
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
            ds.temporal._mode = "time_series"
            ds.temporal._freq = "daily"
            ds.temporal._weighted = "True"
            ds.temporal._time_grouped = xr.DataArray(
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
            result = ds.temporal._get_weights(self.ds["ts"])
            expected = np.ones(15)
            assert np.allclose(result, expected)

        def test_weights_for_hourly_averages(self):
            ds = self.ds.copy()

            # Set object attrs required to test the method.
            ds.temporal._time_bounds = ds.time_bnds.copy()
            ds.temporal._mode = "time_series"
            ds.temporal._freq = "hour"
            ds.temporal._weighted = "True"
            ds.temporal.season_config = {"dec_mode": "JDF"}
            ds.temporal._time_grouped = xr.DataArray(
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
            result = ds.temporal._get_weights(self.ds["ts"])
            expected = np.ones(15)
            assert np.allclose(result, expected)


class TestGroupByFreq:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = generate_dataset(cf_compliant=True, has_bounds=True)

    def test_groups_data_var_for_seasonal_averaging_with_JFD(self):
        ds = self.ds.copy()

        # Set object attrs required to test the method.
        time_grouped = xr.DataArray(
            name="year_season",
            data=np.array(
                [
                    "2000-01-01T00:00:00.000000000",
                    "2000-01-01T00:00:00.000000000",
                    "2000-04-01T00:00:00.000000000",
                    "2000-04-01T00:00:00.000000000",
                    "2000-04-01T00:00:00.000000000",
                    "2000-07-01T00:00:00.000000000",
                    "2000-07-01T00:00:00.000000000",
                    "2000-07-01T00:00:00.000000000",
                    "2000-10-01T00:00:00.000000000",
                    "2000-10-01T00:00:00.000000000",
                    "2000-10-01T00:00:00.000000000",
                    "2001-01-01T00:00:00.000000000",
                    "2001-01-01T00:00:00.000000000",
                    "2001-01-01T00:00:00.000000000",
                    "2002-01-01T00:00:00.000000000",
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
        ds.temporal._time_grouped = time_grouped

        ts = ds.ts.copy()
        expected = ts.copy()
        expected.coords["year_season"] = time_grouped
        expected = expected.groupby("year_season")
        result = ds.temporal._groupby_freq(ts)

        assert result.groups == expected.groups

    def test_groups_data_var_for_seasonal_climatology_with_DJF(self):
        ds = self.ds.copy()

        # Set object attrs required to test the method.
        time_grouped = xr.DataArray(
            name="season",
            data=np.array(
                [
                    # JFD
                    cftime.datetime(1, 1, 1),
                    cftime.datetime(1, 1, 1),
                    cftime.datetime(1, 1, 1),
                    # MAM
                    cftime.datetime(1, 3, 1),
                    cftime.datetime(1, 3, 1),
                    cftime.datetime(1, 3, 1),
                    # JJA
                    cftime.datetime(1, 7, 1),
                    cftime.datetime(1, 7, 1),
                    cftime.datetime(1, 7, 1),
                    # SON
                    cftime.datetime(1, 10, 1),
                    cftime.datetime(1, 10, 1),
                    cftime.datetime(1, 10, 1),
                    # JFD
                    cftime.datetime(1, 1, 1),
                    cftime.datetime(1, 1, 1),
                    cftime.datetime(1, 1, 1),
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
        ds.temporal._time_grouped = time_grouped

        ts = ds.ts.copy()
        expected = ts.copy()
        expected.coords["season"] = time_grouped
        expected = expected.groupby("season")
        result = ds.temporal._groupby_freq(ts)

        assert result.groups == expected.groups


class TestAddOperationAttributes:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = generate_dataset(cf_compliant=True, has_bounds=True)

    def test_adds_attrs_to_data_var_with_DJF(self):
        ds = self.ds.copy()

        # Set object attrs required to test the method.
        ds.temporal._mode = "climatology"
        ds.temporal._freq = "season"
        ds.temporal._weighted = True
        ds.temporal._center_times = True
        ds.temporal._season_config = {
            "dec_mode": "DJF",
            "drop_incomplete_djf": "True",
        }
        ds.temporal._time_grouped = xr.DataArray(
            name="year_season",
            data=np.array(
                [
                    "2000-01-01T00:00:00.000000000",
                    "2000-01-01T00:00:00.000000000",
                    "2000-04-01T00:00:00.000000000",
                    "2000-04-01T00:00:00.000000000",
                    "2000-04-01T00:00:00.000000000",
                    "2000-07-01T00:00:00.000000000",
                    "2000-07-01T00:00:00.000000000",
                    "2000-07-01T00:00:00.000000000",
                    "2000-10-01T00:00:00.000000000",
                    "2000-10-01T00:00:00.000000000",
                    "2000-10-01T00:00:00.000000000",
                    "2001-01-01T00:00:00.000000000",
                    "2001-01-01T00:00:00.000000000",
                    "2001-01-01T00:00:00.000000000",
                    "2002-01-01T00:00:00.000000000",
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
        result = ds.temporal._add_operation_attrs(ds.ts)
        expected = ds.ts.copy()
        expected.attrs.update(
            {
                "operation": "temporal_avg",
                "mode": ds.temporal._mode,
                "freq": ds.temporal._freq,
                "groupby": "year_season",
                "weighted": "True",
                "center_times": "True",
                "dec_mode": "DJF",
                "drop_incomplete_djf": "True",
            }
        )

        assert result.identical(expected)

    def test_adds_attrs_to_data_var_with_custom_seasons(self):
        ds = self.ds.copy()

        # Set object attrs required to test the method.
        ds.temporal._mode = "climatology"
        ds.temporal._freq = "season"
        ds.temporal._weighted = True
        ds.temporal._center_times = True
        ds.temporal._season_config = {
            "custom_seasons": [
                "JanFebMar",
                "AprMayJun",
                "JulAugSep",
                "OctNovDec",
            ]
        }
        ds.temporal._time_grouped = xr.DataArray(
            name="year_season",
            data=np.array(
                [
                    "2000-01-01T00:00:00.000000000",
                    "2000-01-01T00:00:00.000000000",
                    "2000-04-01T00:00:00.000000000",
                    "2000-04-01T00:00:00.000000000",
                    "2000-04-01T00:00:00.000000000",
                    "2000-07-01T00:00:00.000000000",
                    "2000-07-01T00:00:00.000000000",
                    "2000-07-01T00:00:00.000000000",
                    "2000-10-01T00:00:00.000000000",
                    "2000-10-01T00:00:00.000000000",
                    "2000-10-01T00:00:00.000000000",
                    "2001-01-01T00:00:00.000000000",
                    "2001-01-01T00:00:00.000000000",
                    "2001-01-01T00:00:00.000000000",
                    "2002-01-01T00:00:00.000000000",
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
        result = ds.temporal._add_operation_attrs(ds.ts)
        expected = ds.ts.copy()
        expected.attrs.update(
            {
                "operation": "temporal_avg",
                "mode": ds.temporal._mode,
                "freq": ds.temporal._freq,
                "groupby": "year_season",
                "weighted": "True",
                "center_times": "True",
                "custom_seasons": ["JanFebMar", "AprMayJun", "JulAugSep", "OctNovDec"],
            }
        )

        assert result.identical(expected)
