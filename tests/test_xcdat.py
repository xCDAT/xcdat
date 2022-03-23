import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tests.fixtures import generate_dataset, lat_bnds
from xcdat.xcdat import XCDATAccessor


class TestXCDATAccessor:
    # NOTE: We don't have to perform in-depth testing of XCDATAccessor methods.
    # These methods are already tested in other classes.
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = generate_dataset(cf_compliant=True, has_bounds=False)
        self.ds_with_bnds = generate_dataset(cf_compliant=True, has_bounds=True)

    def test_init(self):
        result = XCDATAccessor(self.ds)

        assert result._dataset.identical(self.ds)

    def test_decorator_call(self):

        self.ds.xcdat._dataset.identical(self.ds)


class TestSpatialAvgAccessor:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = generate_dataset(cf_compliant=True, has_bounds=False)
        self.ds_with_bnds = generate_dataset(cf_compliant=True, has_bounds=True)

    def test_weighted_spatial_average_for_lat_and_lon_region(self):
        ds = self.ds_with_bnds.copy()

        # Limit to just 3 data points to simplify testing.
        ds = ds.isel(time=slice(None, 3))

        # Change the value of the first element so that it is easier to identify
        # changes in the output.
        ds["ts"].data[0] = np.full((4, 4), 2.25)

        result = ds.xcdat.spatial_avg(
            "ts", axis=["lat", "lon"], lat_bounds=(-5.0, 5), lon_bounds=(-170, -120.1)
        )
        expected = ds.copy()
        expected["ts"] = xr.DataArray(
            data=np.array([2.25, 1.0, 1.0]),
            coords={"time": expected.time},
            dims="time",
        )

        assert result.identical(expected)


class TestBoundsAccessor:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = generate_dataset(cf_compliant=True, has_bounds=False)
        self.ds_with_bnds = generate_dataset(cf_compliant=True, has_bounds=True)

    def test_bounds_property_returns_expected(self):
        ds = self.ds_with_bnds.copy()
        expected = {
            "T": ds.time_bnds,
            "X": ds.lon_bnds,
            "Y": ds.lat_bnds,
            "lat": ds.lat_bnds,
            "latitude": ds.lat_bnds,
            "lon": ds.lon_bnds,
            "longitude": ds.lon_bnds,
            "time": ds.time_bnds,
        }

        result = ds.xcdat.bounds

        for key in expected.keys():
            assert result[key].identical(expected[key])

    def test_add_missing_bounds_returns_expected(self):
        ds = self.ds_with_bnds.copy()
        ds = ds.drop_vars(["lat_bnds", "lon_bnds"])

        result = ds.xcdat.add_missing_bounds()
        assert result.identical(self.ds_with_bnds)

    def test_get_bounds_returns_expected(self):
        ds = self.ds_with_bnds.copy()
        lat_bnds = ds.xcdat.get_bounds("lat")
        assert lat_bnds.identical(ds.lat_bnds)

        lon_bnds = ds.xcdat.get_bounds("lon")
        assert lon_bnds.identical(ds.lon_bnds)
        assert lon_bnds.is_generated

    def test_add_bounds_returns_expected(self):
        ds = self.ds.copy()
        ds = ds.xcdat.add_bounds("lat")

        assert ds.lat_bnds.equals(lat_bnds)
        assert ds.lat_bnds.is_generated


class TestTemporalAccessor:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = generate_dataset(cf_compliant=True, has_bounds=True)

    def test_weighted_annual_avg(self):
        ds = self.ds.copy()

        result = ds.xcdat.temporal_avg("ts", "time_series", "year")
        expected = ds.copy()
        expected["ts_original"] = ds.ts.copy()
        expected["ts"] = xr.DataArray(
            name="ts",
            data=np.ones((2, 4, 4)),
            coords={
                "lat": self.ds.lat,
                "lon": self.ds.lon,
                "year": pd.MultiIndex.from_tuples(
                    [(2000,), (2001,)],
                ),
            },
            dims=["year", "lat", "lon"],
            attrs={
                "operation": "temporal_avg",
                "mode": "time_series",
                "freq": "year",
                "groupby": "year",
                "weighted": "True",
                "center_times": "False",
            },
        )

        # For some reason, there is a floating point difference between both
        # for ts so we have to use floating point comparison
        xr.testing.assert_allclose(result, expected)
        assert result.ts.attrs == expected.ts.attrs

    def test_weighted_seasonal_departure_with_DJF(self):
        # Create a post-climatology dataset.
        ds = self.ds.copy()
        # Drop incomplete DJF seasons
        ds = ds.isel(time=slice(2, -1))
        ds["ts_original"] = ds.ts.copy()
        ds["ts"] = xr.DataArray(
            data=np.ones((4, 4, 4)),
            coords={
                "lat": ds.lat,
                "lon": ds.lon,
                "season": pd.MultiIndex.from_arrays([["JJA", "MAM", "SON", "DJF"]]),
            },
            dims=["season", "lat", "lon"],
            attrs={
                "operation": "temporal_avg",
                "mode": "climatology",
                "freq": "season",
                "weighted": "True",
                "center_times": "False",
                "groupby": "season",
                "dec_mode": "DJF",
                "drop_incomplete_djf": "False",
            },
        )

        # Run climatology on the post-climatology dataset.
        result = ds.xcdat.departures("ts")

        # Create an expected post-departure dataset.
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
                "drop_incomplete_djf": "False",
            },
        )
        assert result.identical(expected)

    def test_gets_time_as_the_midpoint_between_time_bounds(self):
        ds = self.ds.copy()

        # Compare result of the method against the expected.
        result = ds.xcdat.center_times()
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
        expected["time"] = xr.DataArray(
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
        expected_time_bounds = ds.time_bnds.copy()
        expected_time_bounds.time.data[:] = expected_time_data

        assert result.identical(expected)
