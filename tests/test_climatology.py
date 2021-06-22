import numpy as np
import pytest
import xarray as xr

from tests.fixtures import (
    generate_dataset,
    lat,
    lat_bnds,
    lon,
    lon_bnds,
    time,
    time_bnds,
)
from xcdat.climatology import (
    _calculate_weights,
    _get_months_lengths,
    _group_data,
    climatology,
    departure,
)


class TestClimatology:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds: xr.Dataset = generate_dataset(bounds=True)

        # Unique seasons for a year, which is used for grouped data
        self.season_unique = xr.DataArray(
            data=np.array(["DJF", "JJA", "MAM", "SON"], dtype="object"),
            dims=["season"],
        )

    def test_climatology_throws_error_with_incorrect_period_arg(self):
        with pytest.raises(ValueError):
            climatology(self.ds, "incorrect_period")  # type: ignore

    def test_climatology_throws_error_without_time_dimension(self):
        ds = self.ds.copy()
        ds = ds.drop_dims("time")
        with pytest.raises(KeyError):
            climatology(ds, "season")

    def test_climatology_weighted(self):
        expected = xr.Dataset(
            coords={"time": time, "lat": lat, "lon": lon, "season": self.season_unique},
            data_vars={
                "ts": xr.DataArray(
                    data=np.ones((4, 4, 4)),
                    coords={"lat": lat, "lon": lon, "season": self.season_unique},
                    dims=["season", "lat", "lon"],
                ),
                "lat_bnds": lat_bnds,
                "lon_bnds": lon_bnds,
                "time_bnds": time_bnds,
            },
            attrs={
                "calculation_info": {
                    "type": "climatology",
                    "frequency": "season",
                    "is_weighted": True,
                },
            },
        )

        result = climatology(self.ds, "season")
        assert result.identical(expected)

    def test_climatology_unweighted(self):
        expected = xr.Dataset(
            coords={"time": time, "lat": lat, "lon": lon, "season": self.season_unique},
            data_vars={
                "ts": xr.DataArray(
                    data=np.full((4, 4, 4), 3.0),
                    coords={"lat": lat, "lon": lon, "season": self.season_unique},
                    dims=["season", "lat", "lon"],
                ),
                "lat_bnds": lat_bnds,
                "lon_bnds": lon_bnds,
                "time_bnds": time_bnds,
            },
            attrs={
                "calculation_info": {
                    "type": "climatology",
                    "frequency": "season",
                    "is_weighted": False,
                },
            },
        )
        result = climatology(self.ds, "season", is_weighted=False)
        assert result.identical(expected)


class TestDeparture:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds: xr.Dataset = generate_dataset(bounds=True)

        self.season = xr.DataArray(
            data=np.array(["DJF", "JJA", "MAM", "SON"], dtype="object"),
            dims=["season"],
        )

    def test_departure_weighted(self):
        # Create a copy to update variables
        ds = self.ds.copy()

        ds_climatology_season = xr.Dataset(
            coords={"time": time, "lat": lat, "lon": lon, "season": self.season},
            data_vars={
                "ts": xr.DataArray(
                    data=np.ones((4, 4, 4)),
                    coords={"lat": lat, "lon": lon, "season": self.season},
                    dims=["season", "lat", "lon"],
                ),
                "lat_bnds": lat_bnds,
                "lon_bnds": lon_bnds,
                "time_bnds": time_bnds,
            },
            attrs={
                "calculation_info": {
                    "type": "departure",
                    "frequency": "season",
                    "is_weighted": True,
                },
            },
        )

        # Perform calculation to get expected result
        expected = ds.copy()
        expected = expected.assign_coords(
            {
                "season": xr.DataArray(
                    data=np.array(["DJF", "JJA", "MAM", "SON"]),
                    coords={"season": np.array(["DJF", "JJA", "MAM", "SON"])},
                    dims=["season"],
                ).astype(object),
            }
        )
        expected["ts"] = xr.DataArray(
            data=np.zeros((4, 4, 4)),
            coords={
                "lat": expected.lat,
                "lon": expected.lon,
                "season": expected.season,
            },
            dims=["season", "lat", "lon"],
        )
        expected.attrs.update(
            {
                "calculation_info": {
                    "type": "departure",
                    "frequency": "season",
                    "is_weighted": True,
                },
            }
        )

        # Check that the expected and resulting departure datasets are identical.
        result = departure(ds, ds_climatology_season)
        assert result.identical(expected)

        # Check that the original and result datasets are identical without
        # departure information
        expected = expected.drop_vars(["ts", "season"])
        result = result.drop_vars(["ts", "season"])
        del expected.attrs["calculation_info"]
        del result.attrs["calculation_info"]

        assert result.identical(expected)

    def test_departure_unweighted(self):
        ds = self.ds.copy()

        ds_climatology_season = xr.Dataset(
            coords={
                "time": time,
                "lat": lat,
                "lon": lon,
                "season": self.season,
            },
            data_vars={
                "ts": xr.DataArray(
                    data=np.full((4, 4, 4), 3),
                    coords={"lat": lat, "lon": lon, "season": self.season},
                    dims=["season", "lat", "lon"],
                ),
                "lat_bnds": lat_bnds,
                "lon_bnds": lon_bnds,
                "time_bnds": time_bnds,
            },
            attrs={
                "calculation_info": {
                    "type": "departure",
                    "frequency": "season",
                    "is_weighted": False,
                },
            },
        )

        # Check that the expected and resulting departure datasets are the same.
        expected = ds.copy()
        expected = expected.assign_coords(
            {
                "season": xr.DataArray(
                    data=np.array(["DJF", "JJA", "MAM", "SON"]),
                    coords={"season": np.array(["DJF", "JJA", "MAM", "SON"])},
                    dims=["season"],
                ).astype(object),
            }
        )
        expected["ts"] = xr.DataArray(
            data=np.zeros((4, 4, 4)),
            coords={
                "lat": expected.lat,
                "lon": expected.lon,
                "season": expected.season,
            },
            dims=["season", "lat", "lon"],
        )
        expected.attrs.update(
            {
                "calculation_info": {
                    "type": "departure",
                    "frequency": "season",
                    "is_weighted": False,
                },
            }
        )
        result = departure(ds, ds_climatology_season)
        assert result.identical(expected)

        # Check that the original and result datasets are identical without
        # departure information
        expected = expected.drop_vars(["ts", "season"])
        result = result.drop_vars(["ts", "season"])
        del expected.attrs["calculation_info"]
        del result.attrs["calculation_info"]

        assert result.identical(expected)


class TestGroupData:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds: xr.Dataset = generate_dataset(bounds=True)
        self.ds.attrs.update({"calculation_type": "climatology"})

    def test__group_data_weighted_by_season(self):
        ds = self.ds.copy()

        # Check non-bounds variables were properly grouped and averaged
        result = _group_data(ds.copy(), "climatology", "season", is_weighted=True)
        assert result.attrs["calculation_info"]["type"] == "climatology"
        assert result.attrs["calculation_info"]["frequency"] == "season"
        assert result.attrs["calculation_info"]["is_weighted"] is True

        ts_expected = np.ones((4, 4, 4))
        ts_result = result.ts.data
        assert np.allclose(ts_result, ts_expected)

        # Check that all of the existing coordinates and bounds variables are the same
        ds = ds.drop_vars("ts")
        result = result.drop_dims("season")
        del result.attrs["calculation_info"]
        assert ds.identical(result)

    def test__group_data_unweighted_by_season(self):
        ds = self.ds.copy()

        # Check non-bounds variables were properly grouped and averaged
        result = _group_data(ds.copy(), "climatology", "season", is_weighted=False)
        assert result.attrs["calculation_info"]["type"] == "climatology"
        assert result.attrs["calculation_info"]["frequency"] == "season"
        assert result.attrs["calculation_info"]["is_weighted"] is False

        ts_expected = np.full((4, 4, 4), 3)
        ts_result = result.ts.data
        assert np.allclose(ts_result, ts_expected)

        # Check that all of the existing coordinates and bounds variables are the same
        ds = ds.drop_vars("ts")
        result = result.drop_dims("season")
        del result.attrs["calculation_info"]
        assert ds.identical(result)


class TestCalculateWeights:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds: xr.Dataset = generate_dataset(bounds=True)

    def test__calculate_weights_monthly(self):
        expected = np.array(
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        )
        result = _calculate_weights(self.ds, "time.month")
        assert np.allclose(result, expected)

    def test__calculate_weights_seasonal(self):
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
            ],
        )
        result = _calculate_weights(self.ds, "time.season")
        assert np.allclose(result, expected)

    def test__calculate_weights_annual(self):
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
            ]
        )

        result = _calculate_weights(self.ds, "time.year")
        assert np.allclose(result, expected)

    def test__get_months_lengths_with_time_bounds(self):
        expected = xr.DataArray(
            name="days",
            data=np.array([31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]),
            coords={"time": self.ds.time},
            dims=["time"],
        )
        result = _get_months_lengths(self.ds)

        assert result.identical(expected)

    def test__get_months_lengths_without_time_bounds(self):
        ds = self.ds.copy()
        ds = ds.drop_vars({"time_bnds"})

        expected = xr.DataArray(
            name="days_in_month",
            data=np.array([31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]),
            coords={"time": self.ds.time},
            dims=["time"],
        )

        result = _get_months_lengths(ds)
        assert result.identical(expected)
