import pytest

from tests.fixtures import generate_dataset, lat_bnds, ts_with_bnds
from xcdat.variable import open_variable


class TestOpenVariable:
    def test_raises_error_if_variable_does_not_exist(self):
        ds = generate_dataset(has_bounds=False)

        with pytest.raises(KeyError):
            open_variable(ds, "invalid_var")

    def test_raises_error_if_bounds_dim_is_missing(self):
        ds = generate_dataset(has_bounds=False)

        with pytest.raises(KeyError):
            open_variable(ds, "ts")

    def test_raises_error_if_bounds_are_missing_for_coordinates(self):
        ds = generate_dataset(has_bounds=False)

        # By adding bounds to the parent dataset, it will initiate copying bounds
        # and find that bounds are missing for the other coords (lon and time).
        ds["lat_bnds"] = lat_bnds.copy()
        with pytest.raises(ValueError):
            open_variable(ds, "ts")

    def test_returns_variable_with_bounds(self):
        ds = generate_dataset(has_bounds=True)
        ds.lat.attrs["bounds"] = "lat_bnds"
        ds.lon.attrs["bounds"] = "lon_bnds"
        ds.time.attrs["bounds"] = "time_bnds"

        ts_expected = ts_with_bnds.copy()

        ts_result = open_variable(ds, "ts")
        assert ts_result.identical(ts_expected)
