import pytest

from tests.fixtures import generate_dataset
from xcdat.variable import open_variable


class TestOpenVariable:
    def test_raises_error_if_variable_does_not_exist(self):
        ds = generate_dataset(has_bounds=False)

        with pytest.raises(KeyError):
            open_variable(ds, "invalid_var")

    def test_raises_error_if_bounds_dont_exist(self):
        ds = generate_dataset(has_bounds=False)

        with pytest.raises(ValueError):
            open_variable(ds, "ts")

    def test_bound_variables_are_identical_with_parent_dataset(self):
        ds = generate_dataset(has_bounds=True)
        ds.lat.attrs["bounds"] = "lat_bnds"
        ds.lon.attrs["bounds"] = "lon_bnds"
        ds.time.attrs["bounds"] = "time_bnds"

        ts = open_variable(ds, "ts")

        lat_bounds = ts.bounds.lat
        assert lat_bounds.identical(ds.lat_bnds)

        lon_bounds = ts.bounds.lon
        assert lon_bounds.identical(ds.lon_bnds)

        time_bounds = ts.bounds.time
        assert time_bounds.identical(ds.time_bnds)
