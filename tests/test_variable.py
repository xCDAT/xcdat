import pytest

from tests.fixtures import generate_dataset
from xcdat.variable import copy_variable, open_variable


class TestOpenVariable:
    def test_raises_error_if_variable_does_not_exist(self):
        ds = generate_dataset(cf_compliant=True, has_bounds=False)

        with pytest.raises(KeyError):
            open_variable(ds, "invalid_var")

    def test_raises_error_if_bounds_are_missing_for_coordinates(self):
        ds = generate_dataset(cf_compliant=True, has_bounds=True)

        # Need to add CF attributes in order for cf_xarray to properly
        # identify coordinates and bounds.
        ds = ds.drop_vars("lat_bnds")

        # By adding bounds to the parent dataset, it will initiate copying bounds
        # and find that bounds are missing for the other coords (lon and time).
        with pytest.raises(ValueError):
            open_variable(ds, "ts")

    def test_returns_variable_with_bounds(self):
        ds = generate_dataset(cf_compliant=True, has_bounds=True)
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

        ts = open_variable(ds, "ts")
        result = ts.bounds.bounds

        for key in expected.keys():
            assert result[key].identical(expected[key])


class TestCopyVariable:
    def test_returns_copy_of_variable_with_bounds(self):
        ds = generate_dataset(cf_compliant=True, has_bounds=True)
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

        ts = open_variable(ds, "ts")
        ts_bounds = ts.bounds.bounds

        for key in expected.keys():
            assert ts_bounds[key].identical(expected[key])

        ts_copy = copy_variable(ts)
        ts_copy_bounds = ts_copy.bounds.bounds

        for key in expected.keys():
            assert ts_copy_bounds[key].identical(ts_bounds[key])
