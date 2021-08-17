import pytest

from tests.fixtures import generate_dataset, ts_with_bnds_from_parent_cf
from xcdat.variable import open_variable


class TestOpenVariable:
    def test_raises_error_if_variable_does_not_exist(self):
        ds = generate_dataset(cf_compliant=True, has_bounds=False)

        with pytest.raises(KeyError):
            open_variable(ds, "invalid_var")

    def test_raises_error_if_bounds_dim_is_missing(self):
        ds = generate_dataset(cf_compliant=True, has_bounds=False)

        with pytest.raises(KeyError):
            open_variable(ds, "ts")

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
        ds.lat.attrs["bounds"] = "lat_bnds"
        ds.lon.attrs["bounds"] = "lon_bnds"
        ds.time.attrs["bounds"] = "time_bnds"

        ts_expected = ts_with_bnds_from_parent_cf.copy()

        ts_result = open_variable(ds, "ts")
        assert ts_result.identical(ts_expected)
