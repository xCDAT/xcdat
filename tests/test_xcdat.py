import numpy as np
import xarray as xr

from tests.fixtures import generate_dataset, lat_bnds
from xcdat.regridder import grid
from xcdat.xcdat import XCDATAccessor


class TestXCDATAccessor:
    # NOTE: We don't have to test this class in-depth because its methods
    # are abstracted public methods from other classes, which already has more
    # comprehensive testing.
    def setup(self):
        self.ds = generate_dataset(cf_compliant=True, has_bounds=False)
        self.ds_with_bnds = generate_dataset(cf_compliant=True, has_bounds=True)

    def test_init(self):
        result = XCDATAccessor(self.ds)

        assert result._dataset.identical(self.ds)

    def test_decorator_call(self):

        self.ds.xcdat._dataset.identical(self.ds)

    def test_regrid(self):
        ds = self.ds.copy()

        out_grid = grid.create_uniform_grid(-90, 90, 4.0, -180, 180, 5.0)

        result = ds.xcdat.regrid(out_grid, "xesmf", method="bilinear")

        assert result.ts.shape == (12, 45, 72)

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
