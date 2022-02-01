import sys
from unittest import mock

import numpy as np
import pytest
import xarray as xr

from tests import fixtures
from xcdat.regridder import accessor, base, grid, regrid2, xesmf

np.set_printoptions(threshold=sys.maxsize, suppress=True)


class TestRegrid2Regridder:
    @pytest.fixture(autouse=True)
    def setup(self):
        coarse = np.arange(-90, 90.1, 30)
        self.coarse_bnds = np.zeros((len(coarse) - 1, 2))
        self.coarse_bnds[:, 0] = coarse[:-1]
        self.coarse_bnds[:, 1] = coarse[1:]

        coarse_rev = np.flip(coarse.copy())
        self.coarse_rev_bnds = np.zeros((len(coarse) - 1, 2))
        self.coarse_rev_bnds[:, 0] = coarse_rev[:-1]
        self.coarse_rev_bnds[:, 1] = coarse_rev[1:]

        fine = np.arange(-90, 90.1, 10)
        self.fine_bnds = np.zeros((len(fine) - 1, 2))
        self.fine_bnds[:, 0] = fine[:-1]
        self.fine_bnds[:, 1] = fine[1:]

        self.expected_weights = np.array(
            [
                0.01519224698779198,
                0.0451151322262997,
                0.07366721700146972,
                0.09998096066546058,
                0.12325683343243876,
                0.1427876096865393,
                0.15797985667433123,
                0.16837196565873838,
                0.17364817766693033,
                0.17364817766693033,
                0.16837196565873838,
                0.15797985667433123,
                0.1427876096865393,
                0.12325683343243876,
                0.09998096066546058,
                0.07366721700146972,
                0.0451151322262997,
                0.01519224698779198,
            ]
        )

    def test_extract_bounds(self):
        lower, upper = regrid2.extract_bounds(self.coarse_bnds)

        np.testing.assert_allclose(lower, self.coarse_bnds[:, 0])
        np.testing.assert_allclose(upper, self.coarse_bnds[:, 1])

    def test_extract_bounds_reverse(self):
        lower, upper = regrid2.extract_bounds(self.coarse_rev_bnds)

        np.testing.assert_allclose(lower, self.coarse_rev_bnds[:, 1])
        np.testing.assert_allclose(upper, self.coarse_rev_bnds[:, 0])

    def test_map_fine_to_coarse(self):
        mapping, weights = regrid2.map_latitude(self.fine_bnds, self.coarse_bnds)

        expected_mapping = np.arange(18).reshape((6, 3))

        np.testing.assert_allclose(mapping, expected_mapping)

        np.testing.assert_allclose(weights, self.expected_weights.reshape((6, 3)))

    def test_map_coarse_to_fine(self):
        mapping, weights = regrid2.map_latitude(self.coarse_bnds, self.fine_bnds)

        expected_mapping = np.array(
            [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5]
        )

        np.testing.assert_allclose(mapping, expected_mapping.reshape((18, 1)))

        np.testing.assert_allclose(weights, self.expected_weights.reshape((18, 1)))


# TODO improve testing
class TestXESMFRegridder:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = fixtures.generate_dataset(True, True)

    def test_invalid_method(self):
        ds = self.ds.copy()
        ds.attrs["xcdat_infer"] = "ts"

        new_grid = grid.create_uniform_grid(-90, 90, 4.0, -180, 180, 5.0)

        with pytest.raises(ValueError):
            xesmf.XESMFRegridder(ds, new_grid, "bad value")

    def test_invalid_extra_method(self):
        ds = self.ds.copy()
        ds.attrs["xcdat_infer"] = "ts"

        new_grid = grid.create_uniform_grid(-90, 90, 4.0, -180, 180, 5.0)

        with pytest.raises(ValueError):
            xesmf.XESMFRegridder(ds, new_grid, "bilinear", extrap_method="bad value")


class TestGrid:
    def test_uniform_grid(self):
        new_grid = grid.create_uniform_grid(-90, 90, 4.0, -180, 180, 5.0)

        assert new_grid.lat[0] == -90.0
        assert new_grid.lat[-1] == 86
        assert new_grid.lat.units == "degrees_north"

        assert new_grid.lon[0] == -180
        assert new_grid.lon[-1] == 175
        assert new_grid.lon.units == "degrees_east"

    def test_gaussian_grid(self):
        new_grid = grid.create_gaussian_grid(32)

        assert new_grid.lat.shape == (32,)
        assert new_grid.lon.shape == (64,)

    def test_global_mean_grid(self):
        new_grid = grid.create_gaussian_grid(32)

        mean_grid = grid.create_global_mean_grid(new_grid)

        assert mean_grid.cf["lat"].data == np.array(
            [
                0.0,
            ]
        )
        assert mean_grid.cf["lon"].data == np.array(
            [
                177.1875,
            ]
        )

    def test_zonal_grid(self):
        new_grid = xr.Dataset(
            coords={
                "lat": xr.DataArray(
                    name="lat",
                    data=np.array([-80, -40, 0, 40, 80]),
                    dims=["lat"],
                    attrs={
                        "units": "degrees_north",
                        "axis": "Y",
                        "bounds": "lat_bnds",
                    },
                ),
                "lon": xr.DataArray(
                    name="lon",
                    data=np.array([-160, -80, 0, 80, 160]),
                    dims=["lon"],
                    attrs={
                        "units": "degrees_east",
                        "axis": "X",
                        "bounds": "lon_bnds",
                    },
                ),
            },
            data_vars={
                "lat_bnds": xr.DataArray(
                    name="lat_bnds",
                    data=np.array(
                        [[-90, -60], [-60, -20], [-20, 20], [20, 60], [60, 90]]
                    ),
                    dims=["lat", "bnds"],
                ),
                "lon_bnds": xr.DataArray(
                    name="lon_bnds",
                    data=np.array(
                        [[-180, -120], [-120, -40], [-40, 40], [40, 120], [120, 180]]
                    ),
                    dims=["lon", "bnds"],
                ),
            },
        )

        zonal_grid = grid.create_zonal_grid(new_grid)

        expected_grid = xr.Dataset(
            coords={
                "lat": xr.DataArray(
                    name="lat",
                    data=np.array([-80, -40, 0, 40, 80]),
                    dims=["lat"],
                    attrs={
                        "units": "degrees_north",
                        "axis": "Y",
                        "bounds": "lat_bnds",
                    },
                ),
                "lon": xr.DataArray(
                    name="lon",
                    data=np.array([0.0]),
                    dims=["lon"],
                    attrs={
                        "units": "degrees_east",
                        "axis": "X",
                        "bounds": "lon_bnds",
                    },
                ),
            },
            data_vars={
                "lat_bnds": xr.DataArray(
                    name="lat_bnds",
                    data=np.array(
                        [[-90, -60], [-60, -20], [-20, 20], [20, 60], [60, 90]]
                    ),
                    dims=["lat", "bnds"],
                ),
                "lon_bnds": xr.DataArray(
                    name="lon_bnds",
                    data=np.array([[-180, 180]]),
                    dims=["lon", "bnds"],
                ),
            },
        )

        assert zonal_grid.identical(expected_grid)


class TestAccessor:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.data = mock.MagicMock()
        self.ac = accessor.DatasetRegridderAccessor(self.data)

    def test_valid_tool(self):
        mock_regridder = mock.MagicMock()
        mock_regridder.return_value.regrid.return_value = "output data"

        mock_data = mock.MagicMock()

        with mock.patch.dict(accessor.REGRID_TOOLS, {"test": mock_regridder}):
            output = self.ac.regrid("ts", mock_data, "test")

        assert output == "output data"

        mock_regridder.return_value.regrid.assert_called_with("ts", self.data)

    def test_invalid_tool(self):
        with pytest.raises(
            ValueError, match=r"Tool 'test' does not exist, valid choices"
        ):
            self.ac.regrid("ts", mock.MagicMock(), "test")


class TestBase:
    def test_regridder_implementation(self):
        class NewRegridder(base.BaseRegridder):
            def __init__(self, src_grid, dst_grid, **options):
                super().__init__(src_grid, dst_grid, **options)

            def regrid(self, data_var, ds):
                return ds

        regridder = NewRegridder(mock.MagicMock(), mock.MagicMock())

        assert regridder is not None

        ds_in = mock.MagicMock()

        ds_out = regridder.regrid("ts", ds_in)

        assert ds_in == ds_out
