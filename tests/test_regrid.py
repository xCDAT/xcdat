from unittest import mock

import pytest

from xcdat.regridder import accessor
from xcdat.regridder import base
from xcdat.regridder import grid
from xcdat.regridder import xesmf
from tests import fixtures


# TODO improve testing
class TestXESMFRegridder:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = fixtures.generate_dataset(True, True)

    def test_inferred_data_var(self):
        ds = self.ds.copy()
        ds.attrs["xcdat_infer"] = "ts"

        new_grid = grid.create_uniform_grid(-90, 90, 4.0, -180, 180, 5.0)

        regridder = xesmf.XESMFRegridder(ds, new_grid, "bilinear")

        result = regridder.regrid(ds)

        assert result.ts.shape == (12, 45, 72)

    def test_non_inferred_data_var(self):
        ds = self.ds.copy()

        new_grid = grid.create_uniform_grid(-90, 90, 4.0, -180, 180, 5.0)

        regridder = xesmf.XESMFRegridder(ds, new_grid, "bilinear", "ts")

        result = regridder.regrid(ds)

        assert result.ts.shape == (12, 45, 72)

    def test_non_existant_data_var(self):
        ds = self.ds.copy()

        new_grid = grid.create_uniform_grid(-90, 90, 4.0, -180, 180, 5.0)

        with pytest.raises(KeyError, match="The data variable 'clt' does not exist in the dataset."):
            xesmf.XESMFRegridder(ds, new_grid, "bilinear", "clt")


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
        pass

    def test_global_mean_grid(self):
        pass

    def test_zonal_grid(self):
        pass


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
            output = self.ac.regrid(mock_data, "test", "conservative")

        assert output == "output data"

        mock_regridder.return_value.regrid.assert_called_with(
            self.data
        )

    def test_invalid_tool(self):
        with pytest.raises(ValueError, match=r"Tool 'test' does not exist, valid choices"):
            self.ac.regrid(mock.MagicMock(), "test", "conservative")


class TestBase:
    def test_regridder_implementation(self):
        class NewRegridder(base.BaseRegridder):
            def __init__(self, src_grid, dst_grid, **options):
                super().__init__(src_grid, dst_grid, **options)

            def regrid(self, ds):
                return ds

        regridder = NewRegridder(mock.MagicMock(), mock.MagicMock())

        assert regridder is not None

        ds_in = mock.MagicMock()

        ds_out = regridder.regrid(ds_in)

        assert ds_in == ds_out
