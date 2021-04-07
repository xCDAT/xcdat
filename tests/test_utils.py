import numpy as np
import pytest
import xarray as xr

from xcdat.utils import open_datasets


class TestOpenDatasets:
    @pytest.fixture(autouse=True)
    def setUp(self, tmp_path):
        # Create temporary directory
        self.dir = tmp_path / "input_data"
        self.dir.mkdir()

        # Create dummy dataset
        self.ds = xr.Dataset(
            {"longitude": np.linspace(0, 10), "latitude": np.linspace(0, 20)}
        )
        self.ds.to_netcdf(f"{self.dir}/file.nc")

    def test_returns_files_with_specific_extension(self):
        # Compare expected and result
        expected = {"file.nc": self.ds}
        result = open_datasets(self.dir, extension="nc")

        for filename, dataset in result.items():
            assert dataset.equals(expected[filename])

    def test_returns_all_files(self):
        # Compare expected and result
        expected = {"file.nc": self.ds}
        result = open_datasets(self.dir)

        for filename, dataset in result.items():
            assert dataset.equals(expected[filename])
