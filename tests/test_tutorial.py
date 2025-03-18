import os
from unittest.mock import patch

import pytest
import xarray as xr

from xcdat.tutorial import open_dataset


class TestOpenDataset:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        # Create temporary directory to save files.
        self.cache_dir = tmp_path / "cache"
        self.cache_dir.mkdir()

        # Create a dummy test.nc file in the cache directory.
        with open(self.cache_dir / "test.nc", "w") as f:
            f.write("")

    @patch("pooch.retrieve")
    @patch("xcdat.dataset.open_dataset")
    def test_open_dataset(self, mock_open_dataset, mock_retrieve):
        # Mock the return value of pooch.retrieve
        mock_retrieve.return_value = str(self.cache_dir / "test.nc")

        # Mock the return value of xcdat.open_dataset
        mock_ds = xr.Dataset()
        mock_open_dataset.return_value = mock_ds

        result = open_dataset("tas_amon_access", cache_dir=self.cache_dir)
        xr.testing.assert_identical(result, mock_ds)

        mock_retrieve.assert_called_once()
        mock_open_dataset.assert_called_once_with(
            str(self.cache_dir / "test.nc"), add_bounds=("X", "Y")
        )

    @patch("pooch.retrieve")
    @patch("xcdat.dataset.open_dataset")
    def test_open_dataset_no_cache(self, mock_open_dataset, mock_retrieve):
        # Mock the return value of pooch.retrieve
        mock_retrieve.return_value = str(self.cache_dir / "test.nc")

        # Mock the return value of xcdat.open_dataset
        mock_ds = xr.Dataset()
        mock_open_dataset.return_value = mock_ds

        result = open_dataset("tas_amon_access", cache=False, cache_dir=self.cache_dir)
        xr.testing.assert_identical(result, mock_ds)

        mock_retrieve.assert_called_once()
        mock_open_dataset.assert_called_once_with(
            str(self.cache_dir / "test.nc"), add_bounds=("X", "Y")
        )
        assert not os.path.exists(mock_retrieve.return_value)

    def test_raises_error_with_invalid_name(self):
        with pytest.raises(ValueError):
            open_dataset("invalid_name", cache_dir=self.cache_dir)

    @patch.dict("sys.modules", {"pooch": None})
    def test_raises_error_when_pooch_module_is_not_installed(self):
        with pytest.raises(ImportError):
            open_dataset("tas_amon_access", cache_dir=self.cache_dir)
