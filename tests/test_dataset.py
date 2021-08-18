import numpy as np
import pytest
import xarray as xr

from tests.fixtures import generate_dataset
from xcdat.dataset import decode_time_units, open_dataset, open_mfdataset


class TestOpenDataset:
    @pytest.fixture(autouse=True)
    def setUp(self, tmp_path):
        # Create temporary directory to save files.
        self.dir = tmp_path / "input_data"
        self.dir.mkdir()

        # Paths to the dummy datasets.
        self.file_path = f"{self.dir}/file.nc"

    def test_non_cf_compliant_time_is_decoded(self):
        # Generate dummy datasets with non-CF compliant time units that aren't
        # encoded yet.
        ds = generate_dataset(cf_compliant=False, has_bounds=False)
        ds.to_netcdf(self.file_path)

        result_ds = open_dataset(self.file_path)
        # Replicates decode_times=False, which adds units to "time" coordinate.
        # Refer to xcdat.bounds.DatasetBoundsAccessor._add_bounds() for
        # how attributes propagate from coord to coord bounds.
        result_ds["time_bnds"].attrs["units"] = "months since 2000-01-01"

        # Generate an expected dataset with non-CF compliant time units that are
        # manually encoded
        expected_ds = generate_dataset(cf_compliant=True, has_bounds=True)
        expected_ds.time.attrs["units"] = "months since 2000-01-01"
        expected_ds.time_bnds.attrs["units"] = "months since 2000-01-01"
        expected_ds.time.encoding = {
            "source": None,
            "dtype": np.dtype(np.int64),
            "original_shape": expected_ds.time.data.shape,
            "units": "months since 2000-01-01",
            "calendar": "proleptic_gregorian",
        }

        # Check that non-cf compliant time was decoded and bounds were generated.
        assert result_ds.identical(expected_ds)

    def test_preserves_lat_and_lon_bounds_if_they_exist(self):
        # Create expected dataset which includes bounds.
        expected_ds = generate_dataset(cf_compliant=True, has_bounds=True)
        expected_ds.to_netcdf(self.file_path)

        # Check resulting dataset and expected are identical
        result_ds = open_dataset(self.file_path)
        assert result_ds.identical(expected_ds)

    def test_generates_lat_and_lon_bounds_if_they_dont_exist(self):
        # Create expected dataset without bounds.
        ds = generate_dataset(cf_compliant=True, has_bounds=False)
        ds.to_netcdf(self.file_path)

        # Make sure bounds don't exist
        data_vars = list(ds.data_vars.keys())
        assert "lat_bnds" not in data_vars
        assert "lon_bnds" not in data_vars

        # Check bounds were generated.
        result = open_dataset(self.file_path)
        result_data_vars = list(result.data_vars.keys())
        assert "lat_bnds" in result_data_vars
        assert "lon_bnds" in result_data_vars


class TestOpenMfDataset:
    @pytest.fixture(autouse=True)
    def setUp(self, tmp_path):
        # Create temporary directory to save files.
        self.dir = tmp_path / "input_data"
        self.dir.mkdir()

        # Paths to the dummy datasets.
        self.file_path1 = f"{self.dir}/file1.nc"
        self.file_path2 = f"{self.dir}/file2.nc"

    def test_non_cf_compliant_time_is_decoded(self):
        # Generate two dummy datasets with non-CF compliant time units.
        ds1 = generate_dataset(cf_compliant=False, has_bounds=False)
        ds1.to_netcdf(self.file_path1)
        ds2 = generate_dataset(cf_compliant=False, has_bounds=False)
        ds2 = ds2.rename_vars({"ts": "tas"})
        ds2.to_netcdf(self.file_path2)

        result_ds = open_mfdataset([self.file_path1, self.file_path2])
        # Replicates decode_times=False, which adds units to "time" coordinate.
        # Refer to xcdat.bounds.DatasetBoundsAccessor._add_bounds() for
        # how attributes propagate from coord to coord bounds.
        result_ds.time_bnds.attrs["units"] = "months since 2000-01-01"

        # Generate an expected dataset, which is a combination of both datasets
        # with decoded time units and coordinate bounds.
        expected_ds = generate_dataset(cf_compliant=True, has_bounds=True)
        expected_ds["tas"] = expected_ds["ts"].copy()
        expected_ds.time.attrs["units"] = "months since 2000-01-01"
        expected_ds.time_bnds.attrs["units"] = "months since 2000-01-01"
        expected_ds.time.encoding = {
            "source": None,
            "dtype": np.dtype(np.int64),
            "original_shape": expected_ds.time.data.shape,
            "units": "months since 2000-01-01",
            "calendar": "proleptic_gregorian",
        }

        # Check that non-cf compliant time was decoded and bounds were generated.
        assert result_ds.identical(expected_ds)

    def test_preserves_lat_and_lon_bounds_if_they_exist(self):
        # Generate two dummy datasets.
        ds1 = generate_dataset(cf_compliant=True, has_bounds=True)
        ds1.to_netcdf(self.file_path1)

        ds2 = generate_dataset(cf_compliant=True, has_bounds=True)
        ds2 = ds2.rename_vars({"ts": "tas"})
        ds2.to_netcdf(self.file_path2)

        # Generate expected dataset, which is a combination of the two datasets.
        expected_ds = generate_dataset(cf_compliant=True, has_bounds=True)
        expected_ds["tas"] = expected_ds["ts"].copy()

        # Check that the result is identical to the expected.
        result_ds = open_mfdataset([self.file_path1, self.file_path2])
        assert result_ds.identical(expected_ds)

    def test_generates_lat_and_lon_bounds_if_they_dont_exist(self):
        # Generate two dummy datasets.
        ds1 = generate_dataset(cf_compliant=True, has_bounds=False)
        ds1.to_netcdf(self.file_path1)

        ds2 = generate_dataset(cf_compliant=True, has_bounds=False)
        ds2 = ds2.rename_vars({"ts": "tas"})
        ds2.to_netcdf(self.file_path2)

        # Make sure no bounds exist in the input file.
        data_vars1 = list(ds1.data_vars.keys())
        data_vars2 = list(ds2.data_vars.keys())
        assert "lat_bnds" not in data_vars1 + data_vars2
        assert "lon_bnds" not in data_vars1 + data_vars2

        # Check that bounds were generated.
        result = open_dataset(self.file_path1)
        result_data_vars = list(result.data_vars.keys())
        assert "lat_bnds" in result_data_vars
        assert "lon_bnds" in result_data_vars


class TestDecodeTimeUnits:
    @pytest.fixture(autouse=True)
    def setup(self):
        # Common attributes for the time coordinate. Units are overriden based
        # on the unit that needs to be tested (days (CF compliant) or months
        # (non-CF compliant).
        self.time_attrs = {
            "bounds": "time_bnds",
            "axis": "T",
            "long_name": "time",
            "standard_name": "time",
        }

    def test_throws_error_if_function_is_called_on_already_decoded_cf_compliant_dataset(
        self,
    ):
        ds = generate_dataset(cf_compliant=True, has_bounds=True)

        with pytest.raises(KeyError):
            decode_time_units(ds)

    def test_decodes_cf_compliant_time_units(self):
        # Create a dummy dataset with CF compliant time units.
        time_attrs = self.time_attrs

        # Create an expected dataset with properly decoded time units.
        expected_ds = xr.Dataset(
            {
                "time": xr.DataArray(
                    name="time",
                    data=[
                        np.datetime64("2000-01-01"),
                        np.datetime64("2000-01-02"),
                        np.datetime64("2000-01-03"),
                    ],
                    dims=["time"],
                    attrs=time_attrs,
                )
            }
        )

        # Update the time attrs to mimic decode_times=False
        time_attrs.update({"units": "days since 2000-01-01"})
        time_coord = xr.DataArray(
            name="time", data=[0, 1, 2], dims=["time"], attrs=time_attrs
        )
        input_ds = xr.Dataset({"time": time_coord})

        # Check the resulting dataset is identical to the expected.
        result_ds = decode_time_units(input_ds)
        assert result_ds.identical(expected_ds)

        # Check the encodings are the same.
        expected_ds.time.encoding = {
            # Default entries when `decode_times=True`
            "dtype": np.dtype(np.int64),
            "units": time_attrs["units"],
        }
        assert result_ds.time.encoding == expected_ds.time.encoding

    def test_decodes_non_cf_compliant_time_units_months(self):
        # Create a dummy dataset with non-CF compliant time units.
        time_attrs = self.time_attrs
        time_attrs.update({"units": "months since 2000-01-01"})
        time_coord = xr.DataArray(
            name="time", data=[0, 1, 2], dims=["time"], attrs=time_attrs
        )
        input_ds = xr.Dataset({"time": time_coord})

        # Create an expected dataset with properly decoded time units.
        expected_ds = xr.Dataset(
            {
                "time": xr.DataArray(
                    name="time",
                    data=[
                        np.datetime64("2000-01-01"),
                        np.datetime64("2000-02-01"),
                        np.datetime64("2000-03-01"),
                    ],
                    dims=["time"],
                    attrs=time_attrs,
                )
            }
        )

        # Check the resulting dataset is identical to the expected.
        result_ds = decode_time_units(input_ds)
        assert result_ds.identical(expected_ds)

        # Check result and expected time coordinate encodings are the same.
        expected_ds.time.encoding = {
            "source": None,
            "dtype": np.dtype(np.int64),
            "original_shape": expected_ds.time.data.shape,
            "units": time_attrs["units"],
            "calendar": "proleptic_gregorian",
        }
        assert result_ds.time.encoding == expected_ds.time.encoding

    def test_decodes_non_cf_compliant_time_units_years(self):
        # Create a dummy dataset with non-CF compliant time units.
        time_attrs = self.time_attrs
        time_attrs.update({"units": "years since 2000-01-01"})
        time_coord = xr.DataArray(
            name="time", data=[0, 1, 2], dims=["time"], attrs=time_attrs
        )
        input_ds = xr.Dataset({"time": time_coord})

        # Create an expected dataset with properly decoded time units.
        expected_ds = xr.Dataset(
            {
                "time": xr.DataArray(
                    name="time",
                    data=[
                        np.datetime64("2000-01-01"),
                        np.datetime64("2001-01-01"),
                        np.datetime64("2002-01-01"),
                    ],
                    dims=["time"],
                    attrs=time_attrs,
                )
            }
        )

        # Check the resulting dataset is identical to the expected.
        result_ds = decode_time_units(input_ds)
        assert result_ds.identical(expected_ds)

        # Check result and expected time coordinate encodings are the same.
        expected_ds.time.encoding = {
            "source": None,
            "dtype": np.dtype(np.int64),
            "original_shape": expected_ds.time.data.shape,
            "units": time_attrs["units"],
            "calendar": "proleptic_gregorian",
        }
        assert result_ds.time.encoding == expected_ds.time.encoding
