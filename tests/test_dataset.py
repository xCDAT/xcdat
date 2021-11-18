import logging
import warnings

import numpy as np
import pytest
import xarray as xr

from tests.fixtures import generate_dataset
from xcdat.dataset import (
    _check_dataset_for_cf_compliant_time,
    decode_time_units,
    get_inferred_var,
    infer_or_keep_var,
    open_dataset,
    open_mfdataset,
)
from xcdat.logger import setup_custom_logger

logger = setup_custom_logger("xcdat.dataset", propagate=True)


class TestOpenDataset:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        # Create temporary directory to save files.
        dir = tmp_path / "input_data"
        dir.mkdir()
        self.file_path = f"{dir}/file.nc"

    def test_only_keeps_specified_var(self):
        ds = generate_dataset(cf_compliant=True, has_bounds=True)

        # Create a modified version of the Dataset with a new var
        ds_mod = ds.copy()
        ds_mod["tas"] = ds_mod.ts.copy()

        # Suppress UserWarning regarding missing time.encoding "units" because
        # it is not relevant to this test.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ds_mod.to_netcdf(self.file_path)

        result = open_dataset(self.file_path, data_var="ts")
        expected = ds.copy()
        expected.attrs["xcdat_infer"] = "ts"
        assert result.identical(expected)

    def test_non_cf_compliant_time_is_decoded(self):
        # Generate dummy datasets with non-CF compliant time units that aren't
        # encoded yet.
        ds = generate_dataset(cf_compliant=False, has_bounds=False)
        ds.to_netcdf(self.file_path)

        result = open_dataset(self.file_path, data_var="ts")
        # Replicates decode_times=False, which adds units to "time" coordinate.
        # Refer to xcdat.bounds.BoundsAccessor._add_bounds() for
        # how attributes propagate from coord to coord bounds.
        result["time_bnds"].attrs["units"] = "months since 2000-01-01"

        # Generate an expected dataset with non-CF compliant time units that are
        # manually encoded
        expected = generate_dataset(cf_compliant=True, has_bounds=True)
        expected.attrs["xcdat_infer"] = "ts"
        expected.time.attrs["units"] = "months since 2000-01-01"
        expected.time_bnds.attrs["units"] = "months since 2000-01-01"
        expected.time.encoding = {
            "source": None,
            "dtype": np.dtype(np.int64),
            "original_shape": expected.time.data.shape,
            "units": "months since 2000-01-01",
            "calendar": "proleptic_gregorian",
        }

        # Check that non-cf compliant time was decoded and bounds were generated.
        assert result.identical(expected)

    def test_preserves_lat_and_lon_bounds_if_they_exist(self):
        ds = generate_dataset(cf_compliant=True, has_bounds=True)

        # Suppress UserWarning regarding missing time.encoding "units" because
        # it is not relevant to this test.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ds.to_netcdf(self.file_path)

        result = open_dataset(self.file_path, data_var="ts")
        expected = ds.copy()
        expected.attrs["xcdat_infer"] = "ts"

        assert result.identical(expected)

    def test_generates_lat_and_lon_bounds_if_they_dont_exist(self):
        # Create expected dataset without bounds.
        ds = generate_dataset(cf_compliant=True, has_bounds=False)

        ds.to_netcdf(self.file_path)
        ds.close()

        # Make sure bounds don't exist
        data_vars = list(ds.data_vars.keys())
        assert "lat_bnds" not in data_vars
        assert "lon_bnds" not in data_vars

        # Check bounds were generated.
        result = open_dataset(self.file_path, data_var="ts")
        result_data_vars = list(result.data_vars.keys())
        assert "lat_bnds" in result_data_vars
        assert "lon_bnds" in result_data_vars


class TestCheckTimeCfCompliant:
    @pytest.fixture(autouse=True)
    def setUp(self, tmp_path):
        # Create temporary directory to save files.
        self.dir = tmp_path / "input_data"
        self.dir.mkdir()

        # Paths to the dummy datasets.
        self.file_path = f"{self.dir}/file.nc"

    def test_non_cf_compliant_time(self):
        # Generate dummy datasets with non-CF compliant time units
        ds = generate_dataset(cf_compliant=False, has_bounds=False)
        ds.to_netcdf(self.file_path)

        result = _check_dataset_for_cf_compliant_time(self.file_path)

        # Check that False is returned when the dataset has non-cf_compliant time
        assert result is False

    def test_cf_compliant_time(self):
        # Generate dummy datasets with CF compliant time units
        ds = generate_dataset(cf_compliant=True, has_bounds=False)
        ds.to_netcdf(self.file_path)

        result = _check_dataset_for_cf_compliant_time(self.file_path)

        # Check that True is returned when the dataset has cf_compliant time
        assert result is True

    def test_no_time_axis(self):
        # Generate dummy datasets with CF compliant time
        ds = generate_dataset(cf_compliant=True, has_bounds=False)
        # remove time axis
        ds = ds.isel(time=0)
        ds = ds.squeeze(drop=True)
        ds = ds.reset_coords()
        ds = ds.drop_vars("time")
        ds.to_netcdf(self.file_path)

        result = _check_dataset_for_cf_compliant_time(self.file_path)

        # Check that None is returned when there is no time axis
        assert result is None

    def test_glob_cf_compliant_time(self):
        # Generate dummy datasets with CF compliant time
        ds = generate_dataset(cf_compliant=True, has_bounds=False)
        ds.to_netcdf(self.file_path)

        result = _check_dataset_for_cf_compliant_time(f"{self.dir}" + "/*.nc")

        # Check that the wildcard path input is correctly evaluated
        assert result is True

    def test_list_cf_compliant_time(self):
        # Generate dummy datasets with CF compliant time units
        ds = generate_dataset(cf_compliant=True, has_bounds=False)
        ds.to_netcdf(self.file_path)

        flist = [self.file_path, self.file_path, self.file_path]
        result = _check_dataset_for_cf_compliant_time(flist)

        # Check that the list input is correctly evaluated
        assert result is True


class TestOpenMfDataset:
    @pytest.fixture(autouse=True)
    def setUp(self, tmp_path):
        # Create temporary directory to save files.
        dir = tmp_path / "input_data"
        dir.mkdir()
        self.file_path1 = f"{dir}/file1.nc"
        self.file_path2 = f"{dir}/file2.nc"

    def test_only_keeps_specified_var(self):
        # Generate two dummy datasets with non-CF compliant time units.
        ds1 = generate_dataset(cf_compliant=False, has_bounds=False)
        ds1.to_netcdf(self.file_path1)
        ds2 = generate_dataset(cf_compliant=False, has_bounds=False)
        ds2 = ds2.rename_vars({"ts": "tas"})
        ds2.to_netcdf(self.file_path2)

        result = open_mfdataset([self.file_path1, self.file_path2], data_var="ts")

        # Replicates decode_times=False, which adds units to "time" coordinate.
        # Refer to xcdat.bounds.BoundsAccessor._add_bounds() for
        # how attributes propagate from coord to coord bounds.
        result.time_bnds.attrs["units"] = "months since 2000-01-01"

        # Generate an expected dataset, which is a combination of both datasets
        # with decoded time units and coordinate bounds.
        expected = generate_dataset(cf_compliant=True, has_bounds=True)
        expected.attrs["xcdat_infer"] = "ts"
        expected.time.attrs["units"] = "months since 2000-01-01"
        expected.time_bnds.attrs["units"] = "months since 2000-01-01"
        expected.time.encoding = {
            "source": None,
            "dtype": np.dtype(np.int64),
            "original_shape": expected.time.data.shape,
            "units": "months since 2000-01-01",
            "calendar": "proleptic_gregorian",
        }

        # Check that non-cf compliant time was decoded and bounds were generated.
        assert result.identical(expected)
        result.close()

    def test_non_cf_compliant_time_is_decoded(self):
        # Generate two dummy datasets with non-CF compliant time units.
        ds1 = generate_dataset(cf_compliant=False, has_bounds=False)
        ds2 = generate_dataset(cf_compliant=False, has_bounds=False)
        ds2 = ds2.rename_vars({"ts": "tas"})

        ds1.to_netcdf(self.file_path1)
        ds2.to_netcdf(self.file_path2)

        result = open_mfdataset([self.file_path1, self.file_path2], data_var="ts")
        # Replicates decode_times=False, which adds units to "time" coordinate.
        # Refer to xcdat.bounds.BoundsAccessor._add_bounds() for
        # how attributes propagate from coord to coord bounds.
        result.time_bnds.attrs["units"] = "months since 2000-01-01"

        # Generate an expected dataset, which is a combination of both datasets
        # with decoded time units and coordinate bounds.
        expected = generate_dataset(cf_compliant=True, has_bounds=True)
        expected.attrs["xcdat_infer"] = "ts"
        expected.time.attrs["units"] = "months since 2000-01-01"
        expected.time_bnds.attrs["units"] = "months since 2000-01-01"
        expected.time.encoding = {
            "source": None,
            "dtype": np.dtype(np.int64),
            "original_shape": expected.time.data.shape,
            "units": "months since 2000-01-01",
            "calendar": "proleptic_gregorian",
        }

        # Check that non-cf compliant time was decoded and bounds were generated.
        assert result.identical(expected)

    def test_preserves_lat_and_lon_bounds_if_they_exist(self):
        ds1 = generate_dataset(cf_compliant=True, has_bounds=True)
        ds2 = generate_dataset(cf_compliant=True, has_bounds=True)
        ds2 = ds2.rename_vars({"ts": "tas"})

        # Suppress UserWarning regarding missing time.encoding "units" because
        # it is not relevant to this test.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ds1.to_netcdf(self.file_path1)
            ds2.to_netcdf(self.file_path2)

        # Generate expected dataset, which is a combination of the two datasets.
        expected = generate_dataset(cf_compliant=True, has_bounds=True)
        expected.attrs["xcdat_infer"] = "ts"
        # Check that the result is identical to the expected.
        result = open_mfdataset([self.file_path1, self.file_path2], data_var="ts")
        assert result.identical(expected)

    def test_generates_lat_and_lon_bounds_if_they_dont_exist(self):
        # Generate two dummy datasets.
        ds1 = generate_dataset(cf_compliant=True, has_bounds=False)
        ds2 = generate_dataset(cf_compliant=True, has_bounds=False)
        ds2 = ds2.rename_vars({"ts": "tas"})

        ds1.to_netcdf(self.file_path1)
        ds2.to_netcdf(self.file_path2)

        # Make sure no bounds exist in the input file.
        data_vars1 = list(ds1.data_vars.keys())
        data_vars2 = list(ds2.data_vars.keys())
        assert "lat_bnds" not in data_vars1 + data_vars2
        assert "lon_bnds" not in data_vars1 + data_vars2

        # Check that bounds were generated.
        result = open_dataset(self.file_path1, data_var="ts")
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
        expected = xr.Dataset(
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
        result = decode_time_units(input_ds)
        assert result.identical(expected)

        # Check the encodings are the same.
        expected.time.encoding = {
            # Default entries when `decode_times=True`
            "dtype": np.dtype(np.int64),
            "units": time_attrs["units"],
        }
        assert result.time.encoding == expected.time.encoding

    def test_decodes_non_cf_compliant_time_units_months(self):
        # Create a dummy dataset with non-CF compliant time units.
        time_attrs = self.time_attrs
        time_attrs.update({"units": "months since 2000-01-01"})
        time_coord = xr.DataArray(
            name="time", data=[0, 1, 2], dims=["time"], attrs=time_attrs
        )
        input_ds = xr.Dataset({"time": time_coord})

        # Create an expected dataset with properly decoded time units.
        expected = xr.Dataset(
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
        result = decode_time_units(input_ds)
        assert result.identical(expected)

        # Check result and expected time coordinate encodings are the same.
        expected.time.encoding = {
            "source": None,
            "dtype": np.dtype(np.int64),
            "original_shape": expected.time.data.shape,
            "units": time_attrs["units"],
            "calendar": "proleptic_gregorian",
        }
        assert result.time.encoding == expected.time.encoding

    def test_decodes_non_cf_compliant_time_units_years(self):
        # Create a dummy dataset with non-CF compliant time units.
        time_attrs = self.time_attrs
        time_attrs.update({"units": "years since 2000-01-01"})
        time_coord = xr.DataArray(
            name="time", data=[0, 1, 2], dims=["time"], attrs=time_attrs
        )
        input_ds = xr.Dataset({"time": time_coord})

        # Create an expected dataset with properly decoded time units.
        expected = xr.Dataset(
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
        result = decode_time_units(input_ds)
        assert result.identical(expected)

        # Check result and expected time coordinate encodings are the same.
        expected.time.encoding = {
            "source": None,
            "dtype": np.dtype(np.int64),
            "original_shape": expected.time.data.shape,
            "units": time_attrs["units"],
            "calendar": "proleptic_gregorian",
        }
        assert result.time.encoding == expected.time.encoding


class TestInferOrKeepVar:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = generate_dataset(cf_compliant=True, has_bounds=True)

        self.ds_mod = self.ds.copy()
        self.ds_mod["tas"] = self.ds_mod.ts.copy()

    def tests_raises_logger_debug_if_only_bounds_data_variables_exist(self, caplog):
        caplog.set_level(logging.DEBUG)

        ds = self.ds.copy()
        ds = ds.drop_vars("ts")

        infer_or_keep_var(ds, data_var=None)
        assert "This dataset only contains bounds data variables." in caplog.text

    def test_raises_error_if_specified_data_var_does_not_exist(self):
        ds = self.ds_mod.copy()
        with pytest.raises(KeyError):
            infer_or_keep_var(ds, data_var="nonexistent")

    def test_raises_error_if_specified_data_var_is_a_bounds_var(self):
        ds = self.ds_mod.copy()
        with pytest.raises(KeyError):
            infer_or_keep_var(ds, data_var="lat_bnds")

    def test_returns_dataset_if_it_only_has_one_non_bounds_data_var(self):
        ds = self.ds.copy()

        result = infer_or_keep_var(ds, data_var=None)
        expected = ds.copy()
        expected.attrs["xcdat_infer"] = "ts"

        assert result.identical(expected)

    def test_returns_dataset_if_it_contains_multiple_non_bounds_data_var_with_logger_msg(
        self, caplog
    ):
        caplog.set_level(logging.DEBUG)

        ds = self.ds_mod.copy()
        result = infer_or_keep_var(ds, data_var=None)
        expected = ds.copy()
        expected.attrs["xcdat_infer"] = None

        assert result.identical(expected)
        assert result.attrs.get("xcdat_infer") is None
        assert (
            "This dataset contains more than one regular data variable ('tas', 'ts'). "
            "If desired, pass the `data_var` kwarg to reduce down to one regular data var."
        ) in caplog.text

    def test_returns_dataset_with_specified_data_var_and_inference_attr(self):
        result = infer_or_keep_var(self.ds_mod, data_var="ts")
        expected = self.ds.copy()
        expected.attrs["xcdat_infer"] = "ts"

        assert result.identical(expected)
        assert not result.identical(self.ds_mod)

    def test_bounds_always_persist(self):
        ds = infer_or_keep_var(self.ds_mod, data_var="ts")
        assert ds.get("lat_bnds") is not None
        assert ds.get("lon_bnds") is not None
        assert ds.get("time_bnds") is not None


class TestGetInferredVar:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = generate_dataset(cf_compliant=True, has_bounds=True)

    def test_raises_error_if_inference_attr_is_none(self):
        with pytest.raises(KeyError):
            get_inferred_var(self.ds)

    def test_raises_error_if_inference_attr_is_set_to_nonexistent_data_var(self):
        ds = self.ds.copy()
        ds.attrs["xcdat_infer"] = "nonexistent_var"

        with pytest.raises(KeyError):
            get_inferred_var(ds)

    def test_raises_error_if_inference_attr_is_set_to_bounds_var(self):
        ds = self.ds.copy()
        ds.attrs["xcdat_infer"] = "lat_bnds"

        with pytest.raises(KeyError):
            get_inferred_var(ds)

    def test_returns_inferred_data_var(self, caplog):
        caplog.set_level(logging.DEBUG)

        ds = self.ds.copy()
        ds.attrs["xcdat_infer"] = "ts"

        result = get_inferred_var(ds)
        expected = ds.ts

        assert result.identical(expected)
        assert (
            "The data variable 'ts' was inferred from the Dataset attr 'xcdat_infer' "
            "for this operation."
        ) in caplog.text
