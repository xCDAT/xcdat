import logging
import pathlib
import warnings

import numpy as np
import pytest
import xarray as xr

from tests.fixtures import generate_dataset
from xcdat.dataset import (
    _preprocess_non_cf_dataset,
    _split_time_units_attr,
    decode_non_cf_time,
    get_inferred_var,
    has_cf_compliant_time,
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

    def test_non_cf_compliant_time_is_not_decoded(self):
        ds = generate_dataset(cf_compliant=False, has_bounds=True)
        ds.to_netcdf(self.file_path)

        result = open_dataset(self.file_path, decode_times=False)
        expected = generate_dataset(cf_compliant=False, has_bounds=True)
        expected.attrs["xcdat_infer"] = "ts"

        assert result.identical(expected)

    def test_non_cf_compliant_time_is_decoded(self):
        ds = generate_dataset(cf_compliant=False, has_bounds=False)
        ds.to_netcdf(self.file_path)

        result = open_dataset(self.file_path, data_var="ts")
        expected = generate_dataset(cf_compliant=True, has_bounds=True)
        expected.attrs["xcdat_infer"] = "ts"
        expected.time.attrs["calendar"] = "standard"
        expected.time.attrs["units"] = "months since 2000-01-01"
        expected.time.encoding = {
            "source": result.time.encoding["source"],
            "dtype": np.dtype(np.int64),
            "original_shape": expected.time.data.shape,
            "units": "months since 2000-01-01",
            "calendar": "standard",
        }

        assert result.identical(expected)
        assert result.time.encoding == expected.time.encoding

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

        data_vars = list(ds.data_vars.keys())
        assert "lat_bnds" not in data_vars
        assert "lon_bnds" not in data_vars

        result = open_dataset(self.file_path, data_var="ts")
        result_data_vars = list(result.data_vars.keys())
        assert "lat_bnds" in result_data_vars
        assert "lon_bnds" in result_data_vars


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
        expected = generate_dataset(cf_compliant=True, has_bounds=True)
        expected.attrs["xcdat_infer"] = "ts"
        expected.time.attrs["calendar"] = "standard"
        expected.time.attrs["units"] = "months since 2000-01-01"

        expected.time.encoding = {
            "source": result.time.encoding["source"],
            "dtype": np.dtype(np.int64),
            "original_shape": expected.time.data.shape,
            "units": "months since 2000-01-01",
            "calendar": "standard",
        }

        assert result.identical(expected)
        assert result.time.encoding == expected.time.encoding

    def test_non_cf_compliant_time_is_not_decoded(self):
        ds1 = generate_dataset(cf_compliant=False, has_bounds=True)
        ds1.to_netcdf(self.file_path1)
        ds2 = generate_dataset(cf_compliant=False, has_bounds=True)
        ds2 = ds2.rename_vars({"ts": "tas"})
        ds2.to_netcdf(self.file_path2)

        result = open_mfdataset([self.file_path1, self.file_path2], decode_times=False)

        expected = ds1.merge(ds2)
        expected.attrs["xcdat_infer"] = "None"
        assert result.identical(expected)

    def test_non_cf_compliant_time_is_decoded(self):
        ds1 = generate_dataset(cf_compliant=False, has_bounds=False)
        ds2 = generate_dataset(cf_compliant=False, has_bounds=False)
        ds2 = ds2.rename_vars({"ts": "tas"})

        ds1.to_netcdf(self.file_path1)
        ds2.to_netcdf(self.file_path2)

        result = open_mfdataset([self.file_path1, self.file_path2], data_var="ts")
        expected = generate_dataset(cf_compliant=True, has_bounds=True)
        expected.attrs["xcdat_infer"] = "ts"
        expected.time.attrs["units"] = "months since 2000-01-01"
        expected.time.attrs["calendar"] = "standard"
        expected.time.encoding = {
            "source": result.time.encoding["source"],
            "dtype": np.dtype(np.int64),
            "original_shape": expected.time.data.shape,
            "units": "months since 2000-01-01",
            "calendar": "standard",
        }

        assert result.identical(expected)
        assert result.time.encoding == expected.time.encoding

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

        expected = generate_dataset(cf_compliant=True, has_bounds=True)
        expected.attrs["xcdat_infer"] = "ts"
        result = open_mfdataset([self.file_path1, self.file_path2], data_var="ts")
        assert result.identical(expected)

    def test_generates_lat_and_lon_bounds_if_they_dont_exist(self):
        ds1 = generate_dataset(cf_compliant=True, has_bounds=False)
        ds1.to_netcdf(self.file_path1)
        ds2 = generate_dataset(cf_compliant=True, has_bounds=False)
        ds2 = ds2.rename_vars({"ts": "tas"})

        ds1.to_netcdf(self.file_path1)
        ds2.to_netcdf(self.file_path2)

        data_vars1 = list(ds1.data_vars.keys())
        data_vars2 = list(ds2.data_vars.keys())
        assert "lat_bnds" not in data_vars1 + data_vars2
        assert "lon_bnds" not in data_vars1 + data_vars2

        result = open_dataset(self.file_path1, data_var="ts")
        result_data_vars = list(result.data_vars.keys())
        assert "lat_bnds" in result_data_vars
        assert "lon_bnds" in result_data_vars


class TestHasCFCompliantTime:
    @pytest.fixture(autouse=True)
    def setUp(self, tmp_path):
        # Create temporary directory to save files.
        self.dir = tmp_path / "input_data"
        self.dir.mkdir()

        # Paths to the dummy datasets.
        self.file_path = f"{self.dir}/file.nc"

    def test_non_cf_compliant_time(self):
        # Generate dummy dataset with non-CF compliant time units
        ds = generate_dataset(cf_compliant=False, has_bounds=False)
        ds.to_netcdf(self.file_path)

        result = has_cf_compliant_time(self.file_path)

        # Check that False is returned when the dataset has non-cf_compliant time
        assert result is False

    def test_no_time_axis(self):
        # Generate dummy dataset with CF compliant time
        ds = generate_dataset(cf_compliant=True, has_bounds=False)
        # remove time axis
        ds = ds.isel(time=0)
        ds = ds.squeeze(drop=True)
        ds = ds.reset_coords()
        ds = ds.drop_vars("time")
        ds.to_netcdf(self.file_path)

        result = has_cf_compliant_time(self.file_path)

        # Check that None is returned when there is no time axis
        assert result is None

    def test_glob_cf_compliant_time(self):
        # Generate dummy datasets with CF compliant time
        ds = generate_dataset(cf_compliant=True, has_bounds=False)
        ds.to_netcdf(self.file_path)

        result = has_cf_compliant_time(f"{self.dir}/*.nc")

        # Check that the wildcard path input is correctly evaluated
        assert result is True

    def test_list_cf_compliant_time(self):
        # Generate dummy datasets with CF compliant time units
        ds = generate_dataset(cf_compliant=True, has_bounds=False)
        ds.to_netcdf(self.file_path)

        flist = [self.file_path, self.file_path, self.file_path]
        result = has_cf_compliant_time(flist)

        # Check that the list input is correctly evaluated
        assert result is True

    def test_cf_compliant_time_with_string_path(self):
        # Generate dummy dataset with CF compliant time units
        ds = generate_dataset(cf_compliant=True, has_bounds=False)
        ds.to_netcdf(self.file_path)

        result = has_cf_compliant_time(self.file_path)

        # Check that True is returned when the dataset has cf_compliant time
        assert result is True

    def test_cf_compliant_time_with_pathlib_path(self):
        # Generate dummy dataset with CF compliant time units
        ds = generate_dataset(cf_compliant=True, has_bounds=False)
        ds.to_netcdf(self.file_path)

        result = has_cf_compliant_time(pathlib.Path(self.file_path))

        # Check that True is returned when the dataset has cf_compliant time
        assert result is True

    def test_cf_compliant_time_with_list_of_list_of_strings(self):
        # Generate dummy dataset with CF compliant time units
        ds = generate_dataset(cf_compliant=True, has_bounds=False)
        ds.to_netcdf(self.file_path)

        result = has_cf_compliant_time([self.file_path])

        # Check that True is returned when the dataset has cf_compliant time
        assert result is True

    def test_cf_compliant_time_with_list_of_list_of_pathlib_paths(self):
        # Generate dummy dataset with CF compliant time units
        ds = generate_dataset(cf_compliant=True, has_bounds=False)
        ds.to_netcdf(self.file_path)

        result = has_cf_compliant_time([[pathlib.Path(self.file_path)]])

        # Check that True is returned when the dataset has cf_compliant time
        assert result is True


class TestDecodeNonCFTimeUnits:
    @pytest.fixture(autouse=True)
    def setup(self):
        time = xr.DataArray(
            name="time",
            data=[1, 2, 3],
            dims=["time"],
            attrs={
                "bounds": "time_bnds",
                "axis": "T",
                "long_name": "time",
                "standard_name": "time",
                "calendar": "noleap",
            },
        )
        time_bnds = xr.DataArray(
            name="time_bnds",
            data=[[0, 1], [1, 2], [2, 3]],
            dims=["time", "bnds"],
        )
        time_bnds.encoding = {
            "zlib": False,
            "shuffle": False,
            "complevel": 0,
            "fletcher32": False,
            "contiguous": False,
            "chunksizes": (1, 2),
            "source": "None",
            "original_shape": (1980, 2),
            "dtype": np.dtype("float64"),
        }
        self.ds = xr.Dataset({"time": time, "time_bnds": time_bnds})

    def test_raises_error_if_function_is_called_on_already_decoded_cf_compliant_dataset(
        self,
    ):
        ds = generate_dataset(cf_compliant=True, has_bounds=True)

        with pytest.raises(KeyError):
            decode_non_cf_time(ds)

    def test_decodes_months_with_a_reference_date_at_the_start_of_the_month(self):
        ds = self.ds.copy()
        ds.time.attrs["units"] = "months since 2000-01-01"

        result = decode_non_cf_time(ds)
        expected = xr.Dataset(
            {
                "time": xr.DataArray(
                    name="time",
                    data=np.array(
                        [
                            "2000-02-01",
                            "2000-03-01",
                            "2000-04-01",
                        ],
                        dtype="datetime64",
                    ),
                    dims=["time"],
                    attrs=ds.time.attrs,
                ),
                "time_bnds": xr.DataArray(
                    name="time_bnds",
                    data=np.array(
                        [
                            ["2000-01-01", "2000-02-01"],
                            ["2000-02-01", "2000-03-01"],
                            ["2000-03-01", "2000-04-01"],
                        ],
                        dtype="datetime64",
                    ),
                    dims=["time", "bnds"],
                    attrs=ds.time_bnds.attrs,
                ),
            }
        )
        assert result.identical(expected)

        expected.time.encoding = {
            "source": "None",
            "dtype": np.dtype(np.int64),
            "original_shape": expected.time.data.shape,
            "units": ds.time.attrs["units"],
            "calendar": ds.time.attrs["calendar"],
        }
        expected.time_bnds.encoding = ds.time_bnds.encoding
        assert result.time.encoding == expected.time.encoding
        assert result.time_bnds.encoding == expected.time_bnds.encoding

    def test_decodes_months_with_a_reference_date_at_the_middle_of_the_month(self):
        ds = self.ds.copy()
        ds.time.attrs["units"] = "months since 2000-01-15"

        result = decode_non_cf_time(ds)
        expected = xr.Dataset(
            {
                "time": xr.DataArray(
                    name="time",
                    data=np.array(
                        [
                            "2000-02-15",
                            "2000-03-15",
                            "2000-04-15",
                        ],
                        dtype="datetime64",
                    ),
                    dims=["time"],
                    attrs=ds.time.attrs,
                ),
                "time_bnds": xr.DataArray(
                    name="time_bnds",
                    data=np.array(
                        [
                            ["2000-01-15", "2000-02-15"],
                            ["2000-02-15", "2000-03-15"],
                            ["2000-03-15", "2000-04-15"],
                        ],
                        dtype="datetime64",
                    ),
                    dims=["time", "bnds"],
                    attrs=ds.time_bnds.attrs,
                ),
            }
        )
        assert result.identical(expected)

        expected.time.encoding = {
            "source": "None",
            "dtype": np.dtype(np.int64),
            "original_shape": expected.time.data.shape,
            "units": ds.time.attrs["units"],
            "calendar": ds.time.attrs["calendar"],
        }
        expected.time_bnds.encoding = ds.time_bnds.encoding
        assert result.time.encoding == expected.time.encoding
        assert result.time_bnds.encoding == expected.time_bnds.encoding

    def test_decodes_months_with_a_reference_date_at_the_end_of_the_month(self):
        ds = self.ds.copy()
        ds.time.attrs["units"] = "months since 1999-12-31"

        result = decode_non_cf_time(ds)
        expected = xr.Dataset(
            {
                "time": xr.DataArray(
                    name="time",
                    data=np.array(
                        [
                            "2000-01-31",
                            "2000-02-29",
                            "2000-03-31",
                        ],
                        dtype="datetime64",
                    ),
                    dims=["time"],
                    attrs=ds.time.attrs,
                ),
                "time_bnds": xr.DataArray(
                    name="time_bnds",
                    data=np.array(
                        [
                            ["1999-12-31", "2000-01-31"],
                            ["2000-01-31", "2000-02-29"],
                            ["2000-02-29", "2000-03-31"],
                        ],
                        dtype="datetime64",
                    ),
                    dims=["time", "bnds"],
                    attrs=ds.time_bnds.attrs,
                ),
            }
        )
        assert result.identical(expected)

        expected.time.encoding = {
            "source": "None",
            "dtype": np.dtype(np.int64),
            "original_shape": expected.time.data.shape,
            "units": ds.time.attrs["units"],
            "calendar": ds.time.attrs["calendar"],
        }
        expected.time_bnds.encoding = ds.time_bnds.encoding
        assert result.time.encoding == expected.time.encoding
        assert result.time_bnds.encoding == expected.time_bnds.encoding

    def test_decodes_months_with_a_reference_date_on_a_leap_year(self):
        ds = self.ds.copy()
        ds.time.attrs["units"] = "months since 2000-02-29"

        result = decode_non_cf_time(ds)
        expected = xr.Dataset(
            {
                "time": xr.DataArray(
                    name="time",
                    data=np.array(
                        [
                            "2000-03-29",
                            "2000-04-29",
                            "2000-05-29",
                        ],
                        dtype="datetime64",
                    ),
                    dims=["time"],
                    attrs=ds.time.attrs,
                ),
                "time_bnds": xr.DataArray(
                    name="time_bnds",
                    data=np.array(
                        [
                            ["2000-02-29", "2000-03-29"],
                            ["2000-03-29", "2000-04-29"],
                            ["2000-04-29", "2000-05-29"],
                        ],
                        dtype="datetime64",
                    ),
                    dims=["time", "bnds"],
                    attrs=ds.time_bnds.attrs,
                ),
            }
        )
        assert result.identical(expected)

        expected.time.encoding = {
            "source": "None",
            "dtype": np.dtype(np.int64),
            "original_shape": expected.time.data.shape,
            "units": ds.time.attrs["units"],
            "calendar": ds.time.attrs["calendar"],
        }
        expected.time_bnds.encoding = ds.time_bnds.encoding
        assert result.time.encoding == expected.time.encoding
        assert result.time_bnds.encoding == expected.time_bnds.encoding

    def test_decodes_years_with_a_reference_date_at_the_middle_of_the_year(self):
        ds = self.ds.copy()
        ds.time.attrs["units"] = "years since 2000-06-01"

        result = decode_non_cf_time(ds)
        expected = xr.Dataset(
            {
                "time": xr.DataArray(
                    name="time",
                    data=np.array(
                        [
                            "2001-06-01",
                            "2002-06-01",
                            "2003-06-01",
                        ],
                        dtype="datetime64",
                    ),
                    dims=["time"],
                    attrs=ds.time.attrs,
                ),
                "time_bnds": xr.DataArray(
                    name="time_bnds",
                    data=np.array(
                        [
                            ["2000-06-01", "2001-06-01"],
                            ["2001-06-01", "2002-06-01"],
                            ["2002-06-01", "2003-06-01"],
                        ],
                        dtype="datetime64",
                    ),
                    dims=["time", "bnds"],
                    attrs=ds.time_bnds.attrs,
                ),
            }
        )
        assert result.identical(expected)

        expected.time.encoding = {
            "source": "None",
            "dtype": np.dtype(np.int64),
            "original_shape": expected.time.data.shape,
            "units": ds.time.attrs["units"],
            "calendar": ds.time.attrs["calendar"],
        }
        expected.time_bnds.encoding = ds.time_bnds.encoding
        assert result.time.encoding == expected.time.encoding
        assert result.time_bnds.encoding == expected.time_bnds.encoding

    def test_decodes_years_with_a_reference_date_on_a_leap_year(self):
        ds = self.ds.copy()
        ds.time.attrs["units"] = "years since 2000-02-29"

        result = decode_non_cf_time(ds)
        expected = xr.Dataset(
            {
                "time": xr.DataArray(
                    name="time",
                    data=[
                        np.datetime64("2001-02-28"),
                        np.datetime64("2002-02-28"),
                        np.datetime64("2003-02-28"),
                    ],
                    dims=["time"],
                ),
                "time_bnds": xr.DataArray(
                    name="time_bnds",
                    data=np.array(
                        [
                            ["2000-02-29", "2001-02-28"],
                            ["2001-02-28", "2002-02-28"],
                            ["2002-02-28", "2003-02-28"],
                        ],
                        dtype="datetime64",
                    ),
                    dims=["time", "bnds"],
                    attrs=ds.time_bnds.attrs,
                ),
            }
        )
        expected.time.attrs = ds.time.attrs
        assert result.identical(expected)

        expected.time.encoding = {
            "source": "None",
            "dtype": np.dtype(np.int64),
            "original_shape": expected.time.data.shape,
            "units": ds.time.attrs["units"],
            "calendar": ds.time.attrs["calendar"],
        }
        expected.time_bnds.encoding = ds.time_bnds.encoding
        assert result.time.encoding == expected.time.encoding
        assert result.time_bnds.encoding == expected.time_bnds.encoding


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
        expected.attrs["xcdat_infer"] = "None"

        assert result.identical(expected)
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


class TestPreProcessNonCFDataset:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = generate_dataset(cf_compliant=False, has_bounds=True)

    def test_user_specified_callable_results_in_subsetting_dataset_on_time_slice(self):
        def callable(ds):
            return ds.isel(time=slice(0, 1))

        ds = self.ds.copy()

        result = _preprocess_non_cf_dataset(ds, callable)
        expected = ds.copy().isel(time=slice(0, 1))
        expected["time"] = xr.DataArray(
            name="time",
            data=np.array(
                [
                    "2000-01-01",
                ],
                dtype="datetime64",
            ),
            dims=["time"],
        )
        expected["time_bnds"] = xr.DataArray(
            name="time_bnds",
            data=np.array(
                [
                    ["1999-12-01", "2000-01-01"],
                ],
                dtype="datetime64",
            ),
            dims=["time", "bnds"],
        )

        expected.time.attrs = ds.time.attrs
        expected.time_bnds.attrs = ds.time_bnds.attrs
        assert result.identical(expected)


class TestSplitTimeUnitsAttr:
    def test_raises_error_if_units_attr_is_none(self):
        with pytest.raises(KeyError):
            _split_time_units_attr(None)  # type: ignore

    def test_splits_units_attr_to_unit_and_reference_date(self):
        assert _split_time_units_attr("months since 1800") == ("months", "1800")
        assert _split_time_units_attr("months since 1800-01-01") == (
            "months",
            "1800-01-01",
        )
        assert _split_time_units_attr("months since 1800-01-01 00:00:00") == (
            "months",
            "1800-01-01 00:00:00",
        )
