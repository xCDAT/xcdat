import logging
import warnings

import cftime
import numpy as np
import pytest
import xarray as xr

from tests.fixtures import generate_dataset
from xcdat.dataset import (
    _keep_single_var,
    _postprocess_dataset,
    decode_time,
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

    def test_raises_warning_if_decode_times_but_no_time_coords_found(self, caplog):
        ds = generate_dataset(decode_times=False, cf_compliant=True, has_bounds=True)
        ds = ds.drop_dims("time")
        ds.to_netcdf(self.file_path)

        result = open_dataset(self.file_path)
        expected = generate_dataset(
            decode_times=False,
            cf_compliant=True,
            has_bounds=True,
        )
        expected = expected.drop_dims("time")

        assert result.identical(expected)
        assert (
            "No time coordinates were found in this dataset to decode. Make sure time "
            "coordinates exist and their CF 'axis' or 'standard_name' attribute is set "
            "(e.g., ds['time'].attrs['axis'] = 'T' or "
            "ds['time'].attrs['standard_name'] = 'time'). Afterwards, try decoding "
            "with `decode_time()` again."
        ) in caplog.text

    def test_skip_decoding_time_explicitly(self):
        ds = generate_dataset(decode_times=False, cf_compliant=True, has_bounds=True)
        ds.to_netcdf(self.file_path)

        result = open_dataset(self.file_path, decode_times=False)
        expected = generate_dataset(
            decode_times=False,
            cf_compliant=True,
            has_bounds=True,
        )

        assert result.identical(expected)

    def test_skips_decoding_non_cf_compliant_time_with_unsupported_units(self, caplog):
        # Update logger level to silence the logger warning during test runs.
        caplog.set_level(logging.ERROR)

        ds = generate_dataset(decode_times=False, cf_compliant=False, has_bounds=True)
        ds["time"].attrs["units"] = "year A.D."
        ds.to_netcdf(self.file_path)

        # even though decode_times=True, it should fail to decode unsupported time axis
        result = open_dataset(self.file_path, decode_times=False)
        expected = ds

        assert result.identical(expected)

    def test_decode_time_in_days(self):
        ds = generate_dataset(decode_times=False, cf_compliant=True, has_bounds=True)
        ds.to_netcdf(self.file_path)

        result = open_dataset(self.file_path, data_var="ts", decode_times=True)

        # Generate an expected dataset with decoded CF compliant time units.
        expected = ds.copy()
        expected["time"] = xr.DataArray(
            name="time",
            data=np.array(
                [
                    cftime.DatetimeGregorian(
                        2000, 1, 1, 0, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2000, 1, 2, 0, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2000, 1, 3, 0, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2000, 1, 4, 0, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2000, 1, 5, 0, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2000, 1, 6, 0, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2000, 1, 7, 0, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2000, 1, 8, 0, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2000, 1, 9, 0, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2000, 1, 10, 0, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2000, 1, 11, 0, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2000, 1, 12, 0, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2000, 1, 13, 0, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2000, 1, 14, 0, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2000, 1, 15, 0, 0, 0, 0, has_year_zero=False
                    ),
                ],
                dtype="object",
            ),
            dims="time",
        )
        expected["time_bnds"] = xr.DataArray(
            name="time_bnds",
            data=np.array(
                [
                    [
                        cftime.DatetimeGregorian(
                            1999, 12, 31, 0, 0, 0, 0, has_year_zero=False
                        ),
                        cftime.DatetimeGregorian(
                            2000, 1, 1, 0, 0, 0, 0, has_year_zero=False
                        ),
                    ],
                    [
                        cftime.DatetimeGregorian(
                            2000, 1, 1, 0, 0, 0, 0, has_year_zero=False
                        ),
                        cftime.DatetimeGregorian(
                            2000, 1, 2, 0, 0, 0, 0, has_year_zero=False
                        ),
                    ],
                    [
                        cftime.DatetimeGregorian(
                            2000, 1, 2, 0, 0, 0, 0, has_year_zero=False
                        ),
                        cftime.DatetimeGregorian(
                            2000, 1, 3, 0, 0, 0, 0, has_year_zero=False
                        ),
                    ],
                    [
                        cftime.DatetimeGregorian(
                            2000, 1, 3, 0, 0, 0, 0, has_year_zero=False
                        ),
                        cftime.DatetimeGregorian(
                            2000, 1, 4, 0, 0, 0, 0, has_year_zero=False
                        ),
                    ],
                    [
                        cftime.DatetimeGregorian(
                            2000, 1, 4, 0, 0, 0, 0, has_year_zero=False
                        ),
                        cftime.DatetimeGregorian(
                            2000, 1, 5, 0, 0, 0, 0, has_year_zero=False
                        ),
                    ],
                    [
                        cftime.DatetimeGregorian(
                            2000, 1, 5, 0, 0, 0, 0, has_year_zero=False
                        ),
                        cftime.DatetimeGregorian(
                            2000, 1, 6, 0, 0, 0, 0, has_year_zero=False
                        ),
                    ],
                    [
                        cftime.DatetimeGregorian(
                            2000, 1, 6, 0, 0, 0, 0, has_year_zero=False
                        ),
                        cftime.DatetimeGregorian(
                            2000, 1, 7, 0, 0, 0, 0, has_year_zero=False
                        ),
                    ],
                    [
                        cftime.DatetimeGregorian(
                            2000, 1, 7, 0, 0, 0, 0, has_year_zero=False
                        ),
                        cftime.DatetimeGregorian(
                            2000, 1, 8, 0, 0, 0, 0, has_year_zero=False
                        ),
                    ],
                    [
                        cftime.DatetimeGregorian(
                            2000, 1, 8, 0, 0, 0, 0, has_year_zero=False
                        ),
                        cftime.DatetimeGregorian(
                            2000, 1, 9, 0, 0, 0, 0, has_year_zero=False
                        ),
                    ],
                    [
                        cftime.DatetimeGregorian(
                            2000, 1, 9, 0, 0, 0, 0, has_year_zero=False
                        ),
                        cftime.DatetimeGregorian(
                            2000, 1, 10, 0, 0, 0, 0, has_year_zero=False
                        ),
                    ],
                    [
                        cftime.DatetimeGregorian(
                            2000, 1, 10, 0, 0, 0, 0, has_year_zero=False
                        ),
                        cftime.DatetimeGregorian(
                            2000, 1, 11, 0, 0, 0, 0, has_year_zero=False
                        ),
                    ],
                    [
                        cftime.DatetimeGregorian(
                            2000, 1, 11, 0, 0, 0, 0, has_year_zero=False
                        ),
                        cftime.DatetimeGregorian(
                            2000, 1, 12, 0, 0, 0, 0, has_year_zero=False
                        ),
                    ],
                    [
                        cftime.DatetimeGregorian(
                            2000, 1, 12, 0, 0, 0, 0, has_year_zero=False
                        ),
                        cftime.DatetimeGregorian(
                            2000, 1, 13, 0, 0, 0, 0, has_year_zero=False
                        ),
                    ],
                    [
                        cftime.DatetimeGregorian(
                            2000, 1, 13, 0, 0, 0, 0, has_year_zero=False
                        ),
                        cftime.DatetimeGregorian(
                            2000, 1, 14, 0, 0, 0, 0, has_year_zero=False
                        ),
                    ],
                    [
                        cftime.DatetimeGregorian(
                            2000, 1, 14, 0, 0, 0, 0, has_year_zero=False
                        ),
                        cftime.DatetimeGregorian(
                            2000, 1, 15, 0, 0, 0, 0, has_year_zero=False
                        ),
                    ],
                ],
                dtype="object",
            ),
            dims=["time", "bnds"],
            attrs={"xcdat_bounds": "True"},
        )
        expected.time.attrs = {
            "axis": "T",
            "long_name": "time",
            "standard_name": "time",
            "bounds": "time_bnds",
        }

        assert result.identical(expected)

        # Check encoding is preserved.
        expected.time.encoding = {
            "zlib": False,
            "szip": False,
            "zstd": False,
            "bzip2": False,
            "blosc": False,
            "shuffle": False,
            "complevel": 0,
            "fletcher32": False,
            "contiguous": True,
            "chunksizes": None,
            # Set source as result source because it changes every test run.
            "source": result.time.encoding["source"],
            "original_shape": (15,),
            "dtype": np.dtype("int64"),
            "units": "days since 2000-01-01",
            "calendar": "standard",
        }
        expected.time_bnds.encoding = {
            "zlib": False,
            "szip": False,
            "zstd": False,
            "bzip2": False,
            "blosc": False,
            "shuffle": False,
            "complevel": 0,
            "fletcher32": False,
            "contiguous": True,
            "chunksizes": None,
            "source": result.time.encoding["source"],
            "original_shape": (15, 2),
            "dtype": np.dtype("int64"),
            "units": "days since 2000-01-01",
            "calendar": "standard",
        }

        assert result.time.encoding == expected.time.encoding
        assert result.time_bnds.encoding == expected.time_bnds.encoding

    def test_decode_time_in_months(self):
        ds = generate_dataset(decode_times=False, cf_compliant=False, has_bounds=True)
        ds.to_netcdf(self.file_path)

        result = open_dataset(self.file_path, data_var="ts")

        # Generate an expected dataset with decoded non-CF compliant time units.
        expected = ds.copy()
        expected["time"] = xr.DataArray(
            name="time",
            data=np.array(
                [
                    cftime.DatetimeGregorian(
                        2000, 1, 1, 0, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2000, 2, 1, 0, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2000, 3, 1, 0, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2000, 4, 1, 0, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2000, 5, 1, 0, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2000, 6, 1, 0, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2000, 7, 1, 0, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2000, 8, 1, 0, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2000, 9, 1, 0, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2000, 10, 1, 0, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2000, 11, 1, 0, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2000, 12, 1, 0, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2001, 1, 1, 0, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2001, 2, 1, 0, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2001, 3, 1, 0, 0, 0, 0, has_year_zero=False
                    ),
                ],
                dtype="object",
            ),
            dims="time",
        )

        expected["time_bnds"] = xr.DataArray(
            name="time_bnds",
            data=np.array(
                [
                    [
                        cftime.DatetimeGregorian(
                            1999, 12, 1, 0, 0, 0, 0, has_year_zero=False
                        ),
                        cftime.DatetimeGregorian(
                            2000, 1, 1, 0, 0, 0, 0, has_year_zero=False
                        ),
                    ],
                    [
                        cftime.DatetimeGregorian(
                            2000, 1, 1, 0, 0, 0, 0, has_year_zero=False
                        ),
                        cftime.DatetimeGregorian(
                            2000, 2, 1, 0, 0, 0, 0, has_year_zero=False
                        ),
                    ],
                    [
                        cftime.DatetimeGregorian(
                            2000, 2, 1, 0, 0, 0, 0, has_year_zero=False
                        ),
                        cftime.DatetimeGregorian(
                            2000, 3, 1, 0, 0, 0, 0, has_year_zero=False
                        ),
                    ],
                    [
                        cftime.DatetimeGregorian(
                            2000, 3, 1, 0, 0, 0, 0, has_year_zero=False
                        ),
                        cftime.DatetimeGregorian(
                            2000, 4, 1, 0, 0, 0, 0, has_year_zero=False
                        ),
                    ],
                    [
                        cftime.DatetimeGregorian(
                            2000, 4, 1, 0, 0, 0, 0, has_year_zero=False
                        ),
                        cftime.DatetimeGregorian(
                            2000, 5, 1, 0, 0, 0, 0, has_year_zero=False
                        ),
                    ],
                    [
                        cftime.DatetimeGregorian(
                            2000, 5, 1, 0, 0, 0, 0, has_year_zero=False
                        ),
                        cftime.DatetimeGregorian(
                            2000, 6, 1, 0, 0, 0, 0, has_year_zero=False
                        ),
                    ],
                    [
                        cftime.DatetimeGregorian(
                            2000, 6, 1, 0, 0, 0, 0, has_year_zero=False
                        ),
                        cftime.DatetimeGregorian(
                            2000, 7, 1, 0, 0, 0, 0, has_year_zero=False
                        ),
                    ],
                    [
                        cftime.DatetimeGregorian(
                            2000, 7, 1, 0, 0, 0, 0, has_year_zero=False
                        ),
                        cftime.DatetimeGregorian(
                            2000, 8, 1, 0, 0, 0, 0, has_year_zero=False
                        ),
                    ],
                    [
                        cftime.DatetimeGregorian(
                            2000, 8, 1, 0, 0, 0, 0, has_year_zero=False
                        ),
                        cftime.DatetimeGregorian(
                            2000, 9, 1, 0, 0, 0, 0, has_year_zero=False
                        ),
                    ],
                    [
                        cftime.DatetimeGregorian(
                            2000, 9, 1, 0, 0, 0, 0, has_year_zero=False
                        ),
                        cftime.DatetimeGregorian(
                            2000, 10, 1, 0, 0, 0, 0, has_year_zero=False
                        ),
                    ],
                    [
                        cftime.DatetimeGregorian(
                            2000, 10, 1, 0, 0, 0, 0, has_year_zero=False
                        ),
                        cftime.DatetimeGregorian(
                            2000, 11, 1, 0, 0, 0, 0, has_year_zero=False
                        ),
                    ],
                    [
                        cftime.DatetimeGregorian(
                            2000, 11, 1, 0, 0, 0, 0, has_year_zero=False
                        ),
                        cftime.DatetimeGregorian(
                            2000, 12, 1, 0, 0, 0, 0, has_year_zero=False
                        ),
                    ],
                    [
                        cftime.DatetimeGregorian(
                            2000, 12, 1, 0, 0, 0, 0, has_year_zero=False
                        ),
                        cftime.DatetimeGregorian(
                            2001, 1, 1, 0, 0, 0, 0, has_year_zero=False
                        ),
                    ],
                    [
                        cftime.DatetimeGregorian(
                            2001, 1, 1, 0, 0, 0, 0, has_year_zero=False
                        ),
                        cftime.DatetimeGregorian(
                            2001, 2, 1, 0, 0, 0, 0, has_year_zero=False
                        ),
                    ],
                    [
                        cftime.DatetimeGregorian(
                            2001, 2, 1, 0, 0, 0, 0, has_year_zero=False
                        ),
                        cftime.DatetimeGregorian(
                            2001, 3, 1, 0, 0, 0, 0, has_year_zero=False
                        ),
                    ],
                ],
                dtype="object",
            ),
            dims=["time", "bnds"],
            attrs={"xcdat_bounds": "True"},
        )

        expected.time.attrs = {
            "axis": "T",
            "long_name": "time",
            "standard_name": "time",
            "bounds": "time_bnds",
        }

        assert result.identical(expected)

        # Check encoding is preserved.
        expected.time.encoding = {
            "zlib": False,
            "szip": False,
            "zstd": False,
            "bzip2": False,
            "blosc": False,
            "shuffle": False,
            "complevel": 0,
            "fletcher32": False,
            "contiguous": True,
            "chunksizes": None,
            # Set source as result source because it changes every test run.
            "source": result.time.encoding["source"],
            "original_shape": (15,),
            "dtype": np.dtype("int64"),
            "units": "months since 2000-01-01",
            "calendar": "standard",
        }
        expected.time_bnds.encoding = {
            "zlib": False,
            "szip": False,
            "zstd": False,
            "bzip2": False,
            "blosc": False,
            "shuffle": False,
            "complevel": 0,
            "fletcher32": False,
            "contiguous": True,
            "chunksizes": None,
            # Set source as result source because it changes every test run.
            "source": result.time.encoding["source"],
            "original_shape": (15, 2),
            "dtype": np.dtype("int64"),
            "units": "months since 2000-01-01",
            "calendar": "standard",
        }

        assert result.time.encoding == expected.time.encoding
        assert result.time_bnds.encoding == expected.time_bnds.encoding

    def test_keeps_specified_var_and_preserves_bounds(self):
        ds = generate_dataset(decode_times=True, cf_compliant=True, has_bounds=True)

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

        assert result.identical(expected)


class TestOpenMfDataset:
    @pytest.fixture(autouse=True)
    def setUp(self, tmp_path):
        # Create temporary directory to save files.
        dir = tmp_path / "input_data"
        dir.mkdir()
        self.file_path1 = f"{dir}/file1.nc"
        self.file_path2 = f"{dir}/file2.nc"

    def test_raises_warning_if_decode_times_but_no_time_coords_found(self, caplog):
        ds = generate_dataset(decode_times=False, cf_compliant=True, has_bounds=True)
        ds = ds.drop_dims("time")
        ds.to_netcdf(self.file_path1)

        result = open_mfdataset(self.file_path1)
        expected = generate_dataset(
            decode_times=False,
            cf_compliant=True,
            has_bounds=True,
        )
        expected = expected.drop_dims("time")

        assert result.identical(expected)
        assert (
            "No time coordinates were found in this dataset to decode. Make sure time "
            "coordinates exist and their CF 'axis' or 'standard_name' attribute is set "
            "(e.g., ds['time'].attrs['axis'] = 'T' or "
            "ds['time'].attrs['standard_name'] = 'time'). Afterwards, try decoding "
            "with `decode_time()` again."
        ) in caplog.text

    def test_skip_decoding_times_explicitly(self):
        ds1 = generate_dataset(decode_times=False, cf_compliant=False, has_bounds=True)
        ds1.to_netcdf(self.file_path1)
        ds2 = generate_dataset(decode_times=False, cf_compliant=False, has_bounds=True)
        ds2 = ds2.rename_vars({"ts": "tas"})
        ds2.to_netcdf(self.file_path2)

        result = open_mfdataset([self.file_path1, self.file_path2], decode_times=False)

        expected = ds1.merge(ds2)
        assert result.identical(expected)

    def test_user_specified_callable_results_in_subsetting_dataset_on_time_slice(self):
        def callable(ds):
            return ds.isel(time=slice(0, 1))

        ds = generate_dataset(decode_times=False, cf_compliant=False, has_bounds=True)
        ds.to_netcdf(self.file_path1)

        result = open_mfdataset(self.file_path1, decode_times=True, preprocess=callable)
        expected = ds.copy().isel(time=slice(0, 1))
        expected["time"] = xr.DataArray(
            name="time",
            data=np.array([cftime.datetime(2000, 1, 1)]),
            dims=["time"],
        )
        expected["time_bnds"] = xr.DataArray(
            name="time_bnds",
            data=np.array(
                [[cftime.datetime(1999, 12, 1), cftime.datetime(2000, 1, 1)]],
            ),
            dims=["time", "bnds"],
        )
        expected.time.attrs = {
            "axis": "T",
            "long_name": "time",
            "standard_name": "time",
            "bounds": "time_bnds",
        }

        expected.time_bnds.attrs = {"xcdat_bounds": "True"}

        assert result.identical(expected)

    def test_decode_time_in_months(self):
        # Generate two dataset files with different variables.
        ds1 = generate_dataset(decode_times=False, cf_compliant=False, has_bounds=True)
        ds1.to_netcdf(self.file_path1)

        ds2 = generate_dataset(decode_times=False, cf_compliant=False, has_bounds=True)
        ds2 = ds2.rename_vars({"ts": "tas"})
        ds2.to_netcdf(self.file_path2)

        # Open both dataset files as a single Dataset object.
        result = open_mfdataset([self.file_path1, self.file_path2], data_var="ts")

        # Create an expected Dataset object.
        expected = generate_dataset(
            decode_times=True, cf_compliant=False, has_bounds=True
        )
        expected["time"] = xr.DataArray(
            name="time",
            data=np.array(
                [
                    cftime.DatetimeGregorian(
                        2000, 1, 1, 0, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2000, 2, 1, 0, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2000, 3, 1, 0, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2000, 4, 1, 0, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2000, 5, 1, 0, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2000, 6, 1, 0, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2000, 7, 1, 0, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2000, 8, 1, 0, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2000, 9, 1, 0, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2000, 10, 1, 0, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2000, 11, 1, 0, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2000, 12, 1, 0, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2001, 1, 1, 0, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2001, 2, 1, 0, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2001, 3, 1, 0, 0, 0, 0, has_year_zero=False
                    ),
                ],
                dtype="object",
            ),
            dims="time",
            attrs={
                "axis": "T",
                "long_name": "time",
                "standard_name": "time",
                "bounds": "time_bnds",
            },
        )

        expected["time_bnds"] = xr.DataArray(
            name="time_bnds",
            data=np.array(
                [
                    [
                        cftime.DatetimeGregorian(
                            1999, 12, 1, 0, 0, 0, 0, has_year_zero=False
                        ),
                        cftime.DatetimeGregorian(
                            2000, 1, 1, 0, 0, 0, 0, has_year_zero=False
                        ),
                    ],
                    [
                        cftime.DatetimeGregorian(
                            2000, 1, 1, 0, 0, 0, 0, has_year_zero=False
                        ),
                        cftime.DatetimeGregorian(
                            2000, 2, 1, 0, 0, 0, 0, has_year_zero=False
                        ),
                    ],
                    [
                        cftime.DatetimeGregorian(
                            2000, 2, 1, 0, 0, 0, 0, has_year_zero=False
                        ),
                        cftime.DatetimeGregorian(
                            2000, 3, 1, 0, 0, 0, 0, has_year_zero=False
                        ),
                    ],
                    [
                        cftime.DatetimeGregorian(
                            2000, 3, 1, 0, 0, 0, 0, has_year_zero=False
                        ),
                        cftime.DatetimeGregorian(
                            2000, 4, 1, 0, 0, 0, 0, has_year_zero=False
                        ),
                    ],
                    [
                        cftime.DatetimeGregorian(
                            2000, 4, 1, 0, 0, 0, 0, has_year_zero=False
                        ),
                        cftime.DatetimeGregorian(
                            2000, 5, 1, 0, 0, 0, 0, has_year_zero=False
                        ),
                    ],
                    [
                        cftime.DatetimeGregorian(
                            2000, 5, 1, 0, 0, 0, 0, has_year_zero=False
                        ),
                        cftime.DatetimeGregorian(
                            2000, 6, 1, 0, 0, 0, 0, has_year_zero=False
                        ),
                    ],
                    [
                        cftime.DatetimeGregorian(
                            2000, 6, 1, 0, 0, 0, 0, has_year_zero=False
                        ),
                        cftime.DatetimeGregorian(
                            2000, 7, 1, 0, 0, 0, 0, has_year_zero=False
                        ),
                    ],
                    [
                        cftime.DatetimeGregorian(
                            2000, 7, 1, 0, 0, 0, 0, has_year_zero=False
                        ),
                        cftime.DatetimeGregorian(
                            2000, 8, 1, 0, 0, 0, 0, has_year_zero=False
                        ),
                    ],
                    [
                        cftime.DatetimeGregorian(
                            2000, 8, 1, 0, 0, 0, 0, has_year_zero=False
                        ),
                        cftime.DatetimeGregorian(
                            2000, 9, 1, 0, 0, 0, 0, has_year_zero=False
                        ),
                    ],
                    [
                        cftime.DatetimeGregorian(
                            2000, 9, 1, 0, 0, 0, 0, has_year_zero=False
                        ),
                        cftime.DatetimeGregorian(
                            2000, 10, 1, 0, 0, 0, 0, has_year_zero=False
                        ),
                    ],
                    [
                        cftime.DatetimeGregorian(
                            2000, 10, 1, 0, 0, 0, 0, has_year_zero=False
                        ),
                        cftime.DatetimeGregorian(
                            2000, 11, 1, 0, 0, 0, 0, has_year_zero=False
                        ),
                    ],
                    [
                        cftime.DatetimeGregorian(
                            2000, 11, 1, 0, 0, 0, 0, has_year_zero=False
                        ),
                        cftime.DatetimeGregorian(
                            2000, 12, 1, 0, 0, 0, 0, has_year_zero=False
                        ),
                    ],
                    [
                        cftime.DatetimeGregorian(
                            2000, 12, 1, 0, 0, 0, 0, has_year_zero=False
                        ),
                        cftime.DatetimeGregorian(
                            2001, 1, 1, 0, 0, 0, 0, has_year_zero=False
                        ),
                    ],
                    [
                        cftime.DatetimeGregorian(
                            2001, 1, 1, 0, 0, 0, 0, has_year_zero=False
                        ),
                        cftime.DatetimeGregorian(
                            2001, 2, 1, 0, 0, 0, 0, has_year_zero=False
                        ),
                    ],
                    [
                        cftime.DatetimeGregorian(
                            2001, 2, 1, 0, 0, 0, 0, has_year_zero=False
                        ),
                        cftime.DatetimeGregorian(
                            2001, 3, 1, 0, 0, 0, 0, has_year_zero=False
                        ),
                    ],
                ],
                dtype="object",
            ),
            dims=["time", "bnds"],
            attrs={"xcdat_bounds": "True"},
        )

        # Make sure the expected is chunked.
        expected = expected.chunk(chunks={"time": 15, "bnds": 2})

        # Check encoding is preserved. The extra metadata like "zlib" are from
        # the netCDF4 files.
        expected.time.encoding = {
            "zlib": False,
            "szip": False,
            "zstd": False,
            "bzip2": False,
            "blosc": False,
            "shuffle": False,
            "complevel": 0,
            "fletcher32": False,
            "contiguous": True,
            "chunksizes": None,
            # Set source as result source because it changes every test run.
            "source": result.time.encoding["source"],
            "original_shape": (15,),
            "dtype": np.dtype("int64"),
            "units": "months since 2000-01-01",
            "calendar": "standard",
        }
        expected.time_bnds.encoding = {
            "units": "months since 2000-01-01",
            "calendar": "standard",
        }

        assert result.identical(expected)
        assert result.time.encoding == expected.time.encoding

        # FIXME: For some reason the encoding attributes get dropped only in
        # the test and not real-world datasets.
        assert result.time_bnds.encoding != expected.time_bnds.encoding

    def test_keeps_specified_var_and_preserves_bounds(self):
        # Generate two dataset files with different variables.
        ds1 = generate_dataset(decode_times=False, cf_compliant=False, has_bounds=True)
        ds1.to_netcdf(self.file_path1)

        ds2 = generate_dataset(decode_times=False, cf_compliant=False, has_bounds=True)
        ds2 = ds2.rename_vars({"ts": "tas"})
        ds2.to_netcdf(self.file_path2)

        # Open both dataset files as a single Dataset object.
        result = open_mfdataset(
            [self.file_path1, self.file_path2], data_var="ts", decode_times=False
        )

        # Create an expected Dataset object and check identical with result.
        expected = generate_dataset(
            decode_times=False, cf_compliant=False, has_bounds=True
        )
        expected = expected.chunk(chunks={"time": 15, "bnds": 2})

        assert result.identical(expected)


class TestDecodeTime:
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
            },
        )
        time_bnds = xr.DataArray(
            name="time_bnds",
            data=[[0, 1], [1, 2], [2, 3]],
            dims=["time", "bnds"],
        )
        self.ds = xr.Dataset({"time": time, "time_bnds": time_bnds})

    def test_raises_error_if_no_time_coordinates_could_be_mapped_to(self, caplog):
        ds = generate_dataset(decode_times=False, cf_compliant=False, has_bounds=True)

        # Remove time attributes and rename the coordinate variable before
        # attempting to decode.
        ds.time.attrs = {}
        ds = ds.rename({"time": "invalid_time"})

        with pytest.raises(KeyError):
            decode_time(ds)

    def test_skips_decoding_time_coords_if_units_is_not_set(self, caplog):
        # Update logger level to silence the logger warning during test runs.
        caplog.set_level(logging.ERROR)

        ds = generate_dataset(decode_times=False, cf_compliant=False, has_bounds=True)

        del ds.time.attrs["units"]

        result = decode_time(ds)
        assert ds.identical(result)

    def test_skips_decoding_time_coords_if_units_is_not_supported(self, caplog):
        # Update logger level to silence the logger warning during test runs.
        caplog.set_level(logging.ERROR)

        ds = generate_dataset(decode_times=False, cf_compliant=False, has_bounds=True)

        ds.time.attrs["units"] = "year AD"

        result = decode_time(ds)
        assert ds.identical(result)

    def test_skips_decoding_time_bounds_if_bounds_dont_exist(self):
        ds = xr.Dataset(
            coords={
                "time": xr.DataArray(
                    name="time",
                    data=[1, 2, 3],
                    dims=["time"],
                    attrs={
                        "bounds": "time_bnds",
                        "axis": "T",
                        "long_name": "time",
                        "standard_name": "time",
                        "calendar": "standard",
                        "units": "months since 2000-01-01",
                    },
                ),
                "time2": xr.DataArray(
                    name="time2",
                    data=[1, 2, 3],
                    dims="time",
                    attrs={
                        "bounds": "time_bnds",
                        "axis": "T",
                        "long_name": "time",
                        "standard_name": "time",
                        "calendar": "standard",
                        "units": "months since 2000-01-01",
                    },
                ),
            },
        )

        result = decode_time(ds)
        expected = xr.Dataset(
            coords={
                "time": xr.DataArray(
                    name="time",
                    data=np.array(
                        [
                            cftime.DatetimeGregorian(
                                2000, 2, 1, 0, 0, 0, 0, has_year_zero=False
                            ),
                            cftime.DatetimeGregorian(
                                2000, 3, 1, 0, 0, 0, 0, has_year_zero=False
                            ),
                            cftime.DatetimeGregorian(
                                2000, 4, 1, 0, 0, 0, 0, has_year_zero=False
                            ),
                        ],
                        dtype="object",
                    ),
                    dims=["time"],
                    attrs={
                        "bounds": "time_bnds",
                        "axis": "T",
                        "long_name": "time",
                        "standard_name": "time",
                    },
                ),
                "time2": xr.DataArray(
                    name="time2",
                    data=np.array(
                        [
                            cftime.DatetimeGregorian(
                                2000, 2, 1, 0, 0, 0, 0, has_year_zero=False
                            ),
                            cftime.DatetimeGregorian(
                                2000, 3, 1, 0, 0, 0, 0, has_year_zero=False
                            ),
                            cftime.DatetimeGregorian(
                                2000, 4, 1, 0, 0, 0, 0, has_year_zero=False
                            ),
                        ],
                        dtype="object",
                    ),
                    dims=["time"],
                    attrs={
                        "bounds": "time_bnds",
                        "axis": "T",
                        "long_name": "time",
                        "standard_name": "time",
                    },
                ),
            },
        )

        assert result.identical(expected)

        # Check encoding is preserved.
        expected.time.encoding = {
            "units": "months since 2000-01-01",
            "calendar": "standard",
        }
        expected.time2.encoding = {
            "units": "months since 2000-01-01",
            "calendar": "standard",
        }

        assert result.time.encoding == expected.time.encoding
        assert result.time2.encoding == expected.time.encoding

    def test_decodes_all_time_coordinates_and_time_bounds(self):
        ds = xr.Dataset(
            coords={
                "time": xr.DataArray(
                    name="time",
                    data=[1, 2, 3],
                    dims=["time"],
                    attrs={
                        "bounds": "time_bnds",
                        "axis": "T",
                        "long_name": "time",
                        "standard_name": "time",
                        "calendar": "standard",
                        "units": "months since 2000-01-01",
                    },
                ),
                "time2": xr.DataArray(
                    name="time2",
                    data=[1, 2, 3],
                    dims="time",
                    attrs={
                        "bounds": "time_bnds",
                        "axis": "T",
                        "long_name": "time",
                        "standard_name": "time",
                        "calendar": "standard",
                        "units": "months since 2000-01-01",
                    },
                ),
            },
            data_vars={
                "time_bnds": xr.DataArray(
                    name="time_bnds",
                    data=[[0, 1], [1, 2], [2, 3]],
                    dims=["time", "bnds"],
                )
            },
        )

        result = decode_time(ds)
        expected = xr.Dataset(
            coords={
                "time": xr.DataArray(
                    name="time",
                    data=np.array(
                        [
                            cftime.DatetimeGregorian(
                                2000, 2, 1, 0, 0, 0, 0, has_year_zero=False
                            ),
                            cftime.DatetimeGregorian(
                                2000, 3, 1, 0, 0, 0, 0, has_year_zero=False
                            ),
                            cftime.DatetimeGregorian(
                                2000, 4, 1, 0, 0, 0, 0, has_year_zero=False
                            ),
                        ],
                        dtype="object",
                    ),
                    dims=["time"],
                    attrs={
                        "bounds": "time_bnds",
                        "axis": "T",
                        "long_name": "time",
                        "standard_name": "time",
                    },
                ),
                "time2": xr.DataArray(
                    name="time2",
                    data=np.array(
                        [
                            cftime.DatetimeGregorian(
                                2000, 2, 1, 0, 0, 0, 0, has_year_zero=False
                            ),
                            cftime.DatetimeGregorian(
                                2000, 3, 1, 0, 0, 0, 0, has_year_zero=False
                            ),
                            cftime.DatetimeGregorian(
                                2000, 4, 1, 0, 0, 0, 0, has_year_zero=False
                            ),
                        ],
                        dtype="object",
                    ),
                    dims=["time"],
                    attrs={
                        "bounds": "time_bnds",
                        "axis": "T",
                        "long_name": "time",
                        "standard_name": "time",
                    },
                ),
            },
            data_vars={
                "time_bnds": xr.DataArray(
                    name="time_bnds",
                    data=np.array(
                        [
                            [
                                cftime.DatetimeGregorian(
                                    2000, 1, 1, 0, 0, 0, 0, has_year_zero=False
                                ),
                                cftime.DatetimeGregorian(
                                    2000, 2, 1, 0, 0, 0, 0, has_year_zero=False
                                ),
                            ],
                            [
                                cftime.DatetimeGregorian(
                                    2000, 2, 1, 0, 0, 0, 0, has_year_zero=False
                                ),
                                cftime.DatetimeGregorian(
                                    2000, 3, 1, 0, 0, 0, 0, has_year_zero=False
                                ),
                            ],
                            [
                                cftime.DatetimeGregorian(
                                    2000, 3, 1, 0, 0, 0, 0, has_year_zero=False
                                ),
                                cftime.DatetimeGregorian(
                                    2000, 4, 1, 0, 0, 0, 0, has_year_zero=False
                                ),
                            ],
                        ],
                        dtype="object",
                    ),
                    dims=["time", "bnds"],
                    attrs=ds.time_bnds.attrs,
                ),
            },
        )

        assert result.identical(expected)

        # Check the encoding is preserved.
        expected.time.encoding = {
            "units": "months since 2000-01-01",
            "calendar": "standard",
        }
        expected.time2.encoding = {
            "units": "months since 2000-01-01",
            "calendar": "standard",
        }
        expected.time_bnds.encoding = {
            "units": "months since 2000-01-01",
            "calendar": "standard",
        }

        assert result.time.encoding == expected.time.encoding
        assert result.time2.encoding == expected.time2.encoding
        assert result.time_bnds.encoding == expected.time_bnds.encoding

    def test_decodes_time_coords_and_bounds_without_calendar_attr_set(self, caplog):
        # Update logger level to silence the logger warning during test runs.
        caplog.set_level(logging.ERROR)

        ds = xr.Dataset(
            coords={
                "time": xr.DataArray(
                    name="time",
                    data=[1, 2, 3],
                    dims=["time"],
                    attrs={
                        "bounds": "time_bnds",
                        "axis": "T",
                        "long_name": "time",
                        "standard_name": "time",
                        "units": "months since 2000-01-01",
                    },
                ),
            },
            data_vars={
                "time_bnds": xr.DataArray(
                    name="time_bnds",
                    data=[[0, 1], [1, 2], [2, 3]],
                    dims=["time", "bnds"],
                )
            },
        )

        result = decode_time(ds)
        expected = xr.Dataset(
            coords={
                "time": xr.DataArray(
                    name="time",
                    data=np.array(
                        [
                            cftime.DatetimeGregorian(
                                2000, 2, 1, 0, 0, 0, 0, has_year_zero=False
                            ),
                            cftime.DatetimeGregorian(
                                2000, 3, 1, 0, 0, 0, 0, has_year_zero=False
                            ),
                            cftime.DatetimeGregorian(
                                2000, 4, 1, 0, 0, 0, 0, has_year_zero=False
                            ),
                        ],
                        dtype="object",
                    ),
                    dims=["time"],
                    attrs={
                        "bounds": "time_bnds",
                        "axis": "T",
                        "long_name": "time",
                        "standard_name": "time",
                    },
                ),
            },
            data_vars={
                "time_bnds": xr.DataArray(
                    name="time_bnds",
                    data=np.array(
                        [
                            [
                                cftime.DatetimeGregorian(
                                    2000, 1, 1, 0, 0, 0, 0, has_year_zero=False
                                ),
                                cftime.DatetimeGregorian(
                                    2000, 2, 1, 0, 0, 0, 0, has_year_zero=False
                                ),
                            ],
                            [
                                cftime.DatetimeGregorian(
                                    2000, 2, 1, 0, 0, 0, 0, has_year_zero=False
                                ),
                                cftime.DatetimeGregorian(
                                    2000, 3, 1, 0, 0, 0, 0, has_year_zero=False
                                ),
                            ],
                            [
                                cftime.DatetimeGregorian(
                                    2000, 3, 1, 0, 0, 0, 0, has_year_zero=False
                                ),
                                cftime.DatetimeGregorian(
                                    2000, 4, 1, 0, 0, 0, 0, has_year_zero=False
                                ),
                            ],
                        ],
                        dtype="object",
                    ),
                    dims=["time", "bnds"],
                ),
            },
        )

        assert result.identical(expected)

        # Check the encoding is preserved.
        expected.time.encoding = {
            "units": "months since 2000-01-01",
            "calendar": "standard",
        }
        expected.time_bnds.encoding = {
            "units": "months since 2000-01-01",
            "calendar": "standard",
        }

        assert result.time.encoding == expected.time.encoding
        assert result.time_bnds.encoding == expected.time_bnds.encoding

    def test_decode_time_in_days(self):
        ds = generate_dataset(decode_times=False, cf_compliant=True, has_bounds=True)

        result = decode_time(ds)

        # Generate an expected dataset with decoded CF compliant time units.
        expected = ds.copy()
        expected["time"] = xr.DataArray(
            name="time",
            data=np.array(
                [
                    cftime.DatetimeGregorian(
                        2000, 1, 1, 0, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2000, 1, 2, 0, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2000, 1, 3, 0, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2000, 1, 4, 0, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2000, 1, 5, 0, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2000, 1, 6, 0, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2000, 1, 7, 0, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2000, 1, 8, 0, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2000, 1, 9, 0, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2000, 1, 10, 0, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2000, 1, 11, 0, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2000, 1, 12, 0, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2000, 1, 13, 0, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2000, 1, 14, 0, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2000, 1, 15, 0, 0, 0, 0, has_year_zero=False
                    ),
                ],
                dtype="object",
            ),
            dims="time",
        )
        expected["time_bnds"] = xr.DataArray(
            name="time_bnds",
            data=np.array(
                [
                    [
                        cftime.DatetimeGregorian(
                            1999, 12, 31, 0, 0, 0, 0, has_year_zero=False
                        ),
                        cftime.DatetimeGregorian(
                            2000, 1, 1, 0, 0, 0, 0, has_year_zero=False
                        ),
                    ],
                    [
                        cftime.DatetimeGregorian(
                            2000, 1, 1, 0, 0, 0, 0, has_year_zero=False
                        ),
                        cftime.DatetimeGregorian(
                            2000, 1, 2, 0, 0, 0, 0, has_year_zero=False
                        ),
                    ],
                    [
                        cftime.DatetimeGregorian(
                            2000, 1, 2, 0, 0, 0, 0, has_year_zero=False
                        ),
                        cftime.DatetimeGregorian(
                            2000, 1, 3, 0, 0, 0, 0, has_year_zero=False
                        ),
                    ],
                    [
                        cftime.DatetimeGregorian(
                            2000, 1, 3, 0, 0, 0, 0, has_year_zero=False
                        ),
                        cftime.DatetimeGregorian(
                            2000, 1, 4, 0, 0, 0, 0, has_year_zero=False
                        ),
                    ],
                    [
                        cftime.DatetimeGregorian(
                            2000, 1, 4, 0, 0, 0, 0, has_year_zero=False
                        ),
                        cftime.DatetimeGregorian(
                            2000, 1, 5, 0, 0, 0, 0, has_year_zero=False
                        ),
                    ],
                    [
                        cftime.DatetimeGregorian(
                            2000, 1, 5, 0, 0, 0, 0, has_year_zero=False
                        ),
                        cftime.DatetimeGregorian(
                            2000, 1, 6, 0, 0, 0, 0, has_year_zero=False
                        ),
                    ],
                    [
                        cftime.DatetimeGregorian(
                            2000, 1, 6, 0, 0, 0, 0, has_year_zero=False
                        ),
                        cftime.DatetimeGregorian(
                            2000, 1, 7, 0, 0, 0, 0, has_year_zero=False
                        ),
                    ],
                    [
                        cftime.DatetimeGregorian(
                            2000, 1, 7, 0, 0, 0, 0, has_year_zero=False
                        ),
                        cftime.DatetimeGregorian(
                            2000, 1, 8, 0, 0, 0, 0, has_year_zero=False
                        ),
                    ],
                    [
                        cftime.DatetimeGregorian(
                            2000, 1, 8, 0, 0, 0, 0, has_year_zero=False
                        ),
                        cftime.DatetimeGregorian(
                            2000, 1, 9, 0, 0, 0, 0, has_year_zero=False
                        ),
                    ],
                    [
                        cftime.DatetimeGregorian(
                            2000, 1, 9, 0, 0, 0, 0, has_year_zero=False
                        ),
                        cftime.DatetimeGregorian(
                            2000, 1, 10, 0, 0, 0, 0, has_year_zero=False
                        ),
                    ],
                    [
                        cftime.DatetimeGregorian(
                            2000, 1, 10, 0, 0, 0, 0, has_year_zero=False
                        ),
                        cftime.DatetimeGregorian(
                            2000, 1, 11, 0, 0, 0, 0, has_year_zero=False
                        ),
                    ],
                    [
                        cftime.DatetimeGregorian(
                            2000, 1, 11, 0, 0, 0, 0, has_year_zero=False
                        ),
                        cftime.DatetimeGregorian(
                            2000, 1, 12, 0, 0, 0, 0, has_year_zero=False
                        ),
                    ],
                    [
                        cftime.DatetimeGregorian(
                            2000, 1, 12, 0, 0, 0, 0, has_year_zero=False
                        ),
                        cftime.DatetimeGregorian(
                            2000, 1, 13, 0, 0, 0, 0, has_year_zero=False
                        ),
                    ],
                    [
                        cftime.DatetimeGregorian(
                            2000, 1, 13, 0, 0, 0, 0, has_year_zero=False
                        ),
                        cftime.DatetimeGregorian(
                            2000, 1, 14, 0, 0, 0, 0, has_year_zero=False
                        ),
                    ],
                    [
                        cftime.DatetimeGregorian(
                            2000, 1, 14, 0, 0, 0, 0, has_year_zero=False
                        ),
                        cftime.DatetimeGregorian(
                            2000, 1, 15, 0, 0, 0, 0, has_year_zero=False
                        ),
                    ],
                ],
                dtype="object",
            ),
            dims=["time", "bnds"],
            attrs={"xcdat_bounds": "True"},
        )
        expected.time.attrs = {
            "axis": "T",
            "long_name": "time",
            "standard_name": "time",
            "bounds": "time_bnds",
        }

        assert result.identical(expected)

        # Check encoding is preserved.
        expected.time.encoding = {
            "units": "days since 2000-01-01",
            "calendar": "standard",
        }
        expected.time_bnds.encoding = {
            "units": "days since 2000-01-01",
            "calendar": "standard",
        }

        assert result.time.encoding == expected.time.encoding
        assert result.time_bnds.encoding == expected.time_bnds.encoding

    def test_decodes_time_coords_and_bounds_in_months_with_a_reference_date_at_the_start_of_the_month(
        self,
    ):
        ds = self.ds.copy()
        calendar = "standard"
        ds.time.attrs["calendar"] = calendar
        ds.time.attrs["units"] = "months since 2000-01-01"

        result = decode_time(ds)
        expected = xr.Dataset(
            {
                "time": xr.DataArray(
                    name="time",
                    data=np.array(
                        [
                            cftime.DatetimeGregorian(
                                2000, 2, 1, 0, 0, 0, 0, has_year_zero=False
                            ),
                            cftime.DatetimeGregorian(
                                2000, 3, 1, 0, 0, 0, 0, has_year_zero=False
                            ),
                            cftime.DatetimeGregorian(
                                2000, 4, 1, 0, 0, 0, 0, has_year_zero=False
                            ),
                        ],
                        dtype="object",
                    ),
                    dims=["time"],
                    attrs={
                        "bounds": "time_bnds",
                        "axis": "T",
                        "long_name": "time",
                        "standard_name": "time",
                    },
                ),
                "time_bnds": xr.DataArray(
                    name="time_bnds",
                    data=np.array(
                        [
                            [
                                cftime.DatetimeGregorian(
                                    2000, 1, 1, 0, 0, 0, 0, has_year_zero=False
                                ),
                                cftime.DatetimeGregorian(
                                    2000, 2, 1, 0, 0, 0, 0, has_year_zero=False
                                ),
                            ],
                            [
                                cftime.DatetimeGregorian(
                                    2000, 2, 1, 0, 0, 0, 0, has_year_zero=False
                                ),
                                cftime.DatetimeGregorian(
                                    2000, 3, 1, 0, 0, 0, 0, has_year_zero=False
                                ),
                            ],
                            [
                                cftime.DatetimeGregorian(
                                    2000, 3, 1, 0, 0, 0, 0, has_year_zero=False
                                ),
                                cftime.DatetimeGregorian(
                                    2000, 4, 1, 0, 0, 0, 0, has_year_zero=False
                                ),
                            ],
                        ],
                        dtype="object",
                    ),
                    dims=["time", "bnds"],
                ),
            }
        )
        assert result.identical(expected)

        # Check the encoding is preserved.
        expected.time.encoding = {
            "units": "months since 2000-01-01",
            "calendar": "standard",
        }
        expected.time_bnds.encoding = {
            "units": "months since 2000-01-01",
            "calendar": "standard",
        }

        assert result.time.encoding == expected.time.encoding
        assert result.time_bnds.encoding == expected.time_bnds.encoding

    def test_decodes_time_coords_and_bounds_in_months_with_a_reference_date_at_the_middle_of_the_month(
        self,
    ):
        ds = self.ds.copy()
        calendar = "standard"
        ds.time.attrs["calendar"] = calendar
        ds.time.attrs["units"] = "months since 2000-01-15"

        result = decode_time(ds)
        expected = xr.Dataset(
            {
                "time": xr.DataArray(
                    name="time",
                    data=np.array(
                        [
                            cftime.datetime(2000, 2, 15, calendar=calendar),
                            cftime.datetime(2000, 3, 15, calendar=calendar),
                            cftime.datetime(2000, 4, 15, calendar=calendar),
                        ],
                    ),
                    dims=["time"],
                    attrs={
                        "bounds": "time_bnds",
                        "axis": "T",
                        "long_name": "time",
                        "standard_name": "time",
                    },
                ),
                "time_bnds": xr.DataArray(
                    name="time_bnds",
                    data=np.array(
                        [
                            [
                                cftime.datetime(2000, 1, 15, calendar=calendar),
                                cftime.datetime(2000, 2, 15, calendar=calendar),
                            ],
                            [
                                cftime.datetime(2000, 2, 15, calendar=calendar),
                                cftime.datetime(2000, 3, 15, calendar=calendar),
                            ],
                            [
                                cftime.datetime(2000, 3, 15, calendar=calendar),
                                cftime.datetime(2000, 4, 15, calendar=calendar),
                            ],
                        ],
                    ),
                    dims=["time", "bnds"],
                    attrs=ds.time_bnds.attrs,
                ),
            }
        )
        assert result.identical(expected)

        # Check the encoding is preserved.
        expected.time.encoding = {
            "units": "months since 2000-01-15",
            "calendar": "standard",
        }
        expected.time_bnds.encoding = {
            "units": "months since 2000-01-15",
            "calendar": "standard",
        }

        assert result.time.encoding == expected.time.encoding
        assert result.time_bnds.encoding == expected.time_bnds.encoding

    def test_decodes_time_coords_and_bounds_in_months_with_a_reference_date_at_the_end_of_the_month(
        self,
    ):
        ds = self.ds.copy()
        calendar = "standard"
        ds.time.attrs["calendar"] = calendar
        ds.time.attrs["units"] = "months since 1999-12-31"

        result = decode_time(ds)
        expected = xr.Dataset(
            {
                "time": xr.DataArray(
                    name="time",
                    data=np.array(
                        [
                            cftime.datetime(2000, 1, 31, calendar=calendar),
                            cftime.datetime(2000, 2, 29, calendar=calendar),
                            cftime.datetime(2000, 3, 31, calendar=calendar),
                        ],
                    ),
                    dims=["time"],
                    attrs={
                        "bounds": "time_bnds",
                        "axis": "T",
                        "long_name": "time",
                        "standard_name": "time",
                    },
                ),
                "time_bnds": xr.DataArray(
                    name="time_bnds",
                    data=np.array(
                        [
                            [
                                cftime.datetime(1999, 12, 31, calendar=calendar),
                                cftime.datetime(2000, 1, 31, calendar=calendar),
                            ],
                            [
                                cftime.datetime(2000, 1, 31, calendar=calendar),
                                cftime.datetime(2000, 2, 29, calendar=calendar),
                            ],
                            [
                                cftime.datetime(2000, 2, 29, calendar=calendar),
                                cftime.datetime(2000, 3, 31, calendar=calendar),
                            ],
                        ],
                    ),
                    dims=["time", "bnds"],
                    attrs=ds.time_bnds.attrs,
                ),
            }
        )
        assert result.identical(expected)

        # Check the encoding is preserved.
        expected.time.encoding = {
            "units": "months since 1999-12-31",
            "calendar": "standard",
        }
        expected.time_bnds.encoding = {
            "units": "months since 1999-12-31",
            "calendar": "standard",
        }

        assert result.time.encoding == expected.time.encoding
        assert result.time_bnds.encoding == expected.time_bnds.encoding

    def test_decodes_time_coords_and_bounds_in_months_with_a_reference_date_on_a_leap_year(
        self,
    ):
        ds = self.ds.copy()
        calendar = "standard"
        ds.time.attrs["calendar"] = calendar
        ds.time.attrs["units"] = "months since 2000-02-29"

        result = decode_time(ds)

        expected = xr.Dataset(
            {
                "time": xr.DataArray(
                    name="time",
                    data=np.array(
                        [
                            cftime.datetime(2000, 3, 29, calendar=calendar),
                            cftime.datetime(2000, 4, 29, calendar=calendar),
                            cftime.datetime(2000, 5, 29, calendar=calendar),
                        ],
                    ),
                    dims=["time"],
                    attrs={
                        "bounds": "time_bnds",
                        "axis": "T",
                        "long_name": "time",
                        "standard_name": "time",
                    },
                ),
                "time_bnds": xr.DataArray(
                    name="time_bnds",
                    data=np.array(
                        [
                            [
                                cftime.datetime(2000, 2, 29, calendar=calendar),
                                cftime.datetime(2000, 3, 29, calendar=calendar),
                            ],
                            [
                                cftime.datetime(2000, 3, 29, calendar=calendar),
                                cftime.datetime(2000, 4, 29, calendar=calendar),
                            ],
                            [
                                cftime.datetime(2000, 4, 29, calendar=calendar),
                                cftime.datetime(2000, 5, 29, calendar=calendar),
                            ],
                        ],
                    ),
                    dims=["time", "bnds"],
                    attrs=ds.time_bnds.attrs,
                ),
            }
        )
        assert result.identical(expected)

        # Check the encoding is preserved.
        expected.time.encoding = {
            "units": "months since 2000-02-29",
            "calendar": "standard",
        }
        expected.time_bnds.encoding = {
            "units": "months since 2000-02-29",
            "calendar": "standard",
        }

        assert result.time.encoding == expected.time.encoding
        assert result.time_bnds.encoding == expected.time_bnds.encoding

    def test_decodes_time_coords_and_bounds_in_years_with_a_reference_date_in_the_mid_year(
        self,
    ):
        ds = self.ds.copy()

        calendar = "standard"
        ds.time.attrs["calendar"] = calendar
        ds.time.attrs["units"] = "years since 2000-06-01"

        result = decode_time(ds)

        expected = xr.Dataset(
            {
                "time": xr.DataArray(
                    name="time",
                    data=np.array(
                        [
                            cftime.datetime(2001, 6, 1, calendar=calendar),
                            cftime.datetime(2002, 6, 1, calendar=calendar),
                            cftime.datetime(2003, 6, 1, calendar=calendar),
                        ],
                    ),
                    dims=["time"],
                    attrs={
                        "bounds": "time_bnds",
                        "axis": "T",
                        "long_name": "time",
                        "standard_name": "time",
                    },
                ),
                "time_bnds": xr.DataArray(
                    name="time_bnds",
                    data=np.array(
                        [
                            [
                                cftime.datetime(2000, 6, 1, calendar=calendar),
                                cftime.datetime(2001, 6, 1, calendar=calendar),
                            ],
                            [
                                cftime.datetime(2001, 6, 1, calendar=calendar),
                                cftime.datetime(2002, 6, 1, calendar=calendar),
                            ],
                            [
                                cftime.datetime(2002, 6, 1, calendar=calendar),
                                cftime.datetime(2003, 6, 1, calendar=calendar),
                            ],
                        ],
                    ),
                    dims=["time", "bnds"],
                    attrs=ds.time_bnds.attrs,
                ),
            }
        )
        assert result.identical(expected)

        # Check the encoding is preserved.
        expected.time.encoding = {
            "units": "years since 2000-06-01",
            "calendar": "standard",
        }
        expected.time_bnds.encoding = {
            "units": "years since 2000-06-01",
            "calendar": "standard",
        }

        assert result.time.encoding == expected.time.encoding
        assert result.time_bnds.encoding == expected.time_bnds.encoding

    def test_decodes_time_coords_and_bounds_in_years_with_a_reference_date_on_a_leap_year(
        self,
    ):
        ds = self.ds.copy()

        calendar = "standard"
        ds.time.attrs["calendar"] = calendar
        ds.time.attrs["units"] = "years since 2000-02-29"

        result = decode_time(ds)

        expected = xr.Dataset(
            {
                "time": xr.DataArray(
                    name="time",
                    data=np.array(
                        [
                            cftime.datetime(2001, 2, 28, calendar=calendar),
                            cftime.datetime(2002, 2, 28, calendar=calendar),
                            cftime.datetime(2003, 2, 28, calendar=calendar),
                        ],
                    ),
                    dims=["time"],
                    attrs={
                        "bounds": "time_bnds",
                        "axis": "T",
                        "long_name": "time",
                        "standard_name": "time",
                    },
                ),
                "time_bnds": xr.DataArray(
                    name="time_bnds",
                    data=np.array(
                        [
                            [
                                cftime.datetime(2000, 2, 29, calendar=calendar),
                                cftime.datetime(2001, 2, 28, calendar=calendar),
                            ],
                            [
                                cftime.datetime(2001, 2, 28, calendar=calendar),
                                cftime.datetime(2002, 2, 28, calendar=calendar),
                            ],
                            [
                                cftime.datetime(2002, 2, 28, calendar=calendar),
                                cftime.datetime(2003, 2, 28, calendar=calendar),
                            ],
                        ],
                    ),
                    dims=["time", "bnds"],
                    attrs=ds.time_bnds.attrs,
                ),
            }
        )
        assert result.identical(expected)

        # Check the encoding is preserved.
        expected.time.encoding = {
            "units": "years since 2000-02-29",
            "calendar": "standard",
        }
        expected.time_bnds.encoding = {
            "units": "years since 2000-02-29",
            "calendar": "standard",
        }

        assert result.time.encoding == expected.time.encoding
        assert result.time_bnds.encoding == expected.time_bnds.encoding


class Test_PostProcessDataset:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = generate_dataset(
            decode_times=True, cf_compliant=False, has_bounds=True
        )

    def test_centers_time_coords_and_converts_datetime_dtype_to_cftime_object_type(
        self,
    ):
        ds = generate_dataset(decode_times=True, cf_compliant=False, has_bounds=True)

        # Create a dataset with uncentered time coordinates that are decoded as
        # dtype="datetime[ns]"
        ds_uncentered = ds.copy()
        ds_uncentered["time"] = xr.DataArray(
            data=np.array(
                [
                    "2000-01-31T12:00:00.000000000",
                    "2000-02-29T12:00:00.000000000",
                    "2000-03-31T12:00:00.000000000",
                    "2000-04-30T00:00:00.000000000",
                    "2000-05-31T12:00:00.000000000",
                    "2000-06-30T00:00:00.000000000",
                    "2000-07-31T12:00:00.000000000",
                    "2000-08-31T12:00:00.000000000",
                    "2000-09-30T00:00:00.000000000",
                    "2000-10-16T12:00:00.000000000",
                    "2000-11-30T00:00:00.000000000",
                    "2000-12-31T12:00:00.000000000",
                    "2001-01-31T12:00:00.000000000",
                    "2001-02-28T00:00:00.000000000",
                    "2001-12-31T12:00:00.000000000",
                ],
                dtype="datetime64[ns]",
            ),
            dims=ds.time.dims,
            attrs=ds.time.attrs,
        )
        ds_uncentered.time.encoding = {
            "source": None,
            "original_shape": ds.time.data.shape,
            "dtype": np.dtype("float64"),
            "units": "days since 2000-01-01",
            "calendar": "standard",
            "_FillValue": False,
        }

        # Compare result of the method against the expected.
        result = _postprocess_dataset(ds_uncentered, center_times=True)
        expected = ds.copy()
        expected["time"] = xr.DataArray(
            name="time",
            data=np.array(
                [
                    cftime.DatetimeGregorian(
                        2000, 1, 16, 12, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2000, 2, 15, 12, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2000, 3, 16, 12, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2000, 4, 16, 0, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2000, 5, 16, 12, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2000, 6, 16, 0, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2000, 7, 16, 12, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2000, 8, 16, 12, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2000, 9, 16, 0, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2000, 10, 16, 12, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2000, 11, 16, 0, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2000, 12, 16, 12, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2001, 1, 16, 12, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2001, 2, 15, 0, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2001, 12, 16, 12, 0, 0, 0, has_year_zero=False
                    ),
                ],
                dtype="object",
            ),
            dims="time",
            attrs={
                "long_name": "time",
                "standard_name": "time",
                "axis": "T",
                "bounds": "time_bnds",
            },
        )

        expected.time.encoding = {
            "source": None,
            "original_shape": (15,),
            "dtype": np.dtype("float64"),
            "units": "days since 2000-01-01",
            "calendar": "standard",
            "_FillValue": False,
        }

        # Compare result of the function against the expected.
        assert result.identical(expected)
        assert result.time.encoding == expected.time.encoding

    def test_centers_time_coordinates_and_maintains_cftime_object_type(self):
        # Create a dataset with uncentered time coordinates
        ds = generate_dataset(decode_times=True, cf_compliant=False, has_bounds=True)
        uncentered_time = np.array(
            [
                cftime.DatetimeGregorian(2000, 1, 31, 12, 0, 0, 0),
                cftime.DatetimeGregorian(2000, 2, 29, 12, 0, 0, 0),
                cftime.DatetimeGregorian(2000, 3, 31, 12, 0, 0, 0),
                cftime.DatetimeGregorian(2000, 4, 30, 0, 0, 0, 0),
                cftime.DatetimeGregorian(2000, 5, 31, 12, 0, 0, 0),
                cftime.DatetimeGregorian(2000, 6, 30, 0, 0, 0, 0),
                cftime.DatetimeGregorian(2000, 7, 31, 12, 0, 0, 0),
                cftime.DatetimeGregorian(2000, 8, 31, 12, 0, 0, 0),
                cftime.DatetimeGregorian(2000, 9, 30, 0, 0, 0, 0),
                cftime.DatetimeGregorian(2000, 10, 16, 12, 0, 0, 0),
                cftime.DatetimeGregorian(2000, 11, 30, 0, 0, 0, 0),
                cftime.DatetimeGregorian(2000, 12, 31, 12, 0, 0, 0),
                cftime.DatetimeGregorian(2001, 1, 31, 12, 0, 0, 0),
                cftime.DatetimeGregorian(2001, 2, 28, 0, 0, 0, 0),
                cftime.DatetimeGregorian(2001, 12, 31, 12, 0, 0, 0),
            ],
            dtype="object",
        )
        ds.time.data[:] = uncentered_time
        ds.time.encoding = {
            "source": None,
            "original_shape": ds.time.data.shape,
            "dtype": np.dtype("float64"),
            "units": "days since 2000-01-01",
            "calendar": "standard",
            "_FillValue": False,
        }

        # Compare result of the method against the expected.
        result = _postprocess_dataset(ds, center_times=True)
        expected = ds.copy()
        expected["time"] = xr.DataArray(
            name="time",
            data=np.array(
                [
                    cftime.DatetimeGregorian(
                        2000, 1, 16, 12, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2000, 2, 15, 12, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2000, 3, 16, 12, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2000, 4, 16, 0, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2000, 5, 16, 12, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2000, 6, 16, 0, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2000, 7, 16, 12, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2000, 8, 16, 12, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2000, 9, 16, 0, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2000, 10, 16, 12, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2000, 11, 16, 0, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2000, 12, 16, 12, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2001, 1, 16, 12, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2001, 2, 15, 0, 0, 0, 0, has_year_zero=False
                    ),
                    cftime.DatetimeGregorian(
                        2001, 12, 16, 12, 0, 0, 0, has_year_zero=False
                    ),
                ],
                dtype="object",
            ),
            dims="time",
            attrs={
                "long_name": "time",
                "standard_name": "time",
                "axis": "T",
                "bounds": "time_bnds",
            },
        )

        expected.time.encoding = {
            "source": None,
            "original_shape": (15,),
            "dtype": np.dtype("float64"),
            "units": "days since 2000-01-01",
            "calendar": "standard",
            "_FillValue": False,
        }

        # Update time bounds with centered time coordinates.
        expected["time_bnds"] = ds.time_bnds.copy()
        expected["time_bnds"]["time"] = expected.time

        # Compare result of the function against the expected.
        assert result.identical(expected)
        assert result.time.encoding == expected.time.encoding

    def test_raises_error_if_dataset_has_no_time_coords_but_center_times_is_true(self):
        ds = generate_dataset(decode_times=True, cf_compliant=False, has_bounds=False)
        ds = ds.drop_dims("time")

        with pytest.raises(KeyError):
            _postprocess_dataset(ds, center_times=True)

    def test_adds_missing_lat_and_lon_bounds(self):
        # Create expected dataset without bounds.
        ds = generate_dataset(decode_times=True, cf_compliant=False, has_bounds=False)

        data_vars = list(ds.data_vars.keys())
        assert "lat_bnds" not in data_vars
        assert "lon_bnds" not in data_vars

        result = _postprocess_dataset(ds, add_bounds=True)
        result_data_vars = list(result.data_vars.keys())
        assert "lat_bnds" in result_data_vars
        assert "lon_bnds" in result_data_vars

    def test_orients_longitude_bounds_from_180_to_360_and_sorts_with_prime_meridian_cell(
        self,
    ):
        # Chunk the dataset to test method also works with Dask.
        ds = xr.Dataset(
            coords={
                "lon": xr.DataArray(
                    name="lon",
                    data=np.array([-180, -1, 0, 1, 179]),
                    dims=["lon"],
                    attrs={"units": "degrees_east", "axis": "X", "bounds": "lon_bnds"},
                )
            },
            data_vars={
                "lon_bnds": xr.DataArray(
                    name="lon_bnds",
                    data=np.array(
                        [
                            [-180.5, -1.5],
                            [-1.5, -0.5],
                            [-0.5, 0.5],
                            [0.5, 1.5],
                            [1.5, 179.5],
                        ]
                    ),
                    dims=["lon", "bnds"],
                    attrs={"xcdat_bounds": "True"},
                ),
            },
        ).chunk({"lon": 2})

        result = _postprocess_dataset(ds, lon_orient=(0, 360))
        expected = xr.Dataset(
            coords={
                "lon": xr.DataArray(
                    name="lon",
                    data=np.array([0.0, 1.0, 179.0, 180.0, 359.0, 360.0]),
                    dims=["lon"],
                    attrs={"units": "degrees_east", "axis": "X", "bounds": "lon_bnds"},
                )
            },
            data_vars={
                "lon_bnds": xr.DataArray(
                    name="lon_bnds",
                    data=np.array(
                        [
                            [0, 0.5],
                            [0.5, 1.5],
                            [1.5, 179.5],
                            [179.5, 358.5],
                            [358.5, 359.5],
                            [359.5, 360],
                        ]
                    ),
                    dims=["lon", "bnds"],
                    attrs={"xcdat_bounds": "True"},
                ),
            },
        )
        assert result.identical(expected)

    def test_raises_error_if_dataset_has_no_longitude_coords_but_lon_orient_is_specified(
        self,
    ):
        ds = generate_dataset(decode_times=True, cf_compliant=False, has_bounds=False)
        ds = ds.drop_dims("lon")

        with pytest.raises(KeyError):
            _postprocess_dataset(ds, lon_orient=(0, 360))


class Test_KeepSingleVar:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = generate_dataset(
            decode_times=True, cf_compliant=False, has_bounds=True
        )

        self.ds_mod = self.ds.copy()
        self.ds_mod["tas"] = self.ds_mod.ts.copy()

    def tests_raises_error_if_only_bounds_data_variables_exist(self):
        ds = self.ds.copy()
        ds = ds.drop_vars("ts")

        with pytest.raises(ValueError):
            _keep_single_var(ds, key="ts")

    def test_raises_error_if_specified_data_var_does_not_exist(self):
        ds = self.ds_mod.copy()

        with pytest.raises(ValueError):
            _keep_single_var(ds, key="nonexistent")

    def test_raises_error_if_specified_data_var_is_a_bounds_var(self):
        ds = self.ds_mod.copy()

        with pytest.raises(ValueError):
            _keep_single_var(ds, key="lat_bnds")

    def test_returns_dataset_with_specified_data_var(self):
        result = _keep_single_var(self.ds_mod, key="ts")
        expected = self.ds.copy()

        assert result.identical(expected)
        assert not result.identical(self.ds_mod)

    def test_bounds_always_persist(self):
        ds = _keep_single_var(self.ds_mod, key="ts")
        assert ds.get("lat_bnds") is not None
        assert ds.get("lon_bnds") is not None
        assert ds.get("time_bnds") is not None
