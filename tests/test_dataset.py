import logging
import pathlib
import warnings

import numpy as np
import pytest
import xarray as xr

from tests.fixtures import generate_dataset
from xcdat.dataset import (
    _keep_single_var,
    _preprocess_non_cf_dataset,
    _split_time_units_attr,
    decode_non_cf_time,
    has_cf_compliant_time,
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
        assert result.identical(expected)

    def test_non_cf_compliant_time_is_not_decoded(self):
        ds = generate_dataset(cf_compliant=False, has_bounds=True)
        ds.to_netcdf(self.file_path)

        result = open_dataset(self.file_path, decode_times=False)
        expected = generate_dataset(cf_compliant=False, has_bounds=True)
        assert result.identical(expected)

    def test_non_cf_compliant_time_is_decoded(self):
        ds = generate_dataset(cf_compliant=False, has_bounds=False)
        ds.to_netcdf(self.file_path)

        result = open_dataset(self.file_path, data_var="ts")

        # Generate an expected dataset with decoded non-CF compliant time units.
        expected = generate_dataset(cf_compliant=True, has_bounds=True)
        expected_time_data = np.array(
            [
                "2000-01-01T00:00:00.000000000",
                "2000-02-01T00:00:00.000000000",
                "2000-03-01T00:00:00.000000000",
                "2000-04-01T00:00:00.000000000",
                "2000-05-01T00:00:00.000000000",
                "2000-06-01T00:00:00.000000000",
                "2000-07-01T00:00:00.000000000",
                "2000-08-01T00:00:00.000000000",
                "2000-09-01T00:00:00.000000000",
                "2000-10-01T00:00:00.000000000",
                "2000-11-01T00:00:00.000000000",
                "2000-12-01T00:00:00.000000000",
                "2001-01-01T00:00:00.000000000",
                "2001-02-01T00:00:00.000000000",
                "2001-03-01T00:00:00.000000000",
            ],
            dtype="datetime64[ns]",
        )
        expected["time"] = xr.DataArray(
            name="time",
            data=expected_time_data,
            dims="time",
            attrs={
                "units": "months since 2000-01-01",
                "calendar": "standard",
                "axis": "T",
                "long_name": "time",
                "standard_name": "time",
                "bounds": "time_bnds",
            },
        )
        expected.time_bnds.data[:] = np.array(
            [
                ["1999-12-16T12:00:00.000000000", "2000-01-16T12:00:00.000000000"],
                ["2000-01-16T12:00:00.000000000", "2000-02-15T12:00:00.000000000"],
                ["2000-02-15T12:00:00.000000000", "2000-03-16T12:00:00.000000000"],
                ["2000-03-16T12:00:00.000000000", "2000-04-16T00:00:00.000000000"],
                ["2000-04-16T00:00:00.000000000", "2000-05-16T12:00:00.000000000"],
                ["2000-05-16T12:00:00.000000000", "2000-06-16T00:00:00.000000000"],
                ["2000-06-16T00:00:00.000000000", "2000-07-16T12:00:00.000000000"],
                ["2000-07-16T12:00:00.000000000", "2000-08-16T12:00:00.000000000"],
                ["2000-08-16T12:00:00.000000000", "2000-09-16T00:00:00.000000000"],
                ["2000-09-16T00:00:00.000000000", "2000-10-16T12:00:00.000000000"],
                ["2000-10-16T12:00:00.000000000", "2000-11-16T00:00:00.000000000"],
                ["2000-11-16T00:00:00.000000000", "2000-12-16T12:00:00.000000000"],
                ["2000-12-16T12:00:00.000000000", "2001-01-16T12:00:00.000000000"],
                ["2001-01-16T12:00:00.000000000", "2001-02-15T00:00:00.000000000"],
                ["2001-02-15T00:00:00.000000000", "2001-03-15T00:00:00.000000000"],
            ],
            dtype="datetime64[ns]",
        )
        expected.time.encoding = {
            # Set source as result source because it changes every test run.
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

    def test_swaps_from_180_to_360_and_sorts_with_prime_meridian_cell(self):
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
                    attrs={"is_generated": "True"},
                ),
            },
        )
        ds.to_netcdf(self.file_path)

        result = open_dataset(self.file_path, lon_orient=(0, 360))
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
                    attrs={"is_generated": "True"},
                ),
            },
        )
        assert result.identical(expected)

    def test_centers_time(self):
        ds = generate_dataset(cf_compliant=True, has_bounds=True)

        uncentered_time = np.array(
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
        )
        ds.time.data[:] = uncentered_time
        ds.time.encoding = {
            "source": None,
            "dtype": np.dtype(np.int64),
            "original_shape": ds.time.data.shape,
            "units": "days since 2000-01-01",
            "calendar": "standard",
            "_FillValue": False,
        }
        ds.to_netcdf(self.file_path)

        # Compare result of the method against the expected.
        result = open_dataset(self.file_path, data_var="ts", center_times=True)
        expected = ds.copy()
        expected_time_data = np.array(
            [
                "2000-01-16T12:00:00.000000000",
                "2000-02-15T12:00:00.000000000",
                "2000-03-16T12:00:00.000000000",
                "2000-04-16T00:00:00.000000000",
                "2000-05-16T12:00:00.000000000",
                "2000-06-16T00:00:00.000000000",
                "2000-07-16T12:00:00.000000000",
                "2000-08-16T12:00:00.000000000",
                "2000-09-16T00:00:00.000000000",
                "2000-10-16T12:00:00.000000000",
                "2000-11-16T00:00:00.000000000",
                "2000-12-16T12:00:00.000000000",
                "2001-01-16T12:00:00.000000000",
                "2001-02-15T00:00:00.000000000",
                "2001-12-16T12:00:00.000000000",
            ],
            dtype="datetime64[ns]",
        )
        expected = expected.assign_coords(
            {
                "time": xr.DataArray(
                    name="time",
                    data=expected_time_data,
                    coords={"time": expected_time_data},
                    dims="time",
                    attrs={
                        "long_name": "time",
                        "standard_name": "time",
                        "axis": "T",
                        "bounds": "time_bnds",
                    },
                )
            }
        )
        expected.time.encoding = {
            "zlib": False,
            "shuffle": False,
            "complevel": 0,
            "fletcher32": False,
            "contiguous": True,
            "chunksizes": None,
            "original_shape": (15,),
            "dtype": np.dtype("int64"),
            "_FillValue": 0,
            "units": "days since 2000-01-01",
            "calendar": "standard",
        }

        # Update time bounds with centered time coordinates.
        time_bounds = ds.time_bnds.copy()
        time_bounds["time"] = expected.time
        expected["time_bnds"] = time_bounds

        # Compare result of the function against the expected.
        assert result.identical(expected)

        # Delete source key because the path of the file can change for each
        # time run.
        del result.time.encoding["source"]
        assert result.time.encoding == expected.time.encoding


class TestOpenMfDataset:
    @pytest.fixture(autouse=True)
    def setUp(self, tmp_path):
        # Create temporary directory to save files.
        dir = tmp_path / "input_data"
        dir.mkdir()
        self.file_path1 = f"{dir}/file1.nc"
        self.file_path2 = f"{dir}/file2.nc"

    def test_only_keeps_specified_var(self):
        ds1 = generate_dataset(cf_compliant=True, has_bounds=True)
        ds2 = generate_dataset(cf_compliant=True, has_bounds=True)
        ds2 = ds2.rename_vars({"ts": "tas"})

        # Suppress UserWarning regarding missing time.encoding "units" because
        # it is not relevant to this test.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ds1.to_netcdf(self.file_path1)
            ds2.to_netcdf(self.file_path2)

        result = open_mfdataset([self.file_path1, self.file_path2], data_var="ts")

        # Generate an expected dataset with decoded non-CF compliant time units.
        expected = generate_dataset(cf_compliant=True, has_bounds=True)
        assert result.identical(expected)

    def test_non_cf_compliant_time_is_not_decoded(self):
        ds1 = generate_dataset(cf_compliant=False, has_bounds=True)
        ds1.to_netcdf(self.file_path1)
        ds2 = generate_dataset(cf_compliant=False, has_bounds=True)
        ds2 = ds2.rename_vars({"ts": "tas"})
        ds2.to_netcdf(self.file_path2)

        result = open_mfdataset([self.file_path1, self.file_path2], decode_times=False)

        expected = ds1.merge(ds2)
        assert result.identical(expected)

    def test_non_cf_compliant_time_is_decoded(self):
        ds1 = generate_dataset(cf_compliant=False, has_bounds=False)
        ds2 = generate_dataset(cf_compliant=False, has_bounds=False)
        ds2 = ds2.rename_vars({"ts": "tas"})

        ds1.to_netcdf(self.file_path1)
        ds2.to_netcdf(self.file_path2)

        result = open_mfdataset([self.file_path1, self.file_path2], data_var="ts")

        # Generate an expected dataset, which is a combination of both datasets
        # with decoded time units and coordinate bounds.
        expected = generate_dataset(cf_compliant=True, has_bounds=True)
        expected_time_data = np.array(
            [
                "2000-01-01T00:00:00.000000000",
                "2000-02-01T00:00:00.000000000",
                "2000-03-01T00:00:00.000000000",
                "2000-04-01T00:00:00.000000000",
                "2000-05-01T00:00:00.000000000",
                "2000-06-01T00:00:00.000000000",
                "2000-07-01T00:00:00.000000000",
                "2000-08-01T00:00:00.000000000",
                "2000-09-01T00:00:00.000000000",
                "2000-10-01T00:00:00.000000000",
                "2000-11-01T00:00:00.000000000",
                "2000-12-01T00:00:00.000000000",
                "2001-01-01T00:00:00.000000000",
                "2001-02-01T00:00:00.000000000",
                "2001-03-01T00:00:00.000000000",
            ],
            dtype="datetime64[ns]",
        )
        expected["time"] = xr.DataArray(
            name="time",
            data=expected_time_data,
            dims="time",
            attrs={
                "units": "months since 2000-01-01",
                "calendar": "standard",
                "axis": "T",
                "long_name": "time",
                "standard_name": "time",
                "bounds": "time_bnds",
            },
        )
        expected.time_bnds.data[:] = np.array(
            [
                ["1999-12-16T12:00:00.000000000", "2000-01-16T12:00:00.000000000"],
                ["2000-01-16T12:00:00.000000000", "2000-02-15T12:00:00.000000000"],
                ["2000-02-15T12:00:00.000000000", "2000-03-16T12:00:00.000000000"],
                ["2000-03-16T12:00:00.000000000", "2000-04-16T00:00:00.000000000"],
                ["2000-04-16T00:00:00.000000000", "2000-05-16T12:00:00.000000000"],
                ["2000-05-16T12:00:00.000000000", "2000-06-16T00:00:00.000000000"],
                ["2000-06-16T00:00:00.000000000", "2000-07-16T12:00:00.000000000"],
                ["2000-07-16T12:00:00.000000000", "2000-08-16T12:00:00.000000000"],
                ["2000-08-16T12:00:00.000000000", "2000-09-16T00:00:00.000000000"],
                ["2000-09-16T00:00:00.000000000", "2000-10-16T12:00:00.000000000"],
                ["2000-10-16T12:00:00.000000000", "2000-11-16T00:00:00.000000000"],
                ["2000-11-16T00:00:00.000000000", "2000-12-16T12:00:00.000000000"],
                ["2000-12-16T12:00:00.000000000", "2001-01-16T12:00:00.000000000"],
                ["2001-01-16T12:00:00.000000000", "2001-02-15T00:00:00.000000000"],
                ["2001-02-15T00:00:00.000000000", "2001-03-15T00:00:00.000000000"],
            ],
            dtype="datetime64[ns]",
        )
        expected.time.encoding = {
            # Set source as result source because it changes every test run.
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
        result = open_mfdataset([self.file_path1, self.file_path2], data_var="ts")
        assert result.identical(expected)

    def test_generates_lat_and_lon_bounds_if_they_dont_exist(self):
        ds1 = generate_dataset(cf_compliant=True, has_bounds=False)
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

    def test_swaps_from_180_to_360_and_sorts_with_prime_meridian_cell(self):
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
                    attrs={"is_generated": "True"},
                ),
            },
        )
        ds.to_netcdf(self.file_path1)

        result = open_mfdataset([self.file_path1], lon_orient=(0, 360))
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
                    attrs={"is_generated": "True"},
                ),
            },
        )
        assert result.identical(expected)

    def test_centers_time(self):
        ds1 = generate_dataset(cf_compliant=True, has_bounds=True)

        # Make the time coordinates uncentered.
        uncentered_time = np.array(
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
        )
        ds1.time.data[:] = uncentered_time
        ds1.time.encoding = {
            "source": None,
            "dtype": np.dtype(np.int64),
            "original_shape": ds1.time.data.shape,
            "units": "days since 2000-01-01",
            "calendar": "standard",
            "_FillValue": False,
        }
        ds2 = ds1.copy()
        ds2 = ds2.rename_vars({"ts": "tas"})

        ds1.to_netcdf(self.file_path1)
        ds2.to_netcdf(self.file_path2)

        # Compare result of the method against the expected.
        result = open_mfdataset([self.file_path1, self.file_path2], center_times=True)

        expected = ds1.merge(ds2)
        expected = expected.copy()
        expected_time_data = np.array(
            [
                "2000-01-16T12:00:00.000000000",
                "2000-02-15T12:00:00.000000000",
                "2000-03-16T12:00:00.000000000",
                "2000-04-16T00:00:00.000000000",
                "2000-05-16T12:00:00.000000000",
                "2000-06-16T00:00:00.000000000",
                "2000-07-16T12:00:00.000000000",
                "2000-08-16T12:00:00.000000000",
                "2000-09-16T00:00:00.000000000",
                "2000-10-16T12:00:00.000000000",
                "2000-11-16T00:00:00.000000000",
                "2000-12-16T12:00:00.000000000",
                "2001-01-16T12:00:00.000000000",
                "2001-02-15T00:00:00.000000000",
                "2001-12-16T12:00:00.000000000",
            ],
            dtype="datetime64[ns]",
        )
        expected = expected.assign_coords(
            {
                "time": xr.DataArray(
                    name="time",
                    data=expected_time_data,
                    coords={"time": expected_time_data},
                    dims="time",
                    attrs={
                        "long_name": "time",
                        "standard_name": "time",
                        "axis": "T",
                        "bounds": "time_bnds",
                    },
                )
            }
        )
        expected.time.encoding = {
            "zlib": False,
            "shuffle": False,
            "complevel": 0,
            "fletcher32": False,
            "contiguous": True,
            "chunksizes": None,
            "original_shape": (15,),
            "dtype": np.dtype("int64"),
            "_FillValue": 0,
            "units": "days since 2000-01-01",
            "calendar": "standard",
        }
        # Update time bounds with centered time coordinates.
        time_bounds = expected.time_bnds.copy()
        time_bounds["time"] = expected.time
        expected["time_bnds"] = time_bounds

        # Compare result of the function against the expected.
        assert result.identical(expected)

        # Delete source key because the path of the file can change for each
        # time run.
        del result.time.encoding["source"]
        assert result.time.encoding == expected.time.encoding


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
                        ["2000-02-01", "2000-03-01", "2000-04-01"],
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
                        ["2000-02-15", "2000-03-15", "2000-04-15"],
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
                        ["2000-01-31", "2000-02-29", "2000-03-31"],
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
                        ["2000-03-29", "2000-04-29", "2000-05-29"],
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
                        ["2001-06-01", "2002-06-01", "2003-06-01"],
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


class TestKeepSingleVar:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = generate_dataset(cf_compliant=True, has_bounds=True)

        self.ds_mod = self.ds.copy()
        self.ds_mod["tas"] = self.ds_mod.ts.copy()

    def tests_raises_logger_debug_if_only_bounds_data_variables_exist(self, caplog):
        caplog.set_level(logging.DEBUG)

        ds = self.ds.copy()
        ds = ds.drop_vars("ts")

        _keep_single_var(ds, data_var=None)
        assert "This dataset only contains bounds data variables." in caplog.text

    def test_raises_error_if_specified_data_var_does_not_exist(self):
        ds = self.ds_mod.copy()
        with pytest.raises(KeyError):
            _keep_single_var(ds, data_var="nonexistent")

    def test_raises_error_if_specified_data_var_is_a_bounds_var(self):
        ds = self.ds_mod.copy()
        with pytest.raises(KeyError):
            _keep_single_var(ds, data_var="lat_bnds")

    def test_returns_dataset_if_it_only_has_one_non_bounds_data_var(self):
        ds = self.ds.copy()

        result = _keep_single_var(ds, data_var=None)
        expected = ds.copy()

        assert result.identical(expected)

    def test_returns_dataset_if_it_contains_multiple_non_bounds_data_var_with_logger_msg(
        self, caplog
    ):
        caplog.set_level(logging.DEBUG)

        ds = self.ds_mod.copy()
        result = _keep_single_var(ds, data_var=None)
        expected = ds.copy()

        assert result.identical(expected)
        assert (
            "This dataset contains more than one regular data variable: ['tas', 'ts']. "
            "If desired, pass the `data_var` kwarg to limit the dataset to a single data var."
        ) in caplog.text

    def test_returns_dataset_with_specified_data_var(self):
        result = _keep_single_var(self.ds_mod, data_var="ts")
        expected = self.ds.copy()

        assert result.identical(expected)
        assert not result.identical(self.ds_mod)

    def test_bounds_always_persist(self):
        ds = _keep_single_var(self.ds_mod, data_var="ts")
        assert ds.get("lat_bnds") is not None
        assert ds.get("lon_bnds") is not None
        assert ds.get("time_bnds") is not None


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
                ["2000-01-01"],
                dtype="datetime64",
            ),
            dims=["time"],
        )
        expected["time_bnds"] = xr.DataArray(
            name="time_bnds",
            data=np.array(
                [["1999-12-01", "2000-01-01"]],
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
