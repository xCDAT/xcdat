import logging

import cftime
import numpy as np
import pytest
import xarray as xr

from tests.fixtures import (
    generate_dataset,
    generate_dataset_by_frequency,
    generate_lev_dataset,
    lat_bnds,
    lon_bnds,
)
from xcdat.bounds import BoundsAccessor

logger = logging.getLogger(__name__)


class TestBoundsAccessor:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = generate_dataset(
            decode_times=True, cf_compliant=False, has_bounds=False
        )
        self.ds_with_bnds = generate_dataset(
            decode_times=True, cf_compliant=False, has_bounds=True
        )

    def test__init__(self):
        obj = BoundsAccessor(self.ds)
        assert obj._dataset.identical(self.ds)

    def test_decorator_call(self):
        assert self.ds.bounds._dataset.identical(self.ds)

    def test_map_property_returns_map_of_axis_and_coordinate_keys_to_bounds_dataarray(
        self,
    ):
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

        result = ds.bounds.map

        for key in expected.keys():
            assert result[key].identical(expected[key])

    def test_keys_property_returns_a_list_of_sorted_bounds_keys(self):
        result = self.ds_with_bnds.bounds.keys
        expected = ["lat_bnds", "lon_bnds", "time_bnds"]

        assert result == expected


class TestAddMissingBounds:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = generate_dataset(
            decode_times=True, cf_compliant=False, has_bounds=False
        )
        self.ds_with_bnds = generate_dataset(
            decode_times=True, cf_compliant=False, has_bounds=True
        )

    def test_adds_x_y_and_z_bounds_to_the_dataset_using_midpoints(self):
        ds = generate_lev_dataset()
        ds_no_bnds = ds.drop_vars(["lev_bnds", "lat_bnds", "lon_bnds"])

        result = ds_no_bnds.bounds.add_missing_bounds(axes=["X", "Y", "Z"])
        assert result.identical(ds)

    def test_adds_t_bounds_to_the_dataset_using_time_frequency(self):
        ds = generate_dataset_by_frequency(freq="month")
        ds_no_bnds = ds.drop_vars(["time_bnds"])

        result = ds_no_bnds.bounds.add_missing_bounds(axes=["T"])
        assert result.identical(ds)

    def test_skips_adding_bounds_for_coords_that_are_1_dim_singleton(self, caplog):
        # NOTE: Suppress logger warning to avoid polluting test suite.
        caplog.set_level(logging.CRITICAL)
        ds = xr.Dataset(
            coords={
                "lon": xr.DataArray(
                    data=np.array([0]),
                    dims=["lon"],
                    attrs={"units": "degrees_east", "axis": "X"},
                )
            }
        )

        result = ds.bounds.add_missing_bounds(axes=["X"])
        assert result.identical(ds)

    def test_skips_adding_bounds_for_coords_that_are_0_dim_singleton(self, caplog):
        # NOTE: Suppress logger warning to avoid polluting test suite.
        caplog.set_level(logging.CRITICAL)
        ds = xr.Dataset(
            coords={
                "lon": xr.DataArray(
                    data=float(0),
                    attrs={"units": "degrees_east", "axis": "X"},
                )
            }
        )

        result = ds.bounds.add_missing_bounds(axes=["X"])
        assert result.identical(ds)

    def test_skips_adding_time_bounds_for_coords_that_are_1_dim_singleton(self, caplog):
        # NOTE: Suppress logger warning to avoid polluting test suite.
        caplog.set_level(logging.CRITICAL)
        ds = xr.Dataset(
            coords={
                "time": xr.DataArray(
                    data=np.array(
                        ["2000-01-01T12:00:00.000000000"], dtype="datetime64[ns]"
                    ),
                    dims=["time"],
                    attrs={"calendar": "standard", "units": "days since 1850-01-01"},
                )
            }
        )
        result = ds.bounds.add_missing_bounds(axes=["T"])
        assert result.identical(ds)

    def test_skips_adding_time_bounds_for_coords_that_are_not_datetime_like_objects(
        self, caplog
    ):
        # NOTE: Suppress logger warning to avoid polluting test suite.
        caplog.set_level(logging.CRITICAL)
        ds = xr.Dataset(
            coords={
                "time": xr.DataArray(
                    data=np.array([0, 1, 2]),
                    dims=["time"],
                    attrs={"calendar": "standard", "units": "days since 1850-01-01"},
                )
            }
        )
        result = ds.bounds.add_missing_bounds(axes=["T"])
        assert result.identical(ds)


class TestGetBounds:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = generate_dataset(
            decode_times=True, cf_compliant=False, has_bounds=False
        )
        self.ds_with_bnds = generate_dataset(
            decode_times=True, cf_compliant=False, has_bounds=True
        )

    def test_raises_error_with_invalid_axis_key(self):
        with pytest.raises(ValueError):
            self.ds.bounds.get_bounds("incorrect_axis_argument")

    def test_raises_error_if_no_bounds_are_found_because_none_exist(self):
        ds = xr.Dataset(
            data_vars={"ts": xr.DataArray(data=np.ones(3), dims="lat")},
            coords={
                "lat": xr.DataArray(
                    data=np.ones(3),
                    dims="lat",
                    attrs={"axis": "Y", "bounds": "lat_bnds"},
                )
            },
        )

        # No "Y" axis bounds are found in the entire dataset.
        with pytest.raises(KeyError):
            ds.bounds.get_bounds("Y")

        # No "Y" axis bounds are found for the specified var_key.
        with pytest.raises(KeyError):
            ds.bounds.get_bounds("Y", var_key="ts")

    def test_raises_error_if_no_bounds_are_found_because_bounds_attr_not_set(self):
        ds = xr.Dataset(
            coords={
                "lat": xr.DataArray(
                    data=np.ones(3),
                    dims="lat",
                    attrs={"axis": "Y"},
                )
            },
            data_vars={
                "var": xr.DataArray(data=np.ones((3)), dims=["lat"]),
                "lat_bnds": xr.DataArray(data=np.ones((3, 3)), dims=["lat", "bnds"]),
            },
        )

        with pytest.raises(KeyError):
            ds.bounds.get_bounds("Y")

    def test_raises_error_if_no_bounds_are_found_with_bounds_attr_set_because_none_exist(
        self,
    ):
        ds = xr.Dataset(
            coords={
                "lat": xr.DataArray(
                    data=np.ones(3),
                    dims="lat",
                    attrs={"axis": "Y", "bounds": "lat_bnds"},
                )
            },
            data_vars={
                "var": xr.DataArray(data=np.ones((3)), dims=["lat"]),
            },
        )

        with pytest.raises(KeyError):
            ds.bounds.get_bounds("Y", var_key="var")

    def test_returns_single_coord_var_axis_bounds_as_datarray_object(self):
        ds = xr.Dataset(
            coords={
                "lat": xr.DataArray(
                    data=np.ones(3),
                    dims="lat",
                    attrs={"axis": "Y", "bounds": "lat_bnds"},
                )
            },
            data_vars={
                "lat_bnds": xr.DataArray(data=np.ones((3, 3)), dims=["lat", "bnds"]),
            },
        )

        result = ds.bounds.get_bounds("Y", var_key="lat")
        expected = ds.lat_bnds

        assert result.identical(expected)

    def test_returns_single_data_var_axis_bounds_as_datarray_object(self):
        ds = xr.Dataset(
            coords={
                "lat": xr.DataArray(
                    data=np.ones(3),
                    dims="lat",
                    attrs={"axis": "Y", "bounds": "lat_bnds"},
                )
            },
            data_vars={
                "var": xr.DataArray(data=np.ones((3)), dims=["lat"]),
                "lat_bnds": xr.DataArray(data=np.ones((3, 3)), dims=["lat", "bnds"]),
            },
        )

        result = ds.bounds.get_bounds("Y", var_key="var")
        expected = ds.lat_bnds

        assert result.identical(expected)

    def test_returns_single_dataset_axis_bounds_as_a_dataarray_object(self):
        ds = xr.Dataset(
            coords={
                "lat": xr.DataArray(
                    data=np.ones(3),
                    dims="lat",
                    attrs={"axis": "Y", "bounds": "lat_bnds"},
                )
            },
            data_vars={
                "lat_bnds": xr.DataArray(data=np.ones((3, 3)), dims=["lat", "bnds"])
            },
        )

        result = ds.bounds.get_bounds("Y")
        expected = ds.lat_bnds

        assert result.identical(expected)

    def test_returns_multiple_dataset_axis_bounds_as_a_dataset_object(self):
        ds = xr.Dataset(
            coords={
                "lat": xr.DataArray(
                    data=np.ones(3),
                    dims="lat",
                    attrs={
                        "axis": "Y",
                        "standard_name": "latitude",
                        "bounds": "lat_bnds",
                    },
                ),
                "lat2": xr.DataArray(
                    data=np.ones(3),
                    dims="lat2",
                    attrs={
                        "axis": "Y",
                        "standard_name": "latitude",
                        "bounds": "lat2_bnds",
                    },
                ),
            },
            data_vars={
                "var": xr.DataArray(data=np.ones(3), dims=["lat"]),
                "lat_bnds": xr.DataArray(data=np.ones((3, 3)), dims=["lat", "bnds"]),
                "lat2_bnds": xr.DataArray(data=np.ones((3, 3)), dims=["lat2", "bnds"]),
            },
        )

        result = ds.bounds.get_bounds("Y")
        expected = ds.drop_vars("var")

        assert result.identical(expected)


class TestAddBounds:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = generate_dataset(
            decode_times=True, cf_compliant=False, has_bounds=False
        )
        self.ds_with_bnds = generate_dataset(
            decode_times=True, cf_compliant=False, has_bounds=True
        )

    def test_raises_error_for_singleton_coords(self):
        # Length <=1
        lon = xr.DataArray(
            data=np.array([0]),
            dims=["lon"],
            attrs={"units": "degrees_east", "axis": "X"},
        )
        ds = xr.Dataset(coords={"lon": lon})

        with pytest.raises(ValueError):
            ds.bounds.add_bounds("X")

    def test_raises_error_if_lat_coord_var_units_is_not_in_degrees(self):
        lat = xr.DataArray(
            data=np.array([0, 0, 0]),
            dims=["lat"],
            attrs={"units": "invalid_units", "axis": "Y"},
        )

        ds = xr.Dataset(coords={"lat": lat})

        with pytest.raises(ValueError):
            ds.bounds.add_bounds("Y")

    def test_adds_bounds_and_sets_units_to_degrees_north_if_lat_coord_var_is_missing_units_attr(
        self, caplog
    ):
        # NOTE: Suppress logger warning to avoid polluting test suite.
        caplog.set_level(logging.CRITICAL)

        ds = self.ds.copy()
        del ds.lat.attrs["units"]

        result = ds.bounds.add_bounds("Y")
        assert result.lat_bnds.equals(lat_bnds)
        assert result.lat_bnds.xcdat_bounds == "True"
        assert result.lat.attrs["units"] == "degrees_north"
        assert result.lat.attrs["bounds"] == "lat_bnds"

    def test_skips_adding_bounds_for_coord_vars_with_bounds(self):
        ds = self.ds_with_bnds.copy()
        result = ds.bounds.add_bounds("Y")

        assert ds.identical(result)

    def test_adds_y_and_x_axis_bounds(self):
        ds = self.ds.copy()
        ds = ds.drop_dims("time")
        ds["time"] = xr.DataArray(
            name="time",
            data=np.array(
                [
                    "2000-01-01T12:00:00.000000000",
                    "2000-02-01T12:00:00.000000000",
                    "2000-03-01T12:00:00.000000000",
                ],
                dtype="datetime64[ns]",
            ),
            dims=["time"],
            attrs={
                "axis": "T",
                "long_name": "time",
                "standard_name": "time",
            },
        )

        result = ds.bounds.add_bounds("Y")
        assert result.lat_bnds.equals(lat_bnds)
        assert result.lat_bnds.xcdat_bounds == "True"
        assert result.lat.attrs["bounds"] == "lat_bnds"

        result = result.bounds.add_bounds("X")
        assert result.lon_bnds.equals(lon_bnds)
        assert result.lon_bnds.xcdat_bounds == "True"
        assert result.lon.attrs["bounds"] == "lon_bnds"

        result = result.bounds.add_bounds("T")
        expected_time_bnds = xr.DataArray(
            name="time_bnds",
            data=np.array(
                [
                    ["1999-12-17T00:00:00.000000000", "2000-01-17T00:00:00.000000000"],
                    ["2000-01-17T00:00:00.000000000", "2000-02-16T00:00:00.000000000"],
                    ["2000-02-16T00:00:00.000000000", "2000-03-16T00:00:00.000000000"],
                ],
                dtype="datetime64[ns]",
            ),
            coords={"time": ds.time.assign_attrs({"bounds": "time_bnds"})},
            dims=["time", "bnds"],
            attrs={"xcdat_bounds": "True"},
        )
        assert result.time_bnds.identical(expected_time_bnds)

    def test_ignores_adding_bounds_for_singleton_coordinate_unrelated_to_an_axis(
        self,
    ):
        ds = self.ds.copy()
        ds = ds.drop_dims("time")

        # Add time coordinates with the singleton coordinate, "height"
        height = xr.DataArray(
            name="height",
            data=np.array(2),
            attrs={
                "units": "m",
                "axis": "Z",
                "positive": "up",
                "long_name": "height",
                "standard_name": "height",
            },
        )
        ds["time"] = xr.DataArray(
            name="time",
            data=np.array(
                [
                    "2000-01-01T12:00:00.000000000",
                    "2000-02-01T12:00:00.000000000",
                    "2000-03-01T12:00:00.000000000",
                ],
                dtype="datetime64[ns]",
            ),
            dims=["time"],
            coords={"height": height},
            attrs={
                "axis": "T",
                "long_name": "time",
                "standard_name": "time",
            },
        )

        result = ds.bounds.add_bounds("T")
        expected_time_bnds = xr.DataArray(
            name="time_bnds",
            data=np.array(
                [
                    ["1999-12-17T00:00:00.000000000", "2000-01-17T00:00:00.000000000"],
                    ["2000-01-17T00:00:00.000000000", "2000-02-16T00:00:00.000000000"],
                    ["2000-02-16T00:00:00.000000000", "2000-03-16T00:00:00.000000000"],
                ],
                dtype="datetime64[ns]",
            ),
            coords={
                "time": ds.time.assign_attrs({"bounds": "time_bnds"}),
                "height": ds.time.height,
            },
            dims=["time", "bnds"],
            attrs={"xcdat_bounds": "True"},
        )

        assert result.time_bnds.identical(expected_time_bnds)


class TestAddTimeBounds:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = generate_dataset(
            decode_times=True, cf_compliant=False, has_bounds=False
        )
        self.ds_with_bnds = generate_dataset(
            decode_times=True, cf_compliant=False, has_bounds=True
        )
        # generate datasets of varying temporal frequencies
        self.ds_yearly_with_bnds = generate_dataset_by_frequency(freq="year")
        self.ds_daily_with_bnds = generate_dataset_by_frequency(freq="day")
        self.ds_hourly_with_bnds = generate_dataset_by_frequency(freq="hour")
        self.ds_subhourly_with_bnds = generate_dataset_by_frequency(freq="subhour")

    def test_raises_error_for_non_decoded_time_coords_with_freq_method(self):
        ds = self.ds.copy()
        ds = ds.drop_dims("time")
        ds["time"] = xr.DataArray(
            name="time",
            data=np.array([0, 1, 2], dtype="int64"),
            dims=["time"],
            attrs={
                "axis": "T",
                "long_name": "time",
                "standard_name": "time",
            },
        )

        with pytest.raises(TypeError):
            ds.bounds.add_time_bounds(method="freq")

    def test_skips_adding_bounds_for_coord_vars_with_bounds(self):
        ds = self.ds_with_bnds.copy()
        result = ds.bounds.add_time_bounds(method="freq")

        assert ds.identical(result)

    def test_ignores_adding_bounds_for_singleton_coordinate_unrelated_to_the_time_axis(
        self,
    ):
        ds = self.ds.copy()
        ds = ds.drop_dims("time")

        # Add time coordinates with the singleton coordinate, "height"
        height = xr.DataArray(
            name="height",
            data=np.array(2),
            attrs={
                "units": "m",
                "axis": "Z",
                "positive": "up",
                "long_name": "height",
                "standard_name": "height",
            },
        )
        ds["time"] = xr.DataArray(
            name="time",
            data=np.array(
                [
                    "2000-01-01T12:00:00.000000000",
                    "2000-02-01T12:00:00.000000000",
                    "2000-03-01T12:00:00.000000000",
                ],
                dtype="datetime64[ns]",
            ),
            dims=["time"],
            coords={"height": height},
            attrs={
                "axis": "T",
                "long_name": "time",
                "standard_name": "time",
            },
        )

        result = ds.bounds.add_time_bounds(method="freq")
        # NOTE: The algorithm for generating time bounds doesn't extend the
        # upper bound into the next month.
        expected_time_bnds = xr.DataArray(
            name="time_bnds",
            data=np.array(
                [
                    ["2000-01-01T00:00:00.000000000", "2000-02-01T00:00:00.000000000"],
                    ["2000-02-01T00:00:00.000000000", "2000-03-01T00:00:00.000000000"],
                    ["2000-03-01T00:00:00.000000000", "2000-04-01T00:00:00.000000000"],
                ],
                dtype="datetime64[ns]",
            ),
            coords={
                "time": ds.time.assign_attrs({"bounds": "time_bnds"}),
                "height": ds.time.height,
            },
            dims=["time", "bnds"],
            attrs={"xcdat_bounds": "True"},
        )

        assert result.time_bnds.identical(expected_time_bnds)

    def test_add_bounds_for_np_datetime_time_coords_using_freq(self):
        ds = self.ds.copy()
        ds = ds.drop_dims("time")
        ds["time"] = xr.DataArray(
            name="time",
            data=np.array(
                [
                    "2000-01-01T12:00:00.000000000",
                    "2000-02-01T12:00:00.000000000",
                    "2000-03-01T12:00:00.000000000",
                ],
                dtype="datetime64[ns]",
            ),
            dims=["time"],
            attrs={
                "axis": "T",
                "long_name": "time",
                "standard_name": "time",
            },
        )

        result = ds.bounds.add_time_bounds(method="freq")
        # NOTE: The algorithm for generating time bounds doesn't extend the
        # upper bound into the next month.
        expected_time_bnds = xr.DataArray(
            name="time_bnds",
            data=np.array(
                [
                    ["2000-01-01T00:00:00.000000000", "2000-02-01T00:00:00.000000000"],
                    ["2000-02-01T00:00:00.000000000", "2000-03-01T00:00:00.000000000"],
                    ["2000-03-01T00:00:00.000000000", "2000-04-01T00:00:00.000000000"],
                ],
                dtype="datetime64[ns]",
            ),
            coords={"time": ds.time.assign_attrs({"bounds": "time_bnds"})},
            dims=["time", "bnds"],
            attrs={"xcdat_bounds": "True"},
        )

        assert result.time_bnds.identical(expected_time_bnds)

    def test_add_bounds_for_np_datetime_time_coords_as_midpoints(self):
        ds = self.ds.copy()
        ds = ds.drop_dims("time")
        ds["time"] = xr.DataArray(
            name="time",
            data=np.array(
                [
                    "2000-01-01T12:00:00.000000000",
                    "2000-02-01T12:00:00.000000000",
                    "2000-03-01T12:00:00.000000000",
                ],
                dtype="datetime64[ns]",
            ),
            dims=["time"],
            attrs={
                "axis": "T",
                "long_name": "time",
                "standard_name": "time",
            },
        )
        result = ds.bounds.add_time_bounds(method="midpoint")
        # NOTE: The algorithm for generating time bounds doesn't extend the
        # upper bound into the next month.
        expected_time_bnds = xr.DataArray(
            name="time_bnds",
            data=np.array(
                [
                    ["1999-12-17T00:00:00.000000000", "2000-01-17T00:00:00.000000000"],
                    ["2000-01-17T00:00:00.000000000", "2000-02-16T00:00:00.000000000"],
                    ["2000-02-16T00:00:00.000000000", "2000-03-16T00:00:00.000000000"],
                ],
                dtype="datetime64[ns]",
            ),
            coords={"time": ds.time.assign_attrs({"bounds": "time_bnds"})},
            dims=["time", "bnds"],
            attrs={"xcdat_bounds": "True"},
        )

        assert result.time_bnds.identical(expected_time_bnds)

    def test_returns_bounds_for_cftime_time_coords_using_freq(self):
        ds = self.ds.copy()
        ds = ds.drop_dims("time")
        ds["time"] = xr.DataArray(
            name="time",
            data=np.array(
                [
                    cftime.DatetimeNoLeap(1850, 1, 1),
                    cftime.DatetimeNoLeap(1850, 2, 1),
                    cftime.DatetimeNoLeap(1850, 3, 1),
                ],
            ),
            dims=["time"],
            attrs={"axis": "T", "long_name": "time", "standard_name": "time"},
        )
        ds["time"].encoding["calendar"] = "noleap"

        result = ds.bounds.add_time_bounds(method="freq")
        expected_time_bnds = xr.DataArray(
            name="time_bnds",
            data=np.array(
                [
                    [
                        cftime.DatetimeNoLeap(1850, 1, 1),
                        cftime.DatetimeNoLeap(1850, 2, 1),
                    ],
                    [
                        cftime.DatetimeNoLeap(1850, 2, 1),
                        cftime.DatetimeNoLeap(1850, 3, 1),
                    ],
                    [
                        cftime.DatetimeNoLeap(1850, 3, 1),
                        cftime.DatetimeNoLeap(1850, 4, 1),
                    ],
                ],
            ),
            coords={"time": ds.time.assign_attrs({"bounds": "time_bnds"})},
            dims=["time", "bnds"],
            attrs={"xcdat_bounds": "True"},
        )

        assert result.time_bnds.identical(expected_time_bnds)

    def test_returns_bounds_for_cftime_time_coords_as_midpoints(self):
        ds = self.ds.copy()
        ds = ds.drop_dims("time")
        ds["time"] = xr.DataArray(
            name="time",
            data=np.array(
                [
                    cftime.DatetimeNoLeap(1850, 1, 1),
                    cftime.DatetimeNoLeap(1850, 2, 1),
                    cftime.DatetimeNoLeap(1850, 3, 1),
                ],
            ),
            dims=["time"],
            attrs={
                "axis": "T",
                "long_name": "time",
                "standard_name": "time",
            },
        )

        result = ds.bounds.add_bounds("T")
        expected_time_bnds = xr.DataArray(
            name="time_bnds",
            data=np.array(
                [
                    [
                        cftime.DatetimeNoLeap(1849, 12, 16, 12),
                        cftime.DatetimeNoLeap(1850, 1, 16, 12),
                    ],
                    [
                        cftime.DatetimeNoLeap(1850, 1, 16, 12),
                        cftime.DatetimeNoLeap(1850, 2, 15, 0),
                    ],
                    [
                        cftime.DatetimeNoLeap(1850, 2, 15, 0),
                        cftime.DatetimeNoLeap(1850, 3, 15, 0),
                    ],
                ],
            ),
            coords={"time": ds.time.assign_attrs({"bounds": "time_bnds"})},
            dims=["time", "bnds"],
            attrs={"xcdat_bounds": "True"},
        )

        assert result.time_bnds.identical(expected_time_bnds)

    def test_add_bounds_for_time_coords_using_inferred_freq(self):
        # get reference datasets
        ds_subhrly_with_bnds = self.ds_subhourly_with_bnds.copy()
        ds_hrly_with_bnds = self.ds_hourly_with_bnds.copy()
        ds_daily_with_bnds = self.ds_daily_with_bnds.copy()
        ds_monthly_with_bnds = self.ds_with_bnds.copy()
        ds_yearly_with_bnds = self.ds_yearly_with_bnds.copy()

        # drop bounds for testing
        ds_subhrly_wo_bnds = ds_subhrly_with_bnds.drop_vars("time_bnds")
        ds_hrly_wo_bnds = ds_hrly_with_bnds.drop_vars("time_bnds")
        ds_daily_wo_bnds = ds_daily_with_bnds.drop_vars("time_bnds")
        ds_monthly_wo_bnds = ds_monthly_with_bnds.drop_vars("time_bnds")
        ds_yearly_wo_bnds = ds_yearly_with_bnds.drop_vars("time_bnds")

        # test adding bounds
        hourly_bounds = ds_hrly_wo_bnds.bounds.add_time_bounds(method="freq", freq=None)
        daily_bounds = ds_daily_wo_bnds.bounds.add_time_bounds(method="freq", freq=None)
        monthly_bounds = ds_monthly_wo_bnds.bounds.add_time_bounds(
            method="freq", freq=None
        )
        yearly_bounds = ds_yearly_wo_bnds.bounds.add_time_bounds(
            method="freq", freq=None
        )

        # sub hourly data is not supported
        with pytest.raises(ValueError):
            ds_subhrly_wo_bnds.bounds.add_time_bounds(method="freq", freq=None)

        # ensure identical
        assert hourly_bounds.identical(ds_hrly_with_bnds)
        assert daily_bounds.identical(ds_daily_with_bnds)
        assert monthly_bounds.identical(ds_monthly_with_bnds)
        assert yearly_bounds.identical(ds_yearly_with_bnds)

    def test_add_bounds_for_time_coords_with_different_frequencies(self):
        # get reference datasets
        ds_subhrly_with_bnds = self.ds_subhourly_with_bnds.copy()
        ds_hrly_with_bnds = self.ds_hourly_with_bnds.copy()
        ds_daily_with_bnds = self.ds_daily_with_bnds.copy()
        ds_monthly_with_bnds = self.ds_with_bnds.copy()
        ds_yearly_with_bnds = self.ds_yearly_with_bnds.copy()

        # drop bounds for testing
        ds_subhrly_wo_bnds = ds_subhrly_with_bnds.drop_vars("time_bnds")
        ds_hrly_wo_bnds = ds_hrly_with_bnds.drop_vars("time_bnds")
        ds_daily_wo_bnds = ds_daily_with_bnds.drop_vars("time_bnds")
        ds_monthly_wo_bnds = ds_monthly_with_bnds.drop_vars("time_bnds")
        ds_yearly_wo_bnds = ds_yearly_with_bnds.drop_vars("time_bnds")

        # test adding bounds
        hourly_bounds = ds_hrly_wo_bnds.bounds.add_time_bounds(
            method="freq", freq="hour"
        )
        daily_bounds = ds_daily_wo_bnds.bounds.add_time_bounds(
            method="freq", freq="day"
        )
        monthly_bounds = ds_monthly_wo_bnds.bounds.add_time_bounds(
            method="freq", freq="month"
        )
        yearly_bounds = ds_yearly_wo_bnds.bounds.add_time_bounds(
            method="freq", freq="year"
        )

        # sub hourly data is not supported
        with pytest.raises(ValueError):
            ds_subhrly_wo_bnds.bounds.add_time_bounds(method="freq", freq="hour")

        # ensure identical
        assert hourly_bounds.identical(ds_hrly_with_bnds)
        assert daily_bounds.identical(ds_daily_with_bnds)
        assert monthly_bounds.identical(ds_monthly_with_bnds)
        assert yearly_bounds.identical(ds_yearly_with_bnds)

    def test_add_monthly_bounds_for_end_of_month_set_to_true(self):
        ds_with_bnds = self.ds_with_bnds.copy()

        # Get time axis and create new axis with time set to first day of month.
        time = ds_with_bnds.time
        new_time = []
        for i, t in enumerate(time.values):
            y = t.year
            m = t.month
            nt = cftime.DatetimeGregorian(y, m, 1, 0)
            new_time.append(nt)
        attrs = time.attrs
        time = xr.DataArray(
            name="time",
            data=new_time,
            coords=dict(time=time),
            dims=[*time.dims],
            attrs=attrs,
        )
        time.encoding = {"calendar": "standard"}
        ds_with_bnds["time"] = time

        # test dataset
        ds = ds_with_bnds.drop_vars("time_bnds")
        result = ds.bounds.add_time_bounds(
            method="freq", freq="month", end_of_month=True
        )

        expected = ds_with_bnds.copy()
        # Expect time bounds minus one month
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
                            2001, 11, 1, 0, 0, 0, 0, has_year_zero=False
                        ),
                        cftime.DatetimeGregorian(
                            2001, 12, 1, 0, 0, 0, 0, has_year_zero=False
                        ),
                    ],
                ],
                dtype=object,
            ),
            dims=["time", "bnds"],
            coords={"time": expected.time},
            attrs={"xcdat_bounds": "True"},
        )

        assert result.identical(expected)
