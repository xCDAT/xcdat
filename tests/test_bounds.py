import logging

import cftime
import numpy as np
import pytest
import xarray as xr

from tests.fixtures import generate_dataset, lat_bnds, lon_bnds
from xcdat.bounds import BoundsAccessor

logger = logging.getLogger(__name__)


class TestBoundsAccessor:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = generate_dataset(cf_compliant=True, has_bounds=False)
        self.ds_with_bnds = generate_dataset(cf_compliant=True, has_bounds=True)

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
        self.ds = generate_dataset(cf_compliant=True, has_bounds=False)
        self.ds_with_bnds = generate_dataset(cf_compliant=True, has_bounds=True)

    def test_adds_bounds_to_the_dataset(self):
        ds = self.ds_with_bnds.copy()

        ds = ds.drop_vars(["lat_bnds", "lon_bnds"])

        result = ds.bounds.add_missing_bounds()
        assert result.identical(self.ds_with_bnds)

    def test_adds_bounds_to_the_dataset_skips_nondimensional_axes(self):
        # generate dataset with height coordinate
        ds = generate_dataset(cf_compliant=True, has_bounds=True)
        ds = ds.assign_coords({"height": 2})

        # drop bounds
        dsm = ds.drop_vars(["lat_bnds", "lon_bnds"]).copy()

        # test bounds re-generation
        result = dsm.bounds.add_missing_bounds()

        # dataset with missing bounds added should match dataset with bounds
        # and added height coordinate
        assert result.identical(ds)

    def test_skips_adding_bounds_for_coords_that_are_multidimensional_or_len_of_1(self):
        # Multidimensional
        lat = xr.DataArray(
            data=np.array([[0, 1, 2], [3, 4, 5]]),
            dims=["placeholder_1", "placeholder_2"],
            attrs={"units": "degrees_north", "axis": "Y"},
        )
        # Length <=1
        lon = xr.DataArray(
            data=np.array([0]),
            dims=["lon"],
            attrs={"units": "degrees_east", "axis": "X"},
        )
        ds = xr.Dataset(coords={"lat": lat, "lon": lon})

        result = ds.bounds.add_missing_bounds("Y")

        assert result.identical(ds)


class TestGetBounds:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = generate_dataset(cf_compliant=True, has_bounds=False)
        self.ds_with_bnds = generate_dataset(cf_compliant=True, has_bounds=True)

    def test_raises_error_with_invalid_axis_key(self):
        with pytest.raises(ValueError):
            self.ds.bounds.get_bounds("incorrect_axis_argument")

    def test_raises_error_if_bounds_attr_is_not_set_on_coord_var(self):
        ds = xr.Dataset(
            coords={
                "lat": xr.DataArray(data=np.ones(3), dims="lat", attrs={"axis": "Y"})
            },
            data_vars={
                "lat_bnds": xr.DataArray(data=np.ones((3, 3)), dims=["lat", "bnds"])
            },
        )
        with pytest.raises(KeyError):
            ds.bounds.get_bounds("Y")

    def test_raises_error_if_bounds_attr_is_set_but_no_bounds_data_var_exists(self):
        ds = xr.Dataset(
            coords={
                "lat": xr.DataArray(
                    data=np.ones(3),
                    dims="lat",
                    attrs={"axis": "Y", "bounds": "lat_bnds"},
                )
            }
        )

        with pytest.raises(KeyError):
            ds.bounds.get_bounds("Y")

    def test_returns_bounds(self):
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

        lat_bnds = ds.bounds.get_bounds("Y")

        assert lat_bnds.identical(ds.lat_bnds)


class TestAddBounds:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = generate_dataset(cf_compliant=True, has_bounds=False)
        self.ds_with_bnds = generate_dataset(cf_compliant=True, has_bounds=True)

    def test_raises_error_if_bounds_already_exist(self):
        ds = self.ds_with_bnds.copy()

        with pytest.raises(ValueError):
            ds.bounds.add_bounds("Y")

    def test_raises_errors_for_data_dim_and_length(self, caplog):
        # Multidimensional
        lat = xr.DataArray(
            data=np.array([[0, 1, 2], [3, 4, 5]]),
            dims=["placeholder_1", "placeholder_2"],
            attrs={"units": "degrees_north", "axis": "Y"},
        )
        # Length <=1
        lon = xr.DataArray(
            data=np.array([0]),
            dims=["lon"],
            attrs={"units": "degrees_east", "axis": "X"},
        )
        ds = xr.Dataset(coords={"lat": lat, "lon": lon})

        # If coords dimensions does not equal 1.
        with pytest.raises(ValueError):
            ds.bounds.add_bounds("Y")

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
        # Suppress the warning
        caplog.set_level(logging.CRITICAL)

        ds = self.ds.copy()
        del ds.lat.attrs["units"]

        result = ds.bounds.add_bounds("Y")
        assert result.lat_bnds.equals(lat_bnds)
        assert result.lat_bnds.xcdat_bounds == "True"
        assert result.lat.attrs["units"] == "degrees_north"
        assert result.lat.attrs["bounds"] == "lat_bnds"

    def test_add_bounds_for_dataset_with_time_coords_as_datetime_objects(self):
        ds = self.ds.copy()

        result = ds.bounds.add_bounds("Y")
        assert result.lat_bnds.equals(lat_bnds)
        assert result.lat_bnds.xcdat_bounds == "True"
        assert result.lat.attrs["bounds"] == "lat_bnds"

        result = result.bounds.add_bounds("X")
        assert result.lon_bnds.equals(lon_bnds)
        assert result.lon_bnds.xcdat_bounds == "True"
        assert result.lon.attrs["bounds"] == "lon_bnds"

        result = ds.bounds.add_bounds("T")
        # NOTE: The algorithm for generating time bounds doesn't extend the
        # upper bound into the next month.
        expected_time_bnds = xr.DataArray(
            name="time_bnds",
            data=np.array(
                [
                    ["2000-01-01T12:00:00.000000000", "2000-01-31T12:00:00.000000000"],
                    ["2000-01-31T12:00:00.000000000", "2000-03-01T12:00:00.000000000"],
                    ["2000-03-01T12:00:00.000000000", "2000-03-31T18:00:00.000000000"],
                    ["2000-03-31T18:00:00.000000000", "2000-05-01T06:00:00.000000000"],
                    ["2000-05-01T06:00:00.000000000", "2000-05-31T18:00:00.000000000"],
                    ["2000-05-31T18:00:00.000000000", "2000-07-01T06:00:00.000000000"],
                    ["2000-07-01T06:00:00.000000000", "2000-08-01T00:00:00.000000000"],
                    ["2000-08-01T00:00:00.000000000", "2000-08-31T18:00:00.000000000"],
                    ["2000-08-31T18:00:00.000000000", "2000-10-01T06:00:00.000000000"],
                    ["2000-10-01T06:00:00.000000000", "2000-10-31T18:00:00.000000000"],
                    ["2000-10-31T18:00:00.000000000", "2000-12-01T06:00:00.000000000"],
                    ["2000-12-01T06:00:00.000000000", "2001-01-01T00:00:00.000000000"],
                    ["2001-01-01T00:00:00.000000000", "2001-01-31T06:00:00.000000000"],
                    ["2001-01-31T06:00:00.000000000", "2001-07-17T06:00:00.000000000"],
                    ["2001-07-17T06:00:00.000000000", "2002-05-17T18:00:00.000000000"],
                ],
                dtype="datetime64[ns]",
            ),
            coords={"time": ds.time.assign_attrs({"bounds": "time_bnds"})},
            dims=["time", "bnds"],
            attrs={"xcdat_bounds": "True"},
        )

        assert result.time_bnds.identical(expected_time_bnds)

    def test_returns_bounds_for_dataset_with_time_coords_as_cftime_objects(self):
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
