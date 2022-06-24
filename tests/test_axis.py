import numpy as np
import pytest
import xarray as xr

from tests.fixtures import generate_dataset
from xcdat.axis import center_times, get_axis_coord, get_axis_dim, swap_lon_axis


class TestGetAxisCoord:
    def test_raises_error_if_coord_var_does_not_exist(self):
        ds = xr.Dataset()

        with pytest.raises(KeyError):
            get_axis_coord(ds, "Y")

    def test_raises_error_if_axis_or_standard_name_is_not_set_or_dim_name_is_not_valid(
        self,
    ):
        ds = xr.Dataset(
            coords={
                "invalid_lat_shortname": xr.DataArray(
                    data=np.ones(3), dims="invalid_lat_shortname"
                )
            }
        )

        with pytest.raises(KeyError):
            get_axis_coord(ds, "Y")

    def test_returns_coord_var_if_axis_attr_is_set(self):
        # Set the dimension name to something other than "lat" to make sure
        # axis attr is being used for the match.
        ds = xr.Dataset(
            coords={
                "lat_not_short_name": xr.DataArray(
                    data=np.ones(3), dims="lat_not_short_name", attrs={"axis": "Y"}
                )
            }
        )

        result = get_axis_coord(ds, "Y")
        expected = ds.lat_not_short_name

        assert result.identical(expected)

    def test_returns_coord_var_if_standard_name_attr_is_set(self):
        # Set the dimension name to something other than "lat" to make sure
        # standard_name attr is being used for the match.
        ds = xr.Dataset(
            coords={
                "lat_not_short_name": xr.DataArray(
                    data=np.ones(3),
                    dims="lat_not_short_name",
                    attrs={"standard_name": "latitude"},
                )
            }
        )

        result = get_axis_coord(ds, "Y")
        expected = ds.lat_not_short_name

        assert result.identical(expected)

    def test_returns_coord_var_if_dim_name_is_valid(self):
        ds = xr.Dataset(coords={"lat": xr.DataArray(data=np.ones(3), dims="lat")})

        result = get_axis_coord(ds, "Y")
        expected = ds.lat

        assert result.identical(expected)


class TestGetAxisDim:
    def test_raises_error_if_dim_name_is_not_valid(self):
        ds = xr.Dataset(
            coords={
                "invalid_lat_shortname": xr.DataArray(
                    data=np.ones(3), dims="invalid_lat_shortname"
                )
            }
        )

        with pytest.raises(KeyError):
            get_axis_dim(ds, "Y")

    def test_returns_dim_name(self):
        ds = xr.Dataset(
            coords={
                "lat": xr.DataArray(
                    data=np.ones(3), dims="lat", attrs={"standard_name": "latitude"}
                )
            }
        )

        dim = get_axis_dim(ds, "Y")

        assert dim == "lat"


class TestCenterTimes:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = generate_dataset(cf_compliant=True, has_bounds=True)

    def test_raises_error_if_time_coord_var_does_not_exist_in_dataset(self):
        ds = self.ds.copy()
        ds = ds.drop_dims("time")

        with pytest.raises(KeyError):
            center_times(ds)

    def test_raises_error_if_time_bounds_does_not_exist_in_the_dataset(self):
        ds = self.ds.copy()
        ds = ds.drop_vars("time_bnds")

        with pytest.raises(KeyError):
            center_times(ds)

    def test_gets_time_as_the_midpoint_between_time_bounds(self):
        ds = self.ds.copy()

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
        ds.time.data[:] = uncentered_time

        # Compare result of the method against the expected.
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
        # Update time bounds with centered time coordinates.
        time_bounds = ds.time_bnds.copy()
        time_bounds["time"] = expected.time
        expected["time_bnds"] = time_bounds

        result = center_times(ds)
        assert result.identical(expected)


class TestSwapLonAxis:
    def test_raises_error_with_incorrect_lon_orientation_for_swapping(self):
        ds = generate_dataset(cf_compliant=True, has_bounds=True)
        with pytest.raises(ValueError):
            swap_lon_axis(ds, to=9000)  # type: ignore

    def test_raises_error_if_lon_bounds_contains_more_than_one_prime_meridian_cell(
        self,
    ):
        ds_180 = xr.Dataset(
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
                            [-180.5, 1.5],
                        ]
                    ),
                    dims=["lon", "bnds"],
                    attrs={"xcdat_bounds": "True"},
                ),
                "ts": xr.DataArray(
                    name="ts",
                    data=np.array([0, 1, 2, 3, 4]),
                    dims=["lon"],
                    attrs={"test_attr": "test"},
                ),
            },
        )

        with pytest.raises(ValueError):
            swap_lon_axis(ds_180, to=(0, 360))

    def test_does_not_swap_if_desired_orientation_is_the_same_as_the_existing_orientation(
        self,
    ):
        ds_360 = xr.Dataset(
            coords={
                "lon": xr.DataArray(
                    name="lon",
                    data=np.array([60, 150, 271]),
                    dims=["lon"],
                    attrs={"units": "degrees_east", "axis": "X", "bounds": "lon_bnds"},
                )
            },
            data_vars={
                "lon_bnds": xr.DataArray(
                    name="lon_bnds",
                    data=np.array([[0, 120], [120, 181], [181, 360]]),
                    dims=["lon", "bnds"],
                    attrs={"xcdat_bounds": "True"},
                )
            },
        )

        result = swap_lon_axis(ds_360, to=(0, 360))

        assert result.identical(ds_360)

    def test_swap_from_360_to_180_and_sorts(self):
        ds_360 = xr.Dataset(
            coords={
                "lon": xr.DataArray(
                    name="lon",
                    data=np.array([60, 150, 271]),
                    dims=["lon"],
                    attrs={"units": "degrees_east", "axis": "X", "bounds": "lon_bnds"},
                )
            },
            data_vars={
                "lon_bnds": xr.DataArray(
                    name="lon_bnds",
                    data=np.array([[0, 120], [120, 181], [181, 360]]),
                    dims=["lon", "bnds"],
                    attrs={"xcdat_bounds": "True"},
                )
            },
        )

        result = swap_lon_axis(ds_360, to=(-180, 180))
        expected = xr.Dataset(
            coords={
                "lon": xr.DataArray(
                    data=np.array([-89, 60, 150]),
                    dims=["lon"],
                    attrs={"units": "degrees_east", "axis": "X", "bounds": "lon_bnds"},
                )
            },
            data_vars={
                "lon_bnds": xr.DataArray(
                    name="lon_bnds",
                    data=np.array([[-179, 0], [0, 120], [120, -179]]),
                    dims=["lon", "bnds"],
                    attrs={"xcdat_bounds": "True"},
                )
            },
        )

        assert result.identical(expected)

    def test_swap_from_180_to_360_and_sorts_with_prime_meridian_cell_in_lon_bnds(self):
        ds_180 = xr.Dataset(
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
                "ts": xr.DataArray(
                    name="ts",
                    data=np.array([0, 1, 2, 3, 4]),
                    dims=["lon"],
                    attrs={"test_attr": "test"},
                ),
            },
        )

        result = swap_lon_axis(ds_180, to=(0, 360))
        expected = xr.Dataset(
            coords={
                "lon": xr.DataArray(
                    name="lon",
                    data=np.array([0, 1, 179, 180, 359, 360]),
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
                "ts": xr.DataArray(
                    name="ts",
                    data=np.array([2, 3, 4, 0, 1, 2]),
                    dims=["lon"],
                    attrs={"test_attr": "test"},
                ),
            },
        )

        assert result.identical(expected)
