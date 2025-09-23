import numpy as np
import pytest
import xarray as xr

from tests.fixtures import generate_dataset
from xcdat.axis import (
    CFAxisKey,
    _get_bounds_dim,
    center_times,
    get_coords_by_name,
    get_dim_coords,
    get_dim_keys,
    swap_lon_axis,
)


class TestGetDimKeys:
    def test_raises_error_if_dim_name_is_not_valid(self):
        ds = xr.Dataset(
            coords={
                "invalid_lat_shortname": xr.DataArray(
                    data=np.ones(3), dims="invalid_lat_shortname"
                )
            }
        )

        with pytest.raises(KeyError):
            get_dim_keys(ds, "Y")

    def test_returns_dim_name(self):
        ds = xr.Dataset(
            coords={
                "lat": xr.DataArray(
                    data=np.ones(3), dims="lat", attrs={"standard_name": "latitude"}
                )
            }
        )

        dim = get_dim_keys(ds, "Y")

        assert dim == "lat"

    def test_returns_dim_names(self):
        ds = xr.Dataset(
            coords={
                "ilev": xr.DataArray(data=np.ones(3), dims="ilev", attrs={"axis": "Z"}),
                "lev": xr.DataArray(data=np.ones(3), dims="lev", attrs={"axis": "Z"}),
            }
        )

        dim = get_dim_keys(ds, "Z")

        assert dim == ["ilev", "lev"]


class TestGetDimCoords:
    @pytest.fixture(autouse=True)
    def setup(self):
        # A dataset with "axis" attr set on all dim coord vars.
        self.ds_axis = xr.Dataset(
            data_vars={
                "hyai": xr.DataArray(
                    name="hyai",
                    data=np.ones(3),
                    dims="ilev",
                    attrs={"long_name": "hybrid A coefficient at layer interfaces"},
                ),
                "hyam": xr.DataArray(
                    name="hyam",
                    data=np.ones(3),
                    dims="lev",
                    attrs={"long_name": "hybrid B coefficient at layer interfaces"},
                ),
            },
            coords={
                "ilev": xr.DataArray(
                    data=np.ones(3),
                    dims="ilev",
                    attrs={"axis": "Z"},
                ),
                "lev": xr.DataArray(
                    data=np.ones(3),
                    dims="lev",
                    attrs={"axis": "Z"},
                ),
                "lat": xr.DataArray(
                    data=np.ones(3),
                    dims="lat",
                    attrs={"axis": "Y"},
                ),
            },
        )

        # A dataset with "standard_name" attr set on all dim coord vars.
        self.ds_sn = xr.Dataset(
            data_vars={
                "hyai": xr.DataArray(
                    name="hyai",
                    data=np.ones(3),
                    dims="ilev",
                    attrs={"long_name": "hybrid A coefficient at layer interfaces"},
                ),
                "hyam": xr.DataArray(
                    name="hyam",
                    data=np.ones(3),
                    dims="lev",
                    attrs={"long_name": "hybrid B coefficient at layer interfaces"},
                ),
            },
            coords={
                "ilev": xr.DataArray(
                    data=np.ones(3),
                    dims="ilev",
                    attrs={
                        "standard_name": "atmosphere_hybrid_sigma_pressure_coordinate"
                    },
                ),
                "lev": xr.DataArray(
                    data=np.ones(3),
                    dims="lev",
                    attrs={
                        "standard_name": "atmosphere_hybrid_sigma_pressure_coordinate"
                    },
                ),
                "lat": xr.DataArray(
                    data=np.ones(3),
                    dims="lat",
                    attrs={"standard_name": "latitude"},
                ),
            },
        )

    def test_raises_error_if_dim_does_not_exist(self):
        ds = xr.Dataset()
        dims: CFAxisKey = ["X", "Y", "T", "Z"]  # type: ignore

        for dim in dims:
            with pytest.raises(KeyError):
                get_dim_coords(ds, dim)  # type: ignore

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
            get_dim_coords(ds, "Y")

    def test_raises_error_if_a_dataarray_has_multiple_dims_for_the_same_axis(self):
        da = xr.DataArray(
            coords={
                "ilev": xr.DataArray(
                    data=np.ones(3),
                    dims="ilev",
                    attrs={"axis": "Z"},
                ),
                "lev": xr.DataArray(
                    data=np.ones(3),
                    dims="lev",
                    attrs={"axis": "Z"},
                ),
            },
            dims=["ilev", "lev"],
        )

        with pytest.raises(ValueError):
            get_dim_coords(da, "Z")

    def test_raises_error_if_multidimensional_coords_are_only_present_for_an_axis(self):
        lat = xr.DataArray(
            data=np.array([[0, 1, 2], [3, 4, 5]]),
            dims=["placeholder_1", "placeholder_2"],
            attrs={"units": "degrees_north", "axis": "Y"},
        )
        ds = xr.Dataset(coords={"lat": lat})

        with pytest.raises(KeyError):
            get_dim_coords(ds, "Y")

    def test_returns_dataset_dimension_coordinate_vars_using_common_var_names(
        self,
    ):
        ds = xr.Dataset(
            coords={
                "lat": xr.DataArray(data=np.ones(3), dims="lat"),
                "lon": xr.DataArray(data=np.ones(3), dims="lon"),
                "time": xr.DataArray(data=np.ones(3), dims="time"),
                "lev": xr.DataArray(data=np.ones(3), dims="lev"),
            }
        )

        result = get_dim_coords(ds, "X")
        xr.testing.assert_identical(result, ds["lon"])

        result = get_dim_coords(ds, "Y")
        xr.testing.assert_identical(result, ds["lat"])

        result = get_dim_coords(ds, "T")
        xr.testing.assert_identical(result, ds["time"])

        result = get_dim_coords(ds, "Z")
        xr.testing.assert_identical(result, ds["lev"])

    def test_returns_dataset_dimension_coordinate_vars_using_axis_attr(self):
        # For example, E3SM datasets might have "ilev" and "lev" dimensions
        # with the dim coord var attr "axis" both mapped to "Z".
        result = get_dim_coords(self.ds_axis, "Z")
        expected = xr.Dataset(
            coords={"ilev": self.ds_axis.ilev, "lev": self.ds_axis.lev}
        )

        xr.testing.assert_identical(result, expected)

    def test_returns_dataset_dimension_coordinate_vars_using_standard_name_attr(self):
        # For example, E3SM datasets might have "ilev" and "lev" dimensions
        # with the dim coord var attr "standard_name" both mapped to
        # "atmosphere_hybrid_sigma_pressure_coordinate".
        result = get_dim_coords(self.ds_sn, "Z")
        expected = xr.Dataset(coords={"ilev": self.ds_sn.ilev, "lev": self.ds_sn.lev})

        xr.testing.assert_identical(result, expected)

    def test_returns_dataarray_dimension_coordinate_var_using_axis_attr(self):
        result = get_dim_coords(self.ds_axis.hyai, "Z")
        expected = self.ds_axis.ilev

        assert result.identical(expected)

        result = get_dim_coords(self.ds_axis.hyam, "Z")
        expected = self.ds_axis.lev

        assert result.identical(expected)

    def test_returns_dataarray_dimension_coordinate_var_using_standard_name_attr(self):
        result = get_dim_coords(self.ds_sn.hyai, "Z")
        expected = self.ds_sn.ilev

        assert result.identical(expected)

        result = get_dim_coords(self.ds_sn.hyam, "Z")
        expected = self.ds_sn.lev

        assert result.identical(expected)


class TestGetCoordsByName:
    def test_raises_error_if_coordinate_not_found(self):
        ds = xr.Dataset(
            coords={
                "lat": xr.DataArray(data=np.ones(3), dims="lat"),
                "lon": xr.DataArray(data=np.ones(3), dims="lon"),
            }
        )

        with pytest.raises(
            KeyError, match="Coordinate with name 'T' not found in the dataset."
        ):
            get_coords_by_name(ds, "T")

    def test_returns_coordinate_from_dataset(self):
        ds = xr.Dataset(
            coords={
                "lat": xr.DataArray(data=np.ones(3), dims="lat"),
                "lon": xr.DataArray(data=np.ones(3), dims="lon"),
                "time": xr.DataArray(data=np.arange(3), dims="time"),
            }
        )

        coord = get_coords_by_name(ds, "T")
        assert coord.identical(ds["time"])

    def test_returns_coordinate_from_dataarray(self):
        da = xr.DataArray(
            data=np.random.rand(3, 3),
            dims=["lat", "lon"],
            coords={
                "lat": xr.DataArray(data=np.arange(3), dims="lat"),
                "lon": xr.DataArray(data=np.arange(3), dims="lon"),
            },
        )

        coord = get_coords_by_name(da, "X")
        assert coord.identical(da["lon"])

    def test_returns_coordinate_from_curvilinear_dataset(self):
        ds = xr.Dataset(
            coords={
                "nlat": xr.DataArray(data=np.arange(3), dims="nlat"),
                "nlon": xr.DataArray(data=np.arange(3), dims="nlon"),
                "lat": xr.DataArray(
                    data=np.random.rand(3, 3),
                    dims=["nlat", "nlon"],
                    attrs={"standard_name": "latitude"},
                ),
                "lon": xr.DataArray(
                    data=np.random.rand(3, 3),
                    dims=["nlat", "nlon"],
                    attrs={"standard_name": "longitude"},
                ),
            }
        )

        coord = get_coords_by_name(ds, "X")
        assert coord.identical(ds["lon"])

        coord = get_coords_by_name(ds, "Y")
        assert coord.identical(ds["lat"])


class TestCenterTimes:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = generate_dataset(
            decode_times=True, cf_compliant=False, has_bounds=True
        )

    def test_raises_error_if_time_coord_var_does_not_exist_in_dataset(self):
        ds = self.ds.copy()
        ds = ds.drop_dims("time")

        with pytest.raises(KeyError):
            center_times(ds)

    def test_skips_centering_time_coords_for_a_dimension_if_bounds_do_not_exist(self):
        ds = xr.Dataset(
            coords={
                "time": xr.DataArray(
                    name="time",
                    data=np.array(
                        [
                            "2000-01-31T12:00:00.000000000",
                            "2000-02-29T12:00:00.000000000",
                            "2000-03-31T12:00:00.000000000",
                        ],
                        dtype="datetime64[ns]",
                    ),
                    dims="time",
                    attrs={
                        "long_name": "time",
                        "standard_name": "time",
                        "axis": "T",
                        "bounds": "time_bnds",
                    },
                ),
            },
        )

        # Compare result of the method against the expected.
        result = center_times(ds)
        expected = ds.copy()

        assert result.identical(expected)

    def test_returns_time_coords_as_the_midpoint_between_time_bounds(self):
        ds = xr.Dataset(
            data_vars={
                "time_bnds": xr.DataArray(
                    name="time_bnds",
                    data=np.array(
                        [
                            [
                                "2000-01-01T00:00:00.000000000",
                                "2000-02-01T00:00:00.000000000",
                            ],
                            [
                                "2000-02-01T00:00:00.000000000",
                                "2000-03-01T00:00:00.000000000",
                            ],
                            [
                                "2000-03-01T00:00:00.000000000",
                                "2000-04-01T00:00:00.000000000",
                            ],
                        ],
                        dtype="datetime64[ns]",
                    ),
                    dims=["time", "bnds"],
                    attrs={
                        "xcdat_bounds": "True",
                    },
                ),
            },
            coords={
                "time": xr.DataArray(
                    name="time",
                    data=np.array(
                        [
                            "2000-01-31T12:00:00.000000000",
                            "2000-02-29T12:00:00.000000000",
                            "2000-03-31T12:00:00.000000000",
                        ],
                        dtype="datetime64[ns]",
                    ),
                    dims="time",
                    attrs={
                        "long_name": "time",
                        "standard_name": "time",
                        "axis": "T",
                        "bounds": "time_bnds",
                    },
                ),
                "time2": xr.DataArray(
                    name="time2",
                    data=np.array(
                        [
                            "2000-01-31T12:00:00.000000000",
                            "2000-02-29T12:00:00.000000000",
                            "2000-03-31T12:00:00.000000000",
                        ],
                        dtype="datetime64[ns]",
                    ),
                    dims="time",
                    attrs={
                        "long_name": "time",
                        "standard_name": "time",
                        "axis": "T",
                        "bounds": "time_bnds",
                    },
                ),
            },
        )

        result = center_times(ds)

        # Compare result of the method against the expected.
        expected = xr.Dataset(
            data_vars={
                "time_bnds": xr.DataArray(
                    name="time_bnds",
                    data=np.array(
                        [
                            [
                                "2000-01-01T00:00:00.000000000",
                                "2000-02-01T00:00:00.000000000",
                            ],
                            [
                                "2000-02-01T00:00:00.000000000",
                                "2000-03-01T00:00:00.000000000",
                            ],
                            [
                                "2000-03-01T00:00:00.000000000",
                                "2000-04-01T00:00:00.000000000",
                            ],
                        ],
                        dtype="datetime64[ns]",
                    ),
                    dims=["time", "bnds"],
                    attrs={
                        "xcdat_bounds": "True",
                    },
                ),
            },
            coords={
                "time": xr.DataArray(
                    name="time",
                    data=np.array(
                        [
                            "2000-01-16T12:00:00.000000000",
                            "2000-02-15T12:00:00.000000000",
                            "2000-03-16T12:00:00.000000000",
                        ],
                        dtype="datetime64[ns]",
                    ),
                    dims="time",
                    attrs={
                        "long_name": "time",
                        "standard_name": "time",
                        "axis": "T",
                        "bounds": "time_bnds",
                    },
                ),
                "time2": xr.DataArray(
                    name="time2",
                    data=np.array(
                        [
                            "2000-01-16T12:00:00.000000000",
                            "2000-02-15T12:00:00.000000000",
                            "2000-03-16T12:00:00.000000000",
                        ],
                        dtype="datetime64[ns]",
                    ),
                    dims="time",
                    attrs={
                        "long_name": "time",
                        "standard_name": "time",
                        "axis": "T",
                        "bounds": "time_bnds",
                    },
                ),
            },
        )

        assert result.identical(expected)


class TestSwapLonAxis:
    def test_raises_error_if_no_longitude_axis_exists(self):
        ds = generate_dataset(decode_times=True, cf_compliant=False, has_bounds=True)
        ds = ds.drop_dims("lon")

        with pytest.raises(KeyError):
            swap_lon_axis(ds, to=(-180, 180))

    def test_raises_error_with_incorrect_lon_orientation_for_swapping(self):
        ds = generate_dataset(decode_times=True, cf_compliant=False, has_bounds=True)
        with pytest.raises(ValueError):
            swap_lon_axis(ds, to=9000)  # type: ignore

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

        result = swap_lon_axis(ds_360, to=(0, 360), sort_ascending=False)

        assert result.identical(ds_360)

    def test_does_not_swap_bounds_if_bounds_do_not_exist(self):
        ds_360 = xr.Dataset(
            coords={
                "lon": xr.DataArray(
                    name="lon",
                    data=np.array([60, 150, 271]),
                    dims=["lon"],
                    attrs={"units": "degrees_east", "axis": "X", "bounds": "lon_bnds"},
                ),
            },
            data_vars={
                "ts": xr.DataArray(
                    name="ts",
                    data=np.array([0, 1, 2]),
                    dims=["lon"],
                    attrs={"test_attr": "test"},
                ),
            },
        )

        result = swap_lon_axis(ds_360, to=(-180, 180))
        expected = xr.Dataset(
            coords={
                "lon": xr.DataArray(
                    data=np.array([-89, 60, 150]),
                    dims=["lon"],
                    attrs={"units": "degrees_east", "axis": "X", "bounds": "lon_bnds"},
                ),
            },
            data_vars={
                "ts": xr.DataArray(
                    name="ts",
                    data=np.array([2, 0, 1]),
                    dims=["lon"],
                    attrs={"test_attr": "test"},
                ),
            },
        )

        assert result.identical(expected)

    def test_indempotency_when_converting_0_to_360_to_0_to_360(self):
        ds_360 = xr.Dataset(
            coords={
                "lon": xr.DataArray(
                    name="lon",
                    data=np.array([0, 90, 180, 270, 360]),
                    dims=["lon"],
                    attrs={"units": "degrees_east", "axis": "X", "bounds": "lon_bnds"},
                ),
            },
            data_vars={
                "ts": xr.DataArray(
                    name="ts",
                    data=np.array([0, 1, 2, 3, 4]),
                    dims=["lon"],
                    attrs={"test_attr": "test"},
                ),
                "lon_bnds": xr.DataArray(
                    name="lon_bnds",
                    data=np.array(
                        [
                            [0, 90],
                            [90, 180],
                            [180, 270],
                            [270, 360],
                            [360, 360],
                        ]
                    ),
                    dims=["lon", "bnds"],
                    attrs={"xcdat_bounds": "True"},
                ),
            },
        )

        result = swap_lon_axis(ds_360, to=(0, 360))
        xr.testing.assert_identical(result, ds_360)

    def test_indempotency_when_converting_minus_180_to_180_to_minus_180_to_180(self):
        ds_180 = xr.Dataset(
            coords={
                "lon": xr.DataArray(
                    name="lon",
                    data=np.array([-180, -90, 0, 90, 180]),
                    dims=["lon"],
                    attrs={"units": "degrees_east", "axis": "X", "bounds": "lon_bnds"},
                ),
            },
            data_vars={
                "ts": xr.DataArray(
                    name="ts",
                    data=np.array([0, 1, 2, 3, 4]),
                    dims=["lon"],
                    attrs={"test_attr": "test"},
                ),
                "lon_bnds": xr.DataArray(
                    name="lon_bnds",
                    data=np.array(
                        [
                            [-180, -135],
                            [-135, -45],
                            [-45, 45],
                            [45, 135],
                            [135, 180],
                        ]
                    ),
                    dims=["lon", "bnds"],
                    attrs={"xcdat_bounds": "True"},
                ),
            },
        )

        result = swap_lon_axis(ds_180, to=(-180, 180))

        xr.testing.assert_identical(result, ds_180)

    def test_swaps_single_dim_from_360_to_180_and_sorts(self):
        ds_360 = xr.Dataset(
            coords={
                "lon": xr.DataArray(
                    name="lon",
                    data=np.array([60, 150, 271]),
                    dims=["lon"],
                    attrs={"units": "degrees_east", "axis": "X", "bounds": "lon_bnds"},
                ),
                "lon2": xr.DataArray(
                    name="lon2",
                    data=np.array([60, 150, 271]),
                    dims=["lon"],
                    attrs={"units": "degrees_east", "axis": "X", "bounds": "lon_bnds"},
                ),
            },
            data_vars={
                "ts": xr.DataArray(
                    name="ts",
                    data=np.array([0, 1, 2]),
                    dims=["lon"],
                    attrs={"test_attr": "test"},
                ),
                "lon_bnds": xr.DataArray(
                    name="lon_bnds",
                    data=np.array([[0, 120], [120, 181], [181, 360]]),
                    dims=["lon", "bnds"],
                    attrs={"xcdat_bounds": "True"},
                ),
            },
        )
        result = swap_lon_axis(ds_360, to=(-180, 180))
        expected = xr.Dataset(
            coords={
                "lon": xr.DataArray(
                    data=np.array([-89, 60, 150]),
                    dims=["lon"],
                    attrs={"units": "degrees_east", "axis": "X", "bounds": "lon_bnds"},
                ),
                "lon2": xr.DataArray(
                    name="lon2",
                    data=np.array([-89, 60, 150]),
                    dims=["lon"],
                    attrs={"units": "degrees_east", "axis": "X", "bounds": "lon_bnds"},
                ),
            },
            data_vars={
                "ts": xr.DataArray(
                    name="ts",
                    data=np.array([2, 0, 1]),
                    dims=["lon"],
                    attrs={"test_attr": "test"},
                ),
                "lon_bnds": xr.DataArray(
                    name="lon_bnds",
                    data=np.array([[-179, 0], [0, 120], [120, -179]]),
                    dims=["lon", "bnds"],
                    attrs={"xcdat_bounds": "True"},
                ),
            },
        )

        assert result.identical(expected)

    def test_swaps_single_dim_from_180_to_360_without_prime_meridian_cell(self):
        ds_180 = xr.Dataset(
            coords={
                "lon": xr.DataArray(
                    name="lon",
                    data=np.array([-135, -45, 45, 135]),
                    dims=["lon"],
                    attrs={"units": "degrees_east", "axis": "X", "bounds": "lon_bnds"},
                ),
            },
            data_vars={
                "ts": xr.DataArray(
                    name="ts",
                    data=np.array([0, 1, 2, 3]),
                    dims=["lon"],
                    attrs={"test_attr": "test"},
                ),
                "lon_bnds": xr.DataArray(
                    name="lon_bnds",
                    data=np.array(
                        [
                            [-180, -90],
                            [-90, 0],
                            [0, 90],
                            [90, 180],
                        ]
                    ),
                    dims=["lon", "bnds"],
                    attrs={"xcdat_bounds": "True"},
                ),
            },
        )
        result = swap_lon_axis(ds_180, to=(0, 360))
        expected = xr.Dataset(
            coords={
                "lon": xr.DataArray(
                    name="lon",
                    data=np.array([45, 135, 225, 315]),
                    dims=["lon"],
                    attrs={"units": "degrees_east", "axis": "X", "bounds": "lon_bnds"},
                ),
            },
            data_vars={
                "ts": xr.DataArray(
                    name="ts",
                    data=np.array([2, 3, 0, 1]),
                    dims=["lon"],
                    attrs={"test_attr": "test"},
                ),
                "lon_bnds": xr.DataArray(
                    name="lon_bnds",
                    data=np.array(
                        [
                            [0, 90],
                            [90, 180],
                            [180, 270],
                            [270, 360],
                        ]
                    ),
                    dims=["lon", "bnds"],
                    attrs={"xcdat_bounds": "True"},
                ),
            },
        )

        xr.testing.assert_identical(result, expected)

    def test_swaps_single_dim_from_180_to_360_and_normalizes_prime_meridian_cell_in_lon_bnds_to_360(
        self,
    ):
        ds_180 = xr.Dataset(
            coords={
                "lon": xr.DataArray(
                    name="lon",
                    data=np.array([-179.5, -1.5, -0.5, 1.5, 179.5]),
                    dims=["lon"],
                    attrs={"units": "degrees_east", "axis": "X", "bounds": "lon_bnds"},
                ),
            },
            data_vars={
                "ts": xr.DataArray(
                    name="ts",
                    data=np.array([0, 1, 2, 3, 4]),
                    dims=["lon"],
                    attrs={"test_attr": "test"},
                ),
                "lon_bnds": xr.DataArray(
                    name="lon_bnds",
                    data=np.array(
                        [
                            [-180.0, -179],
                            [-2.0, -1.0],
                            [-1.0, 0.0],  # Prime meridian cell.
                            [1.0, 2.0],
                            [179.0, 180.0],
                        ]
                    ),
                    dims=["lon", "bnds"],
                    attrs={"xcdat_bounds": "True"},
                ),
            },
        )
        result = swap_lon_axis(ds_180, to=(0, 360))
        expected = xr.Dataset(
            coords={
                "lon": xr.DataArray(
                    name="lon",
                    data=np.array([1.5, 179.5, 180.5, 358.5, 359.5]),
                    dims=["lon"],
                    attrs={"units": "degrees_east", "axis": "X", "bounds": "lon_bnds"},
                ),
            },
            data_vars={
                "ts": xr.DataArray(
                    name="ts",
                    data=np.array([3, 4, 0, 1, 2]),
                    dims=["lon"],
                    attrs={"test_attr": "test"},
                ),
                "lon_bnds": xr.DataArray(
                    name="lon_bnds",
                    data=np.array(
                        [
                            [1.0, 2.0],
                            [179.0, 180.0],
                            [180.0, 181.0],
                            [358.0, 359.0],
                            # Instead of [359, 0], normalize to [359, 360].
                            [359.0, 360.0],
                        ]
                    ),
                    dims=["lon", "bnds"],
                    attrs={"xcdat_bounds": "True"},
                ),
            },
        )

        xr.testing.assert_identical(result, expected)

    def test_swaps_multiple_dims_from_180_to_360_and_normalizes_prime_meridian_cell_in_lon_bnds_to_360(
        self,
    ):
        ds_180 = xr.Dataset(
            coords={
                "lon": xr.DataArray(
                    name="lon",
                    data=np.array([-179.5, -1.5, -0.5, 1.5, 179.5]),
                    dims=["lon"],
                    attrs={"units": "degrees_east", "axis": "X", "bounds": "lon_bnds"},
                ),
                "zlon": xr.DataArray(
                    name="zlon",
                    data=np.array([-179.5, -1.5, -0.5, 1.5, 179.5]),
                    dims=["zlon"],
                    attrs={"units": "degrees_east", "axis": "X", "bounds": "zlon_bnds"},
                ),
            },
            data_vars={
                "ts": xr.DataArray(
                    name="ts",
                    data=np.array([0, 1, 2, 3, 4]),
                    dims=["lon"],
                    attrs={"test_attr": "test"},
                ),
                "ts2": xr.DataArray(
                    name="ts2",
                    data=np.array([0, 1, 2, 3, 4]),
                    dims=["zlon"],
                    attrs={"test_attr": "test"},
                ),
                "lon_bnds": xr.DataArray(
                    name="lon_bnds",
                    data=np.array(
                        [
                            [-180.0, -179],
                            [-2.0, -1.0],
                            [-1.0, 0.0],  # Prime meridian cell.
                            [1.0, 2.0],
                            [179.0, 180.0],
                        ]
                    ),
                    dims=["lon", "bnds"],
                    attrs={"xcdat_bounds": "True"},
                ),
                "zlon_bnds": xr.DataArray(
                    name="zlon_bnds",
                    data=np.array(
                        [
                            [-180.0, -179],
                            [-2.0, -1.0],
                            [-1.0, 0.0],  # Prime meridian cell.
                            [1.0, 2.0],
                            [179.0, 180.0],
                        ]
                    ),
                    dims=["zlon", "bnds"],
                    attrs={"xcdat_bounds": "True"},
                ),
            },
        )
        result = swap_lon_axis(ds_180, to=(0, 360))
        expected = xr.Dataset(
            coords={
                "lon": xr.DataArray(
                    name="lon",
                    data=np.array([1.5, 179.5, 180.5, 358.5, 359.5]),
                    dims=["lon"],
                    attrs={"units": "degrees_east", "axis": "X", "bounds": "lon_bnds"},
                ),
                "zlon": xr.DataArray(
                    name="zlon",
                    data=np.array([1.5, 179.5, 180.5, 358.5, 359.5]),
                    dims=["zlon"],
                    attrs={"units": "degrees_east", "axis": "X", "bounds": "zlon_bnds"},
                ),
            },
            data_vars={
                "ts": xr.DataArray(
                    name="ts",
                    data=np.array([3, 4, 0, 1, 2]),
                    dims=["lon"],
                    attrs={"test_attr": "test"},
                ),
                "ts2": xr.DataArray(
                    name="ts",
                    data=np.array([3, 4, 0, 1, 2]),
                    dims=["zlon"],
                    attrs={"test_attr": "test"},
                ),
                "lon_bnds": xr.DataArray(
                    name="lon_bnds",
                    data=np.array(
                        [
                            [1.0, 2.0],
                            [179.0, 180.0],
                            [180.0, 181.0],
                            [358.0, 359.0],
                            # Instead of [359, 0], normalize to [359, 360].
                            [359.0, 360.0],
                        ]
                    ),
                    dims=["lon", "bnds"],
                    attrs={"xcdat_bounds": "True"},
                ),
                "zlon_bnds": xr.DataArray(
                    name="zlon_bnds",
                    data=np.array(
                        [
                            [1.0, 2.0],
                            [179.0, 180.0],
                            [180.0, 181.0],
                            [358.0, 359.0],
                            # Instead of [359, 0], normalize to [359, 360].
                            [359.0, 360.0],
                        ]
                    ),
                    dims=["zlon", "bnds"],
                    attrs={"xcdat_bounds": "True"},
                ),
            },
        )

        xr.testing.assert_identical(result, expected)


class TestGetBoundsDim:
    def test_returns_bounds_dim_for_standard_case(self):
        coords = xr.DataArray([0, 1, 2], dims=["lat"])
        bounds = xr.DataArray([[0, 1], [1, 2], [2, 3]], dims=["lat", "bnds"])

        result = _get_bounds_dim(coords, bounds)

        assert result == "bnds"

    def test_returns_bounds_dim_when_bounds_dim_has_custom_name(self):
        coords = xr.DataArray([10, 20, 30], dims=["lon"])
        bounds = xr.DataArray([[5, 15], [15, 25], [25, 35]], dims=["lon", "boundaries"])

        result = _get_bounds_dim(coords, bounds)

        assert result == "boundaries"

    def test_raises_error_when_bounds_has_no_extra_dim(self):
        coords = xr.DataArray([0, 1, 2], dims=["lat"])
        bounds = xr.DataArray([0, 1, 2], dims=["lat"])

        with pytest.raises(
            ValueError, match="No extra dimension found in bounds variable"
        ):
            _get_bounds_dim(coords, bounds)

    def test_raises_error_when_bounds_has_multiple_extra_dims(self):
        coords = xr.DataArray([0, 1, 2], dims=["lat"])
        bounds = xr.DataArray(np.zeros((3, 2, 2)), dims=["lat", "bnds", "extra"])

        with pytest.raises(
            ValueError, match="Bounds variable must have exactly one more dimension"
        ):
            _get_bounds_dim(coords, bounds)
