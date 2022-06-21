import numpy as np
import pytest
import xarray as xr

from tests import requires_dask
from tests.fixtures import generate_dataset
from xcdat.spatial import SpatialAccessor


class TestSpatialAccessor:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = generate_dataset(cf_compliant=True, has_bounds=True)

    def test__init__(self):
        ds = self.ds.copy()
        obj = SpatialAccessor(ds)

        assert obj._dataset.identical(ds)

    def test_decorator_call(self):
        ds = self.ds.copy()
        obj = ds.spatial

        assert obj._dataset.identical(ds)


class TestAverage:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = generate_dataset(cf_compliant=True, has_bounds=True)

        # Limit to just 3 data points to simplify testing.
        self.ds = self.ds.isel(time=slice(None, 3))

        # Change the value of the first element so that it is easier to identify
        # changes in the output.
        self.ds["ts"].data[0] = np.full((4, 4), 2.25)

    def test_raises_error_if_data_var_not_in_dataset(self):
        with pytest.raises(KeyError):
            self.ds.spatial.average("not_a_data_var", axis=["Y", "incorrect_axis"])

    def test_raises_error_if_axis_list_contains_unsupported_axis(self):
        with pytest.raises(ValueError):
            self.ds.spatial.average("ts", axis=["Y", "incorrect_axis"])

    def test_raises_error_if_lat_axis_does_not_exist(self):
        ds = self.ds.copy()
        ds.lat.attrs["axis"] = None
        with pytest.raises(KeyError):
            ds.spatial.average("ts", axis=["X", "Y"])

    def test_raises_error_if_lon_axis_does_not_exist(self):
        ds = self.ds.copy()
        ds.lon.attrs["axis"] = None
        with pytest.raises(KeyError):
            ds.spatial.average("ts", axis=["X", "Y"])

    def test_raises_error_if_bounds_type_is_not_a_tuple(self):
        with pytest.raises(TypeError):
            self.ds.spatial.average("ts", axis=["Y"], lat_bounds=[1, 1])

        with pytest.raises(TypeError):
            self.ds.spatial.average("ts", axis=["Y"], lat_bounds="str")

    def test_raises_error_if_there_are_0_elements_in_the_bounds(self):
        with pytest.raises(ValueError):
            self.ds.spatial.average("ts", axis=["Y"], lat_bounds=())

    def test_raises_error_if_there_are_more_than_two_elements_in_the_bounds(self):
        with pytest.raises(ValueError):
            self.ds.spatial.average("ts", axis=["Y"], lat_bounds=(1, 2, 3))

    def test_raises_error_if_lower_bound_is_not_a_float_or_int(self):
        with pytest.raises(TypeError):
            self.ds.spatial.average("ts", axis=["Y"], lat_bounds=("invalid", 1))

    def test_raises_error_if_upper_bound_is_not_a_float_or_int(self):
        with pytest.raises(TypeError):
            self.ds.spatial.average("ts", axis=["Y"], lat_bounds=(1, "invalid"))

    def test_raises_error_if_lower_lat_bound_is_larger_than_upper(self):
        with pytest.raises(ValueError):
            self.ds.spatial.average("ts", axis=["Y"], lat_bounds=(2, 1))

    def test_does_not_raise_error_if_lower_lon_bound_is_larger_than_upper(self):
        self.ds.spatial.average("ts", axis=["X"], lon_bounds=(2, 1))

    def test_raises_error_if_lat_axis_is_specified_but_lat_is_not_in_weights_dims(
        self,
    ):
        weights = xr.DataArray(
            data=np.ones(4), coords={"lon": self.ds.lon}, dims=["lon"]
        )
        with pytest.raises(KeyError):
            self.ds.spatial.average("ts", axis=["X", "Y"], weights=weights)

    def test_raises_error_if_lon_axis_is_specified_but_lon_is_not_in_weights_dims(
        self,
    ):
        weights = xr.DataArray(
            data=np.ones(4), coords={"lat": self.ds.lat}, dims=["lat"]
        )
        with pytest.raises(KeyError):
            self.ds.spatial.average("ts", axis=["X", "Y"], weights=weights)

    def test_raises_error_if_weights_lat_and_lon_dims_dont_align_with_data_var_dims(
        self,
    ):
        # Get a slice of the dataset to reduce the size of the dimensions for
        # simpler testing.
        ds = self.ds.isel(lat=slice(0, 3), lon=slice(0, 3))
        weights = xr.DataArray(
            data=np.ones((3, 3)),
            coords={"lat": ds.lat, "lon": ds.lon},
            dims=["lat", "lon"],
        )

        with pytest.raises(ValueError):
            self.ds.spatial.average("ts", axis=["X", "Y"], weights=weights)

    def test_spatial_average_for_lat_region_and_keep_weights(self):
        ds = self.ds.copy()

        result = ds.spatial.average(
            "ts", axis=["Y"], lat_bounds=(-5.0, 5), keep_weights=True
        )

        expected = self.ds.copy()
        expected["ts"] = xr.DataArray(
            data=np.array(
                [[2.25, 2.25, 2.25, 2.25], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]
            ),
            coords={"time": expected.time, "lon": expected.lon},
            dims=["time", "lon"],
        )
        expected["lat_wts"] = xr.DataArray(
            name="lat_wts",
            data=np.array([0.0, 0.08715574, 0.08715574, 0.0]),
            dims=["lat"],
        )

        xr.testing.assert_allclose(result, expected)

    def test_spatial_average_for_lat_region(self):
        ds = self.ds.copy()

        # Specifying axis as a str instead of list of str.
        result = ds.spatial.average("ts", axis=["Y"], lat_bounds=(-5.0, 5))

        expected = self.ds.copy()
        expected["ts"] = xr.DataArray(
            data=np.array(
                [[2.25, 2.25, 2.25, 2.25], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]
            ),
            coords={"time": expected.time, "lon": expected.lon},
            dims=["time", "lon"],
        )

        assert result.identical(expected)

    @requires_dask
    def test_spatial_average_for_lat_region_and_keep_weights_with_dask(self):
        ds = self.ds.copy().chunk(2)

        # Specifying axis as a str instead of list of str.
        result = ds.spatial.average(
            "ts", axis=["Y"], lat_bounds=(-5.0, 5), keep_weights=True
        )

        expected = self.ds.copy()
        expected["ts"] = xr.DataArray(
            data=np.array(
                [[2.25, 2.25, 2.25, 2.25], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]
            ),
            coords={"time": expected.time, "lon": expected.lon},
            dims=["time", "lon"],
        )
        expected["lat_wts"] = xr.DataArray(
            name="lat_wts",
            data=np.array([0.0, 0.08715574, 0.08715574, 0.0]),
            dims=["lat"],
        )

        xr.testing.assert_allclose(result, expected)

    def test_spatial_average_for_lat_and_lon_region_and_keep_weights(self):
        ds = self.ds.copy()
        result = ds.spatial.average(
            "ts",
            axis=["X", "Y"],
            lat_bounds=(-5.0, 5),
            lon_bounds=(-170, -120.1),
            keep_weights=True,
        )

        expected = self.ds.copy()
        expected["ts"] = xr.DataArray(
            data=np.array([2.25, 1.0, 1.0]),
            coords={"time": expected.time},
            dims="time",
        )
        expected["lat_lon_wts"] = xr.DataArray(
            name="ts_weights",
            data=np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 4.34907156, 4.34907156, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            ),
            dims=["lon", "lat"],
        )

        xr.testing.assert_allclose(result, expected)

    def test_spatial_average_for_lat_and_lon_region_with_custom_weights(self):
        ds = self.ds.copy()

        weights = xr.DataArray(
            data=np.array([[1, 2, 3, 4], [2, 4, 6, 8], [3, 6, 9, 12], [4, 8, 12, 16]]),
            coords={"lat": ds.lat, "lon": ds.lon},
            dims=["lat", "lon"],
        )
        result = ds.spatial.average(
            axis=["X", "Y"],
            lat_bounds=(-5.0, 5),
            lon_bounds=(-170, -120.1),
            weights=weights,
            data_var="ts",
        )

        expected = self.ds.copy()
        expected["ts"] = xr.DataArray(
            data=np.array([2.25, 1.0, 1.0]),
            coords={"time": expected.time},
            dims="time",
        )

        assert result.identical(expected)


class TestGetWeights:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = generate_dataset(cf_compliant=True, has_bounds=True)

    def test_raises_error_if_domain_lower_bound_exceeds_upper_bound(self):
        ds = self.ds.copy()
        ds.lon_bnds.data[:] = np.array([[1, 0], [1, 2], [2, 3], [3, 4]])

        with pytest.raises(ValueError):
            ds.spatial.get_weights(axis=["X"])

    def test_weights_for_region_in_lat_and_lon_domains(self):
        result = self.ds.spatial.get_weights(
            axis=["Y", "X"], lat_bounds=(-5, 5), lon_bounds=(-170, -120)
        )
        expected = xr.DataArray(
            data=np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 4.35778714, 0.0],
                    [0.0, 0.0, 4.35778714, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            ),
            coords={"lat": self.ds.lat, "lon": self.ds.lon},
            dims=["lat", "lon"],
        )

        xr.testing.assert_allclose(result, expected)

    def test_area_weights_for_region_in_lat_domain(self):
        result = self.ds.spatial.get_weights(
            axis=["Y", "X"], lat_bounds=(-5, 5), lon_bounds=None
        )
        expected = xr.DataArray(
            data=np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.16341702, 15.52461668, 15.52461668, 0.16341702],
                    [0.16341702, 15.52461668, 15.52461668, 0.16341702],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            ),
            coords={"lat": self.ds.lat, "lon": self.ds.lon},
            dims=["lat", "lon"],
        )

        xr.testing.assert_allclose(result, expected)

    def test_weights_for_region_in_lon_domain(self):
        expected = xr.DataArray(
            data=np.array(
                [
                    [0.0, 0.0, 0.00297475, 0.0],
                    [0.0, 0.0, 49.99702525, 0.0],
                    [0.0, 0.0, 49.99702525, 0.0],
                    [0.0, 0.0, 0.00297475, 0.0],
                ]
            ),
            coords={"lat": self.ds.lat, "lon": self.ds.lon},
            dims=["lat", "lon"],
        )
        result = self.ds.spatial.get_weights(
            axis=["Y", "X"], lat_bounds=None, lon_bounds=(-170, -120)
        )

        xr.testing.assert_allclose(result, expected)

    def test_weights_for_region_in_lon_domain_with_region_spanning_p_meridian(self):
        ds = self.ds.copy()

        result = ds.spatial._get_longitude_weights(
            domain_bounds=ds.lon_bnds,
            # Region spans prime meridian.
            region_bounds=np.array([359, 1]),
        )
        expected = xr.DataArray(
            data=np.array([1.875, 0.0625, 0.0, 0.0625]),
            coords={"lon": ds.lon},
            dims=["lon"],
        )

        xr.testing.assert_allclose(result, expected)

    def test_weights_all_longitudes_for_equal_region_bounds(self):
        expected = xr.DataArray(
            data=np.array(
                [1.875, 178.125, 178.125, 1.875],
            ),
            coords={"lon": self.ds.lon},
            dims=["lon"],
        )
        result = self.ds.spatial.get_weights(
            axis=["X"],
            lat_bounds=None,
            lon_bounds=np.array([0.0, 360.0]),
        )

        xr.testing.assert_allclose(result, expected)

    def test_weights_for_equal_region_bounds_representing_entire_lon_domain(self):
        expected = xr.DataArray(
            data=np.array(
                [1.875, 178.125, 178.125, 1.875],
            ),
            coords={"lon": self.ds.lon},
            dims=["lon"],
        )
        result = self.ds.spatial.get_weights(
            axis=["X"], lat_bounds=None, lon_bounds=np.array([10.0, 10.0])
        )

        xr.testing.assert_allclose(result, expected)


class Test_SwapLonAxis:
    # NOTE: This private method is tested because we might want to consider
    # converting it to a public method in the future.
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = generate_dataset(cf_compliant=True, has_bounds=True)

    def test_raises_error_with_incorrect_orientation_to_swap_to(self):
        domain = xr.DataArray(
            name="lon_bnds",
            data=np.array([[-65, -5], [-5, 0], [0, 120]]),
            dims=["lon", "bnds"],
            attrs={"is_generated": "True"},
        )
        with pytest.raises(ValueError):
            self.ds.spatial._swap_lon_axis(domain, to=9000)

    @requires_dask
    def test_swap_chunked_domain_dataarray_from_180_to_360(self):
        domain = xr.DataArray(
            name="lon_bnds",
            data=np.array([[-65, -5], [-5, 0], [0, 120]]),
            dims=["lon", "bnds"],
            attrs={"is_generated": "True"},
        ).chunk(2)

        result = self.ds.spatial._swap_lon_axis(domain, to=360)
        expected = xr.DataArray(
            name="lon_bnds",
            data=np.array([[295, 355], [355, 0], [0, 120]]),
            dims=["lon", "bnds"],
            attrs={"is_generated": "True"},
        )

        assert result.identical(expected)

    @requires_dask
    def test_swap_chunked_domain_dataarray_from_360_to_180(self):
        domain = xr.DataArray(
            name="lon_bnds",
            data=np.array([[0, 120], [120, 181], [181, 360]]),
            dims=["lon", "bnds"],
            attrs={"is_generated": "True"},
        ).chunk(2)

        result = self.ds.spatial._swap_lon_axis(domain, to=180)
        expected = xr.DataArray(
            name="lon_bnds",
            data=np.array([[0, 120], [120, -179], [-179, 0]]),
            dims=["lon", "bnds"],
            attrs={"is_generated": "True"},
        )

        assert result.identical(expected)

        domain = xr.DataArray(
            name="lon_bnds",
            data=np.array([[-0.25, 120], [120, 359.75]]),
            dims=["lon", "bnds"],
            attrs={"is_generated": "True"},
        ).chunk(2)

        result = self.ds.spatial._swap_lon_axis(domain, to=180)
        expected = xr.DataArray(
            name="lon_bnds",
            data=np.array([[-0.25, 120], [120, -0.25]]),
            dims=["lon", "bnds"],
            attrs={"is_generated": "True"},
        )

        assert result.identical(expected)

    def test_swap_domain_dataarray_from_180_to_360(self):
        domain = xr.DataArray(
            name="lon_bnds",
            data=np.array([[-65, -5], [-5, 0], [0, 120]]),
            dims=["lon", "bnds"],
            attrs={"is_generated": "True"},
        )

        result = self.ds.spatial._swap_lon_axis(domain, to=360)
        expected = xr.DataArray(
            name="lon_bnds",
            data=np.array([[295, 355], [355, 0], [0, 120]]),
            dims=["lon", "bnds"],
            attrs={"is_generated": "True"},
        )

        assert result.identical(expected)

    def test_swap_domain_dataarray_from_360_to_180(self):
        domain = xr.DataArray(
            name="lon_bnds",
            data=np.array([[0, 120], [120, 181], [181, 360]]),
            dims=["lon", "bnds"],
            attrs={"is_generated": "True"},
        )

        result = self.ds.spatial._swap_lon_axis(domain, to=180)
        expected = xr.DataArray(
            name="lon_bnds",
            data=np.array([[0, 120], [120, -179], [-179, 0]]),
            dims=["lon", "bnds"],
            attrs={"is_generated": "True"},
        )

        assert result.identical(expected)

        domain = xr.DataArray(
            name="lon_bnds",
            data=np.array([[-0.25, 120], [120, 359.75]]),
            dims=["lon", "bnds"],
            attrs={"is_generated": "True"},
        )

        result = self.ds.spatial._swap_lon_axis(domain, to=180)
        expected = xr.DataArray(
            name="lon_bnds",
            data=np.array([[-0.25, 120], [120, -0.25]]),
            dims=["lon", "bnds"],
            attrs={"is_generated": "True"},
        )

        assert result.identical(expected)

    def test_swap_region_ndarray_from_180_to_360(self):
        result = self.ds.spatial._swap_lon_axis(np.array([-65, 0, 120]), to=360)
        expected = np.array([295, 0, 120])

        assert np.array_equal(result, expected)

        result = self.ds.spatial._swap_lon_axis(np.array([-180, 0, 180]), to=360)
        expected = np.array([180, 0, 180])

        assert np.array_equal(result, expected)

    def test_swap_region_ndarray_from_360_to_180(self):
        result = self.ds.spatial._swap_lon_axis(np.array([0, 120, 181, 360]), to=180)
        expected = np.array([0, 120, -179, 0])

        assert np.array_equal(result, expected)

        result = self.ds.spatial._swap_lon_axis(np.array([-0.25, 120, 359.75]), to=180)
        expected = np.array([-0.25, 120, -0.25])

        assert np.array_equal(result, expected)


class Test_ScaleDimToRegion:
    # NOTE: This private method is tested because this is an in-house algorithm
    # that has edge cases with some complexities.
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = generate_dataset(cf_compliant=True, has_bounds=True)

    @requires_dask
    def test_scales_chunked_lat_bounds_when_not_wrapping_around_prime_meridian(self):
        domain_bounds = xr.DataArray(
            name="lat_bnds",
            data=np.array(
                [[-90, -89.375], [-89.375, 0.0], [0.0, 89.375], [89.375, 90]]
            ),
            coords={"lat": self.ds.lat},
            dims=["lat", "bnds"],
        ).chunk(2)

        result = self.ds.spatial._scale_domain_to_region(
            domain_bounds=domain_bounds, region_bounds=np.array([-5, 5])
        )
        expected = xr.DataArray(
            name="lat_bnds",
            data=np.array([[-5.0, -5.0], [-5.0, 0.0], [0.0, 5.0], [5.0, 5.0]]),
            coords={"lat": self.ds.lat},
            dims=["lat", "bnds"],
        )

        assert result.identical(expected)

    @requires_dask
    def test_scales_chunked_lon_bounds_when_not_wrapping_around_prime_meridian(self):
        domain_bounds = xr.DataArray(
            name="lon_bnds",
            data=np.array(
                [
                    [359.0625, 360.9375],
                    [0.9375, 179.0625],
                    [179.0625, 357.1875],
                    [357.1875, 359.0625],
                ]
            ),
            coords={"lat": self.ds.lat},
            dims=["lat", "bnds"],
        ).chunk(2)

        result = self.ds.spatial._scale_domain_to_region(
            domain_bounds=domain_bounds, region_bounds=np.array([190, 240])
        )
        expected = xr.DataArray(
            name="lon_bnds",
            data=np.array(
                [[240.0, 240.0], [190.0, 190.0], [190.0, 240.0], [240.0, 240.0]]
            ),
            coords={"lat": self.ds.lat},
            dims=["lat", "bnds"],
        )
        assert result.identical(expected)

    def test_scales_lat_bounds_when_not_wrapping_around_prime_meridian(self):
        domain_bounds = xr.DataArray(
            name="lat_bnds",
            data=np.array(
                [[-90, -89.375], [-89.375, 0.0], [0.0, 89.375], [89.375, 90]]
            ),
            coords={"lat": self.ds.lat},
            dims=["lat", "bnds"],
        )

        result = self.ds.spatial._scale_domain_to_region(
            domain_bounds=domain_bounds, region_bounds=np.array([-5, 5])
        )
        expected = xr.DataArray(
            name="lat_bnds",
            data=np.array([[-5.0, -5.0], [-5.0, 0.0], [0.0, 5.0], [5.0, 5.0]]),
            coords={"lat": self.ds.lat},
            dims=["lat", "bnds"],
        )

        assert result.identical(expected)

    def test_scales_lon_bounds_when_not_wrapping_around_prime_meridian(self):
        domain_bounds = xr.DataArray(
            name="lon_bnds",
            data=np.array(
                [
                    [359.0625, 360.9375],
                    [0.9375, 179.0625],
                    [179.0625, 357.1875],
                    [357.1875, 359.0625],
                ]
            ),
            coords={"lat": self.ds.lat},
            dims=["lat", "bnds"],
        )

        result = self.ds.spatial._scale_domain_to_region(
            domain_bounds=domain_bounds, region_bounds=np.array([190, 240])
        )
        expected = xr.DataArray(
            name="lon_bnds",
            data=np.array(
                [[240.0, 240.0], [190.0, 190.0], [190.0, 240.0], [240.0, 240.0]]
            ),
            coords={"lat": self.ds.lat},
            dims=["lat", "bnds"],
        )

        assert result.identical(expected)

    def test_scales_lon_bounds_when_wrapping_around_prime_meridian(self):
        domain_bounds = xr.DataArray(
            name="lon_bnds",
            data=np.array(
                [
                    # Does not apply to any conditional.
                    [359.0625, 360.9375],
                    # Grid cells stradling upper boundary.
                    [0.9375, 179.0625],
                    # Grid cells in between boundaries.
                    [179.0625, 357.1875],
                    # Grid cell straddling lower boundary.
                    [357.1875, 359.0625],
                ]
            ),
            coords={"lat": self.ds.lat},
            dims=["lat", "bnds"],
        )

        result = self.ds.spatial._scale_domain_to_region(
            domain_bounds=domain_bounds, region_bounds=np.array([357.5, 10.0])
        )
        expected = xr.DataArray(
            name="lon_bnds",
            data=np.array(
                [
                    # Does not apply to any conditional.
                    [359.0625, 360.9375],
                    # Grid cells stradling upper boundary.
                    [0.9375, 10.0],
                    # Grid cells in between boundaries.
                    [10.0, 10.0],
                    # Grid cell straddling lower boundary.
                    [357.5, 359.0625],
                ]
            ),
            coords={"lat": self.ds.lat},
            dims=["lat", "bnds"],
        )

        assert result.identical(expected)
