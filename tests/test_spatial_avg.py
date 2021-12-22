import numpy as np
import pytest
import xarray as xr

from tests import requires_dask
from tests.fixtures import generate_dataset
from xcdat.spatial_avg import SpatialAverageAccessor


class TestSpatialAverageAcccessor:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = generate_dataset(cf_compliant=True, has_bounds=True)

    def test__init__(self):
        ds = self.ds.copy()
        obj = SpatialAverageAccessor(ds)

        assert obj._dataset.identical(ds)

    def test_decorator_call(self):
        ds = self.ds.copy()
        obj = ds.spatial

        assert obj._dataset.identical(ds)


class TestSpatialAvg:
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
            self.ds.spatial.spatial_avg(
                "not_a_data_var",
                axis=["lat", "incorrect_axis"],
            )

    def test_weighted_spatial_average_for_lat_and_lon_region_for_an_inferred_data_var(
        self,
    ):
        ds = self.ds.copy()
        ds.attrs["xcdat_infer"] = "ts"

        # `data_var` kwarg is not specified, so an inference is attempted
        result = ds.spatial.spatial_avg(
            axis=["lat", "lon"], lat_bounds=(-5.0, 5), lon_bounds=(-170, -120.1)
        )

        expected = self.ds.copy()
        expected.attrs["xcdat_infer"] = "ts"
        expected["ts"] = xr.DataArray(
            data=np.array([2.25, 1.0, 1.0]),
            coords={"time": expected.time},
            dims="time",
        )

        assert result.identical(expected)

    def test_weighted_spatial_average_for_lat_and_lon_region_for_explicit_data_var(
        self,
    ):
        ds = self.ds.copy()
        result = ds.spatial.spatial_avg(
            "ts", axis=["lat", "lon"], lat_bounds=(-5.0, 5), lon_bounds=(-170, -120.1)
        )

        expected = self.ds.copy()
        expected["ts"] = xr.DataArray(
            data=np.array([2.25, 1.0, 1.0]),
            coords={"time": expected.time},
            dims="time",
        )

        assert result.identical(expected)

    def test_weighted_spatial_average_for_lat_region(self):
        ds = self.ds.copy()

        # Specifying axis as a str instead of list of str.
        result = ds.spatial.spatial_avg(
            "ts", axis="lat", lat_bounds=(-5.0, 5), lon_bounds=(-170, -120.1)
        )

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
    def test_chunked_weighted_spatial_average_for_lat_region(self):
        ds = self.ds.copy().chunk(2)

        # Specifying axis as a str instead of list of str.
        result = ds.spatial.spatial_avg(
            "ts", axis="lat", lat_bounds=(-5.0, 5), lon_bounds=(-170, -120.1)
        )

        expected = self.ds.copy()
        expected["ts"] = xr.DataArray(
            data=np.array(
                [[2.25, 2.25, 2.25, 2.25], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]
            ),
            coords={"time": expected.time, "lon": expected.lon},
            dims=["time", "lon"],
        )

        assert result.identical(expected)


class TestValidateAxis:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = generate_dataset(cf_compliant=True, has_bounds=True)

    def test_raises_error_if_axis_list_contains_unsupported_axis(self):
        with pytest.raises(ValueError):
            self.ds.spatial._validate_axis(self.ds.ts, axis=["lat", "incorrect_axis"])

    def test_raises_error_if_lat_axis_does_not_exist(self):
        ds = self.ds.copy()
        ds["ts"] = xr.DataArray(data=None, coords={"lon": ds.lon}, dims=["lon"])
        with pytest.raises(KeyError):
            ds.spatial._validate_axis(ds.ts, axis=["lat", "lon"])

    def test_raises_error_if_lon_axis_does_not_exist(self):
        ds = self.ds.copy()
        ds["ts"] = xr.DataArray(data=None, coords={"lat": ds.lat}, dims=["lat"])
        with pytest.raises(KeyError):
            ds.spatial._validate_axis(ds.ts, axis=["lat", "lon"])

    def test_returns_list_of_str_if_axis_is_a_single_supported_str_input(self):
        result = self.ds.spatial._validate_axis(self.ds.ts, axis="lat")
        expected = ["lat"]

        assert result == expected


class TestValidateRegionBounds:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = generate_dataset(cf_compliant=True, has_bounds=True)

    def test_raises_error_if_bounds_type_is_not_a_tuple(self):
        with pytest.raises(TypeError):
            self.ds.spatial._validate_region_bounds("lon", [1, 1])

        with pytest.raises(TypeError):
            self.ds.spatial._validate_region_bounds("lon", "str")

    def test_raises_error_if_there_are_0_elements_in_the_bounds(self):
        with pytest.raises(ValueError):
            self.ds.spatial._validate_region_bounds("lon", ())

    def test_raises_error_if_there_are_more_than_two_elements_in_the_bounds(self):
        with pytest.raises(ValueError):
            self.ds.spatial._validate_region_bounds("lon", (1, 1, 2))

    def test_does_not_raise_error_if_lower_and_upper_bounds_are_floats_or_ints(self):
        self.ds.spatial._validate_region_bounds("lon", (1, 1))
        self.ds.spatial._validate_region_bounds("lon", (1, 1.2))

    def test_raises_error_if_lower_bound_is_not_a_float_or_int(self):
        with pytest.raises(TypeError):
            self.ds.spatial._validate_region_bounds("lat", ("invalid", 1))

    def test_raises_error_if_upper_bound_is_not_a_float_or_int(self):
        with pytest.raises(TypeError):
            self.ds.spatial._validate_region_bounds("lon", (1, "invalid"))

    def test_raises_error_if_lower_lat_bound_is_bigger_than_upper(self):
        with pytest.raises(ValueError):
            self.ds.spatial._validate_region_bounds("lat", (2, 1))

    def test_does_not_raise_error_if_lon_lower_bound_is_larger_than_upper(self):
        self.ds.spatial._validate_region_bounds("lon", (2, 1))


class TestValidateWeights:
    def setup(self):
        self.ds = generate_dataset(cf_compliant=True, has_bounds=True)
        self.weights = xr.DataArray(
            data=np.ones((4, 4)),
            coords={"lat": self.ds.lat, "lon": self.ds.lon},
            dims=["lat", "lon"],
        )

    def test_no_error_is_raised_when_spatial_dim_sizes_align_between_weights_and_data_var(
        self,
    ):
        weights = xr.DataArray(
            data=np.ones((4, 4)),
            coords={"lat": self.ds.lat, "lon": self.ds.lon},
            dims=["lat", "lon"],
        )
        self.ds.spatial._validate_weights(self.ds["ts"], axis="lat", weights=weights)

    def test_error_is_raised_when_lat_axis_is_specified_but_lat_is_not_in_weights_dims(
        self,
    ):
        weights = xr.DataArray(
            data=np.ones(4), coords={"lon": self.ds.lon}, dims=["lon"]
        )
        with pytest.raises(KeyError):
            self.ds.spatial._validate_weights(
                self.ds["ts"], axis=["lon", "lat"], weights=weights
            )

    def test_error_is_raised_when_lon_axis_is_specified_but_lon_is_not_in_weights_dims(
        self,
    ):
        weights = xr.DataArray(
            data=np.ones(4), coords={"lat": self.ds.lat}, dims=["lat"]
        )
        with pytest.raises(KeyError):
            self.ds.spatial._validate_weights(
                self.ds["ts"], axis=["lon", "lat"], weights=weights
            )

    def test_error_is_raised_when_weights_lat_and_lon_dims_dont_align_with_data_var_dims(
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
            self.ds.spatial._validate_weights(
                self.ds["ts"], axis=["lat", "lon"], weights=weights
            )


class TestSwapLonAxis:
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


class TestGetWeights:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = generate_dataset(cf_compliant=True, has_bounds=True)

    def test_weights_for_region_in_lat_and_lon_domains(self):
        result = self.ds.spatial._get_weights(
            axis=["lat", "lon"], lat_bounds=(-5, 5), lon_bounds=(-170, -120)
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
        result = self.ds.spatial._get_weights(
            axis=["lat", "lon"], lat_bounds=(-5, 5), lon_bounds=None
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
        result = self.ds.spatial._get_weights(
            axis=["lat", "lon"], lat_bounds=None, lon_bounds=(-170, -120)
        )

        xr.testing.assert_allclose(result, expected)


class TestGetLongitudeWeights:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = generate_dataset(cf_compliant=True, has_bounds=True)

    def test_weights_for_region_in_lon_domain(self):
        # Longitude axis orientation swaps from (-180, 180) to (0, 360).
        result = self.ds.spatial._get_longitude_weights(
            domain_bounds=self.ds.lon_bnds.copy(),
            region_bounds=np.array([-170.0, -120.0]),
        )
        expected = xr.DataArray(
            data=np.array([0.0, 0.0, 50.0, 0.0]),
            coords={"lon": self.ds.lon},
            dims=["lon"],
        )

        xr.testing.assert_allclose(result, expected)

    def test_weights_for_region_in_lon_domain_with_both_spanning_p_meridian(self):
        ds = self.ds.copy()
        # Domain spans prime meridian.
        ds.lon_bnds.data[:] = np.array([[359, 1], [1, 90], [90, 180], [180, 359]])

        result = ds.spatial._get_longitude_weights(
            domain_bounds=ds.lon_bnds,
            # Region spans prime meridian.
            region_bounds=np.array([359, 1]),
        )
        expected = xr.DataArray(
            data=np.array([2.0, 0.0, 0.0, 0.0]),
            coords={"lon": ds.lon},
            dims=["lon"],
        )

        xr.testing.assert_allclose(result, expected)

    def test_weights_for_region_in_lon_domain_with_domain_spanning_p_meridian(self):
        ds = self.ds.copy()
        # Domain spans prime meridian.
        ds.lon_bnds.data[:] = np.array([[359, 1], [1, 90], [90, 180], [180, 359]])

        # Longitude axis orientation swaps from (-180, 180) to (0, 360).
        result = ds.spatial._get_longitude_weights(
            domain_bounds=ds.lon_bnds,
            region_bounds=np.array([-170.0, -120.0]),
        )
        expected = xr.DataArray(
            data=np.array([0.0, 0.0, 0.0, 50.0]),
            coords={"lon": ds.lon},
            dims=["lon"],
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
        result = self.ds.spatial._get_longitude_weights(
            domain_bounds=self.ds.lon_bnds.copy(),
            region_bounds=np.array([0.0, 360.0]),
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
        result = self.ds.spatial._get_longitude_weights(
            domain_bounds=self.ds.lon_bnds.copy(), region_bounds=np.array([10.0, 10.0])
        )

        xr.testing.assert_allclose(result, expected)


class TestGetLatitudeWeights:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = generate_dataset(cf_compliant=True, has_bounds=True)

    def test_weights_for_region_in_lat_domain(self):
        expected = xr.DataArray(
            data=np.array([0.0, 0.087156, 0.087156, 0.0]),
            coords={"lat": self.ds.lat},
            dims=["lat"],
        )
        result = self.ds.spatial._get_latitude_weights(
            domain_bounds=self.ds.lat_bnds, region_bounds=np.array([-5.0, 5.0])
        )

        xr.testing.assert_allclose(result, expected)


class TestValidateDomainBounds:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = generate_dataset(cf_compliant=True, has_bounds=True)

    def test_raises_error_if_low_bounds_exceeds_high_bound(self):
        domain_bounds = xr.DataArray(
            name="lon_bnds",
            data=np.array([[1, 0], [1, 2], [2, 3], [3, 4]]),
            dims=["lon", "bnds"],
        )
        with pytest.raises(ValueError):
            self.ds.spatial._validate_domain_bounds(domain_bounds)


class TestAlignLongitudeto360Axis:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = generate_dataset(cf_compliant=True, has_bounds=True)

    def test_raises_error_if_bounds_below_0(self):
        domain_bounds = xr.DataArray(
            name="lon_bnds",
            data=np.array([[-1, 1], [1, 90], [90, 180], [180, 359]]),
            dims=["lon", "bnds"],
        )
        with pytest.raises(ValueError):
            self.ds.spatial._align_longitude_to_360_axis(domain_bounds)

    def test_raises_error_if_bounds_above_360(self):
        domain_bounds = xr.DataArray(
            name="lon_bnds",
            data=np.array([[359, 361], [1, 90], [90, 180], [180, 359]]),
            dims=["lon", "bnds"],
        )
        with pytest.raises(ValueError):
            self.ds.spatial._align_longitude_to_360_axis(domain_bounds)

    def test_raises_error_if_multiple_bounds_span_prime_meridian(self):
        domain_bounds = xr.DataArray(
            name="lon_bnds",
            data=np.array([[359, 1], [1, 90], [90, 180], [180, 2]]),
            dims=["lon", "bnds"],
        )
        with pytest.raises(ValueError):
            self.ds.spatial._align_longitude_to_360_axis(domain_bounds)

    def test_extends_bounds_array_for_cell_spanning_prime_meridian(self):
        domain_bounds = xr.DataArray(
            name="lon_bnds",
            data=np.array([[359, 1], [1, 90], [90, 180], [180, 359]]),
            dims=["lon", "bnds"],
        )
        expected_index = 0

        expected_bounds = xr.DataArray(
            name="lon_bnds",
            data=np.array([[0, 1], [1, 90], [90, 180], [180, 359], [359, 360]]),
            dims=["lon", "bnds"],
        )

        result_bounds, result_index = self.ds.spatial._align_longitude_to_360_axis(
            domain_bounds
        )

        assert (result_bounds.identical(expected_bounds)) & (
            result_index == expected_index
        )

    def test_returns_original_array_if_no_cell_spans_prime_meridian(self):
        domain_bounds = xr.DataArray(
            name="lon_bnds",
            data=np.array([[0, 1], [1, 90], [90, 180], [180, 360]]),
            dims=["lon", "bnds"],
        )

        result_bounds, result_index = self.ds.spatial._align_longitude_to_360_axis(
            domain_bounds
        )

        assert result_bounds.identical(domain_bounds)
        assert result_index is None

    def test_retains_total_weight(self):
        # construct array spanning 0 to 360
        domain_bounds = xr.DataArray(
            name="lon_bnds",
            data=np.array([[359, 1], [1, 90], [90, 180], [180, 359]]),
            dims=["lon", "bnds"],
        )

        result_bounds, null = self.ds.spatial._align_longitude_to_360_axis(
            domain_bounds
        )

        dbdiff = np.sum(np.array(result_bounds[:, 1] - result_bounds[:, 0]))

        assert dbdiff == 360.0


class TestCalculateWeights:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = generate_dataset(cf_compliant=True, has_bounds=True)

    def test_returns_weights_as_the_absolute_difference_of_upper_and_lower_bounds(self):
        lat = xr.DataArray(
            name="lat",
            data=np.array([-90.0, -88.75, 88.75, 90.0]),
            coords={"lat": np.array([-90.0, -88.75, 88.75, 90.0])},
            dims=["lat"],
        )
        lat_bounds = xr.DataArray(
            data=np.array(
                [[-90.0, -89.375], [-89.375, 0.0], [0.0, 89.375], [89.375, 90.0]]
            ),
            coords={"lat": lat},
            dims=["lat", "bnds"],
        )

        result = self.ds.spatial._calculate_weights(lat_bounds)
        expected = xr.DataArray(
            data=np.array([0.625, 89.375, 89.375, 0.625]),
            coords={"lat": lat},
            dims=["lat"],
        )
        assert result.identical(expected)


class TestScaleDimToRegion:
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


class TestCombineWeights:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = generate_dataset(cf_compliant=True, has_bounds=True)
        self.axis_weights = {
            "lat": xr.DataArray(
                name="lat_wts",
                data=np.array([1, 2, 3, 4]),
                coords={"lat": self.ds.lat},
                dims=["lat"],
            ),
            "lon": xr.DataArray(
                name="lon_wts",
                data=np.array([1, 2, 3, 4]),
                coords={"lon": self.ds.lon},
                dims=["lon"],
            ),
        }

    def test_weights_for_single_axis_are_identical(self):
        axis_weights = self.axis_weights
        del axis_weights["lon"]

        result = self.ds.spatial._combine_weights(axis_weights=self.axis_weights)
        expected = self.axis_weights["lat"]

        assert result.identical(expected)

    def test_weights_for_multiple_axis_is_the_product_of_matrix_multiplication(self):
        result = self.ds.spatial._combine_weights(axis_weights=self.axis_weights)
        expected = xr.DataArray(
            data=np.array([[1, 2, 3, 4], [2, 4, 6, 8], [3, 6, 9, 12], [4, 8, 12, 16]]),
            coords={"lat": self.ds.lat, "lon": self.ds.lon},
            dims=["lat", "lon"],
        )

        assert result.identical(expected)


class TestAverager:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = generate_dataset(cf_compliant=True, has_bounds=True)

    @requires_dask
    def test_chunked_weighted_avg_over_lat_and_lon_axes(self):
        ds = self.ds.copy().chunk(2)

        weights = xr.DataArray(
            data=np.array([[1, 2, 3, 4], [2, 4, 6, 8], [3, 6, 9, 12], [4, 8, 12, 16]]),
            coords={"lat": ds.lat, "lon": ds.lon},
            dims=["lat", "lon"],
        )

        result = ds.spatial._averager(ds.ts, axis=["lat", "lon"], weights=weights)
        expected = xr.DataArray(
            name="ts", data=np.ones(12), coords={"time": ds.time}, dims=["time"]
        )

        assert result.identical(expected)

    def test_weighted_avg_over_lat_axis(self):
        weights = xr.DataArray(
            name="lat_wts",
            data=np.array([1, 2, 3, 4]),
            coords={"lat": self.ds.lat},
            dims=["lat"],
        )

        result = self.ds.spatial._averager(self.ds.ts, axis=["lat"], weights=weights)
        expected = xr.DataArray(
            name="ts",
            data=np.ones((12, 4)),
            coords={"time": self.ds.time, "lon": self.ds.lon},
            dims=["time", "lon"],
        )

        assert result.identical(expected)

    def test_weighted_avg_over_lon_axis(self):
        weights = xr.DataArray(
            name="lon_wts",
            data=np.array([1, 2, 3, 4]),
            coords={"lon": self.ds.lon},
            dims=["lon"],
        )

        result = self.ds.spatial._averager(self.ds.ts, axis=["lon"], weights=weights)
        expected = xr.DataArray(
            name="ts",
            data=np.ones((12, 4)),
            coords={"time": self.ds.time, "lat": self.ds.lat},
            dims=["time", "lat"],
        )

        assert result.identical(expected)

    def test_weighted_avg_over_lat_and_lon_axis(self):
        weights = xr.DataArray(
            data=np.array([[1, 2, 3, 4], [2, 4, 6, 8], [3, 6, 9, 12], [4, 8, 12, 16]]),
            coords={"lat": self.ds.lat, "lon": self.ds.lon},
            dims=["lat", "lon"],
        )

        result = self.ds.spatial._averager(
            self.ds.ts, axis=["lat", "lon"], weights=weights
        )
        expected = xr.DataArray(
            name="ts", data=np.ones(12), coords={"time": self.ds.time}, dims=["time"]
        )

        assert result.identical(expected)


class TestGetGenericAxisKeys:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = generate_dataset(cf_compliant=True, has_bounds=True)

    def test_generic_keys(self):
        result = self.ds.spatial._get_generic_axis_keys(["lat", "lon"])
        expected = ["Y", "X"]
        assert result == expected
