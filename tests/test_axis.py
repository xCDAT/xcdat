import numpy as np
import pytest
import xarray as xr

from tests.fixtures import generate_dataset
from xcdat.axis import (
    _align_lon_bounds_to_360,
    _get_prime_meridian_index,
    swap_lon_axis,
)


class TestSwapLonAxis:
    def test_raises_error_with_incorrect_lon_orientation_for_swapping(self):
        ds = generate_dataset(cf_compliant=True, has_bounds=True)
        with pytest.raises(ValueError):
            swap_lon_axis(ds, to=9000)  # type: ignore

    def test_swap_from_180_to_360_and_sorts_with_prime_meridian_cell(self):
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
                    attrs={"is_generated": "True"},
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
                    attrs={"is_generated": "True"},
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
                    attrs={"is_generated": "True"},
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
                    attrs={"is_generated": "True"},
                )
            },
        )

        assert result.identical(expected)


class TestAlignLonBoundsto360:
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
            _align_lon_bounds_to_360(domain_bounds, np.array([0]))

    def test_raises_error_if_bounds_above_360(self):
        domain_bounds = xr.DataArray(
            name="lon_bnds",
            data=np.array([[359, 361], [1, 90], [90, 180], [180, 359]]),
            dims=["lon", "bnds"],
        )
        with pytest.raises(ValueError):
            _align_lon_bounds_to_360(domain_bounds, np.array([0]))

    def test_extends_bounds_array_for_cell_spanning_prime_meridian(self):
        domain_bounds = xr.DataArray(
            name="lon_bnds",
            coords={
                "lon": xr.DataArray(
                    name="lon",
                    data=np.array([0, 90, 180, 359]),
                    dims=["lon"],
                    attrs={"axis": "X"},
                )
            },
            data=np.array([[359, 1], [1, 90], [90, 180], [180, 359]]),
            dims=["lon", "bnds"],
        )

        result_bounds = _align_lon_bounds_to_360(domain_bounds, np.array([0]))
        expected_bounds = xr.DataArray(
            name="lon_bnds",
            coords={
                "lon": xr.DataArray(
                    name="lon",
                    data=np.array([0, 90, 180, 359, 0]),
                    dims=["lon"],
                    attrs={"axis": "X"},
                )
            },
            data=np.array([[0, 1], [1, 90], [90, 180], [180, 359], [359, 360]]),
            dims=["lon", "bnds"],
        )
        assert result_bounds.identical(expected_bounds)

    def test_retains_total_weight(self):
        # construct array spanning 0 to 360
        domain_bounds = xr.DataArray(
            name="lon_bnds",
            coords={
                "lon": xr.DataArray(
                    name="lon",
                    data=np.array([0, 90, 180, 359]),
                    dims=["lon"],
                    attrs={"axis": "X"},
                )
            },
            data=np.array([[359, 1], [1, 90], [90, 180], [180, 359]]),
            dims=["lon", "bnds"],
        )

        result_bounds = _align_lon_bounds_to_360(domain_bounds, np.array(0))
        dbdiff = np.sum(np.array(result_bounds[:, 1] - result_bounds[:, 0]))
        assert dbdiff == 360.0


class TestGetPrimeMeridianIndex:
    def test_raises_error_if_multiple_bounds_span_prime_meridian(self):
        domain_bounds = xr.DataArray(
            name="lon_bnds",
            data=np.array([[359, 1], [1, 90], [90, 180], [180, 2]]),
            dims=["lon", "bnds"],
        )
        with pytest.raises(ValueError):
            _get_prime_meridian_index(domain_bounds)

    def test_returns_none_if_there_is_no_prime_meridian(self):
        domain_bounds = xr.DataArray(
            name="lon_bnds",
            data=np.array([[0, 1], [1, 90], [90, 180], [180, 360]]),
            dims=["lon", "bnds"],
        )
        result = _get_prime_meridian_index(domain_bounds)

        assert result is None
