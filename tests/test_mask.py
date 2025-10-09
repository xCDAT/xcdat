import sys
from unittest import mock

import numpy as np
import pytest
import xarray as xr

from tests import fixtures
from xcdat import mask
from xcdat.regridder import grid

np.set_printoptions(threshold=sys.maxsize, suppress=True)

expected_land = [
    [np.nan, np.nan, np.nan, np.nan],
    [np.nan, np.nan, np.nan, np.nan],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
]

expected_sea = [
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [np.nan, np.nan, np.nan, np.nan],
    [np.nan, np.nan, np.nan, np.nan],
]


@pytest.fixture(scope="function")
def mask_da():
    return xr.DataArray(
        [
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
        ],
        dims=["lat", "lon"],
    )


@pytest.fixture(scope="function")
def source_da():
    return xr.DataArray(
        [
            [1, 0.4, 0, 0],
            [0.5, 0.2, 0.4, 0.6],
            [0, 0.2, 0.8, 1],
            [0, 0.1, 1, 1],
        ],
        dims=["lat", "lon"],
    )


@pytest.fixture(scope="function")
def diff_da():
    return xr.DataArray(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dims=["lat", "lon"],
    )


@pytest.fixture(scope="function")
def ds():
    return fixtures.generate_dataset(True, True, True)


class TestMask:
    def test_mask_land(self, ds):
        expected = xr.DataArray(
            expected_land,
            dims=("lat", "lon"),
            coords={
                "lat": ds.lat.copy(),
                "lon": ds.lon.copy(),
                "time": ds.time[0].copy(),
            },
        )

        output = ds.isel(time=0).spatial.mask_land("ts")

        xr.testing.assert_allclose(output.ts, expected)

    def test_mask_sea(self, ds):
        expected = xr.DataArray(
            expected_sea,
            dims=("lat", "lon"),
            coords={
                "lat": ds.lat.copy(),
                "lon": ds.lon.copy(),
                "time": ds.time[0].copy(),
            },
        )

        output = ds.isel(time=0).spatial.mask_sea("ts")

        xr.testing.assert_allclose(output.ts, expected)

    def test_generate_land_sea_mask(self, ds):
        expected = xr.DataArray(
            [
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dims=("lat", "lon"),
            coords={"lat": ds.lat.copy(), "lon": ds.lon.copy()},
        )

        output = ds.spatial.generate_land_sea_mask("ts")

        xr.testing.assert_allclose(output, expected)

    def test_generate_land_sea_mask_from_grid(self):
        ds = grid.create_uniform_grid(-90, 90, 36, 0, 359, 32)

        expected = xr.DataArray(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
                [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dims=("lat", "lon"),
            coords={"lat": ds.lat.copy(), "lon": ds.lon.copy()},
        )

        output = ds.spatial.generate_land_sea_mask()

        xr.testing.assert_allclose(output, expected)

    def test_generate_land_sea_mask_missing_coordinate(self):
        ds = grid.create_grid(x=grid.create_axis("lat", [x for x in range(10)]))

        with pytest.raises(
            KeyError,
            match="Dataset is missing a required coordinate, ensure a lat and lon coordinate exist",
        ):
            ds.spatial.generate_land_sea_mask()


class TestMaskGeneration:
    def test_mask_invalid_data_var(self, ds):
        with pytest.raises(KeyError):
            mask.generate_and_apply_land_sea_mask(ds, "tas")

    def test_mask_invalid_keep(self, ds):
        with pytest.raises(
            ValueError,
            match=r"Keep value 'artic' is not valid, options are 'land, sea'",
        ):
            mask.generate_and_apply_land_sea_mask(ds, "ts", keep="artic")

    def test_mask_output_mask(self, ds):
        output = mask.generate_and_apply_land_sea_mask(ds, "ts", output_mask=True)

        assert "ts_mask" in output

        output = mask.generate_and_apply_land_sea_mask(ds, "ts", output_mask="sea_mask")

        assert "sea_mask" in output

    def test_mask_fractional(self, ds):
        custom_mask = xr.DataArray(
            [
                [0.1, 0.1, 0.1, 0.1],
                [0.1, 0.9, 0.9, 0.1],
                [0.1, 0.9, 0.9, 0.1],
                [0.1, 0.1, 0.1, 0.1],
            ],
            dims=("lat", "lon"),
        )

        expected_sea = xr.DataArray(
            [
                [1, 1, 1, 1],
                [1, np.nan, np.nan, 1],
                [1, np.nan, np.nan, 1],
                [1, 1, 1, 1],
            ],
            dims=("lat", "lon"),
            coords={
                "lat": ds.lat.copy(),
                "lon": ds.lon.copy(),
                "time": ds.time.copy()[0],
            },
        )

        output = mask.generate_and_apply_land_sea_mask(
            ds.isel(time=0), "ts", mask=custom_mask
        )

        xr.testing.assert_allclose(output.ts, expected_sea)

        # invert expected
        expected_land = expected_sea.copy()
        expected_land = xr.where(expected_sea == 1, np.nan, expected_land)
        expected_land = xr.where(np.isnan(expected_sea), 1.0, np.nan)

        output = mask.generate_and_apply_land_sea_mask(
            ds.isel(time=0), "ts", keep="land", mask=custom_mask
        )

        xr.testing.assert_allclose(output.ts, expected_land)

    def test_mask_custom(self, ds):
        custom_mask = xr.DataArray(
            [
                [1, 0, 0, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 0, 0, 1],
            ],
            dims=("lat", "lon"),
        )

        expected = xr.DataArray(
            [
                [np.nan, 1, 1, np.nan],
                [np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan],
                [np.nan, 1, 1, np.nan],
            ],
            dims=("lat", "lon"),
            coords={
                "lat": ds.lat.copy(),
                "lon": ds.lon.copy(),
                "time": ds.time.copy()[0],
            },
        )

        output = mask.generate_and_apply_land_sea_mask(
            ds.isel(time=0), "ts", mask=custom_mask
        )

        xr.testing.assert_allclose(output.ts, expected)

    def test_mask_land(self, ds):
        expected = xr.DataArray(
            expected_land,
            dims=("lat", "lon"),
            coords={
                "lat": ds.lat.copy(),
                "lon": ds.lon.copy(),
                "time": ds.time[0].copy(),
            },
        )

        output = mask.generate_and_apply_land_sea_mask(ds.isel(time=0), "ts")

        xr.testing.assert_allclose(output.ts, expected)

    def test_mask_sea(self, ds):
        expected = xr.DataArray(
            expected_sea,
            dims=("lat", "lon"),
            coords={
                "lat": ds.lat.copy(),
                "lon": ds.lon.copy(),
                "time": ds.time[0].copy(),
            },
        )

        output = mask.generate_and_apply_land_sea_mask(
            ds.isel(time=0), "ts", keep="land"
        )

        xr.testing.assert_allclose(output.ts, expected)


class TestLandSeaMask:
    def test_generate_land_sea_mask_invalid_method(self, ds):
        with pytest.raises(
            ValueError,
            match=r"Method value 'custom' is not valid, options are 'regionmask, pcmdi'",
        ):
            mask.generate_land_sea_mask(ds["ts"], method="custom")

    def test_generate_land_sea_mask_regionmask(self, ds):
        expected = xr.DataArray(
            [
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dims=("lat", "lon"),
            coords={"lat": ds.lat.copy(), "lon": ds.lon.copy()},
        )

        output = mask.generate_land_sea_mask(ds["ts"])

        xr.testing.assert_allclose(output, expected)

    def test_generate_land_sea_mask_pcmdi(self, ds):
        expected = xr.DataArray(
            [
                [1, 1, 1, 1],
                [0, 0, 0, 0],
                [1, 1, 0, 1],
                [0, 0, 0, 0],
            ],
            dims=("lat", "lon"),
            coords={"lat": ds.lat.copy(), "lon": ds.lon.copy()},
            attrs={"Conventions": "CF-1.0"},
        )

        output = mask.generate_land_sea_mask(ds["ts"], method="pcmdi")

        xr.testing.assert_equal(output, expected)

    def test_pcmdi_land_sea_mask_custom_source(self, ds):
        source = xr.DataArray(
            [
                [0.1, 0.1, 0.9, 0.2],
                [0.1, 0.9, 0.9, 0.1],
                [0.0, 0.1, 0.9, 0.9],
                [0.1, 0.1, 0.9, 0.1],
            ],
            dims=("lat", "lon"),
            coords={"lat": ds.lat.copy(), "lon": ds.lon.copy()},
            attrs={"Conventions": "CF-1.0"},
        ).to_dataset(name="highres_mask")

        output = mask.pcmdi_land_sea_mask(
            ds["ts"], source=source, source_data_var="highres_mask"
        )

        expected = xr.DataArray(
            [[0, 0, 1, 0], [0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 1, 0]],
            dims=("lat", "lon"),
            coords={"lat": ds.lat.copy(), "lon": ds.lon.copy()},
            attrs={"Conventions": "CF-1.0"},
        )

        xr.testing.assert_allclose(output, expected)

    def test_pcmdi_land_sea_mask_custom_source_error(self, ds):
        source = xr.DataArray(
            [
                [0.1, 0.1, 0.9, 0.2],
                [0.1, 0.9, 0.9, 0.1],
                [0.0, 0.1, 0.9, 0.9],
                [0.1, 0.1, 0.9, 0.1],
            ],
            dims=("lat", "lon"),
            coords={"lat": ds.lat.copy(), "lon": ds.lon.copy()},
            attrs={"Conventions": "CF-1.0"},
        ).to_dataset(name="highres_mask")

        with pytest.raises(
            ValueError,
            match="The 'source_data_var' value cannot be None when using the 'source' option.",
        ):
            mask.pcmdi_land_sea_mask(ds["ts"], source=source)

    @mock.patch("xcdat.mask._improve_mask")
    def test_pcmdi_land_sea_mask_multiple_iterations(self, _improve_mask, ds):
        mask1 = xr.DataArray(
            [
                [1, 1, 1, 1],
                [0, 0, 0, 0],
                [1, 1, 0, 1],
                [0, 0, 0, 0],
            ],
            dims=("lat", "lon"),
            coords={"lat": ds.lat.copy(), "lon": ds.lon.copy()},
        )
        mask2 = xr.DataArray(
            [
                [1, 1, 1, 1],
                [0, 0, 0, 0],
                [1, 1, 1, 1],
                [0, 0, 0, 0],
            ],
            dims=("lat", "lon"),
            coords={"lat": ds.lat.copy(), "lon": ds.lon.copy()},
        )

        _improve_mask.side_effect = [
            xr.Dataset({"sftlf": mask1.copy()}),
            xr.Dataset({"sftlf": mask2.copy()}),
            xr.Dataset({"sftlf": mask2.copy()}),
        ]

        expected = xr.DataArray(
            [
                [1, 1, 1, 1],
                [0, 0, 0, 0],
                [1, 1, 1, 1],
                [0, 0, 0, 0],
            ],
            dims=("lat", "lon"),
            coords={"lat": ds.lat.copy(), "lon": ds.lon.copy()},
        )

        output = mask.pcmdi_land_sea_mask(ds["ts"])

        xr.testing.assert_equal(output, expected)


class TestUtilities:
    def test_is_circular(self):
        # Circular
        lon = xr.DataArray(data=np.array([0, 90, 180, 270]), dims=["lon"])
        lon_bnds = xr.DataArray(
            data=np.array([[-45, 45], [45, 135], [135, 225], [225, 315]]),
            dims=["lon", "bnds"],
        )
        assert mask._is_circular(lon, lon_bnds) is True

        # Not circular
        lon = xr.DataArray(data=np.array([0, 90, 180, 270]), dims=["lon"])
        lon_bnds = xr.DataArray(
            data=np.array([[-45, 45], [45, 135], [135, 225], [225, 300]]),
            dims=["lon", "bnds"],
        )
        assert mask._is_circular(lon, lon_bnds) is False

    def test_generate_surrounds_non_circular(self, source_da):
        UL, UC, UR, ML, MR, LL, LC, LR = mask._generate_surrounds(
            source_da, is_circular=False
        )

        np.testing.assert_array_equal(UC, source_da[2:, 1:-1])
        np.testing.assert_array_equal(LC, source_da[:-2, 1:-1])
        np.testing.assert_array_equal(ML, source_da[1:-1, :-2])
        np.testing.assert_array_equal(MR, source_da[1:-1, 2:])
        np.testing.assert_array_equal(UL, source_da[2:, :-2])
        np.testing.assert_array_equal(UR, source_da[2:, 2:])
        np.testing.assert_array_equal(LL, source_da[:-2, :-2])
        np.testing.assert_array_equal(LR, source_da[:-2, 2:])

    def test_generate_surrounds_circular(self, source_da):
        UL, UC, UR, ML, MR, LL, LC, LR = mask._generate_surrounds(
            source_da, is_circular=True
        )

        np.testing.assert_array_equal(UC, source_da[2:, :])
        np.testing.assert_array_equal(LC, source_da[:-2, :])
        np.testing.assert_array_equal(ML, np.roll(source_da[1:-1, :], 1, axis=1))
        np.testing.assert_array_equal(MR, np.roll(source_da[1:-1, :], -1, axis=1))
        np.testing.assert_array_equal(UL, np.roll(source_da[2:, :], 1, axis=1))
        np.testing.assert_array_equal(UR, np.roll(source_da[2:, :], -1, axis=1))
        np.testing.assert_array_equal(LL, np.roll(source_da[:-2, :], 1, axis=1))
        np.testing.assert_array_equal(LR, np.roll(source_da[:-2, :], -1, axis=1))

    def test_convert_points_to_land(self, mask_da, source_da, diff_da):
        diff_da[1, 1] = 0.8

        source_da[1, 1] = 0.4

        surrounds = mask._generate_surrounds(source_da, is_circular=False)

        result = mask._convert_points(
            mask_da,
            source_da,
            diff_da,
            threshold1=0.2,
            threshold2=0.3,
            is_circular=False,
            surrounds=surrounds,
            convert_land=True,
        )
        expected = mask_da.copy()
        expected[1, 1] = 1.0
        xr.testing.assert_allclose(result, expected)

    def test_convert_points_to_sea(self, mask_da, source_da, diff_da):
        diff_da[2, 2] = -0.8

        source_da[2, 2] = 0.6

        surrounds = mask._generate_surrounds(source_da, is_circular=False)

        result = mask._convert_points(
            mask_da,
            source_da,
            diff_da,
            threshold1=-0.2,
            threshold2=0.7,
            is_circular=False,
            surrounds=surrounds,
            convert_land=False,
        )
        expected = mask_da.copy()
        expected[2, 2] = 0.0
        xr.testing.assert_allclose(result, expected)
