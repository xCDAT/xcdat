import re
import sys
from unittest import mock

import numpy as np
import pytest
import xarray as xr

from tests import fixtures
from xcdat import mask

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


def test_accessor(ds):
    expected_coords = {
        "lat": ds.lat.copy(),
        "lon": ds.lon.copy(),
        "time": ds.time.copy()[0],
    }

    sea_expected = xr.DataArray(
        expected_land,
        dims=("lat", "lon"),
        coords=expected_coords,
    )

    land_expected = xr.DataArray(
        expected_sea,
        dims=("lat", "lon"),
        coords=expected_coords,
    )

    ac = mask.MaskAccessor(ds.isel(time=0))

    land_output = ac.sea("ts")

    xr.testing.assert_allclose(land_output.ts, land_expected)

    sea_output = ac.land("ts")

    xr.testing.assert_allclose(sea_output.ts, sea_expected)


def test_mask_invalid_data_var(ds):
    with pytest.raises(KeyError):
        mask._mask(ds, "tas")


def test_mask_invalid_keep(ds):
    with pytest.raises(
        ValueError, match=r"Keep value 'artic' is not valid, options are 'land, sea'"
    ):
        mask._mask(ds, "ts", keep="artic")


def test_mask_fractional(ds):
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
        coords={"lat": ds.lat.copy(), "lon": ds.lon.copy(), "time": ds.time.copy()[0]},
    )

    output = mask._mask(ds.isel(time=0), "ts", mask=custom_mask)

    xr.testing.assert_allclose(output.ts, expected_sea)

    # invert expected
    expected_land = expected_sea.copy()
    expected_land = xr.where(expected_sea == 1, np.nan, expected_land)
    expected_land = xr.where(np.isnan(expected_sea), 1.0, np.nan)

    output = mask._mask(ds.isel(time=0), "ts", keep="land", mask=custom_mask)

    xr.testing.assert_allclose(output.ts, expected_land)


def test_mask_custom(ds):
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
        coords={"lat": ds.lat.copy(), "lon": ds.lon.copy(), "time": ds.time.copy()[0]},
    )

    output = mask._mask(ds.isel(time=0), "ts", mask=custom_mask)

    xr.testing.assert_allclose(output.ts, expected)


def test_mask_land(ds):
    expected = xr.DataArray(
        expected_land,
        dims=("lat", "lon"),
        coords={"lat": ds.lat.copy(), "lon": ds.lon.copy(), "time": ds.time[0].copy()},
    )

    output = mask._mask(ds.isel(time=0), "ts")

    xr.testing.assert_allclose(output.ts, expected)


def test_mask_sea(ds):
    expected = xr.DataArray(
        expected_sea,
        dims=("lat", "lon"),
        coords={"lat": ds.lat.copy(), "lon": ds.lon.copy(), "time": ds.time[0].copy()},
    )

    output = mask._mask(ds.isel(time=0), "ts", keep="land")

    xr.testing.assert_allclose(output.ts, expected)


def test_generate_land_sea_mask_invalid_method(ds):
    with pytest.raises(
        ValueError,
        match=r"Method value 'custom' is not valid, options are 'regionmask, pcmdi'",
    ):
        mask.generate_land_sea_mask(ds["ts"], method="custom")


def test_generate_land_sea_mask_regionmask(ds):
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


def test_generate_land_sea_mask_pcmdi(ds):
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

    xr.testing.assert_equal(output.lsmask, expected)


@mock.patch("xcdat.mask._improve_mask")
def test_generate_land_sea_mask_pcmdi_multiple_iterations(_improve_mask, ds):
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

    output = mask.generate_land_sea_mask(ds["ts"], method="pcmdi")

    xr.testing.assert_equal(output.lsmask, expected)


def test_get_resource_path(monkeypatch, tmp_path):
    mock_file = tmp_path / "navy_land.nc"
    mock_file.touch()

    mock_as_file = mock.MagicMock()
    mock_as_file.return_value.__enter__.return_value = mock_file

    monkeypatch.setattr(mask.resources, "as_file", mock_as_file)

    path = mask._get_resource_path("navy_land.nc")

    assert path == mock_file


def test_get_resource_path_fallback_from_exception(monkeypatch, tmp_path):
    mock_file = tmp_path / "xcdat" / "navy_land.nc"
    mock_file.parent.mkdir(parents=True, exist_ok=True)
    mock_file.touch()

    mock_as_file = mock.MagicMock()
    mock_as_file.side_effect = FileNotFoundError("Resource not found")

    monkeypatch.setattr(mask.resources, "as_file", mock_as_file)

    path = mask._get_resource_path("navy_land.nc", tmp_path)

    assert re.match(r".*xcdat/navy_land.nc", str(path))


def test_get_resource_path_fallback_missing(monkeypatch, tmp_path):
    mock_as_file = mock.MagicMock()
    mock_as_file.side_effect = FileNotFoundError("Resource not found")

    monkeypatch.setattr(mask.resources, "as_file", mock_as_file)

    with pytest.raises(
        RuntimeError,
        match=r"Resource file 'navy_land.nc' not found in package or at .*",
    ):
        mask._get_resource_path("navy_land.nc", tmp_path)


def test_is_circular():
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


def test_generate_surrounds_non_circular(source_da):
    # surronds = mask._generate_surrounds(da, is_circular=False)
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


def test_generate_surrounds_circular(source_da):
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


def test_convert_points_to_land(mask_da, source_da, diff_da):
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


def test_convert_points_to_sea(mask_da, source_da, diff_da):
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
