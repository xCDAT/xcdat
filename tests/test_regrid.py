import datetime
import sys
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tests import fixtures, has_xesmf, requires_xesmf
from xcdat.regridder import accessor, base, grid, regrid2

if has_xesmf:
    from xcdat.regridder import xesmf

np.set_printoptions(threshold=sys.maxsize, suppress=True)


def gen_uniform_axis(start, stop, step, name, axis):
    temp = np.arange(start, stop, step)

    bounds = np.zeros((temp.shape[0] - 1, 2))
    bounds[:, 0] = temp[:-1]
    bounds[:, 1] = temp[1:]

    points = np.array(
        [temp[i] + ((temp[i + 1] - temp[i]) / 2.0) for i in range(temp.shape[0] - 1)]
    )

    data = xr.DataArray(
        points, dims=[name], attrs={"bounds": f"{name}_bnds", "axis": axis}
    )

    return xr.DataArray(bounds, dims=[name, "bnds"], coords={name: data})


class TestRegrid2Regridder:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.coarse_lat_bnds = gen_uniform_axis(-90, 90.1, 60, "lat", "Y")

        self.fine_lat_bnds = gen_uniform_axis(-90, 90.1, 45, "lat", "Y")

        self.reversed_lat_bnds = gen_uniform_axis(90, -90.1, -60, "lat", "Y")

        self.coarse_lon_bnds = gen_uniform_axis(-0.5, 360, 180, "lon", "X")

        self.fine_lon_bnds = gen_uniform_axis(-0.5, 360, 90, "lon", "X")

        time = pd.date_range("1970-01-01", periods=3)
        self.time_bnds = np.vstack((time[:-1].to_numpy(), time[1:].to_numpy())).reshape(
            (2, 2)
        )
        time = time[:-1].to_pydatetime() + datetime.timedelta(hours=12)

        self.coarse_2d_ds = xr.Dataset(
            {
                "ts": xr.DataArray(
                    np.ones(
                        (
                            self.coarse_lat_bnds.shape[0],
                            self.coarse_lon_bnds.shape[0],
                        )
                    ),
                    dims=["lat", "lon"],
                    coords={
                        "lat": self.coarse_lat_bnds["lat"],
                        "lon": self.coarse_lon_bnds["lon"],
                    },
                ),
                "lat_bnds": self.coarse_lat_bnds,
                "lon_bnds": self.coarse_lon_bnds,
            }
        )

        self.coarse_3d_ds = xr.Dataset(
            {
                "ts": xr.DataArray(
                    np.ones(
                        (
                            2,
                            self.coarse_lat_bnds.shape[0],
                            self.coarse_lon_bnds.shape[0],
                        )
                    ),
                    dims=["time", "lat", "lon"],
                    coords={
                        "time": ("time", time, {"bounds": "time_bnds", "axis": "T"}),
                        "lat": self.coarse_lat_bnds["lat"],
                        "lon": self.coarse_lon_bnds["lon"],
                    },
                ),
                "time_bnds": (["time", "bnds"], self.time_bnds),
                "lat_bnds": self.coarse_lat_bnds,
                "lon_bnds": self.coarse_lon_bnds,
            }
        )

        self.height_bnds = gen_uniform_axis(0.0, 40000.1, 20000.0, "height", "Z")

        self.coarse_4d_ds = xr.Dataset(
            {
                "ts": xr.DataArray(
                    np.ones(
                        (
                            2,
                            2,
                            self.coarse_lat_bnds.shape[0],
                            self.coarse_lon_bnds.shape[0],
                        )
                    ),
                    dims=["time", "height", "lat", "lon"],
                    coords={
                        "time": ("time", time, {"bounds": "time_bnds", "axis": "T"}),
                        "height": self.height_bnds["height"],
                        "lat": self.coarse_lat_bnds["lat"],
                        "lon": self.coarse_lon_bnds["lon"],
                    },
                ),
                "time_bnds": (["time", "bnds"], self.time_bnds),
                "height_bnds": self.height_bnds,
                "lat_bnds": self.coarse_lat_bnds,
                "lon_bnds": self.coarse_lon_bnds,
            }
        )

        self.fine_2d_ds = xr.Dataset(
            {
                "ts": xr.DataArray(
                    np.ones((self.fine_lat_bnds.shape[0], self.fine_lon_bnds.shape[0])),
                    dims=["lat", "lon"],
                    coords={
                        "lat": self.fine_lat_bnds["lat"],
                        "lon": self.fine_lon_bnds["lon"],
                    },
                ),
                "lat_bnds": self.fine_lat_bnds,
                "lon_bnds": self.fine_lon_bnds,
            }
        )

    @pytest.mark.filterwarnings("ignore:.*invalid value.*true_divide.*:RuntimeWarning")
    def test_output_bounds(self):
        ds = fixtures.generate_dataset(cf_compliant=True, has_bounds=True)

        output_grid = grid.create_gaussian_grid(32)

        regridder = regrid2.Regrid2Regridder(ds, output_grid)

        output_ds = regridder.horizontal("ts", ds)

        assert "lat_bnds" in output_ds
        assert "lon_bnds" in output_ds
        assert "time_bnds" in output_ds

    @pytest.mark.parametrize(
        "src,dst,expected_west,expected_east,expected_shift",
        [
            (
                np.arange(-180, 180.1, 180),
                np.arange(-180, 180.1, 90),
                np.array([-180, 0, 180]),
                np.array([0, 180, 360]),
                0,
            ),
            (
                np.arange(-180, 180.1, 180),
                np.arange(0, 360.1, 90),
                np.array([0, 180, 360]),
                np.array([180, 360, 540]),
                1,
            ),
            (
                np.arange(0, 360.1, 180),
                np.arange(-180, 180.1, 90),
                np.array([-360, -180, 0]),
                np.array([-180, 0, 180]),
                0,
            ),
            (
                np.arange(0, 360.1, 180),
                np.arange(0, 360.1, 90),
                np.array([0, 180, 360]),
                np.array([180, 360, 540]),
                0,
            ),
            (
                np.arange(180.0, -180.1, -180),
                np.arange(-180, 180.1, 90),
                np.array([0, -180, -360]),
                np.array([-180, -360, -540]),
                1,
            ),
            (
                np.arange(-180.0, 180.1, 180),
                np.arange(180, -180.1, -90),
                np.array([-180, 0, 180]),
                np.array([0, 180, 360]),
                0,
            ),
            (
                np.arange(-360, 360.1, 90),
                np.arange(360, -540.1, -180),
                np.array([-360, -270, -180, -90, -360, -270, -180, -90, 0]),
                np.array([-270, -180, -90, 0, -270, -180, -90, 0, 90]),
                0,
            ),
        ],
    )
    def test_align_axis(self, src, dst, expected_west, expected_east, expected_shift):
        src_west, src_east = src[:-1], src[1:]
        dst_west = dst[:-1]

        shifted_west, shifted_east, shift = regrid2._align_axis(
            src_west, src_east, dst_west
        )

        assert np.all(shifted_west == expected_west)
        assert np.all(shifted_east == expected_east)
        assert shift == expected_shift

    def test_unknown_variable(self):
        regridder = regrid2.Regrid2Regridder(self.coarse_2d_ds, self.fine_2d_ds)

        with pytest.raises(KeyError):
            regridder.horizontal("unknown", self.coarse_2d_ds)

    def test_regrid_input_mask(self):
        regridder = regrid2.Regrid2Regridder(self.coarse_2d_ds, self.fine_2d_ds)

        self.coarse_2d_ds["mask"] = (("lat", "lon"), [[0, 0], [1, 1], [0, 0]])

        output_data = regridder.horizontal("ts", self.coarse_2d_ds)

        expected_output = np.array(
            [
                [1.0, 1.0, 1.0, 1.0],
                [1e20, 1e20, 1e20, 1e20],
                [1e20, 1e20, 1e20, 1e20],
                [1.0, 1.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        )

        # need to replace nans since nan != nan
        output_data["ts"] = output_data.ts.fillna(1e20)

        assert np.all(output_data.ts.values == expected_output)

    def test_regrid_output_mask(self):
        output_mask = [
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [0, 0, 0, 0],
        ]

        self.fine_2d_ds["mask"] = (("lat", "lon"), output_mask)

        regridder = regrid2.Regrid2Regridder(self.coarse_2d_ds, self.fine_2d_ds)

        output_data = regridder.horizontal("ts", self.coarse_2d_ds)

        expected_output = np.array(
            [
                [1.0, 1.0, 1.0, 1.0],
                [1e20, 1e20, 1e20, 1e20],
                [1e20, 1e20, 1e20, 1e20],
                [1.0, 1.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        )

        # need to replace nans since nan != nan
        output_data["ts"] = output_data.ts.fillna(1e20)

        assert np.all(output_data.ts.values == expected_output)

    def test_regrid_2d(self):
        regridder = regrid2.Regrid2Regridder(self.coarse_2d_ds, self.fine_2d_ds)

        output_data = regridder.horizontal("ts", self.coarse_2d_ds)

        assert np.all(output_data.ts == 1)

    def test_regrid_fine_coarse_2d(self):
        regridder = regrid2.Regrid2Regridder(self.fine_2d_ds, self.coarse_2d_ds)

        output_data = regridder.horizontal("ts", self.fine_2d_ds)

        assert np.all(output_data.ts == 1)

    def test_regrid_3d(self):
        regridder = regrid2.Regrid2Regridder(self.coarse_3d_ds, self.fine_2d_ds)

        output_data = regridder.horizontal("ts", self.coarse_3d_ds)

        assert np.all(output_data.ts == 1)
        assert "lat_bnds" in output_data
        assert "lon_bnds" in output_data
        assert "time_bnds" in output_data

    def test_regrid_4d(self):
        regridder = regrid2.Regrid2Regridder(self.coarse_4d_ds, self.fine_2d_ds)

        output_data = regridder.horizontal("ts", self.coarse_4d_ds)

        assert np.all(output_data.ts == 1)

    def test_map_longitude_coarse_to_fine(self):
        mapping, weights = regrid2._map_longitude(
            self.coarse_lon_bnds, self.fine_lon_bnds
        )

        expected_mapping = [
            [0],
            [0],
            [1],
            [1],
        ]

        expected_weigths = [
            [[90]],
            [[90]],
            [[90]],
            [[90]],
        ]

        np.testing.assert_allclose(mapping, expected_mapping)
        np.testing.assert_allclose(weights, expected_weigths)

    def test_map_longitude_fine_to_coarse(self):
        mapping, weights = regrid2._map_longitude(
            self.fine_lon_bnds, self.coarse_lon_bnds
        )

        expected_mapping = [
            [0, 1],
            [2, 3],
        ]

        expected_weigths = [[[90, 90]], [[90, 90]]]

        np.testing.assert_allclose(mapping, expected_mapping)
        np.testing.assert_allclose(weights, expected_weigths)

    def test_map_latitude_coarse_to_fine(self):
        mapping, weights = regrid2._map_latitude(
            self.coarse_lat_bnds, self.fine_lat_bnds
        )

        expected_mapping = [
            [
                0,
            ],
            [0, 1],
            [1, 2],
            [
                2,
            ],
        ]

        expected_weigths = [
            [[0.29289322]],
            [[0.20710678], [0.5]],
            [[0.5], [0.20710678]],
            [[0.29289322]],
        ]

        for x, y in zip(mapping, expected_mapping):
            np.testing.assert_allclose(x, y)

        for x2, y2 in zip(weights, expected_weigths):
            np.testing.assert_allclose(x, y)

    def test_map_latitude_fine_to_coarse(self):
        mapping, weights = regrid2._map_latitude(
            self.fine_lat_bnds, self.coarse_lat_bnds
        )

        expected_mapping = [
            [0, 1],
            [1, 2],
            [2, 3],
        ]

        expected_weigths = [
            [[0.29289322], [0.20710678]],
            [[0.5], [0.5]],
            [[0.20710678], [0.29289322]],
        ]

        np.testing.assert_allclose(mapping, expected_mapping)
        np.testing.assert_allclose(weights, expected_weigths)

    def test_extract_bounds(self):
        south, north = regrid2._extract_bounds(self.coarse_lat_bnds)

        assert south.shape == (3,)
        assert south[0], south[-1] == (-90, 60)

        assert north.shape == (3,)
        assert north[0], north[-1] == (60, 90)

    def test_reversed_extract_bounds(self):
        south, north = regrid2._extract_bounds(self.reversed_lat_bnds)

        assert south.shape == (3,)
        assert south[0], south[-1] == (-90, 60)

        assert north.shape == (3,)
        assert north[0], north[-1] == (60, 90)


@requires_xesmf
class TestXESMFRegridder:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = fixtures.generate_dataset(cf_compliant=True, has_bounds=True)
        self.new_grid = grid.create_uniform_grid(-90, 90, 4.0, -180, 180, 5.0)

    @pytest.mark.xfail
    def test_raises_error_if_xesmf_is_not_installed(self):
        # TODO Find a way to mock the value of `_has_xesmf` to False or
        # to remove the `xesmf` module entirely
        with pytest.raises(ModuleNotFoundError):
            xesmf.XESMFRegridder(self.ds, self.new_grid, "bilinear")

    def test_regrid(self):
        ds = self.ds.copy()

        regridder = xesmf.XESMFRegridder(ds, self.new_grid, "bilinear")

        output = regridder.horizontal("ts", ds)

        assert output.ts.shape == (15, 46, 73)
        assert "lat_bnds" in output
        assert "lon_bnds" in output
        assert "time_bnds" in output

    @pytest.mark.parametrize(
        "name,value,attr_name",
        [
            ("periodic", True, "_periodic"),
            ("extrap_method", "inverse_dist", "_extrap_method"),
            ("extrap_method", "nearest_s2d", "_extrap_method"),
            ("extrap_dist_exponent", 0.1, "_extrap_dist_exponent"),
            ("extrap_num_src_pnts", 10, "_extrap_num_src_pnts"),
            ("ignore_degenerate", False, "_ignore_degenerate"),
        ],
    )
    def test_flags(self, name, value, attr_name):
        ds = self.ds.copy()

        options = {name: value}

        regridder = xesmf.XESMFRegridder(ds, self.new_grid, "bilinear", **options)

        assert getattr(regridder, attr_name) == value

    def test_no_variable(self):
        ds = self.ds.copy()

        regridder = xesmf.XESMFRegridder(ds, self.new_grid, "bilinear")

        with pytest.raises(KeyError):
            regridder.horizontal("unknown", ds)

    def test_invalid_method(self):
        ds = self.ds.copy()

        with pytest.raises(ValueError):
            xesmf.XESMFRegridder(ds, self.new_grid, "bad value")

    def test_invalid_extra_method(self):
        ds = self.ds.copy()

        with pytest.raises(ValueError):
            xesmf.XESMFRegridder(
                ds, self.new_grid, "bilinear", extrap_method="bad value"
            )

    def test_preserve_bounds(self):
        ds = fixtures.generate_dataset(cf_compliant=True, has_bounds=True)

        ds = ds.drop_vars(["lat_bnds", "lon_bnds"])

        regridder = xesmf.XESMFRegridder(ds, self.new_grid, method="bilinear")

        output = regridder.horizontal("ts", ds)

        assert "time_bnds" in output


class TestGrid:
    def test_create_grid(self):
        lat = np.array([-45, 0, 45])
        lon = np.array([30, 60, 90, 120, 150])
        lat_bnds = np.array([[-67.5, -22.5], [-22.5, 22.5], [22.5, 67.5]])
        lon_bnds = np.array([[15, 45], [45, 75], [75, 105], [105, 135], [135, 165]])

        new_grid = grid.create_grid(lat, lon)

        assert np.array_equal(new_grid.lat, lat)
        assert np.array_equal(new_grid.lat_bnds, lat_bnds)
        assert new_grid.lat.units == "degrees_north"
        assert np.array_equal(new_grid.lon, lon)
        assert np.array_equal(new_grid.lon_bnds, lon_bnds)
        assert new_grid.lon.units == "degrees_east"

        da_lat = xr.DataArray(
            name="lat",
            data=lat,
            dims=["lat"],
            attrs={"units": "degrees_north", "axis": "Y"},
        )
        da_lon = xr.DataArray(
            name="lon",
            data=lon,
            dims=["lon"],
            attrs={"units": "degrees_east", "axis": "X"},
        )
        da_lat_bnds = xr.DataArray(name="lat_bnds", data=lat_bnds, dims=["lat", "bnds"])
        da_lon_bnds = xr.DataArray(name="lon_bnds", data=lon_bnds, dims=["lon", "bnds"])

        new_grid = grid.create_grid(
            da_lat, da_lon, lat_bnds=da_lat_bnds, lon_bnds=da_lon_bnds
        )

        assert np.array_equal(new_grid.lat, lat)
        assert np.array_equal(new_grid.lat_bnds, lat_bnds)
        assert new_grid.lat.units == "degrees_north"
        assert np.array_equal(new_grid.lon, lon)
        assert np.array_equal(new_grid.lon_bnds, lon_bnds)
        assert new_grid.lon.units == "degrees_east"

    def test_uniform_grid(self):
        new_grid = grid.create_uniform_grid(-90, 90, 4.0, -180, 180, 5.0)

        assert new_grid.lat[0] == -90.0
        assert new_grid.lat[-1] == 90.0
        assert new_grid.lat.shape == (46,)

        assert new_grid.lon[0] == -180
        assert new_grid.lon[-1] == 180
        assert new_grid.lon.shape == (73,)

    def test_gaussian_grid(self):
        small_grid = grid.create_gaussian_grid(32)

        assert small_grid.lat.shape == (32,)
        assert small_grid.lon.shape == (65,)

        large_grid = grid.create_gaussian_grid(128)

        assert large_grid.lat.shape == (128,)
        assert large_grid.lon.shape == (257,)

        uneven_grid = grid.create_gaussian_grid(33)

        assert uneven_grid.lat.shape == (33,)
        assert uneven_grid.lon.shape == (67,)

    def test_global_mean_grid(self):
        source_grid = grid.create_grid(
            np.array([-80, -40, 0, 40, 80]), np.array([0, 45, 90, 180, 270, 360])
        )

        mean_grid = grid.create_global_mean_grid(source_grid)

        assert np.all(mean_grid.lat == np.array([0.0]))
        assert np.all(mean_grid.lat_bnds == np.array([[-90, 90]]))
        assert np.all(mean_grid.lon == np.array([180.0]))
        assert np.all(mean_grid.lon_bnds == np.array([[-22.5, 405]]))

    def test_zonal_grid(self):
        source_grid = grid.create_grid(
            np.array([-80, -40, 0, 40, 80]), np.array([-160, -80, 80, 160])
        )

        zonal_grid = grid.create_zonal_grid(source_grid)

        assert np.all(zonal_grid.lat == np.array([-80, -40, 0, 40, 80]))
        assert np.all(
            zonal_grid.lat_bnds
            == np.array([[-90, -60], [-60, -20], [-20, 20], [20, 60], [60, 90]])
        )
        assert np.all(zonal_grid.lon == np.array([0.0]))
        assert np.all(zonal_grid.lon_bnds == np.array([-200, 200]))


class TestAccessor:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.data = mock.MagicMock()
        self.ac = accessor.RegridderAccessor(self.data)

    def test_grid_missing_axis(self):
        ds = fixtures.generate_dataset(True, True)

        ds_no_lat = ds.drop_dims(["lat"])

        with pytest.raises(KeyError):
            ds_no_lat.regridder.grid

        ds_no_lon = ds.drop_dims(["lon"])

        with pytest.raises(KeyError):
            ds_no_lon.regridder.grid

    def test_grid(self):
        ds_bounds = fixtures.generate_dataset(True, True)

        grid = ds_bounds.regridder.grid

        assert "lat" in grid
        assert "lon" in grid
        assert "lat_bnds" in grid
        assert "lon_bnds" in grid

        ds_no_bounds = fixtures.generate_dataset(True, False)

        grid = ds_no_bounds.regridder.grid

        assert "lat" in grid
        assert "lon" in grid
        assert "lat_bnds" in grid
        assert "lon_bnds" in grid

    def test_valid_tool(self):
        mock_regridder = mock.MagicMock()
        mock_regridder.return_value.horizontal.return_value = "output data"

        mock_data = mock.MagicMock()

        with mock.patch.dict(accessor.REGRID_TOOLS, {"regrid2": mock_regridder}):
            output = self.ac.horizontal("ts", mock_data, "regrid2")

        assert output == "output data"

        mock_regridder.return_value.horizontal.assert_called_with("ts", self.data)

    def test_invalid_tool(self):
        with pytest.raises(
            ValueError, match=r"Tool 'test' does not exist, valid choices"
        ):
            self.ac.horizontal("ts", mock.MagicMock(), "test")  # type: ignore

    @requires_xesmf
    @pytest.mark.filterwarnings("ignore:.*invalid value.*true_divide.*:RuntimeWarning")
    def test_convenience_methods(self):
        ds = fixtures.generate_dataset(cf_compliant=True, has_bounds=True)

        out_grid = grid.create_gaussian_grid(32)

        output_xesmf = ds.regridder.horizontal_xesmf("ts", out_grid, method="bilinear")

        assert output_xesmf.ts.shape == (15, 32, 65)

        output_regrid2 = ds.regridder.horizontal_regrid2("ts", out_grid)

        assert output_regrid2.ts.shape == (15, 32, 65)

    @pytest.mark.xfail
    def test_raises_error_if_xesmf_is_not_installed(self):
        # TODO Find a way to mock the value of `_has_xesmf` to False or
        # to remove the `xesmf` module entirely
        ds = fixtures.generate_dataset(cf_compliant=True, has_bounds=True)

        out_grid = grid.create_gaussian_grid(32)
        with pytest.raises(ModuleNotFoundError):
            ds.regridder.horizontal_xesmf("ts", out_grid, method="bilinear")


class TestBase:
    def test_preserve_bounds(self):
        ds_with_bounds = fixtures.generate_dataset(cf_compliant=True, has_bounds=True)

        ds_without_bounds = ds_with_bounds.drop_vars(["lat_bnds", "lon_bnds"])

        target = xr.Dataset()

        output_ds = base.preserve_bounds(ds_with_bounds, ds_without_bounds, target)

        assert "lat_bnds" not in output_ds
        assert "lon_bnds" not in output_ds
        assert "time_bnds" in output_ds

        target = xr.Dataset()

        output_ds = base.preserve_bounds(ds_without_bounds, ds_without_bounds, target)

        assert "lat_bnds" not in output_ds
        assert "lon_bnds" not in output_ds
        assert "time_bnds" in output_ds

    def test_regridder_implementation(self):
        class NewRegridder(base.BaseRegridder):
            def __init__(self, src_grid, dst_grid, **options):
                super().__init__(src_grid, dst_grid, **options)

            def horizontal(self, data_var, ds):
                return ds

        regridder = NewRegridder(mock.MagicMock(), mock.MagicMock())

        assert regridder is not None

        ds_in = mock.MagicMock()

        ds_out = regridder.horizontal("ts", ds_in)

        assert ds_in == ds_out
