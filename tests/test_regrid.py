import datetime
import sys
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tests import fixtures
from xcdat.regridder import accessor, base, grid, regrid2, xesmf

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
        self.coarse_lat_bnds = gen_uniform_axis(-90, 90.1, 30, "lat", "Y")

        self.fine_lat_bnds = gen_uniform_axis(-90, 90.1, 10, "lat", "Y")

        self.reversed_lat_bnds = gen_uniform_axis(90, -90.1, -30, "lat", "Y")

        self.coarse_lon_bnds = gen_uniform_axis(-0.5, 360, 90, "lon", "X")

        self.fine_lon_bnds = gen_uniform_axis(-0.5, 360, 45, "lon", "X")

        np.random.seed(1337)

        time = pd.date_range("1970-01-01", periods=3)
        time_bnds = np.vstack((time[:-1].to_numpy(), time[1:].to_numpy())).reshape(
            (2, 2)
        )
        time = time[:-1].to_pydatetime() + datetime.timedelta(hours=12)

        self.input_ds = xr.Dataset(
            {
                "ts": xr.DataArray(
                    np.random.random(
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
                "time_bnds": (["time", "bnds"], time_bnds),
                "lat_bnds": self.coarse_lat_bnds,
                "lon_bnds": self.coarse_lon_bnds,
            }
        )

        self.output_ds = xr.Dataset(
            {
                "ts": xr.DataArray(
                    np.random.random(
                        (2, self.fine_lat_bnds.shape[0], self.fine_lon_bnds.shape[0])
                    ),
                    dims=["time", "lat", "lon"],
                    coords={
                        "time": ("time", time, {"bounds": "time_bnds", "axis": "T"}),
                        "lat": self.fine_lat_bnds["lat"],
                        "lon": self.fine_lon_bnds["lon"],
                    },
                ),
                "time_bnds": (["time", "bnds"], time_bnds),
                "lat_bnds": self.fine_lat_bnds,
                "lon_bnds": self.fine_lon_bnds,
            }
        )

    def test_align_axis(self):
        src = gen_uniform_axis(-0.5, 360, 30, "lon", "X")

        dst = gen_uniform_axis(-0.5, 360, 60, "lon", "X")

        src_west, src_east = regrid2.extract_bounds(src)

        dst_west, _ = regrid2.extract_bounds(dst)

        shifted_west, shifted_east, shift = regrid2.align_axis(
            src_west, src_east, dst_west
        )

        assert shift == 0

        src_neg = xr.DataArray(np.roll(src, -2))

        src_neg_west, src_neg_east = regrid2.extract_bounds(src_neg)

        shifted_west, shifted_east, shift = regrid2.align_axis(
            src_neg_west, src_neg_east, dst_west
        )

        assert shift == 11

        src_180 = gen_uniform_axis(-180, 180, 30, "lon", "X")

        src_180_west, src_180_east = regrid2.extract_bounds(src_180)

        shifted_west, shifted_east, shift = regrid2.align_axis(
            src_180_west, src_180_east, dst_west
        )

        assert shift == 5

    def test_unknown_variable(self):
        regridder = regrid2.Regrid2Regridder(self.input_ds, self.output_ds)

        with pytest.raises(KeyError):
            regridder.regrid("unknown", self.output_ds)

    def test_regrid(self):
        regridder = regrid2.Regrid2Regridder(self.input_ds, self.output_ds)

        output_data = regridder.regrid("ts", self.output_ds)

        expected_output = [
            [
                [
                    0.4912290772658786,
                    0.4912290772658786,
                    0.9728723298688062,
                    0.9728723298688062,
                    0.8359678800325482,
                    0.8359678800325482,
                    0.6102373321732014,
                    0.6102373321732014,
                ],
                [
                    0.4912290772658786,
                    0.4912290772658786,
                    0.9728723298688061,
                    0.9728723298688061,
                    0.8359678800325482,
                    0.8359678800325482,
                    0.6102373321732014,
                    0.6102373321732014,
                ],
                [
                    0.4912290772658786,
                    0.4912290772658786,
                    0.9728723298688061,
                    0.9728723298688061,
                    0.8359678800325482,
                    0.8359678800325482,
                    0.6102373321732014,
                    0.6102373321732014,
                ],
                [
                    0.08904506720249383,
                    0.08904506720249383,
                    0.9389826005459538,
                    0.9389826005459538,
                    0.9734593993676922,
                    0.9734593993676922,
                    0.49148152983141974,
                    0.49148152983141974,
                ],
                [
                    0.08904506720249383,
                    0.08904506720249383,
                    0.9389826005459541,
                    0.9389826005459541,
                    0.9734593993676922,
                    0.9734593993676922,
                    0.49148152983141974,
                    0.49148152983141974,
                ],
                [
                    0.08904506720249383,
                    0.08904506720249383,
                    0.938982600545954,
                    0.938982600545954,
                    0.9734593993676923,
                    0.9734593993676923,
                    0.4914815298314197,
                    0.4914815298314197,
                ],
                [
                    0.6714027862126307,
                    0.6714027862126307,
                    0.19044552569632212,
                    0.19044552569632212,
                    0.666464652569817,
                    0.666464652569817,
                    0.9111224015603134,
                    0.9111224015603134,
                ],
                [
                    0.6714027862126307,
                    0.6714027862126307,
                    0.19044552569632212,
                    0.19044552569632212,
                    0.666464652569817,
                    0.666464652569817,
                    0.9111224015603134,
                    0.9111224015603134,
                ],
                [
                    0.6714027862126307,
                    0.6714027862126307,
                    0.19044552569632212,
                    0.19044552569632212,
                    0.666464652569817,
                    0.666464652569817,
                    0.9111224015603134,
                    0.9111224015603134,
                ],
                [
                    0.26545033832019393,
                    0.26545033832019393,
                    0.5189586503858306,
                    0.5189586503858306,
                    0.17746798488170112,
                    0.17746798488170112,
                    0.467845972324444,
                    0.467845972324444,
                ],
                [
                    0.26545033832019393,
                    0.26545033832019393,
                    0.5189586503858306,
                    0.5189586503858306,
                    0.17746798488170112,
                    0.17746798488170112,
                    0.467845972324444,
                    0.467845972324444,
                ],
                [
                    0.26545033832019393,
                    0.26545033832019393,
                    0.5189586503858306,
                    0.5189586503858306,
                    0.17746798488170112,
                    0.17746798488170112,
                    0.467845972324444,
                    0.467845972324444,
                ],
                [
                    0.7239478979864309,
                    0.7239478979864309,
                    0.7930230326291929,
                    0.7930230326291929,
                    0.35152409979850174,
                    0.35152409979850174,
                    0.9096535248282638,
                    0.9096535248282638,
                ],
                [
                    0.7239478979864309,
                    0.7239478979864309,
                    0.7930230326291929,
                    0.7930230326291929,
                    0.35152409979850174,
                    0.35152409979850174,
                    0.9096535248282637,
                    0.9096535248282637,
                ],
                [
                    0.7239478979864309,
                    0.7239478979864309,
                    0.7930230326291929,
                    0.7930230326291929,
                    0.35152409979850174,
                    0.35152409979850174,
                    0.9096535248282638,
                    0.9096535248282638,
                ],
                [
                    0.903927199330352,
                    0.903927199330352,
                    0.6335486059417301,
                    0.6335486059417301,
                    0.532829463625472,
                    0.532829463625472,
                    0.23751076508188718,
                    0.23751076508188718,
                ],
                [
                    0.903927199330352,
                    0.903927199330352,
                    0.63354860594173,
                    0.63354860594173,
                    0.532829463625472,
                    0.532829463625472,
                    0.23751076508188718,
                    0.23751076508188718,
                ],
                [
                    0.903927199330352,
                    0.903927199330352,
                    0.63354860594173,
                    0.63354860594173,
                    0.532829463625472,
                    0.532829463625472,
                    0.23751076508188718,
                    0.23751076508188718,
                ],
            ],
            [
                [
                    0.2152504148025276,
                    0.2152504148025276,
                    0.4083469363533958,
                    0.4083469363533958,
                    0.6552760423503994,
                    0.6552760423503994,
                    0.2703198694818053,
                    0.2703198694818053,
                ],
                [
                    0.2152504148025276,
                    0.2152504148025276,
                    0.4083469363533958,
                    0.4083469363533958,
                    0.6552760423503994,
                    0.6552760423503994,
                    0.2703198694818053,
                    0.2703198694818053,
                ],
                [
                    0.2152504148025276,
                    0.2152504148025276,
                    0.4083469363533958,
                    0.4083469363533958,
                    0.6552760423503994,
                    0.6552760423503994,
                    0.2703198694818053,
                    0.2703198694818053,
                ],
                [
                    0.7241143201319212,
                    0.7241143201319212,
                    0.43316776184776185,
                    0.43316776184776185,
                    0.8205336973552095,
                    0.8205336973552095,
                    0.8243295390756634,
                    0.8243295390756634,
                ],
                [
                    0.7241143201319212,
                    0.7241143201319212,
                    0.43316776184776185,
                    0.43316776184776185,
                    0.8205336973552094,
                    0.8205336973552094,
                    0.8243295390756633,
                    0.8243295390756633,
                ],
                [
                    0.7241143201319212,
                    0.7241143201319212,
                    0.43316776184776185,
                    0.43316776184776185,
                    0.8205336973552095,
                    0.8205336973552095,
                    0.8243295390756634,
                    0.8243295390756634,
                ],
                [
                    0.7860732410584073,
                    0.7860732410584073,
                    0.9371915835783589,
                    0.9371915835783589,
                    0.6581630055733633,
                    0.6581630055733633,
                    0.18749788177521134,
                    0.18749788177521134,
                ],
                [
                    0.7860732410584074,
                    0.7860732410584074,
                    0.9371915835783589,
                    0.9371915835783589,
                    0.6581630055733633,
                    0.6581630055733633,
                    0.18749788177521132,
                    0.18749788177521132,
                ],
                [
                    0.7860732410584074,
                    0.7860732410584074,
                    0.9371915835783587,
                    0.9371915835783587,
                    0.6581630055733632,
                    0.6581630055733632,
                    0.18749788177521132,
                    0.18749788177521132,
                ],
                [
                    0.04670308301948888,
                    0.04670308301948888,
                    0.007838989547189201,
                    0.007838989547189201,
                    0.7495067105618435,
                    0.7495067105618435,
                    0.7335399234388285,
                    0.7335399234388285,
                ],
                [
                    0.04670308301948889,
                    0.04670308301948889,
                    0.007838989547189201,
                    0.007838989547189201,
                    0.7495067105618435,
                    0.7495067105618435,
                    0.7335399234388285,
                    0.7335399234388285,
                ],
                [
                    0.04670308301948889,
                    0.04670308301948889,
                    0.007838989547189201,
                    0.007838989547189201,
                    0.7495067105618435,
                    0.7495067105618435,
                    0.7335399234388285,
                    0.7335399234388285,
                ],
                [
                    0.8231728257419227,
                    0.8231728257419227,
                    0.4037180002921906,
                    0.4037180002921906,
                    0.1791307121550776,
                    0.1791307121550776,
                    0.6314562645575412,
                    0.6314562645575412,
                ],
                [
                    0.8231728257419227,
                    0.8231728257419227,
                    0.4037180002921906,
                    0.4037180002921906,
                    0.1791307121550776,
                    0.1791307121550776,
                    0.6314562645575412,
                    0.6314562645575412,
                ],
                [
                    0.8231728257419227,
                    0.8231728257419227,
                    0.4037180002921906,
                    0.4037180002921906,
                    0.1791307121550776,
                    0.1791307121550776,
                    0.6314562645575412,
                    0.6314562645575412,
                ],
                [
                    0.6172659124628479,
                    0.6172659124628479,
                    0.5738028270065219,
                    0.5738028270065219,
                    0.8261093231490234,
                    0.8261093231490234,
                    0.7447778285204237,
                    0.7447778285204237,
                ],
                [
                    0.6172659124628479,
                    0.6172659124628479,
                    0.5738028270065219,
                    0.5738028270065219,
                    0.8261093231490234,
                    0.8261093231490234,
                    0.7447778285204237,
                    0.7447778285204237,
                ],
                [
                    0.6172659124628479,
                    0.6172659124628479,
                    0.5738028270065219,
                    0.5738028270065219,
                    0.8261093231490234,
                    0.8261093231490234,
                    0.7447778285204237,
                    0.7447778285204237,
                ],
            ],
        ]

        np.testing.assert_allclose(output_data.ts, expected_output)

    def test_map_longitude_coarse_to_fine(self):
        mapping, weights = regrid2.map_longitude(
            self.coarse_lon_bnds, self.fine_lon_bnds
        )

        expected_mapping = [[0], [0], [1], [1], [2], [2], [3], [3]]
        expected_weigths = [
            [45.0],
            [45.0],
            [45.0],
            [45.0],
            [45.0],
            [45.0],
            [45.0],
            [45.0],
        ]

        np.testing.assert_allclose(mapping, expected_mapping)
        np.testing.assert_allclose(weights, expected_weigths)

    def test_map_longitude_fine_to_coarse(self):
        mapping, weights = regrid2.map_longitude(
            self.fine_lon_bnds, self.coarse_lon_bnds
        )

        expected_mapping = [[0, 1], [2, 3], [4, 5], [6, 7]]
        expected_weigths = [[45.0, 45.0], [45.0, 45.0], [45.0, 45.0], [45.0, 45.0]]

        np.testing.assert_allclose(mapping, expected_mapping)
        np.testing.assert_allclose(weights, expected_weigths)

    def test_map_latitude_coarse_to_fine(self):
        mapping, weights = regrid2.map_latitude(
            self.coarse_lat_bnds, self.fine_lat_bnds
        )

        expected_mapping = [
            [0],
            [0],
            [0],
            [1],
            [1],
            [1],
            [2],
            [2],
            [2],
            [3],
            [3],
            [3],
            [4],
            [4],
            [4],
            [5],
            [5],
            [5],
        ]
        expected_weigths = [
            [0.01519224698779198],
            [0.0451151322262997],
            [0.07366721700146972],
            [0.09998096066546058],
            [0.12325683343243876],
            [0.1427876096865393],
            [0.15797985667433123],
            [0.16837196565873838],
            [0.17364817766693033],
            [0.17364817766693033],
            [0.16837196565873838],
            [0.15797985667433123],
            [0.1427876096865393],
            [0.12325683343243876],
            [0.09998096066546058],
            [0.07366721700146972],
            [0.0451151322262997],
            [0.01519224698779198],
        ]

        np.testing.assert_allclose(mapping, expected_mapping)
        np.testing.assert_allclose(weights, expected_weigths)

    def test_map_latitude_fine_to_coarse(self):
        mapping, weights = regrid2.map_latitude(
            self.fine_lat_bnds, self.coarse_lat_bnds
        )

        expected_mapping = [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [9, 10, 11],
            [12, 13, 14],
            [15, 16, 17],
        ]
        expected_weigths = [
            [0.01519224698779198, 0.0451151322262997, 0.07366721700146972],
            [0.09998096066546058, 0.12325683343243876, 0.1427876096865393],
            [0.15797985667433123, 0.16837196565873838, 0.17364817766693033],
            [0.17364817766693033, 0.16837196565873838, 0.15797985667433123],
            [0.1427876096865393, 0.12325683343243876, 0.09998096066546058],
            [0.07366721700146972, 0.0451151322262997, 0.01519224698779198],
        ]

        np.testing.assert_allclose(mapping, expected_mapping)
        np.testing.assert_allclose(weights, expected_weigths)

    def test_extract_bounds(self):
        south, north = regrid2.extract_bounds(self.coarse_lat_bnds)

        assert south.shape == (6,)
        assert south[0], south[-1] == (-90, 60)

        assert north.shape == (6,)
        assert north[0], north[-1] == (60, 90)

    def test_reversed_extract_bounds(self):
        south, north = regrid2.extract_bounds(self.reversed_lat_bnds)

        assert south.shape == (6,)
        assert south[0], south[-1] == (-90, 60)

        assert north.shape == (6,)
        assert north[0], north[-1] == (60, 90)


# TODO improve testing
class TestXESMFRegridder:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = fixtures.generate_dataset(True, True)
        self.new_grid = grid.create_uniform_grid(-90, 90, 4.0, -180, 180, 5.0)

    def test_no_variable(self):
        ds = self.ds.copy()

        regridder = xesmf.XESMFRegridder(ds, self.new_grid, "bilinear")

        with pytest.raises(KeyError):
            regridder.regrid("unknown", ds)

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


class TestGrid:
    def test_uniform_grid(self):
        new_grid = grid.create_uniform_grid(-90, 90, 4.0, -180, 180, 5.0)

        assert new_grid.lat[0] == -90.0
        assert new_grid.lat[-1] == 86
        assert new_grid.lat.units == "degrees_north"

        assert new_grid.lon[0] == -180
        assert new_grid.lon[-1] == 175
        assert new_grid.lon.units == "degrees_east"

    def test_gaussian_grid(self):
        small_grid = grid.create_gaussian_grid(32)

        assert small_grid.lat.shape == (32,)
        assert small_grid.lon.shape == (64,)

        large_grid = grid.create_gaussian_grid(128)

        assert large_grid.lat.shape == (128,)
        assert large_grid.lon.shape == (256,)

        uneven_grid = grid.create_gaussian_grid(33)

        assert uneven_grid.lat.shape == (33,)
        assert uneven_grid.lon.shape == (66,)

    def test_global_mean_grid(self):
        new_grid = grid.create_gaussian_grid(32)

        mean_grid = grid.create_global_mean_grid(new_grid)

        assert mean_grid.cf["lat"].data == np.array(
            [
                0.0,
            ]
        )
        assert mean_grid.cf["lon"].data == np.array(
            [
                177.1875,
            ]
        )

    def test_zonal_grid(self):
        new_grid = xr.Dataset(
            coords={
                "lat": xr.DataArray(
                    name="lat",
                    data=np.array([-80, -40, 0, 40, 80]),
                    dims=["lat"],
                    attrs={
                        "units": "degrees_north",
                        "axis": "Y",
                        "bounds": "lat_bnds",
                    },
                ),
                "lon": xr.DataArray(
                    name="lon",
                    data=np.array([-160, -80, 0, 80, 160]),
                    dims=["lon"],
                    attrs={
                        "units": "degrees_east",
                        "axis": "X",
                        "bounds": "lon_bnds",
                    },
                ),
            },
            data_vars={
                "lat_bnds": xr.DataArray(
                    name="lat_bnds",
                    data=np.array(
                        [[-90, -60], [-60, -20], [-20, 20], [20, 60], [60, 90]]
                    ),
                    dims=["lat", "bnds"],
                ),
                "lon_bnds": xr.DataArray(
                    name="lon_bnds",
                    data=np.array(
                        [[-180, -120], [-120, -40], [-40, 40], [40, 120], [120, 180]]
                    ),
                    dims=["lon", "bnds"],
                ),
            },
        )

        zonal_grid = grid.create_zonal_grid(new_grid)

        expected_grid = xr.Dataset(
            coords={
                "lat": xr.DataArray(
                    name="lat",
                    data=np.array([-80, -40, 0, 40, 80]),
                    dims=["lat"],
                    attrs={
                        "units": "degrees_north",
                        "axis": "Y",
                        "bounds": "lat_bnds",
                    },
                ),
                "lon": xr.DataArray(
                    name="lon",
                    data=np.array([0.0]),
                    dims=["lon"],
                    attrs={
                        "units": "degrees_east",
                        "axis": "X",
                        "bounds": "lon_bnds",
                    },
                ),
            },
            data_vars={
                "lat_bnds": xr.DataArray(
                    name="lat_bnds",
                    data=np.array(
                        [[-90, -60], [-60, -20], [-20, 20], [20, 60], [60, 90]]
                    ),
                    dims=["lat", "bnds"],
                ),
                "lon_bnds": xr.DataArray(
                    name="lon_bnds",
                    data=np.array([[-180, 180]]),
                    dims=["lon", "bnds"],
                ),
            },
        )

        assert zonal_grid.identical(expected_grid)


class TestAccessor:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.data = mock.MagicMock()
        self.ac = accessor.DatasetRegridderAccessor(self.data)

    def test_valid_tool(self):
        mock_regridder = mock.MagicMock()
        mock_regridder.return_value.regrid.return_value = "output data"

        mock_data = mock.MagicMock()

        with mock.patch.dict(accessor.REGRID_TOOLS, {"test": mock_regridder}):
            output = self.ac.regrid("ts", mock_data, "test")

        assert output == "output data"

        mock_regridder.return_value.regrid.assert_called_with("ts", self.data)

    def test_invalid_tool(self):
        with pytest.raises(
            ValueError, match=r"Tool 'test' does not exist, valid choices"
        ):
            self.ac.regrid("ts", mock.MagicMock(), "test")


class TestBase:
    def test_regridder_implementation(self):
        class NewRegridder(base.BaseRegridder):
            def __init__(self, src_grid, dst_grid, **options):
                super().__init__(src_grid, dst_grid, **options)

            def regrid(self, data_var, ds):
                return ds

        regridder = NewRegridder(mock.MagicMock(), mock.MagicMock())

        assert regridder is not None

        ds_in = mock.MagicMock()

        ds_out = regridder.regrid("ts", ds_in)

        assert ds_in == ds_out
