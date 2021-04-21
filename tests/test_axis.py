import numpy as np
import pytest
import xarray as xr

from xcdat.axis import AxisAccessor


class TestAxisAccessor:
    @pytest.fixture(autouse=True)
    def setup(self):
        # Coordinates (axes)
        self.lat = xr.DataArray(
            data=np.arange(-2, 3, 1),
            dims=["lat"],
            attrs={"units": "degrees_north", "axis": "Y"},
        )
        self.lon = xr.DataArray(
            data=np.arange(0, 5, 1),
            dims=["lon"],
            attrs={"units": "degrees_east", "axis": "Y"},
        )

        # Data variables (axes bounds)
        self.lat_bnds = xr.DataArray(
            data=np.array(
                [[-90.0, -1.5], [-1.5, -0.5], [-0.5, 0.5], [0.5, 1.5], [1.5, 90.0]]
            ),
            coords={"lat": np.array(self.lat), "bnds": [0, 1]},
            dims=["lat", "bnds"],
            attrs={"units": "degrees_north", "axis": "Y"},
        )
        self.lon_bnds = xr.DataArray(
            data=np.array(
                [[90.0, 0.5], [0.5, 1.5], [1.5, 2.5], [2.5, 3.5], [3.5, -90.0]]
            ),
            coords={"lon": np.array(self.lon), "bnds": [0, 1]},
            dims=["lon", "bnds"],
            attrs={"units": "degrees_easy", "axis": "Y"},
        )

        # Short axes coordinates names
        self.ds = xr.Dataset(coords={"lat": self.lat, "lon": self.lon})

        # Long axes coordinates names
        self.ds_2 = xr.Dataset(coords={"latitude": self.lat, "longitude": self.lon})
        self.ds_2 = self.ds_2.swap_dims({"lat": "latitude", "lon": "longitude"})

    def test__init__(self):
        obj = AxisAccessor(self.ds)
        assert obj._obj.identical(self.ds)

    def test__calc_bounds_for_short_axis_name(self):
        obj = AxisAccessor(self.ds)

        # Check calculated lat bounds equal existing lat bounds
        lat_bnds = obj._calc_bounds("lat")
        assert lat_bnds.equals(self.lat_bnds)
        assert obj._obj.lat_bnds.is_calculated

        # Check calculated lon bounds equal existing lon bounds
        lon_bnds = obj._calc_bounds("lon")
        assert lon_bnds.equals(self.lon_bnds)
        assert obj._obj.lon_bnds.is_calculated

    def test__calc_bounds_for_long_axis_name(self):
        obj = AxisAccessor(self.ds_2)

        # Check calculated lat bounds equal existing lat bounds
        lat_bnds = obj._calc_bounds("lat")
        assert lat_bnds.equals(self.lat_bnds)
        assert obj._obj.lat_bnds.is_calculated

        # Check calculated lon bounds equal existing lon bounds
        lon_bnds = obj._calc_bounds("lon")
        assert lon_bnds.equals(self.lon_bnds)
        assert obj._obj.lon_bnds.is_calculated

    def test__extract_axis_coords(self):
        obj = AxisAccessor(self.ds)

        lat = obj._extract_axis_coords("lat")
        assert lat is not None

        lon = obj._extract_axis_coords("lon")
        assert lon is not None

    def test__extract_axis_coords_raises_errors(self):
        obj = AxisAccessor(self.ds)

        # Raises error if incorrect axis param is passed
        with pytest.raises(KeyError):
            obj._extract_axis_coords("incorrect_axis")  # type: ignore

        # Raises error if axis param has no matching coords in the Dataset
        with pytest.raises(TypeError):
            obj._obj = self.ds.drop("lat")
            obj._extract_axis_coords("lat")

    def test_get_bounds_for_existing_bounds(self):
        obj = AxisAccessor(self.ds)

        # Assign bounds variables to replicate a DataSet that already has this info
        obj._obj = obj._obj.assign(lat_bnds=self.lat_bnds, lon_bnds=self.lon_bnds)

        lat_bnds = obj.get_bounds("lat")
        lon_bnds = obj.get_bounds("lon")

        assert lat_bnds.identical(obj._obj.lat_bnds)
        assert lon_bnds.identical(obj._obj.lon_bnds)

    def test_get_bounds_for_calculated_bounds(self):
        obj = AxisAccessor(self.ds)

        # Check calculates lat bounds
        lat_bnds = obj.get_bounds("lat")
        lon_bnds = obj.get_bounds("lon")

        assert lat_bnds.equals(self.lat_bnds)
        assert lon_bnds.equals(self.lon_bnds)

    def test_get_bounds_raises_errors(self):
        obj = AxisAccessor(self.ds)

        # Raises error if incorrect axis param is passed
        with pytest.raises(KeyError):
            obj.get_bounds("incorrect_axis")  # type: ignore
