import numpy as np
import pytest
import xarray as xr

from tests.fixtures import generate_dataset
from xcdat.bounds import DataArrayBoundsAccessor, DatasetBoundsAccessor


class TestDatasetBoundsAccessor:
    @pytest.fixture(autouse=True)
    def setup(self):
        # Coordinate information
        lat = xr.DataArray(
            data=np.array([-90, -88.75, 0, 88.75, 90]),
            dims=["lat"],
            attrs={"units": "degrees_north", "axis": "Y"},
        )
        lon = xr.DataArray(
            data=np.array([0, 1.875, 178.125, 356.25, 358.125]),
            dims=["lon"],
            attrs={"units": "degrees_east", "axis": "X"},
        )

        # Generated coordinates bounds
        self.lat_bnds = xr.DataArray(
            name="lat_bnds",
            data=np.array(
                [
                    [-90, -89.375],
                    [-89.375, -44.375],
                    [-44.375, 44.375],
                    [44.375, 89.375],
                    [89.375, 90],
                ]
            ),
            coords={"lat": lat},
            dims=["lat", "bnds"],
            attrs={
                "units": "degrees_north",
                "axis": "Y",
                "is_generated": "True",
            },
        )
        self.lon_bnds = xr.DataArray(
            name="lon_bnds",
            data=np.array(
                [
                    [-0.9375, 0.9375],
                    [0.9375, 90.0],
                    [90.0, 267.1875],
                    [267.1875, 357.1875],
                    [357.1875, 359.0625],
                ]
            ),
            coords={"lon": lon},
            dims=["lon", "bnds"],
            attrs={
                "units": "degrees_east",
                "axis": "X",
                "is_generated": "True",
            },
        )

        # Create Dataset using coordinates
        self.ds = xr.Dataset(coords={"lat": lat, "lon": lon})

    def test__init__(self):
        obj = DatasetBoundsAccessor(self.ds)
        assert obj._dataset.identical(self.ds)

    def test_decorator_call(self):
        assert self.ds.bounds._dataset.identical(self.ds)

    def test_get_bounds_when_bounds_exist_in_dataset(self):
        obj = DatasetBoundsAccessor(self.ds)
        obj._dataset = obj._dataset.assign(
            lat_bnds=self.lat_bnds,
            lon_bnds=self.lon_bnds,
        )

        lat_bnds = obj.get_bounds("lat")
        assert lat_bnds is not None and lat_bnds.identical(self.lat_bnds)
        assert lat_bnds.is_generated

        lon_bnds = obj.get_bounds("lon")
        assert lon_bnds is not None and lon_bnds.identical(self.lon_bnds)
        assert lon_bnds.is_generated

    def test_get_bounds_when_bounds_do_not_exist_in_dataset(self):
        # Check bounds generated if bounds do not exist.
        obj = DatasetBoundsAccessor(self.ds)

        lat_bnds = obj.get_bounds("lat")
        assert lat_bnds is not None
        assert lat_bnds.identical(self.lat_bnds)
        assert lat_bnds.is_generated

        lon_bnds = obj.get_bounds("lon")
        assert lon_bnds is not None
        assert lon_bnds.identical(self.lon_bnds)
        assert lon_bnds.is_generated

        # Check raises error when bounds do not exist and not allowing generated bounds.
        with pytest.raises(ValueError):
            obj._dataset = obj._dataset.drop_vars(["lat_bnds"])
            obj.get_bounds("lat", allow_generating=False)

    def test_get_bounds_raises_error_with_incorrect_axis_argument(self):
        obj = DatasetBoundsAccessor(self.ds)

        with pytest.raises(ValueError):
            obj.get_bounds("incorrect_axis_argument")  # type: ignore

    def test__get_bounds_does_not_drop_attrs_of_existing_coords_when_generating_bounds(
        self,
    ):
        ds = self.ds.copy()

        lat_bnds = ds.bounds.get_bounds("lat", allow_generating=True)
        assert lat_bnds.identical(self.lat_bnds)

        ds = ds.drop("lat_bnds")
        assert ds.identical(self.ds)

    def test__generate_bounds_raises_errors_for_data_dim_and_length(self):
        # Multidimensional
        lat = xr.DataArray(
            data=np.array([[0, 1, 2], [3, 4, 5]]),
            dims=["placeholder_1", "placeholder_2"],
            attrs={"units": "degrees_north", "axis": "Y"},
        )
        # Length <=1
        lon = xr.DataArray(
            data=np.array([0]),
            dims=["lon"],
            attrs={"units": "degrees_east", "axis": "X"},
        )
        ds = xr.Dataset(coords={"lat": lat, "lon": lon})
        obj = DatasetBoundsAccessor(ds)

        # If coords dimensions does not equal 1.
        with pytest.raises(ValueError):
            obj._generate_bounds("lat")
        # If coords are length of <=1.
        with pytest.raises(ValueError):
            obj._generate_bounds("lon")

    def test__generate_bounds_returns_bounds(self):
        obj = DatasetBoundsAccessor(self.ds)

        lat_bnds = obj._generate_bounds("lat")
        assert lat_bnds.equals(self.lat_bnds)
        assert obj._dataset.lat_bnds.is_generated

        lon_bnds = obj._generate_bounds("lon")
        assert lon_bnds.equals(self.lon_bnds)
        assert obj._dataset.lon_bnds.is_generated

    def test__get_coord(self):
        obj = DatasetBoundsAccessor(self.ds)

        # Check lat axis coordinates exist
        lat = obj._get_coord("lat")
        assert lat is not None

        # Check lon axis coordinates exist
        lon = obj._get_coord("lon")
        assert lon is not None

    def test__get_coord_raises_error_if_coord_does_not_exist(self):
        obj = DatasetBoundsAccessor(self.ds)

        with pytest.raises(KeyError):
            obj._dataset = obj._dataset.drop_vars("lat")
            obj._get_coord("lat")


class TestDataArrayBoundsAccessor:
    @pytest.fixture(autouse=True)
    def setUp(self):
        self.ds = generate_dataset(has_bounds=True)
        self.ds.lat.attrs["bounds"] = "lat_bnds"
        self.ds.lon.attrs["bounds"] = "lon_bnds"
        self.ds.time.attrs["bounds"] = "time_bnds"

    def test_decorator_call(self):
        expected = self.ds.ts
        result = self.ds["ts"].bounds._dataarray
        assert result.identical(expected)

    def test__init__(self):
        expected = self.ds.ts
        result = DataArrayBoundsAccessor(self.ds["ts"])
        assert result._dataarray.identical(expected)

        assert result.lat is None
        assert result.lon is None
        assert result.time is None

    def test__copy_from_parent_copies_bounds(self):
        result = DataArrayBoundsAccessor(self.ds["ts"])
        result._copy_from_dataset(self.ds)

        assert self.ds.lat_bnds.identical(result.lat)
        assert self.ds.lon_bnds.identical(result.lon)
        assert self.ds.time_bnds.identical(result.time)
