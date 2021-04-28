import numpy as np
import pytest
import xarray as xr

from xcdat.axis import AxisAccessor


class TestAxisAccessor:
    @pytest.fixture(autouse=True)
    def setup(self):
        # Axes information
        lat = xr.DataArray(
            data=np.array([-90, -88.75, 88.75, 90]),
            dims=["lat"],
            attrs={"units": "degrees_north", "axis": "Y"},
        )
        lon = xr.DataArray(
            data=np.array([0, 1.875, 356.25, 358.125]),
            dims=["lon"],
            attrs={"units": "degrees_east", "axis": "x"},
        )

        # Generated axes boundaries for assertions
        self.lat_bnds = xr.DataArray(
            name="lat_bnds",
            data=np.array(
                [[-90, -89.375], [-89.375, 0.0], [0.0, 89.375], [89.375, 90]]
            ),
            coords={"lat": lat.data},
            dims=["lat", "bnds"],
            attrs={"units": "degrees_north", "is_generated": True},
        )
        self.lon_bnds = xr.DataArray(
            name="lon_bnds",
            data=np.array(
                [
                    [-0.9375, 0.9375],
                    [0.9375, 179.0625],
                    [179.0625, 357.1875],
                    [357.1875, 359.0625],
                ]
            ),
            coords={"lon": lon.data},
            dims=["lon", "bnds"],
            attrs={"units": "degrees_east", "is_generated": True},
        )

        # Create Dataset using coordinates (for testing short axes names)
        self.ds = xr.Dataset(coords={"lat": lat, "lon": lon})

        # Create Dataset using coordinates (for testing long axes names)
        self.ds_2 = xr.Dataset(
            coords={"latitude": self.ds.lat, "longitude": self.ds.lon}
        )
        self.ds_2 = self.ds_2.swap_dims({"lat": "latitude", "lon": "longitude"})

    def test__init__(self):
        obj = AxisAccessor(self.ds)
        assert obj._dataset.identical(self.ds)

    def test_decorator_call(self):
        assert self.ds.axis._dataset.identical(self.ds)

    def test_get_bounds_when_bounds_exist_in_dataset(self):
        obj = AxisAccessor(self.ds)
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
        obj = AxisAccessor(self.ds)

        lat_bnds = obj.get_bounds("lat")
        assert lat_bnds is not None and lat_bnds.identical(self.lat_bnds)
        assert lat_bnds.is_generated

        # Check raises error when bounds do not exist and not allowing generated bounds.
        with pytest.raises(ValueError):
            obj._dataset = obj._dataset.drop_vars(["lat_bnds"])
            obj.get_bounds("lat", allow_generate=False)

    def test_get_bounds_raises_error_with_incorrect_axis_argument(self):
        obj = AxisAccessor(self.ds)

        with pytest.raises(KeyError):
            obj.get_bounds("incorrect_axis_argument")  # type: ignore

    def test__gen_bounds_raises_errors(self):
        obj = AxisAccessor(self.ds)

        # Dataset axis coordinates variable is missing "units" attr
        with pytest.raises(TypeError):
            obj._dataset.lat.attrs.pop("units")
            obj._gen_bounds("lat")

        # Dataset axis coordinates variable "units" attr does not contain "degree" substring
        with pytest.raises(ValueError):
            obj._dataset.lat.attrs["units"] = "incorrect_value"
            obj._gen_bounds("lat")

    def test__gen_bounds_for_short_axis_name(self):
        obj = AxisAccessor(self.ds)

        lat_bnds = obj._gen_bounds("lat")
        assert lat_bnds.equals(self.lat_bnds)
        assert obj._dataset.lat_bnds.is_generated

        lon_bnds = obj._gen_bounds("lon")
        assert lon_bnds.equals(self.lon_bnds)
        assert obj._dataset.lon_bnds.is_generated

    def test__gen_bounds_for_long_axis_name(self):
        obj = AxisAccessor(self.ds_2)

        lat_bnds = obj._gen_bounds("lat")
        assert lat_bnds.equals(self.lat_bnds)
        assert obj._dataset.lat_bnds.is_generated

        lon_bnds = obj._gen_bounds("lon")
        assert lon_bnds.equals(self.lon_bnds)
        assert obj._dataset.lon_bnds.is_generated

    def test__extract_axis_coords(self):
        obj = AxisAccessor(self.ds)

        # Check lat axis coordinates exist
        lat = obj._extract_axis_coords("lat")
        assert lat is not None

        # Check lon axis coordinates exist
        lon = obj._extract_axis_coords("lon")
        assert lon is not None

    def test__extract_axis_coords_raises_errors(self):
        obj = AxisAccessor(self.ds)

        # Raises error if incorrect `axis` param is passed
        with pytest.raises(KeyError):
            obj._extract_axis_coords("incorrect_axis")  # type: ignore

        # Raises error if `axis` param has no matching coords in the Dataset
        with pytest.raises(TypeError):
            obj._dataset = obj._dataset.drop_vars("lat")
            obj._extract_axis_coords("lat")

    def test__gen_base_bounds(self):
        obj = AxisAccessor(self.ds)

        # If len(data) > 1
        data = np.arange(0, 5)
        bounds = np.arange(-0.5, 5, 1)
        expected = np.array(list(zip(*(bounds[i:] for i in range(2)))))

        result = obj._gen_base_bounds(data, width=1)
        assert np.array_equal(result, expected)

        # If len(data) <= 1 and width = 1
        data = np.array([0])
        bounds = np.array([-0.5, 0.5])
        expected = np.array(list(zip(*(bounds[i:] for i in range(2)))))

        result = obj._gen_base_bounds(data, width=1)
        assert np.array_equal(result, expected)

        # If len(data) <= 1 and width = 2
        data = np.array([0])
        bounds = np.array([-1, 1])
        expected = np.array(list(zip(*(bounds[i:] for i in range(2)))))

        result = obj._gen_base_bounds(data, width=2)
        assert np.array_equal(result, expected)

    def test__adjust_lat_bounds_cap_at_90_degrees(self):
        obj = AxisAccessor(self.ds)

        # Generate base lat bounds
        base_bounds = np.array([[-91.0, -88.75], [88.75, 91.0]])

        # Adjust expected result by capping both ends at (-90, 90)
        expected = base_bounds.copy()
        expected[0][0] = -90
        expected[-1][1] = 90

        # Checks adjustments made to cap lat bounds to (-90, 90)
        result = obj._adjust_lat_bounds(base_bounds)
        assert np.array_equal(result, expected)

    def test__adjust_lat_bounds_within_cap_limit(self):
        obj = AxisAccessor(self.ds)

        # Generate expected lat bounds within (-90, 90) limits
        expected = np.array([[-60.0, -58.75], [57.5, 58.75]])

        # Check no adjustments should be made since the bounds are within the cap limit
        result = obj._adjust_lat_bounds(expected)
        assert np.array_equal(result, expected)

    def test__adjust_lon_bounds_when_bounds_not_near_max_degree(
        self,
    ):
        obj = AxisAccessor(self.ds)

        # Generate expected lat bounds
        base_bounds = np.array([[0.0, 1.875], [356.25, 358.125]])
        # Checks no adjustments made since bounds aren't near the max degree
        result = obj._adjust_lon_bounds(base_bounds)
        assert np.array_equal(result, base_bounds)

    def test__adjust_lon_bounds_when_bounds_near_max_degree_and_near_int_value(
        self,
    ):
        obj = AxisAccessor(self.ds)

        # Generate expected lat bounds
        base_bounds = np.array([[-180.0, -178.125], [178.125, 180.0]])
        # Checks adjustments made to cap lat bounds to (-90, 90)
        expected = np.array([[-180.0, 180], [178.125, 180.0]])
        result = obj._adjust_lon_bounds(base_bounds)
        assert np.array_equal(result, expected)

    def test__adjust_lon_bounds_when_bounds_near_max_degree_and_not_near_int_value(
        self,
    ):
        obj = AxisAccessor(self.ds)

        # If max bound right val > min bound left value
        base_bounds = np.array([[-360.0001, -359.9], [-0.1, 0]])
        # Checks adjustments made to ensure circularity
        expected = np.array([[-360.0001, -359.9], [-0.1, -0.0001]])
        result = obj._adjust_lon_bounds(base_bounds)
        assert np.array_equal(result, expected)

        # If max bound right val <= min bound left value
        base_bounds = np.array([[0, -0.1], [-359.9, -360.0001]])
        # Checks adjustments made to ensure circularity
        expected = np.array([[-0.0001, -0.1], [-359.9, -360.0001]])
        result = obj._adjust_lon_bounds(base_bounds)
        assert np.array_equal(result, expected)
