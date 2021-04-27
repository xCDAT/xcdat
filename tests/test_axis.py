import numpy as np
import pytest
import xarray as xr

from xcdat.axis import AxisAccessor

# Static latitude and longitude bounds for DataArray bounds variables
LAT_BNDS = np.array(
    [
        [-90.0, -89.375],
        [-89.375, -88.125],
        [-88.125, -86.875],
        [-86.875, -85.625],
        [-85.625, -84.375],
        [-84.375, -83.125],
        [-83.125, -81.875],
        [-81.875, -80.625],
        [-80.625, -79.375],
        [-79.375, -78.125],
        [-78.125, -76.875],
        [-76.875, -75.625],
        [-75.625, -74.375],
        [-74.375, -73.125],
        [-73.125, -71.875],
        [-71.875, -70.625],
        [-70.625, -69.375],
        [-69.375, -68.125],
        [-68.125, -66.875],
        [-66.875, -65.625],
        [-65.625, -64.375],
        [-64.375, -63.125],
        [-63.125, -61.875],
        [-61.875, -60.625],
        [-60.625, -59.375],
        [-59.375, -58.125],
        [-58.125, -56.875],
        [-56.875, -55.625],
        [-55.625, -54.375],
        [-54.375, -53.125],
        [-53.125, -51.875],
        [-51.875, -50.625],
        [-50.625, -49.375],
        [-49.375, -48.125],
        [-48.125, -46.875],
        [-46.875, -45.625],
        [-45.625, -44.375],
        [-44.375, -43.125],
        [-43.125, -41.875],
        [-41.875, -40.625],
        [-40.625, -39.375],
        [-39.375, -38.125],
        [-38.125, -36.875],
        [-36.875, -35.625],
        [-35.625, -34.375],
        [-34.375, -33.125],
        [-33.125, -31.875],
        [-31.875, -30.625],
        [-30.625, -29.375],
        [-29.375, -28.125],
        [-28.125, -26.875],
        [-26.875, -25.625],
        [-25.625, -24.375],
        [-24.375, -23.125],
        [-23.125, -21.875],
        [-21.875, -20.625],
        [-20.625, -19.375],
        [-19.375, -18.125],
        [-18.125, -16.875],
        [-16.875, -15.625],
        [-15.625, -14.375],
        [-14.375, -13.125],
        [-13.125, -11.875],
        [-11.875, -10.625],
        [-10.625, -9.375],
        [-9.375, -8.125],
        [-8.125, -6.875],
        [-6.875, -5.625],
        [-5.625, -4.375],
        [-4.375, -3.125],
        [-3.125, -1.875],
        [-1.875, -0.625],
        [-0.625, 0.625],
        [0.625, 1.875],
        [1.875, 3.125],
        [3.125, 4.375],
        [4.375, 5.625],
        [5.625, 6.875],
        [6.875, 8.125],
        [8.125, 9.375],
        [9.375, 10.625],
        [10.625, 11.875],
        [11.875, 13.125],
        [13.125, 14.375],
        [14.375, 15.625],
        [15.625, 16.875],
        [16.875, 18.125],
        [18.125, 19.375],
        [19.375, 20.625],
        [20.625, 21.875],
        [21.875, 23.125],
        [23.125, 24.375],
        [24.375, 25.625],
        [25.625, 26.875],
        [26.875, 28.125],
        [28.125, 29.375],
        [29.375, 30.625],
        [30.625, 31.875],
        [31.875, 33.125],
        [33.125, 34.375],
        [34.375, 35.625],
        [35.625, 36.875],
        [36.875, 38.125],
        [38.125, 39.375],
        [39.375, 40.625],
        [40.625, 41.875],
        [41.875, 43.125],
        [43.125, 44.375],
        [44.375, 45.625],
        [45.625, 46.875],
        [46.875, 48.125],
        [48.125, 49.375],
        [49.375, 50.625],
        [50.625, 51.875],
        [51.875, 53.125],
        [53.125, 54.375],
        [54.375, 55.625],
        [55.625, 56.875],
        [56.875, 58.125],
        [58.125, 59.375],
        [59.375, 60.625],
        [60.625, 61.875],
        [61.875, 63.125],
        [63.125, 64.375],
        [64.375, 65.625],
        [65.625, 66.875],
        [66.875, 68.125],
        [68.125, 69.375],
        [69.375, 70.625],
        [70.625, 71.875],
        [71.875, 73.125],
        [73.125, 74.375],
        [74.375, 75.625],
        [75.625, 76.875],
        [76.875, 78.125],
        [78.125, 79.375],
        [79.375, 80.625],
        [80.625, 81.875],
        [81.875, 83.125],
        [83.125, 84.375],
        [84.375, 85.625],
        [85.625, 86.875],
        [86.875, 88.125],
        [88.125, 89.375],
        [89.375, 90.0],
    ]
)

LON_BNDS = np.array(
    [
        [-0.9375, 0.9375],
        [0.9375, 2.8125],
        [2.8125, 4.6875],
        [4.6875, 6.5625],
        [6.5625, 8.4375],
        [8.4375, 10.3125],
        [10.3125, 12.1875],
        [12.1875, 14.0625],
        [14.0625, 15.9375],
        [15.9375, 17.8125],
        [17.8125, 19.6875],
        [19.6875, 21.5625],
        [21.5625, 23.4375],
        [23.4375, 25.3125],
        [25.3125, 27.1875],
        [27.1875, 29.0625],
        [29.0625, 30.9375],
        [30.9375, 32.8125],
        [32.8125, 34.6875],
        [34.6875, 36.5625],
        [36.5625, 38.4375],
        [38.4375, 40.3125],
        [40.3125, 42.1875],
        [42.1875, 44.0625],
        [44.0625, 45.9375],
        [45.9375, 47.8125],
        [47.8125, 49.6875],
        [49.6875, 51.5625],
        [51.5625, 53.4375],
        [53.4375, 55.3125],
        [55.3125, 57.1875],
        [57.1875, 59.0625],
        [59.0625, 60.9375],
        [60.9375, 62.8125],
        [62.8125, 64.6875],
        [64.6875, 66.5625],
        [66.5625, 68.4375],
        [68.4375, 70.3125],
        [70.3125, 72.1875],
        [72.1875, 74.0625],
        [74.0625, 75.9375],
        [75.9375, 77.8125],
        [77.8125, 79.6875],
        [79.6875, 81.5625],
        [81.5625, 83.4375],
        [83.4375, 85.3125],
        [85.3125, 87.1875],
        [87.1875, 89.0625],
        [89.0625, 90.9375],
        [90.9375, 92.8125],
        [92.8125, 94.6875],
        [94.6875, 96.5625],
        [96.5625, 98.4375],
        [98.4375, 100.3125],
        [100.3125, 102.1875],
        [102.1875, 104.0625],
        [104.0625, 105.9375],
        [105.9375, 107.8125],
        [107.8125, 109.6875],
        [109.6875, 111.5625],
        [111.5625, 113.4375],
        [113.4375, 115.3125],
        [115.3125, 117.1875],
        [117.1875, 119.0625],
        [119.0625, 120.9375],
        [120.9375, 122.8125],
        [122.8125, 124.6875],
        [124.6875, 126.5625],
        [126.5625, 128.4375],
        [128.4375, 130.3125],
        [130.3125, 132.1875],
        [132.1875, 134.0625],
        [134.0625, 135.9375],
        [135.9375, 137.8125],
        [137.8125, 139.6875],
        [139.6875, 141.5625],
        [141.5625, 143.4375],
        [143.4375, 145.3125],
        [145.3125, 147.1875],
        [147.1875, 149.0625],
        [149.0625, 150.9375],
        [150.9375, 152.8125],
        [152.8125, 154.6875],
        [154.6875, 156.5625],
        [156.5625, 158.4375],
        [158.4375, 160.3125],
        [160.3125, 162.1875],
        [162.1875, 164.0625],
        [164.0625, 165.9375],
        [165.9375, 167.8125],
        [167.8125, 169.6875],
        [169.6875, 171.5625],
        [171.5625, 173.4375],
        [173.4375, 175.3125],
        [175.3125, 177.1875],
        [177.1875, 179.0625],
        [179.0625, 180.9375],
        [180.9375, 182.8125],
        [182.8125, 184.6875],
        [184.6875, 186.5625],
        [186.5625, 188.4375],
        [188.4375, 190.3125],
        [190.3125, 192.1875],
        [192.1875, 194.0625],
        [194.0625, 195.9375],
        [195.9375, 197.8125],
        [197.8125, 199.6875],
        [199.6875, 201.5625],
        [201.5625, 203.4375],
        [203.4375, 205.3125],
        [205.3125, 207.1875],
        [207.1875, 209.0625],
        [209.0625, 210.9375],
        [210.9375, 212.8125],
        [212.8125, 214.6875],
        [214.6875, 216.5625],
        [216.5625, 218.4375],
        [218.4375, 220.3125],
        [220.3125, 222.1875],
        [222.1875, 224.0625],
        [224.0625, 225.9375],
        [225.9375, 227.8125],
        [227.8125, 229.6875],
        [229.6875, 231.5625],
        [231.5625, 233.4375],
        [233.4375, 235.3125],
        [235.3125, 237.1875],
        [237.1875, 239.0625],
        [239.0625, 240.9375],
        [240.9375, 242.8125],
        [242.8125, 244.6875],
        [244.6875, 246.5625],
        [246.5625, 248.4375],
        [248.4375, 250.3125],
        [250.3125, 252.1875],
        [252.1875, 254.0625],
        [254.0625, 255.9375],
        [255.9375, 257.8125],
        [257.8125, 259.6875],
        [259.6875, 261.5625],
        [261.5625, 263.4375],
        [263.4375, 265.3125],
        [265.3125, 267.1875],
        [267.1875, 269.0625],
        [269.0625, 270.9375],
        [270.9375, 272.8125],
        [272.8125, 274.6875],
        [274.6875, 276.5625],
        [276.5625, 278.4375],
        [278.4375, 280.3125],
        [280.3125, 282.1875],
        [282.1875, 284.0625],
        [284.0625, 285.9375],
        [285.9375, 287.8125],
        [287.8125, 289.6875],
        [289.6875, 291.5625],
        [291.5625, 293.4375],
        [293.4375, 295.3125],
        [295.3125, 297.1875],
        [297.1875, 299.0625],
        [299.0625, 300.9375],
        [300.9375, 302.8125],
        [302.8125, 304.6875],
        [304.6875, 306.5625],
        [306.5625, 308.4375],
        [308.4375, 310.3125],
        [310.3125, 312.1875],
        [312.1875, 314.0625],
        [314.0625, 315.9375],
        [315.9375, 317.8125],
        [317.8125, 319.6875],
        [319.6875, 321.5625],
        [321.5625, 323.4375],
        [323.4375, 325.3125],
        [325.3125, 327.1875],
        [327.1875, 329.0625],
        [329.0625, 330.9375],
        [330.9375, 332.8125],
        [332.8125, 334.6875],
        [334.6875, 336.5625],
        [336.5625, 338.4375],
        [338.4375, 340.3125],
        [340.3125, 342.1875],
        [342.1875, 344.0625],
        [344.0625, 345.9375],
        [345.9375, 347.8125],
        [347.8125, 349.6875],
        [349.6875, 351.5625],
        [351.5625, 353.4375],
        [353.4375, 355.3125],
        [355.3125, 357.1875],
        [357.1875, 359.0625],
    ]
)


class TestAxisAccessor:
    @pytest.fixture(autouse=True)
    def setup(self):
        # Axes information
        lat = xr.DataArray(
            data=np.arange(-90, 91.25, 1.25),
            dims=["lat"],
            attrs={"units": "degrees_north", "axis": "Y"},
        )
        lon = xr.DataArray(
            data=np.arange(0, 360, 1.875),
            dims=["lon"],
            attrs={"units": "degrees_east", "axis": "x"},
        )

        self.lat_bnds = xr.DataArray(
            data=LAT_BNDS,
            coords={"lat": lat.data},
            dims=["lat", "bnds"],
            attrs={"units": "degrees_north", "is_generated": True},
        )
        self.lon_bnds = xr.DataArray(
            data=LON_BNDS,
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
        assert obj._obj.identical(self.ds)
        assert self.ds.identical(self.ds.axis._obj)

    def test_get_bounds_for_existing_bounds(self):
        obj = AxisAccessor(self.ds)

        # Assign bounds variables to replicate a DataSet that already has this info
        obj._obj = obj._obj.assign(lon_bnds=self.lon_bnds)
        lon_bnds = obj.get_bounds("lon")
        assert lon_bnds.identical(obj._obj.lon_bnds)

        obj._obj = obj._obj.assign(lat_bnds=self.lat_bnds)
        lat_bnds = obj.get_bounds("lat")
        assert lat_bnds.identical(obj._obj.lat_bnds)

    def test_get_bounds_for_calculated_bounds(self):
        # Preexisting boundary data for comparison assertions
        obj = AxisAccessor(self.ds)

        # Check calculates lat bounds
        lat_bnds = obj.get_bounds("lat", generate=True)
        lon_bnds = obj.get_bounds("lon", generate=True)

        assert lat_bnds.identical(self.lat_bnds)
        assert lon_bnds.identical(self.lon_bnds)

    def test_get_bounds_raises_value_error_if_bounds_are_nonexistent_and_don(self):
        # Preexisting boundary data for comparison assertions
        obj = AxisAccessor(self.ds)

        # Check calculates lat bounds
        lat_bnds = obj.get_bounds("lat", generate=True)
        lon_bnds = obj.get_bounds("lon", generate=True)

        assert lat_bnds.identical(self.lat_bnds)
        assert lon_bnds.identical(self.lon_bnds)

    def test_get_bounds_raises_errors(self):
        obj = AxisAccessor(self.ds)

        with pytest.raises(KeyError):
            # Raises error if incorrect axis param is passed
            obj.get_bounds("incorrect_axis")  # type: ignore

    def test__gen_bounds_for_short_axis_name(self):
        obj = AxisAccessor(self.ds)
        # Check calculated lat bounds equal existing lat bounds
        lat_bnds = obj._gen_bounds("lat")
        assert lat_bnds.equals(self.lat_bnds)
        assert obj._obj.lat_bnds.is_generated

        # # Check calculated lon bounds equal existing lon bounds
        lon_bnds = obj._gen_bounds("lon")
        assert lon_bnds.equals(self.lon_bnds)
        assert obj._obj.lon_bnds.is_generated

    def test__gen_bounds_for_long_axis_name(self):
        obj = AxisAccessor(self.ds_2)

        # Check calculated lat bounds equal existing lat bounds
        lat_bnds = obj._gen_bounds("lat")
        assert lat_bnds.equals(self.lat_bnds)
        assert obj._obj.lat_bnds.is_generated

        # Check calculated lon bounds equal existing lon bounds
        lon_bnds = obj._gen_bounds("lon")
        assert lon_bnds.equals(self.lon_bnds)
        assert obj._obj.lon_bnds.is_generated

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
            obj._obj = obj._obj.drop_vars("lat")
            obj._extract_axis_coords("lat")

    def test__gen_base_bounds(self):
        # result = AxisAccessor._gen_base_bounds(np.arange(0), 1)
        # print(result)
        pass

    def test__calc_lat_bounds(self):
        pass

    def test__calc_lon_bounds(self):
        pass
