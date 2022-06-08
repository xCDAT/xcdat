from typing import Tuple, Union

import numpy as np
import xarray as xr

# First 50 zeros for the bessel function
# Taken from https://github.com/CDAT/cdms/blob/dd41a8dd3b5bac10a4bfdf6e56f6465e11efc51d/regrid2/Src/_regridmodule.c#L3390-L3402
BESSEL_LOOKUP = [
    2.4048255577,
    5.5200781103,
    8.6537279129,
    11.7915344391,
    14.9309177086,
    18.0710639679,
    21.2116366299,
    24.3524715308,
    27.493479132,
    30.6346064684,
    33.7758202136,
    36.9170983537,
    40.0584257646,
    43.1997917132,
    46.3411883717,
    49.4826098974,
    52.6240518411,
    55.765510755,
    58.9069839261,
    62.0484691902,
    65.1899648002,
    68.3314693299,
    71.4729816036,
    74.6145006437,
    77.7560256304,
    80.8975558711,
    84.0390907769,
    87.1806298436,
    90.3221726372,
    93.4637187819,
    96.605267951,
    99.7468198587,
    102.8883742542,
    106.0299309165,
    109.1714896498,
    112.3130502805,
    115.4546126537,
    118.5961766309,
    121.737742088,
    124.8793089132,
    128.0208770059,
    131.1624462752,
    134.3040166383,
    137.4455880203,
    140.5871603528,
    143.7287335737,
    146.8703076258,
    150.011882457,
    153.1534580192,
    156.2950342685,
]

# Error used for legendre polinomial
# Taken from CDAT's regrid2 module https://github.com/CDAT/cdms/blob/3f8c7baa359f428628a666652ecf361764dc7b7a/regrid2/Src/_regridmodule.c#L3229
EPS = 1e-14
# Constant used in first estimate for legendre polinomial
# Taken from CDAT's regrid2 module https://github.com/CDAT/cdms/blob/3f8c7baa359f428628a666652ecf361764dc7b7a/regrid2/Src/_regridmodule.c#L3254-L3256
ESTIMATE_CONST = 0.25 * (1.0 - np.power(2.0 / np.pi, 2))


def create_gaussian_grid(nlats: int) -> xr.Dataset:
    """
    Creates a grid with Gaussian latitudes and uniform longitudes.

    Parameters
    ----------
    nlats : int
        Number of latitudes.

    Returns
    -------
    xr.Dataset
        Dataset with new grid, containing Gaussian latitudes.

    Examples
    --------
    Create grid with 32 latitudes:

    >>> xcdat.regridder.grid.create_gaussian_grid(32)
    """
    lat_bounds, lat_axis = _create_gaussian_axis(nlats)

    lon_axis = create_uniform_axis(
        0.0,
        360.0,
        (360.0 / (2.0 * nlats)),
        "lon",
        {"units": "degrees_east", "axis": "X"},
    )

    grid = xr.Dataset(
        {
            "lat": lat_axis,
            "lon": lon_axis,
        }
    )

    grid = grid.bounds.add_missing_bounds()

    grid["lat_bnds"] = lat_bounds

    return grid


def _create_gaussian_axis(nlats: int) -> Tuple[xr.DataArray, xr.DataArray]:
    """Create Gaussian axis.

    Creates a Gaussian axis of `nlats`.

    Parameters
    ----------
    nlats : int
        Number of latitude points.

    Returns
    -------
    xr.DataArray
        Containing the latitude bounds.

    xr.DataArray
        Containing the latitude axis.
    """
    mid = int(np.floor(nlats / 2))

    points, weights = _gaussian_axis(mid, nlats)

    bounds = np.zeros((nlats + 1))
    bounds[0], bounds[-1] = 1.0, -1.0

    for i in range(1, mid + 1):
        bounds[i] = bounds[i - 1] - weights[i - 1]
        bounds[nlats - i] = -bounds[i]

    points_data = (180.0 / np.pi) * np.arcsin(points)

    points_da = xr.DataArray(
        name="lat",
        data=points_data,
        dims=["lat"],
        attrs={
            "units": "degrees_north",
            "axis": "Y",
            "bounds": "lat_bnds",
        },
    )

    bounds = (180.0 / np.pi) * np.arcsin(bounds)

    bounds_data = np.zeros((points.shape[0], 2))
    bounds_data[:, 0] = bounds[:-1]
    bounds_data[:, 1] = bounds[1:]

    bounds_da = xr.DataArray(
        name="lat_bnds",
        data=bounds_data,
        dims=["lat", "bnds"],
        coords={
            "lat": points_da,
        },
    )

    return bounds_da, points_da


def _gaussian_axis(mid: int, nlats: int) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates the bounds and weights for a Guassian axis.


    Math is based on CDAT implementation in regrid2 module.

    Related documents:
        - https://github.com/CDAT/cdms/blob/cda99c21098ad01d88c27c6010d4affc8f621863/regrid2/Src/_regridmodule.c#L3310

    Parameters
    ----------
    mid : int
        mid
    nlats : int
        Number of latitude points to calculate.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        First `np.ndarray` contains the angles of the bounds and the second contains the weights.
    """
    points = _bessel_function_zeros(mid + 1)

    points = np.pad(points, (0, nlats - len(points)))

    weights = np.zeros_like(points)

    for x in range(int(nlats / 2 + 1)):
        zero_poly, first_poly, second_poly = _legendre_polinomial(points[x], nlats)

        points[x] = zero_poly

        weights[x] = (
            (1.0 - zero_poly * zero_poly)
            * 2.0
            / ((nlats * first_poly) * (nlats * first_poly))
        )

    points[mid if nlats % 2 == 0 else mid + 1 :] = -1.0 * np.flip(points[:mid])

    weights[mid if nlats % 2 == 0 else mid + 1 :] = np.flip(weights[:mid])

    if nlats % 2 != 0:
        weights[mid + 1] = 0.0

        mid_weight = 2.0 / np.power(nlats, 2)

        for x in range(2, nlats + 1, 2):
            mid_weight = (mid_weight * np.power(nlats, 2)) / np.power(x - 1, 2)

            weights[mid + 1] = mid_weight

    return points, weights


def _bessel_function_zeros(n: int) -> np.ndarray:
    """Zeros of Bessel function.

    Calculates `n` zeros of the Bessel function.

    Math is based on CDAT implementation in regrid2 module.

    Related documents:
        - https://en.wikipedia.org/wiki/Bessel_function
        - https://github.com/CDAT/cdms/blob/cda99c21098ad01d88c27c6010d4affc8f621863/regrid2/Src/_regridmodule.c#L3387
        - https://github.com/CDAT/cdms/blob/cda99c21098ad01d88c27c6010d4affc8f621863/regrid2/Src/_regridmodule.c#L3251

    Parameters
    ----------
    n : int
        Number of zeros to calculate.

    Returns
    -------
    np.ndarray
        Array containing `n` zeros of the Bessel function.
    """
    values = np.zeros(n)

    lookup_n = min(n, 50)

    values[:lookup_n] = BESSEL_LOOKUP[:lookup_n]

    # interpolate remaining values
    if n > 50:
        for x in range(50, n):
            values[x] = values[x - 1] + np.pi

    return values


def _legendre_polinomial(bessel_zero: int, nlats: int) -> Tuple[float, float, float]:
    """Legendre_polynomials.

    Calculates the third legendre polynomial.

    Math is based on CDAT implementation in regrid2 module.

    Related documents:
        - https://en.wikipedia.org/wiki/Legendre_polynomials
        - https://github.com/CDAT/cdms/blob/cda99c21098ad01d88c27c6010d4affc8f621863/regrid2/Src/_regridmodule.c#L3291

    Parameters
    ----------
    bessel_zero : int
        Bessel zero used to calculate the third legendre polynomial.
    nlats : int
        Number of lats.

    Returns
    -------
    Tuple[float, float, float]
        First, second and third legendre polynomial.
    """
    zero_poly = np.cos(bessel_zero / np.sqrt(np.power(nlats + 0.5, 2) + ESTIMATE_CONST))

    for _ in range(10):
        second_poly = 1.0
        first_poly = zero_poly

        for y in range(2, nlats + 1):
            new_poly = (
                (2.0 * y - 1.0) * zero_poly * first_poly - (y - 1.0) * second_poly
            ) / y
            second_poly = first_poly
            first_poly = new_poly

        first_poly = second_poly
        poly_change = (nlats * (first_poly - zero_poly * new_poly)) / (
            1.0 - zero_poly * zero_poly
        )
        poly_slope = new_poly / poly_change
        zero_poly -= poly_slope
        abs_poly_change = np.fabs(poly_slope)

        if abs_poly_change <= EPS:
            break

    return zero_poly, first_poly, second_poly


def create_uniform_axis(
    start: float, stop: float, delta: float, name: str, attrs: dict
) -> xr.DataArray:
    """Create a uniform axis.

    Parameters
    ----------
    start : float
        Starting latitude point.
    stop : float
        Stopping latitude point.
    delta : float
        Distance between points.
    name : str
        Name of the axis.
    attrs : dict
        Axis attributes.

    Returns
    -------
    xr.DataArray
        Containing uniform axis.
    """
    return xr.DataArray(
        name=name,
        data=np.arange(start, stop + delta / 2.0, delta),
        dims=[name],
        attrs=attrs,
    )


def create_uniform_grid(
    lat_start: float,
    lat_stop: float,
    lat_delta: float,
    lon_start: float,
    lon_stop: float,
    lon_delta: float,
) -> xr.Dataset:
    """
    Creates a uniform rectilinear grid and sets appropriate
    the attributes for the lat/lon axis.

    Parameters
    ----------
    lat_start : float
        First latitude.
    lat_stop : float
        Last latitude.
    lat_delta : float
        Difference between two points of axis.
    lon_start : float
        First longitude.
    lon_stop : float
        Last longitude.
    lon_delta : float
        Difference between two points of axis.

    Returns
    -------
    xr.Dataset
        Dataset with uniform lat/lon grid.

    Examples
    --------
    Create 4x5 uniform grid:

    >>> xcdat.regridder.grid.create_uniform_grid(-90, 90, 4.0, -180, 180, 5.0)
    """
    grid = xr.Dataset(
        coords={
            "lat": create_uniform_axis(
                lat_start,
                lat_stop,
                lat_delta,
                "lat",
                {"units": "degrees_north", "axis": "Y"},
            ),
            "lon": create_uniform_axis(
                lon_start,
                lon_stop,
                lon_delta,
                "lon",
                {"units": "degrees_east", "axis": "X"},
            ),
        },
    )

    grid = grid.bounds.add_missing_bounds()

    return grid


def create_global_mean_grid(grid: xr.Dataset) -> xr.Dataset:
    """Creates a global mean grid.

    Parameters
    ----------
    grid : xr.Dataset
        Source grid.

    Returns
    -------
    xr.Dataset
        A dataset containing the global mean grid.
    """
    lat = grid.cf["lat"]
    lat_bnd = grid.bounds.get_bounds("lat")

    out_lat_data = np.array([(lat[0] + lat[-1]) / 2.0])
    out_lat_bnds = np.array([[lat_bnd[0, 0], lat_bnd[-1, 1]]])

    lon = grid.cf["lon"]
    lon_bnd = grid.bounds.get_bounds("lon")

    out_lon_data = np.array([(lon[0] + lon[-1]) / 2.0])
    out_lon_bnds = np.array([[lon_bnd[0, 0], lon_bnd[-1, 1]]])

    lat_dataarray = xr.DataArray(
        name="lat",
        data=out_lat_data,
        dims=["lat"],
        attrs=grid.cf["lat"].attrs.copy(),
    )

    lon_dataarray = xr.DataArray(
        name="lon",
        data=out_lon_data,
        dims=["lon"],
        attrs=grid.cf["lon"].attrs.copy(),
    )

    return xr.Dataset(
        coords={
            "lat": lat_dataarray,
            "lon": lon_dataarray,
        },
        data_vars={
            "lat_bnds": xr.DataArray(
                name="lat_bnds",
                data=out_lat_bnds,
                dims=["lat", "bnds"],
                coords={
                    "lat": lat_dataarray,
                },
                attrs=grid.bounds.get_bounds("lat").attrs.copy(),
            ),
            "lon_bnds": xr.DataArray(
                name="lon_bnds",
                data=out_lon_bnds,
                dims=["lon", "bnds"],
                coords={
                    "lon": lon_dataarray,
                },
                attrs=grid.bounds.get_bounds("lon").attrs.copy(),
            ),
        },
    )


def create_zonal_grid(grid: xr.Dataset) -> xr.Dataset:
    """Creates a zonal grid.

    Parameters
    ----------
    grid : xr.Dataset
        Source grid.

    Returns
    -------
    xr.Dataset
        A dataset containing a zonal grid.
    """
    lon = grid.cf["lon"]
    lon_bnds = grid.bounds.get_bounds("lon")

    out_lon_data = np.array([(lon[0] + lon[-1]) / 2.0])
    out_lon_bnds = np.array([[lon_bnds[0, 0], lon_bnds[-1, 1]]])

    lat_dataarray = xr.DataArray(
        name="lat",
        data=grid.cf["lat"],
        dims=["lat"],
        attrs=grid.cf["lat"].attrs.copy(),
    )

    lon_dataarray = xr.DataArray(
        name="lon",
        data=out_lon_data,
        dims=["lon"],
        attrs=grid.cf["lon"].attrs.copy(),
    )

    return xr.Dataset(
        coords={
            "lat": lat_dataarray,
            "lon": lon_dataarray,
        },
        data_vars={
            "lat_bnds": xr.DataArray(
                name="lat_bnds",
                data=grid.bounds.get_bounds("lat"),
                dims=["lat", "bnds"],
                coords={
                    "lat": lat_dataarray,
                },
                attrs=grid.bounds.get_bounds("lat").attrs.copy(),
            ),
            "lon_bnds": xr.DataArray(
                name="lon_bnds",
                data=out_lon_bnds,
                dims=["lon", "bnds"],
                coords={
                    "lon": lon_dataarray,
                },
                attrs=grid.bounds.get_bounds("lon").attrs.copy(),
            ),
        },
    )


def create_grid(
    lat: Union[np.ndarray, xr.DataArray], lon: Union[np.ndarray, xr.DataArray]
) -> xr.Dataset:
    """Creates a grid for a give latitude and longitude array.

    Parameters
    ----------
    lat : Union[np.ndarray, xr.DataArray]
        Array of latitude values.
    lon : Union[np.ndarray, xr.DataArray]
        Array of longitude values.

    Returns
    -------
    xr.Dataset
        Dataset with lat/lon grid.

    Examples
    --------
    Create uniform 2.5 x 2.5 degree grid
    >>> import xcdat
    >>> import numpy as np
    >>> lat = np.arange(-90, 90, 2.5)
    >>> lon = np.arange(1.25, 360, 2.5)
    >>> xcdat.create_grid(lat, lon)
    """
    if isinstance(lat, np.ndarray):
        lat = xr.DataArray(
            name="lat",
            data=lat,
            dims=["lat"],
            attrs={"units": "degrees_north", "axis": "Y"},
        )

    if isinstance(lon, np.ndarray):
        lon = xr.DataArray(
            name="lon",
            data=lon,
            dims=["lon"],
            attrs={"units": "degrees_east", "axis": "X"},
        )

    grid = xr.Dataset(coords={"lat": lat, "lon": lon})

    grid = grid.bounds.add_missing_bounds()

    return grid
