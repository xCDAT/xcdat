import numpy as np
import xarray as xr


def create_uniform_grid(
    lat_start: float,
    lat_stop: float,
    lat_delta: float,
    lon_start: float,
    lon_stop: float,
    lon_delta: float,
) -> xr.Dataset:
    """
    Creates a uniform rectilinear grid. Sets appropriate attributes
    for lat/lon axis.

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
        coords=dict(
            lat=("lat", np.arange(lat_start, lat_stop, lat_delta)),
            lon=("lon", np.arange(lon_start, lon_stop, lon_delta)),
        )
    )

    grid.lat.attrs["units"] = "degrees_north"

    grid.lon.attrs["units"] = "degrees_east"

    grid = grid.bounds.fill_missing()

    return grid


def create_gaussian_grid(nlats: int) -> xr.Dataset:
    raise NotImplementedError()


def create_global_mean_grid(grid: xr.Dataset) -> xr.Dataset:
    raise NotImplementedError()


def create_zonal_grid(grid: xr.Dataset) -> xr.Dataset:
    raise NotImplementedError()
