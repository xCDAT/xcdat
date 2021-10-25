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
    grid = xr.Dataset(
        coords=dict(
            lat=("lat", np.arange(lat_start, lat_stop, lat_delta)),
            lon=("lon", np.arange(lon_start, lon_stop, lon_delta)),
        )
    )

    grid.lat.attrs["units"] = "degrees_east"

    grid.lon.attrs["units"] = "degrees_north"

    grid = grid.bounds.fill_missing()

    return grid
