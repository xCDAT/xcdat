from typing import List, Optional, Tuple

import numpy as np
import xarray as xr

from xcdat.dataset import get_inferred_var
from xcdat.regridder.base import BaseRegridder


def extract_bounds(bounds):
    if bounds[0, 0] < bounds[0, 1]:
        lower = bounds[:, 0]
        upper = bounds[:, 1]
    else:
        lower = bounds[:, 1]
        upper = bounds[:, 0]

    return lower, upper


def map_latitude(src: xr.DataArray, dst: xr.DataArray) -> Tuple[List, List]:
    src_south, src_north = extract_bounds(src)
    dst_south, dst_north = extract_bounds(dst)

    mapping = []
    weights = []

    for i in range(dst.shape[0]):
        contrib = np.where(
            np.logical_and(src_south < dst_north[i], src_north > dst_south[i])
        )[0]

        mapping.append(contrib)

        north_bounds = np.minimum(dst_north[i], src_north[contrib])
        south_bounds = np.maximum(dst_south[i], src_south[contrib])

        weight = np.sin(np.deg2rad(north_bounds)) - np.sin(np.deg2rad(south_bounds))

        weights.append(weight)

    return mapping, weights


def pertub(value):
    if value >= 0.0:
        offset = np.ceil(value + 0.000001)
    else:
        offset = np.floor(value - 0.000001) + 1.0

    return offset


def get_center_index(src_west, src_east, dst_west) -> Tuple[np.ndarray, np.ndarray]:
    west_most = np.minimum(dst_west[0], dst_west[-1])

    center = pertub((west_most - src_west[-1]) / 360.0)

    if src_west[0] < src_west[-1]:
        center += 1
    else:
        center -= 1

    return (
        west_most,
        np.where(np.logical_and(src_west < center, src_east > center))[0][0] - 1,
    )


def shift_bounds(src_west, src_east, west_most, center_index):
    src_length = len(src_west)

    new_west_index = np.arange(src_length + 1) + center_index

    require_adjust = np.where(new_west_index >= src_length)

    new_west_index[require_adjust] -= src_length

    vectorized_pertub = np.vectorize(pertub)

    value_shift = 360.0 * vectorized_pertub(
        (west_most - src_west[new_west_index]) / 360.0
    )

    new_src_west = src_west[new_west_index] + value_shift
    new_src_east = src_east[new_west_index] + value_shift

    if src_west[-1] > src_west[0]:
        if new_src_west[0] > west_most:
            new_src_west[0] += -360.0
            new_src_east[0] += -360.0
    else:
        if new_src_west[-1] > west_most:
            new_src_west[-1] += -360.0
            new_src_east[-1] += -360.0

    return new_src_west, new_src_east


def map_longitude(src: xr.DataArray, dst: xr.DataArray) -> Tuple[List, List]:
    src_east, src_west = extract_bounds(src)
    dst_east, dst_west = extract_bounds(dst)

    src_length = len(src_west)

    west_most, center_index = get_center_index(src_west, src_east, dst_west)

    new_src_west, new_src_east = shift_bounds(
        src_west, src_east, west_most, center_index
    )

    mapping = []
    weights = []

    for i in range(dst_west.shape[0]):
        contrib = np.where(
            np.logical_and(new_src_west < dst_east[i], new_src_east > dst_west[i])
        )[0]

        weight = np.minimum(dst_east[i], new_src_east[contrib]) - np.maximum(
            dst_west[i], new_src_west[contrib]
        )

        weights.append(weight)

        contrib += center_index

        values_wrapped = contrib > src_length - 1

        contrib[values_wrapped] -= src_length

        mapping.append(contrib)

    return mapping, weights


class Regrid2Regridder(BaseRegridder):
    """Regrid2 regridder class.

    Parameters
    ----------
    src_grid : xr.Dataset
        Contains source grid coordinates.
    dst_grid : xr.Dataset
        Contains desintation grid coordinates.
    """

    def __init__(self, src_grid: xr.Dataset, dst_grid: xr.Dataset):
        src_lat = src_grid.cf.get_bounds("lat")
        self.dst_lat = dst_grid.cf.get_bounds("lat")

        self.lat_mapping, self.lat_weights = map_latitude(src_lat, self.dst_lat)

        src_lon = src_grid.cf.get_bounds("lon")
        self.dst_lon = dst_grid.cf.get_bounds("lon")

        self.lon_mapping, self.lon_weights = map_longitude(src_lon, self.dst_lon)

    def regrid(self, ds: xr.Dataset, data_var: Optional[str] = None) -> xr.Dataset:
        if data_var is None:
            da_data_var = get_inferred_var(ds)
        else:
            da_data_var = ds.get(data_var, None)

            if da_data_var is None:
                raise KeyError(
                    f"The data variable '{data_var}' does not exist in the dataset."
                )

        output_shape = (
            da_data_var.cf["time"].shape[0],
            len(self.lat_mapping),
            len(self.lon_mapping),
        )

        out_data = np.zeros(output_shape)

        for ilat, lat in enumerate(self.lat_mapping):
            wtlat = self.lat_weights[ilat]
            wtlat = wtlat.data.reshape(len(wtlat), 1)

            for ilon, lon in enumerate(self.lon_mapping):
                wtlon = self.lon_weights[ilon]
                wtlon = wtlon.data.reshape(1, len(wtlon))

                weight_dot = np.dot(wtlat, wtlon)

                weight = weight_dot.sum()

                # handle case when order might be time, lon, lat?
                data = da_data_var[:, lat, lon]

                out_data[:, ilat, ilon] = (
                    np.multiply(data, weight_dot).sum(axis=1).sum(axis=1) / weight
                )

        coords = {
            "time": da_data_var["time"],
            "lat": (self.dst_lat[:, 0] + self.dst_lat[:, 1]) / 2,
            "lon": (self.dst_lon[:, 0] + self.dst_lon[:, 1]) / 2,
        }

        out_da = xr.DataArray(out_data, coords=coords, name=da_data_var.name)

        out_ds = xr.Dataset({da_data_var.name: out_da})

        return out_ds
