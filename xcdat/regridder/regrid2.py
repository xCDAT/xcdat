from typing import List, Tuple

import numpy as np
import xarray as xr

from xcdat.regridder.base import BaseRegridder


def extract_bounds(bounds: xr.DataArray) -> Tuple[xr.DataArray, xr.DataArray]:
    """Extract bounds.

    Extract lower and upper bounds from an axis.

    Parameters
    ----------
    bounds : xr.DataArray
        Dataset containing axis with bounds.

    Returns
    -------
    xr.DataArray
        Contains the lower bounds for the axis.

    xr.DataArray
        Contains the upper bounds for the axis.
    """
    if bounds[0, 0] < bounds[0, 1]:
        lower = bounds[:, 0]
        upper = bounds[:, 1]
    else:
        lower = bounds[:, 1]
        upper = bounds[:, 0]

    return lower, upper


def map_latitude(src: xr.DataArray, dst: xr.DataArray) -> Tuple[List, List]:
    """Map source to destination latitude.

    Parameters
    ----------
    src : xr.DataArray
        DataArray containing the source latitude axis.
    dst : xr.DataArray
        DataArray containing the destination latitude axis.

    Returns
    -------
    List
        Containing map of destination to source points.

    List
        Containing map of destination to source weights.

    """
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

        weights.append(weight.values.reshape(contrib.shape[0], 1))

    return mapping, weights


def pertub(value):
    """Pertub a valu.

    Parameters
    ----------
    value :
        Value to pertub.
    """
    if value >= 0.0:
        offset = np.ceil(value + 0.000001)
    else:
        offset = np.floor(value - 0.000001) + 1.0

    return offset


# vectorize version of pertub
vpertub = np.vectorize(pertub)


def align_axis(
    src_west: xr.DataArray, src_east: xr.DataArray, dst_west: xr.DataArray
) -> Tuple[xr.DataArray, xr.DataArray, int]:
    """Align source to destination axis.

    Parameters
    ----------
    src_west : xr.DataArray
        DataArray containing the western source bounds.
    src_east : xr.DataArray
        DataArray containing the eastern source bounds.
    dst_west : xr.DataArray
        DataArray containing the western destination bounds.

    Returns
    -------
    Tuple[xr.DataArray, xr.DataArray, int]
    xr.DataArray
        Containing the shifted western source bounds.

    xr.DataArray
        Containing the shifted eastern source bounds.

    int
        Number of places shifted to align axis.
    """
    west_most = np.minimum(dst_west[0], dst_west[-1])

    alignment_index = pertub((west_most - src_west[-1]) / 360.0).values

    if src_west[0] < src_west[-1]:
        alignment_index += 1
    else:
        alignment_index -= 1

    src_alignment_index = np.where(
        vpertub((west_most - src_west) / 360.0) != alignment_index
    )[0][0]

    if src_west[0] < src_west[-1]:
        if west_most == src_west[src_alignment_index]:
            shift = src_alignment_index
        else:
            shift = src_alignment_index - 1

            if shift < 0:
                shift = src_west.shape[0] - 1
    else:
        shift = src_alignment_index

    src_length = src_west.shape[0]

    shifted_indexes = np.arange(src_length + 1) + shift

    wrapped = np.where(shifted_indexes > src_length - 1)

    shifted_indexes[wrapped] -= src_length

    shifted_src_west = src_west[shifted_indexes] + 360.0 * vpertub(
        (west_most - src_west[shifted_indexes]) / 360.0
    )

    shifted_src_east = src_east[shifted_indexes] + 360.0 * vpertub(
        (west_most - src_west[shifted_indexes]) / 360.0
    )

    if src_west[-1] > src_west[0]:
        if shifted_src_west[0] > west_most:
            shifted_src_west[0] += -360.0
            shifted_src_east[0] += -360.0
    else:
        if shifted_src_west[-1] > west_most:
            shifted_src_west[-1] += -360.0
            shifted_src_east[-1] += -360.0

    return shifted_src_west, shifted_src_east, shift


def map_longitude(src: xr.DataArray, dst: xr.DataArray) -> Tuple[List, List]:
    """Map source to destination longitude.

    Parameters
    ----------
    src : xr.DataArray
        DataArray containing source longitude axis.
    dst : xr.DataArray
        DataArray containing destination longitude axis.

    Returns
    -------
    List
        Contains mapping between axis.

    List
        Contains map of weights.

    """
    src_west, src_east = extract_bounds(src)
    dst_west, dst_east = extract_bounds(dst)

    shifted_src_west, shifted_src_east, shift = align_axis(src_west, src_east, dst_west)

    mapping = []
    weights = []
    src_length = src_west.shape[0]

    for i in range(dst_west.shape[0]):
        contrib = np.where(
            np.logical_and(
                shifted_src_west < dst_east[i], shifted_src_east > dst_west[i]
            )
        )[0]

        weight = np.minimum(dst_east[i], shifted_src_east[contrib]) - np.maximum(
            dst_west[i], shifted_src_west[contrib]
        )

        weights.append(weight.values.reshape(1, contrib.shape[0]))

        contrib += shift

        wrapped = np.where(contrib > src_length - 1)

        contrib[wrapped] -= src_length

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

    def __init__(self, src_grid: xr.Dataset, dst_grid: xr.Dataset, **options):
        super().__init__(src_grid, dst_grid, **options)

        src_lat = src_grid.cf.get_bounds("lat")
        self.dst_lat = dst_grid.cf.get_bounds("lat")

        self.lat_mapping, self.lat_weights = map_latitude(src_lat, self.dst_lat)

        src_lon = src_grid.cf.get_bounds("lon")
        self.dst_lon = dst_grid.cf.get_bounds("lon")

        self.lon_mapping, self.lon_weights = map_longitude(src_lon, self.dst_lon)

    def regrid(self, data_var: str, ds: xr.Dataset) -> xr.Dataset:
        input_data_var = ds.get(data_var, None)

        if input_data_var is None:
            raise KeyError(
                f"The data variable '{data_var}' does not exist in the dataset."
            )

        input_data = input_data_var.values

        axes = {x: y[0] for x, y in input_data_var.cf.axes.items()}

        invert_axes = {y: x for x, y in axes.items()}

        ordered_names = list(input_data_var.sizes)

        ordered_axes = sorted(
            invert_axes.items(), key=lambda x: ordered_names.index(x[0])
        )

        ordered_sizes = {}

        for x, y in ordered_axes:
            if "T" == y or "Z" == y:
                ordered_sizes[y] = input_data_var.sizes[x]
            else:
                ordered_sizes[y] = self._dst_grid.sizes[x]

        ordered_cf_names = list(ordered_sizes)

        output_shape = tuple(ordered_sizes.values())

        output_data = np.zeros(output_shape, dtype=np.float32)

        extra_dims = [x for x in set(ordered_cf_names) - set(["X", "Y"])]

        extra_dims_sizes = [ordered_sizes[x] for x in extra_dims]

        base_dims = [x for x in set(extra_dims) ^ set(ordered_cf_names)]

        base_dims_sizes = [ordered_sizes[x] for x in base_dims]

        number_of_offsets = np.multiply.reduce(extra_dims_sizes)

        offset = np.multiply.reduce(base_dims_sizes)

        base_put_index = (np.arange(number_of_offsets) * offset).astype(np.int64)

        input_lat_index, input_lon_index = ordered_cf_names.index(
            "Y"
        ), ordered_cf_names.index("X")

        # TODO handle lat x lon, lon x lat and height
        for lat_index, lat_map in enumerate(self.lat_mapping):
            lat_weight = self.lat_weights[lat_index]

            input_lat_segment = np.take(input_data, lat_map, axis=input_lat_index)

            for lon_index, lon_map in enumerate(self.lon_mapping):
                lon_weight = self.lon_weights[lon_index]

                dot_weight = np.dot(lat_weight, lon_weight)

                cell_weight = np.sum(dot_weight)

                input_lon_segment = np.take(
                    input_lat_segment, lon_map, axis=input_lon_index
                )

                data = (
                    np.multiply(input_lon_segment, dot_weight).sum(
                        axis=(input_lat_index, input_lon_index)
                    )
                    / cell_weight
                )

                put_index = base_put_index + (
                    (lat_index * ordered_sizes["X"]) + lon_index
                )

                np.put(output_data, put_index, data)

        coords = {}
        data_vars = {}

        for cf_name, name in axes.items():
            if cf_name in ["X", "Y"]:
                coords[name] = self._dst_grid[name]

                bounds = self._dst_grid.cf.get_bounds(cf_name)
            else:
                coords[name] = input_data_var[name]

                bounds = ds.cf.get_bounds(name)

            data_vars[bounds.name] = bounds

        output_da = xr.DataArray(
            output_data,
            dims=[axes[x] for x in ordered_cf_names],
            coords=coords,
        )

        data_vars[input_data_var.name] = output_da

        output_ds = xr.Dataset(data_vars)

        return output_ds
