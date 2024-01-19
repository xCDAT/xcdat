from typing import Any, List, Tuple

import numpy as np
import xarray as xr

from xcdat.regridder.base import BaseRegridder, _preserve_bounds


class Regrid2Regridder(BaseRegridder):
    def __init__(self, input_grid: xr.Dataset, output_grid: xr.Dataset, **options: Any):
        """
        Pure python implementation of the regrid2 horizontal regridder from
        CDMS2's regrid2 module.

        Regrid data from ``input_grid`` to ``output_grid``.

        Available options: None

        Parameters
        ----------
        input_grid : xr.Dataset
            Dataset containing the source grid.
        output_grid : xr.Dataset
            Dataset containing the destination grid.
        options : Any
            Dictionary with extra parameters for the regridder.

        Examples
        --------

        Import xCDAT:

        >>> import xcdat

        Open a dataset:

        >>> ds = xcdat.open_dataset("...")

        Create output grid:

        >>> output_grid = xcdat.create_gaussian_grid(32)

        Regrid data:

        >>> output_data = ds.regridder.horizontal("ts", output_grid)
        """
        super().__init__(input_grid, output_grid, **options)

    def vertical(self, data_var: str, ds: xr.Dataset) -> xr.Dataset:
        """Placeholder for base class."""
        raise NotImplementedError()

    def horizontal(self, data_var: str, ds: xr.Dataset) -> xr.Dataset:
        """See documentation in :py:func:`xcdat.regridder.regrid2.Regrid2Regridder`"""
        try:
            input_data_var = ds[data_var]
        except KeyError:
            raise KeyError(
                f"The data variable {data_var!r} does not exist in the dataset."
            ) from None

        src_lat_bnds = _get_bounds_ensure_dtype(self._input_grid, "Y")
        src_lon_bnds = _get_bounds_ensure_dtype(self._input_grid, "X")

        dst_lat_bnds = _get_bounds_ensure_dtype(self._output_grid, "Y")
        dst_lon_bnds = _get_bounds_ensure_dtype(self._output_grid, "X")

        src_mask = self._input_grid.get("mask", None)

        # apply source mask to input data
        if src_mask is not None:
            input_data_var = input_data_var.where(src_mask != 0.0)

        output_data = _regrid(
            input_data_var, src_lat_bnds, src_lon_bnds, dst_lat_bnds, dst_lon_bnds
        )

        output_ds = _build_dataset(
            ds,
            data_var,
            output_data,
            dst_lat_bnds,
            dst_lon_bnds,
            self._input_grid,
            self._output_grid,
        )

        return output_ds


def _regrid(
    input_data_var: xr.DataArray,
    src_lat_bnds: np.ndarray,
    src_lon_bnds: np.ndarray,
    dst_lat_bnds: np.ndarray,
    dst_lon_bnds: np.ndarray,
) -> np.ndarray:
    lat_mapping, lat_weights = _map_latitude(src_lat_bnds, dst_lat_bnds)
    lon_mapping, lon_weights = _map_longitude(src_lon_bnds, dst_lon_bnds)

    # convert to pure numpy
    input_data = input_data_var.astype(np.float32).data

    y_name, y_index = _get_dimension(input_data_var, "Y")
    x_name, x_index = _get_dimension(input_data_var, "X")

    y_length = len(lat_mapping)
    x_length = len(lon_mapping)

    other_dims = {
        x: y for x, y in input_data_var.sizes.items() if x not in (y_name, x_name)
    }
    other_sizes = list(other_dims.values())

    data_shape = [y_length * x_length] + other_sizes
    output_data = np.zeros(data_shape, dtype=np.float32)

    is_2d = input_data_var.ndim <= 2

    # need to optimize
    for y in range(y_length):
        y_seg = np.take(input_data, lat_mapping[y], axis=y_index)

        for x in range(x_length):
            x_seg = np.take(y_seg, lon_mapping[x], axis=x_index, mode="wrap")

            cell_weight = np.dot(lat_weights[y], lon_weights[x])

            output_seg_index = y * x_length + x

            if is_2d:
                output_data[output_seg_index] = np.divide(
                    np.sum(
                        np.multiply(x_seg, cell_weight),
                        axis=(y_index, x_index),
                    ),
                    np.sum(cell_weight),
                )
            else:
                output_seg = output_data[output_seg_index]

                np.divide(
                    np.sum(
                        np.multiply(x_seg, cell_weight),
                        axis=(y_index, x_index),
                    ),
                    np.sum(cell_weight),
                    out=output_seg,
                )

    output_data_shape = [y_length, x_length] + other_sizes

    output_data = output_data.reshape(output_data_shape)

    output_order = [x + 2 for x in range(input_data_var.ndim - 2)] + [0, 1]

    output_data = output_data.transpose(output_order)

    return output_data.astype(np.float32)


def _build_dataset(
    ds: xr.Dataset,
    data_var: str,
    output_data: np.ndarray,
    dst_lat_bnds,
    dst_lon_bnds,
    input_grid: xr.Dataset,
    output_grid: xr.Dataset,
) -> xr.Dataset:
    input_data_var = ds[data_var]

    output_coords: dict[str, xr.DataArray] = {}
    output_data_vars: dict[str, xr.DataArray] = {}
    output_bnds = {
        "Y": dst_lat_bnds,
        "X": dst_lon_bnds,
    }

    dims = list(input_data_var.dims)

    for cf_axis_name, dim_names in input_data_var.cf.axes.items():
        dim_name = dim_names[0]

        if cf_axis_name in ("X", "Y"):
            output_grid_dim_name = output_grid.cf.axes[cf_axis_name][0]

            output_coords[output_grid_dim_name] = output_grid[
                output_grid_dim_name
            ].copy()

            for i, x in enumerate(dims):
                if x == dim_name:
                    dims[i] = output_grid_dim_name

                    break

            bnds_name = f"{output_grid_dim_name}_bnds"

            output_data_vars[bnds_name] = xr.DataArray(
                output_bnds[cf_axis_name].copy(),
                dims=(dim_name, "bnds"),
                name=bnds_name,
            )
        else:
            output_coords[dim_name] = input_data_var[dim_name].copy()

            bnds_name = input_data_var[dim_name].attrs.get("bnds", None)

            if bnds_name is not None:
                output_data_vars[bnds_name] = input_data_var[bnds_name].copy()

    output_da = xr.DataArray(
        output_data,
        dims=dims,
        coords=output_coords,
        attrs=ds[data_var].attrs.copy(),
        name=data_var,
    )

    output_data_vars[data_var] = output_da

    output_ds = xr.Dataset(
        output_data_vars,
        attrs=input_grid.attrs.copy(),
    )

    output_ds = _preserve_bounds(ds, output_grid, output_ds, ["X", "Y"])

    return output_ds


def _map_latitude(src: np.ndarray, dst: np.ndarray) -> Tuple[List, List]:
    """
    Map source to destination latitude.

    Parameters
    ----------
    src : np.ndarray
        DataArray containing the source latitude bounds.
    dst : np.ndarray
        DataArray containing the destination latitude bounds.

    Returns
    -------
    Tuple[List, List]
        A tuple of cell mappings and cell weights.
    """
    src_south, src_north = _extract_bounds(src)
    dst_south, dst_north = _extract_bounds(dst)

    dst_length = dst_south.shape[0]

    mapping = [
        np.where(np.logical_and(src_south < dst_north[x], src_north > dst_south[x]))[0]
        for x in range(dst_length)
    ]

    bounds = [
        (np.minimum(dst_north[x], src_north[y]), np.maximum(dst_south[x], src_south[y]))
        for x, y in enumerate(mapping)
    ]

    weights = [
        (np.sin(np.deg2rad(x)) - np.sin(np.deg2rad(y))).reshape((-1, 1))
        for x, y in bounds
    ]

    return mapping, weights


def _map_longitude(src: np.ndarray, dst: np.ndarray) -> Tuple[List, List]:
    """
    Map source to destination longitude.

    Parameters
    ----------
    src : np.ndarray
        DataArray containing source longitude bounds.
    dst : np.ndarray
        DataArray containing destination longitude bounds.

    Returns
    -------
    Tuple[List, List]
        A tuple of cell mappings and cell weights.
    """
    src_west, src_east = _extract_bounds(src)
    dst_west, dst_east = _extract_bounds(dst)

    shifted_src_west, shifted_src_east, shift = _align_axis(
        src_west,
        src_east,
        dst_west,
    )

    src_length = src_west.shape[0]
    dst_length = dst_west.shape[0]

    mapping = [
        np.where(
            np.logical_and(
                shifted_src_west < dst_east[x], shifted_src_east > dst_west[x]
            )
        )[0]
        for x in range(dst_length)
    ]

    weights = [
        (
            np.minimum(dst_east[x], shifted_src_east[y])
            - np.maximum(dst_west[x], shifted_src_west[y])
        ).reshape((1, -1))
        for x, y in enumerate(mapping)
    ]

    for x in range(len(mapping)):
        mapping[x] += shift

        wrapped = np.where(mapping[x] > src_length - 1)[0]

        mapping[x][wrapped] -= src_length

    return mapping, weights


def _extract_bounds(bounds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract lower and upper bounds from an axis.

    Parameters
    ----------
    bounds : np.ndarray
     Dataset containing axis with bounds.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
         A tuple containing the lower and upper bounds for the axis.
    """
    if bounds[0, 0] < bounds[0, 1]:
        lower = bounds[:, 0]
        upper = bounds[:, 1]
    else:
        lower = bounds[:, 1]
        upper = bounds[:, 0]

    return lower.astype(np.float32), upper.astype(np.float32)


def _align_axis(
    src_west: np.ndarray,
    src_east: np.ndarray,
    dst_west: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Aligns a longitudinal source axis to the destination axis.

    Parameters
    ----------
    src_west : np.ndarray
        DataArray containing the western source bounds.
    src_east : np.ndarray
        DataArray containing the eastern source bounds.
    dst_west : np.ndarray
        DataArray containing the western destination bounds.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, int]
        A tuple containing the shifted western source bounds, the shifted eastern
        source bounds, and the number of places shifted to align axis.
    """
    west_most = np.minimum(dst_west[0], dst_west[-1])

    alignment_index = _vpertub((west_most - src_west[-1]) / 360.0)

    alignment_index = (
        alignment_index + 1 if src_west[0] < src_west[-1] else alignment_index - 1
    )

    relative_postition = _vpertub((west_most - src_west) / 360.0)

    src_alignment_index = np.where(relative_postition != alignment_index)[0][0]

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

    shifted_src_west = (
        src_west[shifted_indexes] + 360.0 * relative_postition[shifted_indexes]
    )

    shifted_src_east = (
        src_east[shifted_indexes] + 360.0 * relative_postition[shifted_indexes]
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


def _pertub(value: np.ndarray) -> np.ndarray:
    """
    Pertub a value.

    Modifies value with a small constant and returns nearest whole
    number.

    Parameters
    ----------
    value : np.ndarray
        Value to pertub.

    Returns
    -------
    np.ndarray
        Value that's been pertubed.
    """
    if value >= 0.0:
        offset = np.ceil(value + 0.000001)
    else:
        offset = np.floor(value - 0.000001) + 1.0

    return offset


# vectorize version of pertub
_vpertub = np.vectorize(_pertub)


def _get_dimension(input_data_var, cf_axis_name):
    name = input_data_var.cf.axes[cf_axis_name]

    if isinstance(name, list):
        name = name[0]

    index = input_data_var.dims.index(name)

    return name, index


def _get_bounds_ensure_dtype(da, axis):
    # Avoid DataSet.bounds.get_bounds, can be slow
    if axis in da.cf.bounds:
        name = da.cf.bounds[axis][0]

        bounds = da[name]
    else:
        bounds = da.bounds.get_bounds(axis)

    if bounds.dtype != np.float32:
        bounds = bounds.astype(np.float32)

    return bounds.data
