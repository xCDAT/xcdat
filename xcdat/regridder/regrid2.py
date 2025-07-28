from typing import Any

import numpy as np
import sparse as sp
import xarray as xr

import xcdat as xc
from xcdat.axis import get_dim_keys
from xcdat.regridder.base import BaseRegridder, _preserve_bounds
from xcdat.regridder.grid import create_mask, create_nan_mask


class Regrid2Regridder(BaseRegridder):
    def __init__(
        self,
        input_grid: xr.Dataset,
        output_grid: xr.Dataset,
        unmapped_to_nan: bool = True,
        output_weights: bool | str = False,
        create_nan_mask: bool = False,
        **options: Any,
    ):
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
        unmapped_to_nan : bool
            If True, unmapped values are set to NaN. Default is True.
        output_weights : bool | str
            If True, output weights are added to the output dataset as weights.
            If str, the name of the variable to store the weights. Default is False.
        create_nan_mask : bool
            If True, a mask is created using the nan values from source variable. If
            a mask already exists in the Dataset it will be ignored. Default is False.
        **options : Any
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

        self._unmapped_to_nan = unmapped_to_nan
        self._output_weights = output_weights
        self._create_nan_mask = create_nan_mask

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

        if self._create_nan_mask:
            src_mask = create_nan_mask(input_data_var, ["Y", "X"]).values
        else:
            # DataArray to np.ndarray, handle error when None
            try:
                src_mask = self._input_grid.get("mask", None).values  # type: ignore
            except AttributeError:
                # regrid2 requires a mask, so create one
                src_mask = create_mask(self._input_grid, ["Y", "X"]).values

        nan_replace = input_data_var.encoding.get("_FillValue", None)

        if nan_replace is None:
            nan_replace = input_data_var.encoding.get("missing_value", 1e20)

        # exclude alternative of NaN values if there are any
        input_data_var = input_data_var.where(input_data_var != nan_replace)

        lat_mapping, lat_weights = _map_latitude(src_lat_bnds, dst_lat_bnds)
        lon_mapping, lon_weights = _map_longitude(src_lon_bnds, dst_lon_bnds)

        # horizontal regrid
        output_data = _regrid(
            input_data_var,
            lat_mapping,
            lon_mapping,
            lat_weights,
            lon_weights,
            src_mask,
            unmapped_to_nan=self._unmapped_to_nan,
        )

        output_ds = _build_dataset(
            ds,
            data_var,
            output_data,
            self._input_grid,
            self._output_grid,
        )

        if self._output_weights:
            weights = _sparse_weights(
                (len(src_lat_bnds), len(src_lon_bnds)),
                (len(dst_lat_bnds), len(dst_lon_bnds)),
                len(src_lon_bnds),
                len(dst_lon_bnds),
                lat_mapping,
                lon_mapping,
                lat_weights,
                lon_weights,
            )

            if isinstance(self._output_weights, str):
                output_ds[self._output_weights] = weights
            else:
                output_ds["weights"] = weights

        return output_ds


def _regrid(
    input_data_var: xr.DataArray,
    lat_mapping: list[np.ndarray],
    lon_mapping: list[np.ndarray],
    lat_weights: list[np.ndarray],
    lon_weights: list[np.ndarray],
    src_mask: np.ndarray,
    omitted=None,
    unmapped_to_nan=True,
) -> np.ndarray:
    if omitted is None:
        omitted = np.nan

    # convert to pure numpy
    input_data = input_data_var.astype(np.float32).values

    y_name, y_index = _get_dimension(input_data_var, "Y")
    x_name, x_index = _get_dimension(input_data_var, "X")

    y_length = len(lat_mapping)
    x_length = len(lon_mapping)

    other_dims = {
        x: y for x, y in input_data_var.sizes.items() if x not in (y_name, x_name)
    }
    other_sizes = list(other_dims.values())

    data_shape = [y_length * x_length] + other_sizes

    # output data is always float32 in original code
    output_data = np.zeros(data_shape, dtype=np.float32)
    output_mask = np.ones(data_shape, dtype=np.float32)

    is_2d = input_data_var.ndim <= 2

    # TODO: need to optimize further, investigate using ufuncs and dask arrays
    # TODO: how common is lon by lat data? may need to reshape
    for y in range(y_length):
        y_seg = np.take(input_data, lat_mapping[y], axis=y_index)
        y_mask_seg = np.take(src_mask, lat_mapping[y], axis=0)

        for x in range(x_length):
            x_seg = np.take(y_seg, lon_mapping[x], axis=x_index, mode="wrap")
            x_mask_seg = np.take(y_mask_seg, lon_mapping[x], axis=1, mode="wrap")

            cell_weights = np.multiply(
                np.dot(lat_weights[y], lon_weights[x]), x_mask_seg
            )

            cell_weight = np.sum(cell_weights)

            output_seg_index = y * x_length + x

            if cell_weight == 0.0:
                output_mask[output_seg_index] = 0.0

            # using the `out` argument is more performant, places data directly into
            # array memory rather than allocating a new variable. wasn't working for
            # single element output, needs further investigation as we may not need
            # branch
            if is_2d:
                output_data[output_seg_index] = np.divide(
                    np.sum(
                        np.multiply(x_seg, cell_weights),
                        axis=(y_index, x_index),
                    ),
                    cell_weight,
                )
            else:
                output_seg = output_data[output_seg_index]

                np.divide(
                    np.sum(
                        np.multiply(x_seg, cell_weights),
                        axis=(y_index, x_index),
                    ),
                    cell_weight,
                    out=output_seg,
                )

            if cell_weight <= 0.0:
                output_data[output_seg_index] = omitted

    # default for unmapped is nan due to division by zero, use output mask to repalce
    if not unmapped_to_nan:
        output_data[output_mask == 0.0] = 0.0

    output_data_shape = [y_length, x_length] + other_sizes

    output_data = output_data.reshape(output_data_shape)

    # temp dimensional ordering
    temp_dims = [y_name, x_name] + list(other_dims.keys())

    # map temp ordering to input ordering
    output_order = [temp_dims.index(x) for x in input_data_var.dims]

    output_data = output_data.transpose(output_order)

    return output_data.astype(np.float32)


def _build_dataset(
    ds: xr.Dataset,
    data_var: str,
    output_data: np.ndarray,
    input_grid: xr.Dataset,
    output_grid: xr.Dataset,
) -> xr.Dataset:
    """Build a new xarray Dataset with the given output data and coordinates.

    Parameters
    ----------
    ds : xr.Dataset
        The input dataset containing the data variable to be regridded.
    data_var : str
        The name of the data variable in the input dataset to be regridded.
    output_data : np.ndarray
        The regridded data to be included in the output dataset.
    input_grid : xr.Dataset
        The input grid dataset containing the original grid information.
    output_grid : xr.Dataset
        The output grid dataset containing the new grid information.

    Returns
    -------
    xr.Dataset
        A new dataset containing the regridded data variable with updated
        coordinates and attributes.
    """
    dv_input = ds[data_var]

    output_coords = _get_output_coords(dv_input, output_grid)

    output_da = xr.DataArray(
        output_data,
        dims=output_coords.keys(),
        coords=output_coords,
        attrs=ds[data_var].attrs.copy(),
        name=data_var,
    )

    output_ds = output_da.to_dataset()
    output_ds.attrs = input_grid.attrs.copy()
    output_ds = _preserve_bounds(ds, output_grid, output_ds, ["X", "Y"])

    return output_ds


def _sparse_weights(
    in_shape: tuple[int, int],
    out_shape: tuple[int, int],
    in_width: int,
    out_width: int,
    lat_mapping: list[np.ndarray],
    lon_mapping: list[np.ndarray],
    lat_weights: list[np.ndarray],
    lon_weights: list[np.ndarray],
) -> xr.DataArray:
    """
    Generates a sparse weight matrix for regridding.

    Parameters
    ----------
    in_shape : tuple
        Shape of the input grid.
    out_shape : tuple
        Shape of the output grid.
    in_width : int
        Width of the input grid row.
    out_width : int
        Width of the output grid row.
    lat_mapping : list[np.ndarray]
        List of latitude mappings.
    lon_mapping : list[np.ndarray]
        List of longitude mappings.
    lat_weights : list[np.ndarray]
        List of latitude weights.
    lon_weights : list[np.ndarray]
        List of longitude weights.

    Returns
    -------
    xr.DataArray
        A sparse weight matrix for regridding.
    """
    weights = np.zeros((np.prod(out_shape), np.prod(in_shape)), dtype=np.float32)

    for i, y in enumerate(lat_mapping):
        for j, x in enumerate(lon_mapping):
            # destination index
            dst_row = i * out_width + j

            # list of source indexes
            src_col = ((y * in_width).reshape((-1, 1)) + x).flatten()

            # assign weights to matrix
            weights[dst_row, src_col] = np.dot(lat_weights[i], lon_weights[j]).flatten()

    # reshape from 2D (src, dest) to 4D (src y, src x, dest y, dest x), then convert to sparse
    # provides user with simple way to explore weights and mapping
    sparse_weights = sp.COO.from_numpy(weights).reshape(out_shape + in_shape)

    coords = {
        "y_out": np.arange(out_shape[0]),
        "x_out": np.arange(out_shape[1]),
        "y_in": np.arange(in_shape[0]),
        "x_in": np.arange(in_shape[1]),
    }

    return xr.DataArray(
        sparse_weights, dims=["y_out", "x_out", "y_in", "x_in"], coords=coords
    )


def _get_output_coords(
    dv_input: xr.DataArray, output_grid: xr.Dataset
) -> dict[str, xr.DataArray]:
    """
    Generate the output coordinates for regridding based on the input data
    variable and output grid.

    Parameters
    ----------
    dv_input : xr.DataArray
        The input data variable containing the original coordinates.
    output_grid : xr.Dataset
        The dataset containing the target grid coordinates.

    Returns
    -------
    dict[str, xr.DataArray]
        A dictionary where keys are coordinate names and values are the
        corresponding coordinates from the output grid or input data variable,
        aligned with the dimensions of the input data variable.
    """
    output_coords: dict[str, xr.DataArray] = {}
    input_dims = [str(dim) for dim in dv_input.dims]

    # First get the X and Y axes from the output grid.
    for key in ["X", "Y"]:
        input_coord = xc.get_dim_coords(dv_input, key)  # type: ignore
        output_coord = xc.get_dim_coords(output_grid, key)  # type: ignore

        output_coords[str(input_coord.name)] = output_coord  # type: ignore

    # Get the remaining axes the input data variable (e.g., "time").
    for dim in dv_input.dims:
        if dim not in output_coords:
            output_coords[str(dim)] = dv_input[dim]

    # Sort the coords to align with order of input data variable dims. Rename
    # the dictionary keys to match output grid dimensions.
    output_coords = {
        str(output_coords[dim].name): output_coords[dim] for dim in input_dims
    }

    return output_coords


def _map_latitude(
    src: np.ndarray, dst: np.ndarray
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Map source to destination latitude.

    Source cells are grouped by the contribution to each output cell.

    Source cells have new boundaries calculated by finding minimum northern
    and maximum southern boundary between each source cell and the destination
    cell it contributes to.

    The source cell weights are calculated by taking the difference of sin's
    between these new boundary pairs.

    Parameters
    ----------
    src : np.ndarray
        Array containing the source latitude bounds.
    dst : np.ndarray
        Array containing the destination latitude bounds.

    Returns
    -------
    tuple[list[np.ndarray], list[np.ndarray]]
        A tuple of cell mappings and cell weights.
    """
    src_south, src_north = _extract_bounds(src)
    dst_south, dst_north = _extract_bounds(dst)

    dst_length = dst_south.shape[0]

    # finds contributing source cells for each destination cell based on bounds values
    # output is a list of lists containing the contributing cell indexes
    # e.g. let src_south be [90, 45, 0, -45], source_north be [45, 0, -45, -90],
    # dst_north[x] be 70, and dst_south[x] be -70 then the result would be [[1, 2]]
    mapping = [
        np.where(np.logical_and(src_south < dst_north[x], src_north > dst_south[x]))[0]
        for x in range(dst_length)
    ]

    # finds minimum and maximum bounds for each output cell, considers source and
    # destination bounds for each cell
    bounds = [
        (np.minimum(dst_north[x], src_north[y]), np.maximum(dst_south[x], src_south[y]))
        for x, y in enumerate(mapping)
    ]

    # convert latitude to cell weight (difference of height above/below equator)
    weights = _get_latitude_weights(bounds)

    return mapping, weights


def _get_latitude_weights(
    bounds: list[tuple[np.ndarray, np.ndarray]],
) -> list[np.ndarray]:
    weights = []

    for x, y in bounds:
        cell_weight = np.sin(np.deg2rad(x)) - np.sin(np.deg2rad(y))
        cell_weight = cell_weight.reshape((-1, 1))

        weights.append(cell_weight)

    return weights


def _map_longitude(src: np.ndarray, dst: np.ndarray) -> tuple[list, list]:
    """
    Map source to destination longitude.

    Source boundaries are aligned to the most western destination cell.

    Source cells are grouped by the contribution to each output cell.

    The source cell weights are calculated by find the difference of the
    following min/max for each input cell. Minimum of eastern source bounds
    and the eastern bounds of the destination cell it contributes to. Maximum
    of western source bounds and the western bounds of the destination cell
    it contributes to.

    These weights are then shifted to align with the destination longitude.

    Parameters
    ----------
    src : np.ndarray
        Array containing source longitude bounds.
    dst : np.ndarray
        Array containing destination longitude bounds.

    Returns
    -------
    tuple[list, list]
        A tuple of cell mappings and cell weights.
    """
    src_west, src_east = _extract_bounds(src)
    dst_west, dst_east = _extract_bounds(dst)

    # align source and destination longitude
    shifted_src_west, shifted_src_east, shift = _align_axis(
        src_west,
        src_east,
        dst_west,
    )

    src_length = src_west.shape[0]
    dst_length = dst_west.shape[0]

    # finds contributing source cells for each destination cell based on bounds values
    # output is a list of lists containing the contributing cell indexes
    mapping = [
        np.where(
            np.logical_and(
                shifted_src_west < dst_east[x], shifted_src_east > dst_west[x]
            )
        )[0]
        for x in range(dst_length)
    ]

    # weights are just the difference between minimum and maximum of contributing bounds
    # for each destination cell
    weights = [
        (
            np.minimum(dst_east[x], shifted_src_east[y])
            - np.maximum(dst_west[x], shifted_src_west[y])
        ).reshape((1, -1))
        for x, y in enumerate(mapping)
    ]

    # need to adjust the source contributing indexes by the shift required to align
    # source and destination longitude
    for x in range(len(mapping)):
        # shift the mapping indexes by the shift used to determine the weights
        mapping[x] += shift

        # find the contributing indexes that need to be wrapped
        wrapped = np.where(mapping[x] > src_length - 1)[0]

        # wrap the contributing index as all indexes must be <src_length
        mapping[x][wrapped] -= src_length

    return mapping, weights


def _extract_bounds(bounds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract lower and upper bounds from an axis.

    Parameters
    ----------
    bounds : np.ndarray
        A numpy array of bounds values.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
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
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Aligns a source and destination longitude axis.

    Parameters
    ----------
    src_west : np.ndarray
        Array containing the western source bounds.
    src_east : np.ndarray
        Array containing the eastern source bounds.
    dst_west : np.ndarray
        Array containing the western destination bounds.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, int]
        A tuple containing the shifted western source bounds, the shifted eastern
        source bounds, and the number of places shifted to align axis.
    """
    # find smallest western bounds
    west_most = np.minimum(dst_west[0], dst_west[-1])

    # find cell index required to align bounds
    alignment_index = _vpertub((west_most - src_west[-1]) / 360.0)

    # shift index depending on first/last source bounds
    alignment_index = (
        alignment_index + 1 if src_west[0] < src_west[-1] else alignment_index - 1
    )

    # find relative indexes for each source cell to the destinations most western cell
    relative_postition = _vpertub((west_most - src_west) / 360.0)

    # find all index values that are not the alignment index
    src_alignment_index = np.where(relative_postition != alignment_index)[0][0]

    # determine the shift factor required to align source and destination bounds
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

    # shift the source index values
    shifted_indexes = np.arange(src_length + 1) + shift

    # find index values that need to be shift to be within 0 - src_length
    wrapped = np.where(shifted_indexes > src_length - 1)

    # shift the indexes
    shifted_indexes[wrapped] -= src_length

    # reorder src_west and add portion to align
    shifted_src_west = (
        src_west[shifted_indexes] + 360.0 * relative_postition[shifted_indexes]
    )

    # reorder src_east and add portion to align
    shifted_src_east = (
        src_east[shifted_indexes] + 360.0 * relative_postition[shifted_indexes]
    )

    # handle ends of each interval
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
    name = get_dim_keys(input_data_var, cf_axis_name)

    index = input_data_var.dims.index(name)

    return name, index


def _get_bounds_ensure_dtype(ds, axis):
    bounds = None

    try:
        bounds = ds.bounds.get_bounds(axis)
    except KeyError:
        pass

    if bounds is None:
        raise RuntimeError(f"Could not determine {axis!r} bounds")

    if bounds.dtype != np.float32:
        bounds = bounds.astype(np.float32)

    return bounds.values
