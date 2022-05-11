from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import xarray as xr

from xcdat.regridder.base import BaseRegridder


def extract_bounds(bounds: xr.DataArray) -> Tuple[xr.DataArray, xr.DataArray]:
    """
     Extract lower and upper bounds from an axis.

     Parameters
     ----------
     bounds : xr.DataArray
         Dataset containing axis with bounds.

     Returns
     -------
    Tuple[xr.DataArray, xr.DataArray]
         A tuple containing the lower and upper bounds for the axis.
    """
    if bounds[0, 0] < bounds[0, 1]:
        lower = bounds[:, 0]
        upper = bounds[:, 1]
    else:
        lower = bounds[:, 1]
        upper = bounds[:, 0]

    return lower, upper


def map_latitude(src: xr.DataArray, dst: xr.DataArray) -> Tuple[List, List]:
    """
    Map source to destination latitude.

    Parameters
    ----------
    src : xr.DataArray
        DataArray containing the source latitude bounds.
    dst : xr.DataArray
        DataArray containing the destination latitude bounds.

    Returns
    -------
    Tuple[List, List]
        A tuple of cell mappings and cell weights.
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
    """
    Pertub a value.

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
    """
    Aligns a longitudinal source axis to the destination axis.

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
        A tuple containing the shifted western source bounds, the shifted eastern
        source bounds, and the number of places shifted to align axis.
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
    """
    Map source to destination longitude.

    Parameters
    ----------
    src : xr.DataArray
        DataArray containing source longitude bounds.
    dst : xr.DataArray
        DataArray containing destination longitude bounds.

    Returns
    -------
    Tuple[List, List]
        A tuple of cell mappings and cell weights.
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
    def __init__(
        self, src_grid: xr.Dataset, dst_grid: xr.Dataset, **options: Dict[str, Any]
    ):
        """
        Class to generate mapping and apply regridding between source and destination
        grid.

        Supported options: None

        Parameters
        ----------
        src_grid : xr.Dataset
            Dataset containing the source grid.
        dst_grid : xr.Dataset
            Dataset containing the destination grid.
        options : Dict[str, Any]
            Dictionary with extra parameters for the regridder.
        """
        super().__init__(src_grid, dst_grid, **options)

        src_lat = src_grid.bounds.get_bounds("lat")
        self._dst_lat = dst_grid.bounds.get_bounds("lat")

        self._lat_mapping, self._lat_weights = map_latitude(src_lat, self._dst_lat)

        src_lon = src_grid.bounds.get_bounds("lon")
        self._dst_lon = dst_grid.bounds.get_bounds("lon")

        self._lon_mapping, self._lon_weights = map_longitude(src_lon, self._dst_lon)

    def _base_put_indexes(self, axis_sizes: Dict[str, int]) -> np.ndarray:
        """
        Calculates the base indexes to place cell (0, 0).

        Example:
        For a 3D array (time, lat, lon) with the shape (2, 2, 2) the offsets to
        place cell (0, 0) in each time step would be [0, 4].

        For a 4D array (time, plev, lat, lon) with shape (2, 2, 2, 2) the offsets
        to place cell (0, 0) in each time step would be [0, 4, 8, 16].

        Parameters
        ----------
        axis_sizes : Dict[str, int]
            Mapping of axis name e.g. ("X", "Y", etc) to output sizes.

        Returns
        -------
        np.ndarray
            Array containing the base indexes to be used in np.put operations.
        """
        extra_dims = set(axis_sizes) - set(["X", "Y"])

        number_of_offsets = np.multiply.reduce([axis_sizes[x] for x in extra_dims])

        offset = np.multiply.reduce(
            [axis_sizes[x] for x in extra_dims ^ set(axis_sizes)]
        )

        return (np.arange(number_of_offsets) * offset).astype(np.int64)

    def _output_axis_sizes(self, da: xr.DataArray) -> Dict[str, int]:
        """
        Maps axes to output array sizes.

        Parameters
        ----------
        da : xr.DataArray
            Data array containing variable to be regridded.

        Returns
        -------
        Dict
            Mapping of axis name e.g. ("X", "Y", etc) to output sizes.
        """
        output_sizes = {}

        axis_name_map = {y[0]: x for x, y in da.cf.axes.items()}

        for standard_name in da.sizes.keys():
            axis_name = axis_name_map[standard_name]

            if standard_name in self._dst_grid:
                output_sizes[axis_name] = self._dst_grid.sizes[standard_name]
            else:
                output_sizes[axis_name] = da.sizes[standard_name]

        return output_sizes

    def _regrid(
        self,
        input_data: np.ndarray,
        axis_sizes: Dict[str, int],
        ordered_axis_names: List[str],
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Applies regridding to input data.

        Parameters
        ----------
        input_data : np.ndarray
            Input multi-dimensional array on source grid.
        axis_sizes : Dict[str, int]
            Mapping of axis name e.g. ("X", "Y", etc) to output sizes.
        ordered_axis_names : List[str]
            List of axis name in order of dimenions of ``input_data``.
        mask: Optional[np.ndarray], optional
            Multi-dimensional numpy array used for masking, defaults to None.

        Returns
        -------
        np.ndarray
            Multi-dimensional array on destination grid.
        """
        input_lat_index = ordered_axis_names.index("Y")

        input_lon_index = ordered_axis_names.index("X")

        output_shape = [axis_sizes[x] for x in ordered_axis_names]

        output_data = np.zeros(output_shape, dtype=np.float32)

        base_put_index = self._base_put_indexes(axis_sizes)

        mask_lat_index = 0
        mask_lon_index = 1

        if mask is None:
            input_shape = input_data.shape

            mask = np.ones((input_shape[input_lat_index], input_shape[input_lon_index]))

        for lat_index, lat_map in enumerate(self._lat_mapping):
            lat_weight = self._lat_weights[lat_index]

            input_lat_segment = np.take(input_data, lat_map, axis=input_lat_index)

            mask_lat_segment = np.take(mask, lat_map, axis=mask_lat_index)

            for lon_index, lon_map in enumerate(self._lon_mapping):
                lon_weight = self._lon_weights[lon_index]

                dot_weight = np.dot(lat_weight, lon_weight)

                cell_weight = np.sum(dot_weight)

                input_lon_segment = np.take(
                    input_lat_segment, lon_map, axis=input_lon_index
                )

                mask_lon_segment = np.take(
                    mask_lat_segment, lon_map, axis=mask_lon_index
                )

                # apply mask
                input_lon_segment = np.multiply(input_lon_segment, mask_lon_segment)

                data = (
                    np.multiply(input_lon_segment, dot_weight).sum(
                        axis=(input_lat_index, input_lon_index)
                    )
                    / cell_weight
                )

                # This only handles lat by lon and not lon by lat
                put_index = base_put_index + ((lat_index * axis_sizes["X"]) + lon_index)

                np.put(output_data, put_index, data)

        return output_data

    def _create_output_dataset(
        self,
        input_ds: xr.Dataset,
        data_var: str,
        output_data: np.ndarray,
        axis_variable_name_map: Dict[str, str],
        ordered_axis_names: List[str],
    ) -> xr.Dataset:
        """
        Creates the output Dataset containing the new variable on the destination grid.

        Parameters
        ----------
        input_ds : xr.Dataset
            Input dataset containing coordinates and bounds for unmodified axes.
        data_var : str
            The name of the regridded variable.
        output_data : np.ndarray
            Output data array.
        axis_variable_name_map : Dict[str, str]
            Map of axis name e.g. ("X", "Y", etc) to variable name e.g. ("lon", "lat", etc).
        ordered_axis_names : List[str]
            List of axis names in the order observed for ``output_data``.

        Returns
        -------
        xr.Dataset
            Dataset containing the variable on the destination grid.
        """
        variable_axis_name_map = {y: x for x, y in axis_variable_name_map.items()}

        coords = {}
        data_vars = {}

        # Grab coords and bounds from appropriate dataset.
        for variable_name, axis_name in variable_axis_name_map.items():
            if axis_name in ["X", "Y"]:
                coords[variable_name] = self._dst_grid[variable_name].copy()

                bounds = self._dst_grid.bounds.get_bounds(variable_name)
            else:
                coords[variable_name] = input_ds[variable_name].copy()

                bounds = input_ds.bounds.get_bounds(variable_name)

            data_vars[bounds.name] = bounds.copy()

        output_da = xr.DataArray(
            output_data,
            dims=[axis_variable_name_map[x] for x in ordered_axis_names],
            coords=coords,
        )

        data_vars[data_var] = output_da

        return xr.Dataset(data_vars)

    def horizontal(self, data_var: str, ds: xr.Dataset) -> xr.Dataset:
        """
        Regrid data between rectilinear grids.

        Precalculated mapping and weights for the source to destination grid
        are applied to the selected variable in the Dataset.

        Parameters
        ----------
        data_var : str
            The name of the data variable inside the dataset to regrid.
        ds : xr.Dataset
            The dataset containing ``data_var``.

        Returns
        -------
        xr.Dataset
            Dataset with variable on the destination grid.

        Raises
        ------
        KeyError
            If data variable does not exist in the Dataset.

        Examples
        --------
        Import:

        >>> from xcdat.regridder import regrid2

        Create regridder:

        >>> regridder = regrid2.Regrid2Regridder(input_grid, output_grid)

        Regrid variable:

        >>> regridder.regrid("ts", input_ds)
        """
        input_data_var = ds.get(data_var, None)

        if input_data_var is None:
            raise KeyError(
                f"The data variable '{data_var}' does not exist in the dataset."
            )

        # operate on pure numpy
        input_data = input_data_var.values

        axis_variable_name_map = {x: y[0] for x, y in input_data_var.cf.axes.items()}

        output_axis_sizes = self._output_axis_sizes(input_data_var)

        ordered_axis_names = list(output_axis_sizes)

        mask = ds.get("mask", None)

        output_data = self._regrid(
            input_data, output_axis_sizes, ordered_axis_names, mask=mask
        )

        output_ds = self._create_output_dataset(
            ds, data_var, output_data, axis_variable_name_map, ordered_axis_names
        )

        return output_ds
