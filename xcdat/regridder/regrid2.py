from typing import Any, Dict, List, Tuple

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

        self._src_lat = self._input_grid.bounds.get_bounds("Y").values
        self._src_lon = self._input_grid.bounds.get_bounds("X").values

        self._dst_lat = self._output_grid.bounds.get_bounds("Y").values
        self._dst_lon = self._output_grid.bounds.get_bounds("X").values

        self._lat_mapping: Any = None
        self._lon_mapping: Any = None

        self._lat_weights: Any = None
        self._lon_weights: Any = None

    def vertical(self, data_var: str, ds: xr.Dataset) -> xr.Dataset:
        """Placeholder for base class."""
        raise NotImplementedError()

    def horizontal(self, data_var: str, ds: xr.Dataset) -> xr.Dataset:
        """See documentation in :py:func:`xcdat.regridder.regrid2.Regrid2Regridder`"""
        input_data_var = ds.get(data_var, None)

        if input_data_var is None:
            raise KeyError(
                f"The data variable '{data_var}' does not exist in the dataset."
            )

        # Do initial mapping between src/dst latitude and longitude.
        if self._lat_mapping is None and self._lat_weights is None:
            self._lat_mapping, self._lat_weights = _map_latitude(
                self._src_lat, self._dst_lat
            )

        if self._lon_mapping is None and self._lon_weights is None:
            self._lon_mapping, self._lon_weights = _map_longitude(
                self._src_lon, self._dst_lon
            )

        src_mask = self._input_grid.get("mask", None)

        # operate on pure numpy
        input_data: np.ndarray = input_data_var.values

        # apply source mask to input data
        if src_mask is not None:
            input_data_var = input_data_var.where(src_mask != 0.0)

        # operate on pure numpy
        input_data = input_data_var.values

        axis_variable_name_map = {x: y[0] for x, y in input_data_var.cf.axes.items()}

        output_axis_sizes = self._output_axis_sizes(input_data_var)

        ordered_axis_names = list(output_axis_sizes)

        output_data = self._regrid(input_data, output_axis_sizes, ordered_axis_names)

        output_ds = self._create_output_dataset(
            ds, data_var, output_data, axis_variable_name_map, ordered_axis_names
        )

        output_ds = _preserve_bounds(ds, self._output_grid, output_ds, ["X", "Y"])

        return output_ds

    def _output_axis_sizes(self, da: xr.DataArray) -> Dict[str, int]:
        """Maps axes to output array sizes.

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
            try:
                axis_name = axis_name_map[standard_name]
            except KeyError:
                raise RuntimeError(
                    f"Could not find axis {standard_name!r}, ensure {standard_name!r} "
                    "exists and the attributes are correct."
                )

            if standard_name in self._output_grid:
                output_sizes[axis_name] = self._output_grid.sizes[standard_name]
            else:
                output_sizes[axis_name] = da.sizes[standard_name]

        return output_sizes

    def _regrid(
        self,
        input_data: np.ndarray,
        axis_sizes: Dict[str, int],
        ordered_axis_names: List[str],
    ) -> np.ndarray:
        """Applies regridding to input data.

        Parameters
        ----------
        input_data : np.ndarray
            Input multi-dimensional array on source grid.
        axis_sizes : Dict[str, int]
            Mapping of axis name e.g. ("X", "Y", etc) to output sizes.
        ordered_axis_names : List[str]
            List of axis name in order of dimensions of ``input_data``.

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

        for lat_index, lat_map in enumerate(self._lat_mapping):
            lat_weight = self._lat_weights[lat_index]

            input_lat_segment = np.take(input_data, lat_map, axis=input_lat_index)

            for lon_index, lon_map in enumerate(self._lon_mapping):
                lon_weight = self._lon_weights[lon_index]

                dot_weight = np.dot(lat_weight, lon_weight)

                cell_weight = np.sum(dot_weight)

                input_lon_segment = np.take(
                    input_lat_segment, lon_map, axis=input_lon_index
                )

                data = (
                    np.nansum(
                        np.multiply(input_lon_segment, dot_weight),
                        axis=(input_lat_index, input_lon_index),
                    )
                    / cell_weight
                )

                # This only handles lat by lon and not lon by lat
                put_index = base_put_index + ((lat_index * axis_sizes["X"]) + lon_index)

                np.put(output_data, put_index, data)

        return output_data

    def _base_put_indexes(self, axis_sizes: Dict[str, int]) -> np.ndarray:
        """Calculates the base indexes to place cell (0, 0).

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

        # Grab coords and bounds from appropriate dataset.
        for variable_name, axis_name in variable_axis_name_map.items():
            if axis_name in ["X", "Y"]:
                coords[variable_name] = self._output_grid[variable_name].copy()
            else:
                coords[variable_name] = input_ds[variable_name].copy()

        output_da = xr.DataArray(
            output_data,
            dims=[axis_variable_name_map[x] for x in ordered_axis_names],
            coords=coords,
            attrs=input_ds[data_var].attrs.copy(),
        )

        data_vars = {data_var: output_da}

        return xr.Dataset(data_vars, attrs=input_ds.attrs.copy())


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

        weights.append(weight.reshape(contrib.shape[0], 1))

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
        src_west, src_east, dst_west
    )

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

        weights.append(weight.reshape(1, contrib.shape[0]))

        contrib += shift

        wrapped = np.where(contrib > src_length - 1)

        contrib[wrapped] -= src_length

        mapping.append(contrib)

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

    return lower, upper


def _align_axis(
    src_west: np.ndarray, src_east: np.ndarray, dst_west: np.ndarray
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

    if src_west[0] < src_west[-1]:
        alignment_index += 1
    else:
        alignment_index -= 1

    src_alignment_index = np.where(
        _vpertub((west_most - src_west) / 360.0) != alignment_index
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

    shifted_src_west = src_west[shifted_indexes] + 360.0 * _vpertub(
        (west_most - src_west[shifted_indexes]) / 360.0
    )

    shifted_src_east = src_east[shifted_indexes] + 360.0 * _vpertub(
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
