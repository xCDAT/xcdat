import abc
from typing import Any, List, Tuple, Union

import numpy as np
import xarray as xr

import xcdat.bounds  # noqa: F401
from xcdat._logger import _setup_custom_logger
from xcdat.axis import CFAxisKey, get_dim_keys

logger = _setup_custom_logger(__name__)

Coord = Union[np.ndarray, xr.DataArray]

CoordOptionalBnds = Union[Coord, Tuple[Coord, Coord]]


def _preserve_bounds(
    input_ds: xr.Dataset,
    output_grid: xr.Dataset,
    output_ds: xr.Dataset,
    drop_axis: List[CFAxisKey],
) -> xr.Dataset:
    """Preserves existing bounds from datasets.

    Preserves bounds from `ouput_grid` and `input_ds` to `output_ds`.

    Parameters
    ----------
    input_ds : xr.Dataset
        Input Dataset used for regridding.
    output_grid : xr.Dataset
        Output grid Dataset used for regridding.
    output_ds : xr.Dataset
        Dataset bounds will be copied to.
    drop_axis : List[CFAxisKey]
        Axis or axes to drop from `input_ds`, which drops the related coords
        and bounds. For example, dropping the "Y" axis in `input_ds` ensures
        that the "Y" axis in `output_grid` is referenced for bounds.

    Returns
    -------
    xr.Dataset
        Target Dataset with preserved bounds.
    """
    input_ds = _drop_axis(input_ds, drop_axis)

    for ds in (output_grid, input_ds):
        for axis in ("X", "Y", "Z", "T"):
            try:
                bnds = ds.bounds.get_bounds(axis)
            except KeyError:
                pass
            else:
                if bnds.name not in output_ds:
                    output_ds[bnds.name] = bnds.copy()

    return output_ds


def _drop_axis(ds: xr.Dataset, axis: List[CFAxisKey]) -> xr.Dataset:
    """Drops an axis or axes in a dataset.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset.
    axis : List[CFAxisKey]
        The axis or axes to drop.

    Returns
    -------
    xr.Daatset
        The dataset with axis or axes dropped.
    """
    dims: List[str] = []

    for ax in axis:
        try:
            dim = get_dim_keys(ds, ax)
        except KeyError:
            pass
        else:
            if isinstance(dim, str):
                dims.append(dim)
            elif isinstance(dim, list):
                dims = dims + dim

    if len(dims) > 0:
        ds = ds.drop_dims(dims)

    return ds


class BaseRegridder(abc.ABC):
    """BaseRegridder."""

    def __init__(self, input_grid: xr.Dataset, output_grid: xr.Dataset, **options: Any):
        self._input_grid = input_grid
        self._output_grid = output_grid
        self._options = options

    @abc.abstractmethod
    def horizontal(
        self, data_var: str, ds: xr.Dataset
    ) -> xr.Dataset:  # pragma: no cover
        pass

    @abc.abstractmethod
    def vertical(self, data_var: str, ds: xr.Dataset) -> xr.Dataset:  # pragma: no cover
        pass
