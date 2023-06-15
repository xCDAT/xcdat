import abc
from typing import Any, List, Tuple, Union

import numpy as np
import xarray as xr

import xcdat.bounds  # noqa: F401
from xcdat._logger import _setup_custom_logger
from xcdat.axis import CFAxisKey

logger = _setup_custom_logger(__name__)

Coord = Union[np.ndarray, xr.DataArray]

CoordOptionalBnds = Union[Coord, Tuple[Coord, Coord]]


def _preserve_bounds(
    output_ds: xr.Dataset,
    output_grid: xr.Dataset,
    input_ds: xr.Dataset,
    ignore_dims: List[CFAxisKey],
) -> xr.Dataset:
    """Preserves existing bounds from datasets.

    Preserves bounds from `ouput_grid` and `input_ds` to `output_ds`.

    Parameters
    ----------
    output_ds : xr.Dataset
        Dataset bounds will be copied to.
    output_grid : xr.Dataset
        Output grid Dataset used for regridding.
    input_ds : xr.Dataset
        Input Dataset used for regridding.
    ignore_dims : List[CFAxisKey]
        Dimensions to drop from `input_ds`.

    Returns
    -------
    xr.Dataset
        Target Dataset with preserved bounds.
    """
    input_ds = input_ds.drop_dims([input_ds.cf[x].name for x in ignore_dims])

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
