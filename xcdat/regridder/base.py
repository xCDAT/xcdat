import abc
from typing import Any, Tuple, Union

import numpy as np
import xarray as xr

import xcdat.bounds  # noqa: F401
from xcdat.logger import setup_custom_logger

logger = setup_custom_logger(__name__)

Coord = Union[np.ndarray, xr.DataArray]

CoordOptionalBnds = Union[Coord, Tuple[Coord, Coord]]


def preserve_bounds(
    source: xr.Dataset, target_grid: xr.Dataset, target: xr.Dataset
) -> xr.Dataset:
    """Preserves bounds from sources to target.

    Ensure the lat/lon bounds from `target_grid` are included in the `target` dataset.

    Preserve any additional bounds e.g. time, vertical from `source` to `target`.

    Parameters
    ----------
    source : xr.Dataset
        Source Dataset.
    target_grid : xr.Dataset
        Target grid Dataset.
    target : xr.Dataset
        Target Dataset to preserve bounds to.

    Returns
    -------
    xr.Dataset
        Target Dataset with preserved bounds.
    """
    try:
        lat_bnds = target_grid.bounds.get_bounds("Y")
    except KeyError:
        pass
    else:
        target[lat_bnds.name] = lat_bnds.copy()

    try:
        lon_bnds = target_grid.bounds.get_bounds("X")
    except KeyError:
        pass
    else:
        target[lon_bnds.name] = lon_bnds.copy()

    for dim_name in source.cf.axes:
        try:
            source_bnds = source.bounds.get_bounds(dim_name)
        except KeyError:
            logger.debug(f"No bounds for dimension {dim_name!r} found in source")
        else:
            if source_bnds.name in target:
                logger.debug(f"Bounds {source_bnds.name!r} already present")
            elif dim_name in ["X", "Y"]:
                # the X / Y bounds are copied from the target grid above
                continue
            else:
                target[source_bnds.name] = source_bnds.copy()

                logger.debug(f"Preserved bounds {source_bnds.name!r} from source")

    return target


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
