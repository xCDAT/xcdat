import abc
from typing import Any

import xarray as xr

from xcdat.logger import setup_custom_logger

logger = setup_custom_logger(__name__)


def preserve_bounds(
    source: xr.Dataset, source_grid: xr.Dataset, target: xr.Dataset
) -> xr.Dataset:
    """Preserves bounds from sources to target.

    Preserve the lat/lon bounds from `source_grid` to `target`.

    Preserve any additional bounds e.g. time, vertical from `source` to `target`.

    Parameters
    ----------
    source : xr.Dataset
        Source Dataset.
    source_grid : xr.Dataset
        Source grid Dataset.
    target : xr.Dataset
        Target Dataset to preserve bounds to.

    Returns
    -------
    xr.Dataset
        Target Dataset with preserved bounds.
    """
    try:
        lat_bnds = source_grid.cf.get_bounds("lat")
    except KeyError:
        pass
    else:
        target[lat_bnds.name] = lat_bnds.copy()

    try:
        lon_bnds = source_grid.cf.get_bounds("lon")
    except KeyError:
        pass
    else:
        target[lon_bnds.name] = lon_bnds.copy()

    for dim_name in source.cf.axes:
        try:
            source_bnds = source.cf.get_bounds(dim_name)
        except KeyError:
            logger.debug(f"No bounds for dimension {dim_name!r} found in source")
        else:
            if source_bnds.name in target:
                logger.debug(f"Bounds {source_bnds.name!r} already present")
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
