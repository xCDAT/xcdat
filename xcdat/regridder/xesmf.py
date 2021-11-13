from typing import Any

import xarray as xr
import xesmf as xe

from xcdat import dataset
from xcdat.regridder.base import BaseRegridder


VALID_METHODS = [
    "bilinear",
    "conservative",
    "conservative_normed",
    "patch",
    "nearest_s2d",
    "nearest_d2s",
]

VALID_EXTRAP_METHODS = ["inverse_dist", "nearest_s2d"]


class XESMFRegridder(BaseRegridder):
    """Wrapper class for xESMF regridder class.

    Creates a reusable regridder object.

    Parameters
    ----------
    src_grid : xr.Dataset
        Contains source grid coordinates.
    dst_grid : xr.Dataset
        Contains desintation grid coordinates.
    method : str
        Regridding method. Options are
        - bilinear
        - conservative
        - conservative_normed
        - patch
        - nearest_s2d
        - nearest_d2s
    data_var : str
        Target variable to apply regridding to.
    periodic : bool
        Treat longitude as periodic. Used for global grids.
    extrap_method : str
        Extrapolation method. Options are
        - inverse_dist
        - nearest_s2d
    extrap_dist_exponent : float
        The exponent to raise the distance to when calculating weights for the extrapolation method.
    extrap_num_src_pnts : int
        The number of source points to use for the extrapolation methods that use more than one source point.

    Raises
    ------
    KeyError
        If data variable does not exist in the Dataset.
    ValueError
        If `method` is not valid.
    ValueError
        If `extrap_method` is not valid.

    Examples
    --------
    """

    def __init__(
        self,
        src_grid: xr.Dataset,
        dst_grid: xr.Dataset,
        method: str,
        data_var: str = None,
        periodic: bool = False,
        extrap_method: str = None,
        extrap_dist_exponent: float = None,
        extrap_num_src_pnts: int = None,
    ):
        self._src_grid = src_grid
        self._dst_grid = dst_grid

        if data_var is None:
            src_var = dataset.get_inferred_var(src_grid)
        else:
            src_var = src_grid.get(data_var)

            if src_var is None:
                raise KeyError(
                    f"The data variable {data_var!r} does not exist in the dataset."
                )

        if method not in VALID_METHODS:
            raise ValueError(
                f"{method!r} is not valid, possible options: {', '.join(VALID_METHODS)}"
            )

        if extrap_method is not None and extrap_method not in VALID_EXTRAP_METHODS:
            raise ValueError(
                f"{extrap_method!r} is not valid, possible options: {', '.join(VALID_EXTRAP_METHODS)}"
            )

        self._regridder = xe.Regridder(
            src_var,
            self._dst_grid,
            method,
            periodic=periodic,
            extrap_method=extrap_method,
            extrap_dist_exponent=extrap_dist_exponent,
            extrap_num_src_pnts=extrap_num_src_pnts,
        )

    def regrid(self, ds: xr.Dataset) -> xr.Dataset:
        return self._regridder(ds)
