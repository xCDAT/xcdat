from typing import Optional

import xarray as xr

from xcdat.regridder.base import BaseRegridder, _preserve_bounds
from xcdat.utils import _has_module

# TODO: Test this conditional.
_has_xesmf = _has_module("xesmf")
if _has_xesmf:  # pragma: no cover
    import xesmf as xe
else:  # pragma: no cover
    raise ModuleNotFoundError(
        "The `xesmf` package is required for horizontal regridding with `xesmf`. Make "
        "sure your platform supports `xesmf` and it is installed in your conda "
        "environment."
    )


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
    def __init__(
        self,
        input_grid: xr.Dataset,
        output_grid: xr.Dataset,
        method: str = "bilinear",
        periodic: bool = False,
        extrap_method: Optional[str] = None,
        extrap_dist_exponent: Optional[float] = None,
        extrap_num_src_pnts: Optional[int] = None,
        ignore_degenerate: bool = True,
        **options,
    ):
        super().__init__(input_grid, output_grid)

        if method not in VALID_METHODS:
            raise ValueError(
                f"{method!r} is not valid, possible options: {', '.join(VALID_METHODS)}"
            )

        if extrap_method is not None and extrap_method not in VALID_EXTRAP_METHODS:
            raise ValueError(
                f"{extrap_method!r} is not valid, possible options: {', '.join(VALID_EXTRAP_METHODS)}"
            )

        self._method = method
        self._periodic = periodic
        self._extrap_method = extrap_method
        self._extrap_dist_exponent = extrap_dist_exponent
        self._extrap_num_src_pnts = extrap_num_src_pnts
        self._ignore_degenerate = ignore_degenerate
        self._regridder: xe.XESMFRegridder = None
        self._extra_options = options

    def vertical(self, data_var: str, ds: xr.Dataset) -> xr.Dataset:
        """Placeholder for base class."""
        raise NotImplementedError()

    def horizontal(self, data_var: str, ds: xr.Dataset) -> xr.Dataset:
        """Regrid ``data_var`` in ``ds`` to output grid.

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

        Create output grid:

        >>> output_grid = xcdat.create_gaussian_grid(32)

        Create regridder:

        >>> regridder = xesmf.XESMFRegridder(ds, output_grid, method="bilinear")

        Regrid data:

        >>> data_new_grid = regridder.horizontal("ts", ds)
        """
        input_da = ds.get(data_var, None)

        if input_da is None:
            raise KeyError(
                f"The data variable '{data_var}' does not exist in the dataset."
            )

        if self._regridder is None:
            self._regridder = xe.Regridder(
                self._input_grid,
                self._output_grid,
                method=self._method,
                periodic=self._periodic,
                extrap_method=self._extrap_method,
                extrap_dist_exponent=self._extrap_dist_exponent,
                extrap_num_src_pnts=self._extrap_num_src_pnts,
                ignore_degenerate=self._ignore_degenerate,
                **self._extra_options,
            )

        output_da = self._regridder(input_da, keep_attrs=True)

        output_ds = xr.Dataset({data_var: output_da}, attrs=ds.attrs)
        output_ds = _preserve_bounds(ds, self._output_grid, output_ds, ["X", "Y"])

        return output_ds
