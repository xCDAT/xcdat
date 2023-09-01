from typing import Any, Optional

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
        method: str,
        periodic: bool = False,
        extrap_method: Optional[str] = None,
        extrap_dist_exponent: Optional[float] = None,
        extrap_num_src_pnts: Optional[int] = None,
        ignore_degenerate: bool = True,
        **options: Any,
    ):
        """
        Extension of ``xESMF`` regridder.

        This method extends ``xESMF`` by automatically constructing by ``xe.XESMFRegridder``
        object and ensuring bounds and metadata are preserved in the output dataset.

        The ``method`` argument can take any of the following values: `bilinear`, `conservative`,
        `conservative_normed`, `patch`, `nearest_s2d`, or `nearest_d2s`. You can find a comparison
        of the methods `here <https://xesmf.readthedocs.io/en/latest/notebooks/Compare_algorithms.html>`_.

        The ``extrap_method`` argument can take any of the following values: `inverse_dist` or `nearest_s2d`.
        This argument along with ``extrap_dist_exponent`` and ``extrap_num_src_pnts`` can be used to
        configure how extrapolation is applied.

        The ``**options`` arguments are additional values passed to the ``xe.XESMFRegridder``
        constructor. A description of these arguments can be found on `xESMF's documentation <https://github.com/pangeo-data/xESMF/blob/892ac87064d98d98d732ad8a79aa1682b081cdc2/xesmf/frontend.py#L702-L744>`_.

        Parameters
        ----------
        input_grid : xr.Dataset
            Contains source grid coordinates.
        output_grid : xr.Dataset
            Contains desintation grid coordinates.
        method : str
            The regridding method to apply, defaults to "bilinear".
        periodic : bool
            Treat longitude as periodic, used for global grids.
        extrap_method : Optional[str]
            Extrapolation method, useful when moving from a fine to coarse grid.
        extrap_dist_exponent : Optional[float]
            The exponent to raise the distance to when calculating weights for
            the extrapolation method.
        extrap_num_src_pnts : Optional[int]
            The number of source points to use for the extrapolation methods
            that use more than one source point.
        ignore_degenerate : bool
            Ignore degenerate cells when checking the `input_grid` for errors.
            If set False, a degenerate cell produces an error.

            This only applies to "conservative" and "conservative_normed"
            regridding methods.
        **options : Any
            Additional arguments passed to the underlying ``xe.XESMFRegridder``
            constructor.

        Raises
        ------
        KeyError
            If data variable does not exist in the Dataset.
        ValueError
            If ``method`` is not valid.
        ValueError
            If ``extrap_method`` is not valid.

        Examples
        --------
        Import xCDAT:

        >>> import xcdat

        Open a dataset:

        >>> ds = xcdat.open_dataset("...")

        Create output grid:

        >>> output_grid = xcdat.create_gaussian_grid(32)

        Regrid the "ts" variable using the "bilinear" method:

        >>> output_data = ds.regridder.horizontal("ts", output_grid, tool="xesmf", method="bilinear")

        Passing additional values to ``xe.XESMFRegridder``:

        >>> output_data = ds.regridder.horizontal("ts", output_grid, tool="xesmf", method="bilinear", unmapped_to_nan=True)
        """
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
        """See documentaion in :py:func:`xcdat.regridder.xesmf.XESMFRegridder`"""
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
