from typing import Any

import xarray as xr
import xesmf as xe

from xcdat.regridder.base import BaseRegridder, _preserve_bounds
from xcdat.regridder.grid import create_nan_mask

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
        extrap_method: str | None = None,
        extrap_dist_exponent: float | None = None,
        extrap_num_src_pnts: int | None = None,
        ignore_degenerate: bool = True,
        unmapped_to_nan: bool = True,
        output_weights: bool | str = False,
        create_nan_mask: bool = False,
        **options: Any,
    ):
        """Extension of ``xESMF`` regridder.

        This method extends ``xESMF`` by automatically constructing by
        ``xesmf.XESMFRegridder`` object and ensuring bounds and metadata are
        preserved in the output dataset.

        The ``method`` argument can take any of the following values:
        `bilinear`, `conservative`, `conservative_normed`, `patch`,
        `nearest_s2d`, or `nearest_d2s`. You can find a comparison of the
        methods `here <https://xesmf.readthedocs.io/en/latest/notebooks/Compare_algorithms.html>`_.

        The ``extrap_method`` argument can take any of the following values:
        `inverse_dist` or `nearest_s2d`. This argument along with
        ``extrap_dist_exponent`` and ``extrap_num_src_pnts`` can be used to
        configure how extrapolation is applied.

        The ``**options`` arguments are additional values passed to the
        ``xesmf.XESMFRegridder`` constructor. A description of these arguments can
        be found on `xESMF's documentation <https://github.com/pangeo-data/xESMF/blob/892ac87064d98d98d732ad8a79aa1682b081cdc2/xesmf/frontend.py#L702-L744>`_.

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
        extrap_method : str | None
            Extrapolation method, useful when moving from a fine to coarse grid.
        extrap_dist_exponent : float | None
            The exponent to raise the distance to when calculating weights for
            the extrapolation method.
        extrap_num_src_pnts : int | None
            The number of source points to use for the extrapolation methods
            that use more than one source point.
        ignore_degenerate : bool
            Ignore degenerate cells when checking the `input_grid` for errors.
            If set False, a degenerate cell produces an error.

            This only applies to "conservative" and "conservative_normed"
            regridding methods.
        unmapped_to_nan : bool
            Sets values of unmapped points to `np.nan` instead of 0 (ESMF default).
        output_weights : bool | str
            If True, output weights are added to the output dataset as weights.
            If str, the name of the variable to store the weights. Default is False.
        create_nan_mask : bool
            If True, a mask is created using the nan values from source variable. If
            a mask already exists in the Dataset it will be ignored. Default is False.
        **options : Any
            Additional arguments passed to the underlying ``xesmf.XESMFRegridder``
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

        >>> output_data = ds.regridder.horizontal(
        >>>     "ts", output_grid, tool="xesmf", method="bilinear"
        >>> )

        Passing additional values to ``xesmf.XESMFRegridder``:

        >>> output_data = ds.regridder.horizontal(
        >>>     "ts", output_grid, tool="xesmf", method="bilinear", unmapped_to_nan=True
        >>> )
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

        # Re-pack xesmf arguments, broken out for validation/documentation
        options.update(
            periodic=periodic,
            extrap_method=extrap_method,
            extrap_dist_exponent=extrap_dist_exponent,
            extrap_num_src_pnts=extrap_num_src_pnts,
            ignore_degenerate=ignore_degenerate,
            unmapped_to_nan=unmapped_to_nan,
        )

        self._extra_options = options
        self._output_weights = output_weights
        self._create_nan_mask = create_nan_mask

    def vertical(self, data_var: str, ds: xr.Dataset) -> xr.Dataset:
        """Placeholder for base class."""
        raise NotImplementedError()

    def horizontal(self, data_var: str, ds: xr.Dataset) -> xr.Dataset:
        """See documentation in :py:func:`xcdat.regridder.xesmf.XESMFRegridder`"""
        input_da = ds.get(data_var, None)

        if input_da is None:
            raise KeyError(
                f"The data variable '{data_var}' does not exist in the dataset."
            )

        # Create nan mask if requested, xESM will handle exisitng masks on either input or output grid
        if self._create_nan_mask:
            self._input_grid["mask"] = create_nan_mask(input_da, ["X", "Y"])

        # Align output grid dims with input grid dims, self._input_grid is derived from ``ds``.
        self._output_grid = self._output_grid.transpose(
            *[x for x in input_da.dims if x in self._output_grid],
            ...,
            missing_dims="ignore",
        )

        regridder = xe.Regridder(
            self._input_grid,
            self._output_grid,
            method=self._method,
            **self._extra_options,
        )

        output_da = regridder(input_da, keep_attrs=True)

        output_ds = xr.Dataset({data_var: output_da}, attrs=ds.attrs)
        output_ds = _preserve_bounds(ds, self._output_grid, output_ds, ["X", "Y"])

        if self._output_weights:
            if isinstance(self._output_weights, str):
                output_ds[self._output_weights] = regridder.w
            else:
                output_ds["weights"] = regridder.w

        return output_ds
