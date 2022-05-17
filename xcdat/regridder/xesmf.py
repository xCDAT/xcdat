import xarray as xr
import xesmf as xe

from xcdat.logger import setup_custom_logger
from xcdat.regridder.base import BaseRegridder

logger = setup_custom_logger(__name__)

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
        extrap_method: str = None,
        extrap_dist_exponent: float = None,
        extrap_num_src_pnts: int = None,
        **options,
    ):
        """
        Wrapper class for xESMF regridder class.

        Available options: ``method``, ``periodic``, ``extrap_method``, ``extrap_dist_exponent``, ``extrap_num_src_pnts``

        Parameters
        ----------
        input_grid : xr.Dataset
            Contains source grid coordinates.
        output_grid : xr.Dataset
            Contains desintation grid coordinates.
        method : str
            Regridding method. Options are
               - bilinear
               - conservative
               - conservative_normed
               - patch
               - nearest_s2d
               - nearest_d2s
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
            If ``method`` is not valid.
        ValueError
            If ``extrap_method`` is not valid.

        Examples
        --------
        Import xCDAT:

        >>> import xcdat
        >>> from xcdat.regridder import xesmf

        Open a dataset:

        >>> ds = xcdat.open_dataset("ts.nc")

        Create output grid:

        >>> output_grid = xcdat.create_gaussian_grid(32)

        Create regridder:

        >>> regridder = xesmf.XESMFRegridder(ds, output_grid, method="bilinear")

        Regrid data:

        >>> data_new_grid = regridder.horizontal("ts", ds, periodic=True)
        """
        super().__init__(input_grid, output_grid, **options)

        if method not in VALID_METHODS:
            raise ValueError(
                f"{method!r} is not valid, possible options: {', '.join(VALID_METHODS)}"
            )

        if extrap_method is not None and extrap_method not in VALID_EXTRAP_METHODS:
            raise ValueError(
                f"{extrap_method!r} is not valid, possible options: {', '.join(VALID_EXTRAP_METHODS)}"
            )

        self._regridder = xe.Regridder(
            self._input_grid,
            self._output_grid,
            method,
            periodic=periodic,
            extrap_method=extrap_method,
            extrap_dist_exponent=extrap_dist_exponent,
            extrap_num_src_pnts=extrap_num_src_pnts,
        )

    def horizontal(self, data_var: str, ds: xr.Dataset) -> xr.Dataset:
        """
        Regrid ``data_var`` in ``ds`` to output grid.

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

        Notes
        -----

        Examples
        --------
        Import xCDAT:

        >>> import xcdat
        >>> from xcdat.regridder import xesmf

        Open a dataset:

        >>> ds = xcdat.open_dataset("ts.nc")

        Create output grid:

        >>> output_grid = xcdat.create_gaussian_grid(32)

        Create regridder:

        >>> regridder = xesmf.XESMFRegridder(ds, output_grid, method="bilinear")

        Regrid data:

        >>> data_new_grid = regridder.horizontal("ts", ds)
        """
        input_data_var = ds.get(data_var, None)

        if input_data_var is None:
            raise KeyError(
                f"The data variable '{data_var}' does not exist in the dataset."
            )

        output_da = self._regridder(input_data_var, keep_attrs=True)

        data_vars = {}

        for axes, variable_names in output_da.cf.axes.items():
            variable_name = variable_names[0]

            try:
                if axes in ("X", "Y"):
                    bounds = self._output_grid.bounds.get_bounds(variable_name)
                else:
                    bounds = ds.bounds.get_bounds(variable_name)
            except KeyError:
                logger.debug(f"Could not find bounds for {axes!r}")
            else:
                data_vars[bounds.name] = bounds.copy()

        data_vars[data_var] = output_da

        output_ds = xr.Dataset(data_vars)

        return output_ds
