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
    ):
        """Wrapper class for xESMF regridder class.


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
            The exponent to raise the distance to when calculating weights for 
            the extrapolation method.
        extrap_num_src_pnts : int
            The number of source points to use for the extrapolation methods 
            that use more than one source point.

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
        self._regridder: xe.XESMFRegridder = None

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
            )

        output_da = self._regridder(input_da, keep_attrs=True)

        output_ds = xr.Dataset({data_var: output_da}, attrs=ds.attrs)

        for dim_name, var_names in ds.cf.axes.items():
            if dim_name in ("X", "Y"):
                output_ds = output_ds.bounds.add_bounds(var_names[0])
            else:
                try:
                    dim_bounds = ds.cf.get_bounds(dim_name)
                except KeyError:
                    output_ds = output_ds.bounds.add_bounds(var_names[0])
                else:
                    output_ds[dim_bounds.name] = dim_bounds

        return output_ds
