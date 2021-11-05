from typing import Any, Optional

import xarray as xr

from xcdat.regridder import xesmf

REGRID_TOOLS = {
    "xesmf": xesmf.XESMFRegridder,
}


@xr.register_dataset_accessor(name="regridder")
class DatasetRegridderAccessor:
    """xarray dataset accessor for access to regridding."""

    def __init__(self, ds: xr.Dataset):
        self._ds = ds

    def regrid(
        self,
        dst_grid: xr.Dataset,
        tool: str,
        method: str,
        data_var: Optional[str] = None,
        **options: Any,
    ) -> xr.Dataset:
        """
        Applys spatial regridding to variable for rectilinear and curvilinear grids.

        Supported tools:

        - xESMF

        Parameters
        ----------
        dst_grid : xr.Dataset
            Dataset containing destination grid.
        tool : str
            Name of the regridding tool.
        method : str
            Method used for regridding, see specific regridder documentation
            for supported methods.
        data_var : Optional[str]
            The name of the data variable inside the dataset to regrid.
            If None, an inference to the desired data variable inside is
            attempted with the Dataset's "xcdat_infer" attr and
            ``get_inferred_var()``, by default None.
        **options : Any
            These options are passed to the tool being used for regridding.
            See specific regridder documentation for available options.

        Returns
        -------
        xr.Dataset

        Raises
        ------
        ValueError
            If tool is not supported.

        Examples
        --------
        Import:

        >>> import xcdat

        Open a dataset and limit to a single variable:

        >>> ds = xcdat.open_dataset("path/to/file.nc", var="tas")

        Create destination grid:

        >>> out_grid = xcdat.regridder.grid.create_uniform_grid(-90, 90, 4.0, -180, 180, 5.0)

        Regrid variable in dataset:

        >>> ds.regridder.regrid(out_grid, tool="xesmf", method="bilinear")
        """
        try:
            regridder = REGRID_TOOLS[tool](
                self._ds, dst_grid, method=method, data_var=data_var, **options
            )
        except KeyError as e:
            raise ValueError(
                f"Tool {e!s} does not exist, valid choices {list(REGRID_TOOLS)}"
            )

        return regridder.regrid(self._ds)
