from typing import Any, Dict, Literal, Tuple

import xarray as xr

from xcdat.regridder import regrid2, xesmf

RegridTool = Literal["xesmf", "regrid2"]

REGRID_TOOLS = {
    "xesmf": xesmf.XESMFRegridder,
    "regrid2": regrid2.Regrid2Regridder,
}


@xr.register_dataset_accessor(name="regridder")
class RegridderAccessor:
    """xarray dataset accessor for access to regridding."""

    def __init__(self, ds: xr.Dataset):
        self._ds: xr.Dataset = ds

    def _get_axis_data(
        self, name: str, standard_name: str
    ) -> Tuple[xr.DataArray, xr.DataArray]:
        try:
            axis = self._ds.cf[name]
        except KeyError:
            raise KeyError(
                f"{standard_name} axis could not be correctly identified in the Dataset"
            )

        try:
            axis_bnds = self._ds.bounds.get_bounds(axis.name)
        except KeyError:
            axis_bnds = None

        return axis, axis_bnds

    @property
    def grid(self) -> xr.Dataset:
        """
        Extracts grid information from a `xr.Dataset`.

        Returns
        -------
        xr.Dataset
            With data variables describing the grid.

        Raises
        ------
        ValueError
            If axis data variable is not correctly identified.
        """
        x, x_bnds = self._get_axis_data("X", "Longitude")

        y, y_bnds = self._get_axis_data("Y", "Latitude")

        with xr.set_options(keep_attrs=True):
            coords = {x.name: x.copy(), y.name: y.copy()}

            if x_bnds is not None:
                coords[x_bnds.name] = x_bnds.copy()

            if y_bnds is not None:
                coords[y_bnds.name] = y_bnds.copy()

        ds = xr.Dataset(coords, attrs=self._ds.attrs)

        ds = ds.bounds.add_missing_bounds()

        return ds

    def regrid(
        self,
        data_var: str,
        dst_grid: xr.Dataset,
        tool: RegridTool = "xesmf",
        **options: Dict[str, Any],
    ) -> xr.Dataset:
        """
        Applies spatial regridding to variable for rectilinear and curvilinear grids.

        Supported tools:

        - xESMF (https://pangeo-xesmf.readthedocs.io/en/latest/)
        - Regrid2

        Parameters
        ----------
        data_var: str
            Name of the variable in the Dataset to regrid.
        dst_grid : xr.Dataset
            Dataset containing destination grid.
        tool : str
            Name of the regridding tool.
        **options : Dict[str, Any]
            These options are passed to the tool being used for regridding.
            See specific regridder documentation for available options.

        Returns
        -------
        xr.Dataset
            With the ``data_var`` variable on the grid defined in ``dst_grid``.

        Raises
        ------
        ValueError
            If tool is not supported.

        Examples
        --------
        Import:

        >>> import xcdat
        >>> from xcdat.regridder import grid

        Open a dataset:

        >>> ds = xcdat.open_dataset("path/to/file.nc")

        Create destination grid:

        >>> out_grid = grid.create_uniform_grid(-90, 90, 4.0, -180, 180, 5.0)

        Regrid variable using "xesmf":

        >>> ds.regridder.regrid("tas", out_grid, tool="xesmf", method="bilinear")

        Regrid variable using "regrid2":

        >>> ds.regridder.regrid("tas", out_grid, tool="regrid2")
        """
        try:
            regridder = REGRID_TOOLS[tool](self._ds, dst_grid, **options)
        except KeyError as e:
            raise ValueError(
                f"Tool {e!s} does not exist, valid choices {list(REGRID_TOOLS)}"
            )

        return regridder.regrid(data_var, self._ds)
