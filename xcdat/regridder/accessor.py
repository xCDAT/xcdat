from typing import Any, Dict, Literal, Optional, Tuple

import xarray as xr

from xcdat.axis import CFAxisKey, get_dim_coords
from xcdat.regridder import regrid2, xgcm
from xcdat.utils import _has_module

HorizontalRegridTools = Literal["xesmf", "regrid2"]
HORIZONTAL_REGRID_TOOLS = {"regrid2": regrid2.Regrid2Regridder}

# TODO: Test this conditional.
_has_xesmf = _has_module("xesmf")
if _has_xesmf:  # pragma: no cover
    from xcdat.regridder import xesmf

    HORIZONTAL_REGRID_TOOLS["xesmf"] = xesmf.XESMFRegridder  # type: ignore

VerticalRegridTools = Literal["xgcm"]
VERTICAL_REGRID_TOOLS = {"xgcm": xgcm.XGCMRegridder}


@xr.register_dataset_accessor(name="regridder")
class RegridderAccessor:
    """
    An accessor class that provides regridding attributes and methods on xarray
    Datasets through the ``.regridder`` attribute.

    Examples
    --------

    Import RegridderAccessor class:

    >>> import xcdat  # or from xcdat import regridder

    Use RegridderAccessor class:

    >>> ds = xcdat.open_dataset("/path/to/file")
    >>>
    >>> ds.regridder.<attribute>
    >>> ds.regridder.<method>
    >>> ds.regridder.<property>

    Parameters
    ----------
    dataset : xr.Dataset
        A Dataset object.
    """

    def __init__(self, dataset: xr.Dataset):
        self._ds: xr.Dataset = dataset

    @property
    def grid(self) -> xr.Dataset:
        """
        Returns ``xr.Dataset`` containing grid information.

        Returns
        -------
        xr.Dataset
            With data variables describing the grid.

        Raises
        ------
        ValueError
            If axis dimension coordinate variable is not correctly identified.
        ValueError
            If axis has multiple dimensions (only one is expected).
        """
        x, x_bnds = self._get_axis_data("X")
        y, y_bnds = self._get_axis_data("Y")

        with xr.set_options(keep_attrs=True):
            coords = {x.name: x.copy(), y.name: y.copy()}

            if x_bnds is not None:
                coords[x_bnds.name] = x_bnds.copy()

            if y_bnds is not None:
                coords[y_bnds.name] = y_bnds.copy()

        ds = xr.Dataset(coords, attrs=self._ds.attrs)

        ds = ds.bounds.add_missing_bounds()

        return ds

    def _get_axis_data(
        self, name: CFAxisKey
    ) -> Tuple[xr.DataArray, Optional[xr.DataArray]]:
        coord_var = get_dim_coords(self._ds, name)

        if isinstance(coord_var, xr.Dataset):
            raise ValueError(
                f"Multiple '{name}' axis dims were found in this dataset, "
                f"{list(coord_var.dims)}. Please drop the unused dimension(s) before"
                "getting grid information."
            )

        try:
            bounds_var = self._ds.bounds.get_bounds(name, coord_var.name)
        except KeyError:
            bounds_var = None

        return coord_var, bounds_var

    def horizontal_xesmf(
        self,
        data_var: str,
        output_grid: xr.Dataset,
        **options: Dict[str, Any],
    ) -> xr.Dataset:
        """
        Wraps the xESMF library providing access to regridding between
        structured rectilinear and curvilinear grids.

        Regrids ``data_var`` in dataset to ``output_grid``.

        Option documentation :py:func:`xcdat.regridder.xesmf.XESMFRegridder`

        Parameters
        ----------
        data_var: str
            Name of the variable in the `xr.Dataset` to regrid.
        output_grid : xr.Dataset
            Dataset containing output grid.
        options : Dict[str, Any]
            Dictionary with extra parameters for the regridder.

        Returns
        -------
        xr.Dataset
            With the ``data_var`` variable on the grid defined in ``output_grid``.

        Raises
        ------
        ValueError
            If tool is not supported.

        Examples
        --------

        Generate output grid:

        >>> output_grid = xcdat.create_gaussian_grid(32)

        Regrid data to output grid using xesmf:

        >>> ds.regridder.horizontal_xesmf("ts", output_grid)
        """
        # TODO: Test this conditional.
        if _has_xesmf:  # pragma: no cover
            regridder = HORIZONTAL_REGRID_TOOLS["xesmf"](
                self._ds, output_grid, **options
            )

            return regridder.horizontal(data_var, self._ds)
        else:  # pragma: no cover
            raise ModuleNotFoundError(
                "The `xesmf` package is required for horizontal regridding with "
                "`xesmf`. Make sure your platform supports `xesmf` and it is installed "
                "in your conda environment."
            )

    def horizontal_regrid2(
        self,
        data_var: str,
        output_grid: xr.Dataset,
        **options: Dict[str, Any],
    ) -> xr.Dataset:
        """
        Pure python implementation of CDAT's regrid2 horizontal regridder.

        Regrids ``data_var`` in dataset to ``output_grid`` using regrid2's
        algorithm.

        Options documentation :py:func:`xcdat.regridder.regrid2.Regrid2Regridder`

        Parameters
        ----------
        data_var: str
            Name of the variable in the `xr.Dataset` to regrid.
        output_grid : xr.Dataset
            Dataset containing output grid.
        options : Dict[str, Any]
            Dictionary with extra parameters for the regridder.

        Returns
        -------
        xr.Dataset
            With the ``data_var`` variable on the grid defined in ``output_grid``.

        Raises
        ------
        ValueError
            If tool is not supported.

        Examples
        --------
        Generate output grid:

        >>> output_grid = xcdat.create_gaussian_grid(32)

        Regrid data to output grid using regrid2:

        >>> ds.regridder.horizontal_regrid2("ts", output_grid)
        """
        regridder = HORIZONTAL_REGRID_TOOLS["regrid2"](self._ds, output_grid, **options)

        return regridder.horizontal(data_var, self._ds)

    def horizontal(
        self,
        data_var: str,
        output_grid: xr.Dataset,
        tool: HorizontalRegridTools = "xesmf",
        **options: Dict[str, Any],
    ) -> xr.Dataset:
        """
        Apply horizontal regridding to ``data_var`` of the current
        ``xr.Dataset`` to ``output_grid``.

        Supported tools:

        - xESMF (https://pangeo-xesmf.readthedocs.io/en/latest/)
           - Rectilinear and curvilinear grids
           - Find options at :py:func:`xcdat.regridder.xesmf.XESMFRegridder`
        - Regrid2
           - Rectilinear grids
           - Find options at :py:func:`xcdat.regridder.regrid2.Regrid2Regridder`

        Parameters
        ----------
        data_var: str
            Name of the variable in the ``xr.Dataset`` to regrid.
        output_grid : xr.Dataset
            Dataset containing output grid.
        tool : str
            Name of the regridding tool.
        **options : Dict[str, Any]
            These options are passed to the tool being used for regridding.
            See specific regridder documentation for available options.

        Returns
        -------
        xr.Dataset
            With the ``data_var`` variable on the grid defined in ``output_grid``.

        Raises
        ------
        ValueError
            If tool is not supported.

        Examples
        --------

        Create destination grid:

        >>> output_grid = xcdat.create_uniform_grid(-90, 90, 4.0, -180, 180, 5.0)

        Regrid variable using "xesmf":

        >>> ds.regridder.horizontal("ts", output_grid, tool="xesmf", method="bilinear")

        Regrid variable using "regrid2":

        >>> ds.regridder.horizontal("ts", output_grid, tool="regrid2")

        Use convenience methods:

        >>> ds.regridder.horizontal_xesmf("ts", output_grid, method="bilinear")

        >>> ds.regridder.horizontal_regrid2("ts", output_grid)
        """
        # TODO: Test this conditional.
        if tool == "xesmf" and not _has_xesmf:  # pragma: no cover
            raise ModuleNotFoundError(
                "The `xesmf` package is required for horizontal regridding with "
                "`xesmf`. Make sure your platform supports `xesmf` and it is installed "
                "in your conda environment."
            )

        try:
            regrid_tool = HORIZONTAL_REGRID_TOOLS[tool]
        except KeyError as e:
            raise ValueError(
                f"Tool {e!s} does not exist, valid choices {list(HORIZONTAL_REGRID_TOOLS)}"
            )

        regridder = regrid_tool(self._ds, output_grid, **options)
        output_ds = regridder.horizontal(data_var, self._ds)

        return output_ds

    def vertical(
        self,
        data_var: str,
        output_grid: xr.Dataset,
        tool: VerticalRegridTools = "xgcm",
        **options: Dict[str, Any],
    ) -> xr.Dataset:
        """
        Apply vertical regridding to ``data_var`` of the current
        ``xr.Dataset`` to ``output_grid``.

        Supported tools:

        - xgcm (https://xgcm.readthedocs.io/en/latest/index.html)

        Parameters
        ----------
        data_var: str
            Name of the variable in the ``xr.Dataset`` to regrid.
        output_grid : xr.Dataset
            Dataset containing output grid.
        tool : str
            Name of the regridding tool.
        **options : Dict[str, Any]
            These options are passed to the tool being used for regridding.
            See specific regridder documentation for available options.

        Returns
        -------
        xr.Dataset
            With the ``data_var`` variable on the grid defined in ``output_grid``.

        Raises
        ------
        ValueError
            If tool is not supported.

        Examples
        --------

        TODO: Update for vertical regridding
        Create destination grid:

        >>> output_grid = xcdat.create_uniform_grid(-90, 90, 4.0, -180, 180, 5.0)

        Regrid variable using "xesmf":

        >>> ds.regridder.horizontal("ts", output_grid, tool="xesmf", method="bilinear")

        Regrid variable using "regrid2":

        >>> ds.regridder.horizontal("ts", output_grid, tool="regrid2")
        """
        try:
            regrid_tool = VERTICAL_REGRID_TOOLS[tool]
        except KeyError as e:
            raise ValueError(
                f"Tool {e!s} does not exist, valid choices {list(VERTICAL_REGRID_TOOLS)}"
            )
        regridder = regrid_tool(self._ds, output_grid, **options)
        output_ds = regridder.vertical(data_var, self._ds)

        return output_ds
