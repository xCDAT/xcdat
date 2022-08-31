from typing import Any, Dict, Literal, Tuple

import xarray as xr

from xcdat.axis import CFAxisName, get_axis_coord

# esmpy, xesmf dependency not solved for osx-arm64 https://github.com/conda-forge/esmpy-feedstock/issues/55
# _importorskip is another option if these accumulate https://github.com/pydata/xarray/blob/main/xarray/tests/__init__.py#L29-L60
# discussion: https://github.com/xCDAT/xcdat/issues/315
try:
    from xcdat.regridder import regrid2, xesmf
    RegridTool = Literal["xesmf", "regrid2"]
    REGRID_TOOLS = {
        "xesmf": xesmf.XESMFRegridder,
        "regrid2": regrid2.Regrid2Regridder,
    }
    xesmfAvailable = True
except ImportError:
    from xcdat.regridder import regrid2
    RegridTool = Literal["regrid2"]
    REGRID_TOOLS = {
        "regrid2": regrid2.Regrid2Regridder,
    }
    xesmfAvailable = False


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
            If axis data variable is not correctly identified.
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

    def _get_axis_data(self, name: CFAxisName) -> Tuple[xr.DataArray, xr.DataArray]:
        coord_var = get_axis_coord(self._ds, name)

        try:
            bounds_var = self._ds.bounds.get_bounds(name)
        except KeyError:
            bounds_var = None

        return coord_var, bounds_var

    if xesmfAvailable:
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

            Regrid data to output grid using regrid2:

            >>> ds.regridder.horizontal_regrid2("ts", output_grid)
            """
            regridder = REGRID_TOOLS["xesmf"](self._ds, output_grid, **options)

            return regridder.horizontal(data_var, self._ds)

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
        regridder = REGRID_TOOLS["regrid2"](self._ds, output_grid, **options)

        return regridder.horizontal(data_var, self._ds)

    def horizontal(
        self,
        data_var: str,
        output_grid: xr.Dataset,
        tool: RegridTool = "xesmf",
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
        try:
            regrid_tool = REGRID_TOOLS[tool]
        except KeyError as e:
            raise ValueError(
                f"Tool {e!s} does not exist, valid choices {list(REGRID_TOOLS)}"
            )

        regridder = regrid_tool(self._ds, output_grid, **options)

        output_ds = regridder.horizontal(data_var, self._ds)

        return output_ds
