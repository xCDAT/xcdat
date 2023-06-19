import warnings
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import xarray as xr

from xcdat.axis import CFAxisKey, get_dim_coords
from xcdat.regridder import regrid2, xgcm
from xcdat.regridder.xgcm import XGCMVerticalMethods
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

    >>> import xcdat

    Use RegridderAccessor class:

    >>> ds = xcdat.open_dataset("...")
    >>>
    >>> ds.regridder.<attribute>
    >>> ds.regridder.<method>
    >>> ds.regridder.<property>

    Regrid a variable using tool specific methods:

    >>> ds.regridder.horizontal_xesmf("ts", output_grid, method="bilinear")
    >>>
    >>> ds.regridder.vertical_xgcm("ts", output_grid, method="linear")

    Parameters
    ----------
    dataset : xr.Dataset
        The Dataset to attach this accessor.
    """

    def __init__(self, dataset: xr.Dataset):
        self._ds: xr.Dataset = dataset

    @property
    def grid(self) -> xr.Dataset:
        """
        The `X`, `Y`, and `Z` axes are extracted from the Dataset and returned
        in a new ``xr.Dataset``.

        Returns
        -------
        xr.Dataset
            Containing grid axes.

        Raises
        ------
        ValueError
            If axis dimension coordinate variable is not correctly identified.
        ValueError
            If axis has multiple dimensions (only one is expected).
        """
        with xr.set_options(keep_attrs=True):
            coords = {}
            axis_names: List[CFAxisKey] = ["X", "Y", "Z"]

            for axis in axis_names:
                try:
                    data, bnds = self._get_axis_data(axis)
                except KeyError:
                    continue

                coords[data.name] = data.copy()

                if bnds is not None:
                    coords[bnds.name] = bnds.copy()

        ds = xr.Dataset(coords, attrs=self._ds.attrs)

        ds = ds.bounds.add_missing_bounds(axes=["X", "Y", "Z"])

        return ds

    def _get_axis_data(
        self, name: CFAxisKey
    ) -> Tuple[xr.DataArray, Optional[xr.DataArray]]:
        coord_var = get_dim_coords(self._ds, name)

        if isinstance(coord_var, xr.Dataset):
            raise ValueError(
                f"Multiple '{name}' axis dims were found in this dataset, "
                f"{list(coord_var.dims)}. Please drop the unused dimension(s) before "
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
        method: str = "bilinear",
        periodic: bool = False,
        extrap_method: Optional[str] = None,
        extrap_dist_exponent: Optional[float] = None,
        extrap_num_src_pnts: Optional[int] = None,
        ignore_degenerate: bool = True,
        **options: Any,
    ) -> xr.Dataset:
        """
        Extension of ``xESMF`` regridder.

        This method extends ``xESMF`` by automatically constructing the ``xe.XESMFRegridder``
        object and ensuring bounds and metadata are preseved in the output dataset.

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
        data_var : str
            Name of the variable to regrid.
        output_grid : xr.Dataset
            Dataset containing the output grid axes.
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

        >>> output_data = ds.regridder.horizontal_xesmf("ts", output_grid)

        Passing additional values to ``xe.XESMFRegridder``:

        >>> output_data = ds.regridder.horizontal_xesmf("ts", output_grid, unmapped_to_nan=True)
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
        self, data_var: str, output_grid: xr.Dataset, **options: Any
    ) -> xr.Dataset:
        """
        Pure python implementation of CDMS2's Regrid2 horizontal regridder.

        Parameters
        ----------
        data_var : str
            Name of the varibale to regrid.
        output_grid : xr.Dataset
            Dataset containing the output grid axes.
        **options : Any
            Additional arguments passed to Regrid2.

        Examples
        --------
        Import xCDAT:

        >>> import xcdat

        Open a dataset:

        >>> ds = xcdat.open_dataset("...")

        Create output grid:

        >>> output_grid = xcdat.create_gaussian_grid(32)

        Regrid the "ts" variable:

        >>> output_data = ds.regridder.horizontal_regrid2("ts", output_grid)
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
        .. deprecated:: v0.6.0
            `horizontal` will be deprecated, please migrate to using tool specific methods,
            e.g. :py:func:`xarray.Dataset.regridder.horizontal_xesmf` or :py:func:`xarray.Dataset.regridder.horizontal_regrid2`.

        Apply horizontal regridding to ``data_var`` of the current
        ``xr.Dataset`` to ``output_grid``.

        When might ``Regrid2`` be preferred over ``xESMF``?

        If performing conservative regridding from a high/medium resolution lat/lon grid to a
        coarse lat/lon target, ``Regrid2`` may provide better results as it assumes grid cells
        with constant latitudes and longitudes while ``xESMF`` assumes the cells are connected
        by Great Circles [1]_.

        Supported tools, methods and grids:

        - xESMF (https://pangeo-xesmf.readthedocs.io/en/latest/)
           - Methods:

             - Bilinear
             - Conservative
             - Conservative Normed
             - Patch
             - Nearest s2d
             - Nearest d2s
           - Grids:

             - Rectilinear
             - Curvilinear
           - Find options at :py:func:`xcdat.regridder.xesmf.XESMFRegridder`
        - Regrid2
           - Methods:

             - Conservative
           - Grids:

             - Rectilinear
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

        References
        ----------
        .. [1] https://earthsystemmodeling.org/docs/release/ESMF_8_1_0/ESMF_refdoc/node5.html#SECTION05012900000000000000

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
        warnings.warn(
            "`horizontal` will be deprecated, please migrate to using tool "
            "specific methods, e.g. `horizontal_xesmf` or `horizontal_regrid2`",
            DeprecationWarning,
            stacklevel=2,
        )

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

    def vertical_xgcm(
        self,
        data_var: str,
        output_grid: xr.Dataset,
        method: XGCMVerticalMethods = "linear",
        target_data: Optional[Union[str, xr.DataArray]] = None,
        grid_positions: Optional[Dict[str, str]] = None,
        periodic: bool = False,
        extra_init_options: Optional[Dict[str, Any]] = None,
        **options,
    ) -> xr.Dataset:
        """
        Extension of ``xgcm`` regridder.

        This method extends ``xgcm`` by automatically constructing the
        ``Grid`` object, transposing the output data to match the dimensional
        order of the input data, and ensuring bounds and metadata are preserved
        in the output dataset.

        The ``method`` argument can take any of the following values: `linear`, `log`, or
        `conservative`.

        The ``linear`` and ``log`` methods require a single dimension position, which can
        usually be automatically derived. A custom position can be specified using
        the `grid_positions` argument.

        The ``conservative`` method requires multiple dimension positions, e.g.,
        {"center": "xc", "left": "xg"} which can be passed using the `grid_positions`
        argument.

        If ``target_data`` is ``None`` then the regridding process will simply transform
        the ``data_var`` onto the ``output_grid`` and no conversion is done. If ``target_data``
        is provided then the ``data_var`` is transformed onto the ``ouput_grid`` and the data
        is converted with respect to ``target_data``.

        The ``extra_init_options`` argument are additional values passed to the ``xgcm.Grid``
        constructor. A description of these arguments can be found on `XGCM's Grid documentation <https://xgcm.readthedocs.io/en/latest/api.html#xgcm.Grid.__init__>`_.

        The ``**options`` arguments are additional values passed to the ``xgcm.Grid.transform``
        method. A description of these arguments can be found on `XGCM's Grid.transform documentation <https://xgcm.readthedocs.io/en/latest/api.html#xgcm.Grid.transform>`_.

        Parameters
        ----------
        data_var : str
            Name of the variable to regrid.
        output_grid : xr.Dataset
            Dataset containing the output grid axes.
        method : XGCMVerticalMethods
            Regridding method, defaults to "linear".
        target_data : Optional[Union[str, xr.DataArray]]
            Data to transform onto (e.g. a tracer like density or temperature).
            Defaults to ``None``.
        grid_positions : Optional[Dict[str, str]]
            Mapping of axes point position on the grid. Defaults to ``None``,
            which the regridder will try to infer this mapping.
        periodic : bool
            Whether the grid is periodic, defaults to `False`.
        extra_init_options : Optional[Dict[str, Any]]
            Extra values to pass to the ``xgcm.Grid`` constructor, defaults to ``None``.
        **options : Optional[Dict[str, Any]]
            Extra values to pass to the ``xgcm.Grid.transform`` method.

        Raises
        ------
        KeyError
            If data variable does not exist in the Dataset.
        ValueError
            If ``method`` is not valid.

        Examples
        --------
        Import xCDAT:

        >>> import xcdat

        Open a dataset:

        >>> ds = xcdat.open_dataset("...")

        Create output grid:

        >>> output_grid = xcdat.create_grid(lev=np.linspace(1000, 1, 5))

        Regrid "so" variable from input to output levels using the "linear" method:

        >>> output_data = ds.regridder.vertical_xgcm("so", output_grid)

        Regrid "so", additionally convert from model to pressure units:

        >>> # Create pressure variable
        >>> ds["pressure"] = (ds["hyam"] * ds["P0"] + ds["hybm"] * ds["PS"]).transpose(**ds["T"].dims)
        >>>
        >>> output_data = ds.regridder.vertical_xgcm("so", output_grid, target_data="pressure")

        Passing additional values to ``xgcm.Grid`` and ``xgcm.Grid.transform``:

        >>> extra_init_options = {"boundary": "fill", "fill_value": 1e27}
        >>> output_data = ds.regridder.vertical_xgcm("so", output_grid, extra_init_options=extra_init_options, mask_edges=True)
        """
        regridder = VERTICAL_REGRID_TOOLS["xgcm"](
            self._ds,
            output_grid,
            method=method,
            target_data=target_data,
            grid_positions=grid_positions,
            periodic=periodic,
            extra_init_options=extra_init_options,
            **options,
        )

        return regridder.vertical(data_var, self._ds)

    def vertical(
        self,
        data_var: str,
        output_grid: xr.Dataset,
        tool: VerticalRegridTools = "xgcm",
        **options: Any,
    ) -> xr.Dataset:
        """
        .. deprecated:: v0.6.0
            `vertical` is being deprecated, please migrate to using tool specific methods,
            e.g. :py:func:`xarray.Dataset.regridder.vertical_xgcm`.

        Apply vertical regridding to ``data_var`` of the current ``xr.Dataset``
        to ``output_grid``.

        Supported tools:

        - xgcm (https://xgcm.readthedocs.io/en/latest/index.html)
           - Methods:

             - Linear
             - Conservative
             - Log
           - Find options at :py:func:`xcdat.regridder.xgcm.XGCMRegridder`

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

        >>> output_grid = xcdat.create_grid(lev=np.linspace(1000, 1, 20))

        Regrid variable using "xgcm":

        >>> ds.regridder.vertical("so", output_grid, method="linear")
        """
        warnings.warn(
            "`vertical` is being deprecated, please migrate to using "
            "tool specific methods, e.g. `vertical_xgcm`.",
            DeprecationWarning,
            stacklevel=2,
        )

        try:
            regrid_tool = VERTICAL_REGRID_TOOLS[tool]
        except KeyError as e:
            raise ValueError(
                f"Tool {e!s} does not exist, valid choices "
                f"{list(VERTICAL_REGRID_TOOLS)}"
            )
        regridder = regrid_tool(self._ds, output_grid, **options)
        output_ds = regridder.vertical(data_var, self._ds)

        return output_ds
