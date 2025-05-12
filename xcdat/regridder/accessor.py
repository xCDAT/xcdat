from typing import Any, Literal

import xarray as xr

from xcdat.axis import CFAxisKey, get_coords_by_name, get_dim_coords
from xcdat.regridder import regrid2, xesmf, xgcm
from xcdat.regridder.grid import _validate_grid_has_single_axis_dim

HorizontalRegridTools = Literal["xesmf", "regrid2"]
HORIZONTAL_REGRID_TOOLS = {
    "regrid2": regrid2.Regrid2Regridder,
    "xesmf": xesmf.XESMFRegridder,
}

VerticalRegridTools = Literal["xgcm"]
VERTICAL_REGRID_TOOLS = {"xgcm": xgcm.XGCMRegridder}


@xr.register_dataset_accessor(name="regridder")
class RegridderAccessor:
    """
    An accessor class that provides regridding attributes and methods for
    xarray Datasets through the ``.regridder`` attribute.

    Examples
    --------

    Import xCDAT:

    >>> import xcdat

    Use RegridderAccessor class:

    >>> ds = xcdat.open_dataset("...")
    >>>
    >>> ds.regridder.<attribute>
    >>> ds.regridder.<method>
    >>> ds.regridder.<property>

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
        Extract the `X`, `Y`, and `Z` axes from the Dataset and return a new
        ``xr.Dataset``.

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

        Examples
        --------

        Import xCDAT:

        >>> import xcdat

        Open a dataset:

        >>> ds = xcdat.open_dataset("...")

        Extract grid from dataset:

        >>> grid = ds.regridder.grid
        """
        axis_names: list[CFAxisKey] = ["X", "Y", "Z"]

        axis_coords: dict[str, xr.DataArray] = {}
        axis_bounds: dict[str, xr.DataArray] = {}
        axis_has_bounds: dict[CFAxisKey, bool] = {}

        with xr.set_options(keep_attrs=True):
            for axis in axis_names:
                coord, bounds = self._get_axis_coord_and_bounds(axis)

                if coord is not None:
                    axis_coords[str(coord.name)] = coord

                    if bounds is not None:
                        axis_bounds[str(bounds.name)] = bounds
                        axis_has_bounds[axis] = True
                    else:
                        axis_has_bounds[axis] = False

        # Create a new dataset with coordinates and bounds
        ds = xr.Dataset(
            coords=axis_coords,
            data_vars=axis_bounds,
            attrs=self._ds.attrs,
        )

        # Add bounds only for axes that do not already have them. This
        # prevents multiple sets of bounds being added for the same axis.
        # For example, curvilinear grids can have multiple coordinates for the
        # same axis (e.g., (nlat, lat) for X and (nlon, lon) for Y). We only
        # need lat_bnds and lon_bnds for the X and Y axes, respectively, and not
        # nlat_bnds and nlon_bnds.
        for axis, has_bounds in axis_has_bounds.items():
            if not has_bounds:
                ds = ds.bounds.add_bounds(axis=axis)

        return ds

    def _get_axis_coord_and_bounds(
        self, axis: CFAxisKey
    ) -> tuple[xr.DataArray | None, xr.DataArray | None]:
        try:
            coord_var = get_coords_by_name(self._ds, axis)
            if coord_var.size == 1:
                raise ValueError(
                    f"Coordinate '{coord_var}' is a singleton and cannot be used."
                )
        except (ValueError, KeyError):
            try:
                coord_var = get_dim_coords(self._ds, axis)  # type: ignore
                _validate_grid_has_single_axis_dim(axis, coord_var)
            except KeyError:
                coord_var = None

        if coord_var is None:
            return None, None

        bounds_var = None
        bounds_key = coord_var.attrs.get("bounds")
        if bounds_key:
            bounds_var = self._ds.get(bounds_key)

        return coord_var, bounds_var

    def horizontal(
        self,
        data_var: str,
        output_grid: xr.Dataset,
        tool: HorizontalRegridTools = "xesmf",
        **options: Any,
    ) -> xr.Dataset:
        """
        Transform ``data_var`` to ``output_grid``.

        When might ``Regrid2`` be preferred over ``xESMF``?

        If performing conservative regridding from a high/medium resolution lat/lon grid to a
        coarse lat/lon target, ``Regrid2`` may provide better results as it assumes grid cells
        with constant latitudes and longitudes while ``xESMF`` assumes the cells are connected
        by Great Circles [1]_.

        Supported tools, methods and grids:

        - xESMF (https://xesmf.readthedocs.io/en/latest/)
           - Methods: Bilinear, Conservative, Conservative Normed, Patch, Nearest s2d, or Nearest d2s.
           - Grids: Rectilinear, or Curvilinear.
           - Find options at :py:func:`xcdat.regridder.xesmf.XESMFRegridder`
        - Regrid2
           - Methods: Conservative
           - Grids: Rectilinear
           - Find options at :py:func:`xcdat.regridder.regrid2.Regrid2Regridder`

        Parameters
        ----------
        data_var: str
            Name of the variable to transform.
        output_grid : xr.Dataset
            Grid to transform ``data_var`` to.
        tool : str
            Name of the tool to use.
        **options : Any
            These options are passed directly to the ``tool``. See specific
            regridder for available options.

        Returns
        -------
        xr.Dataset
            With the ``data_var`` transformed to the ``output_grid``.

        Raises
        ------
        ValueError
            If tool is not supported.

        References
        ----------
        .. [1] https://earthsystemmodeling.org/docs/release/ESMF_8_1_0/ESMF_refdoc/node5.html#SECTION05012900000000000000

        Examples
        --------

        Import xCDAT:

        >>> import xcdat

        Open a dataset:

        >>> ds = xcdat.open_dataset("...")

        Create output grid:

        >>> output_grid = xcdat.create_uniform_grid(-90, 90, 4.0, -180, 180, 5.0)

        Regrid variable using "xesmf":

        >>> output_data = ds.regridder.horizontal("ts", output_grid, tool="xesmf", method="bilinear")

        Regrid variable using "regrid2":

        >>> output_data = ds.regridder.horizontal("ts", output_grid, tool="regrid2")
        """
        try:
            regrid_tool = HORIZONTAL_REGRID_TOOLS[tool]
        except KeyError as e:
            raise ValueError(
                f"Tool {e!s} does not exist, valid choices {list(HORIZONTAL_REGRID_TOOLS)}"
            ) from e

        input_grid = _get_input_grid(self._ds, data_var, ["X", "Y"])
        regridder = regrid_tool(input_grid, output_grid, **options)
        output_ds = regridder.horizontal(data_var, self._ds)

        return output_ds

    def vertical(
        self,
        data_var: str,
        output_grid: xr.Dataset,
        tool: VerticalRegridTools = "xgcm",
        **options: Any,
    ) -> xr.Dataset:
        """
        Transform ``data_var`` to ``output_grid``.

        Supported tools:

        - xgcm (https://xgcm.readthedocs.io/en/latest/index.html)
           - Methods: Linear, Conservative, Log
           - Find options at :py:func:`xcdat.regridder.xgcm.XGCMRegridder`

        Parameters
        ----------
        data_var: str
            Name of the variable to transform.
        output_grid : xr.Dataset
            Grid to transform ``data_var`` to.
        tool : str
            Name of the tool to use.
        **options : Any
            These options are passed directly to the ``tool``. See specific
            regridder for available options.

        Returns
        -------
        xr.Dataset
            With the ``data_var`` transformed to the ``output_grid``.

        Raises
        ------
        ValueError
            If tool is not supported.

        Examples
        --------

        Import xCDAT:

        >>> import xcdat

        Open a dataset:

        >>> ds = xcdat.open_dataset("...")

        Create output grid:

        >>> output_grid = xcdat.create_grid(lev=np.linspace(1000, 1, 20))

        Regrid variable using "xgcm":

        >>> output_data = ds.regridder.vertical("so", output_grid, method="linear")
        """
        try:
            regrid_tool = VERTICAL_REGRID_TOOLS[tool]
        except KeyError as e:
            raise ValueError(
                f"Tool {e!s} does not exist, valid choices "
                f"{list(VERTICAL_REGRID_TOOLS)}"
            ) from e
        input_grid = _get_input_grid(
            self._ds,
            data_var,
            [
                "Z",
            ],
        )
        regridder = regrid_tool(input_grid, output_grid, **options)
        output_ds = regridder.vertical(data_var, self._ds)

        return output_ds


def _get_input_grid(ds: xr.Dataset, data_var: str, dup_check_dims: list[CFAxisKey]):
    """
    Extract the grid from ``ds``.

    This function will remove any duplicate dimensions leaving only dimensions
    used by the ``data_var``. All extraneous dimensions and variables are
    dropped, returning only the grid.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to extract grid from.
    data_var : str
        Name of target data variable.
    dup_check_dims : list[CFAxisKey]
        List of dimensions to check for duplicates.

    Returns
    -------
    xr.Dataset
        Dataset containing grid dataset.
    """
    to_drop = []

    all_coords = set(ds.coords.keys())

    for dimension in dup_check_dims:
        coords = get_dim_coords(ds, dimension)

        if isinstance(coords, xr.Dataset):
            coord = set([get_dim_coords(ds[data_var], dimension).name])

            dimension_coords = set(ds.cf[[dimension]].coords.keys())

            # need to take the intersection after as `ds.cf[["Z"]]` will hand back data variables
            to_drop += list(dimension_coords.difference(coord).intersection(all_coords))

    input_grid = ds.drop_dims(to_drop)

    # drops extra dimensions on input grid
    grid = input_grid.regridder.grid

    # preserve mask on grid
    if "mask" in ds:
        grid["mask"] = ds["mask"].copy()

    return grid
