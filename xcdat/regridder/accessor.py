from __future__ import annotations

from typing import Any, List, Literal, Tuple

import xarray as xr

from xcdat.axis import CF_ATTR_MAP, CFAxisKey, get_dim_coords
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
    ) -> Tuple[xr.DataArray | xr.Dataset, xr.DataArray]:
        coord_var = get_dim_coords(self._ds, name)

        _validate_grid_has_single_axis_dim(name, coord_var)

        coord_var = self._ensure_cf_compliance(coord_var, name)  # type: ignore

        try:
            bounds_var = self._ds.bounds.get_bounds(name, coord_var.name)
        except KeyError:
            bounds_var = None

        return coord_var, bounds_var

    def _ensure_cf_compliance(
        self, coord_var: xr.DataArray, name: CFAxisKey
    ) -> xr.DataArray:
        """Ensure that the coordinate variable is CF-compliant.

        This function adds the "axis" and "standard_name" attributes to the
        coordinates if they are not already present. Coordinates must be
        CF-compliant in order for xESMF to interpret them using CF-xarray.

        Parameters
        ----------
        coords : xr.DataArray
            Coordinates to make CF compliant.
        name : CFAxisKey
            Name of the axis.

        Returns
        -------
        xr.DataArray
            CF compliant coordinates.
        """
        coord_var_new = coord_var.copy()
        cf_attrs = CF_ATTR_MAP[name]

        if "axis" not in coord_var_new.attrs:
            coord_var_new.attrs["axis"] = cf_attrs["axis"]

        if "standard_name" not in coord_var_new.attrs:
            coord_var_new.attrs["standard_name"] = cf_attrs["coordinate"]

        return coord_var_new

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


def _get_input_grid(ds: xr.Dataset, data_var: str, dup_check_dims: List[CFAxisKey]):
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
    dup_check_dims : List[CFAxisKey]
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
