from typing import Any, Dict, Literal, Optional, Union, get_args

import xarray as xr
from xgcm import Grid

from xcdat.logger import setup_custom_logger
from xcdat.regridder.base import BaseRegridder, preserve_bounds

XGCMVerticalMethods = Literal["linear", "conservative", "log"]

logger = setup_custom_logger(__name__)


class XGCMRegridder(BaseRegridder):
    def __init__(
        self,
        input_grid: xr.Dataset,
        output_grid: xr.Dataset,
        method: XGCMVerticalMethods = "linear",
        target_data: Optional[Union[str, xr.DataArray]] = None,
        grid_positions: Optional[Dict[str, str]] = None,
        periodic: bool = False,
        extra_init_options: Optional[Dict[str, Any]] = None,
        **options,
    ):
        """Extension of ``xgcm`` regridder.

        The ``XGCMRegridder`` extends ``xgcm`` by automatically constructing the
        ``Grid`` object, transposing the output data to match the input data's
        dimensional order, and ensuring missing bounds are generated on the
        output data.

        Linear and log methods require a single dimension position, which can
        usually be automatically derived.

        Conservative regridding requires multiple dimension positions, e.g.,
        {"center": "xc", "left": "xg"}.

        Parameters
        ----------
        input_grid : xr.Dataset
            Contains source grid coordinates.
        output_grid : xr.Dataset
            Contains destination grid coordinates.
        method : XGCMVerticalMethods
            Regridding method, by default "linear". Options are
               - linear (default)
               - log
               - conservative
        target_data : Optional[Union[str, xr.DataArray]]
            Data to transform target data onto, by default None.
        grid_positions : Optional[Dict[str, str]]
            Mapping of dimension positions, by default None. If ``None`` then an
            attempt is made to derive this argument.
        periodic : bool
            Whether the grid is periodic, by default False.
        extra_init_options : Optional[Dict[str, Any]]
            Extra options passed to the ``xgcm.Grid`` constructor, by default
            None.
        options : Optional[Dict[str, Any]]
            Extra options passed to the ``xgcm.Grid.trasnform`` method.

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
        >>> from xcdat.regridder import xgcm

        Open a dataset:

        >>> ds = xcdat.open_dataset("t.nc")

        Create output grid:

        >>> output_grid = xcdat.create_grid(lev=np.linspace(1000, 1, 5))

        Create theta:

        >>> ds["pressure"] = (ds["hyam"] * ds["P0"] + ds["hybm"] * ds["PS"]).transpose(**ds["T"].dims)

        Create regridder:

        >>> regridder = xgcm.XGCMRegridder(ds, output_grid, method="linear", target_data="pressure")

        Regrid data:

        >>> data_new_grid = regridder.vertical("T", ds)
        """
        super().__init__(input_grid, output_grid)

        if method not in get_args(XGCMVerticalMethods):
            raise ValueError(
                f"{method!r} is invalid, possible choices: "
                f"{', '.join(get_args(XGCMVerticalMethods))}"
            )

        self._method = method
        self._target_data = target_data
        self._grid_coords = grid_positions

        if extra_init_options is None:
            extra_init_options = {}

        extra_init_options["periodic"] = periodic

        self._extra_init_options = extra_init_options
        self._extra_options = options

    def horizontal(self, data_var: str, ds: xr.Dataset) -> xr.Dataset:
        """Placeholder for base class."""
        raise NotImplementedError()

    def vertical(self, data_var: str, ds: xr.Dataset) -> xr.Dataset:
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
        RuntimeError
            If "Z" coordinate is not detected.

        Examples
        --------
        Create output grid:

        >>> output_grid = xcdat.create_grid(np.linspace(1000, 1, 20))

        Create regridder:

        >>> regridder = xgcm.XGCMRegridder(ds, output_grid, method="linear", target_data="pressure")

        Regrid data:

        >>> data_new_grid = regridder.verical("T", ds)
        """
        try:
            output_coord_z = self._output_grid.cf["Z"]
        except KeyError:
            raise RuntimeError("Could not determine 'Z' coordinate in output dataset")

        if self._grid_coords is None:
            grid_coords = self._get_grid_coords()
        else:
            # correctly format argument
            grid_coords = {"Z": self._grid_coords}

        grid = Grid(ds, coords=grid_coords, **self._extra_init_options)

        target_data: Union[str, xr.DataArray, None] = None

        try:
            target_data = ds[self._target_data]
        except ValueError:
            target_data = self._target_data
        except KeyError:
            if self._target_data is not None and isinstance(self._target_data, str):
                raise RuntimeError(
                    f"Could not find target variable {self._target_data!r} in dataset"
                )

            target_data = None

        # assumes new verical coordinate has been calculated and stored as `pressure`
        # TODO: auto calculate pressure reference http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#parametric-v-coord
        #       cf_xarray only supports two ocean s-coordinate and ocean_sigma_coordinate
        output_da = grid.transform(
            ds[data_var],
            "Z",
            output_coord_z,
            target_data=target_data,
            method=self._method,
            **self._extra_options,
        )

        input_z_name = ds.cf["Z"].name

        # xgcm always outputs vertical coordinate as 'lev'
        if input_z_name != "lev":
            output_da = output_da.rename({"lev": input_z_name})

        output_da = output_da.transpose(*ds[data_var].dims)

        output_ds = xr.Dataset({data_var: output_da}, attrs=ds.attrs)

        output_ds = preserve_bounds(ds, self._output_grid, output_ds)

        output_ds = output_ds.bounds.add_missing_bounds()

        return output_ds

    def _get_grid_coords(self) -> Dict[str, Dict[str, str]]:
        if self._method == "conservative":
            raise RuntimeError(
                "Conservative regridding requires a second point position, pass these "
                "manually"
            )

        try:
            coord_z = self._input_grid.cf["Z"]
        except KeyError:
            raise RuntimeError("Could not determine 'Z' coordinate in input dataset")

        try:
            bounds_z = self._input_grid.bounds.get_bounds("Z")
        except KeyError:
            raise RuntimeError("Could not determine 'Z' bounds in input dataset")

        # handle simple point positions based on point and bounds
        if (coord_z[0] > bounds_z[0][0] and coord_z[0] < bounds_z[0][1]) or (
            coord_z[0] < bounds_z[0][0] and coord_z[0] > bounds_z[0][1]
        ):
            grid_positions = {"center": coord_z.name}
        elif coord_z[0] == bounds_z[0][0]:
            grid_positions = {"left": coord_z.name}
        elif coord_z[0] == bounds_z[0][1]:
            grid_positions = {"right": coord_z.name}
        else:
            raise RuntimeError(
                "Could not determine the grid point positions, pass these manually "
                "using the `grid_positions` argument"
            )

        return {"Z": grid_positions}
