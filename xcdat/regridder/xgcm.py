from typing import Any, Dict, Hashable, Literal, Optional, Union, get_args

import xarray as xr
from xgcm import Grid

from xcdat._logger import _setup_custom_logger
from xcdat.axis import get_dim_coords
from xcdat.regridder.base import BaseRegridder, _preserve_bounds

XGCMVerticalMethods = Literal["linear", "conservative", "log"]

logger = _setup_custom_logger(__name__)


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
        """
        See documentation at `xcdat.regridder.accessor.RegridderAccessor.vertical_xgcm`.
        """
        super().__init__(input_grid, output_grid)

        if method not in get_args(XGCMVerticalMethods):
            raise ValueError(
                f"{method!r} is invalid, possible choices: "
                f"{', '.join(get_args(XGCMVerticalMethods))}"
            )

        self._method = method
        self._target_data = target_data
        self._grid_positions = grid_positions

        if extra_init_options is None:
            extra_init_options = {}

        extra_init_options["periodic"] = periodic

        self._extra_init_options = extra_init_options
        self._extra_options = options

    def horizontal(self, data_var: str, ds: xr.Dataset) -> xr.Dataset:
        """Placeholder for base class."""
        raise NotImplementedError()

    def vertical(self, data_var: str, ds: xr.Dataset) -> xr.Dataset:
        """
        Apply xgcm vertical regridding to ``data_var`` in ``ds``.

        Parameters
        ----------
        data_var : str
            Name of the variable to regrid.
        ds : xr.Dataset
            Input dataset containing ``data_var``.

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
        """
        try:
            output_coord_z = get_dim_coords(self._output_grid, "Z")
        except KeyError:
            raise RuntimeError("Could not determine 'Z' coordinate in output dataset")

        if self._grid_positions is None:
            grid_coords = self._get_grid_positions()
        else:
            # correctly format argument
            grid_coords = {"Z": self._grid_positions}

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

        # when the order of dimensions are mismatched, the output data will be
        # transposed to match the input dimension order
        if output_da.dims != ds[data_var].dims:
            input_coord_z = get_dim_coords(ds, "Z")

            output_order = [
                x.replace(input_coord_z.name, output_coord_z.name)  # type: ignore[attr-defined]
                for x in ds[data_var].dims
            ]

            output_da = output_da.transpose(*output_order)

        if target_data is None:
            output_da.attrs = ds[data_var].attrs.copy()
        else:
            output_da.attrs = target_data.attrs.copy()  # type: ignore[union-attr]

        output_ds = xr.Dataset({data_var: output_da}, attrs=ds.attrs.copy())
        output_ds = _preserve_bounds(ds, self._output_grid, output_ds, ["Z"])

        return output_ds

    def _get_grid_positions(self) -> Dict[str, Union[Any, Hashable]]:
        if self._method == "conservative":
            raise RuntimeError(
                "Conservative regridding requires a second point position, pass these "
                "manually"
            )

        try:
            coord_z = get_dim_coords(self._input_grid, "Z")
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
