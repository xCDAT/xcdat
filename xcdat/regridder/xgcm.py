from collections.abc import Hashable
from typing import Any, Literal, get_args

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
        target_data: str | xr.DataArray | None = None,
        grid_positions: dict[str, str] | None = None,
        periodic: bool = False,
        extra_init_options: dict[str, Any] | None = None,
        **options: Any,
    ):
        """
        Extension of ``xgcm`` regridder.

        The ``XGCMRegridder`` extends ``xgcm`` by automatically constructing the
        ``Grid`` object, transposing the output data to match the dimensional
        order of the input data, and ensuring bounds and metadata are preserved
        in the output dataset.

        Linear and log methods require a single dimension position, which can
        usually be automatically derived. A custom position can be specified using
        the `grid_positions` argument.

        Conservative regridding requires multiple dimension positions, e.g.,
        {"center": "xc", "left": "xg"} which can be passed using the `grid_positions`
        argument.

        ``xgcm.Grid`` can be passed additional arguments using ``extra_init_options``.
        These arguments can be found on `XGCM's Grid documentation <https://xgcm.readthedocs.io/en/latest/api.html#xgcm.Grid.__init__>`_.

        ``xgcm.Grid.transform`` can be passed additional arguments using ``options``.
        These arguments can be found on `XGCM's Grid.transform documentation <https://xgcm.readthedocs.io/en/latest/api.html#xgcm.Grid.transform>`_.

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
        target_data : str | xr.DataArray | None
                        Data to transform target data onto, either the key of a variable
            in the input dataset or an ``xr.DataArray``, by default None.
        grid_positions : dict[str, str] | None
            Mapping of dimension positions, by default None. If ``None`` then an
            attempt is made to derive this argument.
        periodic : bool
            Whether the grid is periodic, by default False.
        extra_init_options : dict[str, Any] | None
            Extra options passed to the ``xgcm.Grid`` constructor, by default
            None.
        options : dict[str, Any] | None
            Extra options passed to the ``xgcm.Grid.transform`` method.

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

        Regrid data to ``output_grid``:

        >>> output_data = ds.regridder.vertical(
        >>>     "so", output_grid, tool="xgcm", method="linear"
        >>> )

        Create pressure variable:

        >>> ds["pressure"] = (ds["hyam"] * ds["P0"] + ds["hybm"] * ds["PS"]).transpose(
        >>>     **ds["T"].dims
        >>> )

        Regrid data to ``output_grid`` in pressure space:

        >>> output_data = ds.regridder.vertical(
        >>>     "so", output_grid, tool="xgcm", method="linear", target_data="pressure"
        >>> )

        Passing additional arguments to ``xgcm.Grid`` and ``xgcm.Grid.transform``:

        >>> regridder = xgcm.XGCMRegridder(
        >>>     ds,
        >>>     output_grid,
        >>>     method="linear",
        >>>     extra_init_options={"boundary": "fill", "fill_value": 1e27},
        >>>     mask_edges=True
        >>> )
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
        """See documentation in :py:func:`xcdat.regridder.xgcm.XGCMRegridder`"""
        try:
            output_coord_z = get_dim_coords(self._output_grid, "Z")
        except KeyError as e:
            raise RuntimeError(
                "Could not determine 'Z' coordinate in output dataset"
            ) from e

        if self._grid_positions is None:
            grid_coords = self._get_grid_positions()
        else:
            # correctly format argument
            grid_coords = {"Z": self._grid_positions}

        grid = Grid(ds, coords=grid_coords, **self._extra_init_options)

        target_data: str | xr.DataArray | None = None

        try:
            target_data = ds[self._target_data]
        except ValueError:
            target_data = self._target_data
        except KeyError as e:
            if self._target_data is not None and isinstance(self._target_data, str):
                raise RuntimeError(
                    f"Could not find target variable {self._target_data!r} in dataset"
                ) from e

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
            input_coord_z = get_dim_coords(ds[data_var], "Z")

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

    def _get_grid_positions(self) -> dict[str, Any | Hashable]:
        if self._method == "conservative":
            raise RuntimeError(
                "Conservative regridding requires a second point position, pass these "
                "manually."
            )

        try:
            coord_z = get_dim_coords(self._input_grid, "Z")
        except KeyError as e:
            raise RuntimeError("Could not determine `Z` coordinate in dataset.") from e

        if isinstance(coord_z, xr.Dataset):
            coords = ", ".join(sorted(list(coord_z.coords.keys())))  # type: ignore[arg-type]

            raise RuntimeError(
                "Could not determine the `Z` coordinate in the input grid. "
                f"Found multiple axes ({coords}), ensure there is only a "
                "single `Z` axis in the input grid.",
                list(coord_z.coords.keys()),
            )

        try:
            bounds_z = self._input_grid.bounds.get_bounds("Z")
        except KeyError as e:
            raise RuntimeError("Could not determine `Z` bounds in dataset.") from e

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
