import xarray as xr
from xgcm import Grid

from xcdat.regridder.base import BaseRegridder, preserve_bounds

VALID_METHODS = ["linear", "conservative", "log"]


class XGCMRegridder(BaseRegridder):
    def __init__(self, input_grid: xr.Dataset, output_grid: xr.Dataset, method: str, theta: str, **options):
        super().__init__(input_grid, output_grid)

        if method not in VALID_METHODS:
            raise ValueError(f"{method!r} is invalid, possible choices: {', '.join(VALID_METHODS)}")

        self._method = method
        self._theta = theta
        self._extra_options = options

    def horizontal(self, data_var: str, ds: xr.Dataset) -> xr.Dataset:
        """Placeholder for base class."""
        raise NotImplementedError()

    def vertical(self, data_var: str, ds: xr.Dataset) -> xr.Dataset:
        try:
            coord_z = ds.cf["Z"]
        except KeyError:
            raise RuntimeError("Could not determine \"Z\" coordinate in dataset")

        # TODO: how to handle conservative method when we need two points e.g. {"lev": {"center": "z", "outer": "zc"}}
        grid = Grid(ds, coords={coord_z.name: {"center": coord_z.name}}, periodic=False)

        # assumes new verical coordinate has been calculated and stored as `pressure`
        # TODO: auto calculate pressure reference http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#parametric-v-coord
        #       cf_xarray only supports two ocean s-coordinate and ocean_sigma_coordinate
        output_da = grid.transform(ds[data_var], coord_z.name, self._output_grid[coord_z.name], target_data=ds[self._theta], method=self._method)

        output_da = output_da.transpose(*ds[data_var].dims)

        output_ds = xr.Dataset({data_var: output_da}, attrs=ds.attrs)

        output_ds = preserve_bounds(ds, self._output_grid, output_ds)

        output_ds = output_ds.bounds.add_missing_bounds()

        return output_ds
