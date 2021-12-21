"""Bounds module for functions related to coordinate bounds."""
import collections
from typing import Dict, List, Optional

import cf_xarray as cfxr  # noqa: F401
import numpy as np
import xarray as xr
from typing_extensions import Literal

from xcdat.axis import GENERIC_AXIS_MAP
from xcdat.logger import setup_custom_logger

logger = setup_custom_logger(__name__)

#: Tuple of supported CF compliant axis keys for bounds operations.
BoundsAxis = Literal["lat", "latitude", "Y", "lon", "longitude", "X", "time", "T"]


@xr.register_dataset_accessor("bounds")
class BoundsAccessor:
    """A class to represent the BoundsAccessor.

    Examples
    ---------
    Import:

    >>> from xcdat import bounds

    Return dictionary of coordinate keys mapped to bounds DataArrays:

    >>> ds.bounds.bounds

    Add missing coordinate bounds for supported axes in the Dataset:

    >>> ds = ds.bounds.add_missing_bounds()

    Get coordinate bounds if they exist:

    >>> lat_bounds = ds.bounds.get_bounds("lat") # or pass "latitude"
    >>> lon_bounds = ds.bounds.get_bounds("lon") # or pass "longitude"
    >>> time_bounds = ds.bounds.get_bounds("time")

    Add coordinate bounds for a specific axis if they don't exist:

    >>> ds = ds.bounds.add_bounds("lat")
    """

    def __init__(self, dataset: xr.Dataset):
        self._dataset: xr.Dataset = dataset

    @property
    def bounds(self) -> Dict[str, Optional[xr.DataArray]]:
        """Returns a mapping of axis and coordinates keys to their bounds.

        The dictionary provides all valid CF compliant keys for axis and
        coordinates. For example, latitude will includes keys for "lat",
        "latitude", and "Y".

        Returns
        -------
        Dict[str, Optional[xr.DataArray]]
            Dictionary mapping coordinate keys to their bounds.
        """
        ds = self._dataset

        bounds: Dict[str, Optional[xr.DataArray]] = {}
        for axis, bounds_name in ds.cf.bounds.items():
            bound = ds.get(bounds_name[0], None)
            bounds[axis] = bound

        return collections.OrderedDict(sorted(bounds.items()))

    @property
    def names(self) -> List[str]:
        """Returns a list of names for the bounds data variables in the Dataset.

        Returns
        -------
        List[str]
            A list of sorted dounds data variable names.
        """
        return sorted(
            list(
                {
                    name
                    for bound_names in self._dataset.cf.bounds.values()
                    for name in bound_names
                }
            )
        )

    def add_missing_bounds(self) -> xr.Dataset:
        """Adds missing coordinate bounds for supported axes in the Dataset.

        Returns
        -------
        xr.Dataset
        """
        axes = [
            axis for axis in [*self._dataset.coords] if axis in GENERIC_AXIS_MAP.keys()
        ]

        for axis in axes:
            try:
                self.get_bounds(axis)
            except KeyError:
                try:
                    self._dataset = self.add_bounds(axis)
                except ValueError as err:
                    logger.debug(f"{err}")

        return self._dataset

    def get_bounds(self, axis: BoundsAxis) -> xr.DataArray:
        """Get bounds for axis coordinates.

        Parameters
        ----------
        axis : BoundsAxis
            The axis key.

        Returns
        -------
        xr.DataArray
            The coordinate bounds.

        Raises
        ------
        ValueError
            If an incorrect ``axis`` argument is passed.

        ValueError
            If bounds were not found. They must be added.
        """
        if axis not in GENERIC_AXIS_MAP.keys():
            raise ValueError(
                "Incorrect `axis` argument. Supported axes include: "
                f"include: {', '.join(GENERIC_AXIS_MAP.keys())}."
            )

        try:
            axis = GENERIC_AXIS_MAP[axis]
            bounds = self._dataset.cf.get_bounds(axis)
        except KeyError:
            raise KeyError(f"{axis} bounds were not found, they must be added.")

        return bounds

    def add_bounds(self, axis: BoundsAxis, width: float = 0.5) -> xr.Dataset:
        """Add bounds for axis coordinates using coordinate data points.

        If bounds already exist, they must be dropped first.

        Parameters
        ----------
        axis : BoundsAxis
            The axis key.
        width : float, optional
            Width of the bounds relative to the position of the nearest points,
            by default 0.5.

        Returns
        -------
        xr.Dataset
            The dataset with bounds added.

        Raises
        ------
        ValueError
            If bounds already exist. They must be dropped first.
        """
        try:
            self.get_bounds(axis)
            raise ValueError(
                f"{axis} bounds already exist. Drop them first to add new bounds."
            )
        except KeyError:
            dataset = self._add_bounds(axis, width)

        return dataset

    def _add_bounds(self, axis: BoundsAxis, width: float = 0.5) -> xr.Dataset:
        """Add bounds for axis coordinates using coordinate data points.

        Parameters
        ----------
        axis : BoundsAxis
            The axis key.
        width : float, optional
            Width of the bounds relative to the position of the nearest points,
            by default 0.5.

        Returns
        -------
        xr.Dataset
            The dataset with bounds added.

        Raises
        ------
        ValueError
            If coords dimensions does not equal 1.
        ValueError
            If coords are length of <=1.

        Notes
        -----
        Based on [1]_ ``iris.coords._guess_bounds`` and [2]_ ``cf_xarray.accessor.add_bounds``

        References
        ----------

        .. [1] https://scitools-iris.readthedocs.io/en/stable/generated/api/iris/coords.html#iris.coords.AuxCoord.guess_bounds

        .. [2] https://cf-xarray.readthedocs.io/en/latest/generated/xarray.Dataset.cf.add_bounds.html#
        """
        da_coord: xr.DataArray = self._get_coords(axis)

        # Validate coordinate shape and dimensions
        if da_coord.ndim != 1:
            raise ValueError("Cannot generate bounds for multidimensional coordinates.")
        if da_coord.shape[0] <= 1:
            raise ValueError("Cannot generate bounds for a coordinate of length <= 1.")

        # Retrieve coordinate dimension to calculate the diffs between points.
        dim = da_coord.dims[0]
        diffs = da_coord.diff(dim)

        # Add beginning and end points to account for lower and upper bounds.
        diffs = np.insert(diffs, 0, diffs[0])
        diffs = np.append(diffs, diffs[-1])

        # Get lower and upper bounds by using the width relative to nearest point.
        # Transpose both bound arrays into a 2D array.
        lower_bounds = da_coord - diffs[:-1] * width
        upper_bounds = da_coord + diffs[1:] * (1 - width)
        bounds = np.array([lower_bounds, upper_bounds]).transpose()

        # Clip latitude bounds at (-90, 90)
        if (
            da_coord.name in ("lat", "latitude", "grid_latitude")
            and "degree" in da_coord.attrs["units"]
        ):
            if (da_coord >= -90).all() and (da_coord <= 90).all():
                np.clip(bounds, -90, 90, out=bounds)

        # Add coordinate bounds to the dataset
        dataset = self._dataset.copy()
        var_name = f"{axis}_bnds"
        dataset[var_name] = xr.DataArray(
            name=var_name,
            data=bounds,
            coords={axis: da_coord},
            dims=[axis, "bnds"],
            attrs={"is_generated": "True"},
        )
        dataset[da_coord.name].attrs["bounds"] = var_name

        return dataset

    def _get_coords(self, axis: BoundsAxis) -> xr.DataArray:
        """Get the matching coordinates for an axis in the dataset.

        Parameters
        ----------
        axis : BoundsAxis
            The axis key.

        Returns
        -------
        xr.DataArray
            Matching coordinates in the Dataset.

        Raises
        ------
        TypeError
            If no matching coordinates are found in the Dataset.
        """
        try:
            matching_coord = self._dataset.cf[axis]
        except KeyError:
            raise KeyError(f"No matching coordinates for axis: {axis}")

        return matching_coord
