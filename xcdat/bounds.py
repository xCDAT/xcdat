"""Bounds module for functions related to coordinate bounds."""
import collections
import warnings
from typing import Dict, List, Optional

import cf_xarray as cfxr  # noqa: F401
import cftime
import numpy as np
import pandas as pd
import xarray as xr

from xcdat.axis import CF_NAME_MAP, CFAxisName, get_axis_coord
from xcdat.logger import setup_custom_logger

logger = setup_custom_logger(__name__)


@xr.register_dataset_accessor("bounds")
class BoundsAccessor:
    """
    An accessor class that provides bounds attributes and methods on xarray
    Datasets through the ``.bounds`` attribute.

    Examples
    --------

    Import BoundsAccessor class:

    >>> import xcdat  # or from xcdat import bounds

    Use BoundsAccessor class:

    >>> ds = xcdat.open_dataset("/path/to/file")
    >>>
    >>> ds.bounds.<attribute>
    >>> ds.bounds.<method>
    >>> ds.bounds.<property>

    Parameters
    ----------
    dataset : xr.Dataset
        A Dataset object.

    Examples
    ---------
    Import:

    >>> from xcdat import bounds

    Return dictionary of axis and coordinate keys mapped to bounds:

    >>> ds.bounds.map

    Return list of keys for bounds data variables:

    >>> ds.bounds.keys

    Add missing coordinate bounds for supported axes in the Dataset:

    >>> ds = ds.bounds.add_missing_bounds()

    Get coordinate bounds if they exist:

    >>> lat_bounds = ds.bounds.get_bounds("Y")
    >>> lon_bounds = ds.bounds.get_bounds("X")
    >>> time_bounds = ds.bounds.get_bounds("T")

    Add coordinate bounds for a specific axis if they don't exist:

    >>> ds = ds.bounds.add_bounds("Y")
    """

    def __init__(self, dataset: xr.Dataset):
        self._dataset: xr.Dataset = dataset

    @property
    def map(self) -> Dict[str, Optional[xr.DataArray]]:
        """Returns a map of axis and coordinates keys to their bounds.

        The dictionary provides all valid CF compliant keys for axis and
        coordinates. For example, latitude will includes keys for "lat",
        "latitude", and "Y".

        Returns
        -------
        Dict[str, Optional[xr.DataArray]]
            Dictionary mapping axis and coordinate keys to their bounds.
        """
        ds = self._dataset

        bounds: Dict[str, Optional[xr.DataArray]] = {}
        for axis, bounds_keys in ds.cf.bounds.items():
            bound = ds.get(bounds_keys[0], None)
            bounds[axis] = bound

        return collections.OrderedDict(sorted(bounds.items()))

    @property
    def keys(self) -> List[str]:
        """Returns a list of keys for the bounds data variables in the Dataset.

        Returns
        -------
        List[str]
            A list of sorted bounds data variable keys.
        """
        return sorted(
            list(
                {
                    key
                    for bound_keys in self._dataset.cf.bounds.values()
                    for key in bound_keys
                }
            )
        )

    def add_missing_bounds(self, width: float = 0.5) -> xr.Dataset:
        """Adds missing coordinate bounds for supported axes in the Dataset.

        This function loops through the Dataset's axes and adds coordinate
        bounds for an axis that doesn't have any.

        Parameters
        ----------
        width : float, optional
            Width of the bounds relative to the position of the nearest points,
            by default 0.5.

        Returns
        -------
        xr.Dataset
        """
        axes = CF_NAME_MAP.keys()

        for axis in axes:
            coord_var = None

            try:
                coord_var = get_axis_coord(self._dataset, axis)
            except KeyError:
                pass

            if coord_var is not None:
                try:
                    self.get_bounds(axis)
                except KeyError:
                    self._dataset = self.add_bounds(axis, width)
        return self._dataset

    def get_bounds(self, axis: CFAxisName) -> xr.DataArray:
        """Get bounds for axis coordinates if both exist.

        Parameters
        ----------
        axis : CFAxisName
            The CF-compliant axis name ("X", "Y", "T").

        Returns
        -------
        xr.DataArray
            The coordinate bounds.

        Raises
        ------
        ValueError
            If an incorrect ``axis`` argument is passed.

        KeyError
            If the coordinate variable was not found for the ``axis``.

        KeyError
            If the coordinate bounds were not found for the ``axis``.
        """
        self._validate_axis_arg(axis)
        coord_var = get_axis_coord(self._dataset, axis)

        try:
            bounds_key = coord_var.attrs["bounds"]
        except KeyError:
            raise KeyError(
                f"The coordinate variable '{coord_var.name}' has no 'bounds' attr. "
                "Set the 'bounds' attr to the name of the bounds data variable."
            )

        try:
            bounds_var = self._dataset[bounds_key].copy()
        except KeyError:
            raise KeyError(
                f"Bounds were not found for the coordinate variable '{coord_var.name}'. "
                "Add bounds with `Dataset.bounds.add_bounds()`."
            )

        return bounds_var

    def add_bounds(self, axis: CFAxisName, width: float = 0.5) -> xr.Dataset:
        """Add bounds for an axis using its coordinate points.

        Parameters
        ----------
        axis : CFAxisName
            The CF-compliant axis name ("X", "Y", "T").
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
        self._validate_axis_arg(axis)

        try:
            self.get_bounds(axis)
            raise ValueError(
                f"{axis} bounds already exist. Drop them first to add new bounds."
            )
        except KeyError:
            dataset = self._add_bounds(axis, width)

        return dataset

    def _add_bounds(self, axis: CFAxisName, width: float = 0.5) -> xr.Dataset:
        """Add bounds for an axis using its coordinate points.

        Parameters
        ----------
        axis : CFAxisName
            The CF-compliant axis name ("X", "Y", "T").
        width : float, optional
            Width of the bounds relative to the position of the nearest points,
            by default 0.5.

        Returns
        -------
        xr.Dataset
            The dataset with new coordinate bounds for an axis.

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
        # Add coordinate bounds to the dataset
        ds = self._dataset.copy()
        coord_var: xr.DataArray = get_axis_coord(ds, axis)

        # Validate coordinate shape and dimensions
        if coord_var.ndim != 1:
            raise ValueError("Cannot generate bounds for multidimensional coordinates.")
        if coord_var.shape[0] <= 1:
            raise ValueError("Cannot generate bounds for a coordinate of length <= 1.")

        # Retrieve coordinate dimension to calculate the diffs between points.
        dim = coord_var.dims[0]
        diffs = coord_var.diff(dim).values

        # Add beginning and end points to account for lower and upper bounds.
        # np.array of string values with `dtype="timedelta64[ns]"`
        diffs = np.insert(diffs, 0, diffs[0])
        diffs = np.append(diffs, diffs[-1])

        # In xarray and xCDAT, time coordinates with non-CF compliant calendars
        # (360-day, noleap) and/or units ("months", "years") are decoded using
        # `cftime` objects instead of `datetime` objects. `cftime` objects only
        # support arithmetic using `timedelta` objects, so the values of `diffs`
        # must be casted from `dtype="timedelta64[ns]"` to `timedelta`.
        if coord_var.name in ("T", "time") and issubclass(
            type(coord_var.values[0]), cftime.datetime
        ):
            diffs = pd.to_timedelta(diffs)

        # FIXME: These lines produces the warning: `PerformanceWarning:
        # Adding/subtracting object-dtype array to TimedeltaArray not
        # vectorized` after converting diffs to `timedelta`. I (Tom) was not
        # able to find an alternative, vectorized solution at the time of this
        # implementation.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
            # Get lower and upper bounds by using the width relative to nearest point.
            lower_bounds = coord_var - diffs[:-1] * width
            upper_bounds = coord_var + diffs[1:] * (1 - width)

        # Transpose both bound arrays into a 2D array.
        bounds = np.array([lower_bounds, upper_bounds]).transpose()

        # Clip latitude bounds at (-90, 90)
        if coord_var.name in ("lat", "latitude", "grid_latitude"):
            units = coord_var.attrs.get("units")

            if units is None:
                coord_var.attrs["units"] = "degrees_north"
                logger.warning(
                    f"The '{coord_var.name}' coordinate variable is missing "
                    "a 'units' attribute. Assuming 'units' is 'degrees_north'."
                )
            elif "degree" not in units.lower():
                raise ValueError(
                    f"The {coord_var.name} coord variable has a 'units' attribute that "
                    "is not in degrees."
                )

            if (coord_var >= -90).all() and (coord_var <= 90).all():
                np.clip(bounds, -90, 90, out=bounds)

        # Create the bounds data variable and add it to the Dataset.
        bounds_var = xr.DataArray(
            name=f"{coord_var.name}_bnds",
            data=bounds,
            coords={coord_var.name: coord_var},
            dims=[coord_var.name, "bnds"],
            attrs={"xcdat_bounds": "True"},
        )
        ds[bounds_var.name] = bounds_var

        # Update the attributes of the coordinate variable.
        coord_var.attrs["bounds"] = bounds_var.name
        ds[coord_var.name] = coord_var

        return ds

    def _validate_axis_arg(self, axis: CFAxisName):
        cf_axis_names = CF_NAME_MAP.keys()

        if axis not in cf_axis_names:
            keys = ", ".join(f"'{key}'" for key in cf_axis_names)
            raise ValueError(
                f"Incorrect 'axis' argument value. Supported values include {keys}."
            )

        get_axis_coord(self._dataset, axis)
