"""Bounds module for functions related to coordinate bounds."""
import collections
import warnings
from typing import Dict, List, Optional, Union

import cf_xarray as cfxr  # noqa: F401
import cftime
import numpy as np
import pandas as pd
import xarray as xr

from xcdat.axis import CF_ATTR_MAP, CFAxisKey, get_dim_coords
from xcdat.logger import setup_custom_logger
from xcdat.spatial import _get_data_var

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

    def add_missing_bounds(self, axes: List[CFAxisKey] = ["X", "Y"]) -> xr.Dataset:
        """Adds missing coordinate bounds for supported axes in the Dataset.

        This function loops through the Dataset's axes and attempts to adds
        bounds to its coordinates if they don't exist. The coordinates must meet
        the following criteria in order to add bounds:

          1. The axis for the coordinates are "X", "Y", "T", or "Z"
          2. Coordinates are a single dimension, not multidimensional
          3. Coordinates are a length > 1 (not singleton)
          4. Bounds must not already exist.
             * Determined by attempting to map the coordinate variable's
             "bounds" attr (if set) to the bounds data variable of the same key.

        Parameters
        ----------
        axes : List[str], optional
            List of CF axes that function should operate on, default ["X", "Y"].

        Returns
        -------
        xr.Dataset
        """
        ds = self._dataset.copy()

        for axis in axes:
            try:
                coords = get_dim_coords(ds, axis)
            except KeyError:
                continue

            for coord in coords.coords.values():
                try:
                    self.get_bounds(axis, str(coord.name))
                    continue
                except KeyError:
                    pass

                try:
                    bounds = self._create_bounds(axis, coord)
                    ds[bounds.name] = bounds
                    ds[coord.name].attrs["bounds"] = bounds.name
                except ValueError:
                    continue

        return ds

    def get_bounds(
        self, axis: CFAxisKey, var_key: Optional[str] = None
    ) -> Union[xr.Dataset, xr.DataArray]:
        """Gets coordinate bounds.

        Parameters
        ----------
        axis : CFAxisKey
            The CF axis key ("X", "Y", "T", "Z").
        var_key: Optional[str]
            The key of the coordinate or data variable to get axis bounds for.
            This parameter is useful if you only want the single bounds
            DataArray related to the axis on the variable (e.g., "tas" has
            a "lat" dimension and you want "lat_bnds").

        Returns
        -------
        Union[xr.Dataset, xr.DataArray]
            A Dataset of N bounds variables, or a single bounds variable
            DataArray.

        Raises
        ------
        ValueError
            If an incorrect ``axis`` argument is passed.

        KeyError:
            If bounds were not found for the specific ``axis``.
        """
        self._validate_axis_arg(axis)

        if var_key is None:
            # Get all bounds keys in the Dataset for this axis.
            bounds_keys = self._get_bounds_keys(axis)
        else:
            # Get the obj in the Dataset using the key.
            obj = _get_data_var(self._dataset, key=var_key)

            # Check if the object is a data variable or a coordinate variable.
            # If it is a data variable, derive the axis coordinate variable.
            if obj.name in list(self._dataset.data_vars):
                coord = get_dim_coords(obj, axis)
            elif obj.name in list(self._dataset.coords):
                coord = obj

            try:
                bounds_keys = [coord.attrs["bounds"]]
            except KeyError:
                bounds_keys = []

        if len(bounds_keys) == 0:
            raise KeyError(
                f"No bounds data variables were found for the '{axis}' axis. Make sure "
                "the dataset has bound data vars and their names match the 'bounds' "
                "attributes found on their related time coordinate variables. "
                "Alternatively, you can add bounds with `xcdat.add_missing_bounds` "
                "or `xcdat.add_bounds`."
            )

        bounds: Union[xr.Dataset, xr.DataArray] = self._dataset[
            bounds_keys if len(bounds_keys) > 1 else bounds_keys[0]
        ].copy()

        return bounds

    def add_bounds(self, axis: CFAxisKey) -> xr.Dataset:
        """Add bounds for an axis using its coordinate points.

        This method loops over the axis's coordinate variables and attempts to
        add bounds for each of them if they don't exist. The coordinates must
        meet the following criteria in order to add bounds:

          1. The axis for the coordinates are "X", "Y", "T", or "Z"
          2. Coordinates are single dimensional, not multidimensional
          3. Coordinates are a length > 1 (not singleton)
          4. Bounds must not already exist.
             * Determined by attempting to map the coordinate variable's
             "bounds" attr (if set) to the bounds data variable of the same key.

        Parameters
        ----------
        axis : CFAxisKey
            The CF axis key ("X", "Y", "T", or "Z").

        Returns
        -------
        xr.Dataset
            The dataset with bounds added.

        Raises
        ------
        ValueError
            If bounds already exist. They must be dropped first.

        """
        ds = self._dataset.copy()
        self._validate_axis_arg(axis)

        coord_vars: Union[xr.DataArray, xr.Dataset] = get_dim_coords(
            self._dataset, axis
        )

        for coord in coord_vars.coords.values():
            # Check if the coord var has a "bounds" attr and the bounds actually
            # exist in the Dataset. If it does not, then add the bounds.
            try:
                bounds_key = ds[coord.name].attrs["bounds"]
                ds[bounds_key]

                continue
            except KeyError:
                bounds = self._create_bounds(axis, coord)

                ds[bounds.name] = bounds
                ds[coord.name].attrs["bounds"] = bounds.name

        return ds

    def _get_bounds_keys(self, axis: CFAxisKey) -> List[str]:
        """Get bounds keys for an axis's coordinate variables in the dataset.

        This function attempts to map bounds to an axis using ``cf_xarray``
        and its interpretation of the CF "bounds" attribute.

        Parameters
        ----------
        axis : CFAxisKey
            The CF axis key ("X", "Y", "T", or "Z").

        Returns
        -------
        List[str]
            The axis bounds key(s).
        """
        cf_method = self._dataset.cf.bounds
        cf_attrs = CF_ATTR_MAP[axis]

        keys: List[str] = []

        try:
            keys = keys + cf_method[cf_attrs["axis"]]
        except KeyError:
            pass

        try:
            keys = cf_method[cf_attrs["coordinate"]]
        except KeyError:
            pass

        return list(set(keys))

    def _create_bounds(self, axis: CFAxisKey, coord_var: xr.DataArray) -> xr.DataArray:
        """Creates bounds for an axis using its coordinate points.

        Parameters
        ----------
        axis: CFAxisKey
            The CF axis key ("X", "Y", "T" ,"Z").
        coord_var : xr.DataArray
            The coordinate variable for the axis.

        Returns
        -------
        xr.DataArray
            The axis coordinate bounds.

        Raises
        ------
        ValueError
            If coords dimensions does not equal 1.

        Notes
        -----
        Based on [1]_ ``iris.coords._guess_bounds`` and [2]_ ``cf_xarray.accessor.add_bounds``

        References
        ----------

        .. [1] https://scitools-iris.readthedocs.io/en/stable/generated/api/iris/coords.html#iris.coords.AuxCoord.guess_bounds

        .. [2] https://cf-xarray.readthedocs.io/en/latest/generated/xarray.Dataset.cf.add_bounds.html#
        """
        is_singleton = coord_var.size <= 1
        if is_singleton:
            raise ValueError(
                f"Cannot generate bounds for coordinate variable '{coord_var.name}'"
                " which has a length <= 1 (singleton)."
            )

        # Retrieve coordinate dimension to calculate the diffs between points.
        dim = coord_var.dims[0]
        diffs = coord_var.diff(dim).values

        # Add beginning and end points to account for lower and upper bounds.
        # np.array of string values with `dtype="timedelta64[ns]"`
        diffs = np.insert(diffs, 0, diffs[0])
        diffs = np.append(diffs, diffs[-1])

        # `cftime` objects only support arithmetic using `timedelta` objects, so
        # the values of  `diffs` must be casted from `dtype="timedelta64[ns]"`
        # to `timedelta` objects.
        if axis == "T" and issubclass(type(coord_var.values[0]), cftime.datetime):
            diffs = pd.to_timedelta(diffs)

        # width parameter: determines bounds location relative to midpoints
        width = 0.5

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
        data = np.array([lower_bounds, upper_bounds]).transpose()

        # Clip latitude bounds at (-90, 90)
        if axis == "Y":
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
                np.clip(data, -90, 90, out=data)

        # Create the bounds data variable and add it to the Dataset.
        bounds = xr.DataArray(
            name=f"{coord_var.name}_bnds",
            data=data,
            coords={coord_var.name: coord_var},
            dims=[*coord_var.dims, "bnds"],
            attrs={"xcdat_bounds": "True"},
        )

        return bounds

    def _validate_axis_arg(self, axis: CFAxisKey):
        cf_axis_keys = CF_ATTR_MAP.keys()

        if axis not in cf_axis_keys:
            keys = ", ".join(f"'{key}'" for key in cf_axis_keys)
            raise ValueError(
                f"Incorrect 'axis' argument value. Supported values include {keys}."
            )

        get_dim_coords(self._dataset, axis)
