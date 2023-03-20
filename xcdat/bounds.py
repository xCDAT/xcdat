"""Bounds module for functions related to coordinate bounds."""
import collections
import datetime
from typing import Dict, List, Literal, Optional, Union

import cf_xarray as cfxr  # noqa: F401
import cftime
import numpy as np
import pandas as pd
import xarray as xr
from xarray.coding.cftime_offsets import get_date_type

from xcdat.axis import CF_ATTR_MAP, CFAxisKey, get_dim_coords
from xcdat.dataset import _get_data_var
from xcdat.logger import setup_custom_logger
from xcdat.temporal import _infer_freq

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

    >>> ds = ds.bounds.add_missing_bounds(axes=["X", "Y", "T"])

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

    def add_missing_bounds(  # noqa: C901
        self,
        axes: List[CFAxisKey] = ["X", "Y"],
        time_freq: Optional[Literal["year", "month", "day", "hour"]] = None,
        daily_time_freq: Optional[Literal[1, 2, 3, 4, 6, 8, 12, 24]] = None,
    ) -> xr.Dataset:
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
          5. Time axes should be composed of `cftime` objects.

        Parameters
        ----------
        axes : List[str], optional
            List of CF axes that function should operate on, default ["X", "Y"].
        time_freq : Optional[Literal["year", "month", "day", "hour"]]
            If ``axis`` includes "T", this parameter specifies the time
            frequency for creating time bounds, by default None (infer the
            frequency).
        daily_time_freq : Literal[1, 2, 3, 4, 6, 8, 12, 24], optional
            If ``time_freq=="hour"``, this parameter sets the number of
            timepoints per day for bounds, by default None. If greater than 1,
            sub-daily bounds are created.

            * ``daily_time_freq=None`` infers the daily time frequency from the
              time coordinates.
            * ``daily_time_freq=1`` is daily
            * ``daily_time_freq=2`` is twice daily
            * ``daily_time_freq=4`` is 6-hourly
            * ``daily_time_freq=8`` is 3-hourly
            * ``daily_time_freq=12`` is 2-hourly
            * ``daily_time_freq=24`` is hourly


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

                if axis in ["X", "Y", "Z"]:
                    try:
                        bounds = self._create_bounds(axis, coord)
                    except ValueError:
                        continue
                elif axis == "T":
                    bounds = self._create_time_bounds(coord, time_freq, daily_time_freq)

                ds[bounds.name] = bounds
                ds[coord.name].attrs["bounds"] = bounds.name

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

    def add_bounds(
        self,
        axis: CFAxisKey,
        time_freq: Optional[Literal["year", "month", "day", "hour"]] = None,
        daily_time_freq: Optional[Literal[1, 2, 3, 4, 6, 8, 12, 24]] = None,
    ) -> xr.Dataset:
        """Add bounds for an axis using its coordinate points.

        This method loops over the axis's coordinate variables and attempts to
        add bounds for each of them if they don't exist. If ``axis`` is "X",
        "Y", or "Z", each coordinate point is the midpoint between their lower
        and upper bounds. If the axis is "T", bounds will be set to the start
        and end of each time step's period using either the inferred or specified
        time frequency (``time_freq`` parameter).

        Parameters
        ----------
        axis : CFAxisKey
            The CF axis key ("X", "Y", "T", or "Z").
        time_freq : Optional[Literal["year", "month", "day", "hour"]]
            If ``axis="T"``, this parameter specifies the time frequency for
            creating time bounds, by default None (infer the frequency).
        daily_time_freq : Literal[1, 2, 3, 4, 6, 8, 12, 24], optional
            If ``time_freq=="hour"``, this parameter sets the number of
            timepoints per day for bounds, by default None. If greater than 1,
            sub-daily bounds are created.

            * ``daily_time_freq=None`` infers the daily time frequency from the
              time coordinates.
            * ``daily_time_freq=1`` is daily
            * ``daily_time_freq=2`` is twice daily
            * ``daily_time_freq=4`` is 6-hourly
            * ``daily_time_freq=8`` is 3-hourly
            * ``daily_time_freq=12`` is 2-hourly
            * ``daily_time_freq=24`` is hourly

        Returns
        -------
        xr.Dataset
            The dataset with bounds added.

        Raises
        ------
        ValueError
            If bounds already exist. They must be dropped first.

        Note
        ----
        To add time bounds for an axis, its coordinates must be the following
        criteria:

          1. The axis for the coordinates are "X", "Y", "T", or "Z"
          2. Coordinates are single dimensional, not multidimensional
          3. Coordinates are a length > 1 (not singleton)
          4. Bounds must not already exist
             * Determined by attempting to map the coordinate variable's
             "bounds" attr (if set) to the bounds data variable of the same key
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
                if axis in ["X", "Y", "Z"]:
                    bounds = self._create_bounds(axis, coord)
                elif axis == "T":
                    bounds = self._create_time_bounds(coord, time_freq, daily_time_freq)

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

        This method uses each coordinate point as the midpoint between its
        lower and upper bound.

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
        Based on [1]_ ``iris.coords._guess_bounds`` and [2]_
        ``cf_xarray.accessor.add_bounds``.

        For temporal coordinates ``_create_bounds`` will attempt to set the
        bounds to the start and end of each time step's period. Time axes are
        expected to be composed of ``cftime`` objects.

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

        # width parameter: determines bounds location relative to midpoints
        width = 0.5

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

    def _create_time_bounds(
        self,
        time: xr.DataArray,
        freq: Optional[Literal["year", "month", "day", "hour"]] = None,
        daily_freq: Optional[Literal[1, 2, 3, 4, 6, 8, 12, 24]] = None,
    ) -> xr.DataArray:
        """Creates time bounds for each timestep of the time coordinate axis.


        Parameters
        ----------
        time : xr.DataArray
            The temporal coordinate variable for the axis.
        freq : Optional[Literal["year", "month", "day", "hour"]]
            The time frequency for creating time bounds, by default None (infer
            the frequency).
        daily_freq : Literal[1, 2, 3, 4, 6, 8, 12, 24], optional
            If ``freq=="hour"``, this parameter sets the number of timepoints
            per day for bounds, by default None. If greater than 1, sub-daily
            bounds are created.

            * ``daily_freq=None`` infers the freq from the time coords (default)
            * ``daily_freq=1`` is daily
            * ``daily_freq=2`` is twice daily
            * ``daily_freq=4`` is 6-hourly
            * ``daily_freq=8`` is 3-hourly
            * ``daily_freq=12`` is 2-hourly
            * ``daily_freq=24`` is hourly

        Returns
        -------
        xr.DataArray
            A DataArray storing bounds for the time axis.

        Raises
        ------
        TypeError
            If time coordinates are not composed of ``cftime.datetime`` object.
        """
        freq = _infer_freq(time) if freq is None else freq  # type: ignore
        timesteps = time.values

        # Determine the object type for creating time bounds based on the
        # object type/dtype of the time coordinates.
        if np.issubdtype(timesteps.dtype, np.datetime64):
            # Cast time values from `np.datetime64` to `pd.Timestamp` (a
            # sub-class of `np.datetime64`) in order to get access to the
            # pandas time/date components which simplifies creating bounds.
            # https://pandas.pydata.org/docs/user_guide/timeseries.html#time-date-components
            timesteps = pd.to_datetime(timesteps)
            obj_type = pd.Timestamp
        elif issubclass(type(timesteps[0]), cftime.datetime):
            # Get the `cftime.datetime` sub-class object type based on the
            # CF calendar type.
            calendar = time.encoding["calendar"]
            obj_type = get_date_type(calendar)
        else:
            raise TypeError(
                f"Bounds cannot be created for '{time.name}' coordinates because it is "
                "not decoded as `cftime.datetime` or `datetime.datetime`. Try decoding "
                f"'{time.name}' first then adding bounds."
            )

        if freq == "year":
            time_bnds = self._create_yearly_time_bounds(timesteps, obj_type)
        elif freq == "month":
            time_bnds = self._create_monthly_time_bounds(timesteps, obj_type)
        elif freq == "day":
            time_bnds = self._create_daily_time_bounds(timesteps, obj_type)
        elif freq == "hour":
            # Determine the daily frequency for generating time  bounds.
            if daily_freq is None:
                diff = time.values[1] - time.values[0]
                hrs = diff.seconds / 3600
                daily_freq = int(24 / hrs)  # type: ignore

            time_bnds = self._create_daily_time_bounds(timesteps, obj_type, freq=daily_freq)  # type: ignore

        # Create the bounds data array
        da_time_bnds = xr.DataArray(
            name=f"{time.name}_bnds",
            data=time_bnds,
            coords={time.name: time},
            dims=[*time.dims, "bnds"],
            attrs={"xcdat_bounds": "True"},
        )

        return da_time_bnds

    def _create_yearly_time_bounds(
        self,
        timesteps: np.ndarray,
        obj_type: Union[cftime.datetime, pd.Timestamp],
    ) -> List[Union[cftime.datetime, pd.Timestamp]]:
        """Creates time bounds for each timestep with the start and end of the year.

        Bounds for each timestep correspond to Jan. 1 00:00:00 of the year of the
        timestep and Jan. 1 00:00:00 of the subsequent year.

        Parameters
        ----------
        timesteps : np.ndarray
            An array of timesteps, represented as either `cftime.datetime` or
            `pd.Timestamp` (casted from `np.datetime64[ns]` to support pandas
            time/date components).
        obj_type : Union[cftime.datetime, pd.Timestamp]
            The object type for time bounds based on the dtype of
            ``time_values``.

        Returns
        -------
        List[cftime.datetime]
            A list of time bound values.
        """
        time_bnds: List[cftime.datetime] = []

        for step in timesteps:
            year = step.year

            l_bnd = obj_type(year, 1, 1, 0, 0)
            u_bnd = obj_type(year + 1, 1, 1, 0, 0)

            time_bnds.append([l_bnd, u_bnd])

        return time_bnds

    def _create_monthly_time_bounds(
        self,
        timesteps: np.ndarray,
        obj_type: Union[cftime.datetime, pd.Timestamp],
        end_of_month: bool = False,
    ) -> List[Union[cftime.datetime, pd.Timestamp]]:
        """Creates time bounds for each timestep with the start and end of the month.

        Bounds for each timestep correspond to 00:00:00 on the first of the month
        and 00:00:00 on the first of the subsequent month.

        Parameters
        ----------
        timesteps : np.ndarray
            An array of timesteps, represented as either `cftime.datetime` or
            `pd.Timestamp` (casted from `np.datetime64[ns]` to support pandas
            time/date components).
        obj_type : Union[cftime.datetime, pd.Timestamp]
            The object type for time bounds based on the dtype of
            ``time_values``.
        end_of_month : bool, optional
            Flag to note that the timepoint is saved at the end of the monthly
            interval (see Note), by default False.

        Returns
        -------
        List[Union[cftime.datetime, pd.Timestamp]]
            A list of time bound values.

        Note
        ----
        Some timepoints are saved at the end of the interval, e.g., Feb. 1 00:00
        for the time interval Jan. 1 00:00 - Feb. 1 00:00. Since this function
        determines the month and year from the time vector, the bounds will be set
        incorrectly if the timepoint is set to the end of the time interval. For
        these cases, set ``end_of_month=True``.
        """
        time_bnds = []

        for step in timesteps:
            # If end of time interval and first day of year then subtract one
            # month so bounds will be calculated correctly.
            # FIXME: ``end_of_month`` needs to be determined internally because
            # we don't provide this flag with `ds.bounds.add_bounds()` so it can
            # never be set to True. Adding this manual flag would make
            # `.add_bounds` less intuitive. After this logic is implemented, we
            # can remove the ``end_of_month `` parameter here.
            if (end_of_month) & (step.day < 2):
                step = self._add_months_to_timestep(step, obj_type, delta=-1)

            year, month = step.year, step.month

            l_bnd = obj_type(year, month, 1, 0, 0)
            u_bnd = self._add_months_to_timestep(l_bnd, obj_type, delta=1)

            time_bnds.append([l_bnd, u_bnd])

        return time_bnds

    def _add_months_to_timestep(
        self,
        timestep: Union[cftime.datetime, pd.Timestamp],
        obj_type: Union[cftime.datetime, pd.Timestamp],
        delta: int,
    ) -> Union[cftime.datetime, pd.Timestamp]:
        """Adds delta month(s) to a timestep.

        The delta value can be positive or negative (for subtraction). Refer to [1]_
        for logic.

        Parameters
        ----------
        timesep : Union[cftime.datime, pd.Timestamp]
            A timestep represented as ``cftime.datetime`` or ``pd.Timestamp``.
        obj_type : Union[cftime.datetime, pd.Timestamp]
                The object type for time bounds based on the dtype of
                ``timestep``.
        delta : int
            Integer months to be added to times (can be positive or negative)

        Returns
        -------
        Union[cftime.datetime, pd.Timestamp]

        References
        ----------
        [1] https://stackoverflow.com/a/4131114
        """
        # Compute the new month and year with the delta month(s).
        month = timestep.month - 1 + delta
        year = timestep.year + month // 12
        month = month % 12 + 1

        # Re-use existing hour/minute/second/microsecond.
        hour = timestep.hour
        minute = timestep.minute
        second = timestep.second
        microsecond = timestep.microsecond

        # If day is greater than days in month use days in month as day.
        day = timestep.day
        dim = obj_type(year, month, 1).daysinmonth
        day = min(day, dim)

        # Create the new timestep.
        new_timestep = obj_type(year, month, day, hour, minute, second, microsecond)

        return new_timestep

    def _create_daily_time_bounds(
        self,
        timesteps: np.ndarray,
        obj_type: Union[cftime.datetime, pd.Timestamp],
        freq: Literal[1, 2, 3, 4, 6, 8, 12, 24] = 1,
    ) -> List[Union[cftime.datetime, pd.Timestamp]]:
        """Creates time bounds for each timestep with the start and end of the day.

        Bounds for each timestep corresponds to 00:00:00 timepoint on the
        current day and 00:00:00 on the subsequent day.

        Parameters
        ----------
        timesteps : np.ndarray
            An array of timesteps, represented as either `cftime.datetime` or
            `pd.Timestamp` (casted from `np.datetime64[ns]` to support pandas
            time/date components).
        obj_type : Union[cftime.datetime, pd.Timestamp]
            The object type for time bounds based on the dtype of
            ``time_values``.
        freq : Literal[1, 2, 3, 4, 6, 8, 12, 24], optional
            Number of timepoints per day, by default 1. If greater than 1, sub-daily
            bounds are created.

            * ``freq=1`` is daily (default)
            * ``freq=2`` is twice daily
            * ``freq=4`` is 6-hourly
            * ``freq=8`` is 3-hourly
            * ``freq=12`` is 2-hourly
            * ``freq=24`` is hourly

        Returns
        -------
        List[Union[cftime.datetime, pd.Timestamp]]
            A list of time bound values.

        Raises
        ------
        ValueError
            If an incorrect ``freq`` argument is passed. Should be 1, 2, 3, 4, 6, 8,
            12, or 24.

        Notes
        -----
        This function is intended to reproduce CDAT's ``setAxisTimeBoundsDaily``
        method [3]_.

        References
        ----------
        .. [3] https://github.com/CDAT/cdutil/blob/master/cdutil/times.py#L1093
        """
        if (freq > 24) | (np.mod(24, freq)):
            raise ValueError(
                "Incorrect `freq` argument. Supported values include 1, 2, 3, 4, 6, 8, 12, "
                "and 24."
            )

        time_bnds = []

        for step in timesteps:
            y, m, d, h = step.year, step.month, step.day, step.hour

            for f in range(freq):
                if f * (24 // freq) <= h < (f + 1) * (24 // freq):
                    l_bnd = obj_type(y, m, d, f * (24 // freq))
                    u_bnd = l_bnd + datetime.timedelta(hours=(24 // freq))

            time_bnds.append([l_bnd, u_bnd])

        return time_bnds

    def _validate_axis_arg(self, axis: CFAxisKey):
        cf_axis_keys = CF_ATTR_MAP.keys()

        if axis not in cf_axis_keys:
            keys = ", ".join(f"'{key}'" for key in cf_axis_keys)
            raise ValueError(
                f"Incorrect 'axis' argument value. Supported values include {keys}."
            )

        get_dim_coords(self._dataset, axis)
