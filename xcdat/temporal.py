"""Module containing temporal functions."""
from itertools import chain
from typing import Dict, List, Literal, Optional, TypedDict, Union, get_args

import cf_xarray  # noqa: F401
import cftime
import numpy as np
import pandas as pd
import xarray as xr
from xarray.core.groupby import DataArrayGroupBy

from xcdat import bounds  # noqa: F401
from xcdat.dataset import get_data_var
from xcdat.logger import setup_custom_logger
from xcdat.utils import str_to_bool

logger = setup_custom_logger(__name__)

# Type alias for supported time averaging modes.
Mode = Literal["time_series", "climatology", "departures"]
MODES = get_args(Mode)

# Type alias for supported grouping frequencies.
Frequency = Literal["hour", "day", "month", "season", "year"]
#: Tuple of supported grouping frequencies.
FREQUENCIES = get_args(Frequency)

# Configuration specific to the "season" frequency.
SeasonConfig = TypedDict(
    "SeasonConfig",
    {
        "dec_mode": Literal["DJF", "JFD"],
        "drop_incomplete_djf": bool,
        "custom_seasons": Optional[List[str]],
    },
    total=False,
)

SEASON_CONFIG_KEYS = ["dec_mode", "drop_incomplete_djf", "custom_seasons"]

# Type alias representing xarray datetime accessor components.
# https://xarray.pydata.org/en/stable/user-guide/time-series.html#datetime-components
DateTimeComponent = Literal["hour", "day", "month", "season", "year"]

# A dictionary mapping temporal averaging mode and frequency to the xarray
# datetime components used for grouping. Xarray datetime components are
# extracted from the time coordinates of a data variable for grouping. The
# "season" frequency involves additional processing that requires the "year"
# and/or" month" components. These components are removed before grouping.
DATETIME_COMPONENTS = {
    "time_series": {
        "year": ("year",),
        "season": ("year", "season", "month"),  # becomes ("year", "season")
        "month": ("year", "month"),
        "day": ("year", "month", "day"),
        "hour": ("year", "month", "day", "hour"),
    },
    "climatology": {
        "season": ("year", "season", "month"),  # becomes ("season")
        "month": ("month",),
        "day": ("month", "day"),
    },
    "departures": {
        "season": ("year", "season", "month"),  # becomes ("season")
        "month": ("month",),
        "day": ("month", "day"),
    },
}

#: A dictionary mapping month integers to their equivalent 3-letter string.
MONTH_INT_TO_STR: Dict[int, str] = {
    1: "Jan",
    2: "Feb",
    3: "Mar",
    4: "Apr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sep",
    10: "Oct",
    11: "Nov",
    12: "Dec",
}

# A dictionary mapping the season to the middle month. This dictionary is used
# during the creation of datetime objects, which don't support season values.
SEASON_TO_MONTH: Dict[str, int] = {"DJF": 1, "MAM": 4, "JJA": 7, "SON": 10}


@xr.register_dataset_accessor("temporal")
class TemporalAccessor:
    def __init__(self, dataset: xr.Dataset):
        try:
            dataset.cf["T"]
        except KeyError:
            raise KeyError(
                "This dataset does not have a time dimension, which is required for "
                "using the methods in the TemporalAccessor class."
            )

        self._dataset: xr.Dataset = dataset

        # The weights for time coordinates, which are based on a chosen frequency.
        self._weights: Optional[xr.DataArray] = None

    def average(
        self,
        data_var: str,
        freq: Frequency,
        weighted: bool = True,
        center_times: bool = False,
        season_config: SeasonConfig = {
            "dec_mode": "DJF",
            "drop_incomplete_djf": False,
            "custom_seasons": None,
        },
    ):
        """Calculates the time series averages for a data variable.

        Parameters
        ----------
        data_var: str
            The key of the data variable for calculating averages.
        freq : Frequency
            The time frequency to group by.

            * "year": groups by year for yearly averages.
            * "season": groups by (year, season) for seasonal averages.
            * "month": groups by (year, month) for monthly averages.
            * "day": groups by (year, month, day) for daily averages.
            * "hour": groups by (year, month, day, hour) for hourly averages.

        weighted : bool, optional
            Calculate averages using weights, by default True.

            To calculate the weights for the time dimension, first the length of
            time for each coordinate point is calculated using the difference of
            its upper and lower bounds. The time lengths are grouped, then each
            time length is divided by the total sum of the time lengths to get
            the weights.
        center_times: bool, optional
            If True, center time coordinates using the midpoint between its
            upper and lower bounds. Otherwise, use the provided time coordinates,
            by default False.
        season_config: SeasonConfig, optional
            A dictionary for "season" frequency configurations. If configs for
            predefined seasons are passed, configs for custom seasons are
            ignored and vice versa.

            Configs for predefined seasons:

            * "dec_mode" (Literal["DJF", "JFD"], by default "DJF")
                The mode for the season that includes December.

                * "DJF": season includes the previous year December.
                * "JFD": season includes the same year December. Xarray
                    incorrectly labels the season with December as "DJF" when it
                    should be "JFD". Refer to [1]_ for more information on this
                    xarray behavior.

            * "drop_incomplete_djf" (bool, by default False)
                If the "dec_mode" is "DJF", this flag drops (True) or keeps
                (False) time coordinates that fall under incomplete DJF seasons
                Incomplete DJF seasons include the start year Jan/Feb and the
                end year Dec.

            Configs for custom seasons:

            * "custom_seasons" ([List[List[str]]], by default None)
                List of sublists containing month strings, with each sublist
                representing a custom season.

                * Month strings must be in the three letter format (e.g., 'Jan')
                * Each month must be included once in a custom season
                * Order of the months in each custom season does not matter
                * Custom seasons can vary in length

                >>> # Example of custom seasons in a three month format:
                >>> custom_seasons = [
                >>>     ["Jan", "Feb", "Mar"],  # "JanFebMar"
                >>>     ["Apr", "May", "Jun"],  # "AprMayJun"
                >>>     ["Jul", "Aug", "Sep"],  # "JunJulAug"
                >>>     ["Oct", "Nov", "Dec"],  # "OctNovDec"
                >>> ]

        Returns
        -------
        xr.Dataset
            Dataset containing the averaged data variable.

        References
        ----------
        .. [1] https://github.com/pydata/xarray/issues/810

        Examples
        --------
        Import TemporalAccessor class:

        >>> import xcdat

        Call ``average()`` method:

        >>> ds.temporal.average(...)

        Get a data variable's seasonal averages:

        >>> ds_season = ds.temporal.average(
        >>>     "ts",
        >>>     "season",
        >>>     season_config={
        >>>         "dec_mode": "DJF",
        >>>         "drop_incomplete_season": True
        >>>     }
        >>> )
        >>> ds_season.ts
        >>>
        >>> ds_season_with_jfd = ds.temporal.average(
        >>>     "ts",
        >>>     "season",
        >>>     season_config={"dec_mode": "JFD"}
        >>> )
        >>> ds_season_with_jfd.ts

        Get a data variable seasonal averages with custom seasons:

        >>> custom_seasons = [
        >>>     ["Jan", "Feb", "Mar"],  # "JanFebMar"
        >>>     ["Apr", "May", "Jun"],  # "AprMayJun"
        >>>     ["Jul", "Aug", "Sep"],  # "JunJulAug"
        >>>     ["Oct", "Nov", "Dec"],  # "OctNovDec"
        >>> ]
        >>>
        >>> ds_season_custom = ds.temporal.average(
        >>>     "ts",
        >>>     "season",
        >>>     season_config={"custom_seasons": custom_seasons}
        >>> )

        Get the ``average()`` operation attributes:

        >>> ds_season_with_djf.ts.attrs
        {
            'operation': 'temporal_avg',
            'mode': 'average',
            'freq': 'season',
            'groupby': 'season',
            'weighted': 'True',
            'center_times': 'False',
            'dec_mode': 'DJF',
            'drop_incomplete_djf': 'False'
        }
        """
        return self._temporal_avg(
            data_var, "time_series", freq, weighted, center_times, season_config
        )

    def climatology(
        self,
        data_var: str,
        freq: Frequency,
        weighted: bool = True,
        center_times: bool = False,
        season_config: SeasonConfig = {
            "dec_mode": "DJF",
            "drop_incomplete_djf": False,
            "custom_seasons": None,
        },
    ):
        """Calculates the climatology for a data variable.

        Parameters
        ----------
        data_var: str
            The key of the data variable for calculating climatology.
        freq : Frequency
            The time frequency to group by.

            * "season": groups by season for the seasonal cycle climatology.
            * "month": groups by month for the annual cycle climatology.
            * "day": groups by (month, day) for the daily cycle climatology.

        weighted : bool, optional
            Calculate averages using weights, by default True.

            To calculate the weights for the time dimension, first the length of
            time for each coordinate point is calculated using the difference of
            its upper and lower bounds. The time lengths are grouped, then each
            time length is divided by the total sum of the time lengths to get
            the weights.
        center_times: bool, optional
            If True, center time coordinates using the midpoint between its
            upper and lower bounds. Otherwise, use the provided time coordinates,
            by default False.
        season_config: SeasonConfig, optional
            A dictionary for "season" frequency configurations. If configs for
            predefined seasons are passed, configs for custom seasons are
            ignored and vice versa.

            Configs for predefined seasons:

            * "dec_mode" (Literal["DJF", "JFD"], by default "DJF")
               The mode for the season that includes December.

                * "DJF": season includes the previous year December.
                * "JFD": season includes the same year December. Xarray
                  incorrectly labels the season with December as "DJF" when it
                  should be "JFD". Refer to [2]_ for more information on this
                  xarray behavior.

            * "drop_incomplete_djf" (bool, by default False)
                If the "dec_mode" is "DJF", this flag drops (True) or keeps
                (False) time coordinates that fall under incomplete DJF seasons
                Incomplete DJF seasons include the start year Jan/Feb and the
                end year Dec.

            Configs for custom seasons:

            * "custom_seasons" ([List[List[str]]], by default None)
                List of sublists containing month strings, with each sublist
                representing a custom season.

                * Month strings must be in the three letter format (e.g., 'Jan')
                * Each month must be included once in a custom season
                * Order of the months in each custom season does not matter
                * Custom seasons can vary in length

                >>> # Example of custom seasons in a three month format:
                >>> custom_seasons = [
                >>>     ["Jan", "Feb", "Mar"],  # "JanFebMar"
                >>>     ["Apr", "May", "Jun"],  # "AprMayJun"
                >>>     ["Jul", "Aug", "Sep"],  # "JunJulAug"
                >>>     ["Oct", "Nov", "Dec"],  # "OctNovDec"
                >>> ]

        Returns
        -------
        xr.Dataset
            Dataset containing the averaged data variable.

        References
        ----------
        .. [2] https://github.com/pydata/xarray/issues/810

        Examples
        --------
        Import TemporalAccessor class:

        >>> import xcdat

        Call ``climatology()`` method:

        >>> ds.temporal.climatology(...)

        Get a data variable's seasonal climatology:

        >>> ds_season = ds.temporal.climatology(
        >>>     "ts",
        >>>     "season",
        >>>     season_config={
        >>>         "dec_mode": "DJF",
        >>>         "drop_incomplete_season": True
        >>>     }
        >>> )
        >>> ds_season.ts
        >>>
        >>> ds_season = ds.temporal.climatology(
        >>>     "ts",
        >>>     "season",
        >>>     season_config={"dec_mode": "JFD"}
        >>> )
        >>> ds_season.ts

        Get a data variable's seasonal climatology with custom seasons:

        >>> custom_seasons = [
        >>>     ["Jan", "Feb", "Mar"],  # "JanFebMar"
        >>>     ["Apr", "May", "Jun"],  # "AprMayJun"
        >>>     ["Jul", "Aug", "Sep"],  # "JunJulAug"
        >>>     ["Oct", "Nov", "Dec"],  # "OctNovDec"
        >>> ]
        >>>
        >>> ds_season_custom = ds.temporal.climatology(
        >>>     "ts",
        >>>     "season",
        >>>     season_config={"custom_seasons": custom_seasons}
        >>> )

        Get ``climatology()`` operation attributes:

        >>> ds_season_with_djf.ts.attrs
        {
            'operation': 'temporal_avg',
            'mode': 'climatology',
            'freq': 'season',
            'groupby': 'season',
            'weighted': 'True',
            'center_times': 'False',
            'dec_mode': 'DJF',
            'drop_incomplete_djf': 'False'
        }
        """
        return self._temporal_avg(
            data_var, "climatology", freq, weighted, center_times, season_config
        )

    def departures(
        self,
        data_var: str,
        freq: Frequency,
        weighted: bool = True,
        center_times: bool = False,
        season_config: SeasonConfig = {
            "dec_mode": "DJF",
            "drop_incomplete_djf": False,
            "custom_seasons": None,
        },
    ) -> xr.Dataset:
        """Calculates climatological departures ("anomalies").

        In climatology, “anomalies” refer to the difference between the value
        during a given time interval (e.g., the January average surface air
        temperature) and the long-term average value for that time interval
        (e.g., the average surface temperature over the last 30 Januaries).

        This method uses xarray's grouped arithmetic as a shortcut for mapping
        over all unique labels. Grouped arithmetic works by assigning a grouping
        label to each time coordinate of the observation data based on the
        grouping frequency. Afterwards, the corresponding climatology is removed
        from the observation data at each time coordinate based on the matching
        labels.

        xarray's grouped arithmetic operates over each value of the DataArray
        corresponding to each grouping label without changing the size of the
        DataArra. For example,the original monthly time coordinates are
        maintained when calculating seasonal departures on monthly data.
        Visit [3]_ to learn more about how xarray's grouped arithmetic works.

        Parameters
        ----------
        data_var: str
            The key of the data variable to calculate departures for.

        freq : Frequency
            The frequency of time to group by.

            * "season": groups by season for the seasonal cycle departures.
            * "month": groups by month for the annual cycle departures.
            * "day": groups by (month, day) for the daily cycle departures.

        weighted : bool, optional
            Calculate averages using weights, by default True.

            To calculate the weights for the time dimension, first the length of
            time for each coordinate point is calculated using the difference of
            its upper and lower bounds. The time lengths are grouped, then each
            time length is divided by the total sum of the time lengths to get
            the weights.
        center_times: bool, optional
            If True, center time coordinates using the midpoint between its
            upper and lower bounds. Otherwise, use the provided time coordinates,
            by default False.
        season_config: SeasonConfig, optional
            A dictionary for "season" frequency configurations. If configs for
            predefined seasons are passed, configs for custom seasons are
            ignored and vice versa.

            Configs for predefined seasons:

            * "dec_mode" (Literal["DJF", "JFD"], by default "DJF")
               The mode for the season that includes December.

                * "DJF": season includes the previous year December.
                * "JFD": season includes the same year December. Xarray
                  incorrectly labels the season with December as "DJF" when it
                  should be "JFD". Refer to [4]_ for more information on this
                  xarray behavior.

            * "drop_incomplete_djf" (bool, by default False)
                If the "dec_mode" is "DJF", this flag drops (True) or keeps
                (False) time coordinates that fall under incomplete DJF seasons
                Incomplete DJF seasons include the start year Jan/Feb and the
                end year Dec.

            Configs for custom seasons:

            * "custom_seasons" ([List[List[str]]], by default None)
                List of sublists containing month strings, with each sublist
                representing a custom season.

                * Month strings must be in the three letter format (e.g., 'Jan')
                * Each month must be included once in a custom season
                * Order of the months in each custom season does not matter
                * Custom seasons can vary in length

                >>> # Example of custom seasons in a three month format:
                >>> custom_seasons = [
                >>>     ["Jan", "Feb", "Mar"],  # "JanFebMar"
                >>>     ["Apr", "May", "Jun"],  # "AprMayJun"
                >>>     ["Jul", "Aug", "Sep"],  # "JunJulAug"
                >>>     ["Oct", "Nov", "Dec"],  # "OctNovDec"
                >>> ]

        Returns
        -------
        xr.Dataset
            The Dataset containing the departures for a data var's climatology.

        References
        ----------
        .. [3] https://xarray.pydata.org/en/stable/user-guide/groupby.html#grouped-arithmetic
        .. [4] https://github.com/pydata/xarray/issues/810

        Examples
        --------
        Import TemporalAccessor class:

        >>> import xcdat

        Get a data variable's annual cycle departures:

        >>> ds_depart = ds_climo.temporal.departures("ts", "month")

        Get the ``departures()`` operation attributes:

        >>> ds_depart.ts.attrs
        {
            'operation': 'departures',
            'frequency': 'season',
            'groupby': 'season',
            'weighted': 'True',
            'center_times': 'False',
            'dec_mode': 'DJF',
            'drop_incomplete_djf': 'False'
        }
        """
        ds = self._dataset.copy()

        # Calculate the climatology data variable and use its attributes
        # to set the object attributes for calculating departures.
        dv_climo = ds.temporal.climatology(
            data_var, freq, weighted, center_times, season_config
        )[data_var]
        self._set_obj_attrs(
            "departures",
            dv_climo.attrs["freq"],
            str_to_bool(dv_climo.attrs["weighted"]),
            str_to_bool(dv_climo.attrs["center_times"]),
            {
                "dec_mode": dv_climo.attrs.get("dec_mode", "DJF"),
                "drop_incomplete_djf": str_to_bool(
                    dv_climo.attrs.get("drop_incomplete_djf", "False")
                ),
                "custom_seasons": dv_climo.attrs.get("custom_seasons", None),
            },
        )

        # Get the observation data and group it using the time coordinate
        # groups.
        dv_obs = get_data_var(ds, data_var)
        self._time_grouped = self._group_time_coords(ds.cf["T"])
        dv_obs_grouped = self._groupby_freq(dv_obs)

        # Rename the climatology data var's time dimension to align with the
        # grouped observation data var's time dimension so that xarray's
        # grouped subtraction arithmetic works. Otherwise, the error below
        # is thrown: `ValueError: incompatible dimensions for a grouped
        # binary operation: the group variable '<CHOSEN_FREQ>' is not a
        # dimension on the other argument`
        dv_climo = dv_climo.rename({"time": self._time_grouped.name})

        with xr.set_options(keep_attrs=True):
            # Use xarray's grouped arithmetic to subtract the climatology
            # from the observation data based on the groups.
            ds_departs = self._dataset.copy()
            ds_departs[data_var] = dv_obs_grouped - dv_climo
            ds_departs[data_var] = self._add_operation_attrs(ds_departs[data_var])

            # Drop the grouped time coordinates from the final output since
            # it is no longer needed.
            ds_departs = ds_departs.drop_vars(self._time_grouped.name)

        return ds_departs

    def center_times(self, dataset: xr.Dataset) -> xr.Dataset:
        """Centers the time coordinates using the midpoint between time bounds.

        Time coordinates can be recorded using different intervals, including
        the beginning, middle, or end of the interval. By centering time
        coordinates, it ensures any calculation using these values are performed
        reliably, regardless of the recorded interval.

        Parameters
        ----------
        dataset : xr.Dataset
            The Dataset with original time coordinates.

        Returns
        -------
        xr.Dataset
            The Dataset with centered time coordinates.
        """
        ds = dataset.copy()

        if hasattr(self, "_time_bounds") is False:
            self._time_bounds = ds.bounds.get_bounds("time")

        time_bounds = self._time_bounds.copy()
        lower_bounds, upper_bounds = (time_bounds[:, 0].data, time_bounds[:, 1].data)
        bounds_diffs: np.timedelta64 = (upper_bounds - lower_bounds) / 2
        bounds_mids: np.ndarray = lower_bounds + bounds_diffs

        time: xr.DataArray = ds.cf["T"].copy()
        time_centered = xr.DataArray(
            name=time.name,
            data=bounds_mids,
            coords={"time": bounds_mids},
            attrs=time.attrs,
        )
        time_centered.encoding = time.encoding
        ds = ds.assign_coords({"time": time_centered})

        # Update time bounds with centered time coordinates.
        time_bounds[time_centered.name] = time_centered
        self._time_bounds = time_bounds
        ds[time_bounds.name] = self._time_bounds
        return ds

    def _temporal_avg(
        self,
        data_var: str,
        mode: Mode,
        freq: Frequency,
        weighted: bool = True,
        center_times: bool = False,
        season_config: SeasonConfig = {
            "dec_mode": "DJF",
            "drop_incomplete_djf": False,
            "custom_seasons": None,
        },
    ) -> xr.Dataset:
        """Calculates the temporal average for a data variable."""
        self._set_obj_attrs(mode, freq, weighted, center_times, season_config)
        ds = self._dataset.copy()

        # Perform operations on the Dataset's time coordinates before operating
        # on the data variable so that these updates cascade down to it.
        if self._center_times:
            ds = self.center_times(ds)

        if (
            self._freq == "season"
            and self._season_config.get("dec_mode") == "DJF"
            and self._season_config.get("drop_incomplete_djf") is True
        ):
            ds = self._drop_incomplete_djf(ds)

        # Group the time coordinates and average the data variable using them.
        self._time_grouped = self._group_time_coords(ds.cf["T"])
        dv = get_data_var(ds, data_var)
        dv = self._averager(dv)

        # The dataset's original "time" dimension becomes obsolete after
        # calculating the climatology of the data variable, so it is dropped
        # and replaced.
        ds = ds.drop_dims("time")
        ds[dv.name] = dv

        return ds

    def _set_obj_attrs(
        self,
        mode: Mode,
        freq: Frequency,
        weighted: bool,
        center_times: bool,
        season_config: SeasonConfig,
    ):
        """Validates method arguments and sets them as object attributes.

        Parameters
        ----------
        mode : Mode
            The mode for temporal averaging.
        freq : Frequency
            The frequency of time to group by.
        weighted : bool
            Calculate averages using weights.
        center_times: bool
            If True, center time coordinates using the midpoint between of its
            upper and lower bounds. Otherwise, use the provided time
            coordinates, by default False.
        season_config: SeasonConfig
            A dictionary for "season" frequency configurations. If configs for
            predefined seasons are passed, configs for custom seasons are
            ignored and vice versa.

        Raises
        ------
        KeyError
            If the Dataset does not have a "time" dimension.
        ValueError
            If an incorrect ``freq`` arg was passed.
        ValueError
            If an incorrect ``dec_mode`` arg was passed.
        """
        # General configuration attributes.
        if mode not in list(MODES):
            modes = ", ".join(f'"{word}"' for word in MODES)
            raise ValueError(
                f"Incorrect `mode` argument. Supported modes include: " f"{modes}."
            )
        freq_keys = DATETIME_COMPONENTS[mode].keys()
        if freq not in freq_keys and "hour" not in freq:
            raise ValueError(
                f"Incorrect `freq` argument. Supported frequencies for {mode} "
                f"include: {list(freq_keys)}."
            )

        self._time_bounds = self._dataset.bounds.get_bounds("time").copy()
        self._mode = mode
        self._freq = freq
        self._weighted = weighted
        self._center_times = center_times

        # "season" frequency specific configuration attributes.
        for key in season_config.keys():
            if key not in SEASON_CONFIG_KEYS:
                raise KeyError(
                    f"'{key}' is not a supported season config. Supported "
                    f"configs include: {SEASON_CONFIG_KEYS}."
                )
        custom_seasons = season_config.get("custom_seasons", None)
        dec_mode = season_config.get("dec_mode", "DJF")
        drop_incomplete_djf = season_config.get("drop_incomplete_djf", False)

        self._season_config: SeasonConfig = {}
        if custom_seasons is None:
            if dec_mode not in ("DJF", "JFD"):
                raise ValueError(
                    "Incorrect 'dec_mode' key value for `season_config`. "
                    "Supported modes include 'DJF' or 'JFD'."
                )
            self._season_config["dec_mode"] = dec_mode

            if dec_mode == "DJF":
                self._season_config["drop_incomplete_djf"] = drop_incomplete_djf
        else:
            self._season_config["custom_seasons"] = self._form_seasons(custom_seasons)

    def _drop_incomplete_djf(self, dataset: xr.Dataset) -> xr.Dataset:
        """Drops incomplete DJF seasons within a continuous time series.

        This method assumes that the time series is continuous and removes the
        leading and trailing incomplete seasons (e.g., the first January and
        February of a time series that are not complete, because the December of
        the previous year is missing). This method does not account for or
        remove missing time steps anywhere else.

        Parameters
        ----------
        data_var : xr.DataArray
            The data variable with some incomplete DJF seasons.

        Returns
        -------
        xr.DataArray
            The data variable with all complete DJF seasons.
        """
        # Separate the dataset into two datasets, one with and one without
        # the time dimension. This is necessary because the xarray .where()
        # method concatenates the time dimension to non-time dimension data
        # vars, which is not a desired behavior.
        ds = dataset.copy()
        ds_time = ds.get([v for v in ds.data_vars if "time" in ds[v].dims])
        ds_no_time = ds.get([v for v in ds.data_vars if "time" not in ds[v].dims])

        start_year, end_year = (ds.time.dt.year.values[0], ds.time.dt.year.values[-1])
        incomplete_seasons = (f"{start_year}-01", f"{start_year}-02", f"{end_year}-12")
        for year_month in incomplete_seasons:
            try:
                coord_pt = ds.loc[dict(time=year_month)].time[0]
                ds_time = ds_time.where(ds_time.time != coord_pt, drop=True)  # type: ignore
                self._time_bounds = ds_time[self._time_bounds.name]
            except KeyError:
                continue

        ds_final = xr.merge((ds_time, ds_no_time))  # type: ignore
        return ds_final

    def _form_seasons(
        self, custom_seasons: Union[List[str], List[List[str]]]
    ) -> List[str]:
        """Forms custom seasons from a nested list of months.

        This method concatenates the strings in each sublist to form a
        a flat list of custom season strings

        Parameters
        ----------
        custom_seasons : List[List[str]]
            List of sublists containing month strings, with each sublist
            representing a custom season.

        Returns
        -------
        List[str]
           List of strings representing seasons.

        Raises
        ------
        ValueError
            If exactly 12 months are not passed in the list of custom seasons.
        ValueError
            If a duplicate month(s) were found in the list of custom seasons.
        ValueError
            If a month string(s) is not supported.
        """
        predefined_months = list(MONTH_INT_TO_STR.values())
        input_months = list(chain.from_iterable(custom_seasons))

        if len(input_months) != len(predefined_months):
            raise ValueError(
                "Exactly 12 months were not passed in the list of custom seasons."
            )
        if len(input_months) != len(set(input_months)):
            raise ValueError(
                "Duplicate month(s) were found in the list of custom seasons."
            )

        for month in input_months:
            if month not in predefined_months:
                raise ValueError(
                    f"The following month is not supported: '{month}'. "
                    f"Supported months include: {predefined_months}."
                )

        c_seasons = ["".join(season) for season in custom_seasons]
        return c_seasons

    def _averager(self, data_var: xr.DataArray) -> xr.DataArray:
        """Averages a data variable by a grouping frequency.

        This method groups the data variable's values by the time coordinates
        and averages them with or without weights. The parameters for
        ``self._temporal_average()`` are stored as DataArray attributes in the
        averaged data variable.

        Parameters
        ----------
        data_var : xr.DataArray
            The data variable.

        Returns
        -------
        xr.DataArray
            The averaged data variable.
        """
        dv = data_var.copy()

        if self._weighted:
            self._weights = self._get_weights(dv)
            dv *= self._weights
            dv = self._groupby_freq(dv).sum()  # type: ignore
        else:
            dv = self._groupby_freq(dv).mean()  # type: ignore

        # After grouping and aggregating on the grouped time coordinates, the
        # original time dimension is replaced with the grouped time dimension.
        # For example, grouping on "year_season" replaces the "time" dimension
        # with "year_season". This dimension will eventually be renamed back
        # to "time" when the data variable as added back to the dataset.
        dv = dv.rename({self._time_grouped.name: "time"})  # type: ignore

        # After grouping and aggregating, the grouped time dimension's
        # attributes are removed. Unfortunately, `xr.set_options(keep_attrs=True)`,
        # `.sum(keep_attrs=True)`, and `.mean(keep_attrs=True)` only keeps
        # attributes for data variables and not their coordinates so they need
        # to be restored manually
        dv["time"].attrs = self._time_grouped.attrs
        dv["time"].encoding = self._time_grouped.encoding

        dv = self._add_operation_attrs(dv)

        return dv

    def _group_time_coords(self, time_coords: xr.DataArray) -> xr.DataArray:
        """Groups the time coordinates by a frequency.

        This method extracts xarray datetime components from the time
        coordinates and stores them as column values in a pandas DataFrame. A
        pandas DataFrame is the chosen data structure because it simplifies the
        additional steps for processing the component values, specifically for
        the "season" frequency. The DataFrame is then converted to a numpy
        list of cftime.datetime or datetime.datetime that is used as the data
        for the final xarray DataArray of grouped time coordinates.

        Parameters
        ----------
        time_coords : xr.DataArray
            The time coordinates.

        Returns
        -------
        xr.DataArray
            The time coordinates grouped by a frequency.

        Notes
        -----
        Refer to [5]_ for information on xarray datetime accessor components.

        References
        ----------
        .. [5] https://xarray.pydata.org/en/stable/user-guide/time-series.html#datetime-components

        Examples
        --------

        Original daily time coordinates:

        >>> # Original daily time coordinates.
        >>> <xarray.DataArray 'time' (time: 4)>
        >>> array(['2000-01-01T12:00:00.000000000',
        >>>        '2000-01-31T21:00:00.000000000',
        >>>        '2000-03-01T21:00:00.000000000',
        >>>        '2000-04-01T03:00:00.000000000'],
        >>>       dtype='datetime64[ns]')
        >>> Coordinates:
        >>> * time     (time) datetime64[ns] 2000-01-01T12:00:00 ... 2000-04-01T03:00:00
        >>> Attributes:
        >>>     long_name:      time
        >>>     standard_name:  time
        >>>     axis:           T
        >>>     bounds:         time_bnds

        Daily time coordinates grouped by month for time series averaging:

        >>> <xarray.DataArray 'time' (time: 3)>
        >>> array(['2000-01-01T00:00:00.000000000',
        >>>        '2000-03-01T00:00:00.000000000',
        >>>        '2000-04-01T00:00:00.000000000'],
        >>>       dtype='datetime64[ns]')
        >>> Coordinates:
        >>> * time     (time) datetime64[ns] 2000-01-01T00:00:00 ... 2000-04-01T00:00:00
        >>> Attributes:
        >>>     long_name:      time
        >>>     standard_name:  time
        >>>     axis:           T
        >>>     bounds:         time_bnds
        """
        df = pd.DataFrame()

        for component in DATETIME_COMPONENTS[self._mode][self._freq]:
            df[component] = time_coords[f"time.{component}"].values

        if self._freq == "season":
            df = self._process_season_dataframe(df)

        datetime_objs = self._convert_df_to_dt_objects(df)

        time_grouped = xr.DataArray(
            name="_".join(df.columns),
            data=datetime_objs,
            coords={"time": time_coords.time},
            dims=["time"],
            attrs=time_coords.time.attrs,
        )
        time_grouped.encoding = time_coords.time.encoding

        return time_grouped

    def _process_season_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Processes a DataFrame of xarray datetime components for the "season"
        frequency.

        Processing includes:
        * Mapping custom seasons (if applicable).
        * If season with December is "DJF", shift Decembers over to the next
          year so DJF groups are correctly formed.
        * Drop obsolete columns after processing is done.

        Parameters
        ----------
        df : pd.DataFrame
            A DataFrame of xarray datetime components.

        Returns
        -------
        pd.DataFrame
            A DataFrame of processed xarray datetime components.
        """
        df_new = df.copy()
        custom_seasons = self._season_config.get("custom_seasons")
        dec_mode = self._season_config.get("dec_mode")

        if custom_seasons is None:
            if dec_mode == "DJF":
                df_new = self._shift_decembers(df_new)
        else:
            df_new = self._map_custom_seasons(df_new)

        df_new = self._drop_obsolete_columns(df_new)
        return df_new

    def _convert_df_to_dt_objects(self, df: pd.DataFrame) -> np.ndarray:
        """
        Converts a DataFrame of xarray datetime components into a numpy ndarray
        of cftime.datetime or datetime.datetime objects.

        # TODO: Handle custom seasons

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame of xarray datetime components.

        Returns
        -------
        np.ndarray
            A numpy ndarray of datetime.datetime or cftime.datetime objects.

        Examples
        --------
        """
        df_new = df.copy()

        if self._freq == "season":
            # datetime objects don't store seasons as strings, so the middle
            # month is used to represent the season. For example, the middle
            # month "J" is mapped to the integer value 1 to represent the "DJF"
            # season in the datetime object.
            # TODO:
            df_new["season"] = df_new.season.map(SEASON_TO_MONTH)
            df_new = df_new.rename(columns={"season": "month"})

        # Some time frequencies don't require all of the datetime components
        # for grouping, so default values are used for creating the `datetime`
        # objects (which require at least a year, month, and day).
        dt_components_defaults = {"year": 1, "month": 1, "day": 1, "hour": 0}
        for component, default_val in dt_components_defaults.items():
            if component not in df_new.columns:
                df_new[component] = default_val

        if self._mode == "time_series":
            dates = pd.to_datetime(df_new).to_numpy()
        elif self._mode in ["climatology", "departures"]:
            # The "year" values are not considered when grouping the time
            # coordinates for "climatology" and "departures", but are required
            # for creating datetime objects. The fallback value of 1 is
            # used as a placeholder for the year. However, year 1 is outside the
            # Timestamp-valid range so `cftime.datetime` objects are used
            # instead of `datetime.datetime`.
            # https://docs.xarray.dev/en/stable/user-guide/weather-climate.html#non-standard-calendars-and-dates-outside-the-timestamp-valid-range
            # https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timestamp-limitations
            dates = np.array(
                [
                    cftime.datetime(year, month, day, hour)
                    for year, month, day, hour in zip(
                        df_new.year, df_new.month, df_new.day, df_new.hour
                    )
                ]
            )

        return dates

    def _map_custom_seasons(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Maps months to custom seasons in the DataFrame of xarray datetime
        components.

        This method maps each integer value in the "month" column to its string
        represention, which then maps to its respective custom season in the
        "season" column. For example, 1 maps to "Jan", and "Jan" maps to the
        custom "JanFebMar" season.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame of xarray datetime components.

        Returns
        -------
        pd.DataFrame
            The DataFrame of xarray datetime coordinates, with each row mapped
            to a custom season.
        """
        custom_seasons = self._season_config.get("custom_seasons")
        if custom_seasons is None:
            raise ValueError("Custom seasons were not assigned to this object.")

        # Time complexity of O(n^2), but okay with these small data structures.
        seasons_map = {}
        for mon_int, mon_str in MONTH_INT_TO_STR.items():
            for season in custom_seasons:
                if mon_str in season:
                    seasons_map[mon_int] = season

        df_new = df.copy()
        df_new["season"] = df_new["month"].map(seasons_map)
        return df_new

    def _shift_decembers(self, df_season: pd.DataFrame) -> pd.DataFrame:
        """Shifts Decembers over to the next year for "DJF" seasons.

        For "DJF" seasons, Decembers must be shifted over to the next year in
        order for the xarray groupby operation to correctly group the time
        coordinates. Otherwise, grouping is incorrectly performed with the
        native xarray "DJF" season, which is actually "JFD".

        Parameters
        ----------
        df_season : pd.DataFrame
            The DataFrame of xarray datetime components produced using the
            "season" frequency.

        Returns
        -------
        pd.DataFrame
            The DataFrame of xarray datetime components with Decembers shifted
            over to the next year.

        Examples
        --------

        Comparison of "DJF" and "JFD" seasons:

        >>> # "DJF" (shifted Decembers)
        >>> [(2000, "DJF", 1), (2000, "DJF", 2), (2001, "DJF", 12),
        >>>  (2001, "DJF", 1), (2001, "DJF", 2), (2002, "DJF", 12)]

        >>> # "JFD" (native xarray behavior)
        >>> [(2000, "DJF", 1), (2000, "DJF", 2), (2000, "DJF", 12),
        >>>  (2001, "DJF", 1), (2001, "DJF", 2), (2001, "DJF", 12)]
        """
        df_season.loc[df_season["month"] == 12, "year"] = df_season["year"] + 1
        return df_season

    def _drop_obsolete_columns(self, df_season: pd.DataFrame) -> pd.DataFrame:
        """
        Drops obsolete columns from the DataFrame of xarray datetime components.

        For the "season" frequency, processing is required on the DataFrame of
        xarray datetime components, such as mapping custom seasons based on the
        month. Additional datetime component values must be included as
        DataFrame columns, which become obsolete after processing is done. The
        obsolete columns are dropped from the DataFrame before grouping
        time coordinates.

        Parameters
        ----------
        df_season : pd.DataFrame
            The DataFrame of time coordinates for the "season" frequency with
            obsolete columns.

        Returns
        -------
        pd.DataFrame
            The DataFrame of time coordinates for the "season" frequency with
            obsolete columns dropped.
        """
        if self._mode == "time_series":
            df_season = df_season.drop("month", axis=1)
        elif self._mode in ["climatology", "departures"]:
            df_season = df_season.drop(["year", "month"], axis=1)
        else:
            raise ValueError(
                "Unable to drop columns in the datetime components "
                f"DataFrame for unsupported mode, '{self._mode}'."
            )

        return df_season

    def _get_weights(self, data_var: xr.DataArray) -> xr.DataArray:
        """Calculates weights for a data variable using time bounds.

        This method gets the length of time for each coordinate point by using
        the difference in the upper and lower time bounds. This approach ensures
        that the correct time lengths are calculated regardless of how time
        coordinates are recorded (e.g., monthly, daily, hourly) and the calendar
        type used.

        The time lengths are grouped by the grouping frequency, then each time
        length is divided by the total sum of the time lengths in its group to
        get the weights. The sum of the weights for each group is validated to
        ensure it equals 1.0 (100%).

        Parameters
        -------
        data_var : xr.DataArray
            The data variable.

        Returns
        -------
        xr.DataArray
            The weights based on a specified frequency.

        Notes
        -----
        Refer to [6]_ for the supported CF convention calendar types.

        References
        ----------
        .. [6] https://cfconventions.org/cf-conventions/cf-conventions.html#calendar
        """
        with xr.set_options(keep_attrs=True):
            time_lengths: xr.DataArray = (
                self._time_bounds[:, 1] - self._time_bounds[:, 0]
            )

        # Must be convert dtype from timedelta64[ns] to float64, specifically
        # when chunking DataArrays using Dask. Otherwise, the numpy warning
        # below is thrown: `DeprecationWarning: The `dtype` and `signature`
        # arguments to ufuncs only select the general DType and not details such
        # as the byte order or time unit (with rare exceptions see release
        # notes). To avoid this warning please use the scalar types
        # `np.float64`, or string notation.`
        time_lengths = time_lengths.astype(np.float64)
        time_grouped = self._groupby_freq(time_lengths)
        weights: xr.DataArray = time_grouped / time_grouped.sum()  # type: ignore

        self._validate_weights(data_var, weights)
        return weights

    def _validate_weights(self, data_var: xr.DataArray, weights: xr.DataArray):
        """Validates the sums of the weights for each group equals 1.0 (100%).

        Parameters
        ----------
        data_var : xr.DataArray
            The data variable.
        weights : xr.DataArray
            The data variable's time coordinates weights.
        """
        freq_groups = self._groupby_freq(data_var).count()  # type: ignore
        # Sum the frequency group counts by all the dims except the grouped time
        # dimension to get a 1D array of counts.
        summing_dims = tuple(
            x for x in freq_groups.dims if x != self._time_grouped.name
        )
        freq_sums = freq_groups.sum(summing_dims)

        # Replace all non-zero counts with 1.0 (total weight of 100%).
        expected_sum = np.where(freq_sums > 0, 1.0, freq_sums)
        actual_sum = self._groupby_freq(weights).sum().values  # type: ignore
        np.testing.assert_allclose(actual_sum, expected_sum)

    def _groupby_freq(self, data_var: xr.DataArray) -> DataArrayGroupBy:
        """Groups a data variable by a time frequency.

        This method returning a DataArrayGroupBy object, enabling support for
        xarray's grouped arithmetic as a shortcut for mapping over all unique
        labels.

        Parameters
        ----------
        data_var : xr.DataArray
            A data variable.

        Returns
        -------
        DataArrayGroupBy
            A data variable grouped by the frequency.
        """
        dv = data_var.copy()
        dv.coords[self._time_grouped.name] = self._time_grouped
        dv_gb = dv.groupby(self._time_grouped.name)
        return dv_gb

    def _add_operation_attrs(self, data_var: xr.DataArray) -> xr.DataArray:
        """Adds attributes to the data variable describing the operation.
        These attributes distinguish a data variable that has been operated on
        from its original state. The attributes in netCDF4 files do not support
        booleans or nested dictionaries, so booleans are converted to strings
        and nested dictionaries are unpacked.

        Parameters
        ----------
        data_var : xr.DataArray
            The data variable.

        Returns
        -------
        xr.DataArray
            The data variable with a temporal averaging attributes.
        """
        data_var.attrs.update(
            {
                "operation": "temporal_avg",
                "mode": self._mode,
                "freq": self._freq,
                "groupby": self._time_grouped.name,
                "weighted": str(self._weighted),
                "center_times": str(self._center_times),
            }
        )

        if self._freq == "season":
            custom_seasons = self._season_config.get("custom_seasons")
            dec_mode = self._season_config.get("dec_mode")
            drop_incomplete_djf = self._season_config.get("drop_incomplete_djf")

            if custom_seasons is None:
                data_var.attrs["dec_mode"] = dec_mode
                if dec_mode == "DJF":
                    data_var.attrs["drop_incomplete_djf"] = str(drop_incomplete_djf)
            else:
                data_var.attrs["custom_seasons"] = custom_seasons

        return data_var
