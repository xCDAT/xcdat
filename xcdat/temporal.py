"""Module containing temporal functions."""
from itertools import chain
from typing import Dict, List, Literal, Optional, Tuple, TypedDict, Union, get_args

import cf_xarray  # noqa: F401
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

# A dictionary mapping frequencies to xarray datetime accessor components. The
# datetime components are used to create a pandas MultiIndex for grouping
# operations.
# For the "season" frequency, "year" and "month" (for time_series) and "month"
# (for climatology) are included so that additional processing can be performed
# when creating "DJF" or "JFD" groups in the MultiIndex. Before grouping,
# "year" and "month" or just "month" are removed from the MultiIndex.
FREQ_TO_COMPONENTS = {
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
        "season": ("year", "season", "month"),
        "month": ("month",),
        "day": ("month", "day"),
    },
}

#: Dictionary mapping month integer to string.
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

    def temporal_avg(
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
        """Calculates the temporal average for a data variable.

        Parameters
        ----------
        data_var: str
            The key of the data variable to temporally average.
        mode : Mode
            The temporal averaging mode, either "time_series" or "climatology".
        freq : Frequency
            The frequency of time to group by.

            "time_series" frequencies:

            * "year", "season", "month", "day", "hour"

                * "year" groups by year for the yearly average
                * "season" groups by (year, season) for the seasonal average
                * "month" groups by (year, month) for the monthly average
                * "day" groups by (year, month, day) for the daily average
                * "hour" groups by (year, month, day, hour) for the hourly
                  average

            "climatology" frequencies:

            * "season", "month", "day"

                * "season" groups by season for the seasonal cycle
                * "month" groups by month for the annual cycle
                * "day" groups by (month, day) for the daily cycle

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
        Import temporal accessor:

        >>> import xcdat

        Call temporal averaging method:

        >>> # First option
        >>> ds.temporal.temporal_avg(...)
        >>> # Second option
        >>> ds.xcdat.temporal_avg(...)

        Get time series average for a variable:

        >>> ds_year = ds.temporal.temporal_avg("ts", "time_series", "year")
        >>> ds_year.ts

        Get time series average for a variable ("season" freq):

        >>> ds_season_with_djf = ds.temporal.temporal_avg(
        >>>     "ts",
        >>>     "time_series",
        >>>     "season",
        >>>     season_config={
        >>>         "dec_mode": "DJF",
        >>>         "drop_incomplete_season": True
        >>>     }
        >>> )
        >>> ds_season_with_djf.ts
        >>>
        >>> ds_season_with_jfd = ds.temporal.temporal_avg(
        >>>     "ts",
        >>>     "time_series",
        >>>     "season",
        >>>     season_config={"dec_mode": "JFD"}
        >>> )
        >>> ds_season_with_jfd.ts

        Get time series average for a variable ("season" freq & custom seasons):

        >>> custom_seasons = [
        >>>     ["Jan", "Feb", "Mar"],  # "JanFebMar"
        >>>     ["Apr", "May", "Jun"],  # "AprMayJun"
        >>>     ["Jul", "Aug", "Sep"],  # "JunJulAug"
        >>>     ["Oct", "Nov", "Dec"],  # "OctNovDec"
        >>> ]
        >>>
        >>> ds_season_custom = ds.temporal.temporal_avg(
        >>>     "ts",
        >>>     "climatology",
        >>>     "season",
        >>>     season_config={"custom_seasons": custom_seasons}
        >>> )

        Get the temporal averaging operation attributes:

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

        Write temporally averaged dataset to a netCDF file:

        >>> # Xarray cannot serialize MultiIndex objects to netCDF yet. Must
        >>> # call reset_index() on the MultiIndex first.
        >>> ds_season_with_djf.reset_index("season").to_netcdf("name_of_file.nc")
        """
        self._set_obj_attrs(mode, freq, weighted, center_times, season_config)
        ds = self._dataset.copy()
        dv = get_data_var(ds, data_var)

        if self._center_times:
            ds = self.center_times(ds)

        if (
            self._freq == "season"
            and self._season_config.get("dec_mode") == "DJF"
            and self._season_config.get("drop_incomplete_djf") is True
        ):
            ds = self._drop_incomplete_djf(ds)

        # Retrieve the data variable after dataset operations are performed.
        dv = get_data_var(ds, data_var)
        # Save the original variable for calculating climatology departures.
        ds[f"{dv.name}_original"] = dv.copy()
        ds[dv.name] = self._averager(dv)

        # FIXME: Adding weights removes time encoding
        # ds[f"{dv.name}_weights"] = self._weights

        return ds

    def departures(self, data_var: str) -> xr.Dataset:
        """Calculates departures (anomalies) for a climatology data variable.

        In climatology, “anomalies” refer to the difference between the value
        during a given time interval (e.g., the January average surface air
        temperature) and the long-term average value for that time interval
        (e.g., the average surface temperature over the last 30 Januaries).

        This method uses xarray's grouped arithmetic as a shortcut for mapping
        over all unique labels. Grouped arithmetic works by assigning a label
        (the MultiIndex groups) to each time coordinate of the observation data.
        Afterwards, the corresponding climatology is removed from the
        observation data at each time coordinate based on the matching labels.

        xarray's grouped arithmetic operates over each group of the MultiIndex
        without changing the size of the data variable/dataset. For example,
        the original monthly time coordinates are maintained when calculating
        seasonal departures on monthly data. Visit [2]_ to learn more about
        how xarray's grouped arithmetic works.

        Parameters
        ----------
        data_var: str
            The key of the data variable to calculate departures for.

        Returns
        -------
        xr.Dataset
            The Dataset containing the departures for a data var's climatology.

        References
        ----------
        .. [2] https://xarray.pydata.org/en/stable/user-guide/groupby.html#grouped-arithmetic

        Examples
        --------
        Import temporal averaging functionality:

        >>> import xcdat

        Get departures for an annual cycle climatology:

        >>> ds_climo = ds.temporal.temporal_avg("ts", "climatology", "month")
        >>> ds_depart = ds_climo.temporal.departures("ts")

        Get the departures operation attributes:

        >>> ds_depart.ts_departures.attrs
        {
            'operation': 'temporal_avg',
            'mode': 'departures',
            'frequency': 'season',
            'groupby': 'season',
            'weighted': 'True',
            'center_times': 'False',
            'dec_mode': 'DJF',
            'drop_incomplete_djf': 'False'
        }
        """
        ds = self._dataset.copy()
        dv_climo = get_data_var(ds, data_var)
        self._time_bounds: xr.DataArray = ds.time_bnds.copy()

        # Reuse attributes from the climatology data variable except for the
        # 'mode' attribute.
        attrs = dv_climo.attrs
        prev_operation = attrs.get("operation")

        # TODO: This method will most likely need to handle user-supplied climatologies,
        # which might not include an `operation` attribute if it was calculated
        # outside of xcdat.
        if prev_operation is None:
            raise KeyError(
                f"'{dv_climo.name}' does not include the 'operation' attribute to "
                "indicate that it is a climatology data variable. Make sure to "
                f"calculate a climatology for '{dv_climo.name}' before calculating its "
                "departures."
            )
        else:
            if attrs["mode"] != "climatology":
                raise ValueError(
                    "Departures can only be calculated for a climatology data variable."
                )

            self._set_obj_attrs(
                "departures",
                attrs["freq"],
                str_to_bool(attrs["weighted"]),
                str_to_bool(attrs["center_times"]),
                {
                    "dec_mode": attrs.get("dec_mode", "DJF"),
                    "drop_incomplete_djf": str_to_bool(
                        attrs.get("drop_incomplete_djf", "False")
                    ),
                    "custom_seasons": attrs.get("custom_seasons", None),
                },
            )

            # If the MultiIndex in the Dataset was reset to coordinate variables,
            # convert the coordinates variables back to a MultiIndex. This step
            # is required in order for departures to be correctly calculated
            # through index alignment. Otherwise, nans will appear because the
            # indexes for the dimension don't align since one is a MultiIndex
            # and the other are coordinate variables.
            if not self._is_groupby_dim_a_multiindex(dv_climo):
                ds = self._convert_coord_vars_to_multiindex(ds, dv_climo)
                dv_climo = ds[data_var]

            # Perform xarray grouped arithmetic for calculating departures by
            # subtracting the grouped observation data and the climatology.
            departures = ds.copy()
            dv_og = ds[f"{data_var}_original"].copy()
            dv_obs = self._group_obs_by_multiindex(dv_og)
            with xr.set_options(keep_attrs=True):
                dv_departures = dv_obs - dv_climo
                dv_departures = self._add_operation_attrs(dv_departures)
                departures[data_var] = dv_departures

        return departures

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
        freq_keys = FREQ_TO_COMPONENTS[mode].keys()
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
        """Forms custom seasons.

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
            If duplicate months are found in the list of custom seasons.
        ValueError
            If a month string(s) are not supported.
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
        """Averages a data variable using a MultiIndex.

        A pandas MultiIndex is derived from the time coordinates. It enables
        grouping on multiple datetime parameters such as year and month,
        which xarray does not support. After grouping, the parameters passed to
        the averaging operation are stored in the data variable's dictionary of
        attributes.

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
        self._multiindex, self._multiindex_name = self._create_multiindex(dv)

        if self._weighted:
            self._weights = self._get_weights(dv)
            dv *= self._weights
            dv = self._groupby_multiindex(dv).sum()  # type: ignore
        else:
            dv = self._groupby_multiindex(dv).mean()  # type: ignore

        dv = self._add_operation_attrs(dv)
        return dv

    def _create_multiindex(self, data_var: xr.DataArray) -> Tuple[pd.MultiIndex, str]:
        """Creates a MultiIndex to the data variable using time coordinates.

        This method creates a pandas DataFrame from the time coordinates by
        extracting xarray datetime accessor components based on the specified
        grouping frequency. If the "season" frequency is chosen, additional
        processing is performed on the DataFrame, including mapping a custom
        season per coordinate point. The DataFrame is converted to a pandas
        MultiIndex, which is stored in the data variable and used for grouping
        operations.

        Parameters
        ----------
        data_var : xr.DataArray
            The data variable.

        Returns
        -------
        Tuple[pd.MultiIndex, str]
          The MultiIndex and its name. The name is also used for the name of the
          associated Dataset/DataArray dimension.

        Notes
        -----
        Refer to [3]_ for information on xarray datetime accessor components.

        References
        -----------
        .. [3] https://xarray.pydata.org/en/stable/user-guide/time-series.html#datetime-components

        Examples
        --------

        MultiIndex extracted from time coords for daily time series averaging:

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

        >>> <xarray.DataArray 'year_month_day_hour' (year_month_day_hour: 4)>
        >>> array([(2000, 1, 1, 12),
        >>>        (2000, 1, 31, 21),
        >>>        (2000, 3, 1, 21),
        >>>        (2000, 4, 1, 3)],
        >>>       dtype=object)
        >>> Coordinates:
        >>> * year_month_day_hour          (year_month_day_hour) MultiIndex
        >>> - year_month_day_hour_level_0  (year_month_day_hour) int64 2000 2000 2000 2000
        >>> - year_month_day_hour_level_1  (year_month_day_hour) int64 1 1 3 4
        >>> - year_month_day_hour_level_2  (year_month_day_hour) int64 1 31 1 1
        >>> - year_month_day_hour_level_3  (year_month_day_hour) int64 12 21 21 3
        """
        df = pd.DataFrame()
        for component in FREQ_TO_COMPONENTS[self._mode][self._freq]:
            df[component] = data_var[f"time.{component}"].values

        if self._freq == "season":
            df = self._process_season_dataframe(df)

        m_index_lvl_names = [(index, col) for index, col in enumerate(df.columns)]
        m_index = pd.MultiIndex.from_frame(df, names=m_index_lvl_names)
        m_index_name = "_".join(df.columns)
        return m_index, m_index_name

    def _process_season_dataframe(self, df_season: pd.DataFrame) -> pd.DataFrame:
        """Processes the time coordinates DataFrame for the "season" frequency.

        Operations include:
        * Mapping custom seasons if used.
        * If season with December is "DJF", shift Decembers over to the next
          year so DJF groups are correctly formed.
        * Drop obsolete columns after processing is done.

        Parameters
        ----------
        df_season : pd.DataFrame
            The DataFrame of time coordinates.

        Returns
        -------
        pd.DataFrame
            The processed DataFrame of time coordinates.
        """
        if self._season_config.get("custom_seasons") is not None:
            df_season = self._map_custom_seasons(df_season)
        elif self._season_config.get("dec_mode") == "DJF":
            df_season = self._shift_decembers(df_season)

        df_season = self._drop_obsolete_columns(df_season)
        return df_season

    def _map_custom_seasons(self, df_season: pd.DataFrame) -> pd.DataFrame:
        """Maps months to custom seasons in the time coordinates DataFrame.

        This method maps each integer value in the "month" column to its string
        represention, which then maps to its respective custom season in the
        "season" column. For example, 1 maps to "Jan", and "Jan" maps to the
        custom "JanFebMar" season.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame of time coordinates using the "season" frequency.

        Returns
        -------
        pd.DataFrame
            The DataFrame of time coordinates, with each row mapped to a custom
            season.
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

        df = df_season.copy()
        df["season"] = df["month"].map(seasons_map)
        return df

    def _shift_decembers(self, df_season: pd.DataFrame) -> pd.DataFrame:
        """Shifts Decembers over to the next year for "DJF" seasons.

        For "DJF" seasons, Decembers must be shifted over to the next year in
        order for the xarray groupby operation to perform the correct grouping.
        Otherwise, grouping is performed with the native xarray "DJF" season
        (which is actually "JFD").

        Parameters
        ----------
        df_season : pd.DataFrame
            The DataFrame of time coordinates using the "season" frequency.

        Returns
        -------
        pd.DataFrame
            The DataFrame of time coordinates with Decembers shifted over to the
            next year.

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
        """Drops obsolete columns from the time coordinates DataFrame.

        Additional processing is performed on the time coordinates DataFrame
        for the "season" frequency (refer to ``_process_season_dataframe()``).
        Processing requires additonal columns related to the time coordinates,
        which become obsolete after it is done. The obsolete columns must be
        dropped from the DataFrame before creating the time MultiIndex.
        Otherwise, the time MultiIndex will include additional levels,
        resulting in incorrect grouping outputs.

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

        The time lengths are grouped by the time MultiIndex, then each time
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
        Refer to [4]_ for the supported CF convention calendar types.

        References
        -----
        .. [4] https://cfconventions.org/cf-conventions/cf-conventions.html#calendar
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
        time_grouped = self._groupby_multiindex(time_lengths)
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
        freq_groups = self._groupby_multiindex(data_var).count()  # type: ignore
        # Sum the frequency group counts by all the dims except the MultiIndex
        # for a 1D array of counts.
        summing_dims = tuple(x for x in freq_groups.dims if x != self._multiindex.name)
        freq_sums = freq_groups.sum(summing_dims)

        # Replace all non-zero counts with 1.0 (total weight of 100%).
        expected_sum = np.where(freq_sums > 0, 1.0, freq_sums)
        actual_sum = self._groupby_multiindex(weights).sum().values  # type: ignore
        np.testing.assert_allclose(actual_sum, expected_sum)

    def _group_obs_by_multiindex(self, data_var: xr.DataArray) -> DataArrayGroupBy:
        """Groups observation data by the MultiIndex to calculate departures.

        This method returning a DataArrayGroupBy object, enabling support for
        xarray's grouped arithmetic as a shortcut for mapping over all unique
        labels.

        Parameters
        ----------
        data_var : xr.DataArray
            The data variable.

        Returns
        -------
        DataArrayGroupBy
            The data variable grouped by the MultiIndex.
        """
        dv = data_var.copy()
        self._multiindex, self._multiindex_name = self._create_multiindex(dv)
        dv_gb = self._groupby_multiindex(dv)
        return dv_gb

    def _groupby_multiindex(self, data_var: xr.DataArray) -> DataArrayGroupBy:
        """Adds the MultiIndex to the data variable and groups by it.

        Parameters
        ----------
        data_var : xr.DataArray
            The data variable.

        Returns
        -------
        DataArrayGroupBy
            The data variable grouped by the MultiIndex.
        """
        dv = data_var.copy()
        dv.coords[self._multiindex_name] = ("time", self._multiindex)
        dv_gb = dv.groupby(self._multiindex_name)
        return dv_gb

    def _convert_coord_vars_to_multiindex(
        self, dataset: xr.Dataset, data_var: xr.DataArray
    ) -> xr.Dataset:
        """Converts the groupby dimension's coordinate vars to a MultiIndex.

        In xarray, calling ``.reset_index()`` on a MultiIndex will flatten it
        into coordinates variables. A common use case for calling .reset_index()
        is before writing a Dataset with a MultiIndex to a netCDF file. This
        is a required step because MultiIndex serialization is not supported yet
        in xarray (https://github.com/pydata/xarray/issues/1077). Also, there is
        there is no existing xarray method for converting coordinate variables
        back into a MultiIndex after the MultiIndex is reset.

        An example use case for this method is during the calculation of
        departures. The formula for calculating departures involves substracting
        the grouped observation data from the climatology. If the indexes
        between both data variables don't align (caused by resetting the
        MultiIndex), the formula will produce np.nans. Therefore, the grouping
        dimension in both data variables must be a MultiIndex.

        Parameters
        ----------
        dataset : xr.Dataset
            The dataset with a groupby dimension consisting of coordinate vars.
        data_var : xr.DataArray
            The data var with a groupby dimension consisting of coordinate vars.

        Returns
        -------
        xr.Dataset
            The dataset with a groupby dimension of a MultiIndex.
        """
        ds = dataset.copy()
        dv = data_var.copy()
        gb = dv.attrs["groupby"]

        df = pd.DataFrame()
        level_names = list(ds[gb].coords.keys())
        for name in level_names:
            df[name] = ds[name].values
            ds = ds.rename({name: f"{name}_old"})
        m_index = pd.MultiIndex.from_frame(df, names=level_names)

        ds = ds.assign_coords({gb: m_index})
        ds = ds.drop_vars([f"{name}_old" for name in level_names])
        return ds

    def _is_groupby_dim_a_multiindex(self, data_var: xr.DataArray) -> bool:
        """Checks if the groupby dimension is a MultiIndex.

        A groupby dimension can consist of a MultiIndex or coordinate
        variable(s).

        Parameters
        ----------
        data_var : xr.DataArray
            The data variable.

        Returns
        -------
        bool
            True if groupby dimension is a MultiIndex, else False.
        """
        gb_dim = data_var.attrs["groupby"]
        gb_coord: xr.DataArray = data_var[gb_dim]

        # ._level_coords is a dictionary that is populated if the index is
        # a MultiIndex. If it is empty, the index is not a MultiIndex.
        return bool(gb_coord._level_coords)

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
                "groupby": self._multiindex_name,
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
