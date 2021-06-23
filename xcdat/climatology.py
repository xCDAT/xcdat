"""Functions related to calculating climatology cycles and departures."""

from typing import Dict, Optional, Union, get_args

import numpy as np
import xarray as xr
from typing_extensions import Literal

from xcdat.log import logging

# PERIODS
# =======
# Type alias representing climatology periods for the ``frequency`` param.
Period = Literal["month", "season", "year"]
# Tuple for available period groups.
PERIODS = get_args(Period)

# MONTHS
# ======
# Type alias representing months for the ``frequency`` param.
Month = Literal[
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEPT", "OCT", "NOV", "DEC"
]
# Tuple for available months.
MONTHS = get_args(Month)
# Maps str representation of months to integer for xarray operations.
MONTHS_TO_INT = dict(zip(MONTHS, (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)))

# SEASONS
# =======
# Type alias representing seasons for the ``frequency`` param.
Season = Literal["DJF", "MAM", "JJA", "SON"]
# Tuple for available seasons.
SEASONS = get_args(Season)

# ALL FREQUENCIES
# ===============
# Type alias representing available ``frequency`` param options.
Frequency = Union[Period, Month, Season]
#: Tuple of available frequencies for the ``frequency`` param.
FREQUENCIES = PERIODS + MONTHS + SEASONS

# DATETIME COMPONENTS
# ==================
# Type alias representing xarray DateTime components.
DateTimeComponent = Literal["time.month", "time.season", "time.year"]
# Maps available frequencies to xarray DateTime components for xarray operations.
FREQUENCIES_TO_DATETIME: Dict[str, DateTimeComponent] = {
    **{period: f"time.{period}" for period in PERIODS},  # type: ignore
    **{month: "time.month" for month in MONTHS},
    **{season: "time.season" for season in SEASONS},
}


DJFType = Literal["scd", "sdd"]
DJF_TYPES = get_args(DJFType)


def climatology(
    ds: xr.Dataset,
    frequency: Frequency,
    is_weighted: bool = True,
    djf_type: Optional[DJFType] = "scd",
) -> xr.Dataset:
    """Calculates a Dataset's climatology cycle for all data variables.

    The "time" dimension and any existing bounds variables are preserved in the
    dataset.

    # TODO: Daily climatology

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to calculate climatology cycle.
    frequency : Frequency
        The frequency of time to group by. Available aliases:

        - ``"month"`` for monthly climatologies.
        - ``"year"`` for annual climatologies.
        - ``"season"` for seasonal climatologies.
        - ``"JAN", "FEB", ..., or "DEC"`` for specific month climatology.
          - Averages the month across all seasons.
        - ``"DJF", "MAM", "JJA", or "SON"`` for specific season climatology.
          - Average the season across all years.

        Refer to ``FREQUENCIES`` for a complete list of available options.
    is_weighted : bool, optional
        Perform grouping using weighted averages, by default True.
        Time bounds, leap years, and month lengths are considered.
    djf_type : Optional[DJFType], optional
        Whether DJF climatology contains a seasonally continuous or
        discontinuous December, by default "scd".

        - ``"scd"`` for seasonally continuous December.
        - ``"sdd"`` for seasonally discontinuous December.

        Seasonally continuous December (``"scd"``) refers to continuity between
        December and January. DJF starts on the first year Dec and second year
        Jan/Feb, and ending on the second to last year Dec and last year Jan +
        Feb). Incomplete seasons are dropped (first year Jan/ Feb and last year
        Dec).

        - Example Date Range: Jan/2015 - Dec/2017
        - Start -> Dec/2015, Jan/2016, Feb/2016
        - End -> Dec/2016, Jan/2017, Feb/2017
        - Dropped incomplete seasons -> Jan/2015, Feb/2015, and Dec/2017

        Seasonally discontinuous December (``"sdd"``) refers to discontinuity
        between Feb and Dec. DJF starts on the first year Jan/Feb/Dec, and
        ending on the last year Jan/Feb/Dec. This is the default xarray behavior
        when grouping by season.

        - Example Date Range: Jan/2015 - Dec/2017
        - Start -> Jan/2015, Feb/2015, Dec/2015
        - End -> Jan/2017, Feb/2017, Dec/2017

    Returns
    -------
    xr.Dataset
        Climatology cycle for all data variables for a frequency of time.

    Raises
    ------
    ValueError
        If incorrect ``frequency`` argument is passed.
    KeyError
        If the dataset does not have "time" coordinates.

    Examples
    --------
    Import:

    >>> import xarray as xr
    >>> from xcdat.climatology import climatology, departure
    >>> ds = xr.open_dataset("file_path")

    Get monthly, seasonal, or annual weighted climatology:

    >>> ds_climo_monthly = climatology(ds, "month")
    >>> ds_climo_seasonal = climatology(ds, "season")
    >>> ds_climo_annual = climatology(ds, "year")


    Get monthly, seasonal, or annual unweighted climatology:

    >>> ds_climo_monthly = climatology(ds, "month", is_weighted=False)
    >>> ds_climo_seasonal = climatology(ds, "season", is_weighted=False)
    >>> ds_climo_annual = climatology(ds, "year", is_weighted=False)

    Access attribute for info on climatology operation:

    >>> ds_climo_monthly.calculation_info
    {'type': 'climatology', 'frequency': 'month', 'is_weighted': True}
    >>> ds_climo_monthly.attrs["calculation_info"]
    {'type': 'climatology', 'frequency': 'month', 'is_weighted': True}
    """
    if ds.get("time") is None:
        raise KeyError(
            "This dataset does not have 'time' coordinates. Cannot calculate climatology."
        )

    if frequency not in FREQUENCIES:
        raise ValueError(
            f"Incorrect `frequency` argument. Supported frequencies include: {', '.join(FREQUENCIES)}."
        )

    if frequency != "DJF" and djf_type is not None:
        raise ValueError("`djf_type` specified but `frequency` is not set to 'DJF'")

    if djf_type not in DJF_TYPES:
        raise ValueError(
            f"Incorrect `djf_type` argument. Supported DJF types include: {', '.join(DJF_TYPES)}"
        )

    ds_copy = ds.copy(deep=True)
    ds_climatology = _group_data(
        ds_copy, "climatology", frequency, is_weighted, djf_type
    )
    return ds_climatology


def departure(ds_base: xr.Dataset, ds_climatology: xr.Dataset) -> xr.Dataset:
    """Calculates departures for a given climatology.

    First, the base dataset is grouped using the same frequency and weights (if
    weighted) as the climatology dataset. After grouping, it iterates over the
    dataset to get the difference between non-bounds variables in the base
    dataset and the climatology dataset. Bounds variables are preserved.

    Parameters
    ----------
    ds_base : xr.Dataset
        The base dataset.
    ds_climatology : xr.Dataset
        A climatology dataset.

    Returns
    -------
    xr.Dataset
        The climatology departure between the base and climatology datasets.

    Examples
    --------
    Import:

    >>> import xarray as xr
    >>> from xcdat.climatology import climatology, departure

    Get departure for any time frequency:

    >>> ds = xr.open_dataset("file_path")
    >>> ds_climo_monthly = climatology(ds, "month")
    >>> ds_departure = departure(ds, ds_climo_monthly)

    Access attribute for info on departure operation:

    >>> ds_climo_monthly.calculation_info
    {'type': 'departure', 'frequency': 'month', 'is_weighted': True}
    >>> ds_climo_monthly.attrs["calculation_info"]
    {'type': 'departure', 'frequency': 'month', 'is_weighted': True}
    """
    frequency = ds_climatology.attrs["calculation_info"]["frequency"]
    is_weighted = ds_climatology.attrs["calculation_info"]["is_weighted"]
    djf_type = ds_climatology.attrs["calculation_info"]["djf_type"]

    ds_departure = _group_data(
        ds_base.copy(deep=True), "departure", frequency, is_weighted, djf_type
    )

    for key in ds_departure.data_vars.keys():
        if "_bnds" not in str(key):
            ds_departure[key] = ds_departure[key] - ds_climatology[key]

    return ds_departure


def _group_data(
    ds: xr.Dataset,
    calculation_type: Literal["climatology", "departure"],
    frequency: Frequency,
    is_weighted: bool,
    djf_type: Optional[DJFType],
) -> xr.Dataset:
    """Groups data variables by a frequency to get their averages.

    It iterates over each non-bounds variable and groups them. After grouping,
    attributes are added to the dataset to describe the operation performed.
    This distinguishes datasets that have been manipulated from their original source.

    This "time" dimension and any existing bounds variables are preserved in the
    dataset.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to perform group operation on.
    calculation_type : Literal["climatology", "departure"]
        The calculation type.
    frequency : Frequency
        The frequency of time to group on.
    is_weighted : bool
        Perform grouping using weighted averages.
    djf_type : Optional[DJFType], optional
        Whether DJF climatology contains a seasonally continuous or
        discontinuous December.

    Returns
    -------
    xr.Dataset
        The dataset with grouped data variables.
    """
    datetime_component: DateTimeComponent = FREQUENCIES_TO_DATETIME[frequency]

    if frequency in MONTHS:
        month_int = MONTHS_TO_INT[frequency]
        ds = ds.where(ds[datetime_component] == month_int, drop=True)
    elif frequency in SEASONS:
        ds = ds.where(ds[datetime_component] == frequency, drop=True)

    weights = _calculate_weights(ds, datetime_component) if is_weighted else None

    # DJF scd uses a custom grouping operation with rolling windows
    if frequency == "DJF" and djf_type == "scd":
        ds = _group_djf_scd(ds)
        return ds
    else:
        for key in ds.data_vars.keys():
            if "_bnds" not in str(key):
                data_var = ds[key]

                if is_weighted:
                    data_var *= weights

                ds[key] = data_var.groupby(datetime_component).sum(dim="time")

        ds.attrs.update(
            {
                "calculation_info": {
                    "type": calculation_type,
                    "frequency": frequency,
                    "is_weighted": is_weighted,
                    "djf_type": djf_type,
                },
            }
        )
    return ds


def _group_djf_scd(ds: xr.Dataset) -> xr.Dataset:
    """Subsets DJF climatology dataset with seasonally continuous December.

    Parameters
    ----------
    ds : xr.Dataset
        The DJF climatology dataset.

    Returns
    -------
    xr.Dataset
        The DJF seasonally continuous climatology dataset
    """

    # https://github.com/pydata/xarray/issues/810
    # https://stackoverflow.com/questions/59234745/is-there-any-easy-way-to-compute-seasonal-mean-with-xarray
    # https://stackoverflow.com/questions/64976340/keeping-time-series-while-grouping-by-season-in-xarray

    # however, we loose Jan/ Feb in the first year and Dec in the last
    ds_DJF = ds.rolling(min_periods=3, center=True, time=3)
    # make annual mean
    ds_DJF = ds_DJF.groupby("time.year").mean("time")

    return ds


def _calculate_weights(ds: xr.Dataset, dt_accessor: DateTimeComponent) -> xr.DataArray:
    """Calculates weights for a Dataset based on a frequency of time.

    Time bounds, leap years and number of days for each month are considered
    during grouping.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to calculate weights for.
    dt_accessor : DateTimeAccessor
        The frequency of time to group by in xarray notation ("time.<frequency>").

    Returns
    -------
    xr.DataArray
        The weights based on a frequency of time.
    """
    months_lengths = _get_months_lengths(ds)
    weights: xr.DataArray = (
        months_lengths.groupby(dt_accessor) / months_lengths.groupby(dt_accessor).sum()
    )

    _validate_weights(ds, weights, dt_accessor)

    return weights


def _get_months_lengths(ds: xr.Dataset) -> xr.DataArray:
    """Get the months' lengths based on the time coordinates of a dataset.

    If time bounds exist, it will be used to generate the months' lengths. This
    allows for a robust calculation of weights because different datasets could
    record their time differently (e.g., at beginning/end/middle of each time
    interval).

    If time bounds do not exist, use the time variable (which may be less
    accurate based on the previously described time recording differences).

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to get months' lengths from.

    Returns
    -------
    xr.DataArray
        The months' lengths for the dataset.
    """
    time_bounds = ds.get("time_bnds")

    if time_bounds is not None:
        logging.info("Using existing time bounds to calculate weights.")
        months_lengths = (time_bounds[:, 1] - time_bounds[:, 0]).dt.days
    else:
        # TODO: Generate time bounds if they don't exist?
        logging.info("No time bounds found, using time to calculate weights.")
        months_lengths = ds.time.dt.days_in_month

    return months_lengths


def _validate_weights(
    ds: xr.Dataset, weights: xr.DataArray, dt_accessor: DateTimeComponent
):
    """Validate that the sum of the weights for a dataset equals 1.0.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to validate weights for.
    weights : xr.DataArray
        The weights based on a frequency of time.
    dt_accessor : DateTimeAccessor
        The frequency of time to group by in xarray notation ("time.<frequency>").
    """
    # 12 groups for 12 months in a year
    # 4 groups for 4 seasons in year
    # 1 group for a single month or season
    frequency_groups = len(ds.time.groupby(dt_accessor).count())

    expected_sum = np.ones(frequency_groups)
    actual_sum = weights.groupby(dt_accessor).sum().values

    np.testing.assert_allclose(actual_sum, expected_sum)
