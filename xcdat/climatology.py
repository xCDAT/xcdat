"""Functions related to calculating climatology cycles and departures."""

from typing import Union, get_args

import numpy as np
import xarray as xr
from typing_extensions import Literal

from xcdat.log import logging

# GROUPS
# ======
# Type alias representing available groups for the ``period`` param.
PeriodGroup = Literal["ANNUALCYCLE", "SEASONALCYCLE", "YEAR"]
# Tuple for available period groups.
PERIOD_GROUPS = get_args(PeriodGroup)

# MONTHS
# ======
# Type alias representing available months for the ``period`` param.
Month = Literal[
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEPT", "OCT", "NOV", "DEC"
]
# Tuple for available months.
MONTHS = get_args(Month)
# Maps str representation of months to integer for xarray operations.
MONTHS_TO_INT = dict(zip(MONTHS, (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)))

# SEASONS
# =======
# Type alias representing available seasons for the ``period`` param.
Season = Literal["DJF", "MAM", "JJA", "SON"]
# Tuple for available seasons.
SEASONS = get_args(Season)

# ALL PERIODS
# ===========
# Type alias representing all of the available ``period`` param options.
Period = Union[PeriodGroup, Month, Season]
#: Tuple of all available options for the ``period`` param.
PERIODS = PERIOD_GROUPS + MONTHS + SEASONS

# DATETIME ACCESSORS
# ==================
# Type alias representing xarray DateTime accessors.
# http://xarray.pydata.org/en/stable/generated/xarray.core.accessor_dt.DatetimeAccessor.html
DateTimeAccessor = Literal["time.month", "time.season", "time.year"]
# Tuple for available xarray DateTime accessors.
DATETIME_ACCESSORS = get_args(DateTimeAccessor)
# Maps period options to xarray DateTime accessors for xarray operations.
PERIODS_TO_DATETIME = {
    **dict(zip(PERIOD_GROUPS, DATETIME_ACCESSORS)),
    **{month: "time.month" for month in MONTHS},
    **{season: "time.season" for season in SEASONS},
}


def climatology(ds: xr.Dataset, period: Period, is_weighted: bool = True) -> xr.Dataset:
    """Calculates a Dataset's climatology cycle for all data variables.

    The "time" dimension and any existing bounds variables are preserved in the
    dataset.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to calculate climatology cycle.
    period : Period
        The period of time to group by.
        Refer to ``PERIODS`` for list of available options.
    is_weighted : bool, optional
        Perform grouping using weighted averages, by default True.
        Time bounds, leap years, and month lengths are considered.

    Returns
    -------
    xr.Dataset
        Climatology cycle for all data variables for a period of time.

    Raises
    ------
    ValueError
        If incorrect ``period`` argument is passed.
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
    {'type': 'climatology', 'period': 'month', 'is_weighted': True}
    >>> ds_climo_monthly.attrs["calculation_info"]
    {'type': 'climatology', 'period': 'month', 'is_weighted': True}
    """
    if period not in PERIODS:
        raise ValueError(
            f"Incorrect `period` argument passed. Supported periods include: {', '.join(PERIODS)}."
        )

    if ds.get("time") is None:
        raise KeyError(
            "This dataset does not have 'time' coordinates. Cannot calculate climatology."
        )

    ds_copy = ds.copy(deep=True)
    ds_climatology = _group_data(ds_copy, "climatology", period, is_weighted)
    return ds_climatology


def departure(ds_base: xr.Dataset, ds_climatology: xr.Dataset) -> xr.Dataset:
    """Calculates departures for a given climatology.

    First, the base dataset is grouped using the same period and weights (if
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

    Get departure for any time period:

    >>> ds = xr.open_dataset("file_path")
    >>> ds_climo_monthly = climatology(ds, "month")
    >>> ds_departure = departure(ds, ds_climo_monthly)

    Access attribute for info on departure operation:

    >>> ds_climo_monthly.calculation_info
    {'type': 'departure', 'period': 'month', 'is_weighted': True}
    >>> ds_climo_monthly.attrs["calculation_info"]
    {'type': 'departure', 'period': 'month', 'is_weighted': True}
    """
    period = ds_climatology.attrs["calculation_info"]["period"]
    is_weighted = ds_climatology.attrs["calculation_info"]["is_weighted"]
    ds_departure = _group_data(
        ds_base.copy(deep=True), "departure", period, is_weighted
    )

    for key in ds_departure.data_vars.keys():
        if "_bnds" not in str(key):
            ds_departure[key] = ds_departure[key] - ds_climatology[key]

    return ds_departure


def _group_data(
    ds: xr.Dataset,
    calculation_type: Literal["climatology", "departure"],
    period: Period,
    is_weighted: bool,
) -> xr.Dataset:
    """Groups data variables by a period to get their averages.

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
    period : Period
        The period of time to group on.
    is_weighted : bool
        Perform grouping using weighted averages.

    Returns
    -------
    xr.Dataset
        The dataset with grouped data variables.
    """
    # Determine the xarray DateTime accessor for grouping operation and subset
    # if the period is a single month or season.
    dt_accessor: DateTimeAccessor = PERIODS_TO_DATETIME[period]

    if period in MONTHS + SEASONS:
        ds = _subset_data(ds, period, dt_accessor)

    weights = _calculate_weights(ds, dt_accessor) if is_weighted else None
    for key in ds.data_vars.keys():
        if "_bnds" not in str(key):
            data_var = ds[key]

            if is_weighted:
                data_var *= weights

            ds[key] = data_var.groupby(dt_accessor).sum(dim="time")

    ds.attrs.update(
        {
            "calculation_info": {
                "type": calculation_type,
                "period": period,
                "is_weighted": is_weighted,
            },
        }
    )
    return ds


def _subset_data(
    ds: xr.Dataset, period: Period, dt_accessor: DateTimeAccessor
) -> xr.Dataset:
    """Subsets a dataset for a single month or season.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to subset.
    period : Period
        The single month or season to subset with.
    dt_accessor : DateTimeAccessor
        The DateTime accessor to subset with.

    Returns
    -------
    xr.Dataset
        The subsetted dataset and associated DateTime accessor.
    """
    if period in MONTHS:
        month_int = MONTHS_TO_INT[period]
        ds = ds.where(ds[dt_accessor] == month_int, drop=True)
    elif period in SEASONS:
        ds = ds.where(ds[dt_accessor] == period, drop=True)

    return ds


def _calculate_weights(ds: xr.Dataset, dt_accessor: DateTimeAccessor) -> xr.DataArray:
    """Calculates weights for a Dataset based on a period of time.

    Time bounds, leap years and number of days for each month are considered
    during grouping.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to calculate weights for.
    dt_accessor : DateTimeAccessor
        The period of time to group by in xarray notation ("time.<period>").

    Returns
    -------
    xr.DataArray
        The weights based on a period of time.
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
    ds: xr.Dataset, weights: xr.DataArray, dt_accessor: DateTimeAccessor
):
    """Validate that the sum of the weights for a dataset equals 1.0.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to validate weights for.
    weights : xr.DataArray
        The weights based on a period of time.
    dt_accessor : DateTimeAccessor
        The period of time to group by in xarray notation ("time.<period>").
    """
    # 12 groups for 12 months in a year
    # 4 groups for 4 seasons in year
    # 1 group for a single month or season
    period_groups = len(ds.time.groupby(dt_accessor).count())

    expected_sum = np.ones(period_groups)
    actual_sum = weights.groupby(dt_accessor).sum().values

    np.testing.assert_allclose(actual_sum, expected_sum)
