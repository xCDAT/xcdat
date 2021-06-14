"""Functions related to calculating climatology cycles and departures."""
from typing import Tuple, get_args

import numpy as np
import xarray as xr
from typing_extensions import Literal

from xcdat.log import logging

# Available climatology periods
Period = Literal["month", "season", "year"]
# Available climatology periods in xarray notation
TimePeriod = Literal["time.month", "time.season", "time.year"]

PERIODS: Tuple[Period, ...] = get_args(Period)


def climatology(ds: xr.Dataset, period: Period, is_weighted: bool = True) -> xr.Dataset:
    """Calculates a Dataset's climatology cycle for all data variables.

    The "time" dimension is preserved for reference, which would otherwise be
    replaced by the period.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to calculate climatology cycle.
    period : Period
        The period of time to group by.
    is_weighted : bool, optional
        Perform grouping using weighted averages (time bounds, leap years and
        number of days for each month are considered), by default True.

    Returns
    -------
    xr.Dataset
        Climatology cycle for all data variables for a period of time.

    Raises
    ------
    ValueError
        If incorrect ``period`` argument is passed.
    KeyError
        If the dataset does not have a "time" coordinates.

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
            "Incorrect period argument passed. Period must be month, season, or year."
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
    weighted) as the climatology dataset. After grouping, it iterates over each
    non-bounds variable and get the difference between the base dataset and the
    climatology dataset.

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
    """Groups data variables to get their averages over a time period.

    It iterates over each non-bounds variable and groups them. This preserves
    bounds variables if they exist, and the time dimension. Once grouping is
    complete, attributes are added to the dataset to describe the operation
    performed on it. This distinguishes datasets that have been manipulated
    from their original source.

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
    time_period: TimePeriod = f"time.{period}"  # type:ignore
    weights = _calculate_weights(ds, time_period) if is_weighted else None

    for key in ds.data_vars.keys():
        if "_bnds" not in str(key):
            data_var = ds[key]

            if is_weighted:
                data_var *= weights

            ds[key] = data_var.groupby(time_period).sum(dim="time")

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


def _calculate_weights(ds: xr.Dataset, time_period: TimePeriod) -> xr.DataArray:
    """Calculates weights for a Dataset based on a period of time.

    Time bounds, leap years and number of days for each month are considered
    during grouping.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to calculate weights for.
    time_period : TimePeriod
        The period of time to group by in xarray notation ("time.<period>").

    Returns
    -------
    xr.DataArray
        The weights based on a period of time.
    """
    months_lengths = _get_months_lengths(ds)
    weights: xr.DataArray = (
        months_lengths.groupby(time_period) / months_lengths.groupby(time_period).sum()
    )
    _validate_weights(ds, weights, time_period)

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


def _validate_weights(ds: xr.Dataset, weights: xr.DataArray, time_period: TimePeriod):
    """Validate that the sum of the weights for a dataset equals 1.0.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to validate weights for.
    weights : xr.DataArray
        The weights based on a period of time.
    time_period : TimePeriod
        The period of time to group by in xarray notation ("time.<period>").
    """
    expected_count = {
        "time.month": 12,
        "time.season": 4,
        "time.year": np.unique(ds.time.dt.year.data).size,
    }

    expected = np.ones(expected_count[time_period])
    np.testing.assert_allclose(weights.groupby(time_period).sum().values, expected)
