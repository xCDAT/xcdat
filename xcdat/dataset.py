"""Dataset module for functions related to an xarray.Dataset."""
import pathlib
from datetime import datetime
from functools import partial
from glob import glob
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import xarray as xr
from dateutil import parser
from dateutil import relativedelta as rd
from xarray.coding.cftime_offsets import get_date_type
from xarray.coding.times import convert_times

from xcdat import bounds  # noqa: F401
from xcdat.axis import center_times as center_times_func
from xcdat.axis import get_axis_coord, swap_lon_axis
from xcdat.logger import setup_custom_logger

logger = setup_custom_logger(__name__)

#: List of non-CF compliant time units.
NON_CF_TIME_UNITS: List[str] = ["months", "years"]


def open_dataset(
    path: str,
    data_var: Optional[str] = None,
    add_bounds: bool = True,
    decode_times: bool = True,
    center_times: bool = False,
    lon_orient: Optional[Tuple[float, float]] = None,
    **kwargs: Dict[str, Any],
) -> xr.Dataset:
    """Wraps ``xarray.open_dataset()`` with post-processing options.

    Parameters
    ----------
    path : str
        Path to Dataset.
    data_var: Optional[str], optional
        The key of the non-bounds data variable to keep in the Dataset,
        alongside any existing bounds data variables, by default None.
    add_bounds: bool, optional
        If True, add bounds for supported axes (X, Y, T) if they are missing in
        the Dataset, by default True. Bounds are required for many xCDAT
        features.
    decode_times: bool, optional
        If True, attempt to decode times encoded in the standard NetCDF
        datetime format into datetime objects. Otherwise, leave them encoded
        as numbers. This keyword may not be supported by all the backends,
        by default True.
    center_times: bool, optional
        If True, center time coordinates using the midpoint between its upper
        and lower bounds. Otherwise, use the provided time coordinates, by
        default False.
    lon_orient: Optional[Tuple[float, float]], optional
        The orientation to use for the Dataset's longitude axis (if it exists),
        by default None.

        Supported options:

          * None:  use the current orientation (if the longitude axis exists)
          * (-180, 180): represents [-180, 180) in math notation
          * (0, 360): represents [0, 360) in math notation

    kwargs : Dict[str, Any]
        Additional arguments passed on to ``xarray.open_dataset``. Refer to the
        [1]_ xarray docs for accepted keyword arguments.

    Returns
    -------
    xr.Dataset
        Dataset after applying operations.

    Notes
    -----
    ``xarray.open_dataset`` opens the file with read-only access. When you
    modify values of a Dataset, even one linked to files on disk, only the
    in-memory copy you are manipulating in xarray is modified: the original file
    on disk is never touched.

    References
    ----------

    .. [1] https://xarray.pydata.org/en/stable/generated/xarray.open_dataset.html
    """
    if decode_times:
        cf_compliant_time: Optional[bool] = _has_cf_compliant_time(path)
        if cf_compliant_time is False:
            # XCDAT handles decoding time values with non-CF units.
            ds = xr.open_dataset(path, decode_times=False, **kwargs)
            # attempt to decode non-cf-compliant time axis
            ds = decode_non_cf_time(ds)
        else:
            ds = xr.open_dataset(path, decode_times=True, **kwargs)
    else:
        ds = xr.open_dataset(path, decode_times=False, **kwargs)

    ds = _postprocess_dataset(ds, data_var, center_times, add_bounds, lon_orient)

    return ds


def open_mfdataset(
    paths: Union[
        str,
        pathlib.Path,
        List[str],
        List[pathlib.Path],
        List[List[str]],
        List[List[pathlib.Path]],
    ],
    data_var: Optional[str] = None,
    add_bounds: bool = True,
    decode_times: bool = True,
    center_times: bool = False,
    lon_orient: Optional[Tuple[float, float]] = None,
    data_vars: Union[Literal["minimal", "different", "all"], List[str]] = "minimal",
    preprocess: Optional[Callable] = None,
    **kwargs: Dict[str, Any],
) -> xr.Dataset:
    """Wraps ``xarray.open_mfdataset()`` with post-processing options.

    Parameters
    ----------
    path : Union[str, pathlib.Path, List[str], List[pathlib.Path], \
         List[List[str]], List[List[pathlib.Path]]]
        Either a string glob in the form ``"path/to/my/files/*.nc"`` or an
        explicit list of files to open. Paths can be given as strings or as
        pathlib Paths. If concatenation along more than one dimension is desired,
        then ``paths`` must be a nested list-of-lists (see ``combine_nested``
        for details). (A string glob will be expanded to a 1-dimensional list.)
    add_bounds: bool, optional
        If True, add bounds for supported axes (X, Y, T) if they are missing in
        the Dataset, by default True. Bounds are required for many xCDAT
        features.
    data_var: Optional[str], optional
        The key of the data variable to keep in the Dataset, by default None.
    decode_times: bool, optional
        If True, decode times encoded in the standard NetCDF datetime format
        into datetime objects. Otherwise, leave them encoded as numbers.
        This keyword may not be supported by all the backends, by default True.
    center_times: bool, optional
        If True, center time coordinates using the midpoint between its upper
        and lower bounds. Otherwise, use the provided time coordinates, by
        default False.
    lon_orient: Optional[Tuple[float, float]], optional
        The orientation to use for the Dataset's longitude axis (if it exists),
        by default None.

        Supported options:

          * None:  use the current orientation (if the longitude axis exists)
          * (-180, 180): represents [-180, 180) in math notation
          * (0, 360): represents [0, 360) in math notation

    data_vars: Union[Literal["minimal", "different", "all"], List[str]], optional
        These data variables will be concatenated together:
          * "minimal": Only data variables in which the dimension already
            appears are included, the default value.
          * "different": Data variables which are not equal (ignoring
            attributes) across all datasets are also concatenated (as well as
            all for which dimension already appears). Beware: this option may
            load the data payload of data variables into memory if they are not
            already loaded.
          * "all": All data variables will be concatenated.
          * list of str: The listed data variables will be concatenated, in
            addition to the "minimal" data variables.

        The ``data_vars`` kwarg defaults to ``"minimal"``, which concatenates
        data variables in a manner where only data variables in which the
        dimension already appears are included. For example, the time dimension
        will not be concatenated to the dimensions of non-time data variables
        such as "lat_bnds" or "lon_bnds". `data_vars="minimal"` is required for
        some XCDAT functions, including spatial averaging where a reduction is
        performed using the lat/lon bounds.

    preprocess : Optional[Callable], optional
        If provided, call this function on each dataset prior to concatenation.
        You can find the file-name from which each dataset was loaded in
        ``ds.encoding["source"]``.
    kwargs : Dict[str, Any]
        Additional arguments passed on to ``xarray.open_mfdataset``. Refer to
        the [2]_ xarray docs for accepted keyword arguments.

    Returns
    -------
    xr.Dataset
        The Dataset.

    Notes
    -----
    ``xarray.open_mfdataset`` opens the file with read-only access. When you
    modify values of a Dataset, even one linked to files on disk, only the
    in-memory copy you are manipulating in xarray is modified: the original file
    on disk is never touched.

    References
    ----------

    .. [2] https://xarray.pydata.org/en/stable/generated/xarray.open_mfdataset.html
    """
    if decode_times:
        cf_compliant_time: Optional[bool] = _has_cf_compliant_time(paths)
        # XCDAT handles decoding time values with non-CF units using the
        # preprocess kwarg.
        if cf_compliant_time is False:
            decode_times = False
            preprocess = partial(_preprocess_non_cf_dataset, callable=preprocess)

    ds = xr.open_mfdataset(
        paths,
        decode_times=decode_times,
        data_vars=data_vars,
        preprocess=preprocess,
        **kwargs,
    )
    ds = _postprocess_dataset(ds, data_var, center_times, add_bounds, lon_orient)

    return ds


def decode_non_cf_time(dataset: xr.Dataset) -> xr.Dataset:
    """Decodes time coordinates and time bounds with non-CF compliant units.

    By default, ``xarray`` only supports decoding time with CF compliant units
    [3]_. This function enables decoding time with non-CF compliant units.

    The time coordinates must have a "calendar" attribute set to a CF calendar
    type supported by ``cftime`` ("noleap", "360_day", "365_day", "366_day",
    "gregorian", "proleptic_gregorian", "julian", "all_leap", or "standard")
    and a "units" attribute set to a supported format ("months since ..." or
    "years since ...").

    The logic for this function:

    1. Extract units and reference date strings from the "units" attribute.

       * For example with "months since 1800-01-01", the units are "months" and
         reference date is "1800-01-01".

    2. Using the reference date, create a reference ``datetime`` object.
    3. Starting from the reference ``datetime`` object, use the numerically
       encoded time coordinate values (each representing an offset) to create an
       array of ``cftime`` objects based on the calendar type.
    4. Using the array of ``cftime`` objects, create a new xr.DataArray
       of time coordinates to replace the numerically encoded ones.
    5. If it exists, create a time bounds DataArray using steps 3 and 4.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset with numerically encoded time coordinates and time bounds (if
        they exist). If the time coordinates cannot be decoded then the original
        dataset is returned.

    Returns
    -------
    xr.Dataset
        Dataset with decoded time coordinates and time bounds (if they exist) as
        ``cftime`` objects.

    Notes
    -----
    Time coordinates are represented by ``cftime.datetime`` objects because
    it is not restricted by the ``pandas.Timestamp`` range (years 1678 through
    2262). Refer to [4]_ and [5]_ for more information on this limitation.

    References
    -----
    .. [3] https://cfconventions.org/cf-conventions/cf-conventions.html#time-coordinate
    .. [4] https://docs.xarray.dev/en/stable/user-guide/weather-climate.html#non-standard-calendars-and-dates-outside-the-timestamp-valid-range
    .. [5] https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timestamp-limitations

    Examples
    --------

    Decode the time coordinates with non-CF units in a Dataset:

    >>> from xcdat.dataset import decode_non_cf_time
    >>>
    >>> ds.time
    <xarray.DataArray 'time' (time: 3)>
    array([0, 1, 2])
    Coordinates:
    * time     (time) int64 0 1 2
    Attributes:
        units:          years since 2000-01-01
        bounds:         time_bnds
        axis:           T
        long_name:      time
        standard_name:  time
        calendar:       noleap
    >>>
    >>> ds_decoded = decode_non_cf_time(ds)
    >>> ds_decoded.time
    <xarray.DataArray 'time' (time: 3)>
    array([cftime.DatetimeNoLeap(1850, 1, 1, 0, 0, 0, 0, has_year_zero=True),
           cftime.DatetimeNoLeap(1850, 1, 1, 0, 0, 0, 0, has_year_zero=True),
           cftime.DatetimeNoLeap(1850, 1, 1, 0, 0, 0, 0, has_year_zero=True)],
           dtype='object')
    Coordinates:
    * time     (time) datetime64[ns] 2000-01-01 2001-01-01 2002-01-01
    Attributes:
        units:          years since 2000-01-01
        bounds:         time_bnds
        axis:           T
        long_name:      time
        standard_name:  time
        calendar:       noleap

    View time encoding information:

    >>> ds_decoded.time.encoding
    {'source': None,
     'dtype': dtype('int64'),
     'original_shape': (3,),
     'units': 'years since 2000-01-01',
     'calendar': 'noleap'}
    """
    ds = dataset.copy()
    time = get_axis_coord(ds, "T")

    try:
        calendar = time.attrs["calendar"]
    except KeyError:
        logger.warning(
            "This dataset's time coordinates do not have a 'calendar' attribute set, "
            "so time coordinates could not be decoded. Set the 'calendar' attribute "
            "and try decoding the time coordinates again."
        )
        return ds

    try:
        units_attr = time.attrs["units"]
    except KeyError:
        logger.warning(
            "This dataset's time coordinates do not have a 'units' attribute set, "
            "so the time coordinates could not be decoded. Set the 'units' attribute"
            "and try decoding the time coordinates again."
        )
        return ds

    try:
        units, ref_date = _split_time_units_attr(units_attr)
    except ValueError:
        logger.warning(
            f"This dataset's 'units' attribute ('{units_attr}') is not in a "
            "supported format ('months since...' or 'years since...'), so the time "
            "coordinates could not be decoded."
        )
        return ds

    data = _get_cftime_coords(ref_date, time.values, calendar, units)
    decoded_time = xr.DataArray(
        name=time.name,
        data=data,
        dims=time.dims,
        coords={time.name: data},
        attrs=time.attrs,
    )
    decoded_time.encoding = {
        "source": ds.encoding.get("source", "None"),
        "dtype": time.dtype,
        "original_shape": time.shape,
        "units": units_attr,
        "calendar": calendar,
    }
    ds = ds.assign_coords({time.name: decoded_time})

    try:
        time_bounds = ds.bounds.get_bounds("T")
    except KeyError:
        time_bounds = None

    if time_bounds is not None:
        lowers = _get_cftime_coords(ref_date, time_bounds.values[:, 0], calendar, units)
        uppers = _get_cftime_coords(ref_date, time_bounds.values[:, 1], calendar, units)
        data_bounds = np.vstack((lowers, uppers)).T

        decoded_time_bnds = xr.DataArray(
            name=time_bounds.name,
            data=data_bounds,
            dims=time_bounds.dims,
            coords=time_bounds.coords,
            attrs=time_bounds.attrs,
        )
        decoded_time_bnds.coords[time.name] = decoded_time
        decoded_time_bnds.encoding = time_bounds.encoding
        ds = ds.assign({time_bounds.name: decoded_time_bnds})

    return ds


def _has_cf_compliant_time(
    path: Union[
        str,
        pathlib.Path,
        List[str],
        List[pathlib.Path],
        List[List[str]],
        List[List[pathlib.Path]],
    ]
) -> Optional[bool]:
    """Checks if a dataset has time coordinates with CF compliant units.

    If the dataset does not contain a time dimension, None is returned.
    Otherwise, the units attribute is extracted from the time coordinates to
    determine whether it is CF or non-CF compliant.

    Parameters
    ----------
    path : Union[str, pathlib.Path, List[str], List[pathlib.Path], \
         List[List[str]], List[List[pathlib.Path]]]
        Either a file (``"file.nc"``), a string glob in the form
        ``"path/to/my/files/*.nc"``, or an explicit list of files to open.
        Paths can be given as strings or as pathlib Paths. If concatenation
        along more than one dimension is desired, then ``paths`` must be a
        nested list-of-lists (see ``combine_nested`` for details). (A string
        glob will be expanded to a 1-dimensional list.)

    Returns
    -------
    Optional[bool]
        None if time dimension does not exist, True if CF compliant, or False if
        non-CF compliant.

    Notes
    -----
    This function only checks one file for multi-file datasets to optimize
    performance because it is slower to combine all files then check for CF
    compliance.
    """
    first_file: Optional[Union[pathlib.Path, str]] = None

    if isinstance(path, str) and "*" in path:
        first_file = glob(path)[0]
    elif isinstance(path, str) or isinstance(path, pathlib.Path):
        first_file = path
    elif isinstance(path, list):
        if any(isinstance(sublist, list) for sublist in path):
            first_file = path[0][0]  # type: ignore
        else:
            first_file = path[0]  # type: ignore

    ds = xr.open_dataset(first_file, decode_times=False)

    if ds.cf.dims.get("T") is None:
        return None

    time = ds.cf["T"]

    # If the time units attr cannot be split, it is not cf_compliant.
    try:
        units = _split_time_units_attr(time.attrs.get("units"))[0]
    except ValueError:
        return False

    cf_compliant = units not in NON_CF_TIME_UNITS

    return cf_compliant


def _postprocess_dataset(
    dataset: xr.Dataset,
    data_var: Optional[str] = None,
    center_times: bool = False,
    add_bounds: bool = True,
    lon_orient: Optional[Tuple[float, float]] = None,
) -> xr.Dataset:
    """Post-processes a Dataset object.

    Parameters
    ----------
    dataset : xr.Dataset
        The dataset.
    data_var: Optional[str], optional
        The key of the data variable to keep in the Dataset, by default None.
    center_times: bool, optional
        If True, center time coordinates using the midpoint between its upper
        and lower bounds. Otherwise, use the provided time coordinates, by
        default False.
    add_bounds: bool, optional
        If True, add bounds for supported axes (X, Y, T) if they are missing in
        the Dataset, by default False.
    lon_orient: Optional[Tuple[float, float]], optional
        The orientation to use for the Dataset's longitude axis (if it exists),
        by default None.

        Supported options:

          * None:  use the current orientation (if the longitude axis exists)
          * (-180, 180): represents [-180, 180) in math notation
          * (0, 360): represents [0, 360) in math notation

    Returns
    -------
    xr.Dataset
        The Dataset.

    Raises
    ------
    ValueError
        If ``center_times==True`` but there are no time coordinates.
    ValueError
        If ``lon_orient is not None`` but there are no longitude coordinates.
    """
    if data_var is not None:
        dataset = _keep_single_var(dataset, data_var)

    if center_times:
        if dataset.cf.dims.get("T") is not None:
            dataset = center_times_func(dataset)
        else:
            raise ValueError("This dataset does not have a time coordinates to center.")

    if add_bounds:
        dataset = dataset.bounds.add_missing_bounds()

    if lon_orient is not None:
        if dataset.cf.dims.get("X") is not None:
            dataset = swap_lon_axis(dataset, to=lon_orient, sort_ascending=True)
        else:
            raise ValueError(
                "This dataset does not have longitude coordinates to reorient."
            )

    return dataset


def _keep_single_var(dataset: xr.Dataset, key: str) -> xr.Dataset:
    """Keeps a single non-bounds data variable in the Dataset.

    This function checks if the ``data_var`` key exists in the Dataset and
    it is not related to bounds. If those checks pass, it will subset the
    Dataset to retain that non-bounds ``data_var`` and all bounds data vars.

    Parameters
    ----------
    dataset : xr.Dataset
        The Dataset.
    key: str
        The key of the non-bounds data variable to keep in the Dataset.

    Returns
    -------
    xr.Dataset
        The Dataset.

    Raises
    ------
    ValueError
        If the dataset only contains bounds data variables.
    ValueError
        If the specified key does not exist in the dataset.
    ValueError
        If the specified key matches a bounds data variable.
    """
    all_vars = dataset.data_vars.keys()
    bounds_vars = dataset.bounds.keys
    non_bounds_vars = sorted(list(set(all_vars) ^ set(bounds_vars)))

    if len(non_bounds_vars) == 0:
        raise ValueError("This dataset only contains bounds data variables.")

    if key not in all_vars:
        raise ValueError(f"The data variable '{key}' does not exist in the dataset.")

    if key in bounds_vars:
        raise ValueError("Please specify a non-bounds data variable.")

    return dataset[[key] + bounds_vars]


def _get_data_var(dataset: xr.Dataset, key: str) -> xr.DataArray:
    """Get a data variable in the Dataset by key.

    Parameters
    ----------
    dataset : xr.Dataset
        The Dataset.
    key : str
        The data variable key.

    Returns
    -------
    xr.DataArray
        The data variable.

    Raises
    ------
    KeyError
        If the data variable does not exist in the Dataset.
    """
    dv = dataset.get(key, None)

    if dv is None:
        raise KeyError(f"The data variable '{key}' does not exist in the Dataset.")

    return dv.copy()


def _preprocess_non_cf_dataset(
    ds: xr.Dataset, callable: Optional[Callable] = None
) -> xr.Dataset:
    """Preprocessing for each non-CF compliant dataset in ``open_mfdataset()``.

    This function accepts a user specified preprocess function, which is
    executed before additional internal preprocessing functions.

    One call is performed to ``decode_non_cf_time()`` for decoding each
    dataset's time coordinates and time bounds (if they exist) with non-CF
    compliant units. By default, if ``decode_times=False`` is passed, xarray
    will concatenate time values using the first dataset's ``units`` attribute.
    This is an issue for cases where the numerically encoded time values are the
    same and the ``units`` attribute differs between datasets.

    For example, two files have the same time values, but the units of the first
    file is "months since 2000-01-01" and the second is "months since
    2001-01-01". Since the first dataset's units are used in xarray for
    concatenating datasets, the time values corresponding to the second file
    will be dropped since they appear to be the same as the first file.

    Calling ``decode_non_cf_time()`` on each dataset individually before
    concatenating solves the aforementioned issue.

    Parameters
    ----------
    ds : xr.Dataset
        The Dataset.
    callable : Optional[Callable], optional
        A user specified optional callable function for preprocessing.

    Returns
    -------
    xr.Dataset
        The preprocessed Dataset.
    """
    ds_new = ds.copy()

    if callable:
        ds_new = callable(ds)

    # Attempt to decode non-cf-compliant time axis.
    ds_new = decode_non_cf_time(ds_new)

    return ds_new


def _split_time_units_attr(units_attr: str) -> Tuple[str, str]:
    """Splits the time coordinates' units attr into units and reference date.

    Parameters
    ----------
    units_attr : str
        The units attribute (e.g., "months since 1800-01-01").

    Returns
    -------
    Tuple[str, str]
        The units (e.g, "months") and the reference date (e.g., "1800-01-01").

    Raises
    ------
    KeyError
        If the time units attribute was not found.

    ValueError
        If the time units attribute is not of the form `X since Y`.
    """
    if "since" in units_attr:
        units, reference_date = units_attr.split(" since ")
    else:
        raise ValueError(
            "This dataset does not have time coordinates of the form 'X since Y'."
        )

    return units, reference_date


def _get_cftime_coords(
    ref_date: str, offsets: np.ndarray, calendar: str, units: str
) -> np.ndarray:
    """Get an array of `cftime` coordinates starting from a reference date.

    Parameters
    ----------
    ref_date : str
        The starting reference date.
    offsets : np.ndarray
        An array of numerically encoded time offsets from the reference date.
    calendar : str
        The CF calendar type supported by ``cftime``. This includes "noleap",
        "360_day", "365_day", "366_day", "gregorian", "proleptic_gregorian",
        "julian", "all_leap", and "standard".
    units : str
        The time units.

    Returns
    -------
    np.ndarray
        An array of `cftime` coordinates.
    """
    # Starting from the reference date, create an array of `datetime` objects
    # by adding each offset (a numerically encoded value) to the reference date.
    # The `parse.parse` default is set to datetime(2000, 1, 1), with each
    # component being a placeholder if the value does not exist. For example, 1
    # and 1 are placeholders for month and day if those values don't exist.
    ref_datetime: datetime = parser.parse(ref_date, default=datetime(2000, 1, 1))
    offsets = np.array(
        [ref_datetime + rd.relativedelta(**{units: offset}) for offset in offsets],
        dtype="object",
    )

    # Convert the array of `datetime` objects into `cftime` objects based on
    # the calendar type.
    date_type = get_date_type(calendar)
    coords = convert_times(offsets, date_type=date_type)

    return coords
