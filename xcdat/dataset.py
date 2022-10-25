"""Dataset module for functions related to an xarray.Dataset."""
import pathlib
from datetime import datetime
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import cftime
import numpy as np
import xarray as xr
from dateutil import parser
from dateutil import relativedelta as rd
from xarray.coding.cftime_offsets import get_date_type
from xarray.coding.times import convert_times

from xcdat import bounds as bounds_accessor  # noqa: F401
from xcdat.axis import _get_all_coord_keys
from xcdat.axis import center_times as center_times_func
from xcdat.axis import swap_lon_axis
from xcdat.logger import setup_custom_logger

logger = setup_custom_logger(__name__)

#: List of non-CF compliant time units.
NON_CF_TIME_UNITS: List[str] = ["months", "years"]

# Type annotation for the `paths` arg.
Paths = Union[
    str,
    pathlib.Path,
    List[str],
    List[pathlib.Path],
    List[List[str]],
    List[List[pathlib.Path]],
]


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
        datetime format into cftime.datetime objects. Otherwise, leave them
        encoded as numbers. This keyword may not be supported by all the
        backends, by default True.
    center_times: bool, optional
        If True, attempt to center time coordinates using the midpoint between
        its upper and lower bounds. Otherwise, use the provided time
        coordinates, by default False.
    lon_orient: Optional[Tuple[float, float]], optional
        The orientation to use for the Dataset's longitude axis (if it exists).
        Either `(-180, 180)` or `(0, 360)`, by default None.

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
    ds = xr.open_dataset(path, decode_times=False, **kwargs)  # type: ignore
    ds = _postprocess_dataset(
        ds, data_var, decode_times, center_times, add_bounds, lon_orient
    )

    return ds


def open_mfdataset(
    paths: Paths,
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
        If True, attempt to decode times encoded in the standard NetCDF
        datetime format into cftime.datetime objects. Otherwise, leave them
        encoded as numbers. This keyword may not be supported by all the
        backends, by default True.
    center_times: bool, optional
        If True, attempt to center time coordinates using the midpoint between
        its upper and lower bounds. Otherwise, use the provided time
        coordinates, by default False.
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
    ds = xr.open_mfdataset(
        paths,
        decode_times=False,
        data_vars=data_vars,
        preprocess=preprocess,
        **kwargs,  # type: ignore
    )

    ds = _postprocess_dataset(
        ds, data_var, decode_times, center_times, add_bounds, lon_orient
    )

    return ds


def decode_time(dataset: xr.Dataset) -> xr.Dataset:  # noqa: C901
    """Decodes CF and non-CF time coordinates and time bounds using ``cftime``.

    By default, ``xarray`` only supports decoding time with CF compliant units
    [3]_. This function enables also decoding time with non-CF compliant units.
    It skips decoding time coordinates that have already been decoded as
    "datetime64[ns]" or `cftime.datetime`.

    For time coordinates to be decodable, they must have a "calendar" attribute
    set to a CF calendar type supported by ``cftime``. CF calendar types
    include "noleap", "360_day", "365_day", "366_day", "gregorian",
    "proleptic_gregorian", "julian", "all_leap", or "standard". They must also
    have a "units" attribute set to a format supported by xcdat ("months since
    ..." or "years since ...").

    How this function decodes time coordinates:

    1. Extract units and reference date strings from the "units" attribute.

       * For example with "months since 1800-01-01", the units are "months" and
         reference date is "1800-01-01".

    2. Using the reference date, create a reference ``datetime`` object.
    3. Starting from the reference ``datetime`` object, use the integer offset
       values since the reference date to create an array of ``cftime`` objects
       based on the calendar type.
    4. Create a new xr.DataArray of decoded time coordinates to replace the
       numerically encoded ones.
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

    >>> from xcdat.dataset import decode_time
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
    >>> ds_decoded = decode_time(ds)
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
    coord_keys = _get_all_coord_keys(ds, "T")

    for key in coord_keys:
        coord = ds[key].copy()

        if not _is_decoded(coord.values):
            attrs = coord.attrs

            # NOTE: The "calendar" and "units" attributes are stored in the `.attrs`
            # dict when `decode_times=False`, unlike `decode_times=True` which stores
            # them in the`.encoding` dict. xcdat handles decoding non-CF time units by
            # setting xarray to `decode_times=False`. To replicate xarray's decoding
            # behavior, this function pops the "calendar" and "units" from the `.attrs`
            # dict and stores them in the `.encoding` dict.
            calendar = attrs.pop("calendar", None)
            units_attr = attrs.pop("units", None)

            if calendar is None:
                logger.warning(
                    f"{coord.name} does not have a 'calendar' attribute set, so it could "
                    "not be decoded. Set the 'calendar' attribute "
                    f"(`ds.{coord.name}.attrs['calendar]`) and try decoding them again"
                )
                continue

            if units_attr is None:
                logger.warning(
                    f"{coord.name} does not have a 'units' attribute set, it could not be "
                    f"decoded. Set the 'units' attribute (`ds.{coord.name}.attrs['units']`) "
                    "and try decoding them again."
                )
                continue

            try:
                units, ref_date = _split_time_units_attr(units_attr)
            except ValueError:
                logger.warning(
                    f"{coord.name} 'units' attribute ('{units_attr}') is not in a "
                    "supported format ('months since...' or 'years since...') so it "
                    "cannot be decoded."
                )
                continue

            decoded_time = _decode_time_coords(
                ref_date, coord, coord.values, attrs, calendar, units
            )
            decoded_time.encoding = {
                "source": ds.encoding.get("source", "None"),
                "original_shape": coord.shape,
                "dtype": coord.dtype,
                "units": units_attr,
                "calendar": calendar,
            }
            ds = ds.assign_coords({coord.name: decoded_time})

            try:
                bounds = ds.bounds.get_bounds("T", var_key=coord.name)
            except KeyError:
                bounds = None

            if bounds is not None:
                # `.values` converts the underlying array to `np.ndarray` if it
                # is not already an `np.ndarray` (e.g., multi-file Datasets
                # which store data variables as Dask Arrays). Calling `.values`
                # multiple times can result in multiple conversion operations,
                # leading to decreased runtime performance. To avoid this
                # situation, we store `bounds.values` as a variable and reuse it
                # for decoding (if necessary).
                bounds_vals = bounds.values

                if not _is_decoded(bounds_vals):
                    decoded_bounds = _decode_time_bounds(
                        ref_date, bounds, bounds_vals, calendar, units
                    )
                    ds = ds.assign({bounds.name: decoded_bounds})

    return ds


def _decode_time_coords(
    ref_date: str,
    coord: xr.DataArray,
    values: np.ndarray,
    attrs: Dict[str, Any],
    calendar: str,
    units: str,
):
    data = _get_cftime_coords(ref_date, values, calendar, units)

    decoded_time = xr.DataArray(
        name=coord.name,
        data=data,
        dims=coord.dims,
        attrs=attrs,
    )

    return decoded_time


def _decode_time_bounds(
    ref_date: str, bounds: xr.DataArray, values: np.ndarray, calendar: str, units: str
):
    lowers = _get_cftime_coords(ref_date, values[:, 0], calendar, units)
    uppers = _get_cftime_coords(ref_date, values[:, 1], calendar, units)
    data_bounds = np.vstack((lowers, uppers)).T

    decoded_bounds = xr.DataArray(
        name=bounds.name,
        data=data_bounds,
        dims=bounds.dims,
        coords=bounds.coords,
        attrs=bounds.attrs,
    )

    if bounds.chunks is not None:
        decoded_bounds = decoded_bounds.chunk(bounds.chunksizes)

    decoded_bounds.encoding = bounds.encoding

    return decoded_bounds


def _postprocess_dataset(
    dataset: xr.Dataset,
    data_var: Optional[str] = None,
    decode_times: bool = True,
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
    ds = dataset.copy()

    if data_var is not None:
        ds = _keep_single_var(dataset, data_var)

    if decode_times:
        ds = decode_time(ds)

    if center_times:
        ds = center_times_func(dataset)

    if add_bounds:
        ds = ds.bounds.add_missing_bounds()

    if lon_orient is not None:
        ds = swap_lon_axis(ds, to=lon_orient, sort_ascending=True)

    return ds


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


def _is_decoded(time_arr: np.ndarray) -> bool:
    """Check if an array on the time axis is decoded.

    Parameters
    ----------
    arr : np.ndarray
        An array of time axis values (e.g., coords, bounds)

    Returns
    -------
    bool
    """
    if time_arr.ndim == 1:
        vals = time_arr[0]
    else:
        vals = time_arr.flat[0]

    is_decoded = time_arr.dtype == np.dtype("datetime64[ns]") or isinstance(
        vals, cftime.datetime
    )

    return is_decoded


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
        The CF calendar type supported by ``cftime`` . This includes "noleap",
        "360_day", "365_day", "366_day", "gregorian", "proleptic_gregorian",
        "julian", "all_leap", and "standard".
    units : str
        The time units.

    Returns
    -------
    np.ndarray
        An array of `cftime` coordinates.
    """
    if units in NON_CF_TIME_UNITS:
        # Convert offsets to `np.float64` to avoid "TypeError: unsupported type
        # for timedelta days component: numpy.int64".
        offsets = offsets.astype("float64")

        # We don't need to do calendar arithmetic here because the units and
        # offsets are in "months" or "years", which means leap days should not
        # be factored.
        ref_datetime: datetime = parser.parse(ref_date, default=datetime(2000, 1, 1))
        times = np.array(
            [ref_datetime + rd.relativedelta(**{units: offset}) for offset in offsets],
            dtype="object",
        )
        # Convert the array of `datetime` objects into `cftime` objects based on
        # the calendar type.
        date_type = get_date_type(calendar)
        coords = convert_times(times, date_type=date_type)
    else:
        xrunits = units + " since " + ref_date
        coords = xr.coding.times.decode_cf_datetime(
            offsets, xrunits, calendar=calendar, use_cftime=True
        )

    return coords
