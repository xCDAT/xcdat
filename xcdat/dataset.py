"""Dataset module for functions related to an xarray.Dataset."""
import pathlib
from functools import partial
from glob import glob
from typing import Any, Callable, Dict, Hashable, List, Literal, Optional, Tuple, Union

import pandas as pd
import xarray as xr

from xcdat import bounds  # noqa: F401
from xcdat.logger import setup_custom_logger

logger = setup_custom_logger(__name__)

#: List of non-CF compliant time units.
NON_CF_TIME_UNITS: List[str] = ["months", "years"]


def open_dataset(
    path: str,
    data_var: Optional[str] = None,
    decode_times: bool = True,
    **kwargs: Dict[str, Any],
) -> xr.Dataset:
    """Wrapper for ``xarray.open_dataset()`` that applies common operations.

    Operations include:

    - Optional decoding of time coordinates with CF or non-CF compliant units if
      the Dataset has a time dimension
    - Add missing bounds for supported axis
    - Option to limit the Dataset to a single regular (non-bounds) data
      variable, while retaining any bounds data variables

    Parameters
    ----------
    path : str
        Path to Dataset.
    data_var: Optional[str], optional
        The key of the data variable to keep in the Dataset, by default None.
    decode_times: bool
        If True, decode times encoded in the standard NetCDF datetime format
        into datetime objects. Otherwise, leave them encoded as numbers.
        This keyword may not be supported by all the backends, by default True.
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

    Examples
    --------
    Import and call module:

    >>> from xcdat.dataset import open_dataset
    >>> ds = open_dataset("file_path")

    Keep a single variable in the Dataset:

    >>> from xcdat.dataset import open_dataset
    >>> ds = open_dataset("file_path", data_var="tas")
    """
    if decode_times:
        cf_compliant_time: Optional[bool] = has_cf_compliant_time(path)
        if cf_compliant_time is False:
            # XCDAT handles decoding time values with non-CF units.
            ds = xr.open_dataset(path, decode_times=False, **kwargs)
            ds = decode_non_cf_time(ds)
        else:
            ds = xr.open_dataset(path, decode_times=True, **kwargs)
    else:
        ds = xr.open_dataset(path, decode_times=False, **kwargs)

    ds = infer_or_keep_var(ds, data_var)
    ds = ds.bounds.add_missing_bounds()
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
    preprocess: Optional[Callable] = None,
    decode_times: bool = True,
    data_vars: Union[Literal["minimal", "different", "all"], List[str]] = "minimal",
    **kwargs: Dict[str, Any],
) -> xr.Dataset:
    """Wrapper for ``xarray.open_mfdataset()`` that applies common operations.

    Operations include:

    - Optional decoding of time coordinates with CF or non-CF compliant units if
      the Dataset has a time dimension
    - Add missing bounds for supported axis
    - Option to limit the Dataset to a single regular (non-bounds) data
      variable, while retaining any bounds data variables

    ``data_vars`` defaults to ``"minimal"``, which concatenates data variables
    in a manner where only data variables in which the dimension already appears
    are included. For example, the time dimension will not be concatenated to
    the dimensions of non-time data variables such as "lat_bnds" or "lon_bnds".
    `"minimal"` is required for some XCDAT functions, including spatial
    averaging where a reduction is performed using the lat/lon bounds.

    Parameters
    ----------
    path : Union[str, pathlib.Path, List[str], List[pathlib.Path], \
         List[List[str]], List[List[pathlib.Path]]]
        Either a string glob in the form ``"path/to/my/files/*.nc"`` or an
        explicit list of files to open. Paths can be given as strings or as
        pathlib Paths. If concatenation along more than one dimension is desired,
        then ``paths`` must be a nested list-of-lists (see ``combine_nested``
        for details). (A string glob will be expanded to a 1-dimensional list.)
    data_var: Optional[str], optional
        The key of the data variable to keep in the Dataset, by default None.
    preprocess : Optional[Callable], optional
        If provided, call this function on each dataset prior to concatenation.
        You can find the file-name from which each dataset was loaded in
        ``ds.encoding["source"]``.
    decode_times: bool
        If True, decode times encoded in the standard NetCDF datetime format
        into datetime objects. Otherwise, leave them encoded as numbers.
        This keyword may not be supported by all the backends, by default True.
    data_vars: Union[Literal["minimal", "different", "all"], List[str]], optional
        These data variables will be concatenated together:
          * "minimal": Only data variables in which the dimension already
            appears are included, default.
          * "different": Data variables which are not equal (ignoring
            attributes) across all datasets are also concatenated (as well as
            all for which dimension already appears). Beware: this option may
            load the data payload of data variables into memory if they are not
            already loaded.
          * "all": All data variables will be concatenated.
          * list of str: The listed data variables will be concatenated, in
            addition to the "minimal" data variables.
    kwargs : Dict[str, Any]
        Additional arguments passed on to ``xarray.open_mfdataset``. Refer to
        the [2]_ xarray docs for accepted keyword arguments.

    Returns
    -------
    xr.Dataset
        Dataset after applying operations.

    Notes
    -----
    ``xarray.open_mfdataset`` opens the file with read-only access. When you
    modify values of a Dataset, even one linked to files on disk, only the
    in-memory copy you are manipulating in xarray is modified: the original file
    on disk is never touched.

    References
    ----------

    .. [2] https://xarray.pydata.org/en/stable/generated/xarray.open_mfdataset.html

    Examples
    --------
    Import and call module:

    >>> from xcdat.dataset import open_mfdataset
    >>> ds = open_mfdataset(["file_path1", "file_path2"])

    Keep a single variable in the Dataset:

    >>> from xcdat.dataset import open_mfdataset
    >>> ds = open_mfdataset(["file_path1", "file_path2"], data_var="tas")
    """
    if decode_times:
        cf_compliant_time: Optional[bool] = has_cf_compliant_time(paths)
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
    ds = infer_or_keep_var(ds, data_var)
    ds = ds.bounds.add_missing_bounds()
    return ds


def has_cf_compliant_time(
    path: Union[
        str,
        pathlib.Path,
        List[str],
        List[pathlib.Path],
        List[List[str]],
        List[List[pathlib.Path]],
    ]
) -> Optional[bool]:
    """Determine if a dataset has time coordinates with CF compliant units.

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
    units = _split_time_units_attr(time.attrs.get("units"))[0]
    cf_compliant = units not in NON_CF_TIME_UNITS
    return cf_compliant


def decode_non_cf_time(dataset: xr.Dataset) -> xr.Dataset:
    """Decodes time coordinates and time bounds with non-CF compliant units.

    By default, ``xarray`` uses the ``cftime`` module, which only supports
    decoding time with [3]_ CF compliant units. This function fills the gap in
    xarray by being able to decode time with non-CF compliant units such as
    "months since ..." and "years since ...". It extracts the units and
    reference date from the "units" attribute, which are used to convert the
    numerically encoded time values (representing the offset from the reference
    date) to pandas DateOffset objects. These offset values are added to the
    reference date, forming DataArrays of datetime objects that replace the time
    coordinate and time bounds (if they exist) values in the Dataset.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset with numerically encoded time coordinates and time bounds (if
        they exist).

    Returns
    -------
    xr.Dataset
        Dataset with decoded time coordinates and time bounds (if they exist) as
        datetime objects.

    Notes
    -----
    The [4]_ pandas ``DateOffset`` object is a time duration relative to a
    reference date that respects calendar arithmetic. This means it considers
    CF calendar types with or without leap years when adding the offsets to the
    reference date.

    DateOffset is used instead of timedelta64 because timedelta64 does not
    respect calendar arithmetic. One downside of DateOffset (unlike timedelta64)
    is that there is currently no simple way of vectorizing the addition of
    DateOffset objects to Timestamp/datetime64 objects. However, the performance
    of element-wise iteration should be sufficient for datasets that have
    "months" and "years" time units since the number of time coordinates should
    be small compared to "days" or "hours".

    References
    -----
    .. [3] https://cfconventions.org/cf-conventions/cf-conventions.html#time-coordinate
    .. [4] https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects

    Examples
    --------

    Decode the time coordinates with non-CF units in a Dataset:

    >>> from xcdat.dataset import decode_time_units
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
    >>> ds_decoded = decode_time_units(ds)
    >>> ds_decoded.time
    <xarray.DataArray 'time' (time: 3)>
    array(['2000-01-01T00:00:00.000000000', '2001-01-01T00:00:00.000000000',
        '2002-01-01T00:00:00.000000000'], dtype='datetime64[ns]')
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
    {'source': None, 'dtype': dtype('int64'), 'original_shape': (3,), 'units':
    'years since 2000-01-01', 'calendar': 'noleap'}
    """
    time = dataset.cf["T"]
    time_bounds = dataset.get(time.attrs.get("bounds"), None)
    units_attr = time.attrs.get("units")
    units, ref_date = _split_time_units_attr(units_attr)
    ref_date = pd.to_datetime(ref_date)

    data = [ref_date + pd.DateOffset(**{units: offset}) for offset in time.data]
    decoded_time = xr.DataArray(
        name=time.name,
        data=data,
        dims=time.dims,
        coords={time.name: data},
        attrs=time.attrs,
    )
    decoded_time.encoding = {
        "source": dataset.encoding.get("source", "None"),
        "dtype": time.dtype,
        "original_shape": time.shape,
        "units": units_attr,
        "calendar": time.attrs.get("calendar", "none"),
    }
    dataset = dataset.assign_coords({time.name: decoded_time})

    if time_bounds is not None:
        data_bounds = [
            [
                ref_date + pd.DateOffset(**{units: lower}),
                ref_date + pd.DateOffset(**{units: upper}),
            ]
            for [lower, upper] in time_bounds.data
        ]
        decoded_time_bnds = xr.DataArray(
            name=time_bounds.name,
            data=data_bounds,
            dims=time_bounds.dims,
            coords=time_bounds.coords,
            attrs=time_bounds.attrs,
        )
        decoded_time_bnds.coords[time.name] = decoded_time
        decoded_time_bnds.encoding = time_bounds.encoding
        dataset = dataset.assign({time_bounds.name: decoded_time_bnds})

    return dataset


def infer_or_keep_var(dataset: xr.Dataset, data_var: Optional[str]) -> xr.Dataset:
    """Infer or explicitly keep a specific data variable in the Dataset.

    If ``data_var`` is None, then this function checks the number of
    regular (non-bounds) data variables in the Dataset. If there is a single
    regular data var, then it will add an 'xcdat_infer' attr pointing to it in
    the Dataset. XCDAT APIs can then call `get_inferred_var()` to get the data
    var linked to the 'xcdat_infer' attr. If there are multiple regular data
    variables, the 'xcdat_infer' attr is not set and the Dataset is returned
    as is.

    If ``data_var`` is not None, then this function checks if the ``data_var``
    exists in the Dataset and if it is a regular data var. If those checks pass,
    it will subset the Dataset to retain that ``data_var`` and all bounds data
    vars. An 'xcdat_infer' attr pointing to the ``data_var`` is also added
    to the Dataset.

    This utility function is useful for designing XCDAT APIs with an optional
    ``data_var`` kwarg. If ``data_var`` is None, an inference to the desired
    data var is performed with a call to this function. Otherwise, perform the
    API operation explicitly on ``data_var``.

    Parameters
    ----------
    dataset : xr.Dataset
        The Dataset.
    data_var: Optional[str], optional
        The key of the data variable to keep in the Dataset.

    Returns
    -------
    xr.Dataset
        The Dataset.

    Raises
    ------
    KeyError
        If the specified data variable is not found in the Dataset.
    KeyError
        If the user specifies a bounds variable to keep.
    """
    ds = dataset.copy()
    # Make sure the "xcdat_infer" attr is "None" because a Dataset may be
    # written with this attr already set.
    ds.attrs["xcdat_infer"] = "None"

    all_vars = ds.data_vars.keys()
    bounds_vars = ds.bounds.names
    regular_vars: List[Hashable] = list(set(all_vars) ^ set(bounds_vars))

    if len(regular_vars) == 0:
        logger.debug("This dataset only contains bounds data variables.")

    if data_var is None:
        if len(regular_vars) == 1:
            ds.attrs["xcdat_infer"] = regular_vars[0]
        elif len(regular_vars) > 1:
            logger.debug(
                "This dataset contains more than one regular data variable: "
                f"{regular_vars}. If desired, pass the `data_var` kwarg to "
                "limit the dataset to a single data var."
            )
    elif data_var is not None:
        if data_var not in all_vars:
            raise KeyError(
                f"The data variable '{data_var}' does not exist in the dataset."
            )
        if data_var in bounds_vars:
            raise KeyError("Please specify a regular (non-bounds) data variable.")

        ds = dataset[[data_var] + bounds_vars]
        ds.attrs["xcdat_infer"] = data_var

    return ds


def get_inferred_var(dataset: xr.Dataset) -> xr.DataArray:
    """Gets the inferred data variable that is tagged in the Dataset.

    This function looks for the "xcdat_infer" attribute pointing
    to the desired data var in the Dataset, which can be set through
    ``xcdat.open_dataset()``, ``xcdat.open_mf_dataset()``, or manually.

    This utility function is useful for designing XCDAT APIs with an optional
    ``data_var`` kwarg. If ``data_var`` is None, an inference to the desired
    data var is performed with a call to this function. Otherwise, perform the
    API operation explicitly on ``data_var``.

    Parameters
    ----------
    dataset : xr.Dataset
        The Dataset.

    Returns
    -------
    xr.DataArray
        The inferred data variable.

    Raises
    ------
    KeyError
        If the 'xcdat_infer' attr is not set in the Dataset.
    KeyError
        If the 'xcdat_infer' attr points to a non-existent data var.
    KeyError
        If the 'xcdat_infer' attr points to a bounds data var.
    """
    inferred_var = dataset.attrs.get("xcdat_infer", None)
    bounds_vars = dataset.bounds.names

    if inferred_var is None:
        raise KeyError(
            "Dataset attr 'xcdat_infer' is not set so the desired data variable "
            "cannot be inferred. You must pass the `data_var` kwarg to this operation."
        )
    else:
        data_var = dataset.get(inferred_var, None)
        if data_var is None:
            raise KeyError(
                "Dataset attr 'xcdat_infer' is set to non-existent data variable, "
                f"'{inferred_var}'. Either pass the `data_var` kwarg to this operation, "
                "or set 'xcdat_infer' to a regular (non-bounds) data variable."
            )
        if inferred_var in bounds_vars:
            raise KeyError(
                "Dataset attr `xcdat_infer` is set to the bounds data variable, "
                f"'{inferred_var}'. Either pass the `data_var` kwarg, or set "
                "'xcdat_infer' to a regular (non-bounds) data variable."
            )

        return data_var.copy()


def _preprocess_non_cf_dataset(
    ds: xr.Dataset, callable: Optional[Callable] = None
) -> xr.Dataset:
    """Preprocessing for each non-CF compliant dataset in ``open_mfdataset()``.

    This function allows for a user specified preprocess function, in addition
    to XCDAT preprocessing functions.

    One call is performed to ``decode_non_cf_time()`` for decoding each
    dataset's time coordinates and time bounds (if they exist) with non-CF
    compliant units. By default, if ``decode_times=False`` is passed, xarray
    will concatenate time values using the first dataset's "units" attribute.
    This is an issue for cases where the numerically encoded time values are the
    same and the "units" attribute differs between datasets. For example,
    two files have the same time values, but the units of the first file is
    "months since 2000-01-01" and the second is "months since 2001-01-01". Since
    the first dataset's units are used in xarray for concatenating datasets,
    the time values corresponding to the second file will be dropped since they
    appear to be the same as the first file. Calling ``decode_non_cf_time()``
    on each dataset individually before concatenating solves the aforementioned
    issue.

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
        If the units attribute doesn't exist for the time coordinates.
    """
    if units_attr is None:
        raise KeyError("No 'units' attribute found for the dataset's time coordinates.")

    units, reference_date = units_attr.split(" since ")
    return units, reference_date
