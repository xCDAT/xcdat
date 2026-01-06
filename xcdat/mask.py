from typing import Any, Callable

import numpy as np
import regionmask
import xarray as xr

from xcdat import open_dataset
from xcdat._data import _get_pcmdi_mask_path
from xcdat._logger import _setup_custom_logger
from xcdat.axis import get_dim_coords
from xcdat.regridder.accessor import _obj_to_grid_ds
from xcdat.regridder.grid import create_grid
from xcdat.utils import _as_dataarray

logger = _setup_custom_logger(__name__)

VALID_METHODS: list[str] = ["regionmask", "pcmdi"]
VALID_KEEP: list[str] = ["land", "sea"]


def generate_and_apply_land_sea_mask(
    ds: xr.Dataset,
    data_var: str,
    method: str = "regionmask",
    keep: str = "sea",
    threshold: float | None = None,
    mask: xr.DataArray | None = None,
    output_mask: bool | str = False,
    **options: Any,
) -> xr.Dataset:
    """Generate a land-sea mask and apply it to a data variable in a dataset.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to mask.
    data_var : str
        The key of the data variable to mask.
    method : str, optional
        The masking method, by default "regionmask".
        Supported methods: "regionmask", "pcmdi".
    keep : str, optional
        Whether to keep "land" or "sea" points, by default "sea".
    threshold : float | None, optional
        The threshold used to determine cell classification. The default
        value for this argument depends on `keep`. If `keep` is `sea` then
        the default is 0.2 and values less than or equal will be considered
        sea. If `keep` is `land` then the default is 0.8 and values greater
        or equal will be considered land.
    mask : xr.DataArray | None, optional
        A custom mask to apply, by default None. If None, a mask is
        generated using the specified ``method``.
    **options : Any
        These options are passed directly to the ``method``. See
        :func:`xcdat.mask.pcmdi_land_sea_mask` for PCMDI options.

    Returns
    -------
    xr.Dataset
        The dataset with the masked data variable.

    Raises
    ------
    ValueError
        If `keep` is not "land" or "sea".

    Examples
    --------

    Mask a data variable by land using the default method (regionmask):
    >>> ds_masked = generate_mask(ds, "tas", keep="sea")

    Mask a data variable by sea using the PCMDI method with custom threshold:
    >>> ds_masked = generate_mask(ds, "tas", method="pcmdi", keep="land", threshold=0.7)

    Mask a data variable by land using a custom mask and output the mask:
    >>> custom_mask = xr.DataArray(...)  # Define your custom mask here
    >>> ds_masked = generate_mask(ds, "tas", keep="sea", mask=custom_mask, output_mask=True)

    Mask a data variable by sea and add the mask to the dataset with a custom name:
    >>> ds_masked = generate_mask(ds, "tas", keep="land", output_mask="land_mask")
    """
    if keep not in VALID_KEEP:
        raise ValueError(
            f"Keep value {keep!r} is not valid, options are {', '.join(VALID_KEEP)!r}"
        )

    _ds = ds.copy()

    da = _ds[data_var]

    if mask is None:
        mask = generate_land_sea_mask(da, method, **options)

    if keep == "sea":
        _ds[data_var] = da.where(mask <= (threshold or 0.2))
    else:
        _ds[data_var] = da.where(mask >= (threshold or 0.8))

    if output_mask:
        if isinstance(output_mask, str):
            mask_name = output_mask
        else:
            mask_name = f"{data_var}_mask"

        _ds[mask_name] = mask

    return _ds


def generate_land_sea_mask(
    da: xr.DataArray, method: str = "regionmask", **options: Any
) -> xr.DataArray:
    """Generate a land-sea mask.

    Parameters
    ----------
    da : xr.DataArray
        The DataArray to generate the mask for.
    method : str, optional
        The method to use for generating the mask, by default "regionmask".
        Supported methods: "regionmask", "pcmdi".
    **options : Any
        These options are passed directly to the ``method``. See specific
        method documentation for available options:
        :func:`pcmdi_land_sea_mask` for PCMDI options

    Returns
    -------
    xr.DataArray
        The land-sea mask.

    Raises
    ------
    ValueError
        If `method` is not "regionmask" or "pcmdi".

    References
    ----------
    .. _PCMDI's report #58: https://pcmdi.llnl.gov/report/ab58.html

    History
    -------
    2023-06 The [original code](https://github.com/CDAT/cdutil/blob/master/
    cdutil/create_landsea_mask.py) was rewritten using xarray and xcdat by Jiwoo Lee

    Examples
    --------

    Generate a land-sea mask using the default method (regionmask):

    >>> import xcdat
    >>> ds = xcdat.open_dataset("/path/to/file")
    >>> mask = xcdat.mask.generate_land_sea_mask(ds["tas"], method="regionmask")

    Generate a land-sea mask using the PCMDI method with custom options:

    >>> mask = xcdat.mask.generate_land_sea_mask(
    ...     ds["tas"], method="pcmdi", threshold1=0.3, threshold2=0.4
    ... )
    """
    if method not in VALID_METHODS:
        raise ValueError(
            f"Method value {method!r} is not valid, options are {', '.join(VALID_METHODS)!r}"
        )

    if method == "regionmask":
        land_mask = regionmask.defined_regions.natural_earth_v5_0_0.land_110

        lon, lat = get_dim_coords(da, "X"), get_dim_coords(da, "Y")

        land_sea_mask = land_mask.mask(lon, lat=lat)

        land_sea_mask = xr.where(land_sea_mask, 0, 1)
    elif method == "pcmdi":
        land_sea_mask = pcmdi_land_sea_mask(da, **options)

    return land_sea_mask


def pcmdi_land_sea_mask(
    da: xr.DataArray,
    threshold1: float = 0.2,
    threshold2: float = 0.3,
    source: xr.Dataset | None = None,
    source_data_var: str | None = None,
) -> xr.DataArray:
    """
    Generate a land-sea mask using the PCMDI method.

    This method uses a high-resolution land-sea mask and regrids it to the
    resolution of the input DataArray. It then iteratively improves the mask
    based on specified thresholds.

    Parameters
    ----------
    da : xr.DataArray
        The DataArray to generate the mask for.
    threshold1 : float, optional
        The first threshold for improving the mask, by default 0.2.
    threshold2 : float, optional
        The second threshold for improving the mask, by default 0.3.
    source : xr.Dataset | None, optional
        A custom Dataset containing the variable to use as the high-resolution
        source. If not provided, a default high-resolution land-sea mask is used.
    source_data_var : str | None, optional
        The name of the variable in `source` to use as the high-resolution
        source. If `source` is not provided, this defaults to "sftlf".

    Returns
    -------
    xr.DataArray
        The generated land-sea mask.

    Raises
    ------
    ValueError
        If `source` is provided but `source_data_var` is None.

    Notes
    -----
    By default, the ``navy_land.nc`` file is used as the high-resolution land–sea
    mask. This file is distributed by the [1]_ PCMDI (Program for Climate Model
    Diagnosis and Intercomparison) Metrics Package, and is derived from the U.S.
    Navy 1/6° land–sea mask dataset.

    If ``source`` is not provided, the ``navy_land.nc`` file is automatically
    downloaded and cached from the `xcdat-data` repository:
    https://github.com/xCDAT/xcdat-data.

    For more information on caching behavior, refer to the
    :py:func:`xcdat._data._get_pcmdi_mask_path()` function.

    References
    ----------
    .. [1] https://github.com/PCMDI/pcmdi_metrics/blob/main/

    Examples
    --------
    Generate a land-sea mask using the PCMDI method:

    >>> import xcdat
    >>> ds = xcdat.open_dataset("/path/to/file")
    >>> land_sea_mask = xcdat.mask.pcmdi_land_sea_mask(ds["tas"])

    Generate a land-sea mask using the PCMDI method with custom thresholds:

    >>> land_sea_mask = xcdat.mask.pcmdi_land_sea_mask(
    ...     ds["tas"], threshold1=0.3, threshold2=0.4
    ... )

    Generate a land-sea mask using the PCMDI method with a custom high-res source:

    >>> highres_ds = xcdat.open_dataset("/path/to/file")
    >>> land_sea_mask = xcdat.mask.pcmdi_land_sea_mask(
    ...     ds["tas"], source=highres_ds, source_data_var="highres"
    ... )

    For offline workflows, you can pre-download the mask with:

    >>> from xcdat._data import _get_pcmdi_mask_path
    >>> path = _get_pcmdi_mask_path()
    """
    if source is not None and source_data_var is None:
        raise ValueError(
            "The 'source_data_var' value cannot be None when using the 'source' option."
        )

    if source is None:
        source_data_var = "sftlf"

        resource_path = _get_pcmdi_mask_path()

        # Turn off time decoding to prevent logger warning since this dataset
        # does not have a time axis.
        source = open_dataset(resource_path, decode_times=False)

    source_regrid = source.regridder.horizontal(
        source_data_var, _obj_to_grid_ds(da), tool="regrid2"
    )

    mask = source_regrid.copy()
    # Set keep_attrs=False to ensure that attributes from the argument x (1 in
    # this case) are not copied. This preserves the existing attributes of the
    # data variable. The default value is None, which removes all attributes,
    # while True would incorrectly copy attributes from x.
    mask[source_data_var] = xr.where(
        source_regrid[source_data_var] > 0.5, 1, 0, keep_attrs=False
    ).astype("i")

    lon = mask[source_data_var].cf["X"]
    lon_bnds = mask.bounds.get_bounds("X")
    is_circular = _is_circular(lon, lon_bnds)

    surrounds = _generate_surrounds(mask[source_data_var], is_circular)

    i = 0

    while i <= 25:
        logger.debug("Iteration %i", i + 1)

        improved_mask = _improve_mask(
            mask.copy(deep=True),
            source_regrid,
            source_data_var,  # type: ignore[arg-type]
            surrounds,
            is_circular,
            threshold1,
            threshold2,
        )

        if improved_mask.equals(mask):
            break

        mask = improved_mask

        i += 1

    return mask[source_data_var]


def _is_circular(lon: xr.DataArray, lon_bnds: xr.DataArray) -> bool:
    """Check if a longitude axis is circular.

    Parameters
    ----------
    lon : xr.DataArray
        The longitude coordinates.
    lon_bnds : xr.DataArray
        The longitude bounds.

    Returns
    -------
    bool
        True if the longitude axis is circular, False otherwise.
    """
    axis_start, axis_stop = float(lon[0]), float(lon[-1])
    delta = float(lon[-1] - lon[-2])
    alignment = abs(axis_stop + delta - axis_start - 360.0)
    tolerance = 0.01 * delta
    mod_360 = float(lon_bnds[-1][1] - lon_bnds[0][0]) % 360

    return alignment < tolerance and mod_360 == 0


def _improve_mask(
    mask: xr.Dataset,
    source: xr.Dataset,
    data_var: str,
    surrounds: list[np.ndarray],
    is_circular: bool,
    threshold1=0.2,
    threshold2=0.3,
) -> xr.Dataset:
    """Improve a land-sea mask.

    This function improves a land-sea mask by converting points based on
    their surrounding values and a source mask.

    It is useful for enhancing the accuracy of land-sea masks, which are often
    used in climate modeling and geospatial analysis. By considering surrounding
    points and thresholds, it ensures smoother transitions and corrects
    discrepancies between the mask and the source dataset.

    Parameters
    ----------
    mask : xr.Dataset
        The mask to improve.
    source : xr.Dataset
        The source dataset for comparison.
    data_var : str
        The name of the data variable in the mask and source.
    surrounds : list[np.ndarray]
        A list of surrounding points for each point in the mask.
    is_circular : bool
        Whether the longitude axis is circular.
    threshold1 : float, optional
        The first threshold for conversion, by default 0.2.
    threshold2 : float, optional
        The second threshold for conversion, by default 0.3.

    Returns
    -------
    xr.Dataset
        The improved mask.
    """
    mask_approx = _map2four(
        mask,
        data_var,
    )

    diff = source[data_var] - mask_approx[data_var]

    mask_convert_land = _convert_points(
        mask[data_var] * 1.0,
        source[data_var],
        diff,
        threshold1,
        threshold2,
        is_circular,
        surrounds,
    )

    mask_convert_sea = _convert_points(
        mask_convert_land,
        source[data_var],
        diff,
        -threshold1,
        1.0 - threshold2,
        is_circular,
        surrounds,
        convert_land=False,
    )

    mask[data_var] = mask_convert_sea.astype("i")

    return mask


def _map2four(mask: xr.Dataset, data_var: str) -> xr.Dataset:
    """Map a mask to four subgrids and back.

    This function regrids a mask to four subgrids (odd-odd, odd-even,
    even-odd, even-even) and then combines them back into a single mask.
    This is used to approximate the mask at a higher resolution.

    Parameters
    ----------
    mask : xr.Dataset
        The mask to process.
    data_var : str
        The name of the data variable in the mask.

    Returns
    -------
    xr.Dataset
        The processed mask.
    """
    mask_temp = mask.copy()

    lat, lon = mask_temp[data_var].cf["Y"], mask_temp[data_var].cf["X"]
    lat_odd, lat_even = lat[::2], lat[1::2]
    lon_odd, lon_even = lon[::2], lon[1::2]

    odd_odd = create_grid(y=lat_odd, x=lon_odd, add_bounds=True)
    odd_even = create_grid(y=lat_odd, x=lon_even, add_bounds=True)
    even_odd = create_grid(y=lat_even, x=lon_odd, add_bounds=True)
    even_even = create_grid(y=lat_even, x=lon_even, add_bounds=True)

    regrid_odd_odd = mask_temp.regridder.horizontal(data_var, odd_odd, tool="regrid2")
    regrid_odd_even = mask_temp.regridder.horizontal(data_var, odd_even, tool="regrid2")
    regrid_even_odd = mask_temp.regridder.horizontal(data_var, even_odd, tool="regrid2")
    regrid_even_even = mask_temp.regridder.horizontal(
        data_var, even_even, tool="regrid2"
    )

    output = np.zeros(mask_temp[data_var].shape, dtype="f")

    output[::2, ::2] = regrid_odd_odd[data_var].data
    output[::2, 1::2] = regrid_odd_even[data_var].data
    output[1::2, ::2] = regrid_even_odd[data_var].data
    output[1::2, 1::2] = regrid_even_even[data_var].data

    mask_temp[data_var] = (mask_temp[data_var].dims, output)

    return mask_temp


def _convert_points(
    mask: xr.DataArray,
    source: xr.DataArray,
    diff: xr.DataArray,
    threshold1: float,
    threshold2: float,
    is_circular: bool,
    surrounds: list[np.ndarray],
    convert_land=True,
) -> xr.DataArray:
    """Convert points in a mask based on surrounding values.

    This function converts points in a mask from land to sea or sea to land
    based on a set of thresholds and the values of surrounding points.

    Parameters
    ----------
    mask : xr.DataArray
        The mask to modify.
    source : xr.DataArray
        The source data for comparison.
    diff : xr.DataArray
        The difference between the source and an approximated mask.
    threshold1 : float
        Threshold for points in the `diff` DataArray.
    threshold2 : float
        Threshold for points in the `source` DataArray.
    is_circular : bool
        Whether the longitude axis is circular.
    surrounds : list[np.ndarray]
        A list of surrounding points for each point in the mask.
    convert_land : bool, optional
        Whether to convert points to land (True) or sea (False), by default True.

    Returns
    -------
    xr.DataArray
        The modified mask.
    """
    UL, UC, UR, ML, MR, LL, LC, LR = surrounds

    mask_value = 1.0
    compare_func: Callable
    if convert_land:
        compare_func = np.greater
    else:
        compare_func = np.less
        mask_value = 0.0

    flip_value = abs(mask_value - 1.0)

    c1 = compare_func(diff, threshold1)
    c2 = compare_func(source, threshold2)
    c = _as_dataarray(np.logical_and(c1, c2))

    cUL, cUC, cUR, cML, cMR, cLL, cLC, cLR = _generate_surrounds(c, is_circular)

    if is_circular:
        c = c[1:-1]
        temp = source.data[1:-1]
    else:
        c = c[1:-1, 1:-1]
        temp = source.data[1:-1, 1:-1]

    m = np.logical_and(c, compare_func(temp, np.where(cUL, UL, flip_value)))
    m = np.logical_and(m, compare_func(temp, np.where(cUC, UC, flip_value)))
    m = np.logical_and(m, compare_func(temp, np.where(cUR, UR, flip_value)))
    m = np.logical_and(m, compare_func(temp, np.where(cML, ML, flip_value)))
    m = np.logical_and(m, compare_func(temp, np.where(cMR, MR, flip_value)))
    m = np.logical_and(m, compare_func(temp, np.where(cLL, LL, flip_value)))
    m = np.logical_and(m, compare_func(temp, np.where(cLC, LC, flip_value)))
    m = np.logical_and(m, compare_func(temp, np.where(cLR, LR, flip_value)))

    if is_circular:
        mask[1:-1] = xr.where(m, mask_value, mask[1:-1])
    else:
        mask[1:-1, 1:-1] = xr.where(m, mask_value, mask[1:-1, 1:-1])

    return mask


def _generate_surrounds(da: xr.DataArray, is_circular: bool) -> list[np.ndarray]:
    """Generate surrounding points for each point in a DataArray.

    This function returns a list of 8 arrays, each representing the
    values of the 8 surrounding points (UL, UC, UR, ML, MR, LL, LC, LR) for each
    point in the input DataArray.

    Parameters
    ----------
    da : xr.DataArray
        The input DataArray.
    is_circular : bool
        Whether the longitude axis is circular.

    Returns
    -------
    list[np.ndarray]
        A list of 8 arrays representing the surrounding points.
    """
    data = da.data

    y_up, y_mid, y_down = slice(2, None), slice(1, -1), slice(None, -2)

    if is_circular:
        # For circular longitude, roll the horizontal axis.
        UC, LC = data[y_up, :], data[y_down, :]
        ML, MR = np.roll(data[y_mid, :], 1, axis=1), np.roll(data[y_mid, :], -1, axis=1)
        UL, UR = np.roll(data[y_up, :], 1, axis=1), np.roll(data[y_up, :], -1, axis=1)
        LL, LR = (
            np.roll(data[y_down, :], 1, axis=1),
            np.roll(data[y_down, :], -1, axis=1),
        )
    else:
        # For non-circular, slice the horizontal axis.
        x_left, x_mid, x_right = slice(None, -2), slice(1, -1), slice(2, None)
        UC, LC = data[y_up, x_mid], data[y_down, x_mid]
        ML, MR = data[y_mid, x_left], data[y_mid, x_right]
        UL, UR = data[y_up, x_left], data[y_up, x_right]
        LL, LR = data[y_down, x_left], data[y_down, x_right]

    return [UL, UC, UR, ML, MR, LL, LC, LR]
