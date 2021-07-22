"""Averager module that contains functions related to geospatial averaging.
"""

import xarray as xr
import numpy as np

# %%
# Outline:
#     spatial_average(): main spatial averaging function
#         _validate_region_lon_bounds(): check/update regional longitude bounds so they are consistent with the variable array
#             _switch_lon_bounds(): sub-function to actually switch the lon_bounds to a given type (either -180 to 180 or 0 to 360)
#         _get_axis_weights(): get the default axis weights
#         _combine_axis_weights(): combine axis weights into one weighting matrix
#         _averager(): perform weighted average (with appropriate masking)

# %% Notes

# Differences between CDAT and this algorithm are smaller than 10**-4
# CDAT is faster for smaller spatial scales, but this algorithm is faster for larger spatial scales
# Using xarrays data reduction functionality gives similar results, but is slower than the averager function defined here
# Averaging axes are specified with a list (e.g., axis=['lat', 'lon']) rather than the CDAT convention (axis='xy')
# It would be helpful to be able to get axis weights (get_axis_weights()) more generically (some information is hardcoded)
    # If .bounds had a method to return a list of bounds entries, we could create weights for all dimensions
# May want to keep the axis weights or full weight array (i.e., add this information into the variable dataarray properties)
# Is xvariable a good name?
# Need to write unit tests
# Can remove logging around bounds found in dataset
# Functions are too specific to "lat" and "lon" (may have issues if dims are latitude / longitude)
# Should remove these notes and outline...


def _validate_region_lon_bounds(grid_lon_bounds, region_lon_bounds):
    """Determines if values in a list of longitude coordinates falls within
    longitudinal grid cell boundaries of a specified longitude axis.

    Operations include:

    - Check to determine whether longitude values are within the boundaries of
    the target longitude axis

    Parameters
    ----------
    grid_lon_bounds : xr.DataArray
        longitude axis [len(lon), 2]
    region_lon_bounds : float | int | list | array
        list of longitude values to validate

    Returns
    -------
    boolean
        True if all input longitude bounds are within the boundaries of the
        target longitudinal axis

    Examples
    --------
    Import:
    >>> from xcdat.dataset import open_dataset
    >>> from xcdat.variable import open_variable

    Open dataset and dataarray:
    >>> filename = 'test.nc'
    >>> ds = open_dataset(filename)
    >>> tas = open_variable(ds, 'tas')

    Specify some regional bounds:
    >>> regional_bounds = [-170, -120]

    Validate:
    >>> in_bounds = _validate_region_lon_bounds(tas.bounds.lon,
                                                regional_bounds)
    """
    # ensure iterable array
    region_lon_bounds = np.asarray(region_lon_bounds)
    region_lon_bounds = np.atleast_1d(region_lon_bounds)
    # get the max / min bounds for each grid cell
    min_grid_lon_bounds = np.min(grid_lon_bounds, axis=1)
    max_grid_lon_bounds = np.max(grid_lon_bounds, axis=1)
    # iterate over regional bounds and check to see if they fall within
    # any grid cell bounds.
    in_bounds = True
    for lon in region_lon_bounds:
        inds = np.where((min_grid_lon_bounds <= lon) &
                        (max_grid_lon_bounds >= lon))[0]
        if len(inds) == 0:
            in_bounds = False
    return in_bounds


def _switch_lon_bounds(lon, totype='to360'):
    """Switch longitude axis orientation either 'to360' (0 to 360) or 'to180'
    (0 to 180).

    Parameters
    ----------
    lon : array
        array of longitude values to convert
    totype : str, optional
        string specifying conversion type ('to180' or 'to360')

    Returns
    -------
    array
        Array of converted longitude values

    Notes
    -----
    This does not re-order the values in any way; it only converts the values
    in-place between longitude conventions (0 to 360 or -180 to 180).
    """
    orig_type = type(lon)  # note original datatype
    lon = np.array(lon)  # convert to array
    # convert to specified longitude axis orientation
    if totype == 'to360':
        inds = np.where(lon < 0)[0]
        lon[inds] = lon[inds] + 360
    elif totype == 'to180':
        inds = np.where(lon > 180)[0]
        lon[inds] = lon[inds] - 360
    # switch back to original data type
    # REVIEW: Is there an easier way to be flexible to the datatype provided?
    if orig_type == list:
        lon = list(lon)
    elif orig_type == float:
        lon = float(lon)
    elif orig_type == int:
        lon = int(lon)

    return lon


def _get_axis_weights(xvariable,
                      lat_bounds=None,
                      lon_bounds=None,
                      weight='weighted'):
    """Function to generate axis weights for subsequent operations (e.g.,
    area averaging) with the option to generate weights based on a rectilinear
    regional selection.

    Operations include:

    - Generate weights for axes using axis bounds
    - If a region is specified, calculate the weights corresponding to the
    regional bounds

    Parameters
    ----------
    xvariable : xr.DataArray
        xr.DataArray produced using ``open_variable``
    lat_bounds : list, optional
        list of latitude boundaries for regional selection (e.g.,
        lat_bounds=[-5, 5])
    lon_bounds : list, optional
        list of longitude boundaries for regional selection (e.g.,
        lon_bounds=[-170, -120])
    weight : str, optional
        Axis weighting option (either "weighted" or "unweighted"), default
        "weighted"

    Returns
    -------
    dict
        Set of weights for each eligible dimension.

    Notes
    -----
    This is a prototype and needs work to make it work more generically. It
    currently only works with lat / lon. It might be useful to check the
    latitude axis units.

    Examples
    --------
    Import and call module:
    >>> from xcdat.dataset import open_dataset
    >>> from xcdat.variable import open_variable
    >>>
    >>> filename = 'test.nc'
    >>> ds = open_dataset(filename)
    >>> tas = open_variable(ds, 'tas')
    >>> axis_weights = _get_axis_weights(tas)

    Raises
    ------
    ValueError
        Incorrect weight option specified
    """

    # in the case bounds are unweighted, just return ones
    if weight == 'unweighted':
        axis_weights = {}
        for dim in xvariable.dims:
            # uniform weights based on original dimension
            xdim = xvariable[dim]
            dim_wts = xr.DataArray(np.ones(len(xdim)),
                                   dims=xdim.dims,
                                   coords=xdim.coords)
            dim_wts.name = dim + '_wts'
            axis_weights[dim] = dim_wts
        return axis_weights

    # just check a valid weight option is provided
    # if invalid, throw error, else continue (with weighted logic)
    # REVIEW: Should we throw a different exception type?
    if weight not in ['weighted', 'unweighted']:
        raise ValueError('Incorrect weight option. Choose either "weighted"' +
                         ' or "unweighted"')

    # note that xvariables.bounds does not return a list
    # so for now this is hard-coded
    bounds = {}
    bounds['time'] = xvariable.bounds.time.copy(deep=True)
    bounds['lat'] = xvariable.bounds.lat.copy(deep=True)
    bounds['lon'] = xvariable.bounds.lon.copy(deep=True)
    # put region bounds in a dictionary for convenience
    region_bounds = {'lat': lat_bounds, 'lon': lon_bounds}
    # loop over dimensions and produce coordinate weights
    # in reality, we'll need to include an if-statement
    # that skips a dimension or assigns uniform weights
    # if a dimension does not have bounds
    axis_weights = {}
    for dim in xvariable.dims:
        # if a dimension doesn't have bounds, do not calculate weights
        if dim not in bounds.keys():
            continue
        dim_bounds = bounds[dim]
        # adjust bound limits based on regional selection (if applicable)
        if dim in region_bounds.keys():
            if region_bounds[dim]:
                dim_region_bounds = region_bounds[dim]
                dim_bounds[dim_bounds[:, 0] > np.max(dim_region_bounds), 0] = np.max(dim_region_bounds)  # above / right of region
                dim_bounds[dim_bounds[:, 1] > np.max(dim_region_bounds), 1] = np.max(dim_region_bounds)  # above / right of region
                dim_bounds[dim_bounds[:, 0] < np.min(dim_region_bounds), 0] = np.min(dim_region_bounds)  # below / left of region
                dim_bounds[dim_bounds[:, 1] < np.min(dim_region_bounds), 1] = np.min(dim_region_bounds)  # below / left of region
        # latitude weights scale with sine(latitude)
        if dim in ['lat', 'latitude']:
            dim_bounds = np.sin(np.radians(dim_bounds))
        # weights are generally the difference of the boundaries
        # REVIEW: Is this always true for our purposes?
        dim_wts = np.abs(dim_bounds[:, 1] - dim_bounds[:, 0])
        # we may want to attach the bounds to the dimension weights, too
        dim_wts.attrs = dim_bounds.attrs
        dim_wts.name = dim + '_wts'
        axis_weights[dim] = dim_wts
    # return the axis weights
    return axis_weights


def _combine_axis_weights(axis_weights, axis=['lat', 'lon']):
    """Function generically re-scales axis weights for a given region.

    Operations include:

    - Create N-dimensional weighting array based on axis_weights and axis
    dimensions specified

    Parameters
    ----------
    axis_weights : dict
        Dictionary of axis weights (using dims as keys)
    axis : list, optional
        List of axes that should be weighted (default ['lat', 'lon'])

    Returns
    -------
    xr.DataArray
        Array of weights

    Examples
    --------
    Import and call module:
    >>> from xcdat.dataset import open_dataset
    >>> from xcdat.variable import open_variable
    >>>
    >>> filename = 'test.nc'
    >>> ds = open_dataset(filename)
    >>> tas = open_variable(ds, 'tas')
    >>> axis_weights = _get_axis_weights(tas)
    >>> weights = _combine_axis_weights(axis_weights, axis=['lat', 'lon'])
    """
    # ensure averaging axis is a list
    if type(axis) == str:
        axis = list(axis)
    # initialize weight matrix
    weights = axis_weights[axis[0]]
    if len(axis) > 1:
        for i in np.arange(1, len(axis)):
            weights = weights * axis_weights[axis[i]]
    return weights


def _averager(xvariable, weights, axis):
    """Perform a weighted average of a dataarray.

    Operations include:

    - Ensure masked data receives zero weight
    - Perform weighted average over user-specified dimensions

    Parameters
    ----------
    xvariable : xr.DataArray
        xr.DataArray produced using ``open_variable``

    weights : xr.DataArray
        DataArray containing the desired weights for the variable array

    axis : list
        List of string values denoting the dims that should be averaged over
        (e.g., axis=['lat', 'lon'])

    Returns
    -------
    xr.DataArray
        Dataarray that has been reduced via a weighted average.

    Notes
    -----
    Needs to be evaluated for speed and compared to native xarray and CDAT
    functionality.

    """
    # get masked data
    masked = np.isnan(xvariable)
    # ensure weights includes all of the same axes that are in the
    # variable array
    for i, dim in enumerate(xvariable.dims):
        if dim not in weights.dims:
            weights = weights.expand_dims(dim)
    # tranpose to ensure axis ordering is the same
    weights = weights.transpose(*xvariable.dims)
    # expand array so that the variable and weights arrays have identical shape
    # REVIEW: It may be useful to create a function (e.g., genutil.grower)
    #         that does this operation.
    ntile = []
    for i, dim in enumerate(weights.dims):
        if len(weights[dim]) != len(xvariable[dim]):
            ntile.append(len(xvariable[dim]))
        else:
            ntile.append(1)
    weights = np.tile(weights, ntile)
    # zero out missing data
    weights[masked] = 0.
    # get averaging axis indices
    aaxis = []
    for i, dim in enumerate(xvariable.dims):
        if dim in axis:
            aaxis.append(i)
    aaxis = tuple(aaxis)
    # weighted average over appropriate axes
    num = np.sum(weights * xvariable, axis=aaxis)
    den = np.sum(weights, axis=aaxis)
    wavg = num / den

    return wavg


def spatial_average(xvariable,
                    lat_bounds=None,
                    lon_bounds=None,
                    axis=['lat', 'lon'],
                    weight='weighted'):
    """Calculate the spatial average for a rectilinear grid over a (optional)
    specified regional domain.

    Operations include:

    - Check (optional) regional boundary specification to ensure it is within
      the variable domain
    - Get axis weights for standard lat / lon domains
    - Adjust weights to conform to (optional) regional boundary
    - Compute spatial weighted average

    Parameters
    ----------
    xvariable : xr.DataArray
        xr.DataArray produced using ``open_variable``
    lat_bounds : list, optional
        List of upper and lower latitude boundaries [default None]
    lon_bounds : list, optional
        List of upper and lower longitude boundaries [default None]
    axis : list, optional
        List of dimensions to average over (default ['lat', 'lon'])
    weight : str, optional
        Weighting option to calculate weights proportional to area
        ("weighted"), uniform weights ("unweighted"). Alternatively, a labelled
        dataarray can be passed in as a custom set of weights.

    Returns
    -------
    xr.DataArray
        Dataarray that has been reduced via a weighted average.

    Notes
    -----
    This assumes a certain pathology (the axis orientation simply does
    not match). More complex issues may exist.

    Examples
    --------
    Import:
    >>> from xcdat.dataset import open_dataset
    >>> from xcdat.variable import open_variable

    Open dataset and dataarray:
    >>> filename = 'test.nc'
    >>> ds = open_dataset(filename)
    >>> tas = open_variable(ds, 'tas')

    Get global average time series:
    >>> ts_global = spatial_average(tas)

    Get time series in Nino 3.4 domain:
    >>> ts_n34 = spatial_average(tas,
                                 lat_bounds=[-5, 5],
                                 lon_bounds=[-170, -120])

    Get zonal mean time series:
    >>> ts_zonal = spatial_average(tas, axis=['lon'])

    Validate / fix bounds:
    >>> validated_regional_bounds = validate_region_lon_bounds(tas.bounds.lon,
                                                               regional_bounds)

    Raises
    ------
    ValueError
        Thrown if there is a conflict between specified bounds and the
        dataarray bounds
    """
    # Check to see if the region bounds are within the domain of the dataset
    if lon_bounds:
        # if not, guess the axis orientation (i.e., -180 to 180 or 0 to 360) of
        # the dataarray axes and then convert the region bounds to that
        # orientation
        if not _validate_region_lon_bounds(xvariable.bounds.lon, lon_bounds):
            if np.max(xvariable.lon) <= 180:
                lon_bounds = _switch_lon_bounds(lon_bounds, totype='to180')
            else:
                lon_bounds = _switch_lon_bounds(lon_bounds, totype='to360')
            # after region bounds are adjusted, check to ensure they are
            # consistent with the dataarray axes
            if not _validate_region_lon_bounds(xvariable.bounds.lon,
                                               lon_bounds):
                # REVIEW: Should we throw a more specific error?
                raise ValueError('Attempted to fix specified longitude ' +
                                 'bounds, but they are not within ' +
                                 ' the longitude axis domain.')
    # if weights are not supplied, calculate them
    if type(weight) == str:
        axis_weights = _get_axis_weights(xvariable,
                                         lat_bounds,
                                         lon_bounds,
                                         weight=weight)
        # combine all weights into one weighting matrix
        weights = _combine_axis_weights(axis_weights)
    else:
        weights = weight
    wavg = _averager(xvariable, weights, axis=axis)
    return wavg
