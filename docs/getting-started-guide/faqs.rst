==========================
Frequently Asked Questions
==========================

Metadata Interpretation
-----------------------

What types of datasets does ``xcdat`` primarily focus on?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``xcdat`` supports datasets with structured grids that follow the `CF Metadata Convention`_.

.. _CF Metadata Convention: http://cfconventions.org/

What structured grids are supported by  ``xcdat``?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
xCDAT aims to be a generalizable package that is compatible with structured grids that
are **CF-compliant** (e.g., CMIP6). xCDAT's spatial averager currently supports
rectilinear grids, and the horizontal regridder supports curvilinear and rectilinear grids.

How does ``xcdat`` interpret dataset metadata?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``xcdat`` leverages `cf_xarray`_ to interpret CF attributes on ``xarray`` objects.
``xcdat`` methods and functions usually accept an ``axis`` argument (e.g.,
``ds.temporal.average("ts")``). This argument is internally mapped to ``cf_xarray``
mapping tables that interpret the CF attributes.

xCDAT also includes its own "fall-back" mapping table that maps axes to their commonly 
accepted names. Refer to `this section <https://github.com/xCDAT/xcdat/blob/main/xcdat/axis.py#L41-L49>`_ of code for the mapping table.

.. _cf_xarray: https://cf-xarray.readthedocs.io/en/latest/index.html

What CF attributes are interpreted using ``cf_xarray`` mapping tables?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Axis names -- used to map to `dimension coordinates`_

  * For example, any ``xr.DataArray`` that has ``axis: "X"`` in its attrs will be
    identified as the "latitude" coordinate variable by ``cf_xarray``.
  * Refer to the ``cf_xarray`` `Axis Names`_ table for more information.

* Coordinate names -- used to map to `dimension coordinates`_

  * For example, any ``xr.DataArray`` that has ``standard_name: "latitude"`` or
    ``_CoordinateAxisType: "Lat"`` or ``"units": "degrees_north"`` in its attrs will be
    identified as the "latitude" coordinate variable by ``cf_xarray``.
  * Refer to the ``cf_xarray`` `Coordinate Names`_ table for more information.

* Bounds attribute -- used to map to bounds data variables

  * For example, the ``latitude`` coordinate variable has ``bounds: "lat_bnds"``, which
    maps its bounds to the ``lat_bnds`` data variable.
  * Refer to ``cf_xarray`` `Bounds Variables`_ page for more information.

.. _dimension coordinates: https://docs.xarray.dev/en/stable/user-guide/data-structures.html#coordinates
.. _Axis Names: https://cf-xarray.readthedocs.io/en/latest/coord_axes.html#axis-names
.. _Coordinate Names: https://cf-xarray.readthedocs.io/en/latest/coord_axes.html#coordinate-names
.. _Bounds Variables: https://cf-xarray.readthedocs.io/en/latest/bounds.html

Handling Bounds
---------------

How are bounds generated in xCDAT?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
For the X, Y, and Z axes, xCDAT generates bounds by using coordinate points as the
midpoint between their lower and upper bounds.

For the T axis, xCDAT can generate bounds either by 1) time frequency (default method)
or 2) midpoints.

  1. time frequency: create time bounds as the start and end of each timestep's period
     using either the inferred or specified time frequency.
  2. midpoint: create time bounds using time coordinates as the midpoint between their
     upper and lower bounds.

For more information, visit the documentation for these APIs:

- https://xcdat.readthedocs.io/en/stable/generated/xarray.Dataset.bounds.add_missing_bounds.html
- https://xcdat.readthedocs.io/en/stable/generated/xarray.Dataset.bounds.add_time_bounds.html
- https://xcdat.readthedocs.io/en/stable/generated/xarray.Dataset.bounds.add_bounds.html

Does xCDAT support generating bounds for multiple axis coordinate systems in the same dataset?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*For example, there are two sets of coordinates called “lat” and “latitude” in the dataset.*

Yes, xCDAT can generate bounds for axis coordinates if they are  “dimension coordinates”
and have the required CF metadata. Dimension coordinates are also considered "index"
coordinates in Xarray and `coordinate variables`_ in CF terminology. “Non-dimension coordinates”
(`auxiliary coordinate variables`_ in CF terminology) are ignored because they aren't used
as indexes and aren't mapped to the axes (dimensions) of variables.

Visit Xarray’s documentation page on `Coordinates`_ for more info on “dimension
coordinates” vs. “non-dimension coordinates”.

.. _coordinate variables: https://docs.xarray.dev/en/stable/user-guide/data-structures.html#coordinates
.. _auxiliary coordinate variables: https://docs.xarray.dev/en/stable/user-guide/data-structures.html#coordinates
.. _Coordinates: https://docs.xarray.dev/en/stable/user-guide/data-structures.html#coordinates

Temporal Metadata
-----------------

What type of time units are supported?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The units attribute must be in the CF compliant format
``"<units> since <reference_date>"``. For example, ``"days since 1990-01-01"``.

Supported CF compliant units include ``day``, ``hour``, ``minute``, ``second``,
which is inherited from ``xarray`` and ``cftime``. Supported non-CF compliant units
include ``year`` and ``month``, which ``xcdat`` is able to parse. Note, the plural form
of these units are accepted.

References:

* https://cfconventions.org/cf-conventions/cf-conventions#time-coordinate

What type of calendars are supported?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``xcdat`` supports that same CF Metadata Convention calendars as ``xarray`` (based on
``cftime`` and ``netCDF4-python`` package).

Supported calendars include:

* ``'standard'``
* ``'gregorian'``
* ``'proleptic_gregorian'``
* ``'noleap'``
* ``'365_day'``
* ``'360_day'``
* ``'julian'``
* ``'all_leap'``
* ``'366_day'``

References:

* https://cfconventions.org/cf-conventions/cf-conventions#calendar

Why does ``xcdat`` decode time coordinates as ``cftime`` objects instead of ``datetime64[ns]``?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One unfortunate limitation of using ``datetime64[ns]`` is that it limits the native
representation of dates to those that fall between the years 1678 and 2262. This affects
climate modeling datasets that have time coordinates outside of this range.

As a workaround, ``xarray`` uses the ``cftime`` library when decoding/encoding
datetimes for non-standard calendars or for dates before year 1678 or after year 2262.

``xcdat`` opted to decode time coordinates exclusively with ``cftime`` because it
has no timestamp range limitations, simplifies implementation, and the output object
type is deterministic. Another benefit with this approach is that ``xcdat`` has its own
algorithm for lazily decoding ``cftime`` objects, which improves up-front I/O performance.

References:

* https://github.com/pydata/xarray/issues/789
* https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timestamp-limitations
* https://discourse.pangeo.io/t/pandas-dtypes-now-free-from-nanosecond-limitation/3106


xCDAT Does Not Support Model-Specific Data Wrangling
----------------------------------------------------

``xcdat`` aims to implement generalized functionality. This means that data wrangling
functionality to handle model-specific data quality issues is out of scope.

If data quality issues are present, ``xarray`` and ``xcdat`` might not be able to open
the datasets. For example, there might be cases where conflicting floating point values
exist between files of a multi-file dataset, or the dataset contains non-CF compliant
attributes that cannot be interpreted correctly by xCDAT.

A few workarounds include:

1. Configuring ``open_dataset()`` or ``open_mfdataset()`` keyword arguments based on
   your needs.
2. Writing a custom ``preprocess()`` function to feed into ``open_mfdataset()``. This
   function preprocesses each dataset file individually before joining them into a single
   Dataset object.

How do I open a multi-file dataset with bounds values that conflict?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In ``xarray``, the default setting for checking compatibility across a multi-file dataset
is ``compat='no_conflicts'``. In cases where variable values conflict between files,
xarray raises ``MergeError: conflicting values for variable <VARIABLE NAME> on objects
to be combined. You can skip this check by specifying compat="override".``

If you still intend on working with these datasets and recognize the source of the issue
(e.g., minor floating point diffs), follow the workarounds below. **Please proceed with
caution. You should understand the potential implications of these workarounds.**

1. Pick the first bounds variable and keep dimensions the same as the input files

   - This option is recommended if you know bounds values should be the same across all
     files, but one or more files has inconsistent bounds values which breaks the
     concatenation of files into a single `xr.Dataset` object.

    .. code-block:: python

      >>> ds = xcdat.open_mfdataset(
              "path/to/files/*.nc",
              compat="override",
              data_vars="minimal",
              coords="minimal",
              join="override",
          )

    - ``compat="override"``: skip comparing and pick variable from first dataset

      - xarray defaults to ``compat="no_conflicts"``

    - ``data_vars="minimal"``: Only data variables in which the dimension already
      appears are included.

      - xcdat defaults to ``data_vars="minimal"``
      - xarray defaults to ``data_vars="all"``

    - ``coords="minimal"``: Only coordinates in which the dimension already appears
      are included.

      - xarray defaults to ``coord="different"``

    - ``join="override"``: if indexes are of same size, rewrite indexes to be those of
      the first object with that dimension. Indexes for the same dimension must have
      the same size in all objects.

      - Alternatively, ``join="left"``: use indexes from the first object with each
        dimension
      - xarray defaults to ``join="outer"``. This can cause issues where data
        variable values conflict because additional coordinates points are
        concatenated at the point of conflict which can produce ``nan`` values.

2. Drop the conflicting bounds variable(s)

   - This option is recommended if you know don't mind dropping the bounds variable(s).
     xcdat will generate and replace the dropped bounds if add_bounds includes the axis
     for the dropped variable (by default, ``add_bounds=["X", "Y"]``).

    .. code-block:: python

      >>> # Drop single variable
      >>> xcdat.open_mfdataset("path/to/files/*.nc", drop_variables="lon_bnds")
      >>> # Drop multiple variables
      >>> xcdat.open_mfdataset("path/to/files/*.nc", drop_variables=["lon_bnds", "lat_bnds"])


For more information on these options, visit the `xarray.open_mfdataset`_ documentation.

.. _`xarray.open_mfdataset`: https://xarray.pydata.org/en/stable/generated/xarray.open_mfdataset.html#xarray-open-mfdataset

Regridding
----------
``xcdat`` extends and provides a uniform interface to `xESMF`_ and `xgcm`_. In addition,
``xcdat`` provides a port of the ``CDAT`` `regrid2 package`_.

Structured rectilinear and curvilinear grids are supported.

.. _`xESMF`: https://xesmf.readthedocs.io/en/stable/
.. _`xgcm`: https://xgcm.readthedocs.io/en/latest/
.. _`regrid2 package`: https://cdms.readthedocs.io/en/latest/regrid2.html

How can I retrieve the grid from a dataset?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The :py:func:`xcdat.regridder.accessor.RegridderAccessor.grid` property is provided to
extract the grid information from a dataset.

.. code-block:: python

  ds = xcdat.open_dataset(...)
  grid = ds.regridder.grid

How do I perform horizontal regridding?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The :py:func:`xcdat.regridder.accessor.RegridderAccessor.horizontal` method provides
access to the `xESMF`_ and `Regrid2`_ packages.

The arguments for each regridder can be found:

* :py:func:`xcdat.regridder.xesmf.XESMFRegridder`
* :py:func:`xcdat.regridder.regrid2.Regrid2Regridder`

An example of `horizontal`_ regridding can be found in the `gallery`_.

.. _`Regrid2`: generated/xcdat.regridder.regrid2.Regrid2Regridder.html
.. _`horizontal`: examples/regridding-horizontal.html
.. _`gallery`: gallery.html

How do I perform vertical regridding?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The :py:func:`xcdat.regridder.accessor.RegridderAccessor.vertical` method provides
access to the `xgcm`_ package.

The arguments for each regridder can be found:

* :py:func:`xcdat.regridder.xgcm.XGCMRegridder`

An example of `vertical`_ regridding can be found in the `gallery`_.

.. _`vertical`: examples/regridding-vertical.html

Can ``xcdat`` automatically derive Parametric Vertical Coordinates in a dataset?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Automatically deriving `Parametric Vertical Coordinates`_ is a planned feature for ``xcdat``.

.. _`Parametric Vertical Coordinates`: http://cfconventions.org/cf-conventions/cf-conventions.html#parametric-vertical-coordinate

Can I regrid data on unstructured grids?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Regridding data on unstructured grids is a feature we are exploring for ``xcdat``.
