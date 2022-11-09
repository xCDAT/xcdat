==========================
Frequently Asked Questions
==========================

Metadata Interpretation
-----------------------

What types of datasets does ``xcdat`` primarily focus on?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``xcdat`` supports datasets that follow the `CF convention`_, but will also strive
to support datasets with common non-CF compliant metadata (e.g., time units in
"months since ..." or "years since ...").

.. _CF convention: http://cfconventions.org/

How does ``xcdat`` interpret dataset metadata?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``xcdat`` leverages `cf_xarray`_ to interpret CF attributes on ``xarray`` objects.
``xcdat`` methods and functions usually accept an ``axis`` argument (e.g.,
``ds.temporal.average(data_var="ts", axis="T")``). This argument is internally mapped to
``cf_xarray`` mapping tables that interpret the CF attributes.

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

Temporal Metadata
-----------------

What type of time units are supported?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The units attribute must be in the CF compliant format
``"<units> since <reference_date>"``. For example, ``"days since 1990-01-01"``


Supported CF compliant units include ``day``, ``hour``, ``minute``, ``second``,
which is inherited from ``xarray`` and ``cftime``. Supported non-CF compliant units
include ``year``, ``month``, which ``xcdat`` is able to parse. Note, the plural form of
these units are accepted.

References:

* https://cfconventions.org/cf-conventions/cf-conventions#time-coordinate

What type of calendars are supported?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``xcdat`` supports that same CF convention calendars as ``xarray`` (based on
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
type is deterministic.

References:

* https://github.com/pydata/xarray/issues/789
* https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timestamp-limitations

Data Wrangling
--------------

``xcdat`` aims to implement generalized functionality. This means that functionality
intended to handle data quality issues is out of scope, especially for limited cases.

If data quality issues are present, ``xarray`` and ``xcdat`` might not be able to open
the datasets. Examples of data quality issues include conflicting floating point values
between files or non-CF compliant attributes that are not common.

A few workarounds include:

1. Configuring ``open_dataset()`` or ``open_mfdataset()`` keyword arguments based on
   your needs.
2. Writing a custom ``preprocess()`` function to feed into ``open_mfdataset()``. This
   function preprocesses each dataset file individually before joining them into a single
   Dataset object.

How do I open a multi-file dataset with values that conflict?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In ``xarray``, the default setting for checking compatibility across a multi-file dataset
is ``compat='no_conflicts'``. If conflicting values exists between files, xarray raises
``MergeError: conflicting values for variable <VARIABLE NAME> on objects to be combined.
You can skip this check by specifying compat="override".``

If you still intend on working with these datasets and recognize the source of the issue
(e.g., minor floating point diffs), follow the instructions below.
**Please understand the potential implications before proceeding!**

.. code-block:: python

    >>> xcdat.open_mfdataset("path/to/files/*.nc", compat="override", join="override")

1. ``compat="override"``: skip comparing and pick variable from first dataset
2. ``join="override"``:  if indexes are of same size, rewrite indexes to be those of the
   first object with that dimension. Indexes for the same dimension must have the same
   size in all objects.

For more information, visit this page: https://xarray.pydata.org/en/stable/generated/xarray.open_mfdataset.html#xarray-open-mfdataset