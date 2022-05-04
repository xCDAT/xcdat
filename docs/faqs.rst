==========================
Frequently Asked Questions
==========================

Data Wrangling
--------------

Some datasets might have quality issues, such as non-CF compliant attributes and inconsistent values.
As a result, ``xarray`` and ``xcdat`` not being able to open these datasets using the default settings.

How do I open datasets that have data and/or coordinate variables with conflicting values?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In ``xarray``, the default setting for checking compatibility across files is ``compat='no_conflicts'``.
If conflicting values for a data variable exists between the files, xarray raises ``MergeError: conflicting values for variable <DATA VAR NAME> on objects to be combined. You can skip this check by specifying compat="override".``

Let's say you try opening two files using ``xcdat.open_mfdataset()`` and the aforementioned ``MergeError`` appears for the ``lat_bnds`` data var.
You perform floating point comparison for ``lat_bnds`` and find a very small difference at specific coordinates.

To workaround this data quality issue and proceed with opening the files, pass these keyword arguments:

.. code-block:: python

    >>> xcdat.open_mfdataset("path/to/files/*.nc", compat="override", join="override")


1. ``compat="override"``: skip comparing and pick variable from first dataset
2. ``join="override"``:  if indexes are of same size, rewrite indexes to be those of the first object with that dimension. Indexes for the same dimension must have the same size in all objects.

For more information, visit this page: https://xarray.pydata.org/en/stable/generated/xarray.open_mfdataset.html#xarray-open-mfdataset