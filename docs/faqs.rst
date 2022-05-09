==========================
Frequently Asked Questions
==========================

Data Wrangling
--------------

``xcdat`` aims to implement generalized functionality. This means that functionality intended to handle data quality issues is out of scope, especially for limited cases.

If data quality issues are present, ``xarray`` and ``xcdat`` might not be able to open the datasets.
Examples of data quality issues include conflicting floating point values between files or non-CF compliant attributes.

A few workarounds include:

1. Configuring ``open_dataset()`` or ``open_mfdataset()`` keyword arguments based on your needs.
2. Writing a custom ``preprocess()`` function to feed into ``open_mfdataset()``. This function preprocesses each dataset file individually before joining them into a single Dataset object.


How do I open a multifile dataset with values that conflict?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In ``xarray``, the default setting for checking compatibility across a multifile dataset is ``compat='no_conflicts'``.
If conflicting values exists between files, xarray raises ``MergeError: conflicting values for variable <VARIABLE NAME> on objects to be combined. You can skip this check by specifying compat="override".``

If you still intend on working with these datasets and recognize the source of the issue (e.g., minor floating point diffs), follow the instructions below.
**Please understand the potential implications before proceeding!**

.. code-block:: python

    >>> xcdat.open_mfdataset("path/to/files/*.nc", compat="override", join="override")

1. ``compat="override"``: skip comparing and pick variable from first dataset
2. ``join="override"``:  if indexes are of same size, rewrite indexes to be those of the first object with that dimension. Indexes for the same dimension must have the same size in all objects.

For more information, visit this page: https://xarray.pydata.org/en/stable/generated/xarray.open_mfdataset.html#xarray-open-mfdataset