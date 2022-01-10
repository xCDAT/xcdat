==========================
Frequently Asked Questions
==========================

How do I open files that have conflicting values for a data variable(s)?
------------------------------------------------------------------------
In xarray, the default setting for checking compatibility across files is ``compat='no_conflicts'``.
If conflicting values exist for a data variable between the files, xarray raises ``MergeError: conflicting values for variable <DATA VAR NAME> on objects to be combined. You can skip this check by specifying compat="override".``

Let's say you try opening two files using ``xcdat.open_mfdataset()`` and the aforementioned ``MergeError`` appears for the ``lat_bnds`` data var.
You perform a floating point comparison between both files for ``lat_bnds`` and find a very small floating point difference at specific coordinates.

To workaround this data quality issue and proceed with opening the files, pass these keyword arguments:

1. ``compat="override"``: skip comparing and pick variable from first dataset
2. ``join="override"``:  if indexes are of same size, rewrite indexes to be those of the first object with that dimension. Indexes for the same dimension must have the same size in all objects.

   - ``join`` is set to `"outer_join"`, which might not be desired.

``xcdat.open_mfdataset('path/to/files/*.nc', compat="override", join="override")``

More information here: https://xarray.pydata.org/en/stable/generated/xarray.open_mfdataset.html#xarray-open-mfdataset