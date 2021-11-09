=======
History
=======

v0.1.0 (7 October, 2021)
------------------------

New Features
~~~~~~~~~~~~

-  Add geospatial averaging API through
   ``DatasetSpatialAverageAccessor`` class by @pochedls and
   @tomvothecoder in #87

   -  Does not support parallelism with Dask yet

-  Add wrappers for xarray's ``open_dataset`` and ``open_mfdataset`` to
   apply common operations such as:

   -  If the dataset has a time dimension, decode both CF and non-CF
      time units
   -  Generate bounds for supported coordinates if they don’t exist
   -  Option to limit the Dataset to a single regular (non-bounds) data
      variable while retaining any bounds data variables

-  Add ``DatasetBoundsAccessor`` class for filling missing bounds,
   returning mapping of bounds, returning names of bounds keys
-  Add ``BoundsAccessor`` class for accessing xcdat public methods
   from other accessor classes

   -  This will be probably be the API endpoint for most users, unless
      they prefer importing the individual accessor classes

-  Add ability to infer data variables in xcdat APIs based on the
   "xcdat_infer" Dataset attr

   -  This attr is set in ``xcdat.open_dataset()``,
      ``xcdat_mfdataset()``, or manually

-  Utilizes ``cf_xarray`` package
   (https://github.com/xarray-contrib/cf-xarray)


Documentation
~~~~~~~~~~~~~

-  Visit the docs here:
   https://xcdat.readthedocs.io/en/latest/index.html

CI/CD
~~~~~

-  100% code coverage (https://app.codecov.io/gh/XCDAT/xcdat)
-  GH Actions for CI/CD build (https://github.com/XCDAT/xcdat/actions)
-  Pytest and pytest-cov for test suite

**Full Changelog**: https://github.com/XCDAT/xcdat/commits/v0.1.0
