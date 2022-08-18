=======
History
=======

v0.3.1 (18 August 2022)
-----------------------

Bug Fixes
~~~~~~~~~

Bounds
^^^^^^

-  Ignore singleton coordinates without dims when attempting to generate
   bounds by `Stephen Po-Chedley <https://github.com/pochedls>`__ in
   https://github.com/xCDAT/xcdat/pull/281
-  Modify logic to not throw error for singleton coordinates (with no
   bounds) by `Stephen Po-Chedley <https://github.com/pochedls>`__ in
   https://github.com/xCDAT/xcdat/pull/313

Time Axis and Coordinates
^^^^^^^^^^^^^^^^^^^^^^^^^

-  Fix ``TypeError`` with Dask Arrays from multifile datasets in
   temporal averaging by `Stephen
   Po-Chedley <https://github.com/pochedls>`__ in
   https://github.com/xCDAT/xcdat/pull/291
-  Use ``cftime`` to avoid out of bounds ``datetime`` when decoding
   non-CF time coordinates by `Stephen
   Po-Chedley <https://github.com/pochedls>`__ and `Tom
   Vo <https://github.com/tomvothecoder>`__ in
   https://github.com/xCDAT/xcdat/pull/283
-  Use ``cftime`` for temporal averaging operations to avoid out of
   bounds ``datetime`` by `Stephen
   Po-Chedley <https://github.com/pochedls>`__ and `Tom
   Vo <https://github.com/tomvothecoder>`__ in
   https://github.com/xCDAT/xcdat/pull/302
-  Fix ``open_mfdataset()`` dropping time encoding attrs by `Tom
   Vo <https://github.com/tomvothecoder>`__ in
   https://github.com/xCDAT/xcdat/pull/309
-  Replace “time” references with ``self._dim`` in
   ``class TemporalAccessor`` by `Tom
   Vo <https://github.com/tomvothecoder>`__ in
   https://github.com/xCDAT/xcdat/pull/312

Internal Changes
~~~~~~~~~~~~~~~~

-  Filters safe warnings. by `Jason
   Boutte <https://github.com/jasonb5>`__ in
   https://github.com/xCDAT/xcdat/pull/276

Documentation
~~~~~~~~~~~~~

-  update conda install to conda create by `Paul
   Durack <https://github.com/durack1>`__ in
   https://github.com/xCDAT/xcdat/pull/294
-  Update project overview and planned features list by `Tom
   Vo <https://github.com/tomvothecoder>`__ in
   https://github.com/xCDAT/xcdat/pull/298
-  Fix bullet formatting in ``README.rst`` and\ ``index.rst`` by `Tom
   Vo <https://github.com/tomvothecoder>`__ in
   https://github.com/xCDAT/xcdat/pull/299
-  Fix Jupyter headings not rendering with pandoc by `Tom
   Vo <https://github.com/tomvothecoder>`__ in
   https://github.com/xCDAT/xcdat/pull/318

DevOps
~~~~~~

-  Unify workspace settings with ``settings.json`` by `Tom
   Vo <https://github.com/tomvothecoder>`__ in
   https://github.com/xCDAT/xcdat/pull/297

-  Run CI/CD on “push” and “workflow_dispatch” by `Tom
   Vo <https://github.com/tomvothecoder>`__ in
   https://github.com/xCDAT/xcdat/pull/287 and
   https://github.com/xCDAT/xcdat/pull/288

-  Pin ``numba=0.55.2`` in dev env and constrain ``numba>=0.55.2`` in ci
   env by `Tom Vo <https://github.com/tomvothecoder>`__ in
   https://github.com/xCDAT/xcdat/pull/280

-

   -  Update conda env yml files and add missing dependencies by `Tom
      Vo <https://github.com/tomvothecoder>`__ in
      https://github.com/xCDAT/xcdat/pull/307

New Contributors
~~~~~~~~~~~~~~~~

-  `Paul Durack <https://github.com/durack1>`__ made their first
   contribution in https://github.com/xCDAT/xcdat/pull/294

**Full Changelog**: https://github.com/xCDAT/xcdat/compare/v0.3.0…v0.3.1

v0.3.0 (27 June 2022)
------------------------

New Features
~~~~~~~~~~~~

-  Add horizontal regridding by `Jason Boutte <https://github.com/jasonb5>`__ in
   https://github.com/xCDAT/xcdat/pull/164
-  Add averages with time dimension removed by `Tom Vo <https://github.com/tomvothecoder>`__ in
   https://github.com/xCDAT/xcdat/pull/236
-  Update ``_get_weights()`` method in ``class SpatialAccessor`` and
   ``class TemporalAccessor`` by `Tom Vo <https://github.com/tomvothecoder>`__ in
   https://github.com/xCDAT/xcdat/pull/252

   -  Add ``keep_weights`` keyword attr to reduction methods
   -  Make ``_get_weights()`` public in ``class SpatialAccessor``

-  Update ``get_axis_coord()`` to interpret more keys by `Tom Vo <https://github.com/tomvothecoder>`__
   in https://github.com/xCDAT/xcdat/pull/262

   -  Along with the ``axis`` attr, it also now interprets
      ``standard_name`` and the dimension name

Bug Fixes
~~~~~~~~~

-  Fix ``add_bounds()`` breaking when time coords are ``cftime`` objects
   by `Tom Vo <https://github.com/tomvothecoder>`__ in https://github.com/xCDAT/xcdat/pull/241
-  Fix parsing of custom seasons for departures by `Tom Vo <https://github.com/tomvothecoder>`__ in
   https://github.com/xCDAT/xcdat/pull/246
-  Update ``swap_lon_axis`` to ignore same systems, which was causing
   odd behaviors for (0, 360) by `Tom Vo <https://github.com/tomvothecoder>`__ in
   https://github.com/xCDAT/xcdat/pull/257

Breaking Changes
~~~~~~~~~~~~~~~~

-  Remove ``class XCDATAccessor`` by `Tom Vo <https://github.com/tomvothecoder>`__ in
   https://github.com/xCDAT/xcdat/pull/222
-  Update spatial ``axis`` arg supported type and keys by `Tom Vo <https://github.com/tomvothecoder>`__
   in https://github.com/xCDAT/xcdat/pull/226

   -  Now only supports CF-compliant axis names (e.g., “X”, “Y”)

-  Remove ``center_times`` kwarg from temporal averaging methods by
   `Tom Vo <https://github.com/tomvothecoder>`__ in https://github.com/xCDAT/xcdat/pull/254

Documentation
~~~~~~~~~~~~~

-  Revert official project name from “XCDAT” to “xCDAT” by
   `Tom Vo <https://github.com/tomvothecoder>`__ in https://github.com/xCDAT/xcdat/pull/231
-  [DOC] Add CDAT API mapping table and gallery examples by
   `Tom Vo <https://github.com/tomvothecoder>`__ in https://github.com/xCDAT/xcdat/pull/239

Internal Changes
~~~~~~~~~~~~~~~~

-  Update time coordinates object type from ``MultiIndex`` to
   ``datetime``/``cftime`` for ``TemporalAccessor`` reduction methods
   and add convenience methods by `Tom Vo <https://github.com/tomvothecoder>`__ in
   https://github.com/xCDAT/xcdat/pull/221
-  Extract method ``_postprocess_dataset()`` and make bounds generation
   optional by `Tom Vo <https://github.com/tomvothecoder>`__ in https://github.com/xCDAT/xcdat/pull/223
-  Update ``add_bounds`` kwarg default value to ``True`` by
   `Tom Vo <https://github.com/tomvothecoder>`__ in https://github.com/xCDAT/xcdat/pull/230
-  Update ``decode_non_cf_time`` to return input dataset if the time
   “units” attr can’t be split into unit and reference date by `Stephen Po-Chedley <https://github.com/pochedls>`__
   in https://github.com/xCDAT/xcdat/pull/263

v0.2.0 (24 March 2022)
------------------------

New Features
~~~~~~~~~~~~

-  Add support for spatial averaging parallelism via Dask by `Stephen Po-Chedley <https://github.com/pochedls>`__
   in https://github.com/xCDAT/xcdat/pull/132
-  Refactor spatial averaging with more robust handling of longitude
   spanning prime meridian by `Stephen Po-Chedley <https://github.com/pochedls>`__ in
   https://github.com/xCDAT/xcdat/pull/152
-  Update xcdat.open_mfdataset time decoding logic by `Stephen Po-Chedley <https://github.com/pochedls>`__ in
   https://github.com/xCDAT/xcdat/pull/161
-  Add function to swap dataset longitude axis orientation by
   `Tom Vo <https://github.com/tomvothecoder>`__ in https://github.com/xCDAT/xcdat/pull/145
-  Add utility functions by `Tom Vo <https://github.com/tomvothecoder>`__ in
   https://github.com/xCDAT/xcdat/pull/205
-  Add temporal utilities and averaging functionalities by
   `Tom Vo <https://github.com/tomvothecoder>`__ in https://github.com/xCDAT/xcdat/pull/107

Bug Fixes
~~~~~~~~~

-  Add exception for coords of len <= 1 or multidimensional coords in
   ``fill_missing_bounds()`` by `Tom Vo <https://github.com/tomvothecoder>`__ in
   https://github.com/xCDAT/xcdat/pull/141
-  Update ``open_mfdataset()`` to avoid data vars dim concatenation by
   `Tom Vo <https://github.com/tomvothecoder>`__ in https://github.com/xCDAT/xcdat/pull/143
-  Fix indexing on axis keys using generic map (related to spatial
   averaging) by `Tom Vo <https://github.com/tomvothecoder>`__ in
   https://github.com/xCDAT/xcdat/pull/172

Breaking Changes
~~~~~~~~~~~~~~~~

-  Rename accessor classes and methods for API consistency by
   `Tom Vo <https://github.com/tomvothecoder>`__ in https://github.com/xCDAT/xcdat/pull/142
-  Rename ``fill_missing_bounds()`` to ``add_missing_bounds()`` by
   `Tom Vo <https://github.com/tomvothecoder>`__ in https://github.com/xCDAT/xcdat/pull/157
-  Remove data variable inference API by `Tom Vo <https://github.com/tomvothecoder>`__ in
   https://github.com/xCDAT/xcdat/pull/196
-  Rename spatial file and class by `Tom Vo <https://github.com/tomvothecoder>`__ in
   https://github.com/xCDAT/xcdat/pull/207

Documentation
~~~~~~~~~~~~~

-  update README by `Jill Chengzhu Zhang <https://github.com/chengzhuzhang>`__ in
   https://github.com/xCDAT/xcdat/pull/127
-  Update readme by `Jiwoo Lee <https://github.com/lee1043>`__ in https://github.com/xCDAT/xcdat/pull/129
-  Update ``HISTORY.rst`` and fix docstrings by `Tom Vo <https://github.com/tomvothecoder>`__ in
   https://github.com/xCDAT/xcdat/pull/139
-  Update ``README.rst`` content and add logo by `Tom Vo <https://github.com/tomvothecoder>`__ in
   https://github.com/xCDAT/xcdat/pull/153
-  Update API Reference docs to list all APIs by `Tom Vo <https://github.com/tomvothecoder>`__ in
   https://github.com/xCDAT/xcdat/pull/155
-  Add ``config.yml`` for issue templates with link to discussions by
   `Tom Vo <https://github.com/tomvothecoder>`__ in https://github.com/xCDAT/xcdat/pull/176
-  Add FAQs page to docs by `Tom Vo <https://github.com/tomvothecoder>`__ in
   https://github.com/xCDAT/xcdat/pull/181
-  Fix syntax of code examples from PR #181 by `Tom Vo <https://github.com/tomvothecoder>`__ in
   https://github.com/xCDAT/xcdat/pull/182
-  Replace markdown issue templates with GitHub yml forms by
   `Tom Vo <https://github.com/tomvothecoder>`__ in https://github.com/xCDAT/xcdat/pull/186
-  Update ``README.rst``, ``index.rst``, and ``project_maintenance.rst``
   by `Tom Vo <https://github.com/tomvothecoder>`__ in https://github.com/xCDAT/xcdat/pull/211

Deprecations
~~~~~~~~~~~~

Internal Changes
~~~~~~~~~~~~~~~~

-  Update logger levels to debug by `Tom Vo <https://github.com/tomvothecoder>`__ in
   https://github.com/xCDAT/xcdat/pull/148
-  Update and remove logger debug messages by `Tom Vo <https://github.com/tomvothecoder>`__ in
   https://github.com/xCDAT/xcdat/pull/193

DevOps
~~~~~~

-  Add ``requires_dask`` decorator for tests by `Tom Vo <https://github.com/tomvothecoder>`__ in
   https://github.com/xCDAT/xcdat/pull/177
-  Update dependencies in ``setup.py`` and ``dev.yml`` by `Tom Vo <https://github.com/tomvothecoder>`__
   in https://github.com/xCDAT/xcdat/pull/174
-  Add matrix testing and ci specific conda env by `Tom Vo <https://github.com/tomvothecoder>`__ in
   https://github.com/xCDAT/xcdat/pull/178
-  Suppress xarray warning in test suite by `Tom Vo <https://github.com/tomvothecoder>`__ in
   https://github.com/xCDAT/xcdat/pull/179
-  Drop support for Python 3.7 by `Tom Vo <https://github.com/tomvothecoder>`__ in
   https://github.com/xCDAT/xcdat/pull/187
-  Update conda env dependencies by `Tom Vo <https://github.com/tomvothecoder>`__ in
   https://github.com/xCDAT/xcdat/pull/189
-  Add deps to ``pre-commit`` ``mypy`` and fix issues by `Tom Vo <https://github.com/tomvothecoder>`__
   in https://github.com/xCDAT/xcdat/pull/191
-  Add ``matplotlib`` to dev env, update ``ci.yml`` and add Python 3.10
   to build workflow by `Tom Vo <https://github.com/tomvothecoder>`__ in
   https://github.com/xCDAT/xcdat/pull/203
-  Replace conda with mamba in rtd build by `Tom Vo <https://github.com/tomvothecoder>`__ in
   https://github.com/xCDAT/xcdat/pull/209

New Contributors
~~~~~~~~~~~~~~~~

-  `Jill Chengzhu Zhang <https://github.com/chengzhuzhang>`__ made their first contribution in
   https://github.com/xCDAT/xcdat/pull/127
-  `Jiwoo Lee <https://github.com/lee1043>`__ made their first contribution in
   https://github.com/xCDAT/xcdat/pull/129
-  `Stephen Po-Chedley <https://github.com/pochedls>`__ made their first contribution in
   https://github.com/xCDAT/xcdat/pull/132

**Full Changelog**:
https://github.com/xCDAT/xcdat/compare/v0.1.0...v0.2.0

v0.1.0 (7 October 2021)
------------------------

New Features
~~~~~~~~~~~~

-  Add geospatial averaging API through
   ``DatasetSpatialAverageAccessor`` class by `Stephen Po-Chedley <https://github.com/pochedls>`__ and
   `Tom Vo <https://github.com/tomvothecoder>`__ in #87

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

DevOps
~~~~~~

-  100% code coverage (https://app.codecov.io/gh/xCDAT/xcdat)
-  GH Actions for CI/CD build (https://github.com/xCDAT/xcdat/actions)
-  Pytest and pytest-cov for test suite

**Full Changelog**: https://github.com/xCDAT/xcdat/commits/v0.1.0
