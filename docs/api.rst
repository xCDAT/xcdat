API Reference
=============

Overview
--------

Most public ``xcdat`` APIs operate on ``xarray.Dataset`` objects. ``xcdat`` follows this design pattern because coordinate variable bounds are often required to perform robust calculations.
Currently, coordinate variable bounds can only be stored on ``Dataset`` objects and not ``DataArray`` objects. Refer to this `issue`_ for more information.

.. _issue: https://github.com/pydata/xarray/issues/1475

.. currentmodule:: xcdat

Top-level API Functions
-----------------------

Below is a list of top-level API functions available in ``xcdat``.

.. autosummary::
    :toctree: generated/

    axis.swap_lon_axis
    dataset.open_dataset
    dataset.open_mfdataset
    dataset.decode_non_cf_time
    utils.compare_datasets

Accessors
---------

What are accessors?
~~~~~~~~~~~~~~~~~~~

``xcdat`` provides ``Dataset`` accessors, which are implicit namespaces for custom functionality that clearly identifies it as separate from built-in xarray methods.
``xcdat`` implements accessors to extend xarray with custom functionality because it is the officially recommended and most common practice (over sub-classing).

In the example below, custom spatial functionality is exposed by chaining the ``spatial`` accessor attribute to the ``Dataset`` object.
This chaining enables access to the underlying spatial ``average()`` method.

.. figure:: _static/accessor_api.svg
   :alt: Accessor API Diagram


How do I use ``xcdat`` accessors?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, import the package:

.. code-block:: python

    >>> from xcdat

Then open up a dataset file as a ``Dataset`` object:

.. code-block:: python

    >>> ds = xcdat.open_dataset("path/to/file", data_var="ts")


Now chain the accessor attribute to the ``Dataset`` to expose the accessor class attributes, methods, or properties:

.. code-block:: python

    >>> ds = ds.spatial.average("ts", axis=["X", "Y"])

.. note::

   Accessors are created once per Dataset instance. New instances, like those
   created from arithmetic operations will have new accessors created.

Classes
~~~~~~~

.. autosummary::
   :toctree: generated/

    xcdat.bounds.BoundsAccessor
    xcdat.spatial.SpatialAccessor
    xcdat.temporal.TemporalAccessor
    xcdat.regridder.accessor.RegridderAccessor
    xcdat.regridder.regrid2.Regrid2Regridder
    xcdat.regridder.xesmf.XESMFRegridder

.. currentmodule:: xarray

Attributes
~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_attribute.rst

    Dataset.bounds.map
    Dataset.bounds.keys
    Dataset.regridder.grid

.. _dsattr_1:

Methods
~~~~~~~

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_method.rst

    Dataset.bounds.add_bounds
    Dataset.bounds.get_bounds
    Dataset.bounds.add_missing_bounds
    Dataset.spatial.average
    Dataset.temporal.average
    Dataset.temporal.climatology
    Dataset.temporal.departures
    Dataset.temporal.center_times
    Dataset.regridder.horizontal
    Dataset.regridder.horizontal_xesmf
    Dataset.regridder.horizontal_regrid2

.. _dsmeth_1:
