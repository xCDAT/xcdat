API Reference
=============

.. currentmodule:: xcdat

Top-level API Functions
-----------------------

Below is a list of top-level API functions that are available in ``xcdat``.

.. autosummary::
    :toctree: generated/

    dataset.open_dataset
    dataset.open_mfdataset
    dataset.has_cf_compliant_time
    dataset.decode_non_cf_time
    dataset.swap_lon_axis
    dataset.get_data_var
    utils.compare_datasets

.. currentmodule:: xarray


xarray.Dataset Accessors
------------------------

``xcdat`` accessors provide implicit namespaces for custom functionality that clearly identifies it as separate from built-in xarray methods.
These accessors operate directly on  xarray objects, such as ``xarray.Dataset``.

.. note::

   Accessors are created once per DataArray and Dataset instance. New
   instances, like those created from arithmetic operations or when accessing
   a DataArray from a Dataset (ex. ``ds[var_name]``), will have new
   accessors created.

More information on accessors can be found here: https://docs.xarray.dev/en/stable/internals/extending-xarray.html#extending-xarray

.. _dsattr_1:

Attributes
~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_attribute.rst

    Dataset.bounds.bounds

.. _dsmeth_1:

Methods
~~~~~~~

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_method.rst

    Dataset.bounds.add_bounds
    Dataset.bounds.get_bounds
    Dataset.bounds.add_missing_bounds
    Dataset.spatial.average
    Dataset.temporal.temporal_avg
    Dataset.temporal.departures
    Dataset.temporal.center_times
