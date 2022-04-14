API Reference
=============

.. currentmodule:: xcdat

Top-level API
-------------

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


Dataset
------------------------------

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
    Dataset.spatial.spatial_avg
    Dataset.temporal.temporal_avg
    Dataset.temporal.departures
    Dataset.temporal.center_times
