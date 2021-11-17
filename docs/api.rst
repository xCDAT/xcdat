API Reference
=============

.. currentmodule:: xcdat

Top-level API
-------------

.. autosummary::
    :toctree: generated/

    dataset.open_dataset
    dataset.open_mfdataset
    dataset.infer_or_keep_var
    dataset.decode_time_units
    dataset.get_inferred_var

.. currentmodule:: xarray

Dataset (``xcdat`` accessor)
----------------------------

.. _dsattr_1:

Attributes
~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_attribute.rst

    Dataset.xcdat.bounds

.. _dsmeth_1:

Methods
~~~~~~~

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_method.rst

    Dataset.xcdat.add_bounds
    Dataset.xcdat.get_bounds
    Dataset.xcdat.fill_missing_bounds
    Dataset.xcdat.spatial_avg


Dataset (individual accessors)
-------------------------------

.. _dsattr_2:

Attributes
~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_attribute.rst

    Dataset.bounds.bounds

.. _dsmeth_2:

Methods
~~~~~~~

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_method.rst

    Dataset.bounds.add_bounds
    Dataset.bounds.get_bounds
    Dataset.bounds.fill_missing_bounds
    Dataset.spatial.spatial_avg
