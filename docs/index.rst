XCDAT: Xarray Extended with Climate Data Analysis Tools
=======================================================

XCDAT is a Python library built on `xarray`_ for climate data analysis on structured grids.
It serves as a spiritual successor to the Community Data Analysis Tools (`CDAT`_) library with a focus on long-term maintainability and extensibility.

.. _xarray: https://github.com/pydata/xarray
.. _CDAT: https://github.com/CDAT/cdat

Planned Features
-----------------

XCDAT will provide climate domain specific features and utilities that are helpful for general xarray usage.

These features include:

- Support for metadata that is CF compliant, E3SM non-CF compliant, and common
- Robust handling of coordinates and its associated bounds

  - Name-agnostic retrieval of CF compliant coordinates and bounds using ``cf_xarray``
  - Generating a specific or all bounds for supported axes if they don't exist
  - Ability to operate on both [0, 360) and [-180, 180) longitudinal axis orientations

- Temporal averaging (weighted or unweighted)

  - Time series averaging and calculation of climatologies and anomalies
  - Use of time bounds for calculating weights
  - Optional centering of time coordinates using time bounds

- Geospatial weighted averaging over rectilinear grid

  - Optional specification of regional domain

- Horizontal and vertical structured regridding

  - Operate on rectilinear and cuvilinear grids

- Wrappers for opening datasets to apply common operations

  - Generate bounds for all supported axes if they don't exist
  - Decoding of CF and non-CF time units
  - Optional centering of time coordinate using time bounds
  - Optional conversion of longitudinal axes orientation

- Support for parallelism of XCDAT features using Dask

Criteria
~~~~~~~~

The features of this library must meet a set of criteria before being considered for implementation.

1. Climate domain functionality and/or general ``xarray`` utility isn't provided natively with ``xarray``
2. No existing xarray-based packages implement the feature, or the implementation doesn't meet the XCDAT team's defined requirements
3. Feature can be relatively simple to implement and not overly-flexible
4. Feature is often reused

Acknowledgement
---------------

This software is jointly developed by scientists and developers from the Energy Exascale Earth System Model (`E3SM`_) Project and Program for Climate Model Diagnosis and Intercomparison (`PCMDI`_). The work is performed for the E3SM project, which is sponsored by Earth System Model Development (`ESMD`_) program, and the Simplifying ESM Analysis Through Standards (SEATS) project, which is sponsored by the Regional and Global Model Analysis (`RGMA`_) program. ESMD and RGMA are programs for the Earth and Environmental Systems Sciences Division (`EESSD`_) in the Office of Biological and Environmental Research (`BER`_) within the `Department of Energy`_'s `Office of Science`_.

.. _E3SM: https://e3sm.org/
.. _PCMDI: https://pcmdi.llnl.gov/
.. _ESMD: https://climatemodeling.science.energy.gov/program/earth-system-model-development
.. _RGMA: https://climatemodeling.science.energy.gov/program/regional-global-model-analysis
.. _EESSD: https://science.osti.gov/ber/Research/eessd
.. _BER: https://science.osti.gov/ber
.. _Department of Energy: https://www.energy.gov/
.. _Office of Science: https://science.osti.gov/

License
-------

SPDX-License-Identifier: (Apache-2.0)

See `LICENSE <LICENSE>`_ for details

`LLNL-CODE-819717`

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: For users

   Getting Started <getting_started>
   API Reference <api>
   Xarray API Reference <https://xarray.pydata.org/en/stable/api.html>
   Frequently Asked Questions <faqs>

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: For developers/contributors

   Contributing Guide <contributing>
   Project Maintenance <project_maintenance>
   What’s New <history>
   Team <authors>
   GitHub Repository <https://github.com/XCDAT/xcdat>
