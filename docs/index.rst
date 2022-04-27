xCDAT: Xarray Extended with Climate Data Analysis Tools
=======================================================

xCDAT is an extension of `xarray`_ for climate data analysis on structured grids.
It serves as a spiritual successor to the Community Data Analysis Tools (`CDAT`_) library.

.. _xarray: https://github.com/pydata/xarray
.. _CDAT: https://github.com/CDAT/cdat

Planned Features
-----------------

xCDAT aims to provide utilities for general xarray usage and climate domain specific features.

These features include:

- Support for CF compliant, E3SM non-CF compliant, and common metadata

- Extension of xarray's ``open_dataset()`` and ``open_mfdataset()`` to apply common operations

  - Generate bounds for all supported axes if they don't exist
  - Optional decoding of CF (via ``xarray``) and non-CF time units (via ``xcdat``)
  - Optional centering of time coordinate using time bounds
  - Optional conversion of longitudinal axes orientation

- Robust handling of coordinates and its associated bounds

  - Name-agnostic retrieval of CF compliant coordinates and bounds using ``cf_xarray``
  - Generating bounds for an axis or axes if they don't exist
  - Ability to operate on both [0, 360) and [-180, 180) longitudinal axis orientations

- Temporal averaging

  - Calculate time series averages, climatologies, and departures
  - Weighted or unweighted
  - Optional centering of time coordinates using time bounds

- Geospatial weighted averaging over rectilinear grid

  - Optional specification of regional domain

- Horizontal and vertical structured regridding

  - Operate on rectilinear and cuvilinear grids

- Parallelism of xCDAT features using Dask

Feature Criteria
~~~~~~~~~~~~~~~~

Features must meet the following criteria before being considered for implementation:

1. Feature is not implemented by ``xarray``
2. Feature is not implemented in other actively developed xarray-based packages

   - For example, ``cf_xarray`` already handles interpretation of CF convention attributes on xarray objects

3. Feature is not limited to specific use cases (e.g., data quality issues)
4. Feature is generally reusable
5. Feature is relatively simple and lightweight to implement and use

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
   GitHub Repository <https://github.com/xCDAT/xcdat>
