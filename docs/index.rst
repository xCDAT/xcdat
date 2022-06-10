xCDAT: Xarray Climate Data Analysis Tools
=======================================================

xCDAT is an extension of `xarray`_ for climate data analysis on structured grids.
It serves as a spiritual successor to the Community Data Analysis Tools (`CDAT`_) library.

.. _xarray: https://github.com/pydata/xarray
.. _CDAT: https://github.com/CDAT/cdat

Planned Features
-----------------

The goal of xCDAT is to provide climate domain specific features and general utilities with xarray.

Initial features include:

* Extension of xarray's ``open_dataset()`` and ``open_mfdataset()`` with post-processing options

  * Generate bounds for axes supported by ``xcdat`` if they don't exist in the Dataset
  * Optional decoding non-CF time units, in addition to CF time units (already supported in ``xarray``)
  * Optional centering of time coordinates using time bounds
  * Optional conversion of longitudinal axis orientation between [0, 360) and [-180, 180)

* Temporal averaging

  * Time series averages (single snapshot and grouped), climatologies, and departures
  * Weighted or unweighted
  * Optional seasonal configuration

* Geospatial weighted averaging

  * Support rectilinear grid
  * Optional specification of regional domain

* Horizontal structured regridding

  * Support rectilinear and cuvilinear grids
  * Python implementation of `regrid2`_ for handling cartesian latitude longitude grids
  * API that wraps `xesmf`_ with utilities to handle edge cases

* Vertical structured regridding

  * Support rectilinear and cuvilinear grids

Things we are striving for:

* Support for CF compliant, E3SM non-CF compliant, and common metadata

  * Leverage `cf_xarray`_ to interpret `CF convention`_ attributes on ``xarray`` objects

* Robust handling of coordinates and its associated bounds

  * Coordinates and bounds are retrieved with ``cf_xarray`` using the standard ``axis`` and ``coordinate`` name attributes
  * Ability to operate on both longitudinal axis orientations, [0, 360) and [-180, 180)

* Support for parallelism using `dask`_ where it is both possible and makes sense

.. _regrid2: https://cdms.readthedocs.io/en/latest/regrid2.html
.. _xesmf: https://pangeo-xesmf.readthedocs.io/en/latest/
.. _dask: https://dask.org/
.. _cf_xarray: https://cf-xarray.readthedocs.io/en/latest/index.html
.. _CF convention: http://cfconventions.org/

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

   Getting Started <getting-started>
   Gallery <gallery>
   API Reference <api>
   Frequently Asked Questions <faqs>

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: For developers/contributors

   Contributing Guide <contributing>
   Project Maintenance <project-maintenance>
   What’s New <history>
   Team <authors>

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Community

   GitHub Discussions <https://github.com/xCDAT/xcdat/discussions>
