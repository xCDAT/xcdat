.. raw:: html

   <p align="center">
      <img src="./docs/_static/xcdat-logo.png" alt="xCDAT logo" style="width:450px;"/>
   </p>

.. container::

   .. raw:: html

      <h3 align="center">

   Xarray Climate Data Analysis Tools

   +--------------------+------------------------------------------------------+
   |                    | Badges                                               |
   +====================+======================================================+
   | Distribution       | |conda-forge| |platforms| |conda-downloads|          |
   +--------------------+------------------------------------------------------+
   | Citation           | |zenodo-doi|                                         |
   +--------------------+------------------------------------------------------+
   | DevOps             | |CI/CD Build Workflow| |codecov| |docs|              |
   +--------------------+------------------------------------------------------+
   | Quality Assurance  | |pre-commit| |black| |flake8| |mypy|                 |
   +--------------------+------------------------------------------------------+

   .. raw:: html

      </h3>

.. |conda-forge| image:: https://img.shields.io/conda/vn/conda-forge/xcdat.svg
   :target: https://anaconda.org/conda-forge/xcdat
.. |platforms| image:: https://img.shields.io/conda/pn/conda-forge/xcdat.svg
   :target: https://anaconda.org/conda-forge/xcdat
.. |conda-downloads| image:: https://anaconda.org/conda-forge/xcdat/badges/downloads.svg
   :target: https://anaconda.org/conda-forge/xcdat
.. |zenodo-doi| image:: https://zenodo.org/badge/354103918.svg
   :target: https://zenodo.org/badge/latestdoi/354103918
.. |CI/CD Build Workflow| image:: https://github.com/xCDAT/xcdat/actions/workflows/build_workflow.yml/badge.svg
   :target: https://github.com/xCDAT/xcdat/actions/workflows/build_workflow.yml
.. |docs| image:: https://readthedocs.org/projects/xcdat/badge/?version=latest
   :target: https://xcdat.readthedocs.io/en/latest/?badge=latest
.. |codecov| image:: https://codecov.io/gh/xCDAT/xcdat/branch/main/graph/badge.svg?token=UYF6BAURTH
   :target: https://codecov.io/gh/xCDAT/xcdat
.. |pre-commit| image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
   :target: https://github.com/pre-commit/pre-commit
.. |black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
.. |flake8| image:: https://img.shields.io/badge/flake8-enabled-green
   :target: https://github.com/PyCQA/flake8
.. |mypy| image:: http://www.mypy-lang.org/static/mypy_badge.svg
   :target: http://mypy-lang.org/

xCDAT is an extension of `xarray`_ for climate data analysis on structured grids. It
serves as a modern successor to the Community Data Analysis Tools (`CDAT`_) library.

**Useful links**:
`Documentation <https://xcdat.readthedocs.io>`__ |
`Code Repository <https://github.com/xCDAT/xcdat>`__ |
`Issues <https://github.com/xCDAT/xcdat/issues>`__ |
`Discussions <https://github.com/xCDAT/xcdat/discussions>`__ |
`Releases <https://github.com/xCDAT/xcdat/releases>`__ |
`Mailing List <https://groups.google.com/g/xcdat>`__

Project Motivation
------------------

The goal of xCDAT is to provide generalizable features and utilities for simple and
robust analysis of climate data. xCDAT's design philosophy is focused on reducing the
overhead required to accomplish certain tasks in xarray. xCDAT aims to be compatible
with structured grids that are `CF-compliant`_ (e.g., CMIP6). Some key xCDAT features
are inspired by or ported from the core CDAT library, while others leverage powerful
libraries in the xarray ecosystem (e.g., `xESMF`_, `xgcm`_, `cf_xarray`_) to deliver
robust APIs.

The xCDAT core team's mission is to provide a maintainable and extensible package
that serves the needs of the climate community in the long-term. We are excited
to be working on this project and hope to have you onboard!

.. _CF-compliant: https://cfconventions.org/
.. _xarray: https://github.com/pydata/xarray
.. _CDAT: https://github.com/CDAT/cdat

Getting Started
---------------

The best resource for getting started is the `xCDAT documentation website`_.
Our documentation provides general guidance for setting up xCDAT in an Anaconda
environment on your local `computer`_ or on an `HPC/Jupyter`_ environment. We also
include an `API Overview`_ and `Gallery`_ to highlight xCDAT functionality.

.. _xCDAT documentation website: https://xcdat.readthedocs.io/en/stable/
.. _computer: https://xcdat.readthedocs.io/en/stable/getting-started.html
.. _HPC/Jupyter: https://xcdat.readthedocs.io/en/stable/getting-started-hpc-jupyter.html
.. _API Overview: https://xcdat.readthedocs.io/en/stable/api.html
.. _Gallery: https://xcdat.readthedocs.io/en/stable/gallery.html

Community
---------

xCDAT is a community-driven open source project. We encourage discussion on topics such
as version releases, feature suggestions, and architecture design on the
`GitHub Discussions`_ page.

Subscribe to our `mailing list`_ for news and announcements related to xCDAT,
such as software version releases or future roadmap plans.

Please note that xCDAT has a `Code of Conduct`_. By participating in the xCDAT
community, you agree to abide by its rules.

.. _GitHub Discussions: https://github.com/xCDAT/xcdat/discussions
.. _Code of Conduct: CODE-OF-CONDUCT.rst
.. _mailing list: https://groups.google.com/g/xcdat

Contributing
------------

We welcome and appreciate contributions to xCDAT. Users and contributors can view and
open issues on our `GitHub Issue Tracker`_.

For more instructions on how to contribute, please checkout our `Contributing Guide`_.

.. _GitHub Issue Tracker: https://github.com/xCDAT/xcdat/issues
.. _Contributing Guide: https://xcdat.readthedocs.io/en/stable/contributing.html

Features
--------

* Extension of xarray's ``open_dataset()`` and ``open_mfdataset()`` with post-processing options

  * Generate bounds for axes supported by ``xcdat`` if they don't exist in the Dataset
  * Optional selection of single data variable to keep in the Dataset (bounds are also
    kept if they exist)
  * Optional decoding of time coordinates

    * In addition to CF time units, also decodes common non-CF time units
      ("months since ...", "years since ...")

  * Optional centering of time coordinates using time bounds
  * Optional conversion of longitudinal axis orientation between [0, 360) and [-180, 180)

* Temporal averaging

  * Time series averages (single snapshot and grouped), climatologies, and departures
  * Weighted or unweighted
  * Optional seasonal configuration (e.g., DJF vs. JFD, custom seasons)

* Geospatial weighted averaging

  * Supports rectilinear grid
  * Optional specification of regional domain

* Horizontal structured regridding

  * Supports rectilinear and curvilinear grids
  * Extends the `xESMF`_ horizontal regridding API
  * Python implementation of  `regrid2`_ for handling cartesian latitude longitude grids

* Vertical structured regridding

  * Support rectilinear and curvilinear grids
  * Extends the `xgcm`_ vertical regridding API

Things We Are Striving For
--------------------------

* xCDAT supports CF compliant datasets, but will also strive to support datasets with
  common non-CF compliant metadata (e.g., time units in "months since ..." or "years
  since ...")

  * xCDAT leverages `cf_xarray`_ to interpret CF attributes on ``xarray`` objects
  * Refer to `CF Convention`_ for more information on CF attributes

* Robust handling of dimensions and their coordinates and coordinate bounds

  * Coordinate variables are retrieved with ``cf_xarray`` using CF axis names or
    coordinate names found in xarray object attributes. Refer to `Metadata Interpretation`_
    for more information.
  * Bounds are retrieved with ``cf_xarray`` using the ``"bounds"`` attr
  * Ability to operate on both longitudinal axis orientations, [0, 360) and [-180, 180)

* Support for parallelism using `dask`_ where it is both possible and makes sense

.. _Metadata Interpretation: https://xcdat.readthedocs.io/en/stable/faqs.html#metadata-interpretation
.. _xESMF: https://pangeo-xesmf.readthedocs.io/en/latest/
.. _regrid2: https://cdms.readthedocs.io/en/latest/regrid2.html
.. _xgcm: https://xgcm.readthedocs.io/en/latest/index.html
.. _dask: https://dask.org/
.. _cf_xarray: https://cf-xarray.readthedocs.io/en/latest/index.html
.. _CF convention: http://cfconventions.org/

Releases
--------
xCDAT (released as ``xcdat``) follows a feedback-driven release cycle using continuous
integration/continuous deployment. Software releases are performed based on the bandwidth
of the development team, the needs of the community, and the priority of bug fixes or
feature updates.

After releases are performed on `GitHub Releases`_, the corresponding ``xcdat`` package
version will be available to download through Anaconda `conda-forge`_ usually within a day.

Subscribe to our `mailing list`_ to stay notified of new releases.

.. _conda-forge: https://anaconda.org/conda-forge/xcdat
.. _GitHub Releases: https://anaconda.org/conda-forge/xcdat

Useful Resources
-----------------

We highly encourage you to checkout the awesome resources below to learn more about
Xarray and Xarray usage in climate science!

- `Official Xarray Tutorials <https://tutorial.xarray.dev/intro.html>`_
- `Xarray GitHub Discussion Forum <https://github.com/pydata/xarray/discussions>`_
- `Pangeo Forum <https://foundations.projectpythia.org/core/xarray.html>`_
- `Project Pythia <https://foundations.projectpythia.org/core/xarray.html>`_

Acknowledgement
---------------

Huge thank you to all of the xCDAT `contributors`_!

.. _contributors: https://github.com/xCDAT/xcdat/graphs/contributors

xCDAT is jointly developed by scientists and developers from the Energy Exascale
Earth System Model (`E3SM`_) Project and Program for Climate Model Diagnosis and
Intercomparison (`PCMDI`_). The work is performed for the E3SM project, which is
sponsored by Earth System Model Development (`ESMD`_) program, and the Simplifying ESM
Analysis Through Standards (`SEATS`_) project, which is sponsored by the Regional and
Global Model Analysis (`RGMA`_) program. ESMD and RGMA are programs for the Earth and
Environmental Systems Sciences Division (`EESSD`_) in the Office of Biological and
Environmental Research (`BER`_) within the `Department of Energy`_'s `Office of Science`_.

.. _E3SM: https://e3sm.org/
.. _PCMDI: https://pcmdi.llnl.gov/
.. _SEATS: https://www.seatstandards.org/
.. _ESMD: https://climatemodeling.science.energy.gov/program/earth-system-model-development
.. _RGMA: https://climatemodeling.science.energy.gov/program/regional-global-model-analysis
.. _EESSD: https://science.osti.gov/ber/Research/eessd
.. _BER: https://science.osti.gov/ber
.. _Department of Energy: https://www.energy.gov/
.. _Office of Science: https://science.osti.gov/

Projects Using xCDAT
--------------------

xCDAT is actively being integrated as a core component of the `Program for Climate Model
Diagnosis and Intercomparison (PCMDI) Metrics Package`_ and the `Energy Exascale Earth
System Model Diagnostics (E3SM) Package`_. xCDAT is also included in the `E3SM Unified
Anaconda Environment`_` that is deployed on various U.S. Department of Energy
supercomputers to run E3SM software tools.

.. _Program for Climate Model Diagnosis and Intercomparison (PCMDI) Metrics Package: https://pcmdi.github.io/pcmdi_metrics/
.. _Energy Exascale Earth System Model Diagnostics (E3SM) Package: https://e3sm-project.github.io/e3sm_diags/_build/html/main/index.html
.. _E3SM Unified Anaconda Environment: https://e3sm.org/resources/tools/other-tools/e3sm-unified-environment/

License
-------

xCDAT is licensed under the terms of the Apache License (Version 2.0 with LLVM exception).

All new contributions must be made under the Apache-2.0 with LLVM exception license.

See `LICENSE`_ and `NOTICE`_ for details.

.. _LICENSE: https://github.com/xCDAT/xcdat/blob/main/LICENSE
.. _NOTICE: https://github.com/xCDAT/xcdat/blob/main/NOTICE

SPDX-License-Identifier: Apache-2.0

``LLNL-CODE-846944``
