.. raw:: html

   <p align="center">
      <img src="./docs/_static/xcdat_logo.png" alt="xCDAT logo"/>
   </p>

.. container::

   .. raw:: html

      <h3 align="center">

   Xarray Climate Data Analysis Tools

   +--------------------+------------------------------------------------------+
   |                    | Badges                                               |
   +====================+======================================================+
   | Distribution       | |conda| |conda-forge| |platforms| |conda-downloads|  |
   +--------------------+------------------------------------------------------+
   | Citation           | |zenodo-doi|                                         |
   +--------------------+------------------------------------------------------+
   | DevOps             | |CI/CD Build Workflow| |codecov| |docs|              |
   +--------------------+------------------------------------------------------+
   | Quality Assurance  | |pre-commit| |black| |flake8| |mypy|                 |
   +--------------------+------------------------------------------------------+

   .. raw:: html

      </h3>

.. |conda| image:: https://anaconda.org/conda-forge/xcdat/badges/installer/conda.svg
   :target: https://anaconda.org/conda-forge/xcdat
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
serves as the spiritual successor to the Community Data Analysis Tools (`CDAT`_)
library.

The goal of xCDAT is to provide generalizable climate domain features and utilities
that streamline the developer experience for data analysis code. xCDAT's design
philosophy is to reduce the complexity and overhead required by the user to accomplish
specific tasks in xarray. Some xCDAT features are inspired by or ported from core CDAT
functionalities, while others leverage powerful libraries in the xarray ecosystem
(e.g., `xESMF`_ and `cf_xarray`_) to deliver robust APIs.

The xCDAT core team's mission is to provide a maintainable and extensible package
that serves the needs of the climate community in the long-term. We are excited
to be working on this project and hope to have you onboard!

.. _xarray: https://github.com/pydata/xarray
.. _CDAT: https://github.com/CDAT/cdat

Getting Started
---------------

The best resource for getting started with xCDAT is via our `documentation page`_.
There, we provide general guidance for setting up your Anaconda environment on your
local `computer`_ or on an `HPC/Jupyter`_ environment. We also include an
`API Overview`_ and `Gallery`_ to highlight xCDAT functionality.

xCDAT invites discussion on version releases, architecture, new feature suggestions, and
other topics on the `GitHub Discussions`_ page. Users and contributors can also view and
open issues on our `GitHub Issue Tracker`_.

We welcome and appreciate contributions to xCDAT. If you'd like to help improve xCDAT,
please checkout our `Contributing Guide`_.

.. _documentation page: https://xcdat.readthedocs.io/en/latest/
.. _computer: https://xcdat.readthedocs.io/en/latest/getting-started.html
.. _HPC/Jupyter: https://xcdat.readthedocs.io/en/latest/getting-started-hpc-jupyter.html
.. _API Overview: https://xcdat.readthedocs.io/en/latest/api.html
.. _Gallery: https://xcdat.readthedocs.io/en/latest/gallery.html
.. _GitHub Discussions: https://github.com/xCDAT/xcdat/discussions
.. _GitHub Issue Tracker: https://github.com/xCDAT/xcdat/issues
.. _Contributing Guide: https://xcdat.readthedocs.io/en/latest/contributing.html

Stay up to Date with Releases
-----------------------------
xCDAT (released as ``xcdat``) follows a rapid release cycle with continuous
integration/continuous deployment. As a result, releases are performed relatively
frequently based on the priority of commits. The xCDAT core team reviews commits every
two weeks to determine if they warrant a release.

After releases are performed on `GitHub Releases`_, the corresponding ``xcdat`` package
version will be available to download through `conda-forge`_ within 30 minutes to 1
hour.

To be notified of releases through GitHub:

1. Go to the ``xcdat`` repository homepage on GitHub (https://github.com/xCDAT/xcdat)
2. Click the "Watch" button on the upper right-hand corner of the page.

   .. image:: docs/_static/github-watch-releases-1.png

3. Click "Custom" and checkmark "Releases", then click "Apply".

   .. image:: docs/_static/github-watch-releases-2.png

.. _conda-forge: https://anaconda.org/conda-forge/xcdat
.. _GitHub Releases: https://anaconda.org/conda-forge/xcdat

Available Features
------------------

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
  * Python implementation of `regrid2`_ for handling cartesian latitude longitude grids
  * API that wraps `xESMF`_

Planned Features
----------------

* Vertical structured regridding

  * Support rectilinear and curvilinear grids

Things we are striving for:
---------------------------

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

.. _Metadata Interpretation: docs/faqs.rst#metadata-interpretation
.. _regrid2: https://cdms.readthedocs.io/en/latest/regrid2.html
.. _xESMF: https://pangeo-xesmf.readthedocs.io/en/latest/
.. _dask: https://dask.org/
.. _cf_xarray: https://cf-xarray.readthedocs.io/en/latest/index.html
.. _CF convention: http://cfconventions.org/

Acknowledgement
---------------

This software is jointly developed by scientists and developers from the Energy Exascale
Earth System Model (`E3SM`_) Project and Program for Climate Model Diagnosis and
Intercomparison (`PCMDI`_). The work is performed for the E3SM project, which is
sponsored by Earth System Model Development (`ESMD`_) program, and the Simplifying ESM
Analysis Through Standards (SEATS) project, which is sponsored by the Regional and
Global Model Analysis (`RGMA`_) program. ESMD and RGMA are programs for the Earth and
Environmental Systems Sciences Division (`EESSD`_) in the Office of Biological and
Environmental Research (`BER`_) within the `Department of Energy`_'s `Office of Science`_.

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
