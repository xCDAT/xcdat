.. raw:: html

   <p align="center">
      <img src="./docs/_static/xcdat_logo.png" alt="XCDAT logo"/>
   </p>

.. container::

   .. raw:: html

      <h3 align="center">

   Xarray Extended with Climate Data Analysis Tools (XCDAT)

   |conda-forge| |platforms| |CI/CD Build Workflow| |docs| |Codecov|

   |pre-commit| |Code style: black| |flake8| |Checked with mypy|


   .. raw:: html

      </h3>

.. |conda-forge| image:: https://img.shields.io/conda/vn/conda-forge/xcdat.svg
   :target: https://anaconda.org/conda-forge/xcdat
.. |platforms| image:: https://img.shields.io/conda/pn/conda-forge/xcdat.svg
   :target: https://anaconda.org/conda-forge/xcdat
.. |CI/CD Build Workflow| image:: https://github.com/XCDAT/xcdat/actions/workflows/build_workflow.yml/badge.svg
   :target: https://github.com/XCDAT/xcdat/actions/workflows/build_workflow.yml
.. |docs| image:: https://readthedocs.org/projects/xcdat/badge/?version=latest
   :target: https://xcdat.readthedocs.io/en/latest/?badge=latest
.. |Codecov| image:: https://codecov.io/gh/XCDAT/xcdat/branch/main/graph/badge.svg?token=UYF6BAURTH
   :target: https://codecov.io/gh/XCDAT/xcdat
.. |pre-commit| image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
   :target: https://github.com/pre-commit/pre-commit
.. |Code style: black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
.. |flake8| image:: https://img.shields.io/badge/flake8-enabled-green
   :target: https://github.com/PyCQA/flake8
.. |Checked with mypy| image:: http://www.mypy-lang.org/static/mypy_badge.svg
   :target: http://mypy-lang.org/


XCDAT is a Python library built on xarray for climate data analysis on structured grids.
It serves as a spiritual successor to the Community Data Analysis Tools (CDAT) package, with a focus on long-term maintainability and extensibility.

Planned Features
-----------------

XCDAT will include climate domain specific features and utilities that are helpful for general xarray usage.

- Robust handling of CF compliant, common, and E3SM non-CF compliant metadata
- Robust handling of coordinates and its associated bounds

  - Name-agnostic retrieval of bounds and coordinates (utilizes CF-xarray)
  - Functions operate regardless of longitudinal axis orientation (0 to 360 or -180 to 180)
  - Adding bounds and/or filling of all missing bounds

- Temporal averaging

  - Time series averaging and calculation of climatologies and anomalies
  - Use of time bounds for calculating weights
  - Optional centering of time using time bounds (if not already performed)

- Geospatial averaging (weighted) over rectilinear grid

  - Optional specification of regional domain

- Horizontal and vertical regridding
- Wrappers for opening datasets using ``xr.open_dataset()`` and ``xr.open_mfdataset()`` to apply common operations

  - Fill missing bounds
  - Decoding of CF and non-CF time units and coordinate points
  - Optional centering of time axes using time bounds
  - Optional conversion of longitudinal axes orientation

- Support for parallelism of XCDAT features using Dask

Criteria
~~~~~~~~~~~~~~~~

The features of this package must meet a set of criteria before being considered for implementation.

1. Climate domain functionality and/or general xarray utility isn't provided natively with xarray
2. No existing xarray-based packages implement the feature, or the implementation doesn't meet the XCDAT team's defined requirements.
3. Function can be relatively simple to implement and not overly-flexible.
4. Functions are often reused.

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
