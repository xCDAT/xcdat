xCDAT: Xarray Climate Data Analysis Tools
=========================================

xCDAT is an extension of `xarray`_ for climate data analysis on structured grids. It serves as a spiritual successor to the Community Data Analysis Tools (`CDAT`_) library.

The goal of xCDAT is to provide generalizable climate domain features and general utilities in xarray, which includes porting some core CDAT functionalities. xCDAT leverages several powerful libraries in the xarray ecosystem (e.g., `xESMF`_ and `cf_xarray`_) to deliver robust APIs. The xCDAT core team is aiming to provide a maintainable and extensible package that serves the needs of the climate community in the long-term.

A major design philosophy of xCDAT is streamlining the user experience while developing code to analyze climate data. This means reducing the complexity and number of lines required to achieve certain features with xarray.

.. _xarray: https://github.com/pydata/xarray
.. _CDAT: https://github.com/CDAT/cdat

Getting Started
---------------

This `documentation page <https://xcdat.readthedocs.io/en/latest/>`__ provides guidance for setting up your environment on your `computer <https://xcdat.readthedocs.io/en/latest/getting-started.html>`_ generally or on an `HPC/Jupyter <https://xcdat.readthedocs.io/en/latest/getting-started-hpc-jupyter.html>`_ environment. We also include an `API Overview <https://xcdat.readthedocs.io/en/latest/api.html>`_ and `Gallery <https://xcdat.readthedocs.io/en/latest/gallery.html>`_ to highlight xCDAT functionality.

xCDAT invites discussion on version releases, architecture, new feature suggestions, and other topics on the `GitHub discussion <https://github.com/xCDAT/xcdat/discussions>`_ page. Users and contributors can also view and open issues on our `GitHub Issue Tracker <https://github.com/xCDAT/xcdat/issues>`_.

We welcome and appreciate contributions to xCDAT. If you'd like to help improve xCDAT, please checkout our `Contributing Guide <https://xcdat.readthedocs.io/en/latest/contributing.html>`_.

Stay up to Date with Releases
-----------------------------
xCDAT (released as ``xcdat``) follows a rapid release cycle with continuous
integration/continuous deployment. This means releases are made relatively frequently
based on the importance of commits. The xCDAT core team reviews commits every two weeks
to determine if they warrant a release.

To be notified of releases through GitHub:

1. Go to the ``xcdat`` repository homepage on GitHub (https://github.com/xCDAT/xcdat)
2. Click the "Watch" button on the upper right-hand corner of the page.

   .. image:: _static/github-watch-releases-1.png

3. Click "Custom" and checkmark "Releases", then click "Apply".

   .. image:: _static/github-watch-releases-2.png

Available Features
------------------

* Extension of xarray's ``open_dataset()`` and ``open_mfdataset()`` with post-processing options

  * Generate bounds for axes supported by ``xcdat`` if they don't exist in the Dataset
  * Optional selection of single data variable to keep in the Dataset (bounds are also
    kept if they exist)
  * Optional decoding of time coordinates

    * In addition to CF time units, also decodes common non-CF time units ("months since ...",
      "years since ...")

  * Optional centering of time coordinates using time bounds
  * Optional conversion of longitudinal axis orientation between [0, 360) and [-180, 180)

* Temporal averaging

  * Time series averages (single snapshot and grouped), climatologies, and departures
  * Weighted or unweighted
  * Optional seasonal configuration (e.g., DJF vs. JFD, custom seasons)

* Geospatial weighted averaging

  * Support rectilinear grid
  * Optional specification of regional domain

* Horizontal structured regridding

  * Support rectilinear and cuvilinear grids
  * Python implementation of `regrid2`_ for handling cartesian latitude longitude grids
  * API that wraps `xESMF`_

Planned Features
----------------

* Vertical structured regridding

  * Support rectilinear and cuvilinear grids

Things we are striving for:
---------------------------

* Support for CF compliant, E3SM non-CF compliant, and common metadata

  * xCDAT primarily focuses on datasets that follow the `CF convention`_.
  * xCDAT leverages `cf_xarray`_ to interpret CF attributes on ``xarray`` objects
  * Accomodations for specific non-CF compliant situations will be considered on a
    case-by-case basis.

* Robust handling of dimensions and their coordinates and coordinate bounds

  * Coordinate variables are retrieved with ``cf_xarray`` using CF axis names or
    coordinate names found in xarray object attributes. Refer to :ref:`Metadata Interpretation`.
    for more information.
  * Bounds are retrieved with ``cf_xarray`` using the ``"bounds"`` attr
  * Ability to operate on both longitudinal axis orientations, [0, 360) and [-180, 180)

* Support for parallelism using `dask`_ where it is both possible and makes sense

.. _regrid2: https://cdms.readthedocs.io/en/latest/regrid2.html
.. _xESMF: https://pangeo-xesmf.readthedocs.io/en/latest/
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
   xCDAT on HPC / Jupyter <getting-started-hpc-jupyter>
   Gallery <gallery>
   API Reference <api>
   Changelog <history>
   Frequently Asked Questions <faqs>

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: For developers/contributors

   Contributing Guide <contributing>
   Project Maintenance <project-maintenance>
   Team <authors>

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Community

   GitHub Discussions <https://github.com/xCDAT/xcdat/discussions>
