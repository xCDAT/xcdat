.. highlight:: shell

===============
Getting Started
===============

Prerequisites
-------------

1. Familiarity with ``xarray``, since this package is an extension of it

   - We highly recommend visiting the `xarray tutorial`_ and `xarray documentation`_
     pages if you aren't familiar with ``xarray``.

2. xCDAT is distributed available on Anaconda. We recommend installing `mamba`_, which
   is a drop-in replacement of ``conda``.

   We recommend following these steps to install Mambaforge (Linux example):

   .. code-block:: console

      wget https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh

      bash ./Mambaforge-Linux-x86_64.sh

   When you see: ::

      by running conda init? [yes|no]
      [no] >>> yes

   respond with ``yes`` so ``conda`` and ``mamba`` commands are available on
   initializing a new bash terminal.

.. _mamba: https://mamba.readthedocs.io/en/latest/index.html
.. _xarray tutorial: https://tutorial.xarray.dev/intro.html
.. _xarray documentation: https://docs.xarray.dev/en/stable/getting-started-guide/index.html

Installation
------------

1. Create a mamba environment from scratch with ``xcdat`` (`mamba create`_)

   We recommend using the mamba environment creation procedure to install ``xcdat``.
   The advantage with following this approach is that mamba will attempt to resolve
   dependencies (e.g. ``python >= 3.9``) for compatibility.

   To create an ``xcdat`` environment with ``xesmf`` (a recommended dependency), run:

   .. code-block:: console

       >>> mamba create -n <ENV_NAME> -c conda-forge xcdat xesmf
       >>> mamba activate <ENV_NAME>

   Note that ``xesmf`` is an optional dependency, which is required for using ``xesmf``
   based horizontal regridding APIs in ``xcdat``. ``xesmf`` is not currently supported
   on `windows`_ because ``esmpy`` is not yet available on this platform. Windows
   users can try `WSL2`_ as a workaround.

.. _windows: https://github.com/conda-forge/esmf-feedstock/issues/64
.. _WSL2: https://docs.microsoft.com/en-us/windows/wsl/install

2. Install ``xcdat`` in an existing mamba environment (`mamba install`_)

   You can also install ``xcdat`` in an existing mamba environment, granted that mamba
   is able to resolve the compatible dependencies.

   .. code-block:: console

       >>> mamba activate <ENV_NAME>
       >>> mamba install -c conda-forge xcdat xesmf

   Note: As above, ``xesmf`` is an optional dependency.

3. [Optional] Some packages that are commonly used with ``xcdat`` can be installed
   either in step 1 or step 2 above:

   - ``jupyterlab``: a web-based interactive development environment for notebooks,
     code, and data. This package also includes ``ipykernel``.
   - ``matplotlib``: a library for creating visualizations in Python.
   - ``cartopy``: an add-on package for ``matplotlib`` and specialized for geospatial data processing.

.. _mamba create: https://fig.io/manual/mamba/create
.. _mamba install: https://fig.io/manual/mamba/install
