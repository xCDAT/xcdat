.. highlight:: shell

===============
Getting Started
===============

Prerequisites
-------------

1. Familiarity with ``xarray``, since this package is an extension of it

   - We highly recommend visiting the `xarray tutorial`_ and `xarray documentation`_
     pages if you aren't familiar with ``xarray``.

2. xCDAT is distributed through the `conda-forge`_ channel of Anaconda. We recommend
  using Mamba (via `Miniforge`_), a drop-in replacement of Conda that is faster and more
  reliable than Conda. Miniforge ships with `conda-forge` set as the prioritized channel.
  Mamba also uses the same commands and configurations as Conda, and you can swap
  commands between both tools.

   Follow these steps to install Miniforge (Mac OS & Linux):

   .. code-block:: console

      curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
      bash Miniforge3-$(uname)-$(uname -m).sh

   Then follow the instructions for installation. To have conda and mamba added to
   your path you will need to type ``yes`` in response to ``"Do you wish the
   installer to initialize Miniforge by running conda init?"`` (we recommend
   that you do this). Note that this will modify your shell profile (e.g.,
   ``~/.bashrc``) to add ``conda`` to your path.

   Note: After installation completes you may need to type ``bash`` to
   restart your shell (if you use bash). Alternatively, you can log out and
   log back in.

.. _xarray tutorial: https://tutorial.xarray.dev/intro.html
.. _xarray documentation: https://docs.xarray.dev/en/stable/getting-started-guide/index.html
.. _conda-forge: https://anaconda.org/conda-forge/xcdat
.. _Miniforge: https://github.com/conda-forge/miniforge

Installation
------------

1. Create a Mamba environment from scratch with ``xcdat`` (`mamba create`_)

   We recommend using the Mamba environment creation procedure to install ``xcdat``.
   The advantage with following this approach is that Mamba will attempt to resolve
   dependencies (e.g. ``python >= 3.8``) for compatibility.

   To create an ``xcdat`` Mamba environment with ``xesmf`` (a recommended dependency),
   run:

   .. code-block:: console

       >>> mamba create -n <ENV_NAME> -c conda-forge xcdat xesmf
       >>> mamba activate <ENV_NAME>

   Note that ``xesmf`` is an optional dependency, which is required for using ``xesmf``
   based horizontal regridding APIs in ``xcdat``. ``xesmf`` is not currently supported
   on `windows`_ because it depends on ``esmpy``, which also does not support Windows.
   Windows users can try `WSL2`_ as a workaround.

.. _windows: https://github.com/conda-forge/esmf-feedstock/issues/64
.. _WSL2: https://docs.microsoft.com/en-us/windows/wsl/install

2. Install ``xcdat`` in an existing Mamba environment (`mamba install`_)

   You can also install ``xcdat`` in an existing Mamba environment, granted that Mamba
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

.. _mamba create: https://mamba.readthedocs.io/en/latest/user_guide/mamba.html#quickstart
.. _mamba install: https://mamba.readthedocs.io/en/latest/user_guide/mamba.html#quickstart


Updating
--------

New versions of ``xcdat`` will be released periodically. We recommend you use the
latest stable version of ``xcdat`` for the latest features and bug fixes.

.. code-block:: console

   >>> mamba activate <ENV_NAME>
   >>> mamba update xcdat

To update to a specific version of ``xcdat``:

.. code-block:: console

   >>> mamba activate <ENV_NAME>
   >>> mamba update xcdat=<MAJOR.MINOR.PATCH>
