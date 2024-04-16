.. highlight:: shell

=============
Installation
=============

Prerequisites
-------------

1. Familiarity with ``xarray``, since this package is an extension of it

   - We highly recommend visiting the `xarray tutorial`_ and `xarray documentation`_
     pages if you aren't familiar with ``xarray``.

2. xCDAT is distributed through the `conda-forge`_ channel of Anaconda. We recommend
   using Mamba (via `Miniforge`_), a drop-in replacement of Conda that is faster and more
   reliable than Conda. Miniforge ships with ``conda-forge`` set as the prioritized channel.
   Mamba also uses the same commands and configurations as Conda, and you can swap
   commands between both tools.

   Follow these steps to install Miniforge (Mac OS & Linux):

   .. code-block:: bash

      curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
      bash Miniforge3-$(uname)-$(uname -m).sh

   Then follow the instructions for installation. We recommend you type ``yes`` in
   response to ``"Do you wish the installer to initialize Miniforge by running conda init?"``
   to add ``conda`` and ``mamba`` to your path. Note that this will modify your shell
   profile (e.g., ``~/.bashrc``).

   *Note: After installation completes you may need to type ``bash`` to
   restart your shell (if you use bash). Alternatively, you can log out and
   log back in.*

.. _xarray tutorial: https://tutorial.xarray.dev/intro.html
.. _xarray documentation: https://docs.xarray.dev/en/stable/getting-started-guide/index.html
.. _conda-forge: https://anaconda.org/conda-forge/xcdat
.. _Miniforge: https://github.com/conda-forge/miniforge

Instructions
------------

1. Create a Mamba environment from scratch with ``xcdat`` (`mamba create`_)

   We recommend using the Mamba environment creation procedure to install ``xcdat``.
   The advantage with following this approach is that Mamba will attempt to resolve
   dependencies (e.g. ``python >= 3.8``) for compatibility.

   To create an ``xcdat`` Mamba environment,
   run:

   .. code-block:: bash

       >>> mamba create -n <ENV_NAME> -c conda-forge xcdat
       >>> mamba activate <ENV_NAME>

2. Install ``xcdat`` in an existing Mamba environment (`mamba install`_)

   You can also install ``xcdat`` in an existing Mamba environment, granted that Mamba
   is able to resolve the compatible dependencies.

   .. code-block:: bash

       >>> mamba activate <ENV_NAME>
       >>> mamba install -c conda-forge xcdat

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

.. code-block:: bash

   >>> mamba activate <ENV_NAME>
   >>> mamba update xcdat

To update to a specific version of ``xcdat``:

.. code-block:: bash

   >>> mamba activate <ENV_NAME>
   >>> mamba update xcdat=<MAJOR.MINOR.PATCH>

Jupyter Users set ``ESMFMKFILE`` env variable
---------------------------------------------

If you are a Jupyter user, the ``ESMFMKFILE`` environment variable will need to be set
either directly on the machine or through your Jupyter Notebook.

This env variable is normally set when calling ``conda activate`` with the conda
environment that has ``xesmf``. However, Jupyter does not run ``conda activate`` when using
the Python kernel associated with the environment so ``ESMFMKFILE`` is not set, resulting 
in ``ImportError: The ESMFMKFILE environment variable is not available.`` (related `GitHub
Issue <https://github.com/xCDAT/xcdat/issues/574>`_.

To set the ``EMSFMKFILE`` in a Jupyter Notebook add:

.. code-block:: python

   >>> import os
   >>> os.environ['ESMFMKFILE'] = 'conda-envs/xcdat/lib/esmf.mk'
   >>>
   >>> import xcdat
