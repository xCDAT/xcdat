.. highlight:: shell

===============
Getting Started
===============

Prerequisites
-------------

1. Familiarity with ``xarray``, since this package is an extension of it

   - Please visit the `xarray documentation`_ to get started.

2. Anaconda is your development platform

.. _xarray documentation: https://docs.xarray.dev/en/stable/getting-started-guide/index.html

Installation
------------

1. Create a Conda environment from scratch with ``xcdat`` (`conda create`_)

   We recommend using the Conda environment creation procedure to install ``xcdat``.
   The advantage with following this approach is that Conda will attempt to resolve dependencies (e.g. ``python >= 3.8``) for compatibility.

   To create a Conda environment with ``xcdat``, run:

   .. code-block:: console

       >>> conda create -n <ENV_NAME> -c conda-forge xcdat
       >>> conda activate <ENV_NAME>

2. Install ``xcdat`` in an existing Conda environment (`conda install`_)

   You can also install ``xcdat`` in an existing Conda environment, granted that Conda is able to resolve the compatible dependencies.

   .. code-block:: console

       >>> conda activate <ENV_NAME>
       >>> conda install -c conda-forge xcdat

.. _conda create: https://docs.conda.io/projects/conda/en/latest/commands/create.html?highlight=create
.. _conda install: https://docs.conda.io/projects/conda/en/latest/commands/install.html?highlight=install
