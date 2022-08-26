xCDAT on Jupyter and HPC Machines
=================================

xCDAT should be compatible with most high performance computing (HPC)
platforms. In general, xCDAT is available via the conda-forge channel
and should be installed via `conda <https://www.anaconda.com/products/distribution>`_. Note
that xCDAT is available through conda and should follow the same
installation procedure as other conda-based packages. These instructions
follow conda installations from
`NERSC <https://docs.nersc.gov/development/languages/python/nersc-python/>`_,
but setup can vary depending on the exact HPC environment you are
working in so please consult your HPC documentation and/or HPC Support
Resources.

Setting up your xCDAT environment
---------------------------------

Ensure ``conda`` is installed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Option 1: Machines with ``conda`` pre-installed
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Your HPC machine may have ``python`` and ``conda`` pre-installed. You
can check to see whether they are available by entering ``which conda``
and/or ``which python`` in the command line (which will return their
path if they are available).

Some machines make ``python`` and ``conda`` available via modules. For
example, some machines make both available via:

::

   module load python

Option 2: Machines without ``conda`` pre-installed
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   bash Miniconda3-latest-Linux-x86_64.sh

Then follow the instructions for installation. To have conda added to
your path you will need to type ``yes`` in response to "Do you wish the
installer to initialize Miniconda3 by running conda init?" (we recommend
that you do this). Note that this will modify your shell profile (e.g.,
``~/.bashrc``) to add ``conda`` to your path.

Note: After installation completes you may need to type ``bash`` to
restart your shell (if you use bash). Alternatively, you can log out and
log back in.

Creating your environment with ``conda``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once ``conda`` is setup, you can create a new ``xcdat`` environment
with:

::

   conda create -n xcdat -c conda-forge xcdat

You may also want to use ``xcdat`` with some additional packages. For
example, you could instead install ``xcdat`` with ``matplotlib``,
``ipython``, and ``ipykernel`` (see the next section for more about
``ipykernel``):

::

   conda create -n xcdat -c conda-forge xcdat matplotlib ipython ipykernel

You can add packages later with ``conda install ...``.

Adding an ``xcdat`` kernel for use with Jupyter
-----------------------------------------------

HPC systems frequently include a web interface to
`Jupyter <https://docs.jupyter.org/en/latest/>`__, which is a popular
web application that is used to perform analyses in Python. In order to
use ``xcdat`` with Jupyter, you will need to create a kernel in your
``xcdat`` conda environment using ``ipykernel``. These instructions
follow those from
`NERSC <https://docs.nersc.gov/services/jupyter/#conda-environments-as-kernels>`__,
but setup can vary depending on the exact HPC environment you are
working in so please consult your HPC documentation. If you have not
already installed ``ipykernel``, you can install it in your ``xcdat``
environment (created above) with:

::

   conda activate xcdat
   conda install -c conda-forge ipykernel

Once ``ipykernel`` is added to your ``xcdat`` environment, you can
create an ``xcdat`` kernel with:

::

   python -m ipykernel install --user --name xcdat --display-name xcdat

After the kernel is installed, login to the Jupyter instance on your
HPC. Your ``xcdat`` kernel may be available on the home launch page (to
open a new notebook or command line instance). This launcher is
sometimes accessed by clicking the blue plus symbol (see screenshot
below). Alternatively, you may need top open a new Notebook and then
click "Kernel" on the top bar -> click "Change Kernel..." and then
select your ``xcdat`` kernel. You should then be able to use your
``xcdat`` environment on Jupyter.

|image0|

.. |image0| image:: _static/jupyter-launcher-example.png
