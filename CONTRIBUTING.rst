.. highlight:: shell

.. note::

  Large portions of this document came from or are inspired by the `Xarray Contributing
  Guide <https://docs.xarray.dev/en/stable/contributing.html>`_.

============
Overview
============

xCDAT is a community-driven open source project and we welcome everyone who would like to
contribute! Feel free to open a `GitHub Issue`_ or start a `GitHub Discussion`_ for bug
reports, bug fixes, documentation improvements, enhancement suggestions, and other ideas.

We encourage discussion on any xCDAT related topic in the `GitHub Discussion`_ forum.
You can also subscribe to our `mailing list`_ for news and announcements related to xCDAT,
such as software version releases or future roadmap plans.

Please note that xCDAT has a `Code of Conduct`_. By participating in the xCDAT
community, you agree to abide by its rules.

.. _GitHub Issue: https://github.com/xCDAT/xcdat/issues
.. _GitHub Discussion: https://github.com/xCDAT/xcdat/discussions
.. _Code of Conduct: CODE-OF-CONDUCT.rst
.. _mailing list: https://groups.google.com/g/xcdat

Where to start?
===============

If you are brand new to xCDAT or open-source development, we recommend going
through the `GitHub Issues`_ page to find issues that interest you. If you are
interested in opening a GitHub issue, you can use the templates `here <https://github.com/xCDAT/xcdat/issues/new/choose>`_

Some issues are particularly suited for new contributors by the label
`Documentation <https://github.com/xCDAT/xcdat/labels/type%3A%20documentation>`_ and
`good first issue <https://github.com/xCDAT/xcdat/labels/good-first-issue>`_ where
you could start out. These are well documented issues, that do not require a deep
understanding of the internals of xCDAT. Once you've found an interesting issue, you can
return here to get your development environment setup.

**New contributions must be made under the Apache-2.0 with LLVM exception license.**

Documentation Updates
~~~~~~~~~~~~~~~~~~~~~

Contributing to the `documentation <https://xcdat.readthedocs.io/en/latest/>`_ is an
excellent way to help xCDAT. You don't need to be an expert in xCDAT to help with
documentation!

Bug Reports and enhancement requests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Bug reports are an important part of making xCDAT more stable. Having a complete bug
report will allow others to reproduce the bug and provide insight into fixing.

Trying out the bug-producing code on the ``main`` branch is often a worthwhile exercise
to confirm that the bug still exists. It is also worth searching existing bug reports and
pull requests to see if the issue has already been reported and/or fixed.

.. _GitHub Issues: https://github.com/xCDAT/xcdat/issues

Version control, Git, and GitHub
================================

The code is hosted on `GitHub <https://www.github.com/xCDAT/xcdat>`_. To
contribute you will need to sign up for a `free GitHub account
<https://github.com/signup/free>`_. We use `Git <http://git-scm.com/>`_ for
version control to allow many people to work together on the project.

Some great resources for learning Git:

* the `GitHub help pages <https://help.github.com/>`_.
* the `NumPy's documentation <https://numpy.org/doc/stable/dev/index.html>`_.
* Matthew Brett's `Pydagogue <https://matthew-brett.github.io/pydagogue/>`_.

Getting started with Git
------------------------

`GitHub has instructions for setting up Git <https://help.github.com/set-up-git-redirect>`__ including installing git,
setting up your SSH key, and configuring git.  All these steps need to be completed before
you can work seamlessly between your local repository and GitHub.

.. note::

    The following instructions assume you want to learn how to interact with github via the git command-line utility,
    but contributors who are new to git may find it easier to use other tools instead such as
    `Github Desktop <https://desktop.github.com/>`_.

Creating a development environment
==================================

Before starting any development, you'll need to create an isolated xCDAT
development environment:

- Install either `Anaconda <https://www.anaconda.com/download/>`_ or `miniconda
  <https://conda.io/miniconda.html>`_
- Make sure your conda is up to date (``conda update conda``)
- Make sure that you have forked and cloned the repository
- ``cd`` to the *xcdat* source directory

Now we are going through a two-step process:

    1. Install the build dependencies
    2. Build and install `xcdat`

    .. code-block:: bash
       # Create and activate the conda/mamba development environment
       >>> mamba env create -f conda-env/dev.yml
       >>> mamba activate xcdat_dev

       # Build and install xcdat
       >>> make install # or python -m pip install .

At this point you should be able to import ``xcdat`` from your locally built version:

.. code-block:: sh

   $ python  # start an interpreter
   >>> import xcdat
   >>> xcdat.__version__
   '0.6.1'

This will create the new environment, and not touch any of your existing environments,
nor any existing Python installation.

To view your environments::

      conda info -e

To return to your root environment::

      conda deactivate

See the full `conda docs here <http://conda.pydata.org/docs>`__.

Install pre-commit hooks
------------------------

We highly recommend that you setup `pre-commit <https://pre-commit.com/>`_ hooks to automatically
run all the above tools every time you make a git commit. To install the hooks

.. code-block:: bash
    >>> python -m pip install pre-commit
    >>> pre-commit install

This can be done by running:

.. code-block:: bash
    >>> pre-commit run

from the root of the ``xcdat`` repository. You can skip the pre-commit checks with
``git commit --no-verify``.

Note, these hooks are also executed in the GitHub CI/CD build workflow and must
pass before pull requests can be merged.

Contributing to the code base
=============================

.. contents:: Code Base:
   :local:

Pull Request (PR)
-----------------

When you open a `pull request <https://github.com/xCDAT/xcdat/compare>`_ on GitHub,
there a template with a checklist available to use.

Here's a simple checklist for PRs:
- **Properly comment and document your code.** API docstrings are formatted using the
`numpy style guide <https://numpydoc.readthedocs.io/en/latest/format.html>`_
- **Test that the documentation builds correctly** by typing ``make doc`` in the root of the ``xcdat`` directory. This is not strictly necessary, but this may be easier than waiting for CI to catch a mistake. See `"Contributing to the documentation" <https://docs.xarray.dev/en/stable/contributing.html#contributing-to-the-documentation>`_.
- **Test your code**.

  - Write new tests if needed.
  - Test the code using `Pytest <http://doc.pytest.org/en/latest/>`_. Running all tests (type ``make test`` or ``pytest`` in the root directory) takes a while, so feel free to only run the tests you think are needed based on your PR (example: ``pytest xarray/tests/test_dataarray.py``). CI will catch any failing tests.

- **Properly format your code** and verify that it passes the formatting guidelines set by `Black <https://black.readthedocs.io/en/stable/>`_ and `flake8`_. You can use `pre-commit <https://pre-commit.com/>`_ to run these automatically on each commit.

  - Run ``pre-commit run --all-files`` in the root directory. This may modify some files. Confirm and commit any formatting changes.

- **Push your code** and `create a PR on GitHub <https://help.github.com/en/articles/creating-a-pull-request>`_.
- **Use a helpful title for your pull request** by summarizing the main contributions rather than using the latest commit message. If the PR addresses an `issue <https://github.com/xCDAT/xcdat/issues>`_, please `reference it <https://help.github.com/en/articles/autolinked-references-and-urls>`_.

Code Formatting
~~~~~~~~~~~~~~~

xCDAT uses several tools to ensure a consistent code format throughout the project:

- `Black <https://black.readthedocs.io/en/stable/>`_ for standardized
  code formatting,
- `Flake8 <https://flake8.pycqa.org/en/latest/>`_ for code linting
- `isort <https://pycqa.github.io/isort/>`_ for standardized order of imports
- `mypy <http://mypy-lang.org/>`_ for static type checking on `type hints
  <https://docs.python.org/3/library/typing.html>`_.

We highly recommend that you setup `pre-commit hooks <https://pre-commit.com/>`_
to automatically run all the above tools every time you make a git commit. This
can be done by running::

   pre-commit install

from the root of the ``xcdat`` repository. You can skip the pre-commit checks
with ``git commit --no-verify``.

Testing With Continuous Integration
-----------------------------------

The xCDAT `build workflow <https://github.com/xCDAT/xcdat/actions/workflows/build_workflow.yml>`_
runs the test suite automatically via the
`GitHub Actions <https://docs.github.com/en/free-pro-team@latest/actions>`__,
continuous integration service, once your pull request is submitted.

A pull-request will be considered for merging when you have an all 'green' build. If any
tests are failing, then you will get a red 'X', where you can click through to see the
individual failed tests. This is an example of a green build.

.. image:: _static/ci.png

.. note::

   Each time you push to your PR branch, a new run of the tests will be
   triggered on the CI. If they haven't already finished, tests for any older
   commits on the same branch will be automatically cancelled.

Writing tests
~~~~~~~~~~~~~

All tests should go into the ``tests`` subdirectory of the specific package.
This folder contains many current examples of tests, and we suggest looking to these for
inspiration.

The ``xarray.testing`` module has many special ``assert`` functions that
make it easier to make statements about whether DataArray or Dataset objects are
equivalent. The easiest way to verify that your code is correct is to
explicitly construct the result you expect, then compare the actual result to
the expected correct result::

    def test_constructor_from_0d():
        expected = Dataset({None: ([], 0)})[None]
        actual = DataArray(0)
        assert_identical(expected, actual)


Developer Tips
==============

Helpful Commands
----------------

.. note::
    Run ``make help`` in the root of the project for a list of useful commands


xCDAT and Visual Studio Code
----------------------------

We recommend using `VS Code <https://code.visualstudio.com>`_ as your code editor because
it is open-source and has great Python development support.

xCDAT includes a
`VS Code Workspace file <https://github.com/xCDAT/xcdat/blob/main/.vscode/xcdat.code-workspace>`_, which conveniently configures VS Code for quality assurance tools, code line-length rulers, and more.
Just open ``.vscode/xcdat-workspace.code-settings`` using VS Code.

Some recommended extensions include:

    * `Python <https://marketplace.visualstudio.com/items?itemName=ms-python.python>`_
    * `Pylance <https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance>`_
    * `Python Docstring Generator <https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring>`_
    * `Python Type Hint <https://marketplace.visualstudio.com/items?itemName=njqdev.vscode-python-typehint>`_
    * `Better Comments <https://marketplace.visualstudio.com/items?itemName=aaron-bond.better-comments>`_
    * `Jupyter <https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter>`_
    * `Visual Studio Intellicode <https://marketplace.visualstudio.com/items?itemName=VisualStudioExptTeam.vscodeintellicode>`_
    * `autoDocstring <https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring>`_
