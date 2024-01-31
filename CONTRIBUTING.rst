.. highlight:: shell

.. note::

  Large portions of this document came from or are inspired by the `Xarray Contributing
  Guide <https://docs.xarray.dev/en/stable/contributing.html>`_.

============
Overview
============

xCDAT is a community-driven open source project and we welcome everyone who would like to
contribute! Feel free to open a `GitHub Issue`_ for bug reports, bug fixes,
documentation improvements, enhancement suggestions, and other ideas.

We encourage discussion on any xCDAT related topic in the `GitHub Discussion`_ forum.
You can also subscribe to our `mailing list`_ for news and announcements related to xCDAT,
such as software version releases or future roadmap plans.

Please note that xCDAT has a `Code of Conduct`_. By participating in the xCDAT
community, you agree to abide by its rules.

Where to start?
---------------

If you are brand new to xCDAT or open-source development, we recommend going
through the `GitHub Issues`_ page to find issues that interest you. If you are
interested in opening a GitHub Issue, you can use these `templates`_.

Some issues are particularly suited for new contributors by the label
`type: docs <https://github.com/xCDAT/xcdat/labels/type%3A%20docs>`_ and
`good-first-issue <https://github.com/xCDAT/xcdat/labels/good-first-issue>`_ where
you could start out. These are well documented issues, that do not require a deep
understanding of the internals of xCDAT. Once you've found an interesting issue, you can
return here to get your development environment setup.

**New contributions must be made under the Apache-2.0 with LLVM exception license.**

Documentation Updates
~~~~~~~~~~~~~~~~~~~~~

Contributing to the `documentation`_ is an excellent way to help xCDAT. Anything from
fixing typos to improving sentence structure or API docstrings/examples can go a long
way in making xCDAT more user-friendly.

To open a documentation update request, please use the `documentation update template`_.

Bug reports
~~~~~~~~~~~
Bug reports are an important part of making xCDAT more stable. Having a complete bug
report will allow others to reproduce the bug and provide insight into fixing.

Trying out the bug-producing code on the ``main`` branch is often a worthwhile exercise
to confirm that the bug still exists. It is also worth searching existing bug reports and
pull requests to see if the issue has already been reported and/or fixed.

To open a new bug report, please use this issue `bug report template`_.

Enhancement requests
~~~~~~~~~~~~~~~~~~~~
Enhancements are a great way to improve xCDAT's capabilities for the broader scientific
community.

If you are proposing an enhancement:

* Explain in detail how it would work
* Make sure it fits in the domain of geospatial climate science
* Keep the scope as narrow as possible, to make it easier to implement
* Generally reusable among the community (not model or data-specific)
* Remember that this is a open-source project, and that contributions are welcome :)

To open a new enhancement request, please use this issue `enhancement request template`_.

All other inquiries
~~~~~~~~~~~~~~~~~~~~

We welcome comments, questions, or feedback! Please share your thoughts on the
`GitHub Discussions`_ forum!

Version control, Git, and GitHub
--------------------------------

The code is hosted on `GitHub`_. To contribute you will need to sign up for a
`free GitHub account`_. We use `Git`_ for version control to allow many people to work
together on the project.

Some great resources for learning Git:

* the `GitHub help pages`_.
* the `NumPy's documentation`_.
* Matthew Brett's `Pydagogue`_.

Getting started with Git
~~~~~~~~~~~~~~~~~~~~~~~~

`GitHub has instructions for setting up Git`_ including installing git,
setting up your SSH key, and configuring git.  All these steps need to be completed before
you can work seamlessly between your local repository and GitHub.

.. note::

    The following instructions assume you want to learn how to interact with github via the git command-line utility,
    but contributors who are new to git may find it easier to use other tools instead such as
    `GitHub Desktop`_.

.. Links

.. _GitHub has instructions for setting up Git: https://help.github.com/set-up-git-redirect
.. _templates: https://github.com/xCDAT/xcdat/issues/new/choose
.. _documentation: https://xcdat.readthedocs.io/en/latest/
.. _GitHub Issues: https://github.com/xCDAT/xcdat/issues
.. _documentation update template: https://github.com/xCDAT/xcdat/issues/new?assignees=&labels=Type%3A+Documentation&projects=&template=documentation.yml&title=%5BDoc%5D%3A+
.. _bug report template: https://github.com/xCDAT/xcdat/issues/new?assignees=&labels=Type%3A+Bug&projects=&template=bug_report.yml&title=%5BBug%5D%3A+
.. _enhancement request template: https://github.com/xCDAT/xcdat/issues/new?assignees=&labels=Type%3A+Enhancement&projects=&template=feature_request.yml&title=%5BFeature%5D%3A+
.. _GitHub Issue: https://github.com/xCDAT/xcdat/issues
.. _GitHub Issues: https://github.com/xCDAT/xcdat/issues
.. _GitHub Discussion: https://github.com/xCDAT/xcdat/discussions
.. _GitHub Discussions: https://github.com/xCDAT/xcdat/discussions
.. _Code of Conduct: CODE-OF-CONDUCT.rst
.. _mailing list: https://groups.google.com/g/xcdat
.. _GitHub: https://www.github.com/xCDAT/xcdat
.. _free GitHub account: https://github.com/signup/free
.. _Git: http://git-scm.com/
.. _GitHub help pages: https://help.github.com/
.. _NumPy's documentation: https://numpy.org/doc/stable/dev/index.html
.. _Pydagogue: https://matthew-brett.github.io/pydagogue/
.. _GitHub Desktop: https://desktop.github.com/

Creating a development environment
----------------------------------

Before starting any development, you'll need to create an isolated xCDAT
development environment:

- We recommend installing `Miniforge`_ to use ``mamba``. If you prefer, you can install
  `Anaconda`_ or `Miniconda`_ to use ``conda`` instead.
- Make sure your conda is up to date (``conda update conda``)
- Make sure that you have forked and cloned the `repository`_. If you are in the xCDAT
  organization, forking is not needed since you will have direct read access to the repo.
- ``cd`` to the ``xcdat`` source directory

Now we are going through a two-step process:

1. Install the build dependencies

    .. code-block:: bash

       >>> mamba env create -f conda-env/dev.yml
       >>> mamba activate xcdat_dev
       >>>
       >>> make install # or python -m pip install .

2. Build and install ``xcdat``

    At this point you should be able to import ``xcdat`` from your locally built version:

    .. code-block:: bash

      $ python  # start an interpreter
      >>> import xcdat
      >>> xcdat.__version__
      '0.6.1'

    This will create the new environment, and not touch any of your existing environments,
    nor any existing Python installation.

To view your environments:

  .. code-block:: bash

     conda info -e

To return to your root environment:

  .. code-block:: bash

     conda deactivate

See the full `conda docs here`_.

.. _Miniforge: https://github.com/conda-forge/miniforge
.. _Anaconda: https://www.anaconda.com/download
.. _Miniconda: https://docs.conda.io/projects/miniconda/en/latest/
.. _repository: https://github.com/xCDAT/xcdat
.. _conda docs here: http://conda.pydata.org/docs

.. _installing pre-commit hooks:

Install pre-commit hooks
~~~~~~~~~~~~~~~~~~~~~~~~

We highly recommend that you setup `pre-commit`_ hooks to automatically run all
the tools listed in the :ref:`code-formatting` section every time you make a git commit.

To install the hooks

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
-----------------------------

.. contents::
   :local:

Pull Request (PR)
~~~~~~~~~~~~~~~~~

When you open a `pull request`_ on GitHub, there a template with a checklist available to use.

Here's a simple checklist for PRs:
- **Properly comment and document your code.** API docstrings are formatted using the
`NumPy style guide`_
- **Test that the documentation builds correctly** by typing ``make docs`` in the root of the ``xcdat`` directory. This is not strictly necessary, but this may be easier than waiting for CI to catch a mistake.
- **Test your code**.

  - Write new tests if needed.
  - Test the code using `Pytest`_. Running all tests (type ``make test`` or ``pytest`` in the root directory) takes a while, so feel free to only run the tests you think are needed based on your PR (example: ``pytest xarray/tests/test_dataarray.py``). CI will catch any failing tests.

- **Properly format your code** and verify that it passes the formatting guidelines set by `Black`_ and `Flake8`_. You can use `pre-commit`_ to run these automatically on each commit.

  - Run ``pre-commit run --all-files`` in the root directory. This may modify some files. Confirm and commit any formatting changes.

- **Push your code** and `create a PR on GitHub`_.
- **Use a helpful title for your pull request** by summarizing the main contributions rather than using the latest commit message. If the PR addresses a `GitHub Issue`_, please `reference it`_.

.. _code-formatting:

Code Formatting
~~~~~~~~~~~~~~~

xCDAT uses several tools to ensure a consistent code format throughout the project:

- `Black`_ for standardized code formatting
- `Flake8`_ for code linting
- `isort`_ for standardized order of imports
- `mypy`_ for static type checking on `type hints`_

We highly recommend that you setup `pre-commit hooks`_ to automatically run all the
above tools every time you make a git commit. Check out the :ref:`installing pre-commit
hooks` section above for instructions.

.. _pull request: https://github.com/xCDAT/xcdat/compare
.. _create a PR on GitHub: https://help.github.com/en/articles/creating-a-pull-request
.. _reference it: https://help.github.com/en/articles/autolinked-references-and-urls
.. _NumPy style guide: https://numpydoc.readthedocs.io/en/latest/format.html
.. _pre-commit: https://pre-commit.com/
.. _pre-commit hooks: https://pre-commit.com/
.. _Pytest: http://doc.pytest.org/en/latest/
.. _Black: https://black.readthedocs.io/en/stable/
.. _Flake8: https://flake8.pycqa.org/en/latest/
.. _isort: https://pycqa.github.io/isort/
.. _mypy: http://mypy-lang.org/
.. _type hints: https://docs.python.org/3/library/typing.html

Testing With Continuous Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The xCDAT `build workflow`_ runs the test suite automatically via the `GitHub Actions`_,
continuous integration service, once your pull request is submitted.

A pull-request will be considered for merging when you have an all 'green' build. If any
tests are failing, then you will get a red 'X', where you can click through to see the
individual failed tests. This is an example of a green build.

.. image:: _static/ci.png

.. note::

   Each time you push to your PR branch, a new run of the tests will be
   triggered on the CI. If they haven't already finished, tests for any older
   commits on the same branch will be automatically cancelled.

.. _build workflow: https://github.com/xCDAT/xcdat/actions/workflows/build_workflow.yml
.. _GitHub Actions: https://docs.github.com/en/free-pro-team@latest/actions

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
--------------

Helpful Commands
~~~~~~~~~~~~~~~~

.. note::
    * Run ``make help`` in the root of the project for a list of useful commands
    * Run ``make install`` to install a local build of xCDAT into your mamba/conda environment
    * Run ``make clean`` to delete all build, test, coverage and Python artifacts


xCDAT and Visual Studio Code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We recommend using `VS Code`_ as your code editor because it is open-source and has
great Python development support.

xCDAT includes a `VS Code Workspace file`_, which conveniently configures VS Code for
quality assurance tools, code line-length rulers, and more. You just need to open this
file using VS Code, create your development environment, and select the appropriate
Python interpreter for the workspace (from the dev environment).

Some recommended extensions include:

    * `Python <https://marketplace.visualstudio.com/items?itemName=ms-python.python>`_
    * `Pylance <https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance>`_
    * `Python Docstring Generator <https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring>`_
    * `Python Type Hint <https://marketplace.visualstudio.com/items?itemName=njqdev.vscode-python-typehint>`_
    * `Better Comments <https://marketplace.visualstudio.com/items?itemName=aaron-bond.better-comments>`_
    * `Jupyter <https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter>`_
    * `Visual Studio Intellicode <https://marketplace.visualstudio.com/items?itemName=VisualStudioExptTeam.vscodeintellicode>`_
    * `autoDocstring <https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring>`_

.. _VS Code: https://code.visualstudio.com
.. _VS Code Workspace file: https://github.com/xCDAT/xcdat/blob/main/.vscode/xcdat.code-workspace
