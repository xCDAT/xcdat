.. highlight:: shell

============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/tomvothecoder/xcdat/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "Type: Bug" is open to whoever wants to implement it.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for features. Anything tagged with "Type: Enhanacement" is open to whoever wants to implement it.

Write Documentation
~~~~~~~~~~~~~~~~~~~

xCDAT could always use more documentation, whether as part of the
official xCDAT docs, in docstrings, or even on the web in blog posts,
articles, and such.

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at https://github.com/tomvothecoder/xcdat/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a open-source project, and that contributions
  are welcome :)

Get Started!
------------

Ready to contribute? Here's how to set up `xcdat` for local development.


1. Download Conda

    Linux
        ::

            $ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

    MacOS
        ::

            $ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh

2. Install Conda

    Linux
        ::

            $ bash ./Miniconda3-latest-Linux-x86_64.sh

    MacOS
        ::

            $ bash ./Miniconda3-latest-MacOSX-x86_64.sh

    - ``Do you wish the installer to initialize Miniconda3 by running conda init? [yes|no] yes``

3. Fork the ``xcdat`` repo on GitHub.

4. Clone your fork locally::

    $ git clone git@github.com:your_name_here/xcdat.git

5. Create and activate Conda development environment::

    $ cd xcdat
    $ conda env create -f conda-env/dev.yml
    $ conda activate xcdat_dev

6. Install ``pre-commit`` (runs quality assurance tools)::

    $ pre-commit install
    pre-commit installed at .git/hooks/pre-commit

7. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally (and add unit tests if needed).

8. <OPTIONAL> When you're done making changes, check that your changes pass flake8 and the
   tests ::

    $ pre-commit run --all-files
    Trim Trailing Whitespace.................................................Passed
    Fix End of Files.........................................................Passed
    Check Yaml...............................................................Passed
    black....................................................................Passed
    isort....................................................................Passed
    flake8...................................................................Passed
    mypy.....................................................................Passed

9. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    # pre-commit automatically runs on every commit
     Trim Trailing Whitespace.................................................Passed
     Fix End of Files.........................................................Passed
     Check Yaml...............................................................Passed
     black....................................................................Passed
     isort....................................................................Passed
     flake8...................................................................Passed
     mypy.....................................................................Passed


11. Push changes
    $ git push origin name-of-your-bugfix-or-feature

12. Submit a pull request through the GitHub website.

Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. Link issues to pull requests
3. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in README.rst.
4. Use the pull request checklist for further guidance

Tips
----

To run a subset of tests::

$ pytest tests.test_xcdat


Deploying
---------

A reminder for the maintainers on how to deploy.
Make sure all your changes are committed (including an entry in HISTORY.rst).
Then run::

$ bump2version patch # possible: major / minor / patch
$ git push
$ git push --tags

GitHub Actions will then build to Anaconda automatically
