.. highlight:: shell

============
Maintainers
============

This page covers tips for project maintainers.


Deploying
---------

A reminder for the maintainers on how to deploy.
Make sure all your changes are committed (including an entry in HISTORY.rst).

Then run::

$ bump2version patch # possible: major / minor / patch
$ git push
$ git push --tags

Afterwards, GitHub Actions will publish to Anaconda automatically


Continuous Integration / Continuous Delivery (CI/CD)
-----------------------------------------------------

This project uses `GitHub Actions <https://github.com/tomvothecoder/xcdat/actions>`_ to run two CI/CD workflows.

1. CI/CD Build Workflow

  This workflow is triggered by Git ``pull_request`` and ``push`` (merging PRs) events to the the main repo's ``master``.

  Jobs:

    1. Run ``pre-commit`` for formatting, linting, and type checking
    2. Build conda environment and run test suite
    3. Publish latest `master` docs (only on `push`)

2. CI/CD Release Workflow

  This workflow is triggered by the Git ``publish`` event, which occurs when a new release is tagged.

  Jobs:

    1. Publish new release docs
    2. Publish Anaconda package
