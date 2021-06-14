.. highlight:: shell

===================
Project Maintenance
===================

This page covers tips for project maintenance.

Releasing a New Version
-----------------------

1. Add updates to HISTORY.rst
2. Checkout the latest ``master``.
3. Checkout a branch with the name of the version.

  ::

      # Prepend "v" to <version>
      # For release candidates, append "rc" to <version>
      git checkout -b v<version> -t origin/<version>

4. Bump version using tbump.

  ::

      # Exclude "v" and <version> should match step 2
      # --no-tag is required since tagging is handled in step 5.
      tbump <version> --no-tag

5. Create a pull request to the main repo and merge it.
6. Create a GitHub release. GitHub Actions will handle publishing the package to Anaconda.

Continuous Integration / Continuous Delivery (CI/CD)
-----------------------------------------------------

This project uses `GitHub Actions <https://github.com/tomvothecoder/xcdat/actions>`_ to run two CI/CD workflows.

1. CI/CD Build Workflow

  This workflow is triggered by Git ``pull_request`` and ``push`` (merging PRs) events to the the main repo's ``master``.

  Jobs:

    1. Run ``pre-commit`` for formatting, linting, and type checking
    2. Build conda development environment and run test suite

2. CI/CD Release Workflow

  This workflow is triggered by the Git ``publish`` event, which occurs when a new release is tagged.

  Jobs:

    1. Publish Anaconda package
