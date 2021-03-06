.. highlight:: shell

===================
Project Maintenance
===================

This page covers tips for project maintenance.

Releasing a New Version
-----------------------

1. Checkout the latest ``main`` branch.
2. Checkout a branch with the name of the version.

  ::

      # For release candidates, append "rc" to <version>
      git checkout -b <version>
      git push --set-upstream origin <version>

3. Add updates to ``HISTORY.rst`` and commit.

4. Bump version using tbump.

  ::

      # <version> should match step 2
      # --no-tag is required since tagging is handled in step 5.
      tbump <version> --no-tag

5. Create a pull request to the main repo and merge it.
6. Create a GitHub release.
7. Open a PR to the xcdat conda-forge feedstock with the latest changes.

   - https://github.com/conda-forge/xcdat-feedstock

Continuous Integration / Continuous Delivery (CI/CD)
-----------------------------------------------------

This project uses `GitHub Actions <https://github.com/xCDAT/xcdat/actions>`_ to run the CI/CD build workflow.

This workflow is triggered by Git ``pull_request`` and ``push`` (merging PRs) events to the the main repo's ``main`` branch.

Jobs:
    1. Run ``pre-commit`` for formatting, linting, and type checking
    2. Build conda CI/CD environment with different Python versions, install package, and run test suite
