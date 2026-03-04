# AGENTS.md — AI Development Guidelines for xCDAT

This file is the canonical, tool-agnostic source of AI development rules for the
[xCDAT](https://github.com/xCDAT/xcdat) project. Tool-specific files
(`.github/copilot-instructions.md`, `.claude/CLAUDE.md`) contain concise extracts
aligned with this document.

## Project Overview

xCDAT (Xarray Climate Data Analysis Tools) is a Python library that extends
[xarray](https://xarray.dev) for climate data analysis on structured grids. It
provides spatial/temporal averaging, regridding, bounds handling, and dataset
I/O with CF-convention support.

See `pyproject.toml` for the minimum supported Python version and all dependency
constraints.

## Repository Layout

```
xcdat/              # Main package source
  axis.py           # Coordinate axis utilities
  bounds.py         # BoundsAccessor (xarray accessor)
  dataset.py        # Dataset I/O (open_dataset, open_mfdataset)
  spatial.py        # SpatialAccessor for spatial averaging
  temporal.py       # TemporalAccessor for temporal averaging
  mask.py           # Masking operations
  regridder/        # Regridding subpackage (xESMF, xgcm, regrid2)
  utils.py          # Shared utility functions
  _logger.py        # Logger setup
  tutorial.py       # Sample data utilities
tests/              # Pytest test suite (test_*.py)
docs/               # Sphinx documentation
conda-env/          # Conda environment files
```

## Coding Standards

- **Formatter / Linter**: Ruff — handles formatting, linting, and import sorting.
  See `pyproject.toml` under `[tool.ruff]` for configuration.
- **Type Checking**: MyPy — all public APIs must include type annotations.
  See `pyproject.toml` under `[tool.mypy]` for configuration.
- **Docstrings**: Use **NumPy-style** docstrings for all public functions, classes,
  and methods. Include `Parameters`, `Returns`, and `Raises` sections as applicable.
- **Imports**: Use absolute imports within the package
  (e.g., `from xcdat.axis import get_dim_coords`). Import sorting is handled by Ruff.
- **Pre-commit**: The project uses pre-commit hooks. See `.pre-commit-config.yaml`
  for the configured checks (Ruff, MyPy, whitespace, etc.).
- **License**: New contributions must be made under the Apache-2.0 with LLVM exception
  license.

## Architecture

- **xarray Accessors**: `BoundsAccessor`, `SpatialAccessor`, `TemporalAccessor`, and
  `RegridderAccessor` extend `xr.Dataset` via `@xr.register_dataset_accessor`.
  Follow this pattern when adding new dataset-level functionality.
- **CF Conventions**: Use `cf_xarray` for coordinate discovery. Always reference axes
  by CF standard names or axis attributes, not hardcoded dimension names.
- **Module Responsibility**: Each module has a single clear responsibility. Avoid
  circular imports. Shared helpers go in `utils.py`.
- **Public API**: All public symbols are re-exported from `xcdat/__init__.py`. When
  adding a new public function or class, add it to `__init__.py`.

## Testing Conventions

- **Framework**: pytest with pytest-cov. See `pyproject.toml` under
  `[tool.pytest.ini_options]` for configuration.
- **File Naming**: Test files live in `tests/` and follow the pattern `test_<module>.py`.
- **Assertions**: Prefer `xarray.testing.assert_identical` and
  `xarray.testing.assert_allclose` for comparing xarray objects.
- **Fixtures**: Use pytest fixtures for reusable test data. Construct expected results
  explicitly and compare against actual output.
- **Running Tests**: `pytest` (full suite) or `pytest tests/test_<module>.py` (targeted).
  Use `make test` for the full suite with coverage.

## Dependencies

- See `pyproject.toml` under `[project.dependencies]` for runtime dependencies and
  version constraints.
- See `pyproject.toml` under `[project.optional-dependencies]` for dev, test, and
  docs dependencies.
- Do not add new runtime dependencies without discussion. Prefer leveraging existing
  dependencies before introducing new ones.

## Build & Development

```bash
# Create dev environment
conda env create -f conda-env/dev.yml
conda activate xcdat_dev

# Install local build
make install  # or: python -m pip install .

# Quality assurance
make lint          # ruff check + fix
make format        # ruff format
make pre-commit    # run all pre-commit hooks

# Testing
make test                        # full suite with coverage
pytest tests/test_<module>.py    # targeted tests

# Documentation
make docs  # build Sphinx HTML docs
```

## Workflow Rules

- Always run `pre-commit run --all-files` before committing.
- Every PR should include tests for new or changed behavior.
- Do not modify unrelated code or tests.
- Keep changes minimal and focused on the issue being addressed.
- Use meaningful commit messages that summarize the change.
