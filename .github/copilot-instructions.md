# Copilot Instructions for xCDAT

<!-- This file is automatically applied to all GitHub Copilot Chat requests in this workspace. -->
<!-- Canonical guidelines live in AGENTS.md at the repository root. -->

## Project

xCDAT (Xarray Climate Data Analysis Tools) extends xarray for climate data
analysis on structured grids. Python ≥3.11.

## Coding Rules

- Format and lint with **Ruff** (config in `pyproject.toml`). Import sorting is
  handled by Ruff (`ruff check --select I --fix`).
- Type-check with **MyPy** (`check_untyped_defs = true`). All public APIs must
  have type annotations.
- Write **NumPy-style** docstrings with `Parameters`, `Returns`, and `Raises`
  sections.
- Use absolute imports (e.g., `from xcdat.axis import get_dim_coords`).
- License: Apache-2.0 with LLVM exception.

## Architecture

- Dataset functionality uses **xarray accessors** (`@xr.register_dataset_accessor`):
  `BoundsAccessor`, `SpatialAccessor`, `TemporalAccessor`, `RegridderAccessor`.
- Use **cf_xarray** for coordinate discovery — never hardcode dimension names.
- Shared helpers belong in `xcdat/utils.py`. Export new public symbols from
  `xcdat/__init__.py`.

## Testing

- Use **pytest**. Tests live in `tests/test_<module>.py`.
- Prefer `xarray.testing.assert_identical` / `assert_allclose` for comparisons.
- Construct expected results explicitly and compare against actual output.
- Run: `pytest tests/test_<module>.py` (targeted) or `make test` (full suite).

## Dependencies

- Do not add new runtime dependencies without discussion.
- Core: xarray, numpy, cf_xarray, cftime, dask, pandas, scipy, netcdf4, pooch,
  regionmask, sparse, python-dateutil.
- Regridding: xesmf, xgcm.

## Workflow

- Run `pre-commit run --all-files` before committing.
- Every PR must include tests for new or changed behavior.
- Keep changes minimal and focused.
