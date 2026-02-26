# CLAUDE.md — Claude Code Instructions for xCDAT

<!-- See AGENTS.md at the repository root for full AI development guidelines. -->

## Project

xCDAT (Xarray Climate Data Analysis Tools) extends xarray for climate data
analysis on structured grids. See `pyproject.toml` for the supported Python
version and dependency constraints.

## Coding Rules

- Format and lint with **Ruff** (config in `pyproject.toml`). Import sorting is
  handled by Ruff.
- Type-check with **MyPy** (config in `pyproject.toml`). All public APIs must
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
- See `pyproject.toml` for the full dependency list and version constraints.

## Workflow

- Run `pre-commit run --all-files` before committing.
- Every PR must include tests for new or changed behavior.
- Keep changes minimal and focused.

## Build Commands

```bash
make install       # install local build
make lint          # ruff check + fix
make format        # ruff format
make test          # full test suite with coverage
pytest tests/test_<module>.py  # targeted tests
make docs          # build Sphinx documentation
```
