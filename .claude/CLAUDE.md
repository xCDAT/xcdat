# CLAUDE.md — Claude Code Instructions for xCDAT

## Canonical Reference

All coding standards, architecture constraints, testing conventions, and
dependency policies are defined in **`AGENTS.md`** at the repository root.
Apply those rules to all generated code and suggestions.

## Claude-Specific Behavior

- Read `AGENTS.md` at the start of each session for full project context.
- Ground all suggestions in the actual repository structure and existing code.
- Do not invent files, modules, or configuration not present in the repository.
- When uncertain about project conventions, consult `AGENTS.md`, `pyproject.toml`,
  or `.pre-commit-config.yaml`.

## Quick Reference — Build Commands

```bash
make install       # install local build
make lint          # ruff check + fix
make format        # ruff format
make test          # full test suite with coverage
pytest tests/test_<module>.py  # targeted tests
make docs          # build Sphinx documentation
```
