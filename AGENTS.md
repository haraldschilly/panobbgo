# AGENTS.md

This file provides instructions for agents working on the Panobbgo repository.

## General Instructions

*   **Code Style**: Follow PEP 8 guidelines.
*   **Documentation**: Ensure all public functions and classes have docstrings (Google style).
*   **Testing**: All new code must be tested. Existing tests should be migrated to `pytest`.
*   **Dependencies**: Keep dependencies up-to-date. Avoid using deprecated libraries.

## Repository Structure

*   `panobbgo/`: Main package source code.
*   `panobbgo/lib/`: Problem definitions and library functions (formerly `panobbgo.lib`).
*   `tests/`: Directory for tests (to be created/populated).

## Testing

*   Use `pytest` for running tests.
*   Tests are currently scattered in files ending with `_test.py` alongside the source code.
*   Goal: Move tests to a dedicated `tests/` directory.

## Build & Install

*   Use `pip install -e .` for development installation.
*   The project uses `setuptools` (and ideally `pyproject.toml`).
