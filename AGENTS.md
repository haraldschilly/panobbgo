# AGENTS.md

This file provides instructions for agents working on the Panobbgo repository.

## General Instructions

*   **Code Style**: Follow PEP 8 guidelines.
*   **Documentation**: Ensure all public functions and classes have docstrings (Google style).
*   **Testing**: All new code must be tested. Existing tests should be migrated to `pytest`.
*   **Dependencies**: Keep dependencies up-to-date. Avoid using deprecated libraries.

## Development Roadmap

*   **DEVELOPMENT_PROMPT.md**: Contains the comprehensive development roadmap, current priorities, and implementation guidelines.
*   **Immediate Focus**: Framework robustness through realistic testing scenarios, advanced bandit strategies, and constraint handling.
*   **Long-term Goals**: Persistent storage, convergence detection, and advanced optimization features.

## Repository Structure

*   `panobbgo/`: Main package source code.
*   `panobbgo/lib/`: Problem definitions and library functions (formerly `panobbgo.lib`).
*   `tests/`: Directory for tests (to be created/populated).

## Documentation

* The Sphinx based documentation files are in `doc/source/`.
* In particular, the `doc/source/guide.rst` should explain this package.

## Testing

*   Use `pytest` for running tests.
*   Tests are currently scattered in files ending with `_test.py` alongside the source code.
*   Goal: Move tests to a dedicated `tests/` directory.
*   **Priority**: Extend tests with artificial but "realistic" examples for framework robustness validation.

## Build & Install

*   **UV-based setup (recommended)**: Use `uv sync --extra dev` for development installation with all dependencies.
*   **Traditional pip**: Use `pip install -e ".[dev]"` for development installation.
*   The project uses `setuptools` with `pyproject.toml` configuration.

## Running Panobbgo

Panobbgo is designed as a **framework for black-box optimization** but includes **out-of-the-box runnable examples** for testing and demonstration:

*   **Framework nature**: Panobbgo is a library that provides components (strategies, heuristics, analyzers) for building custom optimization pipelines
*   **Runnable demos**: Example scripts in `sketchpad/` demonstrate complete optimization runs (e.g., `python sketchpad/test01.py`)
*   **Testing**: Run `uv run pytest` or `pytest` to execute the test suite and verify functionality
*   **Interactive use**: Import and use components directly in Python scripts for custom optimization problems

The framework runs on **Dask distributed** for parallel evaluation, supporting both local clusters and remote distributed computing.
