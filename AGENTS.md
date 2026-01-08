# AGENTS.md

This file provides instructions for agents working on the Panobbgo repository.

## General Instructions

*   **Code Style**: Follow PEP 8 guidelines.
*   **Documentation**: Ensure all public functions and classes have docstrings (Google style).
*   **Testing**: All new code must be tested. Existing tests should be migrated to `pytest`.
*   Goal: Move tests to a dedicated `tests/` directory.
*   **Priority**: Extend tests with artificial but "realistic" examples for framework robustness validation.
*   **Integration Test**: `tests/test_integration.py` provides end-to-end optimization testing on Rosenbrock function.
*   **Coverage Goal**: Improve test coverage from current 45% via comprehensive integration tests.

## Build & Install

*   **UV-based setup (recommended)**: Use `uv sync --extra dev` for development installation with all dependencies.
*   **Traditional pip**: Use `pip install -e ".[dev]"` for development installation.
*   The project uses `setuptools` with `pyproject.toml` configuration.

## Python Environment Usage

*   **CRITICAL**: When running Python commands, ALWAYS use `uv run python` instead of bare `python`
*   The system Python has version conflicts (NumPy 2.x vs scipy/matplotlib compiled for 1.x)
*   `uv run python` uses the virtual environment with compatible package versions
*   Example: `uv run python script.py` instead of `python script.py`

## Known Issues

*   **Pandas Compatibility**: Framework uses deprecated `DataFrame.append()` method, incompatible with pandas 2.x
*   **Result Hashing**: Result objects need `__hash__` method for Splitter analyzer
*   **Dependency Management**: Some optional dependencies (matplotlib) fail with NumPy 2.x

## Running Panobbgo

Panobbgo is designed as a **framework for black-box optimization** but includes **out-of-the-box runnable examples** for testing and demonstration:

*   **Framework nature**: Panobbgo is a library that provides components (strategies, heuristics, analyzers) for building custom optimization pipelines
*   **Runnable demos**: Example scripts in `sketchpad/` demonstrate complete optimization runs (e.g., `python sketchpad/test01.py`)
*   **Testing**: Run `uv run pytest` or `pytest` to execute the test suite and verify functionality
*   **Interactive use**: Import and use components directly in Python scripts for custom optimization problems

The framework runs on **Dask distributed** for parallel evaluation, supporting both local clusters and remote distributed computing.

## CI/CD and Testing

*   **Local Testing**: Run `./test.sh` to replicate the full CI pipeline locally
*   **CI Status**: Check GitHub Actions status with `gh pr checks <PR_NUMBER>` or `gh run list`
*   **CI Logs**: View detailed CI logs with `gh run view <RUN_ID> --log` or `gh run view --web <RUN_ID>`
*   **Code Formatting**: Use `uv run ruff format` to format code, `./codestyle.sh` for convenience
*   **Linting**: Run `uv run flake8 panobbgo` to check for style issues
