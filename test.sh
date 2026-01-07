#!/bin/bash
set -e

echo "Syncing dependencies..."
uv sync --extra dev

echo "Running flake8 linting..."
# Stop the build if there are Python syntax errors or undefined names
uv run flake8 panobbgo --count --select=E9,F63,F7,F82 --show-source --statistics
# exit-zero treats all errors as warnings. The GitHub editor is 127 chars wide
uv run flake8 panobbgo --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

echo "Running type checking with pyright..."
uv run pyright panobbgo

echo "Running tests with pytest..."
uv run pytest -v --cov=panobbgo --cov-report=xml --cov-report=term
