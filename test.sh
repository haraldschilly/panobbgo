#!/bin/bash
set -e

echo "Syncing dependencies..."
uv sync --extra dev

echo "Running CI equivalent locally..."
uv run python run_ci.py
