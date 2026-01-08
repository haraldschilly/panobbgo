#!/bin/bash
set -e

echo "Syncing dependencies..."
uv sync --extra dev

echo "Running CI equivalent locally..."
python run_ci.py
