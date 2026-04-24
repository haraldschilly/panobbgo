#!/usr/bin/env bash
# Format and lint the codebase with ruff.
# CI enforces `ruff format --check` — run this before committing.

set -euo pipefail

uv run ruff format .
uv run ruff check --fix .
