read [AGENTS.md](./AGENTS.md)

## Development

- Always use `uv run ...` to run tools (ruff, pyright, pytest, etc.), not `.venv/bin/...` directly.
- **Benchmark harness**: When changing optimization strategies, heuristics, or core logic, use `uv run python benchmark_harness.py` to measure before/after impact. See AGENTS.md "Benchmark Harness" section for the full workflow.
