# Environment, commands and CI

## Commands

**Always use `uv run ...`** for Python and every tool (pytest, ruff,
pyright, sphinx-build, scripts).  Never call `.venv/bin/...` or a bare
`python`.

```bash
uv sync --extra dev                       # install (pip: pip install -e ".[dev]")
uv sync --extra dev --extra baselines     # + pycma / Nevergrad / Optuna baselines
uv sync --extra dev --extra baselines --extra baselines-bo  # + BoTorch / TuRBO / SMAC3 / Py-BOBYQA (CPU torch ~0.7 GB)
uv run pytest -q -n 4                     # full suite (~2300 tests, ~1 min)
uv run pytest -q tests/test_core.py       # one file
uv run ruff format .                      # format (CI gate: ruff format --check .)
uv run ruff check --fix .                 # lint; ./codestyle.sh runs both
uv run pyright panobbgo                   # type check (CI gate)
uv run flake8 panobbgo                    # advisory only
uv run sphinx-build -b doctest doc/source doc/build/doctest   # docs doctests (CI gate)
uv run sphinx-build -b html doc/source doc/build/html         # docs HTML (docs.yml)
uv run pytest benchmarks/ --benchmark-min-rounds=1 --benchmark-max-time=0.1 -q  # micro-benchmarks (CI gate)
./test.sh                                 # every CI job locally (run_ci.py; --job NAME, --dry-run)
```

## CI

`.github/workflows/tests.yml` gates on pytest, pyright, `ruff format
--check`, the Sphinx doctest build and the micro-benchmarks; flake8 runs
but does not fail the build.  The `baselines-bo` tests run in their own
job (`test-bo`, with torch); the default test job skips them.  `docs.yml` builds and deploys the HTML
docs.  No PR-side job runs the benchmark harness, so a green PR says
nothing about optimization quality.  Details:
[`.github/workflows/README.md`](../../.github/workflows/README.md).

Useful: `gh pr checks <N> --watch`, `gh run list`,
`gh run view <RUN_ID> --log`.

## Dependencies: newest stable

Keep `requires-python` and the floors in `pyproject.toml` at the newest
**stable** releases (no pre-releases or release candidates), and bump them
deliberately rather than waiting for Dependabot.  A single-maintainer
research codebase has no downstream users to stay compatible with; an old
floor only buys untested version combinations.

To bump: raise the floors, `uv lock`, `uv sync --extra dev --extra dask --extra baselines`,
and bump `env.PYTHON` in **every** workflow in the same PR — the CI pin is
part of the dependency set.  CI then runs the suite, pyright and both
Sphinx builds.

Exception: `tools/ioh_worker/` stays at `>=3.11,<3.13` because the `ioh`
wheels stop at cp312.  It is an isolated uv project precisely so that this
ceiling does not hold the main package back.

## Scripts and local runs

*   Scripts that build `LBFGSB` / `COBYQA` / `LocalPenaltySearch` /
    `QuadraticWlsModel` or use `evaluation_method = "processes"` start
    `"spawn"` subprocesses, which re-import the script: keep the code that
    runs the optimization under `if __name__ == "__main__":` (user guide,
    "Scripts That Start Worker Processes").
*   Multi-seed screens (`benchmarks/*_screen.py`, `arm_sweep.py`,
    `oracle.py`, `np_accept.py`) take `jobs=N`, and
    `scripts/ioh_benchmark.py run` takes `--jobs N`: independent
    (seed, cell) runs go to N `spawn` worker processes
    (`panobbgo.local_run.TaskPool`).  Under `sync_eval` the records do not
    depend on N.
*   On a shared machine, run long jobs under `nice -n 15` and size N to the
    free cores and memory.  Large measurements can go to GitHub runners
    instead ([benchmarking.md](benchmarking.md#re-baselining-on-github-runners)).
