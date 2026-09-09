# GitHub Actions workflows

Three workflows live here. `./test.sh` (via `run_ci.py`) replays the
`tests.yml` commands locally.

## `tests.yml` — CI on every push / PR to `master`

Independent parallel jobs:

- **test**: `uv run pytest -v --cov=panobbgo` (also syncs the
  `tools/ioh_worker` venv so the `requires_worker` IOH tests run)
- **lint**: flake8 — *advisory only* (`continue-on-error: true`)
- **typecheck**: `uv run pyright panobbgo`
- **docs**: `uv run sphinx-build -b doctest doc/source doc/build/doctest`
  (the doctests embedded in the user guide must pass)
- **benchmark**: `uv run pytest benchmarks/ --benchmark-min-rounds=1 --benchmark-max-time=0.1 -q`
  under a 60 s timeout
- **format**: `uv run --extra dev ruff format --check .` — the formatting gate

Note that no PR-side job runs the benchmark harness (`benchmark_harness.py`);
a green PR says nothing about optimization quality. See `AGENTS.md`,
"Agent-driven improve-X PRs".

### Setup and caching

Every job installs UV and runs `uv sync --extra dev`. UV itself and `.venv`
are cached with `actions/cache`, keyed on
`uv-${{ runner.os }}-python-${{ env.PYTHON }}-${{ hashFiles('pyproject.toml', 'uv.lock') }}`,
so the install steps are skipped on a cache hit. The Python version is set
once via `env.PYTHON` (currently 3.12) and referenced everywhere.

If the cache becomes corrupted: GitHub → Actions → Caches, delete the
offending entries, and the next run recreates them.

## `docs.yml` — build and deploy documentation

Runs on pushes and PRs to `master`. The `build` job installs the dev extra
plus Sphinx and runs

```bash
uv run sphinx-build -b html doc/source doc/build/html
```

(without `-W`, so warnings do not fail the build) and uploads the result as
an artifact. On `master` the `deploy` job checks out the `gh-pages` branch,
replaces its contents with the built HTML and pushes — that is what serves
https://haraldschilly.github.io/panobbgo/.

## `self_improve_nightly.yml` — nightly self-improvement loop (disabled)

A scheduled (03:00 UTC) run of `scripts/self_improve.py run` against the
IOH/MA-BBOB AOCC metric (or `composite` via `workflow_dispatch`), which
appends to the metric's ledger under `planning/` and commits the updated
ledger and summary back to `master` with `[skip ci]`.

**This workflow is currently disabled on GitHub** (since 2026-08-13): the
loop's accept rule sits on the harness's measurement-noise floor, see
`planning/done/LOOP_DIAGNOSIS_2026-08-11.md`. The workflow file is kept so
it can be re-enabled once the measurement issues in `TODO.md` are
resolved; the loop's flags are documented in `planning/LOOP_REFERENCE.md`.

## Maintenance

- **Changing the Python version**: update `env.PYTHON` in each workflow.
- **Adding a job**: copy an existing job (including the cache steps) and
  update this README.
- Third-party actions are pinned to major versions (`actions/cache@v5`,
  `actions/checkout@v6`, ...).
