# GitHub Actions workflows

Three workflows live here. `./test.sh` (via `run_ci.py`) replays the `run`
steps of the two CI workflows locally (the on-demand `rebaseline.yml` is skipped): each step's block runs in one `bash -e`, `continue-on-error`
is honoured, `$GITHUB_STEP_SUMMARY` goes to `/dev/null`, and the jobs run in
workflow order (the gh-pages `deploy` job is skipped).
`uv run python run_ci.py --dry-run` prints the plan without running it.

## `tests.yml` — CI on every push / PR to `master`

Independent parallel jobs:

- **test**: `uv run pytest -v --cov=panobbgo` (also syncs the
  `tools/ioh_worker` venv so the `requires_worker` IOH tests run, and the
  `baselines` extra so the pycma / Nevergrad / Optuna adapter tests run)
- **lint**: flake8 — *advisory only* (`continue-on-error: true`)
- **typecheck**: `uv run pyright panobbgo`
- **docs**: `uv run sphinx-build -b doctest doc/source doc/build/doctest`
  (the doctests embedded in the user guide must pass)
- **benchmark**: `uv run pytest benchmarks/ --benchmark-min-rounds=1 --benchmark-max-time=0.1 -q`
  under a 60 s timeout
- **format**: `uv run --extra dev ruff format --check .` — the formatting gate

Note that no PR-side job runs the benchmark harness (`benchmark_harness.py`);
a green PR says nothing about optimization quality. See
`doc/dev/benchmarking.md`, "Evidence for a PR".

### Setup and caching

Every job installs UV and runs `uv sync --extra dev` (the test job adds
`--extra baselines` and syncs on every run, since the shared cache may
lack it). UV itself and `.venv`
are cached with `actions/cache`, keyed on
`uv-${{ runner.os }}-python-${{ env.PYTHON }}-${{ hashFiles('pyproject.toml', 'uv.lock') }}`,
so the install steps are skipped on a cache hit. The Python version is set
once per workflow via `env.PYTHON` (currently 3.14) and referenced everywhere.

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
Concurrency is per ref (`docs-<ref>`): a new push to a PR cancels that PR's
older build only, and deploys share one `pages-deploy` group, so they are
serialised and never cancelled mid-push.

## `rebaseline.yml` — re-measure the reference baselines (manual)

`workflow_dispatch` only (inputs: `suites`, `seeds`, `ref`, `release`).  A
`plan` job turns the inputs into a matrix via `scripts/rebaseline.py plan`;
each `measure` job runs one (suite, seed chunk) shard — synchronous, seeded,
no wall-clock limit — and uploads `shard-<suite>-<shard>`; `aggregate`
builds the reference files, publishes them as a pre-release
`rebaseline-<date>` (the only job with `contents: write`) and uploads
`rebaseline-references`.  Procedure in `doc/dev/benchmarking.md`,
"Re-baselining on GitHub runners".

## Maintenance

- **Changing the Python version**: update `env.PYTHON` in each workflow.
- **Adding a job**: copy an existing job (including the cache steps) and
  update this README.
- Third-party actions are pinned to major versions (`actions/cache@v5`,
  `actions/checkout@v6`, ...).
