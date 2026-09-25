# GitHub Actions workflows

Two workflows live here. `./test.sh` (via `run_ci.py`) replays their `run`
steps locally: each step's block runs in one `bash -e`, `continue-on-error`
is honoured, `$GITHUB_STEP_SUMMARY` goes to `/dev/null`, and the jobs run in
workflow order (the gh-pages `deploy` job is skipped).
`uv run python run_ci.py --dry-run` prints the plan without running it.

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

## Maintenance

- **Changing the Python version**: update `env.PYTHON` in each workflow.
- **Adding a job**: copy an existing job (including the cache steps) and
  update this README.
- Third-party actions are pinned to major versions (`actions/cache@v5`,
  `actions/checkout@v6`, ...).
