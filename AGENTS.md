# AGENTS.md

Instructions for agents working on the Panobbgo repository. Keep this file
actionable; history and design notes live under `planning/`.

## Where things are

*   `TODO.md` — the status tracker (newest-first). Update it when you fix a
    bug, find a new issue, finish a task, or make an architectural decision.
*   `planning/GOAL.md` — the goal contract for open-ended "improve panobbgo"
    work: metric of record, operating loop, research backlog. Read it first
    for any optimisation-quality task.
*   `doc/source/guide_benchmarking.rst` — user-facing benchmarking guide
    (composite score, running the harness, statistical acceptance).
*   `planning/LOOP_REFERENCE.md` — the autonomous self-improvement loop's
    flag/feature reference. The loop is **not currently in use** (nightly
    workflow disabled on GitHub since 2026-08-13); the code and its tests
    stay in the tree. `planning/SELF_IMPROVEMENT_LOOP.md` is its design,
    `planning/SELF_IMPROVEMENT_LOG.md` and `planning/done/` its history.
*   `tools/ioh_worker/README.md` — IOH/MA-BBOB worker setup and protocol.
*   `planning/TEST_PERFORMANCE.md` — test-suite timing notes.

## General rules

*   Follow PEP 8; `ruff` (line length 120) is the formatter and linter.
*   Public functions and classes get Google-style docstrings.
*   All new code must be tested; tests live in `tests/`.
*   Stochastic/integration tests that may intermittently fail are decorated
    with `@pytest.mark.flaky(retries=3)` (`pytest-retry`).
*   Update the copyright banner year on files you edit: `2012-<current year>`.
*   Do not change the composite-score formula (`panobbgo/harness.py`)
    without an architectural decision record — historical comparisons
    depend on it.

## Environment and commands

**Always use `uv run ...`** for Python and every tool (pytest, ruff,
pyright, sphinx-build, scripts). Never call `.venv/bin/...` or bare
`python` directly.

```bash
uv sync --extra dev                       # install (pip: pip install -e ".[dev]")
uv run pytest -q -n 4                     # full suite (~2000 tests, ~1 min)
uv run pytest -q tests/test_core.py       # one file; serial `pytest` also works
uv run ruff format .                      # format (CI gate: ruff format --check .)
uv run ruff check --fix .                 # lint; ./codestyle.sh runs both
uv run pyright panobbgo                   # type check (CI gate)
uv run flake8 panobbgo                    # advisory only (CI: continue-on-error)
uv run sphinx-build -b doctest doc/source doc/build/doctest   # docs doctests (CI gate)
uv run sphinx-build -b html doc/source doc/build/html         # docs HTML (docs.yml)
uv run pytest benchmarks/ --benchmark-min-rounds=1 --benchmark-max-time=0.1 -q  # micro-benchmarks (CI gate)
./test.sh                                 # replicate the whole CI pipeline locally (runs run_ci.py)
```

CI (`.github/workflows/tests.yml`) gates on: pytest, pyright, `ruff format
--check`, the Sphinx doctest build, and the pytest-benchmark suite. flake8
runs but does not fail the build. `.github/workflows/README.md` describes
the workflows.

Useful `gh` commands: `gh pr checks <N>`, `gh run list`,
`gh run view <RUN_ID> --log`.

### Local runs

*   Long benchmark or pytest runs go through `nice -n 15` so the machine
    stays usable, e.g. `nice -n 15 uv run python benchmark_harness.py run --standard ...`.
*   Measure progress in **evaluations**, not wall time. Wall time depends on
    machine load and the evaluator thread pool; evaluation counts are the
    comparable quantity (see "Domain context").

## Running Panobbgo

Panobbgo is a **library**: strategies, heuristics and analyzers are composed
in a Python script (see `README.md` and `doc/source/guide_usage.rst`).
Evaluation is threaded and local by default; Dask is an optional extra for
distributed evaluation. `sketchpad/` holds unpolished scratch scripts, not
curated demos.

## Domain context: black-box noisy optimisation

Panobbgo solves **expensive, noisy black-box optimisation** problems:

*   **The evaluation budget is a hard cap.** `max_eval` is a strict limit,
    not a target. Strategies must respect phase boundaries and never let
    in-flight evaluations overshoot budget allocations.
*   **Generating candidates is cheap; evaluating them is expensive.**
    Produce as many proposals as you like, but control what is submitted
    for evaluation against the budget.
*   **Restartability matters.** The storage backend enables checkpointing
    and resuming, which is essential when one evaluation takes minutes.
*   **Phased strategies** (`StrategyPhased`) must account for pending
    (in-flight) evaluations when enforcing per-phase budgets.

## Benchmark harness

The harness is the **single source of truth** for "is Panobbgo better or
worse than before this change?". It yields one scalar,
`composite_score` ∈ [0, 1]. **Use it whenever you modify a strategy
(`strategies/`), a heuristic (`heuristics/`), core evaluation or constraint
handling, or the benchmark registry.** Full guide:
`doc/source/guide_benchmarking.rst`.

### Workflow

```bash
# 1. Baseline BEFORE changing anything
uv run python benchmark_harness.py run --quick --output before.json
# 2. Make the change
# 3. Measure AFTER
uv run python benchmark_harness.py run --quick --output after.json
# 4. Compare — exit code 2 on regression
uv run python benchmark_harness.py compare before.json after.json --statistical --fail-on-regression
```

### Modes

*   `--quick`: 3 problems × 2 strategies × 3 reps × 75 evals (~30 s) — during development
*   `--standard`: 8 problems × ~6 strategies × 5 reps × 200 evals (minutes) — before merging
*   `--full`: 11 problems × ~10 strategies × 10 reps × 500 evals (~1 h) — thorough validation

Useful flags (see the guide for details):

*   `--baselines` adds external reference solvers (`Baseline_Random`,
    `Baseline_SciPyDE`, `Baseline_SciPyAnneal`) for an *absolute* reference
    (`panobbgo/harness_baselines.py`).
*   `--randomize --randomize-iteration N` swaps the fixed battery for
    parametrically randomised instances (translation / rotation / scaling /
    noise); the same `N` reproduces the same instances so before/after runs
    line up (`panobbgo/harness_randomized.py`).
*   `--seed S` changes the base seed; re-run at a second seed before
    trusting a small delta.

### Score interpretation

*   **1.0** — every run solves at evaluation 1 (theoretical ceiling)
*   **0.7+** — strong; optima found with budget left over
*   **0.3** — weak; rare, late successes
*   **0.0** — never within tolerance

Per-pair metrics: `success_rate`, `ert` (BBOB standard),
`best_func_distance`, `median_func_distance`.

### Statistical rigor

*   `--quick` is **noisy** (3 reps): ±0.02 is within noise. Treat quick
    deltas as trend signals, not proof.
*   For deltas of +0.01…+0.03, re-run with `--seed 43` before accepting.
*   Before merging a significant algorithmic change, run `--standard` or
    `--full` (niced, on a machine you are not otherwise loading).
*   `compare --statistical` bootstraps a 95 % CI on the composite delta and
    accepts iff delta > `eps_accept` (0.005), CI lower bound > 0, and no
    single (problem, strategy) pair regresses by more than `eps_regress`
    (0.05). With `--fail-on-regression` the exit code is 2 on rejection.
    API: `panobbgo.harness.statistical_accept`.
*   The bootstrap is **paired** (rep-aligned) automatically when both sides
    have the same rep count — the `--randomize` case. Force with
    `--paired` / `--unpaired`; use `--unpaired` when reps are not
    instance-aligned (different `base_seed`s).
*   Known measurement-fidelity limit: identical seeded runs are not yet
    bit-reproducible (thread pool, per-heuristic RNGs); see
    `planning/GOAL.md` and `TODO.md` before interpreting deltas below ~0.05.

### IOH / MA-BBOB anytime track (AOCC)

A parallel measurement track scores Panobbgo on the IOHprofiler MA-BBOB
suite with the **AOCC** metric (`scripts/ioh_benchmark.py`,
`panobbgo/harness_ioh.py`). It needs the isolated worker venv — setup and
protocol in `tools/ioh_worker/README.md` (`cd tools/ioh_worker && uv sync`;
without it IOH tests skip via the `requires_worker` marker).

```bash
uv run python scripts/ioh_benchmark.py run --quick --baselines
uv run python scripts/ioh_benchmark.py run --standard --baselines --output ioh_before.json
uv run python scripts/ioh_benchmark.py compare ioh_before.json ioh_after.json
```

`composite_score` and AOCC do not interconvert; a change can improve one and
regress the other — track both. AOCC is the metric of record in
`planning/GOAL.md`.

### Key files

*   `panobbgo/harness.py` — `BenchmarkHarness`, metrics, `compare()`, `statistical_accept()`
*   `panobbgo/harness_baselines.py`, `panobbgo/harness_randomized.py`, `panobbgo/harness_ioh.py`
*   `benchmark_harness.py` — CLI (`run`, `score`, `compare`, `list`)
*   `tests/test_harness*.py` — harness tests
*   `panobbgo/self_improve.py`, `scripts/self_improve.py` — loop driver (dormant; see `planning/LOOP_REFERENCE.md`)

## Agent-driven "improve X" PRs — evidence vs. CI

**First, deduplicate.** Run `gh pr list --state open` (drafts included) and
skim titles before implementing an improvement; if the idea is already in
an open PR, finish that one instead of opening a duplicate.

A green PR proves the change does not break tests / lint / typecheck /
docs / format / micro-benchmarks. **A green PR does NOT prove the change
improved `composite_score` or AOCC** — no PR-side CI workflow executes the
benchmark harness.

When asked to "improve the default strategy", "push to PR", "do not run
locally", or anything else that prevents running the harness before the PR
is opened, the agent must:

1.  **State the evidence form in the PR description.** Acceptable, in
    decreasing order of strength:
    *   a locally captured `before.json` / `after.json` pair compared with
        `compare --statistical --fail-on-regression --paired`, quoting the
        composite delta and CI bounds;
    *   a ledger entry (`planning/self_improve_ledger*.jsonl`) whose
        `proposal` matches the exact change, with `accepted: true`, CI
        lower bound > 0 and no per-pair regression beyond `eps_regress` —
        cite the iteration and `(base_seed, randomize_iteration)`.
    *   *Not acceptable alone:* "another strategy uses this configuration",
        literature analogy, or "the docstring says it should help". These
        are motivations, not measurements.
2.  **Say explicitly when evidence is missing** for part of the change.
3.  **Flag unmeasured changes as "pending validation"** so the owner knows a
    follow-up measurement is required. (The nightly loop that used to
    re-measure merged changes is disabled.)

Cumulative improvement requires each PR's claim to be backed by
measurement or honestly marked as unmeasured; otherwise the project's "is
it better than master?" signal degrades as analogies stack.
