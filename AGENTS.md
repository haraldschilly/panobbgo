# AGENTS.md

This file provides instructions for agents working on the Panobbgo repository.

## General Instructions

*   **Code Style**: Follow PEP 8 guidelines.
*   **Documentation**: Ensure all public functions and classes have docstrings (Google style).
*   **Testing**: All new code must be tested. Existing tests should be migrated to `pytest`.
*   Goal: Move tests to a dedicated `tests/` directory.
*   **Priority**: Extend tests with artificial but "realistic" examples for framework robustness validation.
*   **Integration Test**: `tests/test_integration.py` provides end-to-end optimization testing on Rosenbrock function.
*   **Coverage Goal**: Improve test coverage from current 45% via comprehensive integration tests.
*   **TODO.md Maintenance**: ALWAYS update `TODO.md` when:
    *   Fixing bugs (mark as completed, document the fix)
    *   Identifying new issues (add to appropriate priority section)
    *   Completing tasks (update status, move to completed section)
    *   Discovering technical debt (document in "Known Issues & Technical Debt")
    *   Making architectural decisions (document rationale and alternatives)

## Development

*  Always update the copyright banner year. Starting at 2012-[current year of edit]

## Build & Install

*   **UV-based setup (recommended)**: Use `uv sync --extra dev` for development installation with all dependencies.
*   **Traditional pip**: Use `pip install -e ".[dev]"` for development installation.
*   The project uses `setuptools` with `pyproject.toml` configuration.

## Python Environment Usage

*   **CRITICAL**: When running Python commands, ALWAYS use `uv run python` instead of bare `python`
*   The system Python has version conflicts (NumPy 2.x vs scipy/matplotlib compiled for 1.x)
*   `uv run python` uses the virtual environment with compatible package versions
*   Example: `uv run python script.py` instead of `python script.py`

## Known Issues

*   **Pandas Compatibility**: Framework uses deprecated `DataFrame.append()` method, incompatible with pandas 2.x
*   **Result Hashing**: Result objects need `__hash__` method for Splitter analyzer
*   **Dependency Management**: Some optional dependencies (matplotlib) fail with NumPy 2.x

## Running Panobbgo

Panobbgo is designed as a **framework for black-box optimization** but includes **out-of-the-box runnable examples** for testing and demonstration:

*   **Framework nature**: Panobbgo is a library that provides components (strategies, heuristics, analyzers) for building custom optimization pipelines
*   **Runnable demos**: Example scripts in `sketchpad/` demonstrate complete optimization runs (e.g., `python sketchpad/test01.py`)
*   **Testing**: Run `uv run pytest` or `pytest` to execute the test suite and verify functionality
*   **Interactive use**: Import and use components directly in Python scripts for custom optimization problems

The framework runs on **Dask distributed** for parallel evaluation, supporting both local clusters and remote distributed computing.

## Domain Context: Black-Box Noisy Optimization

Panobbgo solves **expensive, noisy black-box optimization** problems. Key domain constraints:

*   **Evaluation budget is a hard cap**: Each objective function evaluation can be computationally expensive (minutes to hours). The `max_eval` budget is a strict limit, not a soft target. Strategies must respect phase boundaries and never allow in-flight evaluations to overshoot budget allocations.
*   **Candidate generation vs evaluation**: Generating candidate points (proposals) is cheap. *Evaluating* them is expensive. It's fine to generate more candidates than needed, but submitting them for evaluation must be carefully controlled against the budget.
*   **Restart capability**: Long-running optimizations need persistence. The storage/database backend enables checkpointing and restarting from previous results, which is critical when evaluations take significant wall-clock time.
*   **Phased strategies**: When using `StrategyPhased`, each phase has a budget allocation. The system must account for pending (in-flight) evaluations when enforcing phase boundaries to prevent fast-generating strategies from consuming the next phase's budget.

## Benchmark Harness (Self-Improvement Workflow)

A reproducible benchmark harness is the **single source of truth** for
"is Panobbgo better or worse than it was before this change?".  It produces
one scalar, ``composite_score`` ∈ [0, 1], designed to gate every change an
agent makes. **Agents MUST use this when modifying strategies, heuristics,
or core optimization logic.**

Full guide: `doc/source/guide_benchmarking.rst`. Self-improvement plan:
`planning/SELF_IMPROVEMENT_LOOP.md`.

### Workflow

```bash
# 1. Capture baseline BEFORE making changes
uv run python benchmark_harness.py run --quick --output before.json

# 2. Make your changes to panobbgo

# 3. Capture results AFTER changes
uv run python benchmark_harness.py run --quick --output after.json

# 4. Compare — exits with code 2 if regressions detected
uv run python benchmark_harness.py compare before.json after.json --fail-on-regression
```

### Modes

*   `--quick`: 3 problems × 2 strategies × 3 reps × 75 evals (~30s) — use during development
*   `--standard`: 8 problems × ~6 strategies × 5 reps × 200 evals (~few min) — use before merging
*   `--full`: 11 problems × ~10 strategies × 10 reps × 500 evals (~1h) — thorough validation

### External baselines

Add ``--baselines`` to include three external reference solvers
(``Baseline_Random``, ``Baseline_SciPyDE``, ``Baseline_SciPyAnneal``)
alongside the Panobbgo strategies.  These provide an **absolute** reference
point (floor + competitive DE/SA) rather than the purely relative
"Panobbgo vs its previous self" signal.  See
`panobbgo/harness_baselines.py`.

```bash
uv run python benchmark_harness.py run --standard --baselines --output standard.json
uv run python benchmark_harness.py list --standard --baselines
```

### Parametrically randomized problems

Add ``--randomize`` to swap the fixed problem battery for a parametric one
that samples fresh translated / rotated / scaled / noisy instances per
repetition.  This turns ``composite_score`` into a Monte-Carlo estimate
of expected performance on a problem family, preventing the
self-improvement loop from over-fitting to specific instances.  See
`panobbgo/harness_randomized.py`.

```bash
uv run python benchmark_harness.py list --randomize
uv run python benchmark_harness.py run --randomize --randomize-iteration 0 --output rand_before.json
# Make changes ...
uv run python benchmark_harness.py run --randomize --randomize-iteration 0 --output rand_after.json
uv run python benchmark_harness.py compare rand_before.json rand_after.json --statistical
```

``--randomize-iteration`` pins the iteration index — the same iteration
reproduces the *same* sampled instances, so ``before`` and ``after`` runs
line up; different iterations intentionally draw different instances.

Multi-dim families (``dim_choices`` with more than one element) ship with
``stratify_dims=True`` by default, so the dim assigned to rep ``i`` is
``dim_choices[i % k]`` — any contiguous block of ``k`` reps covers every
declared dim exactly once.  This eliminates dim-mix variance in the
composite delta (the bootstrap CI from ``--statistical`` would otherwise
pick up the noise of "this iteration happened to draw more 5-D
instances").  Single-dim families (the entire default battery) are
unaffected.  See :class:`panobbgo.harness_randomized.ProblemFamily` and
``planning/SELF_IMPROVEMENT_LOOP.md`` §10.

### Key files

*   `panobbgo/harness.py` — `BenchmarkHarness` class, metrics, serialization, `compare()`, and `statistical_accept()` (bootstrap-CI acceptance rule)
*   `panobbgo/harness_baselines.py` — external reference strategies (Random, SciPy DE, SciPy dual annealing)
*   `panobbgo/harness_randomized.py` — parametrically randomized problem families, transforms, and `RandomizedProblemSpec`
*   `benchmark_harness.py` — CLI tool (`run`, `score`, `compare`, `list` subcommands; `--baselines`, `--statistical`, `--randomize` flags)
*   `tests/test_harness.py` — comprehensive test suite for the harness itself
*   `tests/test_harness_baselines.py` — tests for the external baselines adapter
*   `tests/test_harness_stats.py` — tests for the statistical acceptance rule
*   `tests/test_harness_randomized.py` — tests for the parametric randomization layer
*   `doc/source/guide_benchmarking.rst` — user-facing guide
*   `planning/SELF_IMPROVEMENT_LOOP.md` — design for autonomous improvement loop

### Score interpretation

*   **1.0** = every run solves at evaluation 1 (theoretical ceiling)
*   **0.7+** = strong; strategies consistently find optima with budget left over
*   **0.3** = weak; rare successes, usually late in the budget
*   **0.0** = never found any optimum within tolerance

Per-pair metrics also reported: ``success_rate``, ``ert`` (BBOB standard),
``best_func_distance``, ``median_func_distance``.

### When to use

*   Modifying any strategy (`strategies/`)
*   Modifying any heuristic (`heuristics/`)
*   Changing core evaluation or constraint handling logic
*   Adding new benchmark problems or strategies to the registry

### Agent-driven "improve X" PRs — evidence vs. CI

**First, deduplicate.** Before implementing an improvement, run
`gh pr list --state open` (drafts included) and skim the open PR titles.
The nightly routine branches from `master` and cannot see unmerged work,
so it will re-pick an idea whose PR is still open — this produced four
duplicate NL-SHADE-RSP PRs (#227–#230) on consecutive nights. If your
idea is already in an open PR, finish/merge that one instead of opening a
duplicate. See `planning/SELF_IMPROVEMENT_LOOP.md` §12.3 step 0.

A green PR proves the change does not break tests / lint / typecheck /
docs / format / the micro pytest-benchmark suite.  **A green PR does NOT
prove the change improved ``composite_score``** — none of the PR-side CI
workflows in `.github/workflows/` execute `benchmark_harness.py`.  Only
the nightly `self_improve_nightly.yml` workflow runs the harness, and it
operates on `master`, not on PR branches.

When the user instructs the agent to "improve the default strategy" /
"push to PR" / "do not run locally" — or any equivalent phrasing that
prevents the agent from running the harness before opening the PR — the
agent must:

1.  **State the evidence form in the PR description**, not just the
    intended improvement.  Acceptable evidence, in decreasing order of
    strength:

    *   A locally-captured `before.json` / `after.json` pair compared
        with `benchmark_harness.py compare --statistical
        --fail-on-regression --paired` (see "Workflow" above).  The PR
        description should include the composite delta and CI bounds.
    *   An entry in `planning/self_improve_ledger.jsonl` whose
        `proposal` matches the exact change being shipped, where
        `accepted: true`, the CI lower bound > 0, and no per-pair
        regression exceeds `eps_regress`.  Cite the iteration number
        and `(base_seed, randomize_iteration)`.
    *   *Not acceptable as the only evidence:* "this matches a
        configuration used by another strategy in the codebase",
        literature analogy, or "the docstring says it should help".
        These are reasonable *motivations* but they do not quantify the
        delta on this strategy on this problem battery.

2.  **Be explicit when an evidence form is missing.**  If part of the
    change is supported by ledger evidence and another part is argued
    by analogy, the PR description must say so.  Future autonomous
    runs will accumulate data only if each PR's claim is honest about
    what was measured.

3.  **Queue the change for follow-up measurement.**  If shipping
    without a harness run, the merged change becomes the new seed
    spec for the next nightly self-improvement run.  That run will
    re-measure the seed on a fresh randomized iteration (and the
    hold-out base_seed) and either accumulate confirming evidence in
    the ledger or surface a regression via the anti-cherry-pick
    guard.  No additional action is required from the agent, but the
    PR description should flag that the change is "pending nightly
    validation" so the user knows where to look for the post-merge
    composite delta.

Cumulative improvement across many such PRs requires that each PR's
claim be either backed by measurement up-front or queued for
post-merge measurement; otherwise the project's "is it better than
master?" signal degrades over time as analogies stack.

### Statistical rigor

The composite score is **noisy** at `--quick` mode (3 reps). A delta of
±0.02 is within noise.

*   Treat quick-mode deltas as *trend signals*, not proof.
*   For deltas in the `+0.01` to `+0.03` range, re-run at a second seed
    (`--seed 43`) before accepting the change.
*   Before merging a significant algorithmic change, run `--standard` or
    `--full` on a machine you are not actively using.
*   The composite-score formula itself is a **stable contract**:  do not
    change it without an architectural decision record — historical
    comparisons depend on it.

For principled accept/reject decisions, add `--statistical` to the
`compare` subcommand:

```bash
uv run python benchmark_harness.py compare before.json after.json \
    --statistical --fail-on-regression
```

This applies the rule from `planning/SELF_IMPROVEMENT_LOOP.md` §6.2 —
bootstrap a 95% confidence interval on the composite delta, then accept
iff (a) `delta > eps_accept` (default `0.005`), (b) the CI lower bound is
`> 0`, and (c) no single `(problem, strategy)` pair regresses by more
than `eps_regress` (default `0.05`).  Exit code is `2` on rejection when
combined with `--fail-on-regression`, so this is usable as a CI gate.
See `panobbgo.harness.statistical_accept` for the programmatic API.

The bootstrap defaults to a **paired (rep-aligned) sampler** when
``n_before == n_after`` on at least one shared pair — the case under
``--randomize`` because the harness keys instance seeds on
``(base_seed, randomize_iteration, family, rep)`` and rep ``i`` on
each side sees the *same* sampled instance.  Paired sampling is
typically 3–10× narrower than the historical independent-resample
scheme on the loop's regime (5 reps × ~3 problems at quick mode) and
is what unblocks acceptance of moderate-but-real improvements.  Force
the scheme explicitly with `--paired` / `--unpaired` (mutually
exclusive) on `compare --statistical` and on
`scripts/self_improve.py run`; use `--unpaired` when reps are NOT
instance-aligned (e.g. comparing two ledgers built with different
`base_seed` values).

### Agent self-improvement loop (in progress)

The harness is the measurement substrate for an autonomous
"measure → propose → apply → measure → accept/revert" loop. See
`planning/SELF_IMPROVEMENT_LOOP.md` for:

*   The parametrically-randomised problem battery (rotations, shifts,
    conditioning, noise) that prevents over-fitting to fixed instances —
    **shipped**, see `--randomize` and `panobbgo/harness_randomized.py`.
*   External absolute baselines (scipy DE, CMA-ES, random search) so the
    number judges Panobbgo in absolute, not just relative, terms —
    **shipped**, see `--baselines` and `panobbgo/harness_baselines.py`.
*   Statistical acceptance rules (bootstrap CI on score delta) —
    **shipped**, see `--statistical` and `panobbgo.harness.statistical_accept`.
*   Loop driver — **shipped** (Phase 5), see `panobbgo/self_improve.py`
    and `scripts/self_improve.py` (`run` / `summary` subcommands).
*   Anti-cherry-pick guard — **shipped** (Phase 6.3), see
    `LoopConfig.guard_interval` / `guard_eps_ladder` /
    `guard_iteration_offset` and the `--guard-interval`,
    `--guard-eps-ladder`, `--guard-iteration-offset` CLI flags.  The
    guard periodically re-measures the top of the accepted ladder on
    a fresh randomized seed and rolls back if the composite drifts
    below tolerance, catching instance cherry-picking.
*   Adaptive mutation sampler (Thompson sampling) — **shipped** (§10),
    see `panobbgo.self_improve.AdaptiveMutationSampler` and the
    `--adaptive`, `--adaptive-prior-alpha`, `--adaptive-prior-beta`,
    `--adaptive-prime-from-ledger` CLI flags.  The sampler treats each
    mutation rule as a Bernoulli arm with reward = "iteration was
    accepted" and biases future samples toward rules with positive
    accept history while still exploring under-tried rules.
    Cold-start (Beta(1, 1) prior) is statistically identical to uniform
    sampling.  Per-class structural bandit arms
    (**shipped 2026-05-18**) split each ``add_heuristic`` /
    ``drop_heuristic`` op into one arm per candidate class so the
    bandit can learn that, e.g., ``add Sobol`` wins while ``add
    Random`` loses — opt in with
    `LoopConfig.structural_per_class_arms` or the
    `--structural-per-class-arms` CLI flag (only effective with
    `--adaptive`).
*   Strategy portfolio composition (§7.2) — **shipped**, see
    `panobbgo.self_improve.StructuralMutationRule` and the
    `default_structural_catalog()` factory.  Two ops join the mutation
    catalog: `add_heuristic` (append from a curated pool —
    Random/Nearby/NelderMead/Center/LatinHypercube/Sobol/Extremal,
    the PSO topologies, the DE family L-SHADE/jSO/NL-SHADE-RSP, and the
    local optimizers COBYQA and LBFGSB (the only gradient-based arm);
    `avoid_duplicates` skips classes already present) and
    `drop_heuristic` (remove subject to a
    `min_heuristics` post-drop floor).  Opt in via
    `scripts/self_improve.py run --structural` or
    `SelfImprover(catalog=default_structural_catalog())`.  Off by
    default so existing CLI invocations remain byte-identical.
*   Hold-out validation set — **shipped** (§10), see
    `panobbgo.self_improve.LoopHoldoutRecord` and the
    `--holdout-base-seed`, `--holdout-iterations`,
    `--holdout-iteration-offset`, `--holdout-eps-overfit`,
    `--fail-on-overfit` CLI flags.  At the end of every loop run, the
    seed and final-top of the ladder are re-measured on instances
    drawn from a completely independent ``base_seed`` SHA-256 stream;
    a shrinking ``top - seed`` gap (``drift < -eps_overfit``) flags
    overfit to the training base_seed family — the failure mode the
    anti-cherry-pick guard cannot see.  Multi-seed hold-out
    (**shipped 2026-05-16**) extends this with
    `LoopConfig.holdout_base_seeds` (list-typed) and
    `--holdout-base-seeds 1234,5678,9012`: one record per seed is
    written and the CLI aggregates with worst-case drift /
    any-overfit semantics — a more robust generalisation check
    than a single independent draw.  Bootstrap-CI aggregation
    (**shipped 2026-05-17**) layers a statistical test on top:
    :func:`panobbgo.self_improve.aggregate_holdout_drift` pools
    per-iteration paired drifts across every hold-out record and
    emits a CI on the mean drift; the `--fail-on-overfit-ci` CLI
    flag fires iff the CI's upper bound falls below
    ``-holdout_eps_overfit`` — a stricter, less-noise-reactive
    sibling of `--fail-on-overfit` that pairs with the
    `statistical_accept` rule already in `panobbgo.harness`.  See
    `doc/source/guide_benchmarking.rst` "Bootstrap CI on the
    aggregated drift".
*   Categorical mutation rule — **shipped**, see
    `panobbgo.self_improve.MutationRule` with
    `kind="categorical_choice"`.  Picks uniformly from a discrete
    `choices` tuple while always excluding the current value (no-op
    mutations are eliminated by construction).  The default catalog
    ships five categorical rules out-of-the-box: `PSO.topology`
    (`"gbest"` ↔ `"lbest"` ↔ `"vonneumann"`), `Sobol.scramble` (`True` ↔ `False`),
    `LSHADE.archive_factor` (`0.0` / `1.0` / `2.6`),
    `LSHADE.F_schedule` (`True` ↔ `False`), and
    `NLSHADE_RSP.adaptive_archive` (`True` ↔ `False`).  Each fires only
    when the target spec sets the kwarg *explicitly* — the
    "param already in kwargs" predicate keeps the rule out of specs
    that left the kwarg at the heuristic's constructor default.
    `LSHADE.F_schedule=True` enables the jSO (Brest et al. 2017)
    three-phase asymmetric F-cap on L-SHADE (`F ≤ 0.7` while
    `progress < 0.6`, `F ≤ 0.8` while `0.6 ≤ progress < 0.9`,
    unclamped in the final 10%); jSO opts into the cap by
    construction so `JSO` is always literature-faithful regardless of
    the catalog rule's verdict on L-SHADE.

Run the loop:

```bash
# 5 quick iterations
uv run python scripts/self_improve.py run --iterations 5

# Long run with the anti-cherry-pick guard every 10 iterations
uv run python scripts/self_improve.py run --iterations 100 \
    --mode standard --guard-interval 10 --guard-eps-ladder 0.02

# Adaptive (Thompson-sampling) mutation sampler primed from a prior ledger
uv run python scripts/self_improve.py run --iterations 100 \
    --adaptive --adaptive-prime-from-ledger

# Structural catalog: kwarg perturbations + add_heuristic / drop_heuristic ops
uv run python scripts/self_improve.py run --iterations 100 \
    --structural --adaptive

# Per-class structural bandit arms — splits each add_heuristic /
# drop_heuristic op into one arm per candidate class so the bandit
# can distinguish "add Sobol" from "add Random".  Only effective
# with --adaptive.
uv run python scripts/self_improve.py run --iterations 100 \
    --structural --adaptive --structural-per-class-arms \
    --adaptive-prime-from-ledger

# End-of-loop hold-out validation against an independent base_seed,
# fail with exit code 3 if the ladder is flagged as overfit
uv run python scripts/self_improve.py run --iterations 100 \
    --mode standard --holdout-base-seed 1234 --fail-on-overfit

# Multi-seed hold-out: more robust drift estimate over several
# independent SHA-256 streams.  Worst-case drift across seeds is
# reported; --fail-on-overfit fires if any single seed flags overfit.
uv run python scripts/self_improve.py run --iterations 100 \
    --mode standard --holdout-base-seeds 1234,5678,9012 \
    --fail-on-overfit

# Bootstrap-CI on the aggregated multi-seed hold-out drift.  Stricter
# exit rule than --fail-on-overfit: fires only when the CI's upper
# bound falls below -holdout_eps_overfit at the configured confidence.
uv run python scripts/self_improve.py run --iterations 100 \
    --mode standard --holdout-base-seeds 1234,5678,9012 \
    --fail-on-overfit-ci --holdout-ci-confidence 0.95

# Inspect the ledger
uv run python scripts/self_improve.py summary
```

## IOH / MA-BBOB Anytime competition harness

A **parallel measurement track** to the composite-score harness above:
this one scores Panobbgo on the IOHprofiler MA-BBOB suite using the
**AOCC** (Area Over the Convergence Curve) metric used by the MA-BBOB
Anytime competition.  Run via `scripts/ioh_benchmark.py`; the data
structures live in `panobbgo/harness_ioh.py` and wrap
`panobbgo/ioh_runner.py` (atomic budget-enforced runner) and
`panobbgo/lib/ioh_wrapper.py` (`IOHProblem` adapter — see below).

### Worker-subprocess architecture

The `ioh` PyPI wheel only ships cp311 / cp312 binaries (no cp313 yet),
so it cannot be installed into a Python 3.13 panobbgo venv without
compiling pybind11 from source.  To keep the panobbgo core free to move
to newer Python, the `ioh` import lives in an **isolated child uv
project** at `tools/ioh_worker/`, pinned to Python 3.12:

```
tools/ioh_worker/
├── pyproject.toml          # panobbgo-ioh-worker, requires-python >=3.11,<3.13
├── .python-version         # 3.12 (cp312 wheels available)
├── README.md               # JSON-Lines protocol spec
└── src/ioh_worker/
    ├── __init__.py
    └── __main__.py         # protocol loop
```

`panobbgo.lib.ioh_wrapper.IOHProblem(kind, instance, dim, ...)` spawns
the worker as `uv run --project tools/ioh_worker python -m ioh_worker`
and proxies `eval(x) -> fx` calls over JSON-Lines on stdin/stdout
behind a per-instance `threading.Lock`.  The parent panobbgo process
never imports `ioh`; the main `pyproject.toml`'s `benchmark` extra
no longer carries `ioh` either.

**First-time setup:**

```bash
cd tools/ioh_worker && uv sync
```

This downloads the cp312 manylinux/macOS wheel directly — no C++
compile, no memory pressure.  After this, all IOH tests and the
benchmarks below run normally.

If `tools/ioh_worker/.venv` does not exist, IOH-related tests skip
with the `requires_worker` marker (the worker is optional from the
main project's standpoint).

```bash
# Quick run (~10s, 3 instances, dim 2, default Panobbgo strategies)
uv run python scripts/ioh_benchmark.py run --quick --baselines

# Standard run (~min) — dims 2 & 5, 5 instances, budget 500*d
uv run python scripts/ioh_benchmark.py run --standard --baselines --output ioh_before.json

# Save & diff
uv run python scripts/ioh_benchmark.py run --quick --output ioh_after.json
uv run python scripts/ioh_benchmark.py compare ioh_before.json ioh_after.json
```

The AOCC metric is computed against log-precision targets `[1e-8, 1e2]`
(IOH default) and **right-pads short trajectories** with the final
best-fx so that a strategy which stops early gets penalised for the
unused budget — this is how IOH itself scores the competition.

`composite_score` and AOCC do not interconvert.  A change can improve
one and regress the other; track both.

The IOH harness ships its own strategy registry —
`panobbgo.harness_ioh.make_ioh_strategies()` — tuned for the anytime
metric (larger Sobol initial design, `Restart` analyzer always on,
no `stop_on_convergence` interference).  This is the default for
`scripts/ioh_benchmark.py run`; pass `--legacy` to score using the
composite-score harness's `_make_quick_strategies` / `_make_standard_strategies`
instead.

### Self-improvement loop on AOCC

`scripts/self_improve.py run` accepts `--metric aocc` to optimise for
the IOH/MA-BBOB anytime metric instead of `composite_score`.  Under
`--metric aocc`:

* The seed strategy registry comes from
  `panobbgo.harness_ioh.make_ioh_strategies()`.
* Per-iteration measurement runs the IOH harness on a battery whose
  size matches `--mode` (quick / standard / full →
  `make_quick_battery` / `make_standard_battery` / `make_full_battery`).
* Each :class:`~panobbgo.harness_ioh.IOHHarnessResult` is adapted via
  :func:`~panobbgo.harness_ioh.aocc_to_harness_result`, encoding per-run
  AOCC into a synthetic `first_success_eval` so the existing bootstrap
  CI, ledger writer, guard, and hold-out machinery all work unchanged.
  The ledger's `baseline_score` / `candidate_score` fields then carry
  **mean AOCC**, not composite_score — interpret accordingly.

```bash
# Five iterations of mutation search against the MA-BBOB anytime metric
uv run python scripts/self_improve.py run --iterations 5 --metric aocc
```

## CI/CD and Testing

*   **Local Testing**: Run `./test.sh` to replicate the full CI pipeline locally
*   **CI Status**: Check GitHub Actions status with `gh pr checks <PR_NUMBER>` or `gh run list`
*   **CI Logs**: View detailed CI logs with `gh run view <RUN_ID> --log` or `gh run view --web <RUN_ID>`
*   **Code Formatting**: Use `uv run ruff format` to format code, `./codestyle.sh` for convenience
*   **Linting**: Run `uv run flake8 panobbgo` to check for style issues
*   **Flaky Tests**: Stochastic/integration tests that may intermittently fail should be decorated with `@pytest.mark.flaky(retries=3)` (provided by `pytest-retry`). This retries the test up to 3 times before reporting failure.
