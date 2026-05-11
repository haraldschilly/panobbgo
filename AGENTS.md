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
    sampling.
*   Strategy portfolio composition (§7.2) — **shipped**, see
    `panobbgo.self_improve.StructuralMutationRule` and the
    `default_structural_catalog()` factory.  Two ops join the mutation
    catalog: `add_heuristic` (append from a curated pool such as
    Random/Nearby/NelderMead/Sobol/PSO/LSHADE/...; `avoid_duplicates`
    skips classes already present) and `drop_heuristic` (remove
    subject to a `min_heuristics` post-drop floor).  Opt in via
    `scripts/self_improve.py run --structural` or
    `SelfImprover(catalog=default_structural_catalog())`.  Off by
    default so existing CLI invocations remain byte-identical.
*   L-SHADE adaptive Differential Evolution (CEC-2014 winner) —
    **shipped**, see `panobbgo.heuristics.lshade.LSHADE`.  Combines
    success-history adaptation of F/CR, *current-to-pbest/1* mutation
    with an external archive, and Linear Population Size Reduction.
    Available in the structural catalog's `add_heuristic` pool and
    via a kwarg rule (`LSHADE.H`).  Opt-in like PSO: not added to
    any default strategy, so existing CLI invocations stay
    byte-identical.

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

# Inspect the ledger
uv run python scripts/self_improve.py summary
```

## CI/CD and Testing

*   **Local Testing**: Run `./test.sh` to replicate the full CI pipeline locally
*   **CI Status**: Check GitHub Actions status with `gh pr checks <PR_NUMBER>` or `gh run list`
*   **CI Logs**: View detailed CI logs with `gh run view <RUN_ID> --log` or `gh run view --web <RUN_ID>`
*   **Code Formatting**: Use `uv run ruff format` to format code, `./codestyle.sh` for convenience
*   **Linting**: Run `uv run flake8 panobbgo` to check for style issues
*   **Flaky Tests**: Stochastic/integration tests that may intermittently fail should be decorated with `@pytest.mark.flaky(retries=3)` (provided by `pytest-retry`). This retries the test up to 3 times before reporting failure.
