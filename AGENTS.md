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
    `--adaptive`).  Graded reward shaping (**shipped 2026-06-13**, §7.4)
    replaces the binary +1/+0 accept/reject signal with a continuous
    reward in `[0, 1]` derived from the bootstrap CI / point delta:
    `0.5 + clip(ci_low/(4·eps_accept), 0, 0.5)` on accept,
    `clip(0.5 + delta/(4·eps_accept), 0, 0.5)` on reject.  The new
    `MutationRuleStats.reward_sum` field accumulates the graded value
    and the Thompson posterior swaps it in for `n_accepts`, so an
    "honest near miss" reject (Δ ≈ 0) carries `r ≈ 0.5` of evidence
    instead of zero.  Opt in via `LoopConfig.bandit_reward_shaping =
    "graded"` or the `--bandit-reward graded` CLI flag.  Each
    `LoopIterationRecord` persists the graded reward in
    `bandit_reward` so `prime_from_ledger` recovers the full
    `reward_sum` state on resume; legacy ledgers (no `bandit_reward`
    key) fall back to the binary semantic byte-identically.
    Archive-aware priming (**shipped 2026-06-15**, V2 §2.6 / §9.5
    step 4) adds
    :meth:`~panobbgo.self_improve.AdaptiveMutationSampler.prime_from_archives`
    plus the matching `LoopConfig.adaptive_prime_include_archives`
    field and `--prime-include-archives` CLI flag: when paired with
    `--adaptive --adaptive-prime-from-ledger`, the bandit
    additionally replays every archived ledger under
    `<dirname(ledger_path)>/done/` matching
    `self_improve_ledger_*.jsonl` (chronological by filename) before
    the live ledger.  Per-record semantics are byte-identical to the
    live ledger path (no-op skip, graded reward, guard / skip
    filter — all share the same `_consume_record` helper).  An
    explicit override is available via the
    `adaptive_prime_archive_dir` field / `--prime-archive-dir` CLI
    flag.  Closes the §2.6 V2 "archives in `planning/done/` are
    invisible" diagnosis: the bandit posterior now accumulates
    evidence across every retained nightly run rather than
    forgetting every pre-rotation observation.  Missing or empty
    archive directories are silent no-ops so the flag is safe to
    enable on first-night runs.
*   Same-night confirmation gate (§6.4) — **shipped 2026-06-14**, see
    `panobbgo.self_improve.LoopConfirmRecord`,
    `LoopConfig.confirm_accepts` / `confirm_iteration_offset`, and the
    `--confirm-accepts` CLI flag.  Every screening-accepted candidate
    is re-measured on a fresh `randomize_iteration` (default offset
    `500_000`, distinct from the guard's `1_000_000`) — and, when at
    least one hold-out base_seed is configured, additionally on the
    *first* hold-out seed — then `statistical_accept` is re-run on the
    *pooled* (screen + confirm) sample.  Promotion happens only when
    the pooled bootstrap CI still clears `eps_accept`; a screening
    noise spike can no longer drive a promotion because the
    confirmation batch is independent and the pooled CI rules it out.
    Failed confirmations land as `record_type="confirm_reject"` records
    carrying screen + confirm + pooled scores so an auditor can trace
    whether the gate caught a noise spike (`screen_Δ ≫ confirm_Δ`) or
    a systematic regression (`screen_Δ ≈ confirm_Δ` but `ci_low ≤ 0`);
    the accompanying `LoopIterationRecord` carries a new
    `confirmed: Optional[bool]` field (`None` / `True` / `False`) so
    codify-scan distinguishes confirmed accepts from overturned
    screening accepts without re-deriving the verdict.  The bandit
    reward path consumes the *post-confirmation* pooled decision, so
    an arm that consistently produces screening noise-spike accepts
    collects the reject-regime reward (binary: `0`; graded:
    `clip(0.5 + pooled_Δ/(4·eps), 0, 0.5)`) rather than the full-
    accept reward the screening alone would have produced.  Off by
    default to keep existing CLI invocations byte-identical; recommended
    on for unattended cron runs since the gate directly addresses the
    V2 §2.2 "Accept → rollback churn" diagnosis (15/16 V1 accepts
    rolled back by the guard).
*   Strategy portfolio composition (§7.2) — **shipped**, see
    `panobbgo.self_improve.StructuralMutationRule` and the
    `default_structural_catalog()` factory.  Four ops join the mutation
    catalog: `add_heuristic` (append from a curated pool —
    Random/Nearby/NelderMead/Center/LatinHypercube/Sobol/Extremal,
    the PSO topologies, the DE family L-SHADE/jSO/NL-SHADE-RSP, and the
    local optimizers COBYQA and LBFGSB (the only gradient-based arm);
    `avoid_duplicates` skips classes already present),
    `drop_heuristic` (remove subject to a
    `min_heuristics` post-drop floor), plus the symmetric analyzer ops
    `add_analyzer` and `drop_analyzer` (**shipped 2026-06-02** —
    candidates `Sensitivity` / `Restart`, `min_analyzers=0` since
    analyzers are non-essential).  Opt in via
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
    aggregated drift".  Vacuous-status telemetry (**shipped
    2026-06-11** — V2 §6.4 / §12.4) adds an explicit
    `LoopHoldoutRecord.status` field with values
    `"ok"` / `"overfit"` / `"vacuous"`, so a hold-out that ran
    against an empty ladder (no accepted mutations to validate)
    no longer mis-reports as `OK drift=+0.0000`.  The aggregator
    filters vacuous records out of the bootstrap (so a single
    negative-drift seed cannot be masked by a vacuous companion),
    `HoldoutDriftAggregate` exposes `vacuous_count` /
    `all_vacuous`, and both `run` and `summary` CLI commands
    surface `VACUOUS` / `VACUOUS_CI` verdicts.  Legacy ledger
    lines (no `status` field on disk) classify correctly via
    :meth:`LoopHoldoutRecord.effective_status`.
*   Categorical mutation rule — **shipped**, see
    `panobbgo.self_improve.MutationRule` with
    `kind="categorical_choice"`.  Picks uniformly from a discrete
    `choices` tuple while always excluding the current value (no-op
    mutations are eliminated by construction).  The default catalog
    ships ten categorical rules out-of-the-box: `PSO.topology`
    (`"gbest"` ↔ `"lbest"` ↔ `"vonneumann"` ↔ `"random"`), `Sobol.scramble` (`True` ↔ `False`),
    `LSHADE.archive_factor` (`0.0` / `1.0` / `2.6`),
    `LSHADE.F_schedule` (`"off"` / `"jso"` / `"early"` / `"strict"` —
    four named asymmetric F-cap regimes, shipped 2026-06-23 broadening
    the original `True` / `False` binary toggle; the bool inputs still
    work as backwards-compatible synonyms for `"jso"` / `"off"`),
    `NLSHADE_RSP.adaptive_archive` (`True` ↔ `False`),
    `NLSHADE_RSP.k_rank` (`0.0` / `3.0` / `5.0` — RSP-off /
    Stanovov default / aggressive regimes, sitting alongside the
    continuous `float_uniform` rule),
    `JSO.p_best_max` (`0.15` / `0.25` / `0.4` — L-SHADE-like /
    jSO default / iLSHADE-like greediness regimes, also sitting
    alongside the continuous `float_uniform` rule on the same
    slot; the L-SHADE setting is raised from the canonical `0.11`
    to `0.15` so it clears jSO's default `p_best_min = 0.125`
    floor — shipped 2026-06-09), `COBYQA.scale`
    (`True` ↔ `False`),
    `NLSHADE_LBC.lbc_regime` (`"cec2022"` / `"lshade"` / `"flat"`
    / `"aggressive"` — four named LBC bias regimes over the
    five-tuple `(p_F_init, p_F_final, p_CR_init, p_CR_final,
    m_lbc)`; shipped 2026-06-24, replaced the five per-field LBC
    `float_uniform` rules with one composite joint-search arm —
    mirrors the `LSHADE.F_schedule` broadening pattern),
    and `Restart.restart_strategy`
    (`"random"` / `"diverse"` / `"sphere"` — uniform-in-box /
    max-min-distance / Gaussian-around-centre center-selection
    regimes, shipped 2026-06-07).  Each fires only
    when the target spec sets the kwarg *explicitly* — the
    "param already in kwargs" predicate keeps the rule out of specs
    that left the kwarg at the heuristic's constructor default.
    `LSHADE.F_schedule="jso"` (formerly `True`) enables the Brest et al.
    2017 three-phase asymmetric F-cap on L-SHADE (`F ≤ 0.7` while
    `progress < 0.6`, `F ≤ 0.8` while `0.6 ≤ progress < 0.9`,
    unclamped in the final 10%).  The 2026-06-23 broadening adds two
    sibling regimes — `"early"` (kicks in earlier and tighter:
    `F ≤ 0.6` while `progress < 0.4`, `F ≤ 0.8` while
    `progress < 0.7`) and `"strict"` (most aggressive: `F ≤ 0.5`
    while `progress < 0.5`, `F ≤ 0.7` while `progress < 0.85`) — so
    the bandit can search the broader cap geometry on a single arm.
    jSO opts into `"jso"` by construction so `JSO` is always
    literature-faithful regardless of the catalog rule's verdict on
    L-SHADE.
*   Numeric `Restart.patience` rule (`integer_add`,
    `bounds=(3, 200)`) — **shipped 2026-06-06**, the more
    impactful of the two :class:`Restart` knobs (alongside
    the existing `Restart.max_restarts`).  Fires only when a
    spec sets `patience` to a concrete integer; the `None`
    auto-default sentinel (= `5 · dim`) is filtered by
    :func:`_find_targets`'s `None`-skip predicate.  Pairs
    with the symmetric `LBFGSB.max_starts` rule shipped in
    the same change.
*   Numeric `LBFGSB.max_starts` rule (`integer_add`,
    `bounds=(1, 50)`) — **shipped 2026-06-06**, caps the
    multi-start L-BFGS-B restart budget.  Fires only when a
    spec sets `max_starts` to a concrete integer; the `None`
    auto-default sentinel (= unlimited until budget) is
    filtered by :func:`_find_targets`'s `None`-skip
    predicate.
*   Numeric `RegionUCB.ucb_c` / `RegionUCB.gauss_fraction` /
    `RegionUCB.gauss_scale` rules — **shipped 2026-06-08**, see
    :class:`panobbgo.heuristics.region_ucb.RegionUCB`.  Three
    catalog arms covering the leaf-bandit knobs of the 2026-06-05
    RegionUCB heuristic: `ucb_c` (`log_uniform_perturb`,
    `bounds=(0.1, 4.0)`) is the UCB1 exploration weight;
    `gauss_fraction` (`float_uniform`, `bounds=(0.0, 1.0)`) is
    the fraction of in-leaf draws taken as Gaussian-around-best
    instead of uniform-over-leaf; `gauss_scale`
    (`log_uniform_perturb`, `bounds=(0.05, 0.5)`) is the Gaussian
    std-dev as a fraction of the leaf's ranges.  All three fire
    only when a spec sets the matching kwarg explicitly; the
    seed `Rewarding_RegionUCB` spec ships them at the constructor
    defaults (`{"ucb_c": 1.0, "gauss_fraction": 0.5,
    "gauss_scale": 0.25}`) so the rules become applicable on the
    standard-mode battery rather than staying dormant.
*   Numeric `PSO.stagnation_threshold` rule (`integer_add`,
    `bounds=(5, 60)`) — **shipped 2026-06-05**, see
    :attr:`panobbgo.heuristics.pso.PSO.stagnation_threshold`.
    Enables the Clerc 2007 / SPSO 2011 stochastic-K stagnation
    rebuild for the ``random`` PSO topology: the informer graph is
    re-sampled mid-run when the swarm fails to lift its global best
    for ``N`` consecutive incoming results (finer-grained than the
    restart-gated rebuild).  Fires only when a spec sets the kwarg
    explicitly; the default `None` value bypasses the policy and
    preserves the static-between-restarts behaviour shipped
    2026-05-29.  Pairs naturally with the structural-catalog
    ``random`` PSO entry.
*   Dedicated **loop seed registry** — **shipped 2026-06-10**, see
    :func:`panobbgo.harness._make_loop_strategies` plus the
    matching `LoopConfig.registry` field and `--registry
    {default,loop}` CLI flag.  Returns the two quick specs plus
    five compact family specs (`Loop_DE_Family` / `Loop_PSO` /
    `Loop_RegionUCB` / `Loop_LocalSearch` / `Loop_Restart`) with
    every tunable kwarg of the rule-bearing classes (LSHADE / JSO /
    NLSHADE_RSP / NLSHADE_LBC / LSHADE_EpSin / PSO / RegionUCB /
    COBYQA / LBFGSB / Restart) explicit at the constructor default
    so the catalog rules — which are gated on `param in kwargs` by
    `_find_targets` — actually fire on the seed instead of staying
    dormant.  Lifts catalog kwarg-rule activation from **4 / 44**
    (quick seed) to **44 / 44** (loop seed).  Closes §9.5 step 1 of
    the V2 plan and the §2.4 "catalog ≫ registry mismatch"
    diagnosis.  Independent of `--mode` (quick / standard / full
    budgets are honoured but the seed specs are the same).  Default
    `registry="default"` preserves the historical mode-based
    selection byte-for-byte.
*   **No-op detection (§12.4)** — **shipped 2026-06-12**, see
    `panobbgo.self_improve.LoopIterationRecord.no_op`, the
    `AdaptiveMutationSampler.discard_outcome` helper, and the
    `_is_no_op` predicate.  Iterations whose per-(problem, strategy)
    candidate scores are bit-identical to baseline carry zero
    information about whether the proposed mutation rule helps or
    hurts; the loop now sets `no_op=True` /
    `reason_skipped="no_op"` and bypasses the bandit pull instead of
    mis-training the Beta posterior on a zero-information event.
    `prime_from_ledger` skips no-op records on resume.  The
    `scripts/self_improve.py summary` view surfaces a separate
    `no-op=N` bucket and computes the accept rate against the
    *informative* denominator (decided − no-op).  Directly addresses
    the §2.1 "34% of mutations measure Δ = exactly 0.0000"
    diagnosis: those iterations no longer mis-train the posterior,
    and an operator can distinguish "bandit starved on dormant
    rules" from "every legitimate proposal got rejected".
    Backwards-compatible field default (`no_op=False` on legacy
    records).
*   **Summary trend block (§12.4)** — **shipped 2026-06-16**,
    `scripts/self_improve.py summary` renders three additive sub-blocks
    after the existing per-record sections: (1) **Trend** table —
    one row per loop run (oldest first) with date / base_seed / mode
    / iters / decided / accepts / no-op / best Δ / seed score columns;
    (2) **Bandit posteriors** ranked by graded `mean_reward`
    descending, configurable via `--top-n` (default 10) / `--bottom-n`
    (default 5) / `--min-attempts` (default 3), replays through the
    same `_proposal_rule_key` collapse used by
    `AdaptiveMutationSampler.prime_from_ledger` so the view matches
    what a freshly-primed nightly bandit would carry; (3)
    **Inactivity** telemetry — inferred `eps_accept` base, longest
    accept drought, relaxed-accept count, mean decay factor at the
    moment of accept.  Directly addresses the §12.3 daily-routine
    contract: an operator can now answer "is the loop accepting
    tonight?", "which arms pay off?" and "is the inactivity-relax
    knob doing anything?" in one screen of text instead of grepping
    the raw JSONL ledger.  All three blocks silently no-op on empty
    input; the Inactivity block additionally no-ops on legacy
    ledgers (pre-2026-05-30) that carry neither
    `effective_eps_accept` nor `iters_since_accept`.  See the
    2026-06-16 entry in `planning/SELF_IMPROVEMENT_LOG.md`.
*   **Cross-night codify-scan (§9.3 / §9.5 step 4)** —
    **shipped 2026-06-17**.  The §11 V2 success criteria are gated on
    *codify PRs* (criterion 2: ≥3 opened, ≥2 merged) — durable
    improvement happens only through codification (§12.2).  The new
    `scripts/self_improve.py codify-scan` subcommand reads
    `planning/self_improve_ledger.jsonl` plus every
    `planning/done/self_improve_ledger_*.jsonl` archive and groups
    every accepted iteration by `(class_name, param_name, direction)` —
    where `direction` is `"up"` / `"down"` for numeric rules,
    `repr(new_value)` for `categorical_choice`, and the op name for
    structural ops.  Groups with at least `--min-nights` (default `2`)
    distinct accept dates and every contributing record's `ci_low > 0`
    are surfaced as `panobbgo.self_improve.CodifyCandidate` objects
    sorted by `(n_distinct_nights, mean_delta, n_accepts)`.  The
    report carries pooled point-delta CI (percentile bootstrap), the
    per-record evidence with `Δ` / `CI` / `old -> new`, and the
    `slot_key` tuple a future `--open-pr` driver will dedup against
    `gh pr list --state open`.  `--confirmed-only` restricts to
    post-V2-§6.4 records; `--json` emits one
    `CodifyCandidate.to_dict()` JSON per line; `--top N` truncates
    the report.  Closes the detection half of V2 §9.5 step 4.  See
    the 2026-06-17 entry in `planning/SELF_IMPROVEMENT_LOG.md`.
*   **Already-codified suppression on codify-scan** —
    **shipped 2026-06-18**.  The scanner runs
    `panobbgo.self_improve.annotate_codified_status` after the
    aggregation pass: it imports the seed-spec factories the nightly
    cron exercises (`_make_quick_strategies` +
    `_make_loop_strategies` — see
    `panobbgo.self_improve.default_codify_registries`), walks every
    spec's `(class, kwargs)` entries, and cross-checks each
    `CodifyCandidate` against the live values.  Categorical
    candidates compare `repr(new_value) == repr(live)`; numeric
    candidates compare the median of `new_values` against the live
    value in the candidate's direction (`"up"` →
    `max(live) >= median(new_values)`; `"down"` →
    `min(live) <= median(new_values)`).  **Structural ops** —
    extended **2026-06-19** with the symmetric class-membership
    predicate (helper `_live_class_membership`):
    `add_heuristic`/`add_analyzer` codify iff *at least one* seed
    spec already lists the class in the matching bucket;
    `drop_heuristic`/`drop_analyzer` codify iff *no* seed spec lists
    it.  `live_codified_values` for a structural candidate surfaces
    the *spec names* carrying the class so the
    `--include-already-codified` audit trail still tells the
    operator where the membership lives.  Already-codified
    candidates are hidden by default so the daily routine's report
    stays on actionable evidence; pass `--include-already-codified`
    to surface the suppressed set tagged `[already codified]` with
    the matching seed kwarg values printed under a `live seed
    value(s):` line.  JSON mode (`--json`) always emits every
    candidate with the `already_codified` / `live_codified_values`
    fields so the consumer can filter itself.  On the live project
    ledger the report shrinks from 5 candidates to 4 (the
    `Sobol.scramble = False` candidate that surfaces from the
    pre-codification archive is suppressed).  See the
    2026-06-18 / 2026-06-19 entries in
    `planning/SELF_IMPROVEMENT_LOG.md`.

*   **Bidirectional-bound widening detection on codify-scan** —
    **shipped 2026-06-19**.  The codify scanner detects each
    `(class, param, direction)` group independently, so a slot whose
    bandit finds value moving the kwarg *up* on some nights and
    *down* on other nights surfaces as two competing default-shift
    candidates.  The right action for these is rarely a default
    shift but a *catalog bound update* that focuses the bandit's
    exploration on the observed range with some headroom outside.
    `panobbgo.self_improve.detect_widening_candidates` pairs every
    bidirectional `(class_name, param_name)` slot — same slot with
    accepts in *both* `"up"` and `"down"` directions across multiple
    nights — into a proposed `MutationRule.bounds` update.  Pass
    `--widen-bounds` to `scripts/self_improve.py codify-scan` to
    append a *Bound-widening candidates* section; `--widen-factor`
    (default `1.5`) controls the multiplicative widening applied to
    the observed range.  Per-kind: `log_uniform_perturb` and
    `float_uniform` use symmetric multiplicative widening;
    `integer_add` uses the same rule rounded outward
    (`floor` on the lower bound, `ceil` on the upper) with the
    lower bound clipped to `1` when observed values are positive.
    JSON mode emits widening candidates on the same line-delimited
    stream tagged `"_type": "widening_candidate"` (codify candidates
    carry the symmetric `"_type": "codify_candidate"` tag).  On the
    live project ledger today, the detector surfaces two
    bidirectional patterns — `Nearby.radius` and `Sobol.n` — both
    *tightening* candidates because the bandit consistently picks
    values in a window 5-10× narrower than the catalog admits.  See
    the 2026-06-19 entry in `planning/SELF_IMPROVEMENT_LOG.md`.

*   **Auto-tuned widen factor** — **shipped 2026-06-22**.  The
    fixed `--widen-factor` is one-size-fits-all across rules whose
    observed-spread / catalog-bound ratios differ by an order of
    magnitude.  `--widen-auto-tune` sizes the factor per candidate
    from that ratio in the rule's natural scale (log for
    `log_uniform_perturb`, linear for `integer_add` / `float_uniform`):
    narrow observed spread (high agreement) → larger factor
    (`--widen-factor-max`, default `2.5`) for exploration headroom;
    wide spread → smaller factor (`--widen-factor-min`, default
    `1.1`) focused on the consensus.  Falls back to `--widen-factor`
    when no catalog rule targets the slot.  Lifts the live
    `Nearby.radius` widen factor from a fixed 1.5 to ~2.31; proposed
    bound widens from `[0.049, 0.203]` to `[0.032, 0.313]`.  See the
    2026-06-22 entry in `planning/SELF_IMPROVEMENT_LOG.md`.

*   **Manual codify of the `Nearby.radius` widening proposal** —
    **shipped 2026-06-26**.  The auto-tuned `[0.032, 0.313]` proposal
    landed as a `default_catalog` change (bounds tightened from
    `(0.005, 0.5)`); the first widening-detector output to land as a
    catalog change.  Pure bound update — no new arms, no constructor
    changes, no behaviour change for the `Nearby` heuristic itself.
    See the 2026-06-26 entry in `planning/SELF_IMPROVEMENT_LOG.md`.

*   **Nightly cron flipped to V2 substrate** — **shipped 2026-06-21**.
    The `self_improve_nightly.yml` workflow now invokes
    `scripts/self_improve.py run` with every zero-cost V2 flag:
    `--registry loop` (catalog kwarg-rule activation 4/44 → 44/44),
    `--prime-include-archives` (bandit posterior compounds across
    rotated ledgers under `planning/done/`), `--structural-per-class-arms`
    (one Thompson arm per (op, candidate class)),
    `--bandit-reward graded` (continuous `[0, 1]` reward derived from
    the bootstrap CI / point delta), `--inactivity-relax-after 10
    --inactivity-relax-factor 0.5` (drought-relaxation per the
    docstring recommendation for unattended cron), `--holdout-base-seeds
    7,1234` (multi-seed hold-out with worst-case / any-overfit
    reduction), and `--guard-interval 10` (relaxed from 5).
    `--confirm-accepts` is *not* flipped in this change because it
    carries ~2-3× per-iteration cost; the follow-up notes flag a
    manual `workflow_dispatch` A/B as the gating step.  Closes V2
    §9.5 step 5 partially (one toggle remaining) and §2.4 "catalog
    ≫ registry mismatch" fully.  See the 2026-06-21 entry in
    `planning/SELF_IMPROVEMENT_LOG.md`.

*   **Nightly cron flipped to `--confirm-accepts`** — **shipped
    2026-06-27** (V2 §9.5 step 5 completion).  The same workflow file
    edit promotes the §6.4 same-night confirmation gate to the cron
    default by appending `--confirm-accepts` to the run command (now
    constructed as a bash array so the toggle composes cleanly).  A
    new `workflow_dispatch.inputs.confirm_accepts` boolean input
    (default `true`) is exposed so the operator can opt back into the
    screen-only regime for an explicit A/B comparison without editing
    the workflow.  The "2-3× per-iteration cost" hedge was
    re-evaluated against the live ~3.6 % accept rate: the gate only
    fires on accepts, so the worst-case per-night overhead is
    ~30-60 s (~0.7 accept events × 2 × 15 s) against the 90-min cap.
    Closes V2 §9.5 step 5 fully (only `--metric aocc` remains queued,
    blocked on IOH worker availability) and structurally closes the
    §2.2 "Accept → rollback churn" V2 diagnosis.  See the 2026-06-27
    entry in `planning/SELF_IMPROVEMENT_LOG.md`.

Run the loop:

```bash
# 5 quick iterations
uv run python scripts/self_improve.py run --iterations 5

# Catalog-exercising loop registry — shipped 2026-06-10, opt in via
# --registry loop.  Ships the two quick specs plus five compact family
# specs (Loop_DE_Family / Loop_PSO / Loop_RegionUCB / Loop_LocalSearch
# / Loop_Restart) with every tunable kwarg of LSHADE / JSO / NLSHADE_RSP
# / NLSHADE_LBC / LSHADE_EpSin / PSO / RegionUCB / COBYQA / LBFGSB /
# Restart explicit at the constructor default.  Lifts catalog
# kwarg-rule activation from 4 / 44 (quick seed) to 44 / 44 (loop seed),
# closing the §2.4 "catalog ≫ registry mismatch" gap.
uv run python scripts/self_improve.py run --iterations 30 \
    --registry loop --adaptive --structural

# Long run with the anti-cherry-pick guard every 10 iterations
uv run python scripts/self_improve.py run --iterations 100 \
    --mode standard --guard-interval 10 --guard-eps-ladder 0.02

# Adaptive (Thompson-sampling) mutation sampler primed from a prior ledger
uv run python scripts/self_improve.py run --iterations 100 \
    --adaptive --adaptive-prime-from-ledger

# Same, but also prime from archived ledgers under planning/done/
# (rotation glob ``self_improve_ledger_*.jsonl``).  Closes the
# V2 §2.6 "archives in planning/done/ are invisible" diagnosis —
# the bandit posterior compounds across nightly rotation boundaries
# rather than forgetting every pre-rotation observation.  Shipped
# 2026-06-15.  Per-record semantics (no-op skip, graded reward) are
# byte-identical to the live ledger path.
uv run python scripts/self_improve.py run --iterations 100 \
    --adaptive --adaptive-prime-from-ledger --prime-include-archives

# Structural catalog: kwarg perturbations + four structural ops
# (add_heuristic / drop_heuristic / add_analyzer / drop_analyzer).
# The analyzer ops shipped 2026-06-02 — Sensitivity / Restart candidate
# pool — extend the loop's reach beyond the heuristics bucket.
uv run python scripts/self_improve.py run --iterations 100 \
    --structural --adaptive

# Per-class structural bandit arms — splits each structural op
# (add_heuristic / drop_heuristic / add_analyzer / drop_analyzer) into
# one arm per candidate class so the bandit can distinguish "add Sobol"
# from "add Random", or "add Restart" from "add Sensitivity".  Only
# effective with --adaptive.
uv run python scripts/self_improve.py run --iterations 100 \
    --structural --adaptive --structural-per-class-arms \
    --adaptive-prime-from-ledger

# Hierarchical bandit over per-class structural arms (kappa = 0.5):
# each per-class arm's Beta posterior borrows half-weighted strength
# from the op-level aggregate so a fresh candidate class warms with
# the op's empirical accept rate instead of the symmetric Beta(1, 1)
# prior.  Only effective with both --adaptive and
# --structural-per-class-arms.
uv run python scripts/self_improve.py run --iterations 100 \
    --structural --adaptive --structural-per-class-arms \
    --structural-borrow-alpha 0.5 --adaptive-prime-from-ledger

# Auto-tuned hierarchical borrow (shipped 2026-06-25): anneal kappa
# down per-arm as that arm's own attempts accumulate.  At the recommended
# horizon h = 5, a cold arm borrows the full kappa = 1.0 from the
# op-level aggregate; at 5 per-class attempts the borrow halves; at
# 20 attempts the borrow shrinks to ~kappa / 5.  Cold-start case is
# unchanged; long-run convergence is no longer dragged toward the
# op-level mean.  Inert when --structural-borrow-alpha = 0 or when
# --structural-per-class-arms is off.
uv run python scripts/self_improve.py run --iterations 100 \
    --structural --adaptive --structural-per-class-arms \
    --structural-borrow-alpha 1.0 --structural-borrow-horizon 5 \
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

# Inactivity-guarded eps_accept relaxation (shipped 2026-05-30).  Break
# out of long accept droughts by geometrically decaying eps_accept
# after every N consecutive non-accepts, floored at min_eps_accept,
# re-tightened on the next accept.  Each iteration ledger record
# persists the effective threshold + streak so the rule is auditable.
# Recommended for the unattended nightly cron where the documented
# accept rate is 1–5%.
uv run python scripts/self_improve.py run --iterations 100 \
    --adaptive --adaptive-prime-from-ledger --structural \
    --guard-interval 10 \
    --inactivity-relax-after 10 \
    --inactivity-relax-factor 0.5 \
    --inactivity-min-eps-accept 0.001

# Inspect the ledger — also surfaces the §12.4 trend / bandit posteriors
# / inactivity-relax sub-blocks (shipped 2026-06-16) that the §12.3
# daily routine reads at a glance.
uv run python scripts/self_improve.py summary

# Wider bandit-posterior leaderboard with stricter min-attempts filter,
# useful when the ledger has accumulated dozens of rules with sparse
# evidence each:
uv run python scripts/self_improve.py summary \
    --top-n 20 --bottom-n 10 --min-attempts 5

# Scan ledger + archives for cross-night codify candidates (V2 §9.3 /
# §9.5 step 4 — shipped 2026-06-17).  Groups every accepted iteration
# by (class, param, direction) and surfaces those that fire on at least
# --min-nights distinct accept dates with every contributing record's
# CI lower bound > 0.  These are the suggestions the §12.3 daily
# routine should consider codifying into seed defaults via a PR.
uv run python scripts/self_improve.py codify-scan

# JSON output for an external dashboard / scripted PR generation:
uv run python scripts/self_improve.py codify-scan --json --top 5

# Strict mode (once V2 §6.4 confirmation gate ships): require the
# `confirmed=True` field on every contributing record.
uv run python scripts/self_improve.py codify-scan --confirmed-only

# Audit the suppressed set (shipped 2026-06-18): include candidates
# whose implied source edit is already live in the seed-spec factories
# (quick + loop registries).  Default behaviour hides these candidates
# so the daily routine sees only actionable evidence; pass this flag
# to inspect the suppressed slots — each one is tagged
# ``[already codified]`` in the report and the matching seed kwarg
# values are printed under a ``live seed value(s):`` line.
uv run python scripts/self_improve.py codify-scan --include-already-codified

# Bidirectional-bound widening detection (shipped 2026-06-19): append a
# *Bound-widening candidates* section that pairs every (class, param)
# slot whose codify-scan reports both ``up`` and ``down`` directions
# into a proposed ``MutationRule.bounds`` update.  Per-pair tag —
# ``[widens current]`` / ``[tightens current — focuses bandit on
# observed range]`` / ``[partial overlap]`` — describes the proposal
# shape.  --widen-factor (default 1.5) controls the multiplicative
# widening; JSON mode emits widening candidates on the same
# line-delimited stream tagged ``"_type": "widening_candidate"``.
uv run python scripts/self_improve.py codify-scan --widen-bounds
uv run python scripts/self_improve.py codify-scan --widen-bounds --widen-factor 2.0

# Auto-tuned widen factor (shipped 2026-06-22) — sizes the factor per
# candidate from observed-spread / catalog-bound ratio.  Narrow spread
# (high agreement) → larger factor; wide spread → smaller.  Default
# range [1.1, 2.5], override with --widen-factor-min / --widen-factor-max.
# Falls back to --widen-factor (default 1.5) when no catalog rule
# targets the slot.
uv run python scripts/self_improve.py codify-scan --widen-bounds --widen-auto-tune
uv run python scripts/self_improve.py codify-scan --widen-bounds --widen-auto-tune \
    --widen-factor-min 1.2 --widen-factor-max 4.0
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
