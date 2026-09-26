# Benchmarking and evidence

How to measure a change and what a PR has to show.  The user-facing guide
with the full background is
[`doc/source/guide_benchmarking.rst`](../source/guide_benchmarking.rst).

## Two metrics

*   **AOCC** on the IOH / MA-BBOB and family batteries is the **metric of
    record** (`planning/GOAL.md`).
*   **`composite_score`** ∈ [0, 1] on the internal battery
    (`benchmark_harness.py`) is a frozen legacy contract: keep it from
    regressing, do not optimize for it.  Its formula (`panobbgo/harness.py`)
    and the default randomized battery change only with an architectural
    decision record — extend through opt-in flags.
*   The two do not interconvert; a change can improve one and regress the
    other.  Measure whenever you change a strategy, a heuristic, core
    evaluation or constraint handling, or a benchmark registry.

## Evaluations, not wall time

Progress is counted in **objective evaluations**.  Every measurement runs
with `sync_eval`, seeded, so a run is bit-reproducible and independent of
machine speed and load.  That is why a busy laptop and a GitHub runner give
the same numbers, and why time-based termination or stall guards are never
a quality signal.

## Evidence for a PR

*   Say what was measured, how, and what was not.  Call unmeasured claims
    unmeasured.
*   **Bug fixes** that restore documented or published behaviour land
    without a benchmark.
*   **Tuning or default changes**: a quick paired multi-seed comparison
    (`harness_ioh.paired_seed_stats`) is enough — quote delta, CI and
    wins/n.
*   Keep evidence rules lean (Harald, 2026-09-25: the former 12-seed roster
    gate is gone).  Two rules remain because they were learned the hard
    way:
    *   **The best spec of a multi-spec screen is a selected maximum.**
        Re-check it on fresh seeds before believing its margin
        (DISCOVERY §30: +0.050 on 3 seeds became −0.006).
    *   **Variants of one arm share a `StrategySpec.seed_name`**, otherwise
        the A/B carries full run-to-run variance and a parameter that is
        never read still shows a delta (DISCOVERY §18).
*   Measured null floor on a 3-seed standard-battery mean: **±0.05** for a
    CMA-ES spec, **±0.03** for a DE arm.  Smaller deltas are a direction,
    not an effect.  A claimed effect also needs a mechanism: which code path
    reads the parameter?

## Composite harness

```bash
uv run python benchmark_harness.py run --quick --output before.json   # on master
uv run python benchmark_harness.py run --quick --output after.json    # with the change
uv run python benchmark_harness.py compare before.json after.json --statistical --fail-on-regression
```

*   `--quick`: 3 problems × 2 strategies × 3 reps × 75 evals (~30 s);
    ±0.02 is noise, re-run at `--seed 43` before trusting a small delta.
*   `--standard`: 7 × 8 × 5 reps × 200 evals (minutes).
*   `--full`: 10 × 12 × 10 reps × 500 evals (~1 h).
*   `--baselines` adds `Baseline_Random`, `Baseline_SciPyDE`,
    `Baseline_SciPyAnneal` as an absolute reference
    (`panobbgo/harness_baselines.py`).  The external baselines — pycma
    IPOP/BIPOP, Nevergrad NGOpt/CMA/TwoPointsDE, Optuna CMA-ES/TPE
    (`uv sync --extra baselines`) — join only when `--strategies` names
    them, e.g. `--baselines --strategies Baseline_NGOpt`; the same works for
    `ioh_benchmark.py run` (guide, "External libraries").
*   `--randomize --randomize-iteration N` uses parametrically randomised
    instances; the same `N` gives the same instances
    (`panobbgo/harness_randomized.py`).
*   `--no-sync-eval` opts into the threaded evaluator; `compare` warns when
    the two sides differ.  The library default (`Config.sync_evaluation`)
    stays asynchronous.
*   `compare --statistical` bootstraps a 95 % CI on the composite delta and
    accepts iff delta > 0.005, the CI lower bound > 0 and no
    (problem, strategy) pair regresses by more than 0.05
    (`panobbgo.harness.statistical_accept`).  The bootstrap is paired when
    both sides have the same rep count; `--paired` / `--unpaired` force it.

Score reading: 1.0 = solved at evaluation 1, 0.7+ strong, 0.3 weak, 0.0
never within tolerance.  Per-pair metrics: `success_rate`, `ert`,
`best_func_distance`, `median_func_distance`.

Key files: `panobbgo/harness.py`, `benchmark_harness.py` (`run`, `score`,
`compare`, `list`), `tests/test_harness*.py`.

## IOH / MA-BBOB and families (AOCC)

`scripts/ioh_benchmark.py` with `panobbgo/harness_ioh.py`.  The IOH
batteries need the isolated worker venv (`cd tools/ioh_worker && uv sync`,
see `tools/ioh_worker/README.md`); without it the IOH tests skip.

```bash
uv run python scripts/ioh_benchmark.py run --quick --baselines
uv run python scripts/ioh_benchmark.py run --standard --baselines --output ioh_before.json
uv run python scripts/ioh_benchmark.py compare ioh_before.json ioh_after.json
```

The same CLI runs generated problem families (`panobbgo/harness_families.py`,
no worker venv): `--families` (5 families × dims 2/5/10 × 3 instances),
`--families-constrained` (4 families × dims 2/5, 1–3 constraints active at
the optimum, AOCC on `f + 100·cv`) and `--families-quick` (smoke test).
Multi-seed screens: `benchmarks/family_screen.py`, `portfolio_screen.py`,
`arm_sweep.py`, `oracle.py` (guide, "Per-arm sweeps").  `family_screen.py`
also has `preset=shapes` (BBOB f6/f7/f12/f21/f24 × dims 2/5/10) and
`preset=failure` (failure regions where evaluations crash or time out,
`lib.families.FailureRegion`; failed calls are spent budget, and a run that
stops below its budget is recorded as `EndedEarly`).

## Comparability

Result files from before **2026-09-25** are not comparable with newer ones:
runs now stop at exactly `max_eval`, composite `success` means "tolerance
met within the budget", measurements run `sync_eval`, and module RNG streams
are keyed by master seed and module name (`StrategyBase.spawn_rng`), which
changed every seeded trajectory.  Re-baseline instead of comparing across
that line (`TODO.md`, "Re-baseline once").

## Re-baselining on GitHub runners

`.github/workflows/rebaseline.yml` (manual) re-measures the references:
composite quick/standard, IOH quick/standard and the family screens,
sharded over (suite × seed chunk), ~28 jobs for 12 seeds.  Every shard is
seeded, `sync_eval`, with no wall-clock limit, so the numbers do not depend
on runner speed.  Suites and chunk sizes: `scripts/rebaseline.py`.

```bash
gh workflow run rebaseline.yml -f suites=all -f seeds=12        # seeds: a count or '42,7'; -f ref=<sha>
gh run download <RUN_ID> --pattern 'shard-*' --dir rebaseline-raw
uv run python scripts/rebaseline.py aggregate rebaseline-raw    # -> planning/results/<UTC date>/ref_*
```

The workflow's last job also aggregates and uploads `rebaseline-references`:
`ref_composite_<mode>_s<seed>.json` (for `benchmark_harness.py compare`),
`ref_ioh_<battery>.json` (multi-seed; compare against a run with the same
`--seeds`) and single-seed `ref_ioh_<battery>_s<seed>.json`,
`ref_family_screen_<preset>.json` (`family_screen.py from=FILE`) and
`ref_MANIFEST.json` (commit, seeds, failed shards).
