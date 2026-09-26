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
*   **Development speed first** (Harald, 2026-09-26).  Panobbgo is in
    development mode with no outside users, so changing a default needs no
    extra gate.  No `--standard`/`--full` runs before merging.
*   **Bug fixes** land without a benchmark.
*   **Improvements** land on any real evidence: a quick paired multi-seed
    comparison (`harness_ioh.paired_seed_stats`) — quote delta, CI and
    wins/n.
*   Either way, a **review agent** checks the PR before merge
    ([process.md](process.md#review-and-merge)); lean on reviews, not on
    long benchmark runs.
*   Two rules stay because they were learned the hard way:
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
`lib.families.FailureRegion`; failed calls are spent budget).  In every IOH
and family run, a run that stops by itself below its budget (every arm
stopped producing) is scored on its short trace but marked `EndedEarly` in
`IOHRunRecord.error`, next to crashed and timed-out runs in the summary.

**Dimensions 30/40 (opt-in).**  The frozen presets keep their dims; every
family preset takes `dims=30,40` in `family_screen.py`.  New presets:
`ioh_benchmark.py run --large` (MA-BBOB, d 30/40, instances 0–2, 500·d),
`--families-large` / `family_screen.py preset=large` (free + shapes
families at d 30/40), and `--largescale` (plain BBOB f2/f8/f10/f15/f21 at
d 80/160, 200·d; full rotations, so not comparable with COCO's
`bbob-largescale` numbers).  Measured cost (2026-09-26, laptop, niced): at
d = 40 and 500·d (20 000 evaluations) a run takes 1.3–3.4 s on the families
and 6.6–8.6 s on MA-BBOB (0.07–0.17 and 0.3–0.45 ms per evaluation); a
d = 160 BBOB run at 100·d takes 5–14 s.  The large presets are ≈ 1–3 min
per strategy and seed, the largescale slice ≈ 10–20 min.

**Parallel behaviour (virtual clock).**  `--virtual-workers Q` runs every
strategy on a deterministic simulation of Q workers
(`panobbgo/virtual_clock.py`, `evaluation.method = "virtual"`; no real
waiting, bit-reproducible) and scores `aocc_time` (AOCC over virtual time,
horizon budget/Q mean durations) next to `aocc`.  Benchmark at
q ∈ {1, 4, 16, 64}, one output file per q, the same q, duration model and
policy on both sides of a compare (`compare` warns on a mismatch and refuses
to gate):

```bash
for q in 1 4 16 64; do
  uv run python scripts/ioh_benchmark.py run --families --virtual-workers $q \
      --duration lognormal --output virtual_q$q.json
done
```

The default `--virtual-policy async` decides at every completion and asks the
strategy for at most the free workers (`StrategyBase.request_cap`; q = 1:
strictly one call at a time) — an idealized pull-when-free loop that the
real threaded loop does not implement yet (`TODO.md`).  `--virtual-policy
sync` is a regression mode: with q = 1, `--duration constant` and
`dask.local.n_workers = 1` it reproduces the `sync_eval` run exactly (then
`aocc_time == aocc`).  Failed and timed-out calls count as spent evaluations
in both metrics.  The ask/tell external baselines run on the same clock; the
SciPy baselines get no `aocc_time`.  Metric, model and the metric's caps
across q: guide, "Parallel behaviour on a virtual clock".

## The sealed test set

A held-out battery for **claims only**: fresh MA-BBOB instances and fresh
family instances of every class, at d 2–40 (`panobbgo/sealed.py`).

```bash
uv run python scripts/ioh_benchmark.py run --sealed --decision-seeds --output claim_mabbob.json
uv run python scripts/ioh_benchmark.py run --families-sealed --output claim_families.json
uv run python benchmarks/family_screen.py claim.json 42 7 1234 preset=sealed
```

*   **Run it only to report a result or back a claim** (a README number, a
    paper table, "panobbgo beats X").  **Never tune, screen, select a spec
    or default, or train a model on it** — not even "just to check".
*   A number from it that steers a decision burns the set: log it, and
    draw a new sealed set (new reserved ids and seed in
    `panobbgo/sealed.py`) before the next claim.
*   The harnesses print a warning banner whenever they run it.  The
    batteries take no knobs; the re-baseline workflow never runs it.
*   Disjoint by construction: MA-BBOB ids come from a reserved range that
    `IOHBatterySpec` refuses outside the sealed battery, and the family
    battery seed is refused outside `make_sealed_families_battery()`
    (`tests/test_large_and_sealed.py` checks both).
*   Cost at 500·d: ≈ 2 min (MA-BBOB) and ≈ 3–6 min (families) per strategy
    and seed.

## Comparability

Result files from before **2026-09-25** are not comparable with newer ones:
runs now stop at exactly `max_eval`, composite `success` means "tolerance
met within the budget", measurements run `sync_eval`, and module RNG streams
are keyed by master seed and module name (`StrategyBase.spawn_rng`), which
changed every seeded trajectory.  Compare against the post-audit
references instead: release `rebaseline-2026-09-26-run36228301268`
(DISCOVERY §54, measured before #344–#346, which are bit-identical on the
default paths; numbers in `planning/results/2026-09-26/SUMMARY.json`).
Unpack it with `scripts/rebaseline.py fetch` (next section).

## Re-baselining on GitHub runners

`.github/workflows/rebaseline.yml` (manual) re-measures the references:
composite quick/standard, IOH quick/standard, `ioh-external` (the IOH
standard battery plus the external baselines of the `baselines` extra,
pycma / Nevergrad / Optuna, selected by the extra their class names) and
the family screens (`free`, `constrained`, `shapes`, `failure`), sharded
over (suite × seed chunk),
27 jobs for 12 seeds.  Every shard is seeded, `sync_eval`, with no
wall-clock limit, so the numbers do not depend on runner speed.  Suites,
chunk sizes and the cost estimates behind them: `scripts/rebaseline.py`
(`ioh-external` ≈ 10 min per seed serially, dominated by NGOpt and Optuna
TPE at d = 5; 4 seeds per job).

```bash
gh workflow run rebaseline.yml -f suites=all -f seeds=12   # seeds: a count or '42,7'; -f ref=<sha>; -f release=...
```

The last job aggregates the shards into the reference files:
`ref_composite_<mode>_s<seed>.json` (for `benchmark_harness.py compare`),
`ref_ioh_<battery>.json` (multi-seed; compare against a run with the same
`--seeds`) and single-seed `ref_ioh_<battery>_s<seed>.json` (`ioh-external`:
`ref_ioh_standard_external*.json`), `ref_family_screen_<preset>.json` (`family_screen.py from=FILE`),
`ref_MANIFEST.json` (commit, seeds, failed shards, shard commands) and
`SUMMARY.json` (the numbers per suite: composite mean/min/max and per seed,
IOH and family mean AOCC and per spec, plus the release tag and URL).

**The raw `ref_*` files are not committed** (Harald, 2026-09-26; ~14 MB
per run).  They live in a GitHub release that the job creates:

*   **Tag** `rebaseline-<UTC date the run started>` (the earliest shard
    start) on the measured commit, marked *pre-release* and never
    *latest*, titled "Re-baseline <date> (reference data)".  The release
    notes carry the run id as a marker: a release with this run's marker is
    reused under whatever tag it has, so re-running the job never
    duplicates it.  A taken tag (another run's release, or a bare git tag
    such as the burned `rebaseline-2026-09-26`) gets `-run<RUN_ID>`
    appended; if that is taken too, the step fails.  An existing tag must
    point at the measured commit.  `-f release=<tag>` names the release
    explicitly (it must start with `rebaseline-`; a smoke test:
    `-f release=rebaseline-smoke-<date>`), `-f release=none` skips it.
    With `auto`, an incomplete run is not published (the step fails; the
    artifact is still uploaded): a shard that failed, a planned shard that
    left no result (timeout, lost runner, cancelled job — `aggregate
    --plan` compares against the matrix), or any `measure` job not
    `success`.  Publish it under an explicit tag if it is still wanted.  A
    local `publish` without a single run id needs `--tag` and has no
    fallback.  Only one aggregate job publishes at a time (`concurrency:
    rebaseline-release`); GitHub keeps one pending job per group, so a third
    queued run's aggregate cancels the pending one — its shard artifacts
    survive (aggregate and publish it locally, below).
*   **Assets**: `<tag>.tar.gz` (every `ref_*.json`, flat), and
    `ref_MANIFEST.json` and `SUMMARY.json` separately.
*   **Immutable.**  The repository has immutable releases enabled: once
    published, a release's assets cannot change, and **a deleted release's
    tag can never be used again** (this is how `rebaseline-2026-09-26` was
    lost; its data is under `rebaseline-2026-09-26-run36228301268`).  So
    `publish` creates a draft, uploads, then publishes; a re-run of the
    job resumes a draft and leaves a published release alone.  Do not
    delete a re-baseline release; a smoke-test release may be deleted
    (`gh release delete <tag> --cleanup-tag`), its name is then gone.

Commit only the small files: download the release into
`planning/results/<date>/` (`.gitignore` excludes the raw `ref_*` there)
and add `SUMMARY.json` and `ref_MANIFEST.json`.  `fetch` unpacks only
`<tag>.tar.gz` and never overwrites an existing `SUMMARY.json` or
manifest: identical copies are skipped, differing ones kept with a
warning.

```bash
uv run python scripts/rebaseline.py fetch rebaseline-2026-09-26-run36228301268   # -> planning/results/2026-09-26/
uv run python scripts/rebaseline.py fetch <tag> --dir /tmp/ref                  # anywhere else
uv run python benchmark_harness.py compare planning/results/2026-09-26/ref_composite_quick_s42.json after.json
```

Without the workflow (e.g. a job that failed after measuring):
`gh run download <RUN_ID> --pattern 'shard-*' --dir rebaseline-raw`, then
`scripts/rebaseline.py aggregate rebaseline-raw --plan plan.json` (the
matrix, `scripts/rebaseline.py plan --suites ... --seeds ... > plan.json`;
→ `planning/results/<UTC date>/`) and `scripts/rebaseline.py publish planning/results/<date> --target
<measured sha>`.  The job also keeps the aggregated directory as the
artifact `rebaseline-references` for 90 days.
