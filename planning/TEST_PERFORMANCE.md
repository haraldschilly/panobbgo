# Test-suite performance notes

The `test` CI job was the long pole: ~20–22 min wall-clock (e.g. PR #233 ran
22m10s), versus seconds for `lint` / `format` / `typecheck` / `build` /
`docs`. This note records *why* and what we measured, so future work has data
to start from.

## RESOLVED (2026-06-05): one test was 83% of the suite — a real deadlock

The first `--durations=25` report from CI (run 27007233339) showed:

```
971.44s  tests/test_heuristic_cmaes.py::test_ipop_cmaes_restarts_triggered
 20.44s  tests/test_heuristics_integration.py::test_lbfgsb_integration
 17.44s  tests/test_constraints_realistic.py::test_pressure_vessel_design_dynamic_penalty
 ...      (total: 1175s)
```

**One test was 971 of 1175 seconds.** Root cause (a genuine production bug,
not a test problem): IPOP restarts double the CMA-ES population (λ 6→12→24→48)
but each heuristic's output queue is capped at `heuristic.capacity = 20`.
`_emit_generation()` used `put_nowait` with a silent `except: pass`, so 38 of
48 generation-7 points were **dropped**, while `_pending` still tracked all 48.
The generation could then never collect its `min_results_fraction · λ` quorum,
CMA-ES never emitted again, and the strategy starved at 58/60 results —
spinning through the 10000-iteration no-progress guard at ~100 ms/loop
≈ 971 s before bailing. The test still "passed" because it only asserted
`restart_count >= 1`.

Fixes (all in the framework, not the test):

1. `Heuristic.ensure_output_capacity(n)` — `CMAES._emit_generation` grows the
   queue to fit λ before emitting; clipped emissions now log a warning and are
   excluded from `_pending`.
2. CMA-ES update quorum is based on the *actually emitted* count per
   generation (`_gen_emitted`), so a clipped generation can never deadlock.
3. Restarts flush stale queued points (`clear_output()`).
4. The main loop's no-progress guard is now **time-based**
   (`config.max_stall_seconds`, default 30 s) instead of 10000 loops — a
   starved/deadlocked run aborts in seconds, not minutes.
5. `EventBus.shutdown()` is called from `_cleanup()` — previously only
   `finished`-key subscriber threads were terminated and ~20 daemon dispatcher
   threads leaked per strategy instance (~1600 over a full suite run).

Result: full suite **1175 s → ~186 s locally** (1373 passed); slowest test is
now ~6 s. Regression tests: `test_emit_generation_grows_queue_beyond_cap`,
`test_partial_generation_triggers_update`, `test_stall_guard_aborts_starved_run`,
plus `test_ipop_cmaes_restarts_triggered` now asserts the run completes its
evaluation budget.

## Why the rest is "slow" (breadth, not hotspots)

The remaining cost is **not** CPU and **not** a few pathological tests doing
huge computations. The two `max_eval = 200000` references in the suite
(`test_core.py`, `test_strategies.py`) are **config-validation** tests that
only check the "unreasonably high" guard fires — they never run an
optimization.

The real cost is **breadth**: ~79 tests each call `strategy.start()`, which
spins up the full asynchronous optimizer (event bus + threaded/dask worker
pool) and drives it to a small budget. These are **latency / event-loop
bound**, not CPU bound — a local profile of the slow CMA-ES integration
cluster showed ~19% CPU while wall-clock ticked up (the process spends most of
its time waiting on the threaded event loop, fixed `time.sleep`s, and dask
`LocalCluster` spin-up/teardown). Coverage instrumentation (`--cov`) adds
more.

The "CI output appears stuck after
`test_ipop_cmaes_integration_rastrigin PASSED [37%]`" symptom was exactly the
971 s deadlock above running silently as the *next* test, plus `-v` output
buffering.

## Timing data

`--durations=25` is now always on (pyproject `addopts`), so every run — local
and CI — prints the 25 slowest tests. Read the CI `test` job log to see the
current hot list. The heaviest are the `.start()` integration tests
(CMA-ES IPOP/BIPOP restarts, the constraint integration suites, the
self-improve structural end-to-end tests).

## Parallelism: measured, promising, not yet shipped to CI

`pytest-xdist` is installed. Locally:

```
uv run pytest -n auto --dist loadscope
```

runs the suite across all cores with a large wall-clock win — because the
work is latency-bound, it parallelises near-linearly. `--dist loadscope`
keeps each test *module* on one worker, which preserves module-level fixtures
and avoids cross-worker file collisions for free. (`pytest-benchmark`
auto-disables under xdist; no benchmark tests use the `benchmark` fixture in
the main suite, so that is a non-issue.)

It is **not yet enabled in CI** because two issues surfaced under full
parallel load:

1. **End-of-run "teardown hang" — RESOLVED.** The earlier diagnosis
   ("non-daemon thread blocks worker shutdown") was wrong: all panobbgo
   threads are daemons. The run that "hung at ~99%" was one worker silently
   sitting in the 971 s `test_ipop_cmaes_restarts_triggered` deadlock (slower
   still under 16-worker contention) until the 600 s timeout killed it
   (RC=124). Fixed by the CMA-ES queue-capacity fix above.

2. **Flaky tests under CPU contention.** `tests/test_harness.py::
   test_statistical_compare` failed once in a full 16-worker run but passes
   reliably in isolation (`pytest -n 2 tests/test_harness.py` → 62/62). A
   handful of `@pytest.mark.flaky`-marked, timing-sensitive tests can tip over
   when 16 workers contend for the CPU. Note `@pytest.mark.flaky` is a
   **no-op in CI today**: it is powered by `pytest-retry`, which lives only in
   `[dependency-groups].dev`, not in the `[project.optional-dependencies].dev`
   set that CI installs via `uv sync --extra dev`. Adding `pytest-retry` to
   the `--extra dev` set (and/or a global `--retries`) would make the existing
   markers effective and absorb contention-induced flakiness.

With (1) fixed and the suite at ~3 min serial, the urgency of xdist in CI is
much lower. If desired later: fix (2) (add `pytest-retry` to the `--extra dev`
set so the existing `@pytest.mark.flaky` markers work in CI), then switch the
CI `test` step to `uv run pytest -n auto --dist loadscope --cov=...` for a
further wall-clock reduction.

## Parallel-safety hygiene already done

The two SQLite storage fixtures (`tests/test_storage.py`,
`tests/test_storage_integration.py`) now write to a per-test `tmp_path`
instead of a fixed filename in the CWD, so parallel workers cannot collide on
them.
