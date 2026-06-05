# Test-suite performance notes

The `test` CI job is the long pole: ~20–22 min wall-clock (e.g. PR #233 ran
22m10s), versus seconds for `lint` / `format` / `typecheck` / `build` /
`docs`. This note records *why* and what we measured, so future work has data
to start from.

## Why it is slow (not what you'd guess)

The cost is **not** CPU and **not** a few pathological tests doing huge
computations. The two `max_eval = 200000` references in the suite
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

A symptom of the buffered, serial run: CI output appears to "stick" after
`test_heuristic_cmaes.py::test_ipop_cmaes_integration_rastrigin` and then jump
ahead — that is `-v` output buffering plus the genuinely slow CMA-ES
IPOP/BIPOP `.start()` restart tests, not a true hang.

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

It is **not yet enabled in CI** because two issues surface under full
parallel load and must be fixed first:

1. **End-of-run teardown hang.** With `-n auto`, the run reaches ~99% (all
   tests effectively done) and then one worker hangs in shutdown — almost
   certainly a non-daemon background thread left running by a strategy/dask
   test (xdist worker processes must exit cleanly; a serial run does not
   because the interpreter exits regardless). A 600 s local run was killed by
   timeout at this point. In CI this would trip the 25 min `test` job timeout.
   Fix: find the offending module (bisect modules under xdist) and ensure its
   strategies/clusters are torn down (the `real_strategy` conftest fixture
   already calls `_cleanup()`; some tests construct strategies without it).

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

Once (1) and (2) are addressed, switch the CI `test` step to
`uv run pytest -n auto --dist loadscope --cov=...` for an expected ~3–4×
wall-clock reduction.

## Parallel-safety hygiene already done

The two SQLite storage fixtures (`tests/test_storage.py`,
`tests/test_storage_integration.py`) now write to a per-test `tmp_path`
instead of a fixed filename in the CWD, so parallel workers cannot collide on
them.
