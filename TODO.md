# TODO

Open work only. Measurements and their reasoning live in
`planning/DISCOVERY_2026-09-09.md` (§n below), the goal contract in
`planning/GOAL.md`, the history of this file in `planning/done/TODO_archive_*`.
Remove an item when it is done; record the result in the planning log, not here.

## Where things stand (2026-09-15, §53)

- **Arms**: `NP_init="auto"` for the DE family, CMA-ES σ-divergence restart —
  both 12-seed accepted defaults. Flagship spec is still `RoundRobin_CMAES`.
- **Sharing portfolio** (`Blocks_warm_CMAES_JSO`: `Archive` + `warm_start` +
  `StrategyBlockBandit`) is a **low-budget effect**: largest at 100·dim
  (§46, 12/12 against all arms), accepted on the 24-fid BBOB axis at 100·dim
  (§53), parity at ≥ 500·dim. The value is the **hand-over** (warm start on
  re-acquisition, both-sided and superadditive — a ratchet, §48/§50); it
  carries m and σ only, never the covariance shape (§49, §51).
- **Bandit** allocation contributes nothing (§31); blocking itself is free (§50).
- **Regime gate** step 1 landed (`regime_gate="table-v1"`, §45): masking is
  cost-free, so the gate is worth exactly what its table is worth.
- **§53**: at 200·dim the better arm flips per function (jSO vs CMA-ES); the
  cell-wise oracle is 0.015–0.039 above the portfolio. Remaining value is in
  **selection**, not in more sharing.

## Waiting for Harald

- [ ] **Defaults** — `regime_gate="oracle:clean"` and `block_evals="auto"` for
      `Blocks_warm_CMAES_JSO` (§45.1, §46.4). Decision of 2026-09-13: only
      after the suite is broadened and the question re-run there.
- [ ] **What "broader suite" means** (`planning/DESIGN_suite_2026-09-14.md`):
      d 10/20, more instances, full BBOB instead of MA-BBOB mixtures,
      constrained/noisy as own axes; tiered (small screen, wide decision) so
      12-seed decision runs stay affordable. The fid axis (§52) is the first step.
- [ ] **Composite registry** — `CMAES_Portfolio`, `IPOP_CMAES`, `BIPOP_CMAES`
      all pair CMA-ES with the `Restart` analyzer (measured −0.067); the
      composite score is a frozen contract.

## Research line

- [ ] **Probe / regime detector** — now the main line (was parked in §45.2):
      target +0.015…+0.039 (§53), signal observable early — how fast the
      first arm progresses (§52.4).
- [ ] **Hand-over without covariance reset** — keep the arm's adapted C and
      only move m/σ (`_reset_covariance` sets C = I today).
- [ ] **Surrogate pre-selection** (lq-CMA-ES) as a new building block.
- [ ] `block_evals="auto"` for the low-budget table row and the portfolio
      spec; fixed-block crossover at 300 or 500·dim (§46).
- [ ] Constrained: `warm_start=None` on the CMA-ES arm only (§44.2 — the
      σ-collapse hypothesis on `ellipsoid_ball`).
- [ ] CMA-ES → warm-started L-BFGS-B polish (never measured: the driver
      lacked a spawn guard).
- [ ] Sweep CMA-ES's own knobs (`sigma0`, `popsize`, `restart_mode`).
- [ ] No further bandit-tuning round without a new mechanism.

## Engineering backlog — triaged 2026-09-25

From a five-way read-only code audit (core, heuristics/strategies,
analyzers/lib, harnesses, self-improve loop). "✓" = reproduced or traced end
to end; "?" = plausible from reading. Tiers are ordered by what to fix first.
T2 items change optimizer trajectories; they restore documented or
published behaviour and land without a benchmark. T1/T3/T4 change only
instruments, which must be re-baselined once.

### T1 — measurement integrity

- [ ] **Re-baseline once: pre-T1 result files are not comparable.**  Since
      2026-09-25 (T1) composite `success`/`success_rate` means "tolerance
      met *within the budget*" (was: the final `strategy.best`, which could
      come from evaluations past it), and the core clamps every run to
      exactly `max_eval` evaluations (runs used to overshoot by up to one
      batch, e.g. 406 on 400) — which also changes trajectories of
      asynchronous runs and the last batch of every run.  Composite
      scores, success rates and AOCC from before T1 must not be compared
      with newer ones; re-measure the baselines (composite quick/standard,
      the IOH/family references) before the next A/B that relies on them.
- [ ] **`sync_eval=True` as the default** of `run_ioh_harness` /
      `scripts/ioh_benchmark.py` and of the composite harness
      (`HarnessConfig.sync_eval`, `--sync-eval`).  Every screen already
      passes it; the defaults stay asynchronous because flipping them moves
      the recorded IOH and composite baselines — a decision for Harald,
      then one re-baseline.

### T2 — algorithm bugs (change trajectories)

- [ ] ✓ **jSO `F_w` is a constant** (0.7/0.8/1.2), paper has `F_w = 0.7·F`
      (`heuristics/jso.py:376`); inherited by NLSHADE_RSP/LBC.
- [ ] ✓ **CMA-ES `_counteval += actual_mu`** (`cma_es.py:1636`) counts μ,
      not λ → h_σ `gen_count`, eigen-update gap and BIPOP budgets off ~2×.
- [ ] ? CMA-ES recombines the projected x but updates paths/C from the
      unprojected y (`cma_es.py:1597-1600`) → σ inflation at bounds.
      Repair y from x_proj (Hansen repair-by-injection, `_clip_injected`).
- [ ] ✓ Region hand-off to arms that ignore it: `MetaAnalyst._accepts_region`
      (`meta.py:668`) accepts `LBFGSB(warm_start=True)` which the bandit
      then drops; PSO `_warm_start_swarm` omits `box=` (`pso.py:752`).
- [ ] ✓ `Splitter.best_box` goes stale between splits (`splitter.py:327-394`)
      — Random/NelderMead/QuadraticWLS consume the wrong box.
- [ ] ? `Restart` never resets its global best (`analyzers/restart.py:118-142`)
      → all `max_restarts` fire early, then the analyzer is inert.
- [ ] ✓ LBFGSB respawns with the same `_worker_seed` (`lbfgsb.py:284,309`)
      → restarts replay the same multi-start sequence.
- [ ] ✓ LSHADE `_fx_of` memoises penalties (`lshade.py:664`) but the
      dynamic-penalty and AL handlers are time-varying.
- [ ] ✓ Constraint handlers disagree: `DefaultConstraintHandler.is_better`
      is cv-first, its `get_penalty_value` is fx+100·cv (`lib/constraints.py`)
      → Archive/Restart/Splitter rank differently from `Best`.
- [ ] ? Bridge races: LBFGSB/COBYQA `on_restart` rebinds the pipe on the
      bus thread (`lbfgsb.py:478`, `cobyqa.py:345`) — defer to `produce` as
      LocalPenaltySearch does; `_bridge_finished` sets `_stopped`, so a
      converged COBYQA ignores every later restart; LocalPenaltySearch abort
      can leave a stale `eval` in the pipe (`local_penalty_search.py:220`).
- [ ] ✓ PSO clips position but not velocity (`pso.py:447`) → wall sticking.

### T3 — runtime bugs and wrong defaults (no effect on the default path)

- [ ] ✓ `evaluation_method="processes"` broken: `sys.path` puts `panobbgo/`
      first so `panobbgo/logging` shadows stdlib (`core.py:2145-2167`);
      failed tasks never leave `pending` → 60 001 idle spins (`:2197-2207`).
      Replace with `ProcessPoolExecutor` or remove the mode.
- [ ] ✓ `HeuristicSubprocess` leaks its process (no `__stop__`, `core.py:930`).
- [ ] ✓ Unstarted strategies leak their EventBus thread (the threaded
      pool's queued evals are cancelled on cleanup since T1).
- [ ] ✓ `Config()` crashes outside a git checkout (`utils.py:153`,
      IndexError) and reports the cwd's repo, not panobbgo's.
- [ ] ✓ Unknown strategy kwargs silently dropped (`core.py:1535`);
      `max_eval > 100000` hard-fails; `cpu_count()` fallback dead.
- [ ] ✓ `TypeError` inside `terminate=True` handlers swallowed and the
      handler unsubscribed (`core.py:1399-1403`).
- [ ] ? `on_finished` delivery races with `__stop__` (`core.py:2471,2501`).
- [ ] ? sqlite storage resumes foreign results — no problem fingerprint.
- [ ] ✓ `Problem(dx=…)` shifts twice and mutates the caller's array
      (`lib/lib.py:259,429`); `Result.__eq__` (fx) vs `__hash__` (x, fx, who)
      (`lib/lib.py:218`).
- [ ] ✓ `lib/wrappers.NoisyProblem` shares one RNG across threads (not
      reproducible) and collides by name with the deterministic
      `lib/noise.NoisyProblem`; `panobbgo.lib` exports the wrong one.
- [ ] ✓ `ioh_runner.run_strategy_on_ioh_problem` sets the budget after
      construction — and has no caller: delete it; rebuild
      `scripts/ioh_smoke.py` on `harness_ioh._run_one`.

### T3b — classic test functions (CI micro-battery; not the AOCC metric)

- [ ] ✓ Wrong formulas / optima in `lib/classic.py`: **Wood** (unbounded,
      −2728), **Branin** (default `t=1` kills the cosine term), **Box**
      (sign, `m=1`), **Step** (missing floor), **Trigonometric**, **Powell**
      (last term ², not ⁴), **RosenbrockModified** (claims 0 at (−1,−1), is
      78; true min ≈ 34.04), **Sargan** (missing D). `RosenbrockStochastic`
      and `NesterovQuadratic` use global `np.random` (Nesterov ignores `dim`).
      Add `x_opt`/`f_opt` class attributes + one parametrized test
      `f(x_opt) == f_opt` with a DE sanity check.

### T4 — performance / memory

- [ ] ✓ `Results.add_results` is O(n) per batch, O(n²) per run
      (`core.py:238-270`): sorts all fx for a progress threshold even when
      the reporter is off (782 µs/add at n = 8k); also `prev_best` off by one.
- [ ] ✓ Splitter ~2.5× faster (0.68 → 0.27 s / 2500 evals): scalar descent,
      key by `id(result)`, lazy debug logging; store results in leaves only
      and drop `result2boxes` / biggest-box bookkeeping (no consumers).
- [ ] ✓ `Grid` is a default analyzer that nothing reads; drop it from
      `initialize` (`core.py:1642`); `Splitter` only when a consumer needs it.
- [ ] ✓ `Config.__init__` per strategy (~3–6 ms; YAML 3.4, `git rev-parse`
      1.2): `lru_cache` the parsed sources keyed on mtime, cache `info()`,
      build the ArgumentParser only with `parse_args=True`, `makedirs`.
- [ ] ✓ LSHADE family sorts the population per trial (`lshade.py:771`,
      `nl_shade_rsp.py:260`): ~40 % of wall on cheap objectives — cache ranks.
- [ ] ✓ `avg_time_per_task` averages all walltimes every 1 ms loop; NaN
      below 2 tasks; Dask records a constant 0.1 (`core.py:2054,2542`).
- [ ] Screens run seeds serially; under `sync_eval` runs are deterministic
      → a `--jobs` process pool gives N× wall-clock for free.
- [ ] IOH worker respawned via `uv run` per run (~0.2 s; 20 % of a quick run).

### T5 — simplification / dead code

- [ ] Shared `benchmarks/_screen.py` (seed loop, cell fold, paired stats,
      `t_ci`, delta table, run health): ~300–400 lines across six screens.
- [ ] Dead: `harness.compare(statistical=True)` (~80), legacy
      `benchmark.BenchmarkSuite` + `run_benchmark.py` (~550, test-only),
      `blocks._maybe_end_prologue` (condition never true), old
      `DifferentialEvolution` (swallows all exceptions, O(NP²)), logging
      `ComponentLogger`/`ErrorReporter`/verbosity toggles, `MockupEventBus`,
      test helpers in library `utils.py`, `config.max_stall_seconds`,
      `Grid`/`Dedensifyer` (test-only).
- [ ] DE family dedup (~180 lines): `JSO._generate_trial` copies LSHADE's;
      three `_update_memory`; reset blocks; EpSin samplers.
- [ ] `phased.py:340-515` copies the UCB/Thompson/LinUCB/Rewarding selectors
      (~150); bridge terminate/join/kill ×4 into `PipeBridgeHeuristic` (~60);
      CMA-ES adaptation constants twice (~25); `Module.budget_progress()` (~30).
- [ ] 71 hand-rolled strategy doubles in tests lack `spawn_rng`
      (`_module_rng` fallback).
- [ ] Document the multiprocessing spawn guard for users (`LBFGSB` /
      `COBYQA` / `LocalPenaltySearch` / `QuadraticWlsModel`).
- [ ] `ruff check`: 221 findings on E4/E7/E9/F (133 auto-fixable); ruff
      0.16's wider defaults ~2000 more — own change.
- [ ] Zoo compaction — parked until the broader suite shows what is good.
