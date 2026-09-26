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
- **Regime gate** step 1 landed as the oracle form
  (`regime_gate="oracle:<class>"`, §45; the in-run probe `"table-v1"` is not
  built): masking is cost-free, so the gate is worth exactly what its table is worth.
- **§53**: at 200·dim the better arm flips per function (jSO vs CMA-ES); the
  cell-wise oracle is 0.015–0.039 above the portfolio. Remaining value is in
  **selection**, not in more sharing.

## Roadmap step 1 — instrument (2026-09-26, `planning/DESIGN_roadmap_2026-09-26.md` §3)

- [ ] External baselines, batch-capable: pycma IPOP/BIPOP, Nevergrad NGOpt,
      Optuna (CMA, TPE); then BoTorch/Ax, SMAC3, HEBO, PDFO in an optional extra.
- [ ] Measure on the virtual clock at q ∈ {1, 4, 16, 64} (simulator built
      2026-09-26: `panobbgo/virtual_clock.py`, `ioh_benchmark.py run
      --virtual-workers Q`, `aocc_time`; default policy "async": a
      decision at every completion, candidates only for the free workers).
- [ ] Virtual clock, capped bandits: a generational arm (CMA-ES, DE)
      drains its queued generation at the bandit's share of the free
      workers, so candidates get stale (max age 49 evaluations vs 7 before
      the request cap).  Prefer an arm with a partly dispatched generation,
      or pull in generation-sized chunks.
- [ ] Bring the *real* asynchronous loop (threaded / processes / dask) to
      the same pull-when-free policy as the virtual clock's "async" policy:
      today it sizes batches by `jobs_per_client` from wall-clock timings and
      can queue past the free workers.
- [ ] Failure-region families (half-space / ball / boxes; crash vs timeout).
- [ ] Dims 30/40; real-world set; sealed test set.
- [ ] Feature logging at checkpoints (training data for the selector).
- [ ] Then: measure panobbgo vs the incumbents on both tracks (roadmap §5.2).

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
- [ ] **Hand-over without covariance reset** — the simple form (keep the
      arm's own C, move only m/σ) was measured in §51.1 and lost everywhere;
      only a new form (e.g. scaling the kept C with the move, or blending)
      or a re-test on the re-baselined instrument is worth a run.
- [ ] **Surrogate pre-selection** (lq-CMA-ES) as a new building block.
- [ ] `block_evals="auto"` for the low-budget table row and the portfolio
      spec; fixed-block crossover at 300 or 500·dim (§46).
- [ ] Constrained: `warm_start=None` on the CMA-ES arm only (§44.2 — the
      σ-collapse hypothesis on `ellipsoid_ball`).
- [ ] CMA-ES → warm-started L-BFGS-B polish (never measured: the driver
      lacked a spawn guard).
- [ ] Sweep CMA-ES's own knobs (`sigma0`, `popsize`, `restart_mode`).
- [ ] No further bandit-tuning round without a new mechanism.

## Engineering backlog

The 2026-09-25 audit backlog (T1–T5) is done: PRs #320–#330 — measurement
integrity, optimizer fidelity (jSO / NL-SHADE / CMA-ES / PSO / constraint
ordering), runtime robustness (process pool, leaks, storage fingerprint,
classic test functions), performance (add_results, Splitter, LSHADE ranks,
analyzers on demand, `--jobs`, IOH worker reuse) and cleanup (dead code,
shared bandit selectors and screen pipeline).  A second full audit round
followed (#331–#334).  Details in the PR descriptions and commit messages.

- [ ] **External baselines in the re-baseline workflow.**  The pycma /
      Nevergrad / Optuna baselines (`panobbgo/harness_baselines.py`, extra
      `baselines`) are opt-in by name, so `scripts/rebaseline.py` does not
      measure them.  Add an `--external-baselines` suite option (IOH run
      with `--baselines --strategies <defaults> <EXTERNAL_BASELINE_NAMES>`)
      and `--extra baselines` in `rebaseline.yml`, so the external
      references come from the sharded workflow.
- [ ] Adopt ruff 0.16's wider default rules (the pinned E4/E7/E9/F
      selection is clean since 2026-09-25) — own change.
- [ ] Zoo compaction — parked until the broader suite shows what is good.
- [ ] The shapes and failure family presets (2026-09-26) have no
      reference numbers yet; add them to `scripts/rebaseline.py` when they
      are to be tracked.

## Decisions (2026-09-25)

Settled with Harald and implemented (#337–#339): harness measurements run
synchronously by default; RNG streams are keyed per module; the deadlock
backstop never cuts an outstanding evaluation, and `evaluation.timeout`
(unset by default) is a per-call limit in every backend whose firing records
a NaN result; warm restarts use the archive only outside the stagnated
basin; QuadraticWLS fits on the pull path.  Kept as they are: old storage
databases without a fingerprint are refused (escape hatch
`storage.adopt_legacy`); block-bandit async credit unchanged; DynamicPenalty /
ALM keep their own rho default (10), ALM multipliers grow linearly while the
incumbent is stuck; the multi-seed regression gate stays pooled.
