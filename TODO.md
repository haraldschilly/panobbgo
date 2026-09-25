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

## Engineering backlog

The 2026-09-25 audit backlog (T1–T5) is done: PRs #320–#330 — measurement
integrity, optimizer fidelity (jSO / NL-SHADE / CMA-ES / PSO / constraint
ordering), runtime robustness (process pool, leaks, storage fingerprint,
classic test functions), performance (add_results, Splitter, LSHADE ranks,
analyzers on demand, `--jobs`, IOH worker reuse) and cleanup (dead code,
shared bandit selectors and screen pipeline).  Details in the PR
descriptions and commit messages.

- [ ] **Re-baseline once.**  Result files from before 2026-09-25 are not
      comparable: runs now stop at exactly `max_eval`, composite `success`
      means "tolerance met within the budget", the DE family / CMA-ES /
      PSO follow their papers, and GP fits are seeded.  Re-measure the
      composite quick/standard and IOH/family references before the next
      comparison that relies on them.
- [ ] Adopt ruff 0.16's wider default rules (the pinned E4/E7/E9/F
      selection is clean since 2026-09-25) — own change.
- [ ] Zoo compaction — parked until the broader suite shows what is good.

## Questions for Harald

- [ ] **`sync_eval=True` as the harness default** (`run_ioh_harness`,
      `scripts/ioh_benchmark.py`, composite `HarnessConfig.sync_eval`).
      Every screen already passes it; flipping the default moves the
      recorded baselines (one re-baseline, see above).
- [ ] **Order-independent RNG streams per module.**  `StrategyBase.initialize`
      still draws one seed per legacy default-analyzer slot
      (`_LEGACY_DEFAULT_ANALYZER_SLOTS`) so trajectories stayed bit-identical
      when Grid/Splitter became optional.  Switching to keyed streams
      (e.g. `SeedSequence(seed, spawn_key=(crc32(name), n))`) removes that
      coupling but changes every trajectory — do it at the next deliberate
      break (e.g. together with the re-baseline)?
- [ ] **Old storage databases without a fingerprint** are refused by default
      now (escape hatch: `storage.adopt_legacy` / `SQLiteStorage.adopt`).
      OK as the default?
