# TODO

Open work only; remove an item when it is done.  Results go to
`planning/DISCOVERY_2026-09-09.md` (§n), the state and plan of record to
`planning/GOAL.md` §2/§2c, older history to `planning/done/TODO_archive_*`.
References for every comparison: the 2026-09-27 re-baseline with the
active-CMA default (§61, release `rebaseline-2026-09-27`, summary in
`planning/results/2026-09-27/SUMMARY.json`).

## 1. Roadmap step 1: finish the instrument

`planning/DESIGN_roadmap_2026-09-26.md` §3.  Done: external baselines
(#344), virtual clock (#345), failure-region families (#346), dims 30/40
with a `bbob-largescale`-style slice and the sealed test set
(`doc/dev/benchmarking.md`), expensive-track baselines (`baselines-bo`
extra).

- [ ] Real-world set, the rest: CEC2020 real-world constrained is in
      (19 problems, `run --realworld`, `panobbgo/lib/realworld.py`).  Open:
      COCO `bbob-constrained` / `bbob-mixint`; ESA GTOP (in `pykep`, no
      Python 3.14 wheels yet); HPO surrogates (YAHPO Gym pins `numpy < 2`,
      HPOBench is not on PyPI) — recheck when the wheels exist.
- [ ] Real-world follow-ups: a `realworld` suite in `scripts/rebaseline.py`
      (first a local timing to size its shards), and a re-baseline of
      `families-constrained` for the baselines (they now minimise
      `f + 100·cv` on constrained problems — that also changes the
      baselines on the constrained half of `--families-sealed`, whose
      numbers before #358 are not comparable for baselines); RC01u/RC02u (8/9 equalities)
      are rarely made feasible at 500·dim, so check whether they
      discriminate at all.
- [ ] Nearby quadratic step scaling at high d (cap auto-rank candidates and
      history): `Nearby(quadratic=True)` takes 7–77 s and up to 1 GB per
      fit at d = 160, on every new best.  Until then `--legacy` is refused
      with the large and sealed batteries.
- [ ] Sealed test set, second half: add part of the real-world set
      (`panobbgo/sealed.py`; MA-BBOB and families are sealed).  The CEC 2020
      problems are fixed, so "sealed" means held-out problems, e.g. a few
      CEC 2020 problems kept out of the development battery.
- [ ] Feature logging follow-ups (#361 records the features): noise
      estimate from repeats and evaluation durations are not logged yet;
      the counterfactual branch labels (roadmap §4 A) are the next step.
- [ ] Py-BOBYQA `seek_global_minimum=True` as a second, global variant of
      the local BOBYQA reference (`panobbgo/harness_baselines_bo.py`).
- [ ] Then measure panobbgo vs the incumbents on both tracks, per COCO
      class, budget and q (roadmap §5.2).

## 2. Follow-ups from the 2026-09-26 PRs

- [ ] **FP pin for the torch path (BO baselines).**  `PIN_ENV` caps torch /
      MKL / oneDNN at AVX2 (`ATEN_CPU_CAPABILITY`, `MKL_CBWR`, ...), but
      no fp-check covers a BO cell yet: add one (a `baselines-bo` job in
      `fp-check.yml`) before claiming bit-identity for BoTorch / TuRBO.

- [ ] **Pull-when-free follow-ups** (the real async loop pulls when free,
      `evaluation.async_policy: pull`, §55).  (a) What is left of candidate
      staleness in pull mode is the heuristics' own output queues: Random
      prefills `heuristic.capacity` = 20 points, ~11 results per candidate
      against 2-3 for Nearby and 2-5 for CMA-ES; harmless for Random, but a
      smaller fill level for reactive heuristics under pull would cut it.
      (b) The dask worker count is `len(scheduler_info()["workers"])`, so a
      cluster with `threads_per_worker > 1` is underfilled; count threads
      if that setting is ever used.  (c) Drop `async_policy: legacy` once
      nothing needs it.
- [ ] **Stale generations under a capped bandit.**  On the virtual clock a
      generational arm (CMA-ES, DE) drains its queued generation at the
      bandit's share of the free workers (max candidate age 49 evaluations,
      7 before the request cap).  Prefer an arm with a partly dispatched
      generation, or pull in generation-sized chunks.
- [ ] **q-sweep measurement**: the instrument is `measure.yml`
      (`scripts/measure.py`; families at 20·d / 100·d, q ∈ {1, 4, 16, 64},
      panobbgo, the cheap-track and the GP baselines,
      `doc/dev/benchmarking.md` "Expensive-track measurement").  Run it and
      log it as a DISCOVERY section; then the `failure` preset.
      **First run done (§62, run 36274781342, pre-active CMA-ES):** Holm
      1 win / 7 losses / 13 open of 21 cells; parity-or-better only at
      100·d, q 4–16; qLogEI leads at 20·d; the q = 64 loss is on the time
      axis only.  Next, in §62.8 order:
      (a) re-run the grid on the active-CMA default, with SMAC's d = 10
          estimate raised;
      (b) worker utilisation of Blocks at q = 64 (idle workers?) and a
          dispatch fix;
      (c) 12 seeds on the 100·d, q 4–16 band;
      (d) the `failure` preset.
- [ ] **Re-baseline suite for the expensive-track baselines** (BoTorch
      qLogEI, TuRBO-1, SMAC3, Py-BOBYQA; extra `baselines-bo`): the slot
      is marked in `SUITES` in `scripts/rebaseline.py`.  `measure.yml`
      covers the families at small budgets; MA-BBOB at 20·d / 100·d is
      still open (`ioh_benchmark.py run --budget-multiplier`).
- [ ] **Optuna 6 drops `CmaEsSampler(x0=)`** (deprecated since 4.9;
      `harness_baselines.py` silences the FutureWarning).  Before bumping
      to 6: find another way to seed the start point, or accept the box
      centre and say so in the guide.
- [ ] **`Blocks_warm_CMAES_JSO` is weaker than plain CMA-ES on
      `ellipsoid_fhs_crash` d5** (seen in the #346 review, not yet a
      DISCOVERY entry).  Measure paired on the failure preset; if it holds,
      find the mechanism (jSO's share of the budget in the crash half-space?).

## 3. Research line (cheap track)

The main line is roadmap §4 (D failure model, A+B learned probe→select
with forecast allocation, C new sharing payloads) after step 2 above.
Cheap-track items, in GOAL §2c order:

- [ ] **Probe / regime detector** (becomes roadmap A): target the per-cell
      oracle gap +0.015…+0.039 (§53); signal observable early, e.g. the
      first arm's progress rate (§52.4).  The oracle gate (§45) is the seam;
      the in-run probe `"table-v1"` is not built.
- [ ] **Hand-over without covariance reset**: the simple form (keep C, move
      m/σ) lost everywhere (§51.1); only a new form (scale the kept C with
      the move, blend) or a re-test on the re-baselined code is worth a run.
- [ ] **Surrogate pre-selection** (lq-CMA-ES) as a building block.
- [ ] `block_evals="auto"` for the low-budget table row and the portfolio
      spec; fixed-block crossover at 300 or 500·dim (§46).
- [ ] Constrained: `warm_start=None` on the CMA-ES arm only (§44.2,
      σ-collapse hypothesis on `ellipsoid_ball`).
- [ ] CMA-ES → warm-started L-BFGS-B polish (never measured).
- [ ] **Recheck the block scheduler's `REGIME_TABLE_V1`**
      (`strategies/blocks.py`): measured with positive-only CMA-ES; with the
      active default the CMA-ES arm is stronger and the sharing portfolio
      now trails `CMAES_alone` / `RoundRobin_CMAES` (§61).
- [ ] **BoundTransform-style genotype mapping** as the principled
      alternative to the active guard (§59): run CMA-ES in an unbounded
      genotype space with a smooth fold into the box for evaluation (pycma's
      default), so no step is ever repaired and active CMA needs no guard.
      §59's probe says what to beat: resample + active is +0.16…+0.21 over
      the guard with the optimum *near* a face, −0.29…−0.41 with it *on*
      the face (f5).
- [ ] Active guard, finer rule (§59): skip only when a repaired offspring
      is among the μ selected; +0.06…+0.08 on two probe cells, within noise
      elsewhere.  Measure on the battery before adopting.
- [ ] **`Baseline_Optuna_CmaEs` crashes on BBOB f14, d = 2, instance 0 at
      500·d** in 3 of 12 seeds (7, 2025, 11): `ValueError: nan is invalid
      value` from Optuna, 0 evaluations, scored 0 (§58; conclusions do not
      change without the cell).  Reproduce and guard the adapter.
- [ ] Why pycma BIPOP (active CMA by default) gains only +0.056 on the
      high-conditioning BBOB class where our `active=True` gains +0.152 (§58).
- [ ] Sweep CMA-ES's own knobs (`sigma0`, `popsize`, `restart_mode`).
- [ ] Standing rule: no further bandit tuning without a new mechanism (§31).

## 4. Engineering backlog

- [ ] **OpenBLAS Zen 4 override (#6021) follow-ups.**  The FP pin now sets
      `OPENBLAS_L2_SIZE=2048` (`panobbgo/fp_env.py`, `doc/dev/benchmarking.md`):
      without it OpenBLAS 0.3.34 segfaults in `dgemm_kernel_HASWELL` on
      AVX-512 EPYC 9V74 runners and gives other GEMM bits there.  Open:
      (a) two shards of the FP-exact re-baseline run 36265786623
      (`composite-quick 01`, `ioh-external 01`) ran on such hosts without
      it — re-run those two shards and compare bit for bit (GEMMs with
      K <= 320 are unaffected, so they are probably identical);
      (b) when numpy / scipy ship an OpenBLAS whose override skips a forced
      coretype (upstream #6021, not the 0.3.35 `NO_AVX512` fix), bump them,
      re-run `fp-check` and drop the variable if it is no longer needed.
- [ ] Adopt ruff 0.16's wider default rules (the pinned E4/E7/E9/F
      selection is clean) — own change.
- [ ] Zoo compaction — parked until the broader suite shows what is good.

## 5. Waiting for Harald

- [ ] **Defaults** `regime_gate="oracle:clean"` and `block_evals="auto"` for
      `Blocks_warm_CMAES_JSO` (§45.1, §46.4): only after the suite is
      broadened and the question re-run there (decision of 2026-09-13).
- [ ] **What "broader suite" means** (`planning/DESIGN_suite_2026-09-14.md`,
      extended by roadmap §3.3): d 10/20, more instances, full BBOB instead of
      MA-BBOB mixtures, constrained/noisy as own axes; tiered (small screen,
      wide decision).  The fid axis (§52) was the first step.
- [ ] **Composite registry**: `CMAES_Portfolio`, `IPOP_CMAES`, `BIPOP_CMAES`
      all pair CMA-ES with the `Restart` analyzer (measured −0.067); the
      composite score is a frozen contract.
- [ ] **Other raw result data in git**: re-baseline `ref_*` files now go to
      GitHub releases (decision 2026-09-26, `doc/dev/benchmarking.md`).
      Still committed: `planning/results/2026-09-1{0,1,3,4}/` (~7.5 MB of
      screen JSON and logs, cited by DISCOVERY §§) and the loop ledgers
      `planning/done/self_improve_ledger_*.jsonl` (~2.4 MB).  Same rule going
      forward (one release per dated results directory), or leave the
      historical ones where they are?
