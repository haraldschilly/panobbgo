# TODO

Open work only; remove an item when it is done.  Results go to
`planning/DISCOVERY_2026-09-09.md` (§n), the state and plan of record to
`planning/GOAL.md` §2/§2c, older history to `planning/done/TODO_archive_*`.
References: cheap track, the 2026-09-27 re-baseline with the active-CMA
default (§61, release `rebaseline-2026-09-27`,
`planning/results/2026-09-27/SUMMARY.json`); §64 changes only CMA-ES runs
with λ > 10 there (≤ 0.002 on the families, other suites unmeasured).
Expensive track: none yet — §62's panobbgo numbers are superseded
(pre-active, pre-§63/§64); the re-run is the q-sweep item in §2.

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
      (first a local timing to size its shards); a reference for the
      baselines on `--families-constrained` (they minimise `f + 100·cv`
      since #358; the `families-constrained` re-baseline suite is the
      family screen, without baselines); check whether RC01u/RC02u (8/9
      equalities, rarely feasible at 500·dim) discriminate at all.
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
- [ ] Then measure panobbgo vs the incumbents per COCO class, budget and
      q (roadmap §5.2; the families part is the q-sweep in §2 below).

## 2. Expensive track and recent follow-ups

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
      7 before the request cap; measured before §63/§64).  Prefer an arm
      with a partly dispatched generation, or pull in generation-sized
      chunks.
- [ ] **q-sweep measurement** (`measure.yml`, `doc/dev/benchmarking.md`
      "Expensive-track measurement").  §62 (run 36274781342) was the first
      run; its panobbgo rows measured a crippled CMA-ES (positive-only,
      and the half quorum of §64), only the externals' numbers stand.
      In order:
      (a) re-run the default grid on master (active CMA, λ ≥ q floor §63,
          dispatched quorum + fold §64) — the confirmatory test of §63/§64
          and the first expensive-track reference;
      (b) with it, decide whether the floor's budget cap stays (it halves
          `RoundRobin_CMAES`'s gain at d2/q64, §63.3) and look at
          Blocks d2/100·d/q16 (−0.012, 1/5, §63): size a block to at least
          one generation of its owner, or apply the floor only where the
          arms cannot fill the workers — untried;
      (c) 12 seeds on the 100·d, q 4–16 band;
      (d) the `failure` preset.
- [ ] **CMA-ES follow-ups from §64.**  (a) At d10/q1 closing at μ + fold
      beat ranking the whole generation (−0.007 [−0.011, −0.003], 0/5):
      a step-size effect to look at.  (b) Fold biases σ down (strongly at
      q 4–16, little at q = 2, not at q = 64); harmless on the free
      families at 100·d, unmeasured on multimodal problems at larger
      budgets.  (c) A fold cap that bounds the late share of the selected
      slots (earliest-arrived or down-weighted, not best-ranked) is
      untried; `"fold_capped"` (best-ranked) measured the same as `"fold"`.
- [ ] **CMA-ES on real async backends** (threaded / process / dask).
      `popsize_min_workers="auto"` raises λ to the worker count on the
      virtual clock only (§63): on real pools λ = q costs sample
      efficiency with cheap objectives, stretches the stagnation window
      and removes BIPOP's small regime; for expensive objectives it should
      help as on the clock.  The §64 quorum and fold rules apply there
      too.  Both unmeasured; decide whether "auto" should look at measured
      evaluation times, or leave it to `True`.
- [ ] **Re-baseline suite for the expensive-track baselines** (BoTorch
      qLogEI, TuRBO-1, SMAC3, Py-BOBYQA; extra `baselines-bo`): the slot
      is marked in `SUITES` in `scripts/rebaseline.py`.  `measure.yml`
      covers the families at small budgets; MA-BBOB at 20·d / 100·d is
      still open (`ioh_benchmark.py run --budget-multiplier`).
- [ ] **Optuna 6 drops `CmaEsSampler(x0=)`** (deprecated since 4.9;
      `harness_baselines.py` silences the FutureWarning).  Before bumping
      to 6: find another way to seed the start point, or accept the box
      centre and say so in the guide.

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
      active default the sharing portfolio trails `CMAES_alone` /
      `RoundRobin_CMAES` at 500·d on every family preset, by 0.02–0.15
      (§61; the #346 review saw it first on `ellipsoid_fhs_crash` d5).
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

- [ ] **OpenBLAS Zen 4 override (upstream #6021).**  The FP pin sets
      `OPENBLAS_L2_SIZE=2048` against it (`doc/dev/benchmarking.md`).  When
      numpy / scipy ship an OpenBLAS whose override skips a forced coretype
      (not the 0.3.35 `NO_AVX512` fix), bump them, re-run `fp-check` and
      drop the variable if it is no longer needed.
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
