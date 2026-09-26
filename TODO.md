# TODO

Open work only; remove an item when it is done.  Results go to
`planning/DISCOVERY_2026-09-09.md` (§n), the state and plan of record to
`planning/GOAL.md` §2/§2c, older history to `planning/done/TODO_archive_*`.
References for every comparison: the FP-exact 2026-09-26 re-baseline
(§56, release `rebaseline-2026-09-26-run36265786623`, summary in
`planning/results/2026-09-26-run36265786623/SUMMARY.json`).

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

- [ ] **Real async loop → pull-when-free.**  The threaded / processes /
      dask loop sizes batches by `jobs_per_client` from wall-clock timings
      and can queue past the free workers; bring it to the virtual clock's
      "async" policy (`StrategyBase.request_cap`).  Until then virtual-clock
      numbers describe that policy, not the real loop.
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
- [ ] **CMA-ES bound handling: resample vs project (§57 H1).**  Optuna's
      CmaEsSampler leads the cheap track only on MA-BBOB (d = 5, inst 2),
      where it finds the global basin 12/12 vs 6/12 for `RoundRobin_CMAES`;
      its one unique trait is resampling out-of-box samples (≤ 10·n
      redraws, then clip).  A/B: a `boundary="resample"` option on `CMAES`
      vs projection, shared `seed_name`, IOH standard 12 seeds, readout the
      (5, 2) reach-1e−1 count and the paired Δ; mirror test: a `cmaes.CMA`
      baseline with `n_max_resampling=1`.  Alongside: the external
      baselines on the BBOB fid battery (§52 axis), to see whether the
      edge exists outside this one mixture.  Then H2 (random first start)
      and H3 (active CMA).
- [ ] Sweep CMA-ES's own knobs (`sigma0`, `popsize`, `restart_mode`).
- [ ] Standing rule: no further bandit tuning without a new mechanism (§31).

## 4. Engineering backlog

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
