# TODO

Open work only; remove an item when it is done.  Results go to
`planning/DISCOVERY_2026-09-09.md` (§n), the state and plan of record to
`planning/GOAL.md` §2/§2c, older history to `planning/done/TODO_archive_*`.
References: cheap track, the 2026-09-27 re-baseline with the active-CMA
default (§61, release `rebaseline-2026-09-27`,
`planning/results/2026-09-27/SUMMARY.json`); §64 changes only CMA-ES runs
with λ > 10 there (≤ 0.002 on the families, other suites unmeasured).
Expensive track (`measure.yml`): §73 for the core specs and the `trq`
candidates at 100·d, q 1/4/16/64 (runs 36418128133 / 36418132989, seeds
3001–3012, release `measure-2026-09-29-confirm-3001`; its headline units
are the spec before the gate default); §67 for q 4/16/64 on seeds
1001–1012 (run 36315576900); §65 for the core specs at 20·d (run
36313485264); §62 for the GP baselines elsewhere (run 36274781342).

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
      `features.fscale_features` (§70) is not in the checkpoint records yet.
- [ ] **Selector, roadmap §4 A step 2 (§70).**  Step 1 is in:
      `panobbgo/selector_data.py` (menu v0, shared 10·d LHS probe,
      continuation via `preload_results`, labels), `benchmarks/selector_labels.py`,
      540 labelled tasks (wide, d 2/5, 100·d, q 1/4).  Headroom: task oracle
      0.078 over the best single arm (RR_TRQ), but leave-one-out the instance
      pick (other seeds) keeps 0.015 and the family pick (other instances)
      −0.003; context alone gives 0.  The learnable headroom on this menu
      and set looks small.  Next: more seeds / instances per family, d = 10
      with a ≥ 14·d probe, 20·d, arms that differ where RR_TRQ loses; then
      xgboost with leave-instance-out / leave-family-out CV judged against the
      LOO gaps; a COBYQA warm start that keeps its point.
- [ ] Py-BOBYQA `seek_global_minimum=True` as a second, global variant of
      the local BOBYQA reference (`panobbgo/harness_baselines_bo.py`).
- [ ] Then measure panobbgo vs the incumbents per COCO class, budget and
      q (roadmap §5.2; the families part is the q-sweep in §2 below).

## 2. Expensive track and recent follow-ups

- [ ] **Failure model, roadmap §4 D step 2 (§71).**  Step 1 is in (opt-in):
      `FailureModel` analyzer (`p_fail` / `in_poison`, kernel classifier),
      the proposal filter (`filter=True`: rejected candidates are answered as
      failures, no budget), `CMAES(failure_aware=True)`,
      `TrustRegionQuadratic(failure_aware=True)`, `IOHRunRecord.n_failed`,
      `benchmarks/failure_screen.py`.  In sample: TRQ wastes 47 % of its
      budget to failures (stuck start in a zone, re-proposed failed geometry)
      and gains +0.037 with model + handling (+0.028 from its own handling);
      population arms waste 3–7 % and gain nothing measurable; the filter
      *alone* costs TRQ −0.28 on the boundary-optimum family (d2 q1).  Next:
      fresh seeds, d = 10, q ≥ 16; then decide TRQ `failure_aware` as its
      default (a defect fix, identical without failures); the filter stays
      off by default until an arm-aware filter or an axis-aligned tree (v1)
      avoids the boundary cost; a separate crash / timeout model.  The
      fresh-seed run is prepared (the `trq` group's `RoundRobin_TRQ_fa` =
      §71.4's `+fm`, `RoundRobin_TRQ_aware` = its `+aware`; plan and
      decision rule, signed off 2026-09-28, on a fresh battery: §2 q-sweep
      (d) below).  No failure-aware headline
      variant: §71.4 measured nothing for one (`CMAES+aware` −0.000 ±
      0.004, `Blocks_warm_CMAES_JSO+fm` +0.002 ± 0.002).

- [ ] **Model-based arms, fresh seeds (§66).**  In sample, the opt-in
      `RoundRobin_TRQ` (d ≤ 5) and `COBYQA` alone (q = 1) lead the pool at
      q ∈ {1, 4}, but much of the margin is the exactly quadratic ellipsoid
      family; without it the 20·d cells stay behind and at d = 10 the
      third arm (`Blocks_warm_CMAES_JSO_TRQ`) costs −0.013…−0.019 against
      Blocks.  The free preset's one exact-quadratic family in five can
      dominate its means: report the ex-ellipsoid view next to the mean.
      **Fresh seeds, free preset: §73** (seeds 3001–3012, 100·d, q
      1/4/16/64, descriptive — not pre-declared for a decision).  With all
      families `RoundRobin_TRQ`, `_r05` and `Blocks_warm_CMAES_JSO_TRQ`
      lead the headline spec 11–12/12 in every cell but d10/q64 and the
      pool's best in all but one (r05 − pool best +0.029 … +0.301).
      Without the ellipsoid they still lead at d ≤ 5 in most cells, but
      at d = 10 all three trail the headline spec in all four q cells
      (−0.025 … −0.005; the d = 10 lead is the ellipsoid, where everything
      else scores 0).  r05 − `RoundRobin_TRQ`: ahead in 8 of 12 cells,
      never behind.
      **Proposal for Harald (not added, not dispatched):** a pre-declared
      confirmation of a TRQ-including headline candidate (e.g.
      `Blocks_warm_CMAES_JSO_TRQ`, or r05 as the third arm) against
      `Blocks_warm_CMAES_JSO` and the pool on the **`wide` preset** (15
      families, §68), fresh optimiser seeds and a fresh battery seed,
      d 2/5/10 (d = 10 is where the free preset's lead is only the
      ellipsoid), a Holm family and a decision rule fixed before the
      dispatch; which candidate, the grid and the rule are Harald's call.
      Also open: MA-BBOB at 20·d / 100·d, and the f-scale quadratic R²
      plus the TR ratio-test rate as logged features (roadmap A's probe).
      The `trq` group's cost row is still the laptop estimate;
      recalibrate from §73's runner runs (`s/run` in their summaries).
- [ ] **Wide family preset (§68).**  `measure.py --presets wide` (opt-in,
      15 families, every x_opt away from the box centre) is in; only a
      local smoke of core + trq at d 2/5, 100·d, q 1/4 exists.  Next: a
      runner run on the reduced grid of §68.3 (q ∈ {1, 4}: core, trq,
      TuRBO1, SMAC in one dispatch, qLogEI at 20·d in a second; ≈ 109
      estimated runner-hours), fresh seeds, then use it as the selector's
      development set (roadmap §4 A).  Open questions from §68.5:
      `schwefel_sep` scores 0 for every arm at d ≥ 5 (rescale, or keep as
      the "nothing works" class); the sealed set does not contain the
      new classes yet; ruggedness (Weierstrass/Katsuura), noise and
      discrete variables are still missing.
- [ ] **TRQ on the wide preset (§69).**  Diagnosed: `levy_embed` is the
      start radius, most likely (COBYQA's first steps move
      min(0.05·max width, 0.5) of each axis — 0.5 on the ±5 family boxes —
      TRQ's 0.1; not the neutral directions), `attractive_sector` a C¹ kink the
      quadratic cannot fit (dropping the curvature prior helps it and costs
      the smooth families; left as is), `styblinski_tang_sep` d2 a restart
      loop into the arm's own tabu ball — fixed (a step that improves into
      a tabu ball makes its centre tabu).  Open: `radius_init = 0.5` as a
      candidate for the fresh-seed / fresh-battery run (in sample +0.05 on
      wide, mixed on free); the catch costs `step_ellipsoid` at d2/q1
      (−0.14 in sample, −0.13 on a fresh battery), `sharp_ridge` (−0.08 on
      the 3 in-sample instances) and `levy_embed` d5/q1 (−0.05); the
      converse check (COBYQA started at 0.1) is not run.  (The `cobyqa.py`
      docstrings on the start radius were corrected in #386.)
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
      "Expensive-track measurement").  The re-run on master is §65
      (Holm 3 / 7 / 11; proposals in §65.6); (c), 12 fresh seeds, is done:
      §67 (Holm 4 / 2 / 3 over its 9 cells).  In order:
      (b) decide whether the floor's budget cap stays (§65.6 and §67.6
          propose to keep it: d5/q64 is parity on fresh seeds; the d2/q64
          loss is likely the block rule and the 3 rounds rather than the
          cap — inferred, not measured) and whether to close Blocks
          d2/100·d/q16 without tuning (§67: gap to qLogEI −0.011, 3/12,
          n.s., mostly ellipsoid; ex-ellipsoid −0.004 is post-hoc and
          descriptive; **§73 on seeds 3001–3012: a Holm loss, −0.014
          [−0.020, −0.008] 1/12, ellipsoid −0.042, ex-ellipsoid −0.007
          (descriptive)**; the §63.4 block-sizing ideas stay untried);
      (d) the `failure` preset (§67.6: fresh seed list, pre-declared Holm
          family; a local in-sample screen is §71, the q-sweep on runners
          with the `trq` group is still open).  **Prepared, not
          dispatched** (#393): §71's candidates `RoundRobin_TRQ_fa`
          (`failure_aware` + the `FailureModel` filter, §71.4's `+fm`,
          +0.037 in sample) and `RoundRobin_TRQ_aware` (`failure_aware`
          alone, `+aware`, +0.028) in the `trq` group, and the summary's
          *Variants − their base spec* table (every spec against the spec
          whose `seed_name` it shares: `_fa` / `_aware` / `_r05` −
          `RoundRobin_TRQ`).  Fresh optimiser seeds **4001–4012** (unused:
          §67 took 1001–1012, §72.4 2001–2005, the §72 confirmation
          3001–3012).  **Fresh instances too** (Harald, 2026-09-28: a
          blind test): the default battery is the 12 instances per d §71
          was debugged on, so the run uses a fresh battery, battery seed
          **20260928** (`measure.py plan --battery-seed`, `-f
          battery_seed`; unused before: the default is 20260910, §69.4's
          wide battery 20260927; `tests/test_measure.py` checks that its
          optima, values and failure regions differ from the default
          battery's on every instance).  The preset still has no d = 10.
          Grid: failure preset (d 2/5), 100·d, q 1/4/16/64 (q = 1 is where
          §71 measured TRQ's gain), all groups and `trq`:
          `gh workflow run measure.yml -f seeds=4001,4002,4003,4004,4005,4006,4007,4008,4009,4010,4011,4012 -f presets=failure -f battery_seed=20260928 -f budgets=100 -f qs=1,4,16,64 -f groups=core,qLogEI,TuRBO1,SMAC,trq`
          (`plan` / `cost` on the battery-seed PR: 155 shards, ~165.7
          estimated runner-hours, 152.8 of them qLogEI; core 4 shards,
          1.8 h; TuRBO1 5, 6.6 h; SMAC 2, 2.0 h; trq 6 shards, 2.5 h —
          the battery seed does not change the cost model; the estimate
          ran 1.56× high on §67.  The failure preset's GP / core costs are
          the free preset's estimates, `doc/dev/benchmarking.md`.)
          Supplement for the free-preset identity, same seeds and the same
          fresh battery (the identity holds on any battery; this keeps the
          whole run on unseen instances), no GP group needed:
          `gh workflow run measure.yml -f seeds=4001,4002,4003,4004,4005,4006,4007,4008,4009,4010,4011,4012 -f presets=free -f battery_seed=20260928 -f budgets=100 -f qs=1,4,16,64 -f groups=trq`
          (13 shards, ~6.5 h).  Aggregate the main run on its own (the
          Holm family is every headline cell of one aggregate: the 8
          failure cells); the supplement (trq only: no pool, no headline)
          may be aggregated with it or apart.  Their cells are
          `failure-20260928/…` and `free-20260928/…`.  (Dispatched
          2026-09-28 at c43c199 as runs 36535041375 / 36535045456, before
          §73: their `trq` shards still run `Blocks_warm_CMAES_JSO_dimbudget`
          (`measure.RETIRED_SPECS` keeps it in the group), and their
          headline units are the ungated spec — identical to the gated one
          at d ≤ 5, so the failure cells are unaffected.)
          **Decision rule (signed off by Harald 2026-09-28 as written,
          before the dispatch):**
          *Holm family:* the 8 failure cells (d 2/5 × q 1/4/16/64, 100·d),
          `Blocks_warm_CMAES_JSO` vs the pool's best on the headline
          metric (AOCC at q = 1, `aocc_time` at q > 1), Holm at 0.05 —
          the roadmap claim on failure regions; no default rides on it.
          The ex-ellipsoid view (here without `ellipsoid_fhs_crash`) is
          pre-declared as descriptive.
          *TRQ `failure_aware=True` as the arm's default* (its handling
          only; judged on `RoundRobin_TRQ_aware`; the flip also changes the
          opt-in `Blocks_warm_CMAES_JSO_TRQ` and `RoundRobin_TRQ_r05`, which
          are read descriptively and not judged here) if all of: (1) in the
          two q = 1 cells (d 2, d 5; §71.4: +0.050, +0.057)
          `_aware − RoundRobin_TRQ` on AOCC (the variants table) has a
          CI95 above 0; (2) no failure cell has an `_aware − RoundRobin_TRQ`
          CI95 entirely below 0 on its headline metric (§71.4's q = 4 cells
          were +0.003, n.s.: a non-loss is enough there); (3) on the free
          supplement `_aware` is identical to `RoundRobin_TRQ` in every
          cell (*equal* + ties at 0 = n; both run in the same `trq` unit,
          so the same host and FP class).  The identity is also pinned by
          tests: `tests/test_failure_model.py::test_failure_aware_is_bit_identical_without_failures`
          (the arm) and
          `tests/test_heuristic_trust_region.py::test_failure_candidates_are_bit_identical_to_trq_without_failures`
          (the shipped specs through the harness, q 1/4).  Caveat:
          `RoundRobin_TRQ` is nearly seed-invariant at q = 1 (box-centre
          start), `_aware` is not where the start fails (a random
          replacement), so the q = 1 CI is mostly the variant's seed spread
          on fixed, in-sample instances; §71.4's 5/5 there makes (1) likely
          on these instances — it guards against a seed-list accident, not
          against overfitting to the battery.
          (Note added with the battery seed, after the sign-off; the rule
          above is unchanged: on the fresh battery the instances are no
          longer §71's, so the caveat's "in-sample" no longer applies and
          (1) also tests the effect on unseen instances — still only 3 per
          family and d.)
          *The filter* (`RoundRobin_TRQ_fa`) is read descriptively
          (`_fa − _aware` = the difference of their `vs_base` means on the
          same common runs); it stays opt-in whatever it shows (§71.6 (b)).
          The TRQ rule's CIs are unadjusted and outside the Holm family.
      §67's two Holm losses (§72): d2/q64 fixed by `first_round_fill`
      (parity on fresh seeds, §73.2); d10/q4 by `regime_gate="dim-budget"`,
      **the headline default since §73** (the pre-declared rule passed on
      seeds 3001–3012: +0.011 [+0.006, +0.016] 11/12 against the ungated
      spec, identical elsewhere; `Blocks_warm_CMAES_JSO_dimbudget` is gone).
      §73's Holm family (ungated spec): 4 wins, 1 loss (d2/q16 vs qLogEI,
      −0.014, p_holm 0.002), 4 unresolved (with the gate d10/q4 reads
      +0.004, p_holm 0.100, post hoc).  Open from it: the d2/q16 loss (b)
      and the q = 1 loss to Py-BOBYQA at d2 (−0.203, descriptive), both
      mostly where a quadratic model acts; the gate's 20·d d10 cells (not
      run).
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
      active default the sharing portfolio trails `CMAES_alone` on every
      family preset by 0.02–0.15 and `RoundRobin_CMAES` on IOH standard by
      0.028 (§61).  The #346 review saw it first on `ellipsoid_fhs_crash`
      d5; a mechanism to check there: jSO's share of the budget in the
      crash half-space?  §72: the `dim >= 10, bpd <= 500` row holds where
      CMA-ES runs at its default λ (q ≤ λ_default), not where the §63
      floor raises λ (q 16/64: level on `aocc_time`, the portfolio ahead
      on AOCC at q64); and "uniform" blocks are not uniform in
      evaluations — jSO takes 65 % at d10/q4 (its queue rarely drains, so
      its blocks run to the 2× cap).
- [ ] **Balanced blocks** (§72.2 lead): end a block at `block_evals`
      dispatches instead of "`block_evals` and a drained queue".  In sample
      it equalises the split (51/49) and matches CMA-ES alone at d10/q4
      (+0.011 vs +0.010), mixed n.s. at d2/d5.  Changes every cell and the
      cheap track the spec was accepted on (§27/§30): full screen first.
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

- [ ] **Flaky `tests/test_constraints_realistic.py::test_pressure_vessel_design_alm`.**
      Failed once in CI on PR #393 (`best.fx` = 1.4e6 > the 1e5 bound,
      feasible), passed on re-run.  The run is unseeded (master seed from
      numpy's global RNG) on threaded evaluation.  A quick local look
      (2026-09-28) did not find the cause: 20 seeds threaded and 20 with
      `sync_evaluation` all end at fx 5.9e3–7.0e3, cv ≈ 0, so 1.4e6 is far
      outside the seed spread seen here — a thread-timing interaction with
      the ALM multiplier updates (`AugmentedLagrangianConstraintHandler` on
      the event bus) on a loaded 4-core runner is the suspect, not a
      tolerance.  Seeding + `sync_evaluation` would hide it without
      explaining it; `flaky(retries=3)` likewise.  The test now reports
      `strategy.seed` (print and assertion message, battery-seed PR), so
      the next failure can be rerun with that seed, threaded, under load.

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
      §72/§73: `first_round_fill=True` and `regime_gate="dim-budget"` (no
      oracle; one row: unconstrained d ≥ 10, ≤ 500·d, q ≤ λ_default →
      CMA-ES alone) are the defaults, the gate since its fresh-seed
      confirmation (§73).  Not measured with it: the cheap track at
      d ≥ 10 (§61's headline numbers there no longer reproduce; the next
      re-baseline records them).
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
