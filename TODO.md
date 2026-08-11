# TODO

## Recent Improvements (continued)

### Per-cell (per-dim) acceptance — 2026-08-11 (fifth session)
- [x] **`--cell-by dim`, on in the nightly** — the objective is no
      longer a scalar.  Per-cell delta + CI reported every iteration; a
      cell blocks only when it is both worse than `-eps_cell_regress`
      (0.01) and has its whole CI below zero.
- [x] **`per_cell` / `blocking_cell` persisted in the ledger** — the
      record a future codify-scan needs to propose a *dimension-gated*
      arm instead of an unconditional one.
- [ ] **Teach codify-scan to read `per_cell`** — it still pools one
      scalar delta per night, so cell-conditional evidence is recorded
      but not yet actionable.  This is the step that turns the #298 d5
      gain into something shippable.
- [ ] **Budget-phase cells** — `IOHRunRecord` already carries
      `trace_evals` / `trace_fx`, but AOCC would have to be recomputed
      on trajectory slices rather than read off the final value (a
      change to the metric path, not the decision path).  #298's
      evidence says this matters: the same arm leaned *positive* at
      2-D×200 evals and negative at 2-D×1000 — a budget effect, not a
      dimension one.
- [ ] **Dimension-gated arm activation** — the original §4.3 follow-up.
      Now that cells exist, a structural mix that activates population
      arms (NLSHADE_LBC, CMA-ES) only above a dim/budget threshold can
      finally be *measured* rather than argued about.

### Rank-based acceptance available — 2026-08-11 (fourth session)
- [x] **`--accept-stat rank`** — one-sided Wilcoxon signed-rank on the
      per-pair deltas shifted by `eps_accept`, replacing both the
      `delta > eps_accept` and `ci_low > 0` conditions.  Graduates
      GOAL §5.3.
- [x] **Hodges-Lehmann** reported as `rank_delta`; `delta` stays the
      mean under both rules so the ledger series is continuous.
- [ ] **A/B the rank rule against the mean rule, then decide** — NOT
      enabled nightly on purpose: tonight already changes the accept
      regime three ways (seed rotation, eps 0.005→0.0125, d5 slice).
      A fourth simultaneous change would make the ledger
      uninterpretable.  Run it via `workflow_dispatch` on a few nights
      once the new instrument has a baseline, compare accept rates and
      codify survival, and only then consider flipping the default.
- [ ] **Guard the n<5 floor** — a rank accept is impossible below 5
      shared pairs (min p is `2**-n`).  Currently only documented and
      tested; a loud warning when a configured battery cannot clear it
      would be better than a rule that silently never fires.

### Nightly loop can see d5 — 2026-08-11 (third session)
- [x] **`--aocc-extra-dims 5` in the nightly** — the quick preset's
      `dims=(2,)` becomes `(2, 5)` for every measurement leg.  Closes
      the blind spot behind the JSO d5 add (08-02) and the NLSHADE_LBC
      per-dim split (08-11).  Cost 8.9s → 11.9s (1.34×).
- [x] **`with_extra_dims` composes, never edits** the frozen presets;
      name gains a `+d5` suffix.
- [ ] **The ledger score level shifts** with this ship (d2 ~0.369 →
      (d2,d5) ~0.309).  `aocc_extra_dims` is recorded per iteration;
      codify-scan and `summary` should group by it before pooling.
- [ ] **Re-measure the CMA-ES arm (GOAL §5.2) at d5** with the 12-seed
      standard instrument — if it splits like NLSHADE_LBC did, one
      dimension-gating mechanism ships both arms.

### Loop measurement fidelity: hold-out metric bug, seed rotation, sync-eval — 2026-08-11 (second session)
- [x] **Hold-out leg measured `composite_score` on AOCC runs** — the
      8.5× "instance-family generalization gap" (`GOAL.md` §2 / §5.1's
      top research priority) was a unit mismatch.  Real gap after the
      fix: **0.3383 training vs 0.3342 hold-out**.
- [x] **Confirm gate now crosses a base seed under `--metric aocc`** —
      the `metric != "aocc"` exclusion meant 0/72 accepts had ever
      tested a second instance family.
- [x] **Nightly rotates `--base-seed`** over 7 values — all 952 prior
      ledger records were `base_seed=42`, so codify's "k≥2 distinct
      nights" was counting one instance draw k times (hit rate 0/5).
- [x] **`--sync-eval` reachable from the loop** and on by default in
      the nightly (noise sd 0.0101 → 0.0063; measured cost ~0).
- [x] **eps_accept recalibrated** 0.005 → 0.0125 on the aocc branch
      (0.5σ → 2σ); relax floor 0.001 → 0.006.
- [ ] **Re-earn the codify backlog** — cross-night evidence banked
      before 2026-08-11 is single-draw.  Let the rotated-seed ledger
      accumulate ≥2 nights on *distinct* base seeds before trusting any
      slot, and consider making codify-scan group by `base_seed` rather
      than by night.
- [ ] **Group cross-night pooling by `sync_eval`** — the field is now
      recorded but codify-scan does not yet split on it; the first
      nights after this ship straddle the boundary.

### NLSHADE_LBC unconditional add rejected; per-dim split measured — 2026-08-11
- [x] **`add_heuristic NLSHADE_LBC` on `Rewarding_Restart` measured and
      rejected** (negative result, PR #298) — 12-seed paired A/B with
      `--sync-eval` both sides: quick +0.0092 (noise), standard −0.0080
      (lean-negative), but the per-dim split is CI-significant *both
      ways*: **d5 +0.0080 [+0.0007,+0.0154]**, **d2 −0.0241
      [−0.0401,−0.0080]**.  Reverted in-branch; see the 2026-08-11 log
      entry.
- [ ] **Dimension/budget-gated arm activation** — the d5 gain is real;
      the shippable form is a structural mix that activates population
      arms (NLSHADE_LBC, CMA-ES) only when dim/budget clears a
      threshold, or an anytime schedule that phases them out at long
      2-D budgets.  Re-measure the CMA-ES arm (GOAL §5.2) at d5 with
      the 12-seed standard instrument first — if it splits the same
      way, one gating mechanism ships two arms.
- [ ] **Nightly regime blind spot confirmed again** — second measured
      case (after the JSO d5 add) where decisive evidence lived at d5,
      invisible to the quick-2-D nightly battery.  Strengthens the
      case for a d5 (or `--extra-highdim`) slice in the nightly loop
      alongside the multi-seed confirm item below.

### JSO drop_heuristic slot rejected; NLSHADE_LBC slot policy-moot — 2026-08-10
- [x] **`drop_heuristic JSO` on `Rewarding_Restart` rejected** (negative
      result) — ledger slot (2 nights, pooled CI95% `[+0.0058,+0.0075]`)
      measured flat on the 12-seed paired quick A/B: mean Δ `+0.0026`,
      sd 0.0249, CI95% `[−0.0132,+0.0185]`, control flat; seed 42 alone
      `+0.0200`.  Fourth consecutive training-seed artifact (0/4 screening
      hit rate).  Would have reversed the standard-battery-validated
      2026-08-02 JSO add (d5 +0.0287).  See the 2026-08-10 log entry.
- [x] **`add_heuristic NLSHADE_LBC` → `RoundRobin_Random` recorded
      policy-moot** — the slot only targets the pure-random reference /
      A/B control spec, which stays untouched by standing judgement call.
- [x] **Measure `add_heuristic NLSHADE_LBC` on `Rewarding_Restart`**
      (§4.3 candidate with a positive prior: 3 confirmed control-spec
      accepts across 2 nights say the arm is strong under AOCC) —
      done 2026-08-11: rejected as an unconditional add (d2 loss
      outweighs the significant d5 gain); see the 2026-08-11 entries
      above.
- [ ] **Multi-seed confirm in the nightly loop** — the 0/4 codify
      screening hit rate bounds the loop's value at the current plateau;
      rotate the nightly base seed and/or add a second-seed confirm gate
      before a screening accept lands in the ledger (extends the
      2026-08-03 "price instance sensitivity into the codify gate" item).

### Sync-eval mode: scheduling noise quantified and halved — 2026-08-09
- [x] **Fixed-seed repeatability measured** (10 identical quick runs,
      seed 42): `Rewarding_Restart` battery-mean AOCC sd **0.0206**
      (range 0.060), `RoundRobin_Random` sd 0.0012 — the "per-seed"
      decision noise is almost entirely *scheduling* nondeterminism in
      the adaptive strategy, not instance sensitivity.  See the
      2026-08-09 log entry for the full source table.
- [x] **`--sync-eval` shipped** (`ioh_benchmark.py run` →
      `config.sync_evaluation` → synchronous future harvest in
      `_run_threaded_evaluation`): pooled repeat sd over seeds
      42/1234/777 drops 0.0183 → 0.0115 (**1.6× sd, 2.5× variance**;
      seed-heterogeneous: 2.3×/1.25×/1.4×).  Opt-in, default-off,
      `compare` warns on mode mismatch, results carry a `sync_eval`
      tag.  Use on both sides of future A/Bs; the N=12 instrument-level
      CI shrink is expected but not yet demonstrated (one null A/B per
      mode couldn't resolve it — keep accumulating nulls).
- [ ] **Event-drain wait measured ineffective** (negative result) —
      waiting for eventbus queues to empty after publishing results did
      not reduce sd further (0.0113 vs 0.0094); queue-empty ≠ handlers
      idle.  A real synchronous stepping mode needs handler-completion
      tracking, not queue polling.
- [ ] **Residual ~0.009 sd: shared global RNG across handler threads** —
      heuristics draw from `np.random` inside per-handler EventBus
      threads, so thread interleaving reorders the stream even at a
      fixed seed.  Next lever: per-heuristic `np.random.Generator`
      seeded from (run seed, heuristic name); mechanical but touches
      ~14 heuristic modules.  Measure with the same 10-repeat protocol.
- [ ] **Adopt `--sync-eval` in the nightly loop / codify verification**
      once a few sessions have used it interactively without surprises
      (it shifts absolute AOCC within noise; ledger continuity says
      switch deliberately, not silently).

### Codify rejection memory: k ≥ 2 fresh-night resurrection bar — 2026-08-08
- [x] **Single-fresh-night resurrection churn stopped** — rejected codify
      slots now stay hidden until the *post-rejection* evidence alone
      reaches `--min-fresh-nights` (default 2) distinct nights; one fresh
      seed-42 night no longer re-opens a slot a 12-seed A/B rejected
      (measured 0/3 hit rate across the 2026-08-03..07 resurrections).
      `--min-fresh-nights 1` restores legacy semantics; audit view shows
      per-slot progress toward the bar.  See the 2026-08-08 log entry.
- [ ] **Rotate the nightly base seed by date** — with every ledger night
      keyed to seed 42, `n_nights >= 2` measures persistence of one
      training-battery draw, not cross-instance generality.  A dated seed
      rotation would make cross-night pooling cross-seed for free (check
      trend-table comparability + hold-out seed disjointness first).
- [ ] **Codify pre-gate: auto 12-seed paired A/B** — before `--apply-top`
      declares a slot actionable, optionally run the
      `ioh_benchmark.py run --decision-seeds` instrument and require a
      CI95 excluding zero (mechanises the manual protocol every session
      currently hand-runs; complements the 2026-08-03 "price instance
      sensitivity into the codify gate" item below).

### Codify slot `NelderMead drop_heuristic` rejected — 2026-08-07
- [x] **`NelderMead drop_heuristic` measured flat and rejected** —
      ledger evidence (2 nights, pooled CI95% `[+0.0067,+0.0083]`)
      did not survive the 12-seed paired quick A/B: `Rewarding_Restart`
      mean Δ `−0.0003`, sd 0.0188, CI95 `[−0.0122,+0.0117]`, control
      flat; seed 42 alone `+0.0174` (training-seed artifact, same
      signature as both 2026-08-03 Sensitivity rejections).  Spec
      unchanged (applied + reverted in-branch); rejection recorded in
      `planning/self_improve_rejections_aocc.json`.  See the
      2026-08-07 log entry.
- [ ] **Multi-seed pre-gate for codify-scan** (raises priority of the
      existing "price instance sensitivity into the codify gate" item)
      — three consecutive ledger-positive slots rejected flat by the
      12-seed instrument means single-seed `min_nights=2` evidence has
      ~0 hit rate at the current plateau.  Cheapest fix: higher
      `min_nights` for structural ops + a small (6-seed) screening A/B
      in the nightly post-loop step before a slot is surfaced
      actionable.
- [ ] **Promote codify verification to `--standard`** when the quick
      battery saturates (GOAL §4 step 4) — `Rewarding_Restart` per-seed
      quick sd (~0.019) is ~9× the control's; effects below ~0.012
      cannot clear a 12-seed quick CI95.

### CMA-ES arm in the structural catalog (GOAL §5.2) — 2026-08-06
- [x] **`CMAES` added to the `add_heuristic` candidate pool** — the
      existing full CMA-ES heuristic (IPOP/BIPOP, `heuristics/cma_es.py`)
      was unreachable by the nightly loop; the bandit can now measure the
      only covariance-adapting family against the DE arms.  Explicit
      `sigma0=0.3` makes the existing `CMAES.sigma0` kwarg rule fire.
      Direct add to `Rewarding_Restart` measured **flat** on a 12-seed
      paired quick A/B (Δ +0.0005, CI95 [-0.0113,+0.0123]) → not shipped
      into the spec; the catalog route lets the ledger/hold-outs decide
      at the regimes (5-D rotated valleys) where the arm should matter.
      See the 2026-08-06 log entry.
- [x] **`drop_analyzer Sensitivity` re-hidden** — resurfaced from a fresh
      2026-08-06 single-seed night but is an apply-guard no-op (last
      analyzer in the bucket) on a spec unchanged since the 2026-08-03
      12-seed rejection; `codify-reject` re-recorded dated 2026-08-06.
- [ ] **Annotate guard-suppressed codify candidates** — `codify-scan`
      surfaces slots whose `--apply-top` would be a safety-guard no-op
      (e.g. dropping the last analyzer) as "actionable"; detect and tag
      (or hide) them so sessions don't burn the codify slot on a no-op.

### Paired multi-seed decision instrument in `ioh_benchmark.py` — 2026-08-05
- [x] **Multi-seed A/B mechanised** — `run --seeds` / `--decision-seeds`
      (canonical 12-seed roster) / `--reps K` and a seed-paired `compare`
      (per-strategy Δmean/sd/CI95 via t-dist, per-seed deltas, verdict
      markers, loud mixed-format error).  The 2026-08-03 decision protocol
      is now one flag instead of a hand-rolled bash loop.  15 new tests;
      single-seed files/workflows unchanged.  See the 2026-08-05 log entry.
- [x] **Deterministic evaluation mode** (follow-up to the nondeterminism
      finding) — `_run_threaded_evaluation` harvests whichever futures are
      `done()` per loop pass, so the strategy's result view depends on OS
      scheduling.  *Partially shipped 2026-08-09 as `--sync-eval`
      (synchronous harvest, 2.3× repeat-sd cut); full determinism blocked
      on the shared-RNG / handler-thread items in the 2026-08-09 section
      above.*

### Codify-scan rejection memory — 2026-08-04
- [x] **Rejection memory shipped** — `codify-scan` now consults a
      per-metric rejections file
      (`planning/self_improve_rejections_<metric>.json`); rejected/moot
      slots are hidden from the report and skipped by `--apply-top`
      until fresh post-rejection evidence resurrects them (tagged for
      re-verification).  New `codify-reject` subcommand records
      decisions; seeded with the five resolved aocc slots
      (LBFGSB/Center/Sobol.n 07-30, both Sensitivity slots 08-03).
      `codify-scan --metric aocc` now truthfully reports 0 actionable
      candidates.  See the 2026-08-04 log entry.

### Codify queue cleared; standard-battery nondeterminism found — 2026-08-03
- [x] **Both remaining `Sensitivity` codify slots rejected** (negative
      results) — `update_interval 25 → 20` (ledger pooled CI95%
      `[+0.0092, +0.0117]`) and `drop_analyzer Sensitivity` (pooled CI95%
      `[+0.0067, +0.0075]`) both measured flat on a 12-seed paired quick
      A/B against the current spec: mean Δ `−0.0012` / `−0.0007`, CI95%
      straddling zero, controls flat.  Training-battery artifacts; see the
      2026-08-03 log entry.  All 6 scan candidates now resolved (JSO →
      PR #289; LBFGSB/Center → rejected 07-30; Sobol.n → moot).
- [x] **Standard battery measured nondeterministic run-to-run** — identical
      tree+seed re-run shifts both arms by ≈ ±0.015 (threaded evaluation).
      Decision protocol updated in the log: ≥ 12 paired quick seeds with
      flat-control check, or ≥ 5 standard replicates per side.
- [x] **Codify-scan rejection memory** — the scan re-surfaces
      A/B-rejected and moot slots every night (no counterpart to the
      already-codified suppression).  Add a rejected-slot suppression
      list (slot key + rejection date + evidence pointer) consulted by
      `codify-scan` so the nightly report and `--apply-top` skip them.
      *Shipped 2026-08-04 — see the section above.*
- [ ] **Fix threaded-evaluation nondeterminism in the IOH harness** — the
      standard battery cannot currently resolve +0.01-scale effects with a
      single run; find the ordering/seeding race and make batteries
      reproducible per seed (quick battery already is, modulo rare
      ±0.01 outliers).
- [ ] **Price instance sensitivity into the codify gate** — nightly
      evidence keys on the seed-42 training battery; per-seed null-change
      sd is ~0.015.  Raise `min_nights` (2 → 3+) and/or add a multi-seed
      confirm to the scan before surfacing a slot as actionable.

### Codify: `JSO({'NP_init': 'auto'})` added to `Rewarding_Restart` — 2026-08-02
- [x] **Codify banked** — `add_heuristic JSO` slot from the aocc ledger
      (2 confirmed nights, pooled CI95% `[+0.0092, +0.0133]`; the 2026-08-01
      accept was measured on the current post-Sobol-drop spec).  Standard-
      battery paired A/B: `Rewarding_Restart` mean AOCC `0.3374 → 0.3596`
      (+0.0222; d2 +0.0156, d5 +0.0287), control flat.  d5 now clearly beats
      the random floor (0.3196 vs 0.2878).  Edit scoped to `Rewarding_Restart`
      only — `RoundRobin_Random` stays the untouched reference.

### Metric-aware codify routing + first aocc codify + goal contract — 2026-07-30
- [x] **Bug fix (the aocc codify stall)** — `default_codify_registries()` and
      `default_codify_apply_sources()` in `panobbgo/self_improve.py` gain a
      `metric: str = "composite"` parameter; `"aocc"` routes suppression to
      `panobbgo.harness_ioh.make_ioh_strategies` and the `--apply-top` edit
      driver to `panobbgo/harness_ioh.py`.  Between 2026-07-09 (nightly metric
      flip) and this fix, aocc evidence could never land as a source edit —
      the driver scanned `harness.py`, found no matching spec, and silently
      no-oped while the bandit re-discovered the same wins nightly (18
      confirmed `drop_analyzer Restart` accepts across 17 nights).
      `codify-scan` threads `--metric` into both call sites.
- [x] **First aocc codify banked** — dropped the `Restart` analyzer from the
      `Rewarding_Restart` spec in `make_ioh_strategies` via the fixed driver.
      Local paired A/B (quick IOH battery): `Rewarding_Restart` mean AOCC
      `0.3538 → 0.3922`, `RoundRobin_Random` control flat.  Post-codify scan
      auto-suppresses the candidate (self-stability verified).
- [x] **Nightly visibility** — `self_improve_nightly.yml` now regenerates and
      commits `planning/self_improve_codify_scan.txt` (metric-aware) alongside
      the summary, so actionable evidence is readable without running anything.
- [x] **Goal contract** — new `planning/GOAL.md`: metric of record, per-session
      operating loop, multi-day escalation ladder, SOTA-informed research
      backlog (MA-BBOB / LLaMEA / modular CMA-ES context).  Pointer added to
      `AGENTS.md`.
- [x] **Validation** — 6 new tests (`TestMetricAwareCodifyRouting`); full
      `tests/test_self_improve.py` suite green (611 passed); ruff clean.
- [x] **Queued codify slots worked through (2026-07-30 second session)** —
      `drop_heuristic Sobol` ACCEPTED (standard battery +0.0176, spec now
      beats random at both dims); `add_heuristic LBFGSB` REJECTED (−0.0146
      on the post-Sobol-drop spec — interaction negative); `drop_heuristic
      Center` REJECTED (−0.0229; Center is load-bearing without Sobol);
      `Sobol.n 32 → 38` moot.  Structural-add driver hardened in the same
      session: missing-comma fix, `structural_kwargs` carried into edits,
      factory-import rewriting, parse-validation net in
      `apply_codify_edits` (8 new tests).  See the log's second 2026-07-30
      entry.
- [ ] **Open weakness** — hold-out base seeds score far below training seed
      (0.04 vs 0.33 on 2026-07-30): instance-family generalization is the
      top research target (see `planning/GOAL.md` §5).

### Opt-in higher-dimensional battery (`extra_families` / `--extra-highdim`) — 2026-07-13
- [x] **Measurement substrate** — added
      `panobbgo/harness_randomized.py::make_highdim_families` (a rotated
      `Rosenbrock_HighDim_family` at `dim_choices=(2, 5)`, stratified, with the
      `rotate` transform the default `Rosenbrock_family` omits) and an opt-in
      `HarnessConfig.extra_families` hook appended to the default randomized
      battery in `get_problems()` when `randomize=True`.  Also threaded into
      `LoopConfig.extra_families` (both `harness_config` and
      `holdout_harness_config`) so the loop, guard, and hold-out all measure the
      extended battery on the composite path.
- [x] **Motivation** — the default battery is `dim_choices=(2,)` everywhere, so
      the whole self-improvement apparatus measures only at dim 2; the
      2026-07-08 → 2026-07-12 curvature-aware `Nearby` work (3.3× lower residual
      on rotated 5-D Rosenbrock) is byte-identical on the 2-D battery and thus
      loop-invisible.  This is the additive first layer of the backlog's
      highest-leverage ticket ("the measured battery is 2-D-only").
- [x] **No composite-contract change** — the default families are untouched
      (`extra_families` defaults to `None`), so a plain `--randomize` run stays
      byte-identical and every historical composite comparison is unchanged.
      `extra_families` is stripped from `HarnessResult.to_dict()` (like
      `strategies_override`) so results stay JSON-serialisable.
- [x] **CLI** — `benchmark_harness.py run/list --extra-highdim` and
      `scripts/self_improve.py run --metric composite --extra-highdim` (inert on
      `--metric aocc`, whose battery lives in `panobbgo.harness_ioh`).  AOCC is
      the recommended signal for this hard family (composite floors at 0.0 on a
      rotated 5-D valley at quick budget).
- [x] **Validation** — 9 new tests (`TestHighDimFamilies` in
      `tests/test_harness_randomized.py`, `TestExtraFamilies` in
      `tests/test_harness.py`); full `test_harness` / `test_harness_randomized` /
      `test_self_improve` suites green (738 passed); ruff + pyright clean; a
      real `--extra-highdim` run of the harness and the loop CLI verified
      end-to-end.
- [x] **Docs** — 2026-07-13 dated entry + follow-up layers in
      `planning/SELF_IMPROVEMENT_LOG.md` (nightly aocc 5-D quick battery;
      default-battery promotion via ADR); `AGENTS.md` bullet;
      `doc/source/guide_benchmarking.rst` "Opt-in extended battery" subsection;
      `SELF_IMPROVEMENT_LOOP.md` §10; module/field docstrings; this entry.
      §7.3-freeze compliant (a measurement/battery hook — no new mutation arms).

### Diagonal-plus-low-rank Hessian for the `Nearby` heuristic — 2026-07-12
- [x] **Heuristic improvement** — added a `hessian_rank` argument to
      `panobbgo/heuristics/nearby.py::fit_quadratic_step` and the matching
      `Nearby.quadratic_hessian_rank` kwarg (default `"auto"`).  Replaces the
      single ridge-regularised *full* quadratic with a **diagonal-plus-low-rank**
      model: the local cloud is rotated into its weighted-PCA basis and a
      quadratic is fitted with coupling terms only among the top-`r` principal
      directions, `r` chosen per fit by **BIC**.  Closes the *5-D
      coordinate-coupling* half of the `Rosenbrock_5D` gap the 2026-07-11
      localisation left open — a full quadratic needs `O(dim²)` local points, so
      from a thin valley cloud the ridge shrinks the cross terms to zero and the
      Newton step points across the valley.  `hessian_rank=None` restores the
      byte-identical legacy full quadratic.
- [x] **Byte-identical on the measured battery** — at `dim ≤ 2` (a single,
      always-well-determined cross term) `"auto"` keeps the legacy full
      quadratic, so it is byte-identical to the previous behaviour on the 2-D
      quick / standard / loop battery (unit-test-enforced via
      `np.testing.assert_array_equal`).  Zero regression risk, no loop / guard /
      codify impact.
- [x] **Measured impact** — the measured battery is 2-D (where the benefit
      cannot appear); the win is at higher dimensions.  Isolated
      `fit_quadratic_step` diagnostic on rotated **5-D Rosenbrock** valleys
      (production `hessian_rank=None` vs `"auto"`): median residual **24.5 → 7.5
      (3.3× lower)**, mean 293 → 167; isotropic sphere 0.106 → 0.033; a slight
      median rise on a rotated dense quadratic (0.42 → 0.57, both trivially
      small).  Quick-mode composite unchanged (2-D → legacy path); the harness
      is non-deterministic run-to-run so a naive 2-D A/B measures only RNG noise.
- [x] **Validation** — 8 new/extended tests in
      `tests/test_heuristic_nearby_quadratic.py` (auto-is-default, low-dim
      byte-identity, full-rank == fixed-rank-dim, rotated-valley coupling win,
      sphere non-regression, fixed rank-0 descent, kwarg validation +
      forwarding, legacy selection).  Full suite green (1867 passed, 11
      skipped); ruff + pyright clean.
- [x] **Docs** — dated entry + follow-up ideas (incl. the 2-D-only-battery
      limitation) in `planning/SELF_IMPROVEMENT_LOG.md`; `AGENTS.md` bullet;
      `doc/source/guide.rst`; module docstrings; this entry.  §7.3-freeze
      compliant (better default for an existing kwarg — no new arms).

### Gaussian-localised quadratic fit for the `Nearby` heuristic — 2026-07-11
- [x] **Heuristic improvement** — added a `weight_sigma` argument to
      `panobbgo/heuristics/nearby.py::fit_quadratic_step` and the matching
      `Nearby.quadratic_weight_sigma` kwarg (default `0.35`).  The local
      quadratic is now fitted with a **Gaussian distance kernel**
      `w_i = exp(-½ (d_i / (σ·median_d))²)` instead of the mild legacy rank
      weights `1/(1+rank)`, so the model tracks the *local* valley floor rather
      than averaging over a wide cloud.  Directly targets the curved-valley
      (Rosenbrock) gap: a quadratic averaged over a wide cloud sends the Newton
      step across (not along) the valley and stalls.  `weight_sigma=None`
      restores the byte-identical legacy weighting; `quadratic=False` remains
      byte-identical to the classic heuristic.
- [x] **Diagnosis** — an isolated `fit_quadratic_step` sweep on a synthetic 5-D
      Rosenbrock valley showed the *locality of the fit sample* (not the ridge
      strength or the `min_r2` gate) dominates step quality; a Gaussian kernel
      with `σ ≈ 0.35` quartered the residual objective (median 0.0062 vs 0.0252
      for the legacy rank weights).
- [x] **Measured impact** — paired `--randomize` A/B of `Rewarding_Diverse`
      on the quick battery (16 randomize-iterations × 10 reps, legacy baseline
      via monkeypatched `weight_sigma=None`): composite **+0.0176** (0.097 →
      0.115, +18%), median +0.0203, 95% bootstrap CI **[+0.0083, +0.0267]**,
      12/4 win/loss.  Clears `statistical_accept` (Δ > eps_accept, CI_low > 0,
      worst per-family regression −0.0028 well inside eps_regress).  Per-family:
      `DeJong` +0.043, `Rosenbrock` +0.029 (smooth/valley), `Ackley` +0.001 /
      `Rastrigin` −0.003 (multimodal, neutral).
- [x] **Validation** — 6 new/extended tests in
      `tests/test_heuristic_nearby_quadratic.py` (legacy-path bit-exactness,
      default wiring, curved-valley localisation, degenerate-cloud guard, kwarg
      validation + forwarding).  Full suite green (1859 passed, 11 skipped);
      ruff + pyright clean.
- [x] **Docs** — dated entry + follow-up ideas in
      `planning/SELF_IMPROVEMENT_LOG.md`; `AGENTS.md` bullet; `doc/source/guide.rst`;
      module docstrings; this entry.  §7.3-freeze compliant (better default for
      an existing kwarg — no new arms).

### Flip the nightly default metric to AOCC (V2 §9.5 step 2 — closes §2.1) — 2026-07-09
- [x] **Metric flip** — flipped the `self_improve_nightly.yml` scheduled
      default `metric` from `composite` to `aocc` (the IOH/MA-BBOB anytime
      metric).  The `--metric aocc` code path, the ioh_worker venv sync, and
      the vacuous-hold-out handling under aocc were all already in place
      (shipped 2026-07-04 as the `workflow_dispatch` A/B lever) — this change
      only flips the default and separates the ledgers.  No Python source
      changed.
- [x] **Metric-separated ledgers** — added a *Resolve ledger path* workflow
      step that routes each metric to its own append-only ledger so composite
      deltas (~0.003 scale) and aocc deltas (~0.3 scale) never mix in
      codify-scan pooling or the graded bandit reward: `aocc` →
      `planning/self_improve_ledger_aocc.jsonl` (fresh canonical ledger),
      `composite` → the historical `planning/self_improve_ledger.jsonl`
      (660-record durability history preserved, grows only on a `composite`
      A/B dispatch).  Threaded `$LEDGER` through the run / summary / commit /
      artifact steps.
- [x] **Measured impact** — in-mode (quick) A/B in the exact scheduled regime:
      aocc median seed score **0.33** (inside §11.1's 0.3–0.6 responsive band),
      **0% Δ=0 rate**, 35% accept rate over a 20-iter loop, vs composite's
      floored **0.036** median, 8% Δ=0 rate, 3.2% accept rate on the
      660-record production ledger.  AOCC meets §11.1 resolution; composite
      does not — exactly the §2.1 "no metric resolution" diagnosis, now
      resolved.
- [x] **Validation** — no tests assert the workflow's metric default or
      ledger path, so the pytest / ruff / pyright suites are unaffected;
      workflow YAML parses; a 3-iteration smoke run of the exact flipped
      invocation (fresh non-existent ledger + `--adaptive-prime-from-ledger`
      `--prime-include-archives` `--confirm-accepts` + hold-out seeds)
      completes cleanly with AOCC scores in the responsive band.
- [x] **Docs** — dated entries + follow-up ideas in
      `planning/SELF_IMPROVEMENT_LOG.md`; §2.1 / §9.4 / §9.5 step 2 / §12.1
      updates in `planning/SELF_IMPROVEMENT_LOOP.md`; `AGENTS.md` bullet +
      usage section; `doc/source/guide.rst` and `guide_benchmarking.rst`; this
      entry.

### Curvature-aware quadratic local step for the `Nearby` heuristic — 2026-07-08
- [x] **Heuristic improvement** — added `quadratic` / `quadratic_trust` /
      `quadratic_min_r2` kwargs to `panobbgo/heuristics/nearby.py::Nearby` plus
      the module-level `fit_quadratic_step`.  When `quadratic=True`, `Nearby`
      keeps a rolling buffer of recent `(x, f(x))` pairs (new `on_new_results`
      hook, lock-guarded) and, on each new best, fits a distance-weighted ridge
      quadratic to the nearest points in box-normalised coordinates and emits
      its trust-region Newton minimiser as the first of its `new` points (rest
      stay isotropic).  Hardened with per-column ridge, PD Hessian
      regularisation, a data-support trust region, and a weighted-R² fit-quality
      gate (`0.8`) that falls back to isotropic exploration on multimodal
      neighbourhoods.  `quadratic=False` (default) is byte-identical to the
      classic heuristic.  Fitted in-process from points the portfolio already
      evaluated → **zero** extra objective evaluations.
- [x] **Seed adoption** — flipped the six Rewarding-family
      `(Nearby, {radius:0.124, axes:"all", new:3})` refinement entries to
      `quadratic=True` in `panobbgo/harness.py` (`Rewarding_Diverse`,
      `Rewarding_RegionUCB`, `UCB_Diverse`, `Thompson_Diverse`,
      `Loop_RegionUCB`, `Loop_Restart`).  GP-specialised `radius=0.05` entries
      left untouched.  §7.3-freeze compliant (better default kwargs; no new
      catalog arms).
- [x] **Measured impact** — paired `--randomize` A/B on `Rewarding_Diverse`
      (quick, reps 12, iter 0): composite **0.0339 → 0.0612**, `statistical_accept`
      **ACCEPT** Δ=+0.0274 95% CI `[+0.0057, +0.0521]`, worst-pair −0.0178.
      Mean Δ ≈ **+0.075** over 20 randomized iters × 2 base_seeds (18/20
      positive).  Effect is specific to the ill-conditioned randomized battery
      (the loop's own metric); net-neutral within noise on the fixed
      natural-conditioning battery.
- [x] **Tests** — 21 new tests in `tests/test_heuristic_nearby_quadratic.py`
      (fit recovery, robustness guards, R²-gate accept/reject, `Nearby`
      wiring).  Full `Nearby` suite (32) + harness suites (130) + loop-registry
      suite green; ruff + pyright clean.
- [x] **Docs** — `doc/source/guide.rst` (catalog clause), dated entry +
      graduated top-priority idea in `planning/SELF_IMPROVEMENT_LOG.md`,
      `AGENTS.md` bullet, seed-site comments, this entry.

### Warm-started memetic L-BFGS-B restarts for the curved-valley class — 2026-07-07
- [x] **Heuristic improvement** — added `warm_start` / `warm_start_sigma`
      kwargs to `panobbgo/heuristics/lbfgsb.py::LBFGSB`.  Every restart after
      the first box-centre descent polishes a small Gaussian perturbation of
      the strategy's best incumbent (tracked via a new `on_new_best` hook)
      instead of a fresh uniform-random point — the memetic recipe scipy
      `dual_annealing` uses.  The subprocess worker requests the restart `x0`
      from the parent over the existing request pipe (`_X0_REQUEST` sentinel);
      `on_start` answers inline with `_warm_start_x0`, falling back to a
      uniform draw before the first result.  `warm_start=False` (default) is
      byte-identical to the historical uniform-restart worker.
- [x] **Structural-catalog adoption** — flipped the
      `default_structural_catalog` LBFGSB candidate from `(LBFGSB, {})` to
      `(LBFGSB, {"warm_start": True})` so the loop's `add_heuristic` op inserts
      the warm variant.  Directly targets the 2026-07-06 negative result
      (cold LBFGSB bolted onto `Rewarding_Diverse` *regressed* the composite —
      wrong restart geometry).  §7.3-freeze compliant (better default kwargs
      for an existing candidate; no new arms).
- [x] **Measured impact** (≥3-seed aggregates): `[Sobol, Random, Nearby,
      LBFGSB]` on the curved-valley battery at full budget — warm **0.198** vs
      cold 0.156 (+0.042); `[Sobol, LBFGSB, NelderMead]` at standard budget —
      tie (0.583 both) but warm's `Rosenbrock_5D` best-distance lower (11.7 vs
      15.8).  No regression anywhere.  Fully crossing the `Rosenbrock_5D`
      tolerance still needs more budget / a dedicated local-search strategy
      (ADR-gated; left as a next idea).
- [x] **Tests** — 17 new tests in `tests/test_heuristic_lbfgsb.py`
      (warm-start construction validation, `on_new_best`/`_warm_start_x0`/
      `on_start` sentinel handling, worker x0-request protocol +
      clean-exit-on-closed-pipe).  Full LBFGSB suite (59) + structural-catalog
      suite (145) green; ruff + pyright clean.
- [x] **Docs** — `AGENTS.md` (structural-catalog mention + dedicated bullet),
      `doc/source/guide.rst` (catalog sentence), dated entry in
      `planning/SELF_IMPROVEMENT_LOG.md` + updated top-priority idea, the
      `default_structural_catalog` comment, this entry.

### Codify: drop the `LatinHypercube` seeder from `Loop_LocalSearch` — 2026-07-06
- [x] **Structural codify applied via the automated pipeline** — removed
      `(LatinHypercube, {"div": 4})` from the `Loop_LocalSearch` seed spec in
      `panobbgo/harness.py::_make_loop_strategies` using
      `scripts/self_improve.py codify-scan --apply-top --apply-format`.  First
      *structural* codify to land through the automated apply driver.
- [x] **Ledger evidence** — two independent `drop_heuristic` accepts:
      2026-06-24 Δ=+0.0511 CI=[+0.0352,+0.0670]; 2026-06-29 Δ=+0.0471
      CI=[+0.0368,+0.0617]; pooled CI [+0.0471,+0.0511].  Flagged as the top
      structural candidate in the 2026-07-01 log entry.
- [x] **Idempotent** — re-run of `--apply-top` derives 0 edits (missing-class
      safety guard); the `--open-pr` driver returns before opening an empty PR.
- [x] **Measured flagship gap + negative result** — standard `--baselines`
      run: every Panobbgo strategy scores 0 on Rosenbrock_5D vs scipy dual
      annealing 0.49.  Bolting `LBFGSB` onto `Rewarding_Diverse` regresses a
      3-seed composite (0.657 → 0.652/0.643); documented so the next iteration
      tries a warm-started curvature-aware polish instead.
- [x] **Tests** — `tests/test_self_improve.py` + `tests/test_harness.py`
      (646 passed).
- [x] **Docs** — `AGENTS.md`, `planning/SELF_IMPROVEMENT_LOG.md` (dated entry
      + two Next iteration ideas), `planning/SELF_IMPROVEMENT_LOOP.md` (§11
      criterion 2), the `_make_loop_strategies` comment, this entry.

### Budget-adaptive `NP_init="auto"` for the DE family + structural-catalog adoption — 2026-07-05
- [x] **`LSHADE` (and subclasses `JSO` / `NLSHADE_RSP` / `NLSHADE_LBC` /
      `LSHADE_EpSin`) accept `NP_init="auto"`** — budget-adaptive population
      sizing `clip(round(min(18·dim, budget/12)), max(NP_min, 6), 400)`,
      resolved in the base constructor via `_resolve_auto_np_init` so every
      subclass and downstream path (validation, `on_start`, LPSR,
      `LSHADE_EpSin` `G_max`) sees a normal `int`.  Falls back to the fixed
      default `30` when the budget is unknown; the `int` default is unchanged
      (byte-identical).  A `bool` is now rejected explicitly (it is an `int`
      subclass and must not size a population).
- [x] **`default_structural_catalog` DE candidates ship `NP_init="auto"`** —
      so a structurally-added DE arm is sized for the strategy budget instead
      of a fixed oversized swarm.  `_find_targets` gained a `rule_kind`
      argument: numeric mutation rules skip non-numeric values (the `"auto"`
      sentinel is ignored, never `int("auto")`-crashed) while categorical
      rules still see strings (`F_schedule` regime flips keep working).
- [x] **Measured impact** — lone `LSHADE` / `Rosenbrock_2D`, 6 reps: at the
      quick-mode budget 75 (the nightly loop's operating budget) `NP_init=30`
      scores **0.036** vs `"auto"` (NP=6) **0.604** — a ~16× win (an
      oversized swarm otherwise burns the budget on the initial random fill);
      at budget 200 a 3-seed sweep of `NP_init ∈ {15,17,30}` measured
      0.42/0.43/0.46 — within noise, no regression.  Respects the §7.3
      catalog freeze (no new arms).
- [x] **Rejected (measured negative result)** — a Hooke-Jeeves pattern-move /
      directional momentum for `Nearby` was implemented and measured across
      three designs at momentum ∈ {0.5, 1.0} over 5 seeds; every variant was
      null-to-negative on composite (`Rosenbrock_2D` degraded — straight
      extrapolation overshoots the curved valley).  Reverted; documented in
      the 2026-07-05 log entry so it is not re-tried.
- [x] **Tests** — 8 new `LSHADEAutoNPInitTests` + 6 new tests
      (`TestNumericRuleSkipsStringSentinel` / `TestStructuralCatalogDEAutoSizing`);
      all affected suites green (802 passed).
- [x] **Docs** — `doc/source/heuristics.rst`, `panobbgo/heuristics/lshade.py`
      (+ 4 subclass docstrings), `AGENTS.md`, `planning/SELF_IMPROVEMENT_LOG.md`
      (dated entry + Next iteration ideas), this entry.

### `--metric aocc` workflow_dispatch A/B mechanism in the nightly cron (V2 §9.5 step 2) — 2026-07-04
- [x] **Workflow surface** — new
      `workflow_dispatch.inputs.metric: choice[composite, aocc]`
      input (default `composite`) on
      `.github/workflows/self_improve_nightly.yml`, sitting next to
      the existing `iterations` / `mode` / `confirm_accepts` inputs.
      Scheduled runs default to `composite` (fall-through pattern
      mirrors `confirm_accepts`), so pre-2026-07-04 cron behaviour
      is byte-identical for the daily trend-table `seed_score`
      column that would otherwise become apples-to-oranges across
      the transition night.
- [x] **IOH worker sync step** — new `Cache IOH worker venv` +
      `Sync IOH worker venv` steps mirroring `tests.yml`.  Key
      derived from `tools/ioh_worker/pyproject.toml` +
      `tools/ioh_worker/uv.lock`; the cp312 manylinux `ioh` wheel
      (~8 MiB) survives normal lockfile refreshes via restore-keys
      degradation.  Kept **eager** (not conditional on
      `METRIC == aocc`) so an operator who flips the dropdown gets
      a warm venv immediately — the ~2 s cold-cache tax is
      amortised across every scheduled run instead of surfacing as
      a spike on the first aocc dispatch.
- [x] **CMD-array append** — two-way conditional at the end of the
      loop invocation (mirrors the `CONFIRM_ACCEPTS` shape shipped
      2026-06-27): `if [ "$METRIC" = "aocc" ]; then CMD+=(--metric
      aocc); fi`.  Nothing else in the CMD array changes; every
      other V2 flag (`--registry loop`, `--adaptive`,
      `--prime-include-archives`, `--structural-per-class-arms`,
      `--bandit-reward graded`, `--inactivity-relax-after 10`,
      `--holdout-base-seeds 7,1234`, `--guard-interval 10`) composes
      cleanly with either metric.
- [x] **Commit-message tag** — aocc-regime dispatch runs get
      `mode=$MODE, metric=aocc` in the commit subject so an auditor
      grepping `git log` can identify A/B nights; the composite
      default preserves the pre-2026-07-04 commit-message shape so
      scheduled runs read identically to the trailing 30 days of
      history.
- [x] **Local smoke test** — `uv run python
      scripts/self_improve.py run --iterations 1 --mode quick
      --metric aocc --base-seed 42 --ledger /tmp/aocc_smoke.jsonl`
      after `cd tools/ioh_worker && uv sync`.  The IOH worker
      spawned successfully, the AOCC harness ran end-to-end against
      the quick IOH battery, and the loop produced a single reject
      iteration (`Rewarding_Restart / Sobol.n: 32 → 36`) with a
      non-zero Δ (`+0.0033`) where the same slot has historically
      produced Δ = 0 exactly on the composite path — smoke-test
      only, not a signal-quality claim, but confirms the wiring is
      intact.
- [x] **Documentation** — `planning/SELF_IMPROVEMENT_LOG.md` new
      2026-07-04 dated entry; `planning/SELF_IMPROVEMENT_LOOP.md`
      §9.5 step 2 progress note plus the summary in §9.5 step 5
      updated to reflect that all V2 flags are now wired in the
      cron; `doc/source/guide.rst` benchmarking summary line
      extended with the new entry; `AGENTS.md` new bullet under
      the V2 ship list + expanded AOCC section with the
      `workflow_dispatch` usage note; this TODO entry.
### `codify-scan --apply-top --apply-format` / `--apply-run-tests` hygiene flags — 2026-07-03
- [x] **Two new CLI flags on `scripts/self_improve.py codify-scan`** —
      `--apply-format` (after write, run `uv run ruff format` on the
      modified files) and `--apply-run-tests` (after the optional
      format step, run `uv run pytest tests/test_self_improve.py`).
      Both inert with `--apply-dry-run` (nothing landed, nothing to
      format or test) and inert when the per-site direction guard
      leaves every candidate site unchanged.  Non-zero subprocess rc
      propagates so a CI wrapper surfaces the failure.
- [x] **New module surface** — `scripts/self_improve.py._run_subprocess`
      indirection over `subprocess.run` so tests can monkeypatch a
      capture-only fake without shelling out to the real `uv` / `ruff`
      / `pytest` binaries.  Matches the queued `--open-pr` driver's
      dependency-injection pattern.  `_apply_top_codify_candidate`
      gains two keyword-only parameters (`run_format` / `run_tests`,
      both default `False`) so existing callers stay byte-identical.
- [x] **Trailing "Next: …" message adapts** — when
      `--apply-run-tests` succeeds, the trailing "Next: commit and
      open a draft PR" line drops the "run pytest" clause (already
      done).  Otherwise the pre-flag message is preserved verbatim,
      matching the 2026-06-30 driver's operator workflow.
- [x] **Why it improves Panobbgo** — three direct effects:
      (1) closes the "run ruff, then run pytest, then commit" gap
      in the §12.3 daily routine (one command replaces the previous
      three-step sequence); (2) prevents "landed but broke tests"
      codify PRs by gating the write behind the smoke-test suite;
      (3) advances the §11 success criteria without adding new arms
      (respects the §7.3 catalog freeze — pure operator-usability
      plumbing).
- [x] **Live-ledger smoke test** — `uv run python scripts/self_improve.py
      codify-scan --apply-top --apply-dry-run --apply-format
      --apply-run-tests` on the live ledger reports every candidate
      is skipped (1 structural + 3 bidirectional — correct outcome
      per the 2026-06-30 safety guards); because no edits landed,
      the two hygiene flags don't fire even though requested,
      matching the "inert when no site needed editing" contract.
- [x] **Tests** — 8 new tests in
      `tests/test_self_improve.py::TestApplyTopHygieneFlags` cover:
      `--apply-format` alone runs ruff on modified files;
      `--apply-run-tests` alone runs pytest + drops "run pytest"
      clause; both together run format-before-tests;
      `--apply-format` failure (rc=3) short-circuits so pytest is
      skipped; `--apply-run-tests` failure (rc=2) propagates after
      format succeeded; `--apply-dry-run` with both flags spawns
      zero subprocesses + reports "inert under --apply-dry-run";
      per-site-guard finds nothing to edit + both flags spawn zero
      subprocesses; argparse round-trip (default False, parses as
      True when passed).  Full `tests/test_self_improve.py` suite:
      541 → 549 tests (+8), all pass; `uv run pytest` (no ignores)
      reports 1762 passed / 11 skipped IOH workers; `ruff check` /
      `ruff format --check` / `pyright` clean.
- [x] **Documentation updated** — `planning/SELF_IMPROVEMENT_LOG.md`
      (2026-07-03 dated entry; the 2026-06-30 entry's
      `--apply-top --auto-format` / `--run-tests` follow-ups
      graduated from queued to shipped);
      `planning/SELF_IMPROVEMENT_LOOP.md` (§9.3 paragraph extended
      with the hygiene-flag mention + recommended-one-liner);
      `doc/source/guide_benchmarking.rst` (new *Hygiene flags*
      sub-block under *Apply the top candidate to the working tree
      (--apply-top)*); `doc/source/guide.rst` (Benchmarking
      summary line extended with the 2026-07-03 entry); `AGENTS.md`
      (new bullet under the V2 ship list + new CLI usage snippet at
      the bottom); `TODO.md` (this entry).
- [x] **Follow-ups** seeded under *Next iteration ideas* in
      `planning/SELF_IMPROVEMENT_LOG.md`: `--apply-open-pr` hygiene
      composition (once PR #275 lands the `--open-pr` driver, a
      single `--apply-format --apply-run-tests --open-pr` chain
      would run format + tests + `gh pr create` from one command);
      custom pytest scope (`--apply-run-tests-scope=STR`) so the
      operator can swap in a broader test path when the codify slot
      touches something outside `test_self_improve.py`.
### Structural-edit primitive for the `codify-scan --apply-top` driver (V2 §9.5 step 4 follow-up) — 2026-07-01
- [x] **Extended library surface in `panobbgo/self_improve.py`** —
      `_scan_source_for_structural_edits(source_path, *,
      factory_names, class_name, op, target_spec_names)` (sibling
      of `_scan_source_for_kwarg_edits` handling `add_/drop_`
      ops on the target `heuristics` / `analyzers` list literal),
      `_byte_to_lineno_col(byte_offset, line_starts)` helper for
      inverting the byte-offset ↔ (lineno, col_offset) mapping,
      and module-level `_STRUCTURAL_OPS_TO_BUCKET` mapping each op
      to its target bucket.  `derive_codify_edits` now dispatches
      structural candidates to the new scanner instead of returning
      an empty list; kwarg candidates route unchanged.
- [x] **Behaviour by op**: `drop_heuristic` / `drop_analyzer` emit
      one `CodifyEdit` per matching `(ClassName, {...})` tuple in
      the target bucket — the removal span covers the tuple plus
      trailing comma and inter-entry whitespace so the surviving
      literal is well-formatted; a corner-case backwards-expansion
      path applies to the "drop last entry of multi-line bucket"
      case so the closing `]` inherits the pre-entry indent.
      `add_heuristic` / `add_analyzer` emit a zero-width insertion
      just after the last entry's trailing comma; the new entry
      ships as `(ClassName, {})` (constructor defaults).  For an
      empty bucket (`analyzers=[]`) the add-primitive inserts inline
      (`analyzers=[(ClassName, {})]`).
- [x] **Three safety guards** keep the primitive conservative:
      (1) `drop_*` skips specs whose bucket has only one entry
      (else the surviving spec has no way to generate points);
      (2) `add_*` skips specs where the class is already in the
      bucket (matches `_structural_already_codified`);
      (3) `drop_*` skips specs where the class is not in the
      bucket (nothing to drop).  Plus a `target_spec_names` filter
      populated from the candidate's `strategy_names` restricts
      edits to the specs the ledger accumulated evidence against —
      unlike kwarg edits which safely propagate across every
      matching spec.
- [x] **CLI updates** in `scripts/self_improve.py` — the
      `--apply-top` handler no longer skips structural candidates
      with a "skipped N structural" note.  Instead it prints
      `selected: X [op]` and `target spec(s): ...` lines for
      structural picks, then delegates to `apply_codify_candidate`
      exactly like the kwarg branch.  Bidirectional-slot skip
      still applies to kwarg candidates only (structural directions
      are the op name and don't collide with the up/down heuristic).
- [x] **Idempotent re-runs** — a second `--apply-top` pass against
      the now-codified source derives an empty edit list because
      every safety guard (drop of missing class, add of present
      class) fires for the codified state.  Matches the
      self-stability shape of `_candidate_already_codified` /
      `_structural_already_codified`.
- [x] **Why it improves Panobbgo** — three direct effects:
      (1) **unblocks the live-ledger's top structural candidate**
      (`LatinHypercube` `drop_heuristic` from `Loop_LocalSearch`,
      `n_nights=2`, `mean_Δ=+0.0491`) — one night away from
      clearing the daily-routine threshold; the operator would
      have had to hand-remove the tuple with the 2026-06-30
      kwarg-only apply driver.  (2) **closes the structural
      codify gap in the daily routine (§12.3)** — every surfaced
      candidate (kwarg, categorical, structural) now translates
      to source edits via `codify-scan --apply-top` alone.
      (3) **advances the §11.2 throughput criterion** — structural
      codification lifts the kwarg-only cadence ceiling; analyzer
      add / heuristic drop candidates from the `--structural`
      mutation catalog can now translate to source edits directly.
- [x] **Tests** — 7 new tests in `tests/test_self_improve.py`:
      `TestApplyCodifyEdits` gains structural drop-missing-class /
      no-strategy-names / drop-actually-removes /
      add-actually-inserts / add-already-present / single-entry-
      bucket-guard / strategy_names-filter-honoured / structural-
      apply-idempotency; `TestApplyTopCLI` gains structural
      no-matching-site graceful-exit / drop-heuristic actually
      removes / add-analyzer actually inserts / drop-last-entry
      preserves closing bracket alignment.  The pre-existing
      `test_apply_top_skips_structural_with_note` and
      `test_derive_edits_structural_returns_empty_list` are
      replaced by the new-semantics versions.  Full suite: 551
      passed (was 544 before — net +7).  `ruff check` / `ruff
      format --check` clean.
- [x] **Documentation updated** — `planning/SELF_IMPROVEMENT_LOG.md`
      (2026-07-01 dated entry; *Next iteration ideas* seeded for
      `--open-pr` structural PR bodies, `add_heuristic` with
      recorded `structural_kwargs`, `--auto-format`, line-wrap
      heuristic), `planning/SELF_IMPROVEMENT_LOOP.md` (§9.3
      paragraph extended noting the structural source-edit layer
      is shipped), `doc/source/guide_benchmarking.rst`
      (*Apply the top candidate to the working tree
      (--apply-top)* sub-section extended with the structural-op
      behaviour + safety-guard rationale + strategy_names filter),
      `doc/source/guide.rst` (Benchmarking summary line extended
      with the 2026-07-01 entry), `AGENTS.md` (new bullet under
      the V2 ship list).
- [x] **Follow-ups** seeded under *Next iteration ideas* in
      `planning/SELF_IMPROVEMENT_LOG.md`:
      `--open-pr` structural PR body population using
      `strategy_names`; `add_heuristic` with recorded
      `structural_kwargs` when they converge across accepts;
      `--apply-top --auto-format` (run `uv run ruff format` on
      modified file); line-wrap heuristic for long constructor
      arguments in structural inserts.

### `codify-scan --apply-top` driver — mechanise the manual codify edit (V2 §9.5 step 4 plumbing) — 2026-06-30
- [x] **New library surface in `panobbgo/self_improve.py`** —
      `CodifyEdit` (frozen dataclass with AST coordinates + old /
      new source text), `derive_codify_edits(candidate, *, sources)`
      (AST-based scan: walks every named factory function in
      `sources`, finds every `(ClassName, {param_name: literal, ...})`
      heuristic / analyzer entry, returns a list of `CodifyEdit`),
      `apply_codify_edits(edits, *, dry_run)` (writes edits to disk
      in reverse byte-offset order so earlier edits don't invalidate
      later coordinates; `dry_run=True` returns the new file
      contents without writing), `apply_codify_candidate(candidate,
      *, sources, dry_run)` (convenience wrapper combining the two),
      and `default_codify_apply_sources()` (default
      `[("panobbgo/harness.py", (factory names ×4))]`).  All four
      added to `__all__`.
- [x] **New CLI flags on `scripts/self_improve.py codify-scan`** —
      `--apply-top` (after the report, apply the top actionable
      kwarg candidate), `--apply-dry-run` (preview the edits
      without writing), `--apply-include-bidirectional` (override
      the default skip-on-bidirectional safety guard).  The CLI
      dispatcher uses `getattr` so existing test invocations with
      hand-rolled NS namespaces continue to work byte-identically.
- [x] **Two safety guards** prevent the driver from shipping
      questionable changes:
      (1) **per-site direction guard** in `_should_apply_at_site`
      skips sites where the current value already sits at-or-beyond
      the proposal in the candidate's direction — so
      `BayesOpt_GP`'s deliberately-tighter `Nearby(radius=0.05)`
      is preserved when the consensus group shifts; (2)
      **bidirectional-slot skip** in the CLI's apply-top
      dispatcher (on by default) — if the same `(class_name,
      param_name)` slot appears with both `"up"` and `"down"`
      directions anywhere in the full candidate list (including
      already-codified ones), the candidate is skipped with a note
      pointing the operator at `--widen-bounds` for the catalog-
      update path.
- [x] **Idempotent re-runs** — a second `--apply-top` pass against
      the now-codified source derives an empty edit list because
      every matching site already satisfies the per-site direction
      guard.  Matches the self-stability invariant of
      `CodifyCandidate.proposed_codify_value` — applying the
      codified value as a live seed value satisfies
      `_candidate_already_codified` on the next scan, so the
      candidate is suppressed at the scan layer too.
- [x] **Live-ledger smoke test** — running `uv run python
      scripts/self_improve.py codify-scan --apply-top
      --apply-dry-run` against the live ledger today reports
      "skipped 1 structural candidate(s)" + "skipped 3
      bidirectional candidate(s)" + "every visible candidate was
      skipped — nothing to apply".  Exactly the correct outcome —
      the four visible candidates today are all either structural
      or bidirectional, so the driver refuses to ship a
      questionable change.
- [x] **Why it improves Panobbgo** — three direct effects:
      (1) **closes the manual-edit gap in the daily routine
      (§12.3)** — the four ledger-evidence-driven codify PRs to
      date each required the operator to hand-find every sibling
      spec literal, edit each one, re-format, and re-test (the
      2026-06-28 PR alone touched six sibling specs across four
      registry tiers); the driver mechanises that step to one
      command; (2) **unblocks the queued `--open-pr` driver
      (V2 §9.5 step 4)** by landing the source-edit primitive
      as a library function — the queued driver wraps the existing
      three layers (detection → value derivation → source editing)
      with a `gh pr create` call; (3) **advances the §11.2
      throughput criterion** — opening a fourth codify PR drops
      from ~30 minutes of careful manual editing to ~30 seconds
      of running one command + reviewing the diff.
- [x] **Tests** — 25 new tests in `tests/test_self_improve.py`:
      `TestApplyCodifyEdits` (18 tests) covering numeric / categorical
      / structural candidates, per-site direction guard (sites already
      at-or-beyond proposal are skipped), dry-run preserves source,
      idempotent re-apply, missing source / invalid Python / unknown
      factory return empty list gracefully, `to_dict` JSON round-trip,
      `default_codify_apply_sources` shape;  `TestApplyTopCLI`
      (7 tests) covering dry-run writes nothing, real apply writes
      the file, bidirectional skip on by default, override flag
      works, structural-only ledger skipped with note, no-candidates
      graceful exit, already-codified yields no edits.  Full suite
      passes (1754 + 25 = 1779 tests + 11 skipped IOH workers);
      `ruff check` / `ruff format --check` clean.
- [x] **Documentation updated** — `planning/SELF_IMPROVEMENT_LOG.md`
      (2026-06-30 dated entry; *Next iteration ideas* seeded for
      the structural-edit primitive, `--auto-format` / `--run-tests`
      hygiene flags), `planning/SELF_IMPROVEMENT_LOOP.md` (§9.3
      paragraph extended noting the source-edit layer is shipped),
      `doc/source/guide_benchmarking.rst` (new *Apply the top
      candidate to the working tree (--apply-top)* sub-section
      with the safety-guard rationale + the operator workflow),
      `doc/source/guide.rst` (Benchmarking summary line extended
      with the 2026-06-30 entry), `AGENTS.md` (new bullet under
      the V2 ship list).
- [x] **Follow-ups** seeded under *Next iteration ideas* in
      `planning/SELF_IMPROVEMENT_LOG.md`: structural-edit primitive
      for the apply driver (extends `derive_codify_edits` to
      support `add_/drop_heuristic` / `add_/drop_analyzer`
      candidates via list-entry insertion / removal — currently
      motivated by the live ledger's `LatinHypercube`
      `drop_heuristic` from `Loop_LocalSearch` candidate one
      night away from clearing the daily-routine threshold);
      `--apply-top --auto-format` flag (run `ruff format` after
      apply); `--apply-top --run-tests` flag (run pytest after
      apply); the full `--open-pr` driver itself (now reduced to
      a `gh pr create` wrapper around the existing primitives,
      plus dedup-against-open-PRs + branch naming).

### `CodifyCandidate.proposed_codify_value()` — codify-value derivation centralised on the dataclass (V2 §9.5 step 4 plumbing) — 2026-06-29
- [x] **New method `CodifyCandidate.proposed_codify_value(*, n_sig=3)`**
      in `panobbgo/self_improve.py` that computes the seed value a
      codify edit would ship.  Per-rule-kind branching: numeric
      `direction="up"`/`"down"` → median of `new_values` rounded
      *outward* in `direction` to `n_sig` significant digits (floats)
      or `math.ceil` / `math.floor` (`integer_add`); categorical →
      the chosen literal verbatim (preserving Python type — `False`
      stays `False`, not `"False"`); structural ops → `None`
      (caller consults `class_name` + `op` directly).
- [x] **New helper `_round_outward_to_significant(value, direction,
      n_sig)`** centralises the float rounding policy.  Handles
      `value == 0.0` (returned unchanged), non-finite inputs (passed
      through), and negative values (the abs-rounding direction is
      inverted so the result still moves in the linear direction —
      `"up"` on `-0.5` returns a less-negative value).  Invalid
      directions raise `ValueError`.
- [x] **`CodifyCandidate.to_dict()` carries the new
      `proposed_codify_value` field** so JSON-mode consumers
      (`codify-scan --json`) and the queued `--open-pr` driver
      share one source of truth for "what value should the codify
      edit ship?".  Value preserves its Python type
      (bool / int / float / None) under `_to_plain`.
- [x] **`_print_codify_candidate` in `scripts/self_improve.py`**
      now emits a `proposed codify value:` line on every actionable
      candidate (suppressed automatically when the value is `None`
      — structural ops and edge cases).  Float formatting uses
      `{value:.6g}` to match the existing `_format_old_new`
      style; bools render as `repr` so `False` reads as `False`
      not `0`.
- [x] **Live ledger verification** — `uv run python
      scripts/self_improve.py codify-scan` surfaces four actionable
      candidates with the proposed values:
      `Nearby.radius` direction=up → `0.124` (matches PR #271
      exactly), `Sobol.n` direction=down → `12` (matches the
      deferred-codify note for the manual companion),
      `Nearby.radius` direction=down → `0.0809`, `Sobol.n`
      direction=up → `22`.
- [x] **Why it improves Panobbgo** — three direct effects:
      (1) closes the manual computation gap (every prior codify
      PR had to hand-compute the median and pick a rounding
      step); (2) unblocks the queued `--open-pr` driver (V2 §9.5
      step 4) by centralising the core "what value to ship?"
      question on the dataclass; (3) the self-stability invariant
      (`proposed >= median` for `up` / `proposed <= median` for
      `down`) ensures the codified value cleanly suppresses the
      candidate on the next scan so the future driver cannot
      re-open the same PR every night.
- [x] **Tests** — 19 new tests in
      `tests/test_self_improve.py::TestProposedCodifyValue` covering:
      direct unit tests of `_round_outward_to_significant` (PR #271
      round-trip, down-rounding, zero / negative / non-finite /
      invalid-direction handling); end-to-end on numeric `up` / `down`
      / `integer_add up` / `integer_add down` (verifies the
      float/int type of the returned value); categorical preserves
      boolean type; structural returns `None`; empty `new_values`
      returns `None`; self-stability invariant
      (`_candidate_already_codified(c, [proposed]) is True`) for
      `up` / `down` / `integer_add`; `to_dict()` carries the new
      field, JSON-serialisable, boolean preserved.  Plus assertion
      lines added to `TestCodifyScanCLI.test_realistic_two_night_pattern_surfaces_candidate`
      (verifies the `proposed codify value: 0.125` line in the
      report) and
      `TestCodifyScanCLI.test_json_mode_emits_one_object_per_candidate`
      (verifies the JSON payload carries
      `proposed_codify_value: 0.125`).  All 19 new tests pass;
      full `test_self_improve.py` suite (516 tests) clean;
      `ruff check` / `ruff format --check` / `pyright` all clean.
- [x] **Documentation updated** — `planning/SELF_IMPROVEMENT_LOG.md`
      (2026-06-29 dated entry; *Follow-up ideas* seeded for the
      `--apply-top` driver, `--open-pr` driver, log-space rounding
      refinement), `planning/SELF_IMPROVEMENT_LOOP.md` (§9.3
      `--open-pr` paragraph extended with a bullet pointing at the
      new method and its self-stability invariant),
      `doc/source/guide_benchmarking.rst` (the codify-scan
      sub-section documents the `proposed codify value:` line, the
      JSON field, and the `proposed_codify_value` method),
      `doc/source/guide.rst` (Benchmarking summary line extended
      with the 2026-06-29 entry), `AGENTS.md` (new bullet under the
      V2 ship list).
- [x] **Follow-ups** seeded under *Next iteration ideas* in
      `planning/SELF_IMPROVEMENT_LOG.md`: `codify-scan --apply-top`
      working-tree-edit driver (mechanises the next-most-tedious
      manual step), `codify-scan --open-pr` PR-creation driver (the
      full V2 §9.5 step 4 closure), log-space rounding for
      `log_uniform_perturb` rules (speculative).

### Codify auto-tuned `Nearby.radius` catalog tightening (manual widening-detector codify) — 2026-06-26
- [x] **Catalog edit** in `panobbgo.self_improve.default_catalog`:
      `Nearby.radius` `MutationRule.bounds` tightened from
      `(0.005, 0.5)` to `(0.032, 0.313)` (the 2026-06-22 auto-tuned
      proposal).  Pure bound update — no new arms, no constructor
      changes, no behaviour change for the `Nearby` heuristic itself.
      First widening-detector output to land as a catalog change.
- [x] **Evidence base** — 13 accepts on the
      `(Nearby, radius, log_uniform_perturb)` arm across 9 distinct
      nights from 2026-05-26 to 2026-06-18.  Every accepted
      `new_value` falls inside the observed window
      `[0.073, 0.135]`; the pre-tightening catalog bounds
      `[0.005, 0.5]` admit values 6.25× below and 1.6× above that
      window.  Auto-tuned widening detector (2026-06-22) recommends
      `[0.0317, 0.3130]` ≈ `[0.032, 0.313]` (~2.31× headroom factor
      around the observed range — wide enough to keep exploration
      headroom on either side, narrow enough that every per-iteration
      pull lands in the productive region).
- [x] **Why it improves Panobbgo** — concentrating the bandit's
      `(Nearby, radius, *)` proposals onto the productive window
      reduces wasted no-op pulls per night (the §11.1 "resolution"
      criterion) and tightens the per-arm Beta posterior on the same
      compute.  Every observed `new_value` from the live ledger sits
      comfortably inside the new bounds, so the bandit's
      accepted-region knowledge survives the change.
- [x] **Self-stabilising** — re-running `codify-scan --widen-bounds
      --widen-auto-tune` against the same ledger after the codify
      shows the auto-tune converges on `[0.0345, 0.287]` (the
      now-narrower catalog yields a smaller spread-ratio so the
      per-candidate factor settles near 2.12 instead of 2.31), which
      sits effectively at the new bounds — the detector won't
      oscillate.
- [x] **Test updates** (assertion-only, 4 tests, no logic change):
      `TestCatalogNumericBounds.test_finds_existing_rule` and
      `TestDetectWideningCandidates.test_looks_up_current_bounds_from_default_catalog`
      now expect `(0.032, 0.313)`;
      `TestDetectWideningCandidatesAutoTune.test_auto_tune_sizes_factor_per_candidate`
      and the JSON-mode CLI sibling
      `TestCodifyScanCLIAutoTuneWidening.test_auto_tune_json_mode_emits_per_candidate_factor`
      relaxed from `2.2 < factor < 2.5` to `2.0 < factor < 2.3` (the
      tighter catalog yields a larger spread-ratio so the
      per-candidate factor sits near 2.12 instead of 2.31).  The
      custom-range siblings of the same two tests relaxed
      symmetrically from `> 3.5` to `> 3.0`.  All 1681 pre-existing
      tests pass; `ruff check` / `ruff format` / `pyright` clean.
- [x] **Documentation updated** — `planning/SELF_IMPROVEMENT_LOG.md`
      (2026-06-26 dated entry; the *Manual codify of the widening
      proposal* follow-up promoted from speculative to shipped),
      `planning/SELF_IMPROVEMENT_LOOP.md` (§9.3 widening-detector
      sub-paragraph extended with the codify note and the
      self-stabilising observation),
      `doc/source/guide_benchmarking.rst` (the *Bidirectional-bound
      widening* sub-section's live-evidence bullet for
      `Nearby.radius` extended with the "manually codified" tag and
      the auto-tune sub-subsection's live-ledger bullet now points
      back to the codify note), `AGENTS.md` (new self-improvement
      loop bullet for the manual codify).
- [x] **Follow-ups** seeded under *Next iteration ideas* in
      `planning/SELF_IMPROVEMENT_LOG.md`: `codify-scan
      --widen-bounds --open-pr` driver (the automation layer) and a
      manual companion for `Sobol.n` (deferred because the
      auto-tune classifies it as `"widens current"` — mixed signal).

### Auto-tune κ for hierarchical structural bandit — 2026-06-25
- [x] **New `AdaptiveMutationSampler.structural_borrow_horizon`
      parameter** (`float ≥ 0`, default `0.0`).  When `> 0` and the
      two borrow preconditions are met (`structural_borrow_alpha > 0`
      and `per_class_structural = True`), each per-class arm's
      effective borrow shrinks toward zero as its own attempts
      accumulate: `κ_eff = κ / (1 + n_class_attempts / h)`.  Cold
      arms borrow the full configured `κ`; at `n_class_attempts = h`
      the borrow halves exactly; saturated arms effectively trust
      the leaf posterior.
- [x] **New helper `AdaptiveMutationSampler._effective_borrow`**
      centralises the annealing math so the sample-path code and
      tests consult the same rule.  Returns the configured `κ`
      unchanged whenever `h = 0`, `κ = 0`, or the arm has no
      attempts (cold-start case).
- [x] **`LoopConfig.structural_borrow_horizon` field + validation**
      mirrors the constructor kwarg with the same default (`0.0` →
      disabled).  `__post_init__` raises `ValueError` on negative
      values.
- [x] **CLI surface on `scripts/self_improve.py run`:
      `--structural-borrow-horizon`** (default `0.0`).  Off by
      default so existing invocations stay byte-identical.
- [x] **Tests** — 16 new tests in the new
      `tests/test_self_improve.py::TestStructuralBorrowAnneal` class
      covering: default constructor, validation paths (negative /
      non-finite), helper math (`h = 0` returns `κ`; `κ = 0` returns
      `0`; cold arm returns full `κ`; halved at `n = h`; vanishes at
      saturation; monotonic decreasing), backwards-compat sampling
      trajectory (`h = 0` byte-identical), cold-sibling still gets
      full borrow with annealing on, saturated-arm uses the reduced
      `κ_eff` (verified by parsing the rationale `Beta(α, β)`
      string), inert without per-class arms, and LoopConfig
      integration (default / validation / propagation through
      `SelfImprover`).
- [x] **Backwards compatibility** — strictly safe.  Default
      `h = 0` keeps every existing invocation byte-identical (the
      annealing path is taken only when the knob is explicitly set
      to a positive value).  Ledger replay: the bandit's arm key
      `(class_name, op, "structural")` is unchanged, so existing
      archives replay onto the same per-class arms regardless of
      whether the consumer enables the annealing knob.  Full
      project test suite (1697 tests) green; `uv run ruff check` /
      `uv run ruff format --check` / `uv run pyright` all clean.
- [x] **Documentation updated** — `planning/SELF_IMPROVEMENT_LOG.md`
      (2026-06-25 dated entry; *Auto-tune ``κ``* idea promoted from
      "Next iteration ideas" to shipped),
      `doc/source/guide_benchmarking.rst` (new "Auto-tune ``κ``
      from observed evidence (``structural_borrow_horizon``)"
      sub-section), `panobbgo/self_improve.py`
      (`AdaptiveMutationSampler` + `LoopConfig` docstrings extended
      for the new parameter and the annealing rule).

### Named regimes for `LSHADE.F_schedule` (categorical broadening) — 2026-06-23
- [x] **New module-level dict
      `panobbgo.heuristics.lshade._F_SCHEDULE_REGIMES`** — maps each
      named regime to a `(phase1_bound, phase2_bound, phase1_cap,
      phase2_cap)` 4-tuple.  Three regimes ship: `"jso"` (Brest et al.
      2017 — `0.6 / 0.9` breakpoints, `0.7 / 0.8` caps), `"early"`
      (kicks in earlier and tighter — `0.4 / 0.7` breakpoints, `0.6 /
      0.8` caps), `"strict"` (most aggressive — `0.5 / 0.85`
      breakpoints, `0.5 / 0.7` caps).  `"off"` collapses onto `None`
      (cap disabled).
- [x] **New helper
      `panobbgo.heuristics.lshade._normalize_F_schedule`** validates
      the constructor argument and maps the legacy bool inputs onto
      the new strings (`True` → `"jso"`, `False` → `"off"` → `None`)
      so ledger replay against the binary toggle shipped 2026-05-21
      and any spec that still passes the boolean form keep working.
- [x] **`LSHADE._apply_F_cap` rewritten** to look up the per-regime
      tuple from `_F_SCHEDULE_REGIMES[self.F_schedule]` instead of
      branching on hard-coded module-level constants.  The canonical
      Brest 2017 constants (`_F_SCHEDULE_PHASE1_BOUND` etc.) stay as
      module-level aliases for the `"jso"` regime tuple so any
      external introspection code that references them by name keeps
      working.
- [x] **`default_catalog` rule broadened** — the `LSHADE.F_schedule`
      `categorical_choice` rule's `choices` flip from `(True, False)`
      to `("off", "jso", "early", "strict")`.  The bandit arm key
      `(LSHADE, F_schedule, categorical_choice)` is unchanged so the
      pre-2026-06-23 Beta posterior accumulates seamlessly across the
      regime broadening — only the proposed value vocabulary expands.
- [x] **JSO updated** — `panobbgo.heuristics.jso.JSO.__init__` now
      passes `F_schedule="jso"` (canonical) instead of
      `F_schedule=True` (legacy synonym).  Behaviour byte-identical.
- [x] **Tests** — 4 new tests in `tests/test_heuristic_lshade.py`
      (`test_apply_F_cap_early_regime` /
      `test_apply_F_cap_strict_regime` /
      `test_apply_F_cap_regime_dict_is_complete` /
      `test_custom_F_schedule_construction_named_regimes`).  Existing
      `_apply_F_cap` tests switched from `F_schedule=True` / `False`
      to the equivalent `"jso"` / `"off"` strings; behaviour
      byte-identical.  Tests in `test_heuristic_jso.py`,
      `test_heuristic_nl_shade_rsp.py`, `test_heuristic_nl_shade_lbc.py`
      that asserted `h.F_schedule is True` updated to assert
      `h.F_schedule == "jso"`.  Full project test suite
      (1681 tests) green.
- [x] **Backwards compatibility** — strictly safe.  Default
      `F_schedule=None` (cap disabled) unchanged.  Legacy bool inputs
      still accepted by the constructor (normalized).  Ledger replay
      against the prior categorical (`(True, False)`) still works —
      bandit arm key is value-independent, and the constructor
      accepts both bool values.
- [x] **Documentation updated** — `planning/SELF_IMPROVEMENT_LOG.md`
      (2026-06-23 dated entry; *Categorical regimes for
      `LSHADE.F_schedule`* idea promoted from "Next iteration ideas"
      to shipped), `panobbgo/heuristics/lshade.py` (module docstring
      + `F_schedule` constructor docstring + `_apply_F_cap`
      docstring), `panobbgo/heuristics/jso.py` (docstring +
      `F_schedule="jso"` call site), `panobbgo/heuristics/lshade_ep_sin.py`
      (docstring reference), `doc/source/guide_benchmarking.rst`
      (categorical-rules bullet + example `MutationRule` literal +
      L-SHADE prose paragraph), `AGENTS.md` (categorical-rules
      bullet).

### Auto-tune widen factor from observed spread (V2 §9.3 follow-up) — 2026-06-22
- [x] **New helper `panobbgo.self_improve._auto_tune_widen_factor`**
      sizes a widen factor from the ratio of observed spread to
      catalog-bound span.  Narrow observed spread (high agreement
      across nights) → larger factor for exploration headroom; wide
      spread (low agreement) → smaller factor focused on the
      consensus.  Spread is measured in the rule's natural scale:
      log-space ratio for `log_uniform_perturb`, linear ratio for
      `integer_add` / `float_uniform`.  Linear interpolation between
      `max_factor` at ratio = 0 and `min_factor` at ratio = 1; falls
      back to a caller-supplied `fallback` when no catalog rule
      targets the slot (relative-spread signal unavailable).
- [x] **`detect_widening_candidates` gains three keyword arguments**:
      `auto_tune: bool = False`, `auto_tune_min_factor: float = 1.1`,
      `auto_tune_max_factor: float = 2.5`.  When `auto_tune=True` the
      per-candidate factor lands in `WideningCandidate.widen_factor`
      so the report and JSON output show the actually-used factor.
      Default `auto_tune=False` keeps every existing invocation
      byte-identical.
- [x] **CLI surface on `scripts/self_improve.py codify-scan`**:
      `--widen-auto-tune` (off by default), `--widen-factor-min`
      (default 1.1), `--widen-factor-max` (default 2.5).  The
      pre-existing `--widen-factor` (default 1.5) is repurposed as
      the fallback for slots with no catalog rule.  The
      *Bound-widening candidates* report header switches from
      `widen_factor=1.5` to `widen_factor=auto-tune [1.1, 2.5]
      (fallback=1.5)` when auto-tune is on.
- [x] **Tests** — 22 new tests across three new test classes
      (`TestAutoTuneWidenFactor` + `TestDetectWideningCandidatesAutoTune`
      + `TestCodifyScanCLIAutoTuneWidening`), all 38 prior widening
      tests pass unchanged.  Full project test suite (1653 tests)
      green; sphinx doctests / ruff / pyright clean.
- [x] **Live-ledger effect** — on the current project ledger
      (`planning/self_improve_ledger.jsonl`) the auto-tuned factor
      lifts `Nearby.radius` from a fixed 1.5 to ~2.31 (proposed bound
      widens from `[0.049, 0.203]` to `[0.032, 0.313]`) and
      `Sobol.n` from 1.5 to ~2.13 (proposed bound flips from
      tightening `[5, 36]` to widening `[3, 52]`).  Both proposals
      use the same ledger evidence the operator was triaging before
      this ship — auto-tune doesn't change the input, just the
      bound-arithmetic.
- [x] **Documentation updated** — `planning/SELF_IMPROVEMENT_LOG.md`
      (2026-06-22 dated entry; *Auto-tune widen factor from observed
      spread* idea promoted from "Next iteration ideas" to shipped),
      `planning/SELF_IMPROVEMENT_LOOP.md` (§9.3 widening-detector
      sub-paragraph extended with the auto-tune lever),
      `doc/source/guide.rst` (quick-nav entry),
      `doc/source/guide_benchmarking.rst` (new
      *Auto-tuned widen factor (--widen-auto-tune)*
      sub-subsection in the *Bidirectional-bound widening*
      subsection), `AGENTS.md` (self-improvement loop bullet +
      new bash example).

### Flip the nightly cron to the V2 substrate (V2 §9.5 step 5) — 2026-06-21
- [x] **Workflow edit** (`.github/workflows/self_improve_nightly.yml`).
      Single-file change to the `Run self-improvement loop` step that
      promotes every zero-cost V2 flag from the planning doc's §9.4
      target invocation into the live cron.  Flips:
      - `--registry loop` (catalog kwarg-rule activation 4/44 → 44/44 on
        the seed specs — closes V2 §2.4 "catalog ≫ registry mismatch").
      - `--prime-include-archives` (bandit posterior compounds across
        rotated ledger archives under `planning/done/` rather than
        forgetting every pre-rotation observation — closes V2 §2.6
        second half in the live cron).
      - `--structural-per-class-arms` (one Thompson arm per (op,
        candidate class), splitting `add_heuristic` into `add_Sobol`
        / `add_Random` / … so the bandit can learn per-class winners).
      - `--bandit-reward graded` (continuous `[0, 1]` reward derived
        from the bootstrap CI / point delta — turns the ~2.5% binary
        information yield into ~65% per the §7.4 lift estimate).
      - `--inactivity-relax-after 10 --inactivity-relax-factor 0.5`
        (geometric eps_accept relaxation per the docstring
        recommendation for the unattended cron).
      - `--holdout-base-seeds 7,1234` (multi-seed hold-out with
        worst-case drift / any-overfit reduction, replacing the
        single-seed `--holdout-base-seed 7`).
      - `--guard-interval 10` (relaxed from 5; matches the §9.4
        target invocation now that the same-night confirm gate is the
        primary noise-spike defence in the V2 architecture even
        though the gate itself is *not* flipped in this PR).
- [x] **Intentional hold-back: `--confirm-accepts`** — the only V2
      flag with a meaningful per-iteration cost (2-3× screening cost
      plus 1× per hold-out seed).  Queued for a manual
      `workflow_dispatch` A/B that measures the confirm-reject rate
      before flipping the cron permanently.  Documented in
      `planning/SELF_IMPROVEMENT_LOG.md` as the *Flip the nightly
      cron to `--confirm-accepts`* follow-up.
- [x] **Intentional hold-back: `--metric aocc`** — §9.5 step 2,
      needs the IOH worker available on the GitHub-hosted runner.
      The current cron stays on `composite_score`.
- [x] **Smoke tests**: two 1-iteration runs against the new
      invocation — fresh ledger (exit 0; 2 hold-out records;
      `worst_drift=+0.0028 overfit=0/2 vacuous=0/2`) and primed from
      the live ledger (exit 0; per-class arms correctly populated
      from legacy collapsed op-level records:
      `Nearby.radius[log_uniform_perturb] -> 6/79 (8%)`,
      `Sensitivity.drop_analyzer[structural] -> 0/29 (0%)`,
      `Restart.add_analyzer[structural] -> 1/23 (4%)`).
- [x] **Backwards compatibility**: strictly safe.  Pure workflow
      edit; no code changes, no test changes, no API changes.
      Existing ledger entries remain valid priors under the new
      invocation (the bandit's `_proposal_rule_key` collapses to
      `(class_name, param_name, rule_kind, ...)` independent of the
      strategy / spec name, structural arm split, or reward shape).
      No ledger rotation needed.
- [x] **Documentation**: dated entry in
      `planning/SELF_IMPROVEMENT_LOG.md`; planning doc §9.5 step 5
      flipped from "open" to "partially shipped"; §2.2 / §2.4 / §2.6
      diagnoses annotated with the partial closure;
      `doc/source/guide_benchmarking.rst` "Live nightly cron" callout
      added under the same-night confirmation gate section;
      `AGENTS.md` self-improvement loop bullet annotated; this TODO
      entry.

### Mutation-bound widening detection for bidirectional codify candidates (V2 §9.3 follow-up) — 2026-06-19
- [x] **New `panobbgo.self_improve.WideningCandidate` dataclass**
      carrying one bidirectional pair: `class_name` / `param_name` /
      `rule_kind`, the catalog's current `bounds` (or `None` when no
      rule targets the slot), the observed range pooled across both
      `"up"` and `"down"` directions, the proposed widened bounds, the
      widen factor used, and the two contributing `CodifyCandidate`
      instances (the up and down flavors).  `proposal_is_wider` /
      `proposal_is_tighter` flags label the proposal direction;
      `slot_key` mirrors :attr:`CodifyCandidate.slot_key` so the
      queued `--open-pr` driver can dedup uniformly across both
      candidate kinds.
- [x] **New `detect_widening_candidates(candidates, *, catalog=None,
      widen_factor=1.5)` function** in `panobbgo.self_improve`.  Walks
      a sequence of :class:`CodifyCandidate` instances, drops
      structural / categorical / single-direction entries, groups by
      `(class_name, param_name, rule_kind)`, and emits one
      :class:`WideningCandidate` per group with both directions
      represented.  Sorted by strongest evidence first
      (`n_distinct_nights desc, n_accepts desc, class_name asc`).
- [x] **Per-rule-kind widening arithmetic** in
      :func:`_widen_numeric_bounds` (private):
      - `log_uniform_perturb` — multiplicative on both ends, floored
        at `1e-12` (the rule rejects non-positive values).
      - `integer_add` — multiplicative + outward rounding
        (:func:`math.floor` on the lower bound, :func:`math.ceil`
        on the upper).  Lower bound clipped to `1` when observed
        values are positive (most integer-typed kwargs are pool sizes
        / iteration counts).
      - `float_uniform` — multiplicative on absolute values, sign
        preserved.  `observed_lo == 0` is preserved at zero.
- [x] **CLI flags on `scripts/self_improve.py codify-scan`**:
      `--widen-bounds` (off by default; appends a *Bound-widening
      candidates* section after the existing codify report) and
      `--widen-factor FLOAT` (default `1.5`).  JSON mode emits each
      widening candidate on its own line tagged
      `"_type": "widening_candidate"`; codify candidates carry the
      symmetric `"_type": "codify_candidate"` tag (additive on the
      existing JSON schema).
- [x] **Per-pair tag in the text report** — `[widens current]` /
      `[tightens current — focuses bandit on observed range]` /
      `[partial overlap]` / `(no rule)` (when no numeric rule targets
      the slot) — so the operator can prioritise at a glance.
- [x] **Live-ledger evidence on the day of ship** — the detector
      surfaces two bidirectional patterns:
      - `Nearby.radius`: observed `[0.073, 0.135]`, current
        `[0.005, 0.5]`, proposed `[0.049, 0.203]` — *tightens
        current*.  The bandit consistently picks values in a window
        5-10× narrower than the catalog admits.
      - `Sobol.n`: observed `[8, 24]`, current `[4, 64]`, proposed
        `[5, 36]` — *tightens current*, same shape.
- [x] **Tests** — 38 new tests across three test classes in
      `tests/test_self_improve.py`:
      - `TestWidenNumericBounds` (10): per-rule-kind bound
        arithmetic, edge cases (tiny positive floor, integer lower
        bound clipping, observed-zero preservation), `widen_factor`
        validation, unsupported rule kind rejection.
      - `TestCatalogNumericBounds` (4): catalog lookup correctness,
        unknown slot returns None, dual-rule slots
        (`NLSHADE_RSP.k_rank`'s `float_uniform` + `categorical_choice`),
        integer rule returns float bounds.
      - `TestDetectWideningCandidates` (17): pairing semantics,
        same-slot vs different-slot grouping, structural / categorical
        skipped, custom catalog override, sort order by evidence
        strength, date deduping, slot_key shape, JSON round-trip.
      - `TestCodifyScanCLIWidening` (5): end-to-end CLI smoke tests
        for the new flags.
      Test totals: 449 in `tests/test_self_improve.py` (411 before +
      38 new); 1645 in `tests/` (11 skipped — unrelated IOH worker
      setup).
- [x] **Backwards compatibility** — strictly safe.  Pure additions
      to `panobbgo/self_improve.py` (one dataclass + one public
      function + two private helpers) and two new CLI flags on the
      existing `codify-scan` subcommand.  Existing invocations
      (without `--widen-bounds`) produce byte-identical output.  The
      JSON-mode schema gains a new `"_type"` field on every emitted
      record but the field is additive — consumers that don't filter
      on it see the same record bodies as before.
- [x] **Documentation updated** — `planning/SELF_IMPROVEMENT_LOG.md`
      (2026-06-19 dated entry; *Mutation-bound widening rule* idea
      promoted from "Next iteration ideas" to shipped),
      `planning/SELF_IMPROVEMENT_LOOP.md` (§9.3 widening-detector
      sub-paragraph), `doc/source/guide.rst` (quick-nav entry),
      `doc/source/guide_benchmarking.rst` (new
      *Bidirectional-bound widening (--widen-bounds)*
      sub-subsection in the *Cross-night codify-scan* section),
      `AGENTS.md` (self-improvement loop bullet + bash examples).
### Structural-op already-codified check (V2 §9.3 follow-up) — 2026-06-19
- [x] **`_live_class_membership` helper** (`panobbgo/self_improve.py`):
      new private helper that walks the seed-spec factories and
      records, for the candidate's class, which spec names already
      list it under `heuristics` / `analyzers`.  Mirrors the
      resilience of `_live_kwarg_values` — factories that raise
      are silently skipped so a misbehaving caller-supplied
      factory cannot break the codify-scan run.
- [x] **`_structural_already_codified` real implementation** —
      replaces the placeholder that always returned `False` with
      the symmetric class-membership predicate:
      `add_heuristic` / `add_analyzer` codify iff at least one
      seed spec already lists the class in the matching bucket;
      `drop_heuristic` / `drop_analyzer` codify iff no spec
      lists it.  Unknown / future op names fall back to "not
      codified" so the candidate continues to surface.
- [x] **`annotate_codified_status` branches by candidate shape** —
      structural candidates route through the new membership
      path; kwarg candidates take the existing
      `_live_kwarg_values` / `_candidate_already_codified` path
      unchanged.  The structural `live_codified_values` field
      now surfaces the *spec names* carrying the class so the
      `--include-already-codified` audit trail still tells the
      operator where the membership lives.
- [x] **Dead-code cleanup in `_candidate_already_codified`** —
      removed the unreachable structural branch that called the
      placeholder (live_values was always empty for structural
      candidates so the earlier `not live_values` guard already
      fired); replaced with a defensive `return False` so a
      mis-routed structural call cannot silently mis-classify.
- [x] **Tests** — 8 net new tests in
      `tests/test_self_improve.py::TestAnnotateCodifiedStatus`
      covering every structural op (`add_heuristic`,
      `drop_heuristic`, `add_analyzer`, `drop_analyzer`) in
      both directions, the heuristic-vs-analyzer bucket distinction,
      the multi-spec membership recording, and the defensive
      unknown-op fallback.  Plus 2 new end-to-end CLI smoke tests
      in `TestCodifyScanCLISuppression`
      (`test_structural_add_heuristic_suppressed_when_already_in_pool`,
      `test_structural_drop_heuristic_surfaces_when_class_in_pool`)
      exercising the suppression behaviour against synthetic
      ledger records.  All 421 self-improve tests pass; the
      live ledger's surfaced-candidate count is byte-identical
      (4 of 5, with the same `Sobol.scramble` candidate hidden).
- [x] **Docs**: `planning/SELF_IMPROVEMENT_LOG.md` dated entry +
      promoted *Structural-op codified check* follow-up from
      queued to shipped; new *Membership-vs-coverage rule for
      structural ops* idea seeded under "Next iteration ideas";
      `doc/source/guide.rst` quick-nav entry extended;
      `doc/source/guide_benchmarking.rst` codify-scan subsection
      describes the structural predicate alongside the kwarg
      rule; `AGENTS.md` self-improvement loop bullet annotated;
      `TODO.md` this entry.

### No-op detection on bandit-pull and ledger telemetry (V2 §12.4) — 2026-06-12
- [x] **`LoopIterationRecord.no_op` field** (`panobbgo/self_improve.py`):
      new boolean field (default `False`), serialised via
      `to_dict`.  Set to `True` when the candidate's per-(problem,
      strategy) scores are bit-identical to baseline — the
      `_is_no_op` helper compares the `problem_strategy_results.score`
      maps directly.  Iterations flagged as no-op also record
      `reason_skipped="no_op"`, `accepted=False`, and an extra
      "no-op" marker in the reasons list.
- [x] **`AdaptiveMutationSampler.discard_outcome`** — new helper
      that clears `last_rule_key` without incrementing
      `n_attempts`.  The driver loop calls this instead of
      `record_outcome` on no-op iterations so the bandit's
      Beta posterior is not pulled on a zero-information event.
- [x] **`prime_from_ledger` skips no-op records** — records
      flagged with `no_op=True` are excluded from priming, with
      legacy records (no `no_op` key on disk) continuing to
      replay byte-identically as before.
- [x] **CLI summary surfaces `no-op=N` bucket** —
      `scripts/self_improve.py run` end-of-run line and the
      `summary` subcommand's `Iterations:` header both report a
      separate no-op count; the accept rate denominator switches
      to **informative** (decided − no-op) so dormant rules
      cannot artificially deflate it.
- [x] **§2.1 diagnosis closure** — the V1 "34% of mutations
      measure Δ = exactly 0.0000" pattern no longer mis-trains
      the bandit posterior; an operator reading the §12.3 daily
      routine can now distinguish "bandit starved on dormant
      rules" from "every legitimate proposal got rejected".
- [x] **Tests** — 10 new tests in
      `tests/test_self_improve.py::TestNoOpDetection`:
      - `test_default_no_op_field_is_false`
      - `test_identical_pair_scores_flag_no_op`
      - `test_distinct_pair_scores_are_not_no_op`
      - `test_no_op_iteration_does_not_pull_bandit`
      - `test_no_op_iteration_increments_streak`
      - `test_prime_from_ledger_skips_no_op_records`
      - `test_prime_from_ledger_legacy_record_replays`
      - `test_discard_outcome_clears_pending_arm`
      - `test_no_op_round_trips_through_ledger`
      - `test_cli_summary_surfaces_no_op_count`
      Plus the existing
      `TestSelfImproverAdaptive::test_adaptive_sampler_records_rejects`
      fixture updated to use distinct baseline/candidate scores
      (it previously relied on the now-detected-as-no-op
      constant-score path).  All 1450 tests pass.
- [x] **Docs**: `planning/SELF_IMPROVEMENT_LOOP.md` §2.1
      annotated, §9.5 step 3 marks the no-op sub-task shipped,
      §12.4 first bullet promoted from open → shipped;
      `planning/SELF_IMPROVEMENT_LOG.md` dated entry + new
      "Next iteration ideas" entry seeded for the *pre-measure
      no-op short-circuit* compute-saving follow-up;
      `doc/source/guide.rst` quick-nav entry;
      `doc/source/guide_benchmarking.rst` new "No-op detection
      (§12.4)" subsection; `AGENTS.md` self-improvement loop
      bullet.

### Categorical `JSO.p_best_max` rule (literature regimes) — 2026-06-09
- [x] **New `categorical_choice` `MutationRule` entry on
      `default_catalog`** (`panobbgo/self_improve.py`):
      `(JSO, p_best_max)` with `choices=(0.15, 0.25, 0.4)` —
      three literature-canonical jSO ``p_best_max`` regimes
      (L-SHADE-like / Brest et al. 2017 jSO default /
      iLSHADE-like).  Probability `0.3` (matches all other
      categorical rules).  Closes the *Categorical mutation rule
      for ``JSO.p_best_max``* next-iteration ticket under *jSO
      follow-ups (after 2026-05-15 ship)*.
- [x] **0.11 → 0.15 substitution** — the L-SHADE-canonical
      ``p_best = 0.11`` lies below jSO's default
      ``p_best_min = 0.125`` and would trip the constructor
      invariant ``p_best_min <= p_best_max``.  Raising to ``0.15``
      preserves the "greedy-regime" semantics (still meaningfully
      narrower than the jSO ``0.25`` default) without requiring a
      coordinated rule that lowers ``p_best_min`` alongside.  The
      planning doc flags the *categorical-with-dependent-kwarg*
      pattern as a future generalisation when motivated by a
      second slot too.
- [x] **Dual-rule shape on the same slot** — the categorical
      rule sits alongside the existing `float_uniform` rule on
      the same `(JSO, p_best_max)` slot (shipped 2026-05-15 with
      the JSO ship).  The two live on distinct bandit arms by
      construction (`_proposal_rule_key` includes `rule_kind`),
      so the bandit can either continuously walk the value or
      jump between regimes — the same pattern that
      `NLSHADE_RSP.k_rank` already uses.
- [x] **Tests** — 4 new tests in
      `tests/test_heuristic_jso.py::JSORegistrationTests`:
      - `test_kwarg_catalog_jso_p_best_max_has_both_kinds`
      - `test_kwarg_catalog_jso_p_best_max_categorical_choices`
      - `test_p_best_max_rule_fires_on_explicit_kwarg`
      - `test_p_best_max_rule_skips_implicit_default`
      Plus `tests/test_self_improve.py::test_default_catalog_has_categorical_rules`
      extended with the `("JSO", "p_best_max")` membership assertion.
- [x] **Docs**: `planning/SELF_IMPROVEMENT_LOOP.md` §13 entry +
      promoted next-iteration ticket; `doc/source/guide.rst`
      quick-nav; `doc/source/guide_benchmarking.rst` categorical-
      rules section bumped to "nine"; `AGENTS.md` rule list.

### Catalog rules for `RegionUCB.ucb_c` / `gauss_fraction` / `gauss_scale` — 2026-06-08
- [x] **Three new `MutationRule` entries on `default_catalog`**
      (`panobbgo/self_improve.py`) covering the three leaf-bandit
      knobs of the 2026-06-05 RegionUCB heuristic:
      - `RegionUCB.ucb_c` (`log_uniform_perturb`,
        `bounds=(0.1, 4.0)`, `log_step=0.15`) — UCB1 exploration
        weight, brackets the literature default of 1.0.
      - `RegionUCB.gauss_fraction` (`float_uniform`,
        `bounds=(0.0, 1.0)`) — fraction of in-leaf draws taken
        as Gaussian around the leaf best (the rest are uniform
        over the leaf box); the full ``[0, 1]`` range
        symmetrically covers the LA-MCTS pure-uniform regime and
        the pure-local-refinement regime.
      - `RegionUCB.gauss_scale` (`log_uniform_perturb`,
        `bounds=(0.05, 0.5)`, `log_step=0.15`) — Gaussian
        std-dev as a fraction of the leaf's ranges; the
        constructor default ``0.25`` lives near the geometric
        centre of the log-uniform window.
- [x] **Seed-spec activation** — `Rewarding_RegionUCB` in
      `_make_standard_strategies()` now ships
      `(RegionUCB, {"ucb_c": 1.0, "gauss_fraction": 0.5,
      "gauss_scale": 0.25})` instead of `(RegionUCB, {})`.  All
      three values match the constructor defaults so RegionUCB
      construction is byte-identical — only the kwarg dict's
      *membership* changes, which is exactly what activates the
      new catalog rules on the standard-mode battery rather than
      letting them sit dormant.  Closes the *Follow-ups: tune
      `ucb_c` / `gauss_fraction` via the self-improvement
      catalog* note in the 2026-06-05 RegionUCB ship.
- [x] **Tests** — 5 new tests in
      `tests/test_heuristic_region_ucb.py`:
      - `test_kwarg_catalog_has_region_ucb_ucb_c_rule`
      - `test_kwarg_catalog_has_region_ucb_gauss_fraction_rule`
      - `test_kwarg_catalog_has_region_ucb_gauss_scale_rule`
      - `test_region_ucb_rules_skip_implicit_default`
      - `test_rewarding_region_ucb_seed_spec_has_explicit_region_ucb_kwargs`
- [x] **Docs**: `planning/SELF_IMPROVEMENT_LOOP.md` §13 entry;
      `doc/source/guide.rst` quick-nav; `doc/source/guide_benchmarking.rst`
      kwarg-catalog list; `AGENTS.md` rule list.

### Catalog rules for `Restart.patience` and `LBFGSB.max_starts` — 2026-06-06
- [x] **Two new `integer_add` `MutationRule` entries on
      `default_catalog`** (`panobbgo/self_improve.py`):
      - `Restart.patience` (`bounds=(3, 200)`,
        `delta_choices=(-20, -10, -5, 5, 10, 20)`) — the more
        impactful of the two `Restart` knobs, alongside the existing
        `Restart.max_restarts` rule.  Closes the *Restart.patience
        mutation rule* next-iteration ticket.
      - `LBFGSB.max_starts` (`bounds=(1, 50)`,
        `delta_choices=(-5, -2, -1, 1, 2, 5)`) — multi-start L-BFGS-B
        restart budget cap.  Closes the *LBFGSB.max_starts catalog
        rule* next-iteration ticket under the *LBFGSB follow-ups*
        block.
- [x] **`_find_targets` None-skip** — the "param already in kwargs"
      predicate now also requires `kwargs[param_name] is not None`.
      `None` is the auto-default sentinel a number of heuristics use
      (`Restart.patience → 5·dim`, `LBFGSB.max_starts → unlimited`),
      and numeric mutation kinds cannot meaningfully perturb it.
      Behaviourally inert for every previously-shipped catalog rule
      (no prior rule's target spec carries a `None`-valued kwarg).
- [x] **Tests** — 5 new tests:
      - `tests/test_self_improve.py::TestMutationCatalog::test_applicable_rules_skips_none_value`
      - `tests/test_analyzer_restart.py::test_kwarg_catalog_has_restart_patience_rule`
      - `tests/test_analyzer_restart.py::test_restart_patience_rule_skips_none_sentinel`
      - `tests/test_heuristic_lbfgsb.py::LBFGSBRegistrationTests::test_kwarg_catalog_has_max_starts_rule`
      - `tests/test_heuristic_lbfgsb.py::LBFGSBRegistrationTests::test_max_starts_rule_skips_none_sentinel`
- [x] **Docs**: `planning/SELF_IMPROVEMENT_LOOP.md` §13 entry +
      promoted next-iteration tickets; `doc/source/guide.rst`
      quick-nav; `doc/source/guide_benchmarking.rst` kwarg-catalog
      list; `AGENTS.md` rule list.

### RegionUCB heuristic — UCB1 allocation over Splitter leaves — 2026-06-05
- [x] **New heuristic `panobbgo/heuristics/region_ucb.py`** (`RegionUCB`):
      treats the Splitter tree's leaves as bandit arms (LA-MCTS / DIRECT
      spirit), selects a leaf via a UCB1-style score
      (rank-based quality from the leaf's best penalty + `ucb_c *
      sqrt(log N / n_leaf)`; empty leaves first), and samples
      candidates *inside* the chosen box (uniform + Gaussian around the
      leaf best, clipped).  Because placement is steered by the heuristic
      itself, the existing strategy-level bandits handle credit
      assignment with no core changes.
- [x] **Harness wiring** — `Rewarding_RegionUCB` spec in
      `_make_standard_strategies()` (= `Rewarding_Diverse` pool +
      `RegionUCB` arm, so the score delta isolates region allocation).
      Deliberately *not* in quick mode: at 75 evals the Splitter yields
      too few leaves to matter (15-rep quick A/B was a tie), and
      `test_quick_strategies_unchanged` guards quick mode at 2 specs.
- [x] **Measured (standard mode, 10 reps, seed 42)** —
      StyblinskiTang_2D +0.302 score (SR 0.20→0.50), Rosenbrock_2D
      −0.167 (exploration tax on a unimodal valley), others tied;
      net per-strategy mean +0.019 vs `Rewarding_Diverse`.  Rosenbrock_5D
      median distance halved (9.64→4.97) without reaching tolerance.
      Artifacts: `ab_region_ucb_standard.json`.
- **Design rationale.**  Replaces the per-region *strategy* prototype
  (parked in `sketchpad/grok_per_region_bandit_strategy.py`): a
  strategy-level per-leaf bandit cannot steer point placement because
  heuristics autonomously push into their output queues — the strategy
  only chooses whose queue to drain — and selection/reward then operate
  on different leaves.  Inverting the decomposition (bandit over
  *regions* inside a heuristic, strategy bandit over *heuristics* as
  before) keeps statistics dense and needs no `StrategyBase` changes.
- **Follow-ups**: tune `ucb_c` / `gauss_fraction` via the
  self-improvement catalog; consider volume-aware exploration term
  (DIRECT-style) and parent-posterior inheritance on splits.

### Stochastic-K stagnation rebuild for the random PSO topology — 2026-06-05
- [x] **Opt-in `PSO.stagnation_threshold` kwarg** in
      `panobbgo/heuristics/pso.py`; closes the *Per-iteration
      re-sampled random PSO topology (stochastic-K)* follow-up below
      the 2026-05-29 random-topology entry in
      `planning/SELF_IMPROVEMENT_LOOP.md`.  When set to a positive
      integer and the topology is `"random"`, the informer adjacency
      is re-sampled mid-run after `N` consecutive incoming results
      land without lifting the global best — a finer-grained
      stagnation rebuild than the restart-gated re-sampling that
      shipped 2026-05-29.  Default `None` preserves the prior
      behaviour bit-for-bit.  Implemented via
      `_maybe_rebuild_random_adjacency` (called from
      `on_new_results` after every global-best refresh) and an
      `_stagnation_counter` reset on improvement, `on_start`, and
      `on_restart`.
  - **Why it matters.**  Under
    :class:`~panobbgo.analyzers.restart.Restart` restarts are rare,
    so the random adjacency can stay locked into a bad realisation
    for hundreds of incoming results without the stochastic-K
    rebuild.  Clerc 2007 / SPSO 2011 standardises this as the
    stagnation-trigger rebuild policy for the random topology.
- [x] **Catalog wiring** — `default_catalog` ships a new
      `PSO.stagnation_threshold` `integer_add` rule
      (`bounds=(5, 60)`, `delta_choices=(-10, -5, 5, 10)`).  Only
      fires when a spec sets the kwarg explicitly (per
      `_find_targets`'s "param already in kwargs" predicate), so the
      built-in `_make_quick_strategies` /
      `_make_standard_strategies` / `_make_full_strategies`
      factories see no behavioural change.
- [x] **Backwards compatibility** — strictly safe.
      `stagnation_threshold=None` (default) bypasses the policy
      entirely; every existing PSO instance retains its prior
      behaviour bit-for-bit (all 68 pre-existing PSO tests pass
      unchanged).  The helper is a no-op for `gbest` / `lbest` /
      `vonneumann` topologies.
- [x] **Tests** — 13 new tests in
      `tests/test_heuristic_pso.py::PSOStochasticKTests` (total 81):
      ctor validation, counter starts at zero, resets on
      improvement, rebuild fires at threshold, no rebuild below
      threshold, no rebuild when policy is off, no rebuild for
      non-random topologies, `on_restart` zeros the counter, first
      global best does not tick the counter.  Plus a catalog
      membership test.
- [x] **Documentation updated** — planning doc §13 entry,
      `guide.rst`, `guide_benchmarking.rst`, `guide_architecture.rst`,
      `heuristics.rst`, `AGENTS.md`.

### Analyzer add/drop structural mutations — 2026-06-02
- [x] **New ops** `add_analyzer` / `drop_analyzer` on
      :class:`panobbgo.self_improve.StructuralMutationRule`, symmetric
      to the existing `add_heuristic` / `drop_heuristic` ops.  Closes
      the *Analyzer add/drop* follow-up below the 2026-05-03
      structural-catalog §13 entry in
      `planning/SELF_IMPROVEMENT_LOOP.md`.
  - **Why it matters.** Before this ship, the loop's reach into the
    strategy spec was asymmetric: it could mutate the *heuristics*
    portfolio but not the *analyzers* — even though analyzers carry
    materially different behaviour (most notably the :class:`Restart`
    analyzer's IPOP-style warm restarts).  Adding the symmetric ops
    extends the bandit's reach without disturbing any existing ledger.
- [x] **`min_analyzers` field** (default `0`) replaces `min_heuristics`
      as the post-drop safety floor for analyzer ops.  Unlike
      heuristics, an empty analyzers list is a valid spec (analyzers
      are non-essential).
- [x] **Catalog wiring** — `default_structural_catalog` gains two new
      rules: `add_analyzer` with candidates
      `(Sensitivity, {"update_interval": 20})` and
      `(Restart, {"patience": None, "restart_strategy": "diverse", "max_restarts": 5})`;
      and `drop_analyzer` with `min_analyzers=0`.  Both at probability
      `0.3` — structural ops sample sparingly relative to kwarg
      retunes.
- [x] **Per-class bandit arms** work identically for the new ops —
      :func:`_proposal_rule_key` keys analyzer ops by their op name in
      :data:`_STRUCTURAL_OPS`.  Setting
      :attr:`LoopConfig.structural_per_class_arms=True` (or
      `--structural-per-class-arms`) splits ``add_analyzer`` into
      `(Sensitivity, add_analyzer, structural)` /
      `(Restart, add_analyzer, structural)` arms.
- [x] **Backwards-compatible.** All 180 pre-existing
      `tests.test_self_improve` tests pass unchanged; the only edit
      was the `TestDefaultStructuralCatalog` expected ops set (two →
      four).  Existing ledger consumers see the two new `rule_kind`
      strings as additional values they may ignore.
- [x] **Tests** — `tests/test_self_improve.py` (+34 tests, total 214):
      validation (5), structural-hit enumeration (6), catalog sampling
      (4), apply-side dispatch (7), per-class bandit arms (5), proposal
      serialisation (2), default catalog (4), end-to-end (1).
- [x] **Documentation updated** — `planning/SELF_IMPROVEMENT_LOOP.md`
      (§13 entry, §2 missing-pieces list, §7 change catalog, §9 Phase 6
      checklist), `doc/source/guide.rst`,
      `doc/source/guide_benchmarking.rst`, and `AGENTS.md`.

### Hierarchical bandit over per-class structural arms — 2026-06-01
- [x] **`AdaptiveMutationSampler.structural_borrow_alpha`** — a
      ``κ ≥ 0`` "borrow" coefficient that turns the per-class
      structural arms (shipped 2026-05-18) into a hierarchical
      Beta-Binomial.  Each per-class arm's Beta posterior borrows
      ``κ · (n_other_class_accepts, n_other_class_failures)`` from the
      op-level aggregate (sum across sibling per-class arms with the
      same op), with deliberate self-exclusion so a per-class arm
      never borrows from itself.  ``κ = 0`` (default) recovers the
      pure per-class semantics; ``κ = 0.5`` is the recommended
      starting point for unattended runs.
- [x] **`LoopConfig.structural_borrow_alpha`** and matching CLI flag
      `--structural-borrow-alpha`, both opt-in.  Only effective with
      both ``--adaptive`` and ``--structural-per-class-arms``;
      otherwise silently inert.
- [x] **14 new tests** in
      `tests/test_self_improve.py::TestHierarchicalStructuralBandit`:
      validation, ``κ = 0`` recovers per-class behaviour, borrow inert
      when ``per_class_structural=False`` or for kwarg rules, fresh
      class warms with op aggregate, self-exclusion verified via the
      rationale-reported ``Beta(α, β)``, ``LoopConfig`` propagation,
      etc.
- [x] **Docs updated**: `planning/SELF_IMPROVEMENT_LOOP.md` §13 entry
      + idea promoted from "open" to "shipped";
      `doc/source/guide_benchmarking.rst` new "Hierarchical bandit
      over per-class structural arms" subsection;
      `doc/source/guide.rst` quick-nav mention; `AGENTS.md`
      run-the-loop bash example.

### Codify `Sobol.scramble=False` in `Rewarding_Diverse` — 2026-05-31
- [x] **First ledger-evidence-driven default change in the panobbgo
      self-improvement loop.**  `panobbgo/harness.py`
      :func:`_make_quick_strategies` now ships ``Rewarding_Diverse``
      with ``(Sobol, {"n": 16, "scramble": False})`` instead of the
      historical ``scramble=True``.  Driven by three independent
      self-improvement loop accepts (iter=9 Δ=+0.0511, iter=15
      Δ=+0.0217, iter=17 Δ=+0.0317), each with a bootstrap-CI lower
      bound strictly above zero and zero per-pair regression — clean
      wins under the §6.2 statistical acceptance rule.
- [x] **Archived the training ledger** at
      `planning/done/self_improve_ledger_2026-05-31.jsonl` and the
      training summary at
      `planning/done/self_improve_summary_2026-05-31.txt` per
      `planning/SELF_IMPROVEMENT_LOOP.md` §12.3 step 5, so the
      bandit primes from a clean slate on the next nightly run
      without conflating the pre- and post-codification accept
      regimes.
- [x] **Catalog rule preserved.**  The
      :class:`~panobbgo.self_improve.MutationRule`
      ``("Sobol", "scramble", "categorical_choice")`` still applies
      to the codified spec (the predicate is "kwarg explicitly set",
      not "kwarg value is True"); the bandit can flip back to
      ``True`` if a future battery prefers Owen scrambling.
- [x] **`BayesOpt_Sobol` (standard) and `harness_ioh.py` unchanged.**
      No ledger evidence on those strategies yet — conservative move
      is to wait for the loop to gather signal before propagating.
- [x] **Documentation** — new §13 entry in
      `planning/SELF_IMPROVEMENT_LOOP.md`,
      `doc/source/guide_benchmarking.rst` categorical-rule section
      callout, `doc/source/guide.rst` quick-nav mention, and this
      TODO entry.

### Inactivity-guarded eps_accept relaxation — 2026-05-30
- [x] **Three new `LoopConfig` knobs** in `panobbgo/self_improve.py`:
      `inactivity_relax_after` (default `0` = disabled),
      `inactivity_relax_factor` (default `0.5`), and
      `inactivity_min_eps_accept` (default `0.001`).  When enabled, the
      loop's accept gate decays the configured `eps_accept`
      geometrically by `factor` for every additional `after`-block of
      consecutive non-accepts, floored at `min_eps_accept`, re-tightened
      on the next accept.  Closes the *Inactivity-guarded loop
      productivity* follow-up in `planning/SELF_IMPROVEMENT_LOOP.md`.
  - **Why it matters.** The most recent unattended ledger
    (`planning/self_improve_summary.txt`) records 15 accepts in 326
    decided iterations (4.6%); earlier windows produced 1 accept in 86
    iterations (~1.2%).  At those rates the Thompson sampler's Beta
    posteriors barely move off the prior — defeating the point of
    adaptive sampling.  A geometric relaxation lets the loop reach for
    borderline improvements (delta between `min_eps_accept` and
    `eps_accept`) that the paired-bootstrap CI rules in as
    statistically distinguishable from zero — exactly the regime where
    the historical point-gate was leaving signal on the floor.
- [x] **New `LoopConfig.effective_eps_accept(iters_since_accept)` helper**
      returning `max(eps_accept · factor^(s // after), min_eps_accept)`
      so the rule is callable directly from tests / callers without
      reaching into the loop driver.
- [x] **Two new `LoopIterationRecord` fields** — `effective_eps_accept`
      and `iters_since_accept` — persist the threshold that
      `statistical_accept` actually saw and the streak length consulted
      to compute it.  Both default to `None` on legacy records so the
      JSONL load path keeps working against historical ledgers.
- [x] **CLI flags** on `scripts/self_improve.py run` —
      `--inactivity-relax-after`, `--inactivity-relax-factor`, and
      `--inactivity-min-eps-accept` — mirror the `LoopConfig` knobs
      with the same defaults (`0`, `0.5`, `0.001`).
- [x] **15 new tests in `tests/test_self_improve.py`** (total 210):
      `TestInactivityRelaxConfig` covers validation (negative `after`,
      out-of-range `factor`, negative / too-large floor) and the
      threshold maths (no-relax before threshold, geometric decay,
      floor clamping); `TestInactivityRelaxIntegration` covers
      end-to-end loop behaviour (records carry effective threshold +
      streak, streak resets on accept, skip-iterations count toward
      streak, borderline +0.04 delta is accepted by relaxed 0.025 gate
      and rejected again after reset, disabled mode populates fields
      with constant `eps_accept`, ledger round-trip, legacy record
      construction).
- [x] **Backwards compatibility** — strictly safe.  Defaults disable
      the feature; when `after = 0`, `effective_eps_accept` is a
      constant equal to `eps_accept` and the loop passes the same
      value to `statistical_accept` as before.  Composite baseline on
      every default battery is byte-identical and existing ledgers
      stay valid (the two new record fields default to `None` for
      legacy records).
- [x] **Documentation updated** — `planning/SELF_IMPROVEMENT_LOOP.md`
      (§13 entry, follow-up promoted to "shipped" with the unshipped
      half left open, new "Inactivity-relax telemetry in summary view"
      follow-up), `doc/source/guide_benchmarking.rst` (new subsection
      with the geometric-decay maths and recommended unattended
      preset), `doc/source/guide.rst` (quick-nav entry), and
      `AGENTS.md` (run-the-loop bash example).

### Random PSO topology (Mendes 2004 / Clerc 2007 / SPSO 2011) — 2026-05-29
- [x] **New `"random"` topology** in `panobbgo/heuristics/pso.py`;
      closes the *Random re-wired topology* PSO follow-up under the
      2026-05-22 entry in `planning/SELF_IMPROVEMENT_LOOP.md`.  Each
      particle is connected to itself plus `k_neighbors` random
      *informers* drawn uniformly with replacement from
      `{0..NP-1} \ {i}`; duplicates are removed so the realised
      neighbourhood size lies in `[2, k_neighbors + 1]`.  Implemented
      via `_init_random_adjacency` (sample the per-particle adjacency
      list once at start) and `_random_neighbors` (lookup helper).
      `_social_best_idx` dispatches the new topology onto the same
      scan-for-best-neighbour-pbest routine already used by `lbest` /
      `vonneumann`.  The adjacency is re-sampled at `on_restart`
      (Clerc 2007 / SPSO 2011 stagnation-rebuild convention).
  - **Why it matters.** Completes the canonical Mendes 2004 topology
    set (gbest / lbest / vonneumann / random); the three geometric
    topologies are all closed-form functions of `NP`, while `random`
    is the structure-free alternative whose diffusion speed depends
    on the realised graph.  Useful when the bandit evidence shows
    neither pure structured topology consistently wins on a given
    battery.  Clerc reports `K=3` as the SPSO 2011 default — matches
    the structural-catalog entry below.
- [x] **Catalog wiring** — `default_structural_catalog` ships a
      fourth PSO entry `(PSO, {"NP": 20, "topology": "random",
      "k_neighbors": 3})` alongside the existing `gbest` / `lbest` /
      `vonneumann` triples.  All four share `cls = PSO` so
      `avoid_duplicates=True` still prevents multiple PSO instances
      per strategy.  `default_catalog`'s `PSO.topology` categorical
      rule grows from three choices to four (`("gbest", "lbest",
      "vonneumann", "random")`) so the bandit can flip an existing
      explicit-topology PSO between all four regimes without dropping
      and re-adding the heuristic.
- [x] **Backwards compatibility** — strictly safe.  `topology`
      defaults to `"gbest"` so every existing PSO instance retains
      its prior behaviour bit-for-bit (all 56 pre-existing PSO tests
      pass unchanged).  The new informer-adjacency field is `None`
      under any other topology, so memory / RNG draws are unchanged
      for `gbest` / `lbest` / `vonneumann`.  The structural catalog
      gains one extra `add_heuristic` candidate; the categorical
      rule's cardinality bumps from 3 to 4 (callers parsing
      `rule.choices` see one extra string they may ignore).
- [x] **Tests** — `tests/test_heuristic_pso.py` (+12 tests, total
      80): random construction round-trip; adjacency built on start;
      every particle is its own informer; realised neighbourhood
      ≤ k+1 with no duplicates; self appears exactly once (i.e. the
      index-shift logic excludes self from the random draws across 50
      seeds); asymmetric graph in general; seed reproducibility;
      adjacency re-sampled on restart; social-best limited to
      informer set (planted-pbest invariant); none-until-evaluated;
      velocity clamp invariant; end-to-end smoke convergence on a
      quadratic; updated structural-catalog test confirming all four
      PSO topology variants appear among `add_heuristic` candidates;
      updated categorical-rule test confirming `default_catalog` now
      ships `choices=("gbest", "lbest", "vonneumann", "random")`.
- [x] **Documentation updated** — `planning/SELF_IMPROVEMENT_LOOP.md`,
      `doc/source/guide.rst`, `doc/source/guide_benchmarking.rst`,
      `doc/source/guide_architecture.rst`, `doc/source/heuristics.rst`,
      `AGENTS.md`, and this `TODO.md` entry.
### NL-SHADE-LBC adaptive DE (CEC 2022 winner) — 2026-05-28
- [x] **New `NLSHADE_LBC` heuristic** in
      `panobbgo/heuristics/nl_shade_lbc.py`; closes the *NL-SHADE-LBC*
      DE-family follow-up in `planning/SELF_IMPROVEMENT_LOOP.md`.
      Direct subclass of
      :class:`~panobbgo.heuristics.nl_shade_rsp.NLSHADE_RSP` (Stanovov,
      Akhmedova & Semenkin, CEC 2022 winner) adding **Linear Bias
      Change** in the success-history memory update: the F / CR
      Lehmer-mean order ``p`` is linearly scheduled across budget
      progress instead of fixed at 2.  Literature defaults
      ``p_F: 3.5 → 1.5``, ``p_CR: 1.0 → 1.5``, spread ``m_lbc = 1.5``;
      at ``p = 2, m = 1`` the formula recovers the standard L-SHADE
      Lehmer mean.
  - **Why it matters.** NL-SHADE-LBC won the CEC-2022 single-objective
    bound-constrained competition and is the direct NL-SHADE-RSP
    descendant — the literature frontier as of the most recent CEC
    competition we can mirror.  Adds a fifth DE-family arm the bandit
    can pick whichever wins on the current battery.
- [x] **Catalog wiring** — `default_structural_catalog` gains
      `(NLSHADE_LBC, {"NP_init": 30, "k_rank": 3.0})` as a fifteenth
      `add_heuristic` candidate; `default_catalog` gains six
      LBC-specific rules (`NLSHADE_LBC.NP_init` integer_add,
      `NLSHADE_LBC.p_F_init` / `p_F_final` / `p_CR_init` / `p_CR_final`
      float_uniform, `NLSHADE_LBC.m_lbc` float_uniform).
- [x] **Deviations from the full CEC-2022 paper** (documented for
      honesty): the adaptive binomial / exponential crossover blend
      and the repetitive-generation bound-constraint handling are
      *not* ported — the same async-pipeline limitations that motivated
      omitting them from NL-SHADE-RSP apply here.  Both queued as
      follow-ups.
- [x] **CR-zero handling** — preserves the L-SHADE terminal sentinel
      rule and filters strict zeros out of the LBC sum (because
      ``s^(p−m)`` with ``p < m`` is undefined at ``s = 0``).
- [x] **Backwards compatibility** — strictly safe.  NLSHADE_LBC is
      opt-in: not added to any default battery, so existing composite
      baselines stay byte-identical.  NL-SHADE-RSP / jSO / L-SHADE
      base classes are untouched — only the LBC subclass overrides
      `_update_memory`; verified by a regression test that
      ``NLSHADE_RSP._update_memory`` still produces the standard
      L-SHADE Lehmer mean output.
- [x] **Tests** — `tests/test_heuristic_nl_shade_lbc.py` (30 tests):
      construction validation, LBC schedule (endpoints, linear
      midpoint, clipping, budget-unknown fallback), memory update
      (anchor-bin skip, pointer mod (H-1), no-op on empty buffer,
      [0,1] clamping, formula recovers Σ(w·F^3.5)/Σ(w·F^2.0) at
      progress=0, p=2/m=1 recovers L-SHADE for both F and CR, CR=0
      terminal sentinel, terminal-bin stays terminal, mixed-zero CR
      filtered, uniform weights on zero-delta), pipeline (on_start,
      smoke convergence, restart resets), inheritance safety
      (NLSHADE_RSP unchanged), and registration.
- [x] **Documentation updated** —
      `planning/SELF_IMPROVEMENT_LOOP.md` (new §13 entry, follow-up
      idea seeded), `doc/source/guide.rst`,
      `doc/source/guide_benchmarking.rst`,
      `doc/source/guide_architecture.rst`,
      `doc/source/heuristics.rst`, and this TODO entry.
### Multi-start L-BFGS-B gradient local optimizer (rescued + catalogued) — 2026-05-27
- [x] **Rewrote `panobbgo/heuristics/lbfgsb.py`** from a one-shot,
  box-centre, restart-blind, *unreferenced* stub into a robust
  **multi-start** bound-constrained quasi-Newton local optimizer.  The
  worker runs `scipy.optimize.fmin_l_bfgs_b` repeatedly — first descent
  from the box centre, subsequent descents from fresh uniform-random
  restarts — using the whole strategy budget instead of going idle after
  one convergence.  `on_restart` warm-starts at the Restart analyzer's
  centre.  Subprocess lifecycle re-modelled on the tested COBYQA adapter
  (`spawn`, `cap=1`, `SystemExit`-on-closed-pipe).  New validated kwargs
  `max_starts` / `maxfun` / `epsilon` / `seed`.
- [x] **Added `LBFGSB` to `default_structural_catalog()`** as the 15th
  `add_heuristic` candidate (`avoid_duplicates=True`) — the only
  gradient-based arm the self-improvement loop can deploy.
- [x] **Why** — the harness shows every Panobbgo *strategy* scores 0.0 on
  `Rosenbrock_5D` (a smooth ill-conditioned valley) while scipy's
  `dual_annealing` solves it via its own L-BFGS-B local search.  A
  *dedicated* LBFGSB strategy now solves Rosenbrock_2D/5D to
  `func_distance ≈ 3e-11` (SR 5/5) in the same A/B.  **Negative result
  recorded:** adding it to the budget-split `Rewarding_Diverse` portfolio
  does *not* crack Rosenbrock_5D (and can regress other problems) — value
  is in dedicated / loop-discovered portfolios, which is why it is
  catalog-only and the default battery is unchanged.
- [x] **Backwards-compatible** — opt-in (not in any default strategy);
  composite baseline byte-identical; integration tests and the
  `on_new_results` penalty contract pass unchanged.
- [x] **Tests** — rewrote `tests/test_heuristic_lbfgsb.py` (29) and
  `tests/test_heuristic_lbfgsb_robustness.py` (9) on the COBYQA template:
  ctor validation, lifecycle, pipe wiring, restart, fake-pipe worker
  multi-start / reproducibility / robustness, registration, and a
  Rosenbrock_5D scipy smoke.
- [x] **Docs** — `heuristics.rst`, `guide_architecture.rst`,
  `guide_benchmarking.rst`, `guide.rst`, `AGENTS.md`, and
  `planning/SELF_IMPROVEMENT_LOOP.md` (§13 + LBFGSB follow-ups).

### NL-SHADE-RSP adaptive DE (CEC 2021 winner) — 2026-05-25
- [x] **New `NLSHADE_RSP` heuristic** in
      `panobbgo/heuristics/nl_shade_rsp.py`; closes the *NL-SHADE-RSP /
      NL-SHADE-LBC* DE-family follow-up in
      `planning/SELF_IMPROVEMENT_LOOP.md`.  Direct subclass of
      :class:`~panobbgo.heuristics.jso.JSO` (Stanovov, Akhmedova &
      Semenkin, CEC 2021 winner) adding three refinements over jSO:
      Non-Linear Population Size Reduction
      (`NP(r) = round((NP_min − NP_init)·r^(1−r) + NP_init)`),
      Rank-based Selective Pressure on the differential `r1` draw
      (`k_rank` default `3`), and a randomised per-generation adaptive
      archive cap (`adaptive_archive`).
  - **Why it matters.** NL-SHADE-RSP won the CEC-2021 single-objective
    bound-constrained competition and is the direct jSO descendant —
    the natural fourth DE-family arm after basic DE / L-SHADE / jSO.
    Gives the self-improvement loop a CEC-2021-class arm the bandit can
    pick whichever wins on the current battery.
- [x] **Behaviour-preserving base-class refactor** — extracted the
      three jSO override points into `LSHADE._select_r1` (r1 selection),
      `LSHADE._lpsr_target` (population-reduction schedule), and
      `LSHADE._archive_cap` (archive cap).  L-SHADE and jSO consume them
      with their exact prior RNG-draw sequence, so both stay
      byte-identical (all 99 pre-existing L-SHADE / jSO tests pass
      unchanged).
- [x] **Catalog wiring** — `default_structural_catalog` gains
      `(NLSHADE_RSP, {"NP_init": 30, "k_rank": 3.0})` as a fourteenth
      `add_heuristic` candidate; `default_catalog` gains three rules
      (`NLSHADE_RSP.NP_init` integer_add, `NLSHADE_RSP.k_rank`
      float_uniform `[1, 5]`, `NLSHADE_RSP.adaptive_archive`
      categorical).
- [x] **Deviations from the full CEC-2021 paper** (documented for
      honesty): the adaptive binomial / exponential crossover blend and
      the success-ratio archive-probability (pA) adaptation are *not*
      ported — they need per-trial bookkeeping the async pipeline does
      not expose.  Binomial crossover (jSO) + randomised archive cap
      used instead.  Both queued as follow-ups.
- [x] **Impact** — A/B vs jSO in the same Rewarding strategy, fixed
      battery, 12 reps × 3 problems × 1000 evals: mean composite delta
      **+0.0004** (statistical tie), seed-dependent complementarity
      (jSO wins seed 42; NL-SHADE-RSP wins 43 & 44).  Component
      decomposition confirmed no bug (same basins as jSO).  The CEC-DE
      refinements are large-budget specialists; within noise at
      panobbgo's small composite-battery budgets.  Backwards-compatible:
      opt-in only, composite baseline unchanged, queued for nightly
      loop validation.
- [x] **Tests** — `tests/test_heuristic_nl_shade_rsp.py` (34 tests):
      construction validation, NLPSR (endpoints / monotonicity /
      faster-than-linear / shrink), RSP (excludes target / empty pool /
      rank bias / k_rank=0 uniform), adaptive archive (fixed when off /
      within-bounds / clip / lazy / resample / never-exceeds), pipeline
      (on_start / trials / win-and-archive / restart / smoke), base-class
      hook safety, and registration.
- [x] **Documentation updated** — `planning/SELF_IMPROVEMENT_LOOP.md`,
      `doc/source/guide.rst`, `doc/source/guide_benchmarking.rst`,
      `doc/source/guide_architecture.rst`, `doc/source/heuristics.rst`,
      and `AGENTS.md`.

### Von Neumann (4-connected 2-D toroidal grid) PSO topology — 2026-05-22
- [x] **New `"vonneumann"` topology** in
      `panobbgo/heuristics/pso.py`; closes the *Random / Von Neumann
      topologies* PSO follow-up under the 2026-05-07 entry in
      `planning/SELF_IMPROVEMENT_LOOP.md`.  :class:`PSO` gains two new
      helpers — :meth:`_vonneumann_grid` (factors `NP` into
      `R × C ≥ NP` with `R ≈ √NP`) and :meth:`_vonneumann_neighbors`
      (returns the 4-connected wrap-around N/S/E/W indices plus
      self, skipping phantom slots on non-rectangular grids).
      :meth:`_social_best_idx` dispatches the new topology onto the
      same scan-for-best-neighbour-pbest routine already used by
      `lbest`.  :class:`PSO` now ships three complementary
      information-diffusion regimes: instantaneous (`gbest`), one-hop
      linear (`lbest`), and two-hop planar (`vonneumann`).
  - **Why it matters.** Kennedy & Mendes (2003) and Mendes (2004)
    identify the 4-connected 2-D toroidal grid as a stable middle
    ground that wins on a broader range of problem classes than
    either pure ring (`lbest`) or pure star (`gbest`).  Shipping all
    three topologies in the structural catalog gives the
    self-improvement loop a third PSO arm the bandit can pick
    whichever wins on the current battery.
- [x] **Default structural catalog grows to three PSO entries** —
      `(PSO, {"NP": 20})`, `(PSO, {"NP": 20, "topology": "lbest",
      "k_neighbors": 2})`, and `(PSO, {"NP": 20, "topology":
      "vonneumann"})`.  All three share `cls = PSO` so
      `avoid_duplicates=True` still prevents multiple PSO instances
      per strategy; the catalog samples uniformly between them when
      PSO is not yet present.
- [x] **`PSO.topology` categorical rule grows from 2 to 3 choices** —
      `("gbest", "lbest", "vonneumann")` so the bandit can flip an
      existing explicit-topology PSO between all three regimes
      without dropping and re-adding the heuristic.
- [x] **Backwards compatibility** — strictly safe.  `topology`
      defaults to `"gbest"`; every existing PSO instance retains
      its prior behaviour bit-for-bit, including the 56 pre-existing
      tests in `tests/test_heuristic_pso.py`.  The categorical rule
      expansion adds one choice; callers passing the prior choices
      tuple get the same uniform-over-the-set draw.
- [x] **Tests** — `tests/test_heuristic_pso.py` (11 new tests, total
      67): vonneumann construction round-trip; grid factoring for
      perfect rectangles and primes / near-primes; 4-connected
      wrap-around correctness on a 4×5 grid; phantom-cell skipping
      on a 3×4 grid (NP=10); duplicate elimination on a 2×2 swarm
      (NP=4); social attractor uses the 2-D neighbourhood, not the
      global best, when a better pbest exists outside the N/S/E/W
      set; social attractor returns None until at least one
      neighbour has a pbest; velocity clamp invariant under
      vonneumann; end-to-end smoke convergence on a quadratic;
      categorical-rule membership test; updated structural-catalog
      test confirming all three PSO topology variants appear among
      the `add_heuristic` candidates.
- [x] **Documentation updated** — `planning/SELF_IMPROVEMENT_LOOP.md`,
      `doc/source/guide.rst`, `doc/source/guide_benchmarking.rst`,
      `doc/source/guide_architecture.rst`, `doc/source/heuristics.rst`,
      and `AGENTS.md` all updated to reflect the tri-topology PSO
      candidate pool.

### jSO Asymmetric F-cap (Three-Phase, Brest 2017) — 2026-05-21
- [x] **New `LSHADE.F_schedule` opt-in kwarg** in
      `panobbgo/heuristics/lshade.py`; closes the *jSO asymmetric
      F-cap during early generations* follow-up under the 2026-05-19
      iLSHADE / jSO `p_best` entry in
      `planning/SELF_IMPROVEMENT_LOOP.md`.  When set to `True`,
      sampled `F` is capped at `0.7` while `progress < 0.6`, at `0.8`
      while `0.6 ≤ progress < 0.9`, and left unclamped in the final
      10% of the budget — the literature-faithful Brest et al. (2017,
      §III-D) three-phase asymmetric cap.  Default `None` is
      byte-identical to the 2026-05-10 L-SHADE ship.
  - **Why it matters.** The 2026-05-15 :class:`JSO` ship implemented
    only the *first* phase of the cap (`F ≤ 0.7` for the first 60%
    of the budget), missing the middle phase that the literature
    documents as part of jSO's CEC-2017-winning recipe.  Adding the
    cap as shared infrastructure on `LSHADE` lets the loop driver
    expose it as a categorical mutation rule and lets `JSO`
    inherit the literature-faithful three-phase cap by construction.
- [x] **New `LSHADE._progress()` helper** — returns
      `len(strategy.results) / max_eval` clipped to `[0, 1]`, or
      `None` when the budget is unknown so each schedule
      (`_current_p_best`, `_apply_F_cap`, `_apply_lpsr`) picks its
      own fall-back.  Replaces the inlined computations in
      `_current_p_best` and `_apply_lpsr`.
- [x] **New `LSHADE._apply_F_cap(F)` helper** — implements the
      three-phase cap.  `_sample_F_CR` calls it once on every draw
      so the cap is shared infrastructure across L-SHADE and
      every subclass.
- [x] **JSO opts in by construction** — `JSO.__init__` passes
      `F_schedule=True` to `super().__init__`.  The old
      `_sample_F_CR` override and module-level `_F_CLAMP_*`
      constants are removed in favour of the inherited
      machinery.  `_progress()` is also removed (inherited from
      LSHADE).  `_current_p_best` / `_current_F_weight` are
      updated to handle the new `_progress()` contract
      (None → early-phase fall-back).
- [x] **New `default_catalog` rule** — `LSHADE.F_schedule`
      (`categorical_choice` over `(True, False)`) joins the existing
      `NP_init` / `H` / `p_best` / `p_best_end` / `archive_factor`
      LSHADE rules.  Only fires when a spec sets `F_schedule`
      explicitly (per `_find_targets`'s "param already in kwargs"
      predicate).  Gives the loop a discrete way to flip an existing
      `LSHADE` instance between the Tanabe-Fukunaga and jSO regimes.
- [x] **15 new tests in `tests/test_heuristic_lshade.py`**
      (`LSHADEAsymmetricFCapTests` test class — total 97):
      default `F_schedule` is `None`, custom construction
      (`True` / `False`), invalid type rejection, `_apply_F_cap`
      disabled-when-off (None and False), three-phase clamping
      (phase 1 ≤ 0.7, phase 2 ≤ 0.8 admits values > 0.7, phase 3
      unclamped), phase-boundary inclusivity (`progress = 0.6` →
      phase 2; `progress = 0.9` → phase 3), bypass when budget
      unknown, end-to-end `_sample_F_CR` respects the cap across
      phases, `_progress` returns `None` without budget, `_progress`
      clipping, and a catalog membership test for `LSHADE.F_schedule`.
- [x] **3 new tests in `tests/test_heuristic_jso.py`** (total 36):
      jSO opts into `F_schedule=True` by construction; jSO
      `_progress()` returns `None` (not 0.0) without budget; jSO
      `_current_p_best` / `_current_F_weight` fall back to the
      early-phase value when the budget is unknown.  Plus updated
      tests for the *three-phase* clamp on jSO (replacing the old
      two-phase tests).
- [x] **Backwards compatible on L-SHADE** — `F_schedule` defaults
      to `None`; every existing L-SHADE instance retains its prior
      behaviour bit-for-bit, all pre-existing L-SHADE tests pass
      unchanged.  jSO's behaviour does change in the middle 30% of
      the budget where the second-phase cap (`F ≤ 0.8`) now
      activates; the jSO unit tests have been updated to reflect
      the three-phase contract.  This is a literature-faithful
      completion rather than a behaviour regression — jSO has
      always been documented as a three-phase asymmetric F-cap
      heuristic since Brest et al. 2017.
- [x] **Documentation updated**
  - `planning/SELF_IMPROVEMENT_LOOP.md`: new §13 dated entry; the
    *jSO asymmetric F-cap during early generations* follow-up
    promoted from "open" to "shipped".
  - `doc/source/guide_benchmarking.rst`: the L-SHADE / jSO entries
    under the structural-catalog candidate pool now describe the
    opt-in jSO F-cap on L-SHADE and the literature-faithful
    three-phase cap on jSO; new `LSHADE.F_schedule` entry under
    the categorical-mutation-rule section.
  - `doc/source/guide.rst`: quick-nav entry mentions
    `LSHADE.F_schedule` and the literature-faithful three-phase
    cap on jSO.
  - `AGENTS.md`: self-improvement loop subsection lists the new
    `LSHADE.F_schedule` categorical rule.

### iLSHADE / jSO Adaptive `p_best` Schedule (2026-05-19)
- [x] **New `LSHADE.p_best_end` opt-in kwarg** in
      `panobbgo/heuristics/lshade.py`; closes the *iLSHADE / jSO*
      follow-up under the L-SHADE entry in
      `planning/SELF_IMPROVEMENT_LOOP.md`.  When set, the effective
      ``p_best`` at evaluation count ``e`` (out of
      ``E = strategy.config.max_eval``) becomes
      ``p_eff(e) = p_best − (p_best − p_best_end) · min(e/E, 1)`` —
      the iLSHADE (Brest et al. 2016) / jSO (Brest et al. 2017)
      linearly-decreasing schedule that shrinks the
      ``current-to-pbest/1`` greediness as the population shrinks
      under LPSR.  Canonical jSO setting: ``p_best = 0.25``,
      ``p_best_end = 0.125``.
  - **Why it matters.** jSO won the CEC-2017 single-objective
    competition, establishing the linearly-decreasing
    ``p_best`` schedule as the literature-best refinement on
    top of L-SHADE.  Without it, the bandit could only tune
    ``p_best`` to a single fixed value — leaving on the table
    the early-exploration / late-exploitation trade-off that
    pairs naturally with LPSR.
- [x] **`LSHADE._current_p_best()` helper** — returns
      ``self.p_best`` when ``p_best_end is None`` (the default),
      otherwise the budget-paced linear interpolation between
      ``self.p_best`` and ``self.p_best_end``.  Falls back to
      constant ``self.p_best`` when the strategy budget is
      unknown.  Mirrors the
      :meth:`PSO._current_inertia` pattern shipped 2026-05-07.
- [x] **`_generate_trial` consults `_current_p_best`** exactly where
      it previously used ``self.p_best``, so the mutation / crossover
      / bounds-reflection paths are shared.
- [x] **New `default_catalog` rule** — ``LSHADE.p_best_end``
      (``float_uniform`` over the literature range
      ``[0.025, 0.15]``) joins the existing
      ``NP_init`` / ``H`` / ``p_best`` LSHADE rules.  Only fires
      when a spec sets ``p_best_end`` explicitly.
- [x] **10 new tests in `tests/test_heuristic_lshade.py`** (total 49):
      construction validation (default ``None``, opt-in round-trip,
      invalid ``p_best_end`` rejected — zero / negative /
      too-large / NaN / inf) plus the
      `LSHADEAdaptivePBestTests` test class covering constant
      when ``p_best_end is None``, linear decrease at canonical
      jSO settings, progress > 1 clipping, linear increase
      (symmetric: end > start), budget-unknown fall-back,
      no-op schedule (end == start), and an end-to-end
      ``_generate_trial`` pool-sizing test, plus a catalog
      membership test confirming the new rule is present.
- [x] **Backwards compatible** — ``p_best_end`` defaults to
      ``None``; every existing :class:`LSHADE` instance retains
      its prior behaviour bit-for-bit, all 39 pre-existing
      tests pass unchanged.
- [x] **Documentation updated**
  - `planning/SELF_IMPROVEMENT_LOOP.md`: new §13 dated entry; the
    *iLSHADE / jSO* follow-up under the L-SHADE entry promoted
    from "open" to "shipped"; a new "next iteration" idea
    (jSO asymmetric F-cap) seeded for the follow-up agent.
  - `doc/source/guide_benchmarking.rst`: the L-SHADE bullet
    under the structural-catalog candidate pool now names the
    opt-in iLSHADE / jSO ``p_best_end`` schedule.
  - `doc/source/guide.rst`: quick-nav entry mentions the
    iLSHADE / jSO schedule.

### Per-Class Bandit Arms for Structural Mutations (2026-05-18)
- [x] **New `panobbgo.self_improve.AdaptiveMutationSampler.per_class_structural`** —
      constructor argument (default `False`) that splits each
      `StructuralMutationRule` into one Thompson arm per candidate
      class at sampling time.  With the flag on, `add_heuristic Sobol`
      lives on `("Sobol", "add_heuristic", "structural")` and is
      a distinct arm from `add_heuristic Random` —
      `("Random", "add_heuristic", "structural")`.  Closes the
      *Per-class arms in the bandit* follow-up below the 2026-05-03
      §13 entry in `planning/SELF_IMPROVEMENT_LOOP.md`.
  - **Why it matters.**  The structural catalog shipped 2026-05-03
    collapses every `add_heuristic` proposal — regardless of
    target class — into the single
    `("*", "add_heuristic", "structural")` bandit arm.  That makes
    cold-start variance small but the bandit cannot learn that, e.g.,
    `add Sobol` is a consistent winner while `add Random` is a
    consistent loser.  Per-class arms split the posterior so the
    bandit can concentrate probability on the winning class.
- [x] **New `panobbgo.self_improve.LoopConfig.structural_per_class_arms`** —
      surface the flag in `LoopConfig` so a CLI / TOML config can
      enable it.  Default `False` keeps the published 2026-05-03
      semantics byte-identical.
- [x] **CLI flag `--structural-per-class-arms`** on
      `scripts/self_improve.py run`.  Only effective with
      `--adaptive`; ignored otherwise (the uniform sampler path
      is unaffected by per-class arms).
- [x] **`_proposal_rule_key(..., per_class_structural=False)`** —
      same flag plumbed through so
      `AdaptiveMutationSampler.prime_from_ledger` rebuilds the
      same arm layout the live sampler would create.  Without this
      wiring, priming a per-class sampler from an existing ledger
      would silently fall back to the wildcard arm.
- [x] **`AdaptiveMutationSampler._structural_arm_key(op, class_name)`** —
      centralised helper for the per-class vs collapsed decision
      so `sample` and `prime_from_ledger` cannot drift out of sync.
- [x] **11 new tests in `tests/test_self_improve.py`** (total 158)
      covering: `_proposal_rule_key` per-class round-trip;
      default `per_class_structural=False`; structural arms split
      per candidate class (both X and Y observed, total attempts
      conserved, wildcard key absent); Thompson sampling
      concentrates probability on the winning class (4x ratio
      threshold over 500 post-training samples); drop ops also
      produce per-class arms; kwarg arms unaffected by the flag;
      `prime_from_ledger` uses per-class keys; off-flag priming
      still collapses to the wildcard arm; `LoopConfig` default
      `False`; flag propagates to sampler via `SelfImprover`; flag
      is inert without adaptive sampling.
- [x] **Documentation updated:**
  - `planning/SELF_IMPROVEMENT_LOOP.md`: new §13 entry; the
    *Per-class arms in the bandit* follow-up below the 2026-05-03
    entry promoted from "open" to "shipped".
  - `doc/source/guide_benchmarking.rst`: new "Per-class
    structural bandit arms" subsection under the adaptive
    sampler.
  - `doc/source/guide.rst`: quick-nav entry mentions the feature.
  - `AGENTS.md`: self-improvement loop subsection lists the
    feature with a run-the-loop bash example.

### Bootstrap CI on Multi-Seed Hold-Out Drift (2026-05-17)
- [x] **New `panobbgo.self_improve.aggregate_holdout_drift`** —
      pools per-iteration paired drift samples across every input
      hold-out record and bootstrap-resamples the mean using the
      same machinery as `panobbgo.harness.statistical_accept`.
      Closes the *Bootstrap CI on the drift estimate* follow-up
      listed under the 2026-05-16 multi-seed hold-out in
      `planning/SELF_IMPROVEMENT_LOOP.md`.  Returns a
      `HoldoutDriftAggregate` carrying `mean_drift`, `ci_low`,
      `ci_high`, `worst_drift`/`worst_seed` (preserving the
      shipped reduction for side-by-side display), `any_overfit` /
      `overfit_count`, `n_samples`, `n_records`, `confidence`,
      `eps_overfit`, and `statistically_overfit`.
- [x] **`LoopHoldoutRecord.seed_iteration_scores` /
      `top_iteration_scores`** — per-iteration paired composite
      scores of the seed and top ladder entries on the hold-out
      instances, persisted alongside the existing aggregate scores.
      Default empty lists keep every legacy call site / ledger
      record byte-identical.
- [x] **CLI flags `--fail-on-overfit-ci`,
      `--holdout-ci-confidence`, `--holdout-ci-n-boot`** on
      `scripts/self_improve.py run`.  `--fail-on-overfit-ci` exits
      with code 3 iff the bootstrap CI's upper bound falls below
      `-holdout_eps_overfit` (i.e. the bootstrap rules out a drift
      better than the tolerance at the configured confidence
      level) — a stricter, less-noise-reactive sibling of
      `--fail-on-overfit`.
- [x] **CLI aggregate output** — both `run` and `summary` print
      the CI line alongside the worst-case reduction, e.g.
      `[self_improve] hold-out drift CI: OK_CI  mean=-0.0012  CI95%=[-0.0037, +0.0000]`.
- [x] **Backwards compatibility** — strictly safe.  All 147 prior
      tests pass unchanged.  Legacy records (without per-iteration
      lists) fall back to one-sample-per-record automatically
      inside `aggregate_holdout_drift`; mixed legacy + modern
      inputs work transparently.  The new CLI flags are all
      opt-in; existing `--fail-on-overfit` behaviour is unchanged.
- [x] **20 new tests in `tests/test_self_improve.py`** (total 167):
      `TestAggregateHoldoutDrift` (15 tests — empty input, per-iter
      pooling, legacy fallback, mixed records, worst-drift /
      any-overfit reductions, statistically_overfit semantics on
      constant-negative and mixed-sign samples, CI widening with
      confidence, reproducibility under fixed seed, distinct seeds
      give distinct CIs, eps_overfit override, unequal-length
      defensive handling, JSON round-trip),
      `TestLoopHoldoutRecordPerIterScores` (2 tests — default empty
      lists, to_dict emits the lists), and
      `TestSelfImproverPersistsPerIterScores` (3 tests — single-seed
      run populates per-iter lists, multi-seed run keeps lists
      per-seed, JSONL ledger round-trips them).

### Multi-Seed Hold-Out Validation (2026-05-16)
- [x] **New `panobbgo.self_improve.LoopConfig.holdout_base_seeds`** —
      list-typed sibling of the scalar `holdout_base_seed` shipped
      2026-05-08; closes the *Multi-seed hold-out for robust drift
      estimation* follow-up in `planning/SELF_IMPROVEMENT_LOOP.md`.
      At the end of every loop run, one `LoopHoldoutRecord` is
      written per seed in the list and the CLI aggregates with
      worst-case drift (`min`) and any-overfit (`any`) semantics —
      strictly more conservative than the single-seed check.
  - **Why it matters.**  The single-seed hold-out reduces the
    entire generalisation question to one independent SHA-256
    draw.  When a ladder overfits in a subtle way — for example,
    exploiting a quirk that happens to repeat across the chosen
    hold-out seed — that one draw can miss it.  Aggregating over
    several seeds catches the failure mode the single seed
    misses while remaining cheap relative to the training cost.
- [x] **`LoopConfig.resolved_holdout_seeds()`** helper — single
      branch that returns the effective seed tuple (list when
      non-empty, else scalar promoted to a 1-tuple, else `()` =
      disabled).  Keeps the multi-seed loop driver simple while
      preserving back-compat for scalar callers.
- [x] **`LoopConfig.holdout_harness_config(..., base_seed=None)`** —
      optional `base_seed` argument drives `HarnessConfig.seed`
      per call rather than reading the scalar attribute.  Without
      this wiring, the multi-seed loop would still measure against
      the scalar.
- [x] **`SelfImprover._run_holdout(ladder, base_seed, verbose)`**
      now takes the seed as a parameter; the main loop iterates
      over `resolved_holdout_seeds()` and writes one record per
      seed to the ledger.  `record_type='holdout'` is unchanged, so
      existing ledger consumers see N records back-to-back instead
      of one.
- [x] **CLI flag `--holdout-base-seeds`** on
      `scripts/self_improve.py run`.  Accepts a comma-separated
      list (e.g. `1234,5678,9012`); the parser tolerates whitespace
      around entries and trailing commas, and rejects non-integer
      tokens with a clear error.  The end-of-run summary line and
      the `summary` subcommand both report the aggregated verdict:
      `OVERFIT` if any record flagged overfit, worst (most negative)
      drift across seeds.
- [x] **Validation rules** — `LoopConfig.__post_init__` rejects
      `0` entries (the disable sentinel), collision with
      `base_seed`, and duplicates in the list.  Each rule has a
      distinct error message.  Normalises `List[int]` input to
      `Tuple[int, ...]` for hash / equality stability.
- [x] **25 new tests in `tests/test_self_improve.py`** (total 147):
      - **`TestLoopConfigMultiSeedHoldout`** — default empty tuple,
        list/tuple normalization, zero entry rejected, collision
        with base_seed rejected, duplicates rejected,
        `resolved_holdout_seeds()` precedence (list > scalar >
        empty), `holdout_harness_config()` explicit-seed override
        and default-to-scalar.
      - **`TestSelfImproverMultiSeedHoldout`** — one record per
        seed in configured order, per-seed harness seeds reach
        the factory, overfit flagged independently per seed,
        list-wins-over-scalar precedence, all records written to
        JSONL ledger, scalar back-compat path unaffected, disable
        when both knobs unset.
      - **`TestCliSeedListParser`** — empty / whitespace / single
        / multiple / whitespace-tolerant / negative-accepted /
        non-integer-rejected / trailing-comma-skipped paths.
- [x] **Documentation updated**
  - `planning/SELF_IMPROVEMENT_LOOP.md`: §2 missing-pieces list
    extended; Phase 6 checklist updated; new §13 dated entry; the
    *Multi-seed hold-out* follow-up promoted from "open" to
    "shipped".
  - `doc/source/guide_benchmarking.rst`: new "Multi-seed hold-out"
    subsection with the aggregation rule, validation rules, CLI
    example, and programmatic example.
  - `doc/source/guide.rst`: quick-nav entry mentions the
    multi-seed hold-out.
  - `AGENTS.md`: self-improvement loop subsection lists the
    multi-seed feature with a run-the-loop bash example.

### Paired bootstrap for `statistical_accept` (2026-05-14)
- [x] **`statistical_accept(..., paired=...)` parameter** in
      `panobbgo/harness.py`.  When the harness keeps reps
      instance-aligned by index (the case under `--randomize`, where
      `derive_instance_seed` is keyed on `(base_seed,
      randomize_iteration, family, rep)`), the per-rep deltas are
      strongly positively correlated and a paired bootstrap is the
      statistically efficient sampler — it draws one shared resample
      index and applies it to both sides, mathematically equivalent to
      bootstrapping the per-rep delta vector `a_frac − b_frac`.
- [x] **Auto-detection** — `paired=None` (default) selects paired when
      at least one shared `(problem, strategy)` pair has matched rep
      counts, falls back to unpaired otherwise.  Asymmetric-rep edge
      cases the old unpaired sampler was written to handle keep their
      prior behaviour.  `paired=True` truncates mismatched reps to the
      common prefix; `paired=False` forces the historical
      independent-resample scheme.
- [x] **`StatisticalDecision.paired: bool`** — the result records which
      scheme actually fired so the JSON payload and `print_summary()`
      can report it (`bootstrap=paired|unpaired`).
- [x] **CLI flags** — mutually-exclusive `--paired` / `--unpaired` on
      both `benchmark_harness.py compare --statistical` and
      `scripts/self_improve.py run`.  Without either flag the
      auto-detect default fires, which is what randomized harness runs
      want by construction.
- [x] **`LoopConfig.paired`** in `panobbgo/self_improve.py` —
      `Optional[bool] = None`, forwarded into `statistical_accept` for
      every iteration's accept/reject decision.
- [x] **Why this matters** — inspecting the recent
      `planning/self_improve_ledger.jsonl` shows every rejection cited
      *"lower CI bound … ≤ 0 — improvement not statistically
      distinguishable from noise"* even on iterations whose composite
      delta was clearly positive.  That is the textbook symptom of an
      under-paired test: the unpaired bootstrap was discarding the
      paired-instance correlation that the randomized harness already
      builds in.  Micro-benchmark on five reps with constant +5-eval
      lift: paired CI collapsed to a point (width 0.0000); unpaired CI
      was 0.5400 wide and rejected the same genuine improvement.
- [x] **13 new tests** — `tests/test_harness_stats.py` (+11, total 33):
      paired-tighter-than-unpaired, paired unblocks acceptance,
      auto-detect picks paired when reps match, auto fallback to
      unpaired on mismatch, force-paired truncation, JSON round-trip
      of the new `paired` field, `print_summary` wording,
      empty-pair edge case, paired bootstrap reproducibility under a
      fixed seed, CLI integration of `--paired` flipping the verdict,
      and `--paired`/`--unpaired` mutually-exclusive argparse.
      `tests/test_self_improve.py` (+2, total 126):
      `LoopConfig.paired` defaults to `None` and accepts explicit
      `True`/`False`.
- [x] **Backwards compatible** — auto-detect default is no behaviour
      change for asymmetric-rep configurations, and
      `StatisticalDecision.paired` is a `False`-defaulted field so old
      ledger consumers parsing the JSON payload continue to work.
- [x] **Documentation updated**
  - `doc/source/guide_benchmarking.rst`: new "Paired vs unpaired
    bootstrap" subsection under Statistical acceptance rule with the
    scheme description, the worked numerical example, the CLI
    examples, and the auto-detect rule.
  - `doc/source/guide.rst`: quick-nav entry now mentions the paired
    bootstrap.
  - `planning/SELF_IMPROVEMENT_LOOP.md`: §6.1 paragraph on the
    paired-vs-unpaired distinction, a §13 entry recording the ship,
    and a "Next iteration ideas" entry on tightening `eps_accept`
    once the paired CI is the loop default.
  - `AGENTS.md`: Statistical rigor subsection now flags
    `--paired` / `--unpaired` and the auto-detect default.
  - This TODO entry.

### jSO Adaptive Differential Evolution (CEC 2017 Winner) (2026-05-15)
- [x] **`panobbgo/heuristics/jso.py`** — new :class:`JSO` heuristic,
      a direct subclass of :class:`~panobbgo.heuristics.lshade.LSHADE`
      that ports the Brest-Maučec-Bošković (CEC 2017) jSO refinement.
      jSO won the CEC-2017 single-objective bound-constrained
      competition and remains the canonical successor to L-SHADE in
      the adaptive-DE family.
- [x] **Three algorithmic refinements over L-SHADE**:
      - **Weighted current-to-pbest mutation** (``current-to-pbest-w/1``).
        The pbest direction is re-weighted by a phase-dependent
        ``F_w`` factor: ``0.7·F`` while ``progress < 0.2``, ``0.8·F``
        while ``progress < 0.4``, ``1.2·F`` afterwards.  The
        differential ``F · (x_r1 − x_r2)`` term keeps the unweighted
        scaling.
      - **Linear ``p_best`` schedule**.  ``p_best`` decreases linearly
        from ``p_best_max = 0.25`` to ``p_best_min = 0.125`` over the
        budget — broader pbest pool early, focused exploitation late.
      - **Cauchy-F clamping**.  When ``progress < 0.6``, sampled ``F``
        values above ``0.7`` are clamped to ``0.7`` to prevent
        pathologically large jumps in the early phase.
- [x] **Two memory tweaks** — ``M_F`` initialised to ``0.3`` (vs
      L-SHADE's ``0.5``) and ``M_CR`` to ``0.8`` (vs ``0.5``), plus a
      *frozen anchor bin* at ``H − 1`` permanently pinned at
      ``M_F = M_CR = 0.9``.  ``_update_memory`` advances the pointer
      through ``[0, H − 2]`` only — the anchor bin is still drawn
      from at sampling time so it stably contributes a "moderately
      greedy" parameter setting regardless of what the live
      success-history has learned.
- [x] **Asynchronous adaptation** — identical to L-SHADE.  jSO inherits
      the per-slot pending dict, generation-by-count update cadence,
      archive trimming, LPSR shrinking, and warm restart unchanged.
      Progress measurement uses ``len(strategy.results) / max_eval``
      (the same idiom L-SHADE uses for LPSR pacing) so the F-clamp
      and ``F_w`` schedules stay in lock-step with the population
      shrink.  When ``max_eval`` is unknown the schedules degrade to
      ``progress = 0.0`` (early-phase regime).
- [x] **Structural catalog integration** — :func:`default_structural_catalog`
      gains JSO as a twelfth ``add_heuristic`` candidate
      (``avoid_duplicates=True`` keeps the catalog from cluttering
      portfolios).  Both L-SHADE and jSO ship side-by-side so the
      bandit picks whichever DE-family variant wins on the current
      battery — exactly the kind of complementarity the structural
      catalog is designed to leverage.
- [x] **Kwarg catalog rules** — :func:`default_catalog` gains two
      jSO-specific rules: ``JSO.NP_init`` (``integer_add`` over
      ``[10, 60]`` with ``±5 / ±10`` deltas) and ``JSO.p_best_max``
      (``float_uniform`` over ``[0.15, 0.4]``).  Each fires only when
      a spec sets the matching kwarg explicitly.
- [x] **Impact** — A/B at quick mode (3 problems × 5 reps × 300 evals)
      against L-SHADE in the same Rewarding strategy:
      - Seed 42: ``Rewarding_LSHADE`` 0.791 / ``Rewarding_JSO`` **0.856**
        (mean **+0.065**).  Rosenbrock pair: 0.374 → **0.568** (success
        rate **40% → 80%**).
      - Seed 43: ``Rewarding_LSHADE`` **0.831** / ``Rewarding_JSO`` 0.801
        (mean -0.030).
      Each variant wins on one seed — the per-seed complementarity
      that motivates carrying both arms in the catalog.  The +0.194
      Rosenbrock spike on seed 42 is the literature-predicted win:
      jSO's weighted mutation term navigates the curved Rosenbrock
      valley faster than fixed-weight ``current-to-pbest/1``.
- [x] **Backwards compatibility** — strictly safe.  jSO is opt-in:
      not added to any default
      :func:`_make_quick_strategies` /
      :func:`_make_standard_strategies` /
      :func:`_make_full_strategies` spec, so existing CLI invocations
      and existing ledgers stay byte-identical.  L-SHADE itself is
      untouched.
- [x] **33 new tests** in `tests/test_heuristic_jso.py`:
      - **Construction validation** (8) — defaults match Brest 2017,
        custom kwargs, subclass-of-LSHADE invariant, ``H >= 2``
        requirement (anchor bin separation), ``p_best_max`` bounds,
        ``p_best_min`` bounds, ordering rule
        ``p_best_min <= p_best_max``.
      - **Memory anchor invariants** (5) — anchor frozen at
        construction, never written by ``_update_memory`` even
        after many cycles, pointer wraps over ``[0, H − 2]`` only,
        writable bin updated via Lehmer mean, no-success leaves
        memory unchanged.
      - **Schedule helpers** (5) — progress clipped to ``[0, 1]``,
        falls back to zero without budget, linear ``p_best``
        schedule, three-phase ``F_w`` schedule, phase-boundary
        inclusivity.
      - **Cauchy-F clamping** (3) — clamped at ``0.7`` in early
        phase, unclamped in late phase, F always in ``(0, 1]``
        regardless of phase.
      - **Initial population emission** (4) — ``NP_init`` points
        emitted, ``on_start`` re-stamps jSO defaults, NaN F/CR on
        initial trials, points stay inside the box.
      - **Generate-trial path** (2) — evolutionary trials emitted
        post-fill, better trial wins and archives parent.
      - **Restart behaviour** (3) — re-stamps jSO memory and
        anchor, ``center=None`` random fallback, before-start no-op.
      - **Smoke convergence** (1) — end-to-end no-regression on a
        quadratic.
      - **Registration** (3) — package re-export, structural catalog
        candidate pool, kwarg rules present in default catalog.

### Categorical Mutation Rule (`categorical_choice`) (2026-05-13)
- [x] **New `MutationRule(kind="categorical_choice", choices=...)`**
      in `panobbgo/self_improve.py`.  Fourth mutation kind alongside
      `log_uniform_perturb` / `integer_add` / `float_uniform`; picks
      uniformly from `choices` *excluding* the current value so the
      mutation always proposes a real change.  No-op samples
      (`new == old`) are eliminated by construction — important for
      two-choice toggles like `(True, False)` where a uniform sample
      would no-op half the time.
- [x] **`bounds` made optional** (defaults to `(0.0, 0.0)`) since
      categorical rules don't use it.  All shipped catalog rules pass
      `bounds` explicitly, so the change is byte-identical for every
      existing call site.
- [x] **Validation** in `MutationRule.__post_init__`:
      `len(choices) >= 2` (single-choice catalogues are forbidden —
      they would always no-op), no duplicate entries.  Numeric kinds
      keep their existing bounds check; categorical skips it.
- [x] **`MutationCatalog._mutate_value` extended** with the
      categorical branch.  Always excludes the current value from
      candidates; falls back to the full set if drift means
      `old not in choices`.  Returns the chosen value verbatim — no
      coercion through `_to_plain` (strings / bools / floats round-
      trip through the dataclass and the JSONL ledger naturally).
- [x] **Three categorical rules in `default_catalog()`**:
      - `PSO.topology` — `("gbest", "lbest")`, fires whenever a spec
        sets `topology` explicitly (typically after the structural
        catalog has added the lbest PSO variant).
      - `Sobol.scramble` — `(True, False)`, fires out-of-the-box on
        `BayesOpt_Sobol` which sets `scramble=True` explicitly.
      - `LSHADE.archive_factor` — `(0.0, 1.0, 2.6)`, dormant on the
        default battery (no spec sets `archive_factor` explicitly) but
        ready for opt-in.
- [x] **Bandit integration** — `_proposal_rule_key` maps a categorical
      rule to its own `(class_name, param_name, "categorical_choice")`
      arm, distinct from any numeric rule on the same kwarg slot.
      The Thompson sampler can therefore learn whether flipping a
      discrete knob is worthwhile, independently of whether tuning the
      same kwarg numerically is.
- [x] **Ledger round-trip** — `MutationProposal.to_dict` emits
      `rule_kind="categorical_choice"` and the literal categorical
      values in `old_value` / `new_value`.  Replay via
      `_proposal_rule_key` recovers the bandit arm losslessly.
- [x] **13 new tests in `tests/test_self_improve.py`** (total 122):
      - **`TestMutationRule`** — categorical constructs, two-choice
        minimum enforced, empty choices rejected, duplicate choices
        rejected, bounds ignored for categorical.
      - **`TestMutationCatalog`** — `default_catalog` ships
        PSO/Sobol/LSHADE categorical rules; sampling always returns a
        value distinct from `old`; two-choice toggle deterministically
        flips; out-of-set drift handled (all choices reachable);
        rationale and `rule_kind` formatting.
      - **`TestApplyMutation`** — categorical string round-trip;
        categorical bool round-trip preserves `isinstance(bool)`.
      - **`TestAdaptiveMutationSampler`** — categorical rule occupies
        its own bandit arm distinct from a numeric rule on the same
        `(class, param)` slot.
      - **`TestStructuralRuleKey`** — `_proposal_rule_key` maps
        categorical to `(class, param, "categorical_choice")`.
- [x] **Backwards compatibility** — strictly safe.  All shipped
      catalog rules use the numeric kinds and pass `bounds` explicitly;
      the new defaulted `choices` field and the new defaulted `bounds`
      are no-ops for existing callers.  Ledger consumers that filter
      on `rule_kind` simply see one extra kind they may ignore.

### COBYQA Derivative-Free Trust-Region Local Optimizer (2026-05-12)
- [x] **`panobbgo/heuristics/cobyqa.py`** — new :class:`COBYQA`
      heuristic, a subprocess-backed adapter around
      ``scipy.optimize.minimize(method="COBYQA")``.  COBYQA
      (*Constrained Optimization BY Quadratic Approximations*,
      Ragonneau-Zhang 2023) is the modern Powell-family successor to
      BOBYQA / COBYLA / NEWUOA / LINCOA: it maintains an interpolation
      set of ``2·n + 1`` points and fits an adaptive *quadratic
      model* of the objective inside a trust region, dominant on
      smooth / near-smooth local refinement.  The asynchronous
      wrapping pattern mirrors :class:`~panobbgo.heuristics.lbfgsb.LBFGSB`:
      a daemon ``spawn`` subprocess drives the synchronous COBYQA
      solver, requests ``f(x)`` over a pipe, and the main thread
      relays the projected point through Panobbgo's evaluator and
      pipes the penalty value back.
  - **Why it matters.**  Before this entry,
    :class:`~panobbgo.heuristics.nelder_mead.NelderMead` was the
    *only* generic derivative-free local refinement step in the
    portfolio; :class:`~panobbgo.heuristics.lbfgsb.LBFGSB` needs a
    finite-difference gradient approximation that breaks on noisy
    objectives, and Nelder-Mead's simplex updates are not
    curvature-aware, so it converges slowly on ill-conditioned
    valleys (Rosenbrock-like landscapes).  COBYQA gives the loop
    driver a *derivative-free **and** curvature-aware* local
    refinement arm the bandit can choose between.  Picking COBYQA
    over the older BOBYQA library keeps the dependency surface
    unchanged — COBYQA ships built-in with ``scipy.optimize.minimize``
    since scipy 1.14 and is the literature-recommended replacement.
- [x] **Configuration knobs** — ``initial_tr_radius`` (auto-derives
      to ``0.1 · max(box_width)`` when ``None``), ``final_tr_radius``
      (default ``1e-6``), ``maxfev`` (``None`` lets the strategy
      budget terminate), ``scale`` (default ``True`` — maps the box
      to ``[-1, 1]`` to keep the interpolation geometry
      well-conditioned for boxes whose axes span very different
      magnitudes).  Construction-time validation rejects negative /
      zero / NaN radii, the ``final >= initial`` ordering, and
      non-integer / non-positive ``maxfev``.
- [x] **Restart support** — :meth:`COBYQA.on_restart(center, reason)`
      tears down the current subprocess (terminate → join → kill on
      timeout) and respawns a fresh COBYQA solve seeded at the
      *clipped* suggested center.  When the strategy is stopped the
      restart is a no-op so the loop never spawns a process during
      shutdown.
- [x] **Structural catalog wiring** — :func:`default_structural_catalog`
      gains COBYQA as an eleventh ``add_heuristic`` candidate
      (``avoid_duplicates=True`` keeps the catalog from cluttering
      portfolios that already include it).
- [x] **Kwarg mutation rules** — :func:`default_catalog` gains two
      rules so the loop driver can also retune
      ``COBYQA.initial_tr_radius`` (``log_uniform_perturb`` over
      ``[0.01, 1.0]``, ``log_step=0.15``) and
      ``COBYQA.final_tr_radius`` (``log_uniform_perturb`` over
      ``[1e-8, 1e-4]``, ``log_step=0.25``).  Both fire only when a
      spec explicitly sets the matching kwarg.
- [x] **Backwards compatibility — strictly safe.**  COBYQA is opt-in:
      it is not added to any default
      :func:`_make_quick_strategies` / :func:`_make_standard_strategies` /
      :func:`_make_full_strategies` spec.  Existing CLI invocations
      and existing ledgers stay byte-identical.
- [x] **Quick A/B impact at ``--quick`` (3 problems × 3 reps × 75
      evaluations)**, comparing the same Rewarding strategy with
      NelderMead vs COBYQA vs both as the local optimizer:
  - Seed 42 — ``NM`` 0.665 / ``COBYQA`` **0.769** (+0.104) /
    ``NM+COBYQA`` 0.699.  Rosenbrock success rate jumps from
    **0/3 with NM** to **2/3 with COBYQA**.
  - Seed 43 — ``NM`` **0.864** / ``COBYQA`` 0.714 / ``NM+COBYQA``
    0.753.  NM happens to win Rosenbrock on this seed.
  - The two seeds together demonstrate complementarity — each
    local optimizer wins on one of them; the categorical
    Rosenbrock success-rate upgrade (0/3 → 2/3) confirms the
    expected property: COBYQA's curvature-aware quadratic model
    crosses the narrow curved valley that Nelder-Mead misses.
- [x] **30 tests in `tests/test_heuristic_cobyqa.py`** —
      construction validation (11 — invalid initial / final TR
      radii / NaN / ordering rule / maxfev type and value), initial
      TR auto-resolution (4), subprocess lifecycle (2 — spawn,
      force-kill), pipe wiring (4 — penalty routed, foreign-who
      ignored, EOF exit, output log), restart behaviour (4 —
      respawn, ``center=None`` fallback, out-of-box clip, stopped
      no-op), registration (3 — package, structural catalog,
      kwarg rules), end-to-end smoke directly through scipy.
- [x] **Documentation updated**
  - `planning/SELF_IMPROVEMENT_LOOP.md`: new §12 dated entry; the
    *BOBYQA / NEWUOA local optimizer* "Next iteration idea" was
    closed and replaced with COBYQA follow-up tickets
    (constraint-aware variant, warm-start interpolation reuse,
    categorical mutation rule for ``scale``).
  - `doc/source/guide.rst`: quick-nav entry mentions COBYQA.
  - `doc/source/guide_benchmarking.rst`: structural catalog
    candidate-pool list extended with ``LSHADE`` and ``COBYQA``.

### Hold-Out Validation Set for Self-Improvement Loop (2026-05-08)
- [x] **New `panobbgo.self_improve.LoopHoldoutRecord`** — third record
      type alongside `LoopIterationRecord` / `LoopGuardRecord`; closes
      the §10 "Hold-out validation set" item in
      `planning/SELF_IMPROVEMENT_LOOP.md`.  At the end of every loop
      run, the seed and final-top of the accepted ladder are
      re-measured on instances drawn from an *independent*
      `base_seed` SHA-256 stream, and the shrinking-gap drift is
      reported on the JSONL ledger as `record_type="holdout"`.
  - **Why it matters.**  The anti-cherry-pick guard catches drift
    *within* the training base_seed family — it varies only
    `randomize_iteration` and keeps `HarnessConfig.seed` constant.  A
    mutation that overfits to peculiarities of the training base_seed
    family slips through silently because the guard's "fresh"
    instances are still drawn from the same SHA-256 stream.  The
    hold-out re-measures on a *completely* independent base_seed, so
    an overfit ladder is exposed by ``drift < -eps_overfit``.
- [x] **`LoopConfig` knobs** — `holdout_base_seed` (0 = disabled),
      `holdout_iterations` (5 by default), `holdout_iteration_offset`
      (0 by default), `holdout_eps_overfit` (0.05 by default).
      `LoopConfig.__post_init__` rejects `holdout_base_seed ==
      base_seed` (other than 0) — equal values would collapse the
      check to a glorified guard with offset 0.
- [x] **`LoopConfig.holdout_harness_config()`** — sibling to
      `harness_config()` that swaps the `seed` field to
      `holdout_base_seed`; every other knob (mode, reps, budget,
      `strategies_override`) matches the training run.
- [x] **`SelfImprover._run_holdout()`** + helpers
      (`_holdout_enabled`, `_measure_holdout`, `_print_holdout`).
      Skipped silently when (a) disabled, (b) the loop ran zero
      iterations, or (c) `randomize=False` (the fixed battery is
      unaffected by base_seed, so a hold-out check would be no
      signal).
- [x] **New public entry-point `SelfImprover.run_full()`** — returns
      `(iter_records, guard_records, holdout_records)` for callers
      that want all three signals.  `SelfImprover.run()` keeps its
      original return type for backward compatibility;
      `run_with_guard_records()` returns just the first two.
- [x] **CLI flags** `--holdout-base-seed`, `--holdout-iterations`,
      `--holdout-iteration-offset`, `--holdout-eps-overfit`,
      `--fail-on-overfit` (exits `3` on a flagged ladder) on
      `scripts/self_improve.py run`; `summary` distinguishes
      hold-out records and prints drift / overfit verdict.
- [x] **17 new tests in `tests/test_self_improve.py`** (total 97):
      - **`TestLoopConfigHoldout`** — defaults, negative-iterations
        validation, negative-eps validation, equal-base-seed
        rejection, zero-zero edge case, `holdout_harness_config`
        propagation.
      - **`TestSelfImproverHoldout`** — disabled-by-default,
        skipped when `randomize=False`, skipped on zero iterations,
        seed-only ladder records zero drift, hold-out uses the
        independent base_seed for measurement, overfit flag fires
        when gap collapses, no flag when gap holds, ledger writes
        `record_type='holdout'` line, `run()` keeps backward
        compatibility.
      - **`TestLoopHoldoutRecord`** — `to_dict` round-trip with JSON
        serialisation.
- [x] **Documentation updated**
  - `planning/SELF_IMPROVEMENT_LOOP.md`: §2 missing-pieces list
    refreshed; §10 hold-out item resolved; Phase 6 checklist updated;
    new §12 dated entry; "Next iteration ideas" replaced with
    multi-seed hold-out / auto-rollback follow-up tickets.
  - `doc/source/guide_benchmarking.rst`: new "Hold-out validation
    set" subsection with algorithm, CLI examples, programmatic
    example, and the independence-from-the-guard note.
  - `doc/source/guide.rst`: quick-nav entry mentions the hold-out.
  - `AGENTS.md`: self-improvement loop subsection lists the
    hold-out feature with run-the-loop bash example.

### PSO adaptive inertia (Shi-Eberhart 1998) (2026-05-07)
- [x] **`panobbgo/heuristics/pso.py`** — :class:`PSO` gains an opt-in
      ``w_end`` keyword argument.  When set, a new
      ``_current_inertia()`` method linearly anneals the inertia from
      ``self.w`` down to ``self.w_end`` paced by
      ``len(strategy.results) / strategy.config.max_eval``; otherwise
      the inertia is constant at ``self.w``, reproducing the prior
      Clerc-Kennedy behaviour byte-for-byte.  Falls back to constant
      ``w`` whenever the budget is unknown, zero, or non-numeric.
- [x] **Catalog rules** — :func:`default_catalog` gains
      ``MutationRule`` entries for ``PSO.w`` (``float_uniform`` over
      ``[0.4, 0.95]``) and ``PSO.w_end`` (``float_uniform`` over
      ``[0.2, 0.6]``) so the loop driver can tune the inertia
      schedule once a spec opts in.
- [x] **6 new tests in `tests/test_heuristic_pso.py`**: default
      ``w_end`` is ``None``; finiteness validation; constant-``w``
      short-circuit; missing-results fall-back; linearly-decreasing
      schedule at four progress points; ``max_eval = 0`` fall-back;
      and a catalog test asserting ``PSO.w`` / ``PSO.w_end`` rules
      are present.
- [x] **Documentation updated** — module docstring gets a new
      "Adaptive inertia" section and a Shi-Eberhart reference;
      ``doc/source/heuristics.rst`` and
      ``doc/source/guide_architecture.rst`` mention ``w_end``.

### PSO ring (`lbest`) topology variant (2026-05-07)
- [x] **`panobbgo/heuristics/pso.py`** — :class:`PSO` gains a
      ``topology: str = "gbest"`` argument and a ``k_neighbors: int = 2``
      half-width.  ``"gbest"`` keeps the canonical Kennedy-Eberhart
      1995 fully-connected swarm (default, byte-identical to the
      2026-05-05 ship); ``"lbest"`` switches every particle's social
      attractor to the best ``pbest`` in a wrap-around *ring* of width
      ``2·k_neighbors + 1`` centred on the particle's index — slower
      information diffusion, stronger multimodal exploration
      (Kennedy & Mendes, CEC 2002).
- [x] **Two new helpers** — ``_ring_neighbors(i)`` returns the
      wrap-around index list; ``_social_best_idx(i)`` returns the
      per-particle social attractor (collapsing to ``_gbest_idx`` for
      gbest, scanning the ring for lbest).  ``_generate_next``
      consults ``_social_best_idx`` exactly where it used
      ``_gbest_idx`` before, so the velocity-update / clamp /
      projection paths are shared between topologies.
- [x] **Structural catalog integration** —
      :func:`default_structural_catalog` ships two PSO entries:
      ``(PSO, {"NP": 20})`` (gbest, default) and
      ``(PSO, {"NP": 20, "topology": "lbest", "k_neighbors": 2})``.
      Both share ``cls = PSO`` so ``avoid_duplicates=True`` still
      installs only one PSO per strategy; the catalog samples
      uniformly between the two when PSO is not yet present and
      skips both afterwards.
- [x] **Backwards compatibility** — ``topology`` defaults to
      ``"gbest"`` so existing PSO instances retain their prior
      behaviour bit-for-bit.  Existing kwarg rule
      (``MutationRule(class_name="PSO", param_name="NP", …)``) and
      bandit ``_proposal_rule_key`` are unchanged.
- [x] **A/B at `--quick`** (3 problems × 5 reps × 150 evals): seed
      42 → gbest 0.183 / **lbest 0.288**; seed 43 → **gbest 0.296** /
      lbest 0.181.  The two topologies are *complementary* (each
      wins on one seed), exactly the literature's prediction —
      shipping both gives the bandit a finer-grained choice without
      regressing the gbest path.
- [x] **13 new tests in `tests/test_heuristic_pso.py`** (total 50):
      construction validation (default topology / lbest construction
      / invalid topology / invalid k_neighbors type / value), ring
      wrap-around correctness, ring size invariant, lbest social
      attractor uses ring (not the global best), gbest social
      attractor degenerates to ``_gbest_idx``, lbest returns ``None``
      before any neighbour pbest exists, lbest velocity clamp
      invariant, lbest end-to-end smoke convergence on a quadratic,
      and the structural catalog now ships both gbest and lbest PSO
      entries.
- [x] **Documentation updated** — ``planning/SELF_IMPROVEMENT_LOOP.md``
      §12 logs the iteration and the PSO follow-ups list now drops
      "Topology variants" (shipped) and adds "Random / Von Neumann
      topologies" + "Categorical / topology mutation rule" as next
      ideas.  ``doc/source/heuristics.rst``,
      ``doc/source/guide_architecture.rst``,
      ``doc/source/guide_research.rst``, and
      ``doc/source/guide_benchmarking.rst`` describe both topologies.

### PSO (Particle Swarm Optimization) heuristic (2026-05-05)
- [x] **`panobbgo/heuristics/pso.py`** — new asynchronous PSO heuristic
      with the canonical Clerc–Kennedy (2002) constriction-coefficient
      parameters (``w = χ ≈ 0.7298``, ``c1 = c2 ≈ 1.49618``).  Each
      particle carries a position, velocity, and personal best; the
      velocity update pulls toward both personal best and global best
      with random per-component weights and is clamped per-dimension to
      a configurable fraction of the box range to prevent explosion.
- [x] **Async event-loop integration** — follows the same pattern as
      :class:`DifferentialEvolution`: each in-flight trial carries a
      unique ``who`` id; ``on_new_results`` matches the id back to its
      particle slot, updates pbest/gbest, and emits the next velocity-
      based trial.  No per-tick busy waiting.
- [x] **IPOP-style warm restart** — ``on_restart(center, reason)``
      drops in-flight trials, scatters particles in a velocity-clamp
      ball around the new center, and resets the global memory while
      the strategy keeps its accumulated history.
- [x] **Catalog integration** — added a kwarg rule in
      :func:`default_catalog` for ``PSO.NP`` (swarm size, range
      ``[8, 60]``, ±4 / ±8 deltas) and added ``PSO`` to the
      ``add_heuristic`` candidate pool in
      :func:`default_structural_catalog`.
- [x] **24 new tests in `tests/test_heuristic_pso.py`** — construction
      validation (8), initial-swarm emission and shape (3), pbest /
      gbest update + follow-up trial (5), velocity clamp invariant (1),
      restart behaviour (3), an end-to-end smoke run on a quadratic, and
      registration tests for ``panobbgo.heuristics`` and the structural
      catalog.
- [x] **Documentation updated** — ``doc/source/heuristics.rst`` and
      ``doc/source/guide_architecture.rst`` (Population-based section)
      now describe PSO; ``doc/source/guide_benchmarking.rst`` mentions
      it among the structural-catalog candidates;
      ``planning/SELF_IMPROVEMENT_LOOP.md`` §12 logs the iteration.

### Strategy Portfolio Composition (`StructuralMutationRule`) (2026-05-03)
- [x] **`panobbgo.self_improve.StructuralMutationRule`** — new dataclass
      that joins :class:`MutationRule` as a first-class catalog entry.
      Closes the §7.2 *Strategy portfolio composition* item in
      `planning/SELF_IMPROVEMENT_LOOP.md` — the loop driver could
      previously only retune existing kwargs; now it can also reshape
      a strategy's heuristics list.
- [x] **Two ops** — ``add_heuristic`` (append a class from a curated
      ``candidate_classes`` pool, ``avoid_duplicates`` by default
      skips classes already present) and ``drop_heuristic`` (remove
      one heuristic, optionally restricted via ``droppable_classes``
      with a ``min_heuristics`` post-drop safety floor — default ``2``).
- [x] **`MutationProposal` extension** — keyword-only ``op`` and
      ``structural_kwargs`` fields, default ``None``.  Kwarg proposals
      serialise byte-identically to before; structural proposals get
      the two extra keys via :meth:`to_dict`.
- [x] **`apply_mutation` dispatch** — branches on ``proposal.op``;
      ``add_heuristic`` recovers the class object via the spec's
      existing classes first and ``panobbgo.heuristics`` package as
      fallback; ``drop_heuristic`` removes the first match and refuses
      to leave the spec empty.
- [x] **Adaptive sampler integration** — ``_proposal_rule_key``
      collapses both structural ops onto a single arm
      (``("*", op, "structural")``) so cold-start variance stays
      bounded.  Per-class arms are listed as a follow-up under "Next
      iteration ideas".
- [x] **`default_structural_catalog()`** — extends
      :func:`default_catalog` with one ``add_heuristic`` rule (pool of
      seven safe generators: Random/Nearby/NelderMead/Center/
      LatinHypercube/Sobol/Extremal) and one ``drop_heuristic`` rule
      (``min_heuristics=2``).  Both at probability ``0.3`` so the
      structural rules don't dominate kwarg perturbations.
- [x] **CLI flag** — `scripts/self_improve.py run --structural`
      switches the loop to the structural catalog.  Off by default so
      existing CLI invocations are byte-identical.
- [x] **29 new tests in `tests/test_self_improve.py`** (total 92):
      rule validation, applicable-hits enumeration (add / drop /
      ``avoid_duplicates`` / ``droppable_classes`` / ``min_heuristics``
      floor / strategy_pattern filter), proposal serialisation, the
      apply-side dispatch (add appends, drop removes, missing class
      raises, empty-strategy refusal, fallback-import path),
      :func:`_proposal_rule_key` collapse for structural ops, the
      Thompson sampler bucketing structural history into one arm, the
      `default_structural_catalog()` factory shape, and an end-to-end
      loop run that accepts a structural drop on a fake harness.
- [x] **Documentation updated**
  - `planning/SELF_IMPROVEMENT_LOOP.md`: Phase 6 checklist marks §7.2
    done; §7 item 2 annotated as shipped; "Next iteration ideas"
    rewritten with per-class-arms, analyzer add/drop, and
    strategy-class-swap follow-ups; new §12 dated entry under the
    iteration log.
  - `doc/source/guide_benchmarking.rst`: new "Strategy portfolio
    composition (§7.2)" subsection covering the ops, safety floors,
    Thompson-sampler bucketing, CLI invocation, and a programmatic
    custom-catalog example.
  - `doc/source/guide.rst`: top-of-page navigation row for the
    benchmarking guide now lists the structural mutations.
  - `AGENTS.md`: structural catalog mentioned in the loop checklist
    and the CLI example block.
  - This TODO entry.

### Stratified Dimension Sampling for Multi-Dim Families (2026-05-02)
- [x] **`panobbgo.harness_randomized.ProblemFamily.stratify_dims`** —
      new bool field (default ``True``) that enables cyclic dim
      stratification for multi-dim families.  Closes the §10 "Composite
      score stability across dimension sampling" open question in
      `planning/SELF_IMPROVEMENT_LOOP.md`.
- [x] **`ProblemFamily.stratified_dim_for_rep(rep)`** — returns
      ``dim_choices[rep % len(dim_choices)]``, so any contiguous block
      of ``k`` reps covers every declared dim exactly once.
- [x] **`ProblemFamily.sample_instance(rng, dim=None)`** — now accepts
      an optional dim override.  Stratified callers pin the dim
      explicitly, so the rng's ``choice`` slot is not consumed and the
      remaining stream (translation, rotation, scaling, noise seed)
      stays comparable to a single-dim family at the same seed.
- [x] **`RandomizedProblemSpec.create_problem_for_rep(rep)`** — calls
      ``stratified_dim_for_rep(rep)`` for multi-dim families with
      ``stratify_dims=True`` and falls back to the rng draw otherwise.
      ``last_sampled_params()`` now exposes a ``stratified_dim: bool``
      flag for ledger introspection.
- [x] **Why it matters.** Without stratification, a family with
      ``dim_choices = (2, 5, 10)`` and 5 reps could draw three ``dim=2``
      instances on iteration 5 and three ``dim=10`` instances on
      iteration 6.  Higher-dim instances are systematically harder, so
      a per-iteration composite delta picks up dim-mix noise on top of
      the actual signal of the underlying mutation, polluting the
      bootstrap CI in :func:`panobbgo.harness.statistical_accept`.
      Cyclic stratification eliminates that noise source by construction
      without changing the per-iteration eval count.
- [x] **Backwards compatibility.** The entire default battery from
      :func:`make_default_families` uses ``dim_choices=(2,)``, so
      stratification is a no-op for byte-level reproducibility of the
      current standard mode.  ``stratify_dims=False`` recovers the
      legacy uniform-draw behaviour for users replicating an old ledger.
- [x] **16 new tests in `tests/test_harness_randomized.py`** (total 68):
      cyclic schedule correctness, balance over a complete cycle,
      imbalance bound on partial cycles, single-dim no-op, dim-override
      validation, rng-stream invariance proof (override does not consume
      the choice slot), end-to-end :class:`RandomizedProblemSpec` round
      trip, ``last_sampled_params`` flag round trip, and the contract
      that default families remain unchanged.
- [x] **Documentation updated**
  - `planning/SELF_IMPROVEMENT_LOOP.md`: §10 stability item resolved;
    Phase 6 checklist updated; new §12 dated entry under iteration log;
    stratification clause added to the §4.1 transforms table; Next
    iteration ideas now lists a "multi-dim default battery" follow-up.
  - `doc/source/guide_benchmarking.rst`: new "Stratified dimension
    sampling" subsection with the cyclic schedule example and the
    ``stratify_dims=False`` escape hatch.
  - `AGENTS.md`: stratification clause in the parametric battery
    section.
  - This TODO entry.

### Adaptive Mutation Sampler (Thompson Sampling) for Self-Improvement Loop (2026-05-01)
- [x] **New `panobbgo.self_improve.AdaptiveMutationSampler`** — Thompson-
      sampling bandit over per-rule Beta posteriors; closes the §10
      "Adaptive mutation sampler" item in
      `planning/SELF_IMPROVEMENT_LOOP.md`.  Each
      :class:`MutationRule` becomes one arm whose reward is "iteration was
      accepted"; on `sample()` the sampler draws one variate from
      ``Beta(prior_alpha + n_accepts, prior_beta + n_attempts -
      n_accepts)`` per applicable rule and picks the arg-max.  Inside the
      chosen rule, hits are still selected uniformly (which spec / which
      slot), exactly like the catalog's uniform sampler.
  - **Why it matters.** The uniform catalog sampler shipped in Phase 5
    wastes iterations on rules that never produce accepts.  Thompson
    sampling concentrates probability on empirically winning rules
    while still exploring under-tried rules — the standard fix for the
    productivity gap of multi-armed bandit problems.  Cold-start
    equivalence to uniform (Beta(1, 1) ≡ U(0, 1), arg-max of i.i.d.
    uniforms is uniform) makes the upgrade strictly safe.
  - **History persistence.** `prime_from_ledger(path)` replays
    iteration records from a prior JSONL ledger so the bandit resumes
    with all the meta-knowledge of which rules have worked so far —
    directly supports unattended multi-hour loops.
- [x] **`MutationRuleStats` dataclass + public `RuleKey` alias** —
      JSON-serialisable per-rule accept/attempt history bucketed by
      ``(class_name, param_name, rule_kind)``.
- [x] **`LoopConfig` knobs** — `adaptive_sampling`,
      `adaptive_prior_alpha`, `adaptive_prior_beta`,
      `adaptive_prime_from_ledger`; all default to off / symmetric prior
      so existing CLI invocations behave identically.  Negative or zero
      priors raise at validation time.
- [x] **`SelfImprover` integration** — accepts an explicit `sampler=`
      keyword for tests; otherwise constructs the sampler from
      `LoopConfig` when `adaptive_sampling=True`.  After each iteration's
      accept/reject decision, the driver calls
      ``sampler.record_outcome()`` so future samples are biased toward
      winning rules.
- [x] **CLI flags** `--adaptive`, `--adaptive-prior-alpha`,
      `--adaptive-prior-beta`, `--adaptive-prime-from-ledger` on
      `scripts/self_improve.py run`; the run summary prints per-rule
      accept rates when the sampler is enabled.
- [x] **23 new tests in `tests/test_self_improve.py`** (total 63):
      invalid priors, cold-start uniform behaviour, arg-max bias toward
      winning rules after biased training, record-outcome correctness
      including no-op after `None` sample / skip iterations, ledger
      priming (with guards / skips correctly ignored), `MutationRuleStats`
      round-trip, `SelfImprover` integration with the `sampler=`
      override, the `adaptive_prime_from_ledger` flag, and
      `LoopConfig` validation.
- [x] **Documentation updated**
  - `planning/SELF_IMPROVEMENT_LOOP.md`: §6 Phase-6 checklist marked
    shipped; new §12 dated entry under iteration log;
    "Next iteration ideas" reduced and gains a hierarchical-bandit
    follow-up ticket.
  - `doc/source/guide_benchmarking.rst`: new "Adaptive mutation sampler
    (§10)" subsection with algorithm, CLI examples, programmatic
    example, cold-start equivalence proof sketch.
  - `AGENTS.md`: self-improvement loop subsection lists the adaptive
    sampler with run-the-loop bash example.
  - This TODO entry.

### Sobol' Quasi-Random Initial Design Heuristic (2026-04-27)
- [x] **New `panobbgo/heuristics/sobol.py`** — `Sobol` heuristic, a one-shot
      low-discrepancy quasi-random sampler that produces space-filling initial
      designs.
  - Backed by `scipy.stats.qmc.Sobol` (no new dependency).
  - Owen-scrambled by default — different seeds produce statistically
    independent point sets so per-rep variance is meaningful, while the
    low-discrepancy property is preserved within each draw.
  - Uses ``random_base2`` when ``n`` is a power of two for the sharpest
    balance properties; falls back to ``random(n)`` otherwise.
  - Pure standalone heuristic following the `LatinHypercube` pattern; no
    event-system hooks needed.
- [x] **`BayesOpt_Sobol` strategy** added to standard harness mode pairing
      ``Sobol(n=16, scramble=True)`` with ``GaussianProcessHeuristic``,
      ``Nearby``, ``NelderMead`` — head-to-head with the existing
      ``BayesOpt_GP`` (which uses ``LatinHypercube``).
- [x] **Mutation rule for Sobol.n** added to the self-improvement loop's
      ``default_catalog()`` (4-step increments inside ``[4, 64]``) so the
      loop driver can also tune the parameter.
- [x] **Measured impact** (standard mode, 5 reps × 7 problems, budget 200):
      mean per-pair score ``BayesOpt_Sobol = 0.314`` vs
      ``BayesOpt_GP = 0.191`` (``+0.123``); wins on 5 / 7 problems, ties on
      Griewank with smaller best-distance.
- [x] **16 tests in `tests/test_heuristic_sobol.py`** — construction
      validation, scaling/sampling primitives, low-discrepancy proxy vs
      uniform sampling, scramble-determinism vs seed-reproducibility,
      ``on_start`` emit path, higher-dimensional problems, registration
      check.
- [x] **Documentation updated**
  - `doc/source/heuristics.rst`: ``Sobol`` listed alongside ``LatinHypercube``.
  - `doc/source/guide_architecture.rst`: Sobol added to the "Space-filling"
    heuristic group.
  - `doc/source/guide_usage.rst`: portfolio table now mentions Sobol; new
    "Bayesian optimization with Sobol' initial design" worked example.
  - `doc/source/guide_benchmarking.rst`: standard-mode strategy count
    bumped from 6 to 7 in the modes table.
  - `planning/SELF_IMPROVEMENT_LOOP.md`: §12 iteration log entry.
  - `AGENTS.md`: heuristics list updated.
  - This TODO entry.

## Setup & Modernization (Completed)
- [x] Restructure repository: Move `panobbgo.lib` to `panobbgo/lib`.
- [x] Modernize `setup.py` / Create `pyproject.toml`.
- [x] Update dependencies in `requirements.txt`.
- [x] Replace `nose` with `pytest`.
- [x] Update imports after restructuring.
- [x] Run and fix existing tests.
- [x] Add type hinting where possible.
- [x] Update `README.md` with new installation and usage instructions.
- [x] Setup CI/CD (GitHub Actions) - *optional but recommended*.

## Recent Improvements

### Anti-Cherry-Pick Guard for Self-Improvement Loop (Phase 6.3) (2026-04-26)
- [x] **New `LoopConfig.guard_interval` / `guard_eps_ladder` /
      `guard_iteration_offset`** in `panobbgo/self_improve.py` —
      implements §6.3 of `planning/SELF_IMPROVEMENT_LOOP.md`.  Every
      ``guard_interval`` iterations the loop re-measures the top of
      the accepted ladder on a *fresh* randomized seed and rolls back
      if the composite drifts more than ``guard_eps_ladder`` below the
      stored ``last_validated_score``.  The seed entry is the trusted
      fallback and is never popped.
  - **Why this matters.**  Even with the parametrically randomized
    battery, a sequence of "lucky" instance draws can inflate
    per-iteration ``after`` scores enough to clear the bootstrap CI.
    The guard catches this drift by validating the ladder against an
    independent instance stream (``randomize_iteration = iteration +
    guard_iteration_offset``).
  - **Disabled by default** (``guard_interval = 0``) for backward
    compatibility; bump to ``5`` or ``10`` for unattended runs.
- [x] **New `LadderEntry` and `LoopGuardRecord` types** —
      `LadderEntry` snapshots ``(iteration, specs,
      last_validated_score, proposal)``; `LoopGuardRecord` records the
      outcome of one guard check and is written to the same JSONL
      ledger with ``record_type = "guard"``.  `LoopIterationRecord`
      gains ``record_type = "iteration"`` for symmetry.
- [x] **CLI flags** `--guard-interval`, `--guard-eps-ladder`,
      `--guard-iteration-offset` on `scripts/self_improve.py run`;
      `summary` now distinguishes iteration and guard records and
      prints rollback details.
- [x] **40 tests in `tests/test_self_improve.py`** — comprehensive
      coverage of `MutationRule` validation, `MutationCatalog` sampling
      (log-uniform / integer-add / float-uniform), `apply_mutation`
      immutability, `LoopConfig` validation, end-to-end
      `SelfImprover` runs with a faked harness (zero iterations, skip,
      accept, reject, STOP sentinel), the new guard
      (cadence, no-rollback when stable, rollback on drift, offset
      iteration id, seed not popped), ledger round-trip, and dataclass
      serialisation.  Phase 5 shipped without tests; this PR fills
      that gap as well.
- [x] **Documentation updated**
  - `planning/SELF_IMPROVEMENT_LOOP.md`: §6.3 marked shipped, §2
    "what's missing" list updated, Phase 5 / Phase 6 checklists
    refreshed, new §12 "Next iteration ideas" with Adaptive Mutation
    Sampler, Stratified Dimension Sampling, Strategy Portfolio
    Composition, and Hold-out Validation Set as carry-over tickets.
  - `doc/source/guide_benchmarking.rst`: new "Anti-cherry-pick guard
    (§6.3)" subsection with algorithm, programmatic example, and
    safety-rail rationale.
  - `doc/source/guide.rst`: quick-nav entry mentions the loop driver
    and guard.
  - `AGENTS.md`: self-improvement loop subsection now lists the loop
    driver (shipped) and the guard (shipped) with run-the-loop bash
    examples.
  - This TODO entry.

### Parametrically Randomized Problem Battery (Self-Improvement Loop Phase 3) (2026-04-22)
- [x] **New `panobbgo/harness_randomized.py`** — Phase 3 of the
      self-improvement loop: the fixed harness battery is replaced with a
      parametric one that samples fresh transformed instances per rep,
      turning `composite_score` into a Monte-Carlo estimate of *expected*
      performance on a problem family.  Without this, an autonomous
      improvement loop would over-fit to specific instances.
  - `TransformedProblem(Problem)` — wraps a base problem with the
    composition `y = Q · Λ · (x - x*) + y_base_star` plus optional
    additive Gaussian noise; by construction `f_new(x*) = f_opt` so the
    existing harness metrics (`func_distance`, `ert`, `composite_score`)
    work unchanged.
  - `ProblemFamily` — declarative spec with per-family
    `supported_transforms` capability flags (`translate`, `rotate`,
    `scale`, `noise`), `log10_cond_max`, `dim_choices`, and tolerance.
  - `RandomizedProblemSpec(ProblemSpec)` — bridge between the family and
    the harness; `create_problem_for_rep(rep)` samples a fresh instance
    from the family, and records the sampled parameters for ledger
    output via `last_sampled_params()`.
  - Haar-uniform orthogonal sampler via QR + Mezzadri sign correction
    (dependency-free).
  - Geometric log-uniform diagonal scaling with configurable condition
    ceiling.
  - Interior-point translation (default 15% per-side margin) so the
    optimum never sits on a box boundary.
  - SHA-256-derived instance seed via
    `derive_instance_seed(base_seed, iteration_id, family_name, rep)`
    — within one iteration `before`/`after` runs see identical
    instances; across iterations they intentionally differ.
- [x] **Default families**: `Rastrigin_family`, `Ackley_family`,
      `Rosenbrock_family`, `DeJong_family`.  Schwefel and Griewank are
      intentionally excluded — rotation would push `y` off their
      sensible domain.
- [x] **`HarnessConfig.randomize` + `HarnessConfig.randomize_iteration`**
      plus `BenchmarkHarness.get_problems()` / `_run_single()` plumbing.
- [x] **CLI flags** `--randomize` and `--randomize-iteration N` on
      `benchmark_harness.py run` / `list`.
- [x] **52 tests in `tests/test_harness_randomized.py`** covering
      sampling primitives, transform invariants (optimum preservation,
      orthogonality, condition-number bounds, noise variance), family
      capability gating, and the before/after reproducibility contract.
- [x] **Documentation updated**
  - `doc/source/guide_benchmarking.rst`: replaced the "planned" section
    with a full shipping section (usage, transform math, default
    families, reproducibility recipe).
  - `doc/source/guide.rst`: quick-nav entry mentions parametric
    randomization.
  - `AGENTS.md`: new "Parametrically randomized problems" subsection and
    key-files list updated.
  - `planning/SELF_IMPROVEMENT_LOOP.md`: Phase 3 checklist flipped to
    shipped; "what's missing" list updated.
  - This TODO entry.
- [ ] **Next in the roadmap** — the loop driver `scripts/self_improve.py`
      (Phase 5) can now build on a randomized battery + statistical
      acceptance rule + external baselines.

### Statistical Acceptance Rule for Self-Improvement Loop Phase 4 (2026-04-21)
- [x] **New `statistical_accept()` in `panobbgo/harness.py`** — principled
      accept/reject decision on two `HarnessResult` objects using bootstrap
      confidence intervals on the composite-score delta.
  - For each shared `(problem, strategy)` pair, per-run **solve fractions**
    (the same quantity averaged into `ProblemStrategyResult.score`) are
    resampled independently on both sides.
  - Composite CI is built by averaging per-pair deltas at *matching*
    bootstrap indices — so pair dependencies are preserved, not implicitly
    decoupled.
  - Decision rule (`planning/SELF_IMPROVEMENT_LOOP.md` §6.2): accept iff
    (a) `delta > eps_accept` (default `0.005`), (b) the CI lower bound is
    `> 0`, and (c) no pair regresses by more than `eps_regress` (default
    `0.05`).  Returns a `StatisticalDecision` with the verdict, overall
    CI, worst regressing pair, reasons, and per-pair `PairCI` entries.
- [x] **New `--statistical` flag on `benchmark_harness.py compare`** plus
      the knobs `--eps-accept`, `--eps-regress`, `--n-boot`,
      `--confidence`, `--stat-seed`.  When combined with
      `--fail-on-regression` the CLI exits `2` on rejection, so this is
      usable as a CI gate or as the accept/revert signal for an autonomous
      loop driver.
- [x] **Machine-readable JSON output** — with `--json --statistical` the
      payload carries a `statistical` block (verdict, CI, worst pair,
      per-pair CIs) so an agent can drill into the cause of a rejection.
- [x] **22 tests in `tests/test_harness_stats.py`** — covers accept /
      reject paths, noise-only rejection, per-pair regression guard,
      CI bracketing, reproducibility under the RNG seed, the no-shared-pairs
      degenerate path, JSON serialisation, and three CLI integration
      tests (accept, reject-regression, JSON payload shape).
- [x] **Documentation updated**
  - `doc/source/guide_benchmarking.rst`: new "Statistical acceptance
    rule" section with decision-rule walkthrough, flag table, sample JSON
    payload, and programmatic API pointer.  "Self-improvement loop"
    section now points at the shipped function.
  - `doc/source/guide.rst`: quick-nav entry updated.
  - `AGENTS.md`: "Statistical rigor" subsection now documents
    `--statistical` and the `statistical_accept()` API; key-files list
    updated.
  - `planning/SELF_IMPROVEMENT_LOOP.md`: Phase 4 marked shipped; missing-
    pieces checklist updated.
  - This TODO entry.

### BIPOP-CMA-ES Restart Mode (2026-04-20)
- [x] **Added BIPOP-CMA-ES restart support to `CMAES` heuristic** (`panobbgo/heuristics/cma_es.py`)
  - New `restart_mode` parameter: ``"ipop"`` (default, existing) or ``"bipop"`` (new)
  - BIPOP alternates two restart regimes following Hansen (2009):
    * **Large regime**: geometric population growth ``λ_l = 2^k · λ_default``
      (where ``k`` is the number of large-regime selections so far), σ resets to default
    * **Small regime**: random small population
      ``λ_s = ⌊λ_default · (½ · λ_l/λ_default)^(U[0,1]²)⌋`` and random small step size
      ``σ_s = σ_default · 10^(-2·U[0,1])``
  - Regime selection: after each restart, the regime that has accumulated *fewer*
    cumulative evaluations is selected next (ties → large)
  - New properties: `bipop_regime`, `bipop_evals_large`, `bipop_evals_small`
  - Refactored common restart bookkeeping into shared `_apply_restart()` helper
  - Reference: N. Hansen (2009). "Benchmarking a BI-Population CMA-ES on the
    BBOB-2009 Function Testbed." GECCO Workshop on BBOB.
- [x] **Updated `BIPOP_CMAES` strategy in full benchmark harness** (`panobbgo/harness.py`)
  - Now uses real BIPOP via `restart_mode="bipop"` (previously was just IPOP with more restarts)
  - Pairs `CMAES(sigma0=0.3, restart_mode="bipop")` with diverse Restart analyzer (max 10 restarts)
- [x] **18 new tests** (`tests/test_heuristic_cmaes.py::TestCMAESBIPOP` + integration test)
  - Parameter validation: default mode is "ipop"; invalid modes raise ValueError
  - Initial state: large regime, zero evals tracked
  - Regime alternation: balances cumulative budget within one delta
  - Large regime: geometric population growth `λ_l = 2^k · λ_default`
  - Small regime: λ ≥ base, σ ≤ default
  - Distribution state resets correctly (paths, covariance, eigendecomposition)
  - Box-clamped emission post-restart, base_lam preserved
  - IPOP path unchanged when `restart_mode="ipop"` (no BIPOP attribution)
  - Integration test: BIPOP-CMA-ES on Rastrigin reaches < 20 within 80 evals
- [x] **Documentation updated**
  - `doc/source/guide_architecture.rst`: CMAES section now documents IPOP and BIPOP
    schemes with mathematical formulas and selection rule
  - `doc/source/guide_usage.rst`: New "Highly multimodal problems with BIPOP-CMA-ES"
    section with worked example and IPOP-vs-BIPOP guidance; portfolio table updated
  - `TODO.md`: this entry

### External Baselines for Harness (Self-Improvement Loop Phase 2) (2026-04-20)
- [x] **New `panobbgo/harness_baselines.py`** — adapter strategies so the harness
      can judge Panobbgo in *absolute* terms, not just relative to itself.
  - `RandomSearchStrategy` — uniform random search (composite-score floor).
  - `SciPyDEStrategy` — wraps `scipy.optimize.differential_evolution`
    (population-based global optimizer).
  - `SciPyAnnealStrategy` — wraps `scipy.optimize.dual_annealing`
    (generalized simulated annealing with L-BFGS-B polish).
  - `BaselineStrategy` base class: minimal duck-typed surface matching what
    `BenchmarkHarness._run_single` actually uses (`config.max_eval`, `start()`,
    `best`, `results.results`) — no `StrategyBase` subclass, no event bus.
  - Hard evaluation-budget enforcement via `_BudgetExhausted` raised from
    the objective wrapper: external solvers can never overshoot the harness
    contract, regardless of their own stopping criteria.
  - Results DataFrame uses the same MultiIndex columns (`("fx", 0)`,
    `("who", 0)`, `("x", j)`, …) as Panobbgo strategies, so the harness'
    convergence extractor and heuristic-count logic work unchanged.
- [x] **`HarnessConfig.include_baselines` flag** — when True, the three
      baseline `StrategySpec`s are appended to the mode's strategy list.
- [x] **`benchmark_harness.py --baselines` CLI flag** on both `run` and `list`.
- [x] **22 tests in `tests/test_harness_baselines.py`**
  - Objective wrapper records / stops / projects into box.
  - Adapter surface (config, add/add_analyzer no-op, abstract `_optimize`,
    MultiIndex results, populated `best`).
  - Per-solver budget enforcement and convergence on simple problems.
  - Harness integration: `include_baselines` append path, filtering,
    end-to-end smoke with `composite_score` in [0, 1].
  - Seed reproducibility of the Random baseline.
- [x] **Documentation updated**
  - `doc/source/guide_benchmarking.rst`: replaced "Absolute baselines
    (planned)" with a full shipping section (usage, design, CMA-ES note).
  - `doc/source/guide.rst`: quick-nav entry mentions baselines.
  - `AGENTS.md`: "External baselines" subsection and key-files list updated.
  - This TODO entry.
- [ ] **Next in the roadmap** — statistical acceptance rule (bootstrap CI in
      `compare`, Phase 4) and parametric randomization (Phase 3).

### Benchmark Harness Documentation & Self-Improvement Plan (2026-04-19)
- [x] **New Sphinx guide** (`doc/source/guide_benchmarking.rst`)
  - Full definition of `composite_score` with math, interpretation table, pitfalls
  - Documents the `quick`/`standard`/`full` modes, reproducibility model, `compare` workflow
  - Sections on statistical caveats, parametric randomization (planned), absolute baselines (planned)
  - Wired into `doc/source/guide.rst` toctree
- [x] **Expanded `AGENTS.md`** Benchmark Harness section
  - Statistical rigor subsection (quick-mode noise, re-run at alt seed before accepting small deltas)
  - Self-improvement loop pointer
  - Explicit "composite score formula is a stable contract" note
- [x] **Enriched module docstrings** (`panobbgo/harness.py`, `benchmark_harness.py`)
  - Explicit composite-score formula with per-run solve fraction `s = 1 - (k* - 1)/B`
  - Stability contract for the formula
  - Pointers to the guide and self-improvement plan
- [x] **New plan** (`planning/SELF_IMPROVEMENT_LOOP.md`)
  - Vision for measure→propose→apply→measure→accept/revert loop against randomized problems
  - Parametric problem battery design (translate/rotate/scale/noise/dim sampling)
  - External absolute baselines (scipy DE, dual_annealing, pycma, random)
  - Bootstrap-CI-based statistical acceptance rule + anti-cherry-pick guard
  - Safety rails (dedicated branch, atomic commits, test gating, STOP sentinel)
  - Six-phase rollout from MVP to production loop + success criteria

### Known gap (tracked in plan)
- [ ] Parametric randomization of benchmark problems — plan Phase 3
- [ ] External absolute baselines in the harness — plan Phase 2
- [ ] Statistical acceptance rule (bootstrap CI) in `compare` — plan Phase 4
- [ ] Loop driver `scripts/self_improve.py` — plan Phase 5

### IPOP-CMA-ES Restart Support (2026-04-19)
- [x] **Added IPOP restart to `CMAES` heuristic** (`panobbgo/heuristics/cma_es.py`)
  - `on_restart(center, reason)` handler: moves search mean to new center, doubles λ (IPOP)
  - Resets covariance matrix C, evolution paths p_c/p_σ, and step size σ to initial values
  - Flushes stale pending/in-flight generation results on restart
  - Recomputes all CMA-ES adaptation constants (c_σ, d_σ, c_c, c_1, c_μ) for new population
  - `ipop_factor` parameter (default 2.0) controls per-restart population growth multiplier
  - `restart_count` property tracks total number of IPOP restarts triggered
  - `_base_lam` records the initial population size (preserved across restarts)
  - Reference: Auger & Hansen (2005). "A restart CMA evolution strategy with increasing
    population size." CEC 2005.
- [x] **Added `IPOP_CMAES` strategy to standard benchmark harness** (`panobbgo/harness.py`)
  - Pairs `CMAES(sigma0=0.3, ipop_factor=2.0)` with `Restart(patience=None, restart_strategy="diverse", max_restarts=5)`
  - `Sensitivity` analyzer included for adaptive Nearby perturbations
- [x] **Added `BIPOP_CMAES` strategy to full benchmark harness**
  - Same as IPOP_CMAES but with `max_restarts=10` for the larger 500-eval budget
- [x] **25 comprehensive tests** (`tests/test_heuristic_cmaes.py`)
  - Unit tests for: default/custom ipop_factor, restart_count tracking, population doubling
  - Correctness tests: mean moves to center, sigma resets, paths reset, covariance resets
  - Behavioral tests: pending queue flushed, new generation emitted, box-constraint preservation
  - Weight renormalization, base_lam preservation, multiple restarts
  - Integration tests: IPOP on Rastrigin 2D (200 evals), restart triggered with short patience
- [x] **Documentation updated**
  - `doc/source/guide_architecture.rst`: CMAES entry now documents IPOP restart capability
  - `doc/source/guide_usage.rst`: New "Multimodal problems with IPOP-CMA-ES" section with
    worked example, parameter guide, and comparison to plain CMA-ES

### CMA-ES Heuristic & Core Reward Fix (2026-04-18)
- [x] **Implemented `CMAES` heuristic** (`panobbgo/heuristics/cma_es.py`)
  - Pure-NumPy implementation of the canonical CMA-ES algorithm (Hansen 2016)
  - Async-compatible: tracks results per generation via `who = "CMAES:g<gen>:i<idx>"` tags
  - Adapts covariance matrix C and step size σ from evaluated offspring
  - Lazy eigendecomposition with condition-number guard (resets to spherical if > 1e7)
  - Parameters: `sigma0` (initial step-size fraction), `popsize` (overrides λ=4+3 ln n),
    `min_results_fraction` (fraction of λ before update trigger, default 0.5 = μ)
  - Gold standard for smooth/ridge-following problems (Rosenbrock, ill-conditioned quadratics)
- [x] **Added `CMAES` to standard and full harness strategies** (`panobbgo/harness.py`)
  - `CMAES_Portfolio` strategy in standard mode: LatinHypercube + CMAES + Nearby + NelderMead
  - `CMAES_GP` strategy in full mode: LatinHypercube + CMAES + GaussianProcessHeuristic + NelderMead
  - Intentionally excluded from quick mode (75 evals) — CMA-ES needs ≥ 100 evals to converge
- [x] **Fixed `StrategyBase.heuristic()` lookup for compound `who` strings** (`panobbgo/core.py`)
  - Heuristics like CMAES and DifferentialEvolution embed generation/UUID info in `who`
    (e.g., `"CMAES:g3:i0"`, `"DifferentialEvolution:abc123"`)
  - `heuristic(who)` now falls back to the prefix before `:` if the full key is not found
  - Prevents spurious `KeyError` in `StrategyRewarding.on_new_best` and `_reward_near_best`
- [x] **20 comprehensive tests** (`tests/test_heuristic_cmaes.py`)
  - Initialisation, parameter validation, point emission within bounds
  - Update triggering from partial results, mean convergence, sigma adaptation
  - Covariance positive-definiteness, weight normalisation, foreign-result handling
  - End-to-end integration test on Rosenbrock 2D (150 evals, fx < 5.0)
- [x] **Documentation updated**
  - `doc/source/guide_architecture.rst`: CMAES added to "Population-based" heuristics section
  - `doc/source/guide_usage.rst`: Added CMA-ES portfolio table entry and usage example

### Bayesian Optimization Harness Integration & UCB Bug Fix (2026-04-17)
- [x] **BayesOpt_GP strategy added to standard harness** (`panobbgo/harness.py`)
  - New `BayesOpt_GP` strategy spec added to `_make_standard_strategies()` (200-eval budget)
  - Uses `GaussianProcessHeuristic(n_restarts=5)` + `LatinHypercube(div=4)` + `Nearby` + `NelderMead`
  - Demonstrates GP-based Bayesian optimization within the reproducible harness
- [x] **BayesOpt_Enhanced added to full harness** (`panobbgo/harness.py`)
  - New `BayesOpt_Enhanced` strategy spec added to `_make_full_strategies()` (500-eval budget)
  - Combines `GaussianProcessHeuristic(n_restarts=10)` + `DifferentialEvolution` + `NelderMead`
  - DifferentialEvolution provides global search; GP provides surrogate-guided exploitation
- [x] **Fixed UCB acquisition function bug** (`panobbgo/heuristics/gaussian_process.py`)
  - `_upper_confidence_bound` was maximising LCB instead of minimising it (wrong for minimisation)
  - Fixed: method now returns `-(μ - κσ)` so the outer maximiser correctly minimises LCB
  - Acquisition functions EI and PI were already correct; only UCB was affected
- [x] **Documentation updated** (`doc/source/guide_architecture.rst`, `doc/source/guide_usage.rst`)
  - `guide_architecture.rst`: Added `GaussianProcessHeuristic`, `DifferentialEvolution`,
    `FeasibleSearch`, `ConstraintGradient`, `LocalPenaltySearch`, `ConstraintRepair` to heuristics
  - `guide_architecture.rst`: Added `StrategyUCB`, `StrategyThompsonSampling`, `StrategyLinUCB`,
    `StrategyPhased` with mathematical descriptions
  - `guide_usage.rst`: Added "Bayesian Optimization with Gaussian Process" section with
    acquisition function details, EIC description, and two-phase BO workflow example
  - `guide_usage.rst`: Updated heuristic portfolio table and recommended configurations

### Sensitivity-Aware Nearby Heuristic & StrategySpec Analyzers (2026-04-15)
- [x] **Sensitivity-Aware `Nearby` Heuristic** (`panobbgo/heuristics/nearby.py`)
  - Added `on_new_sensitivity(importance)` event handler to `Nearby`
  - When `Sensitivity` analyzer is active and has published importance scores,
    `Nearby` scales per-dimension perturbations by importance (normalised so overall
    magnitude is preserved)
  - New `sensitivity_scale` constructor parameter controls contrast sharpness (default 1.0)
  - For `axes="all"`: each dimension's step is multiplied by its (normalised) weight
  - For `axes="one"`: dimension is sampled proportionally to importance weights
  - Both `on_new_best` and `on_restart` use the sensitivity-aware `_make_perturbation` helper
  - Improves local search in high-dimensional problems where only a subset of dimensions matter
  - Added `_perturbation_weights()` helper returning normalised weights (mean = 1)
- [x] **`StrategySpec.analyzers` field** (`panobbgo/benchmark.py`)
  - Added optional `analyzers: List[Tuple[type, dict]]` field to `StrategySpec`
  - `create_strategy()` adds extra analyzers (e.g. `Sensitivity`, `Restart`) alongside heuristics
  - Four required analyzers (Best, Grid, Splitter, Convergence) still added in `initialize()`
- [x] **Sensitivity in Benchmark Strategies** (`panobbgo/harness.py`)
  - Added `Sensitivity(update_interval=20)` to `Rewarding_Diverse`, `UCB_Diverse`, and `Thompson_Diverse`
  - Enables adaptive Nearby perturbations in all adaptive benchmark strategies
- [x] **15 new tests** (`tests/test_heuristic_nearby_sensitivity.py`)
  - Verifies `_perturbation_weights()` normalisation and ordering
  - Confirms sensitivity-aware perturbations statistically bias important dimensions
  - Tests both `axes="all"` and `axes="one"` modes
  - Tests `on_restart` with/without sensitivity and with None center
  - Tests that sensitivity updates are immediately effective
  - Tests `StrategySpec.analyzers` round-trip and creation
- [x] **Documentation updated**
  - `doc/source/guide_architecture.rst`: updated event table and Nearby description
  - `doc/source/guide_usage.rst`: added Sensitivity-Aware Nearby section with example

### Benchmark Harness for Agent Feedback Loops (2026-02-23)
- [x] **Implemented `panobbgo/harness.py` – Reproducible Benchmark Harness**
  - `BenchmarkHarness` class: runs seeded, reproducible benchmark suites
  - Three modes: `quick` (3 problems × 2 strategies × 3 reps, 75 evals), `standard`, `full`
  - Per-run seed derivation for best-effort reproducibility across runs
  - Convergence trace extraction directly from the MultiIndex results DataFrame
  - ERT (Expected Running Time) and per-pair performance score in [0, 1]
  - Composite score = mean of per-pair scores; single scalar for before/after comparison
  - Full JSON serialisation / deserialisation (`save()` / `load()`)
  - `compare()` helper: diff two `HarnessResult` files, flag regressions/improvements
- [x] **`benchmark_harness.py` – CLI for Agent Loop**
  - `run`: execute benchmarks and save a timestamped JSON file
  - `score`: print human-readable summary + optional machine-readable JSON
  - `compare`: side-by-side diff with `--fail-on-regression` exit-code support
  - `list`: enumerate available problems and strategies per mode
- [x] **`tests/test_harness.py` – 60 tests covering all harness components**
  - Unit tests for metrics, serialisation, comparison, seed derivation
  - Smoke integration tests for end-to-end runs (single problem, 30 evals)
  - CLI tests via `main()` invocation

### Contextual Bandit Strategy (2025-01-13)
- [x] **Implemented StrategyLinUCB (Contextual Bandits)**
  - Implemented `StrategyLinUCB` with disjoint linear models for each heuristic.
  - Features include Bias, Budget Progress, and Recent Success Rate.
  - Added unit/integration test `tests/test_strategy_contextual.py`.
  - Updated `panobbgo/lib/classic.py` to support `Rosenbrock(dim=2)` kwargs.

### Thompson Sampling Strategy (2025-01-13)
- [x] **Implemented StrategyThompsonSampling**
  - Added new strategy using Beta-Bernoulli bandit logic
  - Implemented `reward` based on improvement magnitude
  - Implemented `execute` with randomized selection based on Beta samples
  - Added unit tests in `tests/test_strategy_thompson.py`

### PR #43 - Dask Memory Leak Fix & Test Suite Cleanup (2025-01-13)
- [x] **Fixed Critical Memory Leak in Dask Cleanup**
  - Added proper `LocalCluster` cleanup in `_setup_dask_cluster()` and shutdown code
  - Store cluster reference (`self._cluster`) to ensure worker processes are terminated
  - Call both `self._client.close()` AND `self._cluster.close()` during cleanup
  - Prevents memory blowup when running multiple tests that use Dask evaluation
- [x] **Deferred Dask Testing (Future Work - Weeks)**
  - Disabled all Dask-related tests (`test_config_init.py`, `test_dask_evaluation_integration()`)
  - Default test execution model is now "threaded" only
  - Dask evaluation still works in production, just not tested in test suite
  - TODO: Proper Dask test isolation and cleanup testing in future sprint

### PR #42 - FeasibleSearch & Test Warnings (2025-01-13)
- [x] **Test Suite Warnings Resolved**
  - Fixed NumPy RuntimeWarnings in convergence analyzer using `warnings.catch_warnings()`
  - Suppressed warnings for edge cases (identical values, small samples) in std deviation calculations
  - Skipped Dask evaluation integration test (focusing on threaded evaluation for now)
  - All 143 tests now pass with 1 skipped, 0 warnings
- [x] **FeasibleSearch Heuristic Enhanced**
  - Implemented biased line search using Beta(2,1) distribution for more efficient boundary finding
  - Improved comments explaining the line search strategy between feasible/infeasible points
  - Updated copyright year to 2012-2025 per project guidelines
  - All FeasibleSearch tests passing

## Framework Quality Assurance & Completion

### 🔴 CRITICAL: TDD Bug Fixes & Quality Validation (Priority 1)
**TDD Approach**: Write failing tests first, then implement fixes
- [x] **Optimization Loop Stability** - Major hanging issues resolved
  - [x] **FIXED**: Random heuristic infinite wait (main hang cause)
  - [x] **FIXED**: abs() errors in convergence analyzer and progress reporting
  - [x] Basic optimization now completes successfully
  - [ ] Full optimization loop robustness (complex threading - lower priority)
- [x] **Heuristic Functionality** - Core issues resolved
  - [x] **FIXED**: Random heuristic infinite wait (main hang cause)
  - [x] **VALIDATED**: Nearby heuristic generates correct points
  - [x] Added TDD tests for heuristic point generation
  - [ ] Full event system integration (lower priority)
- [x] **Dedensifyer Analyzer** - Fix critical implementation bugs
  - [x] Write TDD tests for proper initialization and grid management
  - [x] Fix constructor (missing strategy parameter)
  - [x] Fix undefined variables and wrong method signatures
  - [x] Validate hierarchical grid functionality
- [x] **Optimization Correctness Validation** - Add tests proving algorithms work
  - [x] Write tests validating convergence to known optima
  - [x] Compare optimization vs random baseline performance
  - [x] Add statistical significance testing

### 🟡 MEDIUM: Coverage Expansion on Validated Code (Priority 2)
**Revised Goal**: 75% coverage on components proven to work correctly
- [x] Expand UCB strategy tests (currently 91% - add edge cases)
- [x] Complete Best analyzer test coverage (currently 34%)
- [x] Add Grid analyzer comprehensive tests (currently 56%)
- [x] Test remaining heuristics: LBFGSB (30%), Nelder-Mead (51%)
- [x] Add integration tests for constrained optimization scenarios

### 🟢 LOW: Documentation & Polish (Priority 3)
- [x] Update documentation references from IPython parallel to Dask
- [x] Review and fix minor naming inconsistencies in guide documentation
- [x] Remove remaining IPython parallel references from code and documentation
- [ ] Review and potentially simplify UI components
- [ ] Add performance benchmarks comparing different strategies
- [ ] Review and optimize threading/event handling

### 🔵 DEFERRED: Dask Testing & Validation (Future Work - Weeks)
**Status**: Completed! Dask tests are isolated, pass locally, and memory leak fix is verified.
- [x] **Dask Test Isolation**: Properly isolate Dask tests to avoid port conflicts
  - Use pytest fixtures to ensure clean Dask cluster setup/teardown
  - Ensure each test gets a fresh LocalCluster with unique dashboard port
  - Test that cluster cleanup properly terminates all worker processes
- [x] **Re-enable Dask Tests**: Currently skipped tests
  - `tests/test_config_init.py` - testing_mode and dashboard configuration
  - `tests/test_integration.py::test_dask_evaluation_integration` - Dask evaluation
- [x] **Verify Memory Leak Fix**: Test that the LocalCluster cleanup fix prevents memory leaks
  - Run repeated Dask evaluations and monitor memory usage
  - Verify worker processes are terminated after cleanup
- [x] **Dask Production Usage**: While tests are disabled, Dask evaluation still works
  - Document current Dask usage patterns for production
  - Consider adding example scripts demonstrating Dask evaluation

## Known Issues & Technical Debt

### Strategy Lifecycle Management (Systemic Issue)
**Problem**: Real strategy instances (StrategyRoundRobin, StrategyRewarding) start background processes (via Dask) that don't clean up properly when tests complete. This causes:
- Test hangs when multiple tests use real strategies (PR #35, PR #32)
- Resource leaks in test suites
- Unreliable benchmark tests

- `strategy.start()` initializes background threads/processes
- [x] **FIXED**: Strategy lifecycle methods (`__stop__`, `_cleanup`) implemented.
- [x] **FIXED**: Context manager support (`__enter__`, `__exit__`) implemented.
- Tests can now properly tear down strategies using `strategy.stop()` or `with` blocks.

**Current Workarounds**:
- Unit tests: Use `@mock.patch("panobbgo.core.StrategyBase")` to avoid real strategies
- Integration tests: Skip tests that hang (e.g., `test_heuristic_tracking` in benchmarks)
- Set `evaluation_method="threaded"` helps but doesn't fully solve cleanup issues

**Proper Solution Needed**:
- [x] **FIXED**: Cleanly terminate background processes.
- [x] **FIXED**: Implementation of `strategy.cleanup()` methods.
- [x] **FIXED**: Context manager support (`__enter__`/`__exit__`).
- [x] Implement pytest fixtures for automatic strategy setup/teardown in tests.
- [ ] Review all Dask distributed usage for best practice cleanup patterns.

**Affected Files**:
- `panobbgo/core.py` - StrategyBase class needs lifecycle methods
- `tests/test_heuristic_feasible.py` - Fixed by using mocked strategies (PR #35)
- `benchmarks/test_benchmarks.py` - Skipped hanging test (PR #32)

### Benchmark Heuristic Tracking Issues (PR #32)
**Bug in convergence_trace logic** (`benchmarks/test_benchmarks.py:88-93`) - **FIXED**:
- ~~When `best_fx == float('inf')` (first evaluation), `old_best_fx` is set to `result.fx`~~
- ~~This causes `improvement = result.fx - result.fx = 0`, which is incorrect~~
- **Fixed**: First improvement now correctly recorded as `result.fx` (function value from baseline)
- **Fixed**: Subsequent improvements correctly calculated as `best_fx - result.fx`

### 🎯 TARGET: 75% Coverage on Validated Components
**Prerequisites**: All Priority 1 items completed with TDD validation
**Quality Metrics**: Correctness + Coverage (not just coverage)
**Status**: Core issues resolved, coverage stands at ~71%.

## Known Issues & Technical Debt

### Strategy.start() Hang Bug (FIXED)
**CRITICAL**: `strategy.start()` doesn't return after reaching `max_eval` evaluations
- **Status**: FIXED by addressing result collection deadlocks and improving cleanup in [PR #38](https://github.com/haraldschilly/panobbgo/pull/38).

### PR #36 Bug Fixes (Merged)
**Fixed Issues** - All good fixes:
- [x] **Splitter.Box.__ranges** - Fixed `.ptp()` call to work with BoundingBox objects (`panobbgo/analyzers/splitter.py:215-220`)
- [x] **memoize decorator** - Added handling for unhashable NumPy arrays by converting to bytes (`panobbgo/utils.py:205-230`)
- [x] **Analyzer name consistency** - Changed "splitter"/"best" to "Splitter"/"Best" (Random, WeightedAverage heuristics)
- [x] **Random heuristic initialization** - Added logic to get root leaf from Splitter on start (`panobbgo/heuristics/random.py:38-48`)
