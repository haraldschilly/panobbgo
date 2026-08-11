# TODO

## Recent Improvements (continued)

### Goal contract corrected; diagnosis written down; planning rotated — 2026-08-11 (sixth session)
- [x] **`GOAL.md` §2 / §5.1 corrected** — the "instance-family
      generalization" research item was built on a metric-unit bug and is
      retracted; the slot now holds *regime-conditional strategy
      selection*.  §5.3 marked shipped.  §4 Diagnose points at `per_cell`.
- [x] **`planning/LOOP_DIAGNOSIS_2026-08-11.md`** — full 34-night audit so
      no future session re-derives it.
- [x] **Rotated into `planning/done/`** — both ledgers, the summary, and
      the pre-2026-07-30 halves of `TODO.md` (3720 → 332) and
      `SELF_IMPROVEMENT_LOG.md` (11126 → 1117).  Nothing deleted; the
      bandit still primes from archives and metric inference on the new
      filenames is verified.
- [ ] **Watch the first post-rotation nights** — the live aocc ledger is
      empty, so codify-scan has no cross-night evidence until ~2 nights
      have run on *distinct* base seeds.  Expect a quiet week; that is
      correct behaviour, not a regression.

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
- [x] ~~**Open weakness** — hold-out base seeds score far below training seed
      (0.04 vs 0.33 on 2026-07-30): instance-family generalization is the
      top research target~~ — **RETRACTED 2026-08-11.**  This was a
      metric-unit bug, not a weakness: `_measure_holdout` never routed
      through the AOCC path, so AOCC runs wrote `composite_score`
      (~0.045 scale) into hold-out records next to mean-AOCC training
      records (~0.34).  Fixed in #299; the real gap is **0.3383 vs
      0.3342**.  See `planning/LOOP_DIAGNOSIS_2026-08-11.md` §3.1.  The
      research slot it occupied in `GOAL.md` §5.1 is now
      *regime-conditional strategy selection*.

---

*Entries before this point were moved to [`planning/done/TODO_archive_pre-2026-07-30.md`](TODO_archive_pre-2026-07-30.md) on 2026-08-11 to keep this file readable. Nothing was deleted — the archive is the same newest-first format.*
