# Self-Improvement Iteration Log

Append-only history of algorithmic improvements applied to Panobbgo
*outside* the autonomous loop, plus the rolling "Next iteration ideas"
backlog at the bottom.  Split out of `SELF_IMPROVEMENT_LOOP.md` (where
it was §13) on 2026-06-09 so the loop document stays a compact,
readable instruction file.

Conventions:

* One dated `###` entry per shipped change — newest first.  Each entry
  references the PR / commit that landed it, the rationale, and a
  measured-impact number when available.
* Section references like §6.2 / §7.2 / §12.3 point into
  `SELF_IMPROVEMENT_LOOP.md`; "see the §13 entry" in older text means
  "see the dated entry in this file".
* Graduate items from "Next iteration ideas" to a dated entry when
  shipped.

### 2026-08-12 — dimension-gated arm activation ships; NLSHADE_LBC enters `Rewarding_Restart` gated to d≥5

* **What** — the §4.3 "shippable form" that PR #298 asked for and could
  not express: `StrategySpec.create_strategy` now honours two reserved
  heuristic-kwargs keys, `gate_min_dim` / `gate_max_dim`, which gate
  whether the arm is instantiated at all based on `problem.dim` (the
  keys are stripped before the heuristic constructor sees them; curated
  `MutationRule`s name (class, param) explicitly, so the reserved keys
  cannot be mutated by accident).  First user:
  `(NLSHADE_LBC, {"NP_init": "auto", "k_rank": 3.0, "gate_min_dim": 5})`
  in `Rewarding_Restart` — the exact structural-catalog kwargs #298
  measured.  At d < 5 the spec is arm-for-arm identical to master; at
  d ≥ 5 the CEC-2022-winner DE arm activates.

* **Why gated** — #298's 12-seed standard A/B measured the
  *unconditional* add at d2 **−0.0241 [−0.0401, −0.0080]** vs d5
  **+0.0080 [+0.0007, +0.0154]** — both CIs exclude zero, in opposite
  directions, so the scalar mean hid a real gain behind a real loss.
  The gate ships the d5 gain without paying the d2 loss.

* **Measurement** (12-seed canonical roster, `--standard`,
  `--sync-eval` both sides, seed-paired; before = master 92b603e):

  | strategy | dim | before | after | Δmean | CI95 | verdict |
  |---|---|---|---|---|---|---|
  | `Rewarding_Restart` | d2 | 0.4330 | 0.4297 | −0.0033 | [−0.0131, +0.0065] | flat (gate off — spec identical to master) |
  | `Rewarding_Restart` | d5 | 0.3026 | 0.3088 | **+0.0062** | [−0.0001, +0.0125] | positive, boundary |
  | `RoundRobin_Random` (control) | d2 | 0.3618 | 0.3613 | −0.0005 | [−0.0019, +0.0009] | flat |
  | `RoundRobin_Random` (control) | d5 | 0.2834 | 0.2846 | +0.0011 | [−0.0052, +0.0074] | flat |

  At d5 the gated spec is byte-identical to the unconditional spec
  #298 measured, so #298's d5 row is an *independent replication of
  the same effect*: fixed-effect inverse-variance pooling of the two
  12-seed estimates (+0.0080 se 0.0033; +0.0062 se 0.0029) gives
  **+0.0070 [+0.0027, +0.0112]** — CI excludes zero.  The quick
  battery (d2-only) was not re-measured: with the gate off the spec
  is identical to master there, and today's d2 row confirms flat.

* **Mechanism, not just the arm** — this is the first deliverable of
  GOAL §5.1(d) (regime-conditional strategy selection).  The same
  two-key mechanism can ship the CMA-ES arm at d5 (GOAL §5.2 open
  question — re-measure with this instrument first) and any future
  cell-conditional codify proposal once codify-scan learns to read
  `per_cell` (§5.1(c), still open).

* **Codify state** — today's scan surfaced 0 actionable candidates
  (13 total: 5 already codified, 8 rejection-suppressed); last
  night's two accepts (add Nearby, add Center → `RoundRobin_Random`)
  both reinforce already-codified slots, so no codify slot competed
  with this session's one-PR budget.

* **Validation** — full pytest green (2014 passed), `ruff format
  --check .` clean, `ruff check` clean on the touched source files
  (the 5 pre-existing `tests/test_benchmark.py` lint findings on
  master are unchanged).  Unit tests pin gate-below-threshold,
  admit-at-threshold, key-stripping, max-dim gating, and
  no-mutation-of-spec-kwargs.

### 2026-08-11 (sixth session) — goal contract corrected, diagnosis written down, planning artifacts rotated

* **`GOAL.md` §2 and §5.1 corrected.**  The state snapshot's "hold-out
  base seeds score 0.04 vs 0.33" and the §5.1 research item built on it
  ("instance-family generalization" — the project's *number-one* stated
  priority since 2026-07-30) were a **metric-unit bug**, not a finding.
  §5.1's slot now holds *regime-conditional strategy selection*, which is
  the real measured structural problem.  §5.3 (rank stats) is marked
  shipped.  §4's Diagnose step now points at the per-cell breakdown first.
  The retracted TODO item is struck through in place rather than deleted —
  a retraction that leaves no trace invites the same conclusion again.

* **`planning/LOOP_DIAGNOSIS_2026-08-11.md`** — the full 34-night audit:
  the numbers, the four instrument defects, the scalar-objective argument,
  what each of #299–#302 changed, how to read the ledger across the
  boundary, and what to do next.  Written so no future session re-derives
  it.

* **Rotation.**  Following the established `planning/done/` convention:

  | from | to | why |
  |---|---|---|
  | `self_improve_ledger_aocc.jsonl` (952 rows) | `done/self_improve_ledger_aocc_2026-08-11.jsonl` | every record predates today's instrument — single base seed, async eval, d2-only, eps 0.005.  Pooling across that boundary is the bug class just fixed. |
  | `self_improve_ledger.jsonl` (811 rows, 2026-06-06..07-08) | `done/self_improve_ledger_composite_2026-07-08.jsonl` | dead since the nightly flipped to aocc on 2026-07-09 |
  | `self_improve_summary.txt` | `done/self_improve_summary_2026-08-11.txt` | regenerated nightly |
  | `TODO.md` 3720 → 332 lines | `done/TODO_archive_pre-2026-07-30.md` | readability; cut at the metric-aware-codify epoch |
  | `SELF_IMPROVEMENT_LOG.md` 11126 → 1117 lines | `done/SELF_IMPROVEMENT_LOG_pre-2026-07-30.md` | same |

  **Nothing is deleted.**  The bandit posterior survives the ledger
  rotation because the nightly passes `--prime-include-archives`, which
  replays `planning/done/`.  Verified that `iter_metric_archives` infers
  the metric correctly from the new filenames — an aocc run primes only
  from `..._aocc_...`, a composite run only from the two composite
  archives — so the rotation cannot cross-contaminate metrics.  What
  *does* reset is codify-scan's evidence base, which is the point: 34
  nights of single-draw evidence should count as one night's worth.

### 2026-08-11 (fifth session) — the objective stops being a scalar: per-cell (per-dim) acceptance

* **What** — `statistical_accept(..., cell_by="dim")` groups the
  `(problem, strategy)` pairs into *cells* by `problem_dim`, reports a
  delta and bootstrap CI per cell, and blocks acceptance when a cell
  **credibly** regresses.  `--cell-by dim` / `--eps-cell-regress`, and
  on in the nightly now that the d5 slice gives it more than one cell.

* **Why this is the deepest of today's changes** — the previous four
  fixed *how well* the loop measures.  This one changes *what it
  optimises*.  A scalar composite is a mean over pairs, so regimes that
  move in opposite directions cancel:

  | | d2 | d5 | scalar composite |
  |---|---|---|---|
  | NL-SHADE-LBC arm (2026-08-11, #298) | **−0.0241** [−0.0401, −0.0080] | **+0.0080** [+0.0007, +0.0154] | −0.0080 |

  Both CIs exclude zero, in opposite directions, and the composite says
  "lean-negative" — which describes neither regime and hides that a
  real gain exists.  If effects are routinely that heterogeneous, then
  the population-mean objective has a **flat optimum by construction**,
  and no amount of extra measurement precision helps: the loop is
  climbing a surface that is genuinely level.  That, not noise alone,
  is the deeper reason 34 nights produced one +0.005 improvement.

* **The gate** — a cell blocks iff its delta is below
  `-eps_cell_regress` (default 0.01) **and** its entire CI sits below
  zero.  Both conditions are required so a merely noisy cell cannot
  veto an otherwise good change; a test pins exactly that (a d2 cell
  whose point estimate is −0.02 but whose CI straddles zero does not
  block).  Under the default the #298 change would have been rejected
  *on its d2 cell*, with the d5 gain recorded, instead of being
  averaged into an ambiguous −0.008.

* **Recording matters as much as gating** — the per-cell breakdown goes
  into the ledger (`per_cell`, `blocking_cell`) on every iteration,
  accept or reject.  Codify-scan currently pools one scalar delta per
  night, so a change that is +0.02 at d5 and −0.001 at d2 is
  indistinguishable in the ledger from one that is +0.01 everywhere.
  With the breakdown persisted, a future scan can propose a
  **dimension-gated arm** rather than an unconditional one — which is
  the shippable form the #298 write-up asked for and could not express.

* **Orthogonal to the rank rule** — one asks whether the typical effect
  is real, the other whether any regime is being sacrificed to achieve
  it.  Both compose; a test covers the combination.

* **Not yet cells: budget phase.**  `IOHRunRecord` already carries a
  down-sampled trajectory (`trace_evals` / `trace_fx`), so phase cells
  are feasible, but AOCC would have to be recomputed on trajectory
  slices rather than read off the final value — a change to the metric
  path, not the decision path.  Queued.  Note the #298 evidence says
  this matters: the same arm *leaned positive* at 2-D×200 evals and
  negative at 2-D×1000, which is a budget effect, not a dimension one.

* **Validation** — 1996 passed, 15 skipped; `ruff format --check`
  clean; pyright 0 errors, 0 warnings.  11 new tests, including that
  cell deltas average back to the composite, that the flat rule accepts
  the very measurement the cell rule blocks, and that a single-dim
  battery degenerates to one cell.

### 2026-08-11 (fourth session) — rank-based acceptance available (GOAL §5.3); **not** enabled nightly

* **What** — `statistical_accept(..., accept_stat="rank")` and
  `--accept-stat rank`.  A one-sided Wilcoxon signed-rank test on the
  per-pair deltas shifted by `eps_accept` replaces *both* mean
  conditions (`delta > eps_accept` **and** `ci_low > 0`): asking whether
  `median(d) > eps_accept` beyond chance is the rank analogue of "the
  mean cleared the bar and its CI excludes zero".  Graduates research
  backlog item GOAL §5.3.

* **Why** — the composite is a *mean* over `(problem, strategy)` pairs,
  so a single pair that happens to solve can carry it past the bar
  alone.  On a battery of 6–12 pairs that is not a remote possibility,
  it is the modal way a noise accept happens.  Demonstrated in the test
  suite: seven pairs drifting −0.005 plus one pair going 0 → 1.0 gives
  a composite of +0.12 with a degenerate (zero-variance) CI — the mean
  rule accepts on the strength of one pair; the rank rule sees 7 losses
  against 1 win, `p = 0.97`, and refuses.

* **It is not merely stricter** — the converse case also holds: eleven
  pairs at +0.020 against one at −0.300 gives a mean of −0.0067 (mean
  rule rejects) but `p = 0.017` (rank accepts).  A broad win dragged
  negative by one catastrophe is exactly what a mean should not be
  trusted with.  The per-pair `eps_regress` guard is orthogonal and
  still applies under both rules.

* **Hard floor on sample size** — the smallest attainable one-sided
  Wilcoxon p on `n` pairs is `2**-n`, so at `confidence=0.95` a rank
  accept is *impossible* below `n = 5` however large the effect.  A
  property of the test, not a bug, but it constrains where the mode may
  be switched on.  The AOCC quick battery gives `3 instances x n_specs`
  — 6 with two specs, 12 with the d5 slice on — comfortably clear.
  Parametrised tests pin both sides of the boundary.

* **Deliberately NOT enabled in the nightly.**  Tonight's run already
  changes the accept regime in three ways at once (seed rotation, eps
  0.005 → 0.0125, the d5 slice, which moves the score *level*).  Adding
  a fourth simultaneous change to the accept *rule* would make the next
  few weeks of ledger uninterpretable — nobody could attribute a change
  in accept rate to any one of them.  This is precisely the discipline
  whose absence produced five consecutive unfalsifiable codify slots.
  Queued in TODO.md as an explicit A/B once the new instrument has a
  baseline.

* **Ledger continuity** — `delta` stays the **mean** under both rules,
  so the ledger series is comparable across a rule switch.  The rank
  location estimate (Hodges-Lehmann, the median of Walsh averages — the
  estimator that pairs with Wilcoxon as the mean pairs with the
  bootstrap) is recorded separately as `rank_delta`, alongside
  `rank_p` and `accept_stat`.

* **Validation** — full suite green (one pre-existing flake in
  `test_storage_integration.py::test_resume_capability`, passes in
  isolation and is untouched by this change); `ruff format --check`
  clean; pyright 0 errors, 0 warnings.  32 new tests.

### 2026-08-11 (third session) — the nightly loop can finally see d5

* **What** — an opt-in `--aocc-extra-dims` widens the AOCC battery for
  *every* measurement the loop makes (screening, confirm, guard,
  hold-out), and the nightly turns it on with `5`.  The quick preset's
  `dims=(2,)` becomes `(2, 5)`.  Mode presets stay frozen per GOAL §4 —
  the widened battery is *composed* by
  :func:`~panobbgo.harness_ioh.with_extra_dims`, which appends, sorts,
  and suffixes the name (`ioh-quick+d5`) so a report can never conflate
  it with the preset.

* **Why** — the loop has been optimising a regime that hides its own
  best results.  Two measured cases, both outside the nightly battery:

  | date | change | where the effect lived |
  |---|---|---|
  | 2026-08-02 | JSO `add_heuristic` | d5 |
  | 2026-08-11 | NLSHADE_LBC `add_heuristic` | d2 −0.0241 [−0.0401,−0.0080] **vs** d5 +0.0080 [+0.0007,+0.0154] |

  The second is the sharper lesson: the two dims moved in *opposite*
  directions with both CIs excluding zero.  A 2-D-only loop reads that
  change as a loss and rejects it; a (2, 5) loop at least sees the
  conflict.  GOAL §4's own cadence guardrail — "if the quick-battery
  score stalls for >1 week with evidence banked, the bottleneck is the
  *measurement regime*" — has been triggered for five weeks.

* **Cost** — measured on the quick battery, 2 specs, sync-eval on:
  **8.9 s → 11.9 s (1.34×)**, 6 runs → 12.  Much less than the 2.5×
  the budget rule (`budget_multiplier * dim`) suggests, because run
  time here is not budget-bound.  The nightly goes ~5.4 → ~7.2 min
  against a 90-minute timeout.

* **Scale discontinuity, deliberate and recorded** — widening moves the
  *level*, not just the noise: on this battery d2 alone reads 0.3685
  while (d2, d5) reads ~0.309, because d5 is simply harder.  Nights
  before and after the switch are therefore **not on one scale**.  The
  per-iteration `aocc_extra_dims` field records which battery produced
  each record, exactly as `sync_eval` records which evaluation mode did;
  any cross-night consumer has to group by both.  Recording this is the
  same discipline whose absence produced the hold-out metric-mismatch
  bug fixed earlier today.

* **Validation** — 1967 passed, 1 skipped; `ruff format --check` clean;
  pyright 0 errors.  9 new tests, including that every measurement leg
  widens identically (a 2-D baseline against a (2,5)-D candidate would
  be a silent catastrophe) and that re-widening does not stack name
  suffixes.

### 2026-08-11 (second session) — measurement fidelity: hold-out metric-mismatch bug fixed, confirm gate crosses a base seed, nightly seed rotation, `--sync-eval`, eps recalibration

* **What** — a review of the full 34-night AOCC ledger (952 records,
  2026-07-09 .. 2026-08-11) found the loop's accept instrument could
  not, in principle, do what the codify pipeline assumed of it.  Four
  fixes ship together because they are one defect seen from four
  sides.  No optimizer-behavior change; this is all measurement
  plumbing.

* **The 34-night picture that motivated it**

  | quantity | value |
  |---|---|
  | first-of-night baseline, 34 nights | mean 0.3398, sd 0.0061 |
  | trend | +0.00015/night, t = +1.39 (n.s.) |
  | first 5 nights → last 5 nights | 0.3433 → 0.3420 |
  | proposals | 680 |
  | screening-accept rate | 210/680 = **0.309** |
  | `P(delta > eps_accept=0.005)` under the measured noise | **0.301** |
  | confirm survival | 72/210 = 0.343 (independent redraw predicts 0.30) |
  | codify slots reaching a 12-seed A/B | 5 |
  | codify slots that survived | **0** |

  The screening bar sat at **0.5 sd** of the single-measurement noise
  (sd 0.0101), so ~30% of *zero-effect* proposals cleared it and the
  confirm gate re-rolled at the same bar.  The observed 10.6% accept
  rate is what pure noise predicts (0.30 × 0.34); there is no room
  left in it for signal.

* **Bug 1 — hold-out measured the wrong metric entirely.**
  :meth:`SelfImprover._measure` branches on ``metric == "aocc"``;
  :meth:`_measure_holdout` did not, so it fell through to
  ``holdout_harness_config`` → the **composite** harness.  Every
  hold-out record in the AOCC ledger carried a ``composite_score``
  while its training record carried mean AOCC.  Measured:
  ``seed_holdout_score`` mean **0.0339** vs ``seed_training_score``
  mean **0.3402** — and the composite ledger's own baseline mean is
  **0.0452**, the same scale.  That 8.5× "instance-family
  generalization gap" is a **unit mismatch**, and it has been
  `planning/GOAL.md` §5.1's *number-one research priority* since
  2026-07.  After the fix, a real quick-battery run measures
  **0.3383 training vs 0.3342 hold-out** — a 0.004 gap, not 0.29.
  ``overfit=False`` on all 66 records is likewise explained: the gate
  was differencing two incommensurable quantities.
  ``_measure_aocc`` grew a ``base_seed_override`` parameter (the AOCC
  counterpart of ``holdout_harness_config``'s seed swap) and
  ``_measure_holdout`` now routes through it.

* **Bug 2 — no accept decision ever crossed an instance-family
  boundary.**  The §6.4 confirm gate's hold-out leg was guarded by
  ``and self.config.metric != "aocc"`` — added precisely because of
  Bug 1.  Consequence: ``confirm_holdout_seed`` is ``None`` in all 138
  confirm records, 0/72 accepts cite a hold-out seed, and the confirm
  re-measurement only drew a fresh ``randomize_iteration`` *within*
  base seed 42.  With Bug 1 fixed the exclusion is removed.

* **Bug 3 — the nightly never set `--base-seed`.**  All **952**
  records are ``base_seed=42``.  Codify-scan's "k≥2 distinct nights"
  gate — and the ``--min-fresh-nights`` resurrection gate shipped
  2026-08-08 — were therefore counting *k re-measurements of one
  instance draw* as k independent confirmations.  This is the direct
  mechanical cause of the 0/5 codify hit rate: five slots with tight
  pooled CIs, every one a seed-42 artifact.  The nightly now rotates
  deterministically over ``(42, 101, 202, 303, 404, 505, 606)`` keyed
  on UTC day-of-year (prime-length, so it cannot alias with a weekly
  cadence; 7 and 1234 stay out of the pool because they are the
  hold-out seeds and the confirm leg must stay independent of the
  screening draw).

* **Gap 4 — `--sync-eval` was unreachable from the loop.**  Shipped
  2026-08-09 on `scripts/ioh_benchmark.py`, where it cut measurement
  noise 1.6×, but never plumbed into `scripts/self_improve.py run` —
  so the process generating *all* the evidence was the one not using
  it.  Now a `LoopConfig.sync_eval` field, a `--sync-eval` flag, and
  on by default in the nightly's aocc branch.  Measured cost on the
  quick battery: **2.6 s vs 2.7 s** — free.  The mode is recorded per
  iteration in the ledger, because sync and async have different
  noise floors and cross-night pooling must not mix them.

* **Recalibration** — with sync-eval the noise floor is ~0.0063, so
  the nightly's aocc branch moves ``--eps-accept`` 0.005 → **0.0125**
  (2σ) and ``--inactivity-min-eps-accept`` 0.001 → **0.006** (the
  relax rule previously walked the bar straight back under the
  noise).  Under the null the joint screen+confirm false-positive rate
  goes from ~9% to ~0.05%; power for a true +0.02 effect stays ~77%.
  Effects below ~0.01 were never shippable anyway — five A/Bs in a row
  proved that — so the loop should stop spending nights on them.
  **Composite keeps its historical 0.005 / 0.001**: its battery mean is
  ~0.045, where the same absolute epsilon would be a 28% relative bar.

* **Validation** — 1958 passed, 1 skipped; `ruff format --check`
  clean (199 files); pyright 0 errors.  10 new regression tests.  A
  live 1-iteration `--metric aocc --sync-eval --base-seed 101` run
  produced the 0.3383 / 0.3342 pair quoted above and a ledger record
  carrying ``sync_eval=true``, ``base_seed=101``,
  ``effective_eps_accept=0.0125``.

* **Consequence for the existing ledger** — the 34 nights of seed-42
  evidence are not *wrong*, they are narrow: they estimate the seed-42
  effect precisely and the population effect not at all
  (`corr(training_delta, holdout_delta) = +0.175`, n=66).  Cross-night
  codify evidence accumulated before tonight should be treated as one
  night's worth, not 34.

### 2026-08-11 — `add_heuristic NLSHADE_LBC` → `Rewarding_Restart` REJECTED unconditionally; first CI-significant per-dim split (d5 gain, d2 loss)

* **What** — Daily session executed the §4.3 measured change queued
  2026-08-10: add the structural-catalog arm
  `(NLSHADE_LBC, {"NP_init": "auto", "k_rank": 3.0})` to
  `Rewarding_Restart` directly (the aocc ledger's `add_heuristic
  NLSHADE_LBC` accepts — 3 confirmed across 2 nights — only target the
  policy-frozen `RoundRobin_Random` control, so no codify slot can
  reach the candidate spec; direct paired A/B is the vehicle).
  Today's codify-scan surfaced **0 actionable candidates** (12 total:
  4 already codified, 8 rejection-suppressed), so this was the
  session's one slot.  Verdict: **rejected as an unconditional add**
  and reverted in-branch (PR #298; apply + revert commits cancel, net
  source diff carries no optimizer-behavior change — same pattern as
  PR #297).

* **Measurement** (12-seed canonical decision roster, `--sync-eval`
  on both sides — first decision made on the 2026-08-09 low-noise
  instrument — seed-paired `compare`):

  Quick battery (2-D, 100·d budget):

  | strategy | before | after | Δmean | sd | CI95 | verdict |
  |---|---|---|---|---|---|---|
  | `Rewarding_Restart` (+ LBC) | 0.3413 | 0.3505 | +0.0092 | 0.0249 | [−0.0066, +0.0251] | ~ noise, 9/12 seeds + |
  | `RoundRobin_Random` (control) | 0.3341 | 0.3351 | +0.0010 | 0.0039 | [−0.0015, +0.0035] | flat |

  Standard battery (d2+d5, 500·d budget, 5 instances):

  | strategy | before | after | Δmean | sd | CI95 | verdict |
  |---|---|---|---|---|---|---|
  | `Rewarding_Restart` (+ LBC) | 0.3734 | 0.3654 | −0.0080 | 0.0160 | [−0.0182, +0.0022] | lean-negative, 8/12 seeds − |
  | `RoundRobin_Random` (control) | 0.3232 | 0.3228 | −0.0004 | 0.0073 | [−0.0050, +0.0043] | flat |

  **Per-dimension split** (paired per-seed deltas, t-dist CI95, n=12):

  | strategy | dim | before | after | Δmean | CI95 | verdict |
  |---|---|---|---|---|---|---|
  | `Rewarding_Restart` | d2 | 0.4451 | 0.4210 | **−0.0241** | **[−0.0401, −0.0080]** | significant loss |
  | `Rewarding_Restart` | d5 | 0.3017 | 0.3097 | **+0.0080** | **[+0.0007, +0.0154]** | significant gain |
  | `RoundRobin_Random` | d2 | 0.3625 | 0.3611 | −0.0014 | [−0.0029, +0.0001] | flat |
  | `RoundRobin_Random` | d5 | 0.2838 | 0.2845 | +0.0007 | [−0.0086, +0.0100] | flat |

* **Diagnosis** — the first per-dim split where *both* CIs exclude
  zero.  The population-based LBC arm pays at 5-D (exactly the GOAL
  §5.2 hypothesis for why the CMA-ES arm was flat at quick-2-D), and
  costs at 2-D×1000-eval where the sixth arm dilutes the bandit's
  budget away from the NelderMead/JSO refinement that dominates
  late-budget 2-D AOCC.  Note the budget interaction: at 2-D×200
  evals (quick) the arm *leaned positive* (+0.0092) — early-phase
  diversity helps, long-budget dilution hurts, which is an anytime
  (AOCC) effect, not a dimension effect alone.  The ledger's
  control-spec accepts generalized in sign only to the regime the
  nightly loop measures (quick 2-D); they said nothing about d5,
  where the real gain turned out to live.

* **Consequences queued** — (a) the gain is real but conditional:
  the natural shippable form is *dimension/budget-gated arm
  activation* (a structural mix that includes population arms only
  when `dim ≥ 3`-ish or budget/dim clears a threshold) — same
  mechanism GOAL §5.5 anytime-scheduling wants, and the CMA-ES §5.2
  open question should be re-measured at d5 with this instrument
  before designing it; (b) the nightly loop's quick-2-D regime
  cannot see d5 effects at all — this is now the second measured
  instance (after the JSO d5 add) where the decisive evidence lived
  outside the nightly battery, reinforcing GOAL §4.4 "move up the
  measurement ladder".

* **Rejection memory** — the slot's record was *superseded*, not
  appended: `codify-reject --metric aocc --class-name NLSHADE_LBC
  --op add_heuristic --date 2026-08-11` replaced the 2026-08-10
  "policy-moot" entry with this A/B's numbers.  The old record
  suppressed the slot on a policy technicality dated one day before
  the measurement, so under the `--min-fresh-nights` resurrection
  gate the slot would have come back citing the technicality rather
  than the evidence that actually settled it; the rejection date now
  matches the decision date, as the gate's contract assumes.  Drive-by:
  the `_comment` literal in `_cmd_codify_reject`'s payload still
  carried the pre-2026-08-08 "all evidence nights on or before
  rejected_on" wording, so every write silently reverted the file's
  header to the stale semantics — synced to the `--min-fresh-nights`
  text.

* **Validation** — full pytest suite green, `ruff check` clean on
  touched files, `ruff format --check` clean.  No optimizer-behavior
  change ships; `planning/` + `TODO.md` + the `codify-reject` header
  literal only.

### 2026-08-10 — `drop_heuristic JSO` codify slot REJECTED (training-seed artifact, #4); NLSHADE_LBC→control slot policy-moot

* **What** — Daily session banked the top actionable codify slot after
  dedup — **`drop_heuristic JSO` on `Rewarding_Restart`** (2 confirmed
  nights 2026-08-04 / 2026-08-10, pooled CI95% `[+0.0058, +0.0075]`) —
  applied it, measured it with the 12-seed paired quick instrument,
  found it **flat**, and reverted in-branch (PR #297; apply + revert
  commits cancel, so the net source diff carries no optimizer-behavior
  change).  Rejection recorded via `codify-reject --metric aocc` dated
  2026-08-10.  A second, policy-moot rejection was recorded for
  `add_heuristic NLSHADE_LBC` → `RoundRobin_Random` (see below).

* **Measurement** (12-seed decision roster, quick battery,
  `run --decision-seeds` + seed-paired `compare`):

  | strategy | before | after | Δmean | sd | CI95 | verdict |
  |---|---|---|---|---|---|---|
  | `Rewarding_Restart` (− JSO) | 0.3476 | 0.3503 | +0.0026 | 0.0249 | [−0.0132, +0.0185] | ~ noise |
  | `RoundRobin_Random` (control) | 0.3345 | 0.3344 | −0.0001 | 0.0042 | [−0.0027, +0.0026] | ~ noise |

  Training seed 42 alone: **+0.0200** (matching the ledger accepts)
  while per-seed deltas flip sign across the roster
  (−0.0471 @ 11 … +0.0369 @ 555) — the same training-battery-artifact
  signature as the three previous rejections (Sensitivity ×2
  2026-08-03, NelderMead 2026-08-07).  **Fourth consecutive**
  single-seed `min_nights=2` slot rejected flat by the multi-seed
  instrument; the codify pipeline's screening evidence has a measured
  0/4 hit rate at the current plateau.  Additional prior against the
  drop: JSO was *added* 2026-08-02 (PR #289) on standard-battery
  evidence (+0.0222 overall, **d5 +0.0287**, d5 finally beating the
  random floor); the quick 2-D battery structurally under-weights the
  regime where the arm earns its keep, so a quick-only positive would
  not have shipped this reversal anyway.

* **NLSHADE_LBC slot** — today's scan also surfaced `add_heuristic
  NLSHADE_LBC` with `strategy_names = [RoundRobin_Random]` only
  (3 confirmed accepts, 2 nights).  Not applied and recorded
  policy-moot: `RoundRobin_Random` is the pure-random reference spec
  and every local A/B's flat control — mutating the measuring stick
  invalidates the measurements (2026-07-30 judgement call, reaffirmed
  2026-08-02, 2026-08-07, and here).  The underlying signal (the
  NLSHADE_LBC arm is strong under AOCC) is real and worth pursuing on
  the *candidate* spec: queued in TODO as a §4.3 measured change
  (12-seed quick + standard d5 A/B, no ledger slot needed).

* **Dedup** — `Sensitivity.update_interval` (1 fresh post-rejection
  night; the resurrection pattern PR #295 gates) skipped;
  `drop_analyzer Sensitivity` (apply-guard no-op; PRs #293/#295)
  skipped; `drop_heuristic NelderMead` (PR #294 rejection) skipped.
  Post-session `codify-scan --metric aocc`: 3 surfaced, all covered by
  open PRs — the actionable queue is empty.

* **Next** — (a) merge the open instrument PRs (#293–#296); with
  #295's k≥2 fresh-night gate and #296's `--sync-eval`, tonight's
  single-seed accepts stop resurrecting dead slots.  (b) Measure
  `add_heuristic NLSHADE_LBC` on `Rewarding_Restart` directly
  (quick 12-seed + standard-battery d5) — first structural candidate
  with a positive prior that does not touch the control.  (c) The 0/4
  screening hit rate says the nightly loop's per-night value is now
  bounded by its single-seed design: prioritise the TODO item
  "multi-seed confirm in the nightly loop" (base-seed rotation or a
  2-seed confirm gate) over more codify attempts.

### 2026-08-09 — Scheduling noise quantified (repeat-sd 0.021 at fixed seed) and halved: `--sync-eval` synchronous-harvest mode

* **What** — Measurement-substrate session (no optimizer behavior
  change; the new mode is opt-in and default-off).  Today's three
  codify-scan candidates were all already covered (NelderMead → open
  PR #294; both `Sensitivity` slots → 2026-08-03 rejections + open
  PRs #293/#295), so per GOAL §4 step 2 this attacked the sharpest
  measured gap: the substrate noise named by the last three sessions.

* **Finding 1: the "per-seed" decision noise is scheduling
  nondeterminism, not instance sensitivity.**  10 repeated quick-battery
  runs on an *identical* tree at the *same* seed (42):

  | strategy | repeat mean | repeat sd | range |
  |---|---|---|---|
  | `Rewarding_Restart` | 0.3536 | **0.0206** | 0.0602 |
  | `RoundRobin_Random` | 0.3333 | 0.0012 | 0.0037 |

  At a fixed seed the entire ~0.019 "per-seed sd" of the 2026-08-03/05
  instrument reproduces — so it was never instance-family noise; the
  adaptive strategy is simply nondeterministic run-to-run.  Per-instance
  breakdown: instances 0 and 2 carry the variance (sd 0.033 each),
  instance 1 is comparatively stable (0.009).  Every +0.01-scale ledger
  accept at seed 42 lives inside this band — which is *why* 67% of
  screening accepts get overturned by the same-night confirm gate and
  all three recent codify candidates re-measured flat on 12 seeds.

* **Finding 2: source ranking (scratch experiments, seed 42, 10
  repeats each).**  Mechanisms, in `panobbgo/core.py`: (a)
  `_run_threaded_evaluation` harvests whichever futures are `done()`
  per loop pass; (b) `jobs_per_client` is derived from wall-clock
  `avg_time_per_task`, so batch-size targets jitter; (c)
  `_collect_points_safely` polls heuristic queues on wall-clock sleeps
  while EventBus handler threads (one per `on_*` method) mutate bandit
  state concurrently; (d) heuristics draw from the *shared global*
  `np.random` inside those threads, so interleaving reorders the
  stream even at a fixed seed.

  | variant | `Rewarding_Restart` repeat sd (seed 42) |
  |---|---|
  | baseline (threaded) | 0.0206 |
  | + synchronous future harvest | **0.0094** |
  | + harvest + eventbus queue-drain wait | 0.0113 (no further gain) |

  The queue-drain result is a real negative: queue-empty ≠ handler
  finished, so polling the queues buys nothing — a true stepping mode
  needs handler-completion tracking.  The residual is consistent with
  (c)+(d).

* **Shipped: `--sync-eval`** — `scripts/ioh_benchmark.py run
  --sync-eval` → `run_ioh_harness(..., sync_eval=True)` →
  `config.sync_evaluation` (YAML `evaluation.sync`, default False) →
  `_run_threaded_evaluation` blocks on all submitted futures before
  harvesting.  Cross-seed validation of the shipped path (8–10 fixed-
  seed repeats per cell, `Rewarding_Restart`):

  | seed | threaded sd | sync sd | ratio |
  |---|---|---|---|
  | 42 | 0.0206 | 0.0090 | 2.3× |
  | 1234 | 0.0190 | 0.0152 | 1.25× |
  | 777 | 0.0138 | 0.0100 | 1.4× |
  | **pooled** | **0.0183** | **0.0115** | **1.6× sd (2.5× variance, F≈2.5, p<0.05)** |

  So the honest claim is a **~1.6× pooled repeat-sd cut,
  heterogeneous by seed** — not the 2.3× the seed-42-only experiment
  suggested.  A single null 12-seed paired A/B per mode could *not*
  resolve the improvement (threaded sd 0.0158 vs sync 0.0193; n=12 sd
  estimates carry ±30% error) — the instrument-level CI shrink is
  expected asymptotically but is **not yet demonstrated at N=12**;
  accumulate nulls before leaning on it.  Result JSONs carry a
  `sync_eval` tag (legacy files read back as False) and `compare`
  warns loudly on a mode mismatch so nobody A/Bs across regimes.
  Sync harvest shifts mean AOCC slightly (+0.005 at 42, +0.015 at
  1234, −0.007 at 777 — different trajectories, expected); per the
  ledger-continuity rule the nightly loop stays on threaded mode
  until a deliberate switch (TODO).

* **Validation** — 7 new tests (round-trip incl. legacy files, CLI
  mismatch warning + silence on match, e2e sync run to ≥90% budget,
  default-off tag); full suite green; ruff clean.  Default-off path:
  the only new code on it is a `getattr(..., False)` check.

* **Next** — (a) per-heuristic `np.random.Generator` streams seeded
  from (run seed, heuristic name) to kill source (d) — mechanical,
  ~14 modules, measure with the same fixed-seed repeat protocol
  (which is the right instrument for substrate work; paired-delta
  nulls at N=12 cannot resolve <2× changes); (b) use `--sync-eval`
  on both sides of codify/frontier A/Bs and accumulate null-A/B
  evidence for the instrument-level gain; (c) only after (b) shows a
  demonstrated CI shrink, promote it into the nightly loop and codify
  verification (deliberate regime switch, logged).

### 2026-08-08 — Rejected codify slots now need k ≥ 2 *post-rejection* nights to resurrect (`--min-fresh-nights`)

* **What** — Tooling-only session (no optimizer behavior change;
  `make_ioh_strategies` untouched).  The rejection memory's
  resurrection rule was "any single evidence night newer than
  `rejected_on` re-surfaces the slot".  It is now gated:
  `CodifyRejection.suppresses()` counts the distinct evidence nights
  strictly *after* the rejection date and keeps suppressing until they
  reach `min_fresh_nights` (new module default
  `DEFAULT_RESURRECT_MIN_FRESH_NIGHTS = 2`, mirroring the §9.3
  `min_nights` actionability bar).  Pre-rejection nights never count —
  they were adjudicated by the rejecting A/B.  CLI:
  `codify-scan --min-fresh-nights N` (default 2; `1` restores the old
  semantics), the gates line renders
  `hide_rejected(resurrect_fresh_nights>=N)`, and the
  `--include-rejected` audit view tags partially-resurrected slots
  with `[rejected <date>; fresh nights since: k, below resurrection
  bar]` so accruing evidence stays visible.  The nightly workflow
  calls `codify-scan` with defaults, so the committed
  `self_improve_codify_scan.txt` picks the gate up automatically.

* **Why (§4 step 2 — sharpest measured gap)** — The single-fresh-night
  resurrection path has a measured **0/3 hit rate** at the current
  plateau: `Sensitivity.update_interval` and `drop_analyzer
  Sensitivity` (both rejected 2026-08-03 by 12-seed paired A/Bs, mean
  d −0.0012 / −0.0007) were each resurrected by one fresh seed-42
  night (2026-08-08 / 2026-08-06), and `drop_heuristic NelderMead`
  went through the same cycle on 2026-08-07 (PR #294: 12-seed A/B
  Δmean −0.0003, seed 42 alone +0.0174).  With per-seed null-change
  sd ≈ 0.015–0.019 and every nightly accept keyed to training seed
  42, a single fresh night is *by construction* the artifact class
  the rejections named — yet it re-opened the slot, and the last
  three daily sessions were consumed re-litigating it (PRs #293,
  #294, plus today's queue).  Requiring the post-rejection evidence
  alone to clear the same k ≥ 2 bar as a brand-new candidate makes
  the queue truthful again while still letting a genuinely changed
  spec resurrect a slot after two independent nights.

* **Effect today** — `codify-scan --metric aocc`: 3 surfaced → 1.
  Both Sensitivity slots are hidden (1/2 fresh nights each); the only
  survivor is `NelderMead drop_heuristic`, which open PR #294 already
  covers and whose rejection record (dated 2026-08-07, evidence
  nights 2026-07-11 / 2026-08-07 → 0 fresh) self-suppresses on merge.

* **Validation** — Updated the two library tests to the new default
  (explicit `min_fresh_nights=1` keeps legacy semantics covered) and
  added four CLI tests (single-fresh-night hidden + gates label,
  audit progress tag, `--min-fresh-nights 1` legacy override, `< 1`
  loud failure).  Full gate green.  No paired A/B needed: the change
  touches scan hygiene only, not optimizer behavior.

* **Next** — The *pre*-rejection actionability gate is still
  single-seed (the open 2026-08-03 "price instance sensitivity into
  the codify gate" TODO): every ledger night is seed 42, so `n_nights`
  measures persistence of the same training-battery draw, not
  cross-instance generality.  Two candidate fixes for a future
  session: rotate the nightly base seed by date (cross-night pooling
  becomes cross-seed pooling for free), or auto-run the 12-seed
  paired A/B as a codify pre-gate before a slot is declared
  actionable.

### 2026-08-07 — Codify slot `NelderMead drop_heuristic` rejected (training-seed artifact; third in a row)

* **What** — Negative result, recorded so the slot stays out of the
  queue.  Today's scan surfaced two candidates.  The top one
  (`drop_analyzer Sensitivity`, resurrected by the 2026-08-06 accept
  night) is an apply-guard no-op already handled by open PR #293's
  re-rejection — skipped per §12.3 step 0 dedup.  The actionable slot
  was **drop `NelderMead` from `Rewarding_Restart`**: 2 confirmed
  accepts on 2 distinct nights (2026-07-11, 2026-08-07), pooled CI95%
  `[+0.0067, +0.0083]`, mean Δ `+0.0075`, the 2026-08-07 night measured
  on the current spec.

* **Measurement (2026-08-03 decision protocol, `--decision-seeds`)** —
  12-seed paired quick A/B of the drop against the current spec:

  | strategy | before | after | Δmean | sd | CI95 | verdict |
  |---|---|---|---|---|---|---|
  | `Rewarding_Restart` (− NelderMead) | 0.3479 | 0.3476 | −0.0003 | 0.0188 | [−0.0122, +0.0117] | ~ noise |
  | `RoundRobin_Random` (control) | 0.3341 | 0.3349 | +0.0008 | 0.0020 | [−0.0005, +0.0020] | ~ noise |

  The per-seed deltas are the diagnosis: the nightly training seed 42
  shows `+0.0174` (consistent with the ledger accepts) while the
  11-seed rest splits ± with roster mean ≈ 0 — the same
  training-seed-artifact signature that rejected both `Sensitivity`
  slots on 2026-08-03.  The drop was applied, measured, and reverted
  in-branch (`codify(aocc): drop NelderMead...` + revert commit);
  the spec is unchanged.  Rejection recorded via `codify-reject
  --metric aocc --class-name NelderMead --op drop_heuristic`
  (dated 2026-08-07) so `codify-scan` hides the slot until
  post-rejection evidence nights accrue.

* **Pattern worth acting on** — this is the **third consecutive**
  ledger-positive codify slot (both `Sensitivity` slots, now
  `NelderMead`) rejected flat by the 12-seed instrument.  At the
  current plateau the nightly single-seed evidence at `min_nights=2`
  has ~0 hit rate against the ~0.015–0.019 per-seed sd; the open
  TODO "price instance sensitivity into the codify gate" (raise
  `min_nights`, or a multi-seed confirm before a slot is surfaced
  actionable) is now the sharpest tooling gap in the codify path.
  Note `Rewarding_Restart`'s per-seed sd (0.0188) is ~9× the control's
  (0.0020) on identical-tree pairs — the adaptive strategy itself is
  the dominant noise source, which is also GOAL §5.1's instance-family
  sensitivity showing up at measurement time.

* **Next** — (a) implement the multi-seed pre-gate for codify-scan
  (cheapest version: `--min-nights 3` for structural drops paired with
  a 6-seed screening A/B in the nightly `post-loop` step); (b) per
  GOAL §4 step 4, the quick battery is saturating as an evidence source
  at this plateau — promote validation to `--standard` for the next
  algorithmic attack; (c) the GOAL §5.2 CMA-ES arm (PR #293) lands a
  new exploration surface for the bandit — let ledger nights accrue on
  it before the next codify session.

### 2026-08-06 — CMA-ES arm enters the structural catalog (GOAL §5.2); direct add to `Rewarding_Restart` measured flat at quick-2-D; `drop_analyzer Sensitivity` re-hidden (apply-guard no-op)

* **What** —

  1. **`CMAES` added to `default_structural_catalog()`'s `add_heuristic`
     candidate pool** as `(CMAES, {"sigma0": 0.3})`.  The full CMA-ES
     implementation (Hansen 2016 tutorial equations, IPOP/BIPOP restart,
     `panobbgo/heuristics/cma_es.py`, shipped in PR #274) existed but was
     unreachable by the nightly loop — the candidate pool had no
     covariance-adapting arm, so the bandit could never measure the
     strongest classical family at this regime against the DE arms.
     The `CMAES.sigma0` kwarg rule already in `default_catalog()`
     (log-uniform on [0.05, 1.0]) becomes reachable too: the catalog
     entry sets `sigma0` explicitly so the rule fires on any spec the
     bandit builds.  Default λ = 4 + ⌊3·ln n⌋ (7 at n=2) needs no
     budget-sizing override at quick scale.  Two membership tests added
     (`TestCMAESCatalogMembership`), following the JSO/LBFGSB pattern.
     No default spec changes — loop-exploration surface only.

  2. **Characterisation A/B (neutral, direct add not shipped)** — 12-seed
     paired quick A/B (`--decision-seeds`, the 2026-08-05 instrument) of
     `Rewarding_Restart + CMAES(sigma0=0.3)` vs the current spec:
     Δmean `+0.0005`, sd `0.0186`, CI95 `[-0.0113, +0.0123]` → `~ noise`
     (before 0.3457 ± 0.0166, after 0.3462; `RoundRobin_Random` control
     flat at `+0.0013` CI `[-0.0003, +0.0030]`).  Per the
     no-unmeasured-or-neutral-ships guardrail the direct spec addition
     was reverted.  Expected: the quick battery is 2-D only, where
     rotation invariance and covariance learning pay least; the arm's
     value hypothesis lives at 5-D rotated valleys (the known weak
     regime), which the nightly ledger, hold-out seeds, and standard
     battery sample — that is exactly what the catalog route measures.

  3. **Codify queue hygiene** — today's scan surfaced exactly one
     candidate: `drop_analyzer Sensitivity`, resurrected from its
     2026-08-03 rejection by the fresh 2026-08-06 single-seed accept
     night.  Re-verification shows re-litigation is unwarranted:
     `--apply-top --apply-dry-run` is an **apply-guard no-op**
     (`Sensitivity` is the last analyzer in `Rewarding_Restart`;
     dropping it would empty the bucket, which the apply engine
     refuses), and the spec is byte-identical to what the 2026-08-03
     12-seed A/B measured (mean d `-0.0007`, CI95 `[-0.0100, +0.0085]`;
     only tooling PRs landed since).  Re-recorded via `codify-reject`
     dated 2026-08-06; `codify-scan --metric aocc` reports 0 actionable
     candidates again.

* **Why (§4 step 3 — attack one gap)** — the last three sessions were
  tooling-only (decision protocol, rejection memory, multi-seed
  instrument); with the instrument mechanised the sharpest *actionable*
  gap on the books is GOAL §5.2: the bandit's structural search space
  simply lacked the one algorithm family (covariance adaptation) that
  competition-winning hybrids at this regime are built on.  This ships
  the missing arm through the measured path: nightly bandit pulls →
  ledger accepts → codify-scan evidence, with hold-out seeds guarding
  against training-seed overfit.

* **Validation** — full suite green (includes the two new catalog
  membership tests), `ruff check` / `ruff format --check` clean; the
  A/B numbers above were captured with
  `scripts/ioh_benchmark.py run --quick --decision-seeds` on both trees
  and the seed-paired `compare`.

* **Next** — (a) watch the nightly `add_heuristic` bandit posterior for
  CMAES pulls (the arm starts cold; the 71-attempt `drop_analyzer` rule
  currently dominates the structural posterior); (b) scan-tooling gap
  noticed today: `codify-scan` surfaces candidates whose apply would be
  guard-suppressed (no-op) as "actionable" — annotate or hide
  guard-suppressed slots so `--apply-top` sessions don't burn a slot on
  them; (c) unchanged from 2026-08-05: deterministic
  `evaluation_method="serial"` benchmark mode, and §5.1 hold-out /
  instance-family generalisation.

### 2026-08-05 — Paired multi-seed decision instrument mechanised in `ioh_benchmark.py` (`--seeds` / `--decision-seeds` / `--reps` + seed-paired `compare`)

* **What** — Tooling-only session (no optimizer behavior change).  The
  2026-08-03 decision protocol — N ≥ 12 paired quick seeds, per-strategy
  mean/sd/CI95 of the per-seed deltas, flat-control check; or ≥ 5
  standard replicates per side — existed only as prose in this log and
  had to be hand-rolled (bash seed loops + manual stats) by every
  session that measured anything.  It is now the tool's native mode:

  1. `panobbgo/harness_ioh.py` gains `IOHMultiSeedResult` (JSON
     round-trip with a `"multi_seed": true` discriminator),
     `run_ioh_harness_multi_seed()`, `paired_seed_stats()` (pairs
     per-seed per-strategy mean AOCC *by seed value*, t-distribution
     CI95 on the deltas), and `DEFAULT_DECISION_SEEDS` — the canonical
     12-seed roster from the 2026-08-03 rejection measurements
     (42, 7, 1234, 2025, 3, 11, 99, 123, 777, 2024, 31337, 555).
  2. `scripts/ioh_benchmark.py run` gains `--seeds N1 N2 ...`,
     `--decision-seeds` (expands to the roster), and `--reps K`
     (battery-replicate override for standard-battery decisions);
     `compare` detects the multi-seed format and prints the paired
     per-strategy table (before/after mean, Δmean, sd, CI95, verdict
     `+ improved` / `- regressed` / `~ noise`) plus per-seed deltas;
     mixed single/multi comparisons fail loudly with rc 2.  Single-seed
     files and workflows are byte-compatible with before.

* **Why (§4 step 2 — sharpest gap)** — Today's codify queue is empty:
  all 5 scan-surfaced candidates are recorded rejections/moot (LBFGSB,
  Center 2026-07-30; Sobol.n moot; both Sensitivity slots 2026-08-03)
  with no post-rejection evidence nights, and PR #291 (open) already
  covers the rejection-memory tooling.  The sharpest measured gap on
  the books is the measurement substrate itself: threaded evaluation
  gives a per-seed null-change sd of ~0.015 AOCC (2026-08-03 finding),
  so single-run A/Bs cannot resolve the ~+0.01 effects codify decisions
  chase.  Until the nondeterminism source is fixed architecturally,
  the multi-seed paired instrument *is* the decision instrument — it
  deserved to be one flag, not a bespoke script per session.

* **Validation** — 15 new tests (multi-seed aggregation/round-trip,
  paired-stats semantics incl. seed-value pairing, partial overlap,
  n=1 NaN CI, no-common-seeds error, CLI dispatch single/multi/mixed +
  `--fail-on-regression`); `tests/test_harness_ioh.py` 41 passed.
  Instrument null-check on the current tree (identical code both
  sides, 12 decision seeds, quick battery): per-strategy mean Δ within
  the noise floor with CI95 straddling zero and `~ noise` verdicts for
  both strategies — numbers in the PR body.

* **Next** — (a) the architectural fix (deterministic evaluation mode)
  is still open — `_run_threaded_evaluation` collects whichever farmed
  futures happen to be `done()` each loop pass, so the strategy's
  result view depends on OS scheduling; a synchronous `evaluation_method
  = "serial"` would remove that source for benchmark runs and is the
  natural follow-up.  (b) With the instrument mechanised, attack GOAL.md
  §5.1 (hold-out / instance-family gap) and §5.2 (CMA-ES arm) using
  `--decision-seeds` from the start.

### 2026-08-04 — Codify-scan rejection memory (`codify-reject` + evidence-scoped suppression)

* **What** — Shipped the "scan hygiene" fix both the 2026-08-02 and
  2026-08-03 sessions named as the top tooling gap: `codify-scan` now
  consults a per-metric *rejection memory* so slots an operator
  A/B-rejected (or declared moot) stop re-surfacing every night.
  Tooling-only change — no optimizer behavior touched, so no paired
  A/B applies (AGENTS.md evidence rules cover optimizer-behavior
  changes); gate was the full test suite + lint.

  * `panobbgo/self_improve.py`: `CodifyRejection` (slot key
    `class/param/op` + optional `direction` restriction + `rejected_on`
    date + reason + log_ref), `load_codify_rejections` (loud
    `ValueError` on any malformed entry — the file gates automated
    suppression), `annotate_rejected_status`, and
    `rejections_path_for_metric` (`composite` →
    `planning/self_improve_rejections.json`, `aocc` → `…_aocc.json`,
    mirroring the ledger-stem convention).  `CodifyCandidate` grows
    `rejected` / `rejected_on` / `rejection_reason` (also in
    `to_dict`, so `--json` consumers can filter).
  * **Suppression is evidence-scoped, not permanent**: a candidate is
    hidden only while *every* contributing evidence night is on or
    before `rejected_on`.  A single accept on a later night is new
    information (the spec changed since the A/B) and resurrects the
    slot, tagged `[fresh evidence since rejection YYYY-MM-DD]` +
    a `rejection:` line so the operator re-verifies instead of
    trusting pooled stats that straddle the spec change.
  * `scripts/self_improve.py codify-scan`: loads the metric's file by
    default (`--rejections` to override, `--include-rejected` to
    audit); hidden slots are reported as `N rejected, hidden` and the
    gates line carries `hide_rejected`.  `--apply-top` picks from the
    visible list, so it now skips rejected slots automatically —
    fixing the 2026-08-02 complaint that the driver could not skip
    past the dead rank-1/rank-2 candidates.  Output is byte-identical
    to before when no rejection matches (nightly report diffs stay
    quiet).
  * New `codify-reject` subcommand appends a record (validates the
    date, refuses an equal-or-newer duplicate for the same slot,
    supersedes an older record, pretty-prints the JSON for reviewable
    diffs).
  * Seeded `planning/self_improve_rejections_aocc.json` with the five
    decided slots: `add_heuristic LBFGSB` and `drop_heuristic Center`
    (rejected 2026-07-30), `Sobol.n` (moot 2026-07-30),
    `Sensitivity.update_interval` down and `drop_analyzer Sensitivity`
    (rejected 2026-08-03).  The `Sensitivity.update_interval` record
    is direction-restricted to `down` — a future `up` signal is a
    different hypothesis and stays actionable.

* **Effect** — `codify-scan --metric aocc` now reports
  `candidates surfaced: 0 (of 9; 4 already codified, 5 rejected,
  hidden)`, which is the truth: the queue is empty until fresh
  post-rejection evidence accrues.  Before this change the same scan
  surfaced 5 "actionable" candidates, all of them re-litigations.
  22 new tests (`TestCodifyRejectionLibrary`,
  `TestCodifyScanCLIRejection`).

* **Session context (evidence review, 2026-08-04 nightly)** — the
  overnight run accepted 1/20 with best Δ +0.0125 (seed score 0.3450);
  hold-out drift CI over 52 records is `+0.0080 [+0.0037, +0.0123]`,
  0 overfit verdicts.  No scan candidate carries a post-rejection
  evidence night, so no codify slot was actionable today (verified by
  cross-checking each candidate's nights against the dated REJECTED
  entries above — now automated by this change).

* **Next** — unchanged from 2026-08-03: (a) ~~rejection-memory
  suppression list~~ **done (this entry)**; (b) find/fix the
  threaded-evaluation nondeterminism so the standard battery becomes a
  usable single-run decision instrument; (c) frontier work: GOAL.md
  §5.1 hold-out / instance-family generalization and §5.2 CMA-ES arm,
  attacked with the 12-seed paired protocol from the start.  Also
  consider pricing instance sensitivity into the scan gate
  (`min_nights` 2 → 3+ / multi-seed confirm) so future ledger evidence
  clears the ~0.015 per-seed null sd before surfacing.

### 2026-08-03 — Codify queue cleared: both `Sensitivity` slots rejected (12-seed paired A/B); standard battery found nondeterministic run-to-run

* **What** — Worked the two remaining actionable codify candidates from
  the nightly scan (both with a post-Sobol-drop 2026-07-31 evidence
  night, so no evidence-baseline mismatch) and rejected both with a
  12-seed paired quick-battery A/B on the current spec.  No optimizer
  behavior change ships from this session; the deliverable is the two
  negative results plus a measurement-substrate finding.

  1. **`Sensitivity.update_interval 25 → 20` — REJECTED** (negative
     result).  Ledger: 2 confirmed nights (2026-07-26, 2026-07-31),
     pooled CI95% `[+0.0092, +0.0117]`, mean Δ `+0.0104`.  Local paired
     A/B over 12 quick-battery seeds (42, 7, 1234, 2025, 3, 11, 99,
     123, 777, 2024, 31337, 555): `Rewarding_Restart` mean Δ
     **−0.0012**, sd 0.0150, CI95% `[−0.0097, +0.0072]`; untouched
     `RoundRobin_Random` control mean Δ −0.0013.  Per-seed deltas
     ranged −0.0404 … +0.0146 with sign flips — the ledger effect is a
     training-battery artifact, not a generalizable gain.
  2. **`drop_analyzer Sensitivity` — REJECTED** (negative result).
     Ledger: 2 confirmed nights (2026-07-14, 2026-07-31), pooled CI95%
     `[+0.0067, +0.0075]`.  Same 12-seed design (before baselines
     reused): mean Δ **−0.0007**, sd 0.0163, CI95%
     `[−0.0100, +0.0085]`; control mean Δ +0.0009.  The `Sensitivity`
     analyzer is AOCC-neutral on the current 4-heuristic spec — keep
     it (it is the only analyzer left feeding the rewarding strategy).

* **Measurement-substrate finding: the standard battery is
  nondeterministic run-to-run.**  Re-running `ioh_benchmark.py run
  --standard` on an *identical* tree and seed moved both arms by
  ≈ −0.015 (`Rewarding_Restart` 0.3291 → 0.3136, control
  0.3406 → 0.3264), and a before/after standard pair moved the
  untouched control by +0.0185.  Threaded evaluation makes a *single*
  standard run unable to resolve the ~+0.01 effects codify decisions
  chase — this bounds the confidence of the single-run standard A/Bs
  used as the 2026-07-30 decision basis (their |Δ| ≥ 0.015 calls stand,
  but +0.01-scale calls made this way would be coin flips).  The quick
  battery is near-deterministic (control |Δ| ≤ 0.001 in 21/24 paired
  runs; occasional ±0.01 outliers), so **the preferred decision
  instrument is now: N ≥ 12 paired quick seeds, report mean/sd/CI95,
  verify the control is flat** (~2.5 min wall-clock).  For standard-
  battery decisions, use ≥ 5 replicates per side until the
  nondeterminism source is fixed.

* **Queue state after this session** — all 6 scan-surfaced candidates
  resolved: `JSO` banked (open PR #289), `LBFGSB` + `drop Center`
  rejected 2026-07-30, `Sobol.n` moot (Sobol dropped), both
  `Sensitivity` slots rejected today.  The scan still re-surfaces the
  rejected/moot slots every night because it has no rejection memory —
  scan hygiene (a rejected-slot suppression list keyed like the
  already-codified suppression) is now the top tooling gap.

* **Instance-family sensitivity, quantified** (GOAL.md §5.1) — the
  per-seed treated-arm delta sd of ~0.015–0.016 on a *null* change is
  the noise floor any real quick-battery improvement must clear; the
  two nightly evidence nights per candidate (both keyed to the seed-42
  training battery) sit well inside it.  Raising the codify gate's
  `min_nights` (2 → 3+) and/or adding a multi-seed confirm step to the
  scan would price this in automatically.

* **Next** — (a) rejection-memory suppression list for codify-scan;
  (b) find/fix the threaded-evaluation nondeterminism so the standard
  battery becomes a usable decision instrument; (c) with the queue
  empty, the frontier is GOAL.md §5.1 (hold-out / instance-family
  generalization) and §5.2 (CMA-ES arm) — attack with the 12-seed
  paired protocol from the start.

### 2026-08-02 — Codify: add `JSO({'NP_init': 'auto'})` to `Rewarding_Restart` (d5 finally clears the random floor)

* **What** — Banked the `JSO [add_heuristic]` codify slot from the aocc
  ledger into `panobbgo/harness_ioh.py::make_ioh_strategies`:
  `(JSO, {"NP_init": "auto"})` appended to the `Rewarding_Restart`
  heuristic bucket.  Applied manually (not via `--apply-top`) because
  the scan's top two candidates — `add_heuristic LBFGSB` and
  `drop_heuristic Center` — are the *already-rejected* 2026-07-30
  negatives that keep resurfacing (see "scan hygiene" below), and the
  driver has no way to skip to the n-th candidate.

* **Evidence (per AGENTS.md "Agent-driven improve X PRs")** —
  * Ledger: 2 confirmed accepts on 2 distinct nights, pooled CI95%
    `[+0.0092, +0.0133]`, mean Δ `+0.0112`; both measured
    `JSO({'NP_init': 'auto'})`.  Crucially the 2026-08-01 accept
    (`Rewarding_Restart`, Δ `+0.0092`, confirmed on a fresh
    randomize_iteration) was measured on the *current* post-Sobol-drop
    spec, so the interaction hazard that sank LBFGSB/Center does not
    apply to that record.  (2026-07-19 accept was on
    `RoundRobin_Random`, pre-drop — supporting, not primary.)
  * Local paired A/B, standard battery (dims 2+5, 5 instances,
    500·d budget): `Rewarding_Restart` mean AOCC `0.3374 → 0.3596`
    (+0.0222), positive at *both* dims — d2 `0.3840 → 0.3996`
    (+0.0156), d5 `0.2909 → 0.3196` (+0.0287).  `RoundRobin_Random`
    control flat (`0.3262 → 0.3252`).  Battery mean `0.3318 → 0.3424`.
  * Quick battery (4 paired seeds: default, 1, 2, 3): mean Δ `+0.0013`
    (noise-flat) — consistent with the documented quick-battery
    instance-sensitivity hazard from the 2026-07-30 Sobol call;
    the standard battery is the decision battery for structural edits.

* **Why it works** — d5 was the spec's weakest regime (barely above
  the random floor since the Sobol drop: 0.2909 vs 0.2898 in this
  session's before-run).  jSO is an L-SHADE-lineage adaptive DE — the
  strongest classical family at exactly this budget/dim regime (GOAL.md
  §5.2 names the lineage) — and `NP_init="auto"` sizes its population
  from the strategy budget (the 2026-07-05 composite-track result).
  With it, d5 jumps +0.0287 and now clearly beats random (0.3196 vs
  0.2878).

* **Guard-rails observed** — edit scoped to `Rewarding_Restart` only;
  `RoundRobin_Random` is the pure-random reference spec and stays
  untouched (2026-07-30 judgement call, reaffirmed).  One codify slot
  in this PR; nothing else shipped.

* **Scan hygiene (open item for the next session)** — `codify-scan`
  re-surfaced `LBFGSB [add]` (rank 1) and `Center [drop]` (rank 2)
  even though both were A/B-rejected on 2026-07-30; their ledger
  evidence all predates the Sobol drop and the scan has no rejection
  memory.  Until a "rejected slots" suppression list exists (natural
  spot: a small JSON consulted by `codify-scan` next to the ledger),
  every session must cross-check scan candidates against the dated
  REJECTED entries in this log before applying.  `Sobol.n → 38` is
  moot (seeder dropped) and also still surfaces.  The two `Sensitivity`
  candidates (tune `update_interval` 25→20 vs drop the analyzer) are
  mutually contradictory — both have a post-drop night (2026-07-31),
  so the right move is an A/B of the two against each other, not
  applying whichever scans higher.

* **Next** — (a) rejected-slot suppression for codify-scan; (b) settle
  the `Sensitivity` contradiction with a two-way A/B; (c) hold-out
  instance-family gap remains the top research target (GOAL.md §5.1);
  (d) with a DE arm now in the portfolio, the GOAL.md §5.2 CMA-ES arm
  is the next structural catalog candidate.

### 2026-07-30 (second session) — Codify queue worked through: drop `Sobol` accepted; add `LBFGSB` / drop `Center` rejected (interaction negatives); structural-add driver hardened

* **What** — Sequential processing of the four queued aocc codify
  candidates, each verified with a paired A/B on the *current* spec
  before landing (the ledger evidence for later queue entries was
  collected against earlier spec states, so interactions must be
  re-checked after every landing):

  1. **`drop_heuristic Sobol` — ACCEPTED.**  3 confirmed nights
     (07-13/27/29, pooled CI95% `[+0.0075, +0.0175]`), each measured on
     that night's ladder *after* the drop-Restart accept, so the
     evidence baseline matches the current spec.  Standard-battery A/B:
     `Rewarding_Restart` mean AOCC `0.3342 → 0.3518` (+0.0176; d2
     +0.0253, d5 +0.0099); the spec now beats the random floor at both
     dims (d5 was *behind* pure random before: 0.2981 vs 0.3124).
     Note: the 3-fixed-instance quick battery said the opposite
     (−0.009 over 3 seeds) — overruled by the standard battery + the
     randomized ledger nights.  Quick-battery instance sensitivity is
     now a documented hazard for accept/reject calls.
  2. **`add_heuristic LBFGSB({"warm_start": True})` — REJECTED**
     (negative result).  2 nights of evidence on `Rewarding_Restart`
     predate the Sobol drop.  On the leaner spec: standard battery
     `0.3518 → 0.3372` (−0.0146, consistent at both dims), control
     flat.  Interpretation: numerical-gradient polish (dim+1 evals per
     L-BFGS-B step) costs more anytime-AOCC than it recovers now that
     `NelderMead` is the sole local refiner on a smaller portfolio.
     Mirrors the 2026-07-06 cold-LBFGSB composite negative.
  3. **`drop_heuristic Center` — REJECTED** (negative result).  2
     nights of evidence predate the Sobol drop.  Standard battery
     `0.3518 → 0.3289` (−0.0229); d5 falls back below random (0.2769
     vs 0.2904).  With Sobol's 32-point sweep gone, `Center`'s single
     deterministic box-centre evaluation is a load-bearing cheap
     prior, not dead weight.
  4. **`Sobol.n 32 → 38` — MOOT** (the seeder it tunes was dropped).

* **Driver hardening (shipped in the same session)** — the first real
  structural *add* through `codify-scan --apply-top` exposed three
  defects, all fixed with 8 new tests
  (`TestStructuralAddDriverFixes`):

  * **Missing-comma bug**: inserting into a compact single-line bucket
    (`heuristics=[(Random, {})]`) produced the call expression
    `(Random, {})(LBFGSB, {})` — syntactically parseable, crashes at
    import.  The insertion now supplies the comma itself.
  * **`structural_kwargs` dropped**: the edit wrote `(LBFGSB, {})`
    where every contributing arm measured
    `LBFGSB({"warm_start": True})` — shipping the *cold* variant the
    2026-07-06 negative result already ruled out.
    `CodifyCandidate` now carries `structural_kwargs_list` +
    `consensus_structural_kwargs()` (majority, ties → most recent) and
    the scanner renders them project-style.
  * **No import management**: the added class was never bound in the
    factory (→ `NameError` at first use).  `add_*` edits now also
    rewrite the factory's `from panobbgo.heuristics import ...` /
    `.analyzers` statement with the class inserted in sorted position
    (one import edit per factory; skipped when already bound).
  * **Parse-validation net**: `apply_codify_edits` refuses to write
    any `.py` result that does not `ast.parse`, so a future primitive
    bug surfaces as a loud skip instead of a corrupted registry.

* **Guard-rail judgement call** — the LBFGSB candidate's
  `strategy_names` included `RoundRobin_Random` (1 night); the edit
  was *not* applied there even while testing: that spec is the pure-
  random reference (GOAL.md §1 criterion 1 and every local A/B's
  control).  Mutating the measuring stick invalidates the measurements.
  Consider a codify-scan exclusion list for reference specs.

* **Net effect on the metric of record** — standard battery
  `Rewarding_Restart`: session start `0.3342` → session end `0.3518`
  (+0.0176 from the Sobol drop; the two rejected candidates would have
  given back −0.015/−0.023 of it had they shipped unverified).

* **Next** — the queue is empty; the nightly loop now mutates a
  3-heuristic spec (`Random` / `Nearby` / `NelderMead`) + `Sensitivity`
  analyzer.  Fresh evidence will key on the new baseline.  Top research
  target remains the hold-out instance-family gap (GOAL.md §5.1);
  d5 remains barely above the random floor (0.3080 vs 0.3039) —
  GOAL.md §5.2 (CMA-ES arm) is the natural attack.

### 2026-07-30 — Metric-aware codify routing + first aocc codify (drop `Restart` from `Rewarding_Restart`) + goal contract

* **What** — Three coupled ships that un-stall the aocc-era codify last
  mile:

  1. **Metric-aware codify routing.**
     :func:`panobbgo.self_improve.default_codify_registries` and
     :func:`panobbgo.self_improve.default_codify_apply_sources` gain a
     `metric: str = "composite"` parameter.  `"aocc"` routes the
     suppression predicate to
     :func:`panobbgo.harness_ioh.make_ioh_strategies` and the
     `--apply-top` edit driver to `panobbgo/harness_ioh.py`
     (`_DEFAULT_APPLY_SOURCES_AOCC`); `"composite"` (default) is
     byte-identical to the historical behaviour.
     `scripts/self_improve.py codify-scan` threads its existing
     `--metric` selector into both call sites.  Six new tests in
     `TestMetricAwareCodifyRouting`.
  2. **First aocc codify.**  Applied via the fixed driver: dropped the
     `(Restart, {...})` analyzer from the `Rewarding_Restart` spec in
     `make_ioh_strategies`.  Evidence: 18 confirmed `drop_analyzer`
     accepts across **17 distinct nights** (2026-07-09 → 2026-07-30),
     pooled CI95% `[+0.0084, +0.0147]`, every record's `ci_low > 0`.
     Local paired A/B on the quick IOH battery: `Rewarding_Restart`
     mean AOCC `0.3538 → 0.3922` with the untouched
     `RoundRobin_Random` control flat (`0.3338 → 0.3339`).
     Interpretation: under the anytime AOCC metric the
     diverse-restart basin jumps cost more mid-budget precision than
     they recover — the Sensitivity analyzer stays.
  3. **Visibility + goal contract.**  The nightly workflow now also
     regenerates and commits `planning/self_improve_codify_scan.txt`
     (metric-aware) so the current actionable evidence is readable
     from the repo without running anything, and `planning/GOAL.md`
     ships as the durable goal contract for agent-driven sessions:
     metric of record, per-session operating loop, multi-day
     escalation ladder, SOTA-informed research backlog (MA-BBOB /
     LLaMEA / modular-CMA-ES context, 2026-07 snapshot).

* **Why** — The 2026-07-09 flip pointed the nightly loop at the IOH
  registry, but every codify-pipeline default still pointed at the
  composite factories in `panobbgo/harness.py`.  Result: three weeks
  of flat seed scores (~0.33) while the bandit re-discovered,
  re-confirmed, and re-forgot the same improvements every night —
  `codify-scan --apply-top` scanned the wrong file, found no matching
  spec, and silently no-oped.  The suppression predicate was equally
  wrong-registried, which both hid a live candidate
  (`add_heuristic LBFGSB`, 3 nights — surfaced by the fix) and
  mis-annotated others.

* **Measured impact** — quick IOH battery mean AOCC `0.3438 → 0.3631`
  (+0.0193); competition-candidate spec `+0.0384`.  Consistent with
  the ledger's per-night mean Δ of `+0.0113` on the same op.

* **Next** — the codify queue behind this one (in evidence order):
  `drop_heuristic Sobol` (3 nights), `add_heuristic LBFGSB`
  (3 nights), `drop_heuristic Center` (2 nights), `Sobol.n 32 → 38`
  (2 nights, only meaningful if Sobol survives).  One slot per PR;
  re-measure after each since the drops interact.  See
  `planning/GOAL.md` §4 for the standing protocol.

---

*Entries before this point were moved to [`planning/done/SELF_IMPROVEMENT_LOG_pre-2026-07-30.md`](SELF_IMPROVEMENT_LOG_pre-2026-07-30.md) on 2026-08-11 to keep this file readable. Nothing was deleted — the archive is the same newest-first format.*
