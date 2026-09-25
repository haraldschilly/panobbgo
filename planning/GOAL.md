# GOAL — Converge on a world-class black-box optimizer

**Audience**: any agent (or human) picking up this repository with the standing
instruction "improve panobbgo". This file is the durable goal contract; read it
first, then act through the operating loop below. The code referenced here
exists and is tested unless it is marked removed — no setup beyond
`uv sync --extra dev` and `cd tools/ioh_worker && uv sync`.

---

## 1. North star (the measurable goal)

Maximize **mean AOCC** (Area Over the Convergence Curve, the IOHprofiler /
MA-BBOB Anytime competition metric) of the best panobbgo strategy on the
MA-BBOB battery, at competition-style budgets, without regressing the frozen
`composite_score` contract on the internal battery.

Concretely, in priority order:

1. **Beat the internal floor**: `RoundRobin_CMAES` (the competition candidate
   spec in `panobbgo/harness_ioh.py::make_ioh_strategies`) must dominate
   `RoundRobin_Random` on every battery tier (quick / standard / full).
   `Rewarding_Restart`, the previous candidate, is kept in the registry as
   a portfolio control.
2. **Beat the external baselines**: `Baseline_SciPyDE` and
   `Baseline_SciPyAnneal` (`panobbgo/harness_baselines.py`) on mean AOCC at
   the standard battery. Random search is the hard floor — never lose to it.
3. **Approach competition level**: the MA-BBOB Anytime competition regime is
   budget `2000·d`, dims 2 and 5, ~1000 affine instances. World-class 2024/2025
   entries were LLaMEA-generated hybrids (LLM-designed metaheuristics — i.e.
   structurally the same loop this repo runs on itself). The long-run target is
   a strategy competitive with tuned modular CMA-ES / L-SHADE-lineage hybrids
   with warm-started local search at that regime.

**Metric of record**: mean AOCC from `scripts/ioh_benchmark.py` and the
paired multi-seed decision runs (`harness_ioh.paired_seed_stats`). `composite_score` is the frozen
legacy contract — keep it green, don't optimize for it.

## 2. State snapshot (2026-09-10 — update when it materially changes)

All section references are to `planning/DISCOVERY_2026-09-09.md`.
This snapshot covers §1–§31.  The later results (§32–§53, through
2026-09-14: the Splitter rework, the families / constrained / noisy /
high-dimension / plain-BBOB batteries, the budget series and the oracle
regime gate) are not folded in yet; read them there.

* **Population size was the single largest effect on this codebase.**
  The L-SHADE family shipped `18·dim` populations — roughly five times
  too large for a few hundred evaluations per dimension, so most of the
  budget went into the initial fill and the success-history adaptation
  never got its generations.  The constructor default is now
  `NP_init="auto"` = `3·dim·(budget/(500·dim))^¼`, floored at 6, with a
  per-class coefficient of `4·dim` for NLSHADE_LBC.  Accepted on the
  12-seed roster, each arm solo and paired on one RNG stream: **L-SHADE
  +0.230, jSO +0.204, NLSHADE_LBC +0.064** and a further **+0.052** at
  `4·dim` (§17, §20, §24).
* **CMA-ES restarts itself.**  Solo CMA-ES had no termination criterion:
  52 % of the budget was spent after the last improvement, and on two of
  five *d* = 5 instances σ diverged against its clamp for 92 % of the
  run.  Hansen's reference criteria are in (worth +0.0001 on their own —
  they are written for 10⁴·d budgets), and the σ-divergence detector is
  on by default: **+0.026 [+0.008, +0.043], 11/12 seeds** (§23).  The
  earlier `ipop_factor` result is **retracted** (§16).
* **The measurement instrument was repaired a second time.**  The
  harness set the budget *after* constructing the heuristics, so
  budget-adaptive arms sized themselves from the config default; and
  run seeds were hashed from the **spec name**, so any two variants of
  an arm ran on different RNG streams and a dead parameter still showed
  a nonzero delta.  Fixed by `StrategySpec.seed_name` (one stream per
  arm).  The **null floor is now measured**: ±0.05 for CMA-ES, ±0.03 for
  a DE arm, on a 3-seed battery mean.  And the CI of a *screen maximum*
  carries no weight — it is the CI of a selected winner, not of a
  pre-registered spec (§18, §18a, §30).
* **The arms are level, and they win different instances.**  Oracle on
  the final defaults, 12 seeds, 120 cells: jSO 0.656, NLSHADE_LBC 0.647,
  L-SHADE 0.642, CMA-ES 0.642 — four arms inside 0.014, i.e. inside the
  floor.  Headroom over the best single arm **+0.074 [+0.048, +0.100],
  12/12**; best *oracle* pairs CMA-ES + jSO ≈ CMA-ES + LBC — though as
  an actual relay pair only CMA-ES + jSO works (§31).  The cells sort by
  *dimension*: LBC owns four of the five *d* = 2 instances, CMA-ES the
  hard *d* = 5 ones, jSO the rest — a context signal available before
  the first evaluation.  PSO drops out of the candidate set (§28).
* **Sharing removes the portfolio's structural penalty; it does not buy
  a lead.**  `StrategyBlockBandit` (one arm owns a block of evaluations,
  scored by the AOCC area it bought) plus the opt-in `Archive` analyzer
  and `warm_start` on CMA-ES, the DE family and PSO.  A **cold** two-arm
  portfolio loses **−0.08** to its best arm with the CI clear of zero;
  with **both** arms warm it reaches parity — `Blocks_uniform_cj_warm2`
  0.685 vs CMA-ES alone 0.666, **+0.019, 8/12, CI including zero**.  The
  lean is entirely at *d* = 5 (+0.03); at *d* = 2 a single arm converges
  before a relay could help (§20, §25, §27).
* **Once sharing is on, rotation beats the bandit.**  D-UCB's job was to
  minimise switching cost, and sharing removes that cost: a switch
  becomes a relay.  `soft_be25`'s +0.050 on three seeds was the winner's
  curse — **−0.006** on the roster (§26, §29, §30).
* **The selection policy is measured out, and it is worth nothing.**
  Screen #7 closed the design space around it: `tail_frac` is flat
  across 0 → 0.6 (retiring §29's "relay then let the leader run"
  mechanism), and at `tail_frac=0` the soft D-UCB *is* round-robin —
  identical in 23 of 30 cells, the mean gap being one cell.  Value
  estimate, bonus, discount, hysteresis and tail contribute nothing
  measurable at any setting tried.  Block length is flat from 20 to 50
  evaluations absolute (with per-cell collapses either side),
  `only_if_better` **hurts** (−0.073 — it starves the arm that most
  needs the relay) and is off by default, a third arm still dilutes at
  *d* = 2, and **CMA-ES + LBC is not the relay pair** (+0.034 at
  *d* = 2, −0.050 at *d* = 5: LBC's `4·dim` population does not fit a
  25-evaluation block).  jSO stays.  **The policy is worth nothing; the
  sharing is worth everything** (§31).
* **Verdict for 500·dim MA-BBOB, *d* ∈ {2, 5}: a two-arm sharing
  portfolio is level with the best single arm, not above it.**
  `RoundRobin_CMAES` stays the flagship.  `Blocks_warm_CMAES_JSO`
  replaces `Rewarding_Restart` as the second harness spec;
  `Rewarding_Restart` at 0.35 is no longer a useful control.
* **§5.2 is retracted** (2026-09-09).  It concluded from a flat A/B that
  the CMA-ES arm did not pay — but the A/B *added* CMA-ES as a seventh
  arm to the same budget.  Nobody had run it alone.
* The first instrument repair (PRs #307–#315) still stands: seeded runs
  are bit-identical (they varied by up to 0.09 AOCC), a default-config
  run spends its whole budget (it used to stop after ~4 %), `Config` no
  longer leaks between specs, population heuristics no longer lose
  emitted points, and the standard battery is ~2× faster.
* The nightly self-improvement loop was disabled on 2026-08-13 and
  **removed on 2026-09-25**: an audit found its screening CI had zero
  width on the nightly battery (reps = 1), its codify gates could not
  fail, and its guard compared scores across instance draws. Its design,
  ledgers and diagnosis are in `planning/done/`.

## 2b. Previous snapshot (2026-08-11)

The nightly-loop era (2026-07-09 … 08-13) is documented in
`planning/done/LOOP_DIAGNOSIS_2026-08-11.md` and
`planning/SELF_IMPROVEMENT_LOG.md`. Its lasting findings: effects are
regime-heterogeneous (the same change can be significantly negative at
d2 and positive at d5, #298), and the "hold-out gap" was a unit bug.

## 2c. Plan of record (set 2026-09-10)

Phase A (strong individual arms) and Phase B (the selection policy) are
both closed on this battery: the arms are level and the sharing
portfolio is level with them.  What is left is the regime where the
measured lean says the gain is real, and the regimes nobody has
measured at all.  In order:

1. **Larger budgets and *d* ≥ 10.**  The portfolio's whole advantage
   sits at *d* = 5 (+0.03 on every roster) and vanishes at *d* = 2,
   where 1000 evaluations end before a relay can matter (§27, §30).
   That gradient predicts a *real* gain where a single arm cannot
   converge inside the budget.  The cheap version, available today, is a
   **dimension-gated spec** — portfolio for *d* ≥ 5, single arm below
   (`gate_min_dim` already exists).  Ship the gate, then measure at
   *d* = 10 / 20 and at `2000·d`.
   *Status 2026-09-14:* a regime gate shipped as
   `StrategyBlockBandit(regime_gate="oracle:<class>")` (§45; the in-run
   probe `"table-v1"` is not implemented); *d* = 10 / 20 at `2000·d` is
   in §38, the budget series 100…2000·dim in §44 and §46.
2. **Do not spend more on the selection policy.**  The `tail_frac` /
   block-length / pair screen is done and came back empty (§31): tune
   nothing there without a new mechanism to point at.  The standing rule
   it leaves behind applies to everything below — the best spec of a
   screen is a *selected maximum*; re-check it on fresh seeds before
   believing its margin (AGENTS.md).  §30 is what happens when that is
   skipped.
3. **Constrained and noisy problems remain unmeasured.**  Every number
   in §2 is continuous, box-constrained MA-BBOB.  Panobbgo's constraint
   machinery, the noisy-objective path and the plain-BBOB suite (§5.4)
   have no battery at all, and a portfolio may well pay where a single
   arm's assumptions break.
   *Status 2026-09-14:* all three batteries exist now, with first
   results — constrained families §34 and §44, noise §38 and §42,
   plain BBOB (`kind=bbob`, 24 functions) §52 and §53.
4. **The composite registry's three CMA-ES specs remain a frozen
   contract**, pending Harald's decision.  Do not touch them to chase an
   AOCC number.

## 3. Operating loop (one agent session ≈ one iteration)

```bash
# 0. Sync + sanity (both venvs; IOH tests skip without the worker venv)
uv sync --extra dev && (cd tools/ioh_worker && uv sync)

# 1. Screen a candidate cheaply (few seeds), e.g. benchmarks/portfolio_screen.py
# 2. Re-check the winner on fresh seeds (paired_seed_stats)
# 3. A/B on the metric of record
uv run python scripts/ioh_benchmark.py run --quick --output /tmp/before.json   # on master
#   ...apply change...
uv run python scripts/ioh_benchmark.py run --quick --output /tmp/after.json
uv run python scripts/ioh_benchmark.py compare /tmp/before.json /tmp/after.json
# 4. Push to a branch; CI gates tests / lint / typecheck / docs / format
# 5. One PR per change, evidence in the body (AGENTS.md
#    "Agent-driven improve X PRs")
```

## 4. Cadence guardrails

* Say in the PR what was measured; bug fixes to documented behaviour need
  no benchmark.
* One change per PR. Independent evidence ≠ joint evidence — don't batch
  three "individually positive" changes into one unmeasured combination.
* The composite-score formula and the default randomized battery are frozen
  contracts. Extend via opt-in flags, never edit.
* Log every measured result — negative ones included — in
  `planning/DISCOVERY_2026-09-09.md`; unlogged negatives get retried.

## 5. Research backlog (SOTA-informed, 2026-07)

Ordered by expected value; each item should enter through the loop above.

1. **Regime-conditional strategy selection** — *promoted 2026-08-11, replacing
   the retracted "instance-family generalization" item.*  The 0.33-vs-0.04
   hold-out gap that headed this list from 2026-07-30 was a **metric-unit
   bug**, not a finding (§2; fixed in #299, real gap 0.3383 vs 0.3342).  The
   genuine, measured structural problem is that effects differ in *sign*
   across regimes: the NL-SHADE-LBC arm moved d2 by −0.0241 [−0.0401, −0.0080]
   and d5 by +0.0080 [+0.0007, +0.0154] (#298), and the same arm leaned
   positive at 2-D×200 evals while losing at 2-D×1000.  A single global spec
   scored by a scalar mean cannot express the improvement that exists — the
   objective has a flat optimum by construction, which is the deeper reason
   the loop's first 34 nights yielded one +0.005 change.  Deliverables in
   order: (a) per-dim cells in the accept rule — **shipped, #302**;
   (b) budget-phase cells (needs AOCC recomputed on trajectory slices, which
   `trace_evals`/`trace_fx` already support); (c) ~~teach codify-scan to
   read the per-cell breakdown~~ — moot, codify-scan was removed with the
   loop on 2026-09-25; (d) dimension-gated arm activation — **shipped
   2026-08-12** (`gate_min_dim`/`gate_max_dim` in `StrategySpec`;
   NLSHADE_LBC gated to d≥5 in `Rewarding_Restart`, pooled d5 evidence
   +0.0070 [+0.0027, +0.0112]); budget-gating and the CMA-ES arm at d5
   remain open.

   **Answered in part 2026-09-10 (§27, §28, §30).**  The regime split is
   now measured on the arms themselves, not just on the accept rule: the
   four level arms win *different cells sorted by dimension* (LBC four of
   five *d* = 2 instances, CMA-ES the hard *d* = 5 ones), and the sharing
   portfolio's entire advantage is at *d* = 5.  Dimension is known before
   the first evaluation, so this is directly actionable — a
   dimension-gated portfolio spec is item 1 of the plan of record (§2c).
   Deliverable (b) and budget-gating remain open.
2. **CMA-ES arm** — ~~*shipped 2026-08-06*~~ **retracted 2026-09-09.**  The
   original item recorded that adding the `CMAES` heuristic to
   `Rewarding_Restart` was flat on a 12-seed paired quick-2-D A/B
   (CI95 [-0.0113, +0.0123]) and asked whether the arm earns its pulls.
   The question was wrong: the A/B added a *seventh* arm to a fixed
   budget.  Run alone, CMA-ES scores 0.580 against the portfolio's 0.352
   on the standard battery.  It is now the competition candidate
   (`RoundRobin_CMAES`).  The open work is the opposite of what this item
   assumed — see the new item 8.

3. **Rank-based acceptance stats** — *shipped 2026-08-11 (#301)* as
   `statistical_accept(accept_stat="rank")` / `--accept-stat rank`: one-sided
   Wilcoxon signed-rank on the per-pair deltas shifted by `eps_accept`, with
   Hodges-Lehmann as the paired location estimate.  Demonstrated to reject an
   outlier-driven composite the mean rule accepts, and to accept a broad win
   the mean rule rejects.  **Removed 2026-09-25** with the loop, never
   enabled.  Friedman across (function, instance) remains unexplored.
4. **Plain-BBOB cross-validation battery** — 24 BBOB functions, dims
   {2, 3, 5, 10}, as an opt-in hold-out suite (the `ioh` package already
   provides them through the same worker protocol).
   *Shipped 2026-09-14* as `IOHBatterySpec.fids` / `portfolio_screen.py
   kind=bbob`, first measured at dims 2 and 5 (§52, §53).
5. **Anytime-aware strategy scheduling** — AOCC rewards early descent;
   panobbgo's rewarding strategy re-weights on "new best" events only.
   Explore time-decayed rewards / explicit budget-phase schedules.

   **Answered 2026-09-10 (§25, §26, §29, §30)** — this shipped as
   `StrategyBlockBandit`: a pull is a *block* of ~50 evaluations rather
   than one point (so a λ-point generation is no longer capped at `1/λ`
   estimated value), and the reward is the AOCC area the block bought,
   anytime and scale-free.  The measured lesson is not the one this item
   expected: once the arms **share** their evaluations through the
   `Archive`, plain rotation beats the D-UCB rule, because a switch stops
   being a cost and becomes a relay — and screen #7 then found the
   bandit contributes **nothing** at any setting tried, `tail_frac`
   included (§31).  What is live is warm start + rotation + a block
   length somewhere in 20–50 evaluations absolute.  Time-decayed rewards
   specifically are still unexplored, but the prior on them is now poor.
6. **Behavior-space diagnostics** (LLaMEA-SAGE direction) — log per-run
   trajectory features (dispersion, basin-jump counts) into the run
   records so a screen can correlate *why* an arm wins, not just that it does.
7. **Learned "intuition layer" (Dynamic Algorithm Configuration)** — a small
   dense policy network that watches run-time progress and re-weights /
   switches heuristics mid-run.  `StrategyRewarding`'s bandit is the
   memoryless special case; the generalization is a stateful policy:

   * **Observations** (per batch of results): budget fraction consumed,
     best-so-far improvement rate over the last k batches, stagnation
     length, per-heuristic reward distribution, point-cloud dispersion,
     Splitter depth stats — cheap features already derivable from the
     eventbus (`on_new_results` / `on_new_best`).
   * **Action**: a weight vector over the active heuristics (drop-in for
     the rewarding strategy's bandit weights), optionally a restart /
     phase-switch signal.
   * **Training**: evolution strategies (CMA-ES / OpenAI-ES) over the
     policy weights — no backprop through the optimization run needed.
     Fitness = mean AOCC over a stratified batch of randomized instances
     (`harness_randomized` families / MA-BBOB instances); hold-out
     base seeds catch policy overfit.  A few thousand policy evaluations × quick-battery cost is
     feasible on the existing parallel harness.
   * **Deliverable path**: (a) feature extractor as an analyzer publishing
     a `progress_features` event, (b) `StrategyLearned` consuming it with
     a hand-set linear policy (sanity baseline), (c) ES meta-training
     script writing the trained weights as a versioned artifact, (d) the
     trained policy enters the decision roster as one more spec —
     measured, not trusted.  Item 6's diagnostics are the natural feature
     source, so build 6 first.

   Literature anchors: Dynamic Algorithm Configuration (Biedenkapp et al.,
   2020+), adaptive operator selection, learning-to-optimize; Nevergrad's
   NGOpt is a hand-crafted (non-learned) version of the same switching
   idea.

8. **Does a portfolio ever pay?** — *new 2026-09-09.*  **Answered
   2026-09-10 (§20, §25, §27, §30): "only if the arms share what they
   paid for, and then it is level, not ahead."**  A cold two-arm
   portfolio loses −0.08 to its best arm with the CI clear of zero — a
   portfolio of independent solvers cannot beat its best member, it can
   only dilute it.  Turning on the shared `Archive` and warm-starting
   **both** arms removes exactly that penalty: `Blocks_uniform_cj_warm2`
   0.685 vs CMA-ES alone 0.666, +0.019, 8/12 seeds, CI including zero.
   Sharing is the mechanism; the bandit is not (rotation beats D-UCB
   once sharing is on, §25.4).  The remaining lean is at *d* = 5, which
   is what item 1 of the plan of record now chases.  The historical
   record of the *cold* result follows.

   *Block allocation was tested and does not rescue it.*  The hypothesis
   was that a mix loses only because interleaving starves the population
   dynamics, so giving each method a contiguous stretch with
   `StrategyPhased` should help.  Measured (3 seeds, standard battery,
   paired against CMA-ES alone): CMA-ES→NLSHADE_LBC **−0.0118**
   [−0.0971, +0.0734], NLSHADE_LBC→CMA-ES **−0.1321**, Sobol→CMA-ES
   **−0.3788**.  Handing CMA-ES a warm start from another method is worse
   than letting it start itself; a Sobol' design phase is much worse.

   What remains open: (a) a problem class where a mix wins — multimodal,
   noisy, constrained, or much higher dimension.  Test with the
   plain-BBOB suite (item 4) and constrained problems, neither of which
   any battery currently covers.  (b) A *warm-started local polish* after
   CMA-ES converges — the one phased variant that could not be measured,
   because `LBFGSB` spawns a subprocess and the driver script lacked an
   ``if __name__ == "__main__":`` guard.  Worth re-running; the
   interleaved form of it (`CMA-ES + LBC + L-BFGS-B`) was +0.12 over the
   old portfolio but well below CMA-ES alone.

References: MA-BBOB generator (Vermetten et al., ACM TELO 2024);
IOHprofiler competitions (iohprofiler.github.io/competitions); LLaMEA
(arXiv:2405.20132); CEC winners longitudinal analysis (arXiv:2603.24140);
benchmarking best practice (arXiv:2007.03488).

## 6. Done criteria (revisit quarterly)

The goal is met for a given quarter when:

* mean AOCC of the best spec on the **standard** IOH battery improves
  quarter-over-quarter with a bootstrap CI excluding zero, and
* the best spec beats `Baseline_SciPyDE` and `Baseline_SciPyAnneal` on the
  same battery, and
* every tuning change says what it was measured on.
