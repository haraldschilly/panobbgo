# GOAL — Converge on a world-class black-box optimizer

**Audience**: any agent (or human) picking up this repository with the standing
instruction "improve panobbgo". This file is the durable goal contract; read it
first, then act through the operating loop below. The code referenced here
exists and is tested unless it is marked removed — no setup beyond
`uv sync --extra dev` and `cd tools/ioh_worker && uv sync`.

---

## 1. North star (the measurable goal)

**Direction since 2026-09-26** (`planning/DESIGN_roadmap_2026-09-26.md`):
the primary battlefield is **expensive, parallel evaluations at small
budgets** (10…200·dim, q workers, failing calls); the cheap-evaluation
MA-BBOB track below stays the **floor** every change must hold.  The
target claim: never much worse than the best single solver on any
class, clearly better on average, on a sealed test set, against the
real incumbents (pycma, NGOpt, Optuna, BoTorch/Ax, SMAC3, HEBO).

The cheap-evaluation track, as set before:

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

## 2. State snapshot (2026-09-25 — update when it materially changes)

All section references are to `planning/DISCOVERY_2026-09-09.md`; this
snapshot covers §1–§53.  **Every number below was measured before the
2026-09-25 audits** and is a pre-audit value: re-measure once on the
current code before comparing anything against it (see the last bullet
and §2c item 1).

* **The arms.**  `NP_init="auto"` for the DE family and the CMA-ES
  σ-divergence restart are 12-seed accepted defaults (§17–§24).  Paired
  oracle on the shipped defaults at 500·dim MA-BBOB: jSO 0.649, CMA-ES
  0.637, L-SHADE 0.631, LBC 0.615; oracle headroom **+0.081, 12/12**,
  78 % of it in the pair CMA-ES + jSO (§43, replacing §28's unpaired
  numbers — `oracle.py` had pinned `seed_name` per arm, §41).
  `RoundRobin_CMAES` is still the flagship.
* **Sharing is a low-budget effect.**  `Blocks_uniform_cj_warm2`
  (CMA-ES + jSO, `Archive` + `warm_start`, block rotation) minus
  CMA-ES alone on MA-BBOB *d* ∈ {2, 5}: **+0.056 at 100·dim (12/12, and
  12/12 against jSO and L-SHADE)**, +0.037 at 200·dim (9/12), +0.011 at
  300, +0.019 at 500 (parity), +0.002 at 2000·dim (§44.1, §46.1).  The
  *d* = 2 dip at 200–300·dim was a short-block stall, not a budget
  effect (§46.2, §50.3).
* **The value is the hand-over, and it is a ratchet.**  Warm vs cold at
  the same block: +0.076 (100·dim) / +0.138 (200·dim), 12/12.  Each
  direction alone is worth a third to two fifths of that; together
  31–47 % more than the sum — each warm arm improves the pool the other re-seeds from
  (§48.1).  Point-level seams (`inject`, `shared_pbest`) are four times
  smaller and need a channel block rotation does not have (§47).  The
  ratchet grows with hand-off count at 100·dim (+0.053 → +0.098 from
  4 to 50 blocks) and saturates at 200·dim; blocking alone costs
  < 0.006 (§50.1–§50.2).
* **The payload is m and σ, nothing else.**  Re-seed from the top-k
  crowd: a diverse selector is indistinguishable from no hand-off
  (§48.2).  All four ways to set C are measured and the identity reset
  wins — archive shape, shrunk archive shape and the arm's own C all
  lose at *d* = 5 (§48.3, §49, §51.1).  The σ floor repairs the
  short-block *d* = 2 stall (7 → 2 cells) but is not a default (§51.2).
  At 100·dim the short block (`n_blocks=50`) beats `block_evals="auto"`;
  at 200·dim `auto` is as good — block length should depend on budget
  (§50.4, §51.3).
* **Allocation carries nothing.**  D-UCB and a third arm still lose at
  200·dim (−0.028 / −0.020, §46.3); rotation is the allocator (§31).
* **Selection by regime is the direction.**  The best arm flips with
  noise, dimension and budget (§38, §41): outliers belong to CMA-ES
  (DE arms −0.135, 0/12), uniform noise to the portfolio (+0.026 vs the
  best arm, 9/12), constrained leans jSO (§42, §44.2).  The oracle gate
  `regime_gate="oracle:<class>"` shipped and is **free** — gated to one
  arm it is bit-identical to that arm alone, so the gate is worth
  exactly its table (§45).  The in-run probe `"table-v1"` is not built.
* **The function axis changes the picture (§52, §53).**  24 plain BBOB
  functions (`kind=bbob`) score 0.27–0.31 against MA-BBOB's 0.49–0.53
  at 200·dim: the mixtures averaged away the dispersion a portfolio
  lives on; per function the portfolio ranges +0.22 … −0.24, losing
  where CMA-ES is already good (f5, f7).  The §53 decision run (6 seeds,
  576 cells per budget): **sharing accepted at 100·dim against both
  arms** (`warm2_auto` +0.019 vs CMA-ES, +0.026 vs jSO, 6/6); at
  200·dim jSO overtakes CMA-ES and the portfolio only ties it.  Against
  the average arm it is +0.02 at both budgets; against the per-cell
  oracle −0.015 / −0.039.  **The remaining value is in selection, not
  in more sharing.**
* **Other batteries.**  Families, constrained, noisy and *d* = 10/20
  batteries exist (§34, §38); no regime there shows the portfolio
  leading a single arm except uniform noise and low budget.  The
  Splitter now scales its resolution with budget (§35, §37, 12-seed
  accepted); the DE arms' cross-process nondeterminism was a thread race
  in `Results.add_results`, fixed (§40).
* **Engineering state, 2026-09-25.**  Two full audit rounds (PRs
  #320–#334) repaired measurement integrity (runs stop at exactly
  `max_eval`, composite `success` means "within the budget", IOH
  aggregates count timeouts/crashes) and optimizer fidelity (jSO,
  NL-SHADE, CMA-ES, PSO follow their papers; the default constraint
  penalty is `fx + 100·cv`), so **all earlier result files must be re-measured
  once before comparisons** (TODO "Re-baseline once").  Since #337
  measurements run with synchronous evaluation and module RNG streams
  are keyed per module.  The 12-seed decision-roster rule is dropped: a
  paired multi-seed comparison with delta, CI and wins/n is enough, and
  a screen's best spec is re-checked on fresh seeds
  (`doc/dev/benchmarking.md`).  The nightly self-improvement loop is
  removed (design in `planning/done/`).

## 2b. Previous snapshot (2026-09-10, §1–§31)

Population size was the largest single effect (`NP_init="auto"`:
L-SHADE +0.230, jSO +0.204, §17–§24); CMA-ES got termination criteria
and a σ-divergence restart (+0.026, 11/12, §23); `StrategySpec.seed_name`
put variants of an arm on one RNG stream and the null floor was measured
(±0.05 CMA-ES, ±0.03 DE, §18).  At 500·dim MA-BBOB the four arms were
level and a two-arm sharing portfolio was level with them (+0.019,
8/12, §27); rotation beat the bandit and the selection policy was worth
nothing (§26–§31).  §5.2 was retracted.  The 2026-08-11 nightly-loop
snapshot is in `planning/done/LOOP_DIAGNOSIS_2026-08-11.md`.

## 2c. Plan of record (set 2026-09-25, re-ordered 2026-09-26)

*Program so far:* 2026-09-09 discovery (robustness and effectiveness
problems, DISCOVERY log) → quality push (docs, simplification) → search
for a strong default setup (§2, §2b) → 2026-09-25 audits → 2026-09-26
roadmap.

**2026-09-26:** the roadmap (`DESIGN_roadmap_2026-09-26.md` §5) comes
first after the re-baseline: instrument (external baselines, virtual-clock
parallel simulator, failure-region families, sealed test set, feature
logging), measure against the incumbents, then failure regions (D),
learned probe→select cycles with forecast allocation (A+B), new sharing
payloads (C).  The items below remain the cheap-track research line.

Sharing is settled as a low-budget mechanism and its payload is fully
characterised; what is left is choosing the arm per problem.  In order
(TODO.md "Research line"):

1. **Re-baseline first.**  Re-measure the composite quick/standard and
   the IOH / family references on the post-audit code, with sync
   evaluation and keyed RNG streams, before any comparison that relies
   on an older number.
2. **Probe / regime detector — the main line.**  Target: the per-cell
   oracle gap of +0.015…+0.039 (§53.2); the signal should be observable
   early, as the first arm's progress rate (§52.4).  The oracle gate
   (§45) is the harness it plugs into; the probe's only price is its
   own evaluations.
3. **Hand-over without covariance reset** — keep the arm's adapted C
   and move only m/σ.  Note §51.1 measured `warm_start_keep_cov`
   (paths zeroed) negative everywhere on the pre-audit code; a re-test
   needs a new form or the re-baselined instrument to say something new.
4. **Surrogate pre-selection** (lq-CMA-ES) as a new building block —
   the seam that reduces evaluations rather than adding ranking
   information (§47.3).
5. **Broader suite before default decisions.**  `regime_gate=
   "oracle:clean"` and `block_evals="auto"` for `Blocks_warm_CMAES_JSO`
   (§45.1, §46.4) wait for Harald's broader suite
   (`planning/DESIGN_suite_2026-09-14.md`: *d* 10/20, full BBOB,
   synthetic families with swept knobs, constrained/noisy as own axes).
6. **Standing rules.**  No further bandit tuning without a new
   mechanism (§31); the composite registry's three CMA-ES specs stay a
   frozen contract pending Harald.

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
# 5. One PR per change, evidence in the body (doc/dev/benchmarking.md,
#    "Evidence for a PR"); review and merge per doc/dev/process.md
```

## 4. Cadence guardrails

Evidence rules and frozen contracts: `doc/dev/benchmarking.md`; one
change per PR and logging every result, negatives included:
`doc/dev/process.md`.

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
   dimension-gated portfolio spec was item 1 of the 2026-09-10 plan of record; it shipped as the oracle regime gate (§45).
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
   is what the 2026-09-10 plan of record chased (answered by the budget series, §44–§46).  The historical
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
