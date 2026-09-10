# Design: a meta level that decides where to look next (2026-09-10)

Read-only analysis of the shared store, the "where to look" heuristics and
the block scheduler, then a design for `MetaAnalyst`.  Produced by an Opus
subagent; citations are against the working tree of
`claude/optimizer-headroom` (`99c1a0b`).  Status: **design, not accepted —
§3 argues that two of the four proposed mechanisms are a re-labelling of
`RegionUCB`, and §4 names the control that decides it.**

The idea, from the project owner: *after a fraction of the budget — a
quarter, a tenth, or when no large progress is happening any more, or a
combination — a heuristic starts that ANALYSES where there is still
something to gain, or where the search has not looked properly yet; it
builds a model and hands candidate points to the solvers.  Jump to a meta
level, analyse, compute a model, break it back down into concrete next
steps.*

## 0. The finding

Three things constrain this design before a line is written.

1. **The design space around *when to decide* is measured out and came
   back empty.**  §31: the bandit contributes nothing at any setting
   tried, `tail_frac` is flat across 0 → 0.6, block length is flat from
   20 to 50.  §30: the one screen CI that excluded zero was the maximum
   of fourteen specs and returned to zero on the roster.  A trigger rule
   (`budget_fraction` vs `stagnation` vs both) is *another* when-to-decide
   knob, and the prior on it is poor.  What has measured is **sharing**:
   cold → warm is +0.051 (§21), +0.094 (§25), and −0.080 → +0.019 for the
   whole portfolio (§20 → §27).
2. **The `Splitter` cannot answer "where has the search not looked" at the
   dimensions that matter.**  `limit = max(20, max_eval/dim²)`
   (`splitter.py:60`), a leaf splits at `limit` points
   (`splitter.py:393`) and each child inherits the parent's points it
   contains (`splitter.py:469-471`), so a leaf holds between `limit/2` and
   `limit` points and the tree settles at **≈ 1.3·dim² leaves,
   independent of the budget**: ~5 at *d* = 2, ~35 at *d* = 5, ~130 at
   *d* = 10, ~530 at *d* = 20.  At *d* = 2 with budget 1000, `limit` is
   **250**, so the root cannot split before evaluation 250 — the proposed
   `budget_fraction(0.25)` trigger fires into a tree with *one leaf*.  At
   *d* = 20, 530 boxes is 530^(1/20) ≈ 1.4 cuts per axis: the partition
   cannot resolve one split per dimension, and "volume per point" is not
   a coverage statistic there.  The window where a kd-tree coverage scan
   means anything is roughly *d* ∈ [5, 10].
3. **Two of the four mechanisms already ship, running continuously.**
   `RegionUCB` is a UCB1 bandit over `Splitter` leaves with in-leaf
   sampling, re-selected on **every** result batch
   (`region_ucb.py:103-125,150-170`); `GaussianProcessHeuristic` refits a
   Matérn GP and emits 5 EI/UCB points on every batch
   (`gaussian_process.py:261-269`); `QuadraticWlsModel` fits a quadratic
   on the best box at every `new_best_box` (`quadratic_wls.py:116-135`);
   `Random` samples inside the best leaf (`random.py:36-40,54-56`).
   Mechanisms (a), (b) and (c) of the brief exist as *streams*.

So the design must state, falsifiably, what a *decision at a chosen
moment* adds over those streams, and it must not spend a 12-seed roster
finding out that the answer is "nothing".

## 1. What "where is there still something to gain" can mean

All four are computed against state the shared store already holds.  Costs
are per *firing*, not per batch — that is the whole point of a meta level.

### (a) Unexplored volume — `Splitter` leaves, count vs volume

`Box.log_volume` is memoised (`splitter.py:315-327`) and `len(box)` is a
list length (`splitter.py:438-439`), so the statistic is free:

    share_i   = log_volume_i − root.log_volume      # log fraction of the box
    deficit_i = log(max(n_i,1)/N) − share_i         # < 0 ⇒ under-sampled

Log form, because `volume` overflows/underflows past *d* ≈ 15 and the
problem box is not a unit cube.  **Cost:** O(#leaves) = O(dim²) ≈ 5 … 530
float operations, microseconds at every dimension and every budget in the
plan of record.  **Emits:** *k* uniform draws inside the `argmin deficit`
leaves, `who = "MetaAnalyst:void"`.

Two caveats that shrink this to near-nothing:

* An **empty leaf essentially cannot exist** under this splitter — a leaf
  is only created by splitting a full box, and both children inherit the
  points they contain (`splitter.py:469-471`).  So `RegionUCB`'s
  `counts == 0 → explore first` branch (`region_ucb.py:111-113`) is close
  to dead code, and *emptiness* is not the signal; *density* is.
* Splits cut at the **mean** coordinate of the widest differing dimension
  (`splitter.py:452-461`), i.e. the partition is drawn *by* where the
  search already looked.  A leaf with a large volume and few points is
  frequently just the outside of the basin the arms are in — which is
  where the objective is worst.  "Under-sampled" and "worth sampling" are
  not the same predicate, and nothing in the store distinguishes them.

### (b) Promising-but-shallow — leaf best near the incumbent, few evaluations

    rank_i    = rank of penalty(box_i.best) among leaves    # scale-free
    shallow_i = rank_i is small AND n_i is small

Rank rather than a distance to the incumbent, because "close to the
incumbent" needs a scale the optimizer does not have; `RegionUCB` already
takes rank for the same reason (`region_ucb.py:115-119`), and the block
strategy's anchored log-precision (`blocks.py:403-417`) is the other
scale-free option.  **Cost:** O(#leaves) plus one `get_penalty_value` per
leaf — the constraint handler call is the only non-trivial term and it is
called ≤ 530 times.  Microseconds.  **Emits:** the leaf's `best` as a
*seed*, or the leaf's **box** as a region (§2), `who =
"MetaAnalyst:shallow"`.

`Archive.per_leaf_best(k)` already returns the best point of each of the
*k* best leaves (`archive.py:189-213`); it sorts by leaf quality and
ignores `n_i` entirely.  Adding the count is a two-line change to a
selector, not a meta level.

### (c) Model-predicted — a global surrogate with an acquisition function

The honest cost table, from the code that exists:

* **GP.**  `GaussianProcessRegressor(..., n_restarts_optimizer=10)`
  (`gaussian_process.py:287-299`) is eleven hyper-parameter optimisations,
  each with O(n³) Cholesky factorisations per likelihood evaluation, then
  `_acquire_candidates(5)` runs five L-BFGS-B ascents with `n_restarts=10`
  starts each (`gaussian_process.py:323-341`).  At *n* = 500 that is
  seconds; at *n* = 2500 (the *d* = 5 budget) tens of seconds; at
  *n* = 5000 it is minutes **per fit**.  This is why no benchmarked spec
  contains the GP arm.
  *Fix, and it is the whole trick:* fit on the **bounded** archive, not on
  everything.  `Archive` keeps `K = 256` by default (`archive.py:57`) at
  `O(log K)` per result (`archive.py:98-109`).  A 256-point GP with
  `n_restarts_optimizer ∈ {0,1}` is tens of milliseconds at any *d* in the
  plan.  Cost is then **independent of the budget**, which is the property
  a once-per-run analysis wants.
  *But:* the archive is the top-256 by penalty, i.e. a sample of the
  basin, not of the box.  A surrogate fitted on it has nothing to say
  about (a).  A meta GP that wants global coverage needs top-K **plus** a
  space-filling subsample of `splitter.root.results` — and the kernel is
  an isotropic Matérn (`gaussian_process.py:287`), which is a poor global
  model at *d* ≥ 10 whatever it is fitted on.
* **Quadratic.**  `QuadraticWlsModel` fits `1 + 2d + d(d−1)/2` terms —
  21 at *d* = 5, 231 at *d* = 20 — so it needs ≥ 231 points to be
  identifiable at *d* = 20 and O(n·p² + p³) ≈ 10⁷ flops to fit.  Cheap,
  runs out of process already (`quadratic_wls.py:35-104`), and *local by
  construction*: it is fitted on `best_box.results` with distance weights
  (`quadratic_wls.py:79-82`).  It answers "where is the bottom of this
  basin", never "where has the search not looked".

**Emits:** *k* points maximising EI/LCB inside the leaves picked by (a) or
(b), `who = "MetaAnalyst:model"`.  Restricting the acquisition optimiser to
one box is what makes it affordable and what makes it different from the
GP arm, which optimises over the whole problem box.

### (d) Divergence / stagnation of the running arms

The only signal in the list that is **not** already a shipped heuristic.
CMA-ES computes σ-divergence and two stagnation criteria internally
(`cma_es.py:88-90,1016-1021,1127`) and the L-SHADE family carries
`_gen_completed` and the success histories.  **Cost: zero** — the block
strategy already derives the portfolio-level version of this from the
result stream alone: `_best_phi`, `_arm_best[name]` per arm
(`blocks.py:423-436`), one dict update per result.  A per-arm "evaluations
since this arm last improved *the run's* best" is three more lines in the
same loop.

The caveat is the sharpest one in this section: **CMA-ES already acts on
its own divergence**, and that action was accepted on the 12-seed roster
at +0.026, 11/12 (§23).  A meta level that fires on the same signal is
double-counting unless it does something the self-restart cannot: the
self-restart re-centres on `"random"` or `"best"`, it cannot re-centre on
*a region the portfolio's other evaluations say is interesting*.  That —
and only that — is the increment.

## 2. The module

**One `Heuristic`, `MetaAnalyst`, not an `Analyzer`+`Heuristic` pair.**
An `Analyzer` has no way to emit points (`emit`/`_output` live on
`Heuristic`, `core.py:644-654,806-828`), and every extra module is a
construction-order perturbation of the RNG streams (`core.py:481-492`,
`spawn_rng` at `core.py:1370-1377`; the same reason `Archive` is opt-in,
`archive.py:36-42`).  One module, appended **last** in the spec's
heuristics list, keeps every arm's stream bit-identical (heuristics are
constructed in list order, `core.py:1311-1313`; `initialize` only *sorts
for registration*, `core.py:1321-1322`, and `subscribe` appends,
`core.py:1013`, so existing handler order is preserved too).  The name
must sort after every arm name for the registration order to be
preserved as well — `"MetaAnalyst"` > `"JSO"` > `"CMAES"` holds.

**Silence is free.**  `StrategyBlockBandit._select` only considers
`h.has_points or self._can_warm_start(h)` (`blocks.py:685`), so a
heuristic with an empty output queue is simply never given a block.  A
`MetaAnalyst` that emits nothing until its trigger is invisible to the
scheduler at zero cost, and goes silent again the moment its queue drains.
*Under `StrategyRoundRobin` this is not true*: `execute` has no
`has_points` gate and retries up to ten times with a 1 ms sleep when a
heuristic yields nothing (`round_robin.py:39-46`), so a permanently silent
arm costs latency on every pass.  **The meta level is a block-scheduler
feature**; under RoundRobin it degrades to exactly `RegionUCB`.

### Trigger

Composable predicates, evaluated in `on_new_results` — the same event the
scheduler accumulates its trace on (`blocks.py:369-393`):

```python
trigger = budget_fraction(0.25) | stagnation(window=0.10, tol=1e-8)
trigger = budget_fraction(0.25) & stagnation(...)      # "both"
trigger = never                                        # the null control, §6
```

* `budget_fraction(f)` — fires once when `len(strategy.results) >= f*max_eval`.
* `stagnation(window, tol)` — **`window` is a fraction of `max_eval`, never
  an absolute count.**  §20 is the lesson: Hansen's absolute stagnation
  window is 95 generations ≈ 30 % of the *d* = 5 budget, so it fires after
  the run has already reached AOCC's log floor.  Resolved as
  `max(4*dim, window*max_eval)`.
* **Refractory period** (`refractory = max(block_evals, 0.05*max_eval)`)
  and a **hard budget cap** (`meta_frac = 0.05`, evaluations emitted by the
  meta arm over the whole run).  Both mandatory, see §5 — `|` with a
  stagnation term can otherwise fire every block for the rest of the run.

### Output: points, or a region

**Points.**  `emit()` (`core.py:806-828`) tags each `Point` with
`self.name` only.  The reason rides in the name, following the existing
convention: `new_who` builds `"<name>:<hex>"` (`core.py:693-701`) and every
consumer splits on the first `":"` (`archive.py:226-229`,
`blocks.py:431-432`).  So a `MetaAnalyst` instance emitting under
`who = "MetaAnalyst:void"` is greppable in the ledger and correctly
attributed by the archive and by `_arm_best` with **no change to `Point`**.

**A region.**  This is the part with a real claim, and it is nearly free:
`Heuristic.archive_seed(k, *, mode=, box=)` **already takes a box**
(`core.py:720-763`), and `Archive.top_k` / `diverse_k` already filter by
one (`archive.py:126-158,160-187`), where "box" is anything with
`contains` — i.e. a `Splitter.Box` — or a `(dim,2)` array
(`archive.py:239-246`).  What is missing is one field:

    Heuristic.warm_start_box: Any = None      # consulted by the seed paths

honoured in `CMAES._warm_start_seeds` (`cma_es.py:551-564`) and
`LSHADE._warm_start_population` (`lshade.py:929`) by passing it through to
`archive_seed(..., box=self.warm_start_box)`.  ~6 lines per arm.  That is
the **only** new arm-side API in this whole design.

**Delivery.**  `MetaAnalyst` publishes `meta_region(arm=…, box=…)`;
`StrategyBlockBandit` gains `on_meta_region` which *records a pending
request* and, at the next `_open_block` for that arm
(`blocks.py:503-512`), sets `h.warm_start_box` and forces
`_should_warm_start` to `True` (`blocks.py:549-571`).  **Do not call
`warm_start_now()` from the event handler.**  The bus dispatches serially
on its own thread (`core.py:942-963,1074`) while `execute()` runs on the
main loop thread; `warm_start_now` clears the arm's output queue
(`lshade.py:1002`, `cma_es.py:671-689`) which `execute()` may be draining
at that instant (`blocks.py:735`).  Under `sync_evaluation` the bus is
idle during `execute` (`core.py:1704-1707`), but the async path is not, and
this design must not add a second reproducibility hazard.  Deferring to
`_open_block` keeps every mutation on the main thread and reuses the
scheduler's existing, tested warm-start path verbatim.

## 3. What this adds over what exists — honestly

`RegionUCB` selects a leaf by rank-quality + UCB bonus and samples inside
it, 5 points per result batch, all run (`region_ucb.py:103-170`).  That is
(a) and (b) already, with a policy.  `GaussianProcessHeuristic` is (c).
So:

* **(a) and (b) as point emitters are a re-labelling.**  A `MetaAnalyst`
  that scans the leaves and emits *k* points is `RegionUCB` with a
  different arrival schedule and a coarser score.  Write that down; do not
  headline it.
* **(c) as a point emitter is a *cost* re-labelling**, and a real one: the
  difference between refitting an O(n³) GP on every batch and fitting one
  bounded 256-point GP once is the difference between an arm nobody ships
  and an arm that is affordable.  But the *decision* it produces is the
  same object the GP arm produces.
* **(d) is genuinely new** — nothing outside CMA-ES reads CMA-ES's
  divergence — but §23 already bought most of its value with a
  self-restart.
* **The region hand-off is the only mechanism with no incumbent.**  Every
  existing heuristic acts by *emitting points that get evaluated*; the
  portfolio then sees them through the shared archive.  The meta level's
  distinctive act is to move an **already-adapted** optimizer — CMA-ES's
  covariance, jSO's success histories — into a region *without spending
  the evaluations to get it there*.  §25/§31 say sharing is the entire
  measured effect and the policy is worth nothing; a mechanism that
  changes *where the sharing points* is the only lever with evidence
  behind it.

There is a second, weaker claim worth stating because it is testable:
**concentration**.  A stream emits ~5 points per batch and competes for
pulls all run; the meta level emits 0 for three quarters of the budget and
then one coherent batch.  On a log-precision anytime metric these are not
the same object — *k* points in one place at time *T* can establish a
basin well enough to warm-start an arm into it, where a dribble of the
same *k* points cannot.  §4's `RegionUCB_arm` control is what separates
the two; if it ties, the concentration claim is dead and only the region
hand-off survives.

**Verdict to carry into the experiment:** build the region hand-off
(mechanism 3 + trigger (d)) as the headline, the leaf scan (a)/(b) as its
*selector* rather than as a point emitter, and treat the point-emitting
path as a fallback that must beat `RegionUCB` to justify its existence.

## 4. The experiment

`benchmarks/portfolio_screen.py` conventions throughout: every spec on one
`seed_name="screen"` stream so a delta carries only the strategy's effect
(`portfolio_screen.py:482-493`, `benchmark.py:143-153`), `sync_eval=True`
(`portfolio_screen.py:508-510`), rows JSON rewritten per seed, paired per
`(seed, dim, inst)` cell, 3 seeds to screen and the **12-seed roster to
decide** — the best spec of a screen is a candidate and its screen CI
carries no weight (§30, GOAL §2c item 2).

**Base spec:** `Blocks_uniform_cj_warm2` — CMA-ES + jSO, both
`warm_start="archive"`, rotation, 0.6854 on the roster, +0.019 over CMA-ES
alone (§27).  Not `soft_be25` (§30: winner's curse).

| spec | what it isolates |
|---|---|
| `Meta_off` | the base, unchanged — the bar |
| `Meta_never` | base + `MetaAnalyst(trigger=never)` — **the null**: must be bit-identical to `Meta_off`, see §6 step 0 |
| `Meta_region_b25` | region hand-off only, `budget_fraction(0.25)`, emits **no** points |
| `Meta_region_stag` | region hand-off, `stagnation(0.10)` |
| `Meta_region_either` | `budget_fraction(0.25) \| stagnation(0.10)` |
| `Meta_region_both` | `& ` — the conservative trigger |
| `Meta_points_b25` | (a)+(b) point emission, same trigger, `k = 0.02·max_eval` |
| **`Meta_random_b25`** | *k* **uniform random** points at the same trigger — the falsifier: if this ties `Meta_points_b25`, the analysis is decoration and only the jolt of exploration mattered |
| **`RegionUCB_arm`** | base + `RegionUCB` as a third arm — the attribution control for the concentration claim; also measures the third-arm tax (§26.2: 0.685 → 0.648) |

**Battery.**  Screen on `dims=5 bm=500` **only**.  Not the standard *d* ∈
{2,5} battery: at *d* = 2 the `Splitter` has ~5 leaves and cannot split
before evaluation 250 (§0.2), and §27/§30 record that *d* = 2 has nothing
to gain at all — including it halves the resolution and guarantees a
"no dimension negative" failure.  Then, per GOAL §2c item 1, the
pre-registered hold-outs: `dims=5,10 bm=500` and `dims=5 bm=2000`.

**Target cells.**  §28's majority table: jSO owns d5i0/d5i1/d5i4, CMA-ES
owns d5i2/d5i3, and §20 measured σ-divergence on two of five *d* = 5
instances for 92 % of the run.  So the cells where the portfolio's CMA-ES
arm is badly wrong are a *minority of instances*, and the screen must
print the **per-instance** *d* = 5 breakdown, not only the battery mean.

**Expected sign and size — and why the experiment may not be able to see
it.**  The oracle headroom is +0.0737 [+0.0477, +0.0998], 12/12 (§28); the
best pair captures 0.7026 and the shipped portfolio reaches 0.6854 (§27).
A meta level that repairs only the "CMA-ES badly wrong" cells is chasing
**≤ +0.017** — below the ±0.05 three-seed null floor and at the edge of
what the roster resolves.  Worse, §23's precedent says the gain will be
**heavy-tailed**: three cells at +0.18…+0.44, most cells unchanged.  The
standing accept rule (paired CI excluding zero **and** ≥ 9/12 seeds **and**
no dimension negative) is structurally hostile to a heavy-tailed
mechanism.  State this before running, not after: if the gain is real but
concentrated, the honest outcome is a **dimension-gated spec**
(`gate_min_dim`, `benchmark.py:118`) or nothing — not a re-argued
threshold.

**Drop rules, pre-registered.**

* `Meta_random_b25` within 0.01 of `Meta_points_b25` ⇒ **drop (a), (b),
  (c) as point emitters.**  The analysis carries nothing.
* `Meta_points_b25` ≤ `RegionUCB_arm` ⇒ **drop the concentration claim.**
  A stream at the same total cost is as good; ship nothing new.
* `Meta_region_*` ≤ `Meta_off` at *d* = 5 on the 12-seed roster ⇒ **drop
  the whole design.**  §31 already measured the scheduling design space
  out; this would be one more knob in it.
* Any spec with runs below 98 % of budget (the screen's own health check,
  `portfolio_screen.py`'s `short` accounting) ⇒ investigate the Splitter
  live-lock (§5) before reading any AOCC number.

## 5. Risks

| risk | mechanism | detection |
|---|---|---|
| **analysis points that never improve the best** | AOCC is a log-precision integral from evaluation 1; a point that does not lower the best contributes *zero* area **and** displaces an evaluation from a descending arm.  At *d* = 5 a 25-point meta batch is one whole block. | hard cap `meta_frac ≤ 0.05`; log AOCC-area credited to `who = "MetaAnalyst:*"` — the block reward (`blocks.py:454-465`) computes exactly this per block |
| **the meta batch also costs the interrupted arm a generation** | the batch opens a block for the meta arm; when the converging arm gets one back, `h.name != self._last_owner` fires `_should_warm_start` (`blocks.py:565-566`) and `warm_start_now` discards its in-flight generation (`lshade.py:1002`).  One firing ≈ *k* + NP evaluations ≈ 40 of 2500 at *d* = 5. | count warm starts per arm (`blocks.py:635-637` already tracks `_warm_started` / `_warm_used`); compare against the `Meta_off` counters |
| **fires too early: nothing to analyse** | at *d* = 2/1000 the root has not split at evaluation 250; at *d* = 5, 25 % of budget is 625 evaluations over ~12 leaves — volume/count on 12 leaves is noise | gate at *d* ≥ 5; assert `len(splitter.leafs) >= 8` as a firing precondition and log refusals |
| **fires too late: nothing to gain** | §20 — by the time an absolute stagnation window elapses the run is at AOCC's log floor | budget-relative window; also refuse to fire in the last `tail`-equivalent fraction |
| **region hand-off is worse than where the arm already is** | this is precisely `warm_start_only_if_better`, which measured **−0.007** on the roster (§30) — the guard is known *not* to help | log the arm's own best vs the region's best at hand-off; crossing traces in `trace_fx` (warm-better-early / worse-late), per `DESIGN_warm_start` §3 |
| **Splitter live-lock** | emitting many points inside one small leaf is exactly the input that deepened the tree without bound in §13 (`splitter.py:417-429`, `MAX_DEPTH = 60` at `:400`) | `n_evals < budget` in the screen's health block; `splitter.max_depth` vs 60 |
| **`Result.__eq__` compares only `fx`** while `__hash__` uses `(x, fx, who)` and `Splitter` keys dicts by `Result` (`DESIGN_warm_start` §1) | a concentrated meta batch raises the rate of equal-`fx` results | already ticketed; watch `result2leaf` size vs `len(results)` |
| **the A/B measures a stream shift, not the mechanism** | adding a module perturbs RNG streams (`core.py:481-492,1370`); §18 is what that costs — a dead parameter produced a tight CI | `Meta_never` **must** be bit-identical to `Meta_off`; if it is not, the module is in the wrong place in the list and every later number is uninterpretable |
| **event-handler / main-loop race** | `warm_start_now` from a bus handler clears a queue `execute()` is draining (`core.py:1074` vs `blocks.py:735`) | never call it from a handler — defer to `_open_block` (§2) |

## 6. Implementation plan, smallest first

| step | what | effort |
|---|---|---|
| **0** | `MetaAnalyst(Heuristic)` skeleton, trigger objects (`budget_fraction`, `stagnation`, `never`, `\|`, `&`), refractory + `meta_frac` cap, `warm_start_box` field on `Heuristic` (unused yet).  **No analysis.**  Ship `Meta_never` and prove it bit-identical to `Meta_off`. | ~0.5 d |
| **1** | (a) leaf volume-vs-count scan + emit *k* points into the least-sampled leaves, `who = "MetaAnalyst:void"`.  No model.  **This is the smallest complete thing** and it is also the one §3 expects to be a re-labelling — build it because `Meta_random_b25` needs something to be compared against. | ~1 d |
| **2** | (b) rank × count scan; reuse `Archive.per_leaf_best` plus the count term.  `who = "MetaAnalyst:shallow"`. | ~0.5 d |
| **3** | **The region hand-off** — `meta_region` event, `StrategyBlockBandit.on_meta_region`, `warm_start_box` honoured in `cma_es.py:551-564` and `lshade.py:929`, forced warm start at the next `_open_block`.  Tests: box reaches the arm; the hand-off never fires from a bus thread; the guard interaction with `warm_start_only_if_foreign` is explicit. | ~1.5 d |
| **4** | (d) arm-state trigger — per-arm "evaluations since this arm last improved the run best", three lines in `blocks.py:378-393`'s loop; read-only `diverged` / `stagnation_ratio` properties on `CMAES` and `LSHADE` for the sharper version. | ~1 d |
| **5** | (c) surrogate mode — GP on `Archive.results` (≤ 256, `n_restarts_optimizer ∈ {0,1}`), EI maximised **inside the chosen box only**.  **Only if steps 1–4 showed a sign at *d* = 5.** | ~2 d |

Tests, following `tests/test_strategy_rewarding_credit.py:33-50` and
`tests/test_reproducibility.py:19-49` (`sync_evaluation=True`, fixed seed,
`testing_mode=True`):

1. `trigger=never` ⇒ trajectory bit-identical to the spec without the module
2. `budget_fraction(f)` fires exactly once, at the first result batch
   crossing `f·max_eval`
3. the refractory period and `meta_frac` cap are both enforced (fire twice
   in a row is impossible; total meta evaluations ≤ `meta_frac·max_eval`)
4. `stagnation` is budget-relative: the same spec at 2× budget waits 2×
   as long
5. the region reaches the arm — a stub arm records the `box` it was handed,
   and `archive_seed(box=…)` returns only points inside it
6. the hand-off runs on the main thread: a handler that fires
   `meta_region` never mutates the arm's queue before `_open_block`
7. `MetaAnalyst` with an empty queue is never selected by
   `StrategyBlockBandit._select` and never opens a block
8. `who` prefixes attribute correctly through `Archive._who_of` and
   `blocks._credit_arm_best`
