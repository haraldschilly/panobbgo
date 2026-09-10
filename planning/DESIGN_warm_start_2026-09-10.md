# Design: warm-starting arms from the shared archive (2026-09-10)

Read-only analysis by an Opus subagent; the three load-bearing claims
(§0) were verified by the orchestrator against the code and by a direct
probe.  Status: **findings accepted; implementation queued behind the
harness fixes and the block scheduler.**

## 0. Verdict on the thesis

"Population heuristics ignore points they did not request" holds
**exactly, and only, for the three arms that matter**: `CMAES`
(`cma_es.py:300` — `if not r.who.startswith("CMAES:"): continue`), the
whole L-SHADE lineage (`lshade.py:868-873`, one `on_start` covers five
arms via `super()`), and `PSO` (`pso.py:701-707`).  All three consult
the shared store only as a scalar budget counter.  Roughly half the
heuristic zoo already consumes foreign points — GP, Nearby, Feasible-
Search, ConstraintGradient, QuadraticWls, NelderMead, WeightedAverage,
RegionUCB, Random — and `LBFGSB` already ships a `warm_start=True` that
restarts from the global incumbent (`lbfgsb.py:441-473`).  Sharing
exists where it buys least and is absent where the performance lives.

Three findings that change the shape of the problem:

* **`on_restart` is dead in every shipped and benchmarked
  configuration.** Only the `Restart` analyzer publishes it
  (`restart.py:149`); `StrategyBase.initialize()` adds four analyzers
  and not that one (`core.py:1234-1236`); `make_ioh_strategies` omits it
  on purpose (it halves CMA-ES).  Therefore **CMA-ES's `ipop_factor`
  is never read in a solo run** — verified: 1.5 / 2.0 / 3.0 give
  bit-identical AOCC.  Every `ipop_factor` number in DISCOVERY §15–§17
  is retracted.  And **solo CMA-ES has no termination or restart
  criterion at all** (see the self-restart work).
* **`StrategyPhased` never warm-started anything.** It constructs every
  phase's heuristics up front and publishes `start` to all of them at
  *t* = 0 (`phased.py:148-158`, `core.py:1253`); the phase-2 arm's first
  generation sits undrained in its queue for all of phase 1; the
  boundary resets only bandit bookkeeping (`phased.py:192-228`).  §12
  measured a *delayed cold start from a stale distribution*, not a warm
  start.
* **The harness sets `max_eval` after constructing the strategy**
  (`harness_ioh.py:869-870`), so `NP_init="auto"` always resolved against
  the default 1000 — 36 at *d* = 2, **83** at *d* = 5 and at *d* = 10
  (not 90 / 180).  Fixed separately.

Also: the per-run seed hashes the spec **name** (`harness_ioh.py:790`),
so every variant-vs-default pair ran on different RNG streams.  That is
how a dead parameter produced a tight CI.  Fixed separately
(`StrategySpec.seed_name`).

## 1. What the shared store exposes

`Results` (`core.py:76-455`): last-*n* only (`get_history`), a DataFrame
that re-concatenates on every access, floats not `Result` objects.  No
top-k, no box query, no threshold, no diversity.  `Best` collapses to
one point on unconstrained problems (`best.py:135`).  **`Splitter` is
the useful surprise**: `Box.results` is a live `list[Result]` of every
point in the box regardless of `who` (`splitter.py:346-354`), and
`add_result` recurses from the root (`:389-392`), so
`splitter.root.results` is a complete, ordered archive in `Result`
form.  Top-k in a box is available today as
`sorted(splitter.get_box(x).results, key=penalty)[:k]`.

Latent hazard: `Result.__eq__` compares only `fx` while `__hash__` uses
`(x, fx, who)` (`lib/lib.py:218-231`); `Splitter` keys dicts by `Result`.
Warm-starting raises the rate of equal-`fx` results.  Ticket it.

## 2. Design

**Query layer**: opt-in `Archive` analyzer (`analyzers/archive.py`,
~80 lines): bounded top-K heap over the result stream, no `who` filter;
`top_k(k, box=, exclude_who=, fx_max=)`, `diverse_k(k, pool=4)` (greedy
max-min over the top `pool·k`), `per_leaf_best(k)` (best of each of the
k best Splitter leaves — the only selector that hands a second arm a
*different basin*, which is what the oracle gap is made of).  Zero-code
fallback for the first experiment: sort `splitter.root.results`.

**Heuristic interface**, after the `LBFGSB` precedent:
`archive_seed(k, mode=, box=) -> list[Result]` (never raises; `[]` means
"use the cold path") and `warm_start_now() -> bool` (default `False`).
Each of CMAES / LSHADE / PSO gains `warm_start: Optional[str] = None`;
`on_start` becomes `seeds = archive_seed(n) if warm_start else []` →
`if not seeds: <existing cold code, untouched>`.  The block scheduler
calls it **by direct method call** (the bus only broadcasts), after
`clear_output()` — dropping the stale *t* = 0 generation is half the
fix on its own.

**Per-arm recipes.**
* CMA-ES: mean = μ-weighted recombination of the top-μ (reusing the
  weights at `cma_es.py:211-212`); σ = `clip(mean std of top-k, 1e-6,
  σ0)` — never wider than cold; `C = I` in v1, `cov(top-k)` normalised
  to unit determinant as a separate `"archive_cov"` v2; paths stay
  zero.  Saves **no** evaluations; the whole gain is "right basin".
* DE family: seed `_population` **directly with `Result` objects —
  zero evaluations** (slot type is already `Optional[Result]`,
  `lshade.py:443`; `_fx_of` memoises on `id`); shortfall via the cold
  path; then `_wake_idle_slots()`.  Seed the external `_archive` with
  the *next* `_archive_cap()` good points from **other arms** — the
  cheapest literal expression of the shared-archive thesis.  Today
  `on_restart` re-emits `NP_init` random points (`:949-955`) — pure
  waste.
* PSO: positions / pbest from the top-NP `Result`s, velocities
  `0.5·(x_π(i) − x_i)` over a random derangement, clipped to `v_max`.

**Reproducibility contract**: `warm_start=None` runs the existing
statements in the existing order; **no RNG draw in `__init__`** (streams
are handed out in construction order, `core.py:1222,1279-1286`); do not
add `Archive` to the default analyzer list — ship via
`StrategySpec.analyzers`; one regression test pinning `(x, fx, who)`.

## 3. Risks → detection

| risk | detection |
|---|---|
| all arms seeded at the same incumbent → premature convergence | log `init_spread`; **crossing traces** warm-better-early / worse-late in `trace_fx` (AOCC alone hides it) |
| DE loses its differential (clustered population + archive) | mean pairwise distance at gen 1 and 5; seed population from `top_k`, archive from `diverse_k` |
| re-emitting seeds for evaluation (what `on_restart` does today) | `n_evals` vs unique `x` rows; seed-directly avoids it by construction |
| Splitter live-lock on duplicate coordinates (§13) | `n_evals < budget`; `splitter.max_depth` vs `MAX_DEPTH=60` |
| stale-basin: top-k *is* the previous arm's exhausted basin | compare `"archive"` vs `"archive_leaf"`; if only the leaf variant pays, the value is diversity, not the archive |
| unmeasurable at 3 seeds | same `seed_name` for both variants; 12-seed roster |

## 4. Experiment

Primary pair, on tuned arms, standard battery, `sync_eval`:
`Phased(60 % CMA-ES → NLSHADE_LBC cold)` vs `(… → LBC warm_start="archive")`,
same `seed_name`, seeds 42/7/1234 for sign then the 12-seed roster.
Re-measure the cold arm — §12's used untuned LBC.  Expected: budget
saved ≈ `NP_init / phase-2 budget` ≈ +0.005; the basin effect is the
hypothesis, **+0.02…+0.06**.  The decision question is sharper than the
delta: does the warm phased spec clear CMA-ES alone?  If not, the
shared archive does not rescue the portfolio for this pair.  Do **not**
headline the Sobol' 10 % → CMA-ES pair: AOCC integrates from evaluation
1, so its −0.38 is mostly metric geometry.

Cheapest sanity check first (minutes): instrument one cold phased run
and log, at the boundary, how many points sit in the phase-2 arm's
queue and the coordinates of its first evaluated point — §0 predicts a
full `NP_init` batch of *t* = 0 uniform-random points.
