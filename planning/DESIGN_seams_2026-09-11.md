# Seams, not shards: where the shared archive can enter a solver's loop

2026-09-11.  Harald's question: is the real strength of panobbgo that
CMA-ES (and the rest) can be *decomposed* into building blocks, rebuilt
inside the framework, and generalised — so that flexibility, not any
one reference implementation, wins on a broad battery?

**Short answer.**  Yes — but the units are *seams*, not *shards*.  A
reference CMA-ES is strong because its parts are tuned to each other
(c₁, c_μ, d_σ, the two evolution paths); decoupling distribution,
selection and update and re-mixing them freely will most likely build a
worse CMA-ES.  Every documented gain in the literature keeps the loop
intact and lets external information in at one **defined seam**
(injection, surrogate pre-screening, model-based ranking, restart
placement).  Those seams are panobbgo's building blocks: each is a
switchable kwarg on an existing arm, and each is measured by the rule.

The observation that makes this concrete (`cma_es.py:718`):

```python
for r in results:
    if not r.who.startswith("CMAES:"):
        continue
```

CMA-ES ignores every point it did not emit.  jSO likewise draws *pbest*
and its archive A only from its own population.  All the sharing
measured so far (§27, §42, §44) happens **at block boundaries** —
`warm_start="archive"` re-fits m/σ/C from the shared archive when an arm
re-acquires the block, ~8 hand-offs in a 200·dim run.  Inside a
generation the loops are closed, exactly as in the reference code.

## 1. The seam catalogue

| seam | block | what the shared archive contributes | status |
|---|---|---|---|
| **update input** | injection (Hansen 2011, *Injecting external solutions into CMA-ES*, arXiv:1110.4181) | foreign evaluated points inside the current support are ranked with the offspring; Mahalanobis length clipped so σ-adaptation is not thrown | **step 1** |
| **pbest pool** (DE side) | shared pbest | jSO's `current-to-pbest-w/1` draws pbest from live-top-p ∪ shared-archive-top-p | **step 1** |
| **pre-evaluation** | surrogate pre-screening (lq-CMA-ES, Hansen 2019) | sample λ′ > λ, rank by a local quadratic fitted to the archive (`quadratic_wls.py` exists), evaluate the best λ | step 2 |
| **selection** | model-blended ranking | under noise, rank by observed value blended with a local model — noise handling without re-evaluation | step 2 |
| **restart / warm start** | `warm_start`, `archive_cov` | m, σ, C from the archive at re-acquisition | done (DESIGN_warm_start) |
| **termination** | basin-aware restart | "another arm already searched this basin" → restart elsewhere; the Splitter / meta line | later |

Step 1 is the pair that turns sharing from "at hand-offs" into "every
generation", on both arms of the portfolio that already wins at
200·dim.  If that pair does not beat block-boundary sharing by the rule,
the within-generation seams are not the lever and steps 2+ do not
inherit the thesis's credit — that is the point of doing it first.

## 2. Step 1, specified

### 2.1 `CMAES(inject=False)` — Hansen-style injection

In `on_new_results`, a result whose `who` does **not** start with this
instance's own prefix (`"CMAES:"` today — make it the instance's own
tag so two CMA-ES arms stay distinct) is a *foreign* point.  With
`inject=True` and an open generation:

1. Take the **oldest open generation** g (the one `_update` will fire
   on next).  Keep a per-generation injected list, capped at
   `inject_max = max(1, λ // 4)` points; beyond the cap, keep the best
   `inject_max` by penalty (replace the worst injected one if the new
   point is better).
2. For a foreign point x_f: `y = (x_f − m) / σ`; Mahalanobis norm
   `‖C^{-1/2} y‖ = ‖diag(1/D) Bᵀ y‖`.  Clip: if it exceeds
   `c_y = √n + 2n/(n+2)` (Hansen's default), rescale y so the norm
   equals `c_y`.  Store `{"penalty", "x": m + σ·y, "y": y}` — the
   *clipped* position, so the mean update cannot be dragged further than
   one clipped step (that is the whole point of the clipping).
3. `_update` receives `bucket + injected` — injected points are ranked
   with the offspring by penalty and may be selected.  They do **not**
   count toward the quorum `min_needed` (emitted-based, unchanged), and
   μ and the weights stay those of λ.  `_record_generation` sees the
   merged list (it is the trace for termination criteria); document
   that stagnation/tolfunhist then include injected values.
4. The step-size path uses `y_w` as before; the clipping is the
   protection.  Hansen's additional σ-correction for the case "the
   injected solution is the *mean shift*" is not needed here — we never
   inject into the mean, only into the ranking.

Invariants to test (`tests/test_cma_es_inject.py`):

* `inject=True` **with no foreign points is byte-identical** to
  `inject=False` — same trajectory (the pattern of
  `tests/test_cross_process_reproducibility.py` / the regime-gate
  identity tests).  This is the guard that the kwarg is inert alone.
* A foreign point far outside the support is clipped to norm `c_y`
  (unit test on the helper).
* With a foreign point that is better than every offspring, it is
  selected (rank 0) in the next update; the mean moves toward it.
* Cap: `λ//4 + 3` foreign points in one generation → exactly
  `inject_max` are injected, the best ones.
* `inject=True` is a dead parameter for `CMAES_alone` and must be
  registered as such in `tests/test_invariants.py`'s dead-parameter
  detector (it legitimately moves nothing without a second arm).

### 2.2 `JSO(shared_pbest=False)`

In `_mutate` (`jso.py:291–345`), with `shared_pbest=True`: the pbest
pool becomes `sorted_live[:p_count]` ∪ the best `p_count` **foreign**
results from the shared `Archive` analyzer (`self.archive_seed(k, mode="archive")`
or the analyzer's `top_k`, filtered by `who` not starting with this
instance's tag; use whatever the base class already provides — see
`core.py:737` `archive_seed` and `analyzers/archive.py` `top_k`), and
pbest is drawn uniformly from the union with the instance RNG.  If the
archive analyzer is absent or has no foreign points, the pool is the
live one and the draw must consume the same RNG calls as today —
**`shared_pbest=True` alone must be byte-identical to `False`**.  Draw
count is the invariant to protect: do not add an RNG call in the empty
case.

Tests (`tests/test_jso_shared_pbest.py`): identity when alone; with a
planted better foreign archive point, it appears as pbest with the
expected frequency; no extra RNG draw in the empty case.

### 2.3 Harness

`benchmarks/portfolio_screen.py` spec table, all on the shared
`"screen"` stream:

| spec | CMA-ES arm | jSO arm |
|---|---|---|
| `Blocks_uniform_cj_warm2` | warm | warm |
| `Blocks_cj_inject` | warm + `inject=True` | warm |
| `Blocks_cj_pbest` | warm | warm + `shared_pbest=True` |
| `Blocks_cj_seams` | warm + inject | warm + shared_pbest |

`CMAES_alone` and `JSO_alone` stay as bars.

## 3. Measurement and falsifier

12-seed roster, `kind=standard bm=200` (the regime where sharing wins,
§44.1) and `kind=noisy-unif` at 500·dim (the other rule win, §42).
Paired deltas of the three new specs vs `Blocks_uniform_cj_warm2` and
vs `CMAES_alone`; the rule as everywhere (CI clear on the positive
side, ≥ 9/12, no dimension negative).  ~6 cells × 8 min.

* **Accept** a seam if it beats `Blocks_uniform_cj_warm2` by the rule
  in one regime and is not negative by the rule in the other.
* **Falsifier.**  Neither `inject` nor `shared_pbest` nor both beats
  block-boundary sharing by the rule at 200·dim → within-generation
  sharing adds nothing that the hand-off did not already collect; the
  seams of §1 step 2 do not inherit the thesis's credit and need their
  own argument.  Record it; the portfolio keeps its §45 default.
* **Diagnostic.**  If `Blocks_cj_inject` is *negative* by the rule
  while `CMAES_alone` identity holds, the clipping is not protecting σ
  — check `sigma_divergence` restart counts in the logs before
  concluding anything about the seam.

## 4. Risks

1. Injected points during a *foreign* block: while jSO holds the block,
   CMA-ES's open generation collects jSO's points; the cap `λ//4` and
   the clipping bound the damage, but the generation that fires on
   re-acquisition is then half foreign.  This interacts with
   `warm_start` at the same moment (which re-fits m/σ/C and *zeroes the
   paths* — check the order: warm start first, then the open generation
   is stale and should be dropped, as `warm_start_now` presumably does
   today; if it does not, the injected list must be cleared with it).
2. Termination criteria (`_check_termination`) read `_record_generation`
   — injected values can trigger `tolfunhist` earlier.  Measured, not
   guessed: the restart counts are in the log.
3. Two CMA-ES arms (§25-style three-arm specs) share the `"CMAES:"`
   prefix today; the own-tag change in §2.1 fixes that as a side
   effect.
