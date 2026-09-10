# Invariant probing of `panobbgo/heuristics` — findings

Date: 2026-09-10 · Test file: `tests/test_invariants.py` (new, 320 tests, ~30 s under `-n 4`)
Method: seeded, synchronous `StrategyRoundRobin` runs (`sync_evaluation=True`,
`testing_mode=True`, `parse_args=False`, `seed=`), `max_stall_seconds=3`, DeJong / Rastrigin /
Rosenbrock at dim 3, 150–1500 evaluations. Every claim below is reproducible in seconds; the
repro snippets use this helper:

```python
# probe.py — the shape every repro below uses
import numpy as np
from panobbgo.lib.classic import DeJong
from panobbgo.strategies import StrategyRoundRobin

def run(factories, problem=None, seed=42, max_eval=150, stall=3.0):
    s = StrategyRoundRobin(problem or DeJong(dims=3), parse_args=False, testing_mode=True, seed=seed)
    s.config.max_eval = max_eval
    s.config.sync_evaluation = True
    s.config.stop_on_convergence = False
    s.config.ui_show = False
    s.config.evaluation_method = "threaded"
    s.config.max_stall_seconds = stall
    for f in factories:
        s.add_heuristic(f(s))
    try:
        s.start()
    except Exception:
        s._cleanup(); raise
    df = s.results.results
    if df is None or len(df) == 0:
        return np.zeros(0), ()
    return df["fx"].to_numpy(dtype=float).ravel(), tuple(str(w) for w in df["who"].to_numpy().ravel())
```

---

## F1 — HIGH — `StrategyRoundRobin` crashes with `ZeroDivisionError` when its last arm goes inactive

**Where:** `panobbgo/strategies/round_robin.py:41`
(`self.current = (self.current + 1) % len(hs)`), with `panobbgo/core.py:1354-1360`.

**Repro**

```python
from panobbgo.heuristics import Center
run([lambda s: Center(s)], max_eval=300)     # ZeroDivisionError: division by zero
```

Same for `Zero` and `Sobol` — every heuristic that finishes its work and unsubscribes.

**Cause.** `StrategyBase.heuristics` (`core.py:1363`) filters on `h.active`; a heuristic that has
emitted everything it will ever emit and is subscribed to nothing is dropped from that list. When
the list becomes empty, `% len(hs)` divides by zero. The exception escapes `_run()` and
`start()` (`core.py:1354-1360`) only catches `KeyboardInterrupt`, so **`_cleanup()` never runs**:
the event-bus dispatcher thread, the evaluator thread pool and the results store are all leaked.
The correct behaviour is to end the run — there is nothing left that can produce a point.

**Why it matters beyond the toy arms.** Solo specs in the harness run under `StrategyRoundRobin`
(`RoundRobin_CMAES` etc.). Any arm that raises `StopHeuristic`, or whose event subscriptions all
terminate, turns a finished run into a crashed one plus leaked threads.

**Pinned by:** `test_exhausted_heuristic_ends_the_run_cleanly[Center|Zero|Sobol]`
(`xfail(raises=ZeroDivisionError, strict=True)`).

---

## F2 — HIGH — `warm_start=` shifts the RNG stream of `NLSHADE_RSP` / `NLSHADE_LBC` even when the archive is empty

**Where:** `panobbgo/heuristics/lshade.py:939` (`cap = self._archive_cap()`), one line *before* the
bail-out at `lshade.py:941-942` (`if not pool: return False  # empty archive: the caller falls
back to the cold path`); the offending override is
`panobbgo/heuristics/nl_shade_rsp.py:231` (`return int(self._rng.integers(0, a_max + 1))`).

**Repro**

```python
from panobbgo.heuristics import NLSHADE_RSP as C
a, _ = run([lambda s: C(s)])
b, _ = run([lambda s: C(s, warm_start="archive")])
np.array_equal(a, b)                       # False   <-- contract says True
# turn off the one override that draws:
c, _ = run([lambda s: C(s, adaptive_archive=False)])
d, _ = run([lambda s: C(s, warm_start="archive", adaptive_archive=False)])
np.array_equal(c, d)                       # True
```

The whole initial population differs (all 6–7 points), not just a tail. Adding an `Archive`
analyzer changes nothing — it is empty at `on_start` either way.

**Cause.** `LSHADE.on_start` documents the contract explicitly (`lshade.py:977-981`): *"with
`warm_start=None` — or an archive that has nothing to give — this is the cold start, statement for
statement as before."* `_warm_start_population` honours that for `LSHADE`, `LSHADE_EpSin` and
`JSO`, whose `_archive_cap()` is pure. `NLSHADE_RSP` (and its subclass `NLSHADE_LBC`) override
`_archive_cap()` to *sample* the cap from `self._rng` when `adaptive_archive=True` — the default.
Because that call sits before the `if not pool` early return, merely *setting* `warm_start=`
consumes one draw from the heuristic's generator and desynchronises everything after it.

**Why it matters.** This is the `ipop_factor` failure of DISCOVERY §18 in a new place: any paired
"warm vs cold" A/B on these two arms compares two different RNG streams, so a no-op change has a
non-zero paired delta. DISCOVERY §25/§27 attribute portfolio gains to "both arms warm"; those
measurements need re-auditing for the NL-SHADE arms specifically.

**Gap in the existing suite.** `tests/test_warm_start.py::test_an_empty_archive_degrades_to_the_cold_path`
pins exactly this contract, but its run (`tests/test_warm_start.py:467-493`) carries only
`JSO`, `PSO` and `CMAES` — none of which has the `adaptive_archive` override. The two classes that
break the contract are the two the test does not cover.

**Suggested fix:** move `cap = self._archive_cap()` below the `if not pool: return False` (it is
only used by `_seed_archive` afterwards), or compute the pool size from `_archive_max()`, which does
not draw.

**Pinned by:** `test_warm_start_with_empty_archive_is_the_cold_path[NLSHADE_RSP|NLSHADE_LBC]`
(`xfail(strict=True)`).

---

## F3 — HIGH — the subprocess-bridge arms (`LBFGSB`, `COBYQA`) contribute *nothing* whenever they share a strategy with any other heuristic

**Where:** `panobbgo/heuristics/lbfgsb.py:403-437` (`on_start` spawns `_pump`, which polls the pipe
with a 0.1 s timeout and emits from its own thread); `panobbgo/heuristics/cobyqa.py` shares the
mechanism; interacts with `panobbgo/strategies/round_robin.py:39-46`.

**Repro**

```python
import collections
from panobbgo.heuristics import Random, LBFGSB
_, who = run([lambda s: Random(s), lambda s: LBFGSB(s)], max_eval=150)
collections.Counter(w.split(":")[0] for w in who)   # Counter({'Random': 150})  — LBFGSB: 0

_, who = run([lambda s: LBFGSB(s)], max_eval=300)   # alone it spends the whole budget
sum(1 for w in who if w.startswith("LBFGSB"))       # 300
```

Same for `COBYQA`, `LocalPenaltySearch` (0 next to Random), and — for a different reason — the
constraint arms (see F7).

**Cause.** The bridge emits asynchronously from a pump thread; its output queue is empty most of
the time. `StrategyRoundRobin.execute` polls one heuristic per attempt and moves on the moment it
gets zero points, so the competitor (which always has 20 queued) satisfies the loop before the pump
ever gets its 0.1 s. Alone, the retry loop's `time.sleep(1e-3)` gives the pump its chance, which is
why the solo measurement looks healthy. This is not just non-determinism (which `lbfgsb.py:410-414`
already acknowledges) — it is total starvation, so any portfolio result that includes an L-BFGS-B or
COBYQA arm has been measuring a portfolio *without* it.

**Pinned by:** `test_subprocess_bridge_contributes_next_to_a_competitor` (`xfail(strict=True)`).

---

## F4 — HIGH — the wall-clock stall guard truncates slow arms, so `sync_evaluation=True` is not machine-independent

**Where:** `panobbgo/core.py:1726-1742` (the `max_stall_seconds` progress guard) and
`panobbgo/core.py:1703-1707` (`eventbus.wait_idle(timeout=self._max_stall_seconds)`).

**Repro**

```python
from panobbgo.heuristics import Random, GaussianProcessHeuristic
fx, _ = run([lambda s: Random(s), lambda s: GaussianProcessHeuristic(s)], max_eval=300, stall=1.5)
len(fx)     # 70 … 240 depending on the run and machine load — never 300
```

Log line: `No progress (no new points, no pending tasks, no new results) for 1.6s (2 consecutive
loops) … Results so far: 240/300. Stopping optimization.`

**Cause.** The event bus dispatches handlers serially. `GaussianProcessHeuristic.on_new_results`
takes ~0.4 s per evaluation's worth of work, so `wait_idle` times out and the *next* main-loop pass
finds every queue empty — including `Random`'s, whose refill handler has not run yet. Two such
passes and the guard declares a stall and stops the run. Consequences:

1. A slow arm silently loses part of its budget (an AOCC computed on 240 of 300 evaluations is not
   comparable to one on 300).
2. **The number of evaluations a seeded run performs becomes a function of machine speed and load.**
   `tests/test_reproducibility.py` only ever compares two runs *on the same machine, back to back*,
   so it cannot see this; the same seed on a busier machine gives a different trajectory length.
3. A slow *handler* starves fast heuristics that had nothing to do with it.

`COBYQA` shows the same shape more sharply: solo, at `max_stall_seconds=1.5`, a 300-evaluation run
produces **zero** evaluations.

**Pinned by:** `test_slow_heuristic_still_spends_its_budget` (`xfail(strict=True)`, COBYQA — GP is
too slow to pin in a seconds-long test; it is documented in the module docstring instead).

---

## F5 — MEDIUM — `PSO(stagnation_threshold=…)` is silently inert under the default topology

**Where:** `panobbgo/heuristics/pso.py:549`
(`if self.topology != "random" or self.stagnation_threshold is None: return`), constructor
validation at `pso.py:348-354`, `self.stagnation_threshold` assigned at `pso.py:367`.

**Repro**

```python
from panobbgo.heuristics import PSO
a, _ = run([lambda s: PSO(s)], max_eval=1500)
b, _ = run([lambda s: PSO(s, stagnation_threshold=4)], max_eval=1500)
np.array_equal(a, b)   # True — bit-identical
```

**Cause.** The stochastic-K stagnation rebuild is only reachable with `topology="random"`; the
default is `"gbest"`. The constructor nevertheless type-checks and range-checks the argument and
stores it, giving every caller the impression it took effect. The class docstring mentions the
coupling in the *random-topology* section (`pso.py:88-96`), but the `stagnation_threshold` argument
entry (`pso.py:272-276`) does not. This is the `ipop_factor` shape exactly: a sweep over
`stagnation_threshold` on a default-topology PSO would measure only RNG noise (or, with the
seed-name fix, exactly zero).

**Suggested fix:** raise (or at minimum warn) in `__init__` when `stagnation_threshold` is set and
`topology != "random"`. Same argument applies to `k_neighbors` under `"gbest"`, which *is*
documented as ignored (`pso.py:262-264`).

**Pinned by:** allowlist entry in `DEAD_PARAM_ALLOWLIST` with the mechanism spelled out (so the
detector will fire again if the gate ever moves).

---

## F6 — MEDIUM — six heuristics lose a paired comparison against uniform `Random`

**Setup.** 3 problems (DeJong, Rastrigin, Rosenbrock, all dim 3) × 2 seeds (42, 7), 300 evaluations,
best `fx` compared per cell, majority (≥ 4 / 6) required.

| arm | verdict | note |
|---|---|---|
| `DifferentialEvolution` | loses | plain DE/rand/1/bin, default `NP=20`; ~15 generations in 300 evals |
| `NelderMead` | loses | |
| `WeightedAverage` | loses | cannot leave the convex hull of what it is given |
| `RegionUCB` | loses | spends the budget exploring boxes |
| `Extremal` | loses | box-corner sampler; not an optimizer |
| `LatinHypercube` | loses | space-filling design; not an optimizer |
| `LSHADE`, `LSHADE_EpSin`, `JSO`, `NLSHADE_RSP`, `NLSHADE_LBC`, `CMAES`, `PSO`, `Nearby`, `ClaudeHeuristic`, `QuadraticWlsModel`, `LBFGSB` | pass | |

The last two rows of the "loses" block are expected by design; the first four are the finding.
`DifferentialEvolution` in particular is the *only* population arm that fails, and it is the one
whose `NP` did not get the `NP_init="auto"` budget-adaptive treatment of DISCOVERY §17 — consistent
with the population law found there (`NP ≈ 3–4·dim`, i.e. ~10 at dim 3, not 20).

**Repro**

```python
from panobbgo.heuristics import Random, DifferentialEvolution
a, _ = run([lambda s: Random(s)], max_eval=300)
b, _ = run([lambda s: DifferentialEvolution(s)], max_eval=300)
min(a), min(b)     # Random wins on all six cells
```

**Pinned by:** `test_beats_random[...]` with `xfail(strict=True)` and a per-arm reason.

---

## F7 — LOW/INFO — four arms are inert on unconstrained problems, and still occupy a scheduler slot

`FeasibleSearch`, `ConstraintGradient`, `ConstraintRepair` and `LocalPenaltySearch` emit **zero**
points next to `Random` on DeJong dim 3 (150 evals). On `RosenbrockConstraint` dim 3 they emit
0 / 3 / 1 / 0 points respectively out of 150. This is by design for the first three, but it means a
portfolio that carries them on an unconstrained battery pays a scheduler slot for an arm that cannot
produce a point — and, under `StrategyRoundRobin`, a wasted `execute()` attempt each round.

**Pinned by:** `test_constraint_heuristic_is_inert_when_unconstrained[...]` (passing — it documents
the contract so a change is visible).

---

## F8 — LOW — knobs that are live code but quantise away at the probe's problem size

These are *not* dead parameters, but they are no-ops in the configuration in which they ship, so a
sweep over them measures nothing. Each is allowlisted with the mechanism.

| knob | mechanism | file:line |
|---|---|---|
| `LSHADE.p_best_end` | `p_count = max(ceil(p_eff·NP), 1)`; at the budget-adaptive `NP` of a dim-3 run (`NP_init` resolves to 6) both `p_best=0.11` and `p_best_end=0.05` give `p_count = 1` | `lshade.py:773` |
| `ClaudeHeuristic.max_clusters` | `k = min(max_clusters, max(1, n_elite // (2·dim)))`; the second term binds for every elite set the probe produces | `claude_heuristic.py:153` |
| `ClaudeHeuristic.min_points` | activation threshold `2·dim+1 = 7`; results arrive in batches of 20, so lowering it to 5 does not move the activation batch | `claude_heuristic.py:78-92` |
| `LBFGSB.maxfun` | per-descent evaluation cap; a 3-d descent never reaches 25 | `lbfgsb.py:318, 367` |
| `Nearby.cap` | `cap` only sizes the output queue, and `Heuristic._put` grows it on demand since the §5 fix; `Nearby` emits from `on_new_best`, never via `fill_queue`, so `cap` has **no** behavioural effect at all | `core.py:681-692`, `nearby.py:567` |

`Nearby.cap` is the strongest of these: since `ensure_output_capacity` grows the queue on every
`_put`, `cap` is now meaningless for every heuristic that does not use `fill_queue`. That is a
side-effect of the (correct) §5 fix and worth a docstring update — `Heuristic.cap` is documented as
"the fill level reactive heuristics top up to", which is only true for `fill_queue` users.

---

## F9 — INFO — CMA-ES: the whole restart/termination machinery is unreachable in a short run

Not a defect, but the measurement hazard that produced the retracted `ipop_factor` result. On
DeJong dim 3 at 300 evaluations, **16 of CMA-ES's 20 constructor arguments** leave the trajectory
bit-identical. Raising the probe to Rastrigin dim 3 / 1500 evaluations revives six of them
(`self_restart`, `restart_from`, `min_results_fraction`, `tolfun`, `stagnation`, `stagnation_frac`);
ten stay inert even there:

- reachable only via the `Restart` analyzer's `restart` event, which no solo spec publishes:
  `ipop_factor`, `restart_mode`;
- Hansen termination criteria that never trip on these problems: `tolx`, `tolfunhist`,
  `conditioncov`, `noeffectaxis`, `noeffectcoord`;
- gated by a sibling at its default: `stagnation_rel_tol` (needs `stagnation_frac`);
- **`sigma_divergence`, `sigma_max_frac`, `sigma_divergence_gens`** — the σ-divergence restart only
  fires once σ exceeds `sigma_max_frac` of the box range, and on a 3-d Rastrigin σ only ever
  shrinks. DISCOVERY §23 records this criterion as *accepted on the 12-seed roster*; given §18's
  lesson, that acceptance deserves the same audit as `ipop_factor` — specifically, a direct check
  that the criterion fires at all on the battery it was accepted on.

**Practical consequence for sweeps:** a per-arm sweep must state the budget and problem class at
which each knob becomes reachable, or it is sweeping over a constant.

---

## Invariants that came out **clean** (now regression-guarded)

- **Output-queue contract (§5).** Every population heuristic (`DifferentialEvolution`, `LSHADE`,
  `LSHADE_EpSin`, `JSO`, `NLSHADE_RSP`, `NLSHADE_LBC`, `PSO`, `CMAES`, `LatinHypercube`, `Sobol`)
  keeps its full generation with `config.capacity = 5` and a generation of 30. The §5 fix holds.
- **Constructor RNG discipline.** Every one of the 28 heuristics advances `strategy.rng` by exactly
  one `spawn_rng()` draw during construction — no constructor reaches for the master generator
  again, so module streams remain a pure function of construction order.
- **Event-handler hygiene.** All 91 `on_*` handlers across the `Module` / strategy /
  constraint-handler classes bind every payload their publisher sends, and no handler listens to an
  event nobody publishes. Checked statically by parsing every literal `publish("key", …)` in the
  package with `ast` and binding the payload against each handler's signature.
  *Related latent hazard, not currently triggered:* `EventBus._dispatch` (`core.py:1098-1101`)
  swallows `TypeError` outright for `terminate` events (`start`, `finished`), so a future signature
  mismatch on `on_start` would make the handler silently never run; and for non-terminate events the
  `TypeError` is caught by the blanket `except Exception` at `core.py:1110` and only logged. The new
  static test is what keeps this from happening.
- **Finiteness / box containment.** No heuristic emits NaN, inf or an out-of-box point at 300
  evaluations. (`Heuristic.emit` projects, so this also confirms nothing bypasses `emit`.)
- **`warm_start=None` ≡ omitted** for every archive-mode class (`LSHADE`, `LSHADE_EpSin`, `JSO`,
  `NLSHADE_RSP`, `NLSHADE_LBC`, `PSO`, `CMAES`).

---

## Test file summary

`tests/test_invariants.py` — 320 tests, **273 passed / 34 skipped / 13 xfailed** in ~30 s with
`-n 4` (three consecutive runs: 29.5 s, 42.0 s, 49.9 s — stable, no flakes).

| test | parametrisation | what it pins |
|---|---|---|
| `test_constructor_kwarg_is_read` | 113 (heuristic, kwarg) pairs from `inspect.signature` | every constructor argument changes the run, or is allowlisted with its mechanism |
| `test_warm_start_none_equals_omitted` | 8 classes | the reproducibility contract |
| `test_warm_start_with_empty_archive_is_the_cold_path` | 8 classes | F2 |
| `test_spends_its_budget` | 18 heuristics | no early stop / stall |
| `test_points_are_finite_and_in_box` | 18 heuristics | no NaN/inf, projection holds |
| `test_beats_random` | 17 heuristics | F6 |
| `test_generation_survives_a_small_queue` | 10 population heuristics | §5 |
| `test_construction_advances_master_rng_exactly_once` | 28 heuristics | module stream order |
| `test_event_handler_signature_matches_publisher` | 91 handlers | static payload/signature match |
| `test_exhausted_heuristic_ends_the_run_cleanly` | 3 | F1 |
| `test_subprocess_bridge_contributes_next_to_a_competitor` | 1 | F3 |
| `test_slow_heuristic_still_spends_its_budget` | 1 | F4 |
| `test_constraint_heuristic_is_inert_when_unconstrained` | 4 | F7 |

Nothing was fixed; every finding is either an `xfail(strict=True)` with the reason in the marker, or
a `skip` carrying the mechanism, so the suite stays green while the fact stays pinned.

## Gates

- `uv run ruff format tests/test_invariants.py` — formatted; `uv run ruff check` — clean.
- `uv run pytest -q -n 4 tests/test_invariants.py` — **273 passed, 34 skipped, 13 xfailed** in
  29.5 s (repeat runs 42.0 s / 32.6 s — no flakes, no `XPASS`).
- `uv run pytest -q -n 4 tests/` (full suite) — **2546 passed, 35 skipped, 13 xfailed** in 133 s,
  exit 0. The new file adds no failures anywhere else.
- Only `tests/test_invariants.py` was created; no existing file was touched.
