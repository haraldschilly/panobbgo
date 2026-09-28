# Discovery session 2026-09-09 — is the default setup robust and effective?

Orientation + measurement pass over master `fac26b8` (last nightly commit
2026-08-13). Everything below was measured on this machine today with
`--sync-eval`, seed 42, unless stated otherwise. Scripts live in the
session scratchpad; the numbers are reproducible from the commands shown.

## 1. Where the flagship stands (absolute, standard IOH battery)

`uv run python scripts/ioh_benchmark.py run --standard --baselines --sync-eval --seed 42`

| strategy | d=2 | d=5 |
|---|---|---|
| Baseline_SciPyDE | **0.5065** | **0.3437** |
| Rewarding_Restart (flagship) | 0.4555 | 0.3011 |
| Baseline_SciPyAnneal | 0.4388 | 0.2682 |
| RoundRobin_Random | 0.3612 | 0.2796 |
| Baseline_Random | 0.3394 | 0.2467 |

* GOAL §1 priority 2 ("beat Baseline_SciPyDE") is **not met** at either
  dimension; the gap (~0.04–0.05) is 4–5× the nightly's `eps_accept`.
* The flagship is ~0.02–0.09 above pure random. On the quick battery
  (d2 × 200 evals) it is 0.3502 vs 0.3294 for random.

## 2. The seeded run is not reproducible

Same spec (`Rewarding_Restart`), same battery (standard), same
`base_seed=42`, `sync_eval=True`, two fresh processes:

| run | d=2 mean | d=5 mean | d=2 per-instance |
|---|---|---|---|
| A | 0.4236 | 0.3112 | 0.474 0.437 0.418 0.413 0.377 |
| B | 0.4318 | 0.3036 | 0.470 0.428 0.325 0.466 0.469 |
| earlier today (5-spec run) | 0.4555 | 0.3011 | |
| earlier today (2-spec run) | 0.4034 | 0.2844 | |
| C (niced, machine under load) | 0.4897 | 0.3083 | 0.525 0.434 0.397 0.577 0.515 |

With the evaluator thread pool reduced to **one** worker (quick battery,
d2): instance AOCCs A = 0.3898 / 0.3869 / 0.2394 vs B = 0.3887 / 0.3801 /
0.2207 — still not identical, so the thread pool is not the only source.

Battery-mean spread across identical invocations ≈ 0.05 at d2; a single
instance moved 0.418 → 0.325. The nightly accept threshold is 0.0125.
**This is the measurement-fidelity ceiling the loop has been fighting for
34 nights.** Verified source: `LSHADE`/`JSO`/`NLSHADE_*`/`PSO`/`LBFGSB` create
`np.random.default_rng(seed)` with `seed=None` by default
(`lshade.py:425`, `pso.py:349`, `lbfgsb.py:273`), so the harness's
`np.random.seed(seed)` never reaches them. Further suspected sources: one
daemon thread per `(module, event)` in `EventBus` makes result-handler
ordering scheduler-dependent; heuristics draw from the global
`np.random` state interleaved across threads; the evaluator uses
`cpu_count()` = 16 worker threads even under `sync_evaluation`.
Making a run bit-reproducible for a given seed would let paired A/Bs
detect effects ~10× smaller than today.

## 3. Default configuration silently cripples an end-user run

Fresh process, `StrategyRewarding(Rosenbrock(dim), max_eval=N)` with the
flagship heuristic set, **no** harness:

| dim | max_eval | evaluations actually used | best f |
|---|---|---|---|
| 2 | 1000 | 131 | 0.12 |
| 5 | 2500 | 105 | 64.4 |

Cause: `StrategyBase.initialize()` force-injects the `Convergence`
analyzer (window 50, std-threshold 1e-6 on best-so-far) and
`core.stop_on_convergence` defaults to `True`, so any 50-evaluation
plateau terminates the run. The benchmark harness sets
`stop_on_convergence = False` by hand (`harness_ioh.py:~848`), so **the
benchmarks never see the behaviour a library user gets.**

## 4. `Config` is a process-wide singleton

`panobbgo/config.py:46-56` (`Config.__new__` returns `_instance`).
Consequences observed today:

* `strategy.config.X = ...` on one strategy changes X for every strategy
  created later in the same process. `StrategySpec.config_overrides`
  therefore leak from one spec to the next inside a single harness run.
* `heuristic.capacity` / `stop_on_convergence` set in a probe leaked into
  the following probe (see §5 numbers: "cap=None" case reported cap=400).

## 5. Population heuristics are silently truncated to the output queue

`Heuristic.emit` / `LSHADE._emit_trial` do `put_nowait` on a
`Queue(config.capacity)` (default **20**) and swallow `Full`. With
`NP_init="auto"` (`18·d`, min 6, max 400):

| dim | budget | NP_init | initial population points dropped |
|---|---|---|---|
| 2 | 200 | 17 | 0 |
| 2 | 1000 | 36 | 16 |
| 5 | 2500 | 90 | **70** |
| 10 | 5000 | 180 | **161** |

Only `CMAES` calls `ensure_output_capacity`. The DE family (JSO,
NLSHADE_*, LSHADE_*) and PSO run with a ~20-member population regardless
of what `NP_init` says, so the last three months of NP tuning in the
loop tuned a parameter that mostly does not take effect.
A/B on the standard battery with `capacity=400` was flat
(Δ = −0.0027, sd 0.035, n=10 paired) — i.e. the DE arms are not
carrying the flagship either way.

## 6. `StrategyRewarding` degenerates to (biased) round-robin

Performance weights over a d5/2500 run (`discount = 0.95` per emitted
point, reward ≤ 1 on new best, smoothing 0.5):

| evals | Random | Nearby | Center | NelderMead | JSO |
|---|---|---|---|---|---|
| 0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| 311 | 0.002 | 2.40 | 1.95 | 0.85 | 0.05 |
| 1244 | 0.0 | 2.53 | 1.95 | 0.06 | 0.01 |
| 2475 | 0.0 | 1.00 | 1.95 | 0.001 | 0.11 |

* `Center` emits exactly one point, gets a reward once, is never
  discounted again, and holds the **highest selection probability
  (44 %) for the entire run** while contributing nothing.
* Any heuristic that emits in batches (DE: 90 points ⇒ 0.95^90 ≈ 0.01)
  is discounted to ~0 immediately; the bandit then runs on the additive
  smoothing term, i.e. near-uniform. This is consistent with
  `Rewarding_*` ≈ `RoundRobin_*` in every battery.
* Credit goes to the emitter of the *best point*, not to improvement
  per evaluation spent — the standard adaptive-operator-selection
  signal (Thompson / UCB / SoftMax on *normalised improvement per pull*)
  is what `StrategyUCB` / `StrategyThompsonSampling` exist for, but the
  flagship does not use them and no A/B in the log compares them.

## 7. Other robustness findings (from code reading, verified)

* `EventBus.publish` (`core.py:~966`) creates one `Event` and hands the
  **same mutable object** to every subscriber; `event.terminate` is
  reassigned in the loop.
* `Heuristic.emit` swallows every exception at DEBUG level, including
  `Full` and non-ndarray input.
* `Heuristic.active` drops a heuristic from `strategy.heuristics` once
  its dispatcher threads are gone and its queue is empty — reactive
  heuristics vanish from the list mid-run (seen in the share probe: 6
  added, 3–4 listed at the end).
* Thread-per-`(module,event)` eventbus: ~25 daemon threads per
  strategy; `eventbus.shutdown()` exists only because tests leaked
  thousands of them.
* Nightly cron `self_improve_nightly.yml` is **disabled_manually** on
  GitHub since 2026-08-13; the live aocc ledger has 48 records.
* Test suite: 2014 passed, 1 skipped, 62 s with `-n 4`. CI is green.
* No niceness / memory awareness anywhere (`os.nice`, `psutil` absent);
  the evaluator thread pool defaults to `cpu_count()` (16 here).
* Tracked top-level clutter: `benchmark_import.py`, `benchmark_imports.py`,
  `debug.py`, `debug_imports.py`, `logging_demo.py`, `test.sh`,
  `fabfile.py`, `run_ci.py`, `test_plan.md`, `DEVELOPMENT_PROMPT.md`,
  `.idea/`.

## 8. What this implies for the harness question

The harness *is* hooked up and elaborate (composite + AOCC tracks,
bootstrap/Wilcoxon acceptance, per-dim cells, hold-out seeds, bandit
over a mutation catalog, codify → PR). What it lacks is not another
statistic but **signal**: §2 (non-reproducible seeded runs) sets a noise
floor that the accept rule sits on, and §5/§6 mean most catalog
mutations act on parameters with little effect. Fixing §2, §5, §6 first
should be cheaper and larger than any further loop machinery.

## 9. The portfolio is worse than its own arms (measured 2026-09-09, later)

Once runs became reproducible and ~2× faster, the obvious experiment
became cheap: run each arm of the flagship *alone* (RoundRobin, no other
heuristic) on the standard IOH battery and compare.

Mean AOCC, standard battery, seeds 42 / 7 / 1234 (30 runs each):

| strategy | all | d=2 | d=5 |
|---|---|---|---|
| CMA-ES alone | **0.5801** | 0.6225 | 0.5377 |
| NLSHADE_LBC alone | 0.5373 | 0.6332 | 0.4414 |
| jSO alone | 0.4587 | 0.5530 | 0.3644 |
| PSO alone | 0.4236 | 0.4805 | 0.3667 |
| L-SHADE alone | 0.4167 | 0.5006 | 0.3328 |
| Baseline_SciPyDE | 0.4156 | 0.4999 | 0.3313 |
| Baseline_SciPyAnneal | 0.3684 | 0.4691 | 0.2677 |
| **`Rewarding_Restart` (the flagship)** | **0.3524** | 0.4051 | 0.2998 |
| Random alone | 0.3185 | 0.3645 | 0.2724 |

The flagship — a six-arm portfolio — scores **below every one of its own
population-based arms run alone**, and only 0.04 above pure random
search.  CMA-ES alone beats it by **+0.228 AOCC**, twenty times the size of any
effect the nightly loop chased in 34 nights.  Five of the nine
contenders — including three of panobbgo's own heuristics — also beat
`Baseline_SciPyDE`, the external reference the goal contract asks the
project to beat; the flagship is the reason that goal looked out of
reach.

Verified not to be a metric artifact: every run spends its full budget,
and on (d2, instance 0) CMA-ES alone reaches 9.5e-10 from the optimum
where the flagship reaches 2.2e-4.

Mechanism: population methods need their whole budget to run their
population dynamics.  Splitting a 1000-evaluation budget six ways gives
CMA-ES ~150 evaluations — fewer than it needs to adapt a covariance
matrix — while Random, Center and Nearby spend the rest on points that
a converging population would never have visited.  The portfolio does
not combine strengths; it starves them.

This also explains a puzzle in the goal contract.  `GOAL.md` §5.2 records
that *adding* CMA-ES to `Rewarding_Restart` measured flat on a 12-seed
A/B, and concluded the arm did not pay.  The arm was never the problem:
adding a seventh mouth to the same budget cannot help.  Nobody had run
it alone.

This reframes the whole improvement problem: the default should be built
around one strong population method, with other arms admitted only if a
paired A/B shows they earn their evaluations.

## 10. It is not a large-budget artifact

The obvious objection to §9 is that CMA-ES needs evaluations to adapt, so
a portfolio might still win when the budget is small.  It does not.
MA-BBOB, d=2, instances 0–2, 8 seeds per cell (24 runs), `--sync-eval`:

| budget (evals) | flagship | CMA-ES alone | NLSHADE_LBC alone |
|---|---|---|---|
| 50 | 0.2879 | **0.3028** | 0.2860 |
| 100 | 0.3137 | **0.3364** | 0.3141 |
| 200 | 0.3412 | **0.3804** | 0.3558 |
| 400 | 0.3605 | **0.4749** | 0.4348 |
| 1000 | 0.4292 | **0.6404** | 0.5841 |

CMA-ES alone wins at every budget tested, including 25 evaluations per
dimension, and the margin grows monotonically with budget (+0.015 →
+0.211).  There is no regime in this range where the six-arm portfolio
is the better choice.

## 11. It holds at higher dimension too

MA-BBOB, instances 0–2, seeds 42 / 7 / 1234, budget 100·d
(1000 evaluations at d10, 2000 at d20 — a *tight* budget for CMA-ES):

| dim | flagship portfolio | CMA-ES alone | NLSHADE_LBC alone | Baseline_SciPyDE |
|---|---|---|---|---|
| 10 | 0.1808 | **0.3005** | 0.1922 | 0.1622 |
| 20 | 0.1018 | **0.1979** | 0.1617 | 0.0804 |

CMA-ES alone is 1.7× the portfolio at d10 and 1.9× at d20, and roughly
double the external DE baseline at both.  The claim now covers dims 2,
5, 10 and 20 at budgets from 25 to 500 evaluations per dimension; the
portfolio is not preferable anywhere in that range.

## 12. Block allocation does not rescue the portfolio

§9 explained the portfolio's loss as starvation: interleaving points
denies a population method the contiguous budget it needs.  The natural
follow-up is to allocate the budget in *blocks* with `StrategyPhased`.
Measured (standard battery, 3 seeds, paired against CMA-ES alone):

| phased variant | Δ vs CMA-ES alone | CI95 |
|---|---|---|
| CMA-ES 60 % → NLSHADE_LBC | −0.0118 | [−0.0971, +0.0734] |
| NLSHADE_LBC 50 % → CMA-ES | −0.1321 | [−0.2579, −0.0063] |
| Sobol' 10 % → CMA-ES | −0.3788 | [−0.5083, −0.2494] |

Blocking is better than interleaving (−0.012 versus −0.27 for the
six-arm mix) but still does not beat the single method.  Handing CMA-ES
a warm start from another method is worse than letting it start itself,
and spending even 10 % of the budget on a Sobol' design is much worse.

A fourth variant — CMA-ES 85 % then warm-started L-BFGS-B — could not be
measured: `LBFGSB` spawns a subprocess and the driver script lacked an
``if __name__ == "__main__":`` guard, so all 30 runs failed to
initialise.  Worth re-running.  Note the sharp edge for users: any script
that builds a strategy at module level and uses `LBFGSB`, `COBYQA`,
`LocalPenaltySearch` or `QuadraticWlsModel` needs that guard.

## 13. A live-lock in the Splitter, found while sweeping CMA-ES

`Box.contains` includes both boundaries, so splitting a box through a
cluster of *identical* points puts every one of them in **both**
children.  Each child is then an over-full leaf that splits again on the
next result, and the tree deepens without bound.  The event-bus
dispatcher never returns from `on_new_results`, so every main-loop pass
pays the full 30 s settle timeout and the run dies on the stall guard
with most of its budget unspent.

Reproducible case (CMA-ES alone, MA-BBOB d5 instance 0, seed 42, budget
1000): the step size diverges to its clamp, most of a generation
projects onto the same box corner, and the run stops at **408 of 1000
evaluations after 154 s**.  With a guard that only splits along a
dimension the results differ in: **1000 of 1000 in 0.5 s**, identical
AOCC.  The whole test suite drops from 233 s to 124 s.

The Splitter is force-injected into every strategy, so any point cloud
that collapses can trigger this.  Measured AOCC is unaffected — a
stalled run's trajectory is right-padded at its final value, which is
what the optimizer would have produced anyway — so the numbers in §9–§11
stand as measured.

## 14. How much is left for a bandit? The oracle bound

With each optimizer measured alone (§9), the ceiling for *any* selection
policy over those arms is computable: take the best arm on every
instance.  Standard battery, seeds 42 / 7 / 1234, 30 instances:

| | mean AOCC |
|---|---|
| best single arm (CMA-ES) | 0.5801 |
| **oracle — best arm per instance** | **0.6523** |
| headroom for a perfect bandit | **+0.0723** |

Which arm wins each instance:

| arm | instances won (of 30) |
|---|---|
| CMA-ES | 19 |
| NLSHADE_LBC | 8 |
| PSO | 3 |
| jSO | 0 |
| L-SHADE | 0 |

Three things follow.

1. **A portfolio is worth building, eventually.** No arm dominates: the
   +0.0723 gap is five times the credit-assignment gain and a quarter of
   the portfolio-to-single-arm fix.  It is an *upper* bound — a real
   policy pays for the budget it spends discovering which arm is best,
   and §12 shows that cost is not small.

2. **Two of the five arms contribute nothing.** jSO and L-SHADE never
   win an instance, so they cannot raise the oracle and can only dilute
   a policy that includes them.  Either they improve or they should not
   be arms.

3. **Improving the arms raises the ceiling and the floor at once.**
   Every point added to a weak arm's score on the instances where it is
   already best raises the oracle; every point added anywhere raises the
   expected value of picking it.  Strengthening the individual
   optimizers therefore comes before tuning the selection policy — the
   policy can only choose among what it is given.

## 15. Phase A, first pass: every arm is over-populated

`benchmarks/arm_sweep.py` run on each of the five arms, standard
battery, seeds 42 / 7 / 1234, each variant paired per seed against that
arm's own current default.  `<--` marks a 95% t-CI that excludes zero.

| arm | variant | mean AOCC | Δ vs default | CI | seeds won |
|---|---|---|---|---|---|
| **cmaes** | `ipop_factor=1.5` | 0.7023 | ~~+0.0886~~ | *retracted — parameter never read, see §16* | |
| | `ipop_factor=3` | 0.6193 | +0.0056 | [−0.0506, +0.0619] | 2/3 |
| | *default* | 0.6137 | — | | |
| | `sigma0=0.2` | 0.6100 | −0.0037 | [−0.4033, +0.3958] | 2/3 |
| | `restart_mode=bipop` | 0.5963 | −0.0174 | [−0.0851, +0.0502] | 1/3 |
| **lbc** | `NP_init=30` | 0.5971 | **+0.0585** | [+0.0294, +0.0877] | 3/3 `<--` |
| | `H=20` | 0.5649 | **+0.0263** | [+0.0018, +0.0508] | 3/3 `<--` |
| | `H=10` | 0.5473 | +0.0088 | [+0.0004, +0.0172] | 3/3 `<--` |
| | `k_rank=6` | 0.5462 | +0.0077 | [+0.0016, +0.0139] | 3/3 `<--` |
| | *default* | 0.5385 | — | | |
| **jso** | `NP_init=30` | 0.5581 | **+0.0997** | [+0.0694, +0.1300] | 3/3 `<--` |
| | `H=20` | 0.4971 | **+0.0387** | [+0.0281, +0.0492] | 3/3 `<--` |
| | `H=10` | 0.4755 | +0.0171 | [−0.0089, +0.0432] | 3/3 |
| | *default* | 0.4584 | — | | |
| | `archive_factor=3` | 0.4135 | −0.0449 | [−0.0767, −0.0130] | 0/3 `<--` |
| **lshade** | `NP_init=30` | 0.4923 | **+0.0791** | [+0.0709, +0.0873] | 3/3 `<--` |
| | `F_schedule=jso` | 0.4336 | +0.0204 | [+0.0020, +0.0387] | 3/3 `<--` |
| | *default* | 0.4132 | — | | |
| | `archive_factor=3` | 0.3937 | −0.0195 | [−0.0317, −0.0072] | 0/3 `<--` |
| **pso** | `v_max_frac=0.2` | 0.4817 | **+0.0660** | [+0.0364, +0.0955] | 3/3 `<--` |
| | `NP=10` | 0.4762 | **+0.0605** | [+0.0360, +0.0850] | 3/3 `<--` |
| | `topology=lbest` | 0.4283 | +0.0126 | [+0.0015, +0.0237] | 3/3 `<--` |
| | *default* | 0.4157 | — | | |
| | `NP=40` | 0.3675 | −0.0482 | [−0.0886, −0.0079] | 0/3 `<--` |

### The single dominant factor is population size

All three L-SHADE-lineage arms default to `NP_init="auto"`, which
resolves to `clip(round(min(18·dim, budget/12)), max(NP_min, 6), 400)`.
A **fixed `NP_init=30` beats that on every arm and every seed**, by
+0.059 (NLSHADE_LBC), +0.079 (L-SHADE) and +0.100 (jSO).  PSO tells the
same story from its own default of 20: `NP=10` gains +0.061 and `NP=40`
loses −0.048, monotonically.

**Which term of `"auto"` is at fault (corrected).** The standard
battery's budget is `500·dim`, so `budget/12` = `41.7·dim` and the
`18·dim` term binds at *every* dimension — the budget term never
participates.  `"auto"` is therefore 36 at *d* = 2 and 83 at
*d* = 5 (see the second correction below), and the per-dimension split
says exactly that:

| arm | Δ from `NP_init=30` at *d* = 2 (auto = 36) | at *d* = 5 (auto = 83) |
|---|---|---|
| jSO | +0.0177 | **+0.1817** |
| L-SHADE | +0.0155 | **+0.1427** |
| NLSHADE_LBC | +0.0148 | **+0.1023** |

The gain is an order of magnitude larger where `"auto"` is further from
30.

**Second correction (2026-09-10).** The harness sets `max_eval` *after*
constructing the strategy (`harness_ioh.py:869-870`), so
`_resolve_auto_np_init` always saw the default 1000, never the battery
budget: `"auto"` was `min(18·dim, 83)` = 36 at *d* = 2 and **83** at
*d* = 5 and at *d* = 10 — not 90 and 180.  The direction of every
conclusion survives; the numbers 90/180 do not.  The harness bug is
fixed separately, and with the real budget in hand `"auto"` would have
been 90 / 180, i.e. even further from the optimum.

This is what §9 was measuring without naming it.  The canonical DE
population sizes come from papers whose budget is 10⁴·*d* evaluations.
Here *d* = 5 gets 2500, so `"auto"`'s 90 members buy about 28
generations against 83 for a population of 30 — not enough for
differential evolution to do much but sample.
`"auto"` already tries to correct for the budget, but on these batteries
its budget term is never the binding one.

Three consequences worth stating plainly:

* **jSO and L-SHADE were not beaten fairly in §14.** With `NP_init=30`
  jSO reaches 0.5581 and L-SHADE 0.4923, against the 0.4584 / 0.4132 that
  produced their 0-out-of-30 win counts.  The oracle bound must be
  recomputed on the tuned arms before concluding that either belongs out
  of the portfolio.
* **`"auto"`'s sizing is a library default, not a benchmark knob.** The
  fix belongs in `_resolve_auto_np_init`, where every user gets it, not
  in the harness specs.

* **Dimension and budget are confounded here.** The battery's budget is
  `500·dim`, so `NP = c·dim` and `NP = budget/k` are the same curve and
  the pooled mean cannot tell them apart.  A fixed-value grid read
  *separately at each dimension* does separate them: if the best fixed
  `NP` is the same at *d* = 2 and *d* = 5 the law is a constant, and if
  it scales by ~2.5× the law is dimensional.  That is what pass 2
  measures.

### What this pass does *not* settle

Each variant was measured alone against the default, so the gains are
not known to compose — `NP_init=30` and `H=20` may well overlap.  The
sweep also brackets rather than locates: `NP_init` was tested only at
`auto` and 30, `ipop_factor` only at 1.5 / 2 / 3, and both winners sit
at the edge of their tested range, so the optimum is plausibly beyond
it.  And the battery is only *d* = 2 and *d* = 5, so nothing here
constrains what a good population is at *d* = 10 or 20.  Three seeds is
enough to see an effect this large and not enough to
accept a default; the 12-seed roster decides.

## 16. Phase A, pass 2: CMA-ES located, the DE arms still off-grid

Same protocol as §15 (standard battery, seeds 42 / 7 / 1234, paired per
seed against each arm's own default), with the grids widened and every
delta broken out per dimension.

### CMA-ES: `ipop_factor` — RETRACTED

The table that stood here reported `ipop_factor = 1.5` as an interior
optimum worth +0.0886.  **It measured nothing.** `ipop_factor` is read
only in `_restart_ipop`, reached only from `on_restart`, published only
by the `Restart` analyzer — which every solo spec omits.  Verified by a
direct probe (same spec name, so the same seed): `ipop_factor` 1.5 /
2.0 / 3.0 give **bit-identical** AOCC.  The "effect" was the maximum of
six RNG-noise draws, because the harness derives each run's seed from
the spec *name* (`harness_ioh.py:790`) and every variant therefore ran
on a different stream.  A tight CI and a clean-looking interior optimum
came out of pure noise; the lesson is recorded in §18.

What is true about solo CMA-ES: it has **no termination or restart
criterion at all** — one CMA-ES runs to the end of the budget and keeps
sampling after σ collapses.  Adding self-restart on Hansen's criteria is
in progress.

### The DE arms: still walking down the grid

| arm | best variant | Δ vs default | *d* = 2 | *d* = 5 |
|---|---|---|---|---|
| lshade | `NP_init=10` | **+0.2186** | +0.1829 | +0.2543 |
| jso | `NP_init=15` | **+0.1859** | +0.1323 | +0.2395 |
| lbc | `NP_init=15` | **+0.1024** | +0.1068 | +0.0981 |
| pso | `NP=6` | **+0.1431** | +0.1557 | +0.1305 |

Every one of these is the *smallest or near-smallest* value tested, and
the gains roughly doubled against pass 1 — `NP_init=30` gave L-SHADE
+0.084, `NP_init=10` gives +0.219.  The population question is not a
tuning detail; it is worth more than every other knob measured so far
combined.  Pass 3 extends the grid down to 4.

**The combinations do not compose.** `NP_init=30` + `H=20` on jSO gives
+0.1336 where `NP_init=15` alone gives +0.1859, and the same holds for
lbc and for PSO's `NP=10 + v_max_frac=0.2` (+0.1169) against `NP=6`
alone (+0.1431).  The secondary knobs of §15 were mostly measuring the
population effect through a different door: once `NP` is right they add
little, and picking one is not free.

### The optimum moves with dimension

Reading each arm's argmax separately at each dimension — the split the
pooled mean cannot show:

| arm | best `NP_init` at *d* = 2 | at *d* = 5 |
|---|---|---|
| lbc | ≤ 10 | ~20 |
| jso | ≤ 10 | ~15 |
| lshade | ≤ 10 | ≤ 10 |
| pso | ≤ 6 | ≤ 6 |

For the two strongest DE arms the optimum roughly doubles from *d* = 2
to *d* = 5, which is the signature of a rule linear in dimension rather
than a constant.  The implied coefficient is **3–4·dim**, against the
`_AUTO_DIM_COEF = 18` in the code — off by a factor of five.  L-SHADE
and PSO want small populations at both dimensions and have not yet been
bracketed at all.

### Two things checked before believing any of this

* **The output-queue truncation of §5 is genuinely gone.** It would have
  invalidated the whole grid by clamping every `NP_init > 20` to 20.
  `Module._put` and `Module.emit` both call `ensure_output_capacity`
  (`core.py:679–690`), so a 60-member population really runs 60.
* **Two dimensions cannot fit a line with confidence.** *d* = 2 and
  *d* = 5 give two points, and the battery's budget is `500·dim`, so a
  dimensional rule and a budget rule remain observationally equivalent
  even here.  A *d* = 10 run (`dims=2,5,10`) is what separates them.

### Caveat: the winner's curse is now material

Three seeds and seven variants per arm means the reported maximum is the
maximum of seven noisy estimates, biased upward by roughly the spread
between neighbouring grid points.  These numbers locate a broad optimum;
they do not size the gain.  Nothing here becomes a default until the
12-seed roster confirms it against the *current* default.

## 17. Phase A, pass 3: the population law is `NP ≈ 3–4·dim`

Grid extended down to `NP_init = 4`.  Same protocol as §15/§16.  Per-
dimension deltas against each arm's `"auto"` default (36 at *d* = 2, 90
at *d* = 5):

| `NP_init` | lbc *d*=2 | lbc *d*=5 | jso *d*=2 | jso *d*=5 | lshade *d*=2 | lshade *d*=5 |
|---|---|---|---|---|---|---|
| 4 | −0.209 | −0.285 | +0.023 | −0.160 | +0.241 | +0.042 |
| 6 | +0.110 | −0.169 | **+0.200** | −0.106 | **+0.274** | +0.128 |
| 8 | **+0.143** | −0.148 | +0.189 | +0.113 | +0.211 | +0.184 |
| 10 | +0.135 | −0.061 | +0.174 | +0.067 | +0.183 | **+0.254** |
| 12 | +0.092 | +0.024 | +0.155 | +0.189 | +0.182 | +0.241 |
| 15 | +0.107 | +0.098 | +0.132 | **+0.240** | +0.136 | +0.232 |
| 20 | +0.063 | **+0.130** | +0.086 | +0.204 | +0.094 | +0.221 |
| 30 | −0.000 | +0.108 | +0.019 | +0.185 | +0.032 | +0.136 |
| 45 | −0.034 | +0.067 | −0.052 | +0.097 | −0.021 | +0.069 |
| 60 | −0.074 | +0.042 | −0.069 | +0.059 | −0.058 | +0.036 |

(Rows 30–60 from §16, same seeds.)  Every column now has an interior
maximum with a collapse below it — `NP_init = 4` is catastrophic for
NLSHADE_LBC at both dimensions — so these are located, not bracketed.

| arm | argmax *d* = 2 | argmax *d* = 5 | ratio | implied rule |
|---|---|---|---|---|
| NLSHADE_LBC | 8 | 20 | 2.5 | **4·dim** |
| jSO | 6 | 15 | 2.5 | **3·dim** |
| L-SHADE | 6 | 10–12 | ~2 | **2.5·dim** |
| PSO | 5 | 6 | ~1 | ≈ 6, constant |

The ratio *d* = 5 : *d* = 2 is 2.5 for the two strongest DE arms —
exactly the dimension ratio — which is the signature of a rule linear in
dimension.  The library's `_AUTO_DIM_COEF` is **18**; the data says
**3–4**.  (Per-dimension deltas above are against `"auto"` = 36 / 83, per
the §16 correction.)  PSO does not scale with dimension in this range and simply
wants a swarm of about six.

Absolute scores at the per-arm optimum (pooled, 3 seeds):

| arm | default | tuned | Δ |
|---|---|---|---|
| CMA-ES | 0.6137 | *(untuned — see §16 retraction)* | — |
| jSO (`NP_init=15`) | 0.4584 | **0.6443** | +0.186 |
| NLSHADE_LBC (`NP_init=15`) | 0.5385 | **0.6410** | +0.102 |
| L-SHADE (`NP_init=10`) | 0.4132 | **0.6318** | +0.219 |
| PSO (`NP=6`) | 0.4157 | **0.5588** | +0.143 |

The arms have gone from a spread of 0.20 to a spread of 0.14, and the
three DE arms are now within 0.013 of each other — they are, after all,
three versions of the same algorithm.  This changes the portfolio
question in two ways: the oracle of §14 is void (it was computed on arms
that were 5× over-populated), and "which DE variant" now matters far
less than "DE or CMA-ES on this instance".

### What a population of six means

With `NP_min = 4` and linear population-size reduction, an arm that
starts at 6 spends most of its budget as a 4-member population.  That is
barely differential evolution; it is closer to a randomised (1+λ) local
search with a memory of successful step directions.  Two readings:

* The honest one: at 500·*d* evaluations on MA-BBOB, AOCC rewards early
  progress, and early progress comes from exploiting the first good
  region hard.  A large population is a bet on multimodality that this
  budget cannot afford.
* The cautionary one: the optimum will move with budget.  At 2000·*d*
  (the competition budget) or on the more deceptive instances, a bigger
  population may pay.  The rule should therefore stay a function of the
  budget as well as of dimension, and the coefficient must be validated
  at the full budget before it ships.

### What is still open before the default changes

* **A third dimension.** Two dimensions fix a line's slope but not its
  form, and budget = 500·dim makes `c·dim` and `budget/k`
  indistinguishable here.  A *d* = 10 run (`dims=10 grid=20,30,45,60`,
  where `"auto"` currently gives 180) is in progress.
* **The 12-seed roster.** Everything above is 3 seeds and a grid of 7–10
  — the reported optimum is biased upward.  The new `"auto"` must be
  accepted against the old one on 12 seeds, as a library default, not as
  a harness spec.
* **The full budget.** One run at `bm=2000` on the standard dims.

## 18. Methodology lesson: a dead parameter passed the sweep's CI

Record of how §15–§17 reported a clean, replicated, CI-excluding-zero
"interior optimum" for a parameter that is never read.

1. The harness derived every run's seed from the spec **name**
   (`harness_ioh.py:790`).  "Variant vs default" therefore always
   compared two different RNG streams, so the paired delta of a
   *no-op* variant is not zero — it is one draw of run-to-run noise.
2. The sweep reports the **maximum** over six such draws.  The winner
   of a null tournament is biased upward by roughly the spread between
   draws.
3. "Replication" in pass 2 re-ran the identical (name, seed) pairs and
   reproduced the identical numbers — determinism mistaken for
   confirmation.
4. The per-dimension split *did* carry the signal — `ipop_1.5` showed
   −0.0001 at *d* = 2, which for a real knob would have been odd — and
   it was read as "the mechanism only acts at *d* = 5" instead of "this
   is noise".

Fixes, all in progress or done: `StrategySpec.seed_name` so variants of
one arm share a stream (a dead parameter then yields exactly 0); a
measured **null floor** (same config, six names, three seeds) recorded
alongside every sweep, so an "effect" below it is not reported as one;
and the rule that a positive result needs a *mechanism* — a claim about
which code path the parameter changes — before it is called located.
The `NP_init` results survive all three tests: the parameter is read
in `__init__`, the effect (+0.2) is far above the spread, and the
collapse at 4 is a mechanism.

## 19. Oracle on the tuned arms (provisional)

`benchmarks/oracle.py`, standard battery, seeds 42 / 7 / 1234, arms at
their pass-3 optima (lbc `NP_init=15`, jso 15, lshade 10, pso `NP=6`,
CMA-ES default — its "tuning" was retracted in §16).

| arm | mean | *d*=2 | *d*=5 | regret | seed-cell wins |
|---|---|---|---|---|---|
| lshade | **0.6342** | 0.6851 | 0.5832 | 0.062 | 7 |
| jso | 0.6232 | 0.6841 | 0.5623 | 0.073 | 5 |
| lbc | 0.6094 | 0.7103 | 0.5085 | 0.087 | 7 |
| cmaes | 0.5664 | 0.6543 | 0.4786 | 0.130 | **11** |
| pso | 0.5245 | 0.6799 | 0.3690 | 0.172 | 0 |
| **oracle** | **0.6963** | 0.7500 | 0.6425 | — | 30 |

Headroom over the best single arm: **+0.0621** [+0.029, +0.095], 3/3
seeds.  Best two-arm oracle: **CMA-ES + L-SHADE = 0.6781**, capturing
71 % of the headroom (+0.044 over L-SHADE alone).  PSO wins nothing and
every pair containing it is worse than L-SHADE alone.

Two readings, and a caveat that outranks both.

* **CMA-ES is the complement, not the champion.** It has the worst
  regret of the four real arms yet wins the most cells (11 of 30).
  It is the arm that is either right or badly wrong — exactly what a
  selector can exploit, and what a single-arm default cannot.
* **The DE arms are interchangeable.** Three variants of one algorithm
  within 0.025 of each other; a portfolio wants *one* of them plus
  CMA-ES.

**Caveat — this table has a noise floor of about ±0.05 on a battery
mean, and the spread between arms is of the same size.** The CMA-ES row
is the same configuration that scored 0.6137 in the arm sweeps; it
scores 0.5664 here because its spec is named `cmaes` instead of
`default` and the harness seeds runs from the name (§18).  A 0.047
swing from a *label* means "which DE arm is best" and "L-SHADE beats
CMA-ES" are not resolved by this run.  The oracle is a max over five
such draws per cell and is therefore biased upward as well.  The
headroom's sign and rough size — a few hundredths, most of it capturable
by two arms — is what survives; the ranking does not.  Rerun after the
`seed_name` fix on the 12-seed roster before any of this sizes Phase B.

### 18a. The null floor, measured

Same configuration under six spec names (six RNG streams), standard
battery, seeds 42 / 7 / 1234, paired per seed exactly as the sweeps are:

| config | sd of battery mean | max − min | largest spurious "gain" vs `default` |
|---|---|---|---|
| CMA-ES (default) | 0.0204 | **0.0500** | **+0.0500** (3/3 seeds!) |
| L-SHADE `NP_init=10` | 0.0121 | 0.0312 | +0.0066 |

So a 3-seed sweep can hand CMA-ES a +0.05 "improvement" with all three
seeds agreeing, from a label change.  The `ipop_factor` retraction in
§16 (+0.089, the max of six draws with one of them doubling as the
reference) is right at the edge of this distribution.

Two things the null runs show beyond the floor:

* **The `default` stream was CMA-ES's unluckiest at *d* = 5.** All five
  null streams beat it there, by +0.057 … +0.100.  CMA-ES's *d* = 5
  standing in §9, §16, §17 and §19 is therefore biased low by roughly
  0.05, and "L-SHADE beats CMA-ES" is unresolved.
* **CMA-ES's variance is bimodal, L-SHADE's is not.** A stream either
  finds the basin at *d* = 5 or it does not; that is the same "right or
  badly wrong" shape the oracle's regret/win table shows, and it is
  what a restart criterion exists to fix.

Rule adopted: a 3-seed result below ~0.05 for CMA-ES or ~0.03 for a DE
arm is not reported as an effect.  This retires pass 1's secondary knobs
(`H`, `k_rank`, `archive_factor`, `F_schedule`, `lbest`) as *unproven*,
not as wrong.  `NP_init` (+0.1 … +0.2, monotone, three arms) stands.

## 20. Phase A closes; Phase B's first screen fails for the reason the thesis predicts

### The population law holds at *d* = 10 and at 4× budget

Grid at *d* = 10 (budget 5000, `"auto"` still resolving to 83 — these
runs predate the harness fix) and L-SHADE at 2000·dim:

| arm | argmax *d*=2 | *d*=5 | **d = 10** | *d*=2 @ 4× | *d*=5 @ 4× |
|---|---|---|---|---|---|
| NLSHADE_LBC | 8 | 20 | **30** (+0.081) | | |
| jSO | 6 | 15 | **20–30** (+0.12) | | |
| L-SHADE | 6 | 10–12 | **20–30** (+0.13) | 10 | 15–20 |

`NP ≈ 3·dim` fits 6 / 15 / 30.  Quadrupling the budget moves the
optimum up by ~1.5× — a fourth-root dependence, mild but real.  The
library rule becomes `3·dim·(budget/(500·dim))^0.25`, floored at 6;
acceptance on the 12-seed roster is running.  At *d* = 10 the `H=20`
combination that looked good at *d* ≤ 5 turns negative (LBC −0.034),
one more reason the secondary knobs stay unproven.

### CMA-ES self-restart: correct, and worth +0.0001

Diagnosis at *d* = 5 / 2500: 52 % of the budget is spent after the
last improvement, 66 % after the last 10⁻³ relative one; on two of five
instances σ *diverges* against its clamp for 92 % of the run.  Hansen's
reference criteria were added and reuse the restart path (so
`ipop_factor` finally does something).  Same-stream paired gain:
**+0.0001**.  The criteria are tuned for 10⁴·d budgets; at *d* = 5 the
stagnation window alone is 95 generations ≈ 30 % of our budget, so they
fire after the run has already reached AOCC's log floor.  The
mechanism works when it fires (inst 4: 0.987 → 2.3e-8); a
budget-relative criterion and σ-divergence detection are in progress.

### The screen: every portfolio loses to its best arm

`benchmarks/portfolio_screen.py`, standard battery, 3 seeds, all specs
on one RNG stream, tuned arms:

| spec | AOCC | *d*=2 | *d*=5 | vs CMA-ES alone |
|---|---|---|---|---|
| CMAES_alone | **0.6526** | 0.7420 | 0.5633 | — |
| LSHADE_alone | 0.6171 | 0.6838 | 0.5503 | −0.036 [−0.071, −0.000] |
| Blocks_uniform_2 | 0.5661 | 0.6183 | 0.5139 | **−0.087** [−0.148, −0.025] |
| Blocks_ducb_2 | 0.5627 | 0.6264 | 0.4991 | −0.090 [−0.147, −0.033] |
| Rewarding_ema_2 | 0.5546 | 0.5933 | 0.5160 | −0.098 [−0.140, −0.056] |
| Blocks_ducb_4 | 0.4655 | | | −0.187 |
| Blocks_ducb_5 | 0.4424 | | | −0.210 |

Gates G1, G2, G3 all fail; G1 and G3 by more than the null floor with
CIs excluding zero.  The scheduler is not at fault: every run spent its
full budget, no generation was cut, D-UCB learned and gave CMA-ES 35 of
45 blocks.  Two arms that do not share information each see half the
budget, and on a log-precision anytime metric halving an arm's budget
shifts its whole curve right.  **A portfolio of independent solvers
cannot beat its best member; it can only dilute it.**  The interleaving
control loses the same amount, so this is the portfolio, not the
scheduling shape.

That is the pre-registered "one strong arm" verdict — with one clause
outstanding.  Every arm in this screen discards results it did not
request (`DESIGN_warm_start` §0); the shared archive, the thing that
distinguishes panobbgo from a bag of solvers, is switched off in all of
them.  The next screen turns it on.  If the warm-started portfolio still
loses, the answer is one arm and the effort goes into CMA-ES's
single-arm path; if it clears CMA-ES alone, sharing is the mechanism
and the policy is worth tuning.

One implementation note for later: `_close_block` discounts only the
owner's statistics, so an unplayed arm's estimate freezes rather than
decays; re-exploration comes only from the `log N / n` bonus.  Not
canonical D-UCB.  Irrelevant to this verdict.

## 21. The thesis test: sharing closes half the gap (2026-09-10, late)

Same screen as §20, warm start now actually firing (gate fix 30585b5),
only the L-SHADE arm warm-starts (CMA-ES warm start unfinished), 3
seeds, one RNG stream:

| spec | AOCC | vs CMA-ES alone |
|---|---|---|
| CMAES_alone | **0.6712** | — |
| Blocks_ducb_2_warm_any (no foreign rule) | 0.6187 | −0.052 [−0.106, +0.001] |
| LSHADE_alone | 0.6171 | |
| Blocks_ducb_2_warm | 0.6142 | |
| Blocks_uniform_2_warm | 0.6068 | |
| Blocks_ducb_2_warm_div / _leaf | 0.5980 / 0.5875 | |
| Blocks_uniform_2 (cold) | 0.5656 | −0.106 |
| Blocks_ducb_2 (cold) | 0.5636 | −0.108 |

**G4 PASS: warm − cold = +0.051**, above the null floor — sharing
evaluations across arms is a real mechanism, the first positive
portfolio result on this codebase.  **G5 FAIL: the best warm portfolio
is still −0.052 below CMA-ES alone**, CI grazing zero.  Sharing closed
half of the −0.108 gap with only one of the two arms warm.

Readings: the foreign-only rule costs a little (`warm_any` > `warm`);
the diverse/leaf selectors are worse than plain top-k here — at this
budget the value is "continue from the incumbent", not "try another
basin".  Next lever, in order: (1) CMA-ES warm start (patch in
`planning/results/2026-09-10/`), so both arms continue from the shared
incumbent; (2) then the 12-seed roster on the best warm spec vs
CMA-ES alone.  If a fully warm two-arm portfolio still cannot clear the
single arm, the answer for this budget class is one arm plus the
σ-divergence restart, and the portfolio is a higher-budget story.

## 22. Oracle on the shipped defaults, 12 seeds, paired (2026-09-10, late)

`benchmarks/oracle.py`, arms at the defaults this branch ships
(`NP_init="auto"`, CMA-ES with σ-divergence restart, PSO `NP=6`),
one RNG stream per arm, 12-seed roster, 120 cells:

| arm | mean | *d*=2 | *d*=5 | regret | seed-cell wins |
|---|---|---|---|---|---|
| **jSO** | **0.6558** | 0.7328 | 0.5788 | 0.076 | **40** |
| L-SHADE | 0.6423 | 0.7270 | 0.5577 | 0.089 | 11 |
| CMA-ES | 0.6419 | 0.7029 | **0.5808** | 0.090 | 32 |
| NLSHADE_LBC | 0.5949 | 0.6575 | 0.5323 | 0.137 | 35 |
| PSO | 0.4972 | | | 0.235 | 2 |
| **oracle** | **0.7317** | 0.8010 | 0.6624 | | 120 |

Headroom over the best single arm: **+0.0760 [+0.0483, +0.1037],
12/12 seeds**.  Best pair **CMA-ES + jSO = 0.7026**, 62 % of the
headroom; CMA-ES + L-SHADE 53 %; CMA-ES + LBC 43 %.

What changed against §19: with the arms properly sized and properly
paired, there is **no champion**.  jSO, L-SHADE and CMA-ES sit within
0.014 — inside the floor — and win *different* cells: LBC owns the
*d* = 2 instances 2–4, CMA-ES the *d* = 5 instances 2–3, jSO the rest.
"CMA-ES alone is the default" (§9, §20) was an artefact of comparing a
tuned CMA-ES with over-populated DE arms.  The portfolio headroom is
now a 12/12 result, not a 3-seed direction.

Consequences for the plan: the two-arm screen should pair **CMA-ES
with jSO**, not L-SHADE; PSO leaves the candidate set; and the honest
single-arm default is a toss-up between jSO and CMA-ES that the
*shipped* flagship spec (`RoundRobin_CMAES`) should be re-decided on
once the CMA-ES warm start and the second warm screen are in.

## 23. CMA-ES σ-divergence restart: accepted on the 12-seed roster

Solo CMA-ES, standard battery, one RNG stream, 12 seeds (8 in the
first run, 4 after the interruption, merged):

| variant | mean | Δ vs `self_restart=False` | seeds | *d*=2 | *d*=5 |
|---|---|---|---|---|---|
| old (no restart) | 0.6285 | — | | 0.6868 | 0.5702 |
| **new (shipped default)** | 0.6543 | **+0.0258 [+0.0083, +0.0433]** | **11/12** | +0.015 [−0.002, +0.033] | **+0.036 [+0.007, +0.066]** |
| new, `restart_from="best"` | 0.6561 | +0.0276 [+0.0103, +0.0448] | 11/12 | +0.016 | +0.040 |

Accept rule met: CI excludes zero on the positive side, 11/12 seeds,
no dimension negative-excluding.  0.68 restarts per run.  The gain is
heavy-tailed as in §20 — three cells +0.18…+0.44, most cells unchanged
— which is why the *d*=2 CI only grazes zero.  `restart_from="best"`
is marginally better and mechanistically the right choice after a
divergence (the divergence path already forces it); the shipped
default keeps `"random"` for the ordinary criteria, per the reference.

## 24. NLSHADE_LBC wants 4·dim — accepted as a per-class coefficient

§17 and §20 put LBC's per-dimension optimum at 8 / 20 / 30 against
6 / 15 / 20–30 for jSO and L-SHADE.  12-seed roster, LBC alone, one RNG
stream, `NP_init = 4·dim` vs the shipped `auto` (3·dim rule):

| | mean | *d*=2 | *d*=5 |
|---|---|---|---|
| auto (3·dim) | 0.5946 | 0.6575 | 0.5317 |
| **4·dim** | **0.6469** | 0.7311 | 0.5627 |
| Δ | **+0.0523 [+0.0098, +0.0948]**, 10/12 | +0.074 [−0.007, +0.154] 9/12 | +0.031 [−0.001, +0.063] 8/12 |

Accept rule met.  LBC moves from the weakest real arm (0.595) to
0.647, level with jSO / L-SHADE / CMA-ES (0.642–0.656, §22).  The
mechanism is plausible: LBC's linear bias-control needs a larger rank
pool than plain L-SHADE's success-history adaptation.  Shipped as a
class attribute (`AUTO_DIM_COEF = 4.0` on `NLSHADE_LBC`); the base rule
stays 3·dim.

## 25. Both arms warm: the portfolio reaches parity — and rotation beats the bandit

Screen #4, standard battery, 3 seeds, one RNG stream, all arms on
shipped defaults, `warm_start_only_if_foreign=False`:

| spec | AOCC | *d*=2 | *d*=5 | vs best single (jSO) |
|---|---|---|---|---|
| **Blocks_uniform_cj_warm2** (CMA-ES + jSO, both warm, rotate every block) | **0.6845** | 0.7354 | 0.6336 | **+0.011** [−0.135, +0.157] |
| JSO_alone | 0.6735 | 0.7467 | 0.6003 | — |
| CMAES_alone | 0.6712 | 0.7379 | 0.6044 | −0.002 |
| LSHADE_alone | 0.6496 | | | −0.024 |
| Blocks_ducb_cj_warm2 (both warm, D-UCB) | 0.6454 | | | −0.028 |
| Blocks_ducb_cl_warm2 (CMA-ES + L-SHADE, both warm) | 0.6454 | | | −0.028 |
| Blocks_ducb_cj_warm2cov (CMA-ES seeds C too) | 0.6269 | | | −0.047 |
| Blocks_ducb_cj_warmJ (jSO warm only) | 0.6191 | | | −0.054 |
| Blocks_ducb_cj (cold) | 0.5911 | | | −0.083 [−0.209, +0.044]; vs CMA-ES −0.080 [−0.100, −0.060] |

Gates: G1 PASS, G4 PASS (+0.094 warm vs cold), **G5 PASS** (+0.011)
— the first portfolio to clear the best single arm.  Parity, not a
win: the margin is inside the ±0.05 floor and the CI spans zero.  The
12-seed roster is running.

What is large enough to believe:

1. **Sharing is the whole effect.** Cold −0.080 (the one CI excluding
   zero); warm start on the same arms +0.054 (D-UCB) and +0.094
   (uniform).  Warm start is worth what blocking costs, and more.
2. **Bidirectional beats unidirectional.** jSO-only warm 0.619 → both
   warm 0.645; CMA-ES's warm start adds as much as jSO's.
3. **`archive_cov` hurts** (−0.019, both dims).  Seed mean and σ,
   leave C = I.
4. **With sharing on, rotation beats the bandit — reversing §21's G2.**
   Uniform beats D-UCB by +0.039 on identical arms.  The counters say
   why: D-UCB commits and warm-starts 5 times in 40 blocks; uniform
   switches every block and warm-starts 39 times in 41.  Once an arm's
   evaluations feed the shared archive, a switch is not a cost — it is
   a relay, each arm continuing from wherever the other got to.  The
   bandit was minimising a cost that sharing removed.  This is the
   thesis in its strongest form: not "pick the right arm", but "make
   every arm pick up where the others left off".

Two hypotheses this implies, being tested now: finer blocks (more
relays) should help monotonically up to the point where a block is
shorter than a generation; and a third arm should now *help*, since
dilution was the cost sharing removed.  Also whether a bandit with much
weaker commitment can beat plain rotation.

## 26. Mechanism test: block length has an interior optimum; more arms still dilute; a soft bandit leads

Screen #5, 3 seeds, one stream, CMA-ES + jSO both `warm_start="archive"`
unless stated, `warm_start_only_if_foreign=False`:

| spec | AOCC | *d*=2 | *d*=5 | warm starts / blocks (*d*=5 probe) |
|---|---|---|---|---|
| **Blocks_ducb_cj_warm2_soft** (`ucb_c=2, hysteresis=1, gamma=0.7`) | **0.6921** | 0.7515 | 0.6326 | 29 / 40 |
| Blocks_uniform_cj_warm2 (nb50, 50 evals) | 0.6845 | 0.7354 | 0.6336 | 39 / 41 |
| JSO_alone | 0.6735 | 0.7467 | 0.6003 | |
| CMAES_alone | 0.6712 | 0.7379 | 0.6044 | |
| Blocks_uniform_cj_warm2_nb25 (100 evals) | 0.6552 | 0.7317 | 0.5788 | 20 / 22 |
| Blocks_uniform_cj_warm2_nb100 (25 evals) | 0.6539 | **0.6340** | **0.6737** | 72 / 74 |
| Blocks_uniform_cjl_warm3 (+ LBC) | 0.6477 | 0.6537 | 0.6417 | 38 / 41 |
| Blocks_uniform_cjls_warm4 (+ L-SHADE) | 0.6039 | 0.6567 | 0.5511 | 37 / 41 |
| Blocks_uniform_cj_warm2_nb200 (12 evals; ~6 at *d*=2) | 0.5589 | 0.5674 | 0.5505 | 137 / 139 |

1. **Block length: an interior optimum, and it is absolute, not a
   fraction of budget.** Every warm start discards the arm's in-flight
   generation (`warm_start_now` clears output and pending), so past a
   point the scheduler throws away more work than the seed is worth,
   and neither arm gets enough consecutive evaluations to adapt.  The
   per-dimension split seemed to locate it at ~20–25 evaluations.
   **Corrected in §29:** that reading was an artefact — `n_blocks=50`
   *is* 20 evaluations at *d* = 2 and 50 at *d* = 5, so one spec's two
   columns were being read as one number.  An absolute grid puts the
   optimum at **~50** under rotation, agreeing at both dimensions.
   `n_blocks` is still the wrong parametrisation; a block length in
   evaluations is right, but the value is 50, not 20.
2. **More arms still dilute.** 2 → 3 → 4 arms: 0.685 → 0.648 → 0.604,
   a gentler slope than cold (§20) but the same sign.  Only at *d* = 5
   does the third arm pay (+0.037); at *d* = 2 it costs −0.084.  If a
   third arm ever ships it is dimension-gated.
3. **A soft bandit beats rotation.** With commitment removed
   (`hysteresis=1`, high `ucb_c`, fast forgetting), D-UCB lands in the
   middle band of switching — 29 warm starts against 5 (default D-UCB,
   §25) and 39 (rotation) — and leads the table, positive at both
   dimensions.  What matters is not maximising relays but landing in
   that band, and a soft bandit finds it adaptively where a fixed
   `n_blocks` has to be tuned per dimension.

All 3-seed rankings inside the floor except the nb200 collapse and the
four-arm loss.  Next: block length in absolute evaluations (12–80) and
a small grid over the soft knobs, then the 12-seed roster on the
winner.

## 27. 12-seed roster: the warm portfolio leads, but does not clear the acceptance bar

`Blocks_uniform_cj_warm2` (CMA-ES + jSO, both `warm_start="archive"`,
rotate every 50-eval block) against the single arms, 12-seed roster,
one RNG stream, standard battery:

| spec | mean | *d*=2 | *d*=5 |
|---|---|---|---|
| **Blocks_uniform_cj_warm2** | **0.6854** | 0.7272 | **0.6436** |
| CMAES_alone | 0.6662 | 0.7205 | 0.6119 |
| Blocks_ducb_cj_warm2 (default D-UCB) | 0.6643 | 0.7118 | 0.6167 |
| JSO_alone | 0.6490 | 0.7241 | 0.5739 |
| LSHADE_alone | 0.6460 | 0.7326 | 0.5593 |

| paired delta | Δ | 95 % CI | seeds | *d*=2 | *d*=5 |
|---|---|---|---|---|---|
| uniform-warm vs CMA-ES alone | +0.0192 | [−0.0156, +0.0541] | 8/12 | +0.007 | +0.032 |
| uniform-warm vs jSO alone | +0.0364 | [−0.0112, +0.0841] | 9/12 | +0.003 | +0.070 |

The portfolio is the best spec on the roster and ahead of every single
arm on a majority of seeds, with no dimension negative — but the CI
against the best single arm includes zero, so **by the rule it is not
accepted as the default**.  Parity with an upward lean, carried by
*d* = 5, where sharing is worth +0.03…+0.07.  At *d* = 2 (1000
evaluations) there is nothing to gain: a single arm converges before a
relay could help.

Also settled: on 12 seeds the best single arm is CMA-ES (0.666), not
jSO as on 3 seeds (§22, §25).  The three arms are level, and which one
"wins" is a property of the seed set.  `RoundRobin_CMAES` stays the
flagship for now.

What could move this above the bar, all measured on 3 seeds in §26 and
being confirmed: the soft bandit (+0.008 over rotation), a block length
in absolute evaluations (~20–25), and not re-seeding the arm that made
the latest progress (§26.1's mechanism).  If those land the portfolio
at +0.03 with a CI clear of zero, it becomes the flagship; if not, the
honest default for 500·dim is one arm, and the portfolio is the answer
for *d* ≥ 5 or larger budgets — a dimension-gated spec.

## 28. Oracle on the final defaults (LBC at 4·dim), 12 seeds

| arm | mean | *d*=2 | *d*=5 | regret | seed-cell wins | majority cells |
|---|---|---|---|---|---|---|
| jSO | **0.6558** | 0.7328 | 0.5788 | 0.074 | 39 | d2i0 d5i0 d5i1 d5i4 |
| NLSHADE_LBC (4·dim) | 0.6469 | 0.7311 | 0.5627 | 0.083 | 34 | **d2i1 d2i2 d2i3 d2i4** |
| L-SHADE | 0.6423 | 0.7270 | 0.5577 | 0.087 | 12 | — |
| CMA-ES | 0.6419 | 0.7029 | **0.5808** | 0.088 | 35 | **d5i2 d5i3** |
| oracle | **0.7295** | 0.8022 | 0.6568 | | 120 | |

Headroom **+0.0737 [+0.0477, +0.0998], 12/12**.  Best pairs: CMA-ES +
jSO 0.7026 (63.6 %) ≈ CMA-ES + LBC 0.7023 (63.1 %).

Four arms within 0.014.  The cells sort by dimension: LBC owns four of
the five *d* = 2 instances, CMA-ES the hard *d* = 5 ones, jSO the rest.
That is a context signal a selector could use (dimension is known
before the first evaluation), and it argues for testing CMA-ES + LBC
as the relay pair alongside CMA-ES + jSO.

## 29. Absolute block length is ~50 under rotation; the "soft bandit" is round-robin plus a greedy tail; first CI clear of zero

Screen #6, 3 seeds, one stream, CMA-ES + jSO both warm, no
`only_if_better` (predates c13748d):

| spec | AOCC | *d*=2 | *d*=5 | vs CMA-ES alone |
|---|---|---|---|---|
| **soft_be25** (soft knobs, `block_evals=25`, tail 0.25) | **0.7211** | 0.7814 | 0.6607 | **+0.0499 [+0.0058, +0.0940]**, 3/3, both dims |
| soft (nb50), and every `ucb_c`/`gamma` variant | 0.6905–0.6930 | | | +0.019…+0.022 |
| uniform be50 | 0.6852 | 0.7369 | 0.6336 | +0.014 |
| uniform nb50 (= be20 at *d*=2, be50 at *d*=5) | 0.6845 | 0.7354 | 0.6336 | +0.013 |
| uniform be80 | 0.6841 | 0.7367 | 0.6315 | +0.013 |
| JSO_alone / CMAES_alone | 0.6735 / 0.6712 | | | |
| uniform be30 / be20 | 0.6596 / 0.6583 | | | −0.012 / −0.013 |
| uniform be12 | 0.5939 | 0.6373 | 0.5505 | −0.077 |

**(A) Block length under rotation: monotone up to ~50, flat beyond.**
The absolute grid is a superset of the old one — `be20` reproduces
nb50's *d* = 2 column exactly and `be50` its *d* = 5 column — which is
how §26's "20–25" is exposed as a reading error.  Realised lengths
(`size=10` draws cannot stop mid-draw): nominal 20 → 24, 50 → 54/56.

**(B) The soft knobs do nothing.** `ucb_c=4` is bit-identical to 2 in
30/30 cells; `ucb_c=1`, `gamma=0.5`, `gamma=0.9` differ by ≤ 0.002.
With `hysteresis=1` and `ucb_c ≥ 2` the exploration bonus swamps the
value estimate, so selection is alternation — and the only thing that
distinguishes "soft D-UCB" from uniform is the **exploit-only tail**
(`tail_frac=0.25`, `c=0` in the last quarter): 30 switches in 40
blocks instead of 40 in 41, ownership 25/15 instead of 20/20.  So the
policy that leads is *relay often, then stop relaying and let the
leader run*.  `block_evals=25` under that policy is worth +0.029 more —
an interaction: with a greedy ending, shorter blocks during the relay
phase pay, where under pure rotation they cost.

**(C) `soft_be25` is the first spec whose CI excludes zero** against the
best single arm, on 3 seeds, positive at both dimensions, sitting at
the ±0.05 floor's edge.  On the 12-seed roster now, in both
`only_if_better` variants.  The next screen puts `tail_frac` itself on
the grid (0…0.6, with 0 as the control that tests whether "soft" is
anything but the tail), block length under the tail, the
`only_if_better` guard, and the CMA-ES + LBC pair from §28.

## 30. 12-seed roster on `soft_be25`: the winner's curse, with a CI as its fig leaf

| spec | mean | vs CMA-ES alone | seeds | *d*=2 | *d*=5 |
|---|---|---|---|---|---|
| CMAES_alone | 0.6662 | — | | 0.7205 | 0.6119 |
| soft_be25 (no `only_if_better`) | 0.6604 | **−0.0058** [−0.0387, +0.0271] | 6/12 | −0.017 | +0.005 |
| soft_be25 (`only_if_better`) | 0.6532 | −0.0130 [−0.0440, +0.0181] | 5/12 | −0.024 | −0.002 |
| JSO_alone | 0.6490 | | | | |

§29's +0.0499 [+0.0058, +0.0940] on 3 seeds was the **maximum of
fourteen specs** in one screen.  A CI computed on a selected maximum is
not the CI of a pre-registered spec; the roster returns it to zero.
§18 named this mechanism for `ipop_factor`; this is the same mechanism
passing through a CI instead of a point estimate.  Rule sharpened: the
best spec of a screen is a *candidate*, its screen CI carries no
weight, and only its roster CI does.

Standing after the roster: `Blocks_uniform_cj_warm2` (plain rotation,
§27) remains the best portfolio on 12 seeds — 0.6854, +0.019 over
CMA-ES, 8/12, CI including zero.  `only_if_better` costs ~0.007 and is
switched off by default again.

**Verdict for 500·dim, MA-BBOB, d ∈ {2, 5}: a two-arm portfolio that
shares its evaluations is level with the best single arm, not above
it.**  Sharing removed the portfolio's structural penalty (−0.08 →
+0.02); it did not buy a lead.  The lean is at *d* = 5 (+0.03 on every
roster), nothing at *d* = 2, where 1000 evaluations end before a relay
can matter.  `RoundRobin_CMAES` stays the flagship; the warm portfolio
becomes the second harness spec so the nightly keeps measuring it, in
place of `Rewarding_Restart` (0.35, no longer a useful control).

## 31. Screen #7: the tail carries nothing, the bandit carries nothing; what is live is warm start + rotation + block length

3 seeds, one stream, CMA-ES + jSO both warm:

| spec | AOCC | *d*=2 | *d*=5 |
|---|---|---|---|
| soft_be25, `tail_frac` 0 / 0.1 / 0.25 / 0.4 / 0.6 | 0.7217 / 0.7217 / 0.7211 / 0.7197 / 0.7190 | 0.781 | 0.662 → 0.658 |
| **uniform_be25** | 0.7068 | 0.7400 | **0.6737** |
| soft_be50 / uniform nb50 | 0.6838 / 0.6783 | | |
| JSO_alone / CMAES_alone | 0.6735 / 0.6712 | | |
| soft_be20 / soft_be35 | 0.6709 / 0.6625 | | |
| CMA-ES + LBC soft_be25 | 0.6627 | **+0.034** | **−0.050** |
| soft_be25 with `only_if_better` | 0.6479 | −0.033 | −0.014 |

* **`tail_frac` is flat** — 0.003 across 0 → 0.6, monotonically down;
  the optimum is no tail.  That retires §29's mechanism claim.
* **At `tail_frac=0` the soft D-UCB *is* round-robin**: the *d* = 5
  probe gives 74 blocks / 73 switches / 37–37 ownership and the same
  AOCC as `uniform_be25`, byte for byte, at both dimensions.  The two
  specs are identical in 23 of 30 cells; the +0.015 mean gap is
  **one cell** (seed 1234, *d* = 2, inst 4: 0.86 vs 0.27), without
  which uniform leads by 0.005.  The bandit — value estimate, bonus,
  discount, hysteresis, tail — contributes nothing measurable at any
  setting tried.
* **Block length is flat-ish from 20 to 50 with per-cell collapses**
  (be20 and be35 each have one collapsed cell that be25 dodged;
  medians 0.82 / 0.83 / 0.76 / 0.80).  "25 mildly preferred, 25–50
  supported" is all the data says.  On the roster (§30) be25 sat at
  −0.006 and nb50 at +0.019 — within noise of each other.
* **`only_if_better` hurts, −0.073**: it cuts warm starts from 72 to
  43, lopsidedly — CMA-ES usually holds the incumbent, so the guard
  starves jSO (12 warm starts instead of 36), the arm that most needs
  the relay.  Default off.
* **CMA-ES + LBC is not the pair**: +0.034 at *d* = 2, −0.050 at
  *d* = 5 — LBC's 4·dim population does not fit a 25-eval block at
  *d* = 5.  jSO stays.

Standing for 500·dim, *d* ∈ {2, 5}: **parity** (§30), and the design
space around the selection policy is measured out — the policy is
worth nothing, the sharing is worth everything.  The remaining
questions are not about the bandit: larger budgets and *d* ≥ 10 (where
the *d* = 5 lean predicts a gain), and a per-arm-relative
`only_if_better` if the guard is ever wanted.

## 32. The Splitter cannot resolve where the search has not looked (found while designing the meta level)

`splitter.py`: `limit = max(20, max_eval/dim²)` and children inherit
the parent's points, so the tree settles at **≈ 1.3·dim² leaves
regardless of budget** — ~5 at *d* = 2 (the root cannot split before
evaluation 250 of 1000), ~35 at *d* = 5, ~530 at *d* = 20 (1.4 cuts
per axis).  The split axis is the widest dimension whose coordinates
differ; function values play no part.  Every "where is something left
to gain" mechanism (Random-in-best-box, RegionUCB, `per_leaf_best`,
the meta heuristic of `DESIGN_meta_level`) reads this tree, and it is
force-injected into every strategy.  Owner's verdict: improve the
Splitter — budget-scaled resolution and a value-aware split rule, in
progress on an isolated worktree, measured through its consumers.

## 33. Meta level, smallest version: exact null, and a structural blind spot in the shared archive

`benchmarks/meta_screen.py`, *d* = 5 only, 3 seeds, one stream,
reference `Blocks_uniform_cj_warm2` (0.6336):

| spec | Δ vs reference | per instance |
|---|---|---|
| Meta_never | **+0.0000** in 15/15 cells (exact null) | |
| Meta_b25 (leaf scan, 50 points at ¼ budget) | +0.0020 | +.007 +.007 +.008 −.004 −.009 |
| Meta_random_b25 (same trigger, uniform points) | **+0.0020, identical in every cell** | same |
| Meta_stag | +0.0044 | 0 / +.006 / +.016 / 0 / 0 (fired on 2 of 5) |
| RegionUCB_arm (stream at the same cost) | −0.0008 | |
| Meta_region_b25 (hand a leaf to CMA-ES) | **−0.0131**, negative in 4/5 | |

Probe of the identical rows: both fire once at evaluation 626 into a
13-leaf tree, both emit 50 points, **zero of those points improve the
best** in either mode, so both runs end at the same value; the +0.002
is the block schedule shifting by 50 diverted evaluations.  The design's
falsifier R1 fires exactly: as a point emitter, the analysis is
decoration.  R3 fails directionally: restricting an arm's warm start to
one leaf is worse than the whole archive — the same sign as
`only_if_better` (§30); narrowing what an arm may re-seed from keeps
measuring negative.

**The structural finding.**  The shared archive is top-K *by value*.
Exploration points are, by construction, worse than the incumbent, so
they never enter it — and the population arms see foreign results
only through it.  Information of the form "this region is empty or
unknown" has no channel into the solvers except the region hand-off,
which is negative.  Whatever a meta level is to contribute, it cannot
be points into the current archive.

Preconditions before this is measured again: the Splitter rework (§32
— 13 leaves at *d* = 5 after 626 evaluations is not a map), and a
model (design step 5, gated off by this screen).  Then `Meta_stag`
with its own random-points control, the only rows with a structure.

## 34. Off MA-BBOB: parametrised families and the first constrained battery

`panobbgo/lib/families.py` builds instances `f(x) = f_base(Λ·R·(x − x_opt)) + f_opt`
from classics with a known minimiser (sphere, rosenbrock, rastrigin,
ackley, griewank, schwefel + BBOB ellipsoid / discus / sharp_ridge),
exact `f(x_opt) == f_opt`; constrained families add linear or ball
constraints built around `x_opt` with the first one **active at the
optimum**.  AOCC on a constrained instance is scored on the penalty
value `f + 100·cv`, the same scalar `Best` and the block scheduler use.
3 seeds, one stream, 500·dim:

| battery | cells | CMA-ES | jSO | L-SHADE | **portfolio** (warm2) | portfolio vs best single |
|---|---|---|---|---|---|---|
| free, d 2/5/10, 5 families | 135 | **0.3947** | 0.3770 | 0.3599 | 0.3484 | −0.046 [−0.135, +0.042], 0/3 |
| constrained, d 2/5, 4 families | 72 | 0.4318 | **0.4649** | 0.4609 | 0.4219 | −0.043 [−0.104, +0.018], 0/3 |

* **The MA-BBOB parity does not travel.** The portfolio is last on both
  batteries, negative on all three dimensions and all five free
  families — inside the floor, but the same sign in eight of nine
  slices, which the MA-BBOB roster never showed.
* **The bar is a property of the regime.** CMA-ES wins the free battery
  at *d* ≥ 5 (+0.06/+0.08) and loses *d* = 2 badly (0.554 vs jSO 0.646);
  on the constrained battery both DE arms beat it (+0.033/+0.029, 3/3).
  `LSHADE − CMAES` on the free battery is the only marked CI
  (−0.035 [−0.069, −0.001]) — a reference-vs-reference result.
* **Constraints work and carry the loudest class structure**: no arm
  errored, mean AOCC is *higher* than on the free battery, and the
  portfolio is −0.21 on `ellipsoid_ball`, −0.10 on `rosenbrock_lin`,
  **+0.09 on `rastrigin_ball`, +0.06 on `sphere_lin`** — it pays where a
  relay past a multimodal landscape helps and loses where a valley
  needs sustained adaptation.
* **Caveat that outranks the tables: DE arms are not bit-reproducible
  across processes.** Same code, seed, `sync_eval`: 3 of 48 cells
  differ (max |Δ| 0.095), all in jSO/L-SHADE-containing specs; CMA-ES
  is identical everywhere; in-process repeats agree.  Being hunted
  (prime suspect: hash-seed-dependent iteration over `who` ids).  Until
  fixed, every DE A/B in this file carries that term.

## 35. Splitter rework: resolution scales with budget

See commit 8a35927.  Leaves at (d, budget): 38 / 105 / 204 / 351 against
6 / 33 / 142 / 602 legacy; the root now splits at evaluation 37, not
250.  Consumers, 3 seeds vs legacy: Random +0.049, RegionUCB +0.038,
`archive_leaf` warm start +0.042 (3/3 seeds each; roster running); the
reference portfolio 0.0000 in every cell.  The value-aware split rule
is +0.006 and ships opt-in.  Next: median cut (the mean cut turns a
contracting cloud into a chain, depth 60), and `RegionUCB` gets an
`on_start` (it cannot start a run alone — a pre-existing defect).

## 36. Invariant tests: nine findings

`tests/test_invariants.py` (320 tests, 30 s) and
`planning/results/2026-09-10/invariants_findings.md`.  HIGH: (F1)
`StrategyRoundRobin` divides by the number of *active* arms — a run
whose last arm goes inactive raises `ZeroDivisionError` and leaks
threads; (F2) `warm_start=` on NLSHADE_RSP/LBC draws from the RNG in
`_archive_cap()` *before* the empty-archive bail-out, so setting the
kwarg alone changes the stream (the contract test covered only
JSO/PSO/CMAES); (F3) L-BFGS-B / COBYQA contribute zero points beside
any competitor (their pump thread never has a point queued when polled);
(F4) the wall-clock stall guard truncates slow arms, so a seeded run is
machine-dependent.  MEDIUM: (F5) `PSO(stagnation_threshold=)` is inert
under the default topology — the `ipop_factor` shape; (F6) six arms
lose to Random (DE without auto sizing, NelderMead, WeightedAverage,
RegionUCB, Extremal, LHS).  The `sigma_divergence` flag in F9 is a
false alarm of the detector — divergence is rare on DeJong at 300
evaluations; on the battery it fired 17 times in 60 runs (§23).
Clean and now guarded: the §5 queue contract for all population arms,
constructor RNG order, all 91 handler signatures, no NaN/out-of-box
points.

## 37. Splitter resolution accepted on the 12-seed roster (2026-09-11)

Consumers of the tree, new resolution vs `legacy=True`, 12 seeds, dims
(2, 5), instances 0–2, one stream (`planning/results/2026-09-11/splitter_roster.*`):

| consumer | Δ new − legacy | 95 % CI | seeds | *d*=2 | *d*=5 |
|---|---|---|---|---|---|
| RoundRobin_Random | +0.0330 | [+0.009, +0.057] | 9/12 | +0.084 | −0.018 |
| RoundRobin_RegionUCB (+Random bootstrap) | **+0.0500** | [+0.023, +0.078] | **12/12** | +0.103 | −0.003 |
| Blocks_uniform_cj_warmleaf | **+0.0530** | [+0.027, +0.079] | 11/12 | +0.054 | +0.053 |

Accept rule met for all three.  The gain is concentrated at *d* = 2,
where the old tree had 5 leaves; at *d* = 5 the two direct consumers
are flat.  `archive_leaf` warm start gains at both dimensions but at
0.502 remains far below the `archive` mode (0.685, §27) — the finer
tree helps the leaf selector, it does not make leaf selection the
right warm-start policy.  The reference portfolio is untouched (§35).

## 38. Noise and higher dimension: the portfolio still does not lead; outliers destroy the DE arms

`panobbgo/lib/noise.py` (BBOB gauss / unif / cauchy, deterministic per
(seed, x); AOCC scored on the true value), presets `noisy`, `highdim`
(d 10/20 at 2000·dim), `noisy-highdim`.  3 seeds, one stream per cell:

| battery | CMA-ES | jSO | L-SHADE | portfolio | vs best single |
|---|---|---|---|---|---|
| noiseless 500·d (control) | 0.671 | 0.674 | 0.650 | 0.685 | +0.011 |
| noisy-gauss 500·d | 0.630 | **0.697** | 0.661 | 0.679 | −0.017 (d=2 −0.105, **d=5 +0.071** 3/3) |
| noisy-unif 500·d | 0.638 | **0.659** | 0.640 | 0.666 | +0.008 (d=2 −0.054, d=5 +0.051) |
| noisy-cauchy 500·d | **0.661** | 0.490 | 0.505 | 0.561 | −0.100 |
| noiseless d=10, 500·d | **0.403** | 0.375 | 0.357 | 0.397 | −0.006 |
| noisy-gauss d=10, 500·d | **0.500** | 0.363 | 0.350 | 0.401 | −0.099 |
| noiseless d=10, 2000·d | 0.519 | 0.600 | **0.652** | 0.623 | −0.029 |
| noiseless d=20, 2000·d | **0.477** | 0.414 | 0.388 | 0.480 | +0.003 |

* **Sharing does not pull ahead in any regime tested.** The lean at
  *d* = 5 is sharper under noise (+0.07) and the loss at *d* = 2 is
  larger (−0.10); the mean does not move.
* **Cauchy outliers collapse the DE arms** (jSO −0.21, L-SHADE −0.16)
  and leave CMA-ES untouched (−0.01), with CIs clear of zero.  A
  rank-based recombination survives a constant shift plus rare
  outliers; greedy per-individual replacement does not.  This has a
  mechanism and counts as located: **under outlier noise, no DE arm in
  the portfolio.**
* **The best arm is a function of the regime**: jSO at *d* ≤ 5 under
  gaussian/uniform noise, CMA-ES at *d* = 10/500·d, *d* = 20 and under
  outliers, L-SHADE at *d* = 10/2000·d.  Dimension and budget are known
  before the first evaluation; the noise class is not, but is
  detectable from re-evaluations.  This is the strongest evidence yet
  for *selection by regime* — a context-gated spec — over any online
  bandit.

Caveats: 3 seeds, screen CIs on selected specs carry no weight (§18,
§30); noisy-vs-noiseless comparisons are unpaired (different
`problem_kind` → different stream).  Raw: `planning/results/2026-09-11/`.

## 39. Splitter v2: the median cut is not the fix for the chain; RegionUCB could never start alone

* A median cut does **not** cure `Random`'s depth-60 chain — mean and
  median give the identical pathology (61 leaves, a 1300-point leaf at
  *d* = 5).  The chain is `Random`'s own: it samples only inside the
  current best leaf, that leaf splits, the child with the best point is
  the next target, the sibling is never revisited.  Fixing it means
  changing the heuristic (or `MAX_DEPTH`), not the geometry.  The
  earlier diagnosis in §35 was wrong and is retracted.
* A *plain* median cut is dangerous: it lands on an observed
  coordinate, and with both-boundaries `contains` a duplicated mass is
  counted into both children — §13 in a softer form.  The shipped
  `cut_rule="median"` cuts the median *gap* between adjacent distinct
  coordinates.  Median vs mean, 3 seeds: RegionUCB +0.033 (3/3), Random
  −0.015; `Blocks_uniform_cj_warm2` at *d* = 5 hits `MAX_DEPTH` with a
  423-point leaf under the median where the mean stays at depth 52.
  `"mean"` stays the default; `"median"` is opt-in.
* **`RegionUCB` had no `on_start`** and emits only from
  `on_new_results`, so alone it produced **0 of 200 evaluations** — a
  pre-existing defect hidden by every benchmark that paired it with
  another arm.  With a six-line initial design it spends its budget
  and is the stronger of the two standalone tree consumers (0.43–0.46
  vs Random's 0.37–0.39).

## 40. The DE arms' cross-process nondeterminism was a thread race in `Results.add_results`

Not hash randomisation (ruled out over five `PYTHONHASHSEED` values).
`add_results` published `new_results` **before** extending the result
buffer; the handler cascade ran on the bus thread while the main thread
was still writing.  The L-SHADE family paces LPSR, the F-schedule and
`p_best` annealing on `len(strategy.results)`, so whether a handler
counted its own batch depended on which thread won the GIL — ~8 stale
reads in 204 batches at the default switch interval.  One stale read
moves an LPSR step a loop earlier, one fewer trial is drawn, and the
stream shifts by exactly 4 bytes (`new_who` draws 16); every point after
that differs.  CMA-ES never reads `len(results)` — hence immune.
`sync_eval` serialises evaluation, not this overlap; in-process repeats
mostly agreed because the race is rare.

Fix f4b6376: publish last.  Worst cell 9/10 → 10/10 subprocesses
identical; family battery seed 42 in three parallel processes 3/48 →
0/48 differing rows.  A handler-time probe now asserts the count
(fails 5/5 on the old ordering).  Effect on measured numbers: small —
the fixed order is the branch that won ~11 times in 12 — but every DE
A/B before f4b6376 carried this term, and no reproducibility test could
see it because they all ran within one process.

## 41. Regime table: selection by regime is the right direction, not yet evidence

`planning/REGIME_TABLE_2026-09-11.md`: 1932 rows over 17 regime cells
(kind × noise × constrained × dim × budget/dim), paired streams only.

* Winners flip: CMA-ES 6 cells, jSO 5, portfolio 3, L-SHADE 3; 9 of
  17 margins clear the floor.
* Fitted regime oracle 0.577 vs fixed CMA-ES 0.538 (+0.039); the
  fixed portfolio (0.528) is *worse* than fixed CMA-ES across regimes.
* **Honest (leave-one-seed-out) gain over CMA-ES: +0.026 [−0.006,
  +0.057], 3/3 — inside the floor.**  The 12-seed control is negative:
  gating by dimension alone on MA-BBOB scores −0.017 [−0.048, +0.015],
  5/12 — where the arms sit within the floor, selection costs more than
  it earns.
* Gating clearly beats the portfolio (+0.036 [+0.011, +0.062]) and
  L-SHADE (+0.055, CI clear).
* The five regimes with an established mechanism (Cauchy ×2, gauss
  d=10, families d=5/10) contribute **zero** against a CMA-ES default —
  CMA-ES is already best there; the whole +0.026 is switches away from
  CMA-ES at d ≤ 5 and d=10/2000·d, all on 3 seeds.
* Dimension alone is negative out of sample; dim × noise (+0.014) and
  dim + budget + noise + constrained (+0.031) carry the signal; budget
  per dim is fully confounded with dimension in this data.  Regime
  gating captures ~64 % of per-instance headroom.
* **Tool defect found**: `benchmarks/oracle.py` pinned `seed_name` per
  arm, so its per-cell maxima were over *unpaired* draws (one cell off
  by 0.52 against the paired stream).  Fixed 40cdd08; §22/§28's oracle
  numbers are upper-biased.

The runs that decide (12 seeds, launched): standard at 2000·dim
(d 2/5), cauchy / gauss / unif, and d=10 at 500·dim.

## 42. 12-seed regime runs, first four: the portfolio wins under uniform noise; outliers belong to CMA-ES

Standard MA-BBOB cube (d 2/5, 500·dim) under the BBOB noise models, and
d = 10 at 500·dim, 12-seed roster, one stream per cell
(`planning/results/2026-09-11/r12_*`):

| regime | CMA-ES | jSO | L-SHADE | portfolio | portfolio vs best single arm |
|---|---|---|---|---|---|
| **cauchy** | **0.640** | 0.505 | 0.506 | 0.539 | −0.101 [−0.124, −0.078], 0/12; DE arms −0.135 (0/12) |
| gauss | 0.643 | **0.660** | 0.641 | 0.666 | +0.005 vs jSO (5/12); +0.022 vs CMA-ES [−0.003, +0.048] 9/12 |
| **unif** | 0.635 | 0.643 | **0.645** | **0.672** | **+0.026 [+0.003, +0.050], 9/12** vs L-SHADE; +0.037 [+0.012, +0.061], 10/12 vs CMA-ES; both dims positive |
| d = 10, 500·dim | **0.422** | 0.367 | 0.353 | 0.400 | −0.023 [−0.070, +0.024] 6/12; DE arms −0.056 / −0.069 (CI clear) |

* **Uniform noise is the first regime where the sharing portfolio
  clears every single arm on the roster by the acceptance rule.**  The
  lean is at *d* = 5 (+0.055 vs CMA-ES) with *d* = 2 also positive.
* **Outliers belong to CMA-ES**, by a margin no other result in this
  file approaches (DE arms −0.135 with 0/12 seeds).  Mechanism as in
  §38; now a 12-seed fact.
* Gaussian noise: the portfolio leads but not by the rule; jSO is level
  with it.
* *d* = 10 at 500·dim: CMA-ES, clearly; the portfolio is level, the DE
  arms lose.

Regime gating now has two branches with 12-seed evidence — outliers →
CMA-ES alone; gaussian/uniform noise at *d* ≤ 5 → the sharing portfolio
— and the noise class is the one regime feature not known a priori: it
needs a probe (re-evaluate a few points; outliers show as a heavy tail
in the differences).  Pending: 2000·dim and 200·dim at *d* 2/5 (does
budget move the noiseless verdict), constrained, and the paired oracle.

## 43. Paired oracle on the shipped defaults, 12 seeds (replaces §28's unpaired numbers)

`benchmarks/oracle.py` with one RNG stream for all arms (40cdd08):

| arm | mean | *d*=2 | *d*=5 | regret | seed-cell wins |
|---|---|---|---|---|---|
| jSO | **0.6488** | 0.7197 | 0.5780 | 0.081 | 31 |
| CMA-ES | 0.6372 | 0.6795 | **0.5949** | 0.092 | **41** |
| L-SHADE | 0.6312 | 0.6990 | 0.5634 | 0.098 | 20 |
| NLSHADE_LBC | 0.6145 | 0.6669 | 0.5620 | 0.115 | 28 |
| oracle | **0.7295** | 0.7922 | 0.6668 | | 120 |

Headroom **+0.0807 [+0.0636, +0.0978], 12/12**.  Best pair CMA-ES +
jSO 0.7118 — **78 % of the headroom** (+0.063 over jSO); every other
pair ≤ 56 %.  Pairing did not change the story of §28, it sharpened
it: CMA-ES is the complement (most wins, worst regret of the four), jSO
the safest single arm, and the two together are the portfolio worth
having.  The measured sharing portfolio of those two arms captures
none of this at 500·dim noiseless (§27) and all of it under uniform
noise (§42) — the headroom is real, the selection is the bottleneck.

**Addendum to §36 (f41bf62):** F6 is superseded — `Random` samples inside
the Splitter's best leaf and was never a null; against a uniform null
DifferentialEvolution, WeightedAverage and LatinHypercube beat it, only
`Extremal` is worse.  `RegionUCB.ucb_c` was under-probed, not dead.
F1–F5 and F7 (for LocalPenaltySearch) are fixed on master.

## 44. Budget decouples the noiseless verdict: the sharing portfolio wins at 200·dim, is level at 500·dim, and is indistinguishable at 2000·dim; constrained on 12 seeds belongs to jSO

Raw: `results/2026-09-11/r12_{bm200,bm2000,constrained}.{json,log}`, git
HEAD 40cdd08, 12-seed roster, one RNG stream, 0 errors.  Numbers below
were recomputed from the JSON rows, not read off the logs.

### 44.1 Budget per dimension, standard battery (MA-BBOB, *d* ∈ {2, 5})

Paired delta of `Blocks_uniform_cj_warm2` (CMA-ES + jSO, both warm from
the archive, 50-eval rotation) against the single arms:

| budget | vs CMA-ES | seeds | *d*=2 | *d*=5 | vs jSO | vs L-SHADE | rule |
|---|---|---|---|---|---|---|---|
| **200·dim** | **+0.0365 [+0.0045, +0.0685]** | **9/12** | +0.014 | +0.059 | +0.0567 [+0.018, +0.096], 10/12 | +0.0542 [+0.014, +0.095], 10/12 | **accepted** |
| 500·dim (§27) | +0.0192 [−0.0156, +0.0541] | 8/12 | +0.007 | +0.032 | +0.0364, 9/12, CI incl. 0 | — | parity |
| 2000·dim | +0.0017 [−0.0191, +0.0226] | 6/12 | +0.002 | +0.001 | −0.005 | −0.001 | nothing |

Means at 200·dim: portfolio **0.5239**, CMA-ES 0.4874, jSO 0.4697,
L-SHADE 0.4672; at 2000·dim all four inside 0.005 (jSO 0.8257, L-SHADE
0.8213, portfolio 0.8204, CMA-ES 0.8187).

**This is the second win by the rule** (after uniform noise, §42), and
the cleaner one: it beats *all three* arms, both dimensions positive,
and the budget series is monotone — +0.037 → +0.019 → +0.002.  Sharing
evaluations is a **low-budget effect**.  The mechanism is the one §27
guessed for *d* = 2: the warm hand-off at block boundaries buys speed
in the early curve, and AOCC at a short budget is *all* early curve.
With 2000·dim every arm converges on its own and the tail, where nothing
is left to gain, dominates the area.  Equally, the §27 "parity" was a
budget artefact, not a verdict on the idea — at the budget panobbgo was
written for (expensive functions, a few hundred evaluations per
dimension) the shared archive pays.

Two caveats.  (i) `NP_init="auto"` scales with the budget, so at 200·dim
the DE arms run with ≈2.4·dim individuals (floor 6) — both alone and
inside the portfolio, so the comparison is fair, but the absolute DE
numbers are those of a small population.  (ii) *Corrected in §46:*
the spec has no `block_evals`, so it runs the `n_blocks=50` default —
block = `max(round(budget/50), 2·dim)` — which at 400 evaluations
(d=2) is a **12-evaluation block and ~37 hand-offs**, not ~8.  The
*d* = 2 number (+0.014) is depressed by short-block stalls; the *d* = 5
number (+0.059) is the robust one.

**Regime consequence.**  Budget per dimension is known *before* the
first evaluation — it needs no probe, unlike the noise class.  It is
the first gate branch that is both 12-seed-accepted and free:
`bpd ≤ 200 → CMA-ES + jSO sharing`.  Goes into `REGIME_TABLE_V1` now
(design §2.4 planned it for `_V2`; there is no reason to wait — the row
carries 12 seeds).  Open: where between 200 and 500 the crossover sits
(one run at 300·dim would place it), and whether 100·dim widens the
gap or the DE arm starves.

### 44.2 Constrained families, 12 seeds

`preset constrained` (ellipsoid_ball, rastrigin_ball, rosenbrock_lin,
sphere_lin; *d* ∈ {2, 5}; 500·dim; 288 cells):

| spec | mean | *d*=2 | *d*=5 |
|---|---|---|---|
| **JSO_alone** | **0.4704** | 0.5806 | 0.3602 |
| LSHADE_alone | 0.4576 | 0.5706 | 0.3446 |
| CMAES_alone | 0.4419 | 0.5226 | **0.3613** |
| Blocks_uniform_cj_warm2 | 0.4392 | 0.5751 | 0.3033 |

| paired delta | Δ | 95 % CI | seeds | *d*=2 | *d*=5 |
|---|---|---|---|---|---|
| jSO vs CMA-ES | +0.0285 | [+0.0077, +0.0493] | 9/12 | +0.058 | −0.001 |
| portfolio vs CMA-ES | −0.0028 | [−0.0190, +0.0135] | 5/12 | +0.053 | −0.058 |
| portfolio vs jSO | −0.0312 | [−0.0553, −0.0072] | 2/12 | −0.006 | −0.057 |

The 3-seed picture (§38: portfolio last, DE arms ahead) survives with a
sharper edge: jSO leads CMA-ES with a CI clear of zero on 9/12 seeds
but *d* = 5 is −0.001, so by the letter of the rule it is a lean, not an
acceptance — the whole jSO advantage is a *d* = 2 effect.  The portfolio
is level with CMA-ES and **loses to jSO by the rule** (2/12).  Per
family vs jSO: ellipsoid_ball **−0.175**, rosenbrock_lin −0.033,
rastrigin_ball +0.033, sphere_lin +0.049 — one family carries the loss,
and it is the ill-conditioned one with the active ball constraint.

Hypothesis, not yet tested: the top-K archive on a constrained problem
is a set of points crowded along the active constraint, so
`warm_start="archive"` hands CMA-ES a mean *on* the boundary with a σ
collapsed along it, from which the penalty gradient (not the objective)
dominates the next generation.  The test is cheap — `warm_start=None`
on the CMA-ES arm only, constrained battery, 3 seeds first — and if it
holds, the constrained row of the table is `("JSO",)` for now and the
archive needs a feasibility-aware K for later.  Constrained-or-not is
free to read (`eval_constraints` is not `None`), so this row, too,
needs no probe; it enters the table as a lean with 12 seeds behind it,
flagged as not rule-accepted.

## 45. Oracle regime gate, 12 seeds: not falsified, and it costs exactly nothing — the gate is worth precisely what the table is worth

Step 1 of `DESIGN_regime_gating_2026-09-11.md` landed (71cd083):
`StrategyBlockBandit(regime_gate="oracle:<class>")`, every arm
constructed up front, a per-arm `_enabled` mask read in one place
(`_select`), `REGIME_TABLE_V1` with the four-field key (noise, dim,
bpd, constrained) and the two probe-free §44 rows.  Raw:
`results/2026-09-11/rg_{cauchy,unif,gauss,standard,d10_bm500,bm200}.json`,
12 seeds, 0 errors, all deltas recomputed from the rows.

| cell | gate → | vs `CMAES_alone` | vs portfolio |
|---|---|---|---|
| cauchy | CMA-ES | **identical, 120/120 cells** | **+0.1009 [+0.078, +0.124], 12/12** — accepted |
| unif | both | **+0.0367 [+0.013, +0.061], 10/12**, d2 +0.018 d5 +0.055 — accepted | identical |
| gauss | both | +0.0224 [−0.003, +0.047], 9/12, d2 −0.003 | identical |
| standard 500·dim | CMA-ES | **identical** (the must-not-lose control) | −0.019, 4/12 (= §27 parity) |
| d=10, 500·dim | CMA-ES | identical, 36/36 | +0.023, 6/12 |
| standard 200·dim | both | **+0.0365 [+0.005, +0.069], 9/12** — accepted (= §44.1 to the 4th decimal) | identical |

Two facts, one of them unexpected.

**Not falsified.**  The oracle beats CMA-ES by the rule on unif and on
200·dim, and on cauchy it *is* CMA-ES — bit for bit — so the cauchy
evidence is the +0.101 recovery over the portfolio at zero cost against
the arm.  Every row fired where it should.

**The design's §2.2 caveat did not materialise.**  A disabled arm still
receives `on_new_results` and advances its own stream, but that stream
never touches the enabled arm's, and a never-selected arm never has a
point evaluated — so "CMA-ES via the gate" equals `CMAES_alone` in all
276 gated-to-one-arm cells, and "both arms via the gate" equals the
ungated portfolio in all 360.  The mask is free.  The consequence is
sharper than the design expected: **the value of regime gating is
exactly the value of the table**, and the only price any future probe
(`table-v1`, steps 2–3) can add is its own k evaluations.

### 45.1 What this makes shippable now

Three of the table's rows need no probe — outlier is the only class
that must be *detected*, and the bounded row is the only one that
*needs* the detection.  Assume `clean` (`regime_gate="oracle:clean"`)
and the gate reduces to: **d ≤ 5 and ≤ 200·dim → CMA-ES + jSO sharing;
constrained → jSO; everything else → CMA-ES.**  Measured against the
flagship `RoundRobin_CMAES` ≡ `CMAES_alone` this is identical in every
clean cell we have (500·dim, 2000·dim, d = 10), better by the rule at
200·dim, a 12-seed lean on constrained, and identical (not worse) under
cauchy and gauss; under unif it forgoes +0.037.  A default that is
never worse by the rule anywhere measured and better by the rule in one
regime is the first candidate to replace the flagship since Phase A.
Proposal: make it the default of the portfolio spec
(`Blocks_warm_CMAES_JSO`), keep `RoundRobin_CMAES` as the plain
reference, and revisit the flagship label once one more low-budget
point (§45.2) is in.

### 45.2 What the probe is worth, and why it is not next

The probe buys the bounded row only: +0.037 (unif) to +0.022 (gauss,
not by the rule), all of it at *d* = 5 (*d* = 2: +0.018 / −0.003), at
the price of k evaluations and a 0.40 cauchy detection rate whose
safety comes from the fallback, not the detector.  A ceiling of ~+0.03
in one regime.  The low-budget line (§44.1) is the larger and cleaner
lead — a monotone series with a rule win at its end and the crossover
between 200 and 500·dim not yet located — and it is the regime panobbgo
was written for.  So: probe deferred; next runs are 100·dim and
300·dim on the standard battery (where does sharing start and stop
paying), and whether a *third* arm or a shorter block helps at 200·dim
(§25's dilution and §29's block length were measured at 500·dim only).
Then the constrained warm-start hypothesis of §44.2.

## 46. Budget series 100…2000·dim: sharing is largest at the lowest budget (+0.056, 12/12), the *d* = 2 dip is a short-block artefact, third arm and D-UCB still lose

Raw: `results/2026-09-11/bs_bm100.json`, `bs_bm300.json`,
`bs_bm200_structure.json` (12 seeds, 0 errors, every row at full
budget; the 240 cells shared with `r12_bm200.json` reproduce bit for
bit).  All deltas recomputed from the rows.

### 46.1 The series, `Blocks_uniform_cj_warm2` − `CMAES_alone`

| bpd | Δ | 95 % CI | seeds | *d*=2 | *d*=5 | rule |
|---|---|---|---|---|---|---|
| **100** | **+0.0558** | [+0.037, +0.074] | **12/12** | +0.052 | +0.060 | **accepted** — also vs jSO +0.057 and L-SHADE +0.059, both 12/12 |
| 200 | +0.0365 | [+0.005, +0.069] | 9/12 | +0.014 | +0.059 | accepted (§44.1) |
| 300 | +0.0114 | [−0.031, +0.054] | 7/12 | **−0.020** | +0.043 | no |
| 500 | +0.0192 | [−0.015, +0.053] | 8/12 | +0.007 | +0.032 | parity (§27) |
| 2000 | +0.0017 | [−0.019, +0.023] | 6/12 | +0.002 | +0.001 | nothing (§44.1) |

The strongest acceptance in the project so far.  At 100·dim the three
single arms are within 0.003 of each other (0.356/0.355/0.354) and the
portfolio is 0.412 — no arm starves (jSO at the NP floor of 6 is level
with CMA-ES alone), the gain is sharing.  The **d = 5 column is
monotone**: +0.060 → +0.059 → +0.043 → +0.032 → +0.001.  The *d* = 2
column is not, and the reason is structural, not statistical.

### 46.2 The block the default spec actually runs

`Blocks_uniform_cj_warm2` carries no `block_evals`, so it uses
`n_blocks=50`: block = `max(round(budget/50), 2·dim)`.  Realised (seed
42, inst 0):

| bpd | *d* | block | hand-offs | | *d* | block | hand-offs |
|---|---|---|---|---|---|---|---|
| 100 | 2 | **6 = one generation** | 34 | | 5 | 16 | 32 |
| 200 | 2 | 12 | 35 | | 5 | 24 | 39 |
| 300 | 2 | 12 | 46 | | 5 | 32 | 41 |
| 500 | 2 | 24 | 43 | | 5 | 56 | 39 |

At 200–300·dim, *d* = 2, the block is two CMA-ES generations and the
portfolio **stalls** in 6–8 of 60 cells (AOCC 0.17–0.30 where CMA-ES
alone reaches 0.55–0.82 and the same cells at 500·dim reach 0.66–0.91);
with a block ≥ 24 the count is 0–2.  The likely mechanism — both arms
re-seeded from a tight top-K into one basin every two generations — is
not measured yet.  So the default's *d* = 2 series mixes a ≈ +0.03
sharing gain with a growing loss from stalled cells; §44.1's caveat
(ii) was wrong and is corrected in place.

**Control with the block pinned at four generations
(`block_evals="auto"`, new spec `Blocks_uniform_cj_warm2_auto`, 70fbdc8):**

| bpd | Δ vs CMA-ES | 95 % CI | seeds | *d*=2 | *d*=5 | rule |
|---|---|---|---|---|---|---|
| 100 | +0.0359 | [+0.018, +0.054] | 10/12 | +0.029 | +0.043 | accepted |
| 200 | +0.0423 | [+0.018, +0.067] | 11/12 | +0.031 | +0.054 | accepted |
| 300 | +0.0248 | [−0.001, +0.051] | 9/12 | +0.026 | +0.023 | misses by 0.001 |

With the block fixed the picture is clean: *d* = 2 flat at ≈ +0.03
across 100–300, *d* = 5 decaying.  The one-generation relay at 100·dim
adds a further +0.020 [−0.003, +0.043], 8/12 over `auto` — a lean.
The crossover sits around 300·dim (lower bound −0.001); a fixed-block
point at 500 would settle whether it is 300 or 500.

### 46.3 Structure at 200·dim (Run B, vs `Blocks_uniform_cj_warm2`)

| spec | Δ | seeds | *d*=2 | *d*=5 | vs CMA-ES |
|---|---|---|---|---|---|
| `_auto` (block 24/48) | +0.006 | 5/12 | +0.017 | −0.005 | **+0.042, 11/12** acc. |
| `_be25` (30/32) | −0.000 | 5/12 | +0.002 | −0.003 | +0.036, 10/12 acc. |
| `_be50` (54/56) | −0.007 | 4/12 | +0.008 | −0.022 | +0.030, 10/12 acc. |
| `cjl_warm3` (+ NLSHADE_LBC) | −0.020 | 3/12 | +0.009 | **−0.048** | +0.017, 8/12 |
| `ducb_cj_warm2` (D-UCB) | −0.028 | 4/12 | −0.007 | **−0.048** | +0.009, 8/12 |

Block length is flat from 12 to 56 evaluations on the pooled mean (all
within 0.013, all accepted vs CMA-ES) — only the *d* = 2 stall of the
12-eval block is structure.  The third arm still dilutes (−0.020,
carried by *d* = 5, same sign and size as §26 at 500·dim) and drops the
pair below the rule.  D-UCB still adds nothing and at a low budget
costs −0.028: a committing bandit has too few blocks to learn from and
forgoes the relays (§31 again).

### 46.4 Consequences

* The low-budget row of `REGIME_TABLE_V1` should carry
  `block_evals="auto"` (or any fixed block ≥ 24), not the `n_blocks`
  default — `auto` is accepted at 100 and 200 with the tightest CIs
  and has no stall.  Same for the portfolio spec's default.
* The regime row `bpd ≤ 200` stands; `bpd ≤ 300` is borderline
  (`auto` lower bound −0.001) and stays out until a fixed-block 500
  point places the crossover.
* Sharing at 100·dim is the regime to aim the seams at
  (`DESIGN_seams_2026-09-11.md`): the seam experiment runs at 100 and
  200·dim on the `auto` block, so the stall cannot masquerade as a seam
  effect.

## 47. Seams, step 1: injection is inert under block rotation; in a channel that carries it, within-generation sharing is real but four times smaller than the hand-off

`DESIGN_seams_2026-09-11.md` §2 landed (733af5c): `CMAES(inject=True)`
(Hansen 2011, clipped, capped λ/4) and `JSO(shared_pbest=True)`, both
proven inert alone (byte-identity tests).  Raw:
`results/2026-09-11/seams_{bm100,bm200,unif}.json` (round 1, on the
`auto` block, 12 seeds, 0 errors) and `seams2_{bm100,bm200}.json`
(round 2, the cold and round-robin channels).  All deltas recomputed
from the rows; the `auto` baseline reproduces §46 cell for cell.

### 47.1 Round 1: `inject` never fired; `shared_pbest` is a small loss

| cell | inject vs `warm2_auto` | pbest vs `warm2_auto` |
|---|---|---|
| 100·dim | **bit-identical, 120/120** | −0.0105 [−0.021, +0.000], 5/12 |
| 200·dim | bit-identical | −0.0136 [−0.043, +0.015], 3/12 |
| unif | bit-identical | −0.0023, 6/12 |

Injection is inert *by construction*, not by bug: the screen runs
`sync_eval=True`, so no point is ever in flight; the only foreign
points CMA-ES sees while it holds an open generation arrive during
jSO's block, and `warm_start_now` on re-acquisition discards that
generation together with its injected list.  Under block rotation +
warm-start-on-resume there is no channel.  (An instrumented threaded
local run *does* inject — via in-flight points at the block switch — an
artefact of the executor, not a mechanism.)  The design's risk 1 was
the whole story.  Shared pbest is a lean loss at low budget: jSO pulls
pbest from CMA-ES's points, which can sit in another basin, and the
differential is mis-scaled.

### 47.2 Round 2: give the seams a channel — cold blocks and per-point round-robin

| pair | 100·dim | 200·dim |
|---|---|---|
| **(a)** `inject_cold_auto` − `cold_auto` | +0.0051 [−0.000, +0.010], 8/12 | +0.0106 [+0.001, +0.020], 9/12 — accepted, barely |
| **(b)** `RoundRobin_cj_seams` − `RoundRobin_cj_cold` | +0.0077 [+0.002, +0.013], 9/12 | **+0.0331 [+0.021, +0.045], 12/12** |
| `cold_auto` − `warm2_auto` (what the hand-off is worth) | **−0.0761 [−0.087, −0.065], 0/12** | **−0.1379 [−0.166, −0.110], 0/12** |
| `RoundRobin_cj_cold` − `CMAES_alone` (the per-point penalty) | −0.0418, 0/12 | −0.0969, 0/12 |
| `RoundRobin_cj_seams` − `CMAES_alone` | −0.0341, 0/12 | −0.0638, 0/12 |

Within-generation sharing is **real** once a channel exists — (b) is
accepted at both budgets, 12/12 at 200·dim — and it recovers about a
third of the per-point structural penalty (−0.097 → −0.064).  But every
cold or per-point spec loses to `CMAES_alone` and to the warm block
portfolio by the rule, 0/12, both dimensions.  The **hand-off** —
re-fitting m/σ/C (and jSO's population) to the shared archive at
re-acquisition — is worth +0.08…+0.14; both seams together are worth
+0.03 at best.  Four to one.

### 47.3 What this says about "decompose CMA-ES"

The sharing that pays is not point-level: it is the *distribution*
being handed the other arm's knowledge in one move — Hansen's
"mean-shift" injection in its strongest form, applied to the whole
search distribution rather than one ranked point.  The block structure
is not a limitation to engineer around; it is the mechanism.  So the
building block worth generalising is **the hand-off itself** — what it
carries (mean, σ, C, population), from which points, how often — and
the seams that reduce *evaluations* (surrogate pre-screening, lq-CMA-ES)
rather than the ones that add information to a ranking.  Step 2 of the
seam catalogue (pre-evaluation) keeps its case; model-blended ranking
does not inherit any credit from this round.

`inject` and `shared_pbest` stay in the code as measured, inert-alone
knobs (they are the falsifier's record); neither is a default.

## 48. What the hand-off carries: both directions or nothing, the top-k and not a spread, and covariance only where the sample supports it

§47.3 named the hand-off as the building block worth generalising, so
this ablation splits the gap it opened.  Raw:
`results/2026-09-13/ho_bm100.json`, `ho_bm200.json` (12 seeds, dims 2
and 5, 5 instances, `block_evals="auto"`, uniform rotation, 120 cells
per budget, 0 errors, every run spent its full budget).  Four new specs
(95edf29) against the baseline `Blocks_uniform_cj_warm2_auto` (both arms
re-seed from the archive top-k) and the bar `Blocks_cj_cold_auto` (same
arms, no hand-off):

* `Blocks_cj_warmC_auto` / `_warmJ_auto` — only CMA-ES / only jSO
  receives the hand-off,
* `Blocks_cj_warm2_cov_auto` — CMA-ES additionally seeds **C** from the
  archive cloud (`archive_cov`, `cma_es.py:673`); jSO unchanged,
* `Blocks_cj_warm2_div_auto` — both arms re-seed from k well-separated
  good points (`archive_diverse`) instead of the top-k.

The baseline reproduces §47 cell for cell (`warm2_auto − cold_auto`
= +0.0761 / +0.1379, the same to four decimals), so the added specs did
not move anyone else's stream: `_derive_seed` hashes the strategy name.

### 48.1 The hand-off is bidirectional, and superadditive

All deltas paired per (seed, dim, instance), t(11) = 2.201.

| over `cold_auto` | 100·dim | 200·dim |
|---|---|---|
| only CMA-ES warm | +0.0269 [+0.022, +0.032], 12/12 | +0.0512 [+0.042, +0.060], 12/12 |
| only jSO warm | +0.0249 [+0.018, +0.032], 12/12 | +0.0539 [+0.040, +0.068], 12/12 |
| sum of the two | +0.0518 | +0.1052 |
| **both warm (baseline)** | **+0.0761** | **+0.1379** |
| superadditivity | **+0.0243 (47 % over the sum)** | **+0.0327 (31 %)** |

Each direction alone is accepted by the rule and each is worth about a
third (100·dim) to two fifths (200·dim) of the full hand-off — but both
one-sided variants lose to the two-arm baseline by the rule (−0.049 /
−0.051 at 100·dim, 0/12; −0.087 / −0.084 at 200·dim, 1/12) and to
`CMAES_alone` as well.  There is no cheap one-sided version to ship.

The superadditivity is the finding.  The hand-off is not a one-way
transfer into an arm; it is a **ratchet**: a warm-started arm writes
better points into the shared archive, which is the pool the *other*
arm's next hand-off draws from.  One-sided warm breaks the loop — the
cold arm keeps feeding the archive its unimproved points.  Prediction to
test: the value should grow with the *number* of hand-offs (more, shorter
blocks) until the switching transient eats it, which is the other end of
§46.3's flat block-length curve.

### 48.2 The selector: the top-k crowd is the point; a spread is worth nothing

| over `cold_auto` (i.e. what the hand-off is still worth) | 100·dim | 200·dim |
|---|---|---|
| top-k (`archive`, baseline) | +0.0761, 12/12 | +0.1379, 12/12 |
| `archive_cov` (top-2n + covariance) | +0.0463, 12/12 | +0.1085, 12/12 |
| **`archive_diverse`** | **−0.0022 [−0.006, +0.001], 3/12** | **−0.0036 [−0.017, +0.010], 7/12** |

`archive_diverse` does not merely underperform the top-k: it is
statistically **indistinguishable from no hand-off at all**, at both
budgets, both CIs straddling zero.  Re-seeding from k well-separated
good points destroys the entire value of the hand-off (−0.078 / −0.142
against the baseline, 0/12).  Whatever the hand-off does, it does by
placing the receiving arm's distribution *on the incumbent*, with a σ
fitted to a tight cloud — the spread version hands it a wide σ and a
mean in no basin at all.  The hand-off is intensification, not
diversification; the exploration in this portfolio comes from the arms,
not from the transfer.

### 48.3 `archive_cov`: location always, shape only where the sample supports it

| `archive_cov` − baseline | overall | d = 2 | d = 5 |
|---|---|---|---|
| 100·dim | −0.0298 [−0.041, −0.019], 1/12 | −0.0067 | −0.0529 |
| 200·dim | −0.0294 [−0.061, +0.003], 3/12 | **+0.0100** | −0.0688 |

The sign splits cleanly by dimension, and at 200·dim `archive_cov` has
the best *d* = 2 mean of every spec measured (0.5822 vs the baseline's
0.5721) while costing −0.069 at *d* = 5.  The mechanism is the sample
size: `_warm_start_seeds` asks for `k = max(λ, 4 + ⌊3 ln n⌋, 2n)`, i.e.
10 points at *d* = 5 for a 5×5 covariance — and those 10 are the archive
*top*-10, so they are strongly correlated by construction.  Nothing
regularises that estimate: `_seed_covariance` normalises to unit
determinant and only falls back to **I** above cond 1e7, which a noisy
10-point cloud never reaches.  So at *d* = 2 (10 points for a 2×2) the
shape is real information and pays; at *d* = 5 it is an over-fitted
ellipse that the arm then has to unlearn.

This is the concrete new building block this ablation produces:
**shrink the seed covariance toward the identity by its own sample
size** — `C = (1−α)·I + α·Ĉ` with α a function of *k*/*n* (Ledoit–Wolf,
or simply `α = clip((k − n − 1)/(c·n), 0, 1)`), and a condition cap of
order 10²–10³ rather than 1e7.  At *d* = 2 that leaves today's winning
behaviour almost untouched (α ≈ 1); at *d* = 5 it collapses gracefully
to the location-only hand-off instead of costing −0.07.  Cheap, local to
`cma_es.py`, and directly falsifiable: `archive_cov` with shrinkage must
be ≥ `archive` at both dimensions, or the shape carries nothing.

### 48.4 Consequences

* The shipped hand-off stays as it is: **both arms warm, `archive`
  top-k**.  Nothing in this ablation is a cheaper or better default.
* `archive_diverse` is out as a warm-start selector for this portfolio.
  It remains available for other uses, but no spec should reach for it
  on the strength of "diversity is good": here it is exactly equal to
  switching sharing off.
* `archive_cov` is *not* dead — it is unregularised.  The shrinkage
  above is the next implementation step, measured on the same two
  budgets against `archive` per dimension.
* The ratchet reading of §48.1 gives the block-length question a
  hypothesis (value grows with hand-off count) that §46.3 measured only
  as a flat curve on the pooled mean; a re-read of those cells against
  the number of realised hand-offs is free.

## 49. Shrinking the seeded covariance: the fix works at 100·dim and is bracketed out at 200·dim — `archive_cov` stays a *d* ≤ 2 knob

§48.3 proposed shrinkage toward the identity by sample size as the repair
for `archive_cov`'s *d* = 5 loss, with the falsifier "≥ plain `archive` at
both dimensions".  Landed in b19c7ab as three opt-in `CMAES` kwargs
(defaults off, `archive_cov` without them bit-identical to §48 — the
unshrunk cells reproduce to four decimals):

* `warm_start_cov_shrink` — `C = (1−α)·I + α·Ĉ` blended on the
  eigenvalues, re-normalised to unit determinant, with
  `α = clip((k − n − 1)/(c·n(n+1)/2), 0, 1)`.  The denominator counts the
  covariance's **free parameters**, not *n*: with `c·n` no single constant
  hits both targets (at *d* = 2 the seed set is 6, so α = 1 needs
  `c ≤ 1.5`, and that same `c` leaves α ≥ 0.53 at *d* = 5).  Counting
  `n(n+1)/2` gives with one constant `c = 1` exactly α = 1 at *d* = 2 and
  α = 4/15 at *d* = 5.  Ledoit–Wolf was rejected in the docstring: its
  intensity is derived for an i.i.d. sample, and the archive top-k is
  selected by objective value from a concentrating search.
* `warm_start_cov_cond_max` — caps cond(C) by clipping the eigenvalue
  ratio (spec: 1e3); the 1e7 hard fallback stays, now read off the raw
  ratio.
* `warm_start_wide_seeds` — the `2n` seed floor in *any* mode: the
  location-only control for §48.3's hypothesis (B).

Raw: `results/2026-09-14/covshrink_{bm100,bm200}.json`, same roster and
cube as §48, 0 errors, full budgets.

### 49.1 The measurement

Paired per-seed deltas against the `archive` baseline
(`Blocks_uniform_cj_warm2_auto`), t(11) = 2.201:

| bm | spec − baseline | overall | *d* = 2 | *d* = 5 |
|---|---|---|---|---|
| 100 | **shrunk cov** | −0.0058 [−0.020, +0.008], 4/12 | −0.0072, 8/12 | **−0.0045 [−0.024, +0.015], 6/12** |
| 100 | unshrunk cov (§48) | −0.0298 [−0.041, −0.019], 1/12 | −0.0067, 8/12 | −0.0529 [−0.073, −0.033], 0/12 |
| 100 | wide seeds, C = I | −0.0012, 5/12 | **0.0000, bit-identical** | −0.0024 [−0.017, +0.012], 5/12 |
| 200 | **shrunk cov** | −0.0177 [−0.046, +0.011], 4/12 | +0.0079, 6/12 | **−0.0434 [−0.067, −0.020], 1/12** |
| 200 | unshrunk cov (§48) | −0.0294 [−0.061, +0.003], 3/12 | +0.0100, 6/12 | −0.0688 [−0.101, −0.037], 1/12 |
| 200 | wide seeds, C = I | −0.0082, 5/12 | **0.0000, bit-identical** | −0.0165 [−0.039, +0.006], 5/12 |

Shrinkage over unshrunk at *d* = 5: **+0.0484 [+0.028, +0.069], 12/12**
at 100·dim, +0.0254, 8/12 at 200·dim.  At *d* = 2 the two are within
0.002 — α = 1 there, so only the condition cap separates them.

### 49.2 Falsifier: failed, and the family is bracketed

At 100·dim shrinkage does what it was designed to do: the −0.053 loss at
*d* = 5 becomes −0.0045 with the CI straddling zero, i.e. parity, while
*d* = 2 keeps its lean.  At 200·dim it is still −0.0434, CI clear of
zero, 1/12 — the rule is not met, so **`archive_cov` does not become a
default**.

And no constant `c` will rescue it, because the endpoints of the
shrinkage path are both measured: `wide seeds, C = I` **is** the α → 0
endpoint (the same seed set, identity shape) and the unshrunk spec is
α = 1.  At *d* = 5 / 200·dim those bracket the family in
[−0.0165, −0.0688] — every value of α lies inside a negative interval.
Tuning the constant against the decision roster would only be picking
the least-negative point of a losing family, which is exactly the
winner's curse the rule exists to prevent.

### 49.3 Which hypothesis, and a new one

Hypothesis **(A)** (over-fitted shape) carries most of it: of the *d* = 5
loss, the wider sample explains −0.002 of −0.053 at 100·dim and −0.017 of
−0.069 at 200·dim.  **(B)** is real but small — and note *what* it is:
the `2n` floor does not move **m** (that is the μ-weighted mean of the
best μ = 4 seeds, the same four points at k = 8 and k = 10), only **σ**.
So the wide control isolates a pure *spread* effect, the same knob that
killed `archive_diverse` in §48.2, and it has the same sign.

The new observation is in the budget column.  The pure shape cost
(shrunk − wide, both on the same seed set) is **−0.0021, 3/12 at
100·dim** and **−0.0269 [−0.048, −0.006], 2/12 at 200·dim**: at the low
budget the seeded shape is free, at the higher budget it is the whole
loss.  The block is what changed — `auto` gives 40 evaluations per block
at *d* = 5 / 100·dim and 48 at 200·dim, but the *budget* gives 12.5
blocks versus 20.8, so at 200·dim the arm has adapted its own **C** over
far more generations before each hand-off, and the hand-off overwrites
it.  The seeded shape does not have to be good; it has to be better than
what it replaces, and what it replaces improves with the budget.

### 49.4 The payload that has never been measured: keep the arm's own C

Every warm-start mode in the code today **discards CMA-ES's adapted
covariance**: `_warm_start_distribution` calls `_reset_covariance(n)`
(C = B = I, D = 1, paths zeroed) unless the mode is `archive_cov`, which
replaces it with the archive's shape instead.  So the three payloads
measured so far are *identity*, *archive shape* and (now) *shrunk archive
shape* — and the fourth, **preserve the arm's own C and move only m and
σ**, does not exist.

§48.2 says the hand-off works by relocating the receiving arm onto the
incumbent; §49.3 says the cost of the payload grows with how much
adaptation is thrown away.  Together they predict that the best payload
is the *relocation without the reset*: keep B, D, C (and, an open sub-
question, the evolution paths), refit m to the archive's μ-weighted best
and σ to the cloud spread.  It is a small change in the same method, it
is falsifiable the same way (≥ `archive` at both dimensions, 100 and
200·dim), and it should pay most exactly where shrinkage failed — at the
higher budget, where the discarded C is worth most.

Consequence for the shipped configuration: unchanged.  Both arms on
`archive`; `archive_cov` (shrunk or not) is a *d* ≤ 2 knob with a lean,
not a default; `warm_start_wide_seeds` is a control, not a feature.

## 50. The ratchet is real at 100·dim and saturates at 200·dim; blocking itself is free; the short-block stall is a hand-off pathology

§48.1's superadditivity suggested the hand-off is a **ratchet** — a
warm-started arm writes better points into the archive, which is the pool
the other arm's next hand-off draws from — and predicted that its value
grows with the *number* of hand-offs.  §46.3 could not test that: it
compared warm specs at four block lengths against `CMAES_alone`, which
mixes the hand-off's value with the switching transient, and its
`_be25`/`_be50` pin `warm_start_only_if_better=False` (`NOB`) while
`_auto` takes the default.  So this run builds **warm/cold pairs at four
block lengths with identical settings** (078114d): `warm − cold` is the
hand-off alone, with the transient held fixed.  Raw:
`results/2026-09-14/ratchet_{bm100,bm200}.json`, same roster and cube,
0 errors, full budgets.  The `auto` pair reproduces §48 exactly
(+0.0761 / +0.1379).

### 50.1 Value of the hand-off against the number of hand-offs

| block | blocks *d*=2 / *d*=5 | `warm − cold` at 100·dim | at 200·dim |
|---|---|---|---|
| 50 | 4 / 10 (bm100), 8 / 20 (bm200) | +0.0526 [+0.043, +0.062] | +0.1197 [+0.106, +0.133] |
| `auto` (24 / 40–48) | 8.3 / 12.5, 16.7 / 20.8 | +0.0761 [+0.065, +0.087] | +0.1379 [+0.110, +0.166] |
| 25 | 8 / 20, 16 / 40 | +0.0794 [+0.070, +0.089] | +0.1307 [+0.113, +0.148] |
| `n_blocks=50` (4–10 / 8–20 evals) | 50 / 50 | **+0.0980 [+0.080, +0.116]** | +0.1327 [+0.103, +0.162] |

All 12/12.  At **100·dim the ratchet is there and monotone**: from 4
blocks to 50 the hand-off nearly doubles in value (+0.053 → +0.098), in
both dimensions separately (d=2 +0.043→+0.097, d=5 +0.063→+0.099).  At
**200·dim it is flat** — every length lands in [+0.120, +0.138], the
ordering scrambled and well inside the CIs.  The ratchet saturates: by
8 blocks at the higher budget the archive already holds what the other
arm can use, and further hand-offs add nothing.

This is also why the shipped default beats `auto` at the low budget
(`Blocks_uniform_cj_warm2` +0.0558, 12/12 vs CMA-ES at 100·dim, against
`auto`'s +0.0359) and loses at 200·dim (+0.0365, 9/12 vs +0.0423,
11/12): the one-generation relay is the *right* configuration exactly
where the ratchet has not saturated.

### 50.2 Blocking is free — the block length matters only through the hand-off

The cold specs are almost independent of the block length: mean AOCC
0.3140 / 0.3153 / 0.3159 / 0.3176 at 100·dim across 50, 16–20, 8–12 and
4–10 blocks (a spread of 0.0036), and 0.3912…0.3974 at 200·dim (0.0062).
Over a 12× range in block count the switching transient costs **under
0.006 AOCC**.

That corrects §46.3's reading.  The flat warm curve there was taken as
"two effects cancelling" — a sharing gain rising with hand-off count
against a transient cost rising with it too.  There is no transient cost
worth the name.  The warm curve is flat at 200·dim because the *gain*
is saturated, and it is steep at 100·dim because the gain is not.  Block
length is not a cost/benefit trade-off in this portfolio; it is a dial on
one quantity, how often the arms exchange.

### 50.3 The short-block stall is caused by the hand-off, not by the block

§46.2 reported that at 200–300·dim, *d* = 2 the `n_blocks=50` default
collapses in 6–8 of 60 cells (AOCC 0.17–0.30 where CMA-ES alone reaches
0.55–0.82) and left the mechanism unmeasured — the *hypothesis* was both
arms re-seeded from a tight top-K into one basin every two generations.
(Note the stall is an AOCC collapse, not an unspent budget: no run in
any file of this campaign finished short of 98 % of its budget.)  The
cold twins settle it.  Cells more than 0.25 AOCC below `CMAES_alone`, of
60 per dimension, at 200·dim:

| block | warm, *d*=2 | cold, *d*=2 |
|---|---|---|
| `n_blocks=50` (8–12 evals) | **7** | 1 |
| 25 | **4** | 0 |
| `auto` (24) | 2 | 1 |
| 50 | 0 | 0 |

The warm count reproduces §46.2's 6–8 exactly, and the cold twin at the
same block length does not stall.  So the collapse **requires the
hand-off**, and it grows as the block shortens.  §46.2's mechanism
survives its first real test.  (At *d* = 5 the pattern inverts — the
*cold* specs collapse in 4–11 cells, the warm ones in 0–1 — which is
just cold portfolios being bad at *d* = 5, where they lose −0.094 to
CMA-ES.)

Note what this costs at 200·dim: seven stalled cells is why the
50-block default does not win there despite having the most hand-offs.
The ratchet gain is real at every length; at *d* = 2 / 200·dim it is
eaten by the pathology its own frequency creates.

### 50.4 Consequences

* **Regime table, low budget:** at ≤ 100·dim the one-generation relay
  (`n_blocks=50`) is the better block, not `block_evals="auto"` — §46.4's
  recommendation was right for 200·dim and wrong for 100·dim.  The row
  should carry the block length, and it should depend on the budget.
* **A σ floor on the hand-off is the building block this points at.**
  The warm start sets σ to the *spread of the seed cloud* (clipped only
  above, by the cold σ₀).  When the top-K have collapsed into one basin
  that spread is tiny, and the receiving arm restarts pinned to a point —
  exactly the stall.  A floor (relative to the arm's own current σ, or to
  σ₀) would keep the ratchet's frequency without its pathology.  This is
  *not* §48.2's rejected diversification: the seeds stay the top-K and the
  mean stays the incumbent; only the step size is prevented from
  collapsing.  Falsifiable the same way, and it should show up precisely
  in the seven stalled cells.
* Together with §49.4 (keep the arm's own **C** instead of resetting it)
  there are now two untested pieces of the hand-off's payload, both
  predicted by the same reading: relocate the receiving arm, but stop
  destroying what it had.

## 51. The payload is location and scale: all four shape payloads are now measured, and the identity wins

§49.4 predicted that keeping CMA-ES's own adapted **C** across a hand-off
would pay, most at the higher budget where more adaptation is discarded;
§50.4 predicted a σ floor would keep the ratchet's frequency without its
stall.  Both landed as opt-in kwargs (6ef2498) and were screened on the
full roster at both budgets, at the short block (`n_blocks=50`, where the
stall lives) and at `auto`.  Raw:
`results/2026-09-14/payload_{bm100,bm200}.json` — the three baselines are
**cell-for-cell identical** to the §50 `ratchet_*` files, so the added
specs perturbed no stream.

### 51.1 Keeping the arm's own C: falsified, in the informative direction

`warm_start_keep_cov` skips `_reset_covariance`, keeping B/D/C
bit-identical while m and σ are refitted; the evolution paths are zeroed
(the mean has just jumped).  Against its own baseline:

| | 100·dim | *d*=2 | *d*=5 | 200·dim | *d*=2 | *d*=5 |
|---|---|---|---|---|---|---|
| keep C, `nb50` | −0.0175, 3/12 | −0.0033 | −0.0316 [−0.052, −0.012] | −0.0253, 2/12 | −0.0012 | **−0.0494 [−0.088, −0.011], 1/12** |
| keep C, `auto` | −0.0053, 5/12 | −0.0068 | −0.0038 | −0.0173, 5/12 | −0.0205 | −0.0141 |

Negative at every budget and dimension, *worse* at the higher budget, and
worst at *d* = 5 / 200·dim — exactly the cell where §49.4 said it should
pay most.  It is also the **only spec in the campaign that stalls at
*d* = 5** (2–3 cells of 60), and at *d* = 2 / 200·dim it stalls in 5 of
60 where its `auto` baseline stalls in 2.  The prediction is refuted as
cleanly as it could be.

The reading that survives: after a hand-off the arm's mean sits on the
*other* arm's incumbent, and its own C describes the curvature it
measured around its *old* mean.  Carrying that shape to a new location is
not conservation, it is a stale prior — and a confidently anisotropic one,
which is why it can stall.  The identity is the honest prior at a place
the arm has never sampled, and CMA-ES re-adapts within a few generations
because σ is already fitted to the cloud.

With this the payload table is complete — four ways to set **C** at a
hand-off, all measured on the same roster:

| payload for C | verdict |
|---|---|
| **identity (reset)** — shipped | **best; the baseline nothing beats** |
| archive shape (`archive_cov`, §48.3) | −0.053 / −0.069 at *d* = 5, cost grows with budget |
| shrunk archive shape (§49) | parity at 100·dim, −0.043 at 200·dim; family bracketed |
| the arm's own C (§51.1) | negative everywhere, worst where predicted to win |

**Shape is not transferable across a relocation — neither the other
arm's nor the arm's own.**  The hand-off carries **m** and **σ**, and
that is the whole payload.  Combined with §48.2 (the top-k crowd, not a
spread) the building block is now fully characterised: *put the receiving
arm on the incumbent, at the scale of the cloud around it, with no memory
of shape.*

### 51.2 The σ floor: a targeted repair, not a default

`warm_start_sigma_floor=f` sets `σ = clip(spread, f·σ_self, σ₀_cold)` —
the floor is relative to the arm's **own current σ**, so it bounds the
*rate* of collapse per hand-off (σ may still fall as `f^k` over k
hand-offs) and binds only on an already-collapsed cloud.  `f = 0.5` was
picked on a 3-seed pilot at 200·dim from {0.25, 0.5, 0.8} before the
roster was spent.

| | overall | *d*=2 | *d*=5 | stalls *d*=2 / 60 |
|---|---|---|---|---|
| floor, `nb50`, 100·dim | +0.0048, 7/12 | +0.0004 | +0.0091 | 0 → 0 |
| floor, `nb50`, 200·dim | +0.0136, 10/12 | **+0.0444 [+0.014, +0.075], 10/12** | −0.0171 [−0.043, +0.009], 6/12 | **7 → 2** |
| floor, `auto`, 100·dim | −0.0206 [−0.031, −0.010], 1/12 | −0.0274 | −0.0138 | 0 → 0 |
| floor, `auto`, 200·dim | −0.0234, 3/12 | −0.0282 | −0.0186 [−0.030, −0.007] | 2 → 2 |

Where the pathology exists — short block, *d* = 2, 200·dim — the floor
removes **five of seven** stalled cells and is not paid for: +0.044 at
*d* = 2 with the CI clear of zero, the best *d* = 2 mean in the run
(0.5998).  Where it does not exist, the floor is a straight loss: at the
`auto` block it costs −0.02 at both budgets and fixes nothing, and at
*d* = 5 / 200·dim it is parity-to-negative.  A rate bound that is right
for a 8-evaluation block is too tight for a 24–48-evaluation one, which
is the obvious next form if it is pursued (scale `f` with the block
length), but the honest verdict today is: **not a default**; a candidate
row for a regime table keyed on short block × low *d*.

### 51.3 What this does to the configuration ranking

At 100·dim the short block plus the floor is the strongest configuration
measured in this project:

| spec | Δ vs `CMAES_alone`, 100·dim |
|---|---|
| `sigfloor_nb50` | **+0.0606 [+0.042, +0.079], 12/12** |
| `Blocks_uniform_cj_warm2` (`nb50`, §46) | +0.0558 [+0.037, +0.074], 12/12 |
| `Blocks_uniform_cj_warm2_auto` | +0.0359 [+0.018, +0.054], 10/12 |

and `sigfloor_nb50 − warm2_auto` = +0.0247 [+0.001, +0.048], 9/12 —
accepted.  But note where that margin comes from: the floor itself adds
only +0.005 (7/12) over `nb50`, and §50 already showed `nb50 − auto` is
+0.020 at this budget.  **The gain is the block length, not the floor.**
At 200·dim `sigfloor_nb50` reaches +0.0502 (9/12) against `auto`'s
+0.0423 (11/12), and the two are level head-to-head (+0.0079, 6/12).

So the configuration story stands where §50 left it — block length by
budget — with the floor as a repair for the *d* = 2 corner of the short
block, and the payload question now closed.

## 52. The BBOB function axis: first look — mixtures were hiding the dispersion the portfolio lives on

`planning/DESIGN_suite_2026-09-14.md` step 1 landed (6282807):
`IOHBatterySpec.fids` makes the 24 standard BBOB functions a battery
dimension, `fid` joins both seed payloads and the run record, and the
screen folds cells on `(seed, fid, dim, instance)` and prints paired
deltas per COCO class.  Backward compatibility is pinned two ways — two
pinned seed constants, and `from=…ratchet_bm200.json` producing
byte-identical output before and after — so every number of §44–§51
stands unchanged.

The BBOB path had **never been exercised**: `ioh.get_problem(fid, inst,
dim, "BBOB")` fails on the current binding, which wants
`ioh.ProblemClass.BBOB`.  Fixed in the worker.

**The run below is 3 seeds.  Nothing here is evidence by the rule; it is
orientation for where to spend the roster.**  432 cells (3 seeds × 24
fids × 2 dims × 3 instances), 3 specs, 200·dim, 0 errors:
`results/2026-09-14/bbob_smoke.json`.

### 52.1 Plain functions are a much harder instrument than the mixtures

Mean AOCC at the same budget: **0.27–0.31 on plain BBOB against
0.49–0.53 on MA-BBOB**.  MA-BBOB instances are affine combinations of
two BBOB functions, so each instance averages two landscapes — and the
average of a landscape CMA-ES owns with one it cannot solve is a
landscape on which every method scores middling.  That compression is
almost certainly why the battery of record read "level" for so long
(§27): the instrument was averaging away the very dispersion a portfolio
exists to exploit.

### 52.2 The portfolio is positive in all five classes — pooled

`Blocks_uniform_cj_warm2_auto` − `CMAES_alone`, per COCO class:

| class | pooled | *d* = 2 | *d* = 5 |
|---|---|---|---|
| separable (f1–5) | +0.019 | +0.058 | −0.021 |
| low conditioning (f6–9) | +0.032 | +0.081 | −0.017 |
| high conditioning (f10–14) | +0.060 | +0.104 | +0.016 |
| multimodal, global structure (f15–19) | +0.027 | +0.068 | −0.014 |
| multimodal, weak structure (f20–24) | +0.063 | +0.115 | +0.012 |

Pooled +0.0405, 0 of 5 classes negative.  The ordering is the one the
thesis predicts — most where a single covariance model is least
sufficient (high conditioning, weak global structure), least on separable
problems — but **the per-dimension split is a sign flip against
MA-BBOB**, where *d* = 5 carried the gain (+0.054) and *d* = 2 was the
smaller half (+0.031).  Here *d* = 2 is +0.085 and *d* = 5 is −0.004.

### 52.3 One function carries the *d* = 5 sign

Per (fid, dim) the portfolio wins **34 of 48** cells.  The losses are not
spread: they concentrate on **f5** (the linear slope — −0.030 at *d* = 2,
**−0.239** at *d* = 5) and **f7** (step ellipsoid — −0.065 / −0.095).

| pooled delta | all 24 | drop f5 | drop f5, f7 |
|---|---|---|---|
| overall | +0.0405 | +0.0481 | +0.0539 |
| *d* = 2 | +0.0852 | +0.0902 | +0.0973 |
| *d* = 5 | **−0.0042** | **+0.0060** | **+0.0106** |

A single function flips the sign of a whole dimension.  That is the
function axis earning its keep on its first run: on the mixtures this
would have been invisible, folded into an average.

f5 is the honest explanation rather than an excuse to drop it: a linear
slope has no interior optimum, CMA-ES walks to the boundary and reaches
AOCC 0.87–0.91, and every evaluation the scheduler hands to jSO is spent
on a problem that is already solved.  f7's plateaus are the case where a
ranking sees ties.  Both are cells where CMA-ES alone does *well*.

### 52.4 The pattern: the portfolio wins where the single arm struggles

Bucketing each cell's delta by CMA-ES's own AOCC in that cell:

| CMA-ES alone | < .15 | .15–.30 | .30–.45 | .45–.60 | .60–.75 | > .75 |
|---|---|---|---|---|---|---|
| portfolio delta | +0.063 | +0.059 | +0.045 | +0.024 | −0.037 | **−0.121** |
| cells | 151 | 160 | 44 | 31 | 16 | 30 |

Bucketing by the baseline's own value invites regression to the mean, so
this table alone would not be trusted — but the per-function cut of
§52.3, which is free of that trap (a fid is a property of the problem,
not of a run), says the same thing: the losses sit exactly on the
functions CMA-ES handles well.

**Consequence.**  This is the strongest argument so far for the probe
that §45.2 deferred.  On the mixtures the dispersion of per-cell outcomes
was small, so a gate had little to win (~+0.03 in one regime).  On plain
functions the same portfolio ranges from **+0.22 to −0.24** depending on
the landscape.  A gate that can tell "CMA-ES alone will handle this" from
"it will not" is worth an order of magnitude more here than the mixtures
suggested — and §52.4 says the signal it needs is observable early: how
fast the first arm is making progress.

### 52.5 What to spend the roster on

1. The 12-seed decision run on the fid axis at 100 and 200·dim, with the
   configurations §50–§51 left standing (`nb50`, `auto`, and the σ floor
   at the short block).  Cost ≈ 100 min; that is the re-test Harald's
   2026-09-13 gate asks for before any default changes.
2. Only then the probe, now with a target worth the complexity.
3. Step 3 of the suite design (the missing synthetic shapes) gains a
   concrete motivation from this run: f7's plateaus and f20–f24's weak
   structure are where the extremes live, and `step_ellipsoid`,
   `lunacek_bi_rastrigin` and `gallagher` are exactly those shapes as
   parametrised families where the knob can be swept.

## 53. The decision run on the function axis: the low-budget claim survives; at 200·dim the arm order changes and the portfolio only ties the winner

The re-test Harald's 2026-09-13 gate asks for, on the axis §52 built.
Raw: `results/2026-09-14/bbobdec_{bm100,bm200}.json` (c5b7d8c) — 6 seeds
`42 7 1234 2025 3 11`, all 24 fids, dims 2 and 5, instances 0–1, 5 specs,
576 cells per budget, 2880 rows each, **0 errors**.  Six seeds means
**t(5) = 2.571**, not the roster's 2.201; the cell count (576 vs the
MA-BBOB cube's 60) is what buys the precision back.

### 53.1 Against each arm

| | vs `CMAES_alone` | vs `JSO_alone` |
|---|---|---|
| **100·dim** | | |
| `warm2_auto` | **+0.0186 [+0.008, +0.029], 6/6** | **+0.0257 [+0.013, +0.038], 6/6** |
| `sigfloor_nb50` | **+0.0169 [+0.010, +0.024], 6/6** | **+0.0240 [+0.018, +0.030], 6/6** |
| `warm2` (`nb50`) | +0.0121 [+0.003, +0.021], 5/6 | +0.0192 [+0.010, +0.028], 6/6 |
| `JSO_alone` | −0.0071 [−0.018, +0.003], 1/6 | — |
| **200·dim** | | |
| `warm2_auto` | +0.0336 [+0.016, +0.051], 6/6 | +0.0074 [−0.010, +0.025], 4/6 |
| `sigfloor_nb50` | +0.0356 [+0.017, +0.055], 6/6 | +0.0094 [−0.004, +0.022], 5/6 |
| `warm2` (`nb50`) | +0.0305 [+0.005, +0.056], 5/6 | +0.0042 [−0.014, +0.022], 4/6 |
| `JSO_alone` | **+0.0263 [+0.012, +0.041], 6/6** | — |

**At 100·dim the sharing portfolio is accepted against both arms** — the
central claim of §46 survives the move from mixtures to the 24 standard
functions, on a battery where the absolute AOCC is 0.20 rather than 0.39.

**At 200·dim it is not**, and the reason is the interesting part: *the
better arm changes*.  jSO is worse than CMA-ES at 100·dim (−0.007) and
clearly better at 200·dim (+0.026, 6/6).  On MA-BBOB CMA-ES was the arm
to beat at every budget, so this could not appear.  The portfolio tracks
whichever arm is ahead — beating the loser by +0.034 and matching the
winner — but the acceptance rule asks it to beat the winner, and it does
not.

### 53.2 Three bars, and where the remaining value is

The rule's bar is *ex post*: it asks the portfolio to beat the arm that
turned out best, which no user could have chosen in advance.  Two other
bars bracket it.  Per cell, CMA-ES is the better arm 277 times of 576 at
100·dim and 262 of 576 at 200·dim — and **in all 24 functions the better
arm is not the same in every cell**, so "pick the right arm" is not even
a per-function decision.

`Blocks_uniform_cj_warm2_auto` against:

| bar | 100·dim | 200·dim |
|---|---|---|
| the **average** arm (what you get without knowledge) | **+0.0221 [+0.012, +0.033], 6/6** | **+0.0205 [+0.005, +0.037], 5/6** |
| the **pooled-best** arm (ex post, but one choice) | +0.0186, 6/6 | +0.0074, 4/6 |
| the **per-cell oracle** arm (perfect selection) | −0.0150 [−0.025, −0.006], 0/6 | −0.0386 [−0.053, −0.025], 0/6 |

Against ignorance the portfolio is worth a steady **+0.02 at both
budgets**.  Against perfect selection it is behind by 0.015 and 0.039 —
and at 200·dim that gap is *five times* the portfolio's own margin over
the pooled-best arm.  **The remaining value in this system is in
selection, not in more sharing.**  That is the strongest case yet for the
probe deferred in §45.2, and it now has a measured target rather than a
hoped-for one.

### 53.3 Per class, and f5 again

Classes with a negative mean delta against CMA-ES: 1 of 5 at 100·dim
(separable, −0.023 for `auto`), **0 of 5** at 200·dim for both `auto` and
`sigfloor_nb50`.  The negative class is the one f5 lives in.  Dropping
the linear slope:

| `warm2_auto` − CMA-ES | all 24 | drop f5 | drop f5, f7 |
|---|---|---|---|
| 100·dim, *d* = 5 | +0.0020 | +0.0159 | +0.0162 |
| 200·dim, *d* = 5 | **−0.0026** | **+0.0059** | +0.0092 |

At 200·dim the *d* = 5 sign flip of §52.3 reproduces at 6 seeds.  Per
(fid, dim) `warm2_auto` wins 38 of 48 at 100·dim and 32 of 48 at
200·dim; `sigfloor_nb50` 41 and 36.  The largest losses are f5 at both
budgets and both dimensions, then f17 and f7.  The largest wins are f22
(Gallagher, +0.30 at 200·dim), f6 (attractive sector) and f20 (Schwefel)
— the weak-structure and asymmetric shapes.

Note the σ floor is not uniformly kind to f5: it makes that function's
*d* = 5 loss worse (−0.477 vs `auto`'s −0.317 at 100·dim) while winning
more cells overall (41 of 48).  A guard that keeps a step size alive is
exactly wrong on a landscape with no interior optimum.

### 53.4 What this settles and what it opens

* **Settled:** the low-budget sharing claim is not an artefact of the
  MA-BBOB mixtures.  It holds on the standard functions, against both
  arms, at 100·dim.
* **Qualified:** at 200·dim the portfolio ties the best arm rather than
  beating it, because which arm is best is itself budget-dependent.  The
  honest statement of the product is *robustness without foreknowledge*,
  and against that bar (+0.02 vs the average arm) it delivers at both
  budgets.
* **Opened:** the selection headroom (−0.015 / −0.039 against the
  per-cell oracle) is now the largest measured quantity on the table, and
  §52.4 says the signal is observable early.  The probe moves from
  "deferred, ceiling ~+0.03 in one regime" to the main line of work.
* Default decisions stay where Harald left them on 2026-09-13: this run
  is the gate's first half (standard functions); the synthetic families
  with swept knobs (`DESIGN_suite_2026-09-14.md` step 3) are the second.

## 54. Re-baseline on the post-audit code (2026-09-26)

Reference files for every comparison from here on; nothing before
2026-09-25 is comparable (TODO history, audit PRs #320–#340).  Measured
on GitHub runners (`rebaseline.yml`, run 36228301268, commit 0184fad,
28 shards, none failed), sync evaluation, keyed RNG streams, no
wall-clock limits, the 12-seed roster.  Reference files: GitHub release
`rebaseline-2026-09-26-run36228301268` (asset
`rebaseline-2026-09-26-run36228301268.tar.gz`; unpack with
`scripts/rebaseline.py fetch <tag>`).  In git: the manifest and
`planning/results/2026-09-26/SUMMARY.json` (every number below).

| Suite | 12-seed mean |
|---|---|
| composite quick | 0.4029 (per seed 0.349 … 0.487) |
| composite standard | 0.4275 (per seed 0.419 … 0.440) |
| IOH quick (mean AOCC, all specs) | 0.3592 |
| IOH standard (mean AOCC, all specs) | 0.4999 |

IOH per spec, quick / standard: `Blocks_warm_CMAES_JSO` 0.4187 / 0.6738,
`RegimeGate_oracle` 0.4187 / 0.6771, `RoundRobin_CMAES` 0.3893 / 0.6673,
`Baseline_SciPyDE` 0.3366 / 0.4228, `RoundRobin_Random` 0.3353 / 0.3797,
`Baseline_Random` 0.3285 / 0.2934, `Baseline_SciPyAnneal` 0.2871 / 0.3855.
Families: free 2160 rows, constrained 1152 rows (re-analyse with
`family_screen.py from=...`).

The code measured predates #344–#346: those change only the external
baselines, failure handling (bit-identical on problems without
failures, verified) and a virtual-clock mode (bit-identical outside it),
so these references stay valid for the default paths.

## 55. The real async loop pulls when free: candidate staleness, legacy vs pull (2026-09-26)

`evaluation.async_policy` (new; default `pull`, `legacy` keeps the old
loop).  Under `pull` the threaded / processes / dask loop without
`evaluation.sync` sets `request_cap` to the free workers within the budget
before `execute()` (skipped at 0), keeps `jobs_per_client = 1`, returns any
surplus to the heuristics' queues, asks again at once after a harvest, and
while every worker is busy blocks on the pool until an evaluation completes
(`LocalPool.wait` / `distributed.wait(FIRST_COMPLETED)`, at most 50 ms)
instead of polling every millisecond.  `evaluation.sync` and the virtual
clock are untouched: 84 seeded traces (RoundRobin_CMAES,
Blocks_warm_CMAES_JSO, Rewarding, UCB, a two-phase Phased, Thompson with
LocalPenaltySearch, UCB with the L-BFGS-B bridge; Rosenbrock d = 3,
Rastrigin d = 2 and constrained Rosenbrock d = 3; 120 evaluations; sync
with 2 and 4 workers, virtual async q = 4 log-normal, virtual sync q = 4)
hash identically on master and the branch.

**Staleness** of a candidate: results delivered to the main loop
(`len(strategy.results)`) between its creation (`Point.__init__`) and the
start of its evaluation on a worker.  Lag in the event bus (results
harvested but not yet seen by the heuristics' handlers) is not counted.
Threaded, Rosenbrock d = 4, 300 evaluations, log-normal sleeps (sigma
0.5), 3 seeds, mean over seeds; "peak" is the most evaluations submitted
and not yet harvested at once; per-heuristic columns are the mean
staleness of that heuristic's points (Random's points dominate the
point-weighted mean, and a Random point does not depend on the archive, so
its staleness is harmless).  Instrument: a scratch script (monkeypatched
`Point.__init__`, the objective records at call start), not in the
repository.  The machine was loaded (load average ~8.7 on 16 threads), so
wall times are indicative.

q = 4, mean 5 ms:

| Strategy | policy | mean | p90 | max | peak | per heuristic | wall s |
|---|---|---|---|---|---|---|---|
| RoundRobin_CMAES | legacy | 3.1 | 6.0 | 7.7 | 12.7 | CMA-ES 3.1 | 0.401 |
| RoundRobin_CMAES | pull | 4.5 | 8.0 | 9.7 | 4 | CMA-ES 4.5 | 0.413 |
| Blocks_warm_CMAES_JSO | legacy | 6.1 | 12.7 | 21.3 | 25.3 | CMA-ES 4.5, jSO 7.8 | 0.409 |
| Blocks_warm_CMAES_JSO | pull | 3.0 | 6.0 | 11.0 | 4 | CMA-ES 3.5, jSO 2.5 | 0.461 |
| RoundRobin (Random + Nearby, size 10) | legacy | 130.5 | 234.9 | 263.3 | 266.3 | Nearby 82.4, Random 132.1 | 0.387 |
| RoundRobin (Random + Nearby, size 10) | pull | 10.7 | 20.0 | 33.0 | 4 | Nearby 3.1, Random 11.6 | 0.422 |
| Rewarding (Random + Nearby) | legacy | 38.7 | 54.3 | 61.7 | 45.7 | Nearby 26.4, Random 39.7 | 0.386 |
| Rewarding (Random + Nearby) | pull | 10.4 | 20.0 | 33.7 | 4 | Nearby 2.6, Random 11.4 | 0.414 |
| UCB (Random + Nearby) | legacy | 28.0 | 38.7 | 45.0 | 24.0 | Nearby 18.3, Random 29.0 | 0.401 |
| UCB (Random + Nearby) | pull | 10.1 | 19.0 | 35.7 | 4 | Nearby 2.2, Random 10.9 | 0.412 |

q = 8, mean 10 ms (mean; Nearby or CMA-ES / jSO; wall s), legacy → pull:
RoundRobin_CMAES 1.2 → 1.8 (0.456 → 0.460); Blocks 4.0 → 2.4, CMA-ES
2.2 → 2.5, jSO 5.8 → 2.2 (0.432 → 0.455); RoundRobin Random + Nearby
126.1 → 11.2, Nearby 74.7 → 3.0 (0.391 → 0.405); Rewarding 52.7 → 11.4,
Nearby 42.0 → 2.9 (0.391 → 0.429); UCB 43.7 → 10.9, Nearby 35.0 → 2.2
(0.391 → 0.409).  Peak in flight 14-266 → 8.

q = 4, a 0 ms objective (no sleep; the loop's own overhead): legacy →
pull mean 0.2 → 5.4 (RoundRobin_CMAES), 5.5 → 3.8 (Blocks), 13.4 → 12.1,
16.6 → 12.3, 12.6 → 12.0 (the Random + Nearby three; Nearby 8.7-15.0 →
3.8-4.3); wall 0.06-0.17 s either way (legacy 0.062-0.165, pull
0.069-0.160, mixed signs).

* Pull never has more in flight than workers; legacy queued up to 266
  evaluations on 4 workers (RoundRobin `size = 10` asked on every ~1 ms
  pass: the whole budget sat in the pool).
* **The staleness gains hold for evaluations of a few ms or more.**  At
  0 ms nothing waits in a pool queue under either policy, so there is
  little to gain (and a lone CMA-ES is staler under pull, below).
* For the heuristics whose points depend on the archive the gain is large:
  Nearby 18-82 → 2-3 at 5-10 ms, jSO 6-8 → 2-2.5.  CMA-ES alone is slightly
  staler under pull (+0.6 to +1.4 at 5-10 ms, +5 at 0 ms): the result that
  frees a worker is delivered before the next queued member of the
  generation starts, so each counts one more delivered result than under
  legacy, where the pool starts the next queued task before the harvest.
* The ~11 left for Random under pull is its own output queue (a 20-point
  prefill, `heuristic.capacity`), not the loop (TODO).
* Wall time: with the pool wait, pull is within 0-13 % of legacy at
  5-10 ms (0.41-0.46 s vs 0.39-0.46 s) and on par at 0 ms.  Before the
  pool wait (1 ms polling) it was 0.44-0.48 s vs 0.38-0.40 s at 5 ms.

## 56. FP-exact re-baseline (2026-09-26, after the FP pin)

§54's references mixed two GitHub runner floating-point classes (AVX2 vs
AVX-512 hosts; last-bit BLAS/libm differences amplified up to 0.08 AOCC per
cell).  #359 pins OpenBLAS kernels (Haswell) and numpy's SIMD dispatch
(AVX-512 targets off) and records `fp_env_id`.  `fp-check.yml`
(run 36265628638): 8 jobs on AMD EPYC 7763 (AVX2) and 9V45 (AVX-512) all
bit-identical, digest `9bdd00b7ac00` — the same digest as Harald's laptop.

Full re-baseline on master f7d7e0d (run 36265786623, 29 jobs, none failed
or missing, one `fp_env_id` 80ee2a0090c4), release `rebaseline-2026-09-26-run36265786623`, summary
`planning/results/2026-09-26-run36265786623/SUMMARY.json`.  12 seeds each.

| Suite | 12-seed mean |
|---|---|
| composite quick | 0.4029 (0.349 … 0.487) |
| composite standard | 0.4257 (0.417 … 0.441) |
| IOH quick (mean AOCC, all specs) | 0.3590 |
| IOH standard (all specs) | 0.5005 |
| IOH standard + external baselines (all 14 specs) | 0.5290 |
| families free / constrained / shapes / failure | 0.3627 / 0.4509 / 0.3546 / 0.3492 |

IOH standard with the external baselines (the cheap track, 500·d):

| Spec | AOCC |
|---|---|
| **Baseline_Optuna_CmaEs** | **0.7052** |
| RegimeGate_oracle | 0.6771 |
| Blocks_warm_CMAES_JSO | 0.6738 |
| RoundRobin_CMAES | 0.6674 |
| Baseline_pycma_BIPOP | 0.6671 |
| Baseline_pycma_IPOP | 0.6546 |
| Baseline_NGOpt | 0.5961 |
| Baseline_NG_CMA | 0.4556 |
| Baseline_SciPyDE | 0.4228 |
| Baseline_Optuna_TPE | 0.4165 |
| Baseline_NG_TwoPointsDE | 0.4066 |
| Baseline_SciPyAnneal | 0.3896 |
| RoundRobin_Random | 0.3797 |
| Baseline_Random | 0.2934 |

Families (4 specs; per-preset best): free CMAES_alone 0.401, constrained
CMAES_alone 0.487, shapes LSHADE_alone 0.371, failure LSHADE_alone 0.405;
`Blocks_uniform_cj_warm2` is last on all four (0.281 on failure).

Reading:

* **On the cheap track the incumbents are at or above us.** Optuna's
  CMA-ES sampler (no restarts, start point at a random box point) leads
  by +0.028 over our best spec (`RegimeGate_oracle`, itself an oracle);
  pycma BIPOP ties `RoundRobin_CMAES`.  The single-seed preview in the
  PR #357 smoke run had BIPOP ahead; with 12 seeds it is Optuna CmaEs.  Not
  yet paired per seed: the next step is a paired comparison
  (`paired_seed_stats`) of the headline specs against Optuna CmaEs and
  BIPOP, per BBOB class, before anything is concluded.
* **The sharing portfolio loses on the new families at 500·d** —
  consistent with §46/§53 (sharing pays at low budget, parity or worse at
  ≥500·d), most strongly under failures (0.281 vs 0.405).  The failure
  families are the roadmap's §4 D ground.
* Composite quick is unchanged from §54 (0.4029); composite standard
  moved 0.4275 → 0.4257 — the §54 run had half its shards on the other FP
  class.

## 57. Optuna's CMA-ES lead on the cheap track is one problem: d = 5 MA-BBOB instance 2, where it finds the global basin 12 times of 12

§56 has `Baseline_Optuna_CmaEs` at the top of IOH standard + external
baselines (500·d): 0.7052 against 0.6771 (`RegimeGate_oracle`), 0.6738
(`Blocks_warm_CMAES_JSO`), 0.6674 (`RoundRobin_CMAES`), 0.6671 (pycma
BIPOP), 0.6546 (pycma IPOP).  This entry pairs those numbers.  Data:
`ref_ioh_standard_external.json` of release
`rebaseline-2026-09-26-run36265786623` (12 seeds × 10 cells × 14 specs,
0 errors; `scripts/rebaseline.py fetch`).  Nothing was re-run.

Method: per seed, the mean AOCC over the cells in scope, differenced
between two specs; t-CI95 over the 12 seeds (t(11) = 2.201), a
percentile bootstrap over seeds (5000 draws) as a check, wins = seeds
where the first spec is ahead, two-sided paired t p.  **The instrument is
small:** the battery is MA-BBOB instances 0–4 at *d* = 2 and 5, and the
instances are fixed across seeds — the seed moves only the optimizer.
The 12 seeds are 12 replicate runs on the same 10 problems, and instance
*i* at *d* = 2 and at *d* = 5 is the same mixture (same BBOB functions
and weights).  There is one budget (500·d) and no noise.

### 57.1 Pooled: Optuna CmaEs is ahead of our CMA-ES specs, pycma is not

Δ = row − column, AOCC:

| | vs Optuna CmaEs | vs pycma BIPOP | vs pycma IPOP |
|---|---|---|---|
| `Blocks_warm_CMAES_JSO` | **−0.031 [−0.059, −0.004], 4/12, p = 0.030** | +0.007 [−0.016, +0.030], 7/12 | +0.019 [−0.013, +0.052], 8/12 |
| `RoundRobin_CMAES` | **−0.038 [−0.069, −0.007], 3/12, p = 0.022** | +0.000 [−0.042, +0.042], 7/12 | +0.013 [−0.030, +0.056], 6/12 |
| `RegimeGate_oracle` | −0.028 [−0.061, +0.005], 3/12, p = 0.088 | +0.010 [−0.020, +0.040], 7/12 | +0.023 [−0.014, +0.060], 8/12 |

(Bootstrap CIs agree in sign: Optuna rows [−0.055, −0.007], [−0.065,
−0.012], [−0.057, −0.000].)  Against both pycma restart variants every
one of our specs is level.  Optuna's lead is real at the 12-seed level
for the two specs that ship, and borderline for the oracle gate.

### 57.2 Per dimension: all of it is *d* = 5

| | *d* = 2 | *d* = 5 |
|---|---|---|
| `Blocks_warm_CMAES_JSO` − Optuna | −0.003 [−0.037, +0.032], 6/12 | **−0.060 [−0.100, −0.021], 2/12, p = 0.007** |
| `RoundRobin_CMAES` − Optuna | −0.002 [−0.048, +0.045], 7/12 | **−0.074 [−0.124, −0.024], 3/12, p = 0.008** |
| `RegimeGate_oracle` − Optuna | −0.014 [−0.060, +0.031], 5/12 | −0.042 [−0.085, +0.001], 3/12, p = 0.054 |
| `RegimeGate_oracle` − BIPOP | −0.032 [−0.071, +0.006], 4/12 | +0.053 [−0.003, +0.108], 9/12 |

Mean AOCC per dimension: at *d* = 2 pycma BIPOP is best (0.743; Optuna
0.724, `RoundRobin_CMAES` 0.723), at *d* = 5 Optuna is best by a margin
(0.686; `RegimeGate_oracle` 0.644, `Blocks` 0.626, `RoundRobin` 0.612,
IPOP 0.597, BIPOP 0.591).

### 57.3 Per problem: one cell carries the sign

Mean AOCC per cell (12 seeds):

| cell (*d*, inst) | mixture (top BBOB weights) | Blocks | RoundRobin | Gate | **Optuna** | BIPOP | IPOP |
|---|---|---|---|---|---|---|---|
| (2, 0) | f23 f5 f21 f17 | 0.674 | 0.696 | 0.695 | 0.721 | 0.731 | 0.639 |
| (2, 1) | f23 f13 f16 f5 | 0.468 | 0.466 | 0.444 | 0.443 | 0.448 | 0.449 |
| (2, 2) | f22 f24 f11 f14 | 0.803 | 0.809 | 0.742 | 0.736 | 0.813 | 0.811 |
| (2, 3) | f1 f3 | 0.904 | 0.873 | 0.879 | 0.870 | 0.875 | 0.836 |
| (2, 4) | f7 f18 f17 f20 | 0.761 | 0.769 | 0.792 | 0.852 | 0.846 | 0.824 |
| (5, 0) | f23 f5 f21 f17 | 0.568 | 0.596 | 0.641 | 0.570 | 0.533 | 0.512 |
| (5, 1) | f23 f13 f16 f5 | 0.419 | 0.392 | 0.418 | 0.442 | 0.407 | 0.409 |
| **(5, 2)** | **f22 f24 f11 f14** | 0.533 | 0.423 | 0.539 | **0.745** | 0.404 | 0.496 |
| (5, 3) | f1 f3 | 0.833 | 0.824 | 0.860 | 0.853 | 0.846 | 0.840 |
| (5, 4) | f7 f18 f17 f20 | 0.774 | 0.825 | 0.762 | 0.821 | 0.767 | 0.731 |

Optuna − `RoundRobin_CMAES` on (5, 2): **+0.322 [+0.173, +0.471], 10/12**;
− `Blocks`: +0.212, 11/12; − BIPOP: +0.341, 11/12; − IPOP: +0.249, 9/12.
The only other cell where Optuna is consistently ahead of all three of
ours is (5, 1) (+0.022…+0.050, 9–11/12, CIs at or just above zero,
small).  **Dropping (5, 2)** —
one of ten cells — every pooled comparison is level:

| without (5, 2) | Δ vs Optuna |
|---|---|
| `Blocks_warm_CMAES_JSO` | −0.011 [−0.042, +0.019], 6/12 |
| `RoundRobin_CMAES` | −0.006 [−0.036, +0.024], 6/12 |
| `RegimeGate_oracle` | −0.008 [−0.040, +0.023], 4/12 |
| pycma BIPOP | −0.005 [−0.032, +0.023], 6/12 |

What happens on (5, 2) (global optimum interior, max |x\*_i| = 3.39;
Gallagher 21 peaks + Lunacek bi-Rastrigin + discus + different powers):
Runs reaching precision 1e−1 by evaluation 911, and 1e−8 by the end
(2500):

| | 1e−1 by 911 | 1e−8 by 2500 |
|---|---|---|
| **Optuna CmaEs** | **12 / 12** | **12 / 12** (median eval 1509) |
| `RoundRobin_CMAES` | 6 / 12 | 3 / 12 |
| `Blocks_warm_CMAES_JSO` | 11 / 12 | 4 / 12 |
| pycma BIPOP | 5 / 12 | 4 / 12 |
| pycma IPOP | 8 / 12 | 6 / 12 |

Two different failures.  **`RoundRobin_CMAES` and pycma fail on basin
selection:** their losing runs end at precision 1e0 … 1e−2, in a local
optimum, and restarts do not rescue them within 2500 evaluations.
**`Blocks` finds the basin (11/12) but converges too slowly** — from
1e−2 to 1e−8 it needs more than the ~1000 evaluations Optuna needs,
because half its evaluations go to jSO; it reaches 1e−8 in 4 runs.  If
the other methods found the basin with p ≈ 0.6, twelve of twelve would
happen with probability ≈ 0.002, so Optuna's rate is not luck — but it
is one problem, and its *d* = 2 sibling (2, 2) is solved by everyone
(Optuna 11/12 to 1e−8).

### 57.4 Early or late: the gap opens after 10 % of the budget

Traces are stored at ~29 log-spaced checkpoints; a step reconstruction
from them overstates each run's precision gap (mean |error| 0.014 AOCC)
but the error cancels in differences (the three segments of Optuna −
`RoundRobin` sum to +0.0376 against the true +0.0378).  Contribution of
each budget segment to Optuna − X:

| segment | − `RoundRobin_CMAES` | − `Blocks` | − BIPOP |
|---|---|---|---|
| evals 0–10 % | −0.001, 3/12 | −0.000, 5/12 | +0.001, 9/12 |
| 10–50 % | +0.010, 9/12 | +0.005, 7/12 | +0.010, 11/12 |
| 50–100 % | **+0.028 [+0.008, +0.048], 9/12** | **+0.026 [+0.007, +0.045], 10/12** | **+0.025 [+0.007, +0.044], 10/12** |

At *d* = 2 our specs are, if anything, *ahead* early (0–10 %: Optuna −
`RoundRobin` −0.0017, 2/12, p = 0.006 — consistent with our CMA-ES
starting at the box centre, see below).  At *d* = 5 three quarters of the
gap is in the second half (Optuna − `RoundRobin`: +0.019 in 10–50 %,
+0.054 in 50–100 %): AOCC accrues for every evaluation spent below the
target, so a basin found at 500 evaluations pays through the whole
second half, and one never found pays nothing.

### 57.5 Mechanism: what differs between the three CMA-ES

Read from `optuna/samplers/_cmaes.py` (Optuna 5.0.0), `cmaes/_cma.py`
(cmaes 0.13.1), `panobbgo/heuristics/cma_es.py` and the pycma adapter in
`panobbgo/harness_baselines.py`:

| | Optuna `CmaEsSampler` (as run) | pycma IPOP/BIPOP (adapter) | panobbgo `CMAES` |
|---|---|---|---|
| search space | [0, 1]^d (`transform_0_1`) | [0, 1]^d | the box |
| σ0 | min(range)/6 = **0.167·range** | **0.25·range** | 0.3·mean(range)/2 = **0.15·range** |
| λ (d = 2 / 5) | 4 + ⌊3 ln n⌋ = 6 / 8 | same | same |
| start mean | uniform random `x0` (the harness passes one); trial 0 is an extra random point that CMA-ES ignores | uniform random `x0` per run | **the box centre** (`on_start`); restarts at a random point |
| restarts | **none** (`restart_strategy` deprecated); the run never stops | IPOP / BIPOP on pycma's criteria | IPOP self-restart on tolx/tolfun/stagnation/conditioncov + σ-divergence |
| bounds | **resampling**: redraw up to 10·n times until inside, then clip | pycma `BoundTransform` (smooth fold-back) | **projection** onto the box; the step that reaches the projected point, Mahalanobis-clipped, enters the update |
| weights / learning rates | Hansen 2016 defaults **with negative weights (active CMA)**, c_m = 1 | defaults, active CMA | Hansen 2016 defaults, **positive weights only** |
| σ clamp | none | none | σ ≤ mean(range) |

Population and σ0 are essentially the same as ours (0.167 vs 0.15 of the
range); pycma's σ0 is larger, and pycma fails on (5, 2) as often as we do,
so a larger initial step is not the cure.  Restarts cannot explain a
first-run basin choice.  What Optuna has that *neither* pycma nor we have
is the resampling bound handling; what it has that we lack but pycma has
is the random start and active CMA.

### 57.6 Hypotheses, each a cheap A/B

1. **H1 — bound handling (top).**  Resampling draws a truncated Gaussian:
   no sample ever sits on a face.  Projection piles every out-of-box
   sample onto the face, so an early generation with a wide σ is ranked
   partly on face points, and a mixture with Gallagher peaks near the
   box edge can pull the mean there.  pycma's fold-back does not pile
   up on faces but still distorts the sampled distribution.  Test:
   a `boundary="resample"` option on `CMAES` (up to 10·n redraws, then
   project), `RoundRobin_CMAES` vs the variant with a shared `seed_name`,
   IOH standard, 12 seeds; primary readout the (5, 2) reach-1e−1 count
   and the paired Δ.  The mirror test is cheaper still and needs no
   panobbgo change: a `cmaes.CMA` baseline with `n_max_resampling=1`
   (clip only) against Optuna's setting — if Optuna loses (5, 2) with
   clipping, H1 is confirmed from the winning side.
2. **H2 — start point.**  Our first run starts at the box centre in every
   seed, so on a given problem all 12 seeds share one starting basin and
   differ only in their samples; Optuna and pycma draw a fresh `x0`.
   The early-budget lead at *d* = 2 (57.4) is the centre start's
   signature.  pycma starts at random and still fails (5, 2), so H2 alone
   cannot explain Optuna — but it is a knob with no downside in
   principle.  Test: `start_from="random"` for the first run.
3. **H3 — active CMA.**  Negative recombination weights shrink the
   covariance along bad directions; they matter on the ill-conditioned
   components (f11 discus, f14) of (5, 2) and in the late budget
   generally.  Evidence is weak here: in-basin convergence 1e−2 → 1e−8 is
   only slightly slower for `RoundRobin` than for Optuna/pycma.  Test:
   negative weights per Hansen 2016 eq. 53, same A/B.

### 57.7 Reading

* **Is Optuna CmaEs better?**  On this battery, yes, by 0.03–0.04 AOCC
  against the shipped specs (3–4 of 12 seeds for us, p ≈ 0.02–0.03) —
  and **all of it is one problem** out of ten: *d* = 5 MA-BBOB
  instance 2.  Without that cell every comparison is level, and at *d* = 2
  the ranking is different (BIPOP first).  With five fixed mixtures per
  dimension this is a finding about one landscape, not about CMA-ES
  implementations in general.
* It is **not** the restart schedule: pycma with restarts is where we
  are.  For plain CMA-ES it is first-run basin selection on a
  Gallagher/Lunacek mixture; for the portfolio it is the halved
  convergence budget once the basin is found.
* §52–§53 already found f22 (Gallagher) to be where the portfolio wins
  most on the fid axis.  Before tuning anything on H1–H3, run the
  external baselines on the BBOB fid battery (24 functions, §52 axis)
  to see whether Optuna's edge exists outside this one mixture; the
  A/B for H1 is cheap enough to run alongside.
* Both our specs remain level with pycma BIPOP/IPOP, the reference
  restart CMA-ES, on every pooled cut.

## 58. The §57 A/B: active CMA carries Optuna's lead; bound handling does not (H1 falsified from both sides)

§57 found Optuna CmaEs ahead of our CMA-ES specs on the cheap track, all
of it on MA-BBOB cell (*d* = 5, inst 2), and named three hypotheses: H1
bound handling (Optuna resamples, we project), H2 the start point (we
start at the box centre), H3 active CMA (negative weights).  #368 made
each an opt-in `CMAES` option, and this is the paired A/B.

**Run.** `rebaseline.yml` run 36275395012 on master 30bb3bd, suites
`ioh-cma-ab`, `ioh-cma-ab-bbob-b200`, `ioh-cma-ab-bbob-b500`, the 12-seed
roster, `release=none`.  12 shards, none failed, one `fp_env_id`
(80ee2a0090c4, the same as §56).  The raw files are in the run's
`rebaseline-references` artifact (90 days); the manifest, the summary
and the analysis script are in
`planning/results/2026-09-27-ab-run36275395012/`.

**Batteries.**
* IOH standard: MA-BBOB, instances 0–4 at *d* 2/5, 500·d.  This is the
  battery §57's hypotheses came from, so it is not independent evidence
  for them.
* The 24 BBOB functions at *d* 2/5/10, instances 0/1, at 200·d and 500·d
  (§52's function axis).  These are the independent test.

**Specs**, all in the same jobs:
* `RoundRobin_CMAES` (RR);
* its variants `_resample` (H1), `_reflect` (a third bound scheme),
  `_randstart` (H2), `_active` (H3) and `_resample_randstart_active`, all
  sharing RR's `seed_name`;
* `Baseline_Optuna_CmaEs`;
* `Baseline_Optuna_CmaEs_clip`: the same sampler with
  `n_max_resampling = 0`, sharing Optuna's seed.  This is §57's "mirror
  test".  **Correction to §57:** in cmaes 0.13.1 `ask()` checks
  `n_max_resampling` draws and then clips one more, so 0 is pure clipping;
  1 would still resample once;
* `Baseline_pycma_BIPOP`.

**Method.** `paired_seed_stats` on the per-seed mean AOCC over the cells
in scope (two specs relabelled to one name, so the pairing is by seed),
t-CI95 over 12 seeds, wins = seeds where the row spec is ahead.  **Bold**
= CI excludes 0.

**Instrument checks.**
* RR, Optuna CmaEs and pycma BIPOP on IOH standard are **bit-identical to
  §56's references**: 360 of 360 runs, AOCC, best f and the full trace.
  That holds for the shard that ran on an AVX-512 EPYC 9V74 as well as
  for the EPYC 7763 one.
* The run predates #370 (FP pin plus `OPENBLAS_L2_SIZE=2048`).  On 9V74
  hosts OpenBLAS 0.3.34 could differ in bits for large matrix products,
  but at *d* ≤ 10 the CMA-ES matrices are far below the sizes involved.
  The 360-run check confirms it for this battery.  Shards ran on EPYC
  7763, 9V74, 9V45 and a Xeon 8370C.
* 3 errors, all `Baseline_Optuna_CmaEs` on BBOB f14, *d* = 2, instance 0
  at 500·d (seeds 7, 2025, 11): Optuna raises `ValueError: nan is invalid
  value` before the first evaluation, and the runs score 0 (the other
  seeds reach 0.76–0.82).  Dropping the cell moves every Optuna delta by
  ≤ 0.002 overall and ≤ 0.011 in the high-conditioning class; no
  conclusion changes.  Logged in `TODO.md`.

### 58.1 IOH standard (the battery §57 came from)

Mean AOCC: RR 0.667, resample 0.671, reflect 0.678, randstart 0.645,
**active 0.705**, all three 0.690, Optuna 0.705, Optuna clip 0.710,
BIPOP 0.667.

| Δ | vs RR | vs Optuna CmaEs |
|---|---|---|
| resample | +0.004 [−0.014, +0.022], 6/12 | **−0.034 [−0.060, −0.008], 1/12** |
| reflect | +0.011 [−0.015, +0.036], 8/12 | **−0.027 [−0.051, −0.003], 3/12** |
| randstart | −0.023 [−0.064, +0.018], 4/12 | **−0.061 [−0.095, −0.027], 1/12** |
| **active** | **+0.038 [+0.014, +0.063], 11/12** | +0.000 [−0.021, +0.021], 6/12 |
| all three | +0.023 [−0.004, +0.050], 7/12 | −0.015 [−0.046, +0.016], 5/12 |
| Optuna CmaEs | **+0.038 [+0.007, +0.069], 9/12** | — |
| Optuna clip | **+0.043 [+0.007, +0.078], 10/12** | +0.005 [−0.018, +0.028], 8/12 |
| pycma BIPOP | −0.000 [−0.042, +0.042], 5/12 | **−0.038 [−0.067, −0.009], 2/12** |

The cuts:
* At *d* = 5, active − RR is +0.061 [+0.016, +0.107], 11/12.
* At *d* = 2 nothing separates: every CI includes 0.
* Without cell (5, 2) every comparison with RR is level (active +0.015
  [−0.009, +0.039]).

So on this battery, too, the lead is concentrated in one cell.

### 58.2 Cell (5, 2): the hit rates

Runs reaching precision 1e−1 by evaluation 911 (the §57 checkpoint) and
by the end (2500), and 1e−8 by the end:

| | AOCC | 1e−1 by 911 | 1e−1 by end | 1e−8 by end |
|---|---|---|---|---|
| RR | 0.423 | 6/12 | 8/12 | 3/12 |
| resample | 0.477 | 8/12 | 8/12 | 3/12 |
| reflect | 0.510 | 7/12 | 10/12 | 5/12 |
| randstart | 0.463 | 9/12 | 9/12 | 3/12 |
| **active** | **0.670** | **11/12** | **12/12** | **9/12** |
| all three | 0.677 | 12/12 | 12/12 | 10/12 |
| Optuna CmaEs | 0.745 | 12/12 | 12/12 | 12/12 |
| Optuna clip | 0.665 | 10/12 | 11/12 | 9/12 |
| pycma BIPOP | 0.404 | 5/12 | 8/12 | 4/12 |

The RR, Optuna and BIPOP counts reproduce §57 exactly (6/12 and 3/12,
12/12 and 12/12, 5/12 and 4/12).

On the cell:
* active − RR: **+0.247 [+0.112, +0.381], 11/12**.
* Optuna clip − Optuna: −0.080 [−0.181, +0.021], 6/12.
* resample − RR: +0.054 [−0.054, +0.162], 8/12.

Clipping costs Optuna some of the cell, not the basin (11/12 reach 1e−1).
Resampling gives our CMA-ES almost none of it.  **Active CMA alone moves
our CMA-ES from 8/12 to 12/12 in the basin and from 3/12 to 9/12 at
1e−8.**

### 58.3 The 24 BBOB functions (independent of §57)

Pooled over *d* 2/5/10, Δ vs RR (without f5 in brackets, see §58.4):

| Δ vs RR | 200·d | 500·d |
|---|---|---|
| resample | **−0.014**, 1/12 (without f5 **+0.005**, 10/12) | −0.008, 2/12 (+0.008, 9/12) |
| reflect | **−0.010**, 1/12 (**+0.007**, 11/12) | +0.000, 7/12 (**+0.010**, 10/12) |
| randstart | **−0.019**, 0/12 (**−0.019**) | **−0.019**, 0/12 (**−0.021**, 0/12) |
| **active** | **+0.011 [+0.003, +0.019], 8/12** (**+0.020**, 11/12) | **+0.041 [+0.031, +0.050], 12/12** (**+0.054**, 12/12) |
| all three | −0.004, 3/12 (**+0.014**, 11/12) | **+0.030**, 12/12 (**+0.052**, 12/12) |
| Optuna CmaEs | −0.002, 4/12 (**+0.017**, 12/12) | **+0.026**, 12/12 (**+0.048**, 12/12) |
| Optuna clip | +0.000, 5/12 (**+0.009**, 11/12) | **+0.026**, 12/12 (**+0.043**, 12/12) |
| pycma BIPOP | **−0.009**, 1/12 | **+0.018**, 11/12 |

Against Optuna CmaEs:
* active is **+0.013 [+0.005, +0.021], 8/12** at 200·d and **+0.015
  [+0.003, +0.026], 8/12** at 500·d.
* Optuna clip − Optuna is +0.002 and +0.001, level.

Per dimension, active − RR:

| | 200·d | 500·d |
|---|---|---|
| *d* = 2 | **+0.028**, 12/12 | **+0.043**, 12/12 |
| *d* = 5 | +0.006, 7/12 | **+0.041**, 11/12 |
| *d* = 10 | −0.001, 5/12 | **+0.038**, 12/12 |

Without f5 active is positive with CI > 0 at every *d* and both budgets.

Per COCO class, Δ vs RR:

| class | active 200·d | active 500·d | resample 200·d | resample 500·d | Optuna 500·d |
|---|---|---|---|---|---|
| separable (f1–5) | **−0.024 [−0.040, −0.009], 2/12** | −0.013 [−0.027, +0.001], 4/12 | **−0.087**, 0/12 | **−0.069**, 0/12 | **−0.060**, 0/12 |
| — without f5 | **+0.017**, 11/12 | **+0.053**, 12/12 | +0.002 | +0.005 | **+0.047**, 12/12 |
| low conditioning (f6–9) | **+0.040**, 11/12 | **+0.071**, 12/12 | **+0.013**, 10/12 | **+0.022**, 8/12 | **+0.033**, 8/12 |
| high conditioning (f10–14) | **+0.049**, 12/12 | **+0.152**, 12/12 | **+0.009**, 10/12 | +0.010, 9/12 | **+0.143**, 12/12 |
| multimodal, global (f15–19) | +0.002, 6/12 | +0.013 [−0.009, +0.034], 8/12 | +0.004 | +0.005 | **+0.026**, 9/12 |
| multimodal, weak (f20–24) | −0.005 [−0.026, +0.015], 6/12 | −0.014 [−0.042, +0.015], 4/12 | −0.003 | +0.000 | −0.011 |

What the classes show:
* **The gain of active CMA is conditioning**: +0.15 on f10–14 at 500·d,
  and per function f11 (discus) +0.29, f2 +0.20, f10 +0.20, f7 +0.14,
  f13 +0.13.  That is exactly the discus / different-powers part of
  cell (5, 2)'s mixture (f22 f24 f11 f14).
* Optuna's gain has the same class profile, which is why the two tie.
* The multimodal classes do not move significantly either way.  f21
  (Gallagher 101 peaks) is the largest single dip, −0.063 / −0.084, CI
  including 0, 4/12.

### 58.4 f5: projection is what solves the linear slope

f5's optimum sits on the box boundary.  Δ vs RR on f5 (RR scores 0.85 /
0.73 / 0.30 at 200·d and 0.94 / 0.88 / 0.60 at 500·d for *d* 2 / 5 / 10):

| f5, 500·d | *d* = 2 | *d* = 5 | *d* = 10 |
|---|---|---|---|
| resample | **−0.37** | **−0.40** | **−0.32** |
| Optuna CmaEs (resamples, active) | **−0.34** | **−0.69** | **−0.44** |
| Optuna clip (clips, active) | +0.03 | **−0.57** | **−0.51** |
| active | +0.00 | **−0.45** | **−0.37** |

Two separate costs, and they add up in Optuna:
* **Resampling** never lets a sample reach the face where f5's optimum
  lies, at any *d*.  Projection lands on it.
* **Active CMA** costs at *d* ≥ 5, not at *d* = 2.  Optuna's clip twin,
  which is active too, shows the same profile.
* A mechanism, not yet tested: on a linear function the best and the
  worst offspring lie along the same gradient axis.  The negative update
  then cancels the rank-μ elongation of C along it, which the run needs
  to travel to the boundary.  This is logged in `TODO.md`.

f5 is also why resample and reflect look negative pooled on BBOB.
Without f5 both are small positives: resample +0.005 / +0.008, reflect
+0.007 / +0.010.

### 58.5 Reading

* **H1 (bound handling) is falsified, from both sides.**
  * Optuna with pure clipping scores as Optuna with resampling on IOH
    standard (+0.005 [−0.018, +0.028]) and on both BBOB budgets (+0.002,
    +0.001).
  * On cell (5, 2) clipping costs Optuna −0.08 (CI including 0), not the
    basin.
  * Our CMA-ES with resampling gains nothing on the headline battery
    (+0.004) and pays on f5.
* **H2 (random first start) is worse, everywhere it is measurable**:
  −0.023 (IOH, CI including 0), −0.019 at both BBOB budgets (0/12), and
  negative in every class.  The centre start is an advantage, not a
  handicap.  Optuna's lead comes despite its random `x0`.
* **H3 (active CMA) carries the whole Optuna lead.**
  * `RoundRobin_CMAES_active` ties Optuna CmaEs on IOH standard (+0.000)
    and on cell (5, 2) (−0.075, CI including 0).
  * It beats RR by +0.038, 11/12.
  * On the independent BBOB battery it beats RR by +0.011 (200·d) and
    +0.041 (500·d, 12/12), and Optuna CmaEs by +0.013 / +0.015.
  * The mechanism is visible in the class profile: conditioning, not
    multimodality.
* **The combination is worse than active alone** (IOH +0.023 vs +0.038;
  BBOB 200·d −0.004 vs +0.011).  Resampling and the random start only
  subtract.
* **Is active "clearly better" by the rule** (a paired CI excluding 0 on
  the headline battery, not worse on any BBOB class)?
  * Headline: yes, +0.038 [+0.014, +0.063].
  * BBOB classes: at 500·d no class is significantly worse.  At 200·d the
    separable class is, −0.024 [−0.040, −0.009].
  * That class loss is all f5 at *d* ≥ 5: the separable class without f5
    is +0.017, 11/12.
  * So active **misses the rule on one class at one budget, for one
    function whose optimum is on the boundary**.
  * It is the clear candidate for the default.  Whether f5's loss is
    accepted, or fixed first (§58.4 hypothesis), is the coordinator's
    call; this entry does not flip the default.
* **Selected-maximum caveat** (benchmarking.md): active is one of five
  pre-registered variants, and IOH standard is where the hypotheses came
  from.  The BBOB battery is the fresh check, 24 functions × 3 dims × 2
  instances × 2 budgets, and it confirms the sign at both budgets (8/12
  and 12/12).
* **Open:** pycma BIPOP also runs active CMA by default (§57.5) and gains
  far less on the high-conditioning class (+0.056 vs +0.152 at 500·d).
  Its σ0 and restart schedule differ, and the gap is not explained here.

## 59. Active CMA at the box face: the f5 loss is the boundary repair, not the gradient; a guard

§58.4 left one weakness of `active=True`: −0.4 to −0.5 AOCC on BBOB f5
(linear slope, optimum on the box face) at *d* ≥ 5.  The hypothesis
logged there — the negative update cancels the rank-μ elongation of C
along the gradient — is **falsified** below.  The cause is an
interaction with our boundary repair, and PR #373 adds a guard.

**Setup.** All runs are local, niced, one BLAS thread, cheap in-process
problems, 500·d, sync evaluation.  None of this is battery evidence; it
locates the mechanism.  BBOB f5 is written out in closed form:
`x_opt = ±5` per coordinate on the face of [−5, 5]^d,
`s_i = sign·10^((i−1)/(d−1))`, f5's plateau beyond the face.  8 seeds;
the instance is drawn per seed.

### 59.1 Reproduction and trace

f5, mean AOCC over 8 seeds:

| | *d* = 5 | *d* = 10 |
|---|---|---|
| positive-only (the default) | 0.938 | 0.685 |
| active, tutorial update (§58's `RoundRobin_CMAES_active`) | 0.525 | 0.199 |

This matches the A/B (§58.4: 0.88 vs 0.43 at *d* = 5).  In a failing seed
(*d* = 5):
* The mean stalls ~3 units from the face in its least-weighted
  coordinate.
* C collapses along the slope direction: the variance along the
  gradient, relative to the mean eigenvalue, falls from ~1.2 to
  0.001–0.006, and cond(C) reaches 1e4.
* σ then runs to its clamp.

The positive-only run of the same seed keeps that ratio near 0.6 and
converges.

**The gradient hypothesis is falsified.** The same f5 in a box of
[−50, 50], where the plateau keeps the optimum value the same and
nothing is ever projected, gives active 0.991 and positive-only 0.991
(*d* = 5).  On a linear function without a box, active CMA is fine.

**The mechanism: the boundary repair.**
* A projected offspring enters the update through the step that
  reaches the projected point (`_emit_generation`), so its step is cut
  short at the face.
* At a face the *best* offspring are exactly the projected ones.  The
  *worst* step inward and keep their full length.
* Along that coordinate, eq. (47)'s negative term then outweighs the
  positive term, and C shrinks there generation after generation.
* The tutorial's balance assumes every ranked step is an unmodified
  sample of N(0, C).  A truncated step breaks it.

### 59.2 Treatments

f5, mean AOCC, 8 seeds:

| treatment | *d* = 5 | *d* = 10 |
|---|---|---|
| positive-only | 0.938 | 0.685 |
| active, tutorial | 0.525 | 0.199 |
| zero only the repaired offspring's own negative weight | 0.655 | 0.581 |
| raw sampled steps in rank-μ (genotype view) | 0.874 | 0.616 |
| raw sampled steps in the negative term only | 0.819 | 0.454 |
| negative weights × unrepaired fraction | 0.661 | 0.490 |
| negative steps masked in the clipped coordinates | 0.480 | 0.204 |
| **positive-only update in a generation with any repaired offspring** | **0.860** | **0.702** |
| … only when a repaired one is among the μ selected | 0.839 | 0.702 |

Zeroing only the repaired offspring's own weight does little, because
the worst offspring are rarely the projected ones.  Masking coordinates
breaks eq. (47) differently and also fails on the ellipsoid.

The adopted rule is `CMAES(active_skip_repaired=True)`, on by default
when `active=True`.  One switch controls two rules:
1. a generation that contains a repaired offspring gets the tutorial's
   positive-only update;
2. injected foreign points never receive a negative weight, pycma's
   `CMA_active_injected = 0`.

The rule is ours, not Hansen 2016's.  pycma never meets the case,
because its default `BoundTransform` runs CMA in the unbounded genotype
space.  The spec names keep §58's meaning:
* `RoundRobin_CMAES_active` is the unguarded update;
* `RoundRobin_CMAES_active_guarded` is the guarded one;
* `…_resample_randstart_active` stays unguarded.

### 59.3 Beyond f5: where the guard helps and what it gives up

The reviewer's probe of #373
(`scratchpad/rv373/probe.py`, re-run here): 8 seeds, 500·d, rotation
and instance per seed.  The problems:
* f5;
* `ell_near`: rotated ellipsoid, cond 1e6, optimum at ±4.5 per
  coordinate, near but inside the faces;
* `ell_mid`: the same at ±3;
* `lin_ell`: slope toward a face in x₁ plus a rotated ellipsoid in the
  rest;
* `ell_out`: cond 1e4, unconstrained optimum outside the box at about
  ±6, so the constrained optimum lies on faces;
* `ell_small`: box [−1, 1], interior optimum.

Mean AOCC. pos = positive-only, tut = tutorial active, guard = the
guard, rs+act = `boundary="resample"` with active:

| problem, *d* | pos | tut | guard | rs+act |
|---|---|---|---|---|
| f5, 5 | 0.803 | 0.566 | 0.851 | 0.444 |
| f5, 10 | 0.622 | 0.279 | 0.722 | 0.430 |
| ell_near, 5 | 0.327 | 0.308 | 0.424 | 0.580 |
| ell_near, 10 | 0.079 | 0.021 | 0.151 | 0.361 |
| ell_mid, 5 | 0.338 | 0.383 | 0.500 | 0.553 |
| ell_mid, 10 | 0.076 | 0.284 | 0.269 | 0.340 |
| lin_ell, 5 | 0.225 | 0.274 | 0.422 | 0.429 |
| lin_ell, 10 | 0.049 | 0.144 | 0.186 | 0.209 |
| ell_out, 5 | 0.417 | 0.125 | 0.425 | 0.239 |
| ell_out, 10 | 0.188 | 0.030 | 0.213 | 0.151 |
| ell_small, 5 | 0.467 | 0.573 | 0.581 | 0.651 |
| ell_small, 10 | 0.164 | 0.398 | 0.378 | 0.421 |

Paired deltas (mean ± SE over 8 seeds, wins):

| problem, *d* | guard − tut | guard − pos | rs+act − guard | top-μ − guard |
|---|---|---|---|---|
| f5, 5 | +0.285 ± 0.102, 8/8 | +0.048 ± 0.090 | −0.407 ± 0.060 | −0.023 |
| f5, 10 | +0.443 ± 0.048, 8/8 | +0.101 ± 0.079 | −0.292 ± 0.019 | −0.033 |
| ell_near, 5 | +0.116 ± 0.091 | +0.097 ± 0.035 | +0.156 ± 0.046 | +0.072 ± 0.044 |
| ell_near, 10 | +0.130 ± 0.039 | +0.072 ± 0.029 | +0.210 ± 0.033 | +0.077 ± 0.026 |
| ell_mid, 5 | +0.117 ± 0.077 | +0.162 ± 0.030 | +0.053 ± 0.013 | +0.032 ± 0.018 |
| ell_mid, 10 | −0.015 ± 0.064 | +0.193 ± 0.025 | +0.071 ± 0.032 | −0.007 |
| lin_ell, 5 | +0.148 ± 0.083 | +0.197 ± 0.019 | +0.007 ± 0.026 | −0.008 |
| lin_ell, 10 | +0.042 ± 0.027 | +0.137 ± 0.020 | +0.023 ± 0.018 | +0.011 |
| ell_out, 5 | +0.300 ± 0.047, 8/8 | +0.008 ± 0.034 | −0.186 ± 0.054 | +0.024 |
| ell_out, 10 | +0.183 ± 0.026, 8/8 | +0.025 ± 0.027 | −0.062 ± 0.037 | −0.007 |
| ell_small, 5 | +0.008 ± 0.061 | +0.115 ± 0.030 | +0.069 ± 0.012 | +0.059 ± 0.010 |
| ell_small, 10 | −0.021 ± 0.048 | +0.213 ± 0.030 | +0.043 ± 0.035 | +0.017 |

### 59.4 Reading

* **The guard is needed beyond f5.** Wherever the optimum is on or near
  a face (f5, `ell_out`, `ell_near`), the tutorial update collapses and
  the guard recovers +0.12…+0.44.  On interior problems (`ell_mid`,
  `ell_small` at *d* = 10) the two are level within noise.
* **The guard never falls below positive-only** (+0.008…+0.213).  On
  every ellipsoid it keeps active's conditioning gain.
* **What the guard gives up.**
  * With the optimum *near* a face but inside, resample + active is
    ahead of the guard: +0.16 / +0.21 on `ell_near`, +0.05 / +0.07 on
    `ell_mid`.  Resampling never produces a repaired step, so active
    runs in every generation.
  * With the optimum *on* the face, the same combination loses heavily:
    f5 −0.41 / −0.29, `ell_out` −0.19 / −0.06.  It never samples the
    face.
  * Neither of the two dominates.  The principled alternative is a
    genotype mapping (pycma's `BoundTransform`): unbounded steps, a
    smooth fold for evaluation, so no repair ever happens and active
    CMA needs no guard.  Logged in `TODO.md`.
* **The any-repaired rule is deliberately conservative.**  The top-μ
  variant is within noise on most cells.  It is ahead by +0.072 /
  +0.077 on `ell_near` and +0.059 on `ell_small` at *d* = 5, and behind
  by 0.02–0.03 on f5.  It is a follow-up, not adopted.
* **Next:** the §58 A/B again with `RoundRobin_CMAES_active_guarded`
  next to the unguarded `RoundRobin_CMAES_active`, dispatched after #373
  merges.  The default decision waits for it.

## 60. The guarded active CMA on the §58 batteries: f5 fixed, no class worse, a small give-back on conditioning

§59 added the repair guard to `active=True` (`active_skip_repaired`,
#373).  This is the §58 A/B again, now with
`RoundRobin_CMAES_active_guarded` next to the unguarded
`RoundRobin_CMAES_active`.

**Run.** `rebaseline.yml` run 36291320296 on master 97d65b4, the same
three suites and the 12-seed roster, `release=none`.  12 shards, none
failed, one `fp_env_id` (80ee2a0090c4, the same as §56 and §58).  The
raw files are in the run's `rebaseline-references` artifact; the
manifest, summary and analysis scripts are in
`planning/results/2026-09-27-ab-run36291320296/`.  The method is §58's:
paired per seed, t-CI95, wins out of 12, **bold** = CI excludes 0.

**Instrument check.** Every spec whose code path did not change is
**bit-identical to §58's run** (36275395012): RR, unguarded active,
resample, the three-way combination, Optuna CmaEs and pycma BIPOP.
That is 21,456 of 21,456 runs over the three batteries (AOCC, best f,
full trace).  So the guarded arm is compared against exactly §58's
numbers.  #370 (the `OPENBLAS_L2_SIZE` pin) came in between and
changes nothing here.  The 3 Optuna crashes of §58 recur (f14,
*d* = 2, instance 0, 500·d).

### 60.1 IOH standard

| Δ | vs RR | vs Optuna CmaEs | vs unguarded active |
|---|---|---|---|
| active (unguarded, §58) | **+0.038 [+0.014, +0.063], 11/12** | +0.000 [−0.021, +0.021], 6/12 | — |
| **active guarded** | +0.028 [−0.001, +0.056], 8/12 | −0.010 [−0.047, +0.027], 5/12 | −0.010 [−0.044, +0.023], 6/12 |

* At *d* = 5 the guarded version is +0.070 [+0.016, +0.123] vs RR,
  10/12 (unguarded +0.061).
* At *d* = 2 it is −0.014 (n.s.; unguarded +0.015).
* On cell (5, 2) it is +0.241 [+0.072, +0.410], 10/12.  Hit rates:
  1e−1 by 911 in 11/12, by the end in 12/12, 1e−8 by the end in 10/12
  (unguarded 11 / 12 / 9, RR 6 / 8 / 3, Optuna 12 / 12 / 12).

### 60.2 The 24 BBOB functions (*d* 2/5/10, instances 0/1)

Pooled:

| | guarded vs RR | vs Optuna | vs unguarded | unguarded vs RR |
|---|---|---|---|---|
| 200·d | **+0.013 [+0.007, +0.019], 11/12** | **+0.015**, 12/12 | +0.002 [−0.001, +0.005], 7/12 | **+0.011**, 8/12 |
| 500·d | **+0.044 [+0.036, +0.051], 12/12** | **+0.018**, 12/12 | +0.003 [−0.002, +0.008], 7/12 | **+0.041**, 12/12 |
| 200·d without f5 | **+0.014**, 10/12 | −0.003 | **−0.005 [−0.009, −0.002], 1/12** | **+0.020** |
| 500·d without f5 | **+0.045**, 12/12 | −0.003 | **−0.009 [−0.014, −0.004], 1/12** | **+0.054** |

Per COCO class, Δ vs RR (the guarded − unguarded delta in brackets):

| class | guarded 200·d | guarded 500·d |
|---|---|---|
| separable (f1–5) | +0.008 [−0.011, +0.027], 7/12 (**+0.033**) | **+0.038**, 11/12 (**+0.051**) |
| low conditioning (f6–9) | **+0.037**, 11/12 (−0.003) | **+0.065**, 12/12 (−0.006) |
| high conditioning (f10–14) | **+0.032**, 12/12 (**−0.017**) | **+0.123**, 12/12 (**−0.029**) |
| multimodal, global (f15–19) | +0.002, 7/12 (+0.000) | +0.011, 8/12 (−0.001) |
| multimodal, weak (f20–24) | −0.010 [−0.025, +0.006], 5/12 (−0.004) | −0.016 [−0.040, +0.009], 3/12 (−0.002) |

The unguarded separable class at 200·d was **−0.024 [−0.040, −0.009]**
(§58.3).  **No class is significantly worse than RR for the guarded
version, at either budget.**

f5, Δ vs RR:

| f5 | *d* = 2 | *d* = 5 | *d* = 10 |
|---|---|---|---|
| unguarded, 200·d | +0.004 | **−0.377** | **−0.193** |
| guarded, 200·d | +0.021 | −0.102 [−0.255, +0.050], 6/12 | +0.033 |
| unguarded, 500·d | +0.002 | **−0.453** | **−0.372** |
| guarded, 500·d | +0.009 | −0.039 [−0.113, +0.034], 5/12 | +0.061 |

Guarded − unguarded on f5 is **+0.28 / +0.23** at 200·d (*d* 5/10) and
**+0.41 / +0.43** at 500·d.

Per function (500·d, pooled over *d*): the guarded gains are f11 +0.23,
f2 +0.17, f10 +0.15, f7 +0.14, f13 +0.11, f14 +0.09 (unguarded +0.29,
+0.20, +0.20, +0.14, +0.13, +0.09).  The largest dip is f21 (Gallagher
101 peaks), −0.10, as in §58 (−0.08).

### 60.3 Reading and recommendation

* **The guard does what §59 said.** f5 goes from −0.38…−0.45 to level
  (no *d* significantly negative), the separable class at 200·d from
  −0.024 to +0.008, and no BBOB class is worse than RR.
* **It gives back a little elsewhere**, consistently: −0.017 / −0.029
  on high conditioning (still +0.032 / +0.123 over RR) and −0.005 /
  −0.009 on all functions without f5.  The likely cause is generations
  that project early, while σ is still wide, running without the active
  term (§59.4; the top-μ variant and a genotype mapping are the logged
  follow-ups).
* **Against the rule** (a paired CI excluding 0 on the headline
  battery, not worse on any BBOB class), neither version passes both
  parts:
  * unguarded passes the headline (+0.038 [+0.014, +0.063]) and fails
    the separable class at 200·d;
  * guarded passes every class and misses the headline CI by a hair
    (+0.028 [**−0.001**, +0.056], 8/12).
* **The recommendation is to make `active=True` (guarded) the default.**
  * The independent battery (24 functions × 3 dims × 2 instances × 2
    budgets, 1,728 cells per budget per spec) has it at **+0.013,
    11/12** and **+0.044, 12/12** over RR, and ahead of Optuna CmaEs at
    both budgets.
  * The 10-cell IOH battery, where the hypothesis came from, has it
    ahead by +0.028 with a lower CI bound of −0.001.
  * It removes the one pathology of the unguarded version.
  * This is the coordinator's call; this entry does not flip the
    default.  A flip changes every CMA-ES trajectory, so it needs a
    re-baseline afterwards.

## 61. Re-baseline with the active-CMA default (2026-09-27)

Master 0629403 (#375: guarded active CMA is the `CMAES` default, §60),
`rebaseline.yml` run 36295705127, 29 jobs, none failed or missing, one
`fp_env_id` 80ee2a0090c4, release `rebaseline-2026-09-27`, summary `planning/results/2026-09-27/SUMMARY.json`,
12 seeds.  Every spec without a CMA-ES arm is bit-identical to §56 (same
per-spec means to the last digit), so every difference below is the new
default.

| Suite / spec | §56 (positive-only) | §61 (active) | Δ |
|---|---|---|---|
| composite quick / standard | 0.4029 / 0.4257 | 0.4029 / 0.4247 | 0 / −0.001 |
| IOH standard `RegimeGate_oracle` | 0.6771 | **0.7097** | +0.033 |
| IOH standard `RoundRobin_CMAES` | 0.6674 | 0.6951 | +0.028 |
| IOH standard `Blocks_warm_CMAES_JSO` | 0.6738 | 0.6669 | −0.007 |
| IOH quick `RoundRobin_CMAES` | 0.3893 | 0.3871 | −0.002 |
| families free `CMAES_alone` | 0.4011 | 0.4641 | +0.063 |
| families constrained `CMAES_alone` | 0.4872 | 0.5286 | +0.041 |
| families shapes `CMAES_alone` | 0.3401 | 0.3698 | +0.030 |
| families failure `CMAES_alone` | 0.3433 | 0.4369 | +0.094 |
| families `Blocks_uniform_cj_warm2` | 0.340 / 0.417 / 0.338 / 0.281 | 0.329 / 0.420 / 0.343 / 0.291 | −0.011 … +0.010 |

External references (unchanged): Optuna CmaEs 0.7052, pycma BIPOP 0.6671,
IPOP 0.6546, NGOpt 0.5961 on IOH standard.

Reading (unpaired means; the paired A/B is §60):

* **On the cheap track panobbgo is now ahead of every external baseline**:
  `RegimeGate_oracle` 0.7097 > Optuna CmaEs 0.7052 (it was −0.028 in §56),
  and plain `RoundRobin_CMAES` 0.695 > pycma BIPOP 0.667.  §57's single
  losing cell was the one active CMA fixes (§58).
* **The sharing portfolio does not profit** from the stronger CMA-ES arm:
  `Blocks_warm_CMAES_JSO` slips −0.007 and now trails `RoundRobin_CMAES`
  by 0.028 at 500·d; on the families `CMAES_alone` leads every preset by
  0.02–0.15 over the portfolio.  Consistent with §46/§53 (sharing pays at
  low budget, not at 500·d) — and a reason to recheck the block scheduler's
  policy table (TODO).
* The failure families gain most (+0.094): active CMA plus the §59 guard
  handles crash regions better than the positive-only update.

## 62. First expensive-track measurement: the free families at 20·d / 100·d, q = 1…64 (positive-only CMA-ES)

> **Pre-active CMA-ES.**  This run is on commit 1778e1b, before #375
> made guarded active CMA the `CMAES` default (§60/§61).  Every
> panobbgo spec here has a CMA-ES arm running the **old positive-only
> update**; the externals are unaffected.  On the cheap track the
> switch moved `RegimeGate_oracle` +0.033 and `RoundRobin_CMAES` +0.028,
> `Blocks_warm_CMAES_JSO` −0.007, and `CMAES_alone` +0.03…+0.09 on the
> families at 500·d (§61).  Nobody has measured the switch at 20·d /
> 100·d or on the virtual clock.  Treat every panobbgo number below as
> pre-active.  Direction and size under the new default are unknown.

**Run.** `measure.yml` run 36274781342 on 1778e1b, the default grid:
the `free` family preset (ackley, ellipsoid, rastrigin, rosenbrock,
sharp_ridge; 3 instances each) at 20·d and 100·d, d = 2/5/10.  Virtual
clock: async pull policy, log-normal durations (σ 0.5), common random
numbers per cell.  q ∈ {1, 4, 16, 64} with q ≤ bm (q = 64 only at 100·d),
5 seeds (3, 7, 42, 1234, 2025).  This gives 21 cells and 323 units in 55
shard jobs, about 90 runner-hours and 9.5 h wall.  Each job had one
`fp_env_id` (80ee2a0090c4).  Two units are missing, none failed:
`SMAC-01` hit the 330-minute `Measure` stop (SMAC at d = 10, 20·d costs
about 1060 s per run, far over the plan's estimate) and lost
`SMAC.free.b100.q1.d2.s3/.s7`.  SMAC is a q = 1 reference row outside
the pool, so no pool member and no headline cell is short (every cell
has n = 75/75).  Coverage as designed: neither qLogEI nor SMAC runs at
d = 10, 100·d; SMAC runs only at q = 1 and not at d = 5, 100·d.  The pool
of a (d, bm) is every external that ran at every q (qLogEI, TuRBO-1,
NGOpt, Optuna CmaEs/TPE, Py-BOBYQA, pycma IPOP/BIPOP; without qLogEI at
d = 10, 100·d).  Files: `summary.md` and `plan.json` in
`planning/results/2026-09-27-measure-run36274781342/`.  `summary.json`
(1.1 MB) is not committed; it stays in the run's `measure-summary`
artifact, and the per-q and per-baseline breakdowns below come from it.

Method (the summary's, pre-declared in `doc/dev/benchmarking.md`,
"Expensive-track measurement").  The headline compares
`Blocks_warm_CMAES_JSO` with the pool's best on AOCC at q = 1 and on
`aocc_time` at q > 1.  Δ is paired over the 5 seeds on the common
(seed, instance) runs, with a t-CI95 and wins/5.  **Holm** is applied over
the 21 headline cells only; every other CI below is unadjusted and
descriptive.  With 5 seeds, 5/5 wins alone has p = 0.0625.  The CIs are
conditional on the 3 fixed instances per family.  The pool's best is a
selected maximum over 7–8 baselines, which biases the headline Δ against
panobbgo.  **bold** = Holm-adjusted p < 0.05.

### 62.1 Headline: `Blocks_warm_CMAES_JSO` − pool best

| cell | metric | pool best | Blocks | Δ [CI95] wins | p_holm | rank |
|---|---|---|---|---|---|---|
| d2 20·d q1 | aocc | TuRBO1 0.145 | 0.093 | **−0.053** [−0.073, −0.032] 0/5 | 0.029 | 6/13 |
| d2 20·d q4 | time | qLogEI 0.100 | 0.085 | −0.016 [−0.025, −0.006] 0/5 | 0.093 | 2/12 |
| d2 20·d q16 | time | qLogEI 0.067 | 0.063 | −0.003 [−0.010, +0.003] 1/5 | 0.919 | 4/12 |
| d2 100·d q1 | aocc | PyBOBYQA 0.440 | 0.215 | **−0.224** [−0.248, −0.200] 0/5 | <0.001 | 4/13 |
| d2 100·d q4 | time | qLogEI 0.209 | 0.214 | +0.005 [−0.005, +0.014] 4/5 | 0.919 | 1/12 |
| d2 100·d q16 | time | qLogEI 0.175 | 0.171 | −0.004 [−0.014, +0.006] 1/5 | 0.919 | 2/12 |
| d2 100·d q64 | time | qLogEI 0.113 | 0.093 | −0.020 [−0.033, −0.007] 0/5 | 0.113 | 4/12 |
| d5 20·d q1 | aocc | NGOpt 0.073 | 0.044 | −0.029 [−0.045, −0.013] 0/5 | 0.080 | 7/13 |
| d5 20·d q4 | time | qLogEI 0.049 | 0.044 | −0.005 [−0.009, −0.001] 0/5 | 0.187 | 2/12 |
| d5 20·d q16 | time | qLogEI 0.040 | 0.034 | **−0.006** [−0.008, −0.004] 0/5 | 0.019 | 2/12 |
| d5 100·d q1 | aocc | PyBOBYQA 0.145 | 0.120 | −0.026 [−0.053, +0.001] 0/5 | 0.402 | 3/12 |
| d5 100·d q4 | time | TuRBO1 0.112 | 0.116 | +0.004 [−0.013, +0.021] 3/5 | 0.919 | 1/12 |
| d5 100·d q16 | time | qLogEI 0.072 | 0.098 | **+0.027** [+0.019, +0.034] 5/5 | 0.008 | 1/12 |
| d5 100·d q64 | time | qLogEI 0.062 | 0.043 | **−0.019** [−0.022, −0.016] 0/5 | 0.001 | 3/12 |
| d10 20·d q1 | aocc | NGOpt 0.051 | 0.028 | **−0.023** [−0.029, −0.017] 0/5 | 0.007 | 7/13 |
| d10 20·d q4 | time | qLogEI 0.034 | 0.027 | **−0.007** [−0.009, −0.006] 0/5 | 0.002 | 4/12 |
| d10 20·d q16 | time | qLogEI 0.031 | 0.026 | **−0.006** [−0.007, −0.004] 0/5 | 0.008 | 3/12 |
| d10 100·d q1 | aocc | TuRBO1 0.096 | 0.080 | −0.017 [−0.026, −0.007] 0/5 | 0.093 | 4/11 |
| d10 100·d q4 | time | TuRBO1 0.085 | 0.081 | −0.004 [−0.010, +0.002] 1/5 | 0.706 | 2/11 |
| d10 100·d q16 | time | TuRBO1 0.060 | 0.067 | +0.007 [−0.004, +0.018] 4/5 | 0.706 | 2/11 |
| d10 100·d q64 | time | Optuna TPE 0.028 | 0.033 | +0.005 [+0.003, +0.008] 5/5 | 0.058 | 1/11 |

(rank: among every strategy of the cell on its headline metric, panobbgo
specs and SMAC included.)

* **Holm: one win, seven losses, thirteen cells unresolved.**
  * The win is d = 5, 100·d, q = 16: +0.027, 5/5, ahead of every
    external (0.098 against qLogEI's 0.072).
  * The losses are three q = 1 cells (d = 2 at 20·d and 100·d, d = 10 at
    20·d), three 20·d cells at q > 1 (d5/q16, d10/q4, d10/q16), and
    d5/100·d/q64.
  * d10/100·d/q64 is +0.005, 5/5, and just misses (p_holm 0.058).
* **Blocks is first in 4 cells** (d2/100·d/q4, d5/100·d/q4 and q16,
  d10/100·d/q64) and first or second in 10 of 21.  All four are at 100·d
  with 4 ≤ q ≤ 64.  At 20·d it never leads.
* **The biggest loss is one family.**  At d2/100·d/q1 the loss is −0.224
  against Py-BOBYQA (0.440), whose quadratic model solves the ellipsoid:
  that family alone is −0.655 [−0.695, −0.614].  Against the next
  externals the same cell is −0.056 (TuRBO1), +0.010 (qLogEI) and +0.034
  (NGOpt).  Py-BOBYQA is sequential: at q > 1 it waits for each call, and
  its `aocc_time` falls to 0.02–0.17 at d = 2.

### 62.2 The other panobbgo specs (Δ vs the same pool best)

* `RegimeGate_oracle` equals Blocks at d ≤ 5.  Its table row
  (`dim <= 5, bpd <= 200`) selects the portfolio, and it runs on the same
  seed stream, so the numbers match to the digit.  At d = 10 the row
  `dim >= 10, bpd <= 500` selects **CMA-ES alone**:

  | d = 10 | 20·d q1 / q4 / q16 | 100·d q1 | q4 | q16 | q64 |
  |---|---|---|---|---|---|
  | Blocks | −0.023 / −0.007 / −0.006 | −0.017 | −0.004 | +0.007 | +0.005 |
  | RegimeGate_oracle | −0.025 / −0.008 / −0.005 | −0.053 (0.044) | −0.041 | +0.019 [+0.013, +0.025] 5/5 | +0.005 |
  | RoundRobin_CMAES | −0.025 / −0.009 / −0.007 | −0.053 (0.044) | −0.041 | −0.023 | −0.003 |

  * At 100·d, q ≤ 4, the gate's CMA-ES-alone choice scores 0.036–0.037
    below the portfolio (unpaired means).  That contradicts the table
    row at this budget.  The row was measured at 500·d (§42), with
    positive-only CMA-ES; the §61 recheck TODO covers it.
  * At q = 16, CMA-ES alone inside the block strategy (0.079) is far
    above the same arm under `RoundRobin_CMAES` (0.037).  That points at
    dispatch: the block strategy with one arm fills the workers
    differently from round-robin.  Unexplained, not investigated.
* `RoundRobin_CMAES` is behind the pool's best in all 21 cells and
  behind Blocks in all 21 (closest: d10/20·d/q1, 0.026 vs 0.028).  At
  100·d the portfolio is ahead of CMA-ES alone by +0.05…+0.07 at d 2/5
  and +0.03…+0.04 at d = 10 (q ≤ 16, unpaired means).  At this budget,
  pre-active, sharing pays.  This is the §46/§53 low-budget regime, and
  the opposite of §61's 500·d picture.
* `RoundRobin_Random` is last or near last at q ≤ 4.  At q ≥ 16 it rises
  into the middle of the table (d2/100·d/q64: 0.088, ahead of NGOpt,
  pycma and `RoundRobin_CMAES`), because the time horizon squeezes
  everyone: 200 evaluations on 64 workers is about 3 rounds.

### 62.3 Which external is the pool's best

| | q = 1 | q = 4 | q = 16 | q = 64 |
|---|---|---|---|---|
| d2 20·d | TuRBO1 | qLogEI | qLogEI | — |
| d5 20·d | NGOpt | qLogEI | qLogEI | — |
| d10 20·d | NGOpt | qLogEI | qLogEI | — |
| d2 100·d | Py-BOBYQA | qLogEI | qLogEI | qLogEI |
| d5 100·d | Py-BOBYQA | TuRBO1 | qLogEI | qLogEI |
| d10 100·d (no qLogEI) | TuRBO1 | TuRBO1 | TuRBO1 | Optuna TPE |

* At q = 1 a sequential local or model-based method leads.
* At q > 1 qLogEI leads wherever it runs, except d5/100·d/q4 (TuRBO1).
* The cheap CMA-ES family (pycma IPOP/BIPOP, Optuna CmaEs) and NGOpt
  never lead at q > 1.  On `aocc_time` Blocks is ahead of them in every
  q > 1 cell by up to +0.08, apart from two ties with Optuna CmaEs
  (−0.000 at d2/20·d/q16, −0.004 at d2/100·d/q64).
* SMAC (q = 1 reference) is below the pool's best in all four cells
  where it ran; no reference-row flag was raised.

### 62.4 Per family: the minimum of each row (the roadmap claim)

The roadmap claim (§2) is "never much worse than the best single solver
on any problem class".  Per cell, the worst family of Blocks − pool best
(the cell's overall best external, not a per-family best, so this is the
lenient version; n.s. = unadjusted CI includes 0):

| cell | q = 1 | q = 4 | q = 16 | q = 64 |
|---|---|---|---|---|
| d2 20·d | ellipsoid −0.071 | ackley −0.032 | ackley −0.010 (n.s.) | — |
| d5 20·d | sharp_ridge −0.089 | sharp_ridge −0.025 | sharp_ridge −0.023 | — |
| d10 20·d | sharp_ridge −0.087 | ackley −0.020 | ackley −0.014 | — |
| d2 100·d | ellipsoid −0.655 | rastrigin −0.038 | ellipsoid −0.023 (n.s.) | ackley −0.035 |
| d5 100·d | ellipsoid −0.114 | rastrigin −0.005 (n.s.) | rastrigin −0.012 | sharp_ridge −0.049 |
| d10 100·d | ackley −0.042 (n.s.) | ackley −0.024 | rastrigin −0.010 (n.s.) | rastrigin −0.001 (n.s.) |

* **The claim fails at q = 1.**  Py-BOBYQA takes the ellipsoid (d = 2/5,
  100·d; −0.66 and −0.11).  NGOpt takes sharp_ridge at 20·d (−0.09 at
  d 5/10).
* **At q > 1 no family minimum is below −0.05.**  But the cells' own
  scale is 0.03–0.2, so −0.02…−0.05 is 10–50 % of the value.
* 16 of the 21 cells have at least one family whose unadjusted CI is
  below 0.  The five with none are d2/20·d/q16, d2/100·d/q16,
  d5/100·d/q4, d10/100·d/q16 and d10/100·d/q64.  All five have a headline
  Δ ≥ −0.004.  The converse does not hold: d2/100·d/q4 and d5/100·d/q16
  lose rastrigin.
* Recurring weak families: **sharp_ridge at 20·d** (d 5/10, every q) and
  **ackley/rastrigin at q > 1**.  **rosenbrock** is the one family where
  Blocks is often ahead: CI above 0 in 7 cells (d 5/10), up to +0.075 at
  d5/100·d/q16.
* **The ellipsoid family is uninformative at d ≥ 5.**  Scores there are
  ≤ 0.015 for everyone except Py-BOBYQA at d5/100·d/q1 (0.120), and
  exactly 0 at d5/20·d and in almost all of d = 10.  The
  `+0.000 [+0.000, +0.000] 0/5` entries in `summary.md` are floors, not
  ties.

### 62.5 How Δ changes with q (the thesis prediction)

Blocks − reference on the headline metric, mean over the d of each row
(qLogEI: d 2/5 at 100·d):

| | q = 1 | q = 4 | q = 16 | q = 64 |
|---|---|---|---|---|
| 20·d vs pool best | −0.035 | −0.009 | −0.005 | — |
| 20·d vs qLogEI | −0.012 | −0.009 | −0.005 | — |
| 20·d vs TuRBO1 | −0.034 | +0.003 | +0.003 | — |
| 100·d vs pool best | −0.089 | +0.002 | +0.010 | −0.011 |
| 100·d vs qLogEI | +0.022 | +0.019 | +0.011 | −0.019 |
| 100·d vs TuRBO1 | −0.029 | +0.014 | +0.030 | +0.003 |
| 100·d vs pycma BIPOP | +0.026 | +0.047 | +0.059 | +0.013 |

Values shrink with q because the horizon is budget/q mean durations.
Ratios are therefore more telling.  Blocks/qLogEI for q = 1 → 4 → 16
(→ 64 at 100·d):

| | d = 2 | d = 5 | d = 10 |
|---|---|---|---|
| 100·d | 1.04 → 1.02 → 0.98 → 0.82 | 1.40 → 1.41 → 1.36 → 0.69 | (qLogEI not run) |
| 20·d | 0.82 → 0.85 → 0.94 | 0.90 → 0.90 → 0.85 | 0.76 → 0.79 → 0.84 |

* **Against TuRBO-1 the thesis holds from q = 1 to q = 16.**  TuRBO-1 is
  batch-synchronous: its `aocc_time`/AOCC ratio is 0.58–0.88 at q > 1.
  Blocks's ratio is 0.85–1.00 up to q = 16, except d2/20·d/q16 (0.73:
  40 evaluations on 16 workers).  So a Δ of −0.029 at q = 1 becomes
  +0.030 at q = 16 (100·d).
* **Against qLogEI it does not.**  At 100·d panobbgo's lead shrinks
  with q and flips at q = 64.  At 20·d the deficit shrinks with q at
  d 2/10 but not at d = 5.
* **The q = 64 loss looks like scheduling, not search.**
  * On AOCC over evaluations Blocks is ahead of qLogEI at q = 64:
    +0.013 [−0.002, +0.028] at d = 2 and +0.035 [+0.028, +0.043] 5/5 at
    d = 5.  It is also ahead of TuRBO1 at d 5/10 (+0.042, +0.021, both
    5/5).
  * It loses on `aocc_time` because its time/evaluation ratio drops to
    0.63 / 0.43 / 0.49 (d 2/5/10), while qLogEI's stays at 0.83 / 0.94.
    Blocks spends its evaluations over far more virtual time than the
    ideal budget/q, which points at idle workers.
  * The hypothesis is that one block-owning generational arm (CMA-ES
    λ ≈ 4 + 3 ln d, jSO's population) cannot fill 64 free workers.  It
    ties in with the open TODO "Stale generations under a capped
    bandit".
  * Per-run worker utilisation is not in the result files, so this is
    untested.

### 62.6 Costs (s/run on the runners)

The virtual clock advances only by evaluation durations.  Proposal time
is not on the clock, so the s/run below is overhead that `aocc_time`
ignores, and ignoring it favours the GP baselines.

| | range over the grid | worst cell |
|---|---|---|
| panobbgo specs (all four) | 0.0–0.8 s | d10/100·d/q64 0.8 s |
| pycma IPOP/BIPOP, Optuna CmaEs | 0.0–3.4 s | Optuna CmaEs d10/100·d/q64 |
| Optuna TPE, NGOpt, Py-BOBYQA | 0.1–16 s | TPE d10/100·d/q64 15.8 s |
| TuRBO-1 | 0.9–576 s | d10/100·d/q1 (falls with q: 204 / 98 / 75 s) |
| SMAC (q = 1) | 27–1061 s | d10/20·d |
| BoTorch qLogEI | 22–3218 s | d5/100·d/q64 (54 min per run, 6.4 s per evaluation) |

* qLogEI's cost grows with q: d5/100·d is 1599 / 1615 / 1862 / 3218 s at
  q 1/4/16/64.  The q = 64 / q = 1 factor is 2.0 at d = 5 and 3.4 at
  d = 2 (755 / 224 s); the plan's guessed factor `1 + 0.05 (q − 1)` gives
  4.15.
* Its proposal cost (up to ~6 s per point) matters only when one
  evaluation takes less than a few times that.
* panobbgo is 2–4 orders of magnitude cheaper to run than the GP tools.

### 62.7 Reading

* **This is not yet the roadmap claim.**  Pre-active and on the free
  families, the flagship portfolio is:
  * behind the pool's best at q = 1 in all six cells (three of them
    Holm-significant);
  * at parity with or ahead of the best BO tool at moderate parallelism
    and 100·d (q 4–16, d 2/5/10: six cells between −0.004 and +0.027,
    one Holm win);
  * clearly behind qLogEI at 20·d (three Holm losses at q > 1) and at
    d5/100·d/q64 on the time axis.
  The "never much worse on any class" half fails at q = 1 (ellipsoid vs
  Py-BOBYQA, sharp_ridge vs NGOpt).  The "clearly better on average"
  half holds only in the 100·d, q 4–16 band.
* **Where it wins is where the thesis says it should.**  Async dispatch
  beats the batch-synchronous or sequential tools (TuRBO1, Py-BOBYQA)
  as q grows.  At q > 1 it is ahead of every cheap-track incumbent
  (pycma, Optuna CmaEs/TPE, NGOpt) in nearly every cell.
* **qLogEI is the real rival on the primary track**, and at 20·d it
  leads at every d and q.  It fits a GP to 40–200 points and conditions
  on pending points; panobbgo has no model at 20·d.
* **The q = 64 deficit is probably an engineering issue** (worker
  utilisation), not an algorithmic one.  AOCC over evaluations says the
  search itself is ahead.
* Caveats:
  * 5 seeds and 3 instances per family;
  * one preset, with no failures yet;
  * Holm only on the headline;
  * a best-of-pool reference that is biased against panobbgo;
  * pre-active CMA-ES.
  The 13 unresolved cells need more seeds, not a reading.

### 62.8 Next steps (recommended, in order)

1. **Re-run the same grid on master (active-CMA default)** with 5 seeds
   (about 90 runner-hours).  Every panobbgo row here is pre-active, and
   the regime gate's d = 10 choice and the portfolio-vs-CMA-ES gap at
   100·d may both move.  This becomes the reference to build on.  Before
   it, raise SMAC's d = 10 cost estimate in `LAPTOP_SECONDS` (measured
   1061 s/run) so `plan` gives it its own shard.
2. **Diagnose q = 64 before measuring more of it.**
   * Log per-run worker utilisation (mean busy workers / q) and candidate
     age on the virtual clock for Blocks at d5/100·d/q64.
   * If idle workers explain the 0.43 time/AOCC ratio, fix the block
     strategy's dispatch.  Options: let a non-owning arm or a
     space-filling fallback fill the idle workers, or size the owning
     arm's generation to the free workers.
   * Then re-measure only the q = 64 cells.  q = 64 at d 5/10 with more
     seeds makes sense only after that.
3. **More seeds on the decisive cells** (12 seeds, core group plus
   qLogEI/TuRBO1 only):
   * the 100·d, q 4–16 band (d2/q4, d5/q4, d10/q16), where the claim is
     decided and five seeds cannot separate;
   * d10/100·d/q64 (p_holm 0.058).
4. Then the `failure` preset (d 2/5), which is roadmap §5 step 3's
   trigger for mechanism D.

## 63. Idle workers at q ≥ 16: measured, and a λ ≥ q floor for CMA-ES (2026-09-27)

§62.5 guessed that `Blocks_warm_CMAES_JSO` loses to qLogEI at q = 64
because its workers sit idle: its `aocc_time`/AOCC ratio is 0.43 at
d5/100·d/q64, qLogEI's 0.94.  This section measures that and fixes most of
it.

**Setup.**  The `measure.py` core configuration, run locally on master
3ce8ee0 (active-CMA default): free preset, 100·d, virtual clock with the
async policy, log-normal durations σ 0.5, common random numbers per cell,
`sync_eval`.  Cells d ∈ {2, 5} × q ∈ {4, 16, 64}, seeds 3/7/42/1234/2025,
all 5 families × 3 instances: 75 runs per cell, the §62 cells exactly.  A
probe (`sketchpad/worker_utilisation_probe.py`) wraps
`VirtualClock.step`/`_advance`.  It records the busy workers between
completions, what was asked (`request_cap`) against what was returned, and
the block owner.  Blocks reproduces §62's `aocc_time` means within 0.001 in
all six cells.  §62 was pre-active, so the active default barely moves
Blocks here.  Tables and paired deltas:
`planning/results/2026-09-27-worker-utilisation/summary.md`.  Deltas are
paired over the 5 seeds, with a t-CI95 and wins/5, unadjusted.

### 63.1 Utilisation before the fix

"busy" is busy worker-time / (q · makespan).  "busy@H" is the same up to
the horizon B/q, the part `aocc_time` scores.  "starved" is idle
worker-time while budget was left, "tail" idle time after the whole budget
was dispatched.  "mk/ideal" is makespan / (B/q).  "empty" is the share of
decision points (a free worker, budget left) where the strategy returned
nothing.

| spec | cell | busy | busy@H | starved | tail | mk/ideal | empty | aocc_time |
|---|---|---|---|---|---|---|---|---|
| Blocks | d2 q4 | 0.99 | 0.98 | 0.00 | 0.01 | 1.01 | 1 % | 0.214 |
| Blocks | d2 q16 | 0.89 | 0.97 | 0.02 | 0.09 | 1.12 | 10 % | 0.171 |
| Blocks | d2 q64 | 0.55 | 0.76 | 0.13 | 0.33 | 1.85 | 33 % | 0.093 |
| Blocks | d5 q4 | 1.00 | 0.99 | 0.00 | 0.01 | 1.01 | 1 % | 0.116 |
| Blocks | d5 q16 | 0.80 | 0.86 | 0.17 | 0.03 | 1.25 | 49 % | 0.098 |
| Blocks | d5 q64 | **0.25** | **0.35** | **0.70** | 0.06 | **4.03** | 82 % | 0.043 |
| RoundRobin_CMAES | d2 q16 / q64 | 0.44 / 0.11 | 0.45 / 0.11 | 0.52 / 0.84 | 0.04 / 0.05 | 2.29 / 9.16 | 85 % | 0.108 / 0.061 |
| RoundRobin_CMAES | d5 q16 / q64 | 0.58 / 0.15 | 0.59 / 0.15 | 0.39 / 0.82 | 0.03 / 0.03 | 1.72 / 6.87 | 89 % | 0.050 / 0.031 |

* **The hypothesis holds.**  At d5/q64, Blocks keeps a quarter of its
  workers busy and never fills all 64 in any run.  70 % of worker-time is
  idle while budget is left.  The run takes 4.0× the ideal time, so only
  about a third of its evaluations complete inside the scored horizon.
  That is the 0.43 ratio of §62.5.
* **Source: the arms run out of points.**  The strategy is asked for the
  free workers (`request_cap`).  The safety net `admit` trimmed nothing in
  all 450 runs, and nothing is lost to dedup or rejection.  At 82 % of the
  decision points the owning arm simply has nothing queued:
  * CMA-ES emits λ = 8 (d = 5) and emits the next generation only when
    λ/2 results are in, so at most about 1.5 λ = 12 points are in flight;
  * jSO has one trial per population slot (NP_init "auto" = 10 at d5/500);
  * blocks are 10 evaluations at d = 5 (4 at d = 2), with a hard cap of 2×
    that.
  While CMA-ES owns a block, 16.8 of the 64 workers are busy on average;
  while jSO does, 16.4 (the old owner's calls still in flight included).
* **d2/q64 is mostly the structural tail**: 200 evaluations on 64 workers
  is about 3 rounds, and 33 % of worker-time is the log-normal tail after
  the last dispatch.  qLogEI has the same tail.
* `RoundRobin_CMAES` is worse still: 11–15 % busy at q = 64, about 7–12
  workers used whatever q is (1.2–1.5 λ).  At d = 2 its AOCC is identical
  at q16 and q64, since the run is the same sequence stretched in time.  In
  Blocks jSO's population and the fast block switches add some workers.

### 63.2 Candidate fixes (prototypes, same cells)

Δ `aocc_time` against master (paired, 5 seeds; the q = 4 cells are
unchanged by every variant except λ ≥ 2q):

| variant | d2 q16 | d2 q64 | d5 q16 | d5 q64 |
|---|---|---|---|---|
| CMA λ ≥ q, warm-start fit at the λ ≥ q size | −0.010 [−0.021, +0.001] 0/5 | −0.002 | −0.001 | +0.007 [+0.004, +0.009] 5/5 |
| **CMA λ ≥ q, warm-start fit at the serial size (shipped, before the budget cap of §63.3)** | **−0.012 [−0.024, −0.000] 1/5** | +0.006 [−0.008, +0.019] 2/5 | +0.005 [−0.002, +0.011] 5/5 | **+0.016 [+0.012, +0.021] 5/5** |
| CMA λ ≥ q/2 | −0.011 [−0.021, −0.001] 1/5 | +0.001 | 0 (λ = 8 already) | +0.010 [+0.006, +0.015] 5/5 |
| CMA λ ≥ 2q | −0.015 (and d2 q4 −0.015) | −0.002 | −0.012 | +0.006 |
| CMA λ ≥ q + jSO NP_init ≥ q | −0.026 [−0.033, −0.019] 0/5 | +0.013 | −0.009 | +0.006 |
| CMA λ ≥ q + fill idle workers from the non-owning arm | −0.010 | +0.016 | −0.004 | +0.010 |
| fill from the non-owning arm only | −0.004 | −0.001 | −0.008 [−0.014, −0.003] 0/5 | +0.001 |

A block length of at least q (or 2q) evaluations on top of λ ≥ q and
NP ≥ q was as good or worse than without it (a smaller run, 3 seeds ×
5 families, instance 0; d5/q64 0.042 against 0.047).

* **Filling the workers is not free.**  More samples per generation means
  fewer generations per evaluation.  Every variant loses AOCC over
  evaluations: the shipped one −0.025 at d5/q64 (−0.027 uncapped), −0.018 at d2/q16.  It
  pays on the time axis only where many workers were idle.
* **The warm-start fit has to stay at the serial size.**  With λ = 64 the
  hand-off fitted the mean to the archive's top 64 (μ = 32) instead of its
  top 8.  That is a far less greedy hand-off.  Keeping the fit at the
  serial λ is worth +0.007 [+0.004, +0.011] (d2/q64) and +0.010
  [+0.006, +0.013] (d5/q64), both 5/5, against the same floor without it.
* **Scaling jSO or filling from the other arm does not help.**  jSO's NP
  and the paused arm's stale queue cost at q = 16 and add little at q = 64.
  Longer blocks lose too.  None of them ships.

### 63.3 The fix and its effect

**This is an in-sample choice.**  The shipped variant was picked from
seven prototypes (§63.2) on the same 450 runs it is evaluated on here, so
the deltas below are optimistic.  The confirmatory test is the
`measure.yml` re-run of the q ≥ 16 cells, on fresh runner runs against
the externals.

`CMAES(popsize_min_workers="auto")` is the new default:

* **What it does.**  It raises the default base λ to the parallel worker
  count, `strategy._n_evaluators()`, **on the virtual clock only**.
* **Where the rule comes from.**  It is the rule the pycma baseline
  adapter already uses (`popsize0 = max(4 + 3 ln n, q)` in
  `harness_baselines`).
* **Budget cap.**  The raised λ is capped at `max(λ_default,
  max_eval // 10)`, so a run keeps at least 10 generations
  (`CMAES.MIN_GENERATIONS`).  This is a convention, not tuned: 10 is the
  shortest window the module's own stagnation tests assume.
* **IPOP/BIPOP and explicit `popsize`.**  IPOP and BIPOP grow λ from the
  raised base.  An explicit `popsize` is never raised.
* **Real pools are off by default (`"auto"`).**  With a cheap objective on
  many threads, or a dask cluster of hundreds of workers, λ = q spends
  sample efficiency and buys no time.  It also stretches the stagnation
  window (`10·λ`) and makes BIPOP's small regime no longer small.  Only
  the virtual clock was measured here.  `True` forces the floor on any
  backend, `False` gives the pre-§63 behaviour, and a raised λ is logged
  at INFO with the reason.
* **Warm start.**  It fits `k = max(λ_serial, 4 + 3 ln n)` seeds with
  μ = λ_serial/2, where λ_serial is λ without the floor.
* **Where the floor does not bind** (q ≤ the default λ: 6 / 8 / 10 at
  d 2 / 5 / 10, so q = 1 and q = 4 everywhere, and every real backend
  under `"auto"`), runs are bit-identical to master.  Two tests pin that,
  one of them with Blocks' archive warm starts at the boundary d = 5,
  q = 8.  With the floor off, the probe reproduces master exactly on all
  450 runs.

Before → after (the shipped, capped version), the same 75 runs per cell.
The cap binds only at q = 64 at d ≤ 5 (λ 64 → 20 at d = 2 and → 50 at
d = 5):

| spec | cell | busy@H | mk/ideal | aocc_time | Δ aocc_time [CI95] wins | Δ AOCC |
|---|---|---|---|---|---|---|
| Blocks | d2 q16 | 0.97 → 0.98 | 1.12 → 1.10 | 0.171 → 0.159 | **−0.012 [−0.024, −0.000] 1/5** | **−0.018 [−0.032, −0.005] 0/5** |
| Blocks | d2 q64 | 0.76 → 0.78 | 1.85 → 1.83 | 0.093 → 0.099 | +0.007 [−0.004, +0.017] 4/5 | +0.001 |
| Blocks | d5 q16 | 0.86 → 0.92 | 1.25 → 1.15 | 0.098 → 0.103 | +0.005 [−0.002, +0.011] 5/5 | −0.002 |
| Blocks | d5 q64 | **0.35 → 0.86** | **4.03 → 1.51** | **0.043 → 0.061** | **+0.018 [+0.015, +0.020] 5/5** | −0.025 [−0.036, −0.014] 0/5 |
| RoundRobin_CMAES | d2 q16 | 0.45 → 0.92 | 2.29 → 1.18 | 0.108 → 0.137 | +0.029 [+0.021, +0.038] 5/5 | +0.002 |
| RoundRobin_CMAES | d2 q64 | 0.11 → 0.34 | 9.16 → 3.27 | 0.061 → 0.081 | +0.020 [+0.013, +0.027] 5/5 | −0.012 |
| RoundRobin_CMAES | d5 q16 | 0.59 → 0.92 | 1.72 → 1.13 | 0.050 → 0.064 | +0.014 [+0.007, +0.022] 5/5 | +0.002 |
| RoundRobin_CMAES | d5 q64 | 0.15 → 0.82 | 6.87 → 1.46 | 0.031 → 0.044 | +0.013 [+0.012, +0.015] 5/5 | −0.016 |

**Effect of the budget cap** (capped against the uncapped λ = q; the
cells where it does not bind are identical):

* Blocks: d2/q64 +0.001, d5/q64 +0.002 (both n.s.).  Utilisation does
  not change there, because Blocks' block rule limits a CMA-ES block to
  20 dispatches anyway.
* `RoundRobin_CMAES` d2/q64 pays for it: +0.041 → +0.020, busy@H 0.89 →
  0.34.  λ = 20 cannot fill 64 workers, while 200 evaluations at λ = 64
  are only about 3 generations.
* The cap therefore trades time-axis speed for generations in the one
  cell where it binds hard.  It is kept on review request (sample
  efficiency on budgets too small for λ = q).  It is a candidate to
  revisit with the `measure.yml` numbers.

d = 10 side check (100·d, 5 seeds × 5 families, instance 0 only; the cap
does not bind at 1000 evaluations):

| spec | q16 Δ aocc_time | q64 Δ aocc_time |
|---|---|---|
| Blocks | +0.003 [−0.011, +0.017] 3/5 | +0.013 [+0.008, +0.017] 5/5 (busy@H 0.46 → 0.81) |
| RoundRobin_CMAES | +0.004 [+0.001, +0.008] 5/5 | +0.004 [+0.002, +0.006] 5/5 |
| RegimeGate_oracle (CMA-ES alone at d = 10) | −0.006 [−0.029, +0.017] 2/5 | +0.009 [+0.005, +0.013] 5/5 |

Against §62's externals on the same cells (the externals' values from the
runner run, the same seeds, instances and CRN durations; indicative, not
paired):

* d5/q64: Blocks 0.061 against qLogEI 0.062.  §62's −0.019 Holm loss
  shrinks to about −0.001.
* d2/q64: 0.099 against qLogEI's 0.113 (was −0.020, now about −0.014).
* d5/q16: 0.103 against qLogEI's 0.072 (the §62 win grows).
* d2/q16: 0.159 against qLogEI's 0.175.  It was −0.004, now about −0.016;
  **this is the fix's cost.**

### 63.4 Reading

* **§62.5's idle-worker hypothesis is confirmed.**  At d5/q64 the
  portfolio used 25 % of its workers, because its arms emit about 1.5 λ
  or NP points and then wait.  With the floor it uses 86 % inside the
  horizon.  Most of the d5/q64 gap to qLogEI (0.018 of 0.019, in-sample)
  was scheduling.  At d2/q64 most of the gap is not, because the budget
  is only 3 rounds.
* **One cost remains: d2/q16 for Blocks.**  `aocc_time` falls by 0.012,
  1/5 seeds, and its unadjusted CI [−0.024, −0.000] excludes 0.  AOCC over
  evaluations falls by 0.018 [−0.032, −0.005], 0/5.  The reason:
  * without the floor the portfolio already filled 97 % of the horizon,
    with 6-point generations in 4-evaluation blocks that switch fast;
  * so a bigger λ only changes the search: by the block rule CMA-ES
    dispatches at most 8 of each 16-point generation before its block
    closes;
  * `RoundRobin_CMAES` gains +0.029 in the same cell.
* **The flagship `RoundRobin_CMAES` gains at every q ≥ 16**
  (+0.004…+0.029, 5/5 in all six cells, d 2/5/10).  Before the fix it
  used 7–12 workers whatever q was.
* AOCC over evaluations falls at q = 64 (−0.012…−0.033) and at Blocks
  d2/q16 (−0.018): more points per generation means fewer generations per
  evaluation.  At q > 1 the headline metric is `aocc_time`, the one that
  improves (except at Blocks d2/q16).
* **Not measured:**
  * 20·d cells and the failure preset;
  * the pycma/Optuna externals re-run;
  * the 500·d batteries — they run at q = 1 or on real pools, where the
    floor does not bind, so they are unchanged by construction;
  * real threaded or dask async with many workers, where the floor is off
    by default.  Whether it should be on there for expensive objectives
    is an open question (TODO).
* **Next:**
  * Re-measure the q ≥ 16 cells of `measure.yml` with the floor (§62.8
    step 2).  That is the confirmatory test.
  * A Blocks-specific follow-up for d2/q16: size the block to at least
    one generation of the owning arm, or keep the floor only where the
    arms cannot fill the workers.  Both are open, not tried.
* **These numbers depend on CMA-ES's half quorum (the §64 work).**
  `_update_if_quorum` closes a generation after its first μ = λ/2
  results, emits the next generation, and drops the other λ/2 results when
  they arrive.  Those results are evaluated and charged but never ranked,
  and active CMA's negative weights never get them.  Two things here rest
  on that rule:
  * **The idle-worker bound.**  "About 1.5 λ in flight" and the
    before-fix utilisation (25 % at d5/q64) come from emitting the next
    generation at λ/2.  With a full-λ quorum, CMA-ES alone would keep at
    most λ points in flight and idle more.
  * **The after-fix utilisation.**  With λ = q the half quorum pipelines
    generations: the next q points are queued while the late half of the
    last generation is still running.  A barrier on all λ results would
    idle workers during each generation's log-normal tail.  So a quorum
    change must be re-measured with this floor, not assumed to keep
    busy@H at 0.86–0.92.
  * **Not affected:** the λ ≥ q floor itself and the serial-size warm-start
    fit.  They only set λ and the hand-off's k/μ, and do not touch
    `_update_if_quorum`, so they compose with any quorum rule.
  * The AOCC-over-evaluations loss at q = 64 (−0.02…−0.03) is partly
    this effect: with λ = 64, 32 evaluated results per generation are
    thrown away.  A quorum fix that ranks late results may win some of it
    back.
* §62.2's "unexplained" gap between CMA-ES alone inside Blocks (0.079)
  and under RoundRobin (0.037) at d10/q16 may partly be this: RoundRobin
  used 12 of 16 workers there (busy@H 0.73).  The block strategy also
  warm-starts CMA-ES whenever its queue is empty at a block boundary.
  That second part is a hypothesis, not measured.

## 64. CMA-ES's half quorum: active CMA never ran on the virtual clock, and late offspring were dropped (2026-09-27)

**The contradiction.**  #375 made guarded active CMA the default (§60/§61),
and the cheap-track re-baseline moved.  But `measure.yml` gave
bit-identical per-run traces for every core unit before and after it:
run 36274781342 (1778e1b) against run 36305967615 (3ce8ee0).  For
example, d5/100·d/q4/seed 3 gives `RoundRobin_CMAES` 0.06453350454091397
and `Blocks_warm_CMAES_JSO` 0.13112648955297093 in both runs.  The
harness builds `CMAES` with the default kwargs, and the workflow checks
out the recorded sha.  Neither explains it.

**Cause.**  `_update_if_quorum` closes a generation once
`max(2, int(λ · min_results_fraction))` results are in (default 0.5,
i.e. μ).  Under `evaluation.sync` without the virtual clock, a
generation of up to 10 points arrives in one batch (`StrategyRoundRobin`
asks for 10), so all λ are ranked.  On the virtual clock (async pull), and
on the real async loop, results arrive one at a time, so the update fires
on the **first μ arrivals**:

1. the ranked set has exactly μ entries, so there is no rank μ+1…λ and
   active CMA's negative update never runs (`active=True` is bit-identical
   to `active=False`);
2. there is no μ-of-λ truncation selection: the μ first arrivals are all
   "selected" and only the rank weights order them;
3. the other λ − μ offspring are evaluated and charged, then ignored when
   they arrive (the generation's bucket is gone).  Half of CMA-ES's
   evaluations never reached its update, at every q including q = 1.

**Evidence** (a counter around `CMAES._update`, scratch script; d5,
100·d, free preset, 15 runs, seed 3, `RoundRobin_CMAES`):

| path | generations | ranked / emitted | active applied | skipped (repaired) | AOCC |
|---|---|---|---|---|---|
| virtual async q = 4 (measure) | 945 | 3780 / 7560 | 0 | 0 | 0.06453350454091397 |
| virtual async q = 4, `active=False` | 945 | 3780 / 7560 | — | — | 0.06453350454091397 |
| virtual async q = 1 | 945 | 3780 / 7560 | 0 | 0 | 0.0710 |
| `sync`, no clock (cheap track) | 945 | 7500 / 7560 | 782 | 148 | 0.11091832778914675 |

The last 15 generations of the sync run are cut by the budget.  At 20·d
the cheap track applies the active update too (114 of 195 generations,
66 skipped by the repair guard), so §61's gain is not a large-budget
effect.  The two runs reproduce the runner's numbers bit for bit.

**The same drop on the cheap track, after IPOP growth.**  A generation
larger than RoundRobin's 10-point request arrives in more than one sync
batch, and the half quorum closes it early.  At d5/500·d/seed 42,
`RoundRobin_CMAES` ranked 35766 of 37512 emitted offspring, with 89
generations at ≤ μ.  At d ≤ 10 the default λ ≤ 10 fits one batch, so only
runs with a larger λ are affected: after an IPOP restart, in BIPOP's large
regime, or with an explicit `popsize` > 10.

### 64.1 The fix

* **Quorum rule (`quorum="dispatched"`, new default; `"fraction"` is the
  old rule).**  A generation may close at `min_results_fraction · λ`
  only once none of its offspring still waits in CMA-ES's own output
  queue (arrived + in flight = emitted).  Until then it waits for the
  whole generation.  Closing early pays only when it lets the next
  generation fill workers that would otherwise idle.  While the
  generation still has queued points the workers have work, and an early
  close only ranks the first arrivals.  Special cases:
  * one worker: every result but the last arrives with points still
    queued, so the whole generation is ranked.  Virtual q = 1 reproduces
    the synchronous trajectory exactly (0.11091832778914675 above; pinned
    in `tests/test_cma_es_quorum.py`);
  * q = 2 against λ = 8 closes after about λ − 2 results;
  * synchronous batches smaller than λ.  The IPOP-grown generations above
    now wait for their second batch, and nothing arrives late on the
    cheap track.
  A first version of this PR used "one worker ⇒ full generation"
  (`_n_evaluators() == 1`).  That version is identical to the new rule at
  q = 1 and at q ≥ 16 in every run below, and within noise at q = 2/4
  (§64.2).
* **Late offspring (`late_results="fold"`, new default; `"drop"` is the
  old behaviour, and `"fold_capped"` keeps at most λ//4, best-ranked).**
  A late result of an early-closed generation joins the next update as
  an injected point:
  * its *evaluated* position is used (`x_eval`, not a repaired `x`),
    converted to the current mean, σ and C, and Mahalanobis-clipped to
    `c_y` (`_clip_injected`, as `inject=True` does);
  * it is added after the quorum check, so it never counts toward the
    quorum;
  * it gets no negative weight under `active_skip_repaired`;
  * it is not charged again.

  Re-ranking the stored step with the next generation was rejected
  because that `y` was drawn from the old distribution.  Counters:
  `n_late_folded`, `n_late_dropped`.  A result for a *flushed* generation
  (restart, warm start) is ignored silently and counted in neither.
* **Rejected: a full-λ barrier at q > 1** (`min_results_fraction = 1`).
  On seed 3 at d5/q4, `RoundRobin_CMAES` `aocc_time` reached 0.088 (fold
  0.105), and `Blocks_warm_CMAES_JSO` fell 0.132 → 0.107.  With #379's
  λ ≥ q floor (§63), the early close is what keeps the next generation
  queued, and fold keeps that pipelining.

### 64.2 A/B on the measure path

Run locally with the `measure.py` core settings: free preset, 100·d,
virtual async, log-normal σ 0.5, CRN per cell, 5 seeds (3, 7, 42, 1234,
2025), 15 instances per seed, both specs, on master a2c5f0e (#379: λ ≥ q
on the virtual clock, capped at `max_eval // 10`).  Δ is paired over the
seeds on the mean of the 15 runs, with a t-CI95 and wins/5, unadjusted.
"Old" is `quorum="fraction", late_results="drop"`.  The old arm at
q ≤ 4 reproduces run 36305967615 bit for bit (the floor does not bind
there).  "id." = bit-identical in every run.

AOCC, Δ against old (old's mean in brackets):

| cell | spec | dispatched + fold (default) | dispatched + fold_capped | dispatched + drop | capped − fold | dispatched − fraction, both fold |
|---|---|---|---|---|---|---|
| d2 q1 | RR (0.147) | +0.045 [+0.021, +0.069] 5/5 | = fold | = fold | id. | +0.010 [−0.009, +0.030] 4/5 |
| d5 q1 | RR (0.069) | +0.042 [+0.035, +0.049] 5/5 | = fold | = fold | id. | +0.001 [−0.007, +0.009] 2/5 |
| d10 q1 | RR (0.044) | +0.043 [+0.036, +0.051] 5/5 | = fold | = fold | id. | **−0.007 [−0.011, −0.003] 0/5** |
| d5 q2 | RR (0.063) | +0.051 [+0.046, +0.056] 5/5 | = fold | +0.041 [+0.035, +0.046] 5/5 | id. | +0.002 [−0.006, +0.009] 4/5 |
| d10 q2 | RR (0.044) | +0.044 [+0.042, +0.047] 5/5 | = fold | +0.039 [+0.035, +0.043] 5/5 | id. | −0.003 [−0.012, +0.006] 2/5 |
| d5 q4 | RR (0.067) | +0.044 [+0.038, +0.049] 5/5 | +0.046 [+0.043, +0.050] 5/5 | +0.021 [+0.017, +0.025] 5/5 | +0.003 [−0.001, +0.006] 4/5 | +0.003 [−0.005, +0.010] 4/5 |
| d10 q4 | RR (0.043) | +0.049 [+0.044, +0.054] 5/5 | +0.050 [+0.047, +0.053] 5/5 | +0.026 [+0.019, +0.032] 5/5 | +0.001 [−0.003, +0.005] 4/5 | −0.003 [−0.010, +0.004] 1/5 |
| d5 q16 | RR (0.069) | +0.016 [+0.013, +0.019] 5/5 | +0.015 [+0.011, +0.020] 5/5 | id. | −0.001 [−0.005, +0.002] 2/5 | id. |
| d10 q16 | RR (0.044) | +0.026 [+0.023, +0.029] 5/5 | +0.029 [+0.027, +0.031] 5/5 | id. | +0.003 [−0.001, +0.006] 5/5 | id. |
| d2 q64 | RR (0.133) | +0.016 [+0.008, +0.025] 5/5 | +0.012 [+0.002, +0.022] 5/5 | id. | −0.004 [−0.011, +0.003] 1/5 | id. |
| d5 q64 | RR (0.050) | +0.003 [+0.001, +0.006] 5/5 | +0.002 [−0.001, +0.006] 4/5 | id. | −0.001 [−0.004, +0.002] 2/5 | id. |
| d10 q64 | RR (0.030) | +0.001 [+0.000, +0.001] 5/5 | +0.001 [+0.000, +0.002] 5/5 | id. | +0.000 [−0.001, +0.001] 3/5 | id. |
| d2 q1 | Blocks (0.215) | +0.011 [−0.025, +0.046] 3/5 | = fold | = fold | id. | +0.011 [−0.025, +0.046] 3/5 |
| d5 q1 | Blocks (0.120) | +0.011 [−0.002, +0.024] 4/5 | = fold | = fold | id. | +0.002 [−0.012, +0.016] 3/5 |
| d10 q1 | Blocks (0.080) | +0.006 [−0.006, +0.019] 4/5 | = fold | = fold | id. | −0.005 [−0.014, +0.004] 1/5 |
| d5 q2 | Blocks (0.117) | +0.007 [−0.008, +0.022] 3/5 | = fold | = fold | id. | +0.004 [−0.007, +0.014] 3/5 |
| d10 q2 | Blocks (0.079) | +0.002 [−0.004, +0.009] 4/5 | = fold | = fold | id. | −0.006 [−0.020, +0.009] 2/5 |
| d5 q4 | Blocks (0.117) | +0.007 [−0.005, +0.020] 3/5 | = fold | = fold | id. | +0.003 [−0.003, +0.009] 4/5 |
| d10 q4 | Blocks (0.081) | +0.002 [−0.008, +0.013] 4/5 | = fold | = fold | id. | −0.005 [−0.020, +0.009] 1/5 |
| q16, d2 q64 | Blocks | id. | id. | id. | id. | id. |
| d5 q64 | Blocks (0.076) | −0.000 [−0.003, +0.003] 2/5 | = fold | = fold | id. | = |
| d10 q64 | Blocks (0.055) | −0.000 [−0.001, +0.001] 1/5 | = fold | = fold | id. | = |

`aocc_time` has the same pattern.  Default against old:
* `RoundRobin_CMAES`: +0.042…+0.050 at q ≤ 4 (5/5), +0.014 / +0.023 at
  d5 / d10 q16 (5/5), +0.001…+0.003 at q64 (4–5/5, CIs touching 0).
* `Blocks_warm_CMAES_JSO`: +0.002…+0.011 at q ≤ 4, and −0.0004…0 at q64
  (CIs spanning 0).

Blocks is bit-identical at q16 and d2/q64, and its q64 deltas come from
the quorum rule alone, not from fold.

**σ and p_σ** (mean over all updates of the 5 seeds × 15 runs: log10
σ/σ0 of the current run, ‖p_σ‖/χ_n, and the share of the μ selected
slots taken by late points; `RoundRobin_CMAES`):

| cell | old | dispatched + drop | dispatched + fold | dispatched + fold_capped |
|---|---|---|---|---|
| d5 q2 | −0.47 / 0.88 / — | −0.73 / 0.80 / — | −0.81 / 0.77 / 10 % | = fold |
| d5 q4 | −0.47 / 0.87 / — | −0.61 / 0.83 / — | −0.87 / 0.76 / 32 % | −0.88 / 0.75 / 31 % |
| d10 q4 | −0.56 / 0.89 / — | −0.77 / 0.84 / — | −1.14 / 0.76 / 25 % | −1.11 / 0.77 / 24 % |
| d5 q16 | −0.24 / 0.87 / — | = old | −0.38 / 0.80 / 42 % | −0.38 / 0.80 / 40 % |
| d10 q16 | −0.40 / 0.88 / — | −0.40 / 0.88 / — | −0.73 / 0.77 / 42 % | −0.74 / 0.77 / 40 % |
| d2 q64 | −0.01 / 0.94 / — | = old | −0.01 / 0.91 / 38 % | −0.01 / 0.90 / 36 % |
| d10 q64 | +0.08 / 0.98 / — | +0.08 / 0.99 / — | +0.08 / 0.97 / 40 % | +0.08 / 0.97 / 39 % |

**Cheap track** (sync, families preset at 500·d, d 2/5/10, the same 5
seeds, old → default): `RoundRobin_CMAES` +0.0019 [−0.0001, +0.0038]
4/5, −0.0002 [−0.0013, +0.0008] 3/5, +0.0003 [+0.0000, +0.0005] 5/5.
`Blocks_warm_CMAES_JSO` is identical in every seed.  No point is folded
on this path.

### 64.3 Reading

* **§62's CMA-ES numbers measure a crippled CMA-ES.**  Every panobbgo
  spec on the expensive track ran with half of CMA-ES's evaluations
  unranked, no truncation selection and no active update, pre- and
  post-#375 alike.  The default gains `RoundRobin_CMAES` +0.04…+0.05 AOCC
  at q ≤ 4, +0.016 / +0.026 at d5 / d10 q16, and +0.001…+0.016 at q64,
  5/5 in every cell.
* **Both parts carry weight.**  The quorum rule alone (dispatched + drop)
  is +0.021…+0.045 at q ≤ 4.  Fold adds +0.005…+0.023 at q = 2/4 and all
  of the gain at q ≥ 16, where the rule changes nothing (λ = q
  generations are dispatched at once).
* **The rule against the first version (fraction + fold at q > 1, full
  generation at q = 1)** is identical at q = 1 and q ≥ 16, and within
  ±0.006 at q = 2/4 with every CI spanning 0.  So it is kept as the
  principled rule.
  * **One contradiction worth a follow-up:** at q = 1, closing at μ and
    folding (fraction + fold) *beats* ranking the whole generation at
    d10: −0.007 [−0.011, −0.003] 0/5 for the full generation, and Blocks
    −0.005 1/5.
  * At d2 it is the other way round (+0.010 4/5), and at d5 even.
  * So at d10 the pipelined, σ-shrinking fold helps even without
    parallelism.  That looks like a step-size effect, not a scheduling
    one.
* **Fold biases σ down, as predicted.**  Late steps are measured from
  the moved mean (the mean shift is not clipped, so a late `y` carries
  about −Δm/σ).  ‖p_σ‖/χ_n falls from 0.83–0.88 to 0.76–0.80 at q = 4–16,
  and mean σ runs 1.4–2.4× smaller than under dispatched + drop
  (log10 −0.14…−0.37).  AOCC gains
  anyway: on these unimodal-leaning free families a smaller σ is not
  costly at 100·d.  On multimodal problems at larger budgets it could
  be.  Unmeasured.
* **The cap does not change the volume that matters.**  λ//4 best-ranked
  keeps about 40 % of the selected slots late (42 % → 40 % at q16),
  because the best late points are the ones that get selected anyway.
  Its AOCC is the same as uncapped (−0.004…+0.003, every CI spanning 0).
  Uncapped `"fold"` stays the default; `"fold_capped"` stays available.
  A cap that bounds the selected share would have to pick
  earliest-arrived or random late points, or down-weight them.  Untried.
* **Blocks barely depends on CMA-ES's own updates here.**  Its blocks
  are 10 evaluations at d = 5 (n_blocks = 50), and it warm-starts on
  every re-acquisition.  A CMA-ES block is about one generation fitted
  from the archive, so the update rule reaches it only when a block
  spans a generation boundary (q ≤ 4).
* The termination histories (TolFun, stagnation) now mix late points
  from the previous distribution with the current offspring, as
  `inject=True` already did.  Not checked separately.
* **Comparability.**
  * Expensive-track (`measure.yml`) numbers of every spec with a CMA-ES
    arm change.
  * Cheap-track references (`rebaseline-2026-09-27`, §61) change only for
    CMA-ES runs whose λ exceeds one synchronous request batch: after an
    IPOP restart, in BIPOP's large regime, or with an explicit `popsize`
    > 10.  That covers `RoundRobin_CMAES`, `CMAES_alone`, the composite
    registry's CMA-ES entries, and `RegimeGate_oracle` where it runs
    CMA-ES alone.  `RoundRobin_CMAES` moves −0.0002…+0.0019 on the
    families; Blocks is unchanged.
  * The pre-§64 pins keep `quorum="fraction", late_results="drop"`
    (`tests/test_cma_es_bounds_active.py`).
* **Not measured:** 20·d cells, the failure preset, MA-BBOB/BBOB at
  500·d, and real (threaded/dask) async, where the same rules apply.
* **Next:** re-run the `measure.yml` grid on this; look at the d10/q1
  step-size effect above.

## 65. The expensive track after the CMA-ES fixes (#375, #379, #380): the d5/q64 loss is gone, the 20·d and q = 1 losses stay (2026-09-27)

**Run.** `measure.yml` run 36313485264 on master 58cf1e7, the `core`
group only, full default grid (free preset, d 2/5/10, 20·d and 100·d,
q ∈ {1, 4, 16, 64}, q ≤ bm, seeds 3/7/42/1234/2025, 3 instances per
family).  That is 105 units in 5 shards, all done, none failed, about
1.4 runner-hours and 26 min wall.  The code under test has
guarded active CMA as the default (#375, §60/§61), the λ ≥ q floor
`popsize_min_workers="auto"` with its budget cap (#379, §63), and the
dispatched quorum with late offspring folded (#380, §64).  §62 measured
none of the three: it ran on 1778e1b, before all of them.  Run
36305967615 on 3ce8ee0 had #375, but the half quorum made active CMA a
no-op there (§64).

**The comparison is a combination of runs.**  The GP groups were not
re-run.  The aggregate combines:

* the 105 new `core` units (the four panobbgo specs plus pycma
  IPOP/BIPOP, NGOpt, Optuna CmaEs/TPE, Py-BOBYQA);
* the qLogEI, TuRBO-1 and SMAC units of §62's run 36274781342 (1778e1b);
* the two SMAC units that §62 lost (`SMAC.free.b100.q1.d2.s3/.s7`),
  re-run in 36305967615 (3ce8ee0).

Why the combination is valid:

* Each unit is a deterministic function of (spec, seed, instance), with
  CRN durations per cell.
* `harness_baselines_bo.py` is unchanged between 1778e1b and 58cf1e7.
  `harness_baselines.py` only gained a new Optuna variant.
* All 325 unit files carry the same `fp_env_id` 80ee2a0090c4.
* The strongest check: every core-group external and `RoundRobin_Random`
  is **bit-identical** to §62 in all 21 × 75 runs.  The new runner
  reproduces the old code path exactly.  Only CMA-ES-bearing specs move.

The combined `summary.md` still shows SMAC at d2/100·d as 3/5.  The two
re-run units enter it as calibration rows (AOCC 0.249 and 0.249), not
as grid units.  SMAC is a q = 1 reference row outside the pool, so no
headline number depends on it.

Files: `planning/results/2026-09-27-measure-confirm/` holds `summary.md`
(the combined aggregate, `scripts/measure.py aggregate` over the merged
unit files) and `plan.json` (the confirmation run's plan, 5 core
shards).  `summary.json` is not committed.  "Before" in this section
is §62 (run 36274781342).  The core units of run 36305967615 (3ce8ee0)
are bit-identical to it (§64).  The method is §62's: Holm over the 21
headline cells, everything else unadjusted, **bold** = p_holm < 0.05.

**Same seeds, so not out of sample.**  The seeds, instances and CRN
durations are §62's.  §63 and §64 chose their variants on those same
100·d cells locally, so wherever the runner repeats a local A/B cell it
reproduces the in-sample number.  It is not an independent confirmation.
The new information is:

* the paired comparison against the externals and the Holm context;
* the 20·d cells, which neither §63 nor §64 ran;
* instances 1 and 2 at d = 10 — but only relative to §63's floor side
  check, which used instance 0.  §64.2 ran all 15 instances at d = 10
  and chose #380 on them, so for the quorum and fold part these
  instances are in sample too.

### 65.1 Headline: `Blocks_warm_CMAES_JSO` − pool best, before and after

"Blocks after − before" is paired over the 5 seeds on the same 75 runs
(unadjusted).  The pool and the pool's best values are unchanged.

| cell | metric | pool best | Blocks before → after | Blocks after − before | Δ before (p_holm) | Δ after [CI95] wins | p_holm |
|---|---|---|---|---|---|---|---|
| d2 20·d q1 | aocc | TuRBO1 0.145 | 0.093 → 0.095 | +0.002 [−0.005, +0.009] 3/5 | **−0.053** (0.029) | **−0.051** [−0.073, −0.028] 0/5 | 0.043 |
| d2 20·d q4 | time | qLogEI 0.100 | 0.085 → 0.090 | +0.005 [−0.009, +0.019] 3/5 | −0.016 (0.093) | −0.011 [−0.022, +0.000] 1/5 | 0.328 |
| d2 20·d q16 | time | qLogEI 0.067 | 0.063 → 0.063 | identical (75/75 runs) | −0.003 (0.919) | −0.003 [−0.010, +0.003] 1/5 | 1.000 |
| d2 100·d q1 | aocc | PyBOBYQA 0.440 | 0.215 → 0.226 | +0.010 [−0.025, +0.046] 3/5 | **−0.224** (<0.001) | **−0.214** [−0.262, −0.166] 0/5 | 0.005 |
| d2 100·d q4 | time | qLogEI 0.209 | 0.214 → 0.203 | −0.010 [−0.037, +0.016] 2/5 | +0.005 (0.919) | −0.006 [−0.037, +0.026] 2/5 | 1.000 |
| d2 100·d q16 | time | qLogEI 0.175 | 0.171 → 0.159 | −0.012 [−0.024, −0.000] 1/5 | −0.004 (0.919) | −0.016 [−0.028, −0.005] 0/5 | 0.136 |
| d2 100·d q64 | time | qLogEI 0.113 | 0.093 → 0.099 | +0.007 [−0.004, +0.017] 4/5 | −0.020 (0.113) | −0.013 [−0.021, −0.006] 0/5 | 0.079 |
| d5 20·d q1 | aocc | NGOpt 0.073 | 0.044 → 0.044 | +0.000 [−0.002, +0.003] 3/5 | −0.029 (0.080) | −0.029 [−0.045, −0.013] 0/5 | 0.068 |
| d5 20·d q4 | time | qLogEI 0.049 | 0.044 → 0.044 | +0.000 [−0.001, +0.001] 2/5 | −0.005 (0.187) | −0.005 [−0.009, −0.001] 0/5 | 0.159 |
| d5 20·d q16 | time | qLogEI 0.040 | 0.034 → 0.035 | +0.001 [−0.000, +0.002] 4/5 | **−0.006** (0.019) | **−0.005** [−0.007, −0.004] 0/5 | 0.020 |
| d5 100·d q1 | aocc | PyBOBYQA 0.145 | 0.120 → 0.131 | +0.011 [−0.002, +0.024] 4/5 | −0.026 (0.402) | −0.015 [−0.047, +0.017] 2/5 | 1.000 |
| d5 100·d q4 | time | TuRBO1 0.112 | 0.116 → 0.123 | +0.007 [−0.005, +0.019] 3/5 | +0.004 (0.919) | +0.011 [+0.006, +0.017] 5/5 | 0.060 |
| d5 100·d q16 | time | qLogEI 0.072 | 0.098 → 0.103 | +0.005 [−0.002, +0.011] 5/5 | **+0.027** (0.008) | **+0.031** [+0.021, +0.041] 5/5 | 0.015 |
| d5 100·d q64 | time | qLogEI 0.062 | 0.043 → 0.061 | **+0.018 [+0.015, +0.021] 5/5** | **−0.019** (0.001) | −0.001 [−0.006, +0.005] 2/5 | 1.000 |
| d10 20·d q1 | aocc | NGOpt 0.051 | 0.028 → 0.028 | +0.000 [−0.000, +0.001] 3/5 | **−0.023** (0.007) | **−0.023** [−0.030, −0.017] 0/5 | 0.010 |
| d10 20·d q4 | time | qLogEI 0.034 | 0.027 → 0.028 | +0.001 [−0.001, +0.002] 4/5 | **−0.007** (0.002) | **−0.007** [−0.009, −0.005] 0/5 | 0.010 |
| d10 20·d q16 | time | qLogEI 0.031 | 0.026 → 0.026 | +0.001 [−0.000, +0.002] 5/5 | **−0.006** (0.008) | **−0.005** [−0.007, −0.003] 0/5 | 0.023 |
| d10 100·d q1 | aocc | TuRBO1 0.096 | 0.080 → 0.086 | +0.006 [−0.006, +0.019] 4/5 | −0.017 (0.093) | **−0.010** [−0.015, −0.006] 0/5 | 0.042 |
| d10 100·d q4 | time | TuRBO1 0.085 | 0.081 → 0.083 | +0.002 [−0.008, +0.013] 4/5 | −0.004 (0.706) | −0.002 [−0.012, +0.009] 2/5 | 1.000 |
| d10 100·d q16 | time | TuRBO1 0.060 | 0.067 → 0.073 | +0.006 [−0.001, +0.012] 4/5 | +0.007 (0.706) | **+0.013** [+0.007, +0.019] 5/5 | 0.049 |
| d10 100·d q64 | time | Optuna TPE 0.028 | 0.033 → 0.046 | **+0.013 [+0.013, +0.014] 5/5** | +0.005 (0.058) | **+0.018** [+0.016, +0.021] 5/5 | 0.001 |

(Bold in "after − before" marks the two unadjusted CIs above 0 that
matter here.  d2/100·d/q16's CI ends at −0.0001.)

### 65.2 Holm count: 1 / 7 / 13 → 3 / 7 / 11

| | wins | losses | unresolved |
|---|---|---|---|
| §62 (before) | 1: d5/100·d/q16 | 7: d2/20·d/q1, d2/100·d/q1, d5/20·d/q16, d10/20·d/q1, d10/20·d/q4, d10/20·d/q16, d5/100·d/q64 | 13 |
| §65 (after) | 3: d5/100·d/q16, **d10/100·d/q16**, **d10/100·d/q64** | 7: d2/20·d/q1, d2/100·d/q1, d5/20·d/q16, d10/20·d/q1, d10/20·d/q4, d10/20·d/q16, **d10/100·d/q1** | 11 |

* **One loss leaves and one enters.**
  * d5/100·d/q64 leaves: Blocks +0.018, and the Δ is now −0.001.
  * d10/100·d/q1 enters, although Blocks *gained* there (+0.006, n.s.).
    Its Δ went from −0.017 to −0.010, but the CI narrowed, so p fell
    from 0.0085 to 0.0030 and p_holm from 0.093 to 0.042.
* **Every remaining loss is at q = 1 or at 20·d.**  No Holm loss is
  left at 100·d with q > 1.  In §62 there was one (d5/q64).
* **Three verdicts sit at the edge:** d10/100·d/q16 (win, p_holm
  0.049), d10/100·d/q1 (loss, 0.042) and d2/20·d/q1 (loss, 0.043).
  Holm is step-down.  Adding small p-values elsewhere changes every
  adjustment: for example d2/20·d/q16 is bit-identical but moves
  0.919 → 1.000.  So this count moves with small changes in cells that
  have nothing to do with a given verdict.
* Near misses: d5/100·d/q4, +0.011 [+0.006, +0.017] 5/5, p_holm 0.060
  (was +0.004, 3/5), and d2/100·d/q64, −0.013, 0/5, 0.079.
* Blocks is first overall in 4 cells, as in §62, but not the same four:
  d5/100·d/q4 (tied with `RegimeGate_oracle`) and q16, d10/100·d/q16 and
  q64.  It gains d10/q16 and loses d2/100·d/q4 (0.203 against qLogEI's
  0.209; it was 0.214).  Ahead of every external it is in 4 cells now
  against 5 in §62 (d10/100·d/q16 was already ahead of the externals,
  behind `RegimeGate_oracle`).  At d10/100·d, q ≤ 4, `RoundRobin_CMAES` and
  `RegimeGate_oracle` now rank above Blocks (§65.4).
* Mean Δ over d per row (the §62.5 view): 100·d is −0.080 / +0.001 /
  +0.009 / +0.001 at q 1/4/16/64 (§62: −0.089 / +0.002 / +0.010 /
  −0.011).  20·d is −0.034 / −0.008 / −0.004 (§62: −0.035 / −0.009 /
  −0.005).  Only q = 64 and the 100·d q = 1 row move.
* Blocks' `aocc_time`/AOCC ratio at q = 64 is now 0.66 / 0.80 / 0.84
  (d 2/5/10), against 0.63 / 0.43 / 0.49 in §62 and qLogEI's 0.83 /
  0.94.
  * The cost is AOCC over evaluations: −0.025 [−0.038, −0.013] at
    d5/q64 and −0.012 [−0.016, −0.009] at d10/q64, both 0/5, as §63
    predicted.
  * Blocks' AOCC lead over qLogEI at d5/q64 (§62.5: +0.035) shrinks to
    +0.010 (0.076 against 0.065, unpaired).

### 65.3 Per family (the §62.4 view, the minimum of each row)

Most rows are unchanged within ±0.004.  These ones change:

| cell | §62 worst family | §65 worst family | families with CI < 0 |
|---|---|---|---|
| d2 100·d q4 | rastrigin −0.038 | rastrigin −0.066 | rastrigin → ellipsoid, rastrigin |
| d2 100·d q16 | ellipsoid −0.023 (n.s.) | ellipsoid −0.051 | none → ellipsoid, sharp_ridge −0.018 |
| d5 100·d q64 | sharp_ridge −0.049 | sharp_ridge −0.025 | ackley, rastrigin, sharp_ridge → ackley, sharp_ridge |
| d10 100·d q4 | ackley −0.024 | ackley −0.022 (n.s.) | ackley, rastrigin → none |
| d10 100·d q64 | rastrigin −0.001 (n.s.) | ellipsoid +0.000 (floor) | none; CI > 0 now in ackley, rastrigin, rosenbrock, sharp_ridge |
| d10 20·d q4 / q16 | ackley | ackley | rastrigin drops out of both |

* **The d2/100·d losses are spread over families.**  In Blocks' paired
  change at d2/q16, only ellipsoid has a CI below 0 (−0.028
  [−0.049, −0.007], 0/5).  The other four families are −0.015…−0.004,
  all n.s.  At d2/q4 no family is significant; the largest drops are
  ellipsoid −0.034 and rastrigin −0.028.
* The count of cells with at least one family CI below 0 is still 16 of
  21.  d2/100·d/q16 joins it; d10/100·d/q4 leaves it.
* The q = 1 failures of the roadmap claim are unchanged: ellipsoid
  −0.655 / −0.116 against Py-BOBYQA (d 2/5, 100·d), and sharp_ridge
  −0.088 / −0.086 against NGOpt (d 5/10, 20·d).  At q > 1 no family
  minimum is below −0.066.

### 65.4 `RoundRobin_CMAES` and `RegimeGate_oracle`

`RoundRobin_CMAES`, after − before (headline metric, paired; 5/5
unless marked):

| | q = 1 | q = 4 | q = 16 | q = 64 |
|---|---|---|---|---|
| 100·d, d 2 / 5 / 10 | +0.045 / +0.042 / +0.043 | +0.035 / +0.043 / +0.049 | +0.041 / +0.028 / +0.027 | +0.023 / +0.015 / +0.004 |
| 20·d, d 2 / 5 / 10 | +0.003 (3/5) / +0.005 / +0.003 | +0.004 (4/5) / +0.004 / +0.003 | +0.001 / +0.004 / +0.002 | — |

* **At 100·d it gains in every cell; at 20·d the gain is small.**  At
  20·d, 40–200 evaluations are only 4–20 generations.
* **Against the pool's best it was behind in all 21 cells; now it is
  ahead in 3, all 5/5, unadjusted:** d5/100·d/q16 +0.006 [+0.003,
  +0.009], d10/100·d/q4 +0.007 [+0.003, +0.012], d10/100·d/q64 +0.002
  [+0.000, +0.003].  At d10/100·d/q16 it is +0.004 (3/5, n.s.).
* **Sharing against CMA-ES alone (RR − Blocks, paired):**

  | 100·d | q = 1 | q = 4 | q = 16 | q = 64 |
  |---|---|---|---|---|
  | d2 | −0.069 → −0.034 | −0.067 → −0.022 | −0.063 → −0.010 | −0.032 → −0.015 |
  | d5 | −0.051 → −0.020 | −0.050 → −0.013 | −0.048 → −0.025 | −0.012 → −0.015 |
  | d10 | −0.036 → +0.001 | −0.037 → **+0.009** [+0.001, +0.018] 4/5 | −0.030 → −0.008 | −0.008 → −0.017 |

  * §62.2's "at 100·d sharing pays by +0.05…+0.07" was mostly the
    crippled CMA-ES.  The portfolio's lead is now 0.010–0.034 at d 2/5.
  * At d = 10 and q ≤ 4 the lead is gone, and CMA-ES alone is ahead at
    q = 4.
  * At q ≥ 16 the portfolio still leads by 0.008–0.025 in every cell
    (d2/q16, 0.010, is n.s.).  The lead shrank by half or more at q = 16
    (0.063 / 0.048 / 0.030 → 0.010 / 0.025 / 0.008) and grew at q = 64
    for d 5/10 (0.012 → 0.015, 0.008 → 0.017), where `RoundRobin_CMAES`
    gains less than Blocks; at d2/q64 it shrank (0.032 → 0.015).
* **`RegimeGate_oracle`** equals Blocks at d ≤ 5, as before.  At d = 10
  it runs CMA-ES alone inside the block strategy:
  * **q = 1 / 4, 100·d:** +0.045 / +0.049, 5/5.  The gate's choice now
    matches (q = 1: RG − Blocks +0.003 [−0.004, +0.010] 4/5) or slightly
    beats (q = 4: +0.010 [+0.0004, +0.019] 5/5, unadjusted, marginal)
    the portfolio.  In §62 it was −0.036 and −0.037.  So §62.2's
    contradiction with the table row (`dim >= 10, bpd <= 500` → CMA-ES
    alone) is consistent with the half quorum, not with the row.
  * **q = 16:** −0.009 [−0.020, +0.003] 0/5, AOCC −0.015 [−0.029,
    −0.002].  RG − Blocks goes from +0.012 5/5 to −0.002.  §63's side
    check (instance 0) showed the same sign: −0.006, 2/5.
  * **q = 64:** +0.012 5/5.  RG − Blocks is −0.002.
  * At d10/20·d it moves by at most 0.003.
  * §62.2's unexplained gap at d10/100·d/q16 is now 0.070 (CMA-ES alone
    inside Blocks) against 0.064 (`RoundRobin_CMAES`), down from 0.079
    against 0.037.  Most of it was dispatch and the half quorum
    (§63/§64).

### 65.5 Did §63's and §64's local in-sample claims hold?

| claim (source) | local | runner, 5 seeds × 15 | verdict |
|---|---|---|---|
| d5/100·d/q64 Blocks, λ ≥ q floor (§63.3) | +0.018 [+0.015, +0.020] 5/5; Δ vs qLogEI "about −0.001" | +0.018 [+0.015, +0.021] 5/5; Δ −0.001 [−0.006, +0.005], p_holm 1.000 | reproduced (same seeds) |
| d2/100·d/q16 Blocks regression (§63.3/§63.4) | −0.012 [−0.024, −0.000] 1/5; AOCC −0.018; Δ vs qLogEI "about −0.016" | −0.012 [−0.024, −0.000] 1/5; AOCC −0.018 [−0.032, −0.004]; Δ −0.016, p_holm 0.136 | reproduced; not a Holm loss |
| d2/100·d/q64 and d5/q16 Blocks (§63.3) | +0.007 4/5; +0.005 5/5 (both n.s.) | +0.007 4/5; +0.005 5/5 | reproduced |
| d10/100·d/q64 Blocks (§63 side check, instance 0) | +0.013 [+0.008, +0.017] 5/5 | +0.013 [+0.013, +0.014] 5/5, all 3 instances; now a Holm win (0.001) | **confirmed on instances 1 and 2**, which are new to §63's floor check but not to §64.2 (it chose #380 on all 15 instances at d = 10) |
| d10/100·d/q16 Blocks (same) | +0.003 [−0.011, +0.017] 3/5 | +0.006 [−0.001, +0.012] 4/5; Holm win at 0.049 | direction holds; the win is mostly §62's +0.007 plus a n.s. gain, and it is marginal |
| `RoundRobin_CMAES` q ≤ 4 at 100·d (§64.2) | +0.042…+0.050 5/5 | +0.035…+0.049 5/5 | reproduced at d 5/10; d2/q4 (+0.035) was not in §64.2 |
| Blocks q ≤ 4 at 100·d (§64.2) | +0.002…+0.011, all n.s. | q1 +0.010 / +0.011 / +0.006, q4 −0.010 / +0.007 / +0.002, all n.s. | holds except **d2/q4, −0.010 [−0.037, +0.016] 2/5**: not measured in §64.2, n.s., from #375 and #380 together (the floor does not bind at q = 4); it turns a Blocks lead (+0.005) into −0.006 |
| 20·d (never measured locally) | — | Blocks +0.000…+0.005, all n.s.; RR +0.001…+0.005 | the fixes barely matter at 20·d |

* **Where the runner repeats a local cell, it matches up to rounding**,
  as it should: deterministic units, same seeds and CRN.  (The d2/q16
  AOCC CI end is −0.0045, printed −0.005 locally and −0.004 here; at
  d5/q64 #380 adds its −0.000 and the CI end moves by 0.001.)  This
  confirms that the shipped code is the code that was measured, and that
  the runner and the laptop agree.  It does **not** remove the selection
  bias of §63.3's in-sample choice.  Only fresh seeds do (§65.6).
* The parts that are out of sample all point the same way as the
  in-sample ones: the 20·d cells (small), the paired comparison with the
  externals, and d10 instances 1 and 2 for the floor (not for #380).

### 65.6 What this means for the q-sweep TODO, steps (b)–(d)

* **(b) The budget cap: keep it.**  This run cannot compare capped with
  uncapped (only the capped code ran).  What it adds:
  * The cap binds in four cells of the grid: at 100·d, d2/q64 (λ 64 →
    20) and d5/q64 (64 → 50); at 20·d, d2/q16 (16 → 6, so no floor at
    all; Blocks is bit-identical to §62 there) and d5/q16 (16 → 10).
  * For the headline spec the cap is immaterial: §63.3 measured +0.001 /
    +0.002 (n.s.) on Blocks.  With the cap on, d5/q64 already reaches
    parity with qLogEI.
  * Its only measured cost is `RoundRobin_CMAES` at d2/100·d/q64 (§63.3:
    +0.020 instead of +0.041).  That is a secondary spec on a 3-round
    budget, still −0.029 behind qLogEI there.
  * Nothing here argues for dropping a convention that protects sample
    efficiency on small budgets.  The decision is proposed for Harald to
    record: keep `MIN_GENERATIONS = 10`.
* **Blocks d2/q16 (the rest of (b)): the cost is real but small, and not
  the priority.**
  * It is −0.012 against its own past, which the runner reproduces.
    Against qLogEI it is −0.016 [−0.028, −0.005] 0/5, p_holm 0.136.
  * d2/100·d/q4 moved −0.010 as well (n.s., from #375 and #380
    together, not the floor).
  * d = 2 at 100·d is where Blocks lost its q = 4 lead.  The two ideas
    in §63.4 (a block of at least one owner generation; the floor only
    where the arms cannot fill the workers) are untried.
  * Rather than tune on these 5 seeds again, run d2/q4 and d2/q16 in the
    12-seed step and decide on fresh seeds.
* **(c) 12 seeds on the q 4–16 band, fresh seeds.**  The band
  (100·d, q 4–16) now has two Holm wins (d5/q16, d10/q16 at 0.049), one
  near miss (d5/q4, 5/5, 0.060), one cell with an unadjusted CI below 0
  that is not a Holm loss (d2/q16, −0.016 [−0.028, −0.005]), and two
  cells with CIs spanning 0 (d2/q4, d10/q4).
  * Because §63/§64 picked their variants on seeds 3/7/42/1234/2025, the
    step should use **new base seeds** (a comma list in `measure.yml`,
    not the first 12 of the roster).  Report the fresh seeds alone as
    the confirmatory result, and pooled with these 5 only as a secondary
    view.
  * Add d2/100·d/q64 (−0.013, 0/5, p_holm 0.079) and d5/100·d/q64
    (parity).
  * Groups: core + qLogEI + TuRBO1.  qLogEI at d 2/5 dominates the
    cost; there is no qLogEI at d = 10, 100·d.
  * Pre-declare the Holm family as the cells of that run.
  * **Done (§67):** run 36315576900, seeds 1001–1012, 100·d, q 4/16/64,
    core + qLogEI + TuRBO1; the pre-declared Holm family is its 9
    headline cells.
* **(d) The `failure` preset** is unchanged as the next step after (c).
  §62's CMA-ES caveat no longer applies to it.
* **For the roadmap claim:**
  * At 100·d and q ≥ 4, over 9 cells: the unadjusted CI is above 0 in 4
    (d5/q4, d5/q16, d10/q16, d10/q64), spans 0 in 3, and is below 0 in
    2 (d2/q16, d2/q64).  No cell is a Holm loss.  (Out of sample,
    §67: over these 9 cells as their own Holm family, 4 wins, 2 losses
    (d2/q64, d10/q4), 3 unresolved.)
  * At q = 1 (a sequential model or local method wins) and at 20·d
    (qLogEI wins at every q > 1, and NGOpt/TuRBO1 at q = 1) nothing
    changed.  Those 7 Holm losses are search, not scheduling, and the
    CMA-ES fixes do not touch them.  They are the gap a model-based arm
    (roadmap §4) has to close.
* **Reference.**  For core specs, this section (run 36313485264) replaces
  §62 as the expensive-track reference; for the GP baselines, §62's run
  stays the reference.  Any later core-only run can be combined with
  those GP units the same way, as long as `harness_baselines_bo.py` and
  the FP pin do not change.  Check the bit-identity of the core-group
  externals first; it is the cheapest proof that the path is unchanged.

## 66. Model-based arms on the expensive track: COBYQA alone and a new quadratic trust region lead at q = 1 and at d ≤ 5 — in sample, much of it on the one exactly quadratic family, and nothing at d = 10 outside it (2026-09-27)

> **In sample.**  Every number here is on seeds 3/7/42/1234/2025, the
> instances and CRN durations of §62/§65 — the cells whose losses
> motivated the work.  The new arm's constants were not tuned on these
> seeds (a few variants were looked at on seeds 101–103; nothing was
> changed), but its review fixes (§66.2) were made after the first
> measurement of these cells, and the rules of §66.4 were chosen after
> seeing the numbers.  Nothing here is a claim until a fresh-seed run
> repeats it.

**Question.**  §65 left 7 Holm losses of `Blocks_warm_CMAES_JSO`, all at
q = 1 or at 20·d; the largest, d2/100·d/q1, is −0.214 against Py-BOBYQA and
−0.655 on ellipsoid alone.  Does a model-based local arm close them — one
panobbgo already has, or a small new one?

**Method.**  Local, niced (`nice -n 10 ionice -c3`, at most 4 processes),
on the `scripts/measure.py` core path: `run_family_harness` with the free
preset (3 instances per family), `sync_eval=True`, the virtual clock
(async, log-normal σ 0.5, CRN per cell), d ∈ {2, 5, 10}, 20·d and 100·d,
q ∈ {1, 4}.  Blocks and the pool are §65's runner units (core: run
36313485264; qLogEI / TuRBO1: run 36274781342).  A local re-run of Blocks
on d2/100·d/q1/s3 is bit-identical to the runner unit, so the local specs
pair with the runner's.  Δ is paired over the 5 seeds, t-CI95, wins/5,
**unadjusted**.  Full tables, all specs, with and without the ellipsoid
family, and the pre-fix TR rows:
`planning/results/2026-09-27-trust-region/tables.md`.

**Read the ex-ellipsoid view.**  The free preset has one exactly
quadratic family in five (ellipsoid: a rotated, conditioned quadratic).
A quadratic model solves it exactly, and every pool member scores ~0 on it
at d = 10, so it can decide a five-family mean on its own.  Every result
below is given with and without it; reports on the free preset should
show the ex-ellipsoid view next to the mean.

### 66.1 Inventory: what panobbgo already has

| arm | kind | on the measure path |
|---|---|---|
| `COBYQA` | SciPy COBYQA (Powell family: quadratic model, trust region) through a subprocess pipe bridge; box-centre start, sequential (one point in flight), no restart | runs; stops itself when converged (`EndedEarly`, scored at its final best) |
| `NelderMead` | randomized NM direction from the Splitter's best box | alone: 0 evaluations (needs a best box from another arm); as a third Blocks arm: runs |
| `QuadraticWlsModel` | WLS quadratic on the best box's points, box-constrained minimiser, in a subprocess | alone: 0 evaluations (same reason) |
| `LBFGSB` | finite-difference gradients through a bridge | not tried (d evaluations per gradient) |
| `GaussianProcessHeuristic`; the analyzers | GP EI; the analyzers (Splitter, Archive, Best, Convergence, Restart, Sensitivity) hold no model | not tried |

**`COBYQA` alone** (`RR_COBYQA`), Δ vs the pool best.  **Single run:** it
is seed-invariant (box-centre start, no randomness), so each "5/5" below
is one COBYQA run paired with 5 reference runs, and the CIs carry only the
reference's seed variance.

| | d2 | d5 | d10 |
|---|---|---|---|
| 20·d q1 | **+0.101** [+0.085, +0.118] | **+0.045** [+0.029, +0.060] | **+0.022** [+0.017, +0.028] |
| 20·d q4 | −0.033 [−0.048, −0.018] | −0.014 | −0.003 |
| 100·d q1 | +0.017 [−0.017, +0.051] 3/5 | **+0.133** [+0.107, +0.160] | **+0.038** [+0.036, +0.040] |
| 100·d q4 | +0.058 [+0.047, +0.069] | +0.026 | −0.005 |

* **At q = 1 an existing arm is ahead of the pool best in all six cells**
  (five with the CI above 0), without ellipsoid too (+0.012 n.s. / +0.055 /
  +0.028 at 20·d and +0.005 n.s. / +0.158 / +0.046 at 100·d, d 2/5/10).
  At q = 4 it is sequential and wins only where it converges before the
  time horizon (d2/d5 at 100·d; it stops at 30 % / 47 % of the budget on
  average).
* It never restarts (`EndedEarly` in 20–100 % of the runs per cell).
  Outside ellipsoid its strength is **ackley** (d5/100·d: 0.657 against at
  most 0.31 for every other spec): the large initial trust region models
  the funnel, not the ripples.
* **As a third Blocks arm it does not help** (`Blocks3_COBYQA`: −0.127 vs
  the pool best at d2/100·d/q1, within ±0.01 of Blocks at d ≥ 5): the
  uniform rotation gives it one block in three, its subprocess keeps its
  own sequence, and it takes no warm start.  `Blocks3_NelderMead` is
  −0.024…+0.000 against Blocks at q = 1: no.

### 66.2 A new arm: `TrustRegionQuadratic` (opt-in)

`panobbgo/heuristics/trust_region.py`, BOBYQA/NEWUOA-lite written for the
event model instead of bridged:

* **model**: full quadratic, weighted least squares on the archive points
  within 3 radii (inf-norm) of the centre, nearest first, at most
  max(2d + 1, 1.5·p) with p = 1 + 2d + d(d−1)/2.  With fewer points than
  coefficients, **approximately** NEWUOA's least change: a Euclidean
  minimum-norm correction over all coefficients (constant, gradient and
  Hessian) around the previous model's Hessian — not NEWUOA's
  Frobenius-norm update of the Hessian change.  The prior is reset to
  zero at every restart and when the centre moves by more than 3 radii
  (carried across kinks or rugged basins it grew to 1e7–1e8).  The archive
  is every result of every arm (`on_new_results`), so another arm's
  points feed the model;
* **rank test**: the model is used only when the displacements of its
  points from the centre have numerical rank d (SVD, relative tolerance
  1e-3); otherwise the arm emits geometry points along the least-covered
  directions first;
* **centre**: the best archive point outside the tabu balls (inf-norm,
  radius_init) of converged centres, whoever found it; the box centre
  (default) or a random point while the archive is empty.  The radius is
  not reset when the centre jumps;
* **step**: the model minimiser in box ∩ trust region (L-BFGS-B from up to
  3 starts); ratio test: shrink ×0.5 below 0.1, grow ×2 above 0.7 at the
  boundary.  Only a full-radius step emitted at the current radius moves
  the radius; a model that predicts no descent shrinks it at most once per
  archive state;
* **geometry**: least-covered directions, then the coordinate design
  c + r·e_i, c − r·e_i (BOBYQA's first 2d + 1 points), then max-min
  space-filling points in the region;
* **restart** below radius 1e-7;
* **batches**: on demand (`produce(limit)`): points the strategy handed
  back undispatched first, then the step, steps at halved radii and
  geometry points, never an evaluated or in-flight point again.  At q = 1
  a sequential TR method; at q > 1 the extra workers get shorter steps and
  geometry points.

**Review fixes (#383).**  The first version judged "enough points" by
count, not rank, and fitted a minimum-norm quadratic: its first 2d + 1
points could lie on a few axes, the fit then had zero gradient and
curvature outside their span, and the arm never moved there (2-D
reproduction: 2‖x − (0.4, −1.2)‖² in [0, 2] × [−2, 2], best 2.88 at
(0.4, 0.0) after 40 evaluations).  Fixed by the rank test, the coordinate
design ordered +e_1…+e_d first, and the least-change Hessian: on d = 10
quadratics with condition 1e3 (600 evaluations) the best value went from
1.5 (separable) / 17 (rotated) with the rank test alone to 4e-24 / 4e-23.
Tests pin both, and a collinear-archive test pins the rank test itself
(with the rank test disabled, every other test still passed).  A
no-descent model used to shrink the radius on every `produce` call
without new data; now once per archive state, and a restart due then
happens inside `produce`.  The second review round added the prior reset
above; it changed 780 of the 2 700 TR runs (restarts and far jumps
happen), and the tables below are from that final version.

Constants are the textbook ones: radius 0.1 of the box (Py-BOBYQA's
rhobeg), fit span 3; not tuned on these seeds.

Opt-in specs (`harness_ioh.make_trust_region_strategies`, named in
`ioh_benchmark.py run --strategies`): `RoundRobin_TRQ` and
`Blocks_warm_CMAES_JSO_TRQ` (Blocks with TR as a third arm, on Blocks'
seed stream).  In the tables they are `RR_TRQ` and `Blocks3_TRQ`, the
names of the local runs: `Blocks_warm_CMAES_JSO_TRQ` reproduces
`Blocks3_TRQ` bit for bit (checked on d5/20·d/q4/s42, 15 runs);
`RoundRobin_TRQ` is `RR_TRQ` on a different RNG stream (the seed hashes the
spec name), so its numbers will differ by seed noise.  No default changes.

### 66.3 (a) Alone and (b) as a third arm: Δ vs the pool best (fixed arm)

| cell | pool best | Blocks | RR_TRQ | Blocks3_TRQ |
|---|---|---|---|---|
| d2 20·d q1 | TuRBO1 0.145 | 0.095 | **+0.157** [+0.141, +0.174] 5/5 | **+0.070** [+0.042, +0.097] 5/5 |
| d2 20·d q4 | qLogEI 0.100 | 0.090 | **+0.128** [+0.103, +0.153] 5/5 | **+0.034** [+0.014, +0.054] 5/5 |
| d5 20·d q1 | NGOpt 0.073 | 0.044 | **+0.110** [+0.095, +0.125] 5/5 | **+0.050** [+0.024, +0.077] 5/5 |
| d5 20·d q4 | qLogEI 0.049 | 0.044 | **+0.089** [+0.054, +0.124] 5/5 | **+0.042** [+0.019, +0.065] 5/5 |
| d10 20·d q1 | NGOpt 0.051 | 0.028 | −0.016 [−0.022, −0.010] 0/5 | **+0.069** [+0.047, +0.090] 5/5 |
| d10 20·d q4 | qLogEI 0.034 | 0.028 | −0.003 [−0.014, +0.008] 1/5 | +0.024 [−0.001, +0.048] 4/5 |
| d2 100·d q1 | PyBOBYQA 0.440 | 0.226 | **+0.089** [+0.055, +0.124] 5/5 | −0.002 [−0.034, +0.029] 2/5 |
| d2 100·d q4 | qLogEI 0.209 | 0.203 | **+0.251** [+0.227, +0.275] 5/5 | **+0.199** [+0.181, +0.218] 5/5 |
| d5 100·d q1 | PyBOBYQA 0.145 | 0.131 | **+0.220** [+0.193, +0.246] 5/5 | **+0.163** [+0.114, +0.212] 5/5 |
| d5 100·d q4 | TuRBO1 0.112 | 0.123 | **+0.219** [+0.204, +0.233] 5/5 | **+0.183** [+0.153, +0.213] 5/5 |
| d10 100·d q1 | TuRBO1 0.096 | 0.086 | **+0.047** [+0.045, +0.049] 5/5 | **+0.145** [+0.114, +0.177] 5/5 |
| d10 100·d q4 | TuRBO1 0.085 | 0.083 | **+0.070** [+0.037, +0.103] 5/5 | **+0.116** [+0.070, +0.162] 5/5 |

(**bold**: unadjusted CI above 0.)

**Without the ellipsoid family** (Δ vs the pool best; **bold** now marks a
CI *below* 0):

| cell | RR_TRQ | Blocks3_TRQ |
|---|---|---|
| d2 20·d q1 | +0.014 [−0.001, +0.029] 5/5 | **−0.038** [−0.061, −0.015] 0/5 |
| d2 20·d q4 | +0.014 [+0.002, +0.027] 4/5 | −0.005 [−0.016, +0.005] 2/5 |
| d5 20·d q1 | −0.001 [−0.020, +0.019] 2/5 | **−0.028** [−0.049, −0.007] 0/5 |
| d5 20·d q4 | +0.013 [+0.010, +0.017] 5/5 | **−0.007** [−0.010, −0.003] 0/5 |
| d10 20·d q1 | **−0.020** [−0.027, −0.013] 0/5 | **−0.028** [−0.036, −0.020] 0/5 |
| d10 20·d q4 | **−0.009** [−0.010, −0.008] 0/5 | **−0.009** [−0.012, −0.006] 0/5 |
| d2 100·d q1 | +0.070 [+0.024, +0.116] 5/5 | −0.033 [−0.072, +0.006] 1/5 |
| d2 100·d q4 | +0.115 [+0.085, +0.145] 5/5 | +0.071 [+0.037, +0.106] 5/5 |
| d5 100·d q1 | +0.077 [+0.055, +0.098] 5/5 | +0.022 [−0.032, +0.076] 4/5 |
| d5 100·d q4 | +0.063 [+0.033, +0.093] 5/5 | +0.023 [−0.015, +0.061] 4/5 |
| d10 100·d q1 | **−0.020** [−0.022, −0.017] 0/5 | **−0.026** [−0.031, −0.021] 0/5 |
| d10 100·d q4 | **−0.041** [−0.051, −0.032] 0/5 | **−0.022** [−0.026, −0.018] 0/5 |

* **All five families.**  RR_TRQ has its CI above 0 in 10 of 12 cells (not
  at d10/20·d, q1 or q4); Blocks3_TRQ also in 10 (not at d2/100·d/q1,
  −0.002, and d10/20·d/q4, +0.024 n.s.).  Against Blocks, Blocks3_TRQ is
  +0.030…+0.211 with 5/5 and the CI above 0 in every cell.  (After the
  first review round, before the prior reset, RR_TRQ was at 9 of 12:
  d10/100·d/q4 was +0.046 [−0.006, +0.098].)
* **Without ellipsoid**, the 20·d cells are **not** closed: Blocks3_TRQ is
  behind the pool best with CI < 0 in 5 of the 6 20·d cells and 7 of 12
  cells overall, ahead only at d2/100·d/q4.  RR_TRQ is ahead with CI > 0
  in 6 cells (d2/20·d/q4, d5/20·d/q4, and d2 and d5 at 100·d, q1 and q4),
  n.s. in 2 (d2 and d5 at 20·d/q1), and behind in all 4 d = 10 cells.
* **The d2/100·d/q1 gap** (Blocks −0.214 against Py-BOBYQA) is closed by
  RR_TRQ: +0.089 [+0.055, +0.124] with all families, +0.070
  [+0.024, +0.116] without ellipsoid.  Per family vs Py-BOBYQA: ellipsoid
  +0.166, sharp_ridge +0.107, ackley +0.100, rosenbrock +0.123, rastrigin
  −0.049.  (Pre-fix it was +0.016 n.s.; the CI claim is the fixed arm's.)
* **"Sharing pays" is entirely ellipsoid.**  At d10/100·d/q1 the third arm
  (0.242) is far above the arm alone (0.143), but on ellipsoid alone
  (0.832 against 0.315); without ellipsoid the arm alone is 0.101 and the
  third arm 0.094.  The shared archive helps the model where the function
  *is* a quadratic; nothing here shows it helping elsewhere.
* **At d = 10 outside ellipsoid neither form helps**: against Blocks the
  third arm is −0.013 [−0.019, −0.007] (q1) and −0.019 [−0.032, −0.007]
  (q4) at 100·d, the arm alone −0.007 and −0.039.
* **The box-centre start** is worth up to 0.06 at d = 2 (RR_TRQ with a
  random start: 0.468 against 0.529 at d2/100·d/q1), 0.03 at d = 5, and
  nothing at d = 10 (0.152 against 0.143).  Py-BOBYQA and TuRBO start at
  random points; CMA-ES and COBYQA at the centre.  The family optimum is
  uniform in the box with a margin, so the centre is a good fixed start,
  not a leak — but part of the lead over Py-BOBYQA is the start point.
* Wall time is small: a unit (3 specs × 15 runs) takes 5–25 s on 3
  processes.

### 66.4 (c) The arm a simple rule would pick

Both rules were defined after seeing the pre-fix numbers and kept
unchanged for the fixed arm (re-picking them again would select twice).

* `Rule_dim`: RR_TRQ at d = 2, Blocks3_TRQ at d ≥ 5.  Against the pool
  best: CI above 0 in 11 of 12 cells with the five families (d10/20·d/q4:
  +0.024 n.s.); without ellipsoid ahead in 3 (d2/20·d/q4, d2/100·d q1 and
  q4), behind in 6 (d5/20·d q1 and q4, every d = 10 cell), n.s. in 3.
  With the fixed arm RR_TRQ is better than Blocks3_TRQ at d = 5 too, with
  and without ellipsoid; the rule is not the best one on these numbers any
  more, and that is in-sample either way.
* `Rule_probe` — an **uncharged oracle upper bound**: as `Rule_dim`, but at
  d = 10 the third arm only on the instances a probe flags as quadratic
  (below), Blocks otherwise; the probe's evaluations are not charged and
  its verdict is taken from knowing the family.  It adds +0.010…+0.016 at
  d10/100·d and removes the ex-ellipsoid cost there (−0.013 / −0.003
  against the pool best instead of −0.026 / −0.022), i.e. it is Blocks.
  At d = 10 the full-quadratic fit needs 2p = 132 points, most of a 20·d
  budget, so a real probe must read the archive the arms fill anyway.
* **The probe feature.**  The rank-based ELA-lite features of `features.py`
  do not separate the families: `r2_quad` (rank R²) is 0.77–0.83 on
  ellipsoid and 0.69–0.96 on the others, because the ranks of a quadratic
  are not quadratic.  The **f-scale** (affine-invariant, not rank) adjusted
  R² of a full quadratic on max(2p, 10·d) uniform points does separate
  them: 1 − R² is 0 to rounding on every ellipsoid probe and at least
  1.4e-2 / 1.0e-2 / 2.5e-3 on every other probe at d 2 / 5 / 10 (20 probes
  × 15 instances per d).  A threshold of 1e-3 separates all of them.  On
  MA-BBOB, whose quadratic functions carry T_osz, no function is an exact
  quadratic; the threshold is a free-family artefact until measured there.

### 66.5 For roadmap A (probe → select → unleash)

1. **Candidates for the confirmation run** — only at the measured q ∈ {1,
   4} and budgets 20·d / 100·d, and only as candidates:
   * `RoundRobin_TRQ` at d ≤ 5 (CI above 0 in all 8 d ≤ 5 cells with the
     five families, in 6 of 8 without ellipsoid, never below 0);
   * `COBYQA` alone at q = 1 (the strongest arm without ellipsoid at d 5
     and 10);
   * `Blocks_warm_CMAES_JSO_TRQ` at d = 10 only behind a probe that says
     "quadratic"; otherwise `Blocks_warm_CMAES_JSO`.
   Nothing here speaks for q ≥ 16, for larger budgets, or for problems
   that are not smooth.
2. **Probe features to add** (`features.py`, `--log-features`): the
   f-scale quadratic R² (the roadmap's "affine in f" class, not
   monotone-invariant), on the archive; and the TR arm's in-run state —
   ratio-test success rate and radius trend.  A model that keeps predicting
   its steps is the regime signal; a shrinking radius with failed steps
   says "not modellable, hand the budget back".  For this decision these
   are more direct than the rank features.
3. **Confirm first**: a fresh-seed runner run (new base seeds, not the
   roster's first 12) of `RoundRobin_TRQ`, `Blocks_warm_CMAES_JSO_TRQ` and
   `COBYQA` alone against §65's pool at q ∈ {1, 4, 16, 64}, with the
   per-family table and the ex-ellipsoid view; and MA-BBOB at 20·d / 100·d,
   where no function is an exact quadratic.
4. **Report the ex-ellipsoid view next to the free-preset mean.**  A
   selector trained on the free preset alone would learn "quadratic model
   everywhere".

### 66.6 Not measured

q = 16 / 64 for any of the new specs; the GP groups (§65's units reused);
MA-BBOB; the `failure` preset (TR drops non-finite results and counts a
failed step as a failed step; untested there); noise (the ratio test
assumes exact values); constraints (the arm minimises the penalty value);
COBYQA with a random start or restarts; any TR constant other than on
seeds 101–103; the COBYQA and NelderMead rows were not re-run (they do not
use the TR arm).

## 67. Out-of-sample confirmation on 12 fresh seeds (100·d, q 4/16/64): 4 Holm wins, 2 losses, 3 unresolved — §65's resolved verdicts all replicate, d10/q4 becomes a loss, "no Holm loss at 100·d, q > 1" does not hold out of sample (2026-09-27)

**Run.**  `measure.yml` run 36315576900, pre-declared in §65.6 (c):
fresh base seeds 1001–1012, the free preset (5 families × 3 instances),
100·d only, q ∈ {4, 16, 64}, d ∈ {2, 5, 10}, groups core + qLogEI +
TuRBO1.  That is 432 units (core 108, TuRBO1 108, qLogEI 216, split per
family) in 207 shards.  All succeeded: none missing, none failed.  The
cost was 137.8 runner-hours, 129 of them qLogEI, and 9 h 15 min wall.
Every unit carries `git_sha` 58cf1e7, **the commit §65 measured**, and
every shard has `fp_env_id` 80ee2a0090c4.  So this is a replication of
§65's code on new seeds, not a test of later master.  **The baselines
ran at 58cf1e7 here too**; in §65 the qLogEI/TuRBO1 units came from §62's
run at 1778e1b (combined there because `harness_baselines_bo.py` did not
change).  Since 58cf1e7 master changed (58cf1e7..c05b4bb, not checked by
a run):
* `TrustRegionQuadratic` (`heuristics/trust_region.py`, new, opt-in) and
  its specs `make_trust_region_strategies` in `harness_ioh.py`, plus the
  opt-in `trq` group in `measure.py`/`measure.yml` (§66, #383/#384);
* `COBYQA`: an opt-in `warm_start="archive"` (default `None`, unchanged)
  and start-radius docs (§69);
* the wide preset: `make_wide_battery` / `WIDE_FAMILIES` in
  `harness_families.py`, new bases and placement knobs in
  `lib/families.py` (opt-in, §68); the free preset's families untouched;
* `StrategyBase.preload_results` in `core.py` (a run without a preload
  is unchanged), `features.py` and `selector_data.py` (new, §70);
* `measure.py`: the ex-ellipsoid table (#384), the `wide` preset and the
  `trq` group — aggregation and reporting, not the runs;
* `cma_es.py`: a docstring only.
None of it is on the code path of the specs measured here, by reading
the diff; no bit-identity check was run.

**Pre-declared test.**  One Holm family: the 9 headline cells,
`Blocks_warm_CMAES_JSO` − pool best on `aocc_time`.  Everything else
below is unadjusted and descriptive.  This is the same pool and method
as §62/§65: t-CI95 over seeds of the per-seed instance-mean delta.

**What is fresh and what is not.**
* Fresh: the optimizer seeds and the CRN duration streams.  The duration
  stream is keyed on the base seed per cell (`virtual_clock.py`).
* **Not fresh: the 15 instances.**  They are fixed by the preset and are
  the same (family, instance) pairs that §62–§65 measured and that §63/§64
  chose their variants on.  So this run tests overfitting to seeds, not
  to instances.  The CIs stay conditional on these 15 instances.

**Files.**  `planning/results/2026-09-27-measure-12seed/`:
* `summary.md`: the 432 units, re-aggregated locally with the current
  `scripts/measure.py`.  It adds the descriptive ex-ellipsoid table
  (#384).  Its headline, per-family and per-cell tables are
  **identical** to the runner's own `measure-summary` (diffed; the only
  differences are the two added passages).
* `plan.json`: the run's plan.
* `summary-pooled-17seeds.md`: the secondary pooled view (§67.3).
  Raw units stay in the run's artifacts.

### 67.1 Headline: `Blocks_warm_CMAES_JSO` − pool best, 12 fresh seeds against §65's 5

The "§65, 5 seeds" column is §65's numbers on the same cells (seeds
3/7/42/1234/2025).  Its p_holm is recomputed **over these 9 cells**
(`measure.py aggregate` over §65's units for them), so it is comparable.
§65.1 printed Holm over 21 cells.

| cell | pool best | Blocks | Δ [CI95] wins/12 | p | p_holm | §65, 5 seeds: Δ [CI95] wins (p_holm, 9 cells) | verdict |
|---|---|---|---|---|---|---|---|
| d2 q4 | qLogEI 0.194 | 0.215 | +0.021 [−0.002, +0.043] 8/12 | 0.067 | 0.183 | −0.006 [−0.037, +0.026] 2/5 (1.000) | unresolved; sign flipped |
| d2 q16 | qLogEI 0.176 | 0.165 | −0.011 [−0.022, +0.001] 3/12 | 0.061 | 0.183 | −0.016 [−0.028, −0.005] 0/5 (0.068) | unresolved; the CI now spans 0 |
| d2 q64 | qLogEI 0.113 | 0.097 | **−0.015 [−0.021, −0.010] 0/12** | 0.0001 | **0.001** | −0.013 [−0.021, −0.006] 0/5 (0.044) | **loss**, replicates |
| d5 q4 | TuRBO1 0.113 | 0.124 | **+0.011 [+0.007, +0.016] 11/12** | 0.0002 | **0.001** | +0.011 [+0.006, +0.017] 5/5 (0.033) | **win**, replicates |
| d5 q16 | qLogEI 0.074 | 0.103 | **+0.030 [+0.025, +0.035] 12/12** | <0.0001 | **<0.001** | +0.031 [+0.021, +0.041] 5/5 (0.007) | **win**, replicates |
| d5 q64 | qLogEI 0.060 | 0.060 | +0.001 [−0.001, +0.002] 8/12 | 0.347 | 0.347 | −0.001 [−0.006, +0.005] 2/5 (1.000) | parity replicates |
| d10 q4 | TuRBO1 0.086 | 0.079 | **−0.008 [−0.012, −0.004] 1/12** | 0.002 | **0.007** | −0.002 [−0.012, +0.009] 2/5 (1.000) | **loss**, new |
| d10 q16 | TuRBO1 0.058 | 0.072 | **+0.014 [+0.010, +0.018] 12/12** | <0.0001 | **<0.001** | +0.013 [+0.007, +0.019] 5/5 (0.029) | **win**, replicates |
| d10 q64 | TuRBO1 0.028 | 0.045 | **+0.017 [+0.016, +0.018] 12/12** | <0.0001 | **<0.001** | +0.018 [+0.016, +0.021] 5/5 (0.000; pool best Optuna TPE 0.028) | **win**, replicates |

**Holm count at 0.05: 4 wins / 2 losses / 3 unresolved.**
* Wins: d5/q4, d5/q16, d10/q16, d10/q64.
* Losses: d2/q64 (−0.015 against qLogEI) and d10/q4 (−0.008 against
  TuRBO1).
* Unresolved: d2/q4, d2/q16, d5/q64.
* The runner's summary gives the same count.  §65's units, Holm over the
  same 9 cells, give 4 / 1 / 4: the same four wins and the d2/q64 loss.

### 67.2 What replicates and what does not

* **Every in-sample verdict that was resolved replicates**, with the same
  sign and a similar size.  The four wins are within ±0.001 of §65; the
  d2/q64 loss moved by 0.002 (−0.013 → −0.015).
  * d5/q4 was §65.2's near miss: p_holm 0.060 over 21 cells, 5/5.  It is
    now 11/12, p_holm 0.001.
  * d10/q16 was the edge win at 0.049.  It is now 12/12, p_holm < 0.001.
  * d2/q64 was a near miss over 21 cells (0.079).  It is now a clear loss,
    0/12.
* **d5/q64 parity replicates**: +0.001 [−0.001, +0.002], 8/12, p 0.35.
  The effect of §63's λ ≥ q floor (the §62 loss of −0.019 closes to
  parity) holds on fresh seeds.
* **d10/q4 is a new Holm loss.**  In §65 it was −0.002 [−0.012, +0.009]
  2/5, with p 0.64.  With 12 seeds it is −0.008, 1/12.  This is not a
  reversal: the 5-seed CI contained −0.008.  §65 was underpowered here.
* **d2/q16: −0.016 → −0.011, n.s.**  This is the gap to qLogEI only;
  the run has no pre-floor arm, so it does not measure §63's before →
  after cost of the floor (−0.012 in sample).  The gap is about
  two-thirds as large on fresh seeds; its CI now touches 0 (upper end
  +0.001).  It is not a Holm loss in either run.
* **d2/q4 flips sign**: −0.006 → +0.021 [−0.002, +0.043], 8/12.  Neither
  side is resolved.  Blocks leads the pool in its per-seed mean, but d = 2
  is the noisiest dimension (CI width 0.045 with 12 seeds).
* **All 9 of the 12-seed means lie inside §65's 5-seed CIs.**  Nothing
  contradicts the in-sample numbers.  Against §65's units with Holm over
  the same 9 cells (4 / 1 / 4), the only change of verdict is d10/q4, and
  it comes from power.  Against §65.2's 21-cell verdicts, d5/q4 and
  d2/q64 also change.  That comes from the smaller family plus more
  seeds; their Δs barely moved.
* **One §65 statement does not hold out of sample: "no Holm loss is
  left at 100·d with q > 1"** (§65.2).  That was 0 Holm losses in the 9 cells, out of 21 over
  which Holm was taken.  On fresh seeds and a 9-cell family there are two
  such losses.  §65.6 read the same 9 cells as "CI above 0 in 4, spans 0
  in 3, below 0 in 2 (d2/q16, d2/q64)".  Out of sample it is 4 above,
  3 spanning, 2 below, but one of the "below" cells changed: d10/q4 in,
  d2/q16 out.

### 67.3 Secondary view: 17 seeds pooled (pre-declared as secondary in §65.6)

The 12 fresh seeds plus §65's 5 give 255 runs per cell.  §65's core
units come from run 36313485264, and its qLogEI/TuRBO1 units from
§62's run 36274781342 (1778e1b); §65 explains why they combine.  The
5 old seeds are the ones §63/§64 selected on, so this view is **not** the
confirmatory result.  It is a precision view only.

| cell | Δ [CI95] wins/17 | p_holm (9 cells) |
|---|---|---|
| d2 q4 | +0.013 [−0.005, +0.031] 10/17 | 0.272 |
| d2 q16 | −0.012 [−0.020, −0.004] 3/17 | 0.015 |
| d2 q64 | −0.015 [−0.019, −0.011] 0/17 | <0.001 |
| d5 q4 | +0.011 [+0.008, +0.015] 16/17 | <0.001 |
| d5 q16 | +0.030 [+0.026, +0.034] 17/17 | <0.001 |
| d5 q64 | +0.000 [−0.001, +0.002] 10/17 | 0.727 |
| d10 q4 | −0.006 [−0.010, −0.002] 3/17 | 0.015 |
| d10 q16 | +0.014 [+0.010, +0.017] 17/17 | <0.001 |
| d10 q64 | +0.018 [+0.017, +0.018] 17/17 | <0.001 |

The file's header comes from the aggregator and reads "Commit 1778e1b,
58cf1e7; run local; 562 unit(s)": the union of the three source runs'
commits, aggregated locally, 432 new + 130 old units (core 45, TuRBO1 45,
qLogEI 40, split per family); no run id applies.

Pooled, d2/q16 would count as a loss (4 / 3 / 2).  That rests on the
in-sample seeds (0/5 there), so it is not claimed.  The confirmatory
count is §67.1's 4 / 2 / 3.

### 67.4 Per family and without ellipsoid (descriptive)

Per family, Blocks − pool best (12 seeds; from `summary.md`), set
against §65.3:

* **d2/q16: the deficit is mostly the ellipsoid family.**
  * Ellipsoid is −0.037 [−0.050, −0.023], 0/12 (§65: −0.051, 0/5).
  * The other four families are −0.014…+0.005, all n.s.  §65's
    sharp_ridge −0.018 [−0.024, −0.012] 0/5 does not replicate: it is now
    −0.014 [−0.038, +0.011].
  * Without ellipsoid the cell is −0.004 [−0.016, +0.008], 5/12
    (post-hoc view, descriptive).
  * §65.3 located Blocks' own before → after change at d2/q16 (the
    floor, in sample) on ellipsoid too (−0.028).  This run measures only
    the gap to qLogEI, and that gap sits mostly on the one exactly
    quadratic family, where qLogEI's GP is strong.
* **d2/q64: broad, not ellipsoid only.**
  * The CI is below 0 for ackley (−0.029, 0/12; §65 −0.034) and for
    ellipsoid (−0.025, 1/12; §65 −0.015 n.s.).  The other three families
    are −0.011…−0.004, all n.s.
  * Without ellipsoid it is −0.017 [−0.022, −0.011], 1/12.  The pool
    best there is Optuna TPE (0.135).
  * Against qLogEI it is a time-axis loss, not a search loss:
    * Blocks' AOCC over evaluations beats qLogEI's, +0.012
      [+0.005, +0.019] 12/12 (0.145 against 0.133).
    * Its `aocc_time`/AOCC ratio is 0.67, against qLogEI's 0.85.
    * This holds against qLogEI only.  On AOCC, TuRBO1 is ahead of
      Blocks (0.159; −0.014 [−0.020, −0.008] 1/12), and the AOCC pool
      best is Py-BOBYQA at 0.405 (sequential).
    * The cause is inferred, not measured: with 200 evaluations at
      q = 64 the run is about 3 rounds, and the loss is consistent with
      the block rule limiting a CMA-ES block to 20 dispatches.  §63.3
      measured the floor's budget cap as immaterial for Blocks here
      (+0.001).
* **d2/q4: §65's family losses do not replicate.**
  * Rastrigin was −0.066 [−0.098, −0.035] 0/5.  It is now +0.022
    [−0.033, +0.076] 6/12.
  * Ellipsoid was −0.028, 1/5, and is now −0.004 n.s.
  * Ackley is +0.077 [+0.047, +0.107], 11/12.
  * Without ellipsoid the cell is +0.027 [+0.005, +0.050], 8/12: an
    unadjusted CI above 0.
* **d5/q64: the parity is a sum of opposite family effects**, as in §65.
  * Rosenbrock is +0.048, 12/12.
  * Ackley is −0.019 (0/12), sharp_ridge −0.014 (0/12) and rastrigin
    −0.012 (1/12).  Rastrigin was n.s. in §65.
  * For the roadmap's "never much worse on any class", this parity cell
    has three family CIs below 0.
* **d10/q4: ackley −0.042 [−0.058, −0.027] 1/12 and rastrigin −0.014
  [−0.023, −0.005] 2/12.**  §65 had −0.022 (n.s.) and −0.014 (CI ending
  at +0.000).  Rosenbrock is +0.019, 11/12.
* d5/q16 rastrigin −0.006 [−0.010, −0.003], 1/12, replicates §65's
  −0.010 (1/5).  In d10/q16 and d10/q64 no family has a CI below 0.  At
  d = 10, ellipsoid is +0.000 in every cell (the floor: every strategy
  scores about 0 there).
* **Cells with a family CI below 0: 5 of 9** (d2/q16, d2/q64, d5/q16,
  d5/q64, d10/q4).  §65 also had 5 of the same 9, but with d2/q4 in
  place of d10/q4.  The worst family entry is
  −0.042 (d10/q4 ackley); in §65 it was −0.066 (d2/q4 rastrigin, which
  did not replicate).
* **Without ellipsoid** (the #384 table, post-hoc), every cell keeps the sign of
  its headline Δ.  d2/q16 shrinks to −0.004, d2/q4 gets a CI above 0, and
  the d = 10 deltas grow by about 1/4 (the ellipsoid runs are about 0 for
  everyone).  This view was added after the run was declared, and it is
  descriptive.

### 67.5 `RoundRobin_CMAES` and `RegimeGate_oracle`

**`RoundRobin_CMAES` − pool best** (12 seeds, unadjusted):
* It is ahead in 4 cells:
  * d5/q16: +0.005 [+0.003, +0.008], 12/12;
  * d10/q4: +0.006 [+0.003, +0.010], 11/12;
  * d10/q16: +0.006 [+0.004, +0.009], 12/12;
  * d10/q64: +0.001 [+0.001, +0.002], 12/12.
* §65.4 had the first, second and fourth of these at 5/5, and d10/q16 at
  +0.004, 3/5.  All four replicate, and d10/q16 is now resolved.
* It is behind at d2 (−0.011 / −0.037 / −0.031 at q 4/16/64) and at
  d5/q64 (−0.015).  d5/q4 is −0.002, n.s.

**Sharing against CMA-ES alone** (RR − Blocks, `aocc_time`, paired):

| | q = 4 | q = 16 | q = 64 |
|---|---|---|---|
| d2 | −0.031 [−0.049, −0.014] 1/12 (§65 −0.022) | −0.027 [−0.035, −0.018] 0/12 (§65 −0.010, n.s.) | −0.015 [−0.022, −0.009] 1/12 (§65 −0.015) |
| d5 | −0.014 [−0.020, −0.008] 1/12 (§65 −0.013) | −0.024 [−0.028, −0.020] 0/12 (§65 −0.025) | −0.016 [−0.017, −0.014] 0/12 (§65 −0.015) |
| d10 | **+0.014 [+0.010, +0.018] 12/12** (§65 +0.009, 4/5) | −0.008 [−0.011, −0.004] 0/12 (§65 −0.008) | −0.016 [−0.017, −0.015] 0/12 (§65 −0.017) |

* §65.4's reading replicates.  The portfolio leads CMA-ES alone in every
  cell except d10/q4, where CMA-ES alone is ahead, now 12/12.
* At d2/q16 the portfolio's lead is larger out of sample: −0.027,
  against §65's −0.010 n.s.

**`RegimeGate_oracle`:**
* At d ≤ 5 it is identical to Blocks in every run.
* At d = 10 it runs CMA-ES alone inside the block strategy.  RG − Blocks:
  * **q = 4: +0.014 [+0.010, +0.018] 12/12** (§65: +0.010 [+0.0004,
    +0.019] 5/5, marginal).  Against TuRBO1 it is +0.006 [+0.003,
    +0.010] 10/12 (unadjusted): the gate's choice leads where Blocks has
    a Holm loss.  `RoundRobin_CMAES` is the same there (0.093 against
    0.093; +0.006, 11/12): at d10/q4 CMA-ES alone and the RR portfolio
    are about equal, both ahead of Blocks.
    * Per family, CMA-ES alone recovers Blocks' two losing families:
      ackley +0.045 (11/12) and rastrigin +0.015 (12/12).
    * Against TuRBO1 it is level on both families: +0.002 and
      +0.001, n.s.
  * **q = 16: −0.002 [−0.005, +0.002] 4/12** (§65 −0.002).
  * **q = 64: −0.001 [−0.003, +0.001] 5/12** (§65 −0.002).  On AOCC it
    is −0.007 [−0.009, −0.005], 0/12.
* So the gate row `dim >= 10, bpd <= 500` → CMA-ES alone is right at
  q = 4 and neutral at q 16/64 on `aocc_time`.  It is behind on AOCC at
  q = 64.

### 67.6 What this means

**For §63/§64/§65's claims.**
* **§63.3, "the floor closes d5/q64 to parity":** confirmed on fresh
  seeds.
* **§63.3/§63.4, the d2/q16 gap to qLogEI (−0.016 in sample):** the
  direction holds (−0.011, 3/12), the size is about a third smaller, and
  it is not significant.  It sits mostly on ellipsoid (§67.4).  The
  floor's own cost (before → after, −0.012 in §63.3) is not re-measured:
  this run has no pre-floor arm.
* **§63.3/§65.1, "the q = 64 wins at d = 10":** confirmed.  d10/q64 is
  +0.017, 12/12, and the aocc_time/AOCC ratios at q = 64 are
  0.67 / 0.80 / 0.85 (§65: 0.66 / 0.80 / 0.84).
* **§64/§65.4, "at d10/q ≤ 4 CMA-ES alone beats the portfolio"**
  (q = 4 measured here): confirmed and sharper.  Where it holds, the
  headline spec loses a Holm cell, and the regime gate (and
  `RoundRobin_CMAES`) leads there, unadjusted.
* **§65.2's "no Holm loss at 100·d, q > 1":** does not hold out of
  sample (d2/q64, d10/q4).
  §65.6's roadmap summary for the 9 cells is replaced by the one below.
* **Roadmap claim at 100·d, q ≥ 4 (confirmatory, free preset, these
  15 instances):**
  * Blocks is ahead of the best of 8 externals (7 at d = 10) at d5/q4,
    d5/q16, d10/q16 and d10/q64, by +0.011…+0.030.
  * It ties at d5/q64.
  * It is behind qLogEI at d2/q64 (−0.015) and TuRBO1 at d10/q4 (−0.008).
  * It is unresolved at d2/q4 (+0.021) and d2/q16 (−0.011).
  * Per family it is "much worse" nowhere by more than 0.042 at q > 1,
    but 5 of 9 cells have a family CI below 0.
* §66's model-based arm results (q ∈ {1, 4}, TRQ, COBYQA) are **not**
  tested here.  The `trq` group was not in the run, and there is no
  q = 1.  They remain in-sample.

**For the q-sweep TODO.**
* **(c) done**: this section.
* **(b) The budget cap: keep it (proposal unchanged, for Harald to
  record).**
  * This run, like §65, ran the capped code only.
  * The cap binds at d2/q64 and d5/q64.  d5/q64 is parity on fresh seeds.
  * d2/q64 is a Holm loss.  Its signature against qLogEI (AOCC ahead,
    time axis behind, 3 rounds) is consistent with the block rule's
    20-dispatch limit and the round count; that cause is inferred, not
    measured.  It is likely not the cap: §63.3 measured uncapping as
    +0.001 for Blocks there.
  * If d2/q64 is to be attacked, the likely lever is the block rule at
    q ≫ λ, not removing the cap.
* **(b) Blocks d2/q16: close it without tuning.**
  * On fresh seeds the gap to qLogEI is −0.011, n.s.; ex-ellipsoid it
    is −0.004 (a post-hoc, descriptive view, not a test).
  * What is left sits mostly on the quadratic family.  A model-based arm (§66) is
    the natural lever there, not block sizing.
  * The two §63.4 ideas (a block of at least one owner generation; the
    floor only where the arms cannot fill the workers) stay untried.
    Nothing here makes them a priority.
* **New: d10/q4.**  This is a Holm loss where a known choice leads:
  CMA-ES alone (the regime-gate row) is +0.006 against TuRBO1 there,
  unadjusted (`RoundRobin_CMAES` the same).  It is
  evidence for the selector work (roadmap §4 A: probe → select), not for
  a new scheduling fix.
* **(d) The `failure` preset** stays the next step.  Suggested design,
  from this run's lessons:
  * 12 seeds, with a seed list disjoint from both 3/7/42/1234/2025 and
    1001–1012;
  * the Holm family pre-declared as that run's headline cells;
  * the ex-ellipsoid-style view pre-declared if the preset has an
    analogous single family.
  * At 12 seeds a d = 2 cell still has CI width about 0.045 at q = 4.
    Expect d = 2 cells to stay unresolved unless the effect is at least
    about 0.02.

**Reference.**  For the core specs at 100·d, q 4/16/64, this run
(36315576900, 12 fresh seeds) is now the confirmatory expensive-track
reference.  §65 (run 36313485264) stays the reference for q = 1 and
20·d, and §62's run for the GP baselines outside these cells.

### 67.7 Multiplicity and other caveats

* **One pre-declared family of 9, Holm at 0.05, and nothing else is
  adjusted.**  The per-family table (45 CIs), the ex-ellipsoid table,
  the RR/RG comparisons, the AOCC deltas and the pooled view come to
  well over 100 unadjusted CIs.  At 95 % a handful are expected to
  exclude 0 by chance.  Read single family cells as descriptive,
  especially the ones that do not replicate across §65 and §67 (d2/q4
  rastrigin, d2/q16 sharp_ridge).
* **The pool best is a per-cell maximum** over 8 externals (7 at
  d = 10).  That favours the baseline side.  At d10/q64 TuRBO1 and
  Optuna TPE tie at 0.028; §65 picked TPE, here TuRBO1.
* **Holm is step-down.**  The two d = 2 unresolved cells share p_holm
  0.183 (monotonicity).  d2/q64's verdict changed between §65.2 (21 cells,
  0.079) and here partly because the family is smaller, not only because
  of the data.
* **Wins/12 as a sign test** (two-sided): 12/12 is p = 0.0005, 11/12
  p = 0.006, 8/12 p = 0.39.  The t-test and the sign test agree on every
  resolved cell.
* **Instances are not fresh** (see the top of this section).  A selection
  effect of §63/§64 on the 15 instances would not show up here.  The
  wide preset (§68) or new instance ids would test it.
* **Only 100·d, q ≥ 4.**  The q = 1 and 20·d cells, where §65's 7 Holm
  losses are, were not re-run.  Their evidence is still 5 in-sample
  seeds.
* **Code: 58cf1e7, the commit §65 measured** (baselines included).
  Later master is not measured by this run.  The changes since (listed
  at the top) are opt-in, reporting or docstrings by reading the diff,
  but no bit-identity check was done.

## 68. A wide family preset for the selector: 15 families across the axes that decide the algorithm, no free box-centre hit (2026-09-27)

> **Numbering.**  §67 (written after this section) is the 12-seed
> confirmation of §65; this is §68.

**Question.**  Roadmap §4 A needs a problem set wide enough to learn a
selector from, with features that transfer.  The expensive track's `free`
preset has 5 families × 3 instances; §66 showed that one of them, the only
exact quadratic (ellipsoid), can decide the mean alone (a quadratic model
0.88 where every baseline scores 0).  Which axes of difficulty does `free`
cover, which are missing, and what would a wider preset look like?

**What was built** (opt-in; `free`, the default grid and every existing
instance are unchanged, bit for bit):

* four new bases in `panobbgo/lib/families.py`: `different_powers`
  (BBOB f14), `styblinski_tang` (minimiser by Newton to float precision),
  `levy` (defined down to d = 1) and `schwefel_box` (Schwefel on a random
  window of its classic domain, x_opt placed so the box never reaches the
  boundary penalty — with a uniformly shifted x_opt at scale 50, the
  existing `schwefel`, about a third of every coordinate's range is the
  quadratic penalty, which a first smoke showed: 1−R² of a quadratic 0.03
  over the box);
* four `Family` knobs, drawn from a stream of their own (spawn key 3), so
  an instance without them is bit-identical and one with them keeps its
  R, f_opt and constraints: `min_centre_dist` (every optimum at least that
  fraction of B·√d from the centre — for an embedding the nearest point of
  the optimal set, ‖R[:k] x_opt‖ — by rejection, so uniform on the rest;
  bounded by 0.6·(1 − opt_margin)·√(k/d), which keeps a draw's acceptance
  above 1 % up to d = 160),
  `boundary_faces` (x_opt on the faces of ceil(fraction·d) coordinates plus
  a linear pull `slope·Σ(B − s_i x_i)`, zero there and positive inside, so
  f_opt stays exact but the gradient does not vanish and the unconstrained
  minimiser lies outside the box), `signed_permutation` (separable, but no
  two instances aligned alike) and `effective_dim` (the base sees only the
  first ceil(fraction·d) coordinates of Λ R (x − x_opt): a random subspace,
  the rest exactly neutral; only for the bases of `EMBEDDABLE_BASES`, each
  checked to stay non-constant with an exact optimum down to k = 1 —
  rosenbrock is constant at k = 1 and is refused);
* `harness_families.WIDE_FAMILIES` / `make_wide_battery()`, the preset
  `wide` in `scripts/measure.py` (`--presets wide`; default `free`), in
  `benchmarks/family_screen.py preset=wide`, and `measure.py cost`, the
  plan's cost per group;
* tests (`tests/test_families_wide.py`, `tests/test_measure.py`): f(x_opt)
  == f_opt bit for bit and nothing below it on random points and the
  corners of the box, finite everywhere, determinism and pickling per
  instance, no optimum (for levy_embed: no point of the optimal set)
  within 0.3·B·√d of the centre, the acceptance bound at d up to 160 and the
  placement's error path, every instance
  rotated (Haar) or signed-permuted, the face optimum really bound by the
  box (just outside is lower, just inside the slope ≥ 0.5), the embedding's
  other directions exactly neutral, the embeddable bases (and the refused
  ones), levy at d = 1, and the shared-label instances equal to `free`'s
  unless redrawn.

**Review #385** (after the first push): the centre test of an embedding now
applies to its optimal set, not to x_opt alone (the old test let a
levy_embed optimum set pass within 0.3·B·√d of the centre); the
`min_centre_dist` bound above replaced a bound that allowed acceptance to
vanish at large d; `effective_dim` is limited to the checked bases;
`schwefel_box` refuses `min_centre_dist` (it places x_opt itself, |x_opt_i|
≥ 4.01).  Only the six levy_embed instances whose optimal set was too close
changed (checked instance by instance against the first version); their
smoke rows below are re-run.

### 68.1 The free preset along the axes that decide the algorithm

Construction first: every family instance is
f(x) = f_base(Λ R (x − x_opt)) + f_opt with R Haar-random, x_opt uniform
in [−4, 4]^d (box [−5, 5]^d, margin 0.2) and a random f_opt.  So the free
preset is already shifted and rotated per instance — but that fixes three
axes for *every* family: never separable, never a boundary optimum, and the
optimum uniform around the centre (the centre is the point with the
smallest expected distance to it).

| family | separable | conditioning | modality | smooth | plateaus | funnel / deceptive | eff. dim |
|---|---|---|---|---|---|---|---|
| ellipsoid | no (rotated) | 1e6, constant | unimodal, **exact quadratic** | C∞ | no | funnel | d |
| rosenbrock | no | varies along the valley | 1 (2 minima at d ≥ 4) | C∞ | no | funnel, curved | d |
| rastrigin | no (rotated) | 1 | 10^d regular | C∞ | no | funnel (global structure) | d |
| ackley | no (rotated) | 1 | many, small | kink at x_opt | nearly flat outer region | funnel, flat far away | d |
| sharp_ridge | no | 100:1 cone | unimodal | kink on a ridge | no | funnel | d |

Rotation: all five Haar-rotated.  Optimum vs centre: uniform with margin;
7 of the 45 instances (5 at d = 2) lie within 0.3·B·√d of the centre
(ellipsoid_d2_i1 at 0.16, sharp_ridge_d2_i2 at 0.13, …; mean 0.42).
Boundary optimum: none.

**Gaps**, against the COCO classes (BBOB f1–f24) and the usual BO test set:

| COCO class | BBOB fids | free covers | missing |
|---|---|---|---|
| 1 separable | f1–f5 | — | separability at all (f3/f4 separable Rastrigin, f5 linear slope = a **boundary optimum**) |
| 2 low/moderate conditioning | f6–f9 | rosenbrock | asymmetry (f6 attractive sector), **plateaus** (f7 step ellipsoid) |
| 3 high conditioning, unimodal | f10–f14 | ellipsoid, sharp_ridge | a conditioned *non-quadratic* (f12 bent cigar, f14 different powers) |
| 4 multimodal, adequate structure | f15–f19 | rastrigin (+ ackley) | ruggedness (f16 Weierstrass, f17/18 Schaffers) |
| 5 multimodal, weak structure | f20–f24 | — | **deception** (f20 Schwefel, f24 Lunacek), random wells (f21/f22 Gallagher), f23 Katsuura |

BO benchmarks (Branin, Hartmann 3/6, Levy, Styblinski–Tang, Michalewicz,
Schwefel, Griewank, step functions, Lunacek, sums of different powers,
REMBO/ALEBO-style embeddings): of these the free preset has only
rastrigin-like structure.  Missing: a few smooth wells of different depth
(Hartmann; Gallagher f22 is its d-dimensional generalisation), Levy and
Styblinski–Tang, low effective dimension, plateaus, a deceptive separable
function.  Branin and Hartmann themselves are fixed-dimensional (and so are
Michalewicz's known optima): not offered, their classes are.

Also missing and **not** in the wide preset either: noise; ruggedness
(Weierstrass, Katsuura, Schaffers — f* is known, left out to keep the
preset at 15; candidates for a next step); discrete / mixed
variables; constraints and failure regions (presets of their own).

### 68.2 The wide preset

`make_wide_battery()`: 15 families × d 2/5/10 × 3 instances (135
instances), battery seed as `free`.  Every family: every optimum at least
0.3·B·√d from the centre (`WIDE_MIN_CENTRE_DIST`; at d = 2 that excludes
the central 22 % of uniform draws, at d = 10 almost none; schwefel_sep
places its own optimum at |x_opt_i| ≥ 4.01, levy_embed tests its optimal
set against min(0.3, 0.48·√(k/d)) — the placement bound — which is 0.3 at
d 2/5/10 and below it at some other d, e.g. 0.28 at d = 30;
`clip_min_centre_dist`, so the preset builds at every d, tested for d 2…40,
80 and 160), a random f_opt,
and a Haar rotation — except the two separable families, which get a random
signed permutation.  The 8 families that share a label with `free` (5) or
`shapes` (3) share their instance seeds: 38 of the 45 `free` instances are identical in
`wide`, the other 7 had x_opt within 0.3·B·√d and were redrawn.

| family | base | axes it adds (✓ = the axis it is there for) |
|---|---|---|
| ellipsoid | BBOB f2 shape, rotated | high conditioning (1e6), exact quadratic — kept, now 1/15 of the mean |
| different_powers | BBOB f14, **new base** | smooth but not quadratic, degenerate curvature, conic bottom ✓ |
| bent_cigar | BBOB f12 | 1e6 anisotropy, non-quadratic (T_asy) ✓ |
| rosenbrock | classic | curved valley, changing conditioning |
| sharp_ridge | BBOB f13 shape | kink on a ridge ✓ (non-smooth) |
| attractive_sector | BBOB f6 | asymmetry around the optimum ✓ |
| step_ellipsoid | BBOB f7 | **plateaus** / neutrality ✓ |
| rastrigin | classic, rotated | regular multimodality, strong global structure |
| ackley | classic, rotated | flat outer region + funnel |
| schwefel_sep | Schwefel, **new base `schwefel_box`**, signed permutation | **separable** ✓, **deceptive** ✓ (best minimum near one face, second-best far away; BBOB f20's design with a random window) |
| styblinski_tang_sep | Styblinski–Tang, **new base**, signed permutation | **separable** ✓, 2^d minima (BO standard) |
| lunacek_box | BBOB f24, placement `box` | **double funnel** ✓, the wrong funnel towards the centre |
| gallagher21 | BBOB f22 (21 peaks, α_opt 1e6) | **weak global structure**, few wells of different depth (Hartmann-like) ✓ |
| levy_embed | Levy, **new base**, `effective_dim = 1/3` | **low effective dimension** ✓ (k = 1/2/4 at d = 2/5/10), neutral directions; Levy is a BO standard |
| rosenbrock_edge | rosenbrock, `boundary_faces = 0.5` | **boundary optimum** ✓ (ceil(d/2) coordinates on a face, non-vanishing gradient; BBOB f5's class) |

Against the COCO classes the preset now has 2 separable families (+ the
boundary case of f5), 3 in class 2, 4 in class 3, 2(+1) in class 4 and 3 in
class 5.  Still absent: ruggedness (f16/f17/f18/f23), noise, discrete
variables; constraints and failures stay presets of their own.

Measured landscape features at d = 5 (means over the 3 instances; the
method and the full table: `planning/results/2026-09-27-wide-preset/tables.md`):

| family | 1−R² quad (box) | 1−R² quad (±1 at opt) | FDC | interaction | neutral | f(centre) quantile | L-BFGS-B hits |
|---|---|---|---|---|---|---|---|
| ellipsoid | 0.00 | 5e-28 | +0.43 | 0.06 | 0 | 0.40 | 1.00 |
| different_powers | 0.09 | 0.21 | +0.68 | 0.08 | 0 | 0.22 | 1.00 |
| bent_cigar | 0.33 | 0.02 | +0.70 | 0.11 | 0 | 0.26 | 1.00 |
| rosenbrock | 0.13 | 0.19 | +0.85 | 0.09 | 0 | 0.15 | 0.85 |
| sharp_ridge | 0.03 | 0.03 | +0.88 | 0.03 | 0 | 0.06 | 0.05 |
| attractive_sector | 0.05 | 0.08 | +0.47 | 0.11 | 0 | 0.19 | 0.92 |
| step_ellipsoid | 0.00 | 0.10 | +0.58 | 0.23 | **0.57** | 0.26 | 0.00 |
| rastrigin | 0.16 | 0.93 | +0.91 | 0.66 | 0 | 0.09 | 0.00 |
| ackley | 0.07 | 0.40 | +0.98 | 0.56 | 0 | 0.13 | 0.00 |
| schwefel_sep | 0.74 | 0.01 | **−0.03** | **0.00** | 0 | 0.53 | 0.00 |
| styblinski_tang_sep | 0.04 | 0.02 | +0.62 | **0.00** | 0 | 0.18 | 0.17 |
| lunacek_box | 0.14 | 0.93 | +0.35 | 0.67 | 0 | 0.05 | 0.00 |
| gallagher21 | 0.75 | 0.26 | +0.35 | 0.23 | 0 | 0.36 | 0.07 |
| levy_embed | 0.34 | 0.04 | +0.53 | 0.31 | 0 | 0.41 | 0.48 |
| rosenbrock_edge | 0.03 | 0.07 | +0.83 | 0.05 | 0 | 0.35 | 0.58 |

* Only `ellipsoid` is a quadratic near its optimum (1−R² 5e-28); the next
  best are 0.01–0.03 (schwefel_sep, bent_cigar, sharp_ridge,
  styblinski_tang_sep).  The features separate what they should: the two
  separable families have interaction 0.00, step_ellipsoid is the only one
  with neutral steps, schwefel_sep the only one with FDC ≈ 0 (deceptive),
  lunacek_box and gallagher21 the lowest positive FDC.
* **The centre is still a head start on some families** (f(centre)
  quantile 0.05–0.15 on sharp_ridge, lunacek_box, rastrigin, ackley,
  rosenbrock): the minimum distance takes away the free *hit*, not the
  fact that a centre point is on average closer to a uniform optimum than
  a random one.  On schwefel_sep (0.53) and levy_embed (0.41) the centre
  is an average point.  Removing the advantage fully would need optima
  biased *away* from the centre, which is a bias of its own.

### 68.3 Cost, and a reduced grid for the GP baselines

`measure.py cost` (the plan's `RUNNER_SECONDS` estimate; runner-hours are
the sum of the shards' estimates, i.e. what the run bills):

| grid (5 seeds, 20·d and 100·d, d 2/5/10) | core | trq | qLogEI | TuRBO1 | SMAC | total | shards |
|---|---|---|---|---|---|---|---|
| free, q 1/4/16/64 (the grid of record) | 2.4 | (2.0) | 124.6 | 13.7 | 13.8 | 154.5 | 140 |
| wide, q 1/4/16/64 | 5.5 | 4.4 | 363.2 | 39.4 | 35.7 | 443.8 | 403 — **refused** (> 256) |
| wide, q 1/4 | 2.8 | 2.3 | 161.9 | 33.2 | 35.7 | 235.9 | 215 |
| **proposed**: wide, q 1/4, qLogEI at 20·d only | 2.8 | 2.3 | 34.7 | 33.2 | 35.7 | **108.7** | 100 |

qLogEI at d = 5, 100·d is 2527 s a run (p90) and 45 runs a unit: it is
two-thirds of any wide grid that contains it.  **Proposal:** run `wide` at
q ∈ {1, 4} (the q range §66 measured and the selector's first target) in
two dispatches — `-f presets=wide -f qs=1,4 -f groups=core,trq,TuRBO1,SMAC`
(65 shards, ≈ 74 runner-h) and `-f presets=wide -f qs=1,4 -f budgets=20
-f groups=qLogEI` (35 shards, ≈ 35 runner-h) — and aggregate the two
together (as §65 combined runs).  The pool at 100·d then has no qLogEI,
as at d10/100·d on free already.  Core alone (plus trq) is cheap at any
grid: 10 runner-h for everything.

### 68.4 Local smoke: does the preset discriminate?

Core + trq at d 2/5, 100·d, q 1/4, the 5 roster seeds, 4 processes niced
(≈ 70 min).  AOCC at q = 1, `aocc_time` at q = 4, means over 5 seeds × 3
instances, descriptive and in sample (unpaired, no CIs).  Full per-family
tables for all four cells: `planning/results/2026-09-27-wide-preset/tables.md`.
d = 5, q = 1:

| family | RR_CMA | Blocks | RR_TRQ | Bl3_TRQ | COBYQA | IPOP | NGOpt | OptCMA | TPE | PyBOBYQA |
|---|---|---|---|---|---|---|---|---|---|---|
| ellipsoid | 0.011 | 0.004 | **0.910** | 0.846 | 0.153 | 0.004 | 0.014 | 0.015 | 0.000 | 0.120 |
| different_powers | 0.338 | 0.368 | 0.509 | 0.405 | **0.512** | 0.306 | 0.306 | 0.323 | 0.301 | 0.491 |
| bent_cigar | 0.012 | 0.057 | **0.208** | 0.132 | 0.084 | 0.004 | 0.038 | 0.015 | 0.000 | 0.134 |
| rosenbrock | 0.103 | 0.124 | 0.367 | 0.182 | **0.435** | 0.069 | 0.097 | 0.099 | 0.095 | 0.189 |
| sharp_ridge | 0.095 | 0.136 | **0.227** | 0.140 | 0.181 | 0.083 | 0.115 | 0.090 | 0.063 | 0.173 |
| attractive_sector | 0.154 | 0.163 | 0.066 | 0.175 | **0.215** | 0.106 | 0.137 | 0.155 | 0.142 | 0.108 |
| step_ellipsoid | **0.219** | 0.164 | 0.111 | 0.129 | 0.101 | 0.185 | 0.202 | 0.214 | 0.182 | 0.053 |
| rastrigin | 0.075 | **0.087** | 0.065 | 0.081 | 0.056 | 0.076 | 0.076 | 0.081 | 0.084 | 0.045 |
| ackley | 0.261 | 0.298 | 0.153 | 0.266 | **0.655** | 0.230 | 0.269 | 0.250 | 0.203 | 0.233 |
| schwefel_sep | 0.000 | **0.010** | 0.000 | 0.000 | 0.000 | 0.000 | 0.008 | 0.000 | 0.000 | 0.000 |
| styblinski_tang_sep | 0.176 | 0.206 | **0.535** | 0.435 | 0.066 | 0.109 | 0.250 | 0.166 | 0.126 | 0.419 |
| lunacek_box | 0.062 | 0.063 | 0.058 | 0.071 | **0.084** | 0.055 | 0.060 | 0.058 | 0.060 | 0.031 |
| gallagher21 | 0.133 | 0.148 | 0.136 | 0.158 | **0.244** | 0.133 | 0.151 | 0.181 | 0.142 | 0.153 |
| levy_embed | 0.547 | 0.700 | 0.225 | 0.774 | **0.932** | 0.547 | 0.661 | 0.574 | 0.517 | 0.457 |
| rosenbrock_edge | 0.088 | 0.095 | **0.540** | 0.392 | 0.472 | 0.111 | 0.128 | 0.050 | 0.059 | 0.334 |
| **mean** | 0.152 | 0.175 | 0.274 | 0.279 | **0.279** | 0.135 | 0.167 | 0.151 | 0.131 | 0.196 |
| mean ex-ellipsoid | 0.162 | 0.187 | 0.229 | 0.239 | **0.288** | 0.144 | 0.178 | 0.161 | 0.141 | 0.201 |

(RR_CMA = `RoundRobin_CMAES`, Blocks = `Blocks_warm_CMAES_JSO`, RR_TRQ =
`RoundRobin_TRQ`, Bl3_TRQ = `Blocks_warm_CMAES_JSO_TRQ`, COBYQA =
`RoundRobin_COBYQA`, one seed-invariant run per instance; IPOP = pycma
IPOP, OptCMA / TPE = Optuna.)

Means over the 15 families, all four cells:

| cell | best mean | best ex-ellipsoid | family winners (arm: families) |
|---|---|---|---|
| d2 q1 | RR_TRQ 0.443 (Bl3_TRQ 0.424, PyBOBYQA 0.417) | RR_TRQ 0.406 | RR_TRQ 5, PyBOBYQA 4, COBYQA 3, Bl3_TRQ, RR_CMA, Blocks 1 each |
| d2 q4 | RR_TRQ 0.396 (Bl3_TRQ 0.390) | Bl3_TRQ 0.360 | RR_TRQ 7, Bl3_TRQ 7, COBYQA 1 |
| d5 q1 | COBYQA 0.2794, Bl3_TRQ 0.2791 (a tie) | COBYQA 0.288 | COBYQA 7, RR_TRQ 5, Blocks 2, RR_CMA 1 |
| d5 q4 | Bl3_TRQ 0.257 (RR_TRQ 0.241) | Bl3_TRQ 0.217 | RR_TRQ 6, Bl3_TRQ 5, RR_CMA 3, COBYQA 1 |

**Reading.**

* **It discriminates.**  Four to six different arms win a family in each
  q = 1 cell, and the per-family spread (best − worst arm) runs from 0.01
  (schwefel_sep at d = 5) to 0.91 (ellipsoid).  The arms have *profiles*
  now, which is what a selector needs: the model-based arms own the smooth
  and the boundary families (ellipsoid, bent_cigar, rosenbrock_edge,
  styblinski_tang_sep), COBYQA's large initial trust region owns the
  funnels with ripples (ackley 0.655, levy_embed 0.932, gallagher21), the
  CMA-ES portfolios are best or level on the plateaus and the regular
  multimodal families (step_ellipsoid, rastrigin), and the baselines lead
  nowhere at d = 5 (PyBOBYQA leads 4 families at d = 2, q = 1).
* **Ellipsoid no longer decides the mean**: at d5/q1 the gap between
  RR_TRQ and Blocks is 0.099 with it and 0.042 without; on free the ex-ellipsoid
  view flipped the sign of several comparisons (§66.3).  The mean still moves with the
  ellipsoid by up to 0.045 (RR_TRQ at d = 5): 1/15 of the weight on a
  0.9 spread.
* **The quadratic arms' lead is broader than §66 could show**, but not
  uniform: RR_TRQ collapses on `levy_embed` at d = 5 (0.225 against 0.932
  for COBYQA and 0.700 for Blocks) — with 3 of 5 directions exactly
  neutral, presumably its model or rank test is misled by directions
  without curvature; not investigated — and on attractive_sector (0.066).  As the third Blocks arm it keeps
  most of Blocks' robustness (levy_embed 0.774).  A wide preset is how
  such failure modes of a new arm show up before a claim.
* **Deception is unsolved by everyone**: schwefel_sep scores 0.000–0.010
  for every arm at d = 5 (and ≤ 0.304 at d = 2); lunacek_box ≤ 0.138 at d = 2
  and ≤ 0.084 at d = 5.
  schwefel_sep at d ≥ 5 is a "nothing works at this budget" class: it adds
  no ranking information there (see 68.5).
* **Separable families reward axis-aligned steps**: RR_TRQ's coordinate
  design (c ± r·e_i) makes it the best arm on styblinski_tang_sep at d = 5
  (0.535; COBYQA 0.066).  That is the separability feature working as
  intended — a signed permutation keeps the axes, so coordinate methods
  may exploit it.  One oddity: RR_TRQ on styblinski_tang_sep at d = 2 is
  0.084 at q = 1 but 0.593 at q = 4; not investigated.

### 68.5 Concerns and next steps

1. **In sample, one local run.**  Seeds 42/7/1234/2025/3, no CIs, no
   pairing, 100·d only, no 20·d, no d = 10, no GP baselines.  Nothing here is
   a claim about any arm; it only shows the preset separates them.
2. **schwefel_sep at d ≥ 5** scores ~0 for every arm (a coordinate in
   the second-best basin costs about 120 or more, above AOCC's 1e2 upper
   target).  Options: keep it as the "no arm works" class (a selector
   should learn to fall back), or rescale its values so partial progress
   registers.  Left as is; decide after the runner run.
3. **The centre advantage is reduced, not removed** (68.2): the free hit
   is gone, a centre start is still on average closer.  COBYQA and CMA-ES
   start at the centre, PyBOBYQA and TuRBO at random points.
4. **The sealed set does not contain the new classes.**  It is built from
   the free, shapes, constrained and failure families; changing it is a
   contract change (roadmap §3) and was not done.  Before a claim on the
   wide classes, a sealed counterpart (a fresh battery seed) is needed.
5. **Duplicates across presets.**  Eight wide families share labels (and
   instances) with free / shapes: results of the same arm on those
   instances are the same runs, not new evidence.
6. **Still missing**: ruggedness (Weierstrass, Katsuura, Schaffers),
   noise, discrete variables, and real-world problems.  Constraints and
   failure regions stay their own presets.
7. **Cost**: the full wide grid is refused (> 256 shards, ≈ 444 runner-h);
   use the reduced grid of 68.3.
8. **Next**: the runner run of 68.3 on fresh seeds, the per-family table
   and the ex-ellipsoid view as in §66; then the wide preset is the
   selector's development set (roadmap §4 A), with the features of 68.2
   (f-scale quadratic R², interaction, neutrality, FDC) as candidates for
   `features.py`.

## 69. Why RoundRobin_TRQ collapses on three wide families: a start radius (levy_embed), a kink the model cannot fit (attractive_sector), and a restart that steps back into its own tabu ball (styblinski_tang_sep) — the last one fixed (2026-09-27)

> **In sample.**  The cases, the diagnosis and the fix come from the §68
> smoke (wide preset, roster seeds 42/7/1234/2025/3), and the before/after
> numbers are on those cells and on §66's free cells.  One check is out of
> sample: a fresh wide battery (battery seed 20260927, §69.4).
> **RoundRobin_TRQ at q = 1 is nearly seed-invariant** (box-centre start;
> the only randomness is a random restart point when the archive offers no
> non-tabu centre, and the space-filling geometry fallback, which leave a
> small residual spread): each q = 1 cell is essentially 3 runs per family,
> not 15, and CIs over its "seeds" are not reported.
> Numbering: §67 is the 12-seed confirmation of §65.
>
> **Erratum pointer for §66.3** (§66 itself is not edited): the same
> near-invariance holds for RR_TRQ at q = 1 on the free preset, so §66.3's
> RR_TRQ q = 1 CIs, and the "CI above 0 in 10 of 12 cells" count that
> includes those cells, carry only the reference's seed variance (and the
> arm's residual restart randomness) — the caveat §66.1 states for COBYQA
> applies to them too.

**Question.**  §68 found RR_TRQ far behind COBYQA on `levy_embed` (d = 5,
q = 1: 0.225 against 0.932) and on `attractive_sector` (0.066), and an
unexplained gap on `styblinski_tang_sep` at d = 2 (0.084 at q = 1, 0.593
at q = 4).  What causes each, and what is a principled fix?

**Method.**  An instrumented copy of the arm (per proposal: radius,
centre, kind, the model's n_fit, weak directions, Hessian eigenvalues,
gradient norm, ‖H_u‖ of the least-change prior; per step result: the ratio
ρ, the radius before and after; restarts; whether a step landed in a tabu
ball) on all 3 instances of each case, then variants screened on the wide
preset at q = 1 (one seed: nearly seed-invariant) and at q = 4 (5 seeds), then
the measurement below.  Tables: `planning/results/2026-09-27-trq-diagnosis/tables.md`.

### 69.1 The three cases

**levy_embed (d = 5, k = 2 effective directions): the start radius, not
the neutral directions.**  Per instance (old arm, q = 1): 284–296 steps,
53–65 % with ρ < 0, 3–4 restarts; inst0 ends in a Levy ripple minimum at
−7.009 (f* −8.170), inst1 at −76.94 (f* −80.48), inst2 reaches f* late
(AOCC 0.338).  The neutral-direction hypothesis does not hold: the model's
Hessian has the expected ~0 eigenvalues in 3 directions, the inf-norm
trust region caps every step at the radius (no huge steps), ‖H_u‖ stays
at 1e3–6e3 (no blow-up), and the radius does not collapse abnormally
(it reaches 1e-7 only by converging).  A direct test: RR_TRQ on Levy
*without* an embedding (`levy_full`, all 5 directions active, same
placement) is as bad at d = 5 (0.19–0.23 per instance) as the embedded one
(0.19).  What differs from COBYQA is the scale the model sees: the bridge
passes `initial_tr_radius` = 0.1·box width, but with `scale=True` SciPy's
COBYQA reads it in [−1, 1]^d and caps it at 1, so its first design is
c ± min(0.05·max width, 0.5) of each axis's width (checked: from 0 in
[−5, 5]², its first points are (±5, 0), (0, ±5); on a box of width 2 the
offset is 0.1 of the axis, on width 100 it is capped at 0.5).  On the
±5 family boxes COBYQA starts at **0.5** of the axis, TRQ at 0.1.  At 0.1 the quadratic fits
Levy's ripples and converges into one; at 0.5 it fits the funnel.  RR_TRQ
with `radius_init = 0.5`: levy_embed d5/q1 0.919 (COBYQA 0.932), levy_full
d5 0.24 (the 5-D problem stays hard), levy_embed d2 0.95.  So the low
effective dimension makes the funnel visible to a *large* model; it is
not what traps the arm.  (The `cobyqa.py` docstring said the first step
explores ~10 % of the largest axis; with scaling it is
min(0.05·max width, 0.5) of each axis — 50 % here.  Corrected in this PR,
documentation only.)

**attractive_sector (d = 5): a C¹ kink the quadratic cannot fit at any
radius.**  No restarts; per instance 40–46 % of the steps have ρ < 0 and
another 11–15 % ρ < 0.1; the radius falls to 2e-5…1e-4 and the arm
crawls along a sector boundary (inst0: f from 150 to 55 over 480
evaluations, f* 42.3).  BBOB f6 scales z_i by 100 on one side of each
hyperplane z_i = 0: the curvature jumps by 1e4 across it, so a quadratic
through points on both sides is biased by a fixed fraction of the
quadratic term whatever the radius, and the least-change prior carries the
curvature of one side into the other (‖H_u‖ 1e6–2e7 in u units, which is
the real curvature, not a runaway).  Screen (q = 1, d = 5, 8 families):
dropping the prior entirely lifts attractive_sector to 0.173 but costs
ellipsoid 0.910→0.802, rosenbrock 0.367→0.237, bent_cigar 0.208→0.145;
resetting the prior after a step with ρ < 0: 0.137 with similar losses;
a NEWUOA-style least change (Frobenius norm of the Hessian change only,
gradient unpenalised): 0.038 and worse on every smooth family; the
ρ/Δ split below: 0.099.  No change here is free, so none was made:
attractive_sector stays a known weak family of the arm (COBYQA 0.215,
Blocks 0.163).

**styblinski_tang_sep (d = 2): a restart that steps back into its own
tabu ball.**  At q = 1 all three instances descend from the box centre
into the wrong basin in one coordinate (f − f* = 14.14 on each, one
coordinate's gap) within ~20 evaluations and converge by ~n = 42.  After
the restart the next centre is the best non-tabu point — just outside the
tabu ball, in the same basin.  Its model steps go straight back into the
ball: they improve on the centre (ρ 0.7–1.0, so the ratio test *grows*
the radius), but no point inside a tabu ball can become a centre, so the
centre never moves.  141/143/138 of ~180 steps per instance are such
steps: the arm spends 150+ of its 200 evaluations in place.  At q = 4 the
first batch holds 2d + 1 points before the first model step (d + 1 at
q = 1), the first descent differs and found the global basin on inst0 and
inst1 (AOCC 0.85) — the loop happens there too (60–124 steps per run),
but after the global minimum was found; inst2 stays at 0.084 at q = 4
too.  So the q1/q4 gap is basin luck of one deterministic path, made
permanent by the loop.  The loop also occurs on levy_embed (2–18 steps per
instance) and on sharp_ridge (137 on d5/inst0).

### 69.2 The fix: a step that improves into a tabu ball makes its centre tabu

`TrustRegionQuadratic._into_tabu`: when a step's result is better than its
centre's value but lies in a tabu ball (the same inf-norm test as the
centre choice), the centre joins the tabu list and the arm restarts from
the next best non-tabu point (or a random one) at `radius_init`, prior
reset — a restart without the convergence.  In-flight steps whose centre
has become tabu no longer move the radius (q > 1); a stale catch (q > 1,
the step's centre is no longer the current one) adds its centre to the
tabu list but resets the radius and prior only when the current centre
lies in the new ball (review of #386; the q = 4 rows below are from that
version).  The test uses the evaluated point.  No new parameter; TRQ
stays opt-in.  Tests (`tests/test_heuristic_trust_region.py`): a 2-D
two-basin function (separable Styblinski–Tang on [−5, 6]², whose box
centre lies in the local basin) where the arm now reaches the global
minimum within 300 evaluations and, with the catch disabled, stays at
the local minimum for the whole budget with one restart (the old
behaviour); the catch by hand; an into-tabu step that does not improve is
an ordinary failure; an in-flight step of an abandoned centre moves
nothing.

Variants screened and **not** adopted (q = 1 on the wide preset, seed 42;
q = 4 5 seeds where given):

| variant | wide d2 q1 | wide d5 q1 | wide d2 q4 | wide d5 q4 | note |
|---|---|---|---|---|---|
| old arm | 0.442 | 0.274 | 0.396 | 0.241 | |
| **tabu catch (adopted)** | **0.474** | **0.276** | **0.409** | **0.245** | |
| BOBYQA ρ/Δ split alone | 0.403 | 0.281 | | | ρ is a floor for Δ; ×0.1 when a step fails at Δ = ρ on a full model |
| ρ/Δ split + tabu catch | 0.474 | 0.283 | 0.408 | 0.247 | adds nothing measurable to the catch |
| a failed step instead of a catch | 0.418 | 0.265 | 0.393 (2 seeds) | | the arm then converges on the ball's rim |
| catch, but lift a ball when a step beats its centre | 0.475 | 0.277 | | | does not recover sharp_ridge |
| radius_init 0.5, old arm (diagnostic) | 0.514 | 0.333 | | | a constant, see 69.5 |

The ρ/Δ split (BOBYQA's resolution ρ as a lower bound of the step radius
Δ, ρ reduced ×0.1 only when a step fails at Δ = ρ with 2d + 1 fit points)
converges faster and restarts sooner, which alone *hurts* at d = 2
(step_ellipsoid 0.698→0.366, levy_embed 0.830→0.703): each earlier restart
fell into the loop.  With the catch it is neutral.  The arm already has
BOBYQA's model-improvement step (a failed step on a thin model asks for a
geometry point before shrinking); requiring 2d + 1 fit points before the
first model step (BOBYQA's initial design) was worse on the d = 5 subset
(0.276 vs 0.331).  A curvature floor in the neutral directions was not
tried: nothing in 69.1 points at them.

### 69.3 Before / after

RoundRobin_TRQ, wide preset, 100·d (per family: tables file; the final
version after review #386 — at q = 1 it differs from the first version on
one run, sharp_ridge d2/inst1, by +0.0004; the q = 4 rows moved by at most
0.011 on a family).  q = 1 has no CI (nearly seed-invariant):

| cell | before | after | Δ | ex-ellipsoid Δ | families that moved > 0.01 |
|---|---|---|---|---|---|
| d2 q1 | 0.443 | 0.475 | +0.032 (n/a) | +0.035 | styblinski_tang_sep 0.084→0.546, rastrigin 0.091→0.253, levy_embed 0.830→0.889, ackley 0.524→0.571; **step_ellipsoid 0.698→0.557, sharp_ridge 0.293→0.214**, gallagher21 0.304→0.277 |
| d2 q4 | 0.396 | 0.410 | +0.015 [+0.004, +0.025] 5/5 | +0.016 | styblinski_tang_sep 0.593→0.710, levy_embed 0.704→0.749, step_ellipsoid 0.285→0.327, gallagher21 0.275→0.300; rastrigin 0.145→0.129 |
| d5 q1 | 0.274 | 0.276 | +0.002 (n/a) | +0.002 | gallagher21 0.136→0.239, ackley 0.153→0.192; **sharp_ridge 0.227→0.150, levy_embed 0.225→0.173** |
| d5 q4 | 0.241 | 0.244 | +0.003 [+0.000, +0.006] 5/5 | +0.003 | styblinski_tang_sep 0.523→0.546, gallagher21 0.122→0.133 |

Blocks_warm_CMAES_JSO_TRQ, wide: within ±0.001 in every cell (the third
arm rarely converges and restarts inside a portfolio).

Free preset (§66's cells, d 2/5/10, 20·d and 100·d, q 1/4): **every
20·d cell and every d = 10 cell is unchanged** for both specs (Δ 0.000;
the arm does not restart there), Bl3_TRQ is unchanged everywhere.
RR_TRQ: d2/100·d/q1 +0.016 (n/a) (rastrigin 0.091→0.253, sharp_ridge
0.285→0.206), d2/100·d/q4 −0.003 [−0.012, +0.006], **d5/100·d/q1 −0.008 (n/a)**
(sharp_ridge 0.227→0.150, ackley 0.156→0.193), d5/100·d/q4 −0.001
[−0.005, +0.002].  (The free and wide sharp_ridge instances are the same
runs.)

**The costs.**  The largest is **step_ellipsoid at d2/q1: −0.141 in
sample (0.698→0.557), −0.126 on the fresh battery (0.337→0.211)** — the
plateaus make "a step that improves into a tabu ball" common, and the
catch leaves regions the old loop kept refining; at q = 4 the same family
gains (0.285→0.327, 0.210→0.221).  Then **sharp_ridge** (−0.079 at d2/q1,
−0.077 at d5/q1, in sample) and **levy_embed at d5/q1 (−0.052,
0.225→0.173)**, the family that motivated part of this work (its real lever
is the start radius, 69.5).

On sharp_ridge the old arm converges
falsely (radius 1e-7 on the kink, 20.30 against f* 17.87 on d5/inst0),
and its loop steps into that ball *did* make progress: one of them
eventually landed outside the ball below the centre, the centre moved,
and the second convergence was f* (AOCC 0.309).  The catch abandons that
region after the first such step (0.145).  Lifting a ball when a step
beats its converged value did not recover it (0.165), and treating the
step as a failure instead of a catch is worse everywhere (table above).
On the fresh battery (69.4) sharp_ridge moves −0.011 (d2) and −0.002
(d5): in sample the loss is concentrated on these 3 instances.  levy_embed
d5/q1 on the fresh battery: +0.016 (0.425→0.441).

### 69.4 Out of sample: a fresh wide battery

Battery seed 20260927 (new instances of all 15 families), RR_TRQ, 100·d;
q = 1 one seed, q = 4 seeds 42/7/1234:

| cell | before | after | Δ |
|---|---|---|---|
| d2 q1 | 0.454 | 0.492 | +0.038 (n/a; styblinski_tang_sep 0.082→0.613, gallagher21 0.128→0.242; step_ellipsoid 0.337→0.211) |
| d2 q4 | 0.412 | 0.426 | +0.014 [+0.008, +0.020] 3/3 |
| d5 q1 | 0.255 | 0.278 | +0.023 (n/a; styblinski_tang_sep 0.339→0.638) |
| d5 q4 | 0.226 | 0.242 | +0.016 [−0.000, +0.031] 3/3 |

The styblinski loop reproduces on fresh instances (0.082 at d2/q1 again)
and the catch removes it.  step_ellipsoid d2/q1 loses on both batteries
(−0.141, −0.126) and gains a little at q = 4 (0.285→0.327, 0.210→0.221):
an open question.

### 69.5 The start radius (diagnostic, not adopted)

RR_TRQ with `radius_init = 0.5` (the first version of the fix included)
against that version alone,
in sample: wide d2 q1 +0.026, d2 q4 +0.059 [+0.014, +0.103] 5/5, d5 q1
+0.053, d5 q4 +0.049 [+0.022, +0.075] 5/5 — levy_embed 0.17→0.92 at
d = 5, ackley 0.19→0.46 (d5 q1); losses on step_ellipsoid d2 q1
(0.557→0.192), bent_cigar d5 q1 (0.208→0.091), styblinski_tang_sep d5
(0.537→0.332).  On free it is mixed: +0.055/+0.048/+0.064 at 100·d q1
(d 2/5/10), but −0.037 [−0.085, +0.010] at d5/20·d/q4 and −0.016 at
d10/100·d/q4 (ellipsoid).  It is one constant chosen after seeing these
cells, so it is a candidate for the fresh-seed / fresh-battery run, not a
default: `TrustRegionQuadratic(radius_init=0.5)` needs no code change.

### 69.6 Reading

* The three collapses have three different causes; only one is a defect
  of the arm's logic (the tabu loop), and that one is fixed.
* The neutral-direction hypothesis is rejected for levy_embed; the
  asymmetry hypothesis holds for attractive_sector (a biased model at every
  radius), but no minimal model change fixes it without losing the smooth
  families; the q1/q4 styblinski gap is basin luck plus the restart loop.
* The fix is a small positive on the wide preset at d = 2 (+0.015…+0.032)
  and neutral at d = 5, confirmed on a fresh battery (+0.014…+0.038), no
  change on the free preset's 20·d and d = 10 cells and on the portfolio
  spec.  Its costs, all at q = 1: step_ellipsoid d = 2 (−0.141 in sample,
  −0.126 out of sample) is the largest, then sharp_ridge (−0.08 on the 3
  in-sample instances, −0.01 out of sample) and levy_embed d = 5 (−0.052
  in sample).
* COBYQA's lead on levy_embed and ackley is consistent with its 5× larger
  start radius on these boxes (RR_TRQ with the same radius closes most of
  it); the converse check, COBYQA started at 0.1, was not run.

### 69.7 Not measured

d = 10 on the wide preset; 20·d on the wide preset; q ≥ 16; the r05
variant for Bl3_TRQ; the r05 variant out of sample; COBYQA with a 0.1
start radius (the converse check); MA-BBOB.

## 70. Selector data pipeline (roadmap §4 A step 1): a shared probe, counterfactual labels per arm, and the headroom — the per-task oracle over five arms is 0.078 above the best single arm, but a pick that does not see the task's own score keeps 0.015 (instance, other seeds) and −0.003 (family, other instances): little learnable headroom on this menu and set (2026-09-27)

> **In sample, descriptive.**  Wide preset (§68), d 2/5, 100·d, q 1/4,
> roster seeds 42/7/1234 — the seeds and battery §68/§69 looked at (RR_TRQ's
> tabu fix of §69 was chosen on them).  No selector is trained; nothing here
> is a claim about an arm.  Numbering: §67 is the 12-seed confirmation.

**Question.**  Roadmap §4 A (Harald, 2026-09-26): probe → select → unleash;
xgboost; the label is the counterfactual normalised regret per arm; features
invariant.  Step 1 is the data: can every arm of a menu be continued from one
shared probe, what do the labels look like, and how much would a perfect
selector gain over the best single arm?

### 70.1 Design

**Menu v0** (`panobbgo/selector_data.py`, `ARM_MENU_V0`, the one place it is
defined): `Blocks_warm_CMAES_JSO`, `RoundRobin_CMAES`, `RoundRobin_TRQ`,
`RoundRobin_COBYQA`, `Blocks_warm_CMAES_JSO_TRQ`.

**Probe.**  A scrambled Latin hypercube of k = 10·d points in the box, seeded
per (base seed, instance), the same for every arm and for q = 1 and q = 4.
Why 10·d: at 100·d it is 10 % of the budget; it is the smallest round
multiple of d with enough points for the full-quadratic fits at the measured
d ≤ 5 (they need 2p: 12 at d = 2, 42 at d = 5, 56 at d = 6 — 10·d suffices
up to d = 6; the §66.4 separator of exact quadratics lives there), whereas 2d + 1 (BOBYQA's initial design) supports only a linear
fit.  ELA practice recommends ~50·d for stable features, which a 100·d budget
cannot pay: the probe features are noisy (the roadmap's risk item).  At
d = 10, 2p = 132 > 100: the full-quadratic features are undefined there with
this probe.  LHS rather than Sobol: stratified on every axis for any k (Sobol
is balanced only at powers of two), and no box-centre point, so no free hit
from the centre bias of §68.2.

**Continuation.**  A new core hook, `StrategyBase.preload_results(results)`:
the probe results are booked in `initialize()` on the storage-resume path —
in the store and published as `new_results` *before* the `start` event, and
counted as dispatched, so with `max_eval = B` the arm evaluates B − k new
points.  It raises after `initialize()`.  With a storage backend the preload
is saved like any booked result (it is part of the run's archive, and a resume
must not re-spend its budget share), so a resumed run gets it from storage —
preloading a run that restored results would book it twice and is refused.
Each arm gets its own copies of the probe results.  How each arm uses the
probe:

* both Blocks specs — unchanged: their CMA-ES / jSO arms already have
  `warm_start="archive"` (m / σ fitted to the archive's top points, jSO's
  population seeded from them); the TR arm's centre is the best archive point;
* `RoundRobin_TRQ` — unchanged: centre = best archive point, the model is
  fitted to the archive near it (probe points included);
* `RoundRobin_CMAES` — continued **with** `warm_start="archive"` + the
  `Archive` analyzer (the registry spec cold-starts at the centre and would
  ignore the probe);
* `RoundRobin_COBYQA` — continued with a new opt-in
  `COBYQA(warm_start="archive")`: at `start` the solve is restarted at the
  best archive point.  **Limitation:** SciPy's COBYQA moves a start within
  its initial radius of a bound onto the bound (or to bound ± radius), and
  the default radius is half of every axis (§69.1), so the start is the best
  probe point *quantised* to {lower face, centre, upper face} per axis — it
  keeps the probe's region, not its point (tested).  COBYQA's interpolation
  set cannot take the archive either.

No arm had to be dropped.  Default behaviour is unchanged (`preload_results`
unused, `COBYQA.warm_start=None`).

**Scores and label.**  Each continuation runs as `harness_families` runs it
(penalty tracker, `sync_evaluation`, virtual clock async, log-normal σ 0.5,
CRN durations per cell, the harness's per-arm seed).  Scores on the remaining
B − k evaluations, with the best-so-far starting at the probe's best: AOCC
over evaluations, `aocc_time` over the continuation's virtual time (horizon
(B − k)·d̄/q; the probe's time is the same for every arm and not simulated),
final log precision.  **Label** = `max_a s_a − s_arm` with s = AOCC at q = 1
and `aocc_time` at q > 1 (0 = the task's best arm; AOCC units, already
normalised by the log-precision range); also stored: the regret on the final
precision, the rank, `best_arm` (first in menu order on a tie) and `n_best`
(arms tied at the top; win counts split a tie equally).

**Features at the probe** (`probe_features`; never raw f or raw x): the
rank-based ELA-lite of `features.landscape_features` (FDC, the five NBC
features, dispersion 10/25 %, rank-R² linear / additive / quadratic,
separability ratio, Hessian condition and sign share) — invariant to
f → a·f + b (a > 0) and to any monotone transform; the new
`features.fscale_features` (adjusted R² linear / additive / quadratic on
standardised f, `flog_quad_gap` = log10(1 − R²_quad), the f-scale
separability ratio, Hessian condition and sign share, y-skewness,
y-kurtosis, tie share) — invariant to f → a·f + b (a > 0), not to other
monotone transforms; both groups are invariant to a rotation, shift and
uniform scaling of x except the additive-model / separability features
(deliberately); context: d, B/d, k/d, (B − k)/d, q.  Tests
(`tests/test_selector_data.py`): invariance under three affine f maps, under
rotation + shift + scale of x, the monotone split (rank features unchanged,
f-scale ones not), a control that the separability features do change under
rotation, and the §66.4 separation (f-scale 1 − R² < 1e-9 on exact quadratics
at d 2/5 while the rank-R² stays < 0.99).  The roadmap's "stuck locally"
features are in-run and per arm; at the probe there is no arm yet — they
belong to the later cycles (B).

### 70.2 The data

`uv run python benchmarks/selector_labels.py run OUT.csv.gz preset=wide dims=2,5 bm=100 qs=1,4 seeds=3 jobs=4 diag=1`
(niced): 90 instances × 3 seeds × 2 q = 540 tasks, 5 arms + 2 diagnostic
cold arms, 11.5 min wall on 4 workers (a COBYQA run adds its solver
subprocess beside its waiting worker).  Every run clean (no exception); every
arm but COBYQA spends exactly B − k; COBYQA ends early (converged, it does not
restart) in 83 % of the tasks and is scored at its final best for the rest,
as in §66.  Its label is therefore a worst case for a COBYQA *selection*: a
selector that picks it would hand the unspent budget to another arm, which
this pipeline does not simulate.  One row per task, 100 columns, 108 KB:
`planning/results/2026-09-27-selector-labels/labels_wide_d2d5_bm100.csv.gz`;
the tables below are from `analysis.md` there (`selector_labels.py analyze`).

### 70.3 Labels

| arm | mean score | mean regret | median | p90 | wins | regret < 0.01 |
|---|---|---|---|---|---|---|
| Blocks | 0.247 | 0.231 | 0.136 | 0.647 | 11 % | 16 % |
| RR_CMAES (warm) | 0.208 | 0.270 | 0.172 | 0.719 | 8 % | 14 % |
| RR_TRQ | **0.400** | **0.078** | 0.000 | 0.246 | 48 % | 57 % |
| RR_COBYQA (warm) | 0.283 | 0.195 | 0.104 | 0.586 | 17 % | 22 % |
| Blocks_TRQ | 0.349 | 0.129 | 0.068 | 0.367 | 15 % | 22 % |

Wins split ties equally: 17 of the 540 tasks tie at the top — all
schwefel_sep (4 at d = 2, 13 at d = 5), all five arms at score 0 — and `best_arm`
gives all of them to the first in menu order, Blocks (the first push of this
section counted them that way: Blocks 14 %, schwefel_sep "Blocks wins 64 %").
RR_TRQ is the best arm on average in every (d, q) cell (mean regret
0.050–0.106) and wins about half the tasks; every arm wins somewhere (8–17 %
each).  Per family RR_TRQ owns the smooth and separable ones (ellipsoid and
levy_embed 92 % of the tasks, rosenbrock_edge 72 %, styblinski_tang_sep 67 %)
and loses where §68/§69 said it would: step_ellipsoid (RR_CMAES wins 47 %,
Blocks best on average), schwefel_sep (COBYQA best on average and 32 % of
the wins, Blocks 26 % — COBYQA's face-snapped start of 70.1 lands near the
face optimum),
rastrigin, attractive_sector and lunacek_box (Blocks_TRQ best on average),
ackley (COBYQA best on average).  The labels are heavy-tailed (p90
0.25–0.72): where an arm loses, it loses a lot.

### 70.4 Headroom: oracle vs single best arm (the key number)

SBS = the arm with the best mean score in hindsight (RR_TRQ in every scope);
gap = oracle − SBS = the SBS's mean regret; CI = cluster bootstrap over the
90 instances.  Instance and family oracles in two forms: **in sample** (the
arm with the best mean over the instance's 3 seeds, resp. the family's tasks
at that d and q, *including the task's own score*) and **leave-one-out**
(the same pick without the task: the instance's other seeds — a mean over
only 2 seeds — resp. the family's other instances).  Only the LOO form is
headroom a selector could in principle reach; the in-sample form, reported
first by this section, credits the pick with the task's own luck.

| scope | tasks | SBS mean | task oracle | **gap** [CI] | instance oracle gap: in sample / **LOO** | family oracle gap: in sample / **LOO** |
|---|---|---|---|---|---|---|
| all | 540 | 0.400 | 0.478 | **0.078** [0.057, 0.104] | 0.051 / **0.015** | 0.032 / **−0.003** |
| ex-ellipsoid | 504 | 0.364 | 0.447 | 0.083 [0.060, 0.108] | 0.055 / **0.016** | 0.034 / **−0.003** |
| d2 q1 | 135 | 0.511 | 0.617 | 0.106 [0.061, 0.161] | 0.072 / **0.022** | 0.039 / **−0.027** |
| d2 q4 | 135 | 0.476 | 0.549 | 0.073 [0.044, 0.111] | 0.041 / **−0.001** | 0.022 / **−0.005** |
| d5 q1 | 135 | 0.321 | 0.402 | 0.081 [0.052, 0.113] | 0.056 / **0.021** | 0.045 / **0.025** |
| d5 q4 | 135 | 0.294 | 0.344 | 0.050 [0.031, 0.070] | 0.036 / **0.018** | 0.022 / **−0.004** |

* **The task oracle is 0.078 AOCC above the best single arm** (0.083
  without ellipsoid) — two to five times §53's cheap-track per-cell oracle
  gap (0.015…0.039).  Its CI barely moves when the SBS is re-chosen in
  every bootstrap resample ([0.057, 0.104] either way; per cell at most
  0.005 narrower): RR_TRQ is the SBS in nearly every resample.
* **Most of it is not learnable from these data.**  Choosing an arm per
  instance from its *other* seeds keeps only **0.015** (the in-sample
  0.051 was mostly the task's own luck), and per family from its *other*
  instances **−0.003** — a perfect family classifier trained on sibling
  instances does no better than always running RR_TRQ.  Caveats that make
  the LOO numbers pessimistic: the instance pick averages only 2 seeds, and
  the family pick sees 2 sibling instances; with more seeds and instances
  both estimates get less noisy.  A probe-feature selector sees less than
  the instance identity, so on this menu and this set the step-2 target is
  small: of the order of the 0.015 instance-LOO gap, not the 0.078 headline.
  Only d5 q1 (0.021 / 0.025) shows both LOO gaps clearly above 0.
* **Context alone gives nothing**: the best arm per (d, q) cell is RR_TRQ
  in every cell, so an NGOpt-style rule on d, budget and q selects the SBS.
  What a selector gains here must come from the landscape features.
* The gap shrinks with q (d5: 0.081 → 0.050): at q = 4 the sequential
  COBYQA drops out of contention.
* In-sample oracle caveats: the SBS is chosen on the same tasks; the task
  oracle is a max over 5 noisy scores (optimistic by construction).  RR_TRQ
  at q = 1 is nearly seed-invariant from a fixed start (§69), but the probe
  varies with the seed here, so its 3 seeds are 3 different runs.

### 70.5 Features vs winners

Spearman ρ of each probe feature with each arm's regret (all 540 tasks) is
weak: |ρ| ≤ 0.30.  The strongest are the f-scale ones: `fr2_quad` /
`flog_quad_gap` (±0.30 for Blocks, ±0.28 for RR_CMAES and RR_COBYQA — these
arms do worse where the probe is well fitted by a quadratic, i.e. where the TR
arms win), `y_skew` (+0.28 / +0.27 / +0.25), `flog10_cond` (+0.27 Blocks),
the rank `nbc_nb_fitness_cor` (+0.22 Blocks); `q` is +0.27 for COBYQA
(sequential).  **RR_TRQ's regret is nearly uncorrelated with every single
feature** (|ρ| ≤ 0.13): where it loses is not visible in one feature — that
is the selector's job (interactions; trees).  By winning arm: the tasks
RR_TRQ wins have the most quadratic probes (mean `flog_quad_gap` −2.41;
Blocks' wins −1.11; weighted by win share) and the most skewed f (1.18 vs
0.61).  `y_ties` is 0 on
every probe (10·d LHS points never land on one step_ellipsoid plateau twice):
a plateau feature needs repeated or nearby points.

### 70.6 Diagnostic: warm vs cold continuation

The registry's cold forms ran beside the menu (same seeds, unlabelled):
RR_CMAES warm − cold +0.014…+0.033 with the CI above 0 in all four cells
(60–73 % of the tasks better); RR_COBYQA +0.031 (d2 q1), −0.010, +0.003,
+0.009, all CIs across 0 — the quantised start (70.1) neither helps nor hurts
on average.  The continuation forms are not unfair to the arms.

### 70.7 Not measured / next

d = 10 (the full quadratic needs a ≥ 14·d probe there); 20·d budgets;
q ≥ 16; free / shapes / MA-BBOB; fresh seeds or a fresh battery; noise,
constraints and failure regions (the probe refuses non-finite values); a
COBYQA warm start with a radius that keeps the point; the bootstrap variance
of the probe features; the "stuck locally" features (in-run, later cycles);
probe sizes other than 10·d.  Next (step 2), with the target stated
honestly: the learnable headroom on this menu and set looks small (70.4:
0.015 instance-LOO, −0.003 family-LOO against 0.078 in-sample per task).
Before training xgboost (leave-instance-out and leave-family-out CV,
reported against the LOO gaps, not the task oracle), widen the data where the
LOO gap could grow — more seeds per instance (the instance-LOO pick averages
only 2), more instances per family, d = 10 with a ≥ 14·d probe, 20·d — and
consider arms that differ more where RR_TRQ loses (step_ellipsoid,
schwefel_sep, rastrigin, attractive_sector) and in-run switching (B), where
the per-task luck a one-shot pick cannot see becomes observable.  The dataset regenerates in
~12 min locally with the command of 70.2; a larger one (> 1 MB) belongs in a
GitHub release asset (`gh release upload <tag> labels.csv.gz`), not in git.

## 71. Failure regions, roadmap §4 D step 1: TRQ loses 47 % of its budget to failures and the population arms 3–6 %; a shared failure model plus TRQ's own handling wins +0.037 AOCC on TRQ, the generic filter alone is neutral on the population arms and hurts TRQ at a boundary optimum (2026-09-27)

> **In sample, descriptive.**  The `failure` preset (4 families × 3
> instances), d 2/5, 100·d, q 1/4, roster seeds 42/7/1234/2025/3, virtual
> clock (async, log-normal σ 0.5), the `measure.py` core path
> (`run_family_harness`).  The same seeds and battery the design was debugged
> on (a few single runs of seed 42, d 2/5); the model's one tuned constant
> (the bandwidth cap, 71.2) was changed once, after a unit test (a partial
> first run with the old cap, d2 q1, 4 seeds, had been looked at — TRQ +0.049,
> the others within noise — and was discarded).  No claim about
> unseen problems.  §67 (the 12-seed write-up) was merged after this
> section was numbered.

**Question.**  Roadmap §4 D (Harald, 2026-09-26): failure regions ("poison
zones") where the objective crashes, returns NaN or times out are modelled,
and every algorithm avoids proposing there.  Step 1: how does each arm
handle failures today, how much budget does each waste, and does a shared
model (opt-in) win it back?

### 71.1 Inventory: what each arm does with a failed point

A crash leaves no result (`failed_evaluations`); a timeout, simulated or
real, is a `NaN` result with `timed_out=True`; both are charged against
`max_eval` at dispatch.  The constraint handlers map a `NaN` value to `+inf`.

| arm | a failed point | 
|---|---|
| `CMAES` | ranked last (penalty `+inf`, crash and timeout alike); an all-failed generation is dropped and resampled, 10 in a row → self-restart; a late failed offspring is dropped.  **Gap:** with more than λ − μ failures in a generation, failed offspring are among the μ recombined parents, with positive weight (the mean is pulled towards the zone).  With active CMA the failures in ranks μ+1…λ get negative weights. |
| `JSO` / `LSHADE` / `DifferentialEvolution` | a failed trial is a lost trial (the target stays, the slot gets its next trial); a failed initial point is redrawn uniformly; a timed-out `NaN` trial loses the comparison.  No memory of where failures were. |
| `TrustRegionQuadratic` | a failed *step* counts as a failed step (radius shrinks, or geometry on a thin model); a failed *geometry* point is dropped — and, since only finite results enter its archive, **proposed again**; a start centre (box centre, or a restart point) that fails is **never replaced**: with no finite point there is no model and no step, and every geometry point around it is drawn in the same zone for the rest of the run. |
| `COBYQA` (subprocess bridge) | the solver gets `+inf` for the point; SciPy's COBYQA then usually stops (`EndedEarly`, 230 of 240 runs, as on the free preset: no restart). |
| `NelderMead` (seeded by `Random`) | builds simplices from the Splitter's results, so crashes never enter; a `NaN` value falls back to an unweighted centroid. |
| `Random` | nothing (uniform in the Splitter's best leaf). |
| `Blocks_warm_CMAES_JSO` | its arms' rules above. |
| external (pycma IPOP/BIPOP, NGOpt, Optuna CMA/TPE, Py-BOBYQA) | the objective answers `NaN`, told to the solver as `+inf` (Optuna: a COMPLETE trial of value `inf`, so TPE learns the region). |

No arm retries a failed point on purpose; TRQ repeats them by accident.

**Waste** (base arms; share of spent evaluations that failed, mean over the
60 runs of a cell; `IOHRunRecord.n_failed`, new):

| arm | d2 q1 | d2 q4 | d5 q1 | d5 q4 | all | ellipsoid hs | rastrigin ball | rosenbrock hs | sharp_ridge boxes |
|---|---|---|---|---|---|---|---|---|---|
| RoundRobin_Random | 0.167 | 0.182 | 0.144 | 0.164 | 0.164 | 0.259 | 0.146 | 0.149 | 0.103 |
| RoundRobin_CMAES | 0.039 | 0.045 | 0.026 | 0.022 | 0.033 | 0.024 | 0.044 | 0.024 | 0.040 |
| RoundRobin_JSO | 0.069 | 0.078 | 0.036 | 0.044 | 0.057 | 0.056 | 0.046 | 0.063 | 0.063 |
| RoundRobin_TRQ | 0.662 | 0.275 | 0.720 | 0.240 | **0.474** | 0.672 | 0.622 | 0.113 | 0.490 |
| RoundRobin_COBYQA | 0.038 | 0.038 | 0.033 | 0.033 | 0.036 | 0.067 | 0.039 | 0.017 | 0.020 |
| RoundRobin_Random_NM | 0.157 | 0.164 | 0.182 | 0.178 | 0.170 | 0.253 | 0.134 | 0.188 | 0.106 |
| Blocks_warm_CMAES_JSO | 0.032 | 0.042 | 0.029 | 0.030 | 0.033 | 0.023 | 0.041 | 0.021 | 0.048 |
| Baseline_pycma_IPOP | 0.059 | 0.059 | 0.033 | 0.033 | 0.046 | 0.034 | 0.039 | 0.059 | 0.051 |
| Baseline_pycma_BIPOP | 0.053 | 0.053 | 0.039 | 0.039 | 0.046 | 0.042 | 0.042 | 0.062 | 0.038 |
| Baseline_NGOpt | 0.133 | 0.127 | 0.028 | 0.191 | 0.120 | 0.208 | 0.067 | 0.129 | 0.074 |
| Baseline_Optuna_CmaEs | 0.067 | 0.074 | 0.034 | 0.043 | 0.054 | 0.052 | 0.055 | 0.043 | 0.068 |
| Baseline_Optuna_TPE | 0.100 | 0.106 | 0.031 | 0.031 | 0.067 | 0.076 | 0.051 | 0.078 | 0.064 |
| Baseline_PyBOBYQA | 0.079 | 0.079 | 0.055 | 0.055 | 0.067 | 0.100 | 0.078 | 0.030 | 0.061 |

* The population methods (panobbgo and external) lose 3–7 %: they sample
  around a mean or a population that moves away from failures by itself
  (failures rank last).  There is little budget to win back there.
* **TRQ loses 47 %**, 66–72 % at q = 1.  72 of its 240 runs spend > 90 % of
  the budget failing: every q = 1 run of `rastrigin` and `sharp_ridge` at
  d = 5 (the box-centre start lies in the ball / a box), `ellipsoid` and
  `sharp_ridge` at d = 2 — the stuck-start defect of the table above.  TRQ
  is nevertheless the best arm on this preset (AOCC 0.345; next Py-BOBYQA
  0.170, COBYQA 0.261 over evaluations), on the instances where it starts
  outside the zones.
* Random-like arms lose about the region's volume share; NGOpt 12 % (its
  d = 2 and d5/q4 configurations are random-search-like).

Reproducibility: the base arms re-run from a second checkout with the
model's code in place gave the same records (624/624 identical), so every
default path is unchanged by the new code.

### 71.2 The shared failure model (v0)

`panobbgo/analyzers/failure_model.py`, `FailureModel` (opt-in analyzer).
It learns from every evaluated point: a finite value is a success; a crash,
a timed-out placeholder or a non-finite value is a failure (kinds are
counted, one model for all; the roadmap's crash/timeout split is left for
later — both cost a full evaluation on this preset).

    p_fail(u) = Σ w_i y_i / (Σ w_i + α),   w_i = exp(-½ |u - u_i|² / h(u)²),
    h(u) = min(h_n, r_k(u)),   h_n = 2 (k / (n V_d))^(1/d)

on box-normalised u, `r_k` the distance to the k-th nearest labelled point
(k = max(3, d + 1)), `h_n` twice the expected k-NN distance of n uniform
points, α = 1 a success pseudo-count; `in_poison(x)` is `p ≥ 0.5`.

* **Few failures:** one isolated failure never reaches 0.5 (except on the
  point itself: a repeat of a known failure has p = 1 — deterministic
  objectives); a zone needs several failures close together.
* **Cheap:** distances to the failures first (the repeat rule reuses them),
  to the successes only for a query within reach of one; growing buffers;
  the guard is amortised (exact on every change up to 50 failures and while
  its share is above 0.8 · `max_share`, otherwise every +10 % of failures or
  points — successes can raise the share too, so far from the threshold a
  cached share may lag; a first amortised version, refreshed on failures
  only, stayed armed at a probe share of 0.535 in the review).  A
  one-point query takes 0.03 ms (d = 2, n = 200) to 0.06 ms (d = 10,
  n = 1000), 0.3 ms at n = 10 000; a guard evaluation 1–110 ms (laptop).
  With no failure `p_fail` is zeros without any work — a run without
  failures is bit-identical (71.5).
* **Never the whole box:** disarmed while more than `max_share` = 0.5 of a
  fixed probe set would be marked (all-fail data, or an early cluster whose
  k-NN vote would cover the box).
* **Sharp near a converging search:** the bandwidth is the local k-NN
  distance, so a boundary optimum is resolved at the search's scale.
* **Honest limit:** within ~h_n of failures it is a smoothed k-NN vote and
  does extrapolate into unsampled space whose nearest labels are failures
  (deeper into a half-space; around an early cluster).  The first version
  (cap `½√d n^{-1/d}`) gave 3 clustered failures a halo of 0.6 box widths;
  a quarter of it could not recognise a box that fails everywhere (unit test
  `all_fail`: 34 % marked); the k-NN-based cap does both.

Not chosen for v0: a half-space fit (logistic / linear SVM) extrapolates a
zone into unexplored space by design — right for this preset's two
half-space families, wrong for the ball and boxes; a GP classifier is too
dear per proposal; an axis-aligned tree is the natural v1 for "a
half-space along one variable".

### 71.3 Integration (opt-in; defaults unchanged)

* **Generic filter** (`FailureModel(filter=True)`): in the main loop, after
  `execute()` and before dispatch, a candidate with `in_poison` is **not
  evaluated and costs no budget**; the proposing heuristic is told through
  a new `predicted_failures` event (relayed to each heuristic's
  `on_failed_evaluations`), so each arm treats it exactly as a failure of
  its own (71.1).  Answered, not resampled: CMA-ES's sampling
  distribution is unchanged — a rejected offspring is ranked like a real
  failure, for free (the alternatives in the task, resampling or treating
  it like a repaired point, would change the distribution or waste the
  sample).  The model learns only from evaluated points.  The filter runs
  before the pull/virtual admission and the budget clamp, so only survivors
  compete for workers and budget.  Liveness: after 10 consecutive
  rejections of one heuristic its next candidate passes; a heuristic
  without a failure hook (`Random`, `NelderMead`), which refills only on
  results, gets `on_new_results([])` for its rejected candidates (without
  it 3 of 120 `Random+fm` q = 1 runs drained their queue and ended early);
  each heuristic's handler is guarded separately in the relay.
  On the virtual clock a pass with rejections asks again at the same
  instant (`VirtualClock.step(retry=True)`), so a rejection costs no
  virtual time.  Budget: `max_eval` counts dispatched points only, as
  before; no conflict (CMA-ES's own IPOP counter counts rejected offspring
  as spent, which only affects its restart schedule).
* **`CMAES(failure_aware=True)`:** failed offspring never get a positive
  recombination weight (the λ − μ gap of 71.1).
* **`TrustRegionQuadratic(failure_aware=True)`:** remembers failed points
  (any arm's) and never proposes one again; uses them as reference points of
  its space-filling geometry; a failed unevaluated start centre is replaced
  by a random one (the stuck start); a model step in a poison zone shrinks
  the radius once per model state and the step at the smaller radius is
  tried (the trust region contracts away from the zone), in the units the
  model was fitted in.

**Erratum (review of #388).**  The first version of the poisoned-step
shrink changed the radius in the middle of a proposal and read the model
in the new units: the retried step had the right length but was the wrong
point — the full-radius minimiser compressed to half its length instead of
the minimiser in the half-radius region — and it carried the full-radius
predicted reduction (the review reproduced pred 0.640 against the model's
0.390 at that point).  Fixed (`_step(..., r_model)`), with a
unit test on the retried step; the TRQ `+fm` / `+filter` / `+aware` cells
and the `Random+fm` / `Random_NM+fm` cells (the relay's refill, above)
were re-run on the fixed code.  The first run had TRQ+fm at +0.038 ± 0.013
(d5 q1 +0.081, ellipsoid d5 q1 +0.140); the others moved by at most 0.001.
A second review fix made the "whole box" guard exact near its threshold
(71.2); it changes the filter's decisions in a few runs, so every arm with
the filter (`+fm`, `+filter`) was re-run once more on the final code: 227
to 240 of 240 records per arm identical, every table cell within 0.003 of
the previous run.  Every number in 71.4 is from this last run (`+aware` has
no filter and is from the run before).

A first `failure_aware` TRQ that also treated every point within 0.3 radii
(geometry) or 10⁻³ radii (steps) of a failure as taken lost 0.36 AOCC on
one d = 2 boundary-optimum run (it could not approach a boundary it had
failed across); only exact repeats are excluded now.

### 71.4 Before / after

Paired on the arm's RNG stream (`seed_name`), per-seed means over the 12
instances, mean ± 95 % t half-width over 5 seeds, seeds up in brackets.
`+fm` = filter + arm handling, `+filter` / `+aware` = one of them.

**Failed share, Δ** (variant − base):

| arm | d2 q1 | d2 q4 | d5 q1 | d5 q4 | all |
|---|---|---|---|---|---|
| RoundRobin_TRQ+fm | −0.495 ± 0.023 | −0.181 ± 0.045 | −0.649 ± 0.008 | −0.152 ± 0.050 | −0.370 ± 0.017 |
| RoundRobin_TRQ+aware | −0.487 ± 0.038 | −0.161 ± 0.061 | −0.618 ± 0.010 | −0.137 ± 0.039 | −0.351 ± 0.015 |
| RoundRobin_TRQ+filter | −0.009 ± 0.015 | −0.037 ± 0.029 | 0 | −0.034 ± 0.032 | −0.020 ± 0.010 |
| RoundRobin_Random+fm | −0.077 ± 0.025 | −0.089 ± 0.014 | −0.031 ± 0.014 | −0.030 ± 0.022 | −0.057 ± 0.008 |
| RoundRobin_Random_NM+fm | −0.078 ± 0.024 | −0.082 ± 0.020 | −0.034 ± 0.033 | −0.029 ± 0.010 | −0.056 ± 0.005 |
| RoundRobin_JSO+fm | −0.018 ± 0.011 | −0.026 ± 0.013 | −0.001 ± 0.002 | +0.001 ± 0.006 | −0.011 ± 0.006 |
| RoundRobin_CMAES+fm | −0.006 ± 0.006 | −0.009 ± 0.005 | +0.000 ± 0.002 | −0.001 ± 0.003 | −0.004 ± 0.002 |
| RoundRobin_CMAES+filter | −0.006 ± 0.006 | −0.006 ± 0.008 | +0.000 ± 0.003 | −0.000 | −0.003 ± 0.004 |
| RoundRobin_CMAES+aware | +0.000 ± 0.006 | −0.005 ± 0.007 | −0.000 ± 0.003 | −0.000 ± 0.003 | −0.001 ± 0.003 |
| Blocks_warm_CMAES_JSO+fm | −0.006 ± 0.003 | −0.011 ± 0.010 | −0.001 ± 0.012 | −0.000 ± 0.004 | −0.005 ± 0.005 |
| RoundRobin_COBYQA+fm | 0 | 0 | 0 | 0 | 0 |

**AOCC, Δ** (`aocc_time` Δ within 0.001 of the AOCC Δ in every cell):

| arm | d2 q1 | d2 q4 | d5 q1 | d5 q4 | all |
|---|---|---|---|---|---|
| RoundRobin_TRQ+fm | +0.050 ± 0.009 (5/5) | +0.013 ± 0.021 (4/5) | +0.077 ± 0.021 (5/5) | +0.009 ± 0.013 (4/5) | **+0.037 ± 0.006 (5/5)** |
| RoundRobin_TRQ+aware | +0.050 ± 0.009 (5/5) | +0.003 ± 0.013 (4/5) | +0.057 ± 0.011 (5/5) | +0.003 ± 0.015 (3/5) | +0.028 ± 0.005 (5/5) |
| RoundRobin_TRQ+filter | −0.070 (0/5) | +0.002 ± 0.003 (3/5) | 0 | +0.000 ± 0.009 (3/5) | −0.017 ± 0.002 (0/5) |
| RoundRobin_Random+fm | +0.005 ± 0.008 (3/5) | +0.000 ± 0.004 (2/5) | −0.000 ± 0.003 (2/5) | +0.000 ± 0.002 (2/5) | +0.001 ± 0.002 (3/5) |
| RoundRobin_Random_NM+fm | +0.009 ± 0.016 (4/5) | +0.003 ± 0.007 (4/5) | +0.000 ± 0.004 (3/5) | +0.002 ± 0.002 (5/5) | +0.004 ± 0.005 (4/5) |
| RoundRobin_JSO+fm | +0.005 ± 0.009 (4/5) | −0.003 ± 0.009 (2/5) | +0.000 ± 0.001 (3/5) | −0.002 ± 0.004 (2/5) | +0.000 ± 0.004 (3/5) |
| RoundRobin_CMAES+fm | +0.003 ± 0.006 (4/5) | −0.003 ± 0.007 (2/5) | −0.001 ± 0.003 (2/5) | −0.001 ± 0.002 (2/5) | −0.000 ± 0.004 (2/5) |
| RoundRobin_CMAES+filter | +0.003 ± 0.004 (5/5) | +0.001 ± 0.007 (3/5) | −0.002 ± 0.003 (0/5) | −0.000 (0/5) | +0.001 ± 0.001 (5/5) |
| RoundRobin_CMAES+aware | +0.002 ± 0.005 (3/5) | −0.001 ± 0.007 (2/5) | −0.001 ± 0.003 (1/5) | −0.001 ± 0.002 (1/5) | −0.000 ± 0.004 (2/5) |
| Blocks_warm_CMAES_JSO+fm | +0.002 ± 0.004 (4/5) | +0.003 ± 0.009 (4/5) | +0.000 ± 0.001 (4/5) | +0.000 ± 0.004 (3/5) | +0.002 ± 0.002 (5/5) |
| RoundRobin_COBYQA+fm | 0 | 0 | 0 | 0 | 0 |

Absolute AOCC (all cells): RoundRobin_TRQ 0.345 → +fm **0.382**; it was
already the best arm here (next: COBYQA 0.261, Py-BOBYQA 0.170 over
evaluations) and the gap grows.

Per family (TRQ): `+aware` and `+fm` fix the stuck starts — `sharp_ridge`
d = 5 q = 1 from 0.000 to 0.142 (failed share −0.95), `rastrigin` d = 5
q = 1 from 0.000 to 0.043 (−0.99), `ellipsoid` d = 5 q = 1 +0.124 (`+fm`,
the filter adds +0.081 there over `+aware`'s +0.044).  **`+filter` alone costs
TRQ −0.280 on `ellipsoid_fhs_crash` d = 2 q = 1** (the optimum on the
boundary of the failing half-space; the q = 1 TRQ is seed-invariant, so
0/5): without the arm handling, rejected steps near the boundary count as
failed steps, the radius collapses, and rejected geometry points are
re-proposed until the streak lets them through.  With the arm handling the
filter's rejections are rare there and the cell is +0.056.

Reading:

1. **The waste is an arm property, not a problem property.**  Population
   methods waste 3–7 %, and the filter's saving on them (0.4–1.1 points)
   buys nothing measurable (|ΔAOCC| ≤ 0.005, every CI across 0).  Random-
   like arms save 3–9 points of budget, also for no measurable AOCC.
2. **The gain is TRQ's own handling** (+0.028 of the +0.037); the shared
   model adds on top of it (+0.009 overall, +0.081 on ellipsoid d5 q1) but
   **hurts without it**.  So "a generic filter that helps every arm" is
   not what was measured: it is neutral on four arms and harmful on one
   whose failure handling is poor.
3. **COBYQA never trips the filter**: it stops after few evaluations with
   isolated failures that never form a zone.
4. `CMAES(failure_aware=True)` changes nothing measurable: generations with
   more than λ − μ failures are rare on this preset.

### 71.5 Free preset: nothing changes without failures

Free preset (5 families × 3 instances), d 2/5, 100·d, q 1/4, seeds 42/7,
every panobbgo arm (the seven above) with and without `+fm`: **840 of 840
paired records identical** (re-checked on the review-fixed code: 840 / 840) (AOCC, `aocc_time`, evaluations, precision,
failures).  With no failure the model is empty, the filter never fires and
the `failure_aware` branches are never taken; the unit test
`test_filter_is_bit_identical_without_failures` checks point-for-point
identity of the evaluated sequences.

### 71.6 Not measured / next

d = 10 and 20·d budgets; q ≥ 16; fresh seeds or a fresh battery (every
number here is in sample); the ball/box shapes at larger shares; timeouts
that cost more (or less) virtual time than a success (here a signalled
timeout takes its drawn duration); a separate crash/timeout model; the
external baselines with the filter (they do not run through the main
loop); rejected-candidate counts per run (the strategy counts
`n_predicted_failures`, the records do not).  Next: (a) take TRQ's
`failure_aware` handling as its default candidate — it is a defect fix
(stuck start, repeated failed geometry) and changes nothing without
failures — after fresh seeds; (b) the filter stays opt-in and off by
default: it pays only together with an arm that handles a failure
sensibly; before a default, a filter that knows the arm (e.g. only
geometry / exploration candidates, never model steps near a boundary) or
an axis-aligned tree model for half-spaces (v1); (c) the `failure` preset
in the `measure.yml` q-sweep (TODO §2 (d)) with the `trq` group.
