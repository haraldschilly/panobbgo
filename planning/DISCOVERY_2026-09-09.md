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
