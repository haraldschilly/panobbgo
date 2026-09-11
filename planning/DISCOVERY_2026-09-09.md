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
