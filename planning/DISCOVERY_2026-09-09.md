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
| **cmaes** | `ipop_factor=1.5` | 0.7023 | **+0.0886** | [+0.0435, +0.1336] | 3/3 `<--` |
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
resolves to `clip(round(min(18·dim, budget/12)), max(NP_min, 6), 400)` —
36 at *d* = 2 and 83 at *d* ≥ 5 for a 1000-evaluation budget.  A **fixed
`NP_init=30` beats that on every arm and every seed**, by +0.059
(NLSHADE_LBC), +0.079 (L-SHADE) and +0.100 (jSO).  PSO tells the same
story from its own default of 20: `NP=10` gains +0.061 and `NP=40` loses
−0.048, monotonically.

This is what §9 was measuring without naming it.  The canonical DE
population sizes come from papers whose budget is 10⁴·*d* evaluations;
at 10³ *total* a population of 83 gets about twelve generations, which
is not enough for differential evolution to do anything but sample.
`"auto"` already tries to correct for the budget, but its divisor of 12
is far too generous — 30 corresponds to roughly `budget/33`.

Two consequences worth stating plainly:

* **jSO and L-SHADE were not beaten fairly in §14.** With `NP_init=30`
  jSO reaches 0.5581 and L-SHADE 0.4923, against the 0.4584 / 0.4132 that
  produced their 0-out-of-30 win counts.  The oracle bound must be
  recomputed on the tuned arms before concluding that either belongs out
  of the portfolio.
* **The `"auto"` divisor is a library default, not a benchmark knob.**
  If a re-sweep confirms the optimum sits near `budget/30`, the fix
  belongs in `_resolve_auto_np_init`, where it helps every user, not in
  the harness specs.

### What this pass does *not* settle

Each variant was measured alone against the default, so the gains are
not known to compose — `NP_init=30` and `H=20` may well overlap.  The
sweep also brackets rather than locates: `NP_init` was tested only at
`auto` and 30, `ipop_factor` only at 1.5 / 2 / 3, and both winners sit
at the edge of their tested range, so the optimum is plausibly beyond
it.  Three seeds is enough to see an effect this large and not enough to
accept a default; the 12-seed roster decides.
