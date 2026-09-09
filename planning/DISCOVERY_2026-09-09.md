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
