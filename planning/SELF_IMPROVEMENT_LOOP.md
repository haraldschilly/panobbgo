# Self-Improvement Loop: Design & Roadmap

**Status:** proposal / living document
**Owner:** Panobbgo maintainers + coding agents
**Related:** `panobbgo/harness.py`, `doc/source/guide_benchmarking.rst`,
`AGENTS.md` — Benchmark Harness section.

## 1. Vision

Run Panobbgo in a loop that *improves itself*: measure → propose a change →
apply → measure → accept if better, revert otherwise — against a battery
of **standardized but parametrically randomized** tests. The loop should be
trustworthy enough to run unattended, and honest enough that a sustained
positive trend means the framework really got better, not that it over-fit
to a fixed benchmark set.

Three properties make this hard, and each maps to a component of the
design below:

1. **Over-fitting**. A fixed benchmark lets an agent tune to quirks
   of the specific instances. ➜ Parametric randomization of problems.
2. **Noise**. Stochastic strategies + a small number of reps produce
   deceptive deltas. ➜ Statistical acceptance rule.
3. **No absolute anchor**. Internal comparisons cannot tell us whether
   Panobbgo is good in absolute terms. ➜ External baselines in the same
   harness.

## 2. Where we are today

What exists (as of 2026-04-19):

- `panobbgo/harness.py` — `BenchmarkHarness` with seeded reproducibility,
  convergence traces, ERT, JSON serialization.
- `benchmark_harness.py` CLI — `run` / `score` / `compare` / `list`.
- `tests/test_harness.py` — ~60 tests.
- Three preset modes: `quick`, `standard`, `full`.
- Single scalar `composite_score` ∈ [0, 1] as the primary metric.
- Manual workflow: a human (or agent) runs `compare` and decides.

What's missing for a true self-improvement loop:

- [ ] Parametric randomization — all problem instances are currently fixed.
- [x] External absolute baselines (Random, scipy DE, scipy dual annealing)
      — shipped 2026-04-20 as `panobbgo/harness_baselines.py` and the
      `--baselines` CLI flag; see `tests/test_harness_baselines.py`.
      `pycma` wrapper still optional — Panobbgo already ships its own
      CMA-ES via the `CMAES` heuristic (`CMAES_Portfolio` / `IPOP_CMAES`
      strategies), so CMA-ES is present on *both* sides of the comparison
      internally.
- [x] Statistically principled accept/reject — shipped 2026-04-21 as
      `panobbgo.harness.statistical_accept` and the `--statistical` flag
      on `benchmark_harness.py compare`.  Bootstrap CI on composite
      delta + per-pair regression guard (§6.2).  Tests in
      `tests/test_harness_stats.py`.
- [ ] A driver that closes the loop: apply change, measure, accept/revert,
      commit.
- [ ] A change catalog — the space of mutations the loop may try.
- [ ] Persistence of the running "ladder" of best composite scores over time.

## 3. Architecture of the loop

```
┌─────────────────────────────────────────────────────────────────┐
│  LOOP DRIVER (new: scripts/self_improve.py)                     │
│                                                                 │
│  for i in range(iterations):                                    │
│      1. baseline = measure(current HEAD)                        │
│      2. proposal = pick_mutation(history)                       │
│      3. apply(proposal)  # in a dedicated git branch/commit     │
│      4. candidate = measure(HEAD with proposal)                 │
│      5. decision = accept_reject(baseline, candidate)           │
│         - statistical test on per-pair scores                   │
│         - hard gate: no pair may regress by > max_regression    │
│      6. if accept: keep commit, advance ladder                  │
│         else:      revert, log, bias sampler away from proposal │
│      7. periodically: full run + external baselines             │
└─────────────────────────────────────────────────────────────────┘
         │                     │                        │
         ▼                     ▼                        ▼
   Parametric            Change catalog          External baselines
   problem sampler       (mutation space)        (scipy DE, pycma, …)
```

## 4. Parametric problem battery

### 4.1 What to randomize

Given a base problem `P` (e.g. Rastrigin), each harness run samples a
**transformed instance** `P̃ = T(P; θ)` with:

| Transform     | Parameter                                | Why                                        |
|---------------|------------------------------------------|--------------------------------------------|
| Translation   | `x* ∼ Uniform(box)`                      | kills hard-coded center exploitation       |
| Rotation      | random orthogonal `Q ∈ O(d)`             | breaks axis-aligned local search advantage |
| Scaling       | diagonal `Λ`, `log₁₀ cond ∼ U[0, 4]`     | stresses second-order / ill-conditioning   |
| Noise         | `f̃(x) = f(x) + σ·ε`, `σ ∈ {0, 1e-3, 1e-2}` | matches noisy black-box use case           |
| Dimension     | `d ∼ choice({2, 5, 10})`                 | prevents per-dim overfit                   |
| Box shift     | box translated with optimum              | the optimum is not at a corner             |

Not every problem supports every transform (Schwefel's optimum is
non-trivial to rotate, Griewank's oscillation pattern couples to rotation).
The sampler declares per-problem capability flags.

### 4.2 Sampler interface (sketch)

```python
@dataclass
class ProblemFamily:
    name: str                        # "Rastrigin"
    base_class: type                 # panobbgo.lib.classic.Rastrigin
    supported_transforms: set[str]   # {"translate", "rotate", "noise", "scale"}
    dim_choices: tuple[int, ...]     # (2, 5, 10)
    f_opt: float = 0.0               # invariant under the supported transforms

class ProblemSampler:
    def sample(self, rng: np.random.Generator) -> ProblemSpec:
        """Draw one concrete instance. Returns a ProblemSpec whose
        `problem_class` is a wrapper that applies T in `eval()` and
        reports the true (transformed) optimum location to the harness."""
```

### 4.3 Reproducibility under randomization

The sampler's `rng` is seeded with
`sha256(base_seed, iteration_id, problem_name, rep) → uint32` — the same
scheme we already use for run seeds. This means:

- An entire loop iteration is replayable from a single `(base_seed,
  iteration_id)` pair.
- A regression flagged by the loop can be reproduced deterministically
  for debugging.
- Across iterations the instances are *different* — which is the point —
  but within an iteration `before` and `after` see the **same** sampled
  instances, so the comparison is apples-to-apples.

### 4.4 Impact on the composite score

The composite-score formula is unchanged. What changes is the *distribution*
it's computed over: it becomes a Monte-Carlo estimate of expected
performance on the problem family, not a point estimate on a fixed
instance. This makes it:

- Higher-variance per iteration (good reps matter more).
- Much less susceptible to over-fitting (the score the agent is climbing
  is the true generalisation signal).
- Comparable across iterations *in expectation* — exact equality is no
  longer expected, and that's correct.

## 5. External absolute baselines

Without external reference solvers, the composite score only answers
"is Panobbgo better than its previous self". We add three baselines, all
fitted into the same `StrategySpec` interface:

| Baseline        | Purpose            | Source                                   |
|-----------------|--------------------|------------------------------------------|
| `Random`        | floor              | uniform samples, best-so-far bookkeeping |
| `SciPyDE`       | competitive ref    | `scipy.optimize.differential_evolution`  |
| `SciPyAnneal`   | competitive ref    | `scipy.optimize.dual_annealing`          |
| `PyCMA`         | state-of-the-art   | `pycma.fmin`                             |

Integration notes:

- Wrap each in an adapter that obeys the harness budget (`max_evaluations`)
  and emits the same convergence trace format.
- Seed each wrapper from the run's SHA-256-derived seed.
- Run baselines only at `--standard` and `--full`, not at `--quick`
  (external solvers have non-trivial per-call overhead).
- Report them in the score table alongside Panobbgo strategies — the
  composite is still averaged only over Panobbgo strategies, but the user
  can see the baseline column.

With baselines in place, the README can quote a single memorable number
like "Panobbgo reaches 82% of CMA-ES's composite score on the standard
battery at equal budget" — which actually tells people something.

## 6. Statistical acceptance rule

Naive "did composite go up?" is too noisy. The loop driver uses:

### 6.1 Per-pair bootstrap CI

For each `(problem, strategy)` pair we have N reps (≥ 5). The pair's score
is a mean of per-rep solve fractions. Bootstrap with 10 000 resamples to
get a 95% CI on the delta `after − before`.

### 6.2 Decision rule

Let `Δ = composite_after − composite_before` and let `r_i` be the per-pair
deltas.

- **Accept** iff:
  - `Δ > ε_accept` (default `0.005`) — moved in the right direction beyond
    measurement noise, **and**
  - lower bound of the bootstrap 95% CI on `Δ` is `> 0` (statistically
    plausible as a real improvement), **and**
  - `min_i r_i > −ε_regress` (default `−0.05`) — no pair regresses
    catastrophically, even if the average improves.

- **Reject** otherwise. Revert the commit.

### 6.3 Anti-cherry-pick guard

Every Kth iteration (default `K = 10`), re-measure the accepted ladder on
a *fresh* random seed. If the ladder drops more than `ε_ladder` (default
`0.02`) on that re-measurement, roll back to the last iteration whose
fresh-seed score is still within tolerance. This catches over-fitting to
the particular stream of sampled problems.

## 7. Change catalog

The mutation space the loop may sample from, in rough order of safety:

1. **Hyperparameter retunes** — `Nearby.radius`, `CMAES.sigma0`,
   `Sensitivity.update_interval`, bandit temperatures. Bounded perturbation
   of current value (log-uniform ±30%).
2. **Strategy portfolio composition** — add/drop a heuristic from a
   strategy, reweight initial priors.
3. **Analyzer parameters** — `Restart.patience`, `Sensitivity` window.
4. **Heuristic code edits** — delegated to a coding agent with a narrow
   task description; applied behind a feature flag if the change is
   non-trivial.
5. **New heuristic / analyzer scaffolds** — manual review required; loop
   can propose but not commit these.

Each mutation has a **rollback plan** (typically `git revert HEAD`). The
loop refuses to apply a mutation whose rollback is unclear.

## 8. Safety rails

- **Dedicated branch** — the loop runs on a branch like
  `auto/improve-YYYYMMDD`. Never touches `main`.
- **Atomic commits** — one commit per accepted mutation. Easy revert.
- **Budget cap** — wall-clock cap per iteration (default 5 minutes for
  quick-mode loop, 1 hour for standard-mode loop).
- **Timeouts** — per-run timeout already enforced by `HarnessConfig.timeout_per_run`.
- **Sanity tests** — before any iteration, run `./test.sh`. A mutation
  that breaks tests is auto-reverted even if composite improved.
- **Human escape hatch** — loop writes a ledger
  (`planning/self_improve_ledger.jsonl`, one line per iteration) that a
  human can audit, and a `STOP` sentinel file that halts the loop.

## 9. Phased rollout

Each phase is independently deliverable and keeps the framework usable.

### Phase 0 — Documentation (this PR)

- `doc/source/guide_benchmarking.rst` — the user-facing guide.
- `AGENTS.md` and `CLAUDE.md` updated with self-improvement context.
- Richer module docstrings on `harness.py` and `benchmark_harness.py`.
- This plan document.

### Phase 1 — Capture an absolute number (~1 day)

- Run `--standard` on main, save as `planning/baseline_standard.json`.
- Publish the current composite score in the README ("Panobbgo current
  score: X.XX on the standard battery at base seed 42").
- Add a `benchmark_harness.py score --json` diff as a CI artefact on
  every PR so reviewers see the before/after.

### Phase 2 — External baselines (shipped 2026-04-20)

- [x] Implemented `Random`, `SciPyDE`, `SciPyAnneal` adapter strategies in
      `panobbgo/harness_baselines.py`.
- [x] Registered as `StrategySpec`s, appended via
      `HarnessConfig(include_baselines=True)` and the `--baselines` flag
      on `benchmark_harness.py run` / `list`.
- [x] Adapters enforce a **hard** evaluation budget via `_BudgetExhausted`
      — external solvers can never overshoot `config.max_eval`.
- [x] Results DataFrame uses the Panobbgo MultiIndex convention so the
      harness convergence extractor works unchanged.
- [x] 22 tests in `tests/test_harness_baselines.py`.
- [ ] `PyCMA` wrapper deferred — Panobbgo's own `CMAES` heuristic already
      provides an internal CMA-ES reference (`CMAES_Portfolio`,
      `IPOP_CMAES`, `BIPOP_CMAES`).  Add later if round-tripping against
      the upstream pycma implementation becomes useful.

### Phase 3 — Parametric randomization (~1-2 weeks)

- Implement `ProblemFamily` and `ProblemSampler`.
- Refactor `_make_*_problems` into family-based factories.
- Add per-family transform capability flags.
- New mode `--randomized-{quick,standard,full}` (fixed modes stay for
  byte-identical reproducibility when needed).
- Extend tests to cover sampler determinism.

### Phase 4 — Statistical acceptance (shipped 2026-04-21)

- [x] Bootstrap CI on the composite delta via
      :func:`panobbgo.harness.statistical_accept` (10 000 resamples by
      default, 95% percentile interval).  Composite CI is built by
      averaging per-pair bootstrap deltas at matching indices, preserving
      the dependence structure between pairs.
- [x] New CLI flag ``compare --statistical`` switches the accept rule to
      §6.2 (delta > `eps_accept`, CI lower bound > 0, no pair regresses
      beyond `eps_regress`).
- [x] Flags: ``--eps-accept``, ``--eps-regress``, ``--n-boot``,
      ``--confidence``, ``--stat-seed``.
- [x] JSON payload (``--json``) gets a ``statistical`` block with composite
      verdict, CI, worst regressing pair, and per-pair CIs.
- [x] 22 tests in ``tests/test_harness_stats.py`` — accept/reject paths,
      regression guard, reproducibility, CLI integration.

### Phase 5 — Loop driver MVP (~1 week)

- `scripts/self_improve.py` implementing §3.
- Start with a **hyperparameter-only** mutation space (§7, item 1).
- Runs for N iterations on a dedicated branch, writes the ledger.
- Produces a human-readable report at the end.

### Phase 6 — Production loop (ongoing)

- Broaden the mutation space.
- Connect the loop to CI (nightly run on a dedicated runner).
- Publish the ladder in the docs.

## 10. Open questions

- **How much compute?** Even `--standard` with 6 strategies × 8 problems ×
  5 reps = 240 runs per iteration. A phase-5 loop aiming for 100 iterations
  is 24 000 runs. Realistic on a single node overnight; less realistic in
  CI. The loop should be cloud-amenable (Dask is already supported).
- **What if the accept rate is too low?** Most proposed mutations will be
  no-ops or regressions. This is normal; the loop needs patience or a
  smarter proposer (a simple Bayesian optimiser over the hyperparameter
  space is a natural upgrade).
- **Composite score stability across dimension sampling.** If we sample
  `d ∈ {2, 5, 10}` we need to *stratify* (same mix of dimensions on both
  sides of the comparison) or the score will be dominated by whichever
  dimension happened to be sampled more.
- **Coordination with `simplify`, `review`, `security-review`.** The loop
  should not run alongside a human PR — race conditions on the branch
  would be ugly. A simple lockfile suffices.

## 11. Success criteria

After Phase 5, the framework is "self-improving" when:

- A loop run of 50 iterations on hyperparameters produces a composite-score
  improvement of ≥ `0.02` on the standard battery, validated against a
  held-out random seed.
- The improvement is *retained* when re-measured a week later on fresh
  samples.
- No regression worse than `−0.02` on any individual pair.
- Total human intervention: approving the PR that merges the accepted
  ladder.

That is the target.
