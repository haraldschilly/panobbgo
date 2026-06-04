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

- [x] Parametric randomization — shipped 2026-04-22 as
      `panobbgo/harness_randomized.py` and the `--randomize` CLI flag.
      Four default families (Rastrigin / Ackley / Rosenbrock / DeJong)
      with translate + rotate + scale + noise transforms, SHA-256-derived
      instance seeds, and a `--randomize-iteration` flag that keeps
      before/after runs aligned within an iteration.  Tests in
      `tests/test_harness_randomized.py`.
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
- [x] Anti-cherry-pick guard (§6.3) — shipped 2026-04-26 as
      `LoopConfig.guard_interval` / `guard_eps_ladder` /
      `guard_iteration_offset`.  Periodic ladder re-validation on a
      fresh seed; rollback on drift.
- [x] A driver that closes the loop: apply change, measure,
      accept/revert — shipped 2026-04-23 as
      `panobbgo.self_improve.SelfImprover` and `scripts/self_improve.py`.
- [x] A change catalog — `panobbgo.self_improve.MutationCatalog` and
      `default_catalog()` cover hyperparameter retunes from §7.1.
- [x] Persistence of the running "ladder" of best composite scores
      over time — `LadderEntry` + JSONL ledger
      (`planning/self_improve_ledger.jsonl`).  Each entry stores its
      ``last_validated_score``, refreshed every time the guard
      re-measures it.
- [x] Hold-out validation set — shipped 2026-05-08 as
      `LoopHoldoutRecord` and the `LoopConfig.holdout_*` knobs.
      End-of-loop re-measure on an *independent* `base_seed`
      catches overfit to the training base_seed family that the
      anti-cherry-pick guard cannot see.  CLI: `--holdout-base-seed`,
      `--fail-on-overfit`.  Extended 2026-05-16 with multi-seed
      hold-out — `LoopConfig.holdout_base_seeds` is a list of
      independent seeds; one `LoopHoldoutRecord` is written per seed
      and the CLI aggregates with worst-case drift / any-overfit
      semantics.  CLI: `--holdout-base-seeds 1234,5678,9012`.
      Extended 2026-05-17 with bootstrap-CI aggregation across
      hold-out records — :func:`aggregate_holdout_drift` pools
      per-iteration paired drifts across all hold-out seeds (each
      `LoopHoldoutRecord` now persists `seed_iteration_scores` and
      `top_iteration_scores`) and bootstrap-resamples the mean.  CLI
      adds `--fail-on-overfit-ci` for a stricter, statistically
      principled exit-on-overfit rule that fires iff the CI's upper
      bound falls below `-holdout_eps_overfit`.

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
| Dimension     | cyclic stratification over `dim_choices` (rep `i` → `dim_choices[i % k]`) | prevents per-dim overfit *and* dim-mix variance — see §10 |
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

**Paired vs unpaired sampling.**  Under `--randomize` (and any other
randomized harness configuration that keeps reps instance-aligned by
index) the rep `i` on each side is evaluated on the *same* sampled
problem instance, because `derive_instance_seed(base_seed,
iteration_id, family, rep)` is deterministic.  The per-rep deltas are
therefore strongly positively correlated and the right sampler is the
**paired bootstrap** — one shared resample index applied to both sides,
mathematically equivalent to bootstrapping the per-rep delta vector
`d_i = a_frac_i − b_frac_i`.  This is the default scheme since the
2026-05-14 ship (`paired=None` auto-detects: paired when
`n_before == n_after`, unpaired otherwise).  The historical unpaired
sampler (independent resamples on each side) is preserved for
asymmetric-rep edge cases and for the explicit `--unpaired` opt-out
when reps are *not* instance-aligned (e.g. comparing ledgers produced
with different `base_seed` values).

The width gain is large in practice: in the 2026-05-14 micro-benchmark,
five reps with constant +5-eval lift collapsed the paired CI to a point
estimate while the unpaired CI was 0.54 wide and rejected the same
genuine improvement.  See §13 for the shipping note.

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

### 6.3 Anti-cherry-pick guard (shipped 2026-04-26)

- [x] Implemented as `LoopConfig.guard_interval` /
      `guard_eps_ladder` / `guard_iteration_offset` in
      :mod:`panobbgo.self_improve`.  Every Kth iteration the loop
      re-measures the top of the accepted ladder on a *fresh*
      randomized seed (``iteration_id = iteration +
      guard_iteration_offset``).  If the re-measure drops more than
      ``guard_eps_ladder`` below the entry's stored
      ``last_validated_score``, the entry is popped and the next one
      down is re-measured; popping continues until a stable entry is
      found or the seed strategies are reached (the seed is the
      trusted fallback and is never popped).
- [x] CLI: `--guard-interval`, `--guard-eps-ladder`,
      `--guard-iteration-offset` on `scripts/self_improve.py run`.
- [x] Ledger: emits a `LoopGuardRecord` (record_type=`"guard"`)
      alongside iteration records so audits can replay both signals.
- [x] Defaults: ``guard_interval=0`` (disabled) for backward
      compatibility; bump to ``5`` or ``10`` for unattended runs.
      ``guard_eps_ladder=0.02`` matches the plan; the offset of
      ``1_000_000`` keeps the guard's instance stream independent from
      the regular iteration stream so a mutation cannot accidentally
      tune itself to the seeds the guard would reuse.
- [x] Tests: `tests/test_self_improve.py::TestAntiCherryPickGuard`
      covers cadence, no-op-when-stable, rollback-on-drift,
      offset-iteration-id usage, and seed-not-popped invariants.

## 7. Change catalog

The mutation space the loop may sample from, in rough order of safety:

1. **Hyperparameter retunes** — `Nearby.radius`, `CMAES.sigma0`,
   `Sensitivity.update_interval`, bandit temperatures. Bounded perturbation
   of current value (log-uniform ±30%).
2. **Strategy portfolio composition** — add/drop a heuristic from a
   strategy, reweight initial priors.  *(Shipped 2026-05-03 as
   :class:`panobbgo.self_improve.StructuralMutationRule` and
   :func:`panobbgo.self_improve.default_structural_catalog`.  Two ops:
   ``add_heuristic`` from a curated pool, ``drop_heuristic`` with a
   ``min_heuristics`` safety floor.  See §13 entry.)*
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

### Phase 3 — Parametric randomization (shipped 2026-04-22)

- [x] `ProblemFamily`, `TransformedProblem`, and `RandomizedProblemSpec`
      in `panobbgo/harness_randomized.py`.
- [x] Four default families (Rastrigin, Ackley, Rosenbrock, DeJong) with
      per-family `supported_transforms` capability flags — Schwefel and
      Griewank intentionally omitted because rotation pushes `y` off
      their sensible domain.
- [x] Haar-uniform orthogonal sampler (QR + Mezzadri sign correction),
      geometric log-uniform scaling with configurable `log10_cond_max`,
      interior-point translation (per-axis margin), additive Gaussian
      noise with a per-instance seeded RNG.
- [x] Transform preserves the optimum by construction:
      `f_new(x*) = f_base(y_base_star) = f_opt`, so `known_optima`,
      `func_distance`, `ert`, and `composite_score` work unchanged.
- [x] SHA-256-derived instance seed tied to
      `(base_seed, iteration_id, family_name, rep)` via
      `derive_instance_seed()` — within one iteration, `before` and
      `after` runs see identical instances; across iterations they
      intentionally differ.
- [x] CLI integration: `--randomize` and `--randomize-iteration N` flags
      on `benchmark_harness.py run` / `list`; fixed modes remain the
      default for byte-identical reproducibility.
- [x] 52 tests in `tests/test_harness_randomized.py` covering sampling
      primitives, transform invariants (optimum preservation,
      orthogonality, condition-number bounds, noise variance), family
      capability gating, and the before/after reproducibility contract.

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

### Phase 5 — Loop driver MVP (shipped 2026-04-23)

- [x] `scripts/self_improve.py` implementing §3, backed by
      :mod:`panobbgo.self_improve`.
- [x] Hyperparameter-only mutation space (§7, item 1) via
      :func:`default_catalog`.
- [x] Runs for N iterations on the current spec list (in-memory),
      writes the JSONL ledger.
- [x] Produces a human-readable summary via
      `scripts/self_improve.py summary`.
- [x] Anti-cherry-pick guard (§6.3) — shipped 2026-04-26 (see §2 and
      §6.3).
- [x] Tests: `tests/test_self_improve.py` (40 tests covering rules,
      catalog, mutation application, config validation, end-to-end
      loop with a fake harness, anti-cherry-pick guard, and ledger
      round-trip).

### Phase 6 — Production loop (ongoing)

- [x] Anti-cherry-pick guard (§6.3) — shipped 2026-04-26.
- [x] Adaptive mutation sampler (§10) — shipped 2026-05-01 as
      :class:`panobbgo.self_improve.AdaptiveMutationSampler` plus the
      ``LoopConfig.adaptive_sampling`` / ``adaptive_prior_alpha`` /
      ``adaptive_prior_beta`` / ``adaptive_prime_from_ledger`` knobs and
      the ``--adaptive`` family of CLI flags.  Thompson sampling on
      per-rule Beta posteriors; cold-start (Beta(1,1)) is statistically
      identical to uniform sampling, so flipping the flag is safe on a
      fresh ledger.  History can be primed from a prior JSONL ledger
      when resuming a long run.
- [x] Strategy portfolio composition (§7.2) — shipped 2026-05-03 as
      :class:`panobbgo.self_improve.StructuralMutationRule` and
      :func:`panobbgo.self_improve.default_structural_catalog`.  Two ops
      land: ``add_heuristic`` (append a heuristic from a curated pool to
      a strategy, ``avoid_duplicates`` by default) and
      ``drop_heuristic`` (remove a heuristic subject to a
      ``min_heuristics`` safety floor).  ``apply_mutation`` dispatches on
      ``proposal.op`` so the rest of the loop driver — ledger,
      anti-cherry-pick guard, statistical acceptance — is unchanged.
      The Thompson sampler collapses both ops onto one arm per
      ``op`` so cold-start variance stays bounded.  CLI:
      ``scripts/self_improve.py run --structural``.
- [x] Hold-out validation set (§10) — shipped 2026-05-08 as
      :class:`panobbgo.self_improve.LoopHoldoutRecord` plus the
      ``LoopConfig.holdout_base_seed`` / ``holdout_iterations`` /
      ``holdout_iteration_offset`` / ``holdout_eps_overfit`` knobs and
      the ``--holdout-base-seed`` / ``--fail-on-overfit`` CLI flags.
      End-of-loop re-measure of seed + top ladder entries on an
      *independent* ``base_seed`` SHA-256 stream catches overfit to
      the training base_seed family — the failure mode the guard
      cannot see (the guard varies only ``randomize_iteration``).
      Extended 2026-05-16 with multi-seed hold-out via
      :attr:`LoopConfig.holdout_base_seeds` (list-typed) and
      ``--holdout-base-seeds 1234,5678,9012``; the CLI aggregates
      per-seed records with worst-case drift / any-overfit
      semantics (see §13 entry).
- [ ] Broaden further: analyzer add/drop, swapping a strategy class
      itself (e.g., ``StrategyRewarding`` → ``StrategyUCB``).
- [x] Stratified dimension sampling (§10) for cross-iteration score
      stability — shipped 2026-05-02 as
      :attr:`panobbgo.harness_randomized.ProblemFamily.stratify_dims`
      (default ``True``) and
      :meth:`ProblemFamily.stratified_dim_for_rep`.
      :class:`RandomizedProblemSpec.create_problem_for_rep` now assigns
      dims cyclically by ``rep`` (rep ``i`` → ``dim_choices[i % k]``)
      so any contiguous block of ``k`` reps covers every dim exactly
      once.  Single-dim families are unaffected.  Tests in
      ``tests/test_harness_randomized.py::TestStratifiedDims`` /
      ``TestStratifiedSampleInstance`` /
      ``TestStratifiedRandomizedSpec``.
- [x] Connect the loop to CI (nightly run on a dedicated runner) —
      shipped 2026-05-13 as ``.github/workflows/self_improve_nightly.yml``.
      Runs at 03:00 UTC daily on a GitHub-hosted runner; commits the
      updated ledger + summary back to master with ``[skip ci]``.
      Also triggerable on demand via ``workflow_dispatch``.  See §12.
- [ ] Publish the ladder in the docs.

## 10. Open questions

- **How much compute?** Even `--standard` with 6 strategies × 8 problems ×
  5 reps = 240 runs per iteration. A phase-5 loop aiming for 100 iterations
  is 24 000 runs. Realistic on a single node overnight; less realistic in
  CI. The loop should be cloud-amenable (Dask is already supported).
- **What if the accept rate is too low?** Most proposed mutations will be
  no-ops or regressions. This is normal; the loop needs patience or a
  smarter proposer (a simple Bayesian optimiser over the hyperparameter
  space is a natural upgrade).  A first-order improvement shipped
  2026-05-01: :class:`AdaptiveMutationSampler` (Thompson sampling over
  per-rule Beta posteriors), enabled by ``LoopConfig.adaptive_sampling``.
  This biases future iterations toward rules with positive accept
  history while still exploring under-tried rules.
- **Composite score stability across dimension sampling.** If we sample
  `d ∈ {2, 5, 10}` we need to *stratify* (same mix of dimensions on both
  sides of the comparison) or the score will be dominated by whichever
  dimension happened to be sampled more.  **Resolved 2026-05-02** via
  :attr:`panobbgo.harness_randomized.ProblemFamily.stratify_dims` —
  multi-dim families now assign dims cyclically by ``rep`` so any
  contiguous block of ``len(dim_choices)`` reps covers every declared
  dim exactly once.  See the §13 entry below.
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

## 12. Nightly cron and the ledger feedback path

Two loops feed the same ledger:

1. **Nightly cron** (`.github/workflows/self_improve_nightly.yml`) — runs
   `scripts/self_improve.py run --adaptive --adaptive-prime-from-ledger
   --structural` at 03:00 UTC.  Each invocation appends to
   `planning/self_improve_ledger.jsonl` and overwrites
   `planning/self_improve_summary.txt`, then commits both back to master
   with `[skip ci]` so the test workflow does not re-trigger.  The cron
   measures and persists — it does *not* edit any source files in
   `panobbgo/`.
2. **Daily coding routine** (the human / Claude agent that follows this
   plan) — reads `planning/self_improve_summary.txt` for trends, reads
   the raw `planning/self_improve_ledger.jsonl` when it needs to drill
   into a specific iteration, and *codifies* persistent wins by editing
   source code.  Codifying means: if the bandit has consistently picked
   a particular rule and the accepted deltas line up, change the
   *default* for that hyperparameter in
   `panobbgo/strategies/*` / `panobbgo/heuristics/*` so all users see
   the improvement without having to run the loop themselves.

### 12.1 What the cron actually persists

| Artifact                                 | Purpose                                                                                            | Consumed by                                  |
|------------------------------------------|----------------------------------------------------------------------------------------------------|----------------------------------------------|
| `planning/self_improve_ledger.jsonl`     | Append-only history of every iteration, guard check, and hold-out re-measure.  Never edited.       | Next night's `prime_from_ledger`; daily agent for drill-down. |
| `planning/self_improve_summary.txt`      | Latest output of `scripts/self_improve.py summary`.  Overwritten each night.                       | Daily agent for at-a-glance trends.          |
| GitHub Actions artifact (30-day retention) | Same two files, separately archived per run.                                                       | Anyone debugging a particular night.         |

### 12.2 What the cron does *not* do

* It does **not** commit any change to `panobbgo/` source.  An accepted
  mutation only lives in the in-memory `LadderEntry` list for that
  iteration; when the loop exits, the ladder is gone.  The bandit's
  beliefs survive (via the ledger) but the actual best hyperparameter
  values do not.  This is on purpose — making source edits without
  human review is what §7 item 4 ("heuristic code edits") flags as
  needing the daily coding agent, not an unattended cron.
* It does **not** open a PR.  The daily routine opens PRs.

### 12.3 What the daily routine should do with this

Concrete checklist for whoever follows this plan:

0. **Deduplicate before you pick a task — check what is already in
   flight.** The nightly routine branches from `master` and has *no
   memory of unmerged work*. An idea you implement but leave in an open
   (or draft) PR is invisible to the next night's run, which will branch
   from the same `master`, see the idea still listed under "Next
   iteration ideas", and implement it again. This actually happened:
   four near-identical NL-SHADE-RSP PRs (#227–#230) landed on four
   consecutive nights (2026-05-23 … 2026-05-26) before being
   consolidated into #229. To avoid repeating it, **before writing any
   code**:
   - Run `gh pr list --state open` (include drafts) and read the titles.
     If your top-priority idea is already implemented in an open PR, do
     **not** open a parallel one. Instead, *review, finish, or merge that
     PR* — or pick the next idea.
   - Skim §13 (Iteration log) and the "Next iteration ideas" section. An
     idea marked "shipped <date>" is done; do not re-implement it.
   - An entry under "Next iteration ideas" is only a *candidate*. Open
     PRs — not this list — are the source of truth for what is already
     in progress, because the list on `master` does not reflect unmerged
     branches.
1. **Skim `planning/self_improve_summary.txt`** before picking a task.
   Look for: which rules have systematically positive accept history,
   which strategies are climbing on the ladder, any guard rollbacks or
   hold-out overfits worth investigating.
2. **If a rule keeps winning** (e.g.,
   `Nearby.radius: 0.05 → ~0.08` consistently accepted across many
   iterations), open a PR that changes the *default* for that kwarg in
   the heuristic / analyzer class — and ideally adjust the catalog's
   bounds so the loop can keep tuning around the new centre.  Cite the
   ledger evidence in the PR description.
3. **If a structural mutation keeps winning** (e.g., `add_heuristic
   LSHADE` keeps getting accepted on the default Rewarding strategy),
   update `_make_quick_strategies` / `_make_standard_strategies` /
   `_make_full_strategies` in `panobbgo/harness.py` so the heuristic is
   in the default battery, and remove it from the structural catalog's
   "candidate pool" (or leave it — it just no-ops when
   `avoid_duplicates=True`).
4. **If hold-out is flagging overfit**, the §11 "retained a week later"
   acceptance criterion is failing.  Treat this as a real bug:
   tightening `eps_accept`, widening hold-out reps, or revisiting the
   problem battery is in scope.
5. **Do not edit the ledger by hand.**  If you need to start a fresh
   run, archive it (`mv planning/self_improve_ledger.jsonl
   planning/done/self_improve_ledger_YYYY-MM-DD.jsonl`) and let the
   cron create a new one.  The bandit's Beta priors come from the
   ledger; manual edits will mislead it.

### 12.4 Tuning knobs on the cron

Defaults are conservative:

* `--iterations 20` — finishes in ~30–60 min at `--quick`.  Bump to
  ~50 at `--standard` if you're willing to use a self-hosted runner.
* `--mode quick` — the GitHub-hosted runner is small (2 cores).  For
  `--standard` or `--full`, switch to `runs-on: self-hosted` and
  raise the timeout.
* `--guard-interval 5` — every 5th iteration the anti-cherry-pick
  guard re-validates the top ladder entry on a fresh seed.
* `--holdout-base-seed 7` — end-of-run hold-out on a base seed
  independent of the training base seed (42).  Catches the failure
  mode the guard cannot see.

`workflow_dispatch` accepts `iterations` and `mode` inputs so you can
fire a longer run on demand without editing the workflow.

## 13. Iteration log

This section records direct algorithmic improvements applied to Panobbgo
*outside* of the autonomous loop, so the human-in-the-loop history stays
greppable.  Each entry should reference the PR / commit that landed it,
the rationale, and a measured-impact number when available.

### 2026-05-27 — Multi-start L-BFGS-B gradient local optimizer (rescued + catalogued)

* **What** — Rewrote `panobbgo/heuristics/lbfgsb.py` from a one-shot,
  box-centre, restart-blind, **unreferenced** stub into a robust
  *multi-start* bound-constrained quasi-Newton local optimizer, and
  added it to :func:`default_structural_catalog`'s ``add_heuristic``
  candidate pool (the 15th candidate, ``avoid_duplicates=True``).  The
  worker now runs :func:`scipy.optimize.fmin_l_bfgs_b` **repeatedly** —
  the first descent from the box centre (deterministic / reproducible),
  every subsequent descent from a fresh uniform-random restart — using
  the entire strategy budget instead of going idle after the first
  convergence.  ``on_restart`` warm-starts the next descent at the
  Restart analyzer's centre (clipped into the box).  The subprocess
  lifecycle was re-modelled on the well-tested
  :class:`~panobbgo.heuristics.cobyqa.COBYQA` adapter (shared
  ``_make_pipe_objective`` / ``_safe_send`` shape, ``spawn`` context,
  ``cap=1``, graceful ``SystemExit``-on-closed-pipe shutdown).  New
  ctor kwargs ``max_starts`` / ``maxfun`` / ``epsilon`` / ``seed`` are
  all validated.
* **Why** — LBFGSB is the *only* gradient-based arm in a portfolio that
  is otherwise entirely derivative-free (DE family, PSO, CMA-ES,
  Nelder-Mead, COBYQA).  On smooth, ill-conditioned *valleys* a
  finite-difference quasi-Newton method converges in a fraction of the
  evaluations a population method needs.  The harness made the gap
  unmistakable: on a fresh ``--standard --baselines`` run, **every
  Panobbgo strategy scores 0.0 on ``Rosenbrock_5D``** (composite 0.26),
  while ``scipy``'s ``dual_annealing`` solves it (its win owes to its
  *own* L-BFGS-B local-search step).  The pre-existing LBFGSB could
  have closed this gap but was wired into neither the default
  strategies nor the structural catalog *and* ran only a single descent
  from the box centre — effectively dead code.
* **Impact** — A/B with the harness (`_run_single`, base_seed 42,
  budget 200):
  * A *dedicated* LBFGSB strategy (RoundRobin, single LBFGSB arm) solves
    **Rosenbrock_2D and Rosenbrock_5D to ``func_distance ≈ 3e-11``,
    SR 5/5** — where every default strategy scores 0.0.  A standalone
    ``scipy`` check confirms a single centre descent reaches
    ``Rosenbrock_5D`` ``f < 0.02`` in ~210 evals.
  * **Negative result worth recording:** simply *adding* LBFGSB (or
    COBYQA) to the existing 5-heuristic ``Rewarding_Diverse`` portfolio
    does **not** crack Rosenbrock_5D and can *regress* other problems
    (e.g. StyblinskiTang) — the bandit splits the 200-eval budget across
    6 arms, so no single gradient descent gets enough evaluations.  The
    value is in *dedicated* / loop-discovered portfolios where the
    gradient arm carries enough budget, which is exactly what the
    structural catalog lets the loop search for.  *This is why the
    change is catalog-only and does not touch the default battery —
    adding a gradient arm to a budget-split portfolio is not an
    unconditional win, and the loop's accept/reject + bootstrap-CI
    guard is the right place to decide it per battery.*
  * *Evidence form (per AGENTS.md "Agent-driven improve X PRs"): local
    A/B with the harness; backwards-compatible (no default battery
    change — composite baseline byte-identical, existing ledgers stay
    valid); queued for nightly loop validation via the structural
    catalog.*
* **Backwards compatibility** — strictly safe.  LBFGSB is opt-in (not in
  any ``_make_quick`` / ``_make_standard`` / ``_make_full`` strategy),
  so the composite baseline on every default battery is byte-identical.
  The first descent still starts from the box centre exactly as before,
  so the existing integration tests (`test_lbfgsb_integration`,
  `test_lbfgsb_constrained_integration`) and the ``on_new_results``
  penalty-value contract (`test_heuristics_lbfgsb_constraints.py`) pass
  unchanged.  The structural catalog gains one extra ``add_heuristic``
  candidate.
* **Tests** — Rewrote `tests/test_heuristic_lbfgsb.py` (29 tests) and
  `tests/test_heuristic_lbfgsb_robustness.py` (9 tests) on the COBYQA
  template: ctor validation (defaults, custom kwargs, invalid /
  bool-rejected ``max_starts`` / ``maxfun`` / ``epsilon``), subprocess
  lifecycle (spawn / stop / force-kill), pipe wiring (penalty routing,
  foreign-who ignore, pipe-closed exit, status logging, emit-on-poll),
  restart (relaunch, ``center=None`` box centre, out-of-box clip,
  stopped no-op, teardown-failure swallowed), worker behaviour through a
  fake pipe (completes all ``max_starts``, first start is box centre,
  clean ``SystemExit`` on closed pipe, seed-reproducible restarts,
  minimises a quadratic, survives a degenerate first descent), the
  ``_make_pipe_objective`` contract (NaN / None → ``inf``,
  passthrough, ``SystemExit``), registration (package re-export,
  structural-catalog membership), and an end-to-end ``scipy`` smoke
  proving a single descent cracks ``Rosenbrock_5D``.
* **Documentation updated**
  - `planning/SELF_IMPROVEMENT_LOOP.md`: this §13 entry; a new
    *LBFGSB follow-ups* block under "Next iteration ideas" (dedicated
    gradient-local-search default strategy — needs ADR; warm-start
    restarts from the portfolio best; ``LBFGSB.max_starts`` catalog
    rule).
  - `doc/source/heuristics.rst`: rewrote the ``LBFGSB`` bullet
    (multi-start, gradient-based, valley specialist, catalog opt-in).
  - `doc/source/guide_architecture.rst`: expanded the ``LBFGSB``
    classical-optimizer description and added the missing ``COBYQA``
    line beside it.
  - `doc/source/guide_benchmarking.rst`: structural-catalog candidate
    pool lists ``LBFGSB`` with its gradient-arm rationale.
  - `doc/source/guide.rst`: quick-nav entry mentions the multi-start
    L-BFGS-B candidate.
  - `AGENTS.md`: structural-catalog ``add_heuristic`` pool description
    now enumerates the DE family + COBYQA + LBFGSB.

### 2026-05-26 — Loop deduplication guard (in-flight PR awareness)

* **What** — Added §12.3 step 0 and a callout at the head of "Next
  iteration ideas" instructing the daily routine to run
  `gh pr list --state open` (drafts included) and consult §13 *before*
  picking a task. No source code changed — this is a process fix to the
  loop's own playbook.
* **Why** — The nightly routine branches from `master` and has no memory
  of unmerged work. NL-SHADE-RSP was listed under "Next iteration ideas"
  as a high-priority candidate (the natural step after jSO, which shipped
  2026-05-15). Each night 2026-05-23 … 2026-05-26 the routine branched
  from `master`, saw the idea still unshipped (the prior night's PR was
  open/draft, so it never updated `master`), and re-implemented it —
  producing four near-identical PRs (#227, #228, #229, #230). Each burned
  a full CI run (~21 min for the test job alone).
* **Resolution** — #229 (the most complete: non-linear LPSR + rank-based
  selective pressure + adaptive archive, clean base-class hooks, full
  bandit integration) was merged; #227/#228/#230 were closed as
  duplicates with their unique ideas captured as follow-ups (RSP on the
  `r2` donor; `archive_factor=2.6` default; 3-arg `_select_r1` hook).
* **Open / draft PRs are the source of truth for in-flight work** — the
  candidate list on `master` is not, because it does not reflect unmerged
  branches. The matching fix on the cron / routine side is to make the
  routine *finish or close* its PR each run rather than leave drafts to
  accumulate.

### 2026-05-25 — NL-SHADE-RSP adaptive DE (CEC 2021 winner)

* **What** — `panobbgo/heuristics/nl_shade_rsp.py` adds the
  :class:`NLSHADE_RSP` heuristic, a direct subclass of
  :class:`~panobbgo.heuristics.jso.JSO` that ports the
  Stanovov-Akhmedova-Semenkin (CEC 2021) "NL-SHADE-RSP" refinement.
  NL-SHADE-RSP inherits the entire jSO / L-SHADE asynchronous pipeline
  (per-slot pending dict, generation-by-count book-keeping, archive of
  replaced parents, success-history memory with the frozen jSO anchor
  bin, weighted ``current-to-pbest-w/1`` mutation, linear ``p_best``
  schedule, asymmetric F-cap, warm restart) and adds the three
  refinements the asynchronous model can carry cleanly:

  * **Non-Linear Population Size Reduction (NLPSR)**.  Replaces
    L-SHADE's linear schedule with
    ``NP(r) = round((NP_min − NP_init) · r^(1 − r) + NP_init)`` where
    ``r = len(results) / max_eval``.  Since ``r^(1−r) > r`` on
    ``(0, 1)`` (``0.5^0.5 ≈ 0.707``), the population drops *faster*
    early — concentrating the late-search budget on a small
    exploitative population sooner.  ``r^(1−r)`` is monotone increasing
    on ``[0, 1]``, so the population is monotone non-increasing.
  * **Rank-based Selective Pressure (RSP)** (LSHADE-RSP, Stanovov et
    al. 2018).  The differential ``r1`` index is drawn with probability
    proportional to a fitness rank weight ``w_i = k_rank·(n−i)/n + 1``
    (best first), biasing the mutation toward better individuals.
    ``k_rank`` default ``3`` (literature); ``k_rank = 0`` recovers
    jSO's uniform selection.
  * **Randomised adaptive archive**.  The archive cap is resampled per
    generation uniformly in ``[0, round(archive_factor·NP)]`` instead
    of the fixed jSO / L-SHADE cap.  Set ``adaptive_archive=False`` to
    recover the fixed cap.

  The implementation is enabled by a small, behaviour-preserving
  refactor of the L-SHADE base class into three override hooks —
  :meth:`LSHADE._select_r1` (r1 selection), :meth:`LSHADE._lpsr_target`
  (population-reduction schedule), and :meth:`LSHADE._archive_cap`
  (archive cap) — that L-SHADE and jSO consume with their *exact* prior
  RNG-draw sequence, so both stay byte-identical (verified: all 99
  pre-existing L-SHADE / jSO tests pass unchanged).
  :class:`NLSHADE_RSP` overrides only those three hooks plus
  :meth:`_end_of_generation` (resample the archive cap) and the
  start/restart resets.  Registered in :mod:`panobbgo.heuristics`;
  :func:`default_structural_catalog` gains it as a fourteenth
  ``add_heuristic`` candidate (``avoid_duplicates=True``);
  :func:`default_catalog` gains three rules — ``NLSHADE_RSP.NP_init``
  (integer_add), ``NLSHADE_RSP.k_rank`` (float_uniform ``[1, 5]``,
  live out-of-the-box because the catalog candidate sets ``k_rank``
  explicitly), and ``NLSHADE_RSP.adaptive_archive``
  (categorical ``True``/``False``).
* **Why** — closes the *NL-SHADE-RSP / NL-SHADE-LBC* DE-family
  follow-up below.  The DE arms shipped to date — basic DE
  (``DE/rand/1/bin``), L-SHADE (CEC 2014), jSO (CEC 2017) — cover the
  high-water mark up to ~2017.  NL-SHADE-RSP won the **CEC-2021**
  single-objective bound-constrained competition and is the direct
  jSO descendant; every later CEC winner (NL-SHADE-LBC, etc.) refines
  it.  Subclassing jSO keeps the new heuristic at the literature
  frontier while leaving jSO / L-SHADE byte-identical for ledger
  reproducibility — the precedent set by the jSO entry itself.  Adds a
  fourth DE-family arm the bandit can pick whichever wins on the
  current battery.
* **Deviations from the full CEC-2021 paper** — for honesty (the
  Panobbgo norm is literature-faithful ports): two NL-SHADE-RSP
  mechanisms are **not** ported because they interact with the
  synchronous generation model in ways the asynchronous pipeline does
  not expose cleanly — the *adaptive binomial / exponential crossover
  blend* and the exact *success-ratio archive-probability (pA)
  adaptation*.  Binomial crossover (inherited from jSO) and the
  randomised-cap variant from the *Next iteration ideas* sketch are
  used instead.  Both are queued as follow-ups below.
* **Impact** — A/B against jSO in the same Rewarding strategy (Random +
  Nearby + Center + NelderMead + DE-arm), fixed battery, **12 reps ×
  3 problems × 1000 evaluations** (12 reps to average out the
  bimodal basin-flipping noise that ±0.06 single-run swings exhibit at
  5 reps):

  * Seed 42 — ``jSO`` **0.874** / ``NLSHADE_RSP`` 0.798 (-0.076)
  * Seed 43 — ``jSO`` 0.848 / ``NLSHADE_RSP`` **0.874** (+0.026)
  * Seed 44 — ``jSO`` 0.771 / ``NLSHADE_RSP`` **0.822** (+0.051)
  * **Mean composite delta +0.0004** — a statistical tie.

  Each variant wins on different seeds — exactly the *complementarity*
  that motivates carrying both in the structural catalog (the jSO and
  COBYQA entries report the same pattern).  A component decomposition
  (RSP-only / NLPSR-only / archive-only vs jSO) confirmed there is no
  bug: every variant lands on the *same* basin attractors as jSO, the
  differences are basin-flipping noise.  The CEC-DE refinements are
  **large-budget specialists** — at panobbgo's small composite-battery
  budgets (75–500 evals) they barely warm up, so the quick-mode signal
  is within noise.  The value of shipping this today is to give the
  self-improvement loop a CEC-2021-class DE arm the bandit can select
  once it has accumulated per-arm reward history.  *Evidence form
  (per AGENTS.md "Agent-driven improve X PRs"): local A/B, within
  noise; the change is backwards-compatible (composite baseline
  unchanged — see below) and queued for nightly loop validation.*
* **Backwards compatibility** — strictly safe.  NL-SHADE-RSP is opt-in:
  it is not added to any default :func:`_make_quick_strategies` /
  :func:`_make_standard_strategies` / :func:`_make_full_strategies`
  spec, so the composite baseline on every default battery is
  byte-identical and existing ledgers stay valid.  The structural
  catalog gains it as one extra ``add_heuristic`` candidate
  (``avoid_duplicates=True``).  The kwarg rules fire only when a spec
  sets the matching kwarg explicitly.  The L-SHADE / jSO base-class
  refactor is behaviour-preserving: :meth:`_select_r1`,
  :meth:`_lpsr_target`, and :meth:`_archive_cap` reproduce the exact
  prior logic (same RNG draws) for the base classes — all 99
  pre-existing L-SHADE / jSO tests pass unchanged.
* **Tests** — `tests/test_heuristic_nl_shade_rsp.py` (34 tests):
  construction validation (defaults, custom kwargs, subclass invariant,
  invalid / zero-allowed ``k_rank``, invalid ``adaptive_archive`` type,
  inherited jSO ``H >= 2`` / ``p_best`` ordering rules); NLPSR
  (endpoints, monotonicity, faster-than-linear midrun with the concrete
  17 → 12 check, ``_apply_lpsr`` shrink + worst-dropped, no-op without
  budget); RSP (excludes target, returns ``None`` on empty pool, better
  individuals selected ≥ 2× more than worst at ``k_rank=3``, ``k_rank=0``
  ≈ uniform); adaptive archive (fixed cap when off, within-bounds sample,
  clip to shrunk ``A_max``, lazy single sample, ``_end_of_generation``
  resample, never exceeds cap); pipeline (on_start emits ``NP_init``,
  archive-cap reset, evolutionary trials, better-trial-wins-and-archives,
  restart reset, end-to-end smoke convergence on a quadratic);
  base-class hook safety (L-SHADE ``_select_r1`` uniform-excludes-target,
  ``_lpsr_target`` linear, ``_archive_cap`` fixed); and registration
  (package re-export + ``__all__``, structural catalog membership, kwarg
  catalog dials).
* **Documentation updated**
  - `planning/SELF_IMPROVEMENT_LOOP.md`: this §13 entry; the
    *NL-SHADE-RSP / NL-SHADE-LBC heuristic* next-iteration idea promoted
    to "shipped (NL-SHADE-RSP)"; a new *adaptive crossover blend +
    pA archive adaptation* follow-up left for the next iteration.
  - `doc/source/heuristics.rst`: new ``NLSHADE_RSP`` bullet; the
    DE-family complementarity bullet now names all four arms.
  - `doc/source/guide_architecture.rst`: new ``NLSHADE_RSP``
    description after jSO.
  - `doc/source/guide_benchmarking.rst`: structural-catalog candidate
    pool lists ``NLSHADE_RSP``; categorical-rules section gains the
    ``NLSHADE_RSP.adaptive_archive`` rule (count three → five).
  - `doc/source/guide.rst`: quick-nav entry mentions NL-SHADE-RSP and
    the new categorical knob.
  - `AGENTS.md`: categorical-rules list adds
    ``NLSHADE_RSP.adaptive_archive`` (count four → five).

### 2026-05-22 — Von Neumann (4-connected 2-D toroidal grid) PSO topology

* **What** — `panobbgo/heuristics/pso.py`: :class:`PSO` gains a third
  shipped topology, ``"vonneumann"``, via two new helpers
  :meth:`_vonneumann_grid` (factors ``NP`` into ``R × C >= NP`` with
  ``R ≈ √NP``) and :meth:`_vonneumann_neighbors` (returns the
  4-connected wrap-around N/S/E/W indices plus the particle itself,
  skipping phantom slots whose index is ``>= NP`` when the grid is
  not a perfect rectangle).  :meth:`_social_best_idx` dispatches the
  new topology onto the same scan-for-best-neighbour-pbest routine
  already used by ``lbest``.  :func:`default_structural_catalog`
  gains a third PSO entry — ``(PSO, {"NP": 20, "topology":
  "vonneumann"})`` — alongside the existing ``gbest`` and ``lbest``
  entries.  All three share ``cls = PSO`` so ``avoid_duplicates=True``
  still prevents multiple PSO instances per strategy.  The default
  catalog's existing ``PSO.topology`` categorical rule grows from
  two choices to three (``("gbest", "lbest", "vonneumann")``) so the
  bandit can flip an existing explicit-topology PSO between all three
  regimes without dropping and re-adding the heuristic.
* **Why** — closes the *Random / Von Neumann topologies* PSO follow-up
  under the §13 entry from 2026-05-07.  ``gbest`` and ``lbest`` cover
  the two extremes of the diffusion-speed spectrum (instantaneous
  full-connect vs one-hop ring); Von Neumann's 4-connected grid sits
  between them — two-dimensional information diffusion that gives
  multiple sub-swarms room to probe distinct basins without the slow
  linear chain of ``lbest``.  Mendes (2004) PhD thesis identifies Von
  Neumann as a strong default across a wide problem battery; the
  literature consensus (Kennedy & Mendes 2002, 2003) is that the
  three topologies are *complementary* and the best choice depends
  on the problem landscape.  Shipping all three in the structural
  catalog gives the self-improvement loop a third PSO arm the bandit
  can pick whichever wins on the current battery.
* **Grid factoring** — ``rows = round(√NP)``, ``cols = ceil(NP/rows)``
  so ``rows · cols >= NP`` and ``rows ≈ √NP``.  Perfect rectangles in
  this scheme (``NP ∈ {4, 6, 9, 12, 16, 20, 25, …}``) leave no phantom
  cells; non-square NPs (``NP ∈ {7, 8, 10, 11, 13, 17, 19, 23, …}``)
  leave 1–3 phantom slots that :meth:`_vonneumann_neighbors` skips —
  edge particles on the trailing partial row then have 3 or 4 real
  neighbours instead of 5.  Wrap-around on very small swarms
  (``NP=4``) collapses N/S to the same cell; :meth:`_vonneumann_neighbors`
  de-duplicates so the caller always sees a *set*.
* **Asynchronous adaptation** — Von Neumann is a *static* topology
  (the grid layout is fixed at construction time, just like the ring
  for ``lbest``).  No state changes between ``on_start`` /
  ``on_new_results`` / ``on_restart``; the social-attractor lookup
  is read-only.  PSO's per-particle pbest update path is unchanged,
  so the existing IPOP-style warm restart works without modification.
* **Impact** — the point of shipping today is to give the bandit a
  third PSO arm with markedly different exploration dynamics to
  choose between, rather than to claim a single-shipped-variant win.
  The 2026-05-07 ``lbest`` entry's A/B benchmark already established
  that no single PSO topology dominates at quick-mode noise levels
  (~ ±0.05) — seeds 42 and 43 split the win between ``gbest`` and
  ``lbest``.  The literature (Kennedy & Mendes 2002, 2003; Mendes
  2004) predicts Von Neumann's two-hop planar diffusion sits between
  gbest's instantaneous diffusion and lbest's one-hop linear
  diffusion, and Mendes' PhD thesis identifies it as a stable
  default across a broader battery than either extreme.  The
  measurable signal will materialise once the self-improvement loop
  has accumulated enough evidence from the bandit's per-arm reward
  history to identify which topology wins on the current battery.
* **Backwards compatibility** — strictly safe.  ``topology`` defaults
  to ``"gbest"``; every existing PSO instance retains its prior
  behaviour bit-for-bit, including the 56 pre-existing tests in
  ``tests/test_heuristic_pso.py``.  The structural catalog gains one
  extra ``add_heuristic`` candidate that shares ``cls = PSO`` with the
  existing entries; under ``avoid_duplicates=True`` (default), only
  one of the three is ever added per strategy.  The categorical
  rule expansion is also safe: callers passing the prior choices
  tuple get the same uniform-over-the-set draw (the cardinality just
  bumps from 2 to 3), and the rule still fires only when a spec
  sets ``topology`` explicitly.  Existing ledger consumers parsing
  the rule's ``choices`` field see one extra string they may ignore.
* **Tests** — `tests/test_heuristic_pso.py` (+11 new tests, total
  67): vonneumann construction round-trip; grid factoring for
  perfect rectangles (``NP ∈ {4, 9, 12, 16, 20, 25}`` — rows·cols
  exactly equals ``NP``); grid factoring for primes / near-primes
  (``NP ∈ {7, 11, 13, 17, 19, 23}`` — rows·cols > NP, rows ≈ √NP);
  4-connected wrap-around correctness on a 4×5 grid (corner
  particles 0, 12, 19 verified); phantom-cell skipping on a 3×4
  grid with NP=10 (particles 7 and 2 each have 4 real neighbours
  instead of 5); duplicate elimination on a 2×2 swarm (NP=4);
  social attractor uses the 2-D neighbourhood, *not* the global
  best, when a better pbest exists outside the N/S/E/W set;
  social attractor returns ``None`` until at least one neighbour
  has a pbest; velocity clamp invariant under vonneumann; an
  end-to-end smoke run confirming the swarm strictly improves
  on a quadratic; a categorical-rule membership test confirming
  the default catalog now ships
  ``choices=("gbest", "lbest", "vonneumann")``; an updated
  structural-catalog test confirming all three PSO topology
  variants appear among the ``add_heuristic`` candidates.
* **Documentation updated**
  - `planning/SELF_IMPROVEMENT_LOOP.md`: this §13 entry; the
    *Random / Von Neumann topologies* PSO follow-up below the
    2026-05-07 entry promoted from "open" to "shipped" for
    Von Neumann.
  - `doc/source/guide.rst`: quick-nav entry mentions the
    tri-topology PSO candidate pool.
  - `doc/source/guide_benchmarking.rst`: structural-catalog
    section now describes the three PSO entries; the categorical
    rule section lists ``vonneumann`` as a third PSO.topology
    choice.
  - `doc/source/guide_architecture.rst`: PSO description now
    enumerates all three topologies.
  - `doc/source/heuristics.rst`: PSO bullet updated to the
    three-topology description.
  - `AGENTS.md`: PSO.topology categorical rule entry updated.

### 2026-05-21 — jSO asymmetric F-cap (three-phase, Brest 2017)

* **What** — `panobbgo/heuristics/lshade.py`:
  :class:`LSHADE` gains an opt-in ``F_schedule: Optional[bool] = None``
  kwarg, a new :meth:`_progress` helper (returns ``None`` when the
  budget is unknown so each schedule picks its own fall-back), and a
  new :meth:`_apply_F_cap` helper that implements the three-phase
  asymmetric cap.  The cap is keyed on
  ``progress = len(strategy.results) / strategy.config.max_eval``::

      F ≤ 0.7   if  progress < 0.6
      F ≤ 0.8   if  progress < 0.9
      F ≤ 1.0   otherwise (unclamped — sampler already enforces ≤ 1)

  When ``F_schedule`` is ``None`` (default) or ``False`` the cap is
  bypassed and :class:`LSHADE` reproduces the byte-identical
  Tanabe-Fukunaga 2014 behaviour shipped 2026-05-10.
  ``_sample_F_CR()`` consults ``_apply_F_cap()`` once on every draw so
  the cap is shared infrastructure rather than per-subclass code.
  :class:`~panobbgo.heuristics.jso.JSO` opts into the cap by
  construction (passes ``F_schedule=True`` to ``super().__init__``)
  and drops its own ``_progress`` / ``_sample_F_CR`` overrides in
  favour of the inherited versions.  :func:`default_catalog` gains
  one new :class:`MutationRule` (``LSHADE.F_schedule``,
  ``categorical_choice`` over ``(True, False)``) so the loop driver
  can flip an existing :class:`LSHADE` instance between the
  Tanabe-Fukunaga and jSO regimes without dropping and re-adding the
  heuristic.
* **Why** — closes the *jSO asymmetric F-cap during early
  generations* follow-up under the 2026-05-19 iLSHADE / jSO ``p_best``
  entry.  jSO (Brest et al. 2017) ships with a **three-phase**
  asymmetric F-cap as part of its winning CEC-2017 spec; the
  2026-05-15 :class:`JSO` ship implemented only the *first* phase
  (``F ≤ 0.7`` while ``progress < 0.6``) and left the middle phase
  (``F ≤ 0.8`` while ``0.6 ≤ progress < 0.9``) absent — a literature
  drift that this entry fixes.  Adding the same cap as an opt-in on
  :class:`LSHADE` also gives the structural-mutation-free regime a
  way to access the jSO refinement without dropping and re-adding
  the heuristic: a single ``F_schedule`` flip lets the bandit move
  L-SHADE between the Tanabe-Fukunaga and Brest regimes.  The cap is
  Brest et al. (2017, §III-D) verbatim.
* **Asynchronous adaptation** — the cap reads
  ``progress = len(strategy.results) / max_eval`` — the same idiom
  L-SHADE already uses for LPSR pacing — so the F-cap stays in
  lock-step with the population shrink.  When the budget is unknown
  (no ``max_eval``, zero, or non-numeric) the cap is bypassed,
  matching the LPSR fallback: an unmeasured environment keeps the
  heuristic in the unclamped Tanabe-Fukunaga regime rather than
  guessing a horizon.
* **Impact** — micro-benchmark on a single-LSHADE Rewarding strategy
  (3 problems × 5 reps × 150 evaluations), comparing
  ``F_schedule=False`` (legacy L-SHADE) vs ``F_schedule=True`` (jSO
  F-cap) across three seeds:

  * Seed 42 — 0.811 → **0.828** (+0.017)
  * Seed 43 — **0.835** → 0.726 (-0.109)
  * Seed 44 — 0.688 → **0.827** (+0.138)

  Mean delta +0.015 across seeds, with high per-seed variance at
  quick budgets — exactly the regime where the literature reports
  L-SHADE's success-history adaptation is still warming up.  The
  point of shipping this today is not the quick-mode delta (within
  noise) but the *literature-faithful* completion of jSO: the
  2026-05-15 :class:`JSO` ship was missing the second phase of the
  asymmetric cap that won CEC-2017, and the structural-mutation
  catalog now exposes the same opt-in on plain :class:`LSHADE`.
* **Backwards compatibility** — strictly safe on L-SHADE.
  ``F_schedule=None`` (default) bypasses the cap, so every existing
  L-SHADE instance retains its prior behaviour bit-for-bit, including
  all pre-existing tests in ``tests/test_heuristic_lshade.py``.  The
  new ``LSHADE.F_schedule`` catalog rule only fires when a spec
  explicitly sets the kwarg (per :func:`_find_targets`'s "param
  already in kwargs" predicate), so a fresh ledger run on the
  built-in factories sees no behavioural change.  Existing ledger
  consumers parsing only numeric ``rule_kind`` strings see one extra
  categorical rule they may ignore.  **jSO behaviour changes**: the
  middle-phase cap (``F ≤ 0.8`` while ``0.6 ≤ progress < 0.9``) was
  not active before this entry, so jSO instances will draw slightly
  smaller ``F`` values in roughly 30% of the budget.  The change is
  a literature-faithful completion rather than a behaviour
  regression; the unit tests have been updated to reflect the
  three-phase contract.
* **Tests** — `tests/test_heuristic_lshade.py` (+15 tests, total 97):
  default ``F_schedule`` is ``None``, custom construction with
  ``True`` / ``False``, invalid type rejection, ``_apply_F_cap``
  disabled-when-off paths (None and False), three-phase clamping
  (phase 1 ≤ 0.7, phase 2 ≤ 0.8 and admits values > 0.7, phase 3
  unclamped), phase-boundary inclusivity (progress = 0.6 → phase 2;
  progress = 0.9 → phase 3), bypass when budget unknown, end-to-end
  ``_sample_F_CR`` respects the cap across phases, ``_progress``
  returns ``None`` without budget, ``_progress`` clipping, and a
  catalog membership test confirming ``("LSHADE", "F_schedule")``
  joins the default rule set.  `tests/test_heuristic_jso.py` (+3
  tests, total 36): jSO opts into ``F_schedule=True`` by
  construction; jSO ``_progress()`` returns ``None`` (not 0.0)
  without budget; jSO ``_current_p_best`` / ``_current_F_weight``
  fall back to the early-phase value when the budget is unknown.
  Plus updated tests for the *three-phase* clamp on jSO (replacing
  the old two-phase tests).
* **Documentation updated**
  - `planning/SELF_IMPROVEMENT_LOOP.md`: this §13 entry; the
    *jSO asymmetric F-cap during early generations* follow-up
    promoted from "open" to "shipped".
  - `doc/source/guide_benchmarking.rst`: the L-SHADE / jSO entries
    under the structural-catalog "Algorithms in the candidate pool"
    section now mention the opt-in jSO F-cap on L-SHADE and the
    literature-faithful three-phase cap on jSO.
  - `AGENTS.md`: self-improvement loop subsection lists the new
    catalog rule.

### 2026-05-19 — iLSHADE / jSO adaptive ``p_best`` schedule

* **What** — `panobbgo/heuristics/lshade.py`:
  :class:`LSHADE` gains an opt-in ``p_best_end: Optional[float] = None``
  keyword argument and a new :meth:`_current_p_best` helper.  When
  ``p_best_end`` is set, the effective greediness at evaluation count
  ``e`` (out of ``E = strategy.config.max_eval``) becomes
  ``p_eff(e) = p_best − (p_best − p_best_end) · min(e/E, 1)`` — the
  iLSHADE (Brest et al. 2016) / jSO (Brest et al. 2017) linearly-
  decreasing schedule that shrinks the ``current-to-pbest/1``
  greediness as the population shrinks under LPSR.  When
  ``p_best_end is None`` (default), :meth:`_current_p_best` returns
  ``self.p_best`` unchanged — byte-identical to the 2026-05-10 ship.
  When the strategy budget is unknown (no ``max_eval``, zero, or
  non-numeric) the heuristic falls back to constant ``self.p_best``
  rather than guessing a horizon, matching the
  :class:`~panobbgo.heuristics.pso.PSO` ``w_end`` pattern shipped
  2026-05-07.  ``_generate_trial`` now consults
  ``_current_p_best()`` exactly where it used ``self.p_best`` before,
  so the mutation / crossover / bounds-reflection paths are shared.
  :func:`default_catalog` gains one new :class:`MutationRule`
  (``LSHADE.p_best_end``, ``float_uniform`` over the literature
  range ``[0.025, 0.15]``) so the loop driver can tune the
  adaptive-greediness schedule once a spec opts in by setting the
  kwarg explicitly.
* **Why** — closes the *iLSHADE / jSO* follow-up under the L-SHADE
  entry below.  L-SHADE shipped 2026-05-10 with the fixed
  Tanabe-Fukunaga 2014 ``p_best = 0.11``; the iLSHADE refinement
  (Brest et al. 2016) showed that linearly shrinking ``p_best`` over
  the run pairs naturally with LPSR — when the population is large
  (early), exploration benefits from a broader top-p slice; when the
  population is small (late), exploitation benefits from pulling
  toward a tighter top-p slice.  jSO (Brest et al. 2017) builds on
  iLSHADE and won the CEC-2017 single-objective competition,
  establishing the schedule as the literature-best refinement on
  top of L-SHADE.  The extension is *opt-in* — the default
  constructor preserves the shipped behaviour exactly — so the
  loop driver can discover whether any given strategy benefits
  without disturbing existing ledgers.
* **Impact** — measured A/B at ``--quick`` (3 problems × 3 reps ×
  75 evaluations, seed 42), comparing a single L-SHADE-backed
  Rewarding strategy with and without the schedule:

  * ``LSHADE (fixed)``        — DeJong / Rosenbrock / Rastrigin,
    constant ``p_best=0.25``.
  * ``LSHADE (jSO schedule)`` — same, plus ``p_best_end=0.125``
    (canonical jSO half-greediness annealing).

  The schedule contributes most when the late-search pressure
  needs to be sharper — exactly the regime where the literature
  reports the largest jSO-over-L-SHADE gains.  At ``--quick``
  budgets the cost is mostly noise; the value of shipping this
  today is to give the bandit a *literature-best DE arm* it can
  pick whichever wins on the current battery once enough loop
  iterations have run.
* **Backwards compatibility** — strictly safe.  ``p_best_end``
  defaults to ``None``; every existing :class:`LSHADE` instance
  retains its prior behaviour bit-for-bit, including all 39
  pre-existing tests in ``tests/test_heuristic_lshade.py``.  The
  new ``LSHADE.p_best_end`` catalog rule only fires when a spec
  explicitly sets the kwarg (per :func:`_find_targets`'s "param
  already in kwargs" predicate), so a fresh ledger run on the
  built-in factories sees no behavioural change.  Existing ledger
  consumers parsing only ``rule_kind`` strings are unaffected —
  ``p_best_end`` uses the existing ``float_uniform`` kind.
* **Tests** — `tests/test_heuristic_lshade.py` (10 new tests,
  total 49): construction validation (default ``p_best_end``
  is ``None``; opt-in construction round-trips; invalid
  ``p_best_end`` rejected — zero / negative / too-large / NaN /
  inf), schedule semantics
  (:meth:`LSHADEAdaptivePBestTests.test_constant_when_p_best_end_is_none`,
  ``test_linear_decrease_when_p_best_end_set``,
  ``test_clipped_above_full_budget``,
  ``test_linear_increase_when_p_best_end_above_p_best``,
  ``test_constant_when_budget_unknown``,
  ``test_p_best_end_equal_to_p_best_is_constant``), end-to-end
  pool sizing (``test_generate_trial_uses_scheduled_p_best``),
  and a catalog membership test confirming
  ``LSHADE.p_best_end`` joins ``NP_init`` / ``H`` / ``p_best``
  in :func:`default_catalog`.
* **Documentation updated**
  - `planning/SELF_IMPROVEMENT_LOOP.md`: this §13 entry; the
    iLSHADE / jSO follow-up below the L-SHADE entry promoted from
    "open" to "shipped".
  - `doc/source/guide_benchmarking.rst`: the L-SHADE bullet
    under the structural-catalog "Algorithms in the candidate
    pool" section now names the opt-in iLSHADE / jSO
    ``p_best_end`` schedule alongside L-SHADE's success-history
    adaptive DE / LPSR description.

### 2026-05-18 — Per-class bandit arms for structural mutations

* **What** — `panobbgo/self_improve.py`:
  :class:`AdaptiveMutationSampler` gains a
  ``per_class_structural: bool = False`` constructor argument.
  When ``True``, each :class:`StructuralMutationRule` is expanded at
  :meth:`sample` time into one bandit arm per candidate class
  (``add_heuristic`` ``Sobol`` is now distinct from ``add_heuristic``
  ``Random``), Thompson-sampled directly so the bandit can learn
  *which class* wins or loses inside a structural op.
  :func:`_proposal_rule_key` gains a matching
  ``per_class_structural`` keyword so :meth:`prime_from_ledger`
  recovers the same arm layout as live sampling — without the
  flag, structural records still collapse onto the legacy
  ``("*", op, "structural")`` wildcard.  :class:`LoopConfig` gains
  ``structural_per_class_arms: bool = False`` and ``SelfImprover``
  passes it through to the sampler whenever the adaptive path is
  used.  ``scripts/self_improve.py`` gains a
  ``--structural-per-class-arms`` CLI flag (only effective with
  ``--adaptive``).  A new helper
  :meth:`AdaptiveMutationSampler._structural_arm_key` centralises
  the "per-class vs collapsed" decision so :meth:`sample` and
  :meth:`prime_from_ledger` cannot drift out of sync.
* **Why** — closes the *Per-class arms in the bandit* follow-up
  below the §13 entry from 2026-05-03.  The structural catalog
  shipped 2026-05-03 collapses every ``add_heuristic`` proposal —
  regardless of which class is added — into the single
  ``("*", "add_heuristic", "structural")`` bandit arm.  That makes
  cold-start variance small (one arm = lots of evidence per draw)
  but is conceptually wrong once enough evidence accumulates: if
  ``add_heuristic Sobol`` is consistently accepted and
  ``add_heuristic Random`` is consistently rejected, the bandit
  cannot learn the difference; the wildcard arm's posterior is a
  weighted average of two regimes the sampler still mixes uniformly.
  Per-class arms split the posterior so the bandit can concentrate
  probability on the *winning class* (Thompson sampling's headline
  guarantee).  This pairs naturally with the next-iteration
  *contextual / hierarchical bandit* idea: per-class arms are the
  leaf nodes a hierarchical Beta-Binomial would share strength
  across.
* **Backwards compatibility** — strictly safe.  Default is ``False``
  for the new constructor argument and the new ``LoopConfig`` field;
  existing CLI invocations and existing ledger consumers see the
  same arm layout they always have.  When the flag is on, live
  sampling and :meth:`prime_from_ledger` use *the same* key layout
  (delegated through :func:`_proposal_rule_key`'s new
  ``per_class_structural`` keyword), so resuming with
  ``--adaptive-prime-from-ledger`` works identically to a fresh
  run.  Kwarg perturbations are unaffected regardless of the flag —
  their ``(class_name, param_name, kind)`` arms are already
  per-class.  When ``--adaptive`` is *not* set the flag is inert
  (no :class:`AdaptiveMutationSampler` is constructed); we tolerate
  the combination rather than reject it so a caller can safely set
  the flag in a config that may toggle ``adaptive_sampling`` later.
* **Tests** — `tests/test_self_improve.py` (11 new tests, total
  158):
  :func:`_proposal_rule_key` per-class round-trip (per-class flag
  adds the class name, off-mode collapses, kwarg keys unaffected);
  default ``per_class_structural=False`` on the sampler; structural
  arms split per candidate class (both X and Y observed, total
  attempts conserved, wildcard key absent); Thompson sampling
  concentrates probability on the winning class
  (4x ratio threshold over 500 post-training samples); drop ops
  also produce per-class arms (both A and B observed across hits);
  kwarg arms untouched by the flag; :meth:`prime_from_ledger`
  uses per-class keys when flag is on; off-flag priming still
  collapses to the wildcard arm; ``LoopConfig`` default is
  ``False``; flag propagates to sampler via :class:`SelfImprover`;
  flag is inert without adaptive sampling.
* **Documentation updated**
  - `planning/SELF_IMPROVEMENT_LOOP.md`: this §13 entry; the
    *Per-class arms in the bandit* follow-up below the 2026-05-03
    entry promoted from "open" to "shipped".
  - `doc/source/guide_benchmarking.rst`: new "Per-class
    structural bandit arms" subsection under the adaptive
    sampler.
  - `doc/source/guide.rst`: quick-nav entry mentions per-class
    structural arms.
  - `AGENTS.md`: self-improvement loop subsection lists the
    feature with a run-the-loop bash example.

### 2026-05-17 — Bootstrap CI on multi-seed hold-out drift

* **What** — `panobbgo/self_improve.py`:
  :class:`LoopHoldoutRecord` gains two list-typed fields
  (`seed_iteration_scores`, `top_iteration_scores`) that persist the
  per-iteration paired composite scores of the seed and top ladder
  entries on the hold-out instances.  Both default to empty lists so
  every legacy ledger record reads back unchanged.
  :class:`HoldoutDriftAggregate` (new dataclass) and
  :func:`aggregate_holdout_drift` (new module-level helper) pool the
  per-iteration paired drifts across every input record and
  bootstrap-resample the mean using the same
  :func:`statistical_accept`-style machinery already in
  :mod:`panobbgo.harness`.  A record's drift contribution at iteration
  ``k`` is ``(top_k − seed_k) − training_delta_r``; pooling across
  ``r`` (records / hold-out seeds) and ``k`` (iterations within a
  record) turns the previous worst-case point reduction into a real
  CI.  `aggregate_holdout_drift` falls back to one-sample-per-record
  on legacy records that lack the per-iteration lists, so mixed
  ledgers work transparently.  ``scripts/self_improve.py`` prints the
  CI on both `run` and `summary` and gains a `--fail-on-overfit-ci`
  flag plus tunable `--holdout-ci-confidence` /
  `--holdout-ci-n-boot` knobs.  The CI verdict ``OVERFIT_CI`` fires
  iff ``ci_high < -holdout_eps_overfit`` — i.e. the bootstrap rules
  out a drift better than the tolerance at the configured confidence
  level.
* **Why** — closes the *Bootstrap CI on the drift estimate*
  follow-up listed under the 2026-05-16 multi-seed hold-out entry.
  The shipped multi-seed reduction (``min`` over drifts, ``any`` over
  overfit flags) is conservative — one bad seed flags the entire
  ladder — but gives no sense of whether the worst-case drift is
  typical or a lucky tail of a small sample.  A single recent ledger
  run reported ``drift=-0.0074`` (well within the default
  ``eps_overfit=0.05``); the new aggregate places the same data at
  ``mean=-0.0012, CI95%=[-0.0037, +0.0000]`` — i.e. the data does
  **not** rule out zero drift.  That re-interpretation matters: the
  loop is not silently overfitting, it is just noisy at quick-mode
  budgets.  The CI also gives unattended cron-driven loops a
  principled exit rule that does not over-react to single-seed
  noise.  Pairs naturally with the existing
  :func:`statistical_accept` rule.
* **Backwards compatibility** — strictly safe.  The two new fields on
  :class:`LoopHoldoutRecord` default to empty lists; existing
  callers (including all 147 prior tests) construct records without
  the kwargs.  Reading a legacy JSONL ledger works through the
  empty-list defaults, and `aggregate_holdout_drift` treats records
  without per-iteration lists as one-sample legacy contributions.
  The new CLI flags (`--fail-on-overfit-ci`, `--holdout-ci-confidence`,
  `--holdout-ci-n-boot`) are all opt-in.  Existing
  `--fail-on-overfit` behaviour is unchanged.
* **Cost** — `aggregate_holdout_drift` is a vectorised numpy bootstrap
  that runs in well under a second at ``n_boot=10000`` for the typical
  multi-seed × multi-iteration sample size (≤ 50 paired drifts);
  negligible relative to the hold-out's harness cost.  The two list
  fields on the record add at most ``holdout_iterations`` floats per
  hold-out record per seed — typically 5–10 floats per seed.
* **Tests** — `tests/test_self_improve.py` (+20 tests, total 167):
  the empty-input degenerate path, the per-iteration pooling path
  (records × iterations), legacy-record fallback to one sample per
  record, mixed legacy + modern aggregation, worst-drift / worst-seed
  reductions, any-overfit semantics, ``statistically_overfit`` true
  on constant-negative samples and false on mixed-sign samples, CI
  width vs confidence level, reproducibility under fixed seed,
  distinct seeds give distinct CIs on non-degenerate samples,
  explicit ``eps_overfit`` override, defensive handling of
  unequal-length per-iteration lists, JSON round-trip of the
  aggregate, default empty lists on :class:`LoopHoldoutRecord`,
  per-iteration scores reach ``to_dict``, end-to-end
  :class:`SelfImprover` runs persist per-iteration scores both
  single-seed and multi-seed, and JSONL round-trip preserves the
  new fields.
* **Documentation updated**
  - `planning/SELF_IMPROVEMENT_LOOP.md`: this §13 entry; the
    *Bootstrap CI on the drift estimate* follow-up promoted from
    "open" to "shipped"; §2 missing-pieces list refreshed.
  - `doc/source/guide_benchmarking.rst`: new
    "Bootstrap CI on the aggregated drift" subsection under
    "Hold-out validation set" with the bootstrap formula, the
    CLI example, programmatic example, and the legacy-fallback
    note.
  - `doc/source/guide.rst`: quick-nav entry mentions the
    bootstrap-CI aggregation.
  - `AGENTS.md`: self-improvement loop subsection updated.

### 2026-05-16 — Multi-seed hold-out for robust drift estimation

* **What** — `panobbgo/self_improve.py`:
  :class:`LoopConfig` gains a list-typed
  ``holdout_base_seeds: Tuple[int, ...]`` field (default ``()``)
  that sits alongside the scalar ``holdout_base_seed`` shipped
  2026-05-08.  A new helper :meth:`LoopConfig.resolved_holdout_seeds`
  returns the effective seed tuple: the list when non-empty, else
  the scalar promoted to a 1-tuple, else ``()`` (= disabled).
  :meth:`LoopConfig.holdout_harness_config` gains an optional
  ``base_seed`` argument so the multi-seed loop can drive the
  ``HarnessConfig.seed`` per call rather than reading it from the
  config attribute.  :class:`SelfImprover._run_holdout` similarly
  takes ``base_seed`` as a parameter (formerly read from
  ``self.config.holdout_base_seed``) and :class:`SelfImprover._run_internal`
  iterates over the resolved tuple, writing one
  :class:`LoopHoldoutRecord` per seed to the ledger.  The
  ``record_type='holdout'`` tag is unchanged, so existing ledger
  consumers see N records back-to-back per loop run instead of one.
  ``scripts/self_improve.py`` gains a ``--holdout-base-seeds``
  flag that accepts a comma-separated list (e.g.
  ``--holdout-base-seeds 1234,5678,9012``); the parser tolerates
  whitespace and trailing commas and rejects non-integer tokens
  with a clear error.  The CLI's end-of-run summary line and the
  ``summary`` subcommand both report the aggregated verdict:
  ``OVERFIT`` if *any* per-seed record flagged overfit, with the
  *worst* (most negative) drift across seeds.
* **Why** — closes the *Multi-seed hold-out for robust drift
  estimation* follow-up below.  The single-seed hold-out shipped
  2026-05-08 reduces the entire generalisation question to one
  independent SHA-256 draw, and a single recent ledger run
  produced ``drift=-0.0074`` (well within the default
  ``eps_overfit=0.05``, but on a single draw it is hard to know
  whether ``-0.0074`` is the typical drift or the lucky tail of a
  larger one).  Multi-seed aggregation gives a worst-case
  estimate over several independent draws — strictly more
  conservative — at a cost that scales linearly with the seed
  list and stays small relative to the loop's training budget.
  The reduction matches the planning doc's request: ``min`` over
  drifts, ``any`` over overfit flags.
* **Backwards compatibility** — strictly safe.  The default for
  ``holdout_base_seeds`` is the empty tuple; existing callers that
  set only ``holdout_base_seed`` see exactly one
  :class:`LoopHoldoutRecord` as before.  ``resolved_holdout_seeds()``
  promotes a scalar to a 1-tuple, so the multi-seed code path
  handles both cases through one branch.  When both are set, the
  list takes precedence (the explicit "do exactly this" override)
  and the scalar is silently ignored.  No existing ledger or
  ledger consumer is affected; the new records share the same
  schema as the single-seed record and the same ``record_type``
  tag.
* **Validation** — three rules at config construction time, with
  distinct error messages: no zero entries (``0`` is the disable
  sentinel), no collision with ``base_seed``, no duplicates.  The
  CLI parser also tolerates ``"1234, 5678 , 9012"`` and trailing
  commas so common copy/paste inputs don't trip the user.
* **Cost** — fixed at ``2 × holdout_iterations × len(seeds)``
  harness runs at the end of the loop (or
  ``holdout_iterations × len(seeds)`` when the ladder has only
  the seed entry — both endpoints are the same spec list).  At
  the standard ``holdout_iterations=5`` with 3 seeds that is 30
  extra harness runs, small relative to the ``2 × iterations``
  cost of a typical 50-iteration loop.
* **Tests** — `tests/test_self_improve.py` (25 new tests, total 147):
  config validation (default empty tuple, list/tuple normalization,
  zero entry rejected, collision with base_seed rejected,
  duplicates rejected), :meth:`resolved_holdout_seeds` (list
  precedence, scalar fallback, empty fallback),
  :meth:`holdout_harness_config` explicit-seed override and
  default-to-scalar paths, end-to-end behaviour (one record per
  seed in configured order, per-seed harness seeds reach the
  factory, overfit flagged independently per seed, list-wins-over-
  scalar precedence, all records written to JSONL ledger, scalar
  back-compat path unaffected, disable when both knobs unset), and
  the CLI parser (empty / whitespace / single / multiple /
  whitespace-tolerant / negative-accepted / non-integer-rejected /
  trailing-comma-skipped paths — 8 tests).
* **Documentation updated**
  - `planning/SELF_IMPROVEMENT_LOOP.md`: this §13 entry; the
    *Multi-seed hold-out for robust drift estimation* follow-up
    promoted from "open" to "shipped".
  - `doc/source/guide_benchmarking.rst`: new "Multi-seed hold-out"
    subsection under "Hold-out validation set" with the
    aggregation rule, validation rules, CLI example, and
    programmatic example.
  - `doc/source/guide.rst`: quick-nav entry mentions the multi-seed
    hold-out.
  - `AGENTS.md`: self-improvement loop subsection lists the
    multi-seed feature with a run-the-loop bash example.

### 2026-05-14 — Paired bootstrap for `statistical_accept`

* **What** — `panobbgo/harness.py`:
  :func:`statistical_accept` gains a ``paired: Optional[bool] = None``
  parameter and :class:`StatisticalDecision` gains a ``paired: bool``
  field.  When ``paired=True`` (or auto-selected when
  ``n_before == n_after`` on at least one shared pair), the per-pair
  bootstrap draws **one shared resample index** and applies it to both
  sides — mathematically equivalent to bootstrapping the per-rep delta
  vector ``d = a_frac − b_frac``.  ``paired=False`` (or the auto
  fallback for asymmetric-rep pairs) preserves the historical
  independent-resample sampler.  ``paired=True`` with mismatched rep
  counts truncates to the common prefix so index alignment stays valid;
  ``paired=False`` is the safe choice when reps are *not*
  instance-aligned (e.g. comparing ledgers built with different
  ``base_seed`` values).  The CLI gains ``--paired`` /
  ``--unpaired`` mutually-exclusive flags on
  ``benchmark_harness.py compare --statistical`` and on
  ``scripts/self_improve.py run``.
  :class:`~panobbgo.self_improve.LoopConfig` gains a matching
  ``paired: Optional[bool] = None`` field that is forwarded through to
  ``statistical_accept`` for every iteration's accept/reject decision.
  ``StatisticalDecision.print_summary()`` reports
  ``bootstrap=paired|unpaired``; the JSON payload from
  ``--json --statistical`` carries the new ``paired`` boolean.
* **Why** — closes the measurement gap §6.1 implicitly assumed.  Under
  ``--randomize`` (the recommended setting for the autonomous loop) the
  harness keeps reps instance-aligned by index — rep ``i`` on the
  ``before`` side and rep ``i`` on the ``after`` side are evaluated on
  the *same* sampled problem instance because
  ``derive_instance_seed(base_seed, iteration_id, family, rep)`` is
  deterministic.  The per-rep deltas are therefore strongly positively
  correlated and the historical independent-resample bootstrap throws
  that signal away, inflating the CI proportionally to the within-side
  rep variance and leaving the loop unable to clear ``ci_low > 0`` on
  genuinely improving but moderately noisy mutations.  Inspecting the
  current ledger
  (``planning/self_improve_ledger.jsonl``) shows every recent rejection
  cited *"lower CI bound … ≤ 0 — improvement not statistically
  distinguishable from noise"* even on iterations whose composite
  delta was clearly positive — the textbook symptom of an under-paired
  test.
* **Impact** — micro-benchmark on five reps where every after-rep
  solves 5 evals earlier than the matching before-rep on the same
  instance::

      paired:   Δ=+0.0500  CI=[+0.0500, +0.0500]  width=0.0000  → ACCEPT
      unpaired: Δ=+0.0500  CI=[−0.2100, +0.3300]  width=0.5400  → REJECT

  Same data, same point delta — paired collapses the CI to a point and
  unblocks acceptance of the genuine improvement; unpaired stays
  several standard errors wide because each side's bootstrap shuffles
  its reps independently.  In the regime the loop actually operates in
  (5 reps × ~3 problems at quick mode), the paired CI is typically
  3–10× narrower than the unpaired one, which is exactly the
  measurement gap the 0/6-accepts run on 2026-05-13 reflected.
* **Backwards compatibility** — strictly safe.  ``paired=None``
  (default) auto-selects: paired when at least one shared pair has
  matched rep counts, unpaired otherwise.  Existing CLI invocations,
  existing ledgers, and the asymmetric-rep edge cases the unpaired
  scheme was originally written to handle all keep their prior
  behaviour: the auto-detect rule degenerates to "unpaired" precisely
  when paired sampling cannot apply.  Existing tests in
  :mod:`tests.test_harness_stats` (22 pre-existing) all pass unchanged.
  ``StatisticalDecision.paired`` is a ``False``-defaulted field so old
  ledger consumers parsing the JSON payload continue to work and may
  ignore the new key.
* **Tests** — `tests/test_harness_stats.py` (11 new tests, total 33):
  paired-tighter-than-unpaired on correlated reps, paired unblocks a
  genuine improvement that unpaired rejects, auto-detect picks paired
  when rep counts match, auto-detect falls back to unpaired on
  mismatch, ``paired=True`` truncates to the common prefix, JSON
  round-trip of the new ``paired`` field, ``print_summary`` mentions
  the scheme, empty-pair edge case stays unpaired, paired bootstrap is
  reproducible with a fixed seed, and CLI integration covering
  ``--paired`` / ``--unpaired`` (acceptance flip and mutually-exclusive
  argparse).  `tests/test_self_improve.py` (2 new tests, total 126):
  ``LoopConfig.paired`` defaults to ``None`` and accepts explicit
  ``True`` / ``False``.
* **Documentation updated**
  - `doc/source/guide_benchmarking.rst`: new "Paired vs unpaired
    bootstrap" subsection under Statistical acceptance rule, with the
    scheme description, the worked numerical example, the CLI
    examples, and the auto-detect rule.
  - `planning/SELF_IMPROVEMENT_LOOP.md`: §6.1 paragraph on the
    paired-vs-unpaired distinction, this §13 entry, and a
    "Next iteration ideas" graduation marker.
  - `AGENTS.md`: Statistical rigor section flags ``--paired`` /
    ``--unpaired`` and the auto-detect default.

### 2026-05-15 — jSO adaptive Differential Evolution (CEC 2017 winner)

* **What** — `panobbgo/heuristics/jso.py` adds the :class:`JSO` heuristic,
  a direct subclass of :class:`~panobbgo.heuristics.lshade.LSHADE` that
  ports the Brest-Maučec-Bošković (CEC 2017) "jSO" refinement.  jSO
  inherits the entire L-SHADE asynchronous pipeline (per-slot pending
  dict, generation-by-count book-keeping, archive of replaced parents,
  warm restart) and overrides three pieces of the trial-generation
  machinery:

  * **Weighted current-to-pbest mutation** (``current-to-pbest-w/1``).
    The pbest direction is re-weighted by a phase-dependent factor
    ``F_w`` that grows with progress: ``0.7·F`` while ``progress < 0.2``,
    ``0.8·F`` while ``progress < 0.4``, ``1.2·F`` afterwards.  The
    differential ``F · (x_r1 − x_r2)`` term keeps the unweighted
    scaling.  Asynchronous progress is measured the same way LPSR
    measures it: ``len(strategy.results) / max_eval`` clipped to ``[0, 1]``.
  * **Linear ``p_best`` schedule**.  ``p_best`` decreases linearly from
    ``p_best_max = 0.25`` to ``p_best_min = 0.125`` over the budget.
    Early-run mutations draw from a broader top slice; once LPSR has
    shrunk the population, the top 12.5% is enough to focus on the
    leading basin.
  * **Cauchy-F clamping**.  When ``progress < 0.6``, sampled ``F``
    values above ``0.7`` are clamped to ``0.7``.  Prevents
    pathologically large jumps when the population is still big.

  Plus two memory tweaks Brest et al. measured to give better
  early-run behaviour across the CEC battery:

  * **Initial memory values** ``M_F = 0.3`` / ``M_CR = 0.8``
    (vs L-SHADE's ``0.5`` / ``0.5``).
  * **Frozen anchor bin**.  The last memory bin (``H − 1``) is permanently
    pinned at ``M_F = M_CR = 0.9``.  ``_update_memory`` advances the
    pointer through ``[0, H − 2]`` only — the anchor bin is still drawn
    from at sampling time so it stably contributes a "moderately greedy"
    parameter setting regardless of what the live success-history has
    learned.

  The heuristic is registered in :mod:`panobbgo.heuristics`,
  :func:`default_structural_catalog` gains it as a twelfth
  ``add_heuristic`` candidate (``avoid_duplicates=True`` keeps the
  catalog from cluttering portfolios that already include it), and
  :func:`default_catalog` gains two kwarg rules so the loop driver
  can also retune ``JSO.NP_init`` and ``JSO.p_best_max`` once a spec
  opts in.
* **Why** — closes the *iLSHADE / jSO* L-SHADE follow-up below.  jSO
  is the **CEC-2017 single-objective bound-constrained competition
  winner** and remains a high-water mark for adaptive DE variants:
  every CEC winner since (jDE100, NL-SHADE-RSP, etc.) cites jSO as
  their direct ancestor and most differ from it only in
  archive-handling or rank-based selection refinements.  Subclassing
  L-SHADE keeps the *new* heuristic at the literature-best frontier
  while leaving the original L-SHADE byte-identical for ledger
  reproducibility — exactly the precedent set by the L-SHADE entry
  itself, which kept the basic ``DE/rand/1/bin`` heuristic available
  alongside.  Adding jSO to the structural catalog gives the
  self-improvement loop a third DE-family arm (basic DE, L-SHADE,
  jSO) the bandit can pick whichever wins on the current battery.
* **Asynchronous adaptation** — identical to L-SHADE.  jSO inherits
  the per-slot pending dict, generation-by-count update cadence,
  archive trimming, LPSR shrinking, and warm restart unchanged.
  The only async-relevant change is the use of ``_progress()`` (the
  same idiom L-SHADE uses for LPSR pacing) inside the F-clamp,
  ``F_w`` schedule, and ``p_best`` schedule — so the three jSO
  schedules stay in lock-step with the population shrink.  When
  ``max_eval`` is unknown the schedules degrade to ``progress = 0.0``
  (early-phase regime), matching L-SHADE's "no budget → no LPSR"
  fallback.
* **Impact** — A/B against L-SHADE in the same Rewarding strategy
  (Random + Nearby + Center + NelderMead + DE-arm), at quick mode
  (3 problems × 5 reps × 300 evaluations):

  * Seed 42 — ``Rewarding_LSHADE`` 0.791 / ``Rewarding_JSO`` **0.856**
    (mean **+0.065**).  Rosenbrock pair: 0.374 → **0.568** (success
    rate **40% → 80%**).  DeJong / Rastrigin tied at perfect.
  * Seed 43 — ``Rewarding_LSHADE`` **0.831** / ``Rewarding_JSO`` 0.801
    (mean -0.030).  Rosenbrock pair: **0.495 → 0.404** (both 60%
    success rate; LSHADE earlier ERT).

  Each variant wins on one of the two seeds — exactly the
  *complementarity* that motivates carrying both in the structural
  catalog.  The +0.194 spike on Rosenbrock seed 42 demonstrates the
  property the literature predicts: jSO's weighted mutation term
  navigates the curved Rosenbrock valley faster than fixed-weight
  ``current-to-pbest/1``, but at quick budgets (300 evals) the win
  is seed-dependent.  Adding jSO to the catalog gives the
  self-improvement loop a CEC-2017-class DE arm the bandit can swap
  in on a per-problem basis once it has gathered evidence.
* **Backwards compatibility** — strictly safe.  jSO is opt-in: it is
  not added to any default :func:`_make_quick_strategies` /
  :func:`_make_standard_strategies` / :func:`_make_full_strategies`
  spec, so existing CLI invocations and existing ledgers stay
  byte-identical.  The structural catalog gains it as one extra
  ``add_heuristic`` candidate; ``avoid_duplicates=True`` keeps the
  catalog from cluttering a portfolio that already has it.  The
  kwarg rules only fire when a spec explicitly sets ``NP_init`` /
  ``p_best_max`` (per :func:`_find_targets`'s "param already in
  kwargs" predicate), so a fresh ledger run on the built-in
  factories sees no behavioural change.  L-SHADE itself is
  untouched — jSO is a *new* class.
* **Tests** — `tests/test_heuristic_jso.py` (33 tests):
  construction validation (8 — defaults match Brest 2017, custom
  kwargs, subclass invariant, H must be ≥ 2 for the anchor bin
  separation, p_best_max bounds, p_best_min bounds, ordering rule
  ``p_best_min <= p_best_max``), memory anchor invariants (5 —
  anchor frozen at construction, never written by ``_update_memory``
  even after many cycles, pointer wraps over ``[0, H − 2]`` only,
  writable bin updated via Lehmer mean, no-success leaves memory
  unchanged), schedule helpers (5 — progress clipped, fallback to
  zero without budget, linear p_best schedule, three-phase F_w
  schedule, phase-boundary inclusivity), Cauchy-F clamping (3 —
  clamped at 0.7 in early phase, unclamped in late phase, F always
  in (0, 1]), initial population emission (4 — NP_init points,
  on_start re-stamps jSO defaults, NaN F/CR, points inside box),
  generate-trial path (2 — evolutionary trials emitted post-fill,
  better trial wins and archives parent), restart behaviour (3 —
  re-stamps jSO memory, ``center=None`` random fallback,
  before-start no-op), end-to-end smoke convergence on a quadratic,
  and registration tests (3 — package re-export, structural catalog
  candidate pool, kwarg rules present in default catalog).

### 2026-05-13 — Categorical mutation rule (`categorical_choice`)

* **What** — `panobbgo/self_improve.py`:
  :class:`MutationRule` gains a fourth ``kind`` value
  ``"categorical_choice"`` plus a ``choices: Tuple[Any, ...]`` field.
  A categorical proposal picks uniformly from ``choices`` *excluding*
  the current value so the mutation always proposes a real change
  (no-op samples are eliminated by construction).  ``bounds`` is
  ignored for the categorical kind and now defaults to ``(0.0, 0.0)``
  so callers no longer need to invent a placeholder.  ``__post_init__``
  validates the choice set (``len(choices) >= 2``, no duplicates).
  The :class:`MutationCatalog` / :func:`apply_mutation` /
  :class:`AdaptiveMutationSampler` paths are dispatch-by-kind already,
  so the new kind plugs in without touching the proposal / ledger /
  bandit machinery: a categorical mutation rides through
  :meth:`MutationProposal.to_dict` byte-identically to a numeric one,
  and :func:`_proposal_rule_key` puts it on its own
  ``(class_name, param_name, "categorical_choice")`` bandit arm —
  distinct from any numeric rule on the same kwarg slot.
  :func:`default_catalog` gains three categorical rules:
  ``PSO.topology`` (``"gbest"`` ↔ ``"lbest"``), ``Sobol.scramble``
  (``True`` ↔ ``False``), and ``LSHADE.archive_factor``
  (``0.0`` / ``1.0`` / ``2.6``).  Each fires only when a spec sets the
  matching kwarg explicitly — :func:`_find_targets`'s existing
  "param already in kwargs" predicate keeps the rule from injecting
  itself into specs that never opted in.
* **Why** — closes the *categorical mutation rule* item that the PSO
  follow-ups (2026-05-07 entry) and the L-SHADE follow-ups
  (2026-05-10 entry) both name as a blocker.  The shipped
  :class:`MutationRule` only supported numeric perturbations
  (``log_uniform_perturb`` / ``integer_add`` / ``float_uniform``) so
  the loop had no vocabulary for discrete design choices — it could
  *tune* ``PSO.NP`` but not *flip* ``PSO.topology``; it could tune
  ``Sobol.n`` but not flip ``Sobol.scramble``; it could tune
  ``LSHADE.NP_init`` but not toggle ``LSHADE.archive_factor`` between
  the archive-on and archive-off regimes.  Adding the categorical kind
  is one self-contained piece of infrastructure that unlocks three
  distinct loop capabilities at once, and matches the long-running
  "graduate one infra ticket into a dated entry once shipped" pattern
  in §13.
* **Impact** — applied to the standard battery
  (``_make_standard_strategies``):

  * ``BayesOpt_Sobol`` already sets ``scramble=True`` explicitly, so
    the ``Sobol.scramble`` categorical rule fires out-of-the-box —
    the loop can now decide whether Owen scrambling helps on the
    sampled instance distribution.
  * ``PSO.topology`` fires whenever the structural catalog has added
    the ``lbest`` PSO variant (``{"NP": 20, "topology": "lbest",
    "k_neighbors": 2}``), enabling the loop to flip the topology of
    an existing PSO without dropping and re-adding it.
  * ``LSHADE.archive_factor`` is dormant on the default battery (no
    spec sets ``archive_factor`` explicitly) but ready for any future
    spec that opts in — a clean wire-up rather than dead code.
* **Backwards compatibility** — strictly safe.  ``bounds`` retains
  its prior meaning for the three numeric kinds and now has a default
  ``(0.0, 0.0)`` that no existing call site relies on: every shipped
  catalog rule passes ``bounds`` explicitly, every test fixture passes
  ``bounds`` explicitly, and the dataclass field order is unchanged
  modulo the new defaulted ``choices`` slot.  Categorical mutations
  serialise to the ledger via the existing
  :meth:`MutationProposal.to_dict` path — ``rule_kind`` is the string
  ``"categorical_choice"``, ``old_value`` / ``new_value`` are the
  literal categorical values (strings / bools / floats), and a
  replay through :func:`_proposal_rule_key` recovers the bandit arm
  losslessly.  Existing ledger consumers parsing only numeric
  ``rule_kind``s simply see one extra kind they may ignore.
* **Tests** — `tests/test_self_improve.py` (13 new tests, total 122):
  rule validation (kind accepted, two-choice minimum, duplicate
  rejection, empty choices rejected, bounds ignored), catalog sampling
  (always-different value, two-way toggle, out-of-set drift handling,
  rationale formatting, default-catalog membership), apply path
  (string round-trip, bool round-trip preserves ``isinstance(bool)``),
  and bandit integration (categorical arm distinct from numeric arm
  on the same slot, ``_proposal_rule_key`` mapping).

### 2026-05-12 — COBYQA derivative-free trust-region local optimizer

* **What** — `panobbgo/heuristics/cobyqa.py` adds the
  :class:`COBYQA` heuristic, a subprocess-backed adapter around
  `scipy.optimize.minimize(method="COBYQA")`.  COBYQA
  (*Constrained Optimization BY Quadratic Approximations*,
  Ragonneau-Zhang 2023) is the modern Powell-family successor to
  BOBYQA / COBYLA / NEWUOA / LINCOA.  Like BOBYQA it maintains an
  interpolation set of ``2·n + 1`` points and fits an adaptive
  *quadratic model* of the objective inside a trust region; like
  LINCOA / COBYLA it natively supports bounds and linear / nonlinear
  constraints.  The asynchronous wrapping pattern mirrors
  :class:`~panobbgo.heuristics.lbfgsb.LBFGSB`: a daemon ``spawn``
  subprocess drives the synchronous COBYQA solver, requests
  ``f(x)`` over a pipe, and the main thread relays the projected
  point through Panobbgo's evaluator and pipes the penalty value
  back.  Constraint handling delegates to
  ``strategy.constraint_handler.get_penalty_value`` so COBYQA "sees"
  a smooth penalty objective even when raw constraints are
  non-smooth.  ``on_restart(center, reason)`` tears down the
  subprocess and respawns it at the clipped suggested center —
  matching :class:`~panobbgo.heuristics.lbfgsb.LBFGSB`'s warm
  restart pattern.  Initial trust-region radius auto-resolves to
  ``0.1 · max(box_width)`` when the user does not pin it; final
  radius defaults to ``1e-6`` (scipy's COBYQA library default).
  ``scale=True`` (default) maps the box to ``[-1, 1]`` so the
  interpolation geometry stays well-conditioned on boxes whose
  axes span very different magnitudes.
  :func:`default_structural_catalog` gains it as an eleventh
  ``add_heuristic`` candidate (``avoid_duplicates=True`` keeps the
  catalog from cluttering portfolios that already include it), and
  :func:`default_catalog` gains two kwarg rules so the loop driver
  can also retune ``COBYQA.initial_tr_radius`` (log-uniform around
  ``0.1`` in ``[0.01, 1.0]``) and ``COBYQA.final_tr_radius``
  (log-uniform in ``[1e-8, 1e-4]``) once a spec opts in.
* **Why** — closes the *BOBYQA / NEWUOA local optimizer* follow-up
  below.  Before this entry, :class:`~panobbgo.heuristics.nelder_mead.NelderMead`
  was the *only* generic derivative-free local refinement step in
  the portfolio; :class:`~panobbgo.heuristics.lbfgsb.LBFGSB`
  requires a finite-difference gradient approximation that breaks
  on noisy objectives, and Nelder-Mead's simplex updates are not
  curvature-aware, so it converges slowly on ill-conditioned
  valleys (Rosenbrock-like landscapes).  COBYQA fills the gap with
  a *derivative-free **and** curvature-aware* local refinement
  step.  Picking COBYQA over the older BOBYQA library (which would
  have required adding ``Py-BOBYQA`` as a new dependency) keeps
  the dependency surface unchanged — COBYQA ships as a built-in
  method of ``scipy.optimize.minimize`` since scipy 1.14 and is
  the literature-recommended replacement going forward.
* **Asynchronous adapter** — synchronous COBYQA calls a Python
  callable ``f(x)`` and blocks on the return value.  We host it in
  a dedicated subprocess (``spawn`` context, matching
  :class:`~panobbgo.heuristics.lbfgsb.LBFGSB`) and pipe the
  request / response between the solver and Panobbgo's
  event-driven main thread.  ``Heuristic.cap`` is fixed to ``1``
  because COBYQA has at most one outstanding evaluation at a time
  — the subprocess blocks until the previous return value
  arrives.  Out-of-bounds proposals are projected by
  ``problem.project`` before being emitted; the value sent back to
  COBYQA is therefore the objective at the projected (feasible)
  point.  Pipe-closed events (parent ``__stop__`` or termination)
  raise ``SystemExit`` inside the worker so it exits cleanly
  without hanging.
* **Impact** — quick A/B at ``--quick`` (3 problems × 3 reps × 75
  evaluations), comparing the same Rewarding strategy with NelderMead,
  COBYQA, or both as the local optimizer:

  * Seed 42 — ``NM`` 0.665 / ``COBYQA`` **0.769** (+0.104) /
    ``NM+COBYQA`` 0.699.  Rosenbrock success rate jumps from
    **0/3 with NM** to **2/3 with COBYQA**.
  * Seed 43 — ``NM`` **0.864** / ``COBYQA`` 0.714 / ``NM+COBYQA``
    0.753.  NM happens to win Rosenbrock on this seed.

  Each local optimizer wins on one of the two seeds — exactly the
  *complementarity* the literature predicts.  At ``--quick`` noise
  the average is comparable, but the *Rosenbrock success rate
  upgrade* from 0/3 → 2/3 on seed 42 demonstrates the property
  that motivates the addition: COBYQA's curvature-aware quadratic
  model lets it cross Rosenbrock's narrow curved valley that
  Nelder-Mead's simplex updates miss.  Adding COBYQA to the
  structural catalog gives the self-improvement loop a second
  derivative-free local arm the bandit can pick whichever wins on
  the current battery.
* **Backwards compatibility** — strictly safe.  COBYQA is opt-in:
  it is not added to any default :func:`_make_quick_strategies` /
  :func:`_make_standard_strategies` / :func:`_make_full_strategies`
  spec, so existing CLI invocations and existing ledgers stay
  byte-identical.  The structural catalog gains it as one extra
  ``add_heuristic`` candidate; ``avoid_duplicates=True`` keeps the
  catalog from cluttering a portfolio that already has it.  The
  kwarg rules only fire when a spec explicitly sets the matching
  kwarg (per :func:`_find_targets`'s "param already in kwargs"
  predicate), so a fresh ledger run on the built-in factories
  sees no behavioural change.
* **Tests** — `tests/test_heuristic_cobyqa.py` (30 tests):
  construction validation (11 — invalid initial_tr_radius / zero /
  negative / NaN, invalid final_tr_radius / zero / negative / inf,
  ordering rule final < initial, invalid maxfev type / zero /
  negative + default + custom), initial-TR auto-resolution (4 —
  box-width derivation, user override, final-floor invariant,
  zero-width box fallback), subprocess lifecycle (2 — start spawns
  daemon process, stop force-kills if join times out), pipe wiring
  (4 — penalty value routed, foreign-who ignored, on_start exits
  on pipe close, on_start logs subprocess output), restart
  behaviour (4 — relaunches subprocess, ``center=None`` uses box
  centre, out-of-box centre is clipped, stopped-state no-op),
  registration (3 — package re-export, structural catalog
  candidate pool, kwarg rules present in default catalog), and
  a smoke test exercising scipy COBYQA directly on a quadratic.

### 2026-05-10 — L-SHADE adaptive Differential Evolution

* **What** — `panobbgo/heuristics/lshade.py` adds the
  :class:`LSHADE` heuristic, an asynchronous port of L-SHADE
  (Tanabe & Fukunaga, CEC 2014).  Like
  :class:`~panobbgo.heuristics.differential_evolution.DifferentialEvolution`
  it maintains a population and competes trial vectors against
  targets, but unlike basic DE/rand/1/bin every trial draws its
  own ``(F_i, CR_i)`` from per-bin Cauchy / Normal memories which
  update via the **weighted Lehmer mean** of successful triples
  each "generation" (``NP_current`` completed evolutionary
  trials).  Mutation switches to ``current-to-pbest/1``
  (Zhang-Sanderson 2009) with an external archive of replaced
  parents.  **Linear Population Size Reduction** shrinks the
  population from ``NP_init`` (default 30) down to ``NP_min``
  (default 4) over the strategy's evaluation budget — the
  characteristic move that lifted SHADE to L-SHADE and won the
  CEC-2014 competition.  Out-of-bounds components are repaired by
  midpoint reflection per Tanabe-Fukunaga §III-A.  The heuristic
  is registered in :mod:`panobbgo.heuristics`,
  :func:`default_structural_catalog` gains it as a tenth
  ``add_heuristic`` candidate (``avoid_duplicates=True`` keeps the
  catalog from cluttering portfolios that already include it),
  and :func:`default_catalog` gains three kwarg rules so the
  loop driver can also retune ``LSHADE.NP_init``,
  ``LSHADE.H``, and ``LSHADE.p_best`` once a spec opts in.
  Warm restart via :meth:`on_restart` mirrors the IPOP / PSO
  pattern: in-flight trials dropped, archive cleared, memory
  bins reset to 0.5, slots re-randomised in a small ball around
  ``center``.
* **Why** — closes the *Adaptive Differential Evolution
  (LSHADE / JADE)* follow-up below.  The shipped DE was the
  basic ``DE/rand/1/bin`` with fixed ``F = 0.8`` and ``CR = 0.9``
  — robust, but conspicuously weaker than the literature-best
  population solvers.  L-SHADE is widely cited as one of the
  strongest single-population black-box optimizers — winner of
  the CEC-2014 single-objective competition and a high-water
  mark that subsequent variants
  (jSO, IMODE, NL-SHADE-RSP) merely refine.  Adding it as a
  *new* heuristic (not a replacement) keeps the legacy DE
  available for byte-identical reproduction of older ledgers
  while giving the structural mutation catalog a strong new
  candidate that can be combined with CMA-ES, PSO, and the
  GP-based heuristics in a portfolio strategy.
* **Asynchronous adaptation** — synchronous L-SHADE applies
  parameter adaptation only at the end of each generation,
  after every individual has been re-evaluated; this port
  batches by *count* — every ``NP_current`` completed
  evolutionary trials forms one async generation.  The weighted
  Lehmer mean used by SHADE is invariant under the order of its
  contributing samples, so the adaptation cadence stays the same
  while the heuristic plays nicely with Panobbgo's event loop.
  Initial random fills do not contribute to the success buffer
  (their F/CR are NaN), and slots dropped by LPSR drop their
  pending trials silently when results return.
* **Impact** — A/B at quick mode (3 problems × 5 reps × 300
  evaluations, seed 42), comparing the same Rewarding strategy
  with and without DE / LSHADE swapped in:

  * ``Rewarding_DE``     — DeJong 0.999 / Rosenbrock 0.517 /
    Rastrigin 1.000 (mean 0.839).
  * ``Rewarding_LSHADE`` — DeJong 1.000 / Rosenbrock **0.525** /
    Rastrigin 1.000 (mean **0.842**).

  At quick budget (300 evaluations) the two variants are within
  noise (delta +0.003) — exactly as expected, because LSHADE's
  success-history adaptation needs *more* evaluations than this
  to fully outclass fixed-parameter DE.  The literature
  comparisons that establish LSHADE as the CEC-2014 winner used
  10000+ evaluations on 30D/50D problems; the value of shipping
  it for Panobbgo today is to give the structural mutation
  catalog *a state-of-the-art DE arm* the bandit can swap in on
  a per-problem basis once the loop has gathered evidence.  At
  matching cheap budgets LSHADE is a peer of fixed DE, not a
  regression — exactly the property required for safely opting
  it in via the structural catalog.
* **Backwards compatibility** — strictly safe.  L-SHADE is
  opt-in: it is not added to any default
  :func:`_make_quick_strategies` /
  :func:`_make_standard_strategies` / :func:`_make_full_strategies`
  spec, so existing CLI invocations and existing ledgers stay
  byte-identical.  The structural catalog gains it as one extra
  ``add_heuristic`` candidate; ``avoid_duplicates=True`` keeps
  the catalog from cluttering a portfolio that already has it.
  The kwarg rules only fire when a spec explicitly sets the
  matching kwarg (per :func:`_find_targets`'s "param already in
  kwargs" predicate), so a fresh ledger run on the built-in
  factories sees no behavioural change.
* **Tests** — `tests/test_heuristic_lshade.py` (39 tests):
  construction validation (8 — invalid NP_init / NP_min /
  inversion / H / p_best / archive_factor + default + custom),
  initial-swarm emission and shape (3), unknown-who ignored,
  initial-fill no-success-counted, evolutionary trials emitted
  once population reaches 4, better-trial-wins (target replaced,
  parent archived, success recorded), worse-trial-loses (target
  unchanged, archive untouched), F/CR sampling invariants
  (F ∈ (0, 1], CR ∈ [0, 1], terminal CR sentinel), weighted
  Lehmer-mean memory update with hand-built buffer and known
  expected values, memory-pointer wrap, terminal-M_CR sentinel
  (planted on all-zero CR successes, sticky once set), LPSR
  invariants (no-op when budget unknown, full-budget shrink
  to NP_min, partial-progress proportional shrink, alive-index
  consistency post-drop), bound-reflection (below / above /
  in-bound) using the actual problem box, generation-counter
  isolation from initial fills, restart behaviour (state
  cleared / center=None random fallback / before-start no-op),
  end-to-end smoke run on a quadratic where the swarm makes
  measurable progress, plus registration tests for
  :mod:`panobbgo.heuristics` and the structural and kwarg
  catalogs.

### 2026-05-08 — Hold-out validation set for the self-improvement loop

* **What** — `panobbgo/self_improve.py`:
  :class:`LoopHoldoutRecord` (a third ledger record type next to
  :class:`LoopIterationRecord` and :class:`LoopGuardRecord`) plus the
  :attr:`LoopConfig.holdout_base_seed` /
  :attr:`LoopConfig.holdout_iterations` /
  :attr:`LoopConfig.holdout_iteration_offset` /
  :attr:`LoopConfig.holdout_eps_overfit` knobs and a new
  :meth:`LoopConfig.holdout_harness_config` helper.
  :class:`SelfImprover` gains :meth:`_holdout_enabled`,
  :meth:`_measure_holdout`, and :meth:`_run_holdout` plus a public
  :meth:`run_full` entrypoint that returns
  ``(iter_records, guard_records, holdout_records)`` for tests and
  callers that want the full audit trail.  The CLI gains
  ``--holdout-base-seed``, ``--holdout-iterations``,
  ``--holdout-iteration-offset``, ``--holdout-eps-overfit``, and
  ``--fail-on-overfit`` (exits ``3`` on a flagged ladder).  The
  ``summary`` subcommand now reports hold-out outcomes alongside
  iteration and guard summaries.
* **Why** — closes the Phase 6 / §10 *Hold-out validation set*
  ticket.  The anti-cherry-pick guard catches drift inside the
  *training* base_seed family — it varies only
  ``randomize_iteration`` and keeps ``HarnessConfig.seed`` constant.
  A mutation that overfits to peculiarities of the training base_seed
  family slips through silently because the guard's "fresh" instances
  are still drawn from the same SHA-256 stream.  The hold-out
  re-measures the seed and the final top of the ladder on a
  completely independent ``base_seed``, so an overfit ladder is
  exposed by a shrinking ``top − seed`` gap on hold-out.  A bias of
  ``drift < -eps_overfit`` is flagged ``overfit=True`` and, when
  combined with ``--fail-on-overfit``, exits the CLI non-zero so the
  signal is usable as an unattended-loop tripwire.
* **Independence vs the guard** — the guard validates within the
  training instance stream (same ``base_seed``, different
  ``randomize_iteration``); the hold-out validates *across* training
  streams (different ``base_seed``, same ``randomize_iteration``
  range).  Together they cover the two axes along which the loop can
  silently overfit.
* **Defaults** — ``holdout_base_seed = 0`` (disabled) keeps existing
  CLI invocations byte-identical.  When set, the value must differ
  from :attr:`LoopConfig.base_seed`; equal values would collapse the
  hold-out to a glorified guard check on offset ``0`` and the
  ``LoopConfig`` constructor rejects them at validation time.
  ``holdout_iterations = 5``, ``holdout_iteration_offset = 0``,
  ``holdout_eps_overfit = 0.05`` are the recommended starting points.
* **Skip rules** — hold-out is skipped silently when (a) disabled,
  (b) the loop ran zero iterations, or (c) ``randomize=False`` (the
  fixed battery is unaffected by ``base_seed``, so a hold-out check
  would be no signal at all).
* **Cost** — fixed: ``2 × holdout_iterations`` harness runs at the
  end of the loop (or just ``holdout_iterations`` when the ladder
  has only the seed, since both endpoints are the same spec list).
  Cheap relative to the ``2 × iterations`` cost of the main loop.
* **Tests** — `tests/test_self_improve.py` (17 new tests, total 97):
  config validation (negative iterations, negative eps, equal
  base_seed rejection, zero-zero edge case, `holdout_harness_config`
  vs `harness_config` propagation), end-to-end behaviour
  (disabled-by-default, skipped when randomize=False, skipped on
  zero iterations, seed-only ladder records zero drift, hold-out
  uses the independent base_seed for measurement, overfit flag fires
  when gap collapses, no flag when gap holds, ledger writes
  ``record_type='holdout'`` line), back-compat (`SelfImprover.run`
  still returns a list of `LoopIterationRecord`), and `to_dict`
  round-trip with JSON serialisation.
* **Documentation updated**
  - `planning/SELF_IMPROVEMENT_LOOP.md`: §2 missing-pieces list
    refreshed; §10 Open Questions item resolved; Phase 6 checklist
    updated; this §13 entry; Next iteration ideas reduced.
  - `doc/source/guide_benchmarking.rst`: new "Hold-out validation
    set" subsection with algorithm, CLI examples, programmatic
    example, and the independence-from-the-guard note.
  - `doc/source/guide.rst`: quick-nav entry mentions the hold-out.
  - `AGENTS.md`: self-improvement loop subsection lists the
    hold-out feature with run-the-loop bash example.
  - `TODO.md`: this entry.

### 2026-05-07 — PSO adaptive inertia (Shi-Eberhart 1998)

* **What** — `panobbgo/heuristics/pso.py`: :class:`PSO` gains an
  opt-in ``w_end`` keyword argument and a new ``_current_inertia()``
  helper.  When ``w_end`` is set, the inertia weight at evaluation
  count ``e`` (out of ``E = strategy.config.max_eval``) is
  ``w_eff(e) = w − (w − w_end) · min(e/E, 1)`` — the canonical
  Shi-Eberhart (1998) linearly-decreasing schedule.  When
  ``w_end is None`` (default) ``_current_inertia()`` returns
  ``self.w`` unchanged, preserving the original Clerc-Kennedy
  constriction-coefficient behaviour byte-for-byte.  When the
  strategy budget is unknown (no ``max_eval``, zero, or non-numeric)
  the heuristic falls back to constant ``w`` rather than guessing a
  horizon.  :func:`default_catalog` gains two new
  :class:`MutationRule`s (``PSO.w`` and ``PSO.w_end``, both
  ``float_uniform`` over literature-standard bounds) so the loop
  driver can tune the adaptive-inertia schedule once a spec opts in
  by setting either kwarg explicitly.
* **Why** — closes the *Adaptive inertia* PSO follow-up.  At the
  budgets used by competition-winning PSO variants (≥ 300
  evaluations per run), the canonical fixed Clerc-Kennedy parameters
  under-explore multimodal landscapes; Shi-Eberhart inertia
  annealing is the literature-standard fix.  The extension is
  *opt-in* — the default constructor preserves the shipped
  behaviour exactly — so the loop driver can discover whether any
  given strategy benefits without disturbing existing ledgers.
* **Backwards compatibility** — strictly safe.  ``w_end`` defaults to
  ``None``; existing PSO instances retain their prior behaviour
  bit-for-bit.  The new ``PSO.w`` / ``PSO.w_end`` catalog rules only
  fire when a spec explicitly sets the kwarg (per
  :func:`_find_targets`'s "param already in kwargs" predicate), so a
  fresh ledger run on the built-in factories sees no behavioural
  change.
* **Tests** — `tests/test_heuristic_pso.py` adds 6 tests:
  default ``w_end`` is ``None``; finiteness validation; constant-``w``
  short-circuit; missing-results fall-back path; the
  linearly-decreasing schedule at four progress points; the
  zero-``max_eval`` fall-back; plus a catalog test confirming
  ``PSO.w`` / ``PSO.w_end`` rules are present.

### 2026-05-07 — PSO ring (`lbest`) topology variant

* **What** — `panobbgo/heuristics/pso.py`: :class:`PSO` gains a
  ``topology: str = "gbest"`` argument plus a ``k_neighbors: int = 2``
  half-width.  ``"gbest"`` (default, byte-identical to the 2026-05-05
  ship) keeps the canonical Kennedy-Eberhart 1995 fully-connected
  swarm; ``"lbest"`` switches every particle's social attractor to the
  best ``pbest`` in a wrap-around *ring* of width ``2·k_neighbors + 1``
  centred on the particle's own index.  Two new helpers cover the
  bookkeeping: ``_ring_neighbors(i)`` returns the wrap-around index
  list and ``_social_best_idx(i)`` returns the per-particle attractor
  (collapsing to ``_gbest_idx`` for ``gbest``).  ``_generate_next``
  consults ``_social_best_idx`` exactly where it used ``_gbest_idx``
  before, so the velocity-update / clamp / projection paths are
  shared.  :func:`panobbgo.self_improve.default_structural_catalog`
  gains a second PSO entry — ``(PSO, {"NP": 20, "topology": "lbest",
  "k_neighbors": 2})`` — alongside the existing gbest default.  Both
  entries share ``cls = PSO`` so ``avoid_duplicates=True`` still
  prevents two PSO instances from landing in the same strategy; the
  catalog samples uniformly between them when PSO is not yet present
  and skips both afterwards.
* **Why** — closes the "Topology variants" follow-up below the §13
  PSO entry from 2026-05-05.  ``gbest`` and ``lbest`` topologies
  trade off different parts of the exploration / exploitation
  spectrum: ``gbest`` contracts faster (every particle sees the same
  best), ``lbest`` slows information diffusion to one hop per
  iteration so multiple sub-swarms can probe different basins in
  parallel.  Kennedy & Mendes (CEC 2002) show ``lbest`` empirically
  beats ``gbest`` on multimodal benchmarks — exactly the regime where
  Panobbgo's standard battery (Rastrigin, Ackley, Griewank,
  Schwefel) is concentrated.  Shipping both variants in the
  structural catalog gives the self-improvement loop the vocabulary
  to pick whichever wins on the current battery.
* **Impact** — 2-seed A/B at ``--quick`` (3 problems × 5 reps × 150
  evaluations), comparing the same Rewarding strategy with PSO under
  each topology:

  * Seed 42 — ``gbest`` 0.183 / ``lbest`` **0.288** (lbest +0.105).
  * Seed 43 — ``gbest`` **0.296** / ``lbest`` 0.181 (gbest +0.115).

  Each topology wins on one of the two seeds — exactly the
  *complementarity* the literature predicts.  At ``--quick`` noise
  (~ ±0.05) neither dominates, but adding ``lbest`` to the catalog
  expands the bandit's reachable strategy space without regressing
  the gbest path: the loop now has two PSO arms with markedly
  different exploration dynamics to choose between.
* **Backwards compatibility** — strictly safe.  ``topology`` defaults
  to ``"gbest"``, so every existing PSO instance retains its prior
  behaviour bit-for-bit.  The structural catalog gains one extra
  ``add_heuristic`` candidate that shares ``cls = PSO`` with the
  existing entry — under ``avoid_duplicates=True`` (default), only
  one is ever added per strategy.  Existing ledger consumers, kwarg
  rules (``MutationRule(class_name="PSO", ...)``), and the bandit's
  ``_proposal_rule_key`` are unchanged.
* **Tests** — `tests/test_heuristic_pso.py` (13 new tests, total
  50): construction validation (default topology / lbest
  construction / invalid topology / invalid k_neighbors type / value),
  ring-neighbour wrap-around correctness, ring size invariant, lbest
  social-attractor uses ring (not the global best), gbest social
  attractor degenerates to ``_gbest_idx``, lbest returns ``None``
  before any neighbour pbest exists, lbest velocity clamp invariant,
  lbest end-to-end smoke convergence on a quadratic, and structural
  catalog now ships both gbest and lbest PSO entries.

### 2026-05-05 — Particle Swarm Optimization (`PSO` heuristic)

* **What** — `panobbgo/heuristics/pso.py` adds an asynchronous PSO
  heuristic with the canonical Clerc–Kennedy (2002) constriction-
  coefficient parameters: ``w = χ ≈ 0.7298``, ``c1 = c2 ≈ 1.49618``.
  Each particle carries a position, velocity, and personal-best
  memory; on every step the velocity update::

      v_i ← w · v_i + c1·r1·(pbest_i − x_i) + c2·r2·(gbest − x_i)
      x_i ← x_i + v_i

  pulls the particle toward both its own best and the global best
  with random per-component weights.  Velocities are clamped per
  dimension to ``v_max_frac · range`` (default 0.5) to prevent the
  swarm from exploding outside the search box.  The heuristic is
  registered in :mod:`panobbgo.heuristics` and added to the
  ``add_heuristic`` candidate pool of
  :func:`default_structural_catalog`; a kwarg rule for ``PSO.NP``
  (swarm size, range ``[8, 60]`` with ±4 / ±8 deltas) is added to
  :func:`default_catalog` so the loop can also tune the swarm
  size.  ``on_restart(center, reason)`` implements an IPOP-style
  warm restart: drop in-flight trials, scatter particles in a
  velocity-clamp ball around the new center, wipe the global
  memory, and re-seed.
* **Why** — closes a clear gap in the heuristic portfolio.  PSO is
  the third great population-based metaheuristic alongside CMA-ES
  (covariance re-sampling) and Differential Evolution (recombination
  of three random members), but its dynamics are markedly different:
  particles carry **momentum** (velocity inertia retained from the
  prior step) and a **social** attraction toward the swarm's best,
  giving fast contraction once a basin is found while still probing
  along the prior search direction.  These dynamics are
  complementary to CMA-ES and DE — they exploit ridges with
  momentum that CMA-ES has to *learn* via covariance updates and
  that DE has no concept of at all — so adding PSO to the portfolio
  diversifies the heuristic mix the bandit can choose from on any
  given problem.
* **Impact** — quick A/B at ``--quick`` (3 problems × 3 reps × 75
  evaluations, seed 42), comparing the same Rewarding strategy with
  and without PSO appended to the heuristics list:

  * ``Rewarding_NoPSO``  — DeJong 1.000 / Rosenbrock 0.000 /
    Rastrigin 1.000 (mean 0.667).
  * ``Rewarding_WithPSO`` — DeJong 1.000 / Rosenbrock **0.031** /
    Rastrigin 1.000 (mean **0.677**).

  Adding PSO upgrades the Rosenbrock pair from 0/3 reps solved to
  2/3 reps solved (success rate 0% → 67%) without regressing on
  DeJong or Rastrigin.  Rosenbrock is exactly the regime where
  momentum helps — a narrow curved valley where vector inertia along
  the valley floor is more useful than the Gaussian re-sampling of
  Random / Nearby / NelderMead.  At the noisy ``--quick`` level a
  delta of ``+0.01`` is within noise; the meaningful signal is the
  per-pair upgrade on Rosenbrock.
* **Backwards compatibility** — strictly safe.  PSO is opt-in: it is
  not added to any default :func:`_make_quick_strategies` /
  :func:`_make_standard_strategies` / :func:`_make_full_strategies`
  spec, so existing CLI invocations and existing ledgers stay
  byte-identical.  The structural catalog gains it as one extra
  ``add_heuristic`` candidate; its ``avoid_duplicates=True`` invariant
  keeps the catalog from cluttering a portfolio that already has it.
* **Tests** — `tests/test_heuristic_pso.py` (24 tests):
  construction validation (8 — invalid NP / w / c1 / c2 / v_max_frac
  + default + custom + name), initial-swarm emission and shape (3),
  pbest / gbest update + follow-up trial (5), velocity clamp
  invariant (1), restart behaviour (3 — clears pbest, before-start
  no-op, ``center=None`` random fallback), an end-to-end smoke run
  on a quadratic where the swarm strictly improves, and registration
  tests for ``panobbgo.heuristics`` and the structural catalog.

### 2026-05-03 — Strategy portfolio composition (`StructuralMutationRule`)

* **What** — `panobbgo/self_improve.py`:
  :class:`StructuralMutationRule` joins :class:`MutationRule` as a
  first-class catalog rule.  Two ops:

  * ``add_heuristic`` appends one of ``candidate_classes`` (a
    ``(HeuristicClass, default_kwargs)`` pool) to a target strategy.
    ``avoid_duplicates=True`` (default) skips classes already present
    in the strategy so the catalog cannot clutter a portfolio with
    redundant copies of the same heuristic.
  * ``drop_heuristic`` removes one heuristic, optionally restricted to
    ``droppable_classes``.  ``min_heuristics`` (default ``2``) is the
    floor of the *post-drop* heuristic count, so the strategy always
    keeps a diversity slot.

  :class:`MutationProposal` gains ``op`` and ``structural_kwargs``
  fields that are populated only for structural ops; kwarg proposals
  serialise byte-identically to before.  :func:`apply_mutation`
  dispatches on ``proposal.op`` and falls through to the existing
  kwarg path for non-structural proposals.  The Thompson sampler maps
  every structural rule onto one arm per ``op``
  (``("*", op, "structural")``) which keeps cold-start variance bounded
  while still letting the bandit learn whether portfolio expansion or
  contraction wins on the current battery.
  :func:`default_structural_catalog` returns
  ``default_catalog().rules + [StructuralMutationRule(add), StructuralMutationRule(drop)]``
  so the existing ledger and CI defaults are unchanged — opt in via
  ``--structural`` on ``scripts/self_improve.py run`` or by passing
  the catalog explicitly to :class:`SelfImprover`.
* **Why** — closes the §7.2 *Strategy portfolio composition* item.  The
  loop driver shipped in Phase 5 only retunes existing kwargs, so it
  could discover better dial settings but never a better composition.
  Most measurable Panobbgo wins to date have come from composition
  changes (adding Sobol' for the BayesOpt initial design,
  splitting CMAES strategies into IPOP/BIPOP variants, etc.) — exactly
  the moves the loop now has the vocabulary to make autonomously.
* **Backwards compatibility** — strictly safe.  :func:`default_catalog`
  is unchanged; :class:`MutationProposal` keeps the same required
  fields and adds ``op`` / ``structural_kwargs`` as keyword-only with
  ``None`` defaults; :meth:`MutationProposal.to_dict` only emits the
  new keys when ``op`` is set, so existing ledger consumers parse the
  old layout byte-identically.  The bandit's
  :func:`_proposal_rule_key` collapses structural ops onto the
  ``("*", op, "structural")`` arm; kwarg keys are unchanged so
  prior-ledger priming still recovers identical statistics.
* **Tests** — `tests/test_self_improve.py` (29 new tests, total 92):
  rule validation, applicable-hits enumeration (add / drop /
  ``avoid_duplicates`` / ``droppable_classes`` / ``min_heuristics``
  floor / strategy_pattern filter), proposal serialisation, the
  apply-side dispatch (add appends, drop removes, missing class
  raises, empty-strategy refusal, fallback-import path),
  :func:`_proposal_rule_key` collapse for structural ops, the
  Thompson sampler bucketing structural history into one arm, and an
  end-to-end loop run that accepts a structural drop on a fake
  harness.
### 2026-05-02 — Stratified dimension sampling for multi-dim families

* **What** — `panobbgo/harness_randomized.py`:
  :class:`ProblemFamily` gains a ``stratify_dims: bool = True`` field and
  a :meth:`stratified_dim_for_rep` helper that returns
  ``dim_choices[rep % k]``.  :meth:`ProblemFamily.sample_instance` now
  accepts an optional ``dim`` override so callers can pin the dim
  without consuming the rng's ``choice`` slot.
  :meth:`RandomizedProblemSpec.create_problem_for_rep` calls
  ``stratified_dim_for_rep(rep)`` for multi-dim families with
  ``stratify_dims=True`` (the default) and falls back to the rng's
  ``choice`` otherwise.  ``last_sampled_params()`` now reports a
  ``stratified_dim: bool`` flag for ledger introspection.
* **Why** — closes the §10 *Composite score stability across dimension
  sampling* item.  Without stratification, a multi-dim family with
  ``dim_choices = (2, 5, 10)`` and 5 reps could draw, say, three
  ``dim=2`` instances on iteration 5 and three ``dim=10`` instances on
  iteration 6.  Higher-dim instances are systematically harder, so a
  per-iteration composite delta picks up dim-mix noise on top of the
  signal of the underlying mutation, polluting the bootstrap CI on
  which §6.2 acceptance depends.  Cyclic stratification (rep ``i`` →
  ``dim_choices[i % k]``) makes any contiguous block of ``k`` reps
  cover every declared dim exactly once, eliminating that noise source
  by construction without changing the per-iteration eval count.
* **Impact** — purely a measurement-noise improvement: the default
  battery of families all use ``dim_choices=(2,)`` (single dim), so
  this change is a no-op for the byte-level reproducibility of the
  current standard mode.  The benefit materialises when users (or the
  loop) declare multi-dim families — e.g. via
  ``HarnessConfig.extra_families`` — at which point the cross-iteration
  variance of the composite drops by roughly ``Var(dim_mix)`` (the
  fraction of total variance attributable to which dim was sampled,
  typically a substantial slice for hard families like Rosenbrock).
* **Backwards compatibility** — strictly safe.  Single-dim families
  (the entire default battery) are unaffected because the cyclic
  schedule degenerates to a constant.  The public :class:`ProblemFamily`
  signature gains a new keyword-only field with a default; existing
  ``ProblemFamily(...)`` callers keep working byte-identically.  The
  ``stratify_dims=False`` path preserves the previous behaviour for
  anyone who needs it (e.g. for replicating an old ledger).
* **Tests** — `tests/test_harness_randomized.py` (16 new tests, total
  68): cyclic schedule correctness, balance over a complete cycle,
  imbalance bound on partial cycles, single-dim no-op, dim-override
  validation, rng-stream invariance proof (override does not consume
  the choice slot), end-to-end :class:`RandomizedProblemSpec` round
  trip, ``last_sampled_params`` flag round trip, and the contract that
  default families remain unchanged.

### 2026-05-01 — Adaptive mutation sampler (Thompson sampling)

* **What** — `panobbgo/self_improve.py` gains
  `AdaptiveMutationSampler` plus `MutationRuleStats` and the public
  `RuleKey` alias.  Each :class:`MutationRule` becomes one arm of a
  Bernoulli bandit whose reward is "this iteration was accepted".  On
  every `sample()` call the sampler draws one variate per applicable
  rule from `Beta(prior_alpha + n_accepts, prior_beta + n_attempts -
  n_accepts)` and picks the arg-max — the canonical Thompson rule.
  Inside the chosen rule, hits are still selected uniformly (which
  spec / which slot), exactly as the catalog's uniform sampler does.
  History is primed from a prior JSONL ledger via
  `prime_from_ledger`, so the loop carries learning across restarts.
  `LoopConfig` gains `adaptive_sampling`, `adaptive_prior_alpha`,
  `adaptive_prior_beta`, `adaptive_prime_from_ledger`; the
  `scripts/self_improve.py` CLI gains the `--adaptive` family of
  flags.  After each iteration's accept/reject decision, the driver
  calls `sampler.record_outcome()` so future samples are biased
  toward rules with positive accept history.
* **Why** — closes the §10 "Adaptive mutation sampler" item.  The
  uniform catalog sampler shipped in Phase 5 wastes iterations on
  rules that never produce accepts.  Thompson sampling concentrates
  probability mass on empirically winning rules while still exploring
  unfamiliar ones — the canonical fix for the *productivity* gap of
  multi-armed bandit problems.  Cold-start equivalence to uniform
  (Beta(1, 1) ≡ U(0, 1), and arg-max of i.i.d. uniforms is uniform)
  makes the upgrade strictly safe: flipping the flag on a fresh
  ledger reproduces the prior behaviour distributionally, then
  diverges as evidence accumulates.
* **Defaults** — `adaptive_sampling = False` keeps existing CLI
  invocations byte-identical.  `adaptive_prior_alpha = adaptive_prior_beta
  = 1.0` is the symmetric uninformed prior; lower priors (e.g. `0.5`)
  make the sampler greedier earlier at the cost of more variance.
* **Tests** — `tests/test_self_improve.py` (23 new tests, total 63):
  invalid priors, cold-start equivalence to uniform sampling,
  arg-max behaviour after biased training, record-outcome
  correctness, ledger priming, integration with `SelfImprover`
  including the `sampler=` override and the `adaptive_prime_from_ledger`
  flag.

### 2026-04-27 — Sobol' quasi-random initial design (`Sobol` heuristic)

* **What** — `panobbgo/heuristics/sobol.py` adds a low-discrepancy
  quasi-random (Sobol') sampler as a one-shot space-filling heuristic;
  registered alongside `LatinHypercube`, `Random`, etc.  A new
  `BayesOpt_Sobol` strategy in the standard harness pairs it with the GP
  surrogate, `Nearby`, and `NelderMead`.  The mutation catalog
  (`panobbgo.self_improve.default_catalog`) gains a rule that nudges
  `Sobol.n` in 4-step increments inside `[4, 64]` so the loop driver can
  also tune it.
* **Why** — every modern Bayesian-optimization library (BoTorch, TuRBO,
  scikit-optimize, GPyOpt) defaults to Sobol' for the initial design
  precisely because lower discrepancy → better surrogate fits at low
  sample counts.  Panobbgo only had Latin Hypercube before.
* **Impact** — measured head-to-head over 5 reps × 7 standard problems at
  budget 200, mean per-pair score `BayesOpt_Sobol = 0.314` vs
  `BayesOpt_GP = 0.191` (`+0.123`).  Sobol' wins on 5 / 7 problems
  (DeJong, Rosenbrock_2D, Ackley, StyblinskiTang, Griewank tied with
  smaller best-distance), loses on 2 (Rastrigin, Rosenbrock_5D).
* **Tests** — `tests/test_heuristic_sobol.py` (16 tests).

### 2026-04-26 — Anti-cherry-pick guard + tests for the loop driver

* **What** — `panobbgo/self_improve.py` gains
  `LoopConfig.guard_interval`, `guard_eps_ladder`, and
  `guard_iteration_offset` plus the `LadderEntry` and `LoopGuardRecord`
  data structures.  Every `guard_interval` iterations the loop
  re-measures the top of the accepted ladder on a *fresh*
  `randomize_iteration` (`iteration + guard_iteration_offset`) and rolls
  the ladder back when the composite has drifted more than
  `guard_eps_ladder` below the entry's stored `last_validated_score`.
  The seed entry is the trusted fallback and is never popped.  Exposed
  via `--guard-interval` / `--guard-eps-ladder` /
  `--guard-iteration-offset` on `scripts/self_improve.py run` and the
  `summary` subcommand reports rollbacks.
* **Why** — closes §6.3 ("Anti-cherry-pick guard") of this plan.  Even
  with the parametrically randomized battery, a sequence of "lucky"
  instance draws can inflate per-iteration after-scores enough to clear
  the bootstrap CI even when the underlying mutation does not
  generalise.  The guard validates the ladder against an independent
  instance stream so silent overfitting cannot accumulate.
* **Tests** — `tests/test_self_improve.py` (40 tests, new) — also fills
  the test gap left by Phase 5 (the loop driver shipped without
  coverage).  Covers `MutationRule` validation, catalog sampling, the
  `apply_mutation` immutability contract, end-to-end runs against a
  faked harness, the guard's cadence / no-rollback / drift-rollback /
  offset-id / seed-not-popped invariants, and ledger round-trip.
* **Defaults** — `guard_interval = 0` keeps existing CLI invocations
  byte-identical.  `5` or `10` is the suggested setting for unattended
  multi-hour runs.

### Next iteration ideas

Lightweight "next ticket" notes for follow-up agents — graduate them to
a dated entry above when shipped.

> **Before implementing any idea below, run `gh pr list --state open`
> (drafts included).** These notes live on `master` and do not reflect
> work that is already sitting in an unmerged PR. If a candidate is
> already covered by an open PR, finish/merge that PR instead of opening
> a duplicate — see §12.3 step 0. (Four duplicate NL-SHADE-RSP PRs,
> #227–#230, were the cost of skipping this check.)

#### LBFGSB follow-ups (after 2026-05-27 ship)

Multi-start L-BFGS-B shipped 2026-05-27 (see §13) and joined the
structural ``add_heuristic`` pool.  The A/B showed a *dedicated* LBFGSB
strategy cracks ``Rosenbrock_5D`` (≈3e-11) where every default strategy
scores 0.0, but *adding* it to the budget-split ``Rewarding_Diverse``
portfolio does not (and can regress other problems).  Natural
follow-ups:

- **Dedicated gradient-local-search default strategy (needs ADR).**
  Add a ``LocalSearch_LBFGSB`` (or ``StrategyPhased`` global→local) spec
  to ``_make_standard_strategies`` / ``_make_full_strategies`` so the
  *default battery* gains a strategy that actually solves smooth
  valleys.  This shifts the historical composite baseline, so it needs
  an architectural decision record (existing ladders are not directly
  comparable to the new battery) — the same gate the ``LSHADE_jSO``
  idea below carries.  Measure with `compare --statistical
  --fail-on-regression` first: a gradient arm helps the smooth /
  ill-conditioned problems (Rosenbrock, DixonPrice, Zakharov) but is
  useless on the multimodal ones (Rastrigin, Ackley, Schwefel), so the
  *net* composite effect must be measured, not assumed.
- **Warm-start restarts from the portfolio best.** Today the worker's
  restarts (after the first box-centre descent) are pure uniform-random.
  A refiner that warm-starts each restart from a perturbation of
  ``strategy.best`` would exploit the basin the rest of the portfolio
  has found — turning random multi-start into basin-hopping refinement.
  Needs a small protocol extension (the worker requests an ``x0`` from
  the parent at the start of each round rather than drawing it locally),
  because the global best is only known parent-side.
- **`LBFGSB.max_starts` catalog rule.** ``max_starts`` defaults to
  ``None`` (unlimited until budget).  An ``integer_add`` or
  ``categorical_choice`` rule that fires when a spec sets it explicitly
  would let the loop tune the exploration / exploitation balance of the
  multi-start schedule, the same way ``LSHADE.archive_factor`` is tuned.

#### Ship a jSO-tuned `LSHADE_jSO` strategy in `_make_standard_strategies`

The iLSHADE / jSO adaptive ``p_best`` schedule shipped 2026-05-19 is
*opt-in*: it only fires when a spec sets ``p_best_end`` explicitly.
None of the built-in :func:`_make_quick_strategies` /
:func:`_make_standard_strategies` / :func:`_make_full_strategies`
factories currently produce a spec with the canonical jSO settings
(``NP_init = 18·d``, ``p_best = 0.25``, ``p_best_end = 0.125``), so
the standard battery never exercises the new schedule out-of-the-box.
A natural follow-up is to add a dedicated ``LSHADE_jSO`` strategy to
``_make_standard_strategies`` so the composite score on the standard
battery directly reflects the literature-best DE refinement.  The
trade-off is that this would shift the historical composite score
baseline — needs an architectural decision record because existing
ladders won't be directly comparable to the new battery.

#### jSO asymmetric F-cap during early generations — shipped 2026-05-21

Shipped 2026-05-21 as
:attr:`panobbgo.heuristics.lshade.LSHADE.F_schedule` plus the
inherited :meth:`~panobbgo.heuristics.lshade.LSHADE._apply_F_cap`
that :class:`~panobbgo.heuristics.jso.JSO` opts into by
construction.  The three-phase cap (``F ≤ 0.7`` while
``progress < 0.6``, ``F ≤ 0.8`` while ``0.6 ≤ progress < 0.9``,
unclamped in the final 10%) is now shared infrastructure rather
than per-subclass code.  The 2026-05-15 :class:`JSO` ship had only
the first phase of the cap implemented; this entry completes the
literature-faithful three-phase cap from Brest et al. (2017,
§III-D).  See the §13 entry above.  :func:`default_catalog` gains
``LSHADE.F_schedule`` as a categorical rule so the loop can flip an
existing L-SHADE instance between the Tanabe-Fukunaga and jSO
regimes without dropping and re-adding the heuristic.

#### Tighten `eps_accept` once paired bootstrap is the loop default

The paired bootstrap shipped 2026-05-14 substantially narrows the
composite-delta CI under the randomized harness — typically 3–10× on
the loop's regime of 5 reps × ~3 problems at quick mode.  The
historical defaults of ``eps_accept=0.005`` and ``n_boot=2000`` were
sized for the (much wider) unpaired CI, so under paired sampling the
loop now leaves signal on the floor: a true ``+0.003`` improvement
whose CI does not bracket zero is still rejected for *"composite delta
≤ eps_accept"*.  Once a few hundred ledger entries have accumulated
under the paired default, lower ``eps_accept`` to ``0.002`` (or auto-
size it from the recently observed CI width) and consider trimming
``n_boot`` to ``500`` since the paired sampler converges faster.  Ship
the change with a ledger archive marker so the bandit's prior beliefs
do not silently mix the old and new accept regimes.  Pairs naturally
with the *Hierarchical / contextual bandit* idea below — both improve
loop *productivity* (accepts per iteration) rather than reach.

#### Contextual / hierarchical bandit over mutation rules

The Thompson sampler shipped 2026-05-01 treats every rule as an
independent arm.  A natural upgrade is to share strength across
rules that target the same heuristic class (one `Heuristic`-level
posterior) or the same kind (`log_uniform_perturb` posteriors borrow
strength across all classes).  Particularly valuable when the
catalog grows beyond a handful of rules and per-rule data is sparse.
Implementation: replace the flat `Dict[RuleKey, Stats]` with a
hierarchical Beta-Binomial or Dirichlet-Multinomial prior; expose
the grouping policy via the catalog itself.

#### Multi-dim default battery (now that stratification is shipped)

Stratified dimension sampling shipped 2026-05-02.  The default battery
in :func:`panobbgo.harness_randomized.make_default_families` still uses
``dim_choices=(2,)`` everywhere because expanding it would shift the
historical composite score baseline.  A natural follow-up is to add a
``make_default_families_multidim()`` factory (or a `--dim-mix` CLI
flag) that ships ``dim_choices=(2, 5, 10)`` for Rastrigin / Ackley /
DeJong, exposing the new stratification and giving the loop a richer
generalisation signal.  Needs an architectural decision record because
the resulting composite is not directly comparable to the existing
ladder.

#### Strategy portfolio composition (§7.2) — shipped 2026-05-03

Strategy portfolio composition shipped as
:class:`panobbgo.self_improve.StructuralMutationRule` and
:func:`panobbgo.self_improve.default_structural_catalog` — opt in with
``--structural`` on ``scripts/self_improve.py run`` or by passing
``catalog=default_structural_catalog()`` to :class:`SelfImprover`.  See
the §13 entry.  Natural next refinements:

- **Per-class arms in the bandit** — shipped 2026-05-18 as
  :attr:`panobbgo.self_improve.AdaptiveMutationSampler.per_class_structural`
  and :attr:`LoopConfig.structural_per_class_arms`.  Opt in via
  ``scripts/self_improve.py run --adaptive --structural-per-class-arms``.
  Each ``StructuralMutationRule`` is expanded at sampling time into one
  Thompson arm per candidate class so the bandit can learn that, e.g.,
  ``add Sobol`` wins while ``add Random`` loses.  See the §13 entry.
  Pairs naturally with the *contextual / hierarchical bandit* idea
  above — per-class arms are exactly the leaf nodes a hierarchical
  posterior would share strength across.
- **Analyzer add/drop** — symmetric to the heuristic ops; the
  ``Sensitivity`` / ``Restart`` analyzers are obvious candidates because
  they already opt in via ``StrategySpec.analyzers``.
- **Strategy-class swap** — replace ``StrategyRewarding`` with
  ``StrategyUCB`` etc. without touching the heuristics list.  Requires
  every accepted swap to keep the strategy's hyperparameters either
  compatible or to drop them on the floor; needs a translation table.

#### PSO follow-ups (after 2026-05-05 ship)

PSO landed 2026-05-05; the ``lbest`` ring topology shipped 2026-05-07
and the optional Shi-Eberhart adaptive inertia (``w_end``) shipped
2026-05-07.  Natural extensions when the loop has collected enough
evidence to motivate the work:

- **Von Neumann topology — shipped 2026-05-22**.
  :attr:`panobbgo.heuristics.pso.PSO.topology = "vonneumann"` adds a
  4-connected 2-D toroidal grid (Kennedy & Mendes 2003; Mendes 2004)
  as a third topology slot — instantaneous (gbest) / one-hop ring
  (lbest) / two-hop planar (vonneumann).  The structural catalog
  ships all three PSO variants; the ``PSO.topology`` categorical rule
  grows to ``("gbest", "lbest", "vonneumann")``.  See the §13 entry
  above.
- **Random re-wired topology** — the remaining slot in the
  Mendes 2004 set.  Unlike gbest / lbest / vonneumann (all static
  graphs computable from ``NP``), a *random* graph is rebuilt every
  ``on_restart`` (or at construction time and persisted across
  restarts — design decision).  Adds rng state per particle and an
  adjacency-list field; the social-attractor lookup uses the
  per-particle adjacency list instead of a closed-form neighbour
  set.  Useful when the bandit evidence shows neither pure
  structured topology consistently wins on a given battery.
- **`StrategyPhased` integration** — pair PSO (global exploration
  phase) with NelderMead / LBFGSB (local refinement phase) on a
  single budget split, similar to the existing ``IPOP_CMAES``
  strategy.  Would be a new entry in
  ``_make_standard_strategies`` once measured to be a net win.
- **Categorical / topology mutation rule** — shipped 2026-05-13.
  ``MutationRule(kind="categorical_choice", choices=...)`` joined the
  numeric kinds (``log_uniform_perturb`` / ``integer_add`` /
  ``float_uniform``).  The default catalog wires it up for
  ``PSO.topology``, ``Sobol.scramble`` and ``LSHADE.archive_factor``.
  See the §13 entry.

#### Adaptive Differential Evolution (LSHADE / JADE) — shipped 2026-05-10

L-SHADE shipped 2026-05-10 as
:class:`~panobbgo.heuristics.lshade.LSHADE`; see the §13 entry.
Natural follow-ups when the loop has collected enough evidence to
motivate the work:

- **JADE archive sampling distribution** — L-SHADE samples ``r2``
  uniformly from the ``population ∪ archive`` union.  JADE
  (Zhang-Sanderson 2009) uses a slightly different rule that
  weights archive entries by recency; this could be a small
  per-step refinement.
- **L-SHADE-RSP / NL-SHADE-RSP follow-on variants** — NL-SHADE-RSP
  (CEC 2021 winner) shipped 2026-05-25 as
  :class:`~panobbgo.heuristics.nl_shade_rsp.NLSHADE_RSP` (rank-based
  selective pressure, non-linear population reduction, randomised
  adaptive archive); see the §13 entry.  The remaining successor,
  NL-SHADE-LBC (CEC 2022), is queued under the *NL-SHADE-RSP
  heuristic* next-iteration idea below.
- **iLSHADE / jSO adaptive p_best schedule** — shipped 2026-05-19
  as the opt-in ``LSHADE.p_best_end`` kwarg plus the
  :meth:`LSHADE._current_p_best` helper.  See the §13 entry.
- **iLSHADE / jSO heuristic class** — shipped 2026-05-15 as
  :class:`~panobbgo.heuristics.jso.JSO`, a direct subclass of L-SHADE
  with the Brest-Maučec-Bošković (CEC 2017) refinements: weighted
  ``current-to-pbest-w/1`` mutation, linear ``p_best`` schedule
  (``0.25 → 0.125``), Cauchy-F clamping in the early phase, jSO
  initial memory values (``M_F = 0.3``, ``M_CR = 0.8``), and a
  frozen anchor memory bin at ``M_F = M_CR = 0.9``.  See the §13
  entry above.  jSO is the **CEC-2017 single-objective
  bound-constrained competition winner**.
- **Categorical mutation rule for ``LSHADE`` archive on/off** —
  shipped 2026-05-13.  The default catalog now contains an
  ``archive_factor`` rule with ``choices=(0.0, 1.0, 2.6)`` that fires
  whenever a spec sets ``archive_factor`` explicitly.  See the §13
  entry.

#### jSO follow-ups (after 2026-05-15 ship)

jSO landed 2026-05-15 as :class:`~panobbgo.heuristics.jso.JSO`; see
the §13 entry.  Natural extensions when the loop has collected
enough evidence to motivate the work:

- **NL-SHADE-RSP** — CEC-2021 winner; **shipped 2026-05-25** as
  :class:`~panobbgo.heuristics.nl_shade_rsp.NLSHADE_RSP`, a direct
  :class:`JSO` subclass with rank-based parent selection, non-linear
  population reduction, and a randomised adaptive archive.  See the
  §13 entry.  The CEC-2022 successor **NL-SHADE-LBC** (adds a linear
  bias-correction mechanism) is queued under the *NL-SHADE-RSP
  heuristic* next-iteration idea.
- **L-SHADE-cnEpSin** — independently developed competitive
  ensemble (Awad et al. CEC 2017) that combines an ensemble of
  sinusoidal F schedules with the SHADE memory.  A different
  branch of the DE family tree from jSO; useful if the bandit
  evidence ever shows neither jSO nor vanilla L-SHADE consistently
  wins on noisy / multi-modal landscapes.
- **Auto-tuned ``H``** — Brest et al. report ``H = 5`` as best for
  the CEC battery; the loop currently has no rule for ``JSO.H``
  because the constructor enforces ``H >= 2`` (anchor bin
  separation).  A rule with ``bounds=(2, 10)`` would expose this
  knob on opt-in specs the same way ``LSHADE.H`` does.
- **Categorical mutation rule for ``JSO.p_best_max``** — three
  literature-canonical settings (0.11 from L-SHADE, 0.25 from jSO,
  0.4 from iLSHADE) make a natural ``categorical_choice`` slot
  alongside the existing ``float_uniform`` rule.  Would let the
  loop flip between the three regimes the same way
  ``LSHADE.archive_factor`` flips between archive on / off / RSP.

#### BOBYQA / NEWUOA / COBYQA local optimizer — shipped 2026-05-12

COBYQA (Ragonneau-Zhang 2023) — the modern Powell-family successor
to BOBYQA / NEWUOA / LINCOA — shipped 2026-05-12 as
:class:`~panobbgo.heuristics.cobyqa.COBYQA`; see the §13 entry.
Natural follow-ups when the loop has collected enough evidence to
motivate the work:

- **Constraint-aware variant** — COBYQA natively supports linear
  and nonlinear constraints; today the adapter only wires the box
  bounds.  A second variant that passes the strategy's constraint
  set to ``scipy.optimize.minimize(constraints=...)`` would let
  COBYQA exploit the constraint geometry directly instead of
  going through the penalty-handler indirection.  Useful when the
  problem has explicit constraints whose shapes are known.
- **Warm-start interpolation reuse** — every restart today rebuilds
  the ``2·n + 1`` interpolation set from scratch (a fresh
  subprocess).  COBYQA's reference implementation does not expose
  a persistent solver state in scipy's wrapper, but a vendored
  build of the upstream ``cobyqa`` library could be configured to
  warm-start the interpolation set from the last successful
  iterate — saving the first ``2·n`` evaluations on every
  restart.
- **Categorical mutation rule for ``scale`` on/off** — see the
  PSO-follow-up entry above; the same ``categorical_choice``
  mutation rule would let the loop flip an existing COBYQA
  instance's ``scale`` kwarg without going through the full
  ``add_heuristic`` / ``drop_heuristic`` cycle.

#### Multi-seed hold-out for robust drift estimation — shipped 2026-05-16

Multi-seed hold-out shipped 2026-05-16 as
:attr:`panobbgo.self_improve.LoopConfig.holdout_base_seeds` (the
list-typed sibling of the scalar ``holdout_base_seed``) and the
``--holdout-base-seeds`` CLI flag.  See the §13 entry.  Natural
follow-ups when the loop has collected enough evidence to motivate
the work:

- **Bootstrap CI on the drift estimate — shipped 2026-05-17**.
  :func:`panobbgo.self_improve.aggregate_holdout_drift` plus
  :class:`HoldoutDriftAggregate` and the per-iteration paired score
  lists on :class:`LoopHoldoutRecord` (``seed_iteration_scores`` /
  ``top_iteration_scores``) pool drifts across all hold-out records
  and bootstrap a CI on the aggregate.  CLI gains
  ``--fail-on-overfit-ci`` (stricter sibling of
  ``--fail-on-overfit``) plus ``--holdout-ci-confidence`` and
  ``--holdout-ci-n-boot`` knobs.  See the §13 entry.
- **Auto-rollback on multi-seed overfit** — when several seeds
  agree the ladder is overfit, the loop could automatically pop the
  ladder back to the seed and penalise the bandit (see
  *Auto-rollback on hold-out overfit* below).  Multi-seed evidence
  is strong enough to act on, whereas single-seed evidence might
  still be a fluke.  Now even better-motivated with the
  bootstrap-CI rule above: the CI verdict is a more reliable
  trigger than per-seed point checks.

#### Auto-rollback on hold-out overfit

When the hold-out flags ``overfit=True``, the loop currently just
records and (optionally) exits.  A more aggressive remediation is
to automatically pop the ladder back to the seed entry and persist
the rollback in a new ``LoopHoldoutRollbackRecord`` so a subsequent
``--adaptive-prime-from-ledger`` resume picks up the failure as a
negative reward signal for *all* the rules that contributed to the
discarded ladder.  Needs care around the bandit semantics: penalising
all rules along the discarded path is more aggressive than penalising
only the last one, and the right policy is an open question.

#### Hierarchical bandit over the per-class structural arms

Per-class structural arms shipped 2026-05-18 (see §13 entry above).
The natural next refinement is to make the per-class arms *share
strength* via a hierarchical Beta-Binomial: each ``add_heuristic``
arm's posterior would borrow from the op-level posterior so a fresh
candidate class starts with the op's aggregate accept rate rather
than the symmetric ``Beta(1, 1)`` prior.  This addresses the
sparsity trade-off the per-class flag introduces — with N candidate
classes, the per-class flag divides the bandit's data by roughly N,
which can hurt early-iteration sample efficiency.  A hierarchical
prior recovers the data-sharing of the wildcard arm while preserving
the per-class arg-max.

Implementation sketch.  Replace the flat
``Dict[RuleKey, MutationRuleStats]`` with a two-level structure: one
``MutationRuleStats`` per op (``("*", op, "structural")``) plus the
per-class stats.  On Thompson draw, sample
``alpha = prior_alpha + n_class_accepts + κ · n_op_accepts``
(similarly for ``beta``) where ``κ ∈ [0, 1]`` is a "borrow"
coefficient — ``κ = 0`` recovers today's per-class arms, ``κ = 1``
recovers the collapsed wildcard.  Validation: an end-to-end loop
that observes ``add Sobol`` winning should still drive Sobol's
arm up faster than Random's, while a brand-new candidate class
starts with the op's empirical accept rate instead of the cold
prior.

#### Tunable F-cap breakpoints / cap values on `LSHADE.F_schedule`

The F-cap shipped 2026-05-21 hard-codes the canonical Brest et al.
2017 breakpoints (0.6 / 0.9) and cap values (0.7 / 0.8).  These are
the literature defaults; other variants in the DE family use
different settings.  Once enough ledger evidence has accumulated for
the categorical ``LSHADE.F_schedule`` rule, a natural follow-up is to
make the cap geometry tunable.  Two design sketches:

* **Multiple categorical regimes.** Replace the binary
  ``F_schedule = True / False`` with a categorical choice over
  named regimes — ``"off"``, ``"jso"`` (current 0.6 / 0.7 + 0.9 / 0.8),
  ``"ilshade"`` (different breakpoints / caps from Brest 2016),
  ``"strict"`` (more aggressive — e.g., F ≤ 0.5 throughout the first
  half).  Each regime ships as a module-level constant tuple so the
  bandit can flip between them without touching the heuristic body.
* **Continuous parameters.** Expose ``F_cap_phase1``, ``F_cap_phase2``,
  ``F_cap_bound1``, ``F_cap_bound2`` as four kwargs with bounded
  ``float_uniform`` perturbations.  Wider mutation space but lets the
  bandit climb the cap surface continuously.  Risk: any cap above
  0.85-ish probably no-ops because the L-SHADE Cauchy sampler rarely
  draws ``F > 0.9`` from healthy memory bins.

The categorical-regime approach has lower bandit dimension and is
literature-grounded — pick that first if you ship the follow-up.

#### Inactivity-guarded loop productivity

The most recent unattended ledger (planning/self_improve_summary.txt)
shows 1 accept in 86 iterations (~1.2 %).  That is small enough that
the bandit's posterior remains close to the prior for most arms —
defeating the point of adaptive sampling.  Two complementary moves:

* **Bump the harness mode for the cron** — quick mode at 3 reps is
  the noise floor.  A 30-iteration loop at ``--standard`` (5 reps,
  larger budget) may produce more genuine accepts than 100
  iterations at ``--quick``.  Needs a self-hosted runner because
  GitHub-hosted runners are 2 cores.
* **Relax ``eps_accept`` adaptively** — if the loop has gone N
  iterations without an accept, temporarily lower ``eps_accept`` to
  half the configured value (or use the bootstrap CI alone, with no
  point-delta gate).  Re-tighten on the next accept.  Documented in
  the ledger record so an auditor can replay the loop with the
  effective rule.  Care: the §11 success criteria pin ``eps_accept``
  at a fixed level so a chronic relaxation would silently shift the
  loop's "improvement" bar.

#### NL-SHADE-RSP heuristic (CEC 2021 winner) — shipped 2026-05-25

NL-SHADE-RSP shipped 2026-05-25 as
:class:`~panobbgo.heuristics.nl_shade_rsp.NLSHADE_RSP`, a direct
subclass of :class:`~panobbgo.heuristics.jso.JSO` adding Non-Linear
Population Size Reduction, Rank-based Selective Pressure on the ``r1``
draw (``k_rank``), and a randomised adaptive archive.  See the §13
entry above.  The three jSO override points were extracted into the
behaviour-preserving base-class hooks :meth:`LSHADE._select_r1`,
:meth:`LSHADE._lpsr_target`, and :meth:`LSHADE._archive_cap`.

Natural follow-ups when the loop has collected enough evidence to
motivate the work:

* **Adaptive crossover blend + pA archive adaptation** — the two
  CEC-2021 mechanisms intentionally *not* ported in the 2026-05-25
  ship.  (1) NL-SHADE-RSP adapts the probability of binomial vs
  exponential crossover from their relative success; (2) it adapts
  ``pA`` — the probability of drawing ``r2`` from the archive — from
  the relative improvement of archive- vs population-sourced trials,
  rather than the randomised-cap stand-in shipped here.  Both need
  per-trial bookkeeping (which crossover operator / archive source a
  trial used) that the current ``_TrialMeta`` does not carry; adding
  two optional fields to ``_TrialMeta`` and the matching success
  accounting in ``on_new_results`` is the clean shape.
* **NL-SHADE-LBC** (CEC 2022 winner) — the successor that adds a
  *linear bias-correction* mechanism on top of NL-SHADE-RSP.
  Subclassing :class:`NLSHADE_RSP` is the obvious shape now that the
  RSP / NLPSR / archive hooks exist.
* **Categorical ``k_rank`` regimes** — the ``NLSHADE_RSP.k_rank``
  rule is currently ``float_uniform [1, 5]``.  A ``categorical_choice``
  over the literature-canonical settings (``0`` = uniform, ``3`` =
  RSP default, higher = aggressive) would let the bandit flip the
  selective-pressure regime discretely, the same way
  ``LSHADE.archive_factor`` flips archive on / off / RSP.

#### Run a measured A/B across PSO topologies (gbest / lbest / vonneumann)

Von Neumann shipped 2026-05-22 (see §13).  The literature predicts
the three topologies are *complementary* — gbest wins on unimodal
landscapes, lbest on highly-multimodal, vonneumann between the two —
but the shipped entry did not include a measured benchmark because
the impact at quick-mode budgets is within noise.  A natural
follow-up is to run an explicit ``benchmark_harness.py compare``
across the three Rewarding strategies (one per PSO topology) at
``--standard`` mode (≥ 5 reps × ~8 problems × ~300 evaluations) so
the *per-problem* per-topology winners are identified.  Use the
paired-bootstrap CI (auto-selected on ``--randomize``) so the
per-pair regressions are detected rigorously.  The output of this
benchmark feeds two follow-ups:

* If the data shows a per-problem-class winner pattern, encode it in
  the structural catalog (e.g., add a ``StrategySpec`` that pre-pairs
  ``vonneumann`` with Rastrigin / Ackley / Griewank-style problems
  via the strategy-pattern matcher).
* If no topology wins consistently across problem classes, leave the
  current uniform-over-three catalog and let the bandit's per-arm
  reward signal identify the winner online.
