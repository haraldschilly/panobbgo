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
      Strategy portfolio composition (§7.2) shipped 2026-05-04 as
      :class:`StructuralMutationRule` + :func:`default_structural_rules`;
      enable via `default_catalog(include_structural=True)` or
      `LoopConfig.structural_mutations=True`.
- [x] Persistence of the running "ladder" of best composite scores
      over time — `LadderEntry` + JSONL ledger
      (`planning/self_improve_ledger.jsonl`).  Each entry stores its
      ``last_validated_score``, refreshed every time the guard
      re-measures it.

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
   strategy, reweight initial priors.  **Shipped 2026-05-04** as
   :class:`StructuralMutationRule` (`add_heuristic` /
   `drop_heuristic`) + :func:`default_structural_rules` +
   `LoopConfig.structural_mutations` + the `--structural` CLI flag.
   Safety rail: drops are refused below `min_heuristics=1` and the
   default catalog uses `min_heuristics=2` to keep at least one
   primer + one local-search heuristic per spec.
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
- [x] Strategy portfolio composition (§7 item 2) — shipped 2026-05-04
      as :class:`panobbgo.self_improve.StructuralMutationRule`,
      :func:`default_structural_rules`, the
      ``LoopConfig.structural_mutations`` knob, and the
      ``--structural`` CLI flag.  Rules add or drop point-emitting
      heuristics from a :class:`StrategySpec`; safety rails refuse to
      empty the heuristic list.  See §7.2 below and the §12 dated
      entry.
- [ ] Analyzer add/drop (§7 item 3).
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
- [ ] Connect the loop to CI (nightly run on a dedicated runner).
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
  dim exactly once.  See the §12 entry below.
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

## 12. Iteration log

This section records direct algorithmic improvements applied to Panobbgo
*outside* of the autonomous loop, so the human-in-the-loop history stays
greppable.  Each entry should reference the PR / commit that landed it,
the rationale, and a measured-impact number when available.

### 2026-05-04 — Strategy portfolio composition (§7.2)

* **What** — `panobbgo/self_improve.py` gains
  :class:`StructuralMutationRule` (kinds ``"add_heuristic"`` and
  ``"drop_heuristic"``), the :func:`default_structural_rules` factory,
  the ``LoopConfig.structural_mutations`` knob, and the
  ``--structural`` CLI flag on `scripts/self_improve.py run`.
  :class:`MutationProposal` gains three new fields (``operation``,
  ``class_module``, ``heuristic_index``) with defaults that preserve
  byte-level backward compatibility for existing ``set_param``
  proposals.  :func:`apply_mutation` dispatches on ``operation``;
  ``add_heuristic`` resolves the inserted class via
  ``importlib.import_module(class_module)`` so proposals round-trip
  through the JSONL ledger.  :class:`AdaptiveMutationSampler` learns to
  key structural rules: ``("Center", "", "add_heuristic")`` for adds,
  ``(droppable_signature, "", "drop_heuristic")`` for drops (single-
  class drop rules ⇒ one bandit arm per class).  The default
  structural catalog ships three add rules (``Nearby`` / ``NelderMead``
  / ``LatinHypercube``) and four single-class drop rules.
  ``min_heuristics=2`` floor prevents drops from over-pruning; the
  applicator hard-fails any drop that would empty a strategy's
  heuristic list.
* **Why** — closes the §7.2 "Strategy portfolio composition" item from
  the §10 Open questions / Phase 6 checklist of this plan.
  Hyperparameter retunes — the only mutation class until today — only
  twiddle existing knobs; structural mutations let the loop discover
  *new* portfolio compositions, which the plan flags as the
  next-most-impactful mutation class.  This is the path from
  "Panobbgo is mostly static, with the loop tuning ε" to "the loop
  reshapes Panobbgo".
* **Defaults** — ``LoopConfig.structural_mutations = False`` keeps
  existing CLI invocations byte-identical on a fresh ledger.  Flip to
  ``True`` (or pass ``--structural`` on the CLI) when you want the
  loop to broaden its reach.  An explicit ``catalog=`` kwarg to
  :class:`SelfImprover` always wins, so tests and advanced users keep
  full control.
* **Safety rails** — drops respect ``min_heuristics`` (default ``1``;
  default catalog uses ``2``), the applicator hard-fails any drop that
  would empty the heuristic list, and adds gate on
  ``skip_if_class_present=True`` so duplicate copies of the same
  class don't accumulate.  All rule-validation errors raise at
  :class:`StructuralMutationRule` construction time so misconfigured
  catalogs fail fast.
* **Tests** — `tests/test_self_improve.py` (46 new tests, total 109).
  Coverage: rule validation (kind, ``heuristic_class``,
  ``min_heuristics``, probability), catalog applicable-rule filtering
  (``skip_if_class_present``, ``min_heuristics``, ``strategy_pattern``,
  empty droppable_classes), proposal sampling (add + drop schemas,
  ``last_sampled_params``-style fields), :func:`apply_mutation` for
  add and drop including index drift fallback, last-heuristic refusal,
  unknown-strategy / unknown-class error paths, dispatch on
  ``operation``, default catalog content, adaptive-sampler keying
  for add / drop / unrestricted-drop / multi-class-drop arms, an
  end-to-end run that accepts a structural mutation and round-trips
  the proposal through the JSONL ledger, and ``LoopConfig`` flag
  precedence (explicit catalog wins, default-on widens, default-off
  preserves backward compat).
* **Backwards compatibility** — strictly safe.  Existing ledger lines
  parse unchanged because :class:`MutationProposal` defaults
  ``operation`` to ``"set_param"``.  ``default_catalog()`` (no
  argument) is byte-identical to the prior 6-rule catalog.  All
  prior tests pass without modification (63 → 109 by addition).

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

#### Analyzer add/drop (§7.3)

The next structural-mutation class once strategy portfolio composition
(shipped 2026-05-04) has bedded in.  Adding / dropping
:class:`Sensitivity` or :class:`Restart` from a strategy reshapes the
event stream the heuristics react to, which can be a much larger lever
than the analyzer's own kwargs.  Needs an analogous
``StructuralAnalyzerRule`` (or a single rule type with a
``"target": "heuristic" | "analyzer"`` discriminator) plus a safety
check that the four mandatory analyzers (Best, Grid, Splitter,
Convergence) cannot be touched.

#### Hold-out validation set

Maintain a small fixed validation set of randomized instances drawn
from a separate `base_seed`.  Use it (read-only) to spot-check the
ladder once at the end of a loop run.  Cheaper than the periodic
guard but complements it.
