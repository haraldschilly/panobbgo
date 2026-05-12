# TODO

## Recent Improvements (continued)

### COBYQA Derivative-Free Trust-Region Local Optimizer (2026-05-12)
- [x] **`panobbgo/heuristics/cobyqa.py`** — new :class:`COBYQA`
      heuristic, a subprocess-backed adapter around
      ``scipy.optimize.minimize(method="COBYQA")``.  COBYQA
      (*Constrained Optimization BY Quadratic Approximations*,
      Ragonneau-Zhang 2023) is the modern Powell-family successor to
      BOBYQA / COBYLA / NEWUOA / LINCOA: it maintains an interpolation
      set of ``2·n + 1`` points and fits an adaptive *quadratic
      model* of the objective inside a trust region, dominant on
      smooth / near-smooth local refinement.  The asynchronous
      wrapping pattern mirrors :class:`~panobbgo.heuristics.lbfgsb.LBFGSB`:
      a daemon ``spawn`` subprocess drives the synchronous COBYQA
      solver, requests ``f(x)`` over a pipe, and the main thread
      relays the projected point through Panobbgo's evaluator and
      pipes the penalty value back.
  - **Why it matters.**  Before this entry,
    :class:`~panobbgo.heuristics.nelder_mead.NelderMead` was the
    *only* generic derivative-free local refinement step in the
    portfolio; :class:`~panobbgo.heuristics.lbfgsb.LBFGSB` needs a
    finite-difference gradient approximation that breaks on noisy
    objectives, and Nelder-Mead's simplex updates are not
    curvature-aware, so it converges slowly on ill-conditioned
    valleys (Rosenbrock-like landscapes).  COBYQA gives the loop
    driver a *derivative-free **and** curvature-aware* local
    refinement arm the bandit can choose between.  Picking COBYQA
    over the older BOBYQA library keeps the dependency surface
    unchanged — COBYQA ships built-in with ``scipy.optimize.minimize``
    since scipy 1.14 and is the literature-recommended replacement.
- [x] **Configuration knobs** — ``initial_tr_radius`` (auto-derives
      to ``0.1 · max(box_width)`` when ``None``), ``final_tr_radius``
      (default ``1e-6``), ``maxfev`` (``None`` lets the strategy
      budget terminate), ``scale`` (default ``True`` — maps the box
      to ``[-1, 1]`` to keep the interpolation geometry
      well-conditioned for boxes whose axes span very different
      magnitudes).  Construction-time validation rejects negative /
      zero / NaN radii, the ``final >= initial`` ordering, and
      non-integer / non-positive ``maxfev``.
- [x] **Restart support** — :meth:`COBYQA.on_restart(center, reason)`
      tears down the current subprocess (terminate → join → kill on
      timeout) and respawns a fresh COBYQA solve seeded at the
      *clipped* suggested center.  When the strategy is stopped the
      restart is a no-op so the loop never spawns a process during
      shutdown.
- [x] **Structural catalog wiring** — :func:`default_structural_catalog`
      gains COBYQA as an eleventh ``add_heuristic`` candidate
      (``avoid_duplicates=True`` keeps the catalog from cluttering
      portfolios that already include it).
- [x] **Kwarg mutation rules** — :func:`default_catalog` gains two
      rules so the loop driver can also retune
      ``COBYQA.initial_tr_radius`` (``log_uniform_perturb`` over
      ``[0.01, 1.0]``, ``log_step=0.15``) and
      ``COBYQA.final_tr_radius`` (``log_uniform_perturb`` over
      ``[1e-8, 1e-4]``, ``log_step=0.25``).  Both fire only when a
      spec explicitly sets the matching kwarg.
- [x] **Backwards compatibility — strictly safe.**  COBYQA is opt-in:
      it is not added to any default
      :func:`_make_quick_strategies` / :func:`_make_standard_strategies` /
      :func:`_make_full_strategies` spec.  Existing CLI invocations
      and existing ledgers stay byte-identical.
- [x] **Quick A/B impact at ``--quick`` (3 problems × 3 reps × 75
      evaluations)**, comparing the same Rewarding strategy with
      NelderMead vs COBYQA vs both as the local optimizer:
  - Seed 42 — ``NM`` 0.665 / ``COBYQA`` **0.769** (+0.104) /
    ``NM+COBYQA`` 0.699.  Rosenbrock success rate jumps from
    **0/3 with NM** to **2/3 with COBYQA**.
  - Seed 43 — ``NM`` **0.864** / ``COBYQA`` 0.714 / ``NM+COBYQA``
    0.753.  NM happens to win Rosenbrock on this seed.
  - The two seeds together demonstrate complementarity — each
    local optimizer wins on one of them; the categorical
    Rosenbrock success-rate upgrade (0/3 → 2/3) confirms the
    expected property: COBYQA's curvature-aware quadratic model
    crosses the narrow curved valley that Nelder-Mead misses.
- [x] **30 tests in `tests/test_heuristic_cobyqa.py`** —
      construction validation (11 — invalid initial / final TR
      radii / NaN / ordering rule / maxfev type and value), initial
      TR auto-resolution (4), subprocess lifecycle (2 — spawn,
      force-kill), pipe wiring (4 — penalty routed, foreign-who
      ignored, EOF exit, output log), restart behaviour (4 —
      respawn, ``center=None`` fallback, out-of-box clip, stopped
      no-op), registration (3 — package, structural catalog,
      kwarg rules), end-to-end smoke directly through scipy.
- [x] **Documentation updated**
  - `planning/SELF_IMPROVEMENT_LOOP.md`: new §12 dated entry; the
    *BOBYQA / NEWUOA local optimizer* "Next iteration idea" was
    closed and replaced with COBYQA follow-up tickets
    (constraint-aware variant, warm-start interpolation reuse,
    categorical mutation rule for ``scale``).
  - `doc/source/guide.rst`: quick-nav entry mentions COBYQA.
  - `doc/source/guide_benchmarking.rst`: structural catalog
    candidate-pool list extended with ``LSHADE`` and ``COBYQA``.

### Hold-Out Validation Set for Self-Improvement Loop (2026-05-08)
- [x] **New `panobbgo.self_improve.LoopHoldoutRecord`** — third record
      type alongside `LoopIterationRecord` / `LoopGuardRecord`; closes
      the §10 "Hold-out validation set" item in
      `planning/SELF_IMPROVEMENT_LOOP.md`.  At the end of every loop
      run, the seed and final-top of the accepted ladder are
      re-measured on instances drawn from an *independent*
      `base_seed` SHA-256 stream, and the shrinking-gap drift is
      reported on the JSONL ledger as `record_type="holdout"`.
  - **Why it matters.**  The anti-cherry-pick guard catches drift
    *within* the training base_seed family — it varies only
    `randomize_iteration` and keeps `HarnessConfig.seed` constant.  A
    mutation that overfits to peculiarities of the training base_seed
    family slips through silently because the guard's "fresh"
    instances are still drawn from the same SHA-256 stream.  The
    hold-out re-measures on a *completely* independent base_seed, so
    an overfit ladder is exposed by ``drift < -eps_overfit``.
- [x] **`LoopConfig` knobs** — `holdout_base_seed` (0 = disabled),
      `holdout_iterations` (5 by default), `holdout_iteration_offset`
      (0 by default), `holdout_eps_overfit` (0.05 by default).
      `LoopConfig.__post_init__` rejects `holdout_base_seed ==
      base_seed` (other than 0) — equal values would collapse the
      check to a glorified guard with offset 0.
- [x] **`LoopConfig.holdout_harness_config()`** — sibling to
      `harness_config()` that swaps the `seed` field to
      `holdout_base_seed`; every other knob (mode, reps, budget,
      `strategies_override`) matches the training run.
- [x] **`SelfImprover._run_holdout()`** + helpers
      (`_holdout_enabled`, `_measure_holdout`, `_print_holdout`).
      Skipped silently when (a) disabled, (b) the loop ran zero
      iterations, or (c) `randomize=False` (the fixed battery is
      unaffected by base_seed, so a hold-out check would be no
      signal).
- [x] **New public entry-point `SelfImprover.run_full()`** — returns
      `(iter_records, guard_records, holdout_records)` for callers
      that want all three signals.  `SelfImprover.run()` keeps its
      original return type for backward compatibility;
      `run_with_guard_records()` returns just the first two.
- [x] **CLI flags** `--holdout-base-seed`, `--holdout-iterations`,
      `--holdout-iteration-offset`, `--holdout-eps-overfit`,
      `--fail-on-overfit` (exits `3` on a flagged ladder) on
      `scripts/self_improve.py run`; `summary` distinguishes
      hold-out records and prints drift / overfit verdict.
- [x] **17 new tests in `tests/test_self_improve.py`** (total 97):
      - **`TestLoopConfigHoldout`** — defaults, negative-iterations
        validation, negative-eps validation, equal-base-seed
        rejection, zero-zero edge case, `holdout_harness_config`
        propagation.
      - **`TestSelfImproverHoldout`** — disabled-by-default,
        skipped when `randomize=False`, skipped on zero iterations,
        seed-only ladder records zero drift, hold-out uses the
        independent base_seed for measurement, overfit flag fires
        when gap collapses, no flag when gap holds, ledger writes
        `record_type='holdout'` line, `run()` keeps backward
        compatibility.
      - **`TestLoopHoldoutRecord`** — `to_dict` round-trip with JSON
        serialisation.
- [x] **Documentation updated**
  - `planning/SELF_IMPROVEMENT_LOOP.md`: §2 missing-pieces list
    refreshed; §10 hold-out item resolved; Phase 6 checklist updated;
    new §12 dated entry; "Next iteration ideas" replaced with
    multi-seed hold-out / auto-rollback follow-up tickets.
  - `doc/source/guide_benchmarking.rst`: new "Hold-out validation
    set" subsection with algorithm, CLI examples, programmatic
    example, and the independence-from-the-guard note.
  - `doc/source/guide.rst`: quick-nav entry mentions the hold-out.
  - `AGENTS.md`: self-improvement loop subsection lists the
    hold-out feature with run-the-loop bash example.

### PSO adaptive inertia (Shi-Eberhart 1998) (2026-05-07)
- [x] **`panobbgo/heuristics/pso.py`** — :class:`PSO` gains an opt-in
      ``w_end`` keyword argument.  When set, a new
      ``_current_inertia()`` method linearly anneals the inertia from
      ``self.w`` down to ``self.w_end`` paced by
      ``len(strategy.results) / strategy.config.max_eval``; otherwise
      the inertia is constant at ``self.w``, reproducing the prior
      Clerc-Kennedy behaviour byte-for-byte.  Falls back to constant
      ``w`` whenever the budget is unknown, zero, or non-numeric.
- [x] **Catalog rules** — :func:`default_catalog` gains
      ``MutationRule`` entries for ``PSO.w`` (``float_uniform`` over
      ``[0.4, 0.95]``) and ``PSO.w_end`` (``float_uniform`` over
      ``[0.2, 0.6]``) so the loop driver can tune the inertia
      schedule once a spec opts in.
- [x] **6 new tests in `tests/test_heuristic_pso.py`**: default
      ``w_end`` is ``None``; finiteness validation; constant-``w``
      short-circuit; missing-results fall-back; linearly-decreasing
      schedule at four progress points; ``max_eval = 0`` fall-back;
      and a catalog test asserting ``PSO.w`` / ``PSO.w_end`` rules
      are present.
- [x] **Documentation updated** — module docstring gets a new
      "Adaptive inertia" section and a Shi-Eberhart reference;
      ``doc/source/heuristics.rst`` and
      ``doc/source/guide_architecture.rst`` mention ``w_end``.

### PSO ring (`lbest`) topology variant (2026-05-07)
- [x] **`panobbgo/heuristics/pso.py`** — :class:`PSO` gains a
      ``topology: str = "gbest"`` argument and a ``k_neighbors: int = 2``
      half-width.  ``"gbest"`` keeps the canonical Kennedy-Eberhart
      1995 fully-connected swarm (default, byte-identical to the
      2026-05-05 ship); ``"lbest"`` switches every particle's social
      attractor to the best ``pbest`` in a wrap-around *ring* of width
      ``2·k_neighbors + 1`` centred on the particle's index — slower
      information diffusion, stronger multimodal exploration
      (Kennedy & Mendes, CEC 2002).
- [x] **Two new helpers** — ``_ring_neighbors(i)`` returns the
      wrap-around index list; ``_social_best_idx(i)`` returns the
      per-particle social attractor (collapsing to ``_gbest_idx`` for
      gbest, scanning the ring for lbest).  ``_generate_next``
      consults ``_social_best_idx`` exactly where it used
      ``_gbest_idx`` before, so the velocity-update / clamp /
      projection paths are shared between topologies.
- [x] **Structural catalog integration** —
      :func:`default_structural_catalog` ships two PSO entries:
      ``(PSO, {"NP": 20})`` (gbest, default) and
      ``(PSO, {"NP": 20, "topology": "lbest", "k_neighbors": 2})``.
      Both share ``cls = PSO`` so ``avoid_duplicates=True`` still
      installs only one PSO per strategy; the catalog samples
      uniformly between the two when PSO is not yet present and
      skips both afterwards.
- [x] **Backwards compatibility** — ``topology`` defaults to
      ``"gbest"`` so existing PSO instances retain their prior
      behaviour bit-for-bit.  Existing kwarg rule
      (``MutationRule(class_name="PSO", param_name="NP", …)``) and
      bandit ``_proposal_rule_key`` are unchanged.
- [x] **A/B at `--quick`** (3 problems × 5 reps × 150 evals): seed
      42 → gbest 0.183 / **lbest 0.288**; seed 43 → **gbest 0.296** /
      lbest 0.181.  The two topologies are *complementary* (each
      wins on one seed), exactly the literature's prediction —
      shipping both gives the bandit a finer-grained choice without
      regressing the gbest path.
- [x] **13 new tests in `tests/test_heuristic_pso.py`** (total 50):
      construction validation (default topology / lbest construction
      / invalid topology / invalid k_neighbors type / value), ring
      wrap-around correctness, ring size invariant, lbest social
      attractor uses ring (not the global best), gbest social
      attractor degenerates to ``_gbest_idx``, lbest returns ``None``
      before any neighbour pbest exists, lbest velocity clamp
      invariant, lbest end-to-end smoke convergence on a quadratic,
      and the structural catalog now ships both gbest and lbest PSO
      entries.
- [x] **Documentation updated** — ``planning/SELF_IMPROVEMENT_LOOP.md``
      §12 logs the iteration and the PSO follow-ups list now drops
      "Topology variants" (shipped) and adds "Random / Von Neumann
      topologies" + "Categorical / topology mutation rule" as next
      ideas.  ``doc/source/heuristics.rst``,
      ``doc/source/guide_architecture.rst``,
      ``doc/source/guide_research.rst``, and
      ``doc/source/guide_benchmarking.rst`` describe both topologies.

### PSO (Particle Swarm Optimization) heuristic (2026-05-05)
- [x] **`panobbgo/heuristics/pso.py`** — new asynchronous PSO heuristic
      with the canonical Clerc–Kennedy (2002) constriction-coefficient
      parameters (``w = χ ≈ 0.7298``, ``c1 = c2 ≈ 1.49618``).  Each
      particle carries a position, velocity, and personal best; the
      velocity update pulls toward both personal best and global best
      with random per-component weights and is clamped per-dimension to
      a configurable fraction of the box range to prevent explosion.
- [x] **Async event-loop integration** — follows the same pattern as
      :class:`DifferentialEvolution`: each in-flight trial carries a
      unique ``who`` id; ``on_new_results`` matches the id back to its
      particle slot, updates pbest/gbest, and emits the next velocity-
      based trial.  No per-tick busy waiting.
- [x] **IPOP-style warm restart** — ``on_restart(center, reason)``
      drops in-flight trials, scatters particles in a velocity-clamp
      ball around the new center, and resets the global memory while
      the strategy keeps its accumulated history.
- [x] **Catalog integration** — added a kwarg rule in
      :func:`default_catalog` for ``PSO.NP`` (swarm size, range
      ``[8, 60]``, ±4 / ±8 deltas) and added ``PSO`` to the
      ``add_heuristic`` candidate pool in
      :func:`default_structural_catalog`.
- [x] **24 new tests in `tests/test_heuristic_pso.py`** — construction
      validation (8), initial-swarm emission and shape (3), pbest /
      gbest update + follow-up trial (5), velocity clamp invariant (1),
      restart behaviour (3), an end-to-end smoke run on a quadratic, and
      registration tests for ``panobbgo.heuristics`` and the structural
      catalog.
- [x] **Documentation updated** — ``doc/source/heuristics.rst`` and
      ``doc/source/guide_architecture.rst`` (Population-based section)
      now describe PSO; ``doc/source/guide_benchmarking.rst`` mentions
      it among the structural-catalog candidates;
      ``planning/SELF_IMPROVEMENT_LOOP.md`` §12 logs the iteration.

### Strategy Portfolio Composition (`StructuralMutationRule`) (2026-05-03)
- [x] **`panobbgo.self_improve.StructuralMutationRule`** — new dataclass
      that joins :class:`MutationRule` as a first-class catalog entry.
      Closes the §7.2 *Strategy portfolio composition* item in
      `planning/SELF_IMPROVEMENT_LOOP.md` — the loop driver could
      previously only retune existing kwargs; now it can also reshape
      a strategy's heuristics list.
- [x] **Two ops** — ``add_heuristic`` (append a class from a curated
      ``candidate_classes`` pool, ``avoid_duplicates`` by default
      skips classes already present) and ``drop_heuristic`` (remove
      one heuristic, optionally restricted via ``droppable_classes``
      with a ``min_heuristics`` post-drop safety floor — default ``2``).
- [x] **`MutationProposal` extension** — keyword-only ``op`` and
      ``structural_kwargs`` fields, default ``None``.  Kwarg proposals
      serialise byte-identically to before; structural proposals get
      the two extra keys via :meth:`to_dict`.
- [x] **`apply_mutation` dispatch** — branches on ``proposal.op``;
      ``add_heuristic`` recovers the class object via the spec's
      existing classes first and ``panobbgo.heuristics`` package as
      fallback; ``drop_heuristic`` removes the first match and refuses
      to leave the spec empty.
- [x] **Adaptive sampler integration** — ``_proposal_rule_key``
      collapses both structural ops onto a single arm
      (``("*", op, "structural")``) so cold-start variance stays
      bounded.  Per-class arms are listed as a follow-up under "Next
      iteration ideas".
- [x] **`default_structural_catalog()`** — extends
      :func:`default_catalog` with one ``add_heuristic`` rule (pool of
      seven safe generators: Random/Nearby/NelderMead/Center/
      LatinHypercube/Sobol/Extremal) and one ``drop_heuristic`` rule
      (``min_heuristics=2``).  Both at probability ``0.3`` so the
      structural rules don't dominate kwarg perturbations.
- [x] **CLI flag** — `scripts/self_improve.py run --structural`
      switches the loop to the structural catalog.  Off by default so
      existing CLI invocations are byte-identical.
- [x] **29 new tests in `tests/test_self_improve.py`** (total 92):
      rule validation, applicable-hits enumeration (add / drop /
      ``avoid_duplicates`` / ``droppable_classes`` / ``min_heuristics``
      floor / strategy_pattern filter), proposal serialisation, the
      apply-side dispatch (add appends, drop removes, missing class
      raises, empty-strategy refusal, fallback-import path),
      :func:`_proposal_rule_key` collapse for structural ops, the
      Thompson sampler bucketing structural history into one arm, the
      `default_structural_catalog()` factory shape, and an end-to-end
      loop run that accepts a structural drop on a fake harness.
- [x] **Documentation updated**
  - `planning/SELF_IMPROVEMENT_LOOP.md`: Phase 6 checklist marks §7.2
    done; §7 item 2 annotated as shipped; "Next iteration ideas"
    rewritten with per-class-arms, analyzer add/drop, and
    strategy-class-swap follow-ups; new §12 dated entry under the
    iteration log.
  - `doc/source/guide_benchmarking.rst`: new "Strategy portfolio
    composition (§7.2)" subsection covering the ops, safety floors,
    Thompson-sampler bucketing, CLI invocation, and a programmatic
    custom-catalog example.
  - `doc/source/guide.rst`: top-of-page navigation row for the
    benchmarking guide now lists the structural mutations.
  - `AGENTS.md`: structural catalog mentioned in the loop checklist
    and the CLI example block.
  - This TODO entry.

### Stratified Dimension Sampling for Multi-Dim Families (2026-05-02)
- [x] **`panobbgo.harness_randomized.ProblemFamily.stratify_dims`** —
      new bool field (default ``True``) that enables cyclic dim
      stratification for multi-dim families.  Closes the §10 "Composite
      score stability across dimension sampling" open question in
      `planning/SELF_IMPROVEMENT_LOOP.md`.
- [x] **`ProblemFamily.stratified_dim_for_rep(rep)`** — returns
      ``dim_choices[rep % len(dim_choices)]``, so any contiguous block
      of ``k`` reps covers every declared dim exactly once.
- [x] **`ProblemFamily.sample_instance(rng, dim=None)`** — now accepts
      an optional dim override.  Stratified callers pin the dim
      explicitly, so the rng's ``choice`` slot is not consumed and the
      remaining stream (translation, rotation, scaling, noise seed)
      stays comparable to a single-dim family at the same seed.
- [x] **`RandomizedProblemSpec.create_problem_for_rep(rep)`** — calls
      ``stratified_dim_for_rep(rep)`` for multi-dim families with
      ``stratify_dims=True`` and falls back to the rng draw otherwise.
      ``last_sampled_params()`` now exposes a ``stratified_dim: bool``
      flag for ledger introspection.
- [x] **Why it matters.** Without stratification, a family with
      ``dim_choices = (2, 5, 10)`` and 5 reps could draw three ``dim=2``
      instances on iteration 5 and three ``dim=10`` instances on
      iteration 6.  Higher-dim instances are systematically harder, so
      a per-iteration composite delta picks up dim-mix noise on top of
      the actual signal of the underlying mutation, polluting the
      bootstrap CI in :func:`panobbgo.harness.statistical_accept`.
      Cyclic stratification eliminates that noise source by construction
      without changing the per-iteration eval count.
- [x] **Backwards compatibility.** The entire default battery from
      :func:`make_default_families` uses ``dim_choices=(2,)``, so
      stratification is a no-op for byte-level reproducibility of the
      current standard mode.  ``stratify_dims=False`` recovers the
      legacy uniform-draw behaviour for users replicating an old ledger.
- [x] **16 new tests in `tests/test_harness_randomized.py`** (total 68):
      cyclic schedule correctness, balance over a complete cycle,
      imbalance bound on partial cycles, single-dim no-op, dim-override
      validation, rng-stream invariance proof (override does not consume
      the choice slot), end-to-end :class:`RandomizedProblemSpec` round
      trip, ``last_sampled_params`` flag round trip, and the contract
      that default families remain unchanged.
- [x] **Documentation updated**
  - `planning/SELF_IMPROVEMENT_LOOP.md`: §10 stability item resolved;
    Phase 6 checklist updated; new §12 dated entry under iteration log;
    stratification clause added to the §4.1 transforms table; Next
    iteration ideas now lists a "multi-dim default battery" follow-up.
  - `doc/source/guide_benchmarking.rst`: new "Stratified dimension
    sampling" subsection with the cyclic schedule example and the
    ``stratify_dims=False`` escape hatch.
  - `AGENTS.md`: stratification clause in the parametric battery
    section.
  - This TODO entry.

### Adaptive Mutation Sampler (Thompson Sampling) for Self-Improvement Loop (2026-05-01)
- [x] **New `panobbgo.self_improve.AdaptiveMutationSampler`** — Thompson-
      sampling bandit over per-rule Beta posteriors; closes the §10
      "Adaptive mutation sampler" item in
      `planning/SELF_IMPROVEMENT_LOOP.md`.  Each
      :class:`MutationRule` becomes one arm whose reward is "iteration was
      accepted"; on `sample()` the sampler draws one variate from
      ``Beta(prior_alpha + n_accepts, prior_beta + n_attempts -
      n_accepts)`` per applicable rule and picks the arg-max.  Inside the
      chosen rule, hits are still selected uniformly (which spec / which
      slot), exactly like the catalog's uniform sampler.
  - **Why it matters.** The uniform catalog sampler shipped in Phase 5
    wastes iterations on rules that never produce accepts.  Thompson
    sampling concentrates probability on empirically winning rules
    while still exploring under-tried rules — the standard fix for the
    productivity gap of multi-armed bandit problems.  Cold-start
    equivalence to uniform (Beta(1, 1) ≡ U(0, 1), arg-max of i.i.d.
    uniforms is uniform) makes the upgrade strictly safe.
  - **History persistence.** `prime_from_ledger(path)` replays
    iteration records from a prior JSONL ledger so the bandit resumes
    with all the meta-knowledge of which rules have worked so far —
    directly supports unattended multi-hour loops.
- [x] **`MutationRuleStats` dataclass + public `RuleKey` alias** —
      JSON-serialisable per-rule accept/attempt history bucketed by
      ``(class_name, param_name, rule_kind)``.
- [x] **`LoopConfig` knobs** — `adaptive_sampling`,
      `adaptive_prior_alpha`, `adaptive_prior_beta`,
      `adaptive_prime_from_ledger`; all default to off / symmetric prior
      so existing CLI invocations behave identically.  Negative or zero
      priors raise at validation time.
- [x] **`SelfImprover` integration** — accepts an explicit `sampler=`
      keyword for tests; otherwise constructs the sampler from
      `LoopConfig` when `adaptive_sampling=True`.  After each iteration's
      accept/reject decision, the driver calls
      ``sampler.record_outcome()`` so future samples are biased toward
      winning rules.
- [x] **CLI flags** `--adaptive`, `--adaptive-prior-alpha`,
      `--adaptive-prior-beta`, `--adaptive-prime-from-ledger` on
      `scripts/self_improve.py run`; the run summary prints per-rule
      accept rates when the sampler is enabled.
- [x] **23 new tests in `tests/test_self_improve.py`** (total 63):
      invalid priors, cold-start uniform behaviour, arg-max bias toward
      winning rules after biased training, record-outcome correctness
      including no-op after `None` sample / skip iterations, ledger
      priming (with guards / skips correctly ignored), `MutationRuleStats`
      round-trip, `SelfImprover` integration with the `sampler=`
      override, the `adaptive_prime_from_ledger` flag, and
      `LoopConfig` validation.
- [x] **Documentation updated**
  - `planning/SELF_IMPROVEMENT_LOOP.md`: §6 Phase-6 checklist marked
    shipped; new §12 dated entry under iteration log;
    "Next iteration ideas" reduced and gains a hierarchical-bandit
    follow-up ticket.
  - `doc/source/guide_benchmarking.rst`: new "Adaptive mutation sampler
    (§10)" subsection with algorithm, CLI examples, programmatic
    example, cold-start equivalence proof sketch.
  - `AGENTS.md`: self-improvement loop subsection lists the adaptive
    sampler with run-the-loop bash example.
  - This TODO entry.

### Sobol' Quasi-Random Initial Design Heuristic (2026-04-27)
- [x] **New `panobbgo/heuristics/sobol.py`** — `Sobol` heuristic, a one-shot
      low-discrepancy quasi-random sampler that produces space-filling initial
      designs.
  - Backed by `scipy.stats.qmc.Sobol` (no new dependency).
  - Owen-scrambled by default — different seeds produce statistically
    independent point sets so per-rep variance is meaningful, while the
    low-discrepancy property is preserved within each draw.
  - Uses ``random_base2`` when ``n`` is a power of two for the sharpest
    balance properties; falls back to ``random(n)`` otherwise.
  - Pure standalone heuristic following the `LatinHypercube` pattern; no
    event-system hooks needed.
- [x] **`BayesOpt_Sobol` strategy** added to standard harness mode pairing
      ``Sobol(n=16, scramble=True)`` with ``GaussianProcessHeuristic``,
      ``Nearby``, ``NelderMead`` — head-to-head with the existing
      ``BayesOpt_GP`` (which uses ``LatinHypercube``).
- [x] **Mutation rule for Sobol.n** added to the self-improvement loop's
      ``default_catalog()`` (4-step increments inside ``[4, 64]``) so the
      loop driver can also tune the parameter.
- [x] **Measured impact** (standard mode, 5 reps × 7 problems, budget 200):
      mean per-pair score ``BayesOpt_Sobol = 0.314`` vs
      ``BayesOpt_GP = 0.191`` (``+0.123``); wins on 5 / 7 problems, ties on
      Griewank with smaller best-distance.
- [x] **16 tests in `tests/test_heuristic_sobol.py`** — construction
      validation, scaling/sampling primitives, low-discrepancy proxy vs
      uniform sampling, scramble-determinism vs seed-reproducibility,
      ``on_start`` emit path, higher-dimensional problems, registration
      check.
- [x] **Documentation updated**
  - `doc/source/heuristics.rst`: ``Sobol`` listed alongside ``LatinHypercube``.
  - `doc/source/guide_architecture.rst`: Sobol added to the "Space-filling"
    heuristic group.
  - `doc/source/guide_usage.rst`: portfolio table now mentions Sobol; new
    "Bayesian optimization with Sobol' initial design" worked example.
  - `doc/source/guide_benchmarking.rst`: standard-mode strategy count
    bumped from 6 to 7 in the modes table.
  - `planning/SELF_IMPROVEMENT_LOOP.md`: §12 iteration log entry.
  - `AGENTS.md`: heuristics list updated.
  - This TODO entry.

## Setup & Modernization (Completed)
- [x] Restructure repository: Move `panobbgo.lib` to `panobbgo/lib`.
- [x] Modernize `setup.py` / Create `pyproject.toml`.
- [x] Update dependencies in `requirements.txt`.
- [x] Replace `nose` with `pytest`.
- [x] Update imports after restructuring.
- [x] Run and fix existing tests.
- [x] Add type hinting where possible.
- [x] Update `README.md` with new installation and usage instructions.
- [x] Setup CI/CD (GitHub Actions) - *optional but recommended*.

## Recent Improvements

### Anti-Cherry-Pick Guard for Self-Improvement Loop (Phase 6.3) (2026-04-26)
- [x] **New `LoopConfig.guard_interval` / `guard_eps_ladder` /
      `guard_iteration_offset`** in `panobbgo/self_improve.py` —
      implements §6.3 of `planning/SELF_IMPROVEMENT_LOOP.md`.  Every
      ``guard_interval`` iterations the loop re-measures the top of
      the accepted ladder on a *fresh* randomized seed and rolls back
      if the composite drifts more than ``guard_eps_ladder`` below the
      stored ``last_validated_score``.  The seed entry is the trusted
      fallback and is never popped.
  - **Why this matters.**  Even with the parametrically randomized
    battery, a sequence of "lucky" instance draws can inflate
    per-iteration ``after`` scores enough to clear the bootstrap CI.
    The guard catches this drift by validating the ladder against an
    independent instance stream (``randomize_iteration = iteration +
    guard_iteration_offset``).
  - **Disabled by default** (``guard_interval = 0``) for backward
    compatibility; bump to ``5`` or ``10`` for unattended runs.
- [x] **New `LadderEntry` and `LoopGuardRecord` types** —
      `LadderEntry` snapshots ``(iteration, specs,
      last_validated_score, proposal)``; `LoopGuardRecord` records the
      outcome of one guard check and is written to the same JSONL
      ledger with ``record_type = "guard"``.  `LoopIterationRecord`
      gains ``record_type = "iteration"`` for symmetry.
- [x] **CLI flags** `--guard-interval`, `--guard-eps-ladder`,
      `--guard-iteration-offset` on `scripts/self_improve.py run`;
      `summary` now distinguishes iteration and guard records and
      prints rollback details.
- [x] **40 tests in `tests/test_self_improve.py`** — comprehensive
      coverage of `MutationRule` validation, `MutationCatalog` sampling
      (log-uniform / integer-add / float-uniform), `apply_mutation`
      immutability, `LoopConfig` validation, end-to-end
      `SelfImprover` runs with a faked harness (zero iterations, skip,
      accept, reject, STOP sentinel), the new guard
      (cadence, no-rollback when stable, rollback on drift, offset
      iteration id, seed not popped), ledger round-trip, and dataclass
      serialisation.  Phase 5 shipped without tests; this PR fills
      that gap as well.
- [x] **Documentation updated**
  - `planning/SELF_IMPROVEMENT_LOOP.md`: §6.3 marked shipped, §2
    "what's missing" list updated, Phase 5 / Phase 6 checklists
    refreshed, new §12 "Next iteration ideas" with Adaptive Mutation
    Sampler, Stratified Dimension Sampling, Strategy Portfolio
    Composition, and Hold-out Validation Set as carry-over tickets.
  - `doc/source/guide_benchmarking.rst`: new "Anti-cherry-pick guard
    (§6.3)" subsection with algorithm, programmatic example, and
    safety-rail rationale.
  - `doc/source/guide.rst`: quick-nav entry mentions the loop driver
    and guard.
  - `AGENTS.md`: self-improvement loop subsection now lists the loop
    driver (shipped) and the guard (shipped) with run-the-loop bash
    examples.
  - This TODO entry.

### Parametrically Randomized Problem Battery (Self-Improvement Loop Phase 3) (2026-04-22)
- [x] **New `panobbgo/harness_randomized.py`** — Phase 3 of the
      self-improvement loop: the fixed harness battery is replaced with a
      parametric one that samples fresh transformed instances per rep,
      turning `composite_score` into a Monte-Carlo estimate of *expected*
      performance on a problem family.  Without this, an autonomous
      improvement loop would over-fit to specific instances.
  - `TransformedProblem(Problem)` — wraps a base problem with the
    composition `y = Q · Λ · (x - x*) + y_base_star` plus optional
    additive Gaussian noise; by construction `f_new(x*) = f_opt` so the
    existing harness metrics (`func_distance`, `ert`, `composite_score`)
    work unchanged.
  - `ProblemFamily` — declarative spec with per-family
    `supported_transforms` capability flags (`translate`, `rotate`,
    `scale`, `noise`), `log10_cond_max`, `dim_choices`, and tolerance.
  - `RandomizedProblemSpec(ProblemSpec)` — bridge between the family and
    the harness; `create_problem_for_rep(rep)` samples a fresh instance
    from the family, and records the sampled parameters for ledger
    output via `last_sampled_params()`.
  - Haar-uniform orthogonal sampler via QR + Mezzadri sign correction
    (dependency-free).
  - Geometric log-uniform diagonal scaling with configurable condition
    ceiling.
  - Interior-point translation (default 15% per-side margin) so the
    optimum never sits on a box boundary.
  - SHA-256-derived instance seed via
    `derive_instance_seed(base_seed, iteration_id, family_name, rep)`
    — within one iteration `before`/`after` runs see identical
    instances; across iterations they intentionally differ.
- [x] **Default families**: `Rastrigin_family`, `Ackley_family`,
      `Rosenbrock_family`, `DeJong_family`.  Schwefel and Griewank are
      intentionally excluded — rotation would push `y` off their
      sensible domain.
- [x] **`HarnessConfig.randomize` + `HarnessConfig.randomize_iteration`**
      plus `BenchmarkHarness.get_problems()` / `_run_single()` plumbing.
- [x] **CLI flags** `--randomize` and `--randomize-iteration N` on
      `benchmark_harness.py run` / `list`.
- [x] **52 tests in `tests/test_harness_randomized.py`** covering
      sampling primitives, transform invariants (optimum preservation,
      orthogonality, condition-number bounds, noise variance), family
      capability gating, and the before/after reproducibility contract.
- [x] **Documentation updated**
  - `doc/source/guide_benchmarking.rst`: replaced the "planned" section
    with a full shipping section (usage, transform math, default
    families, reproducibility recipe).
  - `doc/source/guide.rst`: quick-nav entry mentions parametric
    randomization.
  - `AGENTS.md`: new "Parametrically randomized problems" subsection and
    key-files list updated.
  - `planning/SELF_IMPROVEMENT_LOOP.md`: Phase 3 checklist flipped to
    shipped; "what's missing" list updated.
  - This TODO entry.
- [ ] **Next in the roadmap** — the loop driver `scripts/self_improve.py`
      (Phase 5) can now build on a randomized battery + statistical
      acceptance rule + external baselines.

### Statistical Acceptance Rule for Self-Improvement Loop Phase 4 (2026-04-21)
- [x] **New `statistical_accept()` in `panobbgo/harness.py`** — principled
      accept/reject decision on two `HarnessResult` objects using bootstrap
      confidence intervals on the composite-score delta.
  - For each shared `(problem, strategy)` pair, per-run **solve fractions**
    (the same quantity averaged into `ProblemStrategyResult.score`) are
    resampled independently on both sides.
  - Composite CI is built by averaging per-pair deltas at *matching*
    bootstrap indices — so pair dependencies are preserved, not implicitly
    decoupled.
  - Decision rule (`planning/SELF_IMPROVEMENT_LOOP.md` §6.2): accept iff
    (a) `delta > eps_accept` (default `0.005`), (b) the CI lower bound is
    `> 0`, and (c) no pair regresses by more than `eps_regress` (default
    `0.05`).  Returns a `StatisticalDecision` with the verdict, overall
    CI, worst regressing pair, reasons, and per-pair `PairCI` entries.
- [x] **New `--statistical` flag on `benchmark_harness.py compare`** plus
      the knobs `--eps-accept`, `--eps-regress`, `--n-boot`,
      `--confidence`, `--stat-seed`.  When combined with
      `--fail-on-regression` the CLI exits `2` on rejection, so this is
      usable as a CI gate or as the accept/revert signal for an autonomous
      loop driver.
- [x] **Machine-readable JSON output** — with `--json --statistical` the
      payload carries a `statistical` block (verdict, CI, worst pair,
      per-pair CIs) so an agent can drill into the cause of a rejection.
- [x] **22 tests in `tests/test_harness_stats.py`** — covers accept /
      reject paths, noise-only rejection, per-pair regression guard,
      CI bracketing, reproducibility under the RNG seed, the no-shared-pairs
      degenerate path, JSON serialisation, and three CLI integration
      tests (accept, reject-regression, JSON payload shape).
- [x] **Documentation updated**
  - `doc/source/guide_benchmarking.rst`: new "Statistical acceptance
    rule" section with decision-rule walkthrough, flag table, sample JSON
    payload, and programmatic API pointer.  "Self-improvement loop"
    section now points at the shipped function.
  - `doc/source/guide.rst`: quick-nav entry updated.
  - `AGENTS.md`: "Statistical rigor" subsection now documents
    `--statistical` and the `statistical_accept()` API; key-files list
    updated.
  - `planning/SELF_IMPROVEMENT_LOOP.md`: Phase 4 marked shipped; missing-
    pieces checklist updated.
  - This TODO entry.

### BIPOP-CMA-ES Restart Mode (2026-04-20)
- [x] **Added BIPOP-CMA-ES restart support to `CMAES` heuristic** (`panobbgo/heuristics/cma_es.py`)
  - New `restart_mode` parameter: ``"ipop"`` (default, existing) or ``"bipop"`` (new)
  - BIPOP alternates two restart regimes following Hansen (2009):
    * **Large regime**: geometric population growth ``λ_l = 2^k · λ_default``
      (where ``k`` is the number of large-regime selections so far), σ resets to default
    * **Small regime**: random small population
      ``λ_s = ⌊λ_default · (½ · λ_l/λ_default)^(U[0,1]²)⌋`` and random small step size
      ``σ_s = σ_default · 10^(-2·U[0,1])``
  - Regime selection: after each restart, the regime that has accumulated *fewer*
    cumulative evaluations is selected next (ties → large)
  - New properties: `bipop_regime`, `bipop_evals_large`, `bipop_evals_small`
  - Refactored common restart bookkeeping into shared `_apply_restart()` helper
  - Reference: N. Hansen (2009). "Benchmarking a BI-Population CMA-ES on the
    BBOB-2009 Function Testbed." GECCO Workshop on BBOB.
- [x] **Updated `BIPOP_CMAES` strategy in full benchmark harness** (`panobbgo/harness.py`)
  - Now uses real BIPOP via `restart_mode="bipop"` (previously was just IPOP with more restarts)
  - Pairs `CMAES(sigma0=0.3, restart_mode="bipop")` with diverse Restart analyzer (max 10 restarts)
- [x] **18 new tests** (`tests/test_heuristic_cmaes.py::TestCMAESBIPOP` + integration test)
  - Parameter validation: default mode is "ipop"; invalid modes raise ValueError
  - Initial state: large regime, zero evals tracked
  - Regime alternation: balances cumulative budget within one delta
  - Large regime: geometric population growth `λ_l = 2^k · λ_default`
  - Small regime: λ ≥ base, σ ≤ default
  - Distribution state resets correctly (paths, covariance, eigendecomposition)
  - Box-clamped emission post-restart, base_lam preserved
  - IPOP path unchanged when `restart_mode="ipop"` (no BIPOP attribution)
  - Integration test: BIPOP-CMA-ES on Rastrigin reaches < 20 within 80 evals
- [x] **Documentation updated**
  - `doc/source/guide_architecture.rst`: CMAES section now documents IPOP and BIPOP
    schemes with mathematical formulas and selection rule
  - `doc/source/guide_usage.rst`: New "Highly multimodal problems with BIPOP-CMA-ES"
    section with worked example and IPOP-vs-BIPOP guidance; portfolio table updated
  - `TODO.md`: this entry

### External Baselines for Harness (Self-Improvement Loop Phase 2) (2026-04-20)
- [x] **New `panobbgo/harness_baselines.py`** — adapter strategies so the harness
      can judge Panobbgo in *absolute* terms, not just relative to itself.
  - `RandomSearchStrategy` — uniform random search (composite-score floor).
  - `SciPyDEStrategy` — wraps `scipy.optimize.differential_evolution`
    (population-based global optimizer).
  - `SciPyAnnealStrategy` — wraps `scipy.optimize.dual_annealing`
    (generalized simulated annealing with L-BFGS-B polish).
  - `BaselineStrategy` base class: minimal duck-typed surface matching what
    `BenchmarkHarness._run_single` actually uses (`config.max_eval`, `start()`,
    `best`, `results.results`) — no `StrategyBase` subclass, no event bus.
  - Hard evaluation-budget enforcement via `_BudgetExhausted` raised from
    the objective wrapper: external solvers can never overshoot the harness
    contract, regardless of their own stopping criteria.
  - Results DataFrame uses the same MultiIndex columns (`("fx", 0)`,
    `("who", 0)`, `("x", j)`, …) as Panobbgo strategies, so the harness'
    convergence extractor and heuristic-count logic work unchanged.
- [x] **`HarnessConfig.include_baselines` flag** — when True, the three
      baseline `StrategySpec`s are appended to the mode's strategy list.
- [x] **`benchmark_harness.py --baselines` CLI flag** on both `run` and `list`.
- [x] **22 tests in `tests/test_harness_baselines.py`**
  - Objective wrapper records / stops / projects into box.
  - Adapter surface (config, add/add_analyzer no-op, abstract `_optimize`,
    MultiIndex results, populated `best`).
  - Per-solver budget enforcement and convergence on simple problems.
  - Harness integration: `include_baselines` append path, filtering,
    end-to-end smoke with `composite_score` in [0, 1].
  - Seed reproducibility of the Random baseline.
- [x] **Documentation updated**
  - `doc/source/guide_benchmarking.rst`: replaced "Absolute baselines
    (planned)" with a full shipping section (usage, design, CMA-ES note).
  - `doc/source/guide.rst`: quick-nav entry mentions baselines.
  - `AGENTS.md`: "External baselines" subsection and key-files list updated.
  - This TODO entry.
- [ ] **Next in the roadmap** — statistical acceptance rule (bootstrap CI in
      `compare`, Phase 4) and parametric randomization (Phase 3).

### Benchmark Harness Documentation & Self-Improvement Plan (2026-04-19)
- [x] **New Sphinx guide** (`doc/source/guide_benchmarking.rst`)
  - Full definition of `composite_score` with math, interpretation table, pitfalls
  - Documents the `quick`/`standard`/`full` modes, reproducibility model, `compare` workflow
  - Sections on statistical caveats, parametric randomization (planned), absolute baselines (planned)
  - Wired into `doc/source/guide.rst` toctree
- [x] **Expanded `AGENTS.md`** Benchmark Harness section
  - Statistical rigor subsection (quick-mode noise, re-run at alt seed before accepting small deltas)
  - Self-improvement loop pointer
  - Explicit "composite score formula is a stable contract" note
- [x] **Enriched module docstrings** (`panobbgo/harness.py`, `benchmark_harness.py`)
  - Explicit composite-score formula with per-run solve fraction `s = 1 - (k* - 1)/B`
  - Stability contract for the formula
  - Pointers to the guide and self-improvement plan
- [x] **New plan** (`planning/SELF_IMPROVEMENT_LOOP.md`)
  - Vision for measure→propose→apply→measure→accept/revert loop against randomized problems
  - Parametric problem battery design (translate/rotate/scale/noise/dim sampling)
  - External absolute baselines (scipy DE, dual_annealing, pycma, random)
  - Bootstrap-CI-based statistical acceptance rule + anti-cherry-pick guard
  - Safety rails (dedicated branch, atomic commits, test gating, STOP sentinel)
  - Six-phase rollout from MVP to production loop + success criteria

### Known gap (tracked in plan)
- [ ] Parametric randomization of benchmark problems — plan Phase 3
- [ ] External absolute baselines in the harness — plan Phase 2
- [ ] Statistical acceptance rule (bootstrap CI) in `compare` — plan Phase 4
- [ ] Loop driver `scripts/self_improve.py` — plan Phase 5

### IPOP-CMA-ES Restart Support (2026-04-19)
- [x] **Added IPOP restart to `CMAES` heuristic** (`panobbgo/heuristics/cma_es.py`)
  - `on_restart(center, reason)` handler: moves search mean to new center, doubles λ (IPOP)
  - Resets covariance matrix C, evolution paths p_c/p_σ, and step size σ to initial values
  - Flushes stale pending/in-flight generation results on restart
  - Recomputes all CMA-ES adaptation constants (c_σ, d_σ, c_c, c_1, c_μ) for new population
  - `ipop_factor` parameter (default 2.0) controls per-restart population growth multiplier
  - `restart_count` property tracks total number of IPOP restarts triggered
  - `_base_lam` records the initial population size (preserved across restarts)
  - Reference: Auger & Hansen (2005). "A restart CMA evolution strategy with increasing
    population size." CEC 2005.
- [x] **Added `IPOP_CMAES` strategy to standard benchmark harness** (`panobbgo/harness.py`)
  - Pairs `CMAES(sigma0=0.3, ipop_factor=2.0)` with `Restart(patience=None, restart_strategy="diverse", max_restarts=5)`
  - `Sensitivity` analyzer included for adaptive Nearby perturbations
- [x] **Added `BIPOP_CMAES` strategy to full benchmark harness**
  - Same as IPOP_CMAES but with `max_restarts=10` for the larger 500-eval budget
- [x] **25 comprehensive tests** (`tests/test_heuristic_cmaes.py`)
  - Unit tests for: default/custom ipop_factor, restart_count tracking, population doubling
  - Correctness tests: mean moves to center, sigma resets, paths reset, covariance resets
  - Behavioral tests: pending queue flushed, new generation emitted, box-constraint preservation
  - Weight renormalization, base_lam preservation, multiple restarts
  - Integration tests: IPOP on Rastrigin 2D (200 evals), restart triggered with short patience
- [x] **Documentation updated**
  - `doc/source/guide_architecture.rst`: CMAES entry now documents IPOP restart capability
  - `doc/source/guide_usage.rst`: New "Multimodal problems with IPOP-CMA-ES" section with
    worked example, parameter guide, and comparison to plain CMA-ES

### CMA-ES Heuristic & Core Reward Fix (2026-04-18)
- [x] **Implemented `CMAES` heuristic** (`panobbgo/heuristics/cma_es.py`)
  - Pure-NumPy implementation of the canonical CMA-ES algorithm (Hansen 2016)
  - Async-compatible: tracks results per generation via `who = "CMAES:g<gen>:i<idx>"` tags
  - Adapts covariance matrix C and step size σ from evaluated offspring
  - Lazy eigendecomposition with condition-number guard (resets to spherical if > 1e7)
  - Parameters: `sigma0` (initial step-size fraction), `popsize` (overrides λ=4+3 ln n),
    `min_results_fraction` (fraction of λ before update trigger, default 0.5 = μ)
  - Gold standard for smooth/ridge-following problems (Rosenbrock, ill-conditioned quadratics)
- [x] **Added `CMAES` to standard and full harness strategies** (`panobbgo/harness.py`)
  - `CMAES_Portfolio` strategy in standard mode: LatinHypercube + CMAES + Nearby + NelderMead
  - `CMAES_GP` strategy in full mode: LatinHypercube + CMAES + GaussianProcessHeuristic + NelderMead
  - Intentionally excluded from quick mode (75 evals) — CMA-ES needs ≥ 100 evals to converge
- [x] **Fixed `StrategyBase.heuristic()` lookup for compound `who` strings** (`panobbgo/core.py`)
  - Heuristics like CMAES and DifferentialEvolution embed generation/UUID info in `who`
    (e.g., `"CMAES:g3:i0"`, `"DifferentialEvolution:abc123"`)
  - `heuristic(who)` now falls back to the prefix before `:` if the full key is not found
  - Prevents spurious `KeyError` in `StrategyRewarding.on_new_best` and `_reward_near_best`
- [x] **20 comprehensive tests** (`tests/test_heuristic_cmaes.py`)
  - Initialisation, parameter validation, point emission within bounds
  - Update triggering from partial results, mean convergence, sigma adaptation
  - Covariance positive-definiteness, weight normalisation, foreign-result handling
  - End-to-end integration test on Rosenbrock 2D (150 evals, fx < 5.0)
- [x] **Documentation updated**
  - `doc/source/guide_architecture.rst`: CMAES added to "Population-based" heuristics section
  - `doc/source/guide_usage.rst`: Added CMA-ES portfolio table entry and usage example

### Bayesian Optimization Harness Integration & UCB Bug Fix (2026-04-17)
- [x] **BayesOpt_GP strategy added to standard harness** (`panobbgo/harness.py`)
  - New `BayesOpt_GP` strategy spec added to `_make_standard_strategies()` (200-eval budget)
  - Uses `GaussianProcessHeuristic(n_restarts=5)` + `LatinHypercube(div=4)` + `Nearby` + `NelderMead`
  - Demonstrates GP-based Bayesian optimization within the reproducible harness
- [x] **BayesOpt_Enhanced added to full harness** (`panobbgo/harness.py`)
  - New `BayesOpt_Enhanced` strategy spec added to `_make_full_strategies()` (500-eval budget)
  - Combines `GaussianProcessHeuristic(n_restarts=10)` + `DifferentialEvolution` + `NelderMead`
  - DifferentialEvolution provides global search; GP provides surrogate-guided exploitation
- [x] **Fixed UCB acquisition function bug** (`panobbgo/heuristics/gaussian_process.py`)
  - `_upper_confidence_bound` was maximising LCB instead of minimising it (wrong for minimisation)
  - Fixed: method now returns `-(μ - κσ)` so the outer maximiser correctly minimises LCB
  - Acquisition functions EI and PI were already correct; only UCB was affected
- [x] **Documentation updated** (`doc/source/guide_architecture.rst`, `doc/source/guide_usage.rst`)
  - `guide_architecture.rst`: Added `GaussianProcessHeuristic`, `DifferentialEvolution`,
    `FeasibleSearch`, `ConstraintGradient`, `LocalPenaltySearch`, `ConstraintRepair` to heuristics
  - `guide_architecture.rst`: Added `StrategyUCB`, `StrategyThompsonSampling`, `StrategyLinUCB`,
    `StrategyPhased` with mathematical descriptions
  - `guide_usage.rst`: Added "Bayesian Optimization with Gaussian Process" section with
    acquisition function details, EIC description, and two-phase BO workflow example
  - `guide_usage.rst`: Updated heuristic portfolio table and recommended configurations

### Sensitivity-Aware Nearby Heuristic & StrategySpec Analyzers (2026-04-15)
- [x] **Sensitivity-Aware `Nearby` Heuristic** (`panobbgo/heuristics/nearby.py`)
  - Added `on_new_sensitivity(importance)` event handler to `Nearby`
  - When `Sensitivity` analyzer is active and has published importance scores,
    `Nearby` scales per-dimension perturbations by importance (normalised so overall
    magnitude is preserved)
  - New `sensitivity_scale` constructor parameter controls contrast sharpness (default 1.0)
  - For `axes="all"`: each dimension's step is multiplied by its (normalised) weight
  - For `axes="one"`: dimension is sampled proportionally to importance weights
  - Both `on_new_best` and `on_restart` use the sensitivity-aware `_make_perturbation` helper
  - Improves local search in high-dimensional problems where only a subset of dimensions matter
  - Added `_perturbation_weights()` helper returning normalised weights (mean = 1)
- [x] **`StrategySpec.analyzers` field** (`panobbgo/benchmark.py`)
  - Added optional `analyzers: List[Tuple[type, dict]]` field to `StrategySpec`
  - `create_strategy()` adds extra analyzers (e.g. `Sensitivity`, `Restart`) alongside heuristics
  - Four required analyzers (Best, Grid, Splitter, Convergence) still added in `initialize()`
- [x] **Sensitivity in Benchmark Strategies** (`panobbgo/harness.py`)
  - Added `Sensitivity(update_interval=20)` to `Rewarding_Diverse`, `UCB_Diverse`, and `Thompson_Diverse`
  - Enables adaptive Nearby perturbations in all adaptive benchmark strategies
- [x] **15 new tests** (`tests/test_heuristic_nearby_sensitivity.py`)
  - Verifies `_perturbation_weights()` normalisation and ordering
  - Confirms sensitivity-aware perturbations statistically bias important dimensions
  - Tests both `axes="all"` and `axes="one"` modes
  - Tests `on_restart` with/without sensitivity and with None center
  - Tests that sensitivity updates are immediately effective
  - Tests `StrategySpec.analyzers` round-trip and creation
- [x] **Documentation updated**
  - `doc/source/guide_architecture.rst`: updated event table and Nearby description
  - `doc/source/guide_usage.rst`: added Sensitivity-Aware Nearby section with example

### Benchmark Harness for Agent Feedback Loops (2026-02-23)
- [x] **Implemented `panobbgo/harness.py` – Reproducible Benchmark Harness**
  - `BenchmarkHarness` class: runs seeded, reproducible benchmark suites
  - Three modes: `quick` (3 problems × 2 strategies × 3 reps, 75 evals), `standard`, `full`
  - Per-run seed derivation for best-effort reproducibility across runs
  - Convergence trace extraction directly from the MultiIndex results DataFrame
  - ERT (Expected Running Time) and per-pair performance score in [0, 1]
  - Composite score = mean of per-pair scores; single scalar for before/after comparison
  - Full JSON serialisation / deserialisation (`save()` / `load()`)
  - `compare()` helper: diff two `HarnessResult` files, flag regressions/improvements
- [x] **`benchmark_harness.py` – CLI for Agent Loop**
  - `run`: execute benchmarks and save a timestamped JSON file
  - `score`: print human-readable summary + optional machine-readable JSON
  - `compare`: side-by-side diff with `--fail-on-regression` exit-code support
  - `list`: enumerate available problems and strategies per mode
- [x] **`tests/test_harness.py` – 60 tests covering all harness components**
  - Unit tests for metrics, serialisation, comparison, seed derivation
  - Smoke integration tests for end-to-end runs (single problem, 30 evals)
  - CLI tests via `main()` invocation

### Contextual Bandit Strategy (2025-01-13)
- [x] **Implemented StrategyLinUCB (Contextual Bandits)**
  - Implemented `StrategyLinUCB` with disjoint linear models for each heuristic.
  - Features include Bias, Budget Progress, and Recent Success Rate.
  - Added unit/integration test `tests/test_strategy_contextual.py`.
  - Updated `panobbgo/lib/classic.py` to support `Rosenbrock(dim=2)` kwargs.

### Thompson Sampling Strategy (2025-01-13)
- [x] **Implemented StrategyThompsonSampling**
  - Added new strategy using Beta-Bernoulli bandit logic
  - Implemented `reward` based on improvement magnitude
  - Implemented `execute` with randomized selection based on Beta samples
  - Added unit tests in `tests/test_strategy_thompson.py`

### PR #43 - Dask Memory Leak Fix & Test Suite Cleanup (2025-01-13)
- [x] **Fixed Critical Memory Leak in Dask Cleanup**
  - Added proper `LocalCluster` cleanup in `_setup_dask_cluster()` and shutdown code
  - Store cluster reference (`self._cluster`) to ensure worker processes are terminated
  - Call both `self._client.close()` AND `self._cluster.close()` during cleanup
  - Prevents memory blowup when running multiple tests that use Dask evaluation
- [x] **Deferred Dask Testing (Future Work - Weeks)**
  - Disabled all Dask-related tests (`test_config_init.py`, `test_dask_evaluation_integration()`)
  - Default test execution model is now "threaded" only
  - Dask evaluation still works in production, just not tested in test suite
  - TODO: Proper Dask test isolation and cleanup testing in future sprint

### PR #42 - FeasibleSearch & Test Warnings (2025-01-13)
- [x] **Test Suite Warnings Resolved**
  - Fixed NumPy RuntimeWarnings in convergence analyzer using `warnings.catch_warnings()`
  - Suppressed warnings for edge cases (identical values, small samples) in std deviation calculations
  - Skipped Dask evaluation integration test (focusing on threaded evaluation for now)
  - All 143 tests now pass with 1 skipped, 0 warnings
- [x] **FeasibleSearch Heuristic Enhanced**
  - Implemented biased line search using Beta(2,1) distribution for more efficient boundary finding
  - Improved comments explaining the line search strategy between feasible/infeasible points
  - Updated copyright year to 2012-2025 per project guidelines
  - All FeasibleSearch tests passing

## Framework Quality Assurance & Completion

### 🔴 CRITICAL: TDD Bug Fixes & Quality Validation (Priority 1)
**TDD Approach**: Write failing tests first, then implement fixes
- [x] **Optimization Loop Stability** - Major hanging issues resolved
  - [x] **FIXED**: Random heuristic infinite wait (main hang cause)
  - [x] **FIXED**: abs() errors in convergence analyzer and progress reporting
  - [x] Basic optimization now completes successfully
  - [ ] Full optimization loop robustness (complex threading - lower priority)
- [x] **Heuristic Functionality** - Core issues resolved
  - [x] **FIXED**: Random heuristic infinite wait (main hang cause)
  - [x] **VALIDATED**: Nearby heuristic generates correct points
  - [x] Added TDD tests for heuristic point generation
  - [ ] Full event system integration (lower priority)
- [x] **Dedensifyer Analyzer** - Fix critical implementation bugs
  - [x] Write TDD tests for proper initialization and grid management
  - [x] Fix constructor (missing strategy parameter)
  - [x] Fix undefined variables and wrong method signatures
  - [x] Validate hierarchical grid functionality
- [x] **Optimization Correctness Validation** - Add tests proving algorithms work
  - [x] Write tests validating convergence to known optima
  - [x] Compare optimization vs random baseline performance
  - [x] Add statistical significance testing

### 🟡 MEDIUM: Coverage Expansion on Validated Code (Priority 2)
**Revised Goal**: 75% coverage on components proven to work correctly
- [x] Expand UCB strategy tests (currently 91% - add edge cases)
- [x] Complete Best analyzer test coverage (currently 34%)
- [x] Add Grid analyzer comprehensive tests (currently 56%)
- [x] Test remaining heuristics: LBFGSB (30%), Nelder-Mead (51%)
- [x] Add integration tests for constrained optimization scenarios

### 🟢 LOW: Documentation & Polish (Priority 3)
- [x] Update documentation references from IPython parallel to Dask
- [x] Review and fix minor naming inconsistencies in guide documentation
- [x] Remove remaining IPython parallel references from code and documentation
- [ ] Review and potentially simplify UI components
- [ ] Add performance benchmarks comparing different strategies
- [ ] Review and optimize threading/event handling

### 🔵 DEFERRED: Dask Testing & Validation (Future Work - Weeks)
**Status**: Completed! Dask tests are isolated, pass locally, and memory leak fix is verified.
- [x] **Dask Test Isolation**: Properly isolate Dask tests to avoid port conflicts
  - Use pytest fixtures to ensure clean Dask cluster setup/teardown
  - Ensure each test gets a fresh LocalCluster with unique dashboard port
  - Test that cluster cleanup properly terminates all worker processes
- [x] **Re-enable Dask Tests**: Currently skipped tests
  - `tests/test_config_init.py` - testing_mode and dashboard configuration
  - `tests/test_integration.py::test_dask_evaluation_integration` - Dask evaluation
- [x] **Verify Memory Leak Fix**: Test that the LocalCluster cleanup fix prevents memory leaks
  - Run repeated Dask evaluations and monitor memory usage
  - Verify worker processes are terminated after cleanup
- [x] **Dask Production Usage**: While tests are disabled, Dask evaluation still works
  - Document current Dask usage patterns for production
  - Consider adding example scripts demonstrating Dask evaluation

## Known Issues & Technical Debt

### Strategy Lifecycle Management (Systemic Issue)
**Problem**: Real strategy instances (StrategyRoundRobin, StrategyRewarding) start background processes (via Dask) that don't clean up properly when tests complete. This causes:
- Test hangs when multiple tests use real strategies (PR #35, PR #32)
- Resource leaks in test suites
- Unreliable benchmark tests

- `strategy.start()` initializes background threads/processes
- [x] **FIXED**: Strategy lifecycle methods (`__stop__`, `_cleanup`) implemented.
- [x] **FIXED**: Context manager support (`__enter__`, `__exit__`) implemented.
- Tests can now properly tear down strategies using `strategy.stop()` or `with` blocks.

**Current Workarounds**:
- Unit tests: Use `@mock.patch("panobbgo.core.StrategyBase")` to avoid real strategies
- Integration tests: Skip tests that hang (e.g., `test_heuristic_tracking` in benchmarks)
- Set `evaluation_method="threaded"` helps but doesn't fully solve cleanup issues

**Proper Solution Needed**:
- [x] **FIXED**: Cleanly terminate background processes.
- [x] **FIXED**: Implementation of `strategy.cleanup()` methods.
- [x] **FIXED**: Context manager support (`__enter__`/`__exit__`).
- [x] Implement pytest fixtures for automatic strategy setup/teardown in tests.
- [ ] Review all Dask distributed usage for best practice cleanup patterns.

**Affected Files**:
- `panobbgo/core.py` - StrategyBase class needs lifecycle methods
- `tests/test_heuristic_feasible.py` - Fixed by using mocked strategies (PR #35)
- `benchmarks/test_benchmarks.py` - Skipped hanging test (PR #32)

### Benchmark Heuristic Tracking Issues (PR #32)
**Bug in convergence_trace logic** (`benchmarks/test_benchmarks.py:88-93`) - **FIXED**:
- ~~When `best_fx == float('inf')` (first evaluation), `old_best_fx` is set to `result.fx`~~
- ~~This causes `improvement = result.fx - result.fx = 0`, which is incorrect~~
- **Fixed**: First improvement now correctly recorded as `result.fx` (function value from baseline)
- **Fixed**: Subsequent improvements correctly calculated as `best_fx - result.fx`

### 🎯 TARGET: 75% Coverage on Validated Components
**Prerequisites**: All Priority 1 items completed with TDD validation
**Quality Metrics**: Correctness + Coverage (not just coverage)
**Status**: Core issues resolved, coverage stands at ~71%.

## Known Issues & Technical Debt

### Strategy.start() Hang Bug (FIXED)
**CRITICAL**: `strategy.start()` doesn't return after reaching `max_eval` evaluations
- **Status**: FIXED by addressing result collection deadlocks and improving cleanup in [PR #38](https://github.com/haraldschilly/panobbgo/pull/38).

### PR #36 Bug Fixes (Merged)
**Fixed Issues** - All good fixes:
- [x] **Splitter.Box.__ranges** - Fixed `.ptp()` call to work with BoundingBox objects (`panobbgo/analyzers/splitter.py:215-220`)
- [x] **memoize decorator** - Added handling for unhashable NumPy arrays by converting to bytes (`panobbgo/utils.py:205-230`)
- [x] **Analyzer name consistency** - Changed "splitter"/"best" to "Splitter"/"Best" (Random, WeightedAverage heuristics)
- [x] **Random heuristic initialization** - Added logic to get root leaf from Splitter on start (`panobbgo/heuristics/random.py:38-48`)
