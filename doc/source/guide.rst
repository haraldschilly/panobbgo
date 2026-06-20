Panobbgo User Guide
===================

Welcome to the comprehensive guide for **Panobbgo** (Parallel Noisy Black-Box Global Optimization).

This guide provides everything you need to understand, use, and extend Panobbgo for your black-box optimization problems.

Overview
--------

Panobbgo is a flexible, modular Python framework for optimizing expensive black-box functions where:

- You can only evaluate the function, not analyze its internal structure
- Evaluations may be noisy or stochastic
- You have a limited evaluation budget
- You can leverage parallel computing resources
- You may have box constraints and constraint violations

The framework uses an **event-driven architecture** with multiple point generation **heuristics**
that are coordinated by an adaptive **strategy** implementing a **multi-armed bandit** approach.

Quick Navigation
----------------

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - **Just want to get started?**
     - Follow :doc:`guide_setup` for a step-by-step setup guide with executable tests
   * - **New to Panobbgo?**
     - Start with :doc:`guide_introduction` to understand what it does and when to use it
   * - **Want mathematical details?**
     - See :doc:`guide_mathematical_foundation` for the theory behind the algorithms
   * - **Understanding the design?**
     - Read :doc:`guide_architecture` for the system architecture and component interactions
   * - **Ready to use it?**
     - Follow :doc:`guide_usage` for installation, configuration, and basic examples
   * - **Want to customize?**
     - Check :doc:`guide_extending` for adding custom heuristics, analyzers, and problems
   * - **Interested in research?**
     - Explore :doc:`guide_research` for related work, theoretical properties, and future directions
   * - **Want to measure progress?**
     - Read :doc:`guide_benchmarking` for the composite score, external baselines, parametrically randomised problems, the statistical acceptance rule (with paired and unpaired bootstrap sampling — paired is auto-selected for the rep-aligned randomized harness and shrinks the CI 3-10× over the historical independent-resample scheme), the autonomous self-improvement loop driver, the anti-cherry-pick guard, the hold-out validation set (single- and multi-seed, with bootstrap-CI aggregation across hold-out seeds, plus — shipped 2026-06-11 — the ``LoopHoldoutRecord.status`` field that surfaces empty-ladder hold-outs as ``"vacuous"`` instead of mis-reporting them as ``OK drift=+0.0000``; the bootstrap aggregator filters vacuous records out so a single negative-drift seed can no longer be masked by a vacuous companion, and the ``run`` / ``summary`` CLI commands print ``VACUOUS`` / ``VACUOUS_CI`` verdicts via :meth:`panobbgo.self_improve.LoopHoldoutRecord.effective_status`), the adaptive Thompson-sampling mutation sampler with optional per-class structural bandit arms (and the hierarchical Beta-Binomial ``structural_borrow_alpha`` coefficient that lets each per-class arm borrow strength from the op-level aggregate so a fresh candidate class warms with the op's empirical accept rate instead of the symmetric ``Beta(1, 1)`` prior), the structural ``add_heuristic`` / ``drop_heuristic`` portfolio mutations (plus the symmetric ``add_analyzer`` / ``drop_analyzer`` ops shipped 2026-06-02 that extend the loop's reach to the analyzer bucket — ``Sensitivity`` and ``Restart`` as the default candidate pool), the categorical ``MutationRule`` kind (for discrete knobs like ``PSO.topology``, ``Sobol.scramble`` — the quick-mode default was codified to ``False`` on 2026-05-31 after three ledger-confirmed accepts, see §13, ``LSHADE.archive_factor``, ``LSHADE.F_schedule``, ``NLSHADE_RSP.adaptive_archive``, the literature-regime ``NLSHADE_RSP.k_rank`` toggle — ``0.0`` jSO recovery / ``3.0`` Stanovov default / ``5.0`` aggressive — sitting alongside the continuous ``float_uniform`` rule, the literature-regime ``JSO.p_best_max`` toggle shipped 2026-06-09 — ``0.15`` L-SHADE-like / ``0.25`` jSO default / ``0.4`` iLSHADE-like — also sitting alongside the continuous ``float_uniform`` rule on the same slot (the L-SHADE setting is raised from the canonical ``0.11`` to ``0.15`` so it clears jSO's default ``p_best_min = 0.125`` floor), and ``COBYQA.scale``), the four-topology PSO (``gbest`` / ``lbest`` / ``vonneumann`` / ``random`` — fully-connected, ring, 4-connected 2-D toroidal grid per Kennedy & Mendes 2003 / Mendes 2004, and the stochastic informer graph per Mendes 2004 / Clerc 2007 / SPSO 2011, with the optional stochastic-K stagnation-rebuild ``PSO.stagnation_threshold`` knob for the ``random`` variant, plus the under-tuned analyzer / local-optimizer dials ``Restart.patience`` — the more impactful of the two :class:`~panobbgo.analyzers.restart.Restart` knobs, controlling how aggressively the optimizer restarts when stuck — and ``LBFGSB.max_starts`` — the multi-start L-BFGS-B restart budget cap, both opt-in ``integer_add`` rules that skip the heuristic-internal ``None`` auto-default sentinel, plus the new categorical ``Restart.restart_strategy`` rule shipped 2026-06-07 that lets the bandit flip an existing :class:`Restart` instance between ``"random"`` (uniform-in-box), ``"diverse"`` (max-min distance from previous centres), and ``"sphere"`` (Gaussian around the box centre with ``std = ranges / 6``) without dropping and re-adding the analyzer, plus the three RegionUCB leaf-bandit dials shipped 2026-06-08 — ``RegionUCB.ucb_c`` (UCB1 exploration weight, ``log_uniform_perturb`` in ``(0.1, 4.0)``), ``RegionUCB.gauss_fraction`` (fraction of in-leaf draws taken as Gaussian-around-best instead of uniform-over-leaf, ``float_uniform`` over the full ``[0, 1]`` range so the LA-MCTS pure-uniform regime and the pure-local-refinement regime are both symmetrically reachable), and ``RegionUCB.gauss_scale`` (Gaussian std-dev as a fraction of the leaf's ranges, ``log_uniform_perturb`` in ``(0.05, 0.5)``) so the bandit can learn the exploration / exploitation balance of the per-region allocator on a per-problem basis) candidate pool, the L-SHADE adaptive Differential Evolution heuristic (Tanabe-Fukunaga 2014) with two opt-in jSO refinements (the linearly-decreasing ``p_best`` schedule from iLSHADE / jSO Brest et al. 2016 / 2017 and the three-phase asymmetric F-cap from jSO Brest et al. 2017), the literature-faithful jSO heuristic itself (Brest, Maučec & Bošković 2017 — CEC-2017 winner — inheriting the L-SHADE F-cap machinery by construction), the NL-SHADE-RSP heuristic (Stanovov, Akhmedova & Semenkin 2021 — CEC-2021 winner — a jSO subclass adding non-linear population reduction, rank-based selective pressure, and a randomised adaptive archive), the NL-SHADE-LBC heuristic (Stanovov, Akhmedova & Semenkin 2022 — CEC-2022 winner — a NL-SHADE-RSP subclass adding **Linear Bias Change** in the success-history memory update: the F / CR Lehmer-mean exponents are linearly scheduled across budget progress instead of fixed), the LSHADE-EpSin heuristic (Awad, Ali & Suganthan 2016 — direct precursor of the CEC-2017 co-winner LSHADE-cnEpSin — an L-SHADE subclass that replaces SHADE Cauchy-F memory sampling with an ensemble of two sinusoidal F candidates during the first half of the search), the COBYQA derivative-free trust-region local optimizer (Ragonneau-Zhang 2023), the multi-start L-BFGS-B gradient-based local optimizer (Zhu-Byrd-Lu-Nocedal 1997 — the only gradient-based arm in the catalog, for smooth ill-conditioned valleys like Rosenbrock), and the inactivity-guarded ``eps_accept`` relaxation knob that breaks the loop out of long accept droughts by geometrically decaying the accept threshold after every ``inactivity_relax_after`` consecutive non-accepts (floored at ``inactivity_min_eps_accept``, re-tightened on the next accept, with per-iteration ledger fields for honesty), the §12.4 no-op detection shipped 2026-06-12 (``LoopIterationRecord.no_op`` plus :meth:`AdaptiveMutationSampler.discard_outcome`) that flags iterations whose per-(problem, strategy) candidate scores are bit-identical to baseline, excludes them from bandit pulls so the Thompson posterior is no longer mis-trained on dormant-rule mutations, and surfaces a separate ``no-op=N`` bucket in the ``scripts/self_improve.py summary`` view, the §7.4 graded bandit reward shaping shipped 2026-06-13 (``LoopConfig.bandit_reward_shaping = "graded"`` / ``--bandit-reward graded`` plus the new ``MutationRuleStats.reward_sum`` field and the per-iteration ``LoopIterationRecord.bandit_reward`` field) that replaces the binary +1/+0 accept/reject signal with a continuous reward in ``[0, 1]`` derived from the bootstrap CI / point delta — so a barely-confirmed accept (``r ≈ 0.5``), an honest near-miss reject (``r ≈ 0.5``) and a clearly-harmful reject (``r ≈ 0``) become distinguishable, lifting the bandit's per-night information yield from the ~2.5% accept-rate floor of §2.6 to ~65% of iterations carrying real signal (paired naturally with the §12.4 no-op gate — the two together turn the bandit's sparse 0/1 binary signal into a dense graded ``[0, 1]`` signal on every informative iteration), the V2 §9.3 / §9.5 step 4 ``codify-scan`` CLI shipped 2026-06-17 (``scripts/self_improve.py codify-scan`` plus the underlying :class:`panobbgo.self_improve.CodifyCandidate` dataclass, :func:`panobbgo.self_improve.aggregate_codify_candidates` scanner, and :func:`panobbgo.self_improve.load_ledgers_for_codify_scan` io helper) that scans the live ledger plus every rotated archive under ``planning/done/`` for directionally-consistent accepted mutations and surfaces every ``(class, param, direction)`` (or ``(op, class)`` for structural ops) group that fires on at least ``--min-nights`` distinct accept dates with all contributing per-record CI lower bounds > 0, ranked by ``(n_distinct_nights, mean_delta, n_accepts)`` so the daily-routine operator can codify the strongest evidence first — directly the persistence mechanism the V2 §11 success criterion 2 (≥3 codify PRs opened, ≥2 merged) depends on, with ``--confirmed-only`` opt-in for post-§6.4 ledgers, a ``--json`` mode for external dashboards, and the 2026-06-18 already-codified suppression layer (:func:`panobbgo.self_improve.annotate_codified_status` plus the new ``CodifyCandidate.already_codified`` / ``live_codified_values`` fields and the ``--include-already-codified`` CLI flag) that cross-checks every candidate's predicted source edit against the live seed-spec factories (quick + loop registries) and hides candidates whose edit is a no-op by default — the ``Sobol.scramble = False`` candidate that surfaces from the pre-codification archive even though the seed factory already ships ``scramble=False`` is the motivating example — so the operator's attention stays on actionable evidence, and the 2026-06-19 mutation-bound widening detector (:class:`panobbgo.self_improve.WideningCandidate` plus :func:`panobbgo.self_improve.detect_widening_candidates` and the ``codify-scan --widen-bounds`` / ``--widen-factor`` CLI flags) that pairs every bidirectional ``(class, param)`` slot — same slot accumulating accepts in *both* ``"up"`` and ``"down"`` directions across nights — into a proposed catalog ``MutationRule.bounds`` update, surfacing two actionable patterns on the live project ledger (``Nearby.radius`` and ``Sobol.n``, both bidirectional) that the basic codify-scan reports as competing default-shift proposals: the widening view turns the apparent conflict into a clear bound-update proposal instead, since the right action for a bidirectional pattern is rarely a default shift but a catalog bound focused on the observed range, and the §12.4 summary trend block shipped 2026-06-16 (three additive sub-blocks on the ``scripts/self_improve.py summary`` CLI — a per-run trend table with date / base_seed / mode / iters / decided / accepts / no-op / best Δ / seed-score columns oldest-first, a top-N / bottom-N mutation-rule bandit-posterior leaderboard ranked by graded ``mean_reward`` with configurable ``--top-n`` / ``--bottom-n`` / ``--min-attempts`` thresholds and the same :func:`panobbgo.self_improve._proposal_rule_key` collapse used by :meth:`AdaptiveMutationSampler.prime_from_ledger` so the summary view matches what a freshly-primed nightly bandit would carry, and an Inactivity-relax telemetry block surfacing the inferred ``eps_accept`` base from the maximum observed ``effective_eps_accept`` plus the longest accept drought and the relaxed-accept count / mean decay factor at accept) so the §12.3 daily routine can answer "is the loop accepting tonight?", "which arms pay off?", and "is the inactivity-relax knob doing anything?" in one screen of text instead of grepping the raw JSONL ledger, and the dedicated loop seed registry shipped 2026-06-10 (``LoopConfig.registry = "loop"`` / ``--registry loop``) that ships the two quick specs plus five compact family specs (``Loop_DE_Family`` / ``Loop_PSO`` / ``Loop_RegionUCB`` / ``Loop_LocalSearch`` / ``Loop_Restart``) with every tunable kwarg of the DE family, PSO, RegionUCB, LBFGSB+COBYQA and the :class:`Restart` analyzer explicit at the constructor default — lifts catalog kwarg-rule activation from 4 / 44 (quick seed) to 44 / 44 (loop seed) so the ~30 mutation rules shipped since mid-May 2026 actually fire on the nightly cron's seed instead of staying dormant against a registry that only set Sobol / Nearby / Sensitivity kwargs explicitly, and the §6.4 same-night confirmation gate shipped 2026-06-14 (``LoopConfig.confirm_accepts = True`` / ``--confirm-accepts`` plus the new :class:`panobbgo.self_improve.LoopConfirmRecord` ledger record and the ``LoopIterationRecord.confirmed`` field) that re-measures every screening-accepted candidate on a fresh ``randomize_iteration`` (default offset ``500_000``, distinct from the guard's ``1_000_000`` so the two fresh-seed streams never collide) — and, when at least one hold-out base_seed is configured, additionally on the *first* hold-out seed — then re-runs :func:`panobbgo.harness.statistical_accept` on the pooled (screen + confirm) sample so promotion only happens when the pooled CI still clears ``eps_accept``; failed confirmations land as ``record_type="confirm_reject"`` records carrying screen + confirm scores and the pooled CI so an auditor can trace whether the gate caught a noise spike (``screen_Δ ≫ confirm_Δ``) or a systematic regression (``screen_Δ ≈ confirm_Δ`` but pooled ``ci_low ≤ 0``); the bandit reward path consumes the *post-confirmation* pooled decision so an arm that consistently produces screening noise-spike accepts collects the reject-regime reward (binary: ``0``; graded: ``clip(0.5 + pooled_Δ/(4·eps), 0, 0.5)``) rather than the full-accept reward the screening alone would have produced — directly closes the §2.2 "Accept → rollback churn" V2 diagnosis (15/16 V1 accepts rolled back by the guard) by gating promotion behind an independent re-measurement *before* the accept is recorded, so the guard's job collapses from "roll back ~all accepts" to "catch the rare case where a confirmed accept drifts on the *next* night's fresh seed", and the archive-aware bandit priming shipped 2026-06-15 (``LoopConfig.adaptive_prime_include_archives`` / ``--prime-include-archives`` plus the matching :meth:`panobbgo.self_improve.AdaptiveMutationSampler.prime_from_archives` method) that replays archived ledgers under ``planning/done/`` matching the rotation glob ``self_improve_ledger_*.jsonl`` before the live ledger, so the bandit posterior compounds across nightly rotation boundaries rather than forgetting every pre-rotation observation — closes the V2 §2.6 "archives in ``planning/done/`` are invisible" diagnosis at the bandit-priming layer (the upstream proposal source now shares the same long-memory view of the catalog that the codify-scan step uses)

Guide Contents
--------------

.. toctree::
   :maxdepth: 2

   guide_setup
   guide_introduction
   guide_mathematical_foundation
   guide_architecture
   guide_usage
   guide_extending
   guide_benchmarking
   guide_research

Key Concepts
------------

Before diving in, here are the core concepts you'll encounter throughout Panobbgo:

**Problem**
   The black-box function you want to minimize, defined by its dimensionality and bounding box

**Point**
   A candidate solution (location in search space) generated by a heuristic

**Result**
   The outcome of evaluating a Point, including objective value and constraint violations

**Heuristic**
   A point generation strategy (e.g., random sampling, local search, model-based)

**Analyzer**
   A module that processes results and maintains derived information (e.g., best points, spatial decomposition)

**Strategy**
   The orchestrator that coordinates heuristics, manages evaluation, and tracks the optimization budget

**EventBus**
   The communication backbone enabling decoupled interaction between modules

**Multi-Armed Bandit**
   The adaptive selection mechanism that learns which heuristics work well for your problem

Typical Workflow
----------------

A typical Panobbgo workflow looks like this:

1. **Define your problem** by subclassing :class:`~panobbgo.lib.Problem`

   .. code-block:: python

      class MyProblem(Problem):
          def eval(self, x):
              return my_expensive_function(x)

2. **Choose a strategy** (typically :class:`~panobbgo.strategies.rewarding.StrategyRewarding`, :class:`~panobbgo.strategies.ucb.StrategyUCB`, or :class:`~panobbgo.strategies.thompson.StrategyThompsonSampling`)

   .. code-block:: python

      strategy = StrategyRewarding(problem, max_evaluations=1000)

3. **Add heuristics** to create a diverse portfolio

   .. code-block:: python

      strategy.add(Center)
      strategy.add(LatinHypercube, div=5)
      strategy.add(Random)
      strategy.add(NelderMead)

4. **Run optimization** (local Dask cluster starts automatically)

   .. code-block:: python

      strategy.start()

5. **Analyze results**

   .. code-block:: python

      print(f"Best found: {strategy.best}")
      df = strategy.results.results

Design Philosophy
-----------------

Panobbgo's design is guided by several principles:

**Modularity**
   Components are independent and interchangeable. Add, remove, or replace heuristics without affecting others.

**Extensibility**
   Easy to add custom heuristics, analyzers, or strategies by subclassing base classes.

**Composability**
   Mix and match components to create custom optimization workflows.

**Event-Driven**
   Loose coupling through the EventBus—modules communicate without direct dependencies.

**Parallel-First**
   Built for distributed evaluation on Dask clusters from the ground up.

**Research-Oriented**
   Designed for experimentation and prototyping new optimization ideas.

When to Use Panobbgo
---------------------

Panobbgo excels when you have:

✓ Expensive function evaluations (seconds to hours per evaluation)

✓ No access to gradients or derivatives

✓ Noisy or stochastic evaluations

✓ Box-constrained search space

✓ Optional constraint violations to minimize

✓ Parallel computing resources available

✓ A research or experimental mindset (willing to tune and customize)

Common application domains include:

- Hyperparameter optimization for machine learning
- Simulation calibration and parameter estimation
- Engineering design optimization (aerodynamics, structures, etc.)
- Scientific model fitting
- Industrial process optimization

When NOT to Use Panobbgo
~~~~~~~~~~~~~~~~~~~~~~~~~

Consider alternatives if you have:

✗ Access to gradients (use gradient-based methods instead)

✗ Cheap function evaluations (simple methods may suffice)

✗ Convex problems with known structure (use convex optimization)

✗ Very high dimensions (>50) without special structure

✗ Need for out-of-the-box production-ready solver (Panobbgo is research-oriented)

System Requirements
-------------------

- Python ≥ 3.8
- NumPy, SciPy, pandas, matplotlib, statsmodels, Dask
- Dask distributed cluster for parallel evaluation
- Tested on Linux and macOS (Windows should work but less tested)

Community and Support
---------------------

- **Source Code**: https://github.com/haraldschilly/panobbgo
- **Issue Tracker**: https://github.com/haraldschilly/panobbgo/issues
- **API Documentation**: http://haraldschilly.github.com/panobbgo/html/
- **Development Prompt**: See ``DEVELOPMENT_PROMPT.md`` in repository

Contributing
------------

Panobbgo is open-source (Apache 2.0 license) and welcomes contributions:

- Bug reports and feature requests
- New heuristics or analyzers
- Documentation improvements
- Benchmarking studies
- Research collaborations

See the repository for contribution guidelines.

What's Next?
------------

**If you're new**: Start with :doc:`guide_introduction` to understand black-box optimization and Panobbgo's approach.

**If you want to use it**: Jump to :doc:`guide_usage` for installation and examples.

**If you want to extend it**: Go to :doc:`guide_extending` for customization patterns.

**If you're researching**: Explore :doc:`guide_research` for the academic context and future directions.
