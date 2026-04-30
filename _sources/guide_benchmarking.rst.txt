Benchmarking and the Composite Score
====================================

This chapter explains *how we measure the quality of Panobbgo* — what number
we track, what it means, how to compute it, and how it feeds the planned
self-improvement loop.

.. contents::
   :local:
   :depth: 2


Why a single number?
--------------------

Black-box optimization has no shortage of metrics: success rate, ERT
(Expected Running Time), best distance to optimum, area under the convergence
curve, wall-clock time. Each captures a facet, none captures the whole.

For iterative, agent-driven self-improvement we need **one scalar** that:

1. Lives in a bounded, comparable range (``[0, 1]``).
2. Is monotonic in "doing better" — no artificial ceilings.
3. Penalises both *failing to solve* and *solving slowly*.
4. Is cheap enough to compute that it can gate every code change.

The :class:`~panobbgo.harness.BenchmarkHarness` exposes exactly this number as
``HarnessResult.composite_score``.


Definition
----------

For each optimization run we track a *convergence trace* — the sequence of
evaluation indices at which the best-so-far value improved. From the trace we
extract the **first-hit evaluation** :math:`k^\star`, defined as the first
index whose ``func_distance = |f_best − f_opt|`` is within the tolerance.

Per-run "solve fraction":

.. math::

   s = \begin{cases}
     1 - \dfrac{k^\star - 1}{B} & \text{if the run solved (} k^\star \le B \text{)} \\
     0                          & \text{otherwise}
   \end{cases}

where :math:`B` is the evaluation budget.  This is in :math:`(0, 1]` for
successful runs: ``1`` means "solved on the first evaluation",
:math:`1/B` means "solved on the very last evaluation". Failure always scores
``0``, so even a last-evaluation success is strictly better than no success.

Per **(problem, strategy)** pair we average :math:`s` across repetitions.
The final ``composite_score`` is the unweighted mean over all pairs.

See :meth:`panobbgo.harness.ProblemStrategyResult.compute_metrics` for the
exact implementation.


Interpretation
--------------

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Score
     - Meaning
   * - ``1.0``
     - Theoretical ceiling — every run solves at evaluation 1.
   * - ``0.7+``
     - Strong. Strategies consistently locate the optimum with budget left over.
   * - ``0.4 – 0.7``
     - Mixed. Typical of hard multimodal problems or under-powered strategies.
   * - ``0.1 – 0.3``
     - Weak. Usually succeeds only near the budget limit, or rarely succeeds.
   * - ``0.0``
     - Never found any optimum within tolerance on any problem.


Also reported
~~~~~~~~~~~~~

Alongside ``composite_score`` each pair exposes:

- ``success_rate`` — fraction of reps hitting tolerance.
- ``ert`` — Expected Running Time in evaluations, the BBOB/COCO standard.
  ``inf`` when no rep succeeded.
- ``best_func_distance`` / ``median_func_distance`` — absolute gap to optimum.

Together these let you diagnose *why* the composite score moved, not just that
it moved.


Running benchmarks
------------------

The CLI lives at the repository root.  Always prefix with ``uv run`` so the
correct environment is used.

.. code-block:: bash

   # Quick — ~30 seconds, 3 problems × 2 strategies × 3 reps, 75 evals each.
   uv run python benchmark_harness.py run --quick --output before.json

   # Make changes to panobbgo ...

   uv run python benchmark_harness.py run --quick --output after.json
   uv run python benchmark_harness.py compare before.json after.json

Three preset modes trade cost against statistical power:

.. list-table::
   :header-rows: 1
   :widths: 15 15 15 15 15 25

   * - Mode
     - Problems
     - Strategies
     - Reps
     - Budget
     - Typical wall-clock
   * - ``quick``
     - 3
     - 2
     - 3
     - 75
     - ~30 s
   * - ``standard``
     - 8
     - ~7
     - 5
     - 200
     - few minutes
   * - ``full``
     - 11
     - ~10
     - 10
     - 500
     - ~1 hour

Reproducibility
~~~~~~~~~~~~~~~

Seeds are deterministic: each ``(problem, strategy, rep)`` triple derives its
seed via SHA-256 from a base seed (default ``42``). Re-running with the same
base seed produces byte-identical convergence traces. See
:meth:`panobbgo.harness.BenchmarkHarness._derive_seed`.


The ``compare`` workflow
------------------------

``compare before.json after.json`` reports:

- Composite-score delta and relative percent change.
- Per-pair improved / degraded / unchanged classification, gated by ``--eps``
  (default ``0.01``).
- Pairs that exist in only one file (e.g. you added a new strategy).
- With ``--fail-on-regression``, the process exits ``2`` when the candidate
  scores worse.  Useful for CI gating and automated accept/revert loops.


Statistical acceptance rule
---------------------------

The naive ``|Δ| > eps`` gate is fast but fragile — at quick-mode sample sizes
a single lucky/unlucky run can flip it.  For rigorous gating (the kind an
autonomous self-improvement loop needs), add the ``--statistical`` flag:

.. code-block:: bash

   uv run python benchmark_harness.py compare before.json after.json \
       --statistical --fail-on-regression

The statistical rule follows ``planning/SELF_IMPROVEMENT_LOOP.md`` §6.2.
For every ``(problem, strategy)`` pair present on both sides the per-run
**solve fractions** — the same quantity averaged into the composite score —
are bootstrap-resampled to produce a confidence interval on the mean
difference.  A single bootstrap index yields one composite delta (the mean
of per-pair deltas), and the percentile interval over ``n_boot`` such
indices is the composite CI.

The decision is **accept** iff *all* of:

- ``Δ > eps_accept`` — moved in the right direction beyond noise.
- ``CI_low > 0`` — the improvement is statistically distinguishable from
  zero at the chosen confidence level.
- ``min_i Δ_i > −eps_regress`` — no individual pair crashes, even if the
  composite improved overall.

When ``--fail-on-regression`` is combined with ``--statistical`` the exit
code is ``2`` whenever the rule rejects (composite was noisy, regressed, or
a pair blew up).  Every knob has a flag:

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Flag
     - Default
     - Meaning
   * - ``--eps-accept``
     - ``0.005``
     - Minimum composite delta required.
   * - ``--eps-regress``
     - ``0.05``
     - Maximum tolerated per-pair regression.
   * - ``--n-boot``
     - ``10000``
     - Bootstrap resamples for the CI.  Fewer = faster, noisier CIs.
   * - ``--confidence``
     - ``0.95``
     - Confidence level.
   * - ``--stat-seed``
     - ``42``
     - RNG seed for reproducible bootstraps.

With ``--json --statistical`` the emitted payload carries a
``statistical`` block with the composite verdict, the overall CI, the worst
regressing pair, and per-pair CIs — everything an agent needs to drill
into a rejection:

.. code-block:: json

   {
     "statistical": {
       "accept": false,
       "delta": -0.0002,
       "ci_low": -0.0534,
       "ci_high":  0.0527,
       "worst_pair_regression": -0.34,
       "worst_pair": ["Rastrigin_2D", "UCB_Diverse"],
       "reasons": [
         "lower CI bound -0.0534 ≤ 0 — improvement not statistically distinguishable from noise"
       ],
       "per_pair": [ ... ]
     }
   }

The programmatic API is
:func:`panobbgo.harness.statistical_accept`.  The result is a
:class:`~panobbgo.harness.StatisticalDecision` with the same fields as the
JSON payload, plus a ``print_summary()`` method for human-readable output.

Stability note: the composite-score formula remains a stable contract.
The statistical rule is a *gate* on top of that formula — adding it does not
change the underlying number.


When to run which mode
----------------------

- **During development** — ``--quick`` after each meaningful change.  Fast
  enough to keep you honest without hurting flow.
- **Before opening a PR** — ``--standard``.  Saves the result JSON as a build
  artefact so reviewers can compare.
- **Before merging a significant algorithmic change** — ``--full``, on a
  machine you are not actively using.


Pitfalls and statistical caveats
--------------------------------

The composite score is **noisy**. Three symptoms to watch for:

1. **Few reps** — ``--quick`` uses only 3 reps per pair. A delta of ``±0.02``
   is firmly within noise. Treat quick-mode comparisons as trend signals, not
   proof.
2. **Lucky seeds** — if a regression vanishes when you change the base seed
   (``--seed``), the effect was seed-specific, not a real improvement.
3. **Problem-specific over-fitting** — improving on a fixed problem set does
   not imply generalisation. This is the *central motivation* for the
   parametrically randomised battery described below.

Recommended practice: run the same comparison at two different base seeds
before accepting a ``+0.01`` to ``+0.03`` delta.


Parametrically randomised problems
----------------------------------

The fixed registry (``_make_quick_problems`` etc. in
``panobbgo/harness.py``) is great for A/B reproducibility but vulnerable to
over-fitting: an agent that tunes a heuristic to the specific Rosenbrock
valley at ``(1, 1)`` may regress on the next problem it encounters.  The
harness therefore ships a **parametric problem layer** (Phase 3 of the
self-improvement loop) that samples fresh transformed instances per
repetition, turning ``composite_score`` into a Monte-Carlo estimate of
*expected* performance on a problem family.

Usage
~~~~~

Add ``--randomize`` to ``run`` or ``list``:

.. code-block:: bash

   uv run python benchmark_harness.py list --randomize
   uv run python benchmark_harness.py run --randomize --output rand_before.json
   # Make changes ...
   uv run python benchmark_harness.py run --randomize --output rand_after.json
   uv run python benchmark_harness.py compare rand_before.json rand_after.json --statistical

The ``--randomize-iteration`` flag (default ``0``) mixes an iteration
index into the instance seed.  Within one iteration, ``before`` and
``after`` runs see **identical** sampled instances (apples-to-apples);
across iterations the instances intentionally differ, so repeatedly
"winning" the same iteration is not enough — only sustained wins across
many iterations are real improvements.

What gets randomised
~~~~~~~~~~~~~~~~~~~~

For each :class:`~panobbgo.harness_randomized.ProblemFamily`, the sampler
composes four transforms (filtered by per-family capability flags):

.. list-table::
   :header-rows: 1
   :widths: 18 27 55

   * - Transform
     - Parameter
     - Rationale
   * - Translation
     - :math:`x^\star \sim U(\text{interior box})`
     - Kills hard-coded-centre exploitation; optimum lives away from edges.
   * - Rotation
     - Haar-random :math:`Q \in O(d)`
     - Breaks axis-aligned local-search advantage on separable functions.
   * - Scaling
     - Diagonal :math:`\Lambda`, :math:`\log_{10}\kappa \sim U[0, c_{\max}]`
     - Stresses second-order / ill-conditioning heuristics.
   * - Noise
     - :math:`\tilde f(x) = f(x) + \sigma \varepsilon`
     - Matches the noisy-black-box use case.

The composite transform is

.. math::

   \tilde f(x) = f_{\text{base}}\bigl(Q \, \Lambda \, (x - x^\star)
                                     + y^\star_{\text{base}}\bigr)
                + \sigma \varepsilon,

and by construction :math:`\tilde f(x^\star) = f_{\text{opt}}` so
``known_optima`` for the transformed problem is ``{"x": x*, "fx": f_opt}``
— the existing ``func_distance`` / ``ert`` / ``composite_score`` logic
works unchanged.

Default families
~~~~~~~~~~~~~~~~

:func:`panobbgo.harness_randomized.make_default_families` returns four
families covering the main optimisation challenges:

- ``Rastrigin_family`` — separable, highly multimodal (translate + rotate + scale).
- ``Ackley_family`` — non-separable, multimodal with a funnel (translate + rotate + scale).
- ``Rosenbrock_family`` — non-separable, banana valley (translate + scale).
- ``DeJong_family`` — smooth convex sphere (all transforms; the baseline sanity case).

Schwefel and Griewank are intentionally excluded from the default set —
Schwefel's optimum sits near the box boundary and Griewank's oscillation
couples tightly to the coordinate axes, so rotation pushes ``y`` off the
function's sensible domain.

Reproducibility
~~~~~~~~~~~~~~~

All randomness flows from a single 32-bit seed derived via SHA-256 from
``(base_seed, iteration_id, family_name, rep)`` — the same scheme the
harness already uses for strategy seeds.  A regression flagged by the loop
is deterministically reproducible from the printed tuple:

.. code-block:: python

   from panobbgo.harness_randomized import (
       make_default_families, RandomizedProblemSpec
   )
   fams = make_default_families()
   spec = RandomizedProblemSpec(
       fams[0], iteration_id=5, base_seed=42, max_evaluations=200
   )
   prob = spec.create_problem_for_rep(3)
   params = spec.last_sampled_params()  # dim, translation, rotation_trace, ...

Design details: ``planning/SELF_IMPROVEMENT_LOOP.md`` §4.  Implementation:
:mod:`panobbgo.harness_randomized`.  Tests:
``tests/test_harness_randomized.py``.


Absolute baselines
------------------

The harness runs *Panobbgo strategies vs. Panobbgo strategies* by default.
To judge Panobbgo in **absolute** terms, the
:mod:`panobbgo.harness_baselines` module plugs three external reference
solvers into the same :class:`~panobbgo.benchmark.StrategySpec` interface:

- ``Baseline_Random`` — uniform random search (the **floor**: any serious
  optimizer should beat it most of the time).
- ``Baseline_SciPyDE`` — ``scipy.optimize.differential_evolution``.
- ``Baseline_SciPyAnneal`` — ``scipy.optimize.dual_annealing``.

Enable them with the ``--baselines`` flag on ``run`` or ``list``:

.. code-block:: bash

   uv run python benchmark_harness.py list --standard --baselines
   uv run python benchmark_harness.py run --standard --baselines --output standard_with_refs.json
   uv run python benchmark_harness.py score standard_with_refs.json

They appear in the results table alongside Panobbgo strategies — the
composite score still averages over every ``(problem, strategy)`` pair,
so a fair comparison should either *include baselines on both sides* of
a ``compare`` call or *exclude them from both*.

Design
~~~~~~

Baselines are thin adapters that implement the subset of the strategy
surface the harness actually uses (``.config.max_eval``, ``.start()``,
``.best``, ``.results.results``).  They do **not** subclass
:class:`~panobbgo.core.StrategyBase` — spinning up the event bus and Dask
machinery for a single-shot external solver would be pure overhead.

The shared objective wrapper enforces a **hard evaluation budget**: it
raises :class:`~panobbgo.harness_baselines._BudgetExhausted` once
``max_eval`` evaluations are recorded, giving baselines the same budget
contract as Panobbgo strategies.  Convergence traces are extracted from
the wrapper's log, and the results DataFrame carries the same MultiIndex
columns (``("fx", 0)``, ``("who", 0)``, …) the harness expects.

CMA-ES as a baseline
~~~~~~~~~~~~~~~~~~~~

The pure CMA-ES reference is already available **inside** Panobbgo via
the :class:`~panobbgo.heuristics.cma_es.CMAES` heuristic, used by the
``CMAES_Portfolio`` and ``IPOP_CMAES`` strategies.  That gives a fair
internal comparison without the extra dependency on ``pycma``.  A
dedicated ``pycma`` adapter is easy to add if a round-trip against the
upstream reference becomes valuable.

See ``panobbgo/harness_baselines.py`` for the full interface and
``tests/test_harness_baselines.py`` for the guarantees.


Self-improvement loop
---------------------

The benchmark harness is the *measurement substrate* for an autonomous
improvement loop:

1. Capture baseline composite score.
2. Propose a change (heuristic tweak, new analyzer, parameter retune).
3. Apply in an isolated branch; run the benchmark.
4. Accept with :func:`~panobbgo.harness.statistical_accept` (or the
   ``--statistical`` CLI gate) — the bootstrap-CI rule above; revert
   otherwise.
5. Commit and repeat.

Design and phased roadmap: ``planning/SELF_IMPROVEMENT_LOOP.md`` in the
repository.

The MVP driver is shipped as :mod:`panobbgo.self_improve` plus the
``scripts/self_improve.py`` CLI:

.. code-block:: bash

   # 5 quick iterations, randomized battery, ledger at the default path
   uv run python scripts/self_improve.py run --iterations 5

   # Long unattended run with the anti-cherry-pick guard every 10 iters
   uv run python scripts/self_improve.py run --mode standard \
       --iterations 100 --guard-interval 10 --guard-eps-ladder 0.02

   # Pretty-print a previous ledger (counts accepts, guards, rollbacks)
   uv run python scripts/self_improve.py summary

Anti-cherry-pick guard (§6.3)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Even with the parametrically randomized battery, a sequence of "lucky"
instance draws can inflate per-iteration ``after`` scores enough to
clear the bootstrap CI.  The **anti-cherry-pick guard** mitigates this
by periodically re-measuring the *top of the accepted ladder* on a
fresh randomized seed and rolling the ladder back if the score has
drifted.

Algorithm:

1. Maintain a ladder of accepted spec lists: an entry stores the
   iteration that produced it, the spec snapshot, and the
   ``last_validated_score`` (the composite that originally got it
   promoted, refreshed every time the guard validates the entry).
2. Every :attr:`~panobbgo.self_improve.LoopConfig.guard_interval`
   iterations, re-measure the top entry on
   ``randomize_iteration = iteration + guard_iteration_offset`` (a
   large offset, ``1_000_000`` by default, keeps the guard's instance
   stream independent from the regular iteration stream).
3. If the re-measured composite is below the stored
   ``last_validated_score`` by more than
   :attr:`~panobbgo.self_improve.LoopConfig.guard_eps_ladder`, pop the
   entry and re-measure the next one down — repeat until a stable
   entry is found or the seed is reached.  The seed entry is **never**
   popped: it is the trusted fallback by definition.
4. Each guard check writes a
   :class:`~panobbgo.self_improve.LoopGuardRecord` to the ledger
   (``record_type = "guard"``) so audits can replay both signals.

The guard is **disabled by default** (``guard_interval = 0``) for
backward compatibility.  Bump it to ``5`` or ``10`` for unattended
multi-hour runs where instance cherry-picking is the dominant
risk.

Programmatic use:

.. code-block:: python

   from panobbgo.self_improve import LoopConfig, SelfImprover

   cfg = LoopConfig(
       iterations=50,
       mode="standard",
       guard_interval=10,
       guard_eps_ladder=0.02,
   )
   iter_records, guard_records = SelfImprover(cfg).run_with_guard_records()


Extending the harness
---------------------

Adding a problem
~~~~~~~~~~~~~~~~

1. Append a :class:`~panobbgo.benchmark.ProblemSpec` to the appropriate
   ``_make_*_problems`` factory in ``panobbgo/harness.py``.
2. Provide ``known_optima`` and a ``tolerance`` that is achievable for the
   mode's budget but not trivially so.
3. Run ``--quick`` to confirm the new problem is well-formed.

Adding a strategy
~~~~~~~~~~~~~~~~~

1. Append a :class:`~panobbgo.benchmark.StrategySpec` to
   ``_make_*_strategies``.
2. The strategy must be deterministic given a seed (no external randomness
   outside the seeded ``numpy`` generator).
3. Validate with ``benchmark_harness.py list --standard``.

Adding a metric
~~~~~~~~~~~~~~~

Per-pair aggregates live on
:class:`~panobbgo.harness.ProblemStrategyResult`. Add a field, populate it in
``compute_metrics``, surface it in ``HarnessResult.print_summary`` and
``cmd_score``'s JSON path. Do **not** change the composite score formula
without an ADR — historical comparisons depend on its stability.


See also
--------

- :mod:`panobbgo.harness` — the harness implementation.
- :mod:`panobbgo.harness_baselines` — external reference strategies
  (Random, SciPy DE, SciPy dual annealing).
- ``benchmark_harness.py`` — CLI.
- ``tests/test_harness.py`` — harness test suite (60+ tests).
- ``tests/test_harness_baselines.py`` — baseline adapter tests.
- ``tests/test_harness_stats.py`` — statistical acceptance rule tests.
- ``planning/SELF_IMPROVEMENT_LOOP.md`` — roadmap for the autonomous
  improvement loop.
