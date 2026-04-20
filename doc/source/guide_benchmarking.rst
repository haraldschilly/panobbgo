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
     - ~6
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


Parametrically randomised problems (planned)
---------------------------------------------

The current registry (``_make_quick_problems`` etc. in
``panobbgo/harness.py``) uses *fixed* problem instances. This is great for
A/B reproducibility but vulnerable to over-fitting: an agent that tunes a
heuristic to the specific Rosenbrock valley at ``(1, 1)`` may regress on the
next problem it encounters.

The roadmap adds a **parametric problem layer** that samples fresh instances
from a family each harness run:

- **Translation** — shift the optimum ``x*`` to a random point in the box.
- **Rotation** — for functions separable in the canonical frame (Rastrigin,
  Ackley), apply a random orthogonal transform so that axis-aligned local
  search is no longer privileged.
- **Scaling** — sample ill-conditioning factors (e.g. log-uniform in
  ``[1, 1e4]``) to stress second-order-aware heuristics.
- **Noise injection** — additive Gaussian noise on evaluations, mirroring the
  real noisy-black-box use case.
- **Dimensionality sampling** — draw ``d`` uniformly from a stated set.

Each harness run will log the per-instance parameters so that individual
failures remain reproducible; the *aggregate* score, computed across many
sampled instances, becomes a meaningful generalisation signal.

Design details: see ``planning/SELF_IMPROVEMENT_LOOP.md``.


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
4. Accept if ``delta > eps`` with statistical confidence; revert otherwise.
5. Commit and repeat.

Design and phased roadmap: ``planning/SELF_IMPROVEMENT_LOOP.md`` in the
repository.


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
- ``planning/SELF_IMPROVEMENT_LOOP.md`` — roadmap for the autonomous
  improvement loop.
