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
   * - ``--paired`` / ``--unpaired``
     - auto
     - Force the paired (rep-aligned) or independent bootstrap scheme;
       see *Paired vs unpaired bootstrap* below.

Paired vs unpaired bootstrap
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Under ``--randomize`` the harness keeps reps **instance-aligned by index**
— rep ``i`` on the ``before`` side and rep ``i`` on the ``after`` side
are evaluated on the *same* sampled problem instance (the SHA-256 stream
is keyed on ``(base_seed, randomize_iteration, family, rep)``).  The
per-rep deltas are therefore strongly correlated, and the statistically
efficient sampler is the **paired bootstrap** — draw one shared resample
index and apply it to both sides, equivalent to bootstrapping the per-rep
delta vector ``a_frac − b_frac``.

The historical default of an **unpaired** sampler (independent resamples
on each side) discards the within-rep correlation and inflates the CI by
roughly the within-side variance, often leaving a genuine improvement
indistinguishable from noise:

.. code-block:: text

   # Strongly correlated reps: every rep solves 5 evals earlier on the
   # same instance.  The paired sampler accepts; the unpaired one does
   # not.
   paired:   Δ=+0.0500  CI=[+0.0500, +0.0500]  width=0.0000  → ACCEPT
   unpaired: Δ=+0.0500  CI=[−0.2100, +0.3300]  width=0.5400  → REJECT

The ``--paired`` and ``--unpaired`` flags toggle the scheme explicitly:

.. code-block:: bash

   # Paired — recommended for any --randomize run or any other
   # comparison where reps are instance-aligned by index.
   uv run python benchmark_harness.py compare before.json after.json \
       --statistical --paired

   # Unpaired — required when reps are NOT instance-aligned (e.g. when
   # comparing two ledgers built with different base_seed values).
   uv run python benchmark_harness.py compare before.json after.json \
       --statistical --unpaired

Without either flag the rule auto-selects: paired when at least one
shared pair has ``n_before == n_after``, falling back to unpaired
otherwise.  The auto-detect default is the right choice for the
randomized harness and a safe no-op for the asymmetric-rep edge cases
the unpaired scheme was originally written to handle.

The chosen scheme is reported on the
:class:`~panobbgo.harness.StatisticalDecision`:
``decision.paired`` is ``True`` if the paired sampler fired on at least
one pair, ``False`` otherwise.  ``print_summary()`` and the
``--json`` payload include the field.

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

Stratified dimension sampling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When a :class:`~panobbgo.harness_randomized.ProblemFamily` declares
``dim_choices = (2, 5, 10)``, naive random selection would produce a
different mix of dimensions on each loop iteration — and since
higher-dim instances are systematically harder, that mix-noise
contaminates the cross-iteration deltas the bootstrap CI in
:func:`panobbgo.harness.statistical_accept` (§6.2 of the
self-improvement loop) operates on.

The default ``stratify_dims=True`` flag eliminates that noise source by
construction.
:meth:`~panobbgo.harness_randomized.ProblemFamily.stratified_dim_for_rep`
assigns the dim **cyclically** by repetition index so any contiguous
block of ``len(dim_choices)`` reps covers every declared dim exactly
once:

.. code-block:: python

   from panobbgo.harness_randomized import ProblemFamily
   from panobbgo.lib.classic import DeJong

   fam = ProblemFamily(name="dejong3", base_class=DeJong,
                       dim_choices=(2, 5, 10))  # stratify_dims=True
   [fam.stratified_dim_for_rep(rep) for rep in range(7)]
   # → [2, 5, 10, 2, 5, 10, 2]

When the family's dim-choice tuple has only one element (the default
battery), stratification is a no-op and the per-rep dim is constant.
For backwards compatibility with older ledgers, set
``stratify_dims=False`` to recover the legacy uniform-draw behaviour.

The result of a stratified sample is reflected in
``RandomizedProblemSpec.last_sampled_params()`` via a
``stratified_dim: bool`` field, useful for ledger introspection.

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
   # — also surfaces the V2 §12.4 trend block (one row per loop run),
   # the top-N / bottom-N bandit posteriors, and the inactivity
   # telemetry; see the "Summary trend block" subsection below.
   uv run python scripts/self_improve.py summary
   # Show the top 20 / bottom 10 mutation-rule posteriors (default 10/5)
   # and require at least 5 informative attempts per rule (default 3):
   uv run python scripts/self_improve.py summary \
       --top-n 20 --bottom-n 10 --min-attempts 5

   # Use the loop-tuned seed registry (V2 §9.1) so the dormant catalog
   # mutation rules actually fire on the rule-bearing DE / PSO /
   # RegionUCB / LBFGSB+COBYQA / Restart families.  Lifts catalog
   # kwarg-rule coverage from 4 / 44 (quick seed) to 44 / 44 — see the
   # "Loop registry" section below.
   uv run python scripts/self_improve.py run --iterations 30 \
       --registry loop --adaptive --structural

Loop registry (V2 §9.1)
~~~~~~~~~~~~~~~~~~~~~~~

The default ``quick`` registry — :func:`panobbgo.harness._make_quick_strategies`
— ships two compact specs (``RoundRobin_Random`` and
``Rewarding_Diverse``) that explicitly set only ``Sobol`` / ``Nearby`` /
``Sensitivity`` kwargs.  Mutation rules in
:func:`panobbgo.self_improve.default_catalog` are gated by the
"param already in kwargs" predicate
(:func:`panobbgo.self_improve._find_targets`), so every L-SHADE /
jSO / NL-SHADE-RSP / NL-SHADE-LBC / LSHADE-EpSin / PSO / RegionUCB /
COBYQA / LBFGSB / Restart rule sits dormant against this seed —
four rules fire out of 44 (~9 % catalog reach).

:func:`panobbgo.harness._make_loop_strategies` exists to exercise the
dormant catalog.  It returns the two quick specs **plus** five
compact family specs, every tuneable kwarg explicit at the
constructor default so every catalog rule on the matching class is
immediately applicable:

* ``Loop_DE_Family`` — one Rewarding strategy carrying all five DE
  variants (L-SHADE, jSO, NL-SHADE-RSP, NL-SHADE-LBC, LSHADE-EpSin)
  at ``NP_init = 15``.  Activates every numeric and categorical
  rule across the DE family (~16 rules after the 2026-06-24 LBC
  catalog consolidation), including the L-SHADE ``F_schedule`` and
  ``archive_factor`` categorical toggles, the jSO ``p_best_max``
  regime arm, the NL-SHADE-RSP ``k_rank`` regime arm, and the
  NL-SHADE-LBC ``lbc_regime`` composite categorical arm
  (``"cec2022"`` / ``"lshade"`` / ``"flat"`` / ``"aggressive"``)
  that replaced the five per-field LBC float rules with one
  literature-motivated joint search across the Lehmer-mean exponent
  / spread tuple.
* ``Loop_PSO`` — LatinHypercube + PSO + NelderMead.  PSO ships
  ``NP=15 / w=0.7298 / w_end=0.4 / stagnation_threshold=10 /
  topology="gbest"`` so every PSO rule fires — including the
  four-way ``topology`` categorical and the stochastic-K
  ``stagnation_threshold`` rule shipped 2026-06-05.
* ``Loop_RegionUCB`` — the diverse heuristic mix plus a
  ``RegionUCB`` arm with ``ucb_c / gauss_fraction / gauss_scale``
  explicit (the three 2026-06-08 rules).
* ``Loop_LocalSearch`` — LatinHypercube + COBYQA + LBFGSB +
  NelderMead.  Activates the COBYQA trust-region rules and the
  ``LBFGSB.max_starts`` rule shipped 2026-06-06.
* ``Loop_Restart`` — a CMA-ES strategy with the :class:`Restart`
  analyzer at explicit ``patience / restart_strategy /
  max_restarts``.  Activates all three :class:`Restart` rules
  including the ``restart_strategy`` categorical arm shipped
  2026-06-07.

Opt in with ``--registry loop`` on
``scripts/self_improve.py run`` or
:attr:`panobbgo.self_improve.LoopConfig.registry` ``= "loop"``.  The
registry is independent of ``--mode`` — quick / standard / full
budgets are honoured but the seed specs are the same.  Default
``registry="default"`` preserves the historical mode-based
selection byte-for-byte, so existing CLI invocations are
byte-identical.

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

Same-night confirmation gate (§6.4)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The anti-cherry-pick guard catches drift *after* a screening accept
has already landed on the ladder.  Under the V2 §2.2 diagnosis ("15/16
guard checks rolled the ladder back; all hold-out records ran on an
empty ladder") this is too late: the ladder churns indefinitely
because most screening accepts are upward-noise spikes the guard
subsequently revokes, codify-scan never sees a durable signal, and
the planning doc's success criterion 3 ("zero guard rollbacks of
*confirmed* accepts") is structurally unreachable when no confirmation
step exists.

The **same-night confirmation gate** inverts this — promotion requires
confirmation *before* the accept is recorded.  Setting
:attr:`~panobbgo.self_improve.LoopConfig.confirm_accepts` ``= True``
(CLI ``--confirm-accepts``) makes the loop:

1. Re-measure every screening-accepted candidate on a fresh
   ``randomize_iteration`` (``iteration +
   :attr:`~panobbgo.self_improve.LoopConfig.confirm_iteration_offset```,
   default ``500_000``).  Sits between the regular iteration stream
   (``0..N``) and the guard's offset (``1_000_000``) so the three
   fresh-seed streams never collide at realistic iteration counts.
2. When at least one hold-out base_seed is configured (via
   :attr:`~panobbgo.self_improve.LoopConfig.holdout_base_seed` or
   :attr:`~panobbgo.self_improve.LoopConfig.holdout_base_seeds`),
   additionally re-measure on the *first* hold-out seed at the same
   fresh iteration_id.  Only the first seed is used per iteration to
   cap per-iteration cost at ``≤ 3×`` screening regardless of how
   many hold-out seeds the end-of-loop drift check walks.
3. Pool the screen + confirm (+ optional hold-out) measurements via
   :func:`~panobbgo.self_improve._pool_harness_results` and re-run
   :func:`~panobbgo.harness.statistical_accept` on the pooled sample.
4. Promote only when the pooled bootstrap CI still clears the same
   gate (``Δ > eps_accept``, ``ci_low > 0``, no catastrophic per-pair
   regression).  A screening noise spike can no longer drive a
   promotion because the confirmation batch is independent and the
   pooled CI rules it out.
5. When the gate overturns a screening accept, append a
   :class:`~panobbgo.self_improve.LoopConfirmRecord` (``record_type =
   "confirm_reject"``) to the ledger carrying screen / confirm /
   pooled scores and CI; the accompanying
   :class:`~panobbgo.self_improve.LoopIterationRecord` carries
   ``accepted = False`` and ``confirmed = False`` so codify-scan can
   distinguish noise-spike accepts from durable ones without
   re-deriving the verdict.

The ``LoopIterationRecord.confirmed`` field is a three-valued summary
of the gate's verdict:

* ``None`` — no confirmation step ran (the V1 path:
  ``confirm_accepts = False``, or the iteration was a skip / no-op,
  or screening already rejected — the gate only runs after a
  screening accept).
* ``True`` — confirmation step ran and the pooled CI cleared the gate;
  the candidate was promoted.
* ``False`` — confirmation step ran and the pooled CI did not clear
  the gate; the candidate was *not* promoted and a companion
  ``confirm_reject`` record describes the overturned screening
  accept.

Critically, the bandit reward path consumes the *post-confirmation*
pooled decision, so an arm that consistently produces screening
noise-spike accepts collects the reject-regime reward (binary:
``0``; graded: ``clip(0.5 + pooled_Δ/(4·eps), 0, 0.5)``) rather
than the full-accept reward the screening alone would have produced.
The Thompson posterior on such an arm decays toward the reject
regime over a handful of confirmations, where binary-mode V1
would have inflated it permanently.

The gate is **disabled by default** (``confirm_accepts = False``) to
keep existing CLI invocations byte-identical.  Recommended
unattended cron preset (enables the gate; uses the planning-doc
default offset; pairs with a guard at the longer 10-iteration
cadence so the gate handles screen-time noise and the guard handles
the rare cross-night drift on confirmed accepts):

.. code-block:: bash

   uv run python scripts/self_improve.py run --iterations 35 \
       --adaptive --bandit-reward graded --registry loop \
       --confirm-accepts \
       --guard-interval 10 \
       --holdout-base-seeds 1234,5678

The **live nightly cron**
(``.github/workflows/self_improve_nightly.yml``) ships every flag in
the preset above *except* ``--confirm-accepts``, which carries
~2-3× the per-iteration cost (one re-measurement on a fresh
``randomize_iteration`` plus one per hold-out seed).  The §9.5
step 5 of ``planning/SELF_IMPROVEMENT_LOOP.md`` schedules the
``--confirm-accepts`` flip after a manual ``workflow_dispatch``
A/B that measures the confirm-reject rate before flipping the cron
permanently.  Until then the live cron operates in *promote-on-
screening* mode and §2.2 "Accept → rollback churn" remains the
visible symptom in the ledger.

Programmatic use:

.. code-block:: python

   from panobbgo.self_improve import LoopConfig, SelfImprover

   cfg = LoopConfig(
       iterations=50,
       mode="standard",
       confirm_accepts=True,
       holdout_base_seeds=(1234, 5678),
       guard_interval=10,
       adaptive_sampling=True,
       bandit_reward_shaping="graded",
   )
   iter_records, guard_records, holdout_records = SelfImprover(cfg).run_full()
   confirmed = [r for r in iter_records if r.confirmed is True]
   overturned = [r for r in iter_records if r.confirmed is False]

Inactivity-guarded eps_accept relaxation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Long unattended ledgers tend to show very low accept rates (the cron
that motivated this knob recently recorded *1 accept in 86 iterations*).
That is small enough that the adaptive sampler's posterior remains
close to its prior for most arms — defeating the point of bandit
sampling.  The mitigation is to temporarily lower the accept threshold
after the loop has gone too long without an accept, then re-tighten on
the next accept.

Three knobs on :class:`~panobbgo.self_improve.LoopConfig` (mirrored as
``--inactivity-relax-after`` / ``--inactivity-relax-factor`` /
``--inactivity-min-eps-accept`` on the CLI):

* :attr:`~panobbgo.self_improve.LoopConfig.inactivity_relax_after`
  (default ``0`` = disabled).  Number of consecutive non-accept
  iterations after which the relax rule starts to fire.  Both
  *skip*-iterations (no applicable mutation) and *reject*-iterations
  count toward the streak — the bandit cares about observed accepts,
  not about how the loop reached "no accept".
* :attr:`~panobbgo.self_improve.LoopConfig.inactivity_relax_factor`
  (default ``0.5``).  Multiplicative factor applied to
  ``eps_accept`` per relaxation step.  Each additional
  ``inactivity_relax_after`` block of non-accepts halves the threshold
  again, so after ``k`` blocks the effective threshold is
  ``eps_accept · factor^k``.
* :attr:`~panobbgo.self_improve.LoopConfig.inactivity_min_eps_accept`
  (default ``0.001``).  Floor on the relaxed threshold so a relaxed
  accept still beats a baseline-grade signal.  Picked to match the
  bootstrap CI's noise floor at typical quick-mode rep counts.

Behaviour:

* Disabled (``inactivity_relax_after = 0``) ⇒
  :func:`~panobbgo.self_improve.LoopConfig.effective_eps_accept` is a
  constant equal to ``eps_accept``, byte-identical to the historical
  behaviour.
* Streak length ``s`` ⇒ effective threshold is
  ``max(eps_accept · factor^(s // after), min_eps_accept)``.
* On every accept the streak resets to ``0`` and the next iteration
  starts again at the full ``eps_accept`` — the relaxation is
  genuinely temporary.

Each iteration records both
:attr:`~panobbgo.self_improve.LoopIterationRecord.effective_eps_accept`
and :attr:`~panobbgo.self_improve.LoopIterationRecord.iters_since_accept`
so an auditor can replay the relax rule deterministically.  Old
records (written before the feature shipped) carry ``None`` for both
fields and continue to load unchanged.

Recommended unattended preset, mirroring the planning doc's §10
"inactivity-guarded loop productivity" sketch:

.. code-block:: bash

   uv run python scripts/self_improve.py run --iterations 100 \
       --adaptive --adaptive-prime-from-ledger --structural \
       --guard-interval 10 \
       --inactivity-relax-after 10 \
       --inactivity-relax-factor 0.5 \
       --inactivity-min-eps-accept 0.001

The §11 success criteria pin ``eps_accept`` at a fixed level, so a
chronic relaxation would silently shift the loop's "improvement" bar.
The floor + per-iteration ledger field keep this honest: a reviewer
can grep the ledger for any record whose
``effective_eps_accept`` is below ``eps_accept`` and audit those
accepts separately.

No-op detection (§12.4)
~~~~~~~~~~~~~~~~~~~~~~~

The V2 diagnosis in :doc:`SELF_IMPROVEMENT_LOOP <../../planning/SELF_IMPROVEMENT_LOOP>`
identified "34% of mutations measure Δ = exactly 0.0000" as the
dominant V1 failure mode — proposals targeting kwargs whose effect is
invisible at the quick-mode budget produce baseline and candidate
measurements whose per-pair scores are *bit-identical*.  Those
iterations carry zero information about whether the mutation rule
helps or hurts: pulling the bandit arm on them mis-trains the Beta
posterior toward "this rule keeps rejecting" even though the rule's
value is undetermined.

Two coordinated guards close the loop:

* :class:`~panobbgo.self_improve.LoopIterationRecord` carries a
  ``no_op: bool`` field (default ``False``).  After both
  measurements the loop checks whether the per-(problem, dim,
  strategy) ``score`` maps from ``baseline_result`` and
  ``candidate_result`` are equal across every key
  (:func:`panobbgo.self_improve._is_no_op`).  When they are, the
  record sets ``no_op=True``, ``reason_skipped="no_op"`` and
  ``accepted=False`` regardless of the bootstrap verdict on the
  (vacuously zero) delta.
* :meth:`~panobbgo.self_improve.AdaptiveMutationSampler.discard_outcome`
  clears the sampler's pending ``last_rule_key`` *without*
  incrementing the arm's ``n_attempts`` — the same end-state as
  :meth:`record_outcome` but with no posterior side-effect.  The
  driver loop calls this instead of :meth:`record_outcome` on no-op
  iterations.  :meth:`prime_from_ledger` skips records whose
  ``no_op`` field is ``True``, so resuming from a ledger preserves
  the same gating contract.

The summary view (``scripts/self_improve.py summary``) surfaces a
separate ``no-op=N`` bucket and computes the accept rate against the
**informative** denominator (decided − no-op).  An operator reading
the §12.3 daily routine can now distinguish "the bandit is starved
because most proposed rules are dormant on the seed registry" from
"the bandit is starved because every legitimate proposal got
rejected".  The first calls for a registry change (e.g.
``--registry loop`` — see the *Loop registry* subsection above); the
second calls for a metric change (V2 §9.5 step 2,
``--metric aocc``).

Legacy ledgers (pre 2026-06-12) carry no ``no_op`` key on disk; the
JSONL load path returns ``None`` / ``False`` for missing keys so
prior records classify as informative — matching the historical
semantics exactly.  The :class:`LoopIterationRecord` dataclass
default of ``False`` preserves the same contract for direct
construction.

Summary trend block (§12.4)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The §12.3 daily routine reads ``planning/self_improve_summary.txt``
"at-a-glance" to answer three questions: *is the loop accepting
anything tonight?  Which mutation arms are paying off?  Is the
inactivity-relax knob doing anything?*  The pre-2026-06-16 summary
view was an ever-growing wall of per-record lines (200 iterations × N
nights × M hold-out records) that surfaced the totals but never the
per-night dispersion or the bandit's posterior state.

``scripts/self_improve.py summary`` now renders three additive
sub-blocks **after** the existing per-record sections so the legacy
output is preserved byte-for-byte while the trend signal is one
screen of text:

1. **Trend block** — one row per loop run (oldest first) with
   columns: ``date``, ``seed``, ``mode``, ``iters``, ``dec`` (decided
   = non-skip), ``acc``, ``nop`` (no-op count), ``best_Δ``,
   ``seed_score`` (``baseline_score`` of the first record of the
   run, sourced from a real measurement so the column tracks per-night
   signal rather than a recomputed average).  Runs are detected by
   ``iteration <= prev_iteration`` boundaries
   (:func:`_group_runs`) — the natural inverse of the
   :meth:`SelfImprover.run` writer which restarts the counter at
   ``0`` on every invocation.
2. **Bandit posteriors** — top-N / bottom-N mutation-rule posteriors
   ranked by graded ``mean_reward`` descending (tie-break by
   ``n_attempts`` so dense evidence beats sparse evidence at the same
   mean).  The replay
   (``_replay_bandit_posteriors``) runs through the same
   :func:`panobbgo.self_improve._proposal_rule_key` collapse used by
   :meth:`AdaptiveMutationSampler.prime_from_ledger` (default
   ``per_class_structural=False``), so the summary view matches what
   a freshly-primed nightly bandit would carry into the next run.
   On graded-reward ledgers the ranking carries the full §7.4 signal
   (barely-confirmed accepts at ``~0.5``, honest near-miss rejects
   at ``~0.5``, clearly-harmful rejects at ``~0``); on legacy
   binary-reward ledgers ``mean_reward`` collapses to
   ``accept_rate`` so pre-2026-06-13 evidence is rendered without
   distortion.  Configurable via ``--top-n`` (default ``10``),
   ``--bottom-n`` (default ``5``), and ``--min-attempts`` (default
   ``3`` — filters out one-shot rules so the leaderboard reflects
   actual evidence).
3. **Inactivity** — the inferred ``eps_accept`` base (the maximum
   observed ``effective_eps_accept``, since relaxation only
   *decreases* the threshold and is re-tightened back to the base on
   every accept), the longest accept drought (max
   ``iters_since_accept``), the relaxed-accept count
   (``effective_eps_accept < eps_base``), and the mean decay factor
   at the moment of accept.  Silently no-ops on legacy ledgers
   (pre-2026-05-30) whose iteration records carry neither field.

All three blocks silently no-op on empty input; the bandit block
prints a friendly note ("no rules with >= N informative attempts")
rather than an empty table when the ``--min-attempts`` filter rules
every rule out.  See the 2026-06-16 entry in
``planning/SELF_IMPROVEMENT_LOG.md`` for the rationale and the test
plan.

Cross-night codify-scan (§9.3 / §9.5 step 4)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The §11 V2 success criteria are gated on **codify PRs** — the
persistence mechanism by which short-lived in-memory ladder accepts
translate into durable changes to ``panobbgo/`` defaults that survive
into the next night's seed measurement (§12.2: "the cron never commits
changes under ``panobbgo/``; durable improvement happens only through
codification").  The §9.3 stage of the V2 pipeline turns the live
ledger plus the rotated archives in ``planning/done/`` into a
candidate set of "tune this default" suggestions the operator (or a
future ``--open-pr`` driver) can act on.

``scripts/self_improve.py codify-scan`` ships the detection side of
that stage::

    uv run python scripts/self_improve.py codify-scan

The scanner groups every accepted iteration record by ``(class_name,
param_name, direction)`` — where ``direction`` is ``"up"`` / ``"down"``
for numeric rules, ``repr(new_value)`` for ``categorical_choice`` rules
(so ``Sobol.scramble=False`` is a separate candidate from
``Sobol.scramble=True``), and the op name for structural ops
(``add_heuristic`` / ``drop_heuristic`` / ``add_analyzer`` /
``drop_analyzer``).  Groups are surfaced when:

* They have at least ``--min-nights`` (default ``2``) distinct accept
  dates (``YYYY-MM-DD`` truncated from the iteration timestamp), and
* By default, every contributing record's per-record ``ci_low`` is
  strictly positive — every accept cleared its own screening
  statistical-accept gate, so the pooled signal cannot be a single
  lucky-CI spike.  Disable with ``--no-require-positive-min-ci`` to
  surface weakly-suggestive evidence too.

Each candidate's report carries the pooled point-delta CI (percentile
bootstrap on the per-record deltas), the strongest evidence first
(``(n_distinct_nights desc, mean_delta desc, n_accepts desc)``)::

    - Nearby.radius [log_uniform_perturb] direction=up
        n_accepts=7  n_nights=6  mean_Δ=+0.0566  pooled_CI95%=[+0.0424,+0.0721] …
        nights: 2026-05-26, 2026-05-28, 2026-05-31, 2026-06-04, 2026-06-10, 2026-06-15
        strategies: Rewarding_Diverse
        evidence:
          2026-05-26 06:43:37  Δ=+0.0561  CI=[+0.0039,+0.1233]  Rewarding_Diverse/Nearby.radius: 0.1 -> 0.100885
          …

Pass ``--confirmed-only`` to restrict the input to records carrying
``confirmed=True`` (the V2 §6.4 same-night confirmation gate's ledger
field).  On pre-§6.4 ledgers (the current archive) every record is
treated as legacy evidence; once the confirmation gate ships and the
field is populated, ``--confirmed-only`` will become the recommended
default for an operator opening a codify PR.

``--json`` emits one ``CodifyCandidate.to_dict()`` JSON object per
line for external dashboards or scripted PR generation; ``--top N``
limits the report to the strongest N candidates so a daily routine
can spotlight the highest-conviction codifications.

Already-codified candidates are *suppressed by default* (shipped
2026-06-18).  The scanner runs
:func:`~panobbgo.self_improve.annotate_codified_status` after
aggregation: it imports the seed-spec factories that the nightly
cron exercises (``_make_quick_strategies`` + ``_make_loop_strategies``
— see :func:`~panobbgo.self_improve.default_codify_registries`),
walks every spec's ``(heuristic_cls, kwargs)`` / ``(analyzer_cls,
kwargs)`` entries, and cross-checks each candidate's predicted edit
against the live kwarg values.  Categorical candidates compare
``repr(new_value) == repr(live)``; numeric candidates are flagged
when the live value already meets or exceeds the median of
``new_values`` in the candidate's direction (``"up"`` →
``max(live) >= median(new_values)``; ``"down"`` →
``min(live) <= median(new_values)``).  **Structural ops** —
extended *2026-06-19* via the new helper
:func:`~panobbgo.self_improve._live_class_membership` and the
matching :func:`~panobbgo.self_improve._structural_already_codified`
predicate: ``add_heuristic`` / ``add_analyzer`` candidates are
flagged when *at least one* seed spec already lists the class in
the matching bucket (the symmetric "partially redundant"
semantic), and ``drop_heuristic`` / ``drop_analyzer`` candidates
are flagged when *no* seed spec lists it (already removed
everywhere).  An unknown / future op name falls back to "not
codified" so the candidate continues to surface — defensive against
future catalog extensions.  Pass ``--include-already-codified`` to
audit the suppressed set — every already-codified candidate is
tagged ``[already codified]`` in the report and the
``live seed value(s)`` line surfaces the matching seed kwarg values
(or, for structural candidates, the *spec names* carrying the
class) so the operator can confirm the verdict.

JSON mode (``--json``) always emits every candidate so the consumer
can filter on the new
:attr:`~panobbgo.self_improve.CodifyCandidate.already_codified` and
:attr:`~panobbgo.self_improve.CodifyCandidate.live_codified_values`
fields itself.  The motivating example was
``Sobol.scramble = False``, codified in
:func:`~panobbgo.harness._make_quick_strategies` on 2026-05-31 but
continuing to surface from the pre-codification archive on every
subsequent ``codify-scan`` invocation; the suppression layer
collapses the report to the actionable set.

The scanner's library functions
(:class:`panobbgo.self_improve.CodifyCandidate`,
:func:`panobbgo.self_improve.aggregate_codify_candidates`,
:func:`panobbgo.self_improve.annotate_codified_status`,
:func:`panobbgo.self_improve.default_codify_registries`,
:func:`panobbgo.self_improve.load_ledgers_for_codify_scan`) are
public so a follow-up ``--open-pr`` driver can reuse the same
detection — the slot identifier
:attr:`CodifyCandidate.slot_key` is the ``(class, param, op)`` tuple
the §12.3 deduplication rule should consult against
``gh pr list --state open``.

See the 2026-06-17 / 2026-06-18 / 2026-06-19 entries in
``planning/SELF_IMPROVEMENT_LOG.md`` for the design rationale and
the dated entry's "Follow-up ideas" section for the queued
``--open-pr`` work.

Bidirectional-bound widening (``--widen-bounds``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Shipped 2026-06-19.  The codify scanner detects each ``(class, param,
direction)`` group independently, so a slot whose bandit finds value
moving the kwarg *up* on some nights and *down* on other nights
surfaces as two separate candidates — competing default-shift
proposals the operator cannot reconcile with a single PR.  The right
action on a bidirectional pattern is rarely a default shift but a
``MutationRule.bounds`` update that focuses the bandit's exploration
on the observed range.

Pass ``--widen-bounds`` to append a *Bound-widening candidates*
section to the report::

    uv run python scripts/self_improve.py codify-scan --widen-bounds

The widening detector pairs every ``(class_name, param_name)`` slot
whose direction set contains both ``"up"`` *and* ``"down"``, pools
every observed ``new_value`` across the two directions, and proposes
a new ``MutationRule.bounds`` tuple computed as:

* ``log_uniform_perturb`` — multiplicative on both ends: the lower
  end divides by ``--widen-factor`` (default ``1.5``), the upper end
  multiplies.  Symmetric in log space.
* ``integer_add`` — same rule, then rounded *outward*
  (:func:`math.floor` on the lower bound, :func:`math.ceil` on the
  upper).  Lower bound is clipped to ``1`` when observed values are
  positive — most integer-typed catalog kwargs are pool sizes /
  iteration counts where zero would be degenerate.
* ``float_uniform`` — multiplicative on the absolute values; sign
  preserved so a negative-valued knob widens away from zero on both
  sides.

Categorical and structural candidates have no meaningful "wider
bound" and are skipped silently.

For each surfaced pair the report carries the observed range, the
catalog's current bounds (or ``(no rule)`` when no numeric rule
targets the slot), the proposed bounds, and a one-token tag — one
of ``[widens current]`` / ``[tightens current — focuses bandit on
observed range]`` / ``[partial overlap]`` — so the operator can
prioritise.  On the live project ledger today (2026-06-19) the
detector surfaces two bidirectional patterns:

* ``Nearby.radius`` — observed ``[0.073, 0.135]``, current ``[0.005,
  0.5]``, proposed ``[0.049, 0.203]`` — **tightens current**: the
  bandit consistently picks values in a window 5-10× narrower than
  the catalog admits, so concentrating draws there frees compute.
* ``Sobol.n`` — observed ``[8, 24]``, current ``[4, 64]``, proposed
  ``[5, 36]`` — **tightens current** for the same reason.

In ``--json`` mode every widening candidate rides the same
line-delimited stream tagged ``"_type": "widening_candidate"`` while
codify candidates carry ``"_type": "codify_candidate"`` so a
downstream consumer can filter by type.  Library API:
:class:`panobbgo.self_improve.WideningCandidate` and
:func:`panobbgo.self_improve.detect_widening_candidates`.

Auto-tuned widen factor (``--widen-auto-tune``)
"""""""""""""""""""""""""""""""""""""""""""""""

Shipped 2026-06-22.  The fixed ``--widen-factor`` is one-size-fits-all
across rules whose observed spreads differ by an order of magnitude.
``--widen-auto-tune`` sizes the factor per-candidate from the ratio
of observed spread to catalog-bound span:

* **Narrow observed spread** (high agreement across nights, bandit
  converged) → larger factor — the proposed bound gets generous
  headroom outside the consensus window so the bandit has room to
  discover the next win.
* **Wide observed spread** (low agreement, bandit still exploring) →
  smaller factor — the proposed bound focuses on the consensus
  rather than ballooning further.

The spread is measured in the rule's natural scale:

* ``log_uniform_perturb`` — log-space ratio
  ``log(observed_hi / observed_lo) / log(current_hi / current_lo)``.
* ``integer_add`` / ``float_uniform`` — linear ratio
  ``(observed_hi - observed_lo) / (current_hi - current_lo)``.

Then linearly interpolated:

.. code-block:: text

   factor = max_factor - (max_factor - min_factor) * ratio

with the ratio clipped to ``[0.0, 1.0]``.  The default range is
``[1.1, 2.5]`` (set via ``--widen-factor-min`` / ``--widen-factor-max``).
When no catalog rule targets the slot — the relative-spread signal is
unavailable — the helper falls back to the supplied ``--widen-factor``
(default ``1.5``), so existing operators have a single-flag opt-in
path.

Live-ledger evidence on the day of ship:

* ``Nearby.radius`` (``log_uniform_perturb``) — observed
  ``[0.073, 0.135]``, catalog ``[0.005, 0.5]``, log-ratio ``≈ 0.13`` →
  auto-tuned factor **≈ 2.31** (vs the fixed ``1.5``).  Proposed
  bound widens from ``[0.049, 0.203]`` to ``[0.032, 0.313]`` —
  meaningfully more headroom outside the consensus window the bandit
  has converged into.
* ``Sobol.n`` (``integer_add``) — observed ``[8, 24]``, catalog
  ``[4, 64]``, linear-ratio ``≈ 0.27`` → auto-tuned factor **≈ 2.13**
  (vs ``1.5``).  Proposed bound flips from *tightening*
  (``[5, 36]``) to *widening* (``[3, 52]``) the catalog because the
  observed window covers enough of the catalog that a generous widen
  pushes the upper end past the catalog's current 36.

CLI:

.. code-block:: bash

   # Auto-tuned factor per candidate, default range [1.1, 2.5]:
   uv run python scripts/self_improve.py codify-scan \
       --widen-bounds --widen-auto-tune

   # Custom factor range — e.g. log-scale knobs tolerate larger
   # max_factor; the linear default is conservative:
   uv run python scripts/self_improve.py codify-scan \
       --widen-bounds --widen-auto-tune \
       --widen-factor-min 1.2 --widen-factor-max 4.0

The ``Bound-widening candidates`` report header switches from
``widen_factor=1.5`` to ``widen_factor=auto-tune [1.1, 2.5]
(fallback=1.5)`` when auto-tune is on, so the operator can see at a
glance which sizing rule produced each surfaced bound.  Library API:
:func:`panobbgo.self_improve._auto_tune_widen_factor` (the helper) and
the ``auto_tune`` / ``auto_tune_min_factor`` /
``auto_tune_max_factor`` keyword arguments on
:func:`panobbgo.self_improve.detect_widening_candidates`.

Adaptive mutation sampler (§10)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default the loop draws mutation rules uniformly at random from the
applicable ones in :func:`~panobbgo.self_improve.default_catalog`.  In
practice some rules (e.g. ``Nearby.radius``, ``CMAES.sigma0``) tend to
produce accepts much more often than others, and uniform sampling
wastes iterations on rules that never help.

When ``LoopConfig.adaptive_sampling`` is enabled, the loop substitutes
:class:`~panobbgo.self_improve.AdaptiveMutationSampler` — a
**Thompson-sampling bandit** over per-rule Beta posteriors:

1. For each mutation rule we maintain an accept/attempt counter
   ``(n_accepts, n_attempts)``.  Skip iterations don't count.
2. On every ``sample()`` call the sampler draws one variate from
   ``Beta(prior_alpha + n_accepts, prior_beta + n_attempts -
   n_accepts)`` per *applicable* rule.
3. The arg-max of those variates wins — Thompson's
   exploration/exploitation rule.
4. After the iteration, the loop calls ``record_outcome(accepted)``
   which updates the chosen rule's counters.

Cold-start equivalence to uniform.  With the default symmetric prior
:math:`\\mathrm{Beta}(1, 1)`, every posterior is :math:`\\mathrm{U}(0,
1)` and the arg-max of i.i.d. uniforms is itself uniform — so the very
first sample is statistically indistinguishable from
:meth:`~panobbgo.self_improve.MutationCatalog.sample`.  Flipping the
flag on a fresh ledger is therefore safe; behaviour diverges from
uniform only as evidence accumulates.

Resuming a long run.  Setting
``LoopConfig.adaptive_prime_from_ledger = True`` replays accept
history from any existing JSONL ledger before the first iteration.
Useful when restarting a multi-hour loop after a crash or a manual
stop — the bandit resumes with all the meta-knowledge of which rules
have worked so far.

Crossing nightly boundaries.  By default, priming only sees the
*live* ledger (``planning/self_improve_ledger.jsonl``).  Setting
``LoopConfig.adaptive_prime_include_archives = True`` (CLI
``--prime-include-archives``) additionally replays every archived
ledger under ``planning/done/`` matching the rotation glob
``self_improve_ledger_*.jsonl``.  Files are consumed in chronological
(lexicographic-on-filename) order before the live ledger.  A custom
directory can be passed via
``LoopConfig.adaptive_prime_archive_dir`` (CLI
``--prime-archive-dir``); a missing directory is a silent no-op.
Closes the §2.6 "archives in ``planning/done/`` are invisible"
diagnosis — the bandit posterior now accumulates evidence across
every retained nightly run rather than only the current one.

CLI:

.. code-block:: bash

   # Adaptive sampler with the default symmetric prior, primed from any
   # ledger sitting at the default path
   uv run python scripts/self_improve.py run --iterations 50 \
       --adaptive --adaptive-prime-from-ledger

   # Same, but also prime from archived ledgers (rotation glob
   # ``self_improve_ledger_*.jsonl``) sitting under ``planning/done/``.
   # Per-record semantics are identical to the live ledger — the bandit
   # posterior accumulates evidence across every retained nightly run.
   uv run python scripts/self_improve.py run --iterations 50 \
       --adaptive --adaptive-prime-from-ledger --prime-include-archives

   # Greedier prior for shorter exploratory loops
   uv run python scripts/self_improve.py run --iterations 20 \
       --adaptive --adaptive-prior-alpha 0.5 --adaptive-prior-beta 0.5

   # §7.4 graded reward shaping — turns every informative iteration
   # into evidence on the chosen arm so small-positive arms are
   # identifiable at 20-40 per-night iteration counts.
   uv run python scripts/self_improve.py run --iterations 50 \
       --adaptive --bandit-reward graded

Graded reward shaping (§7.4)
""""""""""""""""""""""""""""

The default ``LoopConfig.bandit_reward_shaping = "binary"`` updates
the Beta posterior with ``+1`` per accept / ``+0`` per reject — the
behaviour shipped 2026-05-01.  At the loop's ~2.5% per-night accept
rate (§2.6 of the V2 diagnosis) this leaves most arms close to the
prior, so the bandit "knows almost nothing" after a night.

Setting ``bandit_reward_shaping = "graded"`` (CLI
``--bandit-reward graded``) substitutes a continuous reward in
``[0, 1]`` derived from the bootstrap CI and point delta:

* ``accepted`` → ``0.5 + clip(ci_low / (4·eps_accept), 0, 0.5)`` —
  barely-confirmed accepts contribute ``~0.5``, clearly-winning
  accepts up to ``1.0``.
* rejected → ``clip(0.5 + Δ / (4·eps_accept), 0, 0.5)`` — a positive
  but sub-eps delta ("honest near miss") contributes ``~0.5``, a
  delta at zero contributes exactly ``0.5``, a clearly-harmful delta
  floors at ``0``.
* no-op / skip → no posterior update (gated by
  :meth:`~panobbgo.self_improve.AdaptiveMutationSampler.discard_outcome`
  upstream, see the §12.4 no-op detection subsection above).

The per-rule
:class:`~panobbgo.self_improve.MutationRuleStats` gains a
``reward_sum`` field accumulating the graded values; the Thompson
sampler swaps it in for ``n_accepts`` in the Beta posterior.  The
binary path is preserved byte-for-byte: ``record_outcome(accepted)``
with no explicit ``reward`` defaults to ``1.0 if accepted else 0.0``
so ``reward_sum`` tracks ``n_accepts`` exactly and the posterior is
identical to the pre-2026-06-13 shape.

Each :class:`~panobbgo.self_improve.LoopIterationRecord` gains an
optional ``bandit_reward`` field carrying the actual reward the
bandit consumed — ``None`` on skip / no-op iterations and on every
binary-mode iteration, the graded float on graded-mode iterations.
:meth:`~panobbgo.self_improve.AdaptiveMutationSampler.prime_from_ledger`
reads the field on resume so a graded run can recover its full
``reward_sum`` state across the persistence boundary.  Legacy
ledgers (no ``bandit_reward`` key) fall back to the binary reward,
so resuming from a pre-2026-06-13 ledger is byte-identical to the
old behaviour.

The summary metric ``stats.mean_reward = reward_sum / n_attempts``
distinguishes arms that produce small-positive deltas (``mean_reward
≈ 0.5`` even at ``accept_rate = 0``) from arms that produce harmful
deltas (``mean_reward ≈ 0``).  Inspect it alongside ``accept_rate``
when auditing the bandit:

.. code-block:: python

   for stats in improver.sampler.stats_snapshot():
       print(
           stats.rule_key,
           f"accept_rate={stats.accept_rate:.2f}",
           f"mean_reward={stats.mean_reward:.2f}",
       )

Programmatic use:

.. code-block:: python

   from panobbgo.self_improve import LoopConfig, SelfImprover

   cfg = LoopConfig(
       iterations=100,
       mode="standard",
       adaptive_sampling=True,
       adaptive_prior_alpha=1.0,
       adaptive_prior_beta=1.0,
       adaptive_prime_from_ledger=True,  # learn from prior runs
       guard_interval=10,
   )
   improver = SelfImprover(cfg)
   improver.run()

   # Inspect the bandit afterwards.
   for stats in improver.sampler.stats_snapshot():
       print(stats.rule_key, stats.n_accepts, "/", stats.n_attempts)

The sampler is **off by default** (``adaptive_sampling = False``) for
backward compatibility.

Strategy portfolio composition (§7.2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default catalog only retunes hyperparameters of heuristics that
already exist in each strategy.  The structural catalog — opt in via
``--structural`` on ``scripts/self_improve.py run`` or by passing
:func:`~panobbgo.self_improve.default_structural_catalog` to
:class:`~panobbgo.self_improve.SelfImprover` — extends the mutation
space with two new ops that change the *shape* of a
:class:`~panobbgo.benchmark.StrategySpec`'s heuristics list:

* ``add_heuristic`` — append a heuristic from a curated pool
  (``Random``, ``Nearby``, ``NelderMead``, ``Center``,
  ``LatinHypercube``, ``Sobol``, ``Extremal``, ``PSO``, ``LSHADE``,
  ``JSO``, ``NLSHADE_RSP``, ``NLSHADE_LBC``, ``LSHADE_EpSin``, ``COBYQA``, ``LBFGSB``) to a target strategy.  The pool ships *four*
  PSO entries — the default fully-connected ``gbest`` topology
  (Kennedy-Eberhart 1995), the ring ``lbest`` topology with
  ``k_neighbors=2`` (Kennedy & Mendes 2002), the 4-connected
  ``vonneumann`` 2-D toroidal grid (Kennedy & Mendes 2003; Mendes
  2004), and the ``random`` stochastic informer graph with
  ``k_neighbors=3`` (Mendes 2004; Clerc 2007 / SPSO 2011) — four
  complementary information-diffusion regimes (instantaneous /
  one-hop linear / two-hop planar / asymmetric stochastic) so the
  bandit can pick whichever exploration / exploitation trade-off
  helps on the current battery.  The ``random`` variant additionally
  exposes an optional ``stagnation_threshold`` kwarg that triggers a
  Clerc 2007 / SPSO 2011 stochastic-K stagnation rebuild — the
  informer graph is re-sampled mid-run whenever the swarm fails to
  lift its global best for ``N`` consecutive incoming results — and
  the default catalog exposes this knob via the ``PSO.stagnation_threshold``
  ``integer_add`` rule so the loop can tune the rebuild cadence on
  any spec that opts in.  L-SHADE (Tanabe-Fukunaga 2014) brings success-history
  adaptive Differential Evolution with linear population reduction
  and two opt-in jSO refinements: the iLSHADE / jSO (Brest 2016 /
  2017) linearly-decreasing ``p_best`` schedule (set ``p_best_end``
  on the spec to enable) and the categorical asymmetric F-cap regime
  ``F_schedule`` whose named choices ``"jso"`` (Brest 2017 —
  ``F ≤ 0.7`` while ``progress < 0.6``, ``F ≤ 0.8`` while
  ``0.6 ≤ progress < 0.9``, unclamped in the final 10%), ``"early"``
  (kicks in earlier and tighter — ``F ≤ 0.6`` while ``progress < 0.4``,
  ``F ≤ 0.8`` while ``progress < 0.7``, unclamped after that), and
  ``"strict"`` (most aggressive — ``F ≤ 0.5`` while ``progress < 0.5``,
  ``F ≤ 0.7`` while ``progress < 0.85``, unclamped in the final 15%)
  each pick a literature-grade three-phase asymmetric F-cap (the
  legacy ``True`` / ``False`` boolean inputs map onto ``"jso"`` /
  ``"off"`` for back-compat with the binary toggle shipped 2026-05-21);
  jSO (Brest, Maučec & Bošković 2017 — CEC-2017 winner) refines L-SHADE
  with a weighted ``current-to-pbest-w/1`` mutation, a linear
  ``p_best`` schedule, the literature-faithful three-phase asymmetric
  F-cap (opted into via the shared L-SHADE machinery by construction),
  and a frozen anchor memory bin.  NL-SHADE-RSP (Stanovov, Akhmedova
  & Semenkin 2021 — CEC-2021 winner) refines jSO further with
  Non-Linear Population Size Reduction (``NP(r) = round((NP_min −
  NP_init)·r^(1−r) + NP_init)``), Rank-based Selective Pressure on the
  differential ``r1`` draw (``k_rank`` default ``3``), and a randomised
  per-generation archive cap.  NL-SHADE-LBC (Stanovov, Akhmedova &
  Semenkin 2022 — CEC-2022 winner) refines NL-SHADE-RSP further with
  **Linear Bias Change** in the success-history memory update: the F /
  CR Lehmer-mean order ``p`` is linearly scheduled across budget
  progress (``p_F: 3.5 → 1.5``, ``p_CR: 1.0 → 1.5``) instead of fixed
  at ``2``; the numerator / denominator exponent spread ``m_lbc`` is
  held constant at ``1.5`` (``m = 1`` and ``p = 2`` everywhere recovers
  the standard L-SHADE Lehmer mean).  The constructor's
  ``lbc_regime`` kwarg (shipped 2026-06-24) wraps four named regime
  presets — ``"cec2022"`` (the Stanovov defaults above), ``"lshade"``
  (recovers the standard L-SHADE Lehmer mean at ``p = 2, m = 1``),
  ``"flat"`` (pure arithmetic mean — ``p = 1`` throughout, default
  spread), and ``"aggressive"`` (strong bias throughout — ``p_F``
  decays ``5 → 3``, ``p_CR`` grows ``3 → 5``, default spread).
  The named regime is mutually exclusive with the explicit per-field
  LBC kwargs and is exposed to the self-improvement bandit as a
  single ``categorical_choice`` rule on the
  ``(NLSHADE_LBC, lbc_regime)`` slot — joint exploration of the
  five-field LBC regime space in one well-curated discrete arm
  rather than five cold-started independent float arms.  LSHADE-EpSin (Awad, Ali & Suganthan
  2016) is the *orthogonal* CEC-2016 sinusoidal-F branch — it replaces
  SHADE Cauchy-from-memory ``F`` sampling with an ensemble of two
  sinusoidal candidates (fixed-frequency / decreasing-envelope vs
  adaptive-frequency / increasing-envelope, mixed by an adaptive
  Sinusoid-1 selection probability ``p_s``) during the first half of the
  search, reverting to SHADE Cauchy in the second half; precursor of the
  CEC-2017 co-winner LSHADE-cnEpSin.  All six DE-family arms share the
  ``add_heuristic`` arm so the bandit picks whichever variant wins on
  the current battery.  COBYQA
  (Ragonneau-Zhang 2023) brings the modern Powell-family
  derivative-free trust-region local optimizer (BOBYQA / NEWUOA
  successor) alongside Nelder-Mead.  LBFGSB (Zhu-Byrd-Lu-Nocedal 1997)
  adds the only *gradient-based* arm — a multi-start, bound-constrained
  quasi-Newton local optimizer (finite-difference gradients) that
  complements the derivative-free generators on smooth ill-conditioned
  valleys (e.g. Rosenbrock), where a dedicated descent reaches the
  optimum in a fraction of a population method's budget.
  ``avoid_duplicates=True``
  (default) skips classes that are already present in the strategy,
  so the catalog cannot litter a portfolio with redundant copies (and
  in particular only ever installs *one* PSO variant per strategy).
* ``drop_heuristic`` — remove an existing heuristic, optionally
  restricted to a tuple of class names via
  ``StructuralMutationRule.droppable_classes``.  The
  ``min_heuristics`` field (default ``2``) is the floor of the
  *post-drop* heuristic count, so the strategy always keeps at least
  one diversity slot beyond the bare minimum.
* ``add_analyzer`` — append an analyzer from the curated pool
  (``Sensitivity`` with ``update_interval=20``, ``Restart`` with the
  IPOP-style ``diverse`` strategy and ``max_restarts=5``) to a target
  strategy.  Shipped 2026-06-02 — mirrors ``add_heuristic`` but
  targets :attr:`~panobbgo.benchmark.StrategySpec.analyzers` rather
  than ``heuristics``.  Useful for letting the loop discover whether
  attaching ``Restart`` (warm restarts on stagnation) or
  ``Sensitivity`` (adaptive sensitivity tracking) helps a given seed
  composition.  ``avoid_duplicates=True`` (default) skips analyzer
  classes already attached.
* ``drop_analyzer`` — remove an existing analyzer, with the same
  optional ``droppable_classes`` filter as ``drop_heuristic``.  The
  matching safety floor is :attr:`StructuralMutationRule.min_analyzers`
  (default ``0`` — analyzers are non-essential, unlike heuristics, so
  stripping :class:`~panobbgo.analyzers.Sensitivity` from a Rewarding
  strategy yields a valid, slightly faster spec).

All four flavours land as one
:class:`~panobbgo.self_improve.MutationProposal` carrying ``op`` and
``structural_kwargs``; :func:`~panobbgo.self_improve.apply_mutation`
dispatches on ``proposal.op`` so the rest of the loop driver — the
ledger, the anti-cherry-pick guard, the statistical acceptance rule —
is unchanged and the JSONL ledger remains backwards compatible.
Analyzer ops use the same ``MutationProposal`` fields and the same
``rule_kind`` namespace (``"add_analyzer"`` / ``"drop_analyzer"``) so
existing ledger consumers see one extra ``rule_kind`` they may
ignore.

The Thompson sampler maps every structural rule onto **one arm per
op** (key ``("*", op, "structural")``) by default — the same flat
collapse for all four ops (``add_heuristic`` / ``drop_heuristic`` /
``add_analyzer`` / ``drop_analyzer``).  This keeps cold-start
variance bounded — a freshly enabled adaptive sampler on a
structural catalog has the same uniform-mix behaviour as on the
kwarg catalog.

Per-class structural bandit arms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once a structural catalog has accumulated evidence, the coarse
one-arm-per-op layout becomes the limiting factor: the bandit cannot
distinguish ``add Sobol`` (which may be a consistent winner) from
``add Random`` (which may not), nor ``add Restart`` (which may help
a multi-modal landscape) from ``add Sensitivity`` (which is mostly
diagnostic).  Setting
:attr:`~panobbgo.self_improve.LoopConfig.structural_per_class_arms`
(or passing ``per_class_structural=True`` to
:class:`~panobbgo.self_improve.AdaptiveMutationSampler` directly)
splits the structural arms by **target candidate class**.  ``add
Sobol`` then lives on the bandit arm
``("Sobol", "add_heuristic", "structural")``; ``add Random`` lives
on ``("Random", "add_heuristic", "structural")``; ``add Restart``
lives on ``("Restart", "add_analyzer", "structural")``; the
Thompson posterior on each arm tracks how often *that specific add*
(or drop) is accepted.

The trade-off is the canonical bandit one: sharper signal vs sparser
per-arm data.  With ``N`` candidate classes in the structural
catalog, the bandit arm space grows by a factor of ``N`` (for each
op) and each arm starts cold-start with the symmetric ``Beta(1, 1)``
prior — i.e. uniform.  In practice this means the first few
iterations will explore each class roughly uniformly, then
concentrate probability on whatever's accepting.

Ledger replay is consistent: :meth:`prime_from_ledger` uses the
same key layout the live sampler will produce, so resuming a long
run with ``--adaptive-prime-from-ledger`` recovers the per-class
posterior intact.  Kwarg perturbations are unaffected by the flag —
their ``(class, param, kind)`` arms are already per-class.

CLI:

.. code-block:: bash

   # Structural catalog + adaptive sampler + per-class structural arms.
   uv run python scripts/self_improve.py run --iterations 100 \
       --structural --adaptive --structural-per-class-arms \
       --adaptive-prime-from-ledger

The flag is **off by default** so existing CLI invocations and the
existing ledger format stay byte-identical.  When ``--adaptive`` is
not set the flag is inert (no
:class:`~panobbgo.self_improve.AdaptiveMutationSampler` is
constructed) but tolerated, which keeps a config-driven workflow
that toggles ``adaptive_sampling`` independently from
``structural_per_class_arms`` safe.

Hierarchical bandit over per-class structural arms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The per-class arms above pay for sharper signal with sparser data: a
candidate pool of ``N`` classes divides the bandit's structural
evidence by roughly ``N``, so a fresh class starts with the symmetric
``Beta(1, 1)`` prior even when its op-level sibling history is
strongly informative.
:attr:`~panobbgo.self_improve.LoopConfig.structural_borrow_alpha`
(``κ ≥ 0``) closes the gap with a **hierarchical Beta-Binomial**:
each per-class arm's Beta posterior borrows
``κ · (n_other_class_accepts, n_other_class_failures)`` from the
op-level aggregate (the sum across every *sibling* per-class arm with
the same op).  The leaf posterior becomes:

.. math::

   \mathrm{Beta}\bigl(
     \alpha_0 + n_{\text{class}}^{\text{accepts}}
              + \kappa \cdot n_{\text{other-class}}^{\text{accepts}},\;
     \beta_0  + n_{\text{class}}^{\text{failures}}
              + \kappa \cdot n_{\text{other-class}}^{\text{failures}}
   \bigr)

The self-exclusion is deliberate: borrowing from one's own evidence
would collapse the hierarchy to a ``κ``-amplified version of the same
per-class posterior.  ``κ = 0`` (default) recovers the pure per-class
semantics shipped 2026-05-18; ``κ = 1`` weights every sibling accept
equally with the class's own.  A useful intermediate is ``κ = 0.5``,
which discounts sibling evidence by half — empirically a robust
default in hierarchical-bandit literature when there is real but
imperfect transfer between arms.

Concretely: with one sibling that has accepted 20/20 times, a fresh
class under ``κ = 1`` starts with effective posterior
``Beta(1 + 0 + 20, 1 + 0 + 0) = Beta(21, 1)`` — mean ≈ 0.95.  The
unhierarchical sampler would start the same class at
``Beta(1, 1)`` (mean 0.5), needing many more arg-max contests to
catch up.

The borrow is **inert** when:

* :attr:`~panobbgo.self_improve.LoopConfig.structural_per_class_arms`
  is ``False`` (no per-class arms exist to borrow between), or
* :attr:`~panobbgo.self_improve.LoopConfig.adaptive_sampling` is
  ``False`` (no :class:`~panobbgo.self_improve.AdaptiveMutationSampler`
  is constructed), or
* the proposed rule is a kwarg perturbation (kwarg arms are not
  grouped by an "op", so there is no aggregate to borrow from).

CLI:

.. code-block:: bash

   # Per-class structural arms with hierarchical borrow (κ = 0.5).
   uv run python scripts/self_improve.py run --iterations 100 \
       --structural --adaptive --structural-per-class-arms \
       --structural-borrow-alpha 0.5 \
       --adaptive-prime-from-ledger

Programmatic:

.. code-block:: python

   from panobbgo.self_improve import LoopConfig, SelfImprover, default_structural_catalog

   cfg = LoopConfig(
       iterations=100,
       adaptive_sampling=True,
       structural_per_class_arms=True,
       structural_borrow_alpha=0.5,  # half-weight sibling evidence
   )
   SelfImprover(cfg, catalog=default_structural_catalog()).run()

Auto-tune ``κ`` from observed evidence (``structural_borrow_horizon``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The fixed-``κ`` posterior above closes the cold-start gap but never
*lets go*: an arm with hundreds of attempts of its own still pays the
``κ ·`` op-aggregate cost, indefinitely pulling its leaf posterior
toward the op-level mean.  Empirically the right behaviour is
"borrow heavily early, vanish as evidence grows" — every
hierarchical-bandit textbook puts this knob behind an annealing rule.

:attr:`~panobbgo.self_improve.LoopConfig.structural_borrow_horizon`
(``h ≥ 0``) provides exactly that.  When ``h > 0`` (and the two
borrow preconditions are met — ``κ > 0`` and per-class arms enabled),
each per-class arm's *effective* borrow shrinks toward zero as that
arm's own attempts accumulate:

.. math::

   \kappa_{\text{eff}}(n_{\text{class}}^{\text{attempts}}) =
     \frac{\kappa}{1 + n_{\text{class}}^{\text{attempts}} / h}

So a cold arm (``n_class_attempts = 0``) borrows the full configured
``κ`` — same as the fixed-``κ`` path.  At ``n_class_attempts = h`` the
borrow halves exactly.  By ``n_class_attempts = 10·h`` the borrow has
shrunk to roughly ``κ / 11`` — the leaf posterior now trusts its own
evidence instead of being pulled toward the op-level aggregate.

The annealing is **inert** under the same preconditions as the borrow
itself: ``κ = 0``, ``structural_per_class_arms = False``, or
``adaptive_sampling = False`` all make ``structural_borrow_horizon``
a silent no-op.  The default ``h = 0`` *also* disables annealing
(every arm always sees the full ``κ``) so the knob is opt-in and
byte-identical to the 2026-06-01 fixed-``κ`` ship when unset.

Recommended values for an unattended cron: ``h = 5`` to ``10``.  The
per-arm posteriors warm up over a couple of nights at the catalog's
typical per-iteration cardinality, beyond which the bandit should
trust the leaf-level signal rather than continue being pulled toward
the op-level mean.

CLI:

.. code-block:: bash

   # Per-class structural arms with annealed hierarchical borrow.
   # Cold arms borrow the full κ=1.0; arms with ≥ 10 attempts borrow ≤ κ/3.
   uv run python scripts/self_improve.py run --iterations 100 \
       --structural --adaptive --structural-per-class-arms \
       --structural-borrow-alpha 1.0 --structural-borrow-horizon 5 \
       --adaptive-prime-from-ledger

Programmatic:

.. code-block:: python

   from panobbgo.self_improve import LoopConfig, SelfImprover, default_structural_catalog

   cfg = LoopConfig(
       iterations=100,
       adaptive_sampling=True,
       structural_per_class_arms=True,
       structural_borrow_alpha=1.0,
       structural_borrow_horizon=5.0,  # halve the borrow at 5 per-class attempts
   )
   SelfImprover(cfg, catalog=default_structural_catalog()).run()

CLI:

.. code-block:: bash

   # Structural catalog, uniform sampler.
   uv run python scripts/self_improve.py run --iterations 50 --structural

   # Structural catalog plus Thompson-sampling adaptive sampler.
   uv run python scripts/self_improve.py run --iterations 100 \
       --structural --adaptive --adaptive-prime-from-ledger

Programmatic use:

.. code-block:: python

   from panobbgo.self_improve import (
       LoopConfig,
       SelfImprover,
       StructuralMutationRule,
       MutationCatalog,
       default_structural_catalog,
   )
   from panobbgo.heuristics import Sobol, Nearby

   cfg = LoopConfig(iterations=100, mode="standard", randomize=True)
   # Built-in structural catalog (kwarg rules + add/drop ops).
   improver = SelfImprover(cfg, catalog=default_structural_catalog())
   improver.run()

   # Or build a focused custom catalog: just propose adding Sobol' to
   # any strategy that doesn't have it yet.
   custom = MutationCatalog([
       StructuralMutationRule(
           strategy_pattern="",
           op="add_heuristic",
           candidate_classes=((Sobol, {"n": 16, "scramble": True}),),
       ),
   ])
   SelfImprover(cfg, catalog=custom).run()

The ``--structural`` flag is **off by default** so existing CLI
invocations and existing ledgers stay byte-identical.

Categorical mutation rule
~~~~~~~~~~~~~~~~~~~~~~~~~

The three original mutation kinds — ``log_uniform_perturb``,
``integer_add``, ``float_uniform`` — all sample from a continuous
numeric space.  Some of the most impactful design choices in
Panobbgo's heuristic portfolio are **discrete** instead:

* ``PSO.topology`` — ``"gbest"`` (fully-connected swarm,
  instantaneous diffusion) vs ``"lbest"`` (ring with one-hop
  diffusion, better on multimodal landscapes) vs ``"vonneumann"``
  (4-connected 2-D toroidal grid, two-hop planar diffusion — Kennedy
  & Mendes 2003 / Mendes 2004 — a stable middle ground) vs
  ``"random"`` (Mendes 2004 / Clerc 2007 / SPSO 2011 stochastic
  informer graph — structure-free middle ground whose diffusion
  speed depends on the realised graph).
* ``Sobol.scramble`` — Owen scrambling on / off; trades a
  pseudo-random "freshness" against the classic Sobol' grid.
* ``LSHADE.archive_factor`` — ``0.0`` (no archive, vanilla
  current-to-pbest/1) vs ``1.0`` (Tanabe-Fukunaga default) vs
  ``2.6`` (L-SHADE-RSP enlarged archive).
* ``LSHADE.F_schedule`` — four named regimes lift the original
  binary toggle into a categorical:
  ``"off"`` (vanilla Tanabe-Fukunaga L-SHADE, unclamped throughout),
  ``"jso"`` (Brest et al. 2017 three-phase asymmetric F-cap:
  ``F ≤ 0.7`` in [0, 0.6), ``F ≤ 0.8`` in [0.6, 0.9), unclamped in
  [0.9, 1.0]), ``"early"`` (earlier and tighter kick-in:
  ``F ≤ 0.6`` in [0, 0.4), ``F ≤ 0.8`` in [0.4, 0.7), unclamped
  after that — useful when small initial steps help converge
  faster), and ``"strict"`` (most aggressive: ``F ≤ 0.5`` in
  [0, 0.5), ``F ≤ 0.7`` in [0.5, 0.85), unclamped in [0.85, 1.0] —
  useful when large F is harmful, e.g. ill-conditioned basins).
  Each regime is a literature-grade three-phase asymmetric cap;
  flipping between them lets the bandit move an existing
  :class:`LSHADE` instance across qualitatively distinct cap
  geometries without dropping and re-adding the heuristic.  The
  legacy ``True`` / ``False`` boolean inputs still work as
  backwards-compatible synonyms for ``"jso"`` / ``"off"``
  (preserving ledger replay against the binary toggle shipped
  2026-05-21).
* ``NLSHADE_RSP.k_rank`` — ``0.0`` (uniform ``r1`` selection,
  recovers jSO behaviour) vs ``3.0`` (Stanovov et al. 2018 / 2021
  RSP default) vs ``5.0`` (aggressive rank pressure).  Sits
  alongside the ``float_uniform`` rule (``bounds=(1.0, 5.0)``) so
  the bandit can either continuously walk ``k_rank`` or jump
  between qualitatively distinct regimes.  In particular ``0.0``
  is unreachable from the continuous rule and gives the loop a way
  to switch off the rank-based pressure entirely on portfolios
  that opt into NL-SHADE-RSP.
* ``COBYQA.scale`` — ``True`` (box variables rescaled to
  ``[-1, 1]``, keeping the Powell interpolation geometry
  well-conditioned) vs ``False`` (raw box).  Useful when the
  problem's box is already isotropic and the rescale adds
  rounding noise that hurts the quadratic-model fit.
* ``NLSHADE_LBC.lbc_regime`` — four named regime presets over the
  five LBC Lehmer-mean kwargs (``p_F_init`` / ``p_F_final`` /
  ``p_CR_init`` / ``p_CR_final`` / ``m_lbc``): ``"cec2022"``
  (Stanovov et al. 2022 defaults — F bias decays ``3.5 → 1.5``,
  CR bias grows ``1.0 → 1.5``, spread ``1.5``), ``"lshade"``
  (recovers the standard L-SHADE Lehmer mean at ``p = 2, m = 1`` —
  turns the LBC mechanism itself off without dropping the
  heuristic), ``"flat"`` (pure arithmetic mean — ``p = 1``
  throughout, default spread), and ``"aggressive"`` (strong bias
  throughout — ``p_F`` decays ``5 → 3``, ``p_CR`` grows ``3 → 5``,
  default spread).  Mutually exclusive with the five per-field
  LBC float kwargs.  Replaces the five per-field
  ``float_uniform`` rules previously on the catalog (shipped
  2026-05-28; retired 2026-06-24) with a single
  literature-motivated composite arm — joint exploration of the
  five-field LBC regime space instead of five cold-started
  independent dial arms.  Mirrors the 2026-06-23 ``F_schedule``
  regime broadening on L-SHADE.

The :class:`~panobbgo.self_improve.MutationRule` ``categorical_choice``
kind closes this gap.  The rule carries a ``choices`` tuple of
candidate values; on every applicable sample the catalog draws
uniformly from ``choices`` *excluding* the current value, so the
mutation always proposes a real change (no-op samples are
eliminated by construction).  The bandit treats categorical rules
as their own arm — distinct from any numeric rule on the same
``(class, param)`` slot — so the Thompson sampler can learn whether
flipping a discrete knob is worthwhile.

The default catalog ships ten categorical rules out-of-the-box:

.. code-block:: python

   MutationRule(
       strategy_pattern="",
       class_name="PSO",
       param_name="topology",
       kind="categorical_choice",
       choices=("gbest", "lbest", "vonneumann", "random"),
       probability=0.3,
   ),
   MutationRule(
       strategy_pattern="",
       class_name="Sobol",
       param_name="scramble",
       kind="categorical_choice",
       choices=(True, False),
       probability=0.3,
   ),
   MutationRule(
       strategy_pattern="",
       class_name="LSHADE",
       param_name="archive_factor",
       kind="categorical_choice",
       choices=(0.0, 1.0, 2.6),
       probability=0.3,
   ),
   MutationRule(
       strategy_pattern="",
       class_name="LSHADE",
       param_name="F_schedule",
       kind="categorical_choice",
       choices=("off", "jso", "early", "strict"),
       probability=0.3,
   ),
   MutationRule(
       strategy_pattern="",
       class_name="NLSHADE_RSP",
       param_name="adaptive_archive",
       kind="categorical_choice",
       choices=(True, False),
       probability=0.3,
   ),
   MutationRule(
       strategy_pattern="",
       class_name="NLSHADE_RSP",
       param_name="k_rank",
       kind="categorical_choice",
       # 0.0 = uniform r1 / jSO recovery; 3.0 = RSP default; 5.0 = aggressive.
       choices=(0.0, 3.0, 5.0),
       probability=0.3,
   ),
   MutationRule(
       strategy_pattern="",
       class_name="JSO",
       param_name="p_best_max",
       kind="categorical_choice",
       # 0.15 ≈ L-SHADE-like (raised from the literature 0.11 so it
       # clears jSO's default p_best_min = 0.125 floor); 0.25 = Brest
       # et al. 2017 jSO default; 0.4 = iLSHADE-like broader pool.
       # Shipped 2026-06-09 — see §13 entry.  Sits alongside the
       # ``float_uniform`` rule on the same slot for continuous /
       # categorical complementarity.
       choices=(0.15, 0.25, 0.4),
       probability=0.3,
   ),
   MutationRule(
       strategy_pattern="",
       class_name="COBYQA",
       param_name="scale",
       kind="categorical_choice",
       choices=(True, False),
       probability=0.3,
   ),
   MutationRule(
       strategy_pattern="",
       class_name="NLSHADE_LBC",
       param_name="lbc_regime",
       kind="categorical_choice",
       # "cec2022" = Stanovov et al. 2022 defaults; "lshade" =
       # standard L-SHADE Lehmer mean (p=2, m=1 — turns LBC off);
       # "flat" = pure arithmetic mean (p=1, default spread);
       # "aggressive" = strong bias throughout.  Shipped 2026-06-24 —
       # replaced the five per-field LBC float_uniform rules with one
       # composite joint-search arm.  See §13 entry.
       choices=("cec2022", "lshade", "flat", "aggressive"),
       probability=0.3,
   ),
   MutationRule(
       strategy_pattern="",
       class_name="Restart",
       param_name="restart_strategy",
       kind="categorical_choice",
       # "random" = uniform-in-box; "diverse" = max-min distance from
       # previous restart centres; "sphere" = Gaussian around the box
       # centre with std = ranges / 6 (clipped to the box).  Shipped
       # 2026-06-07 — see §13 entry.
       choices=("random", "diverse", "sphere"),
       probability=0.3,
   ),

Each fires only when the target spec sets the kwarg *explicitly*
— the catalog's "param already in kwargs" predicate filters out
specs that left the kwarg implicit (the heuristic's constructor
default).  Of the shipped strategies, ``Rewarding_Diverse``
(quick mode) sets ``scramble=False`` and ``BayesOpt_Sobol``
(standard mode) sets ``scramble=True`` — both fire the Sobol'
rule out of the box, so the bandit can flip either spec.  The
``False`` setting on ``Rewarding_Diverse`` was **codified
2026-05-31** after three independent positive accepts in the
self-improvement loop's archived ledger — see the §13 entry
"Codify ``Sobol.scramble=False`` in ``Rewarding_Diverse``" for
the evidence trail.  The PSO and LSHADE rules become applicable
once the structural catalog adds an opt-in PSO / LSHADE entry
with the matching kwarg present.

Likewise, the kwarg catalog also covers two under-tuned
analyzer / local-optimizer dials whose default values are the
``None`` sentinel for the heuristic-internal auto-default:

* ``Restart.patience`` (``integer_add``, ``bounds=(3, 200)``,
  ``delta_choices=(-20, -10, -5, 5, 10, 20)``) — the more
  impactful of the two :class:`~panobbgo.analyzers.restart.Restart`
  knobs, controlling how aggressively the optimizer restarts when
  stuck.  The analyzer's default is ``5 · dim`` (auto-derived at
  ``__start__``); the built-in factories ship ``patience=None`` and
  inherit the auto-default, so the rule stays opt-in until a future
  spec or mutation sets ``patience`` to a concrete integer.
* ``LBFGSB.max_starts`` (``integer_add``, ``bounds=(1, 50)``,
  ``delta_choices=(-5, -2, -1, 1, 2, 5)``) — caps the multi-start
  L-BFGS-B restart budget; ``1`` reduces the heuristic to a pure
  box-centre descent, larger values give the random-restart layer
  more chances to find a different basin.  The heuristic's default
  is ``None`` (unlimited until the strategy budget is exhausted).

Both rules require :func:`~panobbgo.self_improve._find_targets`'s
**``None``-skip** — the "param already in kwargs" predicate also
demands that the value is not ``None``, so a spec that ships the
auto-default sentinel is filtered out instead of crashing
``int(None) + delta``.  The ``None``-skip is uniform across all
rule kinds (not just ``integer_add``) and is behaviourally inert
for every previously-shipped catalog rule.

The 2026-06-05 :class:`~panobbgo.heuristics.region_ucb.RegionUCB`
leaf-bandit heuristic also has three catalog rules covering its
exploration / exploitation dials, shipped 2026-06-08 together
with the byte-identical seed-spec activation of
``Rewarding_RegionUCB``:

* ``RegionUCB.ucb_c`` (``log_uniform_perturb``,
  ``bounds=(0.1, 4.0)``, ``log_step=0.15``) — the UCB1
  exploration weight in the leaf score
  ``quality + ucb_c · sqrt(log(N) / n_leaf)``.  The bounds
  bracket the literature default of ``1.0`` (Auer et al. 2002)
  so a single perturbation can probe both the exploit-heavy
  (``< 1``) and explore-heavy (``> 1``) regimes.
* ``RegionUCB.gauss_fraction`` (``float_uniform``,
  ``bounds=(0.0, 1.0)``) — fraction of in-leaf candidates
  drawn from a Gaussian around the leaf's best point instead
  of uniformly over the leaf box.  The full ``[0, 1]`` range
  is bandit-reachable so the loop can probe the LA-MCTS
  pure-uniform regime (``0.0``) and the pure-local-refinement
  regime (``1.0``) symmetrically.
* ``RegionUCB.gauss_scale`` (``log_uniform_perturb``,
  ``bounds=(0.05, 0.5)``, ``log_step=0.15``) — Gaussian
  std-dev as a fraction of the leaf's per-axis ranges.  The
  constructor default ``0.25`` sits near the geometric centre
  of the log-uniform window so symmetric perturbations can
  both shrink (Nearby-style tight refinement) and widen
  (uniform-leaf baseline) the in-leaf Gaussian cloud.

All three rules fire only when a spec sets the matching kwarg
explicitly.  The seed ``Rewarding_RegionUCB`` spec in
:func:`~panobbgo.harness._make_standard_strategies` ships
``(RegionUCB, {"ucb_c": 1.0, "gauss_fraction": 0.5,
"gauss_scale": 0.25})`` — the three values match the
constructor defaults so RegionUCB construction is byte-
identical to the prior ``(RegionUCB, {})`` form, but the
kwarg dict's membership now activates the catalog rules on
the standard-mode battery rather than letting them sit
dormant.

Adding more categorical rules is a one-liner:

.. code-block:: python

   from panobbgo.self_improve import MutationRule, MutationCatalog, default_catalog

   custom = MutationCatalog(
       list(default_catalog().rules)
       + [
           MutationRule(
               strategy_pattern="",
               class_name="MyHeuristic",
               param_name="mode",
               kind="categorical_choice",
               choices=("aggressive", "conservative"),
           ),
       ]
   )

Ledger serialisation is automatic: the proposal records
``rule_kind="categorical_choice"`` and the literal categorical
values in ``old_value`` / ``new_value``, so a replay through
:func:`panobbgo.self_improve._proposal_rule_key` recovers the
bandit arm losslessly.

Hold-out validation set
~~~~~~~~~~~~~~~~~~~~~~~

The anti-cherry-pick guard catches drift inside the *training*
``base_seed`` family — it varies only ``randomize_iteration`` and keeps
the ``HarnessConfig.seed`` constant.  A mutation that overfits to
peculiarities of the training base-seed family slips through: the
guard's "fresh" instances are still drawn from the same SHA-256 stream.

The **hold-out validation set** closes that gap.  At the end of the
loop run, when
:attr:`~panobbgo.self_improve.LoopConfig.holdout_base_seed` is non-zero
and :attr:`~panobbgo.self_improve.LoopConfig.holdout_iterations` is
positive, the loop re-measures both the **seed** ladder entry and the
**final top** entry on instances drawn from a completely independent
``base_seed`` SHA-256 stream.  The two scores are averaged over
``holdout_iterations`` distinct ``randomize_iteration`` values and
compared to the training-time ``last_validated_score`` recorded on the
ladder.  If the hold-out gap (``top − seed``) is smaller than the
training gap by more than
:attr:`~panobbgo.self_improve.LoopConfig.holdout_eps_overfit`, the
:class:`~panobbgo.self_improve.LoopHoldoutRecord` is flagged
``overfit=True``.

**Three-way verdict** (shipped 2026-06-11 — V2 §6.4 / §12.4).  Every
:class:`~panobbgo.self_improve.LoopHoldoutRecord` carries an explicit
``status`` field with the three permissible values:

* ``"ok"`` — the drift stayed within
  :attr:`~panobbgo.self_improve.LoopConfig.holdout_eps_overfit`; the
  accepted mutations appear to generalise to the independent
  ``base_seed`` family.
* ``"overfit"`` — drift below ``-eps_overfit``; the ladder appears to
  have overfit the *training* ``base_seed`` family and the
  ``--fail-on-overfit`` CLI gate exits with code ``3``.
* ``"vacuous"`` — the ladder kept only the seed entry (no accepted
  mutations to validate), so ``holdout_delta``, ``training_delta``
  and ``drift`` are all ``0.0`` by construction and the record
  carries **no** generalisation signal.  Distinct from ``"ok"``: a
  vacuous record cannot honestly claim that "the improvement
  generalised" because there was no improvement to validate in the
  first place.  Previously this case printed as ``OK drift=+0.0000``
  and silently passed through the aggregator's bootstrap, biasing
  the CI toward zero.

The :func:`~panobbgo.self_improve.aggregate_holdout_drift` function
filters vacuous records out of the bootstrap so a single
negative-drift seed cannot be masked by a vacuous companion;
:attr:`~panobbgo.self_improve.HoldoutDriftAggregate.vacuous_count`
and :attr:`~panobbgo.self_improve.HoldoutDriftAggregate.all_vacuous`
surface the count so a reviewer can see *why* the sample size is
small.  Both ``scripts/self_improve.py run`` and
``scripts/self_improve.py summary`` print ``VACUOUS`` (per-record)
and ``VACUOUS_CI`` (aggregate) verdicts when the underlying records
carry no informative content.  Legacy ledger lines (no ``status``
field on disk, pre-2026-06-11) classify correctly via
:meth:`~panobbgo.self_improve.LoopHoldoutRecord.effective_status`,
which derives the right verdict from the other fields
(``ladder_size <= 1 and top_iteration < 0`` → ``"vacuous"``,
``overfit=True`` → ``"overfit"``, otherwise ``"ok"``).

Algorithm:

1. After all loop iterations have completed (and the guard has done
   its work), run :meth:`~panobbgo.self_improve.SelfImprover._run_holdout`.
2. Build a :class:`~panobbgo.harness.HarnessConfig` whose ``seed`` is
   the **independent** ``holdout_base_seed`` (every other knob — mode,
   reps, budget, ``strategies_override`` — matches the training run).
3. For ``k = 0 … holdout_iterations - 1``, set
   ``randomize_iteration = holdout_iteration_offset + k`` and measure
   both the seed and top spec lists.  Average to get
   ``seed_holdout_score`` and ``top_holdout_score``.
4. ``drift = (top_holdout − seed_holdout) − (top_training −
   seed_training)``.  Negative drift means the gap shrank on hold-out
   (overfit); within tolerance means the improvement generalises.
5. Append a :class:`~panobbgo.self_improve.LoopHoldoutRecord` to the
   ledger (``record_type = "holdout"``) so audits can replay the
   third signal alongside iteration and guard records.

Compute cost is fixed: ``2 × holdout_iterations`` harness runs at the
end of the loop (or just ``holdout_iterations`` when the ladder has
only the seed entry — no accepted mutations to validate).

The hold-out is **disabled by default** (``holdout_base_seed = 0``)
for backward compatibility.  Pick any independent base seed (e.g.
``1234``) for unattended runs where overfitting to the training
``base_seed`` is the dominant remaining risk.

CLI:

.. code-block:: bash

   # Basic: 50 iterations on standard, hold-out at base_seed=1234
   uv run python scripts/self_improve.py run --iterations 50 \
       --mode standard --holdout-base-seed 1234

   # Stricter: fail with exit code 3 if hold-out flags overfit
   uv run python scripts/self_improve.py run --iterations 100 \
       --mode standard --holdout-base-seed 1234 \
       --holdout-eps-overfit 0.03 --fail-on-overfit

Programmatic use:

.. code-block:: python

   from panobbgo.self_improve import LoopConfig, SelfImprover

   cfg = LoopConfig(
       iterations=50,
       mode="standard",
       guard_interval=10,
       holdout_base_seed=1234,        # independent of base_seed=42
       holdout_iterations=5,
       holdout_eps_overfit=0.05,
   )
   iter_records, guard_records, holdout_records = (
       SelfImprover(cfg).run_full()
   )

   if holdout_records:
       ho = holdout_records[-1]
       if ho.overfit:
           print(f"WARNING: ladder overfits training base_seed (drift={ho.drift:+.4f})")
       else:
           print(f"Improvement generalises (drift={ho.drift:+.4f})")

The hold-out is **independent** of the guard.  Both can be on
simultaneously: the guard runs periodically inside the loop and
catches drift between iterations within the training base_seed
family; the hold-out runs once at the end and catches overfit to the
training base_seed family itself.  Together they cover the two main
overfitting modes the loop can suffer from.

Multi-seed hold-out
^^^^^^^^^^^^^^^^^^^

The single-base-seed hold-out described above reduces the entire
generalisation question to one independent draw.  When a ladder
overfits in a subtle way — for example, the accepted mutation
exploits a quirk that happens to repeat across the chosen hold-out
seed — that one draw can miss it.

The list-typed
:attr:`~panobbgo.self_improve.LoopConfig.holdout_base_seeds` knob
trades that single point estimate for a worst-case estimate over
several independent SHA-256 streams.  At the end of the loop, one
:class:`~panobbgo.self_improve.LoopHoldoutRecord` is written per seed
in the list.  The CLI then aggregates:

* **overfit** ⟺ ``any(record.overfit for record in records)``
* **worst drift** is the smallest (most negative) ``drift`` across
  seeds.

This is strictly more conservative than the single-seed check: one
bad seed flags the ladder, even when the average drift across seeds
is comfortably positive.  Cost scales linearly with the number of
seeds (each seed adds ``2 × holdout_iterations`` harness runs at the
end of the loop), still small relative to a typical training-loop
budget.

When both ``holdout_base_seed`` (scalar) and ``holdout_base_seeds``
(list) are configured, the list wins and the scalar is silently
ignored — the list is the "do exactly this" override.

Validation rules:

* Every list entry must be non-zero (``0`` is the disable sentinel).
* Every list entry must differ from
  :attr:`~panobbgo.self_improve.LoopConfig.base_seed`.
* List entries must be distinct (duplicates would re-measure the
  same stream).

CLI:

.. code-block:: bash

   # 3-seed hold-out: worst drift across 1234 / 5678 / 9012 is the
   # number the CLI reports; overfit flags if any seed regresses.
   uv run python scripts/self_improve.py run --iterations 50 \
       --mode standard --holdout-base-seeds 1234,5678,9012 \
       --fail-on-overfit

Programmatic use:

.. code-block:: python

   cfg = LoopConfig(
       iterations=50,
       mode="standard",
       holdout_base_seeds=(1234, 5678, 9012),
       holdout_iterations=5,
   )
   iter_records, guard_records, holdout_records = (
       SelfImprover(cfg).run_full()
   )
   any_overfit = any(r.overfit for r in holdout_records)
   worst_drift = min(r.drift for r in holdout_records)
   if any_overfit:
       print(f"WARNING: ladder overfits at least one hold-out seed"
             f" (worst drift={worst_drift:+.4f})")


Bootstrap CI on the aggregated drift
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The multi-seed worst-case reduction is *conservative* — one bad
seed flags the entire ladder — but it gives no sense of whether
``-0.0074`` is the typical drift, the lucky tail of a larger drift,
or a noisy artefact of a small sample.  The
:func:`~panobbgo.self_improve.aggregate_holdout_drift` helper pools
**per-iteration paired drifts** across every hold-out record and
bootstrap-resamples the mean::

    drift_{r, k} = (top_k - seed_k) - training_delta_r

where ``r`` indexes the record (one per hold-out seed) and ``k``
indexes the hold-out iteration inside that record.  With the default
``holdout_iterations=5`` and three seeds the bootstrap sees
``3 × 5 = 15`` paired samples — enough that the CI quantiles are
real distributional information rather than a degenerate point
estimate.

The CLI prints the CI alongside the worst-case verdict::

    [self_improve] hold-out aggregate: OK  worst_drift=-0.0074 ...
    [self_improve] hold-out drift CI: OK_CI  mean=-0.0012  CI95%=[-0.0037, +0.0000]  ...

Reading the line: the bootstrap places the *expected* drift at
``-0.0012`` and the 95% CI between ``-0.0037`` and ``0`` — i.e. the
data does not statistically rule out zero drift.  Compare to the
worst-case ``-0.0074`` reduction, which can be a lucky-tail
artefact of a single noisy sample.

A stricter exit-on-overfit rule uses the CI: with
``--fail-on-overfit-ci`` the loop exits with code ``3`` only when the
*upper bound* of the CI falls below ``-holdout-eps-overfit`` — i.e.
the bootstrap rules out a drift better than the tolerance at the
configured confidence level.  This is the principled sibling of
``--fail-on-overfit`` and pairs with the
:func:`~panobbgo.harness.statistical_accept` rule used elsewhere in
the loop.

.. code-block:: bash

   uv run python scripts/self_improve.py run --iterations 50 \
       --mode standard \
       --holdout-base-seeds 1234,5678,9012 \
       --fail-on-overfit-ci --holdout-ci-confidence 0.95

The per-iteration paired scores are also persisted to the JSONL
ledger as ``seed_iteration_scores`` and ``top_iteration_scores``
lists on each :class:`~panobbgo.self_improve.LoopHoldoutRecord`, so
the ``summary`` subcommand and any downstream analytics can re-run
the aggregation on stored data:

.. code-block:: python

   from panobbgo.self_improve import (
       aggregate_holdout_drift, LoopHoldoutRecord,
   )

   agg = aggregate_holdout_drift(holdout_records)
   print(
       f"mean drift {agg.mean_drift:+.4f}, "
       f"CI{int(agg.confidence * 100)}%=[{agg.ci_low:+.4f}, "
       f"{agg.ci_high:+.4f}]"
   )
   if agg.statistically_overfit:
       print("CI rules out generalisation at the configured confidence")

Backward compatibility: legacy records (written before the
per-iteration fields existed) contribute one sample each from their
cached ``drift`` value.  Mixed inputs work transparently — the
helper uses high-resolution per-iteration samples when present and
falls back to per-record point drifts otherwise.


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
