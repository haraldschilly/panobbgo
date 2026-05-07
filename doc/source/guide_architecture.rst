Architecture Overview
=====================

This section describes Panobbgo's architecture, explaining how the components fit together
and interact during optimization.

High-Level Design
-----------------

Panobbgo follows an **event-driven, modular architecture** where independent components
communicate through an :class:`~panobbgo.core.EventBus`. This design enables:

- **Extensibility**: Add new heuristics or analyzers without modifying existing code
- **Composability**: Mix and match components to create custom optimization strategies
- **Decoupling**: Components don't depend directly on each other
- **Parallelism**: Events processed asynchronously in separate threads

Data Flow Diagram
~~~~~~~~~~~~~~~~~

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────┐
   │                         Strategy                            │
   │  ┌─────────────┐      ┌──────────────┐     ┌─────────────┐│
   │  │  Heuristic  │─────▶│ Point Queue  │────▶│  Evaluator  ││
   │  │ (generates) │      │              │     │             ││
   │  └─────────────┘      └──────────────┘     └──────┬──────┘│
   │         ▲                                          │      │
   │         │                                          │      │
   │         │                                          ▼      │
   │  ┌──────┴──────┐       ┌──────────────┐    ┌──────────┐  │
   │  │  Analyzer   │◀──────│  EventBus    │◀───│ Results  │  │
   │  │ (processes) │       │ (publishes)  │    │ Database │  │
   │  └─────────────┘       └──────────────┘    └──────────┘  │
   └─────────────────────────────────────────────────────────────┘

Core Components
---------------

StrategyBase
~~~~~~~~~~~~

:class:`~panobbgo.core.StrategyBase` is the main orchestrator that:

- Manages the optimization loop
- Coordinates heuristics, analyzers, and evaluators
- Tracks budget (number of evaluations)
- Manages parallel evaluation of objective functions

**Key methods:**

- ``__init__(problem, **kwargs)``: Initialize with problem definition
- ``add(Heuristic, **kwargs)``: Register a heuristic
- ``add_analyzer(Analyzer, **kwargs)``: Register an analyzer
- ``start()``: Run the optimization loop
- ``execute()``: Abstract method to get next points (implemented by subclasses)

**Properties:**

- ``best``: Current best result
- ``results``: The results database
- ``heuristics``: List of registered heuristics
- ``analyzers``: List of registered analyzers

Results Database
~~~~~~~~~~~~~~~~

:class:`~panobbgo.core.Results` stores all evaluated points in a pandas DataFrame.

**Structure:**

The DataFrame uses MultiIndex columns:

- :math:`(x_0, x_1, \ldots, x_{n-1})`: Coordinate values
- :math:`fx`: Objective function value
- :math:`(cv_0, cv_1, \ldots, cv_{m-1})`: Individual constraint violations
- :math:`cv`: Total constraint violation (L2 norm)
- :math:`who`: Name of heuristic that generated this point
- :math:`error`: Estimated error margin

**Methods:**

- ``add_results(new_results)``: Add new results and publish ``new_results`` event
- ``__len__()``: Number of evaluations performed

EventBus
~~~~~~~~

:class:`~panobbgo.core.EventBus` implements the publisher-subscriber pattern.

**How it works:**

1. Modules define methods named ``on_<event_name>(self, **kwargs)``
2. EventBus automatically discovers these methods via introspection
3. When an event is published, all subscribers are notified in separate threads
4. Each subscription runs in a daemon thread to avoid blocking

**Common events:**

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Event
     - Published By
     - Common Subscribers
   * - ``start``
     - Strategy
     - All modules (initialization)
   * - ``new_results``
     - Results
     - Analyzers (Best, Splitter, Sensitivity, Restart)
   * - ``new_best``
     - Best analyzer
     - Heuristics (NelderMead, Nearby)
   * - ``new_min``
     - Best analyzer
     - UI, statistics collectors
   * - ``new_split``
     - Splitter analyzer
     - Heuristics (Random)
   * - ``new_sensitivity``
     - Sensitivity analyzer
     - Heuristics (Nearby — scales perturbations by dimension importance)
   * - ``restart``
     - Restart analyzer
     - Heuristics (reset and re-explore)
   * - ``finished``
     - Strategy
     - All modules (cleanup)

Module Base Classes
~~~~~~~~~~~~~~~~~~~

:class:`~panobbgo.core.Module` is the abstract parent for :class:`~panobbgo.core.Heuristic`
and :class:`~panobbgo.core.Analyzer`.

**Lifecycle:**

1. ``__init__(strategy, **kwargs)``: Construction with parameter storage
2. ``__start__()``: Called before optimization begins
3. Event handlers: ``on_<event>()`` methods invoked during optimization
4. ``__stop__()``: Called when optimization terminates

**Properties:**

- ``strategy``: Reference to the strategy
- ``config``: Configuration object
- ``eventbus``: EventBus instance
- ``problem``: Problem definition
- ``results``: Results database
- ``logger``: Logger for this module

Heuristics (Point Generators)
------------------------------

:class:`~panobbgo.core.Heuristic` extends Module to generate candidate points.

Architecture
~~~~~~~~~~~~

Each heuristic maintains:

- **Output queue**: FIFO queue with configurable capacity (default: 20)
- **Active state**: Whether it's still generating points
- **Performance score**: Tracked by StrategyRewarding for adaptive selection

**Key methods:**

- ``emit(point)`` or ``emit(points)``: Add points to output queue
- ``get_points(limit)``: Drain up to ``limit`` points from queue
- ``active``: Property indicating if heuristic has more points

**Typical pattern:**

.. code-block:: python

   class MyHeuristic(Heuristic):
       def on_start(self):
           # Generate initial points
           for i in range(10):
               x = self.problem.random_point()
               self.emit(Point(x, self.name))

       def on_new_best(self, best):
           # React to improvements
           x_new = best.x + 0.1 * np.random.randn(self.problem.dim)
           x_new = self.problem.project(x_new)
           self.emit(Point(x_new, self.name))

Implemented Heuristics
~~~~~~~~~~~~~~~~~~~~~~

**Initialization:**

- :class:`~panobbgo.heuristics.center.Center`: Returns center of bounding box
- :class:`~panobbgo.heuristics.zero.Zero`: Returns zero vector

**Space-filling:**

- :class:`~panobbgo.heuristics.latin_hypercube.LatinHypercube`: Stratified sampling with parameter ``div``
- :class:`~panobbgo.heuristics.sobol.Sobol`: Low-discrepancy Sobol' quasi-random
  sequence with parameter ``n`` (powers of two preferred); Owen-scrambled by
  default so different reps see independent point sets
- :class:`~panobbgo.heuristics.extremal.Extremal`: Samples from box boundaries
- :class:`~panobbgo.heuristics.random.Random`: Uniform sampling in best leaf box

**Local refinement:**

- :class:`~panobbgo.heuristics.nearby.Nearby`: Perturbations around best; sensitivity-aware when
  :class:`~panobbgo.analyzers.sensitivity.Sensitivity` is active — scales perturbations along
  each dimension proportionally to its importance score
- :class:`~panobbgo.heuristics.weighted_average.WeightedAverage`: Averages points in best region

**Model-based (surrogate):**

- :class:`~panobbgo.heuristics.quadratic_wls_model.QuadraticWlsModel`: Weighted least-squares quadratic surrogate
- :class:`~panobbgo.heuristics.gaussian_process.GaussianProcessHeuristic`: Gaussian Process surrogate with EI / UCB / PI
  acquisition functions (scikit-learn backend).  Supports constrained Expected Improvement (EIC)
  when the problem has active constraint violations.  Gold standard for expensive black-box
  optimization — builds an accurate probabilistic model and queries it via the acquisition function.
- :class:`~panobbgo.heuristics.claude_heuristic.ClaudeHeuristic`: Mixture of Gaussians over elite points (cluster-based adaptive search)

**Classical optimizers:**

- :class:`~panobbgo.heuristics.nelder_mead.NelderMead`: Randomized simplex method
- :class:`~panobbgo.heuristics.lbfgsb.LBFGSB`: L-BFGS-B in subprocess

**Population-based (global search):**

- :class:`~panobbgo.heuristics.cma_es.CMAES`: Covariance Matrix Adaptation Evolution Strategy
  **with IPOP and BIPOP restart support**.
  The gold standard for derivative-free optimization of continuous functions.  Maintains a
  multivariate Gaussian search distribution N(m, σ²C) and adapts both the step size σ and
  covariance matrix C online.  Invariant under order-preserving objective transformations and
  orthogonal search-space transformations; excels on ill-conditioned and ridge-following problems
  such as Rosenbrock.  Implemented in pure NumPy with asynchronous generation tracking that
  is compatible with panobbgo's threaded evaluation model.

  When paired with the :class:`~panobbgo.analyzers.restart.Restart` analyzer, the heuristic
  supports two restart schemes selected by ``restart_mode``:

  * ``"ipop"`` — **IPOP-CMA-ES** (Increasing Population CMA-ES; Auger & Hansen, CEC 2005):
    each restart multiplies the population size λ → ``ipop_factor`` · λ (default 2.0),
    resets the covariance to identity, and moves the search mean to a new diverse center.
    Good for moderately multimodal problems.

  * ``"bipop"`` — **BIPOP-CMA-ES** (Hansen, GECCO 2009): alternates between two restart
    regimes — a *large* regime with geometric population growth (``λ_l = 2^k · λ_default``)
    and a *small* regime with a small population and a random small step size
    (``σ_s = σ_default · 10^(-2·U[0,1])``).  After each restart, the regime that has
    consumed *fewer* cumulative evaluations is selected next, balancing exploitation
    (large regime fits the local geometry) with exploration (small regime probes diverse
    regions cheaply).  This is the BBOB-2009 winning algorithm and the de-facto gold
    standard for highly multimodal problems with limited evaluation budget.

  Both schemes preserve the accumulated result history; only the CMA-ES distribution
  state is reset.

- :class:`~panobbgo.heuristics.differential_evolution.DifferentialEvolution`: Differential Evolution
  mutation/crossover/selection operators run against the accumulated result database.  Excels on
  multimodal landscapes such as Rastrigin and Schwefel where purely local methods get trapped.

- :class:`~panobbgo.heuristics.pso.PSO`: Asynchronous Particle Swarm Optimization with the canonical
  Clerc–Kennedy (2002) constriction-coefficient parameters (``w = χ ≈ 0.7298``, ``c1 = c2 ≈ 1.49618``).
  Each particle carries a position, a velocity, and a memory of its personal best; on every step
  the velocity is pulled toward the personal best and toward the *swarm best the particle is
  allowed to see* with random per-component weights.  Two swarm topologies select that visible
  attractor: ``"gbest"`` (default) — every particle sees the single global best; or ``"lbest"`` —
  a wrap-around ring of width ``2·k_neighbors + 1`` (Kennedy & Mendes 2002), trading slower
  information diffusion for stronger multimodal exploration.  An opt-in ``w_end`` argument enables
  the Shi–Eberhart (1998) linearly-decreasing inertia schedule paced by the strategy's evaluation
  budget; the default constant ``w`` reproduces the original Clerc-Kennedy behaviour.  PSO's
  *momentum* and *social* dynamics are markedly different from CMA-ES (covariance re-sampling) and
  DE (recombination of three random members) — fast contraction once a basin is found, while
  inertia retains the prior search direction.  Velocities are clamped to a configurable fraction
  of each box dimension to prevent the swarm from exploding outside the search box.  Supports
  IPOP-style warm restarts via the :class:`~panobbgo.analyzers.restart.Restart` analyzer: on a
  ``restart`` event the swarm is scattered around the suggested center and the global memory is
  wiped while the strategy keeps its accumulated result history.

**Constraint-focused:**

- :class:`~panobbgo.heuristics.feasible_search.FeasibleSearch`: Beta(2,1)-biased line search
  between the last feasible and last infeasible point; efficiently locates the constraint boundary.
- :class:`~panobbgo.heuristics.constraint_gradient.ConstraintGradient`: Approximate constraint
  gradient via finite-differences; perturbs the best infeasible point in the direction that most
  reduces constraint violation.
- :class:`~panobbgo.heuristics.local_penalty_search.LocalPenaltySearch`: Local search with an
  adaptive penalty function to steer towards feasibility while minimising the objective.
- :class:`~panobbgo.heuristics.repair.ConstraintRepair`: Repairs infeasible points by projecting
  them back into the feasible region using nearest-feasible-point logic.

Analyzers (Result Processors)
------------------------------

:class:`~panobbgo.core.Analyzer` extends Module to process results and maintain derived information.

Architecture
~~~~~~~~~~~~

Analyzers typically:

- Subscribe to ``new_results`` event
- Maintain internal state (Pareto front, spatial decomposition, statistics)
- Publish derived events to trigger heuristics or other analyzers

Implemented Analyzers
~~~~~~~~~~~~~~~~~~~~~

**Best Tracker**

:class:`~panobbgo.analyzers.best.Best` maintains:

- Best feasible point (:math:`CV(x) = 0`, minimum :math:`f(x)`)
- Best infeasible point (minimum :math:`CV(x)`)
- Pareto front of :math:`(f(x), CV(x))` pairs

**Events published:**

- ``new_best``: New best point (considering constraints)
- ``new_min``: New minimum :math:`f(x)` among feasible points
- ``new_cv``: New minimum :math:`CV(x)`
- ``new_pareto``: Pareto front updated

**Splitter**

:class:`~panobbgo.analyzers.splitter.Splitter` manages hierarchical box decomposition:

- Maintains tree of boxes splitting the search space
- Splits boxes when they contain sufficient points
- Identifies "best leaf box" containing current best point

**Events published:**

- ``new_split``: A box was split into children

**Grid**

:class:`~panobbgo.analyzers.grid.Grid` maintains a simple spatial grid for grouping nearby points.

**Dedensifyer**

:class:`~panobbgo.analyzers.dedensifyer.Dedensifyer` maintains a hierarchical grid to avoid
clustering, keeping only min/max representatives per region.

**Sensitivity**

:class:`~panobbgo.analyzers.sensitivity.Sensitivity` estimates per-dimension importance from
evaluation history using rank correlation. Publishes ``new_sensitivity`` events that heuristics
can use to focus search on the most influential dimensions.

**Events published:**

- ``new_sensitivity``: Importance scores for each dimension (array in [0, 1])

**Restart**

:class:`~panobbgo.analyzers.restart.Restart` detects search stagnation (no improvement in
``patience`` evaluations) and publishes ``restart`` events with a suggested new center point.
Enables multi-start optimization without losing accumulated results.

**Events published:**

- ``restart``: New center point and reason string

Strategies (Orchestration)
---------------------------

Strategy subclasses implement the ``execute()`` method to determine which points to evaluate next.

StrategyRoundRobin
~~~~~~~~~~~~~~~~~~

:class:`~panobbgo.strategies.round_robin.StrategyRoundRobin` cycles through heuristics in fixed order:

.. code-block:: python

   def execute(self):
       points = []
       for h in self.heuristics:
           points.extend(h.get_points(batch_size))
       return points

**Characteristics:**

- Predictable, deterministic
- No adaptation to problem
- Good baseline for comparison

StrategyRewarding
~~~~~~~~~~~~~~~~~

:class:`~panobbgo.strategies.rewarding.StrategyRewarding` implements multi-armed bandit:

.. code-block:: python

   def execute(self):
       points = []
       # Calculate selection probabilities based on performance
       for h in self.heuristics:
           prob = (h.performance + smooth) / (total_performance + smooth * |H|)
           nb_points = round(target * prob)
           points.extend(h.get_points(nb_points))
       return points

**Characteristics:**

- Adaptive: learns which heuristics work
- Probabilistic: maintains exploration
- Reward function: :math:`R(x) = 1 - e^{-(f_{best} - f(x))}`
- Discount factor: causes old successes to fade

StrategyUCB
~~~~~~~~~~~

:class:`~panobbgo.strategies.ucb.StrategyUCB` extends the multi-armed bandit with an Upper
Confidence Bound selection rule (UCB1 / UCB-tuned variant):

.. math::

   \text{score}(h) = \bar{r}_h + C \sqrt{\frac{\ln N}{n_h}}

where :math:`\bar{r}_h` is the average reward, :math:`N` is total selections, :math:`n_h` is
selections of heuristic :math:`h`, and :math:`C` is an exploration constant.  UCB provides
a principled exploration–exploitation trade-off with theoretical regret bounds.

StrategyThompsonSampling
~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~panobbgo.strategies.thompson.StrategyThompsonSampling` uses Thompson Sampling (Beta–
Bernoulli bandit) to select heuristics.  Each heuristic maintains a Beta posterior
:math:`\text{Beta}(\alpha_h, \beta_h)` over its success probability.  At each step, an arm is
selected by drawing a sample from each posterior and picking the largest.

**Advantages:** Naturally balances exploration and exploitation; often outperforms UCB in practice.

StrategyLinUCB
~~~~~~~~~~~~~~

:class:`~panobbgo.strategies.contextual.StrategyLinUCB` implements a contextual bandit using
disjoint linear UCB models.  Each heuristic has its own linear reward model:

.. math::

   \hat{r}_h(s) = \theta_h^\top s + \alpha \sqrt{s^\top A_h^{-1} s}

where :math:`s` is a feature vector capturing budget progress, recent success rate, and a bias
term.  LinUCB adapts selection to the *current optimization context*, not just historical averages.

StrategyPhased
~~~~~~~~~~~~~~

:class:`~panobbgo.strategies.phased.StrategyPhased` divides the evaluation budget into phases.
Each phase can use a different sub-strategy or heuristic portfolio.  The phase transitions are
triggered by evaluation count, enabling a two-stage approach (e.g., initial exploration followed
by focused exploitation).

**Typical usage:** Phase 1 = LatinHypercube global exploration → Phase 2 = GP-guided exploitation.

Execution Flow
--------------

Initialization Phase
~~~~~~~~~~~~~~~~~~~~

1. User creates Strategy with Problem:

   .. code-block:: python

      strategy = StrategyRewarding(problem, max_evaluations=1000)

2. User registers heuristics and analyzers:

   .. code-block:: python

      strategy.add(LatinHypercube, div=5)
      strategy.add(Random)
      strategy.add_analyzer(Best)
      strategy.add_analyzer(Splitter)

3. User calls ``strategy.start()``

Startup
~~~~~~~

1. Strategy initializes the evaluation environment
2. Strategy calls ``__start__()`` on all modules
3. EventBus publishes ``start`` event
4. Modules initialize (e.g., LatinHypercube generates initial grid)

Main Loop
~~~~~~~~~

While budget remaining:

1. **Generate points**:

   .. code-block:: python

      points = strategy.execute()  # Get points from heuristics

2. **Evaluate in parallel**:

   .. code-block:: python

      # Send points to evaluation environment
      results = strategy._evaluate(points)

3. **Store results**:

   .. code-block:: python

      results_db.add_results(results)

4. **Publish event**:

   .. code-block:: python

      eventbus.publish("new_results", results=results)

5. **Analyzers process**:

   - Best checks for improvements → publishes ``new_best`` if found
   - Splitter updates box tree → publishes ``new_split`` if box split

6. **Heuristics react**:

   - NelderMead starts simplex on ``new_best``
   - Random updates sampling region on ``new_split``

7. **Check termination**:

   .. code-block:: python

      if len(results_db) >= max_evaluations:
          break

Termination
~~~~~~~~~~~

1. EventBus publishes ``finished`` event
2. Modules call ``__stop__()`` for cleanup
3. Strategy returns ``best`` result

Threading Model
---------------

Event Handling
~~~~~~~~~~~~~~

Each event subscription runs in a **daemon thread**:

- Non-blocking: publishing returns immediately
- Concurrent: multiple handlers run simultaneously
- Fire-and-forget: no return values from handlers

**Thread safety:**

- Results database uses pandas (generally thread-safe for reads)
- Heuristic queues use thread-safe operations
- Analyzers should use locks if maintaining mutable state

Parallel Evaluation
~~~~~~~~~~~~~~~~~~~

Function evaluations can run in parallel using different engines:

- **Threaded (Default)**: Evaluations run in a local thread pool. Suitable for lightweight functions or when running on a single machine.
- **Dask Distributed**: Connects to a Dask cluster for large-scale distributed evaluation.

**Dask setup (optional):**

.. code-block:: bash

   # Start local cluster with 4 workers
   dask scheduler &
   dask worker localhost:8786 --nprocs 4 &

Configuration
~~~~~~~~~~~~~

Parallelism parameters in ``config.yaml`` or ``~/.panobbgo/config.ini``:

.. code-block:: yaml

   evaluation:
     method: threaded  # or 'dask'
     # Dask specific configuration
     dask:
       address: localhost:8786

.. code-block:: ini

   [optimization]
   queue_capacity = 20      # Heuristic queue size


Extension Points
----------------

To extend Panobbgo:

1. **Add a heuristic**: Subclass :class:`~panobbgo.core.Heuristic`
2. **Add an analyzer**: Subclass :class:`~panobbgo.core.Analyzer`
3. **Add a strategy**: Subclass :class:`~panobbgo.core.StrategyBase`
4. **Define a problem**: Subclass :class:`~panobbgo.lib.Problem`
5. **Create custom events**: Publish with ``eventbus.publish(event_name, **kwargs)``

See :doc:`guide_extending` for detailed examples.

Design Principles
-----------------

The architecture follows these principles:

**Modularity**
  Components are independent and interchangeable

**Event-driven**
  Loose coupling through EventBus

**Composability**
  Mix and match heuristics, analyzers, strategies

**Extensibility**
  Add new components without modifying existing code

**Parallelism**
  Designed for distributed evaluation from the start

**Transparency**
  All events and data flows are observable and logged
