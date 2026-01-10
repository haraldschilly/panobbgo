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
   │  │ (generates) │      │              │     │ (Dask       ││
   │  └─────────────┘      └──────────────┘     │  cluster)   ││
   │         ▲                                   └──────┬──────┘│
   │         │                                          │       │
   │         │                                          ▼       │
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
- Connects to Dask distributed cluster for parallel evaluation

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
     - Analyzers (Best, Splitter)
   * - ``new_best``
     - Best analyzer
     - Heuristics (NelderMead, Nearby)
   * - ``new_min``
     - Best analyzer
     - UI, statistics collectors
   * - ``new_split``
     - Splitter analyzer
     - Heuristics (Random)
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
- :class:`~panobbgo.heuristics.extremal.Extremal`: Samples from box boundaries
- :class:`~panobbgo.heuristics.random.Random`: Uniform sampling in best leaf box

**Local refinement:**

- :class:`~panobbgo.heuristics.nearby.Nearby`: Gaussian perturbations around best
- :class:`~panobbgo.heuristics.weighted_average.WeightedAverage`: Averages points in best region

**Model-based:**

- :class:`~panobbgo.heuristics.quadratic_wls_model.QuadraticWlsModel`: Weighted least-squares quadratic surrogate

**Classical optimizers:**

- :class:`~panobbgo.heuristics.nelder_mead.NelderMead`: Randomized simplex method
- :class:`~panobbgo.heuristics.lbfgsb.LBFGSB`: L-BFGS-B in subprocess

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

1. Strategy connects to Dask distributed cluster
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

   # Send points to Dask cluster
   futures = [client.submit(problem, p) for p in points]
   # Wait for results
   results = client.gather(futures)

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

Dask Distributed
~~~~~~~~~~~~~~~~

Function evaluations run on Dask cluster:

- **Load-balanced scheduling**: Distributes jobs to available workers
- **Asynchronous execution**: Returns Future objects immediately
- **Result collection**: Wait for completion with ``future.result()`` or ``client.gather()``

**Cluster setup:**

.. code-block:: bash

   # Start local cluster with 4 workers
   dask scheduler &
   dask worker localhost:8786 --nprocs 4 &

Configuration
~~~~~~~~~~~~~

Threading and parallelism parameters in ``~/.panobbgo/config.ini``:

.. code-block:: ini

   [ipython]
   profile = default
   max_wait_for_job = 10

   [optimization]
   queue_capacity = 20      # Heuristic queue size
   jobs_per_client = 5      # Batch size per engine

Extension Points
----------------

To extend Panobbgo:

1. **Add a heuristic**: Subclass :class:`~panobbgo.core.Heuristic`
2. **Add an analyzer**: Subclass :class:`~panobbgo.core.Analyzer`
3. **Add a strategy**: Subclass :class:`~panobbgo.core.StrategyBase`
4. **Define a problem**: Subclass :class:`~panobbgo.lib.lib.Problem`
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
