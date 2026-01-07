Introduction to Panobbgo
========================

What is Panobbgo?
-----------------

**Panobbgo** (Parallel Noisy Black-Box Global Optimization) is a Python framework for optimizing black-box functions where:

- **Black-box**: You can only evaluate the function, not analyze its structure
- **Global**: You seek the global optimum, not just local minima
- **Noisy**: Function evaluations may be stochastic or have errors
- **Constrained**: You can handle constraint violations
- **Parallel**: Evaluations can run simultaneously on distributed clusters

The Black-Box Optimization Problem
-----------------------------------

Given:

- An unknown function :math:`f: \mathbb{R}^n \rightarrow \mathbb{R}`
- A bounding box :math:`B = [x_1^{min}, x_1^{max}] \times \cdots \times [x_n^{min}, x_n^{max}]`
- Optional constraint functions :math:`g_i(x) \leq 0` for :math:`i = 1, \ldots, m`
- A limited evaluation budget :math:`N_{max}`

Find:

.. math::

   x^* = \arg\min_{x \in B} f(x) \quad \text{subject to} \quad g_i(x) \leq 0, \; \forall i

Real-World Challenges
~~~~~~~~~~~~~~~~~~~~~

1. **Expensive evaluations**: Each :math:`f(x)` might take minutes to hours
2. **No gradients**: Cannot use derivative-based methods
3. **Noise**: :math:`f(x + \epsilon) \approx f(x)` but not exactly
4. **Unknown smoothness**: :math:`f` might be discontinuous or highly multimodal
5. **Limited budget**: Can only afford 100-10,000 evaluations

Panobbgo's Solution
~~~~~~~~~~~~~~~~~~~

Panobbgo addresses these challenges through:

1. **Multiple search heuristics** working in parallel
2. **Adaptive strategy selection** (multi-armed bandit approach)
3. **Result database** to avoid re-evaluations and learn from history
4. **Event-driven architecture** for modular, extensible design
5. **Dask distributed integration** for parallel evaluation

Key Features
------------

Architecture
~~~~~~~~~~~~

- **Event-driven design**: Modules communicate through an :class:`~panobbgo.core.EventBus`
- **Modular components**: Easy to add new heuristics, analyzers, or strategies
- **Result database**: Pandas-based storage with automatic event publishing
- **Budget tracking**: Automatic termination when evaluation limit reached

Point Generators (Heuristics)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Panobbgo includes 10 heuristics with different search strategies:

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Type
     - Heuristics
     - Strategy
   * - Initialization
     - Center, Zero
     - Start at known points
   * - Random Sampling
     - Random, Extremal, Latin Hypercube
     - Explore broadly
   * - Local Search
     - Nearby, Weighted Average
     - Refine around best
   * - Model-Based
     - Quadratic WLS
     - Fit surrogate, optimize
   * - Classical Optimizers
     - Nelder-Mead, L-BFGS-B
     - Local optimization

Adaptive Selection
~~~~~~~~~~~~~~~~~~

The :class:`~panobbgo.strategies.rewarding.StrategyRewarding` implements a **multi-armed bandit** approach:

- Each heuristic is an "arm" of a bandit machine
- "Pulling an arm" = requesting a point from that heuristic
- "Reward" = improvement in objective value
- Better-performing heuristics are selected more often
- Additive smoothing ensures exploration continues

Result Analysis
~~~~~~~~~~~~~~~

Four analyzers process results and maintain derived information:

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Analyzer
     - Maintains
     - Publishes
   * - :class:`~panobbgo.analyzers.best.Best`
     - Best points, Pareto front
     - ``new_best``, ``new_min``, ``new_cv``, ``new_pareto``
   * - :class:`~panobbgo.analyzers.splitter.Splitter`
     - Hierarchical box decomposition
     - ``new_split``, identifies best leaf box
   * - :class:`~panobbgo.analyzers.grid.Grid`
     - Spatial grid grouping
     - Grid-based neighborhoods
   * - :class:`~panobbgo.analyzers.dedensifyer.Dedensifyer`
     - Hierarchical grid
     - Min/max representatives per region

Benchmark Problems
~~~~~~~~~~~~~~~~~~

The library includes 22 benchmark problems for testing and validation:

- **Classic**: Rosenbrock, Rastrigin, Himmelblau, Beale, Branin
- **Constrained**: RosenbrockConstraint, RosenbrockAbsConstraint
- **Stochastic**: RosenbrockStochastic
- **High-dimensional**: Problems scalable to arbitrary dimensions

When to Use Panobbgo
---------------------

Panobbgo is ideal when you have:

✓ An expensive-to-evaluate objective function
✓ No access to gradients or function derivatives
✓ Potential noise or stochasticity in evaluations
✓ Box constraints on variables
✓ Optional constraint violations to minimize
✓ Access to parallel computing resources
✓ A limited evaluation budget

Common Applications
~~~~~~~~~~~~~~~~~~~

- **Hyperparameter optimization**: Tuning machine learning models
- **Simulation calibration**: Matching simulation outputs to real data
- **Engineering design**: Optimizing complex systems (aerodynamics, structures)
- **Scientific discovery**: Parameter estimation in physical models
- **Industrial processes**: Optimizing manufacturing or chemical processes

When NOT to Use Panobbgo
~~~~~~~~~~~~~~~~~~~~~~~~~

Consider other methods if you have:

✗ Access to gradients (use gradient-based optimization)
✗ Cheap function evaluations (simple methods may suffice)
✗ Convex problems with known structure (use convex optimization)
✗ Very high dimensions (>50) without structure (consider dimension reduction first)
✗ No parallelism available and unlimited budget (simpler methods may be adequate)

Getting Started
---------------

See :doc:`guide_usage` for installation instructions and basic examples.

For a deeper understanding of the mathematical foundations, see :doc:`guide_mathematical_foundation`.

To understand the system architecture, see :doc:`guide_architecture`.
