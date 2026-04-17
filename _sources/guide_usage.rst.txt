Usage Guide
===========

This guide covers installation, basic usage, and common workflows for Panobbgo.

Installation
------------

Requirements
~~~~~~~~~~~~

- Python ≥ 3.8
- NumPy ≥ 2.0
- SciPy ≥ 1.16
- matplotlib ≥ 3.0
- pandas ≥ 2.0
- statsmodels ≥ 0.14
- Dask ≥ 2023.0

Using UV (Recommended)
~~~~~~~~~~~~~~~~~~~~~~

`UV <https://github.com/astral-sh/uv>`_ is a fast Python package manager:

.. code-block:: bash

   # Install UV
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Clone and install panobbgo
   git clone https://github.com/haraldschilly/panobbgo.git
   cd panobbgo
   uv sync --extra dev

Using pip
~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/haraldschilly/panobbgo.git
   cd panobbgo
   pip install -e ".[dev]"

Evaluation Setup
~~~~~~~~~~~~~~~~

**Threaded Evaluation (Default):**

By default, Panobbgo uses local threads for evaluation. This requires no additional setup and is suitable for most local development tasks.

**Dask Cluster (Optional):**

For large-scale distributed optimization, you can use a Dask cluster:

1. **Install Dask**:
   .. code-block:: bash

      pip install dask[distributed]

2. **Start Cluster**:
   .. code-block:: bash

      dask scheduler &
      dask worker localhost:8786 --nprocs 4 &

3. **Configure Panobbgo**:
   Set ``evaluation: method: dask`` in your ``config.yaml``.

See `Dask distributed documentation <https://docs.dask.org/en/stable/deploying.html>`_ for advanced setup
(remote clusters, Kubernetes, SLURM integration, etc.).

Configuration
~~~~~~~~~~~~~

On first run, Panobbgo creates ``~/.panobbgo/config.ini``:

.. code-block:: ini

   [dask]
   cluster_type = local                    # 'local' (auto-start) or 'remote'
   local.n_workers = 2                    # Number of local workers (default: 2)
   local.threads_per_worker = 1           # Threads per worker (default: 1)
   local.memory_limit = 2GB               # Memory per worker (default: 2GB)
   local.dashboard_address = :8787        # Dashboard port (default: :8787)
   remote.scheduler_address = tcp://localhost:8786  # For remote clusters

   [optimization]
   max_evaluations = 1000    # Evaluation budget
   queue_capacity = 20       # Heuristic queue size

   [strategy]
   smooth = 0.1              # Additive smoothing for bandit
   discount = 0.95           # Performance decay factor
   jobs_per_client = 5       # Batch size per engine

   [logging]
   level = INFO              # DEBUG, INFO, WARNING, ERROR

Edit this file to customize behavior.

Basic Usage
-----------

Minimal Example
~~~~~~~~~~~~~~~

.. code-block:: python

   from panobbgo.lib.classic import Rosenbrock
   from panobbgo.strategies.rewarding import StrategyRewarding
   from panobbgo.heuristics import Center, Random, NelderMead

   # Define problem
   problem = Rosenbrock(dim=5)

   # Create strategy
   strategy = StrategyRewarding(problem, max_evaluations=500)

   # Add heuristics
   strategy.add(Center)       # Start at center
   strategy.add(Random)       # Exploration
   strategy.add(NelderMead)   # Exploitation

   # Run optimization
   strategy.start()

   # Get results
   print(f"Best found: {strategy.best}")
   print(f"Best x: {strategy.best.x}")
   print(f"Best f(x): {strategy.best.fx}")

Complete Example
~~~~~~~~~~~~~~~~

.. code-block:: python

   from panobbgo.lib.classic import Rosenbrock
   from panobbgo.strategies.rewarding import StrategyRewarding
   from panobbgo.analyzers import Best, Splitter
   from panobbgo.heuristics import (
       Center, Zero, LatinHypercube, Random,
       Nearby, NelderMead
   )

   # Define 10-dimensional Rosenbrock
   problem = Rosenbrock(dim=10)

    # Create adaptive strategy
    strategy = StrategyRewarding(
        problem,
        max_evaluations=2000      # Budget
    )

   # Add analyzers (optional - Best is default)
   strategy.add_analyzer(Best)      # Track best points
   strategy.add_analyzer(Splitter)  # Spatial decomposition

   # Add diverse heuristics
   strategy.add(Center)                    # Initialize at center
   strategy.add(Zero)                      # Initialize at origin
   strategy.add(LatinHypercube, div=5)    # Space-filling design (5^10 grid)
   strategy.add(Random)                    # Ongoing exploration
   strategy.add(Nearby)                    # Local perturbations
   strategy.add(NelderMead)               # Simplex optimization

   # Run
   strategy.start()

   # Analyze results
   print(f"\nOptimization complete!")
   print(f"Evaluations: {len(strategy.results)}")
   print(f"Best f(x): {strategy.best.fx:.6f}")
   print(f"Best x: {strategy.best.x}")
   print(f"Found by: {strategy.best.who}")

   # Access result database
   df = strategy.results.results
   print(f"\nDataFrame shape: {df.shape}")
   print(df.head())

Defining Custom Problems
-------------------------

Basic Problem
~~~~~~~~~~~~~

Subclass :class:`~panobbgo.lib.Problem`:

.. code-block:: python

   import numpy as np
   from panobbgo.lib import Problem, BoundingBox

   class Sphere(Problem):
       """Simple sphere function: f(x) = sum(x^2)"""

       def __init__(self, dim=5):
           # Define bounding box: each variable in [-10, 10]
           box = BoundingBox(np.array([[-10.0, 10.0]] * dim))
           super().__init__(dim, box)

       def eval(self, x):
           """Evaluate objective function"""
           return np.sum(x ** 2)

   # Use it
   problem = Sphere(dim=10)
   strategy = StrategyRewarding(problem, max_evaluations=500)
   # ... add heuristics and run ...

Problem Wrappers
~~~~~~~~~~~~~~~~

Instead of modifying your problem class, use composable wrappers from :mod:`panobbgo.lib.wrappers`:

.. code-block:: python

   from panobbgo.lib.wrappers import NormalizedProblem, NoisyProblem, LogTransformProblem

   # Normalize all dimensions to [0, 1]
   problem = NormalizedProblem(Rosenbrock(dim=5))

   # Add noise for robustness testing (seed for reproducibility)
   problem = NoisyProblem(Rosenbrock(dim=5), noise_std=0.1, seed=42)

   # Log-transform for objectives spanning orders of magnitude
   problem = LogTransformProblem(MyProblem(), offset=0.0)

   # Compose multiple wrappers
   problem = NormalizedProblem(NoisyProblem(MyProblem(), noise_std=0.05))

Wrappers are transparent — the framework sees a standard :class:`~panobbgo.lib.Problem`
with the transformed box and evaluation.

Noisy Problem (Manual)
~~~~~~~~~~~~~~~~~~~~~~

You can also add stochasticity directly in ``eval()``:

.. code-block:: python

   class NoisySphere(Problem):
       def __init__(self, dim=5, noise_std=0.1):
           box = BoundingBox(np.array([[-10.0, 10.0]] * dim))
           super().__init__(dim, box)
           self.noise_std = noise_std

       def eval(self, x):
           """Noisy evaluation"""
           true_value = np.sum(x ** 2)
           noise = np.random.randn() * self.noise_std
           return true_value + noise

Constrained Problem
~~~~~~~~~~~~~~~~~~~

Override ``eval_constraints()`` to return violation vector:

.. code-block:: python

   class ConstrainedProblem(Problem):
       def __init__(self):
           # 2D problem: x in [-5, 5], y in [-5, 5]
           box = BoundingBox(np.array([[-5, 5], [-5, 5]]))
           super().__init__(dim=2, box=box)

       def eval(self, x):
           """Objective: minimize (x-1)^2 + (y-2)^2"""
           return (x[0] - 1)**2 + (x[1] - 2)**2

       def eval_constraints(self, x):
           """Constraints:
           g1: x + y <= 1  (i.e., x + y - 1 <= 0)
           g2: x >= 0
           """
           g1 = x[0] + x[1] - 1.0
           g2 = -x[0]
           # Return positive violations
           return np.array([max(0, g1), max(0, g2)])

   # Panobbgo will minimize objective while trying to satisfy constraints
   problem = ConstrainedProblem()
   strategy = StrategyRewarding(problem, max_evaluations=300)

   # You can configure the constraint handling method in ~/.panobbgo/config.ini
   # [optimization]
   # constraint_handler = AugmentedLagrangianConstraintHandler

   strategy.add(Center)
   strategy.add(Random)
   # Add FeasibleSearch to actively target feasible regions
   from panobbgo.heuristics import FeasibleSearch, ConstraintGradient
   strategy.add(FeasibleSearch)
   strategy.add(ConstraintGradient)
   strategy.add(NelderMead)
   strategy.start()

   print(f"Best feasible: {strategy.best}")
   print(f"Constraint violation: {strategy.best.cv}")

Constraint Handling Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Panobbgo supports different constraint handling strategies, configurable in ``config.ini``:

1. **DefaultConstraintHandler** (default):
   Lexicographic ordering. Prioritizes feasibility (cv=0) over objective function value.
   Good for general use where feasibility is strict.

2. **PenaltyConstraintHandler**:
   Uses a static penalty: $P(x) = f(x) + \rho \cdot cv(x)^{exponent}$.
   Useful if slight violations are acceptable or gradients lead out of feasible region.

3. **DynamicPenaltyConstraintHandler**:
   Penalty coefficient increases over time. Starts low to allow exploration of infeasible regions, then tightens.

4. **AugmentedLagrangianConstraintHandler**:
   Implements the Augmented Lagrangian Method. Adaptively updates multipliers $\lambda$ and penalty $\mu$ based on progress.
   Can be more robust for equality constraints or hard inequality constraints.

5. **EpsilonConstraintHandler**:
   Uses the $\epsilon$-Constrained Method. Initially treats points with small violations ($cv(x) \le \epsilon(t)$) as feasible.
   $\epsilon(t)$ decreases from `epsilon_start` to 0 over `epsilon_cutoff` evaluations.
   Effective for finding feasible regions in difficult problems by approaching the boundary gradually.

6. **FilterConstraintHandler**:
   Uses a multi-objective filter approach (Pareto dominance on (objective, violation)).
   Accepts points that are not dominated by any previously accepted point in the filter.
   Useful for maintaining a diverse set of trade-off solutions during the search.

FeasibleSearch Heuristic
~~~~~~~~~~~~~~~~~~~~~~~~

When dealing with constraints, it is highly recommended to add the **FeasibleSearch** heuristic.
This heuristic is specifically designed to:

- Repair infeasible solutions by searching towards known feasible regions (Line Search).
- Explore the boundary of the feasible region.
- Adaptively sample around the best point to reduce constraint violations.

.. code-block:: python

   from panobbgo.heuristics import FeasibleSearch
   strategy.add(FeasibleSearch)

ConstraintGradient Heuristic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The **ConstraintGradient** heuristic estimates the gradient of the constraint violation function using finite differences from recent evaluations (or neighbors). It uses this estimated gradient to perform a descent step towards the feasible region. This is particularly useful when constraint violations are smooth.

.. code-block:: python

   from panobbgo.heuristics import ConstraintGradient
   strategy.add(ConstraintGradient)

LocalPenaltySearch Heuristic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For problems with constraints handled via penalties (e.g. Penalty or Augmented Lagrangian), the **LocalPenaltySearch** heuristic can be very effective. It uses Scipy's local optimizers (like L-BFGS-B or Nelder-Mead) to minimize the scalarized penalty function directly.

.. code-block:: python

   from panobbgo.heuristics import LocalPenaltySearch

   # Use L-BFGS-B on the penalized objective
   strategy.add(LocalPenaltySearch, method="L-BFGS-B")

ClaudeHeuristic (Cluster-Based Adaptive Search)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The **ClaudeHeuristic** identifies clusters of elite (top-performing) evaluated points,
fits a local Gaussian distribution to each cluster, and samples new candidates from the
resulting mixture model. The intuition: near groups of good points, even better points
may be hiding — especially in constrained problems where feasibility boundaries create
pockets of promising solutions.

Related to the Cross-Entropy Method and Estimation of Distribution Algorithms (EDAs).

.. code-block:: python

   from panobbgo.heuristics import ClaudeHeuristic

   # Default: top 20% elite, up to 5 clusters, 5 candidates per batch
   strategy.add(ClaudeHeuristic)

   # Customize for your problem
   strategy.add(ClaudeHeuristic,
       elite_fraction=0.3,    # Use top 30% of evaluated points
       max_clusters=3,        # At most 3 clusters
       n_candidates=10,       # Emit 10 candidates per trigger
       regularization=1e-2    # Stronger covariance regularization
   )

This heuristic is particularly effective for:

- **Multimodal landscapes**: Multiple clusters capture distinct promising regions
- **Constrained problems**: Penalty-based elite selection naturally favors feasible points
- **Medium budgets**: Needs enough evaluations to build meaningful clusters (~50+)

Expensive External Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Wrap subprocess or external program:

.. code-block:: python

   import subprocess
   import json

   class ExternalSimulation(Problem):
       def __init__(self):
           # 3D problem
           box = BoundingBox(np.array([
               [0.0, 1.0],
               [0.0, 1.0],
               [0.0, 1.0]
           ]))
           super().__init__(dim=3, box=box)

       def eval(self, x):
           """Call external simulation"""
           # Write input
           input_data = {"parameters": x.tolist()}
           with open("input.json", "w") as f:
               json.dump(input_data, f)

           # Run simulation
           result = subprocess.run(
               ["./my_simulation", "input.json"],
               capture_output=True,
               text=True,
               timeout=300  # 5 minute timeout
           )

           # Parse output
           output_data = json.loads(result.stdout)
           return output_data["objective_value"]

Choosing Heuristics
-------------------

Heuristic Portfolio
~~~~~~~~~~~~~~~~~~~

A good portfolio balances exploration and exploitation:

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Purpose
     - Heuristics
     - When to Use
   * - Initialization
     - Center, Zero
     - Always include one
   * - Space-filling
     - LatinHypercube
     - High-dimensional problems (dim > 5)
   * - Exploration
     - Random, Extremal
     - Always include Random
   * - Local search
     - Nearby, NelderMead
     - Smooth problems
   * - Model-based (surrogate)
     - GaussianProcessHeuristic, QuadraticWLS
     - Few evaluations, smooth/expensive functions (Bayesian optimization)
   * - Cluster-based
     - ClaudeHeuristic
     - Multimodal landscapes, finding hidden optima near good regions
   * - Population-based
     - DifferentialEvolution
     - Multimodal problems (Rastrigin, Schwefel); no surrogate model needed
   * - Gradient-free local
     - LBFGSB, LocalPenaltySearch
     - When local structure suspected
   * - Constraint Handling
     - FeasibleSearch, ConstraintGradient, ConstraintRepair
     - When constraints are present

Recommended Configurations
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Bayesian optimization (≤ 500 evals, smooth/expensive function):**

The gold standard for expensive black-box optimization.  The GP surrogate models the
objective; Expected Improvement (EI) acquisition balances exploration and exploitation.

.. code-block:: python

   from panobbgo.heuristics import GaussianProcessHeuristic, LatinHypercube, NelderMead, Random

   strategy = StrategyRewarding(problem, max_evaluations=200)
   strategy.add(LatinHypercube, div=4)          # Space-filling initial design
   strategy.add(GaussianProcessHeuristic,        # GP + EI acquisition
       n_restarts=5,                             # Acquisition restarts (speed/quality)
       xi=0.01)                                  # EI exploration parameter
   strategy.add(NelderMead)                      # Local refinement
   strategy.add(Random)                          # Fallback exploration

**Low-dimensional (dim ≤ 5):**

.. code-block:: python

   strategy.add(Center)
   strategy.add(LatinHypercube, div=10)
   strategy.add(Random)
   strategy.add(Nearby)
   strategy.add(NelderMead)
   strategy.add(GaussianProcessHeuristic)
   strategy.add(ClaudeHeuristic)  # Cluster-based search

**Medium-dimensional (5 < dim ≤ 20):**

.. code-block:: python

   strategy.add(Center)
   strategy.add(LatinHypercube, div=5)
   strategy.add(Random)
   strategy.add(NelderMead)
   strategy.add(LBFGSB)

**High-dimensional (dim > 20):**

.. code-block:: python

   strategy.add(Center)
   strategy.add(Random)
   strategy.add(NelderMead)
   strategy.add(LBFGSB)

**Multimodal problems (Rastrigin, Schwefel):**

.. code-block:: python

   strategy.add(LatinHypercube, div=5)
   strategy.add(DifferentialEvolution)   # Global search
   strategy.add(Random)                  # Exploration
   strategy.add(NelderMead)              # Local refinement

**Very noisy problems:**

.. code-block:: python

   strategy.add(Center)
   strategy.add(LatinHypercube, div=5)
   strategy.add(Random)
   strategy.add(Nearby)
   # Avoid gradient-based methods (LBFGSB)

Choosing a Strategy
-------------------

StrategyRoundRobin
~~~~~~~~~~~~~~~~~~

Use when:

- You want predictable, deterministic behavior
- Comparing different heuristic portfolios
- Debugging or understanding heuristic behavior

.. code-block:: python

   from panobbgo.strategies.round_robin import StrategyRoundRobin
   strategy = StrategyRoundRobin(problem, max_evaluations=1000)

StrategyRewarding
~~~~~~~~~~~~~~~~~

Use when (recommended for most cases):

- You want adaptive selection based on performance
- Problem structure is unknown
- You have diverse heuristics

.. code-block:: python

   from panobbgo.strategies.rewarding import StrategyRewarding
   strategy = StrategyRewarding(
       problem,
       max_evaluations=1000,
       smooth=0.1,      # Exploration parameter
       discount=0.95    # Performance decay
   )

StrategyThompsonSampling
~~~~~~~~~~~~~~~~~~~~~~~~

Use when:

- You want a principled probabilistic approach (Beta-Bernoulli bandit)
- You want to balance exploration and exploitation automatically without manual tuning
- Suitable for both stationary and non-stationary (if adapted) environments

.. code-block:: python

   from panobbgo.strategies.thompson import StrategyThompsonSampling
   strategy = StrategyThompsonSampling(
       problem,
       max_evaluations=1000
   )

StrategyUCB
~~~~~~~~~~~

Use when:

- You want a deterministic bandit strategy with theoretical guarantees
- You prefer the Upper Confidence Bound (UCB1) algorithm

.. code-block:: python

   from panobbgo.strategies.ucb import StrategyUCB
   strategy = StrategyUCB(
       problem,
       max_evaluations=1000,
       ucb_c=1.414  # Exploration constant (default sqrt(2))
   )

StrategyLinUCB
~~~~~~~~~~~~~~

Use when:

- You want a contextual bandit strategy that adapts based on optimization state
- Uses context features (budget progress, success rate) to select heuristics

.. code-block:: python

   from panobbgo.strategies.contextual import StrategyLinUCB
   strategy = StrategyLinUCB(
       problem,
       max_evaluations=1000,
       linucb_alpha=2.0  # Exploration parameter
   )

StrategyPhased
~~~~~~~~~~~~~~

Use when:

- You want to divide the evaluation budget into distinct phases with different strategies
- Early exploration (e.g., random sampling) should transition to later exploitation (e.g., model-based search)
- Different heuristic portfolios are appropriate at different stages of the optimization

``StrategyPhased`` is a meta-strategy that composes existing strategies across budget phases.
Each phase specifies a fraction of the total budget, a sub-strategy for heuristic selection,
and the heuristics to use in that phase. All heuristics are registered from the start, so
model-building heuristics (e.g., Gaussian Process) accumulate data even before their phase
is active.

.. code-block:: python

   from panobbgo.strategies.phased import StrategyPhased
   from panobbgo.strategies.round_robin import StrategyRoundRobin
   from panobbgo.strategies.rewarding import StrategyRewarding
   from panobbgo.strategies.ucb import StrategyUCB
   from panobbgo.heuristics import (
       Center, Random, LatinHypercube,
       NelderMead, GaussianProcessHeuristic, LBFGSB
   )

   problem = Rosenbrock(dim=5)

   strategy = StrategyPhased(problem, phases=[
       {
           # Phase 1: Explore the space (first 25% of budget)
           "pct": 25,
           "strategy": (StrategyRoundRobin, {"size": 10}),
           "heuristics": [
               (Center, {}),
               (Random, {}),
               (LatinHypercube, {}),
           ],
       },
       {
           # Phase 2: Adaptive exploitation (remaining 75%)
           # No "pct" on the last phase — it gets the remaining budget.
           "strategy": (StrategyRewarding, {}),
           "heuristics": [
               (NelderMead, {}),
               (GaussianProcessHeuristic, {}),
               (LBFGSB, {}),
           ],
       },
   ], max_evaluations=1000)

   strategy.start()

Phase configuration:

- ``pct``: Percentage of total budget for this phase. Optional on the last phase
  (it receives the remaining budget automatically).
- ``strategy``: A ``(StrategyClass, kwargs_dict)`` tuple specifying the selection
  algorithm and its parameters.
- ``heuristics``: A list of ``(HeuristicClass, kwargs_dict)`` tuples.

At phase transitions, bandit/selection statistics (UCB counts, performance scores, etc.)
are reset for the new phase's heuristics, but domain state (GP models, accumulated data)
is preserved.

Three-phase example with gradual refinement:

.. code-block:: python

   strategy = StrategyPhased(problem, phases=[
       {
           "pct": 20,
           "strategy": (StrategyRoundRobin, {"size": 10}),
           "heuristics": [(Random, {}), (LatinHypercube, {})],
       },
       {
           "pct": 30,
           "strategy": (StrategyUCB, {"ucb_c": 1.0}),
           "heuristics": [(Random, {}), (NelderMead, {})],
       },
       {
           "strategy": (StrategyRewarding, {}),
           "heuristics": [(GaussianProcessHeuristic, {}), (LBFGSB, {})],
       },
   ], max_evaluations=2000)

Analyzing Results
-----------------

Accessing the Database
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Get results DataFrame
   df = strategy.results.results

   # Best point
   best = strategy.best
   print(f"f(x) = {best.fx}, x = {best.x}, by {best.who}")

   # All evaluations by a specific heuristic
   random_results = df[df[('who', 0)] == 'Random']

   # Feasible points only
   feasible = df[df[('cv', 0)] == 0]
   best_feasible_fx = feasible[('fx', 0)].min()

Plotting Convergence
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt

   # Get objective values in evaluation order
   fx_values = df[('fx', 0)].values

   # Compute running minimum
   running_min = np.minimum.accumulate(fx_values)

   # Plot
   plt.figure(figsize=(10, 6))
   plt.plot(running_min)
   plt.xlabel('Evaluation')
   plt.ylabel('Best f(x) found')
   plt.title('Convergence Plot')
   plt.grid(True)
   plt.show()

Heuristic Performance
~~~~~~~~~~~~~~~~~~~~~

For :class:`~panobbgo.strategies.rewarding.StrategyRewarding`:

.. code-block:: python

   # Heuristic performance scores
   for h in strategy.heuristics:
       print(f"{h.name}: performance = {h.performance:.4f}")

   # Count points generated by each heuristic
   df[('who', 0)].value_counts()

Pareto Front (Constrained Problems)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Access Pareto front from Best analyzer
   best_analyzer = strategy.analyzer('Best')
   pareto_front = best_analyzer.pareto_front

   # Plot (f, CV) trade-off
   plt.figure(figsize=(8, 6))
   plt.scatter(
       [r.fx for r in pareto_front],
       [r.cv for r in pareto_front],
       c='red', marker='o'
   )
   plt.xlabel('f(x)')
   plt.ylabel('Constraint Violation')
   plt.title('Pareto Front')
   plt.grid(True)
   plt.show()

Advanced Topics
---------------

Budget Management
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Set evaluation budget
   strategy = StrategyRewarding(problem, max_evaluations=500)

   # Check progress during optimization
   print(f"Budget used: {len(strategy.results)} / {strategy.config.max_evaluations}")

Parallel Evaluation
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Control batch size
   strategy = StrategyRewarding(
       problem,
       size=10,               # Jobs per client
       jobs_per_client=5      # Batch size
   )

   # With 4 Dask workers, evaluates up to 4*5 = 20 points simultaneously

Multi-start Optimization
~~~~~~~~~~~~~~~~~~~~~~~~

Use the :class:`~panobbgo.analyzers.restart.Restart` analyzer to automatically restart the search
when it gets stuck in a local optimum:

.. code-block:: python

   from panobbgo.analyzers import Restart

   strategy.add_analyzer(Restart,
       patience=100,              # Restart after 100 evals without improvement
       max_restarts=5,            # Allow up to 5 restarts
       restart_strategy="diverse" # Maximize distance from previous centers
   )

Heuristics that define ``on_restart(center, reason)`` will automatically clear their queues
and begin exploring around the new center. Heuristics without this handler continue as before.

The Restart analyzer pairs well with :class:`~panobbgo.analyzers.convergence.Convergence` —
set ``patience`` lower than ``Convergence.window_size`` so restarts happen before convergence
is declared.

Sensitivity Analysis
~~~~~~~~~~~~~~~~~~~~

Use the :class:`~panobbgo.analyzers.sensitivity.Sensitivity` analyzer to identify which
input dimensions matter most:

.. code-block:: python

   from panobbgo.analyzers import Sensitivity

   strategy.add_analyzer(Sensitivity,
       update_interval=50,  # Recompute every 50 new results
       method="spearman"    # Or "partial" for partial correlation
   )

   # After optimization, check importance:
   sens = strategy.analyzer('Sensitivity')
   print(f"Dimension importance: {sens.importance}")

**Sensitivity-Aware Nearby Heuristic**

When both ``Sensitivity`` and :class:`~panobbgo.heuristics.nearby.Nearby` are active,
``Nearby`` automatically becomes *sensitivity-aware*: it scales its per-dimension perturbations
proportionally to the importance scores. Important dimensions receive larger perturbations,
focusing the local search where it matters most.

This is particularly valuable for **high-dimensional** problems where only a subset of
dimensions drive the objective.  For a 10-D problem where only 3 dimensions are active,
the sensitivity-aware Nearby focuses ≈70 % of search effort on those 3 dimensions:

.. code-block:: python

   from panobbgo.analyzers import Sensitivity
   from panobbgo.heuristics import Nearby

   strategy.add(Nearby,
       radius=0.05,
       axes="all",
       new=3,
       sensitivity_scale=1.5,   # Sharpens dim-importance contrast (default 1.0)
   )
   strategy.add_analyzer(Sensitivity(strategy,
       update_interval=20,       # Recompute every 20 evaluations
   ))

   strategy.start()
   print("Dimension importance:", strategy.analyzer("Sensitivity").importance)

The ``sensitivity_scale`` parameter controls how aggressively important dimensions dominate.
Values > 1 amplify the contrast; 0 disables sensitivity-awareness entirely.

Bayesian Optimization with Gaussian Process
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~panobbgo.heuristics.gaussian_process.GaussianProcessHeuristic` implements full
Bayesian optimization — the gold standard for expensive black-box functions with a limited
evaluation budget.

**How it works:**

1. After each batch of results, a Gaussian Process (GP) is fitted to all observed
   :math:`(x, f(x))` pairs using a Matérn-5/2 kernel (via scikit-learn).
2. The fitted GP gives a probabilistic prediction :math:`(\mu(x), \sigma^2(x))` at
   any un-evaluated point.
3. An *acquisition function* selects the next candidate point by trading off
   predicted quality (:math:`\mu`) against prediction uncertainty (:math:`\sigma`).

**Acquisition functions** (``acquisition_func`` parameter):

- ``AcquisitionFunction.EI`` (**default**) — Expected Improvement:
  :math:`\text{EI}(x) = (\mu^* - \mu)\,\Phi(Z) + \sigma\,\phi(Z)` where
  :math:`Z = (\mu^* - \mu)/\sigma`.  Best for exploitation-heavy search.
- ``AcquisitionFunction.UCB`` — Lower Confidence Bound for minimisation:
  :math:`\text{LCB}(x) = \mu(x) - \kappa\,\sigma(x)`.  Higher ``kappa`` → more exploration.
- ``AcquisitionFunction.PI`` — Probability of Improvement:
  :math:`\text{PI}(x) = \Phi((\mu^* - \mu)/\sigma)`.  Conservative; similar to EI.

**Constrained EI (EIC):** When the problem has active constraint violations the GP
automatically trains a *second* surrogate on :math:`CV(x)` and weights the acquisition by
the probability of feasibility: :math:`\text{EIC}(x) = \text{EI}(x) \cdot P(\text{feas.})`.

.. code-block:: python

   from panobbgo.heuristics import GaussianProcessHeuristic
   from panobbgo.heuristics.gaussian_process import AcquisitionFunction

   # Default: EI acquisition, good for smooth/unimodal functions
   strategy.add(GaussianProcessHeuristic)

   # UCB for more exploratory search (useful for multimodal problems)
   strategy.add(GaussianProcessHeuristic,
       acquisition_func=AcquisitionFunction.UCB,
       kappa=2.576,        # 99% confidence level
       n_restarts=10,      # Acquisition optimisation restarts
   )

   # EI with more exploration (higher xi)
   strategy.add(GaussianProcessHeuristic,
       acquisition_func=AcquisitionFunction.EI,
       xi=0.1,             # Default 0.01 → more exploitative; 0.1 → more explorative
   )

**Recommended pairing** with StrategyPhased for a classic two-phase BO workflow:

.. code-block:: python

   from panobbgo.strategies.phased import StrategyPhased
   from panobbgo.strategies.round_robin import StrategyRoundRobin
   from panobbgo.strategies.rewarding import StrategyRewarding
   from panobbgo.heuristics import LatinHypercube, GaussianProcessHeuristic, NelderMead, Random

   strategy = StrategyPhased(problem, phases=[
       {
           "pct": 20,                                       # 20% = initial design
           "strategy": (StrategyRoundRobin, {"size": 5}),
           "heuristics": [(LatinHypercube, {"div": 4}), (Random, {})],
       },
       {
           "strategy": (StrategyRewarding, {}),              # Remaining 80% = BO
           "heuristics": [
               (GaussianProcessHeuristic, {"n_restarts": 10}),
               (NelderMead, {}),
               (Random, {}),
           ],
       },
   ], max_evaluations=500)

   strategy.start()
   print(f"Best: {strategy.best.fx:.6f} at {strategy.best.x}")

Custom Events
~~~~~~~~~~~~~

.. code-block:: python

   # Publish custom event from a heuristic
   class MyHeuristic(Heuristic):
       def on_new_results(self, results):
           if len(results) > 10:
               self.eventbus.publish("my_custom_event", data=results)

   # Subscribe in another module
   class MyAnalyzer(Analyzer):
       def on_my_custom_event(self, data):
           print(f"Received custom event with {len(data)} results")

Logging
~~~~~~~

.. code-block:: python

   # Configure in ~/.panobbgo/config.ini
   [logging]
   level = DEBUG
   focus = heuristics  # Only log from heuristics module

   # Or programmatically
   import logging
   logging.getLogger('panobbgo').setLevel(logging.DEBUG)

Persistent Storage & Resuming
-----------------------------

Panobbgo supports saving optimization results to an SQLite database. This allows you to:

1. **Pause and resume** long-running optimizations.
2. **Recover** from crashes without losing data.
3. **Analyze results** post-hoc using standard SQL tools or other libraries.

Enabling Storage
~~~~~~~~~~~~~~~~

Add the storage configuration to your `~/.panobbgo/config.ini` or pass it via `config.yaml`:

**config.ini**:

.. code-block:: ini

   [storage]
   backend = sqlite
   uri = my_results.db

**config.yaml**:

.. code-block:: yaml

   storage:
     backend: sqlite
     uri: my_results.db

Resuming Optimization
~~~~~~~~~~~~~~~~~~~~~

When you start a strategy with storage enabled, Panobbgo automatically checks the database file.
If it finds existing results, it loads them into memory and resumes the optimization process,
continuing from where it left off.

.. code-block:: python

   # Run 1: Start optimization
   # (Assume config enables sqlite storage)
   strategy = StrategyRewarding(problem, max_evaluations=100)
   strategy.start()
   # Strategy runs for 100 evals and saves them to 'my_results.db'

   # Run 2: Resume and extend
   # Panobbgo loads the 100 previous results
   strategy = StrategyRewarding(problem, max_evaluations=200)
   strategy.start()
   # Strategy runs for another 100 evals (total 200)

Accessing Stored Data
~~~~~~~~~~~~~~~~~~~~~

The SQLite database contains a ``results`` table with the following schema:

- ``id``: Integer Primary Key
- ``x``: JSON array of coordinates
- ``fx``: Objective function value
- ``cv_vec``: JSON array of constraint violations
- ``who``: Name of the heuristic that generated the point
- ``error``: Error estimate
- ``timestamp``: Unix timestamp of generation

You can query this using the ``sqlite3`` command-line tool or any SQLite client:

.. code-block:: bash

   sqlite3 my_results.db "SELECT id, fx, who FROM results ORDER BY fx ASC LIMIT 5;"

Troubleshooting
---------------

Dask Cluster Not Found
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Error:** ``TimeoutError: Cluster not found`` (Only when using ``evaluation: method: dask``)

**Solution:** Start Dask cluster before running:

.. code-block:: bash

   dask scheduler &
   dask worker localhost:8786 --nprocs 4 &

Function Evaluation Fails
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Error:** Exception during evaluation

**Solution:** Ensure your ``eval()`` method handles all inputs in the bounding box:

.. code-block:: python

   def eval(self, x):
       try:
           return my_calculation(x)
       except Exception as e:
           # Return large penalty value
           return 1e10

Out of Memory
~~~~~~~~~~~~~

**Error:** MemoryError with large result database

**Solution:** Reduce ``max_evaluations`` or implement result pruning.

Slow Convergence
~~~~~~~~~~~~~~~~

**Issue:** Not finding good solutions

**Solutions:**

1. Increase budget: ``max_evaluations=5000``
2. Add more diverse heuristics
3. Adjust bounding box (too large?)
4. Check if problem is feasible

Next Steps
----------

- Learn about the mathematical foundation: :doc:`guide_mathematical_foundation`
- Understand the architecture: :doc:`guide_architecture`
- Extend with custom components: :doc:`guide_extending`
- Explore research context: :doc:`guide_research`
