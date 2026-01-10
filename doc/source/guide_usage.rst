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

Dask Cluster Setup
~~~~~~~~~~~~~~~~~~~

**Automatic Local Cluster (Default):**

Panobbgo automatically starts a local Dask cluster with 2 workers for testing and development.
No manual setup required - just run your optimization script!

**Custom Cluster Setup:**

For production use or custom configurations, you can connect to external clusters:

.. code-block:: bash

   # Install Dask distributed
   pip install dask[distributed]

   # Start local cluster with custom workers
   dask scheduler &
   dask worker localhost:8786 --nprocs 4 &

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

   # Run optimization (requires: dask scheduler & workers)
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

Subclass :class:`~panobbgo.lib.lib.Problem`:

.. code-block:: python

   import numpy as np
   from panobbgo.lib.lib import Problem, BoundingBox

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

Noisy Problem
~~~~~~~~~~~~~

Add stochasticity in ``eval()``:

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
   * - Model-based
     - QuadraticWLS
     - Low-dimensional (dim < 20)
   * - Gradient-free
     - LBFGSB
     - When local structure suspected

Recommended Configurations
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Low-dimensional (dim ≤ 5):**

.. code-block:: python

   strategy.add(Center)
   strategy.add(LatinHypercube, div=10)
   strategy.add(Random)
   strategy.add(Nearby)
   strategy.add(NelderMead)
   strategy.add(QuadraticWLS)

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

Troubleshooting
---------------

Dask Cluster Not Found
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Error:** ``TimeoutError: Cluster not found``

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
