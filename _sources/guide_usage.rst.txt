Usage Guide
===========

This guide covers installation, basic usage, and common workflows for Panobbgo.

Installation
------------

Requirements
~~~~~~~~~~~~

- Python ≥ 3.14
- NumPy ≥ 2.0
- SciPy ≥ 1.16
- matplotlib ≥ 3.0
- pandas ≥ 2.0
- statsmodels ≥ 0.14
- Dask ≥ 2026.1 (optional — only for ``evaluation: method: dask``; install via the ``dask`` extra)

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

Verifying the installation
~~~~~~~~~~~~~~~~~~~~~~~~~~

The snippets below are doctests: they run in CI against the installed
package, so if they pass for you the installation is complete.

.. doctest::

   >>> import sys
   >>> print(f"Python version: {sys.version}")
   Python version: ...

   >>> import panobbgo
   >>> print(f"Panobbgo version: {panobbgo.__version__}")
   Panobbgo version: ...

   >>> import numpy as np
   >>> import scipy
   >>> import pandas as pd
   >>> import matplotlib
   >>> print("All dependencies imported successfully")
   All dependencies imported successfully

   >>> print(f"NumPy: {np.__version__}, SciPy: {scipy.__version__}")
   NumPy: ..., SciPy: ...

To run the framework's own test suite (about 2000 tests, roughly a minute
with four workers):

.. code-block:: bash

   uv run pytest -q -n 4

Evaluation Setup
~~~~~~~~~~~~~~~~

**Threaded Evaluation (Default):**

By default, Panobbgo uses local threads for evaluation. This requires no additional setup and is suitable for most local development tasks.

**Dask Cluster (Optional):**

For large-scale distributed optimization, you can use a Dask cluster:

1. **Install the dask extra** (Dask is not a core dependency; the backend
   lives in :mod:`panobbgo.dask_evaluation` and is imported lazily):

   .. code-block:: bash

      pip install "panobbgo[dask]"     # or: uv sync --extra dask

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

   [core]
   deadlock_seconds = 600    # error backstop for a wedged worker (see below)

   [logging]
   level = INFO              # DEBUG, INFO, WARNING, ERROR

Edit this file to customize behavior.

.. note::

   **There is no wall-clock stall guard.**  A run used to end after
   ``max_stall_seconds`` without progress, which made a seeded run a function
   of machine speed — a slow handler could truncate it anywhere between 70 and
   240 of 300 evaluations.  A run now ends when *nothing can produce a point*:
   no evaluation in flight, no event handler queued, and no heuristic with a
   point to give.  Progress is counted in evaluations; a slow arm makes a run
   take longer, never shorter.

   ``deadlock_seconds`` (default ``600``) is the remaining time-based knob and
   is **not** a scheduling parameter: it is an error backstop for a genuine
   bug — a wedged subprocess, a handler that never returns — and tripping it
   is logged at ``ERROR`` level with the state that caused it.  Leave it
   alone unless an arm of yours legitimately takes minutes per evaluation.
   ``max_stall_seconds`` is still accepted so old config files keep loading,
   but it is **inert**.  See
   ``planning/DESIGN_pump_and_stall_2026-09-11.md`` §2.

Basic Usage
-----------

Minimal Example
~~~~~~~~~~~~~~~

.. code-block:: python

   from panobbgo.lib.classic import Rosenbrock
   from panobbgo.strategies.rewarding import StrategyRewarding
   from panobbgo.heuristics import Center, Random, NelderMead

   # Define problem
   problem = Rosenbrock(dims=5)

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
   problem = Rosenbrock(dims=10)

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

Verified Walkthrough
~~~~~~~~~~~~~~~~~~~~

The following steps are executable doctests (run in CI), so they double as
a check that your installation behaves as documented.  They set up
problems and strategies without starting an optimization run.

**Step 1: Define a problem**

Create a simple optimization problem by subclassing :class:`~panobbgo.lib.Problem`:

.. doctest::

   >>> import numpy as np
   >>> from panobbgo.lib import Problem

   >>> class SphereProblem(Problem):
   ...     """Simple sphere function: f(x) = sum(x^2)"""
   ...     def __init__(self, dim=2):
   ...         # Define search bounds: each variable in [-5, 5]
   ...         box = [(-5.0, 5.0)] * dim
   ...         super().__init__(box)
   ...
   ...     def eval(self, x):
   ...         """Evaluate the objective function"""
   ...         return np.sum(x ** 2)

   >>> # Create an instance
   >>> problem = SphereProblem(dim=2)
   >>> print(f"Problem dimension: {problem.dim}")
   Problem dimension: 2

   >>> # Test evaluation at the origin (global optimum)
   >>> from panobbgo.lib import Point
   >>> point = Point([0.0, 0.0], "test")
   >>> result = problem(point)
   >>> print(f"f([0,0]) = {result.fx}")
   f([0,0]) = 0.0

**Step 2: Evaluate points manually**

Test point evaluation and bounds checking:

.. doctest::

   >>> # Generate a random point within bounds
   >>> random_point = problem.random_point()
   >>> print(f"Random point: {random_point}")
   Random point: ...

   >>> # Evaluate the random point
   >>> random_point_obj = Point(random_point, "test")
   >>> result = problem(random_point_obj)
   >>> print(f"f({random_point}) = {result.fx}")
   f(...) = ...

   >>> # Check that point is within bounds
   >>> in_bounds = all(problem.box[0][0] <= coord <= problem.box[0][1] for coord in random_point)
   >>> print(f"Point within bounds: {in_bounds}")
   Point within bounds: True

**Step 3: Create a strategy and add heuristics**

.. doctest::

   >>> from panobbgo.strategies.rewarding import StrategyRewarding

   >>> # Create strategy
   >>> strategy = StrategyRewarding(problem)
   >>> strategy.config.max_eval = 50  # Set evaluation budget
   >>> print(f"Strategy created with max_evaluations: {strategy.config.max_eval}")
   Strategy created with max_evaluations: 50

   >>> from panobbgo.heuristics import Center, Random, Nearby

   >>> # Add initialization heuristic
   >>> strategy.add(Center)
   >>> strategy.add(Random)
   >>> strategy.add(Nearby, radius=0.1)
   >>> print(f"Total heuristics: {len(strategy._hs)}")
   Total heuristics: 3

**Step 4: Verify the strategy setup**

.. doctest::

   >>> # Check strategy configuration
   >>> print(f"Problem: {strategy.problem.__class__.__name__}")
   Problem: SphereProblem
   >>> print(f"Max evaluations: {strategy.config.max_eval}")
   Max evaluations: 50
   >>> print(f"Number of heuristics: {len(strategy._hs)}")
   Number of heuristics: 3

   >>> # The strategy is ready to run optimization with strategy.start()
   >>> print("Strategy setup complete!")
   Strategy setup complete!

The optimization workflow from here is:

1. Call ``strategy.start()`` to begin optimization
2. The strategy coordinates heuristics to generate points
3. Points are evaluated in parallel (using local threads by default)
4. Results are collected and the best solution is tracked
5. Optimization continues until the evaluation budget is exhausted

**Step 5: Use a built-in test function**

.. doctest::

   >>> from panobbgo.lib.classic import Rosenbrock

   >>> # Create Rosenbrock function (banana-shaped valley)
   >>> rosenbrock = Rosenbrock(dims=2)
   >>> print(f"Rosenbrock problem dimension: {rosenbrock.dim}")
   Rosenbrock problem dimension: 2

   >>> # Evaluate at global optimum
   >>> optimum = Point([1.0, 1.0], "test")
   >>> result = rosenbrock(optimum)
   >>> print(f"Rosenbrock optimum f([1,1]) = {result.fx}")
   Rosenbrock optimum f([1,1]) = 0.0

   >>> # Evaluate at a different point
   >>> test_point = Point([0.0, 0.0], "test")
   >>> result = rosenbrock(test_point)
   >>> print(f"Rosenbrock f([0,0]) = {result.fx:.3f}")
   Rosenbrock f([0,0]) = 1.000

   >>> # Create strategy for Rosenbrock
   >>> strategy2 = StrategyRewarding(rosenbrock)
   >>> strategy2.config.max_eval = 100
   >>> strategy2.add(Center)
   >>> strategy2.add(Random)
   >>> strategy2.add(Nearby, radius=0.1)

   >>> # Strategy is ready for optimization
   >>> print(f"Rosenbrock strategy configured with {len(strategy2._hs)} heuristics")
   Rosenbrock strategy configured with 3 heuristics
   >>> print(f"Ready to optimize with budget of {strategy2.config.max_eval} evaluations")
   Ready to optimize with budget of 100 evaluations

**Step 6: Define a constrained problem**

.. doctest::

   >>> class ConstrainedSphere(Problem):
   ...     """Sphere with constraint: sum(x) <= 1"""
   ...     def __init__(self, dim=2):
   ...         box = [(-2.0, 2.0)] * dim
   ...         super().__init__(box)
   ...
   ...     def eval(self, x):
   ...         return np.sum(x ** 2)
   ...
   ...     def eval_constraints(self, x):
   ...         # Constraint: sum(x) - 1 <= 0 (feasible when sum(x) <= 1)
   ...         return np.array([np.sum(x) - 1.0])

   >>> constrained_problem = ConstrainedSphere(dim=2)
   >>> print("Constrained problem created")
   Constrained problem created

   >>> # Test feasible point
   >>> feasible_point = Point([0.3, 0.3], "test")
   >>> result_feasible = constrained_problem(feasible_point)
   >>> print(f"Feasible point: x = {feasible_point.x}, f(x) = {result_feasible.fx:.3f}")
   Feasible point: x = [0.3 0.3], f(x) = 0.180
   >>> print(f"Constraint violation: {result_feasible.cv_vec}")
   Constraint violation: [-0.4]

   >>> # Test infeasible point
   >>> infeasible_point = Point([1.0, 1.0], "test")
   >>> result_infeasible = constrained_problem(infeasible_point)
   >>> print(f"Infeasible point: x = {infeasible_point.x}, f(x) = {result_infeasible.fx:.3f}")
   Infeasible point: x = [1. 1.], f(x) = 2.000
   >>> print(f"Constraint violation: {result_infeasible.cv_vec}")
   Constraint violation: [1.]

**Step 7: Full setup**

.. doctest::

   >>> # Import everything needed
   >>> from panobbgo.lib.classic import Rosenbrock
   >>> from panobbgo.strategies.rewarding import StrategyRewarding
   >>> from panobbgo.heuristics import Center, Random, Nearby, NelderMead

   >>> # Define problem
   >>> problem = Rosenbrock(dims=3)
   >>> print(f"Optimizing {problem.dim}D Rosenbrock function")
   Optimizing 3D Rosenbrock function

   >>> # Create strategy
   >>> strategy = StrategyRewarding(problem)
   >>> strategy.config.max_eval = 200

   >>> # Add diverse heuristics
   >>> strategy.add(Center)
   >>> strategy.add(Random)
   >>> strategy.add(Nearby, radius=0.1)
   >>> strategy.add(NelderMead)

   >>> # Verify setup
   >>> print(f"Problem: {problem.__class__.__name__} ({problem.dim}D)")
   Problem: Rosenbrock (3D)
   >>> print(f"Strategy: {strategy.__class__.__name__}")
   Strategy: StrategyRewarding
   >>> print(f"Budget: {strategy.config.max_eval} evaluations")
   Budget: 200 evaluations
   >>> print(f"Heuristics: {len(strategy._hs)}")
   Heuristics: 4
   >>> print("Ready to run with: strategy.start()")
   Ready to run with: strategy.start()

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
   problem = NormalizedProblem(Rosenbrock(dims=5))

   # Add noise for robustness testing (seed for reproducibility)
   problem = NoisyProblem(Rosenbrock(dims=5), noise_std=0.1, seed=42)

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
     - LatinHypercube, Sobol
     - High-dimensional problems (dim > 5); prefer ``Sobol`` for Bayesian
       optimization initial designs (lower discrepancy than LHS)
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
     - CMAES (IPOP/BIPOP), DifferentialEvolution, LSHADE, JSO, PSO
     - CMAES: smooth problems, ridges, and ill-conditioned landscapes (Rosenbrock); BIPOP-CMA-ES: highly multimodal (BBOB-2009 winner); DE: multimodal (Rastrigin, Schwefel); LSHADE: literature-best adaptive DE — adapts ``F`` / ``CR`` via success-history memories and shrinks the population linearly with the budget (CEC-2014 winner, Tanabe & Fukunaga 2014); JSO: jSO refinement of L-SHADE with weighted ``current-to-pbest-w/1`` mutation, linear ``p_best`` schedule, Cauchy-F clamping, and a frozen anchor memory bin (CEC-2017 winner, Brest-Maučec-Bošković 2017); PSO: momentum + social attraction, complementary to CMA-ES / DE — strong on narrow-valley problems where vector inertia helps
   * - Gradient-free local
     - LBFGSB, LocalPenaltySearch
     - When local structure suspected
   * - Constraint Handling
     - FeasibleSearch, ConstraintGradient, ConstraintRepair
     - When constraints are present

Recommended Configurations
~~~~~~~~~~~~~~~~~~~~~~~~~~

Start here: one strong optimizer
................................

.. important::

   **Every evaluation you give one heuristic is an evaluation another one
   does not get.**  Population methods (CMA-ES, the L-SHADE/DE family,
   PSO) need a whole budget to run their population dynamics; splitting
   the budget across a portfolio starves them.  This is not a small
   effect — on the MA-BBOB battery a six-arm portfolio scored *below
   every one of its own arms run alone* (see the table below).

   Begin with a single population method and add arms only when a paired
   A/B on your own problem shows they earn their evaluations.

.. code-block:: python

   from panobbgo.heuristics import CMAES
   from panobbgo.strategies import StrategyRoundRobin

   strategy = StrategyRoundRobin(problem, max_evaluations=1000, seed=42)
   strategy.add(CMAES)          # self-adapting covariance, IPOP restarts
   strategy.start()

.. note::

   *Which* single arm is under re-evaluation.  The table below was measured
   before the population-sizing fix of :ref:`population-sizing`, i.e. against
   DE arms that were roughly five times over-populated.  Re-run on the
   12-seed roster with the sizes the library now ships, jSO, NLSHADE_LBC,
   L-SHADE and CMA-ES sit within 0.014 mean AOCC of each other — inside the
   null floor (see :doc:`guide_benchmarking`) — and win *different* instances,
   sorted by dimension (``planning/DISCOVERY_2026-09-09.md`` §28).  "CMA-ES
   alone" is still a safe starting point, but it is no longer a measured
   winner; benchmark the arms on your own problem before committing.

   The same holds for the portfolio.  ``Blocks_warm_CMAES_JSO`` — CMA-ES +
   jSO, both warm-started from the shared archive, rotating every block —
   is now the second shipped harness spec, replacing ``Rewarding_Restart``
   (0.35, no longer a useful control).  It measures **level with**
   ``RoundRobin_CMAES``, not above it (§27, §30).  The plan of record is a
   dimension-gated spec — portfolio for ``d ≥ 5``, single arm below — since
   the portfolio's whole lean sits at ``d=5``; see ``planning/GOAL.md`` §2c.

Measured on the MA-BBOB battery (mean AOCC, dims 2 and 5, budget 500·d,
5 instances, seeds 42 / 7 / 1234, ``sync_evaluation=True``):

.. list-table::
   :header-rows: 1
   :widths: 45 18 18 18

   * - Setup
     - mean AOCC
     - d=2
     - d=5
   * - ``CMAES`` alone
     - **0.580**
     - 0.623
     - 0.538
   * - ``NLSHADE_LBC`` alone
     - 0.537
     - 0.633
     - 0.441
   * - ``JSO`` alone
     - 0.459
     - 0.553
     - 0.364
   * - ``PSO`` alone
     - 0.424
     - 0.481
     - 0.367
   * - ``LSHADE`` alone
     - 0.417
     - 0.501
     - 0.333
   * - scipy ``differential_evolution`` (reference)
     - 0.416
     - 0.500
     - 0.331
   * - six-arm portfolio (Random + Center + Nearby + NelderMead + JSO + NLSHADE_LBC)
     - 0.352
     - 0.405
     - 0.300
   * - ``Random`` alone
     - 0.319
     - 0.365
     - 0.272

The ordering holds at every budget from 25 to 500 evaluations per
dimension; the margin grows with budget.  Higher AOCC is better; see
:doc:`guide_benchmarking` for the metric.

The same ordering holds at higher dimension (MA-BBOB, budget 100·d,
instances 0–2, seeds 42 / 7 / 1234):

.. list-table::
   :header-rows: 1
   :widths: 12 25 22 22 22

   * - dim
     - six-arm portfolio
     - ``CMAES`` alone
     - ``NLSHADE_LBC`` alone
     - scipy DE
   * - 10
     - 0.181
     - **0.301**
     - 0.192
     - 0.162
   * - 20
     - 0.102
     - **0.198**
     - 0.162
     - 0.080

.. note::

   These numbers are for continuous, box-constrained problems from the
   MA-BBOB generator at dims 2, 5, 10 and 20.  For *constrained*
   problems, noisy objectives, or a different problem class, re-measure
   with the harness before trusting any ranking — including this one.

**Bayesian optimization (≤ 500 evals, smooth/expensive function):**

The configurations below have *not* been measured against the
single-optimizer default above.  Treat them as starting points to
benchmark, not as recommendations.

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

**Bayesian optimization with Sobol' initial design:**

Sobol' is a low-discrepancy quasi-random sequence; the first ``2^k`` samples cover
the unit hypercube more uniformly than i.i.d. uniform or Latin Hypercube samples
of the same size.  Modern BO libraries (BoTorch, TuRBO, scikit-optimize) use Sobol'
as their default initial-design generator.  In Panobbgo this is the
``BayesOpt_Sobol`` strategy in the standard harness mode.

.. code-block:: python

   from panobbgo.heuristics import GaussianProcessHeuristic, Sobol, NelderMead, Nearby

   strategy = StrategyRewarding(problem, max_evaluations=200)
   strategy.add(Sobol, n=16, scramble=True)     # 16 Sobol' points; Owen scrambling
   strategy.add(GaussianProcessHeuristic,        # GP + EI acquisition
       n_restarts=5,
       xi=0.01)
   strategy.add(Nearby, radius=0.05)             # Local refinement around best
   strategy.add(NelderMead)                      # Simplex polish

**Tip:** Pick ``n`` as a power of two (``8, 16, 32, 64``) — Sobol's balance
properties are strongest at those counts.  Owen scrambling preserves the
low-discrepancy guarantees within a draw while randomizing across reps so
the harness can compute meaningful repetition statistics.

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

**Smooth/ill-conditioned problems with CMA-ES (≥ 100 evals recommended):**

CMA-ES is the gold standard for derivative-free optimization of continuous functions.
It adapts both step size and covariance to the local geometry, making it exceptionally
effective on problems with elongated valleys (Rosenbrock), ill-conditioned quadratics,
or rotated search spaces.

.. code-block:: python

   from panobbgo.heuristics import CMAES, LatinHypercube, NelderMead, Nearby

   strategy = StrategyRewarding(problem, max_evaluations=200)
   strategy.add(LatinHypercube, div=4)  # Spread initial samples evenly
   strategy.add(CMAES, sigma0=0.3)      # Self-adapting covariance-matrix search
   strategy.add(Nearby, radius=0.05)    # Fine local refinement
   strategy.add(NelderMead)             # Gradient-free simplex local optimizer

**Multimodal problems with IPOP-CMA-ES (competition-winning restart strategy):**

IPOP-CMA-ES (Increasing Population CMA-ES; Auger & Hansen 2005) is the approach used by
competition-winning solvers on the BBOB/COCO benchmark suite.  When stagnation is detected
the population doubles (λ → 2λ) and the search restarts from a diverse new center, enabling
systematic escape from local optima while the full result history is retained.

.. warning::

   Do **not** pair :class:`~panobbgo.heuristics.cma_es.CMAES` with the
   external :class:`~panobbgo.analyzers.restart.Restart` analyzer.  The
   heuristic implements IPOP and BIPOP restarts internally
   (``restart_mode``), keeps its adapted covariance across them, and since
   2026-09 decides *when* to restart on its own (see
   :ref:`cma-es-restarts`).  The analyzer on top *halved* its score on the
   MA-BBOB battery (0.663 → 0.301 mean AOCC, d2+d5, seed 42): its restart
   event discards what the search has learned.  ``CMAES(restart_mode="ipop")``
   on its own is the configuration to use.

.. code-block:: python

   from panobbgo.heuristics import CMAES, LatinHypercube, NelderMead

   strategy = StrategyRewarding(problem, max_evaluations=500)
   strategy.add(LatinHypercube, div=4)
   strategy.add(CMAES, sigma0=0.3, ipop_factor=2.0)
   strategy.add(NelderMead)
   strategy.start()

The :class:`~panobbgo.analyzers.restart.Restart` analyzer remains available
for heuristics that have no restart logic of their own.  Its key parameters:

- ``patience`` (int | None): evaluations without improvement before restart.
  ``None`` uses ``5 * problem.dim`` (recommended).
- ``restart_strategy``: one of:

  * ``"random"`` — uniform draw inside the box (default).
  * ``"diverse"`` — picks the new center to maximize the minimum
    distance to all previous restart centers — better coverage than
    random after a few restarts have fired.
  * ``"sphere"`` — Gaussian draw centered at the box centre with
    ``std = ranges / 6`` (clipped to the box).  Biases the restart
    cloud toward the centroid; useful when the optimum is expected to
    lie in the box interior rather than near its boundary.
- ``max_restarts``: cap on total restarts to prevent budget exhaustion.

**Highly multimodal problems with BIPOP-CMA-ES (BBOB-2009 winner):**

For problems with many local optima or unknown structure, switch to
``restart_mode="bipop"``.  BIPOP-CMA-ES (Hansen 2009) alternates between two restart
regimes — a *large* regime that grows the population geometrically (like IPOP) and a
*small* regime that runs a small population with a random small step size.  After every
restart the regime that has consumed *fewer* cumulative evaluations is selected next,
balancing exploitation and exploration.  BIPOP won the BBOB-2009 competition and remains
the state of the art for limited-budget multimodal black-box optimization.

.. code-block:: python

   strategy = StrategyRewarding(problem, max_evaluations=500)
   strategy.add(LatinHypercube, div=4)
   strategy.add(CMAES, sigma0=0.3, restart_mode="bipop")
   strategy.add(NelderMead)
   strategy.start()

Inspect the BIPOP regime distribution after the run::

   cma = strategy._heuristics["CMAES"]
   print(cma.bipop_regime, cma.bipop_evals_large, cma.bipop_evals_small)

When to choose IPOP vs. BIPOP:

* **IPOP** — moderately multimodal landscapes; you trust that increasing the population
  will eventually find the global structure.
* **BIPOP** — highly multimodal landscapes where a small, randomly-scaled distribution
  occasionally finds basins that the large regime would miss.  Pay-off improves with a
  larger evaluation budget (≥ 200 evals).

**Multimodal problems (Rastrigin, Schwefel) — lighter portfolio:**

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

Tuning the Population Methods
-----------------------------

The three subsections below cover the settings that were measured to matter
most on the MA-BBOB battery.  Every parameter mentioned here is documented in
full on its own class; this is the *why* and the numbers, not a second copy of
the API reference.

.. _population-sizing:

Population Sizing (``NP_init="auto"``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The L-SHADE family — :class:`~panobbgo.heuristics.lshade.LSHADE`,
:class:`~panobbgo.heuristics.jso.JSO`,
:class:`~panobbgo.heuristics.nl_shade_lbc.NLSHADE_LBC` and the other SHADE
derivatives — accepts ``NP_init="auto"``, which sizes the initial population
from the run's evaluation budget and the problem dimension:

.. code-block:: text

   NP = clip( round( 3·dim · (budget / (500·dim))**0.25 ), max(NP_min, 6), 400 )

This is the constructor default for the whole L-SHADE family; the
literature value ``30`` is used only as a fallback when the budget is not
known at construction.  The coefficient is a class attribute
(``AUTO_DIM_COEF``): the base rule uses ``3``, and ``NLSHADE_LBC`` uses
``4`` — its linear bias control wants a larger rank pool, measured at
+0.052 AOCC over the ``3·dim`` rule on the 12-seed roster (10/12 seeds).

.. code-block:: python

   from panobbgo.heuristics import JSO
   from panobbgo.strategies import StrategyRoundRobin

   strategy = StrategyRoundRobin(problem, max_evaluations=2500, seed=42)
   strategy.add(JSO, NP_init="auto")     # d=5, B=2500  →  NP_init = 15
   strategy.start()

**Why.** Fixed-``NP_init`` grid sweeps of the three L-SHADE-lineage arms, run
solo on the standard battery, put the AOCC optimum at 6–8 for ``d=2``, 10–20
for ``d=5`` and 20–30 for ``d=10``: the optimum tracks the *dimension*, which
fixes the coefficient at ``≈ 3·dim``.  Quadrupling the budget moves it up by
only about 1.5×, hence the fourth-root budget term anchored at the reference
``500`` evaluations per dimension.  The floor of 6 keeps
``current-to-pbest/1`` working headroom — at ``NP_init=4`` the mutation
collapses (−0.25 AOCC for ``NLSHADE_LBC``).  The rule this replaced,
``min(18·dim, budget/12)``, took the CEC-2014 upper bound literally; at a few
hundred evaluations per dimension its cap never binds, so it shipped
populations roughly **five times too large** — most of the budget went into
the initial random fill and the success-history adaptation never got enough
generations to pay off.

Accepted on the 12-seed decision roster, each arm solo and paired on one RNG
stream (``planning/DISCOVERY_2026-09-09.md`` §17, §20):

.. list-table::
   :header-rows: 1
   :widths: 30 20 30 20

   * - Arm
     - Δ mean AOCC
     - CI 95 %
     - Seeds positive
   * - ``LSHADE``
     - **+0.230**
     - [+0.215, +0.244]
     - 12 / 12
   * - ``JSO``
     - **+0.204**
     - [+0.175, +0.233]
     - 12 / 12
   * - ``NLSHADE_LBC``
     - +0.064
     - [+0.028, +0.099]
     - 10 / 12

.. important::

   **The budget must be known when the heuristic is constructed.**
   ``"auto"`` is resolved once, in ``__init__``, from
   ``strategy.config.max_eval``.  Pass the budget to the strategy constructor
   (``max_evaluations=``), or let a ``StrategySpec`` do it — its
   ``create_strategy(max_eval=...)`` sets the budget *before* any heuristic is
   built, and the harnesses now pass it.  Assigning
   ``strategy.config.max_eval`` **after** ``strategy.add(...)`` is too late:
   the heuristic has already fallen back to the fixed default of ``30``.

.. _cma-es-restarts:

CMA-ES Restarts
~~~~~~~~~~~~~~~

Solo :class:`~panobbgo.heuristics.cma_es.CMAES` used to have no termination
criterion at all: one CMA-ES run for the whole budget, and once σ had
collapsed it kept resampling the same point.  Diagnosed at ``d=5`` with a
2500-evaluation budget, **52 % of the budget was spent after the last
improvement of any size**, and on two of five instances σ *diverged* against
its clamp and sampled the box boundary for 92 % of the run.

Two mechanisms now fix this, both on by default (``self_restart=True``):

* **Hansen's reference termination set** — ``tolx``, ``tolfun``,
  ``tolfunhist``, ``stagnation``, ``conditioncov``, ``noeffectaxis``,
  ``noeffectcoord``.  When one fires, the heuristic restarts through the
  IPOP/BIPOP path, so ``ipop_factor`` finally has an effect in solo runs.
  These tolerances are written for runs of many thousands of generations and
  rarely fire inside a 500·dim budget; on their own they were worth
  **+0.0001**.
* **σ-divergence detection** (``sigma_divergence=True``, the panobbgo
  analogue of pycma's ``tolupsigma``) — restart when the sampling spread
  ``σ·sqrt(diag C)`` has sat at or above ``sigma_max_frac`` (default ``0.3``)
  of the box range in *every* coordinate for ``sigma_divergence_gens``
  generations.  This is the addition that paid: **+0.026 mean AOCC
  [+0.008, +0.043] on the 12-seed roster, 11 of 12 seeds positive**, at 0.68
  restarts per run.  The gain is heavy-tailed — a handful of cells gain
  +0.18 … +0.44 and most are unchanged — because it only fires on the runs
  that would otherwise diverge (``planning/DISCOVERY_2026-09-09.md`` §23).

``restart_from`` chooses where a *self*-restart re-centres:  ``"random"``
(default, the reference IPOP behaviour), ``"best"`` or ``"center"``.  A
σ-divergence restart always re-centres on the best point seen regardless — a
diverged run has no basin worth keeping.  ``stagnation_frac`` adds a
*budget*-relative stagnation window; it is off by default because it measured
harmful at the fractions where it fires often.

.. code-block:: python

   # The default — nothing to configure.
   strategy.add(CMAES, sigma0=0.3, restart_mode="ipop")

   # Restore the pre-2026-09 behaviour (one run for the whole budget).
   strategy.add(CMAES, self_restart=False)

.. note::

   Because CMA-ES now terminates and restarts itself, the external
   :class:`~panobbgo.analyzers.restart.Restart` analyzer is **not needed**
   with it — and was measured to hurt badly (see the warning under
   *Multimodal problems with IPOP-CMA-ES* above).

.. _shared-archive-warm-start:

Shared Archive and Warm Start
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A portfolio of independent solvers is only worth its switching costs if the
arms *share* the evaluations they paid for.  The
:class:`~panobbgo.analyzers.archive.Archive` analyzer is that sharing layer:
a bounded top-K of the whole result stream, ranked by the constraint
handler's penalty value and **not** filtered by ``who``, so an arm that asks
for a seed gets the best points in the run whoever produced them.

It is **opt-in** — :meth:`~panobbgo.core.StrategyBase.initialize` does not add
it, so a run without warm-started arms keeps exactly the module construction
order, and therefore the RNG streams, it had before:

.. code-block:: python

   from panobbgo.analyzers import Archive
   from panobbgo.heuristics import LSHADE

   strategy.add_analyzer(Archive, k=256)          # or StrategySpec(analyzers=[(Archive, {})])
   strategy.add(LSHADE, NP_init="auto", warm_start="archive")

Three selectors are shared by every arm (see
:meth:`~panobbgo.core.Heuristic.archive_seed`):

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - ``warm_start``
     - Seeds with
   * - ``"archive"``
     - the ``k`` best results in the run
   * - ``"archive_diverse"``
     - ``k`` well-separated good results
   * - ``"archive_leaf"``
     - the best point of each of the ``k`` best ``Splitter`` leaves — ``k``
       different basins

What each arm does with them:

- **The L-SHADE family** places the seeds into the initial population
  *directly, as evaluated results*, and fills any shortfall from the cold
  random path — so warm starting costs **zero evaluations**.
- **PSO** takes positions and personal bests from the top-``NP`` results, also
  at zero cost, and sets velocities to ``0.5·(x_π(i) − x_i)`` over a random
  derangement, so the swarm starts *moving between* known good points instead
  of from a standstill.
- **CMA-ES** fits its initial distribution to the seed cloud.  It additionally
  accepts ``warm_start="archive_cov"``, which also seeds the covariance
  matrix ``C``.  Here the warm start saves no evaluations — CMA-ES has no
  population to fill — the whole gain is starting in the right basin at the
  right scale.

In all cases an empty archive (``t = 0``) means the arm silently cold-starts.
For a custom heuristic, call
:meth:`~panobbgo.core.Heuristic.archive_seed` and treat ``[]`` as "use the
cold path"; override :meth:`~panobbgo.core.Heuristic.warm_start_now` if a
scheduler should be able to re-seed you mid-run.

The recommended warm configuration, if you use one, is **both** arms on plain
``warm_start="archive"`` with both guards off:

.. code-block:: python

   strategy = StrategyBlockBandit(problem, max_evaluations=2500, seed=42,
                                  policy="uniform", block_evals=25,
                                  warm_start_on_resume=True,
                                  warm_start_only_if_foreign=False,
                                  warm_start_only_if_better=False)
   strategy.add_analyzer(Archive)
   strategy.add(CMAES, warm_start="archive")
   strategy.add(JSO, NP_init="auto", warm_start="archive")

Three findings behind those settings
(``planning/DISCOVERY_2026-09-09.md`` §25, §31):

- **Both arms warm, not one.**  One-directional sharing scored 0.619 against
  0.645 for the same pair with both arms warm — CMA-ES's warm start is worth
  as much as the DE arm's.
- **``"archive_cov"`` hurts** (−0.019, at both dimensions).  Seed the mean and
  σ; leave ``C = I``.
- **``warm_start_only_if_better`` hurts** (−0.073 on the 3-seed screen,
  −0.007 on the roster).  It cuts warm starts lopsidedly: CMA-ES usually holds
  the incumbent, so the guard starves the arm that most needs the relay.
  ``warm_start_only_if_foreign`` is the shipped guard, but it too was measured
  off in the specs that lead.

.. warning::

   **Experimental, off by default.**  Sharing is the one portfolio mechanism
   on this codebase that measures: it removes the **−0.08** penalty a *cold*
   two-arm portfolio pays against its own best arm (the one CI clear of zero
   in the whole screen).  It does not buy a lead.  On the 12-seed roster the
   best warm portfolio — ``Blocks_uniform_cj_warm2``, CMA-ES + jSO, both warm,
   rotating every block — reaches **0.685 against CMA-ES alone at 0.666:
   +0.019, 8 of 12 seeds, CI including zero.**  Level with the best single
   arm, not above it.  The whole lean is at ``d=5`` (+0.03); at ``d=2`` a
   single arm converges before a relay can matter
   (``planning/DISCOVERY_2026-09-09.md`` §27, §30).  Treat ``warm_start`` as
   a per-arm experiment to be settled by your own paired A/B, not as a setting
   to switch on.

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

StrategyBlockBandit (experimental)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~panobbgo.strategies.blocks.StrategyBlockBandit` changes what a bandit
*pull* is.  The other strategies treat one point as one pull and pay out only
on a new best, so a population heuristic emitting λ points per generation has
its estimated value bounded by ``1/λ`` by construction — a measurement
problem, not a tuning problem.  Here a pull is a **block**: roughly
``max_eval / n_blocks`` evaluations (default ``n_blocks=50``) handed to a
single arm.  A block closes only when it has spent its evaluations *and* the
owner's queue has run empty, so a generation is never cut in half.

The reward is the AOCC area the arm bought with the block, per evaluation, in
decades of log-precision against a run-local anchor.  It is *anytime* (the
mean over the block, not its endpoint) and *scale-free* (invariant under
``f → c·f + k``), so arms on different objective scales are comparable
without any tuning.

Two policies:

- ``policy="uniform"`` — round-robin over blocks.  **Use this one.**
- ``policy="ducb"`` (the constructor default) — discounted UCB, with a
  one-block prologue per arm, an exploit-only tail, and hysteresis so a
  challenger must beat the incumbent by a factor before the schedule
  switches.

``warm_start_on_resume=True`` (off by default) re-seeds an arm from the shared
archive whenever the scheduler hands it a block back after a gap — see
:ref:`shared-archive-warm-start`.  The trigger is the gap, not an empty queue:
a paused arm's queued generation is stale by construction, so it is cleared
first.

.. code-block:: python

   from panobbgo.strategies import StrategyBlockBandit
   from panobbgo.analyzers import Archive
   from panobbgo.heuristics import CMAES, JSO

   strategy = StrategyBlockBandit(problem, max_evaluations=2500, seed=42,
                                  policy="uniform", block_evals=25,
                                  warm_start_on_resume=True,
                                  warm_start_only_if_foreign=False)
   strategy.add_analyzer(Archive)
   strategy.add(CMAES, warm_start="archive")
   strategy.add(JSO, NP_init="auto", warm_start="archive")
   strategy.start()

What was measured to matter, and what was not
.............................................

Screens #5–#7 took the policy apart on the standard battery
(``planning/DISCOVERY_2026-09-09.md`` §26, §29, §31):

- **The selection policy contributes nothing measurable.**  D-UCB at any
  setting tried — ``ucb_c`` 1/2/4, ``gamma`` 0.5–0.9, ``hysteresis``,
  ``tail_frac`` swept 0 → 0.6 — lands within 0.003 of plain rotation, and at
  ``tail_frac=0`` it *is* rotation (identical in 23 of 30 cells; the mean gap
  is one cell).  Once the arms share their evaluations, a switch stops being a
  cost and becomes a relay, so there is nothing left for a bandit to
  minimise.  ``policy="uniform"`` is therefore the setting to use — it is the
  same behaviour with none of the knobs.
- **Block length is absolute, not a fraction of the budget.**  The useful
  range is **~25–50 evaluations**, flat-ish between them with per-cell
  collapses either side and a hard collapse at 12.  Set it with
  ``block_evals=`` — ``n_blocks=50`` is a budget *fraction*, so on the
  ``500·dim`` battery it means 20 evaluations at ``d=2`` and 50 at ``d=5``,
  two different experiments under one name.  ``block_evals="auto"`` sizes a
  block as four reference generations, which is the mechanism behind the
  range: every block boundary discards the arm's in-flight generation.
- **More than two arms still dilute.**  2 → 3 → 4 arms: 0.685 → 0.648 →
  0.604.  Only at ``d=5`` does a third arm pay (+0.037); at ``d=2`` it costs
  −0.084.  A third arm, if ever, is dimension-gated.

.. warning::

   **Experimental**, and read screen results carefully.  A cold two-arm block
   bandit loses **−0.08** to its best arm, and the interleaving control
   (``StrategyRewarding`` on the same two arms) loses the same amount — so
   that is the portfolio, not the scheduling shape.  The scheduler itself is
   sound: every run spends its full budget and no generation is cut.  With
   both arms warm the portfolio reaches parity but not a lead (§27, §30 and
   the warning under :ref:`shared-archive-warm-start`).

   Also: **the best spec of a three-seed screen is a candidate, not a
   result.**  ``soft_be25`` came out of one screen at +0.050 with a CI
   excluding zero and scored **−0.006** on the 12-seed roster — a CI computed
   on a selected maximum is not the CI of a pre-registered spec
   (``planning/DISCOVERY_2026-09-09.md`` §30).

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

   problem = Rosenbrock(dims=5)

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

Reproducible Runs
~~~~~~~~~~~~~~~~~

Pass ``seed`` to the strategy and enable the synchronous evaluation mode to
make a run a pure function of that seed:

.. code-block:: python

   strategy = StrategyRewarding(problem, seed=1234)
   strategy.config.sync_evaluation = True   # deterministic result batches
   strategy.add(Random)
   strategy.add(JSO, NP_init="auto")
   strategy.start()

Two such runs evaluate exactly the same points in the same order.  Every
module derives its own :class:`numpy.random.Generator` from the master seed
(``self.rng``), the event bus delivers events serially, the main loop
waits for all handlers before drawing new points, and each batch is
evaluated sequentially in submission order (a thread pool would return
results in completion order).  Without ``seed`` the
strategy draws one from numpy's global state, so ``np.random.seed(s)``
before construction also pins the run; ``strategy.seed`` reports which seed
was used.

The guarantee now covers the subprocess solver bridges (``LBFGSB``,
``COBYQA``) too.  They used to relay their pipe on a private "pump" thread
and emit on their own schedule, which put them outside it; they now produce
on the strategy's own thread, when asked, so the point they hand out is a
pure function of the values they were given and their worker's seed
(``planning/DESIGN_pump_and_stall_2026-09-11.md`` §1.5).

Budget Management
~~~~~~~~~~~~~~~~~

A run spends its full ``max_eval`` budget unless its heuristics run out of
points to give.  The :class:`~panobbgo.analyzers.Convergence` analyzer
publishes a ``converged`` event when the best value plateaus, but that event
only ends the run if you opt in with
``strategy.config.stop_on_convergence = True`` (the plateau test fires
routinely on multimodal problems, so stopping on it is not a safe default for
global optimization).

The other way a run ends early is that **nothing can produce a point** — see
the note under *Configuration* above.  That is a fact about the portfolio,
not a timeout, and it is sometimes the correct outcome: a solo
:class:`~panobbgo.heuristics.cobyqa.COBYQA` converges its trust region after
roughly 25 evaluations and has nothing further to propose, so it ends there
whatever budget you gave it.  Pair such an arm with a generator (``Random``,
a population method) if you want the rest of the budget spent.

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

.. warning::

   Do not use it with :class:`~panobbgo.heuristics.cma_es.CMAES`, which
   restarts itself (:ref:`cma-es-restarts`); the analyzer's restart event
   throws away its adapted covariance and halved its score in measurement.

Splitter Resolution
~~~~~~~~~~~~~~~~~~~

:class:`~panobbgo.analyzers.splitter.Splitter` partitions the search space
into a kd-tree of boxes and cuts a leaf once it holds enough points.  How many
points is "enough" used to be ``max(20, max_eval / dim²)`` — independent of
the budget once the dimension is fixed, so a longer run produced the same
coarse tree.  The threshold is now stated on the *leaf population* and
inverted, so the resolution scales with the budget: at ``d=2`` the tree grows
to 38 / 105 / 204 / 351 leaves across the budget range where the old one gave
6 / 33 / 142 / 602, and the root splits at evaluation 37 instead of 250
(``planning/DISCOVERY_2026-09-09.md`` §35).

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Parameter
     - Meaning
   * - ``leaf_size``
     - Target mean results per leaf (default 25); the tree aims at
       ``max_eval / leaf_size`` leaves.
   * - ``min_leaf_size``
     - Floor on the split threshold, so a leaf is never cut before it can say
       anything.  Default ``2·dim + 2``.
   * - ``max_leaves``
     - Cap on the leaf count (default 512).  It raises the threshold rather
       than refusing splits, so the partition stays a proper kd-tree.
   * - ``split_rule``
     - Which dimension to cut: ``"widest"`` (default) or ``"value"``, which
       scores dimensions by how strongly the objective separates across the
       prospective cut.
   * - ``cut_rule``
     - Where along it: ``"mean"`` (default) or ``"median"``, which cuts the
       median *gap* between adjacent distinct coordinates.
   * - ``legacy``
     - ``True`` restores the pre-2026-09-10 tree exactly, ignoring every other
       knob.  Kept so the two trees can be measured paired.

Accepted on the 12-seed roster against ``legacy=True`` (§37): the consumers
that read the tree gain — ``Random`` **+0.033** [+0.009, +0.057],
``RegionUCB`` **+0.050** [+0.023, +0.078] (12/12 seeds), and an
``archive_leaf`` warm start **+0.053** [+0.027, +0.079].  The gain is
concentrated at ``d=2``, where the old tree had five leaves.  **The reference
portfolio is unaffected** — 0.0000 in every cell — because none of its arms
queries the tree, so this change is free if you do not use one that does.

``split_rule="value"`` and ``cut_rule="median"`` both ship **opt-in**: each
measured inside the null floor overall, and the median cut carries a
structural regression (it hits the depth cap with a 423-point leaf on a spec
where the mean stays at depth 52).  ``RegionUCB``, the one consumer that reads
every leaf, is the case worth revisiting on a roster — it gained +0.024 and
+0.033 from them respectively (§39).

.. code-block:: python

   from panobbgo.analyzers import Splitter
   from panobbgo.heuristics import RegionUCB

   strategy.add_analyzer(Splitter, leaf_size=25)   # defaults; budget-scaled
   strategy.add(RegionUCB)                         # UCB1 over the leaves

.. note::

   :class:`~panobbgo.heuristics.region_ucb.RegionUCB` can now be run **alone**.
   It emitted only from ``on_new_results`` and had no ``on_start``, so a solo
   run produced 0 of 200 evaluations — a defect every benchmark hid by pairing
   it with another arm.  With an initial design it spends its budget and is
   the stronger of the two standalone tree consumers (§39).

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
