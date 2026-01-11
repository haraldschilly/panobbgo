Setup and Quick Start Guide
===========================

This guide provides a step-by-step introduction to setting up and using Panobbgo for black-box optimization.
Each step includes executable doctests that verify the setup works correctly.

Prerequisites
-------------

Before starting, ensure you have Python ≥ 3.11 installed:

.. doctest::

   >>> import sys
   >>> print(f"Python version: {sys.version}")
   Python version: ...

Installation
------------

Step 1: Install Panobbgo
~~~~~~~~~~~~~~~~~~~~~~~~~

Install Panobbgo using UV (recommended) or pip:

.. code-block:: bash

   # Using UV (recommended)
   curl -LsSf https://astral.sh/uv/install.sh | sh
   git clone https://github.com/haraldschilly/panobbgo.git
   cd panobbgo
   uv sync --extra dev

   # Using pip
   pip install -e ".[dev]"

Verify the installation:

.. doctest::

   >>> import panobbgo
   >>> print(f"Panobbgo version: {panobbgo.__version__}")
   Panobbgo version: ...

Step 2: Check Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Verify that required dependencies are available:

.. doctest::

   >>> import numpy as np
   >>> import scipy
   >>> import pandas as pd
   >>> import matplotlib
   >>> import dask
   >>> print("All dependencies imported successfully")
   All dependencies imported successfully

   >>> print(f"NumPy: {np.__version__}, SciPy: {scipy.__version__}")
   NumPy: ..., SciPy: ...

Basic Problem Definition
------------------------

Step 3: Define Your First Problem
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a simple optimization problem by subclassing :class:`~panobbgo.lib.lib.Problem`:

.. doctest::

   >>> import numpy as np
   >>> from panobbgo.lib.lib import Problem

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
   >>> from panobbgo.lib.lib import Point
   >>> point = Point([0.0, 0.0], "test")
   >>> result = problem(point)
   >>> print(f"f([0,0]) = {result.fx}")
   f([0,0]) = 0.0

Step 4: Evaluate Points Manually
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

Basic Optimization Setup
------------------------

Step 5: Create a Strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~

Set up an optimization strategy:

.. doctest::

   >>> from panobbgo.strategies.rewarding import StrategyRewarding

   >>> # Create strategy
   >>> strategy = StrategyRewarding(problem)
   >>> strategy.config.max_eval = 50  # Set evaluation budget
   >>> print(f"Strategy created with max_evaluations: {strategy.config.max_eval}")
   Strategy created with max_evaluations: 50

Step 6: Add Heuristics
~~~~~~~~~~~~~~~~~~~~~~~

Add point generation heuristics to the strategy:

.. doctest::

   >>> from panobbgo.heuristics import Center, Random, Nearby

   >>> # Add initialization heuristic
   >>> strategy.add(Center)
   >>> strategy.add(Random)
   >>> strategy.add(Nearby, radius=0.1)
   >>> print(f"Total heuristics: {len(strategy._hs)}")
   Total heuristics: 3

Step 7: Verify Strategy Setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Verify that the strategy is properly configured:

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

Step 8: Basic Optimization Concepts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The optimization workflow is:

1. Call ``strategy.start()`` to begin optimization
2. The strategy coordinates heuristics to generate points
3. Points are evaluated in parallel using Dask
4. Results are collected and the best solution is tracked
5. Optimization continues until the evaluation budget is exhausted

Example results analysis (after optimization):

.. code-block:: python

   # Get results DataFrame
   df = strategy.results.results
   print(f"Total evaluations: {len(df)}")

   # Show best result
   if strategy.best is not None:
       print(f"Best point: {strategy.best.x}")
       print(f"Best value: {strategy.best.fx:.6f}")
       print(f"Found by: {strategy.best.who}")

Using Built-in Problems
-----------------------

Step 9: Try Built-in Test Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use Panobbgo's built-in test problems:

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

Step 10: Setup Optimization for Built-in Problem
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Configure optimization for the Rosenbrock function:

.. doctest::

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

Constrained Problems
--------------------

Step 11: Define Constrained Problem
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a problem with constraints:

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

Complete Example
----------------

Step 13: Full Setup Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Put it all together in a complete setup example:

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
   >>> print("✅ OPTIMIZATION SETUP COMPLETE")
   ✅ OPTIMIZATION SETUP COMPLETE
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

Troubleshooting
---------------

Step 14: Verify Dask Setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Check that Dask is properly configured:

.. doctest::

   >>> import dask
   >>> print(f"Dask version: {dask.__version__}")
   Dask version: ...

   >>> from dask.distributed import Client, LocalCluster
   >>> print("Dask distributed imports successful")
   Dask distributed imports successful

Step 15: Run Framework Tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Verify the framework is working with built-in tests:

.. code-block:: bash

   # Run integration tests
   uv run python tests/test_integration.py

This setup guide ensures that every step of the Panobbgo installation and basic usage is tested and verified.
If all doctests pass, your Panobbgo installation is ready for optimization tasks.

Next Steps
----------

- Learn about advanced strategies: :doc:`guide_usage`
- Customize heuristics: :doc:`guide_extending`
- Understand the mathematics: :doc:`guide_mathematical_foundation`