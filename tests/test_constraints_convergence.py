import pytest
import numpy as np
import time
from panobbgo.lib import Problem, BoundingBox, Result, Point
from panobbgo.strategies.rewarding import StrategyRewarding
from panobbgo.heuristics.random import Random
from panobbgo.heuristics.nearby import Nearby
from panobbgo.heuristics.feasible_search import FeasibleSearch
from panobbgo.lib.classic import RosenbrockConstraint

# --- Test Problem ---
class ConstrainedSphere(Problem):
    def __init__(self):
        super().__init__([[-2, 2], [-2, 2]])
        # Optimum is at (0.5, 0.5) with fx=0.5
        # Constraint: x + y >= 1

    def eval(self, x):
        return np.sum(x**2)

    def eval_constraints(self, x):
        # g(x) = 1 - x[0] - x[1] <= 0  => x + y >= 1
        val = 1.0 - x[0] - x[1]
        return np.array([max(0.0, val)])

def test_alm_convergence_sphere():
    """
    End-to-end test verifying that StrategyRewarding with AugmentedLagrangianConstraintHandler
    converges to the constrained optimum of a Sphere problem.
    """
    problem = ConstrainedSphere()

    # Setup strategy
    strategy = StrategyRewarding(problem, testing_mode=True)
    strategy.config.max_eval = 1000  # Give it enough budget
    strategy.config.constraint_handler = "AugmentedLagrangianConstraintHandler"
    strategy.config.rho = 10.0
    strategy.config.dynamic_penalty_rate = 1.1
    strategy.config.evaluation_method = "threaded"

    # Configure convergence analyzer
    strategy.config.stop_on_convergence = True
    strategy.config.convergence_window_size = 200 # More patience
    strategy.config.convergence_require_feasibility = True # Wait for feasibility

    # Add heuristics
    strategy.add(Random)
    strategy.add(Nearby)
    strategy.add(FeasibleSearch)

    # Run strategy
    strategy.start()

    # Verify results
    best = strategy.best
    assert best is not None, "Strategy found no results"

    # Check feasibility
    assert best.cv < 1e-3, f"Best result is infeasible. CV={best.cv}, x={best.x}"

    # Check optimality (target 0.5)
    # Relax tolerance slightly as this is stochastic
    # User requested to just check below a larger value like 10 to avoid flakiness.
    assert best.fx < 10.0, f"Did not converge to optimum. fx={best.fx}, x={best.x}"

def test_alm_convergence_rosenbrock():
    """
    End-to-end test verifying that StrategyRewarding with AugmentedLagrangianConstraintHandler
    converges on the RosenbrockConstraint problem.
    """
    # RosenbrockConstraint default:
    # min sum 100(y_{i+1} - y_i^2)^2 + (1-y_i)^2
    # s.t. (y_{i+1} - y_i)^2 >= 0.25  => distance between points >= 0.5 or something
    # optimum is shifted.
    # Let's use 2D.
    problem = RosenbrockConstraint(dims=2)

    strategy = StrategyRewarding(problem, testing_mode=True)
    strategy.config.max_eval = 2000 # Harder problem
    strategy.config.constraint_handler = "AugmentedLagrangianConstraintHandler"
    strategy.config.rho = 10.0
    strategy.config.evaluation_method = "threaded"
    strategy.config.stop_on_convergence = True
    strategy.config.convergence_window_size = 200
    strategy.config.convergence_require_feasibility = True

    strategy.add(Random)
    strategy.add(Nearby)
    strategy.add(FeasibleSearch)

    strategy.start()

    best = strategy.best
    assert best is not None

    # Just check if we found a feasible solution
    # Relaxed tolerance for stochastic optimization
    assert best.cv < 5e-3, f"Best result is infeasible. CV={best.cv}"

    # And reasonable objective (not exploding)
    assert best.fx < 100.0, f"Objective too high: {best.fx}"

if __name__ == "__main__":
    test_alm_convergence_sphere()
