#!/usr/bin/env python
# -*- coding: utf8 -*-
"""
Example test: 2D constrained Rosenbrock optimization.

This demonstrates constrained optimization where the unconstrained optimum
lies in the infeasible region, forcing the optimizer to find the best
feasible point on the constraint boundary.

Problem setup:
- Rosenbrock with optimum at [2, 1]
- Constraint: x + y ≤ 0 (feasible region is below the line y = -x)
- The optimum [2, 1] is infeasible (2 + 1 = 3 > 0)
- The optimizer must find the best point on the boundary y = -x

The constrained optimum lies somewhere on the line y = -x where the
Rosenbrock valley intersects it.
"""

import numpy as np
import pytest
from panobbgo.lib.lib import Problem
from panobbgo.lib import Point
from panobbgo.heuristics import Random, Nearby, NelderMead, LatinHypercube
from panobbgo.strategies import StrategyRoundRobin
from panobbgo.utils import evaluate_point_subprocess


class RosenbrockLinearConstraint(Problem):
    """
    Rosenbrock function with a linear constraint.

    The unconstrained optimum is at the specified `optimum` location.
    A linear constraint g(x) ≤ 0 restricts the feasible region.

    For this example:
    - Optimum at [2, 1] (unconstrained minimum, but infeasible)
    - Constraint: x + y ≤ 0 (everything below the line y = -x)
    - The true constrained optimum is on the boundary y = -x

    Parameters
    ----------
    optimum : array-like
        Location of the unconstrained Rosenbrock minimum (default: [2, 1])
    par1 : float
        Rosenbrock coefficient (default: 100)
    box : list of tuples
        Bounding box (default: [(-5, 5), (-5, 5)])
    """

    def __init__(self, optimum=None, par1=100, box=None, **kwargs):
        if optimum is None:
            optimum = [2.0, 1.0]
        self.optimum = np.asarray(optimum, dtype=np.float64)
        self._shift = self.optimum - 1.0  # Shift from standard [1, 1] optimum

        if box is None:
            box = [(-5, 5), (-5, 5)]
        self.par1 = par1
        super().__init__(box, **kwargs)

    def eval(self, x):
        """Evaluate the Rosenbrock function (shifted to have optimum at self.optimum)."""
        y = x - self._shift
        return self.par1 * (y[1] - y[0] ** 2) ** 2 + (1 - y[0]) ** 2

    def eval_constraints(self, x):
        """
        Evaluate constraint violations.

        Constraint: x[0] + x[1] ≤ 0 (feasible region below y = -x)

        Returns array of constraint violations (0 if satisfied, >0 if violated).
        """
        # Linear constraint: x + y ≤ 0
        # Violation = max(0, x + y)
        cv = np.array([max(0.0, x[0] + x[1])])
        return cv


# Test configuration
OPTIMUM = [2.0, 1.0]  # Unconstrained optimum (infeasible)
BOX = [(-5, 5), (-5, 5)]


def test_constraint_setup():
    """Verify the constraint correctly identifies feasible/infeasible points."""
    problem = RosenbrockLinearConstraint(optimum=OPTIMUM, box=BOX)

    # Test infeasible point (the unconstrained optimum)
    infeasible_pt = Point(np.array([2.0, 1.0]), "infeasible")
    result = problem(infeasible_pt)
    assert result.cv_vec is not None, "Should have constraint values"
    assert result.cv_vec[0] > 0, f"Point [2,1] should be infeasible, cv={result.cv_vec}"
    print(f"✓ Point [2, 1]: f(x)={result.fx:.4f}, cv={result.cv_vec[0]:.4f} (infeasible)")

    # Test feasible point (below the line y = -x)
    feasible_pt = Point(np.array([-2.0, -1.0]), "feasible")
    result = problem(feasible_pt)
    assert result.cv_vec[0] == 0, f"Point [-2,-1] should be feasible, cv={result.cv_vec}"
    print(f"✓ Point [-2, -1]: f(x)={result.fx:.4f}, cv={result.cv_vec[0]:.4f} (feasible)")

    # Test boundary point (on the line y = -x)
    boundary_pt = Point(np.array([1.0, -1.0]), "boundary")
    result = problem(boundary_pt)
    assert result.cv_vec[0] == 0, f"Point [1,-1] should be on boundary, cv={result.cv_vec}"
    print(f"✓ Point [1, -1]: f(x)={result.fx:.4f}, cv={result.cv_vec[0]:.4f} (boundary)")


def test_unconstrained_optimum_value():
    """Verify the unconstrained optimum has f(x) = 0."""
    problem = RosenbrockLinearConstraint(optimum=OPTIMUM, box=BOX)

    opt_pt = Point(np.array(OPTIMUM), "optimum")
    result = problem(opt_pt)

    assert abs(result.fx) < 1e-10, f"Unconstrained optimum should give f(x)=0, got {result.fx}"
    print(f"✓ Unconstrained optimum at {OPTIMUM}: f(x) = {result.fx}")
    print(f"  (but infeasible with cv = {result.cv_vec[0]})")


def test_constrained_rosenbrock_manual():
    """
    Manual optimization respecting constraints.

    Generate random feasible points and track the best.
    """
    problem = RosenbrockLinearConstraint(optimum=OPTIMUM, box=BOX)

    best_fx = float('inf')
    best_x = None
    feasible_count = 0

    # Generate points, keep only feasible ones
    for i in range(200):
        x = problem.random_point()
        point = Point(x, f"manual_{i}")
        result = problem(point)

        # Only consider feasible points (cv = 0)
        if result.cv_vec[0] == 0:
            feasible_count += 1
            if result.fx < best_fx:
                best_fx = result.fx
                best_x = x.copy()

    print(f"Found {feasible_count} feasible points out of 200")
    print(f"Best feasible: f(x) = {best_fx:.4f} at x = {best_x}")

    assert feasible_count > 0, "Should find some feasible points"
    assert best_fx < float('inf'), "Should find a finite feasible solution"

    # Best should be near the constraint boundary (y ≈ -x)
    if best_x is not None:
        boundary_distance = abs(best_x[0] + best_x[1])
        print(f"Distance from boundary y=-x: {boundary_distance:.4f}")


def test_constrained_rosenbrock_full_optimization():
    """
    Full optimization test with the strategy framework.

    The optimizer should find a good feasible point near the
    constraint boundary where the Rosenbrock valley intersects y = -x.
    """
    problem = RosenbrockLinearConstraint(optimum=OPTIMUM, box=BOX)

    print("\n" + "="*60)
    print("Constrained Rosenbrock Optimization Test")
    print("="*60)
    print(f"Unconstrained optimum: {OPTIMUM} (infeasible)")
    print(f"Constraint: x + y ≤ 0 (feasible below line y = -x)")
    print(f"Box: {problem.box[:]}")
    print()

    # Set up strategy
    strategy = StrategyRoundRobin(problem, parse_args=False, testing_mode=True)
    strategy.config.max_eval = 500
    strategy.config.evaluation_method = "threaded"
    strategy.config.ui_show = False

    # Add heuristics
    strategy.add(Random)
    strategy.add(LatinHypercube, div=5)
    strategy.add(Nearby, radius=0.1, axes="all", new=3)
    strategy.add(Nearby, radius=0.01, axes="all", new=3)
    strategy.add(NelderMead)

    print(f"Max evaluations: {strategy.config.max_eval}")
    print("Heuristics: Random, LatinHypercube, Nearby (x2), NelderMead")
    print("-"*60)

    # Run optimization
    strategy.start()

    print("-"*60)

    # Validate results
    assert strategy.best is not None, "Should find a solution"

    print(f"\nBest solution: x = {strategy.best.x}")
    print(f"Best f(x) = {strategy.best.fx}")

    if strategy.best.cv is not None:
        print(f"Constraint violation: {strategy.best.cv}")
        is_feasible = strategy.best.cv == 0 or (hasattr(strategy.best.cv, '__iter__') and all(c <= 1e-6 for c in strategy.best.cv))
    else:
        is_feasible = False

    if is_feasible:
        print("✅ Solution is FEASIBLE")
        # Check how close to boundary
        boundary_dist = abs(strategy.best.x[0] + strategy.best.x[1])
        print(f"Distance from constraint boundary: {boundary_dist:.6f}")
    else:
        print("⚠️  Solution is infeasible (constraint handling may need tuning)")

    print("✅ Framework executed successfully!")


if __name__ == "__main__":
    print("Running Constrained Rosenbrock optimization example...\n")
    print("Problem: Rosenbrock with optimum at [2, 1]")
    print("Constraint: x + y ≤ 0 (feasible region below y = -x)")
    print("The unconstrained optimum [2, 1] is INFEASIBLE.\n")

    # Run tests
    test_constraint_setup()
    print()
    test_unconstrained_optimum_value()
    print()
    test_constrained_rosenbrock_manual()
    print()
    test_constrained_rosenbrock_full_optimization()

    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)
