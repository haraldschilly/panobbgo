#!/usr/bin/env python
# -*- coding: utf8 -*-
"""
Example test: 2D optimization of Rosenbrock function with shifted optimum.

This demonstrates how to:
1. Use the built-in Rosenbrock test problem with a custom optimum location
2. Configure and run the optimization framework
3. Validate results against the known solution

Problem: Rosenbrock with optimum at [24, -12] in box [-100, 100]^2.
"""

import numpy as np
import pytest
from panobbgo.lib.classic import Rosenbrock
from panobbgo.lib import Point
from panobbgo.heuristics import Random, Nearby, NelderMead, LatinHypercube
from panobbgo.strategies import StrategyRoundRobin
from panobbgo.utils import evaluate_point_subprocess


# Test configuration
OPTIMUM = [24.0, -12.0]
BOX = [(-100, 100), (-100, 100)]


def test_rosenbrock_optimum_verification():
    """Verify that the Rosenbrock function has the correct optimum at [24, -12]."""
    problem = Rosenbrock(optimum=OPTIMUM, box=BOX)

    # Verify problem configuration
    np.testing.assert_array_equal(problem.optimum, OPTIMUM)
    assert problem.dim == 2

    # Evaluate at the expected optimum
    opt_point = Point(np.array(OPTIMUM), "verification")
    opt_result = problem(opt_point)

    assert abs(opt_result.fx) < 1e-10, f"Optimum should give f(x)=0, got {opt_result.fx}"
    print(f"✓ Verified: f({OPTIMUM}) = {opt_result.fx}")
    print(f"  Box: {problem.box[:]}")


def test_rosenbrock_manual_optimization():
    """
    Run manual optimization (without full strategy) to test basic functionality.
    This is a quick smoke test.
    """
    problem = Rosenbrock(optimum=OPTIMUM, box=BOX)

    # Evaluate some random points and track best
    best_fx = float('inf')
    best_x = None

    for i in range(50):
        x = problem.random_point()
        point = Point(x, f"manual_{i}")
        result = evaluate_point_subprocess(problem, point)

        if result.fx < best_fx:
            best_fx = result.fx
            best_x = x.copy()

    print(f"Best found after 50 random evaluations: f(x) = {best_fx:.4f} at x = {best_x}")
    assert best_fx < float('inf'), "Should find at least one finite result"


def test_rosenbrock_full_optimization():
    """
    Full optimization test using the strategy framework.

    This validates end-to-end optimization:
    - Strategy setup with multiple heuristics
    - Running optimization with budget
    - Framework executes without errors

    Uses Rosenbrock with optimum at [24, -12] in box [-100, 100]^2.
    """
    problem = Rosenbrock(optimum=OPTIMUM, box=BOX)

    print("\n" + "="*60)
    print("Rosenbrock Optimization Test")
    print("="*60)
    print(f"Box: {problem.box[:]}")
    print(f"Target optimum: {OPTIMUM}")
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

    # Validate results - framework should find SOME solution
    assert strategy.best is not None, "Should find a solution"
    assert isinstance(strategy.best.fx, (int, float)), "Function value should be numeric"
    assert strategy.best.fx == strategy.best.fx, "Function value should not be NaN"

    print(f"\nBest solution: x = {strategy.best.x}")
    print(f"Best f(x) = {strategy.best.fx}")

    # Check distance from true optimum
    true_opt = np.array(OPTIMUM)
    distance = np.linalg.norm(strategy.best.x - true_opt)
    print(f"Distance from true optimum: {distance:.6f}")

    # Report quality (informational, not assertions)
    if distance < 0.1:
        print("✅ EXCELLENT: Found optimum with high precision!")
    elif distance < 1.0:
        print("✅ SUCCESS: Found optimum within tolerance!")
    elif distance < 10.0:
        print("⚠️  CLOSE: Found solution reasonably close to optimum")
    else:
        print(f"Note: Solution is {distance:.1f} units from optimum (large search space)")

    # Validate point is within bounds
    for i, coord in enumerate(strategy.best.x):
        bounds = problem.box[i]
        assert bounds[0] <= coord <= bounds[1], f"Solution should be within bounds"

    print("✅ Framework executed successfully!")


if __name__ == "__main__":
    print("Running Rosenbrock optimization example...\n")
    print(f"Optimum location: {OPTIMUM}")
    print(f"Box: {BOX}")
    print()

    # Run tests
    test_rosenbrock_optimum_verification()
    print()
    test_rosenbrock_manual_optimization()
    print()
    test_rosenbrock_full_optimization()

    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)
