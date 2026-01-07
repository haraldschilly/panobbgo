#!/usr/bin/env python
# -*- coding: utf8 -*-

"""
Integration test for full optimization pipeline on Rosenbrock function.
Tests the complete framework from problem setup to optimization result.
"""

import pytest
from panobbgo.lib.classic import Rosenbrock
from panobbgo.strategies import StrategyRewarding
from panobbgo.heuristics import Random, Center, Nearby


@pytest.mark.slow
def test_full_optimization_pipeline():
    """
    Test the complete optimization pipeline on Rosenbrock function.

    This integration test validates:
    - Problem setup and evaluation
    - Strategy initialization and heuristic configuration
    - Full optimization run with multiple heuristics
    - Result quality assessment
    """
    # Set up the "banana-shaped" Rosenbrock function (2D for faster testing)
    problem = Rosenbrock(2)

    # Create strategy with limited evaluations for fast testing
    strategy = StrategyRewarding(problem, parse_args=False)

    # Add diverse heuristics to test different search strategies
    strategy.add(Random)
    strategy.add(Center)
    strategy.add(Nearby, radius=0.1, new=2)

    # Run the optimization
    print(f"Starting optimization with max_eval={strategy.config.max_eval}")
    strategy.start()
    print(f"Optimization completed. Total results: {len(strategy.results)}")

    # Verify we got a result
    assert strategy.best is not None, "Optimization should find a best solution"

    # Check that the result is within bounds
    assert problem.box.contains(strategy.best.x), "Best solution should be within problem bounds"

    # For Rosenbrock, the global minimum is at (1, 1) with f(x) = 0
    # With our limited evaluations, we expect to get reasonably close
    # The function value should be much better than a random point
    assert strategy.best.fx < 10.0, f"Should find a reasonably good solution, got {strategy.best.fx}"

    print(f"Optimization completed successfully!")
    print(f"Best solution: x={strategy.best.x}, f(x)={strategy.best.fx}")
    print(f"Problem: {problem}")
    print(f"Strategy: {strategy.__class__.__name__}")
    print(f"Heuristics used: {[h.__class__.__name__ for h in strategy.heuristics]}")


if __name__ == "__main__":
    test_full_optimization_pipeline()