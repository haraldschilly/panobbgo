"""
Benchmark Tests for Panobbgo Optimization Strategies

This module contains pytest-benchmark tests that evaluate optimization strategies
across various problems, dimensions, and success criteria.
"""

import pytest
import numpy as np
import time
from typing import Dict, Any

from benchmarks.problems import (
    generate_benchmark_battery, SUCCESS_CRITERIA,
    calculate_solution_quality, benchmark_result_to_dict
)
from benchmarks.strategies import BENCHMARK_STRATEGIES, get_benchmark_strategies
from panobbgo.lib.lib import Point


class BenchmarkRunner:
    """Handles running optimization benchmarks."""

    def __init__(self, max_evaluations: int = 1000):
        self.max_evaluations = max_evaluations

    def run_optimization(self, strategy, problem, global_optimum: np.ndarray,
                        global_minimum: float) -> Dict[str, Any]:
        """Run optimization and return results."""
        start_time = time.time()

        # Track best solution found
        best_fx = float('inf')
        best_x = None
        evaluations = 0

        # Manual optimization loop (similar to integration tests)
        while evaluations < self.max_evaluations:
            # Get points from strategy
            try:
                points = strategy.execute()
            except Exception as e:
                # Strategy might not be fully initialized
                break

            if not points:
                break

            # Evaluate points
            for point in points:
                result = problem(point)
                evaluations += 1

                if result.fx < best_fx:
                    best_fx = result.fx
                    best_x = point.x.copy()

                if evaluations >= self.max_evaluations:
                    break

            if evaluations >= self.max_evaluations:
                break

        elapsed_time = time.time() - start_time

        # Calculate solution quality
        if best_x is not None:
            quality = calculate_solution_quality(
                best_x, best_fx, global_optimum, global_minimum
            )
        else:
            # No solution found
            quality = {
                'param_distance': float('inf'),
                'func_distance': float('inf'),
                'relative_error': float('inf'),
                'found_fx': float('inf'),
                'true_fx': global_minimum
            }
            best_x = np.zeros_like(global_optimum)
            best_fx = float('inf')

        return {
            'best_x': best_x,
            'best_fx': best_fx,
            'evaluations': evaluations,
            'time': elapsed_time,
            'quality': quality
        }


# Global benchmark runner
runner = BenchmarkRunner(max_evaluations=1000)


@pytest.mark.parametrize("benchmark_case", generate_benchmark_battery()[:5])  # Limit for testing
@pytest.mark.parametrize("strategy_config", BENCHMARK_STRATEGIES[:3])  # Limit for testing
@pytest.mark.parametrize("success_criteria", SUCCESS_CRITERIA[:2])  # Limit for testing
def test_strategy_benchmark(benchmark, benchmark_case, strategy_config, success_criteria):
    """Benchmark optimization strategies on various problems."""

    # Create problem instance
    problem = benchmark_case.create_problem()

    # Create strategy
    strategy = strategy_config.create_strategy(problem)

    # Run benchmark
    def run_benchmark():
        return runner.run_optimization(
            strategy, problem,
            benchmark_case.global_optimum,
            benchmark_case.global_minimum
        )

    # Use pytest-benchmark
    result = benchmark(run_benchmark)

    # Extract results
    optimization_result = result['result']
    quality = optimization_result['quality']

    # Check success criteria
    success = success_criteria.is_successful(
        quality['func_distance'],
        optimization_result['evaluations']
    )

    # Store results for analysis
    benchmark_result = benchmark_result_to_dict(
        benchmark_case, success_criteria, quality,
        optimization_result['evaluations'], optimization_result['time']
    )

    # Add strategy info
    benchmark_result.update({
        'strategy': strategy_config.name,
        'success': success,
        'benchmark_time': result['stats']['mean']
    })

    # Assert that we found some solution (not infinite)
    assert optimization_result['best_fx'] != float('inf'), "Should find some solution"

    # For strict criteria on easy problems, we expect success
    if (benchmark_case.difficulty == 'easy' and
        success_criteria.name == 'strict_100'):
        assert success, f"Easy problem should succeed with {success_criteria.name}"


# Focused benchmarks for specific scenarios

@pytest.mark.parametrize("problem_name,dimension", [
    ("Sphere", 2),
    ("Sphere", 10),
    ("Rosenbrock", 2),
    ("Rastrigin", 2),
])
def test_dimension_scaling_benchmark(benchmark, problem_name, dimension):
    """Test how strategies scale with problem dimension."""

    # Find matching benchmark case
    cases = generate_benchmark_battery()
    case = next(c for c in cases
               if c.problem_name == problem_name and c.dimension == dimension)

    problem = case.create_problem()
    strategy = BENCHMARK_STRATEGIES[7].create_strategy(problem)  # bandit_basic

    def run_benchmark():
        return runner.run_optimization(
            strategy, problem, case.global_optimum, case.global_minimum
        )

    result = benchmark(run_benchmark)
    optimization_result = result['result']

    # Should find a reasonable solution
    assert optimization_result['best_fx'] < 100.0, f"Should find reasonable solution for {problem_name} in {dimension}D"


@pytest.mark.parametrize("strategy_name,success_name", [
    ("round_robin_random", "moderate_500"),
    ("bandit_basic", "moderate_500"),
    ("bandit_large_pool", "lenient_1000"),
])
def test_strategy_comparison_benchmark(benchmark, strategy_name, success_name):
    """Compare different strategies on the same problem."""

    # Use Rosenbrock 2D as standard test problem
    cases = generate_benchmark_battery()
    case = next(c for c in cases
               if c.problem_name == "Rosenbrock" and c.dimension == 2)

    problem = case.create_problem()

    # Find strategy
    strategy_config = next(s for s in BENCHMARK_STRATEGIES if s.name == strategy_name)
    strategy = strategy_config.create_strategy(problem)

    # Find success criteria
    criteria = next(c for c in SUCCESS_CRITERIA if c.name == success_name)

    def run_benchmark():
        return runner.run_optimization(
            strategy, problem, case.global_optimum, case.global_minimum
        )

    result = benchmark(run_benchmark)
    optimization_result = result['result']
    quality = optimization_result['quality']

    success = criteria.is_successful(quality['func_distance'], optimization_result['evaluations'])

    # For moderate criteria, most strategies should succeed on Rosenbrock
    if success_name == "moderate_500":
        assert success, f"{strategy_name} should succeed on Rosenbrock with {success_name}"


if __name__ == "__main__":
    # Allow running benchmarks manually for debugging
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--manual":
        print("Running manual benchmark test...")

        # Test a simple case
        cases = generate_benchmark_battery()
        case = cases[0]  # First case
        problem = case.create_problem()

        strategy_config = BENCHMARK_STRATEGIES[0]  # First strategy
        strategy = strategy_config.create_strategy(problem)

        result = runner.run_optimization(
            strategy, problem, case.global_optimum, case.global_minimum
        )

        print(f"Problem: {case.problem_name} {case.dimension}D")
        print(f"Strategy: {strategy_config.name}")
        print(f"Best: {result['best_fx']:.6f} at {result['best_x']}")
        print(f"Evaluations: {result['evaluations']}")
        print(f"Time: {result['time']:.3f}s")
        print(f"Quality: {result['quality']}")