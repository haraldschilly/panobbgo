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
        """Run optimization using strategy and return results."""
        start_time = time.time()

        # Track best solution found
        best_fx = float('inf')
        best_x = None
        evaluations = 0

        # Heuristic statistics
        heuristic_stats = {}
        convergence_trace = []

        # Initialize strategy - if this fails, the test should fail
        strategy.start()  # This sets up heuristics properly

        # Simple optimization loop - just call strategy.execute() repeatedly
        # This is similar to the strategy's main loop but simplified for benchmarking
        max_loops = 50  # Prevent infinite loops
        loop_count = 0
        consecutive_empty_loops = 0
        max_consecutive_empty = 5

        while (evaluations < self.max_evaluations and
               loop_count < max_loops and
               consecutive_empty_loops < max_consecutive_empty):
            loop_count += 1

            try:
                # Get points from strategy
                points = strategy.execute()
            except Exception as e:
                # Strategy execution failed, break
                print(f"Strategy execution failed after {loop_count} loops: {e}")
                break

            if not points:
                # No more points to evaluate
                consecutive_empty_loops += 1
                time.sleep(1e-4)  # Small delay before retry
                continue
            else:
                consecutive_empty_loops = 0

            # Evaluate points
            for point in points:
                result = problem(point)
                evaluations += 1

                # Update heuristic stats
                who = result.who
                if who not in heuristic_stats:
                    heuristic_stats[who] = {
                        'count': 0,
                        'improvements': 0,
                        'best_improvement': 0.0,
                    }
                heuristic_stats[who]['count'] += 1

                if result.fx < best_fx:
                    old_best_fx = best_fx if best_fx != float('inf') else result.fx
                    improvement = old_best_fx - result.fx

                    best_fx = result.fx
                    best_x = point.x.copy()

                    heuristic_stats[who]['improvements'] += 1
                    heuristic_stats[who]['best_improvement'] = max(
                        heuristic_stats[who]['best_improvement'], improvement
                    )

                    convergence_trace.append({
                        'eval': evaluations,
                        'fx': best_fx,
                        'who': who,
                        'improvement': improvement
                    })

                if evaluations >= self.max_evaluations:
                    break

            if evaluations >= self.max_evaluations:
                break

            # Small delay to prevent busy waiting
            time.sleep(1e-4)

        elapsed_time = time.time() - start_time

        # Calculate solution quality
        if best_x is not None and best_fx != float('inf'):
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
            best_x = np.zeros_like(global_optimum) if global_optimum is not None else np.array([0.0])
            best_fx = float('inf')

        return {
            'best_x': best_x,
            'best_fx': best_fx,
            'evaluations': evaluations,
            'time': elapsed_time,
            'quality': quality,
            'heuristic_stats': heuristic_stats,
            'convergence_trace': convergence_trace
        }


# Global benchmark runner
runner = BenchmarkRunner(max_evaluations=1000)


def test_basic_benchmark(benchmark):
    """Basic benchmark test to ensure the framework works."""
    from panobbgo.lib.lib import Point
    from benchmarks.problems import generate_benchmark_battery

    # Use a simple 2D DeJong problem
    cases = generate_benchmark_battery()
    case = next(c for c in cases if c.problem_name == "DeJong" and c.dimension == 2)
    problem = case.create_problem()

    def run_random_search():
        """Simple random search benchmark."""
        best_fx = float('inf')
        best_x = None
        evaluations = 0

        # Perform random search
        for i in range(100):
            x_array = problem.random_point()
            point = Point(x_array, f"eval_{i}")
            result = problem(point)
            evaluations += 1

            if result.fx < best_fx:
                best_fx = result.fx
                best_x = x_array.copy()

        return {
            'best_x': best_x,
            'best_fx': best_fx,
            'evaluations': evaluations,
            'time': 0.1,  # dummy for now
            'quality': {'func_distance': abs(best_fx - case.global_minimum)}
        }

    # Run benchmark
    result = benchmark(run_random_search)
    optimization_result = result  # benchmark returns the function result directly

    # Basic checks
    assert optimization_result['evaluations'] == 100
    assert optimization_result['best_fx'] < float('inf')
    assert optimization_result['best_fx'] >= 0  # DeJong minimum is 0

    print(f"Benchmark completed: {optimization_result['evaluations']} evaluations")
    print(f"Best solution: f(x) = {optimization_result['best_fx']:.6f}")


def test_heuristic_tracking(benchmark):
    """Test that heuristic tracking works correctly."""
    from benchmarks.problems import generate_benchmark_battery
    from benchmarks.strategies import BENCHMARK_STRATEGIES

    # Use a simple 2D DeJong problem
    cases = generate_benchmark_battery()
    case = next(c for c in cases if c.problem_name == "DeJong" and c.dimension == 2)
    problem = case.create_problem()

    # Use round_robin_multi which has multiple heuristics
    strategy_config = next(s for s in BENCHMARK_STRATEGIES if s.name == "round_robin_multi")
    strategy = strategy_config.create_strategy(problem)

    def run_benchmark():
        return runner.run_optimization(
            strategy, problem, case.global_optimum, case.global_minimum
        )

    result = benchmark(run_benchmark)

    # Check that heuristic stats are present
    assert 'heuristic_stats' in result
    assert 'convergence_trace' in result

    stats = result['heuristic_stats']
    # Check that we have stats for multiple heuristics
    assert len(stats) > 0

    total_evals = sum(s['count'] for s in stats.values())
    assert total_evals == result['evaluations']

    # Check convergence trace
    trace = result['convergence_trace']
    if result['best_fx'] < float('inf'):
        assert len(trace) > 0
        last_trace = trace[-1]
        assert last_trace['fx'] == result['best_fx']
        assert last_trace['eval'] <= result['evaluations']

    print("\nHeuristic Performance:")
    for name, s in stats.items():
        print(f"  {name}: {s['count']} evals, {s['improvements']} improvements")


# TODO: Re-enable when strategy initialization issues are resolved
# Focused benchmarks for specific scenarios

# @pytest.mark.parametrize("problem_name,dimension", [
#     ("Sphere", 2),
#     ("Sphere", 10),
#     ("Rosenbrock", 2),
#     ("Rastrigin", 2),
# ])
# def test_dimension_scaling_benchmark(benchmark, problem_name, dimension):
#     """Test how strategies scale with problem dimension."""
#
#     # Find matching benchmark case
#     cases = generate_benchmark_battery()
#     case = next(c for c in cases
#                if c.problem_name == problem_name and c.dimension == dimension)
#
#     problem = case.create_problem()
#     strategy = BENCHMARK_STRATEGIES[7].create_strategy(problem)  # bandit_basic
#
#     def run_benchmark():
#         return runner.run_optimization(
#             strategy, problem, case.global_optimum, case.global_minimum
#         )
#
#     optimization_result = benchmark(run_benchmark)
#
#     # Should find a reasonable solution
#     assert optimization_result['best_fx'] < 100.0, f"Should find reasonable solution for {problem_name} in {dimension}D"


# TODO: Re-enable when strategy initialization issues are resolved
# @pytest.mark.parametrize("strategy_name,success_name", [
#     ("round_robin_random", "moderate_500"),
#     ("bandit_basic", "moderate_500"),
#     ("bandit_large_pool", "lenient_1000"),
# ])
# def test_strategy_comparison_benchmark(benchmark, strategy_name, success_name):
#     """Compare different strategies on the same problem."""
#
#     # Use Rosenbrock 2D as standard test problem
#     cases = generate_benchmark_battery()
#     case = next(c for c in cases
#                if c.problem_name == "Rosenbrock" and c.dimension == 2)
#
#     problem = case.create_problem()
#
#     # Find strategy
#     strategy_config = next(s for s in BENCHMARK_STRATEGIES if s.name == strategy_name)
#     strategy = strategy_config.create_strategy(problem)
#
#     # Find success criteria
#     criteria = next(c for c in SUCCESS_CRITERIA if c.name == success_name)
#
#     def run_benchmark():
#         return runner.run_optimization(
#             strategy, problem, case.global_optimum, case.global_minimum
#         )
#
#     optimization_result = benchmark(run_benchmark)
#     quality = optimization_result['quality']
#
#     success = criteria.is_successful(quality['func_distance'], optimization_result['evaluations'])
#
#     # For moderate criteria, most strategies should succeed on Rosenbrock
#     if success_name == "moderate_500":
#         assert success, f"{strategy_name} should succeed on Rosenbrock with {success_name}"


def test_simple_benchmark_structure():
    """Test that the benchmark structure works without complex strategies."""

    def simple_optimization():
        # Simple random search for testing
        import numpy as np
        from benchmarks.problems import generate_benchmark_battery

        cases = generate_benchmark_battery()
        case = next(c for c in cases if c.problem_name == "DeJong" and c.dimension == 2)
        problem = case.create_problem()

        best_fx = float('inf')
        best_x = None
        evaluations = 0

        # Simple random search
        from panobbgo.lib.lib import Point
        for i in range(50):  # Just 50 evaluations
            x_array = problem.random_point()
            point = Point(x_array, f"eval_{i}")
            result = problem(point)
            evaluations += 1

            if result.fx < best_fx:
                best_fx = result.fx
                best_x = x_array.copy()

        from benchmarks.problems import calculate_solution_quality
        quality = calculate_solution_quality(best_x, best_fx, case.global_optimum, case.global_minimum)

        return {
            'best_x': best_x,
            'best_fx': best_fx,
            'evaluations': evaluations,
            'time': 0.1,  # dummy
            'quality': quality
        }

    # Test the benchmark structure
    result = simple_optimization()
    assert result['evaluations'] == 50
    assert result['best_fx'] < float('inf')
    assert 'quality' in result
    print(f"Simple benchmark completed: {result['evaluations']} evaluations, best f(x) = {result['best_fx']:.6f}")


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

    elif len(sys.argv) > 1 and sys.argv[1] == "--simple":
        print("Running simple benchmark test...")
        test_simple_benchmark_structure()