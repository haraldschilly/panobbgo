#!/usr/bin/env python
# -*- coding: utf8 -*-

"""
Tests for the benchmark system.
"""

import numpy as np
import pytest
from panobbgo.benchmark import (
    ProblemSpec, StrategySpec, BenchmarkSuite, BenchmarkRun,
    create_standard_problems, create_standard_strategies
)
from panobbgo.lib.classic import Rosenbrock
from panobbgo.lib import Result, Point


class TestProblemSpec:
    """Test ProblemSpec functionality."""

    def test_create_problem(self):
        """Test creating a problem from spec."""
        spec = ProblemSpec(
            name="test_rosenbrock",
            problem_class=Rosenbrock,
            dims=2,
            known_optima=[{'x': [1.0, 1.0], 'fx': 0.0}],
            tolerance=1e-6
        )

        problem = spec.create_problem()
        assert isinstance(problem, Rosenbrock)
        assert problem.dim == 2

    def test_validate_result_success(self):
        """Test successful validation."""
        spec = ProblemSpec(
            name="test_rosenbrock",
            problem_class=Rosenbrock,
            dims=2,
            known_optima=[{'x': [1.0, 1.0], 'fx': 0.0}],
            tolerance=1e-3  # More reasonable tolerance for optimization
        )

        # Create a result very close to optimum
        result = Result(
            point=Point([1.0001, 1.0001], "test"),
            fx=1.01e-6,  # Actual Rosenbrock value at this point
            cv_vec=None,
            error=0.0
        )

        validation = spec.validate_result(result)
        assert validation['success'] == True
        assert validation['distance'] < spec.tolerance

    def test_validate_result_failure(self):
        """Test failed validation."""
        spec = ProblemSpec(
            name="test_rosenbrock",
            problem_class=Rosenbrock,
            dims=2,
            known_optima=[{'x': [1.0, 1.0], 'fx': 0.0}],
            tolerance=1e-6
        )

        # Create a result far from optimum
        result = Result(
            point=Point([2.0, 2.0], "test"),
            fx=400.0,
            cv_vec=None,
            error=0.0
        )

        validation = spec.validate_result(result)
        assert validation['success'] == False
        assert validation['distance'] > spec.tolerance


class TestBenchmarkSuite:
    """Test BenchmarkSuite functionality."""

    def test_add_problem_and_strategy(self):
        """Test adding problems and strategies to suite."""
        suite = BenchmarkSuite("test_suite")

        problem_spec = ProblemSpec(
            name="test_problem",
            problem_class=Rosenbrock,
            dims=2,
            known_optima=[{'x': [1.0, 1.0], 'fx': 0.0}]
        )

        # Mock strategy spec (we can't easily create real ones in tests)
        class MockStrategySpec:
            def __init__(self):
                self.name = "mock_strategy"
            def create_strategy(self, problem):
                return None

        strategy_spec = MockStrategySpec()

        suite.add_problem(problem_spec)
        suite.add_strategy(strategy_spec)

        assert len(suite.problem_specs) == 1
        assert len(suite.strategy_specs) == 1

    def test_get_summary_dataframe(self):
        """Test creating summary dataframe."""
        suite = BenchmarkSuite("test_suite")

        # Create mock run
        problem_spec = ProblemSpec(
            name="test_problem",
            problem_class=Rosenbrock,
            dims=2,
            known_optima=[{'x': [1.0, 1.0], 'fx': 0.0}]
        )

        class MockStrategySpec:
            def __init__(self):
                self.name = "mock_strategy"

        strategy_spec = MockStrategySpec()

        # Create successful run
        successful_run = BenchmarkRun(
            problem_spec=problem_spec,
            strategy_spec=strategy_spec,
            run_id=0,
            start_time=0.0,
            end_time=1.0,
            best_result=Result(
                point=Point([1.0, 1.0], "test"),
                fx=0.0,
                cv_vec=None,
                error=0.0
            ),
            validation={'success': True, 'distance': 0.0, 'param_distance': 0.0,
                       'func_distance': 0.0, 'tolerance': 1e-6, 'closest_optimum': {'x': [1.0, 1.0], 'fx': 0.0}}
        )

        suite.runs = [successful_run]

        df = suite.get_summary_dataframe()
        assert len(df) == 1
        assert df.iloc[0]['success'] == True
        assert df.iloc[0]['best_fx'] == 0.0


class TestStandardProblems:
    """Test standard problem creation."""

    def test_create_standard_problems(self):
        """Test creating standard problems."""
        problems = create_standard_problems()

        assert len(problems) > 0
        assert all(isinstance(p, ProblemSpec) for p in problems)

        # Check that we have expected problems
        problem_names = [p.name for p in problems]
        assert "Rosenbrock_2D" in problem_names
        assert "Rastrigin_2D" in problem_names
        assert "Ackley_2D" in problem_names

        # Test that problems can be created
        for problem_spec in problems:
            problem = problem_spec.create_problem()
            assert problem is not None
            assert problem.dim == problem_spec.dims


class TestStandardStrategies:
    """Test standard strategy creation."""

    def test_create_standard_strategies(self):
        """Test creating standard strategies."""
        strategies = create_standard_strategies()

        assert len(strategies) > 0
        assert all(isinstance(s, StrategySpec) for s in strategies)

        # Check that we have expected strategies
        strategy_names = [s.name for s in strategies]
        assert "Rewarding_Diverse" in strategy_names
        assert "RoundRobin_Basic" in strategy_names
        assert "UCB_Basic" in strategy_names


class TestBenchmarkIntegration:
    """Integration tests for the complete benchmark system."""

    def test_simple_benchmark_run(self):
        """Test that a simple benchmark runs successfully."""
        from panobbgo.benchmark import BenchmarkSuite, ProblemSpec, StrategySpec
        from panobbgo.lib.classic import Rosenbrock
        from panobbgo.strategies import StrategyRoundRobin
        from panobbgo.heuristics import Random, Nearby

        # Create a minimal benchmark suite
        suite = BenchmarkSuite("integration_test")

        # Add a simple problem
        problem_spec = ProblemSpec(
            name="Rosenbrock_2D_Integration",
            problem_class=Rosenbrock,
            dims=2,
            known_optima=[{'x': [1.0, 1.0], 'fx': 0.0}],
            tolerance=1e-1,  # Relaxed tolerance for integration test
            max_evaluations=20  # Very few evaluations for fast test
        )
        suite.add_problem(problem_spec)

        # Add a simple strategy
        strategy_spec = StrategySpec(
            name="RoundRobin_Simple",
            strategy_class=StrategyRoundRobin,
            heuristics=[
                (Random, {}),
                (Nearby, {'radius': 0.1, 'axes': 'all', 'new': 2}),
            ]
        )
        suite.add_strategy(strategy_spec)

        # Run the benchmark
        runs = suite.run_all(repetitions=1, max_evaluations=20)

        # Verify we got results
        assert len(runs) == 1
        run = runs[0]

        # Check basic structure
        assert run.problem_spec.name == "Rosenbrock_2D_Integration"
        assert run.strategy_spec.name == "RoundRobin_Simple"
        assert run.end_time > run.start_time  # Took some time

        # Check we got some evaluations (even if validation fails)
        assert len(run.all_results) > 0

        # The run should have either succeeded or failed gracefully
        assert run.best_result is not None or run.error is not None

        print(f"Integration test completed: {len(run.all_results)} evaluations, "
              f"best fx = {run.best_result.fx if run.best_result else 'N/A'}")

    def test_benchmark_result_summary(self):
        """Test that benchmark results can be summarized."""
        from panobbgo.benchmark import BenchmarkSuite, ProblemSpec, BenchmarkRun
        from panobbgo.lib.classic import Rosenbrock
        from panobbgo.lib import Result, Point

        suite = BenchmarkSuite("summary_test")

        # Create a mock successful run
        problem_spec = ProblemSpec(
            name="test_problem",
            problem_class=Rosenbrock,
            dims=2,
            known_optima=[{'x': [1.0, 1.0], 'fx': 0.0}],
            tolerance=1e-3
        )

        # Mock strategy spec
        class MockStrategySpec:
            def __init__(self):
                self.name = "mock_strategy"

        strategy_spec = MockStrategySpec()

        # Create successful run with validation
        successful_run = BenchmarkRun(
            problem_spec=problem_spec,
            strategy_spec=strategy_spec,
            run_id=0,
            start_time=0.0,
            end_time=1.0,
            best_result=Result(
                point=Point([1.0, 1.0], "test"),
                fx=0.0,
                cv_vec=None,
                error=0.0
            ),
            all_results=[],
            validation={
                'success': True,
                'distance': 0.0,
                'closest_optimum': {'x': [1.0, 1.0], 'fx': 0.0},
                'param_distance': 0.0,
                'func_distance': 0.0,
                'tolerance': 1e-3
            }
        )

        suite.runs = [successful_run]

        # Test summary generation
        df = suite.get_summary_dataframe()
        assert len(df) == 1
        assert df.iloc[0]['success'] == True
        assert df.iloc[0]['best_fx'] == 0.0

        # Test summary printing (should not crash)
        suite.print_summary()


if __name__ == "__main__":
    pytest.main([__file__])