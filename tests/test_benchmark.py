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


if __name__ == "__main__":
    pytest.main([__file__])