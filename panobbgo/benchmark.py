#!/usr/bin/env python
# -*- coding: utf8 -*-
# Copyright 2012 -- 2026 Harald Schilly <harald.schilly@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Benchmark System
================

Comprehensive benchmarking framework for evaluating optimization algorithms
on well-known test problems with validation against known optima.
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

from panobbgo.lib import Problem, Result
from panobbgo.core import StrategyBase


@dataclass
class ProblemSpec:
    """
    Specification for a benchmark problem including known optima and validation parameters.
    """
    name: str
    problem_class: type
    dims: int
    known_optima: List[Dict[str, Any]]  # List of known optima with 'x' and 'fx' keys
    tolerance: float = 1e-6  # Tolerance for validation
    max_evaluations: int = 1000
    problem_kwargs: Dict[str, Any] = field(default_factory=dict)

    def create_problem(self) -> Problem:
        """Create an instance of the problem."""
        # Some problems (like Himmelblau) don't take dims parameter
        kwargs = self.problem_kwargs.copy()
        if self.problem_class.__name__ not in ['Himmelblau']:  # Add other fixed-dim problems here
            kwargs['dims'] = self.dims
        return self.problem_class(**kwargs)

    def validate_result(self, result: Result) -> Dict[str, Any]:
        """
        Validate if a result is close to any known optimum.

        Returns:
            Dict with validation results including success status and distance metrics.
        """
        if result.x is None or result.fx is None:
            return {
                'success': False,
                'distance': float('inf'),
                'closest_optimum': None,
                'param_distance': float('inf'),
                'func_distance': float('inf'),
                'tolerance': self.tolerance
            }

        best_distance = float('inf')
        best_optimum = None

        for optimum in self.known_optima:
            opt_x = np.array(optimum['x'])
            opt_fx = optimum['fx']

            # Calculate distance in parameter space
            param_distance = np.linalg.norm(result.x - opt_x)

            # Calculate distance in function space
            func_distance = abs(result.fx - opt_fx)

            # Use combined distance metric
            total_distance = param_distance + func_distance

            if total_distance < best_distance:
                best_distance = total_distance
                best_optimum = optimum

        success = best_distance <= self.tolerance

        # We already checked result.x and result.fx are not None
        assert result.x is not None and result.fx is not None
        assert best_optimum is not None

        return {
            'success': success,
            'distance': best_distance,
            'closest_optimum': best_optimum,
            'param_distance': np.linalg.norm(result.x - np.array(best_optimum['x'])),
            'func_distance': abs(result.fx - best_optimum['fx']),
            'tolerance': self.tolerance
        }


@dataclass
class StrategySpec:
    """
    Specification for a strategy configuration in benchmarks.
    """
    name: str
    strategy_class: type
    heuristics: List[Tuple[type, Dict[str, Any]]]  # List of (HeuristicClass, kwargs) tuples
    config_overrides: Dict[str, Any] = field(default_factory=dict)

    def create_strategy(self, problem: Problem) -> StrategyBase:
        """Create and configure a strategy instance."""
        strategy = self.strategy_class(problem, parse_args=False)

        # Apply config overrides
        for key, value in self.config_overrides.items():
            setattr(strategy.config, key, value)

        # Add heuristics
        for heur_class, kwargs in self.heuristics:
            strategy.add(heur_class, **kwargs)

        return strategy


@dataclass
class BenchmarkRun:
    """
    Results from a single benchmark run.
    """
    problem_spec: ProblemSpec
    strategy_spec: StrategySpec
    run_id: int
    start_time: float
    end_time: float
    best_result: Optional[Result] = None
    all_results: List[Any] = field(default_factory=list)  # pandas NamedTuples from itertuples()
    validation: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    @property
    def duration(self) -> float:
        """Duration of the benchmark run in seconds."""
        return self.end_time - self.start_time

    @property
    def success(self) -> bool:
        """Whether the run successfully found the optimum."""
        return self.validation is not None and self.validation.get('success', False)


class BenchmarkSuite:
    """
    A suite of benchmark problems and strategies to run comprehensive evaluations.
    """

    def __init__(self, name: str = "panobbgo_benchmark"):
        self.name = name
        self.problem_specs: List[ProblemSpec] = []
        self.strategy_specs: List[StrategySpec] = []
        self.runs: List[BenchmarkRun] = []

    def add_problem(self, problem_spec: ProblemSpec):
        """Add a problem specification to the suite."""
        self.problem_specs.append(problem_spec)

    def add_strategy(self, strategy_spec: StrategySpec):
        """Add a strategy specification to the suite."""
        self.strategy_specs.append(strategy_spec)

    def run_single(self, problem_spec: ProblemSpec, strategy_spec: StrategySpec,
                   max_evaluations: Optional[int] = None, run_id: int = 0, timeout: Optional[float] = None) -> BenchmarkRun:
        """
        Run a single benchmark experiment.
        """
        start_time = time.time()

        try:
            # Create problem and strategy
            problem = problem_spec.create_problem()
            strategy = strategy_spec.create_strategy(problem)

            # Override max evaluations if specified
            if max_evaluations is not None:
                strategy.config.max_eval = max_evaluations
            elif problem_spec.max_evaluations:
                strategy.config.max_eval = problem_spec.max_evaluations

            # Configure for benchmarking (use threaded evaluation)
            strategy.config.evaluation_method = "threaded"

            # Run optimization with timeout
            if timeout is not None:
                import signal
                def timeout_handler(signum, frame):
                    raise TimeoutError(f"Strategy execution timed out after {timeout} seconds")

                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(int(timeout))

                try:
                    strategy.start()
                finally:
                    signal.alarm(0)  # Cancel the alarm
            else:
                strategy.start()

            # Extract results
            best_result = strategy.best
            all_results = []
            if hasattr(strategy, 'results') and strategy.results is not None:
                # Get all results from the results database
                if hasattr(strategy.results, 'results') and strategy.results.results is not None:
                    all_results = list(strategy.results.results.itertuples(index=False))

            # Validate result
            validation = None
            if best_result:
                validation = problem_spec.validate_result(best_result)

            end_time = time.time()

            benchmark_run = BenchmarkRun(
                problem_spec=problem_spec,
                strategy_spec=strategy_spec,
                run_id=run_id,
                start_time=start_time,
                end_time=end_time,
                best_result=best_result,
                all_results=all_results,
                validation=validation
            )

        except Exception as e:
            end_time = time.time()
            benchmark_run = BenchmarkRun(
                problem_spec=problem_spec,
                strategy_spec=strategy_spec,
                run_id=run_id,
                start_time=start_time,
                end_time=end_time,
                error=str(e)
            )

        return benchmark_run

    def run_all(self, repetitions: int = 1, max_evaluations: Optional[int] = None, timeout: Optional[float] = 60.0) -> List[BenchmarkRun]:
        """
        Run all combinations of problems and strategies.
        """
        runs = []

        for problem_spec in self.problem_specs:
            for strategy_spec in self.strategy_specs:
                for rep in range(repetitions):
                    print(f"Running {problem_spec.name} with {strategy_spec.name} (rep {rep + 1}/{repetitions})")

                    run = self.run_single(problem_spec, strategy_spec, max_evaluations, rep, timeout)
                    runs.append(run)

                    if run.success:
                        if run.best_result and run.best_result.fx is not None:
                            print(f"  ✓ Success: f(x) = {run.best_result.fx:.6f} (validation: {run.validation})")
                        else:
                            print(f"  ✓ Success: (validation: {run.validation})")
                    elif run.error:
                        print(f"  ✗ Error: {run.error}")
                    else:
                        if run.best_result and run.best_result.fx is not None and run.best_result.x is not None:
                            print(f"  ⚠ Validation failed: f(x) = {run.best_result.fx:.6f} at x = {run.best_result.x}")
                            # Try validation manually to debug
                            validation = run.problem_spec.validate_result(run.best_result)
                            print(f"    Validation details: {validation}")
                        else:
                            print(f"  ⚠ No valid result found")

        self.runs.extend(runs)
        return runs

    def get_summary_dataframe(self) -> pd.DataFrame:
        """
        Get a summary DataFrame of all benchmark runs.
        """
        data = []

        for run in self.runs:
            row = {
                'problem': run.problem_spec.name,
                'strategy': run.strategy_spec.name,
                'run_id': run.run_id,
                'duration': run.duration,
                'success': run.success,
                'error': run.error,
                'best_fx': run.best_result.fx if run.best_result else None,
                'evaluations': len(run.all_results),
            }

            if run.validation:
                row.update({
                    'distance': run.validation['distance'],
                    'param_distance': run.validation['param_distance'],
                    'func_distance': run.validation['func_distance'],
                    'tolerance': run.validation['tolerance'],
                })

            data.append(row)

        return pd.DataFrame(data)

    def print_summary(self):
        """
        Print a summary of benchmark results.
        """
        if not self.runs:
            print("No benchmark runs to summarize.")
            return

        df = self.get_summary_dataframe()

        print(f"\n{'='*60}")
        print(f"BENCHMARK SUMMARY: {self.name}")
        print(f"{'='*60}")
        print(f"Total runs: {len(self.runs)}")
        print(f"Successful runs: {df['success'].sum()}")
        print(".1f")
        print(f"Failed runs: {df['error'].notna().sum()}")
        print(f"{'='*60}")

        # Group by problem and strategy
        agg_dict = {
            'success': ['count', 'mean'],
            'duration': ['mean', 'std']
        }

        # Only include columns that exist
        if 'best_fx' in df.columns:
            agg_dict['best_fx'] = ['mean', 'std']
        if 'distance' in df.columns:
            agg_dict['distance'] = ['mean', 'std']

        grouped = df.groupby(['problem', 'strategy']).agg(agg_dict).round(4)

        print("\nResults by Problem and Strategy:")
        print(grouped.to_string())

        # Overall statistics
        if df['success'].any():
            print("\nOverall Statistics (Successful Runs):")
            successful = df[df['success']]
            print(f"Mean duration: {successful['duration'].mean():.2f}s")
            print(f"Mean function value: {successful['best_fx'].mean():.6f}")
            print(f"Mean distance to optimum: {successful['distance'].mean():.6f}")


# Pre-defined benchmark problems with known optima
def create_standard_problems() -> List[ProblemSpec]:
    """
    Create a set of standard benchmark problems with known optima.
    """
    from panobbgo.lib.classic import (
        Rosenbrock, Rastrigin, Ackley, Griewank, StyblinskiTang,
        Schwefel, Himmelblau
    )

    problems = []

    # Rosenbrock function - minimum at (1, 1, ..., 1) with f(x) = 0
    problems.append(ProblemSpec(
        name="Rosenbrock_2D",
        problem_class=Rosenbrock,
        dims=2,
        known_optima=[{'x': [1.0, 1.0], 'fx': 0.0}],
        tolerance=1.0,  # Very lenient tolerance - Rosenbrock is challenging
        max_evaluations=100  # Reduced for faster benchmarking
    ))

    problems.append(ProblemSpec(
        name="Rosenbrock_5D",
        problem_class=Rosenbrock,
        dims=5,
        known_optima=[{'x': [1.0] * 5, 'fx': 0.0}],
        tolerance=1e-1,  # Relaxed tolerance for higher dimensions
        max_evaluations=300  # More evaluations for harder problem
    ))

    # Rastrigin function - minimum at (0, 0, ..., 0) with f(x) = 0
    problems.append(ProblemSpec(
        name="Rastrigin_2D",
        problem_class=Rastrigin,
        dims=2,
        known_optima=[{'x': [0.0, 0.0], 'fx': 0.0}],
        tolerance=1e-3,  # Relaxed tolerance for multimodal function
        max_evaluations=150  # Reasonable budget for multimodal optimization
    ))

    # Ackley function - minimum at (0, 0, ..., 0) with f(x) = 0
    problems.append(ProblemSpec(
        name="Ackley_2D",
        problem_class=Ackley,
        dims=2,
        known_optima=[{'x': [0.0, 0.0], 'fx': 0.0}],
        tolerance=1e-2,  # Relaxed tolerance
        max_evaluations=100
    ))

    # Griewank function - minimum at (0, 0, ..., 0) with f(x) = 0
    problems.append(ProblemSpec(
        name="Griewank_2D",
        problem_class=Griewank,
        dims=2,
        known_optima=[{'x': [0.0, 0.0], 'fx': 0.0}],
        tolerance=1e-3,  # Relaxed tolerance
        max_evaluations=100
    ))

    # Himmelblau function - multiple minima, we'll use the global one
    problems.append(ProblemSpec(
        name="Himmelblau",
        problem_class=Himmelblau,
        dims=2,
        known_optima=[
            {'x': [3.0, 2.0], 'fx': 0.0},  # One of the global minima
            {'x': [-2.805118, 3.131312], 'fx': 0.0},
            {'x': [-3.779310, -3.283186], 'fx': 0.0},
            {'x': [3.584428, -1.848126], 'fx': 0.0}
        ],
        tolerance=1e-3,  # Relaxed tolerance for multimodal function
        max_evaluations=100
    ))

    # Styblinski-Tang function - minimum at approximately (-2.903534, -2.903534, ...)
    styblinski_min_x = -2.903534
    styblinski_min_fx = -39.16617 * 2  # For 2D
    problems.append(ProblemSpec(
        name="StyblinskiTang_2D",
        problem_class=StyblinskiTang,
        dims=2,
        known_optima=[{'x': [styblinski_min_x, styblinski_min_x], 'fx': styblinski_min_fx}],
        tolerance=1e-1,  # Very relaxed tolerance for this function
        max_evaluations=200
    ))

    # Schwefel function - minimum at approximately (420.9687, 420.9687, ...)
    schwefel_min_x = 420.9687
    problems.append(ProblemSpec(
        name="Schwefel_2D",
        problem_class=Schwefel,
        dims=2,
        known_optima=[{'x': [schwefel_min_x, schwefel_min_x], 'fx': 0.0}],
        tolerance=1e-2,  # Relaxed tolerance
        max_evaluations=250
    ))

    return problems


def create_standard_strategies() -> List[StrategySpec]:
    """
    Create a set of standard strategy configurations for benchmarking.
    """
    from panobbgo.strategies import StrategyRewarding, StrategyRoundRobin, StrategyUCB
    from panobbgo.heuristics import (
        Random, Nearby, Zero, LatinHypercube, Extremal, NelderMead,
        Center
    )

    strategies = []

    # Strategy with diverse heuristics
    strategies.append(StrategySpec(
        name="Rewarding_Diverse",
        strategy_class=StrategyRewarding,
        heuristics=[
            (Random, {}),
            (Nearby, {'radius': 0.1, 'axes': 'all', 'new': 3}),
            (Center, {}),
            (NelderMead, {}),
        ]
    ))

    # Strategy focused on local search
    strategies.append(StrategySpec(
        name="Rewarding_Local",
        strategy_class=StrategyRewarding,
        heuristics=[
            (Nearby, {'radius': 0.01, 'axes': 'all', 'new': 5}),
            (Nearby, {'radius': 0.1, 'axes': 'all', 'new': 3}),
            (NelderMead, {}),
            (Center, {}),
        ]
    ))

    # Strategy with global search focus
    strategies.append(StrategySpec(
        name="Rewarding_Global",
        strategy_class=StrategyRewarding,
        heuristics=[
            (Random, {}),
            (LatinHypercube, {'div': 5}),
            (Extremal, {}),
            (Zero, {}),
        ]
    ))

    # Round-robin strategy
    strategies.append(StrategySpec(
        name="RoundRobin_Basic",
        strategy_class=StrategyRoundRobin,
        heuristics=[
            (Random, {}),
            (Nearby, {'radius': 0.1, 'axes': 'all', 'new': 3}),
            (NelderMead, {}),
        ]
    ))

    # UCB strategy
    strategies.append(StrategySpec(
        name="UCB_Basic",
        strategy_class=StrategyUCB,
        heuristics=[
            (Random, {}),
            (Nearby, {'radius': 0.1, 'axes': 'all', 'new': 3}),
            (NelderMead, {}),
        ]
    ))

    return strategies