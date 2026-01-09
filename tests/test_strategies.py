# -*- coding: utf8 -*-
# Copyright 2012 Harald Schilly <harald.schilly@gmail.com>
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

from unittest import mock
import numpy as np

from panobbgo.heuristics.latin_hypercube import LatinHypercube
from panobbgo.heuristics.random import Random
from panobbgo.utils import PanobbgoTestCase
from panobbgo.strategies.round_robin import StrategyRoundRobin
from panobbgo.strategies.ucb import StrategyUCB
from panobbgo.lib.classic import Rosenbrock
from panobbgo.lib.lib import Result, Point


def get_my_setup_cluster():
    def my_setup_cluster(self, problem):
        # Mock Dask client instead of IPython generators/evaluators
        self._client = mock.MagicMock()
        self._problem_future = mock.MagicMock()
        # Initialize pending dict for evaluators property to work
        self.pending = {}

    return my_setup_cluster


class StrategiesTests(PanobbgoTestCase):
    def setUp(self):
        self.problem = Rosenbrock(3)

    @mock.patch(
        "panobbgo.core.StrategyBase._setup_cluster", new_callable=get_my_setup_cluster
    )
    def test_round_robin(self, my_setup_cluster):
        rr = StrategyRoundRobin(self.problem, size=20)
        rr.add(LatinHypercube, div=3)
        assert rr.size == 20
        # rr.start()
        # print rr._heuristics

    @mock.patch(
        "panobbgo.core.StrategyBase._setup_cluster", new_callable=get_my_setup_cluster
    )
    def test_rewarding(self, my_setup_cluster):
        # rwd = StrategyRewarding(self.problem)
        # assert rwd is not None
        pass

    @mock.patch(
        "panobbgo.core.StrategyBase._setup_cluster", new_callable=get_my_setup_cluster
    )
    def test_ucb_initialization(self, my_setup_cluster):
        ucb = StrategyUCB(self.problem)
        assert ucb is not None
        ucb.add(Random)
        ucb.add(LatinHypercube, div=3)

        # Manually register heuristics to simulate start() without running the loop
        for h in ucb._hs:
            ucb.add_heuristic(h)

        for h in ucb.heuristics:
            assert h.ucb_count == 0
            assert h.ucb_total_reward == 0.0

    @mock.patch(
        "panobbgo.core.StrategyBase._setup_cluster", new_callable=get_my_setup_cluster
    )
    def test_ucb_execution(self, my_setup_cluster):
        ucb = StrategyUCB(self.problem)
        ucb.add(Random)
        ucb.add(LatinHypercube, div=3)

        # Manually register heuristics to simulate start() without running the loop
        for h in ucb._hs:
            ucb.add_heuristic(h)

        # Mock evaluators.outstanding to return empty list so it tries to get points
        # The default implementation of evaluators property uses self.pending
        # my_setup_cluster initializes self.pending = {}

        # We need to simulate the loop in execute where it checks for outstanding jobs.
        # ucb.execute() checks: if len(self.evaluators.outstanding) < target:

        # By default jobs_per_client=1, len(evaluators)=len(workers).
        # We need to mock len(evaluators) to be > 0.

        # For direct evaluation, set up multiple processes to simulate multiple evaluators
        ucb._n_processes = 2

        # Initial execution should pick each heuristic at least once
        # Mock get_points to return dummy points
        for h in ucb.heuristics:
            h.get_points = mock.MagicMock(return_value=[mock.MagicMock()])

        points = ucb.execute()

        # Should have selected each at least once (because ucb_count starts at 0)
        assert len(points) >= len(ucb.heuristics)
        for h in ucb.heuristics:
            assert h.ucb_count > 0

    @mock.patch(
        "panobbgo.core.StrategyBase._setup_cluster", new_callable=get_my_setup_cluster
    )
    def test_ucb_reward(self, my_setup_cluster):
        ucb = StrategyUCB(self.problem)
        ucb.add(Random)
        # Random constructor doesn't return the instance, add appends it to _hs
        # We need to get the instance from _hs or after start

        # Manually register heuristics to simulate start() without running the loop
        for h in ucb._hs:
            ucb.add_heuristic(h)

        # After start, heuristics are in self._heuristics
        h_random = ucb.heuristic("Random")

        # Simulate initial execution
        h_random.ucb_count = 1
        h_random.ucb_total_reward = 0.0

        # Simulate a result
        p = Point(np.array([1, 2, 3]), who=h_random.name)
        result_best = Result(p, 10.0)  # fx=10.0

        # First best (no improvement, reward 1.0)
        ucb.on_new_best(result_best)

        assert h_random.ucb_total_reward == 1.0

        # Second result, improvement
        p2 = Point(np.array([1, 2, 3]), who=h_random.name)
        result_better = Result(p2, 5.0)  # fx=5.0

        ucb.on_new_best(result_better)

        # Improvement = 10.0 - 5.0 = 5.0
        # Reward = 1.0 - exp(-5.0) ~= 0.993
        expected_reward = 1.0 - np.exp(-5.0)

        assert abs(h_random.ucb_total_reward - (1.0 + expected_reward)) < 1e-6


class TestEvaluationModes(PanobbgoTestCase):
    """Test different evaluation modes."""

    def setUp(self):
        self.problem = Rosenbrock(2)

    def test_threaded_evaluation_setup(self):
        """Test that threaded evaluation mode sets up correctly."""
        from panobbgo.strategies.round_robin import StrategyRoundRobin

        strategy = StrategyRoundRobin(self.problem, parse_args=False)
        strategy.config.evaluation_method = "threaded"

        # Re-setup cluster with threaded mode
        strategy._setup_threaded_evaluation(self.problem)

        assert hasattr(strategy, '_thread_pool'), "Should have thread pool"
        assert hasattr(strategy, '_futures'), "Should have futures dict"
        assert hasattr(strategy, '_n_processes'), "Should have process count"
        assert strategy._n_processes > 0, "Should have at least 1 thread"

        # Cleanup
        strategy._thread_pool.shutdown(wait=False)

    def test_threaded_evaluation_run(self):
        """Test that threaded evaluation actually evaluates points."""
        from panobbgo.strategies.round_robin import StrategyRoundRobin
        from panobbgo.lib.lib import Point
        import time

        strategy = StrategyRoundRobin(self.problem, parse_args=False)
        strategy.config.evaluation_method = "threaded"

        # Setup threaded mode
        strategy._setup_threaded_evaluation(self.problem)
        strategy.loops = 1

        # Create test points
        points = [
            Point([0.5, 0.5], "test1"),
            Point([1.0, 1.0], "test2"),  # Global optimum
        ]

        # Run threaded evaluation - need to wait for async completion
        strategy._run_threaded_evaluation(points)

        # Wait briefly and poll again for any remaining futures
        # Threaded evaluation is async - need to give time for futures to complete
        for _ in range(10):
            if len(strategy.results) >= 2:
                break
            time.sleep(0.05)
            strategy._run_threaded_evaluation([])  # Poll for completed futures

        # Results should have been added
        assert len(strategy.results) >= 1, f"Should have at least 1 result, got {len(strategy.results)}"

        # Cleanup
        strategy._thread_pool.shutdown(wait=True)

    def test_evaluators_property(self):
        """Test evaluators property for different modes."""
        from panobbgo.strategies.round_robin import StrategyRoundRobin

        strategy = StrategyRoundRobin(self.problem, parse_args=False)
        strategy.config.evaluation_method = "threaded"
        strategy._setup_threaded_evaluation(self.problem)

        evaluators = strategy.evaluators
        assert hasattr(evaluators, 'outstanding'), "Evaluators should have outstanding property"
        assert hasattr(evaluators, '__len__'), "Evaluators should be measurable"

        # Check it returns something reasonable
        assert len(evaluators) > 0, "Should have at least 1 evaluator"

        # Cleanup
        strategy._thread_pool.shutdown(wait=False)


class TestStrategyBase(PanobbgoTestCase):
    """Test StrategyBase functionality."""

    def setUp(self):
        self.problem = Rosenbrock(2)

    @mock.patch(
        "panobbgo.core.StrategyBase._setup_cluster", new_callable=get_my_setup_cluster
    )
    def test_panobbgo_logger_initialization(self, my_setup_cluster):
        """Test that panobbgo_logger is initialized."""
        from panobbgo.strategies.round_robin import StrategyRoundRobin
        from panobbgo.logging import PanobbgoLogger

        strategy = StrategyRoundRobin(self.problem, parse_args=False)

        assert hasattr(strategy, 'panobbgo_logger'), "Should have panobbgo_logger"
        assert isinstance(strategy.panobbgo_logger, PanobbgoLogger), "Should be PanobbgoLogger instance"

    @mock.patch(
        "panobbgo.core.StrategyBase._setup_cluster", new_callable=get_my_setup_cluster
    )
    def test_results_database(self, my_setup_cluster):
        """Test Results database is properly initialized."""
        from panobbgo.strategies.round_robin import StrategyRoundRobin
        from panobbgo.core import Results

        strategy = StrategyRoundRobin(self.problem, parse_args=False)

        assert hasattr(strategy, 'results'), "Should have results"
        assert isinstance(strategy.results, Results), "Should be Results instance"
        assert len(strategy.results) == 0, "Results should start empty"

    @mock.patch(
        "panobbgo.core.StrategyBase._setup_cluster", new_callable=get_my_setup_cluster
    )
    def test_eventbus_initialization(self, my_setup_cluster):
        """Test EventBus is properly initialized."""
        from panobbgo.strategies.round_robin import StrategyRoundRobin
        from panobbgo.core import EventBus

        strategy = StrategyRoundRobin(self.problem, parse_args=False)

        assert hasattr(strategy, 'eventbus'), "Should have eventbus"
        assert isinstance(strategy.eventbus, EventBus), "Should be EventBus instance"

    @mock.patch(
        "panobbgo.core.StrategyBase._setup_cluster", new_callable=get_my_setup_cluster
    )
    def test_best_property(self, my_setup_cluster):
        """Test best property returns None when no results."""
        from panobbgo.strategies.round_robin import StrategyRoundRobin

        strategy = StrategyRoundRobin(self.problem, parse_args=False)

        # Before any evaluations, best should be None
        assert strategy.best is None, "Best should be None before any evaluations"


if __name__ == "__main__":
    import unittest

    unittest.main()
