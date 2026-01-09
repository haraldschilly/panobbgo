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


if __name__ == "__main__":
    import unittest

    unittest.main()
