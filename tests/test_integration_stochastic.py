# -*- coding: utf8 -*-
# Copyright 2025 Panobbgo Contributors
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
Integration Tests for Stochastic Optimization
=============================================

These tests verify the framework's behavior when dealing with noisy objective functions.
"""

import pytest
from panobbgo.strategies.rewarding import StrategyRewarding
from panobbgo.heuristics import Random, Nearby
from panobbgo.lib.classic import RosenbrockStochastic


def test_rosenbrock_stochastic():
    """
    Test optimization of a noisy Rosenbrock function.
    """
    problem = RosenbrockStochastic(dims=2, jitter=0.1)
    strategy = StrategyRewarding(problem, max_eval=300, evaluation_method="threaded", testing_mode=True)
    strategy.config.stop_on_convergence = False

    strategy.add(Random)
    strategy.add(Nearby)

    strategy.start()

    assert strategy.best is not None
    # With noise, we can't expect to find exactly 0.0.
    # But it should be reasonably close to the minimum region.
    # 10.0 is a loose upper bound to ensure we didn't get stuck in a terrible local min.
    assert strategy.best.fx < 10.0, f"Failed to optimize noisy Rosenbrock. Best: {strategy.best.fx}"

    # Check that we actually have multiple evaluations
    assert len(strategy.results) >= 300


def test_stochastic_robustness():
    """
    Verify that the optimizer can run multiple trials without crashing on stochastic problems.
    """
    # Use a slightly higher jitter
    problem = RosenbrockStochastic(dims=2, jitter=0.5)

    # Run a short optimization
    strategy = StrategyRewarding(problem, max_eval=50, evaluation_method="threaded", testing_mode=True)

    strategy.add(Random)

    strategy.start()

    assert strategy.best is not None
    assert strategy.best.fx < float("inf")

    # Check if results show variance for nearby points (difficult to assert directly without re-evaluating same point)
    # But we can check that we have results
    assert len(strategy.results) >= 50
