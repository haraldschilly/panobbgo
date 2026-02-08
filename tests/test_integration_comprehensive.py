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
Comprehensive Integration Tests for Unconstrained Optimization
============================================================

These tests verify the end-to-end performance of strategies on various
unconstrained optimization problems.
"""

import pytest
from panobbgo.strategies.rewarding import StrategyRewarding
from panobbgo.strategies.round_robin import StrategyRoundRobin
from panobbgo.heuristics import (
    Center,
    Random,
    Nearby,
    NelderMead,
    LatinHypercube,
)
from panobbgo.lib.classic import (
    Rosenbrock,
    Rastrigin,
)


def test_rosenbrock_unconstrained_rewarding():
    """
    Test StrategyRewarding on unconstrained Rosenbrock (d=2).
    """
    problem = Rosenbrock(dims=2)
    strategy = StrategyRewarding(
        problem,
        max_eval=200,
        evaluation_method="threaded",
        testing_mode=True
    )
    strategy.config.stop_on_convergence = False

    # Add a mix of heuristics
    strategy.add(Random)
    strategy.add(Nearby)
    strategy.add(NelderMead)

    strategy.start()

    assert strategy.best is not None
    # Rosenbrock min is 0.0. With 200 evals, we should get reasonably close.
    # Typically < 1.0 is easily achievable.
    assert strategy.best.fx < 5.0, f"Failed to converge on Rosenbrock. Best: {strategy.best.fx}"
    assert strategy.best.cv <= 1e-6, "Constraint violation should be 0 for unconstrained problem"


def test_rastrigin_unconstrained_roundrobin():
    """
    Test StrategyRoundRobin on unconstrained Rastrigin (d=2).
    Rastrigin is multimodal, so RoundRobin with global search (LHS/Random) helps.
    """
    problem = Rastrigin(dims=2)
    strategy = StrategyRoundRobin(
        problem,
        max_eval=200,
        evaluation_method="threaded",
        testing_mode=True
    )
    strategy.config.stop_on_convergence = False

    strategy.add(Random)
    strategy.add(LatinHypercube, div=10)
    strategy.add(Nearby)  # Added local search to help find better minima

    strategy.start()

    assert strategy.best is not None
    # Rastrigin min is 0.0.
    # With 200 evals, we might not find global min, but should be reasonable.
    # Initial random points might be around 40-50.
    assert strategy.best.fx < 30.0, f"Failed to find decent solution on Rastrigin. Best: {strategy.best.fx}"


def test_high_dim_rosenbrock():
    """
    Test optimization on a higher dimensional problem (d=10).
    Verifies scalability and that it doesn't crash.
    """
    problem = Rosenbrock(dims=10)
    strategy = StrategyRoundRobin(
        problem,
        max_eval=500,
        evaluation_method="threaded",
        testing_mode=True
    )
    strategy.config.stop_on_convergence = False

    strategy.add(Random)
    strategy.add(Center)

    strategy.start()

    assert strategy.best is not None
    assert strategy.best.fx < float('inf')
    # Just check it ran and produced a valid result
    assert len(strategy.results) >= 500
