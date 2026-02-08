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
Integration Tests for Different Strategy Implementations
======================================================

These tests compare different strategies (UCB, Thompson Sampling, etc.)
and verify they function correctly.
"""

import pytest
from panobbgo.strategies.ucb import StrategyUCB
from panobbgo.strategies.thompson import StrategyThompsonSampling
from panobbgo.lib.classic import Rosenbrock
from panobbgo.heuristics import Random, Nearby


def test_strategy_ucb_execution():
    """
    Test StrategyUCB on Rosenbrock (d=2).
    Verify that UCB statistics (ucb_total_reward) are being updated.
    """
    problem = Rosenbrock(dims=2)
    strategy = StrategyUCB(
        problem,
        max_eval=100,
        evaluation_method="threaded",
        testing_mode=True
    )

    # Add heuristic
    strategy.add(Random)

    strategy.start()

    assert strategy.best is not None
    # We found a solution
    assert strategy.best.fx < float('inf')

    # Check UCB stats
    # Get heuristic by name
    heur = strategy.heuristic(Random.__name__)
    assert hasattr(heur, 'ucb_total_reward'), "Random heuristic should have ucb_total_reward"

    # Total reward should be > 0 if it improved anything.
    # If the first point is the best (often for Random starting from nothing), it gets 1.0.
    assert heur.ucb_total_reward >= 0.0, f"Expected non-negative reward, got {heur.ucb_total_reward}"

    # Check selection count
    assert hasattr(heur, 'ucb_count')
    assert heur.ucb_count > 0


def test_strategy_thompson_execution():
    """
    Test StrategyThompsonSampling on Rosenbrock (d=2).
    Verify that Thompson Sampling stats (ts_alpha, ts_beta) are updated.
    """
    problem = Rosenbrock(dims=2)
    strategy = StrategyThompsonSampling(
        problem,
        max_eval=100,
        evaluation_method="threaded",
        testing_mode=True
    )

    strategy.add(Random)

    strategy.start()

    assert strategy.best is not None
    assert strategy.best.fx < float('inf')

    # Check TS stats
    heur = strategy.heuristic(Random.__name__)
    assert hasattr(heur, 'ts_alpha'), "Random heuristic should have ts_alpha"
    assert hasattr(heur, 'ts_beta'), "Random heuristic should have ts_beta"

    # Initial values are 1.0. If it found improvements, alpha > 1.0.
    # If it failed to improve, beta > 1.0.
    # Since we start with nothing, the first point is an improvement -> alpha should increase.
    assert heur.ts_alpha >= 1.0
    assert heur.ts_beta >= 1.0

    # At least one update should have happened (first point is always best)
    assert (heur.ts_alpha > 1.0) or (heur.ts_beta > 1.0), \
        f"Expected TS parameters to update from (1.0, 1.0). Got ({heur.ts_alpha}, {heur.ts_beta})"
