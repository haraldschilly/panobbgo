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
Reproduction Tests for Strategy Learning
========================================

These tests verify that bandit strategies (Thompson Sampling, UCB) correctly
update their internal statistics based on both success (improvement) and failure
(no improvement).

Specifically, we want to ensure that if a heuristic consistently generates
non-improving points, its probability of selection decreases over time.
"""

import pytest
import numpy as np
from panobbgo.core import StrategyBase, Heuristic, Result
from panobbgo.lib import Problem, Point
from panobbgo.strategies.thompson import StrategyThompsonSampling
from panobbgo.strategies.ucb import StrategyUCB
from panobbgo.strategies.rewarding import StrategyRewarding

class MockBiasedProblem(Problem):
    """
    A mock problem that returns specific values based on the heuristic that
    generated the point. This allows us to deterministically test learning behavior.
    """
    def __init__(self, dim=2):
        super().__init__([(-5, 5)] * dim)
        self.call_count = 0

    def __call__(self, point):
        self.call_count += 1

        # Determine value based on heuristic name
        if point.who == "GoodHeuristic":
            # Always returns good value (improving)
            # We use a slight noise to ensure distinct values if needed,
            # but main property is it's low.
            fx = 0.0 + np.random.random() * 0.1
        elif point.who == "BadHeuristic":
            # Always returns bad value (non-improving)
            fx = 100.0 + np.random.random() * 0.1
        else:
            fx = 50.0

        # Create result (no constraints for simplicity)
        return Result(point, fx, cv_vec=np.array([0.0])) # Adding cv_vec to prevent shape issues

class GoodHeuristic(Heuristic):
    def __init__(self, strategy):
        super().__init__(strategy, name="GoodHeuristic")

    def on_start(self):
        # Emit one point to start
        self.emit(np.array([0.0, 0.0]))

    def on_new_results(self, results):
        # Emit more points when results come back to keep the loop going
        # In a real heuristic, we'd base this on something, but here just spam
        self.emit(np.array([0.0, 0.0]))

class BadHeuristic(Heuristic):
    def __init__(self, strategy):
        super().__init__(strategy, name="BadHeuristic")

    def on_start(self):
        self.emit(np.array([1.0, 1.0]))

    def on_new_results(self, results):
        self.emit(np.array([1.0, 1.0]))


def test_strategy_thompson_learning():
    """
    Test that StrategyThompsonSampling learns to favor GoodHeuristic over BadHeuristic.
    Specifically, check that BadHeuristic's beta parameter increases.
    """
    problem = MockBiasedProblem(dim=2)
    # Use threaded evaluation for speed and simplicity
    strategy = StrategyThompsonSampling(
        problem,
        max_eval=50, # Enough to see learning
        evaluation_method="threaded",
        testing_mode=True
    )

    strategy.add(GoodHeuristic)
    strategy.add(BadHeuristic)

    strategy.start()

    good_h = strategy.heuristic("GoodHeuristic")
    bad_h = strategy.heuristic("BadHeuristic")

    print(f"GoodHeuristic: alpha={getattr(good_h, 'ts_alpha', 'N/A')}, beta={getattr(good_h, 'ts_beta', 'N/A')}")
    print(f"BadHeuristic: alpha={getattr(bad_h, 'ts_alpha', 'N/A')}, beta={getattr(bad_h, 'ts_beta', 'N/A')}")

    # Verify GoodHeuristic learned (alpha should increase)
    assert hasattr(good_h, 'ts_alpha')
    assert good_h.ts_alpha > 1.0, "GoodHeuristic alpha should increase due to successes"

    # Verify BadHeuristic learned (beta should increase)
    # This assertion fails in the buggy implementation because beta is never updated for failures!
    assert hasattr(bad_h, 'ts_beta')
    assert bad_h.ts_beta > 1.0, "BadHeuristic beta should increase due to failures"

    # Ideally, BadHeuristic beta should be significantly higher than alpha
    assert bad_h.ts_beta > bad_h.ts_alpha, "BadHeuristic should have higher beta than alpha"


def test_strategy_ucb_learning():
    """
    Test that StrategyUCB correctly updates statistics.
    BadHeuristic should have high count and low/zero reward.
    """
    problem = MockBiasedProblem(dim=2)
    strategy = StrategyUCB(
        problem,
        max_eval=50,
        evaluation_method="threaded",
        testing_mode=True
    )

    strategy.add(GoodHeuristic)
    strategy.add(BadHeuristic)

    strategy.start()

    good_h = strategy.heuristic("GoodHeuristic")
    bad_h = strategy.heuristic("BadHeuristic")

    print(f"GoodHeuristic: count={getattr(good_h, 'ucb_count', 'N/A')}, reward={getattr(good_h, 'ucb_total_reward', 'N/A')}")
    print(f"BadHeuristic: count={getattr(bad_h, 'ucb_count', 'N/A')}, reward={getattr(bad_h, 'ucb_total_reward', 'N/A')}")

    # GoodHeuristic should have reward
    assert good_h.ucb_total_reward > 0.0

    # BadHeuristic should have count > 0 (it was tried)
    assert bad_h.ucb_count > 0

    # BadHeuristic should have very low or zero reward
    # Note: UCB might explore it a bit, but reward should be minimal compared to count
    # Since BadHeuristic always returns 100.0 and Good returns 0.0,
    # if Good finds 0.0 first, Bad (100.0) will supply 0 reward.
    # If Bad finds 100.0 first (improving over inf), it gets reward.
    # But subsequent Bad points (100.0) won't improve over 0.0 or 100.0.

    # So bad_h reward might be > 0 (initial point), but average reward should be low.
    avg_reward_good = good_h.ucb_total_reward / good_h.ucb_count
    avg_reward_bad = bad_h.ucb_total_reward / bad_h.ucb_count

    assert avg_reward_good > avg_reward_bad, "GoodHeuristic should have higher average reward"


def test_strategy_rewarding_discount():
    """
    Test that StrategyRewarding applies discount when points are generated.
    """
    problem = MockBiasedProblem(dim=2)
    # Use very high discount (punishment) to make it obvious
    # discount=0.5 -> perf halves every time
    strategy = StrategyRewarding(
        problem,
        max_eval=20, # Short run
        discount=0.5,
        evaluation_method="threaded",
        testing_mode=True
    )

    strategy.add(BadHeuristic) # Only add bad heuristic so it keeps getting selected but never rewarded

    strategy.start()

    bad_h = strategy.heuristic("BadHeuristic")

    print(f"BadHeuristic: performance={bad_h.performance}")

    # Initial performance is 1.0.
    # It generates points. Since it's 'Bad', it gets no reward (except maybe first one).
    # But it should be discounted every time it emits points.
    # So performance should be very small (<< 1.0).

    # If discount is NOT applied, performance will be >= 1.0 (1.0 + initial reward).

    assert bad_h.performance < 0.1, "Performance should be discounted significantly"
