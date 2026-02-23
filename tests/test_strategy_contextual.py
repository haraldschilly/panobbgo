# -*- coding: utf8 -*-
import numpy as np
import pytest
from panobbgo.utils import PanobbgoTestCase
from panobbgo.strategies.contextual import StrategyLinUCB
from panobbgo.core import Heuristic
from panobbgo.lib import Point, Result, Problem
import time
import threading

class BiasedHeuristic(Heuristic):
    """A heuristic that generates points with specific quality."""
    def __init__(self, strategy, name):
        super().__init__(strategy, name=name)

    def on_start(self):
        pass

    @property
    def active(self):
        return True

    def get_points(self, limit=None):
        points = []
        limit = limit or 1
        for _ in range(limit):
             # Just random points, values determined by MockContextualProblem
             x = self.problem.random_point()
             points.append(Point(x, self.name))
        return points

class MockContextualProblem(Problem):
    """
    A problem where heuristic performance depends on time/evaluation count.

    Scenario:
    - Heuristic "EarlyBird": Good early (evals < 100), Bad late.
    - Heuristic "LateBloomer": Bad early, Good late (evals > 100).
    """
    def __init__(self, switch_point=100):
        super().__init__([[-10, 10]])
        self.switch_point = switch_point
        self.call_count = 0
        self._lock = threading.Lock()

    def __call__(self, point):
        who = point.who
        with self._lock:
            self.call_count += 1
            current_evals = self.call_count

        # Determine phase
        is_early = current_evals <= self.switch_point

        fx = 100.0 # Default bad value

        if who == "EarlyBird":
            if is_early:
                # Good: Linear descent from 50 to 0 to ensure continuous rewards
                progress = current_evals / self.switch_point
                target = 50.0 * (1.0 - progress)
                fx = target + np.random.normal(0, 0.1)
            else:
                # Bad
                fx = 100.0 + np.random.normal(0, 1.0)
        elif who == "LateBloomer":
            if is_early:
                # Bad
                fx = 100.0 + np.random.normal(0, 1.0)
            else:
                # Good: Linear descent to provide rewards in late phase
                progress = (current_evals - self.switch_point) / self.switch_point
                progress = min(1.0, progress)
                # Ensure it beats EarlyBird's best (approx 0)
                # So it needs to go below 0.
                target = -50.0 * progress
                fx = target + np.random.normal(0, 0.1)

        # Simulate work
        time.sleep(0.0001)

        return Result(point, fx)

    def eval(self, x):
        return 0.0

class TestStrategyContextual(PanobbgoTestCase):

    @pytest.mark.flaky(retries=3)
    def test_linucb_preference_switch(self):
        """
        Test that StrategyLinUCB learns to switch preference based on context (progress).
        Stochastic — the bandit may not always learn fast enough in a single run.
        """
        switch_point = 150
        max_eval = 400
        problem = MockContextualProblem(switch_point=switch_point)

        strategy = StrategyLinUCB(problem, parse_args=False)
        strategy.config.evaluation_method = "threaded"
        strategy.config.max_eval = max_eval
        strategy.config.linucb_alpha = 1.0  # Moderate exploration to allow switching
        # Disable convergence check to ensure we run full budget
        strategy.config.stop_on_convergence = False

        h_early = BiasedHeuristic(strategy, "EarlyBird")
        h_late = BiasedHeuristic(strategy, "LateBloomer")

        strategy.add_heuristic(h_early)
        strategy.add_heuristic(h_late)

        strategy.start()

        # Analyze selections
        df = strategy.results.results
        who_col = df[("who", 0)].values

        # Since evaluation is threaded, indices might not match exactly call counts.
        # But roughly first switch_point are early phase.

        early_selections = who_col[:switch_point]
        late_selections = who_col[switch_point:]

        count_early_h_early = np.sum(early_selections == "EarlyBird")
        count_early_h_late = np.sum(early_selections == "LateBloomer")

        count_late_h_early = np.sum(late_selections == "EarlyBird")
        count_late_h_late = np.sum(late_selections == "LateBloomer")

        print(f"Early Phase (0-{switch_point}): EarlyBird={count_early_h_early}, LateBloomer={count_early_h_late}")
        print(f"Late Phase ({switch_point}-{max_eval}): EarlyBird={count_late_h_early}, LateBloomer={count_late_h_late}")

        # Assertions
        # 1. Early phase: EarlyBird should be preferred
        assert count_early_h_early > count_early_h_late, "Strategy failed to prefer EarlyBird in early phase"

        # 2. Late phase: LateBloomer usage should increase relative to early phase
        late_ratio = count_late_h_late / (count_late_h_late + count_late_h_early + 1e-9)
        early_ratio = count_early_h_late / (count_early_h_late + count_early_h_early + 1e-9)

        print(f"LateBloomer Ratio: Early={early_ratio:.2f}, Late={late_ratio:.2f}")

        assert late_ratio > early_ratio, "Strategy failed to increase usage of LateBloomer in late phase"

    def test_basic_execution(self):
        """
        Verify StrategyLinUCB runs without errors on a standard problem.
        """
        from panobbgo.lib.classic import Rosenbrock
        # Fix: use dims=2
        problem = Rosenbrock(dims=2)
        strategy = StrategyLinUCB(problem, max_eval=50)
        strategy.config.evaluation_method = "threaded"
        strategy.config.stop_on_convergence = False

        from panobbgo.heuristics import Random, Center
        strategy.add(Random)
        strategy.add(Center)

        strategy.start()

        assert len(strategy.results) >= 50
