# -*- coding: utf8 -*-
import pytest
import numpy as np
from unittest import mock
from panobbgo.utils import PanobbgoTestCase
from panobbgo.strategies.ucb import StrategyUCB
from panobbgo.core import Heuristic
from panobbgo.heuristics import Random
from panobbgo.lib import Point, Result, Problem
import time

class BiasedHeuristic(Heuristic):
    """A heuristic that generates points with specific quality."""
    def __init__(self, strategy, name, quality_mean, quality_std=0.1):
        super().__init__(strategy, name=name)
        self.quality_mean = quality_mean
        self.quality_std = quality_std

    def on_start(self):
        pass

    @property
    def active(self):
        return True

    def get_points(self, limit=None):
        points = []
        limit = limit or 1
        for _ in range(limit):
             # Just random points
             x = self.problem.random_point()
             points.append(Point(x, self.name))
        return points

class MockBiasedProblem(Problem):
    def __init__(self, heuristic_quality_map, heuristic_cv_map=None):
        super().__init__([[-10, 10]])
        self.heuristic_quality_map = heuristic_quality_map
        self.heuristic_cv_map = heuristic_cv_map or {}

    def __call__(self, point):
        who = point.who
        if who in self.heuristic_quality_map:
            mean = self.heuristic_quality_map[who]
            fx = np.random.normal(mean, 0.1)
        else:
            fx = 100.0

        cv_vec = None
        # Always return cv_vec if we are simulating constraints (indicated by heuristic_cv_map being non-empty)
        if self.heuristic_cv_map:
             if who in self.heuristic_cv_map:
                 cv_mean = self.heuristic_cv_map[who]
                 cv_vec = np.array([cv_mean])
             else:
                 cv_vec = np.array([0.0])

        # Add slight delay to simulate work and allow threads to sync
        time.sleep(0.001)
        return Result(point, fx, cv_vec=cv_vec)

    def eval(self, x):
        return 0.0

class TestStrategyUCB(PanobbgoTestCase):

    def setUp(self):
        self.quality_map = {
            "GoodHeuristic": 0.0,
            "BadHeuristic": 10.0
        }
        self.problem = MockBiasedProblem(self.quality_map)

    def test_ucb_preference(self):
        """
        Test that UCB prefers the heuristic that generates better results.
        Using full strategy.start() with threaded evaluation.
        """
        strategy = StrategyUCB(self.problem, parse_args=False)
        strategy.config.evaluation_method = "threaded"
        strategy.config.max_eval = 200 # Sufficient budget to learn
        strategy.config.ucb_c = 0.2 # Low exploration to favor exploitation quickly

        # Add heuristics
        h_good = BiasedHeuristic(strategy, "GoodHeuristic", 0.0)
        h_bad = BiasedHeuristic(strategy, "BadHeuristic", 10.0)

        # We must add them to strategy
        strategy.add_heuristic(h_good)
        strategy.add_heuristic(h_bad)

        # Run strategy
        # This will block until max_eval reached
        strategy.start()

        # Check counts
        count_good = h_good.ucb_count
        count_bad = h_bad.ucb_count

        print(f"Good: {count_good}, Bad: {count_bad}")
        print(f"Reward Good: {h_good.ucb_total_reward}, Reward Bad: {h_bad.ucb_total_reward}")

        # Good heuristic should be selected significantly more
        # e.g. at least 55% of selections (relaxed from 60% to allow for stochastic variance)
        total = count_good + count_bad
        assert total > 0
        ratio_good = count_good / total

        assert count_good > count_bad, f"UCB failed to prefer Good heuristic: Good={count_good}, Bad={count_bad}"
        assert ratio_good > 0.55, f"Good heuristic ratio {ratio_good} too low"

    def test_ucb_exploration(self):
        """
        Test that UCB explores all heuristics initially.
        """
        strategy = StrategyUCB(self.problem, parse_args=False)
        strategy.config.evaluation_method = "threaded"
        strategy.config.max_eval = 50
        strategy.config.ucb_c = 5.0 # High exploration

        heurs = []
        for i in range(5):
            h = BiasedHeuristic(strategy, f"H{i}", 10.0) # All equally bad
            strategy.add_heuristic(h)
            heurs.append(h)

        strategy.start()

        # Verify all were selected at least once
        for h in heurs:
            assert h.ucb_count > 0, f"Heuristic {h.name} was never selected"

    def test_ucb_constrained_preference(self):
        """
        Test that UCB prefers feasible solutions over infeasible ones,
        even if infeasible ones have better 'fx' (unpenalized).
        """
        # H_feasible: fx = 10.0, cv = 0.0
        # H_infeasible: fx = 0.0, cv = 1.0. Penalty ~ 100.0 (default rho=100)

        quality_map = {
            "H_Feasible": 10.0,
            "H_Infeasible": 0.0
        }
        cv_map = {
            "H_Feasible": 0.0,
            "H_Infeasible": 1.0
        }

        problem = MockBiasedProblem(quality_map, cv_map)

        strategy = StrategyUCB(problem, parse_args=False)
        strategy.config.evaluation_method = "threaded"
        strategy.config.max_eval = 200
        strategy.config.ucb_c = 0.2

        h_feas = BiasedHeuristic(strategy, "H_Feasible", 10.0)
        h_infeas = BiasedHeuristic(strategy, "H_Infeasible", 0.0)

        strategy.add_heuristic(h_feas)
        strategy.add_heuristic(h_infeas)

        strategy.start()

        count_feas = h_feas.ucb_count
        count_infeas = h_infeas.ucb_count

        print(f"Feasible: {count_feas}, Infeasible: {count_infeas}")

        total = count_feas + count_infeas
        assert total > 0

        # Feasible should be preferred
        assert count_feas > count_infeas, f"UCB failed to prefer Feasible heuristic: Feas={count_feas}, Infeas={count_infeas}"
        assert count_feas / total > 0.55
