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
import threading


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
        self.call_counts = {}
        self._lock = threading.Lock()

    def __call__(self, point):
        who = point.who
        with self._lock:
            count = self.call_counts.get(who, 0) + 1
            self.call_counts[who] = count

        if who in self.heuristic_quality_map:
            mean = self.heuristic_quality_map[who]
            base_val = 100.0 if mean > 5.0 else 20.0
            fx = max(mean, base_val - count * 1.0)
            fx += np.random.normal(0, 0.001)
        else:
            fx = 100.0

        cv_vec = None
        if self.heuristic_cv_map:
            if who in self.heuristic_cv_map:
                cv_mean = self.heuristic_cv_map[who]
                cv_vec = np.array([cv_mean])
            else:
                cv_vec = np.array([0.0])

        time.sleep(0.001)
        return Result(point, fx, cv_vec=cv_vec)

    def eval(self, x):
        return 0.0


class TestStrategyUCB(PanobbgoTestCase):
    def setUp(self):
        self.quality_map = {"GoodHeuristic": 0.0, "BadHeuristic": 10.0}
        self.problem = MockBiasedProblem(self.quality_map)

    def test_ucb_preference(self):
        """
        Test that UCB prefers the heuristic that generates better results.
        Using full strategy.start() with threaded evaluation.
        """
        strategy = StrategyUCB(self.problem, parse_args=False, max_eval=200, ucb_c=0.2)
        strategy.config.evaluation_method = "threaded"

        h_good = BiasedHeuristic(strategy, "GoodHeuristic", 0.0)
        h_bad = BiasedHeuristic(strategy, "BadHeuristic", 10.0)

        strategy.add_heuristic(h_good)
        strategy.add_heuristic(h_bad)

        strategy.start()

        count_good = h_good.ucb_count
        count_bad = h_bad.ucb_count

        total = count_good + count_bad
        assert total > 0
        ratio_good = count_good / total

        assert count_good > count_bad, f"UCB failed to prefer Good heuristic: Good={count_good}, Bad={count_bad}"
        assert ratio_good > 0.55, f"Good heuristic ratio {ratio_good} too low"

    def test_ucb_exploration(self):
        """
        Test that UCB explores all heuristics initially.
        """
        strategy = StrategyUCB(self.problem, parse_args=False, max_eval=50, ucb_c=5.0)
        strategy.config.evaluation_method = "threaded"

        heurs = []
        for i in range(5):
            h = BiasedHeuristic(strategy, f"H{i}", 10.0)  # All equally bad
            strategy.add_heuristic(h)
            heurs.append(h)

        strategy.start()

        for h in heurs:
            assert h.ucb_count > 0, f"Heuristic {h.name} was never selected"

    def test_ucb_constrained_preference(self):
        """
        Test that UCB prefers feasible solutions over infeasible ones.
        """
        quality_map = {"H_Feasible": 10.0, "H_Infeasible": 0.0}
        cv_map = {"H_Feasible": 0.0, "H_Infeasible": 1.0}

        problem = MockBiasedProblem(quality_map, cv_map)

        strategy = StrategyUCB(problem, parse_args=False, max_eval=200, ucb_c=0.2)
        strategy.config.evaluation_method = "threaded"

        h_feas = BiasedHeuristic(strategy, "H_Feasible", 10.0)
        h_infeas = BiasedHeuristic(strategy, "H_Infeasible", 0.0)

        strategy.add_heuristic(h_feas)
        strategy.add_heuristic(h_infeas)

        strategy.start()

        count_feas = h_feas.ucb_count
        count_infeas = h_infeas.ucb_count

        total = count_feas + count_infeas
        assert total > 0

        assert count_feas > count_infeas, (
            f"UCB failed to prefer Feasible heuristic: Feas={count_feas}, Infeas={count_infeas}"
        )
        assert count_feas / total > 0.55

    def test_ucb_edge_cases(self):
        """
        Test UCB edge cases for 100% coverage.
        """
        strategy = StrategyUCB(self.problem, parse_args=False)
        strategy.config.evaluation_method = "threaded"

        # Test reward edge cases
        # 1. First best gives 1.0
        reward1 = strategy.reward(Result(Point(np.zeros(1), "test"), 1.0))
        assert reward1 == 1.0

        # 2. Heuristic missing ucb_total_reward
        class BadHeuristic(Heuristic):
            def __init__(self, strat):
                super().__init__(strat, name="MissingStats")

            def on_start(self):
                pass

        # Manually bypassing add_heuristic which adds the stats
        h_missing = BadHeuristic(strategy)
        strategy._heuristics[h_missing.name] = h_missing

        # Setup last_best and call on_new_best
        strategy.last_best = Result(Point(np.zeros(1), "MissingStats"), 2.0)
        new_best = Result(Point(np.zeros(1), "MissingStats"), 1.0)

        # Should log warning but not crash
        strategy.on_new_best(new_best)

        # 3. Unknown heuristic who
        unknown_best = Result(Point(np.zeros(1), "UnknownWho"), 0.5)
        strategy.on_new_best(unknown_best)

        # 4. execute() edge cases
        # c_val handling: c=invalid
        strategy.config.ucb_c = "invalid_string"
        h1 = BiasedHeuristic(strategy, "H1", 0.0)
        strategy.add_heuristic(h1)
        h1.ucb_count = 1
        h1.ucb_total_reward = 1.0

        # Just calling execute to see if it handles the invalid c and uses default 1.414
        # We need to make sure evaluators outstanding < target
        strategy.jobs_per_client = 1
        with mock.patch.object(StrategyUCB, "evaluators", new_callable=mock.PropertyMock) as mock_evaluators:
            mock_evaluators.return_value.outstanding = []
            mock_evaluators.return_value.__len__ = mock.Mock(return_value=1)
            # Call execute safely to return 1 point
            pts = strategy.execute()
            assert len(pts) > 0

        # 5. execute() when heuristic missing stats during execute
        bh = BadHeuristic(strategy)
        strategy._heuristics[bh.name] = bh  # This one has no ucb_count
        with mock.patch.object(StrategyUCB, "evaluators", new_callable=mock.PropertyMock) as mock_evaluators:
            mock_evaluators.return_value.outstanding = []
            mock_evaluators.return_value.__len__ = mock.Mock(return_value=1)
            pts = strategy.execute()
        # Should initialize it to 0 and give it infinite score -> selected

        # 6. execution when no heuristics
        strategy._heuristics = {}
        with mock.patch.object(StrategyUCB, "evaluators", new_callable=mock.PropertyMock) as mock_evaluators:
            mock_evaluators.return_value.outstanding = []
            mock_evaluators.return_value.__len__ = mock.Mock(return_value=1)
            pts = strategy.execute()
            assert len(pts) == 0
