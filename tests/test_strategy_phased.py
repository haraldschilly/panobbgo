# -*- coding: utf8 -*-
import pytest
from tests.support import PanobbgoTestCase
from panobbgo.strategies.phased import StrategyPhased
from panobbgo.strategies.round_robin import StrategyRoundRobin
from panobbgo.strategies.rewarding import StrategyRewarding
from panobbgo.strategies.ucb import StrategyUCB
from panobbgo.strategies.thompson import StrategyThompsonSampling
from panobbgo.core import Heuristic
from panobbgo.lib import Point, Result, Problem
import inspect
import time

import numpy as np
import threading


class SimpleHeuristic(Heuristic):
    """A heuristic that always generates random points."""

    def __init__(self, strategy, name="SimpleH", **kwargs):
        super().__init__(strategy, name=name)

    def on_start(self):
        pass

    @property
    def active(self):
        return True

    def get_points(self, limit=None):
        limit = limit or 1
        points = []
        for _ in range(limit):
            x = self.problem.random_point()
            points.append(Point(x, self.name))
        return points


class TrackingProblem(Problem):
    """Problem that tracks which heuristic generated each evaluated point."""

    def __init__(self):
        super().__init__([[-10, 10]])
        self.call_counts: dict[str, int] = {}
        self._lock = threading.Lock()
        self._counter = 0

    def __call__(self, point):
        who = point.who
        with self._lock:
            self.call_counts[who] = self.call_counts.get(who, 0) + 1
            self._counter += 1
            # Decreasing function so we always find improvements
            fx = 100.0 - self._counter * 0.1
        time.sleep(0.001)
        return Result(point, fx)

    def eval(self, x):
        return 0.0


class TestStrategyPhasedValidation(PanobbgoTestCase):
    """Test configuration validation."""

    def test_empty_phases_rejected(self):
        problem = TrackingProblem()
        with pytest.raises(ValueError, match="non-empty list"):
            StrategyPhased(problem, phases=[])

    def test_single_phase_rejected(self):
        problem = TrackingProblem()
        with pytest.raises(ValueError, match="at least 2 phases"):
            StrategyPhased(
                problem,
                phases=[
                    {"pct": 100, "strategy": (StrategyRoundRobin, {}), "heuristics": [(SimpleHeuristic, {})]},
                ],
            )

    def test_missing_strategy_rejected(self):
        problem = TrackingProblem()
        with pytest.raises(ValueError, match="missing 'strategy'"):
            StrategyPhased(
                problem,
                phases=[
                    {"pct": 50, "heuristics": [(SimpleHeuristic, {})]},
                    {"heuristics": [(SimpleHeuristic, {})]},
                ],
            )

    def test_missing_heuristics_rejected(self):
        problem = TrackingProblem()
        with pytest.raises(ValueError, match="missing 'heuristics'"):
            StrategyPhased(
                problem,
                phases=[
                    {"pct": 50, "strategy": (StrategyRoundRobin, {})},
                    {"strategy": (StrategyRoundRobin, {}), "heuristics": [(SimpleHeuristic, {})]},
                ],
            )

    def test_pct_over_100_rejected(self):
        problem = TrackingProblem()
        with pytest.raises(ValueError, match="< 100"):
            StrategyPhased(
                problem,
                phases=[
                    {"pct": 60, "strategy": (StrategyRoundRobin, {}), "heuristics": [(SimpleHeuristic, {})]},
                    {"pct": 50, "strategy": (StrategyRoundRobin, {}), "heuristics": [(SimpleHeuristic, {})]},
                ],
            )

    def test_missing_pct_non_last_rejected(self):
        problem = TrackingProblem()
        with pytest.raises(ValueError, match="must have 'pct'"):
            StrategyPhased(
                problem,
                phases=[
                    {"strategy": (StrategyRoundRobin, {}), "heuristics": [(SimpleHeuristic, {})]},
                    {"strategy": (StrategyRoundRobin, {}), "heuristics": [(SimpleHeuristic, {})]},
                ],
            )

    def test_last_phase_pct_inferred(self):
        problem = TrackingProblem()
        strategy = StrategyPhased(
            problem,
            phases=[
                {"pct": 30, "strategy": (StrategyRoundRobin, {}), "heuristics": [(SimpleHeuristic, {})]},
                {"strategy": (StrategyRoundRobin, {}), "heuristics": [(SimpleHeuristic, {})]},
            ],
        )
        assert strategy._phase_configs[-1]["pct"] == 70.0


class TestStrategyPhasedExecution(PanobbgoTestCase):
    """Integration tests for phased execution."""

    def test_two_phase_round_robin(self):
        """Test basic two-phase execution with round-robin in both phases."""
        problem = TrackingProblem()

        strategy = StrategyPhased(
            problem,
            phases=[
                {
                    "pct": 50,
                    "strategy": (StrategyRoundRobin, {"size": 5}),
                    "heuristics": [(SimpleHeuristic, {"name": "H_Phase1"})],
                },
                {
                    "strategy": (StrategyRoundRobin, {"size": 5}),
                    "heuristics": [(SimpleHeuristic, {"name": "H_Phase2"})],
                },
            ],
            parse_args=False,
        )
        strategy.config.evaluation_method = "threaded"
        strategy.config.max_eval = 100

        strategy.start()

        # Both heuristics should have been used
        assert "H_Phase1" in problem.call_counts
        assert "H_Phase2" in problem.call_counts

        total = sum(problem.call_counts.values())
        assert total >= 90  # Allow some slack

    def test_phase_transition_happens(self):
        """Test that phase transition actually occurs at the right time."""
        problem = TrackingProblem()

        strategy = StrategyPhased(
            problem,
            phases=[
                {
                    "pct": 25,
                    "strategy": (StrategyRoundRobin, {"size": 5}),
                    "heuristics": [(SimpleHeuristic, {"name": "H_Early"})],
                },
                {
                    "strategy": (StrategyRoundRobin, {"size": 5}),
                    "heuristics": [(SimpleHeuristic, {"name": "H_Late"})],
                },
            ],
            parse_args=False,
        )
        strategy.config.evaluation_method = "threaded"
        strategy.config.max_eval = 200

        strategy.start()

        # Phase 1 gets 25% = 50 evals, Phase 2 gets 75% = 150 evals
        # H_Late should have significantly more evaluations
        count_early = problem.call_counts.get("H_Early", 0)
        count_late = problem.call_counts.get("H_Late", 0)

        assert count_late > count_early, f"Phase 2 heuristic should dominate: early={count_early}, late={count_late}"

    def test_three_phases(self):
        """Test three-phase setup with different strategies."""
        problem = TrackingProblem()

        strategy = StrategyPhased(
            problem,
            phases=[
                {
                    "pct": 25,
                    "strategy": (StrategyRoundRobin, {"size": 5}),
                    "heuristics": [(SimpleHeuristic, {"name": "H_Explore"})],
                },
                {
                    "pct": 25,
                    "strategy": (StrategyRewarding, {}),
                    "heuristics": [(SimpleHeuristic, {"name": "H_Reward"})],
                },
                {
                    "strategy": (StrategyUCB, {}),
                    "heuristics": [(SimpleHeuristic, {"name": "H_UCB"})],
                },
            ],
            parse_args=False,
        )
        strategy.config.evaluation_method = "threaded"
        strategy.config.max_eval = 200

        strategy.start()

        # All three should have been used
        assert "H_Explore" in problem.call_counts
        assert "H_Reward" in problem.call_counts
        assert "H_UCB" in problem.call_counts

    def test_mixed_strategies(self):
        """Test mixing round-robin exploration with UCB exploitation."""
        problem = TrackingProblem()

        strategy = StrategyPhased(
            problem,
            phases=[
                {
                    "pct": 30,
                    "strategy": (StrategyRoundRobin, {"size": 5}),
                    "heuristics": [(SimpleHeuristic, {"name": "H_RR"})],
                },
                {
                    "strategy": (StrategyThompsonSampling, {}),
                    "heuristics": [(SimpleHeuristic, {"name": "H_TS"})],
                },
            ],
            parse_args=False,
        )
        strategy.config.evaluation_method = "threaded"
        strategy.config.max_eval = 100

        strategy.start()

        total = sum(problem.call_counts.values())
        assert total >= 80

    def test_status_info(self):
        """Test that status info reports phase correctly."""
        problem = TrackingProblem()

        strategy = StrategyPhased(
            problem,
            phases=[
                {
                    "pct": 50,
                    "strategy": (StrategyRoundRobin, {}),
                    "heuristics": [(SimpleHeuristic, {"name": "H1"})],
                },
                {
                    "strategy": (StrategyRewarding, {}),
                    "heuristics": [(SimpleHeuristic, {"name": "H2"})],
                },
            ],
            parse_args=False,
        )
        strategy.config.evaluation_method = "threaded"
        strategy.config.max_eval = 50

        strategy.start()

        info = strategy._get_status_info()
        assert "phase" in info


class TestStrategyPhasedLinUCB(PanobbgoTestCase):
    """A LinUCB phase must learn (regression: its A/b stayed I/0 for the whole phase)."""

    def _run(self):
        from panobbgo.strategies.contextual import StrategyLinUCB

        problem = TrackingProblem()
        strategy = StrategyPhased(
            problem,
            phases=[
                {
                    "pct": 20,
                    "strategy": (StrategyRoundRobin, {"size": 5}),
                    "heuristics": [(SimpleHeuristic, {"name": "H_RR"})],
                },
                {
                    "strategy": (StrategyLinUCB, {}),
                    "heuristics": [(SimpleHeuristic, {"name": "H_First"}), (SimpleHeuristic, {"name": "H_Second"})],
                },
            ],
            parse_args=False,
            seed=3,
        )
        strategy.config.max_eval = 120
        strategy.config.sync_evaluation = True
        strategy.config.stop_on_convergence = False
        strategy.start()
        return strategy, problem

    def test_linucb_phase_updates_and_explores(self):
        strategy, problem = self._run()
        first = strategy.heuristic("H_First")
        second = strategy.heuristic("H_Second")
        # the model was updated with the phase's rewards ...
        assert np.any(first.linucb_b != 0) or np.any(second.linucb_b != 0)
        assert not np.allclose(first.linucb_A, np.eye(3))
        # ... so the picks moved off the first listed arm
        assert problem.call_counts.get("H_Second", 0) > 0, problem.call_counts

    def test_linucb_phase_first_result_earns_no_free_reward(self):
        """A new phase measures improvement against the run's best, not against nothing."""
        strategy, _ = self._run()
        strategy._init_phase_stats(1)
        h = strategy.heuristic("H_First")
        assert strategy.last_best is not None
        worse = Result(Point(np.array([0.0]), "H_First"), strategy.last_best.fx + 1.0)
        worse.point.context_vector = np.array([1.0, 0.5, 0.0])
        strategy.on_new_results([worse])
        assert np.all(h.linucb_b == 0)
        assert not np.allclose(h.linucb_A, np.eye(3))  # the pull still counts

    def test_default_alpha_matches_standalone(self):
        from panobbgo.strategies._bandit import LINUCB_ALPHA
        from panobbgo.strategies.contextual import StrategyLinUCB

        assert inspect.signature(StrategyLinUCB.__init__).parameters["linucb_alpha"].default == LINUCB_ALPHA == 2.0
        # the phase reads the same default
        assert "LINUCB_ALPHA" in inspect.getsource(StrategyPhased._execute_linucb)
