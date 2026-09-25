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

    @property
    def can_produce(self):
        return True  # get_points makes points on demand, the queue stays empty

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


class TestStrategyPhasedSharedHeuristic(PanobbgoTestCase):
    def test_heuristic_listed_in_two_phases_runs(self):
        """Regression: the same heuristic in two phases raised a duplicate-name ValueError."""
        problem = TrackingProblem()
        strategy = StrategyPhased(
            problem,
            phases=[
                {
                    "pct": 40,
                    "strategy": (StrategyRoundRobin, {"size": 5}),
                    "heuristics": [(SimpleHeuristic, {"name": "H_Shared"}), (SimpleHeuristic, {"name": "H_A"})],
                },
                {
                    "strategy": (StrategyUCB, {}),
                    "heuristics": [(SimpleHeuristic, {"name": "H_Shared"}), (SimpleHeuristic, {"name": "H_B"})],
                },
            ],
            parse_args=False,
            seed=1,
        )
        strategy.config.max_eval = 60
        strategy.config.sync_evaluation = True
        strategy.config.stop_on_convergence = False
        strategy.start()
        assert sorted(h.name for h in strategy.heuristics) == ["H_A", "H_B", "H_Shared"]
        assert "H_Shared" in strategy._phase_heuristic_names[0] and "H_Shared" in strategy._phase_heuristic_names[1]
        assert problem.call_counts.get("H_B", 0) > 0


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

    def test_result_from_an_earlier_linucb_phase_does_not_update_the_model(self):
        strategy, _ = self._run()
        strategy._init_phase_stats(1)
        h = strategy.heuristic("H_First")
        better = Result(Point(np.array([0.0]), "H_First"), strategy.last_best.fx - 1.0)
        better.point.context_vector = np.array([1.0, 0.5, 0.0])
        better.point.context_phase = 0  # picked under an earlier phase's (reset) model
        strategy.on_new_results([better])
        assert np.allclose(h.linucb_A, np.eye(3)) and np.all(h.linucb_b == 0)
        assert h.linucb_count == 0

    def test_new_phase_resets_linucb_counters(self):
        strategy, _ = self._run()
        h = strategy.heuristic("H_First")
        assert h.linucb_count > 0
        strategy._init_phase_stats(1)
        assert h.linucb_count == 0 and h.linucb_reward == 0.0


class QueueHeuristic(Heuristic):
    """Emits through the output queue (unlike :class:`SimpleHeuristic`) and counts what it emitted."""

    def __init__(self, strategy, name="QueueH", **kwargs):
        super().__init__(strategy, name=name)
        self.emitted = 0

    def on_start(self):
        self._top_up(10)

    def _top_up(self, n):
        self.emit([self.problem.random_point() for _ in range(n)])
        self.emitted += n

    def get_points(self, limit=None):
        points = super().get_points(limit)
        self._top_up(len(points))  # keep the queue stocked, so the arm stays alive
        return points


class TestStrategyPhasedSurplus(PanobbgoTestCase):
    def test_surplus_over_phase_budget_goes_back_to_the_queue(self):
        """Regression: points cut at the phase cutoff were dropped, not returned to their queue.

        A dropped tagged trial (LSHADE, jSO, PSO) stays in the arm's
        ``_pending`` forever and freezes its population slot.
        """
        problem = TrackingProblem()
        strategy = StrategyPhased(
            problem,
            phases=[
                # cutoff 30 = 4 * 7 + 2: the fifth pull of 7 is cut to 2
                {
                    "pct": 30,
                    "strategy": (StrategyRoundRobin, {"size": 7}),
                    "heuristics": [(QueueHeuristic, {"name": "H_A"})],
                },
                {
                    "strategy": (StrategyRoundRobin, {"size": 7}),
                    "heuristics": [(QueueHeuristic, {"name": "H_B"})],
                },
            ],
            parse_args=False,
            seed=1,
        )
        strategy.config.max_eval = 100
        strategy.config.sync_evaluation = True
        strategy.config.stop_on_convergence = False
        strategy.start()
        h = strategy.heuristic("H_A")
        evaluated = problem.call_counts.get("H_A", 0)
        assert evaluated == 30
        # every emitted point was either evaluated or is still queued (5 were dropped before the fix)
        assert h._output.qsize() == h.emitted - evaluated


class TestStrategyPhasedValidationOfStrategies(PanobbgoTestCase):
    """Unsupported strategies and kwargs fail at construction, not at the first execute()."""

    def _phases(self, strategy):
        return [
            {"pct": 50, "strategy": (StrategyRoundRobin, {}), "heuristics": [(SimpleHeuristic, {"name": "H1"})]},
            {"strategy": strategy, "heuristics": [(SimpleHeuristic, {"name": "H2"})]},
        ]

    def test_unsupported_strategy_class_rejected(self):
        from panobbgo.strategies.blocks import StrategyBlockBandit

        with pytest.raises(ValueError, match="unsupported strategy"):
            StrategyPhased(TrackingProblem(), phases=self._phases((StrategyBlockBandit, {})), parse_args=False)

    def test_unknown_kwarg_rejected(self):
        with pytest.raises(ValueError, match="not supported inside a phase"):
            StrategyPhased(TrackingProblem(), phases=self._phases((StrategyUCB, {"ucb_cc": 2.0})), parse_args=False)

    def test_unknown_credit_rejected(self):
        with pytest.raises(ValueError, match="credit"):
            StrategyPhased(
                TrackingProblem(), phases=self._phases((StrategyRewarding, {"credit": "bogus"})), parse_args=False
            )


class TestStrategyPhasedRewarding(PanobbgoTestCase):
    """A Rewarding phase runs the credit rule StrategyRewarding runs (default: EMA)."""

    def _run(self, kwargs):
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
                    "strategy": (StrategyRewarding, kwargs),
                    "heuristics": [(SimpleHeuristic, {"name": "H_A"}), (SimpleHeuristic, {"name": "H_B"})],
                },
            ],
            parse_args=False,
            seed=2,
        )
        strategy.config.max_eval = 80
        strategy.config.sync_evaluation = True
        strategy.config.stop_on_convergence = False
        strategy.start()
        return strategy

    def test_default_credit_is_ema(self):
        """Regression: the phase always ran the legacy rule, ignoring credit / config.rewarding_credit."""
        strategy = self._run({})
        assert strategy.config.rewarding_credit == "ema"
        credited = sum(strategy.heuristic(n).n_evals for n in ("H_A", "H_B"))
        assert credited > 0  # EMA counts every credited evaluation; legacy never does
        assert all(0.0 <= strategy.heuristic(n).performance <= 1.0 for n in ("H_A", "H_B"))

    def test_legacy_credit_on_request(self):
        strategy = self._run({"credit": "legacy"})
        assert all(strategy.heuristic(n).n_evals == 0 for n in ("H_A", "H_B"))


class TestStrategyPhasedDedupWarning(PanobbgoTestCase):
    def test_duplicate_with_different_kwargs_warns(self):
        import unittest.mock as mock

        problem = TrackingProblem()

        class KwHeuristic(SimpleHeuristic):
            def __init__(self, strategy, name="H_Kw", flavour=0, **kwargs):
                super().__init__(strategy, name=name)

        strategy = StrategyPhased(
            problem,
            phases=[
                {"pct": 50, "strategy": (StrategyRoundRobin, {}), "heuristics": [(KwHeuristic, {"flavour": 1})]},
                {"strategy": (StrategyRoundRobin, {}), "heuristics": [(KwHeuristic, {"flavour": 2})]},
            ],
            parse_args=False,
            seed=1,
        )
        strategy.config.max_eval = 20
        strategy.config.sync_evaluation = True
        with mock.patch.object(strategy.logger, "warning") as warn:
            strategy.start()
        assert any("different kwargs" in str(c.args[0]) for c in warn.call_args_list)
