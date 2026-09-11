# -*- coding: utf8 -*-
import pytest
import numpy as np
from unittest import mock
from panobbgo.utils import PanobbgoTestCase
from panobbgo.core import StrategyBase, Results, EventBus, Event, Module, StopHeuristic, Heuristic
from panobbgo.lib import Point, Result
from panobbgo.lib.classic import Rosenbrock


class TestCoreStrategyBase(PanobbgoTestCase):
    def test_validate_config(self):
        strategy = StrategyBase(self.problem, parse_args=False)
        strategy.config.max_eval = -1
        errors = strategy._validate_config()
        assert any("max_eval must be positive" in e for e in errors)

        strategy.config.max_eval = 200000
        errors = strategy._validate_config()
        assert any("seems unreasonably high" in e for e in errors)

        strategy.config.max_eval = "abc"
        errors = strategy._validate_config()
        assert any("must be a valid integer" in e for e in errors)

        strategy.config.discount = 1.5
        errors = strategy._validate_config()
        assert any("discount must be between 0 and 1" in e for e in errors)

        strategy.config.discount = "abc"
        errors = strategy._validate_config()
        assert any("discount must be a valid float" in e for e in errors)

        strategy.config.smooth = -0.5
        errors = strategy._validate_config()
        assert any("smooth must be non-negative" in e for e in errors)

        strategy.config.smooth = "abc"
        errors = strategy._validate_config()
        assert any("smooth must be a valid float" in e for e in errors)

        strategy.config.evaluation_method = "invalid"
        errors = strategy._validate_config()
        assert any("evaluation_method must be one of" in e for e in errors)

        # Restore configuration state to not affect other tests
        strategy.config.max_eval = 1000
        strategy.config.discount = 0.95
        strategy.config.smooth = 0.5
        strategy.config.evaluation_method = "threaded"


class TestCoreResults(PanobbgoTestCase):
    @mock.patch("panobbgo.core.StrategyBase._setup_cluster")
    def test_load_from_storage(self, mock_setup):
        strategy = StrategyBase(self.problem, parse_args=False)
        strategy.config.storage_backend = "sqlite"
        strategy.config.storage_uri = ":memory:"

        results = Results(strategy)

        # mock storage loading
        mock_backend = mock.MagicMock()
        mock_backend.load.return_value = [Result(Point(np.zeros(2), "test"), 0.0)]
        results.backend = mock_backend

        count = results.load_from_storage()
        assert count == 1

        mock_backend.load.return_value = None
        count = results.load_from_storage()
        assert count == 0

    @mock.patch("panobbgo.core.StrategyBase._setup_cluster")
    def test_get_history(self, mock_setup):
        strategy = StrategyBase(self.problem, parse_args=False)
        results = Results(strategy)

        # Empty
        hist = results.get_history()
        assert len(hist["x"]) == 0

        # Add data
        res1 = Result(Point(np.array([1.0, 2.0]), "test"), 5.0, cv_vec=np.array([0.1]))
        res2 = Result(Point(np.array([3.0, 4.0]), "test"), 2.0, cv_vec=np.array([0.0]))
        results.add_results([res1, res2])

        hist = results.get_history()
        assert len(hist["x"]) == 2
        assert np.array_equal(hist["x"][0], [1.0, 2.0])
        assert np.array_equal(hist["cv_vec"][0], [0.1])

        hist = results.get_history(n=1)
        assert len(hist["x"]) == 1
        assert np.array_equal(hist["x"][0], [3.0, 4.0])

    @mock.patch("panobbgo.core.StrategyBase._setup_cluster")
    def test_flush_buffer_edge_cases(self, mock_setup):
        strategy = StrategyBase(self.problem, parse_args=False)
        results = Results(strategy)

        # Empty buffer flush
        results._buffer = []
        results._flush_buffer()
        assert results._results_df is None

        # Buffer without cv_vec
        res = Result(Point(np.array([1.0]), "test"), 5.0, cv_vec=None)
        results._buffer = [res]
        results._flush_buffer()
        assert results._results_df is not None


class TestCoreEventBus(PanobbgoTestCase):
    @mock.patch("panobbgo.core.StrategyBase._setup_cluster")
    def test_publish_no_subscribers(self, mock_setup):
        strategy = StrategyBase(self.problem, parse_args=False)
        eb = EventBus(strategy.config)

        # Should not crash
        eb.publish("unknown_event")
        eb.publish("unknown_event", event=Event())


from unittest import mock


def test_on_converged_stops_strategy():
    problem = mock.Mock()
    strategy = StrategyBase(problem, parse_args=False)
    strategy.config.stop_on_convergence = True

    assert not strategy._stop_requested
    strategy.on_converged("Test Reason", {"stats": True})
    assert strategy._stop_requested


def test_on_converged_does_not_stop_if_configured():
    problem = mock.Mock()
    strategy = StrategyBase(problem, parse_args=False)

    strategy.config.stop_on_convergence = False
    assert not strategy._stop_requested
    strategy.on_converged("Test Reason", {"stats": True})
    assert not strategy._stop_requested


def test_results_setter_exception():
    import pandas as pd
    from unittest import mock
    from panobbgo.core import StrategyBase, Results

    problem = mock.Mock()
    strategy = StrategyBase(problem, parse_args=False)
    results = Results(strategy)

    # Create a DataFrame missing 'fx' in level 1, with a proper multi-index so xs doesn't fail on index check
    midx = pd.MultiIndex.from_tuples([("other", 0)], names=["prop", "dim"])
    df = pd.DataFrame([[1]], columns=midx)

    # It should catch KeyError and set _best_fx to inf
    results.results = df
    assert results._best_fx == float("inf")


def test_add_results_exception():
    from unittest import mock
    from panobbgo.core import StrategyBase, Results
    from panobbgo.lib import Result, Point
    import pandas as pd

    problem = mock.Mock()
    strategy = StrategyBase(problem, parse_args=False)
    results = Results(strategy)
    results.backend = None  # Disable storage backend so it doesn't fail on save() mock json

    # Create a dummy DataFrame so len(self.results) > 0 check passes
    midx = pd.MultiIndex.from_tuples([("fx", 0)], names=["prop", "dim"])
    df = pd.DataFrame([[1]], columns=midx)

    # Set the dummy df first
    results._results_df = df

    # The progress formatting tries to access xs(0, level=1)
    # If we replace xs with a method that raises an Exception
    # it will hit the try/except block at line 218 in core.py

    import numpy as np

    mock_res = Result(Point(np.array([1.0]), "test"), fx=1.0)

    with mock.patch.object(pd.DataFrame, "xs", side_effect=Exception("Test exception")):
        # Calling add_results should silently catch the exception
        results.add_results([mock_res])


def test_check_dependencies_failure():
    from unittest import mock
    from panobbgo.core import StrategyBase, Module
    import pytest

    problem = mock.Mock()
    strategy = StrategyBase(problem, parse_args=False)

    # Mock a module that fails check_dependencies
    mock_mod1 = mock.Mock()
    mock_mod1.check_dependencies.return_value = False
    mock_mod1._depends_on = []

    # Make sure we use a class type for __class__ to avoid TypeError in set()
    class MockClass:
        pass

    mock_mod1.__class__ = MockClass

    strategy._analyzers = {"mock1": mock_mod1}

    with pytest.raises(Exception, match="does not satisfy dependencies. #1"):
        strategy.check_dependencies()

    # Mock a module that depends on a non-existent class
    class DummyClass:
        pass

    mock_mod2 = mock.Mock()
    mock_mod2.check_dependencies.return_value = True
    mock_mod2._depends_on = [DummyClass]
    mock_mod2.__class__ = MockClass  # Same dummy class so it's added to available modules

    strategy._analyzers = {"mock2": mock_mod2}

    with pytest.raises(Exception, match="depends on.*but missing"):
        strategy.check_dependencies()


def test_starved_run_ends_when_nothing_can_produce():
    """A starved strategy must end the moment nothing *can* produce a point.

    Not after a wall-clock timeout: the guard this replaced
    (``config.max_stall_seconds``) was denominated in seconds, so the number
    of evaluations a seeded run performed was a function of machine speed —
    F4 of ``planning/results/2026-09-10/invariants_findings.md``.  The
    liveness predicate :meth:`StrategyBase._alive` reads the same fact from
    state: every queue empty, the bus drained, nothing in flight.  The claim
    is therefore denominated in *loops*, not seconds.
    """
    import time

    from panobbgo.strategies import StrategyRewarding

    class Silent(Heuristic):
        """Emits a single point on start, then nothing — starving the strategy."""

        def on_start(self):
            self.emit([np.zeros(self.problem.dim)])

    problem = Rosenbrock(2)
    t0 = time.time()
    with StrategyRewarding(problem, max_evaluations=50, evaluation_method="threaded") as strategy:
        strategy.add(Silent)
        strategy.start()
    elapsed = time.time() - t0

    assert len(strategy.results) < 50
    # A handful of passes: one to emit, a few for the async margin
    # (``_max_dead_loops``).  The old guard needed >= max_stall_seconds of
    # spinning to reach the same conclusion.
    assert strategy.loops < 50, f"took {strategy.loops} loops to notice starvation"
    assert elapsed < 20, f"liveness guard too slow: {elapsed:.1f}s"


def test_alive_is_false_only_when_nothing_can_change_it():
    """The liveness predicate's three terms, one at a time."""
    from panobbgo.strategies import StrategyRewarding

    class OneShot(Heuristic):
        """Emits one point on start and is never heard from again."""

        def on_start(self):
            self.emit([np.zeros(self.problem.dim)])

    problem = Rosenbrock(2)
    with StrategyRewarding(problem, max_evaluations=10, evaluation_method="threaded") as strategy:
        strategy.add(OneShot)
        strategy.initialize()
        try:
            h = strategy._heuristics["OneShot"]
            strategy.eventbus.wait_idle()
            assert strategy.eventbus.inflight == 0

            # A queued point: alive, and it is ``can_produce`` that says so.
            assert h.can_produce
            assert strategy._alive()

            # Drained, bus idle, nothing in flight -> the closed state.
            h.get_points()
            assert not h.can_produce
            assert not strategy._alive()

            # An evaluation in flight alone makes it alive again.
            strategy.pending["fake"] = object()
            assert strategy._alive()
            strategy.pending.pop("fake")
            assert not strategy._alive()
        finally:
            strategy._cleanup()
