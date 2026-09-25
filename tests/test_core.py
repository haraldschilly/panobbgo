# -*- coding: utf8 -*-
import pytest
import numpy as np
from unittest import mock
from tests.support import PanobbgoTestCase
from panobbgo.core import StrategyBase, Results, EventBus, Event, Module, StopHeuristic, Heuristic
from panobbgo.lib import Point, Result
from panobbgo.lib.classic import Rosenbrock


class TestCoreStrategyBase(PanobbgoTestCase):
    def test_validate_config(self):
        strategy = StrategyBase(self.problem, parse_args=False)
        strategy.config.max_eval = -1
        errors = strategy._validate_config()
        assert any("max_eval must be positive" in e for e in errors)

        strategy.config.max_eval = 200000  # large is a warning, not an error
        assert strategy._validate_config() == []

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


def test_heuristic_subprocess_stop_terminates_the_worker(strategy):
    """``HeuristicSubprocess`` had no ``__stop__``: its worker lived until interpreter exit."""
    from panobbgo.core import HeuristicSubprocess

    h = HeuristicSubprocess(strategy)
    proc = h._HeuristicSubprocess__subprocess
    assert proc.is_alive()
    h.__stop__()
    assert not proc.is_alive()
    assert h.pipe.closed and h.pipe_child.closed


def test_type_error_inside_a_terminate_handler_is_reported(strategy):
    """A TypeError from a lifecycle handler's *body* used to be swallowed as a signature mismatch."""
    bus = strategy.eventbus

    class Mod:
        name = "Mod"

        def __init__(self):
            self.calls = []

        def on_start(self):
            self.calls.append("start")
            raise TypeError("bug in the body")

        def on_finished(self, unexpected):  # cannot take the payload-less event
            self.calls.append("finished")

    m = Mod()
    bus.register(m)
    with mock.patch.object(bus.logger, "critical") as crit:
        bus.publish("start", terminate=True)
        bus.publish("finished", terminate=True)
        assert bus.wait_idle(timeout=5)
    assert m.calls == ["start"]
    assert crit.call_count == 1 and "bug in the body" in crit.call_args[0][0]
    assert not bus.is_subscribed(m)  # both one-shot subscriptions ended
    bus.shutdown()


def test_on_finished_reaches_every_module_before_stop():
    """``__stop__`` unsubscribed modules while ``finished`` was still queued behind a slow handler."""
    import time

    from panobbgo.core import Analyzer
    from panobbgo.heuristics import Random
    from panobbgo.strategies import StrategyRoundRobin

    got = []

    class Slow(Analyzer):
        def on_finished(self):
            time.sleep(0.3)
            got.append(self.name)

    s = StrategyRoundRobin(Rosenbrock(dim=2), parse_args=False, testing_mode=True, seed=0, max_eval=5)
    s.add_analyzer(Slow(s, name="Slow1"))
    s.add_analyzer(Slow(s, name="Slow2"))
    s.add(Random)
    s.start()
    assert got == ["Slow1", "Slow2"]


def test_unknown_strategy_kwarg_is_a_type_error():
    """``max_evals=`` (typo) used to be dropped silently."""
    from panobbgo.strategies import StrategyRoundRobin, StrategyUCB

    with pytest.raises(TypeError, match="max_evals"):
        StrategyRoundRobin(Rosenbrock(dim=2), parse_args=False, testing_mode=True, max_evals=10)
    s = StrategyRoundRobin(Rosenbrock(dim=2), parse_args=False, testing_mode=True, max_eval=10, rho=5.0)
    assert s.config.max_eval == 10 and s.config.rho == 5.0
    # ucb_c was one of the silently ignored ones.
    assert StrategyUCB(Rosenbrock(dim=2), parse_args=False, testing_mode=True, ucb_c=0.2).ucb_c == 0.2


def test_heuristic_subprocess_unsubscribes_before_closing_its_pipe(strategy):
    from panobbgo.core import HeuristicSubprocess

    h = HeuristicSubprocess(strategy)
    strategy.eventbus.register(h)
    subscribed_at_close = []
    real_close = h.pipe.close

    def close():
        subscribed_at_close.append(strategy.eventbus.is_subscribed(h))
        real_close()

    h.pipe.close = close
    h.__stop__()
    assert subscribed_at_close == [False]


def test_failing_initialize_still_stops_heuristic_subprocesses():
    from panobbgo.core import HeuristicSubprocess
    from panobbgo.strategies import StrategyRoundRobin

    s = StrategyRoundRobin(Rosenbrock(dim=2), parse_args=False, testing_mode=True, seed=0)
    s.add(HeuristicSubprocess)
    proc = s._hs[0]._HeuristicSubprocess__subprocess
    s.config.max_eval = -1  # validate_setup() raises inside initialize()
    with pytest.raises(ValueError, match="max_eval must be positive"):
        s.start()
    assert not proc.is_alive()


def test_unstarted_strategy_owns_no_eventbus_thread():
    """The dispatcher thread starts with the first event, not in ``EventBus.__init__``."""
    import threading

    from panobbgo.heuristics import Random
    from panobbgo.lib.classic import Rosenbrock
    from panobbgo.strategies import StrategyRoundRobin

    def buses():
        return {t for t in threading.enumerate() if t.name == "EventBus"}

    before = buses()
    s = StrategyRoundRobin(Rosenbrock(dim=2), parse_args=False, testing_mode=True, seed=0)
    s.add(Random)
    assert buses() == before
    assert s.eventbus._thread is None
    s.config.max_eval = 5
    s.start()
    assert s.eventbus._thread is not None and not s.eventbus._thread.is_alive()


def test_rank_tracker_matches_sorted_order():
    from panobbgo.core import _RankTracker

    rng = np.random.default_rng(3)
    t = _RankTracker()
    seen = []
    for v in rng.normal(size=500):
        t.push(float(v))
        seen.append(float(v))
        s = sorted(seen)
        assert t.n == len(s) and t.min == s[0]
        assert t.threshold() == s[int(len(s) * 0.1)]
    t.push(float("nan"))
    assert t.n == 500


def _progress_contexts(batches, toggle_at=None):
    """Run batches through ``Results`` with the reporter on; return the contexts."""
    s = StrategyBase(Rosenbrock(dim=2), parse_args=False)
    rep = s.panobbgo_logger.progress_reporter
    rep.enabled = toggle_at is None
    ctxs = []
    rep.report_evaluation = lambda result, context=None: ctxs.append(context)
    r = Results(s)
    for i, fxs in enumerate(batches):
        if i == toggle_at:
            rep.enabled = True
        r.add_results([Result(Point(np.zeros(2), "t"), fx) for fx in fxs])
    return ctxs


def test_progress_context_is_relative_to_history_before_the_batch():
    rng = np.random.default_rng(4)
    batches = [list(rng.exponential(size=int(k))) for k in rng.integers(1, 5, size=60)]
    ctxs = _progress_contexts(batches)
    prior: list = []
    got = iter(ctxs)
    for b in batches:
        srt = sorted(prior)
        for fx in b:
            c = next(got)
            best = bool(srt) and fx < srt[0]
            assert c.is_global_best == best
            assert c.is_significant_improvement == (len(srt) > 10 and fx < srt[int(len(srt) * 0.1)])
        prior.extend(b)


def test_progress_reporter_off_skips_stats_and_rebuilds_when_enabled_later():
    batches = [[5.0], [4.0], [3.0], [6.0], [2.5]]
    ctxs = _progress_contexts(batches, toggle_at=3)
    # Only the batches after the toggle are reported, against the full history.
    assert [c.is_global_best for c in ctxs] == [False, True]


def test_avg_time_per_task_is_a_running_mean_defined_from_one_task():
    s = StrategyBase(Rosenbrock(dim=2), parse_args=False)
    assert np.isnan(s.avg_time_per_task)
    s.record_walltime(0.5)
    assert s.avg_time_per_task == 0.5  # was NaN below two tasks
    s.record_walltime(1.5)
    assert s.avg_time_per_task == 1.0


def _installed_analyzers(*heuristics, analyzers=()):
    """Initialize a seeded strategy; return its analyzer names and the next master draw."""
    from panobbgo.strategies import StrategyRoundRobin

    s = StrategyRoundRobin(Rosenbrock(dim=2), parse_args=False, testing_mode=True, seed=11)
    for a in analyzers:
        s.add_analyzer(a(s))
    for h in heuristics:
        s.add_heuristic(h(s))
    try:
        s.initialize()
        return sorted(a.name for a in s.analyzers), int(s.rng.integers(2**62))
    finally:
        s._cleanup()


def test_splitter_is_installed_only_on_demand_and_grid_never():
    from panobbgo.analyzers import Archive
    from panobbgo.heuristics import CMAES, Nearby, NelderMead, Random, RegionUCB

    names, draw = _installed_analyzers(CMAES, Nearby)
    assert names == ["Best", "Convergence"]
    for consumer in (Random, NelderMead, RegionUCB):
        names_c, draw_c = _installed_analyzers(consumer, Nearby)
        assert names_c == ["Best", "Convergence", "Splitter"], consumer
        # Every skipped default slot still consumes its seed: the master
        # stream after initialize() does not depend on what was installed.
        assert draw_c == draw
    # A warm start reads Splitter leaves / root (Heuristic.archive_seed).
    names_w, draw_w = _installed_analyzers(lambda s: CMAES(s, warm_start="archive"), analyzers=[Archive])
    assert names_w == ["Archive", "Best", "Convergence", "Splitter"]


def test_missing_analyzer_lookup_names_the_declaration():
    from panobbgo.heuristics import CMAES
    from panobbgo.strategies import StrategyRoundRobin

    s = StrategyRoundRobin(Rosenbrock(dim=2), parse_args=False, testing_mode=True, seed=0)
    s.add_heuristic(CMAES(s))
    with pytest.raises(KeyError, match="requires_analyzers"):
        s.analyzer("Splitter")


def test_eventbus_shutdown_timeout_leaves_no_phantom_inflight():
    """Events dropped by a timed-out ``shutdown`` leave the in-flight count."""
    import threading

    from panobbgo.config import Config

    bus = EventBus(Config(parse_args=False, testing_mode=True))
    gate = threading.Event()

    class Slow:
        name = "slow"

        def on_ping(self):
            gate.wait(5)

    bus.register(Slow())
    for _ in range(3):
        bus.publish("ping")
    bus.shutdown(timeout=0.05)  # the first handler is still blocked; two are dropped
    gate.set()
    assert bus.wait_idle(timeout=5)
    assert bus.inflight == 0


def test_results_setter_resets_derived_state():
    """Replacing the frame drops the progress-rank cache built for the old one."""
    s = StrategyBase(Rosenbrock(dim=2), parse_args=False)
    s.panobbgo_logger.progress_reporter.enabled = True
    s.panobbgo_logger.progress_reporter.report_evaluation = lambda *a, **k: None
    r = Results(s)
    r.add_results([Result(Point(np.zeros(2), "t"), float(v)) for v in range(5)])
    r.add_results([Result(Point(np.zeros(2), "t"), 10.0)])
    assert r._progress_ranks is not None
    old = r.results
    r.results = old.iloc[:2]
    assert r._progress_ranks is None and r._last_nb == 2
    assert r._progress_stats()["current_best_fx"] == 0.0


def test_module_requires_a_seeded_strategy_rng():
    """No unseeded fallback: a stand-in without ``spawn_rng`` is an error."""
    from types import SimpleNamespace

    from panobbgo.config import Config
    from tests.support import attach_spawn_rng

    double = SimpleNamespace(config=Config(parse_args=False, testing_mode=True))
    with pytest.raises(TypeError, match="spawn_rng"):
        Module(double)
    a = Module(attach_spawn_rng(double, seed=3)).rng.random()
    b = Module(attach_spawn_rng(double, seed=3)).rng.random()
    assert a == b


@pytest.mark.parametrize(
    "max_eval, known",
    [(200, 200.0), ("150", 150.0), (0, None), (-5, None), (None, None), ("abc", None), (float("inf"), None)],
)
def test_known_budget(max_eval, known):
    from panobbgo.core import known_budget

    assert known_budget(max_eval) == known


def test_budget_progress_and_max_eval_or_on_module_and_strategy():
    """One budget parser for modules and strategies (the ad-hoc copies it replaced)."""
    strategy = StrategyBase(Rosenbrock(dim=2), parse_args=False, seed=1)
    m = Module(strategy)
    for owner in (m, strategy):
        strategy.config.max_eval = 100
        assert owner.max_eval_or(1000) == 100
        assert owner.budget_progress() == 0.0
        strategy.config.max_eval = None
        assert owner.max_eval_or(1000) == 1000
        assert owner.max_eval_or(0) == 0
        assert owner.budget_progress() is None
        strategy.config.max_eval = 0
        assert owner.budget_progress() is None
    strategy.config.max_eval = 4
    strategy.results.add_results([Result(Point(np.zeros(2), "t"), float(i)) for i in range(6)])
    assert m.budget_progress() == strategy.budget_progress() == 1.0  # clipped


def test_default_analyzers_do_not_depend_on_the_package_namespace(monkeypatch):
    """Regression: the slot loop resolved ``getattr(panobbgo.analyzers, name, None)``.

    A class renamed or dropped from ``panobbgo.analyzers`` then silently built
    no default analyzer at all.
    """
    import panobbgo.analyzers
    from panobbgo.heuristics import Random
    from panobbgo.strategies import StrategyRoundRobin

    monkeypatch.delattr(panobbgo.analyzers, "Best")
    monkeypatch.delattr(panobbgo.analyzers, "Convergence")
    s = StrategyRoundRobin(Rosenbrock(dim=2), parse_args=False, seed=1)
    s.config.max_eval = 4
    s.add_heuristic(Random(s))
    try:
        s.initialize()
        assert {"Best", "Convergence"} <= set(s._analyzers)
    finally:
        s._cleanup()
