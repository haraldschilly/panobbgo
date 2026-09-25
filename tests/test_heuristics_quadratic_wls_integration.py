# -*- coding: utf8 -*-
import time
import unittest

import numpy as np
from panobbgo.core import StrategyBase, Result, Point
from panobbgo.heuristics import Random
from panobbgo.lib import Problem
from panobbgo.heuristics.quadratic_wls import QuadraticWlsModel


class MockProblem(Problem):
    def __init__(self, dim=2):
        box_list = [(-5.0, 5.0) for _ in range(dim)]
        super().__init__(box=box_list)

    def eval(self, x):
        return np.sum(x**2)

    def __call__(self, point):
        x = point.x
        return Result(point, self.eval(x), cv_vec=None)


class MockStrategy(StrategyBase):
    def __init__(self):
        problem = MockProblem()
        super().__init__(problem, max_eval=100)
        # self.config.loglevel = 10 # Cannot set config directly in init easily without proper config setup

    def execute(self):
        return []


#: Seconds the slow worker sleeps before each fit.
SLOW_FIT_DELAY = 0.5


class _SlowPipe:
    """Worker-side pipe wrapper: every request takes :data:`SLOW_FIT_DELAY` longer."""

    def __init__(self, pipe):
        self._pipe = pipe

    def poll(self, timeout=None):
        return self._pipe.poll(timeout)

    def recv(self):
        payload = self._pipe.recv()
        time.sleep(SLOW_FIT_DELAY)
        return payload

    def send(self, obj):
        self._pipe.send(obj)


class SlowQuadraticWlsModel(QuadraticWlsModel):
    """The real model, with a worker that is slow to answer."""

    @staticmethod
    def subprocess(pipe):
        QuadraticWlsModel.subprocess(_SlowPipe(pipe))


class MockBestBox:
    def __init__(self, results, best):
        self.results = results
        self.best = best


class TestQuadraticWlsIntegration(unittest.TestCase):
    def test_wls_subprocess_communication(self):
        """
        Test that QuadraticWlsModel subprocess correctly processes data and returns points.
        This verifies that the subprocess loop works and doesn't crash due to scope issues.
        """
        strategy = MockStrategy()
        wls = QuadraticWlsModel(strategy)
        wls.__start__()

        # Create dummy results to simulate a "best box"
        results = []
        dim = 2
        np.random.seed(42)  # Deterministic
        for i in range(20):  # Need enough points for WLS
            x = np.random.uniform(-5, 5, dim)
            p = Point(x, "init")
            r = Result(p, np.sum(x**2), cv_vec=None)
            results.append(r)

        best_res = min(results, key=lambda r: r.fx)
        best_box = MockBestBox(results, best_res)

        # The event handler only records the box; the pull sends it to the
        # worker and (under sync evaluation) waits for the fit.
        strategy.config.sync_evaluation = True
        wls.on_new_best_box(best_box)
        points = wls.produce(10)

        # Assert we got at least one point
        self.assertTrue(len(points) > 0, "QuadraticWlsModel failed to generate points")

        # Check point validity
        for p in points:
            self.assertTrue(isinstance(p, Point))
            # Use 'in' operator instead of .contains()
            self.assertTrue(p in strategy.problem.box)

        # Cleanup
        wls.__stop__()


def _box_of_random_results(n=20, dim=2, seed=42):
    rng = np.random.default_rng(seed)
    results = [Result(Point(x, "init"), float(np.sum(x**2)), cv_vec=None) for x in rng.uniform(-5, 5, size=(n, dim))]
    return MockBestBox(results, min(results, key=lambda r: r.fx))


class TestQuadraticWlsPull(unittest.TestCase):
    def test_event_handler_is_not_blocked_by_a_slow_fit(self):
        """The event-bus handler returns at once; only the main thread's pull waits for the fit."""
        strategy = MockStrategy()
        strategy.config.sync_evaluation = True
        wls = SlowQuadraticWlsModel(strategy)
        wls.__start__()
        try:
            t0 = time.monotonic()
            wls.on_new_best_box(_box_of_random_results())
            wls.on_new_best_box(_box_of_random_results(seed=7))
            handler_time = time.monotonic() - t0
            self.assertLess(handler_time, SLOW_FIT_DELAY / 5)

            t0 = time.monotonic()
            points = wls.produce(10)
            pull_time = time.monotonic() - t0
            self.assertEqual(len(points), 1)  # the two boxes coalesced into one fit
            self.assertGreaterEqual(pull_time, SLOW_FIT_DELAY)  # sync: waited for it, no timeout
        finally:
            wls.__stop__()

    def test_async_pull_does_not_wait_for_a_slow_fit(self):
        strategy = MockStrategy()
        strategy.config.sync_evaluation = False
        wls = SlowQuadraticWlsModel(strategy)
        wls.__start__()
        try:
            wls.on_new_best_box(_box_of_random_results())
            t0 = time.monotonic()
            self.assertEqual(wls.produce(10), [])
            self.assertLess(time.monotonic() - t0, SLOW_FIT_DELAY / 5)
            self.assertTrue(wls.can_produce)  # the fit is on its way

            deadline = time.monotonic() + 30.0
            points = []
            while not points and time.monotonic() < deadline:
                time.sleep(0.05)
                points = wls.produce(10)
            self.assertEqual(len(points), 1)
        finally:
            wls.__stop__()


def _sync_run(wls_cls, seed=3, max_eval=60):
    """``(x, fx, who)`` of a seeded, synchronous round-robin run of ``Random`` + ``wls_cls``."""
    from panobbgo.strategies import StrategyRoundRobin

    strategy = StrategyRoundRobin(MockProblem(), parse_args=False, testing_mode=True, seed=seed)
    cfg = strategy.config
    cfg.max_eval = max_eval
    cfg.sync_evaluation = True
    cfg.stop_on_convergence = False
    cfg.ui_show = False
    cfg.evaluation_method = "threaded"
    strategy.add_heuristic(Random(strategy))
    strategy.add_heuristic(wls_cls(strategy))
    try:
        strategy.start()
    except Exception:
        strategy._cleanup()
        raise
    df = strategy.results.results
    assert df is not None
    x = np.asarray([np.asarray(v, dtype=float) for v in df["x"].to_numpy()])
    fx = df["fx"].to_numpy(dtype=float).ravel()
    who = tuple(str(w).replace("SlowQuadraticWlsModel", "QuadraticWlsModel") for w in df["who"].to_numpy().ravel())
    return x, fx, who


class TestQuadraticWlsReproducible(unittest.TestCase):
    def test_sync_runs_are_reproducible_and_independent_of_fit_speed(self):
        """Same seed twice gives the same trajectory, and so does a worker that is slow to fit."""
        x1, fx1, who1 = _sync_run(QuadraticWlsModel)
        x2, fx2, who2 = _sync_run(QuadraticWlsModel)
        self.assertGreater(sum(w == "QuadraticWlsModel" for w in who1), 0, "the model never contributed")
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(fx1, fx2)
        self.assertEqual(who1, who2)

        x3, fx3, who3 = _sync_run(SlowQuadraticWlsModel)
        np.testing.assert_array_equal(x1, x3)
        np.testing.assert_array_equal(fx1, fx3)
        self.assertEqual(who1, who3)
