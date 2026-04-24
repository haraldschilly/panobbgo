# -*- coding: utf8 -*-
import unittest
import numpy as np
import time
from panobbgo.core import StrategyBase, Result, Point
from panobbgo.lib import Problem, BoundingBox
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

        # Trigger the heuristic
        # This sends data to the subprocess via pipe
        try:
            wls.on_new_best_box(best_box)
        except Exception as e:
            self.fail(f"on_new_best_box failed with exception: {e}")

        # Wait for points (with timeout)
        # Since on_new_best_box calls pipe.recv() internally (blocking),
        # if it returns, it means we got a response.
        points = wls.get_points(limit=10)

        # Assert we got at least one point
        self.assertTrue(len(points) > 0, "QuadraticWlsModel failed to generate points")

        # Check point validity
        for p in points:
            self.assertTrue(isinstance(p, Point))
            # Use 'in' operator instead of .contains()
            self.assertTrue(p in strategy.problem.box)

        # Cleanup
        wls.__stop__()
