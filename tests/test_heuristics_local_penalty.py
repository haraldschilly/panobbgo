# -*- coding: utf8 -*-
from __future__ import unicode_literals

import numpy as np
import time
import pytest
from panobbgo.utils import PanobbgoTestCase
from panobbgo.lib import Point, Result, BoundingBox, Problem

class MockConstraintHandler:
    def get_penalty_value(self, result):
        if result is None or result.fx is None:
            return float('inf')
        cv = result.cv if result.cv is not None else 0.0
        return result.fx + 100.0 * cv

class LocalPenaltySearchTest(PanobbgoTestCase):

    def setUp(self):
        super().setUp()
        self.strategy.constraint_handler = MockConstraintHandler()
        self.strategy.name = "MockStrategy"
        # Increase queue capacity for testing
        self.strategy.config.capacity = 100

    def test_initialization(self):
        from panobbgo.heuristics.local_penalty_search import LocalPenaltySearch
        h = LocalPenaltySearch(self.strategy)
        assert h is not None
        assert h.name == "LocalPenaltySearch"

    def test_start_stop(self):
        from panobbgo.heuristics.local_penalty_search import LocalPenaltySearch
        h = LocalPenaltySearch(self.strategy)
        h.__start__()
        time.sleep(0.5)
        # Process should be alive
        assert h.process is not None
        assert h.process.is_alive()

        h.__stop__()
        time.sleep(0.5)
        assert not h.process.is_alive()

    def test_optimization_flow(self):
        from panobbgo.heuristics.local_penalty_search import LocalPenaltySearch
        import threading

        h = LocalPenaltySearch(self.strategy)
        h.__start__()

        # Start the on_start loop in a thread
        t = threading.Thread(target=h.on_start)
        t.daemon = True
        t.start()

        # Wait for initial point
        start_time = time.time()
        points = []
        while time.time() - start_time < 10.0:
            new_pts = h.get_points()
            if new_pts:
                points.extend(new_pts)
                break
            time.sleep(0.1)

        assert len(points) > 0, "Heuristic did not emit initial point"
        p1 = points[0]

        # Provide result for p1
        # Simple objective: x^2
        fx1 = np.sum(p1.x**2)
        r1 = Result(p1, fx1, cv_vec=None)

        # Feed result back. This should run in a thread or be called directly.
        # Since on_new_results uses a lock and writes to pipe, it should be safe to call directly
        # from main thread while on_start is blocked/polling in another thread.
        h.on_new_results([r1])

        # Expect another point (optimization step)
        start_time = time.time()
        points2 = []
        while time.time() - start_time < 10.0:
            new_pts = h.get_points()
            if new_pts:
                points2.extend(new_pts)
                break
            time.sleep(0.1)

        assert len(points2) > 0, "Heuristic did not emit next point after result"

        # Cleanup
        h._stopped = True
        h.__stop__()
        t.join(timeout=2.0)
