# -*- coding: utf8 -*-
import pytest
from unittest import mock
import numpy as np
from panobbgo.utils import PanobbgoTestCase
from panobbgo.heuristics.nelder_mead import NelderMead
from panobbgo.lib import Result, Point

class TestHeuristicNelderMead(PanobbgoTestCase):
    @mock.patch("panobbgo.core.StrategyBase._setup_cluster")
    def test_on_start_loop_timeout(self, mock_setup):
        nm = NelderMead(self.strategy)

        # Test the loop exits properly when stopped
        def delayed_stop():
            import time
            time.sleep(0.2)
            nm._stopped = True

        import threading
        t = threading.Thread(target=delayed_stop)
        t.start()

        # Should exit loop after timeout when stopped
        nm.on_start()
        t.join()

    @mock.patch("panobbgo.core.StrategyBase._setup_cluster")
    def test_on_start_process_base(self, mock_setup):
        nm = NelderMead(self.strategy)

        # Create a mock best box
        mock_box = mock.MagicMock()

        # Create results for gram schmidt
        pts = self.random_results(self.problem.dim, 10)
        mock_box.results = pts
        mock_box.parent = None

        nm.best_box = mock_box

        # Trigger the wait
        nm.got_bb.set()

        # Make the loop run once and then stop
        def fake_emit(point):
            nm._stopped = True

        with mock.patch.object(nm, 'emit', side_effect=fake_emit):
            nm.on_start()

class TestHeuristicNelderMeadRobustness(PanobbgoTestCase):
    @mock.patch("panobbgo.core.StrategyBase._setup_cluster")
    def test_on_start_loop_timeout(self, mock_setup):
        nm = NelderMead(self.strategy)

        # Test the loop exits properly when stopped
        def delayed_stop():
            import time
            time.sleep(0.2)
            nm._stopped = True

        import threading
        t = threading.Thread(target=delayed_stop)
        t.start()

        # Should exit loop after timeout when stopped
        nm.on_start()
        t.join()

    @mock.patch("panobbgo.core.StrategyBase._setup_cluster")
    def test_on_start_process_base(self, mock_setup):
        nm = NelderMead(self.strategy)

        # Create a mock best box
        mock_box = mock.MagicMock()

        # Create results for gram schmidt
        pts = self.random_results(self.problem.dim, 10)
        mock_box.results = pts
        mock_box.parent = None

        nm.best_box = mock_box

        # Trigger the wait
        nm.got_bb.set()

        # Make the loop run once and then stop
        def fake_emit(point):
            nm._stopped = True

        with mock.patch.object(nm, 'emit', side_effect=fake_emit):
            nm.on_start()
