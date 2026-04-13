# -*- coding: utf8 -*-
import pytest
from unittest import mock
import numpy as np
from panobbgo.utils import PanobbgoTestCase
from panobbgo.heuristics.lbfgsb import LBFGSB

class TestHeuristicLBFGSB(PanobbgoTestCase):
    @mock.patch("panobbgo.core.StrategyBase._setup_cluster")
    def test_start_and_stop(self, mock_setup):
        lbfgsb = LBFGSB(self.strategy)

        # Test __start__
        lbfgsb.__start__()
        assert lbfgsb.lbfgsb is not None
        assert lbfgsb.lbfgsb.is_alive()

        # Test __stop__ graceful
        lbfgsb.__stop__()
        assert not lbfgsb.lbfgsb.is_alive()

    @mock.patch("panobbgo.core.StrategyBase._setup_cluster")
    def test_stop_force_kill(self, mock_setup):
        lbfgsb = LBFGSB(self.strategy)
        lbfgsb.__start__()

        # Mock is_alive to return True even after join to force kill()
        with mock.patch.object(lbfgsb.lbfgsb, 'is_alive', side_effect=[True, True, True, False]):
            with mock.patch.object(lbfgsb.lbfgsb, 'kill') as mock_kill:
                lbfgsb.__stop__()
                mock_kill.assert_called_once()

    @mock.patch("panobbgo.core.StrategyBase._setup_cluster")
    def test_on_start_loop(self, mock_setup):
        lbfgsb = LBFGSB(self.strategy)
        lbfgsb.__start__()

        # We need to simulate the worker sending a message and then stopping the loop
        lbfgsb.p1.send(np.array([1.0, 2.0]))

        # Make the loop run once and then stop
        def fake_poll(timeout):
            lbfgsb._stopped = True
            return True

        with mock.patch.object(lbfgsb.p1, 'poll', side_effect=fake_poll):
            lbfgsb.on_start()

        lbfgsb.__stop__()

class TestHeuristicLBFGSBRobustness(PanobbgoTestCase):
    @mock.patch("panobbgo.core.StrategyBase._setup_cluster")
    def test_start_and_stop(self, mock_setup):
        lbfgsb = LBFGSB(self.strategy)

        # Test __start__
        lbfgsb.__start__()
        assert lbfgsb.lbfgsb is not None
        assert lbfgsb.lbfgsb.is_alive()

        # Test __stop__ graceful
        lbfgsb.__stop__()
        assert not lbfgsb.lbfgsb.is_alive()

    @mock.patch("panobbgo.core.StrategyBase._setup_cluster")
    def test_stop_force_kill(self, mock_setup):
        lbfgsb = LBFGSB(self.strategy)
        lbfgsb.__start__()

        # Mock is_alive to return True even after join to force kill()
        with mock.patch.object(lbfgsb.lbfgsb, 'is_alive', side_effect=[True, True, True, False]):
            with mock.patch.object(lbfgsb.lbfgsb, 'kill') as mock_kill:
                lbfgsb.__stop__()
                mock_kill.assert_called_once()

    @mock.patch("panobbgo.core.StrategyBase._setup_cluster")
    def test_on_start_loop(self, mock_setup):
        lbfgsb = LBFGSB(self.strategy)
        lbfgsb.__start__()

        # We need to simulate the worker sending a message and then stopping the loop
        lbfgsb.p1.send(np.array([1.0, 2.0]))

        # Make the loop run once and then stop
        def fake_poll(timeout):
            lbfgsb._stopped = True
            return True

        with mock.patch.object(lbfgsb.p1, 'poll', side_effect=fake_poll):
            lbfgsb.on_start()

        lbfgsb.__stop__()

class TestHeuristicLBFGSBExtra(PanobbgoTestCase):
    def test_on_new_results(self):
        lbfgsb = LBFGSB(self.strategy)
        lbfgsb.p1 = mock.MagicMock()

        class ResultMock:
            def __init__(self, who, fx):
                self.who = who
                self.fx = fx

        results = [ResultMock("LBFGSB", 5.0), ResultMock("Other", 3.0)]
        self.strategy.constraint_handler.get_penalty_value = lambda r: r.fx

        lbfgsb.on_new_results(results)

        lbfgsb.p1.send.assert_called_once_with(5.0)

    def test_on_start_pipe_closed(self):
        lbfgsb = LBFGSB(self.strategy)
        lbfgsb.out1 = mock.MagicMock()
        lbfgsb.out1.poll.return_value = False
        lbfgsb.p1 = mock.MagicMock()
        lbfgsb.p1.poll.side_effect = EOFError()
        lbfgsb._stopped = False

        lbfgsb.on_start()
        # Should exit loop silently without error

    def test_on_start_output_received(self):
        lbfgsb = LBFGSB(self.strategy)
        lbfgsb.out1 = mock.MagicMock()
        lbfgsb.out1.poll.return_value = True
        lbfgsb.out1.recv.return_value = "Test Output"
        lbfgsb.p1 = mock.MagicMock()
        lbfgsb.p1.poll.side_effect = EOFError() # exit loop after first pass
        lbfgsb._stopped = False
        lbfgsb.logger = mock.MagicMock()

        lbfgsb.on_start()

        lbfgsb.logger.info.assert_called_with("Test Output")
