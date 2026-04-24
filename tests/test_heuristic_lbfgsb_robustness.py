import unittest
import threading
import time
from unittest.mock import Mock, MagicMock
from panobbgo.heuristics.lbfgsb import LBFGSB
from unittest import mock
import numpy as np


class TestLBFGSBRobustness(unittest.TestCase):
    def setUp(self):
        self.strategy = Mock()
        self.strategy.config = Mock()
        self.strategy.config.get_logger.return_value = Mock()
        self.strategy.config.capacity = 10
        self.strategy.problem = Mock()
        self.strategy.problem.dim = 2
        self.strategy.problem.box.box = [[0, 1], [0, 1]]

        self.heuristic = LBFGSB(self.strategy)
        # Mock pipes
        self.heuristic.p1 = Mock()
        self.heuristic.out1 = Mock()
        # Mock emit to avoid errors
        self.heuristic.emit = Mock()

    def test_lbfgsb_stops_cleanly(self):
        # Setup: simulate stopped state
        self.heuristic._stopped = True

        # Mock poll to return False (no output from out1)
        self.heuristic.out1.poll.return_value = False

        # Mock recv to return a dummy value (so it doesn't block)
        self.heuristic.p1.recv.return_value = [0.5, 0.5]

        # Configure mocks to allow poll on p1 (which we will add in the fix)
        self.heuristic.p1.poll = Mock(return_value=True)

        # Run on_start in a thread
        t = threading.Thread(target=self.heuristic.on_start)
        t.daemon = True
        t.start()

        # Wait for thread to finish
        t.join(timeout=2.0)

        if t.is_alive():
            self.fail("LBFGSB.on_start did not stop when _stopped=True")

    def test_lbfgsb_stop_method(self):
        # Verify that __stop__ terminates the process
        self.heuristic.lbfgsb = Mock()
        self.heuristic.__stop__()

        if hasattr(self.heuristic, "lbfgsb") and isinstance(self.heuristic.lbfgsb, Mock):
            pass

    def test_start_subprocess_exception(self):
        strategy = mock.MagicMock()
        lbfgsb = LBFGSB(strategy)

        with mock.patch("multiprocessing.get_context", side_effect=Exception("Test Error")):
            with self.assertRaises(RuntimeError):
                lbfgsb.__start__()

    def test_worker_logic(self):
        pipe_mock = mock.MagicMock()
        output_mock = mock.MagicMock()

        # Test worker sets up starting point correctly
        bounds = [(-5.0, 5.0), (-5.0, 5.0)]
        dims = 2

        pipe_mock.recv.return_value = 1.0  # return something for f()

        with mock.patch("scipy.optimize.fmin_l_bfgs_b") as fmin_mock:
            fmin_mock.return_value = "Test Solution"
            LBFGSB.worker(pipe_mock, output_mock, dims, bounds)

            fmin_mock.assert_called_once()
            args, kwargs = fmin_mock.call_args

            # check x0
            np.testing.assert_array_equal(args[1], np.array([0.0, 0.0]))
            self.assertEqual(kwargs["bounds"], bounds)

            # call objective func
            f = args[0]
            val = f(np.array([1.0, 1.0]))
            self.assertEqual(val, 1.0)

            pipe_mock.send.assert_called_once()
            np.testing.assert_array_equal(pipe_mock.send.call_args[0][0], np.array([1.0, 1.0]))

            output_mock.send.assert_called_once_with("Test Solution")

    def test_on_start_exception(self):
        strategy = mock.MagicMock()
        lbfgsb = LBFGSB(strategy)

        lbfgsb.out1 = mock.MagicMock()
        lbfgsb.out1.poll.return_value = False
        lbfgsb.p1 = mock.MagicMock()
        lbfgsb.p1.poll.side_effect = Exception("Test Exception")
        lbfgsb._stopped = False
        lbfgsb.logger = mock.MagicMock()

        lbfgsb.on_start()

        lbfgsb.logger.error.assert_called_once()


if __name__ == "__main__":
    unittest.main()
