import unittest
import threading
import time
from unittest.mock import Mock, MagicMock
from panobbgo.heuristics.nelder_mead import NelderMead


class TestNelderMeadRobustness(unittest.TestCase):
    def setUp(self):
        self.strategy = Mock()
        self.strategy.config = Mock()
        self.strategy.config.get_logger.return_value = Mock()
        self.strategy.config.capacity = 10
        self.strategy.problem = Mock()
        self.strategy.problem.dim = 2

        self.heuristic = NelderMead(self.strategy)
        # Mock got_bb
        self.heuristic.got_bb = Mock()
        self.heuristic.got_bb.wait = Mock()
        self.heuristic.got_bb.is_set = Mock(return_value=False)
        self.heuristic.got_bb.clear = Mock()

        # Mock emit
        self.heuristic.emit = Mock()

        # Mock best_box to prevent attribute errors if logic accesses it
        self.heuristic.best_box = Mock()
        self.heuristic.best_box.results = []
        self.heuristic.best_box.parent = None

    def test_nelder_mead_stops_cleanly(self):
        # Setup: simulate stopped state
        self.heuristic._stopped = True

        # Mock wait to return immediately (so loop executes)
        self.heuristic.got_bb.wait.return_value = True

        # Run on_start in a thread
        t = threading.Thread(target=self.heuristic.on_start)
        t.daemon = True
        t.start()

        # Wait for thread to finish
        t.join(timeout=2.0)

        if t.is_alive():
            # If still alive, it means the infinite loop didn't break
            # We assume it's stuck in while True loop because _stopped wasn't checked.
            # But in the test setup, we can't easily interrupt the thread.
            pass

        # Assert thread is dead
        self.assertFalse(t.is_alive(), "NelderMead.on_start did not stop when _stopped=True")


if __name__ == "__main__":
    unittest.main()
