import os
import unittest
import unittest.mock as mock

from panobbgo.config import Config


def _no_yaml():
    """Patch context: pretend there is no ./config.yaml and never mkdir."""
    original_exists = os.path.exists

    def mock_exists(path):
        if str(path).endswith("config.yaml"):
            return False
        return original_exists(path)

    return mock.patch("os.path.exists", side_effect=mock_exists), mock.patch("os.mkdir")


class TestConfigInit(unittest.TestCase):
    def test_each_call_is_a_fresh_instance(self):
        c1 = Config(testing_mode=False)
        c2 = Config(testing_mode=False)
        self.assertIsNot(c1, c2)
        self.assertFalse(c1.testing_mode)

    def test_overrides_do_not_leak_between_instances(self):
        c1 = Config(testing_mode=True)
        c2 = Config(testing_mode=True)
        c1.max_eval = 7
        c1.capacity = 400
        self.assertNotEqual(c2.max_eval, 7)
        self.assertNotEqual(c2.capacity, 400)

    def test_testing_mode_changes_defaults(self):
        p1, p2 = _no_yaml()
        with p1, p2:
            c1 = Config(testing_mode=False)
            self.assertFalse(c1.testing_mode)
            self.assertEqual(c1.dask_dashboard_address, ":8787")

            c2 = Config(testing_mode=True)
            self.assertTrue(c2.testing_mode)
            self.assertEqual(c2.dask_dashboard_address, ":0")
            # the first instance is untouched by the second
            self.assertEqual(c1.dask_dashboard_address, ":8787")

    def test_loggers_are_not_duplicated_across_instances(self):
        import logging

        from panobbgo.utils import PanobbgoHandler

        for _ in range(3):
            Config(testing_mode=True).get_logger("CFGT")
        handlers = logging.getLogger("%-5s" % "CFGT").handlers
        self.assertEqual(len([h for h in handlers if isinstance(h, PanobbgoHandler)]), 1)


def test_strategies_own_their_config():
    from panobbgo.lib.classic import Rosenbrock
    from panobbgo.strategies import StrategyRoundRobin

    problem = Rosenbrock(dim=2)
    a = StrategyRoundRobin(problem, parse_args=False, testing_mode=True, max_eval=11)
    b = StrategyRoundRobin(problem, parse_args=False, testing_mode=True)
    a.config.capacity = 123
    assert a.config is not b.config
    assert b.config.max_eval != 11
    assert b.config.capacity != 123


if __name__ == "__main__":
    unittest.main()
