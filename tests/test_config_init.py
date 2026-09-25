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


def test_parsed_sources_are_cached_per_file_version_and_copied(tmp_path, monkeypatch):
    from panobbgo import config as cfg

    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text("core:\n  max_eval: 77\nlogging:\n  level: 10\n")
    c1, c2 = Config(testing_mode=True), Config(testing_mode=True)
    assert c1.max_eval == c2.max_eval == 77
    c1.yaml_config["core"]["max_eval"] = 1
    c1.logging["level"] = 99
    assert c2.yaml_config["core"]["max_eval"] == 77  # a private copy each
    assert c2.logging["level"] == 10
    assert Config(testing_mode=True).max_eval == 77

    hits = cfg._parse_yaml.cache_info().hits
    Config(testing_mode=True)
    assert cfg._parse_yaml.cache_info().hits == hits + 1

    # A new file version (size/mtime) is re-read.
    (tmp_path / "config.yaml").write_text("core:\n  max_eval: 1234\n")
    os.utime(tmp_path / "config.yaml", ns=(1, 1))
    assert Config(testing_mode=True).max_eval == 1234


def test_default_ini_is_written_atomically_into_a_new_directory(tmp_path):
    from configparser import ConfigParser

    from panobbgo.config import _write_default_ini

    path = tmp_path / "a" / "b" / "config.ini"
    _write_default_ini(str(path))
    _write_default_ini(str(path))  # an existing file / directory is fine
    cfgp = ConfigParser()
    cfgp.read(path)
    assert cfgp.getint("core", "max_eval") == 1000
    assert sorted(p.name for p in path.parent.iterdir()) == ["config.ini"]  # no temp left


if __name__ == "__main__":
    unittest.main()
