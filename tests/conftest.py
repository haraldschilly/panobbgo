import os
from pathlib import Path

import pytest

# Resolved at import time, before any test changes HOME.
_REAL_HOME = Path.home()


def pytest_configure(config):
    """Pin BLAS to one thread for the whole test process, as every harness run pins it.

    The first harness run would pin it anyway (``local_run.pin_blas``, never
    lifted); pinning up front makes every test see the same BLAS whatever
    ran before it, and keeps OpenBLAS's thread pool out of the tests: threads
    abandoned by ``evaluation.timeout`` tests keep calling numpy while later
    tests run.  Only the loaded libraries are limited, not the environment,
    so child processes still show whether their own pinning works.
    """
    from panobbgo.local_run import pin_blas

    pin_blas()


@pytest.fixture(autouse=True)
def _hermetic_home_and_cwd(monkeypatch, tmp_path_factory):
    """Run every test with a private ``HOME`` and working directory.

    :class:`panobbgo.config.Config` reads (and, if missing, writes)
    ``~/.panobbgo/config.ini`` and reads ``./config.yaml``.  Without this a
    developer's own settings (say ``[storage] backend = sqlite``) or the
    checked-in ``config.yaml`` would change what the tests run, and every run
    would write into the real home directory.  The repository's
    ``config.yaml`` only restates the built-in defaults, so the tests see the
    same configuration as before.

    Tools that tests start in subprocesses (``uv`` for the IOH worker,
    matplotlib) keep using the real cache and data directories, so a private
    ``HOME`` does not force a fresh download or font-cache rebuild.
    Module-level code that builds a ``Config`` at import (collection) time
    is not covered.
    """
    for var, default in (
        ("XDG_CACHE_HOME", _REAL_HOME / ".cache"),
        ("XDG_DATA_HOME", _REAL_HOME / ".local" / "share"),
    ):
        monkeypatch.setenv(var, os.environ.get(var, str(default)))
    home = tmp_path_factory.mktemp("home")
    # As on a machine where panobbgo has run before (some tests patch os.mkdir).
    (home / ".panobbgo").mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path_factory.mktemp("cwd"))


@pytest.fixture
def dask_cluster():
    """
    Fixture that provides an isolated Dask cluster for testing.
    Ensures cleanup of workers to prevent memory leaks across tests.
    Skips the test when the optional dask extra is not installed.
    """
    distributed = pytest.importorskip("dask.distributed", reason="optional dask extra not installed")
    LocalCluster, Client = distributed.LocalCluster, distributed.Client

    # Start a clean cluster
    # Use different dashboard port to avoid conflicts
    cluster = LocalCluster(n_workers=2, threads_per_worker=1, dashboard_address=":0", silence_logs=True)

    # Optional client if tests need it directly
    client = Client(cluster)

    # Yield the cluster for the test to use
    yield cluster

    # Tear down
    client.close()
    cluster.close()

    # Ensure all workers are actually terminated
    # Dask cleanup can sometimes be asynchronous/lazy
    import time

    time.sleep(0.5)


@pytest.fixture
def strategy():
    """
    Fixture that provides a mocked StrategyBase for testing heuristics and analyzers.
    Avoids resource leaks by mocking the core background threads.
    """
    from unittest import mock
    import numpy as np
    from panobbgo.config import Config
    from panobbgo.core import EventBus

    with mock.patch("panobbgo.core.StrategyBase") as MockStrategy:
        strategy_mock = MockStrategy.return_value
        strategy_mock.config = Config(parse_args=False, testing_mode=True)
        strategy_mock.config.ui_show = False
        strategy_mock.eventbus = EventBus(strategy_mock.config)
        # Modules derive their RNG from the strategy (see Module.__init__).
        strategy_mock.seed = 0
        strategy_mock.rng = np.random.default_rng(0)
        strategy_mock.spawn_rng.side_effect = lambda key="": np.random.default_rng(0)
        yield strategy_mock


@pytest.fixture
def real_strategy():
    """
    Fixture that yields a real Strategy instance factory.
    It takes care of calling `_cleanup()` to prevent hanging background threads/processes.
    Usage:
        def test_my_strategy(real_strategy):
            strategy = real_strategy(StrategyRewarding, problem, max_evaluations=10)
            strategy.start()
    """
    strategies_to_cleanup = []

    def _strategy_factory(strategy_class, problem, **kwargs):
        kwargs.setdefault("parse_args", False)
        kwargs.setdefault("testing_mode", True)

        strategy = strategy_class(problem, **kwargs)
        strategy.config.ui_show = False
        strategy.config.evaluation_method = "threaded"
        strategies_to_cleanup.append(strategy)
        return strategy

    yield _strategy_factory

    for strategy in strategies_to_cleanup:
        strategy._cleanup()
