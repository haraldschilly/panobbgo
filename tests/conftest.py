import pytest
from dask.distributed import LocalCluster, Client

@pytest.fixture
def dask_cluster():
    """
    Fixture that provides an isolated Dask cluster for testing.
    Ensures cleanup of workers to prevent memory leaks across tests.
    """
    # Start a clean cluster
    # Use different dashboard port to avoid conflicts
    cluster = LocalCluster(
        n_workers=2,
        threads_per_worker=1,
        dashboard_address=":0",
        silence_logs=True
    )

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
    from panobbgo.config import Config
    from panobbgo.core import EventBus

    with mock.patch("panobbgo.core.StrategyBase") as MockStrategy:
        strategy_mock = MockStrategy.return_value
        Config._instance = None
        strategy_mock.config = Config(parse_args=False, testing_mode=True)
        strategy_mock.config.ui_show = False
        strategy_mock.eventbus = EventBus(strategy_mock.config)
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
        from panobbgo.config import Config
        Config._instance = None
        kwargs.setdefault('parse_args', False)
        kwargs.setdefault('testing_mode', True)

        strategy = strategy_class(problem, **kwargs)
        strategy.config.ui_show = False
        strategy.config.evaluation_method = "threaded"
        strategies_to_cleanup.append(strategy)
        return strategy

    yield _strategy_factory

    for strategy in strategies_to_cleanup:
        strategy._cleanup()
