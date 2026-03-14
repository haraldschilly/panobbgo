import psutil
from panobbgo.lib.classic import Rosenbrock
from panobbgo.core import StrategyBase
from dask.distributed import Client

class DummyStrategy(StrategyBase):
    def execute(self):
        return []

def get_dask_worker_processes():
    """Returns a list of all currently running dask worker processes."""
    dask_workers = []
    current_process = psutil.Process()
    # Find child processes that might be dask workers
    for p in current_process.children(recursive=True):
        try:
            cmdline = p.cmdline()
            # Dask workers usually have 'dask' or 'worker' in their command line
            if any('worker' in arg for arg in cmdline) or any('dask' in arg for arg in cmdline):
                dask_workers.append(p)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return dask_workers

def test_dask_cluster_isolation_and_cleanup(dask_cluster):
    """
    Test that the dask_cluster fixture properly sets up and tears down.
    Also ensures that no memory leaks or dangling processes remain.
    """
    import time

    # Setup the strategy with dask explicitly
    problem = Rosenbrock(dims=2)
    # create the dummy strategy
    strategy = DummyStrategy(problem, parse_args=False, testing_mode=True)
    strategy.config.evaluation_method = "dask"
    strategy.config.dask_dashboard_address = dask_cluster.dashboard_link

    # manually setup the cluster connection to the fixture
    strategy._cluster = dask_cluster
    strategy._client = Client(strategy._cluster)

    # Assert that dask workers are actually running
    workers_running = len(strategy._client.scheduler_info()['workers'])
    assert workers_running > 0, "Dask workers should be running"

    # We scatter something to ensure work has happened
    strategy._problem_future = strategy._client.scatter(problem, broadcast=True)

    # Now execute a simple evaluation
    from panobbgo.lib.lib import Point

    def evaluate(problem_instance, p):
        return problem_instance(p)

    pt = Point(problem.random_point(), who="test")
    future = strategy._client.submit(evaluate, strategy._problem_future, pt)
    result = future.result(timeout=5)

    assert result is not None

    # We call strategy cleanup to ensure it properly closes its references
    strategy._cleanup()

    # Now we let the fixture teardown (which will happen automatically after the test)
    # We can't test post-fixture cleanup here directly, but we can verify strategy._cleanup
    # closed the client connection it created.
    # Note: strategy._cleanup() calls self._client.close() and self._cluster.close().
    # This might close the fixture's cluster, which is fine (fixture will just close it again or do nothing).

    # Wait a bit for processes to die
    time.sleep(1)

    # Check that Dask worker processes were terminated
    dask_workers_after = get_dask_worker_processes()
    # There might be 0, or they might be zombies shutting down
    # But usually LocalCluster.close() terminates them fully.
    for p in dask_workers_after:
        assert not p.is_running() or p.status() == psutil.STATUS_ZOMBIE, "Worker process should be dead"

    # We do not assert exact memory because python's GC is non-deterministic
    # But we assert the test completes successfully without hanging
