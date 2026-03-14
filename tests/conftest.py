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
