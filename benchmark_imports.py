import time
import numpy as np
from panobbgo.core import Results
from panobbgo.lib import Result, Point
from panobbgo.config import Config
from unittest.mock import MagicMock

def benchmark():
    class MockStrategy:
        def __init__(self):
            self.config = Config(testing_mode=True)
            self.problem = MagicMock()
            self.problem.project = lambda x: x
            self.eventbus = MagicMock()
            # self.panobbgo_logger = MagicMock() # Will be initialized in Results if not present? No.

    strategy = MockStrategy()
    # Results(strategy) expects strategy.problem, strategy.eventbus, strategy.config
    # and maybe panobbgo_logger for _report_evaluation_progress
    strategy.panobbgo_logger = MagicMock()

    results = Results(strategy)

    # Number of iterations
    n_iters = 1000
    n_points_per_iter = 10

    start_time = time.perf_counter()
    for i in range(n_iters):
        batch = []
        for j in range(n_points_per_iter):
            r = Result(Point(np.random.rand(2), "test"), np.random.rand(), cv_vec=np.array([0.0]))
            batch.append(r)
        results.add_results(batch)
    end_time = time.perf_counter()

    print(f"Time taken for {n_iters} batches of {n_points_per_iter} points: {end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    benchmark()
