import time
import numpy as np

# We'll just benchmark the time it takes to call the method repeatedly
# To avoid the full strategy setup, we'll initialize the heuristic directly

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from panobbgo.heuristics.gaussian_process import GaussianProcessHeuristic
from panobbgo.core import StrategyBase, Config
from panobbgo.lib.classic import Rosenbrock


def run_benchmark():
    # Setup mock
    config = Config(testing_mode=True)

    class MockStrategy:
        def __init__(self):
            self.config = config

    strategy = MockStrategy()

    # Initialize heuristic
    heuristic = GaussianProcessHeuristic(strategy=strategy)

    # Fake some data
    heuristic.X_train = np.random.rand(10, 5)
    heuristic.y_train = np.random.rand(10)

    # Warmup
    heuristic._fit_gp_model()

    start = time.time()
    for _ in range(100):
        heuristic._fit_gp_model()
    end = time.time()

    print(f"Time to fit GP model 100 times: {end - start:.4f}s")


if __name__ == "__main__":
    run_benchmark()
