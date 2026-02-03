
import time
import numpy as np
import pandas as pd
from panobbgo.core import Results, StrategyBase
from panobbgo.lib import Result, Point, Problem
from panobbgo.config import Config

class MockStrategy:
    def __init__(self):
        self.config = Config(testing_mode=True)
        self.problem = None
        self.eventbus = MockEventBus()
        self.panobbgo_logger = MockLogger()
        self.results = None

class MockEventBus:
    def publish(self, key, **kwargs):
        pass

class MockLogger:
    def __init__(self):
        self.progress_reporter = MockProgressReporter()

class MockProgressReporter:
    def report_evaluation(self, result, context):
        pass

def run_benchmark():
    strategy = MockStrategy()
    results = Results(strategy)

    # Pre-populate with some results to make the dataframe grow
    # We want to simulate a scenario where `add_results` is called repeatedly or with a large list
    # But `add_results` appends to existing results.

    n_existing = 1000
    existing = []
    for i in range(n_existing):
        r = Result(Point(np.array([float(i)]), "init"), float(i), cv_vec=np.array([0.0]))
        existing.append(r)

    results.add_results(existing)

    # Now benchmark adding new results one by one or in small batches,
    # which triggers the inefficient check repeatedly

    n_new = 500
    new_results = []
    for i in range(n_new):
        # descending fx to trigger updates? Or random?
        # If result.fx < current_best_fx, it does extra work (logging context)
        # But the cost is mainly in finding current_best_fx
        fx = -float(i) # improving
        r = Result(Point(np.array([float(i)]), "new"), fx, cv_vec=np.array([0.0]))
        new_results.append(r)

    start_time = time.time()

    # Adding one by one to maximize the impact of the loop inside add_results
    # if we were calling add_results repeatedly.
    # But add_results takes a list.
    # The loop is inside add_results: `for result in new_results: self._report_evaluation_progress(result)`

    results.add_results(new_results)

    end_time = time.time()

    print(f"Time to add {n_new} results with {n_existing} existing results: {end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    run_benchmark()
