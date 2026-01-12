
from panobbgo.analyzers.convergence import Convergence
from panobbgo.core import StrategyBase, EventBus
from panobbgo.lib import Result, Point
import numpy as np
import threading
import time

class MockProblem:
    def __init__(self, dim=2):
        self.dim = dim
        self.box = None  # Not needed for this test

    def __str__(self):
        return "MockProblem"

class MockConfig:
    def __init__(self):
        self.convergence_window_size = 5
        self.convergence_threshold = 0.1
        self.convergence_mode = 'std'
        self.stop_on_convergence = True
        self.debug = True
        self.rho = 100.0
        self.constraint_exponent = 1.0
        self.dynamic_penalty_rate = 0.01
        self.constraint_handler = "DefaultConstraintHandler"
        self.logging = {}
        self.max_eval = 1000
        self.evaluation_method = 'threaded'
        self.dask_n_workers = 1
        self.ui_show = False
        self.show_interval = 1.0
        self.discount = 0.95
        self.smooth = 0.5
        self.capacity = 20

    def get_logger(self, name):
        import logging
        return logging.getLogger(name)

class MockStrategy(StrategyBase):
    def __init__(self, problem):
        # We override __init__ to avoid full initialization but keep necessary parts
        self._name = "MockStrategy"
        self.config = MockConfig()
        self.logger = self.config.get_logger("STRAT")
        self.slogger = self.config.get_logger("STATS")
        self.problem = problem

        from panobbgo.lib.constraints import DefaultConstraintHandler
        self.constraint_handler = DefaultConstraintHandler(self, rho=100.0)

        self.eventbus = EventBus(self.config)
        self.results = None # We will mock results adding

        self._analyzers = {}
        self._heuristics = {}
        self._threads = []
        self.loops = 0
        self._stop_requested = False

        # We need to manually register on_converged
        self.eventbus.register(self)

    def execute(self):
        return []

def test_strategy_stops_on_convergence():
    problem = MockProblem()
    strategy = MockStrategy(problem)

    # Initialize analyzer
    analyzer = Convergence(strategy, window_size=5, threshold=0.1, mode='std')
    # strategy.add_analyzer calls init_module which calls register
    strategy.add_analyzer(analyzer)

    # Simulate adding results that cause convergence
    # Best values: [1.0, 1.0, 1.0, 1.0, 1.0] -> std=0 -> converged

    from panobbgo.analyzers.best import Best
    best_analyzer = Best(strategy)
    strategy.add_analyzer(best_analyzer) # Needed so strategy.best works

    results = []
    for _ in range(5):
        r = Result(Point(np.array([0.]), "test"), 1.0)
        results.append(r)

    # We need to manually trigger analyzer behavior since we are not running strategy loop
    # 1. Update Best analyzer
    best_analyzer.on_new_results(results)

    # 2. Update Convergence analyzer
    analyzer.on_new_results(results)

    # Allow some time for threads to process events
    time.sleep(0.1)

    # Check if convergence event was fired
    # The MockStrategy.on_converged should have been called

    assert strategy._stop_requested, "Strategy should have requested stop after convergence"

def test_strategy_ignores_convergence_if_disabled():
    problem = MockProblem()
    strategy = MockStrategy(problem)
    strategy.config.stop_on_convergence = False

    # Initialize analyzer
    analyzer = Convergence(strategy, window_size=5, threshold=0.1, mode='std')
    strategy.add_analyzer(analyzer)

    from panobbgo.analyzers.best import Best
    best_analyzer = Best(strategy)
    strategy.add_analyzer(best_analyzer)

    results = []
    for _ in range(5):
        r = Result(Point(np.array([0.]), "test"), 1.0)
        results.append(r)

    best_analyzer.on_new_results(results)
    analyzer.on_new_results(results)

    time.sleep(0.1)

    assert not strategy._stop_requested, "Strategy should NOT have requested stop"
