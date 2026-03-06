from panobbgo.analyzers.convergence import Convergence
from panobbgo.core import Event
from panobbgo.lib import Result, Point
import numpy as np

class MockStrategy:
    def __init__(self, problem=None):
        self.problem = problem
        self.config = MockConfig()
        self.eventbus = MockEventBus()
        self.best = None
        self.results = []

class MockConfig:
    def __init__(self):
        self.convergence_window_size = 5
        self.convergence_threshold = 0.1
        self.convergence_mode = 'std'
        self.convergence_min_evaluations = 5
        self.debug = True

    def get_logger(self, name):
        import logging
        return logging.getLogger(name)

class MockEventBus:
    def __init__(self):
        self.events = []

    def publish(self, key, **kwargs):
        self.events.append((key, kwargs))

def test_convergence_history_none():
    strategy = MockStrategy()
    analyzer = Convergence(strategy, window_size=2, threshold=0.1, mode='std', min_evaluations=2)

    # Values with a None value
    values = [1.0, None]
    for val in values:
        strategy.best = Result(Point(np.array([0.]), "test"), val)
        strategy.results.append(strategy.best)
        analyzer.on_new_results([strategy.best])

    assert not analyzer._converged

def test_check_improv_convergence_zero_start():
    strategy = MockStrategy()
    analyzer = Convergence(strategy, window_size=3, threshold=0.1, mode='improv', min_evaluations=3)

    values = [0.0, 0.0, 0.0]
    for val in values:
        strategy.best = Result(Point(np.array([0.]), "test"), val)
        strategy.results.append(strategy.best)
        analyzer.on_new_results([strategy.best])

    assert analyzer._converged

def test_check_slope_convergence_zero_mean():
    strategy = MockStrategy()
    analyzer = Convergence(strategy, window_size=3, threshold=0.1, mode='slope', min_evaluations=3)

    values = [0.0, 0.0, 0.0]
    for val in values:
        strategy.best = Result(Point(np.array([0.]), "test"), val)
        strategy.results.append(strategy.best)
        analyzer.on_new_results([strategy.best])

    assert analyzer._converged

def test_check_slope_convergence_polyfit_exception(monkeypatch):
    strategy = MockStrategy()
    analyzer = Convergence(strategy, window_size=3, threshold=0.1, mode='slope', min_evaluations=3)

    def mock_polyfit(*args, **kwargs):
        raise np.linalg.LinAlgError("SVD did not converge")

    monkeypatch.setattr(np, "polyfit", mock_polyfit)

    values = [1.0, 1.0, 1.0]
    for val in values:
        strategy.best = Result(Point(np.array([0.]), "test"), val)
        strategy.results.append(strategy.best)
        analyzer.on_new_results([strategy.best])

    assert not analyzer._converged

def test_check_convergence_strategy_results_none():
    strategy = MockStrategy()
    strategy.results = None  # None results list
    analyzer = Convergence(strategy, window_size=3, threshold=0.1, mode='std', min_evaluations=3)

    values = [1.0, 1.0, 1.0]
    for val in values:
        strategy.best = Result(Point(np.array([0.]), "test"), val)
        # Don't append to results, it's None
        analyzer.on_new_results([strategy.best])

    assert analyzer._converged

def test_check_convergence_require_feasibility_infeasible():
    strategy = MockStrategy()
    strategy.config.convergence_require_feasibility = True
    analyzer = Convergence(strategy, window_size=3, threshold=0.1, mode='std', min_evaluations=3)

    # Make sure we initialize with require_feasibility = True
    analyzer.require_feasibility = True

    # Infeasible points
    values = [1.0, 1.0, 1.0]
    for val in values:
        strategy.best = Result(Point(np.array([0.]), "test"), val, cv_vec=np.array([1.0]))
        strategy.results.append(strategy.best)
        analyzer.on_new_results([strategy.best])

    assert not analyzer._converged
