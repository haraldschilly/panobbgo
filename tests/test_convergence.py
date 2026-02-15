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

class MockConfig:
    def __init__(self):
        self.convergence_window_size = 5
        self.convergence_threshold = 0.1
        self.convergence_mode = 'std'
        self.debug = True

    def get_logger(self, name):
        import logging
        return logging.getLogger(name)

class MockEventBus:
    def __init__(self):
        self.events = []

    def publish(self, key, **kwargs):
        self.events.append((key, kwargs))

def test_convergence_std():
    strategy = MockStrategy()
    analyzer = Convergence(strategy, window_size=5, threshold=0.1, mode='std')

    # Simulate a sequence of results that converge
    # First, some improving values
    values = [10.0, 5.0, 2.0, 1.0, 0.5]

    for val in values:
        strategy.best = Result(Point(np.array([0.]), "test"), val)
        analyzer.on_new_results([strategy.best])

    assert not analyzer._converged, "Should not be converged yet (std dev is high)"

    # Now stagnate
    stagnant_values = [0.5, 0.51, 0.49, 0.5, 0.5]
    for val in stagnant_values:
        strategy.best = Result(Point(np.array([0.]), "test"), val)
        analyzer.on_new_results([strategy.best])

    assert analyzer._converged, "Should be converged now"
    assert len(strategy.eventbus.events) > 0
    assert strategy.eventbus.events[-1][0] == "converged"

def test_convergence_improv():
    strategy = MockStrategy()
    # threshold 0.1 means if relative improvement < 10%
    analyzer = Convergence(strategy, window_size=3, threshold=0.1, mode='improv')

    # 10 -> 5 (50% improv) -> 4 (20% improv)
    values = [10.0, 5.0, 4.0]
    for val in values:
        strategy.best = Result(Point(np.array([0.]), "test"), val)
        analyzer.on_new_results([strategy.best])

    assert not analyzer._converged

    # 4.0 -> 3.9 (2.5% improv) -> CONVERGED
    # Window: [5.0, 4.0, 3.9]. Improv = (5-3.9)/5 = 0.22 > 0.1. Not converged.
    strategy.best = Result(Point(np.array([0.]), "test"), 3.9)
    analyzer.on_new_results([strategy.best])

    assert not analyzer._converged

    # Add another 3.9. Window: [4.0, 3.9, 3.9]. Improv = 0.1/4 = 0.025 (2.5%) < 10%
    strategy.best = Result(Point(np.array([0.]), "test"), 3.9)
    analyzer.on_new_results([strategy.best])

    assert analyzer._converged

def test_convergence_batch():
    strategy = MockStrategy()
    analyzer = Convergence(strategy, window_size=5, threshold=0.1, mode='std')

    # Set initial best
    strategy.best = Result(Point(np.array([0.]), "test"), 1.0)

    # Process a batch of 5 results that don't improve
    batch = [Result(Point(np.array([0.]), "test"), 1.1) for _ in range(5)]
    analyzer.on_new_results(batch)

    # Should converge because we had 5 evaluations with same best value (1.0)
    assert analyzer._converged

    # Reset
    strategy = MockStrategy()
    analyzer = Convergence(strategy, window_size=5, threshold=0.1, mode='std')
    strategy.best = Result(Point(np.array([0.]), "test"), 1.0)

    # Process batch of 4 (less than window)
    batch = [Result(Point(np.array([0.]), "test"), 1.1) for _ in range(4)]
    analyzer.on_new_results(batch)
    assert not analyzer._converged

    # Process 1 more
    batch = [Result(Point(np.array([0.]), "test"), 1.1)]
    analyzer.on_new_results(batch)
    assert analyzer._converged
