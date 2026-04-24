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
        self.convergence_mode = "std"
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


def test_convergence_std():
    strategy = MockStrategy()
    analyzer = Convergence(strategy, window_size=5, threshold=0.1, mode="std", min_evaluations=5)

    # Simulate a sequence of results that converge
    # First, some improving values
    values = [10.0, 5.0, 2.0, 1.0, 0.5]

    for val in values:
        strategy.best = Result(Point(np.array([0.0]), "test"), val)
        strategy.results.append(strategy.best)
        analyzer.on_new_results([strategy.best])

    assert not analyzer._converged, "Should not be converged yet (std dev is high)"

    # Now stagnate
    stagnant_values = [0.5, 0.51, 0.49, 0.5, 0.5]
    for val in stagnant_values:
        strategy.best = Result(Point(np.array([0.0]), "test"), val)
        strategy.results.append(strategy.best)
        analyzer.on_new_results([strategy.best])

    assert analyzer._converged, "Should be converged now"
    assert len(strategy.eventbus.events) > 0
    assert strategy.eventbus.events[-1][0] == "converged"


def test_convergence_improv():
    strategy = MockStrategy()
    # threshold 0.1 means if relative improvement < 10%
    analyzer = Convergence(strategy, window_size=3, threshold=0.1, mode="improv", min_evaluations=3)

    # 10 -> 5 (50% improv) -> 4 (20% improv)
    values = [10.0, 5.0, 4.0]
    for val in values:
        strategy.best = Result(Point(np.array([0.0]), "test"), val)
        strategy.results.append(strategy.best)
        analyzer.on_new_results([strategy.best])

    assert not analyzer._converged

    # 4.0 -> 3.9 (2.5% improv) -> CONVERGED
    # Window: [5.0, 4.0, 3.9]. Improv = (5-3.9)/5 = 0.22 > 0.1. Not converged.
    strategy.best = Result(Point(np.array([0.0]), "test"), 3.9)
    strategy.results.append(strategy.best)
    analyzer.on_new_results([strategy.best])

    assert not analyzer._converged

    # Add another 3.9. Window: [4.0, 3.9, 3.9]. Improv = 0.1/4 = 0.025 (2.5%) < 10%
    strategy.best = Result(Point(np.array([0.0]), "test"), 3.9)
    strategy.results.append(strategy.best)
    analyzer.on_new_results([strategy.best])

    assert analyzer._converged


def test_convergence_batch():
    strategy = MockStrategy()
    analyzer = Convergence(strategy, window_size=5, threshold=0.1, mode="std", min_evaluations=5)

    # Set initial best
    strategy.best = Result(Point(np.array([0.0]), "test"), 1.0)

    # Process a batch of 5 results that don't improve
    batch = [Result(Point(np.array([0.0]), "test"), 1.1) for _ in range(5)]
    strategy.results.extend(batch)
    analyzer.on_new_results(batch)

    # Should converge because we had 5 evaluations with same best value (1.0)
    assert analyzer._converged

    # Reset
    strategy = MockStrategy()
    analyzer = Convergence(strategy, window_size=5, threshold=0.1, mode="std", min_evaluations=5)
    strategy.best = Result(Point(np.array([0.0]), "test"), 1.0)

    # Process batch of 4 (less than window)
    batch = [Result(Point(np.array([0.0]), "test"), 1.1) for _ in range(4)]
    strategy.results.extend(batch)
    analyzer.on_new_results(batch)
    assert not analyzer._converged

    # Process 1 more
    batch = [Result(Point(np.array([0.0]), "test"), 1.1)]
    strategy.results.extend(batch)
    analyzer.on_new_results(batch)
    assert analyzer._converged


def test_convergence_slope():
    strategy = MockStrategy()
    # Slope mode. Threshold 0.1
    analyzer = Convergence(strategy, window_size=5, threshold=0.1, mode="slope", min_evaluations=5)

    # Improving: 10, 8, 6, 4, 2. Slope is -2. Normalized by mean(6) -> -0.33. abs(-0.33) > 0.1. Not converged.
    values = [10.0, 8.0, 6.0, 4.0, 2.0]
    for val in values:
        strategy.best = Result(Point(np.array([0.0]), "test"), val)
        strategy.results.append(strategy.best)
        analyzer.on_new_results([strategy.best])

    assert not analyzer._converged, "Slope is significant, should not converge"

    # Flat: 2.0, 2.0, 2.0, 2.0, 2.0. Slope is 0.
    flat_values = [2.0, 2.0, 2.0, 2.0, 2.0]
    for val in flat_values:
        strategy.best = Result(Point(np.array([0.0]), "test"), val)
        strategy.results.append(strategy.best)
        analyzer.on_new_results([strategy.best])

    assert analyzer._converged, "Slope is zero, should converge"


def test_convergence_min_evaluations():
    strategy = MockStrategy()
    # Min evals = 10, window = 5
    analyzer = Convergence(strategy, window_size=5, threshold=0.1, mode="std", min_evaluations=10)

    # 5 flat values. Should converge by window logic, but blocked by min_evals
    values = [1.0] * 5
    for val in values:
        strategy.best = Result(Point(np.array([0.0]), "test"), val)
        strategy.results.append(strategy.best)
        analyzer.on_new_results([strategy.best])

    assert not analyzer._converged, "Blocked by min_evaluations"

    # Add 5 more
    for val in values:
        strategy.best = Result(Point(np.array([0.0]), "test"), val)
        strategy.results.append(strategy.best)
        analyzer.on_new_results([strategy.best])

    assert analyzer._converged, "Should converge now (10 evals)"


def test_convergence_constraint_awareness():
    strategy = MockStrategy()
    # Window 5. Stagnant FX.
    analyzer = Convergence(strategy, window_size=5, threshold=0.1, mode="std", min_evaluations=5)

    # FX is constant (1.0), but CV improves: 1.0 -> 0.9 -> ... -> 0.6
    # Relative improv of CV: (1.0 - 0.6)/1.0 = 0.4 > threshold (0.1). So should NOT converge.

    cv_values = [1.0, 0.9, 0.8, 0.7, 0.6]
    for cv in cv_values:
        strategy.best = Result(Point(np.array([0.0]), "test"), 1.0, cv_vec=np.array([cv]))
        strategy.results.append(strategy.best)
        analyzer.on_new_results([strategy.best])

    assert not analyzer._converged, "CV is improving, should not converge"

    # Now CV stagnates too: 0.6 -> 0.6 -> ...
    cv_stagnant = [0.6] * 5
    for cv in cv_stagnant:
        strategy.best = Result(Point(np.array([0.0]), "test"), 1.0, cv_vec=np.array([cv]))
        strategy.results.append(strategy.best)
        analyzer.on_new_results([strategy.best])

    assert analyzer._converged, "CV stagnated too, should converge"
