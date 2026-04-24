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


def test_convergence_history_none():
    strategy = MockStrategy()
    analyzer = Convergence(strategy, window_size=2, threshold=0.1, mode="std", min_evaluations=2)

    # Values with a None value
    values = [1.0, None]
    for val in values:
        strategy.best = Result(Point(np.array([0.0]), "test"), val)
        strategy.results.append(strategy.best)
        analyzer.on_new_results([strategy.best])

    assert not analyzer._converged


def test_check_improv_convergence_zero_start():
    strategy = MockStrategy()
    analyzer = Convergence(strategy, window_size=3, threshold=0.1, mode="improv", min_evaluations=3)

    values = [0.0, 0.0, 0.0]
    for val in values:
        strategy.best = Result(Point(np.array([0.0]), "test"), val)
        strategy.results.append(strategy.best)
        analyzer.on_new_results([strategy.best])

    assert analyzer._converged


def test_check_slope_convergence_zero_mean():
    strategy = MockStrategy()
    analyzer = Convergence(strategy, window_size=3, threshold=0.1, mode="slope", min_evaluations=3)

    values = [0.0, 0.0, 0.0]
    for val in values:
        strategy.best = Result(Point(np.array([0.0]), "test"), val)
        strategy.results.append(strategy.best)
        analyzer.on_new_results([strategy.best])

    assert analyzer._converged


def test_check_slope_convergence_polyfit_exception(monkeypatch):
    strategy = MockStrategy()
    analyzer = Convergence(strategy, window_size=3, threshold=0.1, mode="slope", min_evaluations=3)

    def mock_polyfit(*args, **kwargs):
        raise np.linalg.LinAlgError("SVD did not converge")

    monkeypatch.setattr(np, "polyfit", mock_polyfit)

    values = [1.0, 1.0, 1.0]
    for val in values:
        strategy.best = Result(Point(np.array([0.0]), "test"), val)
        strategy.results.append(strategy.best)
        analyzer.on_new_results([strategy.best])

    assert not analyzer._converged


def test_check_convergence_strategy_results_none():
    strategy = MockStrategy()
    strategy.results = None  # None results list
    analyzer = Convergence(strategy, window_size=3, threshold=0.1, mode="std", min_evaluations=3)

    values = [1.0, 1.0, 1.0]
    for val in values:
        strategy.best = Result(Point(np.array([0.0]), "test"), val)
        # Don't append to results, it's None
        analyzer.on_new_results([strategy.best])

    assert analyzer._converged


def test_check_convergence_require_feasibility_infeasible():
    strategy = MockStrategy()
    strategy.config.convergence_require_feasibility = True
    analyzer = Convergence(strategy, window_size=3, threshold=0.1, mode="std", min_evaluations=3)

    # Make sure we initialize with require_feasibility = True
    analyzer.require_feasibility = True

    # Infeasible points
    values = [1.0, 1.0, 1.0]
    for val in values:
        strategy.best = Result(Point(np.array([0.0]), "test"), val, cv_vec=np.array([1.0]))
        strategy.results.append(strategy.best)
        analyzer.on_new_results([strategy.best])

    assert not analyzer._converged


def test_check_improv_convergence_abs_start():
    strategy = MockStrategy()
    analyzer = Convergence(strategy, window_size=3, threshold=0.1, mode="improv", min_evaluations=3)

    values = [0.0, 0.0, 0.0]
    for val in values:
        strategy.best = Result(Point(np.array([0.0]), "test"), val)
        strategy.results.append(strategy.best)
        analyzer.on_new_results([strategy.best])

    assert analyzer._converged


def test_check_convergence_history_not_filled():
    strategy = MockStrategy()
    analyzer = Convergence(strategy, window_size=3, threshold=0.1, mode="improv", min_evaluations=3)

    # We only add 2 values
    values = [1.0, 0.9]
    for val in values:
        strategy.best = Result(Point(np.array([0.0]), "test"), val)
        strategy.results.append(strategy.best)
        analyzer.on_new_results([strategy.best])

    assert not analyzer._converged


def test_on_new_results_no_best():
    strategy = MockStrategy()
    analyzer = Convergence(strategy, window_size=3, threshold=0.1, mode="std", min_evaluations=3)

    # Strategy best is None
    strategy.best = None
    analyzer.on_new_results([Result(Point(np.array([0.0]), "test"), 1.0)])

    assert not analyzer._converged
    assert len(analyzer.history) == 0


def test_on_new_results_best_fx_none():
    strategy = MockStrategy()
    analyzer = Convergence(strategy, window_size=3, threshold=0.1, mode="std", min_evaluations=3)

    strategy.best = Result(Point(np.array([0.0]), "test"), None)
    analyzer.on_new_results([Result(Point(np.array([0.0]), "test"), 1.0)])

    assert not analyzer._converged
    assert len(analyzer.history) == 0


def test_check_convergence_early_return():
    strategy = MockStrategy()
    analyzer = Convergence(strategy, window_size=3, threshold=0.1, mode="std", min_evaluations=3)

    # Fill window
    values = [1.0, 1.0, 1.0]
    for val in values:
        strategy.best = Result(Point(np.array([0.0]), "test"), val)
        strategy.results.append(strategy.best)
        analyzer.on_new_results([strategy.best])

    # Now it is converged
    assert analyzer._converged

    # Try calling again, should early return at if self._converged:
    analyzer._check_convergence()
    assert len(analyzer.history) == 3


def test_check_convergence_type_error_len_results():
    strategy = MockStrategy()

    # Replace results list with something that raises TypeError on len()
    class MockResults:
        def __len__(self):
            raise TypeError("Not iterable")

    strategy.results = MockResults()
    analyzer = Convergence(strategy, window_size=3, threshold=0.1, mode="std", min_evaluations=3)

    values = [1.0, 1.0, 1.0]
    for val in values:
        strategy.best = Result(Point(np.array([0.0]), "test"), val)
        analyzer.on_new_results([strategy.best])

    # TypeError is caught, and since len(history) == window_size, it will proceed to check convergence
    assert analyzer._converged


def test_check_convergence_cv_improving():
    strategy = MockStrategy()
    analyzer = Convergence(strategy, window_size=3, threshold=0.1, mode="std", min_evaluations=3)

    # Constant FX, but CV is improving: 1.0 -> 0.8 -> 0.5
    # (1.0 - 0.5) / 1.0 = 0.5 > 0.1 threshold
    cv_values = [1.0, 0.8, 0.5]
    for cv in cv_values:
        strategy.best = Result(Point(np.array([0.0]), "test"), 1.0, cv_vec=np.array([cv]))
        strategy.results.append(strategy.best)
        analyzer.on_new_results([strategy.best])

    assert not analyzer._converged


def test_check_convergence_rel_improv_abs_start():
    strategy = MockStrategy()
    analyzer = Convergence(strategy, window_size=3, threshold=0.1, mode="improv", min_evaluations=3)

    # Start > 1e-9: 10.0 -> 10.0 -> 10.0
    values = [10.0, 10.0, 10.0]
    for val in values:
        strategy.best = Result(Point(np.array([0.0]), "test"), val)
        strategy.results.append(strategy.best)
        analyzer.on_new_results([strategy.best])

    assert analyzer._converged


def test_check_convergence_norm_slope_mean_val():
    strategy = MockStrategy()
    analyzer = Convergence(strategy, window_size=3, threshold=0.1, mode="slope", min_evaluations=3)

    # Mean val > 1e-9: 10.0 -> 10.0 -> 10.0 (Slope is 0.0)
    values = [10.0, 10.0, 10.0]
    for val in values:
        strategy.best = Result(Point(np.array([0.0]), "test"), val)
        strategy.results.append(strategy.best)
        analyzer.on_new_results([strategy.best])

    assert analyzer._converged


def test_check_convergence_min_evals_return():
    strategy = MockStrategy()
    analyzer = Convergence(strategy, window_size=3, threshold=0.1, mode="std", min_evaluations=5)

    # Strategy results has len 3, less than min_evaluations 5
    # Should hit `if len(self.strategy.results) < self.min_evaluations: return`
    values = [1.0, 1.0, 1.0]
    for val in values:
        strategy.best = Result(Point(np.array([0.0]), "test"), val)
        strategy.results.append(strategy.best)
        analyzer.on_new_results([strategy.best])

    assert not analyzer._converged


def test_check_convergence_history_any_none():
    strategy = MockStrategy()
    analyzer = Convergence(strategy, window_size=3, threshold=0.1, mode="std", min_evaluations=3)

    # History has None (simulate values = [1.0, 1.0, None])
    values = [1.0, 1.0, None]
    for val in values:
        strategy.best = Result(Point(np.array([0.0]), "test"), val)
        strategy.results.append(strategy.best)
        analyzer.on_new_results([strategy.best])

    assert not analyzer._converged


def test_check_convergence_handle_none_values():
    strategy = MockStrategy()
    analyzer = Convergence(strategy, window_size=3, threshold=0.1, mode="std", min_evaluations=3)

    # Fill history with None values explicitly
    analyzer.history.extend([1.0, 1.0, None])
    strategy.results = [1.0, 1.0, None]

    analyzer._check_convergence()

    assert not analyzer._converged
