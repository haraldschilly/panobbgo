import pytest
import numpy as np
from panobbgo.analyzers.convergence import Convergence
from panobbgo.lib import Problem, Point, Result, BoundingBox
from unittest import mock


class MockStrategy:
    def __init__(self, problem):
        self.problem = problem
        self.config = mock.MagicMock()
        self.config.max_eval = 100
        self.config.get_logger.return_value = mock.MagicMock()

        # Explicitly set mock config variables
        self.config.convergence_require_feasibility = False
        self.config.convergence_mode = "std"

        self.eventbus = mock.MagicMock()
        self.best = None
        self.results = []


class MockProblem(Problem):
    def __init__(self, dim):
        super().__init__([(-5, 5)] * dim)

    def eval(self, x):
        return np.sum(x**2)

    def eval_constraints(self, x):
        return [0.0]


class MockResult:
    def __init__(self, fx, cv=0.0):
        self.fx = fx
        self.cv = cv


def make_result(fx, cv=0.0):
    return MockResult(fx, cv)


def test_convergence_history_not_full():
    problem = MockProblem(dim=2)
    strategy = MockStrategy(problem)
    conv = Convergence(strategy, window_size=5)

    strategy.best = make_result(10.0)
    strategy.results = [strategy.best] * 3

    # window_size is 5, but we only send 3
    conv.on_new_results([strategy.best] * 3)
    assert not conv._converged


def test_convergence_min_evaluations_not_met():
    problem = MockProblem(dim=2)
    strategy = MockStrategy(problem)
    # window size is 5, min evals is 10
    conv = Convergence(strategy, window_size=5, min_evaluations=10)

    strategy.best = make_result(10.0)
    strategy.results = [strategy.best] * 6  # enough for window_size, not enough for min_evaluations

    conv.on_new_results([strategy.best] * 6)
    assert not conv._converged


def test_convergence_require_feasibility_not_met():
    problem = MockProblem(dim=2)
    strategy = MockStrategy(problem)
    strategy.config.convergence_require_feasibility = True
    conv = Convergence(strategy, window_size=2, min_evaluations=2)

    # Highly infeasible (cv > 1e-6)
    strategy.best = make_result(10.0, cv=1.0)
    strategy.results = [strategy.best] * 5

    conv.on_new_results([strategy.best] * 5)
    assert not conv._converged


def test_convergence_cv_improving():
    problem = MockProblem(dim=2)
    strategy = MockStrategy(problem)
    conv = Convergence(strategy, window_size=2, min_evaluations=2, threshold=0.1)

    # First result with cv=1.0
    r1 = make_result(10.0, cv=1.0)
    strategy.best = r1
    strategy.results = [r1]
    conv.on_new_results([r1])

    # Second result with cv=0.5 -> relative improvement is 0.5 > 0.1
    r2 = make_result(10.0, cv=0.5)
    strategy.best = r2
    strategy.results = [r1, r2]
    conv.on_new_results([r2])

    # Should delay convergence due to CV improvement
    assert not conv._converged


def test_convergence_slope_zero_mean():
    problem = MockProblem(dim=2)
    strategy = MockStrategy(problem)
    strategy.config.convergence_mode = "slope"
    conv = Convergence(strategy, window_size=3, min_evaluations=3, mode="slope", threshold=0.1)

    # Mean of fx is 0 -> test norm_slope logic
    r1 = make_result(0.0)
    strategy.best = r1
    strategy.results = [r1] * 3

    conv.on_new_results([r1] * 3)
    assert conv._converged


def test_convergence_slope_with_polyfit_error():
    problem = MockProblem(dim=2)
    strategy = MockStrategy(problem)
    strategy.config.convergence_mode = "slope"
    conv = Convergence(strategy, window_size=3, min_evaluations=3, mode="slope")

    # Mock polyfit to raise an exception
    with mock.patch("numpy.polyfit", side_effect=Exception("polyfit error")):
        r1 = make_result(1.0)
        strategy.best = r1
        strategy.results = [r1] * 3
        conv.on_new_results([r1] * 3)
        assert not conv._converged


def test_convergence_none_values_in_history():
    problem = MockProblem(dim=2)
    strategy = MockStrategy(problem)
    conv = Convergence(strategy, window_size=2, min_evaluations=2)

    r1 = make_result(1.0)
    strategy.best = r1
    strategy.results = [r1]
    conv.on_new_results([r1])

    r2 = make_result(None)  # we use MockResult to pass None
    strategy.best = r2
    strategy.results = [r1, r2]
    # To test None in history, we'd have to inject it manually
    conv.history.append(None)
    conv._check_convergence()

    assert not conv._converged


def test_convergence_min_evaluations_typeerror():
    # If strategy.results length check raises TypeError
    problem = MockProblem(dim=2)
    strategy = MockStrategy(problem)
    conv = Convergence(strategy, window_size=2, min_evaluations=2)

    # ensure it triggers convergence check successfully afterwards
    conv.threshold = 1.0  # 1.0 > std(1.0, 1.0, 1.0) == 0.0 -> converges

    class MockResults:
        def __len__(self):
            raise TypeError("Not iterable")

    strategy.results = MockResults()
    strategy.best = make_result(1.0)

    conv.history.extend([1.0, 1.0])
    conv.cv_history.extend([0.0, 0.0])

    conv._check_convergence()

    assert conv._converged


def test_convergence_cv_improving_not_enough_history():
    problem = MockProblem(dim=2)
    strategy = MockStrategy(problem)
    conv = Convergence(strategy, window_size=3, min_evaluations=2)

    r1 = make_result(10.0, cv=1.0)
    strategy.best = r1
    strategy.results = [r1] * 2

    # Add only 2 items to cv_history (window size is 3)
    conv.on_new_results([r1] * 2)
    # Will skip the CV improvement check
    assert not conv._converged


def test_convergence_improv_mode():
    problem = MockProblem(dim=2)
    strategy = MockStrategy(problem)
    strategy.config.convergence_mode = "improv"
    conv = Convergence(strategy, window_size=3, min_evaluations=3, mode="improv", threshold=0.1)

    # Send values where start=10.0, end=10.0
    r1 = make_result(10.0)
    strategy.best = r1
    strategy.results = [r1] * 3

    conv.on_new_results([r1] * 3)
    assert conv._converged


def test_convergence_improv_mode_zero_start():
    problem = MockProblem(dim=2)
    strategy = MockStrategy(problem)
    strategy.config.convergence_mode = "improv"
    conv = Convergence(strategy, window_size=3, min_evaluations=3, mode="improv", threshold=0.1)

    # Send values where start=0.0
    r1 = make_result(0.0)
    strategy.best = r1
    strategy.results = [r1] * 3

    conv.on_new_results([r1] * 3)
    assert conv._converged
