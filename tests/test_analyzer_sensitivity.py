# -*- coding: utf8 -*-
import pytest
import numpy as np
from unittest import mock
from panobbgo.analyzers.sensitivity import Sensitivity
from panobbgo.lib import Point, Result, Problem
from panobbgo.config import Config


class DimZeroProblem(Problem):
    """f(x) = 10*x[0] + 0.001*x[1], dim 0 dominates."""

    def __init__(self):
        super().__init__(box=[(-5.0, 5.0), (-5.0, 5.0)])

    def eval(self, x):
        return float(10.0 * x[0] + 0.001 * x[1])


def _make_strategy(problem):
    strategy = mock.MagicMock()
    strategy.problem = problem
    strategy.config = Config(parse_args=False, testing_mode=True)
    strategy.constraint_handler = None
    return strategy


def _make_results(xs, problem):
    """Create Result objects from an array of x vectors."""
    results = []
    for x in xs:
        x = np.asarray(x, dtype=np.float64)
        pt = Point(x, "test")
        fx = problem.eval(x)
        results.append(Result(pt, fx))
    return results


@mock.patch("panobbgo.core.StrategyBase")
def test_no_event_before_min_samples(StrategyBaseMock):
    problem = DimZeroProblem()
    strategy = _make_strategy(problem)
    s = Sensitivity(strategy, min_samples=20, update_interval=5)
    s.__start__()

    # Feed 10 results (less than min_samples=20)
    xs = np.random.default_rng(0).uniform(-5, 5, (10, 2))
    results = _make_results(xs, problem)
    s.on_new_results(results)

    assert s.importance is None
    strategy.eventbus.publish.assert_not_called()


@mock.patch("panobbgo.core.StrategyBase")
def test_dim0_more_important_than_dim1(StrategyBaseMock):
    """For f(x) = x[0]^2, dimension 0 should be much more important than dimension 1."""
    problem = DimZeroProblem()
    strategy = _make_strategy(problem)
    s = Sensitivity(strategy, min_samples=30, update_interval=10)
    s.__start__()

    rng = np.random.default_rng(42)
    xs = rng.uniform(-5, 5, (50, 2))
    results = _make_results(xs, problem)
    s.on_new_results(results)

    assert s.importance is not None
    assert s.importance[0] > s.importance[1]
    # dim 0 should be normalized to 1.0 (the max)
    assert s.importance[0] == pytest.approx(1.0)
    # dim 1 should be small (nearly zero correlation with x[0]^2)
    assert s.importance[1] < 0.3


@mock.patch("panobbgo.core.StrategyBase")
def test_update_interval_triggers(StrategyBaseMock):
    """Sensitivity should not recompute until update_interval new results arrive."""
    problem = DimZeroProblem()
    strategy = _make_strategy(problem)
    s = Sensitivity(strategy, min_samples=10, update_interval=20)
    s.__start__()

    rng = np.random.default_rng(1)

    # Feed 15 results — past min_samples but not past update_interval
    xs = rng.uniform(-5, 5, (15, 2))
    s.on_new_results(_make_results(xs, problem))
    assert s.importance is None  # not yet: only 15 since last update, need 20

    # Feed 10 more — now 25 total since last update, > 20
    xs2 = rng.uniform(-5, 5, (10, 2))
    s.on_new_results(_make_results(xs2, problem))
    assert s.importance is not None


@mock.patch("panobbgo.core.StrategyBase")
def test_partial_method(StrategyBaseMock):
    """Partial correlation method should produce valid importance scores."""
    problem = DimZeroProblem()
    strategy = _make_strategy(problem)
    s = Sensitivity(strategy, min_samples=30, update_interval=10, method="partial")
    s.__start__()

    rng = np.random.default_rng(99)
    xs = rng.uniform(-5, 5, (100, 2))
    results = _make_results(xs, problem)
    s.on_new_results(results)

    assert s.importance is not None
    assert s.importance.shape == (2,)
    # All values should be in [0, 1] with max = 1
    assert np.all(s.importance >= 0.0)
    assert np.max(s.importance) == pytest.approx(1.0)
    # dim 0 should be at least as important (it has 10x coefficient)
    assert s.importance[0] >= s.importance[1]


@mock.patch("panobbgo.core.StrategyBase")
def test_publishes_event(StrategyBaseMock):
    problem = DimZeroProblem()
    strategy = _make_strategy(problem)
    s = Sensitivity(strategy, min_samples=10, update_interval=5)
    s.__start__()

    rng = np.random.default_rng(7)
    xs = rng.uniform(-5, 5, (20, 2))
    s.on_new_results(_make_results(xs, problem))

    strategy.eventbus.publish.assert_called()
    call_args = strategy.eventbus.publish.call_args
    assert call_args[0][0] == "new_sensitivity"
    assert "importance" in call_args[1]
    imp = call_args[1]["importance"]
    assert imp.shape == (2,)


@mock.patch("panobbgo.core.StrategyBase")
def test_default_min_samples(StrategyBaseMock):
    problem = DimZeroProblem()
    strategy = _make_strategy(problem)
    s = Sensitivity(strategy, update_interval=5)
    s.__start__()
    # Default min_samples = 10 * dim = 20
    assert s._min_samples == 20
