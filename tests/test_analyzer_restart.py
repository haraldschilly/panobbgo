# -*- coding: utf8 -*-
import numpy as np
from unittest import mock
from panobbgo.analyzers.restart import Restart
from panobbgo.lib import Point, Result, Problem
from panobbgo.config import Config


class FlatProblem(Problem):
    """f(x) = 0 for all x. Guarantees stagnation."""

    def __init__(self, dim=2):
        super().__init__(box=[(-5.0, 5.0)] * dim)

    def eval(self, x):
        return 0.0


class ImprovingProblem(Problem):
    """f(x) = -sum(x). Easy to improve by increasing x."""

    def __init__(self, dim=2):
        super().__init__(box=[(-5.0, 5.0)] * dim)

    def eval(self, x):
        return float(-np.sum(x))


def _make_strategy(problem):
    strategy = mock.MagicMock()
    strategy.problem = problem
    strategy.config = Config(parse_args=False, testing_mode=True)
    strategy.constraint_handler = None
    return strategy


def _make_results(xs, problem):
    results = []
    for x in xs:
        x = np.asarray(x, dtype=np.float64)
        pt = Point(x, "test")
        fx = problem.eval(x)
        results.append(Result(pt, fx))
    return results


@mock.patch("panobbgo.core.StrategyBase")
def test_restart_fires_after_patience(StrategyBaseMock):
    problem = FlatProblem(dim=2)
    strategy = _make_strategy(problem)
    r = Restart(strategy, patience=10, max_restarts=5)
    r.__start__()

    rng = np.random.default_rng(0)

    # First result sets the baseline (improvement from inf → 0)
    xs0 = rng.uniform(-5, 5, (1, 2))
    r.on_new_results(_make_results(xs0, problem))
    assert r.restart_count == 0

    # Feed 10 more stagnant results — should trigger restart
    xs = rng.uniform(-5, 5, (10, 2))
    r.on_new_results(_make_results(xs, problem))
    assert r.restart_count == 1

    strategy.eventbus.publish.assert_called()
    call_args = strategy.eventbus.publish.call_args
    assert call_args[0][0] == "restart"
    assert "center" in call_args[1]
    assert "reason" in call_args[1]
    center = call_args[1]["center"]
    # Center should be inside the box
    assert np.all(center >= -5.0) and np.all(center <= 5.0)


@mock.patch("panobbgo.core.StrategyBase")
def test_no_restart_when_improving(StrategyBaseMock):
    problem = ImprovingProblem(dim=2)
    strategy = _make_strategy(problem)
    r = Restart(strategy, patience=10, max_restarts=5)
    r.__start__()

    # Feed results with continuously improving fx
    for i in range(20):
        x = np.array([float(i), float(i)])
        results = _make_results([x], problem)
        r.on_new_results(results)

    assert r.restart_count == 0


@mock.patch("panobbgo.core.StrategyBase")
def test_max_restarts_limit(StrategyBaseMock):
    problem = FlatProblem(dim=2)
    strategy = _make_strategy(problem)
    r = Restart(strategy, patience=5, max_restarts=3)
    r.__start__()

    rng = np.random.default_rng(1)

    # First result sets baseline
    xs0 = rng.uniform(-5, 5, (1, 2))
    r.on_new_results(_make_results(xs0, problem))

    # Trigger many restart cycles
    for _ in range(10):
        xs = rng.uniform(-5, 5, (6, 2))
        r.on_new_results(_make_results(xs, problem))

    # Should stop at max_restarts
    assert r.restart_count == 3


@mock.patch("panobbgo.core.StrategyBase")
def test_default_patience(StrategyBaseMock):
    problem = FlatProblem(dim=4)
    strategy = _make_strategy(problem)
    r = Restart(strategy)
    r.__start__()
    assert r._patience == 20  # 5 * dim


@mock.patch("panobbgo.core.StrategyBase")
def test_diverse_strategy(StrategyBaseMock):
    problem = FlatProblem(dim=2)
    strategy = _make_strategy(problem)
    r = Restart(strategy, patience=5, max_restarts=5, restart_strategy="diverse")
    r.__start__()

    rng = np.random.default_rng(2)

    # First result sets baseline
    xs0 = rng.uniform(-5, 5, (1, 2))
    r.on_new_results(_make_results(xs0, problem))

    # Trigger two restarts
    for _ in range(2):
        xs = rng.uniform(-5, 5, (6, 2))
        r.on_new_results(_make_results(xs, problem))

    assert r.restart_count == 2
    # Previous centers should be stored
    assert len(r._previous_centers) == 2
    # Centers should be distinct
    dist = np.linalg.norm(r._previous_centers[0] - r._previous_centers[1])
    assert dist > 0


@mock.patch("panobbgo.core.StrategyBase")
def test_counter_resets_after_restart(StrategyBaseMock):
    """After restart, the patience counter should reset so it takes another patience evals to trigger."""
    problem = FlatProblem(dim=2)
    strategy = _make_strategy(problem)
    r = Restart(strategy, patience=5, max_restarts=10)
    r.__start__()

    rng = np.random.default_rng(3)

    # First result sets baseline
    xs0 = rng.uniform(-5, 5, (1, 2))
    r.on_new_results(_make_results(xs0, problem))

    # Trigger first restart (5 stagnant)
    xs = rng.uniform(-5, 5, (5, 2))
    r.on_new_results(_make_results(xs, problem))
    assert r.restart_count == 1

    # Only 3 more stagnant — should NOT restart yet
    xs2 = rng.uniform(-5, 5, (3, 2))
    r.on_new_results(_make_results(xs2, problem))
    assert r.restart_count == 1

    # 2 more to complete patience — NOW restart
    xs3 = rng.uniform(-5, 5, (2, 2))
    r.on_new_results(_make_results(xs3, problem))
    assert r.restart_count == 2
