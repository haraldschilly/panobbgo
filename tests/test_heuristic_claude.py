# -*- coding: utf8 -*-
import numpy as np
from unittest import mock
from panobbgo.heuristics.claude_heuristic import ClaudeHeuristic
from panobbgo.lib import Point, Result, Problem
from panobbgo.config import Config


class MockProblem(Problem):
    def __init__(self, dim=2):
        box = [(-5.0, 5.0)] * dim
        super().__init__(box=box)

    def eval(self, x):
        fx = np.sum(x**2)
        cv = max(0.0, 1.0 - x[0])
        return Result(Point(x, "eval"), fx, cv_vec=np.array([cv]))


def _make_strategy(problem, penalty_fn=None):
    config = Config(parse_args=False, testing_mode=True)
    strategy = mock.MagicMock()
    strategy.problem = problem
    strategy.config = config
    handler = mock.MagicMock()
    if penalty_fn is None:
        handler.get_penalty_value = lambda r: r.fx if r.cv == 0 else r.fx + 1000 * r.cv
    else:
        handler.get_penalty_value = penalty_fn
    strategy.constraint_handler = handler
    return strategy


def _make_results(n, dim, rng=None):
    """Generate n random results in [-5, 5]^dim with quadratic objective."""
    if rng is None:
        rng = np.random.default_rng(42)
    results = []
    for _ in range(n):
        x = rng.uniform(-5, 5, size=dim)
        fx = float(np.sum(x**2))
        cv = max(0.0, 1.0 - x[0])
        results.append(Result(Point(x, "test"), fx, cv_vec=np.array([cv])))
    return results


def test_initialization():
    problem = MockProblem()
    strategy = _make_strategy(problem)
    h = ClaudeHeuristic(strategy)
    assert h.name == "ClaudeHeuristic"
    assert h.elite_fraction == 0.2
    assert h.max_clusters == 5
    assert h.n_candidates == 5


def test_no_output_before_min_points():
    problem = MockProblem(dim=2)
    strategy = _make_strategy(problem)
    h = ClaudeHeuristic(strategy, min_points=20)
    h.on_start()

    # Feed only 5 results — below min_points
    results = _make_results(5, dim=2)
    h.on_new_results(results)

    points = h.get_points(10)
    assert len(points) == 0


def test_generates_points():
    problem = MockProblem(dim=2)
    strategy = _make_strategy(problem)
    h = ClaudeHeuristic(strategy, n_candidates=5, min_points=10)
    h.on_start()

    results = _make_results(30, dim=2)
    h.on_new_results(results)

    points = h.get_points(10)
    assert len(points) == 5

    for p in points:
        assert Point(p.x, "test") in problem.box


def test_generates_points_incrementally():
    """Feed results in multiple batches, verify it works after accumulating enough."""
    problem = MockProblem(dim=2)
    strategy = _make_strategy(problem)
    h = ClaudeHeuristic(strategy, n_candidates=3, min_points=15)
    h.on_start()

    rng = np.random.default_rng(123)

    # First batch: not enough
    h.on_new_results(_make_results(5, dim=2, rng=rng))
    assert len(h.get_points(10)) == 0

    # Second batch: still not enough
    h.on_new_results(_make_results(5, dim=2, rng=rng))
    assert len(h.get_points(10)) == 0

    # Third batch: now we have 15 total
    h.on_new_results(_make_results(5, dim=2, rng=rng))
    points = h.get_points(10)
    assert len(points) == 3


def test_single_gaussian_fallback():
    """With high dim and few elite points, should fall back to single Gaussian."""
    problem = MockProblem(dim=10)
    strategy = _make_strategy(problem)
    # min_points=5, elite_fraction=0.5 → 5 elite points, but 2*dim=20 > 5
    h = ClaudeHeuristic(strategy, n_candidates=3, min_points=5, elite_fraction=0.5)
    h.on_start()

    results = _make_results(10, dim=10)
    h.on_new_results(results)

    points = h.get_points(10)
    assert len(points) == 3
    for p in points:
        assert Point(p.x, "test") in problem.box


def test_1d_problem():
    """dim=1 edge case: np.cov returns scalar, must be reshaped."""
    problem = MockProblem(dim=1)
    strategy = _make_strategy(problem)
    h = ClaudeHeuristic(strategy, n_candidates=3, min_points=5)
    h.on_start()

    results = _make_results(20, dim=1)
    h.on_new_results(results)

    points = h.get_points(10)
    assert len(points) == 3
    for p in points:
        assert Point(p.x, "test") in problem.box


def test_constraint_aware():
    """Elite selection should prefer feasible points (low cv) when penalty includes cv."""
    problem = MockProblem(dim=2)

    # Penalty heavily penalizes constraint violation
    def penalty_fn(r):
        return r.fx + 100 * r.cv

    strategy = _make_strategy(problem, penalty_fn=penalty_fn)
    h = ClaudeHeuristic(strategy, n_candidates=5, min_points=10, elite_fraction=0.3)
    h.on_start()

    rng = np.random.default_rng(42)
    results = []
    # Create 20 feasible points near [2, 0] (good region: cv=0, fx moderate)
    for _ in range(20):
        x = np.array([2.0 + rng.normal(0, 0.5), rng.normal(0, 0.5)])
        x = np.clip(x, -5, 5)
        fx = float(np.sum(x**2))
        results.append(Result(Point(x, "test"), fx, cv_vec=np.array([0.0])))

    # Create 20 infeasible points near [-2, 0] (bad region: cv=3, fx moderate)
    for _ in range(20):
        x = np.array([-2.0 + rng.normal(0, 0.5), rng.normal(0, 0.5)])
        x = np.clip(x, -5, 5)
        fx = float(np.sum(x**2))
        cv = max(0.0, 1.0 - x[0])
        results.append(Result(Point(x, "test"), fx, cv_vec=np.array([cv])))

    h.on_new_results(results)
    points = h.get_points(10)
    assert len(points) == 5

    # Emitted points should be biased toward the feasible region (x[0] > 0)
    x0_values = [p.x[0] for p in points]
    avg_x0 = np.mean(x0_values)
    assert avg_x0 > 0, f"Expected positive avg x[0] (feasible region), got {avg_x0}"


def test_clear_output_on_new_results():
    """Verify that stale points are cleared when new results arrive."""
    problem = MockProblem(dim=2)
    strategy = _make_strategy(problem)
    h = ClaudeHeuristic(strategy, n_candidates=3, min_points=5)
    h.on_start()

    results = _make_results(20, dim=2)
    h.on_new_results(results)
    assert len(h.get_points(10)) == 3

    # Send more results — should clear old and emit fresh
    h.on_new_results(_make_results(5, dim=2, rng=np.random.default_rng(99)))
    points = h.get_points(10)
    assert len(points) == 3
