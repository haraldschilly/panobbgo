# -*- coding: utf8 -*-
"""
Tests for Nearby heuristic sensitivity integration and on_restart handler,
and on_restart handlers for ClaudeHeuristic and GaussianProcessHeuristic.
"""
import numpy as np
import pytest
from unittest import mock

from panobbgo.heuristics.nearby import Nearby
from panobbgo.lib import Point, Result, Problem
from panobbgo.config import Config


class SimpleProblem(Problem):
    """f(x) = sum(x^2), box [-5, 5]^dim"""

    def __init__(self, dim=3):
        super().__init__(box=[(-5.0, 5.0)] * dim)

    def eval(self, x):
        return float(np.sum(x**2))


def _make_strategy(problem):
    strategy = mock.MagicMock()
    strategy.problem = problem
    strategy.config = Config(parse_args=False, testing_mode=True)
    from panobbgo.lib.constraints import DefaultConstraintHandler
    strategy.constraint_handler = DefaultConstraintHandler(strategy)
    return strategy


# ---------------------------------------------------------------------------
# Nearby + sensitivity tests
# ---------------------------------------------------------------------------


def test_nearby_starts_without_sensitivity():
    """Nearby works normally before any sensitivity scores arrive."""
    problem = SimpleProblem(dim=3)
    strategy = _make_strategy(problem)
    h = Nearby(strategy, new=3, axes="one")
    h.__start__()

    assert h._importance is None
    x = np.array([1.0, 2.0, 3.0])
    best = Result(Point(x, "test"), 14.0)
    h.on_new_best(best)
    points = h.get_points(10)
    assert 1 <= len(points) <= 3
    for p in points:
        assert Point(p.x, "test") in problem.box


def test_nearby_accepts_sensitivity():
    """on_new_sensitivity stores importance correctly."""
    problem = SimpleProblem(dim=4)
    strategy = _make_strategy(problem)
    h = Nearby(strategy)
    h.__start__()

    importance = np.array([1.0, 0.0, 0.5, 0.2])
    h.on_new_sensitivity(importance)

    np.testing.assert_array_equal(h._importance, importance)


def test_nearby_sensitivity_axes_one_favors_important_dims():
    """With axes='one' and importance=[1,0,0], dim 0 should be chosen exclusively."""
    problem = SimpleProblem(dim=3)
    strategy = _make_strategy(problem)
    h = Nearby(strategy, new=1, axes="one", radius=0.1)
    h.__start__()

    # Only dimension 0 is important
    h.on_new_sensitivity(np.array([1.0, 0.0, 0.0]))

    x = np.zeros(3)
    best = Result(Point(x, "test"), 0.0)

    # Generate many perturbations and check that only dim 0 changes
    n_trials = 50
    changed = np.zeros(3, dtype=int)
    for _ in range(n_trials):
        h.on_new_best(best)
        pts = h.get_points(1)
        if pts:
            new_x = pts[0].x
            for i in range(3):
                if abs(new_x[i] - x[i]) > 1e-10:
                    changed[i] += 1

    # Dimension 0 should have changed; dims 1 and 2 should not
    assert changed[0] > 0, "Expected dimension 0 to be perturbed"
    assert changed[1] == 0, "Dimension 1 should never be perturbed (importance=0)"
    assert changed[2] == 0, "Dimension 2 should never be perturbed (importance=0)"


def test_nearby_sensitivity_axes_all_scales_by_importance():
    """With axes='all', perturbation magnitude is zero for importance=0 dims."""
    problem = SimpleProblem(dim=3)
    strategy = _make_strategy(problem)
    h = Nearby(strategy, new=1, axes="all", radius=0.5)
    h.__start__()

    # Dim 2 is irrelevant
    h.on_new_sensitivity(np.array([1.0, 0.5, 0.0]))

    x = np.zeros(3)
    best = Result(Point(x, "test"), 0.0)

    n_trials = 30
    dim2_changed = 0
    for _ in range(n_trials):
        h.on_new_best(best)
        pts = h.get_points(1)
        if pts and abs(pts[0].x[2] - x[2]) > 1e-10:
            dim2_changed += 1

    assert dim2_changed == 0, "Dimension 2 (importance=0) should not be perturbed"


def test_nearby_sensitivity_axes_all_important_dims_do_change():
    """With axes='all' and importance=[1,1,0], dims 0 and 1 do change."""
    problem = SimpleProblem(dim=3)
    strategy = _make_strategy(problem)
    h = Nearby(strategy, new=5, axes="all", radius=0.5)
    h.__start__()

    h.on_new_sensitivity(np.array([1.0, 1.0, 0.0]))

    x = np.zeros(3)
    best = Result(Point(x, "test"), 0.0)
    h.on_new_best(best)
    pts = h.get_points(5)
    assert len(pts) > 0

    # At least some points should differ in dim 0 or 1
    any_changed = any(abs(p.x[0] - x[0]) > 1e-10 or abs(p.x[1] - x[1]) > 1e-10 for p in pts)
    assert any_changed, "Dims 0 or 1 should be perturbed with importance=1"


def test_nearby_restart_uses_new_center():
    """on_restart generates points near the new center."""
    problem = SimpleProblem(dim=3)
    strategy = _make_strategy(problem)
    h = Nearby(strategy, new=2, axes="one", radius=0.01)
    h.__start__()

    center = np.array([3.0, -2.0, 1.5])
    h.on_restart(center, "test restart")

    pts = h.get_points(5)
    assert len(pts) == 2
    for p in pts:
        # Very small radius=0.01 so points must be close to center
        assert np.linalg.norm(p.x - center) < 1.0
        assert Point(p.x, "test") in problem.box


def test_nearby_restart_clears_old_output():
    """on_restart clears stale output before generating new points."""
    problem = SimpleProblem(dim=2)
    strategy = _make_strategy(problem)
    h = Nearby(strategy, new=3, axes="one")
    h.__start__()

    # First, fill queue with "old" points from a best near origin
    best_old = Result(Point(np.zeros(2), "test"), 0.0)
    h.on_new_best(best_old)
    assert h.get_points(10)  # consume old points

    # Generate more "old" points
    h.on_new_best(best_old)
    # Restart should wipe them
    center = np.array([4.0, 4.0])
    h.on_restart(center, "forced restart")

    pts = h.get_points(10)
    # All points should be near the restart center, not origin
    for p in pts:
        assert np.linalg.norm(p.x - center) < np.linalg.norm(p.x - np.zeros(2)) + 0.5


def test_nearby_perturb_stays_in_box():
    """_perturb always produces points inside the bounding box."""
    problem = SimpleProblem(dim=5)
    strategy = _make_strategy(problem)
    h = Nearby(strategy, new=1, axes="all", radius=10.0)  # Very large radius
    h.__start__()

    x = np.array([4.9, -4.9, 0.0, 3.0, -3.0])
    for _ in range(20):
        new_x = h._perturb(x)
        assert np.all(new_x >= -5.0) and np.all(new_x <= 5.0), \
            f"Point {new_x} not in box [-5,5]^5"


# ---------------------------------------------------------------------------
# ClaudeHeuristic on_restart tests
# ---------------------------------------------------------------------------


def test_claude_heuristic_on_restart_clears_history():
    """on_restart clears accumulated training data."""
    from panobbgo.heuristics.claude_heuristic import ClaudeHeuristic

    problem = SimpleProblem(dim=2)
    strategy = _make_strategy(problem)
    h = ClaudeHeuristic(strategy)
    h.on_start()

    # Simulate some accumulated data
    h.X_all = np.random.rand(20, 2)
    h.y_all = np.random.rand(20)

    center = np.array([1.0, 1.0])
    h.on_restart(center, "test restart")

    assert h.X_all is None, "X_all should be cleared on restart"
    assert h.y_all is None, "y_all should be cleared on restart"


def test_claude_heuristic_on_restart_emits_candidates():
    """on_restart emits new candidates near the restart center."""
    from panobbgo.heuristics.claude_heuristic import ClaudeHeuristic

    problem = SimpleProblem(dim=2)
    strategy = _make_strategy(problem)
    h = ClaudeHeuristic(strategy, n_candidates=4)
    h.on_start()

    center = np.array([3.0, -2.0])
    h.on_restart(center, "test restart")

    pts = h.get_points(10)
    assert len(pts) == 4
    for p in pts:
        assert Point(p.x, "test") in problem.box


def test_claude_heuristic_on_restart_with_none_center():
    """on_restart with center=None should not crash."""
    from panobbgo.heuristics.claude_heuristic import ClaudeHeuristic

    problem = SimpleProblem(dim=2)
    strategy = _make_strategy(problem)
    h = ClaudeHeuristic(strategy)
    h.on_start()

    h.X_all = np.random.rand(5, 2)
    h.y_all = np.random.rand(5)

    # Should not raise
    h.on_restart(None, "none center test")

    # Data still cleared even with None center? Let's check — the impl clears then emits
    # Based on the impl: clear_output, clear data, then if center is not None emit
    assert h.X_all is None
    pts = h.get_points(10)
    assert len(pts) == 0  # No candidates emitted when center is None


# ---------------------------------------------------------------------------
# GaussianProcessHeuristic on_restart tests
# ---------------------------------------------------------------------------


def test_gp_heuristic_on_restart_clears_training_data():
    """on_restart clears all GP training data."""
    from panobbgo.heuristics.gaussian_process import GaussianProcessHeuristic

    problem = SimpleProblem(dim=2)
    strategy = _make_strategy(problem)
    h = GaussianProcessHeuristic(strategy)
    h.on_start()

    # Simulate accumulated training data
    h.X_train = np.random.rand(15, 2)
    h.y_fx_train = np.random.rand(15)
    h.y_cv_train = np.zeros(15)
    h.y_train = np.random.rand(15)
    h.best_y = 0.3

    center = np.array([2.0, -1.0])
    h.on_restart(center, "GP restart test")

    assert h.X_train is None, "X_train should be cleared"
    assert h.y_train is None, "y_train should be cleared"
    assert h.y_fx_train is None, "y_fx_train should be cleared"
    assert h.y_cv_train is None, "y_cv_train should be cleared"
    assert h.gp_model is None, "gp_model should be cleared"
    assert h.gp_constraint is None, "gp_constraint should be cleared"
    assert h.best_y == np.inf, "best_y should be reset to inf"


def test_gp_heuristic_on_restart_clears_output_queue():
    """on_restart clears any queued candidate points."""
    from panobbgo.heuristics.gaussian_process import GaussianProcessHeuristic
    from panobbgo.lib import Point

    problem = SimpleProblem(dim=2)
    strategy = _make_strategy(problem)
    h = GaussianProcessHeuristic(strategy)
    h.on_start()

    # Manually emit some stale points
    stale_pts = [Point(np.array([1.0, 2.0]), "stale")]
    h.emit(stale_pts)

    h.on_restart(np.array([0.0, 0.0]), "clear test")

    # Queue should be empty
    pts = h.get_points(10)
    assert len(pts) == 0, "Stale queue should have been cleared by on_restart"
