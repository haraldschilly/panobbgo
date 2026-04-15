# -*- coding: utf8 -*-
# Copyright 2012-2026 Harald Schilly <harald.schilly@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Tests for the sensitivity-aware Nearby heuristic.

The key property being tested: when sensitivity importance scores are provided,
perturbations along important dimensions should be larger than along unimportant
dimensions (on average over many trials).
"""

import numpy as np
from unittest import mock

from panobbgo.heuristics.nearby import Nearby
from panobbgo.lib import Point, Result, Problem
from panobbgo.config import Config


class FlatProblem(Problem):
    """Simple 4D problem with box [-5, 5]^4 for testing."""

    def __init__(self, dim=4):
        box = [(-5.0, 5.0)] * dim
        super().__init__(box=box)

    def eval(self, x):
        return float(np.sum(x**2))


def _make_strategy(dim=4):
    """Create a mock strategy with a FlatProblem."""
    problem = FlatProblem(dim=dim)
    config = Config(parse_args=False, testing_mode=True)
    strategy = mock.MagicMock()
    strategy.problem = problem
    strategy.config = config
    return strategy, problem


# ---------------------------------------------------------------------------
# Basic construction
# ---------------------------------------------------------------------------

def test_nearby_default_no_sensitivity():
    """Nearby starts with no sensitivity data (importance=None)."""
    strategy, _ = _make_strategy()
    h = Nearby(strategy)
    h.__start__()
    assert h._importance is None


def test_nearby_sensitivity_scale_zero_disables():
    """sensitivity_scale=0 means sensitivity weights are ignored even when set."""
    strategy, problem = _make_strategy()
    h = Nearby(strategy, sensitivity_scale=0.0)
    h.__start__()
    importance = np.array([1.0, 0.0, 1.0, 0.0])
    h.on_new_sensitivity(importance)
    # _importance is stored but _perturbation_weights should return None
    assert h._importance is not None
    assert h._perturbation_weights() is None


def test_nearby_on_new_sensitivity_stores_importance():
    """on_new_sensitivity stores the importance array."""
    strategy, problem = _make_strategy(dim=3)
    h = Nearby(strategy)
    h.__start__()
    importance = np.array([0.8, 0.1, 0.5])
    h.on_new_sensitivity(importance)
    np.testing.assert_array_equal(h._importance, importance)


# ---------------------------------------------------------------------------
# _perturbation_weights
# ---------------------------------------------------------------------------

def test_perturbation_weights_normalised_mean():
    """Perturbation weights should have mean == 1 to preserve overall magnitude."""
    strategy, problem = _make_strategy(dim=4)
    h = Nearby(strategy, sensitivity_scale=1.0)
    h.__start__()
    importance = np.array([1.0, 0.5, 0.2, 0.8])
    h.on_new_sensitivity(importance)
    weights = h._perturbation_weights()
    assert weights is not None
    assert abs(weights.mean() - 1.0) < 1e-9


def test_perturbation_weights_preserves_ordering():
    """More important dims get higher weights."""
    strategy, problem = _make_strategy(dim=4)
    h = Nearby(strategy, sensitivity_scale=1.0)
    h.__start__()
    # dim 0 most important, dim 2 least important
    importance = np.array([1.0, 0.6, 0.1, 0.7])
    h.on_new_sensitivity(importance)
    weights = h._perturbation_weights()
    assert weights is not None
    assert weights[0] > weights[1]  # 1.0 > 0.6
    assert weights[1] < weights[3]  # 0.6 < 0.7
    assert weights[2] < weights[1]  # 0.1 < 0.6


def test_perturbation_weights_all_zero_importance():
    """All-zero importance should not cause division errors (min-clipped)."""
    strategy, problem = _make_strategy(dim=3)
    h = Nearby(strategy, sensitivity_scale=1.0)
    h.__start__()
    importance = np.array([0.0, 0.0, 0.0])
    h.on_new_sensitivity(importance)
    weights = h._perturbation_weights()
    assert weights is not None
    # All values clipped to 1e-3, so all weights should be equal
    assert weights is not None
    np.testing.assert_allclose(weights, np.ones(3), rtol=1e-6)


# ---------------------------------------------------------------------------
# on_new_best with sensitivity — axes="all"
# ---------------------------------------------------------------------------

def test_on_new_best_all_axes_no_sensitivity_in_box():
    """Without sensitivity, generated points should stay inside the box."""
    strategy, problem = _make_strategy(dim=4)
    h = Nearby(strategy, cap=15, radius=0.5, axes="all", new=10)
    h.__start__()
    x_best = np.zeros(4)
    best = Result(Point(x_best, "test"), 0.0)
    h.on_new_best(best)
    points = h.get_points(20)
    assert len(points) == 10
    for p in points:
        assert Point(p.x, "t") in problem.box, f"Point {p.x} outside box"


def test_on_new_best_all_axes_sensitivity_in_box():
    """With sensitivity, generated points should still stay inside the box."""
    strategy, problem = _make_strategy(dim=4)
    h = Nearby(strategy, cap=25, radius=0.5, axes="all", new=20)
    h.__start__()
    importance = np.array([1.0, 0.0, 0.5, 0.8])
    h.on_new_sensitivity(importance)
    x_best = np.zeros(4)
    best = Result(Point(x_best, "test"), 0.0)
    h.on_new_best(best)
    points = h.get_points(30)
    assert len(points) == 20
    for p in points:
        assert Point(p.x, "t") in problem.box, f"Point {p.x} outside box"


def test_on_new_best_all_axes_sensitivity_biases_important_dims():
    """
    With sensitivity_scale > 0, the average absolute displacement along important
    dimensions should exceed that of unimportant dimensions when axes='all'.
    """
    np.random.seed(42)

    strategy, problem = _make_strategy(dim=2)
    h = Nearby(strategy, radius=0.3, axes="all", new=1, sensitivity_scale=2.0)
    h.__start__()

    # dim 0 fully important, dim 1 fully unimportant
    importance = np.array([1.0, 0.001])
    h.on_new_sensitivity(importance)

    N = 500
    displacements = np.zeros((N, 2))
    x_best = np.array([0.0, 0.0])
    best = Result(Point(x_best, "test"), 0.0)

    for i in range(N):
        h.on_new_best(best)
        pts = h.get_points(5)
        if pts:
            displacements[i] = np.abs(pts[0].x - x_best)

    mean_disp = displacements.mean(axis=0)
    # Important dim 0 should have much larger mean displacement than dim 1
    assert mean_disp[0] > mean_disp[1] * 3, (
        f"Expected dim 0 displacement >> dim 1, got {mean_disp}"
    )


# ---------------------------------------------------------------------------
# on_new_best with sensitivity — axes="one"
# ---------------------------------------------------------------------------

def test_on_new_best_one_axis_sensitivity_biases_axis_selection():
    """
    With sensitivity (axes='one'), important dimensions should be selected more often.
    """
    np.random.seed(123)
    strategy, problem = _make_strategy(dim=4)
    h = Nearby(strategy, radius=0.1, axes="one", new=1, sensitivity_scale=1.0)
    h.__start__()

    # dim 0 dominant, rest near-zero
    importance = np.array([1.0, 0.01, 0.01, 0.01])
    h.on_new_sensitivity(importance)

    x_best = np.zeros(4)
    best = Result(Point(x_best, "test"), 0.0)

    counts = np.zeros(4)
    N = 400
    for _ in range(N):
        h.on_new_best(best)
        pts = h.get_points(5)
        if pts:
            changed = np.argmax(np.abs(pts[0].x - x_best))
            counts[changed] += 1

    # dim 0 should be selected much more frequently than others
    freq = counts / counts.sum()
    assert freq[0] > 0.6, f"dim 0 frequency {freq[0]:.2f} expected > 0.6"


# ---------------------------------------------------------------------------
# on_restart
# ---------------------------------------------------------------------------

def test_on_restart_with_sensitivity_in_box():
    """on_restart with sensitivity should keep points in box."""
    strategy, problem = _make_strategy(dim=4)
    h = Nearby(strategy, cap=10, radius=0.5, axes="all", new=5)
    h.__start__()
    importance = np.array([1.0, 0.5, 0.2, 0.8])
    h.on_new_sensitivity(importance)
    center = np.zeros(4)
    h.on_restart(center, "test restart")
    points = h.get_points(10)
    assert len(points) == 5
    for p in points:
        assert Point(p.x, "t") in problem.box


def test_on_restart_none_center_no_crash():
    """on_restart with None center should not crash."""
    strategy, problem = _make_strategy(dim=2)
    h = Nearby(strategy, radius=0.1, axes="all", new=3)
    h.__start__()
    h.on_restart(None, "test")  # should not raise
    points = h.get_points(5)
    assert len(points) == 0  # nothing emitted


# ---------------------------------------------------------------------------
# Sensitivity updates are idempotent
# ---------------------------------------------------------------------------

def test_sensitivity_update_updates_weights():
    """Updating sensitivity mid-run should immediately affect new perturbations."""
    np.random.seed(0)
    strategy, problem = _make_strategy(dim=2)
    h = Nearby(strategy, radius=0.5, axes="all", new=50, sensitivity_scale=3.0)
    h.__start__()
    x_best = np.zeros(2)
    best = Result(Point(x_best, "test"), 0.0)

    # Phase 1: dim 0 important
    h.on_new_sensitivity(np.array([1.0, 0.001]))
    h.on_new_best(best)
    pts1 = h.get_points(60)
    disp1 = np.mean([np.abs(p.x - x_best) for p in pts1], axis=0)

    # Phase 2: dim 1 important (swap)
    h.on_new_sensitivity(np.array([0.001, 1.0]))
    h.on_new_best(best)
    pts2 = h.get_points(60)
    disp2 = np.mean([np.abs(p.x - x_best) for p in pts2], axis=0)

    assert disp1[0] > disp1[1], "Phase 1: dim 0 should dominate"
    assert disp2[1] > disp2[0], "Phase 2: dim 1 should dominate"


# ---------------------------------------------------------------------------
# StrategySpec analyzers support in benchmark
# ---------------------------------------------------------------------------

def test_strategy_spec_analyzers_field():
    """StrategySpec.analyzers defaults to empty list and is used by create_strategy."""
    from panobbgo.benchmark import StrategySpec
    from panobbgo.strategies import StrategyRewarding
    from panobbgo.heuristics import Random
    from panobbgo.lib.classic import Rosenbrock
    from panobbgo.analyzers import Sensitivity

    spec = StrategySpec(
        name="test",
        strategy_class=StrategyRewarding,
        heuristics=[(Random, {})],
        analyzers=[(Sensitivity, {"update_interval": 10})],
    )
    problem = Rosenbrock(dims=2)
    strategy = spec.create_strategy(problem)

    # Sensitivity should be registered as an extra analyzer
    analyzer_names = [a.name for a in strategy.analyzers]
    assert "Sensitivity" in analyzer_names, (
        f"Expected 'Sensitivity' in {analyzer_names}"
    )


def test_strategy_spec_no_analyzers_default():
    """StrategySpec without analyzers field defaults to empty list.

    Note: the four required analyzers (Best, Grid, Splitter, Convergence) are
    only added during ``strategy.initialize()`` / ``strategy.start()``, not by
    ``create_strategy()``.  This test only checks that the spec round-trips and
    that no *extra* analyzer is registered before initialization.
    """
    from panobbgo.benchmark import StrategySpec
    from panobbgo.strategies import StrategyRewarding
    from panobbgo.heuristics import Random
    from panobbgo.lib.classic import Rosenbrock

    spec = StrategySpec(
        name="test",
        strategy_class=StrategyRewarding,
        heuristics=[(Random, {})],
    )
    assert spec.analyzers == []
    problem = Rosenbrock(dims=2)
    strategy = spec.create_strategy(problem)
    # No extra analyzers were registered (required ones come via initialize())
    assert strategy.analyzers == []
