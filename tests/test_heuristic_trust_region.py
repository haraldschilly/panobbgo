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

"""Tests for the quadratic trust-region heuristic (:mod:`panobbgo.heuristics.trust_region`)."""

from __future__ import annotations

import numpy as np
import pytest

from panobbgo.heuristics.trust_region import (
    TrustRegionQuadratic,
    minimize_quadratic_in_box,
    quadratic_features,
    unpack_quadratic,
)
from panobbgo.lib import Point, Result
from tests.support import PanobbgoTestCase


# ----------------------------------------------------------------------
# The model pieces
# ----------------------------------------------------------------------


@pytest.mark.parametrize("d", [1, 2, 5])
def test_quadratic_fit_recovers_a_quadratic(d):
    rng = np.random.default_rng(1)
    A = rng.standard_normal((d, d))
    H = A @ A.T + np.eye(d)
    g = rng.standard_normal(d)
    s = rng.uniform(-1, 1, (3 * (1 + 2 * d + d * d), d))
    y = 0.7 + s @ g + 0.5 * np.einsum("ni,ij,nj->n", s, H, s)
    coef, *_ = np.linalg.lstsq(quadratic_features(s), y, rcond=None)
    c, g2, H2 = unpack_quadratic(coef, d)
    assert c == pytest.approx(0.7)
    np.testing.assert_allclose(g2, g, atol=1e-8)
    np.testing.assert_allclose(H2, H, atol=1e-8)


def test_box_minimiser_interior_and_boundary():
    H = np.diag([2.0, 4.0])
    g = np.array([-1.0, -1.0])  # Newton step (0.5, 0.25)
    s = minimize_quadratic_in_box(g, H, np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    np.testing.assert_allclose(s, [0.5, 0.25], atol=1e-6)
    s = minimize_quadratic_in_box(g, H, np.array([-1.0, -1.0]), np.array([0.1, 1.0]))
    np.testing.assert_allclose(s, [0.1, 0.25], atol=1e-6)
    # Negative curvature: the minimiser is on the boundary.
    s = minimize_quadratic_in_box(np.array([0.1, 0.0]), -np.eye(2), -np.ones(2), np.ones(2))
    assert np.max(np.abs(s)) == pytest.approx(1.0)


# ----------------------------------------------------------------------
# The heuristic, driven by hand
# ----------------------------------------------------------------------


def _result(h, x, fx, who="other"):
    return Result(Point(np.asarray(x, dtype=float), who), float(fx))


class TrustRegionTests(PanobbgoTestCase):
    def setUp(self):
        super().setUp()
        self.strategy.constraint_handler.get_penalty_value = lambda r: r.fx

    def _feed(self, h, f, points):
        """Evaluate ``points`` (``Point``) with ``f`` and hand the results back."""
        h.on_new_results([Result(p, float(f(p.x))) for p in points])

    def test_construction_checks(self):
        with pytest.raises(ValueError):
            TrustRegionQuadratic(self.strategy, radius_init=1e-9)
        with pytest.raises(ValueError):
            TrustRegionQuadratic(self.strategy, fit_span=1.0)
        with pytest.raises(ValueError):
            TrustRegionQuadratic(self.strategy, first_start="corner")
        h = TrustRegionQuadratic(self.strategy)
        assert h.on_demand and h.can_produce and not h.has_points

    def test_first_point_is_the_box_centre_then_the_coordinate_design(self):
        h = TrustRegionQuadratic(self.strategy)
        box = np.array(self.problem.box[:, :], dtype=float)
        pts = h.produce(5)
        assert len(pts) == 5
        np.testing.assert_allclose(pts[0].x, box.mean(axis=1))
        offsets = np.array([(p.x - pts[0].x) / (box[:, 1] - box[:, 0]) for p in pts[1:]])
        # the 2d coordinate-design offsets +-0.1 e_i, in some order
        np.testing.assert_allclose(np.sort(np.abs(offsets).sum(axis=1)), [0.1] * 4)
        assert len({tuple(p.x) for p in pts}) == 5
        assert all(p.who.startswith(h.name + ":") for p in pts)

    def test_batches_never_repeat_an_in_flight_point(self):
        h = TrustRegionQuadratic(self.strategy)
        seen = [tuple(p.x) for _ in range(4) for p in h.produce(3)]
        assert len(seen) == len(set(seen)) == 12

    def test_solves_a_convex_quadratic_sequentially(self):
        """At q = 1 on a quadratic the model is exact: few evaluations to high precision."""
        x_opt = np.array([0.7, -0.3])
        H = np.array([[30.0, 4.0], [4.0, 1.0]])

        def f(x):
            d = x - x_opt
            return float(d @ H @ d)

        h = TrustRegionQuadratic(self.strategy)
        best = np.inf
        for _ in range(40):
            pts = h.produce(1)
            self._feed(h, f, pts)
            best = min(best, f(pts[0].x))
        assert best < 1e-10
        assert h.n_steps > 0

    def test_batch_mode_on_a_quadratic(self):
        x_opt = np.array([0.4, 0.2])

        def f(x):
            return float(np.sum((x - x_opt) ** 2 * np.array([1.0, 10.0])))

        h = TrustRegionQuadratic(self.strategy)
        best = np.inf
        for _ in range(15):
            pts = h.produce(4)
            assert len(pts) == 4
            self._feed(h, f, pts)
            best = min(best, min(f(p.x) for p in pts))
        assert best < 1e-8

    def test_uses_foreign_points_and_moves_the_centre_to_the_archive_best(self):
        h = TrustRegionQuadratic(self.strategy)
        x_star = np.array([1.5, 2.0])
        f = lambda x: float(np.sum((x - x_star) ** 2))  # noqa: E731
        rng = np.random.default_rng(3)
        foreign = [x_star + 0.05 * rng.standard_normal(2) for _ in range(12)]
        h.on_new_results([_result(h, x, f(x)) for x in foreign])
        pts = h.produce(1)
        # a model step from the foreign best, not the box centre's first design point
        assert np.linalg.norm(pts[0].x - x_star) < 0.05
        assert h.n_steps == 1

    def test_radius_grows_up_to_radius_max(self):
        """On a linear function every step is a full success at the boundary: the radius doubles, capped."""
        h = TrustRegionQuadratic(self.strategy, radius_init=0.05, radius_max=0.15)
        f = lambda x: float(np.sum(x))  # noqa: E731
        radii = []
        for _ in range(40):
            self._feed(h, f, h.produce(1))
            radii.append(h.radius)
        assert h.n_grow >= 2
        assert max(radii) == pytest.approx(0.15)

    def test_failed_steps_shrink_the_radius_and_restart(self):
        h = TrustRegionQuadratic(self.strategy, radius_min=1e-3)
        f = lambda x: float(np.sum(x**2))  # noqa: E731
        for _ in range(5):
            self._feed(h, f, h.produce(1))
        r0 = h.radius
        # Lie from now on: every point is worse than the centre, so every step fails.
        for _ in range(60):
            pts = h.produce(1)
            h.on_new_results([Result(p, 1e6) for p in pts])
            if h.n_restarts:
                break
        assert h.n_shrink > 0 and h.radius <= r0
        assert h.n_restarts == 1 and h.radius == h.radius_init

    def test_failed_evaluations_and_nan_results(self):
        h = TrustRegionQuadratic(self.strategy)
        pts = h.produce(3)
        h.on_failed_evaluations(pts[:1])
        h.on_new_results([Result(pts[1], float("nan")), Result(pts[2], 1.0)])
        assert not h._pending
        assert len(h._F) == 1  # the NaN is not part of the model

    def test_reproducible(self):
        def run():
            h = TrustRegionQuadratic(self.init_strategy())
            h.strategy.constraint_handler.get_penalty_value = lambda r: r.fx
            xs = []
            for _ in range(20):
                pts = h.produce(2)
                h.on_new_results([Result(p, float(np.sum(np.sin(3 * p.x)) + np.sum(p.x**2))) for p in pts])
                xs.extend(p.x for p in pts)
            return np.array(xs)

        np.testing.assert_array_equal(run(), run())


def _solo_run(problem, f, n_evals, q=1):
    """Drive the arm by hand on ``problem`` with objective ``f``: ``(best fx, arm)``."""
    s = PanobbgoTestCase("__init__")
    s.problem = problem
    strategy = s.init_strategy()
    strategy.constraint_handler.get_penalty_value = lambda r: r.fx
    h = TrustRegionQuadratic(strategy)
    best, n = np.inf, 0
    while n < n_evals:
        pts = h.produce(q)
        res = [Result(p, float(f(p.x))) for p in pts]
        h.on_new_results(res)
        best = min([best] + [r.fx for r in res])
        n += len(pts)
    return best, h


class _Box:
    """A bare problem: a box and ``project``, all the arm reads."""

    def __init__(self, box):
        from panobbgo.lib.lib import BoundingBox

        self._box = BoundingBox(box)
        self.dim = len(box)

    @property
    def box(self):
        return self._box

    def project(self, x):
        return np.minimum(np.maximum(x, self._box[:, 0]), self._box[:, 1])


def test_the_model_never_collapses_into_a_subspace():
    """Review of #383, S1: the first 2d + 1 points used to lie on one axis, the min-norm fit had
    zero gradient along x2, and the arm never varied x2 (best 2.88 at (0.4, 0.0))."""
    x_opt = np.array([0.4, -1.2])
    best, _ = _solo_run(_Box([(0.0, 2.0), (-2.0, 2.0)]), lambda x: 2.0 * float(np.sum((x - x_opt) ** 2)), 40)
    assert best < 1e-8


@pytest.mark.parametrize("rotated", [False, True])
def test_separable_and_rotated_quadratics_at_d10(rotated):
    """S1: a separable d = 10 quadratic used to stall near 1e-5 while the rotated one reached 1e-24."""
    d = 10
    rng = np.random.default_rng(7)
    x_opt = rng.uniform(-3, 3, d)
    lam = np.logspace(0, 3, d)
    R = np.linalg.qr(rng.standard_normal((d, d)))[0] if rotated else np.eye(d)
    H = R @ np.diag(lam) @ R.T

    def f(x):
        z = x - x_opt
        return float(z @ H @ z)

    best, h = _solo_run(_Box([(-5.0, 5.0)] * d), f, 600)
    assert best < 1e-12, (best, h.n_steps, h.n_geometry, h.n_shrink)


def test_no_descent_shrinks_once_per_model_state():
    """S4: produce calls without a new result must not shrink the radius again and again."""
    s = PanobbgoTestCase("__init__")
    s.setUp()
    s.strategy.constraint_handler.get_penalty_value = lambda r: r.fx
    h = TrustRegionQuadratic(s.strategy)
    pts = h.produce(5)  # the box centre and the 2d coordinate-design points
    centre = pts[0].x
    h.on_new_results([Result(p, float(np.sum((p.x - centre) ** 2))) for p in pts])
    # The model's minimum is the centre itself: no descent, a failed iteration.
    for _ in range(10):
        h.produce(1)  # nothing is ever reported back
    assert h.n_shrink == 1


def _arm(**kwargs):
    s = PanobbgoTestCase("__init__")
    s.setUp()
    s.strategy.constraint_handler.get_penalty_value = lambda r: r.fx
    return TrustRegionQuadratic(s.strategy, **kwargs)


def test_rank_test_collinear_points_give_a_geometry_point_off_the_line():
    """Review of #383, S-B: many points on one line are not a model in 2-D.

    With the rank test disabled (``weak = []``) the min-norm fit has no
    gradient across the line, and the next point lies on it again.
    """
    h = _arm()
    box = np.array(h.problem.box[:, :], dtype=float)
    x2_line = box[1].mean()
    b = x2_line + 0.3 * (box[1, 1] - box[1, 0])  # the optimum is off the line
    xs = [np.array([x1, x2_line]) for x1 in np.linspace(box[0, 0] + 0.1, box[0, 1] - 0.1, 9)]
    h.on_new_results([Result(Point(x, "other"), float(x[0] ** 2 + (x[1] - b) ** 2)) for x in xs])
    c, f_c = h._center()
    model, weak = h._fit(c, f_c)
    assert model is None and len(weak) == 1
    np.testing.assert_allclose(np.abs(weak[0]), [0.0, 1.0], atol=1e-9)
    (p,) = h.produce(1)
    assert abs(p.x[1] - x2_line) > 1e-6
    assert h.n_geometry == 1 and h.n_steps == 0


def test_a_shrink_below_radius_min_inside_produce_restarts_at_once():
    """S-B: ``_maybe_restart`` runs after ``_propose``, not only on the next result."""
    h = _arm(radius_init=0.1, radius_min=0.06)
    pts = h.produce(5)  # the box centre and the coordinate design
    centre = pts[0].x
    h._H_u = np.eye(2)
    h.on_new_results([Result(p, float(np.sum((p.x - centre) ** 2))) for p in pts])
    h.produce(1)  # no descent: 0.1 -> 0.05 < radius_min, so a restart right here
    assert h.n_shrink == 1 and h.n_restarts == 1
    assert h.radius == h.radius_init
    assert not np.any(h._H_u)  # S-A: the curvature prior does not survive a restart


def test_the_curvature_prior_is_dropped_after_a_far_jump():
    h = _arm()
    h.produce(1)  # centre at the box centre, u = (0.5, 0.5)
    h._H_u = 5.0 * np.eye(2)
    h._last_center = np.array([0.0, 0.0])  # far more than fit_span radii away
    h.produce(1)
    assert not np.any(h._H_u)


def _two_basins(x):
    """Separable Styblinski–Tang in 2-D: per coordinate a local minimum at 2.7468 and the global one at -2.9035."""
    return float(0.5 * np.sum(x**4 - 16.0 * x**2 + 5.0 * x))


#: On [-5, 6]^2 the box centre (0.5, 0.5) lies in the basin of the local minimum (2.7468, 2.7468).
_TWO_BASINS_BOX = [(-5.0, 6.0)] * 2
_TWO_BASINS_LOCAL = _two_basins(np.full(2, 2.746803))  # -50.06
_TWO_BASINS_GLOBAL = _two_basins(np.full(2, -2.903534))  # -78.33


def test_a_restart_leaves_the_explored_basin(monkeypatch):
    """§69: after converging, the next centre is the best non-tabu point just outside the tabu ball.

    Its steps went back into the ball: they improved on the centre, so the
    ratio test even grew the radius, but no point in a tabu ball can become
    a centre, and the arm spent the rest of the budget there (on the wide
    preset: styblinski_tang_sep at d = 2, q = 1, 0.084 on every instance).
    Now such a centre becomes tabu too.  With that disabled, the arm stays
    at the local minimum for the whole budget.
    """
    best, h = _solo_run(_Box(_TWO_BASINS_BOX), _two_basins, 300)
    assert best < _TWO_BASINS_GLOBAL + 1e-3, (best, h.n_restarts, h.n_tabu_catch)
    assert h.n_tabu_catch >= 1

    monkeypatch.setattr(TrustRegionQuadratic, "_into_tabu", lambda self, info, u, f: False)
    best, h = _solo_run(_Box(_TWO_BASINS_BOX), _two_basins, 300)
    assert best == pytest.approx(_TWO_BASINS_LOCAL, abs=1e-6)
    assert h.n_restarts >= 1


def _step_towards_a_tabu_ball():
    """An arm with one step in flight and a tabu ball that holds the step but not its centre."""
    h = _arm()
    pts = h.produce(5)  # the box centre and the coordinate design
    centre = pts[0].x
    h.on_new_results([Result(p, float(np.sum((p.x - centre - 0.3) ** 2))) for p in pts])
    (step,) = h.produce(1)
    info = h._pending[step.who]
    assert info["kind"] == "step"
    c, u = info["center"], info["u"]
    h._tabu.append(u + 0.9 * h.radius_init * np.sign(u - c))  # beyond the step, seen from the centre
    assert h._in_tabu(u) and not h._in_tabu(c)
    return h, step, info


def test_a_step_that_improves_into_a_tabu_ball_makes_its_centre_tabu():
    h, step, info = _step_towards_a_tabu_ball()
    c = info["center"]
    h.radius = 0.25 * h.radius_init  # so that the reset below is visible
    h._need_geometry = True
    h._H_u = np.eye(2)
    h.on_new_results([Result(step, info["f_center"] - 1.0)])
    assert h.n_tabu_catch == 1 and h._in_tabu(c) and len(h._tabu) == 2
    assert h.radius == h.radius_init and not np.any(h._H_u) and not h._need_geometry
    assert h.n_grow == 0  # the step's "success" does not grow the radius


def _extra_step(h, info, u):
    """Another step of ``info``'s centre in flight (q > 1), towards ``u``."""
    who = h.new_who()
    h._pending[who] = dict(info, u=np.array(u, copy=True), primary=False)
    return Point(h._to_x(u), who)


def test_a_second_improving_step_of_the_same_centre_adds_no_second_ball():
    h, step, info = _step_towards_a_tabu_ball()
    c, u = info["center"], info["u"]
    other = _extra_step(h, info, c + 1.2 * (u - c))  # further into the same tabu ball
    assert h._in_tabu(h._to_u(other.x))
    h.on_new_results([Result(step, info["f_center"] - 1.0), Result(other, info["f_center"] - 2.0)])
    assert h.n_tabu_catch == 1 and len(h._tabu) == 2


def test_a_stale_catch_from_an_old_centre_keeps_the_current_radius():
    """q > 1: the step's centre is no longer the centre; it becomes tabu, the current centre keeps its state."""
    h, step, info = _step_towards_a_tabu_ball()
    c, u = info["center"], info["u"]
    far = np.clip(c - 0.35 * np.sign(u - c), 0.0, 1.0)  # well outside every tabu ball
    h.on_new_results([Result(Point(h._to_x(far), "other"), info["f_center"] - 100.0)])
    current, _ = h._center()
    np.testing.assert_allclose(current, far)
    h.radius = 0.25 * h.radius_init
    h._need_geometry = True
    h._H_u = np.eye(2)
    h.on_new_results([Result(step, info["f_center"] - 1.0)])
    assert h.n_tabu_catch == 1 and h._in_tabu(c)
    assert h.radius == 0.25 * h.radius_init and h._need_geometry and np.any(h._H_u)


def test_a_step_into_a_tabu_ball_that_does_not_improve_is_an_ordinary_failure():
    h, step, info = _step_towards_a_tabu_ball()
    r0 = h.radius
    h.on_new_results([Result(step, info["f_center"] + 1.0)])
    assert h.n_tabu_catch == 0 and len(h._tabu) == 1
    assert h.radius < r0 or h._need_geometry  # the ratio test ran: shrink, or geometry on a thin model


def test_a_step_from_a_centre_that_became_tabu_since_does_not_move_the_radius():
    """At q > 1 steps of an abandoned centre are still in flight; they speak for nothing now."""
    h, step, info = _step_towards_a_tabu_ball()
    h._tabu.append(np.array(info["center"], copy=True))
    h.radius = info["radius"]
    h.on_new_results([Result(step, info["f_center"] + 1.0)])
    assert h.radius == info["radius"] and not h._need_geometry and h.n_shrink == 0


def test_stop_clears_the_pending_bookkeeping():
    h = _arm()
    h.produce(3)
    assert len(h._pending) == 3
    h.__stop__()
    assert not h._pending
    assert h.produce(1) == []


def test_points_handed_back_undispatched_are_produced_again():
    s = PanobbgoTestCase("__init__")
    s.setUp()
    h = TrustRegionQuadratic(s.strategy)
    pts = h.produce(3)
    with h._output.mutex:
        h._output.queue.extendleft(reversed(pts[1:]))
    again = h.produce(3)
    assert [p.who for p in again[:2]] == [p.who for p in pts[1:]]
    assert len(again) == 3 and again[2].who not in {p.who for p in pts}


def test_registered_in_heuristics_package():
    import panobbgo.heuristics as H

    assert H.TrustRegionQuadratic is TrustRegionQuadratic
    assert "TrustRegionQuadratic" in H.__all__


def test_end_to_end_round_robin_rosenbrock():
    """A real strategy run: the arm alone on 2-D Rosenbrock, sequential on the virtual clock.

    One virtual worker makes the run a pure function of the seed; the
    threaded pool splits result batches by timing, and its best value
    varied between 1e-4 and 5e-3 over repeated runs.
    """
    from panobbgo.lib.classic import Rosenbrock
    from panobbgo.strategies import StrategyRoundRobin

    def run():
        s = StrategyRoundRobin(Rosenbrock(dims=2), testing_mode=True)
        s.config.max_eval = 150
        s.config.evaluation_method = "virtual"
        s.config.virtual_workers = 1
        s.config.ui_show = False
        s.add(TrustRegionQuadratic)
        s.start()
        assert len(s.results) == 150
        assert s.best is not None
        return float(s.best.fx)

    best = run()
    assert best < 1e-4
    assert run() == best


def test_opt_in_specs_are_not_in_the_registry_of_record():
    from panobbgo.harness_ioh import TRUST_REGION_NAMES, make_ioh_strategies, make_trust_region_strategies

    record = {s.name for s in make_ioh_strategies()}
    specs = make_trust_region_strategies()
    assert [s.name for s in specs] == list(TRUST_REGION_NAMES)
    assert not record & set(TRUST_REGION_NAMES)
    blocks = next(s for s in make_ioh_strategies() if s.name == "Blocks_warm_CMAES_JSO")
    third = next(s for s in specs if s.name == "Blocks_warm_CMAES_JSO_TRQ")
    assert third.heuristics[:2] == blocks.heuristics
    assert third.heuristics[2][0] is TrustRegionQuadratic
    # the portfolio's config; its dim/budget gate never applies with a third arm (DISCOVERY §72)
    assert third.config_overrides == blocks.config_overrides
    assert third.rng_identity == blocks.rng_identity
    assert [s.name for s in make_trust_region_strategies(["RoundRobin_TRQ", "nope"])] == ["RoundRobin_TRQ"]
    # COBYQA alone, as DISCOVERY §66.1 ran it (``RR_COBYQA``): the bridge with its defaults.
    from panobbgo.heuristics import COBYQA

    [cobyqa] = make_trust_region_strategies(["RoundRobin_COBYQA"])
    assert cobyqa.heuristics == [(COBYQA, {})]
    # §69.5's start radius: RoundRobin_TRQ at radius_init = 0.5 on RoundRobin_TRQ's RNG streams.
    rr, r05 = make_trust_region_strategies(["RoundRobin_TRQ", "RoundRobin_TRQ_r05"])
    assert r05.heuristics == [(TrustRegionQuadratic, {"radius_init": 0.5})]
    assert r05.strategy_class is rr.strategy_class and r05.rng_identity == rr.rng_identity == "RoundRobin_TRQ"


def test_radius_init_half_is_accepted_at_the_default_radius_max():
    """``RoundRobin_TRQ_r05``'s start radius equals the default ``radius_max``, which the constructor allows."""
    h = _arm(radius_init=0.5)
    assert h.radius == h.radius_init == h.radius_max == 0.5
    assert h.can_produce
    with pytest.raises(ValueError):
        _arm(radius_init=0.6)
