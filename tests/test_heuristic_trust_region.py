# -*- coding: utf8 -*-
# Copyright 2026 Harald Schilly <harald.schilly@gmail.com>
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


def test_registered_in_heuristics_package():
    import panobbgo.heuristics as H

    assert H.TrustRegionQuadratic is TrustRegionQuadratic
    assert "TrustRegionQuadratic" in H.__all__


def test_end_to_end_round_robin_rosenbrock():
    """A real strategy run: sync evaluation, the arm alone, 2-D Rosenbrock."""
    from panobbgo.lib.classic import Rosenbrock
    from panobbgo.strategies import StrategyRoundRobin

    s = StrategyRoundRobin(Rosenbrock(dims=2), testing_mode=True)
    s.config.max_eval = 150
    s.config.evaluation_method = "threaded"
    s.config.sync_evaluation = True
    s.config.ui_show = False
    s.add(TrustRegionQuadratic)
    s.start()
    assert len(s.results) == 150
    assert s.best is not None and s.best.fx < 1e-4


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
    assert third.config_overrides == blocks.config_overrides
    assert third.rng_identity == blocks.rng_identity
    assert [s.name for s in make_trust_region_strategies(["RoundRobin_TRQ", "nope"])] == ["RoundRobin_TRQ"]
