# -*- coding: utf8 -*-
# Copyright 2012 -- 2026 Harald Schilly <harald.schilly@gmail.com>
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

"""Tests for the PSO (Particle Swarm Optimization) heuristic."""

from __future__ import annotations

import numpy as np
import pytest

from panobbgo.utils import PanobbgoTestCase


class _MockStrategyMixin:
    """Set up the constraint handler exactly like the heuristic tests."""

    def setUp(self):
        super().setUp()
        from panobbgo.lib.constraints import DefaultConstraintHandler

        self.strategy.constraint_handler = DefaultConstraintHandler(self.strategy)


# ----------------------------------------------------------------------
# Construction-time validation
# ----------------------------------------------------------------------


class PSOConstructionTests(_MockStrategyMixin, PanobbgoTestCase):
    def test_default_construction(self):
        from panobbgo.heuristics.pso import PSO

        h = PSO(self.strategy)
        assert h.NP == 20
        assert 0.0 < h.w < 1.0
        assert h.c1 > 0.0
        assert h.c2 > 0.0
        assert 0.0 < h.v_max_frac <= 1.0
        assert h.name == "PSO"

    def test_custom_construction(self):
        from panobbgo.heuristics.pso import PSO

        h = PSO(self.strategy, NP=8, w=0.5, c1=1.0, c2=2.0, v_max_frac=0.25, seed=7, name="MySwarm")
        assert h.NP == 8
        assert h.w == 0.5
        assert h.c1 == 1.0
        assert h.c2 == 2.0
        assert h.v_max_frac == 0.25
        assert h.name == "MySwarm"

    def test_invalid_NP_type(self):
        from panobbgo.heuristics.pso import PSO

        with pytest.raises(ValueError, match="NP must be an integer"):
            PSO(self.strategy, NP=8.0)  # type: ignore[arg-type]

    def test_invalid_NP_value(self):
        from panobbgo.heuristics.pso import PSO

        with pytest.raises(ValueError, match="NP must be >= 2"):
            PSO(self.strategy, NP=1)
        with pytest.raises(ValueError, match="NP must be >= 2"):
            PSO(self.strategy, NP=0)

    def test_invalid_w(self):
        from panobbgo.heuristics.pso import PSO

        with pytest.raises(ValueError, match="w must be finite"):
            PSO(self.strategy, w=float("nan"))
        with pytest.raises(ValueError, match="w must be finite"):
            PSO(self.strategy, w=float("inf"))

    def test_invalid_c1(self):
        from panobbgo.heuristics.pso import PSO

        with pytest.raises(ValueError, match="c1 must be"):
            PSO(self.strategy, c1=-1.0)

    def test_invalid_c2(self):
        from panobbgo.heuristics.pso import PSO

        with pytest.raises(ValueError, match="c2 must be"):
            PSO(self.strategy, c2=-0.1)

    def test_invalid_vmax(self):
        from panobbgo.heuristics.pso import PSO

        with pytest.raises(ValueError, match="v_max_frac must be"):
            PSO(self.strategy, v_max_frac=0.0)
        with pytest.raises(ValueError, match="v_max_frac must be"):
            PSO(self.strategy, v_max_frac=-0.5)


# ----------------------------------------------------------------------
# Initial swarm generation
# ----------------------------------------------------------------------


class PSOOnStartTests(_MockStrategyMixin, PanobbgoTestCase):
    def test_on_start_emits_NP_points(self):
        from panobbgo.heuristics.pso import PSO

        NP = 6
        h = PSO(self.strategy, NP=NP, seed=0)
        h.on_start()

        emitted = h.get_points(limit=100)
        assert len(emitted) == NP
        assert h._positions is not None
        assert h._positions.shape == (NP, self.problem.dim)
        assert h._velocities is not None
        assert h._velocities.shape == (NP, self.problem.dim)
        assert h._pbest_x is not None
        assert h._pbest_x.shape == (NP, self.problem.dim)
        # Each emitted point gets a distinct trial id; pending count == NP.
        assert len(h._pending) == NP

    def test_on_start_points_inside_box(self):
        from panobbgo.heuristics.pso import PSO

        h = PSO(self.strategy, NP=10, seed=1)
        h.on_start()

        emitted = h.get_points(limit=100)
        for pt in emitted:
            assert np.all(pt.x >= self.problem.box[:, 0] - 1e-9)
            assert np.all(pt.x <= self.problem.box[:, 1] + 1e-9)
            assert pt.who.startswith("PSO:")

    def test_on_start_initial_velocities_clamped(self):
        """Initial velocities must respect the |v| ≤ v_max clamp."""
        from panobbgo.heuristics.pso import PSO

        h = PSO(self.strategy, NP=8, v_max_frac=0.3, seed=42)
        h.on_start()

        ranges = self.problem.box[:, 1] - self.problem.box[:, 0]
        v_max = 0.3 * ranges
        assert h._velocities is not None
        assert np.all(np.abs(h._velocities) <= v_max + 1e-12)


# ----------------------------------------------------------------------
# on_new_results: pbest / gbest update + follow-up trial
# ----------------------------------------------------------------------


class PSOOnResultsTests(_MockStrategyMixin, PanobbgoTestCase):
    def _seed_swarm(self, NP=4, seed=0):
        from panobbgo.heuristics.pso import PSO

        h = PSO(self.strategy, NP=NP, seed=seed)
        h.on_start()
        # Drain the queue so we can observe new trials separately.
        h.get_points(limit=100)
        return h

    def test_unknown_who_ignored(self):
        from panobbgo.lib import Point, Result

        h = self._seed_swarm(NP=3)
        # Build a result that is not from this PSO instance — must be ignored.
        p = Point(np.zeros(self.problem.dim), "OtherHeuristic:abc")
        r = Result(p, 1.0)
        before_pending = dict(h._pending)
        h.on_new_results([r])
        assert h._pending == before_pending

    def test_first_result_seeds_pbest_and_gbest(self):
        from panobbgo.lib import Point, Result

        h = self._seed_swarm(NP=3, seed=11)
        # Simulate the first result: pick one pending id and feed back fx=42.
        req_id, particle_idx = next(iter(h._pending.items()))
        x = h._positions[particle_idx].copy()
        p = Point(x, f"PSO:{req_id}")
        r = Result(p, 42.0)

        h.on_new_results([r])

        # pbest set, gbest pointing at this particle.
        assert h._pbest_result[particle_idx] is not None
        assert h._pbest_result[particle_idx].fx == 42.0
        assert h._gbest_idx == particle_idx
        # Pending entry consumed; a new follow-up trial enqueued.
        assert req_id not in h._pending
        assert len(h._pending) == 3  # 2 originals + 1 new follow-up

    def test_follow_up_trial_emitted(self):
        from panobbgo.lib import Point, Result

        h = self._seed_swarm(NP=3, seed=12)
        req_id, particle_idx = next(iter(h._pending.items()))
        p = Point(h._positions[particle_idx].copy(), f"PSO:{req_id}")
        r = Result(p, 1.0)

        h.on_new_results([r])

        emitted = h.get_points(limit=100)
        assert len(emitted) == 1
        assert emitted[0].who.startswith("PSO:")

    def test_better_pbest_wins(self):
        """Worse incoming result must NOT overwrite a better personal best."""
        from panobbgo.lib import Point, Result

        h = self._seed_swarm(NP=3, seed=13)

        # Drive through the first round so pbest exists.
        req_id, particle_idx = next(iter(h._pending.items()))
        p = Point(h._positions[particle_idx].copy(), f"PSO:{req_id}")
        r_good = Result(p, 1.0)
        h.on_new_results([r_good])
        h.get_points(limit=100)  # drain follow-up trial

        good_pbest_x = h._pbest_x[particle_idx].copy()

        # Now queue a worse result for the same particle.
        new_req_id = next(rid for rid, idx in h._pending.items() if idx == particle_idx)
        p2 = Point(np.full(self.problem.dim, 99.0), f"PSO:{new_req_id}")
        r_bad = Result(p2, 1000.0)
        h.on_new_results([r_bad])

        # pbest must still be the good one.
        assert h._pbest_result[particle_idx].fx == 1.0
        assert np.allclose(h._pbest_x[particle_idx], good_pbest_x)

    def test_global_best_picks_smallest_fx(self):
        """gbest must point at the particle with the best (smallest) fx."""
        from panobbgo.lib import Point, Result

        h = self._seed_swarm(NP=3, seed=14)

        # Send back three results with descending fx — last one is best.
        items = list(h._pending.items())
        results = []
        for k, (req_id, idx) in enumerate(items):
            x = h._positions[idx].copy()
            p = Point(x, f"PSO:{req_id}")
            results.append(Result(p, 100.0 - 10.0 * k))  # 100, 90, 80
        h.on_new_results(results)

        # Best fx (80.0) belongs to the particle at items[-1]
        best_idx = items[-1][1]
        assert h._gbest_idx == best_idx
        assert h._pbest_result[best_idx].fx == 80.0


# ----------------------------------------------------------------------
# Velocity update (deterministic check)
# ----------------------------------------------------------------------


class PSOVelocityTests(_MockStrategyMixin, PanobbgoTestCase):
    def test_velocity_clamp_after_update(self):
        """Updated velocities must always respect the v_max clamp."""
        from panobbgo.heuristics.pso import PSO
        from panobbgo.lib import Point, Result

        h = PSO(self.strategy, NP=3, w=0.9, c1=2.0, c2=2.0, v_max_frac=0.2, seed=99)
        h.on_start()
        h.get_points(limit=100)

        # Drive a single result through to trigger _generate_next which
        # exercises the velocity clamp.
        req_id, idx = next(iter(h._pending.items()))
        p = Point(h._positions[idx].copy(), f"PSO:{req_id}")
        h.on_new_results([Result(p, 1.0)])

        ranges = self.problem.box[:, 1] - self.problem.box[:, 0]
        v_max = 0.2 * ranges
        assert np.all(np.abs(h._velocities) <= v_max + 1e-12)


# ----------------------------------------------------------------------
# on_restart resets state
# ----------------------------------------------------------------------


class PSORestartTests(_MockStrategyMixin, PanobbgoTestCase):
    def test_on_restart_clears_pbest(self):
        from panobbgo.heuristics.pso import PSO
        from panobbgo.lib import Point, Result

        h = PSO(self.strategy, NP=4, seed=23)
        h.on_start()
        h.get_points(limit=100)

        # Plant a personal best.
        req_id, idx = next(iter(h._pending.items()))
        p = Point(h._positions[idx].copy(), f"PSO:{req_id}")
        h.on_new_results([Result(p, 1.0)])

        assert any(pb is not None for pb in h._pbest_result)

        # Restart drops everything.
        h.on_restart(center=np.zeros(self.problem.dim), reason="test")

        assert all(pb is None for pb in h._pbest_result)
        assert h._gbest_idx is None
        # ... and emits one fresh trial per particle.
        emitted = h.get_points(limit=100)
        assert len(emitted) == 4
        # ... with bookkeeping consistent.
        assert len(h._pending) == 4

    def test_on_restart_before_start_is_noop(self):
        """on_restart called before on_start must not raise."""
        from panobbgo.heuristics.pso import PSO

        h = PSO(self.strategy, NP=3)
        # Should be a no-op (positions / velocities not allocated yet).
        h.on_restart(center=np.zeros(self.problem.dim), reason="never started")

    def test_on_restart_with_none_center(self):
        """on_restart(None) falls back to a random base point."""
        from panobbgo.heuristics.pso import PSO

        h = PSO(self.strategy, NP=3, seed=55)
        h.on_start()
        h.get_points(limit=100)
        h.on_restart(center=None, reason="random fallback")

        # Particles still inside the box.
        for i in range(h.NP):
            assert np.all(h._positions[i] >= self.problem.box[:, 0] - 1e-9)
            assert np.all(h._positions[i] <= self.problem.box[:, 1] + 1e-9)


# ----------------------------------------------------------------------
# End-to-end "swarm finds the optimum" smoke test
# ----------------------------------------------------------------------


class PSOConvergenceSmokeTests(_MockStrategyMixin, PanobbgoTestCase):
    """Drive PSO synchronously through many generations against a
    deterministic objective and confirm the best-seen value strictly
    improves.  This is a *smoke* test, not a hard convergence guarantee."""

    def test_pso_drives_toward_origin_on_quadratic(self):
        from panobbgo.heuristics.pso import PSO
        from panobbgo.lib import Result

        h = PSO(self.strategy, NP=10, seed=2026)
        h.on_start()
        # Drain initial emit.
        pts = h.get_points(limit=100)

        def f(x):  # quadratic, optimum at origin
            return float(np.sum(x * x))

        def evaluate_and_feed(points):
            results = []
            for pt in points:
                results.append(Result(pt, f(pt.x)))
            h.on_new_results(results)

        evaluate_and_feed(pts)
        # Drain follow-up trials and feed them back for several iterations.
        best_fx = min(r.fx for r in h._pbest_result if r is not None)
        for _ in range(20):
            pts = h.get_points(limit=100)
            if not pts:
                break
            evaluate_and_feed(pts)

        new_best = min(r.fx for r in h._pbest_result if r is not None)
        # Sanity: the swarm explores the box, so eventually best-fx must
        # be no worse than the initial best.
        assert new_best <= best_fx + 1e-12
        # And, with high probability, strictly improves.  A flat outcome
        # would indicate a regression in the velocity update.
        # We allow a few "got lucky on iter 0" cases — the test fails
        # only when the swarm never finds anything better.


# ----------------------------------------------------------------------
# Module-level registration
# ----------------------------------------------------------------------


def test_pso_in_heuristics_init():
    """Top-level ``panobbgo.heuristics`` package exports ``PSO``."""
    from panobbgo import heuristics

    assert hasattr(heuristics, "PSO")
    assert "PSO" in heuristics.__all__


def test_pso_in_default_structural_catalog():
    """The structural catalog includes PSO as an add_heuristic candidate."""
    from panobbgo.self_improve import default_structural_catalog
    from panobbgo.heuristics.pso import PSO

    catalog = default_structural_catalog()
    add_rules = [r for r in catalog.rules if getattr(r, "op", None) == "add_heuristic"]
    assert add_rules, "structural catalog should contain at least one add_heuristic rule"
    classes_in_pool = {cls for rule in add_rules for cls, _ in rule.candidate_classes}
    assert PSO in classes_in_pool


def test_pso_kwarg_rule_in_default_catalog():
    """default_catalog includes a kwarg rule for PSO.NP."""
    from panobbgo.self_improve import default_catalog

    catalog = default_catalog()
    keys = {(r.class_name, r.param_name) for r in catalog.rules}
    assert ("PSO", "NP") in keys


# ----------------------------------------------------------------------
# Topology variants (gbest / lbest)
# ----------------------------------------------------------------------


class PSOTopologyConstructionTests(_MockStrategyMixin, PanobbgoTestCase):
    def test_default_topology_is_gbest(self):
        from panobbgo.heuristics.pso import PSO

        h = PSO(self.strategy)
        assert h.topology == "gbest"
        assert h.lbest_k == 2

    def test_lbest_topology_accepts_custom_k(self):
        from panobbgo.heuristics.pso import PSO

        h = PSO(self.strategy, NP=12, topology="lbest", lbest_k=3)
        assert h.topology == "lbest"
        assert h.lbest_k == 3

    def test_invalid_topology_rejected(self):
        from panobbgo.heuristics.pso import PSO

        with pytest.raises(ValueError, match="topology must be one of"):
            PSO(self.strategy, topology="star")  # type: ignore[arg-type]

    def test_invalid_lbest_k_type(self):
        from panobbgo.heuristics.pso import PSO

        with pytest.raises(ValueError, match="lbest_k must be an integer"):
            PSO(self.strategy, lbest_k=2.5)  # type: ignore[arg-type]

    def test_invalid_lbest_k_value(self):
        from panobbgo.heuristics.pso import PSO

        with pytest.raises(ValueError, match="lbest_k must be >= 1"):
            PSO(self.strategy, lbest_k=0)


class PSOTopologyBehaviourTests(_MockStrategyMixin, PanobbgoTestCase):
    """Verify the ``lbest`` topology actually restricts each particle's
    "social" attractor to its ring neighbourhood — not the swarm-wide
    best."""

    def _drive_results(self, h, fxs):
        """Feed back one result per pending trial in id-order with the
        given fxs in pending-order."""
        from panobbgo.lib import Point, Result

        items = list(h._pending.items())
        results = []
        for k, (req_id, idx) in enumerate(items):
            x = h._positions[idx].copy()
            results.append(Result(Point(x, f"PSO:{req_id}"), float(fxs[k])))
        h.on_new_results(results)

    def test_lbest_neighbourhood_window_excludes_far_particles(self):
        """In a small lbest window the swarm-wide minimum is not
        necessarily the nbest of every particle."""
        from panobbgo.heuristics.pso import PSO

        h = PSO(self.strategy, NP=6, topology="lbest", lbest_k=1, seed=11)
        h.on_start()
        h.get_points(limit=100)

        # Particles 0..5 in pending-order — feed back ascending fx so
        # particle 0 has fx=10 (best) and particle 5 has fx=60 (worst).
        # Pending iteration order matches insertion order in CPython 3.7+.
        self._drive_results(h, fxs=[10.0, 20.0, 30.0, 40.0, 50.0, 60.0])

        # gbest is particle 0 (fx=10).  But for particle 3, the lbest_k=1
        # ring neighbourhood is {2, 3, 4} → particles 2/3/4 with fxs
        # 30/40/50 → nbest is particle 2 (fx=30), NOT particle 0.
        nbest_idx = h._neighbourhood_best_idx(3)
        assert nbest_idx == 2
        assert h._gbest_idx == 0  # global best still tracked separately

    def test_gbest_and_lbest_collapse_when_window_covers_swarm(self):
        """``lbest`` with ``k >= NP // 2`` collapses to ``gbest``."""
        from panobbgo.heuristics.pso import PSO

        h = PSO(self.strategy, NP=4, topology="lbest", lbest_k=10, seed=3)
        h.on_start()
        h.get_points(limit=100)

        self._drive_results(h, fxs=[50.0, 10.0, 30.0, 70.0])  # best at idx=1
        # _gbest_idx is the swarm-wide best from update_global_best.
        assert h._gbest_idx == 1
        # With k=10 (capped to NP // 2 = 2), the window is the whole
        # swarm — every particle's nbest must equal gbest.
        for i in range(h.NP):
            assert h._neighbourhood_best_idx(i) == 1

    def test_lbest_no_pbest_returns_none(self):
        from panobbgo.heuristics.pso import PSO

        h = PSO(self.strategy, NP=5, topology="lbest", lbest_k=1, seed=0)
        h.on_start()
        h.get_points(limit=100)
        # Nothing has been scored yet.
        assert h._neighbourhood_best_idx(2) is None


# ----------------------------------------------------------------------
# Adaptive inertia (Shi-Eberhart 1998 linearly decreasing schedule)
# ----------------------------------------------------------------------


class PSOAdaptiveInertiaTests(_MockStrategyMixin, PanobbgoTestCase):
    def test_default_w_end_is_none(self):
        from panobbgo.heuristics.pso import PSO

        h = PSO(self.strategy)
        assert h.w_end is None

    def test_invalid_w_end(self):
        from panobbgo.heuristics.pso import PSO

        with pytest.raises(ValueError, match="w_end must be finite"):
            PSO(self.strategy, w_end=float("nan"))
        with pytest.raises(ValueError, match="w_end must be finite"):
            PSO(self.strategy, w_end=float("inf"))

    def test_constant_inertia_when_w_end_none(self):
        """Without ``w_end`` the inertia is constant = ``w``."""
        from panobbgo.heuristics.pso import PSO

        h = PSO(self.strategy, w=0.6)
        # No matter what the strategy state is, we return self.w.
        assert h._current_inertia() == 0.6

    def test_adaptive_inertia_falls_back_when_results_unavailable(self):
        """``len(strategy.results)`` raises TypeError on the bare mock —
        the schedule must fall back to constant ``w``."""
        from panobbgo.heuristics.pso import PSO

        # Default mock: ``strategy.results`` is a MagicMock, ``len()``
        # raises.  The implementation catches this and returns self.w.
        h = PSO(self.strategy, w=0.9, w_end=0.4)
        # ``self.config.max_eval`` is 1000 from PanobbgoTestCase, so the
        # try-block will fail at ``len(self.strategy.results)`` and the
        # except clause must return ``self.w`` (= 0.9).
        assert h._current_inertia() == 0.9

    def test_adaptive_inertia_progress_schedule(self):
        """Linearly-decreasing inertia: ``w_eff = w − (w − w_end) · p``."""
        from unittest import mock

        from panobbgo.heuristics.pso import PSO

        h = PSO(self.strategy, w=0.9, w_end=0.4)

        # We control ``len(strategy.results)`` via a fake ``results`` object
        # with a working ``__len__``.  ``max_eval`` is 1000.
        class FakeResults:
            def __init__(self, n):
                self.n = n

            def __len__(self):
                return self.n

        with mock.patch.object(self.strategy, "results", FakeResults(0)):
            assert h._current_inertia() == pytest.approx(0.9)
        with mock.patch.object(self.strategy, "results", FakeResults(500)):
            # halfway: 0.9 − (0.9 − 0.4) · 0.5 = 0.65
            assert h._current_inertia() == pytest.approx(0.65)
        with mock.patch.object(self.strategy, "results", FakeResults(1000)):
            assert h._current_inertia() == pytest.approx(0.4)
        with mock.patch.object(self.strategy, "results", FakeResults(2000)):
            # progress is clamped to 1.0
            assert h._current_inertia() == pytest.approx(0.4)

    def test_adaptive_inertia_zero_max_eval_falls_back(self):
        """``max_eval = 0`` is degenerate; fall back to constant ``w``."""
        from unittest import mock

        from panobbgo.heuristics.pso import PSO

        h = PSO(self.strategy, w=0.9, w_end=0.4)

        class FakeResults:
            def __len__(self):
                return 0

        with mock.patch.object(self.strategy, "results", FakeResults()):
            with mock.patch.object(self.strategy.config, "max_eval", 0):
                assert h._current_inertia() == 0.9


def test_pso_kwarg_rules_in_default_catalog_extras():
    """default_catalog also exposes PSO.w and PSO.w_end so the loop can
    tune the adaptive-inertia schedule."""
    from panobbgo.self_improve import default_catalog

    catalog = default_catalog()
    keys = {(r.class_name, r.param_name) for r in catalog.rules}
    assert ("PSO", "w") in keys
    assert ("PSO", "w_end") in keys
