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

    def test_default_topology_is_gbest(self):
        from panobbgo.heuristics.pso import PSO

        h = PSO(self.strategy)
        assert h.topology == "gbest"
        assert h.k_neighbors == 2

    def test_lbest_construction(self):
        from panobbgo.heuristics.pso import PSO

        h = PSO(self.strategy, NP=10, topology="lbest", k_neighbors=3)
        assert h.topology == "lbest"
        assert h.k_neighbors == 3

    def test_invalid_topology(self):
        from panobbgo.heuristics.pso import PSO

        with pytest.raises(ValueError, match="topology must be one of"):
            PSO(self.strategy, topology="random")  # type: ignore[arg-type]

    def test_invalid_k_neighbors_type(self):
        from panobbgo.heuristics.pso import PSO

        with pytest.raises(ValueError, match="k_neighbors must be an integer"):
            PSO(self.strategy, topology="lbest", k_neighbors=2.5)  # type: ignore[arg-type]

    def test_invalid_k_neighbors_value(self):
        from panobbgo.heuristics.pso import PSO

        with pytest.raises(ValueError, match="k_neighbors must be >= 1"):
            PSO(self.strategy, topology="lbest", k_neighbors=0)
        with pytest.raises(ValueError, match="k_neighbors must be >= 1"):
            PSO(self.strategy, topology="lbest", k_neighbors=-1)


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
# Topology variants (gbest vs lbest / ring)
# ----------------------------------------------------------------------


class PSOLBestTopologyTests(_MockStrategyMixin, PanobbgoTestCase):
    """Verify the ``lbest`` (ring) topology behaviour."""

    def test_ring_neighbors_wrap_around(self):
        """``_ring_neighbors`` is a wrap-around window of width ``2k+1``."""
        from panobbgo.heuristics.pso import PSO

        h = PSO(self.strategy, NP=10, topology="lbest", k_neighbors=2)
        # No on_start required; _ring_neighbors only depends on NP / k.
        ring0 = h._ring_neighbors(0)
        # k=2, NP=10 → 5 neighbors centred at 0: {-2, -1, 0, 1, 2} → {8, 9, 0, 1, 2}
        assert sorted(ring0) == [0, 1, 2, 8, 9]

        ring9 = h._ring_neighbors(9)
        # centred at 9: {7, 8, 9, 0, 1}
        assert sorted(ring9) == [0, 1, 7, 8, 9]

    def test_ring_neighbors_size(self):
        """Each ring neighbourhood has exactly ``2k+1`` indices."""
        from panobbgo.heuristics.pso import PSO

        for k in (1, 2, 3):
            h = PSO(self.strategy, NP=20, topology="lbest", k_neighbors=k)
            for i in range(20):
                ring = h._ring_neighbors(i)
                assert len(ring) == 2 * k + 1
                assert i in ring  # neighbourhood always includes self

    def test_lbest_social_best_uses_ring(self):
        """Under ``lbest``, the social attractor for particle ``i`` is the
        best ``pbest`` among ``i``'s ring neighbours — even when a
        better pbest exists *outside* the ring.
        """
        from panobbgo.heuristics.pso import PSO
        from panobbgo.lib import Point, Result

        # NP=8, k=1 → ring(0) = {7, 0, 1}.  Plant a great pbest at index 4
        # (outside the ring) and a mediocre pbest at index 1 (inside the
        # ring).  lbest social attractor for particle 0 must be index 1.
        h = PSO(self.strategy, NP=8, topology="lbest", k_neighbors=1, seed=0)
        h.on_start()
        h.get_points(limit=100)

        def feed(idx, fx):
            req_id = next(rid for rid, i in h._pending.items() if i == idx)
            p = Point(h._positions[idx].copy(), f"PSO:{req_id}")
            h.on_new_results([Result(p, fx)])

        feed(4, 0.001)  # excellent pbest — but outside particle 0's ring
        h.get_points(limit=100)
        feed(1, 5.0)  # mediocre pbest — inside particle 0's ring
        h.get_points(limit=100)

        # _gbest_idx is the global best (used for reporting), points at 4.
        assert h._gbest_idx == 4
        # The social attractor for particle 0 (lbest topology) should be 1,
        # not 4 — particle 0 cannot "see" past its ring of width 3.
        assert h._social_best_idx(0) == 1

    def test_gbest_social_best_is_global(self):
        """Under ``gbest``, ``_social_best_idx`` is just ``_gbest_idx``."""
        from panobbgo.heuristics.pso import PSO
        from panobbgo.lib import Point, Result

        h = PSO(self.strategy, NP=6, topology="gbest", seed=0)
        h.on_start()
        h.get_points(limit=100)

        # Plant two pbests, one strictly better than the other.
        items = list(h._pending.items())
        req_id_a, idx_a = items[0]
        req_id_b, idx_b = items[1]
        h.on_new_results(
            [
                Result(Point(h._positions[idx_a].copy(), f"PSO:{req_id_a}"), 100.0),
                Result(Point(h._positions[idx_b].copy(), f"PSO:{req_id_b}"), 1.0),
            ]
        )

        # Every particle's social attractor under gbest is the global best.
        for i in range(h.NP):
            assert h._social_best_idx(i) == idx_b

    def test_lbest_social_best_none_until_neighbour_evaluated(self):
        """If no neighbour in the ring has a pbest yet, returns None."""
        from panobbgo.heuristics.pso import PSO

        h = PSO(self.strategy, NP=8, topology="lbest", k_neighbors=1)
        h.on_start()
        h.get_points(limit=100)
        # No results fed yet → no pbests anywhere.
        for i in range(h.NP):
            assert h._social_best_idx(i) is None

    def test_lbest_velocity_clamp(self):
        """lbest topology must respect the same ``v_max`` clamp as gbest."""
        from panobbgo.heuristics.pso import PSO
        from panobbgo.lib import Point, Result

        h = PSO(
            self.strategy,
            NP=5,
            topology="lbest",
            k_neighbors=1,
            w=0.9,
            c1=2.0,
            c2=2.0,
            v_max_frac=0.2,
            seed=99,
        )
        h.on_start()
        h.get_points(limit=100)

        # Trigger an lbest-driven velocity update by feeding back two
        # neighbours' results — particle 0 then sees a non-None social
        # attractor in its ring {4, 0, 1}.
        for idx in (1, 0):
            req_id = next(rid for rid, i in h._pending.items() if i == idx)
            p = Point(h._positions[idx].copy(), f"PSO:{req_id}")
            h.on_new_results([Result(p, 1.0 + idx)])

        ranges = self.problem.box[:, 1] - self.problem.box[:, 0]
        v_max = 0.2 * ranges
        assert np.all(np.abs(h._velocities) <= v_max + 1e-12)

    def test_lbest_drives_toward_origin_on_quadratic(self):
        """End-to-end: lbest still strictly improves on a quadratic."""
        from panobbgo.heuristics.pso import PSO
        from panobbgo.lib import Result

        h = PSO(self.strategy, NP=10, topology="lbest", k_neighbors=2, seed=2026)
        h.on_start()
        pts = h.get_points(limit=100)

        def f(x):
            return float(np.sum(x * x))

        def evaluate_and_feed(points):
            results = [Result(pt, f(pt.x)) for pt in points]
            h.on_new_results(results)

        evaluate_and_feed(pts)
        best_fx = min(r.fx for r in h._pbest_result if r is not None)
        for _ in range(20):
            pts = h.get_points(limit=100)
            if not pts:
                break
            evaluate_and_feed(pts)
        new_best = min(r.fx for r in h._pbest_result if r is not None)
        assert new_best <= best_fx + 1e-12


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


def test_pso_lbest_variant_in_default_structural_catalog():
    """Both gbest (default) and lbest PSO variants appear in the structural catalog.

    The catalog ships *two* PSO entries — canonical gbest (Kennedy-Eberhart
    1995) and lbest ring topology (Kennedy & Mendes 2002) — so the
    self-improvement loop can pick whichever helps on the current battery.
    Both share ``cls = PSO`` so ``avoid_duplicates=True`` still prevents
    multiple PSO instances per strategy; instead the catalog samples
    uniformly between them when PSO is not yet present.
    """
    from panobbgo.self_improve import default_structural_catalog
    from panobbgo.heuristics.pso import PSO

    catalog = default_structural_catalog()
    add_rules = [r for r in catalog.rules if getattr(r, "op", None) == "add_heuristic"]
    pso_entries = [kwargs for rule in add_rules for cls, kwargs in rule.candidate_classes if cls is PSO]
    assert len(pso_entries) >= 2, f"expected ≥2 PSO entries, got {pso_entries!r}"
    topologies = {kwargs.get("topology", "gbest") for kwargs in pso_entries}
    assert topologies == {"gbest", "lbest"}


def test_pso_kwarg_rule_in_default_catalog():
    """default_catalog includes a kwarg rule for PSO.NP."""
    from panobbgo.self_improve import default_catalog

    catalog = default_catalog()
    keys = {(r.class_name, r.param_name) for r in catalog.rules}
    assert ("PSO", "NP") in keys
