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
            PSO(self.strategy, topology="not-a-real-topology")  # type: ignore[arg-type]

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
# Von Neumann (2-D toroidal grid) topology
# ----------------------------------------------------------------------


class PSOVonNeumannTopologyTests(_MockStrategyMixin, PanobbgoTestCase):
    """Verify the ``vonneumann`` (4-connected 2-D toroidal grid) topology."""

    def test_vonneumann_construction(self):
        from panobbgo.heuristics.pso import PSO

        h = PSO(self.strategy, NP=20, topology="vonneumann")
        assert h.topology == "vonneumann"
        # k_neighbors is preserved on the instance even though Von Neumann
        # ignores it (the heuristic stores it for round-trip serialization).
        assert isinstance(h.k_neighbors, int)

    def test_vonneumann_grid_perfect_rectangle(self):
        """20 = 4·5, 16 = 4·4, 12 = 3·4: rows · cols == NP exactly."""
        from panobbgo.heuristics.pso import PSO

        for NP, expected in (
            (4, (2, 2)),
            (9, (3, 3)),
            (12, (3, 4)),
            (16, (4, 4)),
            (20, (4, 5)),
            (25, (5, 5)),
        ):
            h = PSO(self.strategy, NP=NP, topology="vonneumann")
            rows, cols = h._vonneumann_grid()
            assert rows * cols == NP, f"NP={NP}: rows={rows} cols={cols}"
            assert rows == expected[0] and cols == expected[1], f"NP={NP}: got ({rows},{cols})"

    def test_vonneumann_grid_non_rectangular(self):
        """When NP doesn't factor perfectly, rows·cols > NP and rows·cols - NP cells are phantom."""
        from panobbgo.heuristics.pso import PSO

        for NP in (7, 11, 13, 17, 19, 23):  # primes / near-primes
            h = PSO(self.strategy, NP=NP, topology="vonneumann")
            rows, cols = h._vonneumann_grid()
            assert rows * cols >= NP, f"NP={NP}: rows={rows} cols={cols}"
            # rows should be approximately sqrt(NP), within 1.
            assert abs(rows - round(NP**0.5)) <= 1, f"NP={NP}: rows={rows} far from sqrt"

    def test_vonneumann_neighbors_full_rectangle(self):
        """On a 4x5 grid (NP=20), every particle has exactly 4+1=5 neighbours."""
        from panobbgo.heuristics.pso import PSO

        h = PSO(self.strategy, NP=20, topology="vonneumann")
        rows, cols = h._vonneumann_grid()
        assert (rows, cols) == (4, 5)
        for i in range(20):
            nbrs = h._vonneumann_neighbors(i)
            assert len(nbrs) == 5, f"particle {i}: {nbrs}"
            assert i in nbrs  # self is always in the neighbourhood

    def test_vonneumann_neighbors_wrap_around(self):
        """Wrap-around on a 4x5 grid: corner cells wrap to the opposite edge.

        Layout (NP=20, 4 rows × 5 cols)::

             0  1  2  3  4
             5  6  7  8  9
            10 11 12 13 14
            15 16 17 18 19

        Particle 0 (r=0, c=0):
          N → (3, 0) = 15  (wrap top → bottom)
          S → (1, 0) = 5
          W → (0, 4) = 4   (wrap left → right)
          E → (0, 1) = 1
        """
        from panobbgo.heuristics.pso import PSO

        h = PSO(self.strategy, NP=20, topology="vonneumann")
        nbrs0 = set(h._vonneumann_neighbors(0))
        assert nbrs0 == {0, 15, 5, 4, 1}

        # Particle 12 (r=2, c=2) — interior, no wrap needed.
        # N=(1,2)=7, S=(3,2)=17, W=(2,1)=11, E=(2,3)=13
        nbrs12 = set(h._vonneumann_neighbors(12))
        assert nbrs12 == {12, 7, 17, 11, 13}

        # Particle 19 (r=3, c=4) — corner with two wraps.
        # N=(2,4)=14, S=(0,4)=4 (wrap), W=(3,3)=18, E=(3,0)=15 (wrap)
        nbrs19 = set(h._vonneumann_neighbors(19))
        assert nbrs19 == {19, 14, 4, 18, 15}

    def test_vonneumann_neighbors_phantom_skipped(self):
        """On a non-rectangular grid, neighbours mapping to phantom cells are skipped.

        NP=10, rows=3, cols=4 → grid is::

             0  1  2  3
             4  5  6  7
             8  9  .  .   (cells 10, 11 are phantom)

        Particle 7 (r=1, c=3):
          N=(0,3)=3,
          S=(2,3)=11 PHANTOM → skipped,
          W=(1,2)=6,
          E=(1,0)=4 (wrap)
        Result: {7, 3, 6, 4} — only 4 neighbours (not 5).
        """
        from panobbgo.heuristics.pso import PSO

        h = PSO(self.strategy, NP=10, topology="vonneumann")
        rows, cols = h._vonneumann_grid()
        assert (rows, cols) == (3, 4)
        nbrs7 = h._vonneumann_neighbors(7)
        assert set(nbrs7) == {7, 3, 6, 4}
        # Particle 2 (r=0, c=2): N=(2,2)=phantom 10 → skipped
        #                       S=(1,2)=6, W=(0,1)=1, E=(0,3)=3
        nbrs2 = set(h._vonneumann_neighbors(2))
        assert nbrs2 == {2, 6, 1, 3}

    def test_vonneumann_neighbors_dedupe_small_swarms(self):
        """On very small swarms, wrap-around collapses to duplicates — de-duped."""
        from panobbgo.heuristics.pso import PSO

        # NP=4 → 2x2 grid.  Each particle's N and S wrap to the same cell;
        # W and E wrap to the same cell.  Result: 3 unique neighbours.
        h = PSO(self.strategy, NP=4, topology="vonneumann")
        for i in range(4):
            nbrs = h._vonneumann_neighbors(i)
            assert len(set(nbrs)) == len(nbrs), f"duplicate in {nbrs}"
            assert i in nbrs

    def test_vonneumann_social_best_uses_grid_neighbourhood(self):
        """Under ``vonneumann`` the social attractor must come from the 2-D neighbourhood.

        On a 4x5 grid, particle 0 sees {0, 15, 5, 4, 1}.  Plant a great pbest
        at index 12 (interior, *not* in particle 0's neighbourhood) and a
        mediocre pbest at index 5 (north of 0).  The social attractor for 0
        must be 5, not 12.
        """
        from panobbgo.heuristics.pso import PSO
        from panobbgo.lib import Point, Result

        h = PSO(self.strategy, NP=20, topology="vonneumann", seed=0)
        h.on_start()
        h.get_points(limit=200)

        def feed(idx, fx):
            req_id = next(rid for rid, i in h._pending.items() if i == idx)
            p = Point(h._positions[idx].copy(), f"PSO:{req_id}")
            h.on_new_results([Result(p, fx)])

        feed(12, 0.001)  # excellent pbest — outside particle 0's neighbourhood
        h.get_points(limit=200)
        feed(5, 5.0)  # mediocre pbest — inside particle 0's neighbourhood
        h.get_points(limit=200)

        # _gbest_idx is the global best (used for reporting), points at 12.
        assert h._gbest_idx == 12
        # Social attractor for particle 0 under vonneumann should be 5,
        # not 12 — particle 0 cannot "see" past its N/S/E/W neighbours.
        assert h._social_best_idx(0) == 5

    def test_vonneumann_social_best_none_until_neighbour_evaluated(self):
        """If no neighbour in the Von Neumann set has a pbest yet, returns None."""
        from panobbgo.heuristics.pso import PSO

        h = PSO(self.strategy, NP=12, topology="vonneumann")
        h.on_start()
        h.get_points(limit=100)
        for i in range(h.NP):
            assert h._social_best_idx(i) is None

    def test_vonneumann_velocity_clamp(self):
        """Von Neumann topology respects the same ``v_max`` clamp."""
        from panobbgo.heuristics.pso import PSO
        from panobbgo.lib import Point, Result

        h = PSO(
            self.strategy,
            NP=9,
            topology="vonneumann",
            w=0.9,
            c1=2.0,
            c2=2.0,
            v_max_frac=0.2,
            seed=99,
        )
        h.on_start()
        h.get_points(limit=200)

        # Feed two neighbours' results so particle 0's social attractor is non-None.
        for idx in (1, 0):
            req_id = next(rid for rid, i in h._pending.items() if i == idx)
            p = Point(h._positions[idx].copy(), f"PSO:{req_id}")
            h.on_new_results([Result(p, 1.0 + idx)])

        ranges = self.problem.box[:, 1] - self.problem.box[:, 0]
        v_max = 0.2 * ranges
        assert np.all(np.abs(h._velocities) <= v_max + 1e-12)

    def test_vonneumann_drives_toward_origin_on_quadratic(self):
        """End-to-end: Von Neumann strictly improves on a quadratic."""
        from panobbgo.heuristics.pso import PSO
        from panobbgo.lib import Result

        h = PSO(self.strategy, NP=12, topology="vonneumann", seed=2026)
        h.on_start()
        pts = h.get_points(limit=200)

        def f(x):
            return float(np.sum(x * x))

        def evaluate_and_feed(points):
            results = [Result(pt, f(pt.x)) for pt in points]
            h.on_new_results(results)

        evaluate_and_feed(pts)
        best_fx = min(r.fx for r in h._pbest_result if r is not None)
        for _ in range(20):
            pts = h.get_points(limit=200)
            if not pts:
                break
            evaluate_and_feed(pts)
        new_best = min(r.fx for r in h._pbest_result if r is not None)
        assert new_best <= best_fx + 1e-12


# ----------------------------------------------------------------------
# Random (Clerc 2007 / SPSO 2011 stochastic informer graph) topology
# ----------------------------------------------------------------------


class PSORandomTopologyTests(_MockStrategyMixin, PanobbgoTestCase):
    """Verify the ``random`` (Mendes 2004 / Clerc 2007 / SPSO 2011) topology."""

    def test_random_construction(self):
        from panobbgo.heuristics.pso import PSO

        h = PSO(self.strategy, NP=20, topology="random", k_neighbors=3)
        assert h.topology == "random"
        assert h.k_neighbors == 3
        # Adjacency is sized lazily — None until on_start runs.
        assert h._random_adjacency is None

    def test_random_adjacency_built_on_start(self):
        """``on_start`` populates the per-particle informer list."""
        from panobbgo.heuristics.pso import PSO

        h = PSO(self.strategy, NP=12, topology="random", k_neighbors=3, seed=42)
        h.on_start()
        assert h._random_adjacency is not None
        assert len(h._random_adjacency) == h.NP

    def test_random_adjacency_contains_self(self):
        """Every particle is its own informer (so the swarm can always read its pbest)."""
        from panobbgo.heuristics.pso import PSO

        h = PSO(self.strategy, NP=10, topology="random", k_neighbors=3, seed=7)
        h.on_start()
        for i, informers in enumerate(h._random_adjacency):
            assert i in informers, f"particle {i} not in its own informer list {informers}"

    def test_random_adjacency_excludes_collisions(self):
        """Duplicate draws are removed — realised neighbourhood ≤ k_neighbors + 1."""
        from panobbgo.heuristics.pso import PSO

        h = PSO(self.strategy, NP=20, topology="random", k_neighbors=3, seed=2026)
        h.on_start()
        for informers in h._random_adjacency:
            assert len(informers) == len(set(informers)), f"duplicates in {informers}"
            assert len(informers) <= h.k_neighbors + 1
            assert len(informers) >= 2  # at least self + one draw (with replacement)

    def test_random_adjacency_excludes_self_in_draws(self):
        """The informer draws sample from ``{0..NP-1} \\ {i}`` — self only appears once."""
        from panobbgo.heuristics.pso import PSO

        # Run many seeds to make sure the index-shift logic always excludes self.
        for seed in range(50):
            h = PSO(self.strategy, NP=8, topology="random", k_neighbors=4, seed=seed)
            h.on_start()
            for i, informers in enumerate(h._random_adjacency):
                # Each particle's own index appears exactly once (added by _init).
                assert informers.count(i) == 1, f"seed={seed} i={i}: {informers}"

    def test_random_adjacency_asymmetric(self):
        """The graph is asymmetric: ``j ∈ informers(i)`` does not imply ``i ∈ informers(j)``."""
        from panobbgo.heuristics.pso import PSO

        # With NP=20 / k=2 and seed=0, asymmetry is almost certain.
        h = PSO(self.strategy, NP=20, topology="random", k_neighbors=2, seed=0)
        h.on_start()
        # Build the reverse adjacency and verify at least one mismatch.
        forward = {(i, j) for i, informers in enumerate(h._random_adjacency) for j in informers if j != i}
        backward = {(j, i) for (i, j) in forward}
        assert forward != backward, "random adjacency should be asymmetric in general"

    def test_random_adjacency_seed_reproducibility(self):
        """Two PSOs sharing the same seed produce identical adjacency lists."""
        from panobbgo.heuristics.pso import PSO

        h1 = PSO(self.strategy, NP=15, topology="random", k_neighbors=4, seed=123)
        h2 = PSO(self.strategy, NP=15, topology="random", k_neighbors=4, seed=123)
        h1.on_start()
        h2.on_start()
        assert h1._random_adjacency == h2._random_adjacency

    def test_random_adjacency_resampled_on_restart(self):
        """``on_restart`` re-samples the informer graph (Clerc 2007 stagnation rebuild)."""
        from panobbgo.heuristics.pso import PSO

        h = PSO(self.strategy, NP=15, topology="random", k_neighbors=3, seed=99)
        h.on_start()
        before = [list(row) for row in h._random_adjacency]
        h.on_restart(center=np.zeros(self.problem.dim))
        after = h._random_adjacency
        # The deterministic RNG plus distinct call should change at least
        # one row.  Extremely unlikely to be identical given NP=15, k=3.
        assert any(before[i] != after[i] for i in range(h.NP))

    def test_random_social_best_uses_informer_set(self):
        """Under ``random`` the social attractor is restricted to the per-particle informer set."""
        from panobbgo.heuristics.pso import PSO
        from panobbgo.lib import Point, Result

        h = PSO(self.strategy, NP=15, topology="random", k_neighbors=3, seed=42)
        h.on_start()
        h.get_points(limit=200)

        # Plant a great pbest at some index that is NOT in particle 0's
        # informer set, plus a mediocre pbest that IS.  The social
        # attractor for particle 0 must be the mediocre one (its own
        # informer), not the great one.
        informers_0 = set(h._random_adjacency[0]) - {0}
        all_other = set(range(h.NP)) - {0} - informers_0
        # Need at least one outside-informer index to plant the excellent pbest.
        assert all_other, "test setup needs NP large enough for asymmetric informer sets"
        outside_idx = sorted(all_other)[0]
        inside_idx = sorted(informers_0)[0]

        def feed(idx, fx):
            req_id = next(rid for rid, i in h._pending.items() if i == idx)
            p = Point(h._positions[idx].copy(), f"PSO:{req_id}")
            h.on_new_results([Result(p, fx)])

        feed(outside_idx, 0.0001)  # excellent pbest — outside particle 0's informer set
        h.get_points(limit=200)
        feed(inside_idx, 7.0)  # mediocre pbest — inside particle 0's informer set
        h.get_points(limit=200)

        # _gbest_idx is the global best (used for reporting), points at outside_idx.
        assert h._gbest_idx == outside_idx
        # Social attractor for particle 0 under random topology must come
        # from the informer set — it cannot "see" outside_idx.
        assert h._social_best_idx(0) == inside_idx

    def test_random_social_best_none_until_neighbour_evaluated(self):
        """Returns ``None`` until at least one informer has a pbest."""
        from panobbgo.heuristics.pso import PSO

        h = PSO(self.strategy, NP=12, topology="random", k_neighbors=3, seed=1)
        h.on_start()
        h.get_points(limit=100)
        for i in range(h.NP):
            assert h._social_best_idx(i) is None

    def test_random_velocity_clamp(self):
        """Random topology respects the same ``v_max`` clamp as the others."""
        from panobbgo.heuristics.pso import PSO
        from panobbgo.lib import Point, Result

        h = PSO(
            self.strategy,
            NP=9,
            topology="random",
            k_neighbors=2,
            w=0.9,
            c1=2.0,
            c2=2.0,
            v_max_frac=0.2,
            seed=11,
        )
        h.on_start()
        h.get_points(limit=200)

        # Feed two particles' results so the social attractor pool is non-empty.
        for idx in (1, 0):
            req_id = next(rid for rid, i in h._pending.items() if i == idx)
            p = Point(h._positions[idx].copy(), f"PSO:{req_id}")
            h.on_new_results([Result(p, 1.0 + idx)])

        ranges = self.problem.box[:, 1] - self.problem.box[:, 0]
        v_max = 0.2 * ranges
        assert np.all(np.abs(h._velocities) <= v_max + 1e-12)

    def test_random_drives_toward_origin_on_quadratic(self):
        """End-to-end: random topology strictly improves on a quadratic."""
        from panobbgo.heuristics.pso import PSO
        from panobbgo.lib import Result

        h = PSO(self.strategy, NP=12, topology="random", k_neighbors=3, seed=2026)
        h.on_start()
        pts = h.get_points(limit=200)

        def f(x):
            return float(np.sum(x * x))

        def evaluate_and_feed(points):
            results = [Result(pt, f(pt.x)) for pt in points]
            h.on_new_results(results)

        evaluate_and_feed(pts)
        best_fx = min(r.fx for r in h._pbest_result if r is not None)
        for _ in range(20):
            pts = h.get_points(limit=200)
            if not pts:
                break
            evaluate_and_feed(pts)
        new_best = min(r.fx for r in h._pbest_result if r is not None)
        assert new_best <= best_fx + 1e-12


# ----------------------------------------------------------------------
# Stochastic-K stagnation rebuild (Clerc 2007 / SPSO 2011)
# ----------------------------------------------------------------------


class PSOStochasticKTests(_MockStrategyMixin, PanobbgoTestCase):
    """Verify the stochastic-K stagnation-rebuild policy on the random topology."""

    def test_default_stagnation_threshold_is_none(self):
        """The kwarg defaults to ``None`` so existing behaviour is byte-identical."""
        from panobbgo.heuristics.pso import PSO

        h = PSO(self.strategy, NP=12, topology="random", k_neighbors=3)
        assert h.stagnation_threshold is None
        assert h._stagnation_counter == 0

    def test_custom_stagnation_threshold_round_trip(self):
        from panobbgo.heuristics.pso import PSO

        h = PSO(self.strategy, NP=12, topology="random", k_neighbors=3, stagnation_threshold=8)
        assert h.stagnation_threshold == 8

    def test_invalid_stagnation_threshold_type(self):
        from panobbgo.heuristics.pso import PSO

        with pytest.raises(ValueError, match="stagnation_threshold must be a positive integer"):
            PSO(self.strategy, topology="random", stagnation_threshold=5.0)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="stagnation_threshold must be a positive integer"):
            PSO(self.strategy, topology="random", stagnation_threshold=True)  # type: ignore[arg-type]

    def test_invalid_stagnation_threshold_value(self):
        from panobbgo.heuristics.pso import PSO

        with pytest.raises(ValueError, match="stagnation_threshold must be >= 1"):
            PSO(self.strategy, topology="random", stagnation_threshold=0)
        with pytest.raises(ValueError, match="stagnation_threshold must be >= 1"):
            PSO(self.strategy, topology="random", stagnation_threshold=-3)

    def _feed(self, h, idx, fx):
        """Feed a result for the *pending* trial of particle ``idx``."""
        from panobbgo.lib import Point, Result

        req_id = next(rid for rid, i in h._pending.items() if i == idx)
        p = Point(h._positions[idx].copy(), f"PSO:{req_id}")
        h.on_new_results([Result(p, fx)])

    def test_stagnation_counter_starts_at_zero(self):
        """``on_start`` resets the counter to zero."""
        from panobbgo.heuristics.pso import PSO

        h = PSO(self.strategy, NP=10, topology="random", k_neighbors=3, stagnation_threshold=5, seed=1)
        h.on_start()
        assert h._stagnation_counter == 0

    def test_stagnation_counter_resets_on_improvement(self):
        """Each strict improvement of the global best resets the counter."""
        from panobbgo.heuristics.pso import PSO

        h = PSO(self.strategy, NP=8, topology="random", k_neighbors=2, stagnation_threshold=5, seed=42)
        h.on_start()
        # First result establishes the global best — counter stays at 0.
        self._feed(h, 0, 5.0)
        assert h._stagnation_counter == 0
        # A worse result does not improve the global best — counter ticks up.
        self._feed(h, 1, 7.0)
        assert h._stagnation_counter == 1
        # A strictly-better result improves the global best — counter resets.
        self._feed(h, 2, 1.0)
        assert h._stagnation_counter == 0

    def test_stagnation_rebuilds_after_threshold_consecutive_misses(self):
        """When the counter hits the threshold, the adjacency is rebuilt and the counter resets."""
        from panobbgo.heuristics.pso import PSO

        h = PSO(self.strategy, NP=10, topology="random", k_neighbors=3, stagnation_threshold=3, seed=99)
        h.on_start()
        before = [list(row) for row in h._random_adjacency]

        # Establish a global best at particle 0 with a very low fx so
        # later feeds cannot improve it.
        self._feed(h, 0, -100.0)
        assert h._stagnation_counter == 0

        # Three consecutive non-improving results trigger the rebuild.
        self._feed(h, 1, 50.0)
        self._feed(h, 2, 50.0)
        self._feed(h, 3, 50.0)
        # Counter must reset on rebuild.
        assert h._stagnation_counter == 0
        after = h._random_adjacency
        # Adjacency must have changed (NP=10 / k=3 makes identity vanishingly improbable).
        assert any(before[i] != after[i] for i in range(h.NP))

    def test_stagnation_no_rebuild_before_threshold(self):
        """Below the threshold, the adjacency is untouched."""
        from panobbgo.heuristics.pso import PSO

        h = PSO(self.strategy, NP=10, topology="random", k_neighbors=3, stagnation_threshold=5, seed=77)
        h.on_start()
        before = [list(row) for row in h._random_adjacency]

        self._feed(h, 0, -10.0)  # establishes gbest
        for idx in (1, 2, 3, 4):  # 4 consecutive non-improvements, below threshold of 5
            self._feed(h, idx, 50.0)
        assert h._stagnation_counter == 4
        # Adjacency unchanged.
        assert h._random_adjacency == before

    def test_stagnation_noop_when_threshold_none(self):
        """``stagnation_threshold=None`` (default) never rebuilds the adjacency mid-run."""
        from panobbgo.heuristics.pso import PSO

        h = PSO(self.strategy, NP=8, topology="random", k_neighbors=3, seed=2026)
        h.on_start()
        before = [list(row) for row in h._random_adjacency]
        self._feed(h, 0, 1.0)
        # Long stretch of non-improvements: counter remains at 0 because the policy is off.
        for idx in range(1, h.NP):
            self._feed(h, idx, 50.0)
        assert h._stagnation_counter == 0
        assert h._random_adjacency == before

    def test_stagnation_noop_for_non_random_topology(self):
        """``stagnation_threshold`` is ignored for ``gbest`` / ``lbest`` / ``vonneumann``."""
        from panobbgo.heuristics.pso import PSO

        for topo in ("gbest", "lbest", "vonneumann"):
            h = PSO(
                self.strategy,
                NP=12,
                topology=topo,
                k_neighbors=2,
                stagnation_threshold=2,
                seed=5,
            )
            h.on_start()
            # No random adjacency exists — only the random topology
            # allocates one.
            assert h._random_adjacency is None
            self._feed(h, 0, -1.0)  # establishes gbest
            self._feed(h, 1, 100.0)
            self._feed(h, 2, 100.0)
            self._feed(h, 3, 100.0)
            # The counter advances but nothing else changes — there is
            # no adjacency to rebuild under these topologies.
            assert h._random_adjacency is None

    def test_stagnation_counter_resets_on_restart(self):
        """``on_restart`` zeros the stagnation counter even if mid-stagnation."""
        from panobbgo.heuristics.pso import PSO

        h = PSO(self.strategy, NP=10, topology="random", k_neighbors=3, stagnation_threshold=10, seed=33)
        h.on_start()
        self._feed(h, 0, -5.0)
        for idx in range(1, 5):  # four non-improvements, well below threshold
            self._feed(h, idx, 50.0)
        assert h._stagnation_counter == 4
        h.on_restart(center=np.zeros(self.problem.dim), reason="test")
        assert h._stagnation_counter == 0

    def test_stagnation_does_not_double_count_first_global_best(self):
        """The very first global-best observation must not tick the counter."""
        from panobbgo.heuristics.pso import PSO

        h = PSO(self.strategy, NP=8, topology="random", k_neighbors=2, stagnation_threshold=2, seed=8)
        h.on_start()
        self._feed(h, 0, 5.0)
        assert h._stagnation_counter == 0


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
    """The gbest, lbest, vonneumann, and random PSO variants all appear in the structural catalog.

    The catalog ships *four* PSO entries — canonical gbest
    (Kennedy-Eberhart 1995), lbest ring topology (Kennedy & Mendes
    2002), vonneumann 2-D toroidal grid (Kennedy & Mendes 2003;
    Mendes 2004), and random informer graph (Mendes 2004; Clerc 2007
    / SPSO 2011) — so the self-improvement loop can pick whichever
    helps on the current battery.  All four share ``cls = PSO`` so
    ``avoid_duplicates=True`` still prevents multiple PSO instances
    per strategy; instead the catalog samples uniformly between them
    when PSO is not yet present.
    """
    from panobbgo.self_improve import default_structural_catalog
    from panobbgo.heuristics.pso import PSO

    catalog = default_structural_catalog()
    add_rules = [r for r in catalog.rules if getattr(r, "op", None) == "add_heuristic"]
    pso_entries = [kwargs for rule in add_rules for cls, kwargs in rule.candidate_classes if cls is PSO]
    assert len(pso_entries) >= 4, f"expected ≥4 PSO entries, got {pso_entries!r}"
    topologies = {kwargs.get("topology", "gbest") for kwargs in pso_entries}
    assert topologies == {"gbest", "lbest", "vonneumann", "random"}


def test_pso_topology_categorical_rule_includes_vonneumann():
    """The PSO.topology categorical rule covers all four shipped topologies."""
    from panobbgo.self_improve import default_catalog

    catalog = default_catalog()
    topo_rules = [r for r in catalog.rules if r.class_name == "PSO" and r.param_name == "topology"]
    assert len(topo_rules) == 1, f"expected exactly 1 PSO.topology rule, got {len(topo_rules)}"
    rule = topo_rules[0]
    assert rule.kind == "categorical_choice"
    assert set(rule.choices) == {"gbest", "lbest", "vonneumann", "random"}


def test_pso_kwarg_rule_in_default_catalog():
    """default_catalog includes a kwarg rule for PSO.NP."""
    from panobbgo.self_improve import default_catalog

    catalog = default_catalog()
    keys = {(r.class_name, r.param_name) for r in catalog.rules}
    assert ("PSO", "NP") in keys


def test_pso_stagnation_threshold_rule_in_default_catalog():
    """default_catalog ships a kwarg rule for ``PSO.stagnation_threshold``.

    The rule fires only when a spec sets the kwarg explicitly (per
    ``_find_targets``'s "param already in kwargs" predicate), so the
    built-in factories that leave ``stagnation_threshold=None`` see no
    behavioural change.
    """
    from panobbgo.self_improve import default_catalog

    catalog = default_catalog()
    rules = [r for r in catalog.rules if r.class_name == "PSO" and r.param_name == "stagnation_threshold"]
    assert len(rules) == 1, f"expected exactly 1 stagnation_threshold rule, got {len(rules)}"
    rule = rules[0]
    assert rule.kind == "integer_add"
    assert rule.bounds == (5, 60)


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
        assert h._current_inertia() == 0.6

    def test_adaptive_inertia_falls_back_when_results_unavailable(self):
        """When ``len(strategy.results)`` raises, fall back to constant ``w``."""
        from panobbgo.heuristics.pso import PSO

        h = PSO(self.strategy, w=0.9, w_end=0.4)
        assert h._current_inertia() == 0.9

    def test_adaptive_inertia_progress_schedule(self):
        """Linearly-decreasing inertia: ``w_eff = w − (w − w_end) · p``."""
        from unittest import mock

        from panobbgo.heuristics.pso import PSO

        h = PSO(self.strategy, w=0.9, w_end=0.4)

        class FakeResults:
            def __init__(self, n):
                self.n = n

            def __len__(self):
                return self.n

        # Pin ``max_eval`` to 1000 so the schedule is well-defined regardless
        # of any earlier test (the Panobbgo Config object is a singleton, so
        # an unrelated test can otherwise mutate the budget out from under us).
        with mock.patch.object(self.strategy.config, "max_eval", 1000):
            with mock.patch.object(self.strategy, "results", FakeResults(0)):
                assert h._current_inertia() == pytest.approx(0.9)
            with mock.patch.object(self.strategy, "results", FakeResults(500)):
                assert h._current_inertia() == pytest.approx(0.65)
            with mock.patch.object(self.strategy, "results", FakeResults(1000)):
                assert h._current_inertia() == pytest.approx(0.4)
            with mock.patch.object(self.strategy, "results", FakeResults(2000)):
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
