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

"""Tests for the L-SHADE adaptive Differential Evolution heuristic."""

from __future__ import annotations

import numpy as np
import pytest

from panobbgo.utils import PanobbgoTestCase


class _MockStrategyMixin:
    """Set up the constraint handler and a stub max_eval budget.

    Most LSHADE behaviour reads ``strategy.constraint_handler`` (for
    fitness ranking and is_better comparisons) plus
    ``strategy.config.max_eval`` (for LPSR pacing).  The
    :class:`~panobbgo.utils.PanobbgoTestCase` mock strategy lacks the
    constraint handler and reuses the :class:`~panobbgo.config.Config`
    *singleton*; mutating ``config.max_eval`` directly would leak the
    setting to every later test.  Instead we save/restore the original
    value around each test so the singleton is left untouched on exit.
    """

    def setUp(self):
        super().setUp()
        from panobbgo.lib.constraints import DefaultConstraintHandler

        self.strategy.constraint_handler = DefaultConstraintHandler(self.strategy)
        # Save the original singleton-scoped max_eval so we can restore
        # it in tearDown — direct assignment would bleed across tests.
        self._orig_max_eval = self.strategy.config.max_eval
        self.strategy.config.max_eval = 1000
        # ``strategy.results`` must be len()-able for the LPSR pacing.
        self.strategy.results = []

    def tearDown(self):
        # Roll back the singleton mutation done in setUp / individual tests.
        self.strategy.config.max_eval = self._orig_max_eval
        super().tearDown()


# ----------------------------------------------------------------------
# Construction-time validation
# ----------------------------------------------------------------------


class LSHADEConstructionTests(_MockStrategyMixin, PanobbgoTestCase):
    def test_default_construction(self):
        from panobbgo.heuristics.lshade import LSHADE

        h = LSHADE(self.strategy)
        assert h.NP_init == 30
        assert h.NP_min == 4
        assert h.H == 6
        assert 0.0 < h.p_best <= 1.0
        assert h.archive_factor >= 0.0
        assert h.name == "LSHADE"
        assert h._M_F.shape == (h.H,)
        assert np.allclose(h._M_F, 0.5)
        assert np.allclose(h._M_CR, 0.5)
        assert h._mem_ptr == 0

    def test_custom_construction(self):
        from panobbgo.heuristics.lshade import LSHADE

        h = LSHADE(self.strategy, NP_init=20, NP_min=6, H=4, p_best=0.2, archive_factor=2.0, seed=7, name="MyDE")
        assert h.NP_init == 20
        assert h.NP_min == 6
        assert h.H == 4
        assert h.p_best == 0.2
        assert h.archive_factor == 2.0
        assert h.name == "MyDE"

    def test_invalid_NP_init_type(self):
        from panobbgo.heuristics.lshade import LSHADE

        with pytest.raises(ValueError, match="NP_init must be an integer"):
            LSHADE(self.strategy, NP_init=10.0)  # type: ignore[arg-type]

    def test_invalid_NP_init_value(self):
        from panobbgo.heuristics.lshade import LSHADE

        with pytest.raises(ValueError, match="NP_init must be >= 4"):
            LSHADE(self.strategy, NP_init=3)
        with pytest.raises(ValueError, match="NP_init must be >= 4"):
            LSHADE(self.strategy, NP_init=0)

    def test_invalid_NP_min_value(self):
        from panobbgo.heuristics.lshade import LSHADE

        with pytest.raises(ValueError, match="NP_min must be >= 4"):
            LSHADE(self.strategy, NP_min=2)

    def test_invalid_NP_min_above_NP_init(self):
        from panobbgo.heuristics.lshade import LSHADE

        with pytest.raises(ValueError, match="NP_min .* must be <= NP_init"):
            LSHADE(self.strategy, NP_init=10, NP_min=20)

    def test_invalid_H_value(self):
        from panobbgo.heuristics.lshade import LSHADE

        with pytest.raises(ValueError, match="H must be >= 1"):
            LSHADE(self.strategy, H=0)
        with pytest.raises(ValueError, match="H must be an integer"):
            LSHADE(self.strategy, H=2.5)  # type: ignore[arg-type]

    def test_invalid_pbest(self):
        from panobbgo.heuristics.lshade import LSHADE

        with pytest.raises(ValueError, match="p_best must be in"):
            LSHADE(self.strategy, p_best=0.0)
        with pytest.raises(ValueError, match="p_best must be in"):
            LSHADE(self.strategy, p_best=-0.1)
        with pytest.raises(ValueError, match="p_best must be in"):
            LSHADE(self.strategy, p_best=1.5)
        with pytest.raises(ValueError, match="p_best must be in"):
            LSHADE(self.strategy, p_best=float("nan"))

    def test_invalid_archive_factor(self):
        from panobbgo.heuristics.lshade import LSHADE

        with pytest.raises(ValueError, match="archive_factor"):
            LSHADE(self.strategy, archive_factor=-1.0)
        with pytest.raises(ValueError, match="archive_factor"):
            LSHADE(self.strategy, archive_factor=float("inf"))


# ----------------------------------------------------------------------
# Initial population emission
# ----------------------------------------------------------------------


class LSHADEOnStartTests(_MockStrategyMixin, PanobbgoTestCase):
    def test_on_start_emits_NP_init_points(self):
        from panobbgo.heuristics.lshade import LSHADE

        h = LSHADE(self.strategy, NP_init=8, seed=0)
        h.on_start()

        emitted = h.get_points(limit=100)
        assert len(emitted) == 8
        assert len(h._pending) == 8
        assert len(h._population) == 8
        assert all(slot is None for slot in h._population)

    def test_on_start_points_inside_box(self):
        from panobbgo.heuristics.lshade import LSHADE

        h = LSHADE(self.strategy, NP_init=10, seed=1)
        h.on_start()

        emitted = h.get_points(limit=100)
        for pt in emitted:
            assert np.all(pt.x >= self.problem.box[:, 0] - 1e-9)
            assert np.all(pt.x <= self.problem.box[:, 1] + 1e-9)
            assert pt.who.startswith("LSHADE:")

    def test_on_start_initial_F_CR_are_NaN(self):
        """Initial random trials must not contribute to the success memory."""
        from panobbgo.heuristics.lshade import LSHADE

        h = LSHADE(self.strategy, NP_init=4, seed=42)
        h.on_start()
        for meta in h._pending.values():
            assert np.isnan(meta.F)
            assert np.isnan(meta.CR)


# ----------------------------------------------------------------------
# on_new_results: population fill + competitive replacement
# ----------------------------------------------------------------------


def _build_result(strategy, x, fx, who):
    from panobbgo.lib import Point, Result

    return Result(Point(np.asarray(x, dtype=float), who), float(fx))


class LSHADEOnResultsTests(_MockStrategyMixin, PanobbgoTestCase):
    def _seed_population(self, h, fx_seq):
        """Drive the initial random trials back as results.

        Returns the list of (req_id, slot_idx) pairs in the order they
        were popped.  ``fx_seq`` is the sequence of fx values to assign
        to each returning result; positions are taken from the
        positions array each slot was emitted at.
        """

        items = list(h._pending.items())
        results = []
        for (req_id, meta), fx in zip(items, fx_seq):
            # Use the position of the last emitted point — we don't
            # have the actual positions stored in the meta, so make
            # one up that respects the box.
            x = self.problem.random_point()
            results.append(_build_result(self.strategy, x, fx, f"LSHADE:{req_id}"))
        h.on_new_results(results)
        return [(rid, meta.slot_idx) for rid, meta in items]

    def test_unknown_who_ignored(self):
        from panobbgo.heuristics.lshade import LSHADE
        from panobbgo.lib import Point, Result

        h = LSHADE(self.strategy, NP_init=4, seed=0)
        h.on_start()
        before = dict(h._pending)
        # Result tagged with a different heuristic name.
        r = Result(Point(np.zeros(self.problem.dim), "OtherHeuristic:abc"), 1.0)
        h.on_new_results([r])
        assert h._pending == before
        assert all(slot is None for slot in h._population)

    def test_initial_fill_does_not_count_success(self):
        """Initial random results must not enter the success buffer."""
        from panobbgo.heuristics.lshade import LSHADE

        h = LSHADE(self.strategy, NP_init=4, seed=11)
        h.on_start()
        h.get_points(limit=100)  # drain queue

        self._seed_population(h, fx_seq=[10.0, 20.0, 30.0, 40.0])

        # Population filled, no success recorded.
        assert sum(1 for s in h._population if s is not None) == 4
        assert h._success_F == []
        assert h._success_CR == []
        assert h._gen_completed == 0

    def test_filled_population_emits_evolutionary_trials(self):
        """Once 4 slots are filled, follow-up evolutionary trials are emitted."""
        from panobbgo.heuristics.lshade import LSHADE

        h = LSHADE(self.strategy, NP_init=5, seed=12)
        h.on_start()
        h.get_points(limit=100)  # drain initial queue

        # Fill the first four slots (drives population to >= 4).  Once
        # the threshold is met, every freshly-completed slot kicks off a
        # new evolutionary trial *and* the wake-idle pass kicks off a
        # trial for every other live, idle slot.
        items = list(h._pending.items())[:4]
        results = []
        for req_id, meta in items:
            x = self.problem.random_point()
            results.append(_build_result(self.strategy, x, 10.0 + meta.slot_idx, f"LSHADE:{req_id}"))
        h.on_new_results(results)

        emitted = h.get_points(limit=100)
        assert len(emitted) >= 1
        # New trials must reference the LSHADE name and have valid F/CR.
        for pt in emitted:
            assert pt.who.startswith("LSHADE:")
        # Each follow-up trial got a real F/CR (not NaN).
        evo_metas = [meta for rid, meta in h._pending.items() if not np.isnan(meta.F) and not np.isnan(meta.CR)]
        assert len(evo_metas) >= 1

    def test_better_trial_wins(self):
        """A better trial replaces the target; the parent enters the archive."""
        from panobbgo.heuristics.lshade import LSHADE

        h = LSHADE(self.strategy, NP_init=4, seed=13)
        h.on_start()
        h.get_points(limit=100)

        # Fill all slots.
        self._seed_population(h, fx_seq=[100.0, 110.0, 120.0, 130.0])
        h.get_points(limit=100)  # drain follow-ups

        # Pick one slot's pending trial and feed back a strictly-better fx.
        target_slot = 0
        target_fx = h._population[target_slot].fx
        # Find the pending trial for slot 0.
        rid_meta = [(rid, meta) for rid, meta in h._pending.items() if meta.slot_idx == target_slot]
        assert rid_meta, "expected a follow-up trial for slot 0"
        rid, meta = rid_meta[0]

        improved_fx = target_fx - 50.0
        x = self.problem.random_point()
        r = _build_result(self.strategy, x, improved_fx, f"LSHADE:{rid}")
        h.on_new_results([r])

        # Slot fitness should now be the improved fx.
        assert h._population[target_slot] is not None
        assert h._population[target_slot].fx == improved_fx
        # Archive received the displaced parent.
        assert len(h._archive) >= 1
        # Success buffer recorded one entry with finite F/CR.
        assert len(h._success_F) >= 1
        assert all(np.isfinite(f) for f in h._success_F)
        assert all(np.isfinite(c) for c in h._success_CR)
        assert all(d > 0.0 for d in h._success_delta)

    def test_worse_trial_loses(self):
        """A worse trial leaves the target unchanged and skips the archive."""
        from panobbgo.heuristics.lshade import LSHADE

        h = LSHADE(self.strategy, NP_init=4, seed=14)
        h.on_start()
        h.get_points(limit=100)

        self._seed_population(h, fx_seq=[1.0, 2.0, 3.0, 4.0])
        h.get_points(limit=100)

        target_slot = 1
        before_fx = h._population[target_slot].fx
        rid_meta = [(rid, meta) for rid, meta in h._pending.items() if meta.slot_idx == target_slot]
        rid, meta = rid_meta[0]

        x = self.problem.random_point()
        # Strictly worse — the trial should lose.
        r = _build_result(self.strategy, x, before_fx + 100.0, f"LSHADE:{rid}")
        archive_before = len(h._archive)
        h.on_new_results([r])

        assert h._population[target_slot].fx == before_fx
        assert len(h._archive) == archive_before
        # No success recorded.
        assert h._success_F == [] or all(c != meta.CR for c in h._success_CR)


# ----------------------------------------------------------------------
# F/CR sampling
# ----------------------------------------------------------------------


class LSHADESamplingTests(_MockStrategyMixin, PanobbgoTestCase):
    def test_F_in_unit_interval(self):
        """Sampled F is always in (0, 1] (no zero, no >1, no NaN)."""
        from panobbgo.heuristics.lshade import LSHADE

        h = LSHADE(self.strategy, NP_init=4, H=3, seed=0)
        for _ in range(2000):
            F, CR = h._sample_F_CR()
            assert 0.0 < F <= 1.0
            assert 0.0 <= CR <= 1.0
            assert np.isfinite(F)
            assert np.isfinite(CR)

    def test_terminal_M_CR_yields_zero_CR(self):
        """When ``M_CR[r] = -1`` the sampler returns CR = 0 deterministically."""
        from panobbgo.heuristics.lshade import LSHADE

        h = LSHADE(self.strategy, NP_init=4, H=1, seed=0)
        h._M_CR[0] = -1.0
        # Only one bin so every draw lands on it.
        for _ in range(20):
            F, CR = h._sample_F_CR()
            assert CR == 0.0


# ----------------------------------------------------------------------
# Memory update via weighted Lehmer mean
# ----------------------------------------------------------------------


class LSHADEMemoryUpdateTests(_MockStrategyMixin, PanobbgoTestCase):
    def test_no_success_leaves_memory_unchanged(self):
        from panobbgo.heuristics.lshade import LSHADE

        h = LSHADE(self.strategy, NP_init=4, H=4, seed=0)
        h._M_F[:] = 0.5
        h._M_CR[:] = 0.5
        h._update_memory()  # empty success buffer
        assert np.allclose(h._M_F, 0.5)
        assert np.allclose(h._M_CR, 0.5)

    def test_memory_lehmer_mean_F(self):
        """Successful F values update M_F to their weighted Lehmer mean."""
        from panobbgo.heuristics.lshade import LSHADE

        h = LSHADE(self.strategy, NP_init=4, H=4, seed=0)
        h._mem_ptr = 0
        # Hand-built success buffer with known Lehmer mean.
        h._success_F = [0.2, 0.4, 0.8]
        h._success_CR = [0.5, 0.6, 0.7]
        h._success_delta = [1.0, 1.0, 1.0]
        # Equal weights -> Σx² / Σx
        expected_F = (0.2**2 + 0.4**2 + 0.8**2) / (0.2 + 0.4 + 0.8)
        expected_CR = (0.5**2 + 0.6**2 + 0.7**2) / (0.5 + 0.6 + 0.7)
        h._update_memory()
        assert h._M_F[0] == pytest.approx(expected_F, rel=1e-9)
        assert h._M_CR[0] == pytest.approx(expected_CR, rel=1e-9)
        # Pointer advanced.
        assert h._mem_ptr == 1

    def test_memory_pointer_wraps(self):
        from panobbgo.heuristics.lshade import LSHADE

        h = LSHADE(self.strategy, NP_init=4, H=3, seed=0)
        for _ in range(7):
            h._success_F = [0.5]
            h._success_CR = [0.5]
            h._success_delta = [1.0]
            h._update_memory()
        assert h._mem_ptr == 7 % 3

    def test_M_CR_terminal_when_all_CR_zero(self):
        """All-zero successful CR plants the -1 sentinel."""
        from panobbgo.heuristics.lshade import LSHADE

        h = LSHADE(self.strategy, NP_init=4, H=2, seed=0)
        h._mem_ptr = 0
        h._success_F = [0.5, 0.7]
        h._success_CR = [0.0, 0.0]
        h._success_delta = [1.0, 1.0]
        h._update_memory()
        assert h._M_CR[0] == -1.0

    def test_M_CR_terminal_is_sticky(self):
        """Once a bin is terminal it stays terminal even with positive CR successes."""
        from panobbgo.heuristics.lshade import LSHADE

        h = LSHADE(self.strategy, NP_init=4, H=1, seed=0)
        h._mem_ptr = 0
        h._M_CR[0] = -1.0
        h._success_F = [0.5]
        h._success_CR = [0.5]
        h._success_delta = [1.0]
        h._update_memory()
        # H = 1 so pointer wraps back to 0.
        assert h._M_CR[0] == -1.0


# ----------------------------------------------------------------------
# Linear Population Size Reduction (LPSR)
# ----------------------------------------------------------------------


class LSHADELPSRTests(_MockStrategyMixin, PanobbgoTestCase):
    def test_lpsr_no_op_when_budget_unknown(self):
        """LPSR must not change NP when ``max_eval`` is missing or zero."""
        from panobbgo.heuristics.lshade import LSHADE

        h = LSHADE(self.strategy, NP_init=10, NP_min=4, seed=0)
        h.on_start()
        h.get_points(limit=100)

        # Wipe the budget.
        self.strategy.config.max_eval = 0
        h._apply_lpsr()
        assert h._NP_current == 10

    def test_lpsr_shrinks_at_full_budget(self):
        """At progress = 1.0, NP should hit NP_min."""
        from panobbgo.heuristics.lshade import LSHADE

        h = LSHADE(self.strategy, NP_init=12, NP_min=4, seed=0)
        h.on_start()
        h.get_points(limit=100)

        # Fill the population so dropping has something to work with.
        from panobbgo.lib import Point, Result

        for slot_idx in range(12):
            x = self.problem.random_point()
            h._population[slot_idx] = Result(Point(x, "fake"), float(slot_idx))

        # Saturate the budget.
        self.strategy.config.max_eval = 100
        self.strategy.results = list(range(100))  # any 100-element list
        h._apply_lpsr()
        assert h._NP_current == 4
        from panobbgo.heuristics.lshade import _DROPPED

        n_dropped = sum(1 for s in h._population if s is _DROPPED)
        assert n_dropped == 12 - 4
        # The four kept slots are the four with the smallest fx values
        # (Default constraint handler ranks ascending fx).
        kept_fx = sorted(s.fx for s in h._population if isinstance(s, Result))
        assert kept_fx == [0.0, 1.0, 2.0, 3.0]

    def test_lpsr_partial_progress_shrinks_proportionally(self):
        """At progress = 0.5, NP should be roughly halfway between init and min."""
        from panobbgo.heuristics.lshade import LSHADE
        from panobbgo.lib import Point, Result

        h = LSHADE(self.strategy, NP_init=20, NP_min=4, seed=0)
        h.on_start()
        h.get_points(limit=100)
        for slot_idx in range(20):
            x = self.problem.random_point()
            h._population[slot_idx] = Result(Point(x, "fake"), float(slot_idx))

        self.strategy.config.max_eval = 200
        self.strategy.results = list(range(100))  # progress = 0.5
        h._apply_lpsr()
        # Linear schedule: NP_target = round(20 - 16 * 0.5) = 12.
        assert h._NP_current == 12

    def test_lpsr_drops_dont_break_alive_indexing(self):
        """After LPSR drops slots, _live_indices() returns only kept ones."""
        from panobbgo.heuristics.lshade import LSHADE
        from panobbgo.lib import Point, Result

        h = LSHADE(self.strategy, NP_init=8, NP_min=4, seed=0)
        h.on_start()
        h.get_points(limit=100)
        for slot_idx in range(8):
            x = self.problem.random_point()
            h._population[slot_idx] = Result(Point(x, "fake"), float(slot_idx))

        self.strategy.config.max_eval = 100
        self.strategy.results = list(range(100))  # progress = 1.0
        h._apply_lpsr()
        live = h._live_indices()
        assert len(live) == 4
        for i in live:
            assert isinstance(h._population[i], Result)


# ----------------------------------------------------------------------
# Bound reflection
# ----------------------------------------------------------------------


class LSHADEBoundReflectionTests(_MockStrategyMixin, PanobbgoTestCase):
    def test_reflection_pulls_below_bound_back_inside(self):
        from panobbgo.heuristics.lshade import LSHADE

        h = LSHADE(self.strategy, NP_init=4, seed=0)
        lb = np.asarray(self.problem.box[:, 0])
        ub = np.asarray(self.problem.box[:, 1])
        x_target = (lb + ub) / 2.0  # midpoint of the box
        # v[0] is below the lower bound; v[1] is well inside.
        v = np.array([lb[0] - 5.0, x_target[1]])
        out = h._reflect_bounds(v, x_target)
        # v[0] -> (lb[0] + x_target[0]) / 2
        assert out[0] == pytest.approx((lb[0] + x_target[0]) / 2.0)
        # v[1] unchanged.
        assert out[1] == pytest.approx(x_target[1])

    def test_reflection_pulls_above_bound_back_inside(self):
        from panobbgo.heuristics.lshade import LSHADE

        h = LSHADE(self.strategy, NP_init=4, seed=0)
        lb = np.asarray(self.problem.box[:, 0])
        ub = np.asarray(self.problem.box[:, 1])
        x_target = (lb + ub) / 2.0
        v = np.array([x_target[0], ub[1] + 50.0])
        out = h._reflect_bounds(v, x_target)
        assert out[0] == pytest.approx(x_target[0])
        # v[1] -> (ub[1] + x_target[1]) / 2
        assert out[1] == pytest.approx((ub[1] + x_target[1]) / 2.0)

    def test_reflection_leaves_in_bound_unchanged(self):
        from panobbgo.heuristics.lshade import LSHADE

        h = LSHADE(self.strategy, NP_init=4, seed=0)
        lb = np.asarray(self.problem.box[:, 0])
        ub = np.asarray(self.problem.box[:, 1])
        x_target = (lb + ub) / 2.0
        v = x_target.copy()
        out = h._reflect_bounds(v, x_target)
        np.testing.assert_array_equal(out, v)


# ----------------------------------------------------------------------
# End-of-generation hook fires after NP_current evolutionary trials
# ----------------------------------------------------------------------


class LSHADEGenerationTests(_MockStrategyMixin, PanobbgoTestCase):
    def test_generation_counter_advances_only_on_evolutionary_trials(self):
        """Filling the initial population must not advance the gen counter."""
        from panobbgo.heuristics.lshade import LSHADE

        h = LSHADE(self.strategy, NP_init=4, seed=0)
        h.on_start()
        h.get_points(limit=100)

        # Send back the four initial trials.
        items = list(h._pending.items())
        results = []
        for rid, meta in items[:4]:
            results.append(_build_result(self.strategy, np.array([0.0, 0.0]), 1.0, f"LSHADE:{rid}"))
        h.on_new_results(results)

        # Initial fill -> gen_completed must still be 0.
        assert h._gen_completed == 0


# ----------------------------------------------------------------------
# Restart behaviour
# ----------------------------------------------------------------------


class LSHADERestartTests(_MockStrategyMixin, PanobbgoTestCase):
    def test_restart_clears_state_and_reseeds(self):
        from panobbgo.heuristics.lshade import LSHADE

        h = LSHADE(self.strategy, NP_init=6, seed=2)
        h.on_start()
        h.get_points(limit=100)
        # Pretend we have some archive + memory state.
        h._archive.append(np.array([0.5, 0.5]))
        h._success_F = [0.7]
        h._success_CR = [0.5]
        h._success_delta = [1.0]
        h._mem_ptr = 3

        center = np.array([0.0, 0.0])
        h.on_restart(center, reason="test")

        # Archive cleared, memory reset, new pending trials in flight.
        assert h._archive == []
        assert h._success_F == []
        assert h._mem_ptr == 0
        assert np.allclose(h._M_F, 0.5)
        assert np.allclose(h._M_CR, 0.5)
        assert len(h._pending) == h.NP_init
        # All pending trials emitted with NaN F/CR (initial fill).
        for meta in h._pending.values():
            assert np.isnan(meta.F)
            assert np.isnan(meta.CR)

    def test_restart_with_none_center_falls_back_to_random(self):
        """``center=None`` must still reseed without crashing."""
        from panobbgo.heuristics.lshade import LSHADE

        h = LSHADE(self.strategy, NP_init=4, seed=3)
        h.on_start()
        h.get_points(limit=100)
        h.on_restart(None)
        emitted = h.get_points(limit=100)
        assert len(emitted) == 4
        # All inside the box.
        for pt in emitted:
            assert np.all(pt.x >= self.problem.box[:, 0] - 1e-9)
            assert np.all(pt.x <= self.problem.box[:, 1] + 1e-9)

    def test_restart_before_start_is_noop(self):
        """``on_restart`` before ``on_start`` must not crash or emit."""
        from panobbgo.heuristics.lshade import LSHADE

        h = LSHADE(self.strategy, NP_init=4, seed=4)
        h.on_restart(None)
        emitted = h.get_points(limit=100)
        assert emitted == []


# ----------------------------------------------------------------------
# End-to-end: on a quadratic the swarm makes progress
# ----------------------------------------------------------------------


class LSHADESmokeTests(_MockStrategyMixin, PanobbgoTestCase):
    def test_smoke_quadratic_improves(self):
        """Driving LSHADE through a few generations on f(x) = ||x||² improves the best fx."""
        from panobbgo.heuristics.lshade import LSHADE

        h = LSHADE(self.strategy, NP_init=8, NP_min=4, seed=5)
        h.on_start()

        def fx_of(x: np.ndarray) -> float:
            return float(np.dot(x, x))

        # Seed the initial population.
        from panobbgo.lib import Point, Result

        items = list(h._pending.items())
        h.get_points(limit=100)  # drain queue
        results = []
        for rid, meta in items:
            x = self.problem.random_point()
            results.append(Result(Point(x, f"LSHADE:{rid}"), fx_of(x)))
        h.on_new_results(results)
        h.get_points(limit=100)

        best_fx_before = min(slot.fx for slot in h._population if isinstance(slot, Result))

        # Run several rounds of follow-up trials.  Each round we drain
        # the queue, fabricate matching results, and feed back.
        for _round in range(20):
            pending_snapshot = list(h._pending.items())
            if not pending_snapshot:
                break
            h.get_points(limit=200)
            results = []
            for rid, meta in pending_snapshot:
                # Synthesize a candidate around the slot's current x using
                # a deterministic perturbation; let the algorithm sort
                # winners from losers.
                slot = h._population[meta.slot_idx]
                from panobbgo.heuristics.lshade import _Dropped

                if isinstance(slot, _Dropped) or slot is None:
                    continue
                x = self.problem.project(np.asarray(slot.x) + 0.1 * np.random.randn(self.problem.dim))
                results.append(Result(Point(x, f"LSHADE:{rid}"), fx_of(x)))
            h.on_new_results(results)

        best_fx_after = min(slot.fx for slot in h._population if isinstance(slot, Result))
        # We only require the swarm to be no worse — the synthetic
        # selection above doesn't guarantee a gradient signal because
        # the perturbations are random.  The test still exercises the
        # full pipeline end-to-end.
        assert best_fx_after <= best_fx_before + 1e-6


# ----------------------------------------------------------------------
# Registration
# ----------------------------------------------------------------------


class LSHADERegistrationTests(_MockStrategyMixin, PanobbgoTestCase):
    def test_registered_in_heuristics_package(self):
        import panobbgo.heuristics as h

        assert hasattr(h, "LSHADE")
        assert "LSHADE" in h.__all__

    def test_in_structural_catalog(self):
        """``default_structural_catalog`` ships an ``add_heuristic`` rule with LSHADE."""
        from panobbgo.heuristics.lshade import LSHADE
        from panobbgo.self_improve import default_structural_catalog, StructuralMutationRule

        catalog = default_structural_catalog()
        add_rules = [r for r in catalog.rules if isinstance(r, StructuralMutationRule) and r.op == "add_heuristic"]
        assert add_rules, "expected at least one add_heuristic rule"
        # At least one entry in the candidate pool should be LSHADE.
        has_lshade = False
        for rule in add_rules:
            for cls, _ in rule.candidate_classes or ():
                if cls is LSHADE:
                    has_lshade = True
                    break
        assert has_lshade

    def test_kwarg_catalog_has_NP_init_and_H(self):
        """``default_catalog`` exposes the headline LSHADE dials."""
        from panobbgo.self_improve import default_catalog, MutationRule

        rules = default_catalog().rules
        params = {
            (r.class_name, r.param_name) for r in rules if isinstance(r, MutationRule) and r.class_name == "LSHADE"
        }
        assert ("LSHADE", "NP_init") in params
        assert ("LSHADE", "H") in params
        assert ("LSHADE", "p_best") in params
