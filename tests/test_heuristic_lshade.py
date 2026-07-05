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

    def test_default_p_best_end_is_none(self):
        """Default ``p_best_end`` is ``None`` so behaviour stays byte-identical."""
        from panobbgo.heuristics.lshade import LSHADE

        h = LSHADE(self.strategy)
        assert h.p_best_end is None

    def test_custom_p_best_end_construction(self):
        """Opt-in iLSHADE / jSO schedule via ``p_best_end``."""
        from panobbgo.heuristics.lshade import LSHADE

        h = LSHADE(self.strategy, p_best=0.25, p_best_end=0.125)
        assert h.p_best == 0.25
        assert h.p_best_end == 0.125

    def test_invalid_p_best_end(self):
        """``p_best_end`` must be in ``(0, 1]`` when set."""
        from panobbgo.heuristics.lshade import LSHADE

        with pytest.raises(ValueError, match="p_best_end must be in"):
            LSHADE(self.strategy, p_best_end=0.0)
        with pytest.raises(ValueError, match="p_best_end must be in"):
            LSHADE(self.strategy, p_best_end=-0.05)
        with pytest.raises(ValueError, match="p_best_end must be in"):
            LSHADE(self.strategy, p_best_end=1.5)
        with pytest.raises(ValueError, match="p_best_end must be in"):
            LSHADE(self.strategy, p_best_end=float("nan"))
        with pytest.raises(ValueError, match="p_best_end must be in"):
            LSHADE(self.strategy, p_best_end=float("inf"))

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

    def test_np_init_auto_resolves_from_budget(self):
        """``NP_init="auto"`` sizes the population from budget and dimension."""
        from panobbgo.heuristics.lshade import LSHADE

        # dim=2 (Rosenbrock(2)).  budget/12 dominates when it is below 18*dim=36.
        self.strategy.config.max_eval = 75
        assert LSHADE(self.strategy, NP_init="auto").NP_init == 6  # round(75/12)=6
        self.strategy.config.max_eval = 240
        assert LSHADE(self.strategy, NP_init="auto").NP_init == 20  # round(240/12)=20
        # Large budget: the 18*dim=36 CEC upper bound caps the size.
        self.strategy.config.max_eval = 100000
        assert LSHADE(self.strategy, NP_init="auto").NP_init == 36

    def test_np_init_auto_floors_at_six(self):
        """Auto never resolves below 6 even at tiny budgets (avoids NP=4 degeneracy)."""
        from panobbgo.heuristics.lshade import LSHADE

        self.strategy.config.max_eval = 12  # round(12/12)=1 → floored to 6
        h = LSHADE(self.strategy, NP_init="auto")
        assert h.NP_init == 6
        assert h.NP_init >= h.NP_min

    def test_np_init_auto_respects_custom_np_min_floor(self):
        """A larger ``NP_min`` raises the auto floor so ``NP_min <= NP_init`` holds."""
        from panobbgo.heuristics.lshade import LSHADE

        self.strategy.config.max_eval = 60  # round(60/12)=5, below NP_min=10
        h = LSHADE(self.strategy, NP_init="auto", NP_min=10)
        assert h.NP_init == 10
        assert h.NP_min <= h.NP_init

    def test_np_init_auto_unknown_budget_falls_back(self):
        """Auto degrades to the fixed default when the budget is unknown."""
        from panobbgo.heuristics.lshade import LSHADE

        self.strategy.config.max_eval = 0  # non-positive → unknown
        assert LSHADE(self.strategy, NP_init="auto").NP_init == 30

    def test_np_init_auto_emits_resolved_count_on_start(self):
        """``on_start`` emits exactly the resolved auto population size."""
        from panobbgo.heuristics.lshade import LSHADE

        self.strategy.config.max_eval = 75
        h = LSHADE(self.strategy, NP_init="auto")
        h.__start__()
        h.on_start()
        assert len(h._pending) == h.NP_init == 6

    def test_np_init_invalid_string_raises(self):
        from panobbgo.heuristics.lshade import LSHADE

        with pytest.raises(ValueError, match="NP_init string must be 'auto'"):
            LSHADE(self.strategy, NP_init="big")

    def test_np_init_bool_rejected(self):
        """``True`` / ``False`` are ints but must not size a population."""
        from panobbgo.heuristics.lshade import LSHADE

        with pytest.raises(ValueError, match="NP_init must be an integer or 'auto'"):
            LSHADE(self.strategy, NP_init=True)  # type: ignore[arg-type]

    def test_np_init_auto_inherited_by_subclasses(self):
        """Subclasses (jSO / NL-SHADE-RSP / …) inherit budget-adaptive sizing."""
        from panobbgo.heuristics import JSO, NLSHADE_RSP, NLSHADE_LBC, LSHADE_EpSin

        self.strategy.config.max_eval = 75
        for cls in (JSO, NLSHADE_RSP, NLSHADE_LBC, LSHADE_EpSin):
            assert cls(self.strategy, NP_init="auto").NP_init == 6, cls.__name__

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


class LSHADEAdaptivePBestTests(_MockStrategyMixin, PanobbgoTestCase):
    """Tests for the iLSHADE / jSO adaptive ``p_best`` schedule."""

    def test_constant_when_p_best_end_is_none(self):
        """``_current_p_best`` returns ``self.p_best`` when the schedule is off."""
        from panobbgo.heuristics.lshade import LSHADE

        h = LSHADE(self.strategy, p_best=0.17)
        # Vary the simulated progress; the value must stay constant.
        self.strategy.results = []
        assert h._current_p_best() == pytest.approx(0.17)
        self.strategy.results = [None] * 500
        assert h._current_p_best() == pytest.approx(0.17)
        self.strategy.results = [None] * 10_000
        assert h._current_p_best() == pytest.approx(0.17)

    def test_linear_decrease_when_p_best_end_set(self):
        """Schedule honours the canonical jSO half-greediness annealing."""
        from panobbgo.heuristics.lshade import LSHADE

        h = LSHADE(self.strategy, p_best=0.25, p_best_end=0.125)
        self.strategy.config.max_eval = 1000

        self.strategy.results = []  # progress = 0 → p_best
        assert h._current_p_best() == pytest.approx(0.25)
        self.strategy.results = [None] * 500  # progress = 0.5 → mid
        assert h._current_p_best() == pytest.approx(0.1875)
        self.strategy.results = [None] * 1000  # progress = 1 → p_best_end
        assert h._current_p_best() == pytest.approx(0.125)

    def test_clipped_above_full_budget(self):
        """Progress > 1 clips to 1.0 and pins ``p_best_end``."""
        from panobbgo.heuristics.lshade import LSHADE

        h = LSHADE(self.strategy, p_best=0.25, p_best_end=0.10)
        self.strategy.config.max_eval = 100
        self.strategy.results = [None] * 500  # 5x budget — clip
        assert h._current_p_best() == pytest.approx(0.10)

    def test_linear_increase_when_p_best_end_above_p_best(self):
        """The schedule is symmetric: end > start is also supported."""
        from panobbgo.heuristics.lshade import LSHADE

        h = LSHADE(self.strategy, p_best=0.05, p_best_end=0.20)
        self.strategy.config.max_eval = 100

        self.strategy.results = []
        assert h._current_p_best() == pytest.approx(0.05)
        self.strategy.results = [None] * 50
        assert h._current_p_best() == pytest.approx(0.125)
        self.strategy.results = [None] * 100
        assert h._current_p_best() == pytest.approx(0.20)

    def test_constant_when_budget_unknown(self):
        """Falls back to ``self.p_best`` when ``max_eval`` is unusable."""
        from panobbgo.heuristics.lshade import LSHADE

        h = LSHADE(self.strategy, p_best=0.20, p_best_end=0.10)
        # Mid-run with a non-numeric / zero / non-finite budget.
        self.strategy.results = [None] * 100
        self.strategy.config.max_eval = 0
        assert h._current_p_best() == pytest.approx(0.20)
        self.strategy.config.max_eval = float("inf")
        assert h._current_p_best() == pytest.approx(0.20)
        self.strategy.config.max_eval = "nope"  # type: ignore[assignment]
        assert h._current_p_best() == pytest.approx(0.20)

    def test_p_best_end_equal_to_p_best_is_constant(self):
        """A no-op schedule (end == start) is harmless."""
        from panobbgo.heuristics.lshade import LSHADE

        h = LSHADE(self.strategy, p_best=0.15, p_best_end=0.15)
        self.strategy.config.max_eval = 100
        for n in (0, 25, 50, 75, 100):
            self.strategy.results = [None] * n
            assert h._current_p_best() == pytest.approx(0.15)

    def test_generate_trial_uses_scheduled_p_best(self):
        """End-to-end: ``_generate_trial`` consults the scheduled value.

        We populate a tiny LSHADE swarm by hand and verify the pbest
        pool count tracks the schedule.  At progress 0 (no results),
        ``p_best = 0.40`` over 5 live slots → pool size 2.  At progress
        1, ``p_best = 0.10`` → pool size 1.  We probe ``_current_p_best``
        directly because the pool sizing inside ``_generate_trial``
        uses ``ceil(p_eff · len)``.
        """
        from panobbgo.heuristics.lshade import LSHADE

        h = LSHADE(self.strategy, NP_init=10, p_best=0.40, p_best_end=0.10, seed=42)
        self.strategy.config.max_eval = 100

        self.strategy.results = []
        p_start = h._current_p_best()
        assert int(np.ceil(p_start * 5)) == 2

        self.strategy.results = [None] * 100
        p_end = h._current_p_best()
        assert int(np.ceil(p_end * 5)) == 1


class LSHADEAsymmetricFCapTests(_MockStrategyMixin, PanobbgoTestCase):
    """Opt-in jSO asymmetric F-cap (Brest et al. 2017) on the L-SHADE base.

    The cap is three-phase keyed on ``progress = len(results) / max_eval``:

    * ``progress < 0.6``        →  ``F ≤ 0.7``
    * ``0.6 ≤ progress < 0.9``  →  ``F ≤ 0.8``
    * ``progress ≥ 0.9``         →  ``F`` unclamped (still ≤ 1.0 from sampler)

    Off by default (``F_schedule=None``); jSO opts in by construction.
    """

    def test_default_F_schedule_is_none(self):
        """Default ``F_schedule`` is ``None`` — byte-identical L-SHADE."""
        from panobbgo.heuristics.lshade import LSHADE

        h = LSHADE(self.strategy)
        assert h.F_schedule is None

    def test_custom_F_schedule_construction_bool_compat(self):
        """``True`` / ``False`` are accepted as backwards-compat synonyms."""
        from panobbgo.heuristics.lshade import LSHADE

        h = LSHADE(self.strategy, F_schedule=True)
        assert h.F_schedule == "jso"

        h2 = LSHADE(self.strategy, F_schedule=False)
        assert h2.F_schedule is None

    def test_custom_F_schedule_construction_named_regimes(self):
        """Each named regime survives construction with the canonical name."""
        from panobbgo.heuristics.lshade import LSHADE

        for name in ("jso", "early", "strict"):
            h = LSHADE(self.strategy, F_schedule=name)
            assert h.F_schedule == name

        # ``"off"`` collapses onto ``None`` (the cap-disabled regime).
        h_off = LSHADE(self.strategy, F_schedule="off")
        assert h_off.F_schedule is None

    def test_invalid_F_schedule_type(self):
        """``F_schedule`` must be a known regime, bool, or ``None``."""
        from panobbgo.heuristics.lshade import LSHADE

        with pytest.raises(ValueError, match="F_schedule must be"):
            LSHADE(self.strategy, F_schedule=1)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="F_schedule must be"):
            LSHADE(self.strategy, F_schedule="yes")
        with pytest.raises(ValueError, match="F_schedule must be"):
            LSHADE(self.strategy, F_schedule="")
        with pytest.raises(ValueError, match="F_schedule must be"):
            LSHADE(self.strategy, F_schedule=2.0)  # type: ignore[arg-type]

    def test_apply_F_cap_disabled_by_default(self):
        """``F_schedule=None`` bypasses the cap entirely."""
        from panobbgo.heuristics.lshade import LSHADE

        h = LSHADE(self.strategy)
        self.strategy.config.max_eval = 100
        self.strategy.results = list(range(30))  # progress = 0.3
        # With F_schedule off, every input passes through unchanged.
        for F in (0.0, 0.1, 0.5, 0.7, 0.8, 0.95, 1.0):
            assert h._apply_F_cap(F) == pytest.approx(F)

    def test_apply_F_cap_disabled_explicitly_false(self):
        """``F_schedule=False`` is treated identically to ``None`` and to ``"off"``."""
        from panobbgo.heuristics.lshade import LSHADE

        for value in (False, "off"):
            h = LSHADE(self.strategy, F_schedule=value)
            self.strategy.config.max_eval = 100
            self.strategy.results = list(range(30))
            for F in (0.5, 0.9, 1.0):
                assert h._apply_F_cap(F) == pytest.approx(F)

    def test_apply_F_cap_phase1_clamps_to_07(self):
        """``progress < 0.6`` clamps F at 0.7 under the ``"jso"`` regime."""
        from panobbgo.heuristics.lshade import LSHADE

        h = LSHADE(self.strategy, F_schedule="jso")
        self.strategy.config.max_eval = 100
        self.strategy.results = list(range(30))  # progress = 0.3
        assert h._apply_F_cap(0.5) == pytest.approx(0.5)
        assert h._apply_F_cap(0.7) == pytest.approx(0.7)
        assert h._apply_F_cap(0.85) == pytest.approx(0.7)
        assert h._apply_F_cap(1.0) == pytest.approx(0.7)

    def test_apply_F_cap_phase2_clamps_to_08(self):
        """``0.6 ≤ progress < 0.9`` clamps F at 0.8 under ``"jso"`` — the literature completion."""
        from panobbgo.heuristics.lshade import LSHADE

        h = LSHADE(self.strategy, F_schedule="jso")
        self.strategy.config.max_eval = 100
        self.strategy.results = list(range(75))  # progress = 0.75 ∈ [0.6, 0.9)
        # F up to 0.8 passes through; F > 0.8 clamped to 0.8.
        assert h._apply_F_cap(0.5) == pytest.approx(0.5)
        assert h._apply_F_cap(0.75) == pytest.approx(0.75)
        assert h._apply_F_cap(0.8) == pytest.approx(0.8)
        assert h._apply_F_cap(0.95) == pytest.approx(0.8)
        assert h._apply_F_cap(1.0) == pytest.approx(0.8)

    def test_apply_F_cap_phase3_unclamped(self):
        """``progress ≥ 0.9`` releases the ``"jso"`` cap entirely (still F ≤ 1.0 from sampler)."""
        from panobbgo.heuristics.lshade import LSHADE

        h = LSHADE(self.strategy, F_schedule="jso")
        self.strategy.config.max_eval = 100
        self.strategy.results = list(range(95))  # progress = 0.95 ≥ 0.9
        for F in (0.0, 0.5, 0.7, 0.85, 0.95, 1.0):
            assert h._apply_F_cap(F) == pytest.approx(F)

    def test_apply_F_cap_phase_boundaries(self):
        """Phase boundaries are inclusive-lower: progress == 0.6 belongs to phase 2 (jso)."""
        from panobbgo.heuristics.lshade import LSHADE

        h = LSHADE(self.strategy, F_schedule="jso")
        self.strategy.config.max_eval = 100
        # progress = 0.6 exactly → phase 2 cap (0.8)
        self.strategy.results = list(range(60))
        assert h._apply_F_cap(0.95) == pytest.approx(0.8)
        # progress = 0.9 exactly → phase 3 (unclamped)
        self.strategy.results = list(range(90))
        assert h._apply_F_cap(0.95) == pytest.approx(0.95)

    def test_apply_F_cap_bypassed_when_budget_unknown(self):
        """No ``max_eval`` → ``_apply_F_cap`` is a pass-through (matches LPSR fallback)."""
        from panobbgo.heuristics.lshade import LSHADE

        h = LSHADE(self.strategy, F_schedule="jso")
        self.strategy.config.max_eval = 0
        for F in (0.5, 0.85, 1.0):
            assert h._apply_F_cap(F) == pytest.approx(F)

    def test_apply_F_cap_early_regime(self):
        """``"early"`` regime kicks in earlier and tighter than jSO."""
        from panobbgo.heuristics.lshade import LSHADE, _F_SCHEDULE_REGIMES

        # Sanity-check the regime tuple matches the docstring claim.
        assert _F_SCHEDULE_REGIMES["early"] == (0.4, 0.7, 0.6, 0.8)

        h = LSHADE(self.strategy, F_schedule="early")
        self.strategy.config.max_eval = 100

        # Phase 1: progress 0.2 < 0.4 → clamp at 0.6.
        self.strategy.results = list(range(20))
        assert h._apply_F_cap(0.5) == pytest.approx(0.5)
        assert h._apply_F_cap(0.7) == pytest.approx(0.6)
        assert h._apply_F_cap(1.0) == pytest.approx(0.6)

        # Phase 2: progress 0.55 ∈ [0.4, 0.7) → clamp at 0.8.
        self.strategy.results = list(range(55))
        assert h._apply_F_cap(0.7) == pytest.approx(0.7)
        assert h._apply_F_cap(0.95) == pytest.approx(0.8)

        # Phase 3: progress 0.85 ≥ 0.7 → unclamped.
        self.strategy.results = list(range(85))
        assert h._apply_F_cap(0.95) == pytest.approx(0.95)

    def test_apply_F_cap_strict_regime(self):
        """``"strict"`` regime is the most aggressive cap."""
        from panobbgo.heuristics.lshade import LSHADE, _F_SCHEDULE_REGIMES

        # Sanity-check the regime tuple matches the docstring claim.
        assert _F_SCHEDULE_REGIMES["strict"] == (0.5, 0.85, 0.5, 0.7)

        h = LSHADE(self.strategy, F_schedule="strict")
        self.strategy.config.max_eval = 100

        # Phase 1: progress 0.25 < 0.5 → clamp at 0.5.
        self.strategy.results = list(range(25))
        assert h._apply_F_cap(0.3) == pytest.approx(0.3)
        assert h._apply_F_cap(0.7) == pytest.approx(0.5)
        assert h._apply_F_cap(1.0) == pytest.approx(0.5)

        # Phase 2: progress 0.7 ∈ [0.5, 0.85) → clamp at 0.7.
        self.strategy.results = list(range(70))
        assert h._apply_F_cap(0.6) == pytest.approx(0.6)
        assert h._apply_F_cap(0.9) == pytest.approx(0.7)

        # Phase 3: progress 0.9 ≥ 0.85 → unclamped.
        self.strategy.results = list(range(90))
        assert h._apply_F_cap(0.95) == pytest.approx(0.95)

    def test_apply_F_cap_regime_dict_is_complete(self):
        """Every named regime has a well-formed 4-tuple in :data:`_F_SCHEDULE_REGIMES`."""
        from panobbgo.heuristics.lshade import _F_SCHEDULE_REGIMES

        assert set(_F_SCHEDULE_REGIMES.keys()) == {"jso", "early", "strict"}
        for name, params in _F_SCHEDULE_REGIMES.items():
            bound1, bound2, cap1, cap2 = params
            assert 0.0 < bound1 < bound2 <= 1.0, f"{name}: phase bounds malformed"
            assert 0.0 < cap1 <= cap2 <= 1.0, f"{name}: caps malformed"

    def test_sample_F_CR_respects_F_schedule(self):
        """When ``F_schedule="jso"``, drawn F never exceeds the current cap."""
        from panobbgo.heuristics.lshade import LSHADE

        h = LSHADE(self.strategy, F_schedule="jso", H=2, seed=42)
        self.strategy.config.max_eval = 100
        # Force a high M_F so the Cauchy distribution often produces F > 0.7.
        h._M_F[:] = 0.95
        h._M_CR[:] = 0.5

        # Phase 1.
        self.strategy.results = list(range(20))  # progress = 0.2
        for _ in range(500):
            F, _ = h._sample_F_CR()
            assert F <= 0.7 + 1e-12

        # Phase 2.
        self.strategy.results = list(range(75))  # progress = 0.75
        for _ in range(500):
            F, _ = h._sample_F_CR()
            assert F <= 0.8 + 1e-12

        # Phase 3 — at least one draw should exceed 0.8.
        self.strategy.results = list(range(95))  # progress = 0.95
        any_above_08 = any(h._sample_F_CR()[0] > 0.8 for _ in range(500))
        assert any_above_08

    def test_progress_returns_none_without_budget(self):
        """``_progress()`` returns ``None`` when ``max_eval`` is missing/zero."""
        from panobbgo.heuristics.lshade import LSHADE

        h = LSHADE(self.strategy)
        self.strategy.config.max_eval = 0
        assert h._progress() is None
        self.strategy.config.max_eval = -1
        assert h._progress() is None

    def test_progress_clipped_to_unit_interval(self):
        """``_progress()`` clips overshoots to [0, 1]."""
        from panobbgo.heuristics.lshade import LSHADE

        h = LSHADE(self.strategy)
        self.strategy.config.max_eval = 100
        self.strategy.results = []
        assert h._progress() == pytest.approx(0.0)
        self.strategy.results = list(range(50))
        assert h._progress() == pytest.approx(0.5)
        self.strategy.results = list(range(150))
        assert h._progress() == pytest.approx(1.0)


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
        # iLSHADE / jSO adaptive p_best schedule (opt-in via spec kwarg).
        assert ("LSHADE", "p_best_end") in params
        # jSO asymmetric F-cap (opt-in via spec kwarg).
        assert ("LSHADE", "F_schedule") in params
