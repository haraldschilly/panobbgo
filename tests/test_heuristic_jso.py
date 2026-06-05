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

"""Tests for the jSO (Brest 2017) adaptive Differential Evolution heuristic."""

from __future__ import annotations

import numpy as np
import pytest

from panobbgo.utils import PanobbgoTestCase


class _MockStrategyMixin:
    """Identical setup pattern to :mod:`tests.test_heuristic_lshade`.

    jSO inherits LPSR / constraint-handler / max_eval semantics from
    L-SHADE, so the mock strategy needs the same scaffolding.  The
    singleton ``config.max_eval`` is saved and restored on each test
    to prevent cross-test bleed.
    """

    def setUp(self):
        super().setUp()
        from panobbgo.lib.constraints import DefaultConstraintHandler

        self.strategy.constraint_handler = DefaultConstraintHandler(self.strategy)
        self._orig_max_eval = self.strategy.config.max_eval
        self.strategy.config.max_eval = 1000
        self.strategy.results = []

    def tearDown(self):
        self.strategy.config.max_eval = self._orig_max_eval
        super().tearDown()


# ----------------------------------------------------------------------
# Construction-time validation
# ----------------------------------------------------------------------


class JSOConstructionTests(_MockStrategyMixin, PanobbgoTestCase):
    def test_default_construction(self):
        from panobbgo.heuristics.jso import (
            JSO,
            _ANCHOR_M_CR,
            _ANCHOR_M_F,
            _INIT_M_CR,
            _INIT_M_F,
        )

        h = JSO(self.strategy)
        # Defaults match Brest et al. 2017.
        assert h.NP_init == 30
        assert h.NP_min == 4
        assert h.H == 5  # vs L-SHADE's 6
        assert h.p_best_max == 0.25
        assert h.p_best_min == 0.125
        assert h.archive_factor == 1.0
        assert h.name == "JSO"
        # Memory bins: writable bins seeded with jSO defaults; anchor bin frozen.
        assert h._M_F.shape == (h.H,)
        assert np.allclose(h._M_F[:-1], _INIT_M_F)
        assert np.allclose(h._M_CR[:-1], _INIT_M_CR)
        assert h._M_F[-1] == _ANCHOR_M_F
        assert h._M_CR[-1] == _ANCHOR_M_CR
        assert h._mem_ptr == 0

    def test_custom_construction(self):
        from panobbgo.heuristics.jso import JSO

        h = JSO(
            self.strategy,
            NP_init=20,
            NP_min=6,
            H=4,
            p_best_max=0.3,
            p_best_min=0.1,
            archive_factor=2.0,
            seed=7,
            name="MyJSO",
        )
        assert h.NP_init == 20
        assert h.NP_min == 6
        assert h.H == 4
        assert h.p_best_max == 0.3
        assert h.p_best_min == 0.1
        assert h.archive_factor == 2.0
        assert h.name == "MyJSO"

    def test_subclass_of_lshade(self):
        """jSO inherits the entire LSHADE asynchronous pipeline."""
        from panobbgo.heuristics.jso import JSO
        from panobbgo.heuristics.lshade import LSHADE

        h = JSO(self.strategy)
        assert isinstance(h, LSHADE)

    def test_invalid_H_below_two(self):
        """H must be at least 2 to keep the anchor bin distinct from writable bins."""
        from panobbgo.heuristics.jso import JSO

        with pytest.raises(ValueError, match="H must be >= 2"):
            JSO(self.strategy, H=1)
        with pytest.raises(ValueError, match="H must be an integer"):
            JSO(self.strategy, H=2.5)  # type: ignore[arg-type]

    def test_invalid_p_best_max(self):
        from panobbgo.heuristics.jso import JSO

        with pytest.raises(ValueError, match="p_best_max"):
            JSO(self.strategy, p_best_max=0.0)
        with pytest.raises(ValueError, match="p_best_max"):
            JSO(self.strategy, p_best_max=1.5)
        with pytest.raises(ValueError, match="p_best_max"):
            JSO(self.strategy, p_best_max=float("nan"))

    def test_invalid_p_best_min(self):
        from panobbgo.heuristics.jso import JSO

        with pytest.raises(ValueError, match="p_best_min"):
            JSO(self.strategy, p_best_min=0.0)
        with pytest.raises(ValueError, match="p_best_min"):
            JSO(self.strategy, p_best_min=1.1)

    def test_p_best_min_must_be_at_most_p_best_max(self):
        from panobbgo.heuristics.jso import JSO

        with pytest.raises(ValueError, match="p_best_min .* must be <= p_best_max"):
            JSO(self.strategy, p_best_max=0.2, p_best_min=0.3)


# ----------------------------------------------------------------------
# Memory anchor invariants
# ----------------------------------------------------------------------


class JSOMemoryAnchorTests(_MockStrategyMixin, PanobbgoTestCase):
    def test_anchor_bin_frozen_at_construction(self):
        from panobbgo.heuristics.jso import JSO, _ANCHOR_M_CR, _ANCHOR_M_F

        h = JSO(self.strategy, H=5)
        assert h._M_F[-1] == _ANCHOR_M_F
        assert h._M_CR[-1] == _ANCHOR_M_CR

    def test_update_memory_never_writes_to_anchor(self):
        """Many memory updates in sequence must never touch the anchor bin."""
        from panobbgo.heuristics.jso import JSO, _ANCHOR_M_CR, _ANCHOR_M_F

        h = JSO(self.strategy, H=4, seed=0)
        anchor_idx = h.H - 1
        # Run far more updates than H so the pointer wraps several times.
        for k in range(50):
            h._success_F = [0.5 + 0.001 * k]
            h._success_CR = [0.5]
            h._success_delta = [1.0]
            h._update_memory()
        # Anchor bin still pristine.
        assert h._M_F[anchor_idx] == _ANCHOR_M_F
        assert h._M_CR[anchor_idx] == _ANCHOR_M_CR

    def test_pointer_wraps_over_writable_range_only(self):
        """Pointer must cycle through ``[0, H-2]`` and never land on H-1."""
        from panobbgo.heuristics.jso import JSO

        h = JSO(self.strategy, H=4, seed=0)  # writable range = [0, 1, 2]
        seen = set()
        for k in range(30):
            seen.add(h._mem_ptr)
            h._success_F = [0.5]
            h._success_CR = [0.5]
            h._success_delta = [1.0]
            h._update_memory()
        # Pointer values stayed strictly below H - 1 = 3.
        assert seen.issubset({0, 1, 2})
        # Pointer covered the entire writable range (no skips).
        assert seen == {0, 1, 2}

    def test_writable_bin_updated_via_lehmer_mean(self):
        """``_update_memory`` still writes successful F/CR Lehmer means to bin 0."""
        from panobbgo.heuristics.jso import JSO

        h = JSO(self.strategy, H=3, seed=0)
        h._mem_ptr = 0
        h._success_F = [0.2, 0.4, 0.8]
        h._success_CR = [0.5, 0.6, 0.7]
        h._success_delta = [1.0, 1.0, 1.0]
        expected_F = (0.2**2 + 0.4**2 + 0.8**2) / (0.2 + 0.4 + 0.8)
        expected_CR = (0.5**2 + 0.6**2 + 0.7**2) / (0.5 + 0.6 + 0.7)
        h._update_memory()
        assert h._M_F[0] == pytest.approx(expected_F, rel=1e-9)
        assert h._M_CR[0] == pytest.approx(expected_CR, rel=1e-9)

    def test_no_success_leaves_memory_unchanged(self):
        from panobbgo.heuristics.jso import JSO

        h = JSO(self.strategy, H=4, seed=0)
        before_F = h._M_F.copy()
        before_CR = h._M_CR.copy()
        h._update_memory()
        np.testing.assert_array_equal(h._M_F, before_F)
        np.testing.assert_array_equal(h._M_CR, before_CR)


# ----------------------------------------------------------------------
# Schedules: progress / p_best / F_w
# ----------------------------------------------------------------------


class JSOScheduleTests(_MockStrategyMixin, PanobbgoTestCase):
    def test_progress_clipped_to_unit_interval(self):
        """``_progress`` must clip to [0, 1] regardless of result-count overshoot."""
        from panobbgo.heuristics.jso import JSO

        h = JSO(self.strategy)
        self.strategy.config.max_eval = 100
        self.strategy.results = list(range(50))
        assert h._progress() == pytest.approx(0.5)
        self.strategy.results = list(range(150))  # over-spent budget
        assert h._progress() == pytest.approx(1.0)
        self.strategy.results = []
        assert h._progress() == pytest.approx(0.0)

    def test_progress_returns_none_without_budget(self):
        """No ``max_eval`` → ``progress`` returns ``None`` so each schedule picks its own fall-back.

        The ``_progress()`` helper inherited from L-SHADE returns ``None``
        when the budget is unknown so callers (``_current_p_best``,
        ``_current_F_weight``, ``_apply_F_cap``, ``_apply_lpsr``) can each
        pick the right early-phase fall-back instead of being forced to
        share ``progress = 0.0``.
        """
        from panobbgo.heuristics.jso import JSO

        h = JSO(self.strategy)
        self.strategy.config.max_eval = 0
        assert h._progress() is None

    def test_p_best_schedule_is_linear_decreasing(self):
        """``_current_p_best`` decreases linearly from p_best_max to p_best_min."""
        from panobbgo.heuristics.jso import JSO

        h = JSO(self.strategy, p_best_max=0.25, p_best_min=0.125)
        self.strategy.config.max_eval = 100
        self.strategy.results = []
        assert h._current_p_best() == pytest.approx(0.25)
        self.strategy.results = list(range(50))
        assert h._current_p_best() == pytest.approx((0.25 + 0.125) / 2.0)
        self.strategy.results = list(range(100))
        assert h._current_p_best() == pytest.approx(0.125)

    def test_p_best_falls_back_to_max_when_budget_unknown(self):
        """No ``max_eval`` → ``_current_p_best`` returns ``p_best_max`` (early-phase value)."""
        from panobbgo.heuristics.jso import JSO

        h = JSO(self.strategy, p_best_max=0.4, p_best_min=0.05)
        self.strategy.config.max_eval = 0
        assert h._current_p_best() == pytest.approx(0.4)

    def test_F_weight_three_phase_schedule(self):
        """``_current_F_weight`` returns 0.7 / 0.8 / 1.2 for the three regimes."""
        from panobbgo.heuristics.jso import JSO

        h = JSO(self.strategy)
        self.strategy.config.max_eval = 100
        # Phase 1: progress < 0.2 → 0.7
        self.strategy.results = list(range(10))
        assert h._current_F_weight() == pytest.approx(0.7)
        # Phase 2: 0.2 <= progress < 0.4 → 0.8
        self.strategy.results = list(range(30))
        assert h._current_F_weight() == pytest.approx(0.8)
        # Phase 3: progress >= 0.4 → 1.2
        self.strategy.results = list(range(50))
        assert h._current_F_weight() == pytest.approx(1.2)
        self.strategy.results = list(range(100))
        assert h._current_F_weight() == pytest.approx(1.2)

    def test_F_weight_phase_boundaries_inclusive_lower(self):
        """Phase boundaries: ``progress == 0.2`` belongs to phase 2."""
        from panobbgo.heuristics.jso import JSO

        h = JSO(self.strategy)
        self.strategy.config.max_eval = 100
        # exactly progress = 0.2 → phase 2 (F_w = 0.8)
        self.strategy.results = list(range(20))
        assert h._current_F_weight() == pytest.approx(0.8)
        # exactly progress = 0.4 → phase 3 (F_w = 1.2)
        self.strategy.results = list(range(40))
        assert h._current_F_weight() == pytest.approx(1.2)

    def test_F_weight_falls_back_to_phase1_when_budget_unknown(self):
        """No ``max_eval`` → ``_current_F_weight`` returns the early-phase factor."""
        from panobbgo.heuristics.jso import JSO

        h = JSO(self.strategy)
        self.strategy.config.max_eval = 0
        assert h._current_F_weight() == pytest.approx(0.7)


# ----------------------------------------------------------------------
# Asymmetric Cauchy-F clamping — three-phase schedule (Brest 2017)
# ----------------------------------------------------------------------


class JSOAsymmetricFCapTests(_MockStrategyMixin, PanobbgoTestCase):
    """jSO opts into the L-SHADE three-phase asymmetric F-cap.

    The cap schedule (literature-faithful Brest et al. 2017 §III-D):

    * ``progress < 0.6``        →  ``F ≤ 0.7``
    * ``0.6 ≤ progress < 0.9``  →  ``F ≤ 0.8``
    * ``progress ≥ 0.9``         →  ``F`` unclamped (still ≤ 1.0 from sampler)
    """

    def test_jso_opts_into_F_schedule_by_construction(self):
        """jSO sets ``F_schedule=True`` on the L-SHADE base class."""
        from panobbgo.heuristics.jso import JSO

        h = JSO(self.strategy, seed=0)
        assert h.F_schedule is True

    def test_F_clamped_at_07_when_progress_below_60_percent(self):
        from panobbgo.heuristics.jso import JSO

        h = JSO(self.strategy, H=2, seed=42)
        self.strategy.config.max_eval = 100
        self.strategy.results = list(range(30))  # progress = 0.3 < 0.6
        # Force the parent's bin to a value high enough that draws often exceed 0.7.
        h._M_F[0] = 0.9
        h._M_CR[0] = 0.5
        # The anchor bin (1) is also 0.9 by construction; no need to override.
        for _ in range(500):
            F, _ = h._sample_F_CR()
            assert F <= 0.7 + 1e-12

    def test_F_clamped_at_08_in_middle_phase(self):
        """In the [0.6, 0.9) phase, F is capped at 0.8 (not 0.7, not unclamped)."""
        from panobbgo.heuristics.jso import JSO

        h = JSO(self.strategy, H=2, seed=42)
        self.strategy.config.max_eval = 100
        self.strategy.results = list(range(75))  # progress = 0.75 ∈ [0.6, 0.9)
        h._M_F[0] = 0.95
        h._M_CR[0] = 0.5
        saw_above_07 = False
        for _ in range(500):
            F, _ = h._sample_F_CR()
            assert F <= 0.8 + 1e-12  # never above 0.8
            if F > 0.7:
                saw_above_07 = True
        assert saw_above_07, "phase-2 cap should permit F > 0.7"

    def test_F_unclamped_in_final_10_percent(self):
        """At progress >= 0.9 the clamp is released entirely."""
        from panobbgo.heuristics.jso import JSO

        h = JSO(self.strategy, H=2, seed=42)
        self.strategy.config.max_eval = 100
        self.strategy.results = list(range(95))  # progress = 0.95 >= 0.9
        h._M_F[0] = 0.95
        h._M_CR[0] = 0.5
        # With high M_F and 500 draws we expect at least one F > 0.8.
        any_above_08 = any(h._sample_F_CR()[0] > 0.8 for _ in range(500))
        assert any_above_08

    def test_F_in_unit_interval_always(self):
        """Sampled F is always in (0, 1] regardless of phase."""
        from panobbgo.heuristics.jso import JSO

        h = JSO(self.strategy, H=3, seed=0)
        for progress_pct in (0, 30, 60, 80, 95):
            self.strategy.results = list(range(progress_pct))
            self.strategy.config.max_eval = 100
            for _ in range(500):
                F, CR = h._sample_F_CR()
                assert 0.0 < F <= 1.0
                assert 0.0 <= CR <= 1.0
                assert np.isfinite(F)
                assert np.isfinite(CR)


# ----------------------------------------------------------------------
# Initial population emission
# ----------------------------------------------------------------------


class JSOOnStartTests(_MockStrategyMixin, PanobbgoTestCase):
    def test_on_start_emits_NP_init_points(self):
        from panobbgo.heuristics.jso import JSO

        h = JSO(self.strategy, NP_init=8, seed=0)
        h.on_start()

        emitted = h.get_points(limit=100)
        assert len(emitted) == 8
        assert len(h._pending) == 8
        assert len(h._population) == 8
        assert all(slot is None for slot in h._population)

    def test_on_start_uses_jso_initial_memory(self):
        """``on_start`` re-stamps the memory bins with jSO defaults (not LSHADE's)."""
        from panobbgo.heuristics.jso import JSO, _ANCHOR_M_CR, _ANCHOR_M_F, _INIT_M_CR, _INIT_M_F

        h = JSO(self.strategy, NP_init=4, H=5, seed=0)
        # Mutate memory to non-default values, then on_start should reset it.
        h._M_F[:] = 0.123
        h._M_CR[:] = 0.456
        h.on_start()
        assert np.allclose(h._M_F[:-1], _INIT_M_F)
        assert np.allclose(h._M_CR[:-1], _INIT_M_CR)
        assert h._M_F[-1] == _ANCHOR_M_F
        assert h._M_CR[-1] == _ANCHOR_M_CR

    def test_on_start_initial_F_CR_are_NaN(self):
        from panobbgo.heuristics.jso import JSO

        h = JSO(self.strategy, NP_init=4, seed=42)
        h.on_start()
        for meta in h._pending.values():
            assert np.isnan(meta.F)
            assert np.isnan(meta.CR)

    def test_on_start_points_inside_box(self):
        from panobbgo.heuristics.jso import JSO

        h = JSO(self.strategy, NP_init=10, seed=1)
        h.on_start()
        emitted = h.get_points(limit=100)
        for pt in emitted:
            assert np.all(pt.x >= self.problem.box[:, 0] - 1e-9)
            assert np.all(pt.x <= self.problem.box[:, 1] + 1e-9)
            assert pt.who.startswith("JSO:")


# ----------------------------------------------------------------------
# Generate trial: weighted current-to-pbest-w/1
# ----------------------------------------------------------------------


def _build_result(strategy, x, fx, who):
    from panobbgo.lib import Point, Result

    return Result(Point(np.asarray(x, dtype=float), who), float(fx))


class JSOGenerateTrialTests(_MockStrategyMixin, PanobbgoTestCase):
    def _seed_population(self, h, fx_seq):
        items = list(h._pending.items())
        results = []
        for (req_id, _meta), fx in zip(items, fx_seq):
            x = self.problem.random_point()
            results.append(_build_result(self.strategy, x, fx, f"JSO:{req_id}"))
        h.on_new_results(results)

    def test_filled_population_emits_evolutionary_trials(self):
        from panobbgo.heuristics.jso import JSO

        h = JSO(self.strategy, NP_init=5, seed=12)
        h.on_start()
        h.get_points(limit=100)

        items = list(h._pending.items())[:4]
        results = []
        for req_id, meta in items:
            x = self.problem.random_point()
            results.append(_build_result(self.strategy, x, 10.0 + meta.slot_idx, f"JSO:{req_id}"))
        h.on_new_results(results)

        emitted = h.get_points(limit=100)
        assert len(emitted) >= 1
        for pt in emitted:
            assert pt.who.startswith("JSO:")
        evo_metas = [m for _, m in h._pending.items() if not np.isnan(m.F) and not np.isnan(m.CR)]
        assert len(evo_metas) >= 1

    def test_better_trial_wins_and_archives_parent(self):
        """A better trial replaces the target; the parent enters the archive."""
        from panobbgo.heuristics.jso import JSO

        h = JSO(self.strategy, NP_init=4, seed=13)
        h.on_start()
        h.get_points(limit=100)

        self._seed_population(h, fx_seq=[100.0, 110.0, 120.0, 130.0])
        h.get_points(limit=100)

        target_slot = 0
        target_fx = h._population[target_slot].fx
        rid_meta = [(r, m) for r, m in h._pending.items() if m.slot_idx == target_slot]
        assert rid_meta
        rid, _ = rid_meta[0]

        improved_fx = target_fx - 50.0
        x = self.problem.random_point()
        r = _build_result(self.strategy, x, improved_fx, f"JSO:{rid}")
        h.on_new_results([r])

        assert h._population[target_slot].fx == improved_fx
        assert len(h._archive) >= 1
        assert len(h._success_F) >= 1
        assert all(np.isfinite(f) for f in h._success_F)
        assert all(np.isfinite(c) for c in h._success_CR)


# ----------------------------------------------------------------------
# Restart behaviour
# ----------------------------------------------------------------------


class JSORestartTests(_MockStrategyMixin, PanobbgoTestCase):
    def test_restart_re_stamps_jso_initial_memory(self):
        from panobbgo.heuristics.jso import JSO, _ANCHOR_M_CR, _ANCHOR_M_F, _INIT_M_CR, _INIT_M_F

        h = JSO(self.strategy, NP_init=6, seed=2)
        h.on_start()
        h.get_points(limit=100)
        # Mutate the memory.
        h._M_F[:] = 0.6
        h._M_CR[:] = 0.4
        h._archive.append(np.array([0.5, 0.5]))
        h._success_F = [0.7]

        center = np.array([0.0, 0.0])
        h.on_restart(center, reason="test")

        # Archive cleared, memory reset to jSO defaults, anchor bin restored.
        assert h._archive == []
        assert h._success_F == []
        assert np.allclose(h._M_F[:-1], _INIT_M_F)
        assert np.allclose(h._M_CR[:-1], _INIT_M_CR)
        assert h._M_F[-1] == _ANCHOR_M_F
        assert h._M_CR[-1] == _ANCHOR_M_CR
        assert len(h._pending) == h.NP_init
        for meta in h._pending.values():
            assert np.isnan(meta.F)
            assert np.isnan(meta.CR)

    def test_restart_with_none_center_falls_back_to_random(self):
        from panobbgo.heuristics.jso import JSO

        h = JSO(self.strategy, NP_init=4, seed=3)
        h.on_start()
        h.get_points(limit=100)
        h.on_restart(None)
        emitted = h.get_points(limit=100)
        assert len(emitted) == 4
        for pt in emitted:
            assert np.all(pt.x >= self.problem.box[:, 0] - 1e-9)
            assert np.all(pt.x <= self.problem.box[:, 1] + 1e-9)

    def test_restart_before_start_is_noop(self):
        from panobbgo.heuristics.jso import JSO

        h = JSO(self.strategy, NP_init=4, seed=4)
        h.on_restart(None)
        emitted = h.get_points(limit=100)
        assert emitted == []


# ----------------------------------------------------------------------
# Smoke test: end-to-end progress on a simple landscape
# ----------------------------------------------------------------------


class JSOSmokeTests(_MockStrategyMixin, PanobbgoTestCase):
    def test_smoke_quadratic_no_regression(self):
        """Driving JSO through a few generations on f(x) = ||x||² makes no negative progress."""
        from panobbgo.heuristics.jso import JSO
        from panobbgo.lib import Point, Result

        h = JSO(self.strategy, NP_init=8, NP_min=4, seed=5)
        h.on_start()

        def fx_of(x):
            return float(np.dot(x, x))

        items = list(h._pending.items())
        h.get_points(limit=100)
        results = []
        for rid, _meta in items:
            x = self.problem.random_point()
            results.append(Result(Point(x, f"JSO:{rid}"), fx_of(x)))
        h.on_new_results(results)
        h.get_points(limit=100)

        best_fx_before = min(s.fx for s in h._population if isinstance(s, Result))

        for _round in range(20):
            pending_snapshot = list(h._pending.items())
            if not pending_snapshot:
                break
            h.get_points(limit=200)
            results = []
            from panobbgo.heuristics.lshade import _Dropped

            for rid, meta in pending_snapshot:
                slot = h._population[meta.slot_idx]
                if isinstance(slot, _Dropped) or slot is None:
                    continue
                x = self.problem.project(np.asarray(slot.x) + 0.1 * np.random.randn(self.problem.dim))
                results.append(Result(Point(x, f"JSO:{rid}"), fx_of(x)))
            h.on_new_results(results)

        best_fx_after = min(s.fx for s in h._population if isinstance(s, Result))
        # Population must not regress globally.
        assert best_fx_after <= best_fx_before + 1e-6


# ----------------------------------------------------------------------
# Registration
# ----------------------------------------------------------------------


class JSORegistrationTests(_MockStrategyMixin, PanobbgoTestCase):
    def test_registered_in_heuristics_package(self):
        import panobbgo.heuristics as h

        assert hasattr(h, "JSO")
        assert "JSO" in h.__all__

    def test_in_structural_catalog(self):
        from panobbgo.heuristics.jso import JSO
        from panobbgo.self_improve import StructuralMutationRule, default_structural_catalog

        catalog = default_structural_catalog()
        add_rules = [r for r in catalog.rules if isinstance(r, StructuralMutationRule) and r.op == "add_heuristic"]
        assert add_rules
        has_jso = any(cls is JSO for rule in add_rules for cls, _ in (rule.candidate_classes or ()))
        assert has_jso

    def test_kwarg_catalog_has_jso_dials(self):
        from panobbgo.self_improve import MutationRule, default_catalog

        rules = default_catalog().rules
        params = {(r.class_name, r.param_name) for r in rules if isinstance(r, MutationRule) and r.class_name == "JSO"}
        assert ("JSO", "NP_init") in params
        assert ("JSO", "p_best_max") in params
        # H rule mirrors LSHADE.H; Brest et al. (2017) report H = 5 best.
        assert ("JSO", "H") in params

    def test_kwarg_catalog_jso_H_is_integer_add(self):
        """``JSO.H`` is an integer-add rule with sensible bounds."""
        from panobbgo.self_improve import MutationRule, default_catalog

        rules = [
            r
            for r in default_catalog().rules
            if isinstance(r, MutationRule) and r.class_name == "JSO" and r.param_name == "H"
        ]
        assert len(rules) == 1
        rule = rules[0]
        assert rule.kind == "integer_add"
        lo, hi = rule.bounds
        # H >= 2 is the constructor floor; bounds must stay clear of it.
        assert lo >= 2
        assert hi <= 20
