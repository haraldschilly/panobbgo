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

"""Tests for the NL-SHADE-RSP (Stanovov 2021) adaptive DE heuristic."""

from __future__ import annotations

import numpy as np
import pytest

from panobbgo.utils import PanobbgoTestCase


class _MockStrategyMixin:
    """Identical setup pattern to :mod:`tests.test_heuristic_jso`."""

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


class NLSHADERSPConstructionTests(_MockStrategyMixin, PanobbgoTestCase):
    def test_default_construction(self):
        from panobbgo.heuristics.nlshade_rsp import NLSHADERSP
        from panobbgo.heuristics.jso import (
            _ANCHOR_M_CR,
            _ANCHOR_M_F,
            _INIT_M_CR,
            _INIT_M_F,
        )

        h = NLSHADERSP(self.strategy)
        assert h.NP_init == 30
        assert h.NP_min == 4
        assert h.H == 5  # inherited from jSO
        assert h.p_best_max == 0.25
        assert h.p_best_min == 0.125
        assert h.archive_factor == 1.0
        assert h.name == "NLSHADERSP"
        # jSO memory anchor invariants are inherited.
        assert np.allclose(h._M_F[:-1], _INIT_M_F)
        assert np.allclose(h._M_CR[:-1], _INIT_M_CR)
        assert h._M_F[-1] == _ANCHOR_M_F
        assert h._M_CR[-1] == _ANCHOR_M_CR
        # Adaptive archive cap initialised to A_max.
        assert h._arc_size_t == int(round(1.0 * 30))

    def test_custom_construction(self):
        from panobbgo.heuristics.nlshade_rsp import NLSHADERSP

        h = NLSHADERSP(
            self.strategy,
            NP_init=20,
            NP_min=6,
            H=4,
            p_best_max=0.3,
            p_best_min=0.1,
            archive_factor=2.0,
            seed=7,
            name="MyNLRSP",
        )
        assert h.NP_init == 20
        assert h.NP_min == 6
        assert h.H == 4
        assert h.p_best_max == 0.3
        assert h.p_best_min == 0.1
        assert h.archive_factor == 2.0
        assert h.name == "MyNLRSP"
        # ``A_max = ceil(archive_factor · NP_init)`` at construction time.
        assert h._arc_size_t == int(round(2.0 * 20))

    def test_subclass_of_jso_and_lshade(self):
        from panobbgo.heuristics.jso import JSO
        from panobbgo.heuristics.lshade import LSHADE
        from panobbgo.heuristics.nlshade_rsp import NLSHADERSP

        h = NLSHADERSP(self.strategy)
        assert isinstance(h, JSO)
        assert isinstance(h, LSHADE)

    def test_opts_into_F_schedule_by_inheritance(self):
        """NL-SHADE-RSP keeps the jSO three-phase asymmetric F-cap."""
        from panobbgo.heuristics.nlshade_rsp import NLSHADERSP

        h = NLSHADERSP(self.strategy, seed=0)
        assert h.F_schedule is True

    def test_invalid_p_best_max(self):
        from panobbgo.heuristics.nlshade_rsp import NLSHADERSP

        with pytest.raises(ValueError, match="p_best_max"):
            NLSHADERSP(self.strategy, p_best_max=0.0)
        with pytest.raises(ValueError, match="p_best_max"):
            NLSHADERSP(self.strategy, p_best_max=1.5)
        with pytest.raises(ValueError, match="p_best_max"):
            NLSHADERSP(self.strategy, p_best_max=float("nan"))

    def test_archive_factor_zero_collapses_cap_to_zero(self):
        """``archive_factor=0`` is allowed and pins ``arc_size_t`` at 0."""
        from panobbgo.heuristics.nlshade_rsp import NLSHADERSP

        h = NLSHADERSP(self.strategy, archive_factor=0.0, seed=0)
        assert h._arc_size_t == 0
        # Even after a refresh, the cap stays at 0.
        for _ in range(20):
            h._refresh_arc_size_t()
            assert h._arc_size_t == 0


# ----------------------------------------------------------------------
# Rank-based selection pressure (RSP)
# ----------------------------------------------------------------------


class NLSHADERSPRankWeightsTests(_MockStrategyMixin, PanobbgoTestCase):
    def test_rank_weights_basic_shape(self):
        from panobbgo.heuristics.nlshade_rsp import NLSHADERSP

        h = NLSHADERSP(self.strategy, NP_init=8, seed=0)
        # Five live indices, best-first.
        sorted_live = [10, 11, 12, 13, 14]
        candidates, probs = h._rank_weights(sorted_live, target_idx=12)
        # target_idx removed; rest in best-first order.
        np.testing.assert_array_equal(candidates, [10, 11, 13, 14])
        # NP = 5; rank of each candidate is its position in sorted_live.
        # Weights: NP - rank = 5,4,2,1 → probs proportional.
        expected_weights = np.array([5.0, 4.0, 2.0, 1.0])
        expected_probs = expected_weights / expected_weights.sum()
        np.testing.assert_allclose(probs, expected_probs)

    def test_rank_weights_target_at_end(self):
        from panobbgo.heuristics.nlshade_rsp import NLSHADERSP

        h = NLSHADERSP(self.strategy, NP_init=4, seed=0)
        sorted_live = [0, 1, 2, 3]
        candidates, probs = h._rank_weights(sorted_live, target_idx=3)
        np.testing.assert_array_equal(candidates, [0, 1, 2])
        # Weights: 4, 3, 2
        expected = np.array([4.0, 3.0, 2.0])
        expected = expected / expected.sum()
        np.testing.assert_allclose(probs, expected)

    def test_rank_weights_target_at_start(self):
        from panobbgo.heuristics.nlshade_rsp import NLSHADERSP

        h = NLSHADERSP(self.strategy, NP_init=4, seed=0)
        sorted_live = [0, 1, 2, 3]
        candidates, probs = h._rank_weights(sorted_live, target_idx=0)
        np.testing.assert_array_equal(candidates, [1, 2, 3])
        # Weights: 3, 2, 1 (rank-1 vs rank-2 vs rank-3)
        expected = np.array([3.0, 2.0, 1.0])
        expected = expected / expected.sum()
        np.testing.assert_allclose(probs, expected)

    def test_rank_weights_empty_when_target_is_lone_member(self):
        from panobbgo.heuristics.nlshade_rsp import NLSHADERSP

        h = NLSHADERSP(self.strategy, seed=0)
        candidates, probs = h._rank_weights([5], target_idx=5)
        assert candidates.size == 0
        assert probs.size == 0

    def test_rank_weights_probs_sum_to_one(self):
        from panobbgo.heuristics.nlshade_rsp import NLSHADERSP

        h = NLSHADERSP(self.strategy, seed=0)
        sorted_live = list(range(20))
        for target in (0, 5, 10, 19):
            _, probs = h._rank_weights(sorted_live, target_idx=target)
            assert probs.size == 19
            assert probs.sum() == pytest.approx(1.0)

    def test_rank_weights_favour_best_over_worst(self):
        """Rank-weighted sampling must concentrate on top-ranked individuals.

        With ``NP = 10`` and the best candidate (rank 0) getting weight 10
        vs the worst (rank 9) getting weight 1, we expect ~10x more
        samples of the best than the worst over a large draw.
        """
        from panobbgo.heuristics.nlshade_rsp import NLSHADERSP

        h = NLSHADERSP(self.strategy, NP_init=10, seed=1234)
        sorted_live = list(range(10))
        # Target is not in the candidate set so all 10 ranks are sampled.
        # Use a target outside the range to keep the rank weights pristine.
        candidates, probs = h._rank_weights(sorted_live, target_idx=-1)
        np.testing.assert_array_equal(candidates, np.arange(10))
        # Best = candidate index 0; worst = candidate index 9.
        # Weight ratio = 10 / 1.
        assert probs[0] / probs[-1] == pytest.approx(10.0)
        # Empirically: 2000 draws → ~9.5x best vs worst (binomial slack).
        draws = h._rng.choice(candidates, size=2000, p=probs)
        counts = np.bincount(draws, minlength=10)
        # Allow factor-of-two slack from the 10:1 theoretical ratio.
        assert counts[0] > 5 * counts[-1]


# ----------------------------------------------------------------------
# Adaptive archive size
# ----------------------------------------------------------------------


class NLSHADERSPAdaptiveArchiveTests(_MockStrategyMixin, PanobbgoTestCase):
    def test_refresh_arc_size_t_in_zero_to_a_max(self):
        """``_refresh_arc_size_t`` always produces a value in ``[0, A_max]``."""
        from panobbgo.heuristics.nlshade_rsp import NLSHADERSP

        h = NLSHADERSP(self.strategy, NP_init=30, archive_factor=2.0, seed=1)
        # NP_current = NP_init = 30; A_max = round(2.0 * 30) = 60.
        A_max = int(round(h.archive_factor * h._NP_current))
        seen = set()
        for _ in range(500):
            h._refresh_arc_size_t()
            assert 0 <= h._arc_size_t <= A_max
            seen.add(h._arc_size_t)
        # 500 draws from {0..60} → expect to see most values.
        assert len(seen) >= 40

    def test_refresh_distribution_is_approximately_uniform(self):
        from panobbgo.heuristics.nlshade_rsp import NLSHADERSP

        h = NLSHADERSP(self.strategy, NP_init=10, archive_factor=1.0, seed=42)
        A_max = int(round(h.archive_factor * h._NP_current))
        counts = np.zeros(A_max + 1, dtype=int)
        N = 5000
        for _ in range(N):
            h._refresh_arc_size_t()
            counts[h._arc_size_t] += 1
        expected = N / (A_max + 1)
        # No bucket should be wildly off from uniform.  Allow 4x slack
        # so noise on N=5000 doesn't make this flaky.
        assert np.all(counts > expected / 4)
        assert np.all(counts < expected * 4)

    def test_trim_archive_uses_arc_size_t(self):
        from panobbgo.heuristics.nlshade_rsp import NLSHADERSP

        h = NLSHADERSP(self.strategy, NP_init=8, archive_factor=2.0, seed=2)
        # Stuff the archive with sentinel vectors.
        for i in range(20):
            h._archive.append(np.array([float(i), float(i)]))
        # Now trim to specific arc_size_t values.
        h._arc_size_t = 5
        h._trim_archive()
        assert len(h._archive) == 5
        h._arc_size_t = 2
        h._trim_archive()
        assert len(h._archive) == 2
        h._arc_size_t = 0
        h._trim_archive()
        assert h._archive == []

    def test_end_of_generation_refreshes_arc_size_t(self):
        """``_end_of_generation`` rolls a new ``arc_size_t`` *before* delegating."""
        from panobbgo.heuristics.nlshade_rsp import NLSHADERSP

        h = NLSHADERSP(self.strategy, NP_init=10, archive_factor=1.0, seed=3)
        seen = set()
        for _ in range(50):
            h._end_of_generation()  # no successes → memory/LPSR are no-ops
            seen.add(h._arc_size_t)
            # Mutation happened: most calls will land on a different cap.
            # Allow rare same-value rolls (e.g., when A_max collapses).
        # 50 draws should cover at least a few distinct buckets.
        assert len(seen) >= 3

    def test_archive_empty_when_arc_size_t_zero(self):
        from panobbgo.heuristics.nlshade_rsp import NLSHADERSP

        h = NLSHADERSP(self.strategy, NP_init=6, archive_factor=1.0, seed=4)
        h._archive.append(np.array([1.0, 2.0]))
        h._archive.append(np.array([3.0, 4.0]))
        h._arc_size_t = 0
        h._trim_archive()
        assert h._archive == []


# ----------------------------------------------------------------------
# Schedules inherited from jSO
# ----------------------------------------------------------------------


class NLSHADERSPInheritedScheduleTests(_MockStrategyMixin, PanobbgoTestCase):
    def test_linear_p_best_schedule_inherited(self):
        from panobbgo.heuristics.nlshade_rsp import NLSHADERSP

        h = NLSHADERSP(self.strategy, p_best_max=0.25, p_best_min=0.125)
        self.strategy.config.max_eval = 100
        self.strategy.results = []
        assert h._current_p_best() == pytest.approx(0.25)
        self.strategy.results = list(range(50))
        assert h._current_p_best() == pytest.approx((0.25 + 0.125) / 2.0)
        self.strategy.results = list(range(100))
        assert h._current_p_best() == pytest.approx(0.125)

    def test_three_phase_F_w_schedule_inherited(self):
        from panobbgo.heuristics.nlshade_rsp import NLSHADERSP

        h = NLSHADERSP(self.strategy)
        self.strategy.config.max_eval = 100
        self.strategy.results = list(range(10))
        assert h._current_F_weight() == pytest.approx(0.7)
        self.strategy.results = list(range(30))
        assert h._current_F_weight() == pytest.approx(0.8)
        self.strategy.results = list(range(50))
        assert h._current_F_weight() == pytest.approx(1.2)


# ----------------------------------------------------------------------
# Trial generation: rank-based r1 makes it into the emitted trial
# ----------------------------------------------------------------------


def _build_result(strategy, x, fx, who):
    from panobbgo.lib import Point, Result

    return Result(Point(np.asarray(x, dtype=float), who), float(fx))


class NLSHADERSPGenerateTrialTests(_MockStrategyMixin, PanobbgoTestCase):
    def _seed_population(self, h, fx_seq):
        items = list(h._pending.items())
        results = []
        for (req_id, _meta), fx in zip(items, fx_seq):
            x = self.problem.random_point()
            results.append(_build_result(self.strategy, x, fx, f"NLSHADERSP:{req_id}"))
        h.on_new_results(results)

    def test_initial_population_emission(self):
        from panobbgo.heuristics.nlshade_rsp import NLSHADERSP

        h = NLSHADERSP(self.strategy, NP_init=8, seed=0)
        h.on_start()
        emitted = h.get_points(limit=100)
        assert len(emitted) == 8
        assert all(pt.who.startswith("NLSHADERSP:") for pt in emitted)
        # Initial-fill trials carry NaN F/CR (no success contribution).
        for meta in h._pending.values():
            assert np.isnan(meta.F)
            assert np.isnan(meta.CR)

    def test_filled_population_emits_evolutionary_trials(self):
        from panobbgo.heuristics.nlshade_rsp import NLSHADERSP

        h = NLSHADERSP(self.strategy, NP_init=5, seed=12)
        h.on_start()
        h.get_points(limit=100)

        items = list(h._pending.items())[:4]
        results = []
        for req_id, meta in items:
            x = self.problem.random_point()
            results.append(_build_result(self.strategy, x, 10.0 + meta.slot_idx, f"NLSHADERSP:{req_id}"))
        h.on_new_results(results)
        h.get_points(limit=100)
        # At least one evolutionary trial pending.
        evo_metas = [m for _, m in h._pending.items() if not np.isnan(m.F) and not np.isnan(m.CR)]
        assert len(evo_metas) >= 1

    def test_better_trial_wins_and_archives_parent(self):
        from panobbgo.heuristics.nlshade_rsp import NLSHADERSP

        h = NLSHADERSP(self.strategy, NP_init=4, seed=13)
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
        r = _build_result(self.strategy, x, improved_fx, f"NLSHADERSP:{rid}")
        h.on_new_results([r])

        assert h._population[target_slot].fx == improved_fx
        assert len(h._success_F) >= 1


# ----------------------------------------------------------------------
# Restart behaviour
# ----------------------------------------------------------------------


class NLSHADERSPRestartTests(_MockStrategyMixin, PanobbgoTestCase):
    def test_restart_re_stamps_jso_initial_memory_and_resets_cap(self):
        from panobbgo.heuristics.jso import _ANCHOR_M_CR, _ANCHOR_M_F, _INIT_M_CR, _INIT_M_F
        from panobbgo.heuristics.nlshade_rsp import NLSHADERSP

        h = NLSHADERSP(self.strategy, NP_init=6, archive_factor=1.0, seed=2)
        h.on_start()
        h.get_points(limit=100)
        # Mutate state.
        h._M_F[:] = 0.6
        h._M_CR[:] = 0.4
        h._archive.append(np.array([0.5, 0.5]))
        h._success_F = [0.7]
        h._arc_size_t = 0  # off

        center = np.array([0.0, 0.0])
        h.on_restart(center, reason="test")

        # Archive cleared, jSO memory restored, anchor bin intact.
        assert h._archive == []
        assert h._success_F == []
        assert np.allclose(h._M_F[:-1], _INIT_M_F)
        assert np.allclose(h._M_CR[:-1], _INIT_M_CR)
        assert h._M_F[-1] == _ANCHOR_M_F
        assert h._M_CR[-1] == _ANCHOR_M_CR
        # Archive cap restored to A_max.
        assert h._arc_size_t == int(round(h.archive_factor * h._NP_current))

    def test_restart_with_none_center_falls_back_to_random(self):
        from panobbgo.heuristics.nlshade_rsp import NLSHADERSP

        h = NLSHADERSP(self.strategy, NP_init=4, seed=3)
        h.on_start()
        h.get_points(limit=100)
        h.on_restart(None)
        emitted = h.get_points(limit=100)
        assert len(emitted) == 4
        for pt in emitted:
            assert np.all(pt.x >= self.problem.box[:, 0] - 1e-9)
            assert np.all(pt.x <= self.problem.box[:, 1] + 1e-9)

    def test_restart_before_start_is_noop(self):
        from panobbgo.heuristics.nlshade_rsp import NLSHADERSP

        h = NLSHADERSP(self.strategy, NP_init=4, seed=4)
        h.on_restart(None)
        emitted = h.get_points(limit=100)
        assert emitted == []


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


class NLSHADERSPSmokeTests(_MockStrategyMixin, PanobbgoTestCase):
    def test_smoke_quadratic_no_regression(self):
        """Driving NL-SHADE-RSP through a few generations on f(x) = ||x||² makes no negative progress."""
        from panobbgo.heuristics.nlshade_rsp import NLSHADERSP
        from panobbgo.lib import Point, Result

        h = NLSHADERSP(self.strategy, NP_init=8, NP_min=4, seed=5)
        h.on_start()

        def fx_of(x):
            return float(np.dot(x, x))

        items = list(h._pending.items())
        h.get_points(limit=100)
        results = []
        for rid, _meta in items:
            x = self.problem.random_point()
            results.append(Result(Point(x, f"NLSHADERSP:{rid}"), fx_of(x)))
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
                results.append(Result(Point(x, f"NLSHADERSP:{rid}"), fx_of(x)))
            h.on_new_results(results)

        best_fx_after = min(s.fx for s in h._population if isinstance(s, Result))
        assert best_fx_after <= best_fx_before + 1e-6


# ----------------------------------------------------------------------
# Registration
# ----------------------------------------------------------------------


class NLSHADERSPRegistrationTests(_MockStrategyMixin, PanobbgoTestCase):
    def test_registered_in_heuristics_package(self):
        import panobbgo.heuristics as h

        assert hasattr(h, "NLSHADERSP")
        assert "NLSHADERSP" in h.__all__

    def test_in_structural_catalog(self):
        from panobbgo.heuristics.nlshade_rsp import NLSHADERSP
        from panobbgo.self_improve import StructuralMutationRule, default_structural_catalog

        catalog = default_structural_catalog()
        add_rules = [r for r in catalog.rules if isinstance(r, StructuralMutationRule) and r.op == "add_heuristic"]
        assert add_rules
        has_nlrsp = any(cls is NLSHADERSP for rule in add_rules for cls, _ in (rule.candidate_classes or ()))
        assert has_nlrsp

    def test_kwarg_catalog_has_nlshade_rsp_dials(self):
        from panobbgo.self_improve import MutationRule, default_catalog

        rules = default_catalog().rules
        params = {
            (r.class_name, r.param_name) for r in rules if isinstance(r, MutationRule) and r.class_name == "NLSHADERSP"
        }
        assert ("NLSHADERSP", "NP_init") in params
        assert ("NLSHADERSP", "archive_factor") in params
