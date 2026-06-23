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

"""Tests for the NL-SHADE-RSP (Stanovov et al. 2021) adaptive DE heuristic."""

from __future__ import annotations

import numpy as np
import pytest

from panobbgo.utils import PanobbgoTestCase


class _MockStrategyMixin:
    """Same scaffolding as the L-SHADE / jSO tests.

    NL-SHADE-RSP inherits the LPSR / constraint-handler / max_eval
    semantics from jSO → L-SHADE, so the mock strategy needs the same
    setup.  ``config.max_eval`` is saved / restored to prevent cross-test
    bleed.
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


def _build_result(strategy, x, fx, who):
    from panobbgo.lib import Point, Result

    return Result(Point(np.asarray(x, dtype=float), who), float(fx))


# ----------------------------------------------------------------------
# Construction-time validation
# ----------------------------------------------------------------------


class NLSHADERSPConstructionTests(_MockStrategyMixin, PanobbgoTestCase):
    def test_default_construction(self):
        from panobbgo.heuristics.nl_shade_rsp import NLSHADE_RSP, _DEFAULT_K_RANK

        h = NLSHADE_RSP(self.strategy)
        assert h.NP_init == 30
        assert h.NP_min == 4
        assert h.H == 5
        assert h.p_best_max == 0.25
        assert h.p_best_min == 0.125
        assert h.archive_factor == 1.0
        assert h.k_rank == _DEFAULT_K_RANK == 3.0
        assert h.adaptive_archive is True
        assert h.name == "NLSHADE_RSP"
        assert h._rsp_archive_cap is None
        # jSO opts into the asymmetric F-cap; NL-SHADE-RSP inherits it.
        assert h.F_schedule == "jso"

    def test_custom_construction(self):
        from panobbgo.heuristics.nl_shade_rsp import NLSHADE_RSP

        h = NLSHADE_RSP(
            self.strategy,
            NP_init=20,
            NP_min=6,
            H=4,
            p_best_max=0.3,
            p_best_min=0.1,
            archive_factor=2.0,
            k_rank=1.5,
            adaptive_archive=False,
            seed=7,
            name="MyRSP",
        )
        assert h.NP_init == 20
        assert h.NP_min == 6
        assert h.H == 4
        assert h.p_best_max == 0.3
        assert h.p_best_min == 0.1
        assert h.archive_factor == 2.0
        assert h.k_rank == 1.5
        assert h.adaptive_archive is False
        assert h.name == "MyRSP"

    def test_subclass_of_jso_and_lshade(self):
        from panobbgo.heuristics.jso import JSO
        from panobbgo.heuristics.lshade import LSHADE
        from panobbgo.heuristics.nl_shade_rsp import NLSHADE_RSP

        h = NLSHADE_RSP(self.strategy)
        assert isinstance(h, JSO)
        assert isinstance(h, LSHADE)

    def test_invalid_k_rank(self):
        from panobbgo.heuristics.nl_shade_rsp import NLSHADE_RSP

        with pytest.raises(ValueError, match="k_rank"):
            NLSHADE_RSP(self.strategy, k_rank=-0.1)
        with pytest.raises(ValueError, match="k_rank"):
            NLSHADE_RSP(self.strategy, k_rank=float("nan"))
        with pytest.raises(ValueError, match="k_rank"):
            NLSHADE_RSP(self.strategy, k_rank=float("inf"))

    def test_k_rank_zero_is_valid(self):
        """``k_rank=0`` is the uniform-selection degenerate case — allowed."""
        from panobbgo.heuristics.nl_shade_rsp import NLSHADE_RSP

        h = NLSHADE_RSP(self.strategy, k_rank=0.0)
        assert h.k_rank == 0.0

    def test_invalid_adaptive_archive_type(self):
        from panobbgo.heuristics.nl_shade_rsp import NLSHADE_RSP

        with pytest.raises(ValueError, match="adaptive_archive"):
            NLSHADE_RSP(self.strategy, adaptive_archive="yes")  # type: ignore[arg-type]

    def test_inherits_jso_validation(self):
        """jSO's H >= 2 and p_best ordering rules still apply."""
        from panobbgo.heuristics.nl_shade_rsp import NLSHADE_RSP

        with pytest.raises(ValueError, match="H must be >= 2"):
            NLSHADE_RSP(self.strategy, H=1)
        with pytest.raises(ValueError, match="p_best_min .* must be <= p_best_max"):
            NLSHADE_RSP(self.strategy, p_best_max=0.2, p_best_min=0.3)


# ----------------------------------------------------------------------
# Non-Linear Population Size Reduction (NLPSR)
# ----------------------------------------------------------------------


class NLSHADERSPReductionTests(_MockStrategyMixin, PanobbgoTestCase):
    def test_target_endpoints(self):
        """NLPSR maps progress 0 → NP_init and progress 1 → NP_min."""
        from panobbgo.heuristics.nl_shade_rsp import NLSHADE_RSP

        h = NLSHADE_RSP(self.strategy, NP_init=30, NP_min=4)
        assert h._lpsr_target(0.0) == 30
        assert h._lpsr_target(1.0) == 4

    def test_target_monotone_non_increasing(self):
        from panobbgo.heuristics.nl_shade_rsp import NLSHADE_RSP

        h = NLSHADE_RSP(self.strategy, NP_init=40, NP_min=4)
        grid = np.linspace(0.0, 1.0, 21)
        targets = [h._lpsr_target(p) for p in grid]
        for a, b in zip(targets, targets[1:]):
            assert b <= a

    def test_reduces_faster_than_linear_midrun(self):
        """At progress 0.5 the non-linear schedule drops more than linear."""
        from panobbgo.heuristics.lshade import LSHADE
        from panobbgo.heuristics.nl_shade_rsp import NLSHADE_RSP

        nl = NLSHADE_RSP(self.strategy, NP_init=30, NP_min=4)
        lin = LSHADE(self.strategy, NP_init=30, NP_min=4)
        # r^(1-r) at r=0.5 is ~0.707 > 0.5, so NL target is smaller.
        assert nl._lpsr_target(0.5) < lin._lpsr_target(0.5)
        # Concretely: linear -> 17, non-linear -> 12.
        assert lin._lpsr_target(0.5) == 17
        assert nl._lpsr_target(0.5) == 12

    def test_apply_lpsr_shrinks_population(self):
        from panobbgo.heuristics.nl_shade_rsp import NLSHADE_RSP

        h = NLSHADE_RSP(self.strategy, NP_init=12, NP_min=4, seed=1)
        # Hand-build a full live population with known fitness.
        h._population = [
            _build_result(self.strategy, self.problem.random_point(), float(i), f"x{i}") for i in range(12)
        ]
        h._NP_current = 12
        self.strategy.config.max_eval = 100
        self.strategy.results = list(range(50))  # progress = 0.5
        h._apply_lpsr()
        expected = max(h._lpsr_target(0.5), h.NP_min)
        assert h._NP_current == expected
        live = h._live_indices()
        assert len(live) == expected
        # The dropped slots are the worst by fitness (highest fx).
        survivors_fx = sorted(h._population[i].fx for i in live)
        assert survivors_fx == [float(i) for i in range(expected)]

    def test_apply_lpsr_noop_without_budget(self):
        from panobbgo.heuristics.nl_shade_rsp import NLSHADE_RSP

        h = NLSHADE_RSP(self.strategy, NP_init=10, NP_min=4)
        h._population = [
            _build_result(self.strategy, self.problem.random_point(), float(i), f"x{i}") for i in range(10)
        ]
        h._NP_current = 10
        self.strategy.config.max_eval = 0  # unknown budget
        h._apply_lpsr()
        assert h._NP_current == 10


# ----------------------------------------------------------------------
# Rank-based Selective Pressure (RSP)
# ----------------------------------------------------------------------


class NLSHADERSPRankSelectionTests(_MockStrategyMixin, PanobbgoTestCase):
    def _populate(self, h, fxs):
        h._population = [
            _build_result(self.strategy, self.problem.random_point(), fx, f"x{i}") for i, fx in enumerate(fxs)
        ]
        h._NP_current = len(fxs)
        return h._live_indices()

    def test_select_excludes_target(self):
        from panobbgo.heuristics.nl_shade_rsp import NLSHADE_RSP

        h = NLSHADE_RSP(self.strategy, seed=0)
        live = self._populate(h, [1.0, 2.0, 3.0, 4.0, 5.0])
        for _ in range(200):
            r1 = h._select_r1(live, target_idx=2)
            assert r1 != 2
            assert r1 in live

    def test_returns_none_when_pool_empty(self):
        from panobbgo.heuristics.nl_shade_rsp import NLSHADE_RSP

        h = NLSHADE_RSP(self.strategy, seed=0)
        live = self._populate(h, [1.0])  # only the target slot is live
        assert h._select_r1(live, target_idx=0) is None

    def test_better_individuals_selected_more_often(self):
        """With k_rank > 0 the best slot is drawn far more than the worst."""
        from panobbgo.heuristics.nl_shade_rsp import NLSHADE_RSP

        h = NLSHADE_RSP(self.strategy, k_rank=3.0, seed=123)
        # Slot 0 is best (fx=1), slot 4 is worst (fx=5).  Target is slot 2.
        live = self._populate(h, [1.0, 2.0, 3.0, 4.0, 5.0])
        counts = {i: 0 for i in live}
        for _ in range(3000):
            counts[h._select_r1(live, target_idx=2)] += 1
        # Best candidate (slot 0) is chosen substantially more than worst (slot 4).
        assert counts[0] > counts[4]
        assert counts[0] > 2 * counts[4]

    def test_k_rank_zero_is_uniform(self):
        """``k_rank=0`` gives equal weights → roughly uniform selection."""
        from panobbgo.heuristics.nl_shade_rsp import NLSHADE_RSP

        h = NLSHADE_RSP(self.strategy, k_rank=0.0, seed=7)
        live = self._populate(h, [1.0, 2.0, 3.0, 4.0])  # no target excluded below
        counts = {i: 0 for i in live}
        for _ in range(8000):
            counts[h._select_r1(live, target_idx=99)] += 1  # target not in pool
        # Each of the 4 slots should get ~25% (2000); allow generous slack.
        for i in live:
            assert 1500 < counts[i] < 2500


# ----------------------------------------------------------------------
# Adaptive (randomised) archive
# ----------------------------------------------------------------------


class NLSHADERSPArchiveTests(_MockStrategyMixin, PanobbgoTestCase):
    def test_fixed_cap_when_adaptive_off(self):
        from panobbgo.heuristics.nl_shade_rsp import NLSHADE_RSP

        h = NLSHADE_RSP(self.strategy, archive_factor=2.0, adaptive_archive=False)
        h._NP_current = 10
        assert h._archive_cap() == 20  # round(2.0 * 10)

    def test_adaptive_cap_within_bounds(self):
        from panobbgo.heuristics.nl_shade_rsp import NLSHADE_RSP

        h = NLSHADE_RSP(self.strategy, archive_factor=1.0, adaptive_archive=True, seed=3)
        h._NP_current = 30
        a_max = h._archive_max()
        assert a_max == 30
        seen = set()
        for _ in range(200):
            h._rsp_archive_cap = None  # force a fresh sample
            cap = h._archive_cap()
            assert 0 <= cap <= a_max
            seen.add(cap)
        # Random sampling should produce a spread of cap values.
        assert len(seen) > 5

    def test_adaptive_cap_clipped_to_shrunk_a_max(self):
        """A cached cap is clipped down when NLPSR shrinks ``A_max``."""
        from panobbgo.heuristics.nl_shade_rsp import NLSHADE_RSP

        h = NLSHADE_RSP(self.strategy, archive_factor=1.0, adaptive_archive=True)
        h._NP_current = 30
        h._rsp_archive_cap = 25  # sampled while population was large
        h._NP_current = 8  # NLPSR shrank the population; A_max now 8
        assert h._archive_cap() == 8

    def test_cap_lazily_sampled_once(self):
        from panobbgo.heuristics.nl_shade_rsp import NLSHADE_RSP

        h = NLSHADE_RSP(self.strategy, adaptive_archive=True, seed=5)
        h._NP_current = 20
        assert h._rsp_archive_cap is None
        cap1 = h._archive_cap()
        assert h._rsp_archive_cap is not None
        cap2 = h._archive_cap()  # cached — no re-sample
        assert cap1 == cap2

    def test_end_of_generation_resamples_cap(self):
        from panobbgo.heuristics.nl_shade_rsp import NLSHADE_RSP

        h = NLSHADE_RSP(self.strategy, adaptive_archive=True, seed=9)
        h._population = []
        h._NP_current = 20
        self.strategy.config.max_eval = 1000
        self.strategy.results = []
        h._rsp_archive_cap = None
        h._end_of_generation()
        assert h._rsp_archive_cap is not None
        assert 0 <= h._rsp_archive_cap <= h._archive_max()

    def test_archive_never_exceeds_cap(self):
        from panobbgo.heuristics.nl_shade_rsp import NLSHADE_RSP

        h = NLSHADE_RSP(self.strategy, archive_factor=1.0, adaptive_archive=True, seed=11)
        h._NP_current = 6
        h._rsp_archive_cap = 3
        h._archive = [np.array([0.0, 0.0]) for _ in range(10)]
        h._trim_archive()
        assert len(h._archive) <= 3


# ----------------------------------------------------------------------
# Initial population / generate-trial / restart (inherited pipeline)
# ----------------------------------------------------------------------


class NLSHADERSPPipelineTests(_MockStrategyMixin, PanobbgoTestCase):
    def test_on_start_emits_NP_init_points(self):
        from panobbgo.heuristics.nl_shade_rsp import NLSHADE_RSP

        h = NLSHADE_RSP(self.strategy, NP_init=8, seed=0)
        h.on_start()
        emitted = h.get_points(limit=100)
        assert len(emitted) == 8
        assert len(h._pending) == 8
        assert len(h._population) == 8
        assert all(slot is None for slot in h._population)
        assert all(pt.who.startswith("NLSHADE_RSP:") for pt in emitted)

    def test_on_start_resets_archive_cap(self):
        from panobbgo.heuristics.nl_shade_rsp import NLSHADE_RSP

        h = NLSHADE_RSP(self.strategy, NP_init=4, seed=0)
        h._rsp_archive_cap = 7
        h.on_start()
        assert h._rsp_archive_cap is None

    def test_filled_population_emits_evolutionary_trials(self):
        from panobbgo.heuristics.nl_shade_rsp import NLSHADE_RSP

        h = NLSHADE_RSP(self.strategy, NP_init=5, seed=12)
        h.on_start()
        h.get_points(limit=100)
        items = list(h._pending.items())[:4]
        results = []
        for req_id, meta in items:
            x = self.problem.random_point()
            results.append(_build_result(self.strategy, x, 10.0 + meta.slot_idx, f"NLSHADE_RSP:{req_id}"))
        h.on_new_results(results)
        emitted = h.get_points(limit=100)
        assert len(emitted) >= 1
        evo = [m for _, m in h._pending.items() if not np.isnan(m.F) and not np.isnan(m.CR)]
        assert len(evo) >= 1

    def test_better_trial_wins_and_archives_parent(self):
        from panobbgo.heuristics.nl_shade_rsp import NLSHADE_RSP

        h = NLSHADE_RSP(self.strategy, NP_init=4, seed=13)
        h.on_start()
        h.get_points(limit=100)
        # Fill the four slots.
        items = list(h._pending.items())
        results = []
        for (rid, _m), fx in zip(items, [100.0, 110.0, 120.0, 130.0]):
            results.append(_build_result(self.strategy, self.problem.random_point(), fx, f"NLSHADE_RSP:{rid}"))
        h.on_new_results(results)
        h.get_points(limit=100)

        target_slot = 0
        target_fx = h._population[target_slot].fx
        rid_meta = [(r, m) for r, m in h._pending.items() if m.slot_idx == target_slot]
        assert rid_meta
        rid, _ = rid_meta[0]
        improved = _build_result(self.strategy, self.problem.random_point(), target_fx - 50.0, f"NLSHADE_RSP:{rid}")
        h.on_new_results([improved])
        assert h._population[target_slot].fx == target_fx - 50.0
        assert len(h._success_F) >= 1

    def test_restart_resets_archive_cap_and_memory(self):
        from panobbgo.heuristics.nl_shade_rsp import NLSHADE_RSP

        h = NLSHADE_RSP(self.strategy, NP_init=6, seed=2)
        h.on_start()
        h.get_points(limit=100)
        h._rsp_archive_cap = 4
        h._archive.append(np.array([0.5, 0.5]))
        h.on_restart(np.array([0.0, 0.0]), reason="test")
        assert h._rsp_archive_cap is None
        assert h._archive == []
        assert len(h._pending) == h.NP_init

    def test_smoke_quadratic_no_regression(self):
        """A few generations on f(x)=||x||² makes no negative global progress."""
        from panobbgo.heuristics.lshade import _Dropped
        from panobbgo.heuristics.nl_shade_rsp import NLSHADE_RSP
        from panobbgo.lib import Point, Result

        h = NLSHADE_RSP(self.strategy, NP_init=8, NP_min=4, seed=5)
        h.on_start()

        def fx_of(x):
            return float(np.dot(x, x))

        items = list(h._pending.items())
        h.get_points(limit=100)
        results = [Result(Point(x := self.problem.random_point(), f"NLSHADE_RSP:{rid}"), fx_of(x)) for rid, _m in items]
        h.on_new_results(results)
        h.get_points(limit=100)
        best_before = min(s.fx for s in h._population if isinstance(s, Result))

        for _round in range(20):
            pending = list(h._pending.items())
            if not pending:
                break
            h.get_points(limit=200)
            results = []
            for rid, meta in pending:
                slot = h._population[meta.slot_idx]
                if isinstance(slot, _Dropped) or slot is None:
                    continue
                x = self.problem.project(np.asarray(slot.x) + 0.1 * np.random.randn(self.problem.dim))
                results.append(Result(Point(x, f"NLSHADE_RSP:{rid}"), fx_of(x)))
            h.on_new_results(results)

        best_after = min(s.fx for s in h._population if isinstance(s, Result))
        assert best_after <= best_before + 1e-6


# ----------------------------------------------------------------------
# Byte-identical safety: refactor must not change L-SHADE / jSO behaviour
# ----------------------------------------------------------------------


class DEFamilyHookTests(_MockStrategyMixin, PanobbgoTestCase):
    def test_lshade_select_r1_uniform_excludes_target(self):
        from panobbgo.heuristics.lshade import LSHADE

        h = LSHADE(self.strategy, seed=0)
        h._population = [_build_result(self.strategy, self.problem.random_point(), float(i), f"x{i}") for i in range(5)]
        h._NP_current = 5
        live = h._live_indices()
        for _ in range(100):
            assert h._select_r1(live, target_idx=1) != 1

    def test_lshade_lpsr_target_is_linear(self):
        from panobbgo.heuristics.lshade import LSHADE

        h = LSHADE(self.strategy, NP_init=30, NP_min=4)
        assert h._lpsr_target(0.0) == 30
        assert h._lpsr_target(0.5) == 17
        assert h._lpsr_target(1.0) == 4

    def test_lshade_archive_cap_is_fixed(self):
        from panobbgo.heuristics.lshade import LSHADE

        h = LSHADE(self.strategy, archive_factor=1.0)
        h._NP_current = 12
        assert h._archive_cap() == 12


# ----------------------------------------------------------------------
# Registration
# ----------------------------------------------------------------------


class NLSHADERSPRegistrationTests(_MockStrategyMixin, PanobbgoTestCase):
    def test_registered_in_heuristics_package(self):
        import panobbgo.heuristics as h

        assert hasattr(h, "NLSHADE_RSP")
        assert "NLSHADE_RSP" in h.__all__

    def test_in_structural_catalog(self):
        from panobbgo.heuristics.nl_shade_rsp import NLSHADE_RSP
        from panobbgo.self_improve import StructuralMutationRule, default_structural_catalog

        catalog = default_structural_catalog()
        add_rules = [r for r in catalog.rules if isinstance(r, StructuralMutationRule) and r.op == "add_heuristic"]
        assert add_rules
        has_rsp = any(cls is NLSHADE_RSP for rule in add_rules for cls, _ in (rule.candidate_classes or ()))
        assert has_rsp

    def test_kwarg_catalog_has_rsp_dials(self):
        from panobbgo.self_improve import MutationRule, default_catalog

        rules = default_catalog().rules
        params = {
            (r.class_name, r.param_name) for r in rules if isinstance(r, MutationRule) and r.class_name == "NLSHADE_RSP"
        }
        assert ("NLSHADE_RSP", "NP_init") in params
        assert ("NLSHADE_RSP", "k_rank") in params
        assert ("NLSHADE_RSP", "adaptive_archive") in params
        # H rule mirrors LSHADE.H / JSO.H; inherits the H >= 2 anchor-bin
        # constraint from JSO.
        assert ("NLSHADE_RSP", "H") in params

    def test_kwarg_catalog_nlshade_rsp_k_rank_has_both_kinds(self):
        """``NLSHADE_RSP.k_rank`` ships both a continuous ``float_uniform``
        rule (for fine-tuning around the literature default) and a
        ``categorical_choice`` rule (for jumping between qualitatively
        different RSP regimes — off / default / aggressive).  The two
        live on distinct bandit arms by construction."""
        from panobbgo.self_improve import MutationRule, default_catalog

        kinds = {
            r.kind
            for r in default_catalog().rules
            if isinstance(r, MutationRule) and r.class_name == "NLSHADE_RSP" and r.param_name == "k_rank"
        }
        assert "float_uniform" in kinds
        assert "categorical_choice" in kinds

    def test_kwarg_catalog_nlshade_rsp_k_rank_categorical_choices(self):
        """The categorical ``k_rank`` rule must include ``0.0`` (= jSO /
        RSP-off recovery) and ``3.0`` (Stanovov et al. default)."""
        from panobbgo.self_improve import MutationRule, default_catalog

        cat_rules = [
            r
            for r in default_catalog().rules
            if isinstance(r, MutationRule)
            and r.class_name == "NLSHADE_RSP"
            and r.param_name == "k_rank"
            and r.kind == "categorical_choice"
        ]
        assert len(cat_rules) == 1
        choices = cat_rules[0].choices
        assert 0.0 in choices
        assert 3.0 in choices
        # All choices must be non-negative floats (k_rank constructor).
        assert all(isinstance(c, float) and c >= 0.0 for c in choices)

    def test_kwarg_catalog_nlshade_rsp_H_is_integer_add(self):
        """``NLSHADE_RSP.H`` is an integer-add rule with sensible bounds."""
        from panobbgo.self_improve import MutationRule, default_catalog

        rules = [
            r
            for r in default_catalog().rules
            if isinstance(r, MutationRule) and r.class_name == "NLSHADE_RSP" and r.param_name == "H"
        ]
        assert len(rules) == 1
        rule = rules[0]
        assert rule.kind == "integer_add"
        lo, hi = rule.bounds
        # H >= 2 is the jSO-inherited constructor floor.
        assert lo >= 2
        assert hi <= 20
