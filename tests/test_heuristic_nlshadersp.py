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

"""Tests for the NL-SHADE-RSP (Stanovov 2021) adaptive Differential Evolution heuristic."""

from __future__ import annotations

import numpy as np
import pytest

from panobbgo.utils import PanobbgoTestCase


class _MockStrategyMixin:
    """Same scaffolding as :mod:`tests.test_heuristic_jso`.

    NL-SHADE-RSP inherits the jSO / L-SHADE LPSR / constraint-handler /
    max_eval semantics, so the mock strategy needs the same setup.  The
    singleton ``config.max_eval`` is saved and restored per test.
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
        from panobbgo.heuristics.nlshadersp import NLSHADERSP, _DEFAULT_RANK_GREEDINESS

        h = NLSHADERSP(self.strategy)
        # Inherits jSO defaults.
        assert h.NP_init == 30
        assert h.NP_min == 4
        assert h.H == 5
        assert h.p_best_max == 0.25
        assert h.p_best_min == 0.125
        assert h.archive_factor == 1.0
        assert h.rank_greediness == _DEFAULT_RANK_GREEDINESS == 3.0
        assert h.name == "NLSHADE_RSP"
        # jSO opts into the F-cap by construction — inherited.
        assert h.F_schedule is True

    def test_custom_construction(self):
        from panobbgo.heuristics.nlshadersp import NLSHADERSP

        h = NLSHADERSP(
            self.strategy,
            NP_init=20,
            NP_min=6,
            H=4,
            p_best_max=0.3,
            p_best_min=0.1,
            archive_factor=2.0,
            rank_greediness=1.5,
            seed=7,
            name="MyNL",
        )
        assert h.NP_init == 20
        assert h.NP_min == 6
        assert h.H == 4
        assert h.p_best_max == 0.3
        assert h.p_best_min == 0.1
        assert h.archive_factor == 2.0
        assert h.rank_greediness == 1.5
        assert h.name == "MyNL"

    def test_subclass_of_jso_and_lshade(self):
        from panobbgo.heuristics.jso import JSO
        from panobbgo.heuristics.lshade import LSHADE
        from panobbgo.heuristics.nlshadersp import NLSHADERSP

        h = NLSHADERSP(self.strategy)
        assert isinstance(h, JSO)
        assert isinstance(h, LSHADE)

    def test_invalid_rank_greediness(self):
        from panobbgo.heuristics.nlshadersp import NLSHADERSP

        with pytest.raises(ValueError, match="rank_greediness"):
            NLSHADERSP(self.strategy, rank_greediness=-0.1)
        with pytest.raises(ValueError, match="rank_greediness"):
            NLSHADERSP(self.strategy, rank_greediness=float("nan"))
        with pytest.raises(ValueError, match="rank_greediness"):
            NLSHADERSP(self.strategy, rank_greediness=float("inf"))

    def test_rank_greediness_zero_allowed(self):
        """k = 0 is valid and recovers uniform donor selection."""
        from panobbgo.heuristics.nlshadersp import NLSHADERSP

        h = NLSHADERSP(self.strategy, rank_greediness=0.0)
        assert h.rank_greediness == 0.0

    def test_inherited_jso_validation(self):
        """Inherited jSO validation still fires (H >= 2, p_best ordering)."""
        from panobbgo.heuristics.nlshadersp import NLSHADERSP

        with pytest.raises(ValueError, match="H must be >= 2"):
            NLSHADERSP(self.strategy, H=1)
        with pytest.raises(ValueError, match="p_best_min .* must be <= p_best_max"):
            NLSHADERSP(self.strategy, p_best_max=0.2, p_best_min=0.3)


# ----------------------------------------------------------------------
# Rank-based selective pressure (RSP)
# ----------------------------------------------------------------------


class NLSHADERSPRankWeightTests(_MockStrategyMixin, PanobbgoTestCase):
    def test_rank_weights_best_first_descending(self):
        """sorted_live[0] (best) gets the largest weight; last gets 1."""
        from panobbgo.heuristics.nlshadersp import NLSHADERSP

        h = NLSHADERSP(self.strategy, rank_greediness=3.0)
        sorted_live = [10, 20, 30, 40]  # arbitrary indices, "best" first
        w = h._rank_weights(sorted_live)
        # Rank_i = 3*(NP - i) + 1, NP = 4, i is 1-based position.
        assert w[10] == pytest.approx(3 * 3 + 1)  # best
        assert w[20] == pytest.approx(3 * 2 + 1)
        assert w[30] == pytest.approx(3 * 1 + 1)
        assert w[40] == pytest.approx(3 * 0 + 1)  # worst → 1
        # strictly decreasing with rank
        vals = [w[i] for i in sorted_live]
        assert all(a > b for a, b in zip(vals, vals[1:]))

    def test_rank_weights_uniform_when_greediness_zero(self):
        from panobbgo.heuristics.nlshadersp import NLSHADERSP

        h = NLSHADERSP(self.strategy, rank_greediness=0.0)
        w = h._rank_weights([1, 2, 3, 4, 5])
        assert all(v == pytest.approx(1.0) for v in w.values())

    def test_rank_choice_favours_best(self):
        """With high greediness the best index is sampled far more often."""
        from panobbgo.heuristics.nlshadersp import NLSHADERSP

        h = NLSHADERSP(self.strategy, rank_greediness=3.0, seed=0)
        sorted_live = [0, 1, 2, 3]
        weights = h._rank_weights(sorted_live)
        counts = {i: 0 for i in sorted_live}
        for _ in range(5000):
            counts[h._rank_choice(sorted_live, weights)] += 1
        # best (index 0) chosen more often than worst (index 3).
        assert counts[0] > counts[3]
        # ordering respected throughout
        assert counts[0] > counts[1] > counts[3]

    def test_rank_choice_uniform_when_greediness_zero(self):
        """k = 0 → roughly equal selection counts across the pool."""
        from panobbgo.heuristics.nlshadersp import NLSHADERSP

        h = NLSHADERSP(self.strategy, rank_greediness=0.0, seed=0)
        sorted_live = [0, 1, 2, 3]
        weights = h._rank_weights(sorted_live)
        counts = {i: 0 for i in sorted_live}
        for _ in range(8000):
            counts[h._rank_choice(sorted_live, weights)] += 1
        # each index near 2000 (8000 / 4) within a generous tolerance.
        for c in counts.values():
            assert 1600 < c < 2400


# ----------------------------------------------------------------------
# Non-linear population size reduction (NLPSR)
# ----------------------------------------------------------------------


class NLSHADERSPNonLinearLPSRTests(_MockStrategyMixin, PanobbgoTestCase):
    def test_lpsr_target_endpoints_match_linear(self):
        from panobbgo.heuristics.nlshadersp import NLSHADERSP

        h = NLSHADERSP(self.strategy, NP_init=30, NP_min=4)
        assert h._lpsr_target(0.0) == 30
        assert h._lpsr_target(1.0) == 4

    def test_lpsr_target_shrinks_faster_than_linear_midrun(self):
        """In the interior the non-linear law gives a smaller population."""
        from panobbgo.heuristics.lshade import LSHADE
        from panobbgo.heuristics.nlshadersp import NLSHADERSP

        nl = NLSHADERSP(self.strategy, NP_init=30, NP_min=4)
        lin = LSHADE(self.strategy, NP_init=30, NP_min=4)
        for p in (0.25, 0.5, 0.75):
            assert nl._lpsr_target(p) < lin._lpsr_target(p)

    def test_lpsr_target_monotone_non_increasing(self):
        from panobbgo.heuristics.nlshadersp import NLSHADERSP

        h = NLSHADERSP(self.strategy, NP_init=40, NP_min=4)
        prev = h._lpsr_target(0.0)
        for p in np.linspace(0.0, 1.0, 21):
            cur = h._lpsr_target(float(p))
            assert cur <= prev + 0  # non-increasing
            prev = cur

    def test_apply_lpsr_uses_non_linear_schedule(self):
        """``_apply_lpsr`` drops slots down to the non-linear target."""
        from panobbgo.heuristics.lshade import _Dropped
        from panobbgo.heuristics.nlshadersp import NLSHADERSP

        h = NLSHADERSP(self.strategy, NP_init=20, NP_min=4, seed=1)
        h.on_start()
        h.get_points(limit=100)
        # Fill the whole population.
        items = list(h._pending.items())
        results = [
            _build_result(self.strategy, self.problem.random_point(), 10.0 + i, f"NLSHADE_RSP:{rid}")
            for i, (rid, _m) in enumerate(items)
        ]
        h.on_new_results(results)
        h.get_points(limit=200)
        # Half budget spent.
        self.strategy.results = list(range(500))  # max_eval = 1000 → progress 0.5
        h._apply_lpsr()
        target = h._lpsr_target(0.5)
        assert h._NP_current == target
        live = [s for s in h._population if not isinstance(s, _Dropped) and s is not None]
        assert len(live) <= target

    def test_lpsr_noop_when_budget_unknown(self):
        from panobbgo.heuristics.nlshadersp import NLSHADERSP

        h = NLSHADERSP(self.strategy, NP_init=20, NP_min=4, seed=1)
        h.on_start()
        h.get_points(limit=100)
        self.strategy.config.max_eval = 0  # unknown budget → progress None
        before = h._NP_current
        h._apply_lpsr()
        assert h._NP_current == before


# ----------------------------------------------------------------------
# Generate trial: rank-pressured current-to-pbest-w/1
# ----------------------------------------------------------------------


class NLSHADERSPGenerateTrialTests(_MockStrategyMixin, PanobbgoTestCase):
    def _seed_population(self, h, fx_seq):
        items = list(h._pending.items())
        results = []
        for (req_id, _meta), fx in zip(items, fx_seq):
            x = self.problem.random_point()
            results.append(_build_result(self.strategy, x, fx, f"NLSHADE_RSP:{req_id}"))
        h.on_new_results(results)

    def test_filled_population_emits_evolutionary_trials(self):
        from panobbgo.heuristics.nlshadersp import NLSHADERSP

        h = NLSHADERSP(self.strategy, NP_init=5, seed=12)
        h.on_start()
        h.get_points(limit=100)

        items = list(h._pending.items())[:5]
        results = []
        for req_id, meta in items:
            x = self.problem.random_point()
            results.append(_build_result(self.strategy, x, 10.0 + meta.slot_idx, f"NLSHADE_RSP:{req_id}"))
        h.on_new_results(results)

        emitted = h.get_points(limit=100)
        assert len(emitted) >= 1
        for pt in emitted:
            assert pt.who.startswith("NLSHADE_RSP:")
        evo_metas = [m for _, m in h._pending.items() if not np.isnan(m.F) and not np.isnan(m.CR)]
        assert len(evo_metas) >= 1

    def test_trial_components_inside_box(self):
        """Emitted evolutionary trials respect the box after bounds reflection."""
        from panobbgo.heuristics.nlshadersp import NLSHADERSP

        h = NLSHADERSP(self.strategy, NP_init=6, seed=21)
        h.on_start()
        h.get_points(limit=100)
        self._seed_population(h, fx_seq=[100.0, 90.0, 80.0, 70.0, 60.0, 50.0])
        emitted = h.get_points(limit=100)
        for pt in emitted:
            assert np.all(pt.x >= self.problem.box[:, 0] - 1e-9)
            assert np.all(pt.x <= self.problem.box[:, 1] + 1e-9)

    def test_better_trial_wins_and_archives_parent(self):
        from panobbgo.heuristics.nlshadersp import NLSHADERSP

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
        r = _build_result(self.strategy, x, improved_fx, f"NLSHADE_RSP:{rid}")
        h.on_new_results([r])

        assert h._population[target_slot].fx == improved_fx
        assert len(h._archive) >= 1
        assert len(h._success_F) >= 1
        assert all(np.isfinite(f) for f in h._success_F)
        assert all(np.isfinite(c) for c in h._success_CR)

    def test_archive_members_eligible_as_r2(self):
        """With archive entries present the trial still emits (r2 may be archival)."""
        from panobbgo.heuristics.nlshadersp import NLSHADERSP

        h = NLSHADERSP(self.strategy, NP_init=5, seed=33)
        h.on_start()
        h.get_points(limit=100)
        self._seed_population(h, fx_seq=[100.0, 90.0, 80.0, 70.0, 60.0])
        h.get_points(limit=200)
        # Inject a couple of archive vectors and force a fresh trial.
        h._archive.append(np.array([0.3, 0.3]))
        h._archive.append(np.array([0.6, 0.6]))
        live = h._live_indices()
        assert live
        h._generate_trial(live[0])
        emitted = h.get_points(limit=10)
        # Either a trial was emitted (most likely) or the slot was busy; the
        # key invariant is no exception was raised on the mixed union draw.
        for pt in emitted:
            assert pt.who.startswith("NLSHADE_RSP:")


# ----------------------------------------------------------------------
# Restart behaviour (inherited jSO contract)
# ----------------------------------------------------------------------


class NLSHADERSPRestartTests(_MockStrategyMixin, PanobbgoTestCase):
    def test_restart_re_stamps_jso_memory(self):
        from panobbgo.heuristics.jso import _ANCHOR_M_CR, _ANCHOR_M_F, _INIT_M_CR, _INIT_M_F
        from panobbgo.heuristics.nlshadersp import NLSHADERSP

        h = NLSHADERSP(self.strategy, NP_init=6, seed=2)
        h.on_start()
        h.get_points(limit=100)
        h._M_F[:] = 0.6
        h._M_CR[:] = 0.4
        h._archive.append(np.array([0.5, 0.5]))
        h._success_F = [0.7]

        h.on_restart(np.array([0.0, 0.0]), reason="test")

        assert h._archive == []
        assert h._success_F == []
        assert np.allclose(h._M_F[:-1], _INIT_M_F)
        assert np.allclose(h._M_CR[:-1], _INIT_M_CR)
        assert h._M_F[-1] == _ANCHOR_M_F
        assert h._M_CR[-1] == _ANCHOR_M_CR
        assert len(h._pending) == h.NP_init

    def test_restart_before_start_is_noop(self):
        from panobbgo.heuristics.nlshadersp import NLSHADERSP

        h = NLSHADERSP(self.strategy, NP_init=4, seed=4)
        h.on_restart(None)
        assert h.get_points(limit=100) == []


# ----------------------------------------------------------------------
# Smoke test: end-to-end progress on a simple landscape
# ----------------------------------------------------------------------


class NLSHADERSPSmokeTests(_MockStrategyMixin, PanobbgoTestCase):
    def test_smoke_quadratic_no_regression(self):
        from panobbgo.heuristics.lshade import _Dropped
        from panobbgo.heuristics.nlshadersp import NLSHADERSP
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
            results.append(Result(Point(x, f"NLSHADE_RSP:{rid}"), fx_of(x)))
        h.on_new_results(results)
        h.get_points(limit=100)

        best_fx_before = min(s.fx for s in h._population if isinstance(s, Result))

        for _round in range(20):
            pending_snapshot = list(h._pending.items())
            if not pending_snapshot:
                break
            h.get_points(limit=200)
            results = []
            for rid, meta in pending_snapshot:
                slot = h._population[meta.slot_idx]
                if isinstance(slot, _Dropped) or slot is None:
                    continue
                x = self.problem.project(np.asarray(slot.x) + 0.1 * np.random.randn(self.problem.dim))
                results.append(Result(Point(x, f"NLSHADE_RSP:{rid}"), fx_of(x)))
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
        from panobbgo.heuristics.nlshadersp import NLSHADERSP
        from panobbgo.self_improve import StructuralMutationRule, default_structural_catalog

        catalog = default_structural_catalog()
        add_rules = [r for r in catalog.rules if isinstance(r, StructuralMutationRule) and r.op == "add_heuristic"]
        assert add_rules
        has_nl = any(cls is NLSHADERSP for rule in add_rules for cls, _ in (rule.candidate_classes or ()))
        assert has_nl

    def test_kwarg_catalog_has_nlshadersp_dials(self):
        from panobbgo.self_improve import MutationRule, default_catalog

        rules = default_catalog().rules
        params = {
            (r.class_name, r.param_name) for r in rules if isinstance(r, MutationRule) and r.class_name == "NLSHADERSP"
        }
        assert ("NLSHADERSP", "NP_init") in params
        assert ("NLSHADERSP", "rank_greediness") in params
