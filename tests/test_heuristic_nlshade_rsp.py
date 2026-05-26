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

"""Tests for the NL-SHADE-RSP (Stanovov 2020) adaptive DE heuristic."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from panobbgo.utils import PanobbgoTestCase


class _MockStrategyMixin:
    """Same scaffolding as :mod:`tests.test_heuristic_jso`.

    NL-SHADE-RSP inherits the jSO / L-SHADE LPSR / constraint-handler /
    max_eval semantics, so the mock strategy needs the same setup.  The
    singleton ``config.max_eval`` is saved and restored on each test.
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
        from panobbgo.heuristics.nlshade_rsp import (
            NLSHADE_RSP,
            _DEFAULT_ARCHIVE_FACTOR,
            _DEFAULT_KP,
        )

        h = NLSHADE_RSP(self.strategy)
        # jSO-inherited defaults.
        assert h.NP_init == 30
        assert h.NP_min == 4
        assert h.H == 5
        assert h.p_best_max == 0.25
        assert h.p_best_min == 0.125
        # NL-SHADE-RSP-specific defaults.
        assert h.kp == _DEFAULT_KP == 3.0
        assert h.adaptive_archive is True
        assert h.archive_factor == _DEFAULT_ARCHIVE_FACTOR == 2.6
        assert h.name == "NLSHADE_RSP"
        # jSO opts into the F-cap by construction; NL-SHADE-RSP inherits it.
        assert h.F_schedule is True

    def test_custom_construction(self):
        from panobbgo.heuristics.nlshade_rsp import NLSHADE_RSP

        h = NLSHADE_RSP(
            self.strategy,
            NP_init=20,
            NP_min=6,
            H=4,
            p_best_max=0.3,
            p_best_min=0.1,
            archive_factor=1.0,
            kp=1.5,
            adaptive_archive=False,
            seed=7,
            name="MyRSP",
        )
        assert h.NP_init == 20
        assert h.kp == 1.5
        assert h.adaptive_archive is False
        assert h.archive_factor == 1.0
        assert h.name == "MyRSP"

    def test_subclass_of_jso_and_lshade(self):
        from panobbgo.heuristics.jso import JSO
        from panobbgo.heuristics.lshade import LSHADE
        from panobbgo.heuristics.nlshade_rsp import NLSHADE_RSP

        h = NLSHADE_RSP(self.strategy)
        assert isinstance(h, JSO)
        assert isinstance(h, LSHADE)

    def test_invalid_kp(self):
        from panobbgo.heuristics.nlshade_rsp import NLSHADE_RSP

        with pytest.raises(ValueError, match="kp"):
            NLSHADE_RSP(self.strategy, kp=-1.0)
        with pytest.raises(ValueError, match="kp"):
            NLSHADE_RSP(self.strategy, kp=float("nan"))
        with pytest.raises(ValueError, match="kp"):
            NLSHADE_RSP(self.strategy, kp=float("inf"))

    def test_kp_zero_is_allowed(self):
        from panobbgo.heuristics.nlshade_rsp import NLSHADE_RSP

        h = NLSHADE_RSP(self.strategy, kp=0.0)
        assert h.kp == 0.0

    def test_invalid_adaptive_archive(self):
        from panobbgo.heuristics.nlshade_rsp import NLSHADE_RSP

        with pytest.raises(ValueError, match="adaptive_archive"):
            NLSHADE_RSP(self.strategy, adaptive_archive="yes")  # type: ignore[arg-type]

    def test_inherited_jso_validation_still_fires(self):
        from panobbgo.heuristics.nlshade_rsp import NLSHADE_RSP

        with pytest.raises(ValueError, match="H must be >= 2"):
            NLSHADE_RSP(self.strategy, H=1)
        with pytest.raises(ValueError, match="p_best_min"):
            NLSHADE_RSP(self.strategy, p_best_max=0.2, p_best_min=0.3)


# ----------------------------------------------------------------------
# Rank-based selective pressure (_select_r1)
# ----------------------------------------------------------------------


class NLSHADERSPSelectionTests(_MockStrategyMixin, PanobbgoTestCase):
    def test_kp_positive_biases_toward_best(self):
        """Better-ranked indices are picked far more often than worse ones."""
        from panobbgo.heuristics.nlshade_rsp import NLSHADE_RSP

        h = NLSHADE_RSP(self.strategy, kp=3.0, seed=42)
        live = [0, 1, 2, 3, 4]
        sorted_live = [0, 1, 2, 3, 4]  # best-first
        counts: Counter = Counter()
        for _ in range(8000):
            counts[h._select_r1(live, target_idx=99, sorted_live=sorted_live)] += 1
        # Strictly monotone-ish: the best (0) dominates the worst (4).
        assert counts[0] > counts[4]
        assert counts[0] > counts[2] > counts[4]

    def test_kp_zero_is_uniform(self):
        """kp == 0 defers to the inherited uniform draw."""
        from panobbgo.heuristics.nlshade_rsp import NLSHADE_RSP

        h = NLSHADE_RSP(self.strategy, kp=0.0, seed=42)
        live = [0, 1, 2, 3, 4]
        sorted_live = [0, 1, 2, 3, 4]
        counts: Counter = Counter()
        for _ in range(8000):
            counts[h._select_r1(live, target_idx=99, sorted_live=sorted_live)] += 1
        lo, hi = min(counts.values()), max(counts.values())
        # Uniform-ish: the spread between most and least frequent is small.
        assert hi / lo < 1.3

    def test_excludes_target(self):
        from panobbgo.heuristics.nlshade_rsp import NLSHADE_RSP

        h = NLSHADE_RSP(self.strategy, kp=3.0, seed=1)
        for _ in range(200):
            r1 = h._select_r1([0, 1, 2], target_idx=1, sorted_live=[0, 1, 2])
            assert r1 != 1
            assert r1 in (0, 2)

    def test_empty_pool_returns_none(self):
        from panobbgo.heuristics.nlshade_rsp import NLSHADE_RSP

        h = NLSHADE_RSP(self.strategy, kp=3.0, seed=1)
        assert h._select_r1(live=[5], target_idx=5, sorted_live=[5]) is None

    def test_probabilities_sum_to_one_and_valid(self):
        """Draws always land inside the eligible pool."""
        from panobbgo.heuristics.nlshade_rsp import NLSHADE_RSP

        h = NLSHADE_RSP(self.strategy, kp=2.0, seed=3)
        pool = [10, 11, 12, 13]
        for _ in range(500):
            r1 = h._select_r1(pool, target_idx=99, sorted_live=pool)
            assert r1 in pool


# ----------------------------------------------------------------------
# Adaptive archive size
# ----------------------------------------------------------------------


class NLSHADERSPArchiveTests(_MockStrategyMixin, PanobbgoTestCase):
    def test_max_archive_uses_archive_factor_and_np(self):
        from panobbgo.heuristics.nlshade_rsp import NLSHADE_RSP

        h = NLSHADE_RSP(self.strategy, NP_init=20, archive_factor=2.0, seed=1)
        h._NP_current = 20
        assert h._max_archive() == 40

    def test_refresh_draws_within_bounds(self):
        from panobbgo.heuristics.nlshade_rsp import NLSHADE_RSP

        h = NLSHADE_RSP(self.strategy, NP_init=20, archive_factor=2.0, seed=1)
        h._NP_current = 20
        seen = set()
        for _ in range(500):
            h._refresh_archive_target()
            assert 0 <= h._archive_size_target <= 40
            seen.add(h._archive_size_target)
        # The draw genuinely varies (not pinned to a single value).
        assert len(seen) > 5

    def test_adaptive_cap_returns_target(self):
        from panobbgo.heuristics.nlshade_rsp import NLSHADE_RSP

        h = NLSHADE_RSP(self.strategy, adaptive_archive=True, seed=1)
        h._archive_size_target = 7
        assert h._archive_cap() == 7

    def test_fixed_cap_when_adaptive_disabled(self):
        from panobbgo.heuristics.nlshade_rsp import NLSHADE_RSP

        h = NLSHADE_RSP(self.strategy, NP_init=20, archive_factor=2.0, adaptive_archive=False, seed=1)
        h._NP_current = 20
        # The adaptive target is ignored; the fixed jSO cap is used.
        h._archive_size_target = 3
        assert h._archive_cap() == 40

    def test_trim_archive_respects_adaptive_cap(self):
        from panobbgo.heuristics.nlshade_rsp import NLSHADE_RSP

        h = NLSHADE_RSP(self.strategy, NP_init=10, archive_factor=1.0, seed=1)
        h._NP_current = 10
        h._archive_size_target = 3
        h._archive = [np.zeros(2) for _ in range(10)]
        h._trim_archive()
        assert len(h._archive) == 3

    def test_end_of_generation_refreshes_target(self):
        from panobbgo.heuristics.nlshade_rsp import NLSHADE_RSP

        h = NLSHADE_RSP(self.strategy, NP_init=20, archive_factor=2.0, seed=1)
        h._NP_current = 20
        h._archive_size_target = 10**6  # sentinel out of range
        h._end_of_generation()
        assert h._archive_size_target <= h._max_archive()

    def test_on_start_refreshes_target(self):
        from panobbgo.heuristics.nlshade_rsp import NLSHADE_RSP

        h = NLSHADE_RSP(self.strategy, NP_init=20, archive_factor=2.0, seed=1)
        h._archive_size_target = 10**6
        h.on_start()
        assert h._archive_size_target <= h._max_archive()

    def test_on_restart_refreshes_target(self):
        from panobbgo.heuristics.nlshade_rsp import NLSHADE_RSP

        h = NLSHADE_RSP(self.strategy, NP_init=20, archive_factor=2.0, seed=1)
        h.on_start()
        h.get_points(limit=100)
        h._archive_size_target = 10**6
        h.on_restart(np.array([0.0, 0.0]), reason="test")
        assert h._archive_size_target <= h._max_archive()


# ----------------------------------------------------------------------
# Heuristic interface / integration
# ----------------------------------------------------------------------


class NLSHADERSPInterfaceTests(_MockStrategyMixin, PanobbgoTestCase):
    def test_on_start_emits_NP_init_points(self):
        from panobbgo.heuristics.nlshade_rsp import NLSHADE_RSP

        h = NLSHADE_RSP(self.strategy, NP_init=6, seed=1)
        h.on_start()
        emitted = h.get_points(limit=100)
        assert len(emitted) == 6
        for pt in emitted:
            assert pt.who.startswith("NLSHADE_RSP:")

    def test_filled_population_emits_evolutionary_trials(self):
        from panobbgo.heuristics.nlshade_rsp import NLSHADE_RSP

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
        evo_metas = [m for _, m in h._pending.items() if not np.isnan(m.F) and not np.isnan(m.CR)]
        assert len(evo_metas) >= 1

    def test_better_trial_wins_and_archives_parent(self):
        from panobbgo.heuristics.nlshade_rsp import NLSHADE_RSP

        h = NLSHADE_RSP(self.strategy, NP_init=4, archive_factor=2.0, seed=13)
        h.on_start()
        h.get_points(limit=100)

        items = list(h._pending.items())
        results = []
        for (req_id, _meta), fx in zip(items, [100.0, 110.0, 120.0, 130.0]):
            x = self.problem.random_point()
            results.append(_build_result(self.strategy, x, fx, f"NLSHADE_RSP:{req_id}"))
        h.on_new_results(results)
        h.get_points(limit=100)

        target_slot = 0
        target_fx = h._population[target_slot].fx
        rid_meta = [(r, m) for r, m in h._pending.items() if m.slot_idx == target_slot]
        assert rid_meta
        rid, _ = rid_meta[0]

        improved_fx = target_fx - 50.0
        x = self.problem.random_point()
        r = _build_result(self.strategy, x, improved_fx, f"NLSHADE_RSP:{rid}")
        # Ensure the adaptive cap allows at least one archived parent.
        h._archive_size_target = 5
        h.on_new_results([r])

        assert h._population[target_slot].fx == improved_fx
        assert len(h._archive) >= 1

    def test_smoke_quadratic_no_regression(self):
        """Driving NL-SHADE-RSP on f(x) = ||x||² makes no negative progress."""
        from panobbgo.heuristics.lshade import _Dropped
        from panobbgo.heuristics.nlshade_rsp import NLSHADE_RSP
        from panobbgo.lib import Point, Result

        h = NLSHADE_RSP(self.strategy, NP_init=8, NP_min=4, seed=5)
        h.on_start()

        def fx_of(x):
            return float(np.dot(x, x))

        items = list(h._pending.items())
        h.get_points(limit=100)
        results = [Result(Point(x := self.problem.random_point(), f"NLSHADE_RSP:{rid}"), fx_of(x)) for rid, _ in items]
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

        assert hasattr(h, "NLSHADE_RSP")
        assert "NLSHADE_RSP" in h.__all__

    def test_in_structural_catalog(self):
        from panobbgo.heuristics.nlshade_rsp import NLSHADE_RSP
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
        assert ("NLSHADE_RSP", "kp") in params
        assert ("NLSHADE_RSP", "adaptive_archive") in params
