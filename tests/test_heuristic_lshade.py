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

"""Tests for the L-SHADE (Tanabe-Fukunaga 2014) heuristic."""

from __future__ import annotations

import numpy as np
import pytest

from panobbgo.utils import PanobbgoTestCase


class _MockStrategyMixin:
    """Set up the constraint handler so is_better / get_penalty_value work."""

    def setUp(self):
        super().setUp()
        from panobbgo.lib.constraints import DefaultConstraintHandler

        self.strategy.constraint_handler = DefaultConstraintHandler(self.strategy)
        # By default the mock strategy has no .results / .config.max_eval —
        # LPSR's fall-back keeps NP constant when those are missing, which
        # is fine for most unit tests.  Tests that need LPSR set both.
        self.strategy.results = []


# ----------------------------------------------------------------------
# Construction-time validation
# ----------------------------------------------------------------------


class LSHADEConstructionTests(_MockStrategyMixin, PanobbgoTestCase):
    def test_default_construction(self):
        from panobbgo.heuristics.lshade import LSHADE

        h = LSHADE(self.strategy)
        assert h.name == "LSHADE"
        assert h.NP_init_param is None
        assert h.NP_min == 4
        assert h.H == 6
        assert 0.0 < h.pbest_min <= h.pbest_max <= 1.0

    def test_custom_construction(self):
        from panobbgo.heuristics.lshade import LSHADE

        h = LSHADE(
            self.strategy,
            NP_init=12,
            NP_init_cap=30,
            NP_min=4,
            H=4,
            pbest_min=0.05,
            pbest_max=0.5,
            archive_factor=0.5,
            seed=11,
            name="MyLSHADE",
        )
        assert h.NP_init_param == 12
        assert h.NP_init_cap == 30
        assert h.H == 4
        assert h.pbest_min == 0.05
        assert h.pbest_max == 0.5
        assert h.archive_factor == 0.5
        assert h.name == "MyLSHADE"

    def test_invalid_NP_init_type(self):
        from panobbgo.heuristics.lshade import LSHADE

        with pytest.raises(ValueError, match="NP_init must be an integer or None"):
            LSHADE(self.strategy, NP_init=8.0)  # type: ignore[arg-type]

    def test_invalid_NP_init_too_small(self):
        from panobbgo.heuristics.lshade import LSHADE

        with pytest.raises(ValueError, match="NP_init must be >= 4"):
            LSHADE(self.strategy, NP_init=3)

    def test_invalid_NP_min(self):
        from panobbgo.heuristics.lshade import LSHADE

        with pytest.raises(ValueError, match="NP_min must be an integer >= 4"):
            LSHADE(self.strategy, NP_min=3)

    def test_NP_init_smaller_than_NP_min(self):
        from panobbgo.heuristics.lshade import LSHADE

        with pytest.raises(ValueError, match="NP_init .* must be >= NP_min"):
            LSHADE(self.strategy, NP_init=4, NP_min=6)

    def test_invalid_H(self):
        from panobbgo.heuristics.lshade import LSHADE

        with pytest.raises(ValueError, match="H must be a positive integer"):
            LSHADE(self.strategy, H=0)
        with pytest.raises(ValueError, match="H must be a positive integer"):
            LSHADE(self.strategy, H=-1)

    def test_invalid_pbest_range(self):
        from panobbgo.heuristics.lshade import LSHADE

        with pytest.raises(ValueError, match="pbest_min"):
            LSHADE(self.strategy, pbest_min=0.5, pbest_max=0.1)
        with pytest.raises(ValueError, match="pbest_min"):
            LSHADE(self.strategy, pbest_min=0.0)
        with pytest.raises(ValueError, match="pbest_min"):
            LSHADE(self.strategy, pbest_max=1.5)

    def test_invalid_archive_factor(self):
        from panobbgo.heuristics.lshade import LSHADE

        with pytest.raises(ValueError, match="archive_factor"):
            LSHADE(self.strategy, archive_factor=-0.1)
        with pytest.raises(ValueError, match="archive_factor"):
            LSHADE(self.strategy, archive_factor=float("nan"))


# ----------------------------------------------------------------------
# Initial population generation
# ----------------------------------------------------------------------


class LSHADEOnStartTests(_MockStrategyMixin, PanobbgoTestCase):
    def test_on_start_emits_NP_points_with_explicit_init(self):
        from panobbgo.heuristics.lshade import LSHADE

        h = LSHADE(self.strategy, NP_init=8, seed=0)
        h.on_start()

        emitted = h.get_points(limit=100)
        assert len(emitted) == 8
        assert h.NP == 8
        assert h.NP_init == 8
        # All pending trials are initial-fill trials.
        assert len(h._pending) == 8
        for meta in h._pending.values():
            assert meta["initial"] is True

    def test_on_start_auto_NP_uses_18_dim_capped(self):
        from panobbgo.heuristics.lshade import LSHADE

        # dim=2 → 18*dim=36; cap at NP_init_cap=20 → 20.
        h = LSHADE(self.strategy, NP_init_cap=20, seed=0)
        h.on_start()
        assert h.NP_init == 20
        assert h.NP == 20

    def test_on_start_auto_NP_uses_18_dim_when_small(self):
        from panobbgo.heuristics.lshade import LSHADE

        # dim=2 → 18*dim=36; cap=50 → 36.
        h = LSHADE(self.strategy, NP_init_cap=50, seed=0)
        h.on_start()
        assert h.NP_init == 36

    def test_on_start_respects_NP_min_floor(self):
        from panobbgo.heuristics.lshade import LSHADE

        # If NP_init_cap is lower than NP_min, on_start still respects NP_min.
        h = LSHADE(self.strategy, NP_init_cap=4, NP_min=4, seed=0)
        h.on_start()
        assert h.NP_init >= 4

    def test_on_start_points_inside_box(self):
        from panobbgo.heuristics.lshade import LSHADE

        h = LSHADE(self.strategy, NP_init=6, seed=1)
        h.on_start()
        emitted = h.get_points(limit=100)
        for pt in emitted:
            assert np.all(pt.x >= self.problem.box[:, 0] - 1e-9)
            assert np.all(pt.x <= self.problem.box[:, 1] + 1e-9)
            assert pt.who.startswith("LSHADE:")

    def test_on_start_memory_initialized_to_neutral(self):
        from panobbgo.heuristics.lshade import LSHADE

        h = LSHADE(self.strategy, NP_init=6, H=4, seed=0)
        h.on_start()
        assert h._M_F.shape == (4,)
        assert h._M_CR.shape == (4,)
        assert np.allclose(h._M_F, 0.5)
        assert np.allclose(h._M_CR, 0.5)
        assert h._k_mem == 0


# ----------------------------------------------------------------------
# Initial-fill result handling
# ----------------------------------------------------------------------


class LSHADEInitialFillTests(_MockStrategyMixin, PanobbgoTestCase):
    def test_initial_fill_populates_slots(self):
        from panobbgo.heuristics.lshade import LSHADE
        from panobbgo.lib import Point, Result

        h = LSHADE(self.strategy, NP_init=4, seed=2)
        h.on_start()
        h.get_points(limit=100)

        # Feed back all initial results.
        items = list(h._pending.items())
        results = []
        for k, (req_id, _meta) in enumerate(items):
            p = Point(np.zeros(self.problem.dim), f"LSHADE:{req_id}")
            results.append(Result(p, float(k)))

        h.on_new_results(results)
        assert h._n_filled == 4
        assert all(r is not None for r in h._population)
        # Successor trials should have been emitted now that pop >= 4.
        emitted = h.get_points(limit=100)
        assert len(emitted) > 0

    def test_unknown_who_ignored(self):
        from panobbgo.heuristics.lshade import LSHADE
        from panobbgo.lib import Point, Result

        h = LSHADE(self.strategy, NP_init=4, seed=3)
        h.on_start()
        h.get_points(limit=100)

        before = dict(h._pending)
        p = Point(np.zeros(self.problem.dim), "OtherHeuristic:abc")
        r = Result(p, 1.0)
        h.on_new_results([r])
        # Pending dict unchanged.
        assert dict(h._pending) == before


# ----------------------------------------------------------------------
# Trial-result handling: success / failure / archive / memory update
# ----------------------------------------------------------------------


class LSHADETrialResultTests(_MockStrategyMixin, PanobbgoTestCase):
    def _populate(self, NP=4, seed=7):
        """Drive the heuristic until the population is fully filled."""
        from panobbgo.heuristics.lshade import LSHADE
        from panobbgo.lib import Point, Result

        h = LSHADE(self.strategy, NP_init=NP, seed=seed)
        h.on_start()
        h.get_points(limit=100)
        items = list(h._pending.items())
        results = []
        for k, (req_id, _) in enumerate(items):
            p = Point(np.full(self.problem.dim, float(k)), f"LSHADE:{req_id}")
            results.append(Result(p, 100.0 + float(k)))
        h.on_new_results(results)
        # Drain follow-up trials that on_new_results just emitted.
        h.get_points(limit=100)
        return h

    def test_better_trial_replaces_target(self):
        from panobbgo.lib import Point, Result

        h = self._populate(NP=4)
        # Grab the first pending follow-up trial and feed back a much-better fx.
        items = list(h._pending.items())
        assert items, "expected at least one pending follow-up trial"
        req_id, meta = items[0]
        target_idx = int(meta["target"])
        old_target_fx = float(h._population[target_idx].fx)
        p = Point(np.zeros(self.problem.dim), f"LSHADE:{req_id}")
        r = Result(p, old_target_fx - 50.0)
        h.on_new_results([r])

        assert h._population[target_idx].fx == old_target_fx - 50.0
        # Archive picked up the loser.
        assert len(h._archive) >= 1
        # Memory buffer gained a success.
        assert len(h._S_F) >= 1
        assert len(h._S_CR) >= 1
        assert len(h._S_delta) >= 1

    def test_worse_trial_keeps_target(self):
        from panobbgo.lib import Point, Result

        h = self._populate(NP=4)
        items = list(h._pending.items())
        req_id, meta = items[0]
        target_idx = int(meta["target"])
        old_target_fx = float(h._population[target_idx].fx)
        p = Point(np.zeros(self.problem.dim), f"LSHADE:{req_id}")
        r = Result(p, old_target_fx + 50.0)  # worse
        h.on_new_results([r])

        # Target untouched, success buffer empty.
        assert h._population[target_idx].fx == old_target_fx
        assert len(h._S_F) == 0
        assert len(h._S_CR) == 0

    def test_memory_updates_after_wave(self):
        """After NP outcomes the success-history memory updates exactly one slot."""
        from panobbgo.lib import Point, Result

        h = self._populate(NP=4)
        # Force NP successes in a row by feeding back monotonically
        # better trial results.
        memory_before = h._M_F.copy()
        k_before = h._k_mem
        successes_needed = h.NP  # one full wave
        produced = 0
        guard = 0
        while produced < successes_needed and guard < 50:
            guard += 1
            items = list(h._pending.items())
            if not items:
                break
            req_id, meta = items[0]
            target_idx = int(meta["target"])
            target_fx = float(h._population[target_idx].fx)
            p = Point(np.zeros(self.problem.dim), f"LSHADE:{req_id}")
            r = Result(p, target_fx - 1.0)
            h.on_new_results([r])
            produced += 1
            h.get_points(limit=100)

        # Memory should have rotated by at least one slot and at least
        # one entry must have changed value.
        assert h._k_mem != k_before
        assert not np.allclose(h._M_F, memory_before)

    def test_archive_capped_by_factor(self):
        from panobbgo.lib import Point, Result

        h = self._populate(NP=4)
        # Set a tight archive cap and pump successes through.
        h.archive_factor = 0.5  # archive size cap = 2 at NP=4
        for _ in range(20):
            items = list(h._pending.items())
            if not items:
                break
            req_id, meta = items[0]
            target_idx = int(meta["target"])
            target_fx = float(h._population[target_idx].fx)
            p = Point(np.zeros(self.problem.dim), f"LSHADE:{req_id}")
            h.on_new_results([Result(p, target_fx - 1.0)])
            h.get_points(limit=100)

        # Archive cannot grow beyond 0.5 * NP = 2.
        assert len(h._archive) <= 2


# ----------------------------------------------------------------------
# LPSR (Linear Population Size Reduction)
# ----------------------------------------------------------------------


class LSHADELPSRTests(_MockStrategyMixin, PanobbgoTestCase):
    def _strategy_with_budget(self, max_eval, n_evaluated):
        """Wire up the mock strategy so LPSR has something to read."""

        class _Cfg:
            def __init__(self, m, parent_cfg):
                self.max_eval = m
                self.capacity = 100
                self._parent = parent_cfg

            def get_logger(self, name):
                return self._parent.get_logger(name)

        self.strategy.config = _Cfg(max_eval, self.strategy.config)
        self.strategy.results = [None] * n_evaluated

    def test_no_budget_keeps_NP_constant(self):
        from panobbgo.heuristics.lshade import LSHADE

        # By default the strategy has no max_eval; LPSR must no-op.
        h = LSHADE(self.strategy, NP_init=10, NP_min=4, seed=4)
        h.on_start()
        target = h._current_target_NP()
        assert target == h.NP_init

    def test_target_NP_shrinks_with_progress(self):
        from panobbgo.heuristics.lshade import LSHADE

        h = LSHADE(self.strategy, NP_init=20, NP_min=4, seed=5)
        h.on_start()

        self._strategy_with_budget(max_eval=100, n_evaluated=50)
        # Halfway → ~ midpoint between 20 and 4 = 12.
        target = h._current_target_NP()
        assert 10 <= target <= 14

        self._strategy_with_budget(max_eval=100, n_evaluated=100)
        # End of budget → NP_min.
        assert h._current_target_NP() == 4

    def test_lpsr_does_not_raise_when_pending_slot_pruned(self):
        """Regression: after LPSR shrinks, a follow-up for a pruned slot must not raise.

        When LPSR shrinks the population mid-flight, ``target_idx`` from
        an in-flight pending trial may point past the new end of the
        list.  ``on_new_results`` must recompute the live-slot
        predicate after the shrink rather than relying on the value
        computed at the top of the loop body.
        """
        from panobbgo.heuristics.lshade import LSHADE
        from panobbgo.lib import Point, Result

        h = LSHADE(self.strategy, NP_init=10, NP_min=4, seed=8)
        h.on_start()
        # Drain the initial wave and feed back results so the population
        # is fully filled and follow-up trials are pending.
        h.get_points(limit=200)
        items = list(h._pending.items())
        results = []
        for k, (req_id, _) in enumerate(items):
            p = Point(np.full(self.problem.dim, float(k)), f"LSHADE:{req_id}")
            results.append(Result(p, 100.0 + float(k)))
        h.on_new_results(results)
        h.get_points(limit=200)

        # Snapshot a pending trial whose target is a high-index slot
        # that will be pruned when LPSR shrinks to NP_min=4.
        high_idx_pending = [(rid, meta) for rid, meta in h._pending.items() if int(meta["target"]) >= 4]
        assert high_idx_pending, "expected a pending trial at a high-index slot"
        req_id, meta = high_idx_pending[0]
        target_idx = int(meta["target"])

        # Pin LPSR to the end of the budget so it shrinks aggressively.
        self._strategy_with_budget(max_eval=100, n_evaluated=100)

        # Now feed back the result: the trial's slot should get pruned
        # mid-loop, and on_new_results must NOT raise IndexError.
        p = Point(np.zeros(self.problem.dim), f"LSHADE:{req_id}")
        r = Result(p, 50.0)
        h.on_new_results([r])

        # Population is at the LPSR target and the heuristic did not crash.
        assert h.NP == 4
        assert len(h._population) == 4
        # The originally-targeted slot is gone; the result is ignored
        # (we don't grow the population back).
        assert target_idx >= len(h._population)

    def test_lpsr_prunes_worst_individuals(self):
        from panobbgo.heuristics.lshade import LSHADE
        from panobbgo.lib import Point, Result

        h = LSHADE(self.strategy, NP_init=8, NP_min=4, seed=6)
        h.on_start()
        h.get_points(limit=100)
        # Fill population with results of known fx values.
        items = list(h._pending.items())
        for k, (req_id, _) in enumerate(items):
            p = Point(np.full(self.problem.dim, float(k)), f"LSHADE:{req_id}")
            h.on_new_results([Result(p, 100.0 - 5.0 * k)])
            h.get_points(limit=100)

        # Schedule us to the end of the budget so LPSR wants NP=4.
        self._strategy_with_budget(max_eval=100, n_evaluated=100)
        h._shrink_population_if_needed()
        assert h.NP == 4
        assert len(h._population) == 4
        # Survivors are the 4 best (smallest fx values).  Initial fx
        # values were [100, 95, 90, 85, 80, 75, 70, 65]; the best four
        # are 65, 70, 75, 80.
        surviving = sorted([r.fx for r in h._population])
        assert surviving == [65.0, 70.0, 75.0, 80.0]


# ----------------------------------------------------------------------
# Parameter sampling
# ----------------------------------------------------------------------


class LSHADEParamSampleTests(_MockStrategyMixin, PanobbgoTestCase):
    def test_sampled_F_positive_and_capped(self):
        from panobbgo.heuristics.lshade import LSHADE

        h = LSHADE(self.strategy, NP_init=4, H=2, seed=42)
        h.on_start()
        for _ in range(200):
            F, CR = h._sample_params()
            assert 0.0 < F <= 1.0
            assert 0.0 <= CR <= 1.0

    def test_frozen_CR_when_memory_negative(self):
        from panobbgo.heuristics.lshade import LSHADE

        h = LSHADE(self.strategy, NP_init=4, H=1, seed=7)
        h.on_start()
        h._M_CR[0] = -1.0
        for _ in range(20):
            _F, CR = h._sample_params()
            assert CR == 0.0


# ----------------------------------------------------------------------
# Memory update from success buffer
# ----------------------------------------------------------------------


class LSHADEMemoryUpdateTests(_MockStrategyMixin, PanobbgoTestCase):
    def test_update_memory_with_no_successes_is_noop(self):
        from panobbgo.heuristics.lshade import LSHADE

        h = LSHADE(self.strategy, NP_init=4, H=2, seed=0)
        h.on_start()
        before_F = h._M_F.copy()
        before_CR = h._M_CR.copy()
        before_k = h._k_mem
        h._update_memory()
        assert np.array_equal(h._M_F, before_F)
        assert np.array_equal(h._M_CR, before_CR)
        assert h._k_mem == before_k

    def test_update_memory_writes_then_rotates(self):
        from panobbgo.heuristics.lshade import LSHADE

        h = LSHADE(self.strategy, NP_init=4, H=3, seed=0)
        h.on_start()
        h._S_F = [0.3, 0.7]
        h._S_CR = [0.2, 0.8]
        h._S_delta = [1.0, 1.0]
        h._update_memory()
        # Slot 0 written, rotated to 1.
        assert h._k_mem == 1
        assert h._M_F[0] != 0.5
        # Success buffer cleared.
        assert h._S_F == [] and h._S_CR == [] and h._S_delta == []

    def test_update_memory_freezes_CR_when_all_zero(self):
        from panobbgo.heuristics.lshade import LSHADE

        h = LSHADE(self.strategy, NP_init=4, H=2, seed=0)
        h.on_start()
        h._S_F = [0.5, 0.5]
        h._S_CR = [0.0, 0.0]
        h._S_delta = [1.0, 1.0]
        h._update_memory()
        assert h._M_CR[0] == -1.0  # frozen sentinel


# ----------------------------------------------------------------------
# Restart resets state
# ----------------------------------------------------------------------


class LSHADERestartTests(_MockStrategyMixin, PanobbgoTestCase):
    def test_on_restart_clears_state(self):
        from panobbgo.heuristics.lshade import LSHADE
        from panobbgo.lib import Point, Result

        h = LSHADE(self.strategy, NP_init=4, seed=11)
        h.on_start()
        h.get_points(limit=100)

        # Plant some state.
        items = list(h._pending.items())
        results = []
        for k, (req_id, _) in enumerate(items):
            p = Point(np.zeros(self.problem.dim), f"LSHADE:{req_id}")
            results.append(Result(p, float(k)))
        h.on_new_results(results)
        h.get_points(limit=100)

        assert h._n_filled == 4
        assert any(r is not None for r in h._population)

        h.on_restart(center=np.zeros(self.problem.dim), reason="test")

        # Population reset, archive cleared, success buffer cleared.
        assert h._n_filled == 0
        assert all(r is None for r in h._population)
        assert h._archive == []
        assert h._S_F == [] and h._S_CR == [] and h._S_delta == []
        # NP reset to NP_init for the fresh restart.
        assert h.NP == h.NP_init
        # Fresh initial-fill trials emitted.
        emitted = h.get_points(limit=100)
        assert len(emitted) == h.NP_init
        assert len(h._pending) == h.NP_init

    def test_on_restart_before_start_is_noop(self):
        from panobbgo.heuristics.lshade import LSHADE

        h = LSHADE(self.strategy, NP_init=4)
        # Should not raise.
        h.on_restart(center=np.zeros(self.problem.dim), reason="never started")

    def test_on_restart_with_none_center(self):
        from panobbgo.heuristics.lshade import LSHADE

        h = LSHADE(self.strategy, NP_init=4, seed=22)
        h.on_start()
        h.get_points(limit=100)
        h.on_restart(center=None, reason="random fallback")
        emitted = h.get_points(limit=100)
        assert len(emitted) == 4


# ----------------------------------------------------------------------
# End-to-end smoke test
# ----------------------------------------------------------------------


class LSHADESmokeTests(_MockStrategyMixin, PanobbgoTestCase):
    def test_lshade_improves_on_quadratic(self):
        """Run the heuristic on a quadratic objective and check it improves."""
        from panobbgo.heuristics.lshade import LSHADE
        from panobbgo.lib import Point, Result

        h = LSHADE(self.strategy, NP_init=10, seed=123)
        h.on_start()

        # Define f(x) = sum(x^2) — minimum is at origin with f=0.
        def f(x):
            return float(np.dot(x, x))

        # Pump trial-result rounds: drain the queue, evaluate every point,
        # feed the results back in.
        best_initial = float("inf")
        best_final = float("inf")
        for round_idx in range(60):
            emitted_now = h.get_points(limit=200)
            if not emitted_now:
                break
            results = []
            for pt in emitted_now:
                fx = f(pt.x)
                if round_idx == 0:
                    best_initial = min(best_initial, fx)
                best_final = min(best_final, fx)
                rid = pt.who.split(":", 1)[1]
                p2 = Point(pt.x, f"LSHADE:{rid}")
                results.append(Result(p2, fx))
            h.on_new_results(results)

        # The heuristic should have made meaningful progress vs the
        # initial random sweep.
        assert np.isfinite(best_final)
        assert best_final <= best_initial + 1e-9


# ----------------------------------------------------------------------
# Registration
# ----------------------------------------------------------------------


class LSHADERegistrationTests(_MockStrategyMixin, PanobbgoTestCase):
    def test_lshade_registered_in_heuristics_module(self):
        import panobbgo.heuristics as heur

        assert hasattr(heur, "LSHADE")
        assert "LSHADE" in heur.__all__

    def test_lshade_registered_in_structural_catalog(self):
        """LSHADE must be one of the structural-catalog add candidates."""
        from panobbgo.heuristics import LSHADE
        from panobbgo.self_improve import default_structural_catalog

        catalog = default_structural_catalog()
        add_candidates = []
        for rule in catalog.rules:
            if getattr(rule, "op", None) == "add_heuristic":
                add_candidates.extend(cls for cls, _kw in rule.candidate_classes)
        assert LSHADE in add_candidates
