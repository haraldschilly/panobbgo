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
    """Set up the constraint handler exactly like the heuristic tests.

    :class:`~panobbgo.config.Config` is a process-wide singleton; mutating
    ``self.strategy.config.max_eval`` directly leaks across tests.  We
    save and restore it around each test so unrelated tests (e.g. PSO's
    adaptive-inertia schedule) keep observing their own ``max_eval``.
    """

    def setUp(self):
        super().setUp()
        from panobbgo.lib.constraints import DefaultConstraintHandler

        self.strategy.constraint_handler = DefaultConstraintHandler(self.strategy)
        # Snapshot the singleton attribute we are about to mutate so
        # tearDown can restore it; defaults to whatever the singleton
        # held before this test started.
        self._original_max_eval = self.strategy.config.max_eval
        # Default: max_eval = 0 so LPSR is disabled in unit tests.  Tests
        # that exercise LPSR override this in-place; tearDown restores
        # the original value below.
        self.strategy.config.max_eval = 0
        # Mock results sequence — len(strategy.results) drives LPSR.
        self.strategy.results = []

    def tearDown(self):
        # Restore the singleton's max_eval so subsequent tests are not
        # polluted by our ``max_eval = 0`` override.
        try:
            self.strategy.config.max_eval = self._original_max_eval
        except Exception:
            pass
        super().tearDown()


# ----------------------------------------------------------------------
# Construction-time validation
# ----------------------------------------------------------------------


class LSHADEConstructionTests(_MockStrategyMixin, PanobbgoTestCase):
    def test_default_construction(self):
        from panobbgo.heuristics.lshade import LSHADE

        h = LSHADE(self.strategy)
        assert h.NP_init_user is None
        assert h.NP_min == 4
        assert h.H == 6
        assert h.p_best_rate == pytest.approx(0.11)
        assert h.archive_factor == pytest.approx(2.6)
        assert h.name == "LSHADE"

    def test_custom_construction(self):
        from panobbgo.heuristics.lshade import LSHADE

        h = LSHADE(
            self.strategy,
            NP_init=20,
            NP_min=5,
            H=4,
            p_best_rate=0.2,
            archive_factor=1.5,
            seed=7,
            name="MyDE",
        )
        assert h.NP_init_user == 20
        assert h.NP_min == 5
        assert h.H == 4
        assert h.p_best_rate == 0.2
        assert h.archive_factor == 1.5
        assert h.name == "MyDE"

    def test_invalid_NP_init_type(self):
        from panobbgo.heuristics.lshade import LSHADE

        with pytest.raises(ValueError, match="NP_init must be an integer or None"):
            LSHADE(self.strategy, NP_init=20.0)  # type: ignore[arg-type]

    def test_invalid_NP_init_value(self):
        from panobbgo.heuristics.lshade import LSHADE

        with pytest.raises(ValueError, match="NP_init must be >= 4"):
            LSHADE(self.strategy, NP_init=3)

    def test_NP_init_smaller_than_NP_min(self):
        from panobbgo.heuristics.lshade import LSHADE

        with pytest.raises(ValueError, match="NP_init .* must be >= NP_min"):
            LSHADE(self.strategy, NP_init=4, NP_min=10)

    def test_invalid_NP_min(self):
        from panobbgo.heuristics.lshade import LSHADE

        with pytest.raises(ValueError, match="NP_min must be an integer"):
            LSHADE(self.strategy, NP_min=4.0)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="NP_min must be >= 4"):
            LSHADE(self.strategy, NP_min=2)

    def test_invalid_H(self):
        from panobbgo.heuristics.lshade import LSHADE

        with pytest.raises(ValueError, match="H must be a positive integer"):
            LSHADE(self.strategy, H=0)
        with pytest.raises(ValueError, match="H must be a positive integer"):
            LSHADE(self.strategy, H=-1)

    def test_invalid_p_best_rate(self):
        from panobbgo.heuristics.lshade import LSHADE

        with pytest.raises(ValueError, match="p_best_rate must be in"):
            LSHADE(self.strategy, p_best_rate=0.0)
        with pytest.raises(ValueError, match="p_best_rate must be in"):
            LSHADE(self.strategy, p_best_rate=1.5)
        with pytest.raises(ValueError, match="p_best_rate must be in"):
            LSHADE(self.strategy, p_best_rate=float("nan"))

    def test_invalid_archive_factor(self):
        from panobbgo.heuristics.lshade import LSHADE

        with pytest.raises(ValueError, match="archive_factor must be"):
            LSHADE(self.strategy, archive_factor=-0.1)
        with pytest.raises(ValueError, match="archive_factor must be"):
            LSHADE(self.strategy, archive_factor=float("inf"))


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------


class WeightedLehmerMeanTests:
    """Pure-function tests for the success-history weighted Lehmer mean."""

    def test_uniform_weights_collapses_to_unweighted(self):
        from panobbgo.heuristics.lshade import _weighted_lehmer_mean

        v = np.array([0.2, 0.5, 0.8])
        w = np.ones_like(v)
        # Lehmer mean L_2(v) = Σv² / Σv (with uniform weights)
        expected = float(np.sum(v * v) / np.sum(v))
        assert _weighted_lehmer_mean(v, w) == pytest.approx(expected)

    def test_all_zero_weights_uses_uniform_fallback(self):
        from panobbgo.heuristics.lshade import _weighted_lehmer_mean

        v = np.array([0.3, 0.7])
        w = np.zeros_like(v)
        # Falls back to uniform weights
        expected = float(np.sum(v * v) / np.sum(v))
        assert _weighted_lehmer_mean(v, w) == pytest.approx(expected)

    def test_empty_input_returns_nan(self):
        from panobbgo.heuristics.lshade import _weighted_lehmer_mean

        assert np.isnan(_weighted_lehmer_mean(np.array([]), np.array([])))

    def test_all_zero_values_returns_nan(self):
        from panobbgo.heuristics.lshade import _weighted_lehmer_mean

        v = np.zeros(3)
        w = np.array([1.0, 2.0, 3.0])
        assert np.isnan(_weighted_lehmer_mean(v, w))


class CauchyAndNormalSamplerTests:
    """The Cauchy / Normal samplers honour the L-SHADE clip rules."""

    def test_cauchy_F_always_in_valid_range(self):
        from panobbgo.heuristics.lshade import _sample_cauchy_F

        rng = np.random.default_rng(42)
        for _ in range(1000):
            f = _sample_cauchy_F(rng, mu=0.5, sigma=0.1)
            assert 0.0 < f <= 1.0

    def test_cauchy_F_centred_on_mu_when_well_behaved(self):
        from panobbgo.heuristics.lshade import _sample_cauchy_F

        rng = np.random.default_rng(0)
        samples = [_sample_cauchy_F(rng, mu=0.5, sigma=0.05) for _ in range(2000)]
        # Cauchy is heavy-tailed but the median exists and equals mu (when
        # not clipped).  With sigma=0.05 the median is well-defined.
        assert 0.4 < float(np.median(samples)) < 0.6

    def test_normal_CR_clipped(self):
        from panobbgo.heuristics.lshade import _sample_normal_CR

        rng = np.random.default_rng(99)
        for _ in range(1000):
            cr = _sample_normal_CR(rng, mu=0.5, sigma=0.5)
            assert 0.0 <= cr <= 1.0


# ----------------------------------------------------------------------
# Initial population emission
# ----------------------------------------------------------------------


class LSHADEInitialPopulationTests(_MockStrategyMixin, PanobbgoTestCase):
    def test_on_start_emits_NP_init_points(self):
        from panobbgo.heuristics.lshade import LSHADE

        h = LSHADE(self.strategy, NP_init=10, seed=42)
        h.on_start()

        emitted = h.get_points(limit=100)
        assert len(emitted) == 10
        assert h.NP_init == 10
        assert h.NP_current == 10
        assert len(h._pending) == 10

    def test_dim_scaled_default_NP_init(self):
        from panobbgo.heuristics.lshade import LSHADE

        # Rosenbrock(2) → dim=2 → 18*2 = 36 (clipped to cap if smaller)
        h = LSHADE(self.strategy, seed=42)
        h.on_start()

        # Default 18*dim, but capped at cap // 2 — Rosenbrock(2) with the
        # default cap (100) gives min(36, 50) = 36.
        assert h.NP_init >= h.NP_min
        emitted = h.get_points(limit=200)
        assert len(emitted) == h.NP_init

    def test_initial_memories_are_neutral(self):
        from panobbgo.heuristics.lshade import LSHADE

        h = LSHADE(self.strategy, NP_init=10, H=6)
        h.on_start()

        assert np.all(h.M_F == 0.5)
        assert np.all(h.M_CR == 0.5)
        assert h._mem_pos == 0

    def test_emitted_points_inside_box(self):
        from panobbgo.heuristics.lshade import LSHADE

        h = LSHADE(self.strategy, NP_init=10, seed=1)
        h.on_start()
        for pt in h.get_points(limit=100):
            assert np.all(pt.x >= self.problem.box[:, 0] - 1e-9)
            assert np.all(pt.x <= self.problem.box[:, 1] + 1e-9)


# ----------------------------------------------------------------------
# on_new_results bookkeeping
# ----------------------------------------------------------------------


class LSHADEResultProcessingTests(_MockStrategyMixin, PanobbgoTestCase):
    def _seed(self, NP_init=8, seed=3):
        from panobbgo.heuristics.lshade import LSHADE

        h = LSHADE(self.strategy, NP_init=NP_init, NP_min=4, seed=seed)
        h.on_start()
        # Drain the initial emit so we observe follow-up trials separately.
        self._initial_points = h.get_points(limit=100)
        return h

    def _result_for_pending(self, h, req_id, fx, x=None):
        from panobbgo.lib import Point, Result

        target_idx, _, _, _ = h._pending[req_id]
        if x is None:
            x = self.problem.random_point()
        p = Point(np.asarray(x, dtype=float), f"LSHADE:{req_id}")
        return Result(p, float(fx)), target_idx

    def test_unknown_who_ignored(self):
        from panobbgo.lib import Point, Result

        h = self._seed(NP_init=5)
        before_pending = dict(h._pending)
        # An unrelated heuristic's result must not affect our state.
        p = Point(np.zeros(self.problem.dim), "OtherHeuristic:abc")
        h.on_new_results([Result(p, 1.0)])
        assert h._pending == before_pending

    def test_initial_fill_does_not_record_success(self):
        h = self._seed(NP_init=5, seed=11)
        req_id = next(iter(h._pending))
        r, idx = self._result_for_pending(h, req_id, fx=42.0)
        h.on_new_results([r])

        # Slot is now filled and pending was consumed; success buffer empty.
        assert h.population[idx] is not None
        assert h.population[idx].fx == 42.0
        assert req_id not in h._pending
        assert h._success_F == []
        assert h._success_CR == []
        assert h._success_dF == []

    def test_better_trial_replaces_parent_and_records_success(self):
        h = self._seed(NP_init=5, seed=12)
        # Fill all slots first with mediocre points.
        all_ids = list(h._pending.keys())
        results = [self._result_for_pending(h, rid, fx=100.0)[0] for rid in all_ids]
        h.on_new_results(results)
        # Drain follow-up trials so we can drive a single targeted improvement.
        h.get_points(limit=100)

        # Pick a follow-up trial and feed back a much better fx.
        new_req_id = next(iter(h._pending))
        old_target_idx = h._pending[new_req_id][0]
        old_population_x = np.asarray(h.population[old_target_idx].x).copy()

        r, _ = self._result_for_pending(h, new_req_id, fx=1.0)
        h.on_new_results([r])

        # Trial wins → population updated, parent in archive, success recorded.
        assert h.population[old_target_idx].fx == 1.0
        assert any(np.allclose(a, old_population_x) for a in h.archive)
        assert len(h._success_F) >= 1
        assert len(h._success_CR) >= 1
        assert len(h._success_dF) >= 1
        # Improvement weight is positive.
        assert all(w > 0 for w in h._success_dF)

    def test_worse_trial_does_not_replace_parent(self):
        h = self._seed(NP_init=5, seed=13)
        # Fill slots with very good points.
        all_ids = list(h._pending.keys())
        results = [self._result_for_pending(h, rid, fx=0.001)[0] for rid in all_ids]
        h.on_new_results(results)
        h.get_points(limit=100)

        new_req_id = next(iter(h._pending))
        target_idx = h._pending[new_req_id][0]
        original_fx = h.population[target_idx].fx

        # Worse trial — must not replace parent.
        r, _ = self._result_for_pending(h, new_req_id, fx=999.0)
        h.on_new_results([r])

        assert h.population[target_idx].fx == original_fx
        assert h._success_F == []  # nothing recorded on failure

    def test_follow_up_trials_emitted_when_population_full(self):
        h = self._seed(NP_init=5, seed=14)
        # Fill slots; once at least 4 are filled, follow-up trials emit.
        all_ids = list(h._pending.keys())
        results = [self._result_for_pending(h, rid, fx=10.0 + i)[0] for i, rid in enumerate(all_ids)]
        h.on_new_results(results)

        emitted = h.get_points(limit=100)
        assert len(emitted) >= 1
        assert all(pt.who.startswith("LSHADE:") for pt in emitted)


# ----------------------------------------------------------------------
# Memory updates and archive maintenance
# ----------------------------------------------------------------------


class LSHADEAdaptationTests(_MockStrategyMixin, PanobbgoTestCase):
    def test_memory_updates_after_one_generation(self):
        """Once NP_current outcomes accumulate, M_F / M_CR shift from 0.5."""
        from panobbgo.heuristics.lshade import LSHADE
        from panobbgo.lib import Point, Result

        h = LSHADE(self.strategy, NP_init=4, NP_min=4, H=2, seed=21)
        h.on_start()
        # Fill the initial population with mediocre fitness.
        all_ids = list(h._pending.keys())
        for rid in all_ids:
            x = np.asarray(self.problem.random_point(), dtype=float)
            r = Result(Point(x, f"LSHADE:{rid}"), 100.0)
            h.on_new_results([r])
        h.get_points(limit=100)

        # Now drive 4 successful follow-up trials so the success buffer is
        # populated, then the memory update fires on the 4th outcome.
        for _ in range(4):
            rid = next(iter(h._pending))
            # Improving fx so each is a "success".
            x = np.asarray(self.problem.random_point(), dtype=float)
            r = Result(Point(x, f"LSHADE:{rid}"), 1.0)
            h.on_new_results([r])
            h.get_points(limit=100)

        # mem_pos advanced; at least one of M_F / M_CR may have moved off 0.5
        # (the fresh draw might happen to be near 0.5 — but mem_pos is the
        # cleanest invariant to verify).
        assert h._mem_pos == 1
        # Success buffer cleared after the memory update.
        assert h._success_F == []
        assert h._success_CR == []

    def test_archive_grows_with_replacements(self):
        """Replaced parents are pushed into the archive."""
        from panobbgo.heuristics.lshade import LSHADE
        from panobbgo.lib import Point, Result

        h = LSHADE(self.strategy, NP_init=5, archive_factor=1.0, seed=23)
        h.on_start()
        all_ids = list(h._pending.keys())
        for rid in all_ids:
            x = np.asarray(self.problem.random_point(), dtype=float)
            r = Result(Point(x, f"LSHADE:{rid}"), 100.0)
            h.on_new_results([r])
        h.get_points(limit=100)

        # Improve the slot at idx 0 by a follow-up trial.
        rid = next(iter(h._pending))
        x = np.asarray(self.problem.random_point(), dtype=float)
        r = Result(Point(x, f"LSHADE:{rid}"), 1.0)
        h.on_new_results([r])

        # Archive has at least one entry — the displaced parent.
        assert len(h.archive) >= 1
        # Archive is bounded by archive_factor * NP_current = 5.
        assert len(h.archive) <= 5

    def test_archive_capacity_enforced(self):
        """Archive never exceeds ``archive_factor · NP_current`` entries."""
        from panobbgo.heuristics.lshade import LSHADE

        h = LSHADE(self.strategy, NP_init=5, archive_factor=1.0)
        h.on_start()
        # Force-fill the archive past its cap and verify _archive_add caps it.
        cap = int(np.ceil(1.0 * h.NP_current))
        for _ in range(cap * 3):
            h._archive_add(np.zeros(self.problem.dim))
        assert len(h.archive) <= cap


# ----------------------------------------------------------------------
# Linear Population Size Reduction (LPSR)
# ----------------------------------------------------------------------


class LSHADELPSRTests(_MockStrategyMixin, PanobbgoTestCase):
    def test_lpsr_disabled_without_max_eval(self):
        from panobbgo.heuristics.lshade import LSHADE
        from panobbgo.lib import Point, Result

        # max_eval = 0 → LPSR is a no-op.
        self.strategy.config.max_eval = 0
        self.strategy.results = list(range(50))  # arbitrary "consumed" eval count

        h = LSHADE(self.strategy, NP_init=10, NP_min=4, seed=31)
        h.on_start()
        all_ids = list(h._pending.keys())
        for rid in all_ids:
            x = np.asarray(self.problem.random_point(), dtype=float)
            r = Result(Point(x, f"LSHADE:{rid}"), 100.0)
            h.on_new_results([r])
        # NP_current must stay at NP_init.
        assert h.NP_current == 10

    def test_lpsr_shrinks_population(self):
        """Once budget is partially consumed, LPSR shrinks NP_current."""
        from panobbgo.heuristics.lshade import LSHADE
        from panobbgo.lib import Point, Result

        # 10 initial → 4 minimum.  At 50% budget consumed → ~7.
        self.strategy.config.max_eval = 100
        self.strategy.results = list(range(50))  # simulate 50% consumed

        h = LSHADE(self.strategy, NP_init=10, NP_min=4, seed=33)
        h.on_start()
        # Fill the initial population.
        all_ids = list(h._pending.keys())
        for rid in all_ids:
            x = np.asarray(self.problem.random_point(), dtype=float)
            r = Result(Point(x, f"LSHADE:{rid}"), 100.0 + np.random.rand())
            h.on_new_results([r])
        # The 10th outcome triggers _maybe_shrink: NP_current ∈ [4, 10).
        assert 4 <= h.NP_current < 10
        # population length matches NP_current.
        assert len(h.population) == h.NP_current

    def test_lpsr_drops_worst_individuals(self):
        """When LPSR shrinks, the *worst* (highest-penalty) slots are removed."""
        from panobbgo.heuristics.lshade import LSHADE
        from panobbgo.lib import Point, Result

        self.strategy.config.max_eval = 100
        self.strategy.results = list(range(80))  # 80% budget used → tight shrink

        h = LSHADE(self.strategy, NP_init=10, NP_min=4, seed=35)
        h.on_start()
        # Inject distinct fx so the survivor identity is determined.
        all_ids = list(h._pending.keys())
        for i, rid in enumerate(all_ids):
            x = np.asarray(self.problem.random_point(), dtype=float)
            r = Result(Point(x, f"LSHADE:{rid}"), float(100 - i))  # last is best
            h.on_new_results([r])
        # Survivors should include the best fx values (smallest), not the
        # worst (largest).  Check by max fx in surviving population.
        surviving_fx = sorted(r.fx for r in h.population if r is not None)
        assert surviving_fx[-1] <= 100.0
        # The very worst (fx=100, the first injected) should be dropped.
        assert all(r.fx < 100.0 for r in h.population if r is not None)


# ----------------------------------------------------------------------
# Restart behaviour
# ----------------------------------------------------------------------


class LSHADERestartTests(_MockStrategyMixin, PanobbgoTestCase):
    def test_on_restart_clears_population(self):
        from panobbgo.heuristics.lshade import LSHADE
        from panobbgo.lib import Point, Result

        h = LSHADE(self.strategy, NP_init=6, seed=41)
        h.on_start()
        for rid in list(h._pending.keys()):
            x = np.asarray(self.problem.random_point(), dtype=float)
            r = Result(Point(x, f"LSHADE:{rid}"), 1.0)
            h.on_new_results([r])

        # Population should now be filled.
        assert any(r is not None for r in h.population)
        # Restart drops everything.
        h.on_restart(center=np.zeros(self.problem.dim), reason="test")
        assert all(r is None for r in h.population)
        assert h.archive == []
        assert np.all(h.M_F == 0.5)
        assert np.all(h.M_CR == 0.5)
        # And re-emits NP_init random trials — pending tracks these.
        assert len(h._pending) == h.NP_init
        emitted = h.get_points(limit=100)
        assert len(emitted) == h.NP_init

    def test_on_restart_before_start_is_noop(self):
        from panobbgo.heuristics.lshade import LSHADE

        h = LSHADE(self.strategy, NP_init=5)
        # Should not raise — population is empty so the restart is a no-op.
        h.on_restart(center=np.zeros(self.problem.dim), reason="never started")
        # No emit expected.
        assert h.get_points(limit=10) == []

    def test_on_restart_with_none_center(self):
        from panobbgo.heuristics.lshade import LSHADE
        from panobbgo.lib import Point, Result

        h = LSHADE(self.strategy, NP_init=5, seed=44)
        h.on_start()
        for rid in list(h._pending.keys()):
            x = np.asarray(self.problem.random_point(), dtype=float)
            r = Result(Point(x, f"LSHADE:{rid}"), 1.0)
            h.on_new_results([r])
        h.on_restart(center=None, reason="random fallback")
        # Emitted points stay inside the box.
        for pt in h.get_points(limit=100):
            assert np.all(pt.x >= self.problem.box[:, 0] - 1e-9)
            assert np.all(pt.x <= self.problem.box[:, 1] + 1e-9)


# ----------------------------------------------------------------------
# End-to-end smoke convergence
# ----------------------------------------------------------------------


class LSHADEConvergenceSmokeTests(_MockStrategyMixin, PanobbgoTestCase):
    """Drive L-SHADE synchronously against a quadratic and verify the
    best-seen value strictly improves.  This is a *smoke* test, not a
    hard convergence guarantee."""

    def test_lshade_drives_toward_origin_on_quadratic(self):
        from panobbgo.heuristics.lshade import LSHADE
        from panobbgo.lib import Result

        h = LSHADE(self.strategy, NP_init=10, NP_min=4, seed=2026)
        h.on_start()
        pts = h.get_points(limit=100)

        def f(x):  # quadratic, optimum at origin
            return float(np.sum(x * x))

        def evaluate_and_feed(points):
            results = [Result(pt, f(pt.x)) for pt in points]
            h.on_new_results(results)

        # Run several "generations" of drive-and-feed.
        evaluate_and_feed(pts)
        best_fx_history = []
        for _ in range(40):
            pts = h.get_points(limit=100)
            if not pts:
                break
            evaluate_and_feed(pts)
            current_best = min((r.fx for r in h.population if r is not None), default=float("inf"))
            best_fx_history.append(current_best)

        # The best-seen value should have improved.
        assert best_fx_history[-1] < best_fx_history[0]


# ----------------------------------------------------------------------
# Module-level registration
# ----------------------------------------------------------------------


def test_lshade_in_heuristics_init():
    """Top-level ``panobbgo.heuristics`` package exports ``LSHADE``."""
    from panobbgo import heuristics

    assert hasattr(heuristics, "LSHADE")
    assert "LSHADE" in heuristics.__all__


def test_lshade_in_default_structural_catalog():
    """The structural catalog includes LSHADE as an add_heuristic candidate."""
    from panobbgo.self_improve import default_structural_catalog
    from panobbgo.heuristics.lshade import LSHADE

    catalog = default_structural_catalog()
    add_rules = [r for r in catalog.rules if getattr(r, "op", None) == "add_heuristic"]
    assert add_rules, "structural catalog must have at least one add_heuristic rule"
    candidate_classes = {c for r in add_rules for c, _ in r.candidate_classes}
    assert LSHADE in candidate_classes


def test_lshade_kwargs_in_default_catalog():
    """The kwarg catalog has rules to tune NP_init and p_best_rate."""
    from panobbgo.self_improve import default_catalog

    catalog = default_catalog()
    lshade_rules = [r for r in catalog.rules if getattr(r, "class_name", None) == "LSHADE"]
    param_names = {r.param_name for r in lshade_rules}
    assert "NP_init" in param_names
    assert "p_best_rate" in param_names
