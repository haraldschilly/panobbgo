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
    semantics from L-SHADE, so the mock strategy needs the same
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
        """Paper / reference-code defaults (``nlshade-original.cpp``).

        Updated 2026-09 for the fidelity pass: the port used to inherit jSO's
        ``H = 5``, ``p_best 0.25 → 0.125``, ``archive_factor = 1``, the
        ``"jso"`` F cap and a linear ``k_rank = 3`` weight on ``r1``.
        """
        from panobbgo.heuristics.nl_shade_rsp import NLSHADE_RSP

        h = NLSHADE_RSP(self.strategy)
        # ``NP_init`` defaults to ``"auto"``: 3*dim = 6 at dim=2 / 1000 evals.
        assert h.NP_init == 6
        assert h.NP_min == 4
        assert h.H == 20 * self.problem.dim
        assert h.p_best == 0.2
        assert h.p_best_end == 0.4
        assert h.archive_factor == 2.1
        assert h.k_rank == 1.0
        assert h.p_archive == 0.5
        assert h.name == "NLSHADE_RSP"
        assert h.F_schedule is None  # no jSO F cap
        np.testing.assert_array_equal(h._M_F, 0.2)
        np.testing.assert_array_equal(h._M_CR, 0.2)

    def test_custom_construction(self):
        from panobbgo.heuristics.nl_shade_rsp import NLSHADE_RSP

        h = NLSHADE_RSP(
            self.strategy,
            NP_init=20,
            NP_min=6,
            H=4,
            p_best=0.3,
            p_best_end=0.1,
            archive_factor=2.0,
            k_rank=1.5,
            seed=7,
            name="MyRSP",
        )
        assert h.NP_init == 20
        assert h.NP_min == 6
        assert h.H == 4
        assert h.p_best == 0.3
        assert h.p_best_end == 0.1
        assert h.archive_factor == 2.0
        assert h.k_rank == 1.5
        assert h.name == "MyRSP"

    def test_subclass_of_lshade_not_jso(self):
        """NL-SHADE-RSP is an L-SHADE descendant, not a jSO (no F_w, F cap, anchor bin, ...)."""
        from panobbgo.heuristics.jso import JSO
        from panobbgo.heuristics.lshade import LSHADE
        from panobbgo.heuristics.nl_shade_rsp import NLSHADE_RSP

        h = NLSHADE_RSP(self.strategy)
        assert isinstance(h, LSHADE)
        assert not isinstance(h, JSO)
        assert h._current_F_weight() == 1.0

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

    def test_inherits_lshade_validation(self):
        from panobbgo.heuristics.nl_shade_rsp import NLSHADE_RSP

        with pytest.raises(ValueError, match="H must be >= 1"):
            NLSHADE_RSP(self.strategy, H=0)
        with pytest.raises(ValueError, match="p_best_end"):
            NLSHADE_RSP(self.strategy, p_best_end=1.5)


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
# Rank-based Selective Pressure (RSP) — on r2, not r1
# ----------------------------------------------------------------------


class NLSHADERSPRankSelectionTests(_MockStrategyMixin, PanobbgoTestCase):
    def _populate(self, h, fxs):
        h._population = [
            _build_result(self.strategy, self.problem.random_point(), fx, f"x{i}") for i, fx in enumerate(fxs)
        ]
        h._NP_current = len(fxs)
        return h._live_indices()

    def _sorted(self, h, live):
        return sorted(live, key=lambda i: h._rank_of(h._population[i]))

    def test_r1_is_uniform_and_distinct(self):
        """Regression (paper §III / reference code): ``r1`` is *uniform*, not rank-weighted.

        The old port rank-selected ``r1`` with ``k·(n−i)/n + 1``, so the best
        slot was drawn ~2× as often as the worst.
        """
        from panobbgo.heuristics.nl_shade_rsp import NLSHADE_RSP

        h = NLSHADE_RSP(self.strategy, seed=7)
        live = self._populate(h, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        counts = {i: 0 for i in live}
        for _ in range(8000):
            r1 = h._select_r1(live, target_idx=5, pbest_idx=0)
            counts[r1] += 1
        assert counts[5] == 0 and counts[0] == 0  # target and pbest excluded
        for i in (1, 2, 3, 4):  # the rest uniform: ~2000 each
            assert 1700 < counts[i] < 2300

    def test_returns_none_when_pool_empty(self):
        from panobbgo.heuristics.nl_shade_rsp import NLSHADE_RSP

        h = NLSHADE_RSP(self.strategy, seed=0)
        live = self._populate(h, [1.0])  # only the target slot is live
        assert h._select_r1(live, target_idx=0) is None

    def test_rank_weights_are_exponential(self):
        """``R_i = exp(−k·i/NP)`` (i = 0 best): RSP ``k = 1``, LBC ``k = 4``."""
        from panobbgo.heuristics.nl_shade_lbc import NLSHADE_LBC
        from panobbgo.heuristics.nl_shade_rsp import NLSHADE_RSP

        rsp = NLSHADE_RSP(self.strategy)
        np.testing.assert_allclose(rsp._rank_weights(5), np.exp(-np.arange(5) / 5.0))
        lbc = NLSHADE_LBC(self.strategy)
        np.testing.assert_allclose(lbc._rank_weights(5), np.exp(-4.0 * np.arange(5) / 5.0))

    def test_r2_from_population_is_rank_weighted(self):
        """Population draws of ``r2`` follow ``exp(−i/NP)`` over the ranks, excluding target/pbest/r1."""
        from panobbgo.heuristics.nl_shade_rsp import NLSHADE_RSP

        h = NLSHADE_RSP(self.strategy, k_rank=1.0, seed=11)
        live = self._populate(h, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        srt = self._sorted(h, live)
        h._archive = []  # population only
        n_draws = 20000
        counts = {i: 0 for i in live}
        for _ in range(n_draws):
            x, from_archive = h._select_r2(live, srt, target_idx=6, r1=1, pbest_idx=0)
            assert not from_archive
            idx = next(i for i in live if np.array_equal(h._population[i].x, x))
            counts[idx] += 1
        assert counts[6] == counts[1] == counts[0] == 0
        w = np.exp(-np.arange(7) / 7.0)[[2, 3, 4, 5]]
        expected = w / w.sum() * n_draws
        for pos, i in enumerate((2, 3, 4, 5)):
            assert abs(counts[i] - expected[pos]) < 0.05 * n_draws

    def test_k_rank_zero_is_uniform(self):
        from panobbgo.heuristics.nl_shade_rsp import NLSHADE_RSP

        h = NLSHADE_RSP(self.strategy, k_rank=0.0, seed=7)
        live = self._populate(h, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        srt = self._sorted(h, live)
        counts = {i: 0 for i in live}
        for _ in range(8000):
            x, _ = h._select_r2(live, srt, target_idx=0, r1=1, pbest_idx=None)
            counts[next(i for i in live if np.array_equal(h._population[i].x, x))] += 1
        for i in (2, 3, 4, 5):
            assert 1700 < counts[i] < 2300


# ----------------------------------------------------------------------
# Archive: size 2.1·NP, adaptive usage probability p_A
# ----------------------------------------------------------------------


class NLSHADERSPArchiveTests(_MockStrategyMixin, PanobbgoTestCase):
    def _populate(self, h, n):
        h._population = [_build_result(self.strategy, self.problem.random_point(), float(i), f"x{i}") for i in range(n)]
        h._NP_current = n
        live = h._live_indices()
        return live, sorted(live, key=lambda i: h._rank_of(h._population[i]))

    def test_archive_size_is_2_1_NP_and_no_random_cap(self):
        """Regression: ``N_A = ⌊2.1 · NP⌋`` (at least NP_min), not a random cap in ``[0, NP]``."""
        from panobbgo.heuristics.nl_shade_rsp import NLSHADE_RSP

        h = NLSHADE_RSP(self.strategy, seed=3)
        h._NP_current = 30
        assert {h._archive_cap() for _ in range(50)} == {63}
        h._NP_current = 10
        assert h._archive_cap() == 21
        h._NP_current = 1
        assert h._archive_cap() == h.NP_min
        assert NLSHADE_RSP(self.strategy, archive_factor=0.0)._archive_cap() == 0

    def test_full_archive_replaces_a_random_entry(self):
        from panobbgo.heuristics.nl_shade_rsp import NLSHADE_RSP

        h = NLSHADE_RSP(self.strategy, archive_factor=1.0, NP_min=4, seed=5)
        h._NP_current = 4
        for i in range(10):
            h._archive_insert(_build_result(self.strategy, [float(i), 0.0], 1.0, "p"))
        assert len(h._archive) == 4
        assert any(a[0] >= 4.0 for a in h._archive)  # later parents got in

    def test_r2_archive_probability(self):
        """Regression: ``r2`` comes from the archive with probability ``p_A`` (was uniform over P ∪ A)."""
        from panobbgo.heuristics.nl_shade_rsp import NLSHADE_RSP

        h = NLSHADE_RSP(self.strategy, seed=19)
        live, srt = self._populate(h, 6)
        # A huge archive: under the old uniform-over-union rule nearly every draw hit it.
        h._archive = [np.array([9.0, 9.0]) for _ in range(500)]
        for p_a, lo, hi in ((0.5, 0.45, 0.55), (0.1, 0.07, 0.13), (0.9, 0.87, 0.93)):
            h.p_archive = p_a
            hits = sum(h._select_r2(live, srt, target_idx=0, r1=1, pbest_idx=2)[1] for _ in range(4000))
            assert lo < hits / 4000 < hi
        h._archive = []
        assert not h._select_r2(live, srt, target_idx=0, r1=1, pbest_idx=2)[1]

    def test_p_archive_update(self):
        """``p_A = (Δ_A/n_A) / (Δ_A/n_A + Δ_P/(n − n_A))`` clipped to [0.1, 0.9]; 0.5 without archive success."""
        from panobbgo.heuristics.lshade import _TrialMeta
        from panobbgo.heuristics.nl_shade_rsp import NLSHADE_RSP

        h = NLSHADE_RSP(self.strategy, seed=0)

        def success(from_archive, delta):
            m = _TrialMeta(0, 0.5, 0.5)
            m.from_archive = from_archive
            h._record_success(m, delta)

        success(True, 3.0)
        success(True, 1.0)
        success(False, 2.0)
        h._update_p_archive(n_trials=6)  # A = 4/2 = 2, P = 2/4 = 0.5
        assert h.p_archive == pytest.approx(2.0 / 2.5)

        h._arch_delta, h._arch_n, h._pop_delta = 10.0, 1, 0.0
        h._update_p_archive(n_trials=5)
        assert h.p_archive == pytest.approx(0.9)  # clipped
        h._arch_delta, h._arch_n, h._pop_delta = 0.1, 1, 100.0
        h._update_p_archive(n_trials=5)
        assert h.p_archive == pytest.approx(0.1)  # clipped
        h._arch_delta, h._arch_n, h._pop_delta = 0.0, 0, 5.0
        h._update_p_archive(n_trials=5)
        assert h.p_archive == 0.5  # no archive success

    def test_end_of_generation_updates_p_archive_and_resets(self):
        from panobbgo.heuristics.lshade import _TrialMeta
        from panobbgo.heuristics.nl_shade_rsp import NLSHADE_RSP

        h = NLSHADE_RSP(self.strategy, seed=9)
        h._population = []
        h._NP_current = 4
        m = _TrialMeta(0, 0.5, 0.5)
        m.from_archive = True
        h._record_success(m, 1.0)
        h._gen_completed = 4
        h._cross_exponential = True
        h._end_of_generation()
        assert h.p_archive == pytest.approx(0.9)  # only the archive trial improved
        assert (h._arch_delta, h._arch_n, h._pop_delta) == (0.0, 0, 0.0)
        assert h._cross_exponential is None  # a fresh coin next generation


class NLSHADERSPParameterTests(_MockStrategyMixin, PanobbgoTestCase):
    def _populate(self, h, n):
        h._population = [_build_result(self.strategy, self.problem.random_point(), float(i), f"x{i}") for i in range(n)]
        h._NP_current = n
        live = h._live_indices()
        return live, sorted(live, key=lambda i: h._rank_of(h._population[i]))

    def test_pbest_rises_0_2_to_0_4_and_excludes_target(self):
        """Regression (reference code): ``psize = max(2, NP·(0.2 + 0.2·r))`` rises; pbest ≠ target.

        The old port inherited jSO's falling ``0.25 → 0.125``.
        """
        from panobbgo.heuristics.nl_shade_rsp import NLSHADE_RSP

        h = NLSHADE_RSP(self.strategy, seed=0)
        self.strategy.config.max_eval = 100
        self.strategy.results = []
        assert h._pbest_count(20) == 4
        self.strategy.results = list(range(50))
        assert h._pbest_count(20) == 6
        self.strategy.results = list(range(100))
        assert h._pbest_count(20) == 8
        assert h._pbest_count(4) == 2  # floor of 2
        live, srt = self._populate(h, 10)
        assert srt[0] not in h._pbest_pool(srt, target_idx=srt[0])

    def test_memory_init_and_plain_lehmer_update(self):
        """Regression: M_F = M_CR = 0.2 initially; plain weighted Lehmer mean, no averaging,
        no anchor, no terminal CR; a generation without success resets the bin to 0.5."""
        from panobbgo.heuristics.nl_shade_rsp import NLSHADE_RSP

        h = NLSHADE_RSP(self.strategy, H=4, seed=0)
        np.testing.assert_array_equal(h._M_F, [0.2] * 4)
        h._success_F = [0.2, 0.4, 0.8]
        h._success_CR = [0.0, 0.0, 0.0]
        h._success_delta = [1.0, 1.0, 1.0]
        h._update_memory()
        assert h._M_F[0] == pytest.approx((0.04 + 0.16 + 0.64) / 1.4)
        assert h._M_CR[0] == 0.5  # all-zero CR: Lehmer denominator 0 → 0.5, not the −1 sentinel
        # every bin is writable (no anchor)
        for _ in range(3):
            h._success_F, h._success_CR, h._success_delta = [0.9], [0.9], [1.0]
            h._update_memory()
        assert h._mem_ptr == 0
        np.testing.assert_allclose(h._M_F[1:], 0.9)
        # no success: current bin reset to 0.5 / 0.5, pointer stays
        h._success_F, h._success_CR, h._success_delta = [], [], []
        h._update_memory()
        assert (h._M_F[0], h._M_CR[0], h._mem_ptr) == (0.5, 0.5, 0)

    def test_no_F_cap(self):
        """Regression: RSP does not inherit jSO's ``F ≤ 0.7`` cap."""
        from panobbgo.heuristics.nl_shade_rsp import NLSHADE_RSP

        h = NLSHADE_RSP(self.strategy, seed=1)
        self.strategy.results = []
        h._M_F[:] = 0.95
        assert max(h._sample_F() for _ in range(300)) > 0.9

    def test_CR_is_rank_order_statistic(self):
        """Smaller sampled ``CR`` go to better individuals (sorted hand-out)."""
        from panobbgo.heuristics.nl_shade_rsp import NLSHADE_RSP

        h = NLSHADE_RSP(self.strategy, seed=4)
        h._M_CR[:] = 0.5
        best = np.mean([h._sample_CR_for_rank(0, 20) for _ in range(300)])
        worst = np.mean([h._sample_CR_for_rank(19, 20) for _ in range(300)])
        assert best < 0.4 < 0.6 < worst

    def test_binomial_CR_schedule(self):
        """``CR_b = 0`` in the first half, ``2(r − 0.5)`` after (the sampled CR is not used)."""
        from panobbgo.heuristics.nl_shade_rsp import NLSHADE_RSP

        h = NLSHADE_RSP(self.strategy, seed=0)
        self.strategy.config.max_eval = 100
        self.strategy.results = list(range(40))
        assert h._binomial_CR() == 0.0
        self.strategy.results = list(range(75))
        assert h._binomial_CR() == pytest.approx(0.5)
        h._cross_exponential = False
        self.strategy.results = list(range(10))
        x = np.zeros(2)
        v = np.ones(2)
        for _ in range(50):  # CR_b = 0: exactly the forced component, whatever CR says
            assert np.sum(h._crossover(v, x, 1.0) == 1.0) == 1

    def test_exponential_crossover_contiguous(self):
        """Exponential crossover copies one contiguous run from a random start (no wrap-around)."""
        from unittest import mock

        from panobbgo.heuristics.nl_shade_rsp import NLSHADE_RSP

        h = NLSHADE_RSP(self.strategy, seed=2)
        x = np.zeros(8)
        v = np.ones(8)
        fake = mock.MagicMock(dim=8)
        with mock.patch.object(NLSHADE_RSP, "problem", new_callable=mock.PropertyMock, return_value=fake):
            lengths = []
            for _ in range(300):
                idx = np.flatnonzero(h._exponential_crossover(v, x, 0.7) == 1.0)
                assert len(idx) >= 1
                assert np.all(np.diff(idx) == 1)
                lengths.append(len(idx))
            for _ in range(20):
                assert np.sum(h._exponential_crossover(v, x, 0.0)) == 1.0  # CR = 0: just the start
        assert max(lengths) > 2

    def test_crossover_type_fixed_per_generation(self):
        from panobbgo.heuristics.nl_shade_rsp import NLSHADE_RSP

        h = NLSHADE_RSP(self.strategy, seed=6)
        seen = set()
        for _ in range(40):
            h._cross_exponential = None
            h._crossover(np.ones(2), np.zeros(2), 0.5)
            first = h._cross_exponential
            for _ in range(5):
                h._crossover(np.ones(2), np.zeros(2), 0.5)
                assert h._cross_exponential is first
            seen.add(first)
        assert seen == {True, False}

    def test_bounds_resampled_uniformly(self):
        """Regression: out-of-box components are resampled in the box (was midpoint)."""
        from panobbgo.heuristics.nl_shade_rsp import NLSHADE_RSP

        h = NLSHADE_RSP(self.strategy, seed=8)
        lb, ub = h.problem.box[:, 0], h.problem.box[:, 1]
        x = lb.copy()  # midpoint repair would give exactly lb for u < lb
        u = lb - 5.0
        outs = np.array([h._repair_bounds(u, x) for _ in range(200)])
        assert np.all(outs >= lb) and np.all(outs <= ub)
        assert outs[:, 0].std() > 0.1 * (ub[0] - lb[0])


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

    def test_on_start_resets_archive_probability(self):
        from panobbgo.heuristics.nl_shade_rsp import NLSHADE_RSP

        h = NLSHADE_RSP(self.strategy, NP_init=4, seed=0)
        h.p_archive = 0.9
        h._cross_exponential = True
        h.on_start()
        assert h.p_archive == 0.5
        assert h._cross_exponential is None

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

    def test_restart_resets_archive_probability_and_memory(self):
        from panobbgo.heuristics.nl_shade_rsp import NLSHADE_RSP

        h = NLSHADE_RSP(self.strategy, NP_init=6, seed=2)
        h.on_start()
        h.get_points(limit=100)
        h.p_archive = 0.1
        h._M_F[:] = 0.9
        h._archive.append(np.array([0.5, 0.5]))
        h.on_restart(np.array([0.0, 0.0]), reason="test")
        assert h.p_archive == 0.5
        np.testing.assert_array_equal(h._M_F, 0.2)
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
