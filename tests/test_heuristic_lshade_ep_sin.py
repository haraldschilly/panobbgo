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

"""Tests for the LSHADE-EpSin (Awad, Ali, Suganthan, CEC 2016) heuristic."""

from __future__ import annotations

import numpy as np
import pytest

from panobbgo.utils import PanobbgoTestCase


class _MockStrategyMixin:
    """Same setup as the L-SHADE / jSO / NL-SHADE-RSP tests.

    EpSin inherits ``max_eval``-based progress pacing from L-SHADE, so the
    mock strategy needs ``config.max_eval`` and a results list.  Saved /
    restored to prevent cross-test bleed.
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


class LSHADEEpSinConstructionTests(_MockStrategyMixin, PanobbgoTestCase):
    def test_default_construction(self):
        from panobbgo.heuristics.lshade_ep_sin import LSHADE_EpSin, _DEFAULT_MU_FREQ, _DEFAULT_PS

        h = LSHADE_EpSin(self.strategy)
        assert h.NP_init == 30
        assert h.NP_min == 4
        assert h.H == 6  # inherited L-SHADE default
        assert h.p_best == 0.11
        assert h.archive_factor == 1.0
        assert h.mu_freq_init == _DEFAULT_MU_FREQ == 0.5
        assert h.name == "LSHADE_EpSin"
        assert h._mu_freq == 0.5
        assert h._p_s == _DEFAULT_PS == 0.5
        assert h._ns1 == 0
        assert h._ns2 == 0
        assert h._gen_count == 0
        assert h._gen_success_freq == []
        assert h._last_sin == (0, 0.0)
        # F_schedule is intentionally off by default.
        assert h.F_schedule is None

    def test_custom_construction(self):
        from panobbgo.heuristics.lshade_ep_sin import LSHADE_EpSin

        h = LSHADE_EpSin(
            self.strategy,
            NP_init=20,
            NP_min=6,
            H=4,
            p_best=0.2,
            archive_factor=2.0,
            mu_freq_init=0.3,
            seed=7,
            name="MyEpSin",
        )
        assert h.NP_init == 20
        assert h.NP_min == 6
        assert h.H == 4
        assert h.p_best == 0.2
        assert h.archive_factor == 2.0
        assert h.mu_freq_init == 0.3
        assert h._mu_freq == 0.3
        assert h.name == "MyEpSin"

    def test_subclass_of_lshade(self):
        from panobbgo.heuristics.lshade import LSHADE
        from panobbgo.heuristics.lshade_ep_sin import LSHADE_EpSin

        h = LSHADE_EpSin(self.strategy)
        assert isinstance(h, LSHADE)

    def test_invalid_mu_freq_init(self):
        from panobbgo.heuristics.lshade_ep_sin import LSHADE_EpSin

        with pytest.raises(ValueError, match="mu_freq_init"):
            LSHADE_EpSin(self.strategy, mu_freq_init=0.0)
        with pytest.raises(ValueError, match="mu_freq_init"):
            LSHADE_EpSin(self.strategy, mu_freq_init=-0.1)
        with pytest.raises(ValueError, match="mu_freq_init"):
            LSHADE_EpSin(self.strategy, mu_freq_init=1.5)
        with pytest.raises(ValueError, match="mu_freq_init"):
            LSHADE_EpSin(self.strategy, mu_freq_init=float("nan"))
        with pytest.raises(ValueError, match="mu_freq_init"):
            LSHADE_EpSin(self.strategy, mu_freq_init=float("inf"))

    def test_inherits_lshade_validation(self):
        from panobbgo.heuristics.lshade_ep_sin import LSHADE_EpSin

        with pytest.raises(ValueError, match="NP_init"):
            LSHADE_EpSin(self.strategy, NP_init=3)
        with pytest.raises(ValueError, match="NP_min"):
            LSHADE_EpSin(self.strategy, NP_min=3)
        with pytest.raises(ValueError, match="p_best"):
            LSHADE_EpSin(self.strategy, p_best=0.0)


# ----------------------------------------------------------------------
# Phase split: sinusoidal vs Cauchy-from-memory
# ----------------------------------------------------------------------


class LSHADEEpSinPhaseTests(_MockStrategyMixin, PanobbgoTestCase):
    def test_phase_split_at_half_budget(self):
        from panobbgo.heuristics.lshade_ep_sin import LSHADE_EpSin

        h = LSHADE_EpSin(self.strategy)
        # progress = 0.0 -> sinusoidal phase
        self.strategy.results = []
        assert h._in_sinusoidal_phase()
        # progress = 0.49 -> sinusoidal phase
        self.strategy.results = list(range(490))
        assert h._in_sinusoidal_phase()
        # progress = 0.5 -> Cauchy phase (>=, not <)
        self.strategy.results = list(range(500))
        assert not h._in_sinusoidal_phase()
        # progress = 0.99 -> Cauchy phase
        self.strategy.results = list(range(990))
        assert not h._in_sinusoidal_phase()

    def test_no_budget_falls_back_to_sinusoidal(self):
        from panobbgo.heuristics.lshade_ep_sin import LSHADE_EpSin

        h = LSHADE_EpSin(self.strategy)
        self.strategy.config.max_eval = 0  # unknown budget
        assert h._in_sinusoidal_phase()

    def test_gmax_estimate(self):
        from panobbgo.heuristics.lshade_ep_sin import LSHADE_EpSin

        h = LSHADE_EpSin(self.strategy, NP_init=30, NP_min=4)
        self.strategy.config.max_eval = 1000
        # G_max ~= 1000 / 17 (avg NP) ≈ 58.8
        gmax = h._estimate_gmax()
        assert 50.0 < gmax < 70.0

    def test_gmax_fallback_without_budget(self):
        from panobbgo.heuristics.lshade_ep_sin import LSHADE_EpSin

        h = LSHADE_EpSin(self.strategy, NP_init=20)
        self.strategy.config.max_eval = 0  # unknown budget
        gmax = h._estimate_gmax()
        assert gmax == 20 * 10  # _GMAX_FALLBACK_FACTOR=10


# ----------------------------------------------------------------------
# Sinusoidal F sampling
# ----------------------------------------------------------------------


class LSHADEEpSinSamplerTests(_MockStrategyMixin, PanobbgoTestCase):
    def test_sinusoid1_F_in_unit_interval(self):
        from panobbgo.heuristics.lshade_ep_sin import LSHADE_EpSin

        h = LSHADE_EpSin(self.strategy, seed=0)
        gmax = 100.0
        for g in range(1, 100):
            F = h._sinusoid1_F(float(g), gmax)
            assert 0.0 <= F <= 1.0

    def test_sinusoid2_F_in_unit_interval(self):
        from panobbgo.heuristics.lshade_ep_sin import LSHADE_EpSin

        h = LSHADE_EpSin(self.strategy, seed=0)
        gmax = 100.0
        for g in range(1, 100):
            F = h._sinusoid2_F(float(g), gmax, freq=0.5)
            assert 0.0 <= F <= 1.0

    def test_sinusoid1_envelope_decreasing(self):
        """Sinusoid 1's amplitude envelope is ``(G_max - g)/G_max`` — monotone non-increasing."""
        from panobbgo.heuristics.lshade_ep_sin import LSHADE_EpSin

        h = LSHADE_EpSin(self.strategy)
        gmax = 100.0
        # At g=0 envelope is 1; at g=100 envelope is 0 ⇒ F=0.5 always.
        F0 = h._sinusoid1_F(0.0, gmax)  # sin(0)=0, so F = 0.5
        F_at_gmax = h._sinusoid1_F(gmax, gmax)  # envelope=0 ⇒ F=0.5
        assert F0 == pytest.approx(0.5)
        assert F_at_gmax == pytest.approx(0.5)

    def test_sinusoid2_envelope_increasing(self):
        """Sinusoid 2's amplitude envelope is ``g/G_max`` — monotone non-decreasing."""
        from panobbgo.heuristics.lshade_ep_sin import LSHADE_EpSin

        h = LSHADE_EpSin(self.strategy)
        gmax = 100.0
        # At g=0 envelope is 0 ⇒ F=0.5 always.  At larger g envelope grows.
        F0 = h._sinusoid2_F(0.0, gmax, freq=0.5)
        # sin(π) = 0, so F=0.5 regardless of envelope; force a non-zero phase.
        F0_phase = h._sinusoid2_F(0.0, gmax, freq=0.25)
        assert F0 == pytest.approx(0.5)
        # At g=0, envelope=0 so F is always 0.5 even with different freq.
        assert F0_phase == pytest.approx(0.5)

    def test_sample_freq_cauchy_in_unit_interval(self):
        from panobbgo.heuristics.lshade_ep_sin import LSHADE_EpSin

        h = LSHADE_EpSin(self.strategy, seed=42)
        for _ in range(500):
            f = h._sample_freq_cauchy()
            assert 0.0 < f <= 1.0

    def test_sample_F_sinusoidal_returns_valid_choices(self):
        from panobbgo.heuristics.lshade_ep_sin import LSHADE_EpSin

        h = LSHADE_EpSin(self.strategy, seed=3)
        for _ in range(50):
            F, sin_choice, freq = h._sample_F_sinusoidal()
            assert 0.0 <= F <= 1.0
            assert sin_choice in (1, 2)
            assert 0.0 < freq <= 1.0

    def test_sample_F_CR_routes_by_phase(self):
        """First-half ⇒ sinusoid; second-half ⇒ Cauchy-from-memory."""
        from panobbgo.heuristics.lshade_ep_sin import LSHADE_EpSin

        h = LSHADE_EpSin(self.strategy, seed=11)
        # First half: progress=0
        self.strategy.results = []
        F, CR = h._sample_F_CR()
        assert 0.0 <= F <= 1.0
        assert 0.0 <= CR <= 1.0
        assert h._last_sin[0] in (1, 2)

        # Second half: progress=0.7
        self.strategy.results = list(range(700))
        F, CR = h._sample_F_CR()
        assert 0.0 <= F <= 1.0
        assert 0.0 <= CR <= 1.0
        assert h._last_sin == (0, 0.0)

    def test_p_s_default_balances_sinusoid_selection(self):
        """Initial ``p_s=0.5`` ⇒ Sinusoid 1 and 2 picked roughly equally."""
        from panobbgo.heuristics.lshade_ep_sin import LSHADE_EpSin

        h = LSHADE_EpSin(self.strategy, seed=21)
        counts = {1: 0, 2: 0}
        for _ in range(2000):
            _, sin_choice, _ = h._sample_F_sinusoidal()
            counts[sin_choice] += 1
        # Each ~50%, allow generous slack.
        assert 700 < counts[1] < 1300
        assert 700 < counts[2] < 1300


# ----------------------------------------------------------------------
# Ensemble bandit update (p_s + μ_freq)
# ----------------------------------------------------------------------


class LSHADEEpSinEnsembleUpdateTests(_MockStrategyMixin, PanobbgoTestCase):
    def test_p_s_cold_start_is_half(self):
        from panobbgo.heuristics.lshade_ep_sin import LSHADE_EpSin

        h = LSHADE_EpSin(self.strategy)
        # No successes recorded — Laplace smoothing keeps p_s at 0.5.
        h._update_ensemble_state()
        assert h._p_s == 0.5

    def test_p_s_biases_toward_winning_sinusoid(self):
        from panobbgo.heuristics.lshade_ep_sin import LSHADE_EpSin

        h = LSHADE_EpSin(self.strategy)
        # Sinusoid 1 dominated successes ⇒ p_s should rise above 0.5.
        h._ns1 = 8
        h._ns2 = 2
        h._update_ensemble_state()
        assert h._p_s == pytest.approx((8 + 1) / (8 + 2 + 2))
        assert h._p_s > 0.5

        # Reset: Sinusoid 2 dominated successes ⇒ p_s should drop below 0.5.
        h._ns1 = 2
        h._ns2 = 8
        h._update_ensemble_state()
        assert h._p_s == pytest.approx((2 + 1) / (2 + 8 + 2))
        assert h._p_s < 0.5

    def test_p_s_stays_in_unit_interval(self):
        """All-success-on-one-sinusoid still keeps p_s strictly inside (0, 1)."""
        from panobbgo.heuristics.lshade_ep_sin import LSHADE_EpSin

        h = LSHADE_EpSin(self.strategy)
        h._ns1 = 100
        h._ns2 = 0
        h._update_ensemble_state()
        assert 0.0 < h._p_s < 1.0

        h._ns1 = 0
        h._ns2 = 100
        h._update_ensemble_state()
        assert 0.0 < h._p_s < 1.0

    def test_mu_freq_lehmer_update(self):
        from panobbgo.heuristics.lshade_ep_sin import LSHADE_EpSin

        h = LSHADE_EpSin(self.strategy)
        # All success freqs were 0.3 ⇒ Lehmer mean = 0.3.
        h._gen_success_freq = [0.3, 0.3, 0.3]
        h._update_ensemble_state()
        assert h._mu_freq == pytest.approx(0.3)

        # Reset and feed mixed values.  Lehmer = (0.1² + 0.5² + 0.3²)/(0.1+0.5+0.3)
        h._gen_success_freq = [0.1, 0.5, 0.3]
        h._update_ensemble_state()
        expected = (0.01 + 0.25 + 0.09) / (0.1 + 0.5 + 0.3)
        assert h._mu_freq == pytest.approx(expected)

    def test_mu_freq_untouched_without_sin2_successes(self):
        from panobbgo.heuristics.lshade_ep_sin import LSHADE_EpSin

        h = LSHADE_EpSin(self.strategy, mu_freq_init=0.7)
        h._mu_freq = 0.7
        # Only Sinusoid 1 succeeded — no Sinusoid 2 freqs to average.
        h._ns1 = 5
        h._ns2 = 0
        h._gen_success_freq = []
        h._update_ensemble_state()
        assert h._mu_freq == 0.7

    def test_counters_cleared_after_update(self):
        from panobbgo.heuristics.lshade_ep_sin import LSHADE_EpSin

        h = LSHADE_EpSin(self.strategy)
        h._ns1 = 3
        h._ns2 = 4
        h._gen_success_freq = [0.5, 0.6]
        h._update_ensemble_state()
        assert h._ns1 == 0
        assert h._ns2 == 0
        assert h._gen_success_freq == []

    def test_end_of_generation_bumps_gen_count(self):
        from panobbgo.heuristics.lshade_ep_sin import LSHADE_EpSin

        h = LSHADE_EpSin(self.strategy)
        assert h._gen_count == 0
        h._end_of_generation()
        assert h._gen_count == 1
        h._end_of_generation()
        assert h._gen_count == 2


# ----------------------------------------------------------------------
# Trial meta + success recording
# ----------------------------------------------------------------------


class LSHADEEpSinTrialMetaTests(_MockStrategyMixin, PanobbgoTestCase):
    def test_make_trial_meta_carries_sin_choice(self):
        from panobbgo.heuristics.lshade_ep_sin import _EpSinTrialMeta, LSHADE_EpSin

        h = LSHADE_EpSin(self.strategy)
        h._last_sin = (1, 0.5)
        meta = h._make_trial_meta(slot_idx=3, F=0.4, CR=0.7)
        assert isinstance(meta, _EpSinTrialMeta)
        assert meta.slot_idx == 3
        assert meta.F == 0.4
        assert meta.CR == 0.7
        assert meta.sin_choice == 1
        assert meta.freq == 0.5
        # Sticky reset: _last_sin must be cleared after attachment.
        assert h._last_sin == (0, 0.0)

    def test_make_trial_meta_handles_no_sin_choice(self):
        from panobbgo.heuristics.lshade_ep_sin import LSHADE_EpSin

        h = LSHADE_EpSin(self.strategy)
        meta = h._make_trial_meta(slot_idx=1, F=0.5, CR=0.6)
        assert meta.sin_choice == 0
        assert meta.freq == 0.0

    def test_record_success_increments_sin1(self):
        from panobbgo.heuristics.lshade_ep_sin import _EpSinTrialMeta, LSHADE_EpSin

        h = LSHADE_EpSin(self.strategy)
        meta = _EpSinTrialMeta(slot_idx=0, F=0.4, CR=0.7, sin_choice=1, freq=0.5)
        h._record_success(meta, delta=1.0)
        assert h._ns1 == 1
        assert h._ns2 == 0
        assert h._gen_success_freq == []

    def test_record_success_increments_sin2_with_freq(self):
        from panobbgo.heuristics.lshade_ep_sin import _EpSinTrialMeta, LSHADE_EpSin

        h = LSHADE_EpSin(self.strategy)
        meta = _EpSinTrialMeta(slot_idx=0, F=0.4, CR=0.7, sin_choice=2, freq=0.7)
        h._record_success(meta, delta=1.0)
        assert h._ns1 == 0
        assert h._ns2 == 1
        assert h._gen_success_freq == [0.7]

    def test_record_success_skips_cauchy_phase(self):
        from panobbgo.heuristics.lshade_ep_sin import _EpSinTrialMeta, LSHADE_EpSin

        h = LSHADE_EpSin(self.strategy)
        # sin_choice = 0 ⇒ Cauchy phase, no ensemble tracking.
        meta = _EpSinTrialMeta(slot_idx=0, F=0.4, CR=0.7, sin_choice=0, freq=0.0)
        h._record_success(meta, delta=1.0)
        assert h._ns1 == 0
        assert h._ns2 == 0
        assert h._gen_success_freq == []

    def test_record_success_ignores_plain_trial_meta(self):
        """Defensive: a base-class ``_TrialMeta`` (no sin fields) must not crash."""
        from panobbgo.heuristics.lshade import _TrialMeta
        from panobbgo.heuristics.lshade_ep_sin import LSHADE_EpSin

        h = LSHADE_EpSin(self.strategy)
        meta = _TrialMeta(slot_idx=0, F=0.4, CR=0.7)
        h._record_success(meta, delta=1.0)
        # Should be a no-op.
        assert h._ns1 == 0
        assert h._ns2 == 0


# ----------------------------------------------------------------------
# Pipeline + on_start / on_restart
# ----------------------------------------------------------------------


class LSHADEEpSinPipelineTests(_MockStrategyMixin, PanobbgoTestCase):
    def test_on_start_emits_NP_init_points(self):
        from panobbgo.heuristics.lshade_ep_sin import LSHADE_EpSin

        h = LSHADE_EpSin(self.strategy, NP_init=8, seed=0)
        h.on_start()
        emitted = h.get_points(limit=100)
        assert len(emitted) == 8
        assert len(h._pending) == 8
        assert len(h._population) == 8
        assert all(slot is None for slot in h._population)
        assert all(pt.who.startswith("LSHADE_EpSin:") for pt in emitted)

    def test_on_start_resets_ensemble_state(self):
        from panobbgo.heuristics.lshade_ep_sin import LSHADE_EpSin

        h = LSHADE_EpSin(self.strategy, NP_init=4, mu_freq_init=0.7, seed=0)
        h._mu_freq = 0.1
        h._p_s = 0.9
        h._ns1 = 5
        h._ns2 = 7
        h._gen_success_freq = [0.4, 0.5]
        h._gen_count = 12
        h._last_sin = (2, 0.8)
        h.on_start()
        assert h._mu_freq == 0.7
        assert h._p_s == 0.5
        assert h._ns1 == 0
        assert h._ns2 == 0
        assert h._gen_success_freq == []
        assert h._gen_count == 0
        assert h._last_sin == (0, 0.0)

    def test_on_restart_resets_ensemble_state(self):
        from panobbgo.heuristics.lshade_ep_sin import LSHADE_EpSin

        h = LSHADE_EpSin(self.strategy, NP_init=6, mu_freq_init=0.4, seed=2)
        h.on_start()
        h.get_points(limit=100)
        h._mu_freq = 0.9
        h._p_s = 0.2
        h._ns1 = 3
        h._ns2 = 5
        h._gen_success_freq = [0.1, 0.2]
        h._gen_count = 4
        h._last_sin = (1, 0.5)
        h.on_restart(np.array([0.0, 0.0]), reason="test")
        assert h._mu_freq == 0.4
        assert h._p_s == 0.5
        assert h._ns1 == 0
        assert h._ns2 == 0
        assert h._gen_success_freq == []
        assert h._gen_count == 0
        assert h._last_sin == (0, 0.0)
        assert len(h._pending) == h.NP_init

    def test_initial_fills_carry_no_sin_choice(self):
        """Random-fill trials use ``F = NaN`` and must record ``sin_choice = 0``."""
        from panobbgo.heuristics.lshade_ep_sin import _EpSinTrialMeta, LSHADE_EpSin

        h = LSHADE_EpSin(self.strategy, NP_init=4, seed=0)
        h.on_start()
        for meta in h._pending.values():
            assert isinstance(meta, _EpSinTrialMeta)
            assert meta.sin_choice == 0
            assert meta.freq == 0.0
            assert np.isnan(meta.F)
            assert np.isnan(meta.CR)

    def test_filled_population_emits_evolutionary_trials(self):
        from panobbgo.heuristics.lshade_ep_sin import LSHADE_EpSin

        h = LSHADE_EpSin(self.strategy, NP_init=5, seed=12)
        h.on_start()
        h.get_points(limit=100)
        items = list(h._pending.items())[:4]
        results = []
        for req_id, meta in items:
            x = self.problem.random_point()
            results.append(_build_result(self.strategy, x, 10.0 + meta.slot_idx, f"LSHADE_EpSin:{req_id}"))
        h.on_new_results(results)
        h.get_points(limit=100)
        evo = [m for _, m in h._pending.items() if not np.isnan(m.F) and not np.isnan(m.CR)]
        assert len(evo) >= 1

    def test_better_trial_records_ensemble_success(self):
        """A successful sinusoidal trial must update ``ns1`` or ``ns2``."""
        from panobbgo.heuristics.lshade_ep_sin import _EpSinTrialMeta, LSHADE_EpSin

        h = LSHADE_EpSin(self.strategy, NP_init=4, seed=13)
        h.on_start()
        h.get_points(limit=100)
        # Fill the four slots.
        items = list(h._pending.items())
        results = []
        for (rid, _m), fx in zip(items, [100.0, 110.0, 120.0, 130.0]):
            results.append(_build_result(self.strategy, self.problem.random_point(), fx, f"LSHADE_EpSin:{rid}"))
        h.on_new_results(results)
        h.get_points(limit=100)

        # Pick a pending evolutionary trial and force success.
        target_slot = 0
        target_fx = h._population[target_slot].fx
        evo = [(r, m) for r, m in h._pending.items() if m.slot_idx == target_slot and isinstance(m, _EpSinTrialMeta)]
        assert evo
        rid, meta = evo[0]
        # We need the trial to be one that actually used a sinusoid.
        # In the first half of the budget (progress=0), it should have.
        assert meta.sin_choice in (1, 2)
        sin = meta.sin_choice
        improved = _build_result(self.strategy, self.problem.random_point(), target_fx - 50.0, f"LSHADE_EpSin:{rid}")
        h.on_new_results([improved])
        if sin == 1:
            assert h._ns1 >= 1
        else:
            assert h._ns2 >= 1

    def test_smoke_quadratic_no_regression(self):
        """A few generations on f(x)=||x||² makes no negative global progress."""
        from panobbgo.heuristics.lshade import _Dropped
        from panobbgo.heuristics.lshade_ep_sin import LSHADE_EpSin
        from panobbgo.lib import Point, Result

        h = LSHADE_EpSin(self.strategy, NP_init=8, NP_min=4, seed=5)
        h.on_start()

        def fx_of(x):
            return float(np.dot(x, x))

        items = list(h._pending.items())
        h.get_points(limit=100)
        results = [
            Result(Point(x := self.problem.random_point(), f"LSHADE_EpSin:{rid}"), fx_of(x)) for rid, _m in items
        ]
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
                results.append(Result(Point(x, f"LSHADE_EpSin:{rid}"), fx_of(x)))
            h.on_new_results(results)

        best_after = min(s.fx for s in h._population if isinstance(s, Result))
        assert best_after <= best_before + 1e-6


# ----------------------------------------------------------------------
# Base-class hook safety: LSHADE / jSO / NL-SHADE-RSP unchanged
# ----------------------------------------------------------------------


class DEFamilyHookTests(_MockStrategyMixin, PanobbgoTestCase):
    def test_lshade_make_trial_meta_returns_plain_meta(self):
        from panobbgo.heuristics.lshade import LSHADE, _TrialMeta

        h = LSHADE(self.strategy)
        meta = h._make_trial_meta(slot_idx=2, F=0.5, CR=0.7)
        assert type(meta) is _TrialMeta
        assert meta.slot_idx == 2
        assert meta.F == 0.5
        assert meta.CR == 0.7

    def test_lshade_record_success_is_noop(self):
        from panobbgo.heuristics.lshade import LSHADE, _TrialMeta

        h = LSHADE(self.strategy)
        # No-op hook — should not raise nor mutate state.
        meta = _TrialMeta(slot_idx=0, F=0.5, CR=0.5)
        h._record_success(meta, delta=1.0)

    def test_jso_make_trial_meta_returns_plain_meta(self):
        from panobbgo.heuristics.jso import JSO
        from panobbgo.heuristics.lshade import _TrialMeta

        h = JSO(self.strategy)
        meta = h._make_trial_meta(slot_idx=0, F=0.4, CR=0.6)
        assert type(meta) is _TrialMeta

    def test_nl_shade_rsp_make_trial_meta_returns_plain_meta(self):
        from panobbgo.heuristics.lshade import _TrialMeta
        from panobbgo.heuristics.nl_shade_rsp import NLSHADE_RSP

        h = NLSHADE_RSP(self.strategy)
        meta = h._make_trial_meta(slot_idx=0, F=0.4, CR=0.6)
        assert type(meta) is _TrialMeta


# ----------------------------------------------------------------------
# Registration
# ----------------------------------------------------------------------


class LSHADEEpSinRegistrationTests(_MockStrategyMixin, PanobbgoTestCase):
    def test_registered_in_heuristics_package(self):
        import panobbgo.heuristics as h

        assert hasattr(h, "LSHADE_EpSin")
        assert "LSHADE_EpSin" in h.__all__

    def test_in_structural_catalog(self):
        from panobbgo.heuristics.lshade_ep_sin import LSHADE_EpSin
        from panobbgo.self_improve import StructuralMutationRule, default_structural_catalog

        catalog = default_structural_catalog()
        add_rules = [r for r in catalog.rules if isinstance(r, StructuralMutationRule) and r.op == "add_heuristic"]
        assert add_rules
        has_epsin = any(cls is LSHADE_EpSin for rule in add_rules for cls, _ in (rule.candidate_classes or ()))
        assert has_epsin

    def test_kwarg_catalog_has_epsin_dials(self):
        from panobbgo.self_improve import MutationRule, default_catalog

        rules = default_catalog().rules
        params = {
            (r.class_name, r.param_name)
            for r in rules
            if isinstance(r, MutationRule) and r.class_name == "LSHADE_EpSin"
        }
        assert ("LSHADE_EpSin", "NP_init") in params
        assert ("LSHADE_EpSin", "mu_freq_init") in params
