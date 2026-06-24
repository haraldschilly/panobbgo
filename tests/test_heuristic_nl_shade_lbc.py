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

"""Tests for the NL-SHADE-LBC (Stanovov et al. 2022) adaptive DE heuristic."""

from __future__ import annotations

import numpy as np
import pytest

from panobbgo.utils import PanobbgoTestCase


class _MockStrategyMixin:
    """Same scaffolding as the NL-SHADE-RSP / jSO / L-SHADE tests.

    NL-SHADE-LBC inherits the entire NL-SHADE-RSP pipeline, so the mock
    strategy needs the same constraint-handler / max_eval setup.
    ``config.max_eval`` is saved / restored to prevent cross-test bleed.
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


class NLSHADELBCConstructionTests(_MockStrategyMixin, PanobbgoTestCase):
    def test_default_construction(self):
        from panobbgo.heuristics.nl_shade_lbc import (
            NLSHADE_LBC,
            _DEFAULT_M_LBC,
            _DEFAULT_P_CR_FINAL,
            _DEFAULT_P_CR_INIT,
            _DEFAULT_P_F_FINAL,
            _DEFAULT_P_F_INIT,
        )

        h = NLSHADE_LBC(self.strategy)
        assert h.NP_init == 30
        assert h.NP_min == 4
        assert h.H == 5
        assert h.p_best_max == 0.25
        assert h.p_best_min == 0.125
        assert h.archive_factor == 1.0
        assert h.k_rank == 3.0
        assert h.adaptive_archive is True
        assert h.p_F_init == _DEFAULT_P_F_INIT == 3.5
        assert h.p_F_final == _DEFAULT_P_F_FINAL == 1.5
        assert h.p_CR_init == _DEFAULT_P_CR_INIT == 1.0
        assert h.p_CR_final == _DEFAULT_P_CR_FINAL == 1.5
        assert h.m_lbc == _DEFAULT_M_LBC == 1.5
        assert h.name == "NLSHADE_LBC"
        # F-cap inherited from jSO via NL-SHADE-RSP (jSO regime).
        assert h.F_schedule == "jso"

    def test_custom_construction(self):
        from panobbgo.heuristics.nl_shade_lbc import NLSHADE_LBC

        h = NLSHADE_LBC(
            self.strategy,
            NP_init=24,
            NP_min=6,
            H=4,
            p_best_max=0.3,
            p_best_min=0.1,
            archive_factor=2.0,
            k_rank=2.0,
            adaptive_archive=False,
            p_F_init=4.0,
            p_F_final=2.0,
            p_CR_init=0.8,
            p_CR_final=1.8,
            m_lbc=1.2,
            seed=11,
            name="MyLBC",
        )
        assert h.NP_init == 24
        assert h.NP_min == 6
        assert h.H == 4
        assert h.p_best_max == 0.3
        assert h.k_rank == 2.0
        assert h.adaptive_archive is False
        assert h.p_F_init == 4.0
        assert h.p_F_final == 2.0
        assert h.p_CR_init == 0.8
        assert h.p_CR_final == 1.8
        assert h.m_lbc == 1.2
        assert h.name == "MyLBC"

    def test_subclass_of_nl_shade_rsp_jso_lshade(self):
        from panobbgo.heuristics.jso import JSO
        from panobbgo.heuristics.lshade import LSHADE
        from panobbgo.heuristics.nl_shade_lbc import NLSHADE_LBC
        from panobbgo.heuristics.nl_shade_rsp import NLSHADE_RSP

        h = NLSHADE_LBC(self.strategy)
        assert isinstance(h, NLSHADE_RSP)
        assert isinstance(h, JSO)
        assert isinstance(h, LSHADE)

    def test_invalid_p_F_init(self):
        from panobbgo.heuristics.nl_shade_lbc import NLSHADE_LBC

        with pytest.raises(ValueError, match="p_F_init"):
            NLSHADE_LBC(self.strategy, p_F_init=float("nan"))
        with pytest.raises(ValueError, match="p_F_init"):
            NLSHADE_LBC(self.strategy, p_F_init=float("inf"))

    def test_invalid_p_F_final(self):
        from panobbgo.heuristics.nl_shade_lbc import NLSHADE_LBC

        with pytest.raises(ValueError, match="p_F_final"):
            NLSHADE_LBC(self.strategy, p_F_final=float("nan"))

    def test_invalid_p_CR_init(self):
        from panobbgo.heuristics.nl_shade_lbc import NLSHADE_LBC

        with pytest.raises(ValueError, match="p_CR_init"):
            NLSHADE_LBC(self.strategy, p_CR_init=float("inf"))

    def test_invalid_p_CR_final(self):
        from panobbgo.heuristics.nl_shade_lbc import NLSHADE_LBC

        with pytest.raises(ValueError, match="p_CR_final"):
            NLSHADE_LBC(self.strategy, p_CR_final=float("-inf"))

    def test_invalid_m_lbc(self):
        from panobbgo.heuristics.nl_shade_lbc import NLSHADE_LBC

        with pytest.raises(ValueError, match="m_lbc"):
            NLSHADE_LBC(self.strategy, m_lbc=0.0)
        with pytest.raises(ValueError, match="m_lbc"):
            NLSHADE_LBC(self.strategy, m_lbc=-1.0)
        with pytest.raises(ValueError, match="m_lbc"):
            NLSHADE_LBC(self.strategy, m_lbc=float("nan"))

    def test_inherits_rsp_jso_validation(self):
        """NL-SHADE-RSP / jSO H >= 2 and p_best ordering rules still apply."""
        from panobbgo.heuristics.nl_shade_lbc import NLSHADE_LBC

        with pytest.raises(ValueError, match="H must be >= 2"):
            NLSHADE_LBC(self.strategy, H=1)
        with pytest.raises(ValueError, match="p_best_min .* must be <= p_best_max"):
            NLSHADE_LBC(self.strategy, p_best_max=0.2, p_best_min=0.3)
        with pytest.raises(ValueError, match="k_rank"):
            NLSHADE_LBC(self.strategy, k_rank=-0.1)


# ----------------------------------------------------------------------
# Named LBC regimes (lbc_regime kwarg)
# ----------------------------------------------------------------------


class NLSHADELBCRegimeTests(_MockStrategyMixin, PanobbgoTestCase):
    def test_default_regime_is_none(self):
        """No-argument construction stores ``lbc_regime = None``."""
        from panobbgo.heuristics.nl_shade_lbc import NLSHADE_LBC

        h = NLSHADE_LBC(self.strategy)
        assert h.lbc_regime is None

    def test_regime_cec2022_matches_defaults(self):
        """The CEC 2022 regime is bit-identical to the per-field defaults."""
        from panobbgo.heuristics.nl_shade_lbc import (
            NLSHADE_LBC,
            _DEFAULT_M_LBC,
            _DEFAULT_P_CR_FINAL,
            _DEFAULT_P_CR_INIT,
            _DEFAULT_P_F_FINAL,
            _DEFAULT_P_F_INIT,
        )

        h = NLSHADE_LBC(self.strategy, lbc_regime="cec2022")
        assert h.lbc_regime == "cec2022"
        assert h.p_F_init == _DEFAULT_P_F_INIT
        assert h.p_F_final == _DEFAULT_P_F_FINAL
        assert h.p_CR_init == _DEFAULT_P_CR_INIT
        assert h.p_CR_final == _DEFAULT_P_CR_FINAL
        assert h.m_lbc == _DEFAULT_M_LBC

    def test_regime_lshade_recovers_standard_lehmer(self):
        """The ``lshade`` regime recovers ``p = 2, m = 1`` for both F and CR."""
        from panobbgo.heuristics.nl_shade_lbc import NLSHADE_LBC

        h = NLSHADE_LBC(self.strategy, lbc_regime="lshade")
        assert h.lbc_regime == "lshade"
        assert h.p_F_init == 2.0
        assert h.p_F_final == 2.0
        assert h.p_CR_init == 2.0
        assert h.p_CR_final == 2.0
        assert h.m_lbc == 1.0

    def test_regime_flat_is_constant_arithmetic(self):
        """The ``flat`` regime sets ``p = 1`` throughout (pure arithmetic mean)."""
        from panobbgo.heuristics.nl_shade_lbc import NLSHADE_LBC, _DEFAULT_M_LBC

        h = NLSHADE_LBC(self.strategy, lbc_regime="flat")
        assert h.lbc_regime == "flat"
        assert h.p_F_init == 1.0
        assert h.p_F_final == 1.0
        assert h.p_CR_init == 1.0
        assert h.p_CR_final == 1.0
        assert h.m_lbc == _DEFAULT_M_LBC

    def test_regime_aggressive_is_high_biased(self):
        """The ``aggressive`` regime sets the largest exponents."""
        from panobbgo.heuristics.nl_shade_lbc import NLSHADE_LBC, _DEFAULT_M_LBC

        h = NLSHADE_LBC(self.strategy, lbc_regime="aggressive")
        assert h.lbc_regime == "aggressive"
        assert h.p_F_init == 5.0
        assert h.p_F_final == 3.0
        assert h.p_CR_init == 3.0
        assert h.p_CR_final == 5.0
        assert h.m_lbc == _DEFAULT_M_LBC

    def test_regime_dict_has_expected_keys(self):
        """Sanity-check the regime dict's keys and tuple width."""
        from panobbgo.heuristics.nl_shade_lbc import _LBC_REGIMES

        assert set(_LBC_REGIMES) == {"cec2022", "lshade", "flat", "aggressive"}
        for name, tup in _LBC_REGIMES.items():
            assert len(tup) == 5, f"Regime {name!r}: expected 5 fields, got {len(tup)}"
            assert tup[4] > 0.0, f"Regime {name!r}: m_lbc must be > 0"
            for v in tup:
                assert np.isfinite(v), f"Regime {name!r}: non-finite field {v!r}"

    def test_invalid_regime_string_raises(self):
        """Unknown regime names raise :class:`ValueError`."""
        from panobbgo.heuristics.nl_shade_lbc import NLSHADE_LBC

        with pytest.raises(ValueError, match="lbc_regime"):
            NLSHADE_LBC(self.strategy, lbc_regime="bogus")
        with pytest.raises(ValueError, match="lbc_regime"):
            NLSHADE_LBC(self.strategy, lbc_regime="")

    def test_invalid_regime_type_raises(self):
        """Non-string / non-None ``lbc_regime`` raises :class:`ValueError`."""
        from panobbgo.heuristics.nl_shade_lbc import NLSHADE_LBC

        with pytest.raises(ValueError, match="lbc_regime"):
            NLSHADE_LBC(self.strategy, lbc_regime=42)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="lbc_regime"):
            NLSHADE_LBC(self.strategy, lbc_regime=True)  # type: ignore[arg-type]

    def test_regime_with_explicit_kwargs_raises(self):
        """Regime + any explicit LBC field is mutually exclusive."""
        from panobbgo.heuristics.nl_shade_lbc import NLSHADE_LBC

        for kwarg, value in (
            ("p_F_init", 2.0),
            ("p_F_final", 1.0),
            ("p_CR_init", 1.2),
            ("p_CR_final", 1.3),
            ("m_lbc", 1.2),
        ):
            with pytest.raises(ValueError, match="mutually exclusive"):
                NLSHADE_LBC(self.strategy, lbc_regime="lshade", **{kwarg: value})

    def test_regime_with_unrelated_kwargs_ok(self):
        """Regime composes cleanly with non-LBC kwargs (NP_init / H / etc.)."""
        from panobbgo.heuristics.nl_shade_lbc import NLSHADE_LBC

        h = NLSHADE_LBC(self.strategy, NP_init=15, H=5, lbc_regime="cec2022")
        assert h.NP_init == 15
        assert h.H == 5
        assert h.lbc_regime == "cec2022"

    def test_normalize_helper_collapses_none(self):
        """The ``_normalize_lbc_regime`` helper preserves ``None``."""
        from panobbgo.heuristics.nl_shade_lbc import _normalize_lbc_regime

        assert _normalize_lbc_regime(None) is None

    def test_lshade_regime_memory_update_matches_standard_lehmer(self):
        """``lbc_regime="lshade"`` reproduces the standard ``s^2 / s^1`` mean.

        With ``p = 2, m = 1`` the LBC generalised Lehmer mean collapses to
        the standard L-SHADE / jSO weighted Lehmer mean — concretely
        ``Σ(w_i · F_i^2) / Σ(w_i · F_i^1)``.  This is the analytic
        equivalence the regime documents.
        """
        from panobbgo.heuristics.nl_shade_lbc import NLSHADE_LBC

        h = NLSHADE_LBC(self.strategy, H=4, lbc_regime="lshade")
        h.on_start()
        # Strategy budget set so the schedule is deterministic.
        self.strategy.config.max_eval = 1000
        self.strategy.results = list(range(500))  # any progress; lshade is constant
        F_vals = np.array([0.2, 0.4, 0.6, 0.8])
        weights = np.array([1.0, 2.0, 3.0, 4.0])
        w_norm = weights / weights.sum()
        expected = float(np.sum(w_norm * F_vals**2) / np.sum(w_norm * F_vals**1))
        h._success_F = list(F_vals.tolist())
        h._success_CR = list(F_vals.tolist())
        h._success_delta = list(weights.tolist())
        h._update_memory()
        assert h._M_F[0] == pytest.approx(expected)


# ----------------------------------------------------------------------
# Linear bias change schedule
# ----------------------------------------------------------------------


class NLSHADELBCScheduleTests(_MockStrategyMixin, PanobbgoTestCase):
    def test_exponent_endpoints_F(self):
        from panobbgo.heuristics.nl_shade_lbc import NLSHADE_LBC

        h = NLSHADE_LBC(self.strategy)

        # progress 0 → p_F_init, progress 1 → p_F_final.  Use the
        # strategy.results list to drive ``_progress()``.
        self.strategy.results = []
        self.strategy.config.max_eval = 100
        assert h._lbc_exponent(h.p_F_init, h.p_F_final) == pytest.approx(h.p_F_init)
        self.strategy.results = list(range(100))
        assert h._lbc_exponent(h.p_F_init, h.p_F_final) == pytest.approx(h.p_F_final)

    def test_exponent_linear_midpoint(self):
        from panobbgo.heuristics.nl_shade_lbc import NLSHADE_LBC

        h = NLSHADE_LBC(self.strategy)
        self.strategy.config.max_eval = 100
        self.strategy.results = list(range(50))  # progress = 0.5
        mid = h._lbc_exponent(h.p_F_init, h.p_F_final)
        assert mid == pytest.approx(0.5 * h.p_F_init + 0.5 * h.p_F_final)

    def test_exponent_clipped_to_unit_interval(self):
        from panobbgo.heuristics.nl_shade_lbc import NLSHADE_LBC

        h = NLSHADE_LBC(self.strategy)
        self.strategy.config.max_eval = 100
        # Overspending — progress should clip to 1.0.
        self.strategy.results = list(range(200))
        assert h._lbc_exponent(h.p_F_init, h.p_F_final) == pytest.approx(h.p_F_final)

    def test_exponent_fallback_when_budget_unknown(self):
        """``_progress() is None`` → schedule returns p_init."""
        from panobbgo.heuristics.nl_shade_lbc import NLSHADE_LBC

        h = NLSHADE_LBC(self.strategy)
        self.strategy.config.max_eval = 0  # unknown budget
        assert h._lbc_exponent(h.p_F_init, h.p_F_final) == h.p_F_init
        assert h._lbc_exponent(h.p_CR_init, h.p_CR_final) == h.p_CR_init


# ----------------------------------------------------------------------
# Memory update (LBC Lehmer mean)
# ----------------------------------------------------------------------


class NLSHADELBCMemoryUpdateTests(_MockStrategyMixin, PanobbgoTestCase):
    def _seed_success_buffer(self, h, F_vals, CR_vals, deltas):
        h._success_F = list(map(float, F_vals))
        h._success_CR = list(map(float, CR_vals))
        h._success_delta = list(map(float, deltas))

    def test_update_memory_writes_to_writable_range_only(self):
        """The jSO anchor bin (index H-1) must stay frozen at 0.9."""
        from panobbgo.heuristics.nl_shade_lbc import NLSHADE_LBC

        h = NLSHADE_LBC(self.strategy, H=4)
        h.on_start()
        anchor_F_before = h._M_F[-1]
        anchor_CR_before = h._M_CR[-1]
        # Run several updates to make sure no write ever falls on H-1.
        for _ in range(2 * h.H):
            self._seed_success_buffer(h, [0.5, 0.7], [0.3, 0.6], [1.0, 2.0])
            h._update_memory()
        assert h._M_F[-1] == anchor_F_before
        assert h._M_CR[-1] == anchor_CR_before

    def test_update_memory_advances_pointer_modulo_writable(self):
        from panobbgo.heuristics.nl_shade_lbc import NLSHADE_LBC

        h = NLSHADE_LBC(self.strategy, H=5)
        h.on_start()
        for expected in [1, 2, 3, 0, 1, 2]:  # H-1 = 4 writable bins, wraps mod 4
            self._seed_success_buffer(h, [0.4], [0.6], [1.0])
            h._update_memory()
            assert h._mem_ptr == expected

    def test_no_op_when_buffer_empty(self):
        from panobbgo.heuristics.nl_shade_lbc import NLSHADE_LBC

        h = NLSHADE_LBC(self.strategy, H=4)
        h.on_start()
        snap_F = h._M_F.copy()
        snap_CR = h._M_CR.copy()
        h._update_memory()
        np.testing.assert_array_equal(h._M_F, snap_F)
        np.testing.assert_array_equal(h._M_CR, snap_CR)

    def test_F_memory_in_unit_interval(self):
        from panobbgo.heuristics.nl_shade_lbc import NLSHADE_LBC

        h = NLSHADE_LBC(self.strategy, H=4)
        h.on_start()
        self.strategy.config.max_eval = 1000
        self.strategy.results = list(range(100))  # progress ~ 0.1
        self._seed_success_buffer(
            h,
            F_vals=[0.1, 0.5, 0.9, 0.2, 0.7],
            CR_vals=[0.2, 0.4, 0.8, 0.5, 0.6],
            deltas=[1.0, 2.0, 1.5, 0.5, 0.7],
        )
        h._update_memory()
        # The write went to bin 0 (initial _mem_ptr).
        assert 0.0 <= h._M_F[0] <= 1.0
        assert 0.0 <= h._M_CR[0] <= 1.0

    def test_F_memory_at_progress_zero_uses_p_init(self):
        """At progress=0 the LBC mean uses p=p_F_init, m=m_lbc."""
        from panobbgo.heuristics.nl_shade_lbc import NLSHADE_LBC

        h = NLSHADE_LBC(self.strategy, H=4, p_F_init=3.5, p_F_final=1.5, m_lbc=1.5)
        h.on_start()
        self.strategy.config.max_eval = 1000
        self.strategy.results = []  # progress = 0
        F_vals = np.array([0.3, 0.5, 0.9, 0.6])
        CR_vals = np.array([0.4, 0.5, 0.8, 0.3])
        deltas = np.array([1.0, 1.0, 1.0, 1.0])
        self._seed_success_buffer(h, F_vals, CR_vals, deltas)
        h._update_memory()
        w = deltas / deltas.sum()
        expected = float(np.sum(w * F_vals**3.5) / np.sum(w * F_vals**2.0))
        assert h._M_F[0] == pytest.approx(expected, rel=1e-9)

    def test_F_memory_recovers_standard_lehmer_when_p2_m1(self):
        """At p_F=2 (both ends) and m_lbc=1, the LBC formula recovers L-SHADE."""
        from panobbgo.heuristics.nl_shade_lbc import NLSHADE_LBC

        h = NLSHADE_LBC(
            self.strategy,
            H=4,
            p_F_init=2.0,
            p_F_final=2.0,
            p_CR_init=2.0,
            p_CR_final=2.0,
            m_lbc=1.0,
        )
        h.on_start()
        self.strategy.config.max_eval = 1000
        self.strategy.results = list(range(500))  # progress = 0.5 — schedule still p=2
        F_vals = np.array([0.4, 0.7, 0.2])
        CR_vals = np.array([0.6, 0.3, 0.9])
        deltas = np.array([1.0, 2.0, 3.0])
        self._seed_success_buffer(h, F_vals, CR_vals, deltas)
        h._update_memory()
        w = deltas / deltas.sum()
        expected_F = float(np.sum(w * F_vals * F_vals) / np.sum(w * F_vals))
        expected_CR = float(np.sum(w * CR_vals * CR_vals) / np.sum(w * CR_vals))
        assert h._M_F[0] == pytest.approx(expected_F, rel=1e-9)
        assert h._M_CR[0] == pytest.approx(expected_CR, rel=1e-9)

    def test_CR_zero_terminal_sentinel(self):
        """All-zero CR successes plant the terminal sentinel (-1)."""
        from panobbgo.heuristics.lshade import _CR_TERMINAL
        from panobbgo.heuristics.nl_shade_lbc import NLSHADE_LBC

        h = NLSHADE_LBC(self.strategy, H=4)
        h.on_start()
        self._seed_success_buffer(h, [0.5, 0.7], [0.0, 0.0], [1.0, 1.0])
        h._update_memory()
        assert h._M_CR[0] == _CR_TERMINAL

    def test_CR_terminal_bin_stays_terminal(self):
        """Once the CR bin is the sentinel, subsequent updates leave it alone."""
        from panobbgo.heuristics.lshade import _CR_TERMINAL
        from panobbgo.heuristics.nl_shade_lbc import NLSHADE_LBC

        h = NLSHADE_LBC(self.strategy, H=4)
        h.on_start()
        h._M_CR[0] = _CR_TERMINAL  # plant the sentinel at the writable bin
        self._seed_success_buffer(h, [0.5], [0.6], [1.0])
        h._update_memory()
        assert h._M_CR[0] == _CR_TERMINAL

    def test_CR_zero_entries_filtered_with_mixed_values(self):
        """Mixed CR values: zeros skipped, LBC applied to positive subset."""
        from panobbgo.heuristics.nl_shade_lbc import NLSHADE_LBC

        h = NLSHADE_LBC(self.strategy, H=4, p_CR_init=1.0, p_CR_final=1.0, m_lbc=1.5)
        h.on_start()
        self.strategy.config.max_eval = 1000
        self.strategy.results = []  # progress = 0
        # 1 zero + 3 positives.  CR^(1-1.5)=CR^-0.5 is undefined at 0, so
        # the zero must be filtered.
        F_vals = np.array([0.3, 0.5, 0.7, 0.6])
        CR_vals = np.array([0.0, 0.4, 0.6, 0.8])
        deltas = np.array([1.0, 1.0, 1.0, 1.0])
        self._seed_success_buffer(h, F_vals, CR_vals, deltas)
        # Should not blow up — the LBC update must filter zero CR entries.
        h._update_memory()
        assert np.isfinite(h._M_CR[0])
        assert 0.0 <= h._M_CR[0] <= 1.0
        # Cross-check: the filtered LBC formula applied to the 3 positives.
        pos = CR_vals[1:]
        w_pos = deltas[1:] / deltas[1:].sum()
        expected = float(np.sum(w_pos * pos**1.0) / np.sum(w_pos * pos**-0.5))
        assert h._M_CR[0] == pytest.approx(expected, rel=1e-9)

    def test_uniform_weights_when_delta_total_zero(self):
        """Zero-delta successes fall back to uniform weighting."""
        from panobbgo.heuristics.nl_shade_lbc import NLSHADE_LBC

        h = NLSHADE_LBC(self.strategy, H=4, p_F_init=2.0, p_F_final=2.0, m_lbc=1.0)
        h.on_start()
        self.strategy.config.max_eval = 1000
        self.strategy.results = []
        F_vals = np.array([0.2, 0.4, 0.6])
        CR_vals = np.array([0.3, 0.5, 0.7])
        deltas = np.array([0.0, 0.0, 0.0])
        self._seed_success_buffer(h, F_vals, CR_vals, deltas)
        h._update_memory()
        # Uniform weights → expected = Σ(F^2)/Σ(F).
        expected = float(np.sum(F_vals * F_vals) / np.sum(F_vals))
        assert h._M_F[0] == pytest.approx(expected, rel=1e-9)


# ----------------------------------------------------------------------
# End-to-end pipeline smoke
# ----------------------------------------------------------------------


class NLSHADELBCPipelineTests(_MockStrategyMixin, PanobbgoTestCase):
    def test_on_start_emits_NP_init_points(self):
        from panobbgo.heuristics.nl_shade_lbc import NLSHADE_LBC

        h = NLSHADE_LBC(self.strategy, NP_init=8, seed=0)
        h.on_start()
        emitted = h.get_points(limit=100)
        assert len(emitted) == 8
        assert all(pt.who.startswith("NLSHADE_LBC:") for pt in emitted)

    def test_smoke_quadratic_no_regression(self):
        """A few rounds on f(x)=||x||² makes no negative global progress."""
        from panobbgo.heuristics.lshade import _Dropped
        from panobbgo.heuristics.nl_shade_lbc import NLSHADE_LBC
        from panobbgo.lib import Point, Result

        h = NLSHADE_LBC(self.strategy, NP_init=8, NP_min=4, seed=5)
        h.on_start()

        def fx_of(x):
            return float(np.dot(x, x))

        items = list(h._pending.items())
        h.get_points(limit=100)
        results = [Result(Point(x := self.problem.random_point(), f"NLSHADE_LBC:{rid}"), fx_of(x)) for rid, _m in items]
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
                results.append(Result(Point(x, f"NLSHADE_LBC:{rid}"), fx_of(x)))
            h.on_new_results(results)

        best_after = min(s.fx for s in h._population if isinstance(s, Result))
        assert best_after <= best_before + 1e-6

    def test_restart_resets_memory_and_archive(self):
        from panobbgo.heuristics.nl_shade_lbc import NLSHADE_LBC

        h = NLSHADE_LBC(self.strategy, NP_init=6, seed=2)
        h.on_start()
        h.get_points(limit=100)
        h._rsp_archive_cap = 3
        h._archive.append(np.array([0.5, 0.5]))
        h.on_restart(np.array([0.0, 0.0]), reason="test")
        assert h._rsp_archive_cap is None
        assert h._archive == []
        assert len(h._pending) == h.NP_init


# ----------------------------------------------------------------------
# Byte-identical safety: NL-SHADE-RSP / jSO / L-SHADE behaviour unchanged
# ----------------------------------------------------------------------


class NLSHADELBCInheritanceTests(_MockStrategyMixin, PanobbgoTestCase):
    def test_nl_shade_rsp_uses_unchanged_lehmer_mean(self):
        """NL-SHADE-RSP must not pick up the LBC override."""
        from panobbgo.heuristics.nl_shade_rsp import NLSHADE_RSP

        h = NLSHADE_RSP(self.strategy, H=4)
        h.on_start()
        h._success_F = [0.4, 0.6]
        h._success_CR = [0.5, 0.7]
        h._success_delta = [1.0, 1.0]
        h._update_memory()
        # Standard L-SHADE Lehmer mean with p=2, m=1.
        F_vals = np.array([0.4, 0.6])
        expected = float(np.sum(F_vals * F_vals) / np.sum(F_vals))
        assert h._M_F[0] == pytest.approx(expected, rel=1e-9)


# ----------------------------------------------------------------------
# Registration
# ----------------------------------------------------------------------


class NLSHADELBCRegistrationTests(_MockStrategyMixin, PanobbgoTestCase):
    def test_registered_in_heuristics_package(self):
        import panobbgo.heuristics as h

        assert hasattr(h, "NLSHADE_LBC")
        assert "NLSHADE_LBC" in h.__all__

    def test_in_structural_catalog(self):
        from panobbgo.heuristics.nl_shade_lbc import NLSHADE_LBC
        from panobbgo.self_improve import StructuralMutationRule, default_structural_catalog

        catalog = default_structural_catalog()
        add_rules = [r for r in catalog.rules if isinstance(r, StructuralMutationRule) and r.op == "add_heuristic"]
        assert add_rules
        has_lbc = any(cls is NLSHADE_LBC for rule in add_rules for cls, _ in (rule.candidate_classes or ()))
        assert has_lbc

    def test_kwarg_catalog_has_lbc_dials(self):
        """``default_catalog`` ships the consolidated LBC kwarg rules.

        The five per-field LBC float rules shipped 2026-05-28 were
        retired when the ``lbc_regime`` categorical arm landed: one
        well-curated composite bandit arm subsumes their joint search
        space (see the dated entry in ``planning/SELF_IMPROVEMENT_LOG.md``).
        Today the catalog ships ``NP_init`` (population sizing) and
        ``lbc_regime`` (joint Lehmer-mean regime) — the per-field
        constructors stay available for future opt-in specs, just no
        longer have a dedicated catalog rule.
        """
        from panobbgo.self_improve import MutationRule, default_catalog

        rules = default_catalog().rules
        params = {
            (r.class_name, r.param_name) for r in rules if isinstance(r, MutationRule) and r.class_name == "NLSHADE_LBC"
        }
        assert ("NLSHADE_LBC", "NP_init") in params
        assert ("NLSHADE_LBC", "lbc_regime") in params
        # Retired per-field rules — must not reappear without a deliberate
        # re-introduction motivated by the loop measurement.
        retired = {"p_F_init", "p_F_final", "p_CR_init", "p_CR_final", "m_lbc"}
        assert not (retired & {p for _, p in params}), (
            "Per-field LBC float_uniform rules are retired in favour of the "
            "composite ``lbc_regime`` categorical arm.  Restore them via "
            "a deliberate, ledger-evidence-driven PR rather than carrying "
            "them silently alongside the regime arm."
        )

    def test_kwarg_catalog_lbc_regime_is_categorical(self):
        """The ``lbc_regime`` rule ships the four named regimes."""
        from panobbgo.self_improve import MutationRule, default_catalog

        regime_rules = [
            r
            for r in default_catalog().rules
            if isinstance(r, MutationRule) and r.class_name == "NLSHADE_LBC" and r.param_name == "lbc_regime"
        ]
        assert len(regime_rules) == 1, "Expected exactly one lbc_regime rule"
        rule = regime_rules[0]
        assert rule.kind == "categorical_choice"
        assert set(rule.choices or ()) == {"cec2022", "lshade", "flat", "aggressive"}
