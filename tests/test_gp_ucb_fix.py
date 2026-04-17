#!/usr/bin/env python
# -*- coding: utf8 -*-
# Copyright 2012-2026 Harald Schilly <harald.schilly@gmail.com>
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

"""
Tests for the UCB acquisition fix and new BayesOpt harness strategies.

Verifies:
1. UCB (_upper_confidence_bound) returns the negative LCB so the outer
   maximiser correctly minimises the Lower Confidence Bound — the right
   behaviour for minimisation problems.
2. EI and PI acquisition signs are unchanged and still correct.
3. BayesOpt_GP and BayesOpt_Enhanced appear in the standard / full strategy
   registries of the benchmark harness.
"""

from __future__ import annotations

import numpy as np
import pytest

from panobbgo.heuristics.gaussian_process import AcquisitionFunction, GaussianProcessHeuristic


# ---------------------------------------------------------------------------
# UCB acquisition direction tests — test mathematical properties directly
# ---------------------------------------------------------------------------


class TestUCBAcquisitionMath:
    """Verify UCB acquisition formula is correct for minimisation.

    Tests call the private static/instance methods directly to avoid the need
    for a fully initialised strategy object.
    """

    def _make_gp_with_kappa(self, kappa: float = 2.0) -> GaussianProcessHeuristic:
        """Return a thin wrapper object that exposes only the acquisition methods."""
        # We need a minimal object with kappa set.  Use object.__new__ to bypass
        # __init__ and set only the attributes the methods need.
        gp = object.__new__(GaussianProcessHeuristic)
        gp.kappa = kappa
        gp.xi = 0.01
        gp.best_y = 1.0
        gp.gp_model = None
        gp.gp_constraint = None
        gp.acquisition_func = AcquisitionFunction.UCB
        return gp

    def test_ucb_returns_negative_lcb(self):
        """The raw value returned by _upper_confidence_bound should be -(μ - κσ)."""
        gp = self._make_gp_with_kappa(kappa=2.0)
        y_pred = np.array([2.0, 3.0])
        y_std = np.array([0.5, 1.0])
        result = gp._upper_confidence_bound(y_pred, y_std)
        expected = -(y_pred - gp.kappa * y_std)
        np.testing.assert_allclose(result, expected)

    def test_ucb_prefers_low_mean_equal_uncertainty(self):
        """With equal uncertainty, UCB acquisition is higher for lower predicted mean."""
        gp = self._make_gp_with_kappa(kappa=2.0)
        # Point 0: mean=0.0 (good), Point 1: mean=2.0 (bad), both sigma=0.5
        acq0 = gp._upper_confidence_bound(np.array([0.0]), np.array([0.5]))[0]
        acq1 = gp._upper_confidence_bound(np.array([2.0]), np.array([0.5]))[0]
        assert acq0 > acq1, "UCB: low-mean point should have higher acquisition"

    def test_ucb_prefers_high_uncertainty_equal_mean(self):
        """With equal mean, UCB acquisition is higher for higher uncertainty."""
        gp = self._make_gp_with_kappa(kappa=2.0)
        # Both have mean=1.0; Point 0 has sigma=1.0, Point 1 has sigma=0.01
        acq0 = gp._upper_confidence_bound(np.array([1.0]), np.array([1.0]))[0]
        acq1 = gp._upper_confidence_bound(np.array([1.0]), np.array([0.01]))[0]
        assert acq0 > acq1, "UCB: high-uncertainty point should have higher acquisition"

    def test_ucb_sign_for_zero_uncertainty(self):
        """At zero uncertainty, UCB = -mean (pure exploitation of predicted minimum)."""
        gp = self._make_gp_with_kappa(kappa=2.0)
        result = gp._upper_confidence_bound(np.array([3.0]), np.array([0.0]))[0]
        assert result == pytest.approx(-3.0)

    def test_ucb_kappa_zero_equals_negative_mean(self):
        """kappa=0 → UCB degenerates to -mean (greedy exploitation)."""
        gp = self._make_gp_with_kappa(kappa=0.0)
        y_pred = np.array([1.5, 2.5])
        y_std = np.array([0.3, 0.7])
        result = gp._upper_confidence_bound(y_pred, y_std)
        np.testing.assert_allclose(result, -y_pred)


# ---------------------------------------------------------------------------
# EI and PI regression tests — formulas must be unchanged after UCB patch
# ---------------------------------------------------------------------------


class TestEIandPIFormulas:
    """Regression tests: EI and PI should be exactly as before the UCB patch."""

    def _make_gp(self, best_y: float = 1.0) -> GaussianProcessHeuristic:
        gp = object.__new__(GaussianProcessHeuristic)
        gp.best_y = best_y
        gp.xi = 0.01
        gp.kappa = 1.96
        gp.gp_model = None
        gp.gp_constraint = None
        gp.acquisition_func = AcquisitionFunction.EI
        return gp

    def test_ei_non_negative(self):
        gp = self._make_gp(best_y=1.0)
        y_pred = np.array([0.5, 1.5, 2.0])
        y_std = np.array([0.1, 0.1, 0.1])
        ei = gp._expected_improvement(y_pred, y_std)
        assert np.all(ei >= 0), "EI must be non-negative"

    def test_ei_zero_at_zero_std(self):
        gp = self._make_gp(best_y=1.0)
        ei = gp._expected_improvement(np.array([0.5]), np.array([0.0]))
        assert np.allclose(ei, 0.0)

    def test_ei_larger_for_better_mean(self):
        """Lower predicted mean → higher EI (better chance of improving on best_y)."""
        gp = self._make_gp(best_y=1.0)
        ei_good = gp._expected_improvement(np.array([0.5]), np.array([0.2]))[0]
        ei_bad = gp._expected_improvement(np.array([1.5]), np.array([0.2]))[0]
        assert ei_good > ei_bad

    def test_pi_in_unit_interval(self):
        gp = self._make_gp(best_y=1.0)
        y_pred = np.linspace(-2, 4, 20)
        y_std = np.full(20, 0.5)
        pi = gp._probability_of_improvement(y_pred, y_std)
        assert np.all(pi >= 0) and np.all(pi <= 1)

    def test_pi_larger_for_better_mean(self):
        gp = self._make_gp(best_y=1.0)
        pi_good = gp._probability_of_improvement(np.array([0.5]), np.array([0.2]))[0]
        pi_bad = gp._probability_of_improvement(np.array([1.5]), np.array([0.2]))[0]
        assert pi_good > pi_bad


# ---------------------------------------------------------------------------
# Harness strategy registry tests
# ---------------------------------------------------------------------------


class TestHarnessStrategyRegistry:
    """Verify new BayesOpt strategies appear in standard / full harness modes."""

    def test_bayesopt_gp_in_standard_strategies(self):
        from panobbgo.harness import _make_standard_strategies

        names = [s.name for s in _make_standard_strategies()]
        assert "BayesOpt_GP" in names, (
            "BayesOpt_GP must be registered in standard harness strategies"
        )

    def test_bayesopt_enhanced_in_full_strategies(self):
        from panobbgo.harness import _make_full_strategies

        names = [s.name for s in _make_full_strategies()]
        assert "BayesOpt_Enhanced" in names, (
            "BayesOpt_Enhanced must be registered in full harness strategies"
        )

    def test_quick_strategies_unchanged(self):
        """Quick mode should still have exactly two strategies (no GP overhead)."""
        from panobbgo.harness import _make_quick_strategies

        specs = _make_quick_strategies()
        assert len(specs) == 2
        names = [s.name for s in specs]
        assert "RoundRobin_Random" in names
        assert "Rewarding_Diverse" in names

    def test_bayesopt_gp_uses_gp_heuristic(self):
        from panobbgo.harness import _make_standard_strategies

        specs = {s.name: s for s in _make_standard_strategies()}
        bayes = specs["BayesOpt_GP"]
        heuristic_classes = [h for h, _ in bayes.heuristics]
        assert GaussianProcessHeuristic in heuristic_classes, (
            "BayesOpt_GP must include GaussianProcessHeuristic"
        )

    def test_bayesopt_enhanced_uses_gp_and_de(self):
        from panobbgo.harness import _make_full_strategies
        from panobbgo.heuristics import DifferentialEvolution

        specs = {s.name: s for s in _make_full_strategies()}
        enhanced = specs["BayesOpt_Enhanced"]
        heuristic_classes = [h for h, _ in enhanced.heuristics]
        assert GaussianProcessHeuristic in heuristic_classes
        assert DifferentialEvolution in heuristic_classes

    def test_standard_contains_all_quick_strategies(self):
        """Standard mode must include all quick strategies (backward compatibility)."""
        from panobbgo.harness import _make_quick_strategies, _make_standard_strategies

        quick_names = {s.name for s in _make_quick_strategies()}
        standard_names = {s.name for s in _make_standard_strategies()}
        assert quick_names.issubset(standard_names), (
            "All quick strategies should be present in standard mode"
        )

    def test_full_contains_all_standard_strategies(self):
        """Full mode must include all standard strategies."""
        from panobbgo.harness import _make_standard_strategies, _make_full_strategies

        std_names = {s.name for s in _make_standard_strategies()}
        full_names = {s.name for s in _make_full_strategies()}
        assert std_names.issubset(full_names), (
            "All standard strategies should be present in full mode"
        )
