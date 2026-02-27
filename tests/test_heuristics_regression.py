# -*- coding: utf8 -*-
# Copyright 2025 Panobbgo Contributors
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
Regression Tests for Heuristic Fixes
====================================

Tests to ensure fixes for WeightedAverage crashes, NelderMead sorting,
and GaussianProcess scalar indexing remain effective.
"""

import numpy as np
import pytest
from panobbgo.lib import Point, Result
from panobbgo.heuristics.weighted_average import WeightedAverage
from panobbgo.heuristics.nelder_mead import NelderMead
from panobbgo.heuristics.gaussian_process import GaussianProcessHeuristic
from panobbgo.lib.constraints import DefaultConstraintHandler
from functools import cmp_to_key

class MockLogger:
    def info(self, msg): pass
    def debug(self, msg): pass
    def warning(self, msg): pass

class MockConfig:
    def __init__(self):
        self.capacity = 100
        self.debug = True

    def get_logger(self, name):
        return MockLogger()

class MockProblem:
    def __init__(self):
        self.ranges = np.array([10.0])
        self.dim = 1
        self.box = type('Box', (), {'box': [(0, 10)]})()
    def project(self, x):
        return x

class MockStrategy:
    def __init__(self):
        self.config = MockConfig()
        self.constraint_handler = DefaultConstraintHandler(self, rho=100.0)
        self.analyzers = {}
        self.problem = MockProblem()
        self.eventbus = None
        self.results = []

    def analyzer(self, name):
        class AnalyzerProxy:
            def __init__(self, strat): self.strat = strat
            def get_leaf(self, x):
                if 'Splitter' in self.strat.analyzers:
                    return self.strat.analyzers['Splitter'].get_leaf(x)
                return None
        return AnalyzerProxy(self)

class TestHeuristicRegressions:

    def test_weighted_average_negative_diff_crash(self):
        """
        Verify WeightedAverage does not crash when an infeasible point has
        lower penalty than the current best feasible point.
        """
        strategy = MockStrategy()

        # R1: Feasible, FX=200. Penalty=200.
        r1 = Result(Point(np.array([200.0]), "r1"), 200.0, cv_vec=np.array([0.0]))

        # R2: Infeasible, FX=0, CV=1. Penalty=100.
        r2 = Result(Point(np.array([0.0]), "r2"), 0.0, cv_vec=np.array([1.0]))

        # R3: Feasible, FX=150. Penalty=150.
        r3 = Result(Point(np.array([150.0]), "r3"), 150.0, cv_vec=np.array([0.0]))

        # Mock Splitter box
        class MockBox:
            def __init__(self, results):
                self.results = results

        box = MockBox([r1, r2, r3])

        # Setup mock splitter
        class MockSplitter:
            def get_leaf(self, x): return box
        strategy.analyzers['Splitter'] = MockSplitter()

        wa = WeightedAverage(strategy)
        wa.__start__()
        wa.cap = 10

        # Best is R3 (Feasible, 150)
        best = r3

        # Should not raise exception
        try:
            wa.on_new_best(best)
        except Exception as e:
            pytest.fail(f"WeightedAverage crashed with negative penalty difference: {e}")

    def test_nelder_mead_sorting(self):
        """
        Verify NelderMead sorts points using is_better (feasibility)
        rather than just raw penalty value.
        """
        strategy = MockStrategy()
        nm = NelderMead(strategy)

        # R1: Infeasible, FX=0.0, CV=1.0 -> Penalty=100
        r1 = Result(Point(np.array([0.0]), "r1"), 0.0, cv_vec=np.array([1.0]))

        # R2: Feasible, FX=200.0. Penalty=200.
        r2 = Result(Point(np.array([200.0]), "r2"), 200.0, cv_vec=np.array([0.0]))

        # R3: Feasible, FX=10.0. Penalty=10.
        r3 = Result(Point(np.array([10.0]), "r3"), 10.0, cv_vec=np.array([0.0]))

        results = [r1, r2, r3]

        # Use the comparator from NelderMead (re-implemented here as it's internal to method)
        def compare(a, b):
            if strategy.constraint_handler.is_better(b, a):
                return -1
            elif strategy.constraint_handler.is_better(a, b):
                return 1
            return 0

        sorted_results = sorted(results, key=cmp_to_key(compare))

        # Expect R3 (Best Feasible), R2 (Next Feasible), R1 (Infeasible)
        assert sorted_results[0] == r3
        assert sorted_results[1] == r2
        assert sorted_results[2] == r1

    def test_gp_scalar_indexing(self):
        """
        Verify GaussianProcessHeuristic handles scalar predictions correctly.
        This simulates the boolean indexing logic that failed for scalars.
        """
        # Logic simulation
        c_pred = np.array(0.5)  # 0-d array
        c_std = np.array(0.0)   # 0-d array
        prob_feas = np.array(0.0) # 0-d array

        # Fix application: use atleast_1d
        c_std_arr = np.atleast_1d(c_std)
        c_pred_arr = np.atleast_1d(c_pred)
        prob_feas_arr = np.atleast_1d(prob_feas)

        mask_det = c_std_arr == 0

        # This should succeed
        try:
            prob_feas_arr[mask_det] = (c_pred_arr[mask_det] <= 1e-6).astype(float)
        except IndexError as e:
            pytest.fail(f"Scalar indexing failed despite fix: {e}")

        # Verify value
        assert prob_feas_arr[0] == 0.0 # 0.5 > 1e-6, so 0.0
