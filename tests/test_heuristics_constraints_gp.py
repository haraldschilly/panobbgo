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
Tests for Gaussian Process Heuristic with Constraints
=====================================================

These tests verify that the Gaussian Process heuristic correctly handles
constrained optimization problems by using Expected Improvement with Constraints (EIC)
or falling back to penalized objective function.
"""

import numpy as np
from panobbgo.utils import PanobbgoTestCase
from panobbgo.lib.classic import RosenbrockConstraint
from panobbgo.heuristics.gaussian_process import GaussianProcessHeuristic
from panobbgo.strategies.rewarding import StrategyRewarding
from panobbgo.lib.constraints import PenaltyConstraintHandler


class TestGaussianProcessConstraints(PanobbgoTestCase):

    def test_gp_uses_eic_default(self):
        """
        Test that GP heuristic uses EIC (raw fx + constraint model) by default
        when constraints are present.
        """
        # Setup constrained problem
        problem = RosenbrockConstraint(dims=2)
        strategy = StrategyRewarding(problem, testing_mode=True)
        strategy.config.max_eval = 50

        # Use Penalty Handler explicitly
        handler = PenaltyConstraintHandler(strategy, rho=100.0)
        strategy.constraint_handler = handler

        # Initialize GP Heuristic (EIC enabled by default)
        gp = GaussianProcessHeuristic(strategy)
        gp.__start__()

        # Create some results:
        # 1. Feasible point: fx=10.0, cv=0.0
        # 2. Infeasible point: fx=0.0, cv=1.0. Penalty = 0.0 + 100*1 = 100.0
        from panobbgo.lib import Point, Result

        r1 = Result(Point(np.array([1.0, 1.0], dtype=float), "init"), 10.0,
                    cv_vec=np.array([0.0]))
        r2 = Result(Point(np.array([0.0, 0.0], dtype=float), "init"), 0.0,
                    cv_vec=np.array([1.0]))
        r3 = Result(Point(np.array([0.5, 0.5], dtype=float), "init"), 5.0,
                    cv_vec=np.array([0.5]))

        # Feed results to GP
        gp.on_new_results([r1, r2, r3])

        # Check internal state for EIC
        # y_train should store raw fx: [10.0, 0.0, 5.0]
        if gp.y_train is not None:
            print(f"GP y_train: {gp.y_train}")
            assert np.isclose(gp.y_train[0], 10.0), "First point fx mismatch"
            assert np.isclose(gp.y_train[1], 0.0), "Second point fx mismatch (should be raw fx)"

        # Check constraint model
        assert gp.gp_constraint is not None, "Constraint model should be initialized"
        assert gp.y_cv_train is not None
        assert np.isclose(gp.y_cv_train[0], 0.0)
        assert np.isclose(gp.y_cv_train[1], 1.0)

    def test_gp_fallback_no_constraints(self):
        """
        Test that GP falls back to coupled behavior (or just normal EI)
        if no constraints are observed (all cv=0).
        """
        problem = RosenbrockConstraint(dims=2) # Constrained problem
        strategy = StrategyRewarding(problem, testing_mode=True)
        handler = PenaltyConstraintHandler(strategy, rho=100.0)
        strategy.constraint_handler = handler

        gp = GaussianProcessHeuristic(strategy)
        gp.__start__()

        # Only feasible points
        from panobbgo.lib import Point, Result
        r1 = Result(Point(np.array([1.0, 1.0]), "init"), 10.0, cv_vec=np.array([0.0]))
        r2 = Result(Point(np.array([1.0, 1.0]), "init"), 5.0, cv_vec=np.array([0.0]))
        r3 = Result(Point(np.array([1.0, 1.0]), "init"), 2.0, cv_vec=np.array([0.0]))

        gp.on_new_results([r1, r2, r3])

        # Expect EIC to NOT be active because has_constraints check requires non-zero CV
        # So y_train should be penalized values (which equal fx here)
        # And gp_constraint should be None

        # Note: Implementation detail check
        # has_constraints = np.any(self.y_cv_train > 1e-6) -> False

        assert gp.gp_constraint is None, "Constraint model should be None if no violations observed"
        assert np.array_equal(gp.y_train, gp.y_fx_train), "y_train should match fx (penalized is same as fx here)"

    def test_gp_disabled_eic(self):
        """
        Test that GP uses coupled penalty values when EIC is explicitly disabled.
        """
        problem = RosenbrockConstraint(dims=2)
        strategy = StrategyRewarding(problem, testing_mode=True)
        handler = PenaltyConstraintHandler(strategy, rho=100.0)
        strategy.constraint_handler = handler

        # Initialize GP Heuristic with EIC DISABLED
        gp = GaussianProcessHeuristic(strategy, enable_eic=False)
        gp.__start__()

        from panobbgo.lib import Point, Result
        r1 = Result(Point(np.array([1.0, 1.0]), "init"), 10.0, cv_vec=np.array([0.0]))
        # Infeasible: fx=0, cv=1 -> Penalty=100
        r2 = Result(Point(np.array([0.0, 0.0]), "init"), 0.0, cv_vec=np.array([1.0]))

        gp.on_new_results([r1, r2])

        # Check internal state for Coupled Penalty
        # y_train should store penalized fx: [10.0, 100.0]
        if gp.y_train is not None:
            print(f"GP y_train (Disabled EIC): {gp.y_train}")
            assert np.isclose(gp.y_train[0], 10.0), "First point penalty mismatch"
            assert np.isclose(gp.y_train[1], 100.0), "Second point penalty mismatch"

        # Check constraint model
        assert gp.gp_constraint is None, "Constraint model should be None"
