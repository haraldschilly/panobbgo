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
constrained optimization problems, preferably using Expected Improvement with Constraints (EIC).
"""

import numpy as np
from panobbgo.utils import PanobbgoTestCase
from panobbgo.lib.classic import RosenbrockConstraint
from panobbgo.heuristics.gaussian_process import GaussianProcessHeuristic
from panobbgo.strategies.rewarding import StrategyRewarding
from panobbgo.lib.constraints import PenaltyConstraintHandler


class TestGaussianProcessConstraints(PanobbgoTestCase):

    def test_gp_activates_eic_with_constraints(self):
        """
        Test that GP heuristic activates EIC mode when constraints are present
        and stores raw fx and cv separately.
        """
        # Setup constrained problem
        problem = RosenbrockConstraint(dims=2)
        strategy = StrategyRewarding(problem, testing_mode=True)
        strategy.config.max_eval = 50

        # Initialize GP Heuristic
        gp = GaussianProcessHeuristic(strategy)
        gp.__start__()

        # Create some results:
        # 1. Feasible point
        from panobbgo.lib import Point, Result

        r1 = Result(Point(np.array([1.0, 1.0], dtype=float), "init"), 10.0,
                    cv_vec=np.array([0.0]))
        # 2. Infeasible point
        r2 = Result(Point(np.array([0.0, 0.0], dtype=float), "init"), 0.0,
                    cv_vec=np.array([1.0]))

        # Feed results to GP
        gp.on_new_results([r1, r2])

        # Verify EIC activation
        self.assertTrue(gp._use_eic, "EIC should be activated when cv > 0 is seen")

        # Verify data storage
        # y_train should store raw fx: [10.0, 0.0]
        # y_cv_train should store cv: [0.0, 1.0]

        self.assertEqual(len(gp.y_train), 2)
        self.assertEqual(len(gp.y_cv_train), 2)

        self.assertAlmostEqual(gp.y_train[0], 10.0)
        self.assertAlmostEqual(gp.y_train[1], 0.0) # Raw fx, NOT penalized

        self.assertAlmostEqual(gp.y_cv_train[0], 0.0)
        self.assertAlmostEqual(gp.y_cv_train[1], 1.0)

    def test_gp_fits_constraint_model(self):
        """
        Test that GP fits a constraint model when enough data is available.
        """
        problem = RosenbrockConstraint(dims=2)
        strategy = StrategyRewarding(problem, testing_mode=True)
        gp = GaussianProcessHeuristic(strategy)
        gp.__start__()

        from panobbgo.lib import Point, Result
        results = []
        # Add 5 points to trigger fitting (threshold is 3)
        for i in range(5):
             cv = 1.0 if i % 2 == 0 else 0.0
             r = Result(Point(np.array([float(i), 1.0], dtype=float), "init"), 10.0,
                        cv_vec=np.array([cv]))
             results.append(r)

        gp.on_new_results(results)

        # Check if constraint model exists
        self.assertIsNotNone(gp.gp_constraint, "Constraint GP model should be initialized")

        # Check if we can predict
        x_test = np.array([[0.5, 0.5]])
        try:
            pred_cv, std_cv = gp.gp_constraint.predict(x_test, return_std=True)
            self.assertEqual(len(pred_cv), 1)
        except Exception as e:
            self.fail(f"Prediction with constraint model failed: {e}")

    def test_gp_fallback_behavior(self):
        """
        Test that GP uses scalarization/penalty if no constraints are violated (EIC not active).
        """
        problem = RosenbrockConstraint(dims=2)
        strategy = StrategyRewarding(problem, testing_mode=True)
        handler = PenaltyConstraintHandler(strategy, rho=100.0)
        strategy.constraint_handler = handler

        gp = GaussianProcessHeuristic(strategy)
        gp.__start__()

        from panobbgo.lib import Point, Result
        # Only feasible points, so cv=0 everywhere
        r1 = Result(Point(np.array([1.0, 1.0], dtype=float), "init"), 10.0,
                    cv_vec=np.array([0.0]))
        r2 = Result(Point(np.array([0.0, 0.0], dtype=float), "init"), 5.0,
                    cv_vec=np.array([0.0]))

        gp.on_new_results([r1, r2])

        # EIC should NOT be active because no cv > 0 seen
        # Note: If cv is exactly 0 everywhere, we don't strictly need EIC,
        # but current implementation activates EIC if ANY cv > 0 is seen in history.
        # Here we only have 0s.

        self.assertFalse(gp._use_eic, "EIC should not activate if all cv=0")

        # y_train should use "val = get_val(r)" which is fx + penalty
        # Since cv=0, fx + penalty = fx.
        # But conceptually it follows the "else" branch using get_val.

        self.assertAlmostEqual(gp.y_train[0], 10.0)
        self.assertAlmostEqual(gp.y_train[1], 5.0)

        # y_cv_train is populated in both branches in new implementation?
        # Let's check implementation:
        # "new_cv_list.append(cv)" is unconditional inside the loop.
        # "self.y_cv_train.extend(new_cv_list)" is unconditional.
        self.assertEqual(len(gp.y_cv_train), 2)
