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
constrained optimization problems by optimizing the penalized objective
function.
"""

import numpy as np
from panobbgo.utils import PanobbgoTestCase
from panobbgo.lib.classic import RosenbrockConstraint
from panobbgo.heuristics.gaussian_process import GaussianProcessHeuristic
from panobbgo.strategies.rewarding import StrategyRewarding
from panobbgo.lib.constraints import PenaltyConstraintHandler


class TestGaussianProcessConstraints(PanobbgoTestCase):

    def test_gp_uses_penalty_function(self):
        """
        Test that GP heuristic uses the constraint handler's penalty value.
        """
        # Setup constrained problem
        problem = RosenbrockConstraint(dims=2)
        strategy = StrategyRewarding(problem, testing_mode=True)
        strategy.config.max_eval = 50

        # Use Penalty Handler explicitly
        handler = PenaltyConstraintHandler(strategy, rho=100.0)
        strategy.constraint_handler = handler

        # Initialize GP Heuristic
        gp = GaussianProcessHeuristic(strategy)
        gp.__start__()

        # Create some results:
        # 1. Feasible point with high fx
        # 2. Infeasible point with low fx but high penalty
        from panobbgo.lib import Point, Result

        r1 = Result(Point(np.array([1.0, 1.0], dtype=float), "init"), 10.0,
                    cv_vec=np.array([0.0]))
        # r2 is infeasible. fx=0 (better), but cv=1. Penalty = 100.
        r2 = Result(Point(np.array([0.0, 0.0], dtype=float), "init"), 0.0,
                    cv_vec=np.array([1.0]))

        # If GP uses raw fx, r2 (fx=0) is better than r1 (fx=10).
        # If GP uses penalty, r1 (penalty=10) is better than r2 (penalty=100).

        # Feed results to GP
        gp.on_new_results([r1, r2])

        # We check internal state of GP (y_train) to see what values it stored.
        # This requires white-box testing access to gp.y_train.

        # In current implementation (before fix), it uses r.fx,
        # so y_train will be [10.0, 0.0].
        # After fix, it should be [10.0, 100.0].

        if gp.y_train is not None:
            print(f"GP y_train: {gp.y_train}")

            # Assert that we want penalty values
            # r1 penalty: 10 + 100*0 = 10
            # r2 penalty: 0 + 100*1 = 100

            assert np.isclose(gp.y_train[0], 10.0), \
                "First point penalty mismatch"
            assert np.isclose(gp.y_train[1], 100.0), \
                "Second point penalty mismatch (should use penalty)"
