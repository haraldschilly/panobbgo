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

from panobbgo.utils import PanobbgoTestCase
from panobbgo.lib.constraints import AugmentedLagrangianConstraintHandler
from panobbgo.strategies.rewarding import StrategyRewarding
from panobbgo.lib.classic import RosenbrockConstraint
from panobbgo.lib import Point, Result
from panobbgo.analyzers import Best
import numpy as np
import time
import unittest

class TestALMRobustness(PanobbgoTestCase):
    def setUp(self):
        # We need a strategy with ALM handler
        self.problem = RosenbrockConstraint(dims=2)
        # Setup strategy manually to inject handler properly
        self.strategy = StrategyRewarding(self.problem, max_evaluations=100)

        # Configure ALM
        self.alm = AugmentedLagrangianConstraintHandler(
            self.strategy, rho=1.0, rate=2.0, update_interval=2
        )
        self.strategy.constraint_handler = self.alm

        # Register components
        self.strategy.eventbus.register(self.alm)

        # Add Best analyzer manually
        self.best_analyzer = Best(self.strategy)
        self.strategy.add_analyzer(self.best_analyzer)

    def tearDown(self):
        if hasattr(self, 'strategy'):
            self.strategy._cleanup()

    def test_best_updates_on_param_change(self):
        """
        Test that Best analyzer updates the best point when ALM parameters change,
        even if no new points are added.
        """
        # Point A: Feasible but high objective.
        # Point B: Infeasible but low objective.

        # Point A: fx=10, cv=0 (Feasible)
        pA = Point(np.array([1.0, 1.0]), "testA")
        rA = Result(pA, fx=10.0, cv_vec=np.array([-0.1, -0.1])) # Feasible

        # Point B: fx=0, cv=1 (Infeasible)
        pB = Point(np.array([-1.0, -1.0]), "testB")
        rB = Result(pB, fx=0.0, cv_vec=np.array([1.0, 0.0])) # Infeasible

        # Add results to strategy
        # Best analyzer will process them.
        self.strategy.results.add_results([rA])
        time.sleep(0.1) # Wait for threads

        self.strategy.results.add_results([rB])
        time.sleep(0.1)

        # Check initial best. With rho=1.0, B (L=0.5) is better than A (L=10).
        best_initial = self.best_analyzer.best
        self.assertEqual(best_initial.who, "testB")

        # Case where A becomes better than B.
        # Let's set mu manually to 100.
        self.alm.mu = 100.0
        self.alm.lambdas = np.zeros(2)

        # Trigger explicit history scan (simulate param update)
        self.alm._scan_history_for_new_best()
        time.sleep(0.2)

        # With mu=100, L(A)=10 < L(B)=50. So A should be best.
        self.assertEqual(self.best_analyzer.best.who, "testA", "Point A should be best with high penalty")

        # Now reduce mu to 0.1
        self.alm.mu = 0.1
        self.alm._scan_history_for_new_best()
        time.sleep(0.2)

        # With mu=0.1, L(B)=0.05 < L(A)=10. So B should be best.
        self.assertEqual(self.best_analyzer.best.who, "testB", "Point B should be best with low penalty")

if __name__ == "__main__":
    unittest.main()
