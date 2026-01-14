# -*- coding: utf8 -*-
import unittest
import numpy as np
from panobbgo.lib.classic import RosenbrockConstraint
from panobbgo.strategies.rewarding import StrategyRewarding
from panobbgo.heuristics.random import Random
from panobbgo.heuristics.feasible_search import FeasibleSearch
from panobbgo.heuristics.nearby import Nearby
from panobbgo.config import Config
from panobbgo.lib import Point

class TestConstraintsIntegration(unittest.TestCase):

    def setUp(self):
        # Get the singleton config
        self.config = Config(testing_mode=True)
        # Reset relevant config values to defaults
        self.config.constraint_handler = "DefaultConstraintHandler"
        self.config.rho = 100.0
        self.config.constraint_exponent = 1.0
        self.config.dynamic_penalty_rate = 0.01
        self.config.evaluation_method = "threaded"
        self.config.stop_on_convergence = False

    def run_optimization(self, handler_name, **kwargs):
        # Configure handler
        self.config.constraint_handler = handler_name
        for k, v in kwargs.items():
            setattr(self.config, k, v)

        # Problem: RosenbrockConstraint (min at (1,1) is infeasible)
        # Constraint: |x1 - x0| >= 0.5
        # Feasible region is split.
        problem = RosenbrockConstraint(dims=2)

        # Strategy
        # Increase max_eval to ensure convergence
        self.config.max_eval = 2000
        strategy = StrategyRewarding(problem, testing_mode=True)

        # Add heuristics
        strategy.add(Random)
        strategy.add(Nearby)
        strategy.add(FeasibleSearch) # Important for constraints

        # Start optimization
        try:
            strategy.start()
        finally:
            strategy._cleanup()

        return strategy.best

    def test_alm_optimization(self):
        """Test Augmented Lagrangian Method"""
        # Lower rate to be more stable, start with decent rho
        best = self.run_optimization(
            "AugmentedLagrangianConstraintHandler",
            rho=10.0,
            rate=1.5
        )
        self.assertIsNotNone(best)
        print(f"ALM Best: fx={best.fx}, cv={best.cv}, x={best.x}")

        # Check feasibility
        self.assertAlmostEqual(best.cv, 0.0, delta=1e-3, msg="Result should be feasible")

        # Check objective value
        # Unconstrained min is -50. Constrained should be somewhat higher but still negative ideally.
        # But definitely much better than initial random points (often > 100)
        self.assertLess(best.fx, 20.0, msg="Should minimize the function reasonably well")

    def test_penalty_optimization(self):
        """Test Static Penalty Method"""
        best = self.run_optimization(
            "PenaltyConstraintHandler",
            rho=1000.0, # Needs to be high enough to force feasibility
            constraint_exponent=2.0
        )
        self.assertIsNotNone(best)
        print(f"Penalty Best: fx={best.fx}, cv={best.cv}, x={best.x}")

        # Static penalty might have slight violation
        self.assertLess(best.cv, 1e-2, msg="Result should be feasible")
        self.assertLess(best.fx, 20.0)

    def test_dynamic_penalty_optimization(self):
        """Test Dynamic Penalty Method"""
        best = self.run_optimization(
            "DynamicPenaltyConstraintHandler",
            rho=10.0,
            dynamic_penalty_rate=0.05,
            constraint_exponent=2.0
        )
        self.assertIsNotNone(best)
        print(f"Dynamic Penalty Best: fx={best.fx}, cv={best.cv}, x={best.x}")

        self.assertAlmostEqual(best.cv, 0.0, delta=1e-3, msg="Result should be feasible")
        self.assertLess(best.fx, 20.0)

if __name__ == '__main__':
    unittest.main()
