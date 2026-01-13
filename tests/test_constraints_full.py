# -*- coding: utf8 -*-
import unittest
import numpy as np
from panobbgo.lib.classic import RosenbrockConstraint
from panobbgo.strategies.rewarding import StrategyRewarding
from panobbgo.heuristics.random import Random
from panobbgo.heuristics.feasible_search import FeasibleSearch
from panobbgo.lib import Point

class TestConstraintsIntegration(unittest.TestCase):
    def test_alm_optimization(self):
        """
        Test that StrategyRewarding with AugmentedLagrangianConstraintHandler
        can solve a constrained problem.
        """
        # RosenbrockConstraint: Min at (1,1) with g(x) <= 0
        # Usually constrained by x1^2 + x2^2 <= 2 (disk radius sqrt(2)=1.414)
        # (1,1) has r=sqrt(2)=1.414. So it is on the boundary.

        problem = RosenbrockConstraint(dims=2)

        # Configure for ALM
        # We need to set config via kwargs or by modifying config object?
        # StrategyBase initializes config from args or defaults.
        # We can pass kwargs that config might pick up?
        # StrategyBase uses Config(parse_args).
        # We need to set 'constraint_handler' in config.

        class ConfigMock:
            def __init__(self):
                self.constraint_handler = "AugmentedLagrangianConstraintHandler"
                self.rho = 10.0
                self.constraint_exponent = 2.0
                self.dynamic_penalty_rate = 2.0
                self.max_eval = 200
                self.discount = 0.95
                self.smooth = 0.5
                self.evaluation_method = "threaded"
                self.logging = {}
                self.dask_n_workers = 1
                self.debug = False
                self.capacity = 100
                self.show_interval = 100

            def get_logger(self, name):
                import logging
                return logging.getLogger(name)

        # We can't easily inject ConfigMock into StrategyRewarding because it creates its own Config.
        # However, Config loads from ~/.panobbgo/config.yaml or ./config.yaml.
        # But for tests, we should be able to override?
        # StrategyBase takes **kwargs? No, it takes problem.
        # Ah, StrategyBase.__init__ creates Config(parse_args).

        # Let's try to patch Config.
        from unittest.mock import patch

        with patch('panobbgo.core.Config') as MockConfigClass:
            mock_config = ConfigMock()
            MockConfigClass.return_value = mock_config

            strategy = StrategyRewarding(problem)

            # Add heuristics
            strategy.add(Random)
            strategy.add(FeasibleSearch)

            # Start optimization
            strategy.start()

            # Check results
            best = strategy.best

            print(f"Best: fx={best.fx}, cv={best.cv}, x={best.x}")

            # Expectation:
            # 1. Feasible (cv=0 or very small)
            # 2. Close to optimum (1,1) -> fx=0

            self.assertIsNotNone(best)
            self.assertAlmostEqual(best.cv, 0.0, delta=1e-4)
            self.assertLess(best.fx, 1.0) # Should be reasonably close to 0

if __name__ == '__main__':
    unittest.main()
