import unittest
import numpy as np
from panobbgo.lib.lib import Point, Result, BoundingBox, Problem
from panobbgo.lib.constraints import AugmentedLagrangianConstraintHandler
from panobbgo.core import StrategyBase

class MockStrategy:
    def __init__(self, best=None):
        self.best = best
        self.config = MockConfig()

class MockConfig:
    def get_logger(self, name):
        import logging
        return logging.getLogger(name)

class TestAugmentedLagrangianConstraintHandler(unittest.TestCase):
    def test_init(self):
        handler = AugmentedLagrangianConstraintHandler(rho=5.0, rate=1.5, update_interval=10)
        self.assertEqual(handler.mu, 5.0)
        self.assertEqual(handler.rate, 1.5)
        self.assertEqual(handler.update_interval, 10)
        self.assertEqual(handler.counter, 0)
        self.assertIsNone(handler.lambdas)

    def test_update_parameters(self):
        # Setup
        # Create a "best" result that is violated
        # g(x) = cv_vec. If cv_vec > 0, violation.
        cv_vec = np.array([1.0, 0.5])
        best_point = Point(np.array([0.0, 0.0]), "test")
        best_res = Result(best_point, fx=10.0, cv_vec=cv_vec)

        strategy = MockStrategy(best=best_res)
        handler = AugmentedLagrangianConstraintHandler(strategy=strategy, rho=2.0, rate=2.0, update_interval=2)

        # Test 1: First update (triggered by counter)
        # We need to simulate adding results.
        # on_new_results increments counter.

        # Add 1 result
        handler.on_new_results([Result(best_point, 1.0)])
        self.assertEqual(handler.counter, 1)
        self.assertIsNone(handler.lambdas) # No update yet

        # Add another result -> triggers update
        handler.on_new_results([Result(best_point, 1.0)])
        self.assertEqual(handler.counter, 0)

        # Expected Update:
        # Initial lambda = [0, 0]
        # mu = 2.0
        # cv_vec = [1.0, 0.5]
        # new_lambda = max(0, 0 + 2.0 * [1.0, 0.5]) = [2.0, 1.0]
        # mu update: current_cv_norm > 0.9 * inf -> NO (wait, inf is large)
        # last_cv_norm was inf.
        # So first update, mu shouldn't change unless we handle infinity logic?
        # "if current_cv_norm > 0.9 * self.last_cv_norm:" -> 1.x > 0.9 * inf is False.
        # So mu stays 2.0.

        np.testing.assert_array_equal(handler.lambdas, np.array([2.0, 1.0]))
        self.assertEqual(handler.mu, 2.0)

        # Test 2: Second update
        # Assume best result has NOT improved constraint violation
        # best_res is still the same.

        handler.on_new_results([Result(best_point, 1.0)] * 2)

        # Check mu update
        # last_cv_norm was set to current_cv_norm (sqrt(1^2 + 0.5^2) = sqrt(1.25) approx 1.118)
        # current_cv_norm is same.
        # 1.118 > 0.9 * 1.118 ? Yes.
        # So mu should increase by rate (2.0) -> mu = 4.0

        self.assertEqual(handler.mu, 4.0)

        # Check lambda update
        # Standard ALM: Update lambda using OLD mu, then update mu for next step.
        # old lambda = [2.0, 1.0]
        # mu (used for update) = 2.0
        # new_lambda = max(0, [2.0, 1.0] + 2.0 * [1.0, 0.5]) = [4.0, 2.0]
        np.testing.assert_array_equal(handler.lambdas, np.array([4.0, 2.0]))

    def test_calculate_lagrangian(self):
        handler = AugmentedLagrangianConstraintHandler(rho=2.0)
        handler.lambdas = np.array([1.0, 1.0])
        handler.mu = 2.0

        # Result
        cv_vec = np.array([0.5, -0.5]) # First violated (if we consider > -lambda/mu), second feasible
        res = Result(Point(np.zeros(2), "test"), fx=10.0, cv_vec=cv_vec)

        # L = f(x) + (1/2mu) * sum( max(0, lambda + mu*g)^2 - lambda^2 )
        # term = max(0, [1, 1] + 2*[0.5, -0.5]) = max(0, [1+1, 1-1]) = [2.0, 0.0]
        # penalty_term = (1/4) * ( (2^2 + 0^2) - (1^2 + 1^2) )
        #              = 0.25 * (4 - 2) = 0.5
        # L = 10.0 + 0.5 = 10.5

        L = handler._calculate_lagrangian(res)
        self.assertAlmostEqual(L, 10.5)

    def test_calculate_improvement(self):
        handler = AugmentedLagrangianConstraintHandler(rho=2.0)
        handler.lambdas = np.array([0.0, 0.0])

        # old best: L = 10.5
        # new best: L = 10.0
        # improvement = 0.5

        # Mocking _calculate_lagrangian is hard without mocking the method itself,
        # so let's rely on logic.

        res1 = Result(Point(np.zeros(2), "test"), fx=10.0, cv_vec=np.array([1.0, 1.0]))
        # With lambda=0, mu=2:
        # term = max(0, 0 + 2*[1,1]) = [2, 2]
        # penalty = (1/4) * ( (4+4) - 0 ) = 2.0
        # L1 = 12.0

        res2 = Result(Point(np.zeros(2), "test"), fx=9.0, cv_vec=np.array([1.0, 1.0]))
        # L2 = 11.0

        improv = handler.calculate_improvement(res1, res2)
        self.assertAlmostEqual(improv, 1.0)

        # Regression (L increases)
        improv = handler.calculate_improvement(res2, res1)
        self.assertEqual(improv, 0.0)

if __name__ == '__main__':
    unittest.main()
