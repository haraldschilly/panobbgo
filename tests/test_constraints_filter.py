# -*- coding: utf8 -*-
import unittest
from panobbgo.utils import PanobbgoTestCase
from panobbgo.lib.constraints import FilterConstraintHandler
from panobbgo.lib import Result, Point
import numpy as np

# Define MockStrategy locally since it is not exported from utils
class MockConfig:
    def get_logger(self, name):
        return MockLogger()

class MockLogger:
    def info(self, msg): pass
    def debug(self, msg): pass
    def warning(self, msg): pass

class MockStrategy:
    def __init__(self, problem=None):
        self.problem = problem
        self.config = MockConfig()
        self.results = None
        self.eventbus = None
        self.best = None

class TestFilterConstraintHandler(PanobbgoTestCase):

    def setUp(self):
        self.strategy = MockStrategy()
        self.handler = FilterConstraintHandler(self.strategy)

    def test_initial_filter(self):
        self.assertEqual(len(self.handler.filter), 0)

    def test_filter_dominance(self):
        # Result 1: fx=10, cv=0 (feasible)
        p1 = Point(np.array([1.]), "t1")
        r1 = Result(p1, 10.0, cv_vec=None)

        added = self.handler._update_filter(r1)
        self.assertTrue(added)
        self.assertEqual(len(self.handler.filter), 1)

        # Result 2: fx=12, cv=0 (dominated by r1)
        p2 = Point(np.array([2.]), "t2")
        r2 = Result(p2, 12.0, cv_vec=None)

        added = self.handler._update_filter(r2)
        self.assertFalse(added)
        self.assertEqual(len(self.handler.filter), 1)

        # Result 3: fx=8, cv=0 (dominates r1)
        p3 = Point(np.array([3.]), "t3")
        r3 = Result(p3, 8.0, cv_vec=None)

        added = self.handler._update_filter(r3)
        self.assertTrue(added)
        # r1 should be removed
        self.assertEqual(len(self.handler.filter), 1)
        self.assertEqual(self.handler.filter[0][0], 8.0) # fx

    def test_infeasible_tradeoff(self):
        # Start with feasible fx=100
        p1 = Point(np.array([1.]), "t1")
        r1 = Result(p1, 100.0, cv_vec=np.array([0.]))
        self.handler._update_filter(r1)

        # Infeasible but better fx: fx=50, cv=10
        # Not dominated by r1 (fx=100, cv=0) because fx is better
        # r1 dominates it in cv, but not fx.
        p2 = Point(np.array([2.]), "t2")
        r2 = Result(p2, 50.0, cv_vec=np.array([10.]))

        added = self.handler._update_filter(r2)
        self.assertTrue(added)
        self.assertEqual(len(self.handler.filter), 2)

        # Infeasible, same fx as r2, worse cv: fx=50, cv=20
        # Dominated by r2
        p3 = Point(np.array([3.]), "t3")
        r3 = Result(p3, 50.0, cv_vec=np.array([20.]))

        added = self.handler._update_filter(r3)
        self.assertFalse(added)

        # Infeasible, worse fx than r2, better cv: fx=60, cv=5
        # Not dominated by r2 (50, 10). Not dominated by r1 (100, 0)
        p4 = Point(np.array([4.]), "t4")
        r4 = Result(p4, 60.0, cv_vec=np.array([5.]))

        added = self.handler._update_filter(r4)
        self.assertTrue(added)
        self.assertEqual(len(self.handler.filter), 3)

    def test_calculate_improvement(self):
        # r1: fx=10, cv=0
        p1 = Point(np.array([1.]), "t1")
        r1 = Result(p1, 10.0, cv_vec=None)
        self.handler._update_filter(r1)

        # r2: fx=5, cv=0 (dominates r1)
        p2 = Point(np.array([2.]), "t2")
        r2 = Result(p2, 5.0, cv_vec=None)

        improv = self.handler.calculate_improvement(r1, r2)
        self.assertGreater(improv, 0)

        # r3: fx=15, cv=0 (dominated)
        p3 = Point(np.array([3.]), "t3")
        r3 = Result(p3, 15.0, cv_vec=None)

        improv = self.handler.calculate_improvement(r1, r3)
        self.assertEqual(improv, 0)

    def test_is_better_preserves_feasibility_preference(self):
        # r1: fx=10, cv=0
        p1 = Point(np.array([1.]), "t1")
        r1 = Result(p1, 10.0, cv_vec=np.array([0.]))

        # r2: fx=5, cv=10 (infeasible but better fx)
        # Filter would accept both (pareto front)
        # But is_better (for global best) should prefer feasible
        p2 = Point(np.array([2.]), "t2")
        r2 = Result(p2, 5.0, cv_vec=np.array([10.]))

        self.assertTrue(self.handler.is_better(r2, r1)) # r1 (feasible) > r2 (infeasible)
        self.assertFalse(self.handler.is_better(r1, r2)) # r2 not better than r1

if __name__ == "__main__":
    unittest.main()
