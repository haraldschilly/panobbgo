# -*- coding: utf8 -*-
import unittest
from unittest import mock
import numpy as np
from panobbgo.utils import PanobbgoTestCase
from panobbgo.heuristics.constraint_gradient import ConstraintGradient
from panobbgo.lib import Point, Result, Problem
from panobbgo.lib.constraints import DefaultConstraintHandler

class TestConstraintGradient(PanobbgoTestCase):

    def setUp(self):
        super().setUp()
        self.strategy.constraint_handler = DefaultConstraintHandler(self.strategy)
        self.heuristic = ConstraintGradient(self.strategy)
        # Mock strategy.results as a list or object with results attribute
        self.strategy.results = []
        self.heuristic.__start__()

    def test_init(self):
        self.assertEqual(self.heuristic.name, "ConstraintGradient")

    def test_on_new_best_feasible(self):
        """Should do nothing if best is feasible"""
        best = Result(Point(np.zeros(2), "test"), 0.0, cv_vec=None) # cv=0
        self.heuristic.on_new_best(best)

        points = self.heuristic.get_points()
        self.assertEqual(len(points), 0)

    def test_on_new_best_infeasible_no_history(self):
        """Should do nothing if history is empty (cannot estimate gradient)"""
        best = Result(Point(np.zeros(2), "test"), 0.0, cv_vec=np.array([1.0]))
        self.heuristic.on_new_best(best)

        points = self.heuristic.get_points()
        self.assertEqual(len(points), 0)

    def test_gradient_descent_generation(self):
        """
        Mock history to simulate a gradient in constraint violation.
        CV = x[0] + x[1] + 1.0 (Linear)
        Gradient should be [1, 1]. Descent direction [-1, -1].
        """
        # Override problem to have symmetric bounds around 0
        class SimpleProblem(Problem):
             def __init__(self):
                 super().__init__([(-5, 5), (-5, 5)])
             def eval(self, x): return 0.0

        self.problem = SimpleProblem()
        self.strategy.problem = self.problem

        dim = 2

        # Mock history points around (0,0)
        # CV(x) = x[0] + x[1] + 1.0
        history = []
        # Create a grid of points
        for x1 in np.linspace(-0.1, 0.1, 5):
            for x2 in np.linspace(-0.1, 0.1, 5):
                if x1 == 0 and x2 == 0: continue # Skip center
                x = np.array([x1, x2])
                cv_val = np.sum(x) + 1.0
                r = Result(Point(x, "hist"), 0.0, cv_vec=np.array([cv_val]))
                history.append(r)

        self.strategy.results = history

        # Best point at (0,0) with CV=1.0
        best = Result(Point(np.zeros(2), "best"), 0.0, cv_vec=np.array([1.0]))

        self.heuristic.on_new_best(best)

        points = self.heuristic.get_points()
        self.assertEqual(len(points), 1, f"Expected 1 point, got {len(points)}")

        p = points[0]
        # Check direction
        # Gradient of CV = x0 + x1 + 1 is [1, 1].
        # Descent direction is [-1, -1].
        # Generated point should be x_best + step * descent.
        # x_best = [0, 0].
        # So p.x should be negative in both dimensions

        self.assertTrue(p.x[0] < 0, f"Expected negative step in x[0], got {p.x[0]}")
        self.assertTrue(p.x[1] < 0, f"Expected negative step in x[1], got {p.x[1]}")

    def test_gradient_descent_with_results_object(self):
        """
        Test ConstraintGradient with a mock Results object (mimicking production)
        """
        # Override problem
        class SimpleProblem(Problem):
             def __init__(self):
                 super().__init__([(-5, 5), (-5, 5)])
             def eval(self, x): return 0.0

        self.problem = SimpleProblem()
        self.strategy.problem = self.problem

        # Mock Results object
        class MockResults:
            def get_history(self, n=None):
                # CV(x) = x[0] + x[1] + 1.0
                # Generate simple history around origin
                x_list = []
                cv_list = []
                for x1 in np.linspace(-0.1, 0.1, 5):
                    for x2 in np.linspace(-0.1, 0.1, 5):
                        if x1 == 0 and x2 == 0: continue
                        x = np.array([x1, x2])
                        x_list.append(x)
                        cv_list.append(np.sum(x) + 1.0)

                return {
                    'x': np.array(x_list),
                    'cv': np.array(cv_list),
                    'fx': np.zeros(len(x_list)), # Not used
                    'cv_vec': np.array(cv_list).reshape(-1, 1), # Not used
                    'who': np.array(["hist"] * len(x_list))
                }

            def __len__(self): return 24

        self.strategy.results = MockResults()

        # Best point at (0,0) with CV=1.0
        best = Result(Point(np.zeros(2), "best"), 0.0, cv_vec=np.array([1.0]))

        self.heuristic.on_new_best(best)

        points = self.heuristic.get_points()
        self.assertEqual(len(points), 1)

        p = points[0]
        # Should descent (negative step)
        self.assertTrue(p.x[0] < 0)
        self.assertTrue(p.x[1] < 0)
