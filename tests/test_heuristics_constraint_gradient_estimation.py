# -*- coding: utf8 -*-
import unittest
import numpy as np
from panobbgo.utils import PanobbgoTestCase
from panobbgo.heuristics.constraint_gradient import ConstraintGradient
from panobbgo.lib import Problem, Point, Result
from panobbgo.lib.constraints import DefaultConstraintHandler


class TestConstraintGradientEstimation(PanobbgoTestCase):
    def setUp(self):
        super().setUp()
        self.strategy.constraint_handler = DefaultConstraintHandler(self.strategy)
        # Mock problem
        self.problem = Problem([(-5, 5), (-5, 5)])
        self.strategy.problem = self.problem
        self.heuristic = ConstraintGradient(self.strategy)
        self.heuristic.__start__()

    def test_estimate_gradient_linear(self):
        """
        Verify that _estimate_gradient correctly recovers the gradient of a linear function.
        f(x) = g^T x + c
        """
        dim = 2
        true_gradient = np.array([2.0, -3.0])
        intercept = 5.0

        # Current best point at origin (simplifies centered coordinates)
        best_x = np.array([0.0, 0.0])
        best_val = np.dot(true_gradient, best_x) + intercept  # = 5.0

        # Generate some random points around best
        np.random.seed(42)
        n_points = 10
        X = np.random.randn(n_points, dim)

        # Calculate function values
        y_vals = np.dot(X, true_gradient) + intercept

        # Center points relative to best_x
        X_centered = X - best_x  # X itself since best_x is 0

        # Estimate gradient
        estimated_gradient = self.heuristic._estimate_gradient(X_centered, y_vals, best_val)

        # Check if estimated gradient matches true gradient
        np.testing.assert_allclose(estimated_gradient, true_gradient, rtol=1e-5, atol=1e-5)

    def test_estimate_gradient_quadratic(self):
        """
        Verify that _estimate_gradient approximates the gradient of a quadratic function locally.
        f(x) = x^T A x + b^T x + c
        Gradient at x=0 is b.
        """
        dim = 2
        # Quadratic term: x1^2 + x2^2
        # Linear term: 2*x1 - 3*x2
        true_gradient_at_zero = np.array([2.0, -3.0])

        best_x = np.array([0.0, 0.0])
        best_val = 0.0  # f(0) = 0

        # Generate points very close to 0 to approximate linearity
        n_points = 10
        delta = 1e-4
        X = np.random.randn(n_points, dim) * delta

        # f(x) = x1^2 + x2^2 + 2x1 - 3x2
        y_vals = np.sum(X**2, axis=1) + np.dot(X, true_gradient_at_zero)

        X_centered = X - best_x

        estimated_gradient = self.heuristic._estimate_gradient(X_centered, y_vals, best_val)

        # Should match linear term
        np.testing.assert_allclose(estimated_gradient, true_gradient_at_zero, rtol=1e-2, atol=1e-2)

    def test_estimate_gradient_noisy(self):
        """
        Verify robustness to noise.
        """
        dim = 2
        true_gradient = np.array([1.0, 1.0])
        intercept = 0.0
        best_val = 0.0

        np.random.seed(42)
        n_points = 50
        X = np.random.randn(n_points, dim)

        # Add noise
        noise = np.random.randn(n_points) * 0.1
        y_vals = np.dot(X, true_gradient) + intercept + noise

        X_centered = X  # best is 0

        estimated_gradient = self.heuristic._estimate_gradient(X_centered, y_vals, best_val)

        # With enough points, should be close
        np.testing.assert_allclose(estimated_gradient, true_gradient, rtol=0.2, atol=0.2)
