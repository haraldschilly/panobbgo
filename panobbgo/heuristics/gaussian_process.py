# -*- coding: utf8 -*-
# Copyright 2012 Harald Schilly <harald.schilly@gmail.com>
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
Gaussian Process Surrogate Heuristic
=====================================

This heuristic uses Gaussian Process regression as a surrogate model
for expensive black-box functions. It employs acquisition functions
to balance exploration and exploitation when selecting new evaluation points.

.. codeauthor:: Assistant
"""

from panobbgo.core import Heuristic
import numpy as np
from enum import Enum
from scipy.stats import norm


class AcquisitionFunction(Enum):
    """Acquisition functions for Bayesian optimization."""

    EI = "expected_improvement"  # Expected Improvement
    UCB = "upper_confidence_bound"  # Upper Confidence Bound
    PI = "probability_of_improvement"  # Probability of Improvement


class GaussianProcessHeuristic(Heuristic):
    """
    Bayesian optimization heuristic using Gaussian Process surrogate models.

    This heuristic builds a GP model from evaluated points and uses acquisition
    functions to select promising new points for evaluation. It responds to
    new results by updating the model and suggesting new points.

    Attributes:
        acquisition_func: The acquisition function to use (EI, UCB, or PI)
        kappa: Exploration parameter for UCB acquisition function
        xi: Exploration parameter for EI and PI acquisition functions
        n_restarts: Number of random restarts for acquisition optimization
        gp_model: The fitted GaussianProcessRegressor model
        X_train: Training points (n_samples, n_features)
        y_train: Training function values (n_samples,)
    """

    def __init__(
        self,
        strategy,
        acquisition_func=AcquisitionFunction.EI,
        kappa=1.96,
        xi=0.01,
        n_restarts=10,
    ):
        """
        Initialize the Gaussian Process heuristic.

        Args:
            strategy: The optimization strategy instance
            acquisition_func: Acquisition function to use
            kappa: Exploration parameter for UCB (default: 1.96 for ~95% confidence)
            xi: Exploration parameter for EI/PI (default: 0.01)
            n_restarts: Number of random restarts for optimization (default: 10)
        """
        super().__init__(strategy)
        self.logger = self.config.get_logger("H:GP")

        # Acquisition function parameters
        self.acquisition_func = acquisition_func
        self.kappa = kappa
        self.xi = xi
        self.n_restarts = n_restarts

        # GP model state
        self.gp_model = None
        self.X_train = None
        self.y_train = None
        self.best_y = np.inf

    def on_start(self):
        """Initialize the heuristic at the start of optimization."""
        self.logger.info("Gaussian Process heuristic started")

    def on_new_results(self, results):
        """
        Update the GP model when new results are available and suggest new points.

        Args:
            results: List of Result objects from recent evaluations
        """
        if not results:
            return

        # Update training data with new results
        new_X = np.array([r.x for r in results])
        new_y = np.array([r.fx for r in results])

        if self.X_train is None:
            self.X_train = new_X
            self.y_train = new_y
        else:
            self.X_train = np.vstack([self.X_train, new_X])
            self.y_train = np.append(self.y_train, new_y)  # type: ignore

        self.best_y = np.min(self.y_train)

        if len(self.y_train) < 3:  # Need at least a few points for meaningful GP
            return

        # Fit GP model
        self._fit_gp_model()

        # Generate new candidate points using acquisition function
        candidates = self._acquire_candidates(n_candidates=5)
        for candidate in candidates:
            self.emit(candidate)

    def _fit_gp_model(self):
        """Fit the Gaussian Process model to current training data."""
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import Matern

            # Use Matern kernel (common choice for optimization)
            kernel = Matern(nu=2.5)
            self.gp_model = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,  # Small regularization
                normalize_y=True,
                n_restarts_optimizer=10,
            )

            if self.X_train is not None:
                self.gp_model.fit(self.X_train, self.y_train)
                self.logger.debug(f"GP model fitted with {len(self.X_train)} points")

        except ImportError:
            self.logger.error(
                "scikit-learn not available. Install with: pip install scikit-learn"
            )
            raise

    def _acquire_candidates(self, n_candidates):
        """
        Generate candidate points using the acquisition function.

        Args:
            n_candidates: Number of candidate points to generate

        Returns:
            Array of candidate points
        """
        candidates = []

        for _ in range(n_candidates):
            # Optimize acquisition function with multiple random starts
            candidate = self._optimize_acquisition()
            if candidate is not None:
                candidates.append(candidate)

        return np.array(candidates)

    def _optimize_acquisition(self):
        """
        Optimize the acquisition function to find the next candidate point.

        Returns:
            Best candidate point found, or None if optimization fails
        """
        from scipy.optimize import minimize

        def acquisition(x):
            """Evaluate acquisition function at point x."""
            x = np.atleast_2d(x)
            return -self._evaluate_acquisition(x)[0]  # Negative for minimization

        # Generate random starting points
        bounds = [(low, high) for low, high in self.problem.box.box]
        starts = []
        for _ in range(self.n_restarts):
            start = np.array([np.random.uniform(low, high) for low, high in bounds])
            starts.append(start)

        # Also try current best point as start
        if self.X_train is not None and self.y_train is not None:
            best_idx = int(np.argmin(self.y_train))
            starts.append(self.X_train[best_idx].copy())

        best_candidate = None
        best_acq_value = -np.inf

        for start in starts:
            try:
                result = minimize(
                    acquisition,
                    start,
                    bounds=bounds,
                    method="L-BFGS-B",
                    options={"maxiter": 100},
                )

                if result.success:
                    acq_value = -result.fun  # Convert back from negative
                    if acq_value > best_acq_value:
                        best_acq_value = acq_value
                        best_candidate = result.x

            except Exception as e:
                self.logger.debug(f"Acquisition optimization failed: {e}")
                continue

        return best_candidate

    def _evaluate_acquisition(self, X):
        """
        Evaluate the acquisition function at given points.

        Args:
            X: Points to evaluate (n_points, n_features)

        Returns:
            Acquisition function values at each point
        """
        if self.gp_model is None:
            length = len(X) if hasattr(X, "__len__") else 1
            return np.zeros(length)

        # Get GP predictions and uncertainties
        y_pred, y_std = self.gp_model.predict(X, return_std=True)

        if self.acquisition_func == AcquisitionFunction.EI:
            return self._expected_improvement(y_pred, y_std)
        elif self.acquisition_func == AcquisitionFunction.UCB:
            return self._upper_confidence_bound(y_pred, y_std)
        elif self.acquisition_func == AcquisitionFunction.PI:
            return self._probability_of_improvement(y_pred, y_std)
        else:
            raise ValueError(f"Unknown acquisition function: {self.acquisition_func}")

    def _expected_improvement(self, y_pred, y_std):
        """
        Expected Improvement acquisition function.

        EI(x) = (μ(x) - f*) * Φ(Z) + σ(x) * φ(Z)
        where Z = (μ(x) - f*) / σ(x)
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            z = (self.best_y - y_pred) / y_std
            ei = (self.best_y - y_pred) * self._norm_cdf(z) + y_std * self._norm_pdf(z)
            ei[y_std == 0] = 0.0  # Handle zero variance
            return ei

    def _upper_confidence_bound(self, y_pred, y_std):
        """Upper Confidence Bound acquisition function."""
        return y_pred - self.kappa * y_std  # Negative for maximization

    def _probability_of_improvement(self, y_pred, y_std):
        """
        Probability of Improvement acquisition function.

        PI(x) = Φ((f* - μ(x)) / σ(x))
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            z = (self.best_y - y_pred) / y_std
            pi = self._norm_cdf(z)
            pi[y_std == 0] = 0.0  # Handle zero variance
            return pi

    @staticmethod
    def _norm_pdf(x):
        """Standard normal probability density function."""
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

    @staticmethod
    def _norm_cdf(x):
        """Standard normal cumulative distribution function."""
        return norm.cdf(x)
