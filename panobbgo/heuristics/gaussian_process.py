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
from scipy.special import ndtr
from scipy.optimize import minimize


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
            kappa: Exploration param for UCB (default: 1.96, ~95% confidence)
            xi: Exploration parameter for EI/PI (default: 0.01)
            n_restarts: Number of random restarts for opt (default: 10)
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
        self.X_train = []
        self.y_train = []
        self.best_y = np.inf

    def on_start(self):
        """Initialize the heuristic at the start of optimization."""
        self.logger.info("Gaussian Process heuristic started")

    def on_new_results(self, results):
        """
        Update GP model when new results are available.

        Args:
            results: List of Result objects
        """
        if not results:
            return

        # Use constraint handler to get penalty values (fx + penalty)
        # This handles constrained optimization by scalarizing the problem
        get_val = self.strategy.constraint_handler.get_penalty_value

        new_X_list = []
        new_y_list = []

        for r in results:
            val = get_val(r)
            # Filter out invalid values (inf, NaN) which can break GP
            if np.isfinite(val):
                new_X_list.append(r.x)
                new_y_list.append(val)

        if not new_X_list:
            return

        self.X_train.extend(new_X_list)
        self.y_train.extend(new_y_list)

        self.best_y = min(self.best_y, min(new_y_list))

        if len(self.y_train) < 3:
            # Need at least a few points for meaningful GP
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

            if self.X_train:
                # Convert to numpy arrays for fitting, but keep lists for efficient growth
                X_arr = np.array(self.X_train)
                y_arr = np.array(self.y_train)
                self.gp_model.fit(X_arr, y_arr)
                n_points = len(self.X_train)
                self.logger.debug(
                    "GP model fitted with %d points" % n_points
                )

        except ImportError:
            self.logger.error(
                "scikit-learn not available. Install with: pip install scikit-learn"  # noqa: E501
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

        def acquisition(x):
            """Evaluate acquisition function at point x."""
            x = np.atleast_2d(x)
            # Negative for minimization
            return -self._evaluate_acquisition(x)[0]

        # Generate random starting points
        bounds = [(low, high) for low, high in self.problem.box.box]
        starts = []
        for _ in range(self.n_restarts):
            s = [
                np.random.uniform(low_val, high_val)
                for low_val, high_val in bounds
            ]
            start = np.array(s)
            starts.append(start)

        # Also try current best point as start
        if self.X_train:
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
            raise ValueError(
                f"Unknown acquisition function: {self.acquisition_func}"
            )

    def _expected_improvement(self, y_pred, y_std):
        """
        Expected Improvement acquisition function.

        EI(x) = (μ(x) - f*) * Φ(Z) + σ(x) * φ(Z)
        where Z = (μ(x) - f*) / σ(x)
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            z = (self.best_y - y_pred) / y_std
            ei = (self.best_y - y_pred) * self._norm_cdf(z) + \
                y_std * self._norm_pdf(z)
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
        return ndtr(x)
