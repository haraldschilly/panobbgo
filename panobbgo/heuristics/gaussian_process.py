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
        gp_model: The fitted GaussianProcessRegressor model for the objective
        gp_constraint: The fitted GaussianProcessRegressor model for constraints (if EIC enabled)
        X_train: Training points (n_samples, n_features)
        y_train: Training target values (n_samples,) - used for gp_model
        y_fx_train: Raw objective values (n_samples,)
        y_cv_train: Raw constraint violation values (n_samples,)
    """

    def __init__(
        self,
        strategy,
        acquisition_func=AcquisitionFunction.EI,
        kappa=1.96,
        xi=0.01,
        n_restarts=10,
        enable_eic=True,
    ):
        """
        Initialize the Gaussian Process heuristic.

        Args:
            strategy: The optimization strategy instance
            acquisition_func: Acquisition function to use
            kappa: Exploration param for UCB (default: 1.96, ~95% confidence)
            xi: Exploration parameter for EI/PI (default: 0.01)
            n_restarts: Number of random restarts for opt (default: 10)
            enable_eic: Whether to use Expected Improvement with Constraints (default: True)
        """
        super().__init__(strategy)
        self.logger = self.config.get_logger("H:GP")

        # Acquisition function parameters
        self.acquisition_func = acquisition_func
        self.kappa = kappa
        self.xi = xi
        self.n_restarts = n_restarts
        self.enable_eic = enable_eic

        # GP model state
        self.gp_model = None
        self.gp_constraint = None
        self.X_train = None
        self.y_train = None  # Target for gp_model

        # Raw data storage
        self.y_fx_train = None
        self.y_cv_train = None

        self.best_y = np.inf

    def on_start(self):
        """Initialize the heuristic at the start of optimization."""
        self.gp_model = None
        self.gp_constraint = None
        self.X_train = None
        self.y_train = None
        self.y_fx_train = None
        self.y_cv_train = None
        self.best_y = np.inf
        self.logger.info("Gaussian Process heuristic started")

    def on_new_results(self, results):
        """
        Update GP model when new results are available.

        Args:
            results: List of Result objects
        """
        if not results:
            return

        new_X_list = []
        new_fx_list = []
        new_cv_list = []

        for r in results:
            if r.fx is not None and np.isfinite(r.fx):
                new_X_list.append(r.x)
                new_fx_list.append(r.fx)
                cv = r.cv if r.cv is not None else 0.0
                new_cv_list.append(cv)

        if not new_X_list:
            return

        new_X = np.array(new_X_list)
        new_fx = np.array(new_fx_list)
        new_cv = np.array(new_cv_list)

        # Update raw data arrays
        if self.X_train is None:
            self.X_train = new_X
            self.y_fx_train = new_fx
            self.y_cv_train = new_cv
        else:
            self.X_train = np.vstack([self.X_train, new_X])
            self.y_fx_train = np.append(self.y_fx_train, new_fx)
            self.y_cv_train = np.append(self.y_cv_train, new_cv)

        # Determine if we should use EIC
        # We use EIC if enabled AND we have observed some constraints (cv > 0)
        # Note: If problem is constrained but we only saw feasible points so far,
        # cv is all 0. gp_constraint would predict 0. prob_feas would be ~0.5-1.0 depending on var.
        # It's better to stick to Coupled Penalty if we don't know constraints exist yet?
        # Or just use EIC with 0 CVs?
        # If all CVs are 0, gp_constraint might be unstable or predict 0.
        # Let's check if there's any variation or non-zero value in CV.
        has_constraints = np.any(self.y_cv_train > 1e-6)

        if self.enable_eic and has_constraints:
            # EIC Mode
            self.y_train = self.y_fx_train  # Target is raw objective
            # Update best_y (best FEASIBLE objective)
            feasible_mask = self.y_cv_train <= 1e-6
            if np.any(feasible_mask):
                self.best_y = np.min(self.y_fx_train[feasible_mask])
            else:
                # No feasible points yet.
                # EI usually requires a target.
                # If we use min(y_train), we target improving over best infeasible?
                # EIC formulation: EI(x) * P(feas).
                # EI part needs a target.
                # Standard practice: Use best observed so far.
                self.best_y = np.min(self.y_fx_train)
        else:
            # Standard/Coupled Mode (Penalty)
            # Reconstruct penalized values
            # We need to calculate penalty for ALL points, as dynamic penalty might change rho.
            # But here we just use the current penalty value for new points?
            # Ideally we should re-evaluate penalty for all points if rho changes (DynamicPenalty).
            # But calculating for all history using current strategy handler is safer.

            # Vectorized penalty calculation if possible?
            # Handler usually takes Result object.
            # We can approximate or use loop.
            # Since we need to update y_train, let's regenerate it.

            get_val = self.strategy.constraint_handler.get_penalty_value
            # We need Result objects for get_penalty_value usually.
            # But we only stored arrays.
            # This is a limitation. We can recreate dummy Results or use handler logic manually.
            # Or just update incrementally (assuming static penalty).
            # For robustness, let's use the simple approach: update incrementally for now.
            # (Note: Dynamic penalty might degrade if we don't update history, but GP is robust).

            # Actually, let's use the helper to compute penalty from arrays if possible,
            # or loop over new points.

            # Recalculating all is expensive if N is large.
            # Let's append new penalized values.

            new_penalized = []
            # We iterate over the *new* points we just received
            # (Wait, we need to map back to the original Result objects?
            # We have new_X etc extracted from results.
            # Let's just iterate results again for penalty values)

            # Re-iterate results to get penalty values
            for r in results:
                if r.fx is not None and np.isfinite(r.fx):
                    val = get_val(r)
                    new_penalized.append(val)

            new_penalized = np.array(new_penalized)

            if self.y_train is None:  # Initial call or reset
                self.y_train = new_penalized
            else:
                # Check if we are switching from EIC to Coupled (unlikely sequence but possible)
                if len(self.y_train) != len(self.y_fx_train) - len(new_penalized):
                    # Size mismatch implies we might have been in EIC mode (y_train = y_fx_train)
                    # but now switching to penalty mode?
                    # Or just first initialization.
                    # If we switch, we need to recompute full history.
                    # Since we don't have full history Results, we might be stuck.
                    # Assumption: Mode doesn't flip back and forth frequently.
                    # If has_constraints becomes True, we switch TO EIC.
                    # If it was False, we were in Penalty mode.
                    pass

                self.y_train = np.append(self.y_train, new_penalized)

            self.best_y = np.min(self.y_train)

        if len(self.y_train) < 3:
            return

        # Fit GP model(s)
        self._fit_gp_model(has_constraints)

        # Generate new candidate points using acquisition function
        candidates = self._acquire_candidates(n_candidates=5)
        for candidate in candidates:
            self.emit(candidate)

    def _fit_gp_model(self, fit_constraint=False):
        """
        Fit the Gaussian Process model(s).

        Args:
            fit_constraint (bool): Whether to fit the constraint GP model.
        """
        import warnings
        from sklearn.exceptions import ConvergenceWarning

        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import (
                Matern,
                ConstantKernel,
                WhiteKernel,
            )

            # Use Matern kernel with some white noise for stability
            # ConstantKernel scales the Matern
            kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(
                noise_level=1e-5
            )

            self.gp_model = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=10,
            )

            if self.X_train is not None:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    self.gp_model.fit(self.X_train, self.y_train)

                    if self.enable_eic and fit_constraint and self.y_cv_train is not None:
                        # Fit constraint model
                        # CV is non-negative.
                        # Modeling log(cv + epsilon) might be better for scaling?
                        # Or just cv. Let's stick to cv for simplicity first.
                        self.gp_constraint = GaussianProcessRegressor(
                            kernel=kernel,  # Reuse kernel structure
                            alpha=1e-6,
                            normalize_y=True,
                            n_restarts_optimizer=10,
                        )
                        self.gp_constraint.fit(self.X_train, self.y_cv_train)
                    else:
                        self.gp_constraint = None

                n_points = len(self.X_train)
                self.logger.debug("GP model fitted with %d points" % n_points)

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
            s = [np.random.uniform(low_val, high_val) for low_val, high_val in bounds]
            start = np.array(s)
            starts.append(start)

        # Also try current best point as start
        if self.X_train is not None and self.y_train is not None:
            best_idx = int(np.argmin(self.y_train))
            starts.append(self.X_train[best_idx].copy())

            # Ensure best_y is finite
            if np.isinf(self.best_y) and len(self.y_train) > 0:
                self.best_y = np.min(self.y_train)

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

        # Calculate base acquisition (EI/UCB/PI)
        if self.acquisition_func == AcquisitionFunction.EI:
            acq = self._expected_improvement(y_pred, y_std)
        elif self.acquisition_func == AcquisitionFunction.UCB:
            acq = self._upper_confidence_bound(y_pred, y_std)
        elif self.acquisition_func == AcquisitionFunction.PI:
            acq = self._probability_of_improvement(y_pred, y_std)
        else:
            raise ValueError(f"Unknown acquisition function: {self.acquisition_func}")

        # Apply EIC if enabled and constraint model exists
        if self.gp_constraint is not None:
            # Predict constraint violation
            # We modeled cv. Feasible region is cv <= 0.
            c_pred, c_std = self.gp_constraint.predict(X, return_std=True)

            # Probability of Feasibility: P(cv <= 0)
            # cv ~ N(c_pred, c_std^2)
            # P(cv <= 0) = CDF((0 - c_pred) / c_std)
            with np.errstate(divide="ignore", invalid="ignore"):
                z_feas = (0.0 - c_pred) / c_std
                prob_feas = self._norm_cdf(z_feas)

                # Handle deterministic cases (zero variance)
                # Ensure arrays for indexing safety
                c_std = np.atleast_1d(c_std)
                c_pred = np.atleast_1d(c_pred)
                prob_feas = np.atleast_1d(prob_feas)

                mask_det = c_std == 0
                if np.any(mask_det):
                    prob_feas[mask_det] = (c_pred[mask_det] <= 1e-6).astype(float)

            acq *= prob_feas

        return acq

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
        return ndtr(x)
