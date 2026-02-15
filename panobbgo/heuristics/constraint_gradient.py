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

from panobbgo.core import Heuristic
import numpy as np


class ConstraintGradient(Heuristic):
    """
    Constraint-specific heuristic that uses gradient estimation of constraint violation
    to guide the search towards feasibility.

    Behavior:
    - If the current best point is infeasible (cv > 0), this heuristic estimates the
      gradient of the constraint violation function (sum of violations) using finite differences.
    - It then generates a new point by taking a step in the direction of the negative gradient
      (descent direction for violation).
    - If the current best is feasible, this heuristic is inactive.
    """

    def __init__(self, strategy, step_size=1e-2, descent_step=0.1, samples=1):
        """
        Args:
            strategy: Reference to the strategy.
            step_size (float): Step size for finite difference gradient approximation (h).
            descent_step (float): Step size for moving in the descent direction (relative to box).
            samples (int): Number of points to generate (currently only supports 1 descent step).
        """
        super().__init__(strategy, name="ConstraintGradient")
        self.step_size = step_size
        self.descent_step = descent_step
        self.samples = samples
        self.current_best = None

    def on_new_best(self, best):
        """
        React to a new best point. If infeasible, try to repair it using gradient descent on CV.
        """
        self.current_best = best
        self.clear_output()

        if best.cv > 0:
            self._generate_descent_point(best)

    def _generate_descent_point(self, best):
        """
        Estimates gradient of CV at 'best' and generates a new point.
        """
        x = best.x
        dim = self.problem.dim

        # Collect candidate points from history for gradient estimation
        results_container = self.strategy.results
        if results_container is None or len(results_container) == 0:
            return

        X_candidates = None
        cv_candidates = None

        # Robustly fetch history
        if hasattr(results_container, "get_history"):
            try:
                h = results_container.get_history(n=100)
                X_candidates = h["x"]
                cv_candidates = h["cv"]
            except Exception:
                return
        elif isinstance(results_container, list):
            # Test/Legacy case
            subset = results_container[-100:]
            if not subset:
                return
            try:
                X_candidates = np.array([r.x for r in subset], dtype=float)
                cv_candidates = np.array([r.cv for r in subset], dtype=float)
            except Exception:
                return
        else:
            return

        if X_candidates is None or len(X_candidates) == 0:
            return

        # Calculate distances to current best
        diffs = X_candidates - x
        dists = np.linalg.norm(diffs, axis=1)

        # Exclude the point itself (dist <= 1e-9) to identify neighbors
        # We will re-add 'best' explicitly later for the regression
        mask = dists > 1e-9

        valid_dists = dists[mask]
        valid_X = X_candidates[mask]
        valid_cv = cv_candidates[mask]

        # We need at least 'dim' neighbors to form a simplex with 'best' (dim+1 points)
        # to solve for 'dim' gradients + intercept.
        min_neighbors = dim
        if len(valid_dists) < min_neighbors:
            self.logger.debug(
                f"Not enough neighbors for gradient estimation. Need {min_neighbors}, got {len(valid_dists)}"
            )
            return

        # Select k nearest neighbors
        # Use slightly more neighbors than minimum to be robust to noise (overdetermined system)
        k = min(len(valid_dists), dim * 2)
        idx = np.argpartition(valid_dists, k - 1)[:k]

        neighbors_X = valid_X[idx]
        neighbors_cv = valid_cv[idx]

        # Prepare for Linear Regression: cv(x) = gradient^T (x - best.x) + intercept
        # We include 'best' in the regression set

        # Combine neighbors and best point
        X_reg = np.vstack([neighbors_X, x])
        y_reg = np.concatenate([neighbors_cv, [best.cv]])

        # Shift X to be relative to 'best' (improves numerical stability)
        X_centered = X_reg - x

        # Design matrix: [X_centered, 1]
        n_points = len(y_reg)
        X_design = np.column_stack([X_centered, np.ones(n_points)])

        try:
            # Solve using robust least squares
            # coeffs = [g_1, g_2, ..., g_dim, intercept]
            coeffs, _, _, _ = np.linalg.lstsq(X_design, y_reg, rcond=None)

            gradient = coeffs[:dim]
            intercept = coeffs[dim]

            # Check if gradient is meaningful (not zero)
            grad_norm = np.linalg.norm(gradient)
            if grad_norm < 1e-9:
                return

            # Descent direction is negative gradient
            descent_direction = -gradient / grad_norm

            # Generate points along the descent direction
            # Use pseudo-line search with varying step sizes
            if self.samples == 1:
                scaling_factors = [1.0]
            else:
                scaling_factors = np.linspace(0.5, 1.5, self.samples)

            candidates = []
            for factor in scaling_factors:
                step = descent_direction * (self.descent_step * factor) * self.problem.ranges
                new_x = x + step
                new_x = self.problem.project(new_x)
                candidates.append(new_x)

            self.emit(candidates)

        except np.linalg.LinAlgError:
            self.logger.debug("Linear algebra error in gradient estimation")
            return
