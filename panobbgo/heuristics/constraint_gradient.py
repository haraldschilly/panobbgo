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

        if best.cv > 0:
            self._generate_descent_point(best)

    def _generate_descent_point(self, best):
        """
        Estimates gradient of CV at 'best' and generates a new point.
        """
        x = best.x
        dim = self.problem.dim

        # We need to evaluate the constraint function (cv) at neighboring points.
        # However, we cannot simply evaluate constraints without evaluating the full problem
        # because the interface `problem.eval_constraints(x)` might not be available or
        # evaluating it might be costly/part of the black box.
        # But wait, panobbgo Problems usually have `eval_constraints`.
        # If we assume we can call `problem.eval_constraints(x)` cheaply (or at all), we can do this.
        # In black-box optimization, we usually assume we have to emit points to get their values.
        # BUT, if we emit points for gradient estimation, we use up budget.

        # Strategy:
        # We can't synchronously calculate gradient by evaluating points here.
        # We have to rely on past points or emit points that *might* improve.

        # Alternative approach for Black Box:
        # Use previous results near 'best' to estimate gradient (Weighted Least Squares or simple difference).

        # Simple approach using recent history:
        # Find k nearest neighbors in results.
        # Fit a linear model to CV values: cv(x) ~ a^T x + b
        # Gradient is 'a'.

        # Let's find neighbors.
        # We need access to strategy results.
        results_container = self.strategy.results
        if results_container is None or len(results_container) == 0:
            return

        candidates = []

        # Handle different result container types
        if isinstance(results_container, list):
            # Test/Legacy case: results_container is a list of Result objects
            # Use last 100 results
            subset = results_container[-100:]
            for r in subset:
                candidates.append((r.x, r.cv))
        elif hasattr(results_container, 'results'):
             # Production case: Results object wrapping a DataFrame
            df = results_container.results
            if df is None or df.empty:
                return

            try:
                # Safest way: iterate over last N rows
                n_rows = len(df)
                start_idx = max(0, n_rows - 100)

                # Get 'x' part
                # df['x'] returns a DataFrame with columns 0, 1, ...
                X_subset = df['x'].iloc[start_idx:].values # numpy array of shape (N, dim)

                # Get 'cv' part (scalar)
                # core.py: `[r.cv, r.who, r.error]` are added at the end.
                # So `cv` scalar is 3rd from end.
                cv_values = df.iloc[start_idx:, -3].values

                for i in range(len(X_subset)):
                    xi = X_subset[i]
                    # Ensure xi is float array
                    xi = np.array(xi, dtype=float)
                    cvi = float(cv_values[i])
                    candidates.append((xi, cvi))
            except Exception:
                # Fallback or fail gracefully
                return
        else:
            return

        # Filter for points close to x
        # Compute distances
        dists = []
        for xi, cvi in candidates:
            d = np.linalg.norm(xi - x)
            if d > 1e-9: # Exclude the point itself
                dists.append((d, xi, cvi))

        k = dim + 1
        dists.sort(key=lambda x: x[0])
        neighbors = dists[:k]

        if len(neighbors) < k:
            return

        # Prepare for Linear Regression: X * w = y
        # X is (k, dim), y is (k,)
        # Shift x to origin for stability
        X_mat = np.array([n[1] - x for n in neighbors])
        y_vec = np.array([n[2] - best.cv for n in neighbors])

        # Solve least squares
        # w = (X^T X)^-1 X^T y
        try:
            grad, _, _, _ = np.linalg.lstsq(X_mat, y_vec, rcond=None)
        except np.linalg.LinAlgError:
            return

        # Normalize gradient
        norm = np.linalg.norm(grad)
        if norm < 1e-9:
            return

        grad /= norm

        # Descent direction: -grad
        direction = -grad

        # Generate point
        # Step size relative to problem range
        step = direction * self.descent_step * self.problem.ranges

        new_x = x + step
        new_x = self.problem.project(new_x)

        self.emit(new_x)
