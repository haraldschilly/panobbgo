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
        if not self.strategy.results:
            return

        # Get k nearest neighbors
        k = dim + 1
        results = self.strategy.results

        # Filter for points close to x
        # Compute distances
        # Optimization: Don't compute all if many results.
        # Just take last N results or random subset if N is large?
        # Or use the Splitter structure?

        # Let's use a simple subset for now
        candidates = results[-100:] if len(results) > 100 else results

        dists = []
        for r in candidates:
            d = np.linalg.norm(r.x - x)
            if d > 1e-9: # Exclude the point itself
                dists.append((d, r))

        dists.sort(key=lambda x: x[0])
        neighbors = [p[1] for p in dists[:k]]

        if len(neighbors) < dim:
            # Not enough points for gradient estimation
            # Fallback to random direction?
            return

        # Prepare for Linear Regression: X * w = y
        # X is (k, dim), y is (k,)
        # Shift x to origin for stability
        X = np.array([n.x - x for n in neighbors])
        y = np.array([n.cv - best.cv for n in neighbors])

        # Solve least squares
        # w = (X^T X)^-1 X^T y
        # Use lstsq for robustness
        try:
            grad, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
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
