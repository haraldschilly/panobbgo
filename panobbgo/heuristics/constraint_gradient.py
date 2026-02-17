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
    - If the current best point is infeasible (cv > 0):
      - Estimates the gradient of constraint violations using finite differences from history.
      - If `cv_vec` (vector of constraints) is available, it solves a Linear Programming (LP)
        subproblem to find a descent direction that reduces the maximum violation.
      - If only scalar `cv` is available, it uses the gradient of the sum of violations.
    - If the current best point is feasible (cv == 0):
      - Estimates the gradient of the objective function and constraints.
      - Solves an LP to minimize the objective while respecting linearized constraints.
    - Generates new points by taking steps in the calculated descent direction.
    """

    def __init__(self, strategy, step_size=1e-2, descent_step=0.1, samples=1):
        """
        Args:
            strategy: Reference to the strategy.
            step_size (float): Step size for finite difference gradient approximation (h).
            descent_step (float): Step size for moving in the descent direction (relative to box).
            samples (int): Number of points to generate along the direction (if > 1, uses scaling factors).
        """
        super().__init__(strategy, name="ConstraintGradient")
        self.step_size = step_size
        self.descent_step = descent_step
        self.samples = samples
        self.current_best = None

    def on_new_best(self, best):
        """
        React to a new best point. Try to improve it using gradient information.
        """
        self.current_best = best
        self.clear_output()

        # Try to generate points regardless of feasibility
        # If feasible -> minimize fx s.t. constraints
        # If infeasible -> minimize max violation
        self._generate_descent_point(best)

    def _generate_descent_point(self, best):
        """
        Estimates gradient of CV/FX at 'best' and generates a new point.
        """
        x = best.x
        dim = self.problem.dim

        # Collect candidate points from history for gradient estimation
        results_container = self.strategy.results
        if results_container is None or len(results_container) == 0:
            return

        X_candidates = None
        cv_vec_candidates = None
        cv_candidates = None
        fx_candidates = None

        # Robustly fetch history
        if hasattr(results_container, "get_history"):
            try:
                h = results_container.get_history(n=200)  # Use more history for better estimation
                X_candidates = h["x"]
                fx_candidates = h["fx"]
                cv_candidates = h["cv"]
                cv_vec_candidates = h["cv_vec"]
            except Exception:
                return
        elif isinstance(results_container, list):
            # Test/Legacy case
            subset = results_container[-200:]
            if not subset:
                return
            try:
                X_candidates = np.array([r.x for r in subset], dtype=float)
                fx_candidates = np.array([r.fx for r in subset], dtype=float)
                cv_candidates = np.array([r.cv for r in subset], dtype=float)
                # Handle cv_vec robustly
                cv_vec_list = [r.cv_vec if r.cv_vec is not None else [] for r in subset]
                # Check if all have same length
                lengths = [len(v) for v in cv_vec_list]
                if len(set(lengths)) == 1 and lengths[0] > 0:
                    cv_vec_candidates = np.array(cv_vec_list, dtype=float)
                else:
                    cv_vec_candidates = None
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
        mask = dists > 1e-9

        valid_dists = dists[mask]
        valid_X = X_candidates[mask]
        valid_fx = fx_candidates[mask]
        valid_cv = cv_candidates[mask]
        valid_cv_vec = cv_vec_candidates[mask] if cv_vec_candidates is not None else None

        # We need at least 'dim' neighbors to estimate gradient
        min_neighbors = dim + 1
        if len(valid_dists) < min_neighbors:
            self.logger.debug(
                f"Not enough neighbors for gradient estimation. Need {min_neighbors}, got {len(valid_dists)}"
            )
            return

        # Select k nearest neighbors
        # Use slightly more neighbors than minimum to be robust to noise (overdetermined system)
        k = min(len(valid_dists), max(min_neighbors + 2, dim * 2))
        idx = np.argpartition(valid_dists, k - 1)[:k]

        neighbors_X = valid_X[idx]
        neighbors_fx = valid_fx[idx]
        neighbors_cv = valid_cv[idx]
        neighbors_cv_vec = valid_cv_vec[idx] if valid_cv_vec is not None else None

        # Determine strategy: Scalar (fallback) or Vector (LP)
        # Vector strategy requires cv_vec
        use_vector = (
            neighbors_cv_vec is not None
            and neighbors_cv_vec.size > 0
            and neighbors_cv_vec.shape[1] > 0
            and best.cv_vec is not None
            and len(best.cv_vec) > 0
        )

        try:
            if use_vector:
                self._solve_lp_direction(best, neighbors_X, neighbors_fx, neighbors_cv_vec, x, dim)
            else:
                self._solve_scalar_direction(best, neighbors_X, neighbors_cv, x, dim)

        except Exception as e:
            self.logger.debug(f"Error in gradient estimation: {e}")

    def _estimate_gradient(self, X_centered, y_vals, best_val):
        """
        Estimates gradient using linear regression: y(x) = gradient^T (x - best.x) + best_val
        """
        dim = X_centered.shape[1]
        y_reg = np.concatenate([y_vals, [best_val]])
        X_reg = np.vstack([X_centered, np.zeros((1, dim))]) # Add origin (best point)

        # Solve least squares for gradient
        # We assume intercept is best_val (fixed point), so we solve:
        # y - best_val = gradient^T * X_centered
        # No, wait, if we fix intercept, we solve X * g = y - c

        y_centered = y_reg - best_val

        # Use simple least squares
        gradient, _, _, _ = np.linalg.lstsq(X_reg, y_centered, rcond=None)
        return gradient

    def _solve_scalar_direction(self, best, neighbors_X, neighbors_cv, x, dim):
        """
        Original logic: Minimize scalar CV using gradient descent.
        Only applicable if best.cv > 0.
        """
        if best.cv <= 1e-9:
             # If feasible and no vector info, we can't do much about constraints.
             # Maybe minimize fx? But without constraint gradients we risk violating feasibility.
             return

        X_centered = neighbors_X - x
        gradient = self._estimate_gradient(X_centered, neighbors_cv, best.cv)

        # Check if gradient is meaningful (not zero)
        grad_norm = np.linalg.norm(gradient)
        if grad_norm < 1e-9:
            return

        # Descent direction is negative gradient
        descent_direction = -gradient / grad_norm
        self._emit_candidates(x, descent_direction)

    def _solve_lp_direction(self, best, neighbors_X, neighbors_fx, neighbors_cv_vec, x, dim):
        """
        New logic: Use LP to find descent direction respecting all constraints.
        """
        from scipy.optimize import linprog

        X_centered = neighbors_X - x
        num_constraints = neighbors_cv_vec.shape[1]

        # Estimate gradients
        # fx gradient
        grad_fx = self._estimate_gradient(X_centered, neighbors_fx, best.fx)

        # constraint gradients
        grad_cv = np.zeros((num_constraints, dim))
        for i in range(num_constraints):
            grad_cv[i, :] = self._estimate_gradient(X_centered, neighbors_cv_vec[:, i], best.cv_vec[i])

        # Formulate LP
        # Variables: d (dim components) + gamma (1 component, slack)
        # Total variables: dim + 1

        # Determine bounds for d based on descent_step
        # We want local step, so bound d by box range fraction
        box_ranges = self.problem.ranges
        delta = self.descent_step * box_ranges

        # Objective function coefficients c
        # If feasible (best.cv == 0): Minimize grad_fx^T * d -> c = [grad_fx, 0]
        # If infeasible (best.cv > 0): Minimize gamma -> c = [0...0, 1]

        is_feasible = best.cv <= 1e-9

        c = np.zeros(dim + 1)
        if is_feasible:
            c[:dim] = grad_fx
            # Gamma not used in objective, but used in constraints?
            # If feasible, we want: grad_cv^T * d <= -best.cv_vec (keep <= 0)
            # We don't need gamma for feasible case if we just enforce constraints.
            # But let's use gamma to allow slight relaxation if needed?
            # No, standard SQP uses strict linearization.
            # However, to be robust, let's just minimize fx subject to constraints.
            c[dim] = 0.0
        else:
            c[dim] = 1.0 # Minimize gamma

        # Constraints: A_ub * [d, gamma]^T <= b_ub
        # Condition: grad_cv_i^T * d - gamma <= -best.cv_vec_i
        # Rearranged: grad_cv_i^T * d + (-1)*gamma <= -best.cv_vec_i
        # So A_row = [grad_cv_i, -1]

        # Wait, if feasible, we don't want to minimize gamma. We want to satisfy constraints.
        # But if we use the same formulation, minimizing gamma will push constraints as far negative as possible (into interior).
        # That's not what we want if we optimize fx.

        if is_feasible:
            # Constraints: grad_cv_i^T * d <= -best.cv_vec_i
            # A_ub = [grad_cv_i, 0]
            # b_ub = -best.cv_vec_i
            A_ub = np.hstack([grad_cv, np.zeros((num_constraints, 1))])
            b_ub = -best.cv_vec

            # Bounds for gamma can be anything, say [-1, 1] but it doesn't affect objective
            bounds_gamma = (None, None)
        else:
            # Constraints: grad_cv_i^T * d - gamma <= -best.cv_vec_i
            A_ub = np.hstack([grad_cv, -np.ones((num_constraints, 1))])
            b_ub = -best.cv_vec
            bounds_gamma = (None, None)

        # Bounds for d
        bounds = []
        for i in range(dim):
            bounds.append((-delta[i], delta[i]))
        bounds.append(bounds_gamma) # Gamma

        # Solve LP
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

        if res.success:
            d = res.x[:dim]
            # Check if d is non-zero
            if np.linalg.norm(d) > 1e-9:
                self._emit_candidates(x, d, is_vector=True)
            else:
                 self.logger.debug("LP returned zero step")
        else:
             self.logger.debug(f"LP failed: {res.message}")

    def _emit_candidates(self, x, direction, is_vector=False):
        """
        Emits points along the direction.
        """
        # If vector LP, direction magnitude is already scaled by bounds (delta).
        # If scalar, direction is unit vector, needs scaling.

        candidates = []

        if is_vector:
            # Just emit x + direction (and maybe scaled versions?)
            # Since LP bounds were local, x+d is within trust region.
            if self.samples == 1:
                candidates.append(self.problem.project(x + direction))
            else:
                for factor in np.linspace(0.5, 1.5, self.samples):
                    candidates.append(self.problem.project(x + direction * factor))
        else:
            # Scalar direction is unit vector
            # Use pseudo-line search
            if self.samples == 1:
                scaling_factors = [1.0]
            else:
                scaling_factors = np.linspace(0.5, 1.5, self.samples)

            for factor in scaling_factors:
                step = direction * (self.descent_step * factor) * self.problem.ranges
                new_x = x + step
                new_x = self.problem.project(new_x)
                candidates.append(new_x)

        self.emit(candidates)
