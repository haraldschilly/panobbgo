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


class ConstraintRepair(Heuristic):
    """
    Heuristic that attempts to repair infeasible solutions by projecting them
    onto the feasible region.

    It triggers when a new best point is found that is infeasible (cv > 0).
    It uses Scipy's SLSQP optimizer to minimize the distance to the original point
    subject to the problem's constraints.

    Objective: min ||x - x_start||^2
    Subject to: g(x) <= 0  (evaluated via problem.eval_constraints)
    """

    def __init__(self, strategy, max_iter=20, tolerance=1e-6):
        """
        Args:
            strategy: Reference to strategy.
            max_iter (int): Maximum iterations for the repair optimizer.
            tolerance (float): Feasibility tolerance.
        """
        super().__init__(strategy, name="ConstraintRepair")
        self.max_iter = max_iter
        self.tolerance = tolerance
        self._last_repaired_hash = None

    def on_new_best(self, best):
        """
        Check if the new best point needs repair.
        """
        if best is None:
            return

        # Only repair if infeasible
        if best.cv > self.tolerance:
            # Avoid repairing the same point (or very similar) repeatedly
            x_hash = hash(best.x.tobytes())
            if self._last_repaired_hash == x_hash:
                return

            self._last_repaired_hash = x_hash
            self._repair(best)

    def _repair(self, best):
        try:
            from scipy.optimize import minimize
        except ImportError:
            self.logger.warning("Scipy not installed, ConstraintRepair disabled.")
            return

        x0 = best.x.copy()

        # Define projection objective: minimize distance to starting point
        def obj(x):
            return np.sum((x - x0)**2)

        def jac_obj(x):
            return 2 * (x - x0)

        # Define constraints
        # Scipy 'ineq' means fun(x) >= 0
        # Panobbgo eval_constraints returns g(x) where g(x) <= 0 is feasible
        # So we need -g(x) >= 0
        def constraint_fun(x):
            val = self.problem.eval_constraints(x)
            # Handle case where eval_constraints returns None (unconstrained)
            if val is None:
                return np.array([])
            return -val

        constraints = {
            'type': 'ineq',
            'fun': constraint_fun
        }

        # Box bounds
        bounds = self.problem.box.box

        try:
            res = minimize(
                obj,
                x0,
                method='SLSQP',
                jac=jac_obj,
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': self.max_iter, 'disp': False}
            )

            if res.success or res.message == 'Iteration limit reached':
                # Check if we actually improved feasibility
                # SLSQP might stop without full feasibility if max_iter reached
                x_new = self.problem.project(res.x)

                # We emit the point. The Strategy will evaluate it.
                # If it's feasible, great. If it's just "more feasible", also good.

                # Check distance to avoid emitting same point
                if np.linalg.norm(x_new - x0) > 1e-9:
                    self.emit(x_new)
                    self.logger.debug(f"Emitted repaired point (success={res.success})")
            else:
                self.logger.debug(f"Repair failed: {res.message}")

        except Exception as e:
            self.logger.debug(f"Exception during repair: {e}")
