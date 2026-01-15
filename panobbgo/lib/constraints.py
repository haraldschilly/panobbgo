# -*- coding: utf8 -*-
# Copyright 2024 Panobbgo Contributors
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
Constraint Handling
===================

This module provides classes for handling constraints in optimization strategies,
specifically for calculating "improvement" or "fitness" when comparing results
that may differ in feasibility.
"""

from panobbgo.lib import Result
import numpy as np


class ConstraintHandler:
    """
    Abstract base class for constraint handlers.
    """
    def __init__(self, strategy=None, **kwargs):
        """
        Args:
            strategy (StrategyBase, optional): Reference to the strategy using this handler.
            **kwargs: Additional configuration parameters.
        """
        self.strategy = strategy
        self._threads = []
        self._name = self.__class__.__name__

    @property
    def name(self):
        return self._name

    def calculate_improvement(self, old_best: Result, new_best: Result) -> float:
        """
        Calculates improvement magnitude between old and new best points.

        Args:
            old_best (Result): The previous best result.
            new_best (Result): The new best result.

        Returns:
            float: A non-negative scalar representing the magnitude of improvement.
                   Larger values indicate better improvement.
        """
        raise NotImplementedError

    def is_better(self, old_best: Result, new_result: Result) -> bool:
        """
        Determines if new_result is better than old_best.

        Args:
            old_best (Result): The current best result (can be None).
            new_result (Result): The new result to check.

        Returns:
            bool: True if new_result is better, False otherwise.
        """
        raise NotImplementedError


class DefaultConstraintHandler(ConstraintHandler):
    """
    Handles constraints by prioritizing feasibility (Lexicographic ordering).

    Order of preference:
    1. Feasible points (cv=0) with lower fx.
    2. Infeasible points (cv>0) with lower cv.

    If switching from Infeasible to Feasible, improvement is considered very high.
    """
    def __init__(self, strategy=None, rho=100.0, **kwargs):
        """
        Args:
            strategy: Reference to strategy.
            rho (float): Penalty factor for CV improvement. Defaults to 100.0.
        """
        super().__init__(strategy, **kwargs)
        self.rho = rho

    def calculate_improvement(self, old_best: Result, new_best: Result) -> float:
        if old_best is None:
             # First point found is treated as a baseline improvement
             return 1.0

        old_feasible = old_best.cv == 0
        new_feasible = new_best.cv == 0

        # Case 1: Both Feasible
        if old_feasible and new_feasible:
            # Standard improvement in objective function
            return float(max(0.0, old_best.fx - new_best.fx))

        # Case 2: Old Infeasible, New Feasible
        if not old_feasible and new_feasible:
            # Significant improvement: crossing from infeasible to feasible.
            # We add a base reward plus a term proportional to how bad the old point was.
            # This ensures this transition is valued higher than small fx improvements.
            return float(10.0 + self.rho * old_best.cv)

        # Case 3: Both Infeasible
        if not old_feasible and not new_feasible:
            # Primary goal is to reduce constraint violation
            cv_improv = old_best.cv - new_best.cv
            if cv_improv > 0:
                return float(cv_improv * self.rho)

            # If CV is same (unlikely with floats, but possible), check fx?
            # Usually we stick to CV reduction.
            return 0.0

        # Case 4: Old Feasible, New Infeasible
        # This implies a regression in quality, so 0 improvement.
        return 0.0

    def is_better(self, old_best: Result, new_result: Result) -> bool:
        if old_best is None:
            return True

        cv_old = old_best.cv if old_best.cv is not None else 0.0
        cv_new = new_result.cv if new_result.cv is not None else 0.0

        # Prioritize feasibility (CV)
        if cv_new < cv_old:
            return True
        if cv_new > cv_old:
            return False

        # If CV is equal (e.g. both 0), check FX
        return new_result.fx < old_best.fx


class PenaltyConstraintHandler(ConstraintHandler):
    """
    Handles constraints using a static penalty function.

    The penalized objective function is:
        P(x) = f(x) + rho * cv(x)^exponent

    Improvement is defined as reduction in P(x).
    """
    def __init__(self, strategy=None, rho=100.0, exponent=1.0, **kwargs):
        """
        Args:
            strategy: Reference to strategy.
            rho (float): Penalty coefficient.
            exponent (float): Exponent for constraint violation (1.0 for linear, 2.0 for quadratic).
        """
        super().__init__(strategy, **kwargs)
        self.rho = rho
        self.exponent = exponent

    def calculate_improvement(self, old_best: Result, new_best: Result) -> float:
        if old_best is None:
            return 1.0

        # Calculate penalized values
        # P(x) = f(x) + rho * cv(x)^exponent

        # Handle potential None for cv (though Result.cv returns 0.0 if None)
        cv_old = old_best.cv if old_best.cv is not None else 0.0
        cv_new = new_best.cv if new_best.cv is not None else 0.0

        p_old = old_best.fx + self.rho * (cv_old ** self.exponent)
        p_new = new_best.fx + self.rho * (cv_new ** self.exponent)

        return float(max(0.0, p_old - p_new))

    def is_better(self, old_best: Result, new_result: Result) -> bool:
        if old_best is None:
            return True

        cv_old = old_best.cv if old_best.cv is not None else 0.0
        cv_new = new_result.cv if new_result.cv is not None else 0.0

        p_old = old_best.fx + self.rho * (cv_old ** self.exponent)
        p_new = new_result.fx + self.rho * (cv_new ** self.exponent)

        return p_new < p_old


class DynamicPenaltyConstraintHandler(ConstraintHandler):
    """
    Handles constraints using a dynamic penalty function that increases over time.

    P(x) = f(x) + rho(t) * cv(x)^exponent

    Where rho(t) = rho_start * (1 + rate * t)
    and t is the number of evaluations or loops.
    """
    def __init__(self, strategy=None, rho_start=10.0, rate=0.01, exponent=2.0, **kwargs):
        super().__init__(strategy, **kwargs)
        self.rho_start = rho_start
        self.rate = rate
        self.exponent = exponent

    def _get_current_rho(self):
        if self.strategy is None:
            return self.rho_start

        # Use number of results as time proxy
        t = len(self.strategy.results)
        return self.rho_start * (1.0 + self.rate * t)

    def calculate_improvement(self, old_best: Result, new_best: Result) -> float:
        if old_best is None:
            return 1.0

        rho = self._get_current_rho()

        cv_old = old_best.cv if old_best.cv is not None else 0.0
        cv_new = new_best.cv if new_best.cv is not None else 0.0

        p_old = old_best.fx + rho * (cv_old ** self.exponent)
        p_new = new_best.fx + rho * (cv_new ** self.exponent)

        return float(max(0.0, p_old - p_new))

    def is_better(self, old_best: Result, new_result: Result) -> bool:
        if old_best is None:
            return True

        rho = self._get_current_rho()

        cv_old = old_best.cv if old_best.cv is not None else 0.0
        cv_new = new_result.cv if new_result.cv is not None else 0.0

        p_old = old_best.fx + rho * (cv_old ** self.exponent)
        p_new = new_result.fx + rho * (cv_new ** self.exponent)

        return p_new < p_old


class AugmentedLagrangianConstraintHandler(ConstraintHandler):
    """
    Handles constraints using the Augmented Lagrangian Method.

    L(x, lambda, mu) = f(x) + sum(Psi(c_i(x), lambda_i, mu))

    Where Psi is the penalty-Lagrangian term for inequality constraints g(x) <= 0:
       Psi = (1/2mu) * ( max(0, lambda + mu*g)^2 - lambda^2 )

    Parameters update:
       lambda_i <- max(0, lambda_i + mu * g_i(x))
       mu <- mu * rate (if violation not decreasing sufficiently)
    """
    def __init__(self, strategy=None, rho=10.0, rate=2.0, update_interval=20, **kwargs):
        """
        Args:
            strategy: Reference to strategy.
            rho (float): Initial penalty parameter (mu). Defaults to 10.0.
            rate (float): Multiplier for mu increase. Defaults to 2.0.
            update_interval (int): Number of results between updates. Defaults to 20.
        """
        super().__init__(strategy, **kwargs)
        self.mu = rho
        self.rate = rate
        self.update_interval = update_interval
        self.lambdas = None  # Will be initialized on first result
        self.counter = 0
        self.last_cv_norm = float('inf')

        # Logging
        if strategy:
            self.logger = strategy.config.get_logger("ALM")

    def on_new_results(self, results):
        """
        Called when new results are available.
        Updates Lagrangian parameters based on the new batch of results.
        """
        if not results:
            return

        self.counter += len(results)
        if self.counter >= self.update_interval:
            self.counter = 0
            self._update_parameters()

    def _update_parameters(self):
        # We need the current best point.
        # StrategyBase.best is the best feasible (or min cv) point found so far.
        # However, for ALM, we might want to look at the best point w.r.t the Lagrangian?
        # Standard ALM updates based on the solution of the subproblem.
        # Here we approximate by taking the current global best.

        best = self.strategy.best if self.strategy else None
        if best is None:
            return

        if best.cv_vec is None:
            return

        # Initialize lambdas if needed
        if self.lambdas is None:
            self.lambdas = np.zeros_like(best.cv_vec)

        cv = best.cv_vec
        # Update mu (penalty) first based on constraint violation improvement
        # If constraint violation has not decreased significantly, increase penalty
        current_cv_norm = best.cv  # This is norm of positive parts

        # We only increase penalty if we are still infeasible
        if current_cv_norm > 0:
            if current_cv_norm > 0.9 * self.last_cv_norm:
                self.mu *= self.rate
                if hasattr(self, 'logger'):
                    self.logger.info(f"Increasing penalty mu to {self.mu:.2f} (cv: {current_cv_norm:.4f})")

            self.last_cv_norm = current_cv_norm

        # Update lambdas using the (potentially updated) mu
        # lambda_{k+1} = max(0, lambda_k + mu_k * g(x_k))
        # Note: cv_vec > 0 means violation, so g(x) corresponds to cv_vec
        new_lambdas = np.maximum(0, self.lambdas + self.mu * cv)

        self.lambdas = new_lambdas

        if hasattr(self, 'logger'):
            self.logger.debug(f"Updated AL params: mu={self.mu:.2f}, lambdas={self.lambdas}")

    def calculate_improvement(self, old_best: Result, new_best: Result) -> float:
        # Calculate Lagrangian value for both
        L_old = self._calculate_lagrangian(old_best)
        L_new = self._calculate_lagrangian(new_best)
        return float(max(0.0, L_old - L_new))

    def is_better(self, old_best: Result, new_result: Result) -> bool:
        if old_best is None:
            return True

        L_old = self._calculate_lagrangian(old_best)
        L_new = self._calculate_lagrangian(new_result)

        return L_new < L_old

    def _calculate_lagrangian(self, result: Result):
        if result is None: return float('inf')
        if result.cv_vec is None: return result.fx

        if self.lambdas is None:
             self.lambdas = np.zeros_like(result.cv_vec)

        # L = f(x) + (1/2mu) * sum( max(0, lambda + mu*g)^2 - lambda^2 )
        # term = max(0, lambda + mu * g)
        term = np.maximum(0, self.lambdas + self.mu * result.cv_vec)
        penalty_term = (1.0 / (2.0 * self.mu)) * (np.sum(term**2) - np.sum(self.lambdas**2))

        return result.fx + penalty_term
