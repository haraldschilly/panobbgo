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

from panobbgo.lib.lib import Result
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
            return max(0.0, old_best.fx - new_best.fx)

        # Case 2: Old Infeasible, New Feasible
        if not old_feasible and new_feasible:
            # Significant improvement: crossing from infeasible to feasible.
            # We add a base reward plus a term proportional to how bad the old point was.
            # This ensures this transition is valued higher than small fx improvements.
            return 10.0 + self.rho * old_best.cv

        # Case 3: Both Infeasible
        if not old_feasible and not new_feasible:
            # Primary goal is to reduce constraint violation
            cv_improv = old_best.cv - new_best.cv
            if cv_improv > 0:
                return cv_improv * self.rho

            # If CV is same (unlikely with floats, but possible), check fx?
            # Usually we stick to CV reduction.
            return 0.0

        # Case 4: Old Feasible, New Infeasible
        # This implies a regression in quality, so 0 improvement.
        return 0.0


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

        return max(0.0, p_old - p_new)


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

        return max(0.0, p_old - p_new)
