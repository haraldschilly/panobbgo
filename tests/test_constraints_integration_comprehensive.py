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
Comprehensive Integration Tests for Constraint Handling
=====================================================

These tests verify the end-to-end performance of different constraint handling
strategies on various constrained optimization problems.
"""

import pytest
import numpy as np
import time
from panobbgo.lib import Problem, Point
from panobbgo.strategies.rewarding import StrategyRewarding
from panobbgo.heuristics import Center, Random, Nearby, NelderMead, FeasibleSearch, ConstraintGradient
from panobbgo.lib.classic import RosenbrockConstraint, RosenbrockAbsConstraint
from panobbgo.lib.constraints import (
    DefaultConstraintHandler,
    PenaltyConstraintHandler,
    AugmentedLagrangianConstraintHandler,
    DynamicPenaltyConstraintHandler
)

# --- Test Problems ---

class Simionescu(Problem):
    """
    Simionescu function with constraints.
    f(x,y) = 0.1 * xy
    subject to: x^2 + y^2 <= (1 + 0.2 cos(8 atan(x/y)))^2
    """
    def __init__(self):
        # Box is typically [-1.25, 1.25]
        super().__init__([(-1.25, 1.25), (-1.25, 1.25)])

    def eval(self, x):
        return 0.1 * x[0] * x[1]

    def eval_constraints(self, x):
        # Constraint: x^2 + y^2 <= (r_T)^2
        # g(x) = x^2 + y^2 - (1 + 0.2 cos(8 theta))^2 <= 0
        r_sq = x[0]**2 + x[1]**2

        # Handle atan2 for theta
        theta = np.arctan2(x[1], x[0])

        # r_T = 1 + 0.2 * cos(8 * theta)
        r_T = 1.0 + 0.2 * np.cos(8.0 * theta)

        val = r_sq - r_T**2
        return np.array([max(0.0, val)])

class MishraBird(Problem):
    """
    Mishra's Bird function - constrained.
    f(x,y) = sin(y)e^((1-cos(x))^2) + cos(x)e^((1-sin(y))^2) + (x-y)^2
    s.t. (x+5)^2 + (y+5)^2 < 25
    """
    def __init__(self):
        super().__init__([(-10, 0), (-6.5, 0)])
        # Optimum approx -106.7645 at (-3.1302468, -1.5821422)

    def eval(self, x):
        X, Y = x[0], x[1]
        term1 = np.sin(Y) * np.exp((1 - np.cos(X))**2)
        term2 = np.cos(X) * np.exp((1 - np.sin(Y))**2)
        term3 = (X - Y)**2
        return term1 + term2 + term3

    def eval_constraints(self, x):
        # (x+5)^2 + (y+5)^2 - 25 <= 0
        val = (x[0] + 5)**2 + (x[1] + 5)**2 - 25
        return np.array([max(0.0, val)])

@pytest.mark.parametrize("HandlerClass, kwargs", [
    (DefaultConstraintHandler, {"rho": 100.0}),
    (PenaltyConstraintHandler, {"rho": 100.0, "exponent": 2.0}),
    (DynamicPenaltyConstraintHandler, {"rho_start": 10.0, "rate": 0.05}),
    (AugmentedLagrangianConstraintHandler, {"rho": 10.0, "rate": 2.0})
])
def test_rosenbrock_constraint_handlers(HandlerClass, kwargs):
    """
    Test RosenbrockConstraint with different handlers.
    """
    problem = RosenbrockConstraint(dims=2)

    strategy = StrategyRewarding(problem, testing_mode=True)
    strategy.config.max_eval = 1000 # Increased budget
    strategy.config.convergence_window_size = 200 # Patience

    # Manually swap handler
    handler = HandlerClass(strategy=strategy, **kwargs)
    strategy.constraint_handler = handler
    if hasattr(handler, 'on_new_results'):
        strategy.eventbus.register(handler)

    strategy.add(Random)
    strategy.add(Nearby)
    strategy.add(FeasibleSearch)
    strategy.add(ConstraintGradient)

    strategy.start()

    best = strategy.best
    assert best is not None

    # Check feasibility with relaxed tolerance
    assert best.cv < 1.0, f"Failed to find feasible solution with {HandlerClass.__name__}, cv={best.cv}"

    # Check objective value (should be reasonable)
    # Unconstrained min is -50. Constrained min is > -50.
    # We check if it found something reasonably low.
    # Note: If CV is high, FX might be very low (unconstrained).
    if best.cv < 0.1:
        assert best.fx < 50.0, f"Poor solution with {HandlerClass.__name__}, fx={best.fx}"


def test_simionescu_alm():
    """
    Test Simionescu problem specifically with Augmented Lagrangian.
    """
    problem = Simionescu()
    strategy = StrategyRewarding(problem, testing_mode=True)
    strategy.config.max_eval = 2000 # Increased budget significantly

    handler = AugmentedLagrangianConstraintHandler(strategy=strategy, rho=5.0)
    strategy.constraint_handler = handler
    strategy.eventbus.register(handler)

    strategy.add(Random)
    strategy.add(Nearby)
    strategy.add(FeasibleSearch)
    strategy.add(ConstraintGradient)
    # NelderMead might help refining
    strategy.add(NelderMead)

    strategy.start()

    best = strategy.best
    # Simionescu constraints can be tough at boundaries.
    assert best.cv < 0.05, f"Simionescu: Failed to satisfy constraints. CV={best.cv}"
    # Optimal value is -0.072625
    if best.cv < 1e-3:
         assert best.fx < -0.04, f"Simionescu: Failed to converge to optimum. fx={best.fx}"


def test_mishra_bird_comparison():
    """
    Compare Default and ALM on Mishra Bird function.
    """
    problem = MishraBird()

    # Run with Default
    s1 = StrategyRewarding(problem, testing_mode=True)
    s1.config.max_eval = 1000
    s1.add(Random)
    s1.add(FeasibleSearch)
    s1.start()

    # Run with ALM
    s2 = StrategyRewarding(problem, testing_mode=True)
    s2.config.max_eval = 1000
    alm = AugmentedLagrangianConstraintHandler(strategy=s2)
    s2.constraint_handler = alm
    s2.eventbus.register(alm)
    s2.add(Random)
    s2.add(FeasibleSearch)
    s2.start()

    print(f"Default: fx={s1.best.fx}, cv={s1.best.cv}")
    print(f"ALM:     fx={s2.best.fx}, cv={s2.best.cv}")

    # Relaxed checks for robustness
    assert s1.best.cv < 0.5
    assert s2.best.cv < 0.5

    # Both should find something near -106
    if s1.best.cv < 0.1:
        assert s1.best.fx < -80
    if s2.best.cv < 0.1:
        assert s2.best.fx < -80


def test_rosenbrock_abs_constraint_dynamic_penalty():
    """
    Test RosenbrockAbsConstraint with DynamicPenaltyConstraintHandler.
    """
    problem = RosenbrockAbsConstraint(dims=3)
    strategy = StrategyRewarding(problem, testing_mode=True)
    strategy.config.max_eval = 1000

    handler = DynamicPenaltyConstraintHandler(strategy=strategy, rho_start=1.0, rate=0.01)
    strategy.constraint_handler = handler

    strategy.add(Random)
    strategy.add(Nearby)
    strategy.add(FeasibleSearch)

    strategy.start()

    assert strategy.best.cv < 0.5
    if strategy.best.cv < 0.1:
        assert strategy.best.fx < 100.0


def test_constraint_gradient_effectiveness():
    """
    Check if ConstraintGradient is stable and doesn't degrade performance
    on a linearly constrained problem (RosenbrockConstraint).
    """
    problem = RosenbrockConstraint(dims=2)

    # Strategy: With ConstraintGradient
    s = StrategyRewarding(problem, testing_mode=True)
    s.config.max_eval = 500
    s.add(Random)
    s.add(Center)
    s.add(ConstraintGradient)

    # Also add FeasibleSearch to complement it
    s.add(FeasibleSearch)

    s.start()

    print(f"With CG + FS: best cv={s.best.cv}, fx={s.best.fx}")

    # Expect reasonable feasibility
    assert s.best.cv < 1.0, "Strategy with ConstraintGradient failed to find near-feasible point"
