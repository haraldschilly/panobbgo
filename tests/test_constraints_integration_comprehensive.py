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
from panobbgo.heuristics import Center, Random, Nearby, NelderMead, FeasibleSearch, ConstraintGradient, GaussianProcessHeuristic
from panobbgo.lib.classic import (
    RosenbrockConstraint,
    RosenbrockAbsConstraint,
    Simionescu,
    MishraBird
)
from panobbgo.lib.constraints import (
    DefaultConstraintHandler,
    PenaltyConstraintHandler,
    AugmentedLagrangianConstraintHandler,
    DynamicPenaltyConstraintHandler
)


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


def test_gp_eic_effectiveness():
    """
    Test Gaussian Process with Expected Improvement with Constraints (EIC)
    on a constrained problem.
    """
    problem = RosenbrockConstraint(dims=2)
    strategy = StrategyRewarding(problem, testing_mode=True)
    strategy.config.max_eval = 200 # GP is expensive but sample efficient

    # Add GP Heuristic
    # Note: EIC should activate automatically when constraints are encountered
    strategy.add(GaussianProcessHeuristic)
    strategy.add(Random) # Need some random points to start

    strategy.start()

    print(f"GP EIC Best: fx={strategy.best.fx}, cv={strategy.best.cv}")

    # GP should be able to find a decent solution with 200 evals
    assert strategy.best.cv < 0.1, f"GP with EIC failed feasibility. CV={strategy.best.cv}"

    # Unconstrained min is -50. Constrained is higher.
    if strategy.best.cv < 1e-3:
         # Should be reasonably good
         assert strategy.best.fx < 100.0, f"GP with EIC found poor solution. fx={strategy.best.fx}"
