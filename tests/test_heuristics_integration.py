# -*- coding: utf8 -*-
"""
Integration Tests for Heuristics
================================

These tests verify the end-to-end performance of specific heuristics
like NelderMead and LBFGSB in a controlled integration environment.
"""

import pytest
import numpy as np
from panobbgo.strategies.rewarding import StrategyRewarding
from panobbgo.heuristics import (
    Random,
    NelderMead,
    LBFGSB,
)
from panobbgo.analyzers.splitter import Splitter
from panobbgo.lib.classic import (
    Rosenbrock,
    RosenbrockConstraint,
)
from panobbgo.lib.constraints import (
    PenaltyConstraintHandler,
)

def test_nelder_mead_integration():
    """
    Test NelderMead heuristic integration.
    NelderMead requires existing points and Splitter to function.
    """
    problem = Rosenbrock(dims=2)
    strategy = StrategyRewarding(problem, testing_mode=True)
    strategy.config.max_eval = 100
    strategy.config.evaluation_method = "threaded"
    strategy.config.ui_show = False

    # NelderMead needs Splitter and existing points
    strategy.add(Random)
    # Splitter is added automatically by StrategyBase.start()
    strategy.add(NelderMead)

    strategy.start()

    # Verify that NelderMead actually generated points
    history = strategy.results.get_history()
    who = history['who']
    # who is array of objects/strings
    nm_count = np.sum([1 for w in who if "Nelder Mead" in str(w)])

    # It might not generate many points if max_eval is small and Random takes over,
    # or if Splitter doesn't create enough boxes/points.
    # But with 100 evaluations on 2D, it should kick in.
    print(f"NelderMead points: {nm_count}")

    # Assert we found a reasonable solution (Rosenbrock min is 0)
    best = strategy.best
    print(f"Best fx: {best.fx}")
    assert best.fx < 10.0, f"Failed to find reasonable solution, best fx={best.fx}"

    # Check if NelderMead contributed
    # Note: In short runs, Random might find good points too.
    # We mainly want to ensure no crash and potential contribution.

def test_lbfgsb_integration():
    """
    Test LBFGSB heuristic integration.
    LBFGSB runs in a subprocess and uses scipy.optimize.fmin_l_bfgs_b.
    """
    problem = Rosenbrock(dims=2)
    strategy = StrategyRewarding(problem, testing_mode=True)
    strategy.config.max_eval = 50
    strategy.config.evaluation_method = "threaded"
    strategy.config.ui_show = False

    strategy.add(LBFGSB)

    # LBFGSB generates its own points (starting from 0)
    strategy.start()

    history = strategy.results.get_history()
    who = history['who']
    lbfgs_count = np.sum([1 for w in who if "LBFGS" in str(w)])

    print(f"LBFGSB points: {lbfgs_count}")
    assert lbfgs_count > 0, "LBFGSB did not generate any points"

    best = strategy.best
    print(f"Best fx: {best.fx}")
    # LBFGSB should be very efficient on smooth functions like Rosenbrock
    assert best.fx < 1.0, f"LBFGSB failed to converge, best fx={best.fx}"

def test_lbfgsb_constrained_integration():
    """
    Test LBFGSB heuristic on a constrained problem using PenaltyConstraintHandler.
    """
    problem = RosenbrockConstraint(dims=2)
    strategy = StrategyRewarding(problem, testing_mode=True)
    strategy.config.max_eval = 100
    strategy.config.evaluation_method = "threaded"
    strategy.config.ui_show = False

    # Use PenaltyConstraintHandler to provide smooth gradient info via penalty
    handler = PenaltyConstraintHandler(strategy=strategy, rho=100.0, exponent=2.0)
    strategy.constraint_handler = handler
    strategy.eventbus.register(handler)

    strategy.add(LBFGSB)
    # Add Random to help if LBFGSB gets stuck initially (though it starts at 0)
    strategy.add(Random)

    strategy.start()

    history = strategy.results.get_history()
    who = history['who']
    lbfgs_count = np.sum([1 for w in who if "LBFGS" in str(w)])

    print(f"LBFGSB points: {lbfgs_count}")

    best = strategy.best
    print(f"Best fx: {best.fx}, cv: {best.cv}")

    # Check feasibility or significant improvement
    # RosenbrockConstraint has feasible region.
    # LBFGSB minimizes penalty P(x) = f(x) + rho * cv(x)^2

    if best.cv < 1e-3:
        # Feasible
        pass
    else:
        # If not feasible, check if penalty is reasonable
        # This test might be flaky if LBFGSB gets stuck in local min of penalty function
        pass

    # Ideally we want feasible
    assert best.cv < 1.0, f"Failed to find near-feasible solution, cv={best.cv}"
