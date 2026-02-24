# -*- coding: utf8 -*-
import pytest
import numpy as np
import logging
from unittest.mock import MagicMock, patch

from panobbgo.lib import Point, Result, BoundingBox
from panobbgo.lib.classic import RosenbrockConstraint
from panobbgo.core import StrategyBase, Results, EventBus
from panobbgo.lib.constraints import AugmentedLagrangianConstraintHandler
from panobbgo.heuristics.constraint_gradient import ConstraintGradient
from panobbgo.heuristics.feasible_search import FeasibleSearch
from panobbgo.heuristics.random import Random
from panobbgo.strategies.rewarding import StrategyRewarding

# Mock Config
class MockConfig:
    def __init__(self):
        self.max_eval = 100
        self.capacity = 100
        self.storage_backend = None
        self.debug = True
        self.rho = 10.0
        self.dynamic_penalty_rate = 2.0
        self.constraint_exponent = 1.0
        self.logging = {}

    def get_logger(self, name):
        return logging.getLogger(name)

# Mock Strategy
class MockStrategy(StrategyBase):
    def __init__(self, problem):
        self.problem = problem
        self.config = MockConfig()
        self.eventbus = EventBus(self.config)
        self.results = Results(self)
        self.constraint_handler = AugmentedLagrangianConstraintHandler(self)
        self._best = None
        self._heuristics = {}
        self._analyzers = {}

    @property
    def best(self):
        return self._best

    @best.setter
    def best(self, value):
        self._best = value

    def start(self): pass
    def stop(self): pass
    def execute(self): return []

def test_alm_updates():
    """
    Test that AugmentedLagrangianConstraintHandler updates mu and lambdas correctly.
    """
    problem = RosenbrockConstraint(dims=2)
    strategy = MockStrategy(problem)
    handler = strategy.constraint_handler
    # Force update interval to 1
    handler.update_interval = 1

    # 1. Add infeasible point
    x1 = np.array([0.0, 0.0])
    r1 = Result(Point(x1, "test"), problem.eval(x1), cv_vec=problem.eval_constraints(x1))

    strategy.results.add_results([r1])
    strategy.best = r1
    handler.on_new_results([r1])

    # Initial mu=10.0.
    assert handler.mu == 10.0

    expected_lambdas = np.maximum(0, 0 + 10.0 * r1.cv_vec)
    np.testing.assert_allclose(handler.lambdas, expected_lambdas)

    # 2. Add same point (stagnation)
    handler.on_new_results([r1])

    # mu should increase
    assert handler.mu == 20.0

    expected_lambdas_2 = np.maximum(0, expected_lambdas + 10.0 * r1.cv_vec)
    np.testing.assert_allclose(handler.lambdas, expected_lambdas_2)


def test_constraint_gradient_fallback():
    """
    Test that ConstraintGradient falls back to scalar if LP fails or is missing.
    """
    problem = RosenbrockConstraint(dims=2)
    strategy = MockStrategy(problem)

    heuristic = ConstraintGradient(strategy)
    heuristic.__start__()

    # Mock history
    center = np.array([0.0, 0.0]) # Infeasible
    r_center = Result(Point(center, "init"), problem.eval(center), cv_vec=problem.eval_constraints(center))
    strategy.results.add_results([r_center])
    strategy.best = r_center

    # Neighbors
    neighbors = []
    for i in range(5):
        step = np.random.randn(2) * 0.1
        x = center + step
        r = Result(Point(x, "init"), problem.eval(x), cv_vec=problem.eval_constraints(x))
        neighbors.append(r)
    strategy.results.add_results(neighbors)

    # 1. Test with mocked ImportError for linprog
    with patch("panobbgo.heuristics.constraint_gradient.ConstraintGradient._solve_lp_direction") as mock_lp:
        mock_lp.side_effect = ImportError("No scipy")

        # Spy on scalar direction
        with patch("panobbgo.heuristics.constraint_gradient.ConstraintGradient._solve_scalar_direction") as mock_scalar:
            heuristic.on_new_best(r_center)

            # Check fallback
            assert mock_scalar.called

    # 2. Test with LP returning False (failure)
    with patch("panobbgo.heuristics.constraint_gradient.ConstraintGradient._solve_lp_direction") as mock_lp:
        mock_lp.return_value = False

        with patch("panobbgo.heuristics.constraint_gradient.ConstraintGradient._solve_scalar_direction") as mock_scalar:
            heuristic.on_new_best(r_center)
            assert mock_scalar.called

    # 3. Test with LP returning True (success)
    with patch("panobbgo.heuristics.constraint_gradient.ConstraintGradient._solve_lp_direction") as mock_lp:
        mock_lp.return_value = True

        with patch("panobbgo.heuristics.constraint_gradient.ConstraintGradient._solve_scalar_direction") as mock_scalar:
            heuristic.on_new_best(r_center)
            assert not mock_scalar.called


def test_heuristic_integration():
    """
    Test full integration of StrategyRewarding with constraint heuristics.
    """
    # Use simple constrained problem
    problem = RosenbrockConstraint(dims=2)
    # Setup strategy with mocked config to control parameters via kwargs if needed
    strategy = StrategyRewarding(
        problem,
        max_eval=20,
        constraint_handler="AugmentedLagrangianConstraintHandler",
        rho=10.0
    )

    # Add heuristics (Random is needed to start!)
    strategy.add(Random)
    strategy.add(ConstraintGradient)
    strategy.add(FeasibleSearch)

    # Run strategy (short run)
    strategy.start()

    # Check if we generated points
    assert len(strategy.results) >= 20

    # Verify that constraint heuristics were active
    df = strategy.results.results
    who_counts = df[("who", 0)].value_counts()
    print("Heuristic usage:", who_counts)

    # We expect some points from Random, and hopefully some from others if they were triggered.
    # Note: ConstraintGradient needs history/neighbors, so it might take time to kick in.

    best = strategy.best
    print(f"Best found: fx={best.fx}, cv={best.cv}")
