# -*- coding: utf8 -*-
from panobbgo.lib.constraints import EpsilonConstraintHandler
from panobbgo.lib import Result
import numpy as np
import pytest

class MockStrategy:
    def __init__(self, results_count=0):
        self.results = list(range(results_count))

def test_epsilon_calculation():
    strategy = MockStrategy(results_count=0)
    # epsilon_start=1.0, cp=2.0, cutoff=100
    handler = EpsilonConstraintHandler(strategy=strategy, epsilon_start=1.0, cp=2.0, cutoff=100)

    # t=0
    assert abs(handler._get_current_epsilon() - 1.0) < 1e-6

    strategy.results = list(range(50)) # t=50
    # eps = 1.0 * (1 - 50/100)^2 = 1.0 * 0.5^2 = 0.25
    assert abs(handler._get_current_epsilon() - 0.25) < 1e-6

    strategy.results = list(range(100)) # t=100
    # eps = 0.0
    assert abs(handler._get_current_epsilon() - 0.0) < 1e-6

    strategy.results = list(range(150)) # t=150
    # eps = 0.0
    assert abs(handler._get_current_epsilon() - 0.0) < 1e-6

def test_is_better_both_feasible_under_epsilon():
    strategy = MockStrategy(results_count=0)
    # eps = 1.0
    handler = EpsilonConstraintHandler(strategy=strategy, epsilon_start=1.0, cutoff=100)

    # Both have cv <= 1.0, so treated as feasible. Compare fx.
    r1 = Result(None, 10.0, cv_vec=np.array([0.5])) # cv=0.5
    r2 = Result(None, 9.0, cv_vec=np.array([0.8]))  # cv=0.8, worse cv but better fx

    # r2 is better because 9.0 < 10.0
    assert handler.is_better(r1, r2)
    assert not handler.is_better(r2, r1)

def test_is_better_one_feasible_under_epsilon():
    strategy = MockStrategy(results_count=0)
    # eps = 1.0
    handler = EpsilonConstraintHandler(strategy=strategy, epsilon_start=1.0, cutoff=100)

    # r1 feasible under epsilon, r2 not
    r1 = Result(None, 10.0, cv_vec=np.array([0.9])) # cv=0.9 <= 1.0
    r2 = Result(None, 5.0, cv_vec=np.array([1.1]))  # cv=1.1 > 1.0

    # r1 is better because it is "feasible"
    assert handler.is_better(r2, r1)
    assert not handler.is_better(r1, r2)

def test_is_better_both_infeasible_under_epsilon():
    strategy = MockStrategy(results_count=0)
    # eps = 1.0
    handler = EpsilonConstraintHandler(strategy=strategy, epsilon_start=1.0, cutoff=100)

    # Both have cv > 1.0. Compare cv.
    r1 = Result(None, 10.0, cv_vec=np.array([1.5]))
    r2 = Result(None, 5.0, cv_vec=np.array([1.2])) # Better cv

    # r2 is better because 1.2 < 1.5
    assert handler.is_better(r1, r2)
    assert not handler.is_better(r2, r1)

def test_calculate_improvement():
    strategy = MockStrategy(results_count=0)
    # eps = 1.0
    handler = EpsilonConstraintHandler(strategy=strategy, epsilon_start=1.0, cutoff=100)

    # Case 1: Both feasible under epsilon -> compare fx
    r1 = Result(None, 10.0, cv_vec=np.array([0.5]))
    r2 = Result(None, 8.0, cv_vec=np.array([0.5]))
    # Improvement = 10.0 - 8.0 = 2.0
    assert abs(handler.calculate_improvement(r1, r2) - 2.0) < 1e-6

    # Case 2: One feasible (r2), one infeasible (r1)
    r1 = Result(None, 10.0, cv_vec=np.array([1.5])) # phi = 0.5
    r2 = Result(None, 10.0, cv_vec=np.array([0.5])) # phi = 0.0
    # Improvement should be related to phi reduction.
    # Let's implement calculate_improvement as reduction in generalized penalty P(x).
    # If phi(x) > 0, P(x) = f_max + rho * phi(x)?
    # Or just phi(x) if comparing violations?
    # Simple approach: if transition from Infeasible -> Feasible (phi > 0 -> phi == 0),
    # return base_reward + rho * phi_old.
    # For EpsilonConstraint, "Feasible" means phi==0.

    # Let's assume implementation uses rho=100.0 (default) for violation reduction.
    # Improvement = 10.0 + 100.0 * 0.5 = 60.0
    assert abs(handler.calculate_improvement(r1, r2) - 60.0) < 1e-6

    # Case 3: Both infeasible (phi > 0)
    r1 = Result(None, 10.0, cv_vec=np.array([1.5])) # phi = 0.5
    r2 = Result(None, 10.0, cv_vec=np.array([1.2])) # phi = 0.2
    # Improvement = (0.5 - 0.2) * 100.0 = 30.0
    assert abs(handler.calculate_improvement(r1, r2) - 30.0) < 1e-6
