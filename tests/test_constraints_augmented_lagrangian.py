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

from panobbgo.lib.constraints import AugmentedLagrangianConstraintHandler
from panobbgo.lib import Result
import numpy as np
import pytest

class MockStrategy:
    def __init__(self, best=None):
        self.best = best
        self.config = MockConfig()

class MockConfig:
    def get_logger(self, name):
        return MockLogger()

class MockLogger:
    def info(self, msg): pass
    def debug(self, msg): pass

def test_augmented_lagrangian_initialization():
    handler = AugmentedLagrangianConstraintHandler(rho=10.0, rate=2.0, update_interval=5)
    assert handler.mu == 10.0
    assert handler.rate == 2.0
    assert handler.update_interval == 5
    assert handler.lambdas is None

def test_augmented_lagrangian_calculation():
    handler = AugmentedLagrangianConstraintHandler(rho=2.0)
    # Initialize lambdas
    handler.lambdas = np.array([1.0]) # lambda for one constraint

    # Case 1: Feasible point (cv_vec <= 0, technically cv_vec is violations so it should be 0 if satisfied?)
    # Wait, in Panobbgo, cv_vec contains constraint values.
    # Usually g(x) <= 0.
    # panobbgo.lib.Problem.eval_constraints returns values.
    # Result.cv calculates norm of positive entries.
    # So positive entries are violations.
    # g(x) > 0 is violation.

    # Result: fx=10, cv_vec=[0.0] (feasible)
    res = Result(None, 10.0, cv_vec=np.array([0.0]))

    # L = f(x) + 1/(2mu) * (max(0, lambda + mu*g)^2 - lambda^2)
    # lambda=1, mu=2, g=0
    # term = max(0, 1 + 2*0) = 1
    # penalty = 1/(4) * (1^2 - 1^2) = 0
    # L = 10 + 0 = 10
    L = handler._calculate_lagrangian(res)
    assert abs(L - 10.0) < 1e-6

    # Case 2: Infeasible point
    # Result: fx=10, cv_vec=[1.0]
    res_inf = Result(None, 10.0, cv_vec=np.array([1.0]))

    # lambda=1, mu=2, g=1
    # term = max(0, 1 + 2*1) = 3
    # penalty = 1/(4) * (3^2 - 1^2) = 1/4 * (9 - 1) = 2
    # L = 10 + 2 = 12
    L = handler._calculate_lagrangian(res_inf)
    assert abs(L - 12.0) < 1e-6

def test_augmented_lagrangian_update_parameters():
    strategy = MockStrategy()
    handler = AugmentedLagrangianConstraintHandler(strategy=strategy, rho=2.0, rate=2.0, update_interval=2)

    # Set up initial state
    # Best point is currently infeasible
    strategy.best = Result(None, 10.0, cv_vec=np.array([1.0]))
    handler.lambdas = np.array([0.0])

    # Trigger update (simulating new results)
    # Calling on_new_results twice to trigger update (update_interval=2)
    results = [Result(None, 11.0, cv_vec=np.array([1.0]))]
    handler.on_new_results(results) # counter = 1
    handler.on_new_results(results) # counter = 2 -> update

    # Update logic:
    # 1. Check if mu needs increase.
    #    current_cv_norm = 1.0. last_cv_norm was inf.
    #    last_cv_norm becomes 1.0. mu stays 2.0.
    # 2. Update lambdas.
    #    lambda_new = max(0, lambda + mu * cv)
    #    lambda = 0, mu = 2, cv = 1
    #    lambda_new = max(0, 0 + 2*1) = 2.0

    assert handler.mu == 2.0
    assert np.allclose(handler.lambdas, np.array([2.0]))
    assert handler.last_cv_norm == 1.0

    # Trigger another update
    # Assume best is still same (no progress in feasibility)
    handler.on_new_results(results)
    handler.on_new_results(results)

    # Update logic:
    # 1. Check mu.
    #    current_cv_norm = 1.0. last_cv_norm = 1.0.
    #    1.0 > 0.9 * 1.0 is True.
    #    mu *= rate -> mu = 2.0 * 2.0 = 4.0
    #    last_cv_norm = 1.0
    # 2. Update lambdas (using OLD mu = 2.0)
    #    lambda_prev = 2.0
    #    lambda_new = max(0, 2.0 + 2.0 * 1.0) = 4.0

    assert handler.mu == 4.0
    assert np.allclose(handler.lambdas, np.array([4.0]))


def test_augmented_lagrangian_improvement():
    handler = AugmentedLagrangianConstraintHandler(rho=2.0)
    handler.lambdas = np.array([1.0])

    # Old best (Lagrangian = 12.0 from previous test)
    old = Result(None, 10.0, cv_vec=np.array([1.0]))

    # New best (Feasible, L=10.0)
    new = Result(None, 10.0, cv_vec=np.array([0.0]))

    imp = handler.calculate_improvement(old, new)
    assert abs(imp - 2.0) < 1e-6

def test_is_better():
    handler = AugmentedLagrangianConstraintHandler(rho=2.0)
    handler.lambdas = np.array([1.0])

    # L=12
    r1 = Result(None, 10.0, cv_vec=np.array([1.0]))
    # L=10
    r2 = Result(None, 10.0, cv_vec=np.array([0.0]))

    # r2 is better than r1
    assert handler.is_better(r1, r2)
    assert not handler.is_better(r2, r1)
