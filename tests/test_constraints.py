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

from panobbgo.lib.constraints import (
    DefaultConstraintHandler,
    PenaltyConstraintHandler,
    DynamicPenaltyConstraintHandler,
)
from panobbgo.lib.lib import Result
import numpy as np


class MockStrategy:
    def __init__(self, results_count=0):
        self.results = list(range(results_count))


def test_default_handler_feasible_improvement():
    handler = DefaultConstraintHandler()

    # Old best: fx=10, cv=0
    old = Result(None, 10.0, cv_vec=None)

    # New best: fx=5, cv=0
    new = Result(None, 5.0, cv_vec=None)

    # Improvement should be 5.0
    imp = handler.calculate_improvement(old, new)
    assert imp == 5.0


def test_default_handler_infeasible_to_feasible():
    handler = DefaultConstraintHandler(rho=100.0)

    # Old best: fx=1, cv=0.1
    old = Result(None, 1.0, cv_vec=np.array([0.1]))
    # old.cv will be 0.1

    # New best: fx=100, cv=0 (feasible but worse fx)
    new = Result(None, 100.0, cv_vec=None)

    # Improvement should be 10.0 + 100.0 * 0.1 = 20.0
    imp = handler.calculate_improvement(old, new)
    assert abs(imp - 20.0) < 1e-6


def test_default_handler_cv_reduction():
    handler = DefaultConstraintHandler(rho=100.0)

    # Old best: fx=1, cv=0.5
    old = Result(None, 1.0, cv_vec=np.array([0.5]))

    # New best: fx=1, cv=0.2
    new = Result(None, 1.0, cv_vec=np.array([0.2]))

    # Improvement: (0.5 - 0.2) * 100 = 30.0
    imp = handler.calculate_improvement(old, new)
    assert abs(imp - 30.0) < 1e-6


def test_penalty_handler_linear():
    handler = PenaltyConstraintHandler(rho=10.0, exponent=1.0)

    # Old: fx=10, cv=1. P = 10 + 10*1 = 20
    old = Result(None, 10.0, cv_vec=np.array([1.0]))

    # New: fx=12, cv=0.5. P = 12 + 10*0.5 = 17
    new = Result(None, 12.0, cv_vec=np.array([0.5]))

    # Improvement: 20 - 17 = 3.0
    imp = handler.calculate_improvement(old, new)
    assert abs(imp - 3.0) < 1e-6


def test_penalty_handler_quadratic():
    handler = PenaltyConstraintHandler(rho=10.0, exponent=2.0)

    # Old: fx=10, cv=2. P = 10 + 10*(2^2) = 50
    old = Result(None, 10.0, cv_vec=np.array([2.0]))

    # New: fx=20, cv=1. P = 20 + 10*(1^2) = 30
    new = Result(None, 20.0, cv_vec=np.array([1.0]))

    # Improvement: 50 - 30 = 20.0
    imp = handler.calculate_improvement(old, new)
    assert abs(imp - 20.0) < 1e-6


def test_dynamic_penalty_handler():
    # Start rho=10, rate=1.0 per eval
    # Strategy has 1 result -> rho = 10 * (1 + 1*1) = 20
    strategy = MockStrategy(results_count=1)
    handler = DynamicPenaltyConstraintHandler(strategy=strategy, rho_start=10.0, rate=1.0, exponent=1.0)

    # Old: fx=10, cv=1. P = 10 + 20*1 = 30
    old = Result(None, 10.0, cv_vec=np.array([1.0]))

    # New: fx=5, cv=1. P = 5 + 20*1 = 25
    new = Result(None, 5.0, cv_vec=np.array([1.0]))

    # Improvement: 30 - 25 = 5.0
    imp = handler.calculate_improvement(old, new)
    assert abs(imp - 5.0) < 1e-6

    # Now simulate later stage: 9 results
    strategy.results = list(range(9))
    # rho = 10 * (1 + 1*9) = 100

    # Old: fx=10, cv=1. P = 10 + 100*1 = 110
    old = Result(None, 10.0, cv_vec=np.array([1.0]))

    # New: fx=12, cv=0. P = 12 + 100*0 = 12
    new = Result(None, 12.0, cv_vec=None)

    # Improvement: 110 - 12 = 98.0
    imp = handler.calculate_improvement(old, new)
    assert abs(imp - 98.0) < 1e-6
