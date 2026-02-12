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

from panobbgo.strategies.rewarding import StrategyRewarding
from panobbgo.lib.classic import RosenbrockConstraint
from panobbgo.heuristics import Center, Nearby, NelderMead, Random, FeasibleSearch

def test_augmented_lagrangian_integration():
    """
    Test that the Augmented Lagrangian constraint handler works in a real optimization.
    We use the RosenbrockConstraint problem.

    This is stochastic — retry up to 3 times to reduce flakiness.
    """
    last_error = None
    for attempt in range(3):
        try:
            _run_alm_trial()
            return
        except AssertionError as e:
            last_error = e
    raise last_error  # type: ignore[misc]


def _run_alm_trial():
    problem = RosenbrockConstraint(dims=2)

    strategy = StrategyRewarding(problem)
    strategy.config.max_eval = 1000
    strategy.config.convergence_require_feasibility = True

    from panobbgo.lib.constraints import AugmentedLagrangianConstraintHandler

    alm_handler = AugmentedLagrangianConstraintHandler(strategy=strategy, rho=1.0, rate=2.0, update_interval=10)
    strategy.constraint_handler = alm_handler
    strategy.eventbus.register(alm_handler)

    strategy.add(Center)
    strategy.add(Random)
    strategy.add(NelderMead)
    strategy.add(Nearby)
    strategy.add(FeasibleSearch)

    strategy.start()

    best = strategy.best
    print(f"Best found: {best}")
    print(f"FX: {best.fx}, CV: {best.cv}")

    assert best.cv < 0.1, f"Solution is not feasible! CV={best.cv}"
    assert best.fx < -10.0, f"Solution is feasible but poor quality: {best.fx}"
    assert alm_handler.mu >= 1.0
    if alm_handler.lambdas is not None:
        print(f"Final Lambdas: {alm_handler.lambdas}")

if __name__ == "__main__":
    test_augmented_lagrangian_integration()
