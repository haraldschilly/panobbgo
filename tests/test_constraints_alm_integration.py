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
from panobbgo.lib import Point
import numpy as np
import pytest
import time

def test_augmented_lagrangian_integration():
    """
    Test that the Augmented Lagrangian constraint handler works in a real optimization.
    We use the RosenbrockConstraint problem.
    """

    # 2D Rosenbrock Constraint problem
    # Global minimum is at (-1, -1) with f(x)=0 ?
    # Let's check RosenbrockConstraint docs or code.
    # panobbgo/lib/classic.py:
    # RosenbrockConstraint:
    # ...
    # where y = x - optimum + 1
    # eval: sum( ... ) - 50.
    # Standard Rosenbrock min is 0 at y=1 -> x=optimum.
    # Here, we have constraints.
    # Constraints: (y[i+1] - y[i])^2 >= par2  AND y[i] >= 0.

    # Let's use simple parameters.
    # optimum=1 (default), par1=100, par2=0.25
    # y = x.
    # f(x) = par1*(x2-x1^2)^2 + (1-x1)^2 - 50
    # Constraints:
    # (x2-x1)^2 >= 0.25  => |x2-x1| >= 0.5
    # x >= 0

    # Global unconstrained min is x=(1,1).
    # Check if (1,1) satisfies constraints.
    # |1-1| = 0 < 0.5. Violated!
    # So the constrained minimum is different.

    problem = RosenbrockConstraint(dims=2)

    # Configure strategy to use AugmentedLagrangianConstraintHandler
    # We can pass config via kwargs or set it later?
    # StrategyBase.__init__ uses Config(parse_args=False, ...).
    # We can override config values.

    # Note: StrategyRewarding inherits StrategyBase.
    # StrategyBase.__init__ reads config.constraint_handler.

    # We can hack the config by creating a mock config or setting sys.argv?
    # Better: StrategyBase.__init__ initializes config.
    # If we pass testing_mode=True, it loads defaults.

    # We can rely on StrategyRewarding using StrategyBase's logic.
    # But we need to inject the configuration.

    # StrategyBase looks at `config.constraint_handler`.
    # Let's see how `Config` works. It loads from files or defaults.
    # We can create a config file or use `panobbgo.config.Config` singleton?
    # `panobbgo.core.Config` is a class.

    # Actually, StrategyBase takes `testing_mode`.

    # Let's monkeypatch Config or just set the attribute on the strategy instance
    # and re-initialize handler?

    # The clean way: Create a strategy, then SWAP the handler.
    # StrategyBase does:
    # self.constraint_handler = ...
    # self.eventbus.register(self.constraint_handler)

    # So we can just:
    strategy = StrategyRewarding(problem)
    strategy.config.max_eval = 1000
    strategy.config.convergence_require_feasibility = True

    # Swap handler to AugmentedLagrangian
    from panobbgo.lib.constraints import AugmentedLagrangianConstraintHandler

    # Unregister default handler
    # The default handler is registered to 'new_results' usually? No, handlers don't subscribe usually?
    # Wait, AugmentedLagrangianConstraintHandler implements on_new_results!
    # DefaultConstraintHandler does NOT implement on_new_results (it just has calculate_improvement/is_better).
    # Let's check constraints.py again.
    # DefaultConstraintHandler inherits ConstraintHandler.
    # ConstraintHandler base class does not have on_new_results.
    # AugmentedLagrangianConstraintHandler has on_new_results.

    # StrategyBase registers the handler:
    # self.eventbus.register(self.constraint_handler)

    # So we should unregister the old one (if it was registered).
    # DefaultConstraintHandler has no on_... methods, so register() might not do much (no threads spawned).

    # Create new handler
    alm_handler = AugmentedLagrangianConstraintHandler(strategy=strategy, rho=1.0, rate=2.0, update_interval=10)
    strategy.constraint_handler = alm_handler
    strategy.eventbus.register(alm_handler)

    # Add heuristics
    strategy.add(Center)
    strategy.add(Random)
    strategy.add(NelderMead)
    strategy.add(Nearby)
    strategy.add(FeasibleSearch)

    # Run
    strategy.start()

    # Check results
    best = strategy.best
    print(f"Best found: {best}")
    print(f"FX: {best.fx}, CV: {best.cv}")

    # Check if we found a feasible solution
    assert best.cv < 1e-4, f"Solution is not feasible! CV={best.cv}"

    # Check if it's reasonably good (Rosenbrock constrained usually finds something < -40)
    # The unconstrained min is -50.
    assert best.fx < -40.0, f"Solution is feasible but poor quality: {best.fx}"

    # Check if handler parameters evolved
    assert alm_handler.mu >= 1.0
    if alm_handler.lambdas is not None:
        print(f"Final Lambdas: {alm_handler.lambdas}")

if __name__ == "__main__":
    test_augmented_lagrangian_integration()
