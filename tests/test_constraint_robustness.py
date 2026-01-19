
import pytest
import numpy as np
from panobbgo.lib import Problem, Point, Result
from panobbgo.lib.constraints import AugmentedLagrangianConstraintHandler
from panobbgo.heuristics.constraint_gradient import ConstraintGradient
from panobbgo.heuristics.feasible_search import FeasibleSearch
from panobbgo.strategies.rewarding import StrategyRewarding

class ConstrainedProblem(Problem):
    def __init__(self, dim=2):
        super().__init__([[-5, 5]] * dim)

    def eval(self, x):
        return np.sum(x**2)

    def eval_constraints(self, x):
        # x[0] + x[1] >= 1  =>  1 - x[0] - x[1] <= 0
        return np.array([max(0.0, 1.0 - np.sum(x[:2]))])

def test_alm_update_logic():
    """
    Test that Augmented Lagrangian updates use the correct penalty parameter for multiplier updates.
    """
    class MockStrategy:
        def __init__(self):
            self.config = MockConfig()
            self.best = None
            self.results = []

    class MockConfig:
        def __init__(self):
            self.logging = {}
        def get_logger(self, name):
            import logging
            return logging.getLogger(name)

    strategy = MockStrategy()
    # update_interval=1 to force update
    handler = AugmentedLagrangianConstraintHandler(strategy, rho=10.0, rate=2.0, update_interval=1)

    r1 = Result(Point(np.array([0.0]), "t"), 0.0, cv_vec=np.array([1.0]))
    strategy.best = r1
    strategy.results.append(r1)

    # 1st update: Initial logic
    handler.on_new_results([r1])
    # mu should be 10.0, lambdas [10.0]
    assert handler.mu == 10.0
    assert np.allclose(handler.lambdas, [10.0])

    # 2nd update: Stagnation -> Increase mu
    handler.on_new_results([r1])

    # mu increases to 20.0
    assert handler.mu == 20.0

    # lambda update should use OLD mu (10.0)
    # lambda_new = max(0, 10.0 + 10.0 * 1.0) = 20.0
    # If it used new mu (20.0), it would be 30.0
    assert np.allclose(handler.lambdas, [20.0])

def test_constraint_gradient_dataframe_access():
    """
    Test that ConstraintGradient can access results from DataFrame correctly.
    """
    problem = ConstrainedProblem(dim=2)
    strategy = StrategyRewarding(problem, testing_mode=True)
    strategy.add(ConstraintGradient)

    # Manually populate results
    # We need enough points to fit gradient
    # Gradient of cv = 1-x-y is [-1, -1]. Descent direction [1, 1].

    # p1: [0,0], cv=1
    r1 = Result(Point(np.array([0.0, 0.0]), "t"), 0.0, cv_vec=np.array([1.0]))
    # p2: [0.1, 0], cv=0.9
    r2 = Result(Point(np.array([0.1, 0.0]), "t"), 0.0, cv_vec=np.array([0.9]))
    # p3: [0, 0.1], cv=0.9
    r3 = Result(Point(np.array([0.0, 0.1]), "t"), 0.0, cv_vec=np.array([0.9]))

    strategy.results.add_results([r1, r2, r3])

    # Manually initialize heuristic since we didn't call start()
    for h_class_or_instance in strategy._hs:
        # StrategyBase.add puts instances in _hs
        strategy.add_heuristic(h_class_or_instance)

    # Trigger heuristic with a bad point
    bad = Result(Point(np.array([-1.0, -1.0]), "b"), 0.0, cv_vec=np.array([3.0]))

    cg = strategy.heuristic("ConstraintGradient")
    cg.on_new_best(bad)

    points = cg.get_points(1)
    assert len(points) == 1
    p = points[0]

    # Should move in +x, +y direction from bad point [-1, -1]
    assert p.x[0] > -1.0
    assert p.x[1] > -1.0

def test_feasible_search_integration():
    """
    Test FeasibleSearch integration in a strategy run.
    """
    problem = ConstrainedProblem(dim=2)
    strategy = StrategyRewarding(problem, testing_mode=True)
    strategy.config.max_eval = 50
    strategy.config.evaluation_method = "threaded"

    strategy.add(FeasibleSearch)
    # Add Random to ensure we have points
    from panobbgo.heuristics.random import Random
    strategy.add(Random)

    # Run briefly
    strategy.start()

    assert len(strategy.results) > 0
