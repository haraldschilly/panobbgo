import pytest
import numpy as np
from panobbgo.lib import Problem, BoundingBox, Result, Point
from panobbgo.lib.constraints import (
    DefaultConstraintHandler,
    PenaltyConstraintHandler,
    DynamicPenaltyConstraintHandler,
    AugmentedLagrangianConstraintHandler
)

# --- Mocks and Helpers ---

class MockConfig:
    def __init__(self, **kwargs):
        self.logging = {}
        for k, v in kwargs.items():
            setattr(self, k, v)

    def get_logger(self, name):
        import logging
        return logging.getLogger(name)

class MockStrategy:
    def __init__(self, problem, config=None):
        self.problem = problem
        self.config = config if config else MockConfig()
        self.results = []
        self.best = None
        self.constraint_handler = None

    def update_best(self):
        # Update self.best based on constraint handler
        if not self.results:
            return

        current_best = self.best
        for r in self.results:
            if self.constraint_handler.is_better(current_best, r):
                current_best = r
        self.best = current_best

# --- Test Problem ---

class ConstrainedSphere(Problem):
    """
    Minimize f(x) = x^2 + y^2
    Subject to g(x) = x + y - 1 <= 0
    Global min is at (0,0) which is feasible.
    Wait, global min of sphere is (0,0), 0+0-1 = -1 <= 0. Feasible.
    Let's make it harder.
    Minimize f(x) = x^2 + y^2
    Subject to g(x) = 1 - x - y <= 0  => x + y >= 1.
    Global min should be at (0.5, 0.5), fx = 0.5.
    (0,0) is infeasible (g(0,0) = 1 > 0).
    """
    def __init__(self):
        # Problem expects a list of tuples for box, not (dim, BoundingBox)
        # It initializes BoundingBox internally
        super().__init__([[-2, 2], [-2, 2]])

    def eval(self, x):
        return np.sum(x**2)

    def eval_constraints(self, x):
        # g(x) = 1 - x[0] - x[1] <= 0
        val = 1.0 - x[0] - x[1]
        # Return positive value for violation
        return np.array([max(0.0, val)])

# --- Tests ---

def test_default_constraint_handler():
    # Lexicographic: Feasible < Infeasible
    # If both feasible: smaller fx is better
    # If both infeasible: smaller cv is better

    problem = ConstrainedSphere()
    strategy = MockStrategy(problem)
    handler = DefaultConstraintHandler(strategy)
    strategy.constraint_handler = handler

    # 1. Feasible vs Infeasible
    # Feasible: (0.5, 0.5) -> fx=0.5, cv=0
    # Infeasible: (0, 0) -> fx=0.0, cv=1.0 (violation is 1.0)

    r_feasible = Result(Point(np.array([0.5, 0.5]), "test"), 0.5, cv_vec=np.array([0.0]))
    r_infeasible = Result(Point(np.array([0.0, 0.0]), "test"), 0.0, cv_vec=np.array([1.0]))

    assert handler.is_better(r_infeasible, r_feasible) == True, "Feasible should beat Infeasible"
    assert handler.is_better(r_feasible, r_infeasible) == False, "Infeasible should not beat Feasible"

    # Improvement calculation
    # Crossing from infeasible to feasible should give large reward
    improv = handler.calculate_improvement(r_infeasible, r_feasible)
    assert improv > 10.0

    # 2. Both Feasible
    # (0.5, 0.5) fx=0.5
    # (0.6, 0.6) fx=0.72, feasible
    r_feasible_worse = Result(Point(np.array([0.6, 0.6]), "test"), 0.72, cv_vec=np.array([0.0]))

    assert handler.is_better(r_feasible_worse, r_feasible) == True, "Lower fx should be better"
    assert handler.calculate_improvement(r_feasible_worse, r_feasible) == pytest.approx(0.22)

    # 3. Both Infeasible
    # (0,0) fx=0, cv=1
    # (0.1, 0.1) fx=0.02, cv=0.8
    r_infeasible_better = Result(Point(np.array([0.1, 0.1]), "test"), 0.02, cv_vec=np.array([0.8]))

    assert handler.is_better(r_infeasible, r_infeasible_better) == True, "Lower cv should be better"
    assert handler.calculate_improvement(r_infeasible, r_infeasible_better) == pytest.approx(0.2 * handler.rho)


def test_penalty_constraint_handler():
    # P(x) = f(x) + rho * cv(x)^exponent
    rho = 10.0

    problem = ConstrainedSphere()
    strategy = MockStrategy(problem)
    handler = PenaltyConstraintHandler(strategy, rho=rho, exponent=1.0)
    strategy.constraint_handler = handler

    # r1: fx=0.0, cv=1.0 -> P = 0 + 10*1 = 10
    r1 = Result(Point(np.array([0.0, 0.0]), "test"), 0.0, cv_vec=np.array([1.0]))

    # r2: fx=0.5, cv=0.0 -> P = 0.5 + 10*0 = 0.5
    r2 = Result(Point(np.array([0.5, 0.5]), "test"), 0.5, cv_vec=np.array([0.0]))

    # r3: fx=0.1, cv=0.5 -> P = 0.1 + 10*0.5 = 5.1
    r3 = Result(Point(np.array([0.1, 0.1]), "test"), 0.1, cv_vec=np.array([0.5]))

    assert handler.is_better(r1, r2) == True # 0.5 < 10
    assert handler.is_better(r1, r3) == True # 5.1 < 10
    assert handler.is_better(r3, r2) == True # 0.5 < 5.1

    assert handler.calculate_improvement(r1, r3) == pytest.approx(4.9)

def test_augmented_lagrangian_handler_updates():
    problem = ConstrainedSphere()
    strategy = MockStrategy(problem)
    handler = AugmentedLagrangianConstraintHandler(strategy, rho=2.0, rate=2.0, update_interval=2)
    strategy.constraint_handler = handler

    # Initial state
    assert handler.mu == 2.0
    assert handler.lambdas is None

    # 1. Add some results
    r1 = Result(Point(np.array([0.0, 0.0]), "test"), 0.0, cv_vec=np.array([1.0])) # Infeasible
    strategy.best = r1 # Assume this is best so far (only point)
    strategy.results.append(r1)

    handler.on_new_results([r1])
    # Counter should be 1, no update yet
    assert handler.counter == 1
    assert handler.lambdas is None

    # 2. Add another result to trigger update
    r2 = Result(Point(np.array([0.0, 0.0]), "test"), 0.0, cv_vec=np.array([1.0]))
    # Still best is r1 (same)
    strategy.results.append(r2)

    handler.on_new_results([r2])
    # Counter reset
    assert handler.counter == 0
    # Update should have happened
    assert handler.lambdas is not None

    # Check lambda update
    # lambda_new = max(0, lambda_old + mu * cv)
    # lambda_0 = 0
    # lambda_1 = max(0, 0 + 2.0 * 1.0) = 2.0
    np.testing.assert_array_equal(handler.lambdas, np.array([2.0]))

    # Check mu update
    # cv_norm = 1.0. last_cv_norm was inf.
    # 1.0 < 0.9 * inf -> True. But wait, logic is:
    # if current_cv_norm > 0.9 * self.last_cv_norm: mu *= rate
    # Initial last_cv_norm is inf. So 1.0 is NOT > inf.
    # So mu should NOT increase on first update.
    assert handler.mu == 2.0
    assert handler.last_cv_norm == 1.0

    # 3. Add more results, still high violation
    # Trigger another update
    handler.on_new_results([r1, r2])

    # Now last_cv_norm is 1.0. Current is 1.0.
    # 1.0 > 0.9 * 1.0 -> True.
    # So mu should increase.
    assert handler.mu == 4.0

    # Lambda should update again using the NEW mu (4.0)
    # lambda_2 = max(0, 2.0 + 4.0 * 1.0) = 6.0
    np.testing.assert_array_equal(handler.lambdas, np.array([6.0]))


def test_augmented_lagrangian_improvement():
    problem = ConstrainedSphere()
    strategy = MockStrategy(problem)
    handler = AugmentedLagrangianConstraintHandler(strategy, rho=1.0)
    strategy.constraint_handler = handler

    # Setup lambdas
    handler.lambdas = np.array([1.0])
    handler.mu = 1.0

    # L = f(x) + (1/2mu) * ( max(0, lambda + mu*g)^2 - lambda^2 )

    # Point 1: f=0, g=1
    # term = max(0, 1 + 1*1) = 2
    # penalty = 0.5 * (4 - 1) = 1.5
    # L = 1.5
    r1 = Result(Point(np.array([0,0]), "t"), 0.0, cv_vec=np.array([1.0]))

    # Point 2: f=0.5, g=0
    # term = max(0, 1 + 1*0) = 1
    # penalty = 0.5 * (1 - 1) = 0
    # L = 0.5
    r2 = Result(Point(np.array([0.5,0.5]), "t"), 0.5, cv_vec=np.array([0.0]))

    assert handler.is_better(r1, r2) == True # 0.5 < 1.5
    assert handler.calculate_improvement(r1, r2) == pytest.approx(1.0)
