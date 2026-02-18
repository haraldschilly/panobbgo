
import pytest
import numpy as np
from panobbgo.heuristics.constraint_gradient import ConstraintGradient
from panobbgo.lib import Point, Result, Problem
from panobbgo.strategies.rewarding import StrategyRewarding

class MockProblem(Problem):
    def __init__(self):
        box = [(-5, 5), (-5, 5)]
        super().__init__(box)

    def eval(self, x):
        return np.sum(x**2) # Minimize x^2 + y^2

    def eval_constraints(self, x):
        # Constraint: x[0] + x[1] - 2 <= 0
        return np.array([x[0] + x[1] - 2])

class MockProblemInfeasible(Problem):
    def __init__(self):
        box = [(-5, 5), (-5, 5)]
        super().__init__(box)

    def eval(self, x):
        return np.sum(x**2)

    def eval_constraints(self, x):
        # Constraint: 5 - (x[0] + x[1]) <= 0  =>  x+y >= 5
        return np.array([5 - (x[0] + x[1])])

def test_gradient_estimation_scalar():
    problem = MockProblemInfeasible()
    strategy = StrategyRewarding(problem, testing_mode=True)
    cg = ConstraintGradient(strategy, step_size=1e-2, descent_step=0.1)

    # Add some history points around (0,0)
    # Target: Estimate gradient of CV at (0,0).
    # CV = 5 - (x+y). Grad = (-1, -1).
    # Descent direction = -Grad = (1, 1).

    base_x = np.array([0.0, 0.0])
    points = [
        base_x,
        base_x + [0.1, 0.0],
        base_x + [-0.1, 0.0],
        base_x + [0.0, 0.1],
        base_x + [0.0, -0.1],
    ]

    results = []
    for i, x in enumerate(points):
        p = Point(x, "init")
        fx = problem.eval(x)
        # cv_vec is [5 - (x+y)]
        cv_vec = np.maximum(0, problem.eval_constraints(x))
        r = Result(p, fx, cv_vec=cv_vec)
        results.append(r)

    # Add to strategy results
    strategy.results.add_results(results)

    # Trigger generation from base_x
    best = results[0] # (0,0), cv=5

    emitted = []
    def mock_emit(pts):
        emitted.extend(pts)
    cg.emit = mock_emit

    cg._generate_descent_point(best)

    assert len(emitted) > 0

    p_new = emitted[0]
    direction = p_new - base_x
    print(f"Descent direction: {direction}")

    # Check direction
    # Expected direction is roughly proportional to (1,1)
    assert direction[0] > 0
    assert direction[1] > 0
    assert abs(direction[0] - direction[1]) < 0.1

def test_gradient_estimation_vector():
    # Test LP formulation
    problem = MockProblemInfeasible()
    strategy = StrategyRewarding(problem, testing_mode=True)
    cg = ConstraintGradient(strategy, step_size=1e-2, descent_step=0.1)

    base_x = np.array([0.0, 0.0])
    points = [
        base_x,
        base_x + [0.1, 0.0],
        base_x + [-0.1, 0.0],
        base_x + [0.0, 0.1],
        base_x + [0.0, -0.1],
    ]

    results = []
    for i, x in enumerate(points):
        p = Point(x, "init")
        fx = problem.eval(x)
        cv_vec = np.maximum(0, problem.eval_constraints(x))
        r = Result(p, fx, cv_vec=cv_vec)
        results.append(r)

    strategy.results.add_results(results)
    best = results[0]

    emitted = []
    def mock_emit(pts):
        emitted.extend(pts)
    cg.emit = mock_emit

    cg._generate_descent_point(best)

    assert len(emitted) > 0
    p_new = emitted[0]
    direction = p_new - base_x
    print(f"Vector Descent direction: {direction}")

    # Should also be positive
    assert direction[0] > 0
    assert direction[1] > 0

def test_active_sampling():
    problem = MockProblemInfeasible()
    strategy = StrategyRewarding(problem, testing_mode=True)
    cg = ConstraintGradient(strategy, step_size=1e-2, descent_step=0.1)

    # Start with only one point (best)
    base_x = np.array([0.0, 0.0])
    p = Point(base_x, "init")
    fx = problem.eval(base_x)
    cv_vec = np.maximum(0, problem.eval_constraints(base_x))
    best_res = Result(p, fx, cv_vec=cv_vec)

    strategy.results.add_results([best_res])

    emitted = []
    def mock_emit(pts):
        emitted.extend(pts)
    cg.emit = mock_emit

    # This should trigger active sampling because history is insufficient
    cg.on_new_best(best_res)

    # Should have emitted perturbation points
    assert len(emitted) > 0, "Should emit perturbation points"
    # Wait, on_new_best calls clear_output. But mock_emit captures them.
    # The heuristic calls emit(candidates).

    print(f"Emitted {len(emitted)} points for sampling")

    # Verify we are waiting
    assert cg._waiting_for_samples

    # Capture the emitted points to simulate results
    sample_points = list(emitted)
    emitted.clear()

    # Now simulate results coming back
    results = []
    for x in sample_points:
        p = Point(x, "ConstraintGradient")
        fx = problem.eval(x)
        cv_vec = np.maximum(0, problem.eval_constraints(x))
        r = Result(p, fx, cv_vec=cv_vec)
        results.append(r)

    strategy.results.add_results(results)

    # Trigger on_new_results (mimic EventBus)
    cg.on_new_results(results)

    # Now it should have calculated gradient and emitted descent step
    assert len(emitted) > 0, "Should emit descent step after sampling"
    assert not cg._waiting_for_samples

    # Verify descent step direction
    p_new = emitted[0]
    direction = p_new - base_x
    print(f"Descent direction after sampling: {direction}")

    assert direction[0] > 0
    assert direction[1] > 0
