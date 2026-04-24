import pytest
import numpy as np
from panobbgo.lib import Problem, Point, Result
from panobbgo.heuristics.constraint_gradient import ConstraintGradient
from panobbgo.core import StrategyBase, Results, EventBus
from unittest.mock import MagicMock


class ConstrainedSphere(Problem):
    """
    Minimize (x-center)^2 s.t. x <= bound
    Optimum at boundary if center > bound.
    Returns signed constraint values (negative = satisfied).
    """

    def __init__(self, dim=2, center=2.0, bound=1.0):
        super().__init__([(-5, 5)] * dim)
        self.target = np.full(dim, center)
        self.bound = bound

    def eval(self, x):
        return np.sum((x - self.target) ** 2)

    def eval_constraints(self, x):
        # x_i <= bound  =>  x_i - bound <= 0
        return x - self.bound


def test_constraint_gradient_feasible_direction():
    """
    Test that ConstraintGradient generates feasible descent direction
    when starting from a feasible point but unconstrained gradient points to infeasible region.
    """
    dim = 2
    problem = ConstrainedSphere(dim=dim, center=2.0, bound=1.0)  # Opt at [1,1], constrained by x<=1

    # Mock strategy
    strategy = MagicMock(spec=StrategyBase)
    strategy.problem = problem
    strategy.config = MagicMock()
    strategy.config.capacity = 100  # Set valid capacity
    strategy.config.get_logger = MagicMock()
    strategy.eventbus = MagicMock(spec=EventBus)  # Set eventbus BEFORE Results init
    strategy.results = Results(strategy)  # Use real Results to handle history
    strategy.best = None  # Not used directly by heuristic, passed in on_new_best

    heuristic = ConstraintGradient(strategy, samples=1)

    # Create history:
    # 1. Best point at [0,0] (Feasible). fx high.
    # 2. Neighbors around [0,0] to establish gradients.

    # We need enough neighbors to estimate gradient.
    # fx = (x-2)^2. Grad at 0 is -4.
    # cv = x - 1. Grad is 1.

    # Add points
    points = [[0.0, 0.0], [0.01, 0.0], [0.0, 0.01], [-0.01, 0.0], [0.0, -0.01]]

    results = []
    for p_arr in points:
        p_obj = Point(np.array(p_arr), "init")
        r = problem(p_obj)
        results.append(r)
        strategy.results.add_results([r])

    # The best point is [0,0] (locally for this set, though not globally)
    # Actually [0.01, 0.01] is closer to [2,2] but we treat [0,0] as current best for the test
    best = results[0]

    # Heuristic should use LP to find direction
    # Grad fx at 0: [-4, -4]. Direction should be [1, 1]
    # Grad cv at 0: [1, 0] and [0, 1] (diagonal matrix)
    # Constraint: d_i <= -(0 - 1) = 1.
    # Descent direction d should be positive.

    heuristic.on_new_best(best)

    # Check emitted points
    q = heuristic._output
    assert q.qsize() > 0

    candidates = []
    while not q.empty():
        candidates.append(q.get())

    for cand in candidates:
        # Check if candidate improves objective
        val = problem.eval(cand.x)
        assert val < best.fx

        # Check if candidate is feasible (or close to boundary)
        # x <= 1. Since we step 0.1 * range (range=10), step is 1.0.
        # d is normalized? No, LP output d is bounded by delta.
        # delta = 0.1 * 10 = 1.0.
        # If d = [1, 1], new x = [1, 1]. This is feasible (boundary).
        assert np.all(cand.x <= problem.bound + 1e-5)

        # Also check direction
        # x_new - x_old should be positive (towards [2,2])
        diff = cand.x - best.x
        assert np.all(diff > 0)


def test_constraint_gradient_infeasible_repair():
    """
    Test that ConstraintGradient repairs infeasible point.
    """
    dim = 2
    problem = ConstrainedSphere(dim=dim, center=0.0, bound=1.0)
    # Unconstrained opt at 0,0. Feasible.
    # But we start at [2,2] which is infeasible (bound=1).

    # Mock strategy
    strategy = MagicMock(spec=StrategyBase)
    strategy.problem = problem
    strategy.config = MagicMock()
    strategy.config.capacity = 100
    strategy.config.capacity = 100
    strategy.config.get_logger = MagicMock()
    strategy.eventbus = MagicMock(spec=EventBus)
    strategy.results = Results(strategy)

    heuristic = ConstraintGradient(strategy, samples=1)

    # History around [2,2]
    points = [
        [2.0, 2.0],  # Infeasible, cv=[1,1]
        [2.01, 2.0],
        [2.0, 2.01],
        [1.99, 2.0],
        [2.0, 1.99],
    ]

    results = []
    for p_arr in points:
        p_obj = Point(np.array(p_arr), "init")
        r = problem(p_obj)
        results.append(r)
        strategy.results.add_results([r])

    best = results[0]  # [2,2]

    # Gradient of cv (x-1) is 1.
    # We want to reduce cv. Descent direction should be [-1, -1].

    heuristic.on_new_best(best)

    q = heuristic._output
    assert q.qsize() > 0

    candidates = []
    while not q.empty():
        candidates.append(q.get())

    for cand in candidates:
        # Check if candidate reduces violation
        cv_vec = problem.eval_constraints(cand.x)
        # Original cv at [2,2] is [1,1]
        assert np.all(cv_vec < best.cv_vec)

        # Check direction (should be negative)
        diff = cand.x - best.x
        assert np.all(diff < 0)


def test_constraint_gradient_fallback_scalar():
    """
    Test fallback to scalar CV if vector info missing.
    """
    dim = 2
    problem = ConstrainedSphere(dim=dim, center=0.0, bound=1.0)

    strategy = MagicMock(spec=StrategyBase)
    strategy.problem = problem
    strategy.config = MagicMock()
    strategy.config.capacity = 100
    strategy.config.get_logger = MagicMock()
    strategy.eventbus = MagicMock(spec=EventBus)
    strategy.results = Results(strategy)

    heuristic = ConstraintGradient(strategy, samples=1)

    # Add points with missing cv_vec (simulated by modifying results)
    points = [[2.0, 2.0], [2.01, 2.0], [2.0, 2.01], [1.99, 2.0], [2.0, 1.99]]

    results = []
    for p_arr in points:
        p_obj = Point(np.array(p_arr), "init")
        r = problem(p_obj)
        # Force remove cv_vec
        r._cv_vec = None
        # But ensure cv is set (norm)
        # Wait, Result calculates cv from cv_vec.
        # If cv_vec is None, cv is 0.
        # We need to manually set cv > 0 for this test logic to trigger "repair"
        # Since we can't easily modify Result property behavior, we just use a problem that returns None for cv_vec?
        # But Problem always returns cv_vec.
        # We can hack the result object?
        # Actually, Results._flush_buffer fills 0 if None.

        # Let's just mock get_history to return None for cv_vec
        results.append(r)

    strategy.results.add_results(results)

    # Mock get_history to return scalar cv but no cv_vec
    def mock_get_history(n=None):
        X = np.array(points)
        # cv = norm([1,1]) approx 1.414 for [2,2]
        # gradient of scalar cv (distance to region) points to region.
        cv = np.linalg.norm(X - 1.0, axis=1)  # simplified
        return {
            "x": X,
            "fx": np.zeros(len(X)),  # irrelevant
            "cv": cv,
            "cv_vec": None,  # MISSING VECTOR INFO
        }

    strategy.results.get_history = mock_get_history

    # Set best.cv > 0 manually
    best = results[0]

    # We need to monkeypatch 'cv' property or create a mock Result
    # Result.cv is a property.
    # We can create a dummy object with attributes
    class DummyResult:
        def __init__(self, x, cv):
            self.x = np.array(x)
            self.cv = cv
            self.fx = 0.0
            self.cv_vec = None

    dummy_best = DummyResult([2.0, 2.0], 1.414)

    heuristic.on_new_best(dummy_best)

    q = heuristic._output
    assert q.qsize() > 0

    cand = q.get()
    # Should move towards feasible region (decrease coordinates)
    assert np.all(cand.x < 2.0)
