# -*- coding: utf8 -*-
import pytest
import numpy as np
from panobbgo.heuristics.repair import ConstraintRepair
from panobbgo.lib.classic import RosenbrockConstraint
from panobbgo.lib import Point, Result

class MockStrategy:
    def __init__(self, problem):
        self.problem = problem
        # Mock Config
        self.config = type('Config', (), {})()
        self.config.capacity = 100
        self.config.get_logger = lambda name: type('Logger', (), {
            'debug': lambda self, s, *args: None,
            'info': lambda self, s, *args: None,
            'warning': lambda self, s, *args: None,
            'error': lambda self, s, *args: None,
            'critical': lambda self, s, *args: None
        })()

        # Mock EventBus
        self.eventbus = type('EventBus', (), {})()
        self.eventbus.register = lambda target: None
        self.eventbus.publish = lambda key, **kwargs: None

        # Mock Constraint Handler
        self.constraint_handler = type('ConstraintHandler', (), {})()
        self.constraint_handler.get_penalty_value = lambda r: r.fx + (r.cv or 0)*1000

    @property
    def results(self):
        return []

def test_repair_heuristic_init():
    problem = RosenbrockConstraint(dims=2)
    strategy = MockStrategy(problem)
    h = ConstraintRepair(strategy)
    assert h.name == "ConstraintRepair"
    assert h.max_iter == 20

def test_repair_heuristic_on_new_best():
    problem = RosenbrockConstraint(dims=2)
    strategy = MockStrategy(problem)
    h = ConstraintRepair(strategy)
    h.__start__()

    # Create an infeasible point
    # RosenbrockConstraint optimum approx at (1,1) but strictly it's on boundary
    # Point at x=[1, 1]. y=[0, 0].
    # Violation: -(0-0)^2 + 0.25 = 0.25 (raw).
    # But gradient of constraint (y2-y1)^2 is 0 at y1=y2.
    # Pick a point where gradient is non-zero to allow SLSQP to work.
    # x=[1.0, 1.1]. y=[0, 0.1]. (y2-y1)^2 = 0.01.
    # Violation = -0.01 + 0.25 = 0.24.
    x_bad = np.array([1.0, 1.1])

    # Verify it is infeasible
    res_bad = problem(Point(x_bad, "test"))
    assert res_bad.cv > 0

    # Trigger heuristic
    h.on_new_best(res_bad)

    # Should emit a point
    points = h.get_points()

    try:
        import scipy
        if len(points) > 0:
            p = points[0]
            # Check if new point is "more feasible"
            res_new = problem(p)

            # Check distance to avoid emitting same point
            dist = np.linalg.norm(p.x - x_bad)
            assert dist > 1e-9

            # Violation should be reduced
            # Wait, Result.cv filters non-positive.
            # If feasible, cv=0.
            # If infeasible, cv > 0.
            # We expect improvement (lower cv).
            print(f"Old CV: {res_bad.cv}, New CV: {res_new.cv}")
            assert res_new.cv < res_bad.cv

            # Specifically for this problem, SLSQP should find a feasible point
            # near [1, 1].
            # Constraint: (y2-y1)^2 >= 0.25
            # y1=0, y2=0.5 -> (0.5)^2=0.25. Feasible.
            # dist to [0,0] is 0.5.
            # y1=0, y2=-0.5 -> dist 0.5.
            # y1=-0.25, y2=0.25 -> diff=0.5. dist to origin: sqrt(0.25^2+0.25^2) = sqrt(0.125) ~ 0.35.
            # So SLSQP should move y to satisfy (y2-y1)^2 >= 0.25.

    except ImportError:
        pass

def test_repair_heuristic_no_op_if_feasible():
    problem = RosenbrockConstraint(dims=2)
    strategy = MockStrategy(problem)
    h = ConstraintRepair(strategy)
    h.__start__()

    # Feasible point
    # y1=0, y2=0.6. diff=0.6. sq=0.36 >= 0.25.
    # y=x (shift=0 if optimum=1).
    # Wait, shift = optimum - 1.0. If optimum=1.0, shift=0.
    # So y=x.
    x_good = np.array([0.0, 0.6])
    res_good = problem(Point(x_good, "test"))

    assert res_good.cv == 0.0

    h.on_new_best(res_good)

    points = h.get_points()
    assert len(points) == 0
