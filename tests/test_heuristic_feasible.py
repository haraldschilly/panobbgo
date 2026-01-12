# -*- coding: utf8 -*-
import pytest
import numpy as np
from unittest import mock
from panobbgo.heuristics.feasible_search import FeasibleSearch
from panobbgo.lib import Point, Result, BoundingBox, Problem
from panobbgo.config import Config

class MockProblem(Problem):
    def __init__(self, dim=2):
        # Fix: box must be a list of tuples, not a numpy array
        box = [(-5.0, 5.0)] * dim
        super().__init__(box=box)

    def eval(self, x):
        # Objective: x^2
        fx = np.sum(x**2)
        # Constraint: x[0] > 1.0 => 1.0 - x[0] <= 0
        # cv = max(0, 1.0 - x[0])
        cv = max(0.0, 1.0 - x[0])
        return Result(Point(x, "eval"), fx, cv_vec=np.array([cv]))

@mock.patch("panobbgo.core.StrategyBase")
def test_feasible_search_initialization(StrategyBaseMock):
    problem = MockProblem()
    config = Config(parse_args=False, testing_mode=True)
    strategy = StrategyBaseMock()
    strategy.problem = problem
    strategy.config = config
    h = FeasibleSearch(strategy)
    assert h.name == "FeasibleSearch"
    assert h.radius == 0.1

@mock.patch("panobbgo.core.StrategyBase")
def test_feasible_search_generates_points_when_infeasible(StrategyBaseMock):
    problem = MockProblem()
    config = Config(parse_args=False, testing_mode=True)
    strategy = StrategyBaseMock()
    strategy.problem = problem
    strategy.config = config
    h = FeasibleSearch(strategy, samples=5)
    h.__start__()

    # Create an infeasible result (x=[0,0], cv=1.0)
    infeasible_point = Point(np.array([0.0, 0.0]), "init")
    infeasible_res = Result(infeasible_point, 0.0, cv_vec=np.array([1.0]))

    # Trigger on_new_best
    h.on_new_best(infeasible_res)

    # Should generate 5 points
    points = h.get_points(10)
    assert len(points) == 5

    # Verify points are within box and distinct
    for p in points:
        # p.x is a numpy array. Point(p.x, ...) would wrap it.
        # problem.box.__contains__ expects a Point object
        assert Point(p.x, "test") in problem.box
        assert not np.allclose(p.x, infeasible_point.x)

@mock.patch("panobbgo.core.StrategyBase")
def test_feasible_search_idle_when_feasible(StrategyBaseMock):
    problem = MockProblem()
    config = Config(parse_args=False, testing_mode=True)
    strategy = StrategyBaseMock()
    strategy.problem = problem
    strategy.config = config
    h = FeasibleSearch(strategy, samples=5)
    h.__start__()

    # Create a feasible result (x=[2,2], cv=0.0)
    feasible_point = Point(np.array([2.0, 2.0]), "init")
    feasible_res = Result(feasible_point, 8.0, cv_vec=np.array([0.0]))

    # Trigger on_new_best
    h.on_new_best(feasible_res)

    # Should NOT generate points (based on current implementation logic)
    points = h.get_points(10)
    assert len(points) == 0
