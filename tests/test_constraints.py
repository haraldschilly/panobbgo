import pytest
import numpy as np
from panobbgo.lib.lib import Result, Point
from panobbgo.lib.constraints import DefaultConstraintHandler

class TestDefaultConstraintHandler:

    def setup_method(self):
        self.handler = DefaultConstraintHandler(rho=100.0)

    def create_result(self, fx, cv_vec=None, who="test"):
        p = Point(np.array([0.0]), who)
        return Result(p, fx, cv_vec=cv_vec)

    def test_first_point_improvement(self):
        # Case: old_best is None
        new_best = self.create_result(10.0)
        imp = self.handler.calculate_improvement(None, new_best)
        assert imp > 0.0

    def test_feasible_improvement(self):
        # Case 1: Both Feasible
        old = self.create_result(10.0) # cv=0
        new = self.create_result(5.0)  # cv=0

        imp = self.handler.calculate_improvement(old, new)
        assert imp == 5.0

        # No improvement
        imp = self.handler.calculate_improvement(new, old)
        assert imp == 0.0

    def test_infeasible_to_feasible(self):
        # Case 2: Old Infeasible, New Feasible
        old_cv = np.array([1.0])
        old = self.create_result(-100.0, cv_vec=old_cv) # cv=1.0
        new = self.create_result(10.0) # cv=0

        imp = self.handler.calculate_improvement(old, new)

        # Improvement should be base (10.0) + rho * old_cv (100.0 * 1.0) = 110.0
        assert imp == 10.0 + 100.0 * 1.0

    def test_infeasible_improvement(self):
        # Case 3: Both Infeasible
        # old cv = 2.0, new cv = 1.0
        old = self.create_result(-50.0, cv_vec=np.array([2.0]))
        new = self.create_result(-60.0, cv_vec=np.array([1.0]))

        imp = self.handler.calculate_improvement(old, new)

        # Improvement should be rho * delta_cv = 100 * (2.0 - 1.0) = 100.0
        assert imp == 100.0

        # old cv = 1.0, new cv = 2.0 (worse)
        imp = self.handler.calculate_improvement(new, old)
        assert imp == 0.0

    def test_feasible_to_infeasible(self):
        # Case 4: Old Feasible, New Infeasible (Regression)
        old = self.create_result(10.0)
        new = self.create_result(-100.0, cv_vec=np.array([1.0]))

        imp = self.handler.calculate_improvement(old, new)
        assert imp == 0.0

    def test_custom_rho(self):
        handler = DefaultConstraintHandler(rho=10.0)
        old = self.create_result(-100.0, cv_vec=np.array([1.0]))
        new = self.create_result(10.0)

        imp = handler.calculate_improvement(old, new)
        assert imp == 10.0 + 10.0 * 1.0
