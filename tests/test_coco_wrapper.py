# -*- coding: utf8 -*-
# Copyright 2012-2026 Panobbgo Contributors
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

"""Tests for the COCO/BBOB problem wrapper."""

import numpy as np
import pytest

# Skip entire module if cocoex is not installed
cocoex = pytest.importorskip("cocoex", reason="coco-experiment not installed")

from panobbgo.lib.coco_wrapper import CocoProblem  # noqa: E402
from panobbgo.lib.lib import Point, Result  # noqa: E402


def _get_coco_problem(dim=2, func_idx=1, inst_idx=1, suite_name="bbob"):
    """Helper to get a single COCO problem."""
    suite = cocoex.Suite(
        suite_name,
        "",
        f"dimensions: {dim} function_indices: {func_idx} instance_indices: {inst_idx}",
    )
    return next(iter(suite))


class TestCocoProblemBounds:
    """Test that CocoProblem correctly translates COCO bounds."""

    def setup_method(self):
        self.coco_problem = _get_coco_problem()
        self.problem = CocoProblem(self.coco_problem)

    def test_dimension(self):
        assert self.problem.dim == self.coco_problem.dimension

    def test_box_shape(self):
        assert self.problem.box.box.shape == (self.coco_problem.dimension, 2)

    def test_lower_bounds(self):
        for i in range(self.problem.dim):
            assert self.problem.box[i, 0] == pytest.approx(self.coco_problem.lower_bounds[i])

    def test_upper_bounds(self):
        for i in range(self.problem.dim):
            assert self.problem.box[i, 1] == pytest.approx(self.coco_problem.upper_bounds[i])


class TestCocoProblemEval:
    """Test that eval() delegates correctly to the COCO problem."""

    def setup_method(self):
        self.coco_problem = _get_coco_problem()
        self.problem = CocoProblem(self.coco_problem)

    def test_eval_returns_float(self):
        x = self.problem.initial_solution
        result = self.problem.eval(x)
        assert isinstance(result, float)

    def test_eval_is_finite(self):
        x = self.problem.initial_solution
        result = self.problem.eval(x)
        assert np.isfinite(result)

    def test_call_returns_result(self):
        """Calling problem(point) should return a Result."""
        x = self.problem.initial_solution
        point = Point(x, "test")
        result = self.problem(point)
        assert isinstance(result, Result)
        assert isinstance(result.fx, (float, np.floating))
        assert result.who == "test"

    def test_random_point_in_bounds(self):
        """random_point() should generate points within COCO bounds."""
        for _ in range(10):
            x = self.problem.random_point()
            assert len(x) == self.problem.dim
            for i in range(self.problem.dim):
                assert x[i] >= self.problem.box[i, 0]
                assert x[i] <= self.problem.box[i, 1]

    def test_coco_evaluations_increment(self):
        """Each eval() call should increment COCO's evaluation counter."""
        before = self.coco_problem.evaluations
        x = self.problem.initial_solution
        self.problem.eval(x)
        assert self.coco_problem.evaluations == before + 1


class TestCocoProblemUnconstrained:
    """Test that unconstrained COCO problems have no constraints."""

    def setup_method(self):
        self.coco_problem = _get_coco_problem()
        self.problem = CocoProblem(self.coco_problem)

    def test_has_no_constraints(self):
        assert not self.problem.has_constraints

    def test_eval_constraints_returns_none(self):
        x = self.problem.initial_solution
        assert self.problem.eval_constraints(x) is None


class TestCocoProblemMultipleDimensions:
    """Test wrapper across different COCO dimensions."""

    @pytest.mark.parametrize("dim", [2, 5, 10])
    def test_dimension_mapping(self, dim):
        coco_prob = _get_coco_problem(dim=dim)
        problem = CocoProblem(coco_prob)
        assert problem.dim == dim
        assert problem.box.box.shape == (dim, 2)

    @pytest.mark.parametrize("func_idx", [1, 5, 10, 15, 20, 24])
    def test_different_functions(self, func_idx):
        coco_prob = _get_coco_problem(func_idx=func_idx)
        problem = CocoProblem(coco_prob)
        x = problem.initial_solution
        val = problem.eval(x)
        assert np.isfinite(val)


class TestCocoProblemRepr:
    """Test string representation."""

    def test_repr_contains_id(self):
        coco_prob = _get_coco_problem()
        problem = CocoProblem(coco_prob)
        r = repr(problem)
        assert "CocoProblem" in r
        assert "dim=" in r


class TestCocoProblemConstrained:
    """Test wrapper with constrained COCO problems."""

    @pytest.fixture
    def constrained_problem(self):
        try:
            coco_prob = _get_coco_problem(suite_name="bbob-constrained")
            return CocoProblem(coco_prob)
        except Exception:
            pytest.skip("bbob-constrained suite not available")

    def test_has_constraints(self, constrained_problem):
        assert constrained_problem.has_constraints

    def test_eval_constraints_returns_array(self, constrained_problem):
        x = constrained_problem.initial_solution
        cv = constrained_problem.eval_constraints(x)
        assert isinstance(cv, np.ndarray)
        assert len(cv) > 0

    def test_constraint_values_non_negative(self, constrained_problem):
        """Constraint violations should be >= 0 (clamped)."""
        x = constrained_problem.initial_solution
        cv = constrained_problem.eval_constraints(x)
        assert np.all(cv >= 0)
