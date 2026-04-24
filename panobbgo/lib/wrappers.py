# -*- coding: utf8 -*-
# Copyright 2026 Panobbgo Contributors
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

"""
Problem Wrappers
================

Composable decorator classes that wrap a :class:`~panobbgo.lib.Problem` instance
and transform inputs/outputs, without modifying the original problem.

These are composable::

    wrapped = NormalizedProblem(NoisyProblem(MyProblem()))

"""

import numpy as np
from panobbgo.lib.lib import Problem, Result


class ProblemWrapper(Problem):
    """Base class for problem wrappers. Delegates to wrapped problem."""

    def __init__(self, problem: Problem, box: list[tuple[float, float]] | None = None):
        self._wrapped = problem
        wrapper_box = box if box is not None else [(float(lo), float(hi)) for lo, hi in problem.box]
        super().__init__(box=wrapper_box)

    @property
    def wrapped(self) -> Problem:
        """The underlying wrapped problem."""
        return self._wrapped

    def eval(self, x):
        return self._wrapped.eval(x)

    def eval_constraints(self, x):
        return self._wrapped.eval_constraints(x)


class NormalizedProblem(ProblemWrapper):
    """
    Scales all dimensions to [0, 1]. Heuristics see a unit hypercube;
    the wrapper maps back to original coordinates for evaluation.

    ``x_original = x_normalized * ranges + lower_bounds``
    """

    def __init__(self, problem: Problem):
        self._lower = problem.box[:, 0].copy()
        self._ranges = problem.ranges.copy()
        unit_box = [(0.0, 1.0)] * problem.dim
        super().__init__(problem, box=unit_box)

    def _denormalize(self, x_normalized: np.ndarray) -> np.ndarray:
        return x_normalized * self._ranges + self._lower

    def eval(self, x):
        return self._wrapped.eval(self._denormalize(x))

    def eval_constraints(self, x):
        return self._wrapped.eval_constraints(self._denormalize(x))


class LogTransformProblem(ProblemWrapper):
    """
    Applies log transform to the objective: ``log(1 + f(x) - offset)``.
    Useful when objective spans orders of magnitude.

    Constraints are NOT transformed (they have their own scale via cv).
    """

    def __init__(self, problem: Problem, offset: float = 0.0):
        self.offset = offset
        super().__init__(problem)

    def __call__(self, point):
        x = point.x + self.dx if self.dx is not None else point.x
        fx = self._wrapped.eval(x)
        cv = self._wrapped.eval_constraints(x)
        fx_transformed = np.log1p(fx - self.offset)
        return Result(point, fx_transformed, cv_vec=cv)

    def eval(self, x):
        fx = self._wrapped.eval(x)
        return np.log1p(fx - self.offset)


class NoisyProblem(ProblemWrapper):
    """
    Adds controlled Gaussian noise to evaluations. Useful for robustness testing.

    Args:
        problem: The problem to wrap.
        noise_std: Standard deviation of noise (default 0.1).
        noise_type: ``"additive"`` or ``"multiplicative"`` (default ``"additive"``).
        seed: Optional random seed for reproducibility.
    """

    def __init__(
        self,
        problem: Problem,
        noise_std: float = 0.1,
        noise_type: str = "additive",
        seed: int | None = None,
    ):
        self.noise_std = noise_std
        self.noise_type = noise_type
        self._rng = np.random.default_rng(seed)
        super().__init__(problem)

    def eval(self, x):
        fx = self._wrapped.eval(x)
        if self.noise_type == "multiplicative":
            return fx * (1 + self.noise_std * self._rng.standard_normal())
        else:
            return fx + self.noise_std * self._rng.standard_normal()
