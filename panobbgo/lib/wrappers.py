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

    wrapped = NormalizedProblem(LogTransformProblem(MyProblem()))

"""

import warnings

import numpy as np
from panobbgo.lib.lib import Problem, Result
from panobbgo.lib.noise import (
    AdditiveGaussianNoise,
    MultiplicativeGaussianNoise,
    NoiseModel,
    NoisyProblem as _DeterministicNoisyProblem,
)


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
        x = point.x - self.dx if self.dx is not None else point.x
        fx = self._wrapped.eval(x)
        cv = self._wrapped.eval_constraints(x)
        fx_transformed = np.log1p(fx - self.offset)
        return Result(point, fx_transformed, cv_vec=cv)

    def eval(self, x):
        fx = self._wrapped.eval(x)
        return np.log1p(fx - self.offset)


class NoisyProblem(_DeterministicNoisyProblem):
    """Deprecated: use :class:`panobbgo.lib.noise.NoisyProblem` (exported as ``panobbgo.lib.NoisyProblem``).

    Kept for the old signature.  Gaussian noise on the raw value —
    ``f + noise_std·ε`` (``"additive"``) or ``f·(1 + noise_std·ε)``
    (``"multiplicative"``) — with fresh noise per evaluation, now drawn from
    the deterministic ``(seed, x, k)`` stream of the noise module instead of
    one generator shared by every evaluator thread (whose draws depended on
    thread scheduling).  ``seed=None`` picks a random seed.
    """

    def __init__(
        self,
        problem: Problem,
        noise_std: float = 0.1,
        noise_type: str = "additive",
        seed: int | None = None,
    ):
        warnings.warn(
            "panobbgo.lib.wrappers.NoisyProblem is deprecated; use panobbgo.lib.noise.NoisyProblem "
            "(panobbgo.lib.NoisyProblem) with a NoiseModel.",
            DeprecationWarning,
            stacklevel=2,
        )
        if noise_type == "multiplicative":
            model: NoiseModel = MultiplicativeGaussianNoise(sigma=noise_std)
        elif noise_type == "additive":
            model = AdditiveGaussianNoise(sigma=noise_std)
        else:
            raise ValueError("noise_type must be 'additive' or 'multiplicative', got %r" % (noise_type,))
        if seed is None:
            seed = int(np.random.SeedSequence().generate_state(1)[0])
        self.noise_std = noise_std
        self.noise_type = noise_type
        super().__init__(problem, model, seed=seed, resample=True)

    def apply_noise(self, x: np.ndarray, true_fx: float) -> float:
        # The legacy wrapper corrupts the raw value, not the precision above
        # f_opt (which would clamp every value below 0 to 0).
        if not np.isfinite(true_fx):
            return true_fx
        return float(self.model.apply(true_fx, self._noise_rng(x)))
