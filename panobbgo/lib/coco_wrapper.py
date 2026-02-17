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

"""
COCO/BBOB Problem Wrapper
==========================

Wraps a ``cocoex.Problem`` instance as a panobbgo :class:`~panobbgo.lib.lib.Problem`,
enabling panobbgo strategies and heuristics to be benchmarked on the
COCO/BBOB test suite.

Install the COCO experiment framework with::

    pip install coco-experiment

See https://coco-platform.org/getting-started/ for details.
"""

import numpy as np

from panobbgo.lib.lib import Problem


class CocoProblem(Problem):
    """Wraps a ``cocoex.Problem`` as a panobbgo :class:`Problem`.

    Parameters
    ----------
    coco_problem
        A ``cocoex.Problem`` instance obtained from iterating over
        a ``cocoex.Suite``.

    The bounding box is derived from the COCO problem's ``lower_bounds``
    and ``upper_bounds``. Calling :meth:`eval` delegates to the COCO
    problem callable, which automatically increments the COCO evaluation
    counter and triggers observer logging.
    """

    def __init__(self, coco_problem):
        self._coco_problem = coco_problem

        # Build box from COCO bounds: list of (lower, upper) tuples
        box = list(
            zip(
                coco_problem.lower_bounds.tolist(),
                coco_problem.upper_bounds.tolist(),
            )
        )

        super().__init__(box=box)

        self.coco_id: str = coco_problem.id
        self.coco_name: str = coco_problem.name

    @property
    def has_constraints(self) -> bool:
        """Whether the wrapped COCO problem has constraints."""
        return self._coco_problem.number_of_constraints > 0

    @property
    def initial_solution(self) -> np.ndarray:
        """The COCO-provided initial solution."""
        return np.array(self._coco_problem.initial_solution, dtype=np.float64)

    def eval(self, x):
        """Evaluate the objective function.

        Calls the COCO problem callable, which also triggers COCO's
        internal evaluation counter and observer logging.
        """
        return float(self._coco_problem(x))

    def eval_constraints(self, x):
        """Evaluate constraint violations (if any).

        For constrained COCO suites (e.g. ``bbob-constrained``), delegates
        to ``coco_problem.constraint(x)``. Returns a numpy array where
        positive values indicate violation, matching panobbgo's convention.
        """
        if not self.has_constraints:
            return None
        cv = np.array(self._coco_problem.constraint(x), dtype=np.float64)
        # COCO convention: g(x) <= 0 is feasible. Positive = violation.
        # Clamp satisfied constraints to 0.
        cv[cv < 0] = 0.0
        return cv

    def __repr__(self):
        return f"CocoProblem({self.coco_id}, dim={self.dim})"
