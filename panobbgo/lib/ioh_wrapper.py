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
IOHprofiler Problem Wrapper
===========================

Wraps an ``ioh.problem.RealSingleObjective`` instance as a panobbgo
:class:`~panobbgo.lib.lib.Problem`, enabling panobbgo strategies and
heuristics to be benchmarked on the IOHprofiler test suites (BBOB,
MA-BBOB, ...).

Install the IOH experimenter with::

    uv add --optional benchmark ioh

See https://iohprofiler.github.io/IOHexp/ for details.
"""

from __future__ import annotations

import numpy as np

from panobbgo.lib.lib import Problem


class IOHProblem(Problem):
    """Wraps an ``ioh.problem.RealSingleObjective`` as a panobbgo Problem.

    Parameters
    ----------
    ioh_problem
        An ``ioh.problem.RealSingleObjective`` instance, typically obtained
        from ``ioh.get_problem(fid, instance, dimension, suite)`` or by
        iterating an ``ioh.Suite``.

    Notes
    -----
    The bounding box is derived from the IOH problem's ``bounds.lb`` and
    ``bounds.ub``. Each :meth:`eval` call delegates to the wrapped problem,
    which automatically increments IOH's internal evaluation counter and
    triggers any attached logger.
    """

    def __init__(self, ioh_problem):
        self._ioh_problem = ioh_problem

        lb = np.asarray(ioh_problem.bounds.lb, dtype=np.float64)
        ub = np.asarray(ioh_problem.bounds.ub, dtype=np.float64)
        box = list(zip(lb.tolist(), ub.tolist()))

        super().__init__(box=box)

        meta = ioh_problem.meta_data
        self.ioh_problem_id: int = int(meta.problem_id)
        self.ioh_instance: int = int(meta.instance)
        self.ioh_name: str = str(meta.name)

    @property
    def optimum_y(self) -> float:
        """Known optimum f-value (IOH problems are shifted to a known minimum)."""
        return float(self._ioh_problem.optimum.y)

    def eval(self, x: np.ndarray) -> float:
        return float(self._ioh_problem(x))

    def eval_constraints(self, x: np.ndarray):
        # MA-BBOB / BBOB are unconstrained
        return None

    def reset(self) -> None:
        """Reset the wrapped IOH problem's evaluation counter."""
        self._ioh_problem.reset()

    def __repr__(self) -> str:
        return (
            f"IOHProblem(id={self.ioh_problem_id}, "
            f"instance={self.ioh_instance}, dim={self.dim}, name={self.ioh_name!r})"
        )
