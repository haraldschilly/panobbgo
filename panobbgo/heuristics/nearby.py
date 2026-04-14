# -*- coding: utf8 -*-
# Copyright 2012-2026 Harald Schilly <harald.schilly@gmail.com>
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
import numpy as np

from panobbgo.analyzers.best import Best
from panobbgo.core import Heuristic


class Nearby(Heuristic):
    """
    This provider generates new points based
    on a cheap (i.e. fast) algorithm. For each new best point,
    it and generates ``new`` many nearby point(s).
    The @radius is scaled along each dimension's range in the search box.

    When paired with the :class:`~panobbgo.analyzers.sensitivity.Sensitivity` analyzer,
    this heuristic automatically weights perturbations by dimension importance:
    important dimensions receive larger perturbations while unimportant dimensions
    are nearly frozen. For ``axes="one"``, the axis is chosen proportional to
    importance, so important dimensions are sampled more often.

    Arguments::

    - ``axes``:
       * ``one``: only disturb one axis (chosen proportional to importance if available)
       * ``all``: disturb all axes (scaled by importance if available)

    - ``new``: number of new points to generate (default: 1)
    """

    def __init__(self, strategy, cap=3, radius=1.0 / 100, new=1, axes="one"):
        Heuristic.__init__(
            self, strategy, cap=cap, name="Nearby %.3f/%s" % (radius, axes)
        )
        self.radius = radius
        self.new = new
        self.axes = axes
        self._depends_on = [Best]
        # Sensitivity importance scores — updated by on_new_sensitivity
        self._importance: np.ndarray | None = None

    def on_new_sensitivity(self, importance: np.ndarray) -> None:
        """Update dimension importance weights from the Sensitivity analyzer.

        Args:
            importance: Array of shape ``(dim,)`` with values in ``[0, 1]``,
                where higher means the dimension has more impact on the objective.
        """
        self._importance = importance.copy()

    def on_restart(self, center, reason):
        """Respond to a restart event by resetting the search around the new center."""
        if center is None:
            return
        self.clear_output()
        ret = []
        for _ in range(self.new):
            ret.append(self._perturb(center))
        self.emit(ret)

    def _perturb(self, x: np.ndarray) -> np.ndarray:
        """Generate a single perturbed point near ``x``.

        If sensitivity importance scores are available, perturbations are weighted
        by dimension importance so that irrelevant dimensions are barely changed.

        Returns:
            Projected point inside the bounding box.
        """
        new_x = x.copy()
        if self.axes == "all":
            dx = (2.0 * np.random.rand(self.problem.dim) - 1.0) * self.radius
            dx *= self.problem.ranges
            if self._importance is not None:
                # Scale each dimension's step by its importance (important dims move more)
                dx *= self._importance
            new_x += dx
        elif self.axes == "one":
            if self._importance is not None and self._importance.sum() > 0:
                # Sample axis proportional to importance — important dims chosen more often
                probs = self._importance / self._importance.sum()
                idx = int(np.random.choice(self.problem.dim, p=probs))
            else:
                idx = np.random.randint(self.problem.dim)
            dx = (2.0 * np.random.rand() - 1.0) * self.radius
            dx *= self.problem.ranges[idx]
            new_x[idx] += dx
        else:
            raise ValueError(
                f"Nearby heuristic received invalid 'axes' parameter: '{self.axes}'. "
                f"Valid options are 'one' (perturb one axis) or 'all' (perturb all axes)."
            )
        return self.problem.project(new_x)

    def on_new_best(self, best):
        ret = []
        x = best.x
        if x is None:
            return
        self.clear_output()
        for _ in range(self.new):
            ret.append(self._perturb(x))
        self.emit(ret)
