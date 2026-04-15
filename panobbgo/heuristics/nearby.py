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

"""
Nearby Heuristic
================

Generates new candidate points near the current best point.

Optionally integrates with the :class:`~panobbgo.analyzers.sensitivity.Sensitivity`
analyzer: when importance scores are available, perturbations are scaled by dimension
importance so the heuristic focuses search effort on the dimensions that actually matter.
"""

import numpy as np
from panobbgo.analyzers.best import Best
from panobbgo.core import Heuristic


class Nearby(Heuristic):
    """
    Generates new points based on a cheap, fast algorithm.

    For each new best point, generates ``new`` many nearby points by applying
    a random perturbation of magnitude ``radius`` (relative to dimension ranges).

    When a :class:`~panobbgo.analyzers.sensitivity.Sensitivity` analyzer is active
    and has produced importance scores, the perturbation along each dimension is
    scaled proportionally to that dimension's importance score. This focuses the
    local search on dimensions that have the greatest impact on the objective,
    making it significantly more efficient in high-dimensional problems.

    Arguments:

    - ``axes``:

      * ``one``: perturb only one randomly chosen axis
      * ``all``: perturb all axes simultaneously

    - ``new``: number of new points to generate per best update (default: 1)
    - ``radius``: perturbation size as a fraction of the dimension's range (default: 0.01)
    - ``sensitivity_scale``: when sensitivity data is available, scale perturbations
      by dimension importance raised to this power (default: 1.0). Higher values focus
      more aggressively on important dimensions; 0.0 disables sensitivity scaling.
    """

    def __init__(
        self,
        strategy,
        cap: int = 3,
        radius: float = 1.0 / 100,
        new: int = 1,
        axes: str = "one",
        sensitivity_scale: float = 1.0,
    ):
        Heuristic.__init__(
            self, strategy, cap=cap, name="Nearby %.3f/%s" % (radius, axes)
        )
        self.radius = radius
        self.new = new
        self.axes = axes
        self.sensitivity_scale = sensitivity_scale
        self._depends_on = [Best]

        # Importance scores from the Sensitivity analyzer — None until first update
        self._importance: np.ndarray | None = None

    def on_new_sensitivity(self, importance: np.ndarray) -> None:
        """
        Called when the Sensitivity analyzer produces updated importance scores.

        Args:
            importance: Array of shape ``(dim,)`` with values in ``[0, 1]``.
                Higher means more important. Used to scale perturbation per dimension.
        """
        self._importance = importance

    def _perturbation_weights(self) -> np.ndarray | None:
        """
        Return per-dimension perturbation weights based on sensitivity, or None.

        Weights are non-negative and normalised so that their mean equals 1,
        preserving the overall perturbation magnitude set by ``radius``.
        """
        if self._importance is None or self.sensitivity_scale == 0.0:
            return None

        # Raise to sensitivity_scale power for adjustable contrast
        raw = np.maximum(self._importance, 1e-3) ** self.sensitivity_scale

        # Normalise so mean = 1 → overall perturbation magnitude unchanged
        mean_w = raw.mean()
        if mean_w > 0:
            return raw / mean_w
        return None

    def _make_perturbation(self, x: np.ndarray) -> np.ndarray:
        """
        Apply a single random perturbation to *x* and return the new candidate.

        If sensitivity weights are available the perturbation is scaled per-dimension;
        otherwise the standard uniform perturbation is used.
        """
        new_x = x.copy()
        weights = self._perturbation_weights()

        if self.axes == "all":
            dx = (2.0 * np.random.rand(self.problem.dim) - 1.0) * self.radius
            dx *= self.problem.ranges
            if weights is not None:
                dx *= weights
            new_x += dx

        elif self.axes == "one":
            if weights is not None:
                # Sample dimension proportional to importance
                probs = weights / weights.sum()
                idx = np.random.choice(self.problem.dim, p=probs)
            else:
                idx = np.random.randint(self.problem.dim)
            dx = (2.0 * np.random.rand() - 1.0) * self.radius
            dx *= self.problem.ranges[idx]
            if weights is not None:
                dx *= weights[idx]
            new_x[idx] += dx

        else:
            raise ValueError(
                f"Nearby heuristic received invalid 'axes' parameter: '{self.axes}'. "
                f"Valid options are 'one' (perturb one axis) or 'all' (perturb all axes)."
            )

        return self.problem.project(new_x)

    def on_restart(self, center: np.ndarray, reason: str) -> None:
        """
        Respond to a restart event by generating points around the new center.

        Args:
            center: New search center suggested by the Restart analyzer.
            reason: Human-readable reason string for the restart.
        """
        if center is None:
            return
        self.clear_output()
        ret = [self._make_perturbation(center) for _ in range(self.new)]
        self.emit(ret)

    def on_new_best(self, best) -> None:
        """
        React to a new best result by generating nearby candidate points.

        Args:
            best: The new best :class:`~panobbgo.lib.Result`.
        """
        x = best.x
        if x is None:
            return
        self.clear_output()
        ret = [self._make_perturbation(x) for _ in range(self.new)]
        self.emit(ret)
