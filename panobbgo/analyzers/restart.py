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
Restart Analyzer
================

Detects when the optimizer is stuck in a local optimum and publishes
a ``restart`` event, allowing heuristics and the strategy to shift
their search to a different region — without losing accumulated results.

Heuristics that implement ``on_restart(center, reason)`` should:

1. Call ``clear_output()`` to flush stale points.
2. Reset their internal model/state.
3. Begin generating points around the new ``center``.

Heuristics without ``on_restart`` simply continue as before (graceful degradation).

.. note::

   If both :class:`Restart` and :class:`~panobbgo.analyzers.convergence.Convergence`
   are active, set ``patience`` lower than ``Convergence.window_size`` so that restarts
   trigger *before* convergence declares the optimization done.
"""

from __future__ import unicode_literals

import numpy as np
from panobbgo.core import Analyzer


class Restart(Analyzer):
    """
    Detects search stagnation and publishes ``restart`` events.

    Listens to ``new_results``, tracks the best penalty value in a sliding window,
    and fires a ``restart`` event when no sufficient improvement is found within
    ``patience`` evaluations.

    Configuration parameters:

    - ``patience`` (int): Evaluations without improvement before restart (default: 5 * dim).
    - ``improvement_threshold`` (float): Minimum relative improvement to reset counter (default: 1e-6).
    - ``max_restarts`` (int): Stop restarting after this many (default: 10).
    - ``restart_strategy`` (str): How to pick the new center.  One of:

      * ``"random"`` — uniform draw inside the box (default).
      * ``"diverse"`` — uniform draw inside the box, picked from 20
        candidates to maximize the minimum distance to previous restart
        centers.  Strong coverage signal once a few restarts have fired.
      * ``"sphere"`` — Gaussian draw centered at the box centre with
        ``std = ranges / 6`` (clipped to the box).  Useful when the
        optimum is expected to lie inside the box rather than near its
        boundary; biases the restart cloud toward the centroid.

    Events published:

    - ``restart(center=ndarray, reason=str)``: Suggested new center for search.
    """

    SUPPORTED_RESTART_STRATEGIES = ("random", "diverse", "sphere")

    def __init__(
        self,
        strategy,
        patience=None,
        improvement_threshold=1e-6,
        max_restarts=10,
        restart_strategy="random",
    ):
        super().__init__(strategy)
        self.logger = self.config.get_logger("RSTAR")

        if restart_strategy not in self.SUPPORTED_RESTART_STRATEGIES:
            raise ValueError(
                f"restart_strategy must be one of {self.SUPPORTED_RESTART_STRATEGIES!r}, got {restart_strategy!r}"
            )

        self._patience_cfg = patience
        self._patience: int = 0  # resolved in __start__
        self._improvement_threshold = improvement_threshold
        self._max_restarts = max_restarts
        self._restart_strategy = restart_strategy

        self._best_penalty: float = float("inf")
        self._evals_since_improvement = 0
        self._restart_count = 0
        self._previous_centers: list[np.ndarray] = []

    def __start__(self):
        self._patience = self._patience_cfg if self._patience_cfg is not None else 5 * self.problem.dim

    @property
    def restart_count(self) -> int:
        """Number of restarts triggered so far."""
        return self._restart_count

    def on_new_results(self, results):
        if self._restart_count >= self._max_restarts:
            return

        handler = getattr(self.strategy, "constraint_handler", None)

        improved = False
        for r in results:
            if r.fx is None:
                continue
            penalty = handler.get_penalty_value(r) if handler else r.fx

            if penalty < self._best_penalty:
                if self._best_penalty == float("inf"):
                    self._best_penalty = penalty
                    improved = True
                else:
                    rel_improvement = (self._best_penalty - penalty) / max(abs(self._best_penalty), 1e-12)
                    if rel_improvement > self._improvement_threshold:
                        self._best_penalty = penalty
                        improved = True

        if improved:
            self._evals_since_improvement = 0
        else:
            self._evals_since_improvement += len(results)

        if self._evals_since_improvement >= self._patience:
            self._trigger_restart()

    def _trigger_restart(self):
        center = self._pick_new_center()
        self._restart_count += 1
        self._evals_since_improvement = 0
        self._previous_centers.append(center)

        reason = f"No improvement for {self._patience} evaluations (restart {self._restart_count}/{self._max_restarts})"
        self.logger.info(f"Restart triggered: {reason}")
        self.eventbus.publish("restart", center=center, reason=reason)

    def _pick_new_center(self) -> np.ndarray:
        if self._restart_strategy == "diverse" and self._previous_centers:
            return self._diverse_center()
        if self._restart_strategy == "sphere":
            return self.problem.random_point(distribution="normal")
        return self.problem.random_point()

    def _diverse_center(self) -> np.ndarray:
        """Pick the point that maximizes minimum distance to previous centers."""
        best_candidate = None
        best_min_dist = -1.0

        for _ in range(20):  # try 20 random candidates
            candidate = self.problem.random_point()
            min_dist = min(float(np.linalg.norm(candidate - c)) for c in self._previous_centers)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_candidate = candidate

        assert best_candidate is not None
        return best_candidate
