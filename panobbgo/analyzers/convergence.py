# -*- coding: utf8 -*-
# Copyright 2025 Harald Schilly <harald.schilly@gmail.com>
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

from __future__ import unicode_literals

import numpy as np
from panobbgo.core import Analyzer
from collections import deque


class Convergence(Analyzer):
    """
    Analyzes the progress of the optimization to detect convergence or stagnation.

    It listens to ``new_results`` events and tracks the history of the best objective function values.
    If the improvement over a specified window of evaluations is below a threshold,
    or if the standard deviation of the best values in the window is small enough,
    it publishes a ``converged`` event.

    Configuration parameters (via ``strategy.config`` or kwargs):
    - ``convergence.window_size`` (int): Number of recent best values to consider (default: 50).
    - ``convergence.threshold`` (float): Threshold for relative improvement or std dev (default: 1e-6).
    - ``convergence.mode`` (str): 'std' (standard deviation) or 'improv' (relative improvement) (default: 'std').

    Events published:
    - ``converged``: When convergence criteria are met.
      The event carries ``reason`` (str) and ``stats`` (dict) with details.
    """

    def __init__(self, strategy, window_size=None, threshold=None, mode=None):
        super(Convergence, self).__init__(strategy)
        self.logger = self.config.get_logger("CONVG")

        # Configuration with fallbacks
        self.window_size = int(window_size or getattr(self.config, 'convergence_window_size', 50))
        self.threshold = float(threshold or getattr(self.config, 'convergence_threshold', 1e-6))
        self.mode = mode or getattr(self.config, 'convergence_mode', 'std')

        self.history = deque(maxlen=self.window_size)
        self._converged = False

    def on_new_results(self, results):
        """
        Listen to all results to detect stagnation (no improvement for many evaluations).
        """
        current_best = self.strategy.best
        if current_best is None:
            return

        # We append the current best value for each new result.
        # Note: If a large batch arrives and no improvement happens, we might fill the window
        # with identical values. This is intentional: it represents stagnation over N evaluations.
        for _ in results:
            self.history.append(current_best.fx)

        self._check_convergence()

    def _check_convergence(self):
        if self._converged:
            return

        if len(self.history) < self.window_size:
            return

        values = np.array(self.history)

        # To avoid premature convergence detection (e.g., initial batch with no improvement),
        # we can require at least some variation in the history if we are in 'std' mode?
        # Or simply rely on the fact that if 50 points failed, we ARE stagnant.
        # However, to be safe, let's ensure we are not just looking at the very first batch
        # if the window size equals the batch size.
        # But we don't know the batch size here.

        if self.mode == 'std':
            std = np.std(values)
            if std < self.threshold:
                self._trigger_convergence(f"Standard deviation {std:.2e} < {self.threshold:.2e}")

        elif self.mode == 'improv':
            # Check relative improvement between start and end of window
            start = values[0]
            end = values[-1]
            if abs(start) > 1e-9:
                rel_improv = (start - end) / abs(start)
            else:
                rel_improv = start - end

            if rel_improv < self.threshold:
                self._trigger_convergence(f"Relative improvement {rel_improv:.2e} < {self.threshold:.2e}")

    def _trigger_convergence(self, reason):
        self._converged = True
        self.logger.info(f"Converged! {reason}")

        # We publish the event. Strategies or other listeners can decide to stop.
        self.eventbus.publish("converged", reason=reason, stats={'best_fx': self.history[-1]})
