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
import warnings
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
    - ``convergence.min_evaluations`` (int): Minimum number of evaluations before checking convergence (default: window_size).
    - ``convergence.threshold`` (float): Threshold for relative improvement or std dev (default: 1e-6).
    - ``convergence.mode`` (str): 'std' (standard deviation), 'improv' (relative improvement), or 'slope' (linear regression slope) (default: 'std').

    Events published:
    - ``converged``: When convergence criteria are met.
      The event carries ``reason`` (str) and ``stats`` (dict) with details.
    """

    def __init__(self, strategy, window_size=None, threshold=None, mode=None, min_evaluations=None):
        super(Convergence, self).__init__(strategy)
        self.logger = self.config.get_logger("CONVG")

        # Configuration with fallbacks
        self.window_size = int(window_size or getattr(self.config, 'convergence_window_size', 50))
        self.min_evaluations = int(min_evaluations or getattr(self.config, 'convergence_min_evaluations', self.window_size))
        self.threshold = float(threshold or getattr(self.config, 'convergence_threshold', 1e-6))
        self.mode = mode or getattr(self.config, 'convergence_mode', 'std')
        self.require_feasibility = getattr(self.config, 'convergence_require_feasibility', False)

        self.history = deque(maxlen=self.window_size)
        self.cv_history = deque(maxlen=self.window_size)
        self._converged = False

    def on_new_results(self, results):
        """
        Listen to all results to detect stagnation (no improvement for many evaluations).
        """
        current_best = self.strategy.best
        if current_best is None or current_best.fx is None:
            return

        # We append the current best value for each new result.
        # Note: If a large batch arrives and no improvement happens, we might fill the window
        # with identical values. This is intentional: it represents stagnation over N evaluations.
        for _ in results:
            self.history.append(current_best.fx)
            # Track constraint violation if available, else 0.0
            cv = current_best.cv if current_best.cv is not None else 0.0
            self.cv_history.append(cv)

        self._check_convergence()

    def _check_convergence(self):
        if self._converged:
            return

        # Ensure we have enough data points
        if len(self.history) < self.window_size:
            return

        # Ensure we have processed enough total evaluations
        # We can estimate total evals by strategy results count, or just by history length if we assume
        # history started filling from 0. But history is capped.
        # Strategy results length is better.
        if hasattr(self.strategy, 'results') and self.strategy.results is not None:
            try:
                if len(self.strategy.results) < self.min_evaluations:
                    return
            except TypeError:
                pass

        # Check feasibility if required
        if self.require_feasibility:
            best = self.strategy.best
            if best and best.cv > 1e-6:
                # Still significantly infeasible, do not trigger convergence yet
                return

        # Check if constraints are improving
        # If CV is decreasing, we are making progress towards feasibility, so don't stop.
        if len(self.cv_history) >= self.window_size:
            cv_values = np.array(self.cv_history)
            # If start > end significantly, we are improving
            # Check relative improvement in CV
            start_cv = cv_values[0]
            end_cv = cv_values[-1]
            if start_cv > 1e-6:
                cv_improv = (start_cv - end_cv) / start_cv
                if cv_improv > self.threshold:
                    # Significant CV improvement, reset/delay convergence
                    return

        values = np.array(self.history)

        # Handle None values
        if any(v is None for v in values):
            return

        if self.mode == 'std':
            self._check_std_convergence(values)
        elif self.mode == 'improv':
            self._check_improv_convergence(values)
        elif self.mode == 'slope':
            self._check_slope_convergence(values)

    def _check_std_convergence(self, values):
        # Suppress warnings for edge cases
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            std = np.std(values)

        if std < self.threshold:
            self._trigger_convergence(f"Standard deviation {std:.2e} < {self.threshold:.2e}")

    def _check_improv_convergence(self, values):
        start = values[0]
        end = values[-1]

        if abs(start) > 1e-9:
            rel_improv = (start - end) / abs(start)
        else:
            rel_improv = start - end

        if rel_improv < self.threshold:
            self._trigger_convergence(f"Relative improvement {rel_improv:.2e} < {self.threshold:.2e}")

    def _check_slope_convergence(self, values):
        """
        Check convergence using linear regression slope.
        """
        n = len(values)
        x = np.arange(n)

        # Fit y = mx + c
        try:
            slope, _ = np.polyfit(x, values, 1)
        except Exception:
            return

        # Normalize slope by the range of values or absolute mean to make it scale-independent?
        # Absolute slope depends on the scale of y.
        # Relative slope: slope / mean(y) (if mean != 0)
        mean_val = np.mean(np.abs(values))
        if mean_val > 1e-9:
            norm_slope = slope / mean_val
        else:
            norm_slope = slope

        # We converge if the slope is effectively zero (flat) or positive (no improvement possible for best so far)
        # Note: best so far is non-increasing, so slope <= 0.
        # If slope is very close to 0 (e.g. > -threshold), we are stagnant.

        # Also check residuals/variance to ensure we are not fluctuating wildly?
        # But best so far is monotonic, so it can't fluctuate. It is a step function.

        if abs(norm_slope) < self.threshold:
             self._trigger_convergence(f"Slope {norm_slope:.2e} < {self.threshold:.2e}")

    def _trigger_convergence(self, reason):
        self._converged = True
        self.logger.info(f"Converged! {reason}")

        # We publish the event. Strategies or other listeners can decide to stop.
        self.eventbus.publish("converged", reason=reason, stats={'best_fx': self.history[-1]})
