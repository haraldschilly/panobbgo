# -*- coding: utf8 -*-
from __future__ import division
from __future__ import unicode_literals

# Copyright 2025-2026 Panobbgo Contributors
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

import threading

import numpy as np

from panobbgo.core import StrategyBase
from panobbgo.strategies._bandit import (
    LINUCB_ALPHA,
    LINUCB_DIM,
    collect_pulls,
    init_linucb,
    linucb_context,
    linucb_observe,
    linucb_reward,
    linucb_select,
    linucb_update,
    push_recent,
)


class StrategyLinUCB(StrategyBase):
    """
    This strategy uses the Linear Upper Confidence Bound (LinUCB) algorithm (disjoint)
    to select heuristics based on context features.

    Context features ($x_t$):
    - Bias (1.0)
    - Budget Progress (t / T_max)
    - Recent Global Success Rate (last 100 evaluations)

    Model for each heuristic $a$:
    - $\\theta_a^* = \\arg\\min_{\\theta} \\sum (r_t - \\theta^T x_{t,a})^2 + \\lambda \\|\\theta\\|^2$
    - Expected reward: $\\hat{r}_{t,a} = x_t^T \\theta_a$
    - UCB Score: $p_{t,a} = \\hat{r}_{t,a} + \\alpha \\sqrt{x_t^T A_a^{-1} x_t}$

    where $A_a = \\sum x_s x_s^T + I$ and $b_a = \\sum r_s x_s$.
    """

    def __init__(self, problem, linucb_alpha: float = LINUCB_ALPHA, **kwargs):
        self.local_best = None  # Track best internally to avoid event bus race conditions
        #: Exploration weight (default 2.0 to encourage exploration).  An
        #: explicit parameter: read from ``**kwargs`` it was also forwarded
        #: to StrategyBase, which rejects keys that are not config attributes.
        self.alpha = float(linucb_alpha)
        self._lock = threading.RLock()

        # LinUCB state per heuristic: A (d x d), b (d), theta (d)
        # Context dimension d = 3 (Bias, Progress, SuccessRate)
        self.context_dim = LINUCB_DIM

        StrategyBase.__init__(self, problem, **kwargs)

    def add_heuristic(self, h):
        StrategyBase.add_heuristic(self, h)
        init_linucb(h, self.context_dim)

    def _get_context_vector(self):
        """
        Constructs the context vector $x_t$ for the current state.

        Features:
        1. Bias (1.0)
        2. Budget Progress [0, 1]
        3. Recent Success Rate [0, 1]
        """
        if not hasattr(self, "_recent_rewards"):
            self._recent_rewards = []
        with self._lock:
            return linucb_context(len(self.results), self.config.max_eval, self._recent_rewards)

    def reward(self, improvement):
        """
        Calculate reward based on improvement magnitude.

        Args:
            improvement (float): improvement metric from constraint handler.

        Returns:
            float: Reward in [0, 1]
        """
        return linucb_reward(improvement)

    def on_new_best(self, best):
        """
        Logging only. Reward calculation is handled in on_new_results to avoid race conditions.
        """
        # This method is triggered by the Best analyzer via EventBus.
        # We don't use it for logic anymore, but we log the event.
        pass

    def on_new_results(self, results):
        """
        Process new results to update LinUCB models.
        """
        if not results:
            return

        if not hasattr(self, "_recent_rewards"):
            self._recent_rewards = []

        with self._lock:
            for result in results:
                # Reward against the internally tracked best (not the event-bus best)
                reward_val, self.local_best = linucb_observe(self.constraint_handler, self.local_best, result)
                if reward_val > 0:
                    self.logger.info("\u2318 %s | \u0394 %.7f %s (LinUCB)" % (result, reward_val, result.who))
                push_recent(self._recent_rewards, reward_val)

                # The context the point was selected under, attached in execute();
                # points generated without one (e.g. at startup) teach nothing.
                if not hasattr(result.point, "context_vector"):
                    continue
                try:
                    h = self.heuristic(result.who)
                except KeyError:
                    continue
                linucb_update(h, result.point.context_vector, reward_val)

    def execute(self):
        # One context for the whole batch; each point carries it back to on_new_results.
        context = self._get_context_vector()
        return collect_pulls(
            self, lambda _target: linucb_select(self.heuristics, context, self.alpha, self.context_dim)
        )

    def _get_status_info(self):
        """Return strategy-specific status info."""
        info = {}
        # Report max theta magnitude
        max_theta = 0
        best_h_name = ""
        for h in self.heuristics:
            if hasattr(h, "linucb_A_inv") and hasattr(h, "linucb_b"):
                theta = h.linucb_A_inv @ h.linucb_b
                norm_theta = np.linalg.norm(theta)
                if norm_theta > max_theta:
                    max_theta = norm_theta
                    best_h_name = h.name

        if best_h_name:
            info["max_theta"] = f"{best_h_name} ({max_theta:.2f})"
        return info
