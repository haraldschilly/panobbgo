# -*- coding: utf8 -*-
from __future__ import division
from __future__ import unicode_literals

# Copyright 2025 Panobbgo Contributors
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

from panobbgo.core import StrategyBase
import numpy as np
import threading

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

    def __init__(self, problem, **kwargs):
        self.local_best = None  # Track best internally to avoid event bus race conditions
        # Default alpha increased to 2.0 to encourage exploration
        self.alpha = kwargs.get('linucb_alpha', 2.0)
        self._lock = threading.RLock()

        # LinUCB state per heuristic: A (d x d), b (d), theta (d)
        # Context dimension d = 3 (Bias, Progress, SuccessRate)
        self.context_dim = 3

        StrategyBase.__init__(self, problem, **kwargs)

    def add_heuristic(self, h):
        StrategyBase.add_heuristic(self, h)
        # Initialize LinUCB matrices for this heuristic
        # A = Identity(d)
        h.linucb_A = np.eye(self.context_dim)
        # b = Zeros(d)
        h.linucb_b = np.zeros(self.context_dim)
        # Cached inverse of A for speed (optional, but A is small 3x3 so inversion is fast)
        h.linucb_A_inv = np.eye(self.context_dim)

    def _get_context_vector(self):
        """
        Constructs the context vector $x_t$ for the current state.

        Features:
        1. Bias (1.0)
        2. Budget Progress [0, 1]
        3. Recent Success Rate [0, 1]
        """
        # 1. Bias
        feat_bias = 1.0

        # 2. Budget Progress
        n_evals = len(self.results)
        max_evals = float(self.config.max_eval) if self.config.max_eval else 1000.0
        feat_progress = min(1.0, n_evals / max_evals)

        # 3. Recent Success Rate
        if not hasattr(self, '_recent_rewards'):
            self._recent_rewards = []

        if self._recent_rewards:
            # Average of recent binary success (reward > 0)
            feat_success = sum(1 for r in self._recent_rewards if r > 0) / len(self._recent_rewards)
        else:
            feat_success = 0.0

        return np.array([feat_bias, feat_progress, feat_success])

    def reward(self, improvement):
        """
        Calculate reward based on improvement magnitude.

        Args:
            improvement (float): improvement metric from constraint handler.

        Returns:
            float: Reward in [0, 1]
        """
        if improvement <= 0:
            return 0.0
        # Bounded reward in [0, 1]
        return 1.0 - np.exp(-1.0 * improvement)

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

        # Update recent rewards buffer
        if not hasattr(self, '_recent_rewards'):
            self._recent_rewards = []

        with self._lock:
            for result in results:
                # Calculate reward locally using self.local_best
                if self.local_best is None:
                    # First point is effectively a success
                    improvement = 1.0
                    self.local_best = result
                else:
                    improvement = self.constraint_handler.calculate_improvement(self.local_best, result)
                    if self.constraint_handler.is_better(self.local_best, result):
                        self.local_best = result

                reward_val = self.reward(improvement)

                if reward_val > 0:
                    self.logger.info("\u2318 %s | \u0394 %.7f %s (LinUCB)" % (result, reward_val, result.who))

                # Update recent rewards (keep last 100)
                self._recent_rewards.append(reward_val)
                if len(self._recent_rewards) > 100:
                    self._recent_rewards.pop(0)

                # Get context vector from point
                # It should have been attached in execute()
                if hasattr(result.point, 'context_vector'):
                    x_t = result.point.context_vector

                    try:
                        h = self.heuristic(result.who)
                    except KeyError:
                        continue

                    # LinUCB Update
                    # A += x x^T
                    h.linucb_A += np.outer(x_t, x_t)
                    # b += r * x
                    h.linucb_b += reward_val * x_t

                    # Recompute inverse (or use Sherman-Morrison for O(d^2) update, but d=3 is tiny)
                    h.linucb_A_inv = np.linalg.inv(h.linucb_A)

                    # Update stats for logging
                    if not hasattr(h, 'linucb_count'): h.linucb_count = 0
                    if not hasattr(h, 'linucb_reward'): h.linucb_reward = 0.0
                    h.linucb_count += 1
                    h.linucb_reward += reward_val

                else:
                    # Point generated without context (e.g. at startup or by other means)
                    pass

    def execute(self):
        points = []
        target = self.jobs_per_client * len(self.evaluators)

        if len(self.evaluators.outstanding) < target:

            # Calculate context once for this batch
            current_context = self._get_context_vector()

            def until(points, target):
                return len(self.evaluators.outstanding) + len(points) >= target

            def selector():
                heurs = self.heuristics
                if not heurs:
                    return None

                # Calculate LinUCB scores
                scores = []
                for h in heurs:
                    # Ensure initialization
                    if not hasattr(h, "linucb_A"):
                        h.linucb_A = np.eye(self.context_dim)
                        h.linucb_b = np.zeros(self.context_dim)
                        h.linucb_A_inv = np.eye(self.context_dim)

                    # Theta = A^-1 b
                    theta = h.linucb_A_inv @ h.linucb_b

                    # Predicted reward: x^T theta
                    mean = current_context @ theta

                    # Confidence interval: alpha * sqrt(x^T A^-1 x)
                    variance = current_context @ h.linucb_A_inv @ current_context
                    std = np.sqrt(variance)

                    ucb_score = mean + self.alpha * std

                    scores.append((ucb_score, h))

                # Sort by score (descending)
                scores.sort(key=lambda x: x[0], reverse=True)

                for score, h in scores:
                    # Request points from the selected heuristic
                    new_points = h.get_points(1)
                    if new_points:
                        # Attach context to points!
                        for p in new_points:
                            p.context_vector = current_context
                        return new_points

                return []

            points = self._collect_points_safely(target, selector, until=until)

        return points

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
