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

from panobbgo.core import StrategyBase
from panobbgo.strategies._bandit import collect_pulls, improvement_reward, init_thompson, thompson_select


class StrategyThompsonSampling(StrategyBase):
    """
    This strategy uses Thompson Sampling (Beta-Bernoulli bandit) to select heuristics.

    It maintains a Beta distribution Beta(alpha, beta) for each heuristic, representing
    the probability of success (finding a better point).

    Update rule:
    - Reward r in [0, 1] derived from improvement magnitude.
    - alpha = 1 + total_reward (successes)
    - beta = 1 + total_attempts - total_reward (failures)

    Selection:
    - Sample theta ~ Beta(alpha, beta) for each heuristic.
    - Select heuristic with highest theta.
    """

    def __init__(self, problem, **kwargs):
        self.last_best = None
        # Keep track of total selections if needed for stats, though not used in algorithm
        self.total_selections = 0
        StrategyBase.__init__(self, problem, **kwargs)

    def add_heuristic(self, h):
        StrategyBase.add_heuristic(self, h)
        init_thompson(h)

    def reward(self, best):
        """
        Calculate reward for a heuristic based on the improvement.

        Args:
        - ``best``: new (best) result

        Returns:
            float: Reward value in [0, 1]
        """
        return improvement_reward(self.constraint_handler, self.last_best, best)

    def on_new_best(self, best):
        """
        Called when a new best solution is found.
        Updates the Beta parameters for the heuristic that generated the solution.
        """
        reward = self.reward(best)
        self.last_best = best

        # Update the heuristic's statistics
        try:
            h = self.heuristic(best.who)
            if hasattr(h, "ts_total_reward"):
                # Accumulate reward (successes)
                h.ts_total_reward += reward

                # Calculate current alpha/beta for logging
                alpha = 1.0 + h.ts_total_reward
                beta = 1.0 + max(0.0, h.ts_counts - h.ts_total_reward)

                h.ts_alpha = alpha
                h.ts_beta = beta

                self.logger.info(f"Updated {h.name}: reward={reward:.4f} -> Beta({alpha:.2f}, {beta:.2f})")
            else:
                self.logger.warning(f"Heuristic {h.name} missing ts_total_reward")
        except KeyError:
            self.logger.warning(f"Heuristic '{best.who}' not found in strategy.")

        self.logger.info("\u2318 %s | \u0394 %.7f %s (Thompson)" % (best, reward, best.who))

    def _get_status_info(self):
        """Return strategy-specific status info."""
        info = {}
        # Report max alpha (best heuristic confidence)
        max_alpha = 0
        best_h_name = ""
        for h in self.heuristics:
            if hasattr(h, "ts_alpha") and h.ts_alpha > max_alpha:
                max_alpha = h.ts_alpha
                best_h_name = h.name

        if best_h_name:
            info["best_heuristic"] = f"{best_h_name} (\u03b1={max_alpha:.1f})"
        return info

    def execute(self):
        return collect_pulls(self, lambda _target: thompson_select(self, self.heuristics, self.rng))
