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


class StrategyThompsonSampling(StrategyBase):
    """
    This strategy uses Thompson Sampling (Beta-Bernoulli bandit) to select heuristics.

    It maintains a Beta distribution Beta(alpha, beta) for each heuristic, representing
    the probability of success (finding a better point).

    Update rule:
    - Reward r in [0, 1] derived from improvement magnitude.
    - alpha <- alpha + r
    - beta <- beta + (1 - r)

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
        # Initialize Beta parameters (Uniform prior)
        h.ts_alpha = 1.0
        h.ts_beta = 1.0

    def reward(self, best):
        """
        Calculate reward for a heuristic based on the improvement.

        Args:
        - ``best``: new (best) result

        Returns:
            float: Reward value in [0, 1]
        """
        if self.last_best is None:
            # First point found is treated as a baseline success (max reward)
            return 1.0

        # Calculate improvement using constraint handler logic
        improvement = self.constraint_handler.calculate_improvement(self.last_best, best)

        # Bounded reward in [0, 1] based on improvement magnitude
        # improvement is >= 0
        reward = 1.0 - np.exp(-1.0 * improvement)

        return reward

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
            if hasattr(h, "ts_alpha"):
                # Bernoulli trial update with "soft" outcome r in [0,1]
                h.ts_alpha += reward
                h.ts_beta += (1.0 - reward)

                self.logger.info(
                    f"Updated {h.name}: reward={reward:.4f} -> Beta({h.ts_alpha:.2f}, {h.ts_beta:.2f})"
                )
            else:
                self.logger.warning(f"Heuristic {h.name} missing ts_alpha/beta")
        except KeyError:
            self.logger.warning(f"Heuristic '{best.who}' not found in strategy.")

        self.logger.info("\u2318 %s | \u0394 %.7f %s (Thompson)" % (best, reward, best.who))

    def execute(self):
        points = []
        target = self.jobs_per_client * len(self.evaluators)

        if len(self.evaluators.outstanding) < target:

            def until(points, target):
                return len(self.evaluators.outstanding) + len(points) >= target

            def selector():
                heurs = self.heuristics
                if not heurs:
                    return None

                # Sample from Beta for all heuristics
                samples = []
                for h in heurs:
                    # Ensure initialization
                    if not hasattr(h, "ts_alpha"):
                        h.ts_alpha = 1.0
                        h.ts_beta = 1.0

                    # Sample theta
                    theta = np.random.beta(h.ts_alpha, h.ts_beta)
                    samples.append((theta, h))

                # Sort by sampled value (highest first)
                samples.sort(key=lambda x: x[0], reverse=True)

                for theta, h in samples:
                    # Request points from the selected heuristic
                    new_points = h.get_points(1)
                    if new_points:
                        # Update selection stats
                        self.total_selections += len(new_points)
                        # Note: In pure Thompson Sampling, we select based on samples.
                        # We don't update parameters until we see a result (reward).
                        return new_points

                return []

            points = self._collect_points_safely(target, selector, until=until)

        return points
