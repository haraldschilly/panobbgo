from __future__ import division
from __future__ import unicode_literals
# -*- coding: utf8 -*-

from panobbgo.core import StrategyBase
import numpy as np


class StrategyUCB(StrategyBase):
    """
    This strategy uses the Upper Confidence Bound (UCB1) algorithm to select
    heuristics. It balances exploration (trying less used heuristics) and
    exploitation (using heuristics that have performed well).

    The Multi-Armed Bandit formulation here is:
    - Arms: Heuristics
    - Pull: Generating points from a heuristic
    - Reward: 0 if point doesn't improve best; >0 if it improves best.
      Reward calculation: R(x) = 1 - e^{-(improvement)}.
    - Value Q_t(a): Average reward per point generated.
    """

    def __init__(self, problem, **kwargs):
        self.last_best = None
        self.total_selections = 0
        StrategyBase.__init__(self, problem, **kwargs)

    def add_heuristic(self, h):
        StrategyBase.add_heuristic(self, h)
        # Initialize UCB statistics
        h.ucb_count = 0        # Number of points generated (Selections)
        h.ucb_total_reward = 0.0 # Accumulated reward

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
        reward = 1.0 - np.exp(-1.0 * improvement)

        return reward

    def on_new_best(self, best):
        """
        Called when a new best solution is found.
        Updates the reward statistics for the heuristic that generated the solution.
        """
        reward = self.reward(best)
        self.last_best = best

        # Update the heuristic's statistics
        try:
            h = self.heuristic(best.who)
            if hasattr(h, "ucb_total_reward"):
                # Accumulate reward.
                # Note: ucb_count is incremented in execute() when points are generated.
                # Points that don't become 'best' contribute 0 to total_reward
                # but increment ucb_count, thus lowering the average reward (Q value).
                h.ucb_total_reward += reward
                self.logger.info(f"Updated {h.name} reward: +{reward:.4f} -> {h.ucb_total_reward:.4f}")
            else:
                self.logger.warning(f"Heuristic {h.name} missing ucb_total_reward")
        except KeyError:
            self.logger.warning(f"Heuristic '{best.who}' not found in strategy.")

        self.logger.info("\u2318 %s | \u0394 %.7f %s (UCB)" % (best, reward, best.who))

    def execute(self):
        points = []
        target = self.jobs_per_client * len(self.evaluators)

        if len(self.evaluators.outstanding) < target:
            # UCB Parameter c (exploration weight)
            # Default to sqrt(2) approx 1.414
            c_val = getattr(self.config, "ucb_c", 1.414)
            try:
                c = float(c_val) # type: ignore
            except (ValueError, TypeError):
                c = 1.414

            def until(points, target):
                return len(self.evaluators.outstanding) + len(points) >= target

            def selector():
                heurs = self.heuristics
                if not heurs:
                    return None

                # Calculate UCB scores for all heuristics
                scores = []
                for h in heurs:
                    # Ensure initialization if added dynamically or somehow missed
                    if not hasattr(h, "ucb_count"):
                        h.ucb_count = 0
                        h.ucb_total_reward = 0.0

                    if h.ucb_count == 0:
                        # Infinite score for unselected arms to force exploration
                        score = float("inf")
                    else:
                        # Q_t(a) = Average Reward
                        average_reward = h.ucb_total_reward / h.ucb_count
                        # UCB1 exploration term
                        exploration_term = c * np.sqrt(
                            np.log(max(1, self.total_selections)) / h.ucb_count
                        )
                        score = average_reward + exploration_term
                    scores.append((score, h))

                # Try heuristics in order of score (highest first)
                scores.sort(key=lambda x: x[0], reverse=True)

                for score, h in scores:
                    # Request points from the selected heuristic
                    new_points = h.get_points(1)
                    if new_points:
                        # Update selection counts immediately
                        count = len(new_points)
                        h.ucb_count += count
                        self.total_selections += count
                        return new_points

                return []

            points = self._collect_points_safely(target, selector, until=until)

        return points
