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
    """

    def __init__(self, problem, **kwargs):
        self.last_best = None
        self.total_selections = 0
        StrategyBase.__init__(self, problem, **kwargs)

    def add_heuristic(self, h):
        StrategyBase.add_heuristic(self, h)
        h.ucb_count = 0
        h.ucb_total_reward = 0.0

    def reward(self, best):
        """
        Calculate reward for a heuristic based on the improvement.

        Args:
        - ``best``: new (best) result
        """
        if self.last_best is None:
            # First point, can't really calculate improvement yet, or treat as infinite improvement
            return 1.0

        # Reward calculation similar to StrategyRewarding
        # R(x) = 1 - e^{-(f_{best} - f(x))}
        # However, 'best' here is the NEW best. self.last_best is the OLD best.
        # Improvement is (self.last_best.fx - best.fx).

        improvement = self.constraint_handler.calculate_improvement(self.last_best, best)
        reward = 1.0 - np.exp(-1.0 * improvement)

        # Update the heuristic that produced this point
        h = self.heuristic(best.who)
        if hasattr(h, "ucb_count"):
            # Update running mean of rewards
            # Q_{n+1} = Q_n + (R - Q_n) / (n + 1)
            # But here ucb_count tracks selections, not necessarily successes.
            # Usually UCB updates on selection. If we only update on success, we might have issues.
            # In standard bandit, you select an arm, observe reward.
            # Here, we select a heuristic to generate points. We might get points later.
            # The reward comes when a result is processed.

            # Let's accumulate reward to the heuristic.
            # Note: This implementation simplifies things by updating value on success.
            # But we should also account for selections that didn't yield a new best.
            # StrategyRewarding discounts on emission.

            # Let's try to stick to standard UCB as much as possible.
            # Q_t(a) = average reward obtained from arm a
            pass

        return reward

    def on_new_best(self, best):
        reward = self.reward(best)
        self.last_best = best

        # Update the heuristic's statistics
        h = self.heuristic(best.who)

        # We need to be careful: if we only update on new_best, heuristics that never find new bests
        # will have 0 reward. This is fine.
        # But we need to count selections properly.

        if hasattr(h, "ucb_total_reward"):
            # Update average reward.
            # We treat every point generated as a 'trial'.
            # If a heuristic generates N points and one of them is a new best with reward R,
            # effectively we have N trials, 1 success (reward R), N-1 failures (reward 0).
            # But we don't know exactly when the failures happened vs success.
            # A simple approximation: Add R to the total reward sum.
            # Q_t(a) = Total Reward / Number of Selections

            h.ucb_total_reward += reward
            # h.ucb_count is updated in execute() when we ask for points.

        self.logger.info("\u2318 %s | \u0394 %.7f %s (UCB)" % (best, reward, best.who))

    def on_new_result(self, result):
        # We might want to penalize or at least count evaluations that are not best?
        # In standard UCB, every pull gives a reward. Here reward is sparse (only when improving).
        # We can define reward = 0 for non-improving points.
        pass

    def execute(self):
        points = []
        target = self.jobs_per_client * len(self.evaluators)

        if len(self.evaluators.outstanding) < target:
            # UCB Parameter c (exploration weight)
            # A common value is sqrt(2), but can be tuned.
            c = float(self.config.ucb_c) if hasattr(self.config, "ucb_c") else 1.414

            # Ensure every heuristic is tried at least once
            heurs = self.heuristics

            # First pass: try any heuristic that hasn't been tried enough
            # Or just give them a boost.

            # In this framework, execute() is called repeatedly.
            # We need to decide which heuristic to ask for points.
            # We can ask for 1 point from the winner of UCB.

            while len(self.evaluators.outstanding) + len(points) < target:
                best_ucb = -float("inf")
                selected_h = None

                # Calculate UCB for each heuristic
                for h in heurs:
                    if not hasattr(h, "ucb_count"):
                        h.ucb_count = 0
                        h.ucb_total_reward = 0.0

                    if h.ucb_count == 0:
                        # Infinite UCB for untried arms
                        selected_h = h
                        break

                    average_reward = h.ucb_total_reward / h.ucb_count
                    exploration_term = c * np.sqrt(
                        np.log(self.total_selections) / h.ucb_count
                    )
                    ucb_score = average_reward + exploration_term

                    if ucb_score > best_ucb:
                        best_ucb = ucb_score
                        selected_h = h

                if selected_h:
                    # Request 1 point (or a small batch)
                    n_points = 1
                    new_points = selected_h.get_points(n_points)
                    points.extend(new_points)

                    # Update counts
                    selected_h.ucb_count += n_points
                    self.total_selections += n_points

                else:
                    # Should not happen if heuristics list is not empty
                    break

                # Break if we have enough points
                if len(points) >= target:  # simplified logic, target is rough
                    break

        return points
