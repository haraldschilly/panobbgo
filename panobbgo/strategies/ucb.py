from __future__ import division
from __future__ import unicode_literals
# -*- coding: utf8 -*-

from panobbgo.core import StrategyBase
from panobbgo.strategies._bandit import collect_pulls, improvement_reward, init_ucb, ucb_select


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

    def __init__(self, problem, ucb_c: float = 1.414, **kwargs):
        #: Exploration weight ``c`` (default ≈ √2).  Was read from a config
        #: key that does not exist, so ``StrategyUCB(ucb_c=...)`` was ignored.
        self.ucb_c = float(ucb_c)
        self.last_best = None
        self.total_selections = 0
        StrategyBase.__init__(self, problem, **kwargs)

    def add_heuristic(self, h):
        StrategyBase.add_heuristic(self, h)
        init_ucb(h)

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
        return collect_pulls(self, lambda _target: ucb_select(self, self.heuristics, self.ucb_c))
