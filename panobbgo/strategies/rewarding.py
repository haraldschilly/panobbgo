# -*- coding: utf8 -*-
from __future__ import division
from __future__ import unicode_literals

# Copyright 2012-2026 Harald Schilly <harald.schilly@gmail.com>
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

import numpy as np

from panobbgo.core import StrategyBase
from panobbgo.strategies._bandit import (
    collect_pulls,
    discount_factor,
    ema_credit,
    ema_discount,
    ema_select,
    init_rewarding,
    near_best_rewards,
    rewarding_select,
)


#: Credit-assignment modes of :class:`StrategyRewarding`.
REWARDING_CREDITS = ("legacy", "ema")


def rewarding_params(config, credit=None, explore=None):
    """``(credit, explore)`` of a Rewarding policy: the arguments, else the config, else ``"ema"`` / 0.2.

    Shared by :class:`StrategyRewarding` and a Rewarding phase of
    :class:`~.phased.StrategyPhased`, so both resolve the same defaults.
    Raises ``ValueError`` for an unknown *credit*.
    """
    credit = str(credit if credit is not None else getattr(config, "rewarding_credit", "ema"))
    if credit not in REWARDING_CREDITS:
        raise ValueError("credit must be 'legacy' or 'ema', got %r" % credit)
    explore = float(explore if explore is not None else getattr(config, "rewarding_explore", 0.2))
    return credit, explore


class StrategyRewarding(StrategyBase):
    """
    This strategy rewards given :mod:`.heuristics` by selecting
    those more often, which produce better search points.

    Two credit-assignment modes (``credit=`` kwarg or ``config.rewarding_credit``):

    ``"ema"`` (default)
        ``performance`` is an exponential moving average of the reward
        *per evaluation spent* by that heuristic (reward ``1 - exp(-improvement)``
        when the evaluation improves the best, else 0; smoothing
        ``1 - config.discount``).  Selection is probability matching over the
        heuristics that currently have points, with an exploration floor of
        ``explore`` (``config.rewarding_explore``, default 0.2) spread evenly.

    ``"legacy"``
        Every point drawn from a heuristic multiplies its ``performance`` by
        ``config.discount`` (0.95); a new best adds ``1 - exp(-improvement)``
        to the emitter's performance.  Selection is proportional to
        ``performance + config.smooth``.  This credits the *emitter of the
        best point* rather than improvement per evaluation spent, so a
        heuristic that emits a generation of 90 points is discounted by
        ``0.95**90`` at once while a one-shot heuristic keeps its weight
        for the whole run — the strategy degenerates to a biased
        round-robin (planning/DISCOVERY_2026-09-09.md §6).

    Measured on the standard IOH battery (mean AOCC, 12-seed decision
    roster, paired per seed, same arm set): ``"ema"`` with ``explore=0.2``
    scores **+0.0135 [+0.0032, +0.0238]** over ``"legacy"`` (9/12 seeds),
    the gain sitting at 2-D (**+0.0248 [+0.0049, +0.0447]**) and flat at
    5-D (+0.0022 [-0.0030, +0.0074]).
    """

    def __init__(self, problem, credit=None, explore=None, **kwargs):
        self.last_best = None
        StrategyBase.__init__(self, problem, **kwargs)
        self.credit, self.explore = rewarding_params(self.config, credit, explore)
        self._ema_alpha = 1.0 - self._discount_factor()

    def _discount_factor(self):
        return ema_discount(self.config.discount)

    def add_heuristic(self, h):
        StrategyBase.add_heuristic(self, h)
        init_rewarding(h)

    def discount(self, heur, discount=None, times=1):
        """
        Discount the given heuristic after emitting a point.

        Args:

        - ``discount``: positive float, default ``config.default``
        - ``times``: how often
        """
        val = discount if discount is not None else self.config.discount
        heur.performance *= discount_factor(val) ** times

    def reward(self, best):
        """
        Give this heuristic a reward (e.g. when it finds a new point)

        Args:

        - ``best``: new (best) result
        """
        if self.last_best is None:
            return 1.0

        # currently, only reward if better point found.
        # (near best values are rewarded in on_new_results)
        fx_delta, reward = 0.0, 0.0
        # fx_delta = np.log1p(self.best.fx - r.fx) # log1p ok?

        improvement = self.constraint_handler.calculate_improvement(self.last_best, best)
        fx_delta = 1.0 - np.exp(-1.0 * improvement)  # saturates to 1
        fx_delta = 0.0 if fx_delta <= 0 else fx_delta
        # if self.fx_delta_last == None: self.fx_delta_last = fx_delta
        reward = fx_delta  # / self.fx_delta_last
        self.heuristic(best.who).performance += reward
        # self.fx_delta_last = fx_delta
        return reward

    def on_new_best(self, best):
        if self.credit == "ema":
            self.logger.info("\u2318 %s by %s" % (best, best.who))
            return
        reward = self.reward(best)
        self.logger.info("\u2318 %s | \u0394 %.7f %s" % (best, reward, best.who))
        self.last_best = best

    def _get_status_info(self):
        """Return strategy-specific status info."""
        info = {}
        # Report best performance
        max_perf = 0
        best_h_name = ""
        for h in self.heuristics:
            if hasattr(h, "performance") and h.performance > max_perf:
                max_perf = h.performance
                best_h_name = h.name

        if best_h_name:
            info["best_heuristic"] = f"{best_h_name} (perf={max_perf:.2f})"
        return info

    def on_new_results(self, results):
        if self.credit == "ema":
            self._credit_ema(results)
            return
        if self.last_best is None:
            return

        for r, reward in near_best_rewards(self.constraint_handler, self.problem, self.last_best, results):
            self.heuristic(r.who).performance += reward

    def _credit_ema(self, results):
        """Update each emitter's EMA reward-per-evaluation with this batch."""
        self.last_best = ema_credit(self.constraint_handler, self.heuristic, self.last_best, results, self._ema_alpha)

    def execute(self):
        try:
            s = float(self.config.smooth)
        except (ValueError, TypeError):
            s = 0.5

        def selector(target):
            if self.credit == "ema":
                heurs = self.heuristics
                return self._select_ema(heurs, target) if heurs else None
            return rewarding_select(self.heuristics, target, s, self.config.discount, rng=self._exact_split_rng())

        return collect_pulls(self, selector, count_outstanding=False)

    def _select_ema(self, heurs, target):
        """Probability matching with an exploration floor over heuristics that have points."""
        return ema_select(heurs, target, self.explore, rng=self._exact_split_rng())
