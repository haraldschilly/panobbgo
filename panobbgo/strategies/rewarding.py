from __future__ import division
from __future__ import unicode_literals
# -*- coding: utf8 -*-
# Copyright 2012 Harald Schilly <harald.schilly@gmail.com>
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


class StrategyRewarding(StrategyBase):
    """
    This strategy rewards given :mod:`.heuristics` by selecting
    those more often, which produce better search points.
    """

    def __init__(self, problem, **kwargs):
        self.last_best = None
        StrategyBase.__init__(self, problem, **kwargs)

    def __start__(self):
        for h in self.heuristics:
            h.performance = 1.0

    def discount(self, heur, discount=None, times=1):
        """
        Discount the given heuristic after emitting a point.

        Args:

        - ``discount``: positive float, default ``config.default``
        - ``times``: how often
        """
        # Ensure discount is a float. If None, fetch from config which might return a string/float.
        val = discount if discount is not None else self.config.discount
        if val is None:
            # Fallback if config returns None, though typically it has a default
            d = 0.95
        else:
            try:
                d = float(val)
            except (ValueError, TypeError):
                # Handle cases where val cannot be converted to float
                d = 0.95

        d = d**times
        heur.performance *= d

    def reward(self, best):
        """
        Give this heuristic a reward (e.g. when it finds a new point)

        Args:

        - ``best``: new (best) result
        """
        if self.last_best is None:
            return 1.0
        import numpy as np

        # currently, only reward if better point found.
        # TODO in the future also reward if near the best value (but
        # e.g. not in the proximity of the best x)
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
        reward = self.reward(best)
        self.logger.info("\u2318 %s | \u0394 %.7f %s" % (best, reward, best.who))
        self.last_best = best

    def execute(self):
        points = []
        target = self.jobs_per_client * len(self.evaluators)
        # self.logger.debug(
        #     "per_client = %s | target = %s" % (self.jobs_per_client, target)
        # )
        if len(self.evaluators.outstanding) < target:
            try:
                s = float(self.config.smooth)
            except (ValueError, TypeError):
                s = 0.5

            def selector():
                heurs = self.heuristics
                if not heurs:
                    return None

                batch = []
                perf_sum = sum(h.performance for h in heurs)
                for h in heurs:
                    # calc probability based on performance with additive
                    # smoothing
                    prob = (h.performance + s) / (perf_sum + s * len(heurs))
                    nb_h = max(1, round(target * prob))
                    h_pts = h.get_points(nb_h)
                    batch.extend(h_pts)
                    # print "  %16s -> %s" % (h, nb_h)
                return batch

            points = self._collect_points_safely(target, selector)

        return points
        return points
