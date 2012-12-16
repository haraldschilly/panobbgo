# -*- coding: utf8 -*-
# Copyright 2012 Harald Schilly <harald.schilly@univie.ac.at>
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

r'''
Strategies
==========

This part outlines the coordination between the point-producing
heuristics, the interaction with the cluster and the
:class:`DB of evaluated points <panobbgo.core.Results>`.

Basically, one or more threads produce points where to search,
and another one consumes them and dispatches tasks.
Subclass the :class:`~panobbgo.core.StrategyBase` class to implement
a new strategy.

.. inheritance-diagram:: panobbgo.strategies

.. codeauthor:: Harald Schilly <harald.schilly@univie.ac.at>
'''
from core import StrategyBase


class StrategyRewarding(StrategyBase):
    '''
    This strategy rewards given :mod:`.heuristics` by selecting
    those more ofte, which produce better search points.
    '''
    def __init__(self, problem, heurs):
        self.last_best = None
        StrategyBase.__init__(self, problem, heurs)

    def _init_(self):
        for h in self.heuristics:
            h.performance = 1.0

    def discount(self, heur, discount=None, times=1):
        '''
        Discount the given heuristic after emitting a point.

        Args:

        - ``discount``: positive float, default ``config.default``
        - ``times``: how often
        '''
        d = discount if discount is not None else self.config.discount
        d = d ** times
        heur.performance *= d

    def reward(self, best):
        '''
        Give this heuristic a reward (e.g. when it finds a new point)

        Args:

        - ``best``: new (best) result
        '''
        if self.last_best is None:
            return 1.0
        import numpy as np
        # currently, only reward if better point found.
        # TODO in the future also reward if near the best value (but
        # e.g. not in the proximity of the best x)
        fx_delta, reward = 0.0, 0.0
        # fx_delta = np.log1p(self.best.fx - r.fx) # log1p ok?
        fx_delta = 1.0 - np.exp(-1.0 * (self.last_best.fx -
                                best.fx))  # saturates to 1
        fx_delta = 0.0 if fx_delta <= 0 else fx_delta
        # if self.fx_delta_last == None: self.fx_delta_last = fx_delta
        reward = fx_delta  # / self.fx_delta_last
        self.heuristic(best.who).performance += reward
        # self.fx_delta_last = fx_delta
        return reward

    def on_new_best(self, best):
        reward = self.reward(best)
        self.logger.info(
            u"\u2318 %s | \u0394 %.7f %s" % (best, reward, best.who))
        self.last_best = best

    def execute(self):
        points = []
        target = self.jobs_per_client * len(self.evaluators)
        self.logger.debug(
            "per_client = %s | target = %s" % (self.jobs_per_client, target))
        if len(self.evaluators.outstanding) < target:
            s = self.config.smooth
            while True:
                heurs = self.heuristics
                perf_sum = sum(h.performance for h in heurs)
                for h in heurs:
                    # calc probability based on performance with additive
                    # smoothing
                    prob = (h.performance + s) / (perf_sum + s * len(heurs))
                    nb_h = max(1, round(target * prob))
                    h_pts = h.get_points(nb_h)
                    points.extend(h_pts)
                    # print "  %16s -> %s" % (h, nb_h)
                # stopping criteria
                if len(points) >= target:
                    break
        return points


class StrategyRoundRobin(StrategyBase):
    r'''
    This is a very primitive strategy for testing purposes only.
    It selects the heuristics based on a fixed
    `round-robin <http://en.wikipedia.org/wiki/Round-robin_scheduling>`_
    scheme.
    '''
    def __init__(self, problem, heurs, size=10):
        self.size = size
        self.current = 0
        StrategyBase.__init__(self, problem, heurs)

    def execute(self):
        from IPython.utils.timing import time
        points = []
        while len(points) == 0:
            hs = self.heuristics
            self.current = (self.current + 1) % len(hs)
            points.extend(hs[self.current].get_points(self.size))
            time.sleep(1e-3)
        return points
