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
from config import get_config
from core import Results, EventBus, StrategyBase
import numpy as np

class StrategyRewarding(StrategyBase):
  '''
  This strategy rewards given :mod:`.heuristics` by selecting
  those more ofte, which produce better search points.
  '''
  def __init__(self, problem, heurs):
    StrategyBase.__init__(self, problem, heurs)

  def execute(self):
    points = []
    target = self.per_client * len(self.evaluators)
    self.logger.debug("per_client = %s | target = %s" % (self.per_client, target))
    new_tasks = None
    if len(self.evaluators.outstanding) < target:
      s = self.config.smooth
      while True:
        heurs = self.heuristics
        perf_sum = sum(h.performance for h in heurs)
        for h in heurs:
          # calc probability based on performance with additive smoothing
          prob = (h.performance + s)/(perf_sum + s * len(heurs))
          nb_h = max(1, round(target * prob))
          points.extend(h.get_points(nb_h))
          #print "  %16s -> %s" % (h, nb_h)
        # stopping criteria
        if len(points) >= target: break
    return points

class StrategyRoundRobin(StrategyBase):
  r'''
  This is a very primitive strategy for testing purposes only.
  It selects the heuristics based on a fixed
  `round-robin <http://en.wikipedia.org/wiki/Round-robin_scheduling>`_
  scheme.
  '''
  def __init__(self, problem, heurs, size = 10):
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
