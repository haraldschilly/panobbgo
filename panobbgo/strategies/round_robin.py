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


class StrategyRoundRobin(StrategyBase):
    r"""
    This is a very primitive strategy for testing purposes only.
    It selects the heuristics based on a fixed
    `round-robin <http://en.wikipedia.org/wiki/Round-robin_scheduling>`_
    scheme.
    """

    def __init__(self, problem, size=10, **kwargs):
        self.size = size
        self.current = 0
        StrategyBase.__init__(self, problem, **kwargs)

    def execute(self):
        points = []
        attempts = 0
        max_attempts = 10  # Prevent infinite loop
        while len(points) == 0 and attempts < max_attempts:
            hs = self.heuristics
            if not hs:
                # Every arm went inactive (emitted everything it will ever
                # emit and unsubscribed).  There is nothing left to rotate
                # over, and ``% len(hs)`` used to raise ZeroDivisionError
                # here — which escaped ``start()`` and leaked the event bus,
                # the evaluator pool and every subprocess.  Returning ``[]``
                # lets ``StrategyBase._alive`` end the run cleanly.
                return []
            self.current = (self.current + 1) % len(hs)
            # ``produce``, not ``get_points``: an on-demand arm (a solver
            # bridge) has an empty queue between round trips and would be
            # skipped forever beside any arm that keeps one stocked.
            points.extend(hs[self.current].produce(self.size))
            attempts += 1
        return points
