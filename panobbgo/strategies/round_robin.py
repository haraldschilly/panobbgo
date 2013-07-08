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

from panobbgo.core import StrategyBase

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
