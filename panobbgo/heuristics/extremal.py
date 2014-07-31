from __future__ import division
from __future__ import unicode_literals
from future.builtins import range
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

from panobbgo.core import Heuristic


class Extremal(Heuristic):

    """
    This heuristic is specifically seeking for points at the
    border of the box and around 0.
    The @where parameter takes a list or tuple, which has values
    from 0 to 1, which indicate the probability for sampling from the
    minimum, zero, center and the maximum. default = ( 1, .2, .2, 1 )
    """

    def __init__(self, strategy, diameter=1. / 10, prob=None):
        Heuristic.__init__(self, strategy, name="Extremal")
        import numpy as np
        if prob is None:
            prob = (1, .2, .2, 1)
        prob = np.array(prob) / float(np.sum(prob))
        self.probabilities = prob.cumsum()
        self.diameter = diameter  # inside the box or around zero
        self.vals = None

    def __start__(self):
        import numpy as np
        problem = self.problem
        low = problem.box[:, 0]
        high = problem.box[:, 1]
        zero = np.zeros(problem.dim)
        center = low + (high - low) / 2.
        self.vals = np.row_stack((low, zero, center, high))

    def on_start(self):
        import numpy as np
        while True:
            ret = np.empty(self.problem.dim)
            for i in range(self.problem.dim):
                r = np.random.rand()
                for idx, val in enumerate(self.probabilities):
                    if val > r:
                        radius = self.problem.ranges[i] * self.diameter
                        # jitter = radius * (np.random.rand() - .5)
                        jitter = np.random.normal(0, radius)
                        if idx == 0:
                            # minimum border
                            ret[i] = self.vals[idx, i] + abs(jitter)
                        elif idx == len(self.probabilities) - 1:
                            # maximum border
                            ret[i] = self.vals[idx, i] - abs(jitter)
                        else:
                            # around center or zero
                            ret[i] = self.vals[idx, i] + jitter
                        break  # since we found the idx, break!
            self.emit(ret)
            # stop early, if run by unittests
            if self.strategy.config.testing_mode:
                return
