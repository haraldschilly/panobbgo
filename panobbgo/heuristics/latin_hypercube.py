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

from panobbgo.core import Heuristic, StopHeuristic


class LatinHypercube(Heuristic):

    '''
    Partitions the search box into n x n x ... x n cubes.
    Selects randomly in such a way, that there is only one cube in each dimension.
    Then, it randomly selects one point from inside such a cube.

    e.g. with div=4 and dim=2::

      +---+---+---+---+
      | X |   |   |   |
      +---+---+---+---+
      |   |   |   | X |
      +---+---+---+---+
      |   | X |   |   |
      +---+---+---+---+
      |   |   | X |   |
      +---+---+---+---+
    '''

    def __init__(self, div):
        '''
        Args:
           - `div`: number of divisions, positive integer.
        '''
        cap = div
        Heuristic.__init__(self, cap=cap, name="Latin Hypercube")
        if not isinstance(div, int):
            raise Exception("LH: div needs to be an integer")
        self.div = div

    def _init_(self):
        # length of each box'es dimension
        self.lengths = self.problem.ranges / float(self.div)

    def on_start(self):
        import numpy as np
        div = self.div
        dim = self.problem.dim
        while True:
            pts = np.repeat(
                np.arange(div, dtype=np.float64), dim).reshape(div, dim)
            pts += np.random.rand(div, dim)  # add [0,1) jitter
            pts *= self.lengths             # scale with length, already divided by div
            pts += self.problem.box[:, 0]    # shift with min
            [np.random.shuffle(pts[:, i]) for i in range(dim)]
            self.emit([p for p in pts])  # needs to be a list of np.ndarrays
