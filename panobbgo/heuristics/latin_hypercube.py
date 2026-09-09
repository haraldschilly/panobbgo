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

import numpy as np

from panobbgo.core import Heuristic


class LatinHypercube(Heuristic):
    """
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
    """

    def __init__(self, strategy, div):
        """
        Args:
           - `div`: number of divisions, positive integer.
        """
        cap = div
        Heuristic.__init__(self, strategy, cap=cap, name="Latin Hypercube")
        if not isinstance(div, int):
            raise Exception("LH: div needs to be an integer")
        self.div = div

    def __start__(self):
        # length of each box'es dimension
        self.lengths = self.problem.ranges / float(self.div)

    def _design(self):
        """One Latin-hypercube design of ``div`` points (a list of arrays)."""
        div = self.div
        dim = self.problem.dim
        pts = np.repeat(np.arange(div, dtype=np.float64), dim).reshape(div, dim)
        pts += self.rng.random((div, dim))  # add [0,1) jitter
        pts *= self.lengths  # scale with length, already divided by div
        pts += self.problem.box[:, 0]  # shift with min
        for _ in range(dim):
            self.rng.shuffle(pts[:, _])
        return [p for p in pts]  # needs to be a list of np.ndarrays

    def _fill(self):
        """Emit whole designs until the queue holds at least ``div`` points.

        The unit of this heuristic is a design, not a point, so it refills in
        design-sized blocks rather than topping up to ``cap`` point by point.
        """
        while self._output.qsize() < self.div:
            self.emit(self._design())

    def on_start(self):
        self._fill()

    def on_new_results(self, results):
        self._fill()
