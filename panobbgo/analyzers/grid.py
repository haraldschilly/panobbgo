from __future__ import division
from __future__ import unicode_literals
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

from panobbgo.core import Analyzer


class Grid(Analyzer):
    """
    packs nearby points into grid boxes
    """

    def __init__(self, strategy):
        Analyzer.__init__(self, strategy)

    def __start__(self):
        # grid for storing points which are nearby.
        # maps from rounded coordinates tuple to point
        self._grid = dict()
        self._grid_div = 5.0
        self._grid_lengths = self.problem.ranges / float(self._grid_div)

    def in_same_grid(self, point):
        key = tuple(self._grid_mapping(point.x))
        return self._grid.get(key, [])

    def _grid_mapping(self, x):
        from numpy import floor

        l = self._grid_lengths
        # m = self._problem.box[:,0]
        return tuple(floor(x / l) * l)

    def _grid_add(self, r):
        key = self._grid_mapping(r.x)
        box = self._grid.get(key, [])
        box.append(r)
        self._grid[key] = box
        # print ' '.join('%2s' % str(len(self._grid[k])) for k in
        # sorted(self._grid.keys()))

    def on_new_results(self, results):
        for result in results:
            self._grid_add(result)
