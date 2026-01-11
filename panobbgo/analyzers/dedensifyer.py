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

from panobbgo.core import Analyzer


class Box:
    r"""
    :class:`Dedensifyer's <.Dedensifyer>` helper class, that registeres points for the given box.
    """

    def __init__(self):
        self.count = 0
        self.max_fx = None
        self.min_fx = None
        self.max_cv = None
        self.min_cv = None

    def register(self, result):
        self.count += 1

        if self.max_fx is None or self.max_fx.fx < result.fx:
            self.max_fx = result

        if self.max_cv is None or self.max_cv.cv < result.cv:
            self.max_cv = result

        if self.min_fx is None or self.min_fx.fx > result.fx:
            self.min_fx = result

        if self.min_cv is None or self.min_cv.cv > result.cv:
            self.min_cv = result


class Dedensifyer(Analyzer):
    r"""
    This analyzer stores points in a fixed hierarchical grid.
    It discards previously added points in order to avoid a high number of points
    in a close neighbourhood. The rules for discarding older points take
    the function value and the constraint violation into account to store
    the minimal and maximal representants for that region.
    """

    def __init__(self, strategy, max_depth=5):
        Analyzer.__init__(self, strategy)
        self.max_depth = max_depth
        # boxes[depth] is a dict mapping gridkey -> Box
        self.boxes = [dict() for _ in range(self.max_depth)]
        self.box_dims = {}

    def __start__(self):
        # Initialize grid dimensions for each depth level.
        # We use a hierarchical decomposition where each depth k divides the space
        # into 2^k segments per dimension (or cell size scales by 1/2^k).
        for depth in range(self.max_depth):
             self.box_dims[depth] = self.problem.ranges / (2.0 ** depth)

    def gridkey(self, x, depth):
        # Calculate grid coordinates for point x at given depth.
        # x is a point coordinate (numpy array)
        # box_dims[depth] is the size of a grid cell at this depth

        # box[:, 0] is the vector of lower bounds.
        box_min = self.problem.box[:, 0]

        idx = np.floor((x - box_min) / self.box_dims[depth])
        return tuple(idx.astype(int))

    def get_box(self, x, depth):
        key = self.gridkey(x, depth)
        return self.boxes[depth].get(key, None)

    def register(self, result):
        """
        analyzes a new result
        """
        for depth in range(self.max_depth):
            key = self.gridkey(result.x, depth)
            if key not in self.boxes[depth]:
                self.boxes[depth][key] = Box()
            self.boxes[depth][key].register(result)

    def on_new_results(self, results):
        for r in results:
            self.register(r)
