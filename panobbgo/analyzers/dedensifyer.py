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

from panobbgo.config import get_config
from panobbgo.core import Analyzer

class Box(object):
    r'''
    :class:`Dedensifyer's <.Dedensifyer>` helper class, that registeres points for the given box.
    '''
    def __init__(self):
        self.count = 0
        self.max_fx = None
        self.min_fx = None
        self.max_cv = None
        self.min_cv = None

    def register(self, result):
        count += 1

        if self.max_fx is None or self.max_fx.fx < result.fx:
            self.max_fx = result

        if self.max_cv is None or self.max_cv.cv < result.cv:
            self.max_cv = result

        if self.min_fx is None or self.min_fx.fx > result.fx:
            self.min_fx = result

        if self.min_cv is None or self.min_cv.cv > result.cv:
            self.min_cv = result

class Dedensifyer(Analyzer):
    r'''
    This analyzer stores points in a fixed hierarchical grid.
    It discards previously added points in order to avoid a high number of points 
    in a close neighbourhood. The rules for discarding older points take
    the function value and the constraint violation into account to store
    the minimal and maximal representants for that region.
    '''

    def __init__(self, max_depth = 5):
        Analyzer.__init__(self)
        self.max_depth = max_depth
        self.boxes = dict()
        for depth in range(self.max_depth):
            self.boxes[depth] = dict()
        self._

    def __start__(self):
        for depth in range(self.max_depth):
          self.box_dims[depth] = self.problem.ranges / float(depth)

    def gridkey(self, x, depth):
        from numpy import floor
        return tuple(floor(x / self.box_dims[depth]) * self.box_dims[depth])

    def get_box(self, x, depth):
        key = self.gridkey(x)
        return self._boxes.get(key, None)

    def register(result):
        '''
        analyzes a new result
        '''
        key = self.gridkey(result.x)
        if key not in self._boxes:
            self._boxes[key] = Box()
        self._boxes[key].register(result)

    def on_new_results(self, results):
        map(self.register, results)
