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


class Random(Heuristic):

    """
    always generates random points inside the box of the
    "best leaf" (see "Splitter") until the capped queue is full.
    """

    def __init__(self, cap=None, name=None):
        name = "Random" if name is None else name
        self.leaf = None
        from threading import Event
        # used in on_start, to continue when we have a leaf.
        self.first_split = Event()
        Heuristic.__init__(self, name=name)

    def on_start(self):
        import numpy as np
        self.first_split.wait()
        splitter = self.strategy.analyzer("splitter")
        while True:
            r = self.leaf.ranges * np.random.rand(
                splitter.dim) + self.leaf.box[:, 0]
            self.emit(r)

    def on_new_split(self, box, children, dim):
        """
        we are only interested in the (possibly new)
        leaf around the best point
        """
        best = self.strategy.analyzer("best").best
        self.leaf = self.strategy.analyzer("splitter").get_leaf(best)
        self.clear_output()
        self.first_split.set()
