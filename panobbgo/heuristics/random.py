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

from panobbgo.core import Heuristic


class Random(Heuristic):
    """
    always generates random points inside the box of the
    "best leaf" (see "Splitter") until the capped queue is full.
    """

    def __init__(self, strategy, cap=None, name=None):
        name = "Random" if name is None else name
        self.leaf = None
        from threading import Event

        # used in on_start, to continue when we have a leaf.
        self.first_split = Event()
        Heuristic.__init__(self, strategy, name=name)

    def on_start(self):
        import numpy as np

        # Initialize by trying to get the root leaf from splitter
        # This handles the initial case where no splits have happened yet
        # but the root box exists
        try:
            splitter = self.strategy.analyzer("Splitter")
            # If no specific leaf is set, start with the root leaf
            if self.leaf is None and hasattr(splitter, 'root'):
                 self.leaf = splitter.root
                 self.first_split.set()
        except Exception:
             pass

        # Wait for splitter to initialize, but with timeout to avoid hanging
        split_available = self.first_split.wait(timeout=5.0)  # 5 second timeout

        if not split_available:
            # No splits available yet, generate from full problem space
            self.logger.warning("No search leaf available, generating from full problem space")
            while True:
                r = self.problem.random_point()
                self.emit(r)
            return

        splitter = self.strategy.analyzer("Splitter")
        if self.leaf is None:
            # Fallback: generate from full problem space
            self.logger.warning("No search leaf available, generating from full problem space")
            while True:
                r = self.problem.random_point()
                self.emit(r)
            return

        # Generate from the current best leaf
        while True:
            r = self.leaf.ranges * np.random.rand(splitter.dim) + self.leaf.box[:, 0]
            self.emit(r)

    def on_new_split(self, box, children, dim):
        """
        we are only interested in the (possibly new)
        leaf around the best point
        """
        best = self.strategy.analyzer("Best").best
        self.leaf = self.strategy.analyzer("Splitter").get_leaf(best)
        self.clear_output()
        self.first_split.set()
