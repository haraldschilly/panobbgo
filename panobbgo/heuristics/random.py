# -*- coding: utf8 -*-
# Copyright 2012 - 2026 Harald Schilly <harald.schilly@univie.ac.at>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from panobbgo.core import Heuristic


class Random(Heuristic):
    """
    Generates uniformly random points inside the box of the current
    "best leaf" (see :class:`~panobbgo.analyzers.Splitter`), or inside the
    whole problem box until the Splitter has published one.

    The heuristic is *reactive*: it fills its output queue on ``start`` and
    tops it up whenever results arrive or the best leaf changes.  It never
    sleeps or polls, so the sequence of points it produces is a pure
    function of its :attr:`rng` and the events it receives.
    """

    def __init__(self, strategy, cap=None, name=None):
        name = "Random" if name is None else name
        self.leaf = None
        Heuristic.__init__(self, strategy, name=name, cap=cap)

    def _draw(self):
        leaf = self.leaf
        if leaf is None:
            return self.problem.random_point(rng=self.rng)
        return leaf.ranges * self.rng.random(len(leaf.ranges)) + leaf.box[:, 0]

    def on_start(self):
        try:
            splitter = self.strategy.analyzer("Splitter")
            if self.leaf is None and getattr(splitter, "root", None) is not None:
                self.leaf = splitter.root
        except Exception:
            pass
        self.fill_queue(self._draw)

    def on_new_results(self, results):
        self.fill_queue(self._draw)

    def on_new_best_box(self, best_box):
        self.leaf = best_box
        self.fill_queue(self._draw)

    def on_new_split(self, box, children, dim):
        """Track the (possibly new) leaf around the best point."""
        best = self.strategy.analyzer("Best").best
        self.leaf = self.strategy.analyzer("Splitter").get_leaf(best) if best is not None else None
        self.clear_output()
        self.fill_queue(self._draw)

    def on_restart(self, center, reason):
        """Reset the search area after a restart event.

        ``Splitter.get_leaf`` only locates leaves around *observed*
        results and a restart proposes an unobserved ``center``, so fall
        back to the Splitter's root box (the whole search space) until the
        next split assigns a tighter leaf around the new incumbent.
        """
        self.clear_output()
        splitter = self.strategy.analyzer("Splitter")
        self.leaf = getattr(splitter, "root", None)
        self.fill_queue(self._draw)
