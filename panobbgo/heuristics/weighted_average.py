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
from panobbgo.analyzers.best import Best

from panobbgo.core import Heuristic


class WeightedAverage(Heuristic):
    """
    This strategy calculates the weighted average of all points
    in the box around the best point of the :class:`~panobbgo.analyzers.Splitter`.
    """

    def __init__(self, strategy, k=0.1):
        Heuristic.__init__(self, strategy)
        self.k = k
        self.logger = self.config.get_logger("WAvg")

    def __start__(self):
        self.minstd = min(self.problem.ranges) / 1000.0

    def check_dependencies(self, analyzers, heuristics):
        return any(isinstance(a, Best) for a in analyzers)

    def on_new_best(self, best):
        assert best is not None and best.x is not None
        box = self.strategy.analyzer("Splitter").get_leaf(best)
        if len(box.results) < 3:
            return

        # actual calculation
        import numpy as np

        get_val = self.strategy.constraint_handler.get_penalty_value
        xx = np.array([r.x for r in box.results])
        yy = np.array([get_val(r) for r in box.results])
        best_val = get_val(best)
        weights = np.log1p(yy - best_val)
        weights = -weights + (1 + self.k) * weights.max()
        # weights = np.log1p(np.arange(len(yy) + 1, 1, -1))
        # self.logger.info("weights: %s" % zip(weights, yy))
        self.clear_output()
        avg_ret = np.average(xx, axis=0, weights=weights)
        std = xx.std(axis=0)
        # std must be > 0
        std[std < self.minstd] = self.minstd
        # self.logger.info("std: %s" % std)
        for i in range(self.cap):
            ret = avg_ret.copy()
            ret += (float(i) / self.cap) * np.random.normal(0, std)
            if np.linalg.norm(best.x - ret) > 0.01:
                # self.logger.info("out: %s" % ret)
                self.emit(ret)
