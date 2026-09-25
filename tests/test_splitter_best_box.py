# Copyright 2026 Harald Schilly <harald.schilly@gmail.com>
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

"""``Splitter.best_box`` follows the incumbent between splits."""

import numpy as np

from panobbgo.analyzers.splitter import Splitter
from panobbgo.lib import Point, Result
from panobbgo.lib.constraints import DefaultConstraintHandler
from tests.support import PanobbgoTestCase


class TestSplitterBestBox(PanobbgoTestCase):
    def setUp(self):
        from panobbgo.lib.classic import Rosenbrock

        self.problem = Rosenbrock(2)
        self.strategy = self.init_strategy()
        self.strategy.constraint_handler = DefaultConstraintHandler(self.strategy)

    def _published_boxes(self):
        return [
            c.kwargs["best_box"] for c in self.strategy.eventbus.publish.call_args_list if c.args == ("new_best_box",)
        ]

    def test_best_box_set_before_first_split(self):
        splitter = Splitter(self.strategy)
        splitter.__start__()
        splitter.on_new_results([Result(Point(np.array([0.5, 0.5]), "t"), 3.0)])
        self.assertIs(splitter.best_box, splitter.root)
        self.assertEqual(self._published_boxes(), [splitter.root])

    def test_best_box_follows_better_result_into_another_leaf(self):
        splitter = Splitter(self.strategy, min_leaf_size=4, leaf_size=4)
        splitter.__start__()
        rng = np.random.default_rng(1)
        lo, hi = self.problem.box[:, 0], self.problem.box[:, 1]
        # Fill the tree until it has split a few times.
        for i in range(40):
            x = lo + (hi - lo) * rng.random(2)
            splitter.on_new_results([Result(Point(x, "t%d" % i), 100.0 + i)])
        self.assertGreater(len(splitter.leafs), 2)
        # Route through the split events the way the event bus would.
        splitter.best_box = min(splitter.leafs, key=lambda b: b.fx)
        old_box = splitter.best_box
        other = next(b for b in splitter.leafs if b is not old_box)
        center = other.box[:, 0] + 0.5 * other.ranges
        self.strategy.eventbus.publish.reset_mock()

        r_new = Result(Point(center, "better"), -1.0)
        splitter.on_new_results([r_new])

        leaf = splitter.get_leaf(r_new)
        self.assertIs(splitter.best_box, leaf)
        self.assertIs(splitter.best_box.best, r_new)
        self.assertEqual(self._published_boxes(), [leaf])

    def test_worse_result_does_not_move_best_box(self):
        splitter = Splitter(self.strategy)
        splitter.__start__()
        splitter.on_new_results([Result(Point(np.array([0.5, 0.5]), "a"), 1.0)])
        self.strategy.eventbus.publish.reset_mock()
        splitter.on_new_results([Result(Point(np.array([0.2, 0.2]), "b"), 5.0)])
        self.assertIs(splitter.best_box, splitter.root)
        self.assertEqual(self._published_boxes(), [])
