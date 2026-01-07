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
from panobbgo.utils import memoize

import numpy as np


"""
Splitter
--------

Inside is its Box class.
"""


class Splitter(Analyzer):

    """
    Manages a tree of splits.
    Each split in this tree is a :class:`box <.Splitter.Box>`, which
    partitions the search space into smaller boxes and can have children.
    Boxes without children are :attr:`leafs <.Splitter.Box.leaf>`.

    The goal for this splitter is to balance between the
    depth level of splits and the number of points inside such a box.

    A heuristic can build upon this hierarchy
    to investigate interesting subregions.
    """

    def __init__(self, strategy):
        Analyzer.__init__(self, strategy)
        # split, if there are more than this number of points in the box
        self.leafs = []
        self._id = 0  # block id
        self.logger = self.config.get_logger('SPLIT')  # , 10)
        self.max_eval = self.config.max_eval
        # _new_result used to signal get_leaf and others when there
        # are updates regarding box/split/leaf status
        from threading import Condition
        self._new_result = Condition()

    def __start__(self):
        # root box is equal to problem's box
        self.dim = self.problem.dim
        self.limit = max(20, self.max_eval / self.dim ** 2)
        self.logger.debug("limit = %s" % self.limit)
        self.root = Splitter.Box(None, self, self.problem.box.copy())
        self.leafs.append(self.root)
        # big boxes
        self.biggest_leaf = self.root
        self.big_by_depth = dict()
        self.big_by_depth[self.root.depth] = self.root
        self.max_depth = self.root.depth
        # best box (with best f(x))
        self.best_box = None
        # in which box (a list!) is each point?
        from collections import defaultdict
        self.result2boxes = defaultdict(list)
        self.result2leaf = {}

    def _new_box(self, new_box):
        """
        Called for each new box when there is a split.
        E.g. it updates the ``biggest`` box and related
        information for each depth level.
        """
        self.max_depth = max(new_box.depth, self.max_depth)

        old_biggest_leaf = self.biggest_leaf
        self.biggest_leaf = max(self.leafs, key=lambda l: l.log_volume)
        if old_biggest_leaf is not self.biggest_leaf:
            self.eventbus.publish('new_biggest_leaf', box=new_box)

        dpth = new_box.depth
        # also consider the parent depth level
        for d in [dpth - 1, dpth]:
            old_big_by_depth = self.big_by_depth.get(d, None)
            if old_big_by_depth is None:
                self.big_by_depth[d] = new_box
            else:
                leafs_at_depth = list(
                    [l for l in self.leafs if l.depth == d])
                if len(leafs_at_depth) > 0:
                    self.big_by_depth[d] = max(
                        leafs_at_depth, key=lambda l: l.log_volume)

            if self.big_by_depth[d] is not old_big_by_depth:
                self.eventbus.publish('new_biggest_by_depth',
                                      depth=d, box=self.big_by_depth[d])

    def on_new_biggest_leaf(self, box):
        self.logger.debug("biggest leaf at depth %d -> %s" % (box.depth, box))

    def on_new_biggest_by_depth(self, depth, box):
        self.logger.debug("big by depth: %d -> %s" % (depth, box))

    def get_box(self, point):
        """
        return "leftmost" leaf box, where given point is contained in
        """
        box = self.root
        while not box.leaf:
            box = box.get_child_boxes(point)[0]
        return box

    def get_all_boxes(self, result):
        """
        return all boxes, where point is contained in
        """
        from panobbgo.lib.lib import Result
        assert isinstance(result, Result)
        return self.result2boxes[result]

    def get_leaf(self, result):
        """
        returns the leaf box, where given result is currently sitting in
        """
        from panobbgo.lib.lib import Result
        assert isinstance(result, Result)
        # it might happen, that the result isn't in the result2leaf map
        # then we have to wait until on_new_results got it
        with self._new_result:
            while result not in self.result2leaf:
                # logger.info("RESULT NOT FOUND %s" % result)
                # logger.info("BOXES: %s" % self.get_all_boxes(result))
                self._new_result.wait()
        return self.result2leaf[result]

    def on_new_results(self, results):
        with self._new_result:
            for result in results:
                self.root += result
            self._new_result.notify_all()
        # logger.info("leafs: %s" % map(lambda x:(x.depth, len(x)), self.leafs))
        # logger.info("point %s in boxes: %s" % (result.x, self.get_all_boxes(result)))
        # logger.info("point %s in leaf: %s" % (result.x, self.get_leaf(result)))
        # assert self.get_all_boxes(result)[-1] == self.get_leaf(result)

    def on_new_split(self, box, children, dim):
        self.logger.debug("Split: %s" % box)
        for i, chld in enumerate(children):
            self.logger.debug(" +ch%d: %s" % (i, chld))
        # logger.info("children: %s" % map(lambda x:(x.depth, len(x)),
        # children))

        # update self.best_box
        # check if new box contains the best point (>= because it could
        # be a child box)
        for new_box in children:
            if self.best_box is None or self.best_box.fx >= new_box.fx:
                self.best_box = new_box
        self.eventbus.publish('new_best_box', best_box=self.best_box)

    class Box:

        """
        Used by :class:`.Splitter`, therefore nested.

        Most important routine is :meth:`.split`.

        .. Note::

          In the future, this might be refactored to allow different
          splitting methods.
        """

        def __init__(self, parent, splitter, box):
            self.parent = parent
            self.logger = splitter.logger
            self.depth = parent.depth + 1 if parent else 0
            self.box = box
            self.splitter = splitter
            self.limit = splitter.limit
            self.dim = splitter.dim
            self.best = None              # best point
            self.results = []
            self.children = []
            self.split_dim = None
            self.id = splitter._id
            splitter._id += 1

        @property
        def leaf(self):
            """
            returns ``true``, if this box is a leaf. i.e. no children
            """
            return len(self.children) == 0

        @property
        def fx(self):
            """
            Function value of best point in this particular box.
            """
            if self.best is None:
                return float('inf')
            return self.best.fx

        @memoize
        def __ranges(self):
            return self.box.ptp(axis=1)  # self.box[:,1] - self.box[:,0]

        @property
        def ranges(self):
            """
            Gives back a vector with all the ranges of this box,
            i.e. upper - lower bound.
            """
            return self.__ranges()

        @memoize
        def __log_volume(self):
            return np.sum(np.log(self.ranges))

        @property
        def log_volume(self):
            """
            Returns the `logarithmic` volume of this box.
            """
            return self.__log_volume()

        @memoize
        def __volume(self):
            return np.exp(self.log_volume)

        @property
        def volume(self):
            """
            Returns the volume of the box.

            .. Note::

              Currently, the exponential of :attr:`.log_volume`
            """
            return self.__volume()

        def _register_result(self, result):
            """
            This updates the splitter and box specific datatypes,
            i.e. the maps from a result to the corresponding boxes or leafs.
            """
            from panobbgo.lib.lib import Result
            assert isinstance(result, Result)
            self.results.append(result)

            # new best result in box? (for best fx value, too)
            if self.best is not None:
                if self.best.fx > result.fx:
                    self.best = result
            else:
                self.best = result

            self.splitter.result2boxes[result].append(self)
            if self.leaf:
                self.splitter.result2leaf[result] = self

        def add_result(self, result):
            """
            Registers and adds a new :class:`~panobbgo.lib.lib.Result`.
            In particular, it adds the given ``result`` to the
            current box and it's children (also all descendents).

            If the current box is a leaf and too big, the :meth:`.split`
            routine is called.

            .. Note::

              ``box += result`` is fine, too.
            """
            self._register_result(result)
            if not self.leaf:
                for child in self.get_child_boxes(result.x):
                    child += result  # recursive
            elif self.leaf and len(self.results) >= self.limit:
                self.split()

        def __iadd__(self, result):
            """
            Convenience wrapper for :meth:`.add_result`.
            """
            self.add_result(result)
            return self

        def __len__(self):
            return len(self.results)

        def split(self, dim=None):
            """
            Arguments::

            - ``dim``: Dimension, along which to split. (default: `None`, and calculated)
            """
            assert self.leaf, 'only leaf boxes are allowed to be split'
            if dim is None:
                # scaled_coords = np.vstack(map(lambda r:r.x, self.results)) / self.ranges
                # dim = np.argmax(np.std(scaled_coords, axis=0))
                dim = np.argmax(self.ranges)
            # self.logger.debug("dim: %d" % dim)
            assert dim >= 0 and dim < self.dim, 'dimension along where to split is %d' % dim
            b1 = Splitter.Box(self, self.splitter, self.box.copy())
            b2 = Splitter.Box(self, self.splitter, self.box.copy())
            self.split_dim = dim
            # split_point = np.median(map(lambda r:r.x[dim], self.results))
            split_point = np.average([r.x[dim] for r in self.results])
            b1.box[dim, 1] = split_point
            b2.box[dim, 0] = split_point
            self.children.extend([b1, b2])
            self.splitter.leafs.remove(self)
            list(map(self.splitter.leafs.append, self.children))
            for c in self.children:
                self.splitter._new_box(c)
                for r in self.results:
                    if c.contains(r.x):
                        c._register_result(r)
            self.splitter.eventbus.publish('new_split',
                                           box=self, children=self.children, dim=dim)

        def contains(self, point):
            """
            true, if given point is inside this box (including boundaries).
            """
            l, u = self.box[:, 0], self.box[:, 1]
            return (l <= point).all() and (u >= point).all()

        def get_child_boxes(self, point):
            """
            returns all immediate child boxes, which contain given point.
            """
            assert not self.leaf, 'not applicable for "leaf" box'
            ret = [c for c in self.children if c.contains(point)]
            assert len(ret) > 0, "no child box containing %s found!" % point
            return ret

        def __repr__(self):
            v = self.volume
            l = ',leaf' if self.leaf else ''
            l = '(%d,%.3f%s) ' % (len(self), v, l)
            b = ','.join('%s' % _ for _ in self.box)
            return 'Box-%d %s[%s]' % (self.id, l, b)
