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

r'''
Analyzers
=========

Analyzers, just like :mod:`.heuristics`, listen to events
and change their internal state based on local and
global data. They can emit :class:`events <panobbgo.core.Event>`
on their own and they are accessible via the
:meth:`~panobbgo.core.StrategyBase.analyzer` method of
the strategy.

.. inheritance-diagram:: panobbgo.analyzers

.. codeauthor:: Harald Schilly <harald.schilly@univie.ac.at>
'''
from config import get_config
from panobbgo_lib import Result
from core import Analyzer
from utils import memoize
import numpy as np

class Best(Analyzer):
  '''
  Listens on all results and emits the following events:

  - ``new_best``: when a new "best" point,
  - ``new_min``: a point with smallest objective function value,
  - ``new_cv``: one with a smaller
    :attr:`~panobbgo_lib.lib.Result.constraint_violation`, or
  - ``new_pareto``: when a best point on the pareto front of the
    objective value and constraint violation

  has been found.

  The best point (pareto) is available via the :attr:`.best` attribute.
  There is also :attr:`.cv` and :attr:`.pareto`.
  '''
  def __init__(self):
    Analyzer.__init__(self)
    self.logger   = get_config().get_logger("BEST")
    r             = Result(None, np.infty, cv = np.infty)
    self._min     = r
    self._cv      = r
    self._pareto  = r

  @property
  def best(self):
    '''
    Currently best :class:`~panobbgo_lib.lib.Result`.

    .. Note::

      At the moment, this is :attr:`.pareto` but might change.
    '''
    return self._pareto

  @property
  def cv(self):
    '''
    The point with currently minimal constraint violation, likely 0.0.
    If there are several points with the same minimal constraint violation,
    the value of the objective function is a secondary selection argument.
    '''
    return self._cv

  @property
  def pareto(self):
    return self._pareto

  @property
  def min(self):
    '''
    The point, with currently smallest objective function value.
    '''
    return self._min

  def on_new_result(self, result):
    r = result
    if r.fx < self._min.fx:
      #self.logger.info(u"\u2318 %s by %s" %(r, r.who))
      self._min = r
      self.eventbus.publish("new_min", min = r)
    if r.cv < self._cv.cv or (r.cv == self._cv.cv and r.fx < self._cv.fx):
      self._cv = r
      self.eventbus.publish("new_cv", cv = r)
    if np.hypot(self._pareto.cv, self._pareto.fx) > np.hypot(r.cv, r.fx):
      self._pareto = r
      self.eventbus.publish("new_pareto", pareto = r)
      self.eventbus.publish("new_best", best = r)

  def on_new_pareto(self, pareto):
    #self.logger.info("pareto: %s" % pareto)
    pass

  def on_new_cv(self, cv):
    #self.logger.info("cv: %s" % cv)
    pass

  def on_new_min(self, min):
    #self.logger.info("min: %s" % cv)
    pass

class Grid(Analyzer):
  '''
  packs nearby points into grid boxes
  '''
  def __init__(self):
    Analyzer.__init__(self)

  def _init_(self):
    # grid for storing points which are nearby.
    # maps from rounded coordinates tuple to point
    self._grid = dict()
    self._grid_div = 5.
    self._grid_lengths = self.problem.ranges / float(self._grid_div)

  def in_same_grid(self, point):
    key = tuple(self._grid_mapping(point.x))
    return self._grid.get(key, [])

  def _grid_mapping(self, x):
    from numpy import floor
    l = self._grid_lengths
    #m = self._problem.box[:,0]
    return tuple(floor(x / l) * l)

  def _grid_add(self, r):
    key = self._grid_mapping(r.x)
    box = self._grid.get(key, [])
    box.append(r)
    self._grid[key] = box
    #print ' '.join('%2s' % str(len(self._grid[k])) for k in sorted(self._grid.keys()))

  def on_new_results(self, results):
    for result in results:
      self._grid_add(result)

#
# Splitter + inside is its Box class
#

class Splitter(Analyzer):
  '''
  Manages a tree of splits.
  Each split in this tree is a :class:`box <.Splitter.Box>`, which
  partitions the search space into smaller boxes and can have children.
  Boxes without children are :attr:`leafs <.Splitter.Box.leaf>`.

  The goal for this splitter is to balance between the 
  depth level of splits and the number of points inside such a box.

  A heuristic can build upon this hierarchy
  to investigate interesting subregions.
  '''
  def __init__(self):
    Analyzer.__init__(self)
    # split, if there are more than this number of points in the box
    self.leafs = []
    self._id = 0 # block id
    self.logger = get_config().get_logger('SPLIT') #, 10)
    self.max_eval = get_config().max_eval
    # _new_result used to signal get_leaf and others when there
    # are updates regarding box/split/leaf status
    from threading import Condition
    self._new_result = Condition()

  def _init_(self):
    # root box is equal to problem's box
    self.dim  = self.problem.dim
    self.limit = max(20, self.max_eval / self.dim ** 2)
    self.logger.info("limit = %s" % self.limit)
    self.root = Splitter.Box(None, self, self.problem.box.copy())
    self.leafs.append(self.root)
    # big boxes
    self.biggest_leaf = self.root
    self.big_by_depth = {}
    self.big_by_depth[self.root.depth] = self.root
    self.max_depth = self.root.depth
    # in which box (a list!) is each point?
    from collections import defaultdict
    self.result2boxes = defaultdict(list)
    self.result2leaf  = {}

  def _new_box(self, new_box):
    '''
    Called for each new box when there is a split.
    E.g. it updates the ``biggest`` box and related
    information for each depth level.
    '''
    self.max_depth = max(new_box.depth, self.max_depth)

    old_biggest_leaf = self.biggest_leaf
    self.biggest_leaf = max(self.leafs, key = lambda l:l.log_volume)
    if old_biggest_leaf is not self.biggest_leaf:
      self.eventbus.publish('new_biggest_leaf', box = new_box)

    dpth = new_box.depth
    # also consider the parent depth level
    for d in [dpth - 1, dpth]:
      old_big_by_depth = self.big_by_depth.get(d, None)
      if old_big_by_depth is None:
        self.big_by_depth[d] = new_box
      else:
        leafs_at_depth = list(filter(lambda l:l.depth == d, self.leafs))
        if len(leafs_at_depth) > 0:
          self.big_by_depth[d] = min(leafs_at_depth, key = lambda l:l.log_volume)

      if self.big_by_depth[d] is not old_big_by_depth:
        self.eventbus.publish('new_biggest_by_depth',
            depth=d, box = self.big_by_depth[d])

  def on_new_biggest_leaf(self, box):
    self.logger.debug("biggest leaf at depth %d -> %s" % (box.depth, box))

  def on_new_biggest_by_depth(self, depth, box):
    self.logger.debug("big by depth: %d -> %s" % (depth, box))

  def get_box(self, point):
    '''
    return "leftmost" leaf box, where given point is contained in
    '''
    box = self.root
    while not box.leaf:
      box = box.get_child_boxes(point)[0]
    return box

  def get_all_boxes(self, result):
    '''
    return all boxes, where point is contained in
    '''
    assert isinstance(result, Result)
    return self.result2boxes[result]

  def get_leaf(self, result):
    '''
    returns the leaf box, where given result is currently sitting in
    '''
    assert isinstance(result, Result)
    # it might happen, that the result isn't in the result2leaf map
    # then we have to wait until on_new_results got it
    with self._new_result:
      while not self.result2leaf.has_key(result):
        #logger.info("RESULT NOT FOUND %s" % result)
        #logger.info("BOXES: %s" % self.get_all_boxes(result))
        self._new_result.wait()
    return self.result2leaf[result]

  def in_same_leaf(self, result):
    l = self.get_leaf(result)
    return l.results, l

  def on_new_results(self, results):
    with self._new_result:
      for result in results:
        self.root += result
      self._new_result.notify_all()
    #logger.info("leafs: %s" % map(lambda x:(x.depth, len(x)), self.leafs))
    #logger.info("point %s in boxes: %s" % (result.x, self.get_all_boxes(result)))
    #logger.info("point %s in leaf: %s" % (result.x, self.get_leaf(result)))
    #assert self.get_all_boxes(result)[-1] == self.get_leaf(result)

  def on_new_split(self, box, children, dim):
    self.logger.debug("Split: %s" % box)
    for i, chld in enumerate(children):
      self.logger.debug(" +ch%d: %s" % (i, chld))
    #logger.info("children: %s" % map(lambda x:(x.depth, len(x)), children))
    pass

  class Box(object):
    '''
    Used by :class:`.Splitter`, therefore nested.

    Most important routine is :meth:`.split`. 

    .. Note::

      In the future, this might be refactored to allow different
      splitting methods.
    '''
    def __init__(self, parent, splitter, box, depth = 0):
      self.parent    = parent
      self.logger    = splitter.logger
      self.depth     = depth
      self.box       = box
      self.splitter  = splitter
      self.limit     = splitter.limit
      self.dim       = splitter.dim
      self.best      = None              # best point
      self.results   = []
      self.children  = []
      self.split_dim = None
      self.id        = splitter._id
      splitter._id  += 1

    @property
    def leaf(self):
      '''
      returns ``true``, if this box is a leaf. i.e. no children
      '''
      return len(self.children) == 0

    @property
    def fx(self):
      '''
      Function value of best point in this particular box.
      '''
      return self.best.fx

    @memoize
    def __ranges(self):
      return self.box[:,1] - self.box[:,0]

    @property
    def ranges(self):
      '''
      Gives back a vector with all the ranges of this box,
      i.e. upper - lower bound.
      '''
      return self.__ranges()

    @memoize
    def __log_volume(self):
      return np.sum(np.log(self.ranges))

    @property
    def log_volume(self):
      '''
      Returns the `logarithmic` volume of this box.
      '''
      return self.__log_volume()

    @memoize
    def __volume(self):
      return np.exp(self.log_volume)

    @property
    def volume(self):
      '''
      Returns the volume of the box.

      .. Note::

        Currently, the exponential of :attr:`.log_volume`
      '''
      return self.__volume()

    def _register_result(self, result):
      '''
      This updates the splitter and box specific datatypes,
      i.e. the maps from a result to the corresponding boxes or leafs.
      '''
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
      '''
      Registers and adds a new :class:`~panobbgo_lib.lib.Result`.
      In particular, it adds the given ``result`` to the
      current box and it's children (also all descendents).

      If the current box is a leaf and too big, the :meth:`.split`
      routine is called.

      .. Note::

        ``box += result`` is fine, too.
      '''
      self._register_result(result)
      if not self.leaf:
        for child in self.get_child_boxes(result.x):
          child += result # recursive
      elif self.leaf and len(self.results) >= self.limit:
        self.split()

    def __iadd__(self, result):
      '''Convenience wrapper for :meth:`.add_result`.'''
      self.add_result(result)
      return self

    def __len__(self): return len(self.results)

    def split(self, dim = None):
      assert self.leaf, 'only leaf boxes are allowed to be split'
      if dim is None:
        scaled_coords = np.vstack(map(lambda r:r.x, self.results)) / self.ranges
        dim = np.argmax(np.std(scaled_coords, axis=0))
      #self.logger.debug("dim: %d" % dim)
      assert dim >=0 and dim < self.dim, 'dimension along where to split is %d' % dim
      next_depth = self.depth + 1
      b1 = Splitter.Box(self, self.splitter, self.box.copy(), depth = next_depth)
      b2 = Splitter.Box(self, self.splitter, self.box.copy(), depth = next_depth)
      self.split_dim = dim
      #split_point = np.median(map(lambda r:r.x[dim], self.results))
      split_point = np.average(map(lambda r:r.x[dim], self.results))
      b1.box[dim, 1] = split_point
      b2.box[dim, 0] = split_point
      self.children.extend([ b1, b2 ])
      self.splitter.leafs.remove(self)
      map(self.splitter.leafs.append, self.children)
      for c in self.children:
        self.splitter._new_box(c)
        for r in self.results:
          if c.contains(r.x):
            c._register_result(r)
      self.splitter.eventbus.publish('new_split', \
          box = self, children = self.children, dim = dim)

    def contains(self, point):
      '''
      true, if given point is inside this box (including boundaries).
      '''
      l, u = self.box[:,0], self.box[:,1]
      return (l <= point).all() and (u >= point).all()

    def get_child_boxes(self, point):
      '''
      returns all immediate child boxes, which contain given point.
      '''
      assert not self.leaf, 'not applicable for "leaf" box'
      ret = filter(lambda c : c.contains(point), self.children)
      assert len(ret) > 0, "no child box containing %s found!" % point
      return ret

    def __repr__(self):
      v = self.volume
      l = ',leaf' if self.leaf else ''
      l = '(%d,%.3f%s) ' % (len(self), v, l)
      b = ','.join('%s'%_ for _ in self.box)
      return 'Box-%d %s[%s]' % (self.id, l, b)

# end Splitter
