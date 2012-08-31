# -*- coding: utf8 -*-
from config import get_config
from panobbgo_lib import Result
from core import Analyzer
import numpy as np

class Best(Analyzer):
  '''
  listens on all results and emits a "new_best" event,
  if a new best point has been found.
  The best point is also available via the .best field.
  '''
  def __init__(self):
    Analyzer.__init__(self)
    self.best = Result(None, np.infty)

  def on_new_result(self, result):
    r = result
    if r.fx < self.best.fx:
      #logger.info(u"\u2318 %s | \u0394 %.7f %s" %(r, 0.0, r.who))
      self.best = r
      self.eventbus.publish("new_best", best = r)

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
  manages a tree of splits. a node in a tree can have children, which
  partitions the search space into smaller boxes. nodes without
  children are leafs.
  the idea is to balance between the depth level of splits and the number
  of points inside such a box. a heuristic can build upon this
  to investigate interesting subregions.
  '''
  def __init__(self):
    Analyzer.__init__(self)
    # split, if there are more than this number of points in the box
    self.leafs = []
    self._id = 0 # block id
    self.logger = get_config().get_logger('SPLT')
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
    # in which box (a list!) is each point?
    from collections import defaultdict
    self.result2boxes = defaultdict(list)
    self.result2leaf  = {}

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
    #logger.info("Split: %s -> %s" % (box, ','.join(map(str, children))))
    #logger.info("children: %s" % map(lambda x:(x.depth, len(x)), children))
    pass

  class Box(object):
    '''
    used by Splitter, therefore nested.
    (accessed via Splitter.Box)
    '''
    def __init__(self, parent, splitter, box, depth = 0):
      self.parent    = parent
      self.depth     = depth
      self.box       = box
      self.splitter  = splitter
      self.limit     = splitter.limit
      self.dim       = splitter.dim
      self.results   = []
      self.children  = []
      self.split_dim = None
      self.id        = splitter._id
      splitter._id  += 1

    @property
    def leaf(self):
      '''return true, if this box is a leaf. i.e. no children'''
      return len(self.children) == 0

    @property
    def ranges(self):
      return self.box[:,1] - self.box[:,0]

    def register_result(self, result):
      assert isinstance(result, Result)
      self.results.append(result)
      self.splitter.result2boxes[result].append(self)
      if self.leaf:
        self.splitter.result2leaf[result] = self

    def __iadd__(self, result):
      self.register_result(result)
      if not self.leaf:
        for child in self.get_child_boxes(result.x):
          child += result # recursive
      elif self.leaf and len(self.results) >= self.limit:
        self.split()
      return self

    def __len__(self): return len(self.results)

    def split(self, dim = None):
      assert self.leaf, 'only leaf boxes are allowed to be split'
      if dim is None:
        scaled_coords = np.vstack(map(lambda r:r.x, self.results)) / self.ranges
        dim = np.argmax(np.std(scaled_coords, axis=0))
      assert dim >=0 and dim < self.dim, 'dimension along where to split is %d' % dim
      next_depth = self.depth + 1
      b1 = Splitter.Box(self, self.splitter, self.box.copy(), depth = next_depth)
      b2 = Splitter.Box(self, self.splitter, self.box.copy(), depth = next_depth)
      self.split_dim = dim
      split_point = np.median(map(lambda r:r.x[dim], self.results))
      b1.box[dim, 1] = split_point
      b2.box[dim, 0] = split_point
      self.children.extend([ b1, b2 ])
      self.splitter.leafs.remove(self)
      map(self.splitter.leafs.append, self.children)
      for r in self.results:
        for c in self.children:
          if c.contains(r.x):
            c.register_result(r)
      self.splitter.eventbus.publish('new_split', \
          box = self, children = self.children, dim = dim)

    def contains(self, point):
      '''
      true, if given point is inside (including boundaries) this box
      '''
      l, u = self.box[:,0], self.box[:,1]
      return (l <= point).all() and (u >= point).all()

    def get_child_boxes(self, point):
      assert not self.leaf, 'not applicable for "leaf" box'
      ret = filter(lambda c : c.contains(point), self.children)
      assert len(ret) > 0, "no child box containing %s found!" % point
      return ret

    def __repr__(self):
      l = '(%d,leaf) ' if self.leaf else '%d '
      l = l % len(self)
      b = ','.join('%s'%_ for _ in self.box)
      return 'Box-%d %s[%s]' % (self.id, l, b)

# end Splitter

class Rewarder(Analyzer):
  '''
  list for new points and rewards the heuristic, e.g. if it is better
  than the currently known best point
  '''
  def __init__(self):
    Analyzer.__init__(self)
    self.logger = get_config().get_logger('RWRD')

  @property
  def best(self):
    return self.strategy.analyzer("best").best

  def _reward_heuristic(self, result):
    '''
    for each new result, decide if there is a reward and also
    calculate its amount.
    '''
    import numpy as np
    # currently, only reward if better point found.
    # TODO in the future also reward if near the best value (but
    # e.g. not in the proximity of the best x)
    fx_delta, reward = 0.0, 0.0
    if result.fx < self.best.fx:
      # ATTN: always take care of self.best.fx == np.infty
      #fx_delta = np.log1p(self.best.fx - r.fx) # log1p ok?
      fx_delta = 1.0 - np.exp(-1.0 * (self.best.fx - result.fx)) # saturates to 1
      #if self.fx_delta_last == None: self.fx_delta_last = fx_delta
      reward = fx_delta #/ self.fx_delta_last
      self.strategy.heuristic(result.who).reward(reward)
      #self.fx_delta_last = fx_delta
    return reward

  def on_new_results(self, results):
    for result in results:
      if result.fx < self.best.fx:
        reward = self._reward_heuristic(result)
        self.logger.info(u"\u2318 %s | \u0394 %.7f %s" %(result, reward, result.who))
