# -*- coding: utf8 -*-
import config
logger = config.get_logger('ALYZ')
from panobbgo_problems import Result
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

  def on_new_result(self, events):
    for event in events:
      r = event.result
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

  def on_new_results(self, events):
    for event in events:
      for result in event.results:
        self._grid_add(result)

#
# Splitter + its classes
#

class Box(object):
  '''
  used by Splitter
  '''
  def __init__(self, parent, splitter, box):
    self.parent    = parent
    self.box       = box
    self.splitter  = splitter
    self.limit     = splitter.limit
    self.dim       = splitter.dim
    self.results   = []
    self.children  = []
    self.dim_split = None

  @property
  def leaf(self):
    '''return true, if this box is a leaf. i.e. no children'''
    return len(self.children) == 0

  def __iadd__(self, result):
    assert isinstance(result, Result)
    self.results.append(result)
    if not self.leaf:
      child = self.get_box(result.x)
      child += result
    elif self.leaf and len(self.results) >= self.limit:
      if self.parent is None:
        d = 0  # we are the root box
      else:
        d = (self.parent.dim_split + 1) % self.dim
      self.split(d)
    return self

  def split(self, dim, where = .5):
    assert dim >=0 and dim < self.dim, 'dimension along where to split is %d' % dim
    assert where >= 0.0 and where <= 1.0, 'where must be between 0 and 1, not %s' % where
    b1 = Box(self, self.splitter, self.box.copy())
    b2 = Box(self, self.splitter, self.box.copy())
    self.dim_split = dim
    l, u = self.box[dim, 0], self.box[dim, 1]
    split_point = l + (u - l) * where
    b1.box[dim, 1] = split_point
    b2.box[dim, 0] = split_point
    self.children = [ b1, b2 ]
    self.splitter.eventbus.publish('new_split', boxes = self.children)

  def get_box(self, point):
    assert not self.leaf, 'not applicable for "leaf" box'
    # assume non-overlapping children
    for c in self.children:
      l, u = c.box[:,0], c.box[:,1]
      if (l <= point).all() and (u >= point).all():
        return c
    raise Exception("no child box containing %s found!" % point)

  def __repr__(self):
    l = '(leaf) ' if self.leaf else ''
    b = ','.join('%s'%_ for _ in self.box)
    return 'Box %s[%s]' % (l, b)

class Splitter(Analyzer):
  '''
  manages a tree of splits. a split is a node in a tree, that
  partitions the search space into smaller boxes. the idea is
  to balance between the depth level of splits and the number
  of points inside such a box. a heuristic can build upon this
  to investigate interesting subregions.
  '''
  def __init__(self):
    Analyzer.__init__(self)
    # split, if there are more than this number of points
    # in the box
    self.limit = 10

  def _init_(self):
    # root box is equal to problem's box
    self.dim  = self.problem.dim
    self.root = Box(None, self, self.problem.box.copy())

  def ranges(self, box):
    return box[:,1] - box[:,0]

  def get_box(self, point):
    '''return box, where point is contained in'''
    box = self.root
    while not box.leaf:
      box = box.get_box(point)
    return box

  def on_new_results(self, events):
    for event in events:
      for result in event.results:
        box = self.get_box(result.x)
        box += result

  def on_new_split(self, events):
    for e in events:
      logger.info("Split: %s" % ','.join(map(str, e.boxes)))

# end Splitter

class Rewarder(Analyzer):
  '''
  list for new points and rewards the heuristic, e.g. if it is better
  than the currently known best point
  '''
  def __init__(self):
    Analyzer.__init__(self)

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

  def on_new_results(self, events):
    for event in events:
      for result in event.results:
        if result.fx < self.best.fx:
          reward = self._reward_heuristic(result)
          logger.info(u"\u2318 %s | \u0394 %.7f %s" %(result, reward, result.who))

