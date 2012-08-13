# -*- coding: utf8 -*-

'''
This is the core part, currently only managing the global
DB of point evaluations. For more, look into the strategy.py file.
'''

from config import loggers
logger = loggers['core']
from panobbgo_problems import Result

class Results(object):
  '''
  List of results w/ notificaton for new results.
  Later on, this will be a cool database.
  '''
  def __init__(self, problem):
    import numpy as np
    self._problem = problem
    self._results = []
    self._last_nb = 0 #for logging
    # a listener just needs a .notify([..]) method
    self._listener = set() 
    self.fx_delta_last = None
    self._best = Result(None, np.infty)

    # grid for storing points which are nearby.
    # maps from rounded coordinates tuple to point
    self._grid = dict()
    self._grid_div = 10.
    self._grid_lengths = self._problem.ranges / float(self._grid_div)

  def add_listener(self, listener):
    self._listener.add(listener)

  def in_same_grid(self, point):
    key = tuple(self._grid_mapping(point.x))
    points = self._grid.get(key, [])
    return points

  def _grid_mapping(self, x):
    from numpy import floor
    l = self._grid_lengths
    #m = self._problem.box[:,0]
    return floor((x) / l) * l

  def _grid_add(self, r):
    key = tuple(self._grid_mapping(r.x))
    bin = self._grid.get(key, [])
    bin.append(r.point)
    self._grid[key] = bin

  def _reward_heuristic(self, r):
    '''
    for each new result, decide if there is a reward and also
    calculate its amount.
    '''
    import numpy as np
    from heuristics import Heuristic
    # currently, only reward if better point found.
    # TODO in the future also reward if near the best value (but
    # e.g. not in the proximity of the best x)
    fx_delta, reward = 0.0, 0.0
    if r.fx < self.best.fx:
      # ATTN: always take care of self.best.fx == np.infty
      #fx_delta = np.log1p(self.best.fx - r.fx) # log1p ok?
      fx_delta = 1.0 - np.exp(-1.0 * (self.best.fx - r.fx)) # saturates to 1
      #if self.fx_delta_last == None: self.fx_delta_last = fx_delta
      reward = fx_delta #/ self.fx_delta_last
      Heuristic.lookup[r.who].reward(reward)
      #self.fx_delta_last = fx_delta
    return reward

  def add_results(self, new_results):
    '''
    add one single or a list of new @Result objects.
    * calc some statistics, then
    * listeners will get notified.
    '''
    import heapq
    if isinstance(new_results, Result):
      new_results = [ new_results ]
    for r in new_results:
      assert isinstance(r, Result), "Got object of type %s != Result" % type(r)
      heapq.heappush(self._results, r)
      self._grid_add(r)
      reward = self._reward_heuristic(r)
      # new best solution found?
      if r.fx < self.best.fx:
        logger.info(u"\u2318 %s | \u0394 %.7f %s" %(r, reward, r.who))
        self._best = r # set the new best point
    if len(self._results) / 100 > self._last_nb / 100:
      #self.info()
      self._last_nb = len(self._results)

    # notification
    for l in self._listener:
      l.notify(new_results)

  def info(self):
    logger.info("%d results in DB" % len(self._results))

  def __iadd__(self, results):
    self.add_results(results)
    return self

  def __len__(self):
    return len(self._results)

  @property
  def best(self): return self._best

  def n_best(self, n):
    import heapq
    return heapq.nsmallest(n, self._results)


