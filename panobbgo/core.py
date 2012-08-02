# -*- coding: utf8 -*-

'''
This is the core part, currently only managing the global
DB of point evaluations. For more, look into the strategy.py file.
'''

# time.time & time.clock for cpu time
# from IPython.utils.timing import time

from utils import logger
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
    # a listener just needs a .notify([..]) method
    self._listener = set() 
    self.fx_delta_last = None
    self._best = Result(None, np.infty)

    # distance matrix, lower triangle
    from scipy import sparse
    self._distance = sparse.dok_matrix((1,1), dtype=np.float32)
    self._all = [ ]          # all results, for distance matrix
    self._dist_idx = { }     # reverse lookup, for distance matrix

  def add_listener(self, listener):
    self._listener.add(listener)

  def _update_distances(self, r):
    import numpy as np
    from IPython.utils.timing import time
    t1 = time.clock()
    self._all.append(r.x)
    r_idx = len(self._all)
    self._dist_idx[r] = r_idx
    self._distance.resize((r_idx, r_idx))
    row = r_idx - 1
    for col in range(r_idx - 1):
      d = np.linalg.norm(r.x - self._all[col], np.infty)
      if d < .1: self._distance[row, col] = d
    t = time.clock() - t1
    logger.info("t: %s" % t)
    #if len(self._all) <=6: 
    #  logger.info("distance\n%s" % self._distance.todense())
    #if len(self._all) == 6:
    #  v1 = self._all[0]
    #  v2 = self._all[5]
    #  d = np.linalg.norm(v1.x - v2.x)
    #  logger.info("v1: %s | v2: %s | d: %s" % (v1, v2, d))

  def add_results(self, new_results):
    '''
    add a new list of @Result objects.
    * calc some statistics, then
    * listeners will get notified.
    '''
    import heapq
    import numpy as np
    from heuristics import Heuristic
    if isinstance(new_results, Result):
      new_results = [ new_results ]
    for r in new_results:
      assert isinstance(r, Result), "Got object of type %s != Result" % type(r)
      heapq.heappush(self._results, r)
      self._update_distances(r)
      if r.fx < self.best.fx:
        fx_delta, reward = 0.0, 0.0
        # ATTN: always think of self.best.fx == np.infty
        #fx_delta = np.log1p(self.best.fx - r.fx) # log1p ok?
        fx_delta = 1.0 - np.exp(- 1.0 * (self.best.fx - r.fx)) # saturated to 1
        if self.fx_delta_last == None: self.fx_delta_last = fx_delta
        reward = fx_delta #/ self.fx_delta_last
        Heuristic.lookup[r.who].reward(reward)
        self.fx_delta_last = fx_delta
        logger.info(u"* %-20s %s | \u0394 %.7f" %('[%s]' % r.who, r, reward))
        self._best = r # set the new best point

    # notification
    for l in self._listener:
      l.notify(new_results)

  def __iadd__(self, results):
    self.add_results(results)
    return self

  @property
  def best(self): return self._best

  def n_best(self, n):
    import heapq
    return heapq.nsmallest(n, self._results)


