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
  def __init__(self):
    self._results = []
    # a listener just needs a .notify([..]) method
    self._listener = set() 
    #self.fx_delta_last = None
    from numpy import infty
    self._best = Result(None, infty)

  def add_listener(self, listener):
    self._listener.add(listener)

  def add_results(self, results):
    '''
    add a new list of @Result objects.
    * calc some statistics, then
    * listeners will get notified.
    '''
    import heapq
    import numpy as np
    from heuristics import Heuristic
    if isinstance(results, Result):
      results = [ results ]
    for r in results:
      assert isinstance(r, Result), "Got object of type %s != Result" % type(r)
      heapq.heappush(self._results, r)
      if r.fx < self.best.fx:
        fx_delta, reward = 0.0, 0.0
        if self.best.fx < np.infty:
          fx_delta = np.log1p(self.best.fx - r.fx) # TODO log1p ok?
          #if self.fx_delta_last == None: self.fx_delta_last = fx_delta
          reward = fx_delta# / self.fx_delta_last
          Heuristic.lookup[r.who].reward(reward)
          self.fx_delta_last = fx_delta
        logger.info(u"* %-20s %s | \u0394 %.7f" %('[%s]' % r.who, r, reward))
        self._best = r # set the new best point

    # notification
    for l in self._listener:
      l.notify(results)

  def __iadd__(self, results):
    self.add_results(results)
    return self

  @property
  def best(self): return self._best

  def n_best(self, n):
    import heapq
    return heapq.nsmallest(n, self._results)

  def n_worst(self, n):
    import heapq
    return heapq.nlargest(n, self._results)


