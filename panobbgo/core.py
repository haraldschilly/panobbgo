# -*- coding: utf8 -*-

'''
This is the core part, responsible for directing the 
tasks, workers and flow of information.

Basically, one or more threads produce points where to search,
and another one consumes them and dispatches tasks.
'''

import config
import threading
import heapq
#from Queue import PriorityQueue, Empty, Queue
import numpy as np
# time.time & time.clock for cpu time
from IPython.utils.timing import time

from utils import stats, logger
from heuristics import *
from panobbgo_problems import Result


class Results(object):
  '''
  list of results w/ notificaton for 
  new results. later on, this will be a cool
  database
  '''
  def __init__(self):
    self._results = []
    # a listener just needs a .notify([..]) method
    self._listener = set() 
    self._best = Result(None, np.infty)

  def add_listener(self, listener):
    self._listener.add(listener)

  def add_results(self, results):
    '''
    add a new list of @Result objects.
    listeners will get notified.
    '''
    if isinstance(results, Result):
      results = [ results ]
    for r in results:
      heapq.heappush(self._results, r)
      if r.fx < self.best.fx: 
        if self.best.fx < np.infty:
          fx_delta = np.log1p(self.best.fx - r.fx) # TODO log1p ok?
          r.who.reward(fx_delta)
        logger.info("* %-20s %s" %('[%s]' % r.who, r))
        self._best = r # set the new best point
    for l in self._listener:
      l.notify(results)

  def __iadd__(self, results):
    self.add_results(results)
    return self

  @property
  def best(self): return self._best

  def n_best(self, n):
    return heapq.nsmallest(n, self._results)

  def n_worst(self, n):
    return heapq.nlargest(n, self._results)


#class Strategy0(object):
#  '''
#  This thread selects new points from the point generating
#  threads and submits the calculations to the cluster.
#  '''
#  def __init__(self, problem, results, heurs, nb_gens=1):
#    self.problem = problem
#    self.results = results
#    for h in heurs:
#      if not isinstance(h, Heuristic):
#        raise Exception('List must contain Heuristics')
#    self.heurs = heurs
#    self.rounds = 0
#    self._setup_cluster(nb_gens)
#
#  def run(self):
#    '''
#    This method consumes entries from search_points
#    and dispatches them to the workers in parallel.
#    '''
#    while stats.cnt < config.max_eval:
#      while True:
#        self.evaluators.spin() # check outstanding tasks
#        stats.update_finished(self.evaluators.outstanding)
#        if stats.pending < 10:
#          break
#        time.sleep(1e-3)
#
#      new_points = []
#
#      target = 10 #target 10 new points
#      perf_sum = sum(h.perf for h in self.heurs)
#      while True:
#        for h in self.heurs:
#          # calc probability based on perfomance with additive smoothing
#          delta = .5
#          prob = (h.perf + delta)/(perf_sum + delta * len(self.heurs)) 
#          np_h = int(target * prob) + 1
#          #logger.info("  %s -> %s" % (h, np_h))
#          if self.rounds < 5:
#            np_h = int(float(target) / len(self.heurs)) + 1
#          new_points.extend(h.get_points(np_h))
#
#        # TODO make this more intelligent
#        if len(new_points) == 0:
#          time.sleep(1e-3)
#        else:
#          break
#
#
#      #for point in new_points:
#      #  stats.add_cnt()
#      #  #logger.info(" new point: %s" % p)
#      #  # TODO later there is a separat thread collecting
#      #  # results and adding them to the @result list
#      #  # and notifying all generating threads
#      #  #time.sleep(1e-3)
#      #  res = Result(point, fx=self.problem(point))
#      #  self.results += res
#
#      fff = lambda x,y : x(y)
#      from IPython.parallel import Reference
#      ref = [ Reference("problem") ] * len(new_points)
#      def fff(x):
#        return 2*x
#      new_tasks = self.evaluators.map_async(fff, ref, new_points)#, chunksize = 3, ordered=False)
#      print ">>>"
#      new_tasks.wait()
#      print new_tasks.result
#      #import pdb; pdb.set_trace()
#      #for t in new_tasks:
#      # print t
#      stats.add_tasks(new_tasks)
#      stats.update_finished(self.evaluators.outstanding)
#
#      # discount all heuristics after each round
#      for h in self.heurs:
#        h.discount()
#
#      logger.debug('  '.join(('%s:%.3f' % (h, h.perf) for h in self.heurs)))
#      self.rounds += 1

#  def _setup_cluster(self, nb_gens):
#    from IPython.parallel import Client#, require, Reference
#    self._c = c = Client(profile=config.ipy_profile)
#    c.clear() # clears remote engines
#    c.purge_results('all') # all results are memorized in the hub
#
#    if len(c.ids) < nb_gens + 1:
#      raise Exception('I need at least %d clients.' % (nb_gens + 1))
#    dv_evaluators = c[nb_gens:]
#    dv_evaluators['problem'] = self.problem
#    self.generators = c.load_balanced_view(c.ids[:nb_gens])
#    self.evaluators = c.load_balanced_view(c.ids[nb_gens:])
#
#    # import some packages  (also locally)
#    with c[:].sync_imports():
#      from IPython.utils.timing import time
#      import numpy
#      import math


#class Collector(threading.Thread):
#  '''
#  This thread collects new results from the cluster
#  and sends them in the @Results list.
#  '''
#  def __init__(self, controller):
#    self.controller = controller
#    self.stop = False
#    threading.Thread.__init__(self, name='collector')
#    self.start()
#
#  def run(self):
#    while not self.stop:
#      for msg_id in stats.get_finished():
#        # we know these are done, so don't worry about blocking
#        res = self.controller.evaluators.get_result(msg_id)
#
#        # each job returns a list of length chunksize
#        for r in res.result:
#          self.controller.results += r
#
#      time.sleep(1e-3)

