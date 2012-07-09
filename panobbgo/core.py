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
from Queue import PriorityQueue, Empty, Queue
import numpy as np
# time.time & time.clock for cpu time
from IPython.utils.timing import time

from utils import logger, stats
from heuristics import *

class Problem(object):
  '''
  this is used to store the objective function, 
  information about the problem, etc.
  '''
  def __init__(self, box):
    '''
    box must be a list of tuples, which specify
    the range of each variable. 

    example: [(-1,1), (-100, 0), (0, 0.01)]
    '''
    # validate
    if not isinstance(box, (list, tuple)):
      raise Exception("box argument must be a list or tuple")
    for entry in box:
      if not len(entry) == 2:
        raise Exception("box entries must be of length 2")
      for e in entry:
        import numbers
        if not isinstance(e, numbers.Number):
          raise Exception("box entries must be numbers")
      if entry[0] > entry[1]:
        raise Exception("box entries must be non decreasing")

    self._dim = len(box)
    self._box = np.array(box, dtype=np.float)
    self._ranges = self._box[:,1] - self._box[:,0]

  @property
  def dim(self): return self._dim

  @property
  def ranges(self): return self._ranges

  @property
  def box(self): return self._box

  def project(self, point):
    '''
    projects given point into the search box. 
    e.g. [-1.1, 1] with box [(-1,1),(-1,1)] gives [-1,1] 
    '''
    return np.minimum(np.maximum(point, self.box[:,0]), self.box[:,1])

  def random_point(self):
    '''
    generates a random point inside the given search box (ranges).
    '''
    # uniformly
    return self._ranges * np.random.rand(self.dim) + self._box[:,0]
    # TODO other distributions, too?

  def eval(self, point):
    raise Exception("You have to subclass and overwrite the eval function")

  def __call__(self, point):
    return self.eval(point.x)


class Result(object):
  '''
  class for one result, mapping of x to fx
  '''
  def __init__(self, point, fx):
    if point and not isinstance(point, Point): 
      raise Exception("point must be a Point")
    self._point = point
    self._fx = fx
    self._time = time.time()

  @property
  def x(self): return self.point.x

  @property
  def point(self): return self._point

  @property
  def fx(self): return self._fx

  @property
  def who(self): return self.point.who

  @property
  def error(self): return 0.0

  def __cmp__(self, other):
    # assume other instance of @Result
    return cmp(self._fx, other._fx)

  def __repr__(self):
    return '%11.6f @ f(%s)' % (self.fx, self.x)

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


class Controller(threading.Thread):
  '''
  This thread selects new points from the point generating
  threads and submits the calculations to the cluster.
  '''
  def __init__(self, problem, results, heurs, nb_gens=1):
    threading.Thread.__init__(self, name="controller")
    self.problem = problem
    self.results = results
    for h in heurs:
      if not isinstance(h, Heuristic):
        raise Exception('List must contain Heuristics')
    self.heurs = heurs
    self.rounds = 0
    self._setup_cluster(nb_gens)

  def run(self):
    '''
    This thread consumes entries from search_points
    and dispatches them to the workers in parallel.
    '''
    while stats.cnt < config.max_eval:
      while True:
        self.evaluators.spin() # check outstanding tasks
        stats.update_finished(self.evaluators.outstanding)
        if stats.pending < 10:
          break
        time.sleep(1e-3)

      new_points = []

      target = 10 #target 10 new points
      perf_sum = sum(h.perf for h in self.heurs)
      while True:
        for h in self.heurs:
          # calc probability based on perfomance with additive smoothing
          delta = .5
          prob = (h.perf + delta)/(perf_sum + delta * len(self.heurs)) 
          np_h = int(target * prob) + 1
          #logger.info("  %s -> %s" % (h, np_h))
          if self.rounds < 5:
            np_h = int(float(target) / len(self.heurs)) + 1
          new_points.extend(h.get_points(np_h))

        # TODO make this more intelligent
        if len(new_points) == 0:
          time.sleep(1e-3)
        else:
          break


      #for point in new_points:
      #  stats.add_cnt()
      #  #logger.info(" new point: %s" % p)
      #  # TODO later there is a separat thread collecting
      #  # results and adding them to the @result list
      #  # and notifying all generating threads
      #  #time.sleep(1e-3)
      #  res = Result(point, fx=self.problem(point))
      #  self.results += res

      new_tasks = self.evaluators.map_async(np.sin, [ 2 ] , chunksize = 3, ordered=False)
      stats.add_tasks(new_tasks)
      stats.update_finished(self.evaluators.outstanding)

      # discount all heuristics after each round
      for h in self.heurs:
        h.discount()

      logger.debug('  '.join(('%s:%.3f' % (h, h.perf) for h in self.heurs)))
      self.rounds += 1
    self.collector.stop = True

  def _setup_cluster(self, nb_gens):
    from IPython.parallel import Client, require
    self._c = c = Client(profile=config.ipy_profile)
    c.clear() # clears remote engines
    c.purge_results('all') # all results are memorized in the hub

    if len(c.ids) < nb_gens + 1:
      raise Exception('I need at least %d clients.' % (nb_gens + 1))
    self.generators = c.load_balanced_view(c.ids[:nb_gens])
    self.evaluators = c.load_balanced_view(c.ids[nb_gens:])

    # import some packages  (also locally)
    with c[:].sync_imports():
      from IPython.utils.timing import time
      import numpy
      import math

    self._collector = Collector(self)


class Collector(threading.Thread):
  '''
  This thread collects new results from the cluster
  and sends them in the @Results list.
  '''
  def __init__(self, controller):
    self.controller = controller
    self.stop = False
    threading.Thread.__init__(self, name='collector')
    self.start()

  def run(self):
    while not self.stop:
      for msg_id in stats.get_finished():
        # we know these are done, so don't worry about blocking
        res = self.controller.evaluators.get_result(msg_id)

        # each job returns a list of length chunksize
        for r in res.result:
          self.controller.results += r

      time.sleep(1e-3)

