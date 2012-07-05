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

from utils import logger
from heuristics import *

# global vars
cnt = 0
MAX = 100
DIM = 5


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
    return self.eval(point)


class Result(object):
  '''
  class for one result, mapping of x to fx
  '''
  def __init__(self, x, fx):
    self._x = x
    self._fx = fx
    self._time = time.time()

  @property
  def x(self): return self._x

  @property
  def fx(self): return self._fx

  @property
  def error(self): return 0.0

  def __cmp__(self, other):
    # assume other instance of @Result
    return cmp(self._fx, other._fx)

  def __repr__(self):
    return 'f(%s) -> %s' % (self._x, self._fx)

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
    for r in results:
      heapq.heappush(self._results, r)
      if r.fx < self._best.fx: 
        self._best = r
    for l in self._listener:
      l.notify(results)

  def __iadd__(self, results):
    self.add_results(results)

  def best(self):
    return self._best

  def n_best(self, n):
    return heapq.nsmallest(n, self._results)

  def n_worst(self, n):
    return heapq.nlargest(n, self._results)


      

class Controller(threading.Thread):
  '''
  This thread selects new points from the point generating
  threads and submits the calculations to the cluster.
  '''
  def __init__(self, problem, results, rand_pts, heur_pts, calc_pts):
    threading.Thread.__init__(self, name="controller")
    self._problem = problem
    self._results = results
    self._rand_pts = rand_pts
    self._heur_pts = heur_pts
    self._calc_pts = calc_pts
    self._setup_cluster()
    self.start()

  def run(self):
    '''
    This thread consumes entries from search_points
    and dispatches them to the workers in parallel.
    '''
    global cnt, logger
    while cnt < MAX:
      new_points = []

      # add all calculated points
      new_points.extend(self._calc_pts.get_points(5))

      # heuristic points (get all)
      new_points.extend(self._heur_pts.get_points(5))

      if cnt < MAX:
        #nb_new = max(0, min(CAP - len(new_points), MAX - cnt))
        nb_new = 1
        logger.info("+++ %d" % nb_new)
        nrp = self._rand_pts.get_points(nb_new)
        new_points.extend(nrp)

      # TODO this is just a demo
      logger.info('   new points: %d' % len(new_points))
      for p in new_points: 
        cnt += 1
        logger.info(" new point: %s" % p)
        # TODO later there is a separat thread collecting
        # results and adding them to the @result list
        # and notifying all generating threads
        time.sleep(1e-3)
        import random
        r = Result(x=p, fx=random.random())
        self._results.add_results([r])

  def _setup_cluster(self):
    from IPython.parallel import Client, require
    self._c = c = Client(profile=config.ipy_profile)
    c.clear() # clears remote engines
    c.purge_results('all') # all results are memorized in the hub
    
    nbGens = 1
    if len(c.ids) < nbGens + 1:
      raise Exception('I need at least %d clients.' % (nbGens + 1))
    self.generators = c.load_balanced_view(c.ids[:nbGens])
    self.evaluators = c.load_balanced_view(c.ids[nbGens:])
  
    # import some packages  (also locally)
    with c[:].sync_imports():
      from IPython.utils.timing import time 
      import numpy
      import math

      
class Collector(threading.Thread):
  '''
  This thread collects new results from the cluster
  and sends them in the @Results list.
  '''
  def __init__(self):
    threading.Thread.__init__(self)
    
  def run(self):
    pass

