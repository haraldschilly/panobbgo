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

# global vars
cnt = 0
MAX = 100
CAP = 10
DIM = 5

def gen_random_point():
  return np.random.rand(DIM)


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

  def best(self):
    return self._best

  def n_best(self, n):
    return heapq.nsmallest(n, self._results)

  def n_worst(self, n):
    return heapq.nlargest(n, self._results)

class PointProvider(threading.Thread):
  '''
  abstract parent class for all types of point generating classes
  '''
  def __init__(self, name, results, q = None, cap = CAP, start = True):
    threading.Thread.__init__(self, name=name)
    self._results = results
    self._results.add_listener(self)
    self._q = q if q else Queue(cap)
    self._r = Queue()
    self.daemon = True
    # and start me
    if start: self.start()

  def run(self): raise Exception('NYI')

  def notify(self, results):
    '''
    notify is called by @Results if there is a new @Result
    '''
    for r in sorted(results):
      self._r.put(r)

  def get_points(self, limit=None):
    '''
    this drains the self._q Queue until @limit
    elements are removed or the Queue is empty.
    '''
    new_points = []
    try:
      while not limit or len(new_points) < limit:
        new_points.append(self._q.get(block=False))
    except Empty:
      pass
    return new_points

class RandomPoints(PointProvider):
  '''
  always generates random points until the
  capped queue is full.
  '''
  def __init__(self, results, cap = 10):
    PointProvider.__init__(self, cap=cap, name="random", results=results)

  def run(self):
    while True:
      self._q.put(gen_random_point())

class HeuristicPoints(PointProvider):
  '''
  This provider generates new points based
  on a cheap (i.e. fast) algorithm.
  '''
  def __init__(self, results, cap = 3):
    PointProvider.__init__(self, cap=cap, name="heuristic", results=results)

  def run(self):
    while True:
      if self._r:
        _ = self._r.get()
        best = self._results.best()
        x = best.x
        # generate new points near best x 
        for _ in range(1): 
          dx = ( np.random.rand(len(x)) - .5 ) / 20.0
          x_new = x + dx
          self._q.put(x_new)
      else:
        time.sleep(1e-3)


class CalculatedPoints(PointProvider):
  '''
  This is the thread that generates points by
  dispatching tasks. -- NYI
  '''
  def __init__(self, results, cap = 10):
    PointProvider.__init__(self, cap=cap, name="calculated", results=results)

  def run(self):
    while True:
      # TODO see if there are new calculated points
      # and then add them to queue
      self._q.put([99]*DIM)
      time.sleep(1e-1)



      

class Controller(threading.Thread):
  '''
  This thread selects new points from the point generating
  threads and submits the calculations to the cluster.
  '''
  def __init__(self, results, rand_pts, heur_pts, calc_pts):
    threading.Thread.__init__(self, name="controller")
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
      new_points.extend(self._calc_pts.get_points())

      # heuristic points (get all)
      new_points.extend(self._heur_pts.get_points())

      if len(new_points) < CAP and cnt < MAX:
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



if __name__=="__main__":
  # spawning threads
  #calc_points_thread   = threading.Thread(target=calculated_points, name='calc_points')
  #calc_points_thread.daemon   = True

  results = Results()

  rand_pts = RandomPoints(results)
  heur_pts = HeuristicPoints(results)
  calc_pts = CalculatedPoints(results)

  controller = Controller(results, rand_pts,heur_pts,calc_pts)

  # keep main thread alive until all created points are also consumed 
  # and processed by the evaluator_thread
  controller.join()
  #print "remaining in search_points_q: %s [ == 0 ?]" % search_points_q.unfinished_tasks
  print "cnt: %d" % cnt
