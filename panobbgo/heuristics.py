# -*- coding: utf8 -*-
import threading
from Queue import PriorityQueue, Empty, Queue, LifoQueue
import numpy as np
from IPython.utils.timing import time

class PointProvider(threading.Thread):
  '''
  abstract parent class for all types of point generating classes
  '''
  def __init__(self, name, problem, results, q = None, cap = None, start = True):
    threading.Thread.__init__(self, name=name)
    self._problem = problem
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
  def __init__(self, problem, results, cap = 10):
    PointProvider.__init__(self, cap=cap, name="random", problem=problem, results=results)

  def run(self):
    while True:
      self._q.put(self._problem.random_point())

class HeuristicPoints(PointProvider):
  '''
  This provider generates new points based
  on a cheap (i.e. fast) algorithm.
  '''
  def __init__(self, problem, results, cap = 3):
    PointProvider.__init__(self, cap=cap, name="heuristic", problem=problem, results=results)

  def run(self):
    while True and self._r:
        _ = self._r.get() # one for each new result
        best = self._results.best()
        x = best.x
        # generate new points near best x 
        for _ in range(1): 
          dx = ( np.random.rand(len(x)) - .5 ) / 20.0
          x_new = x + dx
          self._q.put(x_new)


class CalculatedPoints(PointProvider):
  '''
  This is the thread that generates points by
  dispatching tasks. -- NYI
  '''
  def __init__(self, problem, results, cap = 10):
    PointProvider.__init__(self, cap=cap, name="calculated", problem=problem, results=results)

  def run(self):
    while True:
      # TODO see if there are new calculated points
      # and then add them to queue
      self._q.put([99]*self._problem.dim)


