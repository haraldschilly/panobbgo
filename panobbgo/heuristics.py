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

  def _add(self, point):
    point = self._problem.project(point)
    self._q.put(point)

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
      while limit is None or len(new_points) < limit:
        new_points.append(self._q.get(block=False))
    except Empty:
      pass
    return new_points

  @property
  def problem(self): return self._problem

class RandomPoints(PointProvider):
  '''
  always generates random points until the
  capped queue is full.
  '''
  def __init__(self, problem, results, cap = 10):
    PointProvider.__init__(self, cap=cap, name="random", problem=problem, results=results)

  def run(self):
    while True:
      self._add(self.problem.random_point())

class LatinHypercube(PointProvider):
  '''
  partitions the search box into n x n x ... x n cubes.
  selects randomly in such a way, that there is only one cube in each dimension.
  then randomly selects one point from inside such a cube.


  e.g. with n=4:

  +---+---+---+---+
  | X |   |   |   |
  +---+---+---+---+
  |   |   |   | X |
  +---+---+---+---+
  |   | X |   |   |
  +---+---+---+---+
  |   |   | X |   |
  +---+---+---+---+
  '''
  def __init__(self, problem, results, division, cap = 10):
    self.division = division
    # length of each box'es dimension
    self.lengths = problem.ranges / float(division)
    PointProvider.__init__(self, cap=cap, name="latin hypercube", problem=problem, results=results)
  
  def run(self):
    dim = self.problem.dim
    divs = self.division
    while True:
      pts = np.empty((divs, dim))
      for i in range(dim): 
        pts[:,i] = (np.arange(divs) + np.random.rand(divs)) * self.lengths[i]
        np.random.shuffle(pts[:,i])
        pts[:,i] += np.repeat(self.problem.box[i,0], divs)  #add min

      for p in pts: self._add(p)
      

class HeuristicPoints(PointProvider):
  '''
  This provider generates new points based
  on a cheap (i.e. fast) algorithm.
  '''
  def __init__(self, problem, results, cap = 3):
    q = LifoQueue(cap)
    PointProvider.__init__(self, q = q, cap=cap, name="heuristic", problem=problem, results=results)

  def run(self):
    while True and self._r:
        _ = self._r.get() # one for each new result
        best = self._results.best()
        x = best.x
        # generate new points near best x 
        for _ in range(1): 
          dx = ( np.random.rand(len(x)) - .5 ) / 20.0
          x_new = x + dx
          self._add(x_new)


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
      p = np.array([99]*self._problem.dim)
      self._add(p)


