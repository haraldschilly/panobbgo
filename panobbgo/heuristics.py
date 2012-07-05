# -*- coding: utf8 -*-
import threading
from Queue import PriorityQueue, Empty, Queue, LifoQueue
import numpy as np
from IPython.utils.timing import time
from core import logger

class PointProvider(threading.Thread):
  '''
  abstract parent class for all types of point generating classes
  '''
  def __init__(self, name, problem, results, q = None, cap = None, start = True):
    threading.Thread.__init__(self, name=name)
    self._problem = problem
    if results:
      self._results = results
      self._results.add_listener(self)
    self._q = q if q else Queue(cap)
    self._r = Queue()
    self.daemon = True
    # and start me
    if start: self.start()

  def run(self): 
    while True:
      map(self._add, self.calc_points())

  def calc_points(self): 
    raise Exception('You have to overwrite the calc_points method.')

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

  @property
  def results(self): return self._results

class RandomPoints(PointProvider):
  '''
  always generates random points until the
  capped queue is full.
  '''
  def __init__(self, problem, results, cap = 10):
    PointProvider.__init__(self, cap=cap, name="random", problem=problem, results=results)

  def calc_points(self):
    return [ self.problem.random_point() ]

class LatinHypercube(PointProvider):
  '''
  partitions the search box into n x n x ... x n cubes.
  selects randomly in such a way, that there is only one cube in each dimension.
  then randomly selects one point from inside such a cube.


  e.g. with div=4 and dim=2:

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
  def __init__(self, problem, results, div, cap = 10):
    if not isinstance(div, int):
      raise Exception("LH: div needs to be an integer")
    self.div = div
    # length of each box'es dimension
    self.lengths = problem.ranges / float(div)
    PointProvider.__init__(self, cap=cap, name="latin hypercube", \
                           problem=problem, results=results)
 
  def calc_points(self):
    div = self.div
    dim = self.problem.dim
    pts = np.repeat(np.arange(div, dtype=np.float), dim).reshape(div,dim)
    pts += np.random.rand(div, dim) # add [0,1) jitter
    pts *= self.lengths             # scale with length, already divided by div
    pts += self.problem.box[:,0]    # shift with min
    [ np.random.shuffle(pts[:,i]) for i in range(dim) ]
    return pts
      
class HeuristicPoints(PointProvider):
  '''
  This provider generates new points based
  on a cheap (i.e. fast) algorithm.
  '''
  def __init__(self, problem, results, cap = 3, radius = 1./100):
    q = LifoQueue(cap)
    self.radius = radius
    PointProvider.__init__(self, q = q, cap=cap, name="heuristic",\
                           problem=problem, results=results)

  def calc_points(self):
    ret = []
    while self._r.qsize() > 0:
      _ = self._r.get() # one for each new result
      best = self.results.best()
      x = best.x
      # generate new points near best x 
      for _ in range(2): 
        dx = ( np.random.rand(self.problem.dim) - .5 ) / self.radius
        dx *= self.problem.ranges
        ret.append(x + dx)
    return ret

class ExtremalPoints(PointProvider):
  '''
  This heuristic is specially seeking for points at the
  border of the box and around 0. 
  The @where parameter takes a list or tuple, which has values 
  from 0 to 1, which indicate the probability for sampling from the
  minimum, zero or the maximum. default = ( 1, .2, 1 )
  '''
  def __init__(self, problem, results, cap = 10):
    if where is None: where = (1, .2, 1)
    self.l = len(where)
    for i in where:
      if i < 0 or i > 1:
        raise Exception("entries in where must be in [0, 1]")
    where =  np.array(where) / float(max(where))
    PointProvider.__init__(self, cap=cap, name="Extremal",\
                           problem=problem, results=results)

  def calc_points(self):
    m = self.problem.box[:,0].copy()
    for e in self.steps:
      m += e[np.random.randint(0, self.l)]
    return m

class ZeroPoint(PointProvider):
  '''
  This heuristic only returns the 0 vector once.
  '''
  def __init__(self, problem):
    PointProvider.__init__(self, name="zero", cap=1, problem=problem, results=None)
  
  def run(self):
    self._add(np.zeros(self.problem.dim))

class CalculatedPoints(PointProvider):
  '''
  This is the thread that generates points by
  dispatching tasks. -- NYI
  '''
  def __init__(self, problem, results, cap = 10):
    self.machines = None
    PointProvider.__init__(self, cap=cap, name="calculated",\
                  problem=problem, results=results, start=False)

  def set_machines(self, machines):
    self.machines = machines # this is already a load_balanced view
    

  def calc_points(self):
    return np.array([99]*self._problem.dim)


