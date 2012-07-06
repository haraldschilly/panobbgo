# -*- coding: utf8 -*-
import threading
from Queue import PriorityQueue, Empty, Queue, LifoQueue
import numpy as np
from IPython.utils.timing import time
import config
from core import logger

class Point(object):
  '''
  This contains the x vector for a new point and a 
  reference to who has generated it.
  '''
  def __init__(self, x, who):
    if not isinstance(who, Heuristic):
      raise Exception('who needs to be a Heuristic')
    if not isinstance(x, np.ndarray):
      raise Exception('x must be a numpy ndarray')
    self._x   = x
    self._who = who

  def __str__(self):
    return '%s by %s' % (self.x, self.who)

  @property
  def x(self): return self._x

  @property
  def who(self): return self._who

class Heuristic(threading.Thread):
  '''
  abstract parent class for all types of point generating classes
  '''
  def __init__(self, name, problem, results, q = None, cap = None, start = True):
    self._name = name
    threading.Thread.__init__(self, name=name)
    self._problem = problem
    if results:
      self._new_results = LifoQueue()
      self._results = results
      self._results.add_listener(self)
    self._q = q if q else Queue(cap)

    # statistics
    self._perf = 1.0

    self.daemon = True
    # and start me
    if start: self.start()

  def run(self): 
    '''
    This calls the calc_points() method repeatedly. You can also overwrite
    the run() method if you like. Also note, that you can iterate over the
    self.new_results queue to be notified about new points.
    '''
    while True:
      map(self.emit, self.calc_points())

  def calc_points(self): 
    raise Exception('You have to overwrite the calc_points method.')

  def emit(self, point):
    x = self.problem.project(point)
    point = Point(x, self)
    self._q.put(point)

  def reward(self, reward):
    '''
    Give this heuristic a reward (e.g. when it finds a new point)
    '''
    self._perf += reward

  def discount(self, discount = None):
    '''
    Discount the heuristic's reward. Default is set in the configuration.
    '''
    self._perf *= discount if discount else config.discount 

  def notify(self, results):
    '''
    notify is called by @Results if there is a new @Result
    '''
    for r in sorted(results):
      self.new_results.put(r)

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

  def __str__(self):
    return '%s' % self.name

  @property
  def problem(self): return self._problem

  @property
  def results(self): return self._results

  @property
  def new_results(self): return self._new_results

  @property
  def name(self): return self._name

  @property
  def perf(self): return self._perf

class RandomPoints(Heuristic):
  '''
  always generates random points until the
  capped queue is full.
  '''
  def __init__(self, problem, results, cap = 10):
    Heuristic.__init__(self, cap=cap, name="Random", problem=problem, results=results)

  def calc_points(self):
    return [ self.problem.random_point() ]

class LatinHypercube(Heuristic):
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
    Heuristic.__init__(self, cap=cap, name="Latin Hypercube", \
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
      
class NearbyPoints(Heuristic):
  '''
  This provider generates new points based
  on a cheap (i.e. fast) algorithm. For each new result,
  it picks the so far best point (regardless of the new result)
  and generates @new many nearby point(s). 
  The @radius is scaled along each dimension's range in the search box.
  

  Arguments::
  
    - axes: 
       * one: only desturb one axis
       * all: desturb all axes
  '''
  def __init__(self, problem, results, cap = 3, radius = 1./100, new = 1, axes = 'one'):
    q = LifoQueue(cap)
    self.radius = radius
    self.new    = new
    self.axes   = axes
    Heuristic.__init__(self, q = q, cap=cap, name="Nearby %.3f/%s" % (radius, axes),\
                           problem=problem, results=results)

  def calc_points(self):
    ret = []
    while self.new_results.qsize() > 0:
      _ = self.new_results.get() # one for each new result
      best = self.results.best
      x = best.x
      # generate @new many new points near best x 
      for _ in range(self.new): 
        if self.axes == 'all':
          dx = ( np.random.rand(self.problem.dim) - .5 ) * self.radius
          dx *= self.problem.ranges
          new_x = x + dx
        elif self.axes == 'one':
          idx = np.random.randint(self.problem.dim)
          dx = (np.random.rand() - .5) * self.radius
          dx *= self.problem.ranges[idx]
          new_x = x.copy()
          new_x[idx] += dx
        else:
          raise Exception("axis parameter not 'one' or 'all'")
        ret.append(new_x)
    return ret

class ExtremalPoints(Heuristic):
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
    Heuristic.__init__(self, cap=cap, name="Extremal",\
                           problem=problem, results=results)

  def calc_points(self):
    m = self.problem.box[:,0].copy()
    for e in self.steps:
      m += e[np.random.randint(0, self.l)]
    return m

class ZeroPoint(Heuristic):
  '''
  This heuristic only returns the 0 vector once.
  '''
  def __init__(self, problem):
    Heuristic.__init__(self, name="Zero", cap=1, problem=problem, results=None)
  
  def run(self):
    self.emit(np.zeros(self.problem.dim))

class CalculatedPoints(Heuristic):
  '''
  This is the thread that generates points by
  dispatching tasks. -- NYI
  '''
  def __init__(self, problem, results, cap = 10):
    self.machines = None
    Heuristic.__init__(self, cap=cap, name="Calculated",\
                  problem=problem, results=results, start=False)

  def set_machines(self, machines):
    self.machines = machines # this is already a load_balanced view
    

  def calc_points(self):
    #return np.array([99]*self._problem.dim)
    pass


