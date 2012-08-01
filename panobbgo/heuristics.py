# -*- coding: utf8 -*-
import threading
from Queue import PriorityQueue, Empty, Queue, LifoQueue
import numpy as np
#from IPython.utils.timing import time
import config
from core import logger
from panobbgo_problems import Point

class StopHeuristic(Exception):
  '''
  Used to indicate, that the heuristic has finished and should be ignored.
  '''
  pass

class Heuristic(threading.Thread):
  '''
  abstract parent class for all types of point generating classes
  '''
  # global mapping of heuristic names to their instances.
  # names must be unique!
  import collections
  lookup = collections.OrderedDict()

  #@classmethod
  #def normalize_performances(cls):
  #  perf_sum = sum(h.performance for h in cls.heuristics())
  #  for v in cls.lookup.values():
  #    v._performance /= perf_sum

  @classmethod
  def register_heuristics(cls, heurs):
    '''
    Call it with a list of Heuristic-instances before starting the Strategy.
    '''
    for h in sorted(heurs, key = lambda h : h.name):
      name = h.name
      assert name not in cls.lookup, 'Names of heuristics need to be unique. "%s" is already used' % name
      cls.lookup[name] = h

  @classmethod
  def heuristics(self):
    '''
    Get active (thread is alive) heuristics.
    '''
    return filter(lambda h:h.isAlive(), Heuristic.lookup.values())

  def __init__(self, name, problem, results, q = None, cap = None, start = True):
    self._name = name
    self._problem = problem
    if results:
      self._new_results = LifoQueue()
      self._results = results
      self._results.add_listener(self)
    self._q = q if q else Queue(cap)

    # statistics; performance
    self._performance = 0.0

    threading.Thread.__init__(self, name=name)
    # daemonize and start me
    self.daemon = True
    if start: self.start()

  def run(self):
    '''
    This calls the calc_points() method repeatedly.
    Stop executing the heuristic by raising a StopHeuristic exception.

    Don't overwrite this run method.

    Also note, that you can iterate over the
    self.new_results queue to be notified about new points.
    '''
    try:
      while True:
        map(self.emit, self.calc_points())
    except StopHeuristic:
      pass

  def calc_points(self): 
    raise Exception('You have to overwrite the calc_points method.')

  def emit(self, point):
    '''
    this is used in the heuristic's thread. 
    '''
    x = self.problem.project(point)
    point = Point(x, self.name)
    self.discount()
    self._q.put(point)

  def reward(self, reward = 1):
    '''
    Give this heuristic a reward (e.g. when it finds a new point)
    '''
    logger.debug("Reward of %s for '%s'" % (reward, self.name))
    self._performance += reward

  def discount(self, discount = config.discount):
    '''
    Discount the heuristic's reward. Default is set in the configuration.
    '''
    self._performance *= discount

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
    for each actually emitted point,
    the performance value is discounted (i.e. "punishment" or "energy
    consumption")
    '''
    new_points = []
    try:
      while limit is None or len(new_points) < limit:
        new_points.append(self._q.get(block=False))
        self.discount()
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
  def performance(self): return self._performance

class RandomPoints(Heuristic):
  '''
  always generates random points until the
  capped queue is full.
  '''
  def __init__(self, problem, results, cap = 10, name=None):
    name = "Random" if name is None else name
    Heuristic.__init__(self, cap=cap, name=name, problem=problem, results=results)

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
    Heuristic.__init__(self, q = q, cap=cap, name="Nearby %.3f/%s" % (radius, axes),
                           problem=problem, results=results)

  def calc_points(self):
    ret = []
    while self.new_results.qsize() > 0:
      _ = self.new_results.get() # one for each new result
      best = self.results.best
      x = best.x
      # generate self.new many new points near best x
      for _ in range(self.new): 
        if self.axes == 'all':
          dx = (2.0 * np.random.rand(self.problem.dim) - 1.0) * self.radius
          dx *= self.problem.ranges
          new_x = x + dx
        elif self.axes == 'one':
          idx = np.random.randint(self.problem.dim)
          dx = (2.0 * np.random.rand() - 1.0) * self.radius
          dx *= self.problem.ranges[idx]
          new_x = x.copy()
          new_x[idx] += dx
        else:
          raise Exception("axis parameter not 'one' or 'all'")
        ret.append(new_x)
    return ret

class ExtremalPoints(Heuristic):
  '''
  This heuristic is specifically seeking for points at the
  border of the box and around 0.
  The @where parameter takes a list or tuple, which has values 
  from 0 to 1, which indicate the probability for sampling from the
  minimum, zero, center and the maximum. default = ( 1, .2, .2, 1 )
  '''
  def __init__(self, problem, cap = 10, diameter = 1./10, prob = None):
    if prob is None: prob = (1, .2, .2, 1)
    for i in prob:
      if i < 0 or i > 1:
        raise Exception("entries in prob must be in [0, 1]")
    prob =  np.array(prob) / float(sum(prob))
    self.prob = prob.cumsum()
    self.diameter = diameter # inside the box or around zero
    low  = problem.box[:,0]
    high = problem.box[:,1]
    zero = np.zeros(problem.dim)
    center = low + (high-low) / 2.0
    self.vals = np.row_stack((low, zero, center, high))

    Heuristic.__init__(self, cap=cap, name="Extremal",\
                           problem=problem, results=None)

  def calc_points(self):
    ret = np.empty(self.problem.dim)
    for i in range(self.problem.dim):
      r = np.random.rand()
      for idx, val in enumerate(self.prob):
        if val > r:
          radius = self.problem.ranges[i] * self.diameter
          jitter = radius * (np.random.rand() - .5)
          shift = 0.0
          if idx == 0: shift = 1
          elif idx == len(self.prob) - 1: shift = -1
          else: shift = 0
          ret[i] = self.vals[idx, i] + shift * radius * .5 + jitter
          break

    return ret

class ZeroPoint(Heuristic):
  '''
  This heuristic only returns the 0 vector once.
  '''
  def __init__(self, problem):
    self.done = False
    Heuristic.__init__(self, name="Zero", cap=1, problem=problem, results=None)

  def calc_points(self):
    if not self.done:
      self.done = True
      return np.zeros(self.problem.dim)
    else:
      raise StopHeuristic()

class CenterPoint(Heuristic):
  '''
  This heuristic checks the point in the center of the box.
  '''
  def __init__(self, problem):
    self.done = False
    Heuristic.__init__(self, name="Center", cap=1, problem=problem, results=None)

  def calc_points(self):
    if not self.done:
      self.done = True
      box = self.problem.box
      return box[:,0] + (box[:,1]-box[:,0]) / 2.0
    else:
      raise StopHeuristic()

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
    return []


