# -*- coding: utf8 -*-
import threading
from Queue import Empty, Queue, LifoQueue # PriorityQueue
import config
from config import loggers
logger = loggers['heuristic']
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
  def register_heuristics(cls, heurs, problem, results):
    '''
    Call it with a list of Heuristic-instances before starting the Strategy.
    '''
    for h in sorted(heurs, key = lambda h : h.name):
      name = h.name
      assert name not in cls.lookup, 'Names of heuristics need to be unique. "%s" is already used' % name
      cls.lookup[name] = h
      h._problem = problem
      if h._listen_results:
        h._results = results
        h.new_results = LifoQueue()
        results.add_listener(h)
      h._init_()
      if h._start: h.start()

  @classmethod
  def heuristics(self):
    '''
    Get active (thread is alive) heuristics.
    '''
    return filter(lambda h:h.isAlive(), Heuristic.lookup.values())

  def __init__(self, name = None, q = None, cap = None, start = True):
    name = name if name else self.__class__.__name__
    threading.Thread.__init__(self, name=name)
    # daemonize and start me
    self.daemon = True
    self._start = start
    self._name = name
    self._problem = None
    self._listen_results = False
    self._q = q if q else Queue(cap)

    # statistics; performance
    self._performance = 0.0

  def _init_(self):
    '''
    2nd initialization, after registering and hooking up the heuristic.
    e.g. self._problem is available.
    '''
    pass

  def run(self):
    '''
    This calls the calc_points() method repeatedly.
    Stop executing the heuristic by raising a StopHeuristic exception.

    Don't overwrite this run method.

    Also note, that you can set _listen_results to True and 
    iterate over the new_results queue to be notified about new points.
    '''
    try:
      while True:
        pnts = self.calc_points()
        if pnts == None: raise StopHeuristic()
        if not isinstance(pnts, list): pnts = [ pnts ]
        if len(pnts) == 0:
          #logger.warning("empty list of points, heuristic '%s' shouldn't do this" % self.name)
          from IPython.utils.timing import time
          time.sleep(1e-3)
        else:
          map(self.emit, pnts)
    except StopHeuristic:
      logger.info("'%s' heuristic stopped." % self.name)

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

  def reward(self, reward):
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

  def __repr__(self):
    return '%s' % self.name

  @property
  def problem(self): return self._problem

  @property
  def results(self): return self._results

  @property
  def name(self): return self._name

  @property
  def performance(self): return self._performance

class Random(Heuristic):
  '''
  always generates random points until the
  capped queue is full.
  '''
  def __init__(self, cap = 10, name=None):
    name = "Random" if name is None else name
    Heuristic.__init__(self, cap=cap, name=name)

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
  def __init__(self, div, cap = 10):
    Heuristic.__init__(self, cap=cap, name="Latin Hypercube")
    if not isinstance(div, int):
      raise Exception("LH: div needs to be an integer")
    self.div = div

  def _init_(self):
    # length of each box'es dimension
    self.lengths = self.problem.ranges / float(self.div)

  def calc_points(self):
    import numpy as np
    div = self.div
    dim = self.problem.dim
    pts = np.repeat(np.arange(div, dtype=np.float), dim).reshape(div,dim)
    pts += np.random.rand(div, dim) # add [0,1) jitter
    pts *= self.lengths             # scale with length, already divided by div
    pts += self.problem.box[:,0]    # shift with min
    [ np.random.shuffle(pts[:,i]) for i in range(dim) ]
    return [ p for p in pts ] # needs to be a list of np.ndarrays

class Nearby(Heuristic):
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
  def __init__(self, cap = 3, radius = 1./100, new = 1, axes = 'one'):
    q = LifoQueue(cap)
    Heuristic.__init__(self, q = q, cap=cap, name="Nearby %.3f/%s" % (radius, axes))
    self.radius = radius
    self.new    = new
    self.axes   = axes
    self._listen_results = True

  def calc_points(self):
    import numpy as np
    ret = []
    block = True # block for first item, then until queue empty
    while True:
      try:
        _ = self.new_results.get(block = block) # one loop for each new result
      except Empty:
        break
      block = False
      best = self.results.best
      x = best.x
      # generate self.new many new points near best x
      for _ in range(self.new): 
        new_x = x.copy()
        if self.axes == 'all':
          dx = (2.0 * np.random.rand(self.problem.dim) - 1.0) * self.radius
          dx *= self.problem.ranges
          new_x += dx
        elif self.axes == 'one':
          idx = np.random.randint(self.problem.dim)
          dx = (2.0 * np.random.rand() - 1.0) * self.radius
          dx *= self.problem.ranges[idx]
          new_x[idx] += dx
        else:
          raise Exception("axis parameter not 'one' or 'all'")
        ret.append(new_x)
    return ret

class Extremal(Heuristic):
  '''
  This heuristic is specifically seeking for points at the
  border of the box and around 0.
  The @where parameter takes a list or tuple, which has values 
  from 0 to 1, which indicate the probability for sampling from the
  minimum, zero, center and the maximum. default = ( 1, .2, .2, 1 )
  '''
  def __init__(self, cap = 10, diameter = 1./10, prob = None):
    Heuristic.__init__(self, cap=cap, name="Extremal")
    import numpy as np
    if prob is None: prob = (1, .2, .2, 1)
    for i in prob:
      if i < 0 or i > 1:
        raise Exception("entries in prob must be in [0, 1]")
    prob =  np.array(prob) / float(sum(prob))
    self.prob = prob.cumsum()
    self.diameter = diameter # inside the box or around zero

  def _init_(self):
    import numpy as np
    problem = self.problem
    low  = problem.box[:,0]
    high = problem.box[:,1]
    zero = np.zeros(problem.dim)
    center = low + (high-low) / 2.0
    self.vals = np.row_stack((low, zero, center, high))

  def calc_points(self):
    import numpy as np
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

class Zero(Heuristic):
  '''
  This heuristic only returns the 0 vector once.
  '''
  def __init__(self):
    Heuristic.__init__(self, name="Zero", cap=1)
    self.done = False

  def calc_points(self):
    import numpy as np
    if not self.done:
      self.done = True
      return np.zeros(self.problem.dim)
    else:
      raise StopHeuristic()

class Center(Heuristic):
  '''
  This heuristic checks the point in the center of the box.
  '''
  def __init__(self):
    Heuristic.__init__(self, name="Center", cap=1)
    self.done = False

  def calc_points(self):
    if not self.done:
      self.done = True
      box = self.problem.box
      return box[:,0] + (box[:,1]-box[:,0]) / 2.0
    else:
      raise StopHeuristic()

class QadraticModelMockup(Heuristic):
  '''
  '''
  def __init__(self, cap = 10):
    Heuristic.__init__(self, cap=cap) #, start=False)
    self._listen_results = True
    self.machines = None

  #def set_machines(self, machines):
  #  self.machines = machines # this is already a load_balanced view

  def calc_points(self):
    logger.warning("%s is broken, don't use it" % self.name)
    return None

    best = self.results.best
    if best is None or best.x is None: return []
    nbrs = self.results.in_same_grid(best)
    if len(nbrs) < 3: return []

    # actual calculation
    import numpy as np
    from scipy.optimize import fmin_bfgs

    N = self.problem.dim
    def approx(x, *params):
      a, b, c = params
      return a * x.dot(x) + x.dot(np.repeat(b, N)) + c

    xx = np.array([r.x  for r in nbrs])
    yy = np.array([r.fx for r in nbrs])

    def residual(params):
      fx = np.array([approx(x, *params) - y for x,y in zip(xx,yy)])
      return fx.dot(fx)

    def gradient(params):
      a, b, c = params
      ret = np.empty(3)
      v1 = 2. * np.array([approx(x, *params) - y for x,y in zip(xx,yy)])
      ret[0] = v1.dot(xx.dot(xx.T).diagonal())
      ret[1] = v1.dot(xx.sum(axis=1))
      ret[2] = v1.sum()
      return ret

    params = np.random.normal(0,1,size=3)
    sol = fmin_bfgs(residual, params, fprime=gradient)
    logger.info("params: %s %s %s" % (sol[0], sol[1], sol[2]))

    return []

class WeightedAverage(Heuristic):
  '''
  '''
  def __init__(self, cap = 10, k = 2.):
    Heuristic.__init__(self, cap=cap) #, start=False)
    self._listen_results = True
    self.k = k

  def calc_points(self):
    pass

  def on_new_best(self, best):
    if best is None or best.x is None: return []
    #nbrs = self.results.in_same_grid(best)
    nbrs = self.results.n_best(4)
    if len(nbrs) < 3: return

    # actual calculation
    import numpy as np
    xx = np.array([r.x   for r in nbrs])
    yy = np.array([r.fx  for r in nbrs])
    #weights = np.abs(best.fx - yy)
    #weights = -weights + self.k * weights.max()
    weights = np.log1p(np.arange(len(yy) + 1, 1, -1))
    #logger.info("weights: %s" % weights)
    for i in range(10):
      ret = np.average(xx, axis=0, weights=weights)
      ret += 1 * np.random.normal(0, xx.std(axis=0))
      if np.linalg.norm(best.x - ret) > .01:
        self.emit(ret)

class Testing(Heuristic):
  '''
  '''
  def __init__(self, cap = 10):
    Heuristic.__init__(self, cap=cap) #, start=False)
    self.i = 0
    self.j = 0

  def calc_points(self):
    raise StopHeuristic()

  def on_new_best(self, best):
    #logger.info("TEST best: %s" % best)
    self.i += 1
    import numpy as np
    p = np.random.normal(size=self.problem.dim)
    self.emit(p)
    if self.i > 2:
      #logger.info('TEST best i = %s'%self.i)
      raise StopHeuristic()

  def on_new_result(self, r):
    #logger.info("TEST results: %s" % r)
    self.j += 1
    import numpy as np
    p = np.random.normal(size=self.problem.dim)
    self.emit(p)
    #if self.j > 5:
    #  raise StopHeuristic()

