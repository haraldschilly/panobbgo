# -*- coding: utf8 -*-
import config
logger = config.get_logger('HEUR')
from core import Heuristic, StopHeuristic
import numpy as np

class Random(Heuristic):
  '''
  always generates random points inside the box of the
  "best leaf" (see "Splitter") until the capped queue is full.
  '''
  def __init__(self, cap = None, name=None):
    name = "Random" if name is None else name
    self.leaf = None
    from threading import Event
    self.have_split = Event()
    Heuristic.__init__(self, name=name)

  def on_start(self):
    self.have_split.wait()
    splitter = self.strategy.analyzer("splitter")
    while True:
      r = self.leaf.ranges * np.random.rand(splitter.dim) + self.leaf.box[:,0]
      self.emit(r)

  def on_new_split(self, box, children, dim):
    '''
    we are only interested in the (possibly new)
    leaf around the best point
    '''
    best = self.strategy.analyzer("best").best
    self.leaf = self.strategy.analyzer("splitter").get_leaf(best)
    self.clear_queue()
    self.have_split.set()


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
  def __init__(self, div):
    cap = div
    Heuristic.__init__(self, cap=cap, name="Latin Hypercube")
    if not isinstance(div, int):
      raise Exception("LH: div needs to be an integer")
    self.div = div

  def _init_(self):
    # length of each box'es dimension
    self.lengths = self.problem.ranges / float(self.div)

  def on_start(self):
    import numpy as np
    div = self.div
    dim = self.problem.dim
    while True:
      pts = np.repeat(np.arange(div, dtype=np.float), dim).reshape(div,dim)
      pts += np.random.rand(div, dim) # add [0,1) jitter
      pts *= self.lengths             # scale with length, already divided by div
      pts += self.problem.box[:,0]    # shift with min
      [ np.random.shuffle(pts[:,i]) for i in range(dim) ]
      self.emit([ p for p in pts ]) # needs to be a list of np.ndarrays

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
    Heuristic.__init__(self, cap=cap, name="Nearby %.3f/%s" % (radius, axes))
    self.radius = radius
    self.new    = new
    self.axes   = axes

  def on_new_result(self, result):
    import numpy as np
    ret = []
    x = self.strategy.analyzer('best').best.x
    if x is None: return
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
    self.emit(ret)

class Extremal(Heuristic):
  '''
  This heuristic is specifically seeking for points at the
  border of the box and around 0.
  The @where parameter takes a list or tuple, which has values 
  from 0 to 1, which indicate the probability for sampling from the
  minimum, zero, center and the maximum. default = ( 1, .2, .2, 1 )
  '''
  def __init__(self, diameter = 1./10, prob = None):
    Heuristic.__init__(self, name="Extremal")
    import numpy as np
    if prob is None: prob = (1, .2, .2, 1)
    for i in prob:
      if i < 0 or i > 1:
        raise Exception("entries in prob must be in [0, 1]")
    prob =  np.array(prob) / float(sum(prob))
    self.probabilities = prob.cumsum()
    self.diameter = diameter # inside the box or around zero

  def _init_(self):
    import numpy as np
    problem = self.problem
    low  = problem.box[:,0]
    high = problem.box[:,1]
    zero = np.zeros(problem.dim)
    center = low + (high-low) / 2.0
    self.vals = np.row_stack((low, zero, center, high))

  def on_start(self):
    import numpy as np
    while True:
      ret = np.empty(self.problem.dim)
      for i in range(self.problem.dim):
        r = np.random.rand()
        for idx, val in enumerate(self.probabilities):
          if val > r:
            radius = self.problem.ranges[i] * self.diameter
            jitter = radius * (np.random.rand() - .5)
            shift = 0.0
            if idx == 0:
              shift = 1
            elif idx == len(self.probabilities) - 1:
              shift = -1
            else:
              shift = 0
            ret[i] = self.vals[idx, i] + shift * radius * .5 + jitter
            break
      self.emit(ret)

class Zero(Heuristic):
  '''
  This heuristic only returns the 0 vector once.
  '''
  def __init__(self):
    Heuristic.__init__(self, name="Zero", cap=1)

  def on_start(self):
    from numpy import zeros
    return zeros(self.problem.dim)

class Center(Heuristic):
  '''
  This heuristic checks the point in the center of the box.
  '''
  def __init__(self):
    Heuristic.__init__(self, name="Center", cap=1)

  def on_start(self):
    box = self.problem.box
    return box[:,0] + (box[:,1]-box[:,0]) / 2.0

class QuadraticModelMockup(Heuristic):
  '''
  '''
  def __init__(self):
    Heuristic.__init__(self)
    self.machines = None

  #def set_machines(self, machines):
  #  self.machines = machines # this is already a load_balanced view

  def on_start(self):
    self.eventbus.publish('calc_quadratic_model')

  def on_calc_quadratic_model(self):
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
    #self.emit(...)
    self.eventbus.publish('calc_quadratic_model')

class WeightedAverage(Heuristic):
  '''
  '''
  def __init__(self, k = .1):
    Heuristic.__init__(self)
    self.k = k

  def _init_(self):
    self.minstd = min(self.problem.ranges) / 100.

  def on_new_best(self, best):
    if best is None or best.x is None: return
    nbrs = self.strategy.analyzer('splitter').in_same_leaf(best)
    #nbrs = self.results.n_best(4)
    if len(nbrs) < 3: return
    #logger.info("WA: %s" % len(nbrs))

    # actual calculation
    import numpy as np
    xx = np.array([r.x   for r in nbrs])
    yy = np.array([r.fx  for r in nbrs])
    weights = np.log1p(yy - best.fx)
    weights = -weights + (1+self.k) * weights.max()
    #weights = np.log1p(np.arange(len(yy) + 1, 1, -1))
    #logger.info("weights: %s" % zip(weights, yy))
    for i in range(self.cap - self._q.qsize()):
      ret = np.average(xx, axis=0, weights=weights)
      std = xx.std(axis=0)
      # std must be > 0
      std[std < self.minstd] = self.minstd
      ret += 1 * np.random.normal(0, std)
      if np.linalg.norm(best.x - ret) > .01:
        self.emit(ret)

class Testing(Heuristic):
  '''
  just to try some ideas ...
  '''
  def __init__(self):
    Heuristic.__init__(self)
    self.i = 0
    self.j = 0

  def on_start(self):
    self.eventbus.publish('calling_testing')

  def on_calling_testing(self):
    #self.eventbus.publish('calling_testing')
    pass

  def on_new_best(self, best):
    #logger.info("TEST best: %s" % best)
    self.i += 1
    import numpy as np
    p = np.random.normal(size=self.problem.dim)
    self.emit(p)
    if self.i > 2:
      #logger.info('TEST best i = %s'%self.i)
      raise StopHeuristic()

  def on_new_result(self, result):
    #logger.info("TEST results: %s" % r)
    self.j += 1
    import numpy as np
    p = np.random.normal(size=self.problem.dim)
    self.emit(p)
    #if self.j > 5:
    #  raise StopHeuristic()

