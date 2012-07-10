# -*- coding: utf8 -*-
import numpy as np
from IPython.utils.timing import time

class Point(object):
  '''
  This contains the x vector for a new point and a 
  reference to who has generated it.
  '''
  def __init__(self, x, who):
    if not isinstance(who, basestring):
      raise Exception('who needs to be a string describing the heuristic')
    if not isinstance(x, np.ndarray):
      raise Exception('x must be a numpy ndarray')
    self._x   = x
    self._who = who # heuristic.name

  def __str__(self):
    return '%s by %s' % (self.x, self.who)

  @property
  def x(self): return self._x

  @property
  def who(self): return self._who


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
  def x(self): return self.point.x if self.point else None

  @property
  def point(self): return self._point

  @property
  def fx(self): return self._fx

  @property
  def who(self): return self.point.who

  @property
  def error(self):
    '''
    error margin of function evaluation, usually 0.0.
    overwrite error property in subclass if needed.
    '''
    return 0.0

  def __cmp__(self, other):
    # assume other instance of @Result
    return cmp(self._fx, other._fx)

  def __repr__(self):
    return '%11.6f @ %s' % (self.fx, self.x)

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

  #def __getstate__(self):
  #  return self._dim, self._box, self._ranges

  #def __setstate__(self, s):
  #  self._dim = s[0]
  #  self._box = s[1]
  #  self._ranges = s[2]

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

  def eval(self, x):
    raise Exception("You have to subclass and overwrite the eval function")

  def __call__(self, point):
    return Result(point, self.eval(point.x))

class Rosenbrock(Problem):
  '''
  f(x) = sum_i (100 (x_{i+1} - x_i^2)^2 + (1-x_i)^2)
  '''
  def __init__(self, dims, par1 = 100):
    box = [(-4,4)] * dims
    box[0] = (0,2) # for cornercases + testing
    self.par1 = par1
    Problem.__init__(self, box)

  def eval(self, x):
    return sum(self.par1 * (x[1:] - x[:-1]**2)**2 + (1-x[:-1])**2)

class RosenbrockAbs(Problem):
  '''
  f(x) = sum_i (100 | x_{i+1} - |x_i| | + | 1 - x_i |
  '''
  def __init__(self, dims, par1 = 100):
    box = [(-5,5)] * dims
    box[0] = (0,2) # for cornercases + testing
    self.par1 = par1
    Problem.__init__(self, box)

  def eval(self, x):
    return sum(self.par1 * np.abs(x[1:] - np.abs(x[:-1])) + \
               np.abs(1-x[:-1]))
    
class RosenbrockStochastic(Problem):
  '''
  f(x) = sum_i (100 eps_i (x_{i+1} - x_i^2)^2 + (1-x_i)^2)

  where eps_i is uniformly random in [0, 1)
  '''
  def __init__(self, dims, par1 = 100):
    box = [(-2,2)] * dims
    box[0] = (0,2) # for cornercases + testing
    self.par1 = par1
    Problem.__init__(self, box)

  def eval(self, x):
    eps = np.random.rand(self.dim - 1)
    return sum(self.par1 * eps * (x[1:] - x[:-1]**2)**2 + (1-x[:-1])**2)

class Himmelblau(Problem):
  '''
  f(x,y) = (x^2+y-11)^2 + (x+y^2-7)^2

  Lit. http://en.wikipedia.org/wiki/Himmelblau%27s_function
  '''
  def __init__(self):
    Problem.__init__(self, [(-5,5)] * 2)

  def eval(self, x):
    x, y = x[0], x[1]
    return (x**2+y-11)**2+(x+y**2-7)**2

class Rastrigin(Problem):
  '''
  f(x) = 10*n + sum_i (x_i^2 - 10 cos(2 pi x_i) )
  '''
  def __init__(self, dims, par1 = 10, offset=0):
    box = [(-2,2)] * dims
    self.offset = offset
    self.par1 = par1
    Problem.__init__(self, box)

  def eval(self, x):
    x = x - self.offset
    return self.par1 * self.dim + \
           sum(x**2 - self.par1 * np.cos(2 * np.pi * x))
