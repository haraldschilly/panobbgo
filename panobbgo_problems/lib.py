# -*- coding: utf8 -*-
'''
This file contains the basic objects to build a problem and to do a single evaluation.
'''

# ATTN: make sure, that this doesn't depend on the config or threading modules.
#       the serialization and reconstruction won't work!
import numpy as np
from IPython.utils.timing import time

class Point(object):
  '''
  This contains the x vector for a new point and a 
  reference to who has generated it.
  '''
  def __init__(self, x, who):
    if not isinstance(who, basestring):
      raise Exception('who needs to be a string describing the heuristic, was %s of type %s' % (who, type(who)))
    if not isinstance(x, np.ndarray):
      raise Exception('x must be a numpy ndarray')
    self._x   = x
    self._who = who # heuristic.name, a string

  def __repr__(self):
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
    self._error = 0.0
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
    '''
    return self._error

  def __cmp__(self, other):
    assert isinstance(other, Result)
    return cmp(self._fx, other._fx)

  def __repr__(self):
    x = ' '.join('%11.6f' % _ for _ in self.x) if self.x != None else None
    return '%11.6f @ [%s]' % (self.fx, x)

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

  def project(self, point):
    '''
    projects given point into the search box. 
    e.g. [-1.1, 1] with box [(-1,1),(-1,1)] gives [-1,1] 
    '''
    assert isinstance(point, np.ndarray), 'point must be a numpy ndarray'
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

  def __repr__(self):
    descr = "Problem '%s' has %d dimensions. " % (self.__class__.__name__, self._dim)
    descr += "Box: [%s]" % ', '.join('[%.2f %.2f]' % (l,u) for l,u in self._box)
    return descr

