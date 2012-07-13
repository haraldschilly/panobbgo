# -*- coding: utf8 -*-
'''
This file contains the basic objects to build a problem and to do a single evaluation.
'''

# ATTN: make sure, that this doesn't depend on the config or threading modules.
#       the serialization and reconstruction won't work!
import numpy as np
from problem_lib import Point, Result, Problem
from IPython.utils.timing import time

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
