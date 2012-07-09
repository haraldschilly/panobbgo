#!/usr/bin/env python
# -*- coding: utf8 -*-

import sys
sys.path.append(".")
from panobbgo import *
import numpy as np


# define problem
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

#problem = Rosenbrock(2)
problem = RosenbrockStochastic(2)
#problem = Rosenbrock(3, 100)
#problem = RosenbrockAbs(2)
#problem = Rastrigin(2, offset=1.11111)
#problem = Himmelblau()

results = Results()

rand      = RandomPoints(problem, results)
near_1000 = NearbyPoints(problem, results, radius=1./1000, axes='all')
near_100  = NearbyPoints(problem, results, radius=1./100,  axes='all')
near_10_all = NearbyPoints(problem, results, radius=1./10, axes='all')
near_10   = NearbyPoints(problem, results, radius=1./10)
calc      = CalculatedPoints(problem, results)
zero      = ZeroPoint(problem)

# target of max_eval generated points is the inverse of the gamma function
if False:
  from scipy import special as sp
  from scipy.optimize import fmin
  m = fmin(lambda x : (sp.gamma(x) - config.max_eval / 3.0)**2, [5])
  div = max(1, int(m[0]))
else:
  div = 5 # for 1000, should be 7 to 8
lhyp= LatinHypercube(problem, results, div)

heurs = [ rand, near_10_all, near_100, lhyp, zero]
controller = Controller(problem, results, heurs)
calc.set_machines(controller.generators) #use nb_machines for calc. new points
calc.start()

controller.start()
# keep main thread alive until all created points are also consumed 
# and processed by the controller thread.
controller.join()

if not results.best is None:
  print "best: ", results.best
else:
  print "no best solution found"
