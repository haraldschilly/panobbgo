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
    box = [(-5,5)] * dims
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
    


class Rastrigin(Problem):
  '''
  f(x) = 10*n + sum_i (x_i^2 - 10 cos(2 pi x_i) )
  '''
  def __init__(self, dims, par1 = 10, offset=0):
    box = [(-5,5)] * dims
    self.offset = offset
    self.par1 = par1
    Problem.__init__(self, box)

  def eval(self, x):
    x = x - self.offset
    return self.par1 * self.dim + \
           sum(x**2 - self.par1 * np.cos(2 * np.pi * x))
  
#problem = Rosenbrock(3)
#problem = RosenbrockAbs(3)
problem = Rastrigin(2, offset=1.1)

results = Results()

rand      = RandomPoints(problem, results)
near_1000 = NearbyPoints(problem, results, radius=1./1000)
near_100  = NearbyPoints(problem, results, radius=1./100)
near_10   = NearbyPoints(problem, results, radius=1./10)
calc      = CalculatedPoints(problem, results)
zero      = ZeroPoint(problem)

# target of max_eval generated points is the inverse of the gamma function
if False:
  from scipy import special as sp
  from scipy.optimize import fmin
  m = fmin(lambda x : (sp.gamma(x) - config.max_eval)**2, [5])
  div = max(1, int(m[0]))
else:
  div = 7 # for 1000, should be 7 to 8
lhyp= LatinHypercube(problem, results, div)

controller = Controller(problem, results, rand, near_10, near_100, calc, lhyp, zero)
calc.set_machines(controller.generators) #use nb_machines for calc. new points
calc.start()

controller.start()
# keep main thread alive until all created points are also consumed 
# and processed by the controller thread.
controller.join()

print "best: ", results.best
