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
    self._dims = dims
    self.par1 = par1
    Problem.__init__(self, box)

  def eval(self, x):
    x -= self.offset
    return self.par1 * self.dim + \
           sum(x**2 - self.par1 * np.cos(2 * np.pi * x))
  
problem = Rosenbrock(3)
#problem = RosenbrockAbs(3)
#problem = Rastrigin(3, offset=1.1)

results = Results()

rand      = RandomPoints(problem, results)
heur_1000 = NearbyPoints(problem, results, radius=1./1000)
heur_100  = NearbyPoints(problem, results, radius=1./100)
heur_10   = NearbyPoints(problem, results, radius=1./10)
calc      = CalculatedPoints(problem, results)
zero      = ZeroPoint(problem)

# target of 1000 generated points is the inverse of the gamma function
from scipy import special as sp
from scipy.optimize import fmin
div = max(1, int(fmin(lambda x : (sp.gamma(x) - 1000)**2, [5])[0]))
#print div # for 1000, should be 7
lhyp_pts = LatinHypercube(problem, results, div)

controller = Controller(problem, results, rand_pts,heur_pts_100,calc_pts, lhyp_pts)
calc_pts.set_machines(controller.generators) #use nb_machines for calc. new points
calc_pts.start()

controller.start()
# keep main thread alive until all created points are also consumed 
# and processed by the controller thread.
controller.join()

print "best: ", results.best()
