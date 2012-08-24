#!/usr/bin/env python
# -*- coding: utf8 -*-

import sys
sys.path.append(".")
#from panobbgo.core import Results
from panobbgo.heuristics import Random, Nearby, \
     Zero, LatinHypercube, Extremal, \
     Center, WeightedAverage, Testing
from panobbgo.strategies import Strategy0
#import numpy as np

from panobbgo_problems.classic import *

#problem = Rosenbrock(2)
problem = RosenbrockStochastic(2)
#problem = Rosenbrock(2, 100)
#problem = RosenbrockAbs(2)
#problem = Rastrigin(2, offset=1.11111)
#problem = Himmelblau()

#class LocalProblem(Problem):
#  def __init__(self):
#    box = [(-5,5)]
#    Problem.__init__(self, box)
#
#  def eval(self, x):
#    return np.cos(np.abs(x))**2.0
#
#problem = LocalProblem()

rand        = Random()
near_1000   = Nearby(radius=1./1000, axes='all')
near_100    = Nearby(radius=1./100,  axes='all')
near_10_all = Nearby(radius=1./10,   axes='all')
near_10     = Nearby(radius=1./10)
zero        = Zero()
extremal    = Extremal()
center      = Center()
avg         = WeightedAverage()
testing     = Testing()

# target of max_eval generated points is the inverse of the gamma function
if False:
  from scipy import special as sp
  from scipy.optimize import fmin
  m = fmin(lambda x : (sp.gamma(x) - config.max_eval / 3.0)**2, [5])
  div = max(1, int(m[0]))
else:
  div = 5 # for 1000, should be 7 to 8
lhyp= LatinHypercube(div)

heurs = [ center, rand, near_10, lhyp, zero, extremal, avg, testing]
#heurs = [ rand, avg ]

strategy0 = Strategy0(problem, heurs)
#calc.set_machines(strategy0.generators) #use nb_machines for calc. new points
#calc.start()

if not strategy0.best is None:
  print "best: ", strategy0.best
else:
  print "best solution is None"
