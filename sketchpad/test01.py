#!/usr/bin/env python
# -*- coding: utf8 -*-

import sys
sys.path.append(".")
from panobbgo.problem import Problem
from panobbgo.core import Results
from panobbgo.heuristics import RandomPoints, NearbyPoints, ZeroPoint, LatinHypercube
from panobbgo.strategies import strategy1
import numpy as np

from panobbgo.problem import *

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
#calc      = CalculatedPoints(problem, results)
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
#strategy0 = Strategy0(problem, results, heurs)
#calc.set_machines(strategy0.generators) #use nb_machines for calc. new points
#calc.start()

#strategy0.run()

print strategy1(problem, results, heurs)

if not results.best is None:
  print "best: ", results.best
else:
  print "no best solution found"
