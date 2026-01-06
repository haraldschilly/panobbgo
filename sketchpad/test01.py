#!/usr/bin/env python
# -*- coding: utf8 -*-

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys
sys.path.append(".")
#from panobbgo.core import Results
from panobbgo.heuristics import Random, Nearby, Zero, LatinHypercube, Extremal, NelderMead, Center, WeightedAverage, QuadraticWlsModel
from panobbgo.strategies import StrategyRewarding  # , StrategyRoundRobin
#import numpy as np

from panobbgo.lib.classic import Rosenbrock

#problem = Shekel(3)
problem = Rosenbrock(4, par1=10)
#problem = RosenbrockConstraint(3, par1 = 10, par2 = .5)
#problem = RosenbrockStochastic(3)
#problem = Rosenbrock(2, 100)
#problem = RosenbrockAbs(2)
#problem = RosenbrockAbsConstraint(2)
#problem = Rastrigin(2, offset=1.11111)
#problem = Himmelblau()

# class LocalProblem(Problem):
#  def __init__(self):
#    box = [(-5,5)]
#    Problem.__init__(self, box)
#
#  def eval(self, x):
#    return np.cos(np.abs(x))**2.0
#
#problem = LocalProblem()

strategy = StrategyRewarding(problem, parse_args=True)
#strategy = StrategyRoundRobin(problem, parse_args = True)

strategy.add(Random)
strategy.add(Nearby, radius=1. / 1000, axes='all', new=3)
strategy.add(Nearby, radius=1. / 100, axes='all', new=3)
strategy.add(Nearby, radius=1. / 10, axes='all', new=3)
strategy.add(Nearby, radius=1. / 10, new=3)
strategy.add(Zero)
strategy.add(Extremal)
strategy.add(Center)
strategy.add(WeightedAverage)
strategy.add(NelderMead)
strategy.add(QuadraticWlsModel)

# target of max_eval generated points is the inverse of the gamma function
if False:
    from scipy import special as sp
    from scipy.optimize import fmin
    from panobbgo.config import get_config
    config = get_config()
    m = fmin(lambda x: (sp.gamma(x) - config.max_eval / 3.0) ** 2, [5])
    div = max(1, int(m[0]))
else:
    div = 5  # for 1000, should be 7 to 8
strategy.add(LatinHypercube, div=div)

strategy.start()

if strategy.best is None:
    print("no solution found")
else:
    print(u"best: %s" % strategy.best)
