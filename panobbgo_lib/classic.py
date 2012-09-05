# -*- coding: utf8 -*-
# Copyright 2012 Harald Schilly <harald.schilly@univie.ac.at>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
Classic Problems
================
This file contains the basic objects to build a problem and to do a single evaluation.

.. inheritance-diagram:: panobbgo_lib.classic

.. codeauthor:: Harald Schilly <harald.schilly@univie.ac.at>
'''

# ATTN: make sure, that this doesn't depend on the config or threading modules.
#       the serialization and reconstruction won't work!
import numpy as np
from lib import Problem

class Rosenbrock(Problem):
  r'''
  Rosenbrock function with parameter ``par1``.

  .. math::

    f(x) = \sum_i (\mathit{par}_1 (x_{i+1} - x_i^2)^2 + (1-x_i)^2)

  '''
  def __init__(self, dims, par1 = 100):
    box = [(-2,2)] * dims
    box[0] = (0,2) # for cornercases + testing
    self.par1 = par1
    Problem.__init__(self, box)

  def eval(self, x):
    return sum(self.par1 * (x[1:] - x[:-1]**2)**2 + (1-x[:-1])**2)

class RosenbrockAbs(Problem):
  r'''
  Absolute Rosenbrock function.

  .. math::

   f(x) = \sum_i (\mathit{par}_1 \Big\| x_{i+1} - \| x_i \| \Big\| + \| 1 - x_i \|

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
  r'''
  Stochastic variant of Rosenbrock function.

  .. math ::

     f(x) = \sum_i (\mathit{par}_1 \mathit{eps}_i (x_{i+1} - x_i^2)^2 + (1-x_i)^2)

  where :math:`\mathit{eps}_i` is a uniformly random (n-1)-dimensional
  vector in :math:`\left[0, 1\right)^{n-1}`.
  '''
  def __init__(self, dims, par1 = 100, jitter = .1):
    box = [(-5,5)] * dims
    box[0] = (-1,2) # for cornercases + testing
    self.par1 = par1
    self.jitter = jitter
    Problem.__init__(self, box)

  def eval(self, x):
    eps = self.jitter * np.random.rand(self.dim - 1)
    ret = sum(self.par1 * eps * (x[1:] - x[:-1]**2)**2 + (1-x[:-1])**2)
    return ret

class Himmelblau(Problem):
  '''
  Himmelblau [HB]_ testproblem.

  .. math::
  
    f(x,y) = (x^2+y-11)^2 + (x+y^2-7)^2
  '''
  def __init__(self):
    Problem.__init__(self, [(-5,5)] * 2)

  def eval(self, x):
    x, y = x[0], x[1]
    return (x**2+y-11)**2+(x+y**2-7)**2

class Rastrigin(Problem):
  '''
  Rastrigin

  .. math::

    f(x) = \mathit{par}_1 \cdot n + \sum_i (x_i^2 - 10 \cos(2 \pi x_i) )

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
