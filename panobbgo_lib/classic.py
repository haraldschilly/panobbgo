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

"""
Classic Problems
================
This file contains the basic objects to build a problem and to do a single evaluation.

.. inheritance-diagram:: panobbgo_lib.classic

.. codeauthor:: Harald Schilly <harald.schilly@univie.ac.at>
"""

# ATTN: make sure, that this doesn't depend on the config or threading modules.
#       the serialization and reconstruction won't work!
import numpy as np
from lib import Problem


class Rosenbrock(Problem):

    r"""
    Rosenbrock function with parameter ``par1``.

    .. math::

      f(x) = \sum_i (\mathit{par}_1 (x_{i+1} - x_i^2)^2 + (1-x_i)^2)

    """

    def __init__(self, dims, par1=100, **kwargs):
        box = [(-2, 2)] * dims
        box[0] = (0, 2)  # for cornercases + testing
        self.par1 = par1
        Problem.__init__(self, box, **kwargs)

    def eval(self, x):
        return np.sum(self.par1 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


class RosenbrockConstraint(Problem):

    r"""
    Constraint Rosenbrock function with parameter ``par1`` and ``par2``.

    .. math::

      \min f(x) & = \sum_i \mathit{par}_1 (x_{i+1} - x_i^2)^2 + (1-x_i)^2 \\
      \mathit{s.t.} \;\; & (x_{i+1} - x_{i})^2 \geq \mathit{par}_2 \;\; \forall i \in \{0,\dots,\mathit{dim}-1\} \\
                         & x_i \geq 0 \;\;                              \forall i

    """

    def __init__(self, dims, par1=100, par2=0.25, **kwargs):
        box = [(-2, 2)] * dims
        box[0] = (0, 2)  # for cornercases + testing
        self.par1 = par1
        self.par2 = par2
        Problem.__init__(self, box, **kwargs)

    def eval(self, x):
        return sum(self.par1 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2) - 50

    def eval_constraints(self, x):
        cv = - (x[1:] - x[:-1]) ** 2.0 + self.par2
        cv[cv < 0] = 0.0
        pos = -x.copy()  # note the -
        pos[pos < 0] = 0.0
        return np.concatenate([cv, pos])


class RosenbrockAbs(Problem):

    r"""
    Absolute Rosenbrock function.

    .. math::

     f(x) = \sum_i \mathit{par}_1 \Big\| x_{i+1} - \| x_i \| \Big\| + \| 1 - x_i \|

    """

    def __init__(self, dims, par1=100, **kwargs):
        box = [(-5, 5)] * dims
        box[0] = (0, 2)  # for cornercases + testing
        self.par1 = par1
        Problem.__init__(self, box, **kwargs)

    def eval(self, x):
        return np.sum(self.par1 * np.abs(x[1:] - np.abs(x[:-1])) +
                      np.abs(1 - x[:-1]))


class RosenbrockAbsConstraint(Problem):

    r"""
    Absolute Rosenbrock function.

    .. math::

     \min f(x) & = \sum_i \mathit{par}_1 \Big\| x_{i+1} - \| x_i \| \Big\| + \| 1 - x_i \| \\
     \mathit{s.t.} \;\; & \|x_{i+1} - x_{i}\| \geq \mathit{par}_2 \;\; \forall i \in \{0,\dots,\mathit{dim}-1\} \\
                        & x_i \geq 0 \;\;                          \forall i

    """

    def __init__(self, dims, par1=100, par2=0.1, **kwargs):
        box = [(-5, 5)] * dims
        box[0] = (0, 2)  # for cornercases + testing
        self.par1 = par1
        self.par2 = par2
        Problem.__init__(self, box, **kwargs)

    def eval(self, x):
        return sum(self.par1 * np.abs(x[1:] - np.abs(x[:-1])) +
                   np.abs(1 - x[:-1]))

    def eval_constraints(self, x):
        cv = - np.abs(x[1:] - x[:-1]) + self.par2
        cv[cv < 0] = 0.0
        pos = -x.copy()  # note the -
        pos[pos < 0] = 0.0
        return np.concatenate([cv, pos])


class RosenbrockStochastic(Problem):

    r"""
    Stochastic variant of Rosenbrock function.

    .. math ::

       f(x) = \sum_i (\mathit{par}_1 \mathit{eps}_i (x_{i+1} - x_i^2)^2 + (1-x_i)^2)

    where :math:`\mathit{eps}_i` is a uniformly random (n-1)-dimensional
    vector in :math:`\left[0, 1\right)^{n-1}`.
    """

    def __init__(self, dims, par1=100, jitter=.1, **kwargs):
        box = [(-5, 5)] * dims
        box[0] = (-1, 2)  # for cornercases + testing
        self.par1 = par1
        self.jitter = jitter
        Problem.__init__(self, box, **kwargs)

    def eval(self, x):
        eps = self.jitter * np.random.rand(self.dim - 1)
        ret = sum(
            self.par1 * eps * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)
        return ret


class Himmelblau(Problem):

    r"""
    Himmelblau [HB]_ testproblem.

    .. math::

      f(x,y) = (x^2+y-11)^2 + (x+y^2-7)^2

    .. [HB] http://en.wikipedia.org/wiki/Himmelblau%27s_function
    """

    def __init__(self, **kwargs):
        Problem.__init__(self, [(-5, 5)] * 2, **kwargs)

    def eval(self, x):
        x, y = x[0], x[1]
        return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2


class Rastrigin(Problem):

    r"""
    Rastrigin

    .. math::

      f(x) = \mathit{par}_1 \cdot n + \sum_i (x_i^2 - 10 \cos(2 \pi x_i) )

    """

    def __init__(self, dims, par1=10, offset=0, box=None, **kwargs):
        box = box or [(-2, 2)] * dims
        self.offset = offset
        self.par1 = par1
        Problem.__init__(self, box, **kwargs)

    def eval(self, x):
        x = x - self.offset
        return self.par1 * self.dim + \
            np.sum(x ** 2 - self.par1 * np.cos(2 * np.pi * x))


class Shekel(Problem):

    r"""
    Shekel Function [SH]_.

    For :math:`m` minima in :math:`n` dimensions:

    .. math::

      f(\vec{x}) = \sum_{i = 1}^{m} \tfrac{1}{c_{i} + \sum\limits_{j = 1}^{n} (x_{j} - a_{ji})^2 }

    .. [SH] http://en.wikipedia.org/wiki/Shekel_function
    """

    def __init__(self, dims, m=10, a=None, c=None, box=None, **kwargs):
        box = box or [(-2, 2)] * dims
        Problem.__init__(self, box, **kwargs)
        self.m = m

        if a is None:
            a = np.empty((dims, m), dtype=np.float)
            phi = np.linspace(0, 2 * np.pi, num=dims, endpoint=False)
            for i in range(m):
                a[:, i] = np.sin(phi * ((1. + i) / m))

        assert a.shape == (dims, m)

        if c is None:
            from itertools import cycle
            cc = cycle([.1, .2, .2, .4, .4, .6, .3, .7, .5, .5])
            c = [cc.next() for _ in range(m)]

        assert len(c) == m

        self.a = a
        self.c = c

    def eval(self, x):
        def denom(i):
            d = x - self.a[:, i]
            return self.c[i] + d.dot(d)
        return - np.sum([1. / denom(i) for i in range(self.m)])


class DeJong(Problem):

    r"""
    De Jong function:

    .. math::

        \operatorname{DJ}(x) = c \, \sum_{i=1}^n (x - dx)_i^2

    with defaults :math:`c = 1` and :math:`dx = \vec{0}`.
    """

    def __init__(self, dims, c=1, box=None, **kwargs):
        box = box or [(-5, 5)] * dims
        self.c = c
        Problem.__init__(self, box, **kwargs)

    def eval(self, x):
        return self.c * np.dot(x, x)


class Quadruple(Problem):

    r"""
    Quadruple Function [QuadF]_

    .. math::

        Q(x) = c \, \sum_{i=1}^n \left( \frac{x_i - dx}{4} \right)^4

    with defaults :math:`c = 1` and :math:`dx = \vec{0}`.

    .. [QuadF] Hewlett, Joel D., Bogdan M. Wilamowski, and Gunhan Dundar.
        "Optimization using a modified second-order approach with evolutionary enhancement."
        Industrial Electronics, IEEE Transactions on 55.9 (2008): 3374-3380.
    """

    def __init__(self, dims, c=1, box=None, **kwargs):
        box = box or [(-10, 10)] * dims
        self.c = c
        Problem.__init__(self, box, **kwargs)

    def eval(self, x):
        return self.c * np.sum((x / 4.) ** 4)


class Powell(Problem):

    r"""
    Powell singular function [UncTest]_

    .. math::
        P(x) = (x_1 + 10 x_2)^2 +
               (\sqrt{5} (x_3 - x_4))^2 +
               ((x_2 + 2 x_3)^2)^2 +
               (\sqrt{10} (x_1 - x_4)^2)^2

    .. [UncTest] Moré, Jorge J., Burton S. Garbow, and Kenneth E. Hillstrom.
        "Testing unconstrained optimization software."
        ACM Transactions on Mathematical Software (TOMS) 7.1 (1981): 17-41.
    """

    def __init__(self, box=None, **kwargs):
        box = box or [(-10, 10)] * 4
        Problem.__init__(self, box, **kwargs)

    def eval(self, x):
        f = (x[0] + 10 * x[1]) ** 2 + \
            5 * (x[2] - x[3]) ** 2 + \
            ((x[1] - 2 * x[2]) ** 2) ** 2 +\
            10 * (x[0] - x[3]) ** 2
        return f


class Trigonometric(Problem):

    r"""
    Trigonometric function [UncTest]_

    .. math::

        f_i(x) = n - \sum_{j=1}^{n} \cos x_j  + i (1-\cos x_i)-\sin x_i

        f(x, n) = \sum_{i=1}^{m} f_i^2

    with :math:`n = m`.
    """

    def __init__(self, dims, box=None, **kwargs):
        box = box or [(-1, 1)] * dims
        Problem.__init__(self, box, **kwargs)

    def eval(self, x):
        ret = 0
        n = m = self.dim
        for i in range(n):
            tmp = i * (1 - np.cos(x[i])) - np.sin(x[i])
            fi = n - np.sum(np.cos(x) - tmp)
            ret += fi ** 2
        return ret


class SumDifferentPower(Problem):

    r"""
    Sum of different power function [CompStudy]_

    .. :math:

        F(x) = \sum_{i=1}^n |x_i|^{i+1}

    .. [CompStudy] Pham, Nam, A. Malinowski, and T. Bartczak.
        "Comparative study of derivative free optimization algorithms."
        Industrial Informatics, IEEE Transactions on 7.4 (2011): 592-600.
    """

    def __init__(self, dims, box=None, **kwargs):
        box = box or [(-5, 5)] * dims
        Problem.__init__(self, box, **kwargs)

    def eval(self, x):
        return np.abs(np.power(x, np.arange(self.dim) + 2)).sum()


class Step(Problem):

    r"""
    Step function [CompStudy]_

    .. :math:

        F(x) = \sum_{i=1}^n |x_i + 0.5|^2
    """

    def __init__(self, dims, box=None, **kwargs):
        box = box or [(-5, 5)] * dims
        Problem.__init__(self, box, **kwargs)

    def eval(self, x):
        return np.sum(np.abs(x + 0.5) ** 2)


class Box(Problem):

    r"""
    Box function [UncTest]_

    :param int m: positive integer (default 1)

    .. :math:

        F(x) = \sum_{i=1}^m \left(e^{-t_i x_1} - e^{-t_i x_2} - x_3(e^{-t_i} - e^{-10 t_i}\right)^2

        \text{where}

        t_i = i / 10
    """

    def __init__(self, m=1, box=None, **kwargs):
        box = box or [(-5, 5)] * 3
        self.m = m
        Problem.__init__(self, box, **kwargs)

    def eval(self, x):
        ret = 0.
        for i in range(1, self.m + 1):
            ti = i / 10.
            tmp = np.exp(ti * x[0]) - np.exp(-ti * x[1]) - \
                x[2] * (np.exp(-ti) - np.exp(-10 * ti))
            ret += tmp ** 2
        return ret


class Wood(Problem):

    r"""
    Wood function [UncTest]_

     .. :math:

        F(x) = 100 (x_2 - x_1^2)^2 + (1-x_1)^2 +
            90 (x_4-x_3^2)^2 + (1-x_3)^2 +
            10 (x_2 + x_4 - 2)^2 + 10 (x_2-x_4)^2

    """

    def __init__(self, box=None, **kwargs):
        box = box or [(-5, 5)] * 4
        Problem.__init__(self, box, **kwargs)

    def eval(self, x):
        ret = 10 * (x[1] - x[0] ** 2) + (1 - x[0]) ** 2 + \
            90 * (x[3] - x[2] ** 2) + (1 - x[2]) ** 2 + \
            10 * (x[1] + x[3] - 2) ** 2 + 10 * (x[1] - x[3])
        return ret
