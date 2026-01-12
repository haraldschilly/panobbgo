# -*- coding: utf8 -*-
# Copyright 2012 Harald Schilly <harald.schilly@gmail.com>
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

.. codeauthor:: Harald Schilly <harald.schilly@gmail.com>
"""

# ATTN: make sure, that this doesn't depend on the config or threading modules.
#       the serialization and reconstruction won't work!
import numpy as np
from .lib import Problem


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

     f(x) = \sum_i \mathit{par}_1 \Big\lVert x_{i+1} - \lVert x_i \rVert \Big\rVert + \lVert 1 - x_i \rVert

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

     \min f(x) & = \sum_i \mathit{par}_1 \Big\lVert x_{i+1} - \lVert x_i \rVert \Big\rVert + \lVert 1 - x_i \rVert \\
     \mathit{s.t.} \;\; & \lVert x_{i+1} - x_{i}\rVert \geq \mathit{par}_2 \;\; \forall i \in \{0,\dots,\mathit{dim}-1\} \\
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

    .. math::

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


class Ackley(Problem):

    r"""
    Ackley function.

    .. math::

      f(\mathbf{x}) = -20 \exp\left(-0.2 \sqrt{\frac{1}{n} \sum_{i=1}^{n} x_i^2}\right) - \exp\left(\frac{1}{n} \sum_{i=1}^{n} \cos(2\pi x_i)\right) + 20 + e

    The Ackley function is a multimodal test function for optimization algorithms.
    It has a nearly flat outer region and a large hole at the center. The function
    is continuous, convex and unimodal in a narrow valley.

    Global minimum: :math:`f(0,\dots,0) = 0`

    References
    ----------
    .. [Ackley] https://en.wikipedia.org/wiki/Ackley_function
    """

    def __init__(self, dims, box=None, **kwargs):
        box = box or [(-5, 5)] * dims
        Problem.__init__(self, box, **kwargs)

    def eval(self, x):
        n = self.dim
        term1 = -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / n))
        term2 = -np.exp(np.sum(np.cos(2 * np.pi * x)) / n)
        return term1 + term2 + 20 + np.e


class Griewank(Problem):

    r"""
    Griewank function.

    The Griewank function is a multimodal test function for optimization algorithms.
    It combines a quadratic term with a product of cosine terms, creating many
    local minima that make it challenging for optimization algorithms.

    .. math::

      f(\mathbf{x}) = 1 + \frac{1}{4000} \sum_{i=1}^{n} x_i^2 - \prod_{i=1}^{n} \cos\left(\frac{x_i}{\sqrt{i}}\right)

    The function has a global minimum at the origin and many local minima
    due to the cosine product term. The division by :math:`\sqrt{i}` in the
    cosine arguments creates different scales for different dimensions.

    Global minimum: :math:`f(0,\dots,0) = 0`

    References
    ----------
    .. [Griewank] https://en.wikipedia.org/wiki/Griewank_function
    """

    def __init__(self, dims, box=None, **kwargs):
        box = box or [(-600, 600)] * dims  # Common bounds for Griewank
        Problem.__init__(self, box, **kwargs)

    def eval(self, x):
        n = self.dim
        sum_term = np.sum(x**2) / 4000
        prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, n+1))))
        return 1 + sum_term - prod_term


class StyblinskiTang(Problem):

    r"""
    Styblinski-Tang function.

    The Styblinski-Tang function is a multimodal test function for optimization algorithms.
    It is defined as the sum of quartic polynomials for each dimension, creating
    multiple local minima that challenge optimization algorithms.

    .. math::

      f(\mathbf{x}) = \frac{1}{2} \sum_{i=1}^{n} (x_i^4 - 16 x_i^2 + 5 x_i)

    Each dimension contributes a quartic term with multiple local optima.
    The function has a global minimum at :math:`x_i \approx -2.903534` for all dimensions.

    Global minimum: :math:`f(-2.903534,\dots,-2.903534) \approx -39.16617 \cdot n`

    References
    ----------
    .. [StyblinskiTang] https://en.wikipedia.org/wiki/Styblinski%E2%80%93Tang_function
    """

    def __init__(self, dims, box=None, **kwargs):
        box = box or [(-5, 5)] * dims
        Problem.__init__(self, box, **kwargs)

    def eval(self, x):
        return 0.5 * np.sum(x**4 - 16 * x**2 + 5 * x)


class Schwefel(Problem):

    r"""
    Schwefel function.

    The Schwefel function is a highly multimodal test function for optimization algorithms.
    It has many local minima arranged in a complex pattern, making it extremely challenging
    for optimization algorithms to find the global minimum.

    .. math::

      f(\mathbf{x}) = 418.9829 \cdot n + \sum_{i=1}^{n} \left( -x_i \sin\left(\sqrt{|x_i|}\right) \right)

    The function has a global minimum at :math:`x_i \approx 420.9687` for all dimensions,
    where :math:`f(\mathbf{x}) = 0`. The many local minima and the narrow valley leading
    to the global optimum make this function particularly difficult to optimize.

    Global minimum: :math:`f(420.9687,\dots,420.9687) = 0`

    References
    ----------
    .. [Schwefel] https://en.wikipedia.org/wiki/Schwefel_function
    """

    def __init__(self, dims, box=None, **kwargs):
        box = box or [(-500, 500)] * dims  # Common bounds for Schwefel
        Problem.__init__(self, box, **kwargs)

    def eval(self, x):
        n = self.dim
        return 418.9829 * n + np.sum(-x * np.sin(np.sqrt(np.abs(x))))


class DixonPrice(Problem):

    r"""
    Dixon & Price function.

    The Dixon & Price function is a unimodal test function for optimization algorithms.
    It has a long, narrow valley that makes it challenging for algorithms to converge
    to the global minimum along the valley floor.

    .. math::

      f(\mathbf{x}) = (x_1 - 1)^2 + \sum_{i=2}^{D} i (2x_i^2 - x_{i-1})^2

    The function creates a valley that becomes increasingly narrow as the dimension increases,
    making it difficult for optimization algorithms to follow the valley to the minimum.

    Global minimum: :math:`f(2^{(\frac{2^i-2}{2^i})}, \dots, 2^{(\frac{2^D-2}{2^D})}) = 0`

    References
    ----------
    .. [DixonPrice] Momin Jamil and Xin-She Yang,
                    A literature survey of benchmark functions for global optimization problems,
                    {\it Int. Journal of Mathematical Modelling and Numerical Optimisation},
                    Vol.~4, No.~2, pp. 150--194 (2013).
                    DOI: 10.1504/IJMMNO.2013.055204
    """

    def __init__(self, dims, box=None, **kwargs):
        box = box or [(-10, 10)] * dims
        Problem.__init__(self, box, **kwargs)

    def eval(self, x):
        if self.dim < 2:
            raise ValueError("Dixon-Price function requires at least 2 dimensions")

        result = (x[0] - 1.0) ** 2
        for i in range(1, self.dim):
            result += (i + 1) * (2 * x[i]**2 - x[i-1]) ** 2
        return result


class Zakharov(Problem):

    r"""
    Zakharov function.

    The Zakharov function is a multimodal test function for optimization algorithms.
    It combines a quadratic term with polynomial terms involving weighted sums,
    creating multiple local optima that challenge optimization algorithms.

    .. math::

      f(\mathbf{x}) = \sum_{i=1}^{n} x_i^2 + \left(\frac{1}{2} \sum_{i=1}^{n} i x_i \right)^2 + \left(\frac{1}{2} \sum_{i=1}^{n} i x_i \right)^4

    The function has a global minimum at the origin and multiple local minima
    due to the higher-order polynomial terms.

    Global minimum: :math:`f(0,\dots,0) = 0`

    References
    ----------
    .. [Zakharov] Momin Jamil and Xin-She Yang,
                   A literature survey of benchmark functions for global optimization problems,
                   {\it Int. Journal of Mathematical Modelling and Numerical Optimisation},
                   Vol.~4, No.~2, pp. 150--194 (2013).
                   DOI: 10.1504/IJMMNO.2013.055204
    """

    def __init__(self, dims, box=None, **kwargs):
        box = box or [(-5, 10)] * dims
        Problem.__init__(self, box, **kwargs)

    def eval(self, x):
        n = self.dim
        # Calculate the weighted sum: sum_{i=1}^n i * x_i
        weighted_sum = np.sum((np.arange(1, n+1)) * x)

        # Quadratic term
        quad_term = np.sum(x**2)

        # Polynomial terms
        poly_term = 0.5 * weighted_sum
        poly_squared = poly_term**2
        poly_fourth = poly_term**4

        return quad_term + poly_squared + poly_fourth


class RosenbrockModified(Problem):

    r"""
    Rosenbrock Modified function.

    The Rosenbrock Modified function is a multimodal test function that adds a Gaussian bump
    to the standard Rosenbrock function, creating a local minimum that makes optimization more challenging.

    .. math::

      f(\mathbf{x}) = 74 + 100(x_2 - x_1^2)^2 + (1 - x_1)^2 - 400 \exp\left(-\frac{(x_1 + 1)^2 + (x_2 + 1)^2}{0.1}\right)

    The Gaussian term creates a local minimum at (1,1) while the global minimum remains at (-1,-1).
    This makes the function difficult to optimize because the local minimum has a larger basin of attraction.

    Global minimum: :math:`f(-1,-1) = 0`

    References
    ----------
    .. [RosenbrockModified] Momin Jamil and Xin-She Yang,
                            A literature survey of benchmark functions for global optimization problems,
                            {\it Int. Journal of Mathematical Modelling and Numerical Optimisation},
                            Vol.~4, No.~2, pp. 150--194 (2013).
                            DOI: 10.1504/IJMMNO.2013.055204
    """

    def __init__(self, box=None, **kwargs):
        box = box or [(-2, 2), (-2, 2)]
        Problem.__init__(self, box, **kwargs)

    def eval(self, x):
        x1, x2 = x[0], x[1]
        return (74 + 100 * (x2 - x1**2)**2 + (1 - x1)**2
                - 400 * np.exp(-((x1 + 1)**2 + (x2 + 1)**2) / 0.1))


class RotatedEllipse(Problem):

    r"""
    Rotated Ellipse function.

    The Rotated Ellipse function is a unimodal quadratic test function
    that represents an ellipse rotated in the coordinate system.

    .. math::

      f(\mathbf{x}) = 7x_1^2 - 6\sqrt{3}x_1 x_2 + 13x_2^2

    This is a quadratic form that creates an elliptical contour when rotated.
    It is unimodal with a single global minimum.

    Global minimum: :math:`f(0,0) = 0`

    References
    ----------
    .. [RotatedEllipse] Momin Jamil and Xin-She Yang,
                        A literature survey of benchmark functions for global optimization problems,
                        {\it Int. Journal of Mathematical Modelling and Numerical Optimisation},
                        Vol.~4, No.~2, pp. 150--194 (2013).
                        DOI: 10.1504/IJMMNO.2013.055204
    """

    def __init__(self, box=None, **kwargs):
        box = box or [(-500, 500), (-500, 500)]
        Problem.__init__(self, box, **kwargs)

    def eval(self, x):
        x1, x2 = x[0], x[1]
        return 7*x1**2 - 6*np.sqrt(3)*x1*x2 + 13*x2**2


class RotatedEllipse2(Problem):

    r"""
    Rotated Ellipse 2 function.

    The Rotated Ellipse 2 function is another unimodal quadratic test function
    that represents a different elliptical form.

    .. math::

      f(\mathbf{x}) = x_1^2 - x_1 x_2 + x_2^2

    This quadratic form creates a rotated elliptical contour.
    It is unimodal with a single global minimum.

    Global minimum: :math:`f(0,0) = 0`

    References
    ----------
    .. [RotatedEllipse2] Momin Jamil and Xin-She Yang,
                         A literature survey of benchmark functions for global optimization problems,
                         {\it Int. Journal of Mathematical Modelling and Numerical Optimisation},
                         Vol.~4, No.~2, pp. 150--194 (2013).
                         DOI: 10.1504/IJMMNO.2013.055204
    """

    def __init__(self, box=None, **kwargs):
        box = box or [(-500, 500), (-500, 500)]
        Problem.__init__(self, box, **kwargs)

    def eval(self, x):
        x1, x2 = x[0], x[1]
        return x1**2 - x1*x2 + x2**2


class Ripple1(Problem):

    r"""
    Ripple 1 function.

    The Ripple 1 function is a highly multimodal test function with many local minima.
    It creates a landscape with 25 holes in a 5x5 grid plus additional ripples.

    .. math::

      f(\mathbf{x}) = \sum_{i=1}^{2} -e^{-2\ln2(\frac{x_i-0.1}{0.8})^2} (\sin^6(5\pi x_i) + 0.1\cos^2(500\pi x_i))

    The function has one global minimum and 252,004 local minima.
    The landscape consists of 25 holes forming a 5×5 regular grid plus high-frequency ripples.

    Global minimum: Located within bounds (complex)

    References
    ----------
    .. [Ripple1] Momin Jamil and Xin-She Yang,
                 A literature survey of benchmark functions for global optimization problems,
                 {\it Int. Journal of Mathematical Modelling and Numerical Optimisation},
                 Vol.~4, No.~2, pp. 150--194 (2013).
                 DOI: 10.1504/IJMMNO.2013.055204
    """

    def __init__(self, box=None, **kwargs):
        box = box or [(0, 1), (0, 1)]
        Problem.__init__(self, box, **kwargs)

    def eval(self, x):
        result = 0.0
        for i in range(2):
            xi = x[i]
            exp_term = np.exp(-2*np.log(2) * ((xi - 0.1)/0.8)**2)
            sin_term = np.sin(5*np.pi*xi)**6
            cos_term = 0.1 * np.cos(500*np.pi*xi)**2
            result += -exp_term * (sin_term + cos_term)
        return result


class Ripple25(Problem):

    r"""
    Ripple 25 function.

    The Ripple 25 function is a multimodal test function similar to Ripple 1
    but without the high-frequency cosine ripples.

    .. math::

      f(\mathbf{x}) = \sum_{i=1}^{2} -e^{-2\ln2(\frac{x_i-0.1}{0.8})^2} \sin^6(5\pi x_i)

    The function creates 25 holes in a 5×5 grid pattern without additional ripples.
    It has fewer local minima compared to Ripple 1.

    Global minimum: Located within bounds (complex)

    References
    ----------
    .. [Ripple25] Momin Jamil and Xin-She Yang,
                  A literature survey of benchmark functions for global optimization problems,
                  {\it Int. Journal of Mathematical Modelling and Numerical Optimisation},
                  Vol.~4, No.~2, pp. 150--194 (2013).
                  DOI: 10.1504/IJMMNO.2013.055204
    """

    def __init__(self, box=None, **kwargs):
        box = box or [(0, 1), (0, 1)]
        Problem.__init__(self, box, **kwargs)

    def eval(self, x):
        result = 0.0
        for i in range(2):
            xi = x[i]
            exp_term = np.exp(-2*np.log(2) * ((xi - 0.1)/0.8)**2)
            sin_term = np.sin(5*np.pi*xi)**6
            result += -exp_term * sin_term
        return result


class Shekel(Problem):

    r"""
    Shekel Function [SH]_.

    The Shekel function is a multimodal test function for optimization algorithms.
    It creates :math:`m` local minima in :math:`n` dimensions by summing inverse
    quadratic terms centered at different points.

    .. math::

      f(\vec{x}) = \sum_{i = 1}^{m} \frac{1}{c_{i} + \sum_{j = 1}^{n} (x_{j} - a_{ji})^2}

    where :math:`a_{ji}` are the coordinates of the :math:`i`-th minimum in dimension :math:`j`,
    and :math:`c_i` controls the depth/width of each minimum.

    The function is highly multimodal with many local optima, making it challenging
    for optimization algorithms to find the global minimum.

    Parameters
    ----------
    m : int
        Number of local minima (default: 10)
    a : array_like, shape (n, m)
        Positions of the m minima in n dimensions. If None, generated automatically.
    c : array_like, shape (m,)
        Scaling factors for each minimum. If None, generated automatically.

    Global minimum: Depends on parameters, typically :math:`f(\vec{x}^*) \approx -m`

    References
    ----------
    .. [SH] https://en.wikipedia.org/wiki/Shekel_function
    """

    def __init__(self, dims, m=10, a=None, c=None, box=None, **kwargs):
        box = box or [(-2, 2)] * dims
        Problem.__init__(self, box, **kwargs)
        self.m = m

        if a is None:
            a = np.empty((dims, m), dtype=float)
            phi = np.linspace(0, 2 * np.pi, num=dims, endpoint=False)
            for i in range(m):
                a[:, i] = np.sin(phi * ((1. + i) / m))

        assert a.shape == (dims, m)

        if c is None:
            from itertools import cycle
            cc = cycle([.1, .2, .2, .4, .4, .6, .3, .7, .5, .5])
            c = [next(cc) for _ in range(m)]

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
               5 (x_3 - x_4)^2 +
               ((x_2 + 2 x_3)^2)^2 +
               10 ((x_1 - x_4)^2)^2
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
        n = self.dim
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

    .. :math:

        F(x) = \sum_{i=1}^m \left(e^{-t_i x_1} - e^{-t_i x_2} - x_3(e^{-t_i} - e^{-10 t_i}\right)^2

        \text{where}

        t_i = i / 10
    """

    def __init__(self, m=1, box=None, **kwargs):
        """
        :param int m: positive integer (default 1)
        """
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

     .. math::

        F(x) = 100 (x_2 - x_1^2)^2 + (1-x_1)^2 +
            90 (x_4-x_3^2)^2 + (1-x_3)^2 +
            10 (x_2 + x_4 - 2)^2 + 10 (x_2-x_4)^2

    """

    def __init__(self, box=None, **kwargs):
        box = box or [(-5, 5)] * 4
        Problem.__init__(self, box, **kwargs)

    def eval(self, x):
        return 10 * (x[1] - x[0] ** 2) + (1 - x[0]) ** 2 + \
            90 * (x[3] - x[2] ** 2) + (1 - x[2]) ** 2 + \
            10 * (x[1] + x[3] - 2) ** 2 + 10 * (x[1] - x[3])


class HelicalValley(Problem):

    r"""
    Helical valley function [UncTest]_

    .. math::

        F(x) = \left(10(x_3 - 10 \, \Theta (x_1, x_2) \right)^2 +
            \left(10 \sqrt{ x_1^2 + x_2^2 } - 1\right)^2 + x_3^2

    where

    .. math::

        \Theta(x_1, x_2) =
            \begin{cases}
                \frac{1}{2 \pi} \arctan (\frac{x_2}{x_1}),      & \text{if} & x_1 > 0 \\
                \frac{1}{2 \pi} \arctan (\frac{x_2}{x_1}) + .5, & \text{if} & x_1 < 0
            \end{cases}

    """

    def __init__(self, box=None, **kwargs):
        box = box or [(-5, 5)] * 3
        Problem.__init__(self, box, **kwargs)

    @staticmethod
    def _theta(x0, x1):
        ret = 1. / (2 * np.pi) * np.arctan(x1 / x0)
        if x0 < 0:
            ret += 0.5
        return ret

    def eval(self, x):
        f0 = 10 * (x[2] - 10 * self._theta(x[0], x[1]))
        f1 = 10 * (np.sqrt(x[0] ** 2 + x[1] ** 2) - 1)
        f2 = x[2]
        return f0 ** 2 + f1 ** 2 + f2 ** 2


class Beale(Problem):

    r"""
    Beale function [UncTest]_

    .. math::

        F(x) = \sum_{i=1}^3 \left( y_i - x_1 ( 1-x_2^i) \right) ^2

    where

    .. math::

        y_1 = 1.5, \, y_2 = 2.25, \, y_3 = 2.625
    """

    def __init__(self, box=None, **kwargs):
        box = box or [(-5, 5)] * 2
        Problem.__init__(self, box, **kwargs)

    def eval(self, x):
        y = [1.5, 2.25, 2.625]
        v = [y[i] - x[0] * (1 - x[1] ** (i + 1)) for i in range(3)]
        return sum(_ ** 2 for _ in v)


class NesterovQuadratic(Problem):

    r"""
    Nesterov's Quadratic Function [NQuad]_

    .. math::

            F(x) = \frac{1}{2} \lVert A x - b \rVert_2^2 + \lVert x \rVert_1
    """

    def __init__(self, dim=None, box=None, A=None, b=None, nonsmooth=True, **kwargs):
        r"""
        :param boolean nonsmooth: add the nonsmooth :math:`\lVert x\rVert_1` part (default: True)
        """
        self.nonsmooth = nonsmooth
        if A is None and b is None:
            dim = 2
            A = np.random.randn(dim, dim)
            b = np.random.randn(dim)
        else:
            assert b is not None, "b must be provided if A is provided"
            dim = b.shape[0]
            if A is None:
                A = np.random.randn(dim, dim)
        box = box or [(-5, 5)] * dim
        self.A, self.b = A, b
        Problem.__init__(self, box, **kwargs)

    def eval(self, x):
        ret = .5 * ((self.A.dot(x) - self.b) ** 2).sum()
        if self.nonsmooth:
            ret += np.abs(x).sum()
        return ret


class Arwhead(Problem):

    r"""
    ARWHEAD test problem [Conn]_

    .. math::
        F(x) = \sum_{i=1}^{n-1} \left( (x_i^2 + x_n^2)^2 - 4 x_i + 3 \right)
    """

    def __init__(self, dim=None, box=None, **kwargs):
        if dim is None:
            dim = box.shape[0] if box else 3
        box = box or [(-5, 5)] * dim
        Problem.__init__(self, box, **kwargs)

    def eval(self, x):
        return ((x[:-1] ** 2 + x[-1] ** 2) ** 2 - 4 * x[:-1] + 3).sum()


class Branin(Problem):

    r"""
    BRANIN test problem [branin]_

    .. math::
        F(X) = a(x_2 - b x_1^2 + c x_1 - r)^2 + s(1-t) \cos(x_1) + s

        a = 1,\,b = 5.1 ⁄ (4 π^2),\, c = 5 ⁄ π,\, r = 6,\, s = 10 and t = 1 ⁄ (8π)
    """

    def __init__(self,
            a = 1,
            b = 5.1 / (4 * np.pi**2),
            c = 5 / np.pi,
            r = 6,
            s = 10,
            t = 1,
            **kwargs):
        box = [[-5, 10], [0, 15]]
        self.a = a
        self.b = b
        self.c = c
        self.r = r
        self.s = s
        self.t = t
        Problem.__init__(self, box, **kwargs)

    def eval(self, x):
        x1, x2 = x
        term1 = self.a * (x2 - self.b*x1**2 + self.c*x1 - self.r)**2
        term2 = self.s*(1-self.t)*np.cos(x1)
        y = term1 + term2 + self.s
        return y


class GoldsteinPrice(Problem):
    r"""
    Goldstein-Price test function [branin]_

    .. math:
        f(x,y) = \left(1+\left(x+y+1\right)^{2}\left(19-14x+3x^{2}-14y+6xy+3y^{2}\right)\right)
                 \left(30+\left(2x-3y\right)^{2}\left(18-32x+12x^{2}+48y-36xy+27y^{2}\right)\right)
    """
    def __init__(self, **kwargs):
        box = [(-2, 2), (-2, 2)]
        Problem.__init__(self, box, **kwargs)

    def eval(self, x):
        x1, x2 = x
        a = 1+(x1+x2+1)**2*(19-14*x1+3*x1**2-14*x2+6*x1*x2+3*x2**2)
        b = 30+(2*x1-3*x2)**2*(18-32*x1+12*x1**2+48*x2-36*x1*x2+27*x2**2)
        return a*b


