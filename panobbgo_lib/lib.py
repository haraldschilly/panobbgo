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
Library Classes
===============

This file contains the basic objects to build a problem and to do a single evaluation.

.. inheritance-diagram:: panobbgo_lib.lib

.. Note:: This is used by :mod:`panobbgo` and :mod:`panobbgo_lib`.

.. codeauthor:: Harald Schilly <harald.schilly@univie.ac.at>
"""

# ATTN: make sure, that this doesn't depend on the config or threading modules.
#       the serialization and reconstruction won't work!
import numpy as np
from IPython.utils.timing import time


class Point(object):

    """
    This contains the x vector for a new point and a
    reference to :attr:`.who` has generated it.
    """

    def __init__(self, x, who):
        if not isinstance(who, basestring):
            raise Exception(
                'who needs to be a string describing the heuristic, was %s of type %s'
                % (who, type(who)))
        if not isinstance(x, np.ndarray):
            raise Exception('x must be a numpy ndarray')
        self._x = x
        self._who = who  # heuristic.name, a string

    def __repr__(self):
        """
        >>> Point
        <class 'panobbgo_lib.lib.Point'>

        >>> x = np.array([1,2])
        >>> repr(Point(x, 'doctest'))
        '[1 2] by doctest'
        """
        return '%s by %s' % (self.x, self.who)

    @property
    def x(self):
        "The vector :math:`x`, a :class:`numpy.ndarray`"
        return self._x

    @property
    def who(self):
        """
        A string, which is the :attr:`~panobbgo.core.Module.name` of a heuristic.

        To get the actual heuristic, use the
        :meth:`strategie's heuristic <panobbgo.core.StrategyBase.heuristic>` method.
        """
        return self._who

    def __getitem__(self, item):
        """
        get x vector

        >>> p = Point(np.array([1,42]), "doctest")
        >>> p[1]
        42
        """
        return self._x[item]


class Result(object):

    r"""
    This represents one result, wich is a mapping of a :class:`.Point`
    :math:`x \rightarrow f(x)`.

    Additionally, there is also

    - :attr:`.error`: estimated or calculated :math:`\Delta f(x)`.
    - :attr:`.cv_vec`: a possibly empty vector listing the constraint violation for
      each constraint.
    - :attr:`.cnt`: An integer counter, starting at 0.
    """

    def __init__(self, point, fx, cv_vec=None, cv_norm=None, error=0.0):
        """
        Args:

        - ``cv``: the constraint violation vector
        - ``cv_norm``: the norm used to calculate :attr:`.cv`.
          (see :func:`numpy.linalg.norm`, default ``None`` means 2-norm)
        """
        if point and not isinstance(point, Point):
            raise Exception("point must be a Point")
        self._point = point
        self._fx = fx
        self._error = error
        self._cv_vec = cv_vec
        self._cv_norm = cv_norm
        self._time = time.time()
        self._cnt = None

    @property
    def cnt(self):
        """
        Integer ID for this result.
        """
        return self._cnt

    @property
    def x(self):
        """
        Point :math:`x` where this result has been evaluated.
        """
        return self.point.x if self.point else None

    @property
    def point(self):
        """
        Returns the actual :class:`.Point` object.
        """
        return self._point

    @property
    def fx(self):
        """
        The function value :math:`f(x)` after :meth:`evaluating <panobbgo_lib.lib.Problem.eval>` it.
        """
        return self._fx

    @property
    def cv_vec(self):
        """
        Vector of constraint violations for each constraint, or None.

        .. Note::

           Be aware, that entries could be negative. This is useful if you want to know
           how well a point is satisfied. The `.cv` property just looks at the positive
           entries, though.
        """
        return self._cv_vec

    @property
    def cv(self):
        """
        The chosen norm of :attr:`.cv_vec`; see ``cv_norm`` in constructor.

        .. Note::

            Only the positive entries are used to calculate the norm!
        """
        if self._cv_vec is None:
            return 0.0
        from numpy.linalg import norm
        return norm(self._cv_vec[self._cv_vec > 0.0], self._cv_norm)

    @property
    def pp(self):
        """
        pareto point, i.e. array([cv, fx])
        """
        return np.array([self.cv, self.fx])

    @property
    def who(self):
        """
        The :attr:`~panobbgo.core.Module.name` of the heuristic, who
        did generate this point (String).
        """
        return self.point.who

    @property
    def error(self):
        """
        Error margin of function evaluation, usually 0.0.
        """
        return self._error

    def __cmp__(self, other):
        """
        Compare with other point by fx (and fx only!).

        .. Note ::

          This is also used by mechanism
          like Best -> pareto_front
        """
        assert isinstance(other, Result)
        return cmp(self._fx, other._fx)

    def __repr__(self):
        x = ' '.join(
            '%11.6f' % _ for _ in self.x) if self.x is not None else None
        cv = '' if self._cv_vec is None else u'\u22DB%8.4f ' % self.cv
        return '%11.6f %s@ [%s]' % (self.fx, cv, x)


class Problem(object):

    """
    this is used to store the objective function,
    information about the problem, etc.
    """

    def __init__(self, box=None, dx=None):
        r"""
        :param list box: list of tuples for the bounding box with length n,
                          e.g.: :math:`\left[ (-1,1), (-100, 0), (0, 0.01) \right]`.
        :param np.ndarray dx: translational offset which also affects the box,
                   n-dimensional vector (default: None)
        """
        assert isinstance(box, (list, tuple)), "box argument must be a list or tuple"

        from numbers import Number
        for entry in box:
            assert len(entry) == 2, "box entries must be of length 2"
            for e in entry:
                assert isinstance(e, Number), "box entries must be numbers"
            assert entry[0] <= entry[1], "box entries must be non-decreasing"

        self._dim = len(box)
        self._box = np.array(box, dtype=np.float64)
        assert self._box.shape == (self._dim, 2), "converting box to n x 2 array failed"
        self._ranges = self._box.ptp(axis=1)  # self._box[:,1] - self._box[:,0]

        if dx:
            assert len(dx) == self._dim, "dx vector must have dimension n"
            dx = np.array(dx, dtype=np.float64)
            self._box -= dx

        self.dx = dx

    @property
    def dim(self):
        """
        The number of dimensions.
        """
        return self._dim

    @property
    def ranges(self):
        """
        The ranges along each dimension, a :class:`numpy.ndarray`.
        """
        return self._ranges

    @property
    def box(self):
        r"""
        The bounding box for this problem, a :math:`(\mathit{dim},2)`-:class:`array <numpy.ndarray>`.

        .. Note::

          This might change to a more sophisticated ``Box`` object.
        """
        return self._box

    def project(self, point):
        r"""
        projects given point into the search box.
        e.g. :math:`[-1.1, 1]` with box :math:`[(-1,1),(-1,1)]`
        gives :math:`[-1,1]`
        """
        assert isinstance(point, np.ndarray), 'point must be a numpy ndarray'
        return np.minimum(np.maximum(point, self.box[:, 0]), self.box[:, 1])

    def random_point(self):
        """
        generates a random point inside the given search box (ranges).
        """
        # uniformly
        return self._ranges * np.random.rand(self.dim) + self._box[:, 0]
        # TODO other distributions, too?

    def eval(self, x):
        """
        This is called to evaluate the given black-box function.
        The problem should be called directly (``__call__`` special function wraps this)
        and the given problem should subclass this ``eval`` method.

        :rtype: numpy.float64
        """
        raise Exception("You have to subclass and overwrite the eval function")

    def eval_constraints(self, x):
        """
        This method is optionally overwritten by the problem to calculate the constraint violations.
        It has to return a :class:`numpy.ndarray` of ``floats``.

        :rtype: numpy.ndarray
        """
        pass

    def __call__(self, point):
        x = point.x - self.dx if self.dx else point.x
        fx = self.eval(x)
        cv = self.eval_constraints(x)
        return Result(point, fx, cv_vec=cv)

    def __repr__(self):
        descr = "Problem '{}': {:d} dims, ".format(
            self.__class__.__name__, self._dim)
        p = filter(
            lambda _: not _[0].startswith("_"), self.__dict__.iteritems())
        descr += "params: %s, " % dict(p)
        descr += "box: [%s]" % ', '.join(
            '[%.2f %.2f]' % (l, u) for l, u in self._box)
        return descr
