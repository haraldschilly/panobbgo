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
import unittest
import nose
import numpy as np

from classic import *
from lib import *
from panobbgo.utils import expected_failure


def Disturbance(dim, nb=10, sd=.001, minimum=0.0001):
    r"""
    A generator for random vectors.

    It generates ``nb`` many ``dim`` sized "disturbance" vectors,
    based on :func:`~numpy.random.randn` and
    have a minimum value of ``minimum``.

    :param int dim: dimension of the disturbance vector.
    :param nb: a positive integer or None (default: 10).
               If set to None, the generator will run infinitely.
    :param double sd: the standard deviation of the generated numbers
    :param double minimum: minimum value for each entry.
    """
    while nb is None or nb > 0:
        x = np.random.randn(dim) * float(sd)
        if np.any(np.abs(x) < minimum):
            continue
        yield x
        nb -= 1


class Lib(unittest.TestCase):

    def test_point(self):
        x = np.array([5, -2.2, 0, 1.1], dtype=np.float)
        p = Point(x, 'test')
        self.assertEqual(p.who, 'test')
        np.testing.assert_array_equal(p.x, x)
        self.assertEqual(repr(p), '[ 5.  -2.2  0.   1.1] by test')

    @expected_failure(Exception, "who needs to be a string describing the heuristic, "
                                 "was 0 of type <type 'int'>")
    def test_point_who(self):
        x = np.array([5, -2.2, 0, 1.1])
        Point(x, 0)

    def test_disturbance(self):
        d = list(Disturbance(3))
        assert all(len(_) == 3 for _ in d)
        assert all(np.all(np.abs(_) > 0.0001) for _ in d)

    def test_problem(self):
        rbrk = Rosenbrock(4)
        assert rbrk.dim == 4
        p = Point([1., 1.], "nose")
        assert rbrk(p).fx == 0.
        assert np.allclose(
            rbrk.project(np.array([22, 0, -33, 2], dtype=np.float64)),
            [2., 0., - 2., 2.])
        assert rbrk.dx is None
        assert np.allclose(rbrk.ranges, [2., 4., 4., 4.])

        # with dx
        rbrk = Rosenbrock(2, dx=[2.41, 3.14])
        assert np.allclose(rbrk.box,
                           [[2.41, 5.14],
                            [0.41, 5.14]])
        assert rbrk(p).fx > 5000.
        p = Point([-1.41, -2.14], "nose")
        assert np.isclose(rbrk(p).fx, 0.)
        rp = rbrk.random_point()
        assert np.all(rbrk.box[:, 0] <= rp)
        assert np.all(rbrk.box[:, 1] >= rp)
        assert repr(rbrk) == "Problem 'Rosenbrock': 2 dims, " \
                             "params: {'par1': 100, 'dx': array([ 2.41,  3.14])}, " \
                             "box: [[2.41 5.14], [0.41 5.14]]"

    @expected_failure(ValueError, "assignment destination is read-only")
    def test_problem_dx_assign(self):
        rbrk = Rosenbrock(2, dx=[2.41, 3.14])
        rbrk.dx[0] = 1.

    @expected_failure(ValueError, "assignment destination is read-only")
    def test_problem_box_assign(self):
        rbrk = Rosenbrock(2, dx=[2.41, 3.14])
        rbrk._box[0, 0] = 0.

    @expected_failure(ValueError, "assignment destination is read-only")
    def test_problem_ranges_assign(self):
        rbrk = Rosenbrock(2, dx=[2.41, 3.14])
        rbrk._ranges[0] = 1.

    @expected_failure(ValueError, "point must be an instance of lib.Point")
    def test_result_error(self):
        Result([1., 1.], 1.1)

    def test_result(self):
        r0 = Result(Point([1., 1.], "nose"), 1.0)
        assert r0.who == "nose"
        assert r0.cv == 0.
        r = Result(Point([1., 1.], "nose"), 1.1,
                   cv_vec=np.array([2., 3., -1.]),
                   error=1e-5)
        assert r.fx == 1.1
        assert np.allclose(r.x, [1., 1.])
        assert np.allclose(r.cv_vec, [2., 3., -1])
        assert np.allclose(r.cv, np.sqrt(13))
        assert np.allclose(r.pp, [np.sqrt(13), 1.1])
        assert r.error == 1e-5
        assert r0 < r
        assert unicode(r) == u"   1.100000 \u22DB  3.6056 @ [   1.000000    1.000000]"


class Classics(unittest.TestCase):

    def setUp(self):
        pass

    def is_optimal(self, func, opt_pt):
        r"""
        This heuristic disturbs a known optimal point by a random
        :func:`~.Disturbance` to assess, if the given ``opt_pt``
        is really optimal.
        """
        dim = func.dim
        opt_val = func.eval(opt_pt)
        self.assertTrue(
            all(func.eval(opt_pt + d) > opt_val for d in Disturbance(dim)))

    def test_rosenbrock(self):
        dim = 3
        rbrk = Rosenbrock(dim)
        opt_pt = np.array([1.] * dim)
        self.is_optimal(rbrk, opt_pt)

    def test_helicalvalley(self):
        assert np.isclose(0., HelicalValley._theta(1, 0))
        hv = HelicalValley()
        x0 = np.array([1.0, 0., 0.])
        x = Point(x0, "test")
        r = hv(x)
        self.is_optimal(hv, x0)
        assert np.isclose(0., hv(x).fx)

    def test_nesterovquadratic(self):
        np.random.seed(1)
        A = np.random.randn(10, 10)
        b = np.random.randn(10)
        x = np.random.randn(10)
        x = Point(x, "test")
        nq = NesterovQuadratic(A=A, b=b)
        assert np.isclose(nq(x).fx, 45.961712020840039)

    def test_arwhead(self):
        x = Point(np.arange(10) - 5, "test")
        arwhead = Arwhead(dim=10)

        # slow sum
        sum = 0
        for i in range(9):
            sum += (x[i] ** 2 + x[9] ** 2) ** 2 - 4 * x[i] + 3

        assert np.isclose(arwhead(x).fx, sum)


if __name__ == '__main__':
    unittest.main()
