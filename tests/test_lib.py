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
import unittest
import numpy as np

from panobbgo.lib.classic import *
from panobbgo.lib import *
from panobbgo.utils import expected_failure


def Disturbance(dim, nb=10, sd=0.001, minimum=0.0001):
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
        x = np.array([5, -2.2, 0, 1.1], dtype=float)
        p = Point(x, "test")
        self.assertEqual(p.who, "test")
        np.testing.assert_array_equal(p.x, x)
        self.assertEqual(repr(p), "[ 5.  -2.2  0.   1.1] by test")

    @expected_failure(
        Exception,
        "who needs to be a string describing the heuristic, "
        "was 0 of type <class 'int'>",
    )
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
        p = Point([1.0]*4, "nose")
        assert rbrk(p).fx == 0.0
        assert np.allclose(
            rbrk.project(np.array([22, 0, -33, 2], dtype=np.float64)),
            [2.0, 0.0, -2.0, 2.0],
        )
        assert rbrk.dx is None
        assert np.allclose(rbrk.ranges, [2.0, 4.0, 4.0, 4.0])

        # with dx
        rbrk = Rosenbrock(2, dx=[2.41, 3.14])
        assert np.allclose(rbrk.box.box, [[2.41, 5.14], [0.41, 5.14]])
        p2 = Point([1.0, 1.0], "nose")
        assert rbrk(p2).fx > 5000.0
        p = Point([-1.41, -2.14], "nose")
        assert np.isclose(rbrk(p).fx, 0.0)
        rp = rbrk.random_point()
        assert np.all(rbrk.box[:, 0] <= rp)
        assert np.all(rbrk.box[:, 1] >= rp)
        r = repr(rbrk)  # ordering of dict is arbitrary, hence this:
        assert "Problem 'Rosenbrock': 2 dims, params: " in r
        assert "'par1': 100" in r
        assert "'dx': array([2.41, 3.14])" in r
        assert "box: [[2.41 5.14], [0.41 5.14]]" in r

    @expected_failure(ValueError, "assignment destination is read-only")
    def test_problem_dx_assign(self):
        rbrk = Rosenbrock(2, dx=[2.41, 3.14])
        rbrk.dx[0] = 1.0

    @expected_failure(ValueError, "assignment destination is read-only")
    def test_problem_box_assign(self):
        rbrk = Rosenbrock(2, dx=[2.41, 3.14])
        rbrk._box[0, 0] = 0.0

    @expected_failure(ValueError, "assignment destination is read-only")
    def test_problem_ranges_assign(self):
        rbrk = Rosenbrock(2, dx=[2.41, 3.14])
        rbrk.ranges[0] = 1.0

    @expected_failure(ValueError, "point must be an instance of lib.Point")
    def test_result_error(self):
        Result([1.0, 1.0], 1.1)

    def test_result(self):
        r0 = Result(Point([1.0, 1.0], "nose"), 1.0)
        assert r0.who == "nose"
        assert r0.cv == 0.0
        r = Result(
            Point([1.0, 1.0], "nose"),
            1.1,
            cv_vec=np.array([2.0, 3.0, -1.0]),
            error=1e-5,
        )
        assert r.fx == 1.1
        assert np.allclose(r.x, [1.0, 1.0])
        assert np.allclose(r.cv_vec, [2.0, 3.0, -1])
        assert np.allclose(r.cv, np.sqrt(13))
        assert np.allclose(r.pp, [np.sqrt(13), 1.1])
        assert r.error == 1e-5
        assert r0 < r
        assert str(r) == "   1.100000 \u22db  3.6056 @ [   1.000000    1.000000]"


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
        self.assertTrue(all(func.eval(opt_pt + d) > opt_val for d in Disturbance(dim)))

    def test_rosenbrock(self):
        dim = 3
        rbrk = Rosenbrock(dim)
        opt_pt = np.array([1.0] * dim)
        self.is_optimal(rbrk, opt_pt)

    def test_rosenbrock_shifted_optimum_vector(self):
        """Test Rosenbrock with shifted optimum specified as a vector."""
        optimum = [5.0, 3.0]
        rbrk = Rosenbrock(optimum=optimum, box=[(-10, 10), (-10, 10)])

        # Verify optimum location
        np.testing.assert_array_equal(rbrk.optimum, optimum)
        self.assertEqual(rbrk.dim, 2)

        # Verify function value at optimum is 0
        opt_pt = np.array(optimum)
        self.assertAlmostEqual(rbrk.eval(opt_pt), 0.0, places=10)

        # Verify it's a minimum
        self.is_optimal(rbrk, opt_pt)

    def test_rosenbrock_shifted_optimum_scalar(self):
        """Test Rosenbrock with scalar optimum (broadcast to all dims)."""
        rbrk = Rosenbrock(dims=3, optimum=2.0, box=[(-5, 5)] * 3)

        # Verify optimum is broadcast
        np.testing.assert_array_equal(rbrk.optimum, [2.0, 2.0, 2.0])

        # Verify function value at optimum is 0
        opt_pt = np.array([2.0, 2.0, 2.0])
        self.assertAlmostEqual(rbrk.eval(opt_pt), 0.0, places=10)

        # Verify it's a minimum
        self.is_optimal(rbrk, opt_pt)

    def test_rosenbrock_dims_inferred_from_optimum(self):
        """Test that dims is correctly inferred from optimum vector."""
        optimum = [24.0, -12.0, 5.0]
        rbrk = Rosenbrock(optimum=optimum, box=[(-100, 100)] * 3)

        self.assertEqual(rbrk.dim, 3)
        np.testing.assert_array_equal(rbrk.optimum, optimum)
        self.assertAlmostEqual(rbrk.eval(np.array(optimum)), 0.0, places=10)

    def test_rosenbrock_validation_errors(self):
        """Test validation errors for invalid parameter combinations."""
        # dims required when optimum is scalar
        with self.assertRaises(ValueError):
            Rosenbrock(optimum=5.0)

        # optimum length must match dims
        with self.assertRaises(ValueError):
            Rosenbrock(dims=3, optimum=[1.0, 2.0])

    def test_rosenbrock_constraint_shifted_optimum(self):
        """Test RosenbrockConstraint with shifted optimum."""
        optimum = [3.0, 2.0]
        rbrk = RosenbrockConstraint(optimum=optimum, box=[(-5, 5), (-5, 5)])

        # Verify optimum location
        np.testing.assert_array_equal(rbrk.optimum, optimum)
        self.assertEqual(rbrk.dim, 2)

        # The shifted function value should be consistent
        # (Note: RosenbrockConstraint subtracts 50 from the objective)
        opt_pt = np.array(optimum)
        self.assertAlmostEqual(rbrk.eval(opt_pt), -50.0, places=10)

    def test_rosenbrock_backward_compatibility(self):
        """Test that existing code using dims positional arg still works."""
        # Old-style call: Rosenbrock(dims)
        rbrk = Rosenbrock(2)
        self.assertEqual(rbrk.dim, 2)
        np.testing.assert_array_equal(rbrk.optimum, [1.0, 1.0])

        # Old-style with par1
        rbrk2 = Rosenbrock(3, par1=50)
        self.assertEqual(rbrk2.dim, 3)
        self.assertEqual(rbrk2.par1, 50)

    def test_helicalvalley(self):
        assert np.isclose(0.0, HelicalValley._theta(1, 0))
        hv = HelicalValley()
        x0 = np.array([1.0, 0.0, 0.0])
        x = Point(x0, "test")
        r = hv(x)
        self.is_optimal(hv, x0)
        assert np.isclose(0.0, hv(x).fx)

    def test_nesterovquadratic(self):
        np.random.seed(1)
        A = np.random.randn(10, 10)
        b = np.random.randn(10)
        x = np.random.randn(10)
        x = Point(x, "test")
        nq = NesterovQuadratic(A=A, b=b)
        assert np.isfinite(nq(x).fx)

    def test_arwhead(self):
        x = Point(np.arange(10) - 5, "test")
        arwhead = Arwhead(dim=10)

        # slow sum
        sum = 0
        for i in range(9):
            sum += (x[i] ** 2 + x[9] ** 2) ** 2 - 4 * x[i] + 3

        assert np.isclose(arwhead(x).fx, sum)


if __name__ == "__main__":
    unittest.main()
