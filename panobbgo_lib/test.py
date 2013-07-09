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
import numpy as np

from classic import *
from lib import *


def Disturbance(dim, nb=10, sd=.001, minimum=0.0001):
    r"""
    Returns a generator.

    It generates ``nb`` many ``dim`` sized "disturbance" vectors,
    which are based on :func:`~numpy.random.randn` and
    have a minimum value of ``minimum``.
    ``sd`` is the standard deviation of the generated numbers.
    """
    while nb > 0:
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

    def test_disturbance(self):
        d = list(Disturbance(3))
        assert all(len(_) == 3 for _ in d)
        assert all(np.all(np.abs(_) > 0.0001) for _ in d)


class Classics(unittest.TestCase):

    def setUp(self):
        pass

    def is_optimal(self, func, opt_pt):
        r'''
        This heuristic disturbs a known optimal point by a random
        :func:`~.Disturbance` to assess, if the given ``opt_pt``
        is really optimal.
        '''
        dim = func.dim
        opt_val = func.eval(opt_pt)
        self.assertTrue(
            all(func.eval(opt_pt + d) > opt_val for d in Disturbance(dim)))

    def test_rosenbrock(self):
        dim = 3
        rbrk = Rosenbrock(dim)
        opt_pt = np.array([1.] * dim)
        self.is_optimal(rbrk, opt_pt)

if __name__ == '__main__':
    unittest.main()
