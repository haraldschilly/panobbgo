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
from __future__ import unicode_literals

from panobbgo.utils import PanobbgoTestCase
import numpy as np


class HeuristicTests(PanobbgoTestCase):

    def test_weighted_average(self):
        from . import WeightedAverage
        avg = WeightedAverage(self.strategy)
        assert avg is not None

    def test_random(self):
        from . import Random
        rnd = Random(self.strategy)
        assert rnd is not None

    def test_latin_hypercube(self):
        from . import LatinHypercube
        lhyp = LatinHypercube(self.strategy, 3)
        assert lhyp is not None

    def test_nelder_mead(self):
        from . import NelderMead
        nm = NelderMead(self.strategy)
        assert nm is not None
        dim = 5
        pts = self.random_results(dim, 10)
        # make it ill conditioned
        pts.insert(0, pts[0])
        base = nm.gram_schmidt(dim, pts)
        M = np.array([_.x for _ in base])
        assert np.linalg.matrix_rank(M) == dim

    def test_center(self):
        from . import Center
        cntr = Center(self.strategy)
        assert cntr is not None
        #box = cntr.on_start()
        #assert np.allclose(box, [1, 0.])

    def test_extremal(self):
        from . import Extremal
        extr = Extremal(self.strategy, prob=range(10))
        assert isinstance(extr.probabilities, np.ndarray)
        assert np.isclose(sum(np.diff(extr.probabilities)), 1.)
        extr = Extremal(self.strategy)
        extr.__start__()

        # min, max, center, zero
        box = self.problem.box
        assert np.allclose(extr.vals[0], box[:, 0])
        assert np.allclose(extr.vals[1], np.zeros_like(extr.vals[1]))
        assert np.allclose(extr.vals[2], self.problem.center)
        assert np.allclose(extr.vals[3], box[:, 1])

        # simulate on_start, produces one point in testing mode
        extr.on_start()
        from Queue import Queue
        assert isinstance(extr._output, Queue)
        p = extr._output.get()
        from panobbgo_lib import Point
        assert isinstance(p, Point)
        assert p in self.problem.box
