from panobbgo.utils import PanobbgoTestCase
import numpy as np


class HeuristicTests(PanobbgoTestCase):

    def test_weighted_average(self):
        from panobbgo.heuristics import WeightedAverage
        avg = WeightedAverage()
        strategy = self.init_strategy(avg)
        assert avg is not None

    def test_random(self):
        from panobbgo.heuristics import Random
        rnd = Random()
        strategy = self.init_strategy(rnd)
        assert rnd is not None

    def test_latin_hypercube(self):
        from panobbgo.heuristics import LatinHypercube
        lhyp = LatinHypercube(3)
        strategy = self.init_strategy(lhyp)
        assert lhyp is not None

    def test_nelder_mead(self):
        from panobbgo.heuristics import NelderMead
        nm = NelderMead()
        strategy = self.init_strategy(nm)
        assert nm is not None
        dim = 5
        pts = self.random_results(dim, 10)
        # make it ill conditioned
        pts.insert(0, pts[0])
        base = nm.gram_schmidt(dim, pts)
        M = np.array([_.x for _ in base])
        assert np.linalg.matrix_rank(M) == dim