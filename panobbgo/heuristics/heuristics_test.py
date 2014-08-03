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
        box = cntr.on_start()
        assert np.allclose(box, [1, 0.])
