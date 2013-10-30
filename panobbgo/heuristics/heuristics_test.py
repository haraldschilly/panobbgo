from panobbgo.utils import PanobbgoTestCase


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
        dim = 3
        pts = self.random_results(3, 10)
        base = nm.gram_schmidt(dim, pts)
        