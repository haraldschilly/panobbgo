from panobbgo.utils import PanobbgoTestCase


class HeuristicTests(PanobbgoTestCase):

    def test_weighted_average(self):
        from panobbgo.heuristics import WeightedAverage
        avg = WeightedAverage()
        strategy = self.get_strategy(avg)
        assert avg is not None

    def test_random(self):
        from panobbgo.heuristics import Random
        rnd = Random()
        strategy = self.get_strategy(rnd)

        assert rnd is not None

    def test_latin_hypercube(self):
        from panobbgo.heuristics import LatinHypercube
        lhyp = LatinHypercube(3)
        strategy = self.get_strategy(lhyp)

        assert lhyp is not None
