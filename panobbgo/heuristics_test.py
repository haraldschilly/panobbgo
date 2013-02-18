from panobbgo.utils import PanobbgoTestCase


class HeuristicTests(PanobbgoTestCase):

    def test_weighted_average(self):
        from panobbgo.heuristics import WeightedAverage
        avg = WeightedAverage()
        avg._init_module(self.strategy)

        assert avg is not None

    def test_random(self):
        from panobbgo.heuristics import Random
        rnd = Random()
        rnd._init_module(self.strategy)

        assert rnd is not None
