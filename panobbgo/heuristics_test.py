from panobbgo.utils import PanobbgoTestCase


class MockupEventBus(object):
    def __init__(self):
        self.targets = []

    def register(self, who):
        self.targets.append(who)


class MockupStrategy(object):

    def __init__(self, problem):
        self.problem = problem
        self._eventbus = MockupEventBus()

    @property
    def eventbus(self):
        return self._eventbus


class HeuristicTests(PanobbgoTestCase):

    def setUp(self):
        from panobbgo_lib.classic import Rosenbrock
        self.problem = Rosenbrock(2)
        self.strategy = MockupStrategy(self.problem)

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
