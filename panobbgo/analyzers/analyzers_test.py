from panobbgo.utils import PanobbgoTestCase


class AnalyzersUtils(PanobbgoTestCase):

    def test_best(self):
        from panobbgo.analyzers import Best
        best = Best(self.strategy)
        assert best is not None

if __name__ == '__main__':
    import unittest
    unittest.main()
