import unittest
import unittest.mock as mock
import numpy as np

from panobbgo.heuristics.zero import Zero

class TestZeroHeuristic(unittest.TestCase):
    def test_on_start(self):
        strategy_mock = mock.MagicMock()
        strategy_mock.problem.dim = 3

        with mock.patch('panobbgo.core.Heuristic.problem', new_callable=mock.PropertyMock) as prop_mock:
            prop_mock.return_value = strategy_mock.problem
            zero_heuristic = Zero(strategy_mock)

            result = zero_heuristic.on_start()

            self.assertEqual(len(result), 3)
            np.testing.assert_array_equal(result, np.array([0.0, 0.0, 0.0]))
