import unittest
import unittest.mock as mock
import numpy as np

from panobbgo.heuristics.center import Center


class TestCenterHeuristic(unittest.TestCase):
    def test_on_start(self):
        strategy_mock = mock.MagicMock()
        strategy_mock.problem.box = np.array([[-1.0, 1.0], [0.0, 2.0], [1.0, 5.0]])

        with mock.patch("panobbgo.core.Heuristic.problem", new_callable=mock.PropertyMock) as prop_mock:
            prop_mock.return_value = strategy_mock.problem
            center_heuristic = Center(strategy_mock)

            result = center_heuristic.on_start()

            self.assertEqual(len(result), 3)
            np.testing.assert_array_equal(result, np.array([0.0, 1.0, 3.0]))
