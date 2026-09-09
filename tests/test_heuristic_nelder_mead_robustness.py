# -*- coding: utf8 -*-
"""Behavioural tests for the reactive NelderMead heuristic."""

import types
import unittest
from unittest.mock import Mock

import numpy as np

from panobbgo.heuristics.nelder_mead import NelderMead


class TestNelderMeadReactive(unittest.TestCase):
    def setUp(self):
        self.strategy = Mock()
        self.strategy.config = Mock()
        self.strategy.config.get_logger.return_value = Mock()
        self.strategy.config.capacity = 6
        self.strategy.problem = Mock()
        self.strategy.problem.dim = 2
        self.strategy.spawn_rng = lambda: np.random.default_rng(0)

        self.heuristic = NelderMead(self.strategy)
        self.heuristic.emit = Mock()
        self.heuristic.clear_output = Mock()

    def _box(self, base, parent=None):
        return types.SimpleNamespace(results=["r"] * 3, parent=parent)

    def test_no_base_no_emission(self):
        self.heuristic.gram_schmidt = Mock(return_value=None)
        self.heuristic.on_new_best_box(self._box(None))
        self.heuristic.emit.assert_not_called()
        self.heuristic.on_new_results([])
        self.heuristic.emit.assert_not_called()

    def test_new_best_box_flushes_and_fills_queue(self):
        self.heuristic.gram_schmidt = Mock(return_value=["a", "b", "c"])
        worst = types.SimpleNamespace(x=np.zeros(2))
        self.heuristic.nelder_mead_init = Mock(return_value=(worst, np.ones(2)))

        self.heuristic.on_new_best_box(self._box(None))

        self.heuristic.clear_output.assert_called_once()
        self.heuristic.emit.assert_called_once()
        points = self.heuristic.emit.call_args[0][0]
        self.assertEqual(len(points), 6)  # capacity

    def test_walks_up_to_parent_box(self):
        parent = self._box(None)
        child = self._box(None, parent=parent)
        calls = []

        def gs(dim, results):
            calls.append(results)
            return ["a", "b"] if len(calls) == 2 else None

        self.heuristic.gram_schmidt = gs
        self.heuristic.nelder_mead_init = Mock(return_value=(types.SimpleNamespace(x=np.zeros(2)), np.ones(2)))
        self.heuristic.on_new_best_box(child)
        self.assertEqual(len(calls), 2)
        self.heuristic.emit.assert_called_once()

    def test_restart_pauses_until_next_best_box(self):
        self.heuristic.gram_schmidt = Mock(return_value=["a", "b", "c"])
        self.heuristic.nelder_mead_init = Mock(return_value=(types.SimpleNamespace(x=np.zeros(2)), np.ones(2)))
        self.heuristic.on_new_best_box(self._box(None))
        self.heuristic.emit.reset_mock()

        self.heuristic.on_restart(np.array([0.5, 0.5]), "stagnation")
        self.heuristic.on_new_results([])

        self.assertIsNone(self.heuristic.best_box)
        self.heuristic.emit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
