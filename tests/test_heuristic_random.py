# -*- coding: utf8 -*-
"""Branch tests for the Random heuristic's on_start paths.

Random's point-generation loops run on an eventbus thread and depend on the
Splitter analyzer's state; these tests drive `on_start` directly with scripted
splitter stand-ins and stop the loops via a timer flipping `_stopped`.
"""

from __future__ import unicode_literals

import threading
import types

import numpy as np

from panobbgo.utils import PanobbgoTestCase


def _stop_soon(h, delay=0.15):
    """Flip the heuristic's stop flag shortly after the loop starts."""
    threading.Timer(delay, lambda: setattr(h, "_stopped", True)).start()


class RandomHeuristicBranchTest(PanobbgoTestCase):
    def make_random(self):
        from panobbgo.heuristics import Random

        return Random(self.strategy)

    def test_no_splitter_falls_back_to_full_problem_space(self):
        """No Splitter analyzer at all → warning + sampling from the full box."""
        h = self.make_random()
        self.strategy.analyzer.side_effect = Exception("no analyzer registered")
        # Don't actually wait 5 s for a split that will never come.
        h.first_split.wait = lambda timeout=None: False

        _stop_soon(h)
        h.on_start()

        points = h.get_points()
        assert len(points) > 0, "fallback loop emitted no points"

    def test_split_available_but_no_leaf_falls_back(self):
        """first_split set but leaf is None → second fallback loop."""
        h = self.make_random()
        # Splitter without a `root` attribute → leaf stays None in on_start.
        self.strategy.analyzer.side_effect = None
        self.strategy.analyzer.return_value = object()
        h.first_split.set()

        _stop_soon(h)
        h.on_start()

        assert h.leaf is None
        points = h.get_points()
        assert len(points) > 0, "leafless fallback loop emitted no points"

    def test_samples_from_leaf_box(self):
        """With a root leaf available, points come from the leaf's box."""
        h = self.make_random()
        leaf = types.SimpleNamespace(
            ranges=np.array([1.0, 1.0]),
            box=np.array([[0.0, 1.0], [0.0, 1.0]]),
        )
        splitter = types.SimpleNamespace(root=leaf, dim=2)
        self.strategy.analyzer.side_effect = None
        self.strategy.analyzer.return_value = splitter

        _stop_soon(h)
        h.on_start()

        assert h.leaf is leaf
        points = h.get_points()
        assert len(points) > 0, "leaf-based loop emitted no points"
        for p in points:
            assert np.all(p.x >= 0.0) and np.all(p.x <= 1.0), f"point {p.x} outside leaf box"

    def test_on_restart_resets_to_root_leaf(self):
        h = self.make_random()
        root = object()
        self.strategy.analyzer.side_effect = None
        self.strategy.analyzer.return_value = types.SimpleNamespace(root=root)

        h.first_split.clear()
        h.on_restart(np.array([0.5, 0.5]), "stagnation")

        assert h.leaf is root
        assert h.first_split.is_set()


class RandomNewSplitTest(PanobbgoTestCase):
    def test_on_new_split_tracks_best_leaf(self):
        from unittest import mock

        from panobbgo.heuristics import Random

        h = Random(self.strategy)
        leaf = object()
        best = types.SimpleNamespace(x=np.array([0.5, 0.5]))

        def analyzer(name):
            if name == "Best":
                return types.SimpleNamespace(best=best)
            return types.SimpleNamespace(get_leaf=mock.Mock(return_value=leaf))

        self.strategy.analyzer.side_effect = analyzer
        h.first_split.clear()
        h.on_new_split(box=None, children=[], dim=0)

        assert h.leaf is leaf
        assert h.first_split.is_set()
