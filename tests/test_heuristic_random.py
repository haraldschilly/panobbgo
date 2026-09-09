# -*- coding: utf8 -*-
"""Tests for the reactive Random heuristic.

Random fills its output queue on ``start`` and tops it up on every result
batch / best-box change.  It never sleeps or polls, so its behaviour is a
pure function of ``self.rng`` and the events it receives.
"""

from __future__ import unicode_literals

import types
from unittest import mock

import numpy as np

from panobbgo.utils import PanobbgoTestCase


def _leaf(lo, hi):
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    return types.SimpleNamespace(ranges=hi - lo, box=np.column_stack([lo, hi]))


class RandomHeuristicTest(PanobbgoTestCase):
    def make_random(self, **kw):
        from panobbgo.heuristics import Random

        return Random(self.strategy, **kw)

    def test_no_splitter_samples_full_box(self):
        """No Splitter analyzer → sampling from the full problem box, queue filled to capacity."""
        h = self.make_random(cap=8)
        self.strategy.analyzer.side_effect = Exception("no analyzer registered")

        h.on_start()

        assert h.leaf is None
        points = h.get_points()
        assert len(points) == 8
        for p in points:
            assert np.all(p.x >= self.problem.box[:, 0]) and np.all(p.x <= self.problem.box[:, 1])

    def test_samples_from_root_leaf_box(self):
        """With a root leaf available, points come from the leaf's box."""
        h = self.make_random(cap=8)
        leaf = _leaf([0.0, 0.0], [0.25, 0.25])
        self.strategy.analyzer.side_effect = None
        self.strategy.analyzer.return_value = types.SimpleNamespace(root=leaf, dim=2)

        h.on_start()

        assert h.leaf is leaf
        points = h.get_points()
        assert len(points) == 8
        for p in points:
            assert np.all(p.x >= 0.0) and np.all(p.x <= 0.25), f"point {p.x} outside leaf box"

    def test_tops_up_after_results(self):
        h = self.make_random(cap=6)
        self.strategy.analyzer.side_effect = Exception("no analyzer")
        h.on_start()
        assert len(h.get_points(4)) == 4
        assert h._output.qsize() == 2

        h.on_new_results([])

        assert h._output.qsize() == 6

    def test_new_best_box_switches_leaf_and_refills(self):
        h = self.make_random(cap=4)
        self.strategy.analyzer.side_effect = Exception("no analyzer")
        h.on_start()
        h.get_points()
        leaf = _leaf([1.0, 1.0], [1.5, 1.5])

        h.on_new_best_box(best_box=leaf)

        assert h.leaf is leaf
        points = h.get_points()
        assert len(points) == 4
        for p in points:
            assert np.all(p.x >= 1.0) and np.all(p.x <= 1.5)

    def test_on_new_split_tracks_best_leaf(self):
        h = self.make_random(cap=4)
        leaf = _leaf([0.0, 0.0], [0.5, 0.5])
        best = types.SimpleNamespace(x=np.array([0.5, 0.5]))

        def analyzer(name):
            if name == "Best":
                return types.SimpleNamespace(best=best)
            return types.SimpleNamespace(get_leaf=mock.Mock(return_value=leaf))

        self.strategy.analyzer.side_effect = analyzer
        h.on_new_split(box=None, children=[], dim=0)

        assert h.leaf is leaf
        assert h._output.qsize() == 4

    def test_on_restart_resets_to_root_leaf(self):
        h = self.make_random(cap=4)
        root = _leaf([-1.0, -1.0], [1.0, 1.0])
        self.strategy.analyzer.side_effect = None
        self.strategy.analyzer.return_value = types.SimpleNamespace(root=root)
        h.leaf = _leaf([0.0, 0.0], [0.1, 0.1])

        h.on_restart(np.array([0.5, 0.5]), "stagnation")

        assert h.leaf is root
        assert h._output.qsize() == 4

    def test_same_rng_seed_same_points(self):
        self.strategy.analyzer.side_effect = Exception("no analyzer")
        a, b = self.make_random(cap=5), self.make_random(cap=5)
        a.rng = np.random.default_rng(7)
        b.rng = np.random.default_rng(7)
        a.on_start()
        b.on_start()

        xa = [p.x for p in a.get_points()]
        xb = [p.x for p in b.get_points()]
        assert len(xa) == 5
        np.testing.assert_array_equal(np.array(xa), np.array(xb))
