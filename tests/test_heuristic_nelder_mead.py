# -*- coding: utf8 -*-
"""NelderMead against real results: base finding and sampling end to end."""

from unittest import mock

import numpy as np

from panobbgo.heuristics.nelder_mead import NelderMead
from panobbgo.utils import PanobbgoTestCase


class TestHeuristicNelderMead(PanobbgoTestCase):
    def _box(self, n=10):
        box = mock.MagicMock()
        box.results = self.random_results(self.problem.dim, n)
        box.parent = None
        return box

    def test_new_best_box_emits_from_real_base(self):
        nm = NelderMead(self.strategy)
        emitted = []
        nm.emit = lambda pts: emitted.extend(pts)

        nm.on_new_best_box(self._box())

        assert nm._worst is not None and nm._centroid is not None
        assert len(emitted) == nm.cap
        for p in emitted:
            assert p.shape == (self.problem.dim,)

    def test_results_top_up_queue(self):
        nm = NelderMead(self.strategy)
        nm.on_new_best_box(self._box())
        drained = nm.get_points(3)
        assert len(drained) == 3

        nm.on_new_results([])

        assert nm._output.qsize() == nm.cap

    def test_too_few_results_means_no_base(self):
        nm = NelderMead(self.strategy)
        nm.emit = mock.Mock()

        nm.on_new_best_box(self._box(n=1))

        assert nm._worst is None
        nm.emit.assert_not_called()

    def test_samples_are_seeded(self):
        a, b = NelderMead(self.strategy), NelderMead(self.strategy)
        a.rng, b.rng = np.random.default_rng(3), np.random.default_rng(3)
        box = self._box()
        a.on_new_best_box(box)
        b.on_new_best_box(box)
        xa = np.array([p.x for p in a.get_points()])
        xb = np.array([p.x for p in b.get_points()])
        np.testing.assert_array_equal(xa, xb)
