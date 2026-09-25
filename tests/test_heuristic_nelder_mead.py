# -*- coding: utf8 -*-
"""NelderMead against real results: base finding and sampling end to end."""

from unittest import mock

import numpy as np

from panobbgo.heuristics.nelder_mead import NelderMead
from tests.support import PanobbgoTestCase


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

    def _results(self, xs, fxs=None):
        from panobbgo.lib import Point, Result

        fxs = range(len(xs)) if fxs is None else fxs
        return [Result(Point(np.array(x, dtype=float), "t"), float(f)) for x, f in zip(xs, fxs)]

    def test_collinear_points_are_no_simplex(self):
        """Regression: absolute positions were orthogonalised, so collinear points off the origin passed."""
        nm = NelderMead(self.strategy)
        assert nm.gram_schmidt(2, self._results([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])) is None

    def test_base_is_a_full_simplex_and_translation_invariant(self):
        """dim + 1 affinely independent vertices, best first; shifting every point shifts nothing else."""
        from panobbgo.lib.constraints import DefaultConstraintHandler

        self.strategy.constraint_handler = DefaultConstraintHandler(self.strategy, rho=100.0)  # a real best-first sort
        nm = NelderMead(self.strategy)
        xs = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0]]
        base = nm.gram_schmidt(2, self._results(xs))
        assert base is not None and len(base) == 3
        assert [list(r.x) for r in base] == [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
        offsets = np.array([r.x - base[0].x for r in base[1:]])
        assert np.linalg.matrix_rank(offsets) == 2

        shift = np.array([5.0, -3.0])
        shifted = nm.gram_schmidt(2, self._results([np.array(x) + shift for x in xs]))
        assert [list(r.x - shift) for r in shifted] == [list(r.x) for r in base]
