from __future__ import division
from __future__ import unicode_literals
# -*- coding: utf8 -*-
# Copyright 2012-2026 Harald Schilly <harald.schilly@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from panobbgo.core import Heuristic

import numpy as np
from functools import cmp_to_key


class NelderMead(Heuristic):
    r"""
    This heuristic is inspired by the
    `Nelder Mead Method <http://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method>`_

    Algorithm:

    * If there are enough result points available, it tries to find a
      subset of ``dim + 1`` points, which are affinely independent (hence, span a simplex)
      and have the best (so far) function values, and are
      close (in the same :class:`Box <panobbgo.analyzers.Splitter>`).

    * Then, it applies the NM heuristic in a randomized fashion, i.e. it generates
      several promising points into the same direction as
      the implied search direction. See :meth:`here <.nelder_mead>`.

    The heuristic is *reactive*: it recomputes its base whenever the
    :class:`~panobbgo.analyzers.Splitter` publishes a new best box and tops
    its output queue up on every result batch.  It never sleeps or polls.
    """

    #: Reads Splitter boxes / subscribes to its events (installed on demand).
    requires_analyzers = ("Splitter",)

    def __init__(self, strategy):
        Heuristic.__init__(self, strategy, name="Nelder Mead")
        self.logger = self.config.get_logger("H:NM")
        self.best_box = None
        self._worst = None
        self._centroid = None

    def gram_schmidt(self, dim, results, tol=1e-4):
        """
        Pick ``dim + 1`` affinely independent :class:`Results <panobbgo.lib.Result>`
        to span a simplex, best first.

        The results are sorted best first; the best one is the simplex's
        origin, and each further result is accepted iff its offset
        ``p.x - first.x`` keeps a component above ``tol`` after Gram-Schmidt
        against the offsets accepted so far.  Working on offsets (affine
        independence) rather than on the absolute positions makes the
        choice translation-invariant: collinear points such as ``(1, 0),
        (1, 1), (1, 2)`` never pass as a basis, wherever the origin lies.
        Returns ``None`` if there are not enough such points.  The actual
        basis is not important, only the points for it are; they are used
        in :meth:`~.nelder_mead`.
        """
        if len(results) < dim + 1:
            return None

        def compare(a, b):
            # Sort using constraint_handler.is_better logic to prioritize feasible points
            if self.strategy.constraint_handler.is_better(b, a):
                return -1
            elif self.strategy.constraint_handler.is_better(a, b):
                return 1
            else:
                return 0

        results = sorted(results, key=cmp_to_key(compare))

        first = results.pop(0)
        ret = [first]  # the simplex's vertices, best first
        base = []  # orthogonal basis of the accepted offsets
        base_norms_sq = []
        for p in results:
            d = np.asarray(p.x, dtype=float) - first.x
            w = d.copy()
            for v, v_norm_sq in zip(base, base_norms_sq):
                w -= (v.dot(d) / v_norm_sq) * v
            if np.any(np.abs(w) > tol):
                base.append(w)
                base_norms_sq.append(w.dot(w))
                ret.append(p)
                if len(ret) >= dim + 1:
                    return ret
        return None

    def nelder_mead_init(self, base):
        """
        Calculates the worst point and the centroid of the remaining points
        from the given base.
        """
        get_val = self.strategy.constraint_handler.get_penalty_value

        def compare(a, b):
            if self.strategy.constraint_handler.is_better(b, a):
                return -1
            elif self.strategy.constraint_handler.is_better(a, b):
                return 1
            return 0

        # Find the worst point (last in sorted list)
        sorted_base = sorted(enumerate(base), key=cmp_to_key(lambda x, y: compare(x[1], y[1])))
        worst_idx, worst = sorted_base[-1]

        others = [p for i, p in enumerate(base) if i != worst_idx]
        others_x = [p.x for p in others]

        # Calculate weights based on penalty values
        worst_val = get_val(worst)
        vals = [get_val(r) for r in others]
        weights = []
        for v in vals:
            # Use absolute difference to robustly handle cases where worst point might have lower penalty
            # (e.g. if constraint handler prioritizes feasibility over penalty magnitude)
            diff = worst_val - v
            weights.append(np.log1p(abs(diff)))

        if not weights or not np.all(np.isfinite(weights)) or np.sum(weights) < 1e-4:
            weights = None  # fall back to normal average (also for an infinite penalty)

        # Calculate centroid of other points
        centroid = np.average(others_x, axis=0, weights=weights)
        return worst, centroid

    def nelder_mead_sample(self, worst, centroid, scale=3, offset=0):
        """
        Generates a new randomized search point based on worst point and centroid.
        """
        factor = self.rng.rayleigh(scale=scale) - offset
        return worst.x + factor * (centroid - worst.x)

    def nelder_mead(self, base, scale=3, offset=0):
        """
        Retuns a new *randomized* search point for the given set of results (``base``),
        which are linearly independent enough to form a orthonormal base,
        using the Nelder-Mead Method.

        Optional Arguments:

        - ``scale``: Used when sampling the new points via the :func:`~numpy.random.rayleigh` method.
        - ``offset``: This is subtracted from the sample factor; i.e. negative
          values account for the "contraction".
        """
        worst, centroid = self.nelder_mead_init(base)
        return self.nelder_mead_sample(worst, centroid, scale, offset)

    def _refresh_base(self) -> None:
        """Derive ``worst`` / ``centroid`` from the current best box.

        Walks up the box hierarchy until :meth:`gram_schmidt` finds enough
        linearly independent results; leaves ``_worst`` / ``_centroid`` as
        ``None`` when there are not enough.
        """
        dim = self.problem.dim
        bb = self.best_box
        while bb is not None:
            base = self.gram_schmidt(dim, bb.results)
            if base:
                self._worst, self._centroid = self.nelder_mead_init(base)
                return
            bb = bb.parent
        self._worst = self._centroid = None

    def _fill(self) -> None:
        """Top the output queue up with samples from the current base."""
        if self._worst is None or self._centroid is None:
            return
        self.fill_queue(lambda: self.nelder_mead_sample(self._worst, self._centroid))

    def on_new_best_box(self, best_box):
        """A new best box (from the :class:`~.analyzers.Splitter`) resets the
        search direction: the queue is flushed and refilled from the new base."""
        self.best_box = best_box
        self.clear_output()
        self._refresh_base()
        self._fill()

    def on_new_results(self, results):
        """Keep the queue topped up; the base only changes with the best box."""
        self._fill()

    def on_restart(self, center, reason):
        """Flush points and pause until a new best box is found."""
        self.clear_output()
        self.best_box = None
        self._worst = self._centroid = None
