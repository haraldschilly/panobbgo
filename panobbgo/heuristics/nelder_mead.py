from __future__ import division
from __future__ import unicode_literals
# -*- coding: utf8 -*-
# Copyright 2012 Harald Schilly <harald.schilly@gmail.com>
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
      subset of points, which are linear independent (hence, suiteable for NM)
      and have the best (so far) function values, and are
      close (in the same :class:`Box <panobbgo.analyzers.Splitter>`).

    * Then, it applies the NM heuristic in a randomized fashion, i.e. it generates
      several promising points into the same direction as
      the implied search direction. See :meth:`here <.nelder_mead>`.

    The heuristic is *reactive*: it recomputes its base whenever the
    :class:`~panobbgo.analyzers.Splitter` publishes a new best box and tops
    its output queue up on every result batch.  It never sleeps or polls.
    """

    def __init__(self, strategy):
        Heuristic.__init__(self, strategy, name="Nelder Mead")
        self.logger = self.config.get_logger("H:NM")
        self.best_box = None
        self._worst = None
        self._centroid = None

    def gram_schmidt(self, dim, results, tol=1e-4):
        """
        Tries to calculate an orthogonal base of dimension `dim`
        with given list of :class:`Results <panobbgo.lib.Result>` points.
        Retuns `None`, if not enough points or impossible.
        The actual basis is not important, only the points for it are.
        They are used in :meth:`~.nelder_mead`.
        """
        # start empty, and append in each iteration
        # sort points ascending by fx -> calc gs -> skip if <= tol
        import numpy as np

        base = []  # orthogonal system basis
        ret = []  # list of results, which will be returned
        if len(results) < dim:
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

        # better? randomize results to diversify
        # from random import shuffle
        # shuffle(results)
        first = results.pop(0)
        base.append(first.x)
        # Cache squared norms of basis vectors
        base_norms_sq = [first.x.dot(first.x)]
        ret.append(first)
        for p in results:
            # Avoid division by zero or near-zero in Gram-Schmidt orthogonalization
            # Start with original vector and subtract projections
            w = p.x.copy()

            for i, v in enumerate(base):
                v_norm_sq = base_norms_sq[i]
                if abs(v_norm_sq) > 1e-12:  # Check for near-zero norms
                    # Project p.x onto v: (v . p.x / |v|^2) * v
                    # Standard Gram-Schmidt uses original vector p.x in dot product
                    coeff = v.dot(p.x) / v_norm_sq
                    w -= coeff * v
                else:
                    # Skip degenerate vectors
                    continue

            if np.any(np.abs(w) > tol):
                base.append(w)
                base_norms_sq.append(w.dot(w))
                ret.append(p)
                if len(ret) >= dim:
                    return ret
            else:
                # self.logger.info("below tol: %s (base: %s)" % (np.abs(w),
                # base))
                pass
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

        if not weights or np.sum(weights) < 1e-4:
            weights = None  # fall back to normal average

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

    def _refresh_base(self) -> bool:
        """Derive ``worst`` / ``centroid`` from the current best box.

        Walks up the box hierarchy until :meth:`gram_schmidt` finds enough
        linearly independent results.  Returns ``True`` when a base exists.
        """
        dim = self.problem.dim
        bb = self.best_box
        while bb is not None:
            base = self.gram_schmidt(dim, bb.results)
            if base:
                self._worst, self._centroid = self.nelder_mead_init(base)
                return True
            bb = bb.parent
        self._worst = self._centroid = None
        return False

    def _fill(self) -> None:
        """Top the output queue up with samples from the current base."""
        if self._worst is None or self._centroid is None:
            return
        free = self.cap - self._output.qsize()
        if free > 0:
            self.emit([self.nelder_mead_sample(self._worst, self._centroid) for _ in range(free)])

    def on_new_best_box(self, best_box):
        """A new best box (from the :class:`~.analyzers.Splitter`) resets the
        search direction: the queue is flushed and refilled from the new base."""
        self.best_box = best_box
        self.clear_output()
        if self._refresh_base():
            self._fill()

    def on_new_results(self, results):
        """Keep the queue topped up; the base only changes with the best box."""
        self._fill()

    def on_restart(self, center, reason):
        """Flush points and pause until a new best box is found."""
        self.clear_output()
        self.best_box = None
        self._worst = self._centroid = None
