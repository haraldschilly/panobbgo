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
import time
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
    """

    def __init__(self, strategy):
        Heuristic.__init__(self, strategy, name="Nelder Mead")
        self.logger = self.config.get_logger("H:NM")
        from threading import Event

        self.got_bb = Event()

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
        factor = np.random.rayleigh(scale=scale) - offset
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

    def on_start(self):
        """
        Algorithm Outline:

        #. Wait until a first or new best box has been found.

        #. Clear the ``got_bb`` flag, later on we use this to be notified
           about new best boxes via :meth:`.on_new_best_box`.

        #. ``bb`` is the currently used best box, it might be ``None`` if
           we have to look up the parents when searching for more result points.

        #. Inside the outer while, we try to find a suiteable base via :meth:`.gram_schmidt`.

        #. If we got such a base, we generate new search points via :meth:`.nelder_mead`
           until the queue is full (which blocks) or there is a new best box (breaks inner loop).

        #. The ``break`` exits the outer while and we start fresh with the new best box.
        """
        dim = self.problem.dim
        while not self._stopped:
            # Wait with timeout to allow checking _stopped
            if not self.got_bb.wait(timeout=0.1):
                continue

            bb = self.best_box
            self.got_bb.clear()
            while bb is not None and not self._stopped:
                base = self.gram_schmidt(dim, bb.results)
                if base:  # was able to find a base
                    if len(base) == 0:
                        break

                    worst, centroid = self.nelder_mead_init(base)

                    while not self.got_bb.is_set() and not self._stopped:
                        if self._output.full():
                            time.sleep(0.1)
                            continue
                        new_point = self.nelder_mead_sample(worst, centroid)
                        # self.logger.info("new point: %s" % new_point)
                        self.emit(new_point)
                    break
                else:  # not able to find base, try with parent of current best box
                    bb = bb.parent
                    if bb is None:
                        self.got_bb.clear()  # the "wait()" at the top is now active

    def on_restart(self, center, reason):
        """
        Respond to a restart event by flushing points and pausing until a new best box is found.
        """
        self.clear_output()
        self.got_bb.clear()

    def on_new_best_box(self, best_box):
        """
        When a new best box has been found by the :class:`~.analyzers.Splitter`, the
        ``got_bb`` :class:`~threading.Event` is set and the output queue is cleared.
        """
        self.best_box = best_box
        self.got_bb.set()
        self.clear_output()  # clearing must come last
