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
        # sort the results by asc. f(x)
        results = sorted(results, key=lambda p: p.fx)
        # better? randomize results to diversify
        # from random import shuffle
        # shuffle(results)
        first = results.pop(0)
        base.append(first.x)
        ret.append(first)
        for p in results:
            w = p.x - np.sum([((v.dot(p.x) / v.dot(v)) * v) for v in base], axis=0)
            if np.any(np.abs(w) > tol):
                base.append(w)
                ret.append(p)
                if len(ret) >= dim:
                    return ret
            else:
                # self.logger.info("below tol: %s (base: %s)" % (np.abs(w),
                # base))
                pass
        return None

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
        # base = sorted(base, key = lambda r : r.fx)
        # worst = base.pop() # worst point, base is the remainder

        # get worst point and it's index (to remove it)
        worst_idx, worst = max(enumerate(base), key=lambda _: _[1].fx)
        del base[worst_idx]

        # TODO f(x) values are available and could be used for weighting (or their rank number)
        # weights = [ np.log1p(worst.fx - r.fx) for r in base ]
        # weights = 1 + .1 * np.random.randn(len(base))
        centroid = np.average([p.x for p in base], axis=0)  # , weights = weights)
        factor = np.random.rayleigh(scale=scale) - offset
        return worst.x + factor * (centroid - worst.x)

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
        while True:
            self.got_bb.wait()
            bb = self.best_box
            self.got_bb.clear()
            while bb is not None:
                base = self.gram_schmidt(dim, bb.results)
                if base:  # was able to find a base
                    if len(base) == 0:
                        break
                    # TODO split nelder_mead into a init phase, and a sampling
                    # routine
                    while not self.got_bb.is_set():
                        new_point = self.nelder_mead(base[:])
                        # self.logger.info("new point: %s" % new_point)
                        self.emit(new_point)
                    break
                else:  # not able to find base, try with parent of current best box
                    bb = bb.parent
                    if bb is None:
                        self.got_bb.clear()  # the "wait()" at the top is now active

    def on_new_best_box(self, best_box):
        """
        When a new best box has been found by the :class:`~.analyzers.Splitter`, the
        ``got_bb`` :class:`~threading.Event` is set and the output queue is cleared.
        """
        self.best_box = best_box
        self.got_bb.set()
        self.clear_output()  # clearing must come last
