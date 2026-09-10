# -*- coding: utf8 -*-
# Copyright 2026 Harald Schilly <harald.schilly@gmail.com>
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

r"""
Archive
=======

A bounded, *shared* top-K of the whole result stream — the query layer the
population heuristics need in order to warm-start from points somebody else
paid for (``planning/DESIGN_warm_start_2026-09-10.md`` §2).

What the store offers today is either too little or too much: ``Results``
only replays the last *n* rows as floats, ``Best`` collapses to a single
incumbent, and ``Splitter.root.results`` is a complete but unsorted,
unbounded ``list[Result]``.  This analyzer keeps the *K* best results seen so
far, ranked by the constraint handler's penalty value and **not filtered by
``who``** — an arm that asks for a seed gets the best points in the run,
whoever produced them.

Cost: one heap operation per result, ``O(log K)``.  Queries are ``O(K log K)``
and are made a handful of times per run, so they are deliberately simple.

.. Note::

  This analyzer is **opt-in**: :meth:`~panobbgo.core.StrategyBase.initialize`
  does not add it.  Ship it via ``StrategySpec.analyzers`` (or
  ``add_analyzer``) so a run without warm-started arms keeps the exact
  module construction order — and therefore the exact RNG streams — it had
  before.

.. codeauthor:: Harald Schilly <harald.schilly@gmail.com>
"""

from __future__ import annotations

import heapq
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from panobbgo.core import Analyzer
from panobbgo.lib import Result

#: Default number of results retained.
DEFAULT_K = 256


class Archive(Analyzer):
    r"""
    Bounded top-K of the shared result stream, ranked by penalty value.

    :param strategy: the owning strategy.
    :param k: how many results to retain (default :data:`DEFAULT_K`).
    :param name: optional module name; defaults to ``"Archive"``.
    """

    def __init__(self, strategy, k: int = DEFAULT_K, name: Optional[str] = None) -> None:
        if isinstance(k, bool) or not isinstance(k, int):
            raise ValueError(f"Archive: k must be an integer, got {k!r}")
        if k < 1:
            raise ValueError(f"Archive: k must be >= 1, got {k}")
        Analyzer.__init__(self, strategy, name=name)
        self.K: int = int(k)
        #: min-heap of ``(-penalty, -seq, result)``; the root is therefore the
        #: *worst* retained entry, which is the one an incoming better result
        #: replaces.  ``-seq`` breaks ties deterministically in favour of the
        #: older result and keeps :class:`~panobbgo.lib.Result` out of the
        #: comparison (``Result.__eq__`` only looks at ``fx``).
        self._heap: List[Tuple[float, int, Result]] = []
        self._seq: int = 0

    # -- ingestion ---------------------------------------------------------

    def _penalty(self, r: Result) -> Optional[float]:
        """The scalar the constraint handler minimises, or ``None`` if unusable."""
        handler = getattr(self.strategy, "constraint_handler", None)
        try:
            if handler is None:
                value = float("inf") if r.fx is None else float(r.fx)
            else:
                value = float(handler.get_penalty_value(r))
        except (TypeError, ValueError):
            return None
        return value if np.isfinite(value) else None

    def on_new_results(self, results: Iterable[Result]) -> None:
        """Fold every result into the bounded top-K.  ``O(log K)`` per result."""
        for r in results:
            phi = self._penalty(r)
            if phi is None:
                continue  # inf / nan / failed evaluation: no information
            self._seq += 1
            item = (-phi, -self._seq, r)
            if len(self._heap) < self.K:
                heapq.heappush(self._heap, item)
            elif item > self._heap[0]:
                heapq.heapreplace(self._heap, item)

    # -- queries -----------------------------------------------------------

    def __len__(self) -> int:
        return len(self._heap)

    @property
    def results(self) -> List[Result]:
        """Every retained result, best first."""
        return [r for _, _, r in sorted(self._heap, reverse=True)]

    def penalty_of(self, r: Result) -> float:
        """Penalty value of ``r``, ``inf`` when it cannot be evaluated."""
        phi = self._penalty(r)
        return float("inf") if phi is None else phi

    def top_k(
        self,
        k: int,
        *,
        box: Any = None,
        exclude_who: Any = None,
        fx_max: Optional[float] = None,
    ) -> List[Result]:
        """The ``k`` best retained results, best first.

        :param box: restrict to results inside this region — either a
            :class:`~panobbgo.analyzers.splitter.Splitter.Box` (anything with a
            ``contains`` method) or a ``(dim, 2)`` array of bounds.
        :param exclude_who: a ``who`` tag, or an iterable of them, to skip.
            Matching is by heuristic name, so ``"PSO"`` also drops
            ``"PSO:1a2b…"``.
        :param fx_max: keep only results whose penalty is ``<= fx_max``.
        """
        if k <= 0:
            return []
        excluded = self._who_filter(exclude_who)
        out: List[Result] = []
        for neg_phi, _, r in sorted(self._heap, reverse=True):
            if fx_max is not None and -neg_phi > fx_max:
                break  # sorted ascending in penalty: nothing later can pass
            if excluded and self._who_of(r) in excluded:
                continue
            if box is not None and not self._in_box(r, box):
                continue
            out.append(r)
            if len(out) >= k:
                break
        return out

    def diverse_k(self, k: int, *, pool: int = 4, box: Any = None) -> List[Result]:
        """``k`` well-separated good results: greedy max–min over the top ``pool*k``.

        The incumbent is always the first pick; each further pick maximises
        the minimum distance to everything picked so far, measured in
        box-normalised coordinates so the selection does not depend on the
        units of a dimension.  This is the selector that hands a second arm a
        *spread* rather than a cluster around one basin.
        """
        if k <= 0:
            return []
        candidates = self.top_k(max(k * max(int(pool), 1), k), box=box)
        if len(candidates) <= k:
            return candidates

        xs = np.asarray([np.asarray(r.x, dtype=float) for r in candidates])
        scale = self._scale(xs.shape[1])
        xs = xs / scale

        chosen = [0]
        dist = np.linalg.norm(xs - xs[0], axis=1)
        while len(chosen) < k:
            nxt = int(np.argmax(dist))
            if nxt in chosen:
                break  # everything left is a duplicate of something chosen
            chosen.append(nxt)
            dist = np.minimum(dist, np.linalg.norm(xs - xs[nxt], axis=1))
        return [candidates[i] for i in chosen]

    def per_leaf_best(self, k: int) -> List[Result]:
        """Best point of each of the ``k`` best :class:`Splitter` leaves.

        Returns ``[]`` when no ``Splitter`` is registered — the only selector
        that depends on a second analyzer, and the only one that can hand out
        points from *different* basins by construction.
        """
        if k <= 0:
            return []
        try:
            splitter = self.strategy.analyzer("Splitter")
            leafs = [box for box in splitter.leafs if getattr(box, "best", None) is not None]
        except Exception:
            return []
        leafs.sort(key=lambda box: self.penalty_of(box.best))
        out: List[Result] = []
        seen: Dict[int, bool] = {}
        for box in leafs:
            if id(box.best) in seen:
                continue  # a point can be the best of several boxes
            seen[id(box.best)] = True
            out.append(box.best)
            if len(out) >= k:
                break
        return out

    # -- helpers -----------------------------------------------------------

    def _scale(self, dim: int) -> np.ndarray:
        """Per-dimension normaliser: the problem's box ranges, or ones."""
        try:
            ranges = np.asarray(self.problem.box[:, 1] - self.problem.box[:, 0], dtype=float)
        except Exception:
            return np.ones(dim)
        ranges = np.where(np.isfinite(ranges) & (ranges > 0.0), ranges, 1.0)
        return ranges if ranges.shape == (dim,) else np.ones(dim)

    @staticmethod
    def _who_of(r: Result) -> str:
        who = getattr(r, "who", "") or ""
        return who.split(":", 1)[0]

    @staticmethod
    def _who_filter(exclude_who: Any) -> set:
        if exclude_who is None:
            return set()
        if isinstance(exclude_who, str):
            exclude_who = [exclude_who]
        return {str(w).split(":", 1)[0] for w in exclude_who}

    @staticmethod
    def _in_box(r: Result, box: Any) -> bool:
        x = np.asarray(r.x, dtype=float)
        contains = getattr(box, "contains", None)
        if callable(contains):
            return bool(contains(x))
        bounds = np.asarray(box, dtype=float)
        return bool((bounds[:, 0] <= x).all() and (x <= bounds[:, 1]).all())
