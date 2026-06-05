# -*- coding: utf8 -*-
# Copyright 2012-2026 Panobbgo Contributors
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

"""
RegionUCB Heuristic
===================

A region-allocation heuristic in the spirit of LA-MCTS / DIRECT: it treats
the leaves of the :class:`~panobbgo.analyzers.splitter.Splitter` tree as
bandit arms, selects a leaf via a UCB1-style score, and samples new
candidate points *inside* the chosen leaf.

This differs from the existing :class:`~panobbgo.heuristics.Random`
heuristic (which only samples the leaf around the global best) by
allocating evaluations across *all* regions — exploiting good leaves while
still visiting under-sampled ones.

Because it is a plain :class:`~panobbgo.core.Heuristic`, it composes with
every existing strategy: the strategy-level bandit (Rewarding / UCB /
Thompson) learns *whether* region-level allocation helps, while RegionUCB
learns *where* to sample.  Placement is steered directly (points are drawn
inside the selected box), so credit assignment needs no extra machinery.

**Leaf score** (higher is better)::

    score(leaf) = quality(leaf) + ucb_c * sqrt(log(N) / n_leaf)

- ``quality``: rank-based in [0, 1] from the leaf's best penalty value
  (constraint-aware via the strategy's constraint handler).  Rank-based
  normalization is scale-free, so it works for arbitrary objectives.
- ``n_leaf``: number of evaluated points inside the leaf; ``N``: total.
- Leaves with no points score ``+inf`` and are explored first
  (random tie-break).

**In-leaf sampling**: a mix of uniform draws over the leaf box and
Gaussian draws around the leaf's best point (clipped to the box), so both
fresh exploration and local refinement happen inside the chosen region.
"""

from __future__ import annotations

import numpy as np

from panobbgo.core import Heuristic


class RegionUCB(Heuristic):
    """
    UCB1 allocation over Splitter leaves with in-leaf sampling.

    Args:
        strategy: The strategy instance (provides Splitter, constraint handler).
        ucb_c (float): Exploration weight in the UCB score (default 1.0).
        n_candidates (int): Points emitted per ``new_results`` event (default 5).
        gauss_fraction (float): Fraction of candidates drawn from a Gaussian
            around the leaf's best point instead of uniformly (default 0.5).
        gauss_scale (float): Std-dev of that Gaussian, relative to the leaf's
            ranges (default 0.25).
    """

    def __init__(self, strategy, ucb_c=1.0, n_candidates=5, gauss_fraction=0.5, gauss_scale=0.25, name=None):
        name = "RegionUCB" if name is None else name
        Heuristic.__init__(self, strategy, name=name)
        self.logger = self.config.get_logger("RUCB")
        self.ucb_c = float(ucb_c)
        self.n_candidates = int(n_candidates)
        self.gauss_fraction = float(gauss_fraction)
        self.gauss_scale = float(gauss_scale)

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _box_array(leaf):
        """Return the (dim, 2) bounds array of a Splitter.Box."""
        box = leaf.box
        return box.box if hasattr(box, "box") else box

    def _leaf_penalty(self, leaf):
        """Best penalty value in the leaf (constraint-aware), or +inf if empty."""
        if leaf.best is None:
            return float("inf")
        handler = getattr(self.strategy, "constraint_handler", None)
        if handler is not None:
            try:
                return handler.get_penalty_value(leaf.best)
            except Exception:
                pass
        fx = leaf.best.fx
        return float("inf") if fx is None else fx

    def select_leaf(self, leafs):
        """
        Pick a leaf by UCB1 score; empty leaves win immediately (random
        tie-break among them).
        """
        counts = np.array([len(leaf) for leaf in leafs], dtype=float)
        total = counts.sum()

        empty = np.where(counts == 0)[0]
        if len(empty) > 0:
            return leafs[int(np.random.choice(empty))]

        # rank-based quality in [0, 1]: best penalty -> 1, worst -> 0
        penalties = np.array([self._leaf_penalty(leaf) for leaf in leafs])
        order = np.argsort(np.argsort(penalties))  # rank, 0 = best
        if len(leafs) > 1:
            quality = 1.0 - order / (len(leafs) - 1.0)
        else:
            quality = np.ones(1)

        explore = self.ucb_c * np.sqrt(np.log(max(total, 2.0)) / counts)
        scores = quality + explore
        return leafs[int(np.argmax(scores))]

    def sample_in_leaf(self, leaf):
        """Draw ``n_candidates`` points inside the leaf's box."""
        box = self._box_array(leaf)
        lo, ranges = box[:, 0], leaf.ranges
        dim = len(lo)

        n_gauss = 0
        if leaf.best is not None:
            n_gauss = int(round(self.n_candidates * self.gauss_fraction))
        n_uniform = self.n_candidates - n_gauss

        points = [lo + ranges * np.random.rand(dim) for _ in range(n_uniform)]
        if n_gauss > 0:
            sigma = np.maximum(ranges * self.gauss_scale, 1e-12)
            for _ in range(n_gauss):
                p = leaf.best.x + sigma * np.random.randn(dim)
                points.append(np.clip(p, box[:, 0], box[:, 1]))
        return points

    # -- event handlers ----------------------------------------------------

    def on_new_results(self, results):
        """Re-select a region and refill the output queue on every batch."""
        try:
            splitter = self.strategy.analyzer("Splitter")
            leafs = list(splitter.leafs)
        except Exception:
            leafs = []

        if not leafs:
            # No Splitter (yet) — sample the full problem box instead.
            self.clear_output()
            self.emit([self.problem.random_point() for _ in range(self.n_candidates)])
            return

        leaf = self.select_leaf(leafs)
        points = self.sample_in_leaf(leaf)
        self.clear_output()
        self.emit(points)
        self.logger.debug(
            "leaf id=%s depth=%s n=%d -> emitted %d points" % (leaf.id, leaf.depth, len(leaf), len(points))
        )
