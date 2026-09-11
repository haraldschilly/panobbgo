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

from panobbgo.core import Analyzer
from panobbgo.utils import memoize

import numpy as np


"""
Splitter
--------

Inside is its Box class.
"""

#: Target *mean* number of results per leaf.  The tree's resolution is
#: ``max_eval / LEAF_SIZE`` leaves, i.e. it grows with the budget instead of
#: being pinned to the dimension (``planning/DESIGN_meta_level_2026-09-10.md``
#: §1: the old ``limit = max(20, max_eval/dim**2)`` settles at ≈1.3·dim²
#: leaves whatever the budget is — 5 leaves at *d* = 2, and the root cannot
#: split before evaluation 250 of 1000).  25 is the smallest population for
#: which a leaf's ``best`` is a statistic rather than a single draw, and it
#: puts the *d* = 2 tree at ~40 leaves, the resolution the consumers
#: (``Random``, ``RegionUCB``, ``Archive.per_leaf_best``) are written for.
LEAF_SIZE = 25

#: Hard cap on the number of leaves.  Everything in the analyzer that is not
#: O(1) per result is O(#leaves) per *split* (``_new_box``'s scans,
#: ``leafs.remove``), so the tree costs O(L²) over a run; 512 keeps that
#: below a millisecond-scale contribution on the dispatcher thread at every
#: (dim, budget) in the plan of record.
MAX_LEAVES = 512

#: Measured mean leaf population divided by the split threshold.  A leaf
#: splits at ``limit`` points and each child inherits the parent's points it
#: contains, so a leaf holds between ``limit/2`` and ``limit`` points.  The
#: realised mean is ≈0.67·``limit`` — measured over uniform and contracting
#: point clouds at (dim, budget) ∈ {(2,1000), (5,2500), (10,5000),
#: (20,10000)}, where it stayed in 0.59 … 0.72 for both split rules and for
#: the legacy threshold of 250.  Used to turn a target leaf *population*
#: into the split threshold; a mis-calibration only scales the leaf count,
#: it cannot break the tree.
LEAF_FILL = 0.67

#: How the split dimension is chosen.  ``"widest"`` is the historical rule
#: (widest dimension the results differ in, function values ignored);
#: ``"value"`` scores each candidate dimension by how strongly the objective
#: separates across the prospective cut.  See :meth:`Splitter.Box._split_dim`.
SPLIT_RULES = ("widest", "value")

#: The default split rule.  ``"widest"``, deliberately: a paired 3-seed
#: screen (42/7/1234, MA-BBOB standard battery, dims 2 and 5 at 500·d,
#: instances 0-2, ``seed_name`` shared) put ``"value"`` minus ``"widest"``,
#: *both* on the budget-scaled resolution, at +0.001 for ``Random``, +0.024
#: for ``RegionUCB``, 0.000 for the reference portfolio and −0.000 for the
#: ``archive_leaf`` portfolio — every one of them inside the ±0.03 null
#: floor, every CI straddling zero, and it costs ~8% more analyzer time per
#: result.  The whole measured gain of this change is the *resolution*
#: (+0.048 / +0.014 / 0.000 / +0.042 against the legacy tree).  ``"value"``
#: ships opt-in so a 12-seed roster can revisit ``RegionUCB``, which is the
#: only consumer that reads more than one leaf per decision.
DEFAULT_SPLIT_RULE = "widest"

#: Where along the chosen dimension the cut goes.  ``"mean"`` is the
#: historical point; ``"median"`` cuts through the middle *order statistic*.
#: See :meth:`Splitter.Box._split_point`.
CUT_RULES = ("mean", "median")

#: The default cut.  ``"mean"``, and the reasoning that said otherwise was
#: **wrong**, so it is written down here rather than repeated.
#:
#: The observation was real: ``Random`` at *d* = 5 (Rosenbrock, 2500
#: evaluations) builds a tree of depth 60 — the :attr:`Box.MAX_DEPTH` cap —
#: holding 61 leaves against a target of 101, with a single leaf of ~1300
#: points.  The *diagnosis* — "the mean is dragged towards the stragglers of
#: a contracting cloud, so each cut only peels a few points off" — was not:
#: a median cut produces the identical depth 60 / 61 leaves / 1301-point leaf
#: on the same spec.  The chain is :class:`~panobbgo.heuristics.Random`'s
#: own doing: it samples *only inside the current best leaf*, so every point
#: lands in one leaf, that leaf splits, the child holding the best point
#: becomes the next target and the sibling is never visited again.  Depth
#: then grows once per split whatever the cut rule is.  Fixing it means
#: changing the heuristic or the depth cap, not the geometry.
#:
#: What the median *does* change was then measured, paired, 3 seeds
#: (42/7/1234, MA-BBOB standard battery, dims 2 and 5 at 500·d, instances
#: 0-2): ``Random`` **−0.0145** (1/3 seeds), ``RegionUCB`` **+0.0327**
#: (3/3) — one inside the ±0.03 null floor and negative, the other barely
#: outside it, mean +0.009.  No mandate.  And it carries one concrete
#: structural regression: on ``Blocks_uniform_cj_warm2`` at *d* = 5 the
#: median cut reaches ``MAX_DEPTH`` with a 423-point leaf where the mean
#: stays at depth 52 with no leaf above 36.
#:
#: So ``"median"`` ships opt-in — it is genuinely the better *geometry* (see
#: :meth:`Splitter.Box._split_point`) and ``RegionUCB``, the one consumer
#: that reads every leaf, is the case worth revisiting on a 12-seed roster.
DEFAULT_CUT_RULE = "mean"


class Splitter(Analyzer):
    """
    Manages a tree of splits.
    Each split in this tree is a :class:`box <.Splitter.Box>`, which
    partitions the search space into smaller boxes and can have children.
    Boxes without children are :attr:`leafs <.Splitter.Box.leaf>`.

    The goal for this splitter is to balance between the
    depth level of splits and the number of points inside such a box.

    A heuristic can build upon this hierarchy
    to investigate interesting subregions.

    Args:
        strategy: the owning strategy.
        split_rule: ``"widest"`` (default, see :data:`DEFAULT_SPLIT_RULE`)
            or ``"value"``; see :data:`SPLIT_RULES`.
        leaf_size: target mean number of results per leaf
            (default :data:`LEAF_SIZE`).  The resolution of the tree is
            ``max_eval / leaf_size`` leaves.
        min_leaf_size: floor on the split threshold, so a leaf is never cut
            before it holds enough points to say anything.  Default
            ``2 * dim + 2`` — the number of points a linear trend in *dim*
            variables needs, plus a margin.
        max_leaves: cap on the number of leaves
            (default :data:`MAX_LEAVES`); it raises the split threshold
            rather than refusing splits, so the partition stays a proper
            kd-tree.
        cut_rule: ``"mean"`` (default, see :data:`DEFAULT_CUT_RULE`; measured in DISCOVERY §39) or
            ``"mean"`` — where along the chosen dimension the cut falls.
        legacy: ``True`` restores the pre-2026-09-10 analyzer exactly —
            ``limit = max(20, max_eval / dim**2)``, ``split_rule =
            "widest"`` and ``cut_rule = "mean"``.  Every other knob is
            ignored.  Kept so both trees are runnable from one code base
            and a measurement can be paired.
    """

    def __init__(
        self,
        strategy,
        split_rule=None,
        cut_rule=None,
        leaf_size=None,
        min_leaf_size=None,
        max_leaves=None,
        legacy=False,
        name=None,
    ):
        Analyzer.__init__(self, strategy, name=name)
        # split, if there are more than this number of points in the box
        self.leafs = []
        self._id = 0  # block id
        self.logger = self.config.get_logger("SPLIT")  # , 10)
        self.max_eval = self.config.max_eval
        self.legacy = bool(legacy)
        if split_rule is None:
            split_rule = "widest" if self.legacy else DEFAULT_SPLIT_RULE
        if split_rule not in SPLIT_RULES:
            raise ValueError("split_rule must be one of %s, got %r" % (SPLIT_RULES, split_rule))
        self.split_rule = "widest" if self.legacy else split_rule
        if cut_rule is None:
            cut_rule = "mean" if self.legacy else DEFAULT_CUT_RULE
        if cut_rule not in CUT_RULES:
            raise ValueError("cut_rule must be one of %s, got %r" % (CUT_RULES, cut_rule))
        self.cut_rule = "mean" if self.legacy else cut_rule
        self.leaf_size = float(LEAF_SIZE if leaf_size is None else leaf_size)
        self.min_leaf_size = None if min_leaf_size is None else int(min_leaf_size)
        self.max_leaves = int(MAX_LEAVES if max_leaves is None else max_leaves)
        # _new_result used to signal get_leaf and others when there
        # are updates regarding box/split/leaf status
        from threading import Condition

        self._new_result = Condition()

    def _resolve_limit(self):
        """The number of points at which a leaf is cut.

        Legacy: ``max(20, max_eval / dim**2)`` — independent of the budget
        once ``dim`` is fixed, which is the defect this replaces.

        Otherwise the rule is stated on the *leaf population* and inverted::

            target_leaves = clip(max_eval / leaf_size, 1, max_leaves)
            limit         = max_eval / (LEAF_FILL * target_leaves)
                          = leaf_size / LEAF_FILL      (when no cap binds)
            limit         = clip(round(limit), min_leaf_size, max_eval)

        so the resolution scales with the budget and the two clips are the
        only dimension-dependent terms: ``min_leaf_size = 2·dim + 2`` keeps
        a leaf big enough to carry a local statistic at high *dim* (it is
        what makes *d* = 20 coarser than *d* = 2 instead of finer), and
        ``max_leaves`` is the cost guard.
        """
        if self.legacy:
            return max(20, self.max_eval / self.dim**2)
        min_leaf = 2 * self.dim + 2 if self.min_leaf_size is None else self.min_leaf_size
        target = min(max(self.max_eval / max(self.leaf_size, 1.0), 1.0), float(max(self.max_leaves, 1)))
        limit = self.max_eval / (LEAF_FILL * target)
        return int(min(max(round(limit), min_leaf, 4), max(self.max_eval, 4)))

    def target_leaves(self):
        """Leaf count this configuration aims at — ``max_eval / leaf_size``,
        clipped, and re-derived from the *realised* integer threshold so it
        matches what a run actually builds."""
        return max(1.0, self.max_eval / (LEAF_FILL * self.limit))

    def __start__(self):
        # root box is equal to problem's box
        self.dim = self.problem.dim
        self.limit = self._resolve_limit()
        self.logger.debug(
            "limit = %s (rule=%s, legacy=%s, target leafs ~ %.0f)"
            % (self.limit, self.split_rule, self.legacy, self.target_leaves())
        )
        self.root = Splitter.Box(None, self, self.problem.box.copy())
        self.leafs.append(self.root)
        # leafs bucketed by depth — same insertion order as ``leafs``, so
        # ``big_by_depth`` picks the identical box the O(#leafs) filter did.
        from collections import defaultdict

        self._leafs_by_depth = defaultdict(list)
        self._leafs_by_depth[self.root.depth].append(self.root)
        # big boxes
        self.biggest_leaf = self.root
        self.big_by_depth = dict()
        self.big_by_depth[self.root.depth] = self.root
        self.max_depth = self.root.depth
        # best box (with best f(x))
        self.best_box = None
        # in which box (a list!) is each point?
        self.result2boxes = defaultdict(list)
        self.result2leaf = {}

    def _replace_leaf(self, parent, children):
        """``parent`` stopped being a leaf; ``children`` became ones."""
        self.leafs.remove(parent)
        try:
            self._leafs_by_depth[parent.depth].remove(parent)
        except ValueError:  # a hand-built box that never entered the buckets
            pass
        for c in children:
            self.leafs.append(c)
            self._leafs_by_depth[c.depth].append(c)

    def _new_box(self, new_box):
        """
        Called for each new box when there is a split.
        E.g. it updates the ``biggest`` box and related
        information for each depth level.
        """
        self.max_depth = max(new_box.depth, self.max_depth)

        old_biggest_leaf = self.biggest_leaf
        # A child is contained in its parent, so it can never be larger than
        # the incumbent; the only way the biggest leaf changes is that it was
        # the box just split.  ``max`` keeps the first of equal volumes and
        # children are appended at the end, so this is exactly what a full
        # rescan returns — at O(1) instead of O(#leafs) per new box.
        if not old_biggest_leaf.leaf:
            self.biggest_leaf = max(self.leafs, key=lambda l: l.log_volume)
        if old_biggest_leaf is not self.biggest_leaf:
            self.eventbus.publish("new_biggest_leaf", box=new_box)

        dpth = new_box.depth
        # also consider the parent depth level
        for d in [dpth - 1, dpth]:
            old_big_by_depth = self.big_by_depth.get(d, None)
            if old_big_by_depth is None:
                self.big_by_depth[d] = new_box
            else:
                leafs_at_depth = self._leafs_by_depth.get(d, ())
                if len(leafs_at_depth) > 0:
                    self.big_by_depth[d] = max(leafs_at_depth, key=lambda l: l.log_volume)

            if self.big_by_depth[d] is not old_big_by_depth:
                self.eventbus.publish("new_biggest_by_depth", depth=d, box=self.big_by_depth[d])

    def on_new_biggest_leaf(self, box):
        self.logger.debug("biggest leaf at depth %d -> %s" % (box.depth, box))

    def on_new_biggest_by_depth(self, depth, box):
        self.logger.debug("big by depth: %d -> %s" % (depth, box))

    def get_box(self, point):
        """
        return "leftmost" leaf box, where given point is contained in
        """
        box = self.root
        while not box.leaf:
            box = box.get_child_boxes(point)[0]
        return box

    def get_all_boxes(self, result):
        """
        return all boxes, where point is contained in
        """
        from panobbgo.lib import Result

        assert isinstance(result, Result)
        return self.result2boxes[result]

    def get_leaf(self, result):
        """
        returns the leaf box, where given result is currently sitting in
        """
        from panobbgo.lib import Result

        assert isinstance(result, Result)
        # The eventbus delivers events serially, so a result published
        # before this call is already registered.  Never block here: a
        # waiting handler would stall the whole dispatcher.
        with self._new_result:
            return self.result2leaf.get(result)

    def on_new_results(self, results):
        with self._new_result:
            for result in results:
                self.root += result
            self._new_result.notify_all()
        # logger.info("leafs: %s" % map(lambda x:(x.depth, len(x)), self.leafs))
        # logger.info("point %s in boxes: %s" % (result.x, self.get_all_boxes(result)))
        # logger.info("point %s in leaf: %s" % (result.x, self.get_leaf(result)))
        # assert self.get_all_boxes(result)[-1] == self.get_leaf(result)

    def on_new_split(self, box, children, dim):
        self.logger.debug("Split: %s" % box)
        for i, chld in enumerate(children):
            self.logger.debug(" +ch%d: %s" % (i, chld))
        # logger.info("children: %s" % map(lambda x:(x.depth, len(x)),
        # children))

        # update self.best_box
        # check if new box contains the best point (>= because it could
        # be a child box)

        # If the best box was the one being split, we prefer to point to the child
        # that contains the best point now.
        best_box_was_parent = self.best_box is box

        for new_box in children:
            if self.best_box is None:
                self.best_box = new_box
                continue

            if new_box.best is None:
                continue

            if self.best_box.best is None:
                self.best_box = new_box
                continue

            is_better = False
            if hasattr(self.strategy, "constraint_handler") and self.strategy.constraint_handler:
                is_better = self.strategy.constraint_handler.is_better(self.best_box.best, new_box.best)
            else:
                is_better = new_box.fx < self.best_box.fx

            if is_better:
                self.best_box = new_box
            elif best_box_was_parent:
                # If we are splitting the best box, and this child contains the best point
                # (or an equally good one), we move ownership to the child.
                # Result equality checks fx. We might need checking identity or value equality.
                # If fx is same and cv is same (implied by not is_better and not reverse is_better),
                # we can assume it's the same point or equivalent.

                # Check if new_box.best is effectively "equal" to best_box.best
                # (Since is_better returned False)
                # We simply check if new_box.best.fx <= best_box.best.fx (and handling constraints if needed)
                # But is_better already checked that.
                # If the parent contained the best point, one child MUST contain it.
                # So we just need to find WHICH child contains it.

                # Simple check: if fx matches (and cv matches)
                if new_box.best == self.best_box.best:  # Result.__eq__ checks fx
                    # Ideally check identity
                    if new_box.best is self.best_box.best:
                        self.best_box = new_box
                    # Or check if cv also matches
                    elif new_box.best.cv == self.best_box.best.cv:
                        self.best_box = new_box
        self.eventbus.publish("new_best_box", best_box=self.best_box)

    def on_refresh_best(self, candidates):
        """
        Called when the definition of "best" changes (e.g. ALM parameter update).
        We need to re-evaluate the local best for all leaf boxes and update the global best box.
        """
        # 1. Re-evaluate local best in each leaf box
        for box in self.leafs:
            if not box.results:
                continue

            # Re-find best in this box
            current_best = box.results[0]
            for res in box.results[1:]:
                is_better = False
                if hasattr(self.strategy, "constraint_handler") and self.strategy.constraint_handler:
                    is_better = self.strategy.constraint_handler.is_better(current_best, res)
                else:
                    is_better = res.fx < current_best.fx

                if is_better:
                    current_best = res
            box.best = current_best

        # 2. Re-evaluate global best box from all leafs
        new_best_box = None
        for box in self.leafs:
            if box.best is None:
                continue

            if new_best_box is None:
                new_best_box = box
                continue

            is_better = False
            if hasattr(self.strategy, "constraint_handler") and self.strategy.constraint_handler:
                is_better = self.strategy.constraint_handler.is_better(new_best_box.best, box.best)
            else:
                is_better = box.fx < new_best_box.fx

            if is_better:
                new_best_box = box

        if new_best_box is not None and new_best_box is not self.best_box:
            self.best_box = new_best_box
            self.eventbus.publish("new_best_box", best_box=self.best_box)

    class Box:
        """
        Used by :class:`.Splitter`, therefore nested.

        Most important routine is :meth:`.split`.

        .. Note::

          In the future, this might be refactored to allow different
          splitting methods.
        """

        def __init__(self, parent, splitter, box):
            self.parent = parent
            self.logger = splitter.logger
            self.depth = parent.depth + 1 if parent else 0
            self.box = box
            self.splitter = splitter
            self.limit = splitter.limit
            self.dim = splitter.dim
            self.best = None  # best point
            self.results = []
            self.children = []
            self.split_dim = None
            self.id = splitter._id
            splitter._id += 1
            # Running component-wise min/max of the results in this box.
            # ``_can_split`` runs on *every* result once the box is over-full
            # (``add_result``), so the "do the points differ anywhere?"
            # question has to be O(dim), not O(#results·dim) — otherwise a
            # box that can never be split re-scans its whole point cloud on
            # every arrival and the degenerate case costs O(n²).
            self._xmin = None
            self._xmax = None

        @property
        def leaf(self):
            """
            returns ``true``, if this box is a leaf. i.e. no children
            """
            return len(self.children) == 0

        @property
        def fx(self):
            """
            Function value of best point in this particular box.
            """
            if self.best is None:
                return float("inf")
            return self.best.fx

        @memoize
        def __ranges(self):
            # If self.box is a BoundingBox (custom object), get the underlying array
            box_array = self.box.box if hasattr(self.box, "box") else self.box
            return np.ptp(box_array, axis=1)  # self.box[:,1] - self.box[:,0]

        @property
        def ranges(self):
            """
            Gives back a vector with all the ranges of this box,
            i.e. upper - lower bound.
            """
            from typing import Any, cast

            return cast(Any, self).__ranges()

        @memoize
        def __log_volume(self):
            with np.errstate(divide="ignore"):
                return np.sum(np.log(self.ranges))

        @property
        def log_volume(self):
            """
            Returns the `logarithmic` volume of this box.
            """
            from typing import Any, cast

            return cast(Any, self).__log_volume()

        @memoize
        def __volume(self):
            return np.exp(self.log_volume)

        @property
        def volume(self):
            """
            Returns the volume of the box.

            .. Note::

              Currently, the exponential of :attr:`.log_volume`
            """
            from typing import Any, cast

            return cast(Any, self).__volume()

        def _register_result(self, result):
            """
            This updates the splitter and box specific datatypes,
            i.e. the maps from a result to the corresponding boxes or leafs.
            """
            from panobbgo.lib import Result

            assert isinstance(result, Result)
            self.results.append(result)

            x = np.asarray(result.x, dtype=float)
            xmin, xmax = self._xmin, self._xmax
            if xmin is None or xmax is None:
                self._xmin = x.copy()
                self._xmax = x.copy()
            else:
                np.minimum(xmin, x, out=xmin)
                np.maximum(xmax, x, out=xmax)

            # new best result in box? (for best fx value, too)
            if self.best is not None:
                is_better = False
                if hasattr(self.splitter.strategy, "constraint_handler") and self.splitter.strategy.constraint_handler:
                    is_better = self.splitter.strategy.constraint_handler.is_better(self.best, result)
                else:
                    if result.fx is None or self.best.fx is None:
                        is_better = False
                    else:
                        is_better = result.fx < self.best.fx

                if is_better:
                    self.best = result
            else:
                self.best = result

            self.splitter.result2boxes[result].append(self)
            if self.leaf:
                self.splitter.result2leaf[result] = self

        def add_result(self, result):
            """
            Registers and adds a new :class:`~panobbgo.lib.Result`.
            In particular, it adds the given ``result`` to the
            current box and it's children (also all descendents).

            If the current box is a leaf and too big, the :meth:`.split`
            routine is called.

            .. Note::

              ``box += result`` is fine, too.
            """
            self._register_result(result)
            if not self.leaf:
                for child in self.get_child_boxes(result.x):
                    child += result  # recursive
            elif self.leaf and len(self.results) >= self.limit and self._can_split():
                self.split()

        #: A box deeper than this is never split again.  Depth grows by one
        #: per split, so the bound is generous for any real search; it exists
        #: only so a pathological point cloud cannot deepen the tree without
        #: end.
        MAX_DEPTH = 60

        def _usable(self):
            """Per-dimension width where a cut can separate the results, ``-1``
            elsewhere.

            "Elsewhere" is the live-lock guard: a dimension in which every
            result has the *same* coordinate cannot be cut, because
            :meth:`contains` includes both boundaries and would put the whole
            cluster into both children (see :meth:`_can_split`).
            """
            if len(self.results) < 2 or self._xmin is None or self._xmax is None:
                return None
            spread = self._xmax - self._xmin
            usable = np.where(spread > 0.0, self.ranges, -1.0)
            return usable if bool((usable > 0.0).any()) else None

        def _split_dim_widest(self, usable):
            """The historical rule: widest dimension the results differ in.

            Function values are ignored, so the partition is drawn purely by
            where the search already looked.
            """
            dim = int(np.argmax(usable))
            return dim if usable[dim] > 0.0 else None

        def _penalties(self):
            """Constraint-aware objective values of this box's results.

            Non-finite values (``NaN``, ``+/-inf``) and missing ``fx`` map to
            ``+inf``, i.e. "worst"; ranking them rather than using them
            keeps the split rule scale-free and immune to a single blown-up
            evaluation.
            """
            handler = getattr(self.splitter.strategy, "constraint_handler", None)
            out = np.empty(len(self.results), dtype=float)
            for i, r in enumerate(self.results):
                v = None
                if handler is not None:
                    try:
                        v = handler.get_penalty_value(r)
                    except Exception:
                        v = None
                if v is None:
                    v = r.fx
                try:
                    v = float(v)  # pyright: ignore[reportArgumentType]
                except (TypeError, ValueError):
                    v = np.inf
                out[i] = v if np.isfinite(v) else np.inf
            return out

        @staticmethod
        def _ranks(vals):
            """Mid-ranks of ``vals`` in ``[0, n-1]``, ties sharing their mean rank.

            Tie-awareness is not cosmetic: ``argsort(argsort(v))`` invents a
            strict order out of equal values, so a box in which every result
            has the same objective would be "ranked" by *arrival order* and
            the split rule would cut on noise instead of falling back to
            width.
            """
            n = len(vals)
            order = np.argsort(vals, kind="stable")
            srt = vals[order]
            fresh = np.empty(n, dtype=bool)
            fresh[0] = True
            np.not_equal(srt[1:], srt[:-1], out=fresh[1:])
            group = np.cumsum(fresh) - 1
            counts = np.bincount(group)
            starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
            mid = starts + (counts - 1) / 2.0
            ranks = np.empty(n, dtype=float)
            ranks[order] = mid[group]
            return ranks

        def _split_dim_value(self, usable):
            """Dimension along which the objective separates most across the cut.

            For every candidate dimension the prospective cut is the same one
            :meth:`split` would make (the mean coordinate), and the two halves
            are compared by the *rank* of their penalty values — scale-free,
            so it survives an objective spanning many orders of magnitude and
            a constraint handler whose penalty units are arbitrary::

                score_j = |mean_rank(left) - mean_rank(right)|
                          * sqrt(n_left * n_right / n)

            That is the standardised two-sample rank-sum statistic: the first
            factor is the separation, the second rewards a balanced cut, so a
            dimension does not win by shaving off two outliers.  Ties (in
            particular *every* score zero, i.e. a box in which the objective
            is flat or all values are equal) fall back to the widest
            dimension, so this rule degrades exactly into ``"widest"`` when
            there is no value signal to use.

            What it scores is the **cut**, not the dependence: a cloud that
            mirrors exactly about the cut — a bowl sampled symmetrically
            around its own centre — puts equally good and equally bad points
            on both sides, scores zero, and width decides.  That is the
            honest semantics for a rule about to make exactly this cut: a cut
            that separates nothing gains nothing by being made on that axis.
            In practice the mean of a finite sample is not the axis of
            symmetry and the residual asymmetry is enough — a plain
            ``x_j**2`` still wins its axis by a factor of three over the
            noise on 200 uniform points.
            """
            n = len(self.results)
            cand = np.flatnonzero(usable > 0.0)
            xs = np.vstack([r.x for r in self.results])[:, cand]
            ranks = self._ranks(self._penalties())
            if n > 1:
                ranks /= n - 1.0

            cuts = xs.mean(axis=0)
            left = xs <= cuts
            n_l = left.sum(axis=0).astype(float)
            n_r = n - n_l
            sum_l = ranks @ left
            with np.errstate(divide="ignore", invalid="ignore"):
                mean_l = sum_l / n_l
                mean_r = (ranks.sum() - sum_l) / n_r
                score = np.abs(mean_l - mean_r) * np.sqrt(n_l * n_r / n)
            # ``spread > 0`` puts min and max strictly on opposite sides of
            # the mean, so both halves are non-empty; be defensive anyway.
            score = np.where(np.isfinite(score), score, -1.0)

            best = float(score.max())
            tied = score >= best - 1e-12
            widths = np.where(tied, usable[cand], -1.0)
            return int(cand[int(np.argmax(widths))])

        def _split_dim(self):
            """Dimension to cut this box along, or ``None`` if no cut helps."""
            usable = self._usable()
            if usable is None:
                return None
            if self.splitter.split_rule == "value":
                return self._split_dim_value(usable)
            return self._split_dim_widest(usable)

        def _can_split(self):
            """``False`` when splitting cannot make progress.

            ``Box.contains`` includes both boundaries, so a cut through a
            cluster of *identical* points puts every one of them in *both*
            children.  Each child is then an over-full leaf that splits
            again on the next result, and the tree deepens without bound —
            a live-lock that costs the rest of the evaluation budget.
            (Measured: CMA-ES on MA-BBOB d5 with a diverged step size
            projects most of a generation onto the same box corner and
            stalls the run at 408 of 1000 evaluations.)

            Asks :meth:`_usable` rather than :meth:`_split_dim`: the two
            agree on *whether* a cut exists (both rules return a dimension
            whenever one does), and this one is O(dim) from the running
            min/max instead of O(#results · dim).  It runs on every result
            of an over-full leaf, so the difference is the whole cost of the
            degenerate case.
            """
            return self.depth < self.MAX_DEPTH and self._usable() is not None

        def __iadd__(self, result):
            """
            Convenience wrapper for :meth:`.add_result`.
            """
            self.add_result(result)
            return self

        def __len__(self):
            return len(self.results)

        def _split_point(self, dim):
            """Where to cut along ``dim``.

            **The invariant, and the only thing that keeps the tree finite.**
            :meth:`contains` includes both boundaries, so a point sitting
            exactly on the cut lands in *both* children.  A child is
            therefore a strict subset of its parent only when the cut ``s``
            is *strictly interior* to the observed coordinates::

                left  = {x_dim <= s}  is proper  <=>  s < max(x_dim)
                right = {x_dim >= s}  is proper  <=>  s > min(x_dim)

            Violate it and one child inherits the whole parent, is over-full
            on arrival, splits again — the §13 live-lock, one level up from
            the identical-points case :meth:`_can_split` already guards.

            ``"mean"`` satisfies it for free: the average of values that are
            not all equal lies strictly between the smallest and the largest,
            and ``_can_split`` has already established ``spread > 0``.

            A **plain median does not**, in two separate ways, and both were
            measured rather than reasoned about:

            1. It can *be* the minimum — on ``[0,0,0,1,1]`` the median is 0,
               so the right child inherits all five points.
            2. Worse, and the one that actually bit: the median is an
               *observed coordinate*, and a converging search produces long
               runs of duplicates (a clipped bound, a stalled component).
               Every point on the cut goes into **both** children, so the two
               children together hold ``n + (duplicates at the cut)`` points
               and neither shrinks much.  Replaying one real ``Random``
               *d* = 5 cloud (2500 points) through a plain median cut gave
               **61 leaves at depth 60** — the ``MAX_DEPTH`` cap — with a
               leaf holding **1302** points, against 113 leaves at depth 31
               and a largest leaf of 36 for the mean.

            So ``"median"`` here is the median *boundary*: of all the gaps
            between adjacent **distinct** coordinates, take the one that
            comes closest to halving the population and cut in its middle.
            With distinct coordinates that is exactly the median; with ties
            it is the nearest cut that no point can sit on, so a duplicate
            mass lands wholly on one side.  The mean stays the fallback for
            the float-degenerate case where the gap is too narrow to hold a
            representable point.
            """
            if self.splitter.cut_rule == "mean":
                # ``np.average`` over the list, verbatim: legacy trees are
                # pinned bit-for-bit on this expression.
                return np.average([r.x[dim] for r in self.results])
            n = len(self.results)
            xd = np.fromiter((r.x[dim] for r in self.results), dtype=float, count=n)
            values, counts = np.unique(xd, return_counts=True)
            if len(values) >= 2:
                # ``below[k]`` = points at or below ``values[k]``; the cut
                # between ``values[k]`` and ``values[k+1]`` splits the box
                # ``below[k]`` against ``n - below[k]``.
                below = np.cumsum(counts)[:-1]
                k = int(np.argmin(np.abs(2 * below - n)))
                lo_v, hi_v = float(values[k]), float(values[k + 1])
                cut = 0.5 * (lo_v + hi_v)
                if lo_v < cut < hi_v:
                    return cut
            lo, hi = float(xd.min()), float(xd.max())
            cut = float(np.average(xd))
            # The last resort is bounded by ``MAX_DEPTH``, not by this line:
            # if no representable point is strictly interior the box simply
            # keeps splitting until the depth cap stops it.
            return cut if lo < cut < hi else 0.5 * (lo + hi)

        def split(self, dim=None):
            """
            Arguments::

            - ``dim``: Dimension, along which to split. (default: `None`, and calculated)
            """
            assert self.leaf, "only leaf boxes are allowed to be split"
            if dim is None:
                # Split along the widest dimension in which the results
                # actually differ; a cut through identical coordinates
                # separates nothing (see :meth:`_can_split`).
                dim = self._split_dim()
                if dim is None:
                    dim = int(np.argmax(self.ranges))
            # self.logger.debug("dim: %d" % dim)
            assert dim >= 0 and dim < self.dim, "dimension along where to split is %d" % dim
            b1 = Splitter.Box(self, self.splitter, self.box.copy())
            b2 = Splitter.Box(self, self.splitter, self.box.copy())
            self.split_dim = dim
            split_point = self._split_point(dim)
            b1.box[dim, 1] = split_point
            b2.box[dim, 0] = split_point
            self.children.extend([b1, b2])
            self.splitter._replace_leaf(self, self.children)
            for c in self.children:
                self.splitter._new_box(c)
                for r in self.results:
                    if c.contains(r.x):
                        c._register_result(r)
            self.splitter.eventbus.publish("new_split", box=self, children=self.children, dim=dim)

        def contains(self, point):
            """
            true, if given point is inside this box (including boundaries).
            """
            l, u = self.box[:, 0], self.box[:, 1]
            return (l <= point).all() and (u >= point).all()

        def get_child_boxes(self, point):
            """
            returns all immediate child boxes, which contain given point.
            """
            assert not self.leaf, 'not applicable for "leaf" box'
            ret = [c for c in self.children if c.contains(point)]
            assert len(ret) > 0, "no child box containing %s found!" % point
            return ret

        def __repr__(self):
            v = self.volume
            l = ",leaf" if self.leaf else ""
            l = "(%d,%.3f%s) " % (len(self), v, l)
            b = ",".join("%s" % _ for _ in self.box)
            return "Box-%d %s[%s]" % (self.id, l, b)
