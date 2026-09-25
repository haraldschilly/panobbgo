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

"""Unit tests for the RegionUCB heuristic (UCB over Splitter leaves)."""

from __future__ import unicode_literals

import numpy as np

from tests.support import PanobbgoTestCase


class _FakeLeaf:
    """Minimal stand-in for Splitter.Box: bounds, results, best."""

    _next_id = 0

    def __init__(self, box, results=None, best=None, depth=1):
        self.box = np.asarray(box, dtype=float)
        self.results = results if results is not None else []
        self.best = best
        self.depth = depth
        self.id = _FakeLeaf._next_id
        _FakeLeaf._next_id += 1

    @property
    def ranges(self):
        return np.ptp(self.box, axis=1)

    def __len__(self):
        return len(self.results)


class _FakeBest:
    def __init__(self, x, fx):
        self.x = np.asarray(x, dtype=float)
        self.fx = fx
        self.cv = 0.0


class RegionUCBTest(PanobbgoTestCase):
    def make_heuristic(self, **kwargs):
        from panobbgo.heuristics import RegionUCB

        h = RegionUCB(self.strategy, **kwargs)
        # PanobbgoTestCase mocks the strategy; constraint handler off
        self.strategy.constraint_handler = None
        return h

    def test_empty_leaf_selected_first(self):
        """A leaf with no points must win over any populated leaf."""
        h = self.make_heuristic()
        full = _FakeLeaf([[0, 1], [0, 1]], results=[object()] * 10, best=_FakeBest([0.5, 0.5], 0.0))
        empty = _FakeLeaf([[1, 2], [1, 2]])
        assert h.select_leaf([full, empty]) is empty

    def test_best_leaf_preferred_at_equal_counts(self):
        """With equal counts, the leaf with better fx has the higher score."""
        h = self.make_heuristic(ucb_c=0.1)
        good = _FakeLeaf([[0, 1], [0, 1]], results=[object()] * 5, best=_FakeBest([0.5, 0.5], 1.0))
        bad = _FakeLeaf([[1, 2], [1, 2]], results=[object()] * 5, best=_FakeBest([1.5, 1.5], 100.0))
        assert h.select_leaf([good, bad]) is good

    def test_exploration_term_lifts_undersampled_leaf(self):
        """A much-less-sampled leaf wins when ucb_c is large."""
        h = self.make_heuristic(ucb_c=10.0)
        exploited = _FakeLeaf([[0, 1], [0, 1]], results=[object()] * 100, best=_FakeBest([0.5, 0.5], 0.0))
        fresh = _FakeLeaf([[1, 2], [1, 2]], results=[object()] * 2, best=_FakeBest([1.5, 1.5], 50.0))
        assert h.select_leaf([exploited, fresh]) is fresh

    def test_samples_stay_inside_leaf_box(self):
        """All sampled points (uniform + gaussian) lie within the leaf bounds."""
        h = self.make_heuristic(n_candidates=50, gauss_fraction=0.5)
        leaf = _FakeLeaf([[2.0, 3.0], [-1.0, 0.5]], best=_FakeBest([2.9, 0.4], 1.0))
        pts = h.sample_in_leaf(leaf)
        assert len(pts) == 50
        for p in pts:
            assert (p >= leaf.box[:, 0]).all() and (p <= leaf.box[:, 1]).all()

    def test_no_best_means_pure_uniform(self):
        """Without a best point, all candidates are uniform draws in the box."""
        h = self.make_heuristic(n_candidates=20, gauss_fraction=0.5)
        leaf = _FakeLeaf([[0.0, 1.0], [0.0, 1.0]])
        pts = h.sample_in_leaf(leaf)
        assert len(pts) == 20
        for p in pts:
            assert (p >= 0.0).all() and (p <= 1.0).all()

    def test_on_new_results_emits_into_queue(self):
        """End-to-end handler: leaf from fake splitter -> points in output queue."""
        h = self.make_heuristic(n_candidates=5)
        leaf = _FakeLeaf([[0, 1], [0, 1]], results=[object()] * 3, best=_FakeBest([0.5, 0.5], 1.0))

        class _FakeSplitter:
            leafs = [leaf]

        self.strategy.analyzer.side_effect = None
        self.strategy.analyzer.return_value = _FakeSplitter()

        h.on_new_results([])
        pts = h.get_points()
        assert len(pts) == 5

    def test_on_new_results_without_splitter_falls_back(self):
        """No Splitter analyzer -> samples the full problem box."""
        h = self.make_heuristic(n_candidates=5)
        self.strategy.analyzer.side_effect = Exception("no analyzer")

        h.on_new_results([])
        pts = h.get_points()
        assert len(pts) == 5


# ----------------------------------------------------------------------
# The heuristic must be able to open a run on its own
# ----------------------------------------------------------------------


def test_region_ucb_alone_spends_its_budget():
    """``RegionUCB`` used to emit nothing when run without a second arm.

    Every other handler is driven by ``on_new_results``, and the Splitter
    tree it allocates over does not exist until results arrive — so with no
    ``on_start`` the strategy stalled at **0 evaluations** and died on the
    no-progress guard.  It was only ever benchmarked next to another arm,
    which hid the defect.
    """
    from panobbgo.heuristics import RegionUCB
    from panobbgo.lib.classic import Rosenbrock
    from panobbgo.strategies import StrategyRoundRobin

    strategy = StrategyRoundRobin(Rosenbrock(dim=3), parse_args=False, testing_mode=True, seed=0)
    strategy.config.max_eval = 200
    strategy.config.sync_evaluation = True
    strategy.config.stop_on_convergence = False
    strategy.add_heuristic(RegionUCB(strategy))
    strategy.start()
    assert len(strategy.results) >= 200, f"only {len(strategy.results)}/200 evaluations"


def test_region_ucb_on_start_fills_the_queue_from_its_own_rng():
    """The opening design is uniform in the problem box and uses ``self.rng``.

    ``self.rng`` rather than ``strategy.rng``: the strategy-level stream
    belongs to the strategy's own draws (Thompson sampling, phases), which
    a module drawing from it would shift.
    """
    import numpy as np

    from panobbgo.heuristics import RegionUCB
    from panobbgo.lib.classic import Rosenbrock
    from panobbgo.strategies import StrategyRoundRobin

    strategy = StrategyRoundRobin(Rosenbrock(dim=4), parse_args=False, testing_mode=True, seed=0)
    strategy.config.max_eval = 100
    try:
        h = RegionUCB(strategy)
        before = strategy.rng.bit_generator.state
        h.on_start()
        assert strategy.rng.bit_generator.state == before, "on_start drew from the master stream"
        points = h.get_points()
        assert len(points) > 0
        box = np.asarray(strategy.problem.box.box)
        for p in points:
            x = np.asarray(getattr(p, "x", p), dtype=float)
            assert (x >= box[:, 0]).all() and (x <= box[:, 1]).all()
    finally:
        strategy._cleanup()
