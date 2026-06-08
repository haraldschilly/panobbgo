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

from panobbgo.utils import PanobbgoTestCase


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
# Self-improvement loop catalog rules
# ----------------------------------------------------------------------


def test_kwarg_catalog_has_region_ucb_ucb_c_rule():
    """``default_catalog`` ships a ``RegionUCB.ucb_c`` log-uniform rule.

    The rule lets the autonomous loop tune the UCB1 exploration weight
    on specs that opt into a concrete ``ucb_c`` value.  The seed
    :func:`~panobbgo.harness._make_standard_strategies`
    ``Rewarding_RegionUCB`` spec sets the kwarg explicitly (matching
    the constructor default of ``1.0``) so the rule fires on the
    standard-mode battery out-of-the-box.
    """
    from panobbgo.self_improve import MutationRule, default_catalog

    rules = [
        r
        for r in default_catalog().rules
        if isinstance(r, MutationRule) and r.class_name == "RegionUCB" and r.param_name == "ucb_c"
    ]
    assert len(rules) == 1, "expected exactly one RegionUCB.ucb_c rule"
    rule = rules[0]
    assert rule.kind == "log_uniform_perturb"
    assert rule.bounds == (0.1, 4.0)
    # Bounds bracket the literature default of 1.0.
    assert rule.bounds[0] < 1.0 < rule.bounds[1]


def test_kwarg_catalog_has_region_ucb_gauss_fraction_rule():
    """``default_catalog`` ships a ``RegionUCB.gauss_fraction`` float_uniform rule.

    ``gauss_fraction`` is the fraction of in-leaf draws taken as a
    Gaussian around the leaf's best point instead of uniform over the
    leaf box.  The full ``[0, 1]`` range is bandit-reachable so the
    loop can probe the LA-MCTS-style pure-uniform regime
    (``gauss_fraction = 0.0``) and the pure-local-refinement regime
    (``gauss_fraction = 1.0``) symmetrically.
    """
    from panobbgo.self_improve import MutationRule, default_catalog

    rules = [
        r
        for r in default_catalog().rules
        if isinstance(r, MutationRule) and r.class_name == "RegionUCB" and r.param_name == "gauss_fraction"
    ]
    assert len(rules) == 1, "expected exactly one RegionUCB.gauss_fraction rule"
    rule = rules[0]
    assert rule.kind == "float_uniform"
    assert rule.bounds == (0.0, 1.0)
    assert rule.low == 0.0 and rule.high == 1.0


def test_kwarg_catalog_has_region_ucb_gauss_scale_rule():
    """``default_catalog`` ships a ``RegionUCB.gauss_scale`` log_uniform rule.

    ``gauss_scale`` is the Gaussian-around-best std-dev expressed as a
    fraction of the leaf's ranges; ``0.25`` (the constructor default)
    lives near the geometric centre of the ``[0.05, 0.5]`` log-uniform
    window so symmetric perturbations can both shrink and widen the
    in-leaf Gaussian cloud.
    """
    from panobbgo.self_improve import MutationRule, default_catalog

    rules = [
        r
        for r in default_catalog().rules
        if isinstance(r, MutationRule) and r.class_name == "RegionUCB" and r.param_name == "gauss_scale"
    ]
    assert len(rules) == 1, "expected exactly one RegionUCB.gauss_scale rule"
    rule = rules[0]
    assert rule.kind == "log_uniform_perturb"
    assert rule.bounds == (0.05, 0.5)


def test_region_ucb_rules_skip_implicit_default():
    """The ``RegionUCB.*`` rules must not fire on specs that omit the
    kwarg from the kwargs dict (i.e. specs that left the kwarg at the
    heuristic's constructor default).  The ``_find_targets`` "param
    already in kwargs" predicate keeps the rule out of those specs.
    """
    import numpy as np

    from panobbgo.benchmark import StrategySpec
    from panobbgo.heuristics import RegionUCB
    from panobbgo.self_improve import MutationCatalog, MutationRule
    from panobbgo.strategies.rewarding import StrategyRewarding

    rule = MutationRule(
        strategy_pattern="",
        class_name="RegionUCB",
        param_name="ucb_c",
        kind="log_uniform_perturb",
        bounds=(0.1, 4.0),
        log_step=0.15,
        probability=1.0,
    )

    spec_implicit = StrategySpec(
        name="StratImplicit",
        strategy_class=StrategyRewarding,
        heuristics=[(RegionUCB, {})],
        analyzers=[],
    )
    spec_explicit = StrategySpec(
        name="StratExplicit",
        strategy_class=StrategyRewarding,
        heuristics=[(RegionUCB, {"ucb_c": 1.0})],
        analyzers=[],
    )
    cat = MutationCatalog([rule])
    rng = np.random.default_rng(0)

    for _ in range(30):
        prop = cat.sample(rng, [spec_implicit, spec_explicit])
        assert prop is not None
        assert prop.strategy_name == "StratExplicit"


def test_rewarding_region_ucb_seed_spec_has_explicit_region_ucb_kwargs():
    """The seed ``Rewarding_RegionUCB`` spec in
    :func:`~panobbgo.harness._make_standard_strategies` must ship the
    catalog-discoverable kwargs explicitly so the
    ``RegionUCB.ucb_c`` / ``RegionUCB.gauss_fraction`` /
    ``RegionUCB.gauss_scale`` rules become applicable on the
    standard-mode battery rather than staying dormant.

    The explicit values match the constructor defaults so RegionUCB
    construction stays byte-identical — only the kwarg dict's
    *membership* changes.
    """
    from panobbgo.harness import _make_standard_strategies
    from panobbgo.heuristics import RegionUCB

    specs = {s.name: s for s in _make_standard_strategies()}
    assert "Rewarding_RegionUCB" in specs
    spec = specs["Rewarding_RegionUCB"]
    region_kwargs = [kw for cls, kw in spec.heuristics if cls is RegionUCB]
    assert len(region_kwargs) == 1
    kw = region_kwargs[0]
    assert kw["ucb_c"] == 1.0
    assert kw["gauss_fraction"] == 0.5
    assert kw["gauss_scale"] == 0.25
