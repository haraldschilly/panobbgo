# -*- coding: utf8 -*-
"""Budget-scaled resolution and the value-aware split rule.

``planning/DESIGN_meta_level_2026-09-10.md`` §1: the old
``limit = max(20, max_eval/dim**2)`` pinned the tree at ≈1.3·dim² leaves
*independent of the budget* — 5 leaves at *d* = 2, where the root could not
split before evaluation 250 of 1000 — and the split dimension was chosen
without ever looking at a function value.  These tests hold the two new
rules, the knobs that steer them, and the live-lock guard of §13 that both
of them have to keep.
"""

import hashlib
import json
import time

import numpy as np
import pytest

from panobbgo.analyzers import Splitter
from panobbgo.analyzers.splitter import LEAF_FILL, LEAF_SIZE
from panobbgo.lib import Point, Result
from panobbgo.lib.classic import Rosenbrock
from panobbgo.strategies import StrategyRoundRobin


def _splitter(dim, max_eval, **kwargs):
    s = StrategyRoundRobin(Rosenbrock(dim=dim), parse_args=False, testing_mode=True, seed=0)
    s.config.max_eval = max_eval
    s.config.sync_evaluation = True
    s.config.stop_on_convergence = False
    sp = Splitter(s, **kwargs)
    sp.__start__()
    return s, sp


def _feed(sp, xs, fxs):
    for x, fx in zip(xs, fxs):
        sp.root += Result(Point(np.asarray(x, dtype=float), "test"), float(fx), cv_vec=np.zeros(1))


def _uniform_cloud(strategy, n, seed=3):
    box = np.asarray(strategy.problem.box.box)
    lo, rg = box[:, 0], box[:, 1] - box[:, 0]
    rng = np.random.default_rng(seed)
    return lo + rg * rng.random((n, len(lo)))


# --- 1. the leaf-count rule ------------------------------------------------

#: ``(dim, budget)``, budget = 500·dim — the standard battery, extended.
BUDGETS = [(2, 1000), (5, 2500), (10, 5000), (20, 10000)]


@pytest.mark.parametrize("dim,budget", BUDGETS)
def test_limit_is_budget_scaled_not_dimension_pinned(dim, budget):
    """``limit`` follows ``leaf_size`` until the ``2·dim+2`` floor bites."""
    _, sp = _splitter(dim, budget)
    expected = max(round(LEAF_SIZE / LEAF_FILL), 2 * dim + 2)
    assert sp.limit == expected
    # The whole point: the target resolution is set by the budget.
    assert sp.target_leaves() == pytest.approx(budget / (LEAF_FILL * sp.limit))


@pytest.mark.parametrize("dim,budget", BUDGETS)
def test_realised_leaf_count_matches_the_target(dim, budget):
    """A uniform cloud of ``budget`` points lands within 20% of the target."""
    strategy, sp = _splitter(dim, budget)
    xs = _uniform_cloud(strategy, budget)
    _feed(sp, xs, np.sum((xs - 0.3) ** 2, axis=1))
    target = sp.target_leaves()
    assert 0.8 * target <= len(sp.leafs) <= 1.2 * target, f"d={dim}: {len(sp.leafs)} leafs, target {target:.0f}"
    # ... and it is *budget*-scaled, not dimension-pinned: the legacy rule
    # gives 1.3·dim² leaves, which at d=2 is single digits.
    if dim == 2:
        assert len(sp.leafs) > 20


def test_legacy_leaf_count_is_the_documented_defect():
    """The control: 1000 evaluations at d=2 buy a handful of leaves."""
    strategy, sp = _splitter(2, 1000, legacy=True)
    assert sp.limit == 250.0
    xs = _uniform_cloud(strategy, 1000)
    _feed(sp, xs, np.sum((xs - 0.3) ** 2, axis=1))
    assert len(sp.leafs) <= 8


def test_knobs_move_the_resolution():
    _, coarse = _splitter(5, 2500, leaf_size=250)
    _, fine = _splitter(5, 2500, leaf_size=12)
    _, capped = _splitter(5, 2500, leaf_size=1, max_leaves=10)
    assert coarse.limit > fine.limit
    assert coarse.target_leaves() < fine.target_leaves()
    assert capped.target_leaves() == pytest.approx(10, rel=0.15)
    _, floored = _splitter(5, 2500, min_leaf_size=400)
    assert floored.limit == 400


def test_split_rule_is_validated():
    with pytest.raises(ValueError):
        _splitter(3, 500, split_rule="nonsense")


def test_legacy_forces_the_widest_rule():
    _, sp = _splitter(3, 500, legacy=True, split_rule="value")
    assert sp.split_rule == "widest"


# --- 2. the value-aware split dimension ------------------------------------


def _separable_cloud(strategy, n, informative, seed=5):
    """A cloud whose objective depends, monotonically, on one coordinate."""
    xs = _uniform_cloud(strategy, n, seed=seed)
    fxs = xs[:, informative].copy()
    return xs, fxs


@pytest.mark.parametrize("informative", [1, 2, 3])
def test_value_rule_finds_the_informative_axis(informative):
    """f depends on one coordinate only; the cut has to land on it."""
    strategy, sp = _splitter(4, 400, split_rule="value", leaf_size=1000)
    xs, fxs = _separable_cloud(strategy, 120, informative)
    _feed(sp, xs, fxs)
    assert sp.root.leaf, "leaf_size=1000 should keep the root unsplit"
    assert sp.root._split_dim() == informative


def test_widest_rule_ignores_the_values():
    """The control: the historical rule cuts dimension 1 whatever f does.

    Rosenbrock's box is ``[0,2] x [-2,2]^3``, so dimension 0 is the narrow
    one and 1 is the first of the three widest — the widest rule returns it
    for every objective, which is exactly the blind spot ``"value"`` fixes.
    """
    strategy, sp = _splitter(4, 400, split_rule="widest", leaf_size=1000)
    xs, fxs = _separable_cloud(strategy, 120, informative=3)
    _feed(sp, xs, fxs)
    assert sp.root._split_dim() == 1


def test_value_rule_degrades_to_widest_without_a_signal():
    """A flat objective carries no separation, so width decides — as before.

    Tie-aware ranks are what makes this hold: ``argsort(argsort(v))`` would
    turn 120 equal values into ranks 0…119 in *arrival order* and the rule
    would cut on nothing at all.
    """
    strategy, sp = _splitter(4, 400, split_rule="value", leaf_size=1000)
    xs = _uniform_cloud(strategy, 120)
    _feed(sp, xs, np.ones(len(xs)))
    assert sp.root._split_dim() == 1


def test_value_rule_is_about_the_cut_not_the_dependence():
    """A cloud mirrored about the cut hides its own informative axis.

    Documented limitation, not a defect: the score asks whether *this* cut
    separates good points from bad.  Mirror the sample about the midpoint of
    axis 3 and every pair contributes the same objective value to both
    halves, so axis 3 — the only one the objective depends on — scores zero
    and loses to the sampling noise on the others.  See
    ``Box._split_dim_value``.  On a finite, non-mirrored sample the residual
    asymmetry is enough:
    :func:`test_value_rule_still_finds_a_symmetric_axis_on_a_real_sample`.
    """
    strategy, sp = _splitter(4, 400, split_rule="value", leaf_size=1000)
    half = _uniform_cloud(strategy, 100, seed=17)
    box = np.asarray(strategy.problem.box.box)
    mirror = half.copy()
    mirror[:, 3] = box[3].sum() - mirror[:, 3]  # reflect axis 3 about its midpoint
    xs = np.vstack([half, mirror])
    _feed(sp, xs, (xs[:, 3] - box[3].mean()) ** 2)
    assert sp.root._split_dim() != 3


def test_value_rule_still_finds_a_symmetric_axis_on_a_real_sample():
    """The mean of a finite sample is not the axis of symmetry, and the
    residual is enough: a plain quadratic in one coordinate still wins."""
    strategy, sp = _splitter(4, 400, split_rule="value", leaf_size=1000)
    xs = _uniform_cloud(strategy, 200, seed=17)
    _feed(sp, xs, xs[:, 3] ** 2)
    assert sp.root._split_dim() == 3


def test_value_rule_prefers_a_balanced_cut():
    """Two outliers on one axis must not outscore a real trend on another.

    Dimension 1 separates a single point from the rest (a huge rank gap on
    a 1-vs-n split); dimension 3 carries a monotone trend across the whole
    cloud.  The ``sqrt(n_l*n_r/n)`` factor of the rank-sum statistic is what
    makes the second win.
    """
    strategy, sp = _splitter(4, 400, split_rule="value", leaf_size=1000)
    n = 100
    xs = _uniform_cloud(strategy, n, seed=9)
    xs[:, 1] = -2.0
    xs[0, 1] = 2.0  # one point far away on axis 1
    fxs = xs[:, 3].copy()
    fxs[0] = -100.0  # ... and it is the best point
    _feed(sp, xs, fxs)
    assert sp.root._split_dim() == 3


def test_value_rule_survives_nan_and_inf():
    """Non-finite objective values rank as "worst" instead of poisoning the cut."""
    strategy, sp = _splitter(4, 400, split_rule="value", leaf_size=1000)
    xs, fxs = _separable_cloud(strategy, 120, informative=2)
    fxs = np.asarray(fxs, dtype=float)
    fxs[3] = np.nan
    fxs[7] = np.inf
    fxs[11] = -np.inf
    _feed(sp, xs, fxs)
    dim = sp.root._split_dim()
    assert dim is not None and 0 <= dim < 4


# --- 3. legacy is byte-identical -------------------------------------------

#: Pinned on ``2b835f5`` (``master``) *before* any change to the analyzer:
#: sha256 over the pre-order (id, depth, split_dim, #results, box) of every
#: box in the tree.  ``legacy=True`` must keep reproducing these exactly.
LEGACY_PINS = {
    (2, 300, 1000): ("91a06f5194e440e0cb376442ebe386664683d9de0ea1a67a7e85affd99e115f3", 2, 250.0),
    (3, 200, 400): ("bb425fbfc5ded8a3a676d6702ad3acfe56485e5d478398e8086bbf6aaafa9218", 7, 400 / 9),
    (5, 500, 2500): ("68db92cec9ff5a1dfab5511f53931d66450e2271f9d97ca8d065def1d125eae3", 8, 100.0),
}


def _tree_signature(sp):
    out = []

    def walk(b):
        out.append(
            [
                b.id,
                b.depth,
                b.split_dim,
                len(b.results),
                [
                    [round(float(v), 12) for v in row]
                    for row in np.asarray(b.box.box if hasattr(b.box, "box") else b.box)
                ],
            ]
        )
        for c in b.children:
            walk(c)

    walk(sp.root)
    return hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest()


@pytest.mark.parametrize("key", sorted(LEGACY_PINS))
def test_legacy_tree_is_byte_identical_to_the_pinned_one(key):
    dim, n, max_eval = key
    sha, n_leafs, limit = LEGACY_PINS[key]
    strategy, sp = _splitter(dim, max_eval, legacy=True)
    rng = np.random.default_rng(11)
    box = np.asarray(strategy.problem.box.box)
    lo, rg = box[:, 0], box[:, 1] - box[:, 0]
    for _ in range(n):
        x = lo + rg * rng.random(dim)
        _feed(sp, [x], [float(np.sum((x - 0.3) ** 2) + 0.1 * np.sin(5 * x[0]))])
    assert sp.limit == pytest.approx(limit)
    assert len(sp.leafs) == n_leafs
    assert _tree_signature(sp) == sha


# --- 4. the live-lock stays shut -------------------------------------------


@pytest.mark.parametrize("rule", ["widest", "value"])
def test_identical_points_never_split(rule):
    """§13's live-lock: a cut through identical points separates nothing."""
    _, sp = _splitter(3, 400, split_rule=rule)
    x = np.array([0.25, -0.5, 1.0])
    started = time.perf_counter()
    _feed(sp, [x.copy() for _ in range(10 * int(sp.limit))], np.arange(10 * int(sp.limit), dtype=float))
    assert time.perf_counter() - started < 10.0
    assert sp.root.leaf
    assert len(sp.leafs) == 1


@pytest.mark.parametrize("rule", ["widest", "value"])
def test_points_on_the_box_boundary_do_not_live_lock(rule):
    """Every point on a face of the box, half of them on each of two corners.

    ``contains`` includes both boundaries, so the cut through the mean puts
    nothing in both children *here* — but the children then hold clusters of
    identical coordinates, which is where the guard has to hold.
    """
    strategy, sp = _splitter(3, 400, split_rule=rule)
    box = np.asarray(strategy.problem.box.box)
    lo, hi = box[:, 0], box[:, 1]
    pts, fxs = [], []
    for i in range(12 * int(sp.limit)):
        x = lo.copy() if i % 2 else hi.copy()
        pts.append(x)
        fxs.append(float(i))
    started = time.perf_counter()
    _feed(sp, pts, fxs)
    assert time.perf_counter() - started < 10.0
    assert sp.max_depth <= Splitter.Box.MAX_DEPTH
    assert len(sp.leafs) <= 8, "two distinct corners cannot produce a deep tree"


@pytest.mark.parametrize("rule", ["widest", "value"])
def test_points_exactly_on_the_cut(rule):
    """A pile of points sitting exactly on the split point goes into *both*
    children; the tree must still terminate."""
    strategy, sp = _splitter(3, 400, split_rule=rule)
    box = np.asarray(strategy.problem.box.box)
    mid = box.mean(axis=1)
    rng = np.random.default_rng(4)
    pts, fxs = [], []
    for i in range(8 * int(sp.limit)):
        x = mid.copy()
        if i % 3 == 0:  # a few movers, so a cut exists at all
            x = box[:, 0] + (box[:, 1] - box[:, 0]) * rng.random(3)
        pts.append(x)
        fxs.append(float(rng.random()))
    started = time.perf_counter()
    _feed(sp, pts, fxs)
    assert time.perf_counter() - started < 15.0
    assert sp.max_depth <= Splitter.Box.MAX_DEPTH


def test_can_split_is_cheap_on_a_degenerate_leaf():
    """``_can_split`` must not re-scan the point cloud on every arrival.

    An unsplittable leaf sees ``_can_split`` on *every* result, so an
    O(#results) predicate makes the degenerate case quadratic.
    """
    _, sp = _splitter(10, 400, split_rule="value")
    x = np.zeros(10)
    n = 4000
    started = time.perf_counter()
    _feed(sp, [x.copy() for _ in range(n)], np.zeros(n))
    elapsed = time.perf_counter() - started
    assert sp.root.leaf
    assert elapsed < 5.0, f"{n} identical points took {elapsed:.1f}s"


# --- 5. through the consumers ----------------------------------------------


@pytest.mark.parametrize("rule", ["widest", "value"])
@pytest.mark.parametrize("consumer", ["random", "region_ucb"])
def test_consumers_spend_the_whole_budget(rule, consumer):
    """The two most direct consumers still run a full budget on the new tree.

    ``RegionUCB`` has no ``on_start`` and only emits from ``on_new_results``,
    so it cannot bootstrap itself and is paired with a capped ``Random``.
    That is a pre-existing property of the heuristic, not of the tree — the
    same spec produces zero evaluations under ``legacy=True``.
    """
    from panobbgo.heuristics import Random, RegionUCB

    s = StrategyRoundRobin(Rosenbrock(dim=3), parse_args=False, testing_mode=True, seed=0)
    s.config.max_eval = 300
    s.config.sync_evaluation = True
    s.config.stop_on_convergence = False
    s.add_analyzer(Splitter(s, split_rule=rule))
    if consumer == "random":
        s.add_heuristic(Random(s))
    else:
        s.add_heuristic(RegionUCB(s))
        s.add_heuristic(Random(s, cap=2))
    s.start()
    assert len(s.results) >= 300, f"{consumer}/{rule}: {len(s.results)}/300"
    splitter = s.analyzer("Splitter")
    assert splitter.split_rule == rule
    assert len(splitter.leafs) > 1


def test_archive_per_leaf_best_sees_the_finer_tree():
    """``archive_leaf`` warm starts hand out one point per leaf, so the number
    of *different basins* it can offer is the tree's resolution."""
    from panobbgo.analyzers import Archive
    from panobbgo.heuristics import Random

    seeds = {}
    for name, kwargs in (("old", {"legacy": True}), ("new", {})):
        s = StrategyRoundRobin(Rosenbrock(dim=2), parse_args=False, testing_mode=True, seed=0)
        s.config.max_eval = 600
        s.config.sync_evaluation = True
        s.config.stop_on_convergence = False
        s.add_analyzer(Splitter(s, **kwargs))
        s.add_analyzer(Archive(s))
        s.add_heuristic(Random(s))
        s.start()
        seeds[name] = len(s.analyzer("Archive").per_leaf_best(32))
    assert seeds["new"] > seeds["old"], seeds
    assert seeds["old"] <= 8, "the legacy tree at d=2 cannot offer more than a handful"


# --- 6. where the cut falls: mean vs median --------------------------------


def _cut(sp, dim=0):
    """The split point ``Box.split`` would use on the root."""
    return float(sp.root._split_point(dim))


def test_cut_rule_is_validated_and_legacy_forces_the_mean():
    with pytest.raises(ValueError):
        _splitter(3, 500, cut_rule="quartile")
    _, sp = _splitter(3, 500, legacy=True, cut_rule="median")
    assert sp.cut_rule == "mean"


def test_mean_cut_is_unchanged():
    """``cut_rule="mean"`` is the historical expression, bit for bit."""
    strategy, sp = _splitter(3, 5000, cut_rule="mean", leaf_size=10000)
    xs = _uniform_cloud(strategy, 40, seed=2)
    _feed(sp, xs, np.arange(40, dtype=float))
    assert _cut(sp, 1) == np.average([r.x[1] for r in sp.root.results])


@pytest.mark.parametrize(
    "coords",
    [
        [0.0] * 30 + [1.0] * 3,  # the plain-median trap: median == min
        [0.0] * 3 + [1.0] * 30,  # ... and its mirror: median == max
        [0.0, 1.0],  # the smallest possible split
        [0.0] * 20 + [0.5] * 20 + [1.0] * 20,  # a median sitting on a big tie
        list(np.linspace(0.0, 1.0, 41)),  # all distinct, odd count
        list(np.linspace(0.0, 1.0, 40)),  # all distinct, even count
        [0.0] * 19 + [1e-12] + [1.0] * 20,  # a near-degenerate gap
    ],
)
def test_median_cut_is_strictly_interior_and_off_the_data(coords):
    """The two invariants a cut has to satisfy, on adversarial coordinates.

    *Strictly interior* keeps each child a proper subset of its parent.
    *Not equal to any observed coordinate* is the second one, and the one a
    plain median violates: ``contains`` includes both boundaries, so points
    sitting on the cut are counted into **both** children and neither box
    shrinks.
    """
    strategy, sp = _splitter(3, 100000, cut_rule="median", leaf_size=1000000)
    coords = np.asarray(coords, dtype=float)
    box = np.asarray(strategy.problem.box.box)
    lo, hi = box[1, 0], box[1, 1]
    rng = np.random.default_rng(1)
    xs = _uniform_cloud(strategy, len(coords), seed=8)
    xs[:, 1] = lo + (hi - lo) * coords  # put the pattern on dimension 1
    _feed(sp, xs, rng.random(len(coords)))

    cut = _cut(sp, 1)
    observed = xs[:, 1]
    assert observed.min() < cut < observed.max(), "cut is not strictly interior"
    assert not np.any(observed == cut), "cut sits on an observed coordinate"
    # Both children are proper subsets, which is what the two invariants buy.
    assert 0 < (observed <= cut).sum() < len(observed)
    assert 0 < (observed >= cut).sum() < len(observed)


def test_median_cut_balances_a_skewed_cloud_better_than_the_mean():
    """On a cloud with a long tail the median halves it; the mean does not."""
    strategy, _ = _splitter(3, 100, leaf_size=1000000)
    box = np.asarray(strategy.problem.box.box)
    lo, hi = box[1, 0], box[1, 1]
    # A dense cluster holding 3/4 of the points, plus a far tail: the mean is
    # dragged out to the tail and cuts 60-against-20, the median halves the
    # cluster itself and cuts 40-against-40.
    coords = np.concatenate([np.linspace(0.0, 0.05, 60), np.linspace(0.9, 1.0, 20)])

    balance = {}
    for rule in ("mean", "median"):
        strategy, sp = _splitter(3, 100000, cut_rule=rule, leaf_size=1000000)
        xs = _uniform_cloud(strategy, len(coords), seed=8)
        xs[:, 1] = lo + (hi - lo) * coords
        _feed(sp, xs, np.arange(len(coords), dtype=float))
        cut = _cut(sp, 1)
        left = int((xs[:, 1] <= cut).sum())
        balance[rule] = abs(2 * left - len(coords))
    assert balance["median"] < balance["mean"], balance


@pytest.mark.parametrize("rule", ["mean", "median"])
def test_identical_points_never_split_under_either_cut(rule):
    """The §13 guard is upstream of the cut and must hold for both."""
    _, sp = _splitter(3, 400, cut_rule=rule)
    x = np.array([0.25, -0.5, 1.0])
    n = 10 * int(sp.limit)
    _feed(sp, [x.copy() for _ in range(n)], np.arange(n, dtype=float))
    assert sp.root.leaf
    assert len(sp.leafs) == 1


@pytest.mark.parametrize("rule", ["mean", "median"])
def test_a_contracting_search_keeps_its_leaves_bounded(rule):
    """No leaf may exceed ``limit`` unless the depth cap stopped it."""
    strategy, sp = _splitter(5, 2500, cut_rule=rule)
    rng = np.random.default_rng(0)
    box = np.asarray(strategy.problem.box.box)
    lo, rg = box[:, 0], box[:, 1] - box[:, 0]
    centre = lo + 0.37 * rg
    for i in range(2500):
        scale = max(1e-3, 1.0 - i / 2500)
        x = np.clip(centre + scale * rg * (rng.random(5) - 0.5), lo, lo + rg)
        _feed(sp, [x], [float(np.sum((x - centre) ** 2))])
    for box_ in sp.leafs:
        assert len(box_) < sp.limit or box_.depth >= Splitter.Box.MAX_DEPTH
