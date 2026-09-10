# -*- coding: utf8 -*-
"""The meta level: :class:`~panobbgo.heuristics.meta.MetaAnalyst`.

``planning/DESIGN_meta_level_2026-09-10.md``.  The properties pinned here,
in the order the design's §6 asks for them:

* **the null** — ``trigger="never"`` produces a trajectory bit-identical to
  the same spec *without* the module.  Every later number in the experiment
  is uninterpretable if this fails (design §5, last row);
* the trigger algebra: ``budget_fraction`` fires once at the fraction,
  ``stagnation`` is budget-relative and fires on a flat trace but not on an
  improving one, ``&`` / ``|`` compose;
* the two mandatory guards — a refractory period and the hard ``meta_frac``
  cap on the evaluations the module may emit;
* the scan: leaves with a high volume-per-point ratio *and* a good rank
  win, and the emitted points land inside the chosen leaf's box;
* the region hand-off: ``meta_region`` only *records* a request, and
  ``_open_block`` applies it exactly once;
* and that a run with the module still spends its whole budget, under the
  block scheduler and under round-robin.
"""

from __future__ import annotations

import numpy as np
import pytest

from panobbgo.analyzers import Archive, Splitter
from panobbgo.core import Heuristic
from panobbgo.heuristics.meta import (
    MetaAnalyst,
    MetaContext,
    Trigger,
    budget_fraction,
    stagnation,
)
from panobbgo.lib import Point, Result
from panobbgo.lib.classic import Rosenbrock
from panobbgo.strategies import StrategyBlockBandit, StrategyRoundRobin

MAX_EVAL = 400


class Always(Trigger):
    """A trigger that always fires — isolates the guards from the rule."""

    @property
    def reason(self):
        return "always"

    def check(self, ctx):
        return self


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def _res(x, fx, who="A"):
    return Result(Point(np.asarray(x, dtype=float), who), float(fx))


def _batch(n, fx=1.0, who="A"):
    return [_res(np.zeros(2), fx, who) for _ in range(n)]


def _probe(max_eval=MAX_EVAL, dim=2, **kw):
    """An unstarted strategy with a ``Splitter`` and a hand-driven meta arm."""
    s = StrategyRoundRobin(Rosenbrock(dim=dim), parse_args=False, testing_mode=True, seed=3)
    s.config.max_eval = max_eval
    s.config.sync_evaluation = True
    s.config.stop_on_convergence = False
    s.add_analyzer(Splitter(s))
    m = MetaAnalyst(s, **kw)
    s.add_heuristic(m)
    return s, m


def _leaf(splitter, bounds, results):
    """A synthetic ``Splitter.Box`` with a hand-set point set."""
    b = Splitter.Box(None, splitter, np.array(bounds, dtype=float))
    b.results = list(results)
    b.best = min(results, key=lambda r: r.fx) if results else None
    return b


def _three_leaves(s):
    """A: sparse + good, B: less sparse + bad, C: dense + best.

    All three sit inside ``Rosenbrock(dim=2)``'s box ``[0,2] x [-2,2]`` (log
    volume 2.079), so ``Problem.project`` cannot move an emitted point out of
    the leaf it was drawn in.  Hand-computed against the documented score:
    ``deficit`` −1.73 / −1.32 / +6.57 ⇒ ``sparsity`` 1.0 / 0.5 / 0.0, leaf
    bests 1.0 / 100.0 / 0.5 ⇒ ``quality`` 0.5 / 0.0 / 1.0.  So **A wins the
    blend** (0.75 vs 0.25 vs 0.50), A wins the pure volume-vs-count scan
    (``w = 0``) and C wins pure quality (``w = 1``).
    """
    sp = s.analyzer("Splitter")
    a = _leaf(sp, [[0.0, 1.0], [-2.0, 0.0]], [_res([0.5, -1.0], 1.0), _res([0.2, -1.5], 3.0)])
    b = _leaf(
        sp,
        [[1.0, 2.0], [-2.0, 0.0]],
        [_res([1.5, -1.0], 100.0), _res([1.2, -1.5], 200.0), _res([1.8, -0.5], 300.0)],
    )
    c = _leaf(sp, [[0.9, 1.0], [0.9, 1.0]], [_res([0.95, 0.95], 0.5 + i) for i in range(40)])
    sp.leafs = [a, b, c]
    return a, b, c


def _trajectory(s):
    df = s.results.results
    assert df is not None
    x = df["x"].to_numpy(dtype=float)
    fx = df[("fx", 0)].to_numpy(dtype=float).ravel()
    who = df[("who", 0)].to_numpy().ravel().astype(str)
    return x, fx, who


def _portfolio(with_meta, *, seed=7, max_eval=200, dim=2, meta_kwargs=None):
    """The reference two-arm warm portfolio, optionally plus the meta arm.

    Built exactly as ``StrategySpec.create_strategy`` builds it — heuristics
    in list order, extra analyzers afterwards — so the module construction
    order (and therefore every arm's RNG stream) is the one the benchmark
    would produce.
    """
    from panobbgo.heuristics import CMAES, JSO

    s = StrategyBlockBandit(
        Rosenbrock(dim=dim),
        parse_args=False,
        testing_mode=True,
        seed=seed,
        policy="uniform",
        warm_start_on_resume=True,
        warm_start_only_if_foreign=False,
        warm_start_only_if_better=False,
    )
    s.config.max_eval = max_eval
    s.config.sync_evaluation = True
    s.config.stop_on_convergence = False
    s.config.evaluation_method = "threaded"
    s.add(CMAES, warm_start="archive")
    s.add(JSO, NP_init=8, warm_start="archive")
    if with_meta:
        s.add(MetaAnalyst, **(meta_kwargs or {"trigger": "never"}))
    s.add_analyzer(Archive(s))
    s.start()
    return s


# ----------------------------------------------------------------------
# 1 -- the null: trigger="never" changes nothing at all
# ----------------------------------------------------------------------


def test_never_is_bit_identical_to_the_spec_without_the_module():
    xa, fa, wa = _trajectory(_portfolio(False))
    xb, fb, wb = _trajectory(_portfolio(True))

    assert len(fa) == len(fb)
    np.testing.assert_array_equal(xa, xb)
    np.testing.assert_array_equal(fa, fb)
    assert list(wa) == list(wb)
    # ... and the module really was there, silent
    assert "Meta" not in set(wb)


def test_never_short_circuits_before_it_reads_anything():
    _, m = _probe(trigger="never")
    m.on_new_results(_batch(MAX_EVAL))
    assert m._n_evals == 0 and not m.firings and not m._trace


def test_a_silent_meta_arm_is_never_given_a_block():
    s = _portfolio(True)
    assert "Meta" in s._heuristics
    assert all(b["owner"] != "Meta" for b in s._blocks)


# ----------------------------------------------------------------------
# 2 -- triggers
# ----------------------------------------------------------------------


def test_budget_fraction_fires_once_at_the_fraction():
    s, m = _probe(trigger=budget_fraction(0.25), k=8, min_leafs=1)
    _three_leaves(s)

    m.on_new_results(_batch(99))
    assert m.firings == []
    m.on_new_results(_batch(1))  # crosses 0.25 * 400
    assert len(m.firings) == 1 and m.firings[0]["eval"] == 100
    assert m.firings[0]["reason"] == "b25"

    # one-shot: no second firing, however long the run goes on
    for _ in range(10):
        m.on_new_results(_batch(20))
    assert len(m.firings) == 1


def test_who_carries_the_reason():
    s, m = _probe(trigger=budget_fraction(0.25), k=4, min_leafs=1)
    _three_leaves(s)
    m.on_new_results(_batch(100))
    points = m.get_points()
    assert len(points) == 4
    assert {p.who for p in points} == {"Meta:b25"}


def test_refractory_period_is_enforced():
    s, m = _probe(trigger=Always(), k=1, min_leafs=1, meta_frac=1.0)
    _three_leaves(s)
    assert m.refractory() == 20  # max(block_evals=0, 0.05 * 400)

    m.on_new_results(_batch(30))
    assert len(m.firings) == 1
    for _ in range(19):  # 19 more evaluations: still inside the window
        m.on_new_results(_batch(1))
    assert len(m.firings) == 1
    m.on_new_results(_batch(1))
    assert len(m.firings) == 2
    assert m.firings[1]["eval"] - m.firings[0]["eval"] == m.refractory()


def test_meta_frac_caps_the_emitted_share_of_the_budget():
    s, m = _probe(trigger=Always(), k=8, min_leafs=1, meta_frac=0.05)
    _three_leaves(s)
    cap = int(0.05 * MAX_EVAL)

    for _ in range(12):
        m.on_new_results(_batch(20))
    assert m._emitted == cap == 20
    assert len(m.get_points()) == cap
    assert m.refusals.get("meta_frac", 0) > 0


def test_stagnation_fires_on_a_flat_trace_and_not_on_an_improving_one():
    trig = stagnation(0.10, tol=1e-8)
    flat = MetaContext(400, 400, 2, [1.0] * 200)
    assert trig.check(flat) is trig
    improving = MetaContext(400, 400, 2, [1.0 / (i + 1) for i in range(200)])
    assert trig.check(improving) is None
    # not enough history yet -> no verdict
    assert trig.check(MetaContext(10, 400, 2, [1.0] * 10)) is None


def test_stagnation_window_is_budget_relative():
    trig = stagnation(0.10)
    assert trig.window(MetaContext(0, 1000, 5, [])) == 100
    assert trig.window(MetaContext(0, 2000, 5, [])) == 200
    # ... with an absolute floor of 4*dim for tiny budgets
    assert trig.window(MetaContext(0, 10, 5, [])) == 20


def test_triggers_compose():
    early = MetaContext(200, 400, 2, [1.0] * 200)
    late = MetaContext(300, 400, 2, [1.0] * 300)
    b, st = budget_fraction(0.6), stagnation(0.10)

    assert (b | st).check(early) is st  # only stagnation holds
    assert (b & st).check(early) is None
    assert (b & st).check(late) is not None
    assert (b & st).reason == "b60&stag"
    assert (b | st).reason == "b60|stag"


def test_trigger_argument_validation():
    with pytest.raises(ValueError):
        budget_fraction(1.5)
    with pytest.raises(ValueError):
        stagnation(0.0)
    with pytest.raises(ValueError):
        MetaAnalyst(_probe()[0], trigger="whenever")
    with pytest.raises(ValueError):
        MetaAnalyst(_probe()[0], mode="magic")


# ----------------------------------------------------------------------
# 3 -- the scan
# ----------------------------------------------------------------------


def test_scan_prefers_high_volume_per_point_and_a_good_rank():
    s, m = _probe(trigger=budget_fraction(0.25), k=6, min_leafs=1, quality_weight=0.5)
    a, b, c = _three_leaves(s)

    ordered = m.scan()
    assert [leaf.id for _, leaf in ordered] == [a.id, c.id, b.id]
    scores = {leaf.id: sc for sc, leaf in ordered}
    assert scores[a.id] == pytest.approx(0.75)
    assert scores[c.id] == pytest.approx(0.50)
    assert scores[b.id] == pytest.approx(0.25)


def test_the_two_terms_of_the_score_can_be_isolated():
    s, m = _probe(trigger=budget_fraction(0.25), min_leafs=1, quality_weight=0.0)
    a, _, _ = _three_leaves(s)
    assert m.scan()[0][1].id == a.id  # pure volume-vs-count

    s2, m2 = _probe(trigger=budget_fraction(0.25), min_leafs=1, quality_weight=1.0)
    _, _, c2 = _three_leaves(s2)
    assert m2.scan()[0][1].id == c2.id  # pure leaf quality


def test_emitted_points_lie_inside_the_chosen_leaf():
    s, m = _probe(trigger=budget_fraction(0.25), k=12, n_leaves=1, min_leafs=1)
    a, _, _ = _three_leaves(s)

    m.on_new_results(_batch(100))
    points = m.get_points()
    assert len(points) == 12
    assert all(a.contains(p.x) for p in points), [p.x for p in points]


def test_random_mode_ignores_the_analysis():
    """The falsifier arm: same trigger, same ``k``, no analysis."""
    s, m = _probe(trigger=budget_fraction(0.25), k=20, n_leaves=1, min_leafs=1, mode="random")
    a, _, _ = _three_leaves(s)
    m.on_new_results(_batch(100))
    points = m.get_points()
    assert len(points) == 20
    # uniform over the whole box: essentially impossible to stay in one leaf
    assert not all(a.contains(p.x) for p in points)


def test_too_few_leaves_refuses_to_fire():
    s, m = _probe(trigger=budget_fraction(0.25), k=4, min_leafs=8)
    _three_leaves(s)  # only three
    m.on_new_results(_batch(100))
    assert m.firings == [] and m.refusals.get("leafs", 0) == 1


def test_the_tail_of_the_budget_refuses_to_fire():
    s, m = _probe(trigger=budget_fraction(0.90), k=4, min_leafs=1, tail_frac=0.25)
    _three_leaves(s)
    m.on_new_results(_batch(int(0.90 * MAX_EVAL)))
    assert m.firings == [] and m.refusals.get("tail", 0) == 1


# ----------------------------------------------------------------------
# 4 -- the region hand-off
# ----------------------------------------------------------------------


class Boxed(Heuristic):
    """A warm-startable stub that records the box it is handed."""

    def __init__(self, strategy, name="Boxed"):
        Heuristic.__init__(self, strategy, name=name, cap=8)
        self.warm_start = "archive"
        self.warm_start_box = None
        self.seen: list = []

    def _generation(self):
        self.emit([self.problem.random_point(rng=self.rng) for _ in range(4)])

    def on_start(self):
        self._generation()

    def on_new_results(self, results):
        pass

    def warm_start_now(self) -> bool:
        self.seen.append(self.warm_start_box)
        self._generation()
        return True


def _scheduler(**kw):
    s = StrategyBlockBandit(
        Rosenbrock(dim=2),
        parse_args=False,
        testing_mode=True,
        seed=5,
        warm_start_on_resume=True,
        warm_start_only_if_foreign=False,
        warm_start_only_if_better=False,
        **kw,
    )
    s.config.max_eval = 200
    s.config.sync_evaluation = True
    s.config.stop_on_convergence = False
    h = Boxed(s)
    s.add_heuristic(h)
    s.on_new_results([_res([0.1, 0.1], 1.0, "Other")])  # something to warm-start from
    return s, h


def test_meta_region_is_only_recorded_by_the_handler():
    s, h = _scheduler()
    box = np.array([[-1.0, 1.0], [-1.0, 1.0]])
    s.on_meta_region(arm="Boxed", box=box)
    # the handler runs on the bus thread: it must not have touched the arm
    assert s._pending_region == {"Boxed": box}
    assert h.seen == [] and h.warm_start_box is None

    s.on_meta_region(arm="Nobody", box=box)  # unknown arm: ignored
    s.on_meta_region(arm="Boxed", box=None)  # no box: ignored
    assert list(s._pending_region) == ["Boxed"]


def test_open_block_applies_the_region_exactly_once():
    s, h = _scheduler()
    box = np.array([[-1.0, 1.0], [-1.0, 1.0]])
    s.on_meta_region(arm="Boxed", box=box)

    s._open_block(h)
    assert h.seen == [box], "the arm must have been warm-started with the box"
    assert h.warm_start_box is None, "the hand-off is one-shot"
    assert s._pending_region == {}
    assert s._regions_applied == {"Boxed": 1}

    s._close_block()
    assert s._blocks[-1]["region"] is True
    s._open_block(h)  # a consecutive block, no new request
    assert h.seen == [box]
    assert s._regions_applied == {"Boxed": 1}


def test_region_forces_a_warm_start_the_gap_rule_would_skip():
    s, h = _scheduler()
    # a *consecutive* block for the same owner with a full queue: the gap
    # rule (``_should_warm_start``) would not re-seed here
    s._open_block(h)
    s._close_block()
    assert h.seen == []

    s.on_meta_region(arm="Boxed", box=np.array([[-1.0, 1.0], [-1.0, 1.0]]))
    s._open_block(h)
    assert len(h.seen) == 1


def test_meta_publishes_a_region_for_a_supported_leaf():
    s = StrategyBlockBandit(
        Rosenbrock(dim=2),
        parse_args=False,
        testing_mode=True,
        seed=5,
        warm_start_on_resume=True,
        warm_start_only_if_foreign=False,
        warm_start_only_if_better=False,
    )
    s.config.max_eval = MAX_EVAL
    s.config.sync_evaluation = True
    s.config.stop_on_convergence = False
    s.add_analyzer(Splitter(s))
    s.add_heuristic(Boxed(s, name="CMAES"))
    m = MetaAnalyst(s, trigger=budget_fraction(0.25), mode="none", region=True, min_leafs=1, min_region_points=2)
    s.add_heuristic(m)
    s.eventbus.subscribe("meta_region", s)  # ``initialize`` does this for a real run
    a, _, _ = _three_leaves(s)

    m.on_new_results(_batch(100))
    assert len(m.firings) == 1
    assert m.firings[0]["points"] == 0, "mode='none' spends no evaluations"
    assert m.firings[0]["region"] == {"arm": "CMAES", "leaf": a.id}
    s.eventbus.wait_idle(timeout=5.0)
    assert s._pending_region.get("CMAES") is a


def test_region_is_refused_when_the_box_holds_too_little():
    s = StrategyBlockBandit(
        Rosenbrock(dim=2),
        parse_args=False,
        testing_mode=True,
        seed=5,
        warm_start_on_resume=True,
    )
    s.config.max_eval = MAX_EVAL
    s.config.sync_evaluation = True
    s.config.stop_on_convergence = False
    s.add_analyzer(Splitter(s))
    s.add_heuristic(Boxed(s, name="CMAES"))
    m = MetaAnalyst(s, trigger=budget_fraction(0.25), mode="none", region=True, min_leafs=1, min_region_points=1000)
    s.add_heuristic(m)
    s.eventbus.subscribe("meta_region", s)
    _three_leaves(s)

    m.on_new_results(_batch(100))
    assert m.refusals.get("region_unsupported", 0) == 1
    assert m.firings[0]["region"] is None


# ----------------------------------------------------------------------
# 5 -- whole runs still spend their budget
# ----------------------------------------------------------------------


def test_block_bandit_run_with_a_firing_meta_spends_its_budget():
    s = _portfolio(
        True,
        max_eval=300,
        meta_kwargs={"trigger": budget_fraction(0.25), "k": 10, "min_leafs": 1},
    )
    m = s._heuristics["Meta"]
    assert len(m.firings) == 1, m.refusals
    _, _, who = _trajectory(s)
    assert len(who) >= 300
    assert any(w.startswith("Meta:") for w in who), "the meta batch must reach the evaluator"


def test_round_robin_run_with_a_firing_meta_spends_its_budget():
    from panobbgo.heuristics import LSHADE

    s = StrategyRoundRobin(Rosenbrock(dim=2), parse_args=False, testing_mode=True, seed=11)
    s.config.max_eval = 200
    s.config.sync_evaluation = True
    s.config.stop_on_convergence = False
    s.config.evaluation_method = "threaded"
    s.add(LSHADE, NP_init=8)
    s.add(MetaAnalyst, trigger=budget_fraction(0.25), k=10, min_leafs=1)
    s.start()

    assert len(s.results) >= 200
    assert len(s._heuristics["Meta"].firings) == 1
