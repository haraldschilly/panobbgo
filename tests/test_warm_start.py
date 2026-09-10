# -*- coding: utf8 -*-
"""Warm-starting an arm from the shared archive.

``planning/DESIGN_warm_start_2026-09-10.md`` §2: the population heuristics
ignore every point they did not request themselves.  The
:class:`~panobbgo.analyzers.archive.Archive` analyzer is the query layer that
lets them look, :meth:`~panobbgo.core.Heuristic.archive_seed` is how they ask,
and the ``warm_start=`` constructor argument of the L-SHADE family and PSO is
how an experiment opts in.

The properties pinned here:

* the three selectors (``top_k`` / ``diverse_k`` / ``per_leaf_best``) on
  synthetic results;
* a warm-started L-SHADE fills its population with the archived
  :class:`~panobbgo.lib.Result` objects **themselves** — no evaluation is
  requested for a point that was already paid for;
* a warm-started PSO takes its personal bests from the seeds;
* and the reproducibility contract: ``warm_start=None`` — explicit or
  omitted — and an *empty* archive both reproduce the cold trajectory
  exactly.
"""

from __future__ import annotations

import numpy as np

from panobbgo.analyzers import Archive, Splitter
from panobbgo.lib import Point, Result
from panobbgo.lib.classic import Rosenbrock


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def _strategy(seed=1, max_eval=100):
    """An unstarted strategy: enough context for a module, no main loop."""
    from panobbgo.strategies import StrategyRoundRobin

    s = StrategyRoundRobin(Rosenbrock(dim=2), parse_args=False, testing_mode=True, seed=seed)
    s.config.max_eval = max_eval
    s.config.sync_evaluation = True
    s.config.stop_on_convergence = False
    return s


def _results(problem, n, *, seed=0, who=("A", "B", "C")):
    """``n`` synthetic results spread over the problem box, ``fx = ||x||^2``."""
    rng = np.random.default_rng(seed)
    lo, hi = problem.box[:, 0], problem.box[:, 1]
    out = []
    for i in range(n):
        x = lo + (hi - lo) * rng.random(problem.dim)
        out.append(Result(Point(x, who[i % len(who)]), float(np.sum(x**2))))
    return out


def _archive_of(s, results, k=64):
    a = Archive(s, k=k)
    s.add_analyzer(a)
    a.on_new_results(results)
    return a


# ----------------------------------------------------------------------
# (a) the Archive analyzer
# ----------------------------------------------------------------------


def test_archive_top_k_is_the_best_k_regardless_of_who():
    s = _strategy()
    res = _results(s.problem, 40)
    a = _archive_of(s, res, k=16)

    assert len(a) == 16  # bounded
    top = a.top_k(5)
    fxs = [r.fx for r in top]
    assert fxs == sorted(fxs)
    assert fxs == sorted(r.fx for r in res)[:5]
    # no ``who`` filter: the archive is shared by construction
    assert len({r.who for r in a.top_k(16)}) > 1


def test_archive_bound_keeps_the_best_not_the_last():
    s = _strategy()
    res = _results(s.problem, 60)
    a = _archive_of(s, res, k=8)
    assert sorted(r.fx for r in a.results) == sorted(r.fx for r in res)[:8]


def test_archive_top_k_filters():
    s = _strategy()
    res = _results(s.problem, 40)
    a = _archive_of(s, res, k=32)

    assert all(r.who != "A" for r in a.top_k(10, exclude_who="A"))
    assert all(r.who not in ("A", "B") for r in a.top_k(10, exclude_who=["A", "B"]))

    cutoff = float(np.median([r.fx for r in a.results]))
    assert all(r.fx <= cutoff for r in a.top_k(32, fx_max=cutoff))

    half = np.column_stack([s.problem.box[:, 0], np.zeros(s.problem.dim)])
    assert all((r.x <= 0.0).all() for r in a.top_k(32, box=half))

    assert a.top_k(0) == []


def test_archive_diverse_k_spreads_out():
    s = _strategy()
    res = _results(s.problem, 60)
    a = _archive_of(s, res, k=48)

    k = 5
    clustered = a.top_k(k)
    spread = a.diverse_k(k, pool=6)
    assert len(spread) == k
    assert spread[0] is clustered[0]  # the incumbent is always taken

    def min_gap(rs):
        xs = np.asarray([r.x for r in rs])
        d = np.linalg.norm(xs[:, None, :] - xs[None, :, :], axis=-1)
        return float(np.min(d + np.eye(len(rs)) * 1e9))

    assert min_gap(spread) > min_gap(clustered)


def test_archive_per_leaf_best_needs_a_splitter():
    s = _strategy(max_eval=40)
    res = _results(s.problem, 40)
    a = _archive_of(s, res)
    assert a.per_leaf_best(3) == []  # no Splitter registered -> no basins

    sp = Splitter(s)
    s.add_analyzer(sp)
    sp.on_new_results(res)
    assert len(sp.leafs) >= 2, "the fixture must produce more than one leaf"

    best = a.per_leaf_best(2)
    assert len(best) == 2
    leaf_bests = {id(box.best) for box in sp.leafs}
    assert all(id(r) in leaf_bests for r in best)
    assert best[0].fx <= best[1].fx
    assert best[0] is a.top_k(1)[0]  # the best leaf's best point is the incumbent
    assert len({id(r) for r in best}) == 2


def test_archive_ignores_unusable_results():
    s = _strategy()
    good = _results(s.problem, 4)
    bad = [Result(Point(np.zeros(2), "X"), float("nan")), Result(Point(np.ones(2), "X"), float("inf"))]
    a = _archive_of(s, good + bad, k=16)
    assert len(a) == len(good)


# ----------------------------------------------------------------------
# (b) L-SHADE seeds its population without spending evaluations
# ----------------------------------------------------------------------


def test_lshade_warm_start_fills_the_population_with_archived_results():
    from panobbgo.heuristics import LSHADE

    s = _strategy()
    res = _results(s.problem, 24)
    a = _archive_of(s, res)

    NP = 6
    h = LSHADE(s, NP_init=NP, warm_start="archive")
    seeds = a.top_k(NP)
    h.on_start()

    # the population *is* the seeds -- the objects, not copies of them
    assert [id(r) for r in h._population] == [id(r) for r in seeds]

    # ... and nothing was queued for re-evaluation: what comes out is one
    # follow-up trial per slot, not a fresh initial fill.
    points = h.get_points()
    assert len(points) == NP
    seed_x = np.asarray([r.x for r in seeds])
    for p in points:
        assert not np.any(np.all(np.isclose(seed_x, p.x), axis=1)), "a seed was re-evaluated"

    # the external archive is primed from the *next* good points, foreign first
    assert len(h._archive) == h._archive_cap()
    known = {tuple(np.round(r.x, 12)) for r in a.top_k(NP + h._archive_cap())[NP:]}
    assert all(tuple(np.round(x, 12)) in known for x in h._archive)


def test_lshade_warm_start_fills_a_shortfall_from_the_cold_path():
    from panobbgo.heuristics import LSHADE

    s = _strategy()
    res = _results(s.problem, 3)
    a = _archive_of(s, res)

    h = LSHADE(s, NP_init=8, warm_start="archive")
    h.on_start()
    seeds = a.top_k(3)
    assert [id(r) for r in h._population[:3]] == [id(r) for r in seeds]
    assert all(slot is None for slot in h._population[3:])
    # only the five uncovered slots cost an evaluation
    assert len(h.get_points()) == 5


def test_lshade_warm_start_now_reseeds_a_running_population():
    from panobbgo.heuristics import LSHADE

    s = _strategy()
    a = _archive_of(s, _results(s.problem, 24))

    h = LSHADE(s, NP_init=6, warm_start="archive")
    h.on_start()
    h.get_points()  # drain the first generation

    later = _results(s.problem, 12, seed=7)
    for r in later:
        r._fx = r.fx * 1e-6  # much better than anything in the archive so far
    a.on_new_results(later)

    assert h.warm_start_now() is True
    assert [id(r) for r in h._population] == [id(r) for r in a.top_k(6)]
    assert len(h.get_points()) == 6

    # not opted in -> the default hook, and nothing moves
    cold = LSHADE(s, NP_init=6, name="Cold")
    cold.on_start()
    before = list(cold._population)
    assert cold.warm_start_now() is False
    assert cold._population == before


def test_lshade_warm_start_validates_its_mode():
    import pytest

    from panobbgo.heuristics import JSO, LSHADE, NLSHADE_LBC, NLSHADE_RSP, LSHADE_EpSin

    s = _strategy()
    with pytest.raises(ValueError, match="warm_start must be None or one of"):
        LSHADE(s, warm_start="best")

    # The whole family inherits the behaviour: every subclass' ``on_start``
    # goes through ``LSHADE.on_start`` via ``super()``, so seeding one seeds
    # them all.
    a = _archive_of(s, _results(s.problem, 24))
    seeds = a.top_k(6)
    for i, cls in enumerate((LSHADE, JSO, NLSHADE_RSP, NLSHADE_LBC, LSHADE_EpSin)):
        assert cls(s, name="C%d" % i).warm_start is None
        h = cls(s, NP_init=6, warm_start="archive", name="W%d" % i)
        assert h.warm_start == "archive"
        h.on_start()
        assert [id(r) for r in h._population] == [id(r) for r in seeds], cls.__name__


# ----------------------------------------------------------------------
# (d) PSO
# ----------------------------------------------------------------------


def test_pso_warm_start_takes_its_swarm_from_the_seeds():
    from panobbgo.heuristics import PSO

    s = _strategy()
    a = _archive_of(s, _results(s.problem, 24))

    NP = 5
    h = PSO(s, NP=NP, warm_start="archive")
    seeds = a.top_k(NP)
    h.on_start()

    # ``_emit_trial`` advances ``_positions`` to the first move, so the
    # seeds are pinned where they survive: the personal bests.
    assert [id(r) for r in h._pbest_result] == [id(r) for r in seeds]
    np.testing.assert_allclose(h._pbest_x, np.asarray([r.x for r in seeds]))
    assert h._gbest_idx == 0  # top_k is sorted, so particle 0 holds the incumbent

    v_max = h._v_max()
    assert np.all(np.abs(h._velocities) <= v_max + 1e-12)
    assert np.all(np.linalg.norm(h._velocities, axis=1) > 0.0), "a derangement has no fixed point"

    assert len(h.get_points()) == NP  # one move each, zero evaluations for the seeds


def test_pso_warm_start_validates_its_mode():
    import pytest

    from panobbgo.heuristics import PSO

    s = _strategy()
    with pytest.raises(ValueError, match="warm_start must be None or one of"):
        PSO(s, warm_start="yes")
    assert PSO(s, name="P2").warm_start is None


# ----------------------------------------------------------------------
# (c) + (e) the reproducibility contract
# ----------------------------------------------------------------------


def _run(seed=1234, max_eval=60, warm_start="__omitted__", with_archive=False):
    """A small seeded round-robin run; returns its ``(x, fx, who)`` trajectory."""
    from panobbgo.heuristics import JSO, PSO, Random
    from panobbgo.strategies import StrategyRoundRobin

    problem = Rosenbrock(dim=2)
    s = StrategyRoundRobin(problem, parse_args=False, seed=seed)
    s.config.max_eval = max_eval
    s.config.sync_evaluation = True
    s.config.stop_on_convergence = False

    kw = {} if warm_start == "__omitted__" else {"warm_start": warm_start}
    s.add_heuristic(Random(s))
    s.add_heuristic(JSO(s, NP_init=8, **kw))
    s.add_heuristic(PSO(s, NP=6, **kw))
    if with_archive:
        s.add_analyzer(Archive(s))
    s.start()

    df = s.results.results
    assert df is not None and len(df) >= max_eval
    return (
        df["x"].to_numpy(dtype=float),
        df["fx"].to_numpy(dtype=float).ravel(),
        df["who"].to_numpy().ravel().astype(str),
    )


def _assert_same(a, b):
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    assert list(a[2]) == list(b[2])


def test_explicit_none_is_the_same_run_as_omitting_the_argument():
    """The new keyword must not perturb the RNG streams (design §2)."""
    _assert_same(_run(warm_start="__omitted__"), _run(warm_start=None))


def test_an_empty_archive_degrades_to_the_cold_path():
    """At ``t = 0`` nothing has been evaluated, so a warm arm starts cold."""
    _assert_same(
        _run(warm_start=None, with_archive=True),
        _run(warm_start="archive", with_archive=True),
    )
