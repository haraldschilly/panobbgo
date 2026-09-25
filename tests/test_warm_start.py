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
* a warm-started CMA-ES fits ``m`` / ``σ`` / ``C`` to the seed cloud — saving
  no evaluations, but starting in the right basin at the right scale;
* and the reproducibility contract: ``warm_start=None`` — explicit or
  omitted — and an *empty* archive both reproduce the cold trajectory
  exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from panobbgo.analyzers import Archive, Splitter
from panobbgo.lib import Point, Result
from panobbgo.lib.classic import Rosenbrock


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def _strategy(seed=1, max_eval=100, dim=2):
    """An unstarted strategy: enough context for a module, no main loop."""
    from panobbgo.strategies import StrategyRoundRobin

    s = StrategyRoundRobin(Rosenbrock(dim=dim), parse_args=False, testing_mode=True, seed=seed)
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


def test_pso_warm_start_honours_the_region_box():
    """Regression: a MetaAnalyst region hand-off restricts PSO's seeds to the box.

    ``_warm_start_swarm`` used to call ``archive_seed`` without
    ``box=self.warm_start_box``, so the hand-off was silently unrestricted.
    """
    from panobbgo.heuristics import PSO

    s = _strategy()
    a = _archive_of(s, _results(s.problem, 60))
    lo, hi = s.problem.box[:, 0], s.problem.box[:, 1]
    box = np.column_stack([lo + 0.6 * (hi - lo), hi])  # away from the best (origin) points
    inside = a.top_k(64, box=box)
    assert 0 < len(inside) < 60 and all(r not in inside for r in a.top_k(3))

    h = PSO(s, NP=4, warm_start="archive")
    h.warm_start_box = box
    h.on_start()
    seeded = [r for r in h._pbest_result if r is not None]
    assert seeded, "the box holds archive points, so the swarm must be seeded"
    for r in seeded:
        assert np.all(r.x >= box[:, 0]) and np.all(r.x <= box[:, 1])


def test_pso_warm_start_validates_its_mode():
    from panobbgo.heuristics import PSO

    s = _strategy()
    with pytest.raises(ValueError, match="warm_start must be None or one of"):
        PSO(s, warm_start="yes")
    assert PSO(s, name="P2").warm_start is None


# ----------------------------------------------------------------------
# CMA-ES: m / sigma / C fitted to the seed cloud
# ----------------------------------------------------------------------


def _cmaes_seeds(h, archive):
    """The seed positions ``h`` would have used, as an ``(k, dim)`` array."""
    return np.vstack([r.x for r in archive.top_k(len(h._warm_start_seeds()))])


def test_cmaes_warm_start_mean_is_the_weighted_recombination():
    from panobbgo.heuristics import CMAES

    s = _strategy()
    a = _archive_of(s, _results(s.problem, 40))

    h = CMAES(s, warm_start="archive")
    h.on_start()

    X = _cmaes_seeds(h, a)
    assert len(X) == max(h._lam, 4 + int(3 * np.log(2)))
    np.testing.assert_allclose(h._m, h._w @ X[: h._mu])
    # exactly one generation went out, from the fitted distribution
    assert len(h.get_points()) == h._lam


def test_cmaes_warm_start_sigma_is_the_seed_spread_and_never_wider_than_cold():
    from panobbgo.heuristics import CMAES

    s = _strategy()
    a = _archive_of(s, _results(s.problem, 40))

    h = CMAES(s, warm_start="archive")
    h.on_start()
    X = _cmaes_seeds(h, a)

    sigma0_cold = h._sigma0_default()
    np.testing.assert_allclose(h._sigma, float(np.mean(np.std(X, axis=0))))
    assert h._sigma <= sigma0_cold
    # ``_sigma0`` — what ``tolx`` is relative to — follows the warm sigma
    assert h._sigma0 == h._sigma

    # a cloud wider than the cold sigma0 is clipped, never widened
    wide = _strategy(seed=2)
    _archive_of(wide, _results(wide.problem, 40, seed=11))
    g = CMAES(wide, warm_start="archive", sigma0=0.01)
    g.on_start()
    assert g._sigma == g._sigma0_default()

    # a collapsed cloud is floored, not zeroed
    tight = _strategy(seed=3)
    x0 = np.zeros(tight.problem.dim)
    same = [Result(Point(x0 + 1e-18 * i, "T"), float(i)) for i in range(8)]
    _archive_of(tight, same)
    t = CMAES(tight, warm_start="archive")
    t.on_start()
    assert t._sigma == 1e-6 * float(np.mean(t._ranges))


def test_cmaes_warm_start_leaves_c_and_the_paths_at_their_cold_values():
    from panobbgo.heuristics import CMAES

    s = _strategy()
    _archive_of(s, _results(s.problem, 40))
    n = s.problem.dim

    h = CMAES(s, warm_start="archive")
    h.on_start()
    np.testing.assert_array_equal(h._C, np.eye(n))
    np.testing.assert_array_equal(h._B, np.eye(n))
    np.testing.assert_array_equal(h._D, np.ones(n))
    np.testing.assert_array_equal(h._p_c, np.zeros(n))
    np.testing.assert_array_equal(h._p_sigma, np.zeros(n))
    assert h._counteval == 0


def test_cmaes_archive_cov_seeds_a_unit_determinant_covariance():
    from panobbgo.heuristics import CMAES

    s = _strategy()
    # a deliberately anisotropic cloud: wide in x0, narrow in x1
    rng = np.random.default_rng(5)
    cloud = [
        Result(Point(s.problem.project(np.array([rng.normal(0.0, 0.8), rng.normal(0.0, 0.02)])), "T"), float(i))
        for i in range(20)
    ]
    a = _archive_of(s, cloud)

    h = CMAES(s, warm_start="archive_cov")
    h.on_start()
    X = _cmaes_seeds(h, a)

    # unit determinant: the *scale* of the cloud lives in sigma alone
    assert np.linalg.det(h._C) == pytest.approx(1.0)
    # ... and the shape is the cloud's, decomposed the way ``_update`` does
    cov = np.cov(X, rowvar=False)
    eig = np.linalg.eigvalsh((cov + cov.T) / 2.0)
    eig = np.maximum(eig, 1e-20)
    eig = eig / float(np.exp(np.mean(np.log(eig))))
    np.testing.assert_allclose(np.sort(h._D), np.sqrt(np.sort(eig)))
    np.testing.assert_allclose(h._C, (h._B * (h._D**2)) @ h._B.T, atol=1e-12)
    assert h._D.max() / h._D.min() > 2.0, "the anisotropy must survive"

    # "archive" alone must NOT pick the shape up — that is the whole point of
    # keeping the covariance seed a separate mode.
    plain = CMAES(s, warm_start="archive")
    plain.on_start()
    np.testing.assert_array_equal(plain._C, np.eye(s.problem.dim))


def test_cmaes_archive_cov_degrades_to_identity_on_a_degenerate_cloud():
    from panobbgo.heuristics import CMAES

    s = _strategy()
    # every seed on one line -> the sample covariance is rank 1
    base = np.array([0.1, 0.1])
    line = [Result(Point(s.problem.project(base * (1 + i)), "T"), float(i)) for i in range(12)]
    _archive_of(s, line)

    h = CMAES(s, warm_start="archive_cov")
    h.on_start()
    np.testing.assert_array_equal(h._C, np.eye(s.problem.dim))
    np.testing.assert_array_equal(h._D, np.ones(s.problem.dim))


# ----------------------------------------------------------------------
# (e) §48.3: shrinking the seeded covariance by its own sample size
# ----------------------------------------------------------------------
#
# ``archive_cov`` won *d* = 2 and lost *d* = 5 (§48.3, −0.069 at 200·dim).
# The estimator is a 5x5 sample covariance of the archive top-10, with
# nothing between it and the search distribution.  ``warm_start_cov_shrink``
# blends it toward **I** by the sample size, ``warm_start_cov_cond_max``
# clips the eigenvalue ratio, and ``warm_start_wide_seeds`` is the
# location-only control that gives plain ``archive`` the *same* seed set so
# the wider sample can be measured apart from the shape it buys.


def _anisotropic_cloud(s, *, n=20, ratio=40.0, seed=5):
    """Results on an axis-aligned cloud whose covariance is badly conditioned."""
    rng = np.random.default_rng(seed)
    dim = s.problem.dim
    scales = np.geomspace(0.8, 0.8 / ratio, dim)
    out = []
    for i in range(n):
        x = s.problem.project(rng.normal(0.0, 1.0, dim) * scales)
        out.append(Result(Point(x, "T"), float(i)))
    return out


def _cond_of(h):
    """``cond(C)`` read off the decomposition.

    Not ``h._cond``: ``_reset_run_state`` runs *after* the warm start and
    zeroes that field, so the seeded condition number is only visible in
    ``_D`` (which is what the sampling actually uses).
    """
    return float(h._D.max() / h._D.min()) ** 2


def test_shrinkage_alpha_counts_free_parameters_not_dimensions():
    from panobbgo.heuristics import CMAES

    s = _strategy()
    h = CMAES(s, warm_start="archive_cov", warm_start_cov_shrink=1.0)

    # d=2, k=6 (λ): (6-2-1)/(1·3) = 1 exactly — the winning d=2 behaviour
    # of §48.3 is preserved, not merely approximated.
    assert h._shrinkage_alpha(6, 2) == 1.0
    # d=5, k=10 (the 2n floor): (10-5-1)/(1·15) = 4/15
    assert h._shrinkage_alpha(10, 5) == pytest.approx(4.0 / 15.0)
    # the constant scales the whole rule
    assert h._shrinkage_alpha(10, 5) == pytest.approx(
        3.0 * CMAES(s, warm_start="archive_cov", warm_start_cov_shrink=3.0)._shrinkage_alpha(10, 5)
    )
    # clipped into [0, 1] at both ends
    assert h._shrinkage_alpha(3, 2) == 0.0  # k = n+1: nothing to estimate with
    assert h._shrinkage_alpha(1000, 2) == 1.0
    # off by default: no shrinkage at all, whatever the sample size
    assert CMAES(s, warm_start="archive_cov")._shrinkage_alpha(10, 5) == 1.0


def test_cmaes_cov_shrinkage_is_inert_at_alpha_one():
    """α = 1 reproduces the unshrunk matrix — bit for bit, at d = 2."""
    from panobbgo.heuristics import CMAES

    s = _strategy()
    _archive_of(s, _anisotropic_cloud(s))

    plain = CMAES(s, warm_start="archive_cov")
    plain.on_start()
    shrunk = CMAES(s, warm_start="archive_cov", warm_start_cov_shrink=1.0)
    shrunk.on_start()

    assert shrunk._shrinkage_alpha(len(shrunk._warm_start_seeds()), s.problem.dim) == 1.0
    np.testing.assert_array_equal(shrunk._C, plain._C)
    np.testing.assert_array_equal(shrunk._D, plain._D)


def test_cmaes_cov_shrinkage_at_alpha_zero_is_the_identity():
    from panobbgo.heuristics import CMAES

    s = _strategy(dim=5)
    _archive_of(s, _anisotropic_cloud(s, n=30))

    h = CMAES(s, warm_start="archive_cov", warm_start_cov_shrink=1.0)
    h._shrinkage_alpha = lambda k, n: 0.0  # type: ignore[method-assign]
    h.on_start()

    np.testing.assert_allclose(h._C, np.eye(s.problem.dim), atol=1e-12)
    np.testing.assert_allclose(h._D, np.ones(s.problem.dim), atol=1e-12)
    assert _cond_of(h) == pytest.approx(1.0)


def test_cmaes_cov_shrinkage_stays_spd_with_unit_determinant():
    from panobbgo.heuristics import CMAES

    s = _strategy(dim=5)
    _archive_of(s, _anisotropic_cloud(s, n=30))

    plain = CMAES(s, warm_start="archive_cov")
    plain.on_start()
    h = CMAES(s, warm_start="archive_cov", warm_start_cov_shrink=1.0)
    h.on_start()

    n = s.problem.dim
    # the 2n floor binds at d=5: 10 seeds for a 5x5, alpha = 4/15
    assert len(h._warm_start_seeds()) == 2 * n
    eig = np.linalg.eigvalsh(h._C)
    assert np.all(eig > 0.0)  # positive definite
    np.testing.assert_allclose(h._C, h._C.T, atol=1e-12)  # symmetric
    assert np.linalg.det(h._C) == pytest.approx(1.0)
    np.testing.assert_allclose(h._C, (h._B * (h._D**2)) @ h._B.T, atol=1e-12)
    # and it really is closer to I than the unshrunk estimate
    assert np.linalg.norm(h._C - np.eye(n)) < np.linalg.norm(plain._C - np.eye(n))
    assert _cond_of(h) < _cond_of(plain)


def test_cmaes_cov_condition_cap_binds_without_discarding_the_estimate():
    from panobbgo.heuristics import CMAES

    s = _strategy(dim=5)
    # ratio 1e4 per axis pair: ill-conditioned, but nowhere near the 1e7
    # hard fallback, so the cap is the only thing that can act.
    _archive_of(s, _anisotropic_cloud(s, n=30, ratio=1e4))

    plain = CMAES(s, warm_start="archive_cov")
    plain.on_start()
    assert _cond_of(plain) > 1e3, "the uncapped estimate must be worse than the cap"

    capped = CMAES(s, warm_start="archive_cov", warm_start_cov_cond_max=1e3)
    capped.on_start()
    assert _cond_of(capped) <= 1e3 * (1.0 + 1e-9)
    assert np.linalg.det(capped._C) == pytest.approx(1.0)
    # clipped, not thrown away: the anisotropy survives, and the principal
    # directions are the cloud's.
    assert _cond_of(capped) > 1.0
    np.testing.assert_allclose(np.abs(capped._B), np.abs(plain._B), atol=1e-8)


def test_cmaes_cov_hard_fallback_survives_the_new_knobs():
    """A rank-deficient cloud still degrades to I, cap and shrinkage or not."""
    from panobbgo.heuristics import CMAES

    s = _strategy()
    base = np.array([0.1, 0.1])
    line = [Result(Point(s.problem.project(base * (1 + i)), "T"), float(i)) for i in range(12)]
    _archive_of(s, line)

    h = CMAES(s, warm_start="archive_cov", warm_start_cov_shrink=1.0, warm_start_cov_cond_max=1e3)
    h.on_start()
    np.testing.assert_array_equal(h._C, np.eye(s.problem.dim))
    np.testing.assert_array_equal(h._D, np.ones(s.problem.dim))


def test_cmaes_wide_seeds_is_the_location_only_control():
    """``archive`` with the ``2n`` floor: the cov sample, none of the shape."""
    from panobbgo.heuristics import CMAES

    s = _strategy(dim=5)
    _archive_of(s, _anisotropic_cloud(s, n=30))
    n = s.problem.dim

    narrow = CMAES(s, warm_start="archive")
    narrow.on_start()
    wide = CMAES(s, warm_start="archive", warm_start_wide_seeds=True)
    wide.on_start()
    cov = CMAES(s, warm_start="archive_cov")
    cov.on_start()

    # the confound §48.3 names: at d=5 ``archive_cov`` fits 10 points where
    # ``archive`` fits 8.  The two extra points move sigma -- and *only*
    # sigma: m is the mu-weighted recombination of the best mu = lambda//2 = 4
    # seeds, which are the same four either way.  So the wider sample is a
    # pure *spread* effect, which is the hand-off's one measured failure mode
    # (DISCOVERY §48.2: a wider seed cloud is what killed ``archive_diverse``).
    assert len(narrow._warm_start_seeds()) == narrow._lam == 8
    assert len(wide._warm_start_seeds()) == 2 * n == 10
    assert len(cov._warm_start_seeds()) == 2 * n
    np.testing.assert_array_equal(wide._m, narrow._m)
    assert wide._sigma != narrow._sigma
    # ... but the control keeps C = I, so its delta is location only
    np.testing.assert_array_equal(wide._C, np.eye(n))
    np.testing.assert_array_equal(wide._D, np.ones(n))
    np.testing.assert_allclose(wide._m, cov._m)
    np.testing.assert_allclose(wide._sigma, cov._sigma)

    # at d=2 the floor does not bind, so the control is a no-op there
    two = _strategy(dim=2, seed=4)
    _archive_of(two, _anisotropic_cloud(two, n=30))
    a = CMAES(two, warm_start="archive")
    a.on_start()
    b = CMAES(two, warm_start="archive", warm_start_wide_seeds=True)
    b.on_start()
    assert len(a._warm_start_seeds()) == len(b._warm_start_seeds()) == 6
    np.testing.assert_array_equal(a._m, b._m)
    assert a._sigma == b._sigma


def test_cmaes_cov_default_path_is_unchanged():
    """The knobs are opt-in: omitted or ``None`` is the pre-2026-09-14 run."""
    from panobbgo.heuristics import CMAES

    for dim in (2, 5):
        s = _strategy(dim=dim)
        _archive_of(s, _anisotropic_cloud(s, n=30))
        omitted = CMAES(s, warm_start="archive_cov")
        omitted.on_start()
        explicit = CMAES(
            s,
            warm_start="archive_cov",
            warm_start_cov_shrink=None,
            warm_start_cov_cond_max=None,
            warm_start_wide_seeds=False,
        )
        explicit.on_start()
        np.testing.assert_array_equal(omitted._C, explicit._C)
        np.testing.assert_array_equal(omitted._m, explicit._m)
        np.testing.assert_array_equal(omitted._D, explicit._D)
        assert omitted._sigma == explicit._sigma


@pytest.mark.parametrize(
    "kw",
    [
        {"warm_start_cov_shrink": 0.0},
        {"warm_start_cov_shrink": -1.0},
        {"warm_start_cov_cond_max": 1.0},
        {"warm_start_cov_cond_max": 0.5},
    ],
)
def test_cmaes_cov_knobs_validate_their_arguments(kw):
    from panobbgo.heuristics import CMAES

    s = _strategy()
    with pytest.raises(ValueError):
        CMAES(s, warm_start="archive_cov", **kw)


def test_cmaes_warm_start_now_only_fires_with_seeds_and_reloads_the_queue():
    from panobbgo.heuristics import CMAES

    s = _strategy()
    a = Archive(s, k=64)
    s.add_analyzer(a)

    h = CMAES(s, warm_start="archive")
    h.on_start()  # empty archive -> cold start
    cold_m = h._m.copy()
    stale = h.get_points()
    assert len(stale) == h._lam
    assert h._pending  # the cold generation is in flight

    # empty archive: nothing to seed with, and nothing is thrown away
    assert h.warm_start_now() is False
    assert h._pending
    np.testing.assert_array_equal(h._m, cold_m)

    a.on_new_results(_results(s.problem, 40))
    assert h.warm_start_now() is True
    assert not np.array_equal(h._m, cold_m)
    # the stale generation is gone and a fresh one is queued
    assert all(who.startswith("CMAES:g2:") for who in h._pending)
    assert len(h.get_points()) == h._lam
    assert h._counteval == 0  # a re-seed is not a restart
    assert h.restart_count == 0

    # not opted in -> the base-class hook
    assert CMAES(s, warm_start=None).warm_start_now() is False


def test_cmaes_warm_start_validates_its_mode():
    from panobbgo.heuristics import CMAES

    s = _strategy()
    with pytest.raises(ValueError, match="warm_start must be None or one of"):
        CMAES(s, warm_start="archive_covariance")
    assert CMAES(s).warm_start is None
    assert "archive_cov" in CMAES.SUPPORTED_WARM_START


# ----------------------------------------------------------------------
# (c) + (e) the reproducibility contract
# ----------------------------------------------------------------------


def _run(seed=1234, max_eval=60, warm_start="__omitted__", with_archive=False):
    """A small seeded round-robin run; returns its ``(x, fx, who)`` trajectory."""
    from panobbgo.heuristics import CMAES, JSO, PSO, Random
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
    s.add_heuristic(CMAES(s, **kw))
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
    """The new keyword must not perturb the RNG streams (design §2).

    Covers all three warm-startable families at once — the run carries a
    JSO, a PSO and a CMAES.
    """
    _assert_same(_run(warm_start="__omitted__"), _run(warm_start=None))


def test_an_empty_archive_degrades_to_the_cold_path():
    """At ``t = 0`` nothing has been evaluated, so a warm arm starts cold."""
    _assert_same(
        _run(warm_start=None, with_archive=True),
        _run(warm_start="archive", with_archive=True),
    )


# ``_run`` above carries a JSO, a PSO and a CMAES, so the two contracts it
# pins were only ever checked on those three.  The rest of the L-SHADE
# family overrides pieces of the warm-start path, and one of those overrides
# was randomised: until 2026-09 ``NLSHADE_RSP._archive_cap`` sampled a
# per-generation archive cap from ``self._rng`` (the removed
# ``adaptive_archive`` option, inherited by ``NLSHADE_LBC``; both now use a
# fixed paper cap), and
# ``LSHADE._warm_start_population`` used to call it one line *before* the
# "empty archive -> cold path" bail-out.  So on those two arms, passing
# ``warm_start="archive"`` against an empty archive consumed a draw and
# shifted the entire initial population — the warm start never happened,
# but the run moved anyway, which makes every paired warm-vs-cold A/B on
# them a comparison of two different RNG streams.  One arm per run here, so
# a regression names the arm that broke.
_DE_FAMILY = ("LSHADE", "JSO", "NLSHADE_RSP", "NLSHADE_LBC", "LSHADE_EpSin")


def _run_arm(name, seed=1234, max_eval=60, warm_start="__omitted__", with_archive=False):
    """``_run``, but with a single named arm of the L-SHADE family."""
    import panobbgo.heuristics as H
    from panobbgo.strategies import StrategyRoundRobin

    s = StrategyRoundRobin(Rosenbrock(dim=2), parse_args=False, seed=seed)
    s.config.max_eval = max_eval
    s.config.sync_evaluation = True
    s.config.stop_on_convergence = False

    kw = {} if warm_start == "__omitted__" else {"warm_start": warm_start}
    s.add_heuristic(getattr(H, name)(s, NP_init=8, **kw))
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


@pytest.mark.parametrize("name", _DE_FAMILY)
def test_empty_archive_is_the_cold_path_for_the_whole_de_family(name):
    """Every L-SHADE variant, not just the two ``_run`` happens to carry."""
    _assert_same(
        _run_arm(name, warm_start=None, with_archive=True),
        _run_arm(name, warm_start="archive", with_archive=True),
    )


@pytest.mark.parametrize("name", _DE_FAMILY)
@pytest.mark.parametrize("mode", ("archive", "archive_diverse", "archive_leaf"))
def test_every_selector_is_free_against_an_empty_archive(name, mode):
    """The bail-out must cost nothing whichever selector asked for it."""
    _assert_same(
        _run_arm(name, warm_start=None, with_archive=True),
        _run_arm(name, warm_start=mode, with_archive=True),
    )


@pytest.mark.parametrize("name", _DE_FAMILY)
def test_explicit_none_is_the_omitted_run_for_the_whole_de_family(name):
    """And the keyword itself stays inert, arm by arm."""
    _assert_same(_run_arm(name, warm_start="__omitted__"), _run_arm(name, warm_start=None))


# ----------------------------------------------------------------------
# (f) §50.4 / §49.4: the two untested pieces of the hand-off's payload
# ----------------------------------------------------------------------
#
# ``warm_start_sigma_floor`` bounds how far one hand-off may collapse the
# step size (§50.3's stall: the archive top-k fall into one basin, their
# spread is tiny, and the receiving arm restarts pinned to a point).
# ``warm_start_keep_cov`` keeps the arm's own adapted B/D/C instead of
# resetting it to **I** (§49.4: the relocation without the reset).  Both are
# opt-in and default-off, so every number measured before 2026-09-14 stands.


def _cluster(s, *, center=(1.0, 0.0), jitter=1e-4, n=20, seed=7):
    """Results collapsed into one basin — the cloud that pins the hand-off."""
    rng = np.random.default_rng(seed)
    c = np.asarray(center, dtype=float)[: s.problem.dim]
    out = []
    for i in range(n):
        x = s.problem.project(c + rng.uniform(-jitter, jitter, s.problem.dim))
        out.append(Result(Point(x, "T"), float(i)))
    return out


def _warm_arm(s, **kw):
    """A started CMA-ES on an empty archive: cold state, warm start armed."""
    from panobbgo.heuristics import CMAES

    h = CMAES(s, warm_start="archive", **kw)
    h.on_start()
    return h


def test_cmaes_sigma_floor_binds_when_the_seed_cloud_has_collapsed():
    s = _strategy()
    a = _archive_of(s, [])
    plain = _warm_arm(s)
    floored = _warm_arm(s, warm_start_sigma_floor=0.5)
    sigma_self = plain._sigma
    assert floored._sigma == sigma_self  # both cold-started identically

    a.on_new_results(_cluster(s))
    assert plain.warm_start_now() is True
    assert floored.warm_start_now() is True

    # without the floor the arm inherits the cloud's (tiny) spread ...
    assert plain._sigma < 1e-3
    # ... with it, it may lose at most half of what it had
    assert floored._sigma == pytest.approx(0.5 * sigma_self)
    # and the floor moves nothing else: same seeds, same incumbent mean
    np.testing.assert_array_equal(floored._m, plain._m)


def test_cmaes_sigma_floor_is_inert_when_the_cloud_is_not_collapsed():
    """A spread above the floor is taken unchanged — bit for bit."""
    s = _strategy()
    a = _archive_of(s, [])
    plain = _warm_arm(s)
    floored = _warm_arm(s, warm_start_sigma_floor=0.5)
    sigma_self = plain._sigma

    a.on_new_results(_cluster(s, jitter=0.5, n=30))
    assert plain.warm_start_now() is True
    assert floored.warm_start_now() is True

    assert 0.5 * sigma_self < plain._sigma < plain._sigma0_default()
    assert floored._sigma == plain._sigma
    np.testing.assert_array_equal(floored._C, plain._C)


def test_cmaes_sigma_floor_never_beats_the_cold_sigma0():
    """The upper clip wins: a warm start may narrow, never widen."""
    s = _strategy()
    _archive_of(s, _cluster(s))

    h = _warm_arm(s, warm_start_sigma_floor=1.0)
    # an arm whose own sigma has grown past the cold sigma0 (the step-size
    # path may take it up to mean(range); sigma0 is 0.3 * range / 2)
    h._sigma = float(np.mean(h._ranges))
    assert h._sigma > h._sigma0_default()

    assert h.warm_start_now() is True
    assert h._sigma == pytest.approx(h._sigma0_default())


def test_cmaes_sigma_floor_bounds_the_rate_not_the_depth():
    """Repeated hand-offs from a collapsed archive still intensify — slowly."""
    s = _strategy()
    a = _archive_of(s, [])
    h = _warm_arm(s, warm_start_sigma_floor=0.5)
    plain = _warm_arm(s)
    sigma0 = h._sigma
    a.on_new_results(_cluster(s))

    sigmas = []
    for _ in range(4):
        assert h.warm_start_now() is True
        assert plain.warm_start_now() is True
        sigmas.append(h._sigma)
    # geometric decay at exactly the floor's rate, four hand-offs deep ...
    np.testing.assert_allclose(sigmas, sigma0 * 0.5 ** np.arange(1, 5), rtol=1e-12)
    # ... where the unfloored arm was pinned to the cloud by the first one
    assert plain._sigma < 1e-3
    assert h._sigma > plain._sigma


@pytest.mark.parametrize("f", [0.0, -0.5, 1.5])
def test_cmaes_sigma_floor_validates_its_argument(f):
    from panobbgo.heuristics import CMAES

    s = _strategy()
    with pytest.raises(ValueError, match="warm_start_sigma_floor"):
        CMAES(s, warm_start="archive", warm_start_sigma_floor=f)


def _adapted(h, *, cond=9.0, seed=3):
    """Give *h* a non-identity C/B/D and non-zero evolution paths."""
    n = h.problem.dim
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    eig = np.geomspace(1.0 / np.sqrt(cond), np.sqrt(cond), n)
    h._C = (q * eig) @ q.T
    h._B = q
    h._D = np.sqrt(eig)
    h._p_c = rng.normal(size=n)
    h._p_sigma = rng.normal(size=n)
    return h


def test_cmaes_keep_cov_moves_m_and_sigma_and_nothing_else():
    s = _strategy()
    a = _archive_of(s, [])

    h = _adapted(_warm_arm(s, warm_start_keep_cov=True))
    C, B, D = h._C.copy(), h._B.copy(), h._D.copy()
    m, sigma = h._m.copy(), h._sigma
    a.on_new_results(_cluster(s, jitter=0.3, n=30))

    assert h.warm_start_now() is True

    # the payload: B, D and C survive bit for bit ...
    np.testing.assert_array_equal(h._C, C)
    np.testing.assert_array_equal(h._B, B)
    np.testing.assert_array_equal(h._D, D)
    # ... the evolution paths are zeroed (the mean has just jumped) ...
    np.testing.assert_array_equal(h._p_c, np.zeros(s.problem.dim))
    np.testing.assert_array_equal(h._p_sigma, np.zeros(s.problem.dim))
    # ... and the relocation itself happened
    assert not np.array_equal(h._m, m)
    assert h._sigma != sigma


def test_cmaes_keep_cov_is_the_only_branch_that_keeps_a_shape():
    """The default and ``archive_cov`` both overwrite what the arm adapted."""
    s = _strategy()
    a = _archive_of(s, [])
    n = s.problem.dim

    reset = _adapted(_warm_arm(s))
    kept = _adapted(_warm_arm(s, warm_start_keep_cov=True))
    a.on_new_results(_cluster(s, jitter=0.3, n=30))
    assert reset.warm_start_now() is True
    assert kept.warm_start_now() is True

    np.testing.assert_array_equal(reset._C, np.eye(n))
    np.testing.assert_array_equal(reset._D, np.ones(n))
    assert not np.allclose(kept._C, np.eye(n))
    # m and sigma are the same in both: only the shape payload differs
    np.testing.assert_array_equal(kept._m, reset._m)
    assert kept._sigma == reset._sigma


def test_cmaes_keep_cov_leaves_the_lazy_eigen_schedule_alone():
    """Nothing rewrote the decomposition, so nothing restarts its clock."""
    s = _strategy()
    a = _archive_of(s, [])

    kept = _adapted(_warm_arm(s, warm_start_keep_cov=True))
    reset = _adapted(_warm_arm(s))
    a.on_new_results(_cluster(s, jitter=0.3, n=30))
    for h in (kept, reset):
        h._counteval = 40
        h._eigeneval = 7

    assert kept.warm_start_now() is True
    assert reset.warm_start_now() is True

    assert kept._eigeneval == 7  # the arm's own schedule, uninterrupted
    assert reset._eigeneval == 40  # B/D were just made consistent with C


def test_cmaes_keep_cov_rejects_the_archive_cov_payload():
    from panobbgo.heuristics import CMAES

    s = _strategy()
    with pytest.raises(ValueError, match="mutually exclusive"):
        CMAES(s, warm_start="archive_cov", warm_start_keep_cov=True)
    # ... and is accepted on every payload that does not seed C itself
    for mode in ("archive", "archive_diverse", "archive_leaf"):
        assert CMAES(s, warm_start=mode, warm_start_keep_cov=True)._warm_start_keep_cov is True


def test_cmaes_handoff_payload_default_path_is_unchanged():
    """Both knobs are opt-in: omitted or off is the pre-2026-09-14 hand-off."""
    from panobbgo.heuristics import CMAES

    for cloud in (_cluster, lambda s: _cluster(s, jitter=0.3, n=30)):
        s = _strategy()
        a = _archive_of(s, [])
        omitted = CMAES(s, warm_start="archive")
        omitted.on_start()
        explicit = CMAES(s, warm_start="archive", warm_start_sigma_floor=None, warm_start_keep_cov=False)
        explicit.on_start()
        for h in (omitted, explicit):
            _adapted(h)
        a.on_new_results(cloud(s))

        assert omitted.warm_start_now() is True
        assert explicit.warm_start_now() is True
        np.testing.assert_array_equal(omitted._m, explicit._m)
        np.testing.assert_array_equal(omitted._C, explicit._C)
        np.testing.assert_array_equal(omitted._D, explicit._D)
        assert omitted._sigma == explicit._sigma
        # the default really is the reset, floor or no floor
        np.testing.assert_array_equal(omitted._C, np.eye(s.problem.dim))
        assert omitted._sigma == pytest.approx(
            float(np.clip(omitted._sigma, 1e-6 * float(np.mean(omitted._ranges)), omitted._sigma0_default()))
        )
