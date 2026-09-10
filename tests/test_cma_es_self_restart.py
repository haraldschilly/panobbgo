# -*- coding: utf8 -*-
"""CMA-ES restarts itself when its own termination criteria fire.

Before 2026-09 the restart path of :class:`~panobbgo.heuristics.cma_es.CMAES`
was reachable only from the ``restart`` event of the
:class:`~panobbgo.analyzers.restart.Restart` analyzer, which is in no
benchmarked configuration.  A solo CMA-ES therefore had *no* termination
criterion: measured on five MA-BBOB instances at d=5 with a 2500-evaluation
budget, 52 % of the budget on average went to evaluations after the last
improvement of any size (91 % after 99 % of the total improvement).

These tests pin the four properties that fix has to keep:

a) the criteria actually fire and restart the search,
b) ``self_restart=False`` reproduces the pre-change trajectory bit for bit,
c) the ``restart`` event path is untouched,
d) ``restart_from="random"`` draws from the heuristic's own generator, so a
   seeded run stays reproducible.
"""

from __future__ import annotations

import numpy as np

from panobbgo.heuristics import CMAES
from panobbgo.lib.classic import DeJong
from panobbgo.strategies import StrategyRoundRobin

# A sphere whose box centre (3, 3, 3) is *not* the optimum (0, 0, 0), so the
# run has something to do before it converges.
BOX = [(-2.0, 8.0)] * 3


def _run(seed=42, max_eval=400, analyzers=(), **kw):
    """One solo-CMAES run; returns the heuristic and the evaluated fx vector."""
    problem = DeJong(3, box=list(BOX))
    np.random.seed(seed)
    s = StrategyRoundRobin(problem, max_evaluations=max_eval, seed=seed, parse_args=False, testing_mode=True)
    s.config.sync_evaluation = True
    s.config.stop_on_convergence = False
    h = CMAES(s, **kw)
    s.add_heuristic(h)
    for factory in analyzers:
        s.add_analyzer(factory(s))
    s.start()
    fx = s.results.results["fx"].to_numpy(dtype=float).ravel()
    return h, fx


def test_self_restart_fires_on_a_sphere():
    """(a) A generous budget on a sphere converges, so a criterion must fire."""
    h, fx = _run(max_eval=1200)
    assert h.n_restarts >= 1
    assert h.n_self_restarts == h.n_restarts
    assert h.last_stop_reason in (
        "tolx",
        "tolfun",
        "tolfunhist",
        "stagnation",
        "conditioncov",
        "noeffectaxis",
        "noeffectcoord",
    )
    # IPOP: every restart multiplies λ by ipop_factor (2.0 by default).
    assert h._lam == 7 * 2**h.n_restarts
    assert float(fx.min()) < 1e-6


def test_self_restart_off_reproduces_the_pre_change_trajectory():
    """(b) ``self_restart=False`` is the old code, to the last bit.

    The three expected numbers were recorded with the *pre-change* module on
    2026-09-10 (commit c2e4b6f, seed 42, DeJong(3) on [-2, 8]^3, 400 evals):
    they are the regression pin for "one flag away from the old behaviour".
    """
    h, fx = _run(self_restart=False)
    assert h.n_restarts == 0
    assert len(fx) == 406
    assert float(np.min(fx)) == 5.523971143576693e-09
    assert float(np.sum(fx)) == 701.12430982615172


def test_restart_event_still_restarts():
    """(c) The analyzer-driven ``restart`` event works exactly as before."""
    from panobbgo.analyzers.restart import Restart

    # ``patience`` far below the budget, so the analyzer is guaranteed to fire
    # once the sphere has converged; self-restart is off so every restart
    # counted here came through the event bus.
    h, _ = _run(max_eval=800, self_restart=False, analyzers=(lambda s: Restart(s, patience=40),))
    assert h.n_self_restarts == 0
    assert h.n_restarts >= 1

    # ... and a direct call re-centres the distribution on the given point.
    h2, _ = _run(max_eval=200, self_restart=False)
    before = h2.n_restarts
    center = np.array([1.0, -1.0, 2.0])
    h2.on_restart(center, reason="test")
    assert h2.n_restarts == before + 1
    np.testing.assert_allclose(h2._m, center)


def test_restart_from_is_reproducible_and_respected():
    """(d) The start point comes from ``self.rng``: same seed → same point."""

    # The spy needs the heuristic instance, so the run is built inline here
    # rather than through ``_run``.
    def run_and_record(seed=42, **kw):
        problem = DeJong(3, box=list(BOX))
        np.random.seed(seed)
        s = StrategyRoundRobin(problem, max_evaluations=1200, seed=seed, parse_args=False, testing_mode=True)
        s.config.sync_evaluation = True
        s.config.stop_on_convergence = False
        h = CMAES(s, **kw)
        recorded: list[np.ndarray] = []
        orig = h._restart_center

        def wrapped():
            c = orig()
            recorded.append(np.asarray(c, dtype=float).copy())
            return c

        h._restart_center = wrapped  # type: ignore[method-assign]
        s.add_heuristic(h)
        s.start()
        return h, recorded

    h_a, a = run_and_record(restart_from="random")
    h_b, b = run_and_record(restart_from="random")
    assert len(a) == len(b) >= 1
    assert h_a.n_self_restarts == len(a)
    for ca, cb in zip(a, b):
        np.testing.assert_array_equal(ca, cb)

    box_center = np.full(3, 3.0)
    # A uniform draw in [-2, 8]^3 is (almost surely) not the box centre.
    assert not np.allclose(a[0], box_center)

    _, c = run_and_record(restart_from="center")
    for ci in c:
        np.testing.assert_allclose(ci, box_center)

    h_best, d = run_and_record(restart_from="best")
    assert len(d) >= 1
    # The best point is inside the box and better than the box centre.
    for di in d:
        assert np.all(di >= -2.0) and np.all(di <= 8.0)
    assert float(np.dot(d[0], d[0])) < float(np.dot(box_center, box_center))
    assert h_best.n_restarts == len(d)


def test_invalid_kwargs_are_rejected():
    """Constructor validation matches the file's existing style."""
    import pytest

    problem = DeJong(3, box=list(BOX))
    s = StrategyRoundRobin(problem, max_evaluations=100, seed=1, parse_args=False, testing_mode=True)
    for kw in (
        {"restart_from": "nowhere"},
        {"tolx": -1.0},
        {"tolfun": -1e-3},
        {"tolfunhist": -1.0},
        {"conditioncov": 0.5},
        {"stagnation": -5},
    ):
        with pytest.raises(ValueError):
            CMAES(s, **kw)
