# -*- coding: utf8 -*-
# Copyright 2012-2026 Harald Schilly <harald.schilly@gmail.com>
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
"""``CMAES(boundary=, first_start=, active=)`` — the three options of DISCOVERY §57.

* H1 ``boundary``: ``"project"`` (default), ``"resample"`` (the ``cmaes`` /
  Optuna scheme: up to ``10·n`` redraws, then one more draw projected),
  ``"reflect"`` (periodic reflection at the faces).
* H2 ``first_start``: ``"center"`` (default) or ``"random"``.
* H3 ``active``: negative recombination weights, Hansen (2016),
  arXiv:1604.00772, eq. 46–53.  The default since DISCOVERY §60 (with the
  repair guard of §59); ``active=False`` is the positive-only update.

Pinned here: ``active=False`` with the other defaults is the old code bit for
bit (a numeric pin recorded with the pre-change module), the new default has
its own numeric pin, every draw comes from the heuristic's own keyed stream,
and the active weights and covariance update are the tutorial's (checked
against a from-scratch formula and, when the ``baselines`` extra is installed,
against the ``cmaes`` library Optuna uses).
"""

from __future__ import annotations

import numpy as np
import pytest

from panobbgo.heuristics import CMAES
from panobbgo.lib.classic import DeJong, Rosenbrock
from panobbgo.strategies import StrategyRoundRobin

# A sphere whose optimum (0, 0, 0) sits near the lower face -2 and far from
# the box centre (3, 3, 3): the positive-only run projects 50 of its 1200
# points onto that face (the default, active run 63), so every boundary path
# is exercised.
BOX = [(-2.0, 8.0)] * 3


def _run(seed=42, max_eval=1200, problem=None, **kw):
    """One solo-CMAES sync run; returns ``(heuristic, X, fx, who)``."""
    problem = problem or DeJong(3, box=list(BOX))
    s = StrategyRoundRobin(problem, max_evaluations=max_eval, seed=seed, parse_args=False, testing_mode=True)
    s.config.sync_evaluation = True
    s.config.stop_on_convergence = False
    h = CMAES(s, **kw)
    s.add_heuristic(h)
    s.start()
    df = s.results.results
    assert df is not None
    X = np.stack(list(df["x"].to_numpy())).astype(float)
    return h, X, df["fx"].to_numpy(dtype=float).ravel(), list(df["who"].to_numpy().ravel().astype(str))


def _on_face(X, lo=-2.0, hi=8.0):
    return int(np.sum(np.any((X == lo) | (X == hi), axis=1)))


# ---------------------------------------------------------------------------
# Defaults: bit-identical
# ---------------------------------------------------------------------------


def test_explicit_defaults_equal_omitted():
    """Passing the documented defaults is the same run as not passing them."""
    _, X0, fx0, who0 = _run()
    _, X1, fx1, who1 = _run(boundary="project", first_start="center", active=True, active_skip_repaired=True)
    np.testing.assert_array_equal(X0, X1)
    np.testing.assert_array_equal(fx0, fx1)
    assert who0 == who1


def test_positive_only_trajectory_pinned_to_the_pre_change_module():
    """``active=False`` reproduces the run recorded before the options existed.

    Recorded 2026-09-26 with the module at 097d797 (seed 42, DeJong(3) on
    [-2, 8]^3, 1200 evaluations, one IPOP self-restart, 50 points projected
    onto the lower face).  That was the default path until DISCOVERY §60
    made active CMA the default; the pin keeps the positive-only update
    bit-identical.  ``test_self_restart_off_reproduces_the_pre_change_trajectory``
    pins the no-restart path the same way.  ``quorum="fraction",
    late_results="drop"``: after the restart λ = 14 exceeds RoundRobin's
    10-point request, and since §64 such a generation waits for its second
    batch instead of closing on the first.
    """
    h, X, fx, _ = _run(active=False, quorum="fraction", late_results="drop")
    assert len(fx) == 1200
    assert h.n_restarts == 1
    assert _on_face(X) == 50
    assert float(np.min(fx)) == pytest.approx(7.273024102978022e-17, rel=1e-6)  # BLAS order differs across CPUs
    assert float(np.sum(fx)) == pytest.approx(2522.28979759724, rel=1e-9)


def test_active_drop_trajectory_pinned():
    """Guarded active CMA (DISCOVERY §60) with the pre-§64 quorum: the default until §64.

    Recorded 2026-09-27 with the commit that made ``active=True`` the default
    (seed 42, DeJong(3) on [-2, 8]^3, 1200 evaluations, one IPOP self-restart,
    63 points projected onto the lower face).  After the restart 64 late
    offspring are dropped.
    """
    h, X, fx, _ = _run(quorum="fraction", late_results="drop")
    assert len(fx) == 1200
    assert h.n_restarts == 1
    assert _on_face(X) == 63
    assert h.n_late_dropped == 64
    assert float(np.min(fx)) == pytest.approx(1.5349983286946815e-15, rel=1e-6)  # BLAS order differs across CPUs
    assert float(np.sum(fx)) == pytest.approx(6155.544241568712, rel=1e-9)


def test_default_trajectory_pinned():
    """The default path (guarded active CMA, ``quorum="dispatched"``, DISCOVERY §64) on the same run.

    Recorded 2026-09-27: the same best value as the pre-§64 run above; after
    the IPOP restart each λ = 14 generation waits for its second synchronous
    batch, so nothing arrives late.
    """
    h, X, fx, _ = _run()
    assert len(fx) == 1200
    assert h.n_restarts == 1
    assert _on_face(X) == 67
    assert (h.n_late_folded, h.n_late_dropped) == (0, 0)
    assert float(np.min(fx)) == pytest.approx(1.5349983286946815e-15, rel=1e-6)  # BLAS order differs across CPUs
    assert float(np.sum(fx)) == pytest.approx(6078.039386907837, rel=1e-9)


@pytest.mark.parametrize(
    "kw",
    [
        {"boundary": "resample"},
        {"boundary": "reflect"},
        {"first_start": "random"},
        {"active": False},
        {"active_skip_repaired": False},
    ],
)
def test_each_option_changes_the_run_and_is_reproducible(kw):
    """Every option is read (not a dead parameter), and a seeded run stays reproducible."""
    _, _, fx0, _ = _run()
    _, X1, fx1, who1 = _run(**kw)
    _, X2, fx2, who2 = _run(**kw)
    assert not (len(fx0) == len(fx1) and np.array_equal(fx0, fx1))
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(fx1, fx2)
    assert who1 == who2


def test_invalid_values_raise():
    s = StrategyRoundRobin(DeJong(2), parse_args=False, testing_mode=True, seed=1)
    with pytest.raises(ValueError, match="boundary"):
        CMAES(s, boundary="clip")
    with pytest.raises(ValueError, match="first_start"):
        CMAES(s, first_start="best")


# ---------------------------------------------------------------------------
# H1: boundary handling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("boundary", ["resample", "reflect"])
def test_resample_and_reflect_put_no_point_on_a_face(boundary):
    """Projection piles points onto the face; resampling and reflecting do not."""
    _, X, fx, _ = _run(boundary=boundary)
    assert len(fx) == 1200
    assert np.all(X >= -2.0) and np.all(X <= 8.0)
    assert _on_face(X) == 0
    assert float(np.min(fx)) < 1e-10


class _CountingRng:
    """Wraps a Generator and counts ``standard_normal`` calls."""

    def __init__(self, rng):
        self._rng = rng
        self.normal_calls = 0

    def standard_normal(self, *a, **k):
        self.normal_calls += 1
        return self._rng.standard_normal(*a, **k)

    def __getattr__(self, name):
        return getattr(self._rng, name)


def _started(seed=5, dim=3, **kw):
    """A heuristic whose ``on_start`` has run (a 1-generation run); state is live."""
    problem = DeJong(dim, box=[(-2.0, 8.0)] * dim)
    s = StrategyRoundRobin(problem, max_evaluations=7, seed=seed, parse_args=False, testing_mode=True)
    s.config.sync_evaluation = True
    s.config.stop_on_convergence = False
    h = CMAES(s, **kw)
    s.add_heuristic(h)
    s.start()
    return h


def test_resample_fallback_draw_count_and_projection():
    """No draw inside: N = 10·n draws are checked (the caller's is the first), then one more is returned.

    ``cmaes.CMA.ask``: ``for i in range(n_max_resampling): sample; if
    feasible: return`` — then one more sample, clipped.  So the helper makes
    ``10·n − 1`` checked draws plus the extra one: ``10·n`` calls.
    """
    h = _started(boundary="resample")
    n = h.problem.dim
    h._m = np.full(n, 1e6)  # far outside: every draw is out of the box
    h._sigma = 1e-3
    counter = _CountingRng(h.rng)
    h.rng = counter  # type: ignore[assignment]
    x_raw, y = h._resample_inside(h._m + h._sigma * np.ones(n), np.ones(n))
    assert counter.normal_calls == CMAES.RESAMPLE_PER_DIM * n
    assert not h._inside(x_raw)
    np.testing.assert_allclose(x_raw, h._m + h._sigma * y)


def test_a_point_on_a_face_is_inside():
    """``cmaes`` accepts ``lo <= x <= hi`` (``_is_feasible``), faces included; so does ``_inside``."""
    h = _started(boundary="resample")  # box [-2, 8]^3
    assert h._inside(np.array([-2.0, 8.0, 3.0]))
    assert h._inside(np.array([-2.0, -2.0, -2.0]))
    assert not h._inside(np.array([-2.0, 8.0 + 1e-12, 3.0]))
    assert not h._inside(np.array([np.nextafter(-2.0, -np.inf), 3.0, 3.0]))
    # A resample draw that lands exactly on a face is kept, not redrawn.
    h._m, h._sigma = np.array([-2.0, 3.0, 8.0]), 1.0
    h._B, h._D = np.eye(3), np.ones(3)

    class _Zero:
        calls = 0

        def standard_normal(self, n):
            _Zero.calls += 1
            return np.zeros(n)

    h.rng = _Zero()  # type: ignore[assignment]
    x_raw, y = h._resample_inside(np.full(3, 100.0), np.full(3, 97.0))
    assert _Zero.calls == 1
    np.testing.assert_array_equal(x_raw, [-2.0, 3.0, 8.0])
    np.testing.assert_array_equal(y, 0.0)


def test_resample_returns_the_first_inside_draw():
    h = _started(boundary="resample")
    n = h.problem.dim
    h._m = np.full(n, 3.0)
    h._sigma = 1.0
    counter = _CountingRng(h.rng)
    h.rng = counter  # type: ignore[assignment]
    x_raw, y = h._resample_inside(np.full(n, 100.0), np.full(n, 97.0))
    assert counter.normal_calls == 1  # σ = 1 around the centre of a width-10 box: the first redraw is inside
    assert h._inside(x_raw)
    np.testing.assert_allclose(x_raw, h._m + h._sigma * y)


def test_reflect_is_periodic_and_keeps_inside_coordinates():
    h = _started(boundary="reflect")  # box [-2, 8]^3, range 10
    x = np.array([9.0, -2.5, 3.25])
    out = h._reflect_into_box(x)
    np.testing.assert_allclose(out[:2], [7.0, -1.5], atol=1e-12)
    assert out[2] == 3.25  # bit for bit
    far = h._reflect_into_box(np.array([8.0 + 10.0 + 2.0, -2.0 - 23.0, 0.0]))  # 2 past the far side, 23 below
    np.testing.assert_allclose(far[:2], [0.0, 1.0], atol=1e-12)  # -25: up 23 from -2 is 21, back from 8 to -5, up to 1
    inside = np.array([-2.0, 8.0, 1.0])
    assert h._reflect_into_box(inside) is inside


# ---------------------------------------------------------------------------
# H2: first start
# ---------------------------------------------------------------------------


def test_first_start_random_draws_from_the_keyed_stream():
    """The first mean is a uniform box point from ``self.rng``, not the centre."""
    means = []
    for _ in range(2):
        problem = DeJong(3, box=list(BOX))
        s = StrategyRoundRobin(problem, max_evaluations=7, seed=42, parse_args=False, testing_mode=True)
        s.config.sync_evaluation = True
        h = CMAES(s, first_start="random")
        orig = h._emit_generation

        def spy(h=h, orig=orig):
            if h._gen == 0:
                means.append(np.array(h._m, copy=True))
            return orig()

        h._emit_generation = spy
        s.add_heuristic(h)
        s.start()
    np.testing.assert_array_equal(means[0], means[1])
    assert not np.allclose(means[0], 3.0)
    assert np.all(means[0] >= -2.0) and np.all(means[0] <= 8.0)


# ---------------------------------------------------------------------------
# H3: active CMA
# ---------------------------------------------------------------------------


def _tutorial(lam, n):
    """Hansen (2016) Table 1 / eq. 49–53, written out from the paper."""
    mu = lam // 2
    wp = np.array([np.log((lam + 1) / 2) - np.log(i) for i in range(1, lam + 1)])
    mu_eff = wp[:mu].sum() ** 2 / (wp[:mu] ** 2).sum()
    mu_eff_m = wp[mu:].sum() ** 2 / (wp[mu:] ** 2).sum()
    c1 = 2 / ((n + 1.3) ** 2 + mu_eff)
    cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + 2 * mu_eff / 2))
    a = min(1 + c1 / cmu, 1 + 2 * mu_eff_m / (mu_eff + 2), (1 - c1 - cmu) / (n * cmu))
    pos = wp[wp >= 0].sum()
    neg = np.abs(wp[wp < 0]).sum()
    w = np.where(wp >= 0, wp / pos, a * wp / neg)
    return w, mu_eff, c1, cmu


@pytest.mark.parametrize("lam,n", [(6, 2), (8, 5), (10, 10), (7, 3), (14, 3), (5, 20)])
def test_active_weights_follow_the_tutorial(lam, n):
    h = _started(dim=n, active=True)
    h._set_population(lam)
    w, mu_eff, c1, cmu = _tutorial(lam, n)
    mu = lam // 2
    np.testing.assert_allclose(h._w, w[:mu], rtol=1e-13)
    np.testing.assert_allclose(h._w_neg, w[mu:], rtol=1e-13, atol=1e-16)
    assert h._mu_eff == pytest.approx(mu_eff, rel=1e-13)
    assert h._c_1 == pytest.approx(c1, rel=1e-13)
    assert h._c_mu == pytest.approx(cmu, rel=1e-13)
    assert float(np.sum(h._w)) == pytest.approx(1.0, rel=1e-14)
    assert np.all(h._w_neg <= 0.0)
    if lam % 2 == 0:
        # even λ: the positive weights are the ones the positive-only path uses
        np.testing.assert_allclose(h._w, CMAES._recombination_weights(mu)[0], rtol=1e-13)


@pytest.mark.parametrize("lam,n", [(6, 2), (8, 5), (10, 10), (7, 3)])
def test_active_weights_match_the_cmaes_library(lam, n):
    """Cross-check against ``cmaes.CMA`` (the library Optuna's CmaEsSampler runs)."""
    cmaes = pytest.importorskip("cmaes", reason="the baselines extra is not installed")
    ref = cmaes.CMA(mean=np.zeros(n), sigma=1.0, population_size=lam, seed=1)
    h = _started(dim=n, active=True)
    h._set_population(lam)
    np.testing.assert_allclose(np.concatenate([h._w, h._w_neg]), ref._weights, rtol=1e-12, atol=1e-16)
    assert h._c_1 == pytest.approx(ref._c1, rel=1e-12)
    assert h._c_mu == pytest.approx(ref._cmu, rel=1e-12)


def test_positive_only_path_has_no_negative_weights():
    h = _started(active=False)
    assert h._w_neg.size == 0


def test_default_is_guarded_active_cma():
    """DISCOVERY §60: ``active=True`` with the repair guard is the default."""
    h = _started()
    assert h._active and h._active_skip_repaired
    assert h._w_neg.size == h._lam - h._mu and np.all(h._w_neg <= 0.0)


def _synthetic_generation(h, rng):
    n, lam = h.problem.dim, h._lam
    Y = [h._B @ (h._D * rng.standard_normal(n)) for _ in range(lam)]
    return [
        {"penalty": float(i), "x": h._m + h._sigma * y, "x_eval": h._m + h._sigma * y, "y": y} for i, y in enumerate(Y)
    ]


def _update_by_hand(h, entries, weights):
    """Snapshot ``h``, run ``h._update(entries)`` and return (expected C, expected p_c) per eq. (43)-(47).

    ``entries`` are in rank order; ``weights`` are the recombination weights
    applied to them (the μ positive ones, then any negative ones).
    """
    n, lam, mu = h.problem.dim, h._lam, h._mu
    # C is updated every generation but B, D only lazily, so the C an update
    # starts from is in general not B·D²·Bᵀ.  Make that explicit: eq. (46)
    # must use the *sampling* C^{-1/2} = B·D⁻¹·Bᵀ, and eq. (47) the current C.
    u = np.arange(1.0, n + 1.0)
    h._C = h._C + 0.3 * np.outer(u, u) / float(u @ u)
    C0, B, D, p_c0, p_s0 = h._C.copy(), h._B.copy(), h._D.copy(), h._p_c.copy(), h._p_sigma.copy()
    assert not np.allclose(C0, B @ np.diag(D**2) @ B.T)
    counteval0 = h._counteval
    h._eigeneval = h._counteval + 10**9  # keep the lazy eigendecomposition from rewriting C
    h._update(list(entries), n_offspring=lam)

    ys = [d["y"] for d in entries]
    y_w = sum(weights[i] * ys[i] for i in range(mu))
    C_is = B @ np.diag(1 / D) @ B.T
    p_s = (1 - h._c_sigma) * p_s0 + np.sqrt(h._c_sigma * (2 - h._c_sigma) * h._mu_eff) * C_is @ y_w
    g = counteval0 / lam + 1
    h_sig = np.linalg.norm(p_s) / np.sqrt(1 - (1 - h._c_sigma) ** (2 * g)) / h._chi_n < 1.4 + 2 / (n + 1)
    p_c = (1 - h._c_c) * p_c0 + h_sig * np.sqrt(h._c_c * (2 - h._c_c) * h._mu_eff) * y_w
    delta = (1 - h_sig) * h._c_c * (2 - h._c_c)
    rank = np.zeros((n, n))
    for i, wi in enumerate(weights):
        if wi < 0:
            wi = wi * n / float(np.linalg.norm(C_is @ ys[i]) ** 2)  # eq. (46)
        rank += wi * np.outer(ys[i], ys[i])
    C = (1 + h._c_1 * delta - h._c_1 - h._c_mu * sum(weights)) * C0 + h._c_1 * np.outer(p_c, p_c) + h._c_mu * rank
    return C, p_c


@pytest.mark.parametrize("dim", [3, 5])  # λ = 7 (odd: the first negative weight is 0) and 8
@pytest.mark.parametrize("active", [False, True])
def test_covariance_update_is_equation_47(active, dim):
    """One ``_update`` on a full synthetic generation equals eq. (47) computed by hand."""
    h = _started(dim=dim, active=active)
    entries = _synthetic_generation(h, np.random.default_rng(0))
    weights = list(h._w) + (list(h._w_neg) if active else [])
    C, p_c = _update_by_hand(h, entries, weights)
    np.testing.assert_allclose(h._C, C, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(h._p_c, p_c, rtol=1e-12, atol=1e-15)
    if active:
        assert sum(weights) < 1.0  # the negative part shrinks C along the worst directions


def test_active_partial_generation_uses_the_leading_negative_weights():
    """A generation that came back short (μ + 1 of λ) applies the negative weight of rank μ + 1 only."""
    h = _started(dim=5, active=True)  # λ = 8
    entries = _synthetic_generation(h, np.random.default_rng(1))[: h._mu + 1]
    weights = list(h._w) + [h._w_neg[0]]
    assert weights[-1] < 0
    C, _ = _update_by_hand(h, entries, weights)
    np.testing.assert_allclose(h._C, C, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("dim", [3, 5])
def test_active_guard_falls_back_to_positive_only_with_a_repaired_offspring(dim):
    """One repaired (projected) offspring anywhere in the generation: the update is eq. (47) without negatives."""
    h = _started(dim=dim, active=True)
    entries = _synthetic_generation(h, np.random.default_rng(2))
    entries[1]["repaired"] = True  # a selected one, as at a box face
    C, _ = _update_by_hand(h, entries, list(h._w))
    np.testing.assert_allclose(h._C, C, rtol=1e-12, atol=1e-15)


def test_active_guard_off_keeps_the_tutorial_update_with_repaired_offspring():
    h = _started(dim=5, active=True, active_skip_repaired=False)
    entries = _synthetic_generation(h, np.random.default_rng(2))
    entries[1]["repaired"] = True
    C, _ = _update_by_hand(h, entries, list(h._w) + list(h._w_neg))
    np.testing.assert_allclose(h._C, C, rtol=1e-12, atol=1e-15)


def test_active_guard_gives_injected_points_no_negative_weight():
    """pycma's CMA_active_injected = 0: an injected point ranked below μ contributes nothing negative."""
    h = _started(dim=5, active=True)  # λ = 8, μ = 4
    entries = _synthetic_generation(h, np.random.default_rng(3))
    entries[6]["injected"] = True
    weights = list(h._w) + list(h._w_neg)
    weights[6] = 0.0
    C, _ = _update_by_hand(h, entries, weights)
    np.testing.assert_allclose(h._C, C, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("skip", [True, False])
def test_injected_point_through_the_production_path(skip):
    """A foreign result → ``_maybe_inject`` → quorum → ``_update``: the flag, the zero weight, and the coupling.

    One switch controls both rules (the docstring's rule 2): with
    ``active_skip_repaired=True`` the injected point, ranked 7th of 9 (inside
    the negative ranks μ+1 … λ), gets weight 0; with ``False`` it keeps the
    tutorial's negative weight of its rank.
    """
    from panobbgo.lib import Point, Result
    from panobbgo.lib.constraints import DefaultConstraintHandler

    s = StrategyRoundRobin(DeJong(2, box=[(-5.0, 5.0)] * 2), parse_args=False, testing_mode=True, seed=7)
    s.constraint_handler = DefaultConstraintHandler(s)
    h = CMAES(s, inject=True, active=True, active_skip_repaired=skip, popsize=8, min_results_fraction=1.0)
    h.on_start()
    h.get_points()
    h._drop_generation()
    h._sigma = 1e-3  # deep inside the box: no offspring is repaired, so only rule 2 can act
    h._emit_generation()
    pts = h.get_points()
    assert len(pts) == 8

    captured = []
    h._update = lambda collected, n_offspring: captured.append(sorted(collected, key=lambda d: d["penalty"]))
    assert h._m is not None
    h.on_new_results([Result(Point(h._m + 1e-3, "OTHER:x"), 5.5)])  # between own offspring 5 and 6
    [entry] = h._injected[min(h._gen_results)]
    assert entry["injected"] is True  # set by _maybe_inject itself
    for i, p in enumerate(pts):
        h.on_new_results([Result(p, float(i))])
    del h._update  # the real method again
    [entries] = captured
    assert [e["penalty"] for e in entries] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 5.5, 6.0, 7.0]
    assert entries[6] is entry and not any(e.get("repaired", False) for e in entries)

    weights = list(h._w) + list(h._w_neg)
    assert weights[6] < 0
    if skip:
        weights[6] = 0.0
    C, _ = _update_by_hand(h, entries, weights)
    np.testing.assert_allclose(h._C, C, rtol=1e-12, atol=1e-15)


def test_emitted_offspring_carry_the_repaired_flag():
    """The flag is set exactly when the boundary repair changed the step (the positive-only path included)."""
    for active in (False, True):
        h = _started(dim=3, active=active)
        h._m = np.array([7.9, 3.0, 3.0])  # next to the upper face of [-2, 8]^3
        h._sigma = 1.0
        h._pending.clear()
        h._emit_generation()
        flags = [info["repaired"] for info in h._pending.values()]
        assert any(flags) and not all(flags)
        for info in h._pending.values():
            assert info["repaired"] == bool(np.any(info["x_eval"] >= 8.0))


def _linear_slope_aocc(seed, **kw):
    """BBOB f5 (linear slope, optimum on the face of [-5, 5]^5) at 500·d — the §58.4 failure case."""
    from panobbgo.ioh_runner import aocc
    from panobbgo.lib.lib import Problem

    class LinearSlope(Problem):
        def __init__(self, dim, rng):
            sgn = np.where(rng.random(dim) < 0.5, -1.0, 1.0)
            self.x_opt = 5.0 * sgn
            self.s = sgn * 10.0 ** (np.arange(dim) / (dim - 1))
            Problem.__init__(self, [(-5.0, 5.0)] * dim)

        def eval(self, x):
            z = np.where(self.x_opt * x < 25.0, x, self.x_opt)
            return float(np.sum(5.0 * np.abs(self.s) - self.s * z))

    _, _, fx, _ = _run(seed=seed, max_eval=2500, problem=LinearSlope(5, np.random.default_rng(1000 + seed)), **kw)
    return aocc(np.minimum.accumulate(fx), 0.0, budget=2500)


def test_active_guard_repairs_the_linear_slope():
    """Unguarded active CMA collapses C along a face-bound slope; the guard keeps it working.

    Measured 2026-09-27 on these seeds: guarded 0.97 / 0.58 / 0.98 / 0.73,
    unguarded 0.30 / 0.21 / 0.12 / 0.29 (positive-only 0.96 / 0.96 / 0.98 /
    0.88).  The margin is ~0.45 in the mean, far above any FP-environment
    drift of a chaotic trajectory.
    """
    seeds = (3, 4, 5, 6)
    guarded = np.mean([_linear_slope_aocc(s, active=True) for s in seeds])
    tutorial = np.mean([_linear_slope_aocc(s, active=True, active_skip_repaired=False) for s in seeds])
    assert guarded > tutorial + 0.2
    assert guarded > 0.6


def test_active_run_keeps_c_positive_definite_and_converges():
    h, _, fx, _ = _run(problem=Rosenbrock(dim=5), max_eval=3000, active=True)
    assert float(np.linalg.eigvalsh(h._C).min()) > 0.0
    assert float(np.min(fx)) < 1e-8


def test_all_three_combined():
    h, X, fx, _ = _run(boundary="resample", first_start="random", active=True)
    assert _on_face(X) == 0
    assert h._w_neg.size == h._lam - h._mu
    assert float(np.min(fx)) < 1e-10


# ---------------------------------------------------------------------------
# Harness wiring: the opt-in specs and the --bbob battery
# ---------------------------------------------------------------------------


def test_variant_specs_are_opt_in_and_share_the_flagship_seed_name():
    from panobbgo.harness_ioh import (
        CMAES_VARIANT_NAMES,
        CMAES_VARIANT_OPTIONS,
        make_cmaes_variant_strategies,
        make_ioh_strategies,
    )

    assert not set(CMAES_VARIANT_NAMES) & {s.name for s in make_ioh_strategies()}
    flagship = [s for s in make_ioh_strategies() if s.name == "RoundRobin_CMAES"][0]
    specs = make_cmaes_variant_strategies()
    assert [s.name for s in specs] == list(CMAES_VARIANT_NAMES)
    for spec in specs:
        assert spec.rng_identity == flagship.rng_identity == "RoundRobin_CMAES"
        assert spec.strategy_class is flagship.strategy_class
        [(cls, kw)] = spec.heuristics
        assert cls is CMAES and kw == CMAES_VARIANT_OPTIONS[spec.name] and kw
        CMAES(StrategyRoundRobin(DeJong(2), parse_args=False, testing_mode=True, seed=1), **kw)  # valid kwargs
    assert [s.name for s in make_cmaes_variant_strategies(["RoundRobin_CMAES_active", "nope"])] == [
        "RoundRobin_CMAES_active"
    ]
    # The §58 names keep their §58 meaning (unguarded); the guard has its own name (§59).
    assert CMAES_VARIANT_OPTIONS["RoundRobin_CMAES_resample_randstart_active"] == {
        "boundary": "resample",
        "first_start": "random",
        "active": True,
        "active_skip_repaired": False,
    }
    assert CMAES_VARIANT_OPTIONS["RoundRobin_CMAES_active"] == {"active": True, "active_skip_repaired": False}
    # Spelled out, so neither name depends on the constructor default (active=True since §60).
    assert CMAES_VARIANT_OPTIONS["RoundRobin_CMAES_active_guarded"] == {"active": True, "active_skip_repaired": True}
    assert CMAES_VARIANT_OPTIONS["RoundRobin_CMAES_positive"] == {"active": False}


def _ioh_cli():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "scripts" / "ioh_benchmark.py"
    spec = importlib.util.spec_from_file_location("ioh_benchmark_cma_ab", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_ioh_cli_adds_a_variant_only_when_named():
    import argparse

    cli = _ioh_cli()
    base = dict(legacy=False, standard=True, full=False, baselines=False)
    names = [s.name for s in cli._resolve_strategies(argparse.Namespace(strategies=None, **base))]
    assert not any(n.startswith("RoundRobin_CMAES_") for n in names)
    picked = cli._resolve_strategies(
        argparse.Namespace(strategies=["RoundRobin_CMAES", "RoundRobin_CMAES_resample"], **base)
    )
    assert [s.name for s in picked] == ["RoundRobin_CMAES", "RoundRobin_CMAES_resample"]


def test_ioh_cli_bbob_battery():
    import argparse

    cli = _ioh_cli()
    ns = argparse.Namespace(
        full=False,
        standard=False,
        noisy=None,
        noisy_highdim=None,
        highdim=False,
        large=False,
        largescale=False,
        sealed=False,
        bbob=True,
        bbob_dims=[5, 10],
        bbob_instances=[0, 1],
        bbob_fids=None,
        reps=None,
        budget_multiplier=500,
    )
    battery = cli._resolve_battery(ns)
    assert battery.problem_kind == "BBOB" and battery.dims == (5, 10) and battery.instances == (0, 1)
    assert battery.fids is not None and len(battery.fids) == 24
    assert battery.budget_multiplier == 500 and battery.name == "ioh-bbob-b500"
    ns.bbob_dims, ns.bbob_instances, ns.bbob_fids, ns.budget_multiplier = None, None, [22, 24], None
    battery = cli._resolve_battery(ns)
    assert battery.dims == (2, 5) and battery.instances == (0, 1, 2) and battery.fids == (22, 24)
    assert battery.budget_multiplier == 200
    with pytest.raises(SystemExit):
        cli.main(["run", "--standard", "--bbob-dims", "5"])
