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

"""Tests for the BBOB bases and the failure regions of :mod:`panobbgo.lib.families`.

The BBOB bases are checked against a *literal* transcription of the
definitions in Hansen et al. (2009), "Real-parameter black-box optimization
benchmarking 2009: noiseless functions definitions" (RR-6829), written in
BBOB's own coordinates, plus hand-computed reference values.  The failure
regions are checked for geometry (share of the box, optimum outside) and
for how the evaluation paths book them (crash: a failed evaluation;
timeout: a ``Result.timed_out`` placeholder, without waiting).
"""

import pickle

import numpy as np
import pytest

from panobbgo.harness_families import (
    PenaltyTracker,
    describe_instances,
    make_failure_battery,
    make_shapes_battery,
    run_family_harness,
)
from panobbgo.harness_ioh import make_ioh_strategies
from panobbgo.lib import EvaluationCrashed, EvaluationTimedOut, Point
from panobbgo.lib.families import (
    BASE_FUNCTIONS,
    CONTEXT_BASES,
    BaseContext,
    FailureRegion,
    Family,
    FamilyConfig,
    make_family_instances,
    t_asy,
    t_osz,
)

BBOB_BASES = sorted(CONTEXT_BASES)


def _lam(alpha, dim):
    return np.array([alpha ** (0.5 * i / (dim - 1)) for i in range(dim)])


# ---------------------------------------------------------------------------
# Transformations
# ---------------------------------------------------------------------------


def test_t_osz_fixed_points_and_monotone():
    assert list(t_osz(np.array([0.0, 1.0, -1.0]))) == [0.0, 1.0, -1.0]
    xs = np.linspace(-50.0, 50.0, 2001)
    assert np.all(np.diff(t_osz(xs)) > 0.0)
    # One value by hand: x = e, x_hat = 1, c1 = 10, c2 = 7.9.
    assert t_osz(np.e)[0] == pytest.approx(np.exp(1.0 + 0.049 * (np.sin(10.0) + np.sin(7.9))))


def test_t_asy_reference_values():
    # D = 3, beta = 0.5: exponents 1, 1 + 0.25 sqrt(x), 1 + 0.5 sqrt(x) on positive entries.
    out = t_asy(np.array([4.0, 4.0, -4.0]), 0.5)
    assert out[0] == 4.0
    assert out[1] == pytest.approx(4.0**1.5)
    assert out[2] == -4.0  # negative entries are left alone


# ---------------------------------------------------------------------------
# The optimum
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("base", BBOB_BASES)
@pytest.mark.parametrize("dim", [2, 5, 10])
@pytest.mark.parametrize("rotate", [True, False])
def test_bbob_optimum_is_exact_and_global(base, dim, rotate):
    p = Family(base, dim=dim, seed=321, rotate=rotate)
    assert p.eval(p.x_opt) == p.f_opt
    rng = np.random.default_rng(4)
    xs = np.vstack([rng.uniform(-5.0, 5.0, size=(1500, dim)), p.x_opt + rng.normal(0.0, 0.05, size=(500, dim))])
    values = np.array([p.eval(x) for x in xs])
    assert values.min() >= p.f_opt


def test_bbob_bases_draw_from_their_own_stream():
    """A BBOB base leaves the instance stream alone: same x_opt, R, f_opt as any other base."""
    ref = Family("sphere", dim=4, seed=77)
    for base in BBOB_BASES:
        p = Family(base, dim=4, seed=77)
        assert np.array_equal(p.x_opt, ref.x_opt)
        assert np.array_equal(p.rotation, ref.rotation)
        assert p.f_opt == ref.f_opt


# ---------------------------------------------------------------------------
# Literal BBOB transcriptions (RR-6829)
# ---------------------------------------------------------------------------


def _probes(p, n=200):
    rng = np.random.default_rng(9)
    return np.vstack([rng.uniform(-5.0, 5.0, size=(n, p.dim)), p.x_opt + rng.normal(0.0, 0.3, size=(n, p.dim))])


@pytest.mark.parametrize("dim", [2, 5])
def test_attractive_sector_matches_bbob_f6(dim):
    p = Family("attractive_sector", dim=dim, seed=11)
    q, r, xopt = p._base_fn.q, p.rotation, p.x_opt
    for x in _probes(p):
        z = q @ (_lam(10.0, dim) * (r @ (x - xopt)))
        s = np.where(z * xopt > 0.0, 100.0, 1.0)
        want = t_osz(np.sum((s * z) ** 2))[0] ** 0.9 + p.f_opt
        assert p.eval(x) == pytest.approx(want, rel=1e-12)


def test_attractive_sector_reference_values():
    """Unrotated, ``x_opt = 0`` (sign pattern +1): T_osz(1) = 1, so both points give exactly 1."""
    f = BASE_FUNCTIONS["attractive_sector"](
        2, BaseContext(rng=np.random.default_rng(0), x_opt=np.zeros(2), rotate=False)
    )
    assert f(np.array([-1.0, 0.0])) == pytest.approx(1.0)
    assert f(np.array([0.01, 0.0])) == pytest.approx(1.0)  # the penalised side: 100 * 0.01 = 1
    assert f(np.array([1.0, 0.0])) > 1e3  # asymmetric: the mirror point is far worse


@pytest.mark.parametrize("dim", [2, 5])
def test_step_ellipsoid_matches_bbob_f7(dim):
    p = Family("step_ellipsoid", dim=dim, seed=12)
    q, r, xopt = p._base_fn.q, p.rotation, p.x_opt
    for x in _probes(p):
        zh = _lam(10.0, dim) * (r @ (x - xopt))
        zt = np.array([np.floor(0.5 + v) if abs(v) > 0.5 else np.floor(0.5 + 10.0 * v) / 10.0 for v in zh])
        z = q @ zt
        inner = sum(10.0 ** (2.0 * i / (dim - 1)) * z[i] ** 2 for i in range(dim))
        want = 0.1 * max(abs(zh[0]) / 1e4, inner) + p.f_opt
        assert p.eval(x) == pytest.approx(want, rel=1e-12)


def test_step_ellipsoid_reference_values():
    f = BASE_FUNCTIONS["step_ellipsoid"](2, BaseContext(rng=np.random.default_rng(0), x_opt=np.zeros(2), rotate=False))
    assert f(np.array([1.0, 0.0])) == pytest.approx(0.1)
    assert f(np.array([0.7, 0.0])) == pytest.approx(0.1)  # rounded to 1: a plateau
    assert f(np.array([0.03, 0.0])) == pytest.approx(0.1 * 0.03 / 1e4)  # only the |z_1| term
    assert f(np.array([0.0, 0.01])) == 0.0  # on the optimal slab
    assert f(np.array([0.0, 2.0 / np.sqrt(10.0)])) == pytest.approx(0.1 * 100.0 * 4.0)


@pytest.mark.parametrize("dim", [2, 5])
def test_bent_cigar_matches_bbob_f12(dim):
    p = Family("bent_cigar", dim=dim, seed=13)
    r, xopt = p.rotation, p.x_opt
    for x in _probes(p):
        z = r @ t_asy(r @ (x - xopt), 0.5)
        want = z[0] ** 2 + 1e6 * np.sum(z[1:] ** 2) + p.f_opt
        assert p.eval(x) == pytest.approx(want, rel=1e-12)


def test_bent_cigar_reference_values():
    f = BASE_FUNCTIONS["bent_cigar"](3)  # the default context is unrotated
    assert f(np.array([1.0, 0.0, 0.0])) == 1.0
    assert f(np.array([-2.0, 0.0, 0.0])) == 4.0
    assert f(np.array([0.0, 1.0, 0.0])) == 1e6
    assert f(np.array([0.0, 4.0, 0.0])) == pytest.approx(1e6 * 8.0**2)  # 4 ** (1 + 0.25 * 2)


@pytest.mark.parametrize("dim", [2, 5])
def test_gallagher_matches_bbob_f21(dim):
    p = Family("gallagher", dim=dim, seed=14)
    g = p._base_fn
    r = p.rotation
    ys, cs, ws = g.peaks_x, g.c, g.weights
    for x in _probes(p, n=100):
        vals = [w * np.exp(-((r @ (x - y)) @ (c * (r @ (x - y)))) / (2.0 * dim)) for y, c, w in zip(ys, cs, ws)]
        want = t_osz(10.0 - max(vals))[0] ** 2 + p.f_opt
        assert p.eval(x) == pytest.approx(want, rel=1e-9, abs=1e-12)


def test_gallagher_structure_is_bbob_f21_and_f22():
    dim = 4
    g = Family("gallagher", dim=dim, seed=15)._base_fn
    assert g.peaks.shape == (101, dim)
    assert g.weights[0] == 10.0 and g.weights[1] == pytest.approx(1.1) and g.weights[-1] == pytest.approx(9.1)
    # C_1 = Lambda^1000 / 1000^(1/4), permuted; the others use 1000^(2j/99), j = 0..99.
    np.testing.assert_allclose(np.sort(g.c[0]), np.sort(_lam(1000.0, dim) / 1000.0**0.25))
    assert np.all(np.abs(g.peaks_x) <= 5.0)

    f22 = Family("gallagher", dim=dim, seed=15, base_params={"n_peaks": 21, "alpha_opt": 1e6})._base_fn
    assert f22.peaks.shape == (21, dim)
    assert f22.weights[-1] == pytest.approx(9.1)
    np.testing.assert_allclose(np.sort(f22.c[0]), np.sort(_lam(1e6, dim) / 1e6**0.25))


def test_lunacek_matches_bbob_f24():
    """With BBOB's own ``x_opt = mu0/2 * 1`` the recentred base *is* f24."""
    dim = 5
    rng = np.random.default_rng(16)
    r = np.linalg.qr(rng.standard_normal((dim, dim)))[0]
    xopt = np.full(dim, 1.25)
    ctx = BaseContext(rng=np.random.default_rng(3), x_opt=xopt, rotation=r)
    f = BASE_FUNCTIONS["lunacek_bi_rastrigin"](dim, ctx)
    mu0, s = 2.5, 1.0 - 1.0 / (2.0 * np.sqrt(dim + 20.0) - 8.2)
    mu1 = -np.sqrt((mu0**2 - 1.0) / s)
    for x in rng.uniform(-5.0, 5.0, size=(200, dim)):
        xh = 2.0 * np.sign(xopt) * x
        z = f.q @ (_lam(100.0, dim) * (r @ (xh - mu0)))
        want = min(np.sum((xh - mu0) ** 2), dim + s * np.sum((xh - mu1) ** 2)) + 10.0 * (
            dim - np.sum(np.cos(2.0 * np.pi * z))
        )
        assert f(r @ (x - xopt)) == pytest.approx(want, rel=1e-12)


def test_lunacek_second_funnel_and_knobs():
    dim = 10
    f = BASE_FUNCTIONS["lunacek_bi_rastrigin"](dim)
    assert f.s == pytest.approx(1.0 - 1.0 / (2.0 * np.sqrt(30.0) - 8.2))
    assert f.mu1 == pytest.approx(-np.sqrt((6.25 - 1.0) / f.s))
    # Every instance keeps the second funnel inside the box (towards the centre).
    for seed in range(20):
        p = Family("lunacek_bi_rastrigin", dim=dim, seed=seed)
        second = p.x_opt - (2.5 - p._base_fn.mu1) / 2.0 * np.where(p.x_opt >= 0, 1.0, -1.0)
        assert np.all(np.abs(second) <= 5.0)
    deep = Family("lunacek_bi_rastrigin", dim=2, seed=1, base_params={"d": 0.1, "s": 0.5})
    assert deep._base_fn.d == 0.1 and deep._base_fn.s == 0.5
    with pytest.raises(ValueError):
        Family("lunacek_bi_rastrigin", dim=2, seed=1, base_params={"d": 7.0})


def test_base_params_are_validated():
    with pytest.raises(ValueError, match="takes no knob"):
        Family("gallagher", dim=2, seed=1, base_params={"peaks": 3})
    with pytest.raises(ValueError, match="takes no knob"):
        Family("sphere", dim=2, seed=1, base_params={"n_peaks": 3})


@pytest.mark.parametrize("base", BBOB_BASES)
def test_bbob_instances_are_deterministic_and_pickle(base):
    a = Family(base, dim=3, seed=5)
    b = Family(base, dim=3, seed=5)
    c = pickle.loads(pickle.dumps(a))
    for x in _probes(a, n=20):
        assert a.eval(x) == b.eval(x) == c.eval(x)


# ---------------------------------------------------------------------------
# Failure regions: geometry
# ---------------------------------------------------------------------------


def _mc_share(p, n=20000, seed=123):
    xs = np.random.default_rng(seed).uniform(-5.0, 5.0, size=(n, p.dim))
    return float(np.mean([p.failure_at(x) is not None for x in xs]))


@pytest.mark.parametrize("shape", ["halfspace", "ball", "boxes"])
@pytest.mark.parametrize("dim", [2, 5])
@pytest.mark.parametrize("share", [0.1, 0.3])
def test_failure_share_is_right_and_the_optimum_outside(shape, dim, share):
    for seed in range(3):
        p = Family("sphere", dim=dim, seed=seed, failure=FailureRegion(shape, share=share))
        assert p.failure_at(p.x_opt) is None
        assert p.eval(p.x_opt) == p.f_opt
        assert p.failure_share == pytest.approx(share, abs=0.01)
        assert _mc_share(p) == pytest.approx(share, abs=0.015)


def test_halfspace_boundary_through_the_optimum():
    """``boundary_gap=0``: the optimum sits on the stability limit; one step past it fails."""
    for seed in range(5):
        p = Family("ellipsoid", dim=5, seed=seed, failure=FailureRegion("halfspace", share=0.25, boundary_gap=0.0))
        geom = p._failure_geom
        assert geom.t == p.x_opt[geom.axis]
        assert p.failure_at(p.x_opt) is None
        step = np.zeros(5)
        step[geom.axis] = 1e-9 * geom.sign
        assert p.failure_at(p.x_opt + step) == "crash"
        assert p.failure_at(p.x_opt - step) is None
        assert 0.0 < p.failure_share <= 1.0
    near = Family("sphere", dim=3, seed=1, failure=FailureRegion("halfspace", share=0.2, boundary_gap=0.1))
    assert abs(near._failure_geom.t - near.x_opt[near._failure_geom.axis]) == pytest.approx(0.5)


def test_failure_region_is_deterministic_and_pairs_with_the_plain_instance():
    region = FailureRegion("boxes", share=0.2, n_boxes=3)
    a = Family("rastrigin", dim=3, seed=9, failure=region)
    b = Family("rastrigin", dim=3, seed=9, failure=region)
    plain = Family("rastrigin", dim=3, seed=9)
    np.testing.assert_array_equal(a._failure_geom.lo, b._failure_geom.lo)
    np.testing.assert_array_equal(a._failure_geom.hi, b._failure_geom.hi)
    # The region has a stream of its own: the instance itself is unchanged.
    assert np.array_equal(a.x_opt, plain.x_opt) and a.f_opt == plain.f_opt
    c = pickle.loads(pickle.dumps(a))
    xs = np.random.default_rng(0).uniform(-5.0, 5.0, size=(200, 3))
    assert [a.failure_at(x) for x in xs] == [c.failure_at(x) for x in xs]
    # Outside the region, the objective is the plain instance's.
    for x in xs:
        if a.failure_at(x) is None:
            assert a.eval(x) == plain.eval(x)


def test_failure_region_validation_and_labels():
    with pytest.raises(ValueError):
        FailureRegion("cube")
    with pytest.raises(ValueError):
        FailureRegion("ball", mode="hang")
    with pytest.raises(ValueError):
        FailureRegion("ball", share=0.8)
    with pytest.raises(ValueError):
        FailureRegion("ball", boundary_gap=0.1)
    p = Family("sphere", dim=2, seed=1, failure=FailureRegion("ball", mode="timeout"))
    assert p.family == "sphere_fball_tmo"
    cfg = FamilyConfig(base="sphere", failure=FailureRegion("halfspace"))
    assert cfg.name() == "sphere_fhs_crash"
    # A plain and a failure family of one base can share a battery.
    names = [n for n, _p in make_family_instances(["sphere", cfg], dims=(2,), n_instances=1)]
    assert names == ["sphere_d2_i0", "sphere_fhs_crash_d2_i0"]


# ---------------------------------------------------------------------------
# Failure regions: booking
# ---------------------------------------------------------------------------


def _inside(p):
    xs = np.random.default_rng(1).uniform(-5.0, 5.0, size=(2000, p.dim))
    return next(x for x in xs if p.failure_at(x) is not None)


def test_crash_raises_and_timeout_becomes_a_timed_out_result():
    crash = Family("sphere", dim=2, seed=3, failure=FailureRegion("ball", share=0.3, mode="crash"))
    with pytest.raises(EvaluationCrashed):
        crash(Point(_inside(crash), "test"))

    tmo = Family("sphere", dim=2, seed=3, failure=FailureRegion("ball", share=0.3, mode="timeout"))
    x = _inside(tmo)
    with pytest.raises(EvaluationTimedOut):
        tmo.eval(x)
    r = tmo(Point(x, "test"))
    assert r.timed_out and np.isnan(r.fx) and r.cv == float("inf")
    ok = tmo(Point(tmo.x_opt, "test"))
    assert not ok.timed_out and ok.fx == tmo.f_opt


def test_tracker_counts_a_failed_call_as_spent_budget():
    p = Family("sphere", dim=2, seed=3, failure=FailureRegion("ball", share=0.3, mode="crash"))
    bad = _inside(p)
    tracker = PenaltyTracker(p, budget=10)
    try:
        p.eval(p.x_opt)
        with pytest.raises(EvaluationCrashed):
            p.eval(bad)
        assert tracker.n_evals == 2
        assert tracker.best_so_far == [p.f_opt, p.f_opt]  # spent, no progress
    finally:
        tracker.restore()


@pytest.mark.parametrize("mode", ["crash", "timeout"])
def test_failed_evaluations_are_booked_by_a_strategy(mode):
    """Through a real strategy: crashes leave no result, timeouts a NaN placeholder; both use budget."""
    p = Family("sphere", dim=2, seed=3, failure=FailureRegion("halfspace", share=0.4, mode=mode))
    spec = [s for s in make_ioh_strategies() if s.name == "RoundRobin_Random"][0]
    strategy = spec.create_strategy(p, seed=1, max_eval=60)
    strategy.config.sync_evaluation = True
    strategy.start()
    df = strategy.results.results
    xs = df["x"].to_numpy()
    inside = np.array([p.failure_at(x) is not None for x in xs])
    if mode == "crash":
        assert not inside.any()  # a crash leaves no result ...
        assert strategy.n_finished == 60 > len(strategy.results)  # ... but was paid for
        assert strategy.n_timed_out == 0
    else:
        timed_out = df[("timed_out", 0)].to_numpy().astype(bool)
        assert len(strategy.results) == 60
        assert np.array_equal(timed_out, inside) and inside.any()
        assert strategy.n_timed_out == int(inside.sum())
        assert np.all(np.isnan(df[("fx", 0)].to_numpy()[timed_out]))


def test_failure_run_through_the_harness_spends_the_budget():
    instances = [x for x in make_failure_battery(dims=(2,), n_instances=1)]
    spec = [s for s in make_ioh_strategies() if s.name == "RoundRobin_Random"]
    result = run_family_harness(spec, instances, budget_multiplier=20, base_seed=3, progress=False)
    assert len(result.runs) == 4
    for run in result.runs:
        assert run.error is None
        assert run.n_evals == run.budget == 40
        assert run.precision >= -1e-9


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


def test_new_presets_have_the_documented_shape():
    shapes = describe_instances(make_shapes_battery())
    assert shapes["families"] == sorted(BBOB_BASES)
    assert shapes["dims"] == [2, 5, 10] and shapes["n_instances"] == 5 * 3 * 3
    assert shapes["failure"] == []

    failure = make_failure_battery()
    d = describe_instances(failure)
    assert d["dims"] == [2, 5] and d["n_instances"] == 4 * 2 * 3
    assert d["failure"] == ["fball_crash", "fbox_tmo", "fhs_crash", "fhs_tmo"]
    for _n, p in failure:
        assert p.failure_at(p.x_opt) is None and p.eval(p.x_opt) == p.f_opt
