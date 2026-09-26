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
        # Lunacek's default puts x_opt at BBOB's +-1.25 (same signs); "box" keeps the draw.
        params = {"placement": "box"} if base == "lunacek_bi_rastrigin" else {}
        p = Family(base, dim=4, seed=77, base_params=params)
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


@pytest.mark.parametrize("rotate", [True, False])
def test_lunacek_matches_bbob_f24(rotate):
    """The default instance *is* f24: ``x_opt = mu0/2 * sign``, the literal formula in x."""
    dim = 5
    p = Family("lunacek_bi_rastrigin", dim=dim, seed=16, rotate=rotate)
    xopt = p.x_opt
    assert np.all(np.abs(xopt) == 1.25)
    ref = Family("sphere", dim=dim, seed=16)
    assert np.array_equal(np.sign(xopt), np.sign(ref.x_opt))  # the signs of the instance draw
    r = p.rotation if rotate else np.eye(dim)
    q = p._base_fn.q
    mu0, s = 2.5, 1.0 - 1.0 / (2.0 * np.sqrt(dim + 20.0) - 8.2)
    mu1 = -np.sqrt((mu0**2 - 1.0) / s)
    rng = np.random.default_rng(16)
    for x in rng.uniform(-5.0, 5.0, size=(200, dim)):
        xh = 2.0 * np.sign(xopt) * x
        z = q @ (_lam(100.0, dim) * (r @ (xh - mu0)))
        want = min(np.sum((xh - mu0) ** 2), dim + s * np.sum((xh - mu1) ** 2)) + 10.0 * (
            dim - np.sum(np.cos(2.0 * np.pi * z))
        )
        assert p.eval(x) == pytest.approx(want + p.f_opt, rel=1e-12)


def test_lunacek_second_funnel_and_knobs():
    dim = 10
    f = BASE_FUNCTIONS["lunacek_bi_rastrigin"](dim)
    assert f.s == pytest.approx(1.0 - 1.0 / (2.0 * np.sqrt(30.0) - 8.2))
    assert f.mu1 == pytest.approx(-np.sqrt((6.25 - 1.0) / f.s))
    # Both placements keep the second funnel inside the box (towards the centre).
    for placement in ("bbob", "box"):
        for seed in range(20):
            p = Family("lunacek_bi_rastrigin", dim=dim, seed=seed, base_params={"placement": placement})
            second = p.x_opt - (2.5 - p._base_fn.mu1) / 2.0 * np.where(p.x_opt >= 0, 1.0, -1.0)
            assert np.all(np.abs(second) <= 5.0)
            assert p.eval(p.x_opt) == p.f_opt
    box = Family("lunacek_bi_rastrigin", dim=3, seed=2, base_params={"placement": "box"})
    assert np.array_equal(box.x_opt, Family("sphere", dim=3, seed=2).x_opt)
    deep = Family("lunacek_bi_rastrigin", dim=2, seed=1, base_params={"d": 0.1, "s": 0.5})
    assert deep._base_fn.d == 0.1 and deep._base_fn.s == 0.5
    with pytest.raises(ValueError):
        Family("lunacek_bi_rastrigin", dim=2, seed=1, base_params={"d": 7.0})
    with pytest.raises(ValueError):
        Family("lunacek_bi_rastrigin", dim=2, seed=1, base_params={"placement": "corner"})


@pytest.mark.parametrize("base", BBOB_BASES)
def test_bbob_bases_refuse_an_instance_condition(base):
    with pytest.raises(ValueError, match="own BBOB conditioning"):
        Family(base, dim=3, seed=1, condition=100.0)


def test_own_streams_are_spawned_children():
    """The base / failure streams are SeedSequence children, not ``[seed, k]`` (which is ``seed + k * 2**32``)."""
    p = Family("gallagher", dim=2, seed=5)
    want = np.random.default_rng(np.random.SeedSequence(5, spawn_key=(1,))).standard_normal(3)
    assert np.array_equal(p._base_rng().standard_normal(3), want)
    clash = np.random.default_rng(5 + 2**32).standard_normal(3)
    assert not np.array_equal(p._base_rng().standard_normal(3), clash)
    assert not np.array_equal(p._failure_rng().standard_normal(3), p._base_rng().standard_normal(3))


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


# ---------------------------------------------------------------------------
# Review round 1: booking in every path, robust placement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["crash", "timeout"])
def test_failed_evaluations_are_booked_in_worker_processes(mode):
    """The same booking with ``evaluation_method = "processes"``: the signal crosses the pipe."""
    p = Family("sphere", dim=2, seed=3, failure=FailureRegion("halfspace", share=0.4, mode=mode))
    spec = [s for s in make_ioh_strategies() if s.name == "RoundRobin_Random"][0]
    strategy = spec.create_strategy(p, seed=1, max_eval=30)
    strategy.config.sync_evaluation = True
    strategy.config.evaluation_method = "processes"
    strategy.start()
    df = strategy.results.results
    inside = np.array([p.failure_at(x) is not None for x in df["x"].to_numpy()])
    if mode == "crash":
        assert not inside.any()
        assert strategy.n_finished == 30 > len(strategy.results)
    else:
        timed_out = df[("timed_out", 0)].to_numpy().astype(bool)
        assert len(strategy.results) == 30
        assert np.array_equal(timed_out, inside) and inside.any()
        assert strategy.n_timed_out == int(inside.sum())


def _de_spec():
    from panobbgo.benchmark import StrategySpec
    from panobbgo.heuristics import DifferentialEvolution
    from panobbgo.strategies import StrategyRoundRobin

    return StrategySpec(
        name="RoundRobin_DE", strategy_class=StrategyRoundRobin, heuristics=[(DifferentialEvolution, {})]
    )


def test_cmaes_and_de_survive_a_crash_region():
    """A generation / slot that loses points to crashes keeps going, and converges.

    ``ellipsoid_fhs_crash`` d2, instance 0, optimum on the boundary: without
    ``CMAES.on_failed_evaluations`` the run ends at 480/1000 evaluations;
    updating from an all-crashed generation (instead of dropping it) blew
    sigma up and ended at precision ~0.75.
    """
    inst = [x for x in make_failure_battery(dims=(2,), n_instances=1) if x[1].family == "ellipsoid_fhs_crash"]
    specs = [s for s in make_ioh_strategies() if s.name == "RoundRobin_CMAES"] + [_de_spec()]
    result = run_family_harness(specs, inst, budget_multiplier=500, base_seed=1, progress=False)
    assert len(result.runs) == 2
    for run in result.runs:
        assert run.error is None, (run.strategy_name, run.error)
        assert run.n_evals == run.budget == 1000
    cma = next(r for r in result.runs if r.strategy_name == "RoundRobin_CMAES")
    assert cma.precision < 1e-6


def test_a_baseline_survives_a_crash_region():
    from panobbgo.harness_baselines import make_baseline_strategies

    inst = [x for x in make_failure_battery(dims=(2,), n_instances=1) if "crash" in x[0]]
    specs = [s for s in make_baseline_strategies() if s.name == "Baseline_SciPyDE"]
    result = run_family_harness(specs, inst, budget_multiplier=50, base_seed=1, progress=False)
    for run in result.runs:
        assert run.error is None, run.error
        assert run.n_evals == run.budget == 100


def test_a_run_that_ends_early_is_marked():
    from panobbgo.benchmark import StrategySpec
    from panobbgo.core import Heuristic
    from panobbgo.harness_ioh import EARLY_END_ERROR_PREFIX
    from panobbgo.strategies import StrategyRoundRobin

    class ThreePoints(Heuristic):
        def __init__(self, strategy):
            super().__init__(strategy, name="ThreePoints")

        def on_start(self):
            for _ in range(3):
                self._put(Point(self.problem.random_point(rng=self.rng), self.name))

    spec = StrategySpec(name="ThreePoints", strategy_class=StrategyRoundRobin, heuristics=[(ThreePoints, {})])
    inst = [("sphere_d2_i0", Family("sphere", dim=2, seed=1))]
    result = run_family_harness([spec], inst, budget_multiplier=20, base_seed=1, progress=False)
    run = result.runs[0]
    assert run.n_evals == 3 < run.budget
    assert run.error is not None and run.error.startswith(EARLY_END_ERROR_PREFIX)
    assert run.ended_early and not run.crashed and not run.timed_out
    assert result.per_strategy_counts()["ThreePoints"]["ended_early"] == 1


@pytest.mark.parametrize("shape", ["ball", "boxes"])
def test_unplaceable_regions_fall_back_deterministically(shape):
    """A ball of 20 % of the box cannot avoid a central optimum at d = 10: fallback, not an error."""
    region = FailureRegion(shape, share=0.2)
    p = Family("sphere", dim=10, seed=3, shift=False, failure=region)
    q = Family("sphere", dim=10, seed=3, shift=False, failure=region)
    assert p._failure_geom.fallback
    assert p.failure_at(p.x_opt) is None and p.eval(p.x_opt) == p.f_opt
    assert 0.0 < p.failure_share <= 0.2 + 0.01
    assert p.failure_share == q.failure_share
    assert _mc_share(p, n=5000) == pytest.approx(p.failure_share, abs=0.02)
    battery = make_family_instances([FamilyConfig(base="sphere", shift=False, failure=region)], dims=(10,))
    assert len(battery) == 3  # the battery builds
    if shape == "ball":  # the ray search reaches the target where a half-way centre gave 4 %
        assert p.failure_share == pytest.approx(0.2, abs=0.02)
    hs = Family("sphere", dim=10, seed=3, shift=False, failure=FailureRegion("halfspace", share=0.2))
    assert hs._failure_geom.fallback is False


def test_failure_share_is_measured_out_of_sample():
    p = Family("sphere", dim=5, seed=4, failure=FailureRegion("ball", share=0.1))
    calib = p._failure_rng().uniform(-5.0, 5.0, size=(20000, 5))  # the calibration sample: first draw
    in_sample = float(np.mean(p._failure_geom.contains_many(calib)))
    assert p.failure_share != in_sample  # a different sample ...
    assert p.failure_share == pytest.approx(0.1, abs=0.01)  # ... that still measures the target


def test_a_timeout_abandoned_in_flight_is_not_an_early_end(monkeypatch):
    """Threads + ``evaluation.timeout``: an abandoned call is never recorded, but its slot was spent.

    Two runs back to back: calls abandoned by the first are still running
    when the second starts, and must not write into the first (finished,
    scored) run's tracker.
    """
    import time

    from panobbgo.benchmark import StrategySpec
    from panobbgo.harness_ioh import _early_end_error
    from panobbgo.heuristics import Random
    from panobbgo.strategies import StrategyRoundRobin

    assert _early_end_error(24, 30, 30) is None
    assert _early_end_error(8, 8, 30) is not None

    in_flight = _InFlight()

    class Slow(Family):
        def eval(self, x):
            with in_flight:
                if x[0] > 2.0:
                    time.sleep(0.3)
                return super().eval(x)

    spec = StrategySpec(
        name="R",
        strategy_class=StrategyRoundRobin,
        heuristics=[(Random, {})],
        config_overrides={"evaluation_timeout": 0.05},
    )
    late_writes = []
    real_record = PenaltyTracker._record

    def record(self, x, measured):
        if getattr(self, "_closed", False):
            late_writes.append(float(x[0]))
        real_record(self, x, measured)

    monkeypatch.setattr(PenaltyTracker, "_record", record)
    instances = [("slow1", Slow("sphere", dim=2, seed=1)), ("slow2", Slow("sphere", dim=2, seed=2))]
    try:
        res = run_family_harness([spec], instances, budget_multiplier=15, progress=False)
    finally:
        in_flight.drain()  # abandoned calls must not run on into the next tests
    assert len(res.runs) == 2
    for run in res.runs:
        assert run.n_evals < run.budget  # abandoned calls never reached the trace ...
        assert run.error is None  # ... but the run spent its budget
    assert late_writes == []  # nothing recorded into a finished run


class _InFlight:
    """Counts the calls inside a ``with`` block; :meth:`drain` waits (bounded) until none is left."""

    def __init__(self) -> None:
        import threading

        self._cv = threading.Condition()
        self.n = 0
        self.peak = 0

    def __enter__(self) -> "_InFlight":
        with self._cv:
            self.n += 1
            self.peak = max(self.peak, self.n)
        return self

    def __exit__(self, *exc) -> None:
        with self._cv:
            self.n -= 1
            self._cv.notify_all()

    def drain(self, timeout: float = 10.0) -> None:
        with self._cv:
            assert self._cv.wait_for(lambda: self.n == 0, timeout), f"{self.n} call(s) still running"


def test_a_call_that_finishes_after_the_run_is_not_recorded():
    """A call abandoned in flight that returns after ``restore()`` leaves the finished run's tracker untouched.

    Threads abandoned by ``evaluation.timeout`` cannot be killed; they
    finish after the harness has scored the run (and while the next run or
    test is going).  The tracker is closed at ``restore()``: the late
    result is dropped, a late call is not admitted, and the problem is not
    called for it.
    """
    import threading

    entered, release = threading.Event(), threading.Event()
    calls = []

    class Held(Family):
        def eval(self, x):
            calls.append(float(x[0]))
            if x[0] > 2.0:
                entered.set()
                assert release.wait(10)
            return super().eval(x)

    p = Held("sphere", dim=2, seed=1, shift=False)
    tracker = PenaltyTracker(p, budget=10)
    bound = p.eval  # what a pool thread holds while its call is in flight
    bound(np.full(2, 1.0))
    late = threading.Thread(target=bound, args=(np.full(2, 3.0),))
    late.start()
    try:
        assert entered.wait(10)
        tracker.restore()  # the run ends while the call is in flight
        frozen = (tracker.n_evals, list(tracker.best_so_far), tracker.best_fx, tracker._reserved)
    finally:
        release.set()
        late.join(10)
    assert not late.is_alive()
    assert (tracker.n_evals, list(tracker.best_so_far), tracker.best_fx, tracker._reserved) == frozen
    assert frozen[0] == 1 and frozen[3] == 2  # one recorded, one admitted in flight
    assert bound(np.zeros(2)) == tracker.best_fx  # a late call: not admitted ...
    assert len(calls) == 2  # ... and the problem was not called
    assert tracker.n_evals == 1
