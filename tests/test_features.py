# -*- coding: utf8 -*-
# Copyright 2012-2026 Panobbgo Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Feature logging at budget checkpoints (``panobbgo.features``, roadmap §3.4)."""

import json
import math

import numpy as np
import pytest

from panobbgo.features import (
    FeatureLogSpec,
    compute_features,
    landscape_features,
    rank_order,
    spearman,
)
from panobbgo.harness_families import (
    PenaltyTracker,
    make_families_battery,
    make_failure_battery,
    make_sealed_families_battery,
    run_family_harness,
)
from panobbgo.harness_ioh import (
    IOHHarnessResult,
    _run_tracked,
    make_ioh_strategies,
    make_sealed_battery,
    run_ioh_harness,
)
from panobbgo.virtual_clock import VirtualSpec

# Rotation-invariant landscape keys (with equal box ranges).  ``sep_ratio``
# (and ``r2_add``) are deliberately not; the coverage keys depend on the
# fixed probe set and the box, not on the problem's orientation.
ROTATION_INVARIANT = (
    "fdc",
    "nbc_mean_ratio",
    "nbc_sd_ratio",
    "nbc_rank_cor",
    "disp_10",
    "disp_25",
    "r2_lin",
    "r2_quad",
    "log10_cond",
    "hess_pos",
)


def _spec(name):
    return next(s for s in make_ioh_strategies() if s.name == name).with_regime_class("clean")


def _same(a, b, rel=0.0):
    """Equal dicts of floats, NaN == NaN, nested."""
    assert a.keys() == b.keys()
    for k in a:
        if isinstance(a[k], dict):
            _same(a[k], b[k], rel)
        elif a[k] is None or (isinstance(a[k], float) and math.isnan(a[k])):
            assert b[k] is None or (isinstance(b[k], float) and math.isnan(b[k])), k
        else:
            assert b[k] == pytest.approx(a[k], rel=rel, abs=rel), k


def _cloud(seed=3, d=4, n=400):
    rng = np.random.default_rng(seed)
    y = rng.uniform(-3, 3, (n, d))
    q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    who = np.array(["CMAES:g1", "JSO:ab12", "Random"])[rng.integers(0, 3, n)]
    return y, q, who, np.array([[-5.0, 5.0]] * d)


def _ellipsoid(z, cond=100.0):
    w = cond ** np.linspace(0, 1, z.shape[1])
    return (w * z**2).sum(1)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def test_rank_order_ties_constraints_and_spearman():
    assert rank_order(np.array([3.0, 1.0, 1.0, 2.0])).tolist() == [3, 0, 1, 2]
    # feasible first by f, then infeasible by violation
    fx = np.array([0.0, 5.0, -1.0, 2.0])
    cv = np.array([1.0, 0.0, 0.5, 0.0])
    assert rank_order(fx, cv).tolist() == [3, 1, 2, 0]
    assert spearman(np.arange(5.0), np.arange(5.0) ** 3) == pytest.approx(1.0)
    assert math.isnan(spearman(np.arange(2.0), np.arange(2.0)))


def test_spec_validation_and_parse():
    assert FeatureLogSpec.parse(None).checkpoints == (0.05, 0.1, 0.2, 0.4, 0.7)
    assert FeatureLogSpec.parse("0.1, 0.5,1").checkpoints == (0.1, 0.5, 1.0)
    for bad in ((), (0.5, 0.2), (0.0,), (1.2,), (0.2, 0.2)):
        with pytest.raises(ValueError):
            FeatureLogSpec(checkpoints=bad)


def test_small_archives_give_undefined_not_errors():
    box = np.array([[0.0, 1.0]] * 3)
    for n in (0, 1, 3):
        x = np.full((n, 3), 0.5)
        feats = compute_features(x, np.arange(n, dtype=float), ["A"] * n, box)
        assert math.isnan(feats["land"]["fdc"])
    # the quadratic fit needs more points than coefficients (1 + 2d + d(d-1)/2 = 10 at d = 3)
    rng = np.random.default_rng(0)
    u = rng.random((10, 3))
    land = landscape_features(u, rank_order(_ellipsoid(u - 0.5)))
    assert math.isnan(land["r2_quad"]) and math.isnan(land["log10_cond"])
    assert math.isfinite(land["r2_lin"])


# ---------------------------------------------------------------------------
# invariances
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "transform",
    [
        lambda f: 3.7 * f - 11.0,  # affine
        lambda f: np.log1p(f),  # monotone
        lambda f: f**3 + f,  # monotone
    ],
)
def test_rank_based_features_invariant_to_monotone_f_transforms(transform):
    y, _q, who, box = _cloud()
    f = _ellipsoid(y)
    base = compute_features(y, f, who, box)
    moved = compute_features(y, transform(f), who, box)
    # Everything is rank-based: bit-identical, meta-model R² included.
    _same(base["land"], moved["land"])
    _same(base["traj"], moved["traj"])
    _same(base["arms"], moved["arms"])


def test_rotation_invariant_features_and_separability():
    y, q, who, box = _cloud()
    base = compute_features(y, _ellipsoid(y), who, box)
    # the same landscape, rotated about the (centred) box: x = Q y, f(x) = f_sep(Qᵀ x)
    xr = y @ q.T
    rot = compute_features(xr, _ellipsoid(xr @ q), who, box)
    for k in ROTATION_INVARIANT:
        assert rot["land"][k] == pytest.approx(base["land"][k], rel=1e-7, abs=1e-9), k
    for arm in base["arms"]:
        assert rot["arms"][arm]["revisit"] == base["arms"][arm]["revisit"]
    # The axis-aligned ellipsoid is additive; rotated it is not.
    assert base["land"]["sep_ratio"] == pytest.approx(1.0, abs=0.02)
    assert rot["land"]["sep_ratio"] < 0.7
    # The Hessian condition estimate sees the conditioning (100) either way.
    assert base["land"]["log10_cond"] == pytest.approx(2.0, abs=0.3)


def test_stuck_arm_reads_as_stuck():
    """A contracting arm resampling one point: small spread, contracting, low novelty, revisits."""
    rng = np.random.default_rng(1)
    d = 3
    explore = rng.random((60, d))
    scales = np.r_[np.full(50, 1e-2), np.full(10, 0.0)]  # the last 10 (its recent window) collapse
    stuck = 0.3 + scales[:, None] * rng.standard_normal((60, d))
    x = np.vstack([explore, stuck])
    who = ["Random"] * 60 + ["CMAES"] * 60
    fx = np.r_[rng.random(60) + 1.0, np.full(60, 0.5)]
    feats = compute_features(x, fx, who, np.array([[0.0, 1.0]] * d))
    cma, rnd = feats["arms"]["CMAES"], feats["arms"]["Random"]
    assert cma["revisit"] == 0.9 and rnd["revisit"] == 0.0  # the first collapsed point is new
    assert cma["spread"] < 1e-6 < rnd["spread"]
    assert cma["spread_trend"] < -3
    assert cma["share"] == rnd["share"] == 0.5
    assert feats["traj"]["stall_per_d"] > 10


# ---------------------------------------------------------------------------
# in a run
# ---------------------------------------------------------------------------


def _tracked(spec, problem, budget, log, virtual=None, seed=7):
    tracker = PenaltyTracker(problem, budget=budget)
    out = _run_tracked(
        spec,
        problem,
        tracker,
        f_opt=float(problem.f_opt),
        budget=budget,
        seed=seed,
        sync_eval=True,
        log_lo=-8.0,
        log_hi=2.0,
        timeout_s=None,
        virtual=virtual,
        log_features=log,
    )
    return tracker, out


@pytest.mark.parametrize(
    "name,battery,virtual",
    [
        ("Blocks_warm_CMAES_JSO", make_families_battery, None),
        ("RoundRobin_CMAES", make_failure_battery, None),
        ("Blocks_warm_CMAES_JSO", make_failure_battery, VirtualSpec(workers=4, duration="lognormal")),
    ],
)
def test_logging_leaves_the_run_bit_identical(name, battery, virtual):
    """Same evaluations, same RNG consumption: the whole trace, incumbent and count are identical."""
    _n, problem = battery(dims=(5,), n_instances=1)[0]
    spec = _spec(name)
    t_off, off = _tracked(spec, problem, 100 * 5, None, virtual)
    t_on, on = _tracked(spec, problem, 100 * 5, FeatureLogSpec(), virtual)
    assert t_on.best_so_far == t_off.best_so_far
    assert np.array_equal(np.array(t_on.timeline), np.array(t_off.timeline), equal_nan=True)
    assert np.array_equal(t_on.best_x, t_off.best_x)
    assert (on.n_evals, on.aocc, on.aocc_time) == (off.n_evals, off.aocc, off.aocc_time)
    assert off.features is None and on.features is not None
    assert t_on.best_x is not None
    assert [r["checkpoint"] for r in on.features] == [0.05, 0.1, 0.2, 0.4, 0.7]
    evals = [r["ctx"]["evals"] for r in on.features]
    assert evals == sorted(evals) and all(e >= math.ceil(c * 500) for e, c in zip(evals, (0.05, 0.1, 0.2, 0.4, 0.7)))
    last = on.features[-1]
    assert last["ctx"]["q"] == (1 if virtual is None else 4)
    assert set(last) == {"checkpoint", "ctx", "land", "traj", "arms"}
    if problem.failure is not None:
        assert last["traj"]["fail_share"] > 0
    if name.startswith("Blocks"):
        assert {"CMAES", "JSO"} <= set(last["arms"])
    cma = last["arms"]["CMAES"]
    assert "sigma_rel" in cma and "log10_cond_c" in cma  # the CMA-ES extras
    json.dumps(on.features, allow_nan=False)  # compact and strict JSON: no NaN, no numpy scalars


def test_logging_is_deterministic_and_off_by_default():
    spec = [_spec("RoundRobin_CMAES")]
    inst = make_families_battery(dims=(3,), n_instances=1)[:2]
    plain = run_family_harness(spec, inst, budget_multiplier=60, progress=False)
    a = run_family_harness(spec, inst, budget_multiplier=60, progress=False, log_features=FeatureLogSpec())
    b = run_family_harness(spec, inst, budget_multiplier=60, progress=False, log_features=FeatureLogSpec())
    assert [r.features for r in a.runs] == [r.features for r in b.runs]
    assert a.runs[0].features is not None
    assert [r.trace_fx for r in a.runs] == [r.trace_fx for r in plain.runs]
    # Default result files keep their shape: no ``features`` key at all.
    assert all("features" not in row for row in plain.to_dict()["runs"])
    rows = a.to_dict()["runs"]
    assert all(len(row["features"]) == 5 for row in rows)
    back = IOHHarnessResult.from_dict(json.loads(a.to_json()))
    assert [r.features for r in back.runs] == [r.features for r in a.runs]
    for rec in a.runs[0].features:
        text = json.dumps(rec)
        assert len(text) < 2000  # compact


def test_sealed_sets_refuse_feature_logging():
    spec = [_spec("RoundRobin_Random")]
    with pytest.raises(ValueError, match="feature logging is refused"):
        run_family_harness(spec, make_sealed_families_battery(), progress=False, log_features=FeatureLogSpec())
    with pytest.raises(ValueError, match="feature logging is refused"):
        run_ioh_harness(spec, make_sealed_battery(), progress=False, log_features=FeatureLogSpec())
    cli = _cli()
    for flag in ("--sealed", "--families-sealed"):
        with pytest.raises(SystemExit, match="--log-features"):
            cli.main(["run", flag, "--log-features", "--quiet"])


def _cli():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "scripts" / "ioh_benchmark.py"
    spec = importlib.util.spec_from_file_location("ioh_benchmark_cli_features", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cli_log_features_on_the_constrained_realworld_track(tmp_path):
    out = tmp_path / "rw.json"
    argv = ["run", "--realworld-quick", "--strategies", "RoundRobin_Random", "--quiet", "--output", str(out)]
    assert _cli().main([*argv, "--log-features", "0.25,0.5,1.0"]) == 0
    rows = json.loads(out.read_text())["runs"]
    for row in rows:
        assert [f["checkpoint"] for f in row["features"]] == [0.25, 0.5, 1.0]
        assert all(f["ctx"]["constrained"] for f in row["features"])
        assert row["features"][-1]["ctx"]["evals"] == row["budget"]
    plain = tmp_path / "plain.json"
    assert _cli().main([*argv[:-1], str(plain)]) == 0
    assert all("features" not in row for row in json.loads(plain.read_text())["runs"])


@pytest.mark.flaky(retries=3)
def test_logging_overhead_is_small_at_d10():
    """Feature computation stays below ~10 % of the run at d = 10 (500·d, the cheapest objective)."""
    _n, problem = make_families_battery(dims=(10,), n_instances=1)[0]
    t0 = __import__("time").perf_counter()
    _tracker, out = _tracked(_spec("Blocks_warm_CMAES_JSO"), problem, 500 * 10, FeatureLogSpec())
    total = __import__("time").perf_counter() - t0
    assert out.features is not None and len(out.features) == 5
    assert out.features_s < 0.10 * total, (out.features_s, total)
