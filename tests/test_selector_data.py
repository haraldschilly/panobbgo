# -*- coding: utf8 -*-
# Copyright 2012-2026 Panobbgo Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Selector training data (``panobbgo.selector_data``, roadmap §4 A step 1, DISCOVERY §70)."""

import math

import numpy as np
import pytest

from panobbgo.benchmark import StrategySpec
from panobbgo.features import fscale_features
from panobbgo.lib.families import Family
from panobbgo.selector_data import (
    ARM_MENU_V0,
    DIAGNOSTIC_ARMS,
    PROBE_WHO,
    add_labels,
    arm_menu_v0,
    continue_from_probe,
    evaluate_probe,
    probe_design,
    probe_features,
    run_task,
)
from panobbgo.virtual_clock import VirtualSpec

# ---------------------------------------------------------------------------
# features: invariance
# ---------------------------------------------------------------------------

#: Invariant to f -> a·f + b (a > 0): every feature.
#: Invariant to rotation / shift / uniform scaling of x: all but these.
NOT_ROTATION_INVARIANT = {"r2_add", "sep_ratio", "fr2_add", "fsep_ratio"}
#: f-scale features: not invariant to a non-affine monotone transform of f.
FSCALE = {
    "fr2_lin",
    "fr2_add",
    "fr2_quad",
    "flog_quad_gap",
    "fsep_ratio",
    "flog10_cond",
    "fhess_pos",
    "y_skew",
    "y_kurt",
}


def _sample(d=4, n=60, seed=0, fn="rotated"):
    rng = np.random.default_rng(seed)
    u = rng.random((n, d))
    if fn == "rotated":  # a rotated, conditioned quadratic plus a ripple: nothing is exactly zero or one
        q, _ = np.linalg.qr(rng.normal(size=(d, d)))
        z = (u - 0.3) @ q.T
        f = (np.logspace(0, 2, d) * z**2).sum(axis=1) + 0.3 * np.sin(7 * u).sum(axis=1)
    elif fn == "additive":  # a separable, conditioned quadratic: additive in u, not in rotated coordinates
        f = (np.logspace(0, 2, d) * (u - 0.5) ** 2).sum(axis=1)
    else:
        raise ValueError(fn)
    return u, f


def _feats(u, f):
    out = probe_features(u, f, dim=u.shape[1], budget=100 * u.shape[1], q=1)
    return {k: v for k, v in out.items() if k not in ("dim", "budget_per_d", "probe_per_d", "remaining_per_d", "q")}


def _assert_close(a, b, keys, rtol=1e-7, atol=1e-9):
    for k in keys:
        va, vb = a[k], b[k]
        assert (va is None) == (vb is None), k
        if va is not None:
            assert math.isclose(va, vb, rel_tol=rtol, abs_tol=atol), (k, va, vb)


def test_probe_features_defined():
    u, f = _sample()
    feats = _feats(u, f)
    missing = [k for k, v in feats.items() if v is None]
    assert missing == [], missing  # d = 4, n = 60 >= 2p = 30: every feature is defined


@pytest.mark.parametrize("a,b", [(1e-3, 5.0), (7.5e4, -3e6), (1.0, 1e3)])
def test_features_invariant_to_affine_f(a, b):
    u, f = _sample()
    base = _feats(u, f)
    other = _feats(u, a * f + b)
    _assert_close(base, other, base.keys(), rtol=1e-6)


def test_rank_features_invariant_to_monotone_f_but_fscale_ones_are_not():
    u, f = _sample()
    base = _feats(u, f)
    other = _feats(u, np.exp(f / f.std()))
    rank_keys = [k for k in base if k not in FSCALE]
    _assert_close(base, other, rank_keys, rtol=1e-9)
    # control: the f-scale R² does see the transform (the test would catch a rank leak)
    assert abs(base["fr2_quad"] - other["fr2_quad"]) > 1e-3


@pytest.mark.parametrize("seed", [1, 2])
def test_features_invariant_to_rotation_shift_scale_of_x(seed):
    u, f = _sample(seed=seed)
    rng = np.random.default_rng(100 + seed)
    q, _ = np.linalg.qr(rng.normal(size=(u.shape[1], u.shape[1])))
    u2 = 2.5 * (u @ q.T) + rng.normal(size=u.shape[1])
    base, other = _feats(u, f), _feats(u2, f)
    _assert_close(base, other, [k for k in base if k not in NOT_ROTATION_INVARIANT], rtol=1e-6, atol=1e-8)


def test_separability_features_see_a_rotation():
    """Control for the rotation test: the features claimed *not* rotation-invariant do change."""
    u, f = _sample(fn="additive")
    rng = np.random.default_rng(5)
    q, _ = np.linalg.qr(rng.normal(size=(u.shape[1], u.shape[1])))
    base, other = _feats(u, f), _feats(u @ q.T, f)
    assert abs(base["fr2_add"] - other["fr2_add"]) > 0.05
    assert abs(base["r2_add"] - other["r2_add"]) > 0.05


def test_fscale_r2_separates_an_exact_quadratic_and_rank_r2_does_not():
    """DISCOVERY §66.4: 1 - R² is 0 to rounding on an exact quadratic only when fitted to f, not to ranks."""
    for d in (2, 5):
        u = probe_design(d, 10 * d, seed=d)
        rng = np.random.default_rng(d)
        q, _ = np.linalg.qr(rng.normal(size=(d, d)))
        z = (u - 0.4) @ q.T
        f = (np.logspace(0, 6, d) * z**2).sum(axis=1)
        feats = _feats(u, f)
        assert feats["flog_quad_gap"] < -9
        assert feats["r2_quad"] < 0.99
        # a non-quadratic (the same plus a ripple) is far from it
        g = f + 0.05 * f.std() * np.sin(9 * u).sum(axis=1)
        assert _feats(u, g)["flog_quad_gap"] > -5


def test_fscale_features_edge_cases():
    out = fscale_features(np.zeros((3, 2)), np.ones(3))
    assert all(math.isnan(v) for v in out.values())
    u = np.random.default_rng(0).random((10, 2))
    out = fscale_features(u, np.ones(10))  # constant f: only the tie share is defined
    assert out["y_ties"] == pytest.approx(0.9)
    assert math.isnan(out["fr2_lin"])
    fx = np.r_[np.nan, np.arange(9.0)]  # a failed call is dropped
    assert fscale_features(u, fx)["y_ties"] == 0.0


# ---------------------------------------------------------------------------
# probe and menu
# ---------------------------------------------------------------------------


def test_probe_design_is_a_latin_hypercube():
    u = probe_design(3, 20, seed=11)
    assert u.shape == (20, 3)
    assert np.all((u >= 0) & (u <= 1))
    for j in range(3):  # one point per stratum on every axis
        assert sorted(np.floor(u[:, j] * 20).astype(int)) == list(range(20))
    assert np.array_equal(u, probe_design(3, 20, seed=11))
    assert not np.array_equal(u, probe_design(3, 20, seed=12))


def test_arm_menu_v0_is_the_continuation_form():
    specs = arm_menu_v0()
    assert tuple(s.name for s in specs) == ARM_MENU_V0
    by = {s.name: s for s in specs}
    ((cls, kw),) = by["RoundRobin_CMAES"].heuristics
    assert cls.__name__ == "CMAES" and kw == {"warm_start": "archive"}
    assert [a.__name__ for a, _ in by["RoundRobin_CMAES"].analyzers] == ["Archive"]
    ((cls, kw),) = by["RoundRobin_COBYQA"].heuristics
    assert cls.__name__ == "COBYQA" and kw == {"warm_start": "archive"}
    assert by["Blocks_warm_CMAES_JSO_TRQ"].rng_identity == "Blocks_warm_CMAES_JSO"
    assert [s.name for s in arm_menu_v0(["RoundRobin_TRQ"])] == ["RoundRobin_TRQ"]
    cold = {s.name: s for s in arm_menu_v0(DIAGNOSTIC_ARMS)}
    assert cold["RoundRobin_CMAES_cold"].heuristics[0][1] == {}
    assert cold["RoundRobin_CMAES_cold"].rng_identity == "RoundRobin_CMAES"
    assert cold["RoundRobin_COBYQA_cold"].heuristics[0][1] == {}
    assert cold["RoundRobin_COBYQA_cold"].rng_identity == "RoundRobin_COBYQA"
    with pytest.raises(ValueError):
        arm_menu_v0(["nope"])


def test_add_labels():
    row = {"q": 1}
    for a, s, lp in zip(ARM_MENU_V0, [0.2, 0.5, 0.5, 0.1, 0.4], [-1.0, -3.0, -4.0, 0.0, -2.0]):
        row[f"{a}:aocc"] = s
        row[f"{a}:final_logp"] = lp
    add_labels(row)
    assert row["best_arm"] == "RoundRobin_CMAES"  # first of the tie in menu order
    assert row["RoundRobin_TRQ:regret"] == 0.0
    assert row["RoundRobin_COBYQA:regret"] == pytest.approx(0.4)
    assert row["RoundRobin_TRQ:regret_final"] == 0.0
    assert row["RoundRobin_COBYQA:regret_final"] == pytest.approx(0.4)
    assert row["RoundRobin_CMAES:rank"] == 1.5
    # q > 1 labels the time score; a missing score leaves the task unlabelled
    row2 = {"q": 4, **{f"{a}:aocc_time": 0.3 for a in ARM_MENU_V0}, **{f"{a}:final_logp": 0.0 for a in ARM_MENU_V0}}
    row2[f"{ARM_MENU_V0[0]}:aocc_time"] = None
    assert add_labels(row2)["best_arm"] is None


# ---------------------------------------------------------------------------
# continuation from a shared probe
# ---------------------------------------------------------------------------


def _problem(d=2):
    return Family("ellipsoid", dim=d, seed=3)


def _probe(problem, k, seed=0):
    return evaluate_probe(problem, probe_design(problem.dim, k, seed))


def test_preloaded_probe_counts_against_the_budget():
    from panobbgo.harness_ioh import make_ioh_strategies

    p = _problem()
    probe = _probe(p, 20)
    spec = next(s for s in make_ioh_strategies() if s.name == "RoundRobin_Random")
    res = continue_from_probe(spec, p, probe, budget=60, seed=1, virtual=VirtualSpec(workers=4, duration="lognormal"))
    assert res["n_evals"] == 40 and res["error"] is None
    assert 0.0 <= res["aocc"] <= 1.0 and res["aocc_time"] is not None
    # the score starts at the probe's best: never worse than holding it
    from panobbgo.ioh_runner import aocc

    hold = aocc([min(float(r.fx) for r in probe)], f_opt=p.f_opt, budget=40)
    assert res["aocc"] >= hold - 1e-12


def test_preload_results_books_before_start():
    p = _problem()
    probe = _probe(p, 12)
    spec = next(s for s in arm_menu_v0(["RoundRobin_CMAES"]))
    strat = spec.create_strategy(p, seed=5, max_eval=30)
    strat.config.sync_evaluation = True
    strat.config.stop_on_convergence = False
    strat.preload_results(probe)
    strat.start()
    hist = strat.results.get_history()
    assert len(hist["fx"]) == 30
    assert list(hist["who"][:12]) == [PROBE_WHO] * 12
    assert all(w != PROBE_WHO for w in hist["who"][12:])
    with pytest.raises(TypeError):
        strat.preload_results([1.0])


def _cmaes_mean(preload):
    p = _problem(3)
    spec = arm_menu_v0(["RoundRobin_CMAES"])[0]
    strat = spec.create_strategy(p, seed=5, max_eval=100)
    strat.config.sync_evaluation = True
    probe = _probe(p, 30, seed=2)
    if preload:
        strat.preload_results(probe)
    try:
        strat.initialize()
        m = np.array(strat.heuristic("CMAES")._m, dtype=float)
    finally:
        strat._cleanup()
    return m, probe, p


def test_cmaes_continuation_warm_starts_from_the_probe():
    m_cold, _, p = _cmaes_mean(False)
    assert np.allclose(m_cold, p.box.box.mean(axis=1))  # empty archive: the box centre
    m_warm, probe, p = _cmaes_mean(True)
    best = sorted(probe, key=lambda r: float(r.fx))[:3]
    xs = np.array([r.x for r in best])
    assert not np.allclose(m_warm, m_cold)
    # the recombined mean lies in the hull of the best probe points' box
    assert np.all(m_warm >= xs.min(axis=0) - 1e-9) and np.all(m_warm <= xs.max(axis=0) + 1e-9)


def _cobyqa_first_point(cobyqa_kwargs):
    """The first point a COBYQA-alone run evaluates after a preloaded 10-point probe, and the best probe point."""
    from panobbgo.heuristics import COBYQA
    from panobbgo.strategies import StrategyRoundRobin

    p = _problem()
    probe = _probe(p, 10, seed=4)
    spec = StrategySpec(
        name="RoundRobin_COBYQA", strategy_class=StrategyRoundRobin, heuristics=[(COBYQA, cobyqa_kwargs)]
    )
    strat = spec.create_strategy(p, seed=1, max_eval=14)
    strat.config.sync_evaluation = True
    strat.config.stop_on_convergence = False
    strat.preload_results(probe)
    strat.start()
    hist = strat.results.get_history()
    first = np.asarray(hist["x"], dtype=float).reshape(len(hist["fx"]), 2)[10]
    return first, np.asarray(min(probe, key=lambda r: float(r.fx)).x, dtype=float), p


def test_cobyqa_warm_start_starts_at_the_best_probe_point():
    # A small, unscaled start radius, so SciPy keeps x0 as given (see the next test).
    first, best, p = _cobyqa_first_point({"warm_start": "archive", "initial_tr_radius": 0.05, "scale": False})
    box = p.box.box
    assert np.all(best - box[:, 0] > 0.05) and np.all(box[:, 1] - best > 0.05)
    assert np.allclose(first, best)
    cold, _, _ = _cobyqa_first_point({"initial_tr_radius": 0.05, "scale": False})
    assert np.allclose(cold, box.mean(axis=1))


def test_cobyqa_warm_start_with_the_default_radius_snaps_to_the_start_grid():
    """SciPy's COBYQA moves x0 within rhobeg of a bound onto the bound (or to bound ± rhobeg).

    With the default radius (1 in the scaled box [-1, 1]) that quantises the
    start to {-1, 0, +1} per scaled axis: the warm start keeps only the
    region of the best probe point (DISCOVERY §70.1).
    """
    first, best, p = _cobyqa_first_point({"warm_start": "archive"})
    lo, hi = p.box.box[:, 0], p.box.box[:, 1]
    s = (best - 0.5 * (lo + hi)) / (0.5 * (hi - lo))
    grid = np.where(s <= -0.5, -1.0, np.where(s >= 0.5, 1.0, 0.0))
    assert np.allclose(first, 0.5 * (lo + hi) + grid * 0.5 * (hi - lo))


def test_cobyqa_rejects_an_unknown_warm_start():
    from panobbgo.heuristics import COBYQA
    from panobbgo.strategies import StrategyRoundRobin

    with pytest.raises(ValueError):
        StrategyRoundRobin(_problem(), testing_mode=True).add(COBYQA, warm_start="best")


def test_run_task_is_deterministic_and_labelled():
    p = _problem()
    arms = ["RoundRobin_CMAES", "RoundRobin_TRQ"]
    a = run_task(p, base_seed=7, budget_multiplier=20, q=4, arms=arms)
    b = run_task(p, base_seed=7, budget_multiplier=20, q=4, arms=arms)
    drop = [k for k in a if k.endswith(":elapsed_s")]
    assert {k: v for k, v in a.items() if k not in drop} == {k: v for k, v in b.items() if k not in drop}
    assert a["k"] == 20 and a["budget"] == 40
    for arm in arms:
        assert a[f"{arm}:n_evals"] == 20
    assert a["f_probe_per_d"] == 10.0 and a["f_q"] == 4
    add_labels(a, arms)
    assert min(a[f"{arm}:regret"] for arm in arms) == 0.0


def test_selector_labels_analyze_runs_on_labelled_rows():
    """``benchmarks/selector_labels.py analyze``: the headroom table on synthetic rows (SBS regret = gap)."""
    import importlib.util
    from pathlib import Path

    import pandas as pd

    path = Path(__file__).resolve().parent.parent / "benchmarks" / "selector_labels.py"
    spec = importlib.util.spec_from_file_location("selector_labels", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rng = np.random.default_rng(0)
    rows = []
    for fam in ("a", "b", "ellipsoid"):
        for inst in range(2):
            for seed in (1, 2):
                for q in (1, 4):
                    row = {"family": fam, "dim": 2, "instance": inst, "seed": seed, "q": q, "f_x": rng.random()}
                    for arm in ARM_MENU_V0:
                        row[f"{arm}:aocc"] = row[f"{arm}:aocc_time"] = rng.random()
                        row[f"{arm}:final_logp"] = rng.random()
                        row[f"{arm}:error"] = None
                    rows.append(add_labels(row))
    df = pd.DataFrame(rows)
    text = mod.analyze(df, ARM_MENU_V0)
    s = df[[f"{arm}:score" for arm in ARM_MENU_V0]].to_numpy()
    gap = (s.max(axis=1) - s[:, s.mean(axis=0).argmax()]).mean()
    assert f"**{gap:.3f}**" in text
