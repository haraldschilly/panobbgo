# -*- coding: utf8 -*-
# Copyright 2012 -- 2026 Harald Schilly <harald.schilly@gmail.com>
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

"""The real-world set (``panobbgo.lib.realworld``) and its AOCC harness (``panobbgo.harness_realworld``)."""

import json
import pickle

import numpy as np
import pytest

from panobbgo.harness_ioh import make_ioh_strategies
from panobbgo.harness_realworld import (
    QUICK_PROBLEMS,
    FeasibleGapTracker,
    make_realworld_battery,
    make_realworld_quick_battery,
    run_realworld_harness,
)
from panobbgo.lib import EvaluationCrashed, Point
from panobbgo.lib.realworld import EQ_TOL, REALWORLD_SPECS, RealWorldProblem, make_realworld_instances

SPECS = list(REALWORLD_SPECS.values())
IDS = [s.name for s in SPECS]


# ---------------------------------------------------------------------------
# Reference values and the published optimum
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_reference_point_value_feasibility_and_best_known(spec):
    """f(x_ref) is f_ref, x_ref is feasible (CEC 2020 rule) and f_ref matches the best-known value."""
    p = RealWorldProblem(spec)
    x = np.asarray(spec.x_ref, dtype=np.float64)
    assert p.dim == spec.dim == len(spec.upper)
    assert np.all(x >= p.box[:, 0]) and np.all(x <= p.box[:, 1])
    f = p.eval(x)
    assert f == pytest.approx(spec.f_ref, rel=1e-12, abs=0.0)
    assert p.is_feasible(x)
    assert p.violation(x) == 0.0
    assert abs(f - spec.f_best) / abs(spec.f_best) <= spec.best_rtol
    assert p.relative_gap(f) == pytest.approx((f - spec.f_best) / abs(spec.f_best))


#: Points published for the classic formulations, with their published value
#: and the relative tolerance the published digits allow.
LITERATURE_POINTS = [
    # Haverly (1978) pooling optimum: all flow through the pool to product Y.
    ("rc05_haverly_pooling", (0, 200, 0, 100, 0, 100, 0, 100, 1), -400.0, 1e-15),
    # CEC 2006 g04 (Liang et al. 2006): the published optimum, full digits.
    ("rc32_himmelblau", (78, 33, 29.9952560256815985, 45, 36.7758129057882073), -30665.5386717834, 1e-12),
    # Integer pressure vessel optimum: z = (0.8125, 0.4375), R = 42.098446, L = 176.636596.
    ("rc18_pressure_vessel", (13, 7, 42.0984455958549, 176.6365958424394), 6059.714335, 1e-9),
    # Commonly reported optima of the spring, the three-bar truss, the gas compressor
    # and the flow-sheeting problem, to the digits they are usually printed with.
    ("rc17_spring", (0.051689061, 0.356717741, 11.28896566), 0.012665233, 1e-7),
    ("rc20_three_bar_truss", (0.788675, 0.408248), 263.8958, 1e-6),
    ("rc29_gas_compressor", (50, 1.178284, 24.592590, 0.388353), 2964895.4173, 1e-6),
    ("rc10_process_flow_sheeting", (0.94194, -2.1, 1), 1.076543, 2e-5),
]


@pytest.mark.parametrize("name,x,f_pub,rtol", LITERATURE_POINTS, ids=[t[0] for t in LITERATURE_POINTS])
def test_published_points(name, x, f_pub, rtol):
    """The transcription reproduces published values at published points (near-feasible at printed digits)."""
    p = RealWorldProblem(REALWORLD_SPECS[name])
    x = np.asarray(x, dtype=np.float64)
    assert p.eval(x) == pytest.approx(f_pub, rel=rtol)
    # Printed digits may miss an active constraint by a rounding error, never by more.
    assert p.violation(x) <= 1e-6


def test_registry_shape_and_names():
    assert len(SPECS) == 18
    assert len({s.cec_id for s in SPECS}) == 18
    assert all(s.dim <= 14 for s in SPECS)
    by_id = make_realworld_instances(["RC17", "rc01"])
    assert [n for n, _p in by_id] == ["rc17_spring", "rc01_heat_exchanger_1"]
    with pytest.raises(KeyError):
        make_realworld_instances(["rc99"])
    for spec in SPECS:
        assert spec.f_best_source and spec.ref_source


def test_constraint_layout_and_equality_tolerance():
    p = RealWorldProblem(REALWORLD_SPECS["rc05_haverly_pooling"])
    assert (p.n_ineq, p.n_eq) == (2, 4)
    x = np.array([0, 200, 0, 100, 0, 100, 0, 100, 1], dtype=np.float64)
    c = p.eval_constraints(x)
    assert c.shape == (6,)
    assert np.all(c[2:] == -EQ_TOL)  # |h| = 0 exactly
    # h2 = x1 - x5 - x7: a violation of 5e-5 is inside the tolerance, 2e-4 is not.
    y = x.copy()
    y[0] = 5e-5
    assert p.is_feasible(y) and p.violation(y) == 0.0
    y[0] = 2e-4
    assert not p.is_feasible(y)
    assert p.violation(y) == pytest.approx(2e-4 / 6)


# ---------------------------------------------------------------------------
# Integer variables and failure regions
# ---------------------------------------------------------------------------


def test_integer_variables_are_rounded_like_matlab():
    p = RealWorldProblem(REALWORLD_SPECS["rc18_pressure_vessel"])
    base = np.array([13.0, 7.0, 42.0984455958549, 176.63659585])
    for d in (-0.49, -0.2, 0.3, 0.49):
        y = base.copy()
        y[0] += d
        assert p.eval(y) == p.eval(base)  # a plateau
    q = RealWorldProblem(REALWORLD_SPECS["rc10_process_flow_sheeting"])
    # x3 in [-0.51, 1.49]: -0.5 rounds away from zero (MATLAB), to -1 -> a different value than 0.
    assert q.eval(np.array([0.5, -1.5, -0.5])) != q.eval(np.array([0.5, -1.5, -0.49]))


def test_failure_region_raises_and_constraints_are_unknown():
    p = RealWorldProblem(REALWORLD_SPECS["rc01_heat_exchanger_1"])
    bad = np.array(p.spec.x_ref)
    bad[6], bad[8] = 400.0, 300.0  # x9 < x7: log of a negative temperature difference in h8
    with pytest.raises(EvaluationCrashed):
        p.eval(bad)
    assert np.all(np.isnan(p.eval_constraints(bad)))
    assert not p.is_feasible(bad)
    assert p.violation(bad) == float("inf")
    with pytest.raises(EvaluationCrashed):
        p(Point(bad, "test"))  # the evaluation path books it as a failed evaluation


@pytest.mark.parametrize(
    "name,lo,hi",
    [("rc01_heat_exchanger_1", 0.25, 0.38), ("rc02_heat_exchanger_2", 0.42, 0.58)],
)
def test_failure_share_of_the_heat_exchangers(name, lo, hi):
    p = RealWorldProblem(REALWORLD_SPECS[name])
    rng = np.random.default_rng(0)
    n, failed = 4000, 0
    for _ in range(n):
        try:
            p.eval(p.random_point(rng=rng))
        except EvaluationCrashed:
            failed += 1
    assert lo < failed / n < hi


def test_no_failures_on_the_other_problems():
    rng = np.random.default_rng(1)
    for spec in SPECS:
        if spec.failure:
            continue
        p = RealWorldProblem(spec)
        for _ in range(300):
            p.eval(p.random_point(rng=rng))  # never raises


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_deterministic_and_picklable():
    rng = np.random.default_rng(7)
    for spec in SPECS:
        a, b = RealWorldProblem(spec), pickle.loads(pickle.dumps(RealWorldProblem(spec)))
        for _ in range(20):
            x = a.random_point(rng=rng)
            try:
                fa = a.eval(x)
            except EvaluationCrashed:
                with pytest.raises(EvaluationCrashed):
                    b.eval(x)
                continue
            assert b.eval(x) == fa == a.eval(x)
            np.testing.assert_array_equal(a.eval_constraints(x), b.eval_constraints(x))


# ---------------------------------------------------------------------------
# Tracker and harness
# ---------------------------------------------------------------------------


def test_tracker_scores_the_feasible_relative_gap():
    p = RealWorldProblem(REALWORLD_SPECS["rc17_spring"])
    tracker = FeasibleGapTracker(p, budget=10)
    try:
        infeasible = np.array([0.05, 0.25, 2.0])
        assert not p.is_feasible(infeasible)
        p.eval(infeasible)
        assert tracker.best_so_far == [float("inf")]
        p.eval(np.array(p.spec.x_ref))
        assert tracker.best_so_far[-1] == pytest.approx(p.relative_gap(p.spec.f_ref))
        assert tracker.best_raw_fx == p.spec.f_ref
    finally:
        tracker.restore()


def test_tracker_counts_a_crash_as_spent_budget():
    p = RealWorldProblem(REALWORLD_SPECS["rc01_heat_exchanger_1"])
    bad = np.array(p.spec.x_ref)
    bad[6], bad[8] = 400.0, 300.0
    tracker = FeasibleGapTracker(p, budget=10)
    try:
        p.eval(np.array(p.spec.x_ref))
        with pytest.raises(EvaluationCrashed):
            p.eval(bad)
        assert tracker.n_evals == 2
        assert tracker.best_so_far[0] == tracker.best_so_far[1] < 1e-7
    finally:
        tracker.restore()


def test_batteries():
    assert [n for n, _p in make_realworld_battery()] == IDS
    assert [n for n, _p in make_realworld_quick_battery()] == list(QUICK_PROBLEMS)


def test_harness_smoke_run():
    """A tiny run through the harness: every cell spends its budget and is scored."""
    spec = [s for s in make_ioh_strategies() if s.name == "RoundRobin_Random"]
    result = run_realworld_harness(
        spec, make_realworld_quick_battery(), budget_multiplier=10, base_seed=3, progress=False
    )
    assert result.problem_kind == "realworld" and len(result.runs) == 3
    for run in result.runs:
        assert run.error is None, run.error
        assert run.n_evals == run.budget == 10 * run.dim
        assert run.f_opt == 0.0 and 0.0 <= run.aocc <= 1.0
    again = run_realworld_harness(
        spec, make_realworld_quick_battery(), budget_multiplier=10, base_seed=3, progress=False
    )
    assert [r.best_fx for r in again.runs] == [r.best_fx for r in result.runs]
    assert [r.aocc for r in again.runs] == [r.aocc for r in result.runs]


def _cli():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "scripts" / "ioh_benchmark.py"
    spec = importlib.util.spec_from_file_location("ioh_benchmark_cli_realworld", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cli_realworld_quick_writes_a_result(tmp_path):
    from panobbgo.harness_ioh import IOHHarnessResult

    out = tmp_path / "rw.json"
    argv = ["run", "--realworld-quick", "--strategies", "RoundRobin_Random", "--quiet", "--output", str(out)]
    assert _cli().main(argv) == 0
    result = IOHHarnessResult.from_dict(json.loads(out.read_text()))
    assert result.battery_name == "realworld-quick"
    assert sorted(r.problem_kind for r in result.runs) == sorted(QUICK_PROBLEMS)


def test_cli_realworld_subset_and_errors():
    import argparse

    cli = _cli()
    ns = argparse.Namespace(realworld=True, realworld_quick=False, realworld_problems=["RC17", "rc20_three_bar_truss"])
    name, instances, mult = cli._resolve_realworld_battery(ns)
    assert name == "realworld-subset" and mult == 500
    assert [n for n, _p in instances] == ["rc17_spring", "rc20_three_bar_truss"]
    ns = argparse.Namespace(realworld=False, realworld_quick=False, realworld_problems=None)
    assert cli._resolve_realworld_battery(ns) is None
    with pytest.raises(SystemExit):
        cli._resolve_realworld_battery(
            argparse.Namespace(realworld=True, realworld_quick=False, realworld_problems=["x"])
        )
    with pytest.raises(SystemExit):
        cli._resolve_realworld_battery(
            argparse.Namespace(realworld=False, realworld_quick=False, realworld_problems=["RC17"])
        )
