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

from panobbgo.harness_baselines import BASELINE_PENALTY_RHO, _EvaluationLog, _make_objective, penalized_value
from panobbgo.harness_ioh import (
    SCORED_OBJECTIVE,
    SCORED_RELATIVE_FEASIBLE_GAP,
    IOHHarnessResult,
    make_ioh_strategies,
)
from panobbgo.harness_realworld import (
    QUICK_PROBLEMS,
    REALWORLD_LOG_HI,
    FeasibleGapTracker,
    make_realworld_battery,
    make_realworld_quick_battery,
    run_realworld_harness,
)
from panobbgo.lib import EvaluationCrashed, Point
from panobbgo.lib.realworld import (
    EQ_TOL,
    REALWORLD_SPECS,
    RealWorldProblem,
    RealWorldSpec,
    make_realworld_instances,
)
from panobbgo.local_run import BLAS_THREADS
from panobbgo.virtual_clock import VirtualSpec

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
    # CEC 2006 g04 (Liang et al. 2006): f(x*) = -3.066553867178332e+004 at the published optimum.
    ("rc32_himmelblau", (78, 33, 29.9952560256815985, 45, 36.7758129057882073), -30665.53867178332, 1e-14),
    # The suite's Revision.docx (16 Nov 2019): proven optimum of the integer pressure vessel.
    ("rc18_pressure_vessel", (13, 7, 42.0984455958549, 176.6365958424394), 6059.714335048436, 1e-14),
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
    assert len(SPECS) == 19
    assert len({s.cec_id for s in SPECS}) == 19
    assert all(s.dim <= 14 for s in SPECS)
    by_id = make_realworld_instances(["RC17", "rc01u"])
    assert [n for n, _p in by_id] == ["rc17_spring", "rc01u_heat_exchanger_1"]
    with pytest.raises(KeyError):
        make_realworld_instances(["rc99"])
    with pytest.raises(KeyError):
        make_realworld_instances(["RC01"])  # the unguarded variant is RC01u, never plain RC01
    for spec in SPECS:
        assert spec.f_best_source and spec.ref_source
        assert spec.f_best != 0.0
    # The unguarded heat exchangers say so in every name.
    assert {s.cec_id for s in SPECS if "unguarded" in s.title} == {"RC01u", "RC02u"}


def test_spec_refuses_a_zero_best_known_value():
    spec = REALWORLD_SPECS["rc17_spring"]
    with pytest.raises(ValueError, match="non-zero"):
        RealWorldSpec(**{**spec.__dict__, "f_best": 0.0})


def test_rc28_uses_the_paper_bound_for_z():
    """Z = 4 is admissible (lower bound 3.51 before rounding), as Table 3's value needs."""
    p = RealWorldProblem(REALWORLD_SPECS["rc28_rolling_bearing"])
    assert p.box[2, 0] == 3.51
    x = np.array(p.spec.x_ref)
    x[2] = 3.6  # rounds to 4
    assert p.eval(x) == p.spec.f_ref and p.is_feasible(x)


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
    # x3 in [-0.51, 1.49]: [-0.51, -0.5] rounds away from zero (MATLAB), to -1 -> a different value than 0.
    assert q.eval(np.array([0.5, -1.5, -0.5])) == q.eval(np.array([0.5, -1.5, -0.51]))
    assert q.eval(np.array([0.5, -1.5, -0.5])) != q.eval(np.array([0.5, -1.5, -0.49]))


def test_failure_region_raises_and_constraints_are_unknown():
    p = RealWorldProblem(REALWORLD_SPECS["rc01u_heat_exchanger_1"])
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
    [("rc01u_heat_exchanger_1", 0.25, 0.38), ("rc02u_heat_exchanger_2", 0.42, 0.58)],
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
    p = RealWorldProblem(REALWORLD_SPECS["rc01u_heat_exchanger_1"])
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


def test_tracker_feasibility_record_and_closed_value():
    p = RealWorldProblem(REALWORLD_SPECS["rc17_spring"])
    tracker = FeasibleGapTracker(p, budget=3)
    try:
        infeasible = np.array([0.05, 0.25, 2.0])
        p.eval(infeasible)
        assert tracker.first_feasible_eval is None
        assert tracker.best_violation == p.violation(infeasible) > 0
        assert tracker._closed_value() == float("inf")  # nothing feasible yet: no objective value
        p.eval(np.array(p.spec.x_ref))
        p.eval(infeasible)
        assert tracker.first_feasible_eval == 2 and tracker.best_violation == 0.0
        # Past the budget the strategy gets the best feasible *objective* value, not the gap.
        assert p.eval(infeasible) == p.spec.f_ref
        assert tracker.n_evals == 3
    finally:
        tracker.restore()


def test_negative_gap_is_clamped_to_full_credit():
    """A point below the best-known value scores like the best-known value, never above 1."""
    from panobbgo.ioh_runner import aocc

    gap = [-1e-3] * 10
    assert aocc(gap, f_opt=0.0, log_hi=REALWORLD_LOG_HI, budget=10) == 1.0
    assert aocc([float("inf")] * 10, f_opt=0.0, log_hi=REALWORLD_LOG_HI, budget=10) == 0.0
    assert aocc([1.0] * 10, f_opt=0.0, log_hi=REALWORLD_LOG_HI, budget=10) == 0.0  # 100 % gap: the ceiling


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


def _small_run(**kw):
    spec = [s for s in make_ioh_strategies() if s.name == "RoundRobin_Random"]
    battery = make_realworld_instances(["RC17", "RC20"])
    return run_realworld_harness(spec, battery, budget_multiplier=15, base_seed=5, progress=False, **kw)


def test_result_fields_round_trip():
    result = _small_run()
    assert result.scored == SCORED_RELATIVE_FEASIBLE_GAP
    assert result.log_hi == REALWORLD_LOG_HI == 0.0
    assert result.blas_threads == BLAS_THREADS
    for run in result.runs:
        assert run.feasible is not None and run.best_violation is not None
        if run.feasible:
            assert run.best_violation == 0.0 and run.first_feasible_eval is not None
            assert 1 <= run.first_feasible_eval <= run.n_evals
            assert np.isfinite(run.best_fx)
        else:
            assert run.first_feasible_eval is None and run.best_fx == float("inf") and run.aocc == 0.0
    back = IOHHarnessResult.from_dict(json.loads(result.to_json()))
    assert back.scored == SCORED_RELATIVE_FEASIBLE_GAP and back.log_hi == 0.0
    assert [(r.feasible, r.best_violation, r.first_feasible_eval) for r in back.runs] == [
        (r.feasible, r.best_violation, r.first_feasible_eval) for r in result.runs
    ]
    # An older file (no "scored" key) reads as scoring the objective.
    d = json.loads(result.to_json())
    del d["scored"]
    assert IOHHarnessResult.from_dict(d).scored == SCORED_OBJECTIVE


def test_worker_pool_gives_the_same_records():
    serial, pooled = _small_run(), _small_run(jobs=2)
    key = [(r.problem_kind, r.aocc, r.best_fx, r.first_feasible_eval, r.n_evals) for r in serial.runs]
    assert [(r.problem_kind, r.aocc, r.best_fx, r.first_feasible_eval, r.n_evals) for r in pooled.runs] == key


def test_virtual_clock_scores_time_too():
    result = _small_run(virtual=VirtualSpec(workers=4))
    assert result.virtual is not None
    for run in result.runs:
        assert run.error is None and run.n_evals == run.budget
        assert run.aocc_time is not None and 0.0 <= run.aocc_time <= 1.0


# ---------------------------------------------------------------------------
# Baselines on constrained problems
# ---------------------------------------------------------------------------


def test_baseline_objective_penalises_constraints():
    p = RealWorldProblem(REALWORLD_SPECS["rc17_spring"])
    log = _EvaluationLog(who="test", max_eval=5)
    objective = _make_objective(p, log)
    feasible = np.array(p.spec.x_ref)
    assert objective(feasible) == p.spec.f_ref  # feasible: f, unchanged
    infeasible = np.array([0.05, 0.25, 2.0])
    c = p.eval_constraints(infeasible)
    cv = float(np.linalg.norm(c[c > 0]))
    assert objective(infeasible) == pytest.approx(p.eval(infeasible) + BASELINE_PENALTY_RHO * cv)
    assert log.fxs[1] > p.eval(infeasible)  # the log (and so the baseline's best) holds the penalty value
    # An unknown violation (NaN, a failure region) is an infinite penalty.
    assert penalized_value(1.0, np.array([np.nan])) == float("inf")
    assert penalized_value(1.0, None) == 1.0 and penalized_value(1.0, np.array([-1.0, 0.0])) == 1.0


def test_baseline_objective_is_unchanged_without_constraints():
    """Unconstrained problems: bit-identical to the bare objective (classic and family problems)."""
    from panobbgo.lib.classic import Rosenbrock
    from panobbgo.lib.families import Family

    for p in (Rosenbrock(dims=3), Family("rastrigin", dim=3, seed=1)):
        assert p.eval_constraints(p.center) is None
        log = _EvaluationLog(who="test", max_eval=20)
        objective = _make_objective(p, log)
        rng = np.random.default_rng(0)
        for _ in range(20):
            x = p.random_point(rng=rng)
            assert objective(x) == p.eval(p.project(x))
        assert log.fxs == [p.eval(x) for x in log.xs]


def test_baseline_objective_penalises_a_constrained_family():
    """Every constrained track: the families-constrained baselines get f + 100 cv as well."""
    from panobbgo.lib.families import Family

    p = Family("sphere", dim=2, seed=3, n_constraints=2, constraint_kind="linear")
    log = _EvaluationLog(who="test", max_eval=200)
    objective = _make_objective(p, log)
    rng = np.random.default_rng(1)
    n_infeasible = 0
    for _ in range(200):
        x = p.random_point(rng=rng)
        c = p.eval_constraints(x)
        assert c is not None
        cv = float(np.linalg.norm(c[c > 0])) if np.any(c > 0) else 0.0
        n_infeasible += cv > 0
        assert objective(x) == pytest.approx(p.eval(x) + BASELINE_PENALTY_RHO * cv, rel=1e-15)
    assert n_infeasible > 0


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
    with pytest.raises(SystemExit, match="not --realworld-quick"):
        cli._resolve_realworld_battery(
            argparse.Namespace(realworld=False, realworld_quick=True, realworld_problems=["RC17"])
        )


def test_cli_realworld_budget_multiplier():
    import argparse

    cli = _cli()
    ns = argparse.Namespace(realworld=True, realworld_quick=False, realworld_problems=None, budget_multiplier=20)
    name, instances, mult = cli._resolve_realworld_battery(ns)
    assert (name, mult, len(instances)) == ("realworld-b20", 20, len(SPECS))
    ns = argparse.Namespace(realworld=False, realworld_quick=True, realworld_problems=None, budget_multiplier=7)
    assert cli._resolve_realworld_battery(ns)[0::2] == ("realworld-quick-b7", 7)


def test_cli_compare_refuses_a_different_scored_quantity(tmp_path, capsys):
    from panobbgo.harness_ioh import IOHRunRecord

    def result(scored):
        run = IOHRunRecord("p", 2, 0, "A", 0, 10, 10, 0.1, 0.0, 0.5, 0.0, 1)
        return IOHHarnessResult("b", "k", -8.0, 0.0, [run], sync_eval=True, scored=scored)

    rw, obj, rw2 = tmp_path / "rw.json", tmp_path / "obj.json", tmp_path / "rw2.json"
    rw.write_text(result(SCORED_RELATIVE_FEASIBLE_GAP).to_json())
    rw2.write_text(result(SCORED_RELATIVE_FEASIBLE_GAP).to_json())
    obj.write_text(result(SCORED_OBJECTIVE).to_json())
    cli = _cli()
    assert cli.main(["compare", str(rw), str(obj)]) == 2
    assert "different quantities" in capsys.readouterr().err
    assert cli.main(["compare", str(rw), str(rw2)]) == 0
