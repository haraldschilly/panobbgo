#!/usr/bin/env python
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

"""
Tests for the expensive-track baselines in ``panobbgo.harness_baselines_bo``
(BoTorch qLogEI, TuRBO-1, SMAC3 BlackBox, Py-BOBYQA).

The library-backed tests skip without the optional ``baselines-bo`` extra;
CI runs them in the separate ``test-bo`` job (``.github/workflows/tests.yml``).
Budgets stay tiny (about 20 evaluations at dim 2): GP fits are slow.
"""

from __future__ import annotations

import importlib.util
import os
import threading
from pathlib import Path
from typing import Any, List

import numpy as np
import pandas as pd
import pytest

from panobbgo.harness_baselines import (
    ALL_EXTERNAL_BASELINE_NAMES,
    BO_BASELINE_NAMES,
    EXTERNAL_BASELINE_NAMES,
    make_baseline_strategies,
    make_external_baseline_strategies,
)
from panobbgo.harness_baselines_bo import (
    BO_BASELINE_CLASSES,
    FUNCMAX,
    BoTorchQLogEIStrategy,
    BoTorchTuRBOStrategy,
    PyBOBYQAStrategy,
    SMACBlackBoxStrategy,
    impute_worst,
    moderated_extreme_barrier,
)
from panobbgo.lib import EvaluationCrashed
from panobbgo.lib.classic import Rosenbrock


def _has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _needs(*modules: str):
    missing = [m for m in modules if not _has(m)]
    return pytest.mark.skipif(bool(missing), reason=f"needs the 'baselines-bo' extra ({', '.join(missing)})")


TORCH = ("torch", "gpytorch", "botorch")
ADAPTERS = [
    pytest.param(BoTorchQLogEIStrategy, marks=_needs(*TORCH), id="BoTorch_qLogEI"),
    pytest.param(BoTorchTuRBOStrategy, marks=_needs(*TORCH), id="TuRBO1"),
    pytest.param(SMACBlackBoxStrategy, marks=_needs("smac"), id="SMAC_BB"),
    pytest.param(PyBOBYQAStrategy, marks=_needs("pybobyqa"), id="PyBOBYQA"),
]


class FailingRosenbrock(Rosenbrock):
    """Rosenbrock (box [0, 2] x [-2, 2]) whose calls crash right of x0 = 0.6, the optimum included."""

    def eval(self, x: Any) -> float:
        if float(x[0]) > 0.6:
            raise EvaluationCrashed("failure region")
        return super().eval(x)


def _run(cls, *, max_eval: int = 20, seed: int = 5, batch_size: int = 1, problem: Any = None):
    problem = problem or Rosenbrock(dims=2)
    strategy = cls(problem, seed=seed, batch_size=batch_size)
    strategy.config.max_eval = max_eval
    strategy.start()
    return problem, strategy


# ---------------------------------------------------------------------------
# Library-backed adapters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", ADAPTERS)
@pytest.mark.parametrize("batch_size", [1, 4])
def test_exact_budget_and_well_formed_frame(cls, batch_size):
    # 19 is not a multiple of any batch or design size involved.
    problem, strategy = _run(cls, max_eval=19, batch_size=batch_size)
    df = strategy.results.results
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 19
    assert set(df[("who", 0)]) == {cls.who}
    xs = df[[("x", 0), ("x", 1)]].to_numpy()
    assert np.all(xs >= problem.box[:, 0]) and np.all(xs <= problem.box[:, 1])
    fx = df[("fx", 0)].to_numpy()
    assert np.all(np.isfinite(fx))
    assert strategy.best is not None and strategy.best.fx == pytest.approx(fx.min())


@pytest.mark.parametrize("cls", ADAPTERS)
def test_deterministic_for_fixed_seed_and_restores_global_rngs(cls):
    np.random.seed(0)
    np_state = np.random.get_state()[1].copy()
    torch = pytest.importorskip("torch") if _has("torch") else None
    torch_state = torch.get_rng_state().clone() if torch is not None else None
    _, a = _run(cls, max_eval=12, seed=11, batch_size=2)
    _, b = _run(cls, max_eval=12, seed=11, batch_size=2)
    pd.testing.assert_frame_equal(a.results.results, b.results.results)
    _, c = _run(cls, max_eval=6, seed=12, batch_size=2)
    assert not np.allclose(a.results.results[("x", 0)].to_numpy()[:6], c.results.results[("x", 0)].to_numpy())
    assert np.array_equal(np.random.get_state()[1], np_state)
    if torch is not None:
        assert torch.equal(torch.get_rng_state(), torch_state)


@pytest.mark.parametrize("cls", ADAPTERS)
@pytest.mark.parametrize("batch_size", [1, 3])
def test_failed_evaluations_are_survived_and_recorded_as_nan(cls, batch_size):
    """Half the box fails: the run spends its whole budget, the frame keeps NaN, the model never sees one."""
    _, strategy = _run(cls, max_eval=20, seed=2, batch_size=batch_size, problem=FailingRosenbrock(dims=2))
    fx = strategy.results.results[("fx", 0)].to_numpy()
    assert len(fx) == 20
    assert np.isnan(fx).any() and np.isfinite(fx).any()
    assert strategy.best is not None and np.isfinite(strategy.best.fx)


def test_impute_worst_and_extreme_barrier():
    assert np.array_equal(impute_worst(np.array([1.0, np.nan, 3.0, np.inf, -2.0])), [1.0, 3.0, 3.0, 3.0, -2.0])
    with pytest.raises(ValueError):
        impute_worst(np.array([np.nan, np.inf]))
    assert moderated_extreme_barrier(float("nan")) == FUNCMAX
    assert moderated_extreme_barrier(float("inf")) == FUNCMAX
    assert moderated_extreme_barrier(-float("inf")) == -FUNCMAX
    assert moderated_extreme_barrier(2.5) == 2.5


@_needs(*TORCH)
def test_qlogei_gp_sees_the_worst_value_for_failures(monkeypatch):
    import botorch

    seen: List[Any] = []
    real = botorch.models.SingleTaskGP

    def spy(X, Y, *a, **k):
        seen.append(Y.detach().numpy().ravel().copy())
        return real(X, Y, *a, **k)

    monkeypatch.setattr(botorch.models, "SingleTaskGP", spy)
    adapter = BoTorchQLogEIStrategy(Rosenbrock(dims=2), seed=1).make_adapter(1, 20, 1)
    values = [4.0, float("nan"), 1.0, float("inf"), 7.0]  # the 5-point Sobol design at dim 2
    for fx in values:
        ((key, _x),) = adapter.ask(1)
        adapter.tell(key, fx)
    adapter.ask(1)  # first model-based proposal
    assert np.allclose(seen[-1], [-4.0, -7.0, -1.0, -7.0, -7.0])  # negated; failures = worst (7)


@_needs(*TORCH)
def test_qlogei_batches_and_pending_points():
    adapter = BoTorchQLogEIStrategy(Rosenbrock(dims=2), seed=4).make_adapter(4, 30, 4)
    assert adapter.n_init == 5
    assert adapter.min_observed == 3  # Ax: half the design, at least 2
    first = adapter.ask(4)  # design
    rest = adapter.ask(4)  # nothing observed yet: design point 5, then more Sobol points
    assert len(first) == 4 and len(rest) == 4
    problem = Rosenbrock(dims=2)
    for key, x in first + rest:
        adapter.tell(key, float(problem.eval(x)))
    batch = adapter.ask(3)
    assert len(batch) == 3 and len({tuple(np.round(x, 9)) for _k, x in batch}) == 3
    # Asked but untold points are X_pending: a later single ask avoids them.
    (one,) = adapter.ask(1)
    assert set(adapter._pending) == {k for k, _ in batch} | {one[0]}
    assert min(np.linalg.norm(one[1] - x) for _k, x in batch) > 1e-6


@_needs(*TORCH)
def test_qlogei_gets_the_pending_points_as_x_pending(monkeypatch):
    import botorch

    shapes: List[Any] = []
    real = botorch.acquisition.logei.qLogExpectedImprovement

    def spy(*a, X_pending=None, **k):
        shapes.append(None if X_pending is None else tuple(X_pending.shape))
        return real(*a, X_pending=X_pending, **k)

    monkeypatch.setattr(botorch.acquisition.logei, "qLogExpectedImprovement", spy)
    problem = Rosenbrock(dims=2)
    adapter = BoTorchQLogEIStrategy(problem, seed=4).make_adapter(4, 30, 2)
    for key, x in adapter.ask(5):  # the design, all told
        adapter.tell(key, float(problem.eval(x)))
    adapter.ask(2)  # a joint 2-batch, nothing pending
    adapter.ask(1)  # one more, with the 2-batch pending
    assert shapes == [None, (2, 2)]


@_needs(*TORCH)
def test_qlogei_waits_for_half_the_design_before_modelling():
    adapter = BoTorchQLogEIStrategy(Rosenbrock(dims=2), seed=4).make_adapter(4, 30, 1)
    keys = [k for k, _x in adapter.ask(5)]
    for k in keys[:2]:
        adapter.tell(k, 1.0)
    assert not adapter._can_model()  # 2 of the needed 3 observed: more Sobol points
    adapter.tell(keys[2], 2.0)
    assert adapter._can_model()


@_needs(*TORCH)
def test_qlogei_initial_design_fills_the_first_batch():
    adapter = BoTorchQLogEIStrategy(Rosenbrock(dims=2), seed=4).make_adapter(4, 100, 16)
    assert adapter.n_init == 16 and len(adapter.ask(16)) == 16


@_needs(*TORCH)
def test_turbo_is_synchronous_per_batch_and_restarts():
    adapter = BoTorchTuRBOStrategy(Rosenbrock(dims=2), seed=6).make_adapter(6, 200, 3)
    assert adapter.n_init == 4
    init = adapter.ask(10)
    assert len(init) == 4 and adapter.ask(2) == []  # waits for the design
    for key, _x in init:
        adapter.tell(key, 1.0)
    assert adapter.state is not None and adapter.state.failure_tolerance == 2  # ceil(max(4/3, 2/3))
    for _ in range(40):  # a flat objective: every batch fails, the region shrinks to a restart
        batch = adapter.ask(5)
        assert len(batch) <= 3
        assert adapter.ask(1) == []
        for key, _x in batch:
            adapter.tell(key, 1.0)
        if adapter.restarts:
            break
    assert adapter.restarts == 1 and adapter.state is None
    assert len(adapter.ask(10)) == 4  # a fresh design


@_needs("smac")
def test_smac_parallel_asks_and_conservative_failure_cost():
    adapter = SMACBlackBoxStrategy(Rosenbrock(dims=2), seed=3).make_adapter(3, 20, 4)
    tmp = adapter._tmp
    try:
        trials = adapter.ask(4)
        assert len({tuple(x) for _k, x in trials}) == 4
        (k0, _), (k1, _), (k2, _), (k3, _) = trials
        adapter.tell(k0, float("nan"))  # no finite value yet: waits as a running trial
        assert len(adapter._deferred) == 1
        adapter.tell(k1, 5.0)  # releases k0: worst = best = 5, margin |5|·1e-6
        adapter.tell(k2, 2.0)
        adapter.tell(k3, float("inf"))  # 5 + (5 - 2) = 8: below every real point
        costs = sorted(v.cost for v in adapter._smac.runhistory._data.values())
        assert costs == [2.0, 5.0, pytest.approx(5.000005, abs=1e-12), 8.0]
        # Equal values of 0: the absolute floor keeps the failure strictly worse.
        adapter._worst = adapter._best = 0.0
        assert adapter.failure_cost() == 1e-12
        adapter._worst, adapter._best = 1e308, -1e308
        assert adapter.failure_cost() == np.finfo(float).max
    finally:
        adapter.close()
    assert not os.path.exists(tmp)


@_needs("pybobyqa")
def test_pybobyqa_is_sequential_and_close_stops_its_thread():
    problem = Rosenbrock(dims=2)
    before = set(threading.enumerate())
    adapter = PyBOBYQAStrategy(problem, seed=1).make_adapter(1, 50, 4)
    ((key, x),) = adapter.ask(4)
    assert adapter.ask(4) == []  # one point in flight at most
    adapter.tell(key, float(problem.eval(x)))
    assert len(adapter.ask(4)) == 1
    started = [t for t in set(threading.enumerate()) - before if t.is_alive()]
    assert started
    adapter.close()
    for thread in started:
        thread.join(timeout=10)
        assert not thread.is_alive()


@_needs("pybobyqa")
def test_pybobyqa_run_without_evaluations_raises_instead_of_restarting_forever(monkeypatch):
    import pybobyqa

    from panobbgo.harness_baselines_bo import _PyBOBYQAAdapter

    with pytest.raises(ValueError, match="hi > lo"):
        _PyBOBYQAAdapter(np.zeros(2), np.zeros(2), seed=1, budget=10)
    # Py-BOBYQA's EXIT_INPUT_ERROR returns before the first evaluation.  The
    # stub fails the test on a second run (a restart loop), so it cannot hang.
    runs = []

    def solve(*a, **k):
        runs.append(1)
        assert len(runs) == 1, "restarted after a run without evaluations"

    monkeypatch.setattr(pybobyqa, "solve", solve)
    adapter = _PyBOBYQAAdapter(np.zeros(2), np.ones(2), seed=1, budget=10)
    try:
        with pytest.raises(RuntimeError, match="without evaluating"):
            adapter.ask(1)
    finally:
        adapter.close()


@_needs("pybobyqa")
def test_pybobyqa_restarts_from_a_new_point_after_convergence():
    """A tiny quadratic converges long before 300 evaluations: the adapter starts new runs."""
    strategy = PyBOBYQAStrategy(Rosenbrock(dims=2), seed=1)
    adapter = strategy.make_adapter(1, 300, 1)
    try:
        for _ in range(300):
            ((key, x),) = adapter.ask(1)
            adapter.tell(key, float(np.sum((x - 0.3) ** 2)))
        assert adapter.runs >= 2
    finally:
        adapter.close()


@pytest.mark.parametrize("cls", ADAPTERS)
def test_runs_on_the_virtual_clock(cls):
    from panobbgo.benchmark import StrategySpec
    from panobbgo.harness_ioh import _run_tracked
    from panobbgo.ioh_runner import IOHTracker
    from panobbgo.virtual_clock import VirtualSpec

    kw = dict(f_opt=0.0, budget=20, seed=4, sync_eval=True, log_lo=-8.0, log_hi=6.0, timeout_s=None)
    spec = StrategySpec(name=cls.who, strategy_class=cls, heuristics=[])
    tracker = IOHTracker((p := Rosenbrock(dim=2)), budget=20)
    run = _run_tracked(spec, p, tracker, virtual=VirtualSpec(workers=4, duration="lognormal"), **kw)
    assert run.error is None and run.n_evals == 20 and run.aocc_time is not None
    times = [t for t, _ in tracker.timeline]
    assert times == sorted(times)


@_needs("pybobyqa")
def test_pybobyqa_on_the_virtual_clock_keeps_one_in_flight():
    """At q = 4 Py-BOBYQA uses one worker: completions are strictly sequential and aocc equals q = 1."""
    from panobbgo.benchmark import StrategySpec
    from panobbgo.harness_ioh import _run_tracked
    from panobbgo.ioh_runner import IOHTracker
    from panobbgo.virtual_clock import VirtualSpec

    kw = dict(f_opt=0.0, budget=20, seed=4, sync_eval=True, log_lo=-8.0, log_hi=6.0, timeout_s=None)
    spec = StrategySpec(name="PyBOBYQA", strategy_class=PyBOBYQAStrategy, heuristics=[])
    runs = {}
    for q in (1, 4):
        tracker = IOHTracker((p := Rosenbrock(dim=2)), budget=20)
        runs[q] = _run_tracked(spec, p, tracker, virtual=VirtualSpec(workers=q), **kw)
        # Constant durations of 1: one point in flight means completions at 1, 2, ..., 20.
        assert [t for t, _ in tracker.timeline] == [float(k + 1) for k in range(20)]
    assert runs[4].aocc == pytest.approx(runs[1].aocc)


# ---------------------------------------------------------------------------
# Registry (no optional dependency)
# ---------------------------------------------------------------------------


def test_bo_names_match_the_classes_and_join_the_registry():
    assert BO_BASELINE_NAMES == tuple(f"Baseline_{cls.who}" for cls in BO_BASELINE_CLASSES)
    assert ALL_EXTERNAL_BASELINE_NAMES[-len(BO_BASELINE_NAMES) :] == BO_BASELINE_NAMES
    specs = {s.name: s.strategy_class for s in make_external_baseline_strategies()}
    assert [specs[n] for n in BO_BASELINE_NAMES] == list(BO_BASELINE_CLASSES)


@pytest.mark.parametrize("name", BO_BASELINE_NAMES)
def test_missing_bo_extra_names_the_bo_extra(monkeypatch, name):
    real = importlib.util.find_spec
    gone = {"torch", "smac", "pybobyqa"}
    monkeypatch.setattr(importlib.util, "find_spec", lambda mod, *a: None if mod in gone else real(mod, *a))
    with pytest.raises(ImportError, match="`baselines-bo`.*--extra baselines-bo"):
        make_baseline_strategies([name])


def test_cheap_external_names_need_no_torch():
    assert not set(EXTERNAL_BASELINE_NAMES) & set(BO_BASELINE_NAMES)
    assert ALL_EXTERNAL_BASELINE_NAMES == EXTERNAL_BASELINE_NAMES + BO_BASELINE_NAMES
    assert all(getattr(cls, "no_wall_timeout", False) for cls in BO_BASELINE_CLASSES)


@_needs("pybobyqa")
def test_composite_harness_does_not_cut_bo_baselines_at_the_wall_timeout():
    from panobbgo.harness import BenchmarkHarness, HarnessConfig

    config = HarnessConfig(
        mode="quick",
        budget=20,
        reps=1,
        seed=42,
        problems=["Rosenbrock_2D"],
        strategies=["Baseline_PyBOBYQA"],
        include_baselines=True,
        timeout_per_run=1e-6,  # would cut any other strategy at once
    )
    result = BenchmarkHarness(config).run(verbose=False)
    (run,) = result.problem_strategy_results[0].runs
    assert run.error is None
    assert run.evaluations_used == 20


@_needs("pybobyqa")
def test_family_track_does_not_cut_bo_baselines_at_the_wall_timeout():
    from panobbgo.benchmark import StrategySpec
    from panobbgo.harness_baselines import RandomSearchStrategy
    from panobbgo.harness_families import _run_one, make_families_battery
    from panobbgo.harness_ioh import wall_timeout_for

    bo = StrategySpec(name="Baseline_PyBOBYQA", strategy_class=PyBOBYQAStrategy, heuristics=[])
    other = StrategySpec(name="Baseline_Random", strategy_class=RandomSearchStrategy, heuristics=[])
    assert wall_timeout_for(bo, 5.0) is None and wall_timeout_for(other, 5.0) == 5.0
    _name, problem = make_families_battery(dims=(2,), n_instances=1)[0]
    rec = _run_one(bo, problem, 0, 20, 1, -8.0, 2.0, True, timeout_s=1e-9)
    assert rec.error is None and rec.n_evals == 20


def _ioh_cli():
    path = Path(__file__).resolve().parent.parent / "scripts" / "ioh_benchmark.py"
    spec = importlib.util.spec_from_file_location("ioh_benchmark_cli_bo", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cli_budget_multiplier_overrides_ioh_and_family_batteries():
    import argparse

    cli = _ioh_cli()
    flags = dict(full=False, standard=True, noisy=None, noisy_highdim=None, highdim=False, reps=None)
    assert cli._resolve_battery(argparse.Namespace(**flags, budget_multiplier=None)).budget_multiplier == 500
    battery = cli._resolve_battery(argparse.Namespace(**flags, budget_multiplier=20))
    assert battery.budget_multiplier == 20 and battery.name.endswith("-b20")
    fam = dict(families_quick=False, families_constrained=False, families=True)
    name, _instances, bm = cli._resolve_family_battery(argparse.Namespace(**fam, budget_multiplier=100))
    assert (name, bm) == ("families-b100", 100)
    assert cli._resolve_family_battery(argparse.Namespace(**fam, budget_multiplier=None))[2] == 500
    with pytest.raises(SystemExit) as e:
        cli.main(["run", "--budget-multiplier", "0"])
    assert e.value.code == 2
