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
from typing import Any, List

import numpy as np
import pandas as pd
import pytest

from panobbgo.harness_baselines import (
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
    first = adapter.ask(4)  # design
    rest = adapter.ask(4)  # 1 design point + a 3-point q-batch (nothing told yet: waits)
    assert len(first) == 4 and len(rest) == 1
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
def test_smac_parallel_asks_and_deferred_failures():
    adapter = SMACBlackBoxStrategy(Rosenbrock(dims=2), seed=3).make_adapter(3, 20, 3)
    tmp = adapter._tmp
    try:
        trials = adapter.ask(3)
        assert len({tuple(x) for _k, x in trials}) == 3
        (k0, _), (k1, _), (k2, _) = trials
        adapter.tell(k0, float("nan"))  # no finite value yet: waits as a running trial
        assert len(adapter._deferred) == 1
        adapter.tell(k1, 5.0)
        adapter.tell(k2, float("inf"))
        costs = sorted(v.cost for v in adapter._smac.runhistory._data.values())
        assert costs == [5.0, 5.0, 5.0]  # both failures told with the worst finite value
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


# ---------------------------------------------------------------------------
# Registry (no optional dependency)
# ---------------------------------------------------------------------------


def test_bo_names_match_the_classes_and_join_the_registry():
    assert BO_BASELINE_NAMES == tuple(f"Baseline_{cls.who}" for cls in BO_BASELINE_CLASSES)
    assert EXTERNAL_BASELINE_NAMES[-len(BO_BASELINE_NAMES) :] == BO_BASELINE_NAMES
    specs = {s.name: s.strategy_class for s in make_external_baseline_strategies()}
    assert [specs[n] for n in BO_BASELINE_NAMES] == list(BO_BASELINE_CLASSES)


def test_missing_bo_extra_names_the_bo_extra(monkeypatch):
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None if name == "torch" else real(name, *a))
    with pytest.raises(ImportError, match="--extra baselines-bo"):
        make_baseline_strategies(["Baseline_TuRBO1"])
