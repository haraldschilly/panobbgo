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
Tests for the batch-capable external baselines in ``panobbgo.harness_baselines``
(pycma IPOP/BIPOP, Nevergrad, Optuna).

The library-backed tests skip when the optional ``baselines`` extra is not
installed; the driver and registry tests run without it.
"""

from __future__ import annotations

import importlib.util
import threading
from typing import List, Tuple

import numpy as np
import pandas as pd
import pytest

from panobbgo.harness import BenchmarkHarness, HarnessConfig
from panobbgo.harness_baselines import (
    EXTERNAL_BASELINE_NAMES,
    AskTellAdapter,
    AskTellBaselineStrategy,
    NevergradCMAStrategy,
    NevergradNGOptStrategy,
    NevergradTwoPointsDEStrategy,
    OptunaCmaEsStrategy,
    OptunaTPEStrategy,
    PycmaBIPOPStrategy,
    PycmaIPOPStrategy,
    _PycmaRestartAdapter,
    make_baseline_strategies,
    make_external_baseline_strategies,
)
from panobbgo.lib.classic import Rosenbrock


def _has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _needs(*modules: str):
    missing = [m for m in modules if not _has(m)]
    return pytest.mark.skipif(bool(missing), reason=f"needs the 'baselines' extra ({', '.join(missing)})")


ADAPTERS = [
    pytest.param(PycmaIPOPStrategy, marks=_needs("cma"), id="pycma_IPOP"),
    pytest.param(PycmaBIPOPStrategy, marks=_needs("cma"), id="pycma_BIPOP"),
    pytest.param(NevergradNGOptStrategy, marks=_needs("nevergrad"), id="NGOpt"),
    pytest.param(NevergradCMAStrategy, marks=_needs("nevergrad"), id="NG_CMA"),
    pytest.param(NevergradTwoPointsDEStrategy, marks=_needs("nevergrad"), id="NG_TwoPointsDE"),
    pytest.param(OptunaCmaEsStrategy, marks=_needs("optuna", "cmaes"), id="Optuna_CmaEs"),
    pytest.param(OptunaTPEStrategy, marks=_needs("optuna"), id="Optuna_TPE"),
]


def _run(cls, *, max_eval: int, seed: int, batch_size: int = 1):
    problem = Rosenbrock(dims=2)
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
    # 23 is not a multiple of any population size or batch size involved.
    problem, strategy = _run(cls, max_eval=23, seed=5, batch_size=batch_size)
    df = strategy.results.results
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 23
    for col in [("x", 0), ("x", 1), ("fx", 0), ("cv", 0), ("who", 0), ("error", 0)]:
        assert col in df.columns
    assert set(df[("who", 0)]) == {cls.who}
    xs = df[[("x", 0), ("x", 1)]].to_numpy()
    assert np.all(xs >= problem.box[:, 0]) and np.all(xs <= problem.box[:, 1])
    fx = df[("fx", 0)].to_numpy()
    assert np.all(np.isfinite(fx))
    assert strategy.best is not None
    assert strategy.best.fx == pytest.approx(fx.min())


@pytest.mark.parametrize("cls", ADAPTERS)
def test_deterministic_for_fixed_seed(cls):
    _, a = _run(cls, max_eval=20, seed=11, batch_size=2)
    _, b = _run(cls, max_eval=20, seed=11, batch_size=2)
    pd.testing.assert_frame_equal(a.results.results, b.results.results)


@_needs("nevergrad")
def test_ngopt_metamodel_cma_is_deterministic():
    """dim 5 / 300 evaluations: NGOpt runs MetaModel(CmaFmin2), i.e. ``cma.fmin`` in a Nevergrad thread."""
    runs = []
    for _ in range(2):
        strategy = NevergradNGOptStrategy(Rosenbrock(dims=5), seed=3)
        strategy.config.max_eval = 300
        strategy.start()
        runs.append(strategy.results.results)
    pd.testing.assert_frame_equal(runs[0], runs[1])


@_needs("nevergrad")
@pytest.mark.parametrize(("dim", "budget"), [(2, 30), (5, 1000)])
def test_ngopt_close_stops_recast_threads(dim, budget):
    """NGOpt picks Cobyla (dim 2) or MetaModel(CmaFmin2) (dim 5); both run in a Nevergrad thread."""
    problem = Rosenbrock(dims=dim)
    strategy = NevergradNGOptStrategy(problem, seed=1)
    before = set(threading.enumerate())
    adapter = strategy.make_adapter(1, budget, 1)  # held: GC must not be what stops the thread
    for _ in range(20):
        for key, x in adapter.ask(1):
            adapter.tell(key, float(problem.eval(x)))
    started = [t for t in set(threading.enumerate()) - before if t.is_alive()]
    assert started, "expected a recast worker thread"
    adapter.close()
    for thread in started:
        thread.join(timeout=10)
        assert not thread.is_alive()


@_needs("cma")
def test_pycma_popsize_schedule_and_batch_floor():
    """IPOP populations follow fmin2 (int of a float base) and q raises the base."""
    adapter = _PycmaRestartAdapter(np.zeros(5), np.ones(5), seed=1, bipop=False)
    sizes = []
    while len(sizes) < 4:
        for key, _x in adapter.ask(100):
            adapter.tell(key, 1.0)  # flat: every run stops quickly
        if adapter.popsize not in sizes:
            sizes.append(adapter.popsize)
    assert sizes == [8, 17, 35, 70]
    assert _PycmaRestartAdapter(np.zeros(5), np.ones(5), seed=1, bipop=True, batch_size=20).popsize == 20


@_needs("cma")
def test_run_restores_the_global_rng():
    np.random.seed(0)
    state = np.random.get_state()[1].copy()
    _run(PycmaBIPOPStrategy, max_eval=50, seed=4)
    assert np.array_equal(np.random.get_state()[1], state)


def _record_library_tells(adapter, told):
    """Wrap the library call behind ``adapter.tell`` so it appends every told value to ``told``."""
    if isinstance(adapter, _PycmaRestartAdapter):
        es = adapter._es
        inner = es.tell
        es.tell = lambda xs, fs, *a, **k: (told.extend(fs), inner(xs, fs, *a, **k))[1]
    elif hasattr(adapter, "_study"):
        inner = adapter._study.tell
        adapter._study.tell = lambda key, value=None, *a, **k: (told.append(value), inner(key, value, *a, **k))[1]
    else:
        inner = adapter._opt.tell
        adapter._opt.tell = lambda cand, loss, *a, **k: (told.append(loss), inner(cand, loss, *a, **k))[1]


@pytest.mark.parametrize("cls", ADAPTERS)
def test_nan_is_told_as_worst(cls):
    """One failed-value rule: the library receives +inf for NaN, other values unchanged."""
    strategy = cls(Rosenbrock(dims=2), seed=2)
    adapter = strategy.make_adapter(2, 30, 1)
    told: List[float] = []
    _record_library_tells(adapter, told)
    sent: List[float] = []
    try:
        for i in range(12):  # two full pycma generations (popsize 6 at dim 2)
            for key, _x in adapter.ask(1):
                fx = float("nan") if i % 3 == 0 else float(i)
                sent.append(fx)
                adapter.tell(key, fx)
        assert len(told) == len(sent) == 12
        for fx, got in zip(sent, told):
            assert got == (float("inf") if np.isnan(fx) else fx)
        if hasattr(adapter, "_study"):
            assert all(t.state.name == "COMPLETE" for t in adapter._study.trials)
    finally:
        adapter.close()


def _start_point(adapter) -> np.ndarray:
    """The start point the library itself holds, in the unit cube of the box."""
    if isinstance(adapter, _PycmaRestartAdapter):
        return np.asarray(adapter._es.mean, dtype=np.float64)
    box = Rosenbrock(dims=2).box
    if hasattr(adapter, "_study"):
        x0 = adapter._study.sampler._x0
        x = np.array([x0["x0"], x0["x1"]])
    else:
        x = np.asarray(adapter._opt.parametrization.value, dtype=np.float64)
    return (x - box[:, 0]) / (box[:, 1] - box[:, 0])


@pytest.mark.parametrize("cls", [p for p in ADAPTERS if p.id != "Optuna_TPE"])  # TPE has no start point
def test_start_point_is_random_not_the_centre(cls):
    starts = []
    for seed in (9, 10):
        adapter = cls(Rosenbrock(dims=2), seed=seed).make_adapter(seed, 50, 1)
        try:
            starts.append(_start_point(adapter))
        finally:
            adapter.close()
    assert not np.allclose(starts[0], 0.5)
    assert not np.allclose(starts[0], starts[1])  # drawn per seed


@_needs("cma")
def test_pycma_first_run_maxiter_matches_fmin2():
    adapter = _PycmaRestartAdapter(np.zeros(5), np.ones(5), seed=1, bipop=False)
    assert adapter._es.opts["maxiter"] == 3330


@_needs("nevergrad")
def test_fmin_seed_slot_is_released_only_by_its_owner():
    import panobbgo.harness_baselines as hb

    a = NevergradNGOptStrategy(Rosenbrock(dims=2), seed=1).make_adapter(1, 30, 1)
    b = NevergradNGOptStrategy(Rosenbrock(dims=2), seed=2).make_adapter(2, 30, 1)
    assert hb._CMA_FMIN_SEEDS is b._fmin_seeds
    a.close()  # not the owner: leaves b's generator in place
    assert hb._CMA_FMIN_SEEDS is b._fmin_seeds
    b.close()
    assert hb._CMA_FMIN_SEEDS is None


@_needs("cma")
@pytest.mark.parametrize("bipop", [False, True])
def test_pycma_restarts_on_a_flat_objective(bipop):
    """A flat objective stops each CMA-ES run quickly, so the restart schedule runs."""
    adapter = _PycmaRestartAdapter(np.zeros(2), np.ones(2), seed=3, bipop=bipop)
    for _ in range(400):
        for key, _x in adapter.ask(3):
            adapter.tell(key, 1.0)
    assert adapter.restarts >= 2


# ---------------------------------------------------------------------------
# Driver (no optional dependency)
# ---------------------------------------------------------------------------


class _FakeAdapter(AskTellAdapter):
    """Generational stub: hands out ``gen`` points, then waits for all tells."""

    def __init__(self, gen: int) -> None:
        self.gen = gen
        self.next_key = 0
        self.open: List[int] = []
        self.handed = 0
        self.ask_sizes: List[int] = []
        self.told: List[Tuple[int, float]] = []

    def ask(self, n):
        if self.handed == self.gen and not self.open:
            self.handed = 0
        out = []
        while len(out) < n and self.handed < self.gen:
            key = self.next_key
            self.next_key += 1
            self.handed += 1
            self.open.append(key)
            out.append((key, np.full(2, 0.01 * key)))
        self.ask_sizes.append(len(out))
        return out

    def tell(self, key, fx):
        self.open.remove(key)
        self.told.append((key, fx))


class _FakeStrategy(AskTellBaselineStrategy):
    who = "Fake"

    def make_adapter(self, seed, budget, batch_size):
        self.adapter = _FakeAdapter(gen=5)
        return self.adapter


def test_driver_batches_in_dispatch_order_and_trims_the_last_batch():
    strategy = _FakeStrategy(Rosenbrock(dims=2), seed=0, batch_size=3)
    strategy.config.max_eval = 11
    strategy.start()
    adapter = strategy.adapter
    # Generations of 5 with q=3: 3, 2 | 3, 2 | 1 (budget-trimmed).
    assert adapter.ask_sizes == [3, 2, 3, 2, 1]
    assert [k for k, _ in adapter.told] == list(range(11))
    df = strategy.results.results
    assert np.allclose(df[("x", 0)].to_numpy(), 0.01 * np.arange(11))


@_needs("nevergrad")
def test_batch_size_via_config_override():
    spec = [s for s in make_baseline_strategies(["Baseline_NGOpt"]) if s.name == "Baseline_NGOpt"][0]
    spec.config_overrides = {"batch_size": 4}
    strategy = spec.create_strategy(Rosenbrock(dims=2), seed=1, max_eval=10)
    assert strategy.config.batch_size == 4
    assert strategy.config.max_eval == 10


# ---------------------------------------------------------------------------
# Registry: opt-in by name, default --baselines set unchanged
# ---------------------------------------------------------------------------


def test_default_baseline_set_is_unchanged():
    names = [s.name for s in make_baseline_strategies()]
    assert names == ["Baseline_Random", "Baseline_SciPyDE", "Baseline_SciPyAnneal"]


@_needs("nevergrad", "cma")
def test_external_baselines_join_by_name():
    names = [s.name for s in make_baseline_strategies(["Baseline_NGOpt", "Baseline_pycma_BIPOP", "Other"])]
    assert names[3:] == ["Baseline_pycma_BIPOP", "Baseline_NGOpt"]
    assert len(set(EXTERNAL_BASELINE_NAMES)) == 7


def test_names_derive_from_the_classes():
    specs = make_external_baseline_strategies()
    assert tuple(s.name for s in specs) == EXTERNAL_BASELINE_NAMES
    assert all(s.name == f"Baseline_{s.strategy_class.who}" for s in specs)


def test_missing_extra_fails_when_the_spec_is_built(monkeypatch):
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None if name == "optuna" else real(name, *a))
    with pytest.raises(ImportError, match="--extra baselines"):
        make_baseline_strategies(["Baseline_Optuna_TPE"])
    make_baseline_strategies(["Baseline_pycma_IPOP"] if real("cma") else None)  # others unaffected


def test_baseline_name_without_baselines_flag_is_an_error_with_a_hint():
    harness = BenchmarkHarness(HarnessConfig(mode="quick", strategies=["Baseline_NGOpt"]))
    with pytest.raises(ValueError, match="--baselines"):
        harness.get_strategies()


@_needs("optuna")
def test_harness_selects_external_baseline_through_the_filter():
    config = HarnessConfig(mode="quick", include_baselines=True, strategies=["Baseline_Optuna_TPE"])
    names = [s.name for s in BenchmarkHarness(config).get_strategies()]
    assert names == ["Baseline_Optuna_TPE"]


@_needs("cma")
def test_harness_runs_an_external_baseline_end_to_end():
    config = HarnessConfig(
        mode="quick",
        budget=20,
        reps=1,
        seed=42,
        problems=["Rosenbrock_2D"],
        strategies=["Baseline_pycma_BIPOP"],
        include_baselines=True,
        timeout_per_run=30.0,
    )
    result = BenchmarkHarness(config).run(verbose=False)
    (run,) = result.problem_strategy_results[0].runs
    assert run.error is None
    assert run.evaluations_used == 20
