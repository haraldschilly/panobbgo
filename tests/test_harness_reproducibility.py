# -*- coding: utf8 -*-
"""Two identical harness invocations must give identical AOCC values.

This is the property every paired A/B in the harness rests on
(planning/DISCOVERY_2026-09-09.md §2).  Needs the IOH worker venv
(``cd tools/ioh_worker && uv sync``); skipped otherwise.
"""

import dataclasses

import pytest

from panobbgo.lib.ioh_wrapper import worker_available

pytestmark = pytest.mark.skipif(not worker_available(), reason="IOH worker venv not installed")


def _run():
    from panobbgo.harness_ioh import make_ioh_strategies, make_quick_battery, run_ioh_harness

    specs = [s for s in make_ioh_strategies() if s.name == "Rewarding_Restart"]
    battery = dataclasses.replace(make_quick_battery(), instances=(1,))
    res = run_ioh_harness(specs, battery, base_seed=42, progress=False, sync_eval=True)
    return [(r.aocc, r.n_evals, tuple(r.trace_fx)) for r in res.runs]


def test_same_seed_same_aocc():
    a, b = _run(), _run()
    assert a == b


def test_config_overrides_reach_the_constructor():
    from panobbgo.benchmark import StrategySpec
    from panobbgo.heuristics import Random
    from panobbgo.lib.classic import Rosenbrock
    from panobbgo.strategies import StrategyRoundRobin

    spec = StrategySpec(
        name="x",
        strategy_class=StrategyRoundRobin,
        heuristics=[(Random, {})],
        config_overrides={"dask_n_workers": 1, "max_eval": 17},
    )
    s = spec.create_strategy(Rosenbrock(dim=2), seed=3)
    assert s.config.max_eval == 17
    assert s._n_processes == 1
