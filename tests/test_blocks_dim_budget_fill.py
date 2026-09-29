# -*- coding: utf8 -*-
# Copyright 2012-2026 Harald Schilly <harald.schilly@gmail.com>
"""DISCOVERY §72: the block scheduler's non-oracle regime gate and first-round fill.

``regime_gate="dim-budget"`` applies only the dimension / budget rows of
``REGIME_TABLE_V1``, and only while the workers do not exceed the kept arms'
serial generation size.  ``first_round_fill`` fills the workers the arms
leave idle before the first result.  Both are on in the headline spec
``Blocks_warm_CMAES_JSO``: the fill since §72, the gate since §73 (opt-in
as ``Blocks_warm_CMAES_JSO_dimbudget`` from §72.8 until the fresh-seed
confirmation).
"""

from __future__ import annotations

from typing import Any, List

import numpy as np
import pytest

from panobbgo.core import Analyzer
from panobbgo.lib.classic import Rosenbrock
from panobbgo.strategies import StrategyBlockBandit
from panobbgo.strategies.blocks import DIM_BUDGET_GATE, REGIME_TABLE_V1, is_dim_budget_row


class _Recorder(Analyzer):
    """Keeps every delivered result, in delivery order."""

    def __init__(self, strategy: Any) -> None:
        Analyzer.__init__(self, strategy, name="Recorder")
        self.seen: List[Any] = []

    def on_new_results(self, results: List[Any]) -> None:
        self.seen.extend(results)


class _Constrained(Rosenbrock):
    """Rosenbrock with one (always satisfied) constraint."""

    def eval_constraints(self, x):
        return np.array([-1.0])


def _portfolio(
    *,
    dim: int,
    max_eval: int,
    q: int,
    seed: int = 3,
    problem: Any = None,
    virtual: bool = True,
    **overrides: Any,
) -> Any:
    """The headline spec's strategy (CMA-ES + jSO, archive warm starts), with ``overrides`` on its config."""
    from panobbgo.harness_ioh import make_ioh_strategies
    from panobbgo.virtual_clock import VirtualSpec

    spec = {sp.name: sp for sp in make_ioh_strategies()}["Blocks_warm_CMAES_JSO"]
    config = {**spec.config_overrides, **overrides}
    s = spec.strategy_class(problem or Rosenbrock(dim=dim), parse_args=False, testing_mode=True, seed=seed, **config)
    s.config.max_eval = max_eval
    s.config.stop_on_convergence = False
    if virtual:
        VirtualSpec(workers=q, duration="lognormal").apply(s)
    else:
        s.config.sync_evaluation = True
        s.config.evaluation_method = "threaded"
    for h, hk in spec.heuristics:
        s.add(h, **hk)
    for a, ak in spec.analyzers:
        s.add_analyzer(a(s, **ak))
    rec = _Recorder(s)
    s.add_analyzer(rec)
    s.rec = rec
    return s


# -- the dim/budget rows -------------------------------------------------------


def test_dim_budget_rows_of_the_table():
    rows = [key for key in REGIME_TABLE_V1 if is_dim_budget_row(key)]
    assert rows == [(None, "dim >= 10", "bpd <= 500", None), (None, "dim <= 5", "bpd <= 200", None)]
    assert not is_dim_budget_row((None, None, None, None))  # the catch-all says nothing about dim/budget
    assert not is_dim_budget_row(("bounded", "dim <= 5", None, None))  # needs the noise class
    assert not is_dim_budget_row((None, None, None, True))  # the constrained row


def test_constructor_accepts_dim_budget():
    s = StrategyBlockBandit(Rosenbrock(dim=2), parse_args=False, testing_mode=True, regime_gate=DIM_BUDGET_GATE)
    assert s.regime_gate == "dim-budget" and s._gate_class is None and s._gate_dim_budget
    s._cleanup()


@pytest.mark.parametrize(
    "dim, max_eval, q, constrained, applied, enabled",
    [
        # unconstrained, d >= 10, <= 500·d, q <= λ_default = 10 (d = 10): CMA-ES alone
        (10, 1000, 4, False, True, ["CMAES"]),
        (10, 200, 10, False, True, ["CMAES"]),
        # more workers than λ_default: the floor raises λ, the row is not applied
        (10, 1000, 16, False, False, ["CMAES", "JSO"]),
        # above 500·d at d = 10: the catch-all row, not a dim/budget row
        (10, 6000, 4, False, False, ["CMAES", "JSO"]),
        # d <= 5 at <= 200·d: a dim/budget row that keeps both arms
        (5, 500, 4, False, True, ["CMAES", "JSO"]),
        # d <= 5 at 500·d: the catch-all, nothing is disabled
        (2, 1000, 4, False, False, ["CMAES", "JSO"]),
        # constrained: the constrained row wins the lookup, and it is not a dim/budget row
        (10, 1000, 4, True, False, ["CMAES", "JSO"]),
    ],
)
def test_dim_budget_gate_decision(dim, max_eval, q, constrained, applied, enabled):
    problem = _Constrained(dim=dim) if constrained else Rosenbrock(dim=dim)
    s = _portfolio(dim=dim, max_eval=max_eval, q=q, problem=problem, regime_gate="dim-budget")
    s.config.max_eval = 40  # a short run: only the decision is under test
    s._max_eval = lambda: max_eval  # the budget the gate reads (bpd), independent of the short run
    s.start()
    d = s.regime_decision
    assert d is not None and d["gate"] == "dim-budget" and d["noise_class"] is None
    assert (d["dim"], d["workers"], d["constrained"]) == (dim, q, constrained)
    assert d["applied"] is applied, d["reason"]
    assert d["enabled"] == enabled
    assert {r.who.split(":")[0] for r in s.rec.seen} <= set(enabled)
    if applied and enabled == ["CMAES"]:
        assert {b["owner"] for b in s._blocks} == {"CMAES"}


def test_dim_budget_gate_that_keeps_both_arms_is_bit_identical_to_no_gate():
    """d = 5 at 100·d (both arms kept) and d = 10, q = 16 (not applied) change nothing."""
    for dim, max_eval, q in ((5, 500, 4), (10, 300, 16)):
        runs = []
        for gate in (None, "dim-budget"):
            s = _portfolio(dim=dim, max_eval=max_eval, q=q, regime_gate=gate, first_round_fill=False)
            s.start()
            runs.append(s.rec.seen)
        a, b = runs
        assert [tuple(r.x) for r in a] == [tuple(r.x) for r in b]
        assert [r.fx for r in a] == [r.fx for r in b]


def test_dim_budget_gate_matches_the_oracle_at_d10_q4():
    """Where the row applies, the non-oracle gate is the oracle gate (noiseless) run for run."""
    runs = []
    for gate in ("oracle:clean", "dim-budget"):
        s = _portfolio(dim=10, max_eval=300, q=4, regime_gate=gate, first_round_fill=False)
        s._max_eval = lambda: 1000  # bpd 100: the d >= 10 row; the run itself is short
        s.start()
        runs.append(s.rec.seen)
    a, b = runs
    assert [tuple(r.x) for r in a] == [tuple(r.x) for r in b]


def test_serial_generation():
    from panobbgo.heuristics import CMAES, JSO

    s = _portfolio(dim=5, max_eval=700, q=64)
    s.start()
    cma = next(h for h in s.heuristics if isinstance(h, CMAES))
    jso = next(h for h in s.heuristics if isinstance(h, JSO))
    assert cma._lam == 64  # the worker floor raised it (budget cap 700 // 10 = 70)
    assert StrategyBlockBandit._serial_generation(cma) == 8  # 4 + 3 ln 5
    np_jso = StrategyBlockBandit._serial_generation(jso)
    assert np_jso is not None and 0 < np_jso <= jso.NP_init
    assert StrategyBlockBandit._serial_generation(object()) is None  # type: ignore[arg-type]


# -- the first-round fill ------------------------------------------------------


def _busy_at_zero(seen: List[Any]) -> int:
    return sum(1 for r in seen if r.t_dispatch == 0.0)


def test_first_round_fill_fills_every_worker_at_the_start():
    """d = 2, 200 evaluations, q = 64: the arms' first generations fill 26 workers, the fill all 64."""
    off = _portfolio(dim=2, max_eval=200, q=64, first_round_fill=False)
    off.start()
    on = _portfolio(dim=2, max_eval=200, q=64, first_round_fill=True)
    on.start()
    assert len(off.rec.seen) == len(on.rec.seen) == 200
    assert _busy_at_zero(off.rec.seen) < 64
    assert _busy_at_zero(on.rec.seen) == 64
    design = [r for r in on.rec.seen if r.who == StrategyBlockBandit.INITIAL_DESIGN_WHO]
    assert len(design) == on._initial_design_n == 64 - _busy_at_zero(off.rec.seen)
    assert all(r.t_dispatch == 0.0 for r in design)  # the first round only
    # one Latin hypercube: one point per stratum in every coordinate
    box = np.asarray(on.problem.box.box, dtype=float)
    u = (np.array([r.x for r in design]) - box[:, 0]) / (box[:, 1] - box[:, 0])
    for j in range(u.shape[1]):
        assert sorted(np.floor(u[:, j] * len(design)).astype(int)) == list(range(len(design)))
    assert on._design_queue == []
    # no arm is credited with the design's points
    assert StrategyBlockBandit.INITIAL_DESIGN_WHO not in on._arm_best


def test_first_round_fill_is_deterministic():
    runs = []
    for _ in range(2):
        s = _portfolio(dim=2, max_eval=200, q=64, first_round_fill=True)
        s.start()
        runs.append(s.rec.seen)
    assert [tuple(r.x) for r in runs[0]] == [tuple(r.x) for r in runs[1]]
    assert [r.t_complete for r in runs[0]] == [r.t_complete for r in runs[1]]


@pytest.mark.parametrize("q", [4, 16])
def test_first_round_fill_is_a_no_op_when_the_arms_fill_the_workers(q):
    """d = 2, q <= 16: CMA-ES's first generation (λ = max(6, q)) and jSO fill every worker already."""
    runs = []
    for fill in (False, True):
        s = _portfolio(dim=2, max_eval=200, q=q, first_round_fill=fill)
        s.start()
        runs.append(s)
    a, b = runs
    assert b._initial_design_n == 0
    assert [tuple(r.x) for r in a.rec.seen] == [tuple(r.x) for r in b.rec.seen]
    assert [r.fx for r in a.rec.seen] == [r.fx for r in b.rec.seen]


def test_first_round_fill_needs_a_request_cap():
    """Synchronous (no request cap): nothing is filled, the run is the unfilled one."""
    runs = []
    for fill in (False, True):
        s = _portfolio(dim=2, max_eval=60, q=1, virtual=False, first_round_fill=fill)
        s.start()
        runs.append(s)
    a, b = runs
    assert b._initial_design_n == 0
    assert [tuple(r.x) for r in a.rec.seen] == [tuple(r.x) for r in b.rec.seen]


def test_first_round_fill_respects_the_budget():
    """A budget smaller than q: the fill stops at max_eval."""
    s = _portfolio(dim=2, max_eval=40, q=64, first_round_fill=True)
    s.start()
    assert len(s.rec.seen) == 40
    assert _busy_at_zero(s.rec.seen) == 40


# -- the registry ---------------------------------------------------------------


def test_headline_spec_defaults():
    from panobbgo.harness_ioh import make_ioh_strategies, make_trust_region_strategies

    specs = {sp.name: sp for sp in make_ioh_strategies() + make_trust_region_strategies()}
    blocks = specs["Blocks_warm_CMAES_JSO"].config_overrides
    # first-round fill (§72) and the dim/budget gate (§73: the pre-declared rule passed) on by default
    assert blocks["regime_gate"] == "dim-budget" and blocks["first_round_fill"] is True
    # the opt-in variant is gone: it would be the headline spec under another name
    assert "Blocks_warm_CMAES_JSO_dimbudget" not in specs
    # RegimeGate_oracle: the portfolio's config but the gate
    assert specs["RegimeGate_oracle"].config_overrides == {**blocks, "regime_gate": "oracle"}
    # the TRQ portfolio copies the config, gate included; the gate never applies there (test below)
    assert specs["Blocks_warm_CMAES_JSO_TRQ"].config_overrides == blocks


def test_ioh_cli_no_longer_knows_the_gated_variant(capsys):
    import argparse
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "scripts" / "ioh_benchmark.py"
    mspec = importlib.util.spec_from_file_location("ioh_benchmark_blocks_variant", path)
    assert mspec is not None and mspec.loader is not None
    cli = importlib.util.module_from_spec(mspec)
    mspec.loader.exec_module(cli)
    base = dict(legacy=False, standard=True, full=False, baselines=False)
    wanted = ["Blocks_warm_CMAES_JSO", "Blocks_warm_CMAES_JSO_dimbudget"]
    picked = cli._resolve_strategies(argparse.Namespace(strategies=wanted, **base))
    assert [s.name for s in picked] == ["Blocks_warm_CMAES_JSO"]
    assert picked[0].config_overrides["regime_gate"] == "dim-budget"
    assert "Blocks_warm_CMAES_JSO_dimbudget" in capsys.readouterr().err  # reported as unknown


def test_the_shipped_headline_spec_gates_at_d10_q4():
    """No override: the shipped config runs CMA-ES alone where §72's row binds."""
    s = _portfolio(dim=10, max_eval=40, q=4)
    s._max_eval = lambda: 1000  # 100·d, the cell the §73 rule judged
    s.start()
    d = s.regime_decision
    assert d is not None and d["gate"] == "dim-budget" and d["applied"] is True
    assert d["enabled"] == ["CMAES"] and d["disabled"] == ["JSO"]
    assert {r.who.split(":")[0] for r in s.rec.seen} == {"CMAES"}


def test_dim_budget_gate_is_not_applied_with_an_arm_outside_the_table():
    """The TRQ portfolio at d10/q4: the row would switch the TR arm off, so the gate stays off."""
    from panobbgo.harness_ioh import make_trust_region_strategies
    from panobbgo.virtual_clock import VirtualSpec

    spec = make_trust_region_strategies(["Blocks_warm_CMAES_JSO_TRQ"])[0]
    config = {**spec.config_overrides, "regime_gate": "dim-budget"}
    s = spec.strategy_class(Rosenbrock(dim=10), parse_args=False, testing_mode=True, seed=2, **config)
    s.config.max_eval = 60
    s.config.stop_on_convergence = False
    VirtualSpec(workers=4, duration="lognormal").apply(s)
    for h, hk in spec.heuristics:
        s.add(h, **hk)
    for a, ak in spec.analyzers:
        s.add_analyzer(a(s, **ak))
    s._max_eval = lambda: 1000  # bpd 100: the d >= 10 row
    s.start()
    d = s.regime_decision
    assert d is not None and d["row"] == (None, "dim >= 10", "bpd <= 500", None)
    assert d["applied"] is False and "play no role" in d["reason"]
    assert d["disabled"] == []


def _trq_portfolio(*, dim: int, max_eval: int, q: int, seed: int, **overrides: Any) -> Any:
    """``Blocks_warm_CMAES_JSO_TRQ`` as shipped, with ``overrides`` on its config."""
    from panobbgo.harness_ioh import make_trust_region_strategies
    from panobbgo.virtual_clock import VirtualSpec

    spec = make_trust_region_strategies(["Blocks_warm_CMAES_JSO_TRQ"])[0]
    config = {**spec.config_overrides, **overrides}
    s = spec.strategy_class(Rosenbrock(dim=dim), parse_args=False, testing_mode=True, seed=seed, **config)
    s.config.max_eval = max_eval
    s.config.stop_on_convergence = False
    VirtualSpec(workers=q, duration="lognormal").apply(s)
    for h, hk in spec.heuristics:
        s.add(h, **hk)
    for a, ak in spec.analyzers:
        s.add_analyzer(a(s, **ak))
    rec = _Recorder(s)
    s.add_analyzer(rec)
    s.rec = rec
    return s


@pytest.mark.parametrize("q", [1, 4])
def test_trq_portfolio_is_unchanged_by_the_gate_default(q):
    """§73: ``Blocks_warm_CMAES_JSO_TRQ`` inherits ``regime_gate="dim-budget"`` from the portfolio, but at
    d10/100·d, where the gate binds for the headline spec, it keeps every arm and runs the ungated config's
    trajectory point for point."""
    runs = []
    for overrides in ({}, {"regime_gate": None}):
        s = _trq_portfolio(dim=10, max_eval=1000, q=q, seed=4, **overrides)
        s.start()
        runs.append(s)
    shipped, ungated = runs
    assert shipped.regime_gate == "dim-budget" and ungated.regime_decision is None
    d = shipped.regime_decision
    assert d is not None and d["applied"] is False and d["disabled"] == [] and "play no role" in d["reason"]
    assert len(shipped.rec.seen) == len(ungated.rec.seen) == 1000
    assert [tuple(r.x) for r in shipped.rec.seen] == [tuple(r.x) for r in ungated.rec.seen]
    assert [r.fx for r in shipped.rec.seen] == [r.fx for r in ungated.rec.seen]
    assert [r.who for r in shipped.rec.seen] == [r.who for r in ungated.rec.seen]
    assert [r.t_complete for r in shipped.rec.seen] == [r.t_complete for r in ungated.rec.seen]


# -- review of #390: bit-identity at the real budgets, preload, the design's stream --


@pytest.mark.parametrize("dim, q", [(10, 16), (10, 64), (5, 16)])
def test_shipped_and_ungated_specs_are_bit_identical_to_the_old_where_neither_option_binds(dim, q):
    """At 100·d (the real budget: a shorter one changes CMA-ES's λ cap) the shipped config (gated
    since §73) and the §72 one (fill, no gate) both reproduce the config before §72."""
    runs = []
    for overrides in ({}, {"regime_gate": None}, {"regime_gate": None, "first_round_fill": False}):
        s = _portfolio(dim=dim, max_eval=100 * dim, q=q, seed=5, **overrides)
        s.start()
        runs.append(s)
    shipped, ungated, old = runs
    assert shipped.regime_decision is not None and shipped.regime_decision["applied"] is False
    assert ungated.regime_decision is None
    for new in (shipped, ungated):
        assert new._initial_design_n == 0 and new._filled == {}
        assert len(new.rec.seen) == len(old.rec.seen) == 100 * dim
        assert [tuple(r.x) for r in new.rec.seen] == [tuple(r.x) for r in old.rec.seen]
        assert [r.fx for r in new.rec.seen] == [r.fx for r in old.rec.seen]
        assert [r.t_complete for r in new.rec.seen] == [r.t_complete for r in old.rec.seen]


def test_a_preloaded_run_never_fills():
    """Results before the first pass (a resume, ``preload_results``): the first round is over."""
    from panobbgo.lib.families import Family
    from panobbgo.selector_data import evaluate_probe, probe_design

    problem = Family("ellipsoid", dim=2, seed=3)
    probe = evaluate_probe(problem, probe_design(2, 10, 0))
    s = _portfolio(dim=2, max_eval=200, q=64, problem=problem, first_round_fill=True)
    s.preload_results(probe)
    s.start()
    assert s._initial_design_n == 0 and s._design_queue is None
    assert all(r.who != StrategyBlockBandit.INITIAL_DESIGN_WHO for r in s.rec.seen)


def test_the_design_draws_from_its_own_keyed_stream():
    """The Latin hypercube leaves the strategy-level stream untouched and uses the keyed one."""
    from panobbgo.core import keyed_rng, rng_stream_key

    s = _portfolio(dim=2, max_eval=200, q=64, seed=9, first_round_fill=True)
    before = s.rng.bit_generator.state
    design = s._latin_hypercube(8)
    assert s.rng.bit_generator.state == before
    rng = keyed_rng(s.seed, rng_stream_key("first_round_fill", 0))
    box = np.asarray(s.problem.box.box, dtype=float)
    u = (np.argsort(rng.random((8, 2)), axis=0) + rng.random((8, 2))) / 8
    expected = np.array([s.problem.project(box[:, 0] + ui * (box[:, 1] - box[:, 0])) for ui in u])
    np.testing.assert_array_equal(np.array([p.x for p in design]), expected)
    s._cleanup()


def test_status_reports_the_fill():
    s = _portfolio(dim=2, max_eval=200, q=64, first_round_fill=True)
    s.start()
    assert "design 38" in s._get_status_info()["first_round_fill"]
