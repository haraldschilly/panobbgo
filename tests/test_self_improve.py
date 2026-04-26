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
Tests for the self-improvement loop driver (:mod:`panobbgo.self_improve`).

Covers:

* :class:`MutationRule` validation
* :class:`MutationCatalog` sampling and applicable-rule filtering
* :func:`apply_mutation` correctness and immutability
* :class:`LoopConfig` validation
* :class:`SelfImprover` end-to-end with a faked harness
* The anti-cherry-pick guard (§6.3 of the plan):

  - Guard does not fire when ``guard_interval == 0``.
  - Guard fires every K iterations otherwise.
  - Guard refreshes ``last_validated_score`` when within tolerance.
  - Guard rolls back the ladder when re-measure drifts beyond
    ``guard_eps_ladder``.
  - Guard cannot pop the seed (always falls back to it).
* Ledger writing (mixed iteration + guard records) and :func:`load_ledger`.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pytest

from panobbgo.benchmark import StrategySpec
from panobbgo.harness import (
    HarnessConfig,
    HarnessResult,
    ProblemStrategyResult,
    RunRecord,
)
from panobbgo.self_improve import (
    LadderEntry,
    LoopConfig,
    LoopGuardRecord,
    LoopIterationRecord,
    MutationCatalog,
    MutationProposal,
    MutationRule,
    SelfImprover,
    apply_mutation,
    default_catalog,
    load_ledger,
)


# ===========================================================================
# Test fixtures: dummy heuristic / analyzer / strategy classes
# ===========================================================================


class _DummyHeuristicA:
    """Fake heuristic class used as a mutation target."""

    pass


class _DummyHeuristicB:
    """Fake heuristic class with a different name."""

    pass


class _DummyAnalyzerC:
    """Fake analyzer class used as a mutation target."""

    pass


class _DummyStrategy:
    """Strategy stand-in (StrategySpec only references the class object)."""

    pass


def _make_specs() -> List[StrategySpec]:
    """Two strategies, each with a heuristic and an analyzer carrying knobs."""
    return [
        StrategySpec(
            name="StratX",
            strategy_class=_DummyStrategy,
            heuristics=[
                (_DummyHeuristicA, {"radius": 0.1}),
                (_DummyHeuristicB, {"sigma0": 0.3}),
            ],
            analyzers=[(_DummyAnalyzerC, {"update_interval": 20})],
        ),
        StrategySpec(
            name="StratY",
            strategy_class=_DummyStrategy,
            heuristics=[(_DummyHeuristicA, {"radius": 0.05})],
            analyzers=[],
        ),
    ]


# ===========================================================================
# MutationRule
# ===========================================================================


class TestMutationRule:
    def test_constructs_with_valid_kind(self):
        rule = MutationRule(
            strategy_pattern="",
            class_name="X",
            param_name="p",
            kind="log_uniform_perturb",
            bounds=(0.0, 1.0),
        )
        assert rule.kind == "log_uniform_perturb"

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError, match="Unknown mutation kind"):
            MutationRule(
                strategy_pattern="",
                class_name="X",
                param_name="p",
                kind="weird",
                bounds=(0.0, 1.0),
            )

    def test_inverted_bounds_raise(self):
        with pytest.raises(ValueError, match="bounds not ordered"):
            MutationRule(
                strategy_pattern="",
                class_name="X",
                param_name="p",
                kind="float_uniform",
                bounds=(1.0, 0.0),
            )

    def test_non_positive_probability_raises(self):
        with pytest.raises(ValueError, match="probability"):
            MutationRule(
                strategy_pattern="",
                class_name="X",
                param_name="p",
                kind="float_uniform",
                bounds=(0.0, 1.0),
                probability=0.0,
            )


# ===========================================================================
# MutationCatalog: sampling, applicable rules, mutation kinds
# ===========================================================================


class TestMutationCatalog:
    def test_empty_catalog_raises(self):
        with pytest.raises(ValueError):
            MutationCatalog([])

    def test_applicable_rules_filters_by_class(self):
        rules = [
            MutationRule(
                strategy_pattern="",
                class_name="_DummyHeuristicA",
                param_name="radius",
                kind="log_uniform_perturb",
                bounds=(0.005, 0.5),
            ),
            MutationRule(
                strategy_pattern="",
                class_name="_DummyHeuristicB",
                param_name="sigma0",
                kind="log_uniform_perturb",
                bounds=(0.05, 1.0),
            ),
            MutationRule(
                strategy_pattern="",
                class_name="DoesNotExist",
                param_name="foo",
                kind="float_uniform",
                bounds=(0.0, 1.0),
            ),
        ]
        cat = MutationCatalog(rules)
        applicable = cat.applicable_rules(_make_specs())
        names = {r.class_name for r, _ in applicable}
        assert names == {"_DummyHeuristicA", "_DummyHeuristicB"}

    def test_applicable_rules_skips_missing_kwarg(self):
        """Rule for an existing class but a missing kwarg should be skipped."""
        rules = [
            MutationRule(
                strategy_pattern="",
                class_name="_DummyHeuristicA",
                param_name="not_present",
                kind="float_uniform",
                bounds=(0.0, 1.0),
            ),
        ]
        cat = MutationCatalog(rules)
        assert cat.applicable_rules(_make_specs()) == []

    def test_strategy_pattern_filters(self):
        rules = [
            MutationRule(
                strategy_pattern="StratY",
                class_name="_DummyHeuristicA",
                param_name="radius",
                kind="log_uniform_perturb",
                bounds=(0.005, 0.5),
            ),
        ]
        cat = MutationCatalog(rules)
        applicable = cat.applicable_rules(_make_specs())
        assert len(applicable) == 1
        rule, hits = applicable[0]
        # Only StratY should match — single hit.
        assert len(hits) == 1

    def test_sample_returns_none_when_no_applicable_rules(self):
        rules = [
            MutationRule(
                strategy_pattern="",
                class_name="DoesNotExist",
                param_name="foo",
                kind="float_uniform",
                bounds=(0.0, 1.0),
            ),
        ]
        cat = MutationCatalog(rules)
        rng = np.random.default_rng(0)
        assert cat.sample(rng, _make_specs()) is None

    def test_sample_log_uniform_stays_within_bounds(self):
        rule = MutationRule(
            strategy_pattern="",
            class_name="_DummyHeuristicA",
            param_name="radius",
            kind="log_uniform_perturb",
            bounds=(0.005, 0.5),
            log_step=0.5,
        )
        cat = MutationCatalog([rule])
        rng = np.random.default_rng(7)
        for _ in range(50):
            prop = cat.sample(rng, _make_specs())
            assert prop is not None
            assert 0.005 <= prop.new_value <= 0.5

    def test_sample_integer_add_clamps_to_bounds(self):
        rule = MutationRule(
            strategy_pattern="",
            class_name="_DummyAnalyzerC",
            param_name="update_interval",
            kind="integer_add",
            bounds=(5, 60),
            delta_choices=(-100, 100),
        )
        cat = MutationCatalog([rule])
        rng = np.random.default_rng(0)
        for _ in range(20):
            prop = cat.sample(rng, _make_specs())
            assert prop is not None
            assert 5 <= prop.new_value <= 60
            assert isinstance(prop.new_value, int)

    def test_sample_float_uniform(self):
        rule = MutationRule(
            strategy_pattern="",
            class_name="_DummyHeuristicA",
            param_name="radius",
            kind="float_uniform",
            bounds=(0.0, 1.0),
            low=0.2,
            high=0.4,
        )
        cat = MutationCatalog([rule])
        rng = np.random.default_rng(1)
        for _ in range(30):
            prop = cat.sample(rng, _make_specs())
            assert prop is not None
            assert 0.2 <= prop.new_value <= 0.4

    def test_log_uniform_requires_positive_old(self):
        rule = MutationRule(
            strategy_pattern="",
            class_name="_DummyHeuristicA",
            param_name="radius",
            kind="log_uniform_perturb",
            bounds=(0.0, 1.0),
        )
        cat = MutationCatalog([rule])
        rng = np.random.default_rng(0)
        bad_specs = [
            StrategySpec(
                name="S",
                strategy_class=_DummyStrategy,
                heuristics=[(_DummyHeuristicA, {"radius": 0.0})],
            )
        ]
        with pytest.raises(ValueError, match="positive"):
            cat.sample(rng, bad_specs)

    def test_default_catalog_has_rules(self):
        cat = default_catalog()
        assert len(cat.rules) >= 5


# ===========================================================================
# apply_mutation
# ===========================================================================


class TestApplyMutation:
    def test_applies_to_named_strategy(self):
        specs = _make_specs()
        proposal = MutationProposal(
            strategy_name="StratX",
            class_name="_DummyHeuristicA",
            param_name="radius",
            old_value=0.1,
            new_value=0.2,
            rule_kind="log_uniform_perturb",
            rationale="test",
        )
        out = apply_mutation(specs, proposal)
        # New StratX has updated radius.
        new_x = next(s for s in out if s.name == "StratX")
        assert new_x.heuristics[0][1]["radius"] == 0.2
        # Other strategy untouched.
        new_y = next(s for s in out if s.name == "StratY")
        assert new_y.heuristics[0][1]["radius"] == 0.05

    def test_input_specs_not_mutated(self):
        specs = _make_specs()
        original_x_radius = specs[0].heuristics[0][1]["radius"]
        proposal = MutationProposal(
            strategy_name="StratX",
            class_name="_DummyHeuristicA",
            param_name="radius",
            old_value=0.1,
            new_value=0.42,
            rule_kind="log_uniform_perturb",
            rationale="t",
        )
        apply_mutation(specs, proposal)
        # Original spec object unchanged.
        assert specs[0].heuristics[0][1]["radius"] == original_x_radius

    def test_applies_to_analyzer(self):
        specs = _make_specs()
        proposal = MutationProposal(
            strategy_name="StratX",
            class_name="_DummyAnalyzerC",
            param_name="update_interval",
            old_value=20,
            new_value=30,
            rule_kind="integer_add",
            rationale="t",
        )
        out = apply_mutation(specs, proposal)
        new_x = next(s for s in out if s.name == "StratX")
        assert new_x.analyzers[0][1]["update_interval"] == 30

    def test_unknown_strategy_raises(self):
        proposal = MutationProposal(
            strategy_name="Nope",
            class_name="_DummyHeuristicA",
            param_name="radius",
            old_value=0.1,
            new_value=0.2,
            rule_kind="log_uniform_perturb",
            rationale="t",
        )
        with pytest.raises(ValueError, match="not in the input spec list"):
            apply_mutation(_make_specs(), proposal)

    def test_unknown_param_raises(self):
        proposal = MutationProposal(
            strategy_name="StratX",
            class_name="_DummyHeuristicA",
            param_name="ghost",
            old_value=0.1,
            new_value=0.2,
            rule_kind="log_uniform_perturb",
            rationale="t",
        )
        with pytest.raises(ValueError, match="not found"):
            apply_mutation(_make_specs(), proposal)


# ===========================================================================
# LoopConfig validation
# ===========================================================================


class TestLoopConfig:
    def test_defaults_are_valid(self):
        cfg = LoopConfig()
        assert cfg.iterations == 5
        assert cfg.guard_interval == 0  # disabled by default

    def test_negative_iterations_raise(self):
        with pytest.raises(ValueError):
            LoopConfig(iterations=-1)

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            LoopConfig(mode="extreme")

    def test_negative_guard_interval_raises(self):
        with pytest.raises(ValueError, match="guard_interval"):
            LoopConfig(guard_interval=-1)

    def test_negative_guard_eps_raises(self):
        with pytest.raises(ValueError, match="guard_eps_ladder"):
            LoopConfig(guard_eps_ladder=-0.001)


# ===========================================================================
# Fake harness used to drive SelfImprover deterministically
# ===========================================================================


def _fake_run_record(score: float, budget: int = 100) -> RunRecord:
    """A run that hits exactly at the eval implied by ``score``.

    ``score`` is the fraction ``1 - (hit - 1)/budget`` we want, so
    ``hit = budget * (1 - score) + 1``.  When score ≤ 0 the run "fails"
    (no convergence).
    """
    if score <= 0.0:
        return RunRecord(
            problem_name="P",
            problem_dim=2,
            strategy_name="S",
            rep=0,
            seed=0,
            budget=budget,
            evaluations_used=budget,
            best_fx=999.0,
            f_opt=0.0,
            func_distance=1.0,
            tolerance=0.1,
            success=False,
            convergence=[],
            heuristic_counts={},
            duration=0.01,
        )
    hit = max(1, int(round(budget * (1.0 - score) + 1)))
    from panobbgo.harness import ConvergencePoint

    return RunRecord(
        problem_name="P",
        problem_dim=2,
        strategy_name="S",
        rep=0,
        seed=0,
        budget=budget,
        evaluations_used=budget,
        best_fx=0.0,
        f_opt=0.0,
        func_distance=0.0,
        tolerance=0.1,
        success=True,
        convergence=[ConvergencePoint(eval_idx=hit, fx=0.0, func_distance=0.0)],
        heuristic_counts={},
        duration=0.01,
    )


def _fake_psr(prob: str, strat: str, scores: List[float], budget: int = 100) -> ProblemStrategyResult:
    runs = []
    for s in scores:
        r = _fake_run_record(s, budget=budget)
        r.problem_name = prob
        r.strategy_name = strat
        runs.append(r)
    psr = ProblemStrategyResult(
        problem_name=prob,
        problem_dim=2,
        strategy_name=strat,
        f_opt=0.0,
        tolerance=0.1,
        budget=budget,
        runs=runs,
    )
    psr.compute_metrics()
    return psr


def _fake_harness_result(
    score: float,
    strategy_names: Sequence[str],
    problem_names: Sequence[str] = ("P1",),
    n_reps: int = 5,
) -> HarnessResult:
    """A HarnessResult whose composite_score equals ``score``."""
    psrs: List[ProblemStrategyResult] = []
    for prob in problem_names:
        for strat in strategy_names:
            psrs.append(_fake_psr(prob, strat, [score] * n_reps))
    composite = float(np.mean([p.score for p in psrs])) if psrs else 0.0
    return HarnessResult(
        config=HarnessConfig(mode="quick"),
        timestamp="2026-01-01T00:00:00+00:00",
        total_runs=sum(len(p.runs) for p in psrs),
        total_duration=0.0,
        problem_strategy_results=psrs,
        composite_score=composite,
    )


@dataclass
class _FakeHarness:
    """Stand-in for :class:`BenchmarkHarness` in :class:`SelfImprover` tests.

    The factory pattern in :class:`SelfImprover` lets us swap this in via
    ``_harness_factory``.  Each call to ``run`` returns a deterministic
    score chosen by ``score_fn(config) -> float`` so we can test the
    accept/reject and guard paths precisely.
    """

    config: HarnessConfig
    score_fn: Callable[[HarnessConfig], float] = field(default=lambda c: 0.5)
    strategy_names: List[str] = field(default_factory=lambda: ["S"])
    call_log: List[Dict[str, Any]] = field(default_factory=list)

    def run(self, verbose: bool = False) -> HarnessResult:
        score = float(self.score_fn(self.config))
        self.call_log.append(
            {
                "score": score,
                "randomize_iteration": self.config.randomize_iteration,
                "n_strategies": (
                    len(self.config.strategies_override) if self.config.strategies_override is not None else 0
                ),
            }
        )
        return _fake_harness_result(score, self.strategy_names)

    def get_strategies(self) -> List[StrategySpec]:
        # Returned only when the loop has no explicit seed_strategies.
        return _make_specs()


def _make_factory(
    score_fn: Callable[[HarnessConfig], float],
    call_log: Optional[List[Dict[str, Any]]] = None,
) -> Callable[[HarnessConfig], _FakeHarness]:
    log = call_log if call_log is not None else []
    strat_names = [s.name for s in _make_specs()]

    def factory(config: HarnessConfig) -> _FakeHarness:
        return _FakeHarness(
            config=config,
            score_fn=score_fn,
            strategy_names=strat_names,
            call_log=log,
        )

    return factory


# ===========================================================================
# SelfImprover end-to-end with fake harness
# ===========================================================================


class TestSelfImproverBasic:
    def test_runs_zero_iterations(self, tmp_path):
        cfg = LoopConfig(
            iterations=0,
            ledger_path=str(tmp_path / "ledger.jsonl"),
            stop_sentinel_path="",
            randomize=False,
        )
        si = SelfImprover(cfg, seed_strategies=_make_specs())
        si._harness_factory = _make_factory(lambda c: 0.5)
        records = si.run()
        assert records == []

    def test_skip_records_when_no_applicable_mutations(self, tmp_path):
        # Catalog with a rule that targets a class no spec has.
        catalog = MutationCatalog(
            [
                MutationRule(
                    strategy_pattern="",
                    class_name="DoesNotExist",
                    param_name="x",
                    kind="float_uniform",
                    bounds=(0.0, 1.0),
                ),
            ]
        )
        cfg = LoopConfig(
            iterations=2,
            ledger_path=str(tmp_path / "ledger.jsonl"),
            stop_sentinel_path="",
            randomize=False,
        )
        si = SelfImprover(cfg, catalog=catalog, seed_strategies=_make_specs())
        si._harness_factory = _make_factory(lambda c: 0.5)
        records = si.run()
        assert len(records) == 2
        assert all(r.proposal is None for r in records)
        assert all(r.reason_skipped is not None for r in records)

    def test_accept_strong_improvement(self, tmp_path):
        # Catalog that always proposes a perturbation on radius.
        catalog = MutationCatalog(
            [
                MutationRule(
                    strategy_pattern="",
                    class_name="_DummyHeuristicA",
                    param_name="radius",
                    kind="log_uniform_perturb",
                    bounds=(0.005, 0.5),
                ),
            ]
        )
        # Score depends on call order: baseline call (0.3), candidate (0.7),
        # then alternating.  Each iteration runs baseline then candidate.
        counter = {"n": 0}

        def score_fn(config: HarnessConfig) -> float:
            n = counter["n"]
            counter["n"] += 1
            return 0.3 if n % 2 == 0 else 0.7

        cfg = LoopConfig(
            iterations=2,
            n_boot=200,
            eps_accept=0.005,
            ledger_path=str(tmp_path / "ledger.jsonl"),
            stop_sentinel_path="",
            randomize=False,
        )
        si = SelfImprover(cfg, catalog=catalog, seed_strategies=_make_specs())
        si._harness_factory = _make_factory(score_fn)
        records = si.run()
        assert len(records) == 2
        # First iteration should accept the strong improvement.
        assert records[0].accepted is True
        assert records[0].delta > 0
        assert records[0].baseline_score == pytest.approx(0.3)
        assert records[0].candidate_score == pytest.approx(0.7)

    def test_reject_when_score_unchanged(self, tmp_path):
        catalog = MutationCatalog(
            [
                MutationRule(
                    strategy_pattern="",
                    class_name="_DummyHeuristicA",
                    param_name="radius",
                    kind="log_uniform_perturb",
                    bounds=(0.005, 0.5),
                ),
            ]
        )
        cfg = LoopConfig(
            iterations=1,
            n_boot=100,
            ledger_path=str(tmp_path / "ledger.jsonl"),
            stop_sentinel_path="",
            randomize=False,
        )
        si = SelfImprover(cfg, catalog=catalog, seed_strategies=_make_specs())
        # Both baseline and candidate score identically.
        si._harness_factory = _make_factory(lambda c: 0.5)
        records = si.run()
        assert len(records) == 1
        assert records[0].accepted is False

    def test_stop_sentinel_halts_loop(self, tmp_path):
        sentinel = tmp_path / "STOP"
        sentinel.write_text("")
        cfg = LoopConfig(
            iterations=5,
            ledger_path=str(tmp_path / "ledger.jsonl"),
            stop_sentinel_path=str(sentinel),
            randomize=False,
        )
        si = SelfImprover(cfg, seed_strategies=_make_specs())
        si._harness_factory = _make_factory(lambda c: 0.5)
        records = si.run()
        assert records == []


# ===========================================================================
# Anti-cherry-pick guard
# ===========================================================================


class TestAntiCherryPickGuard:
    def _accept_catalog(self) -> MutationCatalog:
        return MutationCatalog(
            [
                MutationRule(
                    strategy_pattern="",
                    class_name="_DummyHeuristicA",
                    param_name="radius",
                    kind="log_uniform_perturb",
                    bounds=(0.005, 0.5),
                ),
            ]
        )

    def test_disabled_by_default(self, tmp_path):
        """guard_interval=0 must produce no guard records."""
        cfg = LoopConfig(
            iterations=2,
            n_boot=100,
            ledger_path=str(tmp_path / "ledger.jsonl"),
            stop_sentinel_path="",
            randomize=False,
            guard_interval=0,
        )
        si = SelfImprover(cfg, catalog=self._accept_catalog(), seed_strategies=_make_specs())
        si._harness_factory = _make_factory(lambda c: 0.4)
        iter_records, guard_records = si.run_with_guard_records()
        assert len(iter_records) == 2
        assert guard_records == []

    def test_fires_at_correct_cadence(self, tmp_path):
        """guard_interval=2 fires after iter 1 and iter 3 of 4 iterations."""
        cfg = LoopConfig(
            iterations=4,
            n_boot=50,
            ledger_path=str(tmp_path / "ledger.jsonl"),
            stop_sentinel_path="",
            randomize=False,
            guard_interval=2,
        )
        si = SelfImprover(cfg, catalog=self._accept_catalog(), seed_strategies=_make_specs())
        si._harness_factory = _make_factory(lambda c: 0.4)
        _, guard_records = si.run_with_guard_records()
        assert [g.iteration for g in guard_records] == [1, 3]

    def test_no_rollback_when_within_tolerance(self, tmp_path):
        """Guard re-measure within eps_ladder must not pop the ladder."""
        cfg = LoopConfig(
            iterations=1,
            n_boot=100,
            ledger_path=str(tmp_path / "ledger.jsonl"),
            stop_sentinel_path="",
            randomize=False,
            guard_interval=1,
            guard_eps_ladder=0.1,
        )
        si = SelfImprover(cfg, catalog=self._accept_catalog(), seed_strategies=_make_specs())
        # Constant score 0.5 — no drift between any measurements.
        si._harness_factory = _make_factory(lambda c: 0.5)
        _, guard_records = si.run_with_guard_records()
        assert len(guard_records) == 1
        assert guard_records[0].rolled_back is False
        assert guard_records[0].pops == 0

    def test_rollback_when_drift_exceeds_eps(self, tmp_path):
        """When the guard re-measure drops below eps_ladder, the ladder rolls back."""
        # Each call: baseline=0.3, candidate=0.8 — so iter 0 will accept
        # decisively (Δ ≈ +0.5).  Then the guard re-measures the top of the
        # ladder at iter_id = 1_000_000; we make that one return 0.1.
        # Any other call (regular iteration) keeps the alternating pattern.
        counter = {"n": 0}

        def score_fn(config: HarnessConfig) -> float:
            iter_id = config.randomize_iteration
            if iter_id >= 1_000_000:
                # Guard re-measure on the fresh seed — collapse.
                return 0.1
            n = counter["n"]
            counter["n"] += 1
            return 0.3 if n % 2 == 0 else 0.8

        cfg = LoopConfig(
            iterations=1,
            n_boot=100,
            ledger_path=str(tmp_path / "ledger.jsonl"),
            stop_sentinel_path="",
            randomize=False,
            guard_interval=1,
            guard_eps_ladder=0.05,
            guard_iteration_offset=1_000_000,
        )
        si = SelfImprover(cfg, catalog=self._accept_catalog(), seed_strategies=_make_specs())
        si._harness_factory = _make_factory(score_fn)
        iter_records, guard_records = si.run_with_guard_records()
        # Iteration accepted (mutated >> baseline on regular seed).
        assert iter_records[0].accepted is True
        # Guard rolled back — re-measure on the fresh seed showed collapse.
        assert len(guard_records) == 1
        assert guard_records[0].rolled_back is True
        # Drift target was iter 0 (the just-accepted entry).
        assert guard_records[0].pre_guard_top_iteration == 0
        # We pop back to the seed (-1) because the ladder only has one
        # accepted entry above the seed.
        assert guard_records[0].rolled_back_to_iteration == -1
        assert guard_records[0].pops == 1

    def test_guard_uses_offset_iteration_id(self, tmp_path):
        """The guard must measure at ``iter + guard_iteration_offset``."""
        call_log: List[Dict[str, Any]] = []
        cfg = LoopConfig(
            iterations=1,
            n_boot=50,
            ledger_path=str(tmp_path / "ledger.jsonl"),
            stop_sentinel_path="",
            randomize=False,
            guard_interval=1,
            guard_iteration_offset=42_424,
        )
        si = SelfImprover(cfg, catalog=self._accept_catalog(), seed_strategies=_make_specs())
        si._harness_factory = _make_factory(lambda c: 0.5, call_log=call_log)
        si.run()
        # Last call (the guard) should be at the offset iteration id.
        assert call_log[-1]["randomize_iteration"] == 42_424

    def test_guard_does_not_pop_seed(self, tmp_path):
        """Even if every entry drifts, the guard never pops below the seed."""

        # All measurements just return 0.0 so every entry "drifts" — but the
        # ladder only ever has the seed (no accept), so nothing should be
        # popped.
        cfg = LoopConfig(
            iterations=1,
            n_boot=50,
            ledger_path=str(tmp_path / "ledger.jsonl"),
            stop_sentinel_path="",
            randomize=False,
            guard_interval=1,
        )
        catalog = MutationCatalog(
            [
                MutationRule(
                    strategy_pattern="",
                    class_name="DoesNotExist",
                    param_name="foo",
                    kind="float_uniform",
                    bounds=(0.0, 1.0),
                ),
            ]
        )
        si = SelfImprover(cfg, catalog=catalog, seed_strategies=_make_specs())
        si._harness_factory = _make_factory(lambda c: 0.0)
        _, guard_records = si.run_with_guard_records()
        # Skip-only iteration; guard still runs but ladder has one entry.
        assert len(guard_records) == 1
        # No accepted entries → ladder is just the seed → no pops possible.
        assert guard_records[0].pops == 0


# ===========================================================================
# Ledger writing & loading
# ===========================================================================


class TestLedger:
    def test_writes_iteration_and_guard_records(self, tmp_path):
        cfg = LoopConfig(
            iterations=2,
            n_boot=50,
            ledger_path=str(tmp_path / "ledger.jsonl"),
            stop_sentinel_path="",
            randomize=False,
            guard_interval=1,
        )
        si = SelfImprover(
            cfg,
            catalog=MutationCatalog(
                [
                    MutationRule(
                        strategy_pattern="",
                        class_name="_DummyHeuristicA",
                        param_name="radius",
                        kind="log_uniform_perturb",
                        bounds=(0.005, 0.5),
                    ),
                ]
            ),
            seed_strategies=_make_specs(),
        )
        si._harness_factory = _make_factory(lambda c: 0.5)
        si.run()

        records = load_ledger(cfg.ledger_path)
        # 2 iterations + 2 guard records.
        assert len(records) == 4
        types = [r.get("record_type") for r in records]
        # Pattern: iter, guard, iter, guard.
        assert types == ["iteration", "guard", "iteration", "guard"]

    def test_load_ledger_missing_file_returns_empty(self, tmp_path):
        out = load_ledger(str(tmp_path / "does-not-exist.jsonl"))
        assert out == []

    def test_iteration_record_to_dict_round_trip(self):
        rec = LoopIterationRecord(
            iteration=3,
            timestamp="2026-01-01T00:00:00+00:00",
            duration_seconds=1.0,
            proposal={"strategy_name": "S", "class_name": "C", "param_name": "p"},
            accepted=True,
            baseline_score=0.4,
            candidate_score=0.6,
            delta=0.2,
            ci_low=0.05,
            ci_high=0.35,
            worst_pair_regression=-0.01,
            worst_pair=("P", "S"),
            reasons=["ok"],
            base_seed=42,
            randomize_iteration=3,
            mode="quick",
        )
        d = rec.to_dict()
        assert d["record_type"] == "iteration"
        assert d["iteration"] == 3
        assert d["worst_pair"] == ["P", "S"]
        # JSON-serialisable end-to-end.
        assert json.loads(json.dumps(d))

    def test_guard_record_to_dict_round_trip(self):
        rec = LoopGuardRecord(
            iteration=4,
            timestamp="2026-01-01T00:00:00+00:00",
            duration_seconds=0.5,
            guard_score=0.45,
            pre_guard_top_score=0.55,
            pre_guard_top_iteration=2,
            rolled_back=True,
            rolled_back_to_iteration=1,
            pops=1,
            ladder_size_before=3,
            ladder_size_after=2,
            guard_iteration_id=1_000_004,
            reasons=["drift"],
        )
        d = rec.to_dict()
        assert d["record_type"] == "guard"
        assert d["pops"] == 1
        # JSON-serialisable end-to-end.
        assert json.loads(json.dumps(d))


# ===========================================================================
# LadderEntry sanity
# ===========================================================================


class TestLadderEntry:
    def test_construct(self):
        e = LadderEntry(iteration=-1, specs=_make_specs(), last_validated_score=float("nan"))
        assert e.iteration == -1
        assert np.isnan(e.last_validated_score)
        assert e.proposal is None
