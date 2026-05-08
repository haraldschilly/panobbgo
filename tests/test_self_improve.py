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
    AdaptiveMutationSampler,
    LadderEntry,
    LoopConfig,
    LoopGuardRecord,
    LoopHoldoutRecord,
    LoopIterationRecord,
    MutationCatalog,
    MutationProposal,
    MutationRule,
    MutationRuleStats,
    SelfImprover,
    StructuralMutationRule,
    apply_mutation,
    default_catalog,
    default_structural_catalog,
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
                "seed": self.config.seed,
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


# ===========================================================================
# AdaptiveMutationSampler (Thompson sampling)
# ===========================================================================


def _two_rule_catalog() -> MutationCatalog:
    """Two rules pointing at independent kwargs in :func:`_make_specs`."""
    return MutationCatalog(
        [
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
        ]
    )


class TestAdaptiveMutationSampler:
    def test_invalid_priors_raise(self):
        cat = _two_rule_catalog()
        with pytest.raises(ValueError, match="prior_alpha and prior_beta"):
            AdaptiveMutationSampler(cat, prior_alpha=0.0)
        with pytest.raises(ValueError, match="prior_alpha and prior_beta"):
            AdaptiveMutationSampler(cat, prior_beta=-1.0)

    def test_cold_start_returns_proposals(self):
        """A fresh sampler must successfully produce proposals."""
        samp = AdaptiveMutationSampler(_two_rule_catalog())
        rng = np.random.default_rng(0)
        for _ in range(5):
            prop = samp.sample(rng, _make_specs())
            assert prop is not None
            assert prop.class_name in {"_DummyHeuristicA", "_DummyHeuristicB"}

    def test_returns_none_when_no_applicable_rules(self):
        cat = MutationCatalog(
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
        samp = AdaptiveMutationSampler(cat)
        rng = np.random.default_rng(0)
        assert samp.sample(rng, _make_specs()) is None
        # ``last_rule_key`` is reset so a stray record_outcome is a no-op.
        assert samp.last_rule_key is None
        samp.record_outcome(True)  # must not raise
        assert samp.stats_snapshot() == []

    def test_record_outcome_increments_stats(self):
        samp = AdaptiveMutationSampler(_two_rule_catalog())
        rng = np.random.default_rng(0)
        for _ in range(10):
            samp.sample(rng, _make_specs())
            samp.record_outcome(True)
        total_attempts = sum(s.n_attempts for s in samp.stats_snapshot())
        total_accepts = sum(s.n_accepts for s in samp.stats_snapshot())
        assert total_attempts == 10
        assert total_accepts == 10

    def test_record_outcome_no_op_after_none_sample(self):
        samp = AdaptiveMutationSampler(_two_rule_catalog())
        # Without a prior sample(), record is a no-op.
        samp.record_outcome(True)
        assert samp.stats_snapshot() == []

    def test_thompson_biases_toward_winning_rule(self):
        """If one rule always accepts and the other always rejects,
        post-training samples must heavily favor the winning rule.

        This is the headline guarantee of Thompson sampling: the bandit
        must concentrate probability on the empirically better arm.
        """
        cat = _two_rule_catalog()
        samp = AdaptiveMutationSampler(cat)
        rng = np.random.default_rng(123)

        # Train: A accepts, B rejects.
        for _ in range(50):
            prop = samp.sample(rng, _make_specs())
            assert prop is not None
            samp.record_outcome(prop.class_name == "_DummyHeuristicA")

        # Now count picks over a fresh sampling phase (record_outcome
        # disabled so stats freeze).
        counts = {"_DummyHeuristicA": 0, "_DummyHeuristicB": 0}
        for _ in range(500):
            prop = samp.sample(rng, _make_specs())
            assert prop is not None
            counts[prop.class_name] += 1
            # Reset last_rule_key without recording.
            samp._last_rule_key = None

        assert counts["_DummyHeuristicA"] > 4 * counts["_DummyHeuristicB"], (
            f"Thompson should heavily favor the winning rule, got {counts}"
        )

    def test_uniform_prior_matches_uniform_sampler_distribution(self):
        """Beta(1, 1) cold-start must distribute over rules ~uniformly.

        Statistical, not deterministic: the difference between the two
        counts should be small relative to the total over many draws.
        """
        cat = _two_rule_catalog()
        samp = AdaptiveMutationSampler(cat, prior_alpha=1.0, prior_beta=1.0)
        rng = np.random.default_rng(7)

        counts = {"_DummyHeuristicA": 0, "_DummyHeuristicB": 0}
        for _ in range(1000):
            prop = samp.sample(rng, _make_specs())
            counts[prop.class_name] += 1
            # Don't record any outcomes — keep posterior at the prior.
            samp._last_rule_key = None

        # Both buckets should be roughly equal under U(0, 1) arg-max.
        # Tolerance ±10% generous against rng quirks.
        a, b = counts["_DummyHeuristicA"], counts["_DummyHeuristicB"]
        ratio = a / (a + b)
        assert 0.4 <= ratio <= 0.6, f"Expected near-uniform split, got {counts}"

    def test_stats_snapshot_is_sorted(self):
        samp = AdaptiveMutationSampler(_two_rule_catalog())
        rng = np.random.default_rng(0)
        for _ in range(20):
            samp.sample(rng, _make_specs())
            samp.record_outcome(True)
        snap = samp.stats_snapshot()
        keys = [s.rule_key for s in snap]
        assert keys == sorted(keys)

    def test_rule_stats_to_dict_round_trip(self):
        s = MutationRuleStats(rule_key=("Foo", "bar", "log_uniform_perturb"), n_attempts=4, n_accepts=1)
        d = s.to_dict()
        assert d["class_name"] == "Foo"
        assert d["accept_rate"] == pytest.approx(0.25)
        # JSON serialisable.
        assert json.loads(json.dumps(d))

    def test_accept_rate_zero_when_no_attempts(self):
        s = MutationRuleStats(rule_key=("A", "b", "kind"))
        assert s.accept_rate == 0.0

    def test_prime_from_ledger_replays_history(self, tmp_path):
        """Iteration records must be replayed; guards / skips ignored."""
        ledger = tmp_path / "old.jsonl"
        # Two iteration records (one accept), one skip, one guard.
        records = [
            {
                "record_type": "iteration",
                "iteration": 0,
                "proposal": {
                    "class_name": "_DummyHeuristicA",
                    "param_name": "radius",
                    "rule_kind": "log_uniform_perturb",
                },
                "accepted": True,
            },
            {
                "record_type": "iteration",
                "iteration": 1,
                "proposal": {
                    "class_name": "_DummyHeuristicA",
                    "param_name": "radius",
                    "rule_kind": "log_uniform_perturb",
                },
                "accepted": False,
            },
            {
                "record_type": "iteration",
                "iteration": 2,
                "proposal": None,
                "accepted": False,
            },
            {
                "record_type": "guard",
                "iteration": 2,
                "rolled_back": False,
            },
        ]
        ledger.write_text("\n".join(json.dumps(r) for r in records) + "\n")

        samp = AdaptiveMutationSampler(_two_rule_catalog())
        consumed = samp.prime_from_ledger(str(ledger))
        # Only the two iteration records with non-null proposals count.
        assert consumed == 2
        snap = samp.stats_snapshot()
        assert len(snap) == 1
        assert snap[0].n_attempts == 2
        assert snap[0].n_accepts == 1
        assert snap[0].rule_key == ("_DummyHeuristicA", "radius", "log_uniform_perturb")

    def test_prime_from_ledger_missing_file_returns_zero(self, tmp_path):
        samp = AdaptiveMutationSampler(_two_rule_catalog())
        consumed = samp.prime_from_ledger(str(tmp_path / "nope.jsonl"))
        assert consumed == 0
        assert samp.stats_snapshot() == []

    def test_proposal_rationale_includes_thompson_marker(self):
        samp = AdaptiveMutationSampler(_two_rule_catalog())
        rng = np.random.default_rng(0)
        prop = samp.sample(rng, _make_specs())
        assert prop is not None
        assert "Thompson" in prop.rationale


# ===========================================================================
# SelfImprover wired with the adaptive sampler
# ===========================================================================


class TestSelfImproverAdaptive:
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

    def test_adaptive_off_by_default(self, tmp_path):
        cfg = LoopConfig(
            iterations=0,
            ledger_path=str(tmp_path / "ledger.jsonl"),
            stop_sentinel_path="",
            randomize=False,
        )
        si = SelfImprover(cfg, seed_strategies=_make_specs())
        assert si.sampler is None

    def test_adaptive_creates_sampler(self, tmp_path):
        cfg = LoopConfig(
            iterations=0,
            ledger_path=str(tmp_path / "ledger.jsonl"),
            stop_sentinel_path="",
            randomize=False,
            adaptive_sampling=True,
        )
        si = SelfImprover(cfg, seed_strategies=_make_specs())
        assert isinstance(si.sampler, AdaptiveMutationSampler)
        assert si.sampler.prior_alpha == 1.0
        assert si.sampler.prior_beta == 1.0

    def test_adaptive_propagates_priors(self, tmp_path):
        cfg = LoopConfig(
            iterations=0,
            ledger_path=str(tmp_path / "ledger.jsonl"),
            stop_sentinel_path="",
            randomize=False,
            adaptive_sampling=True,
            adaptive_prior_alpha=2.5,
            adaptive_prior_beta=0.5,
        )
        si = SelfImprover(cfg, seed_strategies=_make_specs())
        assert si.sampler is not None
        assert si.sampler.prior_alpha == 2.5
        assert si.sampler.prior_beta == 0.5

    def test_adaptive_prime_from_ledger(self, tmp_path):
        ledger = tmp_path / "ledger.jsonl"
        # Pre-populate ledger with one accepted iteration on the catalog rule.
        ledger.write_text(
            json.dumps(
                {
                    "record_type": "iteration",
                    "proposal": {
                        "class_name": "_DummyHeuristicA",
                        "param_name": "radius",
                        "rule_kind": "log_uniform_perturb",
                    },
                    "accepted": True,
                }
            )
            + "\n"
        )
        cfg = LoopConfig(
            iterations=0,
            ledger_path=str(ledger),
            stop_sentinel_path="",
            randomize=False,
            adaptive_sampling=True,
            adaptive_prime_from_ledger=True,
        )
        si = SelfImprover(cfg, catalog=self._accept_catalog(), seed_strategies=_make_specs())
        assert si.sampler is not None
        snap = si.sampler.stats_snapshot()
        assert len(snap) == 1
        assert snap[0].n_attempts == 1
        assert snap[0].n_accepts == 1

    def test_adaptive_sampler_records_outcomes(self, tmp_path):
        """One iteration with adaptive sampling must update the sampler stats."""
        # Each call returns 0.3 then 0.8 — so the candidate is much better.
        counter = {"n": 0}

        def score_fn(config: HarnessConfig) -> float:
            n = counter["n"]
            counter["n"] += 1
            return 0.3 if n % 2 == 0 else 0.8

        cfg = LoopConfig(
            iterations=1,
            n_boot=100,
            ledger_path=str(tmp_path / "ledger.jsonl"),
            stop_sentinel_path="",
            randomize=False,
            adaptive_sampling=True,
        )
        si = SelfImprover(cfg, catalog=self._accept_catalog(), seed_strategies=_make_specs())
        si._harness_factory = _make_factory(score_fn)
        records = si.run()
        assert records[0].accepted is True
        assert si.sampler is not None
        snap = si.sampler.stats_snapshot()
        assert len(snap) == 1
        assert snap[0].n_attempts == 1
        assert snap[0].n_accepts == 1

    def test_adaptive_sampler_records_rejects(self, tmp_path):
        """Reject paths must increment n_attempts but not n_accepts."""

        # Constant score — no improvement so iteration rejects.
        def score_fn(config: HarnessConfig) -> float:
            return 0.5

        cfg = LoopConfig(
            iterations=2,
            n_boot=100,
            ledger_path=str(tmp_path / "ledger.jsonl"),
            stop_sentinel_path="",
            randomize=False,
            adaptive_sampling=True,
        )
        si = SelfImprover(cfg, catalog=self._accept_catalog(), seed_strategies=_make_specs())
        si._harness_factory = _make_factory(score_fn)
        si.run()
        assert si.sampler is not None
        snap = si.sampler.stats_snapshot()
        assert len(snap) == 1
        assert snap[0].n_attempts == 2
        assert snap[0].n_accepts == 0

    def test_explicit_sampler_overrides_config(self, tmp_path):
        """Passing ``sampler=`` must take priority over ``adaptive_sampling``."""
        cat = self._accept_catalog()
        explicit = AdaptiveMutationSampler(cat, prior_alpha=4.0, prior_beta=4.0)
        cfg = LoopConfig(
            iterations=0,
            ledger_path=str(tmp_path / "ledger.jsonl"),
            stop_sentinel_path="",
            randomize=False,
            adaptive_sampling=False,  # explicit sampler should still win
        )
        si = SelfImprover(cfg, catalog=cat, sampler=explicit, seed_strategies=_make_specs())
        assert si.sampler is explicit
        assert si.sampler.prior_alpha == 4.0


class TestLoopConfigAdaptive:
    def test_invalid_prior_alpha_raises(self):
        with pytest.raises(ValueError, match="adaptive_prior_alpha"):
            LoopConfig(adaptive_prior_alpha=0.0)

    def test_invalid_prior_beta_raises(self):
        with pytest.raises(ValueError, match="adaptive_prior_beta"):
            LoopConfig(adaptive_prior_beta=-1.0)

    def test_defaults(self):
        cfg = LoopConfig()
        assert cfg.adaptive_sampling is False
        assert cfg.adaptive_prior_alpha == 1.0
        assert cfg.adaptive_prior_beta == 1.0
        assert cfg.adaptive_prime_from_ledger is False


# ===========================================================================
# StructuralMutationRule (§7.2 — strategy portfolio composition)
# ===========================================================================


class _NewHeuristicX:
    """A class not yet present in :func:`_make_specs` — usable as an add target."""

    pass


class _NewHeuristicY:
    """Second add-target so add_heuristic has a non-trivial pool to pick from."""

    pass


class TestStructuralMutationRule:
    def test_unknown_op_raises(self):
        with pytest.raises(ValueError, match="Unknown structural op"):
            StructuralMutationRule(strategy_pattern="", op="rename_heuristic")

    def test_min_heuristics_below_one_raises(self):
        with pytest.raises(ValueError, match="min_heuristics"):
            StructuralMutationRule(strategy_pattern="", op="drop_heuristic", min_heuristics=0)

    def test_non_positive_probability_raises(self):
        with pytest.raises(ValueError, match="probability"):
            StructuralMutationRule(strategy_pattern="", op="drop_heuristic", probability=0.0)

    def test_add_requires_candidates(self):
        with pytest.raises(ValueError, match="candidate_classes"):
            StructuralMutationRule(strategy_pattern="", op="add_heuristic", candidate_classes=())

    def test_drop_does_not_require_candidates(self):
        # Construction should not raise; drop allows empty candidate pool.
        rule = StructuralMutationRule(strategy_pattern="", op="drop_heuristic")
        assert rule.candidate_classes == ()

    def test_rule_key_collapses_by_op(self):
        rule_add = StructuralMutationRule(
            strategy_pattern="",
            op="add_heuristic",
            candidate_classes=((_NewHeuristicX, {}),),
        )
        rule_drop = StructuralMutationRule(strategy_pattern="", op="drop_heuristic")
        assert rule_add.rule_key() == ("*", "add_heuristic", "structural")
        assert rule_drop.rule_key() == ("*", "drop_heuristic", "structural")


class TestStructuralCatalogSampling:
    def _add_catalog(self) -> MutationCatalog:
        return MutationCatalog(
            [
                StructuralMutationRule(
                    strategy_pattern="",
                    op="add_heuristic",
                    candidate_classes=((_NewHeuristicX, {"k": 7}), (_NewHeuristicY, {})),
                    avoid_duplicates=True,
                ),
            ]
        )

    def _drop_catalog(self) -> MutationCatalog:
        return MutationCatalog(
            [
                StructuralMutationRule(
                    strategy_pattern="",
                    op="drop_heuristic",
                    min_heuristics=1,
                ),
            ]
        )

    def test_add_proposal_carries_op_and_kwargs(self):
        cat = self._add_catalog()
        rng = np.random.default_rng(0)
        prop = cat.sample(rng, _make_specs())
        assert prop is not None
        assert prop.op == "add_heuristic"
        assert prop.rule_kind == "add_heuristic"
        assert prop.class_name in {"_NewHeuristicX", "_NewHeuristicY"}
        assert isinstance(prop.structural_kwargs, dict)
        # When the chosen class has default kwargs, they round-trip.
        if prop.class_name == "_NewHeuristicX":
            assert prop.structural_kwargs == {"k": 7}

    def test_avoid_duplicates_skips_existing_classes(self):
        # Add a spec that already contains _NewHeuristicX; avoid_duplicates
        # should yield only _NewHeuristicY for that strategy.
        specs = [
            StrategySpec(
                name="StratZ",
                strategy_class=_DummyStrategy,
                heuristics=[(_NewHeuristicX, {})],
            ),
        ]
        cat = self._add_catalog()
        rng = np.random.default_rng(0)
        # With avoid_duplicates, every sample must pick _NewHeuristicY.
        for _ in range(20):
            prop = cat.sample(rng, specs)
            assert prop is not None
            assert prop.class_name == "_NewHeuristicY"

    def test_drop_proposal_targets_existing_class(self):
        cat = self._drop_catalog()
        rng = np.random.default_rng(0)
        prop = cat.sample(rng, _make_specs())
        assert prop is not None
        assert prop.op == "drop_heuristic"
        assert prop.rule_kind == "drop_heuristic"
        # StratX has _DummyHeuristicA and _DummyHeuristicB; StratY only A.
        assert prop.class_name in {"_DummyHeuristicA", "_DummyHeuristicB"}

    def test_drop_respects_min_heuristics_floor(self):
        # ``min_heuristics`` is the floor *after* dropping (the spec
        # always keeps that many).  With the floor at 1, StratX (2
        # heuristics → 1) qualifies but StratY (1 → 0) does not.
        cat = MutationCatalog(
            [
                StructuralMutationRule(
                    strategy_pattern="",
                    op="drop_heuristic",
                    min_heuristics=1,
                ),
            ]
        )
        rng = np.random.default_rng(0)
        for _ in range(30):
            prop = cat.sample(rng, _make_specs())
            assert prop is not None
            assert prop.strategy_name == "StratX"

    def test_drop_min_heuristics_two_requires_three_to_drop(self):
        """min_heuristics=2 means at-least-2-remain; the spec must have ≥3 to start."""
        fat_specs = [
            StrategySpec(
                name="Big",
                strategy_class=_DummyStrategy,
                heuristics=[
                    (_DummyHeuristicA, {"radius": 0.1}),
                    (_DummyHeuristicB, {"sigma0": 0.3}),
                    (_NewHeuristicX, {}),
                ],
            ),
            # A 2-heuristic spec — *not* eligible because dropping breaches
            # the post-drop floor of 2.
            StrategySpec(
                name="Small",
                strategy_class=_DummyStrategy,
                heuristics=[
                    (_DummyHeuristicA, {}),
                    (_DummyHeuristicB, {}),
                ],
            ),
        ]
        cat = MutationCatalog(
            [
                StructuralMutationRule(
                    strategy_pattern="",
                    op="drop_heuristic",
                    min_heuristics=2,
                ),
            ]
        )
        rng = np.random.default_rng(0)
        for _ in range(20):
            prop = cat.sample(rng, fat_specs)
            assert prop is not None
            assert prop.strategy_name == "Big"

    def test_drop_returns_none_when_no_strategy_qualifies(self):
        # Every strategy has only one heuristic → min_heuristics=2 forbids drops.
        skinny_specs = [
            StrategySpec(
                name="OnlyOne",
                strategy_class=_DummyStrategy,
                heuristics=[(_DummyHeuristicA, {})],
            ),
        ]
        cat = MutationCatalog(
            [
                StructuralMutationRule(
                    strategy_pattern="",
                    op="drop_heuristic",
                    min_heuristics=2,
                ),
            ]
        )
        rng = np.random.default_rng(0)
        assert cat.sample(rng, skinny_specs) is None

    def test_droppable_classes_filter(self):
        cat = MutationCatalog(
            [
                StructuralMutationRule(
                    strategy_pattern="",
                    op="drop_heuristic",
                    droppable_classes=("_DummyHeuristicB",),
                    min_heuristics=1,
                ),
            ]
        )
        rng = np.random.default_rng(0)
        for _ in range(10):
            prop = cat.sample(rng, _make_specs())
            assert prop is not None
            assert prop.class_name == "_DummyHeuristicB"

    def test_strategy_pattern_filters_structural(self):
        # Two strategies that both have ≥2 heuristics; the rule's
        # ``strategy_pattern`` filters down to "StratX" only.
        cat = MutationCatalog(
            [
                StructuralMutationRule(
                    strategy_pattern="StratX",
                    op="drop_heuristic",
                    min_heuristics=1,
                ),
            ]
        )
        rng = np.random.default_rng(0)
        for _ in range(10):
            prop = cat.sample(rng, _make_specs())
            assert prop is not None
            assert prop.strategy_name == "StratX"

    def test_kwarg_default_kwargs_are_independent_per_hit(self):
        """The catalog must not share mutable kwargs dicts across proposals."""
        cat = self._add_catalog()
        rng = np.random.default_rng(0)
        proposals = [cat.sample(rng, _make_specs()) for _ in range(20)]
        x_proposals = [p for p in proposals if p is not None and p.class_name == "_NewHeuristicX"]
        if not x_proposals:
            pytest.skip("rng never picked _NewHeuristicX in this seed")
        # Mutating one structural_kwargs must not leak into others.
        x_proposals[0].structural_kwargs["k"] = 999
        for p in x_proposals[1:]:
            assert p.structural_kwargs["k"] == 7


class TestStructuralApplyMutation:
    def test_add_appends_to_heuristics(self):
        specs = _make_specs()
        proposal = MutationProposal(
            strategy_name="StratX",
            class_name="_NewHeuristicX",
            param_name="",
            old_value=None,
            new_value=None,
            rule_kind="add_heuristic",
            rationale="t",
            op="add_heuristic",
            structural_kwargs={"k": 3},
        )
        # Stub out the heuristics-package lookup by monkey-patching the
        # name into the spec's existing classes — apply_mutation will find
        # it there before falling back to the package import.
        specs[0] = StrategySpec(
            name="StratX",
            strategy_class=_DummyStrategy,
            heuristics=[
                (_DummyHeuristicA, {"radius": 0.1}),
                (_NewHeuristicX, {}),  # a sibling we can drop later, but here just for type lookup
            ],
        )
        out = apply_mutation(specs, proposal)
        new_x = next(s for s in out if s.name == "StratX")
        assert any(cls is _NewHeuristicX and kw == {"k": 3} for cls, kw in new_x.heuristics)
        # Original spec untouched.
        assert len(specs[0].heuristics) == 2

    def test_add_falls_back_to_heuristics_package(self):
        # Use a real class from panobbgo.heuristics so the package-import
        # fallback path is exercised.
        from panobbgo.heuristics import Random

        specs = [
            StrategySpec(
                name="S",
                strategy_class=_DummyStrategy,
                heuristics=[(_DummyHeuristicA, {"radius": 0.1})],
            ),
        ]
        proposal = MutationProposal(
            strategy_name="S",
            class_name="Random",
            param_name="",
            old_value=None,
            new_value=None,
            rule_kind="add_heuristic",
            rationale="t",
            op="add_heuristic",
            structural_kwargs={},
        )
        out = apply_mutation(specs, proposal)
        assert len(out[0].heuristics) == 2
        assert out[0].heuristics[1][0] is Random

    def test_add_unknown_class_raises(self):
        specs = [
            StrategySpec(
                name="S",
                strategy_class=_DummyStrategy,
                heuristics=[(_DummyHeuristicA, {"radius": 0.1})],
            ),
        ]
        proposal = MutationProposal(
            strategy_name="S",
            class_name="DoesNotExistAnywhere",
            param_name="",
            old_value=None,
            new_value=None,
            rule_kind="add_heuristic",
            rationale="t",
            op="add_heuristic",
            structural_kwargs={},
        )
        with pytest.raises(ValueError, match="not exported"):
            apply_mutation(specs, proposal)

    def test_drop_removes_first_match(self):
        specs = _make_specs()
        proposal = MutationProposal(
            strategy_name="StratX",
            class_name="_DummyHeuristicB",
            param_name="",
            old_value=None,
            new_value=None,
            rule_kind="drop_heuristic",
            rationale="t",
            op="drop_heuristic",
            structural_kwargs={"sigma0": 0.3},
        )
        out = apply_mutation(specs, proposal)
        new_x = next(s for s in out if s.name == "StratX")
        assert all(cls.__name__ != "_DummyHeuristicB" for cls, _ in new_x.heuristics)
        # Original spec untouched.
        assert any(cls.__name__ == "_DummyHeuristicB" for cls, _ in specs[0].heuristics)

    def test_drop_missing_class_raises(self):
        specs = _make_specs()
        proposal = MutationProposal(
            strategy_name="StratX",
            class_name="DoesNotExist",
            param_name="",
            old_value=None,
            new_value=None,
            rule_kind="drop_heuristic",
            rationale="t",
            op="drop_heuristic",
            structural_kwargs={},
        )
        with pytest.raises(ValueError, match="no heuristic"):
            apply_mutation(specs, proposal)

    def test_drop_refuses_to_empty_strategy(self):
        skinny = [
            StrategySpec(
                name="S",
                strategy_class=_DummyStrategy,
                heuristics=[(_DummyHeuristicA, {})],
            ),
        ]
        proposal = MutationProposal(
            strategy_name="S",
            class_name="_DummyHeuristicA",
            param_name="",
            old_value=None,
            new_value=None,
            rule_kind="drop_heuristic",
            rationale="t",
            op="drop_heuristic",
            structural_kwargs={},
        )
        with pytest.raises(ValueError, match="no heuristics"):
            apply_mutation(skinny, proposal)


class TestStructuralProposalToDict:
    def test_round_trips_op_and_kwargs(self):
        proposal = MutationProposal(
            strategy_name="S",
            class_name="Random",
            param_name="",
            old_value=None,
            new_value=None,
            rule_kind="add_heuristic",
            rationale="t",
            op="add_heuristic",
            structural_kwargs={"radius": 0.1, "div": np.int64(4)},
        )
        d = proposal.to_dict()
        # Op + kwargs surface in the dict; numpy scalars are coerced.
        assert d["op"] == "add_heuristic"
        assert d["structural_kwargs"]["radius"] == 0.1
        assert d["structural_kwargs"]["div"] == 4
        assert isinstance(d["structural_kwargs"]["div"], int)
        # JSON-serialisable.
        assert json.loads(json.dumps(d))

    def test_kwarg_proposal_omits_structural_fields(self):
        proposal = MutationProposal(
            strategy_name="S",
            class_name="Nearby",
            param_name="radius",
            old_value=0.1,
            new_value=0.2,
            rule_kind="log_uniform_perturb",
            rationale="t",
        )
        d = proposal.to_dict()
        assert "op" not in d
        assert "structural_kwargs" not in d


class TestStructuralRuleKey:
    def test_proposal_rule_key_collapses_structural(self):
        from panobbgo.self_improve import _proposal_rule_key

        # Both ops should collapse to the wildcard key — distinguished
        # only by the param_name slot which carries the op name.
        assert _proposal_rule_key("Sobol", "", "add_heuristic") == ("*", "add_heuristic", "structural")
        assert _proposal_rule_key("Random", "", "drop_heuristic") == ("*", "drop_heuristic", "structural")
        # Kwarg keys are unchanged.
        assert _proposal_rule_key("Nearby", "radius", "log_uniform_perturb") == (
            "Nearby",
            "radius",
            "log_uniform_perturb",
        )

    def test_adaptive_sampler_buckets_structural_history(self):
        rule_add = StructuralMutationRule(
            strategy_pattern="",
            op="add_heuristic",
            candidate_classes=((_NewHeuristicX, {}), (_NewHeuristicY, {})),
        )
        cat = MutationCatalog([rule_add])
        samp = AdaptiveMutationSampler(cat)
        rng = np.random.default_rng(0)
        for _ in range(10):
            prop = samp.sample(rng, _make_specs())
            assert prop is not None
            samp.record_outcome(True)
        snap = samp.stats_snapshot()
        # All 10 attempts/accepts collapse into one structural arm.
        assert len(snap) == 1
        assert snap[0].rule_key == ("*", "add_heuristic", "structural")
        assert snap[0].n_attempts == 10
        assert snap[0].n_accepts == 10


class TestDefaultStructuralCatalog:
    def test_returns_catalog_with_structural_rules(self):
        cat = default_structural_catalog()
        kinds = {type(r).__name__ for r in cat.rules}
        assert "MutationRule" in kinds
        assert "StructuralMutationRule" in kinds
        # At least the two structural rules from §7.2 — add + drop.
        ops = {r.op for r in cat.rules if isinstance(r, StructuralMutationRule)}
        assert ops == {"add_heuristic", "drop_heuristic"}

    def test_structural_catalog_is_superset_of_default(self):
        base = default_catalog()
        ext = default_structural_catalog()
        assert len(ext.rules) > len(base.rules)


class TestStructuralEndToEnd:
    """Exercise SelfImprover with a structural-rule catalog end-to-end."""

    def _structural_catalog(self) -> MutationCatalog:
        return MutationCatalog(
            [
                StructuralMutationRule(
                    strategy_pattern="",
                    op="drop_heuristic",
                    min_heuristics=1,
                ),
            ]
        )

    def test_loop_accepts_structural_improvement(self, tmp_path):
        # Baseline 0.3, candidate 0.8 — strong improvement, must accept.
        counter = {"n": 0}

        def score_fn(config: HarnessConfig) -> float:
            n = counter["n"]
            counter["n"] += 1
            return 0.3 if n % 2 == 0 else 0.8

        cfg = LoopConfig(
            iterations=1,
            n_boot=200,
            ledger_path=str(tmp_path / "ledger.jsonl"),
            stop_sentinel_path="",
            randomize=False,
        )
        si = SelfImprover(cfg, catalog=self._structural_catalog(), seed_strategies=_make_specs())
        si._harness_factory = _make_factory(score_fn)
        records = si.run()
        assert len(records) == 1
        rec = records[0]
        assert rec.accepted is True
        assert rec.proposal is not None
        assert rec.proposal["op"] == "drop_heuristic"

    def test_loop_skips_when_structural_rule_inapplicable(self, tmp_path):
        # Skinny specs (1 heuristic each) + min_heuristics=2 → no drop possible.
        skinny = [
            StrategySpec(
                name="S1",
                strategy_class=_DummyStrategy,
                heuristics=[(_DummyHeuristicA, {})],
            ),
            StrategySpec(
                name="S2",
                strategy_class=_DummyStrategy,
                heuristics=[(_DummyHeuristicA, {})],
            ),
        ]
        cat = MutationCatalog(
            [
                StructuralMutationRule(
                    strategy_pattern="",
                    op="drop_heuristic",
                    min_heuristics=2,
                ),
            ]
        )
        cfg = LoopConfig(
            iterations=2,
            ledger_path=str(tmp_path / "ledger.jsonl"),
            stop_sentinel_path="",
            randomize=False,
        )
        si = SelfImprover(cfg, catalog=cat, seed_strategies=skinny)
        si._harness_factory = _make_factory(lambda c: 0.5)
        records = si.run()
        assert len(records) == 2
        assert all(r.proposal is None for r in records)
        assert all(r.reason_skipped is not None for r in records)


# ===========================================================================
# Hold-out validation set
# ===========================================================================


def _accept_radius_catalog() -> MutationCatalog:
    """Catalog whose only rule perturbs ``_DummyHeuristicA.radius``.

    Used by hold-out tests so the loop reliably proposes mutations
    against ``_make_specs()`` and the bandit's accept rate is governed
    purely by the ``score_fn`` we hand the fake harness.
    """
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


class TestLoopConfigHoldout:
    """Validation of the hold-out config knobs."""

    def test_defaults_disabled(self):
        cfg = LoopConfig()
        assert cfg.holdout_base_seed == 0
        assert cfg.holdout_iterations == 5
        assert cfg.holdout_iteration_offset == 0
        assert cfg.holdout_eps_overfit == 0.05

    def test_negative_iterations_raises(self):
        with pytest.raises(ValueError, match="holdout_iterations"):
            LoopConfig(holdout_iterations=-1)

    def test_negative_eps_overfit_raises(self):
        with pytest.raises(ValueError, match="holdout_eps_overfit"):
            LoopConfig(holdout_eps_overfit=-0.1)

    def test_holdout_base_seed_equal_to_base_seed_raises(self):
        # The whole point is an *independent* SHA-256 stream.  Equal
        # values would silently collapse the check; reject loudly.
        with pytest.raises(ValueError, match="holdout_base_seed must differ"):
            LoopConfig(base_seed=42, holdout_base_seed=42)

    def test_holdout_base_seed_zero_does_not_collide_with_base_seed(self):
        # 0 means "disabled" — the equality check must skip 0 even when
        # base_seed is also 0.
        cfg = LoopConfig(base_seed=0, holdout_base_seed=0)
        assert cfg.holdout_base_seed == 0

    def test_holdout_harness_config_uses_holdout_seed(self):
        cfg = LoopConfig(base_seed=42, holdout_base_seed=99)
        hc = cfg.holdout_harness_config([], iteration_id=7)
        assert hc.seed == 99
        assert hc.randomize_iteration == 7

    def test_regular_harness_config_still_uses_base_seed(self):
        cfg = LoopConfig(base_seed=42, holdout_base_seed=99)
        hc = cfg.harness_config([], iteration_id=3)
        assert hc.seed == 42  # unchanged
        assert hc.randomize_iteration == 3


class TestSelfImproverHoldout:
    """End-to-end behaviour of the hold-out validation pass."""

    def test_disabled_by_default(self, tmp_path):
        """holdout_base_seed=0 must produce no hold-out records."""
        cfg = LoopConfig(
            iterations=2,
            n_boot=100,
            ledger_path=str(tmp_path / "ledger.jsonl"),
            stop_sentinel_path="",
            randomize=True,
            # holdout_base_seed defaults to 0 (disabled).
        )
        si = SelfImprover(cfg, catalog=_accept_radius_catalog(), seed_strategies=_make_specs())
        si._harness_factory = _make_factory(lambda c: 0.4)
        _, _, holdout_records = si.run_full()
        assert holdout_records == []

    def test_skipped_when_randomize_false(self, tmp_path):
        """randomize=False makes hold-out vacuous; the loop must skip it.

        Without randomization, ``base_seed`` does not affect the
        instances drawn — every measurement returns identical scores
        and a hold-out check would be no signal at all.  The helper
        ``_holdout_enabled`` guards against this so we never write a
        meaningless ``LoopHoldoutRecord``.
        """
        cfg = LoopConfig(
            iterations=2,
            n_boot=100,
            ledger_path=str(tmp_path / "ledger.jsonl"),
            stop_sentinel_path="",
            randomize=False,
            base_seed=42,
            holdout_base_seed=99,
            holdout_iterations=3,
        )
        si = SelfImprover(cfg, catalog=_accept_radius_catalog(), seed_strategies=_make_specs())
        si._harness_factory = _make_factory(lambda c: 0.4)
        _, _, holdout_records = si.run_full()
        assert holdout_records == []

    def test_skipped_when_zero_iterations_run(self, tmp_path):
        """Zero iterations means no ladder activity; hold-out is moot."""
        cfg = LoopConfig(
            iterations=0,
            ledger_path=str(tmp_path / "ledger.jsonl"),
            stop_sentinel_path="",
            randomize=True,
            base_seed=42,
            holdout_base_seed=99,
            holdout_iterations=3,
        )
        si = SelfImprover(cfg, catalog=_accept_radius_catalog(), seed_strategies=_make_specs())
        si._harness_factory = _make_factory(lambda c: 0.4)
        _, _, holdout_records = si.run_full()
        assert holdout_records == []

    def test_seed_only_ladder_records_zero_drift(self, tmp_path):
        """When no mutation is accepted, holdout_delta == 0 by construction.

        With ``score_fn`` constant the iteration produces zero delta and
        the rule is rejected.  The ladder still has only the seed —
        hold-out reports it as a no-op (drift 0) and overfit=False.
        """
        cfg = LoopConfig(
            iterations=1,
            n_boot=50,
            ledger_path=str(tmp_path / "ledger.jsonl"),
            stop_sentinel_path="",
            randomize=True,
            base_seed=42,
            holdout_base_seed=99,
            holdout_iterations=3,
            holdout_iteration_offset=0,
        )
        si = SelfImprover(cfg, catalog=_accept_radius_catalog(), seed_strategies=_make_specs())
        # Constant score: the iteration delta is 0 and the rule is rejected.
        si._harness_factory = _make_factory(lambda c: 0.4)
        iter_records, _, holdout_records = si.run_full()
        assert iter_records[0].accepted is False
        assert len(holdout_records) == 1
        rec = holdout_records[0]
        assert rec.top_iteration == -1  # ladder stayed at the seed
        assert rec.ladder_size == 1
        assert rec.holdout_delta == pytest.approx(0.0)
        assert rec.training_delta == pytest.approx(0.0)
        assert rec.drift == pytest.approx(0.0)
        assert rec.overfit is False
        assert any("only the seed entry" in r for r in rec.reasons)

    def test_uses_holdout_base_seed_for_measurement(self, tmp_path):
        """Hold-out calls must use ``holdout_base_seed``, not the training seed.

        Sanity-checks that the SHA-256 instance stream is genuinely
        independent: no hold-out call should go through with the
        training base_seed.
        """
        call_log: List[Dict[str, Any]] = []
        cfg = LoopConfig(
            iterations=1,
            n_boot=50,
            ledger_path=str(tmp_path / "ledger.jsonl"),
            stop_sentinel_path="",
            randomize=True,
            base_seed=42,
            holdout_base_seed=4242,
            holdout_iterations=2,
            holdout_iteration_offset=10,
        )
        si = SelfImprover(cfg, catalog=_accept_radius_catalog(), seed_strategies=_make_specs())
        si._harness_factory = _make_factory(lambda c: 0.5, call_log=call_log)
        si.run_full()

        training_calls = [c for c in call_log if c["seed"] == 42]
        holdout_calls = [c for c in call_log if c["seed"] == 4242]
        # 1 iter × 2 measurements (baseline + candidate) = 2 training calls.
        assert len(training_calls) == 2
        # 2 hold-out iterations × 2 ladder slots (seed + top) = 4.
        # Note: when ladder has only seed, top_specs IS seed_specs so the
        # branch only does one measurement per iter — handle both cases.
        assert len(holdout_calls) in (2, 4)
        # All hold-out iter ids must come from offset=10 sweep [10, 11].
        assert {c["randomize_iteration"] for c in holdout_calls} == {10, 11}

    def test_overfit_flagged_when_gap_collapses(self, tmp_path):
        """A mutation that wins on training but collapses on hold-out is flagged.

        Setup: training measurements have a +0.5 gap (seed=0.3,
        top=0.8), so the iteration accepts decisively.  Hold-out
        measurements (those using ``seed=99``) collapse: seed=0.3,
        top=0.3 — zero gap.  Drift = 0 - 0.5 = -0.5 < -0.05, so the
        record reports ``overfit=True``.
        """
        # Training calls alternate baseline (0.3) / candidate (0.8).
        # Hold-out calls all return 0.3 — total collapse on the
        # independent base_seed family.
        counter = {"n": 0}

        def score_fn(config: HarnessConfig) -> float:
            if config.seed == 99:
                # Hold-out — collapse.
                return 0.3
            n = counter["n"]
            counter["n"] += 1
            return 0.3 if n % 2 == 0 else 0.8

        cfg = LoopConfig(
            iterations=1,
            n_boot=50,
            ledger_path=str(tmp_path / "ledger.jsonl"),
            stop_sentinel_path="",
            randomize=True,
            base_seed=42,
            holdout_base_seed=99,
            holdout_iterations=3,
            holdout_eps_overfit=0.05,
        )
        si = SelfImprover(cfg, catalog=_accept_radius_catalog(), seed_strategies=_make_specs())
        si._harness_factory = _make_factory(score_fn)
        iter_records, _, holdout_records = si.run_full()
        assert iter_records[0].accepted is True
        assert len(holdout_records) == 1
        rec = holdout_records[0]
        assert rec.top_iteration == 0  # the mutation we just accepted
        assert rec.ladder_size == 2
        assert rec.training_delta == pytest.approx(0.5, abs=1e-6)
        assert rec.holdout_delta == pytest.approx(0.0, abs=1e-6)
        assert rec.drift == pytest.approx(-0.5, abs=1e-6)
        assert rec.overfit is True
        assert any("overfit" in r for r in rec.reasons)

    def test_generalises_when_gap_holds(self, tmp_path):
        """Improvement that holds on hold-out is *not* flagged as overfit."""
        counter = {"n": 0}

        def score_fn(config: HarnessConfig) -> float:
            if config.seed == 99:
                # Hold-out preserves the gap — slightly noisy but well within
                # the eps_overfit tolerance.
                ho = counter.get("ho", 0)
                counter["ho"] = ho + 1
                # Alternate seed-eval and top-eval; seed=0.3, top=0.78.
                return 0.3 if ho % 2 == 0 else 0.78
            n = counter["n"]
            counter["n"] += 1
            return 0.3 if n % 2 == 0 else 0.8

        cfg = LoopConfig(
            iterations=1,
            n_boot=50,
            ledger_path=str(tmp_path / "ledger.jsonl"),
            stop_sentinel_path="",
            randomize=True,
            base_seed=42,
            holdout_base_seed=99,
            holdout_iterations=4,  # even so seed/top alternation balances
            holdout_eps_overfit=0.05,
        )
        si = SelfImprover(cfg, catalog=_accept_radius_catalog(), seed_strategies=_make_specs())
        si._harness_factory = _make_factory(score_fn)
        iter_records, _, holdout_records = si.run_full()
        assert iter_records[0].accepted is True
        rec = holdout_records[0]
        assert rec.overfit is False
        # Drift |delta| stays within tolerance.
        assert abs(rec.drift) < cfg.holdout_eps_overfit
        assert any("generalise" in r for r in rec.reasons)

    def test_writes_holdout_record_to_ledger(self, tmp_path):
        """The hold-out record is appended to the JSONL ledger as
        ``record_type='holdout'``.

        Auditors loading the ledger must be able to filter to just the
        hold-out records, so the type tag is the contract.
        """
        cfg = LoopConfig(
            iterations=1,
            n_boot=50,
            ledger_path=str(tmp_path / "ledger.jsonl"),
            stop_sentinel_path="",
            randomize=True,
            base_seed=42,
            holdout_base_seed=99,
            holdout_iterations=2,
        )
        si = SelfImprover(cfg, catalog=_accept_radius_catalog(), seed_strategies=_make_specs())
        si._harness_factory = _make_factory(lambda c: 0.4)
        si.run_full()

        records = load_ledger(cfg.ledger_path)
        types = [r.get("record_type") for r in records]
        # Order: 1 iter, then 1 holdout record at the end.
        assert types[-1] == "holdout"
        assert types.count("holdout") == 1

    def test_run_keeps_back_compatibility(self, tmp_path):
        """:meth:`SelfImprover.run` still returns just iteration records."""
        cfg = LoopConfig(
            iterations=1,
            n_boot=50,
            ledger_path=str(tmp_path / "ledger.jsonl"),
            stop_sentinel_path="",
            randomize=True,
            base_seed=42,
            holdout_base_seed=99,
            holdout_iterations=2,
        )
        si = SelfImprover(cfg, catalog=_accept_radius_catalog(), seed_strategies=_make_specs())
        si._harness_factory = _make_factory(lambda c: 0.4)
        records = si.run()
        # Type stable: list of LoopIterationRecord, not a tuple.
        assert isinstance(records, list)
        assert all(isinstance(r, LoopIterationRecord) for r in records)


class TestLoopHoldoutRecord:
    """Round-trip and surface checks for the new dataclass."""

    def test_to_dict_round_trip(self):
        rec = LoopHoldoutRecord(
            timestamp="2026-05-08T00:00:00+00:00",
            duration_seconds=1.5,
            holdout_base_seed=99,
            holdout_iterations=4,
            holdout_iteration_offset=0,
            seed_holdout_score=0.30,
            top_holdout_score=0.32,
            seed_training_score=0.30,
            top_training_score=0.80,
            holdout_delta=0.02,
            training_delta=0.50,
            drift=-0.48,
            overfit=True,
            eps_overfit=0.05,
            top_iteration=4,
            ladder_size=3,
            base_seed=42,
            mode="quick",
            reasons=["hold-out drift -0.4800 below -eps_overfit"],
        )
        d = rec.to_dict()
        assert d["record_type"] == "holdout"
        assert d["overfit"] is True
        assert d["drift"] == pytest.approx(-0.48)
        assert d["holdout_iterations"] == 4
        assert d["base_seed"] == 42
        assert d["mode"] == "quick"
        # Round-trip through JSON to validate JSON-default coverage.
        s = json.dumps(d)
        parsed = json.loads(s)
        assert parsed["record_type"] == "holdout"
        assert parsed["holdout_base_seed"] == 99
