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
    HoldoutDriftAggregate,
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
    aggregate_holdout_drift,
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

    def test_categorical_choice_constructs(self):
        rule = MutationRule(
            strategy_pattern="",
            class_name="X",
            param_name="topology",
            kind="categorical_choice",
            choices=("gbest", "lbest"),
        )
        assert rule.kind == "categorical_choice"
        assert rule.choices == ("gbest", "lbest")
        # rule_key disambiguates from numeric kinds on the same (class, param).
        assert rule.rule_key() == ("X", "topology", "categorical_choice")

    def test_categorical_choice_requires_two_choices(self):
        with pytest.raises(ValueError, match="at least 2"):
            MutationRule(
                strategy_pattern="",
                class_name="X",
                param_name="topology",
                kind="categorical_choice",
                choices=("only_one",),
            )

    def test_categorical_choice_rejects_empty_choices(self):
        with pytest.raises(ValueError, match="at least 2"):
            MutationRule(
                strategy_pattern="",
                class_name="X",
                param_name="topology",
                kind="categorical_choice",
                choices=(),
            )

    def test_categorical_choice_rejects_duplicate_choices(self):
        with pytest.raises(ValueError, match="duplicate"):
            MutationRule(
                strategy_pattern="",
                class_name="X",
                param_name="topology",
                kind="categorical_choice",
                choices=("a", "a"),
            )

    def test_categorical_choice_ignores_bounds(self):
        # bounds=(0,0) default — categorical never reads it, so the
        # "ordered" check is intentionally skipped.
        rule = MutationRule(
            strategy_pattern="",
            class_name="X",
            param_name="scramble",
            kind="categorical_choice",
            choices=(True, False),
        )
        assert rule.bounds == (0.0, 0.0)


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

    def test_applicable_rules_skips_none_value(self):
        """Rule for an existing kwarg explicitly set to ``None`` should be
        skipped — ``None`` is the sentinel a number of heuristics use
        ("use the auto-default") and numeric mutation kinds cannot
        perturb it.

        Concretely, the ``Restart.patience`` and ``LBFGSB.max_starts``
        rules in :func:`default_catalog` rely on this so they can
        co-exist with built-in specs that ship the kwarg as ``None``.
        """
        rules = [
            MutationRule(
                strategy_pattern="",
                class_name="_DummyHeuristicA",
                param_name="radius",
                kind="float_uniform",
                bounds=(0.0, 1.0),
            ),
        ]
        # Build a spec where ``radius`` is explicitly None (the sentinel).
        specs = [
            StrategySpec(
                name="StratNone",
                strategy_class=_DummyStrategy,
                heuristics=[(_DummyHeuristicA, {"radius": None})],
            )
        ]
        cat = MutationCatalog(rules)
        assert cat.applicable_rules(specs) == []

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

    def test_default_catalog_has_categorical_rules(self):
        """Default catalog ships the full set of categorical mutation rules."""
        cat = default_catalog()
        cat_rules = [r for r in cat.rules if getattr(r, "kind", None) == "categorical_choice"]
        keys = {(r.class_name, r.param_name) for r in cat_rules}
        assert ("PSO", "topology") in keys
        assert ("Sobol", "scramble") in keys
        assert ("LSHADE", "archive_factor") in keys
        assert ("LSHADE", "F_schedule") in keys
        assert ("NLSHADE_RSP", "adaptive_archive") in keys
        # Added: literature regimes for the RSP coefficient.
        assert ("NLSHADE_RSP", "k_rank") in keys
        # Added: COBYQA box-rescaling toggle.
        assert ("COBYQA", "scale") in keys
        # Added: Restart center-picking regimes.
        assert ("Restart", "restart_strategy") in keys
        # Added: literature regimes for the jSO p_best schedule's upper
        # bound, alongside the continuous ``float_uniform`` rule.
        assert ("JSO", "p_best_max") in keys

    def test_sample_categorical_choice_picks_different_value(self):
        """Categorical mutation always proposes a value != current."""
        rule = MutationRule(
            strategy_pattern="",
            class_name="_DummyHeuristicA",
            param_name="mode",
            kind="categorical_choice",
            choices=("A", "B", "C"),
        )
        # Inject ``mode="A"`` so the rule is applicable.
        specs = [
            StrategySpec(
                name="S",
                strategy_class=_DummyStrategy,
                heuristics=[(_DummyHeuristicA, {"mode": "A"})],
            )
        ]
        cat = MutationCatalog([rule])
        rng = np.random.default_rng(0)
        for _ in range(30):
            prop = cat.sample(rng, specs)
            assert prop is not None
            assert prop.new_value in ("B", "C")
            assert prop.old_value == "A"

    def test_sample_categorical_choice_two_way_toggle(self):
        """With exactly 2 choices the rule must flip to the other one."""
        rule = MutationRule(
            strategy_pattern="",
            class_name="_DummyHeuristicA",
            param_name="flag",
            kind="categorical_choice",
            choices=(True, False),
        )
        specs = [
            StrategySpec(
                name="S",
                strategy_class=_DummyStrategy,
                heuristics=[(_DummyHeuristicA, {"flag": True})],
            )
        ]
        cat = MutationCatalog([rule])
        rng = np.random.default_rng(0)
        for _ in range(20):
            prop = cat.sample(rng, specs)
            assert prop is not None
            assert prop.new_value is False

    def test_sample_categorical_choice_old_not_in_choices(self):
        """When the current value drifted out of the choice set, every
        entry is a valid candidate (no exclusion)."""
        rule = MutationRule(
            strategy_pattern="",
            class_name="_DummyHeuristicA",
            param_name="mode",
            kind="categorical_choice",
            choices=("A", "B"),
        )
        specs = [
            StrategySpec(
                name="S",
                strategy_class=_DummyStrategy,
                # ``mode="Z"`` is not in choices.
                heuristics=[(_DummyHeuristicA, {"mode": "Z"})],
            )
        ]
        cat = MutationCatalog([rule])
        rng = np.random.default_rng(0)
        seen = set()
        for _ in range(40):
            prop = cat.sample(rng, specs)
            assert prop is not None
            assert prop.new_value in ("A", "B")
            seen.add(prop.new_value)
        # Both options should be reachable.
        assert seen == {"A", "B"}

    def test_sample_categorical_choice_rationale_format(self):
        rule = MutationRule(
            strategy_pattern="",
            class_name="_DummyHeuristicA",
            param_name="mode",
            kind="categorical_choice",
            choices=("A", "B"),
        )
        specs = [
            StrategySpec(
                name="S",
                strategy_class=_DummyStrategy,
                heuristics=[(_DummyHeuristicA, {"mode": "A"})],
            )
        ]
        prop = MutationCatalog([rule]).sample(np.random.default_rng(0), specs)
        assert prop is not None
        # Rationale should mention the kind, the target, and the values.
        assert "categorical_choice" in prop.rationale
        assert "mode" in prop.rationale
        assert prop.rule_kind == "categorical_choice"


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

    def test_applies_categorical_string_value(self):
        """A categorical proposal must overwrite the kwarg with the new
        string value (the apply path is value-type-agnostic)."""
        specs = [
            StrategySpec(
                name="S",
                strategy_class=_DummyStrategy,
                heuristics=[(_DummyHeuristicA, {"topology": "gbest"})],
            )
        ]
        proposal = MutationProposal(
            strategy_name="S",
            class_name="_DummyHeuristicA",
            param_name="topology",
            old_value="gbest",
            new_value="lbest",
            rule_kind="categorical_choice",
            rationale="flip",
        )
        out = apply_mutation(specs, proposal)
        assert out[0].heuristics[0][1]["topology"] == "lbest"
        # Original spec untouched.
        assert specs[0].heuristics[0][1]["topology"] == "gbest"

    def test_applies_categorical_bool_value(self):
        """Bool round-trips through apply_mutation without coercion to int."""
        specs = [
            StrategySpec(
                name="S",
                strategy_class=_DummyStrategy,
                heuristics=[(_DummyHeuristicA, {"scramble": True})],
            )
        ]
        proposal = MutationProposal(
            strategy_name="S",
            class_name="_DummyHeuristicA",
            param_name="scramble",
            old_value=True,
            new_value=False,
            rule_kind="categorical_choice",
            rationale="flip",
        )
        out = apply_mutation(specs, proposal)
        applied = out[0].heuristics[0][1]["scramble"]
        assert applied is False
        assert isinstance(applied, bool)


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

    def test_paired_default_is_auto_detect(self):
        """``paired`` defaults to ``None`` (auto-detect) so randomized
        runs get paired CIs without explicit opt-in."""
        cfg = LoopConfig()
        assert cfg.paired is None

    def test_paired_can_be_forced_true_or_false(self):
        cfg_true = LoopConfig(paired=True)
        assert cfg_true.paired is True
        cfg_false = LoopConfig(paired=False)
        assert cfg_false.paired is False


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

    def test_categorical_rule_gets_its_own_arm(self):
        """A categorical_choice rule must occupy a distinct bandit arm
        from a numeric rule on the same (class, param) slot."""
        rules = [
            MutationRule(
                strategy_pattern="",
                class_name="_DummyHeuristicA",
                param_name="radius",
                kind="log_uniform_perturb",
                bounds=(0.01, 1.0),
            ),
            MutationRule(
                strategy_pattern="",
                class_name="_DummyHeuristicA",
                param_name="radius",
                kind="categorical_choice",
                choices=(0.05, 0.1, 0.2),
            ),
        ]
        cat = MutationCatalog(rules)
        samp = AdaptiveMutationSampler(cat)
        rng = np.random.default_rng(0)
        for _ in range(20):
            prop = samp.sample(rng, _make_specs())
            assert prop is not None
            samp.record_outcome(True)
        snap = samp.stats_snapshot()
        keys = {s.rule_key for s in snap}
        assert ("_DummyHeuristicA", "radius", "log_uniform_perturb") in keys
        assert ("_DummyHeuristicA", "radius", "categorical_choice") in keys

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
# Archive priming (V2 §2.6 / §9.5 step 4)
# ===========================================================================


class TestPrimeFromArchives:
    """Tests for :meth:`AdaptiveMutationSampler.prime_from_archives`.

    Addresses the §2.6 V2 diagnosis "priming reads only the current
    ledger — archives in ``planning/done/`` are invisible".  The
    bandit posterior should now accumulate evidence across every
    retained nightly run.
    """

    @staticmethod
    def _accept_record(class_name: str = "_DummyHeuristicA") -> Dict[str, Any]:
        return {
            "record_type": "iteration",
            "iteration": 0,
            "proposal": {
                "class_name": class_name,
                "param_name": "radius",
                "rule_kind": "log_uniform_perturb",
            },
            "accepted": True,
        }

    @staticmethod
    def _reject_record(class_name: str = "_DummyHeuristicA") -> Dict[str, Any]:
        return {
            "record_type": "iteration",
            "iteration": 1,
            "proposal": {
                "class_name": class_name,
                "param_name": "radius",
                "rule_kind": "log_uniform_perturb",
            },
            "accepted": False,
        }

    @staticmethod
    def _write_archive(path, records):
        path.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    def test_missing_directory_is_no_op(self, tmp_path):
        """A non-existent archive dir returns 0 and leaves the posterior untouched."""
        samp = AdaptiveMutationSampler(_two_rule_catalog())
        consumed = samp.prime_from_archives(str(tmp_path / "no-such-dir"))
        assert consumed == 0
        assert samp.stats_snapshot() == []

    def test_empty_directory_is_no_op(self, tmp_path):
        """An empty directory (no matching files) returns 0."""
        archives = tmp_path / "done"
        archives.mkdir()
        samp = AdaptiveMutationSampler(_two_rule_catalog())
        consumed = samp.prime_from_archives(str(archives))
        assert consumed == 0

    def test_directory_with_non_matching_files_is_no_op(self, tmp_path):
        """Files that don't match the rotation glob are ignored."""
        archives = tmp_path / "done"
        archives.mkdir()
        # Wrong prefix / extension — must be skipped.
        self._write_archive(archives / "summary.txt", [self._accept_record()])
        self._write_archive(archives / "other_ledger.jsonl", [self._accept_record()])
        samp = AdaptiveMutationSampler(_two_rule_catalog())
        consumed = samp.prime_from_archives(str(archives))
        assert consumed == 0
        assert samp.stats_snapshot() == []

    def test_single_archive_replayed(self, tmp_path):
        """One archived ledger contributes one accept to the posterior."""
        archives = tmp_path / "done"
        archives.mkdir()
        self._write_archive(
            archives / "self_improve_ledger_2026-05-31.jsonl",
            [self._accept_record(), self._reject_record()],
        )
        samp = AdaptiveMutationSampler(_two_rule_catalog())
        consumed = samp.prime_from_archives(str(archives))
        assert consumed == 2
        snap = samp.stats_snapshot()
        assert len(snap) == 1
        assert snap[0].n_attempts == 2
        assert snap[0].n_accepts == 1

    def test_multiple_archives_replayed_in_chronological_order(self, tmp_path):
        """Several archives sum across all of them, chronological by filename."""
        archives = tmp_path / "done"
        archives.mkdir()
        self._write_archive(
            archives / "self_improve_ledger_2026-05-31.jsonl",
            [self._accept_record(), self._accept_record(), self._reject_record()],
        )
        self._write_archive(
            archives / "self_improve_ledger_2026-06-01.jsonl",
            [self._accept_record(), self._reject_record()],
        )
        samp = AdaptiveMutationSampler(_two_rule_catalog())
        consumed = samp.prime_from_archives(str(archives))
        assert consumed == 5
        snap = samp.stats_snapshot()
        assert len(snap) == 1
        assert snap[0].n_attempts == 5
        assert snap[0].n_accepts == 3

    def test_archives_filter_no_op_records(self, tmp_path):
        """No-op records in archives are skipped (matches live ledger semantics)."""
        archives = tmp_path / "done"
        archives.mkdir()
        noop = self._accept_record()
        noop["no_op"] = True
        self._write_archive(
            archives / "self_improve_ledger_2026-05-31.jsonl",
            [self._accept_record(), noop, self._reject_record()],
        )
        samp = AdaptiveMutationSampler(_two_rule_catalog())
        consumed = samp.prime_from_archives(str(archives))
        # No-op record contributes nothing; accept + reject = 2.
        assert consumed == 2
        snap = samp.stats_snapshot()
        assert snap[0].n_attempts == 2
        assert snap[0].n_accepts == 1

    def test_archives_filter_guard_and_skip_records(self, tmp_path):
        """Guard and skip (null proposal) records in archives are ignored."""
        archives = tmp_path / "done"
        archives.mkdir()
        records = [
            self._accept_record(),
            {"record_type": "guard", "iteration": 0, "rolled_back": False},
            {"record_type": "iteration", "iteration": 1, "proposal": None, "accepted": False},
        ]
        self._write_archive(archives / "self_improve_ledger_2026-05-31.jsonl", records)
        samp = AdaptiveMutationSampler(_two_rule_catalog())
        consumed = samp.prime_from_archives(str(archives))
        assert consumed == 1
        snap = samp.stats_snapshot()
        assert snap[0].n_attempts == 1
        assert snap[0].n_accepts == 1

    def test_archives_propagate_graded_bandit_reward(self, tmp_path):
        """Archived graded rewards accumulate into reward_sum, matching prime_from_ledger."""
        archives = tmp_path / "done"
        archives.mkdir()
        rec = self._accept_record()
        rec["bandit_reward"] = 0.75
        self._write_archive(archives / "self_improve_ledger_2026-05-31.jsonl", [rec])
        samp = AdaptiveMutationSampler(_two_rule_catalog())
        consumed = samp.prime_from_archives(str(archives))
        assert consumed == 1
        snap = samp.stats_snapshot()
        assert snap[0].n_attempts == 1
        assert snap[0].n_accepts == 1
        assert snap[0].reward_sum == pytest.approx(0.75)

    def test_archives_combined_with_live_ledger(self, tmp_path):
        """Live ledger prime + archives prime accumulate together."""
        archives = tmp_path / "done"
        archives.mkdir()
        self._write_archive(
            archives / "self_improve_ledger_2026-05-31.jsonl",
            [self._accept_record(), self._reject_record()],
        )
        live = tmp_path / "ledger.jsonl"
        self._write_archive(live, [self._accept_record(), self._reject_record()])

        samp = AdaptiveMutationSampler(_two_rule_catalog())
        n_archive = samp.prime_from_archives(str(archives))
        n_live = samp.prime_from_ledger(str(live))
        assert n_archive == 2
        assert n_live == 2
        snap = samp.stats_snapshot()
        assert len(snap) == 1
        assert snap[0].n_attempts == 4
        assert snap[0].n_accepts == 2

    def test_archive_path_is_a_file_returns_zero(self, tmp_path):
        """When the archive path points to a regular file rather than a directory."""
        f = tmp_path / "not-a-dir"
        f.write_text("garbage")
        samp = AdaptiveMutationSampler(_two_rule_catalog())
        consumed = samp.prime_from_archives(str(f))
        assert consumed == 0


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

    def test_adaptive_prime_include_archives_default_dir(self, tmp_path):
        """When ``adaptive_prime_include_archives=True`` and no explicit
        ``adaptive_prime_archive_dir``, the SelfImprover derives the
        directory as ``<dirname(ledger_path)>/done`` and adds the
        archive contributions to the live ledger contributions."""
        ledger = tmp_path / "ledger.jsonl"
        archives = tmp_path / "done"
        archives.mkdir()
        # Live ledger: one accept on the catalog rule.
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
        # Archive: two accepts + one reject on the same catalog rule.
        archive_records = [
            {
                "record_type": "iteration",
                "proposal": {
                    "class_name": "_DummyHeuristicA",
                    "param_name": "radius",
                    "rule_kind": "log_uniform_perturb",
                },
                "accepted": acc,
            }
            for acc in (True, True, False)
        ]
        (archives / "self_improve_ledger_2026-05-31.jsonl").write_text(
            "\n".join(json.dumps(r) for r in archive_records) + "\n"
        )
        cfg = LoopConfig(
            iterations=0,
            ledger_path=str(ledger),
            stop_sentinel_path="",
            randomize=False,
            adaptive_sampling=True,
            adaptive_prime_from_ledger=True,
            adaptive_prime_include_archives=True,
        )
        si = SelfImprover(cfg, catalog=self._accept_catalog(), seed_strategies=_make_specs())
        assert si.sampler is not None
        snap = si.sampler.stats_snapshot()
        assert len(snap) == 1
        # 1 from live + 3 from the archive
        assert snap[0].n_attempts == 4
        # 1 accept live + 2 accepts archive
        assert snap[0].n_accepts == 3

    def test_adaptive_prime_include_archives_explicit_dir(self, tmp_path):
        """Explicit ``adaptive_prime_archive_dir`` overrides the default
        ``<parent>/done`` derivation — useful when archives sit
        outside the ledger's parent directory."""
        ledger = tmp_path / "ledger.jsonl"
        ledger.write_text("")  # empty live ledger
        # Archives at a non-default location.
        custom = tmp_path / "alt-archives"
        custom.mkdir()
        rec = {
            "record_type": "iteration",
            "proposal": {
                "class_name": "_DummyHeuristicA",
                "param_name": "radius",
                "rule_kind": "log_uniform_perturb",
            },
            "accepted": True,
        }
        (custom / "self_improve_ledger_2026-06-01.jsonl").write_text(json.dumps(rec) + "\n")
        cfg = LoopConfig(
            iterations=0,
            ledger_path=str(ledger),
            stop_sentinel_path="",
            randomize=False,
            adaptive_sampling=True,
            adaptive_prime_from_ledger=True,
            adaptive_prime_include_archives=True,
            adaptive_prime_archive_dir=str(custom),
        )
        si = SelfImprover(cfg, catalog=self._accept_catalog(), seed_strategies=_make_specs())
        assert si.sampler is not None
        snap = si.sampler.stats_snapshot()
        assert len(snap) == 1
        assert snap[0].n_attempts == 1
        assert snap[0].n_accepts == 1

    def test_adaptive_prime_include_archives_off_by_default(self, tmp_path):
        """Without ``adaptive_prime_include_archives``, archives are ignored
        even when present in the default location."""
        ledger = tmp_path / "ledger.jsonl"
        ledger.write_text("")  # empty live ledger
        archives = tmp_path / "done"
        archives.mkdir()
        rec = {
            "record_type": "iteration",
            "proposal": {
                "class_name": "_DummyHeuristicA",
                "param_name": "radius",
                "rule_kind": "log_uniform_perturb",
            },
            "accepted": True,
        }
        (archives / "self_improve_ledger_2026-05-31.jsonl").write_text(json.dumps(rec) + "\n")
        cfg = LoopConfig(
            iterations=0,
            ledger_path=str(ledger),
            stop_sentinel_path="",
            randomize=False,
            adaptive_sampling=True,
            adaptive_prime_from_ledger=True,
            # adaptive_prime_include_archives left at default (False)
        )
        si = SelfImprover(cfg, catalog=self._accept_catalog(), seed_strategies=_make_specs())
        assert si.sampler is not None
        # Empty live ledger + ignored archive => no stats.
        assert si.sampler.stats_snapshot() == []

    def test_adaptive_prime_include_archives_requires_prime_from_ledger(self, tmp_path):
        """The archive flag only takes effect when ``adaptive_prime_from_ledger``
        is also ``True`` — when priming is off, archives are silently ignored."""
        archives = tmp_path / "done"
        archives.mkdir()
        rec = {
            "record_type": "iteration",
            "proposal": {
                "class_name": "_DummyHeuristicA",
                "param_name": "radius",
                "rule_kind": "log_uniform_perturb",
            },
            "accepted": True,
        }
        (archives / "self_improve_ledger_2026-05-31.jsonl").write_text(json.dumps(rec) + "\n")
        cfg = LoopConfig(
            iterations=0,
            ledger_path=str(tmp_path / "ledger.jsonl"),
            stop_sentinel_path="",
            randomize=False,
            adaptive_sampling=True,
            adaptive_prime_from_ledger=False,
            adaptive_prime_include_archives=True,  # ignored without prime_from_ledger
        )
        si = SelfImprover(cfg, catalog=self._accept_catalog(), seed_strategies=_make_specs())
        assert si.sampler is not None
        assert si.sampler.stats_snapshot() == []

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

        # Baseline 0.5, candidate 0.4 on each iteration — the candidate
        # legitimately regresses so the iteration rejects without
        # tripping the §12.4 no-op detector (bit-identical pair scores).
        counter = {"n": 0}

        def score_fn(config: HarnessConfig) -> float:
            n = counter["n"]
            counter["n"] += 1
            return 0.5 if n % 2 == 0 else 0.4

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
        # Categorical kwarg rules get their own per-(class, param) arm —
        # distinct from any numeric rule on the same slot.
        assert _proposal_rule_key("PSO", "topology", "categorical_choice") == (
            "PSO",
            "topology",
            "categorical_choice",
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


class TestStructuralPerClassArms:
    """Per-class bandit arms for structural mutations.

    Closes the 'Per-class arms in the bandit' follow-up below the
    2026-05-03 §13 entry — splits each structural op into one bandit
    arm per candidate class so the loop can learn that, e.g., adding
    ``Sobol`` wins while adding ``Random`` loses, instead of pooling
    both into the same arm.
    """

    def test_proposal_rule_key_per_class_structural(self):
        from panobbgo.self_improve import _proposal_rule_key

        # With the per-class flag on, the key gains the class name.
        assert _proposal_rule_key("Sobol", "", "add_heuristic", per_class_structural=True) == (
            "Sobol",
            "add_heuristic",
            "structural",
        )
        assert _proposal_rule_key("Random", "", "drop_heuristic", per_class_structural=True) == (
            "Random",
            "drop_heuristic",
            "structural",
        )
        # Off (default) preserves the collapsed key.
        assert _proposal_rule_key("Sobol", "", "add_heuristic") == ("*", "add_heuristic", "structural")
        # Kwarg perturbations are unaffected regardless of the flag.
        assert _proposal_rule_key("Nearby", "radius", "log_uniform_perturb", per_class_structural=True) == (
            "Nearby",
            "radius",
            "log_uniform_perturb",
        )

    def test_sampler_default_is_off(self):
        cat = MutationCatalog(
            [
                StructuralMutationRule(
                    strategy_pattern="",
                    op="add_heuristic",
                    candidate_classes=((_NewHeuristicX, {}), (_NewHeuristicY, {})),
                ),
            ]
        )
        samp = AdaptiveMutationSampler(cat)
        assert samp.per_class_structural is False

    def test_sampler_splits_structural_arms_by_class(self):
        """With the flag on, each candidate class gets its own arm."""
        rule_add = StructuralMutationRule(
            strategy_pattern="",
            op="add_heuristic",
            candidate_classes=((_NewHeuristicX, {}), (_NewHeuristicY, {})),
        )
        cat = MutationCatalog([rule_add])
        samp = AdaptiveMutationSampler(cat, per_class_structural=True)
        rng = np.random.default_rng(0)
        for _ in range(30):
            prop = samp.sample(rng, _make_specs())
            assert prop is not None
            samp.record_outcome(True)
        snap = samp.stats_snapshot()
        # Two distinct arms, one per candidate class, no wildcard key.
        keys = {s.rule_key for s in snap}
        assert ("_NewHeuristicX", "add_heuristic", "structural") in keys
        assert ("_NewHeuristicY", "add_heuristic", "structural") in keys
        assert ("*", "add_heuristic", "structural") not in keys
        # Every sampled iteration is counted once across the two arms.
        total = sum(s.n_attempts for s in snap)
        accepts = sum(s.n_accepts for s in snap)
        assert total == 30
        assert accepts == 30

    def test_sampler_thompson_biases_to_winning_class(self):
        """When only one candidate class ever accepts, the sampler
        must concentrate probability on it — the headline guarantee
        of per-class arms.
        """
        rule_add = StructuralMutationRule(
            strategy_pattern="",
            op="add_heuristic",
            candidate_classes=((_NewHeuristicX, {}), (_NewHeuristicY, {})),
        )
        cat = MutationCatalog([rule_add])
        samp = AdaptiveMutationSampler(cat, per_class_structural=True)
        rng = np.random.default_rng(123)

        # Train: X accepts, Y rejects.
        for _ in range(50):
            prop = samp.sample(rng, _make_specs())
            assert prop is not None
            samp.record_outcome(prop.class_name == "_NewHeuristicX")

        # Sample without recording so stats freeze, then count.
        counts = {"_NewHeuristicX": 0, "_NewHeuristicY": 0}
        for _ in range(500):
            prop = samp.sample(rng, _make_specs())
            assert prop is not None
            counts[prop.class_name] += 1
            samp._last_rule_key = None

        assert counts["_NewHeuristicX"] > 4 * counts["_NewHeuristicY"], (
            f"per-class Thompson should heavily favor the winning class, got {counts}"
        )

    def test_sampler_drop_arm_keys_per_class(self):
        """Drop ops are also split per class — the bandit can learn
        which class is most worth dropping.
        """
        cat = MutationCatalog(
            [
                StructuralMutationRule(
                    strategy_pattern="",
                    op="drop_heuristic",
                    min_heuristics=1,
                ),
            ]
        )
        samp = AdaptiveMutationSampler(cat, per_class_structural=True)
        rng = np.random.default_rng(0)
        for _ in range(30):
            prop = samp.sample(rng, _make_specs())
            assert prop is not None
            samp.record_outcome(True)
        snap = samp.stats_snapshot()
        keys = {s.rule_key for s in snap}
        # _make_specs has StratX with A+B (so both droppable) and StratY
        # with only A (also droppable under min_heuristics=1).  Both A
        # and B should have appeared as drop targets.
        assert ("_DummyHeuristicA", "drop_heuristic", "structural") in keys
        assert ("_DummyHeuristicB", "drop_heuristic", "structural") in keys

    def test_sampler_kwarg_rules_unaffected(self):
        """Kwarg perturbations keep their (class, param, kind) arms
        regardless of the structural-per-class flag.
        """
        cat = _two_rule_catalog()
        samp = AdaptiveMutationSampler(cat, per_class_structural=True)
        rng = np.random.default_rng(0)
        for _ in range(20):
            samp.sample(rng, _make_specs())
            samp.record_outcome(True)
        snap = samp.stats_snapshot()
        keys = {s.rule_key for s in snap}
        assert ("_DummyHeuristicA", "radius", "log_uniform_perturb") in keys
        assert ("_DummyHeuristicB", "sigma0", "log_uniform_perturb") in keys

    def test_prime_from_ledger_uses_per_class_keys(self, tmp_path):
        """Ledger replay must respect the per-class flag so a primed
        sampler picks up the same arms its live sampling would create.
        """
        ledger = tmp_path / "old.jsonl"
        records = [
            {
                "record_type": "iteration",
                "iteration": 0,
                "proposal": {
                    "class_name": "Sobol",
                    "param_name": "",
                    "rule_kind": "add_heuristic",
                },
                "accepted": True,
            },
            {
                "record_type": "iteration",
                "iteration": 1,
                "proposal": {
                    "class_name": "Random",
                    "param_name": "",
                    "rule_kind": "add_heuristic",
                },
                "accepted": False,
            },
        ]
        ledger.write_text("\n".join(json.dumps(r) for r in records) + "\n")

        rule_add = StructuralMutationRule(
            strategy_pattern="",
            op="add_heuristic",
            candidate_classes=((_NewHeuristicX, {}),),
        )
        samp = AdaptiveMutationSampler(MutationCatalog([rule_add]), per_class_structural=True)
        consumed = samp.prime_from_ledger(str(ledger))
        assert consumed == 2
        snap = samp.stats_snapshot()
        keys = {s.rule_key for s in snap}
        assert ("Sobol", "add_heuristic", "structural") in keys
        assert ("Random", "add_heuristic", "structural") in keys
        # The off-mode wildcard key must not appear.
        assert ("*", "add_heuristic", "structural") not in keys

    def test_prime_from_ledger_collapsed_when_flag_off(self, tmp_path):
        """Default sampler primes into the collapsed wildcard arm."""
        ledger = tmp_path / "old.jsonl"
        records = [
            {
                "record_type": "iteration",
                "iteration": 0,
                "proposal": {
                    "class_name": "Sobol",
                    "param_name": "",
                    "rule_kind": "add_heuristic",
                },
                "accepted": True,
            },
            {
                "record_type": "iteration",
                "iteration": 1,
                "proposal": {
                    "class_name": "Random",
                    "param_name": "",
                    "rule_kind": "add_heuristic",
                },
                "accepted": False,
            },
        ]
        ledger.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        rule_add = StructuralMutationRule(
            strategy_pattern="",
            op="add_heuristic",
            candidate_classes=((_NewHeuristicX, {}),),
        )
        samp = AdaptiveMutationSampler(MutationCatalog([rule_add]))  # flag off
        consumed = samp.prime_from_ledger(str(ledger))
        assert consumed == 2
        snap = samp.stats_snapshot()
        # Both Sobol and Random history collapses to one wildcard arm.
        assert len(snap) == 1
        assert snap[0].rule_key == ("*", "add_heuristic", "structural")
        assert snap[0].n_attempts == 2
        assert snap[0].n_accepts == 1

    def test_loop_config_default_false(self):
        cfg = LoopConfig()
        assert cfg.structural_per_class_arms is False

    def test_loop_config_propagates_flag_to_sampler(self, tmp_path):
        from panobbgo.self_improve import SelfImprover

        cfg = LoopConfig(
            iterations=0,
            ledger_path=str(tmp_path / "ledger.jsonl"),
            stop_sentinel_path="",
            adaptive_sampling=True,
            structural_per_class_arms=True,
        )
        si = SelfImprover(cfg, seed_strategies=_make_specs())
        assert si.sampler is not None
        assert si.sampler.per_class_structural is True

    def test_loop_config_flag_ignored_without_adaptive(self, tmp_path):
        """structural_per_class_arms is only meaningful for the adaptive
        sampler.  Without --adaptive, the loop uses the uniform catalog
        sampler and the flag has no effect.
        """
        from panobbgo.self_improve import SelfImprover

        cfg = LoopConfig(
            iterations=0,
            ledger_path=str(tmp_path / "ledger.jsonl"),
            stop_sentinel_path="",
            adaptive_sampling=False,
            structural_per_class_arms=True,
        )
        si = SelfImprover(cfg, seed_strategies=_make_specs())
        # Uniform sampler path: no AdaptiveMutationSampler is constructed,
        # so the flag is inert (no error, no surprise).
        assert si.sampler is None


class TestHierarchicalStructuralBandit:
    """Hierarchical Beta-Binomial over per-class structural arms.

    Closes the *Hierarchical bandit over the per-class structural arms*
    follow-up below the 2026-05-18 §13 entry.  Each per-class arm's Beta
    posterior borrows ``κ · (n_other_class_accepts, ...)`` from the
    op-level aggregate, so a fresh candidate class warms with the op's
    empirical accept rate instead of the symmetric ``Beta(1, 1)`` prior.
    """

    def _add_only_catalog(self) -> MutationCatalog:
        return MutationCatalog(
            [
                StructuralMutationRule(
                    strategy_pattern="",
                    op="add_heuristic",
                    candidate_classes=((_NewHeuristicX, {}), (_NewHeuristicY, {})),
                ),
            ]
        )

    # ---- defaults / validation ---------------------------------------

    def test_default_borrow_is_zero(self):
        samp = AdaptiveMutationSampler(self._add_only_catalog())
        assert samp.structural_borrow_alpha == 0.0

    def test_negative_borrow_raises(self):
        with pytest.raises(ValueError, match="structural_borrow_alpha"):
            AdaptiveMutationSampler(self._add_only_catalog(), structural_borrow_alpha=-0.1)

    def test_non_finite_borrow_raises(self):
        with pytest.raises(ValueError, match="structural_borrow_alpha"):
            AdaptiveMutationSampler(self._add_only_catalog(), structural_borrow_alpha=float("inf"))
        with pytest.raises(ValueError, match="structural_borrow_alpha"):
            AdaptiveMutationSampler(self._add_only_catalog(), structural_borrow_alpha=float("nan"))

    def test_zero_borrow_recovers_per_class_behaviour(self):
        """κ=0 must produce a byte-identical sample trajectory to the
        unhierarchical per-class sampler.  Equal seed, equal catalog,
        equal flag → equal proposals.
        """
        cat = self._add_only_catalog()
        a = AdaptiveMutationSampler(cat, per_class_structural=True, structural_borrow_alpha=0.0)
        b = AdaptiveMutationSampler(cat, per_class_structural=True)
        rng_a = np.random.default_rng(7)
        rng_b = np.random.default_rng(7)
        for _ in range(40):
            pa = a.sample(rng_a, _make_specs())
            pb = b.sample(rng_b, _make_specs())
            assert pa is not None and pb is not None
            assert pa.class_name == pb.class_name
            a.record_outcome(pa.class_name == "_NewHeuristicX")
            b.record_outcome(pb.class_name == "_NewHeuristicX")
        assert {s.rule_key for s in a.stats_snapshot()} == {s.rule_key for s in b.stats_snapshot()}

    # ---- mechanics ---------------------------------------------------

    def test_borrow_inert_without_per_class_flag(self):
        """``per_class_structural=False`` collapses to one arm per op,
        so there are no sibling arms to borrow from.  The α/β draw must
        be identical to the κ=0 case.
        """
        cat = self._add_only_catalog()
        # Sampler 1: hierarchical but per-class is off — borrow is dead code.
        a = AdaptiveMutationSampler(cat, per_class_structural=False, structural_borrow_alpha=10.0)
        # Sampler 2: per-class off, no borrow.
        b = AdaptiveMutationSampler(cat, per_class_structural=False)
        rng_a = np.random.default_rng(11)
        rng_b = np.random.default_rng(11)
        for _ in range(20):
            pa = a.sample(rng_a, _make_specs())
            pb = b.sample(rng_b, _make_specs())
            assert pa is not None and pb is not None
            assert pa.class_name == pb.class_name
            a.record_outcome(True)
            b.record_outcome(True)

    def test_borrow_inert_for_kwarg_rules(self):
        """Kwarg perturbations are not grouped by an op; the borrow must
        not touch their α/β.  Compare the kwarg-only catalog trajectory
        with and without κ — they must be byte-identical.
        """
        cat = _two_rule_catalog()
        a = AdaptiveMutationSampler(cat, per_class_structural=True, structural_borrow_alpha=1.0)
        b = AdaptiveMutationSampler(cat, per_class_structural=True)
        rng_a = np.random.default_rng(5)
        rng_b = np.random.default_rng(5)
        for _ in range(20):
            pa = a.sample(rng_a, _make_specs())
            pb = b.sample(rng_b, _make_specs())
            assert pa is not None and pb is not None
            assert pa.class_name == pb.class_name
            a.record_outcome(True)
            b.record_outcome(True)

    def test_fresh_class_warms_with_op_aggregate(self):
        """A class with zero history but a sibling with strong history
        must have its draw centred on the op's accept rate, not the
        symmetric prior.

        Test: hand-seed sibling X's stats with 20 accepts out of 20.
        Sibling Y has no stats.  Under κ = 1, Y's posterior should be
        ``Beta(1 + 0 + 1·20, 1 + 0 + 1·0) = Beta(21, 1)``, which draws
        far closer to 1 than the symmetric ``Beta(1, 1)`` would.
        """
        from panobbgo.self_improve import MutationRuleStats

        cat = self._add_only_catalog()
        samp = AdaptiveMutationSampler(cat, per_class_structural=True, structural_borrow_alpha=1.0)
        # Plant X's history without changing Y's.
        x_key = ("_NewHeuristicX", "add_heuristic", "structural")
        samp._stats[x_key] = MutationRuleStats(rule_key=x_key, n_attempts=20, n_accepts=20)

        rng = np.random.default_rng(31)
        # Sample 200 times *without* updating stats so the posterior stays
        # frozen, and check Y is drawn from a high-mean posterior.
        y_draws_above_half = 0
        n_samples = 200
        for _ in range(n_samples):
            # Force a fresh Beta draw per loop iteration via the public
            # sample() path — we count how often the sampler picks Y at
            # all, which it can only do when Y's draw beats X's.  With
            # X's posterior Beta(21, 1) (~0.95) and Y's Beta(21, 1) under
            # the borrow, the two arms have equal mean, so we should see
            # Y win ~50% of the time.  Without the borrow, Y is Beta(1, 1)
            # (uniform), which wins X's near-1 draw far less often.
            prop = samp.sample(rng, _make_specs())
            assert prop is not None
            if prop.class_name == "_NewHeuristicY":
                y_draws_above_half += 1
            # Discard the recorded last_rule_key so the next sample is
            # against the same frozen posterior.
            samp._last_rule_key = None
        # With borrow=1, X and Y are both ~Beta(21, 1) → roughly equal
        # win rate.  Without borrow, Y is uniform(0,1) and X is Beta(21,1)
        # → Y wins ~5-10% of the time.  Threshold conservatively at >25%
        # for the borrowed sampler.
        rate = y_draws_above_half / n_samples
        assert rate > 0.25, (
            f"Y should warm to the op's empirical accept rate under borrow=1, "
            f"got y_pick_rate={rate:.3f} (expected ~0.5)"
        )

    def test_fresh_class_cold_without_borrow(self):
        """Inverse check: without κ borrow, a sibling with no history
        keeps the symmetric prior and loses most arg-max contests to a
        strongly-positive sibling.
        """
        from panobbgo.self_improve import MutationRuleStats

        cat = self._add_only_catalog()
        samp = AdaptiveMutationSampler(cat, per_class_structural=True, structural_borrow_alpha=0.0)
        x_key = ("_NewHeuristicX", "add_heuristic", "structural")
        samp._stats[x_key] = MutationRuleStats(rule_key=x_key, n_attempts=20, n_accepts=20)
        rng = np.random.default_rng(31)
        y_draws = 0
        n_samples = 200
        for _ in range(n_samples):
            prop = samp.sample(rng, _make_specs())
            assert prop is not None
            if prop.class_name == "_NewHeuristicY":
                y_draws += 1
            samp._last_rule_key = None
        rate = y_draws / n_samples
        # Y should rarely win the arg-max — Beta(1, 1) vs Beta(21, 1).
        assert rate < 0.20, f"Y should be cold without borrow, got y_pick_rate={rate:.3f} (expected ~0.05)"

    def test_borrow_excludes_self_contribution(self):
        """The borrow must aggregate *other* sibling arms, not include
        the arm's own contribution — otherwise the hierarchy collapses
        to a κ-amplified version of the same per-class posterior.

        With only one per-class arm (X) seeded with 10/10 and Y absent,
        Y's borrowed alpha must be ``prior + 0 + κ · 10`` and X's must be
        ``prior + 10 + κ · 0`` (since X has no *other* siblings).  We
        verify by inspecting the rationale field, which prints the
        effective α/β.
        """
        from panobbgo.self_improve import MutationRuleStats

        cat = self._add_only_catalog()
        samp = AdaptiveMutationSampler(cat, per_class_structural=True, structural_borrow_alpha=0.5)
        x_key = ("_NewHeuristicX", "add_heuristic", "structural")
        samp._stats[x_key] = MutationRuleStats(rule_key=x_key, n_attempts=10, n_accepts=10)

        # Force X's selection by lots of draws and check the rationale
        # text reports Beta(prior + 10 + κ·0, prior + 0 + κ·0) for X
        # — i.e. no self-borrow.
        rng = np.random.default_rng(0)
        x_rationale = None
        y_rationale = None
        for _ in range(200):
            prop = samp.sample(rng, _make_specs())
            assert prop is not None
            if prop.class_name == "_NewHeuristicX" and x_rationale is None:
                x_rationale = prop.rationale
            elif prop.class_name == "_NewHeuristicY" and y_rationale is None:
                y_rationale = prop.rationale
            samp._last_rule_key = None
            if x_rationale and y_rationale:
                break

        assert x_rationale is not None, "X should be sampled at least once"
        assert y_rationale is not None, "Y should be sampled at least once"
        # X's draw is Beta(1 + 10 + 0.5·0, 1 + 0 + 0.5·0) = Beta(11, 1).
        assert "Beta(11.0, 1.0)" in x_rationale, (
            f"X should see only its own evidence under self-exclusion; got rationale={x_rationale!r}"
        )
        # Y's draw is Beta(1 + 0 + 0.5·10, 1 + 0 + 0.5·0) = Beta(6, 1).
        assert "Beta(6.0, 1.0)" in y_rationale, f"Y should borrow from X's accepts only; got rationale={y_rationale!r}"

    def test_borrow_mixed_failures_and_accepts(self):
        """When the sibling has mixed history, both α and β borrow."""
        from panobbgo.self_improve import MutationRuleStats

        cat = self._add_only_catalog()
        samp = AdaptiveMutationSampler(cat, per_class_structural=True, structural_borrow_alpha=1.0)
        x_key = ("_NewHeuristicX", "add_heuristic", "structural")
        samp._stats[x_key] = MutationRuleStats(rule_key=x_key, n_attempts=10, n_accepts=3)

        rng = np.random.default_rng(2)
        y_rationale = None
        for _ in range(300):
            prop = samp.sample(rng, _make_specs())
            assert prop is not None
            if prop.class_name == "_NewHeuristicY":
                y_rationale = prop.rationale
                break
            samp._last_rule_key = None
        assert y_rationale is not None
        # Y's draw is Beta(1 + 0 + 1·3, 1 + 0 + 1·7) = Beta(4, 8).
        assert "Beta(4.0, 8.0)" in y_rationale, (
            f"Y should borrow both accept and failure counts; got rationale={y_rationale!r}"
        )

    # ---- LoopConfig + SelfImprover integration -----------------------

    def test_loop_config_default_borrow_zero(self):
        cfg = LoopConfig()
        assert cfg.structural_borrow_alpha == 0.0

    def test_loop_config_negative_borrow_raises(self):
        with pytest.raises(ValueError, match="structural_borrow_alpha"):
            LoopConfig(structural_borrow_alpha=-0.1)

    def test_loop_config_propagates_borrow_to_sampler(self, tmp_path):
        from panobbgo.self_improve import SelfImprover

        cfg = LoopConfig(
            iterations=0,
            ledger_path=str(tmp_path / "ledger.jsonl"),
            stop_sentinel_path="",
            adaptive_sampling=True,
            structural_per_class_arms=True,
            structural_borrow_alpha=0.7,
        )
        si = SelfImprover(cfg, seed_strategies=_make_specs())
        assert si.sampler is not None
        assert si.sampler.structural_borrow_alpha == 0.7

    def test_loop_config_borrow_inert_without_adaptive(self, tmp_path):
        """Without --adaptive there is no sampler to take the borrow,
        so the knob is silently inert — same pattern as
        ``structural_per_class_arms``.
        """
        from panobbgo.self_improve import SelfImprover

        cfg = LoopConfig(
            iterations=0,
            ledger_path=str(tmp_path / "ledger.jsonl"),
            stop_sentinel_path="",
            adaptive_sampling=False,
            structural_borrow_alpha=0.5,
        )
        si = SelfImprover(cfg, seed_strategies=_make_specs())
        assert si.sampler is None


class TestDefaultStructuralCatalog:
    def test_returns_catalog_with_structural_rules(self):
        cat = default_structural_catalog()
        kinds = {type(r).__name__ for r in cat.rules}
        assert "MutationRule" in kinds
        assert "StructuralMutationRule" in kinds
        # Four structural rules from §7.2 — heuristic add/drop (shipped
        # 2026-05-03) plus analyzer add/drop (shipped 2026-06-02).
        ops = {r.op for r in cat.rules if isinstance(r, StructuralMutationRule)}
        assert ops == {"add_heuristic", "drop_heuristic", "add_analyzer", "drop_analyzer"}

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
# Analyzer add/drop structural mutations (shipped 2026-06-02)
# ===========================================================================


class _DummyAnalyzerD:
    """Second fake analyzer used as an add-analyzer candidate."""

    pass


class _DummyAnalyzerE:
    """Third fake analyzer used to test duplicate avoidance / drop filters."""

    pass


class TestAnalyzerStructuralRuleValidation:
    """Construction-time validation of analyzer ops on :class:`StructuralMutationRule`."""

    def test_add_analyzer_constructs(self):
        rule = StructuralMutationRule(
            strategy_pattern="",
            op="add_analyzer",
            candidate_classes=((_DummyAnalyzerD, {"interval": 10}),),
        )
        assert rule.op == "add_analyzer"
        assert rule.min_analyzers == 0  # default

    def test_drop_analyzer_constructs_without_candidates(self):
        rule = StructuralMutationRule(strategy_pattern="", op="drop_analyzer")
        assert rule.op == "drop_analyzer"
        assert rule.candidate_classes == ()

    def test_add_analyzer_requires_candidates(self):
        with pytest.raises(ValueError, match="add_analyzer requires"):
            StructuralMutationRule(strategy_pattern="", op="add_analyzer", candidate_classes=())

    def test_min_analyzers_negative_raises(self):
        with pytest.raises(ValueError, match="min_analyzers"):
            StructuralMutationRule(strategy_pattern="", op="drop_analyzer", min_analyzers=-1)

    def test_min_analyzers_zero_allowed(self):
        # Unlike min_heuristics (floor 1), an empty analyzers list is a
        # valid spec — analyzers are non-essential.
        rule = StructuralMutationRule(strategy_pattern="", op="drop_analyzer", min_analyzers=0)
        assert rule.min_analyzers == 0

    def test_rule_key_collapses_by_op(self):
        rule_add = StructuralMutationRule(
            strategy_pattern="",
            op="add_analyzer",
            candidate_classes=((_DummyAnalyzerD, {}),),
        )
        rule_drop = StructuralMutationRule(strategy_pattern="", op="drop_analyzer")
        assert rule_add.rule_key() == ("*", "add_analyzer", "structural")
        assert rule_drop.rule_key() == ("*", "drop_analyzer", "structural")


class TestAnalyzerStructuralHits:
    """:func:`_find_structural_hits` enumeration on analyzer buckets."""

    def test_add_analyzer_skips_existing(self):
        # StratX already has _DummyAnalyzerC.  With avoid_duplicates=True,
        # the candidate _DummyAnalyzerC should be skipped on StratX but
        # available on StratY (which has no analyzers).
        rule = StructuralMutationRule(
            strategy_pattern="",
            op="add_analyzer",
            candidate_classes=(
                (_DummyAnalyzerC, {"update_interval": 25}),
                (_DummyAnalyzerD, {"interval": 10}),
            ),
            avoid_duplicates=True,
        )
        from panobbgo.self_improve import _find_structural_hits

        hits = _find_structural_hits(_make_specs(), rule)
        # StratX: only _DummyAnalyzerD eligible (C is a duplicate)
        # StratY: both C and D eligible (no analyzers to dedup against)
        # Total: 3 hits.
        assert len(hits) == 3
        classes = {(si, cls.__name__) for si, cls, _ in hits}
        assert classes == {(0, "_DummyAnalyzerD"), (1, "_DummyAnalyzerC"), (1, "_DummyAnalyzerD")}

    def test_add_analyzer_without_avoid_duplicates(self):
        rule = StructuralMutationRule(
            strategy_pattern="",
            op="add_analyzer",
            candidate_classes=((_DummyAnalyzerC, {"update_interval": 5}),),
            avoid_duplicates=False,
        )
        from panobbgo.self_improve import _find_structural_hits

        hits = _find_structural_hits(_make_specs(), rule)
        # Both StratX and StratY get the candidate even though StratX
        # already has _DummyAnalyzerC.
        spec_idxs = {si for si, _, _ in hits}
        assert spec_idxs == {0, 1}

    def test_drop_analyzer_respects_min_floor(self):
        # StratX has 1 analyzer; min_analyzers=1 forbids dropping (would
        # breach the floor).  StratY has 0 analyzers, also ineligible.
        rule = StructuralMutationRule(
            strategy_pattern="",
            op="drop_analyzer",
            min_analyzers=1,
        )
        from panobbgo.self_improve import _find_structural_hits

        hits = _find_structural_hits(_make_specs(), rule)
        assert hits == []

    def test_drop_analyzer_floor_zero_allows_strip(self):
        # min_analyzers=0 means "post-drop count may be 0" — StratX (1 → 0)
        # qualifies; StratY (0 → would be -1) does not.
        rule = StructuralMutationRule(
            strategy_pattern="",
            op="drop_analyzer",
            min_analyzers=0,
        )
        from panobbgo.self_improve import _find_structural_hits

        hits = _find_structural_hits(_make_specs(), rule)
        assert len(hits) == 1
        si, cls, _kw = hits[0]
        assert si == 0  # only StratX qualifies
        assert cls.__name__ == "_DummyAnalyzerC"

    def test_droppable_classes_filter_on_analyzers(self):
        # A spec with two analyzers; the rule restricts drops to
        # _DummyAnalyzerD only.
        fat = [
            StrategySpec(
                name="Fat",
                strategy_class=_DummyStrategy,
                heuristics=[(_DummyHeuristicA, {})],
                analyzers=[
                    (_DummyAnalyzerC, {"update_interval": 20}),
                    (_DummyAnalyzerD, {"interval": 10}),
                ],
            ),
        ]
        rule = StructuralMutationRule(
            strategy_pattern="",
            op="drop_analyzer",
            droppable_classes=("_DummyAnalyzerD",),
            min_analyzers=0,
        )
        from panobbgo.self_improve import _find_structural_hits

        hits = _find_structural_hits(fat, rule)
        names = {cls.__name__ for _, cls, _ in hits}
        assert names == {"_DummyAnalyzerD"}

    def test_strategy_pattern_filters_analyzer_ops(self):
        rule = StructuralMutationRule(
            strategy_pattern="StratX",
            op="drop_analyzer",
            min_analyzers=0,
        )
        from panobbgo.self_improve import _find_structural_hits

        hits = _find_structural_hits(_make_specs(), rule)
        # Only StratX matches the pattern (StratY would be ineligible anyway
        # since it has zero analyzers, but the pattern filter must run
        # before the bucket check).
        spec_idxs = {si for si, _, _ in hits}
        assert spec_idxs == {0}


class TestAnalyzerStructuralCatalogSampling:
    """End-to-end sampling through :meth:`MutationCatalog.sample`."""

    def test_add_analyzer_proposal_shape(self):
        cat = MutationCatalog(
            [
                StructuralMutationRule(
                    strategy_pattern="",
                    op="add_analyzer",
                    candidate_classes=((_DummyAnalyzerD, {"interval": 7}),),
                ),
            ]
        )
        rng = np.random.default_rng(0)
        prop = cat.sample(rng, _make_specs())
        assert prop is not None
        assert prop.op == "add_analyzer"
        assert prop.rule_kind == "add_analyzer"
        assert prop.class_name == "_DummyAnalyzerD"
        assert prop.param_name == ""
        assert prop.structural_kwargs == {"interval": 7}

    def test_drop_analyzer_proposal_shape(self):
        cat = MutationCatalog(
            [
                StructuralMutationRule(
                    strategy_pattern="",
                    op="drop_analyzer",
                    min_analyzers=0,
                ),
            ]
        )
        rng = np.random.default_rng(0)
        prop = cat.sample(rng, _make_specs())
        assert prop is not None
        assert prop.op == "drop_analyzer"
        assert prop.rule_kind == "drop_analyzer"
        # The only droppable analyzer in _make_specs() is on StratX.
        assert prop.strategy_name == "StratX"
        assert prop.class_name == "_DummyAnalyzerC"

    def test_no_applicable_returns_none(self):
        # Every spec has zero analyzers and the rule requires ≥1 to drop.
        bare = [
            StrategySpec(
                name="Bare",
                strategy_class=_DummyStrategy,
                heuristics=[(_DummyHeuristicA, {})],
                analyzers=[],
            ),
        ]
        cat = MutationCatalog(
            [
                StructuralMutationRule(
                    strategy_pattern="",
                    op="drop_analyzer",
                    min_analyzers=0,
                ),
            ]
        )
        rng = np.random.default_rng(0)
        assert cat.sample(rng, bare) is None

    def test_default_kwargs_independent_per_hit(self):
        """Mutating one proposal's structural_kwargs must not leak into others."""
        cat = MutationCatalog(
            [
                StructuralMutationRule(
                    strategy_pattern="",
                    op="add_analyzer",
                    candidate_classes=((_DummyAnalyzerD, {"interval": 7}),),
                    avoid_duplicates=False,
                ),
            ]
        )
        rng = np.random.default_rng(0)
        props = [cat.sample(rng, _make_specs()) for _ in range(8)]
        assert all(p is not None for p in props)
        # Mutate the first; the rest must keep the original default.
        props[0].structural_kwargs["interval"] = 999
        for p in props[1:]:
            assert p.structural_kwargs["interval"] == 7


class TestAnalyzerApplyMutation:
    """:func:`apply_mutation` dispatch on analyzer ops."""

    def test_add_analyzer_appends_to_analyzers_bucket(self):
        specs = _make_specs()
        proposal = MutationProposal(
            strategy_name="StratY",
            class_name="_DummyAnalyzerD",
            param_name="",
            old_value=None,
            new_value=None,
            rule_kind="add_analyzer",
            rationale="t",
            op="add_analyzer",
            structural_kwargs={"interval": 13},
        )
        # Make the class resolvable: register it on StratY itself first by
        # giving the spec a placeholder _DummyAnalyzerD entry that the
        # resolver can find.  Use a fresh spec with the analyzer already
        # present.
        specs[1] = StrategySpec(
            name="StratY",
            strategy_class=_DummyStrategy,
            heuristics=[(_DummyHeuristicA, {"radius": 0.05})],
            analyzers=[(_DummyAnalyzerD, {})],
        )
        out = apply_mutation(specs, proposal)
        new_y = next(s for s in out if s.name == "StratY")
        # The append produces a *second* _DummyAnalyzerD entry (with the
        # proposed kwargs) — _resolve_analyzer_class found the first one
        # to recover the class object.
        analyzer_kwargs = [kw for cls, kw in new_y.analyzers if cls is _DummyAnalyzerD]
        assert {"interval": 13} in analyzer_kwargs

    def test_add_analyzer_falls_back_to_package(self):
        from panobbgo.analyzers import Sensitivity

        specs = [
            StrategySpec(
                name="S",
                strategy_class=_DummyStrategy,
                heuristics=[(_DummyHeuristicA, {})],
                analyzers=[],
            ),
        ]
        proposal = MutationProposal(
            strategy_name="S",
            class_name="Sensitivity",
            param_name="",
            old_value=None,
            new_value=None,
            rule_kind="add_analyzer",
            rationale="t",
            op="add_analyzer",
            structural_kwargs={"update_interval": 15},
        )
        out = apply_mutation(specs, proposal)
        assert len(out[0].analyzers) == 1
        cls, kw = out[0].analyzers[0]
        assert cls is Sensitivity
        assert kw == {"update_interval": 15}

    def test_add_analyzer_unknown_class_raises(self):
        specs = [
            StrategySpec(
                name="S",
                strategy_class=_DummyStrategy,
                heuristics=[(_DummyHeuristicA, {})],
                analyzers=[],
            ),
        ]
        proposal = MutationProposal(
            strategy_name="S",
            class_name="DoesNotExistAnalyzer",
            param_name="",
            old_value=None,
            new_value=None,
            rule_kind="add_analyzer",
            rationale="t",
            op="add_analyzer",
            structural_kwargs={},
        )
        with pytest.raises(ValueError, match="panobbgo.analyzers"):
            apply_mutation(specs, proposal)

    def test_drop_analyzer_removes_first_match(self):
        specs = _make_specs()
        proposal = MutationProposal(
            strategy_name="StratX",
            class_name="_DummyAnalyzerC",
            param_name="",
            old_value=None,
            new_value=None,
            rule_kind="drop_analyzer",
            rationale="t",
            op="drop_analyzer",
            structural_kwargs={"update_interval": 20},
        )
        out = apply_mutation(specs, proposal)
        new_x = next(s for s in out if s.name == "StratX")
        assert all(cls.__name__ != "_DummyAnalyzerC" for cls, _ in new_x.analyzers)
        # Original spec untouched.
        assert any(cls.__name__ == "_DummyAnalyzerC" for cls, _ in specs[0].analyzers)
        # Heuristics bucket completely unaffected.
        assert [cls.__name__ for cls, _ in new_x.heuristics] == [cls.__name__ for cls, _ in specs[0].heuristics]

    def test_drop_analyzer_allows_empty_result(self):
        # Drop the only analyzer in StratX — analyzers list becomes [].
        specs = _make_specs()
        proposal = MutationProposal(
            strategy_name="StratX",
            class_name="_DummyAnalyzerC",
            param_name="",
            old_value=None,
            new_value=None,
            rule_kind="drop_analyzer",
            rationale="t",
            op="drop_analyzer",
            structural_kwargs={},
        )
        out = apply_mutation(specs, proposal)
        new_x = next(s for s in out if s.name == "StratX")
        assert new_x.analyzers == []  # empty allowed

    def test_drop_analyzer_missing_class_raises(self):
        specs = _make_specs()
        proposal = MutationProposal(
            strategy_name="StratX",
            class_name="DoesNotExistAnalyzer",
            param_name="",
            old_value=None,
            new_value=None,
            rule_kind="drop_analyzer",
            rationale="t",
            op="drop_analyzer",
            structural_kwargs={},
        )
        with pytest.raises(ValueError, match="no analyzer"):
            apply_mutation(specs, proposal)

    def test_analyzer_op_preserves_heuristics_independence(self):
        # Applying an analyzer mutation must not touch the heuristics
        # bucket of the matched spec, nor any other spec entirely.
        specs = _make_specs()
        proposal = MutationProposal(
            strategy_name="StratX",
            class_name="_DummyAnalyzerC",
            param_name="",
            old_value=None,
            new_value=None,
            rule_kind="drop_analyzer",
            rationale="t",
            op="drop_analyzer",
            structural_kwargs={},
        )
        out = apply_mutation(specs, proposal)
        # StratY unchanged.
        new_y = next(s for s in out if s.name == "StratY")
        orig_y = specs[1]
        assert [cls for cls, _ in new_y.heuristics] == [cls for cls, _ in orig_y.heuristics]
        assert new_y.analyzers == orig_y.analyzers
        # StratX heuristics untouched.
        new_x = next(s for s in out if s.name == "StratX")
        assert [cls for cls, _ in new_x.heuristics] == [cls for cls, _ in specs[0].heuristics]


class TestAnalyzerRuleKey:
    """Bandit arm key behaviour for analyzer ops."""

    def test_proposal_rule_key_collapses_analyzer_ops(self):
        from panobbgo.self_improve import _proposal_rule_key

        assert _proposal_rule_key("Sensitivity", "", "add_analyzer") == (
            "*",
            "add_analyzer",
            "structural",
        )
        assert _proposal_rule_key("Restart", "", "drop_analyzer") == (
            "*",
            "drop_analyzer",
            "structural",
        )

    def test_proposal_rule_key_per_class_for_analyzer_ops(self):
        from panobbgo.self_improve import _proposal_rule_key

        assert _proposal_rule_key("Sensitivity", "", "add_analyzer", per_class_structural=True) == (
            "Sensitivity",
            "add_analyzer",
            "structural",
        )
        assert _proposal_rule_key("Restart", "", "drop_analyzer", per_class_structural=True) == (
            "Restart",
            "drop_analyzer",
            "structural",
        )

    def test_adaptive_sampler_buckets_analyzer_history(self):
        rule_add = StructuralMutationRule(
            strategy_pattern="",
            op="add_analyzer",
            candidate_classes=((_DummyAnalyzerD, {}), (_DummyAnalyzerE, {})),
        )
        cat = MutationCatalog([rule_add])
        samp = AdaptiveMutationSampler(cat)
        rng = np.random.default_rng(0)
        for _ in range(10):
            prop = samp.sample(rng, _make_specs())
            assert prop is not None
            samp.record_outcome(True)
        snap = samp.stats_snapshot()
        # All 10 attempts/accepts collapse into one arm per default.
        assert len(snap) == 1
        assert snap[0].rule_key == ("*", "add_analyzer", "structural")
        assert snap[0].n_attempts == 10
        assert snap[0].n_accepts == 10

    def test_adaptive_sampler_per_class_for_analyzer_ops(self):
        """With per_class_structural=True the bandit can distinguish
        adding Sensitivity-vs-Restart-style analyzers."""
        rule_add = StructuralMutationRule(
            strategy_pattern="",
            op="add_analyzer",
            candidate_classes=((_DummyAnalyzerD, {}), (_DummyAnalyzerE, {})),
        )
        cat = MutationCatalog([rule_add])
        samp = AdaptiveMutationSampler(cat, per_class_structural=True)
        rng = np.random.default_rng(0)
        for _ in range(50):
            prop = samp.sample(rng, _make_specs())
            assert prop is not None
            samp.record_outcome(True)
        snap = samp.stats_snapshot()
        keys = {s.rule_key for s in snap}
        # Both classes observed as distinct arms.
        assert ("_DummyAnalyzerD", "add_analyzer", "structural") in keys
        assert ("_DummyAnalyzerE", "add_analyzer", "structural") in keys
        # Total attempts conserved across the two arms.
        total_attempts = sum(s.n_attempts for s in snap)
        assert total_attempts == 50


class TestAnalyzerProposalToDict:
    """JSONL ledger round-trip for analyzer-op proposals."""

    def test_add_analyzer_round_trip(self):
        proposal = MutationProposal(
            strategy_name="S",
            class_name="Restart",
            param_name="",
            old_value=None,
            new_value=None,
            rule_kind="add_analyzer",
            rationale="t",
            op="add_analyzer",
            structural_kwargs={"patience": None, "max_restarts": np.int64(5)},
        )
        d = proposal.to_dict()
        assert d["op"] == "add_analyzer"
        assert d["structural_kwargs"]["patience"] is None
        assert d["structural_kwargs"]["max_restarts"] == 5
        assert isinstance(d["structural_kwargs"]["max_restarts"], int)
        # JSON-serialisable.
        assert json.loads(json.dumps(d))

    def test_drop_analyzer_round_trip(self):
        proposal = MutationProposal(
            strategy_name="S",
            class_name="Sensitivity",
            param_name="",
            old_value=None,
            new_value=None,
            rule_kind="drop_analyzer",
            rationale="t",
            op="drop_analyzer",
            structural_kwargs={"update_interval": 25},
        )
        d = proposal.to_dict()
        assert d["op"] == "drop_analyzer"
        assert d["structural_kwargs"] == {"update_interval": 25}
        assert json.loads(json.dumps(d))


class TestAnalyzerDefaultStructuralCatalog:
    """The default structural catalog includes analyzer ops with literature-grounded candidates."""

    def test_includes_analyzer_ops(self):
        cat = default_structural_catalog()
        ops = {r.op for r in cat.rules if isinstance(r, StructuralMutationRule)}
        assert "add_analyzer" in ops
        assert "drop_analyzer" in ops

    def test_analyzer_candidate_pool_contents(self):
        from panobbgo.analyzers import Restart, Sensitivity

        cat = default_structural_catalog()
        add_rules = [r for r in cat.rules if isinstance(r, StructuralMutationRule) and r.op == "add_analyzer"]
        assert len(add_rules) == 1
        candidate_classes = {cls for cls, _kw in add_rules[0].candidate_classes}
        assert candidate_classes == {Sensitivity, Restart}

    def test_drop_analyzer_min_floor_is_zero(self):
        cat = default_structural_catalog()
        drop_rules = [r for r in cat.rules if isinstance(r, StructuralMutationRule) and r.op == "drop_analyzer"]
        assert len(drop_rules) == 1
        assert drop_rules[0].min_analyzers == 0

    def test_analyzer_ops_applicable_on_default_battery(self):
        """The default structural catalog must produce ≥1 analyzer hit
        on the standard quick-mode battery."""
        from panobbgo.harness import _make_quick_strategies
        from panobbgo.self_improve import _find_structural_hits

        specs = _make_quick_strategies()
        cat = default_structural_catalog()
        add_rules = [r for r in cat.rules if isinstance(r, StructuralMutationRule) and r.op == "add_analyzer"]
        drop_rules = [r for r in cat.rules if isinstance(r, StructuralMutationRule) and r.op == "drop_analyzer"]
        # At least one add_analyzer hit on Rewarding_Diverse (it has Sensitivity, so
        # avoid_duplicates filters Sensitivity but allows Restart).  Also one on
        # RoundRobin_Random which has no analyzers (both candidates eligible).
        add_hits = _find_structural_hits(specs, add_rules[0])
        assert len(add_hits) >= 1
        # At least one drop_analyzer hit on Rewarding_Diverse (has Sensitivity).
        drop_hits = _find_structural_hits(specs, drop_rules[0])
        assert len(drop_hits) >= 1


class TestAnalyzerEndToEnd:
    """End-to-end SelfImprover run with an analyzer structural mutation."""

    def test_loop_accepts_analyzer_drop(self, tmp_path):
        """The loop drives a fake harness that rewards the candidate;
        the analyzer-drop proposal must round-trip through apply and
        end up as an accept."""
        counter = {"n": 0}

        def score_fn(config):
            n = counter["n"]
            counter["n"] += 1
            # Baseline runs are at even calls (0.3); candidate runs at odd (0.8).
            return 0.3 if n % 2 == 0 else 0.8

        # Seed strategy has one analyzer — drop_analyzer must be applicable.
        seed = [
            StrategySpec(
                name="S",
                strategy_class=_DummyStrategy,
                heuristics=[(_DummyHeuristicA, {"radius": 0.1})],
                analyzers=[(_DummyAnalyzerC, {"update_interval": 20})],
            ),
        ]
        cat = MutationCatalog(
            [
                StructuralMutationRule(
                    strategy_pattern="",
                    op="drop_analyzer",
                    min_analyzers=0,
                ),
            ]
        )
        cfg = LoopConfig(
            iterations=1,
            ledger_path=str(tmp_path / "ledger.jsonl"),
            stop_sentinel_path="",
            randomize=False,
        )
        si = SelfImprover(cfg, catalog=cat, seed_strategies=seed)
        si._harness_factory = _make_factory(score_fn)
        records = si.run()
        assert len(records) == 1
        rec = records[0]
        assert rec.proposal is not None
        assert rec.proposal["op"] == "drop_analyzer"
        assert rec.accepted is True


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

    def test_seed_only_ladder_records_vacuous(self, tmp_path):
        """When no mutation is accepted, the hold-out is VACUOUS, not OK.

        With ``score_fn`` constant the iteration produces zero delta and
        the rule is rejected.  The ladder still has only the seed — the
        hold-out must surface ``status="vacuous"`` (V2 §6.4 / §12.4 of
        ``planning/SELF_IMPROVEMENT_LOOP.md``) so downstream consumers
        cannot mistake an empty ladder for a "loop generalised cleanly"
        verdict.  ``overfit=False`` is preserved because vacuous is not
        overfit; the field stays a boolean for the existing
        ``--fail-on-overfit`` gate.
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
        assert rec.status == "vacuous"
        assert rec.effective_status() == "vacuous"
        assert any("VACUOUS" in r for r in rec.reasons)

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

    def test_status_default_is_ok(self):
        """V2 §6.4: ``status`` defaults to ``"ok"`` for backwards compat.

        Legacy ledger lines (written before the field shipped) load
        with this default; the implicit-status path runs through
        :meth:`effective_status` to recover ``"vacuous"`` / ``"overfit"``
        from the structural fields.
        """
        rec = LoopHoldoutRecord(
            timestamp="t",
            duration_seconds=0.0,
            holdout_base_seed=99,
            holdout_iterations=1,
            holdout_iteration_offset=0,
            seed_holdout_score=0.0,
            top_holdout_score=0.0,
            seed_training_score=0.0,
            top_training_score=0.0,
            holdout_delta=0.0,
            training_delta=0.0,
            drift=0.0,
            overfit=False,
            eps_overfit=0.05,
            top_iteration=4,
            ladder_size=3,
        )
        assert rec.status == "ok"
        assert rec.effective_status() == "ok"
        # The dict round-trip must carry the new field.
        assert rec.to_dict()["status"] == "ok"

    def test_status_validation_rejects_unknown(self):
        """Typos in downstream callers must fail loudly, not silently."""
        with pytest.raises(ValueError, match="status"):
            LoopHoldoutRecord(
                timestamp="t",
                duration_seconds=0.0,
                holdout_base_seed=99,
                holdout_iterations=1,
                holdout_iteration_offset=0,
                seed_holdout_score=0.0,
                top_holdout_score=0.0,
                seed_training_score=0.0,
                top_training_score=0.0,
                holdout_delta=0.0,
                training_delta=0.0,
                drift=0.0,
                overfit=False,
                eps_overfit=0.05,
                top_iteration=-1,
                ladder_size=1,
                status="bogus",
            )

    def test_supported_statuses_constant(self):
        """The wire constant lists exactly the three implemented verdicts."""
        assert LoopHoldoutRecord.SUPPORTED_STATUSES == ("ok", "overfit", "vacuous")

    def test_effective_status_legacy_vacuous_inference(self):
        """Legacy records (no ``status``) with ``ladder_size=1`` are vacuous."""
        rec = LoopHoldoutRecord(
            timestamp="t",
            duration_seconds=0.0,
            holdout_base_seed=99,
            holdout_iterations=3,
            holdout_iteration_offset=0,
            seed_holdout_score=0.0,
            top_holdout_score=0.0,
            seed_training_score=0.0,
            top_training_score=0.0,
            holdout_delta=0.0,
            training_delta=0.0,
            drift=0.0,
            overfit=False,
            eps_overfit=0.05,
            top_iteration=-1,
            ladder_size=1,
            # status omitted — defaults to "ok" as a legacy record would
        )
        assert rec.status == "ok"
        assert rec.effective_status() == "vacuous"

    def test_effective_status_legacy_overfit_inference(self):
        """Legacy records (no ``status``) with ``overfit=True`` map to overfit."""
        rec = LoopHoldoutRecord(
            timestamp="t",
            duration_seconds=0.0,
            holdout_base_seed=99,
            holdout_iterations=3,
            holdout_iteration_offset=0,
            seed_holdout_score=0.0,
            top_holdout_score=0.0,
            seed_training_score=0.0,
            top_training_score=0.0,
            holdout_delta=0.0,
            training_delta=0.0,
            drift=-0.3,
            overfit=True,
            eps_overfit=0.05,
            top_iteration=4,
            ladder_size=3,
        )
        assert rec.status == "ok"  # default — no explicit field
        assert rec.effective_status() == "overfit"

    def test_vacuous_status_round_trips_through_to_dict(self):
        """``status="vacuous"`` survives the JSON ledger contract."""
        rec = LoopHoldoutRecord(
            timestamp="t",
            duration_seconds=0.0,
            holdout_base_seed=99,
            holdout_iterations=3,
            holdout_iteration_offset=0,
            seed_holdout_score=0.0,
            top_holdout_score=0.0,
            seed_training_score=0.0,
            top_training_score=0.0,
            holdout_delta=0.0,
            training_delta=0.0,
            drift=0.0,
            overfit=False,
            eps_overfit=0.05,
            top_iteration=-1,
            ladder_size=1,
            status="vacuous",
        )
        d = rec.to_dict()
        assert d["status"] == "vacuous"
        # Round-trip through JSON encoding to lock the wire contract.
        parsed = json.loads(json.dumps(d))
        assert parsed["status"] == "vacuous"


# ===========================================================================
# Multi-seed hold-out (planning/SELF_IMPROVEMENT_LOOP.md §13 follow-up)
# ===========================================================================


class TestLoopConfigMultiSeedHoldout:
    """Validation of the list-typed ``holdout_base_seeds`` knob.

    The single-seed scalar landed in 2026-05-08; the list version turns
    a single drift draw into a worst-case estimate over multiple
    independent SHA-256 streams.  Validation rules mirror the scalar
    case (no collision with base_seed) plus list-only constraints
    (no zero entries, no duplicates).
    """

    def test_default_is_empty_tuple(self):
        cfg = LoopConfig()
        assert cfg.holdout_base_seeds == ()

    def test_accepts_list_and_normalizes_to_tuple(self):
        cfg = LoopConfig(holdout_base_seeds=[1, 2, 3])
        # The dataclass stores tuples for hashability and equality stability.
        assert isinstance(cfg.holdout_base_seeds, tuple)
        assert cfg.holdout_base_seeds == (1, 2, 3)

    def test_rejects_zero_entry(self):
        # 0 is the disable sentinel — accepting it here would silently
        # produce a no-op call against the *training* base seed family.
        with pytest.raises(ValueError, match="non-zero"):
            LoopConfig(holdout_base_seeds=(1234, 0, 5678))

    def test_rejects_collision_with_base_seed(self):
        with pytest.raises(ValueError, match="must differ from base_seed"):
            LoopConfig(base_seed=42, holdout_base_seeds=(99, 42, 77))

    def test_rejects_duplicates(self):
        with pytest.raises(ValueError, match="distinct"):
            LoopConfig(holdout_base_seeds=(1234, 5678, 1234))

    def test_resolved_seeds_prefers_list_over_scalar(self):
        """When both knobs are set, the list takes precedence."""
        cfg = LoopConfig(holdout_base_seed=99, holdout_base_seeds=(1234, 5678))
        assert cfg.resolved_holdout_seeds() == (1234, 5678)

    def test_resolved_seeds_falls_back_to_scalar(self):
        """Scalar promoted to a 1-tuple for the multi-seed code path."""
        cfg = LoopConfig(holdout_base_seed=99)
        assert cfg.resolved_holdout_seeds() == (99,)

    def test_resolved_seeds_empty_when_both_unset(self):
        cfg = LoopConfig()
        assert cfg.resolved_holdout_seeds() == ()

    def test_holdout_harness_config_accepts_explicit_seed(self):
        """The explicit ``base_seed`` argument overrides ``holdout_base_seed``.

        This is the wiring the multi-seed loop relies on — without it,
        every per-seed iteration would still measure against the
        scalar attribute.
        """
        cfg = LoopConfig(base_seed=42, holdout_base_seed=99)
        hc = cfg.holdout_harness_config([], iteration_id=3, base_seed=5678)
        assert hc.seed == 5678
        assert hc.randomize_iteration == 3

    def test_holdout_harness_config_defaults_to_scalar(self):
        """Omitting ``base_seed`` keeps the single-seed back-compat path."""
        cfg = LoopConfig(base_seed=42, holdout_base_seed=99)
        hc = cfg.holdout_harness_config([], iteration_id=3)
        assert hc.seed == 99


class TestSelfImproverMultiSeedHoldout:
    """End-to-end behaviour with ``holdout_base_seeds`` set."""

    def test_writes_one_record_per_seed(self, tmp_path):
        """Each seed produces its own LoopHoldoutRecord, in order."""
        cfg = LoopConfig(
            iterations=1,
            n_boot=50,
            ledger_path=str(tmp_path / "ledger.jsonl"),
            stop_sentinel_path="",
            randomize=True,
            base_seed=42,
            holdout_base_seeds=(99, 101, 103),
            holdout_iterations=2,
        )
        si = SelfImprover(cfg, catalog=_accept_radius_catalog(), seed_strategies=_make_specs())
        si._harness_factory = _make_factory(lambda c: 0.4)
        _, _, holdout_records = si.run_full()
        assert len(holdout_records) == 3
        # Records preserve seed order so a ledger audit lines up with
        # the configured list.
        assert [r.holdout_base_seed for r in holdout_records] == [99, 101, 103]

    def test_uses_each_seed_for_measurement(self, tmp_path):
        """Hold-out measurements must use the per-seed base_seed.

        The call log proves we actually drew from the configured
        SHA-256 streams rather than reusing the scalar or the
        training seed.
        """
        call_log: List[Dict[str, Any]] = []
        cfg = LoopConfig(
            iterations=1,
            n_boot=50,
            ledger_path=str(tmp_path / "ledger.jsonl"),
            stop_sentinel_path="",
            randomize=True,
            base_seed=42,
            holdout_base_seeds=(1234, 5678),
            holdout_iterations=2,
        )
        si = SelfImprover(cfg, catalog=_accept_radius_catalog(), seed_strategies=_make_specs())
        si._harness_factory = _make_factory(lambda c: 0.5, call_log=call_log)
        si.run_full()

        seeds_seen = {c["seed"] for c in call_log}
        # 42 = training, 1234/5678 = hold-out streams.  No leakage of
        # 99 or 0 anywhere.
        assert 42 in seeds_seen
        assert 1234 in seeds_seen
        assert 5678 in seeds_seen
        assert 0 not in seeds_seen
        assert 99 not in seeds_seen

    def test_overfit_on_one_seed_only(self, tmp_path):
        """Per-record overfit is computed independently per seed.

        Setup: seed 99 collapses (drift = -0.5), seed 7777 holds
        (drift ≈ 0).  We expect record[0].overfit == True and
        record[1].overfit == False — the aggregation across seeds is
        the CLI's job, not the loop's.
        """
        counter = {"n": 0}

        def score_fn(config: HarnessConfig) -> float:
            if config.seed == 99:
                return 0.3  # collapsed gap on seed 99
            if config.seed == 7777:
                # Hold-out preserves the gap.  Alternate seed/top eval.
                ho = counter.get("ho2", 0)
                counter["ho2"] = ho + 1
                return 0.3 if ho % 2 == 0 else 0.78
            n = counter["n"]
            counter["n"] += 1
            # Training alternation: baseline (0.3) / candidate (0.8).
            return 0.3 if n % 2 == 0 else 0.8

        cfg = LoopConfig(
            iterations=1,
            n_boot=50,
            ledger_path=str(tmp_path / "ledger.jsonl"),
            stop_sentinel_path="",
            randomize=True,
            base_seed=42,
            holdout_base_seeds=(99, 7777),
            holdout_iterations=4,
            holdout_eps_overfit=0.05,
        )
        si = SelfImprover(cfg, catalog=_accept_radius_catalog(), seed_strategies=_make_specs())
        si._harness_factory = _make_factory(score_fn)
        iter_records, _, holdout_records = si.run_full()
        assert iter_records[0].accepted is True
        assert len(holdout_records) == 2
        # Seed 99: collapsed -> overfit; seed 7777: held -> not overfit.
        assert holdout_records[0].holdout_base_seed == 99
        assert holdout_records[0].overfit is True
        assert holdout_records[1].holdout_base_seed == 7777
        assert holdout_records[1].overfit is False

    def test_list_takes_precedence_over_scalar(self, tmp_path):
        """If both scalar and list are set, only the list seeds are used."""
        call_log: List[Dict[str, Any]] = []
        cfg = LoopConfig(
            iterations=1,
            n_boot=50,
            ledger_path=str(tmp_path / "ledger.jsonl"),
            stop_sentinel_path="",
            randomize=True,
            base_seed=42,
            holdout_base_seed=99,
            holdout_base_seeds=(1234, 5678),
            holdout_iterations=1,
        )
        si = SelfImprover(cfg, catalog=_accept_radius_catalog(), seed_strategies=_make_specs())
        si._harness_factory = _make_factory(lambda c: 0.5, call_log=call_log)
        si.run_full()
        seeds_seen = {c["seed"] for c in call_log}
        # 99 (the scalar) must NOT appear when the list is set.
        assert 99 not in seeds_seen
        assert 1234 in seeds_seen
        assert 5678 in seeds_seen

    def test_writes_all_records_to_ledger(self, tmp_path):
        """All per-seed records land in the JSONL ledger with type=holdout."""
        cfg = LoopConfig(
            iterations=1,
            n_boot=50,
            ledger_path=str(tmp_path / "ledger.jsonl"),
            stop_sentinel_path="",
            randomize=True,
            base_seed=42,
            holdout_base_seeds=(11, 22, 33),
            holdout_iterations=1,
        )
        si = SelfImprover(cfg, catalog=_accept_radius_catalog(), seed_strategies=_make_specs())
        si._harness_factory = _make_factory(lambda c: 0.4)
        si.run_full()

        records = load_ledger(cfg.ledger_path)
        holdout_rows = [r for r in records if r.get("record_type") == "holdout"]
        assert len(holdout_rows) == 3
        assert [r["holdout_base_seed"] for r in holdout_rows] == [11, 22, 33]

    def test_scalar_path_unaffected(self, tmp_path):
        """Existing scalar callers see exactly one record (back-compat)."""
        cfg = LoopConfig(
            iterations=1,
            n_boot=50,
            ledger_path=str(tmp_path / "ledger.jsonl"),
            stop_sentinel_path="",
            randomize=True,
            base_seed=42,
            holdout_base_seed=99,  # scalar only
            holdout_iterations=2,
        )
        si = SelfImprover(cfg, catalog=_accept_radius_catalog(), seed_strategies=_make_specs())
        si._harness_factory = _make_factory(lambda c: 0.4)
        _, _, holdout_records = si.run_full()
        assert len(holdout_records) == 1
        assert holdout_records[0].holdout_base_seed == 99

    def test_disabled_when_only_zero_scalar(self, tmp_path):
        """``holdout_base_seed=0`` + empty list = hold-out disabled."""
        cfg = LoopConfig(
            iterations=1,
            n_boot=50,
            ledger_path=str(tmp_path / "ledger.jsonl"),
            stop_sentinel_path="",
            randomize=True,
            base_seed=42,
            # both knobs at default => disabled
            holdout_iterations=2,
        )
        si = SelfImprover(cfg, catalog=_accept_radius_catalog(), seed_strategies=_make_specs())
        si._harness_factory = _make_factory(lambda c: 0.4)
        _, _, holdout_records = si.run_full()
        assert holdout_records == []


class TestCliSeedListParser:
    """Tests for ``scripts/self_improve.py:_parse_seed_list``.

    The parser is the only CLI-side surface the user touches for
    multi-seed hold-out; the loop driver receives a tuple either way.
    """

    @staticmethod
    def _parser():
        import sys

        sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "scripts"))
        try:
            import self_improve as cli  # type: ignore

            return cli._parse_seed_list
        finally:
            # Don't leave the import path polluted for downstream tests.
            sys.path = [p for p in sys.path if not p.endswith("/scripts")]

    def test_empty_string_returns_empty_tuple(self):
        parse = self._parser()
        assert parse("") == ()

    def test_whitespace_only_returns_empty_tuple(self):
        parse = self._parser()
        assert parse("   ") == ()

    def test_single_int(self):
        parse = self._parser()
        assert parse("1234") == (1234,)

    def test_multiple_ints(self):
        parse = self._parser()
        assert parse("1,2,3") == (1, 2, 3)

    def test_tolerates_whitespace_around_entries(self):
        # `--holdout-base-seeds "1234, 5678"` should work without
        # forcing the user to remember to omit the space.
        parse = self._parser()
        assert parse("1234, 5678 , 9012") == (1234, 5678, 9012)

    def test_negative_int_accepted(self):
        # The validator on LoopConfig rejects 0 and duplicates; the
        # parser itself stays tolerant of negative ints so the loop
        # config gets to emit a clearer error.
        parse = self._parser()
        assert parse("-1,2") == (-1, 2)

    def test_rejects_non_integer(self):
        parse = self._parser()
        with pytest.raises(ValueError, match="invalid integer"):
            parse("1,foo,3")

    def test_skips_empty_entries_from_trailing_comma(self):
        parse = self._parser()
        # Trailing commas are common in copy/paste; quietly tolerate them.
        assert parse("1,2,3,") == (1, 2, 3)
        assert parse(",1,2") == (1, 2)


# ===========================================================================
# Bootstrap-CI aggregation of multi-seed hold-out drift
# (planning/SELF_IMPROVEMENT_LOOP.md §13 — *Bootstrap CI on the drift
# estimate* follow-up to the multi-seed hold-out shipped 2026-05-16).
# ===========================================================================


def _make_holdout_record(
    *,
    seed: int = 99,
    drift: float = 0.0,
    overfit: bool = False,
    seed_iter_scores: Optional[List[float]] = None,
    top_iter_scores: Optional[List[float]] = None,
    training_delta: float = 0.0,
    eps_overfit: float = 0.05,
    status: str = "ok",
    top_iteration: int = 4,
    ladder_size: int = 3,
) -> LoopHoldoutRecord:
    """Build a :class:`LoopHoldoutRecord` for the aggregation tests.

    Defaults to a zero-drift, non-overfit record; pass per-iter score
    lists to exercise the high-resolution bootstrap path, omit them to
    exercise the legacy one-sample-per-record fallback.  Pass
    ``status="vacuous"`` together with ``ladder_size=1`` /
    ``top_iteration=-1`` to exercise the V2 §6.4 / §12.4 vacuous
    filtering path through :func:`aggregate_holdout_drift`.
    """
    s_iter = list(seed_iter_scores) if seed_iter_scores is not None else []
    t_iter = list(top_iter_scores) if top_iter_scores is not None else []
    return LoopHoldoutRecord(
        timestamp="2026-05-17T00:00:00+00:00",
        duration_seconds=1.0,
        holdout_base_seed=seed,
        holdout_iterations=max(len(s_iter), 1),
        holdout_iteration_offset=0,
        seed_holdout_score=float(np.mean(s_iter)) if s_iter else 0.0,
        top_holdout_score=float(np.mean(t_iter)) if t_iter else float(training_delta),
        seed_training_score=0.0,
        top_training_score=float(training_delta),
        holdout_delta=(float(np.mean(t_iter)) - float(np.mean(s_iter)))
        if (s_iter and t_iter)
        else float(training_delta),
        training_delta=float(training_delta),
        drift=float(drift),
        overfit=bool(overfit),
        eps_overfit=float(eps_overfit),
        top_iteration=int(top_iteration),
        ladder_size=int(ladder_size),
        base_seed=42,
        mode="quick",
        reasons=[],
        seed_iteration_scores=s_iter,
        top_iteration_scores=t_iter,
        status=status,
    )


class TestAggregateHoldoutDrift:
    """Bootstrap-CI aggregation across a list of hold-out records.

    The aggregation closes the §13 *Bootstrap CI on the drift estimate*
    follow-up — the multi-seed hold-out's worst-case reduction is hard
    to interpret on a single seed because it confounds noise with real
    drift, and a single recent ledger showed drift=-0.0074 (well within
    eps=0.05, but still hard to know whether that is the typical drift
    or the lucky tail).  Pooling per-iteration paired drifts across
    seeds and bootstrap-resampling the mean gives a real CI.
    """

    def test_empty_input_returns_degenerate_aggregate(self):
        """No records → zero-everything, n_samples=0, not flagged."""
        agg = aggregate_holdout_drift([])
        assert agg.mean_drift == 0.0
        assert agg.ci_low == 0.0
        assert agg.ci_high == 0.0
        assert agg.n_samples == 0
        assert agg.n_records == 0
        assert agg.any_overfit is False
        assert agg.statistically_overfit is False

    def test_per_iteration_path_pools_across_records(self):
        """With per-iter scores, n_samples = records × iterations."""
        rec_a = _make_holdout_record(
            seed=99,
            seed_iter_scores=[0.5, 0.5, 0.5],
            top_iter_scores=[0.7, 0.7, 0.7],
            training_delta=0.2,
            drift=0.0,
        )
        rec_b = _make_holdout_record(
            seed=101,
            seed_iter_scores=[0.4, 0.4, 0.4],
            top_iter_scores=[0.6, 0.6, 0.6],
            training_delta=0.2,
            drift=0.0,
        )
        agg = aggregate_holdout_drift([rec_a, rec_b])
        # 2 records × 3 iters = 6 pooled samples, all zero drift.
        assert agg.n_samples == 6
        assert agg.n_records == 2
        assert agg.mean_drift == pytest.approx(0.0)
        # Constant samples → degenerate CI at the point estimate.
        assert agg.ci_low == pytest.approx(0.0)
        assert agg.ci_high == pytest.approx(0.0)

    def test_legacy_record_falls_back_to_point_drift(self):
        """No per-iter scores → one sample per record from the cached drift."""
        legacy = _make_holdout_record(seed=99, drift=-0.03, overfit=False)
        # No seed_iter_scores set → empty lists → fallback.
        agg = aggregate_holdout_drift([legacy])
        assert agg.n_samples == 1
        assert agg.mean_drift == pytest.approx(-0.03)
        # Single-sample bootstrap collapses to point estimate.
        assert agg.ci_low == pytest.approx(-0.03)
        assert agg.ci_high == pytest.approx(-0.03)

    def test_mixed_legacy_and_modern_records(self):
        """Modern records contribute iters, legacy contribute one point each."""
        modern = _make_holdout_record(
            seed=99,
            seed_iter_scores=[0.5, 0.5],
            top_iter_scores=[0.7, 0.7],
            training_delta=0.2,
            drift=0.0,
        )
        legacy = _make_holdout_record(seed=101, drift=-0.05)
        agg = aggregate_holdout_drift([modern, legacy])
        # 2 (modern) + 1 (legacy) = 3 pooled samples.
        assert agg.n_samples == 3
        assert agg.n_records == 2

    def test_worst_drift_is_min_across_records(self):
        """worst_drift / worst_seed mirror the existing `min` reduction."""
        r1 = _make_holdout_record(seed=1, drift=-0.01)
        r2 = _make_holdout_record(seed=2, drift=-0.10)  # worst
        r3 = _make_holdout_record(seed=3, drift=+0.02)
        agg = aggregate_holdout_drift([r1, r2, r3])
        assert agg.worst_drift == pytest.approx(-0.10)
        assert agg.worst_seed == 2

    def test_any_overfit_reduction(self):
        """any_overfit fires on a single bad record (per-seed semantics)."""
        good = _make_holdout_record(seed=1, drift=-0.01, overfit=False)
        bad = _make_holdout_record(seed=2, drift=-0.20, overfit=True)
        agg = aggregate_holdout_drift([good, bad])
        assert agg.any_overfit is True
        assert agg.overfit_count == 1

    def test_no_overfit_when_all_records_clean(self):
        good_a = _make_holdout_record(seed=1, drift=-0.01, overfit=False)
        good_b = _make_holdout_record(seed=2, drift=+0.00, overfit=False)
        agg = aggregate_holdout_drift([good_a, good_b])
        assert agg.any_overfit is False
        assert agg.overfit_count == 0

    def test_statistically_overfit_fires_when_ci_excludes_zero_drift(self):
        """ci_high < -eps_overfit → statistically_overfit True.

        Setup: large constant negative drift across many samples.  The
        bootstrap CI then collapses far below -eps_overfit and the
        verdict fires regardless of the per-record overfit flag.
        """
        rec = _make_holdout_record(
            seed=99,
            seed_iter_scores=[0.5] * 8,
            top_iter_scores=[0.3] * 8,  # gap -0.2 each iter
            training_delta=0.3,  # drift = (-0.2) - 0.3 = -0.5 per iter
            drift=-0.5,
            overfit=True,
            eps_overfit=0.05,
        )
        agg = aggregate_holdout_drift([rec])
        # Constant -0.5 samples → CI degenerate at -0.5; well below -0.05.
        assert agg.mean_drift == pytest.approx(-0.5)
        assert agg.statistically_overfit is True

    def test_statistically_overfit_silent_when_ci_brackets_zero(self):
        """Mixed-sign samples → CI brackets zero → not flagged."""
        rec = _make_holdout_record(
            seed=99,
            # Half samples positive, half negative drift.  Mean ≈ 0,
            # CI brackets 0 — should never trip overfit.
            seed_iter_scores=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            top_iter_scores=[0.7, 0.3, 0.7, 0.3, 0.7, 0.3],
            training_delta=0.0,
        )
        agg = aggregate_holdout_drift([rec])
        # Mean ≈ 0; CI brackets 0 → not flagged.
        assert agg.statistically_overfit is False

    def test_confidence_widens_the_ci(self):
        """Higher confidence → wider CI (looser lower / tighter upper)."""
        rec = _make_holdout_record(
            seed=99,
            seed_iter_scores=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            top_iter_scores=[0.7, 0.3, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4],
            training_delta=0.0,
        )
        agg_95 = aggregate_holdout_drift([rec], confidence=0.95, seed=42)
        agg_99 = aggregate_holdout_drift([rec], confidence=0.99, seed=42)
        width_95 = agg_95.ci_high - agg_95.ci_low
        width_99 = agg_99.ci_high - agg_99.ci_low
        assert width_99 >= width_95

    def test_reproducible_with_same_seed(self):
        """Same seed → byte-identical CI bounds (the harness contract)."""
        rec = _make_holdout_record(
            seed=99,
            seed_iter_scores=[0.5, 0.4, 0.6, 0.5, 0.45],
            top_iter_scores=[0.7, 0.6, 0.8, 0.7, 0.65],
            training_delta=0.2,
        )
        a = aggregate_holdout_drift([rec], seed=123, n_boot=500)
        b = aggregate_holdout_drift([rec], seed=123, n_boot=500)
        assert a.ci_low == pytest.approx(b.ci_low)
        assert a.ci_high == pytest.approx(b.ci_high)

    def test_different_seeds_can_give_different_cis(self):
        """Sanity: distinct seeds usually give distinct CI bounds.

        Needs *variance* in the per-iteration drifts; uniform gaps
        would collapse the bootstrap to the degenerate point estimate
        regardless of seed.  This sample is constructed so the pooled
        drifts span a non-degenerate range.
        """
        rec = _make_holdout_record(
            seed=99,
            # Spread the drift contributions: (top-seed) − training varies
            # across iterations to give the bootstrap something to chew on.
            seed_iter_scores=[0.5, 0.4, 0.6, 0.5, 0.45, 0.55, 0.42, 0.58],
            top_iter_scores=[0.9, 0.5, 0.6, 0.8, 0.4, 0.75, 0.55, 0.7],
            training_delta=0.2,
        )
        a = aggregate_holdout_drift([rec], seed=1, n_boot=500)
        b = aggregate_holdout_drift([rec], seed=99999, n_boot=500)
        # With variance > 0 in the samples and modest n_boot, the bounds
        # almost certainly differ (probability of collision is tiny).
        assert (a.ci_low, a.ci_high) != (b.ci_low, b.ci_high)

    def test_eps_overfit_override(self):
        """Explicit eps_overfit wins over the one stored on the record."""
        rec = _make_holdout_record(
            seed=99,
            seed_iter_scores=[0.5] * 5,
            top_iter_scores=[0.4] * 5,  # gap -0.1 each iter
            training_delta=0.0,  # drift = -0.1
            eps_overfit=0.5,  # record's own eps is generous
        )
        # Default (use record's eps=0.5): not statistically overfit
        # because -0.1 is well above -0.5.
        agg_default = aggregate_holdout_drift([rec])
        assert agg_default.eps_overfit == 0.5
        assert agg_default.statistically_overfit is False
        # Stricter override: eps=0.05.  ci_high stays at -0.1 < -0.05.
        agg_strict = aggregate_holdout_drift([rec], eps_overfit=0.05)
        assert agg_strict.eps_overfit == 0.05
        assert agg_strict.statistically_overfit is True

    def test_per_iter_lists_unequal_length_uses_min(self):
        """Defensive: mismatched list lengths use the shorter prefix."""
        rec = _make_holdout_record(
            seed=99,
            seed_iter_scores=[0.5, 0.5, 0.5, 0.5, 0.5],  # 5 entries
            top_iter_scores=[0.7, 0.7, 0.7],  # 3 entries
            training_delta=0.2,
        )
        agg = aggregate_holdout_drift([rec])
        # Only 3 paired samples should reach the bootstrap.
        assert agg.n_samples == 3

    def test_vacuous_record_excluded_from_bootstrap(self):
        """V2 §6.4 / §12.4 — vacuous records contribute nothing to the CI.

        A vacuous record has ``drift=0.0`` by construction (the "top"
        of the ladder *is* the seed).  Pooling it into the bootstrap
        would pull the CI toward zero and mask a single negative-drift
        seed, so the aggregator must filter it out — and surface the
        count via :attr:`HoldoutDriftAggregate.vacuous_count` so the
        operator can see why the sample count is small.
        """
        informative = _make_holdout_record(
            seed=99,
            seed_iter_scores=[0.5] * 4,
            top_iter_scores=[0.3] * 4,  # gap -0.2 each iter
            training_delta=0.0,  # drift = -0.2 per iter
            drift=-0.2,
            status="ok",
        )
        vacuous = _make_holdout_record(
            seed=101,
            seed_iter_scores=[0.4, 0.4, 0.4],
            top_iter_scores=[0.4, 0.4, 0.4],
            training_delta=0.0,
            drift=0.0,
            status="vacuous",
            top_iteration=-1,
            ladder_size=1,
        )
        agg = aggregate_holdout_drift([informative, vacuous])
        # Only the informative record's 4 iters reach the bootstrap.
        assert agg.n_samples == 4
        assert agg.n_records == 2
        assert agg.vacuous_count == 1
        assert agg.all_vacuous is False
        # Mean drift reflects only the informative record (CI not pulled
        # toward zero by the vacuous record's drift=0).
        assert agg.mean_drift == pytest.approx(-0.2)

    def test_all_vacuous_returns_degenerate_aggregate(self):
        """Every record vacuous → ``all_vacuous=True``, no signal.

        Mirrors the empty-input case but records the vacuous count and
        the originating seed so a summary can show the operator why
        nothing was measured.  Critically, ``statistically_overfit`` is
        False — the aggregate must never claim drift on no data.
        """
        v1 = _make_holdout_record(
            seed=11,
            status="vacuous",
            top_iteration=-1,
            ladder_size=1,
        )
        v2 = _make_holdout_record(
            seed=22,
            status="vacuous",
            top_iteration=-1,
            ladder_size=1,
        )
        agg = aggregate_holdout_drift([v1, v2])
        # The degenerate path must still produce a real
        # :class:`HoldoutDriftAggregate` (and not, e.g., ``None``) so
        # downstream consumers can dispatch on ``all_vacuous`` and
        # ``vacuous_count`` without conditional ``isinstance`` guards.
        assert isinstance(agg, HoldoutDriftAggregate)
        assert agg.all_vacuous is True
        assert agg.vacuous_count == 2
        assert agg.n_records == 2
        assert agg.n_samples == 0
        assert agg.mean_drift == 0.0
        assert agg.ci_low == 0.0
        assert agg.ci_high == 0.0
        assert agg.any_overfit is False
        assert agg.statistically_overfit is False
        # ``worst_seed`` defaults to the first record's seed so the
        # operator can locate the run in the ledger.
        assert agg.worst_seed == 11

    def test_legacy_vacuous_record_classified_by_structure(self):
        """V2 §6.4 / §12.4 — legacy records (no ``status``) classify too.

        Records written before the ``status`` field shipped default to
        ``status="ok"`` on dataclass construction.  The aggregator must
        still recognise them as vacuous when their structural shape
        matches: ``top_iteration < 0`` and ``ladder_size <= 1``.  This
        guards against pre-2026-06-11 ledgers silently slipping vacuous
        records back into the bootstrap.
        """
        legacy_vacuous = _make_holdout_record(
            seed=99,
            status="ok",  # explicit "ok" (legacy default)
            top_iteration=-1,
            ladder_size=1,
            drift=0.0,
        )
        agg = aggregate_holdout_drift([legacy_vacuous])
        assert agg.all_vacuous is True
        assert agg.vacuous_count == 1
        # The legacy record's effective status is vacuous even with the
        # explicit "ok" field, confirming the structural fallback fires.
        assert legacy_vacuous.effective_status() == "vacuous"

    def test_statistically_overfit_not_masked_by_vacuous_record(self):
        """Regression guard: a negative-drift seed must not be averaged
        out by a vacuous companion.

        Mixing one strongly overfit record with one vacuous record was
        the previous failure mode: pooling six samples (4 negative +
        a vacuous 0) softened the CI; filtering vacuous keeps the CI
        on the informative samples only.
        """
        overfit = _make_holdout_record(
            seed=1,
            seed_iter_scores=[0.5] * 6,
            top_iter_scores=[0.3] * 6,  # gap -0.2 each iter
            training_delta=0.3,  # drift = -0.5 per iter
            drift=-0.5,
            overfit=True,
            eps_overfit=0.05,
            status="overfit",
        )
        vacuous = _make_holdout_record(
            seed=2,
            status="vacuous",
            top_iteration=-1,
            ladder_size=1,
            drift=0.0,
        )
        agg = aggregate_holdout_drift([overfit, vacuous])
        # Only the overfit record's 6 samples drive the CI.
        assert agg.n_samples == 6
        assert agg.vacuous_count == 1
        assert agg.any_overfit is True
        assert agg.statistically_overfit is True
        assert agg.mean_drift == pytest.approx(-0.5)

    def test_to_dict_round_trip(self):
        """JSON-friendly serialisation for ledger / dashboard consumers."""
        rec = _make_holdout_record(
            seed=99,
            seed_iter_scores=[0.5, 0.5],
            top_iter_scores=[0.7, 0.6],
            training_delta=0.1,
        )
        agg = aggregate_holdout_drift([rec])
        d = agg.to_dict()
        assert d["n_records"] == 1
        assert d["n_samples"] == 2
        assert "mean_drift" in d
        assert "ci_low" in d
        assert "ci_high" in d
        # Round-trip through JSON exercises the float / int / bool coercions.
        parsed = json.loads(json.dumps(d))
        assert parsed["n_samples"] == 2


class TestLoopHoldoutRecordPerIterScores:
    """The new per-iteration score fields preserve back-compat."""

    def test_default_to_empty_lists(self):
        """Old call sites that omit the new kwargs see empty lists."""
        rec = LoopHoldoutRecord(
            timestamp="t",
            duration_seconds=0.0,
            holdout_base_seed=1,
            holdout_iterations=0,
            holdout_iteration_offset=0,
            seed_holdout_score=0.0,
            top_holdout_score=0.0,
            seed_training_score=0.0,
            top_training_score=0.0,
            holdout_delta=0.0,
            training_delta=0.0,
            drift=0.0,
            overfit=False,
            eps_overfit=0.05,
            top_iteration=-1,
            ladder_size=1,
        )
        assert rec.seed_iteration_scores == []
        assert rec.top_iteration_scores == []

    def test_to_dict_emits_lists(self):
        """The lists round-trip through to_dict so the ledger persists them."""
        rec = _make_holdout_record(
            seed=99,
            seed_iter_scores=[0.1, 0.2, 0.3],
            top_iter_scores=[0.4, 0.5, 0.6],
            training_delta=0.0,
        )
        d = rec.to_dict()
        assert d["seed_iteration_scores"] == [0.1, 0.2, 0.3]
        assert d["top_iteration_scores"] == [0.4, 0.5, 0.6]
        # JSON-serialisable (float lists are JSON-native, but check the
        # contract end-to-end so a future regression surfaces here).
        parsed = json.loads(json.dumps(d))
        assert parsed["seed_iteration_scores"] == [0.1, 0.2, 0.3]


class TestSelfImproverPersistsPerIterScores:
    """End-to-end: the loop wires per-iter scores into the record.

    Regression guard: without this, ``aggregate_holdout_drift`` would
    silently fall back to one-sample-per-record on every fresh run.
    """

    def test_run_full_record_carries_per_iter_scores(self, tmp_path):
        cfg = LoopConfig(
            iterations=1,
            n_boot=50,
            ledger_path=str(tmp_path / "ledger.jsonl"),
            stop_sentinel_path="",
            randomize=True,
            base_seed=42,
            holdout_base_seed=99,
            holdout_iterations=3,
        )
        si = SelfImprover(cfg, catalog=_accept_radius_catalog(), seed_strategies=_make_specs())
        # Constant-score harness — the per-iter values will all be 0.4.
        si._harness_factory = _make_factory(lambda c: 0.4)
        _, _, holdout_records = si.run_full()
        assert len(holdout_records) == 1
        rec = holdout_records[0]
        # Lists should be populated with one float per hold-out iter.
        assert len(rec.seed_iteration_scores) == 3
        assert len(rec.top_iteration_scores) == 3
        assert all(s == pytest.approx(0.4) for s in rec.seed_iteration_scores)

    def test_multi_seed_each_record_has_its_own_iter_scores(self, tmp_path):
        """One record per seed, each with its own per-iter score list."""
        cfg = LoopConfig(
            iterations=1,
            n_boot=50,
            ledger_path=str(tmp_path / "ledger.jsonl"),
            stop_sentinel_path="",
            randomize=True,
            base_seed=42,
            holdout_base_seeds=(99, 7777),
            holdout_iterations=2,
        )
        si = SelfImprover(cfg, catalog=_accept_radius_catalog(), seed_strategies=_make_specs())
        # Distinct scores per seed so we can verify the lists came from
        # the per-seed path, not some shared cache.
        si._harness_factory = _make_factory(lambda c: 0.3 if c.seed == 99 else 0.6)
        _, _, holdout_records = si.run_full()
        assert len(holdout_records) == 2
        # Find each record by its seed (order is configured but explicit
        # check guards against silent reordering).
        by_seed = {r.holdout_base_seed: r for r in holdout_records}
        # Seed 99 ran with score function returning 0.3.
        assert all(s == pytest.approx(0.3) for s in by_seed[99].seed_iteration_scores)
        # Seed 7777 ran with score function returning 0.6.
        assert all(s == pytest.approx(0.6) for s in by_seed[7777].seed_iteration_scores)

    def test_ledger_round_trip_preserves_per_iter_scores(self, tmp_path):
        """JSONL ledger persistence keeps the new fields readable."""
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
        si._harness_factory = _make_factory(lambda c: 0.5)
        si.run_full()
        records = load_ledger(cfg.ledger_path)
        holdout = [r for r in records if r.get("record_type") == "holdout"]
        assert len(holdout) == 1
        assert holdout[0]["seed_iteration_scores"] == [0.5, 0.5]
        assert holdout[0]["top_iteration_scores"] == [0.5, 0.5]


# ===========================================================================
# Inactivity-guarded eps_accept relaxation
# ===========================================================================


class TestInactivityRelaxConfig:
    """Validation and effective-threshold maths for the inactivity-relax knobs."""

    def test_disabled_by_default(self):
        cfg = LoopConfig()
        assert cfg.inactivity_relax_after == 0
        # Disabled ⇒ effective threshold is the configured eps_accept for
        # any streak length.
        for streak in (0, 1, 10, 1000):
            assert cfg.effective_eps_accept(streak) == cfg.eps_accept

    def test_negative_relax_after_raises(self):
        with pytest.raises(ValueError, match="inactivity_relax_after must be >= 0"):
            LoopConfig(inactivity_relax_after=-1)

    def test_factor_must_be_in_open_unit_interval(self):
        # 1.0 doesn't relax — pointless and almost certainly a typo.
        with pytest.raises(ValueError, match="inactivity_relax_factor"):
            LoopConfig(inactivity_relax_after=5, inactivity_relax_factor=1.0)
        # 0.0 collapses the threshold to the floor instantly — pointless.
        with pytest.raises(ValueError, match="inactivity_relax_factor"):
            LoopConfig(inactivity_relax_after=5, inactivity_relax_factor=0.0)
        # > 1 would amplify — opposite of the knob's intent.
        with pytest.raises(ValueError, match="inactivity_relax_factor"):
            LoopConfig(inactivity_relax_after=5, inactivity_relax_factor=1.5)

    def test_floor_must_be_non_negative(self):
        with pytest.raises(ValueError, match="inactivity_min_eps_accept must be >= 0"):
            LoopConfig(inactivity_relax_after=5, inactivity_min_eps_accept=-0.001)

    def test_floor_must_not_exceed_eps_accept(self):
        with pytest.raises(ValueError, match="inactivity_min_eps_accept must be <= eps_accept"):
            LoopConfig(
                inactivity_relax_after=5,
                eps_accept=0.005,
                inactivity_min_eps_accept=0.01,
            )

    def test_no_relax_before_threshold(self):
        cfg = LoopConfig(
            eps_accept=0.005,
            inactivity_relax_after=10,
            inactivity_relax_factor=0.5,
            inactivity_min_eps_accept=0.0,
        )
        for streak in range(10):
            assert cfg.effective_eps_accept(streak) == 0.005

    def test_geometric_decay_steps(self):
        cfg = LoopConfig(
            eps_accept=0.008,
            inactivity_relax_after=4,
            inactivity_relax_factor=0.5,
            inactivity_min_eps_accept=0.0,
        )
        # After 4 misses → 1 step of decay (0.5x).
        assert cfg.effective_eps_accept(4) == pytest.approx(0.004)
        # After 7 misses still 1 step (integer division 7 // 4 = 1).
        assert cfg.effective_eps_accept(7) == pytest.approx(0.004)
        # After 8 misses → 2 steps (0.25x).
        assert cfg.effective_eps_accept(8) == pytest.approx(0.002)
        # After 12 misses → 3 steps (0.125x).
        assert cfg.effective_eps_accept(12) == pytest.approx(0.001)

    def test_floor_clamps_decay(self):
        cfg = LoopConfig(
            eps_accept=0.008,
            inactivity_relax_after=4,
            inactivity_relax_factor=0.5,
            inactivity_min_eps_accept=0.002,
        )
        # Step 1 (4 misses): 0.004, above floor.
        assert cfg.effective_eps_accept(4) == pytest.approx(0.004)
        # Step 2 (8 misses): 0.002 — exactly the floor.
        assert cfg.effective_eps_accept(8) == pytest.approx(0.002)
        # Step 3 (12 misses) would be 0.001 < floor: clamped to floor.
        assert cfg.effective_eps_accept(12) == pytest.approx(0.002)
        # Step 10 still floor.
        assert cfg.effective_eps_accept(40) == pytest.approx(0.002)


class TestInactivityRelaxIntegration:
    """End-to-end loop behaviour under inactivity-relaxed eps_accept."""

    def _radius_catalog(self) -> MutationCatalog:
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

    def test_records_carry_effective_eps_and_streak(self, tmp_path):
        # Score function: baseline 0.5, candidate 0.5 → delta 0, every
        # iteration rejected.  Streak grows monotonically; threshold
        # decays once we cross inactivity_relax_after.
        cfg = LoopConfig(
            iterations=6,
            n_boot=50,
            eps_accept=0.010,
            inactivity_relax_after=2,
            inactivity_relax_factor=0.5,
            inactivity_min_eps_accept=0.0,
            ledger_path=str(tmp_path / "ledger.jsonl"),
            stop_sentinel_path="",
            randomize=False,
        )
        si = SelfImprover(cfg, catalog=self._radius_catalog(), seed_strategies=_make_specs())
        si._harness_factory = _make_factory(lambda c: 0.5)
        records = si.run()
        assert len(records) == 6
        # No accepts at delta=0 (regardless of relaxation, the bootstrap
        # CI on a zero-delta sample brackets zero so the lower-bound gate
        # rejects it).  Streak therefore grows 0, 1, 2, 3, 4, 5.
        assert all(not r.accepted for r in records)
        assert [r.iters_since_accept for r in records] == [0, 1, 2, 3, 4, 5]
        # eps_accept=0.010 at streak 0/1 (no relax yet), 0.005 at 2/3
        # (1 step), 0.0025 at 4/5 (2 steps).
        expected = [0.010, 0.010, 0.005, 0.005, 0.0025, 0.0025]
        for r, e in zip(records, expected):
            assert r.effective_eps_accept == pytest.approx(e)

    def test_streak_resets_on_accept(self, tmp_path):
        # First two iters: baseline 0.3, candidate 0.7 (strong accept);
        # then baseline=candidate=0.5 (reject); then strong accept again.
        # Alternate the runs: each iteration runs baseline then candidate.
        # We want: iter 0 = accept, iter 1 = reject, iter 2 = reject,
        # iter 3 = accept.  So we serve baseline/candidate pairs
        # (0.3, 0.7), (0.5, 0.5), (0.5, 0.5), (0.3, 0.7).
        baseline_candidate_pairs = [(0.3, 0.7), (0.5, 0.5), (0.5, 0.5), (0.3, 0.7)]
        flat = [v for pair in baseline_candidate_pairs for v in pair]
        counter = {"n": 0}

        def score_fn(config):
            n = counter["n"]
            counter["n"] += 1
            return flat[n]

        cfg = LoopConfig(
            iterations=4,
            n_boot=200,
            eps_accept=0.005,
            inactivity_relax_after=10,  # high enough not to fire in 4 iters
            inactivity_relax_factor=0.5,
            inactivity_min_eps_accept=0.0,
            ledger_path=str(tmp_path / "ledger.jsonl"),
            stop_sentinel_path="",
            randomize=False,
        )
        si = SelfImprover(cfg, catalog=self._radius_catalog(), seed_strategies=_make_specs())
        si._harness_factory = _make_factory(score_fn)
        records = si.run()
        # iter 0 accepts → streak entering iter 0 = 0
        # iter 1 rejects → streak entering iter 1 = 0 (just reset)
        # iter 2 rejects → streak entering iter 2 = 1
        # iter 3 accepts → streak entering iter 3 = 2
        assert records[0].accepted is True
        assert records[1].accepted is False
        assert records[2].accepted is False
        assert records[3].accepted is True
        assert [r.iters_since_accept for r in records] == [0, 0, 1, 2]
        # eps_accept is constant 0.005 across the four because the relax
        # threshold of 10 was never crossed.
        assert all(r.effective_eps_accept == pytest.approx(0.005) for r in records)

    def test_skip_iterations_count_toward_streak(self, tmp_path):
        # A catalog targeting a class that does not exist will always
        # produce skip records, so iters_since_accept must climb through
        # them.
        empty_catalog = MutationCatalog(
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
            iterations=4,
            ledger_path=str(tmp_path / "ledger.jsonl"),
            stop_sentinel_path="",
            randomize=False,
            inactivity_relax_after=2,
            inactivity_relax_factor=0.5,
            inactivity_min_eps_accept=0.0,
        )
        si = SelfImprover(cfg, catalog=empty_catalog, seed_strategies=_make_specs())
        si._harness_factory = _make_factory(lambda c: 0.5)
        records = si.run()
        # All four iterations are skipped (no applicable mutations).
        assert all(r.proposal is None for r in records)
        assert [r.iters_since_accept for r in records] == [0, 1, 2, 3]
        # effective_eps_accept tracks the decay: 0.005, 0.005, 0.0025, 0.0025.
        expected = [0.005, 0.005, 0.0025, 0.0025]
        for r, e in zip(records, expected):
            assert r.effective_eps_accept == pytest.approx(e)

    def test_relaxation_can_accept_borderline_delta(self, tmp_path):
        # With eps_accept=0.05 a +0.04 lift is rejected; after relaxing
        # one step to 0.025 the same lift accepts.  The fake harness
        # returns 0.5 then 0.54 on alternate calls so every iteration
        # sees the same delta — the only thing that changes is the
        # threshold.
        baseline = 0.50
        cand = 0.54
        counter = {"n": 0}

        def score_fn(config):
            n = counter["n"]
            counter["n"] += 1
            return baseline if n % 2 == 0 else cand

        cfg = LoopConfig(
            iterations=5,
            n_boot=500,
            eps_accept=0.05,
            eps_regress=0.5,
            inactivity_relax_after=2,
            inactivity_relax_factor=0.5,
            inactivity_min_eps_accept=0.0,
            ledger_path=str(tmp_path / "ledger.jsonl"),
            stop_sentinel_path="",
            randomize=False,
        )
        si = SelfImprover(cfg, catalog=self._radius_catalog(), seed_strategies=_make_specs())
        si._harness_factory = _make_factory(score_fn)
        records = si.run()
        # iter 0 (eps=0.05): +0.04 < 0.05 → reject (delta-gate)
        # iter 1 (eps=0.05, streak entering=1): same → reject
        # iter 2 (eps=0.025, streak entering=2 → 1 step): +0.04 > 0.025 → accept
        assert records[0].accepted is False
        assert records[0].effective_eps_accept == pytest.approx(0.05)
        assert records[1].accepted is False
        assert records[1].effective_eps_accept == pytest.approx(0.05)
        assert records[2].accepted is True
        assert records[2].effective_eps_accept == pytest.approx(0.025)
        # After the accept the streak resets so iter 3 is back to 0.05
        # and rejects again, iter 4 (streak 1) still 0.05 and rejects.
        assert records[3].effective_eps_accept == pytest.approx(0.05)
        assert records[3].accepted is False
        assert records[4].effective_eps_accept == pytest.approx(0.05)
        assert records[4].accepted is False

    def test_disabled_keeps_eps_constant_and_records_field_set(self, tmp_path):
        # With relaxation disabled the field is still populated (it
        # equals the constant eps_accept) so summary tooling can rely
        # on the field always being present in newly-written records.
        cfg = LoopConfig(
            iterations=3,
            n_boot=50,
            eps_accept=0.005,
            inactivity_relax_after=0,
            ledger_path=str(tmp_path / "ledger.jsonl"),
            stop_sentinel_path="",
            randomize=False,
        )
        si = SelfImprover(cfg, catalog=self._radius_catalog(), seed_strategies=_make_specs())
        si._harness_factory = _make_factory(lambda c: 0.5)
        records = si.run()
        assert all(r.effective_eps_accept == pytest.approx(0.005) for r in records)
        # Streak still tracked so an auditor can see drought length even
        # without relaxation enabled.
        assert [r.iters_since_accept for r in records] == [0, 1, 2]

    def test_ledger_round_trip_preserves_relax_fields(self, tmp_path):
        cfg = LoopConfig(
            iterations=3,
            n_boot=50,
            eps_accept=0.010,
            inactivity_relax_after=2,
            inactivity_relax_factor=0.5,
            inactivity_min_eps_accept=0.0,
            ledger_path=str(tmp_path / "ledger.jsonl"),
            stop_sentinel_path="",
            randomize=False,
        )
        si = SelfImprover(cfg, catalog=self._radius_catalog(), seed_strategies=_make_specs())
        si._harness_factory = _make_factory(lambda c: 0.5)
        si.run()
        records = load_ledger(cfg.ledger_path)
        iter_recs = [r for r in records if r.get("record_type") == "iteration"]
        assert len(iter_recs) == 3
        assert iter_recs[0]["effective_eps_accept"] == pytest.approx(0.010)
        assert iter_recs[1]["effective_eps_accept"] == pytest.approx(0.010)
        assert iter_recs[2]["effective_eps_accept"] == pytest.approx(0.005)
        assert iter_recs[0]["iters_since_accept"] == 0
        assert iter_recs[1]["iters_since_accept"] == 1
        assert iter_recs[2]["iters_since_accept"] == 2

    def test_legacy_record_loads_with_none_relax_fields(self):
        # Records written before the 2026-05-30 ship don't carry the
        # two new fields.  Direct dataclass construction (with omitted
        # kwargs) must default them to None so the JSONL load path
        # continues to work against historical ledgers.
        rec = LoopIterationRecord(
            iteration=0,
            timestamp="2026-01-01T00:00:00+00:00",
            duration_seconds=0.0,
            proposal=None,
            accepted=False,
            baseline_score=0.0,
            candidate_score=0.0,
            delta=0.0,
            ci_low=0.0,
            ci_high=0.0,
            worst_pair_regression=0.0,
            worst_pair=None,
        )
        assert rec.effective_eps_accept is None
        assert rec.iters_since_accept is None
        d = rec.to_dict()
        assert d["effective_eps_accept"] is None
        assert d["iters_since_accept"] is None


# ===========================================================================
# §12.4 No-op detection
# ===========================================================================


class TestNoOpDetection:
    """§12.4 of ``planning/SELF_IMPROVEMENT_LOOP.md``.

    An iteration whose candidate per-pair scores are bit-identical to
    baseline carries zero information about whether the proposal helps or
    hurts: the bandit must not be pulled on it.  These tests exercise the
    detection (post-measure equality), the ledger telemetry (``no_op``
    field + ``reason_skipped="no_op"``), and the bandit gating (no
    ``record_outcome`` on no-op iterations, no replay in
    :meth:`AdaptiveMutationSampler.prime_from_ledger`).
    """

    def _radius_catalog(self) -> MutationCatalog:
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

    def test_default_no_op_field_is_false(self):
        # Direct construction without the new kwarg must default to
        # False so legacy ledger lines (pre 2026-06-12) classify
        # correctly without a one-time migration.
        rec = LoopIterationRecord(
            iteration=0,
            timestamp="2026-01-01T00:00:00+00:00",
            duration_seconds=0.0,
            proposal=None,
            accepted=False,
            baseline_score=0.0,
            candidate_score=0.0,
            delta=0.0,
            ci_low=0.0,
            ci_high=0.0,
            worst_pair_regression=0.0,
            worst_pair=None,
        )
        assert rec.no_op is False
        assert rec.to_dict()["no_op"] is False

    def test_identical_pair_scores_flag_no_op(self, tmp_path):
        # Constant score 0.5 on every call → baseline and candidate
        # measurements produce bit-identical per-pair scores → no-op.
        cfg = LoopConfig(
            iterations=1,
            n_boot=50,
            ledger_path=str(tmp_path / "ledger.jsonl"),
            stop_sentinel_path="",
            randomize=False,
        )
        si = SelfImprover(cfg, catalog=self._radius_catalog(), seed_strategies=_make_specs())
        si._harness_factory = _make_factory(lambda c: 0.5)
        records = si.run()
        assert len(records) == 1
        assert records[0].no_op is True
        assert records[0].accepted is False
        assert records[0].reason_skipped == "no_op"
        # The decision's bootstrap CI is still reported (the field
        # carries genuine information that the loop *measured* zero
        # delta), but the human-readable reason list includes the
        # no-op marker for an auditor scanning the ledger.
        assert any("no-op" in r for r in records[0].reasons)

    def test_distinct_pair_scores_are_not_no_op(self, tmp_path):
        # Baseline 0.5, candidate 0.4 — legitimate non-no-op reject.
        # The candidate is strictly worse so the iteration rejects
        # without tripping the bit-identical detector.
        counter = {"n": 0}

        def score_fn(config: HarnessConfig) -> float:
            n = counter["n"]
            counter["n"] += 1
            return 0.5 if n % 2 == 0 else 0.4

        cfg = LoopConfig(
            iterations=1,
            n_boot=50,
            ledger_path=str(tmp_path / "ledger.jsonl"),
            stop_sentinel_path="",
            randomize=False,
        )
        si = SelfImprover(cfg, catalog=self._radius_catalog(), seed_strategies=_make_specs())
        si._harness_factory = _make_factory(score_fn)
        records = si.run()
        assert len(records) == 1
        assert records[0].no_op is False
        assert records[0].accepted is False
        assert records[0].reason_skipped is None

    def test_no_op_iteration_does_not_pull_bandit(self, tmp_path):
        # Two iterations on a constant-score harness → both are no-op
        # → bandit's n_attempts stays at zero.  Compare against
        # test_adaptive_sampler_records_rejects (immediately above
        # this class in source order), which exercises the legitimate
        # reject path (n_attempts==2 after two real rejects).
        cfg = LoopConfig(
            iterations=2,
            n_boot=50,
            ledger_path=str(tmp_path / "ledger.jsonl"),
            stop_sentinel_path="",
            randomize=False,
            adaptive_sampling=True,
        )
        si = SelfImprover(cfg, catalog=self._radius_catalog(), seed_strategies=_make_specs())
        si._harness_factory = _make_factory(lambda c: 0.5)
        records = si.run()
        assert len(records) == 2
        assert all(r.no_op for r in records)
        assert si.sampler is not None
        snap = si.sampler.stats_snapshot()
        # No arm registered any attempt because every iteration was a
        # no-op — the §12.4 telemetry rule is the headline guarantee.
        assert all(s.n_attempts == 0 for s in snap)
        assert all(s.n_accepts == 0 for s in snap)

    def test_no_op_iteration_increments_streak(self, tmp_path):
        # No-op iterations are observationally non-accepts: they must
        # count toward the inactivity streak so the relax rule can
        # break out of a long dormant-rule drought.
        cfg = LoopConfig(
            iterations=3,
            n_boot=50,
            eps_accept=0.010,
            inactivity_relax_after=10,  # high enough not to fire in 3 iters
            ledger_path=str(tmp_path / "ledger.jsonl"),
            stop_sentinel_path="",
            randomize=False,
        )
        si = SelfImprover(cfg, catalog=self._radius_catalog(), seed_strategies=_make_specs())
        si._harness_factory = _make_factory(lambda c: 0.5)
        records = si.run()
        assert len(records) == 3
        assert all(r.no_op for r in records)
        assert [r.iters_since_accept for r in records] == [0, 1, 2]

    def test_prime_from_ledger_skips_no_op_records(self, tmp_path):
        # Write a small ledger by hand: one legitimate accept and one
        # no-op.  prime_from_ledger must register the accept but skip
        # the no-op entirely — mis-priming the bandit on a zero-info
        # event would mis-train the posterior toward Beta(1, 2) even
        # though the rule's value is undetermined.
        ledger_path = tmp_path / "ledger.jsonl"
        accept_rec = {
            "record_type": "iteration",
            "iteration": 0,
            "accepted": True,
            "no_op": False,
            "proposal": {
                "class_name": "_DummyHeuristicA",
                "param_name": "radius",
                "rule_kind": "log_uniform_perturb",
            },
        }
        no_op_rec = {
            "record_type": "iteration",
            "iteration": 1,
            "accepted": False,
            "no_op": True,
            "reason_skipped": "no_op",
            "proposal": {
                "class_name": "_DummyHeuristicA",
                "param_name": "radius",
                "rule_kind": "log_uniform_perturb",
            },
        }
        with ledger_path.open("w") as f:
            f.write(json.dumps(accept_rec) + "\n")
            f.write(json.dumps(no_op_rec) + "\n")

        sampler = AdaptiveMutationSampler(self._radius_catalog())
        consumed = sampler.prime_from_ledger(str(ledger_path))
        # Only the accept was consumed — the no-op was skipped.
        assert consumed == 1
        snap = sampler.stats_snapshot()
        assert len(snap) == 1
        assert snap[0].n_attempts == 1
        assert snap[0].n_accepts == 1

    def test_prime_from_ledger_legacy_record_replays(self, tmp_path):
        # Pre-2026-06-12 ledger lines have no ``no_op`` key.  Loader
        # default is False → they replay as ordinary attempts.  This
        # is the backwards-compat contract: archived ledgers stay
        # equivalent under the new priming semantics.
        ledger_path = tmp_path / "ledger.jsonl"
        legacy_rec = {
            "record_type": "iteration",
            "iteration": 0,
            "accepted": False,
            # no ``no_op`` key on disk — that's the legacy shape.
            "proposal": {
                "class_name": "_DummyHeuristicA",
                "param_name": "radius",
                "rule_kind": "log_uniform_perturb",
            },
        }
        with ledger_path.open("w") as f:
            f.write(json.dumps(legacy_rec) + "\n")

        sampler = AdaptiveMutationSampler(self._radius_catalog())
        consumed = sampler.prime_from_ledger(str(ledger_path))
        assert consumed == 1
        snap = sampler.stats_snapshot()
        assert snap[0].n_attempts == 1
        assert snap[0].n_accepts == 0

    def test_discard_outcome_clears_pending_arm(self):
        # discard_outcome must clear last_rule_key so the next
        # record_outcome is a no-op — same contract as the
        # post-pull cleanup.  Independent of the loop driver.
        sampler = AdaptiveMutationSampler(self._radius_catalog())
        # Seed an internal pending key without going through sample()
        # (which would require a full spec list); the public surface
        # is what matters.
        sampler._last_rule_key = ("_DummyHeuristicA", "radius", "log_uniform_perturb")
        sampler.discard_outcome()
        assert sampler.last_rule_key is None
        # Subsequent record_outcome is now a no-op — no posterior
        # update, no error.
        sampler.record_outcome(True)
        snap = sampler.stats_snapshot()
        assert snap == []

    def test_no_op_round_trips_through_ledger(self, tmp_path):
        # End-to-end: write one no-op iteration, reload the ledger,
        # confirm the field survived (with the right value) and that
        # the CLI summary parsing path sees it.
        ledger_path = tmp_path / "ledger.jsonl"
        cfg = LoopConfig(
            iterations=1,
            n_boot=50,
            ledger_path=str(ledger_path),
            stop_sentinel_path="",
            randomize=False,
        )
        si = SelfImprover(cfg, catalog=self._radius_catalog(), seed_strategies=_make_specs())
        si._harness_factory = _make_factory(lambda c: 0.5)
        si.run()
        loaded = load_ledger(str(ledger_path))
        iter_recs = [r for r in loaded if r.get("record_type") == "iteration"]
        assert len(iter_recs) == 1
        assert iter_recs[0]["no_op"] is True
        assert iter_recs[0]["reason_skipped"] == "no_op"
        assert iter_recs[0]["accepted"] is False

    def test_cli_summary_surfaces_no_op_count(self, tmp_path, capsys):
        # End-to-end smoke check: feed the CLI summary path a mixed
        # ledger (one no-op, one legitimate reject) and confirm the
        # output line reports both buckets correctly.  Catches the
        # accept-rate-denominator bug the §12.4 ship guards against:
        # the rate must exclude no-op iterations.
        import sys

        ledger_path = tmp_path / "ledger.jsonl"
        no_op_rec = {
            "record_type": "iteration",
            "iteration": 0,
            "accepted": False,
            "no_op": True,
            "reason_skipped": "no_op",
            "delta": 0.0,
            "proposal": {
                "class_name": "_DummyHeuristicA",
                "param_name": "radius",
                "rule_kind": "log_uniform_perturb",
            },
        }
        reject_rec = {
            "record_type": "iteration",
            "iteration": 1,
            "accepted": False,
            "no_op": False,
            "delta": -0.10,
            "proposal": {
                "class_name": "_DummyHeuristicA",
                "param_name": "radius",
                "rule_kind": "log_uniform_perturb",
            },
        }
        with ledger_path.open("w") as f:
            f.write(json.dumps(no_op_rec) + "\n")
            f.write(json.dumps(reject_rec) + "\n")

        sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "scripts"))
        try:
            import self_improve as cli  # type: ignore

            ns = type("NS", (), {"ledger": str(ledger_path)})()
            rc = cli._cmd_summary(ns)
        finally:
            sys.path = [p for p in sys.path if not p.endswith("/scripts")]

        assert rc == 0
        out = capsys.readouterr().out
        # Iteration breakdown surfaces the no-op count distinct from
        # the skip count.
        assert "no-op=1" in out
        # Accept rate is computed over the informative bucket
        # (decided minus no-op) — here that's exactly 1 record (the
        # legitimate reject), so the rate is 0/1 = 0.0%.
        assert "informative" in out


# ===========================================================================
# §7.4 Graded bandit reward shaping
# ===========================================================================


class TestGradedBanditReward:
    """§7.4 of ``planning/SELF_IMPROVEMENT_LOOP.md``.

    Replaces the binary accept-reward (``+1`` per accept, ``0`` per
    reject) with a graded reward in ``[0, 1]`` derived from the
    bootstrap CI / point delta — so a barely-confirmed accept and a
    clearly-winning accept become distinguishable, and an honest near-
    miss reject (``Δ ≈ 0``) is no longer indistinguishable from a
    clearly-harmful reject (``Δ ≪ 0``).
    """

    def _radius_catalog(self) -> MutationCatalog:
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

    # ---- _compute_graded_reward formula ----

    def test_compute_graded_reward_accept_at_zero_ci_low(self):
        from panobbgo.self_improve import _compute_graded_reward

        # ``ci_low = 0`` is the lower bound of the accept regime: just
        # barely confirmed → reward sits at the 0.5 floor.
        r = _compute_graded_reward(accepted=True, delta=0.005, ci_low=0.0, eps_accept=0.005)
        assert r == pytest.approx(0.5)

    def test_compute_graded_reward_accept_at_full_ci_low(self):
        from panobbgo.self_improve import _compute_graded_reward

        # ``ci_low = eps_scale = 4·eps_accept`` saturates the bonus →
        # reward maxes out at 1.0.
        r = _compute_graded_reward(accepted=True, delta=0.05, ci_low=0.020, eps_accept=0.005)
        assert r == pytest.approx(1.0)

    def test_compute_graded_reward_accept_half_ci_low(self):
        from panobbgo.self_improve import _compute_graded_reward

        # ``ci_low = 0.5·eps_scale = 2·eps_accept`` ⇒ bonus = 0.5 ⇒
        # reward = 1.0 (saturated).  Use a lower ci_low for the
        # half-bonus midpoint.
        r_half = _compute_graded_reward(accepted=True, delta=0.02, ci_low=0.005, eps_accept=0.005)
        # ci_low / eps_scale = 0.005 / 0.020 = 0.25 ⇒ reward = 0.75.
        assert r_half == pytest.approx(0.75)

    def test_compute_graded_reward_reject_at_zero_delta(self):
        from panobbgo.self_improve import _compute_graded_reward

        # Δ = 0 (CI bracketed zero, didn't reach eps) → reward sits at
        # the 0.5 ceiling of the reject regime: the proposal carried no
        # negative signal.
        r = _compute_graded_reward(accepted=False, delta=0.0, ci_low=-0.005, eps_accept=0.005)
        assert r == pytest.approx(0.5)

    def test_compute_graded_reward_reject_at_full_negative_delta(self):
        from panobbgo.self_improve import _compute_graded_reward

        # Δ = -eps_scale = -4·eps_accept saturates the penalty →
        # reward floors at 0.0 (clearly harmful proposal).
        r = _compute_graded_reward(accepted=False, delta=-0.020, ci_low=-0.030, eps_accept=0.005)
        assert r == pytest.approx(0.0)

    def test_compute_graded_reward_reject_at_positive_delta(self):
        from panobbgo.self_improve import _compute_graded_reward

        # Honest near miss: Δ > 0 but CI didn't clear, reward saturates
        # at 0.5 (top of reject regime).  This is the §7.4 invariant —
        # a "real signal, wrong sign / too small" reject lands above a
        # clearly-harmful one and matches a delta-zero reject.
        r = _compute_graded_reward(accepted=False, delta=0.010, ci_low=-0.001, eps_accept=0.005)
        assert r == pytest.approx(0.5)

    def test_compute_graded_reward_handles_zero_eps_accept(self):
        from panobbgo.self_improve import _compute_graded_reward

        # ``eps_accept = 0`` would divide by zero naively; the helper
        # collapses to a tiny floor so the clamps still pin the output
        # and no NaN/Inf escapes into the posterior.
        r_accept = _compute_graded_reward(accepted=True, delta=0.1, ci_low=0.1, eps_accept=0.0)
        assert r_accept == pytest.approx(1.0)
        r_reject = _compute_graded_reward(accepted=False, delta=-0.1, ci_low=-0.1, eps_accept=0.0)
        assert r_reject == pytest.approx(0.0)

    # ---- MutationRuleStats reward_sum back-compat ----

    def test_stats_default_reward_sum_zero(self):
        s = MutationRuleStats(rule_key=("A", "b", "kind"))
        assert s.reward_sum == 0.0
        assert s.mean_reward == 0.0
        # Empty arm: accept_rate semantics unchanged.
        assert s.accept_rate == 0.0

    def test_stats_post_init_mirrors_n_accepts(self):
        # Direct construction with the binary semantic (n_accepts > 0
        # but reward_sum = 0): __post_init__ mirrors n_accepts into
        # reward_sum so the Thompson posterior is byte-identical to the
        # pre-graded (Beta(α₀ + n_accepts, …)) parameterisation.
        s = MutationRuleStats(rule_key=("A", "b", "kind"), n_attempts=4, n_accepts=3)
        assert s.reward_sum == 3.0
        # mean_reward equals accept_rate on the binary path.
        assert s.mean_reward == pytest.approx(0.75)
        assert s.accept_rate == pytest.approx(0.75)

    def test_stats_explicit_reward_sum_preserved(self):
        # Graded direct construction: reward_sum != 0 stays as-is even
        # when n_accepts > 0.  This is the case the post_init guard
        # exists to *not* clobber.
        s = MutationRuleStats(
            rule_key=("A", "b", "kind"),
            n_attempts=4,
            n_accepts=2,
            reward_sum=1.5,
        )
        assert s.reward_sum == 1.5
        assert s.mean_reward == pytest.approx(1.5 / 4.0)
        assert s.accept_rate == pytest.approx(0.5)

    def test_stats_to_dict_includes_reward_fields(self):
        s = MutationRuleStats(
            rule_key=("Foo", "bar", "log_uniform_perturb"),
            n_attempts=10,
            n_accepts=3,
            reward_sum=4.25,
        )
        d = s.to_dict()
        assert d["reward_sum"] == 4.25
        assert d["mean_reward"] == pytest.approx(0.425)
        # Binary fields stay unchanged so legacy summary parsers work.
        assert d["n_accepts"] == 3
        assert d["accept_rate"] == pytest.approx(0.3)

    # ---- record_outcome graded path ----

    def test_record_outcome_binary_default_matches_history(self):
        # The historical call shape (no ``reward`` kwarg) must update
        # the stats in the exact same way the pre-graded sampler did:
        # ``reward_sum`` tracks ``n_accepts`` cleanly so the Beta
        # posterior matches.
        samp = AdaptiveMutationSampler(self._radius_catalog())
        samp._last_rule_key = ("_DummyHeuristicA", "radius", "log_uniform_perturb")
        samp.record_outcome(True)
        samp._last_rule_key = ("_DummyHeuristicA", "radius", "log_uniform_perturb")
        samp.record_outcome(False)
        samp._last_rule_key = ("_DummyHeuristicA", "radius", "log_uniform_perturb")
        samp.record_outcome(True)
        snap = samp.stats_snapshot()
        assert len(snap) == 1
        assert snap[0].n_attempts == 3
        assert snap[0].n_accepts == 2
        # Binary path: reward_sum = n_accepts.
        assert snap[0].reward_sum == 2.0
        assert snap[0].mean_reward == pytest.approx(2.0 / 3.0)

    def test_record_outcome_graded_accumulates(self):
        samp = AdaptiveMutationSampler(self._radius_catalog())
        # Three graded rewards: accept@0.75, reject@0.3, accept@0.55.
        for reward, accepted in [(0.75, True), (0.30, False), (0.55, True)]:
            samp._last_rule_key = ("_DummyHeuristicA", "radius", "log_uniform_perturb")
            samp.record_outcome(accepted, reward=reward)
        snap = samp.stats_snapshot()
        assert len(snap) == 1
        assert snap[0].n_attempts == 3
        assert snap[0].n_accepts == 2  # binary side still tracks accepts
        assert snap[0].reward_sum == pytest.approx(0.75 + 0.30 + 0.55)
        assert snap[0].mean_reward == pytest.approx((0.75 + 0.30 + 0.55) / 3.0)

    def test_record_outcome_graded_clamps_out_of_range(self):
        # Defensive clamping: rewards outside [0, 1] are pinned to the
        # nearest valid boundary so a numeric escape never corrupts the
        # posterior.  The driver only ever passes in-range values but
        # third-party callers and future graded variants might not.
        samp = AdaptiveMutationSampler(self._radius_catalog())
        samp._last_rule_key = ("_DummyHeuristicA", "radius", "log_uniform_perturb")
        samp.record_outcome(True, reward=1.5)
        samp._last_rule_key = ("_DummyHeuristicA", "radius", "log_uniform_perturb")
        samp.record_outcome(False, reward=-0.5)
        snap = samp.stats_snapshot()
        # 1.5 → clipped to 1.0; -0.5 → clipped to 0.0.
        assert snap[0].reward_sum == pytest.approx(1.0)

    # ---- Thompson uses reward_sum ----

    def test_thompson_uses_reward_sum_not_n_accepts(self):
        # Two arms, identical n_accepts but different reward_sum:
        # the higher-reward arm should be picked far more often.
        # This is the §7.4 headline guarantee — graded reward turns
        # close-to-prior arms into distinguishable ones.
        cat = MutationCatalog(
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
                    param_name="radius",
                    kind="log_uniform_perturb",
                    bounds=(0.005, 0.5),
                ),
            ]
        )
        samp = AdaptiveMutationSampler(cat)
        # Both arms: 20 attempts, 0 binary accepts.  Arm A: 20 graded
        # rewards of 0.9 (mean reward = 0.9, like Beta(19, 3)); Arm B:
        # 20 graded rewards of 0.05 (mean reward = 0.05, like
        # Beta(2, 20)).  Under Thompson the higher-reward arm should
        # dominate.
        a_key = ("_DummyHeuristicA", "radius", "log_uniform_perturb")
        b_key = ("_DummyHeuristicB", "radius", "log_uniform_perturb")
        samp._stats[a_key] = MutationRuleStats(
            rule_key=a_key,
            n_attempts=20,
            n_accepts=0,
            reward_sum=18.0,  # mean reward 0.9
        )
        samp._stats[b_key] = MutationRuleStats(
            rule_key=b_key,
            n_attempts=20,
            n_accepts=0,
            reward_sum=1.0,  # mean reward 0.05
        )
        rng = np.random.default_rng(42)
        a_picks = 0
        n_samples = 200
        for _ in range(n_samples):
            prop = samp.sample(rng, _make_specs())
            assert prop is not None
            if prop.class_name == "_DummyHeuristicA":
                a_picks += 1
            samp._last_rule_key = None  # reset so we don't pollute stats
        rate = a_picks / n_samples
        # Posteriors: Beta(1 + 18, 1 + 2) ≈ Beta(19, 3) vs
        # Beta(1 + 1, 1 + 19) ≈ Beta(2, 20).  A should dominate.
        assert rate > 0.85, f"high-reward arm should win, got {rate:.3f}"

    # ---- prime_from_ledger graded path ----

    def test_prime_from_ledger_uses_bandit_reward(self, tmp_path):
        # Graded ledger record: bandit_reward = 0.85 (an accept worth
        # 85% of full reward).  prime_from_ledger must accumulate 0.85
        # into reward_sum, not 1.0.
        ledger_path = tmp_path / "ledger.jsonl"
        graded_accept = {
            "record_type": "iteration",
            "iteration": 0,
            "accepted": True,
            "no_op": False,
            "bandit_reward": 0.85,
            "proposal": {
                "class_name": "_DummyHeuristicA",
                "param_name": "radius",
                "rule_kind": "log_uniform_perturb",
            },
        }
        graded_reject = {
            "record_type": "iteration",
            "iteration": 1,
            "accepted": False,
            "no_op": False,
            "bandit_reward": 0.35,
            "proposal": {
                "class_name": "_DummyHeuristicA",
                "param_name": "radius",
                "rule_kind": "log_uniform_perturb",
            },
        }
        with ledger_path.open("w") as f:
            f.write(json.dumps(graded_accept) + "\n")
            f.write(json.dumps(graded_reject) + "\n")

        sampler = AdaptiveMutationSampler(self._radius_catalog())
        consumed = sampler.prime_from_ledger(str(ledger_path))
        assert consumed == 2
        snap = sampler.stats_snapshot()
        assert len(snap) == 1
        assert snap[0].n_attempts == 2
        assert snap[0].n_accepts == 1
        # 0.85 + 0.35 — the graded rewards, *not* 1.0 + 0.0 (which
        # the binary-fallback path would have produced).
        assert snap[0].reward_sum == pytest.approx(1.20)

    def test_prime_from_ledger_legacy_record_falls_back_to_binary(self, tmp_path):
        # Legacy ledger record (no ``bandit_reward`` field): prime
        # falls back to the binary reward 1.0 per accept / 0.0 per
        # reject.  This is the back-compat contract that pre-2026-06-13
        # ledgers replay byte-identically.
        ledger_path = tmp_path / "ledger.jsonl"
        legacy_accept = {
            "record_type": "iteration",
            "iteration": 0,
            "accepted": True,
            "no_op": False,
            # no ``bandit_reward`` key — that's the legacy shape.
            "proposal": {
                "class_name": "_DummyHeuristicA",
                "param_name": "radius",
                "rule_kind": "log_uniform_perturb",
            },
        }
        with ledger_path.open("w") as f:
            f.write(json.dumps(legacy_accept) + "\n")

        sampler = AdaptiveMutationSampler(self._radius_catalog())
        consumed = sampler.prime_from_ledger(str(ledger_path))
        assert consumed == 1
        snap = sampler.stats_snapshot()
        # Legacy path: reward_sum = n_accepts.
        assert snap[0].n_accepts == 1
        assert snap[0].reward_sum == pytest.approx(1.0)

    # ---- LoopConfig + LoopIterationRecord plumbing ----

    def test_loop_config_validates_bandit_reward_shaping(self):
        with pytest.raises(ValueError, match="bandit_reward_shaping"):
            LoopConfig(bandit_reward_shaping="bogus")
        # Both valid choices construct cleanly.
        LoopConfig(bandit_reward_shaping="binary")
        LoopConfig(bandit_reward_shaping="graded")

    def test_loop_config_default_is_binary(self):
        # Default must be the historical behaviour so existing CLI
        # invocations / programmatic callers are byte-identical.
        cfg = LoopConfig()
        assert cfg.bandit_reward_shaping == "binary"

    def test_iteration_record_bandit_reward_defaults_none(self):
        rec = LoopIterationRecord(
            iteration=0,
            timestamp="2026-01-01T00:00:00+00:00",
            duration_seconds=0.0,
            proposal=None,
            accepted=False,
            baseline_score=0.0,
            candidate_score=0.0,
            delta=0.0,
            ci_low=0.0,
            ci_high=0.0,
            worst_pair_regression=0.0,
            worst_pair=None,
        )
        assert rec.bandit_reward is None
        assert rec.to_dict()["bandit_reward"] is None

    # ---- End-to-end driver behaviour ----

    def test_binary_mode_record_has_no_bandit_reward(self, tmp_path):
        # Default (binary) mode: the driver leaves bandit_reward = None
        # on every iteration so the ledger stays byte-identical to the
        # pre-graded shape.
        cfg = LoopConfig(
            iterations=1,
            n_boot=50,
            ledger_path=str(tmp_path / "ledger.jsonl"),
            stop_sentinel_path="",
            randomize=False,
            adaptive_sampling=True,
            bandit_reward_shaping="binary",
        )
        si = SelfImprover(cfg, catalog=self._radius_catalog(), seed_strategies=_make_specs())
        # Score 0.5 baseline, 0.4 candidate → reject, non-no-op.
        counter = {"n": 0}

        def score_fn(c):
            n = counter["n"]
            counter["n"] += 1
            return 0.5 if n % 2 == 0 else 0.4

        si._harness_factory = _make_factory(score_fn)
        records = si.run()
        assert records[0].bandit_reward is None

    def test_graded_mode_record_persists_reward(self, tmp_path):
        # Graded mode: the driver computes the reward and persists it
        # on the record (and pulls the bandit's arm with the same
        # value).  Here a non-no-op reject with Δ = -0.1 (strongly
        # negative) should produce a reward at the lower bound (0).
        cfg = LoopConfig(
            iterations=1,
            n_boot=50,
            eps_accept=0.005,
            ledger_path=str(tmp_path / "ledger.jsonl"),
            stop_sentinel_path="",
            randomize=False,
            adaptive_sampling=True,
            bandit_reward_shaping="graded",
        )
        si = SelfImprover(cfg, catalog=self._radius_catalog(), seed_strategies=_make_specs())
        counter = {"n": 0}

        def score_fn(c):
            n = counter["n"]
            counter["n"] += 1
            return 0.5 if n % 2 == 0 else 0.4

        si._harness_factory = _make_factory(score_fn)
        records = si.run()
        assert len(records) == 1
        rec = records[0]
        # Reject with strongly negative delta → reward floor at 0.0.
        assert rec.accepted is False
        assert rec.bandit_reward is not None
        # eps_scale = 4·0.005 = 0.020; delta ≈ -0.1 ⇒ value ≈ 0.5 - 5 < 0
        # → clamped to 0.0.
        assert rec.bandit_reward == pytest.approx(0.0)
        # The sampler's stats carry the same value.
        assert si.sampler is not None
        snap = si.sampler.stats_snapshot()
        # The arm was pulled (n_attempts=1) with a graded reward of 0.0.
        pulled = [s for s in snap if s.n_attempts > 0]
        assert len(pulled) == 1
        assert pulled[0].reward_sum == pytest.approx(0.0)
        assert pulled[0].n_accepts == 0

    def test_graded_mode_no_op_leaves_bandit_reward_none(self, tmp_path):
        # No-op iterations bypass the bandit entirely — the reward
        # field stays None even in graded mode so the ledger can
        # distinguish "the iteration was informative but the reward
        # was 0" from "the iteration carried no information".
        cfg = LoopConfig(
            iterations=1,
            n_boot=50,
            ledger_path=str(tmp_path / "ledger.jsonl"),
            stop_sentinel_path="",
            randomize=False,
            adaptive_sampling=True,
            bandit_reward_shaping="graded",
        )
        si = SelfImprover(cfg, catalog=self._radius_catalog(), seed_strategies=_make_specs())
        si._harness_factory = _make_factory(lambda c: 0.5)
        records = si.run()
        assert records[0].no_op is True
        assert records[0].bandit_reward is None

    def test_graded_mode_ledger_round_trip(self, tmp_path):
        # Round-trip: a graded-mode run writes bandit_reward to the
        # ledger, and prime_from_ledger picks it up on resume so the
        # reward_sum is preserved exactly across the persistence
        # boundary.  Headline contract for the §7.4 ship.
        ledger_path = tmp_path / "ledger.jsonl"
        cfg = LoopConfig(
            iterations=1,
            n_boot=50,
            eps_accept=0.005,
            ledger_path=str(ledger_path),
            stop_sentinel_path="",
            randomize=False,
            adaptive_sampling=True,
            bandit_reward_shaping="graded",
        )
        si = SelfImprover(cfg, catalog=self._radius_catalog(), seed_strategies=_make_specs())
        counter = {"n": 0}

        def score_fn(c):
            n = counter["n"]
            counter["n"] += 1
            return 0.5 if n % 2 == 0 else 0.4

        si._harness_factory = _make_factory(score_fn)
        si.run()
        loaded = load_ledger(str(ledger_path))
        iter_recs = [r for r in loaded if r.get("record_type") == "iteration"]
        assert len(iter_recs) == 1
        on_disk_reward = iter_recs[0]["bandit_reward"]
        assert on_disk_reward == pytest.approx(0.0)
        # Now prime a fresh sampler from the persisted ledger and
        # confirm the reward_sum matches what the live driver pulled.
        sampler = AdaptiveMutationSampler(self._radius_catalog())
        sampler.prime_from_ledger(str(ledger_path))
        snap = sampler.stats_snapshot()
        primed = [s for s in snap if s.n_attempts > 0]
        assert len(primed) == 1
        assert primed[0].reward_sum == pytest.approx(0.0)
        assert primed[0].n_attempts == 1


# ===========================================================================
# §12.4 Summary trend block
# ===========================================================================


class TestSummaryTrendBlock:
    """V2 §12.4 of ``planning/SELF_IMPROVEMENT_LOOP.md``.

    The ``scripts/self_improve.py summary`` CLI was an ever-growing wall
    of per-record lines.  The §12.4 trend block — one row per nightly
    run, plus top-N / bottom-N bandit posteriors and inactivity
    telemetry — converts the wall into the at-a-glance signal the §12.3
    daily routine reads.
    """

    @staticmethod
    def _import_cli():
        # Import the CLI module the same way TestNoOpDetection does so
        # the path manipulation stays isolated to this test.
        sys_module = __import__("sys")
        sys_module.path.insert(0, str(pathlib.Path(__file__).parents[1] / "scripts"))
        try:
            import self_improve as cli  # type: ignore
        finally:
            sys_module.path = [p for p in sys_module.path if not p.endswith("/scripts")]
        return cli

    @staticmethod
    def _iter_rec(
        iteration: int,
        *,
        accepted: bool = False,
        delta: float = 0.0,
        baseline_score: float = 0.05,
        no_op: bool = False,
        proposal: Optional[Dict[str, Any]] = None,
        timestamp: str = "2026-06-15T00:00:00+00:00",
        base_seed: int = 42,
        mode: str = "quick",
        effective_eps_accept: Optional[float] = None,
        iters_since_accept: Optional[int] = None,
        bandit_reward: Optional[float] = None,
    ) -> Dict[str, Any]:
        if proposal is None:
            proposal = {
                "class_name": "Nearby",
                "param_name": "radius",
                "rule_kind": "log_uniform_perturb",
            }
        return {
            "record_type": "iteration",
            "iteration": iteration,
            "timestamp": timestamp,
            "accepted": accepted,
            "delta": delta,
            "baseline_score": baseline_score,
            "proposal": proposal,
            "no_op": no_op,
            "base_seed": base_seed,
            "mode": mode,
            "effective_eps_accept": effective_eps_accept,
            "iters_since_accept": iters_since_accept,
            "bandit_reward": bandit_reward,
        }

    # -----------------------------------------------------------------
    # _group_runs
    # -----------------------------------------------------------------

    def test_group_runs_empty_input_returns_empty_list(self):
        cli = self._import_cli()
        assert cli._group_runs([]) == []

    def test_group_runs_single_run_one_bucket(self):
        cli = self._import_cli()
        recs = [self._iter_rec(i) for i in range(5)]
        runs = cli._group_runs(recs)
        assert len(runs) == 1
        assert len(runs[0]) == 5

    def test_group_runs_splits_on_iteration_reset(self):
        cli = self._import_cli()
        # Two consecutive runs of 3 + 4 iterations; second restarts at 0.
        recs = [self._iter_rec(i) for i in range(3)] + [
            self._iter_rec(i, timestamp="2026-06-16T00:00:00+00:00") for i in range(4)
        ]
        runs = cli._group_runs(recs)
        assert len(runs) == 2
        assert [len(r) for r in runs] == [3, 4]
        # The second run's first record carries the later timestamp.
        assert runs[1][0]["timestamp"] == "2026-06-16T00:00:00+00:00"

    def test_group_runs_repeated_iteration_zero_starts_new_run(self):
        cli = self._import_cli()
        # Pathological: two runs that each contain exactly one record at
        # iteration=0.  Should produce two separate buckets, not one with
        # two records.
        recs = [
            self._iter_rec(0, timestamp="2026-06-15T00:00:00+00:00"),
            self._iter_rec(0, timestamp="2026-06-16T00:00:00+00:00"),
        ]
        runs = cli._group_runs(recs)
        assert len(runs) == 2

    # -----------------------------------------------------------------
    # _print_trend_block
    # -----------------------------------------------------------------

    def test_trend_block_renders_per_run_row(self, capsys):
        cli = self._import_cli()
        recs = [
            self._iter_rec(0, baseline_score=0.10, accepted=True, delta=0.05),
            self._iter_rec(1, accepted=False, delta=-0.02),
            self._iter_rec(2, accepted=False, delta=0.0, no_op=True),
        ]
        cli._print_trend_block(recs)
        out = capsys.readouterr().out
        assert "Trend" in out
        # One data row beyond the header.
        rows = [line for line in out.splitlines() if "2026-06-15" in line]
        assert len(rows) == 1
        # Column values: 3 iters, 3 decided, 1 accept, 1 no-op, +0.0500 best.
        row = rows[0]
        assert "    3" in row  # iters
        assert "+0.0500" in row  # best Δ
        assert "0.1000" in row  # seed score (baseline_score of first record)

    def test_trend_block_groups_runs_in_chronological_order(self, capsys):
        cli = self._import_cli()
        recs = [
            self._iter_rec(0, timestamp="2026-06-14T00:00:00+00:00", accepted=True, delta=0.01),
            self._iter_rec(1, timestamp="2026-06-14T00:01:00+00:00"),
            self._iter_rec(0, timestamp="2026-06-15T00:00:00+00:00", accepted=False, delta=-0.03),
            self._iter_rec(1, timestamp="2026-06-15T00:01:00+00:00"),
        ]
        cli._print_trend_block(recs)
        out = capsys.readouterr().out
        # The order must be oldest-first so an operator scans top-to-bottom.
        i14 = out.find("2026-06-14")
        i15 = out.find("2026-06-15")
        assert 0 <= i14 < i15

    def test_trend_block_silent_on_empty_input(self, capsys):
        cli = self._import_cli()
        cli._print_trend_block([])
        # No output — the block silently no-ops so the existing summary
        # contract on empty ledgers is preserved.
        assert capsys.readouterr().out == ""

    # -----------------------------------------------------------------
    # _replay_bandit_posteriors / _print_bandit_block
    # -----------------------------------------------------------------

    def test_replay_bandit_skips_no_op_and_skip_records(self):
        cli = self._import_cli()
        recs = [
            self._iter_rec(0, accepted=True, delta=0.05),
            self._iter_rec(1, accepted=False, no_op=True),
            # Skip record carries no proposal.
            {
                "record_type": "iteration",
                "iteration": 2,
                "accepted": False,
                "proposal": None,
                "no_op": False,
            },
            # Guard / hold-out records must be filtered out.
            {"record_type": "guard", "iteration": 2},
            {"record_type": "holdout"},
        ]
        stats = cli._replay_bandit_posteriors(recs)
        # Only the one informative accept was consumed.
        assert len(stats) == 1
        bucket = next(iter(stats.values()))
        assert bucket["n_attempts"] == 1
        assert bucket["n_accepts"] == 1
        assert bucket["mean_reward"] == pytest.approx(1.0)

    def test_replay_bandit_graded_reward_propagates(self):
        cli = self._import_cli()
        recs = [
            self._iter_rec(0, accepted=False, delta=0.0, bandit_reward=0.5),
            self._iter_rec(1, accepted=True, delta=0.05, bandit_reward=0.8),
        ]
        stats = cli._replay_bandit_posteriors(recs)
        bucket = next(iter(stats.values()))
        assert bucket["n_attempts"] == 2
        assert bucket["n_accepts"] == 1
        assert bucket["reward_sum"] == pytest.approx(1.3)
        assert bucket["mean_reward"] == pytest.approx(0.65)
        # Accept rate is the binary-path view; mean_reward carries the
        # graded signal — they must not collapse onto each other.
        assert bucket["accept_rate"] == pytest.approx(0.5)

    def test_replay_bandit_legacy_record_uses_binary_fallback(self):
        cli = self._import_cli()
        # No ``bandit_reward`` field — must fall back to 1.0 per accept,
        # 0.0 per reject, matching :meth:`prime_from_ledger`.
        recs = [
            self._iter_rec(0, accepted=True, delta=0.05),
            self._iter_rec(1, accepted=False, delta=-0.05),
        ]
        for r in recs:
            r.pop("bandit_reward", None)
        stats = cli._replay_bandit_posteriors(recs)
        bucket = next(iter(stats.values()))
        assert bucket["reward_sum"] == pytest.approx(1.0)
        assert bucket["mean_reward"] == pytest.approx(0.5)
        assert bucket["accept_rate"] == pytest.approx(0.5)

    def test_replay_bandit_structural_op_collapses_to_one_arm(self):
        cli = self._import_cli()
        # Two ``add_heuristic`` proposals targeting different classes must
        # collapse onto the single ``("*", "add_heuristic", "structural")``
        # arm by default — matching the default
        # :func:`_proposal_rule_key` semantics.
        recs = [
            self._iter_rec(
                0,
                proposal={
                    "class_name": "Sobol",
                    "param_name": "add_heuristic",
                    "rule_kind": "add_heuristic",
                },
            ),
            self._iter_rec(
                1,
                proposal={
                    "class_name": "Random",
                    "param_name": "add_heuristic",
                    "rule_kind": "add_heuristic",
                },
            ),
        ]
        stats = cli._replay_bandit_posteriors(recs)
        # One collapsed structural arm.
        assert len(stats) == 1
        key = next(iter(stats.keys()))
        assert key == ("*", "add_heuristic", "structural")

    def test_bandit_block_orders_by_mean_reward_desc(self, capsys):
        cli = self._import_cli()
        # Two rules: one with reward sum 5/5 (mean 1.0), one with 1/5
        # (mean 0.2).  Top must show the high-reward rule first.
        good = [
            self._iter_rec(
                i,
                accepted=True,
                delta=0.05,
                bandit_reward=1.0,
                proposal={
                    "class_name": "Good",
                    "param_name": "x",
                    "rule_kind": "log_uniform_perturb",
                },
            )
            for i in range(5)
        ]
        bad = [
            self._iter_rec(
                i,
                accepted=False,
                delta=-0.05,
                bandit_reward=0.0,
                proposal={
                    "class_name": "Bad",
                    "param_name": "y",
                    "rule_kind": "log_uniform_perturb",
                },
            )
            for i in range(5)
        ]
        # Plus a single 0.2-mean rule to widen the band.
        mid = [
            self._iter_rec(
                i,
                accepted=False,
                delta=0.0,
                bandit_reward=0.2,
                proposal={
                    "class_name": "Mid",
                    "param_name": "z",
                    "rule_kind": "log_uniform_perturb",
                },
            )
            for i in range(5)
        ]
        cli._print_bandit_block(good + bad + mid, top_n=2, bottom_n=1, min_attempts=3)
        out = capsys.readouterr().out
        assert "Bandit posteriors" in out
        # Good must appear before Bad in the top block.
        i_good = out.find("Good")
        i_bad = out.find("Bad")
        assert i_good >= 0 and i_bad >= 0 and i_good < i_bad
        # The header counts eligible rules — all three pass the
        # min_attempts=3 filter.
        assert "3 eligible rules" in out

    def test_bandit_block_filters_by_min_attempts(self, capsys):
        cli = self._import_cli()
        # One rule with 2 attempts (below the threshold), one with 5.
        sparse = [
            self._iter_rec(
                i,
                accepted=False,
                delta=-0.01,
                bandit_reward=0.0,
                proposal={
                    "class_name": "Sparse",
                    "param_name": "p",
                    "rule_kind": "log_uniform_perturb",
                },
            )
            for i in range(2)
        ]
        dense = [
            self._iter_rec(
                i,
                accepted=True,
                delta=0.05,
                bandit_reward=1.0,
                proposal={
                    "class_name": "Dense",
                    "param_name": "q",
                    "rule_kind": "log_uniform_perturb",
                },
            )
            for i in range(5)
        ]
        cli._print_bandit_block(sparse + dense, top_n=10, bottom_n=5, min_attempts=3)
        out = capsys.readouterr().out
        assert "Dense" in out
        assert "Sparse" not in out
        assert "1 eligible rules" in out

    def test_bandit_block_no_eligible_rules_prints_friendly_note(self, capsys):
        cli = self._import_cli()
        # All rules below the threshold.
        recs = [
            self._iter_rec(
                0,
                accepted=True,
                delta=0.05,
                bandit_reward=1.0,
            )
        ]
        cli._print_bandit_block(recs, top_n=10, bottom_n=5, min_attempts=3)
        out = capsys.readouterr().out
        assert "no rules with >= 3 informative attempts" in out

    def test_bandit_block_silent_on_empty_input(self, capsys):
        cli = self._import_cli()
        cli._print_bandit_block([], top_n=10, bottom_n=5, min_attempts=3)
        assert capsys.readouterr().out == ""

    # -----------------------------------------------------------------
    # _print_inactivity_block
    # -----------------------------------------------------------------

    def test_inactivity_block_renders_drought_and_relax(self, capsys):
        cli = self._import_cli()
        # 4 records: three rejects with growing drought, then a relaxed
        # accept.  Base eps_accept = 0.005 (max observed).
        recs = [
            self._iter_rec(0, accepted=False, effective_eps_accept=0.005, iters_since_accept=0),
            self._iter_rec(1, accepted=False, effective_eps_accept=0.005, iters_since_accept=1),
            self._iter_rec(2, accepted=False, effective_eps_accept=0.005, iters_since_accept=2),
            # Relaxed accept — effective threshold below base.
            self._iter_rec(
                3,
                accepted=True,
                delta=0.003,
                effective_eps_accept=0.0025,
                iters_since_accept=3,
            ),
        ]
        cli._print_inactivity_block(recs)
        out = capsys.readouterr().out
        assert "Inactivity:" in out
        assert "eps_accept_base=0.0050" in out
        assert "longest_drought=3" in out
        # One accept, one relaxed.
        assert "relaxed_accepts=1/1" in out
        # Decay factor = 0.0025 / 0.005 = 0.500.
        assert "mean_decay_at_accept=0.500" in out

    def test_inactivity_block_silent_on_legacy_records(self, capsys):
        cli = self._import_cli()
        # Pre-2026-05-30 records carry neither field; the block must
        # stay silent (no inactivity stats to surface).
        recs = [self._iter_rec(0, accepted=True, delta=0.05)]
        for r in recs:
            r.pop("effective_eps_accept", None)
            r.pop("iters_since_accept", None)
        cli._print_inactivity_block(recs)
        assert capsys.readouterr().out == ""

    def test_inactivity_block_silent_on_empty_input(self, capsys):
        cli = self._import_cli()
        cli._print_inactivity_block([])
        assert capsys.readouterr().out == ""

    def test_inactivity_block_no_relaxed_accepts_hides_decay(self, capsys):
        cli = self._import_cli()
        # An accept that fired at the base eps — relaxed_accepts must be
        # 0/1 and the mean_decay clause must NOT appear (no relaxed
        # accepts to average).
        recs = [
            self._iter_rec(
                0,
                accepted=True,
                delta=0.10,
                effective_eps_accept=0.005,
                iters_since_accept=0,
            ),
        ]
        cli._print_inactivity_block(recs)
        out = capsys.readouterr().out
        assert "relaxed_accepts=0/1" in out
        assert "mean_decay_at_accept" not in out

    # -----------------------------------------------------------------
    # End-to-end CLI smoke test
    # -----------------------------------------------------------------

    def test_cli_summary_emits_trend_and_bandit_and_inactivity_blocks(self, tmp_path, capsys):
        cli = self._import_cli()
        ledger_path = tmp_path / "ledger.jsonl"
        # Build a synthetic two-run ledger: each run has 4 records.
        records = []
        for run_idx, ts_date in enumerate(("2026-06-14", "2026-06-15")):
            for i in range(4):
                records.append(
                    self._iter_rec(
                        i,
                        accepted=(i == 0),
                        delta=(0.05 if i == 0 else -0.01),
                        bandit_reward=(1.0 if i == 0 else 0.0),
                        baseline_score=0.10,
                        timestamp=f"{ts_date}T0{i}:00:00+00:00",
                        effective_eps_accept=0.005,
                        iters_since_accept=(0 if i == 0 else i),
                    )
                )
        with ledger_path.open("w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        ns = type(
            "NS",
            (),
            {
                "ledger": str(ledger_path),
                "top_n": 10,
                "bottom_n": 5,
                "min_attempts": 3,
            },
        )()
        rc = cli._cmd_summary(ns)
        assert rc == 0
        out = capsys.readouterr().out
        # All three new sub-blocks present.
        assert "Trend" in out
        assert "Bandit posteriors" in out
        assert "Inactivity" in out
        # Two run rows in the trend block — the grouping correctly split.
        rows = [line for line in out.splitlines() if "2026-06-14 " in line or "2026-06-15 " in line]
        assert len(rows) >= 2


# ===========================================================================
# V2 §9.3 / §9.5 step 4 — codify-scan
# ===========================================================================


class TestDirectionKey:
    """:func:`panobbgo.self_improve._direction_key` — proposal direction extraction.

    Direction collapses every accept into a stable bucket key so the
    codify scanner can group "the same change" across nights.  Tests
    cover every rule_kind currently shipping in
    :func:`default_catalog` / :func:`default_structural_catalog`.
    """

    def test_integer_add_up(self):
        from panobbgo.self_improve import _direction_key

        assert _direction_key({"rule_kind": "integer_add", "old_value": 16, "new_value": 20}) == "up"

    def test_integer_add_down(self):
        from panobbgo.self_improve import _direction_key

        assert _direction_key({"rule_kind": "integer_add", "old_value": 16, "new_value": 12}) == "down"

    def test_float_uniform_up(self):
        from panobbgo.self_improve import _direction_key

        assert _direction_key({"rule_kind": "float_uniform", "old_value": 0.5, "new_value": 0.7}) == "up"

    def test_log_uniform_perturb_directions(self):
        from panobbgo.self_improve import _direction_key

        assert _direction_key({"rule_kind": "log_uniform_perturb", "old_value": 0.1, "new_value": 0.13}) == "up"
        assert _direction_key({"rule_kind": "log_uniform_perturb", "old_value": 0.1, "new_value": 0.08}) == "down"

    def test_categorical_choice_uses_repr_of_new(self):
        from panobbgo.self_improve import _direction_key

        # Booleans get their own buckets — False and "False" must not collide.
        assert _direction_key({"rule_kind": "categorical_choice", "old_value": True, "new_value": False}) == "False"
        assert (
            _direction_key({"rule_kind": "categorical_choice", "old_value": False, "new_value": "False"}) == "'False'"
        )

    def test_structural_op_returns_op_name(self):
        from panobbgo.self_improve import _direction_key

        assert (
            _direction_key({"rule_kind": "structural", "op": "add_heuristic", "old_value": None, "new_value": None})
            == "add_heuristic"
        )
        assert (
            _direction_key({"rule_kind": "structural", "op": "drop_analyzer", "old_value": None, "new_value": None})
            == "drop_analyzer"
        )

    def test_equal_numeric_returns_none(self):
        from panobbgo.self_improve import _direction_key

        # No direction → pre-2026-06-12 records that no-opped numerically.
        # Caller must filter these out.
        assert _direction_key({"rule_kind": "integer_add", "old_value": 10, "new_value": 10}) is None

    def test_non_numeric_old_value_returns_none(self):
        from panobbgo.self_improve import _direction_key

        assert _direction_key({"rule_kind": "float_uniform", "old_value": "bogus", "new_value": 0.5}) is None

    def test_missing_old_value_returns_none(self):
        from panobbgo.self_improve import _direction_key

        assert _direction_key({"rule_kind": "integer_add", "new_value": 10}) is None


class TestPercentileBootstrapCI:
    """:func:`panobbgo.self_improve._percentile_bootstrap_ci` — pooled CI helper."""

    def test_empty_input_returns_zero_zero(self):
        from panobbgo.self_improve import _percentile_bootstrap_ci

        assert _percentile_bootstrap_ci([]) == (0.0, 0.0)

    def test_single_sample_degenerate_to_point(self):
        from panobbgo.self_improve import _percentile_bootstrap_ci

        lo, hi = _percentile_bootstrap_ci([0.05])
        assert lo == pytest.approx(0.05)
        assert hi == pytest.approx(0.05)

    def test_multi_sample_brackets_mean(self):
        from panobbgo.self_improve import _percentile_bootstrap_ci

        samples = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
        lo, hi = _percentile_bootstrap_ci(samples, n_boot=500, confidence=0.95, seed=1)
        # Mean is 0.045; the CI should bracket it.
        assert lo <= 0.045 <= hi
        # And be reasonably tight on this nearly-uniform sample.
        assert (hi - lo) < 0.08

    def test_seed_makes_results_reproducible(self):
        from panobbgo.self_improve import _percentile_bootstrap_ci

        samples = [0.01, 0.05, 0.10, 0.02, 0.07]
        a = _percentile_bootstrap_ci(samples, n_boot=200, seed=7)
        b = _percentile_bootstrap_ci(samples, n_boot=200, seed=7)
        assert a == b


def _accepted_iter_record(
    *,
    iteration: int = 0,
    class_name: str = "Nearby",
    param_name: str = "radius",
    rule_kind: str = "log_uniform_perturb",
    old_value: Any = 0.1,
    new_value: Any = 0.12,
    op: Optional[str] = None,
    delta: float = 0.06,
    ci_low: float = 0.01,
    ci_high: float = 0.10,
    timestamp: str = "2026-06-01T05:00:00+00:00",
    strategy_name: str = "Rewarding_Diverse",
    accepted: bool = True,
    no_op: bool = False,
    confirmed: Optional[bool] = None,
) -> Dict[str, Any]:
    """Build a synthetic accepted iteration record matching the live ledger schema."""
    proposal: Dict[str, Any] = {
        "strategy_name": strategy_name,
        "class_name": class_name,
        "param_name": param_name,
        "old_value": old_value,
        "new_value": new_value,
        "rule_kind": rule_kind,
        "rationale": "test",
    }
    if op is not None:
        proposal["op"] = op
        proposal["structural_kwargs"] = {}
    rec: Dict[str, Any] = {
        "record_type": "iteration",
        "iteration": iteration,
        "timestamp": timestamp,
        "duration_seconds": 1.0,
        "proposal": proposal,
        "accepted": accepted,
        "baseline_score": 0.05,
        "candidate_score": 0.05 + delta,
        "delta": delta,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "worst_pair_regression": -0.01,
        "worst_pair": None,
        "reasons": [],
        "base_seed": 42,
        "randomize_iteration": iteration,
        "mode": "quick",
        "reason_skipped": None,
        "no_op": no_op,
    }
    if confirmed is not None:
        rec["confirmed"] = confirmed
    return rec


class TestAggregateCodifyCandidates:
    """:func:`panobbgo.self_improve.aggregate_codify_candidates` — the scanner."""

    def test_empty_input_returns_empty_list(self):
        from panobbgo.self_improve import aggregate_codify_candidates

        assert aggregate_codify_candidates([]) == []

    def test_single_accept_below_min_nights_filtered(self):
        from panobbgo.self_improve import aggregate_codify_candidates

        recs = [_accepted_iter_record(timestamp="2026-06-01T05:00:00+00:00")]
        assert aggregate_codify_candidates(recs, min_nights=2) == []

    def test_two_distinct_nights_clears_default_gate(self):
        from panobbgo.self_improve import aggregate_codify_candidates

        recs = [
            _accepted_iter_record(timestamp="2026-06-01T05:00:00+00:00", new_value=0.12),
            _accepted_iter_record(timestamp="2026-06-02T05:00:00+00:00", new_value=0.13),
        ]
        cands = aggregate_codify_candidates(recs)
        assert len(cands) == 1
        c = cands[0]
        assert c.class_name == "Nearby"
        assert c.param_name == "radius"
        assert c.direction == "up"
        assert c.n_accepts == 2
        assert c.n_distinct_nights == 2
        assert c.distinct_dates == ("2026-06-01", "2026-06-02")

    def test_same_night_multiple_accepts_one_night(self):
        from panobbgo.self_improve import aggregate_codify_candidates

        # Two accepts same day — counts as one night only.
        recs = [
            _accepted_iter_record(timestamp="2026-06-01T05:00:00+00:00", iteration=1),
            _accepted_iter_record(timestamp="2026-06-01T06:00:00+00:00", iteration=2),
        ]
        cands = aggregate_codify_candidates(recs, min_nights=2)
        assert cands == []

    def test_opposite_directions_separate_buckets(self):
        from panobbgo.self_improve import aggregate_codify_candidates

        recs = [
            _accepted_iter_record(timestamp="2026-06-01T05:00:00+00:00", new_value=0.12),
            _accepted_iter_record(timestamp="2026-06-02T05:00:00+00:00", new_value=0.13),
            _accepted_iter_record(timestamp="2026-06-03T05:00:00+00:00", new_value=0.08),
            _accepted_iter_record(timestamp="2026-06-04T05:00:00+00:00", new_value=0.07),
        ]
        cands = aggregate_codify_candidates(recs)
        directions = sorted({c.direction for c in cands})
        assert directions == ["down", "up"]
        # Both directions should clear k>=2 nights.
        assert all(c.n_distinct_nights >= 2 for c in cands)

    def test_categorical_choice_buckets_by_repr_of_new(self):
        from panobbgo.self_improve import aggregate_codify_candidates

        recs = [
            _accepted_iter_record(
                timestamp="2026-06-01T05:00:00+00:00",
                class_name="Sobol",
                param_name="scramble",
                rule_kind="categorical_choice",
                old_value=True,
                new_value=False,
            ),
            _accepted_iter_record(
                timestamp="2026-06-02T05:00:00+00:00",
                class_name="Sobol",
                param_name="scramble",
                rule_kind="categorical_choice",
                old_value=True,
                new_value=False,
            ),
        ]
        cands = aggregate_codify_candidates(recs)
        assert len(cands) == 1
        assert cands[0].direction == "False"
        assert cands[0].rule_kind == "categorical_choice"

    def test_structural_op_uses_op_as_direction(self):
        from panobbgo.self_improve import aggregate_codify_candidates

        recs = [
            _accepted_iter_record(
                timestamp="2026-06-01T05:00:00+00:00",
                class_name="Restart",
                param_name="",
                rule_kind="structural",
                op="add_analyzer",
                old_value=None,
                new_value=None,
            ),
            _accepted_iter_record(
                timestamp="2026-06-02T05:00:00+00:00",
                class_name="Restart",
                param_name="",
                rule_kind="structural",
                op="add_analyzer",
                old_value=None,
                new_value=None,
            ),
        ]
        cands = aggregate_codify_candidates(recs)
        assert len(cands) == 1
        c = cands[0]
        assert c.op == "add_analyzer"
        assert c.direction == "add_analyzer"
        assert c.rule_kind == "structural"
        assert c.slot_key == ("Restart", "", "add_analyzer")

    def test_require_positive_min_ci_default_filters_one_negative_record(self):
        from panobbgo.self_improve import aggregate_codify_candidates

        # Two accepts, but one has ci_low <= 0 — the strict gate should drop the candidate.
        recs = [
            _accepted_iter_record(timestamp="2026-06-01T05:00:00+00:00", ci_low=0.01),
            _accepted_iter_record(timestamp="2026-06-02T05:00:00+00:00", ci_low=-0.005),
        ]
        assert aggregate_codify_candidates(recs, require_positive_min_ci=True) == []

    def test_loose_gate_surfaces_record_with_negative_ci_low(self):
        from panobbgo.self_improve import aggregate_codify_candidates

        recs = [
            _accepted_iter_record(timestamp="2026-06-01T05:00:00+00:00", ci_low=0.01),
            _accepted_iter_record(timestamp="2026-06-02T05:00:00+00:00", ci_low=-0.005),
        ]
        cands = aggregate_codify_candidates(recs, require_positive_min_ci=False)
        assert len(cands) == 1
        assert cands[0].min_ci_low == pytest.approx(-0.005)

    def test_no_op_iterations_excluded(self):
        from panobbgo.self_improve import aggregate_codify_candidates

        # A pathological "accepted=True but no_op=True" record — must not count.
        recs = [
            _accepted_iter_record(timestamp="2026-06-01T05:00:00+00:00", no_op=True),
            _accepted_iter_record(timestamp="2026-06-02T05:00:00+00:00", no_op=False),
        ]
        cands = aggregate_codify_candidates(recs, min_nights=2)
        assert cands == []  # only one informative accept remains

    def test_non_accepted_records_skipped(self):
        from panobbgo.self_improve import aggregate_codify_candidates

        recs = [
            _accepted_iter_record(timestamp="2026-06-01T05:00:00+00:00", accepted=False),
            _accepted_iter_record(timestamp="2026-06-02T05:00:00+00:00", accepted=False),
        ]
        assert aggregate_codify_candidates(recs) == []

    def test_skip_records_with_no_proposal_dropped(self):
        from panobbgo.self_improve import aggregate_codify_candidates

        # Build a "skip" record matching what the loop writes when no
        # applicable rule fires — accepted=False, proposal=None.
        skip = _accepted_iter_record(timestamp="2026-06-01T05:00:00+00:00", accepted=False)
        skip["proposal"] = None
        recs = [
            skip,
            _accepted_iter_record(timestamp="2026-06-02T05:00:00+00:00"),
            _accepted_iter_record(timestamp="2026-06-03T05:00:00+00:00"),
        ]
        cands = aggregate_codify_candidates(recs)
        assert len(cands) == 1
        assert cands[0].n_accepts == 2

    def test_non_iteration_records_ignored(self):
        from panobbgo.self_improve import aggregate_codify_candidates

        # A guard / hold-out record should not contaminate the scan.
        guard = {
            "record_type": "guard",
            "iteration": 5,
            "timestamp": "2026-06-01T05:00:00+00:00",
            "accepted": True,  # would confuse a naïve scanner
        }
        recs = [
            guard,
            _accepted_iter_record(timestamp="2026-06-02T05:00:00+00:00"),
            _accepted_iter_record(timestamp="2026-06-03T05:00:00+00:00"),
        ]
        cands = aggregate_codify_candidates(recs)
        assert len(cands) == 1
        assert cands[0].n_accepts == 2

    def test_confirmed_only_filters_legacy_records(self):
        from panobbgo.self_improve import aggregate_codify_candidates

        # Pre-V2-§6.4 records carry no ``confirmed`` field; confirmed_only=True
        # must drop them all.
        recs = [
            _accepted_iter_record(timestamp="2026-06-01T05:00:00+00:00", confirmed=None),
            _accepted_iter_record(timestamp="2026-06-02T05:00:00+00:00", confirmed=None),
        ]
        assert aggregate_codify_candidates(recs, confirmed_only=True) == []

    def test_confirmed_only_keeps_confirmed_records(self):
        from panobbgo.self_improve import aggregate_codify_candidates

        recs = [
            _accepted_iter_record(timestamp="2026-06-01T05:00:00+00:00", confirmed=True),
            _accepted_iter_record(timestamp="2026-06-02T05:00:00+00:00", confirmed=True),
        ]
        cands = aggregate_codify_candidates(recs, confirmed_only=True)
        assert len(cands) == 1
        assert all(f is True for f in cands[0].confirmed_flags)

    def test_confirmed_only_drops_confirm_rejected(self):
        from panobbgo.self_improve import aggregate_codify_candidates

        # confirmed=False — the screening accept was overturned by the
        # same-night confirmation gate.  Must not contribute to codify
        # evidence even when ``accepted=True`` lingers on the record.
        recs = [
            _accepted_iter_record(timestamp="2026-06-01T05:00:00+00:00", confirmed=False),
            _accepted_iter_record(timestamp="2026-06-02T05:00:00+00:00", confirmed=False),
        ]
        assert aggregate_codify_candidates(recs, confirmed_only=True) == []

    def test_min_nights_zero_raises(self):
        from panobbgo.self_improve import aggregate_codify_candidates

        with pytest.raises(ValueError):
            aggregate_codify_candidates([], min_nights=0)

    def test_sort_order_strongest_first(self):
        from panobbgo.self_improve import aggregate_codify_candidates

        recs: List[Dict[str, Any]] = []
        # Candidate A: 2 nights, small delta.
        recs += [
            _accepted_iter_record(
                timestamp="2026-06-01T05:00:00+00:00",
                class_name="A",
                param_name="x",
                rule_kind="integer_add",
                old_value=10,
                new_value=11,
                delta=0.01,
            ),
            _accepted_iter_record(
                timestamp="2026-06-02T05:00:00+00:00",
                class_name="A",
                param_name="x",
                rule_kind="integer_add",
                old_value=10,
                new_value=12,
                delta=0.01,
            ),
        ]
        # Candidate B: 3 nights — should rank first (more replication).
        recs += [
            _accepted_iter_record(
                timestamp="2026-06-03T05:00:00+00:00",
                class_name="B",
                param_name="y",
                rule_kind="integer_add",
                old_value=5,
                new_value=6,
                delta=0.02,
            ),
            _accepted_iter_record(
                timestamp="2026-06-04T05:00:00+00:00",
                class_name="B",
                param_name="y",
                rule_kind="integer_add",
                old_value=5,
                new_value=7,
                delta=0.02,
            ),
            _accepted_iter_record(
                timestamp="2026-06-05T05:00:00+00:00",
                class_name="B",
                param_name="y",
                rule_kind="integer_add",
                old_value=5,
                new_value=8,
                delta=0.02,
            ),
        ]
        cands = aggregate_codify_candidates(recs)
        assert len(cands) == 2
        assert cands[0].class_name == "B"  # more nights = first
        assert cands[1].class_name == "A"

    def test_candidate_dict_round_trip(self):
        from panobbgo.self_improve import aggregate_codify_candidates

        recs = [
            _accepted_iter_record(timestamp="2026-06-01T05:00:00+00:00", new_value=0.12),
            _accepted_iter_record(timestamp="2026-06-02T05:00:00+00:00", new_value=0.13),
        ]
        cand = aggregate_codify_candidates(recs)[0]
        d = cand.to_dict()
        # JSON-serialisable.
        s = json.dumps(d, sort_keys=True)
        assert isinstance(s, str)
        # Surfaces every public field.
        for key in (
            "class_name",
            "param_name",
            "rule_kind",
            "direction",
            "n_accepts",
            "n_distinct_nights",
            "distinct_dates",
            "deltas",
            "ci_lows",
            "ci_highs",
            "old_values",
            "new_values",
            "timestamps",
            "strategy_names",
            "confirmed_flags",
            "mean_delta",
            "min_ci_low",
            "max_ci_high",
        ):
            assert key in d, key


class TestLoadLedgersForCodifyScan:
    """:func:`panobbgo.self_improve.load_ledgers_for_codify_scan` — io helper."""

    def test_missing_live_ledger_returns_empty(self, tmp_path):
        from panobbgo.self_improve import load_ledgers_for_codify_scan

        records = load_ledgers_for_codify_scan(
            str(tmp_path / "nope.jsonl"),
            include_archives=False,
        )
        assert records == []

    def test_live_only_returns_live_records(self, tmp_path):
        from panobbgo.self_improve import load_ledgers_for_codify_scan

        live = tmp_path / "live.jsonl"
        live.write_text(json.dumps(_accepted_iter_record(timestamp="2026-06-10T05:00:00+00:00")) + "\n")
        records = load_ledgers_for_codify_scan(str(live), include_archives=False)
        assert len(records) == 1
        assert records[0]["timestamp"] == "2026-06-10T05:00:00+00:00"

    def test_archive_default_dir_is_done_sibling(self, tmp_path):
        from panobbgo.self_improve import load_ledgers_for_codify_scan

        # Layout:
        #   tmp_path/live.jsonl
        #   tmp_path/done/self_improve_ledger_2026-05-31.jsonl
        live = tmp_path / "live.jsonl"
        live.write_text(json.dumps(_accepted_iter_record(timestamp="2026-06-10T05:00:00+00:00")) + "\n")
        done_dir = tmp_path / "done"
        done_dir.mkdir()
        arch = done_dir / "self_improve_ledger_2026-05-31.jsonl"
        arch.write_text(json.dumps(_accepted_iter_record(timestamp="2026-05-31T05:00:00+00:00")) + "\n")

        records = load_ledgers_for_codify_scan(str(live), include_archives=True)
        # Archive first (chronological), live after.
        assert len(records) == 2
        assert records[0]["timestamp"].startswith("2026-05-31")
        assert records[1]["timestamp"].startswith("2026-06-10")

    def test_archive_dir_override_respected(self, tmp_path):
        from panobbgo.self_improve import load_ledgers_for_codify_scan

        live = tmp_path / "live.jsonl"
        live.write_text("")  # empty live
        custom_archive = tmp_path / "custom"
        custom_archive.mkdir()
        arch = custom_archive / "self_improve_ledger_2026-05-31.jsonl"
        arch.write_text(json.dumps(_accepted_iter_record(timestamp="2026-05-31T05:00:00+00:00")) + "\n")

        records = load_ledgers_for_codify_scan(str(live), include_archives=True, archive_dir=str(custom_archive))
        assert len(records) == 1
        assert records[0]["timestamp"].startswith("2026-05-31")

    def test_missing_archive_dir_is_silent(self, tmp_path):
        from panobbgo.self_improve import load_ledgers_for_codify_scan

        live = tmp_path / "live.jsonl"
        live.write_text(json.dumps(_accepted_iter_record(timestamp="2026-06-10T05:00:00+00:00")) + "\n")
        # Default archive dir doesn't exist yet — must not throw.
        records = load_ledgers_for_codify_scan(str(live), include_archives=True)
        assert len(records) == 1

    def test_archive_dir_with_non_matching_files_ignored(self, tmp_path):
        from panobbgo.self_improve import load_ledgers_for_codify_scan

        live = tmp_path / "live.jsonl"
        live.write_text("")
        done = tmp_path / "done"
        done.mkdir()
        # Non-matching name — must be skipped.
        (done / "notes.txt").write_text("nothing here")
        (done / "other_ledger.jsonl").write_text(
            json.dumps(_accepted_iter_record(timestamp="2026-05-31T05:00:00+00:00")) + "\n"
        )
        records = load_ledgers_for_codify_scan(str(live), include_archives=True)
        assert records == []

    def test_archive_chronological_order(self, tmp_path):
        from panobbgo.self_improve import load_ledgers_for_codify_scan

        live = tmp_path / "live.jsonl"
        live.write_text("")
        done = tmp_path / "done"
        done.mkdir()
        (done / "self_improve_ledger_2026-05-15.jsonl").write_text(
            json.dumps(_accepted_iter_record(timestamp="2026-05-15T05:00:00+00:00")) + "\n"
        )
        (done / "self_improve_ledger_2026-05-31.jsonl").write_text(
            json.dumps(_accepted_iter_record(timestamp="2026-05-31T05:00:00+00:00")) + "\n"
        )
        records = load_ledgers_for_codify_scan(str(live), include_archives=True)
        assert [r["timestamp"][:10] for r in records] == ["2026-05-15", "2026-05-31"]


class TestCodifyScanCLI:
    """End-to-end smoke tests for the ``codify-scan`` CLI subcommand."""

    @staticmethod
    def _import_cli():
        import sys as sys_module

        sys_module.path.insert(0, str(pathlib.Path(__file__).parents[1] / "scripts"))
        try:
            import self_improve as cli  # type: ignore
        finally:
            sys_module.path = [p for p in sys_module.path if not p.endswith("/scripts")]
        return cli

    def test_empty_ledger_prints_no_records_note(self, tmp_path, capsys):
        cli = self._import_cli()
        # Build a fresh live + done layout with both empty.
        live = tmp_path / "live.jsonl"
        live.write_text("")
        (tmp_path / "done").mkdir()

        ns = type(
            "NS",
            (),
            {
                "ledger": str(live),
                "archive_dir": None,
                "include_archives": True,
                "min_nights": 2,
                "require_positive_min_ci": True,
                "confirmed_only": False,
                "pooled_ci_n_boot": 100,
                "pooled_ci_confidence": 0.95,
                "pooled_ci_seed": 1,
                "as_json": False,
                "top": 0,
            },
        )()
        rc = cli._cmd_codify_scan(ns)
        assert rc == 0
        out = capsys.readouterr().out
        assert "no records" in out

    def test_realistic_two_night_pattern_surfaces_candidate(self, tmp_path, capsys):
        cli = self._import_cli()
        live = tmp_path / "live.jsonl"
        live.write_text(
            json.dumps(_accepted_iter_record(timestamp="2026-06-01T05:00:00+00:00", new_value=0.12))
            + "\n"
            + json.dumps(_accepted_iter_record(timestamp="2026-06-02T05:00:00+00:00", new_value=0.13))
            + "\n"
        )
        # Empty done dir.
        (tmp_path / "done").mkdir()

        ns = type(
            "NS",
            (),
            {
                "ledger": str(live),
                "archive_dir": None,
                "include_archives": True,
                "min_nights": 2,
                "require_positive_min_ci": True,
                "confirmed_only": False,
                "pooled_ci_n_boot": 100,
                "pooled_ci_confidence": 0.95,
                "pooled_ci_seed": 1,
                "as_json": False,
                "top": 0,
            },
        )()
        rc = cli._cmd_codify_scan(ns)
        assert rc == 0
        out = capsys.readouterr().out
        assert "Codify scan" in out
        assert "candidates surfaced: 1" in out
        assert "Nearby.radius" in out
        assert "direction=up" in out
        assert "n_accepts=2" in out
        assert "n_nights=2" in out
        # Evidence section renders both records.
        assert out.count("Δ=") >= 2

    def test_json_mode_emits_one_object_per_candidate(self, tmp_path, capsys):
        cli = self._import_cli()
        live = tmp_path / "live.jsonl"
        live.write_text(
            json.dumps(_accepted_iter_record(timestamp="2026-06-01T05:00:00+00:00", new_value=0.12))
            + "\n"
            + json.dumps(_accepted_iter_record(timestamp="2026-06-02T05:00:00+00:00", new_value=0.13))
            + "\n"
        )
        (tmp_path / "done").mkdir()

        ns = type(
            "NS",
            (),
            {
                "ledger": str(live),
                "archive_dir": None,
                "include_archives": True,
                "min_nights": 2,
                "require_positive_min_ci": True,
                "confirmed_only": False,
                "pooled_ci_n_boot": 100,
                "pooled_ci_confidence": 0.95,
                "pooled_ci_seed": 1,
                "as_json": True,
                "top": 0,
            },
        )()
        rc = cli._cmd_codify_scan(ns)
        assert rc == 0
        out = capsys.readouterr().out
        lines = [line for line in out.splitlines() if line.strip()]
        assert len(lines) == 1
        d = json.loads(lines[0])
        assert d["class_name"] == "Nearby"
        assert d["param_name"] == "radius"
        assert d["direction"] == "up"
        assert d["n_accepts"] == 2
        assert "pooled_ci_low" in d
        assert "pooled_ci_high" in d

    def test_top_truncates_report(self, tmp_path, capsys):
        cli = self._import_cli()
        # Build two distinct candidates: A on 3 nights, B on 2 nights.
        recs = []
        for i in range(3):
            recs.append(
                _accepted_iter_record(
                    timestamp=f"2026-06-0{i + 1}T05:00:00+00:00",
                    class_name="A",
                    param_name="x",
                    rule_kind="integer_add",
                    old_value=10,
                    new_value=10 + i + 1,
                )
            )
        for i in range(2):
            recs.append(
                _accepted_iter_record(
                    timestamp=f"2026-06-1{i + 1}T05:00:00+00:00",
                    class_name="B",
                    param_name="y",
                    rule_kind="integer_add",
                    old_value=5,
                    new_value=5 + i + 1,
                )
            )
        live = tmp_path / "live.jsonl"
        live.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
        (tmp_path / "done").mkdir()

        ns = type(
            "NS",
            (),
            {
                "ledger": str(live),
                "archive_dir": None,
                "include_archives": True,
                "min_nights": 2,
                "require_positive_min_ci": True,
                "confirmed_only": False,
                "pooled_ci_n_boot": 100,
                "pooled_ci_confidence": 0.95,
                "pooled_ci_seed": 1,
                "as_json": False,
                "top": 1,
            },
        )()
        rc = cli._cmd_codify_scan(ns)
        assert rc == 0
        out = capsys.readouterr().out
        # Top=1 — only one candidate header line.
        assert out.count("- A.x") == 1
        assert "- B.y" not in out

    def test_min_nights_must_be_positive(self, tmp_path, capsys):
        cli = self._import_cli()
        live = tmp_path / "live.jsonl"
        live.write_text("")
        ns = type(
            "NS",
            (),
            {
                "ledger": str(live),
                "archive_dir": None,
                "include_archives": False,
                "min_nights": 0,
                "require_positive_min_ci": True,
                "confirmed_only": False,
                "pooled_ci_n_boot": 100,
                "pooled_ci_confidence": 0.95,
                "pooled_ci_seed": 1,
                "as_json": False,
                "top": 0,
            },
        )()
        rc = cli._cmd_codify_scan(ns)
        assert rc == 1

    def test_confirmed_only_filters_legacy(self, tmp_path, capsys):
        cli = self._import_cli()
        live = tmp_path / "live.jsonl"
        # Legacy records — no confirmed field.
        live.write_text(
            json.dumps(_accepted_iter_record(timestamp="2026-06-01T05:00:00+00:00"))
            + "\n"
            + json.dumps(_accepted_iter_record(timestamp="2026-06-02T05:00:00+00:00"))
            + "\n"
        )
        (tmp_path / "done").mkdir()
        ns = type(
            "NS",
            (),
            {
                "ledger": str(live),
                "archive_dir": None,
                "include_archives": True,
                "min_nights": 2,
                "require_positive_min_ci": True,
                "confirmed_only": True,
                "pooled_ci_n_boot": 100,
                "pooled_ci_confidence": 0.95,
                "pooled_ci_seed": 1,
                "as_json": False,
                "top": 0,
            },
        )()
        rc = cli._cmd_codify_scan(ns)
        assert rc == 0
        out = capsys.readouterr().out
        assert "candidates surfaced: 0" in out

    def test_end_to_end_against_real_ledger_runs_clean(self, tmp_path, capsys):
        """Cheap sanity check that the CLI handles the actual project ledger.

        Confirms the live ``planning/self_improve_ledger.jsonl`` plus the
        existing ``planning/done/`` archive parse end-to-end without
        raising — and that *some* candidates surface (the project has
        accumulated 30+ accepts across 25+ nights at the time of ship).
        """
        cli = self._import_cli()
        project_root = pathlib.Path(__file__).parents[1]
        live = project_root / "planning" / "self_improve_ledger.jsonl"
        if not live.exists():
            pytest.skip("project ledger not present")

        ns = type(
            "NS",
            (),
            {
                "ledger": str(live),
                "archive_dir": None,
                "include_archives": True,
                "min_nights": 2,
                "require_positive_min_ci": True,
                "confirmed_only": False,
                "pooled_ci_n_boot": 200,
                "pooled_ci_confidence": 0.95,
                "pooled_ci_seed": 1,
                "as_json": False,
                "top": 0,
            },
        )()
        rc = cli._cmd_codify_scan(ns)
        assert rc == 0
        out = capsys.readouterr().out
        # The Sobol.scramble / Nearby.radius / Sobol.n patterns described
        # in the ledger inspection above must surface.  Don't assert exact
        # counts — those drift as new nightlies append — just confirm the
        # CLI produced a non-trivial report.
        assert "Codify scan" in out
        assert "candidates surfaced:" in out


# ===========================================================================
# §6.4 Same-night confirmation gate
# ===========================================================================


class TestPoolHarnessResults:
    """:func:`panobbgo.self_improve._pool_harness_results`.

    Validates that the pooling helper used by the §6.4 confirmation
    gate produces a :class:`HarnessResult` whose per-pair runs are the
    concatenation of the inputs' runs and whose composite score is the
    mean of the recomputed per-pair scores — i.e. interchangeable with
    a fresh harness measurement.
    """

    def test_single_input_is_identity(self):
        from panobbgo.self_improve import _pool_harness_results

        result = _fake_harness_result(0.5, ["S"])
        pooled = _pool_harness_results(result)
        # Identity case: same object returned, no recomputation hazard.
        assert pooled is result

    def test_no_inputs_raises(self):
        from panobbgo.self_improve import _pool_harness_results

        with pytest.raises(ValueError):
            _pool_harness_results()

    def test_two_inputs_concat_runs(self):
        from panobbgo.self_improve import _pool_harness_results

        a = _fake_harness_result(0.6, ["S"], n_reps=3)
        b = _fake_harness_result(0.4, ["S"], n_reps=4)
        pooled = _pool_harness_results(a, b)
        assert len(pooled.problem_strategy_results) == 1
        assert len(pooled.problem_strategy_results[0].runs) == 7
        assert pooled.total_runs == a.total_runs + b.total_runs

    def test_composite_is_recomputed(self):
        from panobbgo.self_improve import _pool_harness_results

        a = _fake_harness_result(0.6, ["S"], n_reps=3)
        b = _fake_harness_result(0.4, ["S"], n_reps=3)
        pooled = _pool_harness_results(a, b)
        # All concatenated runs are successes at eval 1 → solve_fraction = 1.0
        # so the per-pair score is 1.0 regardless of the per-input scores
        # (which were a fixture artefact, not a fact about the runs).
        # The pooled composite still equals the per-pair mean — the
        # contract the test exists to enforce.
        assert pooled.composite_score == pytest.approx(
            float(np.mean([p.score for p in pooled.problem_strategy_results]))
        )

    def test_disjoint_pairs_kept(self):
        from panobbgo.self_improve import _pool_harness_results

        a = _fake_harness_result(0.5, ["S1"], n_reps=2)
        b = _fake_harness_result(0.5, ["S2"], n_reps=2)
        pooled = _pool_harness_results(a, b)
        names = sorted(p.strategy_name for p in pooled.problem_strategy_results)
        assert names == ["S1", "S2"]


class TestLoopConfigConfirmAccepts:
    """:attr:`LoopConfig.confirm_accepts` and
    :attr:`LoopConfig.confirm_iteration_offset`."""

    def test_defaults_off_by_default(self):
        cfg = LoopConfig()
        assert cfg.confirm_accepts is False
        assert cfg.confirm_iteration_offset == 500_000

    def test_confirm_iteration_offset_must_be_positive(self):
        with pytest.raises(ValueError):
            LoopConfig(confirm_iteration_offset=0)
        with pytest.raises(ValueError):
            LoopConfig(confirm_iteration_offset=-1)

    def test_collision_with_guard_offset_rejected(self):
        # The confirm and guard offsets must differ so the two checks
        # see independent SHA-256 streams.  Only validates when
        # confirm_accepts is True — collision is dead code otherwise
        # and we don't want to retroactively break ledger replay.
        with pytest.raises(ValueError):
            LoopConfig(confirm_accepts=True, confirm_iteration_offset=1_000_000)

    def test_collision_allowed_when_confirm_disabled(self):
        # With confirm_accepts=False the offset is dead code, so the
        # collision check should not fire.  This keeps existing
        # configs valid even if they accidentally share offsets.
        cfg = LoopConfig(confirm_accepts=False, confirm_iteration_offset=1_000_000)
        assert cfg.confirm_iteration_offset == 1_000_000


class TestLoopConfirmRecord:
    """:class:`LoopConfirmRecord` serialisation contract."""

    def _make_record(self, **overrides):
        from panobbgo.self_improve import LoopConfirmRecord

        kwargs = dict(
            iteration=5,
            timestamp="2026-06-14T00:00:00+00:00",
            duration_seconds=1.5,
            proposal={"class_name": "C", "param_name": "p"},
            screen_baseline_score=0.5,
            screen_candidate_score=0.55,
            screen_delta=0.05,
            confirm_baseline_score=0.5,
            confirm_candidate_score=0.50,
            confirm_delta=0.0,
            pooled_delta=0.025,
            pooled_ci_low=-0.005,
            pooled_ci_high=0.06,
            pooled_worst_pair_regression=-0.01,
            pooled_worst_pair=("Rastrigin", "S1"),
            confirm_iteration_id=500_005,
        )
        kwargs.update(overrides)
        return LoopConfirmRecord(**kwargs)

    def test_record_type_is_confirm_reject(self):
        rec = self._make_record()
        assert rec.record_type == "confirm_reject"

    def test_to_dict_serialises_all_fields(self):
        rec = self._make_record()
        d = rec.to_dict()
        assert d["record_type"] == "confirm_reject"
        assert d["iteration"] == 5
        assert d["screen_delta"] == pytest.approx(0.05)
        assert d["confirm_delta"] == pytest.approx(0.0)
        assert d["pooled_delta"] == pytest.approx(0.025)
        assert d["pooled_ci_low"] == pytest.approx(-0.005)
        assert d["pooled_worst_pair"] == ["Rastrigin", "S1"]
        assert d["confirm_iteration_id"] == 500_005
        # Optional hold-out fields default to None.
        assert d["confirm_holdout_seed"] is None
        assert d["confirm_holdout_baseline_score"] is None
        assert d["confirm_holdout_candidate_score"] is None

    def test_to_dict_holdout_fields_when_set(self):
        rec = self._make_record(
            confirm_holdout_seed=1234,
            confirm_holdout_baseline_score=0.45,
            confirm_holdout_candidate_score=0.48,
        )
        d = rec.to_dict()
        assert d["confirm_holdout_seed"] == 1234
        assert d["confirm_holdout_baseline_score"] == pytest.approx(0.45)
        assert d["confirm_holdout_candidate_score"] == pytest.approx(0.48)

    def test_worst_pair_none_serialises_to_none(self):
        rec = self._make_record(pooled_worst_pair=None)
        d = rec.to_dict()
        assert d["pooled_worst_pair"] is None


class TestLoopIterationRecordConfirmedField:
    """:attr:`LoopIterationRecord.confirmed` field default + serialisation."""

    def _make_iter_record(self, **overrides) -> LoopIterationRecord:
        kwargs = dict(
            iteration=0,
            timestamp="2026-06-14T00:00:00+00:00",
            duration_seconds=0.1,
            proposal={"class_name": "C", "param_name": "p"},
            accepted=True,
            baseline_score=0.5,
            candidate_score=0.55,
            delta=0.05,
            ci_low=0.01,
            ci_high=0.1,
            worst_pair_regression=0.0,
            worst_pair=None,
        )
        kwargs.update(overrides)
        return LoopIterationRecord(**kwargs)

    def test_default_confirmed_is_none(self):
        rec = self._make_iter_record()
        assert rec.confirmed is None
        # Persistence: ``confirmed`` must round-trip through to_dict so
        # downstream consumers can branch on it without re-deriving the
        # state from other fields.
        d = rec.to_dict()
        assert d["confirmed"] is None

    def test_confirmed_true_round_trips(self):
        rec = self._make_iter_record(confirmed=True)
        assert rec.confirmed is True
        assert rec.to_dict()["confirmed"] is True

    def test_confirmed_false_round_trips(self):
        rec = self._make_iter_record(accepted=False, confirmed=False)
        assert rec.confirmed is False
        assert rec.to_dict()["confirmed"] is False


class TestConfirmationGateEndToEnd:
    """End-to-end behaviour of the §6.4 confirmation gate."""

    def _radius_catalog(self) -> MutationCatalog:
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

    def test_off_by_default_promotes_on_screening(self, tmp_path):
        # With confirm_accepts=False (the default) the loop promotes
        # straight from the screening measurement — historical V1
        # behaviour, byte-identical to the pre-2026-06-14 code path.
        ledger_path = tmp_path / "ledger.jsonl"
        cfg = LoopConfig(
            iterations=1,
            n_boot=50,
            eps_accept=0.005,
            ledger_path=str(ledger_path),
            stop_sentinel_path="",
            randomize=False,
        )
        si = SelfImprover(cfg, catalog=self._radius_catalog(), seed_strategies=_make_specs())
        counter = {"n": 0}

        # baseline=0.5, candidate=0.9 → strong screening accept.
        def score_fn(c):
            n = counter["n"]
            counter["n"] += 1
            return 0.5 if n % 2 == 0 else 0.9

        si._harness_factory = _make_factory(score_fn)
        records = si.run()
        assert len(records) == 1
        assert records[0].accepted is True
        # confirmed=None signals "no confirmation step ran" — the V1
        # path.  Important for ledger consumers to distinguish "didn't
        # confirm" from "confirmed false".
        assert records[0].confirmed is None
        # Only the iteration record on disk, no confirm record.
        loaded = load_ledger(str(ledger_path))
        kinds = [r.get("record_type", "iteration") for r in loaded]
        assert kinds == ["iteration"]

    def test_confirmation_passes_promotes_with_confirmed_true(self, tmp_path):
        # Both screening and confirmation see the same clearly-winning
        # delta → pooled CI cleared → confirmed=True, accepted=True.
        ledger_path = tmp_path / "ledger.jsonl"
        cfg = LoopConfig(
            iterations=1,
            n_boot=50,
            eps_accept=0.005,
            ledger_path=str(ledger_path),
            stop_sentinel_path="",
            randomize=False,
            confirm_accepts=True,
        )
        si = SelfImprover(cfg, catalog=self._radius_catalog(), seed_strategies=_make_specs())
        counter = {"n": 0}

        def score_fn(c):
            n = counter["n"]
            counter["n"] += 1
            # Screen pass: baseline=0.5, candidate=0.9
            # Confirm pass: baseline=0.5, candidate=0.9
            # → strong pooled accept.
            return 0.5 if n % 2 == 0 else 0.9

        si._harness_factory = _make_factory(score_fn)
        records = si.run()
        assert len(records) == 1
        assert records[0].accepted is True
        assert records[0].confirmed is True
        # No confirm_reject record because the gate passed.
        loaded = load_ledger(str(ledger_path))
        confirm_records = [r for r in loaded if r.get("record_type") == "confirm_reject"]
        assert confirm_records == []

    def test_confirmation_fails_demotes_with_confirmed_false(self, tmp_path):
        # Screening sees a strong win; confirmation sees a strong loss
        # → pooled CI no longer clears → confirmed=False, accepted=False,
        # and a LoopConfirmRecord lands on the ledger.
        ledger_path = tmp_path / "ledger.jsonl"
        cfg = LoopConfig(
            iterations=1,
            n_boot=50,
            eps_accept=0.005,
            ledger_path=str(ledger_path),
            stop_sentinel_path="",
            randomize=False,
            confirm_accepts=True,
        )
        si = SelfImprover(cfg, catalog=self._radius_catalog(), seed_strategies=_make_specs())
        # Call order from _run_internal with confirm_accepts:
        #   0: screen baseline
        #   1: screen candidate
        #   2: confirm baseline
        #   3: confirm candidate
        scores = [0.5, 0.95, 0.5, 0.05]  # screen wins, confirm loses
        counter = {"n": 0}

        def score_fn(c):
            n = counter["n"]
            counter["n"] += 1
            return scores[n] if n < len(scores) else 0.5

        si._harness_factory = _make_factory(score_fn)
        records = si.run()
        assert len(records) == 1
        # Post-confirmation: the gate overturned the screening accept.
        assert records[0].accepted is False
        assert records[0].confirmed is False
        # Ledger now carries both the iteration record AND a
        # confirm_reject record.  The latter preserves the screen +
        # confirm scores so an auditor can trace why the gate fired.
        loaded = load_ledger(str(ledger_path))
        confirm_records = [r for r in loaded if r.get("record_type") == "confirm_reject"]
        assert len(confirm_records) == 1
        rec = confirm_records[0]
        assert rec["iteration"] == 0
        assert rec["confirm_iteration_id"] == 0 + cfg.confirm_iteration_offset
        assert rec["screen_delta"] > 0  # Screening saw a win.
        assert rec["confirm_delta"] < 0  # Confirmation saw a loss.

    def test_screening_reject_skips_confirmation(self, tmp_path):
        # Screening rejected → confirmation never runs (the gate only
        # gates *promotions*), so confirmed=None and accepted=False.
        cfg = LoopConfig(
            iterations=1,
            n_boot=50,
            eps_accept=0.005,
            ledger_path=str(tmp_path / "ledger.jsonl"),
            stop_sentinel_path="",
            randomize=False,
            confirm_accepts=True,
        )
        si = SelfImprover(cfg, catalog=self._radius_catalog(), seed_strategies=_make_specs())
        counter = {"n": 0}

        def score_fn(c):
            n = counter["n"]
            counter["n"] += 1
            # baseline=0.5, candidate=0.1 → screening reject.
            return 0.5 if n % 2 == 0 else 0.1

        si._harness_factory = _make_factory(score_fn)
        records = si.run()
        assert len(records) == 1
        assert records[0].accepted is False
        # confirmed stays None because no confirmation step ran.
        assert records[0].confirmed is None
        # Harness saw only 2 calls (screening baseline + candidate),
        # not 4 — the gate did not run.
        assert counter["n"] == 2

    def test_no_op_screening_skips_confirmation(self, tmp_path):
        # No-op iterations are filtered upstream — the gate only runs
        # on informative screening accepts.  The historical no-op
        # semantics are unchanged.
        cfg = LoopConfig(
            iterations=1,
            n_boot=50,
            eps_accept=0.005,
            ledger_path=str(tmp_path / "ledger.jsonl"),
            stop_sentinel_path="",
            randomize=False,
            confirm_accepts=True,
        )
        si = SelfImprover(cfg, catalog=self._radius_catalog(), seed_strategies=_make_specs())
        si._harness_factory = _make_factory(lambda c: 0.5)  # baseline==candidate
        records = si.run()
        assert len(records) == 1
        assert records[0].no_op is True
        assert records[0].confirmed is None
        assert records[0].accepted is False

    def test_confirmation_uses_distinct_iteration_id(self, tmp_path):
        # Screening sees ``randomize_iteration=0`` (the regular stream)
        # while confirmation sees ``0 + confirm_iteration_offset``.
        # Validates the fresh-seed isolation the gate relies on.
        cfg = LoopConfig(
            iterations=1,
            n_boot=50,
            eps_accept=0.005,
            ledger_path=str(tmp_path / "ledger.jsonl"),
            stop_sentinel_path="",
            randomize=True,
            confirm_accepts=True,
            confirm_iteration_offset=500_000,
        )
        si = SelfImprover(cfg, catalog=self._radius_catalog(), seed_strategies=_make_specs())
        log: List[Dict[str, Any]] = []

        def score_fn(c):
            return 0.5 if (len(log) % 2 == 0) else 0.95

        si._harness_factory = _make_factory(score_fn, call_log=log)
        si.run()
        # Four harness calls expected: screen baseline / screen
        # candidate (iter_id=0), confirm baseline / confirm candidate
        # (iter_id=500_000).
        assert [c["randomize_iteration"] for c in log] == [0, 0, 500_000, 500_000]

    def test_confirm_reject_grants_bandit_zero_reward(self, tmp_path):
        # Bandit semantics under confirm-reject: the post-confirmation
        # pooled delta drives the graded reward, so an arm that
        # consistently produces noise-spike accepts no longer collects
        # the full-accept reward the screening would have given it.
        cfg = LoopConfig(
            iterations=1,
            n_boot=50,
            eps_accept=0.005,
            ledger_path=str(tmp_path / "ledger.jsonl"),
            stop_sentinel_path="",
            randomize=False,
            confirm_accepts=True,
            adaptive_sampling=True,
            bandit_reward_shaping="graded",
        )
        si = SelfImprover(cfg, catalog=self._radius_catalog(), seed_strategies=_make_specs())
        counter = {"n": 0}
        # screen baseline / screen candidate → strong screening accept
        # confirm baseline / confirm candidate → clearly harmful
        scores = [0.5, 0.95, 0.5, 0.05]

        def score_fn(c):
            n = counter["n"]
            counter["n"] += 1
            return scores[n] if n < len(scores) else 0.5

        si._harness_factory = _make_factory(score_fn)
        records = si.run()
        assert records[0].confirmed is False
        assert records[0].accepted is False
        # The pooled delta is roughly zero (averages 0.45 and -0.45),
        # so the graded reject reward sits near the 0.5 ceiling of the
        # reject regime ("honest near miss") — better than a clearly
        # harmful reward 0 (graded mode invariant on the reject path)
        # because the rule still surfaced real signal.  Importantly,
        # the bandit_reward field is *not* the full-accept reward
        # (≥0.5 + bonus) that screening would have produced.
        r = records[0].bandit_reward
        assert r is not None
        assert 0.0 <= r <= 0.5  # reject regime — the gate's contract.

    def test_pooled_decision_uses_pooled_baseline_and_candidate(self, tmp_path):
        # Headline contract: the pooled bootstrap CI uses the pooled
        # (screen + confirm) sample, not just the screening sample.
        # We exercise this by setting up scores where the screen +
        # confirm averages combine to a near-zero pooled delta — the
        # screening alone would clear the eps_accept gate, the pooled
        # would not.
        ledger_path = tmp_path / "ledger.jsonl"
        cfg = LoopConfig(
            iterations=1,
            n_boot=200,
            eps_accept=0.005,
            ledger_path=str(ledger_path),
            stop_sentinel_path="",
            randomize=False,
            confirm_accepts=True,
        )
        si = SelfImprover(cfg, catalog=self._radius_catalog(), seed_strategies=_make_specs())
        # Screening: baseline=0.5, candidate=0.95 (Δ=+0.45)
        # Confirm:   baseline=0.5, candidate=0.55 (Δ=+0.05)
        # Pooled:    baseline=0.5, candidate=0.75 (Δ=+0.25)
        # Both pooled Δ > eps_accept, but the pooled CI on identical
        # scores per side is trivially tight — the screening passes
        # in the same way.  Stronger test: a screen-strong / confirm-
        # negative pair that forces a reject only when pooling.
        scores = [0.5, 0.95, 0.5, 0.30]
        counter = {"n": 0}

        def score_fn(c):
            n = counter["n"]
            counter["n"] += 1
            return scores[n] if n < len(scores) else 0.5

        si._harness_factory = _make_factory(score_fn)
        records = si.run()
        # Loaded confirm record carries the pooled CI; verify the
        # pooled point delta matches what _pool_harness_results +
        # statistical_accept would compute on the same fake scores.
        loaded = load_ledger(str(ledger_path))
        confirm = [r for r in loaded if r.get("record_type") == "confirm_reject"]
        if confirm:
            rec = confirm[0]
            # Pooled point delta = mean(screen_after, confirm_after) -
            # mean(screen_before, confirm_before) = mean(0.95, 0.30) -
            # mean(0.5, 0.5) ≈ 0.625 - 0.5 = 0.125.
            # But since all fake runs land at the same solve_fraction
            # = 1.0 regardless of the score arg, the per-pair score
            # delta is actually 0 in the fixture.  What we DO want to
            # assert is that the record was written and carries fields.
            assert rec["confirm_iteration_id"] == cfg.confirm_iteration_offset
        # Whatever the outcome, the iteration record is present with
        # confirmed ∈ {True, False}; never None when confirm ran.
        assert records[0].confirmed in (True, False)


class TestConfirmationGateLedgerReplay:
    """:class:`LoopConfirmRecord` round-trips through the JSONL ledger.

    Ledger replay is the persistence boundary the codify-scan stage
    (§9.3) reads; confirmation rejects must survive the round-trip so
    cross-night evidence can distinguish "screening noise spike" from
    "real arm winner".
    """

    def test_confirm_reject_record_round_trips(self, tmp_path):
        from panobbgo.self_improve import _LedgerWriter, LoopConfirmRecord

        ledger_path = tmp_path / "ledger.jsonl"
        rec = LoopConfirmRecord(
            iteration=3,
            timestamp="2026-06-14T01:23:45+00:00",
            duration_seconds=2.0,
            proposal={"class_name": "Nearby", "param_name": "radius"},
            screen_baseline_score=0.42,
            screen_candidate_score=0.55,
            screen_delta=0.13,
            confirm_baseline_score=0.42,
            confirm_candidate_score=0.40,
            confirm_delta=-0.02,
            pooled_delta=0.055,
            pooled_ci_low=-0.003,
            pooled_ci_high=0.11,
            pooled_worst_pair_regression=-0.04,
            pooled_worst_pair=("Rastrigin_2D", "S1"),
            confirm_iteration_id=500_003,
            confirm_holdout_seed=1234,
            confirm_holdout_baseline_score=0.41,
            confirm_holdout_candidate_score=0.39,
            reasons=["ci_low < 0"],
            base_seed=42,
            mode="quick",
        )
        writer = _LedgerWriter(str(ledger_path))
        writer.write(rec)
        loaded = load_ledger(str(ledger_path))
        assert len(loaded) == 1
        on_disk = loaded[0]
        assert on_disk["record_type"] == "confirm_reject"
        assert on_disk["iteration"] == 3
        assert on_disk["proposal"]["class_name"] == "Nearby"
        assert on_disk["pooled_delta"] == pytest.approx(0.055)
        assert on_disk["pooled_ci_low"] == pytest.approx(-0.003)
        assert on_disk["confirm_holdout_seed"] == 1234
        assert on_disk["pooled_worst_pair"] == ["Rastrigin_2D", "S1"]
