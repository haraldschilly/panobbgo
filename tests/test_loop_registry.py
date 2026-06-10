#!/usr/bin/env python
# -*- coding: utf8 -*-
# Copyright 2012-2026 Harald Schilly <harald.schilly@gmail.com>
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
Tests for the self-improvement loop registry (§9.1 of
``planning/SELF_IMPROVEMENT_LOOP.md``).

The loop registry — :func:`panobbgo.harness._make_loop_strategies` —
ships the two ``quick`` specs plus one compact spec per rule-bearing
catalog family so the ~30 mutation rules in
:func:`panobbgo.self_improve.default_catalog` actually fire on the seed
specs (rather than staying dormant against a registry that only sets
``Sobol`` / ``Nearby`` / ``Sensitivity`` kwargs explicitly).
"""

from __future__ import annotations

import pytest

from panobbgo.harness import (
    BenchmarkHarness,
    HarnessConfig,
    _make_loop_strategies,
    _make_quick_strategies,
)


class TestLoopRegistryComposition:
    """Structural assertions on the loop registry itself."""

    def test_loop_registry_includes_quick_specs(self):
        """Loop registry carries forward both quick specs unchanged."""
        quick_names = {s.name for s in _make_quick_strategies()}
        loop_names = {s.name for s in _make_loop_strategies()}
        assert quick_names.issubset(loop_names), (
            f"Loop registry should include all quick specs; missing: {quick_names - loop_names}"
        )

    def test_loop_registry_has_compact_family_specs(self):
        """Loop registry adds one compact spec per rule-bearing family."""
        names = {s.name for s in _make_loop_strategies()}
        required = {
            "Loop_DE_Family",  # LSHADE / JSO / NLSHADE_RSP / NLSHADE_LBC / LSHADE_EpSin
            "Loop_PSO",  # PSO with all kwargs explicit
            "Loop_RegionUCB",  # RegionUCB with leaf-bandit kwargs explicit
            "Loop_LocalSearch",  # COBYQA + LBFGSB
            "Loop_Restart",  # Restart analyzer with kwargs explicit
        }
        missing = required - names
        assert not missing, f"Loop registry missing required family specs: {missing}"

    def test_loop_registry_size(self):
        """Loop registry has the documented 7 specs (2 quick + 5 family)."""
        specs = _make_loop_strategies()
        assert len(specs) == 7, f"Expected 7 specs in loop registry, got {len(specs)}"


class TestCatalogRuleCoverage:
    """The loop registry must exercise every kwarg mutation rule.

    These tests pin the §9.1 ship's primary contract: the loop registry
    activates *every* :class:`MutationRule` in :func:`default_catalog`
    (except :class:`StructuralMutationRule` which add classes rather
    than perturb existing kwargs).  If a new catalog rule lands on a
    class that the loop registry does not expose with an explicit
    kwarg, this test fails — the catalog freeze (§7.3) ensures the
    failure is loud rather than silent.
    """

    def test_every_kwarg_rule_fires_on_loop_registry(self):
        """All :class:`MutationRule` entries match at least one loop spec."""
        from panobbgo.self_improve import (
            MutationRule,
            StructuralMutationRule,
            _find_targets,
            default_catalog,
        )

        specs = _make_loop_strategies()
        catalog = default_catalog()
        dormant = []
        for rule in catalog.rules:
            if isinstance(rule, StructuralMutationRule):
                continue
            assert isinstance(rule, MutationRule)
            hits = _find_targets(specs, rule.strategy_pattern, rule.class_name, rule.param_name)
            if not hits:
                dormant.append(f"{rule.class_name}.{rule.param_name}[{rule.kind}]")
        assert not dormant, (
            "Loop registry leaves these catalog rules dormant — extend "
            "_make_loop_strategies() with an explicit kwarg, or remove the "
            f"rule:\n  {dormant}"
        )

    def test_quick_registry_covers_few_rules(self):
        """Sanity check: the quick registry is *intentionally* sparse.

        Before §9.1 the loop ran against the quick registry which sets
        only :class:`Sobol` / :class:`Nearby` / :class:`Sensitivity`
        kwargs explicitly.  If this baseline ever shifts, re-read the
        loop diagnosis in §2.4 and update the expected count.
        """
        from panobbgo.self_improve import (
            MutationRule,
            StructuralMutationRule,
            _find_targets,
            default_catalog,
        )

        specs = _make_quick_strategies()
        catalog = default_catalog()
        active = 0
        for rule in catalog.rules:
            if isinstance(rule, StructuralMutationRule):
                continue
            assert isinstance(rule, MutationRule)
            hits = _find_targets(specs, rule.strategy_pattern, rule.class_name, rule.param_name)
            if hits:
                active += 1
        # Today: Sobol.n / Sobol.scramble / Nearby.radius /
        # Sensitivity.update_interval = 4 rules.  The catalog freeze
        # (§7.3) keeps this at <=10 until the loop can resolve the
        # backlog; if you change this number, document why.
        assert active <= 10, (
            f"Quick registry should fire <= 10 kwarg rules (got {active});"
            " if you added explicit kwargs to a quick spec, raise the bound"
            " or move the spec to the loop registry."
        )


class TestHarnessConfigRegistryWiring:
    """``HarnessConfig.registry = 'loop'`` routes through ``get_strategies()``."""

    def test_default_registry_matches_quick_mode(self):
        cfg = HarnessConfig(mode="quick", registry="default")
        names = [s.name for s in BenchmarkHarness(cfg).get_strategies()]
        assert names == [s.name for s in _make_quick_strategies()]

    def test_loop_registry_ignores_mode(self):
        """The loop registry returns the same specs regardless of mode."""
        loop_specs = [s.name for s in _make_loop_strategies()]
        for mode in ("quick", "standard", "full"):
            cfg = HarnessConfig(mode=mode, registry="loop")
            names = [s.name for s in BenchmarkHarness(cfg).get_strategies()]
            assert names == loop_specs, f"registry='loop' should ignore mode={mode!r}"

    def test_unknown_registry_raises(self):
        cfg = HarnessConfig(mode="quick", registry="bogus")
        with pytest.raises(ValueError, match="Unknown registry"):
            BenchmarkHarness(cfg).get_strategies()

    def test_strategies_override_wins_over_registry(self):
        """When ``strategies_override`` is set the registry name is ignored."""
        override = _make_quick_strategies()[:1]
        cfg = HarnessConfig(mode="quick", registry="loop", strategies_override=override)
        names = [s.name for s in BenchmarkHarness(cfg).get_strategies()]
        assert names == [s.name for s in override]


class TestLoopConfigRegistryWiring:
    """``LoopConfig.registry`` propagates to the seed-strategy loader."""

    def test_loop_config_default_uses_mode_registry(self):
        from panobbgo.self_improve import LoopConfig, SelfImprover

        cfg = LoopConfig(iterations=0, mode="quick", registry="default")
        improver = SelfImprover(cfg)
        seeds = improver._load_seed_strategies()
        assert [s.name for s in seeds] == [s.name for s in _make_quick_strategies()]

    def test_loop_config_registry_loop_loads_loop_specs(self):
        from panobbgo.self_improve import LoopConfig, SelfImprover

        cfg = LoopConfig(iterations=0, mode="quick", registry="loop")
        improver = SelfImprover(cfg)
        seeds = improver._load_seed_strategies()
        assert [s.name for s in seeds] == [s.name for s in _make_loop_strategies()]

    def test_loop_config_validates_registry(self):
        from panobbgo.self_improve import LoopConfig

        with pytest.raises(ValueError, match="registry must be"):
            LoopConfig(registry="bogus")


class TestSelfImproveCliRegistryFlag:
    """The ``scripts/self_improve.py run --registry loop`` CLI surface."""

    @staticmethod
    def _parser():
        import pathlib
        import sys

        sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "scripts"))
        try:
            import self_improve as cli  # type: ignore

            return cli._build_parser()
        finally:
            sys.path = [p for p in sys.path if not p.endswith("/scripts")]

    def test_default_is_default_registry(self):
        parser = self._parser()
        args = parser.parse_args(["run", "--iterations", "0"])
        assert args.registry == "default"

    def test_loop_registry_accepted(self):
        parser = self._parser()
        args = parser.parse_args(["run", "--iterations", "0", "--registry", "loop"])
        assert args.registry == "loop"

    def test_unknown_registry_rejected(self):
        parser = self._parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["run", "--iterations", "0", "--registry", "bogus"])
