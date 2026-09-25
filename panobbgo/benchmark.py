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
Benchmark specifications
========================

:class:`ProblemSpec` and :class:`StrategySpec` describe the problems and
strategy configurations the benchmark harnesses (:mod:`panobbgo.harness`,
:mod:`panobbgo.harness_ioh`, ...) run.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, replace

from panobbgo.lib import Problem
from panobbgo.core import StrategyBase


@dataclass
class ProblemSpec:
    """
    Specification for a benchmark problem including known optima and validation parameters.
    """

    name: str
    problem_class: type
    dims: int
    known_optima: List[Dict[str, Any]]  # List of known optima with 'x' and 'fx' keys
    tolerance: float = 1e-6  # Tolerance for validation
    max_evaluations: int = 1000
    problem_kwargs: Dict[str, Any] = field(default_factory=dict)

    def create_problem(self) -> Problem:
        """Create an instance of the problem."""
        # Some problems (like Himmelblau) don't take dims parameter
        kwargs = self.problem_kwargs.copy()
        if self.problem_class.__name__ not in ["Himmelblau"]:  # Add other fixed-dim problems here
            kwargs["dims"] = self.dims
        return self.problem_class(**kwargs)


#: Reserved keys in a :class:`StrategySpec` heuristic kwargs dict that gate
#: whether the arm is instantiated at all, based on the problem's dimension.
#: They are consumed by :meth:`StrategySpec.create_strategy` and never reach
#: the heuristic constructor.  Motivation: measured effects can be
#: regime-conditional in *sign* — the 2026-08-11 NLSHADE_LBC A/B (PR #298)
#: found d2 −0.0241 [−0.0401, −0.0080] vs d5 +0.0080 [+0.0007, +0.0154] for
#: the same arm — so the shippable form of such a gain is an arm that only
#: activates in the regime where it pays.
GATE_MIN_DIM_KEY = "gate_min_dim"
GATE_MAX_DIM_KEY = "gate_max_dim"


@dataclass
class StrategySpec:
    """
    Specification for a strategy configuration in benchmarks.

    Attributes:
        name: Human-readable strategy name shown in reports.
        strategy_class: The :class:`~panobbgo.core.StrategyBase` subclass to use.
        heuristics: List of ``(HeuristicClass, kwargs)`` pairs added via
            :meth:`~panobbgo.core.StrategyBase.add`.  Two reserved kwargs keys —
            ``"gate_min_dim"`` / ``"gate_max_dim"`` — are stripped before
            construction and instead gate the arm on the problem dimension:
            the heuristic is only added when
            ``gate_min_dim <= problem.dim <= gate_max_dim`` (either bound may
            be absent).
        analyzers: Optional list of ``(AnalyzerClass, kwargs)`` pairs added via
            :meth:`~panobbgo.core.StrategyBase.add_analyzer`.  ``Best`` and
            ``Convergence`` are always added by the strategy, ``Splitter`` whenever a
            module declares it (``requires_analyzers``); only supply *extra* analyzers here.
        config_overrides: Key/value pairs applied to ``strategy.config`` before the
            run starts.
        seed_name: Optional RNG identity for the harnesses' per-run seed
            derivation.  The harnesses hash a strategy label into every run's
            seed; by default that label is :attr:`name`, so two specs that
            differ *only* in their display name run on different RNG streams
            and an A/B between them carries the full run-to-run variance —
            a parameter that is never read still shows a nonzero delta.
            Setting ``seed_name`` to one constant across all variants of an
            arm makes them share the RNG stream per cell, so a dead parameter
            yields *exactly* zero delta and a live one shows only its own
            effect.  ``None`` (the default) keeps the historical behaviour of
            seeding from :attr:`name`.
    """

    name: str
    strategy_class: type
    heuristics: List[Tuple[type, Dict[str, Any]]]  # List of (HeuristicClass, kwargs) tuples
    analyzers: List[Tuple[type, Dict[str, Any]]] = field(default_factory=list)
    config_overrides: Dict[str, Any] = field(default_factory=dict)
    seed_name: Optional[str] = None

    @property
    def rng_identity(self) -> str:
        """Label the harnesses hash into each run's seed — ``seed_name or name``."""
        return self.seed_name or self.name

    def with_regime_class(self, noise_class: str) -> "StrategySpec":
        """Resolve a bare ``regime_gate="oracle"`` override to ``"oracle:<class>"``.

        The oracle form of :class:`~panobbgo.strategies.StrategyBlockBandit`'s
        regime gate takes the noise class as given.  A benchmark knows it —
        it built the noisy problem — but the spec is written once for every
        battery, so it says ``"oracle"`` and the harness fills the class in
        per run (:func:`panobbgo.harness_ioh.noise_class_of`).  Specs
        without that exact override are returned unchanged; a spec that
        already names a class keeps it (a deliberate override of the
        battery's tag).
        """
        if self.config_overrides.get("regime_gate") != "oracle":
            return self
        overrides = dict(self.config_overrides)
        overrides["regime_gate"] = "oracle:%s" % noise_class
        return replace(self, config_overrides=overrides)

    def create_strategy(
        self,
        problem: Problem,
        seed: Optional[int] = None,
        max_eval: Optional[int] = None,
    ) -> StrategyBase:
        """Create and configure a strategy instance.

        ``seed`` pins the strategy's master RNG (see
        :attr:`panobbgo.core.StrategyBase.seed`); ``None`` lets the strategy
        draw one from numpy's global state.

        ``max_eval`` is the run's evaluation budget.  It must be passed here
        rather than assigned to ``strategy.config`` afterwards: heuristics are
        constructed inside this method, and constructor-time logic that sizes
        itself from the horizon — notably
        :func:`panobbgo.heuristics.lshade._resolve_auto_np_init` for
        ``NP_init="auto"`` — would otherwise read the *default* ``max_eval``
        from :class:`panobbgo.config.Config` instead of the battery's budget.
        ``None`` leaves the budget to ``config_overrides`` / the config default.
        """
        # Config overrides go in through the constructor so they are in
        # effect *before* the evaluation backend is set up (e.g.
        # ``evaluation_method``, ``dask_n_workers``) and before any heuristic
        # is constructed (e.g. ``max_eval``).  Only StrategyBase
        # accepts them; the external baselines in
        # :mod:`panobbgo.harness_baselines` take no config kwargs, so their
        # overrides are written onto the config afterwards.  ``strategy_class``
        # may also be a plain factory callable (used to pre-bind arguments a
        # spec cannot express, e.g. ``StrategyPhased``'s phase list), which
        # takes the same path as a baseline.
        overrides = dict(self.config_overrides)
        if max_eval is not None:
            # The caller's budget wins over a spec-level override, matching the
            # harnesses' historical ``strategy.config.max_eval = budget``.
            overrides["max_eval"] = int(max_eval)
        if isinstance(self.strategy_class, type) and issubclass(self.strategy_class, StrategyBase):
            strategy = self.strategy_class(problem, parse_args=False, seed=seed, **overrides)
        else:
            strategy = self.strategy_class(problem, parse_args=False, seed=seed)
            for key, value in overrides.items():
                setattr(strategy.config, key, value)

        # Add heuristics (dimension-gated arms are skipped outside their regime)
        for heur_class, kwargs in self.heuristics:
            kwargs = dict(kwargs)
            min_dim = kwargs.pop(GATE_MIN_DIM_KEY, None)
            max_dim = kwargs.pop(GATE_MAX_DIM_KEY, None)
            if min_dim is not None and problem.dim < int(min_dim):
                continue
            if max_dim is not None and problem.dim > int(max_dim):
                continue
            strategy.add(heur_class, **kwargs)

        # Add optional extra analyzers (e.g. Sensitivity, Restart)
        for analyzer_class, kwargs in self.analyzers:
            strategy.add_analyzer(analyzer_class(strategy, **kwargs))

        return strategy
