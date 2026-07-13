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
Benchmark Harness
=================

Reproducible benchmark harness for automated agent feedback loops.

This module is the *measurement substrate* of Panobbgo: it produces one
scalar, :attr:`HarnessResult.composite_score` ∈ ``[0, 1]``, that captures
how well the framework is doing on a fixed battery of problems under a
fixed set of strategies. Every code change that touches optimisation logic
is expected to be gated against this number.

For the full user-facing guide (interpretation, workflow, pitfalls,
roadmap) see ``doc/source/guide_benchmarking.rst``. For the planned
autonomous self-improvement loop that builds on top of this harness, see
``planning/SELF_IMPROVEMENT_LOOP.md``.

Key features
------------

- **Single composite score** in ``[0, 1]`` for easy before/after comparison.
- **Reproducible** via SHA-256-derived seeds per ``(problem, strategy, rep)``.
- **Convergence tracking** per run — all improvement events are recorded,
  so the first-hit evaluation is always recoverable.
- **ERT (Expected Running Time)** — the BBOB / COCO standard metric.
- **JSON serialisation** so results can be diffed by a coding agent or CI.
- **Three modes**: ``quick`` (fast iteration), ``standard``, ``full``.
- **Comparison tool** with per-pair classification of improvements and
  regressions.

Composite score — formula
-------------------------

For each run with budget ``B`` we extract the first-hit evaluation
``k* = first index where |f_best - f_opt| ≤ tolerance``.  The per-run
"solve fraction" is::

    s = 1 - (k* - 1) / B    if the run solved (k* ≤ B)
        0                   otherwise

- ``1`` means "solved on evaluation 1" (theoretical ceiling).
- ``1/B`` means "solved on the very last evaluation".
- Failure scores ``0`` — strictly worse than any successful run, even a
  last-evaluation one.

Per ``(problem, strategy)`` pair we average ``s`` across repetitions.
The final composite score is the unweighted mean over all pairs.

Score interpretation
--------------------

- ``1.0`` — theoretical best; every run solves at evaluation 1.
- ``0.7+`` — strong; consistently finds optima with budget left over.
- ``0.3`` — weak; rare successes, usually late in the budget.
- ``0.0`` — never found any optimum within tolerance.

**Stability contract**: the composite score formula is a stable contract.
Historical before/after comparisons depend on it.  Do not change it without
an architectural decision record.

Typical agent workflow
----------------------

::

    uv run python benchmark_harness.py run --quick --output before.json
    # ... make changes ...
    uv run python benchmark_harness.py run --quick --output after.json
    uv run python benchmark_harness.py compare before.json after.json

The exit code of ``compare --fail-on-regression`` is ``2`` if the
candidate scores worse, which makes this usable as a CI gate or an
automated revert trigger inside a self-improvement loop.
"""

from __future__ import annotations

import hashlib
import json
import time
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

from panobbgo.benchmark import ProblemSpec, StrategySpec

if TYPE_CHECKING:
    from panobbgo.harness_randomized import ProblemFamily


# ---------------------------------------------------------------------------
# Problem and strategy registry for the harness
# ---------------------------------------------------------------------------


def _make_quick_problems() -> List[ProblemSpec]:
    """Three problems: easy, medium, hard – fast to run."""
    from panobbgo.lib.classic import Rosenbrock, Rastrigin

    return [
        ProblemSpec(
            name="DeJong_2D",
            problem_class=_DejongProxy,
            dims=2,
            known_optima=[{"x": [0.0, 0.0], "fx": 0.0}],
            tolerance=0.1,
            max_evaluations=75,
        ),
        ProblemSpec(
            name="Rosenbrock_2D",
            problem_class=Rosenbrock,
            dims=2,
            known_optima=[{"x": [1.0, 1.0], "fx": 0.0}],
            tolerance=0.1,
            max_evaluations=75,
        ),
        ProblemSpec(
            name="Rastrigin_2D",
            problem_class=Rastrigin,
            dims=2,
            known_optima=[{"x": [0.0, 0.0], "fx": 0.0}],
            tolerance=0.1,
            max_evaluations=75,
        ),
    ]


def _make_standard_problems() -> List[ProblemSpec]:
    """Eight representative problems across easy/medium/hard."""
    from panobbgo.lib.classic import (
        Rosenbrock,
        Rastrigin,
        Ackley,
        Griewank,
        StyblinskiTang,
    )

    return [
        ProblemSpec(
            name="DeJong_2D",
            problem_class=_DejongProxy,
            dims=2,
            known_optima=[{"x": [0.0, 0.0], "fx": 0.0}],
            tolerance=0.1,
            max_evaluations=200,
        ),
        ProblemSpec(
            name="Rosenbrock_2D",
            problem_class=Rosenbrock,
            dims=2,
            known_optima=[{"x": [1.0, 1.0], "fx": 0.0}],
            tolerance=0.1,
            max_evaluations=200,
        ),
        ProblemSpec(
            name="Rastrigin_2D",
            problem_class=Rastrigin,
            dims=2,
            known_optima=[{"x": [0.0, 0.0], "fx": 0.0}],
            tolerance=0.1,
            max_evaluations=200,
        ),
        ProblemSpec(
            name="Ackley_2D",
            problem_class=Ackley,
            dims=2,
            known_optima=[{"x": [0.0, 0.0], "fx": 0.0}],
            tolerance=0.1,
            max_evaluations=200,
        ),
        ProblemSpec(
            name="Griewank_2D",
            problem_class=Griewank,
            dims=2,
            known_optima=[{"x": [0.0, 0.0], "fx": 0.0}],
            tolerance=0.1,
            max_evaluations=200,
        ),
        ProblemSpec(
            name="StyblinskiTang_2D",
            problem_class=StyblinskiTang,
            dims=2,
            known_optima=[{"x": [-2.903534, -2.903534], "fx": -78.33234}],
            tolerance=0.5,
            max_evaluations=200,
        ),
        ProblemSpec(
            name="Rosenbrock_5D",
            problem_class=Rosenbrock,
            dims=5,
            known_optima=[{"x": [1.0, 1.0, 1.0, 1.0, 1.0], "fx": 0.0}],
            tolerance=1.0,
            max_evaluations=200,
        ),
    ]


def _make_full_problems() -> List[ProblemSpec]:
    """Extended problem set including higher-dimensional variants."""
    from panobbgo.lib.classic import (
        Schwefel,
        DixonPrice,
        Zakharov,
    )

    base = _make_standard_problems()
    extra = [
        ProblemSpec(
            name="Schwefel_2D",
            problem_class=Schwefel,
            dims=2,
            known_optima=[{"x": [420.9687, 420.9687], "fx": 0.0}],
            tolerance=0.5,
            max_evaluations=500,
        ),
        ProblemSpec(
            name="DixonPrice_5D",
            problem_class=DixonPrice,
            dims=5,
            known_optima=[{"x": [2 ** (-(2**i - 2) / 2**i) for i in range(1, 6)], "fx": 0.0}],
            tolerance=0.5,
            max_evaluations=500,
        ),
        ProblemSpec(
            name="Zakharov_5D",
            problem_class=Zakharov,
            dims=5,
            known_optima=[{"x": [0.0] * 5, "fx": 0.0}],
            tolerance=0.1,
            max_evaluations=500,
        ),
    ]
    for spec in base:
        spec.max_evaluations = 500
    return base + extra


def _make_quick_strategies() -> List[StrategySpec]:
    """Minimal strategy set: one baseline + one adaptive.

    The adaptive strategy (``Rewarding_Diverse``) combines a Sobol' low-
    discrepancy initial design with a continuous-exploration ``Random`` arm,
    a sensitivity-aware ``Nearby`` perturbation, a ``Center`` anchor, and a
    ``NelderMead`` local refiner under the rewarding bandit.  The Sobol'
    block covers the box more uniformly than uniform i.i.d. sampling at the
    same evaluation count — the same rationale the standard-mode
    ``BayesOpt_Sobol`` / ``CMAES_Portfolio`` strategies use — giving the
    downstream local heuristics a higher-quality grid of "first looks" at
    the landscape.  ``scramble=False`` was selected by the self-improvement
    loop as a robust win over the literature-default ``True`` (three
    independent accepts with bootstrap CIs strictly above zero — see the
    ``planning/done/`` ledger archive); at the quick-mode budget the
    unscrambled Sobol' grid is already optimally space-filling and Owen
    scrambling adds variance the local heuristics then have to absorb.
    The :class:`~panobbgo.analyzers.sensitivity.Sensitivity` analyzer feeds
    dimension importances to ``Nearby`` once enough evaluations have
    accumulated; ``update_interval=25`` was selected by the self-
    improvement loop as a small but reproducible gain over the earlier
    ``20`` (see ``planning/self_improve_ledger.jsonl`` iter 6).

    CMA-ES is intentionally excluded from the quick strategy: with only 75 evaluations
    its covariance adaptation has too little data to converge, and the population overhead
    dilutes the budget for the faster local heuristics.  Use ``CMAES_Portfolio`` in
    standard or full mode for CMA-ES benchmarking.
    """
    from panobbgo.strategies import StrategyRoundRobin, StrategyRewarding
    from panobbgo.heuristics import Random, Nearby, NelderMead, Center, Sobol
    from panobbgo.analyzers import Sensitivity

    return [
        StrategySpec(
            name="RoundRobin_Random",
            strategy_class=StrategyRoundRobin,
            heuristics=[(Random, {})],
        ),
        StrategySpec(
            name="Rewarding_Diverse",
            strategy_class=StrategyRewarding,
            heuristics=[
                # ``scramble=False`` codified 2026-05-31 from three independent
                # self-improvement loop accepts (deltas +0.022 / +0.051 / +0.032,
                # each with bootstrap-CI lower bound > 0 and no per-pair
                # regression — see planning/self_improve_ledger.jsonl iter=9 /
                # iter=15 / iter=17 in the 2026-05 ledger window).  At n=16 in
                # the quick-mode 2-D battery, the deterministic Sobol' grid is
                # already optimally space-filling; Owen scrambling perturbs
                # those grid points and slightly degrades the "first looks"
                # the downstream local heuristics build on.  The catalog rule
                # ``Sobol.scramble`` (kind=categorical_choice) keeps the
                # bandit free to flip back to ``True`` once enough fresh
                # evidence accumulates.
                (Sobol, {"n": 16, "scramble": False}),
                (Random, {}),
                # ``radius=0.124`` codified 2026-06-28 from nine independent
                # self-improvement loop accepts in the ``"up"`` direction
                # across eight distinct nights (2026-05-26 → 2026-06-18).
                # Median accepted ``new_value`` is ``0.123105`` (most
                # frequent: three accepts at exactly that value); the
                # shipped seed is rounded slightly outward to ``0.124`` so
                # the ``max(live) >= median(new_values)`` predicate in
                # :func:`panobbgo.self_improve._candidate_already_codified`
                # cleanly suppresses the now-redundant codify candidate
                # next night.  Pooled per-record CI ``[+0.0365, +0.0658]``
                # — every contributing accept cleared its own per-record
                # statistical-accept gate.  Pairs with the 2026-06-26
                # catalog-bound tightening ``(0.032, 0.313)`` — the
                # bandit now explores a productive ~2.5× window around
                # the new seed.  See the 2026-06-28 entry in
                # ``planning/SELF_IMPROVEMENT_LOG.md``.
                #
                # ``quadratic=True`` (2026-07-08) turns the local refinement
                # step curvature-aware: on each new best, when a local quadratic
                # fits the recent evaluated points well (weighted R² ≥ 0.8), the
                # first of the three emitted points is the trust-region Newton
                # minimiser of that model instead of an isotropic perturbation.
                # Isotropic perturbation is worst exactly under ill-conditioning,
                # which the randomized battery injects into *every* problem via
                # log-uniform diagonal scaling + Haar rotation — so a
                # curvature-aware step wins broadly there.  Paired randomized
                # A/B on this exact spec (quick registry, --randomize, reps 12,
                # iter 0): composite 0.0339 → 0.0612, statistical_accept ACCEPT
                # Δ=+0.0274 95% CI [+0.0057, +0.0521], worst-pair −0.0178.  Over
                # 20 randomized iterations × 2 base_seeds the mean Δ is ≈ +0.075
                # (18/20 iterations positive).  See the 2026-07-08 entry in
                # ``planning/SELF_IMPROVEMENT_LOG.md``.
                (Nearby, {"radius": 0.124, "axes": "all", "new": 3, "quadratic": True}),
                (Center, {}),
                (NelderMead, {}),
            ],
            analyzers=[(Sensitivity, {"update_interval": 25})],
        ),
    ]


def _make_standard_strategies() -> List[StrategySpec]:
    """Seven strategies: baseline, adaptive, region bandit, UCB bandit, Bayesian GP, CMA-ES portfolio, IPOP-CMA-ES.

    Rewarding_RegionUCB is ``Rewarding_Diverse`` plus the
    :class:`~panobbgo.heuristics.RegionUCB` arm (UCB1 allocation over
    Splitter leaves with in-leaf sampling), so the per-strategy score delta
    between the two isolates the effect of region-level allocation.
    Registered in standard mode (not quick) because at 75 evals the
    Splitter produces too few leaves for region allocation to matter
    (measured 2026-06-05: quick A/B with 15 reps was a tie; standard A/B
    with 10 reps gave +0.30 on StyblinskiTang_2D, -0.17 on Rosenbrock_2D,
    net +0.02 — the classic multimodal-win / unimodal-tax signature).

    BayesOpt_GP uses a Gaussian Process surrogate (Expected Improvement acquisition)
    paired with LatinHypercube initialization.  With 200 evaluations the GP model
    has enough data to build an accurate surrogate and guide search efficiently.

    CMAES_Portfolio pairs CMA-ES with LatinHypercube initialization and NelderMead
    local refinement inside a Rewarding strategy.  CMA-ES adapts its covariance
    matrix to the local geometry, providing strong performance on smooth functions.

    IPOP_CMAES combines IPOP-CMA-ES (Increasing Population restart) with the
    Restart analyzer so that when stagnation is detected the population doubles
    and the search restarts from a new diverse center.  This systematically
    escapes local optima and is the approach used by competition-winning solvers
    on the BBOB/COCO benchmark suite.
    """
    from panobbgo.strategies import StrategyUCB, StrategyRewarding
    from panobbgo.heuristics import (
        Random,
        Nearby,
        NelderMead,
        LatinHypercube,
        Sobol,
        GaussianProcessHeuristic,
        CMAES,
        Center,
        RegionUCB,
    )
    from panobbgo.analyzers import Sensitivity, Restart

    quick = _make_quick_strategies()
    region_ucb = StrategySpec(
        name="Rewarding_RegionUCB",
        strategy_class=StrategyRewarding,
        heuristics=[
            # Identical pool to ``Rewarding_Diverse`` plus the RegionUCB arm —
            # see the docstring above and panobbgo/heuristics/region_ucb.py.
            # ``radius=0.124`` matches the 2026-06-28 codify on
            # ``Rewarding_Diverse`` (sibling spec, same heuristic mix); see
            # the rationale comment there.
            (Sobol, {"n": 16, "scramble": False}),
            (Random, {}),
            # ``quadratic=True`` curvature-aware refinement — same codify as
            # ``Rewarding_Diverse`` (2026-07-08); statistical_accept ACCEPT on the
            # randomized battery.  See that spec above and the 2026-07-08 entry in
            # ``planning/SELF_IMPROVEMENT_LOG.md``.
            (Nearby, {"radius": 0.124, "axes": "all", "new": 3, "quadratic": True}),
            (Center, {}),
            (NelderMead, {}),
            # Explicit kwargs match the constructor defaults
            # (``ucb_c=1.0`` / ``gauss_fraction=0.5`` /
            # ``gauss_scale=0.25``) so RegionUCB construction is
            # byte-identical to the prior ``(RegionUCB, {})`` form;
            # the explicit dict makes the ``default_catalog``
            # mutation rules (shipped as a follow-up to the
            # 2026-06-05 RegionUCB entry) immediately applicable to
            # this seed spec rather than dormant.
            (RegionUCB, {"ucb_c": 1.0, "gauss_fraction": 0.5, "gauss_scale": 0.25}),
        ],
        analyzers=[(Sensitivity, {"update_interval": 25})],
    )
    ucb = StrategySpec(
        name="UCB_Diverse",
        strategy_class=StrategyUCB,
        heuristics=[
            (Random, {}),
            # ``radius=0.124`` matches the 2026-06-28 codify on
            # ``Rewarding_Diverse``; the Nearby refinement step plays the
            # same role here under the UCB bandit as under the rewarding
            # one.  See the rationale comment there.
            # ``quadratic=True`` curvature-aware refinement — same codify as
            # ``Rewarding_Diverse`` (2026-07-08); statistical_accept ACCEPT on the
            # randomized battery.  See that spec above and the 2026-07-08 entry in
            # ``planning/SELF_IMPROVEMENT_LOG.md``.
            (Nearby, {"radius": 0.124, "axes": "all", "new": 3, "quadratic": True}),
            (LatinHypercube, {"div": 4}),
            (NelderMead, {}),
        ],
        analyzers=[(Sensitivity, {"update_interval": 20})],
    )
    bayes_gp = StrategySpec(
        name="BayesOpt_GP",
        strategy_class=StrategyRewarding,
        heuristics=[
            (LatinHypercube, {"div": 4}),
            (GaussianProcessHeuristic, {"n_restarts": 5}),
            (Nearby, {"radius": 0.05, "axes": "all", "new": 3}),
            (NelderMead, {}),
        ],
        analyzers=[(Sensitivity, {"update_interval": 20})],
    )
    # Sobol' counterpart of BayesOpt_GP: low-discrepancy quasi-random initial design
    # plus the same GP / Nearby / NelderMead ensemble.  Sobol' samples the search
    # box more uniformly than LatinHypercube at the same evaluation count, giving
    # the GP surrogate a higher-quality "first look" at the landscape.  See
    # ``panobbgo.heuristics.sobol`` for references.
    bayes_sobol = StrategySpec(
        name="BayesOpt_Sobol",
        strategy_class=StrategyRewarding,
        heuristics=[
            (Sobol, {"n": 16, "scramble": True}),
            (GaussianProcessHeuristic, {"n_restarts": 5}),
            (Nearby, {"radius": 0.05, "axes": "all", "new": 3}),
            (NelderMead, {}),
        ],
        analyzers=[(Sensitivity, {"update_interval": 20})],
    )
    cmaes_portfolio = StrategySpec(
        name="CMAES_Portfolio",
        strategy_class=StrategyRewarding,
        heuristics=[
            (LatinHypercube, {"div": 4}),
            (CMAES, {"sigma0": 0.3}),
            (Nearby, {"radius": 0.05, "axes": "all", "new": 3}),
            (NelderMead, {}),
        ],
        analyzers=[(Sensitivity, {"update_interval": 20})],
    )
    ipop_cmaes = StrategySpec(
        name="IPOP_CMAES",
        strategy_class=StrategyRewarding,
        heuristics=[
            (LatinHypercube, {"div": 4}),
            (CMAES, {"sigma0": 0.3, "ipop_factor": 2.0}),
            (NelderMead, {}),
        ],
        # Restart analyzer: patience=20*dim, diverse centers to maximize distance
        # from previous restarts (better coverage of the search space)
        analyzers=[
            (
                Restart,
                {"patience": None, "restart_strategy": "diverse", "max_restarts": 5},
            ),
            (Sensitivity, {"update_interval": 20}),
        ],
    )
    return quick + [region_ucb, ucb, bayes_gp, bayes_sobol, cmaes_portfolio, ipop_cmaes]


def _make_full_strategies() -> List[StrategySpec]:
    """Full strategy set including Thompson Sampling, enhanced Bayesian GP, and CMA-ES+GP.

    BayesOpt_Enhanced pairs GP (EI acquisition, 10 restarts) with DifferentialEvolution
    for global exploration and NelderMead for local refinement.  The larger 500-evaluation
    budget lets the GP build a highly accurate surrogate model.

    CMAES_GP combines CMA-ES with the Gaussian Process heuristic inside a Rewarding
    strategy.  CMA-ES provides efficient local adaptation while GP suggests globally
    promising regions, and the bandit allocates budget according to performance.
    """
    from panobbgo.strategies import StrategyThompsonSampling, StrategyRewarding
    from panobbgo.heuristics import (
        Random,
        Nearby,
        NelderMead,
        LatinHypercube,
        GaussianProcessHeuristic,
        DifferentialEvolution,
        CMAES,
    )
    from panobbgo.analyzers import Sensitivity, Restart

    base = _make_standard_strategies()
    thompson = StrategySpec(
        name="Thompson_Diverse",
        strategy_class=StrategyThompsonSampling,
        heuristics=[
            (Random, {}),
            # ``radius=0.124`` matches the 2026-06-28 codify on
            # ``Rewarding_Diverse`` (same heuristic mix shape); see the
            # rationale comment there.
            # ``quadratic=True`` curvature-aware refinement — same codify as
            # ``Rewarding_Diverse`` (2026-07-08); statistical_accept ACCEPT on the
            # randomized battery.  See that spec above and the 2026-07-08 entry in
            # ``planning/SELF_IMPROVEMENT_LOG.md``.
            (Nearby, {"radius": 0.124, "axes": "all", "new": 3, "quadratic": True}),
            (LatinHypercube, {"div": 4}),
            (NelderMead, {}),
        ],
        analyzers=[(Sensitivity, {"update_interval": 20})],
    )
    bayes_enhanced = StrategySpec(
        name="BayesOpt_Enhanced",
        strategy_class=StrategyRewarding,
        heuristics=[
            (LatinHypercube, {"div": 4}),
            (GaussianProcessHeuristic, {"n_restarts": 10}),
            (DifferentialEvolution, {}),
            (NelderMead, {}),
        ],
        analyzers=[(Sensitivity, {"update_interval": 20})],
    )
    cmaes_gp = StrategySpec(
        name="CMAES_GP",
        strategy_class=StrategyRewarding,
        heuristics=[
            (LatinHypercube, {"div": 4}),
            (CMAES, {"sigma0": 0.3}),
            (GaussianProcessHeuristic, {"n_restarts": 5}),
            (NelderMead, {}),
        ],
        analyzers=[(Sensitivity, {"update_interval": 20})],
    )
    # True BIPOP-CMA-ES (Hansen 2009): alternates between large-population (IPOP)
    # and small-population (random small sigma) regimes.  The regime that has
    # consumed fewer evaluations is selected after every restart, balancing
    # exploitation and exploration.  This is the BBOB-2009 winning algorithm.
    bipop_cmaes = StrategySpec(
        name="BIPOP_CMAES",
        strategy_class=StrategyRewarding,
        heuristics=[
            (LatinHypercube, {"div": 4}),
            (CMAES, {"sigma0": 0.3, "restart_mode": "bipop"}),
            (NelderMead, {}),
        ],
        analyzers=[
            (
                Restart,
                {"patience": None, "restart_strategy": "diverse", "max_restarts": 10},
            ),
            (Sensitivity, {"update_interval": 20}),
        ],
    )
    return base + [thompson, bayes_enhanced, cmaes_gp, bipop_cmaes]


def _make_loop_strategies() -> List[StrategySpec]:
    """Dedicated registry for the self-improvement loop.

    Returns the two ``quick`` specs (``RoundRobin_Random`` and
    ``Rewarding_Diverse``) plus one *compact* spec per rule-bearing
    catalog family — Differential Evolution, Particle Swarm, RegionUCB,
    LBFGSB / COBYQA local optimizers, and the :class:`Restart` analyzer.
    Each new spec ships every tunable kwarg of its targeted classes
    explicitly at the constructor default so the :mod:`panobbgo.self_improve`
    catalog rules immediately apply (see the ``_find_targets`` "param
    already in kwargs" predicate — kwargs left at the constructor default
    are filtered out and the rule stays dormant).

    Motivation: §9.1 of ``planning/SELF_IMPROVEMENT_LOOP.md`` (V2 plan).
    The nightly loop runs in ``quick`` mode whose default registry covers
    only ``Sobol`` / ``Nearby`` / ``Sensitivity`` — three of the ~15
    classes the catalog has rules for.  Every ``LSHADE`` / ``JSO`` /
    ``NLSHADE_RSP`` / ``NLSHADE_LBC`` / ``LSHADE_EpSin`` / ``PSO`` /
    ``RegionUCB`` / ``COBYQA`` / ``LBFGSB`` / ``Restart`` rule shipped
    since mid-May 2026 sat dormant because the seed specs never set
    those kwargs explicitly.  This factory exists to exercise the
    dormant catalog (§2.4 "catalog ≫ registry mismatch" in the loop
    diagnosis) without touching the existing ``quick`` / ``standard`` /
    ``full`` registries.

    The new specs use compact populations (``NP_init = 15`` for
    population-based DE variants, ``NP = 15`` for PSO) so a single
    strategy with multiple heuristics still fits the ``quick`` 75-eval
    budget.  All values land inside the matching catalog rule's
    ``bounds`` so a mutation can move in either direction.

    Opt in via ``scripts/self_improve.py run --registry loop`` or by
    passing ``seed_strategies=_make_loop_strategies()`` directly to
    :class:`~panobbgo.self_improve.SelfImprover`.  The default
    self-improvement CLI invocations are byte-identical because
    ``--registry`` defaults to the historical mode-based selection.
    """
    from panobbgo.strategies import StrategyRewarding
    from panobbgo.heuristics import (
        Random,
        Nearby,
        NelderMead,
        Center,
        LatinHypercube,
        Sobol,
        CMAES,
        PSO,
        RegionUCB,
        LBFGSB,
        COBYQA,
    )
    from panobbgo.heuristics.lshade import LSHADE
    from panobbgo.heuristics.jso import JSO
    from panobbgo.heuristics.nl_shade_rsp import NLSHADE_RSP
    from panobbgo.heuristics.nl_shade_lbc import NLSHADE_LBC
    from panobbgo.heuristics.lshade_ep_sin import LSHADE_EpSin
    from panobbgo.analyzers import Sensitivity, Restart

    quick = _make_quick_strategies()

    # DE family: all five literature-best adaptive Differential Evolution
    # variants share a single Rewarding strategy.  The strategy-level
    # bandit allocates budget across them at runtime; the explicit kwargs
    # are tuned to the canonical literature defaults *and* sized down to
    # ``NP_init = 15`` so even at the quick-mode 75-eval budget every
    # heuristic can complete at least one full generation.  Every value
    # sits inside the matching catalog rule's bounds (``NP_init`` ∈
    # ``[10, 60]`` etc.) so the bandit can mutate in either direction.
    #
    # Per-heuristic kwarg coverage:
    # * ``LSHADE``     — NP_init / H / p_best / p_best_end / archive_factor /
    #                    F_schedule (six rules including two categorical)
    # * ``JSO``        — NP_init / p_best_max / H (three rules + the
    #                    `p_best_max` categorical regime arm)
    # * ``NLSHADE_RSP``— NP_init / k_rank / H / adaptive_archive (four
    #                    rules + the `k_rank` categorical regime arm)
    # * ``NLSHADE_LBC``— NP_init + lbc_regime (two rules: the
    #                    ``NP_init`` ``integer_add`` and the joint
    #                    ``lbc_regime`` ``categorical_choice``).
    #                    The composite categorical arm shipped
    #                    2026-06-24 replaced the five per-field LBC
    #                    float rules with one literature-motivated
    #                    joint search across the LBC Lehmer-mean
    #                    exponent / spread tuple.
    # * ``LSHADE_EpSin``— NP_init / mu_freq_init (two rules)
    #
    # ``p_best_end`` for LSHADE is set to half ``p_best`` (the jSO
    # iLSHADE-style schedule); ``F_schedule = "jso"`` opts the heuristic
    # into the Brest et al. 2017 three-phase asymmetric F-cap (one of the
    # four named regimes shipped 2026-06-23 — see
    # :data:`panobbgo.heuristics.lshade._F_SCHEDULE_REGIMES`).
    loop_de = StrategySpec(
        name="Loop_DE_Family",
        strategy_class=StrategyRewarding,
        heuristics=[
            (Random, {}),
            (
                LSHADE,
                {
                    "NP_init": 15,
                    "H": 6,
                    "p_best": 0.11,
                    "p_best_end": 0.055,
                    "archive_factor": 1.0,
                    "F_schedule": "jso",
                },
            ),
            (JSO, {"NP_init": 15, "H": 5, "p_best_max": 0.25}),
            (
                NLSHADE_RSP,
                {"NP_init": 15, "H": 5, "k_rank": 3.0, "adaptive_archive": True},
            ),
            (
                NLSHADE_LBC,
                {
                    "NP_init": 15,
                    "H": 5,
                    "lbc_regime": "cec2022",
                },
            ),
            (LSHADE_EpSin, {"NP_init": 15, "mu_freq_init": 0.5}),
            (NelderMead, {}),
        ],
        analyzers=[(Sensitivity, {"update_interval": 20})],
    )

    # PSO family: a single Rewarding spec with PSO at every tunable kwarg
    # explicit.  Covers PSO.NP / w / w_end / stagnation_threshold /
    # topology (five rules including the four-way topology categorical).
    # ``topology = "gbest"`` is the Kennedy-Eberhart 1995 default; the
    # categorical mutation rule can flip it to ``lbest`` / ``vonneumann``
    # / ``random``.  ``w_end = 0.4`` enables the Shi-Eberhart 1998 linear
    # inertia schedule (``w`` decays from ``0.7298`` to ``0.4`` over the
    # budget).  ``stagnation_threshold = 10`` opts in to the Clerc 2007 /
    # SPSO 2011 stochastic-K rebuild for the ``random`` topology — inert
    # on the seed ``gbest`` setting but pre-staged so the bandit can flip
    # both knobs simultaneously without re-adding the heuristic.
    loop_pso = StrategySpec(
        name="Loop_PSO",
        strategy_class=StrategyRewarding,
        heuristics=[
            (LatinHypercube, {"div": 4}),
            (
                PSO,
                {
                    "NP": 15,
                    "w": 0.7298,
                    "w_end": 0.4,
                    "stagnation_threshold": 10,
                    "topology": "gbest",
                },
            ),
            (NelderMead, {}),
        ],
        analyzers=[(Sensitivity, {"update_interval": 20})],
    )

    # RegionUCB: the three leaf-bandit dials shipped 2026-06-08 (ucb_c /
    # gauss_fraction / gauss_scale) all live on the same heuristic class.
    # Matches the seeded ``Rewarding_RegionUCB`` spec in standard mode
    # (the same explicit kwargs) so the catalog mutations stay applicable.
    loop_region_ucb = StrategySpec(
        name="Loop_RegionUCB",
        strategy_class=StrategyRewarding,
        heuristics=[
            # ``radius=0.124`` matches the 2026-06-28 codify on
            # ``Rewarding_Diverse`` (sibling spec — same heuristic mix
            # plus RegionUCB).  See the rationale comment there.
            (Sobol, {"n": 16, "scramble": False}),
            (Random, {}),
            # ``quadratic=True`` curvature-aware refinement — same codify as
            # ``Rewarding_Diverse`` (2026-07-08); statistical_accept ACCEPT on the
            # randomized battery.  See that spec above and the 2026-07-08 entry in
            # ``planning/SELF_IMPROVEMENT_LOG.md``.
            (Nearby, {"radius": 0.124, "axes": "all", "new": 3, "quadratic": True}),
            (Center, {}),
            (NelderMead, {}),
            (RegionUCB, {"ucb_c": 1.0, "gauss_fraction": 0.5, "gauss_scale": 0.25}),
        ],
        analyzers=[(Sensitivity, {"update_interval": 25})],
    )

    # Local-search pair: COBYQA (derivative-free trust region) and
    # LBFGSB (gradient-based quasi-Newton).  Both ship the kwargs the
    # catalog mutates explicitly.  COBYQA: initial_tr_radius /
    # final_tr_radius / scale (three rules including the binary
    # categorical).  LBFGSB: max_starts (the only LBFGSB rule today).
    # NelderMead stays as a cheap simplex fallback when the trust-region
    # / quasi-Newton arms exhaust their budget.
    #
    # The LatinHypercube seeder was *dropped* here on 2026-07-06 by the
    # codify pipeline: both local optimizers already start their first
    # descent from the box centre (COBYQA / LBFGSB) and multi-start from
    # fresh points thereafter, so the LHC "first looks" only diluted the
    # tight quick-mode budget without giving the refiners a better
    # anchor.  Two independent self-improvement ``drop_heuristic`` accepts
    # (2026-06-24 Δ=+0.0511, 2026-06-29 Δ=+0.0471; each bootstrap-CI lower
    # bound > 0) confirmed the win.  See the 2026-07-06 entry in
    # ``planning/SELF_IMPROVEMENT_LOG.md``.
    loop_local = StrategySpec(
        name="Loop_LocalSearch",
        strategy_class=StrategyRewarding,
        heuristics=[
            (
                COBYQA,
                {"initial_tr_radius": 0.1, "final_tr_radius": 1e-6, "scale": True},
            ),
            (LBFGSB, {"max_starts": 5}),
            (NelderMead, {}),
        ],
        analyzers=[(Sensitivity, {"update_interval": 20})],
    )

    # Restart analyzer: every kwarg the catalog tunes lives on the
    # :class:`Restart` analyzer — patience / max_restarts /
    # restart_strategy (three rules including the categorical regime
    # arm).  The heuristic mix is the lightweight "diverse" stack
    # (Random / Nearby / NelderMead) so the analyzer's stagnation /
    # restart cycle is the dominant signal.  Pairs naturally with CMAES
    # whose own ``sigma0`` kwarg is exposed by an existing catalog rule.
    loop_restart = StrategySpec(
        name="Loop_Restart",
        strategy_class=StrategyRewarding,
        heuristics=[
            # ``radius=0.124`` matches the 2026-06-28 codify on
            # ``Rewarding_Diverse``; the Nearby refinement step plays the
            # same role here (between Restart-driven jumps) as in the
            # baseline diverse stack.  See the rationale comment there.
            (LatinHypercube, {"div": 4}),
            (CMAES, {"sigma0": 0.3}),
            (Random, {}),
            # ``quadratic=True`` curvature-aware refinement — same codify as
            # ``Rewarding_Diverse`` (2026-07-08); statistical_accept ACCEPT on the
            # randomized battery.  See that spec above and the 2026-07-08 entry in
            # ``planning/SELF_IMPROVEMENT_LOG.md``.
            (Nearby, {"radius": 0.124, "axes": "all", "new": 3, "quadratic": True}),
            (NelderMead, {}),
        ],
        analyzers=[
            (
                Restart,
                {"patience": 20, "restart_strategy": "random", "max_restarts": 5},
            ),
            (Sensitivity, {"update_interval": 20}),
        ],
    )

    return quick + [loop_de, loop_pso, loop_region_ucb, loop_local, loop_restart]


# ---------------------------------------------------------------------------
# Simple DeJong sphere proxy (avoids importing a missing class by name)
# ---------------------------------------------------------------------------


class _DejongProxy:
    """Minimal wrapper so ProblemSpec.create_problem() works for DeJong."""

    def __new__(cls, dims: int = 2, **kwargs):  # type: ignore[override]
        from panobbgo.lib.classic import DeJong

        return DeJong(dims=dims)


# Patch Himmelblau into namespace for the _make_standard_problems function
try:
    from panobbgo.lib.classic import Himmelblau  # pyright: ignore[reportUnusedImport] # noqa: F401
except ImportError:
    Himmelblau = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Default budgets (max evaluations per run) for each mode.
_MODE_BUDGETS: Dict[str, int] = {
    "quick": 75,
    "standard": 200,
    "full": 500,
}

#: Default repetitions per (problem, strategy) pair for each mode.
_MODE_REPS: Dict[str, int] = {
    "quick": 3,
    "standard": 5,
    "full": 10,
}


@dataclass
class HarnessConfig:
    """
    Configuration for the benchmark harness.

    Args:
        mode: Preset mode controlling default problems, strategies, budget, and
            repetitions.  One of ``"quick"``, ``"standard"``, or ``"full"``.
        budget: Maximum number of evaluations per run.  ``None`` uses the mode
            default.
        reps: Number of independent repetitions per (problem, strategy) pair.
            ``None`` uses the mode default.
        seed: Base random seed.  Each (problem, strategy, rep) combination
            derives its own seed from this value for reproducibility.
        problems: List of problem names to include.  ``None`` uses the mode
            defaults (see :func:`get_problems`).
        strategies: List of strategy names to include.  ``None`` uses the mode
            defaults.
        timeout_per_run: Per-run wall-clock timeout in seconds.  Set to
            ``None`` to disable.
        randomize: If True, replace the fixed problem battery with a
            parametrically randomized one (Phase 3 of the self-improvement
            loop).  Each repetition draws a fresh translated / rotated /
            scaled / noisy instance from a :class:`~panobbgo.harness_randomized.ProblemFamily`.
            See :mod:`panobbgo.harness_randomized`.
        randomize_iteration: Iteration index mixed into the instance seed
            so that ``before`` and ``after`` runs within a single
            self-improvement iteration see *identical* sampled instances,
            while different iterations see different ones.
        strategies_override: Explicit list of :class:`StrategySpec` objects
            to run instead of the mode's default registry.  Used by the
            self-improvement loop driver to evaluate perturbed copies of the
            strategy registry without touching global state.  When set,
            :attr:`strategies` (the name filter) and :attr:`include_baselines`
            still apply, but no fallback to the built-in mode strategies
            happens.
        registry: Named strategy-registry override.  ``"default"`` (the
            historical behaviour) selects ``quick`` / ``standard`` /
            ``full`` strategies based on :attr:`mode`.  ``"loop"`` selects
            the catalog-exercising :func:`_make_loop_strategies` registry
            regardless of :attr:`mode` — used by the self-improvement loop
            so the rule-bearing DE / PSO / RegionUCB / LBFGSB / COBYQA /
            Restart catalog entries actually fire on the seed specs.  See
            §9.1 of ``planning/SELF_IMPROVEMENT_LOOP.md``.  Ignored when
            :attr:`strategies_override` is set.
    """

    mode: str = "quick"
    budget: Optional[int] = None
    reps: Optional[int] = None
    seed: int = 42
    problems: Optional[List[str]] = None
    strategies: Optional[List[str]] = None
    timeout_per_run: Optional[float] = 120.0
    #: If True, append the external baseline strategies (Random, SciPy DE,
    #: SciPy dual annealing) to the strategy list.  See
    #: :mod:`panobbgo.harness_baselines` for the adapters and Phase 2 of
    #: ``planning/SELF_IMPROVEMENT_LOOP.md`` for the motivation.
    include_baselines: bool = False
    #: If True, the problem battery is replaced with randomized families.
    #: See :mod:`panobbgo.harness_randomized` (Phase 3).
    randomize: bool = False
    #: Iteration index for the randomized sampler.  Constant across a
    #: ``before``/``after`` pair so instances line up.
    randomize_iteration: int = 0
    #: Opt-in extra randomized families appended to
    #: :func:`~panobbgo.harness_randomized.make_default_families` when
    #: :attr:`randomize` is set.  Used to probe regimes the frozen 2-D
    #: default battery cannot reach (e.g. the rotated higher-dimensional
    #: families from
    #: :func:`~panobbgo.harness_randomized.make_highdim_families`) *without*
    #: perturbing the historical ``composite_score`` contract, since the
    #: default families are untouched.  ``None`` (default) keeps the battery
    #: byte-identical to the pre-``extra_families`` behaviour.  Ignored when
    #: :attr:`randomize` is ``False``.
    extra_families: Optional[List["ProblemFamily"]] = None
    #: Caller-supplied strategy list that overrides the mode's default
    #: registry.  Used by :mod:`panobbgo.self_improve` so a loop iteration
    #: can evaluate a mutated copy of the strategy registry without
    #: monkey-patching the built-in factories.  ``None`` keeps the legacy
    #: behaviour (factories select based on ``mode``).
    strategies_override: Optional[List[StrategySpec]] = None
    #: Named strategy-registry override; ``"default"`` (historical) maps
    #: ``mode`` to ``quick`` / ``standard`` / ``full`` factories;
    #: ``"loop"`` selects :func:`_make_loop_strategies` regardless of
    #: ``mode``.  Ignored when :attr:`strategies_override` is set.
    registry: str = "default"

    def effective_budget(self) -> int:
        """Return the resolved evaluation budget."""
        return self.budget if self.budget is not None else _MODE_BUDGETS[self.mode]

    def effective_reps(self) -> int:
        """Return the resolved repetition count."""
        return self.reps if self.reps is not None else _MODE_REPS[self.mode]


# ---------------------------------------------------------------------------
# Result data structures
# ---------------------------------------------------------------------------


@dataclass
class ConvergencePoint:
    """A single improvement event in a convergence trace.

    Args:
        eval_idx: 1-indexed evaluation number when this improvement occurred.
        fx: Best function value seen up to this point.
        func_distance: Absolute distance ``|fx - f_opt|`` from the global optimum.
    """

    eval_idx: int
    fx: float
    func_distance: float


@dataclass
class RunRecord:
    """
    Complete record of a single optimization run.

    Args:
        problem_name: Name of the benchmark problem.
        problem_dim: Dimensionality of the problem.
        strategy_name: Name of the strategy used.
        rep: Zero-indexed repetition number.
        seed: Numpy random seed used for this run.
        budget: Maximum evaluations allowed.
        evaluations_used: Actual number of evaluations performed.
        best_fx: Best function value found.
        f_opt: True global optimum function value.
        func_distance: ``|best_fx - f_opt|``.
        tolerance: Acceptance tolerance for "success".
        success: Whether ``func_distance <= tolerance``.
        convergence: Ordered list of improvement events.
        heuristic_counts: Map from heuristic name to evaluation count.
        duration: Wall-clock time in seconds.
        error: Error message if the run failed, else ``None``.
    """

    problem_name: str
    problem_dim: int
    strategy_name: str
    rep: int
    seed: int
    budget: int
    evaluations_used: int
    best_fx: float
    f_opt: float
    func_distance: float
    tolerance: float
    success: bool
    convergence: List[ConvergencePoint]
    heuristic_counts: Dict[str, int]
    duration: float
    error: Optional[str] = None

    @property
    def first_success_eval(self) -> Optional[int]:
        """Return the evaluation index at which tolerance was first met, or None."""
        for pt in self.convergence:
            if pt.func_distance <= self.tolerance:
                return pt.eval_idx
        return None


@dataclass
class ProblemStrategyResult:
    """
    Aggregated results for all repetitions of a (problem, strategy) pair.

    Args:
        problem_name: Problem name.
        problem_dim: Dimensionality.
        strategy_name: Strategy name.
        f_opt: True global optimum value.
        tolerance: Success tolerance (same across all runs).
        budget: Evaluation budget per run.
        runs: Individual :class:`RunRecord` instances.
        success_rate: Fraction of runs where ``func_distance <= tolerance``.
        ert: Expected Running Time in evaluations (lower is better).
            ``inf`` when no run succeeded.
        score: Composite performance score in ``[0, 1]`` (higher is better).
        best_func_distance: Minimum ``func_distance`` across all runs.
        median_func_distance: Median ``func_distance`` across all runs.
    """

    problem_name: str
    problem_dim: int
    strategy_name: str
    f_opt: float
    tolerance: float
    budget: int
    runs: List[RunRecord]

    # Filled by compute_metrics()
    success_rate: float = 0.0
    ert: float = float("inf")
    score: float = 0.0
    best_func_distance: float = float("inf")
    median_func_distance: float = float("inf")

    def compute_metrics(self) -> None:
        """Compute derived metrics from the run records.  Called automatically by
        :class:`BenchmarkHarness`.
        """
        n = len(self.runs)
        if n == 0:
            return

        distances = [r.func_distance for r in self.runs]
        self.best_func_distance = float(np.min(distances))
        self.median_func_distance = float(np.median(distances))

        successes = [r for r in self.runs if r.success]
        self.success_rate = len(successes) / n

        # ERT: sum of evaluations-to-success / n_successes.
        # Failed runs contribute the full budget as penalty.
        total_evals = 0
        n_success = 0
        for run in self.runs:
            hit = run.first_success_eval
            if hit is not None:
                total_evals += hit
                n_success += 1
            else:
                total_evals += self.budget
        self.ert = total_evals / n_success if n_success > 0 else float("inf")

        # Composite score: mean over runs of the "hitting fraction".
        # For each run: solve_fraction = 1 - (hit_eval - 1) / budget   if solved
        #                                0                               otherwise
        # This is in (0, 1]: 1 = solved at eval 1, 1/budget = solved at last eval.
        # Failure always scores 0.0, so even solving at the last eval is
        # strictly better than not solving at all.
        solve_fractions = []
        for run in self.runs:
            hit = run.first_success_eval
            if hit is not None:
                frac = 1.0 - (hit - 1) / max(1, self.budget)
                solve_fractions.append(max(0.0, min(1.0, frac)))
            else:
                solve_fractions.append(0.0)
        self.score = float(np.mean(solve_fractions))


@dataclass
class HarnessResult:
    """
    Complete output of a :class:`BenchmarkHarness` run.

    Args:
        config: The configuration used to produce this result.
        timestamp: ISO-8601 timestamp (UTC) of when the run was completed.
        total_runs: Total number of individual optimization runs performed.
        total_duration: Total wall-clock time in seconds.
        problem_strategy_results: One :class:`ProblemStrategyResult` per
            (problem, strategy) pair.
        composite_score: Single scalar in ``[0, 1]``.  This is the primary
            metric for agent feedback – higher is better.
    """

    config: HarnessConfig
    timestamp: str
    total_runs: int
    total_duration: float
    problem_strategy_results: List[ProblemStrategyResult]
    composite_score: float

    # ------------------------------------------------------------------ #
    # Serialisation                                                        #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain Python dictionary (JSON-compatible)."""

        def _clean(obj: Any) -> Any:
            if isinstance(obj, float) and (obj == float("inf") or obj == float("-inf")):
                return None
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: _clean(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_clean(v) for v in obj]
            return obj

        raw = asdict(self)
        # ``strategies_override`` and ``extra_families`` carry live class /
        # dataclass objects (the latter's ``base_class`` is a raw ``type``)
        # that are not JSON-serialisable.  Both are runtime hooks used by
        # the self-improvement loop driver / opt-in extended battery, not
        # facts about the run that should be persisted; strip them before
        # serialising.
        if isinstance(raw.get("config"), dict):
            raw["config"].pop("strategies_override", None)
            raw["config"].pop("extra_families", None)
        return _clean(raw)

    def save(self, path: str) -> None:
        """Save results to a JSON file.

        Args:
            path: File path to write.
        """
        import pathlib

        pathlib.Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "HarnessResult":
        """Load a previously saved result file.

        Args:
            path: File path to read.

        Returns:
            A :class:`HarnessResult` instance restored from JSON.
        """
        import pathlib

        data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "HarnessResult":
        config_data = data["config"]
        config = HarnessConfig(**{k: v for k, v in config_data.items() if k in HarnessConfig.__dataclass_fields__})

        psr_list = []
        for psr_data in data.get("problem_strategy_results", []):
            runs = []
            for r in psr_data.get("runs", []):
                conv = [ConvergencePoint(**cp) for cp in r.get("convergence", [])]
                run = RunRecord(
                    problem_name=r["problem_name"],
                    problem_dim=r["problem_dim"],
                    strategy_name=r["strategy_name"],
                    rep=r["rep"],
                    seed=r["seed"],
                    budget=r["budget"],
                    evaluations_used=r["evaluations_used"],
                    best_fx=r["best_fx"],
                    f_opt=r["f_opt"],
                    func_distance=r["func_distance"] if r["func_distance"] is not None else float("inf"),
                    tolerance=r["tolerance"],
                    success=r["success"],
                    convergence=conv,
                    heuristic_counts=r.get("heuristic_counts", {}),
                    duration=r["duration"],
                    error=r.get("error"),
                )
                runs.append(run)

            psr = ProblemStrategyResult(
                problem_name=psr_data["problem_name"],
                problem_dim=psr_data["problem_dim"],
                strategy_name=psr_data["strategy_name"],
                f_opt=psr_data["f_opt"],
                tolerance=psr_data["tolerance"],
                budget=psr_data["budget"],
                runs=runs,
                success_rate=psr_data.get("success_rate", 0.0),
                ert=psr_data["ert"] if psr_data.get("ert") is not None else float("inf"),
                score=psr_data.get("score", 0.0),
                best_func_distance=(
                    psr_data["best_func_distance"] if psr_data.get("best_func_distance") is not None else float("inf")
                ),
                median_func_distance=(
                    psr_data["median_func_distance"]
                    if psr_data.get("median_func_distance") is not None
                    else float("inf")
                ),
            )
            psr_list.append(psr)

        return cls(
            config=config,
            timestamp=data.get("timestamp", ""),
            total_runs=data.get("total_runs", 0),
            total_duration=data.get("total_duration", 0.0),
            problem_strategy_results=psr_list,
            composite_score=data.get("composite_score", 0.0),
        )

    # ------------------------------------------------------------------ #
    # Human-readable output                                               #
    # ------------------------------------------------------------------ #

    def print_summary(self, width: int = 72) -> None:
        """Print a formatted summary table to stdout.

        Args:
            width: Character width for the output table.
        """
        bar = "=" * width
        print(bar)
        print(
            f"  HARNESS RESULTS  |  mode={self.config.mode}"
            f"  budget={self.config.effective_budget()}"
            f"  reps={self.config.effective_reps()}"
        )
        print(f"  Composite Score: {self.composite_score:.4f}   ({self.timestamp})")
        print(bar)

        # Header
        print(f"  {'Problem':<22} {'Strategy':<22} {'Score':>6} {'SR':>5} {'ERT':>7} {'BestDist':>10}")
        print("-" * width)

        for psr in self.problem_strategy_results:
            ert_str = f"{psr.ert:7.1f}" if psr.ert < float("inf") else "    inf"
            dist_str = f"{psr.best_func_distance:10.4f}" if psr.best_func_distance < float("inf") else "       inf"
            print(
                f"  {psr.problem_name:<22} {psr.strategy_name:<22}"
                f" {psr.score:6.3f} {psr.success_rate:5.2f} {ert_str} {dist_str}"
            )

        print(bar)
        print(f"  Total runs: {self.total_runs}  |  Wall time: {self.total_duration:.1f}s")
        print(bar)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


@dataclass
class ComparisonResult:
    """
    Side-by-side comparison of two :class:`HarnessResult` instances.

    Args:
        before: File path or label for the first (baseline) result.
        after: File path or label for the second (candidate) result.
        score_before: Composite score of the baseline.
        score_after: Composite score of the candidate.
        delta: Absolute score change (after - before).
        relative_delta: Relative change as a percentage.
        improved: Problem-strategy pairs where ``score_after > score_before + eps``.
        degraded: Problem-strategy pairs where ``score_after < score_before - eps``.
        unchanged: Problem-strategy pairs with no significant change.
        only_before: Pairs present only in the baseline (not compared).
        only_after: Pairs present only in the candidate (not compared).
    """

    before: str
    after: str
    score_before: float
    score_after: float
    delta: float
    relative_delta: float
    improved: List[Tuple[str, str, float, float]]  # (problem, strategy, before, after)
    degraded: List[Tuple[str, str, float, float]]
    unchanged: List[Tuple[str, str, float, float]]
    only_before: List[Tuple[str, str, float]] = field(default_factory=list)
    only_after: List[Tuple[str, str, float]] = field(default_factory=list)

    # Statistical test outputs (if requested)
    statistical: bool = False
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    pair_cis: Dict[Tuple[str, str], Tuple[float, float]] = field(default_factory=dict)
    accepted: Optional[bool] = None

    def print_summary(self, width: int = 72) -> None:
        """Print a formatted comparison to stdout."""
        bar = "=" * width
        arrow = "▲" if self.delta > 0 else ("▼" if self.delta < 0 else "—")
        sign = "+" if self.delta >= 0 else ""
        print(bar)
        print(f"  HARNESS COMPARISON  |  {self.before}  vs  {self.after}")
        print(
            f"  Score: {self.score_before:.4f} → {self.score_after:.4f}"
            f"  ({sign}{self.delta:.4f}, {sign}{self.relative_delta:.1f}%)  {arrow}"
        )
        print(bar)

        if self.improved:
            print(f"  Improved ({len(self.improved)}):")
            for prob, strat, b, a in self.improved:
                print(f"    {prob} / {strat}: {b:.3f} → {a:.3f} (+{a - b:.3f})")
        if self.degraded:
            print(f"  Degraded ({len(self.degraded)}):")
            for prob, strat, b, a in self.degraded:
                print(f"    {prob} / {strat}: {b:.3f} → {a:.3f} ({a - b:.3f})")
        if self.unchanged:
            print(f"  Unchanged ({len(self.unchanged)}): " + ", ".join(f"{p}/{s}" for p, s, _, _ in self.unchanged))
        if self.only_before:
            print(
                f"  Only in baseline ({len(self.only_before)}): "
                + ", ".join(f"{p}/{s}" for p, s, _ in self.only_before)
            )
        if self.only_after:
            print(
                f"  Only in candidate ({len(self.only_after)}): " + ", ".join(f"{p}/{s}" for p, s, _ in self.only_after)
            )

        if self.statistical:
            print(bar)
            print("  STATISTICAL ACCEPTANCE (95% Bootstrap CI)")
            if self.ci_lower is not None and self.ci_upper is not None:
                print(f"  Overall CI:    [{self.ci_lower:+.4f}, {self.ci_upper:+.4f}]")
            else:
                print("  Overall CI:    N/A (No matching pairs)")
            print(f"  Decision:      {'✅ ACCEPT' if self.accepted else '❌ REJECT'}")

            if self.pair_cis:
                print(f"  Per-pair CIs ({len(self.pair_cis)}):")
                for key, (ci_l, ci_u) in sorted(self.pair_cis.items()):
                    p, s = key
                    print(f"    {p} / {s}: [{ci_l:+.3f}, {ci_u:+.3f}]")

        print(bar)


def compare(
    before: HarnessResult,
    after: HarnessResult,
    eps: float = 0.01,
    label_before: str = "before",
    label_after: str = "after",
    statistical: bool = False,
    eps_accept: float = 0.005,
    eps_regress: float = -0.05,
) -> ComparisonResult:
    """
    Compare two :class:`HarnessResult` objects and identify regressions /
    improvements.

    Args:
        before: Baseline result.
        after: Candidate result.
        eps: Minimum absolute score change to be considered significant.
        label_before: Display label for the baseline.
        label_after: Display label for the candidate.

    Returns:
        A :class:`ComparisonResult` with per-pair breakdown.
    """
    # Index results by (problem, strategy)
    before_map: Dict[Tuple[str, str], float] = {
        (psr.problem_name, psr.strategy_name): psr.score for psr in before.problem_strategy_results
    }
    after_map: Dict[Tuple[str, str], float] = {
        (psr.problem_name, psr.strategy_name): psr.score for psr in after.problem_strategy_results
    }

    improved = []
    degraded = []
    unchanged = []

    # Only compare pairs present in both results to avoid false
    # regressions/improvements from differing problem/strategy sets.
    both_keys = sorted(set(before_map) & set(after_map))
    ob_keys = sorted(set(before_map) - set(after_map))
    oa_keys = sorted(set(after_map) - set(before_map))

    for key in both_keys:
        prob, strat = key
        b = before_map[key]
        a = after_map[key]
        if a - b > eps:
            improved.append((prob, strat, b, a))
        elif b - a > eps:
            degraded.append((prob, strat, b, a))
        else:
            unchanged.append((prob, strat, b, a))

    only_before = [(p, s, before_map[(p, s)]) for p, s in ob_keys]
    only_after = [(p, s, after_map[(p, s)]) for p, s in oa_keys]

    delta = after.composite_score - before.composite_score
    rel_delta = (delta / before.composite_score * 100.0) if before.composite_score > 0 else float("inf")

    ci_lower = None
    ci_upper = None
    pair_cis = {}
    accepted = None

    if statistical:
        # We need the per-run solve fractions for all matched pairs.
        # This requires matching the problem/strategy results.

        # Helper: Extract all run solve fractions for a PSR
        def get_fractions(psr):
            fracs = []
            for run in psr.runs:
                if run.first_success_eval is not None:
                    f = 1.0 - (run.first_success_eval - 1) / max(1, psr.budget)
                    fracs.append(max(0.0, min(1.0, f)))
                else:
                    fracs.append(0.0)
            return np.array(fracs)

        before_psrs = {(psr.problem_name, psr.strategy_name): psr for psr in before.problem_strategy_results}
        after_psrs = {(psr.problem_name, psr.strategy_name): psr for psr in after.problem_strategy_results}

        all_diffs = []
        pair_min_diff = float("inf")

        rng = np.random.default_rng(42)
        n_resamples = 10000
        alpha = 0.05

        for key in both_keys:
            b_psr = before_psrs[key]
            a_psr = after_psrs[key]

            # Note: We assume len(b_psr.runs) == len(a_psr.runs)
            b_fracs = get_fractions(b_psr)
            a_fracs = get_fractions(a_psr)

            # Pad with nan or truncate to match length if they differ, though they shouldn't
            min_len = min(len(b_fracs), len(a_fracs))
            b_fracs = b_fracs[:min_len]
            a_fracs = a_fracs[:min_len]

            diffs = a_fracs - b_fracs
            all_diffs.extend(diffs)

            # Pair-level bootstrap CI
            means = []
            for _ in range(n_resamples):
                resample = rng.choice(diffs, size=len(diffs), replace=True)
                means.append(np.mean(resample))

            p_lower = float(np.percentile(means, 100 * (alpha / 2)))
            p_upper = float(np.percentile(means, 100 * (1 - alpha / 2)))
            pair_cis[key] = (p_lower, p_upper)

            # We care about the actual mean difference for the acceptance rule minimum
            pair_mean_diff = float(np.mean(diffs))
            if pair_mean_diff < pair_min_diff:
                pair_min_diff = pair_mean_diff

        # Overall bootstrap CI over all paired runs
        if all_diffs:
            all_diffs_arr = np.array(all_diffs)
            means = []
            for _ in range(n_resamples):
                resample = rng.choice(all_diffs_arr, size=len(all_diffs_arr), replace=True)
                means.append(np.mean(resample))

            ci_lower = float(np.percentile(means, 100 * (alpha / 2)))
            ci_upper = float(np.percentile(means, 100 * (1 - alpha / 2)))

            # Statistical acceptance rule:
            # 1. Delta > eps_accept
            # 2. Lower bound of bootstrap 95% CI > 0
            # 3. No pair regresses more than eps_regress
            accepted = (delta > eps_accept) and (ci_lower > 0) and (pair_min_diff > eps_regress)
        else:
            accepted = False

    return ComparisonResult(
        before=label_before,
        after=label_after,
        score_before=before.composite_score,
        score_after=after.composite_score,
        delta=delta,
        relative_delta=rel_delta,
        improved=improved,
        degraded=degraded,
        unchanged=unchanged,
        only_before=only_before,
        only_after=only_after,
        statistical=statistical,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        pair_cis=pair_cis,
        accepted=accepted,
    )


# ---------------------------------------------------------------------------
# Statistical acceptance rule (bootstrap CI on composite-score delta)
# ---------------------------------------------------------------------------


def _solve_fractions(psr: ProblemStrategyResult) -> np.ndarray:
    """Return the per-run solve fractions for a (problem, strategy) pair.

    The solve fraction is the same quantity averaged by
    :meth:`ProblemStrategyResult.compute_metrics` to produce
    :attr:`ProblemStrategyResult.score`.  Exposed separately here so the
    statistical machinery can bootstrap over it.
    """
    fractions: List[float] = []
    budget = max(1, psr.budget)
    for run in psr.runs:
        hit = run.first_success_eval
        if hit is not None:
            frac = 1.0 - (hit - 1) / budget
            fractions.append(max(0.0, min(1.0, frac)))
        else:
            fractions.append(0.0)
    return np.asarray(fractions, dtype=np.float64)


@dataclass
class PairCI:
    """Bootstrap confidence interval for a single ``(problem, strategy)`` pair.

    Args:
        problem: Problem name.
        strategy: Strategy name.
        score_before: Baseline per-pair score (mean solve fraction).
        score_after: Candidate per-pair score.
        delta: ``score_after − score_before``.
        ci_low: Lower bound of the bootstrap CI on the delta.
        ci_high: Upper bound of the bootstrap CI on the delta.
        n_before: Number of baseline reps for this pair.
        n_after: Number of candidate reps for this pair.
    """

    problem: str
    strategy: str
    score_before: float
    score_after: float
    delta: float
    ci_low: float
    ci_high: float
    n_before: int
    n_after: int


@dataclass
class StatisticalDecision:
    """
    Result of :func:`statistical_accept` — a principled accept / reject decision
    on ``after`` vs ``before`` harness results.

    The decision rule follows ``planning/SELF_IMPROVEMENT_LOOP.md`` §6.2:

    - **Accept** iff *all* of:
        - ``delta > eps_accept`` (moved in the right direction beyond noise),
        - the lower bound of the bootstrap CI on the composite delta is ``> 0``
          (statistically plausible as a real improvement),
        - no individual pair regresses by more than ``eps_regress``.

    Args:
        accept: Final decision — ``True`` to keep the change, ``False`` to
            revert.
        delta: Composite score delta (``after − before``).
        ci_low: Lower bound of the bootstrap CI on the composite delta.
        ci_high: Upper bound of the bootstrap CI on the composite delta.
        confidence: Confidence level used (e.g. ``0.95``).
        n_boot: Number of bootstrap resamples used.
        eps_accept: Minimum composite delta required for acceptance.
        eps_regress: Maximum tolerated per-pair regression (as a positive
            number; the actual threshold on the per-pair delta is
            ``-eps_regress``).
        worst_pair_regression: The most negative per-pair delta observed,
            or ``0.0`` if no pair regressed.
        worst_pair: ``(problem, strategy)`` tuple identifying the pair in
            ``worst_pair_regression``, or ``None`` when no regression exists.
        reasons: Human-readable bullet points explaining the decision.
        per_pair: Per-pair confidence intervals.
        seed: Base RNG seed used for the bootstrap (for reproducibility).
        paired: ``True`` if the bootstrap used the paired (rep-aligned)
            scheme on at least one pair; ``False`` if every shared pair was
            sampled with the unpaired (independent before/after) scheme.
            See :func:`statistical_accept` for the auto-detection rule.
    """

    accept: bool
    delta: float
    ci_low: float
    ci_high: float
    confidence: float
    n_boot: int
    eps_accept: float
    eps_regress: float
    worst_pair_regression: float
    worst_pair: Optional[Tuple[str, str]]
    reasons: List[str]
    per_pair: List[PairCI]
    seed: int
    paired: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""
        return {
            "accept": self.accept,
            "delta": self.delta,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "confidence": self.confidence,
            "n_boot": self.n_boot,
            "eps_accept": self.eps_accept,
            "eps_regress": self.eps_regress,
            "worst_pair_regression": self.worst_pair_regression,
            "worst_pair": (list(self.worst_pair) if self.worst_pair is not None else None),
            "reasons": list(self.reasons),
            "seed": self.seed,
            "paired": self.paired,
            "per_pair": [
                {
                    "problem": p.problem,
                    "strategy": p.strategy,
                    "score_before": p.score_before,
                    "score_after": p.score_after,
                    "delta": p.delta,
                    "ci_low": p.ci_low,
                    "ci_high": p.ci_high,
                    "n_before": p.n_before,
                    "n_after": p.n_after,
                }
                for p in self.per_pair
            ],
        }

    def print_summary(self, width: int = 72) -> None:
        """Print a formatted decision report to stdout."""
        bar = "=" * width
        verdict = "ACCEPT" if self.accept else "REJECT"
        arrow = "▲" if self.delta > 0 else ("▼" if self.delta < 0 else "—")
        pct = int(self.confidence * 100)
        print(bar)
        print(
            f"  STATISTICAL DECISION: {verdict}  "
            f"Δ={self.delta:+.4f}  {pct}% CI=[{self.ci_low:+.4f}, {self.ci_high:+.4f}]"
            f"  {arrow}"
        )
        scheme = "paired" if self.paired else "unpaired"
        print(
            f"  eps_accept={self.eps_accept:.4f}  eps_regress={self.eps_regress:.4f}"
            f"  n_boot={self.n_boot}  bootstrap={scheme}"
        )
        if self.worst_pair is not None:
            prob, strat = self.worst_pair
            print(f"  Worst pair regression: {prob} / {strat}  Δ={self.worst_pair_regression:+.4f}")
        print(bar)
        if self.reasons:
            print("  Reasons:")
            for r in self.reasons:
                print(f"    • {r}")
            print(bar)


def statistical_accept(
    before: HarnessResult,
    after: HarnessResult,
    eps_accept: float = 0.005,
    eps_regress: float = 0.05,
    n_boot: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
    paired: Optional[bool] = None,
) -> StatisticalDecision:
    """Principled accept/reject decision for a candidate :class:`HarnessResult`.

    Implements the statistical acceptance rule from
    ``planning/SELF_IMPROVEMENT_LOOP.md`` §6.2.  For every
    ``(problem, strategy)`` pair present in both results, the per-run solve
    fractions are bootstrapped to produce a confidence interval on the
    mean-difference.  The composite delta and its CI are obtained by averaging
    the per-pair deltas across the *same* bootstrap resample indices, so the
    composite CI correctly accounts for dependence between pairs that share
    noise sources.

    The decision is ``accept`` iff **all** of:

    - ``delta > eps_accept`` — moved beyond measurement noise,
    - ``ci_low > 0`` — statistically plausible as a real improvement,
    - ``min_i delta_i > -eps_regress`` — no individual pair regresses
      catastrophically.

    Args:
        before: Baseline harness result.
        after: Candidate harness result.
        eps_accept: Minimum composite delta required for acceptance.
            Default ``0.005``.
        eps_regress: Maximum tolerated per-pair regression.  A pair whose
            delta is below ``-eps_regress`` blocks acceptance.
            Default ``0.05``.
        n_boot: Number of bootstrap resamples.  Default ``10_000``.
        confidence: Confidence level for the bootstrap CI.  Default ``0.95``.
        seed: Base RNG seed (derived per-pair via SHA-256 for independence).
            Default ``42``.
        paired: Bootstrap scheme for per-pair CIs.  Under
            ``--randomize`` (and any other run topology where reps are
            instance-aligned by index — same ``base_seed`` /
            ``randomize_iteration`` / ``rep`` slot on both sides) the
            ``i``-th run on each side is evaluated on the *same* problem
            instance, so paired sampling preserves the strong positive
            within-rep correlation between ``before`` and ``after``.  An
            unpaired bootstrap (the historical default) shuffles the two
            sides independently and inflates the CI proportionally to the
            instance-to-instance variance, which can leave genuine
            improvements indistinguishable from noise.

            Values:

            - ``None`` (default) — auto-detect: use paired sampling on
              every pair where ``n_before == n_after`` and at least one
              such pair exists; fall back to unpaired otherwise.  This is
              the right default for the randomized harness and a no-op
              for the asymmetric-rep edge cases the unpaired scheme was
              originally written to handle.
            - ``True`` — force paired sampling.  Pairs whose rep counts
              differ are truncated to ``min(n_before, n_after)`` reps so
              the index alignment stays valid; an empty intersection
              degenerates to a point estimate.
            - ``False`` — force the unpaired scheme (independent
              resamples on each side).  Useful when reps are *not*
              instance-aligned (e.g. comparing two ledgers built with
              different ``base_seed`` values).

    Returns:
        A populated :class:`StatisticalDecision` describing the verdict and
        carrying per-pair CIs so an agent can drill into the cause of a
        regression.  :attr:`StatisticalDecision.paired` records which
        scheme was actually used.
    """
    # Index each result by (problem, strategy) to find shared pairs.
    before_map: Dict[Tuple[str, str], ProblemStrategyResult] = {
        (psr.problem_name, psr.strategy_name): psr for psr in before.problem_strategy_results
    }
    after_map: Dict[Tuple[str, str], ProblemStrategyResult] = {
        (psr.problem_name, psr.strategy_name): psr for psr in after.problem_strategy_results
    }
    shared = sorted(set(before_map) & set(after_map))

    # Auto-detect paired scheme: any shared pair with matched rep counts is
    # eligible for paired sampling.  Pairs whose rep counts differ fall back
    # to the unpaired (independent-resample) scheme on a per-pair basis so the
    # mode-detection cannot crash on asymmetric-rep edge cases.
    if paired is None:
        paired_auto = False
        for key in shared:
            b_n = len(before_map[key].runs)
            a_n = len(after_map[key].runs)
            if b_n > 0 and b_n == a_n:
                paired_auto = True
                break
        paired_default = paired_auto
    else:
        paired_default = bool(paired)

    rng = np.random.default_rng(seed)

    per_pair: List[PairCI] = []
    # Joint bootstrap: for each pair compute a matrix of resampled deltas
    # (shape: n_boot,).  Composite delta resamples are obtained by averaging
    # across pairs at matching bootstrap indices.
    pair_delta_samples: List[np.ndarray] = []
    used_paired_anywhere = False

    for key in shared:
        prob, strat = key
        b_psr = before_map[key]
        a_psr = after_map[key]
        b_frac = _solve_fractions(b_psr)
        a_frac = _solve_fractions(a_psr)

        if b_frac.size == 0 or a_frac.size == 0:
            # No reps on one side — skip bootstrap, use point estimate only.
            point = float((a_frac.mean() if a_frac.size else 0.0) - (b_frac.mean() if b_frac.size else 0.0))
            per_pair.append(
                PairCI(
                    problem=prob,
                    strategy=strat,
                    score_before=float(b_psr.score),
                    score_after=float(a_psr.score),
                    delta=point,
                    ci_low=point,
                    ci_high=point,
                    n_before=int(b_frac.size),
                    n_after=int(a_frac.size),
                )
            )
            pair_delta_samples.append(np.full(n_boot, point, dtype=np.float64))
            continue

        nb, na = b_frac.size, a_frac.size
        # Decide the scheme for this pair: paired requires equal rep counts
        # (after auto-truncating to min when paired=True is forced).
        if paired_default and nb != na:
            if paired is True:
                # Forced paired with mismatched reps — truncate to the
                # common prefix so index alignment stays valid.  This is
                # the conservative interpretation; the alternative (raise)
                # would be hostile to the common case where one rep timed
                # out on one side but not the other.
                cut = min(nb, na)
                b_frac = b_frac[:cut]
                a_frac = a_frac[:cut]
                nb = na = cut
                use_paired = True
            else:
                # Auto-detect mode: this particular pair has mismatched
                # reps so fall back to unpaired sampling for it; other
                # pairs may still be paired.
                use_paired = False
        else:
            use_paired = paired_default

        if use_paired and nb > 0:
            # Paired bootstrap: resample one shared index vector and apply
            # to both sides so the within-rep correlation is preserved.
            # Mathematically equivalent to bootstrapping the per-rep delta
            # vector ``a_frac - b_frac``.
            d = a_frac - b_frac
            idx = rng.integers(0, nb, size=(n_boot, nb))
            deltas = d[idx].mean(axis=1)
            used_paired_anywhere = True
        else:
            # Unpaired bootstrap (historical scheme): independently resample
            # each side.  Correct when reps are *not* instance-aligned.
            idx_b = rng.integers(0, nb, size=(n_boot, nb))
            idx_a = rng.integers(0, na, size=(n_boot, na))
            means_b = b_frac[idx_b].mean(axis=1)
            means_a = a_frac[idx_a].mean(axis=1)
            deltas = means_a - means_b

        alpha = (1.0 - confidence) / 2.0
        lo = float(np.quantile(deltas, alpha))
        hi = float(np.quantile(deltas, 1.0 - alpha))
        point = float(a_frac.mean() - b_frac.mean())

        per_pair.append(
            PairCI(
                problem=prob,
                strategy=strat,
                score_before=float(b_psr.score),
                score_after=float(a_psr.score),
                delta=point,
                ci_low=lo,
                ci_high=hi,
                n_before=int(nb),
                n_after=int(na),
            )
        )
        pair_delta_samples.append(deltas)

    # Composite delta distribution: average per-pair deltas at each bootstrap
    # index.  If there are no shared pairs, fall back to the raw composite
    # delta with a degenerate CI.
    if not pair_delta_samples:
        raw_delta = float(after.composite_score - before.composite_score)
        ci_low = ci_high = raw_delta
        composite_deltas = np.array([raw_delta])
    else:
        composite_deltas = np.mean(np.vstack(pair_delta_samples), axis=0)
        alpha = (1.0 - confidence) / 2.0
        ci_low = float(np.quantile(composite_deltas, alpha))
        ci_high = float(np.quantile(composite_deltas, 1.0 - alpha))

    # Point estimate of the composite delta: mean of per-pair point deltas
    # across shared pairs (apples-to-apples).  If no shared pairs, use the raw
    # composite score delta.
    if per_pair:
        delta = float(np.mean([p.delta for p in per_pair]))
    else:
        delta = float(after.composite_score - before.composite_score)

    # Worst per-pair regression (most negative delta).  Zero when everything
    # improved or stayed flat.
    worst_pair_regression = 0.0
    worst_pair: Optional[Tuple[str, str]] = None
    for p in per_pair:
        if p.delta < worst_pair_regression:
            worst_pair_regression = p.delta
            worst_pair = (p.problem, p.strategy)

    # Decision rule.
    reasons: List[str] = []
    cond_shared = bool(per_pair)
    cond_delta = delta > eps_accept
    cond_ci = ci_low > 0.0
    cond_regress = worst_pair_regression > -eps_regress

    if not cond_shared:
        reasons.append(
            "no (problem, strategy) pairs are shared between before and after — cannot form a statistical decision"
        )
    if not cond_delta:
        reasons.append(f"composite delta {delta:+.4f} ≤ eps_accept {eps_accept:.4f}")
    if not cond_ci:
        reasons.append(f"lower CI bound {ci_low:+.4f} ≤ 0 — improvement not statistically distinguishable from noise")
    if not cond_regress:
        if worst_pair is not None:
            prob, strat = worst_pair
            reasons.append(
                f"pair {prob} / {strat} regressed by {worst_pair_regression:+.4f} (> eps_regress {eps_regress:.4f})"
            )
        else:  # pragma: no cover — defensive; cannot trigger with the rule above
            reasons.append(f"worst regression {worst_pair_regression:+.4f} exceeds eps_regress {eps_regress:.4f}")

    accept = cond_shared and cond_delta and cond_ci and cond_regress
    if accept and not reasons:
        reasons.append(
            f"composite delta {delta:+.4f} > eps_accept {eps_accept:.4f},"
            f" CI lower bound {ci_low:+.4f} > 0,"
            f" worst pair regression {worst_pair_regression:+.4f}"
            f" > -{eps_regress:.4f}"
        )

    return StatisticalDecision(
        accept=accept,
        delta=delta,
        ci_low=ci_low,
        ci_high=ci_high,
        confidence=confidence,
        n_boot=n_boot,
        eps_accept=eps_accept,
        eps_regress=eps_regress,
        worst_pair_regression=worst_pair_regression,
        worst_pair=worst_pair,
        reasons=reasons,
        per_pair=per_pair,
        seed=seed,
        paired=used_paired_anywhere,
    )


# ---------------------------------------------------------------------------
# Core harness
# ---------------------------------------------------------------------------


class BenchmarkHarness:
    """
    Reproducible benchmark harness for automated agent feedback loops.

    Usage::

        config = HarnessConfig(mode="quick")
        harness = BenchmarkHarness(config)
        result = harness.run()
        result.print_summary()
        result.save("results.json")

    Args:
        config: Harness configuration.  Defaults to ``HarnessConfig()``
            (quick mode).
    """

    def __init__(self, config: Optional[HarnessConfig] = None):
        self.config = config or HarnessConfig()

    # ------------------------------------------------------------------ #
    # Problem / strategy factories                                        #
    # ------------------------------------------------------------------ #

    def get_problems(self) -> List[ProblemSpec]:
        """Return the list of :class:`~panobbgo.benchmark.ProblemSpec` objects
        for the configured mode, filtered by ``config.problems`` if set.

        When :attr:`HarnessConfig.randomize` is set, the fixed mode battery
        is replaced with :class:`~panobbgo.harness_randomized.RandomizedProblemSpec`
        instances that sample a fresh translated / rotated / scaled / noisy
        instance per repetition.  See :mod:`panobbgo.harness_randomized`
        and Phase 3 of ``planning/SELF_IMPROVEMENT_LOOP.md``.

        When :attr:`HarnessConfig.extra_families` is set (and
        ``randomize`` is on), those families are appended to the default
        battery — used to probe regimes the frozen 2-D default battery
        cannot reach without perturbing the historical composite baseline.
        """
        # Validate the mode eagerly so that callers get an early error
        # even when randomize=True (which does not itself look at mode).
        mode = self.config.mode
        if mode not in _MODE_BUDGETS:
            raise ValueError(f"Unknown mode {mode!r}. Use 'quick', 'standard', or 'full'.")
        budget = self.config.effective_budget()

        if self.config.randomize:
            from panobbgo.harness_randomized import (
                make_default_families,
                make_randomized_specs,
            )

            families = make_default_families()
            if self.config.extra_families:
                families = families + list(self.config.extra_families)
            specs: List[ProblemSpec] = list(
                make_randomized_specs(
                    families,
                    iteration_id=self.config.randomize_iteration,
                    base_seed=self.config.seed,
                    max_evaluations=budget,
                )
            )
        else:
            if mode == "quick":
                specs = _make_quick_problems()
            elif mode == "standard":
                specs = _make_standard_problems()
            else:  # mode == "full" (already validated above)
                specs = _make_full_problems()

        if self.config.problems:
            keep = set(self.config.problems)
            specs = [s for s in specs if s.name in keep]

        # Override budget from config
        for spec in specs:
            spec.max_evaluations = budget

        return specs

    def get_strategies(self) -> List[StrategySpec]:
        """Return the list of :class:`~panobbgo.benchmark.StrategySpec` objects
        for the configured mode, filtered by ``config.strategies`` if set.

        When :attr:`HarnessConfig.include_baselines` is ``True``, the three
        external reference solvers from :mod:`panobbgo.harness_baselines`
        (``Baseline_Random``, ``Baseline_SciPyDE``, ``Baseline_SciPyAnneal``)
        are appended to the mode's strategy list.  This gives every
        ``HarnessResult`` an absolute-performance reference, not just the
        relative "better than previous Panobbgo" signal.

        When :attr:`HarnessConfig.strategies_override` is set, that list
        replaces the mode's defaults; the name filter and baseline
        appender still apply on top of it.
        """
        if self.config.strategies_override is not None:
            specs = list(self.config.strategies_override)
        elif self.config.registry == "loop":
            # The catalog-exercising loop registry (§9.1 of
            # ``planning/SELF_IMPROVEMENT_LOOP.md``).  Independent of
            # :attr:`mode` so the same seed specs measure under quick /
            # standard / full budgets.
            specs = _make_loop_strategies()
        elif self.config.registry == "default":
            mode = self.config.mode
            if mode == "quick":
                specs = _make_quick_strategies()
            elif mode == "standard":
                specs = _make_standard_strategies()
            elif mode == "full":
                specs = _make_full_strategies()
            else:
                raise ValueError(f"Unknown mode {mode!r}.")
        else:
            raise ValueError(f"Unknown registry {self.config.registry!r}; expected 'default' or 'loop'.")

        if self.config.include_baselines:
            from panobbgo.harness_baselines import make_baseline_strategies

            specs = specs + make_baseline_strategies()

        if self.config.strategies:
            keep = set(self.config.strategies)
            specs = [s for s in specs if s.name in keep]

        return specs

    # ------------------------------------------------------------------ #
    # Run                                                                 #
    # ------------------------------------------------------------------ #

    def run(self, verbose: bool = True) -> HarnessResult:
        """Execute all benchmark runs and return aggregated results.

        Args:
            verbose: If ``True``, print progress and a summary to stdout.

        Returns:
            A :class:`HarnessResult` with per-pair scores and the composite
            score.
        """
        config = self.config
        problems = self.get_problems()
        strategies = self.get_strategies()
        budget = config.effective_budget()
        reps = config.effective_reps()

        total_planned = len(problems) * len(strategies) * reps
        if verbose:
            print(
                f"Harness [{config.mode}] "
                f"{len(problems)} problems × {len(strategies)} strategies × {reps} reps"
                f" = {total_planned} runs  (budget={budget})"
            )

        harness_start = time.time()
        all_psr: List[ProblemStrategyResult] = []
        run_counter = 0

        for prob_spec in problems:
            for strat_spec in strategies:
                psr = ProblemStrategyResult(
                    problem_name=prob_spec.name,
                    problem_dim=prob_spec.dims,
                    strategy_name=strat_spec.name,
                    f_opt=prob_spec.known_optima[0]["fx"],
                    tolerance=prob_spec.tolerance,
                    budget=budget,
                    runs=[],
                )

                for rep in range(reps):
                    seed = self._derive_seed(config.seed, prob_spec.name, strat_spec.name, rep)
                    if verbose:
                        run_counter += 1
                        print(
                            f"  [{run_counter}/{total_planned}] "
                            f"{prob_spec.name} + {strat_spec.name} rep={rep} seed={seed}",
                            end="  ",
                            flush=True,
                        )

                    record = self._run_single(prob_spec, strat_spec, rep, seed, budget)
                    psr.runs.append(record)

                    if verbose:
                        status = "OK" if record.error is None else f"ERR:{record.error[:30]}"
                        suc = "✓" if record.success else "✗"
                        print(
                            f"{suc} fx={record.best_fx:.4g}"
                            f" dist={record.func_distance:.4g}"
                            f" evals={record.evaluations_used}"
                            f" [{status}]"
                        )

                psr.compute_metrics()
                all_psr.append(psr)

        total_duration = time.time() - harness_start
        total_runs = sum(len(psr.runs) for psr in all_psr)

        # Composite score: mean of per-pair scores
        scores = [psr.score for psr in all_psr]
        composite = float(np.mean(scores)) if scores else 0.0

        result = HarnessResult(
            config=config,
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            total_runs=total_runs,
            total_duration=total_duration,
            problem_strategy_results=all_psr,
            composite_score=composite,
        )

        if verbose:
            print()
            result.print_summary()

        return result

    # ------------------------------------------------------------------ #
    # Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _derive_seed(base: int, problem: str, strategy: str, rep: int) -> int:
        """Derive a deterministic seed from run parameters.

        Uses SHA-256 (not Python's ``hash()``) so that the result is stable
        across interpreter invocations regardless of ``PYTHONHASHSEED``.

        The result is always a non-negative 32-bit integer so it can be passed
        directly to ``numpy.random.seed``.
        """
        raw = hashlib.sha256(f"{base}:{problem}:{strategy}:{rep}".encode()).hexdigest()
        return int(raw, 16) % (2**32)

    def _run_single(
        self,
        prob_spec: ProblemSpec,
        strat_spec: StrategySpec,
        rep: int,
        seed: int,
        budget: int,
    ) -> RunRecord:
        """Execute one (problem, strategy, rep) run and return a :class:`RunRecord`.

        Runs the strategy directly (without going through :class:`BenchmarkSuite`)
        so that the full results DataFrame is accessible for convergence extraction.

        Args:
            prob_spec: Problem specification.
            strat_spec: Strategy specification.
            rep: Repetition index.
            seed: Numpy random seed for this run.
            budget: Maximum number of evaluations.

        Returns:
            A populated :class:`RunRecord`.
        """
        f_opt = prob_spec.known_optima[0]["fx"]
        tolerance = prob_spec.tolerance

        # Seed numpy for reproducibility (best-effort; threaded eval means
        # some non-determinism remains, but repeated runs are comparable).
        np.random.seed(seed)

        start = time.time()

        try:
            # Create problem and strategy.  RandomizedProblemSpec draws a
            # fresh transformed instance per rep; other specs return a
            # fixed instance.
            from panobbgo.harness_randomized import RandomizedProblemSpec

            if isinstance(prob_spec, RandomizedProblemSpec):
                problem = prob_spec.create_problem_for_rep(rep)
            else:
                problem = prob_spec.create_problem()
            strategy = strat_spec.create_strategy(problem)

            # Configure evaluation budget and method
            strategy.config.max_eval = budget
            strategy.config.evaluation_method = "threaded"

            # Run with optional wall-clock timeout.
            # We use a daemon thread + join(timeout) instead of SIGALRM
            # because SIGALRM can corrupt state in threaded evaluation workers.
            timeout = self.config.timeout_per_run
            run_error: Optional[Exception] = None

            def _run_strategy() -> None:
                nonlocal run_error
                try:
                    strategy.start()
                except Exception as e:
                    run_error = e

            runner = threading.Thread(target=_run_strategy, daemon=True)
            runner.start()
            runner.join(timeout=timeout)

            if runner.is_alive():
                # Timed out — signal the strategy to stop gracefully
                try:
                    strategy._stopped = True  # type: ignore[attr-defined]
                except Exception:
                    pass
                runner.join(timeout=5.0)
                raise TimeoutError(f"Run timed out after {timeout:.0f}s")

            if run_error is not None:
                raise run_error

            # Best result
            best_result = strategy.best
            best_fx_raw = best_result.fx if best_result is not None and best_result.fx is not None else float("inf")
            best_fx = float(best_fx_raw)
            func_distance = abs(best_fx - f_opt)
            success = func_distance <= tolerance

            # Access the results DataFrame directly for reliable column access.
            # The DataFrame uses MultiIndex columns: ("fx", 0), ("who", 0), etc.
            convergence: List[ConvergencePoint] = []
            heuristic_counts: Dict[str, int] = {}
            evaluations_used = 0

            if (
                hasattr(strategy, "results")
                and strategy.results is not None
                and hasattr(strategy.results, "results")
                and strategy.results.results is not None
                and not strategy.results.results.empty
            ):
                df = strategy.results.results
                evaluations_used = len(df)

                fx_values = self._get_column(df, "fx")
                who_values = self._get_column(df, "who")

                convergence = self._build_convergence_from_arrays(fx_values, f_opt)
                heuristic_counts = self._build_heuristic_counts_from_array(who_values)

            duration = time.time() - start

            return RunRecord(
                problem_name=prob_spec.name,
                problem_dim=prob_spec.dims,
                strategy_name=strat_spec.name,
                rep=rep,
                seed=seed,
                budget=budget,
                evaluations_used=evaluations_used,
                best_fx=best_fx,
                f_opt=f_opt,
                func_distance=func_distance,
                tolerance=tolerance,
                success=success,
                convergence=convergence,
                heuristic_counts=heuristic_counts,
                duration=duration,
                error=None,
            )

        except Exception as exc:
            duration = time.time() - start
            return RunRecord(
                problem_name=prob_spec.name,
                problem_dim=prob_spec.dims,
                strategy_name=strat_spec.name,
                rep=rep,
                seed=seed,
                budget=budget,
                evaluations_used=0,
                best_fx=float("inf"),
                f_opt=f_opt,
                func_distance=float("inf"),
                tolerance=tolerance,
                success=False,
                convergence=[],
                heuristic_counts={},
                duration=duration,
                error=str(exc),
            )

    @staticmethod
    def _get_column(df: Any, col_name: str) -> Any:
        """Extract a column from a potentially MultiIndex DataFrame.

        The results DataFrame uses MultiIndex tuples like ``("fx", 0)`` and
        ``("who", 0)``.  This helper tries the multi-level key first, then
        falls back to a plain string key.

        Args:
            df: A pandas DataFrame.
            col_name: Top-level column name (e.g. ``"fx"`` or ``"who"``).

        Returns:
            A numpy array of values (or empty array if column not found).
        """
        import pandas as pd

        try:
            # MultiIndex: ("col_name", 0) is the standard pattern
            if isinstance(df.columns, pd.MultiIndex):
                # Find the first sub-column index for this top-level name
                top_levels = df.columns.get_level_values(0)
                if col_name in top_levels:
                    return df[col_name].values.ravel()
        except Exception:
            pass

        try:
            return df[col_name].values.ravel()
        except (KeyError, Exception):
            return np.array([], dtype=object)

    @staticmethod
    def _build_convergence_from_arrays(fx_values: Any, f_opt: float) -> List[ConvergencePoint]:
        """Build a convergence trace from an ordered array of function values.

        Records an improvement event every time the running minimum decreases.

        Args:
            fx_values: Ordered array of function values (one per evaluation).
            f_opt: True global optimum value.

        Returns:
            List of :class:`ConvergencePoint` objects, one per improvement event.
        """
        trace: List[ConvergencePoint] = []
        best_fx = float("inf")

        for i, fx_raw in enumerate(fx_values):
            try:
                fx = float(fx_raw)
            except (TypeError, ValueError):
                continue

            if np.isnan(fx) or np.isinf(fx):
                continue

            if fx < best_fx:
                best_fx = fx
                trace.append(
                    ConvergencePoint(
                        eval_idx=i + 1,
                        fx=float(best_fx),
                        func_distance=abs(float(best_fx) - f_opt),
                    )
                )

        return trace

    @staticmethod
    def _build_heuristic_counts_from_array(who_values: Any) -> Dict[str, int]:
        """Count evaluations per heuristic from an array of heuristic names.

        Args:
            who_values: Array of heuristic name strings.

        Returns:
            Dict mapping heuristic name to evaluation count.
        """
        counts: Dict[str, int] = {}
        for who_raw in who_values:
            if who_raw is None:
                continue
            who = str(who_raw)
            counts[who] = counts.get(who, 0) + 1
        return counts

    # Keep backward-compatible static methods for external callers / tests
    @staticmethod
    def _extract_convergence(all_results: List[Any], f_opt: float) -> List[ConvergencePoint]:
        """Build a convergence trace from itertuples NamedTuples.

        .. deprecated::
            Prefer :meth:`_build_convergence_from_arrays` when a raw
            DataFrame is available.  This method exists for external callers
            that hold pre-extracted itertuples results.

        The NamedTuple field for ``fx`` depends on the pandas version and the
        MultiIndex column naming (``"fx_0"`` after flattening ``("fx", 0)``).

        Args:
            all_results: Ordered NamedTuple rows from ``itertuples()``.
            f_opt: True global optimum value.

        Returns:
            List of :class:`ConvergencePoint` objects.
        """
        trace: List[ConvergencePoint] = []
        best_fx = float("inf")

        for i, row in enumerate(all_results):
            # NamedTuple field name for ("fx", 0) is "fx_0" in pandas
            fx = None
            for attr in ("fx_0", "fx"):
                try:
                    val = getattr(row, attr)
                    if val is not None:
                        fx = float(val)
                        break
                except (AttributeError, TypeError, ValueError):
                    pass

            if fx is None:
                continue
            if np.isnan(fx) or np.isinf(fx):
                continue

            if fx < best_fx:
                best_fx = fx
                trace.append(
                    ConvergencePoint(
                        eval_idx=i + 1,
                        fx=float(best_fx),
                        func_distance=abs(float(best_fx) - f_opt),
                    )
                )

        return trace

    @staticmethod
    def _extract_heuristic_counts(all_results: List[Any]) -> Dict[str, int]:
        """Count evaluations per heuristic from itertuples NamedTuples.

        .. deprecated::
            Prefer :meth:`_build_heuristic_counts_from_array`.

        Args:
            all_results: Ordered NamedTuple rows from ``itertuples()``.

        Returns:
            Dict mapping heuristic name to evaluation count.
        """
        counts: Dict[str, int] = {}
        for row in all_results:
            who = None
            for attr in ("who_0", "who"):
                try:
                    val = getattr(row, attr)
                    if val is not None:
                        who = str(val)
                        break
                except AttributeError:
                    pass
            if who:
                counts[who] = counts.get(who, 0) + 1
        return counts


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


def run_quick(seed: int = 42, verbose: bool = True) -> HarnessResult:
    """Run a quick benchmark and return the result.

    This is the entry-point most useful for interactive experimentation.

    Args:
        seed: Base random seed.
        verbose: Print progress.

    Returns:
        :class:`HarnessResult` with a ``composite_score`` attribute.
    """
    return BenchmarkHarness(HarnessConfig(mode="quick", seed=seed)).run(verbose=verbose)


def run_standard(seed: int = 42, verbose: bool = True) -> HarnessResult:
    """Run a standard benchmark suite.  Takes longer than :func:`run_quick`.

    Args:
        seed: Base random seed.
        verbose: Print progress.
    """
    return BenchmarkHarness(HarnessConfig(mode="standard", seed=seed)).run(verbose=verbose)


def compute_ert(
    runs: List[RunRecord],
    tolerance: Optional[float] = None,
    budget: Optional[int] = None,
) -> float:
    """Compute Expected Running Time for a list of runs.

    This is a stand-alone helper for external analysis scripts.

    Args:
        runs: List of :class:`RunRecord` from a single (problem, strategy) pair.
        tolerance: Override the per-run tolerance.  ``None`` uses each run's own
            ``tolerance`` field.
        budget: Override the budget used as penalty for failed runs.  ``None``
            uses each run's ``budget`` field.

    Returns:
        ERT in number of evaluations, or ``inf`` if no run succeeded.
    """
    total = 0
    n_success = 0
    for run in runs:
        tol = tolerance if tolerance is not None else run.tolerance
        bud = budget if budget is not None else run.budget
        hit = run.first_success_eval
        if hit is not None and run.func_distance <= tol:
            total += hit
            n_success += 1
        else:
            total += bud
    return total / n_success if n_success > 0 else float("inf")
