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

This module provides a self-contained, reproducible benchmark system designed
for iterative improvement cycles where a coding agent runs benchmarks, makes
changes, and needs concrete quantitative feedback on whether improvements occurred.

Key features:

- **Single composite score** in [0, 1] for easy before/after comparison
- **Reproducible** via seeded random states per run
- **Convergence tracking** per run for diagnosing optimization behaviour
- **ERT (Expected Running Time)** computation – the standard BBOBench metric
- **JSON serialization** for agent-readable structured output
- **Three modes**: ``quick`` (fast iteration), ``standard``, ``full``
- **Comparison tool** to diff two result files

Typical agent workflow::

    # Run benchmarks, save results
    uv run python benchmark_harness.py run --quick --output before.json

    # ... make changes to panobbgo ...

    # Run again and compare
    uv run python benchmark_harness.py run --quick --output after.json
    uv run python benchmark_harness.py compare before.json after.json

Score interpretation:

- ``1.0`` – every run found the global optimum on the first evaluation (theoretical)
- ``0.7`` – good; strategies consistently find optima efficiently
- ``0.3`` – poor; strategies rarely find optima or only after many evaluations
- ``0.0`` – never found any optimum
"""

from __future__ import annotations

import hashlib
import json
import time
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from panobbgo.benchmark import ProblemSpec, StrategySpec


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
            known_optima=[
                {"x": [2 ** (-(2**i - 2) / 2**i) for i in range(1, 6)], "fx": 0.0}
            ],
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

    The adaptive strategy includes a :class:`~panobbgo.analyzers.sensitivity.Sensitivity`
    analyzer so that the sensitivity-aware :class:`~panobbgo.heuristics.nearby.Nearby`
    heuristic can scale perturbations by dimension importance once enough evaluations
    have been accumulated.

    CMA-ES is intentionally excluded from the quick strategy: with only 75 evaluations
    its covariance adaptation has too little data to converge, and the population overhead
    dilutes the budget for the faster local heuristics.  Use ``CMAES_Portfolio`` in
    standard or full mode for CMA-ES benchmarking.
    """
    from panobbgo.strategies import StrategyRoundRobin, StrategyRewarding
    from panobbgo.heuristics import Random, Nearby, NelderMead, Center
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
                (Random, {}),
                (Nearby, {"radius": 0.1, "axes": "all", "new": 3}),
                (Center, {}),
                (NelderMead, {}),
            ],
            analyzers=[(Sensitivity, {"update_interval": 20})],
        ),
    ]


def _make_standard_strategies() -> List[StrategySpec]:
    """Six strategies: baseline, adaptive, UCB bandit, Bayesian GP, CMA-ES portfolio, IPOP-CMA-ES.

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
        GaussianProcessHeuristic,
        CMAES,
    )
    from panobbgo.analyzers import Sensitivity, Restart

    quick = _make_quick_strategies()
    ucb = StrategySpec(
        name="UCB_Diverse",
        strategy_class=StrategyUCB,
        heuristics=[
            (Random, {}),
            (Nearby, {"radius": 0.1, "axes": "all", "new": 3}),
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
    return quick + [ucb, bayes_gp, cmaes_portfolio, ipop_cmaes]


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
            (Nearby, {"radius": 0.1, "axes": "all", "new": 3}),
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
    """

    mode: str = "quick"
    budget: Optional[int] = None
    reps: Optional[int] = None
    seed: int = 42
    problems: Optional[List[str]] = None
    strategies: Optional[List[str]] = None
    timeout_per_run: Optional[float] = 120.0

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
        return _clean(raw)

    def save(self, path: str) -> None:
        """Save results to a JSON file.

        Args:
            path: File path to write.
        """
        import pathlib

        pathlib.Path(path).write_text(
            json.dumps(self.to_dict(), indent=2), encoding="utf-8"
        )

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
        config = HarnessConfig(
            **{
                k: v
                for k, v in config_data.items()
                if k in HarnessConfig.__dataclass_fields__
            }
        )

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
                    func_distance=r["func_distance"]
                    if r["func_distance"] is not None
                    else float("inf"),
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
                ert=psr_data["ert"]
                if psr_data.get("ert") is not None
                else float("inf"),
                score=psr_data.get("score", 0.0),
                best_func_distance=(
                    psr_data["best_func_distance"]
                    if psr_data.get("best_func_distance") is not None
                    else float("inf")
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
        print(
            f"  {'Problem':<22} {'Strategy':<22} {'Score':>6} "
            f"{'SR':>5} {'ERT':>7} {'BestDist':>10}"
        )
        print("-" * width)

        for psr in self.problem_strategy_results:
            ert_str = f"{psr.ert:7.1f}" if psr.ert < float("inf") else "    inf"
            dist_str = (
                f"{psr.best_func_distance:10.4f}"
                if psr.best_func_distance < float("inf")
                else "       inf"
            )
            print(
                f"  {psr.problem_name:<22} {psr.strategy_name:<22}"
                f" {psr.score:6.3f} {psr.success_rate:5.2f} {ert_str} {dist_str}"
            )

        print(bar)
        print(
            f"  Total runs: {self.total_runs}  |  Wall time: {self.total_duration:.1f}s"
        )
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
            print(
                f"  Unchanged ({len(self.unchanged)}): "
                + ", ".join(f"{p}/{s}" for p, s, _, _ in self.unchanged)
            )
        if self.only_before:
            print(
                f"  Only in baseline ({len(self.only_before)}): "
                + ", ".join(f"{p}/{s}" for p, s, _ in self.only_before)
            )
        if self.only_after:
            print(
                f"  Only in candidate ({len(self.only_after)}): "
                + ", ".join(f"{p}/{s}" for p, s, _ in self.only_after)
            )
        print(bar)


def compare(
    before: HarnessResult,
    after: HarnessResult,
    eps: float = 0.01,
    label_before: str = "before",
    label_after: str = "after",
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
        (psr.problem_name, psr.strategy_name): psr.score
        for psr in before.problem_strategy_results
    }
    after_map: Dict[Tuple[str, str], float] = {
        (psr.problem_name, psr.strategy_name): psr.score
        for psr in after.problem_strategy_results
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
    rel_delta = (
        (delta / before.composite_score * 100.0)
        if before.composite_score > 0
        else float("inf")
    )

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
        """
        mode = self.config.mode
        if mode == "quick":
            specs = _make_quick_problems()
        elif mode == "standard":
            specs = _make_standard_problems()
        elif mode == "full":
            specs = _make_full_problems()
        else:
            raise ValueError(
                f"Unknown mode {mode!r}. Use 'quick', 'standard', or 'full'."
            )

        if self.config.problems:
            keep = set(self.config.problems)
            specs = [s for s in specs if s.name in keep]

        # Override budget from config
        budget = self.config.effective_budget()
        for spec in specs:
            spec.max_evaluations = budget

        return specs

    def get_strategies(self) -> List[StrategySpec]:
        """Return the list of :class:`~panobbgo.benchmark.StrategySpec` objects
        for the configured mode, filtered by ``config.strategies`` if set.
        """
        mode = self.config.mode
        if mode == "quick":
            specs = _make_quick_strategies()
        elif mode == "standard":
            specs = _make_standard_strategies()
        elif mode == "full":
            specs = _make_full_strategies()
        else:
            raise ValueError(f"Unknown mode {mode!r}.")

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
                    seed = self._derive_seed(
                        config.seed, prob_spec.name, strat_spec.name, rep
                    )
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
                        status = (
                            "OK" if record.error is None else f"ERR:{record.error[:30]}"
                        )
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
            # Create problem and strategy
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
            best_fx_raw = (
                best_result.fx
                if best_result is not None and best_result.fx is not None
                else float("inf")
            )
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
    def _build_convergence_from_arrays(
        fx_values: Any, f_opt: float
    ) -> List[ConvergencePoint]:
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
    def _extract_convergence(
        all_results: List[Any], f_opt: float
    ) -> List[ConvergencePoint]:
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
    return BenchmarkHarness(HarnessConfig(mode="standard", seed=seed)).run(
        verbose=verbose
    )


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
