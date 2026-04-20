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
External baseline strategies for the benchmark harness
======================================================

Phase 2 of :doc:`../planning/SELF_IMPROVEMENT_LOOP.md`: without external
reference solvers, the composite score only answers *"is Panobbgo better
than its previous self"*.  This module adds adapter strategies that plug
three well-known external optimizers into the same
:class:`~panobbgo.benchmark.StrategySpec` interface the harness already
uses:

- :class:`RandomSearchStrategy` — pure uniform random search (score floor).
- :class:`SciPyDEStrategy` — ``scipy.optimize.differential_evolution``
  (competitive, general-purpose global optimizer).
- :class:`SciPyAnnealStrategy` — ``scipy.optimize.dual_annealing``
  (competitive simulated-annealing hybrid).

All adapters respect a strict evaluation budget (``config.max_eval``),
record every evaluation into a MultiIndex results DataFrame compatible
with :class:`panobbgo.harness.BenchmarkHarness`, and honour the harness
per-run SHA-256-derived seed via ``np.random.seed`` / the optimizer's own
``seed`` argument.

Design notes
------------

* Baselines do **not** subclass :class:`~panobbgo.core.StrategyBase`.
  Instead they implement the subset of the strategy protocol the harness
  actually uses (``config.max_eval``, ``config.evaluation_method``,
  ``start()``, ``best``, ``results.results``) as a small duck-typed shim.
  This keeps the wrappers small, avoids spinning up the Dask/event-bus
  infrastructure which is meaningless for non-portfolio solvers, and makes
  budget enforcement deterministic.
* Budget enforcement is **hard**: the objective wrapper raises
  :class:`_BudgetExhausted` as soon as ``max_eval`` is reached.  The
  external solver's own stopping criterion may terminate earlier, but it
  can never overshoot, matching the harness contract.
* The results DataFrame uses the same MultiIndex columns
  ``(("fx", 0), ("who", 0), ("x", j), ...)`` that Panobbgo's own strategies
  emit, so :meth:`panobbgo.harness.BenchmarkHarness._get_column` and the
  convergence extractor work unchanged.

Usage
-----

The baselines are registered as additional strategies that can be included
via the ``--baselines`` CLI flag (see ``benchmark_harness.py``) or via the
``HarnessConfig(include_baselines=True)`` flag directly::

    uv run python benchmark_harness.py run --standard --baselines

With baselines in the picture, the harness output lets the user read
Panobbgo's score side-by-side with the external references — e.g. *"the
``CMAES_Portfolio`` strategy reaches 0.62 on the standard battery versus
0.58 for SciPy DE and 0.42 for pure Random search at equal budget"*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from panobbgo.benchmark import StrategySpec
from panobbgo.lib import Point, Problem, Result


# ---------------------------------------------------------------------------
# Utility: hard-budget objective wrapper
# ---------------------------------------------------------------------------


class _BudgetExhausted(Exception):
    """Raised by the objective wrapper once ``max_eval`` evaluations have
    been recorded.  Baseline adapters catch this to terminate cleanly."""


@dataclass
class _EvaluationLog:
    """Accumulates every ``(x, fx)`` pair the external solver evaluates.

    Used to build the harness-compatible results DataFrame and to compute
    the best-so-far record without leaning on the solver's own bookkeeping
    (which is implementation-specific and not always accessible).
    """

    who: str
    max_eval: int
    xs: List[np.ndarray] = field(default_factory=list)
    fxs: List[float] = field(default_factory=list)
    best_fx: float = float("inf")
    best_x: Optional[np.ndarray] = None

    def record(self, x: np.ndarray, fx: float) -> None:
        if len(self.fxs) >= self.max_eval:
            raise _BudgetExhausted()
        x_copy = np.asarray(x, dtype=np.float64).copy()
        fx_val = float(fx)
        self.xs.append(x_copy)
        self.fxs.append(fx_val)
        if np.isfinite(fx_val) and fx_val < self.best_fx:
            self.best_fx = fx_val
            self.best_x = x_copy


def _make_objective(
    problem: Problem, log: _EvaluationLog
) -> Callable[[np.ndarray], float]:
    """Return a scalar objective that projects into the box, evaluates the
    problem, logs the result, and enforces the evaluation budget."""

    def objective(x: np.ndarray) -> float:
        x_arr = np.asarray(x, dtype=np.float64)
        # Respect the problem's box — scipy solvers already pass candidates
        # inside bounds, but project defensively to cover edge cases with
        # solvers that occasionally probe slightly outside (``basinhopping``
        # has historically done this at tolerance boundaries).
        x_proj = problem.project(x_arr)
        fx = float(problem.eval(x_proj))
        log.record(x_proj, fx)
        return fx

    return objective


# ---------------------------------------------------------------------------
# Minimal strategy-like adapter for the benchmark harness
# ---------------------------------------------------------------------------


class _BaselineConfig:
    """The subset of :class:`panobbgo.config.Config` the harness touches.

    ``max_eval`` is respected as a hard budget.  ``evaluation_method`` is
    accepted (to match the harness contract) but ignored: baselines run
    the objective synchronously inside ``start()``.
    """

    def __init__(self, max_eval: int = 1000) -> None:
        self.max_eval: int = max_eval
        self.evaluation_method: str = "threaded"


class _BaselineResults:
    """Exposes ``.results`` as a pandas DataFrame with the MultiIndex
    columns the harness' ``_get_column`` expects."""

    def __init__(self) -> None:
        self.results: Optional[pd.DataFrame] = None

    def build(self, log: _EvaluationLog) -> None:
        """Materialize the MultiIndex DataFrame from accumulated evaluations.

        Columns follow the Panobbgo convention so that
        :meth:`panobbgo.harness.BenchmarkHarness._get_column` and the
        convergence extractor work without special-casing.
        """
        n = len(log.fxs)
        if n == 0:
            self.results = None
            return

        dim = log.xs[0].size
        data: Dict[Any, Any] = {}
        x_stack = np.vstack(log.xs)
        for j in range(dim):
            data[("x", j)] = x_stack[:, j]
        data[("fx", 0)] = np.asarray(log.fxs, dtype=np.float64)
        data[("cv", 0)] = np.zeros(n, dtype=np.float64)
        data[("who", 0)] = np.asarray([log.who] * n, dtype=object)
        data[("error", 0)] = np.zeros(n, dtype=np.float64)

        midx = pd.MultiIndex.from_tuples(list(data.keys()))
        self.results = pd.DataFrame(data, columns=midx)


class BaselineStrategy:
    """Minimal adapter that presents the strategy surface the benchmark
    harness uses.

    Subclasses override :meth:`_optimize` to implement the actual search
    algorithm, leaning on the provided :class:`_EvaluationLog` for
    bookkeeping and :func:`_make_objective` for the budget-safe wrapper.
    """

    #: Short identifier written to the ``who`` column of the results
    #: DataFrame.  Subclasses override.
    who: str = "Baseline"

    def __init__(self, problem: Problem, parse_args: bool = False) -> None:
        # ``parse_args`` accepted for API compatibility with
        # ``StrategyBase.__init__`` — baselines never parse CLI.
        del parse_args
        self.problem: Problem = problem
        self.config: _BaselineConfig = _BaselineConfig()
        self.results: _BaselineResults = _BaselineResults()
        self._best: Optional[Result] = None
        self._stopped: bool = False

    # -- harness compatibility shims ---------------------------------------

    def add(self, heuristic_class: type, **kwargs: Any) -> None:
        """No-op: baselines ignore Panobbgo heuristics entirely.

        Accepted so :meth:`panobbgo.benchmark.StrategySpec.create_strategy`
        can iterate through a (possibly empty) ``heuristics`` list without
        special-casing baselines.
        """
        del heuristic_class, kwargs

    def add_analyzer(self, analyzer: Any) -> None:
        """No-op: baselines have no event bus for analyzers to subscribe to."""
        del analyzer

    # -- run ---------------------------------------------------------------

    def start(self) -> None:
        """Execute the wrapped solver under the harness budget cap.

        Catches :class:`_BudgetExhausted` so that exceeding the budget from
        inside the external solver is a clean termination, not a crash.
        """
        log = _EvaluationLog(who=self.who, max_eval=max(1, int(self.config.max_eval)))

        try:
            self._optimize(log)
        except _BudgetExhausted:
            # Reached the hard budget mid-optimization — this is expected
            # and is exactly how the harness' own strategies terminate.
            pass
        except Exception:
            # Surface unexpected errors so the harness records them in the
            # RunRecord.error field.
            self.results.build(log)
            self._best = self._build_best_result(log)
            raise

        self.results.build(log)
        self._best = self._build_best_result(log)

    def _optimize(self, log: _EvaluationLog) -> None:
        """Subclass hook: run the actual optimizer, calling the objective
        returned by :func:`_make_objective` up to ``log.max_eval`` times."""
        raise NotImplementedError

    # -- results -----------------------------------------------------------

    @property
    def best(self) -> Optional[Result]:
        """Return the best :class:`~panobbgo.lib.Result` seen, or ``None``
        if the solver produced no finite evaluation (e.g. immediate error)."""
        return self._best

    def _build_best_result(self, log: _EvaluationLog) -> Optional[Result]:
        if log.best_x is None:
            return None
        point = Point(log.best_x, self.who)
        return Result(point, log.best_fx)


# ---------------------------------------------------------------------------
# Concrete baselines
# ---------------------------------------------------------------------------


class RandomSearchStrategy(BaselineStrategy):
    """Pure uniform random search over the problem box.

    This is the **score floor**: any serious optimizer should beat it on
    most problems.  Its composite score on the standard battery gives us a
    lower reference point for what "doing nothing adaptive" achieves under
    the harness formula.
    """

    who: str = "Random"

    def _optimize(self, log: _EvaluationLog) -> None:
        objective = _make_objective(self.problem, log)
        lo = self.problem.box[:, 0]
        hi = self.problem.box[:, 1]
        rng = np.random.default_rng()  # seeded by the harness' np.random.seed
        for _ in range(log.max_eval):
            # np.random.default_rng() is independent of the legacy
            # np.random.seed() call the harness makes, so use the legacy
            # draw path for reproducibility with that seed.
            x = np.random.uniform(lo, hi, size=self.problem.dim)
            objective(x)
            if self._stopped:
                break
        # ``rng`` unused but kept for future migration to modern RNG API.
        del rng


class SciPyDEStrategy(BaselineStrategy):
    """Adapter for :func:`scipy.optimize.differential_evolution`.

    Differential Evolution is a strong population-based global optimizer
    and a standard baseline in the black-box optimization literature.  We
    configure it with a small population so that the harness' 75-to-500
    evaluation budgets produce meaningful generations.
    """

    who: str = "SciPyDE"

    def __init__(
        self,
        problem: Problem,
        parse_args: bool = False,
        popsize: int = 10,
        tol: float = 0.0,
    ) -> None:
        super().__init__(problem, parse_args=parse_args)
        self._popsize = popsize
        self._tol = tol

    def _optimize(self, log: _EvaluationLog) -> None:
        from scipy.optimize import differential_evolution

        objective = _make_objective(self.problem, log)
        bounds = [(float(lo), float(hi)) for lo, hi in self.problem.box]

        # DE uses popsize * dim candidates per generation.  With a hard
        # evaluation cap, set ``maxiter`` generously — the _BudgetExhausted
        # exception inside the objective is the real stopping criterion.
        #
        # ``tol=0`` forces DE to use the full budget; otherwise it will
        # terminate early on convergence of the population standard
        # deviation, wasting evaluations we could spend exploring.
        #
        # Seed derivation: the harness calls ``np.random.seed(seed)`` just
        # before this runs, so ``int(np.random.randint(...))`` gives us a
        # deterministic integer we can pass to scipy's RNG.
        de_seed = int(np.random.randint(0, 2**31 - 1))
        max_generations = max(1, log.max_eval // max(1, self._popsize * self.problem.dim))

        # ``differential_evolution`` uses ``rng`` in scipy >= 1.15 (``seed``
        # is retained as a deprecated alias).  Prefer ``rng``.
        try:
            differential_evolution(
                objective,
                bounds=bounds,
                popsize=self._popsize,
                maxiter=max_generations,
                tol=self._tol,
                rng=de_seed,
                polish=False,  # we enforce budget ourselves; polish would exceed
                init="sobol",
                updating="deferred",
            )
        except _BudgetExhausted:
            raise


class SciPyAnnealStrategy(BaselineStrategy):
    """Adapter for :func:`scipy.optimize.dual_annealing`.

    Dual annealing combines generalized simulated annealing with a local
    search step — a competitive global optimizer on multimodal problems.
    We pair it with ``no_local_search=False`` so it can exploit the
    built-in L-BFGS-B polish, which is fair because CMA-ES and BayesOpt
    harness strategies also include a local-refinement heuristic.
    """

    who: str = "SciPyAnneal"

    def _optimize(self, log: _EvaluationLog) -> None:
        from scipy.optimize import dual_annealing

        objective = _make_objective(self.problem, log)
        bounds = [(float(lo), float(hi)) for lo, hi in self.problem.box]

        da_seed = int(np.random.randint(0, 2**31 - 1))

        # ``dual_annealing`` uses ``rng`` in scipy >= 1.15.
        try:
            dual_annealing(
                objective,
                bounds=bounds,
                maxfun=log.max_eval,
                # ``maxiter`` is the cap on the outer SA loop; we let the
                # hard ``maxfun`` + our _BudgetExhausted guard terminate it.
                maxiter=10_000,
                rng=da_seed,
                no_local_search=False,
            )
        except _BudgetExhausted:
            raise


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def make_baseline_strategies() -> List[StrategySpec]:
    """Return the :class:`StrategySpec` list for the three baseline solvers.

    These are plugged into the harness when a run is requested with
    baselines enabled.  All three are budget-agnostic — they inherit
    ``config.max_eval`` from the harness — so they can be dropped into any
    of the ``quick`` / ``standard`` / ``full`` modes.
    """
    return [
        StrategySpec(
            name="Baseline_Random",
            strategy_class=RandomSearchStrategy,
            heuristics=[],
        ),
        StrategySpec(
            name="Baseline_SciPyDE",
            strategy_class=SciPyDEStrategy,
            heuristics=[],
        ),
        StrategySpec(
            name="Baseline_SciPyAnneal",
            strategy_class=SciPyAnnealStrategy,
            heuristics=[],
        ),
    ]


__all__ = [
    "BaselineStrategy",
    "RandomSearchStrategy",
    "SciPyDEStrategy",
    "SciPyAnnealStrategy",
    "make_baseline_strategies",
]
