# -*- coding: utf8 -*-
# Copyright 2012-2026 Panobbgo Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""
IOHprofiler runner
==================

Drives a panobbgo strategy against an IOHprofiler problem (BBOB / MA-BBOB /
...), records the convergence trajectory, and computes the AOCC (Area Over
the Convergence Curve) metric used by the MA-BBOB Anytime competition.

The competition rules (2025 edition, used here as a forward-compatible
spec) are:

- dimensions 2 and 5
- budget = 2000 * d evaluations per (problem, instance)
- metric: average AOCC across all (problem, instance) pairs
- log-precision targets in [1e-8, 1e2] (10 orders of magnitude)

This module deliberately stays thin and free of harness/Dask infrastructure
so it can be invoked from a notebook or a unit test without spinning up
the full benchmark machinery.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# AOCC
# ---------------------------------------------------------------------------

#: IOH/MA-BBOB anytime competition default: 10 orders of magnitude.
AOCC_LOG_LO: float = -8.0
AOCC_LOG_HI: float = 2.0


def aocc(
    best_so_far: Sequence[float],
    f_opt: float = 0.0,
    log_lo: float = AOCC_LOG_LO,
    log_hi: float = AOCC_LOG_HI,
    budget: Optional[int] = None,
) -> float:
    """Area Over the Convergence Curve, normalised to ``[0, 1]``.

    For each evaluation step ``t``, the precision ``p(t) = best_so_far(t) - f_opt``
    is mapped onto a log scale and clipped to ``[log_lo, log_hi]``.  AOCC is
    then ``1 - mean_t (log10 p(t) - log_lo) / (log_hi - log_lo)``, i.e. the
    fraction of the log-precision range we are *below* on average.  Higher
    is better.

    Parameters
    ----------
    best_so_far
        Monotonically non-increasing trajectory of best-so-far objective
        values, one entry per evaluation.
    f_opt
        Known optimum value (IOH problems carry this via ``optimum.y``).
    log_lo, log_hi
        Log10 bounds of the precision-target range.  Defaults match the
        MA-BBOB anytime competition.
    budget
        If given and larger than ``len(best_so_far)``, the trajectory is
        right-padded with its last value to ``budget`` evaluations before
        the mean is taken.  This matches the IOH convention: an algorithm
        that stops early is held responsible for the remaining budget at
        its final best value.
    """
    arr = np.asarray(best_so_far, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    if budget is not None and arr.size < budget:
        pad = np.full(budget - arr.size, arr[-1], dtype=np.float64)
        arr = np.concatenate([arr, pad])
    precision = np.maximum(arr - f_opt, 10.0**log_lo)
    log_p = np.log10(precision)
    log_p = np.clip(log_p, log_lo, log_hi)
    normalised_gap = (log_p - log_lo) / (log_hi - log_lo)
    return float(1.0 - normalised_gap.mean())


# ---------------------------------------------------------------------------
# Budget-enforced objective adapter
# ---------------------------------------------------------------------------


class _BudgetExhausted(Exception):
    """Raised when the per-problem evaluation budget is reached.

    Retained for backwards-compatibility with code that catches it
    (e.g. external solver adapters that want a hard-stop signal).
    The default :class:`IOHTracker` mode is *soft*: once the budget is
    exhausted it simply returns ``+inf`` and stops recording, so the
    strategy's own ``max_eval`` check terminates the run cleanly without
    raising from evaluator threads.
    """


@dataclass
class Trajectory:
    """Convergence trace from one optimisation run."""

    best_so_far: List[float]
    n_evals: int
    best_x: Optional[np.ndarray]
    best_fx: float

    def aocc(self, f_opt: float = 0.0, budget: Optional[int] = None) -> float:
        return aocc(self.best_so_far, f_opt=f_opt, budget=budget)


class IOHTracker:
    """Wraps an :class:`~panobbgo.lib.ioh_wrapper.IOHProblem` so that every
    evaluation is recorded and the budget is enforced.

    The tracker exposes a ``problem``-shaped object (drop-in for panobbgo
    strategies) whose ``eval`` records best-so-far.

    Budget enforcement is **soft by default** (``hard=False``): once the
    budget is exhausted, further calls return the last best-fx (or
    ``+inf`` if nothing has been recorded yet) and are not counted —
    they are no-ops from the metric's point of view.  This keeps panobbgo
    strategies running cleanly to their own ``max_eval`` check without
    raising from evaluator threads (which would otherwise be logged as
    errors in the threaded executor).

    Set ``hard=True`` for the legacy behaviour of raising
    :class:`_BudgetExhausted` past budget, used by adapters that want a
    hard-stop control-flow signal (e.g. scipy DE inside a baseline
    strategy).
    """

    def __init__(self, problem: Any, budget: int, *, hard: bool = False) -> None:
        self.problem = problem
        self.budget = int(budget)
        self.hard = bool(hard)
        self.n_evals: int = 0
        self.best_fx: float = float("inf")
        self.best_x: Optional[np.ndarray] = None
        self.best_so_far: List[float] = []

        self._orig_eval: Callable[[np.ndarray], float] = problem.eval
        problem.eval = self._tracked_eval  # type: ignore[method-assign]

    def _tracked_eval(self, x: np.ndarray) -> float:
        if self.n_evals >= self.budget:
            if self.hard:
                raise _BudgetExhausted()
            # Soft mode: don't fail the evaluation; just signal "no useful
            # value" so the strategy treats it as a non-improvement.
            return self.best_fx if np.isfinite(self.best_fx) else float("inf")
        fx = float(self._orig_eval(x))
        self.n_evals += 1
        if np.isfinite(fx) and fx < self.best_fx:
            self.best_fx = fx
            self.best_x = np.asarray(x, dtype=np.float64).copy()
        self.best_so_far.append(self.best_fx)
        return fx

    def restore(self) -> None:
        """Restore the original ``eval`` so the problem can be reused."""
        self.problem.eval = self._orig_eval  # type: ignore[method-assign]

    def trajectory(self) -> Trajectory:
        return Trajectory(
            best_so_far=list(self.best_so_far),
            n_evals=self.n_evals,
            best_x=self.best_x,
            best_fx=self.best_fx,
        )


# ---------------------------------------------------------------------------
# Strategy driver
# ---------------------------------------------------------------------------


def run_strategy_on_ioh_problem(
    ioh_problem: Any,
    *,
    strategy_factory: Callable[[Any], Any],
    budget: Optional[int] = None,
) -> Trajectory:
    """Run a panobbgo strategy against a wrapped IOH problem.

    Parameters
    ----------
    ioh_problem
        An ``ioh.problem.RealSingleObjective`` (or compatible) instance.
    strategy_factory
        Callable ``problem -> strategy`` that returns a configured panobbgo
        strategy ready to ``start()``.  The factory is given the wrapped
        :class:`IOHProblem` (not the raw IOH object) and should attach all
        heuristics before returning.
    budget
        Override evaluation budget.  Defaults to ``2000 * dim`` per the
        MA-BBOB anytime competition rules.

    Returns
    -------
    Trajectory
        The best-so-far trace, evaluation count, and best ``(x, fx)``.
    """
    from panobbgo.lib.ioh_wrapper import IOHProblem

    wrapped = IOHProblem(ioh_problem)
    if budget is None:
        budget = 2000 * wrapped.dim

    tracker = IOHTracker(wrapped, budget=budget)
    try:
        strategy = strategy_factory(wrapped)
        # The strategy's own max_eval should be set via panobbgo config,
        # but the tracker enforces budget hard-stop regardless.
        if hasattr(strategy, "config"):
            if hasattr(strategy.config, "max_eval"):
                strategy.config.max_eval = budget
            # Anytime metric: don't let convergence detection forfeit the
            # remaining budget — the tracker is the only authority that
            # ends the run.
            if hasattr(strategy.config, "stop_on_convergence"):
                strategy.config.stop_on_convergence = False
        try:
            strategy.start()
        except _BudgetExhausted:
            pass
    finally:
        tracker.restore()

    return tracker.trajectory()
