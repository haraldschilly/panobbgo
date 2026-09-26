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

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, cast

import numpy as np

from panobbgo.lib.lib import EvaluationFailed

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
        its final best value.  A trajectory *longer* than ``budget`` is
        truncated to it: evaluations past the budget are never scored.
    """
    arr = np.asarray(best_so_far, dtype=np.float64)
    if budget is not None and arr.size > budget:
        arr = arr[: max(0, int(budget))]
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


def aocc_virtual_time(
    timeline: Sequence[Tuple[float, float]],
    *,
    budget: int,
    workers: int,
    f_opt: float = 0.0,
    mean_duration: float = 1.0,
    log_lo: float = AOCC_LOG_LO,
    log_hi: float = AOCC_LOG_HI,
) -> float:
    r"""AOCC over *virtual time* for a run on ``workers`` simulated workers.

    ``timeline`` holds one ``(t_complete, value)`` pair per evaluation: the
    virtual time at which its result became known and the metric's value
    for it, ``NaN`` for a spent call that failed or timed out (as
    :attr:`IOHTracker.timeline` records them, :mod:`panobbgo.virtual_clock`).  The best-so-far at time :math:`t` is the
    minimum over the evaluations completed by :math:`t` (``+inf`` before the
    first).  It is sampled on the grid

    .. math::  t_j = j \cdot \bar d / q, \qquad j = 1, \dots, B

    (:math:`\bar d` = ``mean_duration``, :math:`q` = ``workers``,
    :math:`B` = ``budget``) and scored with :func:`aocc`.  Time is thus in
    units of the mean duration, the horizon is :math:`B \bar d / q` — the
    time :math:`q` perfectly busy workers need to spend the budget — and
    the score is the right Riemann sum of the log-precision curve over it,
    with :math:`B` samples like the per-evaluation AOCC.  Evaluations
    completing after the horizon are not scored; a run that ends early is
    held at its last best value.

    At ``workers = 1`` with a constant duration of ``mean_duration`` the grid
    points are exactly the completion times, so the value equals
    :func:`aocc` over evaluations.  Idle workers, stale candidates and slow
    points (under an x-dependent duration model) all lower it.

    Structural caps — read comparisons *across* ``q`` with them in mind:
    no call can complete before :math:`\bar d` (with a constant duration),
    so the first :math:`q - 1` grid points are always ``+inf`` (the worst
    gap), which caps the score at about :math:`1 - (q-1)/B` of the ideal;
    and with a heavy-tailed duration model (log-normal) the last calls
    usually complete after the horizon and are not scored.  Both effects
    grow with ``q``.  Comparisons at the *same* ``q`` and duration model are
    unaffected.
    """
    budget = int(budget)
    if budget <= 0 or not timeline:
        return 0.0
    if int(workers) < 1 or not mean_duration > 0:
        raise ValueError("workers must be >= 1 and mean_duration > 0")
    arr = np.asarray(timeline, dtype=np.float64).reshape(-1, 2)
    order = np.argsort(arr[:, 0], kind="stable")
    times = arr[order, 0]
    vals = arr[order, 1]
    vals = np.where(np.isnan(vals), np.inf, vals)
    best = np.minimum.accumulate(vals)
    grid = np.arange(1, budget + 1, dtype=np.float64) * (float(mean_duration) / int(workers))
    # A completion at exactly a grid time counts at that time; the relative
    # slack absorbs the rounding of accumulated virtual times.
    n_done = np.searchsorted(times, grid * (1.0 + 1e-9), side="right")
    sampled = np.where(n_done > 0, best[np.maximum(n_done - 1, 0)], np.inf)
    return aocc(sampled.tolist(), f_opt=f_opt, log_lo=log_lo, log_hi=log_hi, budget=budget)


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
    """Convergence trace from one optimisation run.

    On a noiseless problem there is one trace and ``best_so_far`` is it.
    On a noisy one (:class:`~panobbgo.lib.noise.NoisyProblem`) three
    traces are recorded and they say different things:

    * ``best_so_far`` — the best **observed** (noisy) value.  What the
      optimizer believes it has achieved; biased optimistically, because
      the minimum of many noisy readings is below the minimum of their
      means.  Not a metric.
    * ``best_so_far_true`` — the best **true** value over all evaluated
      points.  The BBOB noisy-suite convention and the metric of record:
      it credits the algorithm for having *visited* a good point, whether
      or not the noise let it recognise one.
    * ``best_so_far_reco`` — the true value of the current
      **recommendation**, i.e. of the point with the best observed value.
      Not monotone.  The gap to ``best_so_far_true`` is exactly the cost
      of being fooled by the noise, which no other trace measures.
    """

    best_so_far: List[float]
    n_evals: int
    best_x: Optional[np.ndarray]
    best_fx: float
    best_so_far_true: Optional[List[float]] = None
    best_so_far_reco: Optional[List[float]] = None
    best_true_fx: float = float("nan")

    def aocc(self, f_opt: float = 0.0, budget: Optional[int] = None) -> float:
        return aocc(self.best_so_far, f_opt=f_opt, budget=budget)

    def aocc_true(self, f_opt: float = 0.0, budget: Optional[int] = None) -> float:
        """AOCC on the true trace, falling back to the observed one."""
        trace = self.best_so_far_true if self.best_so_far_true is not None else self.best_so_far
        return aocc(trace, f_opt=f_opt, budget=budget)


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

    ``timeout_s`` adds a wall-clock deadline that ends the run the same
    way the budget does: past it, evaluations are no longer counted and
    :attr:`timed_out` is set.  The trajectory up to the deadline is kept,
    so the run is still scored — penalised, like any run that stops
    early, at its final best-fx for the rest of the budget.

    Thread-safe: an asynchronous strategy evaluates from a pool, so the
    budget slot is *reserved* under a lock before the objective is called
    (no evaluation past the budget ever reaches the problem) and the
    best-so-far update is recorded under the same lock (the trace stays
    monotone and exactly ``budget`` long at most).
    """

    def __init__(self, problem: Any, budget: int, *, hard: bool = False, timeout_s: Optional[float] = None) -> None:
        self.problem = problem
        self.budget = int(budget)
        self.hard = bool(hard)
        self._deadline: Optional[float] = None if timeout_s is None else time.monotonic() + float(timeout_s)
        self.timed_out: bool = False
        self.n_evals: int = 0
        self.best_fx: float = float("inf")
        self.best_x: Optional[np.ndarray] = None
        self.best_so_far: List[float] = []
        self._lock = threading.Lock()
        #: Called once (after the lock is released) when the deadline passes — a driver
        #: points it at ``strategy.request_stop`` so a timed-out run ends
        #: instead of spinning through no-op evaluations to ``max_eval``.
        self.on_timeout: Optional[Callable[[], None]] = None
        #: Evaluations admitted against the budget: recorded plus in flight.
        self._reserved: int = 0
        #: Virtual-clock observer state (:mod:`panobbgo.virtual_clock`): while
        #: a simulated call runs, the key of that call; its measurements wait
        #: in ``_deferred`` until the call *completes* (:meth:`complete_call`),
        #: so the traces are recorded in completion order.
        self._defer_key: Optional[int] = None
        self._deferred: Dict[int, List[Tuple[np.ndarray, Tuple[float, ...]]]] = {}
        #: ``(t_complete, value)`` per counted evaluation of a virtual-clock
        #: run, in completion order — ``value`` is the one the metric scores
        #: (the true value on a noisy problem), ``NaN`` for a spent call that
        #: failed or timed out.  Input of :func:`aocc_virtual_time`.
        self.timeline: List[Tuple[float, float]] = []

        # Noisy problems expose ``eval_pair(x) -> (noisy, true)``: both
        # values out of *one* inner evaluation, so the true trace costs no
        # extra worker round-trips.  A plain problem has no such method and
        # the true trace is simply not recorded (``has_true`` is False).
        pair = getattr(problem, "eval_pair", None)
        self._eval_pair: Optional[Callable[[np.ndarray], Tuple[float, float]]] = (
            cast("Callable[[np.ndarray], Tuple[float, float]]", pair) if callable(pair) else None
        )
        self.has_true: bool = self._eval_pair is not None
        self.best_true_fx: float = float("inf")
        self.best_so_far_true: List[float] = []
        #: True value of the incumbent — the point with the best *observed*
        #: value.  Non-monotone under noise; see :class:`Trajectory`.
        self.best_so_far_reco: List[float] = []
        self._incumbent_true: float = float("inf")

        self._orig_eval: Callable[[np.ndarray], float] = problem.eval
        problem.eval = self._tracked_eval  # type: ignore[method-assign]

    def _tracked_eval(self, x: np.ndarray) -> float:
        fire_timeout = False
        with self._lock:
            if not self.timed_out and self._deadline is not None and time.monotonic() > self._deadline:
                self.timed_out = True
                fire_timeout = True
            admitted = self._reserved < self.budget and not self.timed_out
            if admitted:
                self._reserved += 1
            last_best = self.best_fx
        if fire_timeout and self.on_timeout is not None:
            # Outside the lock: the callback may call back into the problem.
            self.on_timeout()
        if not admitted:
            if self.hard:
                raise _BudgetExhausted()
            # Soft mode: don't fail the evaluation; just signal "no useful
            # value" so the strategy treats it as a non-improvement.
            return last_best if np.isfinite(last_best) else float("inf")
        try:
            measured = self._measure(x)
        except EvaluationFailed:
            # A simulated failure (crash or timeout, see lib.EvaluationFailed)
            # is a call that was made and paid for: it counts as one spent
            # evaluation that makes no progress, so the trace index stays
            # aligned with the budget the strategy spent.  Re-raised for the
            # evaluation path to book.
            with self._lock:
                self.n_evals += 1
                self._record_failed()
            raise
        except BaseException:
            with self._lock:
                self._reserved -= 1  # the slot was never used
            raise
        with self._lock:
            if self._defer_key is not None:
                # A simulated call: counted when it completes (complete_call).
                self._deferred.setdefault(self._defer_key, []).append(
                    (np.asarray(x, dtype=np.float64).copy(), measured)
                )
            else:
                self.n_evals += 1
                self._record(x, measured)
        return measured[0]

    # ── virtual-clock observer (panobbgo.virtual_clock) ──

    def begin_call(self, key: int) -> None:
        """A simulated call ``key`` starts evaluating: defer what it measures."""
        self._defer_key = int(key)

    def end_call(self) -> None:
        """The objective call of the current simulated call returned (or raised)."""
        self._defer_key = None

    def complete_call(self, key: int, t_complete: float, ok: bool) -> None:
        """Simulated call ``key`` completed at virtual time ``t_complete``; fold it into the traces.

        A successful call records its deferred measurement(s).  A failed or
        timed-out one (``ok`` false) is one spent, non-improving evaluation
        (:meth:`record_spent`): it used a budget slot and a worker, whatever
        it may have measured before failing — extra slots its measurements
        reserved are released.  Called in completion order.
        """
        with self._lock:
            entries = self._deferred.pop(int(key), [])
            if ok and entries:
                for x, measured in entries:
                    self.n_evals += 1
                    self._record(x, measured)
                    self.timeline.append((float(t_complete), float(measured[1])))
                return
            # The failed call's measurements give their slots back; it is
            # charged exactly one below.
            self._reserved -= len(entries)
        self.record_spent(t_complete)

    def record_spent(self, t_complete: float) -> None:
        """Count one spent, non-improving evaluation (a failed or timed-out call) completing at ``t_complete``.

        Subject to the budget and the wall-clock deadline like any evaluation.
        """
        fire_timeout = False
        with self._lock:
            if not self.timed_out and self._deadline is not None and time.monotonic() > self._deadline:
                self.timed_out = True
                fire_timeout = True
            if not self.timed_out and self._reserved < self.budget:
                self._reserved += 1
                self._spend(float(t_complete))
        if fire_timeout and self.on_timeout is not None:
            self.on_timeout()

    def _spend(self, t_complete: float) -> None:
        """One spent evaluation: the traces repeat their last value (called under the lock)."""
        self.n_evals += 1
        self.best_so_far.append(self.best_fx)
        if self.has_true:
            self.best_so_far_true.append(self.best_true_fx)
            self.best_so_far_reco.append(self._incumbent_true)
        self.timeline.append((t_complete, float("nan")))

    def _measure(self, x: np.ndarray) -> Tuple[float, ...]:
        """Evaluate ``x`` (outside the lock); element 0 goes back to the strategy, element 1 is scored."""
        if self._eval_pair is not None:
            noisy, true_fx = self._eval_pair(x)
            return float(noisy), float(true_fx)
        fx = float(self._orig_eval(x))
        return fx, fx

    def _record(self, x: np.ndarray, measured: Tuple[float, ...]) -> None:
        """Fold one admitted evaluation into the traces (called under the lock)."""
        fx, tfx = measured
        if np.isfinite(fx) and fx < self.best_fx:
            self.best_fx = fx
            self.best_x = np.asarray(x, dtype=np.float64).copy()
            self._incumbent_true = tfx
        if np.isfinite(tfx) and tfx < self.best_true_fx:
            self.best_true_fx = tfx
        self.best_so_far.append(self.best_fx)
        if self.has_true:
            self.best_so_far_true.append(self.best_true_fx)
            self.best_so_far_reco.append(self._incumbent_true)

    def _record_failed(self) -> None:
        """Fold one failed evaluation into the traces: spent, no progress (called under the lock)."""
        self.best_so_far.append(self.best_fx)
        if self.has_true:
            self.best_so_far_true.append(self.best_true_fx)
            self.best_so_far_reco.append(self._incumbent_true)

    def restore(self) -> None:
        """Restore the original ``eval`` so the problem can be reused."""
        self.problem.eval = self._orig_eval  # type: ignore[method-assign]

    def trajectory(self) -> Trajectory:
        with self._lock:
            return self._trajectory_locked()

    def _trajectory_locked(self) -> Trajectory:
        return Trajectory(
            best_so_far=list(self.best_so_far),
            n_evals=self.n_evals,
            best_x=self.best_x,
            best_fx=self.best_fx,
            best_so_far_true=list(self.best_so_far_true) if self.has_true else None,
            best_so_far_reco=list(self.best_so_far_reco) if self.has_true else None,
            best_true_fx=self.best_true_fx if self.has_true else float("nan"),
        )
