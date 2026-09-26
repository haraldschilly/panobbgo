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

r"""
AOCC on the real-world problems
===============================

An in-process AOCC runner for :mod:`panobbgo.lib.realworld` (CEC 2020
real-world constrained problems).  It is the measurement of
:mod:`panobbgo.harness_families` — the same run driver
(``harness_ioh._run_tracked``), :func:`~panobbgo.ioh_runner.aocc`, records
(:class:`~panobbgo.harness_ioh.IOHRunRecord`), per-run seeds and worker
pool — with a different scored quantity, because a real problem has
neither a known global minimum over the box nor a common scale.

The scored value: feasible relative gap
---------------------------------------

For every evaluation the tracker records

.. math::

    v(x) = \begin{cases}
        \dfrac{f(x) - f_{\mathrm{best}}}{|f_{\mathrm{best}}|} & \text{if } x \text{ is feasible,} \\
        +\infty & \text{otherwise,}
    \end{cases}

where :math:`f_{\mathrm{best}}` is the best-known value
(:attr:`RealWorldProblem.f_best <panobbgo.lib.realworld.RealWorldProblem.f_best>`)
and *feasible* is the CEC 2020 rule: every :math:`g_i \le 0` and every
:math:`|h_j| \le 10^{-4}`.  AOCC is then taken with ``f_opt = 0``, so the
log-precision targets :math:`10^{-8} \dots 10^{2}` are *relative* gaps.
Why not the penalty value of the family track
(:class:`~panobbgo.harness_families.PenaltyTracker`)?

* **The penalty is exploitable here.**  On a family instance the optimum
  is the minimum of :math:`f` over the whole box, so
  :math:`f + \rho\,\mathrm{cv} \ge f_{\mathrm{opt}}` everywhere.  On a real
  problem the constrained optimum is not the unconstrained one: an
  infeasible point with a small violation and a large Lagrange multiplier
  scores below :math:`f_{\mathrm{best}}` for any fixed :math:`\rho` — full
  AOCC credit without a feasible point.  The suite ranks feasible points
  first; so does this metric.
* **Scales differ by eight orders of magnitude** (0.0127 for the spring,
  2.96e6 for the gas compressor).  An absolute precision would make the
  AOCC depend on the units of each problem; the relative gap does not.

Consequences: a run that never finds a feasible point scores 0; an
evaluation that fails (:class:`~panobbgo.lib.lib.EvaluationCrashed`, a
real failure region of the model) is spent budget without progress; and a
point *below* the best-known value (negative gap) gets full credit at that
step.  The record's ``f_opt`` is ``0`` and ``best_fx`` the best relative
gap (``inf`` if nothing feasible was found), so
:attr:`IOHRunRecord.precision <panobbgo.harness_ioh.IOHRunRecord.precision>`
reads as the relative gap.

Public surface
--------------

* :func:`run_realworld_harness` — the entry point.
* :func:`make_realworld_battery` — every problem of
  :data:`~panobbgo.lib.realworld.REALWORLD_SPECS`;
  :func:`make_realworld_quick_battery` — a three-problem smoke test.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from panobbgo.benchmark import StrategySpec
from panobbgo.harness_ioh import (
    AOCC_LOG_HI,
    AOCC_LOG_LO,
    IOHHarnessResult,
    IOHRunRecord,
    IOHTracker,
    _derive_seed,
    _print_run,
    _run_tasks_in_pool,
    _run_tracked,
    _TrackedRun,
    warn_missing_time_scores,
)
from panobbgo.lib.realworld import RealWorldProblem, make_realworld_instances
from panobbgo.virtual_clock import VirtualSpec

#: ``(name, problem)`` pairs, as returned by :func:`~panobbgo.lib.realworld.make_realworld_instances`.
RealWorldInstances = List[Tuple[str, RealWorldProblem]]

#: Budget multiplier of the real-world battery (``budget = 500 * dim``), as on the family track.
REALWORLD_BUDGET_MULTIPLIER: int = 500


class FeasibleGapTracker(IOHTracker):
    r""":class:`IOHTracker` that records the feasible relative gap (module docstring).

    The strategy still receives the true objective ``f(x)``; panobbgo's own
    constraint handling reads the constraints through
    :meth:`RealWorldProblem.eval_constraints
    <panobbgo.lib.realworld.RealWorldProblem.eval_constraints>`.  Only the
    metric's best-so-far trace is the gap.
    """

    def __init__(self, problem: RealWorldProblem, budget: int, *, timeout_s: Optional[float] = None) -> None:
        super().__init__(problem, budget, timeout_s=timeout_s)
        #: Objective value at the best feasible point (``inf`` until one is found).
        self.best_raw_fx: float = float("inf")

    def _measure(self, x: np.ndarray) -> Tuple[float, ...]:
        fx = float(self._orig_eval(x))  # EvaluationCrashed propagates: a spent, failed call
        c = self.problem.eval_constraints(x)
        feasible = bool(np.all(np.asarray(c) <= 0.0)) and np.isfinite(fx)
        gap = self.problem.relative_gap(fx) if feasible else float("inf")
        return fx, gap

    def _record(self, x: np.ndarray, measured: Tuple[float, ...]) -> None:
        fx, gap = measured
        if gap < self.best_fx:
            self.best_fx = gap
            self.best_raw_fx = fx
            self.best_x = np.asarray(x, dtype=np.float64).copy()
        self.best_so_far.append(self.best_fx)


def make_realworld_battery(names: Optional[Sequence[str]] = None) -> RealWorldInstances:
    """Every real-world problem (or the named ones), in registry order.

    Opt-in (``ioh_benchmark.py run --realworld``); not part of any frozen
    preset or of ``scripts/rebaseline.py``.  At ``500·D`` the whole battery
    is 52 500 evaluations per strategy and seed (18 problems, dims 2–14).
    """
    return make_realworld_instances(names)


#: The smoke-test subset: one equality-constrained chemical process with a
#: failure region (RC01), one mixed-integer process synthesis problem, one classic
#: mechanical design.
QUICK_PROBLEMS: Tuple[str, ...] = ("rc01_heat_exchanger_1", "rc10_process_flow_sheeting", "rc17_spring")


def make_realworld_quick_battery() -> RealWorldInstances:
    """Three small problems for a smoke test (:data:`QUICK_PROBLEMS`); not a measurement."""
    return make_realworld_instances(QUICK_PROBLEMS)


def _run_one(
    strategy_spec: StrategySpec,
    problem: RealWorldProblem,
    rep: int,
    budget: int,
    seed: int,
    log_lo: float,
    log_hi: float,
    sync_eval: bool,
    timeout_s: Optional[float] = None,
    virtual: Optional[VirtualSpec] = None,
) -> IOHRunRecord:
    """Run one strategy on one real-world problem; same driver as ``harness_families._run_one``."""
    t0 = time.time()
    tracked = _TrackedRun(n_evals=0, best_fx=float("inf"), aocc=0.0, trace_evals=[], trace_fx=[])
    try:
        tracked = _run_tracked(
            strategy_spec.with_regime_class("clean"),
            problem,
            FeasibleGapTracker(problem, budget=budget, timeout_s=timeout_s),
            f_opt=0.0,
            budget=budget,
            seed=seed,
            sync_eval=sync_eval,
            log_lo=log_lo,
            log_hi=log_hi,
            timeout_s=timeout_s,
            virtual=virtual,
        )
    except Exception as e:  # noqa: BLE001 — record and continue, as the other tracks do
        tracked.error = f"{type(e).__name__}: {e}"
    return IOHRunRecord(
        problem_kind=problem.family,
        dim=problem.dim,
        instance=problem.instance,
        strategy_name=strategy_spec.name,
        rep=rep,
        budget=budget,
        n_evals=tracked.n_evals,
        best_fx=tracked.best_fx,
        f_opt=0.0,
        aocc=tracked.aocc,
        elapsed_s=time.time() - t0,
        seed=seed,
        error=tracked.error,
        trace_evals=tracked.trace_evals,
        trace_fx=tracked.trace_fx,
        aocc_time=tracked.aocc_time,
    )


def run_realworld_harness(
    specs: Sequence[StrategySpec],
    instances: Sequence[Tuple[str, RealWorldProblem]],
    *,
    budget_multiplier: int = REALWORLD_BUDGET_MULTIPLIER,
    base_seed: int = 42,
    sync_eval: bool = True,
    reps: int = 1,
    log_lo: float = AOCC_LOG_LO,
    log_hi: float = AOCC_LOG_HI,
    progress: bool = True,
    battery_name: str = "realworld",
    timeout_s: Optional[float] = None,
    jobs: int = 1,
    virtual: Optional[VirtualSpec] = None,
) -> IOHHarnessResult:
    """Score every spec on every real-world problem and return an AOCC result.

    The parameters mean what they mean in
    :func:`~panobbgo.harness_families.run_family_harness`: the budget per
    run is ``budget_multiplier * problem.dim``, ``base_seed`` seeds the
    optimiser (the problems are fixed), the per-run seed is derived from
    ``spec.rng_identity``, ``jobs > 1`` runs the cells in worker processes
    and ``virtual`` runs on the virtual clock.  Each record's
    ``problem_kind`` is the problem name (``"rc17_spring"``), ``instance``
    is 0, ``f_opt`` is 0 and ``best_fx`` the best feasible relative gap.
    """
    total = len(instances) * len(specs) * int(reps)
    runs: List[IOHRunRecord] = []
    tasks: List[Dict[str, Any]] = []
    idx = 0
    for _name, problem in instances:
        budget = int(budget_multiplier) * problem.dim
        for spec in specs:
            for rep in range(int(reps)):
                idx += 1
                seed = _derive_seed(base_seed, problem.family, problem.dim, problem.instance, spec.rng_identity, rep)
                task: Dict[str, Any] = dict(
                    strategy_spec=spec,
                    problem=problem,
                    rep=rep,
                    budget=budget,
                    seed=seed,
                    log_lo=log_lo,
                    log_hi=log_hi,
                    sync_eval=sync_eval,
                    timeout_s=timeout_s,
                    virtual=virtual,
                )
                if jobs > 1:
                    tasks.append(task)
                    continue
                if progress:
                    print(
                        f"  [{idx:>3d}/{total:>3d}] {problem.family:<28s} dim={problem.dim:<2d} rep={rep} {spec.name}",
                        flush=True,
                    )
                rec = _run_one(**task)
                if progress:
                    _print_run(rec, budget)
                runs.append(rec)
    if tasks:
        runs = _run_tasks_in_pool(tasks, jobs, total, progress, fn=_run_one)

    result = IOHHarnessResult(
        battery_name=battery_name,
        problem_kind="realworld",
        log_lo=log_lo,
        log_hi=log_hi,
        sync_eval=sync_eval or virtual is not None,
        runs=runs,
        virtual=None if virtual is None else virtual.to_dict(),
    )
    warn_missing_time_scores(result)
    return result
