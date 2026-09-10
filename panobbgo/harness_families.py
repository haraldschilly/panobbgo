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
AOCC on generated problem families
==================================

An in-process AOCC runner for :mod:`panobbgo.lib.families` instances.  It
is the *same* measurement as :mod:`panobbgo.harness_ioh` — the same
:class:`~panobbgo.ioh_runner.IOHTracker`, the same
:func:`~panobbgo.ioh_runner.aocc`, the same
:class:`~panobbgo.harness_ioh.IOHRunRecord` /
:class:`~panobbgo.harness_ioh.IOHHarnessResult` records, the same
``_derive_seed`` per-run seeds keyed on ``StrategySpec.rng_identity`` —
pointed at a different set of problems.  Everything is imported rather
than re-implemented so the two tracks cannot drift apart, and so the
existing analysis code (``paired_seed_stats``, ``aocc_to_harness_result``,
``benchmarks/portfolio_screen.py``'s cell folding) works on these results
unchanged.

Two things differ from the IOH track, both by necessity:

1.  **No worker subprocess.**  The problems are pure Python, so there is
    no ``tools/ioh_worker`` venv to install and the ``requires_worker``
    pytest marker does not apply.  A family battery therefore runs in CI
    and on a machine where ``ioh`` cannot be built.
2.  **Constrained instances need AOCC to be defined.**  See below.

AOCC on a constrained instance
------------------------------

AOCC needs a scalar, monotone best-so-far trace and a target ``f_opt``.
The scalar this module tracks is the **penalty value**

.. math::

    \varphi(x) = f(x) + \rho \cdot \mathrm{cv}(x), \qquad \rho = 100,

which is exactly
:meth:`ConstraintHandler.get_penalty_value
<panobbgo.lib.constraints.ConstraintHandler.get_penalty_value>` under the
shipped :class:`~panobbgo.lib.constraints.DefaultConstraintHandler`
(``rho = 100.0``) — the same number
:class:`~panobbgo.strategies.StrategyBlockBandit` already rewards its arms
on (``strategies/blocks.py``), and the same
:math:`\mathrm{cv} = \lVert \max(g, 0) \rVert_2` that
:attr:`Result.cv <panobbgo.lib.Result.cv>` and hence the ``Best`` analyzer
use.  Nothing new is invented here.

What that definition means in practice:

* On a feasible point ``cv = 0``, so :math:`\varphi = f`: for the
  unconstrained families the metric is bit-identical to the IOH track's.
* :math:`\varphi(x) - f_{\mathrm{opt}} \ge 0` **everywhere**, because
  :math:`x_{\mathrm{opt}}` is the global minimiser of :math:`f` over the
  whole box (see :class:`~panobbgo.lib.families.Family`) and
  :math:`\rho\,\mathrm{cv} \ge 0`.  So ``f_opt`` remains a true AOCC
  target and no infeasible point can score better than the optimum.
* **Infeasible points do not count as progress**, up to the
  :math:`\rho`-weighting: an infeasible point registers only if
  :math:`f(x) + 100\,\mathrm{cv}(x)` beats the incumbent, and at the
  1e-8 end of the AOCC target range that requires
  :math:`\mathrm{cv} \le 10^{-10}` — feasibility in all but name.  A
  strictly lexicographic "feasible-only best-so-far" would differ only in
  the coarse first orders of magnitude, and would *not* be the quantity
  the strategies are optimising; the penalty value is.

Public surface
--------------

* :func:`run_family_harness` — the entry point.
* :func:`make_families_battery` / :func:`make_constrained_battery` — the
  two presets.
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
    _BudgetExhausted,
    _derive_seed,
    _downsample_trajectory,
    aocc,
)
from panobbgo.lib.families import Family, FamilyConfig, FamilyLike, make_family_instances

#: Penalty coefficient of :class:`DefaultConstraintHandler
#: <panobbgo.lib.constraints.DefaultConstraintHandler>`.  Pinned here as a
#: constant rather than read off the handler because the metric must not
#: silently change meaning when a default is retuned — a battery measured
#: at one ``rho`` is not comparable to one measured at another.
PENALTY_RHO: float = 100.0

#: Default battery seed.  Fixed so a family battery is a stable contract
#: the way an IOH instance id is: ``base_seed`` varies the *optimiser*,
#: never the problems.
DEFAULT_BATTERY_SEED: int = 20260910

#: ``(name, problem)`` pairs, as returned by
#: :func:`~panobbgo.lib.families.make_family_instances`.
FamilyInstances = List[Tuple[str, Family]]


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------


class PenaltyTracker(IOHTracker):
    r""":class:`IOHTracker` that records the penalty value, not ``f``.

    The strategy still *receives* the true objective ``f(x)`` from
    ``eval`` — panobbgo's own machinery evaluates the constraints
    separately (:meth:`Problem.__call__ <panobbgo.lib.Problem.__call__>`
    builds ``Result.cv_vec`` from ``eval_constraints``), and handing it a
    pre-penalised objective would double-count.  Only the *metric's*
    best-so-far trace is penalised, with
    :math:`\varphi = f + \rho \cdot \lVert \max(g, 0) \rVert_2`.

    ``best_fx`` therefore holds the best :math:`\varphi`, which is what
    :attr:`IOHRunRecord.precision` must be computed from.
    """

    def __init__(self, problem: Any, budget: int, *, rho: float = PENALTY_RHO, hard: bool = False) -> None:
        super().__init__(problem, budget, hard=hard)
        self.rho = float(rho)
        #: Objective value at the best *penalty* point (for reporting).
        self.best_raw_fx: float = float("inf")
        #: Constraint violation at the best penalty point.
        self.best_cv: float = float("inf")

    def _tracked_eval(self, x: np.ndarray) -> float:
        if self.n_evals >= self.budget:
            if self.hard:
                raise _BudgetExhausted()
            return self.best_fx if np.isfinite(self.best_fx) else float("inf")
        fx = float(self._orig_eval(x))
        cv_vec = self.problem.eval_constraints(x)
        if cv_vec is None:
            cv = 0.0
        else:
            positive = np.asarray(cv_vec, dtype=np.float64)
            positive = positive[positive > 0.0]
            cv = float(np.linalg.norm(positive)) if positive.size else 0.0
        phi = fx + self.rho * cv
        self.n_evals += 1
        if np.isfinite(phi) and phi < self.best_fx:
            self.best_fx = phi
            self.best_raw_fx = fx
            self.best_cv = cv
            self.best_x = np.asarray(x, dtype=np.float64).copy()
        self.best_so_far.append(self.best_fx)
        return fx


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------
#
# Both presets are frozen contracts in the sense of ``planning/GOAL.md``
# §4: extend by composing a new instance list, do not edit these.  The
# instance *seed* is fixed (DEFAULT_BATTERY_SEED) so the problems are the
# same on every run; only ``base_seed`` — the optimiser's RNG — moves.


def make_families_battery(
    dims: Sequence[int] = (2, 5, 10),
    n_instances: int = 3,
    seed: int = DEFAULT_BATTERY_SEED,
) -> FamilyInstances:
    """Unconstrained battery: 5 families x ``dims`` x ``n_instances``.

    The five families span the shapes the MA-BBOB battery of record
    samples only incidentally, one per failure mode:

    * ``ellipsoid`` — ill-conditioned (1e6) unimodal, non-separable once
      rotated: the case a covariance model exists for.
    * ``rosenbrock`` — a curved valley: conditioning that *changes* along
      the path.
    * ``rastrigin`` — regular multimodality at a scale a population can
      still see.
    * ``ackley`` — multimodality on a nearly flat outer basin: the case
      where a local model has nothing to fit.
    * ``sharp_ridge`` — non-smooth, gradient not vanishing at the
      optimum: the case a smooth surrogate is systematically wrong about.

    ``sphere``, ``discus``, ``griewank`` and ``schwefel`` are available in
    :data:`~panobbgo.lib.families.BASE_FUNCTIONS` and deliberately left
    out of the preset — the sphere adds no discrimination, and the other
    three duplicate a shape already covered.

    Note the cost asymmetry: the budget is ``budget_multiplier * dim``,
    so the ``d = 10`` third of the battery is half of its evaluations.
    """
    families: List[FamilyLike] = [
        FamilyConfig(base="ellipsoid"),
        FamilyConfig(base="rosenbrock"),
        FamilyConfig(base="rastrigin"),
        FamilyConfig(base="ackley"),
        FamilyConfig(base="sharp_ridge"),
    ]
    return make_family_instances(families, dims=dims, n_instances=n_instances, seed=seed)


def make_constrained_battery(
    dims: Sequence[int] = (2, 5),
    n_instances: int = 3,
    seed: int = DEFAULT_BATTERY_SEED,
) -> FamilyInstances:
    """Constrained battery: 4 families x ``dims`` x ``n_instances``, ``k = 1, 2, 3``.

    The instance index cycles the number of constraints, so each family
    contributes one 1-constraint, one 2-constraint and one 3-constraint
    instance.  The first constraint of every instance is exactly active
    at the optimum (see :class:`~panobbgo.lib.families.Family`), so the
    feasible region always touches the optimum and the constraints are
    never decoration.

    Two families use linear constraints (half-spaces through the
    optimum), two use balls whose surface passes through it — a curved
    active set is a materially different thing to satisfy than a flat
    one, and ``classic.py`` only has the curved case in fixed 2-D
    problems (``Simionescu``, ``MishraBird``).

    ``d = 10`` is left out: nothing in the constraint-handling code has
    ever been measured on AOCC at all, and a first battery that is cheap
    enough to run repeatedly is worth more than a wide one.
    """
    ks = (1, 2, 3)
    families: List[FamilyLike] = [
        FamilyConfig(base="sphere", n_constraints=ks, constraint_kind="linear"),
        FamilyConfig(base="ellipsoid", n_constraints=ks, constraint_kind="ball"),
        FamilyConfig(base="rosenbrock", n_constraints=ks, constraint_kind="linear"),
        FamilyConfig(base="rastrigin", n_constraints=ks, constraint_kind="ball"),
    ]
    return make_family_instances(families, dims=dims, n_instances=n_instances, seed=seed)


# ---------------------------------------------------------------------------
# Atomic run
# ---------------------------------------------------------------------------


def _run_one(
    strategy_spec: StrategySpec,
    problem: Family,
    rep: int,
    budget: int,
    seed: int,
    log_lo: float,
    log_hi: float,
    sync_eval: bool,
) -> IOHRunRecord:
    """Run one strategy on one family instance; mirror of ``harness_ioh._run_one``."""
    t0 = time.time()
    err: Optional[str] = None
    n_evals = 0
    best_fx = float("inf")
    f_opt = float(problem.f_opt)
    trace_evals: List[int] = []
    trace_fx: List[float] = []
    score = 0.0

    try:
        np.random.seed(seed)
        tracker = PenaltyTracker(problem, budget=budget)
        try:
            # Budget before construction: ``NP_init="auto"`` sizes itself
            # from ``config.max_eval`` in the heuristic's constructor.
            strategy = strategy_spec.create_strategy(problem, seed=seed, max_eval=budget)
            strategy.config.max_eval = budget
            strategy.config.sync_evaluation = bool(sync_eval)
            # Anytime metric: the tracker's budget is the only stop.
            strategy.config.stop_on_convergence = False
            try:
                strategy.start()
            except _BudgetExhausted:
                pass
        finally:
            tracker.restore()

        n_evals = tracker.n_evals
        best_fx = tracker.best_fx
        score = aocc(tracker.best_so_far, f_opt=f_opt, log_lo=log_lo, log_hi=log_hi, budget=budget)
        trace_evals, trace_fx = _downsample_trajectory(tracker.best_so_far, budget=budget)
    except Exception as e:  # noqa: BLE001 — record and continue, as the IOH track does
        err = f"{type(e).__name__}: {e}"

    return IOHRunRecord(
        problem_kind=problem.family,
        dim=problem.dim,
        instance=problem.instance,
        strategy_name=strategy_spec.name,
        rep=rep,
        budget=budget,
        n_evals=n_evals,
        best_fx=best_fx,
        f_opt=f_opt,
        aocc=score,
        elapsed_s=time.time() - t0,
        seed=seed,
        error=err,
        trace_evals=trace_evals,
        trace_fx=trace_fx,
    )


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


def run_family_harness(
    specs: Sequence[StrategySpec],
    instances: Sequence[Tuple[str, Family]],
    *,
    budget_multiplier: int = 500,
    base_seed: int = 42,
    sync_eval: bool = True,
    reps: int = 1,
    log_lo: float = AOCC_LOG_LO,
    log_hi: float = AOCC_LOG_HI,
    progress: bool = True,
    battery_name: str = "families",
) -> IOHHarnessResult:
    """Score every spec on every instance and return an AOCC result.

    Parameters
    ----------
    specs
        Strategy specs.  The per-run seed is derived from
        ``spec.rng_identity`` (``seed_name or name``), so variants of one
        arm that share a ``seed_name`` run on the identical RNG stream
        per cell and their delta carries only the parameter's own effect
        (``DISCOVERY_2026-09-09.md`` §18).
    instances
        ``(name, problem)`` pairs from
        :func:`~panobbgo.lib.families.make_family_instances` or one of
        the presets above.
    budget_multiplier
        Budget per run is ``budget_multiplier * problem.dim``, as in
        :meth:`IOHBatterySpec.budget_for
        <panobbgo.harness_ioh.IOHBatterySpec.budget_for>`.
    base_seed
        Seed of the *optimiser*, not of the problems.
    sync_eval
        Default ``True`` here (the IOH track defaults to ``False`` for
        backwards compatibility): deterministic result batches roughly
        halve run-to-run measurement noise, and there is no historical
        family-battery number to stay comparable with.
    reps
        Independent repetitions per (instance, spec).

    Returns
    -------
    IOHHarnessResult
        Each run's ``problem_kind`` is the *family* label and ``instance``
        the instance index, so the ``f"{kind}_d{dim}_i{inst}"`` key used
        throughout the IOH analysis code stays unique.

    Runs serially; the strategies use internal threading.
    """
    total = len(instances) * len(specs) * int(reps)
    runs: List[IOHRunRecord] = []
    idx = 0
    for _name, problem in instances:
        budget = int(budget_multiplier) * problem.dim
        for spec in specs:
            for rep in range(int(reps)):
                idx += 1
                seed = _derive_seed(base_seed, problem.family, problem.dim, problem.instance, spec.rng_identity, rep)
                if progress:
                    print(
                        f"  [{idx:>3d}/{total:>3d}] {problem.family:<18s} "
                        f"dim={problem.dim:<2d} inst={problem.instance:<2d} rep={rep} {spec.name}",
                        flush=True,
                    )
                rec = _run_one(
                    strategy_spec=spec,
                    problem=problem,
                    rep=rep,
                    budget=budget,
                    seed=seed,
                    log_lo=log_lo,
                    log_hi=log_hi,
                    sync_eval=sync_eval,
                )
                if progress:
                    tag = "ERR " if rec.error else ""
                    print(
                        f"      {tag}AOCC={rec.aocc:.4f}  evals={rec.n_evals}/{budget}  "
                        f"prec={rec.precision:.3e}  t={rec.elapsed_s:.1f}s" + (f"  ({rec.error})" if rec.error else ""),
                        flush=True,
                    )
                runs.append(rec)

    return IOHHarnessResult(
        battery_name=battery_name,
        problem_kind="families",
        log_lo=log_lo,
        log_hi=log_hi,
        sync_eval=sync_eval,
        runs=runs,
    )


def describe_instances(instances: Sequence[Tuple[str, Family]]) -> Dict[str, Any]:
    """Summary of a battery: sizes, families, dims, constraint counts.

    Useful in a report header, and in a test that wants to assert the
    shape of a preset without running it.
    """
    dims = sorted({p.dim for _n, p in instances})
    families = sorted({p.family for _n, p in instances})
    ks = sorted({p.n_constraints for _n, p in instances})
    return {
        "n_instances": len(instances),
        "families": families,
        "dims": dims,
        "n_constraints": ks,
        "constrained": any(k > 0 for k in ks),
    }
