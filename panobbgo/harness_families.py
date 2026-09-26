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
existing analysis code (``paired_seed_stats``,
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
  two frozen presets; :func:`make_shapes_battery` (the BBOB shapes) and
  :func:`make_failure_battery` (failure regions) — two more.  Every
  preset takes ``dims=``, including 30 and 40.
* :func:`make_large_families_battery` — the free and shapes families at
  *d* = 30 and 40 (opt-in).
* :func:`make_sealed_families_battery` — the **sealed test set**
  (:mod:`panobbgo.sealed`), for claims only.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from panobbgo.benchmark import StrategySpec
from panobbgo.local_run import BLAS_THREADS
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
from panobbgo.lib.families import FailureRegion, Family, FamilyConfig, make_family_instances
from panobbgo.sealed import SEALED_FAMILY_SEED, print_sealed_banner
from panobbgo.virtual_clock import VirtualSpec

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

    def __init__(
        self,
        problem: Any,
        budget: int,
        *,
        rho: float = PENALTY_RHO,
        hard: bool = False,
        timeout_s: Optional[float] = None,
    ) -> None:
        super().__init__(problem, budget, hard=hard, timeout_s=timeout_s)
        self.rho = float(rho)
        #: Objective value at the best *penalty* point (for reporting).
        self.best_raw_fx: float = float("inf")
        #: Constraint violation at the best penalty point.
        self.best_cv: float = float("inf")

    def _measure(self, x: np.ndarray) -> Tuple[float, ...]:
        fx = float(self._orig_eval(x))
        cv_vec = self.problem.eval_constraints(x)
        if cv_vec is None:
            cv = 0.0
        else:
            positive = np.asarray(cv_vec, dtype=np.float64)
            positive = positive[positive > 0.0]
            cv = float(np.linalg.norm(positive)) if positive.size else 0.0
        return fx, fx + self.rho * cv, cv

    def _record(self, x: np.ndarray, measured: Tuple[float, ...]) -> None:
        fx, phi, cv = measured
        if np.isfinite(phi) and phi < self.best_fx:
            self.best_fx = phi
            self.best_raw_fx = fx
            self.best_cv = cv
            self.best_x = np.asarray(x, dtype=np.float64).copy()
        self.best_so_far.append(self.best_fx)


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------
#
# The free and constrained presets are frozen contracts in the sense of
# ``planning/GOAL.md`` §4: extend by composing a new instance list, do not
# edit these (the shapes and failure presets below are new, 2026-09-26).  The
# instance *seed* is fixed (DEFAULT_BATTERY_SEED) so the problems are the
# same on every run; only ``base_seed`` — the optimiser's RNG — moves.


#: The families of each preset, in battery order.  Shared by the
#: development presets and the sealed set, so the two measure the same
#: classes on different instances.
FREE_FAMILIES: Tuple[FamilyConfig, ...] = (
    FamilyConfig(base="ellipsoid"),
    FamilyConfig(base="rosenbrock"),
    FamilyConfig(base="rastrigin"),
    FamilyConfig(base="ackley"),
    FamilyConfig(base="sharp_ridge"),
)
_KS = (1, 2, 3)
CONSTRAINED_FAMILIES: Tuple[FamilyConfig, ...] = (
    FamilyConfig(base="sphere", n_constraints=_KS, constraint_kind="linear"),
    FamilyConfig(base="ellipsoid", n_constraints=_KS, constraint_kind="ball"),
    FamilyConfig(base="rosenbrock", n_constraints=_KS, constraint_kind="linear"),
    FamilyConfig(base="rastrigin", n_constraints=_KS, constraint_kind="ball"),
)
SHAPES_FAMILIES: Tuple[FamilyConfig, ...] = (
    FamilyConfig(base="lunacek_bi_rastrigin"),
    FamilyConfig(base="gallagher"),
    FamilyConfig(base="attractive_sector"),
    FamilyConfig(base="step_ellipsoid"),
    FamilyConfig(base="bent_cigar"),
)
FAILURE_FAMILIES: Tuple[FamilyConfig, ...] = (
    FamilyConfig(base="ellipsoid", failure=FailureRegion("halfspace", share=0.25, mode="crash", boundary_gap=0.0)),
    FamilyConfig(base="rosenbrock", failure=FailureRegion("halfspace", share=0.25, mode="timeout", boundary_gap=0.05)),
    FamilyConfig(base="rastrigin", failure=FailureRegion("ball", share=0.2, mode="crash")),
    FamilyConfig(base="sharp_ridge", failure=FailureRegion("boxes", share=0.2, mode="timeout", n_boxes=3)),
)


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
    return make_family_instances(list(FREE_FAMILIES), dims=dims, n_instances=n_instances, seed=seed)


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
    return make_family_instances(list(CONSTRAINED_FAMILIES), dims=dims, n_instances=n_instances, seed=seed)


def make_shapes_battery(
    dims: Sequence[int] = (2, 5, 10),
    n_instances: int = 3,
    seed: int = DEFAULT_BATTERY_SEED,
) -> FamilyInstances:
    """The BBOB shapes the free battery lacks: 5 families x ``dims`` x ``n_instances``.

    ``planning/DESIGN_suite_2026-09-14.md`` Gap 2, at BBOB's default knobs
    (see :data:`~panobbgo.lib.families.CONTEXT_BASES` for the knobs):

    * ``lunacek_bi_rastrigin`` (f24) — a deceptive double funnel: the
      class where restarts and portfolios should pay most.
    * ``gallagher`` (f21, 101 peaks) — random peaks, weak global structure.
    * ``attractive_sector`` (f6) — strongly asymmetric around the optimum.
    * ``step_ellipsoid`` (f7) — plateaus: ties for a ranking method, a
      systematically wrong smooth surrogate.
    * ``bent_cigar`` (f12) — one soft, curved direction.
    """
    return make_family_instances(list(SHAPES_FAMILIES), dims=dims, n_instances=n_instances, seed=seed)


def make_failure_battery(
    dims: Sequence[int] = (2, 5),
    n_instances: int = 3,
    seed: int = DEFAULT_BATTERY_SEED,
) -> FamilyInstances:
    """Failure regions (``DESIGN_roadmap_2026-09-26.md`` §4 D): 4 families x ``dims`` x ``n_instances``.

    Every shape and both modes of :class:`~panobbgo.lib.families.FailureRegion`,
    each on a base whose own difficulty is already in the free battery, so
    the failure region is the new variable:

    * ``ellipsoid`` + half-space, *crash*, the optimum **on** the boundary
      (``boundary_gap=0``): the stability-limit case.
    * ``rosenbrock`` + half-space, *timeout*, the boundary 5 % of the
      half-width past the optimum.

      With a ``boundary_gap`` the boundary is tied to ``x_opt``, so the
      realised share of these two is *not* 25 %: it is the share closest
      to 25 % among the 2d axis/side choices the optimum allows, and varies
      per instance (``Family.failure_share``; roughly 0.1–0.4 at d = 2).
    * ``rastrigin`` + ball, *crash*, 20 % of the box.
    * ``sharp_ridge`` + 3 boxes, *timeout*, 20 % of the box.

    Every failed call is spent budget (the AOCC trace counts it, without
    progress).  ``d = 10`` is left out for the reason the constrained
    battery leaves it out: a first battery cheap enough to run often.
    """
    return make_family_instances(list(FAILURE_FAMILIES), dims=dims, n_instances=n_instances, seed=seed)


def make_large_families_battery(
    dims: Sequence[int] = (30, 40),
    n_instances: int = 3,
    seed: int = DEFAULT_BATTERY_SEED,
) -> FamilyInstances:
    """The free and shapes families at *d* = 30 and 40: 10 families x ``dims`` x ``n_instances``.

    Opt-in (``family_screen.py preset=large``, ``ioh_benchmark.py run
    --families-large``); the frozen presets keep their dims.  The instances
    are exactly those :func:`make_families_battery` and
    :func:`make_shapes_battery` build when asked for ``dims=(30, 40)``: an
    instance seed depends on (battery seed, family, dim, index) only.

    The constrained and failure families scale to these dims as well
    (``family_screen.py preset=failure dims=30,40``); they are left out
    here because nothing in their handling has been measured above
    ``d = 5`` yet.

    Cost, measured 2026-09-26 (16-core laptop, ``nice -n 15``,
    ``sync_eval``, one BLAS thread, light load): a run at *d* = 40 and
    ``500·d`` (20 000 evaluations) takes 1.3-1.5 s for CMA-ES alone and
    2.4-3.4 s for the two portfolio specs of
    :func:`~panobbgo.harness_ioh.make_ioh_strategies`, 0.07-0.17 ms per
    evaluation.  The 60 runs of the preset are ≈ 1.2-3 min per strategy and
    seed.

    Resolution limit: at this budget CMA-ES scores AOCC exactly 0 on the
    ellipsoid and Lunacek families at *d* = 40 — its best value never gets
    below the ``1e2`` upper target.  Those cells rank nothing; read the
    preset per family, not only as a mean.
    """
    families = list(FREE_FAMILIES) + list(SHAPES_FAMILIES)
    return make_family_instances(families, dims=dims, n_instances=n_instances, seed=seed)


#: Dimensions of the sealed family set: unconstrained, and constrained or failing.
SEALED_FAMILY_DIMS: Tuple[int, ...] = (2, 5, 10, 20, 30, 40)
SEALED_FAMILY_DIMS_HARD: Tuple[int, ...] = (2, 5, 10)


def make_sealed_families_battery() -> FamilyInstances:
    """The sealed family test set: **for claims only, never for tuning.**

    The classes of every development preset on fresh instances (battery
    seed :data:`panobbgo.sealed.SEALED_FAMILY_SEED`, which
    :func:`~panobbgo.lib.families.make_family_instances` refuses outside
    this function):

    * the free and shapes families (10) at *d* ∈ {2, 5, 10, 20, 30, 40},
      3 instances each — 180 instances;
    * the constrained and failure families (8) at *d* ∈ {2, 5, 10},
      3 instances each — 72 instances.

    :func:`run_family_harness` prints a warning banner when it runs it,
    names the result ``sealed-…`` and marks every record ``sealed``; it
    refuses a sub-selection of the set, a mix with development instances,
    ``reps != 1`` and a budget other than ``500·d``.  Rules:
    ``doc/dev/benchmarking.md``, "The sealed test set".  Cost at ``500·d``
    (1.8 M evaluations per strategy and seed): ≈ 3-6 min per strategy and
    seed at 0.07-0.17 ms per evaluation (one BLAS thread).  No arguments, on
    purpose (see :func:`~panobbgo.harness_ioh.make_sealed_battery`).
    """
    easy = list(FREE_FAMILIES) + list(SHAPES_FAMILIES)
    hard = list(CONSTRAINED_FAMILIES) + list(FAILURE_FAMILIES)
    out = make_family_instances(easy, dims=SEALED_FAMILY_DIMS, n_instances=3, seed=SEALED_FAMILY_SEED, sealed=True)
    out += make_family_instances(
        hard, dims=SEALED_FAMILY_DIMS_HARD, n_instances=3, seed=SEALED_FAMILY_SEED, sealed=True
    )
    return out


#: Budget multiplier of the sealed family set.
SEALED_FAMILY_BUDGET_MULTIPLIER: int = 500


def sealed_family_keys() -> set:
    """``{(family, dim, instance)}`` of :func:`make_sealed_families_battery`, without building it."""
    easy = list(FREE_FAMILIES) + list(SHAPES_FAMILIES)
    hard = list(CONSTRAINED_FAMILIES) + list(FAILURE_FAMILIES)
    return {(cfg.name(), d, i) for cfg in easy for d in SEALED_FAMILY_DIMS for i in range(3)} | {
        (cfg.name(), d, i) for cfg in hard for d in SEALED_FAMILY_DIMS_HARD for i in range(3)
    }


def _check_sealed_run(instances: Sequence[Tuple[str, Family]], budget_multiplier: int, reps: int) -> bool:
    """``True`` for the whole sealed set; ``False`` for none of it; ``ValueError`` for anything between."""
    marks = [bool(getattr(p, "sealed", False)) for _n, p in instances]
    if not any(marks):
        return False
    if not all(marks):
        raise ValueError("a battery mixing sealed and development instances is refused (panobbgo.sealed)")
    keys = [(p.family, p.dim, p.instance) for _n, p in instances]
    if len(keys) != len(set(keys)) or set(keys) != sealed_family_keys():
        raise ValueError("the sealed family set runs whole: no sub-selection (panobbgo.sealed)")
    if int(budget_multiplier) != SEALED_FAMILY_BUDGET_MULTIPLIER or int(reps) != 1:
        raise ValueError(
            f"the sealed family set runs at {SEALED_FAMILY_BUDGET_MULTIPLIER}*d with reps=1, "
            f"not {budget_multiplier}*d, reps={reps}"
        )
    return True


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
    timeout_s: Optional[float] = None,
    virtual: Optional[VirtualSpec] = None,
) -> IOHRunRecord:
    """Run one strategy on one family instance; same driver as ``harness_ioh._run_one``."""
    t0 = time.time()
    f_opt = float(problem.f_opt)
    tracked = _TrackedRun(n_evals=0, best_fx=float("inf"), aocc=0.0, trace_evals=[], trace_fx=[])

    try:
        # The families are noiseless, so an oracle regime gate reads
        # ``"clean"`` (constrained-or-not it reads off the problem).
        tracked = _run_tracked(
            strategy_spec.with_regime_class("clean"),
            problem,
            PenaltyTracker(problem, budget=budget, timeout_s=timeout_s),
            f_opt=f_opt,
            budget=budget,
            seed=seed,
            sync_eval=sync_eval,
            log_lo=log_lo,
            log_hi=log_hi,
            timeout_s=timeout_s,
            virtual=virtual,
        )
    except Exception as e:  # noqa: BLE001 — record and continue, as the IOH track does
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
        f_opt=f_opt,
        aocc=tracked.aocc,
        elapsed_s=time.time() - t0,
        seed=seed,
        error=tracked.error,
        trace_evals=tracked.trace_evals,
        trace_fx=tracked.trace_fx,
        aocc_time=tracked.aocc_time,
        sealed=bool(getattr(problem, "sealed", False)),
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
    timeout_s: Optional[float] = None,
    jobs: int = 1,
    virtual: Optional[VirtualSpec] = None,
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
    timeout_s
        Per-run wall-clock deadline, as in :func:`run_ioh_harness
        <panobbgo.harness_ioh.run_ioh_harness>`: evaluations past it are
        not counted, the strategy is stopped, and the run keeps its AOCC
        up to the deadline but is recorded with a ``TimeoutError``.
    jobs
        ``> 1`` runs the (instance, spec, rep) cells in that many worker
        processes, as in :func:`run_ioh_harness
        <panobbgo.harness_ioh.run_ioh_harness>`; the records are the same
        for every ``jobs`` (except ``elapsed_s``) and in cell order.
    virtual
        Run on the virtual clock (:class:`~panobbgo.virtual_clock.VirtualSpec`)
        and score ``aocc_time`` too, as in :func:`run_ioh_harness
        <panobbgo.harness_ioh.run_ioh_harness>`.

    Returns
    -------
    IOHHarnessResult
        Each run's ``problem_kind`` is the *family* label and ``instance``
        the instance index, so the ``f"{kind}_d{dim}_i{inst}"`` key used
        throughout the IOH analysis code stays unique.

    Runs serially unless ``jobs > 1``; the strategies use internal threading.
    """
    sealed = _check_sealed_run(instances, budget_multiplier, reps)
    if sealed:
        if not battery_name.startswith("sealed"):
            battery_name = f"sealed-{battery_name}"
        print_sealed_banner(battery_name)
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
                        f"  [{idx:>3d}/{total:>3d}] {problem.family:<18s} "
                        f"dim={problem.dim:<2d} inst={problem.instance:<2d} rep={rep} {spec.name}",
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
        problem_kind="families",
        log_lo=log_lo,
        log_hi=log_hi,
        sync_eval=sync_eval or virtual is not None,
        runs=runs,
        virtual=None if virtual is None else virtual.to_dict(),
        sealed=sealed,
        blas_threads=BLAS_THREADS,
    )

    warn_missing_time_scores(result)
    return result


def describe_instances(instances: Sequence[Tuple[str, Family]]) -> Dict[str, Any]:
    """Summary of a battery: sizes, families, dims, constraint counts.

    Useful in a report header, and in a test that wants to assert the
    shape of a preset without running it.
    """
    dims = sorted({p.dim for _n, p in instances})
    families = sorted({p.family for _n, p in instances})
    ks = sorted({p.n_constraints for _n, p in instances})
    failures = sorted({p.failure.tag() for _n, p in instances if p.failure is not None})
    return {
        "n_instances": len(instances),
        "families": families,
        "dims": dims,
        "n_constraints": ks,
        "constrained": any(k > 0 for k in ks),
        "failure": failures,
    }
