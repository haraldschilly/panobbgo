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
IOHprofiler-driven benchmark harness
====================================

Multi-instance, multi-strategy scoring of Panobbgo against problems
from the IOHprofiler suite — in particular, the MA-BBOB Anytime
competition family.  This is a parallel measurement track to
:mod:`panobbgo.harness`: the existing harness produces ``composite_score``
on Panobbgo's own problem battery (a stable internal contract), while
this module produces **mean AOCC** on IOH problems (the metric the
external community competes on).

Why a separate harness?

* ``composite_score`` is a "fraction of budget left when we solved" metric
  with a fixed tolerance.  AOCC (Area Over the Convergence Curve) is a
  log-precision area metric defined over a target range ``[1e-8, 1e2]``
  with no notion of "solved/unsolved".  They reward different things and
  do not interconvert cleanly.
* The IOH suite ships its own problem instances (and the MA-BBOB
  competition draws from a documented instance distribution); they are
  not in Panobbgo's own problem registry and there is no need to mix the
  registries.
* Forking the measurement track keeps the existing self-improvement
  ledger (``planning/self_improve_ledger.jsonl``) honest: a change can be
  good for ``composite_score`` and bad for AOCC, and we want to see both.

Public surface
--------------

* :class:`IOHBatterySpec` — declares the (problem kind, dim, instances,
  budget multiplier, reps) cube to evaluate.
* :func:`make_quick_battery` / :func:`make_standard_battery` /
  :func:`make_full_battery` — preset batteries matching the
  ``quick``/``standard``/``full`` modes of the legacy harness.
* :class:`IOHRunRecord` — per (problem kind, dim, instance, strategy, rep)
  result, including the convergence trajectory.
* :class:`IOHHarnessResult` — aggregate over a battery, with mean AOCC and
  per-pair detail.  Serialises to JSON for diffing.
* :func:`run_ioh_harness` — main entry point.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from panobbgo.benchmark import StrategySpec
from panobbgo.ioh_runner import AOCC_LOG_HI, AOCC_LOG_LO, IOHTracker, _BudgetExhausted, aocc  # noqa: F401


# ---------------------------------------------------------------------------
# Adapter: IOHHarnessResult -> HarnessResult (for the self-improvement loop)
# ---------------------------------------------------------------------------
#
# The composite-score harness machinery (statistical_accept, the ledger
# format, ProblemStrategyResult.compute_metrics, ...) operates on
# :class:`~panobbgo.harness.HarnessResult` with per-run "solve fractions"
# derived from ``first_success_eval``.  To reuse that machinery for the
# AOCC track, we encode each (problem, strategy, rep) AOCC as a synthetic
# :class:`~panobbgo.harness.RunRecord` whose only convergence event sits
# at evaluation ``k* = round((1 - aocc) * budget) + 1`` with
# ``func_distance = 0``.  Then ``_solve_fractions`` reads back exactly
# ``aocc`` from that run and the bootstrap CI on the composite delta
# operates on AOCC values without any further code duplication.
#
# Trade-offs:
#   * `ert` and the convergence trace in the encoded result are
#     meaningless under this scheme; only ``score`` /
#     ``composite_score`` (== mean AOCC) carries semantics.
#   * The encoded result is **not** for human consumption — it exists
#     to plug AOCC measurements into the existing statistical /
#     ledger machinery.  Use :class:`IOHHarnessResult` directly when
#     reporting numbers to a user.


def aocc_to_harness_result(ioh_result, mode: str = "ioh", base_seed: int = 42):
    """Encode an :class:`IOHHarnessResult` as a :class:`HarnessResult`.

    See module-level note: only ``composite_score`` and the per-pair
    ``score`` carry meaning in the returned object; convergence traces
    and ERT are synthetic.

    Parameters
    ----------
    ioh_result
        Result returned by :func:`run_ioh_harness`.
    mode, base_seed
        Forwarded to the synthetic :class:`HarnessConfig` so the wrapper
        carries enough metadata for downstream serialisation.
    """
    from panobbgo.harness import (
        ConvergencePoint,
        HarnessConfig,
        HarnessResult,
        ProblemStrategyResult,
        RunRecord,
    )

    # Group runs by (problem-key, strategy_name) → produces one
    # ProblemStrategyResult per pair.  Problem key encodes
    # (problem_kind, dim, instance) so different (dim, instance)
    # tuples appear as different "problems" in the harness sense.
    pair_buckets: Dict[Tuple[str, str], List[Any]] = {}
    pair_meta: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in ioh_result.runs:
        pname = f"{r.problem_kind}_d{r.dim}_i{r.instance}"
        key = (pname, r.strategy_name)
        pair_buckets.setdefault(key, []).append(r)
        pair_meta[key] = {
            "dim": r.dim,
            "budget": r.budget,
            "f_opt": r.f_opt,
        }

    psr_list: List[ProblemStrategyResult] = []
    for (pname, sname), runs in pair_buckets.items():
        meta = pair_meta[(pname, sname)]
        budget = int(meta["budget"])
        run_records: List[RunRecord] = []
        for run in runs:
            score = max(0.0, min(1.0, float(run.aocc)))
            # Map AOCC -> first_success_eval so _solve_fractions reads
            # it back unchanged.  Inverting frac = 1 - (k* - 1) / budget:
            #   k* = round((1 - aocc) * budget) + 1, clipped to [1, budget].
            k_star = max(1, min(budget, int(round((1.0 - score) * budget)) + 1))
            conv = [
                ConvergencePoint(
                    eval_idx=k_star,
                    fx=float(run.best_fx),
                    func_distance=0.0,
                )
            ]
            run_records.append(
                RunRecord(
                    problem_name=pname,
                    problem_dim=int(meta["dim"]),
                    strategy_name=sname,
                    rep=int(run.rep),
                    seed=int(run.seed),
                    budget=budget,
                    evaluations_used=int(run.n_evals),
                    best_fx=float(run.best_fx),
                    f_opt=float(meta["f_opt"]),
                    func_distance=0.0,  # synthetic — see note above
                    tolerance=1e-9,
                    success=True,
                    convergence=conv,
                    heuristic_counts={},
                    duration=float(run.elapsed_s),
                    error=run.error,
                )
            )
        psr = ProblemStrategyResult(
            problem_name=pname,
            problem_dim=int(meta["dim"]),
            strategy_name=sname,
            f_opt=float(meta["f_opt"]),
            tolerance=1e-9,
            budget=budget,
            runs=run_records,
        )
        psr.compute_metrics()  # populates .score = mean AOCC by construction
        psr_list.append(psr)

    # The synthetic HarnessConfig only needs the mode / base_seed /
    # budget so downstream serialisation has consistent metadata.
    fake_cfg = HarnessConfig(mode=mode, seed=base_seed, budget=None, reps=None)

    composite = float(np.mean([p.score for p in psr_list])) if psr_list else 0.0
    return HarnessResult(
        config=fake_cfg,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ioh_result.timestamp)),
        total_runs=len(ioh_result.runs),
        total_duration=float(sum(r.elapsed_s for r in ioh_result.runs)),
        problem_strategy_results=psr_list,
        composite_score=composite,
    )


# ---------------------------------------------------------------------------
# Problem-kind registry
# ---------------------------------------------------------------------------
#
# The ``ioh`` C++ binding is *not* imported in this process — it lives in
# the isolated child venv under ``tools/ioh_worker/`` and is reached via
# :class:`~panobbgo.lib.ioh_wrapper.IOHProblem`.  All this module knows
# about a problem kind is the short tag and which extra kwargs the worker
# needs to build it.


#: Recognised problem-kind tags.  These are forwarded to the worker as
#: the ``kind`` field of the ``create`` JSON-Lines request, where the
#: actual ``ioh`` builder lives.  Keep in sync with
#: ``tools/ioh_worker/src/ioh_worker/__main__.py::_build_problem``.
SUPPORTED_PROBLEM_KINDS: Tuple[str, ...] = ("MA-BBOB", "BBOB")


# ---------------------------------------------------------------------------
# Battery / spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IOHBatterySpec:
    """A (problem kind, dims, instances, budget) cube to evaluate.

    Parameters
    ----------
    name
        Human-readable battery name shown in reports.
    problem_kind
        One of :data:`SUPPORTED_PROBLEM_KINDS`.  ``"MA-BBOB"`` is the
        MA-BBOB anytime competition family.
    dims
        Tuple of dimensions to evaluate.  The competition uses ``(2, 5)``.
    instances
        Tuple of instance ids.  ``range(N)`` becomes ``tuple(range(N))``.
    reps
        Independent repetitions per (dim, instance, strategy).  Each rep
        uses a SHA-256-derived seed so before/after runs see the same
        problem realisation but different RNG state across reps.
    budget_multiplier
        Evaluation budget per run = ``budget_multiplier * dim``.  The
        MA-BBOB anytime rules use ``2000``.
    extra_builder_kwargs
        Optional dict of additional kwargs passed to the builder
        (e.g. ``{"fid": 1}`` for the BBOB sphere).
    """

    name: str
    problem_kind: str
    dims: Tuple[int, ...]
    instances: Tuple[int, ...]
    reps: int = 1
    budget_multiplier: int = 2000
    extra_builder_kwargs: Tuple[Tuple[str, Any], ...] = ()

    def pair_count(self, n_strategies: int) -> int:
        return n_strategies * len(self.dims) * len(self.instances) * self.reps

    def budget_for(self, dim: int) -> int:
        return self.budget_multiplier * dim

    def builder_kwargs(self) -> Dict[str, Any]:
        return dict(self.extra_builder_kwargs)


def make_quick_battery() -> IOHBatterySpec:
    """Small battery for fast iteration (~seconds, used in tests and CI)."""
    return IOHBatterySpec(
        name="ioh-quick",
        problem_kind="MA-BBOB",
        dims=(2,),
        instances=(0, 1, 2),
        reps=1,
        budget_multiplier=100,  # 100 * 2 = 200 evals; ~1s per run
    )


def make_standard_battery() -> IOHBatterySpec:
    """Mid-sized battery: a meaningful AOCC estimate without overnight runs."""
    return IOHBatterySpec(
        name="ioh-standard",
        problem_kind="MA-BBOB",
        dims=(2, 5),
        instances=tuple(range(5)),
        reps=1,
        budget_multiplier=500,  # 500 * d evals
    )


def make_full_battery() -> IOHBatterySpec:
    """Competition-budget battery: 2000*d evals, 10 instances per dim.

    The actual MA-BBOB anytime competition uses many more instances; this
    is the largest preset that still finishes on a developer machine in
    minutes rather than hours.  Use a custom :class:`IOHBatterySpec` for
    a serious pre-submission run.
    """
    return IOHBatterySpec(
        name="ioh-full",
        problem_kind="MA-BBOB",
        dims=(2, 5),
        instances=tuple(range(10)),
        reps=1,
        budget_multiplier=2000,  # competition budget
    )


# ---------------------------------------------------------------------------
# IOH-tuned strategy registry
# ---------------------------------------------------------------------------
#
# These specs are tuned for the *anytime* metric (AOCC), in contrast to
# the strategy registry in :mod:`panobbgo.harness` which is tuned for
# ``composite_score`` (find-and-stop).  The key differences:
#
# 1.  Always include the :class:`~panobbgo.analyzers.Restart` analyzer so
#     stagnation triggers re-diversification rather than a flat tail
#     (the strategy's ``stop_on_convergence`` is already disabled by the
#     IOH runner — see :func:`_run_one` below — but Restart actively
#     *uses* the budget by jumping to a fresh basin).
# 2.  A larger Sobol initial design relative to the budget: spending the
#     first 5-10% of budget on a low-discrepancy sweep pays off on the
#     anytime metric, which weights early-budget exploration heavily on
#     the log-precision target scale.
# 3.  Heuristics that implement ``on_restart`` (Random, Nearby, NelderMead,
#     Sobol, ...) participate; others degrade gracefully.


def make_ioh_strategies() -> List[StrategySpec]:
    """IOH-tuned strategy registry — primary entry point for AOCC runs.

    Returns a small list rather than the full panobbgo zoo so each
    iteration of the harness stays cheap.  Add baselines via
    :func:`panobbgo.harness_baselines.make_baseline_strategies` if you
    need an absolute reference.
    """
    from panobbgo.analyzers import Sensitivity
    from panobbgo.heuristics import Center, Nearby, NelderMead, Random, Sobol
    from panobbgo.strategies import StrategyRewarding, StrategyRoundRobin

    return [
        # Pure-random reference inside the panobbgo strategy framework.
        # Equivalent to the harness_baselines Random except it goes
        # through panobbgo's eventbus and Splitter, so it's a fair
        # apples-to-apples baseline for any improvements above.
        StrategySpec(
            name="RoundRobin_Random",
            strategy_class=StrategyRoundRobin,
            heuristics=[(Random, {})],
        ),
        # Adaptive heuristic mix.  This is the working candidate for the
        # MA-BBOB competition entry: roughly the same heuristics as
        # ``Rewarding_Diverse`` from the composite-score harness.  The
        # Restart analyzer it originally shipped with was dropped by the
        # 2026-07-30 codify (18 confirmed drop_analyzer accepts across 17
        # distinct nights on the aocc ledger, pooled CI95%
        # [+0.0084, +0.0147]) — under the anytime AOCC metric the
        # diverse-restart jumps cost more mid-budget precision than the
        # basin-hopping recovers.  The spec name is kept for ledger /
        # bandit-arm continuity.
        StrategySpec(
            name="Rewarding_Restart",
            strategy_class=StrategyRewarding,
            heuristics=[
                (Sobol, {"n": 32, "scramble": True}),
                (Random, {}),
                (Nearby, {"radius": 0.1, "axes": "all", "new": 3}),
                (Center, {}),
                (NelderMead, {}),
            ],
            analyzers=[
                (Sensitivity, {"update_interval": 25}),
            ],
        ),
    ]


# ---------------------------------------------------------------------------
# Run records & aggregate result
# ---------------------------------------------------------------------------


@dataclass
class IOHRunRecord:
    """Result of one (problem kind, dim, instance, strategy, rep) run."""

    problem_kind: str
    dim: int
    instance: int
    strategy_name: str
    rep: int
    budget: int
    n_evals: int
    best_fx: float
    f_opt: float
    aocc: float
    elapsed_s: float
    seed: int
    error: Optional[str] = None
    # Down-sampled convergence trace so JSON dumps don't blow up:
    # store best_fx at evenly-spaced budget fractions.
    trace_evals: List[int] = field(default_factory=list)
    trace_fx: List[float] = field(default_factory=list)

    @property
    def precision(self) -> float:
        return float(self.best_fx - self.f_opt)


@dataclass
class IOHHarnessResult:
    """Aggregate over a battery."""

    battery_name: str
    problem_kind: str
    log_lo: float
    log_hi: float
    runs: List[IOHRunRecord]
    timestamp: float = field(default_factory=time.time)

    @property
    def mean_aocc(self) -> float:
        scores = [r.aocc for r in self.runs if r.error is None]
        return float(np.mean(scores)) if scores else 0.0

    def per_strategy_aocc(self) -> Dict[str, float]:
        """Return ``{strategy_name: mean AOCC over all runs}``."""
        by_strat: Dict[str, List[float]] = {}
        for r in self.runs:
            if r.error is not None:
                continue
            by_strat.setdefault(r.strategy_name, []).append(r.aocc)
        return {k: float(np.mean(v)) for k, v in by_strat.items()}

    def per_strategy_per_dim_aocc(self) -> Dict[Tuple[str, int], float]:
        by: Dict[Tuple[str, int], List[float]] = {}
        for r in self.runs:
            if r.error is not None:
                continue
            by.setdefault((r.strategy_name, r.dim), []).append(r.aocc)
        return {k: float(np.mean(v)) for k, v in by.items()}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "battery_name": self.battery_name,
            "problem_kind": self.problem_kind,
            "log_lo": self.log_lo,
            "log_hi": self.log_hi,
            "mean_aocc": self.mean_aocc,
            "per_strategy_aocc": self.per_strategy_aocc(),
            "timestamp": self.timestamp,
            "runs": [asdict(r) for r in self.runs],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=float)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "IOHHarnessResult":
        runs = [IOHRunRecord(**r) for r in d.get("runs", [])]
        return cls(
            battery_name=d["battery_name"],
            problem_kind=d["problem_kind"],
            log_lo=d.get("log_lo", AOCC_LOG_LO),
            log_hi=d.get("log_hi", AOCC_LOG_HI),
            runs=runs,
            timestamp=d.get("timestamp", time.time()),
        )

    def print_summary(self) -> None:
        print(f"\nIOH battery: {self.battery_name}  ({self.problem_kind})")
        print(f"  log targets:  [10^{self.log_lo:.0f}, 10^{self.log_hi:.0f}]")
        print(f"  mean AOCC:    {self.mean_aocc:.4f}    over {len(self.runs)} run(s)")
        print("\n  per strategy:")
        for name, val in sorted(self.per_strategy_aocc().items(), key=lambda kv: -kv[1]):
            print(f"    {name:32s}  {val:.4f}")
        per_dim = self.per_strategy_per_dim_aocc()
        dims = sorted({d for _, d in per_dim})
        if len(dims) > 1:
            print("\n  per (strategy, dim):")
            strategies = sorted({s for s, _ in per_dim})
            header = "    " + "strategy".ljust(32) + "  " + "  ".join(f"  d={d:<2d}" for d in dims)
            print(header)
            for s in strategies:
                row = (
                    "    " + s.ljust(32) + "  " + "  ".join(f"  {per_dim.get((s, d), float('nan')):.4f}" for d in dims)
                )
                print(row)


# ---------------------------------------------------------------------------
# Seed derivation — same SHA-256 scheme as panobbgo.harness
# ---------------------------------------------------------------------------


def _derive_seed(base_seed: int, problem_kind: str, dim: int, instance: int, strategy_name: str, rep: int) -> int:
    payload = f"{base_seed}|{problem_kind}|{dim}|{instance}|{strategy_name}|{rep}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


# ---------------------------------------------------------------------------
# Trajectory down-sampling
# ---------------------------------------------------------------------------


def _downsample_trajectory(traj: Sequence[float], budget: int, k: int = 32) -> Tuple[List[int], List[float]]:
    """Return ``k`` log-spaced (eval_idx, best_fx) samples from a trajectory.

    Log spacing emphasises early-budget behaviour, which is what an
    anytime metric values.  The final budget step is always included.
    """
    if len(traj) == 0:
        return [], []
    arr = np.asarray(traj, dtype=np.float64)
    last_idx = len(arr) - 1
    # log-spaced indices from 1..len(arr), unique & sorted
    raw = np.unique(np.round(np.geomspace(1, max(last_idx + 1, 2), num=k)).astype(int))
    raw = np.clip(raw, 1, last_idx + 1) - 1
    eval_indices = [int(i + 1) for i in raw]
    fxs = [float(arr[i]) for i in raw]
    # pad up to budget with the final value so plots show the flat tail
    if last_idx + 1 < budget:
        eval_indices.append(budget)
        fxs.append(float(arr[last_idx]))
    return eval_indices, fxs


# ---------------------------------------------------------------------------
# Atomic run
# ---------------------------------------------------------------------------


def _run_one(
    strategy_spec: StrategySpec,
    problem_kind: str,
    dim: int,
    instance: int,
    rep: int,
    budget: int,
    seed: int,
    builder_kwargs: Dict[str, Any],
    log_lo: float,
    log_hi: float,
    timeout_s: Optional[float] = None,
) -> IOHRunRecord:
    """Run one strategy on one (problem, instance) and return its record."""
    from panobbgo.lib.ioh_wrapper import IOHProblem

    if problem_kind not in SUPPORTED_PROBLEM_KINDS:
        raise ValueError(f"Unknown problem kind {problem_kind!r}; known: {list(SUPPORTED_PROBLEM_KINDS)}")

    t0 = time.time()
    err: Optional[str] = None
    n_evals = 0
    best_fx = float("inf")
    f_opt = 0.0
    trace_evals: List[int] = []
    trace_fx: List[float] = []
    score = 0.0
    wrapped: Optional[IOHProblem] = None

    try:
        wrapped = IOHProblem(
            kind=problem_kind,
            instance=instance,
            dim=dim,
            **builder_kwargs,
        )
        f_opt = float(wrapped.optimum_y)

        np.random.seed(seed)

        tracker = IOHTracker(wrapped, budget=budget)
        try:
            strategy = strategy_spec.create_strategy(wrapped)
            strategy.config.max_eval = budget
            # IOH/AOCC is an anytime metric: stopping early on convergence
            # leaves the remaining budget penalised at the final best-fx.
            # Force the strategy to keep producing points until the
            # tracker enforces the budget hard-stop.  (The `Convergence`
            # analyzer still fires its event for any listeners; we simply
            # tell the strategy not to honour it as a stop signal.)
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
    except Exception as e:  # noqa: BLE001  — record and continue
        err = f"{type(e).__name__}: {e}"
    finally:
        if wrapped is not None:
            try:
                wrapped.close()
            except Exception:
                pass

    return IOHRunRecord(
        problem_kind=problem_kind,
        dim=dim,
        instance=instance,
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


def run_ioh_harness(
    strategies: Sequence[StrategySpec],
    battery: IOHBatterySpec,
    *,
    base_seed: int = 42,
    log_lo: float = AOCC_LOG_LO,
    log_hi: float = AOCC_LOG_HI,
    timeout_s: Optional[float] = None,
    progress: bool = True,
) -> IOHHarnessResult:
    """Run every strategy against every (dim, instance, rep) in ``battery``.

    Runs serially; the underlying strategies use internal threading.  For
    large batteries, drive multiple ``run_ioh_harness`` calls from outside
    if you need outer parallelism.
    """
    if battery.problem_kind not in SUPPORTED_PROBLEM_KINDS:
        raise ValueError(f"Unknown problem kind {battery.problem_kind!r}; known: {list(SUPPORTED_PROBLEM_KINDS)}")
    builder_kwargs = battery.builder_kwargs()
    total = battery.pair_count(len(strategies))
    runs: List[IOHRunRecord] = []
    idx = 0
    for dim in battery.dims:
        budget = battery.budget_for(dim)
        for instance in battery.instances:
            for spec in strategies:
                for rep in range(battery.reps):
                    idx += 1
                    seed = _derive_seed(base_seed, battery.problem_kind, dim, instance, spec.name, rep)
                    if progress:
                        print(
                            f"  [{idx:>3d}/{total:>3d}] {battery.problem_kind} "
                            f"dim={dim:<2d} inst={instance:<2d} rep={rep} "
                            f"{spec.name}",
                            flush=True,
                        )
                    rec = _run_one(
                        strategy_spec=spec,
                        problem_kind=battery.problem_kind,
                        dim=dim,
                        instance=instance,
                        rep=rep,
                        budget=budget,
                        seed=seed,
                        builder_kwargs=builder_kwargs,
                        log_lo=log_lo,
                        log_hi=log_hi,
                        timeout_s=timeout_s,
                    )
                    if progress:
                        tag = "ERR " if rec.error else ""
                        print(
                            f"      {tag}AOCC={rec.aocc:.4f}  evals={rec.n_evals}/{budget}  "
                            f"prec={rec.precision:.3e}  t={rec.elapsed_s:.1f}s"
                            + (f"  ({rec.error})" if rec.error else ""),
                            flush=True,
                        )
                    runs.append(rec)

    return IOHHarnessResult(
        battery_name=battery.name,
        problem_kind=battery.problem_kind,
        log_lo=log_lo,
        log_hi=log_hi,
        runs=runs,
    )
