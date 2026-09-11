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
* :func:`make_noisy_battery` / :func:`make_highdim_battery` /
  :func:`make_noisy_highdim_battery` — the regimes ``planning/GOAL.md``
  §2c asks for: noise on the objective (scored on the true value, BBOB
  noisy-suite style) and *d* ∈ {10, 20}.  Noise is applied in *this*
  process by :mod:`panobbgo.lib.noise`; the ``ioh`` worker is untouched.
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
from dataclasses import asdict, dataclass, field, replace
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


#: Problem-kind tags the *worker* understands.  These are forwarded as
#: the ``kind`` field of the ``create`` JSON-Lines request, where the
#: actual ``ioh`` builder lives.  Keep in sync with
#: ``tools/ioh_worker/src/ioh_worker/__main__.py::_build_problem``.
WORKER_PROBLEM_KINDS: Tuple[str, ...] = ("MA-BBOB", "BBOB")

#: Noisy problem kinds — ``tag -> (worker kind, noise-model tag)``.
#:
#: The noise is applied **in this process** by
#: :class:`~panobbgo.lib.noise.NoisyProblem`, on top of a plain worker
#: problem; the worker and its JSON-Lines protocol are untouched.  That
#: keeps the ``ioh`` child venv (pinned to Python 3.12) free of panobbgo
#: code and means a noisy battery costs exactly the same number of worker
#: round-trips as a noiseless one — the wrapper's ``eval_pair`` hands the
#: tracker the noisy *and* the true value from a single ``eval`` call.
#:
#: The noise level (``"moderate"`` / ``"severe"``, the BBOB f101–f106 vs.
#: f107–f130 settings) is a field of :class:`IOHBatterySpec`, not part of
#: the kind, so one battery preset can be re-run at a second level
#: without a second tag.
NOISY_PROBLEM_KINDS: Dict[str, Tuple[str, str]] = {
    "MA-BBOB-noisy-gauss": ("MA-BBOB", "gauss"),
    "MA-BBOB-noisy-unif": ("MA-BBOB", "unif"),
    "MA-BBOB-noisy-cauchy": ("MA-BBOB", "cauchy"),
}

#: Everything :func:`run_ioh_harness` accepts.
SUPPORTED_PROBLEM_KINDS: Tuple[str, ...] = WORKER_PROBLEM_KINDS + tuple(NOISY_PROBLEM_KINDS)


def resolve_problem_kind(kind: str) -> Tuple[str, Optional[str]]:
    """Split a battery's ``problem_kind`` into ``(worker kind, noise tag)``.

    ``("MA-BBOB", None)`` for a noiseless kind,
    ``("MA-BBOB", "gauss")`` for ``"MA-BBOB-noisy-gauss"``.
    """
    if kind in NOISY_PROBLEM_KINDS:
        return NOISY_PROBLEM_KINDS[kind]
    return kind, None


#: Noise-model tag -> the regime class an oracle regime gate takes as given
#: (:data:`panobbgo.strategies.blocks.NOISE_CLASSES`).  gauss and unif are
#: *one* class: a probe cannot separate them (design §1.3) and §42 sends
#: both to the same arm set.  ``None`` is a noiseless kind.
NOISE_CLASS_OF_TAG: Dict[Optional[str], str] = {
    None: "clean",
    "gauss": "bounded",
    "unif": "bounded",
    "cauchy": "outlier",
}


def noise_class_of(problem_kind: str) -> str:
    """The regime noise class of a battery kind — ``"clean"``, ``"bounded"`` or ``"outlier"``.

    This is what the harness injects into a spec whose ``config_overrides``
    say ``regime_gate="oracle"`` (:meth:`StrategySpec.with_regime_class`):
    the *upper bound* of regime gating, the class known rather than probed.
    """
    _, tag = resolve_problem_kind(problem_kind)
    try:
        return NOISE_CLASS_OF_TAG[tag]
    except KeyError:
        raise ValueError(f"no regime noise class for noise tag {tag!r} (kind {problem_kind!r})") from None


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
    noise_level
        Only read for a kind in :data:`NOISY_PROBLEM_KINDS`:
        ``"moderate"`` (BBOB f101–f106) or ``"severe"`` (f107–f130).
    noise_resample
        Only read for a noisy kind.  ``False`` (default) freezes the noise
        per point — re-evaluating the same ``x`` cannot average it away.
        ``True`` draws fresh noise per call.  See
        :mod:`panobbgo.lib.noise`.
    """

    name: str
    problem_kind: str
    dims: Tuple[int, ...]
    instances: Tuple[int, ...]
    reps: int = 1
    budget_multiplier: int = 2000
    extra_builder_kwargs: Tuple[Tuple[str, Any], ...] = ()
    noise_level: str = "moderate"
    noise_resample: bool = False

    @property
    def is_noisy(self) -> bool:
        return self.problem_kind in NOISY_PROBLEM_KINDS

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


def with_extra_dims(battery: IOHBatterySpec, extra_dims: Sequence[int]) -> IOHBatterySpec:
    """Return ``battery`` widened with ``extra_dims``, preserving everything else.

    The battery presets are frozen contracts (``planning/GOAL.md`` §4:
    "extend via opt-in flags, never edit"), so a caller that wants a
    regime the preset cannot reach composes one instead of editing the
    factory.  Dims already present are ignored and the result is sorted,
    so the call is idempotent and order-insensitive.

    The name gains a ``+d<k>`` suffix per added dim so a report or
    ledger record cannot silently conflate a widened battery with the
    preset it came from — the two measure different things and their
    mean AOCC is not comparable.

    This exists because the nightly loop runs the quick battery, which
    is ``dims=(2,)``.  Two of the sharpest measured results of 2026-08
    (the JSO d5 add on 2026-08-02 and the NLSHADE_LBC per-dim split on
    2026-08-11, where d2 lost 0.0241 while d5 gained 0.0080) lived
    entirely at d5 — invisible to the regime the loop actually samples.

    Note the budget interaction: ``budget_for`` is
    ``budget_multiplier * dim``, so adding dim 5 to the quick battery
    (multiplier 100) buys 500-eval runs alongside the 200-eval ones.
    The added dim costs more per run than the ones already there.
    """
    merged = tuple(sorted(set(battery.dims) | {int(d) for d in extra_dims}))
    if merged == battery.dims:
        return battery
    added = [d for d in merged if d not in battery.dims]
    suffix = "".join(f"+d{d}" for d in added)
    return replace(battery, name=f"{battery.name}{suffix}", dims=merged)


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
# The regimes §2c asks for: noise, and dimension
# ---------------------------------------------------------------------------
#
# ``planning/GOAL.md`` §2c items 1 and 3.  Every number in §2 was measured
# on noiseless, unconstrained MA-BBOB at *d* <= 5, and on that battery a
# two-arm sharing portfolio is *level* with the best single arm
# (§27/§30/§31).  The lean it does have is entirely at *d* = 5, which is
# the gradient these two batteries extend: a portfolio should pay where a
# single arm cannot converge inside the budget (higher *d*) or where its
# model assumptions break (noise).
#
# They are separate presets rather than flags on the standard battery
# because the standard battery is a frozen contract — a mean AOCC from a
# noisy or *d* = 10 run is not comparable with one from §2 and must not
# be able to masquerade as one.


def make_noisy_battery(noise: str = "gauss", *, level: str = "moderate") -> IOHBatterySpec:
    """Standard battery shape, with BBOB-style noise on the objective.

    Same cube as :func:`make_standard_battery` (dims 2 and 5, instances
    0–4, budget ``500·d``) so the *only* difference against the §2
    numbers is the noise, and one battery's cost is one standard
    battery's cost.

    Parameters
    ----------
    noise
        ``"gauss"`` (multiplicative log-normal), ``"unif"`` (the
        endgame-inflating uniform model) or ``"cauchy"`` (rare heavy-tail
        outliers).  See :mod:`panobbgo.lib.noise` for the definitions.
    level
        ``"moderate"`` (default) or ``"severe"``.

    AOCC is scored on the **true** value (the BBOB noisy-suite
    convention); the AOCC an optimizer would compute from its own
    observations is reported alongside as ``aocc_observed``.
    """
    kind = f"MA-BBOB-noisy-{noise}"
    if kind not in NOISY_PROBLEM_KINDS:
        raise ValueError(
            f"unknown noise model {noise!r}; known: {sorted(k.rsplit('-', 1)[1] for k in NOISY_PROBLEM_KINDS)}"
        )
    return IOHBatterySpec(
        name=f"ioh-noisy-{noise}-{level}",
        problem_kind=kind,
        dims=(2, 5),
        instances=tuple(range(5)),
        reps=1,
        budget_multiplier=500,
        noise_level=level,
    )


def make_highdim_battery() -> IOHBatterySpec:
    """Noiseless MA-BBOB at *d* = 10 and 20, at the competition budget.

    Three instances rather than five, and ``2000·d`` evaluations
    (20 000 at *d* = 10, 40 000 at *d* = 20) — the regime §2c item 1
    predicts a sharing portfolio should finally pay in, because a single
    arm cannot converge inside the budget.

    Cost, measured 2026-09-10 on a 16-core laptop (``nice -n 15``,
    ``sync_eval=True``, two screens running side by side): **≈ 18 s per
    run at *d* = 10 and ≈ 55 s at *d* = 20**, for both a CMA-ES arm and
    the two-arm warm portfolio.  The full 2 × 3 cube is therefore ≈ 3.7
    min per strategy per seed — a 4-spec, 3-seed screen is ≈ 45 min, and
    the *d* = 10 half of it alone is ≈ 11 min.  Screen at ``dims=(10,)``
    first.  Most of the wall time is the JSON-Lines round-trip to the
    ``ioh`` worker (one subprocess call per evaluation), so the cost
    scales with the budget, not with the optimizer.
    """
    return IOHBatterySpec(
        name="ioh-highdim",
        problem_kind="MA-BBOB",
        dims=(10, 20),
        instances=(0, 1, 2),
        reps=1,
        budget_multiplier=2000,
    )


def make_noisy_highdim_battery(noise: str = "gauss", *, level: str = "moderate") -> IOHBatterySpec:
    """Both regimes at once — noise at *d* = 10, at the standard budget.

    ``dims=(10,)``, instances 0–2, ``budget_multiplier=500`` (5000
    evaluations per run): deliberately the cheap corner of the cross, so
    the two effects can be crossed for far less than
    :func:`make_highdim_battery` costs.  Measured 2026-09-10: **≈ 3 s per
    run**, i.e. a 4-spec 3-seed screen (36 runs) in under two minutes.

    Read it against ``make_highdim_battery()`` with ``bm=500``, not
    against the 2000·d one: at 500·d a *d* = 10 run has not converged,
    and which arm leads depends on the budget as much as on the noise.
    """
    kind = f"MA-BBOB-noisy-{noise}"
    if kind not in NOISY_PROBLEM_KINDS:
        raise ValueError(f"unknown noise model {noise!r}")
    return IOHBatterySpec(
        name=f"ioh-noisy-highdim-{noise}-{level}",
        problem_kind=kind,
        dims=(10,),
        instances=(0, 1, 2),
        reps=1,
        budget_multiplier=500,
        noise_level=level,
    )


# ---------------------------------------------------------------------------
# IOH-tuned strategy registry
# ---------------------------------------------------------------------------
#
# These specs are tuned for the *anytime* metric (AOCC), in contrast to
# the strategy registry in :mod:`panobbgo.harness` which is tuned for
# ``composite_score`` (find-and-stop).  What that tuning arrived at:
#
# 1.  Few arms, each with a real share of the budget.  A population method
#     needs the whole horizon to adapt, so the six-arm mix that won under
#     composite_score scored 0.35 here against 0.67 for CMA-ES alone
#     (§9-§10) and was retired.
# 2.  No external :class:`~panobbgo.analyzers.Restart` analyzer and no
#     Sobol initial design: both were measured off (Restart discards
#     CMA-ES's adapted covariance, halving it).  ``stop_on_convergence``
#     is disabled by the IOH runner instead — see :func:`_run_one` below
#     — so a converged strategy still spends its budget.
# 3.  Where a second arm helps at all it is *blocked and warm-started*,
#     not interleaved per point: see ``Blocks_warm_CMAES_JSO`` below and
#     ``planning/DISCOVERY_2026-09-09.md`` §27/§30.


def make_ioh_strategies() -> List[StrategySpec]:
    """IOH-tuned strategy registry — primary entry point for AOCC runs.

    Three specs: a pure-random floor (``RoundRobin_Random``), the
    competition candidate (``RoundRobin_CMAES``) and the best portfolio
    found so far (``Blocks_warm_CMAES_JSO``), kept as the control that the
    flagship has to beat.  Returns a small list rather than the full
    panobbgo zoo so each iteration of the harness stays cheap.  Add
    baselines via
    :func:`panobbgo.harness_baselines.make_baseline_strategies` if you
    need an absolute reference.
    """
    from panobbgo.analyzers import Archive
    from panobbgo.heuristics import CMAES, JSO, Random
    from panobbgo.strategies import StrategyBlockBandit, StrategyRoundRobin

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
        # Competition candidate since 2026-09-09: one strong population
        # method with the whole budget.
        #
        # Every arm of the previous six-arm candidate scores higher run
        # *alone* than the portfolio does (standard battery, seeds
        # 42/7/1234): CMA-ES 0.580, NLSHADE_LBC 0.537, jSO 0.459, PSO
        # 0.424, L-SHADE 0.417, Baseline_SciPyDE 0.416, the portfolio
        # 0.352, Random 0.319.  Population methods need the whole budget
        # for their population dynamics; six arms sharing 1000
        # evaluations leave CMA-ES ~150, fewer than it needs to adapt a
        # covariance matrix.  The ordering holds at every budget from 25
        # to 500 evaluations per dimension and the margin grows with
        # budget, so this is not a large-budget artifact.  See
        # planning/DISCOVERY_2026-09-09.md §9-§10.
        #
        # CMA-ES restarts itself (IPOP); the Restart *analyzer* on top
        # halves it (0.663 -> 0.301, seed 42) because its restart event
        # discards the adapted covariance.  Deliberately absent.
        StrategySpec(
            name="RoundRobin_CMAES",
            strategy_class=StrategyRoundRobin,
            heuristics=[(CMAES, {})],
        ),
        # Best portfolio found so far, and the control the flagship is
        # measured against on every battery run.  This is
        # ``Blocks_uniform_cj_warm2`` from ``benchmarks/portfolio_screen.py``
        # (§27/§30): CMA-ES + jSO, blocked, *no* learning rule
        # (``policy="uniform"``), both arms re-seeded from the shared
        # ``Archive`` on every re-acquisition.
        #
        # On the 12-seed roster it scores 0.685 vs 0.666 for CMA-ES alone
        # (+0.019, better on 8/12 seeds, CI includes zero) — parity, not a
        # win, so ``RoundRobin_CMAES`` stays the flagship and this is the
        # thing to beat.  It replaced ``Rewarding_Restart`` (0.35), which
        # was no longer a useful control at half the score of either.
        #
        # ``Archive`` must be listed: without it ``archive_seed`` silently
        # falls back to the Splitter root.  ``Splitter`` is not listed —
        # ``StrategyBase.initialize`` always installs it.  The strategy
        # kwargs travel through ``config_overrides``, which
        # ``StrategySpec.create_strategy`` passes to the constructor.
        StrategySpec(
            name="Blocks_warm_CMAES_JSO",
            strategy_class=StrategyBlockBandit,
            heuristics=[
                (CMAES, {"warm_start": "archive"}),
                (JSO, {"NP_init": "auto", "warm_start": "archive"}),
            ],
            analyzers=[(Archive, {})],
            config_overrides={
                "policy": "uniform",
                "warm_start_on_resume": True,
                # Re-seed on *every* re-acquisition (§21: the ``_any``
                # variants beat the foreign-only default by +0.005).
                "warm_start_only_if_foreign": False,
                # Pinned rather than inherited: §30 measured this guard at
                # -0.007, which made it the default's `False`, but the spec
                # states what it ran.
                "warm_start_only_if_better": False,
            },
        ),
        # The same portfolio behind the **oracle regime gate**
        # (``planning/DESIGN_regime_gating_2026-09-11.md`` §2, §4): both
        # arms are still constructed, but ``REGIME_TABLE_V1`` decides per
        # run which of them may own a block, with the battery's noise class
        # handed in as known (``"oracle"`` is resolved to
        # ``"oracle:<class>"`` by :func:`noise_class_of` in ``_run_one``).
        # On a noiseless 500·dim battery the row is CMA-ES alone; under
        # bounded noise at d <= 5 and at <= 200·dim it is the portfolio.
        # ``seed_name`` pins it to the portfolio's RNG stream so the delta
        # between the two carries only the gate.
        StrategySpec(
            name="RegimeGate_oracle",
            strategy_class=StrategyBlockBandit,
            heuristics=[
                (CMAES, {"warm_start": "archive"}),
                (JSO, {"NP_init": "auto", "warm_start": "archive"}),
            ],
            analyzers=[(Archive, {})],
            config_overrides={
                "policy": "uniform",
                "warm_start_on_resume": True,
                "warm_start_only_if_foreign": False,
                "warm_start_only_if_better": False,
                "regime_gate": "oracle",
            },
            seed_name="Blocks_warm_CMAES_JSO",
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
    # --- noisy batteries only (None on a noiseless kind) ----------------
    #: Seed of the noise realisation.  Shared by every strategy on this
    #: (dim, instance, rep) cell, so the comparison stays paired: all
    #: arms face the *same* noisy function.
    noise_seed: Optional[int] = None
    #: AOCC an optimizer would compute from its own (noisy) observations.
    #: Optimistically biased — the minimum of many noisy readings sits
    #: below the minimum of their means — and reported only as a
    #: diagnostic.  ``aocc`` above is on the true value.
    aocc_observed: Optional[float] = None
    #: AOCC of the *recommendation* trace: the true value of whichever
    #: point currently has the best observed value.  ``aocc - aocc_reco``
    #: is the price of being fooled by the noise.
    aocc_reco: Optional[float] = None

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
    #: Whether the synchronous-harvest evaluation mode was active.
    #: Results measured under different modes are not comparable —
    #: ``ioh_benchmark.py compare`` warns on a mismatch.
    sync_eval: bool = False

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
            "sync_eval": self.sync_eval,
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
            sync_eval=bool(d.get("sync_eval", False)),
        )

    def print_summary(self) -> None:
        print(f"\nIOH battery: {self.battery_name}  ({self.problem_kind})")
        if self.sync_eval:
            print("  eval mode:    sync (deterministic result batches)")
        print(f"  log targets:  [10^{self.log_lo:.0f}, 10^{self.log_hi:.0f}]")
        print(f"  mean AOCC:    {self.mean_aocc:.4f}    over {len(self.runs)} run(s)")
        obs = [r.aocc_observed for r in self.runs if r.error is None and r.aocc_observed is not None]
        if obs:
            reco = [r.aocc_reco for r in self.runs if r.error is None and r.aocc_reco is not None]
            print(f"  (noisy: AOCC is on the TRUE value; observed {float(np.mean(obs)):.4f}", end="")
            print(f", recommendation {float(np.mean(reco)):.4f})" if reco else ")")
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
# Multi-seed batteries — the paired decision instrument
# ---------------------------------------------------------------------------
#
# The 2026-08-03 measurement-substrate finding (see
# planning/SELF_IMPROVEMENT_LOG.md): panobbgo's threaded evaluation makes
# a *single* battery run unable to resolve the ~+0.01 AOCC effects codify
# decisions chase — the per-seed null-change sd on the quick battery is
# ~0.015.  The preferred decision instrument is therefore a *paired*
# multi-seed A/B: run the same battery across N base seeds before and
# after a change, pair the per-seed per-strategy means, and report
# mean/sd/CI95 of the deltas (verifying that untouched control strategies
# stay flat).  These types mechanise that protocol.

#: Canonical seed roster for paired decision A/Bs — the 12 seeds used by
#: the 2026-08-03 rejection measurements.  ``ioh_benchmark.py run
#: --decision-seeds`` expands to this list.
DEFAULT_DECISION_SEEDS: Tuple[int, ...] = (42, 7, 1234, 2025, 3, 11, 99, 123, 777, 2024, 31337, 555)


@dataclass
class IOHMultiSeedResult:
    """One battery evaluated across several base seeds.

    ``results[i]`` is the :class:`IOHHarnessResult` for ``base_seeds[i]``.
    Serialises to JSON with a ``"multi_seed": true`` discriminator so
    ``ioh_benchmark.py compare`` can dispatch on the file format.
    """

    battery_name: str
    problem_kind: str
    log_lo: float
    log_hi: float
    base_seeds: List[int]
    results: List[IOHHarnessResult]
    timestamp: float = field(default_factory=time.time)
    #: Whether the synchronous-harvest evaluation mode was active (see
    #: :class:`IOHHarnessResult.sync_eval`).
    sync_eval: bool = False

    @property
    def mean_aocc(self) -> float:
        """Mean AOCC with equal weight per seed."""
        vals = [r.mean_aocc for r in self.results]
        return float(np.mean(vals)) if vals else 0.0

    def per_strategy_seed_aocc(self) -> Dict[str, List[float]]:
        """Return ``{strategy: [mean AOCC at base_seeds[i]]}``.

        Only strategies present in every per-seed result are returned —
        a ragged strategy (all its runs errored at some seed) cannot be
        paired and is dropped.
        """
        out: Dict[str, List[float]] = {}
        for res in self.results:
            for name, val in res.per_strategy_aocc().items():
                out.setdefault(name, []).append(val)
        n = len(self.results)
        return {k: v for k, v in out.items() if len(v) == n}

    def per_strategy_aocc(self) -> Dict[str, float]:
        """Return ``{strategy: mean AOCC across seeds}``."""
        return {k: float(np.mean(v)) for k, v in self.per_strategy_seed_aocc().items()}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "multi_seed": True,
            "battery_name": self.battery_name,
            "problem_kind": self.problem_kind,
            "log_lo": self.log_lo,
            "log_hi": self.log_hi,
            "base_seeds": list(self.base_seeds),
            "mean_aocc": self.mean_aocc,
            "per_strategy_aocc": self.per_strategy_aocc(),
            "timestamp": self.timestamp,
            "sync_eval": self.sync_eval,
            "results": [r.to_dict() for r in self.results],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=float)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "IOHMultiSeedResult":
        return cls(
            battery_name=d["battery_name"],
            problem_kind=d["problem_kind"],
            log_lo=d.get("log_lo", AOCC_LOG_LO),
            log_hi=d.get("log_hi", AOCC_LOG_HI),
            base_seeds=[int(s) for s in d["base_seeds"]],
            results=[IOHHarnessResult.from_dict(r) for r in d.get("results", [])],
            timestamp=d.get("timestamp", time.time()),
            sync_eval=bool(d.get("sync_eval", False)),
        )

    def print_summary(self) -> None:
        print(f"\nIOH multi-seed battery: {self.battery_name}  ({self.problem_kind})")
        if self.sync_eval:
            print("  eval mode:    sync (deterministic result batches)")
        print(f"  seeds ({len(self.base_seeds)}): {', '.join(str(s) for s in self.base_seeds)}")
        n_runs = sum(len(r.runs) for r in self.results)
        print(f"  mean AOCC:    {self.mean_aocc:.4f}    over {len(self.base_seeds)} seed(s), {n_runs} run(s)")
        print("\n  per strategy (mean ± sd across seeds):")
        matrix = self.per_strategy_seed_aocc()
        for name, vals in sorted(matrix.items(), key=lambda kv: -float(np.mean(kv[1]))):
            arr = np.asarray(vals, dtype=np.float64)
            sd = float(arr.std(ddof=1)) if arr.size >= 2 else float("nan")
            print(f"    {name:32s}  {float(arr.mean()):.4f} ± {sd:.4f}")


def run_ioh_harness_multi_seed(
    strategies: Sequence[StrategySpec],
    battery: IOHBatterySpec,
    base_seeds: Sequence[int],
    *,
    log_lo: float = AOCC_LOG_LO,
    log_hi: float = AOCC_LOG_HI,
    timeout_s: Optional[float] = None,
    progress: bool = True,
    sync_eval: bool = False,
) -> IOHMultiSeedResult:
    """Run :func:`run_ioh_harness` once per seed in ``base_seeds``."""
    if not base_seeds:
        raise ValueError("base_seeds must be non-empty")
    results: List[IOHHarnessResult] = []
    for i, seed in enumerate(base_seeds):
        if progress:
            print(f"== base seed {seed}  ({i + 1}/{len(base_seeds)}) ==", flush=True)
        results.append(
            run_ioh_harness(
                strategies,
                battery,
                base_seed=int(seed),
                log_lo=log_lo,
                log_hi=log_hi,
                timeout_s=timeout_s,
                progress=progress,
                sync_eval=sync_eval,
            )
        )
    return IOHMultiSeedResult(
        battery_name=battery.name,
        problem_kind=battery.problem_kind,
        log_lo=log_lo,
        log_hi=log_hi,
        base_seeds=[int(s) for s in base_seeds],
        results=results,
        sync_eval=sync_eval,
    )


def paired_seed_stats(before: IOHMultiSeedResult, after: IOHMultiSeedResult) -> Dict[str, Dict[str, Any]]:
    """Per-strategy paired delta statistics across the common base seeds.

    Pairs the per-seed per-strategy mean AOCC of ``after`` against
    ``before`` *by seed value* (order need not match), and returns::

        {strategy: {
            "seeds":          [common seeds, in before's order],
            "n":              number of paired seeds,
            "before_mean":    mean AOCC across common seeds (before),
            "after_mean":     mean AOCC across common seeds (after),
            "mean_delta":     mean of per-seed deltas (after - before),
            "sd":             sample sd of the deltas (NaN when n < 2),
            "ci_low"/"ci_high": t-distribution CI95 of the mean delta
                              (NaN when n < 2),
            "per_seed_delta": [delta at each common seed],
        }}

    Only strategies present in both results are returned.  Raises
    :class:`ValueError` when the two results share no base seeds.
    """
    after_seed_set = set(after.base_seeds)
    common = [s for s in before.base_seeds if s in after_seed_set]
    if not common:
        raise ValueError(
            f"no common base seeds between the two results (before={before.base_seeds}, after={after.base_seeds})"
        )
    b_idx = {s: i for i, s in enumerate(before.base_seeds)}
    a_idx = {s: i for i, s in enumerate(after.base_seeds)}
    b_mat = before.per_strategy_seed_aocc()
    a_mat = after.per_strategy_seed_aocc()

    stats: Dict[str, Dict[str, Any]] = {}
    for name in sorted(set(b_mat) & set(a_mat)):
        b_vals = np.asarray([b_mat[name][b_idx[s]] for s in common], dtype=np.float64)
        a_vals = np.asarray([a_mat[name][a_idx[s]] for s in common], dtype=np.float64)
        deltas = a_vals - b_vals
        n = len(common)
        mean_delta = float(deltas.mean())
        if n >= 2:
            sd = float(deltas.std(ddof=1))
            from scipy.stats import t as t_dist

            half = float(t_dist.ppf(0.975, n - 1)) * sd / float(np.sqrt(n))
            ci_low, ci_high = mean_delta - half, mean_delta + half
        else:
            sd = float("nan")
            ci_low = ci_high = float("nan")
        stats[name] = {
            "seeds": list(common),
            "n": n,
            "before_mean": float(b_vals.mean()),
            "after_mean": float(a_vals.mean()),
            "mean_delta": mean_delta,
            "sd": sd,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "per_seed_delta": [float(d) for d in deltas],
        }
    return stats


# ---------------------------------------------------------------------------
# Seed derivation — same SHA-256 scheme as panobbgo.harness
# ---------------------------------------------------------------------------


def _derive_seed(
    base_seed: int,
    problem_kind: str,
    dim: int,
    instance: int,
    strategy_name: str,
    rep: int,
    noise_seed: Optional[int] = None,
) -> int:
    """Per-run RNG seed.

    ``noise_seed`` is appended only when it is not ``None``, so every
    noiseless battery keeps the seeds it had before noisy kinds existed —
    ``tests/test_harness_reproducibility.py`` and every historical
    comparison depend on that.

    Consequence worth knowing before reading a table: a *noisy* battery
    and its noiseless counterpart run on **different RNG streams** (the
    kind differs, and the noise seed is appended), so
    "same arm, noise vs no noise" is an *unpaired* comparison and carries
    the full null floor of ±0.05 for a CMA-ES-containing spec (§18a).
    Comparisons *within* one battery — every ``delta vs <ref>`` the
    screen prints — are paired per (seed, dim, instance) cell and are the
    ones that resolve small effects.
    """
    payload = f"{base_seed}|{problem_kind}|{dim}|{instance}|{strategy_name}|{rep}"
    if noise_seed is not None:
        payload += f"|n{noise_seed}"
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:4], "little")


def _derive_noise_seed(base_seed: int, problem_kind: str, dim: int, instance: int, rep: int) -> int:
    """Seed of the noise realisation for one (dim, instance, rep) cell.

    Deliberately **not** a function of the strategy: every arm on a cell
    must face the identical noisy function or the paired comparison the
    whole harness rests on (``AGENTS.md`` "Statistical rigor",
    ``StrategySpec.seed_name``) leaks the noise realisation into the
    delta.  It *is* a function of the instance and the rep, so the five
    instances of a noisy battery are five different noise realisations
    rather than one repeated.
    """
    payload = f"noise|{base_seed}|{problem_kind}|{dim}|{instance}|{rep}".encode()
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
    sync_eval: bool = False,
    noise_seed: Optional[int] = None,
    noise_level: str = "moderate",
    noise_resample: bool = False,
) -> IOHRunRecord:
    """Run one strategy on one (problem, instance) and return its record."""
    from panobbgo.lib.ioh_wrapper import IOHProblem

    if problem_kind not in SUPPORTED_PROBLEM_KINDS:
        raise ValueError(f"Unknown problem kind {problem_kind!r}; known: {list(SUPPORTED_PROBLEM_KINDS)}")
    worker_kind, noise_tag = resolve_problem_kind(problem_kind)

    t0 = time.time()
    err: Optional[str] = None
    n_evals = 0
    best_fx = float("inf")
    f_opt = 0.0
    trace_evals: List[int] = []
    trace_fx: List[float] = []
    score = 0.0
    aocc_observed: Optional[float] = None
    aocc_reco: Optional[float] = None
    problem: Optional[Any] = None

    try:
        problem = IOHProblem(
            kind=worker_kind,
            instance=instance,
            dim=dim,
            **builder_kwargs,
        )
        f_opt = float(problem.optimum_y)

        if noise_tag is not None:
            from panobbgo.lib.noise import NoisyProblem, make_noise_model

            # Wrapped *in this process*, on top of the untouched worker
            # problem.  The strategy sees only the noisy value; the
            # tracker pulls both out of one inner evaluation.
            problem = NoisyProblem(
                problem,
                make_noise_model(noise_tag, dim=dim, level=noise_level),
                seed=int(noise_seed if noise_seed is not None else 0),
                resample=noise_resample,
                f_opt=f_opt,
            )

        np.random.seed(seed)

        tracker = IOHTracker(problem, budget=budget)
        try:
            # The budget must reach the config *before* the heuristics are
            # constructed — budget-adaptive arms (``NP_init="auto"``) size
            # themselves from ``config.max_eval`` in their constructor, and
            # would otherwise read Config's default (1000) instead of the
            # battery's ``budget_multiplier * dim``.
            # A spec gated by ``regime_gate="oracle"`` learns the battery's
            # noise class here — the one regime feature the strategy cannot
            # read off the problem itself.
            strategy = strategy_spec.with_regime_class(noise_class_of(problem_kind)).create_strategy(
                problem, seed=seed, max_eval=budget
            )
            # Harmless belt-and-braces: keeps the invariant for factory-built
            # strategies that rebuild their own config.
            strategy.config.max_eval = budget
            # Deterministic result batches for the threaded evaluator —
            # cuts adaptive-strategy measurement noise roughly in half
            # (2026-08-09 repeat-sd experiment); opt-in via --sync-eval.
            strategy.config.sync_evaluation = bool(sync_eval)
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
        if tracker.has_true:
            # BBOB-noisy convention: the metric is scored on the true,
            # noise-free value; what the optimizer *observed* is a
            # diagnostic (and an optimistically biased one).
            best_fx = tracker.best_true_fx
            score = aocc(tracker.best_so_far_true, f_opt=f_opt, log_lo=log_lo, log_hi=log_hi, budget=budget)
            aocc_observed = aocc(tracker.best_so_far, f_opt=f_opt, log_lo=log_lo, log_hi=log_hi, budget=budget)
            aocc_reco = aocc(tracker.best_so_far_reco, f_opt=f_opt, log_lo=log_lo, log_hi=log_hi, budget=budget)
            trace_evals, trace_fx = _downsample_trajectory(tracker.best_so_far_true, budget=budget)
        else:
            best_fx = tracker.best_fx
            score = aocc(tracker.best_so_far, f_opt=f_opt, log_lo=log_lo, log_hi=log_hi, budget=budget)
            trace_evals, trace_fx = _downsample_trajectory(tracker.best_so_far, budget=budget)
    except Exception as e:  # noqa: BLE001  — record and continue
        err = f"{type(e).__name__}: {e}"
    finally:
        if problem is not None:
            try:
                problem.close()
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
        noise_seed=noise_seed,
        aocc_observed=aocc_observed,
        aocc_reco=aocc_reco,
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
    sync_eval: bool = False,
) -> IOHHarnessResult:
    """Run every strategy against every (dim, instance, rep) in ``battery``.

    Runs serially; the underlying strategies use internal threading.  For
    large batteries, drive multiple ``run_ioh_harness`` calls from outside
    if you need outer parallelism.

    ``sync_eval=True`` enables the synchronous-harvest evaluation mode
    (``config.sync_evaluation``) on every strategy: deterministic result
    batches, roughly halving run-to-run measurement noise for adaptive
    strategies.  Only compare results measured under the same mode.
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
                    # One noise realisation per (dim, instance, rep) cell,
                    # identical for every strategy — see _derive_noise_seed.
                    noise_seed = (
                        _derive_noise_seed(base_seed, battery.problem_kind, dim, instance, rep)
                        if battery.is_noisy
                        else None
                    )
                    # ``rng_identity`` is ``spec.seed_name or spec.name``: variants
                    # of one arm can opt into a shared RNG stream so an A/B
                    # measures the parameter, not the run-to-run variance.
                    seed = _derive_seed(
                        base_seed, battery.problem_kind, dim, instance, spec.rng_identity, rep, noise_seed
                    )
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
                        sync_eval=sync_eval,
                        noise_seed=noise_seed,
                        noise_level=battery.noise_level,
                        noise_resample=battery.noise_resample,
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
        sync_eval=sync_eval,
        runs=runs,
    )
