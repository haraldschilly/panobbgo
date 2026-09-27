# -*- coding: utf8 -*-
# Copyright 2012-2026 Panobbgo Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

r"""
Training data for a learned arm selector: shared probe, counterfactual labels
=============================================================================

Roadmap ``planning/DESIGN_roadmap_2026-09-26.md`` §4 A, step 1 (DISCOVERY §70):
the data pipeline of *probe → select → unleash*.  No selector here.

For one **task** — a problem instance, its dimension ``d``, a budget ``B``, a
number of workers ``q`` and a seed — :func:`run_task`

1. evaluates a **probe** of ``k`` points (:func:`probe_design`: a scrambled
   Latin hypercube of ``PROBE_PER_DIM · d`` points in the box; the same
   points for every arm, and for q = 1 and q = 4);
2. computes the **features** of the probe (:func:`probe_features`);
3. **continues every arm of the menu** (:data:`ARM_MENU_V0`) from that same
   probe archive to the full budget (:func:`continue_from_probe`: the probe
   results are booked with
   :meth:`~panobbgo.core.StrategyBase.preload_results` before the first
   pass, so they count against ``max_eval`` and every module sees them before
   it emits a point), on the virtual clock with ``q`` workers;
4. scores each continuation on the **remaining** ``B - k`` evaluations
   (best-so-far starting at the probe's best): AOCC over evaluations,
   ``aocc_time`` over the continuation's virtual time, and the final
   precision.

:func:`add_labels` turns the scores into the **label: the counterfactual
normalised regret of each arm**, ``max_a s_a - s_i`` for the cell's score
``s`` (AOCC at q = 1, ``aocc_time`` at q > 1; both are normalised to [0, 1]
by the log-precision range, so easy and hard instances, d = 2 and d = 40
share one scale).  0 = the best arm on that task.

How each arm uses the probe (the continuation form of each menu entry,
:func:`arm_menu_v0`):

* ``Blocks_warm_CMAES_JSO``, ``Blocks_warm_CMAES_JSO_TRQ`` — unchanged:
  their CMA-ES and jSO arms already have ``warm_start="archive"``, which at
  the ``start`` event fits CMA-ES's m / σ to the archive's top points and
  seeds jSO's population from them; the TR arm's centre is the best archive
  point.
* ``RoundRobin_TRQ`` — unchanged: its centre is the best archive point, and
  its quadratic model is fitted to every archive point near it (the probe's
  included).
* ``RoundRobin_CMAES`` — **with** ``warm_start="archive"`` and the
  ``Archive`` analyzer (the registry spec cold-starts at the box centre and
  would ignore the probe): m and σ fitted to the probe's best points, the
  recipe the Blocks arms use.  Same RNG identity as the registry spec.
* ``RoundRobin_COBYQA`` — **with** ``warm_start="archive"``: SciPy's COBYQA
  cannot take an interpolation set, so only its start point is warm (the
  best probe point instead of the box centre); its own initial design of
  ``2d + 1`` points follows.  SciPy moves a start within the initial radius
  of a bound onto the bound (or to bound ± radius), and the default radius
  is half of every axis (DISCOVERY §69.1), so the start is the best probe
  point *quantised* to {lower face, centre, upper face} per axis: the warm
  start keeps the probe's region, not its point (DISCOVERY §70.1).

The registry's cold forms of the last two (:data:`DIAGNOSTIC_ARMS`) can run
beside the menu to measure what the warm start is worth; they are never
labelled.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import replace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from panobbgo.benchmark import StrategySpec
from panobbgo.features import avg_ranks, fscale_features, landscape_features

#: The arm menu v0 (roadmap §4 A step 1; DISCOVERY §66.5, §68.4): the headline
#: portfolio, CMA-ES alone, the quadratic trust region alone, COBYQA alone and
#: the portfolio with the trust region as a third arm.  The one place it is defined.
ARM_MENU_V0: Tuple[str, ...] = (
    "Blocks_warm_CMAES_JSO",
    "RoundRobin_CMAES",
    "RoundRobin_TRQ",
    "RoundRobin_COBYQA",
    "Blocks_warm_CMAES_JSO_TRQ",
)

#: Not in the menu: the registry's cold forms of the two arms whose continuation
#: form differs (box-centre start, the probe only spent), run beside the menu
#: to measure what the warm start is worth.  Never labelled (:func:`add_labels`).
DIAGNOSTIC_ARMS: Tuple[str, ...] = ("RoundRobin_CMAES_cold", "RoundRobin_COBYQA_cold")

#: Probe size per dimension: ``k = PROBE_PER_DIM · d`` (DISCOVERY §70.1 says why 10).
PROBE_PER_DIM: int = 10

#: ``who`` of the probe's results in the archive.
PROBE_WHO: str = "Probe"

#: The identity the probe's seed is derived under (in place of a strategy name).
PROBE_STREAM_IDENTITY: str = "<selector probe>"

#: Score used for the label, per ``q``: AOCC over evaluations at q = 1, over virtual time otherwise.
SCORE_Q1: str = "aocc"
SCORE_QN: str = "aocc_time"


def arm_menu_v0(names: Optional[Iterable[str]] = None) -> List[StrategySpec]:
    """The continuation form of every arm of :data:`ARM_MENU_V0` (module docstring), in menu order.

    ``names`` restricts the list and may name :data:`DIAGNOSTIC_ARMS` too
    (appended after the menu); an unknown name raises.
    """
    from panobbgo.analyzers import Archive
    from panobbgo.harness_ioh import make_ioh_strategies, make_trust_region_strategies
    from panobbgo.heuristics import CMAES, COBYQA
    from panobbgo.strategies import StrategyRoundRobin

    registry = {s.name: s for s in make_ioh_strategies() + make_trust_region_strategies()}
    specs: Dict[str, StrategySpec] = {
        "Blocks_warm_CMAES_JSO": registry["Blocks_warm_CMAES_JSO"],
        "RoundRobin_CMAES": StrategySpec(
            name="RoundRobin_CMAES",
            strategy_class=StrategyRoundRobin,
            heuristics=[(CMAES, {"warm_start": "archive"})],
            analyzers=[(Archive, {})],
        ),
        "RoundRobin_TRQ": registry["RoundRobin_TRQ"],
        "RoundRobin_COBYQA": StrategySpec(
            name="RoundRobin_COBYQA",
            strategy_class=StrategyRoundRobin,
            heuristics=[(COBYQA, {"warm_start": "archive"})],
        ),
        "Blocks_warm_CMAES_JSO_TRQ": registry["Blocks_warm_CMAES_JSO_TRQ"],
        # diagnostics: the registry's cold forms, on the menu arm's RNG stream
        "RoundRobin_CMAES_cold": replace(
            registry["RoundRobin_CMAES"], name="RoundRobin_CMAES_cold", seed_name="RoundRobin_CMAES"
        ),
        "RoundRobin_COBYQA_cold": replace(
            registry["RoundRobin_COBYQA"], name="RoundRobin_COBYQA_cold", seed_name="RoundRobin_COBYQA"
        ),
    }
    assert tuple(specs) == ARM_MENU_V0 + DIAGNOSTIC_ARMS
    wanted = list(ARM_MENU_V0) if names is None else list(names)
    unknown = [n for n in wanted if n not in specs]
    if unknown:
        raise ValueError(
            f"unknown arm(s) {unknown}; the menu: {list(ARM_MENU_V0)}, diagnostics: {list(DIAGNOSTIC_ARMS)}"
        )
    return [spec for n, spec in specs.items() if n in wanted]


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------


def probe_seed(base_seed: int, family: str, dim: int, instance: int) -> int:
    """The probe's seed for one (base seed, instance) — independent of ``q`` and of the arm."""
    from panobbgo.harness_ioh import _derive_seed

    return _derive_seed(int(base_seed), str(family), int(dim), int(instance), PROBE_STREAM_IDENTITY, 0)


def probe_design(dim: int, k: int, seed: int) -> np.ndarray:
    """``k`` points of a scrambled Latin hypercube in the unit box ``[0, 1]^dim`` (``k x dim``).

    One stratum per point on every axis, for any ``k`` (a Sobol sequence is
    balanced only at powers of two), no box-centre point (no free hit on a
    centred optimum), deterministic in ``seed``.
    """
    from scipy.stats import qmc

    if int(k) < 1 or int(dim) < 1:
        raise ValueError(f"need dim >= 1 and k >= 1, got dim={dim}, k={k}")
    return qmc.LatinHypercube(d=int(dim), rng=np.random.default_rng(int(seed))).random(int(k))


def evaluate_probe(problem: Any, u: np.ndarray) -> List[Any]:
    """Evaluate the unit-box points ``u`` on ``problem``; one :class:`~panobbgo.lib.Result` each, ``who`` = :data:`PROBE_WHO`.

    Evaluated directly (no strategy, no tracker).  A problem with failure
    regions is not supported (v0: the probe must be fully evaluated).
    """
    from panobbgo.lib import Point

    box = np.asarray(problem.box.box, dtype=np.float64)
    x = box[:, 0] + np.asarray(u, dtype=np.float64) * (box[:, 1] - box[:, 0])
    out = []
    for xi in x:
        r = problem(Point(np.asarray(xi, dtype=np.float64), PROBE_WHO))
        if r.fx is None or not math.isfinite(float(r.fx)):
            raise ValueError(f"probe point {xi} has no finite value ({r.fx}): failure regions are not supported")
        out.append(r)
    return out


def probe_features(u: np.ndarray, fx: np.ndarray, *, dim: int, budget: int, q: int) -> Dict[str, Any]:
    """Features of a probe: ``u`` the points in the unit box (``k x d``), ``fx`` their values.

    Never raw ``f`` values or raw coordinates.  Every feature and its invariance
    (tested in ``tests/test_selector_data.py``):

    * **rank-based** (:func:`~panobbgo.features.landscape_features`) —
      invariant to ``f -> a·f + b`` (``a > 0``) and to every strictly
      increasing transform of ``f``: ``fdc``; the NBC group
      ``nbc_mean_ratio``, ``nbc_sd_ratio``, ``nbc_nn_nb_cor``,
      ``nbc_dist_ratio_cv``, ``nbc_nb_fitness_cor``; the dispersion
      ``disp_10``, ``disp_25``; the rank-R² ``r2_lin``, ``r2_add``,
      ``r2_quad``, ``sep_ratio``, ``log10_cond``, ``hess_pos``.  Distances are
      Euclidean in ``u`` (/ √d) and enter only as ratios or ranks, so these
      are also invariant to a rotation, a shift and a uniform scaling of
      ``x`` — except ``r2_add`` and ``sep_ratio``, which are deliberately not
      rotation-invariant (separability).
    * **f-scale** (:func:`~panobbgo.features.fscale_features`) — invariant to
      ``f -> a·f + b`` (``a > 0``) only: ``fr2_lin``, ``fr2_add``,
      ``fr2_quad``, ``flog_quad_gap`` (the DISCOVERY §66.4 separator of exact
      quadratics), ``fsep_ratio``, ``flog10_cond``, ``fhess_pos``,
      ``y_skew``, ``y_kurt``; ``y_ties`` is monotone-invariant.  Rotation /
      shift of ``x``: invariant except ``fr2_add`` and ``fsep_ratio``.
    * **context** — ``dim``, ``budget_per_d`` (``B / d``), ``probe_per_d``
      (``k / d``), ``remaining_per_d`` (``(B - k) / d``), ``q``.

    ``None`` stands for an undefined feature (the full quadratic needs
    ``2p`` points: at ``k = 10 d`` that is d ≤ 6; see §70).
    """
    u = np.asarray(u, dtype=np.float64)
    fx = np.asarray(fx, dtype=np.float64)
    k = int(fx.size)
    feats: Dict[str, Any] = {
        "dim": int(dim),
        "budget_per_d": budget / float(dim),
        "probe_per_d": k / float(dim),
        "remaining_per_d": (budget - k) / float(dim),
        "q": int(q),
    }
    ok = np.isfinite(fx)
    feats.update(landscape_features(u[ok], avg_ranks(fx[ok]), max_points=max(k, 10)))
    feats.update(fscale_features(u, fx))
    return {key: (None if isinstance(v, float) and not math.isfinite(v) else v) for key, v in feats.items()}


# ---------------------------------------------------------------------------
# continuation
# ---------------------------------------------------------------------------


def _penalty(r: Any, rho: float) -> float:
    """The metric's value of a result: ``f + rho·cv`` (the families' penalty tracker)."""
    cv = float(r.cv) if r.cv_vec is not None else 0.0
    return float(r.fx) + rho * cv


def continue_from_probe(
    spec: StrategySpec,
    problem: Any,
    probe: Sequence[Any],
    *,
    budget: int,
    seed: int,
    virtual: Any,
    log_lo: Optional[float] = None,
    log_hi: Optional[float] = None,
) -> Dict[str, Any]:
    """Run ``spec`` on ``problem`` from the probe archive ``probe`` to ``budget`` evaluations in total.

    The probe results are booked before the first pass
    (:meth:`~panobbgo.core.StrategyBase.preload_results`); ``max_eval`` is the
    full ``budget``, so the arm evaluates ``budget - k`` new points.  The run
    goes through the families' penalty tracker on the virtual clock
    (``virtual``, a :class:`~panobbgo.virtual_clock.VirtualSpec` with the
    cell's duration stream), as ``harness_families`` runs it, with
    ``sync_evaluation`` and ``stop_on_convergence = False``.

    Returns the scores on the remaining budget, each from a best-so-far that
    starts at the probe's best: ``aocc`` (over the ``budget - k``
    evaluations), ``aocc_time`` (over the continuation's own virtual time,
    ``(budget - k)·d̄/q``; the probe's time is the same for every arm and not
    simulated), ``final_logp`` (``log10`` of the final precision, clipped to
    the AOCC range), ``n_evals`` (new evaluations), ``error`` (``None`` for a
    clean run; ``EndedEarly …`` when the arm stopped by itself, scored like
    any short run) and ``elapsed_s``.
    """
    from panobbgo.harness_families import PENALTY_RHO, PenaltyTracker
    from panobbgo.harness_ioh import _early_end_error
    from panobbgo.ioh_runner import AOCC_LOG_HI, AOCC_LOG_LO, _BudgetExhausted, aocc, aocc_virtual_time
    from panobbgo.local_run import pin_blas

    lo = AOCC_LOG_LO if log_lo is None else float(log_lo)
    hi = AOCC_LOG_HI if log_hi is None else float(log_hi)
    pin_blas()
    t0 = time.time()
    k = len(probe)
    rest = int(budget) - k
    if rest <= 0:
        raise ValueError(f"the probe ({k}) must be smaller than the budget ({budget})")
    f_opt = float(problem.f_opt)
    best_probe = min(_penalty(r, PENALTY_RHO) for r in probe)
    np.random.seed(int(seed))
    tracker = PenaltyTracker(problem, budget=rest)
    error: Optional[str] = None
    try:
        strategy = spec.with_regime_class("clean").create_strategy(problem, seed=int(seed), max_eval=int(budget))
        strategy.config.max_eval = int(budget)
        strategy.config.sync_evaluation = True
        strategy.config.stop_on_convergence = False
        virtual.apply(strategy, observer=tracker)
        # copies: a Result carries mutable per-run state (virtual times), and the same probe feeds every arm
        strategy.preload_results([copy.copy(r) for r in probe])
        tracker.on_timeout = getattr(strategy, "request_stop", None)
        try:
            strategy.start()
        except _BudgetExhausted:
            pass
    except Exception as e:  # noqa: BLE001 — record and continue, as the harness does
        error = f"{type(e).__name__}: {e}"
    finally:
        tracker.restore()
    trace = np.minimum(best_probe, np.asarray(tracker.best_so_far, dtype=np.float64))
    if trace.size == 0:
        trace = np.array([best_probe])
    out: Dict[str, Any] = {
        "aocc": aocc(trace.tolist(), f_opt=f_opt, log_lo=lo, log_hi=hi, budget=rest),
        "aocc_time": None,
        "final_logp": float(np.clip(math.log10(max(float(trace[-1]) - f_opt, 10.0**lo)), lo, hi)),
        "n_evals": int(tracker.n_evals),
        "error": error,
        "elapsed_s": time.time() - t0,
    }
    if tracker.timeline and len(tracker.timeline) == tracker.n_evals:
        out["aocc_time"] = aocc_virtual_time(
            [(0.0, best_probe)] + list(tracker.timeline),
            budget=rest,
            workers=int(virtual.workers),
            f_opt=f_opt,
            mean_duration=virtual.model().mean,
            log_lo=lo,
            log_hi=hi,
        )
    if out["error"] is None:
        out["error"] = _early_end_error(tracker.n_evals, getattr(tracker, "_reserved", tracker.n_evals), rest)
    return out


def run_task(
    problem: Any,
    *,
    base_seed: int,
    budget_multiplier: int,
    q: int,
    probe_per_dim: int = PROBE_PER_DIM,
    arms: Optional[Sequence[str]] = None,
    duration: str = "lognormal",
    sigma: float = 0.5,
) -> Dict[str, Any]:
    """One task: probe, features, every arm continued from the probe.  One flat row (``dict``).

    Keys: the task (``family``, ``dim``, ``instance``, ``seed``, ``q``,
    ``budget``, ``k``), ``probe_best_logp`` (the probe's precision, for the
    analysis only — it needs ``f_opt`` and is **not** a feature),
    ``f_<feature>`` for every :func:`probe_features` key, and per arm
    ``<arm>:<score>`` for ``aocc``, ``aocc_time``, ``final_logp``,
    ``n_evals``, ``error``, ``elapsed_s``.  Labels: :func:`add_labels`.
    ``arms`` defaults to the menu; :data:`DIAGNOSTIC_ARMS` may be named too.

    Seeds as in ``harness_families.run_family_harness``: each arm's seed is
    ``_derive_seed(base_seed, family, dim, instance, arm, 0)``, the duration
    stream is the cell's (common random numbers across the arms, and across
    q), the probe's is :func:`probe_seed` (the same probe at every q).
    """
    from panobbgo.harness_ioh import _derive_seed
    from panobbgo.ioh_runner import AOCC_LOG_HI, AOCC_LOG_LO
    from panobbgo.virtual_clock import DURATION_STREAM_IDENTITY, VirtualSpec

    fam, dim, inst = str(problem.family), int(problem.dim), int(problem.instance)
    budget = int(budget_multiplier) * dim
    k = int(probe_per_dim) * dim
    u = probe_design(dim, k, probe_seed(base_seed, fam, dim, inst))
    probe = evaluate_probe(problem, u)
    fx = np.array([float(r.fx) for r in probe])
    row: Dict[str, Any] = {
        "family": fam,
        "dim": dim,
        "instance": inst,
        "seed": int(base_seed),
        "q": int(q),
        "budget": budget,
        "k": k,
        "probe_best_logp": float(
            np.clip(
                math.log10(max(float(fx.min()) - float(problem.f_opt), 10.0**AOCC_LOG_LO)), AOCC_LOG_LO, AOCC_LOG_HI
            )
        ),
    }
    for key, v in probe_features(u, fx, dim=dim, budget=budget, q=q).items():
        row[f"f_{key}"] = v
    cell = _derive_seed(int(base_seed), fam, dim, inst, DURATION_STREAM_IDENTITY, 0)
    virtual = VirtualSpec(workers=int(q), duration=duration, sigma=float(sigma), policy="async").with_cell(cell)
    for spec in arm_menu_v0(arms):
        seed = _derive_seed(int(base_seed), fam, dim, inst, spec.rng_identity, 0)
        res = continue_from_probe(spec, problem, probe, budget=budget, seed=seed, virtual=virtual)
        for key, v in res.items():
            row[f"{spec.name}:{key}"] = v
    return row


def score_key(q: int) -> str:
    """The score a task's label is taken of: :data:`SCORE_Q1` at q = 1, :data:`SCORE_QN` otherwise."""
    return SCORE_Q1 if int(q) == 1 else SCORE_QN


def add_labels(row: Dict[str, Any], arms: Sequence[str] = ARM_MENU_V0) -> Dict[str, Any]:
    """Add the labels to one task row (in place; returned too).

    * ``<arm>:score`` — the cell's score (:func:`score_key`);
    * ``<arm>:regret`` — **the label**: ``max_a score_a - score_arm`` (0 = best arm);
    * ``<arm>:regret_final`` — the same on the final precision, in units of
      the log range (``(logp_arm - min_a logp_a) / (log_hi - log_lo)``);
    * ``<arm>:rank`` — 1 = best score (average ranks for ties);
    * ``best_arm`` — the arm with the best score (the first in menu order on a tie);
    * ``n_best`` — how many arms tie at the best score (a win share is ``1 / n_best``).

    A task where some arm has no score gets ``None`` labels.
    """
    from scipy.stats import rankdata

    from panobbgo.ioh_runner import AOCC_LOG_HI, AOCC_LOG_LO

    key = score_key(int(row["q"]))
    scores = [row.get(f"{a}:{key}") for a in arms]
    logps = [row.get(f"{a}:final_logp") for a in arms]
    if any(s is None or not math.isfinite(float(s)) for s in scores):
        for a in arms:
            row[f"{a}:score"] = row[f"{a}:regret"] = row[f"{a}:regret_final"] = row[f"{a}:rank"] = None
        row["best_arm"] = None
        row["n_best"] = None
        return row
    s = np.asarray(scores, dtype=np.float64)
    lp = np.asarray(logps, dtype=np.float64)
    ranks = rankdata(-s, method="average")
    for i, a in enumerate(arms):
        row[f"{a}:score"] = float(s[i])
        row[f"{a}:regret"] = float(s.max() - s[i])
        row[f"{a}:regret_final"] = float((lp[i] - lp.min()) / (AOCC_LOG_HI - AOCC_LOG_LO))
        row[f"{a}:rank"] = float(ranks[i])
    row["best_arm"] = arms[int(np.argmax(s))]
    row["n_best"] = int((s == s.max()).sum())
    return row
