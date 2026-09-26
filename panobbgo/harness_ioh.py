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
* Forking the measurement track kept the self-improvement ledgers
  honest (the loop was removed 2026-09-25; its dated ledgers are
  ``planning/done/self_improve_ledger_*.jsonl``): a change can be good for
  ``composite_score`` and bad for AOCC, and we want to see both.

Public surface
--------------

* :class:`IOHBatterySpec` — declares the (problem kind, dim, instances,
  budget multiplier, reps) cube to evaluate.
* :func:`make_quick_battery` / :func:`make_standard_battery` /
  :func:`make_full_battery` — preset batteries matching the
  ``quick``/``standard``/``full`` modes of the legacy harness.
* :func:`make_bbob_battery` — the **function axis**: the 24 noiseless
  BBOB functions as a dimension of the cube, with
  :func:`bbob_class_of` / :data:`BBOB_CLASS_OF_FID` giving each run its
  COCO class so results can be reported per landscape group
  (``planning/DESIGN_suite_2026-09-14.md``).
* :func:`make_noisy_battery` / :func:`make_highdim_battery` /
  :func:`make_noisy_highdim_battery` — the regimes ``planning/GOAL.md``
  §2c asks for: noise on the objective (scored on the true value, BBOB
  noisy-suite style) and *d* ∈ {10, 20}.  Noise is applied in *this*
  process by :mod:`panobbgo.lib.noise`; the ``ioh`` worker is untouched.
* :func:`make_large_battery` / :func:`make_largescale_battery` — opt-in
  dimensions 30/40 (MA-BBOB) and a ``bbob-largescale``-style slice (plain
  BBOB, a few functions at *d* 80/160).
* :func:`make_sealed_battery` — the **sealed test set**
  (:mod:`panobbgo.sealed`): fresh MA-BBOB instances, run only for claims.
* :class:`IOHRunRecord` — per (problem kind, dim, instance, strategy, rep)
  result, including the convergence trajectory.
* :class:`IOHHarnessResult` — aggregate over a battery, with mean AOCC and
  per-pair detail.  Serialises to JSON for diffing.
* :func:`run_ioh_harness` — main entry point.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import time
import warnings
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from panobbgo.benchmark import StrategySpec
from panobbgo.ioh_runner import (  # noqa: F401
    AOCC_LOG_HI,
    AOCC_LOG_LO,
    IOHTracker,
    _BudgetExhausted,
    aocc,
    aocc_virtual_time,
)
from panobbgo import fp_env
from panobbgo.local_run import BLAS_THREADS, pin_blas
from panobbgo.sealed import (
    DEV_INSTANCE_LIMIT,
    SEALED_MABBOB_DIMS,
    SEALED_MABBOB_INSTANCES,
    check_sealed_mabbob_instances,
    is_dev_instance_id,
    print_sealed_banner,
)
from panobbgo.features import FeatureLogger, FeatureLogSpec
from panobbgo.virtual_clock import DURATION_STREAM_IDENTITY, VirtualSpec


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
# The BBOB function axis and its COCO classes
# ---------------------------------------------------------------------------
#
# Same table shape as NOISE_CLASS_OF_TAG above: a flat mapping plus a
# lookup helper, so a report can group runs by a *landscape* label the way
# it already groups them by noise class.
#
# The five groups are the COCO/BBOB ones (Hansen et al., "Real-Parameter
# Black-Box Optimization Benchmarking 2009: Noiseless Functions
# Definitions"), in the canonical order:
#
#   f1–f5    separable
#   f6–f9    low or moderate conditioning
#   f10–f14  high conditioning and unimodal
#   f15–f19  multi-modal with adequate global structure
#   f20–f24  multi-modal with weak global structure
#
# The short tags below are what the reports print.

#: The five groups, in COCO order, as ``tag -> (first fid, last fid)``.
BBOB_CLASS_RANGES: Tuple[Tuple[str, int, int], ...] = (
    ("separable", 1, 5),
    ("low-cond", 6, 9),
    ("high-cond", 10, 14),
    ("multimodal-global", 15, 19),
    ("multimodal-weak", 20, 24),
)

#: Every BBOB function id 1..24 -> its COCO class tag.  Complete and
#: gap-free by construction; ``tests/test_harness_ioh.py`` pins that.
BBOB_CLASS_OF_FID: Dict[int, str] = {fid: tag for tag, lo, hi in BBOB_CLASS_RANGES for fid in range(lo, hi + 1)}

#: The 24 noiseless BBOB functions — the full function axis.
ALL_BBOB_FIDS: Tuple[int, ...] = tuple(BBOB_CLASS_OF_FID)

#: Class tags in COCO order, for deterministic report ordering.
BBOB_CLASS_ORDER: Tuple[str, ...] = tuple(tag for tag, _lo, _hi in BBOB_CLASS_RANGES)


def bbob_class_of(fid: int) -> str:
    """The COCO class tag of BBOB function ``fid`` (1..24).

    Raises :class:`ValueError` outside 1..24 — there is no 25th BBOB
    function and ``fid=0`` is not an id, so a silent ``"unknown"`` bucket
    would only hide a caller's off-by-one.
    """
    try:
        return BBOB_CLASS_OF_FID[int(fid)]
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"not a BBOB function id (expected 1..24): {fid!r}") from None


def bbob_classes_present(fids: Iterable[int]) -> List[str]:
    """The COCO class tags covered by ``fids``, in COCO order."""
    present = {bbob_class_of(f) for f in fids}
    return [c for c in BBOB_CLASS_ORDER if c in present]


# ---------------------------------------------------------------------------
# Battery / spec
# ---------------------------------------------------------------------------


#: Every field of the one sealed MA-BBOB battery (:func:`make_sealed_battery`);
#: :class:`IOHBatterySpec` refuses ``sealed=True`` with anything else.
_SEALED_FIELDS: Dict[str, Any] = dict(
    name="sealed-mabbob",
    problem_kind="MA-BBOB",
    dims=SEALED_MABBOB_DIMS,
    instances=SEALED_MABBOB_INSTANCES,
    reps=1,
    budget_multiplier=500,
    extra_builder_kwargs=(),
    noise_level="moderate",
    noise_resample=False,
    fids=(),
    sealed=True,
    log_lo=None,
    log_hi=None,
)


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
    fids
        The **function axis**: BBOB function ids (1..24) the battery runs
        over.  Empty (the default) means "the kind has no function axis" —
        every battery written before 2026-09-14 keeps exactly the cube,
        the seeds and the numbers it had.  Non-empty requires a
        ``"BBOB"``-derived ``problem_kind`` and turns the run cube into
        ``fids × dims × instances × reps``; ``fid`` then joins the seed
        payloads, the run record and the report keys.  Use
        :data:`ALL_BBOB_FIDS` for the whole suite and
        :func:`bbob_class_of` to group the rows.
    reps
        Independent repetitions per (fid, dim, instance, strategy).  Each
        rep uses a SHA-256-derived seed so before/after runs see the same
        problem realisation but different RNG state across reps.
    budget_multiplier
        Evaluation budget per run = ``budget_multiplier * dim``.  The
        MA-BBOB anytime rules use ``2000``.
    extra_builder_kwargs
        Optional ``(key, value)`` pairs passed straight to the worker's
        problem builder — the escape hatch for a builder argument that has
        no field here.  A single fixed function is expressible as
        ``(("fid", 1),)``, but a battery that wants the function as an
        *axis* (and therefore per-fid seeds, records and class labels)
        must use ``fids`` instead; giving both is an error.
    noise_level
        Only read for a kind in :data:`NOISY_PROBLEM_KINDS`:
        ``"moderate"`` (BBOB f101–f106) or ``"severe"`` (f107–f130).
    noise_resample
        Only read for a noisy kind.  ``False`` (default) freezes the noise
        per point — re-evaluating the same ``x`` cannot average it away.
        ``True`` draws fresh noise per call.  See
        :mod:`panobbgo.lib.noise`.
    sealed
        ``True`` only for the sealed test set (:mod:`panobbgo.sealed`),
        and then the spec must be exactly :func:`make_sealed_battery` —
        a ``dataclasses.replace`` sub-selection or any other change is
        refused.  Without it every instance id must lie in the development
        window ``0 <= id < DEV_INSTANCE_LIMIT`` (``2**20``), outside which
        ``ioh`` ids alias each other (:mod:`panobbgo.sealed`).
        :func:`run_ioh_harness` prints a warning banner for a sealed spec.
    log_lo, log_hi
        The AOCC target range of this battery, ``None`` for the standard
        :data:`~panobbgo.ioh_runner.AOCC_LOG_LO` /
        :data:`~panobbgo.ioh_runner.AOCC_LOG_HI`.  A battery whose runs sit
        above the standard upper bound for most of the budget (the
        largescale slice) widens it here, so the bound travels with the
        battery into every result file (``IOHHarnessResult.log_hi``) and
        cannot be mixed up with a standard-bound number.
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
    fids: Tuple[int, ...] = ()
    sealed: bool = False
    log_lo: Optional[float] = None
    log_hi: Optional[float] = None

    def __post_init__(self) -> None:
        # ``fids`` is normalised the way callers already hand ``instances``
        # in — any iterable of ints (``range(1, 25)``, a list, a generator)
        # becomes a plain tuple of ``int`` — so a frozen spec is hashable
        # and comparable regardless of how it was built.  The frozen
        # dataclass needs object.__setattr__ for that.
        fids = tuple(int(f) for f in self.fids)
        bad = [f for f in fids if f not in BBOB_CLASS_OF_FID]
        if bad:
            raise ValueError(f"fids must be BBOB function ids 1..24; got {bad}")
        if fids:
            worker_kind, _tag = resolve_problem_kind(self.problem_kind)
            if worker_kind != "BBOB":
                raise ValueError(
                    f"a fid axis needs a BBOB problem kind, not {self.problem_kind!r} "
                    "(MA-BBOB mixtures have no single function id)"
                )
            if "fid" in dict(self.extra_builder_kwargs):
                raise ValueError(
                    "fids and extra_builder_kwargs['fid'] both set — use fids for the axis "
                    "and extra_builder_kwargs only for a builder argument that has no field"
                )
        object.__setattr__(self, "fids", fids)
        if self.sealed:
            check_sealed_mabbob_instances(self.instances)
            if any(getattr(self, k) != v for k, v in _SEALED_FIELDS.items()):
                raise ValueError(
                    "a sealed battery is exactly make_sealed_battery(): no sub-selection or other "
                    "change (dims, reps, budget, ...) of the sealed set (panobbgo.sealed)"
                )
        else:
            bad = [int(i) for i in self.instances if not is_dev_instance_id(int(i))]
            if bad:
                raise ValueError(
                    f"instance ids {bad} are outside the development window 0 <= id < {DEV_INSTANCE_LIMIT} "
                    "(panobbgo.sealed: ids outside it alias the sealed set's); the sealed test set is "
                    "make_sealed_battery(), for claims only"
                )

    def aocc_bounds(self, log_lo: float = AOCC_LOG_LO, log_hi: float = AOCC_LOG_HI) -> Tuple[float, float]:
        """The AOCC bounds to score this battery with: its own where it has them.

        A caller's bound that differs from both the standard one and the
        battery's is refused rather than silently overridden.
        """
        out = []
        for mine, given, std, label in (
            (self.log_lo, log_lo, AOCC_LOG_LO, "log_lo"),
            (self.log_hi, log_hi, AOCC_LOG_HI, "log_hi"),
        ):
            if mine is None:
                out.append(float(given))
            elif float(given) in (float(std), float(mine)):
                out.append(float(mine))
            else:
                raise ValueError(f"battery {self.name!r} is scored with {label}={mine}, not {given}")
        return out[0], out[1]

    @property
    def is_noisy(self) -> bool:
        return self.problem_kind in NOISY_PROBLEM_KINDS

    @property
    def fid_axis(self) -> Tuple[Optional[int], ...]:
        """The function axis to iterate — ``(None,)`` when there is none.

        Lets one loop cover both shapes: without a fid axis the single
        ``None`` reproduces exactly the old single pass, and ``None`` is
        also what lands in the record and the seed payloads (where it is
        omitted), so nothing about a fid-less battery changes.
        """
        return self.fids if self.fids else (None,)

    def pair_count(self, n_strategies: int) -> int:
        return n_strategies * len(self.fid_axis) * len(self.dims) * len(self.instances) * self.reps

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
# The function axis: plain BBOB, all 24 functions
# ---------------------------------------------------------------------------


def make_bbob_battery(
    dims: Sequence[int] = (2, 5),
    instances: Sequence[int] = (0, 1, 2),
    budget_multiplier: int = 200,
    fids: Sequence[int] = ALL_BBOB_FIDS,
) -> IOHBatterySpec:
    """The 24 noiseless BBOB functions as a first-class battery axis.

    ``planning/DESIGN_suite_2026-09-14.md`` gap 1.  Every number of
    §44–§51 was measured on MA-BBOB *mixtures* — affine combinations of
    two BBOB functions, broad by construction but not attributable to a
    landscape class.  This battery is the plain suite instead, so a
    result can be reported per COCO class (:func:`bbob_class_of`) and the
    regime table can be asked whether it keys on the landscape rather
    than only on the wallet.

    Defaults: ``24 × 2 × 3 = 144`` cells per strategy and seed — 14× the
    standard battery's 10 — at ``200·dim``, the budget where §44/§50 put
    the sharing effect.  At the measured ~0.7 s per run that is ~1.7 min
    per strategy per seed.  Screens stay on the small MA-BBOB cube; this
    is the *re-test* instrument.
    """
    return IOHBatterySpec(
        name="ioh-bbob",
        problem_kind="BBOB",
        dims=tuple(int(d) for d in dims),
        instances=tuple(int(i) for i in instances),
        reps=1,
        budget_multiplier=int(budget_multiplier),
        fids=tuple(fids),
    )


# ---------------------------------------------------------------------------
# Dimensions 30/40, a bbob-largescale slice, and the sealed test set
# ---------------------------------------------------------------------------
#
# ``planning/DESIGN_roadmap_2026-09-26.md`` §3.3.  Opt-in presets: none of
# the batteries above changes.  Measured cost (2026-09-26, 16-core laptop,
# ``nice -n 15``, ``sync_eval``, one BLAS thread, light load; one run of each
# ``make_ioh_strategies`` spec): MA-BBOB at d = 40 and 500·d (20 000
# evaluations) takes 6.6 s (CMA-ES) to 8.6 s (the portfolios), 0.3-0.45 ms
# per evaluation, most of it the worker round-trip; plain BBOB at 500·d
# takes ≈ 6 s per run at d = 80 and 20-42 s at d = 160.  Building an ``ioh``
# problem at d = 160 takes 0.03 s (BBOB) and 0.8 s (MA-BBOB).
#
# None of the GP / QuadraticWLS / ``Nearby(quadratic=True)`` heuristics is
# for d >= 30 (a Nearby quadratic fit takes seconds and up to 1 GB per new
# best at d = 160); ``ioh_benchmark.py`` refuses ``--legacy`` with these
# batteries.


#: One function per COCO class for the largescale slice: separable
#: ellipsoid, Rosenbrock, rotated ellipsoid, rotated Rastrigin, Gallagher 101.
LARGESCALE_FIDS: Tuple[int, ...] = (2, 8, 10, 15, 21)


def make_large_battery(
    dims: Sequence[int] = (30, 40),
    instances: Sequence[int] = (0, 1, 2),
    budget_multiplier: int = 500,
) -> IOHBatterySpec:
    """Noiseless MA-BBOB at *d* = 30 and 40, at the standard budget ``500·d``.

    Three instances per dim (the :func:`make_highdim_battery` shape) and
    15 000 / 20 000 evaluations per run.  Cost: ≈ 7-9 s per run at
    *d* = 40 (measured 2026-09-26, laptop, one BLAS thread), so the 2 × 3
    cube is ≈ 1 min per strategy per seed.  At this budget a *d* = 40 run
    is far from converged: this battery asks how fast a strategy makes
    progress, not whether it finishes.
    """
    return IOHBatterySpec(
        name="ioh-large",
        problem_kind="MA-BBOB",
        dims=tuple(int(d) for d in dims),
        instances=tuple(int(i) for i in instances),
        reps=1,
        budget_multiplier=int(budget_multiplier),
    )


#: Upper AOCC bound of the largescale slice (standard: ``AOCC_LOG_HI = 2``).
LARGESCALE_LOG_HI: float = 6.0

#: Budget multiplier of the largescale slice.
LARGESCALE_BUDGET_MULTIPLIER: int = 500


def make_largescale_battery(
    dims: Sequence[int] = (80, 160),
    instances: Sequence[int] = (0, 1, 2),
    budget_multiplier: int = LARGESCALE_BUDGET_MULTIPLIER,
    fids: Sequence[int] = LARGESCALE_FIDS,
) -> IOHBatterySpec:
    """A ``bbob-largescale``-style slice: five BBOB functions at *d* = 80 and 160.

    Plain ``ioh`` BBOB with its full ``d × d`` rotations.  COCO's
    ``bbob-largescale`` suite replaces those with permuted block-diagonal
    rotations (and ``ioh`` does not ship it), so the function values are
    **not** comparable with published ``bbob-largescale`` data; the
    landscapes are the same classes.  A rotation at *d* = 160 is 200 KB and
    built in ~0.03 s, so the full rotation costs nothing here.

    **Scored on a wider AOCC range**, targets ``[1e-8, 1e6]``
    (:data:`LARGESCALE_LOG_HI`, stored in the spec and in every result):
    with the standard ``1e2`` upper bound CMA-ES scored exactly 0 on f2,
    f10 and f15 at *d* = 80 (200·d) and on four of the five functions at
    *d* = 160 — the battery could not rank anything.  At ``500·d`` and
    ``log_hi = 6`` every function scores (CMA-ES, instance 0: 0.10-0.43 at
    *d* = 80, 0.08-0.36 at *d* = 160).  Not comparable with a
    standard-bound AOCC; ``ioh_benchmark.py compare`` refuses the mix.

    ``5 × 2 × 3 = 30`` runs per strategy and seed at ``500·d`` (40 000 /
    80 000 evaluations).  Measured 2026-09-26 (laptop, one BLAS thread):
    ≈ 6 s per run at *d* = 80 and 20-42 s at *d* = 160, so ≈ 7-10 min per
    strategy per seed.
    """
    return IOHBatterySpec(
        name="ioh-bbob-largescale",
        problem_kind="BBOB",
        dims=tuple(int(d) for d in dims),
        instances=tuple(int(i) for i in instances),
        reps=1,
        budget_multiplier=int(budget_multiplier),
        fids=tuple(fids),
        log_hi=LARGESCALE_LOG_HI,
    )


def make_sealed_battery() -> IOHBatterySpec:
    """The sealed MA-BBOB test set: **for claims only, never for tuning.**

    Twenty fresh MA-BBOB instances (:data:`panobbgo.sealed.SEALED_MABBOB_INSTANCES`,
    from a reserved id range no development battery can reach) at
    *d* ∈ {2, 5, 10, 20, 30, 40}, ``500·d``.  :func:`run_ioh_harness`
    prints a warning banner when it runs it, and marks the result
    ``sealed``.  Rules and the unit of inference:
    ``doc/dev/benchmarking.md``, "The sealed test set".

    ``6 × 20 = 120`` runs per strategy and seed, 1.07 M evaluations: ≈ 7 min
    per strategy per seed at the 0.3-0.45 ms per evaluation measured with one
    BLAS thread.  It takes no arguments, and :class:`IOHBatterySpec` refuses
    any variant of it: a sealed set with knobs becomes a family of sets, and
    a set picked from a family is a tuned set.
    """
    return IOHBatterySpec(**_SEALED_FIELDS)


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
        # both arms warm-start, so their ``required_analyzers`` declare it
        # and ``StrategyBase.initialize`` installs it on demand.  The strategy
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


#: Start of ``IOHRunRecord.error`` for a run the tracker cut off at its
#: deadline (still scored).  A wedged worker's ``TimeoutError: IOH worker did
#: not answer in time`` is a crash, not this.
TIMEOUT_ERROR_PREFIX = "TimeoutError: stopped after"

#: Start of ``IOHRunRecord.error`` for a run that stopped by itself below its
#: budget (no deadline, no exception) — typically every arm stopped emitting.
#: It is scored like any short run (the rest of the budget at its final
#: best), but it is not a clean run and must not read as one.
EARLY_END_ERROR_PREFIX = "EndedEarly: stopped at"


#: What a result's AOCC is taken of (:attr:`IOHHarnessResult.scored`).  The
#: IOH and family tracks score their objective (the true value on a noisy
#: battery, the penalty value ``f + 100·cv`` on a constrained family); the
#: real-world track (:mod:`panobbgo.harness_realworld`) scores the feasible
#: relative gap.  ``compare`` refuses to put two different quantities side by side.
SCORED_OBJECTIVE: str = "objective"
SCORED_RELATIVE_FEASIBLE_GAP: str = "relative_feasible_gap"


@dataclass
class IOHRunRecord:
    """Result of one (problem kind, fid, dim, instance, strategy, rep) run."""

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
    #: BBOB function id of this run, or ``None`` on a battery with no fid
    #: axis (every MA-BBOB and families row, and every result file written
    #: before 2026-09-14 — which is why it is optional and defaults to
    #: ``None``: ``IOHRunRecord(**row)`` keeps loading old JSON unchanged).
    fid: Optional[int] = None
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
    #: AOCC over *virtual time* (:func:`~panobbgo.ioh_runner.aocc_virtual_time`)
    #: of a virtual-clock run (``IOHHarnessResult.virtual``); ``None`` otherwise.
    aocc_time: Optional[float] = None
    #: ``True`` for a run on the sealed test set (:mod:`panobbgo.sealed`).
    sealed: bool = False
    # --- constrained real-world runs only (None elsewhere) ---------------
    #: Did the run find a feasible point?
    feasible: Optional[bool] = None
    #: Smallest mean constraint violation seen (the CEC 2020 ``nu``; 0 once feasible).
    best_violation: Optional[float] = None
    #: 1-based index of the first feasible evaluation, ``None`` if there was none.
    first_feasible_eval: Optional[int] = None
    #: Feature dicts at the budget checkpoints (:mod:`panobbgo.features`), one
    #: per checkpoint reached, with ``--log-features``; ``None`` (and left out
    #: of the JSON) otherwise.
    features: Optional[List[Dict[str, Any]]] = None

    @property
    def precision(self) -> float:
        return float(self.best_fx - self.f_opt)

    @property
    def timed_out(self) -> bool:
        """Cut off at its wall-clock deadline; ``aocc`` is scored on the trace up to it."""
        return self.error is not None and self.error.startswith(TIMEOUT_ERROR_PREFIX)

    @property
    def ended_early(self) -> bool:
        """Stopped by itself below its budget; ``aocc`` is scored on the short trace."""
        return self.error is not None and self.error.startswith(EARLY_END_ERROR_PREFIX)

    @property
    def crashed(self) -> bool:
        """Ended by an exception; ``aocc`` is then 0 (nothing was scored)."""
        return self.error is not None and not self.timed_out and not self.ended_early


def time_score(r: IOHRunRecord) -> Optional[float]:
    """A run's AOCC over virtual time for the aggregates: 0 for a crash, ``None`` if it has none."""
    return 0.0 if r.crashed else r.aocc_time


def run_record_to_dict(r: IOHRunRecord) -> Dict[str, Any]:
    """JSON row of a record; ``features`` only when logged, so default result files keep their shape."""
    row = asdict(r)
    if row.get("features") is None:
        row.pop("features", None)
    return row


def run_record_from_dict(row: Dict[str, Any]) -> IOHRunRecord:
    """:class:`IOHRunRecord` from a JSON row, ignoring keys this version does not know (newer files)."""
    known = {f.name for f in fields(IOHRunRecord)}
    return IOHRunRecord(**{k: v for k, v in row.items() if k in known})


def warn_missing_time_scores(result: "IOHHarnessResult") -> None:
    """Warn when strategies of a virtual-clock result have no time score (see ``strategies_without_time_score``)."""
    missing = result.strategies_without_time_score()
    if missing:
        warnings.warn(
            "no AOCC over virtual time for %s: these strategies did not run on the virtual clock "
            "and are left out of the time aggregates" % ", ".join(missing),
            RuntimeWarning,
            stacklevel=2,
        )


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
    #: Virtual-clock settings (:meth:`VirtualSpec.to_dict
    #: <panobbgo.virtual_clock.VirtualSpec.to_dict>`: workers, duration
    #: model) when the runs were simulated on ``q`` virtual workers, else
    #: ``None``.  Each run then also carries ``aocc_time``.
    virtual: Optional[Dict[str, Any]] = None
    #: ``True`` when the battery is (or contains) the sealed test set.
    sealed: bool = False
    #: BLAS / OpenMP threads the runs were pinned to
    #: (:data:`panobbgo.local_run.BLAS_THREADS`); ``None`` in older files.
    blas_threads: Optional[int] = None
    #: The quantity the AOCC is taken of: :data:`SCORED_OBJECTIVE` (every
    #: track but one, and every older file) or
    #: :data:`SCORED_RELATIVE_FEASIBLE_GAP` (the real-world track).
    scored: str = SCORED_OBJECTIVE
    #: The floating-point environment the runs were measured in
    #: (:func:`panobbgo.fp_env.collect`) and its short id
    #: (:func:`panobbgo.fp_env.fp_env_id`); ``None`` in older files.  A
    #: seeded result is bit-reproducible only within one ``fp_env_id``.
    fp_env: Optional[Dict[str, Any]] = None
    fp_env_id: Optional[str] = None

    # Every run counts, the way ``benchmarks/_screen.fold`` counts them: a
    # timed-out run with its AOCC up to the deadline, a crashed run with
    # AOCC 0.  Dropping them would reward a strategy that dies on its
    # hardest cells and pair means over different cell sets.

    @property
    def mean_aocc(self) -> float:
        """Mean AOCC over all runs (crashes score 0, timeouts their partial AOCC)."""
        scores = [r.aocc for r in self.runs]
        return float(np.mean(scores)) if scores else 0.0

    @property
    def mean_aocc_time(self) -> Optional[float]:
        """Mean AOCC over virtual time over the runs that have one (:func:`time_score`).

        ``None`` unless a virtual-clock run with at least one time score.
        Runs without one (a strategy that does not run on the clock) are
        left out, not counted as 0: see :attr:`n_aocc_time` and
        :meth:`strategies_without_time_score`.
        """
        if self.virtual is None:
            return None
        vals = [v for v in (time_score(r) for r in self.runs) if v is not None]
        return float(np.mean(vals)) if vals else None

    @property
    def n_aocc_time(self) -> int:
        """Number of runs behind :attr:`mean_aocc_time`."""
        return sum(time_score(r) is not None for r in self.runs) if self.virtual is not None else 0

    def per_strategy_aocc_time(self) -> Dict[str, float]:
        """``{strategy_name: mean AOCC over virtual time}`` over the runs that have one."""
        if self.virtual is None:
            return {}
        return self._mean_aocc_by(lambda r: r.strategy_name, value=time_score)

    def strategies_without_time_score(self) -> List[str]:
        """Strategies of a virtual-clock result with no time score at all (they did not run on the clock)."""
        if self.virtual is None:
            return []
        scored = set(self.per_strategy_aocc_time())
        return sorted({r.strategy_name for r in self.runs} - scored)

    def per_strategy_aocc(self) -> Dict[str, float]:
        """Return ``{strategy_name: mean AOCC over all runs}`` (crashes score 0)."""
        return self._mean_aocc_by(lambda r: r.strategy_name)

    def per_strategy_counts(self) -> Dict[str, Dict[str, int]]:
        """``{strategy: {"n": runs, "crashed": ..., "timed_out": ..., "ended_early": ...}}`` behind each mean."""
        out: Dict[str, Dict[str, int]] = {}
        for r in self.runs:
            c = out.setdefault(r.strategy_name, {"n": 0, "crashed": 0, "timed_out": 0, "ended_early": 0})
            c["n"] += 1
            c["crashed"] += int(r.crashed)
            c["timed_out"] += int(r.timed_out)
            c["ended_early"] += int(r.ended_early)
        return out

    def _mean_aocc_by(
        self,
        key_fn: Callable[[IOHRunRecord], Any],
        value: Callable[[IOHRunRecord], Optional[float]] = lambda r: r.aocc,
    ) -> Dict[Any, float]:
        """Mean ``value(run)`` (default AOCC) grouped by ``key_fn(run)``; a ``None`` key or value skips the run."""
        by: Dict[Any, List[float]] = {}
        for r in self.runs:
            key = key_fn(r)
            v = value(r)
            if key is not None and v is not None:
                by.setdefault(key, []).append(v)
        return {k: float(np.mean(v)) for k, v in by.items()}

    def per_strategy_per_dim_aocc(self) -> Dict[Tuple[str, int], float]:
        return self._mean_aocc_by(lambda r: (r.strategy_name, r.dim))

    def per_strategy_per_class_aocc(self) -> Dict[Tuple[str, str], float]:
        """``{(strategy, COCO class tag): mean AOCC}`` — empty without a fid axis."""
        return self._mean_aocc_by(lambda r: None if r.fid is None else (r.strategy_name, bbob_class_of(r.fid)))

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
            "sealed": self.sealed,
            "blas_threads": self.blas_threads,
            "scored": self.scored,
            "fp_env_id": self.fp_env_id,
            "fp_env": self.fp_env,
            **(
                {}
                if self.virtual is None
                else {
                    "virtual": self.virtual,
                    "mean_aocc_time": self.mean_aocc_time,
                    "per_strategy_aocc_time": self.per_strategy_aocc_time(),
                }
            ),
            "runs": [run_record_to_dict(r) for r in self.runs],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=float)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "IOHHarnessResult":
        runs = [run_record_from_dict(r) for r in d.get("runs", [])]
        return cls(
            battery_name=d["battery_name"],
            problem_kind=d["problem_kind"],
            log_lo=d.get("log_lo", AOCC_LOG_LO),
            log_hi=d.get("log_hi", AOCC_LOG_HI),
            runs=runs,
            timestamp=d.get("timestamp", time.time()),
            sync_eval=bool(d.get("sync_eval", False)),
            virtual=d.get("virtual"),
            sealed=bool(d.get("sealed", False)),
            blas_threads=d.get("blas_threads"),
            scored=d.get("scored", SCORED_OBJECTIVE),
            fp_env=d.get("fp_env"),
            fp_env_id=d.get("fp_env_id"),
        )

    def print_summary(self) -> None:
        print(f"\nIOH battery: {self.battery_name}  ({self.problem_kind})")
        if self.sealed:
            print("  SEALED TEST SET: for claims only, never for tuning")
        if self.virtual is not None:
            print(f"  eval mode:    virtual clock {self.virtual}")
        elif self.sync_eval:
            print("  eval mode:    sync (deterministic result batches)")
        print(f"  log targets:  [10^{self.log_lo:.0f}, 10^{self.log_hi:.0f}]")
        print(f"  mean AOCC:    {self.mean_aocc:.4f}    over {len(self.runs)} run(s)")
        if self.mean_aocc_time is not None:
            print(
                f"  AOCC (time):  {self.mean_aocc_time:.4f}    over virtual time, horizon budget/q mean durations "
                f"({self.n_aocc_time} run(s))"
            )
        missing = self.strategies_without_time_score()
        if missing:
            print(f"  (no time score, not on the virtual clock: {', '.join(missing)})")
        n_crash = sum(r.crashed for r in self.runs)
        n_timeout = sum(r.timed_out for r in self.runs)
        n_early = sum(r.ended_early for r in self.runs)
        if n_crash or n_timeout or n_early:
            print(
                f"  incl.:        {n_crash} crashed (AOCC 0), {n_timeout} timed out (AOCC up to the deadline), "
                f"{n_early} ended early (below budget)"
            )
        obs = [r.aocc_observed for r in self.runs if r.aocc_observed is not None]
        if obs:
            reco = [r.aocc_reco for r in self.runs if r.aocc_reco is not None]
            print(f"  (noisy: AOCC is on the TRUE value; observed {float(np.mean(obs)):.4f}", end="")
            print(f", recommendation {float(np.mean(reco)):.4f})" if reco else ")")
        print("\n  per strategy:")
        counts = self.per_strategy_counts()
        per_time = self.per_strategy_aocc_time()
        for name, val in sorted(self.per_strategy_aocc().items(), key=lambda kv: -kv[1]):
            c = counts[name]
            bad = [f"{c.get(k, 0)} {k.replace('_', ' ')}" for k in ("crashed", "timed_out", "ended_early") if c.get(k)]
            t = f"  time {per_time[name]:.4f}" if name in per_time else ""
            print(f"    {name:32s}  {val:.4f}{t}  (n={c['n']}{', ' + ', '.join(bad) if bad else ''})")
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
        per_class = self.per_strategy_per_class_aocc()
        if per_class:
            # Only the classes actually present in the run: a battery cut
            # to a few fids would otherwise print three columns of NaN.
            classes = bbob_classes_present(r.fid for r in self.runs if r.fid is not None)
            print("\n  per (strategy, COCO class):")
            print("    " + "strategy".ljust(32) + "  " + "  ".join(f"{c:>19s}" for c in classes))
            for s in sorted({s for s, _ in per_class}):
                vals = "  ".join(f"{per_class.get((s, c), float('nan')):19.4f}" for c in classes)
                print("    " + s.ljust(32) + "  " + vals)
        if self.scored != SCORED_OBJECTIVE:
            self.print_problem_table()

    def print_problem_table(self) -> None:
        """Per problem: mean AOCC, and how many runs found a feasible point, per strategy.

        Printed by :meth:`print_summary` for the real-world track, where a
        problem is a named model rather than an instance of a class.
        """
        problems = list(dict.fromkeys(r.problem_kind for r in self.runs))
        strategies = sorted({r.strategy_name for r in self.runs})
        if not problems:
            return
        print(f"\n  per problem (AOCC, feasible runs / runs; scored: {self.scored}):")
        width = max(12, max(len(s) for s in strategies))
        print("    " + "problem".ljust(34) + "  ".join(s[:width].rjust(width) for s in strategies))
        for prob in problems:
            cells = []
            for strat in strategies:
                rs = [r for r in self.runs if r.problem_kind == prob and r.strategy_name == strat]
                if not rs:
                    cells.append("-".rjust(width))
                    continue
                n_feas = sum(1 for r in rs if r.feasible)
                cells.append(f"{float(np.mean([r.aocc for r in rs])):.4f} {n_feas}/{len(rs)}".rjust(width))
            print("    " + prob[:33].ljust(34) + "  ".join(cells))


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
    #: Virtual-clock settings (see :class:`IOHHarnessResult.virtual`).
    virtual: Optional[Dict[str, Any]] = None
    #: See :attr:`IOHHarnessResult.sealed`, :attr:`~IOHHarnessResult.blas_threads`,
    #: :attr:`~IOHHarnessResult.fp_env` and :attr:`~IOHHarnessResult.fp_env_id`.
    sealed: bool = False
    blas_threads: Optional[int] = None
    fp_env: Optional[Dict[str, Any]] = None
    fp_env_id: Optional[str] = None

    @property
    def mean_aocc(self) -> float:
        """Mean AOCC with equal weight per seed."""
        vals = [r.mean_aocc for r in self.results]
        return float(np.mean(vals)) if vals else 0.0

    @property
    def mean_aocc_time(self) -> Optional[float]:
        """Mean AOCC over virtual time with equal weight per seed; ``None`` without one."""
        vals = [v for v in (r.mean_aocc_time for r in self.results) if v is not None]
        return float(np.mean(vals)) if vals else None

    def per_strategy_seed_aocc(self) -> Dict[str, List[float]]:
        """Return ``{strategy: [mean AOCC at base_seeds[i]]}``.

        Only strategies present in every per-seed result are returned —
        a ragged strategy (missing from some seed's battery) cannot be
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
            "sealed": self.sealed,
            "blas_threads": self.blas_threads,
            "fp_env_id": self.fp_env_id,
            "fp_env": self.fp_env,
            **({} if self.virtual is None else {"virtual": self.virtual, "mean_aocc_time": self.mean_aocc_time}),
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
            virtual=d.get("virtual"),
            sealed=bool(d.get("sealed", False)),
            blas_threads=d.get("blas_threads"),
            fp_env=d.get("fp_env"),
            fp_env_id=d.get("fp_env_id"),
        )

    def per_strategy_per_dim_aocc(self) -> Dict[Tuple[str, int], List[float]]:
        """``{(strategy, dim): [mean AOCC at base_seeds[i]]}`` — the per-dim report of a multi-dim battery."""
        out: Dict[Tuple[str, int], List[float]] = {}
        for res in self.results:
            for key, val in res.per_strategy_per_dim_aocc().items():
                out.setdefault(key, []).append(val)
        return out

    def print_summary(self) -> None:
        print(f"\nIOH multi-seed battery: {self.battery_name}  ({self.problem_kind})")
        if self.sealed:
            print("  SEALED TEST SET: for claims only, never for tuning")
        if self.sync_eval:
            print("  eval mode:    sync (deterministic result batches)")
        print(f"  seeds ({len(self.base_seeds)}): {', '.join(str(s) for s in self.base_seeds)}")
        n_runs = sum(len(r.runs) for r in self.results)
        print(f"  mean AOCC:    {self.mean_aocc:.4f}    over {len(self.base_seeds)} seed(s), {n_runs} run(s)")
        if self.mean_aocc_time is not None:
            print(f"  AOCC (time):  {self.mean_aocc_time:.4f}    over virtual time ({self.virtual})")
        print("\n  per strategy (mean ± sd across seeds):")
        matrix = self.per_strategy_seed_aocc()
        for name, vals in sorted(matrix.items(), key=lambda kv: -float(np.mean(kv[1]))):
            arr = np.asarray(vals, dtype=np.float64)
            sd = float(arr.std(ddof=1)) if arr.size >= 2 else float("nan")
            print(f"    {name:32s}  {float(arr.mean()):.4f} ± {sd:.4f}")
        per_dim = self.per_strategy_per_dim_aocc()
        dims = sorted({d for _s, d in per_dim})
        if len(dims) > 1:
            print("\n  per dimension (mean across seeds):")
            print("    " + "strategy".ljust(32) + "  " + "  ".join(f"  d={d:<3d}" for d in dims))
            for name in sorted(matrix, key=lambda n: -float(np.mean(matrix[n]))):
                cells = [per_dim.get((name, d)) for d in dims]
                print(
                    "    "
                    + name.ljust(32)
                    + "  "
                    + "  ".join(f"  {float(np.mean(c)):.4f}" if c else "     nan" for c in cells)
                )


def run_ioh_harness_multi_seed(
    strategies: Sequence[StrategySpec],
    battery: IOHBatterySpec,
    base_seeds: Sequence[int],
    *,
    log_lo: float = AOCC_LOG_LO,
    log_hi: float = AOCC_LOG_HI,
    timeout_s: Optional[float] = None,
    progress: bool = True,
    sync_eval: bool = True,
    jobs: int = 1,
    virtual: Optional[VirtualSpec] = None,
    log_features: Optional[FeatureLogSpec] = None,
) -> IOHMultiSeedResult:
    """Run :func:`run_ioh_harness` once per seed in ``base_seeds``.

    ``jobs > 1`` runs each seed's cells in that many worker processes (see
    :func:`run_ioh_harness`).
    """
    if not base_seeds:
        raise ValueError("base_seeds must be non-empty")
    log_lo, log_hi = battery.aocc_bounds(log_lo, log_hi)
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
                jobs=jobs,
                virtual=virtual,
                log_features=log_features,
            )
        )
    return IOHMultiSeedResult(
        battery_name=battery.name,
        problem_kind=battery.problem_kind,
        log_lo=log_lo,
        log_hi=log_hi,
        base_seeds=[int(s) for s in base_seeds],
        results=results,
        sync_eval=sync_eval or virtual is not None,
        virtual=None if virtual is None else virtual.to_dict(),
        sealed=battery.sealed,
        blas_threads=BLAS_THREADS,
        **fp_env.current(),
    )


def t_ci(deltas: Sequence[float], level: float = 0.95) -> Tuple[float, float]:
    """``(mean, half-width)`` of a two-sided Student-t CI on the mean of ``deltas``.

    The one t-interval every screen and :func:`paired_seed_stats` use: the
    critical value is ``scipy.stats.t.ppf`` at ``len(deltas) - 1`` degrees
    of freedom — keyed on the deltas actually present, not on the number of
    seeds requested (a cell missing on one seed shrinks ``n``).  The
    half-width is NaN below two deltas; the mean is NaN for none.
    """
    arr = np.asarray(list(deltas), dtype=np.float64)
    n = int(arr.size)
    if n == 0:
        return float("nan"), float("nan")
    mean = float(arr.mean())
    if n < 2:
        return mean, float("nan")
    from scipy.stats import t as t_dist

    crit = float(t_dist.ppf(0.5 + level / 2.0, n - 1))
    return mean, crit * float(arr.std(ddof=1)) / float(np.sqrt(n))


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
        mean_delta, half = t_ci(deltas)
        if n >= 2:
            sd = float(deltas.std(ddof=1))
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


def _hash_seed(payload: str, fid: Optional[int]) -> int:
    """SHA-256 seed of ``payload``, with the ``|f<fid>`` segment appended only when a fid is given."""
    if fid is not None:
        payload += f"|f{fid}"
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:4], "little")


def _derive_seed(
    base_seed: int,
    problem_kind: str,
    dim: int,
    instance: int,
    strategy_name: str,
    rep: int,
    noise_seed: Optional[int] = None,
    fid: Optional[int] = None,
) -> int:
    """Per-run RNG seed.

    ``noise_seed`` is appended only when it is not ``None``, so every
    noiseless battery keeps the seeds it had before noisy kinds existed —
    ``tests/test_harness_reproducibility.py`` and every historical
    comparison depend on that.  ``fid`` follows the same rule for the same
    reason: a battery with no function axis passes ``None`` and its
    payload is byte-identical to the pre-2026-09-14 one.  When there *is*
    a function axis the segment is mandatory — without it every function
    of a battery would share one RNG stream and the runs on f1 and f2
    would be the same draw, which is exactly the bug the axis must not
    introduce.

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
    return _hash_seed(payload, fid)


def _derive_noise_seed(
    base_seed: int,
    problem_kind: str,
    dim: int,
    instance: int,
    rep: int,
    fid: Optional[int] = None,
) -> int:
    """Seed of the noise realisation for one (fid, dim, instance, rep) cell.

    Deliberately **not** a function of the strategy: every arm on a cell
    must face the identical noisy function or the paired comparison the
    whole harness rests on (``AGENTS.md`` "Statistical rigor",
    ``StrategySpec.seed_name``) leaks the noise realisation into the
    delta.  It *is* a function of the instance and the rep, so the five
    instances of a noisy battery are five different noise realisations
    rather than one repeated.

    ``fid`` is appended only when present (same backward-compatibility
    rule as :func:`_derive_seed`), and for the same statistical reason:
    two functions of one battery must not share a noise realisation.
    """
    return _hash_seed(f"noise|{base_seed}|{problem_kind}|{dim}|{instance}|{rep}", fid)


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


#: Seconds past a run's ``timeout_s`` that an in-flight IOH worker call
#: may still take before the worker is killed (:attr:`IOHProblem.deadline
#: <panobbgo.lib.ioh_wrapper.IOHProblem.deadline>`).
WORKER_GRACE_S: float = 30.0


@dataclass
class _TrackedRun:
    """What :func:`_run_tracked` measured; the caller wraps it in a record."""

    n_evals: int
    best_fx: float
    aocc: float
    trace_evals: List[int]
    trace_fx: List[float]
    aocc_observed: Optional[float] = None
    aocc_reco: Optional[float] = None
    error: Optional[str] = None
    aocc_time: Optional[float] = None
    features: Optional[List[Dict[str, Any]]] = None
    #: Wall time the feature logger took (not recorded: records stay deterministic).
    features_s: float = 0.0


def refuse_sealed_features(log_features: Optional[FeatureLogSpec], name: str) -> None:
    """Raise if features are to be logged on a sealed battery: they are training data, and the set is for claims only."""
    if log_features is not None:
        raise ValueError(f"feature logging is refused on the sealed battery {name!r}: never train on the sealed set")


def _is_sealed_problem(problem: Any) -> bool:
    """A sealed problem: marked ``sealed`` (families), or an IOH problem on a sealed MA-BBOB id (wrappers unwrapped)."""
    p = problem
    for _ in range(8):
        if getattr(p, "sealed", False) or getattr(p, "ioh_instance", None) in SEALED_MABBOB_INSTANCES:
            return True
        inner = getattr(p, "inner", None)
        if inner is None or inner is p:
            return False
        p = inner
    return False


def wall_timeout_for(strategy_spec: StrategySpec, timeout_s: Optional[float]) -> Optional[float]:
    """The per-run wall-clock deadline for ``strategy_spec``: ``None`` for a ``no_wall_timeout`` class.

    The expensive-track baselines (:mod:`panobbgo.harness_baselines_bo`)
    spend minutes per run in GP fits by design; a deadline would score them
    on a stub.  Every other strategy keeps ``timeout_s``.
    """
    if getattr(strategy_spec.strategy_class, "no_wall_timeout", False):
        return None
    return timeout_s


def _run_tracked(*args: Any, **kwargs: Any) -> _TrackedRun:
    """:func:`_run_tracked_unpinned` with BLAS pinned to :data:`~panobbgo.local_run.BLAS_THREADS`.

    Every AOCC run of both tracks goes through here, so no entry point can
    measure with an unpinned BLAS pool: at ``d >= 80`` CMA-ES's ``eigh``
    gives thread-count-dependent results, and an oversubscribed pool is
    7-70x slower under load (``panobbgo.local_run``).

    The pin is process-wide and stays after the run
    (:func:`~panobbgo.local_run.pin_blas`): a thread abandoned by
    ``evaluation.timeout`` can still be inside BLAS when the run returns,
    and raising the OpenBLAS thread count under it is not safe.
    """
    pin_blas()
    return _run_tracked_unpinned(*args, **kwargs)


def _run_tracked_unpinned(
    strategy_spec: StrategySpec,
    problem: Any,
    tracker: IOHTracker,
    *,
    f_opt: float,
    budget: int,
    seed: int,
    sync_eval: bool,
    log_lo: float,
    log_hi: float,
    timeout_s: Optional[float],
    virtual: Optional[VirtualSpec] = None,
    log_features: Optional[FeatureLogSpec] = None,
) -> _TrackedRun:
    """Run one strategy under an installed tracker and score its trace.

    The one run driver of the AOCC tracks (IOH / MA-BBOB here, the
    generated families in :mod:`panobbgo.harness_families`): strategy
    construction with the budget, the anytime settings, the wall-clock
    timeout, AOCC and trace downsampling.  ``tracker`` is already wrapped
    around ``problem`` (an :class:`IOHTracker` or a subclass such as the
    families' penalty tracker) and is restored here.  Exceptions
    propagate; the caller records them.

    With ``virtual`` the strategy runs on the virtual clock
    (:mod:`panobbgo.virtual_clock`) with the tracker as its observer: the
    traces are recorded in completion order, failed and timed-out calls
    count as spent evaluations, and ``aocc_time`` is scored as well
    (:func:`~panobbgo.ioh_runner.aocc_virtual_time`).

    With ``log_features`` a :class:`~panobbgo.features.FeatureLogger`
    observes the main loop and the run's ``features`` holds its checkpoint
    dicts; the run itself is unchanged (same evaluations, same random
    streams).  A strategy without a main loop (the external baselines)
    records none.
    """
    np.random.seed(seed)
    logger: Optional[FeatureLogger] = None
    try:
        if _is_sealed_problem(problem):
            # Defence in depth behind the battery-level refusals (inside the
            # ``try``: the tracker is restored either way).
            refuse_sealed_features(log_features, str(getattr(problem, "family", problem)))
        # The budget must reach the config *before* the heuristics are
        # constructed — budget-adaptive arms (``NP_init="auto"``) size
        # themselves from ``config.max_eval`` in their constructor, and
        # would otherwise read Config's default (1000) instead of the
        # battery's ``budget_multiplier * dim``.
        strategy = strategy_spec.create_strategy(problem, seed=seed, max_eval=budget)
        # Harmless belt-and-braces: keeps the invariant for factory-built
        # strategies that rebuild their own config.
        strategy.config.max_eval = budget
        # Deterministic result batches: a seeded run is reproducible
        # (and adaptive-strategy measurement noise roughly halves,
        # 2026-08-09 repeat-sd experiment).  The harness default since
        # 2026-09-25; opt out with --no-sync-eval.
        strategy.config.sync_evaluation = bool(sync_eval)
        # IOH/AOCC is an anytime metric: stopping early on convergence
        # leaves the remaining budget penalised at the final best-fx.
        # Force the strategy to keep producing points until the
        # tracker enforces the budget hard-stop.  (The `Convergence`
        # analyzer still fires its event for any listeners; we simply
        # tell the strategy not to honour it as a stop signal.)
        strategy.config.stop_on_convergence = False
        if virtual is not None:
            virtual.apply(strategy, observer=tracker)
        if log_features is not None and hasattr(strategy, "add_pass_observer"):
            logger = FeatureLogger(
                log_features,
                budget=budget,
                spent=lambda: tracker.n_evals,
                failed=lambda: tracker.n_failed,
                context={"q": 1 if virtual is None else virtual.workers, "noisy": tracker.has_true},
            )
            strategy.add_pass_observer(logger)
        # Past the deadline the tracker stops counting; this also ends the
        # main loop instead of letting it spin through no-op evaluations.
        tracker.on_timeout = getattr(strategy, "request_stop", None)
        try:
            strategy.start()
        except _BudgetExhausted:
            pass
    finally:
        tracker.restore()

    def score(trace: Sequence[float]) -> float:
        return aocc(trace, f_opt=f_opt, log_lo=log_lo, log_hi=log_hi, budget=budget)

    out = _TrackedRun(n_evals=tracker.n_evals, best_fx=tracker.best_fx, aocc=0.0, trace_evals=[], trace_fx=[])
    if log_features is not None:
        out.features = [] if logger is None else logger.records
        out.features_s = 0.0 if logger is None else logger.elapsed_s
        if logger is not None and logger.error is not None:
            warnings.warn(f"feature logging stopped: {logger.error}", RuntimeWarning, stacklevel=2)
    if tracker.has_true:
        # BBOB-noisy convention: the metric is scored on the true,
        # noise-free value; what the optimizer *observed* is a
        # diagnostic (and an optimistically biased one).
        out.best_fx = tracker.best_true_fx
        out.aocc = score(tracker.best_so_far_true)
        out.aocc_observed = score(tracker.best_so_far)
        out.aocc_reco = score(tracker.best_so_far_reco)
        out.trace_evals, out.trace_fx = _downsample_trajectory(tracker.best_so_far_true, budget=budget)
    else:
        out.aocc = score(tracker.best_so_far)
        out.trace_evals, out.trace_fx = _downsample_trajectory(tracker.best_so_far, budget=budget)
    if virtual is not None and tracker.timeline and len(tracker.timeline) == tracker.n_evals:
        # Every counted call carries its completion time (a strategy that
        # evaluates outside the main loop, e.g. an external baseline, does
        # not run on the clock and gets no time score).
        out.aocc_time = aocc_virtual_time(
            tracker.timeline,
            budget=budget,
            workers=virtual.workers,
            f_opt=f_opt,
            mean_duration=virtual.model().mean,
            log_lo=log_lo,
            log_hi=log_hi,
        )
    if tracker.timed_out:
        # Scored above on the trajectory up to the deadline; the error
        # marks it so no table mistakes a cut-off run for a finished one.
        out.error = f"{TIMEOUT_ERROR_PREFIX} {timeout_s:g}s at {out.n_evals}/{budget} evals"
    else:
        out.error = _early_end_error(out.n_evals, getattr(tracker, "_reserved", out.n_evals), budget)
    return out


def _early_end_error(n_evals: int, admitted: int, budget: int) -> Optional[str]:
    """The ``EndedEarly`` marker, or ``None`` for a run that used its budget.

    No deadline and no exception, yet the budget was not spent: the strategy
    stopped by itself (every arm stopped producing).  Scored like any short
    run, but marked so no table reads it as clean.  The test is on the
    evaluations the tracker *admitted* (recorded plus in flight), not only
    the recorded ones: a call abandoned by ``evaluation.timeout`` in threaded
    mode keeps running after the run and is never recorded, but its budget
    slot was dispatched — that run did not end early.
    """
    if max(n_evals, admitted) >= budget:
        return None
    return f"{EARLY_END_ERROR_PREFIX} {n_evals}/{budget} evals"


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
    sync_eval: bool = True,
    noise_seed: Optional[int] = None,
    noise_level: str = "moderate",
    noise_resample: bool = False,
    fid: Optional[int] = None,
    virtual: Optional[VirtualSpec] = None,
    sealed: bool = False,
    log_features: Optional[FeatureLogSpec] = None,
) -> IOHRunRecord:
    """Run one strategy on one (problem, fid, instance) and return its record.

    ``instance`` must be a development id (``0 <= id < 2**20``), or, with
    ``sealed=True``, a sealed MA-BBOB id.  ``sealed=True`` is what
    :func:`run_ioh_harness` passes for the sealed battery; it is an escape
    hatch, not a lock — a caller that sets it deliberately can run one
    sealed cell.  The guard stops *accidental* use (a development id range
    that drifts into the sealed ids, ``scripts/ioh_smoke.py --instance``),
    and a run made this way is at least marked ``sealed`` in its record.
    """
    from panobbgo.lib.ioh_wrapper import IOHProblem

    timeout_s = wall_timeout_for(strategy_spec, timeout_s)

    if problem_kind not in SUPPORTED_PROBLEM_KINDS:
        raise ValueError(f"Unknown problem kind {problem_kind!r}; known: {list(SUPPORTED_PROBLEM_KINDS)}")
    if sealed:
        refuse_sealed_features(log_features, f"{problem_kind} {instance}")
        if int(instance) not in SEALED_MABBOB_INSTANCES or problem_kind != "MA-BBOB":
            raise ValueError(f"sealed=True is for the sealed MA-BBOB instances only, not {problem_kind} {instance}")
    elif not is_dev_instance_id(int(instance)):
        raise ValueError(
            f"instance {instance} is outside the development window 0 <= id < {DEV_INSTANCE_LIMIT} "
            "(panobbgo.sealed); the sealed set runs only as make_sealed_battery()"
        )
    worker_kind, noise_tag = resolve_problem_kind(problem_kind)

    t0 = time.time()
    f_opt = 0.0
    tracked = _TrackedRun(n_evals=0, best_fx=float("inf"), aocc=0.0, trace_evals=[], trace_fx=[])
    if log_features is not None:
        tracked.features = []  # a run that raises still says "logged, nothing recorded"
    problem: Optional[Any] = None

    try:
        # ``fid=None`` is left out entirely rather than forwarded: the
        # wrapper's ``create`` request then carries no ``fid`` key and the
        # worker builds exactly what it built before the axis existed.
        if fid is not None:
            builder_kwargs = {**builder_kwargs, "fid": int(fid)}
        problem = IOHProblem(
            kind=worker_kind,
            instance=instance,
            dim=dim,
            **builder_kwargs,
        )
        f_opt = float(problem.optimum_y)
        if timeout_s is not None:
            # A hung worker must not outlive the run's deadline by more than
            # a grace period: past it, the round-trip is abandoned and the
            # worker killed (the tracker already stops calling it at the
            # deadline, so this only fires on a wedged call).
            problem.deadline = time.monotonic() + float(timeout_s) + WORKER_GRACE_S

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

        # A spec gated by ``regime_gate="oracle"`` learns the battery's
        # noise class here — the one regime feature the strategy cannot
        # read off the problem itself.
        tracked = _run_tracked(
            strategy_spec.with_regime_class(noise_class_of(problem_kind)),
            problem,
            IOHTracker(problem, budget=budget, timeout_s=timeout_s),
            f_opt=f_opt,
            budget=budget,
            seed=seed,
            sync_eval=sync_eval,
            log_lo=log_lo,
            log_hi=log_hi,
            timeout_s=timeout_s,
            virtual=virtual,
            log_features=log_features,
        )
    except Exception as e:  # noqa: BLE001  — record and continue
        tracked.error = f"{type(e).__name__}: {e}"
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
        n_evals=tracked.n_evals,
        best_fx=tracked.best_fx,
        f_opt=f_opt,
        aocc=tracked.aocc,
        elapsed_s=time.time() - t0,
        seed=seed,
        error=tracked.error,
        trace_evals=tracked.trace_evals,
        trace_fx=tracked.trace_fx,
        noise_seed=noise_seed,
        aocc_observed=tracked.aocc_observed,
        aocc_reco=tracked.aocc_reco,
        fid=fid,
        aocc_time=tracked.aocc_time,
        sealed=bool(sealed),
        features=tracked.features,
    )


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


def _print_run(rec: IOHRunRecord, budget: int, prefix: str = "") -> None:
    tag = "ERR " if rec.error else ""
    print(
        f"{prefix}      {tag}AOCC={rec.aocc:.4f}  evals={rec.n_evals}/{budget}  "
        f"prec={rec.precision:.3e}  t={rec.elapsed_s:.1f}s" + (f"  ({rec.error})" if rec.error else ""),
        flush=True,
    )


def _run_tasks_in_pool(
    tasks: List[Dict[str, Any]],
    jobs: int,
    total: int,
    progress: bool,
    fn: Optional[Callable[..., IOHRunRecord]] = None,
) -> List[IOHRunRecord]:
    """Run ``fn(**task)`` (default :func:`_run_one`) for every task in a worker pool; results in task order."""
    from panobbgo.local_run import shared_pool

    done = [0]

    def on_done(i: int, rec: IOHRunRecord) -> None:
        done[0] += 1
        if progress:
            print(
                f"  [{done[0]:>4d}/{total:>4d}] #{i + 1} {rec.problem_kind} dim={rec.dim} "
                f"inst={rec.instance} rep={rec.rep} {rec.strategy_name}",
                flush=True,
            )
            _print_run(rec, rec.budget)

    return shared_pool(jobs).map(fn or _run_one, tasks, on_done=on_done)


def run_ioh_harness(
    strategies: Sequence[StrategySpec],
    battery: IOHBatterySpec,
    *,
    base_seed: int = 42,
    log_lo: float = AOCC_LOG_LO,
    log_hi: float = AOCC_LOG_HI,
    timeout_s: Optional[float] = None,
    progress: bool = True,
    sync_eval: bool = True,
    jobs: int = 1,
    virtual: Optional[VirtualSpec] = None,
    log_features: Optional[FeatureLogSpec] = None,
) -> IOHHarnessResult:
    """Run every strategy against every (fid, dim, instance, rep) in ``battery``.

    Runs serially by default; the underlying strategies use internal
    threading.  ``jobs > 1`` runs the cells in that many ``spawn`` worker
    processes (:func:`panobbgo.local_run.shared_pool`, niced, memory-floor
    throttled).  Every cell derives its own seed, so with ``sync_eval`` the
    records are the same for every ``jobs`` and come back in cell order;
    only ``elapsed_s`` differs.

    ``sync_eval=True`` (the default) runs every strategy in the
    synchronous-harvest evaluation mode (``config.sync_evaluation``):
    deterministic result batches, so a seeded run is reproducible —
    that, not speed, is why measurements default to it.  ``False`` opts
    into the threaded evaluator, whose trajectory depends on thread
    scheduling.  Only compare results measured under the same mode.

    ``timeout_s`` is a per-run wall-clock deadline, enforced by the
    tracker: evaluations past it are not counted, and the run keeps its
    AOCC up to the deadline but is recorded with a ``TimeoutError``.

    ``virtual`` runs every strategy on the virtual clock
    (:class:`~panobbgo.virtual_clock.VirtualSpec`: ``q`` simulated workers
    and a duration model) and scores ``aocc_time`` next to ``aocc``.  The
    run seeds do not depend on it, so a sweep over ``q`` is paired.

    ``log_features`` records checkpoint features in every run record
    (:mod:`panobbgo.features`) without changing the runs.  Refused on the
    sealed battery: its features would be training data.
    """
    if battery.problem_kind not in SUPPORTED_PROBLEM_KINDS:
        raise ValueError(f"Unknown problem kind {battery.problem_kind!r}; known: {list(SUPPORTED_PROBLEM_KINDS)}")
    if battery.sealed:
        refuse_sealed_features(log_features, battery.name)
        print_sealed_banner(battery.name)
    log_lo, log_hi = battery.aocc_bounds(log_lo, log_hi)
    builder_kwargs = battery.builder_kwargs()
    total = battery.pair_count(len(strategies))
    runs: List[IOHRunRecord] = []
    tasks: List[Dict[str, Any]] = []
    # ``fid_axis`` is ``(None,)`` on a battery with no function axis, so
    # this is the same single pass the loop has always made.
    cells = itertools.product(battery.fid_axis, battery.dims, battery.instances, strategies, range(battery.reps))
    for idx, (fid, dim, instance, spec, rep) in enumerate(cells, start=1):
        budget = battery.budget_for(dim)
        # One noise realisation per (fid, dim, instance, rep) cell,
        # identical for every strategy — see _derive_noise_seed.
        noise_seed = (
            _derive_noise_seed(base_seed, battery.problem_kind, dim, instance, rep, fid) if battery.is_noisy else None
        )
        # ``rng_identity`` is ``spec.seed_name or spec.name``: variants
        # of one arm can opt into a shared RNG stream so an A/B
        # measures the parameter, not the run-to-run variance.
        seed = _derive_seed(base_seed, battery.problem_kind, dim, instance, spec.rng_identity, rep, noise_seed, fid)
        # Common random numbers: the duration stream is keyed on the cell, not the strategy.
        cell_virtual = (
            None
            if virtual is None
            else virtual.with_cell(
                _derive_seed(
                    base_seed, battery.problem_kind, dim, instance, DURATION_STREAM_IDENTITY, rep, noise_seed, fid
                )
            )
        )
        fid_tag = f"f{fid:<2d} " if fid is not None else ""
        label = f"{battery.problem_kind} {fid_tag}dim={dim:<2d} inst={instance:<2d} rep={rep} {spec.name}"
        if progress and jobs <= 1:
            print(f"  [{idx:>4d}/{total:>4d}] {label}", flush=True)
        task: Dict[str, Any] = dict(
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
            fid=fid,
            virtual=cell_virtual,
            sealed=battery.sealed,
            log_features=log_features,
        )
        if jobs > 1:
            tasks.append(task)
            continue
        rec = _run_one(**task)
        if progress:
            _print_run(rec, budget)
        runs.append(rec)

    if tasks:
        runs = _run_tasks_in_pool(tasks, jobs, total, progress)

    result = IOHHarnessResult(
        battery_name=battery.name,
        problem_kind=battery.problem_kind,
        log_lo=log_lo,
        log_hi=log_hi,
        sync_eval=sync_eval or virtual is not None,
        runs=runs,
        virtual=None if virtual is None else virtual.to_dict(),
        sealed=battery.sealed,
        blas_threads=BLAS_THREADS,
        **fp_env.current(),
    )
    warn_missing_time_scores(result)
    return result
