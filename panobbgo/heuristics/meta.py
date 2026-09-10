# -*- coding: utf8 -*-
# Copyright 2026 Harald Schilly <harald.schilly@gmail.com>
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
Meta level
==========

One heuristic that stays **silent** for most of a run, wakes up when a
composable *trigger* says the moment has come, spends one bounded look at
the shared store, and turns that look into concrete next steps —
``planning/DESIGN_meta_level_2026-09-10.md``.

Why a :class:`~panobbgo.core.Heuristic` and not an
:class:`~panobbgo.core.Analyzer`: only a heuristic can emit points
(``core.py``), and every extra module perturbs the per-module RNG streams
(``spawn_rng`` is called once per module, in construction order).  **Append
this module last** in a spec's heuristics list and give it a name that
sorts after every arm's — ``"Meta"`` sorts after ``"CMAES"``, ``"JSO"``,
``"LSHADE"`` — and both the construction order and the event-bus
registration order of the arms are preserved bit for bit.  With
``trigger="never"`` the module is then provably inert: :meth:`on_new_results`
returns before it touches anything.

Silence is free under :class:`~panobbgo.strategies.blocks.StrategyBlockBandit`
— ``_select`` only considers arms that ``has_points`` — and costs a little
latency under :class:`~panobbgo.strategies.round_robin.StrategyRoundRobin`,
which polls every registered heuristic.  The meta level is a block-scheduler
feature (design §2).

What it does when it fires
--------------------------

**The scan** (design §1a + §1b), over the
:class:`~panobbgo.analyzers.splitter.Splitter` leaves, both terms
rank-normalised into ``[0, 1]`` so they are scale-free and commensurable:

.. math::

    \mathrm{share}_i   &= \log V_i - \log V_{\mathrm{root}} \\
    \mathrm{deficit}_i &= \log(n_i / N) - \mathrm{share}_i

``deficit < 0`` means the leaf holds a smaller share of the evaluations than
of the volume, i.e. it is under-sampled.  ``sparsity`` is the rank of
``-deficit``, ``quality`` the rank of the leaf's best penalty value, and

.. math::

    \mathrm{score}_i = (1 - w)\,\mathrm{sparsity}_i + w\,\mathrm{quality}_i

with ``w = quality_weight``.  ``w = 0`` is the pure volume-vs-count scan of
§1a ("void"), ``w > 0`` mixes in §1b's "promising but shallow" (the count
term is already inside ``sparsity``).  Cost is ``O(#leaves)`` — a few
hundred floats plus one ``get_penalty_value`` per leaf, i.e. microseconds,
once per firing.

**The output** is one of

``mode="scan"``
    ``k`` uniform draws spread over the top ``n_leaves`` leaves,
    ``who = "Meta:<reason>"``.
``mode="random"``
    ``k`` uniform draws over the whole problem box, at the identical firing
    times.  The design's falsifier: if this ties ``mode="scan"``, the
    analysis carries nothing and only the jolt of exploration mattered.
``mode="none"``
    no points at all — for the region hand-off, which costs zero
    evaluations.

**The region hand-off** (``region=True``, design §3 — the one mechanism
with no incumbent).  Instead of emitting points, the meta level publishes
``meta_region(arm=…, box=…)``.  The block scheduler records it and, at the
next ``_open_block`` for that arm, sets the arm's ``warm_start_box`` and
forces a warm start, so an **already-adapted** optimizer (CMA-ES's
covariance, jSO's success histories) is moved into the region *without
spending the evaluations to get it there*.  The hand-off is never applied
from the bus thread; see :meth:`panobbgo.strategies.blocks.StrategyBlockBandit.on_meta_region`.

Triggers
--------

Composable, **stateless** predicates (all firing state lives in the
heuristic, so a trigger object can be shared across runs of a benchmark)::

    trigger = budget_fraction(0.25)
    trigger = budget_fraction(0.25) | stagnation(0.10)
    trigger = budget_fraction(0.25) & stagnation(0.10)
    trigger = "never"                                    # the null control

``stagnation``'s window is a **fraction of the budget**, never an absolute
count (``planning/DISCOVERY_2026-09-09.md`` §20: an absolute 95-generation
window is 30 % of the *d* = 5 budget and fires after the run has reached
AOCC's log floor), resolved as ``max(4*dim, window_frac*max_eval)``.

Three guards are mandatory and always on (design §5):

* a **refractory period**, ``max(block_evals, refractory_frac*max_eval)``,
  so a ``|``-composed stagnation term cannot fire every block;
* a hard **budget cap**, ``meta_frac`` of ``max_eval``, on the evaluations
  the meta level may ever emit;
* **preconditions**: at least ``min_leafs`` leaves to scan (below that,
  volume-per-point is noise), and never in the last ``tail_frac`` of the
  budget (by then there is nothing left to gain).

.. codeauthor:: Harald Schilly <harald.schilly@gmail.com>
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from panobbgo.core import Heuristic
from panobbgo.lib import Point

__all__ = [
    "MetaAnalyst",
    "Trigger",
    "Never",
    "BudgetFraction",
    "Stagnation",
    "AnyOf",
    "AllOf",
    "never",
    "budget_fraction",
    "stagnation",
    "as_trigger",
]


# ----------------------------------------------------------------------
# triggers
# ----------------------------------------------------------------------


class MetaContext:
    """The read-only state a :class:`Trigger` is evaluated against.

    Deliberately tiny and free of the strategy object: a trigger is a pure
    predicate over "how far into the budget are we" and "what has the run's
    best done lately", which is all that makes a *when to decide* rule
    testable in isolation.
    """

    __slots__ = ("n_evals", "max_eval", "dim", "trace")

    def __init__(self, n_evals: int, max_eval: int, dim: int, trace: Sequence[float]) -> None:
        #: evaluations seen so far
        self.n_evals: int = int(n_evals)
        #: the run's budget
        self.max_eval: int = int(max_eval)
        #: the problem's dimension
        self.dim: int = int(dim)
        #: running best penalty value after each result, oldest first
        self.trace: Sequence[float] = trace

    @property
    def fraction(self) -> float:
        """Fraction of the budget spent, in ``[0, 1]``."""
        return self.n_evals / self.max_eval if self.max_eval > 0 else 0.0


class Trigger:
    """A composable, stateless "is it time?" predicate.

    :meth:`check` returns the **leaf trigger that fired** (so the caller
    learns *why*, for the ``who`` tag and the one-shot bookkeeping) or
    ``None``.  ``|`` and ``&`` build :class:`AnyOf` / :class:`AllOf`; a bare
    ``"never"`` string is accepted anywhere a trigger is (see
    :func:`as_trigger`).
    """

    #: fires at most once per run (the heuristic remembers :attr:`reason`)
    one_shot: bool = False

    @property
    def reason(self) -> str:
        """Short, greppable tag; rides in ``who`` as ``"Meta:<reason>"``."""
        return "meta"

    def check(self, ctx: MetaContext) -> Optional["Trigger"]:
        raise NotImplementedError

    def __or__(self, other: Any) -> "Trigger":
        return AnyOf(self, as_trigger(other))

    def __and__(self, other: Any) -> "Trigger":
        return AllOf(self, as_trigger(other))

    def __repr__(self) -> str:
        return "%s(%s)" % (type(self).__name__, self.reason)


class Never(Trigger):
    """The null control: never fires, and the heuristic short-circuits on it.

    ``MetaAnalyst(trigger="never")`` must produce a trajectory bit-identical
    to the same spec *without* the module — the check that every later
    number in the experiment rests on (design §6 step 0).
    """

    @property
    def reason(self) -> str:
        return "never"

    def check(self, ctx: MetaContext) -> Optional[Trigger]:
        return None


class BudgetFraction(Trigger):
    """Fires **once**, on the first batch that crosses ``f * max_eval``."""

    one_shot = True

    def __init__(self, f: float) -> None:
        f = float(f)
        if not 0.0 <= f <= 1.0:
            raise ValueError("budget_fraction: f must be in [0, 1], got %r" % f)
        self.f: float = f

    @property
    def reason(self) -> str:
        return "b%02d" % int(round(self.f * 100))

    def check(self, ctx: MetaContext) -> Optional[Trigger]:
        return self if ctx.n_evals >= self.f * ctx.max_eval else None


class Stagnation(Trigger):
    r"""Fires while the run's best has been flat over a budget-relative window.

    The window is ``max(4*dim, window_frac*max_eval)`` evaluations — never
    an absolute generation count (§20).  "Flat" is measured *relatively*,
    ``old - new <= tol * max(|old|, 1)``, so the predicate does not depend
    on the units of the objective; the design's ``tol = 1e-8`` reads as
    "eight significant digits of no progress".
    """

    def __init__(self, window_frac: float = 0.10, tol: float = 1e-8) -> None:
        window_frac = float(window_frac)
        if not 0.0 < window_frac <= 1.0:
            raise ValueError("stagnation: window_frac must be in (0, 1], got %r" % window_frac)
        self.window_frac: float = window_frac
        self.tol: float = float(tol)

    @property
    def reason(self) -> str:
        return "stag"

    def window(self, ctx: MetaContext) -> int:
        return max(4 * ctx.dim, int(round(self.window_frac * ctx.max_eval)), 1)

    def check(self, ctx: MetaContext) -> Optional[Trigger]:
        w = self.window(ctx)
        trace = ctx.trace
        if len(trace) <= w:
            return None
        old, new = float(trace[-w - 1]), float(trace[-1])
        if not (np.isfinite(old) and np.isfinite(new)):
            return None
        return self if (old - new) <= self.tol * max(abs(old), 1.0) else None


class AnyOf(Trigger):
    """``a | b`` — the first child that fires wins, and lends its reason."""

    def __init__(self, *children: Trigger) -> None:
        self.children: Tuple[Trigger, ...] = tuple(children)

    @property
    def reason(self) -> str:
        return "|".join(c.reason for c in self.children)

    def check(self, ctx: MetaContext) -> Optional[Trigger]:
        for c in self.children:
            hit = c.check(ctx)
            if hit is not None:
                return hit
        return None


class AllOf(Trigger):
    """``a & b`` — the conservative trigger: every child must fire."""

    def __init__(self, *children: Trigger) -> None:
        self.children: Tuple[Trigger, ...] = tuple(children)

    @property
    def one_shot(self) -> bool:  # type: ignore[override]
        """One-shot iff *every* child is: an ``&`` with a recurring term
        stays recurring, and the refractory period is what bounds it."""
        return all(c.one_shot for c in self.children)

    @property
    def reason(self) -> str:
        return "&".join(c.reason for c in self.children)

    def check(self, ctx: MetaContext) -> Optional[Trigger]:
        return self if all(c.check(ctx) is not None for c in self.children) else None


def never() -> Trigger:
    """The null trigger (see :class:`Never`)."""
    return Never()


def budget_fraction(f: float) -> Trigger:
    """Fire once, at ``f`` of the budget (see :class:`BudgetFraction`)."""
    return BudgetFraction(f)


def stagnation(window_frac: float = 0.10, tol: float = 1e-8) -> Trigger:
    """Fire on a flat run best over a budget-relative window (:class:`Stagnation`)."""
    return Stagnation(window_frac, tol)


#: What a ``trigger=`` argument may be, besides a :class:`Trigger`.
_ALIASES = {"never": Never, "none": Never}


def as_trigger(spec: Any) -> Trigger:
    """Coerce ``spec`` to a :class:`Trigger`.

    Accepts a :class:`Trigger`, ``"never"``/``"none"``/``None``, or a plain
    float, which reads as :func:`budget_fraction`.
    """
    if isinstance(spec, Trigger):
        return spec
    if spec is None:
        return Never()
    if isinstance(spec, str):
        key = spec.strip().lower()
        if key in _ALIASES:
            return _ALIASES[key]()
        raise ValueError("unknown trigger %r (known: %s)" % (spec, ", ".join(sorted(_ALIASES))))
    if isinstance(spec, (int, float)) and not isinstance(spec, bool):
        return BudgetFraction(float(spec))
    raise TypeError("trigger must be a Trigger, 'never' or a budget fraction, got %r" % (spec,))


# ----------------------------------------------------------------------
# the heuristic
# ----------------------------------------------------------------------


class MetaAnalyst(Heuristic):
    """
    Analyse the shared store *once*, at a chosen moment, and turn the
    analysis into points or into a region hand-off.

    :param strategy: the owning strategy.
    :param trigger: a :class:`Trigger`, or ``"never"`` (default — the null
        control), or a float read as :func:`budget_fraction`.
    :param mode: ``"scan"`` (points from the leaf scan), ``"random"`` (the
        falsifier: uniform points over the whole box at the same moments) or
        ``"none"`` (emit nothing; for the pure region hand-off).
    :param region: publish ``meta_region`` so the block scheduler moves an
        adapted arm into the chosen leaf at its next block open.
    :param region_arm: name of the arm to hand the region to; ``None``
        picks the arm that is furthest behind on its own best (the leader is
        left alone).
    :param k: points per firing; ``None`` resolves ``k_frac * max_eval``,
        floored at ``dim``.
    :param k_frac: the fraction behind ``k=None`` (design §4: ``0.02``).
    :param meta_frac: hard cap on the share of the budget this module may
        ever emit (design §5).
    :param refractory: evaluations between two firings; ``None`` resolves
        ``max(block_evals, refractory_frac * max_eval)``.
    :param n_leaves: how many top-scoring leaves the ``k`` points are spread
        over — concentrated enough to establish a basin, spread enough not
        to feed the ``Splitter`` live-lock of §13.
    :param quality_weight: ``w`` in the score; ``0`` is the pure
        volume-vs-count scan, ``1`` pure leaf quality.
    :param min_leafs: refuse to fire with fewer leaves than this — on a
        handful of boxes volume-per-point is noise (design §5).
    :param min_region_points: refuse a region hand-off unless the archive
        holds at least this many points inside the box, so the receiving arm
        is not handed an empty query after its queue was cleared.
    :param tail_frac: refuse to fire in the last fraction of the budget.
    """

    #: default name; sorts after ``CMAES`` / ``JSO`` / ``LSHADE``, which is
    #: what keeps the arms' registration order untouched (module docstring).
    DEFAULT_NAME = "Meta"

    def __init__(
        self,
        strategy,
        *,
        trigger: Any = "never",
        mode: str = "scan",
        region: bool = False,
        region_arm: Optional[str] = None,
        k: Optional[int] = None,
        k_frac: float = 0.02,
        meta_frac: float = 0.05,
        refractory: Optional[int] = None,
        refractory_frac: float = 0.05,
        n_leaves: int = 3,
        quality_weight: float = 0.5,
        min_leafs: int = 8,
        min_region_points: int = 4,
        tail_frac: float = 0.25,
        name: Optional[str] = None,
    ) -> None:
        if mode not in ("scan", "random", "none"):
            raise ValueError("mode must be 'scan', 'random' or 'none', got %r" % mode)
        Heuristic.__init__(self, strategy, name=name or self.DEFAULT_NAME)
        self.logger = self.config.get_logger("META")

        self.trigger: Trigger = as_trigger(trigger)
        self.mode: str = str(mode)
        self.region: bool = bool(region)
        self.region_arm: Optional[str] = region_arm
        self.k: Optional[int] = None if k is None else int(k)
        self.k_frac: float = float(k_frac)
        self.meta_frac: float = float(meta_frac)
        self._refractory: Optional[int] = None if refractory is None else int(refractory)
        self.refractory_frac: float = float(refractory_frac)
        self.n_leaves: int = max(1, int(n_leaves))
        self.quality_weight: float = float(np.clip(float(quality_weight), 0.0, 1.0))
        self.min_leafs: int = int(min_leafs)
        self.min_region_points: int = int(min_region_points)
        self.tail_frac: float = float(tail_frac)

        #: ``True`` iff the module is provably inert — checked first in
        #: :meth:`on_new_results` so the null control touches nothing at all.
        self._never: bool = isinstance(self.trigger, Never)

        # -- run state (no randomness here: ``__init__`` must not draw) ----
        self._n_evals: int = 0
        self._best: float = float("inf")
        self._trace: List[float] = []
        self._emitted: int = 0
        self._last_fire: Optional[int] = None
        self._fired: set[str] = set()
        #: one record per firing, for the screen's health block
        self.firings: List[Dict[str, Any]] = []
        #: firings refused by a precondition, by cause
        self.refusals: Dict[str, int] = {}

    # -- budget bookkeeping ------------------------------------------------

    def _max_eval(self) -> int:
        try:
            return int(self.config.max_eval)
        except (TypeError, ValueError):
            return 1000

    def _resolve_k(self) -> int:
        max_eval = self._max_eval()
        k = self.k if self.k is not None else max(self.problem.dim, int(round(self.k_frac * max_eval)))
        budget = int(self.meta_frac * max_eval)
        return max(0, min(int(k), budget - self._emitted))

    def refractory(self) -> int:
        """Evaluations that must pass between two firings.

        ``max(block_evals, refractory_frac*max_eval)``: never more often
        than the scheduler makes a decision, and never more often than
        5 % of the budget.
        """
        if self._refractory is not None:
            return self._refractory
        block = 0
        try:
            value = getattr(self.strategy, "block_evals", 0)
            block = int(value) if isinstance(value, (int, np.integer)) else 0
        except Exception:
            block = 0
        return max(block, int(round(self.refractory_frac * self._max_eval())), 1)

    # -- the event handler -------------------------------------------------

    def on_new_results(self, results) -> None:
        """Advance the trace, ask the trigger, fire at most once per batch.

        The very first statement is the null short-circuit: with
        ``trigger="never"`` this module reads nothing, writes nothing and
        draws nothing, so the run is bit-identical to one without it.
        """
        if self._never:
            return
        self._n_evals += len(results)
        for r in results:
            phi = self._penalty(r)
            if phi is not None and phi < self._best:
                self._best = phi
            self._trace.append(self._best)

        hit = self._due()
        if hit is None:
            return
        self._fire(hit)

    def _penalty(self, r) -> Optional[float]:
        handler = getattr(self.strategy, "constraint_handler", None)
        try:
            if handler is None:
                value = float("inf") if r.fx is None else float(r.fx)
            else:
                value = float(handler.get_penalty_value(r))
        except (TypeError, ValueError):
            return None
        return value if np.isfinite(value) else None

    def context(self) -> MetaContext:
        """The :class:`MetaContext` the trigger is evaluated against."""
        return MetaContext(self._n_evals, self._max_eval(), self.problem.dim, self._trace)

    def _refuse(self, cause: str) -> None:
        self.refusals[cause] = self.refusals.get(cause, 0) + 1

    def _due(self) -> Optional[Trigger]:
        """The trigger that fires now, or ``None`` — guards first, rule last."""
        ctx = self.context()
        if self._last_fire is not None and ctx.n_evals - self._last_fire < self.refractory():
            return None  # refractory: silent, not a refusal worth logging
        hit = self.trigger.check(ctx)
        if hit is None:
            return None
        if hit.one_shot and hit.reason in self._fired:
            return None
        # -- preconditions (design §5).  Evaluated *after* the trigger so a
        # refusal is attributable to a moment the rule actually wanted.
        if ctx.max_eval - ctx.n_evals < self.tail_frac * ctx.max_eval:
            self._refuse("tail")
            return None
        if self._emits_points() and self._resolve_k() <= 0:
            self._refuse("meta_frac")
            return None
        if len(self._leafs()) < self.min_leafs:
            self._refuse("leafs")
            return None
        return hit

    def _emits_points(self) -> bool:
        return self.mode != "none"

    # -- the scan ----------------------------------------------------------

    def _splitter(self) -> Any:
        try:
            return self.strategy.analyzer("Splitter")
        except Exception:
            return None

    def _leafs(self) -> List[Any]:
        splitter = self._splitter()
        if splitter is None:
            return []
        return [leaf for leaf in getattr(splitter, "leafs", []) if leaf is not None]

    def _leaf_penalty(self, leaf) -> float:
        """Best penalty value inside ``leaf``, ``+inf`` when it holds nothing."""
        best = getattr(leaf, "best", None)
        if best is None:
            return float("inf")
        phi = self._penalty(best)
        return float("inf") if phi is None else phi

    @staticmethod
    def _rank01(values: np.ndarray) -> np.ndarray:
        """Rank-normalise into ``[0, 1]``: the *smallest* value scores 1.

        Rank rather than a scale, for the same reason ``RegionUCB`` and the
        block reward take ranks: the optimizer has no units for either the
        objective or the log-volume deficit.  ``kind="stable"`` so ties
        break deterministically.
        """
        n = len(values)
        if n <= 1:
            return np.ones(n)
        order = np.argsort(np.argsort(values, kind="stable"), kind="stable")
        return 1.0 - order / (n - 1.0)

    def scan(self) -> List[Tuple[float, Any]]:
        """Score every ``Splitter`` leaf, best first.

        ``O(#leaves)`` floats plus one ``get_penalty_value`` per leaf — the
        whole point of a meta level is that this runs *per firing*, not per
        result batch.
        """
        leafs = self._leafs()
        if not leafs:
            return []
        splitter = self._splitter()
        root = getattr(splitter, "root", None)
        root_log_volume = float(getattr(root, "log_volume", 0.0)) if root is not None else 0.0

        counts = np.array([max(len(leaf), 1) for leaf in leafs], dtype=float)
        total = float(counts.sum())
        share = np.array([float(leaf.log_volume) for leaf in leafs]) - root_log_volume
        with np.errstate(divide="ignore", invalid="ignore"):
            deficit = np.log(counts / total) - share
        deficit = np.where(np.isfinite(deficit), deficit, 0.0)

        sparsity = self._rank01(deficit)  # most under-sampled (smallest deficit) -> 1
        quality = self._rank01(np.array([self._leaf_penalty(leaf) for leaf in leafs]))

        w = self.quality_weight
        scores = (1.0 - w) * sparsity + w * quality
        order = np.argsort(-scores, kind="stable")
        return [(float(scores[i]), leafs[i]) for i in order]

    # -- emission ----------------------------------------------------------

    @staticmethod
    def _box_array(leaf) -> np.ndarray:
        box = leaf.box
        return box.box if hasattr(box, "box") else box

    def _draw_in(self, leaf) -> np.ndarray:
        box = self._box_array(leaf)
        return box[:, 0] + np.asarray(leaf.ranges, dtype=float) * self.rng.random(self.problem.dim)

    def _scan_points(self, leafs: Sequence[Any], k: int) -> List[np.ndarray]:
        """``k`` uniform draws, spread round-robin over ``leafs``."""
        if not leafs:
            return []
        return [self._draw_in(leafs[i % len(leafs)]) for i in range(k)]

    def _random_points(self, k: int) -> List[np.ndarray]:
        return [self.problem.random_point(rng=self.rng) for _ in range(k)]

    def _put_points(self, points: Sequence[np.ndarray], who: str) -> int:
        """Queue ``points`` under an explicit ``who``.

        :meth:`~panobbgo.core.Heuristic.emit` tags with the bare module
        name; the reason has to ride along, so the ledger stays greppable
        and ``Archive._who_of`` / ``blocks._credit_arm_best`` still attribute
        the points to this arm (both split on the first ``":"``).
        """
        for x in points:
            self._put(Point(self.problem.project(np.asarray(x, dtype=float)), who))
        return len(points)

    # -- the region hand-off -----------------------------------------------

    def _region_support(self, leaf) -> int:
        """How many *known good* points already sit inside ``leaf``."""
        archive = self._archive_analyzer()
        if archive is not None:
            try:
                return len(archive.top_k(self.min_region_points, box=leaf))
            except Exception:
                return 0
        return len(getattr(leaf, "results", []))

    def _accepts_region(self, h) -> bool:
        """Arms that can act on a box: the ``warm_start`` *mode-string* form.

        The callable ``warm_start(results)`` hook takes results, not a
        region, so an arm that only has that one cannot be handed a box.
        """
        return bool(getattr(h, "warm_start", None)) and not callable(getattr(h, "warm_start", None))

    def _region_recipient(self) -> Optional[Any]:
        """The arm to hand the region to — furthest behind, ties by name.

        Explicit ``region_arm`` wins.  Otherwise the arm with the *worst*
        own best is chosen: the leader is the one whose adaptation is
        currently paying off, and moving it is the expensive mistake.
        """
        arms = [h for h in self.strategy.heuristics if h.name != self.name and self._accepts_region(h)]
        if not arms:
            return None
        if self.region_arm is not None:
            return next((h for h in arms if h.name == self.region_arm), None)
        arm_best: Dict[str, float] = dict(getattr(self.strategy, "_arm_best", {}) or {})
        return max(arms, key=lambda h: (arm_best.get(h.name, float("inf")), h.name))

    def _hand_off(self, ordered: Sequence[Tuple[float, Any]]) -> Optional[Dict[str, Any]]:
        """Publish ``meta_region`` for the best *supported* leaf.

        Only records the request — the scheduler applies it on the main
        thread at the next ``_open_block``.  Never calls ``warm_start_now``:
        that clears a queue ``execute()`` may be draining (design §2).
        """
        arm = self._region_recipient()
        if arm is None:
            self._refuse("no_region_arm")
            return None
        for _, leaf in ordered:
            if self._region_support(leaf) >= self.min_region_points:
                self.eventbus.publish("meta_region", arm=arm.name, box=leaf)
                return {"arm": arm.name, "leaf": getattr(leaf, "id", None)}
        self._refuse("region_unsupported")
        return None

    # -- firing ------------------------------------------------------------

    def _fire(self, hit: Trigger) -> None:
        who = "%s:%s" % (self.name, hit.reason)
        ordered = self.scan()
        n_points = 0
        if self._emits_points():
            k = self._resolve_k()
            if self.mode == "random":
                points = self._random_points(k)
            else:
                points = self._scan_points([leaf for _, leaf in ordered[: self.n_leaves]], k)
            n_points = self._put_points(points, who)
            self._emitted += n_points

        handed = self._hand_off(ordered) if self.region else None

        self._last_fire = self._n_evals
        self._fired.add(hit.reason)
        record: Dict[str, Any] = {
            "eval": self._n_evals,
            "reason": hit.reason,
            "points": n_points,
            "leafs": len(ordered),
            "region": handed,
        }
        self.firings.append(record)
        self.logger.info(
            "meta fired at %d/%d (%s): %d points over %d leaves, region=%s",
            self._n_evals,
            self._max_eval(),
            hit.reason,
            n_points,
            len(ordered),
            handed,
        )
