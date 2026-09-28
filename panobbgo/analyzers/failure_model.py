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
Failure model
=============

A shared estimate of where evaluations fail ("poison zones",
``planning/DESIGN_roadmap_2026-09-26.md`` §4 D): regions of the box where
the objective crashes, returns ``NaN`` or times out.  Every heuristic can
ask it, and the strategy can use it to answer a candidate in such a region
without evaluating it.

The model (v0)
--------------

A **kernel classifier with an adaptive (balloon) bandwidth and a success
prior**, on box-normalised coordinates :math:`u \in [0, 1]^d`, from every
labelled point the run has seen (a result with a finite value is a success;
a crash, a timed-out ``NaN`` placeholder or any non-finite value is a
failure):

.. math::

    p_{\mathrm{fail}}(u) = \frac{\sum_i w_i y_i}{\sum_i w_i + \alpha},
    \qquad w_i = \exp\!\left(-\tfrac12 \lVert u - u_i \rVert^2 / h(u)^2\right),
    \qquad h(u) = \min\bigl(h_n,\ r_k(u)\bigr)

with :math:`y_i = 1` for a failure, :math:`r_k(u)` the distance to the
:math:`k`-th nearest labelled point (:math:`k = d + 1`, at least 3) and the
cap :math:`h_n = 2\,(k / (n V_d))^{1/d}`, twice the expected distance to
the :math:`k`-th neighbour among :math:`n` uniform points (:math:`V_d` the
volume of the unit ball).  The choices, each for one of the requirements:

* **Unexplored is not poisoned — beyond the data's own scale.**
  :math:`\alpha` (``prior``, default 1) is a pseudo-count of success, so a
  zone needs several failures close together: :math:`p \ge 1/2` needs a
  failure weight of at least :math:`1 + \alpha` plus the success weight;
  with all :math:`k` neighbours failed, :math:`p \approx 0.6` (d = 2) to
  0.8 (d = 5); one isolated failure gives :math:`p < 1 / (1 + \alpha)`
  (below the threshold) anywhere but on the point itself.  Further than a
  few :math:`h_n` from every failure all failure weights vanish and
  :math:`p \to 0`.  Within that reach the estimate is a smoothed
  :math:`k`-nearest-neighbour vote, and it does extrapolate: a point whose
  nearest labelled points are all failures is marked even where nothing has
  been evaluated (deeper inside a half-space than any failure, or, early
  in a run, around a first cluster of failures).  That is the price of a
  model that also recognises a zone it has sampled sparsely; the
  "whole box" guard below bounds it.
* **Sharp where the data are dense.**  The bandwidth is the distance to
  the :math:`k`-th neighbour, so near a converging search (thousands of
  points within :math:`10^{-3}` of an optimum on the boundary of a
  failure region) the boundary is resolved at the scale of the search, not
  at the scale of the box.
* **Deterministic failures.**  A point within ``repeat_tol`` (Euclidean, in
  normalised units) of a known failure has :math:`p = 1`: evaluating it
  again would fail again.
* **Cheap.**  A query of :math:`m` points costs the distances to the
  failures, :math:`O(m\, n_{\mathrm{fail}} d)` (the repeat rule reuses
  them); only the queries within reach of a failure also compute the
  distances to the successes, :math:`O(m' n_{\mathrm{ok}} d)`.  Points are
  stored in growing buffers (amortised :math:`O(d)` per result).  The
  guard below costs ``N_PROBES`` queries, amortised and exact near its
  threshold (:meth:`FailureModel.poisoned_share`).  No spatial index: at the budgets
  measured a one-point query takes 0.03 ms (d = 2, n = 200) to 0.06 ms
  (d = 10, n = 1000) and 0.3 ms at n = 10 000, one guard evaluation 1 ms to
  110 ms (laptop, 2026-09-27); a KD-tree would pay off beyond that.  With no failure at all
  :meth:`p_fail` returns zeros without touching anything else, so a run
  without failures is bit-identical with the model on.
* **Never the whole box.**  :meth:`in_poison` is ``p >= threshold``
  (default 0.5), and it is *disarmed* (``False`` everywhere) while the
  share of a fixed set of probe points in the box that it would mark
  exceeds ``max_share`` (default 0.5, the largest failure share the
  :class:`~panobbgo.lib.families.FailureRegion` families use).

Alternatives considered and not chosen for v0 (DISCOVERY §71): a half-space
fit (logistic / linear SVM on the failures) extrapolates a zone into
unexplored territory — exactly right for a half-space, wrong for a ball or
boxes, and against "unexplored is not poisoned"; a GP classifier is too
expensive to query on every proposal; an axis-aligned tree is a natural v1
for half-spaces along one variable.

The proposal filter
-------------------

With ``filter=True`` the strategy's main loop
(:meth:`~panobbgo.core.StrategyBase._filter_poisoned`) passes every
candidate through :meth:`reject`: a candidate in a poison zone is **not
evaluated and costs no budget**; the heuristic that proposed it is told, by
the ``predicted_failures`` event, that it failed — which each heuristic
answers with its own failure handling (``on_failed_evaluations``: CMA-ES
ranks it last, the DE family loses the trial, the trust region shrinks, a
solver bridge answers ``inf``).  The candidate is *answered*, not
resampled: the heuristic's sampling distribution is not changed, and a
CMA-ES generation sees the rejected offspring exactly as it would see a real
failure, only for free.  Liveness: after ``max_rejects`` consecutive
rejections of one heuristic's candidates, its next candidate is evaluated
whatever the model says (which also lets the model correct itself); a
heuristic without an ``on_failed_evaluations`` hook (it refills only on new
results, like ``Random``) gets ``on_new_results([])`` for its rejected
candidates, so rejections never drain its queue.

The model only learns from *evaluated* points; a rejected candidate is not
a failure the model has seen.

.. codeauthor:: Harald Schilly <harald.schilly@gmail.com>
"""

from __future__ import annotations

import math
import threading
from typing import Any, Dict, List, Optional

import numpy as np

from panobbgo.core import Analyzer

#: Probe points of the "never the whole box" guard.
N_PROBES = 256

#: :meth:`FailureModel.poisoned_share` is recomputed on every new failure up to this many, then every +10 %.
SHARE_EXACT_UP_TO = 50

#: Above ``SHARE_NEAR * max_share`` the guard's share is recomputed exactly on every data change.
SHARE_NEAR = 0.8

#: The bandwidth cap is ``H_FACTOR`` times the expected ``k``-th neighbour distance of ``n`` uniform points.
H_FACTOR = 2.0


class FailureModel(Analyzer):
    """Shared failure-region model: ``p_fail(x)`` / ``in_poison(x)`` from observed failures and successes.

    Opt-in: add it to a strategy (``StrategySpec.analyzers``); nothing in a
    strategy without it changes.  Thread-safety: results arrive on the
    event-bus thread, queries come from the main loop and from heuristics;
    one lock covers the labelled data.

    Args:
        strategy: The owning strategy.
        filter: Reject candidates in poison zones in the strategy's main
            loop (see the module docstring).  ``False`` (default): the model
            only answers queries.
        threshold: :meth:`in_poison` is ``p_fail >= threshold``.
        prior: The success pseudo-count :math:`\\alpha`.
        k: Neighbour rank of the adaptive bandwidth; ``None``: ``max(3, d + 1)``.
        repeat_tol: Euclidean distance (normalised units) within which a
            point repeats a known failure.
        max_share: Largest share of the box :meth:`in_poison` may mark before
            it is disarmed.
        max_rejects: Consecutive rejections of one heuristic's candidates
            after which its next candidate passes.
        name: Module name; defaults to ``"FailureModel"``.
    """

    def __init__(
        self,
        strategy: Any,
        filter: bool = False,
        threshold: float = 0.5,
        prior: float = 1.0,
        k: Optional[int] = None,
        repeat_tol: float = 1e-9,
        max_share: float = 0.5,
        max_rejects: int = 10,
        name: Optional[str] = None,
    ) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("FailureModel: threshold must be in (0, 1]")
        if prior <= 0.0:
            raise ValueError("FailureModel: prior must be > 0")
        if not 0.0 < max_share <= 1.0:
            raise ValueError("FailureModel: max_share must be in (0, 1]")
        if int(max_rejects) < 1:
            raise ValueError("FailureModel: max_rejects must be >= 1")
        Analyzer.__init__(self, strategy, name=name or "FailureModel")
        self.filter = bool(filter)
        self.threshold = float(threshold)
        self.prior = float(prior)
        dim = int(self.problem.dim)
        self.k = int(k) if k is not None else max(3, dim + 1)
        if self.k < 1:
            raise ValueError("FailureModel: k must be >= 1")
        self.repeat_tol = float(repeat_tol)
        self.max_share = float(max_share)
        self.max_rejects = int(max_rejects)
        box = np.array(self.problem.box[:, :], dtype=float)
        self._lo = box[:, 0]
        self._width = np.where(box[:, 1] > box[:, 0], box[:, 1] - box[:, 0], 1.0)
        self._lock = threading.RLock()
        self._fail = _Rows(dim)
        self._ok = _Rows(dim)
        #: Failure kinds seen: ``"crash"``, ``"timeout"``, ``"nan"`` -> count.
        self.kinds: Dict[str, int] = {}
        # A fixed probe set of its own stream: querying it never touches another module's randomness.
        self._probes = np.random.default_rng(0x9015011).random((N_PROBES, dim))
        self._share: Optional[float] = None  # poisoned probe share (see :meth:`poisoned_share`)
        self._share_at = 0  # n_fail when it was computed
        self._share_at_total = 0  # n_fail + n_ok when it was computed
        #: Consecutive rejections per heuristic name (see :meth:`reject`).
        self._streak: Dict[str, int] = {}
        #: Diagnostics.
        self.n_rejected = 0
        self.n_passed_streak = 0

    # -- data --------------------------------------------------------------

    @property
    def n_fail(self) -> int:
        return len(self._fail)

    @property
    def n_ok(self) -> int:
        return len(self._ok)

    @property
    def _U_fail(self) -> np.ndarray:
        return self._fail.view()

    @property
    def _U_ok(self) -> np.ndarray:
        return self._ok.view()

    def _to_u(self, x: Any) -> np.ndarray:
        return (np.atleast_2d(np.asarray(x, dtype=float)) - self._lo) / self._width

    def add_failure(self, x: Any, kind: str = "crash") -> None:
        """Record a failed evaluation at ``x`` (also the entry point for tests and other drivers)."""
        with self._lock:
            self._fail.extend(self._to_u(x))
            self.kinds[kind] = self.kinds.get(kind, 0) + 1

    def add_success(self, x: Any) -> None:
        """Record a successful evaluation at ``x``."""
        with self._lock:
            self._ok.extend(self._to_u(x))

    def on_new_results(self, results: List[Any]) -> None:
        """Label every result: finite value -> success; timed out / non-finite -> failure."""
        fail, ok, kinds = [], [], []
        for r in results:
            x = getattr(r, "x", None)
            if x is None:
                continue
            fx = getattr(r, "fx", None)
            if getattr(r, "timed_out", False) or fx is None or not np.isfinite(float(fx)):
                fail.append(np.asarray(x, dtype=float))
                kinds.append("timeout" if getattr(r, "timed_out", False) else "nan")
            else:
                ok.append(np.asarray(x, dtype=float))
        with self._lock:
            if ok:
                self._ok.extend(self._to_u(np.asarray(ok)))
            if fail:
                self._fail.extend(self._to_u(np.asarray(fail)))
                for kind in kinds:
                    self.kinds[kind] = self.kinds.get(kind, 0) + 1

    def on_failed_evaluations(self, points: List[Any]) -> None:
        """A crashed evaluation (no result) is a failure at its point."""
        xs = [np.asarray(p.x, dtype=float) for p in points if getattr(p, "x", None) is not None]
        if not xs:
            return
        with self._lock:
            self._fail.extend(self._to_u(np.asarray(xs)))
            self.kinds["crash"] = self.kinds.get("crash", 0) + len(xs)

    # -- queries -----------------------------------------------------------

    def _p_u(self, U: np.ndarray) -> np.ndarray:
        """``p_fail`` of normalised points ``U`` (``m x d``); callers hold the lock."""
        m, d = U.shape
        F = self._U_fail
        out = np.zeros(m)
        if F.shape[0] == 0 or m == 0:
            return out
        S = self._U_ok
        n = F.shape[0] + S.shape[0]
        h_n = H_FACTOR * (self.k / (n * _unit_ball_volume(d))) ** (1.0 / d)
        dF = np.sqrt(((U[:, None, :] - F[None, :, :]) ** 2).sum(axis=2))  # (m, n_fail)
        # Only a query within a few global bandwidths of a failure can reach the threshold.
        near = dF.min(axis=1) <= 4.0 * h_n
        if near.any():
            idx = np.flatnonzero(near)
            dFn = dF[idx]
            if S.shape[0]:
                dS = np.sqrt(((U[idx, None, :] - S[None, :, :]) ** 2).sum(axis=2))
                D = np.hstack([dFn, dS])
            else:
                D = dFn
            kk = min(self.k, D.shape[1])
            r_k = np.partition(D, kk - 1, axis=1)[:, kk - 1]
            h = np.maximum(np.minimum(h_n, r_k), 1e-12)[:, None]
            wF = np.exp(-0.5 * (dFn / h) ** 2).sum(axis=1)
            wS = np.exp(-0.5 * (D[:, dFn.shape[1] :] / h) ** 2).sum(axis=1)
            out[idx] = wF / (wF + wS + self.prior)
        # A repeat of a known failure fails again (Euclidean, from the distances above).
        out[dF.min(axis=1) <= self.repeat_tol] = 1.0
        return out

    def p_fail(self, x: Any) -> Any:
        """Estimated probability that evaluating ``x`` fails: a float for one point, an array for ``m x d``."""
        X = np.asarray(x, dtype=float)
        with self._lock:
            p = self._p_u(self._to_u(X))
        return float(p[0]) if X.ndim == 1 else p

    def poisoned_share(self) -> float:
        """Share of the fixed probe points the model marks (``p >= threshold``), before the guard.

        Computed lazily, on the first query after the data changed, and
        amortised: exactly on every change (a new failure *or* success) while
        the last share is above ``SHARE_NEAR * max_share`` (close to the
        guard) or the model knows at most :data:`SHARE_EXACT_UP_TO` failures;
        otherwise only once the failures or the labelled points have grown by
        10 %.  New successes can *raise* the share as well as lower it (they
        shrink the bandwidths ``h_n`` and ``r_k``), so a share cached far
        below the guard can lag the exact one; that lag is what the
        ``SHARE_NEAR`` band absorbs.  The guard is therefore not a hard
        guarantee between recomputes, only near the threshold: tested on a
        half-box failure region sampled uniformly and then by successes only
        (``tests/test_failure_model.py``).  One evaluation costs ``N_PROBES``
        queries.
        """
        with self._lock:
            n_f, n = self.n_fail, self.n_fail + self.n_ok
            changed = (n_f, n) != (self._share_at, self._share_at_total)
            stale = self._share is None or (
                changed
                and (
                    self._share > SHARE_NEAR * self.max_share
                    or (n_f != self._share_at and n_f <= SHARE_EXACT_UP_TO)
                    or n_f >= 1.1 * self._share_at
                    or n >= 1.1 * self._share_at_total
                )
            )
            if stale:
                self._share = float(np.mean(self._p_u(self._probes) >= self.threshold)) if n_f else 0.0
                self._share_at, self._share_at_total = n_f, n
            assert self._share is not None
            return self._share

    @property
    def armed(self) -> bool:
        """``False`` while the model would mark more than ``max_share`` of the box (or knows no failure)."""
        return self.n_fail > 0 and self.poisoned_share() <= self.max_share

    def in_poison(self, x: Any) -> Any:
        """``p_fail(x) >= threshold`` while :attr:`armed`, else ``False``; a bool or a bool array like :meth:`p_fail`."""
        X = np.asarray(x, dtype=float)
        if not self.armed:
            return False if X.ndim == 1 else np.zeros(X.shape[0], dtype=bool)
        p = self.p_fail(X)
        return bool(p >= self.threshold) if X.ndim == 1 else p >= self.threshold

    # -- the proposal filter ---------------------------------------------------

    def reject(self, points: List[Any]) -> List[bool]:
        """Which of ``points`` (candidates, in order) the filter rejects; updates the per-heuristic streaks.

        A candidate of a heuristic whose last ``max_rejects`` candidates were
        all rejected passes (liveness; see the module docstring).  Returns
        all ``False`` without any work while the model knows no failure.
        """
        if not points or not self.armed:
            for p in points:
                self._streak.pop(_owner(p), None)
            return [False] * len(points)
        poisoned = self.in_poison(np.asarray([p.x for p in points], dtype=float))
        out = []
        for p, bad in zip(points, poisoned):
            who = _owner(p)
            if bad and self._streak.get(who, 0) < self.max_rejects:
                self._streak[who] = self._streak.get(who, 0) + 1
                self.n_rejected += 1
                out.append(True)
            else:
                if bad:
                    self.n_passed_streak += 1
                self._streak.pop(who, None)
                out.append(False)
        return out


class _Rows:
    """A growing ``n x d`` float buffer (amortised O(1) appends, no copy per result)."""

    def __init__(self, dim: int) -> None:
        self._a = np.empty((16, dim))
        self._n = 0

    def __len__(self) -> int:
        return self._n

    def extend(self, rows: np.ndarray) -> None:
        rows = np.atleast_2d(rows)
        need = self._n + rows.shape[0]
        if need > self._a.shape[0]:
            grown = np.empty((max(need, 2 * self._a.shape[0]), self._a.shape[1]))
            grown[: self._n] = self._a[: self._n]
            self._a = grown
        self._a[self._n : need] = rows
        self._n = need

    def view(self) -> np.ndarray:
        return self._a[: self._n]


def _unit_ball_volume(d: int) -> float:
    """Volume of the unit ball in ``d`` dimensions."""
    return float(math.exp(0.5 * d * math.log(math.pi) - math.lgamma(0.5 * d + 1.0)))


def _owner(point: Any) -> str:
    """The heuristic name of a candidate: its ``who`` up to the first ``:``."""
    return str(getattr(point, "who", "") or "").split(":")[0]
