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

"""
Quadratic trust region on the shared archive
============================================

A small model-based local search in the spirit of Powell's BOBYQA / NEWUOA,
written for panobbgo's event model instead of wrapped around a sequential
solver (compare :class:`~panobbgo.heuristics.cobyqa.COBYQA`, which bridges
SciPy's COBYQA through a subprocess and can only ever have one point in
flight).

* **Model.**  A quadratic fitted by weighted least squares to the archive
  points nearest to the centre, within ``fit_span`` trust radii.  With
  fewer points than coefficients the fit is approximately NEWUOA's least
  change of the previous model's Hessian: a Euclidean minimum-norm
  correction over *all* coefficients (constant, gradient and Hessian
  together) around the previous Hessian, not NEWUOA's Frobenius-norm
  update of the Hessian change alone.  The curvature learned so far is
  kept in the directions the new points do not determine.  The prior is
  dropped (reset to zero) at every restart and whenever the centre moves
  by more than ``fit_span`` radii: carried across kinks or rugged basins it
  grew to 1e7–1e8 in review.  The archive is the *shared* one: every result of every arm
  (``on_new_results`` sees them all) is a candidate interpolation point,
  so a CMA-ES generation that lands near the centre improves this arm's
  model for free.
* **Centre.**  The best archive point outside the tabu balls of earlier
  converged centres — whoever found it.
* **Step.**  The model minimiser in the box trust region
  ``|u - c|_inf <= radius`` intersected with the problem box (coordinates
  normalised to ``[0, 1]^d``).  The usual ratio test on the step's result
  grows (``rho > 0.7`` at the boundary), keeps, or shrinks
  (``rho < 0.1``) the radius.
* **Geometry.**  The model is used only when the displacements of its
  points from the centre span all ``d`` directions (numerical rank ``d``
  of their SVD).  Counting points is not enough: a min-norm fit sets the
  gradient and curvature to zero outside the span, so the arm would never
  move there.  While the span is short, or after a failed step on a thin
  model, the arm emits geometry points instead: first along the
  least-covered directions (the smallest right singular vectors), then
  the coordinate design ``c + radius·e_i``, ``c - radius·e_i`` (BOBYQA's
  first ``2d + 1`` points), then space-filling points in the trust region.
* **Restart.**  When the radius falls below ``radius_min`` the centre
  becomes tabu and the radius is reset.  The tabu ball is an inf-norm ball
  of ``radius_init`` (box-normalised units) around the converged centre:
  no archive point inside any tabu ball can become a centre again, so the
  next centre is the best non-tabu archive point, or a uniform random
  point when there is none.
* **A moving centre keeps its radius.**  When the centre jumps (another
  arm found a better point far away) the radius is *not* reset.  The fit
  only uses points within ``fit_span`` radii of the new centre, so a small
  radius after a far jump means a geometry phase first.
* **Batches.**  :meth:`produce` is on demand (:attr:`on_demand`): each call
  refits on everything known and returns up to ``limit`` points: the
  model step, then steps at halved radii, then geometry points, never
  repeating a point that is already evaluated or in flight.  At ``q = 1``
  it is a plain sequential trust-region method; at ``q > 1`` the extra
  workers get the shorter steps and geometry points.

The arm is opt-in: no registry spec uses it by default
(``planning/DISCOVERY_2026-09-09.md`` §66).
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from panobbgo.core import Heuristic
from panobbgo.lib import Point


def quadratic_features(s: np.ndarray) -> np.ndarray:
    """Design matrix of a full quadratic in ``d`` variables: ``1, s_i, s_i^2 / 2, s_i s_j (i < j)``.

    ``s`` has shape ``(n, d)``; the result has ``1 + 2d + d(d-1)/2`` columns.
    """
    n, d = s.shape
    iu = np.triu_indices(d, k=1)
    cross = (s[:, :, None] * s[:, None, :])[:, iu[0], iu[1]] if d > 1 else np.zeros((n, 0))
    return np.hstack([np.ones((n, 1)), s, 0.5 * s * s, cross])


def unpack_quadratic(coef: np.ndarray, d: int) -> Tuple[float, np.ndarray, np.ndarray]:
    """``(c, g, H)`` of the model ``c + g·s + s·H·s / 2`` from :func:`quadratic_features` coefficients."""
    c = float(coef[0])
    g = np.asarray(coef[1 : 1 + d], dtype=float)
    H = np.diag(np.asarray(coef[1 + d : 1 + 2 * d], dtype=float))
    if d > 1:
        iu = np.triu_indices(d, k=1)
        cross = np.asarray(coef[1 + 2 * d :], dtype=float)
        H[iu[0], iu[1]] = cross
        H[iu[1], iu[0]] = cross
    return c, g, H


def pack_hessian(H: np.ndarray) -> np.ndarray:
    """The :func:`quadratic_features` coefficient vector of ``s·H·s / 2`` (constant and gradient zero)."""
    d = H.shape[0]
    iu = np.triu_indices(d, k=1)
    return np.concatenate([np.zeros(1 + d), np.diag(H), H[iu[0], iu[1]]])


def minimize_quadratic_in_box(g: np.ndarray, H: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Approximate minimiser of ``g·s + s·H·s / 2`` over the box ``lo <= s <= hi`` (``lo <= 0 <= hi``).

    L-BFGS-B from a few starts (the origin, the clipped Newton step when
    ``H`` is positive definite, the clipped steepest-descent corner); the
    best end point wins.  Deterministic.
    """

    def fun(s: np.ndarray) -> Tuple[float, np.ndarray]:
        hs = H @ s
        return float(g @ s + 0.5 * s @ hs), g + hs

    starts = [np.zeros_like(g)]
    try:
        if np.all(np.linalg.eigvalsh(H) > 1e-12):
            starts.append(np.clip(-np.linalg.solve(H, g), lo, hi))
    except np.linalg.LinAlgError:
        pass
    gn = np.max(np.abs(g))
    if gn > 0:
        starts.append(np.clip(-g / gn, lo, hi))
    bounds = list(zip(lo, hi))
    best_s, best_v = starts[0], 0.0
    for s0 in starts:
        res = minimize(fun, s0, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": 200})
        s = np.clip(np.asarray(res.x, dtype=float), lo, hi)
        v = fun(s)[0]
        if v < best_v:
            best_s, best_v = s, v
    return best_s


class TrustRegionQuadratic(Heuristic):
    """Quadratic-model trust-region local search on the shared archive (see the module docstring).

    Args:
        strategy: The owning :class:`~panobbgo.core.StrategyBase`.
        radius_init: Initial (and restart) trust radius, as a fraction of
            the box width per axis.  Default ``0.1`` (Py-BOBYQA's ``rhobeg``
            with ``scaling_within_bounds``).
        radius_min: Radius below which the centre is declared converged and
            the arm restarts.  Default ``1e-7``.
        radius_max: Upper bound of the radius.  Default ``0.5``.
        fit_span: Only archive points within ``fit_span * radius`` (inf-norm)
            of the centre enter the model.  Default ``3``.
        first_start: Where the first centre is when the archive is empty:
            ``"center"`` of the box (default, like ``CMAES`` and ``COBYQA``)
            or a uniform ``"random"`` point (like the Py-BOBYQA baseline).
            Restarts always draw a random point when the archive offers no
            non-tabu centre.
        name: Override the heuristic's name.
    """

    on_demand = True

    #: Ratio-test thresholds and factors.
    ETA_BAD, ETA_GOOD = 0.1, 0.7
    SHRINK, GROW = 0.5, 2.0

    def __init__(
        self,
        strategy: Any,
        radius_init: float = 0.1,
        radius_min: float = 1e-7,
        radius_max: float = 0.5,
        fit_span: float = 3.0,
        first_start: str = "center",
        name: Optional[str] = None,
    ) -> None:
        if not (0 < radius_min < radius_init <= radius_max):
            raise ValueError("TrustRegionQuadratic: need 0 < radius_min < radius_init <= radius_max")
        if not fit_span > 1:
            raise ValueError("TrustRegionQuadratic: fit_span must be > 1")
        if first_start not in ("center", "random"):
            raise ValueError("TrustRegionQuadratic: first_start must be 'center' or 'random'")
        Heuristic.__init__(self, strategy, name=name or "TrustRegionQuadratic")
        self.radius_init = float(radius_init)
        self.radius_min = float(radius_min)
        self.radius_max = float(radius_max)
        self.fit_span = float(fit_span)
        self.first_start = first_start
        self._lock = threading.RLock()
        box = np.array(self.problem.box[:, :], dtype=float)
        self._lo = box[:, 0]
        self._width = np.where(box[:, 1] > box[:, 0], box[:, 1] - box[:, 0], 1.0)
        dim = self.problem.dim
        self._U = np.empty((0, dim))  # evaluated points, normalised
        self._F = np.empty(0)  # their penalty values (finite only)
        #: The last model's Hessian in normalised coordinates and f units (the least-change prior).
        self._H_u = np.zeros((dim, dim))
        #: Centre of the previous proposal (the prior is dropped after a far jump).
        self._last_center: Optional[np.ndarray] = None
        self.radius = self.radius_init
        self._tabu: List[np.ndarray] = []
        self._restart_u: Optional[np.ndarray] = None  # unevaluated restart centre
        #: who -> what an in-flight point was emitted as.
        self._pending: Dict[str, Dict[str, Any]] = {}
        self._need_geometry = False
        #: Archive size at the last no-descent shrink: one model state shrinks the radius at most once.
        self._no_descent_state: Optional[int] = None
        #: Diagnostics.
        self.n_steps = 0
        self.n_geometry = 0
        self.n_restarts = 0
        self.n_grow = 0
        self.n_shrink = 0

    # -- coordinates -------------------------------------------------------

    def _to_u(self, x: np.ndarray) -> np.ndarray:
        return np.clip((np.asarray(x, dtype=float) - self._lo) / self._width, 0.0, 1.0)

    def _to_x(self, u: np.ndarray) -> np.ndarray:
        return self.problem.project(self._lo + np.clip(u, 0.0, 1.0) * self._width)

    # -- archive -----------------------------------------------------------

    def _penalty(self, r: Any) -> float:
        try:
            v = float(self.strategy.constraint_handler.get_penalty_value(r))
        except Exception:
            v = float(getattr(r, "fx", float("nan")))
        return v

    def _center(self) -> Tuple[Optional[np.ndarray], float]:
        """Best finite archive point outside every tabu ball, or the restart point."""
        if len(self._F):
            ok = np.ones(len(self._F), dtype=bool)
            for t in self._tabu:
                ok &= np.max(np.abs(self._U - t), axis=1) > self.radius_init
            if ok.any():
                i = int(np.flatnonzero(ok)[np.argmin(self._F[ok])])
                self._restart_u = None
                return self._U[i], float(self._F[i])
        if self._restart_u is None:
            first = self.n_restarts == 0 and self.first_start == "center"
            self._restart_u = np.full(self.problem.dim, 0.5) if first else self.rng.random(self.problem.dim)
        return self._restart_u, float("inf")

    def on_new_results(self, results: List[Any]) -> None:
        """Add every result (any arm's) to the archive; run the ratio test on this arm's steps."""
        with self._lock:
            new_u, new_f = [], []
            for r in results:
                f = self._penalty(r)
                info = self._pending.pop(str(getattr(r, "who", "") or ""), None)
                if not np.isfinite(f):
                    if info is not None and info["kind"] == "step":
                        self._step_failed(info)
                    continue
                new_u.append(self._to_u(r.x))
                new_f.append(f)
                if info is not None and info["kind"] == "step":
                    rho = (info["f_center"] - f) / info["pred"] if info["pred"] > 0 else -1.0
                    if rho < self.ETA_BAD:
                        self._step_failed(info)
                    elif rho > self.ETA_GOOD and info["at_boundary"] and self._current(info):
                        self.radius = min(self.GROW * self.radius, self.radius_max)
                        self.n_grow += 1
            if new_u:
                self._U = np.vstack([self._U, np.asarray(new_u)])
                self._F = np.concatenate([self._F, np.asarray(new_f)])
            self._maybe_restart()

    def on_failed_evaluations(self, points: List[Any]) -> None:
        """A failed step counts as a failed step; a failed geometry point is just dropped."""
        with self._lock:
            for p in points:
                info = self._pending.pop(str(getattr(p, "who", "") or ""), None)
                if info is not None and info["kind"] == "step":
                    self._step_failed(info)
            self._maybe_restart()

    def _current(self, info: Dict[str, Any]) -> bool:
        """Does a step's outcome still speak for the current radius?

        Only a full-radius step emitted at the radius in force now moves the
        radius: a shorter batch step, or one that was in flight while the
        radius already changed (``q > 1``), must not shrink or grow it twice.
        """
        return bool(info.get("primary")) and info.get("radius") == self.radius

    def _step_failed(self, info: Dict[str, Any]) -> None:
        if not self._current(info):
            return
        if info["n_fit"] < 2 * self.problem.dim + 1:
            self._need_geometry = True  # a thin model: improve it before shrinking
        else:
            self.radius *= self.SHRINK
            self.n_shrink += 1

    def _maybe_restart(self) -> None:
        if self.radius >= self.radius_min:
            return
        c, _ = self._center()
        if c is not None:
            self._tabu.append(np.array(c, copy=True))
        self.radius = self.radius_init
        self._need_geometry = False
        self._H_u = np.zeros_like(self._H_u)  # a new basin: no curvature prior
        self.n_restarts += 1

    # -- proposals ---------------------------------------------------------

    #: Relative singular-value tolerance of the rank test in :meth:`_fit`.
    RANK_RTOL = 1e-3

    def _fit(self, c: np.ndarray, f_c: float) -> Tuple[Optional[Tuple[np.ndarray, np.ndarray, float, int]], List[Any]]:
        """``(model, weak)``: the model ``(g, H, scale, n_fit)`` in ``s = (u - c) / radius`` units, or ``None``.

        ``weak`` lists the unit directions the displacements of the fit
        points from the centre do not cover, least covered first.  The model
        is ``None`` whenever ``weak`` is not empty: a rank-deficient fit has
        zero gradient and curvature outside its span.
        """
        d = self.problem.dim
        if not len(self._F) or not np.isfinite(f_c):
            return None, []
        dist = np.max(np.abs(self._U - c), axis=1)
        p = 1 + 2 * d + d * (d - 1) // 2
        cap = max(2 * d + 1, int(1.5 * p))
        near = np.flatnonzero(dist <= self.fit_span * self.radius)
        near = near[np.argsort(dist[near], kind="stable")][:cap]
        s = (self._U[near] - c) / self.radius
        _, sv, vt = np.linalg.svd(s, full_matrices=True)
        sv = np.concatenate([sv, np.zeros(d - len(sv))])
        tol = self.RANK_RTOL * max(float(sv[0]), 1e-300)
        weak = [vt[i] for i in range(d - 1, -1, -1) if sv[i] <= tol]
        if weak:
            return None, weak
        y = self._F[near] - f_c
        scale = float(np.max(np.abs(y))) or 1.0
        w = 1.0 / (1.0 + (dist[near] / self.radius) ** 2)
        A = quadratic_features(s) * np.sqrt(w)[:, None]
        # Approximately NEWUOA's least change: with fewer points than
        # coefficients, the Euclidean minimum-norm correction (all
        # coefficients) around the previous model's Hessian, not around zero.
        # Overdetermined fits do not depend on it.
        base = pack_hessian(self._H_u * self.radius**2 / scale)
        delta, *_ = np.linalg.lstsq(A, (y / scale) * np.sqrt(w) - A @ base, rcond=None)
        _, g, H = unpack_quadratic(base + delta, d)
        if not (np.all(np.isfinite(g)) and np.all(np.isfinite(H))):
            return None, []
        self._H_u = H * scale / self.radius**2
        return (g, H, scale, len(near)), []

    def _taken(self, u: np.ndarray, extra: List[np.ndarray], tol: float) -> bool:
        """Is ``u`` within ``tol`` (inf-norm) of an evaluated, in-flight or just-proposed point?"""
        pts = [self._U] if len(self._U) else []
        pend = [i["u"] for i in self._pending.values()] + extra
        if pend:
            pts.append(np.asarray(pend))
        return any(bool(np.any(np.max(np.abs(P - u), axis=1) <= tol)) for P in pts if len(P))

    def _geometry_point(self, c: np.ndarray, extra: List[np.ndarray]) -> np.ndarray:
        """Next coordinate-design point ``c ± radius·e_i`` not yet present, else a space-filling one."""
        r = self.radius
        d = self.problem.dim
        tol = 0.3 * r
        if not self._taken(c, extra, 1e-12 + 1e-9 * r):
            return c.copy()
        for sign in (1.0, -1.0):  # c + r e_1 ... c + r e_d first: together they span every direction
            for i in range(d):
                u = c.copy()
                u[i] += sign * r
                if not 0.0 <= u[i] <= 1.0:
                    u[i] = c[i] - sign * r  # BOBYQA's flip into the box
                u = np.clip(u, 0.0, 1.0)
                if not self._taken(u, extra, tol):
                    return u
        lo, hi = np.clip(c - r, 0.0, 1.0), np.clip(c + r, 0.0, 1.0)
        cand = lo + self.rng.random((20 * d, d)) * (hi - lo)
        ref = [self._U[np.max(np.abs(self._U - c), axis=1) <= 2 * r]] if len(self._U) else []
        pend = [i["u"] for i in self._pending.values()] + extra
        if pend:
            ref.append(np.asarray(pend))
        ref = [R for R in ref if len(R)]
        if not ref:
            return cand[0]
        R = np.vstack(ref)
        md = np.min(np.max(np.abs(cand[:, None, :] - R[None, :, :]), axis=2), axis=1)
        return cand[int(np.argmax(md))]

    def _step(self, c: np.ndarray, f_c: float, model: Tuple[np.ndarray, np.ndarray, float, int], radius: float):
        """The model minimiser in the trust region of ``radius``: ``(u, info)`` or ``None`` (no descent)."""
        g, H, scale, n_fit = model
        k = radius / self.radius  # the model lives in s = (u - c) / self.radius units
        lo = np.maximum(-k, (0.0 - c) / self.radius)
        hi = np.minimum(k, (1.0 - c) / self.radius)
        s = minimize_quadratic_in_box(g, H, lo, hi)
        pred = -float(g @ s + 0.5 * s @ H @ s) * scale
        if not pred > 1e-14 * max(1.0, abs(f_c)):
            return None
        u = np.clip(c + s * self.radius, 0.0, 1.0)
        at_boundary = bool(np.max(np.abs(s)) >= 0.9 * k)
        info = {
            "kind": "step",
            "pred": pred,
            "f_center": f_c,
            "radius": radius,
            "primary": radius == self.radius,
            "at_boundary": at_boundary,
            "n_fit": n_fit,
        }
        return u, info

    def _direction_point(self, c: np.ndarray, v: np.ndarray, extra: List[np.ndarray]) -> Optional[np.ndarray]:
        """``c ± radius·v`` inside the box and not yet present, or ``None``."""
        r = self.radius
        for sign in (1.0, -1.0):
            u = c + sign * r * v
            if np.all(u >= 0.0) and np.all(u <= 1.0) and not self._taken(u, extra, 0.3 * r):
                return u
        # Both ends leave the box: the clipped end that still moves furthest along v.
        best, moved = None, 0.3 * r
        for sign in (1.0, -1.0):
            u = np.clip(c + sign * r * v, 0.0, 1.0)
            m = abs(float((u - c) @ v))
            if m > moved and not self._taken(u, extra, 0.3 * r):
                best, moved = u, m
        return best

    def _propose(self, k: int) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
        c, f_c = self._center()
        assert c is not None
        last = self._last_center
        if last is not None and float(np.max(np.abs(c - last))) > self.fit_span * self.radius:
            self._H_u = np.zeros_like(self._H_u)  # a far jump: the old curvature is not local here
        self._last_center = np.array(c, copy=True)
        model, weak = (None, []) if self._need_geometry else self._fit(c, f_c)
        out: List[Tuple[np.ndarray, Dict[str, Any]]] = []
        extra: List[np.ndarray] = []
        radius = self.radius
        for _ in range(k):
            prop = None
            while model is not None and radius >= self.radius_min and prop is None:
                step = self._step(c, f_c, model, radius)
                radius *= 0.5
                if step is None:
                    if not out:
                        self._no_descent(model)
                    break
                if not self._taken(step[0], extra, 1e-3 * step[1]["radius"]):
                    prop = step
            while prop is None and weak:
                u = self._direction_point(c, weak.pop(0), extra)
                if u is not None:
                    prop = (u, {"kind": "geometry"})
            if prop is None:
                prop = (self._geometry_point(c, extra), {"kind": "geometry"})
            if prop[1]["kind"] == "geometry":
                self._need_geometry = False
            out.append(prop)
            extra.append(prop[0])
        return out

    def _no_descent(self, model: Tuple[np.ndarray, np.ndarray, float, int]) -> None:
        """A full-rank model predicts no descent at the full radius: one failed iteration.

        It shrinks the radius (or asks for geometry on a thin model) at most
        once per model state -- the archive size -- so repeated
        :meth:`produce` calls with no new result in between do not shrink it
        again and again without evaluating anything.
        """
        if self._no_descent_state == len(self._F):
            return
        self._no_descent_state = len(self._F)
        self._step_failed({"n_fit": model[3], "radius": self.radius, "primary": True})

    def produce(self, limit: Optional[int] = None, timeout: Optional[float] = None) -> List[Point]:
        """Up to ``limit`` fresh points (one when ``limit`` is ``None``), computed now from the archive."""
        del timeout
        if self._stopped:
            return []
        k = 1 if limit is None else int(limit)
        if k <= 0:
            return []
        with self._lock:
            # Points the strategy handed back undispatched (``_return_to_queues``) come first;
            # they are still in ``_pending`` and keep their meaning.
            points = self.get_points(k)
            proposals = self._propose(k - len(points)) if k > len(points) else []
            self._maybe_restart()
            for u, info in proposals:
                who = self.new_who()
                info["u"] = u
                self._pending[who] = info
                if info["kind"] == "step":
                    self.n_steps += 1
                else:
                    self.n_geometry += 1
                points.append(Point(self._to_x(u), who))
            return points

    def __stop__(self) -> None:
        """Forget the in-flight bookkeeping: points still pending when the run ends are never reported."""
        with self._lock:
            self._pending.clear()
        Heuristic.__stop__(self)

    @property
    def can_produce(self) -> bool:
        """Always, until stopped: a geometry point is always available."""
        return not self._stopped
