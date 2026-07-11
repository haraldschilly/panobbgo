# -*- coding: utf8 -*-
# Copyright 2012-2026 Harald Schilly <harald.schilly@gmail.com>
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
Nearby Heuristic
================

Generates new candidate points near the current best point.

Optionally integrates with the :class:`~panobbgo.analyzers.sensitivity.Sensitivity`
analyzer: when importance scores are available, perturbations are scaled by dimension
importance so the heuristic focuses search effort on the dimensions that actually matter.

Curvature-aware quadratic step (``quadratic=True``)
---------------------------------------------------

The default isotropic perturbation is an *un*\\ informed local move: it treats
every direction the same regardless of how the objective actually curves.  On
smooth, ill-conditioned *valleys* — the Rosenbrock family, and any instance the
randomized battery hits with its log-uniform diagonal scaling — an isotropic
step is badly inefficient: most random perturbations of the best point step
*across* the narrow valley (uphill) rather than *along* it.  This is the
sharpest measured competitive gap (every Panobbgo strategy scores ``0`` on
``Rosenbrock_5D`` while stock ``scipy`` dual annealing solves it, because dual
annealing owes that win to a curvature-aware local-search step).

With ``quadratic=True`` the heuristic keeps a small rolling buffer of the most
recent evaluated ``(x, f(x))`` pairs and, on each new best, fits a **local
quadratic model** to the points nearest the incumbent (distance-weighted ridge
least squares in box-normalised coordinates).  It then emits the model's
trust-region-constrained Newton minimiser as the *first* refinement point — a
curvature-aware step that follows the valley floor — while the remaining
``new - 1`` points stay isotropic perturbations so exploration is preserved.
When too few points are buffered, or the fit is not usable, every point falls
back to the isotropic perturbation, so the heuristic degrades gracefully to its
classic behaviour.  ``quadratic=False`` (the default) is byte-for-byte the
historical heuristic — no buffer is kept and no model is ever fitted.

Unlike :class:`~panobbgo.heuristics.lbfgsb.LBFGSB` (a subprocess-based
finite-difference quasi-Newton descent) the model here is fitted **in process**
from points the *rest of the portfolio* already evaluated, so it spends **zero**
extra objective evaluations building its curvature estimate — the L-BFGS-B
worker, by contrast, spends ``O(dim)`` evaluations per gradient.
"""

import threading

import numpy as np
from panobbgo.analyzers.best import Best
from panobbgo.core import Heuristic


def fit_quadratic_step(
    points: np.ndarray,
    values: np.ndarray,
    center: np.ndarray,
    ranges: np.ndarray,
    trust_radius: float,
    ridge: float = 1e-2,
    min_r2: float = 0.0,
    max_points: int | None = None,
    weight_sigma: float | None = 0.35,
) -> np.ndarray | None:
    """Fit a local quadratic model and return a trust-region Newton step.

    Fits ``f(x) ≈ b + gᵀu + ½ uᵀ H u`` (with ``u = (x - center) / ranges`` a
    box-normalised offset) to the ``(points, values)`` sample by
    distance-weighted ridge least squares, then returns the step
    ``p`` (in *raw* coordinates, to be **added to** ``center``) that minimises
    the model, clipped to a ball of radius ``trust_radius`` in normalised
    space.  Returns ``None`` when no reliable step is available.

    Working in normalised coordinates keeps the design matrix well conditioned
    regardless of the box scale — crucial under the randomized battery's
    log-uniform diagonal scaling, which is exactly the ill-conditioning a
    curvature-aware step is meant to exploit.

    Args:
        points: ``(n, dim)`` array of evaluated points.
        values: ``(n,)`` objective (penalty) values; non-finite rows dropped.
        center: ``(dim,)`` incumbent the model is expanded around.
        ranges: ``(dim,)`` per-dimension box widths (``> 0``); used to
            normalise coordinates.
        trust_radius: Maximum ‖p‖ in normalised units.  Must be ``> 0``.
        ridge: Ridge (Tikhonov) strength on the non-intercept coefficients,
            relative to the mean design-matrix scale.  Larger shrinks the
            model toward flat (weaker, safer curvature).
        min_r2: Minimum weighted coefficient of determination (``R²``) the
            fitted model must achieve on its own sample to be trusted.  When
            the local data is not well explained by a single quadratic — the
            multimodal case, where the neighbourhood straddles several basins
            — the ``R²`` is low and ``None`` is returned so the caller falls
            back to isotropic exploration.  ``0.0`` (default) disables the
            gate.  A valley (Rosenbrock) fits with ``R² ≈ 1``.
        max_points: Cap on the nearest points used for the fit; defaults to
            ``4 · n_features`` so the fit stays local and cheap.
        weight_sigma: Bandwidth of the distance kernel that weights each sample
            in the least-squares fit, expressed as a fraction of the *median*
            neighbour distance.  When a float, a Gaussian kernel
            ``w_i = exp(-½ (d_i / (weight_sigma · median_d))²)`` down-weights
            points far from the incumbent, so the quadratic is fitted
            preferentially to the *local* neighbourhood — the regime a
            curvature-aware step must model well on a curved valley.  A
            quadratic averaged over a wide cloud is a poor model of a quartic
            Rosenbrock valley and yields a Newton step that stalls; localising
            the fit roughly quartered the residual objective on a synthetic 5-D
            Rosenbrock valley (see the 2026-07-11 entry in
            ``planning/SELF_IMPROVEMENT_LOG.md``).  ``None`` selects the legacy
            rank-based weights ``1 / (1 + rank)`` (byte-for-byte the pre-
            2026-07-11 behaviour) for callers that need it.  Default ``0.35``.

    Returns:
        The raw-coordinate step ``p`` (shape ``(dim,)``), or ``None``.
    """
    points = np.asarray(points, dtype=float)
    values = np.asarray(values, dtype=float)
    center = np.asarray(center, dtype=float)
    ranges = np.asarray(ranges, dtype=float)

    if points.ndim != 2 or points.shape[0] == 0:
        return None
    dim = points.shape[1]
    if center.shape[0] != dim or ranges.shape[0] != dim:
        return None
    if not np.isfinite(trust_radius) or trust_radius <= 0.0:
        return None

    finite = np.isfinite(values) & np.all(np.isfinite(points), axis=1)
    points = points[finite]
    values = values[finite]
    if points.shape[0] == 0:
        return None

    safe_ranges = np.where(ranges > 0.0, ranges, 1.0)
    u = (points - center) / safe_ranges  # (n, dim) normalised offsets

    n_features = 1 + dim + dim * (dim + 1) // 2
    if max_points is None:
        max_points = 4 * n_features

    # Keep the points nearest the incumbent so the model stays local.
    dists = np.linalg.norm(u, axis=1)
    order = np.argsort(dists)[:max_points]
    u = u[order]
    values = values[order]
    dists = dists[order]

    n = u.shape[0]
    # Need at least enough points to estimate per-axis curvature; the ridge
    # covers the (denser) cross terms when the sample is thin.
    min_points = 2 * dim + 1
    if n < min_points:
        return None

    # Design matrix: [1, u_i, u_i·u_j (i ≤ j)] in the same order the Hessian
    # extraction below assumes.
    triu_i, triu_j = np.triu_indices(dim)
    quad = u[:, triu_i] * u[:, triu_j]  # (n, dim*(dim+1)/2)
    phi = np.concatenate([np.ones((n, 1)), u, quad], axis=1)  # (n, n_features)

    # Sample weights.  A Gaussian distance kernel (default) localises the fit
    # so the quadratic models the immediate neighbourhood of the incumbent —
    # essential on a curved valley, where a model averaged over a wide cloud
    # points the Newton step across (not along) the valley and stalls.  The
    # bandwidth is a fraction of the *median* neighbour distance, so the scheme
    # is scale-free and adapts to however tightly the portfolio has clustered.
    # ``weight_sigma=None`` restores the legacy rank weights (nearest point 1,
    # then 1/2, 1/3, …) for byte-for-byte backward compatibility.
    if weight_sigma is None:
        rank = np.argsort(np.argsort(dists))
        w = 1.0 / (1.0 + rank)
    else:
        scale = float(np.median(dists))
        if not np.isfinite(scale) or scale <= 0.0:
            scale = float(np.max(dists)) if dists.size else 0.0
        if not np.isfinite(scale) or scale <= 0.0:
            # Degenerate cloud (all points coincide) — fall back to uniform.
            w = np.ones_like(dists)
        else:
            w = np.exp(-0.5 * (dists / (weight_sigma * scale)) ** 2)

    phi_w = phi * w[:, None]
    ata = phi.T @ phi_w  # weighted Gram matrix (n_features, n_features)
    aty = phi_w.T @ values

    # Per-column (standardization-equivalent) ridge: shrink each coefficient
    # relative to its *own* column scale, leaving the intercept unpenalised.
    # A single trace-based lambda would be dominated by the intercept column
    # (whose scale is ~1) and would swamp the much smaller-scaled linear /
    # quadratic columns, biasing the recovered minimiser badly.
    reg_diag = np.diag(ata).copy()
    reg_diag[0] = 0.0
    try:
        beta = np.linalg.solve(ata + ridge * np.diag(reg_diag), aty)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(beta)):
        return None

    # Fit-quality gate: only trust the model where a single quadratic actually
    # explains the local sample.  On a smooth valley R² ≈ 1; on a multimodal
    # neighbourhood spanning several basins it is low, and we bail out so the
    # caller keeps exploring isotropically instead of over-exploiting a bad
    # model.
    if min_r2 > 0.0:
        pred = phi @ beta
        wsum = w.sum()
        wmean = float((w * values).sum() / wsum) if wsum > 0 else 0.0
        ss_res = float((w * (values - pred) ** 2).sum())
        ss_tot = float((w * (values - wmean) ** 2).sum())
        if ss_tot <= 1e-30:
            return None
        r2 = 1.0 - ss_res / ss_tot
        if r2 < min_r2:
            return None

    # Unpack gradient g and Hessian H of the model at the incumbent.
    g = beta[1 : 1 + dim]
    quad_coef = beta[1 + dim :]
    hess = np.zeros((dim, dim))
    for k, (i, j) in enumerate(zip(triu_i, triu_j)):
        c = quad_coef[k]
        if i == j:
            hess[i, i] = 2.0 * c
        else:
            hess[i, j] = c
            hess[j, i] = c

    # Regularise the Hessian to positive definite so the Newton step is a
    # descent direction even in indefinite (saddle) regions.
    try:
        eigval, eigvec = np.linalg.eigh(hess)
    except np.linalg.LinAlgError:
        return None
    scale = max(np.max(np.abs(eigval)), 1e-8)
    eig_floor = 1e-3 * scale
    eigval = np.maximum(eigval, eig_floor)
    hess_inv = eigvec @ np.diag(1.0 / eigval) @ eigvec.T

    step_u = -hess_inv @ g  # normalised-space Newton step
    if not np.all(np.isfinite(step_u)):
        return None

    # Trust region: never step further than the local data supports.  A
    # quadratic fitted to a small cloud is only trustworthy *inside* that
    # cloud; stepping beyond it is extrapolation, which blows up on strongly
    # non-quadratic landscapes (e.g. Rosenbrock's quartic valley).  Cap the
    # step at the smaller of the caller's ``trust_radius`` and the radius of
    # the kept data cloud.
    support_radius = float(np.max(dists))
    eff_trust = min(trust_radius, support_radius)
    if not np.isfinite(eff_trust) or eff_trust <= 0.0:
        return None

    norm = np.linalg.norm(step_u)
    if norm <= 1e-12:
        return None
    if norm > eff_trust:
        step_u = step_u * (eff_trust / norm)

    return step_u * safe_ranges


class Nearby(Heuristic):
    """
    Generates new points based on a cheap, fast algorithm.

    For each new best point, generates ``new`` many nearby points by applying
    a random perturbation of magnitude ``radius`` (relative to dimension ranges).

    When a :class:`~panobbgo.analyzers.sensitivity.Sensitivity` analyzer is active
    and has produced importance scores, the perturbation along each dimension is
    scaled proportionally to that dimension's importance score. This focuses the
    local search on dimensions that have the greatest impact on the objective,
    making it significantly more efficient in high-dimensional problems.

    Arguments:

    - ``axes``:

      * ``one``: perturb only one randomly chosen axis
      * ``all``: perturb all axes simultaneously

    - ``new``: number of new points to generate per best update (default: 1)
    - ``radius``: perturbation size as a fraction of the dimension's range (default: 0.01)
    - ``sensitivity_scale``: when sensitivity data is available, scale perturbations
      by dimension importance raised to this power (default: 1.0). Higher values focus
      more aggressively on important dimensions; 0.0 disables sensitivity scaling.
    - ``quadratic``: when ``True``, keep a rolling buffer of recent evaluated
      ``(x, f(x))`` pairs and, on each new best, emit the trust-region Newton
      minimiser of a locally-fitted quadratic model as the first refinement
      point (a curvature-aware step; see the module docstring).  The remaining
      ``new - 1`` points stay isotropic perturbations.  ``False`` (default) is
      byte-for-byte the classic behaviour — no buffer, no model.
    - ``quadratic_trust``: trust-region radius for the quadratic step, as a
      multiple of ``radius`` in per-axis box-normalised units (default: 2.0).
      Only consulted when ``quadratic=True``.
    - ``quadratic_min_r2``: minimum weighted ``R²`` the local quadratic model
      must achieve to be trusted (default: 0.8).  Below it the step is skipped
      and the point falls back to an isotropic perturbation — this is what
      keeps the curvature step from over-exploiting on multimodal landscapes
      whose neighbourhood straddles several basins.  Only consulted when
      ``quadratic=True``.
    - ``quadratic_weight_sigma``: bandwidth of the Gaussian distance kernel
      that localises the quadratic fit, as a fraction of the median neighbour
      distance (default: 0.35).  Smaller focuses the model more tightly on the
      immediate neighbourhood of the incumbent (a better local model of a
      curved valley, but noisier if pushed too small); larger approaches an
      unweighted fit over the whole buffer.  Only consulted when
      ``quadratic=True``.  See :func:`fit_quadratic_step`.
    """

    def __init__(
        self,
        strategy,
        cap: int = 3,
        radius: float = 1.0 / 100,
        new: int = 1,
        axes: str = "one",
        sensitivity_scale: float = 1.0,
        quadratic: bool = False,
        quadratic_trust: float = 2.0,
        quadratic_min_r2: float = 0.8,
        quadratic_weight_sigma: float = 0.35,
    ):
        if not isinstance(quadratic, bool):
            raise ValueError(f"Nearby: quadratic must be a bool, got {quadratic!r}")
        if not np.isfinite(quadratic_trust) or quadratic_trust <= 0.0:
            raise ValueError(f"Nearby: quadratic_trust must be a positive finite float, got {quadratic_trust!r}")
        if not np.isfinite(quadratic_min_r2) or not (0.0 <= quadratic_min_r2 <= 1.0):
            raise ValueError(f"Nearby: quadratic_min_r2 must be a float in [0, 1], got {quadratic_min_r2!r}")
        if not np.isfinite(quadratic_weight_sigma) or quadratic_weight_sigma <= 0.0:
            raise ValueError(
                f"Nearby: quadratic_weight_sigma must be a positive finite float, got {quadratic_weight_sigma!r}"
            )
        name = "Nearby %.3f/%s" % (radius, axes)
        if quadratic:
            name += "+quad"
        Heuristic.__init__(self, strategy, cap=cap, name=name)
        self.radius = radius
        self.new = new
        self.axes = axes
        self.sensitivity_scale = sensitivity_scale
        self.quadratic = quadratic
        self.quadratic_trust = float(quadratic_trust)
        self.quadratic_min_r2 = float(quadratic_min_r2)
        self.quadratic_weight_sigma = float(quadratic_weight_sigma)
        self._depends_on = [Best]

        # Importance scores from the Sensitivity analyzer — None until first update
        self._importance: np.ndarray | None = None

        # Rolling buffer of recent evaluated (x, penalty-value) pairs, used only
        # when ``quadratic`` is enabled.  Guarded by ``_hist_lock`` because the
        # EventBus dispatches ``on_new_results`` (writer) and ``on_new_best``
        # (reader) on separate threads.
        self._hist_lock = threading.Lock()
        self._hist_x: list[np.ndarray] = []
        self._hist_f: list[float] = []
        # Cap sized to comfortably hold a full quadratic fit's worth of points
        # (n_features = 1 + dim + dim(dim+1)/2) with headroom for the nearest-
        # point selection; resolved lazily on the first result batch.
        self._hist_cap: int | None = None

    def on_new_sensitivity(self, importance: np.ndarray) -> None:
        """
        Called when the Sensitivity analyzer produces updated importance scores.

        Args:
            importance: Array of shape ``(dim,)`` with values in ``[0, 1]``.
                Higher means more important. Used to scale perturbation per dimension.
        """
        self._importance = importance

    def _perturbation_weights(self) -> np.ndarray | None:
        """
        Return per-dimension perturbation weights based on sensitivity, or None.

        Weights are non-negative and normalised so that their mean equals 1,
        preserving the overall perturbation magnitude set by ``radius``.
        """
        if self._importance is None or self.sensitivity_scale == 0.0:
            return None

        # Raise to sensitivity_scale power for adjustable contrast
        raw = np.maximum(self._importance, 1e-3) ** self.sensitivity_scale

        # Normalise so mean = 1 → overall perturbation magnitude unchanged
        mean_w = raw.mean()
        if mean_w > 0:
            return raw / mean_w
        return None

    def _make_perturbation(self, x: np.ndarray) -> np.ndarray:
        """
        Apply a single random perturbation to *x* and return the new candidate.

        If sensitivity weights are available the perturbation is scaled per-dimension;
        otherwise the standard uniform perturbation is used.
        """
        new_x = x.copy()
        weights = self._perturbation_weights()

        if self.axes == "all":
            dx = (2.0 * np.random.rand(self.problem.dim) - 1.0) * self.radius
            dx *= self.problem.ranges
            if weights is not None:
                dx *= weights
            new_x += dx

        elif self.axes == "one":
            if weights is not None:
                # Sample dimension proportional to importance
                probs = weights / weights.sum()
                idx = np.random.choice(self.problem.dim, p=probs)
            else:
                idx = np.random.randint(self.problem.dim)
            dx = (2.0 * np.random.rand() - 1.0) * self.radius
            dx *= self.problem.ranges[idx]
            if weights is not None:
                dx *= weights[idx]
            new_x[idx] += dx

        else:
            raise ValueError(
                f"Nearby heuristic received invalid 'axes' parameter: '{self.axes}'. "
                f"Valid options are 'one' (perturb one axis) or 'all' (perturb all axes)."
            )

        return self.problem.project(new_x)

    def _hist_capacity(self) -> int:
        """Rolling-buffer size — lazily sized to hold a full quadratic fit."""
        if self._hist_cap is None:
            dim = self.problem.dim
            n_features = 1 + dim + dim * (dim + 1) // 2
            self._hist_cap = max(8 * dim, 4 * n_features, 64)
        return self._hist_cap

    def on_new_results(self, results) -> None:
        """Accumulate recent ``(x, penalty-value)`` pairs for the quadratic fit.

        A no-op unless ``quadratic`` is enabled, so ``quadratic=False`` keeps
        the classic heuristic free of any per-result bookkeeping.
        """
        if not self.quadratic:
            return
        get_val = self.strategy.constraint_handler.get_penalty_value
        cap = self._hist_capacity()
        with self._hist_lock:
            for result in results:
                x = getattr(result, "x", None)
                if x is None:
                    continue
                try:
                    fx = float(get_val(result))
                except (TypeError, ValueError):
                    continue
                self._hist_x.append(np.asarray(x, dtype=float))
                self._hist_f.append(fx)
            # Trim to the most recent ``cap`` entries.
            overflow = len(self._hist_x) - cap
            if overflow > 0:
                del self._hist_x[:overflow]
                del self._hist_f[:overflow]

    def _quadratic_candidate(self, center: np.ndarray) -> np.ndarray | None:
        """Return the projected trust-region Newton step around ``center``, or None."""
        with self._hist_lock:
            if len(self._hist_x) < 2 * self.problem.dim + 1:
                return None
            points = np.array(self._hist_x, dtype=float)
            values = np.array(self._hist_f, dtype=float)
        ranges = self.problem.ranges
        # Trust radius in normalised units: a Newton step up to ``quadratic_trust``
        # times a full isotropic perturbation (magnitude ``radius`` per axis).
        trust_radius = self.quadratic_trust * self.radius * np.sqrt(self.problem.dim)
        step = fit_quadratic_step(
            points,
            values,
            center,
            ranges,
            trust_radius,
            min_r2=self.quadratic_min_r2,
            weight_sigma=self.quadratic_weight_sigma,
        )
        if step is None:
            return None
        return self.problem.project(center + step)

    def on_restart(self, center: np.ndarray, reason: str) -> None:
        """
        Respond to a restart event by generating points around the new center.

        Args:
            center: New search center suggested by the Restart analyzer.
            reason: Human-readable reason string for the restart.
        """
        if center is None:
            return
        self.clear_output()
        ret = [self._make_perturbation(center) for _ in range(self.new)]
        self.emit(ret)

    def on_new_best(self, best) -> None:
        """
        React to a new best result by generating nearby candidate points.

        When ``quadratic`` is enabled and a local model can be fitted, the first
        emitted point is the trust-region Newton minimiser of that model (a
        curvature-aware step); the remaining points stay isotropic perturbations
        so exploration is preserved.

        Args:
            best: The new best :class:`~panobbgo.lib.Result`.
        """
        x = best.x
        if x is None:
            return
        self.clear_output()
        ret: list[np.ndarray] = []
        if self.quadratic:
            step = self._quadratic_candidate(x)
            if step is not None:
                ret.append(step)
        while len(ret) < self.new:
            ret.append(self._make_perturbation(x))
        self.emit(ret)
