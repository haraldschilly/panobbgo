# -*- coding: utf8 -*-
# Copyright 2012-2026 Panobbgo Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

r"""
Feature logging at budget checkpoints
=====================================

Training data for the learned selector and the forecast allocation
(``planning/DESIGN_roadmap_2026-09-26.md`` §3.4 and §4 A/B).  At fixed
fractions of the budget a run records a compact feature dict: landscape
features of the points evaluated so far (ELA-lite), trajectory features,
per-arm features (the heuristics) and the run context.  Nothing learns from
them yet.

Recording never changes the run: :class:`FeatureLogger` is a pass observer
of the strategy's main loop (:meth:`panobbgo.core.StrategyBase.add_pass_observer`)
that only *reads* the archive and the heuristics' state, and draws no random
numbers from any shared stream (its one random draw, the coverage probes,
comes from a private generator with a fixed seed).

**Invariance** (roadmap §4 A, the table there):

* Every ``f``-based quantity is computed on **ranks** (constrained problems:
  feasible points by ``f``, then infeasible ones by violation), so it is
  invariant to ``f -> a·f + b`` (``a > 0``) and to every strictly monotone
  transform of ``f`` — including the meta-model R² values ("rank-R²"),
  which are fitted to the normalised ranks, not to raw ``f``.  Landscape and
  arm features use *average* ranks, so ties (plateaus) do not depend on the
  order the points were sampled in.
* ``x`` is normalised to the unit box and every distance is divided by
  :math:`\sqrt d`.  With equal box ranges, the distance-based features and
  the linear / full-quadratic R² and Hessian condition are invariant to a
  rotation of ``x`` (so is ``spread_iso``); the separability ratio (additive
  vs full quadratic R²) and the per-axis spread / region features are
  *deliberately* not.
* Budget quantities are per dimension (``evals / d``, rates per ``d``
  evaluations).

Never raw ``f`` values or raw coordinates.  ``None`` stands for a feature
that is undefined at this checkpoint (too few points, a degenerate fit).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

#: Default checkpoints, as fractions of the budget.
DEFAULT_CHECKPOINTS: Tuple[float, ...] = (0.05, 0.1, 0.2, 0.4, 0.7)

#: At most this many points enter the landscape features (evenly spaced over
#: the archive, the incumbent always included).
DEFAULT_MAX_POINTS: int = 500

#: Uniform probe points for the coverage features (private, fixed-seed generator).
N_PROBES: int = 128
_PROBE_SEED: int = 20260926

#: Top fractions for the dispersion features.
DISPERSION_FRACTIONS: Tuple[float, ...] = (0.1, 0.25)

#: A point closer than this (unit box, per-axis RMS) to an earlier point is a revisit.
REVISIT_TOL: float = 1e-8

#: Significant digits kept in the JSON (compactness; the features are noisy far above this).
SIG_DIGITS: int = 4


@dataclass(frozen=True)
class FeatureLogSpec:
    """What to record: the checkpoints (budget fractions) and the landscape subsample size."""

    checkpoints: Tuple[float, ...] = DEFAULT_CHECKPOINTS
    max_points: int = DEFAULT_MAX_POINTS

    def __post_init__(self) -> None:
        cps = tuple(float(c) for c in self.checkpoints)
        if not cps or any(not 0.0 < c <= 1.0 for c in cps) or list(cps) != sorted(set(cps)):
            raise ValueError(f"checkpoints must be strictly increasing fractions in (0, 1], got {self.checkpoints}")
        if int(self.max_points) < 10:
            raise ValueError("max_points must be >= 10")
        object.__setattr__(self, "checkpoints", cps)

    @classmethod
    def parse(cls, text: Optional[str]) -> "FeatureLogSpec":
        """From a CLI value: ``None``/empty for the defaults, else comma-separated fractions (``"0.1,0.5"``)."""
        if not text:
            return cls()
        return cls(checkpoints=tuple(float(t) for t in text.split(",") if t.strip()))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _c(v: Any) -> Any:
    """A compact JSON value: ``None`` for non-finite, floats to :data:`SIG_DIGITS` significant digits."""
    if v is None:
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    v = float(v)
    if not math.isfinite(v):
        return None
    return float(f"{v:.{SIG_DIGITS}g}")


def _compact(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: (_compact(v) if isinstance(v, dict) else _c(v)) for k, v in d.items()}


def _key(fx: np.ndarray, cv: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """``(infeasible, key)``: feasible points keyed by ``fx``, infeasible ones by ``cv`` (a NaN ``cv`` is infeasible)."""
    fx = np.asarray(fx, dtype=np.float64)
    if cv is None:
        return np.zeros(fx.size, dtype=bool), fx
    cv = np.nan_to_num(np.asarray(cv, dtype=np.float64), nan=np.inf)
    infeasible = cv > 0
    return infeasible, np.where(infeasible, cv, fx)


def rank_order(fx: np.ndarray, cv: Optional[np.ndarray] = None) -> np.ndarray:
    """Ordinal ranks ``0..n-1`` (0 = best): feasible by ``fx``, then infeasible by ``cv``; ties by index.

    Used only where an order is needed (best-so-far and improvement detection
    in :func:`trajectory_features`); the landscape and arm features use
    :func:`avg_ranks`, which do not depend on the sampling order.  Invariant
    to every strictly increasing transform of ``fx`` (and of ``cv``).
    ``fx`` must be finite.
    """
    infeasible, key = _key(fx, cv)
    n = key.size
    order = np.lexsort((np.arange(n), key, infeasible))
    ranks = np.empty(n, dtype=np.int64)
    ranks[order] = np.arange(n)
    return ranks


def avg_ranks(fx: np.ndarray, cv: Optional[np.ndarray] = None) -> np.ndarray:
    """0-based average ranks on the key of :func:`rank_order`: tied points share their mean rank.

    Invariant to monotone transforms of ``fx`` *and* to the order the points
    were sampled in (a plateau is one rank, whoever reached it first).
    """
    from scipy.stats import rankdata

    infeasible, key = _key(fx, cv)
    r = np.empty(key.size, dtype=np.float64)
    feas = ~infeasible
    n_feas = int(feas.sum())
    if n_feas:
        r[feas] = rankdata(key[feas], method="average") - 1.0
    if n_feas < key.size:
        r[infeasible] = n_feas + rankdata(key[infeasible], method="average") - 1.0
    return r


def _avg_ranks(a: np.ndarray) -> np.ndarray:
    """Average ranks (ties share the mean rank), for Spearman correlations."""
    from scipy.stats import rankdata

    return rankdata(np.asarray(a, dtype=np.float64), method="average")


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation, ``NaN`` for fewer than 3 points or a constant input."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 3:
        return float("nan")
    ra, rb = _avg_ranks(a), _avg_ranks(b)
    ra -= ra.mean()
    rb -= rb.mean()
    den = math.sqrt(float(ra @ ra) * float(rb @ rb))
    return float(ra @ rb) / den if den > 0 else float("nan")


def _sqdist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances between the rows of ``a`` and ``b`` (one BLAS product)."""
    d2 = a @ b.T
    d2 *= -2.0
    d2 += np.einsum("ij,ij->i", a, a)[:, None]
    d2 += np.einsum("ij,ij->i", b, b)[None, :]
    return np.maximum(d2, 0.0, out=d2)


def _adj_r2(design: np.ndarray, y: np.ndarray) -> Tuple[float, Optional[np.ndarray]]:
    """Adjusted R² of a least-squares fit (first column the intercept); ``NaN`` unless ``n > p``.

    Large designs (the full quadratic at d >= 15) are solved through the
    normal equations with a vanishing ridge (Cholesky; ~6x faster than an SVD
    at d = 40); a failed factorisation falls back to :func:`numpy.linalg.lstsq`.
    """
    n, p = design.shape
    if n <= p:
        return float("nan"), None
    coef: Optional[np.ndarray] = None
    if p > 100:
        from scipy.linalg import LinAlgError, cho_factor, cho_solve

        gram = design.T @ design
        gram[np.diag_indices(p)] += 1e-12 * float(np.trace(gram)) / p
        try:
            coef = cho_solve(cho_factor(gram, check_finite=False), design.T @ y, check_finite=False)
        except LinAlgError:
            coef = None
    if coef is None:
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef
    sst = float(((y - y.mean()) ** 2).sum())
    if sst <= 0:
        return float("nan"), None
    r2 = 1.0 - float(resid @ resid) / sst
    return 1.0 - (1.0 - r2) * (n - 1) / (n - p), coef


def n_quad_coefficients(d: int) -> int:
    """Coefficients of the full quadratic model in ``d`` dimensions: ``1 + 2d + d(d-1)/2``."""
    return 1 + 2 * d + d * (d - 1) // 2


def _subsample(n: int, k: int, keep: int) -> np.ndarray:
    """At most ``k`` evenly spaced indices of ``range(n)``, always including ``keep``; no RNG."""
    if n <= k:
        return np.arange(n)
    idx = np.unique(np.round(np.linspace(0, n - 1, k)).astype(np.int64))
    if keep not in idx:
        idx[np.argmin(np.abs(idx - keep))] = keep
        idx = np.unique(idx)
    return idx


# ---------------------------------------------------------------------------
# feature groups
# ---------------------------------------------------------------------------


def _quad_design(c: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    m, d = c.shape
    ii, jj = np.triu_indices(d, 1)
    return np.hstack([np.ones((m, 1)), c, c * c, c[:, ii] * c[:, jj]]), ii, jj


def landscape_features(u: np.ndarray, ranks: np.ndarray, max_points: int = DEFAULT_MAX_POINTS) -> Dict[str, float]:
    """ELA-lite on points ``u`` (unit box, ``n x d``) with ``ranks`` (0 = best; ties allowed).

    Every quantity uses average ranks, so it depends neither on monotone
    transforms of ``f`` nor on the order the points were sampled in (up to
    the evenly spaced subsample of at most ``max_points`` points).
    Distances are divided by √d.  Keys:

    * ``fdc`` — Spearman correlation of rank vs distance to the nearest of
      the best points.
    * Nearest-better clustering (Kerschke et al. 2015; *nearest better* is
      strictly better): ``nbc_mean_ratio`` / ``nbc_sd_ratio`` — mean / sd of
      nearest-neighbour over nearest-better distances; ``nbc_nn_nb_cor`` —
      their Pearson correlation; ``nbc_dist_ratio_cv`` — coefficient of
      variation of nn / nb; ``nbc_nb_fitness_cor`` — Spearman correlation of
      each point's nearest-better in-degree with its rank.
    * ``disp_10``, ``disp_25`` — mean pairwise distance among the top 10 / 25 %
      (ties at the cut included) over that among all.
    * ``r2_lin``, ``r2_add``, ``r2_quad`` — **rank-R²**: adjusted R² of a
      linear, an additive quadratic (no interactions) and a full quadratic
      model of the normalised ranks (not of ``f``).  The full quadratic
      (``p = 1 + 2d + d(d-1)/2`` coefficients) is fitted only with at least
      ``2p`` points — above ``max_points`` from a larger subsample of its
      own (at d = 30 / 40: 992 / 1722 points) — else ``None``.
    * ``sep_ratio`` — ``max(r2_add, 0) / r2_quad`` on the quadratic's sample,
      ``None`` unless ``r2_quad > 0.05`` (1 = separable; not rotation-invariant).
    * ``log10_cond`` — log10 of max|λ|/min|λ| of the fitted quadratic's
      Hessian (capped at 12); ``hess_pos`` — share of its positive
      eigenvalues.  A condition estimate from a fit near its sample limit is
      noisy and grows with d even on a sphere: compare it within one d.
    """
    from scipy.stats import rankdata

    n, d = u.shape
    nan = float("nan")
    keys = ("fdc", "nbc_mean_ratio", "nbc_sd_ratio", "nbc_nn_nb_cor", "nbc_dist_ratio_cv", "nbc_nb_fitness_cor")
    keys += tuple(f"disp_{round(q * 100)}" for q in DISPERSION_FRACTIONS)
    keys += ("r2_lin", "r2_add", "r2_quad", "sep_ratio", "log10_cond", "hess_pos")
    out = {k: nan for k in keys}
    if n < 4:
        return out
    ranks = np.asarray(ranks, dtype=np.float64)
    keep = int(np.argmin(ranks))
    idx = _subsample(n, max_points, keep)
    us = u[idx]
    m = us.shape[0]
    r = rankdata(ranks[idx], method="average") - 1.0  # average ranks within the subsample
    sq = math.sqrt(d)
    dist = np.sqrt(_sqdist(us, us)) / sq
    best = r == r.min()
    others = ~best
    if others.sum() >= 3:
        out["fdc"] = spearman(r[others], dist[best].min(axis=0)[others])

    # nearest-better clustering: nearest *strictly* better point
    np.fill_diagonal(dist, np.inf)
    nn = dist.min(axis=1)
    masked = np.where(r[None, :] < r[:, None], dist, np.inf)  # [i, j]: j strictly better than i
    nb_idx = masked.argmin(axis=1)
    nb = masked[np.arange(m), nb_idx]
    ok = others & np.isfinite(nb) & np.isfinite(nn)
    if ok.sum() >= 3:
        nn_o, nb_o = nn[ok], nb[ok]
        if nb_o.mean() > 0:
            out["nbc_mean_ratio"] = float(nn_o.mean() / nb_o.mean())
        if nb_o.std() > 0:
            out["nbc_sd_ratio"] = float(nn_o.std() / nb_o.std())
            if nn_o.std() > 0:
                out["nbc_nn_nb_cor"] = float(np.corrcoef(nn_o, nb_o)[0, 1])
        pos = nb_o > 0
        if pos.sum() >= 3:
            ratio = nn_o[pos] / nb_o[pos]
            if ratio.mean() > 0:
                out["nbc_dist_ratio_cv"] = float(ratio.std() / ratio.mean())
        indegree = np.bincount(nb_idx[ok], minlength=m).astype(np.float64)
        out["nbc_nb_fitness_cor"] = spearman(indegree, r)
    np.fill_diagonal(dist, 0.0)

    # dispersion of the top q vs all (symmetric, zero diagonal: pair mean = sum / (m (m - 1)))
    mean_all = float(dist.sum()) / (m * (m - 1))
    r_sorted = np.sort(r)
    for q in DISPERSION_FRACTIONS:
        k = max(2, int(math.ceil(q * m)))
        top = np.flatnonzero(r <= r_sorted[k - 1])  # ties at the cut included: order-free
        kk = top.size
        if kk < m and mean_all > 0:
            out[f"disp_{round(q * 100)}"] = float(dist[np.ix_(top, top)].sum()) / (kk * (kk - 1)) / mean_all

    # meta-models on normalised ranks (rank-R²)
    c = us - 0.5
    y = r / max(m - 1, 1)
    ones = np.ones((m, 1))
    out["r2_lin"], _ = _adj_r2(np.hstack([ones, c]), y)
    out["r2_add"], _ = _adj_r2(np.hstack([ones, c, c * c]), y)
    need = 2 * n_quad_coefficients(d)
    if n >= need:
        if m >= need:
            cq, yq, r2_add_q = c, y, out["r2_add"]
        else:
            idq = _subsample(n, need, keep)
            cq = u[idq] - 0.5
            rq = rankdata(ranks[idq], method="average") - 1.0
            yq = rq / max(idq.size - 1, 1)
            r2_add_q, _ = _adj_r2(np.hstack([np.ones((idq.size, 1)), cq, cq * cq]), yq)
        design, ii, jj = _quad_design(cq)
        r2q, coef = _adj_r2(design, yq)
        out["r2_quad"] = r2q
        if math.isfinite(r2q) and r2q > 0.05 and math.isfinite(r2_add_q):
            out["sep_ratio"] = max(r2_add_q, 0.0) / r2q
        if coef is not None:
            h = np.zeros((d, d))
            h[np.diag_indices(d)] = 2.0 * coef[1 + d : 1 + 2 * d]
            h[ii, jj] = coef[1 + 2 * d :]
            h[jj, ii] = coef[1 + 2 * d :]
            ev = np.linalg.eigvalsh(h)
            a = np.abs(ev)
            if a.max() > 0:
                out["log10_cond"] = float(min(12.0, math.log10(a.max() / max(a.min(), a.max() * 1e-12))))
                out["hess_pos"] = float((ev > 0).mean())
    return out


def coverage_features(u: np.ndarray, probes: np.ndarray) -> Dict[str, float]:
    """How much of the unit box is unsampled (global dispersion).

    * ``probe_mean``, ``probe_max`` — mean / max distance (/ √d) from uniform
      probe points to the nearest evaluated point.
    * ``coverage_ratio`` — ``probe_mean`` over the nearest-neighbour distance
      expected for as many uniform points in an *unbounded* region (a Poisson
      approximation).  Boundary effects make a uniform design score above 1,
      increasingly with d (about 1.00 / 1.10 / 1.15 / 1.22 / 1.31 at d = 2 /
      5 / 10 / 20 / 40); larger values mean the samples cluster and leave
      regions empty.  Compare it within one d.
    """
    n, d = u.shape
    nan = float("nan")
    if n == 0:
        return {"probe_mean": nan, "probe_max": nan, "coverage_ratio": nan}
    dmin = np.sqrt(_sqdist(probes, u).min(axis=1))
    mean = float(dmin.mean())
    # E[nearest-neighbour distance] of n uniform points in a unit-volume region:
    # Γ(1 + 1/d) / (n · V_d)^(1/d), V_d the unit-ball volume.
    log_vd = 0.5 * d * math.log(math.pi) - math.lgamma(0.5 * d + 1.0)
    expected = math.exp(math.lgamma(1.0 + 1.0 / d) - (math.log(n) + log_vd) / d)
    sq = math.sqrt(d)
    return {"probe_mean": mean / sq, "probe_max": float(dmin.max()) / sq, "coverage_ratio": mean / expected}


def trajectory_features(
    ranks: np.ndarray, finite: np.ndarray, d: int, n_failed: int, n_spent: int
) -> Tuple[Dict[str, float], np.ndarray, int]:
    """Progress features of the archive in booking order.

    ``ranks`` holds the ordinal rank (:func:`rank_order`, ties by index) of
    every *finite* result, in archive order (``finite`` marks which archive
    rows those are): an order is what best-so-far needs.  Returns the
    features, the archive indices that improved the best-so-far, and the
    start of the recent window.

    * ``progress_rate`` — minus the slope of ``log10(g_t)`` per ``d``
      evaluations over the second half of the run so far, where ``g_t`` is
      the share of the archive at least as good as the incumbent at ``t``
      (a rank proxy of the gap to the optimum; ``f_opt`` is unknown at run
      time).  Positive = still improving.  ``progress_rate_all``: over the
      whole run.
    * ``stall_per_d`` — evaluations since the last improvement, per ``d``;
      ``stall_share`` — the same over all evaluations so far.
    * ``improve_recent`` — improvements per evaluation in the recent window
      (the last ``max(d, 20 %)`` evaluations).
    * ``fail_share`` — failed over spent evaluations.  With a tracker,
      "failed" is its ``n_failed``: calls whose objective raised
      :class:`~panobbgo.lib.lib.EvaluationFailed` (a simulated crash or
      timeout) and failed / timed-out calls on the virtual clock; an
      ``evaluation.timeout`` placeholder whose objective still returned is
      not counted.  Without one, the archive's non-finite rows.
    """
    nan = float("nan")
    n = finite.size
    out: Dict[str, float] = {
        "progress_rate": nan,
        "progress_rate_all": nan,
        "stall_per_d": nan,
        "stall_share": nan,
        "improve_recent": nan,
        "fail_share": (n_failed / n_spent) if n_spent > 0 else nan,
    }
    window_start = max(0, n - max(d, int(math.ceil(0.2 * n))))
    if ranks.size == 0:
        return out, np.zeros(0, dtype=np.int64), window_start
    pos = np.flatnonzero(finite)  # archive index of each finite result
    bsf = np.minimum.accumulate(ranks)
    improved = np.r_[True, bsf[1:] < bsf[:-1]]
    imp_idx = pos[improved]
    # g_t over the archive: rows before the first finite value have no incumbent
    g = np.full(n, np.nan)
    g[pos] = (bsf + 1.0) / ranks.size
    # forward-fill across non-finite rows (they leave the incumbent unchanged)
    src = np.maximum.accumulate(np.where(np.isnan(g), -1, np.arange(n)))
    g = np.where(src >= 0, g[np.maximum(src, 0)], np.nan)
    t = np.arange(1, n + 1) / float(d)
    lg = np.log10(g)

    def slope(lo: int) -> float:
        sel = np.isfinite(lg[lo:])
        if sel.sum() < 3:
            return nan
        tt, yy = t[lo:][sel], lg[lo:][sel]
        if np.ptp(tt) == 0:
            return nan
        return -float(np.polyfit(tt, yy, 1)[0])

    out["progress_rate"] = slope(n // 2)
    out["progress_rate_all"] = slope(0)
    since = n - 1 - int(imp_idx[-1])
    out["stall_per_d"] = since / float(d)
    out["stall_share"] = since / float(n)
    out["improve_recent"] = float((imp_idx >= window_start).sum()) / float(n - window_start)
    return out, imp_idx, window_start


def arm_of(who: str) -> str:
    """The arm (heuristic name) of a ``who`` tag: ``"CMAES:g3:i0"`` → ``"CMAES"``."""
    return str(who).split(":", 1)[0]


def arm_features(
    u: np.ndarray,
    arms: np.ndarray,
    rank_norm: np.ndarray,
    imp_idx: np.ndarray,
    window_start: int,
) -> Dict[str, Dict[str, float]]:
    """Per-arm features, including the "stuck locally" group (roadmap §4 A).

    ``u`` is the whole archive in the unit box, ``arms`` the arm of every
    row, ``rank_norm`` the normalised rank per row (``NaN`` for a failed one).
    Per arm, over its recent samples (its last ``max(10, 2d)`` points):

    * ``share`` — the arm's share of the archive;
    * ``credit`` — its share of the improvements in the recent window
      (``None`` if there were none); ``best_rank`` — its best normalised rank
      (0 = it holds the incumbent);
    * ``spread`` — geometric mean over the axes of the samples' std (unit box;
      per axis, so not rotation-invariant); ``spread_iso`` — its
      rotation-invariant companion ``det(Cov)^(1/2d)``, the geometric-mean std
      along the principal axes; ``spread_trend`` — log10 of
      ``spread`` over that of the arm's previous as many points (negative =
      contracting);
    * ``novelty`` — median distance (/ √d) from each recent point to the
      nearest *earlier* point of the archive, over ``spread``; ``revisit`` —
      the share of recent points within :data:`REVISIT_TOL` of an earlier one;
    * ``region`` — geometric-mean side length of the recent points' bounding
      box, i.e. its share of the box to the power 1/d.
    """
    n, d = u.shape
    nan = float("nan")
    sq = math.sqrt(d)
    w = max(10, 2 * d)
    out: Dict[str, Dict[str, float]] = {}
    n_imp_recent = int((imp_idx >= window_start).sum())
    imp_arms = arms[imp_idx[imp_idx >= window_start]] if n_imp_recent else np.array([], dtype=object)
    for arm in sorted(set(arms.tolist())):
        rows = np.flatnonzero(arms == arm)
        f: Dict[str, float] = {
            "share": rows.size / float(n),
            "credit": (float((imp_arms == arm).sum()) / n_imp_recent) if n_imp_recent else nan,
            "best_rank": float(np.nanmin(rank_norm[rows])) if np.isfinite(rank_norm[rows]).any() else nan,
            "spread": nan,
            "spread_iso": nan,
            "spread_trend": nan,
            "novelty": nan,
            "revisit": nan,
            "region": nan,
        }
        recent = rows[-w:]
        if recent.size >= 3:
            ur = u[recent]
            std = ur.std(axis=0)
            spread = float(np.exp(np.log(np.maximum(std, 1e-12)).mean()))
            f["spread"] = spread
            ev = np.linalg.eigvalsh(np.atleast_2d(np.cov(ur, rowvar=False, bias=True)))
            f["spread_iso"] = float(np.exp(0.5 * np.log(np.maximum(ev, 1e-24)).mean()))
            prev = rows[-2 * w : -w] if rows.size > w else rows[:0]
            if prev.size >= 3:
                sp_prev = float(np.exp(np.log(np.maximum(u[prev].std(axis=0), 1e-12)).mean()))
                f["spread_trend"] = math.log10(spread / sp_prev)
            ext = np.ptp(ur, axis=0)
            f["region"] = float(np.exp(np.log(np.maximum(ext, 1e-12)).mean()))
            # nearest earlier point of the whole archive, per recent point
            d2 = _sqdist(ur, u[: int(recent[-1])])
            cols = np.arange(d2.shape[1])[None, :]
            d2 = np.where(cols < recent[:, None], d2, np.inf)
            dn = np.sqrt(d2.min(axis=1)) / sq
            dn = dn[np.isfinite(dn)]
            if dn.size:
                f["novelty"] = float(np.median(dn)) / spread
                f["revisit"] = float((dn < REVISIT_TOL).mean())
        out[str(arm)] = f
    return out


def heuristic_state(strategy: Any) -> Dict[str, Dict[str, float]]:
    """Optional per-arm internals, where a heuristic exposes them (CMA-ES).

    On the box-normalised covariance ``Cn = diag(1/range) · C · diag(1/range)``
    (a read-only copy of the heuristic's current ``C``, not its lazily
    refreshed eigendecomposition): ``sigma_rel`` — the step size in the unit
    box, ``sigma · det(Cn)^(1/2d)`` (geometric mean over the principal axes);
    ``log10_cond_c`` — log10 of ``cond(Cn)``; ``restarts`` — restarts so far.
    Read only; never calls into the heuristic.
    """
    out: Dict[str, Dict[str, float]] = {}
    try:
        heuristics = list(getattr(strategy, "_heuristics", {}).values())
        ranges = np.asarray(strategy.problem.ranges, dtype=np.float64)
    except Exception:
        return out
    inv = np.where(ranges > 0, 1.0 / np.where(ranges > 0, ranges, 1.0), 1.0)
    for h in heuristics:
        sigma = getattr(h, "_sigma", None)
        cmat = getattr(h, "_C", None)
        if not isinstance(sigma, (float, int)) or cmat is None:
            continue
        cmat = np.array(cmat, dtype=np.float64, copy=True)
        if cmat.shape != (ranges.size, ranges.size) or not np.all(np.isfinite(cmat)):
            continue
        cn = inv[:, None] * cmat * inv[None, :]
        ev = np.linalg.eigvalsh(0.5 * (cn + cn.T))
        st: Dict[str, float] = {}
        if ev.min() > 0:
            st["sigma_rel"] = float(sigma) * float(np.exp(0.5 * np.log(ev).mean()))
            st["log10_cond_c"] = math.log10(float(ev.max() / ev.min()))
        restarts = getattr(type(h), "n_restarts", None)
        if isinstance(restarts, property):
            try:
                st["restarts"] = float(h.n_restarts)
            except Exception:
                pass
        out[str(h.name)] = st
    return out


def compute_features(
    x: np.ndarray,
    fx: np.ndarray,
    who: Union[Sequence[str], np.ndarray],
    box: np.ndarray,
    *,
    cv: Optional[np.ndarray] = None,
    n_spent: Optional[int] = None,
    n_failed: Optional[int] = None,
    max_points: int = DEFAULT_MAX_POINTS,
    probes: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """All archive-based feature groups for one checkpoint (uncompacted floats).

    ``x`` (``n x d``, search coordinates), ``fx`` (``NaN`` for a failed
    evaluation) and ``who`` are the archive in booking order, ``box`` the
    ``d x 2`` search box.  ``n_spent`` / ``n_failed`` default to the archive's
    size / its non-finite rows (a crashed call leaves no result, so a tracker
    passes its own counts).
    """
    box = np.asarray(box, dtype=np.float64)
    fx = np.asarray(fx, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64).reshape(fx.size, box.shape[0])
    n, d = x.shape
    lo, rng = box[:, 0], np.where(np.ptp(box, axis=1) > 0, np.ptp(box, axis=1), 1.0)
    u = (x - lo) / rng
    finite = np.isfinite(fx)
    if cv is not None:
        cv = np.asarray(cv, dtype=np.float64)
    cvf = None if cv is None else cv[finite]
    ranks = rank_order(fx[finite], cvf)  # ordinal: best-so-far and improvements only
    aranks = avg_ranks(fx[finite], cvf)  # order-free: landscape and arm features
    rank_norm = np.full(n, np.nan)
    if ranks.size:
        rank_norm[finite] = aranks / max(ranks.size - 1, 1)
    spent = n if n_spent is None else int(n_spent)
    failed = int((~finite).sum()) if n_failed is None else int(n_failed)
    traj, imp_idx, window_start = trajectory_features(ranks, finite, d, failed, spent)
    land = landscape_features(u[finite], aranks, max_points=max_points)
    if probes is None:
        probes = np.random.Generator(np.random.PCG64(_PROBE_SEED)).random((N_PROBES, d))
    land.update(coverage_features(u, probes))
    tags, inverse = np.unique(np.asarray(who, dtype=str), return_inverse=True)
    arms = np.array([arm_of(t) for t in tags], dtype=object)[inverse.reshape(-1)]
    return {"land": land, "traj": traj, "arms": arm_features(u, arms, rank_norm, imp_idx, window_start)}


def _provenance(strategy: Any) -> Dict[str, Any]:
    """Where in the run a snapshot was taken: main-loop pass, dispatched and in-flight calls, virtual time."""
    out: Dict[str, Any] = {
        "pass": int(getattr(strategy, "loops", 0)),
        "dispatched": int(getattr(strategy, "_dispatched", 0)),
    }
    clock = getattr(strategy, "_virtual_clock", None)
    if clock is not None and getattr(strategy.config, "evaluation_method", None) == "virtual":
        out["in_flight"] = int(clock.busy)
        out["vtime"] = float(clock.now)
    else:
        out["in_flight"] = len(getattr(strategy, "pending", None) or ())
    return out


class FeatureLogger:
    """Pass observer that records one feature dict per checkpoint reached.

    Register it with :meth:`panobbgo.core.StrategyBase.add_pass_observer`.
    After every pass of the main loop it compares the evaluations spent so
    far with the next checkpoint (``ceil(fraction · budget)``); when one or
    more are reached it computes the features once, on the archive as it
    stands after that pass, and appends one dict per reached checkpoint to
    :attr:`records`.  With ``sync_evaluation`` a pass ends only after every
    handler has reacted to its batch, so the snapshot — the heuristics'
    state included — is deterministic.

    ``spent`` and ``failed`` count evaluations as the metric does (a
    tracker's ``n_evals`` / ``n_failed``); without them the archive is
    counted.  ``context`` holds the run-level keys the strategy cannot see
    (``q``, ``noisy``).

    Each record carries the provenance a counterfactual branch needs to
    restart from exactly this state: ``ctx.pass`` (the main-loop pass,
    ``strategy.loops``), ``ctx.dispatched`` (evaluations charged against the
    budget), ``ctx.in_flight`` (dispatched, not yet booked) and, on the
    virtual clock, ``ctx.vtime``.

    Ranks are of what the *strategy* observes (the noisy value on a noisy
    battery; constrained: feasible by ``f``, then infeasible by violation),
    not of the tracker's metric (the true value, the penalty ``f + 100·cv``,
    or the feasible gap): the features describe what a selector can see at
    run time.

    """

    def __init__(
        self,
        spec: FeatureLogSpec,
        *,
        budget: int,
        spent: Optional[Callable[[], int]] = None,
        failed: Optional[Callable[[], int]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.spec = spec
        self.budget = int(budget)
        self._spent = spent
        self._failed = failed
        self.context = dict(context or {})
        self._targets = [max(1, int(math.ceil(c * self.budget))) for c in spec.checkpoints]
        self._next = 0
        self._probes: Optional[np.ndarray] = None
        #: One compact dict per checkpoint reached, in order.
        self.records: List[Dict[str, Any]] = []
        #: Wall time spent computing features (not part of the records: they stay deterministic).
        self.elapsed_s: float = 0.0
        #: The first exception a snapshot raised (the run goes on without features).
        self.error: Optional[str] = None

    def __call__(self, strategy: Any) -> None:
        if self._next >= len(self._targets) or self.error is not None:
            return
        spent = int(self._spent()) if self._spent is not None else len(strategy.results)
        if spent < self._targets[self._next]:
            return
        import time

        t0 = time.perf_counter()
        try:
            snap = self._snapshot(strategy, spent)
        except Exception as e:  # noqa: BLE001 — logging must never end a run
            self.error = f"{type(e).__name__}: {e}"
            self._next = len(self._targets)
            return
        finally:
            self.elapsed_s += time.perf_counter() - t0
        while self._next < len(self._targets) and spent >= self._targets[self._next]:
            self.records.append({"checkpoint": self.spec.checkpoints[self._next], **snap})
            self._next += 1

    def _snapshot(self, strategy: Any, spent: int) -> Dict[str, Any]:
        hist = strategy.results.get_history()
        x = np.asarray(hist["x"], dtype=np.float64)
        fx = np.asarray(hist["fx"], dtype=np.float64)
        n = fx.size
        d = int(strategy.problem.dim)
        x = x.reshape(n, d)
        cv_vec = np.asarray(hist.get("cv_vec", np.zeros((n, 0))))
        constrained = cv_vec.ndim == 2 and cv_vec.shape[1] > 0
        cv = np.asarray(hist["cv"], dtype=np.float64) if constrained else None
        if self._probes is None:
            self._probes = np.random.Generator(np.random.PCG64(_PROBE_SEED)).random((N_PROBES, d))
        failed = int(self._failed()) if self._failed is not None else None
        feats = compute_features(
            x,
            fx,
            list(hist["who"]),
            np.asarray(strategy.problem.box.box, dtype=np.float64),
            cv=cv,
            n_spent=spent,
            n_failed=failed,
            max_points=self.spec.max_points,
            probes=self._probes,
        )
        for arm, st in heuristic_state(strategy).items():
            if arm in feats["arms"]:
                feats["arms"][arm].update(st)
        ctx = {
            "dim": d,
            "budget": self.budget,
            "evals": spent,
            "frac": spent / self.budget if self.budget else float("nan"),
            "evals_per_d": spent / d,
            "remaining_per_d": max(self.budget - spent, 0) / d,
            "archive": n,
            "n_arms": len(feats["arms"]),
            "q": int(self.context.get("q", 1)),
            "noisy": bool(self.context.get("noisy", False)),
            "constrained": bool(constrained),
            **_provenance(strategy),
        }
        return _compact({"ctx": ctx, "land": feats["land"], "traj": feats["traj"], "arms": feats["arms"]})
