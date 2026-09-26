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

r"""
Parametrised problem families
=============================

Cheap, generated test-problem *instances* with a **known optimum**, so the
AOCC track (:mod:`panobbgo.harness_families`) can score them the same way
it scores MA-BBOB.

Why this module exists
----------------------

The AOCC battery of record (``planning/GOAL.md`` §2) measures one regime:
noiseless, unconstrained MA-BBOB at :math:`d \le 5`.  In that regime a
sharing portfolio is *level* with the best single arm
(``planning/DISCOVERY_2026-09-09.md`` §27/§31) — which says nothing about
the thing a portfolio is actually for: robustness across problem
*classes*.  This module supplies many cheap parametrised classes,
including constrained ones, which no existing battery covers at all.

The construction (BBOB-style)
-----------------------------

Every instance is one base function put through an affine transform:

.. math::

    f(x) = f_{\mathrm{base}}\bigl(\Lambda \, R \, (x - x_{\mathrm{opt}})\bigr)
           + f_{\mathrm{opt}}

* every base function is normalised so that
  :math:`f_{\mathrm{base}}(0) = 0` **exactly** (the raw value at the base's
  own minimiser is subtracted at construction time, so the identity holds
  bit-for-bit, not just to float tolerance) and so that ``0`` is its
  *global* minimiser over :math:`\mathbb{R}^d`;
* :math:`x_{\mathrm{opt}}` is drawn uniformly inside the box, with a
  margin to the boundary, so the optimum is never a corner artefact;
* :math:`R` is a Haar-random orthogonal matrix (QR of a Gaussian matrix
  with the sign correction, so it is uniform on :math:`O(d)`);
* :math:`\Lambda = \mathrm{diag}(\kappa^{i/(d-1)})_{i=0}^{d-1}` is the
  optional ill-conditioning, ``condition`` :math:`= \kappa`;
* :math:`f_{\mathrm{opt}}` is a random vertical offset, so nothing can
  assume ``f_opt == 0``.

Because the transform is a bijection and the base's minimiser is global,
:math:`x_{\mathrm{opt}}` is the global minimiser of the instance and
:math:`f(x_{\mathrm{opt}}) = f_{\mathrm{opt}}` — which is exactly the
``(x_opt, f_opt)`` pair AOCC needs.

Constrained instances
---------------------

See :class:`Family` for the constraint construction; the short version is
that every constraint is built *around* :math:`x_{\mathrm{opt}}` so the
optimum stays feasible and the constrained optimum equals the
unconstrained one, and the first constraint is exactly **active** at the
optimum (zero slack) so the constraints are not decoration.

BBOB bases
----------

Five bases follow the BBOB noiseless definitions [BBOB2009]_ literally —
``attractive_sector`` (f6), ``step_ellipsoid`` (f7), ``bent_cigar``
(f12), ``gallagher`` (f21/f22) and ``lunacek_bi_rastrigin`` (f24),
including their internal oscillation (:math:`T_{\mathrm{osz}}`),
asymmetry (:math:`T_{\mathrm{asy}}^\beta`), conditioning
(:math:`\Lambda^\alpha`) and second rotation :math:`Q`.  The instance's
:math:`R` plays BBOB's :math:`R` and its :math:`x_{\mathrm{opt}}` BBOB's
:math:`x_{\mathrm{opt}}`; what these bases need of the instance (the
second rotation, the peaks, the sign pattern) is drawn from a stream of
their own (:class:`BaseContext`), so the instance draws of the other
bases are unchanged.  The deviations — all forced by a shifted,
arbitrary :math:`x_{\mathrm{opt}}` or by the instance box standing in for
BBOB's :math:`f_{\mathrm{pen}}` — are listed at each base.

.. [BBOB2009] N. Hansen, S. Finck, R. Ros and A. Auger, *Real-Parameter
   Black-Box Optimization Benchmarking 2009: Noiseless Functions
   Definitions*, INRIA research report RR-6829 (2009, updated 2010),
   https://hal.inria.fr/inria-00362633.

Failure regions
---------------

``Family(..., failure=FailureRegion(...))`` adds a region of the box
where an evaluation returns no value: :meth:`Family.eval` raises
:class:`~panobbgo.lib.lib.EvaluationCrashed` (booked as a failed
evaluation) or :class:`~panobbgo.lib.lib.EvaluationTimedOut` (booked as a
timed-out ``NaN`` result, without any waiting).  See
:class:`FailureRegion`.

.. codeauthor:: Harald Schilly <harald.schilly@gmail.com>
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from panobbgo.lib.classic import Ackley, DeJong, Griewank, Rastrigin, Rosenbrock, Schwefel
from panobbgo.lib.lib import EvaluationCrashed, EvaluationTimedOut, Problem

#: Callable ``dim -> base function`` (the base's minimiser is folded in by
#: :func:`_normalise`).  The BBOB bases in :data:`CONTEXT_BASES` also take
#: a :class:`BaseContext` and their knobs: ``factory(dim, ctx, **knobs)``.
BaseFactory = Callable[..., Callable[[np.ndarray], float]]

#: Argmin of ``-u sin(sqrt|u|)``, i.e. the per-coordinate minimiser of the
#: Schwefel function, to Brent's precision.  The textbook value
#: ``420.9687`` is off by ~5e-5, which would leave the "known" optimum
#: about 3e-11 above the true one — inside the AOCC floor, but there is no
#: reason to carry the error.
SCHWEFEL_ARGMIN: float = 420.9687463319553


# ---------------------------------------------------------------------------
# Base functions
# ---------------------------------------------------------------------------
#
# A base function must have its **global** minimum over all of R^d at the
# origin.  That is a stronger requirement than "documented optimum": the
# affine transform moves points far outside the instance box, so a
# function whose optimum is only box-local (Schwefel) needs a boundary
# penalty to keep the known optimum global.  Functions that cannot be
# fixed that way are simply not offered here.


def _boundary_penalty(u: np.ndarray, half_width: float) -> float:
    """BBOB-style boundary penalty ``sum_i max(0, |u_i| - half_width)^2``.

    Zero inside the base's own reference box, quadratic outside.  Used
    only for bases whose minimum is box-local; the coefficient (1.0) is
    verified sufficient for Schwefel — see :func:`_schwefel_base`.
    """
    over = np.maximum(0.0, np.abs(u) - half_width)
    return float(np.dot(over, over))


def _normalise(
    raw: Callable[[np.ndarray], float],
    minimiser: np.ndarray,
) -> Callable[[np.ndarray], float]:
    """Shift ``raw`` so the returned ``f`` has ``f(0) == 0.0`` exactly.

    The offset is the *evaluated* value at ``minimiser``, so the identity
    is exact in floating point rather than "zero up to 4e-16" (Ackley's
    ``-20 - e + 20 + e`` does not cancel exactly, and an inexact ``f_opt``
    would show up as a spurious 1e-16 precision floor in AOCC).
    """
    shift = float(raw(minimiser))

    def f(z: np.ndarray) -> float:
        return float(raw(z + minimiser)) - shift

    return f


def _sphere_base(dim: int) -> Callable[[np.ndarray], float]:
    """De Jong / sphere, :math:`\\sum_i z_i^2`.  Minimum ``0`` at the origin."""
    prob = DeJong(dims=dim)
    return _normalise(prob.eval, np.zeros(dim))


def _rosenbrock_base(dim: int) -> Callable[[np.ndarray], float]:
    """Rosenbrock.  Documented minimum ``0`` at :math:`(1, \\ldots, 1)`."""
    prob = Rosenbrock(dims=dim)
    return _normalise(prob.eval, np.ones(dim))


def _rastrigin_base(dim: int) -> Callable[[np.ndarray], float]:
    """Rastrigin, ``par1 = 10``.  Documented minimum ``0`` at the origin."""
    prob = Rastrigin(dims=dim)
    return _normalise(prob.eval, np.zeros(dim))


def _ackley_base(dim: int) -> Callable[[np.ndarray], float]:
    """Ackley.  Documented minimum ``0`` at the origin."""
    prob = Ackley(dims=dim)
    return _normalise(prob.eval, np.zeros(dim))


def _griewank_base(dim: int) -> Callable[[np.ndarray], float]:
    """Griewank on its natural ``[-600, 600]`` scale.

    Documented minimum ``0`` at the origin.  The argument is scaled by
    ``120`` so the instance box ``[-5, 5]`` covers the classic domain;
    without it the cosine product never completes a period and the
    function degenerates into a quadratic.
    """
    prob = Griewank(dims=dim)

    def scaled(u: np.ndarray) -> float:
        return float(prob.eval(u * 120.0))

    return _normalise(scaled, np.zeros(dim))


def _schwefel_base(dim: int) -> Callable[[np.ndarray], float]:
    """Schwefel, recentred on its minimiser and fenced by a boundary penalty.

    Documented minimum ``0`` at :math:`(420.9687, \\ldots)` — but only
    *within* ``[-500, 500]``: the raw function is unbounded below over
    :math:`\\mathbb{R}^d`, and the affine transform of a family instance
    routinely leaves that box.  So this base adds
    :func:`_boundary_penalty` with coefficient ``1.0``, which was checked
    numerically on a 4-million-point grid over ``[-2000, 2000]`` to keep
    the recentred minimum global (the next ``sin(sqrt u) = 1`` peak past
    the boundary sits at ``u ~ 713``, where the penalty is ``4.5e4``
    against a ``713`` gain).  Beyond that the penalty grows quadratically
    and the function linearly, so the margin only widens.

    The argument is scaled by ``50``, mapping the instance box's *full*
    reach — ``|x - x_opt| <= 9`` per coordinate, and the rotation is
    norm-preserving — onto the classic ``[-500, 500]`` domain.  ``100``
    (BBOB f20's factor, for a box that is not also shifted by up to 4)
    would put the penalty term in charge: it fires on 59 % of uniform
    draws at ``d = 5`` and 93 % at ``d = 10``, which turns the family
    into a quadratic bowl with a multimodal speck in the middle.  At
    ``50`` it fires on 1 % / 3 % and the raw Schwefel range is unchanged.
    """
    prob = Schwefel(dims=dim)

    def raw(u: np.ndarray) -> float:
        return float(prob.eval(u)) + _boundary_penalty(u, 500.0)

    def scaled(z: np.ndarray) -> float:
        return raw(z * 50.0)

    # The minimiser lives in the *scaled* coordinate, hence /50.
    return _normalise(scaled, np.full(dim, SCHWEFEL_ARGMIN / 50.0))


def _ellipsoid_base(dim: int) -> Callable[[np.ndarray], float]:
    """BBOB f2 shape: :math:`\\sum_i 10^{6 i/(d-1)} z_i^2`.  Minimum ``0`` at ``0``.

    Ill-conditioned by construction (condition number ``1e6``) and — once
    rotated by the family transform — non-separable.
    """
    exps = np.zeros(dim) if dim < 2 else 6.0 * np.arange(dim) / (dim - 1)
    weights = np.power(10.0, exps)

    def f(z: np.ndarray) -> float:
        return float(np.dot(weights, z * z))

    return f


def _discus_base(dim: int) -> Callable[[np.ndarray], float]:
    """BBOB f11 shape: :math:`10^6 z_0^2 + \\sum_{i>0} z_i^2`.  Minimum ``0`` at ``0``.

    One direction is a million times more sensitive than all the others —
    the opposite anisotropy to the sharp ridge.
    """

    def f(z: np.ndarray) -> float:
        return float(1e6 * z[0] * z[0] + np.dot(z[1:], z[1:]))

    return f


def _sharp_ridge_base(dim: int) -> Callable[[np.ndarray], float]:
    """BBOB f13 shape: :math:`z_0^2 + 100\\sqrt{\\sum_{i>0} z_i^2}`.

    Minimum ``0`` at the origin.  Non-smooth at the ridge: the gradient
    does not vanish as the ridge is approached, so a method that models a
    smooth valley is systematically misled.
    """

    def f(z: np.ndarray) -> float:
        return float(z[0] * z[0] + 100.0 * np.sqrt(float(np.dot(z[1:], z[1:]))))

    return f


# ---------------------------------------------------------------------------
# BBOB bases (Hansen et al. 2009, RR-6829)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BaseContext:
    r"""What a BBOB base needs of its instance.

    ``rng`` is the base's *own* stream (:meth:`Family._base_rng`), so the
    draws a base makes (a second rotation, Gallagher's peaks) do not shift
    the instance stream that ``x_opt``, ``R``, ``f_opt`` and the
    constraints come from.  ``x_opt``, ``rotation`` and ``scaling`` are the
    instance's transform :math:`u = \Lambda R (x - x_{\mathrm{opt}})`; a
    base that places structure in the *box* (Gallagher's peaks) or in
    unrotated coordinates (Lunacek's funnels) maps through it.
    ``rotate=False`` makes every internal rotation the identity, so an
    unrotated instance stays unrotated all the way down.
    """

    rng: np.random.Generator
    x_opt: np.ndarray
    rotation: Optional[np.ndarray] = None
    scaling: Optional[np.ndarray] = None
    half_width: float = 5.0
    rotate: bool = True

    @classmethod
    def default(cls, dim: int) -> "BaseContext":
        """A fixed context (``x_opt = 0``, no instance transform) for using a base on its own."""
        return cls(rng=np.random.default_rng(0), x_opt=np.zeros(dim))

    def to_base(self, x: np.ndarray) -> np.ndarray:
        """Box coordinates -> base coordinates, :math:`\\Lambda R (x - x_{\\mathrm{opt}})`."""
        u = np.asarray(x, dtype=np.float64) - self.x_opt
        if self.rotation is not None:
            u = self.rotation @ u
        if self.scaling is not None:
            u = self.scaling * u
        return u

    def from_base(self, u: np.ndarray) -> np.ndarray:
        """Base coordinates -> the offset :math:`x - x_{\\mathrm{opt}}` (inverse of :meth:`to_base`)."""
        d = np.asarray(u, dtype=np.float64)
        if self.scaling is not None:
            d = d / self.scaling
        if self.rotation is not None:
            d = self.rotation.T @ d
        return d

    def internal_rotation(self, dim: int) -> np.ndarray:
        """BBOB's second rotation :math:`Q`: Haar-random, or ``I`` when ``rotate=False``.

        Always drawn, so ``rotate`` does not shift the base stream.
        """
        q = _haar_rotation(self.rng, dim)
        return q if self.rotate else np.eye(dim)

    def sign_pattern(self) -> np.ndarray:
        """``sign(x_opt)`` with ``0 -> +1``: the BBOB sign vector of f6 and f24."""
        return np.where(self.x_opt >= 0.0, 1.0, -1.0)


def _haar_rotation(rng: np.random.Generator, dim: int) -> np.ndarray:
    """Haar-uniform orthogonal matrix (QR of a Gaussian, with the sign correction)."""
    q, r = np.linalg.qr(rng.standard_normal((dim, dim)))
    return q * np.sign(np.diag(r))


def _lambda_diag(alpha: float, dim: int) -> np.ndarray:
    r"""Diagonal of BBOB's :math:`\Lambda^\alpha`: :math:`\alpha^{\frac12 \frac{i}{D-1}}`, ``i = 0..D-1``."""
    return np.power(float(alpha), 0.5 * np.arange(dim) / max(dim - 1, 1))


def t_osz(v: Union[float, np.ndarray]) -> np.ndarray:
    r"""BBOB's oscillation :math:`T_{\mathrm{osz}}`, element-wise.

    :math:`x \mapsto \mathrm{sign}(x)\exp\bigl(\hat x + 0.049(\sin c_1\hat x
    + \sin c_2\hat x)\bigr)` with :math:`\hat x = \log|x|` (``0`` at ``0``),
    :math:`c_1 = 10, c_2 = 7.9` for :math:`x > 0` and :math:`5.5, 3.1`
    otherwise.  Fixed points ``0``, ``1`` and ``-1``; strictly increasing.
    """
    a = np.atleast_1d(np.asarray(v, dtype=np.float64))
    out = np.zeros_like(a)
    nz = a != 0.0
    xh = np.log(np.abs(a[nz]))
    pos = a[nz] > 0.0
    c1 = np.where(pos, 10.0, 5.5)
    c2 = np.where(pos, 7.9, 3.1)
    out[nz] = np.sign(a[nz]) * np.exp(xh + 0.049 * (np.sin(c1 * xh) + np.sin(c2 * xh)))
    return out


def t_asy(v: np.ndarray, beta: float) -> np.ndarray:
    r"""BBOB's asymmetry :math:`T_{\mathrm{asy}}^\beta`: :math:`x_i^{1 + \beta \frac{i}{D-1}\sqrt{x_i}}` for :math:`x_i > 0`."""
    a = np.asarray(v, dtype=np.float64)
    dim = a.shape[0]
    out = a.copy()
    pos = a > 0.0
    exps = 1.0 + float(beta) * np.arange(dim) / max(dim - 1, 1) * np.sqrt(np.where(pos, a, 0.0))
    out[pos] = np.power(a[pos], exps[pos])
    return out


class _AttractiveSector:
    r"""BBOB f6, attractive sector: :math:`T_{\mathrm{osz}}\bigl(\sum_i (s_i z_i)^2\bigr)^{0.9}`.

    :math:`z = Q \Lambda^{10} u` with :math:`u = R(x - x_{\mathrm{opt}})`
    the instance's own transform, and :math:`s_i = 100` where
    :math:`z_i \, x_{\mathrm{opt},i} > 0`, else ``1``: only a cone of
    roughly :math:`2^{-D}` of the directions around the optimum is cheap,
    so the function is strongly asymmetric around it.  Minimum ``0`` at
    ``0`` (:math:`T_{\mathrm{osz}}` is increasing and fixes ``0``).

    Deviation from BBOB: the sign pattern is ``sign(x_opt)`` with
    ``0 -> +1``, so an unshifted instance (``x_opt = 0``) keeps a sector.
    """

    def __init__(self, dim: int, ctx: Optional[BaseContext] = None) -> None:
        ctx = ctx if ctx is not None else BaseContext.default(dim)
        self.q = ctx.internal_rotation(dim)
        self.lam = _lambda_diag(10.0, dim)
        self.sign = ctx.sign_pattern()

    def __call__(self, u: np.ndarray) -> float:
        z = self.q @ (self.lam * u)
        s = np.where(z * self.sign > 0.0, 100.0, 1.0)
        sz = s * z
        return float(t_osz(float(np.dot(sz, sz)))[0] ** 0.9)


class _StepEllipsoid:
    r"""BBOB f7, step ellipsoid: plateaus.

    :math:`0.1 \max\bigl(|\hat z_1| / 10^4, \sum_i 10^{2\frac{i}{D-1}} z_i^2\bigr)`
    with :math:`\hat z = \Lambda^{10} u`, :math:`\tilde z_i = \lfloor 0.5 +
    \hat z_i \rfloor` for :math:`|\hat z_i| > 0.5` and :math:`\lfloor 0.5 +
    10 \hat z_i \rfloor / 10` otherwise, and :math:`z = Q \tilde z`.  The
    rounding makes the function piecewise constant; the
    :math:`|\hat z_1|` term only keeps the plateau containing the optimum
    from being flat along the first axis.  Minimum ``0`` at ``0`` — and on
    the rest of the slab :math:`\hat z_1 = 0, |\hat z_{i>1}| < 0.05`, so
    the minimiser is not unique; ``x_opt`` is one of them.

    Deviation from BBOB: no :math:`f_{\mathrm{pen}}` term — the instance
    box bounds the search, and the base is non-negative everywhere.
    """

    def __init__(self, dim: int, ctx: Optional[BaseContext] = None) -> None:
        ctx = ctx if ctx is not None else BaseContext.default(dim)
        self.q = ctx.internal_rotation(dim)
        self.lam = _lambda_diag(10.0, dim)
        self.weights = np.power(10.0, 2.0 * np.arange(dim) / max(dim - 1, 1))

    def __call__(self, u: np.ndarray) -> float:
        zh = self.lam * u
        zt = np.where(np.abs(zh) > 0.5, np.floor(0.5 + zh), np.floor(0.5 + 10.0 * zh) / 10.0)
        z = self.q @ zt
        return float(0.1 * max(abs(float(zh[0])) / 1e4, float(np.dot(self.weights, z * z))))


class _BentCigar:
    r"""BBOB f12, bent cigar: :math:`z_1^2 + 10^6 \sum_{i>1} z_i^2`.

    :math:`z = R\, T_{\mathrm{asy}}^{0.5}(u)`, :math:`u = R(x -
    x_{\mathrm{opt}})`, with the instance's :math:`R` on *both* sides as in
    BBOB (``I`` when unrotated).  One direction is a million times softer
    than the others and the asymmetry bends it into a curved ridge.
    Minimum ``0`` at ``0``.
    """

    def __init__(self, dim: int, ctx: Optional[BaseContext] = None) -> None:
        ctx = ctx if ctx is not None else BaseContext.default(dim)
        self.r = ctx.rotation if ctx.rotation is not None else np.eye(dim)

    def __call__(self, u: np.ndarray) -> float:
        z = self.r @ t_asy(u, 0.5)
        return float(z[0] * z[0] + 1e6 * np.dot(z[1:], z[1:]))


class _Gallagher:
    r"""BBOB f21 / f22, Gallagher's Gaussian peaks: weak global structure.

    :math:`T_{\mathrm{osz}}\bigl(10 - \max_i w_i \exp(-\tfrac{1}{2D}
    (u - \tilde y_i)^\top C_i (u - \tilde y_i))\bigr)^2`, where the peaks
    :math:`y_i` live in the *box*: :math:`y_1 = x_{\mathrm{opt}}` and
    :math:`y_{i \ge 2}` uniform in it, mapped to base coordinates
    (:math:`\tilde y_i = \Lambda R (y_i - x_{\mathrm{opt}})`, so the
    shared :math:`R` of BBOB's quadratic form is the instance's).
    :math:`w_1 = 10`, :math:`w_i = 1.1 + 8 \frac{i-2}{n-2}`;
    :math:`C_i = \Lambda^{\alpha_i} / \alpha_i^{1/4}` with its diagonal
    randomly permuted, :math:`\alpha_1` = ``alpha_opt`` and the other
    :math:`\alpha_i` a random permutation of :math:`1000^{2j/(n-2)}`,
    :math:`j = 0..n-2`.  Every other peak is lower than :math:`w_1`, so the
    global minimum is ``0`` at :math:`y_1` alone.

    Knobs: ``n_peaks`` (``n``; BBOB f21 is ``101``, f22 is ``21`` with
    ``alpha_opt = 1e6``) and ``alpha_opt``.

    Deviations from BBOB: the peaks are drawn in the whole instance box
    (f21's ``[-5, 5]``; f22 uses ``[-4.9, 4.9]``), and ``x_opt`` is the
    instance's (f21 draws :math:`y_1` in ``[-4, 4]``, which is exactly the
    instance's default ``opt_margin``); no :math:`f_{\mathrm{pen}}` term.
    """

    def __init__(
        self, dim: int, ctx: Optional[BaseContext] = None, n_peaks: int = 101, alpha_opt: float = 1000.0
    ) -> None:
        ctx = ctx if ctx is not None else BaseContext.default(dim)
        n = int(n_peaks)
        if n < 2:
            raise ValueError(f"gallagher needs n_peaks >= 2, got {n_peaks}")
        rng = ctx.rng
        b = float(ctx.half_width)
        others = rng.uniform(-b, b, size=(n - 1, dim))
        pool = np.power(1000.0, 2.0 * np.arange(n - 1) / max(n - 2, 1))
        alphas = np.concatenate([[float(alpha_opt)], rng.permutation(pool)])
        self.c = np.empty((n, dim))
        for i, a in enumerate(alphas):
            self.c[i] = rng.permutation(_lambda_diag(a, dim)) / a**0.25
        self.peaks_x = np.vstack([ctx.x_opt, others])
        self.peaks = np.array([ctx.to_base(y) for y in self.peaks_x])
        self.peaks[0] = 0.0  # exactly: to_base(x_opt) is 0 anyway
        self.weights = np.concatenate([[10.0], 1.1 + 8.0 * np.arange(n - 1) / max(n - 2, 1)])
        self.dim = dim

    def __call__(self, u: np.ndarray) -> float:
        d = u - self.peaks
        q = np.sum(self.c * d * d, axis=1)
        m = float(np.max(self.weights * np.exp(-q / (2.0 * self.dim))))
        return float(t_osz(10.0 - m)[0] ** 2)


class _LunacekBiRastrigin:
    r"""BBOB f24, Lunacek bi-Rastrigin: a deceptive double funnel.

    .. math::

        \min\Bigl(\sum_i (\hat x_i - \mu_0)^2,\; d D + s \sum_i (\hat x_i - \mu_1)^2\Bigr)
        + 10\Bigl(D - \sum_i \cos 2\pi z_i\Bigr)

    with :math:`\hat x - \mu_0 = 2\,\sigma \otimes (x - x_{\mathrm{opt}})`,
    :math:`\sigma = \mathrm{sign}(x_{\mathrm{opt}})` (BBOB's :math:`\hat x
    = 2\sigma \otimes x` with :math:`x_{\mathrm{opt}} = \frac{\mu_0}{2}\sigma`,
    written relative to the optimum), :math:`z = Q \Lambda^{100} R (\hat x
    - \mu_0)`, :math:`\mu_0 = 2.5`, :math:`\mu_1 = -\sqrt{(\mu_0^2 - d)/s}`.

    The optimum's funnel is narrow; the second funnel, centred
    :math:`(\mu_0 - \mu_1)/2` per coordinate from :math:`x_{\mathrm{opt}}`
    *towards the box centre*, is broader (:math:`s < 1`) and only
    :math:`d D` higher — most of the box drains into the wrong funnel.
    Minimum ``0`` at ``0``: the second funnel is at least :math:`dD > 0`
    and the Rastrigin term is non-negative.

    Knobs (the funnels' depth/width trade-off): ``d`` (``1``, how much
    higher the second funnel's floor lies; ``0 < d < mu_0^2``) and ``s``
    (default BBOB's :math:`1 - 1/(2\sqrt{D + 20} - 8.2)`; smaller is a
    broader second funnel); and ``placement`` of the optimum:

    * ``"bbob"`` (default) — BBOB's own :math:`x_{\mathrm{opt}} =
      \pm\mu_0/2 = \pm 1.25` per coordinate, the signs from the instance's
      draw (:meth:`Family.__init__` places it).  With it the base **is**
      f24 (up to :math:`f_{\mathrm{pen}}`), comparable with the literature.
    * ``"box"`` — the instance's uniform :math:`x_{\mathrm{opt}}` in
      ``[-4, 4]``.  **Harder than f24**: the optimum can sit far out and
      the second funnel still lies :math:`(\mu_0 - \mu_1)/2` towards the
      centre (inside the box for every draw).

    No :math:`10^4 f_{\mathrm{pen}}` term (the box bounds the search).
    """

    MU0: float = 2.5
    PLACEMENTS: Tuple[str, ...] = ("bbob", "box")

    def __init__(
        self,
        dim: int,
        ctx: Optional[BaseContext] = None,
        d: float = 1.0,
        s: Optional[float] = None,
        placement: str = "bbob",
    ) -> None:
        ctx = ctx if ctx is not None else BaseContext.default(dim)
        if placement not in self.PLACEMENTS:
            raise ValueError(f"lunacek_bi_rastrigin placement must be one of {self.PLACEMENTS}, got {placement!r}")
        self.placement = placement
        self.d = float(d)
        self.s = float(s) if s is not None else 1.0 - 1.0 / (2.0 * np.sqrt(dim + 20.0) - 8.2)
        if not 0.0 < self.d < self.MU0**2:
            raise ValueError(f"lunacek_bi_rastrigin needs 0 < d < {self.MU0**2}, got {d}")
        if self.s <= 0.0:
            raise ValueError(f"lunacek_bi_rastrigin needs s > 0, got {s}")
        self.mu1 = -float(np.sqrt((self.MU0**2 - self.d) / self.s))
        self.q = ctx.internal_rotation(dim)
        self.r = ctx.rotation if ctx.rotation is not None else np.eye(dim)
        self.lam = _lambda_diag(100.0, dim)
        self.sign = ctx.sign_pattern()
        self.ctx = ctx
        self.dim = dim

    @classmethod
    def bbob_x_opt(cls, x_draw: np.ndarray) -> np.ndarray:
        """BBOB's optimum :math:`\\frac{\\mu_0}{2}\\,\\mathrm{sign}` with the signs of a uniform draw."""
        return 0.5 * cls.MU0 * np.where(x_draw >= 0.0, 1.0, -1.0)

    def __call__(self, u: np.ndarray) -> float:
        w = 2.0 * self.sign * self.ctx.from_base(u)  # \hat x - mu0
        f1 = float(np.dot(w, w))
        v = w + (self.MU0 - self.mu1)  # \hat x - mu1
        f2 = self.d * self.dim + self.s * float(np.dot(v, v))
        z = self.q @ (self.lam * (self.r @ w))
        ras = 10.0 * (self.dim - float(np.sum(np.cos(2.0 * np.pi * z))))
        return min(f1, f2) + ras


#: ``base name -> factory``.  Every entry has a **global** minimum of ``0``
#: at the origin over the whole of :math:`\mathbb{R}^d` (see the module
#: note on Schwefel's boundary penalty).
#:
#: Deliberately *not* offered as bases, having checked ``classic.py``:
#: ``StyblinskiTang`` (minimiser ``-2.903534...`` is a rounded root, so
#: ``f_opt`` would carry a ~1e-9 error), ``Himmelblau`` /
#: ``GoldsteinPrice`` / ``Branin`` (fixed 2-D, no ``dims``),
#: ``DixonPrice`` / ``Zakharov`` / ``Salomon`` / ``Trigonometric``
#: (minimiser documented only implicitly or drifting with ``dim``), and
#: everything whose optimum ``classic.py`` does not state at all.
BASE_FUNCTIONS: Dict[str, BaseFactory] = {
    "sphere": _sphere_base,
    "rosenbrock": _rosenbrock_base,
    "rastrigin": _rastrigin_base,
    "ackley": _ackley_base,
    "griewank": _griewank_base,
    "schwefel": _schwefel_base,
    "ellipsoid": _ellipsoid_base,
    "discus": _discus_base,
    "sharp_ridge": _sharp_ridge_base,
    "attractive_sector": _AttractiveSector,
    "step_ellipsoid": _StepEllipsoid,
    "bent_cigar": _BentCigar,
    "gallagher": _Gallagher,
    "lunacek_bi_rastrigin": _LunacekBiRastrigin,
}

#: ``base -> knob names`` for the bases built with a :class:`BaseContext`
#: (``factory(dim, ctx, **knobs)``); every other base is ``factory(dim)``
#: and takes no knobs.  A knob reaches the base through
#: ``Family(..., base_params={...})``.
CONTEXT_BASES: Dict[str, Tuple[str, ...]] = {
    "attractive_sector": (),
    "step_ellipsoid": (),
    "bent_cigar": (),
    "gallagher": ("n_peaks", "alpha_opt"),
    "lunacek_bi_rastrigin": ("d", "s", "placement"),
}

#: Constraint constructions understood by :class:`Family`.
CONSTRAINT_KINDS: Tuple[str, ...] = ("linear", "ball", "mixed")


# ---------------------------------------------------------------------------
# Failure regions
# ---------------------------------------------------------------------------

#: Failure-region shapes understood by :class:`FailureRegion`.
FAILURE_SHAPES: Tuple[str, ...] = ("halfspace", "ball", "boxes")

#: What a call inside the region does: raise :class:`EvaluationCrashed` or
#: :class:`EvaluationTimedOut`.
FAILURE_MODES: Tuple[str, ...] = ("crash", "timeout")

#: Monte-Carlo points used to calibrate a ball or box region to its share of
#: the box volume (standard error of the share ~0.2 % at a 10 % share).
FAILURE_MC_POINTS: int = 20000

#: Redraws of a ball region that contains ``x_opt`` before the deterministic
#: fallback placement (:meth:`_FailureGeometry._fallback`).
FAILURE_MAX_TRIES: int = 200

#: The same for a box set (each draw is a bisection, so fewer).
FAILURE_MAX_TRIES_BOXES: int = 20


@dataclass(frozen=True)
class FailureRegion:
    r"""A region of the box where an evaluation returns no value.

    Real simulators fail in *regions* — a solver diverges above some
    pressure, a mesh breaks below some thickness
    (``planning/DESIGN_roadmap_2026-09-26.md`` §4 D).  A
    :class:`Family` with a failure region raises from :meth:`Family.eval`
    for every point inside it, before the objective is computed:

    * ``mode="crash"`` raises :class:`~panobbgo.lib.lib.EvaluationCrashed`.
      The evaluation paths book it like any raising objective: no result,
      charged against ``max_eval``, published as ``failed_evaluations``.
    * ``mode="timeout"`` raises :class:`~panobbgo.lib.lib.EvaluationTimedOut`,
      which :meth:`Problem.__call__ <panobbgo.lib.lib.Problem.__call__>`
      turns into the ``NaN`` placeholder of a real ``evaluation.timeout``
      (``Result.timed_out``) — deterministically and without sleeping.

    Shapes (``share`` is the fraction of the box *volume* inside the
    region, ``0 < share <= 0.5``):

    * ``"halfspace"`` — :math:`x_i > t` or :math:`x_i < t` for one random
      axis :math:`i`.  With ``boundary_gap=None`` the threshold gives the
      share exactly.  With ``boundary_gap=g`` the boundary is put at
      distance :math:`g B` from :math:`x_{\mathrm{opt}}` instead (``0``:
      the optimum lies *on* the boundary — engineering optima often sit
      right at a stability limit), on the axis and side whose resulting
      share is closest to ``share``; the realised share is then
      :attr:`Family.failure_share`.
    * ``"ball"`` — a Euclidean ball with a uniform random centre, its
      radius calibrated so that ``share`` of the box lies inside it (the
      ball may stick out of the box).
    * ``"boxes"`` — ``n_boxes`` axis-aligned boxes with random centres and
      aspect ratios, scaled together so that their *union* covers
      ``share`` of the box.

    In high dimension the shapes change character: a box of a fixed
    volume share is a *slab* (at ``d = 10`` a 7 % box is ~77 % of the
    width per axis, so it cuts through nearly the whole range of every
    coordinate), and a ball holding ``share`` of the cube is mostly a
    *cap* sticking out of it, its radius comparable to the half-width.

    The optimum is always outside (the inequalities are strict, so a
    half-space boundary through ``x_opt`` leaves it outside); a ball or a
    box set that contains it is redrawn, and where no draw avoids it (a
    large ball around a central ``x_opt`` at ``d = 10``) a deterministic
    fallback places it (:meth:`_FailureGeometry._fallback`), with a
    realised share below ``share`` — :attr:`Family.failure_share`, which
    is measured on a Monte-Carlo sample independent of the calibration.  ``x_opt`` therefore stays the
    minimiser of every point that can be evaluated, and ``f_opt`` the
    AOCC target.  All draws come from a stream of their own
    (:meth:`Family._failure_rng`), so an instance with a region has the
    same ``x_opt``, ``R``, ``f_opt`` and constraints as without one: the
    two form a pair.
    """

    shape: str
    share: float = 0.1
    mode: str = "crash"
    n_boxes: int = 3
    boundary_gap: Optional[float] = None

    def __post_init__(self) -> None:
        if self.shape not in FAILURE_SHAPES:
            raise ValueError(f"unknown failure shape {self.shape!r}; known: {list(FAILURE_SHAPES)}")
        if self.mode not in FAILURE_MODES:
            raise ValueError(f"unknown failure mode {self.mode!r}; known: {list(FAILURE_MODES)}")
        if not 0.0 < float(self.share) <= 0.5:
            raise ValueError(f"failure share must be in (0, 0.5], got {self.share}")
        if int(self.n_boxes) < 1:
            raise ValueError(f"n_boxes must be >= 1, got {self.n_boxes}")
        if self.boundary_gap is not None and (self.shape != "halfspace" or float(self.boundary_gap) < 0.0):
            raise ValueError("boundary_gap is a non-negative fraction of the half-width, for shape='halfspace' only")

    def tag(self) -> str:
        """Short label part: ``fhs_crash``, ``fball_tmo``, ``fbox_crash``."""
        shape = {"halfspace": "hs", "ball": "ball", "boxes": "box"}[self.shape]
        mode = {"crash": "crash", "timeout": "tmo"}[self.mode]
        return f"f{shape}_{mode}"


class _FailureGeometry:
    """The drawn region of one instance; plain arrays, so it pickles."""

    def __init__(self, region: FailureRegion, x_opt: np.ndarray, half_width: float, rng: np.random.Generator) -> None:
        self.shape = region.shape
        self.mode = region.mode
        b = float(half_width)
        dim = x_opt.shape[0]
        target = float(region.share)
        # halfspace: sign * (x[axis] - t) > 0
        self.axis = 0
        self.sign = 1.0
        self.t = 0.0
        # ball
        self.centre = np.zeros(dim)
        self.radius = 0.0
        # boxes
        self.lo = np.zeros((0, dim))
        self.hi = np.zeros((0, dim))
        #: ``True`` when no random draw kept ``x_opt`` outside and the
        #: deterministic fallback placed the region (the realised share is
        #: then below ``share``; see :attr:`share`).  Never for a half-space.
        self.fallback = False

        if self.shape == "halfspace":
            axis = int(rng.integers(dim))
            sign = float(rng.choice([-1.0, 1.0]))
            order = rng.permutation(2 * dim)  # candidate order for the boundary_gap placement
            if region.boundary_gap is None:
                t = sign * (b - 2.0 * b * target)
                if sign * (x_opt[axis] - t) > 0.0:  # x_opt inside: the mirrored half-space is disjoint
                    sign, t = -sign, -t
                self.axis, self.sign, self.t = axis, sign, t
                self.share = target
            else:
                gap = float(region.boundary_gap) * b
                best: Optional[Tuple[float, int, float, float]] = None
                for c in order:
                    ax, sg = int(c) // 2, (1.0 if int(c) % 2 else -1.0)
                    t = float(x_opt[ax]) + sg * gap
                    share = (b - sg * t) / (2.0 * b)
                    if share <= 0.0:
                        continue
                    if best is None or abs(share - target) < best[0]:
                        best = (abs(share - target), ax, sg, t)
                if best is None:
                    raise ValueError(
                        f"boundary_gap={region.boundary_gap} puts every half-space boundary outside the box "
                        f"(share={target}, dim={dim})"
                    )
                _, self.axis, self.sign, self.t = best
                self.share = (b - self.sign * self.t) / (2.0 * b)
            return

        sample = rng.uniform(-b, b, size=(FAILURE_MC_POINTS, dim))
        # A box draw is a 30-step bisection, a ball draw one quantile: the
        # boxes give up sooner.
        tries = FAILURE_MAX_TRIES if self.shape == "ball" else FAILURE_MAX_TRIES_BOXES
        for _ in range(tries):
            if self.shape == "ball":
                self.centre = rng.uniform(-b, b, size=dim)
                dist = np.linalg.norm(sample - self.centre, axis=1)
                self.radius = float(np.quantile(dist, target))
            else:
                self._draw_boxes(region, sample, b, target, rng)
            if not self.contains(x_opt):
                break
        else:
            self._fallback(x_opt, sample, b, target)
        if self.contains(x_opt):  # pragma: no cover - the fallback excludes x_opt by construction
            raise ValueError(f"could not place a {self.shape} failure region outside x_opt (share={target}, dim={dim})")
        # Measured on points the calibration never saw: the calibration
        # sample's own share is the target by construction.
        check = rng.uniform(-b, b, size=(FAILURE_MC_POINTS, dim))
        self.share = float(np.mean(self.contains_many(check)))

    def _fallback(self, x_opt: np.ndarray, sample: np.ndarray, b: float, target: float) -> None:
        """Deterministic placement when every random draw contained ``x_opt``.

        Happens for large shares where the region cannot avoid a central
        optimum (a ball of 20 % of the box at ``d = 10`` around an unshifted
        ``x_opt = 0``).  Ball: the centre is searched along the ray from
        ``x_opt`` to the corner *away* from it (up to one half-width past
        the corner), the radius capped so ``x_opt`` stays outside, and the
        centre whose capped ball comes closest to the target share wins.  Boxes: the
        last draw, each box containing ``x_opt`` cut at ``x_opt`` along the
        axis where that removes the thinnest slab.  Either way the realised
        share (:attr:`share`) is at most the target.
        """
        self.fallback = True
        if self.shape == "ball":
            corner = -b * np.where(x_opt >= 0.0, 1.0, -1.0)
            ray = corner - x_opt
            best: Optional[Tuple[float, np.ndarray, float]] = None
            for t in np.linspace(0.25, 1.0 + b / max(float(np.linalg.norm(ray)), 1e-12), 16):
                centre = x_opt + t * ray
                dist = np.linalg.norm(sample - centre, axis=1)
                cap = float(np.linalg.norm(x_opt - centre))
                radius = min(float(np.quantile(dist, target)), cap)
                share = float(np.mean(dist < radius))
                if best is None or abs(share - target) < abs(best[0] - target):
                    best = (share, centre, radius)
            assert best is not None
            _, self.centre, self.radius = best
            return
        for j in range(self.lo.shape[0]):
            lo, hi = self.lo[j], self.hi[j]
            if not (np.all(x_opt > lo) and np.all(x_opt < hi)):
                continue
            below, above = x_opt - lo, hi - x_opt
            ax = int(np.argmin(np.minimum(below, above)))
            if below[ax] <= above[ax]:
                lo[ax] = x_opt[ax]  # strict inequality: x_opt on the face is outside
            else:
                hi[ax] = x_opt[ax]

    def _draw_boxes(
        self, region: FailureRegion, sample: np.ndarray, b: float, target: float, rng: np.random.Generator
    ) -> None:
        """``n_boxes`` random boxes, scaled together until their union covers ``target`` of ``sample``."""
        k = int(region.n_boxes)
        dim = sample.shape[1]
        centres = rng.uniform(-b, b, size=(k, dim))
        log_aspect = rng.uniform(-0.5, 0.5, size=(k, dim))
        log_aspect -= log_aspect.mean(axis=1, keepdims=True)  # volume-neutral aspect ratios
        half = b * (target / k) ** (1.0 / dim) * np.exp(log_aspect)
        lo_s, hi_s = 0.0, 2.0 * b / float(half.min())  # at hi_s every box covers the whole box
        for _ in range(30):  # 2**-30 of the scale range: far below the Monte-Carlo resolution
            mid = 0.5 * (lo_s + hi_s)
            self.lo, self.hi = centres - mid * half, centres + mid * half
            if np.mean(self.contains_many(sample)) < target:
                lo_s = mid
            else:
                hi_s = mid
        self.lo, self.hi = centres - hi_s * half, centres + hi_s * half

    def contains_many(self, xs: np.ndarray) -> np.ndarray:
        """Row-wise :meth:`contains`."""
        xs = np.atleast_2d(xs)
        if self.shape == "halfspace":
            return self.sign * (xs[:, self.axis] - self.t) > 0.0
        if self.shape == "ball":
            return np.linalg.norm(xs - self.centre, axis=1) < self.radius
        inside = (xs[:, None, :] > self.lo[None]) & (xs[:, None, :] < self.hi[None])
        return inside.all(axis=2).any(axis=1)

    def contains(self, x: np.ndarray) -> bool:
        """``True`` iff ``x`` (box coordinates) is inside the region; the boundary is outside."""
        return bool(self.contains_many(np.asarray(x, dtype=np.float64))[0])


# ---------------------------------------------------------------------------
# One instance
# ---------------------------------------------------------------------------


class Family(Problem):
    r"""One problem *instance* drawn from a parametrised family.

    ``Family("rastrigin", dim=5, seed=7)`` is a ready-to-optimise
    :class:`~panobbgo.lib.lib.Problem` with

    .. math::

        f(x) = f_{\mathrm{base}}\bigl(\Lambda R (x - x_{\mathrm{opt}})\bigr)
               + f_{\mathrm{opt}},

    and it records :attr:`x_opt` and :attr:`f_opt`, so AOCC is computable.
    Everything random (``x_opt``, ``R``, ``f_opt``, the constraints) is
    drawn from ``numpy.random.default_rng(seed)`` and nothing else, so two
    instances with the same arguments are identical.

    Parameters
    ----------
    base
        Key into :data:`BASE_FUNCTIONS`.
    dim
        Dimension, ``>= 2``.
    seed
        Instance seed.  *Not* the optimiser's seed — this only draws the
        problem.
    shift
        Draw ``x_opt`` inside the box (``True``) or pin it at the centre.
    rotate
        Apply a Haar-random orthogonal ``R`` (``True``) or ``R = I``.
    condition
        Ill-conditioning :math:`\kappa`: the diagonal scaling is
        :math:`\Lambda_{ii} = \kappa^{i/(d-1)}`, i.e. ``10**(alpha*i/(d-1))``
        for :math:`\kappa = 10^\alpha`.  ``1.0`` disables it.  Applied
        *after* the rotation, so it is not itself rotated away.  The
        classic bases (including the BBOB-shaped ``ellipsoid``, ``discus``
        and ``sharp_ridge``) get it *on top of* their own shape; the BBOB
        bases of :data:`CONTEXT_BASES` carry BBOB's own conditioning and
        refuse any ``condition != 1``.
    box_half_width
        The box is ``[-box_half_width, box_half_width]^dim``.
    opt_margin
        ``x_opt`` is drawn uniformly in
        ``[-(1 - opt_margin) * B, (1 - opt_margin) * B]^dim``.
    f_opt_range
        ``f_opt`` is drawn uniformly in ``[-f_opt_range, f_opt_range]``
        and rounded to 4 decimals.  ``0.0`` pins it to zero.
    n_constraints
        ``k >= 0`` inequality constraints ``g_j(x) <= 0``.  ``0`` gives an
        unconstrained instance and ``eval_constraints`` returns ``None``,
        exactly like an unconstrained ``classic.py`` problem.
    constraint_kind
        ``"linear"``, ``"ball"``, or ``"mixed"`` (alternating).
    slack
        Slack of the *inactive* constraints, as a fraction of
        ``box_half_width``.  The first constraint always has zero slack.
    ball_radius
        Radius of the ball constraints, as a fraction of
        ``box_half_width``.
    family
        Label used in reports and in the AOCC record's ``problem_kind``.
        Defaults to ``base`` (plus a constraint tag when ``k > 0`` and a
        failure tag with a ``failure`` region).
    instance
        Instance index, carried through to the record.  Purely a label.
    base_params
        Knobs of a BBOB base (:data:`CONTEXT_BASES`), e.g.
        ``{"n_peaks": 21}`` for ``gallagher``.  Other bases take none.
    failure
        A :class:`FailureRegion`, or ``None`` (every point evaluates).

    The constraint construction
    ---------------------------

    Constraints are built *around* ``x_opt``, which is what makes them
    both non-trivial and compatible with a known optimum:

    * **linear** — :math:`g_j(x) = \bigl(a_j \cdot (x - x_{\mathrm{opt}})
      - b_j\bigr) / B` with :math:`a_j` a random unit vector and
      :math:`b_j \ge 0` the slack.  Then
      :math:`g_j(x_{\mathrm{opt}}) = -b_j / B \le 0`.
    * **ball** — :math:`g_j(x) = (\lVert x - c_j \rVert^2 - r_j^2) /
      \max(r_j^2, 1)` with the centre :math:`c_j = x_{\mathrm{opt}} +
      \rho\, u_j` at distance :math:`\rho` in a random direction
      :math:`u_j`, and :math:`r_j = \rho + b_j`.  The feasible set is the
      *inside* of the ball, and :math:`x_{\mathrm{opt}}` sits on its
      surface at zero slack.

    Both are divided by their natural scale so the violations come out
    ``O(1)``, the way ``classic.py``'s ``PressureVessel`` normalises its
    ``g3``/``g4`` — a penalty term is only meaningful against a
    comparable objective.

    Two properties hold by construction and are asserted in the tests:

    1.  **The optimum is feasible.**  Every :math:`g_j(x_{\mathrm{opt}})
        \le 0`.
    2.  **The first constraint is active there** (:math:`b_0 = 0`, so
        :math:`g_0(x_{\mathrm{opt}}) = 0` exactly), and ``Result.cv``
        counts only *strictly positive* entries, so an exactly-active
        constraint is feasible.  Without this the constraints would be
        slack at the optimum and would do nothing at all.

    Consequently the constrained optimum **equals** the unconstrained
    one: :math:`x_{\mathrm{opt}}` is the global minimiser of :math:`f`
    over the whole box, so it is a fortiori the minimiser over any
    feasible subset that contains it.  ``f_opt`` therefore stays the AOCC
    target for the constrained instances too.
    """

    def __init__(
        self,
        base: str,
        dim: int,
        seed: int,
        *,
        shift: bool = True,
        rotate: bool = True,
        condition: float = 1.0,
        box_half_width: float = 5.0,
        opt_margin: float = 0.2,
        f_opt_range: float = 100.0,
        n_constraints: int = 0,
        constraint_kind: str = "linear",
        slack: float = 0.2,
        ball_radius: float = 1.0,
        family: Optional[str] = None,
        instance: int = 0,
        base_params: Optional[Dict[str, Any]] = None,
        failure: Optional[FailureRegion] = None,
    ) -> None:
        if base not in BASE_FUNCTIONS:
            raise ValueError(f"unknown base function {base!r}; known: {sorted(BASE_FUNCTIONS)}")
        if dim < 2:
            raise ValueError(f"dim must be >= 2, got {dim}")
        if constraint_kind not in CONSTRAINT_KINDS:
            raise ValueError(f"unknown constraint_kind {constraint_kind!r}; known: {list(CONSTRAINT_KINDS)}")
        if n_constraints < 0:
            raise ValueError(f"n_constraints must be >= 0, got {n_constraints}")
        params = dict(base_params or {})
        unknown = sorted(set(params) - set(CONTEXT_BASES.get(base, ())))
        if unknown:
            raise ValueError(f"base {base!r} takes no knob(s) {unknown}; it takes {list(CONTEXT_BASES.get(base, ()))}")
        if base in CONTEXT_BASES and float(condition) != 1.0:
            # A BBOB base carries its own conditioning (Lambda^alpha) at a
            # defined place in its transform; an instance scaling on top
            # would condition some of its terms and not others (Lunacek's
            # funnels vs its Rastrigin term).
            raise ValueError(f"base {base!r} has its own BBOB conditioning; condition must be 1.0, got {condition}")

        rng = np.random.default_rng(int(seed))
        b = float(box_half_width)
        Problem.__init__(self, [(-b, b)] * dim)

        self.base: str = base
        self.instance: int = int(instance)
        self.seed: int = int(seed)
        self.condition: float = float(condition)
        self.n_constraints: int = int(n_constraints)
        self.constraint_kind: str = constraint_kind
        self.family: str = (
            family if family is not None else _family_label(base, n_constraints, constraint_kind, failure)
        )
        self.base_params: Dict[str, Any] = params
        self.failure: Optional[FailureRegion] = failure
        self._half_width: float = b

        # -- the affine transform -------------------------------------------
        #
        # Every random quantity is *drawn* unconditionally and only then
        # used or discarded, so ``shift`` and ``rotate`` do not shift the
        # RNG stream: instances that differ only in a toggle share their
        # ``x_opt``, ``f_opt`` and constraints, and a test can attribute a
        # difference to the toggle rather than to a re-draw.
        reach = (1.0 - float(opt_margin)) * b
        x_opt_draw = rng.uniform(-reach, reach, size=dim)
        gaussian = rng.standard_normal((dim, dim))
        q, r = np.linalg.qr(gaussian)
        # Sign correction: without it QR is not Haar-uniform on O(d).
        rotation = q * np.sign(np.diag(r))

        x_opt = x_opt_draw if shift else np.zeros(dim)
        if shift and base == "lunacek_bi_rastrigin" and params.get("placement", "bbob") == "bbob":
            # BBOB f24's own optimum, +-mu0/2, with the signs of the draw
            # (no extra draw: the instance stream is unchanged).
            x_opt = _LunacekBiRastrigin.bbob_x_opt(x_opt_draw)
        self._x_opt: np.ndarray = x_opt
        self._rotation: Optional[np.ndarray] = rotation if rotate else None

        if self.condition != 1.0:
            exps = np.zeros(dim) if dim < 2 else np.arange(dim) / (dim - 1)
            self._scaling: Optional[np.ndarray] = np.power(self.condition, exps)
        else:
            self._scaling = None

        self._f_opt: float = round(float(rng.uniform(-f_opt_range, f_opt_range)), 4) if f_opt_range else 0.0

        # -- the constraints -------------------------------------------------
        # ``kinds[0]`` gets zero slack, so it is exactly active at x_opt.
        self._con_kinds: List[str] = []
        self._con_a: List[np.ndarray] = []  # linear normals / ball centres
        self._con_b: List[float] = []  # linear offsets / ball *squared* radii
        for j in range(self.n_constraints):
            kind = constraint_kind if constraint_kind != "mixed" else ("linear", "ball")[j % 2]
            slack_j = 0.0 if j == 0 else float(rng.uniform(0.0, float(slack) * b))
            direction = rng.standard_normal(dim)
            direction /= np.linalg.norm(direction)
            if kind == "linear":
                self._con_a.append(direction)
                self._con_b.append(slack_j)
            else:
                rho = float(ball_radius) * b
                centre = x_opt + rho * direction
                # The squared radius is the *evaluated* squared distance
                # from the optimum, computed by the same expression
                # ``eval_constraints`` uses.  ``rho**2`` would be off by
                # ~3e-16 (``direction`` is a unit vector only to float
                # precision), and ``Result.cv`` counts any strictly
                # positive entry as a violation — the optimum itself would
                # come out infeasible.
                offset = x_opt - centre
                r_sq = float(np.dot(offset, offset))
                if slack_j:
                    r_sq = (np.sqrt(r_sq) + slack_j) ** 2
                self._con_a.append(centre)
                self._con_b.append(r_sq)
            self._con_kinds.append(kind)

        # -- the base and the failure region ----------------------------------
        # Both draw from streams of their own (``_base_rng`` /
        # ``_failure_rng``), after everything above: the instance stream is
        # the same for every base and with or without a failure region.
        self._rotate: bool = bool(rotate)
        self._base_fn: Callable[[np.ndarray], float] = self._build_base()
        self._failure_geom: Optional[_FailureGeometry] = (
            None if failure is None else _FailureGeometry(failure, x_opt, b, self._failure_rng())
        )

    def _base_rng(self) -> np.random.Generator:
        """The stream a BBOB base draws its own structure from (second rotation, peaks).

        A spawned child of the instance seed (``SeedSequence(seed,
        spawn_key=(1,))``): unlike ``default_rng([seed, 1])`` it cannot
        coincide with the plain stream of another seed (``[s, 1]`` is the
        entropy of ``s + 2**32``).
        """
        return np.random.default_rng(np.random.SeedSequence(self.seed, spawn_key=(1,)))

    def _failure_rng(self) -> np.random.Generator:
        """The stream the failure region is drawn from (spawn key ``2``, see :meth:`_base_rng`)."""
        return np.random.default_rng(np.random.SeedSequence(self.seed, spawn_key=(2,)))

    def _build_base(self) -> Callable[[np.ndarray], float]:
        """The base callable; deterministic in the instance data, so a pickle can rebuild it."""
        factory = BASE_FUNCTIONS[self.base]
        if self.base not in CONTEXT_BASES:
            return factory(self.dim)
        ctx = BaseContext(
            rng=self._base_rng(),
            x_opt=self._x_opt,
            rotation=self._rotation,
            scaling=self._scaling,
            half_width=self._half_width,
            rotate=getattr(self, "_rotate", self._rotation is not None),
        )
        return factory(self.dim, ctx, **getattr(self, "base_params", {}))

    # -- what makes the instance scoreable ----------------------------------

    @property
    def x_opt(self) -> np.ndarray:
        """The global minimiser, a copy (the stored vector stays private)."""
        return self._x_opt.copy()

    @property
    def f_opt(self) -> float:
        """``f(x_opt)`` — the AOCC precision target."""
        return self._f_opt

    @property
    def rotation(self) -> Optional[np.ndarray]:
        """The orthogonal matrix ``R``, or ``None`` when ``rotate=False``."""
        return None if self._rotation is None else self._rotation.copy()

    @property
    def scaling(self) -> Optional[np.ndarray]:
        """The diagonal of ``Lambda``, or ``None`` when ``condition == 1``."""
        return None if self._scaling is None else self._scaling.copy()

    @property
    def name(self) -> str:
        """``<family>_d<dim>_i<instance>`` — the report / record label."""
        return f"{self.family}_d{self.dim}_i{self.instance}"

    @property
    def failure_share(self) -> float:
        """Realised share of the box volume inside the failure region (``0.0`` without one).

        Exact for a half-space; a Monte-Carlo estimate (``FAILURE_MC_POINTS``
        points) for a ball or boxes.
        """
        geom = getattr(self, "_failure_geom", None)
        return 0.0 if geom is None else float(geom.share)

    def failure_at(self, x: np.ndarray) -> Optional[str]:
        """``"crash"`` / ``"timeout"`` if evaluating ``x`` fails, else ``None``.

        The same test :meth:`eval` applies, without evaluating anything: a
        simulator or a duration model can ask it beforehand.  ``x`` is in
        :meth:`eval`'s coordinates.
        """
        geom = getattr(self, "_failure_geom", None)
        if geom is None or not geom.contains(np.asarray(x, dtype=np.float64)):
            return None
        return geom.mode

    # -- evaluation ----------------------------------------------------------

    def eval(self, x: np.ndarray) -> float:
        """``f(x)``; raises inside a failure region (see :class:`FailureRegion`)."""
        mode = self.failure_at(x)
        if mode == "crash":
            raise EvaluationCrashed(f"{self.name}: evaluation crashed (failure region)")
        if mode == "timeout":
            raise EvaluationTimedOut(f"{self.name}: evaluation timed out (failure region)")
        z = np.asarray(x, dtype=np.float64) - self._x_opt
        if self._rotation is not None:
            z = self._rotation @ z
        if self._scaling is not None:
            z = self._scaling * z
        return self._base_fn(z) + self._f_opt

    def eval_constraints(self, x: np.ndarray) -> Optional[np.ndarray]:
        """Violation vector ``g(x)``; ``<= 0`` is feasible, as in ``classic.py``.

        Returns ``None`` for an unconstrained instance so the instance is
        indistinguishable from an unconstrained ``classic.py`` problem
        (``Result.cv_vec is None`` -> ``Result.cv == 0``).
        """
        if not self._con_kinds:
            return None
        xa = np.asarray(x, dtype=np.float64)
        out = np.empty(len(self._con_kinds), dtype=np.float64)
        for j, kind in enumerate(self._con_kinds):
            if kind == "linear":
                out[j] = (float(np.dot(self._con_a[j], xa - self._x_opt)) - self._con_b[j]) / self._half_width
            else:
                d = xa - self._con_a[j]
                r_sq = self._con_b[j]
                out[j] = (float(np.dot(d, d)) - r_sq) / max(r_sq, 1.0)
        return out

    def __repr__(self) -> str:
        tail = "" if not self.n_constraints else f", {self.n_constraints} {self.constraint_kind} constraint(s)"
        failure = getattr(self, "failure", None)
        if failure is not None:
            tail += f", {failure.mode} {failure.shape} failure region ({self.failure_share:.0%})"
        return f"Family '{self.name}': base={self.base}, dim={self.dim}, f_opt={self._f_opt:g}{tail}"

    # The base function is a closure (not picklable); it is a pure function
    # of the instance data (``base``, ``dim``, and for a BBOB base the seed,
    # the transform and ``base_params``), so a pickle carries the drawn
    # instance data and the receiver rebuilds the closure.  Needed to run a
    # family battery in worker processes (``jobs=N``).
    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state.pop("_base_fn", None)
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._base_fn = self._build_base()


# ---------------------------------------------------------------------------
# Instance generation
# ---------------------------------------------------------------------------


def _family_label(base: str, n_constraints: int, constraint_kind: str, failure: Optional[FailureRegion] = None) -> str:
    """``rastrigin`` / ``rastrigin_lin`` / ``rastrigin_ball`` / ``rastrigin_mix``, plus ``_fhs_crash`` etc."""
    label = base
    if n_constraints > 0:
        label += "_" + {"linear": "lin", "ball": "ball", "mixed": "mix"}[constraint_kind]
    if failure is not None:
        label += "_" + failure.tag()
    return label


def _instance_seed(base_seed: int, label: str, dim: int, index: int) -> int:
    """SHA-256 instance seed — the same scheme the harnesses use for run seeds.

    Deterministic in ``(base_seed, label, dim, index)`` and nothing else,
    so adding a dimension or a family to a battery does not re-draw the
    instances that were already in it.
    """
    payload = f"{base_seed}|{label}|{dim}|{index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


@dataclass(frozen=True)
class FamilyConfig:
    """One family in a battery: a base function plus its instance knobs.

    ``n_constraints`` may be a single ``int`` or a sequence, in which case
    it is *cycled over the instance index* — that is how a battery gets
    ``k = 1, 2, 3`` constraints on consecutive instances of the same
    family without three near-duplicate configs.

    ``base_params`` are the knobs of a BBOB base (:data:`CONTEXT_BASES`)
    and ``failure`` an optional :class:`FailureRegion`; the failure region
    tags the default label, the knobs do not: a sweep over a knob needs
    an explicit ``label`` per config (two equal labels in one battery are
    refused).
    """

    base: str
    label: Optional[str] = None
    condition: float = 1.0
    rotate: bool = True
    shift: bool = True
    n_constraints: Union[int, Sequence[int]] = 0
    constraint_kind: str = "linear"
    extra: Dict[str, Any] = field(default_factory=dict)
    base_params: Dict[str, Any] = field(default_factory=dict)
    failure: Optional[FailureRegion] = None

    def k_for(self, index: int) -> int:
        if isinstance(self.n_constraints, int):
            return self.n_constraints
        ks = list(self.n_constraints)
        return int(ks[index % len(ks)]) if ks else 0

    def name(self) -> str:
        if self.label is not None:
            return self.label
        # Any instance carrying a constraint tags the whole family, so a
        # cycled ``(0, 1, 2)`` cannot collide with the unconstrained family
        # of the same base.
        if isinstance(self.n_constraints, int):
            k = self.n_constraints
        else:
            k = max((int(v) for v in self.n_constraints), default=0)
        return _family_label(self.base, k, self.constraint_kind, self.failure)


FamilyLike = Union[str, FamilyConfig]


def make_family_instances(
    families: Sequence[FamilyLike],
    dims: Sequence[int],
    n_instances: int = 3,
    seed: int = 42,
) -> List[Tuple[str, Family]]:
    """Build ``(name, problem)`` pairs, deterministically.

    The product ``families x dims x range(n_instances)`` in that order.
    Each instance's own seed is
    ``_instance_seed(seed, family_label, dim, index)``, so the same
    ``seed`` reproduces the same battery bit-for-bit, and a battery that
    gains a dimension keeps the instances it already had.

    Parameters
    ----------
    families
        Base-function names (plain unconstrained instances) or
        :class:`FamilyConfig` objects (constraints, conditioning, ...).
    dims
        Dimensions; each must be ``>= 2``.
    n_instances
        Instances per (family, dim).
    seed
        Battery seed.

    Returns
    -------
    list of (str, Family)
        ``name`` is ``problem.name`` (``<family>_d<dim>_i<index>``), so a
        caller can key on it without reaching into the object.

    Raises
    ------
    ValueError
        When two families share a label (:meth:`FamilyConfig.name`): they
        would get the same instance seeds and the same record names.
    """
    cfgs = [FamilyConfig(base=f) if isinstance(f, str) else f for f in families]
    # The label is the instance seed's and the record name's only family
    # part, and it does not see ``condition`` / ``rotate`` / ``shift`` /
    # ``extra``: two configs differing only there would draw the same seeds
    # and write their results under the same names.
    seen: Dict[str, FamilyConfig] = {}
    for cfg in cfgs:
        label = cfg.name()
        if label in seen:
            raise ValueError(
                f"two families in one battery share the label {label!r} ({seen[label]!r} and {cfg!r}); "
                "give them distinct FamilyConfig(label=...)"
            )
        seen[label] = cfg
    out: List[Tuple[str, Family]] = []
    for cfg in cfgs:
        label = cfg.name()
        for dim in dims:
            for index in range(int(n_instances)):
                k = cfg.k_for(index)
                problem = Family(
                    cfg.base,
                    dim=int(dim),
                    seed=_instance_seed(seed, label, int(dim), index),
                    shift=cfg.shift,
                    rotate=cfg.rotate,
                    condition=cfg.condition,
                    n_constraints=k,
                    constraint_kind=cfg.constraint_kind,
                    family=label,
                    instance=index,
                    base_params=cfg.base_params,
                    failure=cfg.failure,
                    **cfg.extra,
                )
                out.append((problem.name, problem))
    return out
