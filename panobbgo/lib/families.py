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

.. codeauthor:: Harald Schilly <harald.schilly@gmail.com>
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from panobbgo.lib.classic import Ackley, DeJong, Griewank, Rastrigin, Rosenbrock, Schwefel
from panobbgo.lib.lib import Problem

#: Callable ``dim -> (base function, base minimiser)``.  The minimiser is
#: given in the base's own coordinates; :class:`Family` never sees it,
#: because :func:`_normalise` folds it into the returned callable.
BaseFactory = Callable[[int], Callable[[np.ndarray], float]]

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
}

#: Constraint constructions understood by :class:`Family`.
CONSTRAINT_KINDS: Tuple[str, ...] = ("linear", "ball", "mixed")


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
        *after* the rotation, so it is not itself rotated away.
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
        Defaults to ``base`` (plus a constraint tag when ``k > 0``).
    instance
        Instance index, carried through to the record.  Purely a label.

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
    ) -> None:
        if base not in BASE_FUNCTIONS:
            raise ValueError(f"unknown base function {base!r}; known: {sorted(BASE_FUNCTIONS)}")
        if dim < 2:
            raise ValueError(f"dim must be >= 2, got {dim}")
        if constraint_kind not in CONSTRAINT_KINDS:
            raise ValueError(f"unknown constraint_kind {constraint_kind!r}; known: {list(CONSTRAINT_KINDS)}")
        if n_constraints < 0:
            raise ValueError(f"n_constraints must be >= 0, got {n_constraints}")

        rng = np.random.default_rng(int(seed))
        b = float(box_half_width)
        Problem.__init__(self, [(-b, b)] * dim)

        self.base: str = base
        self.instance: int = int(instance)
        self.seed: int = int(seed)
        self.condition: float = float(condition)
        self.n_constraints: int = int(n_constraints)
        self.constraint_kind: str = constraint_kind
        self.family: str = family if family is not None else _family_label(base, n_constraints, constraint_kind)

        self._base_fn: Callable[[np.ndarray], float] = BASE_FUNCTIONS[base](dim)
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

    # -- evaluation ----------------------------------------------------------

    def eval(self, x: np.ndarray) -> float:
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
        return f"Family '{self.name}': base={self.base}, dim={self.dim}, f_opt={self._f_opt:g}{tail}"


# ---------------------------------------------------------------------------
# Instance generation
# ---------------------------------------------------------------------------


def _family_label(base: str, n_constraints: int, constraint_kind: str) -> str:
    """``rastrigin`` / ``rastrigin_lin`` / ``rastrigin_ball`` / ``rastrigin_mix``."""
    if n_constraints <= 0:
        return base
    tag = {"linear": "lin", "ball": "ball", "mixed": "mix"}[constraint_kind]
    return f"{base}_{tag}"


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
    """

    base: str
    label: Optional[str] = None
    condition: float = 1.0
    rotate: bool = True
    shift: bool = True
    n_constraints: Union[int, Sequence[int]] = 0
    constraint_kind: str = "linear"
    extra: Dict[str, Any] = field(default_factory=dict)

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
        return _family_label(self.base, k, self.constraint_kind)


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
    """
    cfgs = [FamilyConfig(base=f) if isinstance(f, str) else f for f in families]
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
                    **cfg.extra,
                )
                out.append((problem.name, problem))
    return out
