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
Parametrically randomized benchmark problems
============================================

Phase 3 of :doc:`../planning/SELF_IMPROVEMENT_LOOP.md`: turn the fixed
harness battery into one that samples *fresh transformed instances* per
repetition so the composite score becomes a Monte-Carlo estimate of
**expected performance on a problem family**, not a point estimate on a
memorized instance.  Without this layer, an autonomous improvement loop
would over-fit to quirks of the fixed instances (the exact Rosenbrock
valley at ``(1, 1)``, the axis-aligned symmetry of Rastrigin, …).

What gets randomized
--------------------

Given a base problem :math:`f_{base}(y)` with optimum at
:math:`y^\\star_{base}` (e.g. Rastrigin's origin, Rosenbrock's
:math:`(1, \\ldots, 1)`), a transformed instance :math:`\\tilde f` is
built from four independent transforms:

- **Translation**: a random optimum location :math:`x^\\star` is sampled
  uniformly from an *interior* region of the box (default: central 70%)
  so the optimum never sits on a boundary.
- **Rotation**: a Haar-random orthogonal matrix :math:`Q \\in O(d)` is
  drawn via the Gram–Schmidt on a standard-normal matrix (equivalent to
  :func:`scipy.stats.ortho_group` but dependency-free).
- **Scaling**: a diagonal ill-conditioning matrix :math:`\\Lambda` with
  :math:`\\log_{10}` condition number uniform in ``[0, cond_max]``.
- **Noise**: additive Gaussian :math:`\\varepsilon \\sim \\mathcal N(0, \\sigma^2)`
  on each evaluation (per-eval, using an instance-local :class:`numpy.random.Generator`).

The composite transform is

.. math::

   \\tilde f(x) = f_{base}\\bigl(Q \\, \\Lambda \\, (x - x^\\star)
                                + y^\\star_{base}\\bigr)
                 + \\sigma \\varepsilon.

By construction, :math:`\\tilde f(x^\\star) = f_{base}(y^\\star_{base}) = f_{opt}`,
so ``known_optima`` for the transformed problem is ``{x: x*, fx: f_opt}``
and the existing harness metrics (``func_distance``, ``ert``,
``composite_score``) work unchanged.

Reproducibility contract
------------------------

All randomness flows from a single 32-bit seed derived via SHA-256 from
``(base_seed, iteration_id, family_name, rep)`` — the same scheme
:meth:`panobbgo.harness.BenchmarkHarness._derive_seed` already uses for
strategy seeds.  Consequences:

* Within one ``iteration_id``, a ``before`` and ``after`` run of a
  self-improvement loop see the **same** sampled instances — the
  comparison is apples-to-apples.
* Different ``iteration_id`` values intentionally draw different
  instances — that is the point.  Aggregate across many iterations to get
  a generalisation estimate.
* A regression flagged by the loop is deterministically reproducible from
  the printed ``(base_seed, iteration_id, family_name, rep)`` tuple.

Per-family capability flags
---------------------------

Not every transform is sensible for every problem.  Schwefel's optimum is
at the box boundary and rotation would push ``y`` off the function's
visible domain; Griewank's oscillation pattern couples directly to the
coordinate axes and rotation breaks the published global-minimum
location.  Each :class:`ProblemFamily` declares which transforms it
supports; the sampler silently skips unsupported ones.

See also
--------

* :mod:`panobbgo.harness` — the harness this plugs into.
* :mod:`panobbgo.harness_baselines` — Phase 2 external reference solvers.
* ``planning/SELF_IMPROVEMENT_LOOP.md`` — design, §4.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from panobbgo.benchmark import ProblemSpec
from panobbgo.lib import Problem


# ---------------------------------------------------------------------------
# Seed derivation
# ---------------------------------------------------------------------------


def derive_instance_seed(base_seed: int, iteration_id: int, family_name: str, rep: int) -> int:
    """Derive the 32-bit seed used to sample a family instance.

    Uses SHA-256 over a ``|``-separated tuple so the result is stable across
    interpreter invocations (unaffected by ``PYTHONHASHSEED``).

    Args:
        base_seed: The harness base seed.
        iteration_id: Self-improvement loop iteration index.  Within one
            iteration, ``before``/``after`` runs see identical instances.
        family_name: Human-readable family identifier (e.g. ``"Rastrigin"``).
        rep: Repetition index within the iteration.

    Returns:
        A non-negative 32-bit integer suitable for ``numpy.random.default_rng``.
    """
    raw = hashlib.sha256(f"instance|{base_seed}|{iteration_id}|{family_name}|{rep}".encode()).hexdigest()
    return int(raw, 16) % (2**32)


# ---------------------------------------------------------------------------
# TransformedProblem wrapper
# ---------------------------------------------------------------------------


class TransformedProblem(Problem):
    """A :class:`~panobbgo.lib.Problem` that wraps a base problem with the
    composition of translation, rotation, scaling, and noise.

    The transformed objective is::

        y = Q @ (Lambda * (x - x_star)) + y_base_star
        f_new(x) = base.eval(y) + sigma * N(0, 1)

    so ``f_new(x_star) == base.eval(y_base_star) == f_opt`` by construction.

    Args:
        base_problem: An already-constructed base :class:`Problem` instance
            whose ``eval`` will be wrapped.
        x_star: Target optimum location in the new coordinate system.
            ``len(x_star)`` defines the dimensionality.
        y_base_star: Optimum location in the base problem's frame (default:
            zeros).  For Rastrigin / Ackley / Sphere this is ``0``.  For
            Rosenbrock the translation is handled via its ``optimum``
            kwarg so ``y_base_star = 0`` works as long as the base
            problem has been instantiated with ``optimum=0``-equivalent.
        Q: Orthogonal rotation matrix.  ``None`` disables rotation.
        scale: Diagonal scaling vector (one factor per dimension).
            ``None`` disables scaling.
        noise_sigma: Standard deviation of additive Gaussian noise on
            each evaluation.  ``0`` disables noise.
        noise_seed: Seed for the per-instance noise generator.  Each
            instance carries its own RNG so noise is reproducible within
            a run (modulo evaluation order in threaded execution).
        box: Bounding box for the transformed problem.  Defaults to the
            base problem's box.
        name: Optional name string recorded for debugging.
    """

    # ``eval`` relies on these attributes.  They are set in __init__ but
    # we annotate them here for type checkers.
    _base: Problem
    _x_star: np.ndarray
    _y_base_star: np.ndarray
    _Q: Optional[np.ndarray]
    _scale: Optional[np.ndarray]
    _noise_sigma: float
    _noise_rng: np.random.Generator
    _transform_name: str

    def __init__(
        self,
        base_problem: Problem,
        x_star: Any,
        y_base_star: Any = None,
        Q: Optional[np.ndarray] = None,
        scale: Any = None,
        noise_sigma: float = 0.0,
        noise_seed: int = 0,
        box: Optional[Sequence[Tuple[float, float]]] = None,
        name: str = "Transformed",
    ) -> None:
        x_star_arr = np.asarray(x_star, dtype=np.float64)
        dims = x_star_arr.size

        if base_problem.dim != dims:
            raise ValueError(f"base_problem.dim={base_problem.dim} does not match len(x_star)={dims}")

        if y_base_star is None:
            y_base_star_arr = np.zeros(dims, dtype=np.float64)
        else:
            y_base_star_arr = np.asarray(y_base_star, dtype=np.float64)
            if y_base_star_arr.size != dims:
                raise ValueError(f"y_base_star has length {y_base_star_arr.size}, expected {dims}")

        if Q is not None:
            Q_arr = np.asarray(Q, dtype=np.float64)
            if Q_arr.shape != (dims, dims):
                raise ValueError(f"Q has shape {Q_arr.shape}, expected ({dims}, {dims})")
        else:
            Q_arr = None

        if scale is not None:
            scale_arr = np.asarray(scale, dtype=np.float64)
            if scale_arr.size != dims:
                raise ValueError(f"scale has length {scale_arr.size}, expected {dims}")
        else:
            scale_arr = None

        if noise_sigma < 0.0:
            raise ValueError(f"noise_sigma must be non-negative, got {noise_sigma}")

        # Use the base problem's box by default.
        if box is None:
            box_raw = base_problem.box.box
            # Convert to list of tuples (Problem.__init__ expects this).
            box_list: List[Tuple[float, float]] = [(float(lo), float(hi)) for lo, hi in box_raw]
        else:
            box_list = [(float(lo), float(hi)) for lo, hi in box]

        Problem.__init__(self, box_list)

        self._base = base_problem
        self._x_star = x_star_arr
        self._y_base_star = y_base_star_arr
        self._Q = Q_arr
        self._scale = scale_arr
        self._noise_sigma = float(noise_sigma)
        self._noise_rng = np.random.default_rng(noise_seed)
        self._transform_name = name

        # Public attributes used by the sampler / harness bookkeeping.
        self.optimum = x_star_arr.copy()

    # ------------------------------------------------------------------
    # Transform plumbing
    # ------------------------------------------------------------------

    def transform_input(self, x: np.ndarray) -> np.ndarray:
        """Map an input point in the transformed frame to the base frame."""
        y = x - self._x_star
        if self._scale is not None:
            y = self._scale * y
        if self._Q is not None:
            y = self._Q @ y
        y = y + self._y_base_star
        return y

    def eval(self, x: np.ndarray) -> float:
        y = self.transform_input(np.asarray(x, dtype=np.float64))
        fx = float(self._base.eval(y))
        if self._noise_sigma > 0.0:
            fx = fx + float(self._noise_sigma * self._noise_rng.standard_normal())
        return fx

    def eval_constraints(self, x: np.ndarray) -> Optional[np.ndarray]:
        # Most classical test problems have no constraints, and mixing
        # constraints with non-linear rotations is non-trivial.  We simply
        # skip constraints for transformed problems.
        return None

    def __repr__(self) -> str:
        return f"TransformedProblem(name={self._transform_name!r}, dim={self.dim}, sigma={self._noise_sigma})"


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------


def sample_orthogonal(rng: np.random.Generator, dim: int) -> np.ndarray:
    """Draw a Haar-uniform orthogonal matrix in :math:`O(d)`.

    QR-decompose a standard-normal matrix and correct the sign of ``R``'s
    diagonal so ``Q`` has the Haar distribution (Mezzadri 2007).
    """
    if dim < 1:
        raise ValueError(f"dim must be >= 1, got {dim}")
    A = rng.standard_normal(size=(dim, dim))
    Q, R = np.linalg.qr(A)
    # Mezzadri's correction: multiply columns of Q by sign(diag(R)) so the
    # distribution is Haar-invariant.  Without this, Q is only approximately
    # uniform.
    d = np.sign(np.diag(R))
    d[d == 0] = 1.0
    Q = Q * d
    return Q


def sample_diagonal_scaling(rng: np.random.Generator, dim: int, log10_cond_max: float) -> np.ndarray:
    """Draw a diagonal scaling vector with a sampled condition number.

    The condition number :math:`\\kappa` is drawn so that :math:`\\log_{10}
    \\kappa \\sim U[0, \\mathtt{log10\\_cond\\_max}]`; the per-coordinate
    scales are spaced geometrically between :math:`1` and :math:`\\kappa`
    and then shuffled so no dimension is privileged.  A ``log10_cond_max``
    of ``0`` returns an all-ones vector (no scaling).
    """
    if dim < 1:
        raise ValueError(f"dim must be >= 1, got {dim}")
    if log10_cond_max < 0:
        raise ValueError(f"log10_cond_max must be non-negative, got {log10_cond_max}")
    if log10_cond_max == 0 or dim == 1:
        return np.ones(dim, dtype=np.float64)

    log10_cond = rng.uniform(0.0, log10_cond_max)
    cond = 10.0**log10_cond
    # Geometric spacing from 1 to cond across dims.
    exponents = np.linspace(0.0, 1.0, num=dim)
    scales = cond**exponents  # range: [1, cond]
    # Shuffle so the largest scale is not always the first axis.
    rng.shuffle(scales)
    return scales


def sample_interior_point(
    rng: np.random.Generator,
    box: np.ndarray,
    margin_fraction: float,
) -> np.ndarray:
    """Sample a point uniformly from the *interior* of the box.

    ``margin_fraction`` shrinks each axis by that fraction on *each* side
    so the sampled optimum never sits on a boundary.  A value of ``0.15``
    means the sample range is the central 70% of the box.

    Args:
        rng: Numpy generator.
        box: Array of shape ``(d, 2)`` with ``(lo, hi)`` rows.
        margin_fraction: Per-side margin as a fraction of the axis range.
            Must be in ``[0, 0.5)``.
    """
    if not 0.0 <= margin_fraction < 0.5:
        raise ValueError(f"margin_fraction must be in [0, 0.5), got {margin_fraction}")
    box = np.asarray(box, dtype=np.float64)
    lo = box[:, 0]
    hi = box[:, 1]
    rng_width = hi - lo
    new_lo = lo + margin_fraction * rng_width
    new_hi = hi - margin_fraction * rng_width
    return new_lo + rng.random(size=box.shape[0]) * (new_hi - new_lo)


# ---------------------------------------------------------------------------
# ProblemFamily and the sampler
# ---------------------------------------------------------------------------


#: Transforms the sampler knows about.  Each family declares a subset of
#: these.  ``"translate"`` shifts the optimum, ``"rotate"`` applies a
#: Haar-random orthogonal matrix, ``"scale"`` multiplies by a diagonal
#: ill-conditioning factor, ``"noise"`` adds Gaussian evaluation noise.
SUPPORTED_TRANSFORMS: Tuple[str, ...] = ("translate", "rotate", "scale", "noise")


@dataclass
class ProblemFamily:
    """Declarative specification of a parametrically-randomized problem family.

    The sampler draws one :class:`ProblemSpec` per ``(family, rep)`` using
    the transforms listed in :attr:`supported_transforms`.  Transforms not
    in the set are silently skipped — this is how, e.g., Schwefel opts out
    of rotation (its optimum is near the box boundary and rotation would
    push ``y`` off the function's sensible domain).

    Args:
        name: Display name shown in the results table (e.g. ``"Rastrigin_family"``).
        base_class: The :class:`~panobbgo.lib.Problem` subclass to wrap.
        base_factory: Optional callable ``(dim, rng) -> Problem`` used
            instead of ``base_class(dims=dim)`` when the base problem
            needs custom construction kwargs.  If set, ``base_class`` is
            ignored for construction (still reported for introspection).
        y_base_star: Location of the base problem's optimum (in the base
            problem's own frame).  Default: zeros.  For Rosenbrock the
            canonical ``(1, …, 1)`` optimum is handled by constructing the
            base problem with ``optimum=0`` (supported by
            :class:`~panobbgo.lib.classic.Rosenbrock`), so the default of
            zeros is correct.
        f_opt: Objective value at the optimum — invariant under the
            transforms in :attr:`supported_transforms`.
        dim_choices: Dimensions the sampler may draw from.  Using a single
            dimension gives a stratified battery; a multi-element tuple
            samples a dimension per instance.
        stratify_dims: When ``True`` (default) and ``len(dim_choices) > 1``,
            :class:`RandomizedProblemSpec` assigns dims **cyclically** by
            ``rep`` (rep ``i`` gets ``dim_choices[i % k]``) so any
            contiguous block of ``k`` reps covers every dim exactly once.
            This eliminates dim-mix variance across iterations of the
            self-improvement loop — see §10 of
            ``planning/SELF_IMPROVEMENT_LOOP.md``.  When ``False``, the
            dim is drawn uniformly from ``dim_choices`` per instance,
            which dilutes cross-iteration deltas with sampling noise.
            Single-dim families are unaffected by this flag.
        supported_transforms: Subset of :data:`SUPPORTED_TRANSFORMS`.
        tolerance: Success tolerance for ``func_distance`` in the harness.
        log10_cond_max: Ceiling on :math:`\\log_{10} \\kappa` when
            ``"scale"`` is enabled.  ``4`` is a representative value from
            the BBOB suite.
        noise_sigma: Standard deviation of additive Gaussian noise when
            ``"noise"`` is enabled.
        margin_fraction: Per-side margin of the translation region as a
            fraction of the axis range (see :func:`sample_interior_point`).
    """

    name: str
    base_class: type
    base_factory: Optional[Callable[[int, np.random.Generator], Problem]] = None
    y_base_star: Optional[Sequence[float]] = None
    f_opt: float = 0.0
    dim_choices: Tuple[int, ...] = (2,)
    stratify_dims: bool = True
    supported_transforms: Set[str] = field(default_factory=lambda: {"translate", "rotate", "scale"})
    tolerance: float = 0.1
    log10_cond_max: float = 2.0
    noise_sigma: float = 0.0
    margin_fraction: float = 0.15

    def __post_init__(self) -> None:
        unknown = set(self.supported_transforms) - set(SUPPORTED_TRANSFORMS)
        if unknown:
            raise ValueError(f"Unknown transforms in family {self.name!r}: {sorted(unknown)}")
        if not self.dim_choices:
            raise ValueError(f"family {self.name!r} has empty dim_choices")

    def _build_base(self, dim: int, rng: np.random.Generator) -> Problem:
        if self.base_factory is not None:
            return self.base_factory(dim, rng)
        return self.base_class(dims=dim)

    def stratified_dim_for_rep(self, rep: int) -> int:
        """Return the dim assigned to rep ``rep`` under cyclic stratification.

        Cycles through :attr:`dim_choices` so that any contiguous block of
        ``len(dim_choices)`` reps covers every declared dimension exactly
        once.  When ``len(dim_choices) == 1`` the result is the constant
        family dim.  See §10 of ``planning/SELF_IMPROVEMENT_LOOP.md``.

        Args:
            rep: Non-negative repetition index.

        Returns:
            The assigned dim as a Python ``int``.
        """
        if rep < 0:
            raise ValueError(f"rep must be non-negative, got {rep}")
        return int(self.dim_choices[rep % len(self.dim_choices)])

    def sample_instance(
        self,
        rng: np.random.Generator,
        dim: Optional[int] = None,
    ) -> Tuple[TransformedProblem, Dict[str, Any]]:
        """Draw one transformed instance and its parameter record.

        Args:
            rng: Numpy generator used for *all* random choices in this
                instance (translation, rotation, scaling, noise seed).
            dim: Optional dimension override.  When ``None`` (default),
                the dim is drawn from :attr:`dim_choices` (constant for
                single-dim families, uniform draw for multi-dim).
                Stratified callers (e.g. :class:`RandomizedProblemSpec`)
                pass an explicit dim so the rng is not consumed by the
                dim choice and the stratified schedule is honoured.
        """
        # Choose a dimension.
        if dim is not None:
            dim = int(dim)
            if dim not in self.dim_choices:
                raise ValueError(f"family {self.name!r}: dim={dim} is not in dim_choices={self.dim_choices}")
        elif len(self.dim_choices) == 1:
            dim = int(self.dim_choices[0])
        else:
            dim = int(rng.choice(self.dim_choices))

        base = self._build_base(dim, rng)
        box = np.asarray(base.box.box, dtype=np.float64)

        params: Dict[str, Any] = {"dim": dim, "family": self.name}

        if "translate" in self.supported_transforms:
            x_star = sample_interior_point(rng, box, self.margin_fraction)
            params["translation"] = x_star.tolist()
        else:
            # No translation: place the optimum at the base optimum (so the
            # instance is literally the base problem).
            if self.y_base_star is None:
                x_star = np.zeros(dim, dtype=np.float64)
            else:
                x_star = np.asarray(self.y_base_star, dtype=np.float64).copy()

        if "rotate" in self.supported_transforms and dim >= 2:
            Q = sample_orthogonal(rng, dim)
            params["rotation_trace"] = float(np.trace(Q))
        else:
            Q = None

        if "scale" in self.supported_transforms and self.log10_cond_max > 0:
            scale = sample_diagonal_scaling(rng, dim, self.log10_cond_max)
            params["scale_cond"] = float(scale.max() / scale.min())
        else:
            scale = None

        if "noise" in self.supported_transforms and self.noise_sigma > 0:
            sigma = float(self.noise_sigma)
            params["noise_sigma"] = sigma
        else:
            sigma = 0.0

        # Noise RNG seed is drawn from the same stream so a full instance
        # is replayable from the single derived seed.
        noise_seed = int(rng.integers(0, 2**32 - 1))

        y_base_star: Optional[Sequence[float]]
        if self.y_base_star is None:
            y_base_star = None
        else:
            y_base_star = list(self.y_base_star)

        problem = TransformedProblem(
            base_problem=base,
            x_star=x_star,
            y_base_star=y_base_star,
            Q=Q,
            scale=scale,
            noise_sigma=sigma,
            noise_seed=noise_seed,
            name=self.name,
        )

        return problem, params


# ---------------------------------------------------------------------------
# RandomizedProblemSpec: a ProblemSpec that resamples per rep
# ---------------------------------------------------------------------------


class RandomizedProblemSpec(ProblemSpec):
    """A :class:`~panobbgo.benchmark.ProblemSpec` that samples a *fresh*
    transformed instance on every call to :meth:`create_problem_for_rep`.

    Serves as the bridge between :class:`ProblemFamily` (declarative) and
    the harness (which iterates over :class:`ProblemSpec` instances).  The
    ``known_optima`` and ``tolerance`` fields carry the family's invariants
    — the optimum value :math:`f_{opt}` is constant, only ``x^\\star``
    (and thus the landscape) changes across reps.

    The usual :meth:`~panobbgo.benchmark.ProblemSpec.create_problem` is
    overridden to raise — callers must use :meth:`create_problem_for_rep`
    so we know the rep index that drives the seed.

    Attributes beyond :class:`ProblemSpec`:
        family: The :class:`ProblemFamily` this spec wraps.
        iteration_id: Passed through to :func:`derive_instance_seed`.
        base_seed: Passed through to :func:`derive_instance_seed`.
        _last_sampled_params: Populated after each sample, useful for
            debugging and ledger output.
    """

    def __init__(
        self,
        family: ProblemFamily,
        iteration_id: int,
        base_seed: int,
        max_evaluations: int,
        name: Optional[str] = None,
    ) -> None:
        # Represent the "representative" dim as the first in dim_choices.
        # The actual dim is re-drawn per rep; we only use this for the
        # legacy ``dims`` field used in summary output.
        repr_dim = int(family.dim_choices[0])

        super().__init__(
            name=name or family.name,
            problem_class=family.base_class,
            dims=repr_dim,
            known_optima=[{"x": [0.0] * repr_dim, "fx": family.f_opt}],
            tolerance=family.tolerance,
            max_evaluations=max_evaluations,
        )
        self.family = family
        self.iteration_id = iteration_id
        self.base_seed = base_seed
        self._last_sampled_params: Optional[Dict[str, Any]] = None

    def create_problem(self) -> Problem:  # type: ignore[override]
        raise RuntimeError(
            "RandomizedProblemSpec requires a rep index; call create_problem_for_rep(rep) instead of create_problem()."
        )

    def create_problem_for_rep(self, rep: int) -> TransformedProblem:
        """Sample a fresh instance for the given repetition.

        For multi-dim families with :attr:`ProblemFamily.stratify_dims`
        enabled, the dim is assigned cyclically by ``rep`` so any
        contiguous block of ``len(dim_choices)`` reps covers every
        declared dim exactly once.  Single-dim families are unaffected.
        """
        seed = derive_instance_seed(self.base_seed, self.iteration_id, self.family.name, rep)
        rng = np.random.default_rng(seed)
        if self.family.stratify_dims and len(self.family.dim_choices) > 1:
            dim_override: Optional[int] = self.family.stratified_dim_for_rep(rep)
        else:
            dim_override = None
        problem, params = self.family.sample_instance(rng, dim=dim_override)
        params["seed"] = seed
        params["rep"] = rep
        params["stratified_dim"] = dim_override is not None
        self._last_sampled_params = params
        return problem

    def last_sampled_params(self) -> Optional[Dict[str, Any]]:
        """Return the parameters of the most recently sampled instance, or ``None``."""
        return self._last_sampled_params


# ---------------------------------------------------------------------------
# Pre-packaged family registry
# ---------------------------------------------------------------------------


def _rosenbrock_factory(dim: int, rng: np.random.Generator) -> Problem:
    """Build a Rosenbrock base problem whose optimum is at the origin.

    The standard Rosenbrock optimum is at ``(1, …, 1)``; passing
    ``optimum=0`` to :class:`~panobbgo.lib.classic.Rosenbrock` shifts it
    to the origin so the transform with ``y_base_star=0`` works cleanly.
    We also supply a symmetric box around the origin.
    """
    from panobbgo.lib.classic import Rosenbrock

    box = [(-2.0, 2.0)] * dim
    return Rosenbrock(dims=dim, optimum=0.0, box=box)


def make_default_families() -> List[ProblemFamily]:
    """Return the canonical set of parametric families used by the harness.

    These cover the three axes the self-improvement loop needs to
    generalize across:

    - **Separable** multimodal functions (Rastrigin, Ackley) — tests that
      local search is not privileged by axis alignment.
    - **Non-separable** unimodal (Rosenbrock-family) — tests gradient /
      valley-following.
    - **Smooth convex** (Sphere/DeJong) — the base-line sanity case.

    Families that do not compose cleanly with rotation or extreme scaling
    (Schwefel, Griewank) are intentionally excluded from the default set.
    Downstream callers can extend the list via
    :attr:`panobbgo.harness.HarnessConfig.extra_families`.
    """
    from panobbgo.lib.classic import Rastrigin, Ackley, DeJong

    return [
        ProblemFamily(
            name="Rastrigin_family",
            base_class=Rastrigin,
            f_opt=0.0,
            dim_choices=(2,),
            supported_transforms={"translate", "rotate", "scale"},
            tolerance=0.2,
            log10_cond_max=1.0,
        ),
        ProblemFamily(
            name="Ackley_family",
            base_class=Ackley,
            f_opt=0.0,
            dim_choices=(2,),
            supported_transforms={"translate", "rotate", "scale"},
            tolerance=0.2,
            log10_cond_max=1.0,
        ),
        ProblemFamily(
            name="Rosenbrock_family",
            base_class=type("_Rosenbrock_origin", (), {}),
            base_factory=_rosenbrock_factory,
            f_opt=0.0,
            dim_choices=(2,),
            supported_transforms={"translate", "scale"},
            tolerance=0.2,
            log10_cond_max=1.0,
        ),
        ProblemFamily(
            name="DeJong_family",
            base_class=DeJong,
            f_opt=0.0,
            dim_choices=(2,),
            supported_transforms={"translate", "rotate", "scale"},
            tolerance=0.1,
            log10_cond_max=2.0,
        ),
    ]


def make_randomized_specs(
    families: Sequence[ProblemFamily],
    iteration_id: int,
    base_seed: int,
    max_evaluations: int,
) -> List[RandomizedProblemSpec]:
    """Bundle a list of families into :class:`RandomizedProblemSpec` instances.

    Used by :class:`~panobbgo.harness.BenchmarkHarness` when
    :attr:`~panobbgo.harness.HarnessConfig.randomize` is set.
    """
    return [
        RandomizedProblemSpec(
            family=f,
            iteration_id=iteration_id,
            base_seed=base_seed,
            max_evaluations=max_evaluations,
        )
        for f in families
    ]
