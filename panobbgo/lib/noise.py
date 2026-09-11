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
Noisy objectives
================

Panobbgo is "Parallel **Noisy** Black-Box Global Optimization", but every
number in ``planning/GOAL.md`` §2 was measured on a *noiseless*,
unconstrained battery.  This module supplies the missing half: a
deterministic noise wrapper that turns any
:class:`~panobbgo.lib.lib.Problem` into a noisy one while keeping the
true, noiseless value available for scoring.

Two objects live here:

* the **noise models** (:class:`NoiseModel` subclasses) — pure
  ``precision -> noisy precision`` maps that consume a
  :class:`numpy.random.Generator`;
* the **wrapper** (:class:`NoisyProblem`) — a ``Problem`` that delegates
  to an inner problem, applies a model, and exposes both values.

Where the noise acts
--------------------

All models act on the **precision** ``f_raw = max(0, f(x) - f_opt)``, not
on the raw objective value, and the wrapper adds ``f_opt`` back
afterwards.  This mirrors the BBOB noisy suite, where the noise-free
part of a function is non-negative by construction and the noise models
are (mostly) multiplicative — applying them to a value that has been
shifted by an arbitrary ``f_opt`` would make the noise level depend on
the shift.  ``f_opt`` is taken from the inner problem's ``optimum_y``
when it has one, else it is ``0.0``, and may be overridden.

Determinism: noise is a pure function of ``(seed, x)``
------------------------------------------------------

By default (``resample=False``) the noise added at a point is a pure
function of ``(seed, x)``: the bytes of ``x`` are hashed together with
the seed and the resulting digest seeds a fresh
:class:`numpy.random.Generator` for that one call.  Consequences, both
deliberate:

* **Reproducibility.**  A seeded run under ``config.sync_evaluation`` is
  bit-identical across invocations even though evaluations are handed
  out by a thread pool: the noise does not depend on *when* a point is
  evaluated, only on *which* point it is.  Without this, every paired
  A/B in :mod:`panobbgo.harness_ioh` would carry the noise realisation
  as an extra variance term and the 12-seed roster would resolve
  nothing.
* **Re-evaluating the same ``x`` returns the same value**, so an
  optimizer cannot average the noise away by resampling.  That is
  *realistic* for a deterministic simulator with discretisation or
  solver-tolerance error — the usual "noisy black box" in engineering —
  and *wrong* for a genuinely stochastic simulator (a Monte-Carlo
  model, a stochastic controller, a measurement), where resampling is
  the obvious variance-reduction move.

For the second regime pass ``resample=True``.  The wrapper then counts
how often each distinct ``x`` has been evaluated and mixes that counter
into the hash, so the *k*-th evaluation of a point draws fresh noise
while the whole run stays reproducible as long as the sequence of
points is (per point) deterministic — which it is under
``sync_evaluation``.  Under the default asynchronous evaluator the
counter is assigned in arrival order, so a run with ``resample=True``
is reproducible only up to the ordering of *identical* points.

Scoring
-------

The wrapper exposes three ways to read one evaluation:

* :meth:`NoisyProblem.eval` — the noisy value, what the optimizer sees;
* :meth:`NoisyProblem.true_eval` — the noiseless value;
* :meth:`NoisyProblem.eval_pair` — both, from **one** call to the inner
  problem.  :class:`~panobbgo.ioh_runner.IOHTracker` prefers this, so
  wrapping an :class:`~panobbgo.lib.ioh_wrapper.IOHProblem` costs no
  extra worker round-trips.

Scoring AOCC on the *true* value is what the BBOB noisy suite does (its
logger records the noise-free value and measures target hits on it) and
is the only way to get a metric that is not itself a random variable of
the noise.

.. codeauthor:: Harald Schilly <harald.schilly@gmail.com>
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from panobbgo.lib.lib import Problem


#: BBOB's guard against division by zero in the noise definitions.
_EPS: float = 1e-99

#: BBOB adds this constant to every noisy value so the observed value is
#: strictly positive (and, incidentally, never below the 1e-8 AOCC floor).
_BBOB_FLOOR: float = 1.01e-8


# ---------------------------------------------------------------------------
# Deterministic per-point RNG
# ---------------------------------------------------------------------------


def _x_bytes(x: np.ndarray) -> bytes:
    """Canonical byte image of ``x`` — the hash key of a point."""
    return np.ascontiguousarray(np.asarray(x, dtype=np.float64)).tobytes()


def rng_for_point(seed: int, x: np.ndarray, draw: int = 0) -> np.random.Generator:
    """Return the generator belonging to ``(seed, x, draw)``.

    Keyed BLAKE2b over the float64 bytes of ``x`` — a *pure function*,
    independent of evaluation order, thread, or process.  ``draw`` is the
    resampling counter (0 for the frozen-noise default).
    """
    return _rng_for_bytes(seed, _x_bytes(x), draw)


def _rng_for_bytes(seed: int, xb: bytes, draw: int) -> np.random.Generator:
    key = (int(seed) & 0xFFFFFFFFFFFFFFFF).to_bytes(8, "little")
    digest = hashlib.blake2b(xb + int(draw).to_bytes(8, "little"), digest_size=16, key=key).digest()
    return np.random.default_rng(int.from_bytes(digest, "little"))


# ---------------------------------------------------------------------------
# Noise models
# ---------------------------------------------------------------------------


class NoiseModel:
    """A ``precision -> noisy precision`` map driven by a seeded generator.

    Subclasses implement :meth:`apply`.  They must be pure: everything
    random comes out of the ``rng`` argument, so that
    :class:`NoisyProblem`'s ``(seed, x)`` determinism holds.
    """

    #: Short tag used in battery names and reports.
    name: str = "none"

    def apply(self, f_raw: float, rng: np.random.Generator) -> float:
        """Return the noisy precision for a noise-free precision ``f_raw >= 0``."""
        raise NotImplementedError

    def describe(self) -> str:
        return self.name


@dataclass(frozen=True)
class NoNoise(NoiseModel):
    """Identity — useful as a control that exercises the wrapper itself."""

    name: str = "none"

    def apply(self, f_raw: float, rng: np.random.Generator) -> float:
        return float(f_raw)


@dataclass(frozen=True)
class GaussianNoise(NoiseModel):
    r"""BBOB's Gaussian (log-normal, multiplicative) noise model.

    .. math:: f_{GN}(f, \beta) = f \cdot \exp(\beta \, \mathcal{N}(0,1))

    Multiplicative in the precision, so the *relative* error is constant
    over the whole descent: an optimizer at precision ``1e-6`` sees the
    same signal-to-noise ratio as one at ``1e+2``.  This is why the BBOB
    noisy suite remains solvable to high precision at all.

    ``beta`` is the noise level: 0.01 = "moderate", 1.0 = "severe" in the
    BBOB terminology.
    """

    beta: float = 0.01
    name: str = "gauss"

    def apply(self, f_raw: float, rng: np.random.Generator) -> float:
        return float(f_raw * np.exp(self.beta * rng.standard_normal())) + _BBOB_FLOOR

    def describe(self) -> str:
        return f"gauss(beta={self.beta:g})"


@dataclass(frozen=True)
class UniformNoise(NoiseModel):
    r"""BBOB's uniform noise model.

    .. math::

        f_{UN}(f, \alpha, \beta) = f \cdot U(0,1)^{\beta}
            \cdot \max\!\left(1, \left(\frac{10^9}{f + \epsilon}\right)^{\alpha U(0,1)}\right)

    The first factor shrinks the value (an optimistic bias that grows
    with ``beta``); the second is a *severe* inflation that only switches
    on once ``f`` drops below ``1e9``, i.e. exactly in the endgame.  It is
    the model that punishes an optimizer for trusting a single good
    reading near the optimum.

    BBOB scales ``alpha`` with the dimension: moderate noise uses
    ``alpha = 0.01 * (0.49 + 1/D)``, severe ``alpha = 0.49 + 1/D``.  Use
    :func:`make_noise_model`, which does that arithmetic.
    """

    alpha: float = 0.01
    beta: float = 0.01
    name: str = "unif"

    def apply(self, f_raw: float, rng: np.random.Generator) -> float:
        u1 = float(rng.random())
        u2 = float(rng.random())
        blowup = max(1.0, (1e9 / (f_raw + _EPS)) ** (self.alpha * u2))
        return float(f_raw * (u1**self.beta) * blowup) + _BBOB_FLOOR

    def describe(self) -> str:
        return f"unif(alpha={self.alpha:g}, beta={self.beta:g})"


@dataclass(frozen=True)
class CauchyNoise(NoiseModel):
    r"""BBOB's "seldom Cauchy" outlier model.

    .. math::

        f_{CN}(f, \alpha, p) = f + \alpha \cdot \max\!\left(0,\;
            1000 + \mathbb{1}_{U(0,1) < p} \cdot
            \frac{\mathcal{N}(0,1)}{|\mathcal{N}'(0,1)| + \epsilon}\right)

    *Additive*, and heavy-tailed: with probability ``p`` the evaluation
    is corrupted by a Cauchy draw (a ratio of two normals) that can be
    arbitrarily large — and, because of the ``max(0, ...)``, occasionally
    cancels the constant ``1000`` shift instead.  With probability
    ``1 - p`` the value is simply shifted by ``1000 * alpha``, a constant
    that is invisible to any rank-based optimizer.

    Moderate: ``alpha = 0.01, p = 0.05``.  Severe: ``alpha = 1, p = 0.2``.
    """

    alpha: float = 0.01
    p: float = 0.05
    name: str = "cauchy"

    def apply(self, f_raw: float, rng: np.random.Generator) -> float:
        hit = 1.0 if float(rng.random()) < self.p else 0.0
        cauchy = float(rng.standard_normal()) / (abs(float(rng.standard_normal())) + _EPS)
        return float(f_raw + self.alpha * max(0.0, 1000.0 + hit * cauchy)) + _BBOB_FLOOR

    def describe(self) -> str:
        return f"cauchy(alpha={self.alpha:g}, p={self.p:g})"


@dataclass(frozen=True)
class AdditiveGaussianNoise(NoiseModel):
    r"""Plain additive Gaussian noise, :math:`f + \sigma\varepsilon`.

    Not a BBOB model — the textbook one.  ``sigma`` is **absolute**
    unless ``f_range`` is given, in which case the standard deviation is
    ``sigma * f_range`` (use it when the caller knows the objective's
    span; MA-BBOB does not publish one, so the batteries in
    :mod:`panobbgo.harness_ioh` use the BBOB models instead).

    Note the consequence that BBOB's multiplicative models avoid: an
    absolute noise floor makes every precision below ``sigma``
    indistinguishable, so AOCC on the *observed* trace saturates while
    AOCC on the true trace keeps improving.  That gap is the point of
    the model, not a defect.
    """

    sigma: float = 1e-2
    f_range: Optional[float] = None
    name: str = "add"

    def apply(self, f_raw: float, rng: np.random.Generator) -> float:
        scale = self.sigma if self.f_range is None else self.sigma * self.f_range
        return float(f_raw + scale * rng.standard_normal())

    def describe(self) -> str:
        rel = "" if self.f_range is None else f", f_range={self.f_range:g}"
        return f"add(sigma={self.sigma:g}{rel})"


@dataclass(frozen=True)
class MultiplicativeGaussianNoise(NoiseModel):
    r"""Relative Gaussian noise, :math:`f\,(1 + \sigma\varepsilon)`.

    The linearisation of :class:`GaussianNoise` for small ``sigma``
    (``exp(σε) ≈ 1 + σε``), kept separate because it is what most
    engineering write-ups mean by "5 % measurement noise" and because it
    can return a *negative* precision for ``sigma > 1/3`` or so, which
    the log-normal form cannot.
    """

    sigma: float = 0.1
    name: str = "mult"

    def apply(self, f_raw: float, rng: np.random.Generator) -> float:
        return float(f_raw * (1.0 + self.sigma * rng.standard_normal()))

    def describe(self) -> str:
        return f"mult(sigma={self.sigma:g})"


#: BBOB noise levels.  ``alpha`` for the uniform model is
#: dimension-dependent, so it is filled in by :func:`make_noise_model`.
_BBOB_LEVELS: Dict[str, Dict[str, Dict[str, float]]] = {
    "moderate": {
        "gauss": {"beta": 0.01},
        "unif": {"alpha_coeff": 0.01, "beta": 0.01},
        "cauchy": {"alpha": 0.01, "p": 0.05},
    },
    "severe": {
        "gauss": {"beta": 1.0},
        "unif": {"alpha_coeff": 1.0, "beta": 1.0},
        "cauchy": {"alpha": 1.0, "p": 0.2},
    },
}


def make_noise_model(kind: str, *, dim: int, level: str = "moderate") -> NoiseModel:
    """Build a noise model by short tag.

    Parameters
    ----------
    kind
        ``"gauss"`` / ``"unif"`` / ``"cauchy"`` for the BBOB trio,
        ``"add"`` / ``"mult"`` for the plain Gaussian models, ``"none"``
        for the identity.
    dim
        Problem dimension — the BBOB uniform model's ``alpha`` is
        ``coeff * (0.49 + 1/D)``.
    level
        ``"moderate"`` (the BBOB f101–f106 setting) or ``"severe"``
        (f107–f130).  Ignored by ``add`` / ``mult`` / ``none``.

    Notes
    -----
    The BBOB parameterisations are transcribed from the noisy-suite
    definitions (Hansen, Finck, Ros, Auger 2009, *Real-Parameter
    Black-Box Optimization Benchmarking: Noisy Functions Definitions*).
    They are reproduced here rather than imported because the ``ioh``
    binding lives in an isolated child venv (see
    :mod:`panobbgo.lib.ioh_wrapper`) and its noisy suite is not exposed
    over the worker protocol.  Check them against the reference before
    quoting a number as "BBOB f10x".
    """
    k = kind.lower()
    if k in ("none", ""):
        return NoNoise()
    if k == "add":
        return AdditiveGaussianNoise()
    if k == "mult":
        return MultiplicativeGaussianNoise()
    if level not in _BBOB_LEVELS:
        raise ValueError(f"unknown noise level {level!r}; known: {sorted(_BBOB_LEVELS)}")
    params = _BBOB_LEVELS[level]
    if k == "gauss":
        return GaussianNoise(beta=params["gauss"]["beta"])
    if k == "unif":
        coeff = params["unif"]["alpha_coeff"]
        return UniformNoise(alpha=coeff * (0.49 + 1.0 / float(dim)), beta=params["unif"]["beta"])
    if k == "cauchy":
        return CauchyNoise(alpha=params["cauchy"]["alpha"], p=params["cauchy"]["p"])
    raise ValueError(f"unknown noise kind {kind!r}; known: gauss, unif, cauchy, add, mult, none")


# ---------------------------------------------------------------------------
# The wrapper
# ---------------------------------------------------------------------------


class NoisyProblem(Problem):
    """Wrap ``problem`` so ``eval`` returns a noisy value and the true one stays reachable.

    Parameters
    ----------
    problem
        The inner :class:`~panobbgo.lib.lib.Problem`.  Its box is copied,
        its ``eval_constraints`` and (if present) ``close`` / ``reset`` /
        ``optimum_y`` are delegated.
    model
        A :class:`NoiseModel`.
    seed
        Seed of the noise stream.  Two wrappers with the same seed over
        the same inner problem are the same noisy function; different
        seeds are different *realisations* of the same noise process, and
        that is how the harness makes instances differ.
    resample
        ``False`` (default): the noise at ``x`` is frozen — a pure
        function of ``(seed, x)``.  ``True``: the *k*-th evaluation of
        ``x`` draws fresh noise from ``(seed, x, k)``.  See the module
        docstring for why the default is the frozen one.
    f_opt
        Override the value the precision is measured against.  Defaults
        to the inner problem's ``optimum_y``, else ``0.0``.

    Notes
    -----
    The class deliberately does **not** count evaluations — that is
    :class:`~panobbgo.ioh_runner.IOHTracker`'s job, and the wrapper is
    installed *under* it.
    """

    def __init__(
        self,
        problem: Problem,
        model: NoiseModel,
        *,
        seed: int,
        resample: bool = False,
        f_opt: Optional[float] = None,
    ) -> None:
        box: List[Tuple[float, float]] = [(float(lo), float(hi)) for lo, hi in np.asarray(problem.box.box)]
        super().__init__(box=box)
        self.inner: Problem = problem
        self.model: NoiseModel = model
        self.noise_seed: int = int(seed)
        self.resample: bool = bool(resample)
        if f_opt is None:
            f_opt = float(getattr(problem, "optimum_y", 0.0))
        self._f_opt: float = float(f_opt)
        self._counts: Dict[bytes, int] = {}
        self._counts_lock = threading.Lock()

    # ------------------------------------------------------------------
    # panobbgo.Problem API
    # ------------------------------------------------------------------

    @property
    def optimum_y(self) -> float:
        """The inner problem's optimum — noise leaves it where it was."""
        return self._f_opt

    def eval(self, x: np.ndarray) -> float:
        """The noisy value — what an optimizer running on this problem sees."""
        return self.eval_pair(x)[0]

    def true_eval(self, x: np.ndarray) -> float:
        """The noiseless value.  Costs one inner evaluation."""
        return float(self.inner.eval(np.asarray(x, dtype=np.float64)))

    def eval_pair(self, x: np.ndarray) -> Tuple[float, float]:
        """Return ``(noisy, true)`` from a *single* inner evaluation."""
        x = np.asarray(x, dtype=np.float64)
        true_fx = float(self.inner.eval(x))
        return self.apply_noise(x, true_fx), true_fx

    def apply_noise(self, x: np.ndarray, true_fx: float) -> float:
        """Corrupt a known true value at ``x`` — the pure ``(seed, x)`` map."""
        if not np.isfinite(true_fx):
            return true_fx
        xb = _x_bytes(x)
        draw = 0
        if self.resample:
            with self._counts_lock:
                draw = self._counts.get(xb, 0)
                self._counts[xb] = draw + 1
        rng = _rng_for_bytes(self.noise_seed, xb, draw)
        f_raw = max(0.0, true_fx - self._f_opt)
        return self._f_opt + float(self.model.apply(f_raw, rng))

    def eval_constraints(self, x: np.ndarray) -> Optional[np.ndarray]:
        return self.inner.eval_constraints(x)

    # ------------------------------------------------------------------
    # Lifecycle passthrough (IOHProblem owns a subprocess)
    # ------------------------------------------------------------------

    def close(self) -> None:
        close = getattr(self.inner, "close", None)
        if callable(close):
            close()

    def reset(self) -> None:
        reset = getattr(self.inner, "reset", None)
        if callable(reset):
            reset()
        with self._counts_lock:
            self._counts.clear()

    def __getattr__(self, item: str) -> Any:
        # Only called when normal lookup fails, so this never shadows the
        # Problem API above.  Forwards descriptive attributes of the inner
        # problem (``ioh_name``, ``kind``, ...) to reports and repr().
        if item.startswith("_"):
            raise AttributeError(item)
        try:
            inner = self.__dict__["inner"]
        except KeyError:  # pragma: no cover — during __init__ only
            raise AttributeError(item) from None
        return getattr(inner, item)

    def __repr__(self) -> str:
        return (
            f"NoisyProblem({self.inner!r}, {self.model.describe()}, seed={self.noise_seed}, resample={self.resample})"
        )
