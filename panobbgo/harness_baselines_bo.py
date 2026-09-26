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
Expensive-track external baselines (the ``baselines-bo`` extra)
===============================================================

Reference solvers for the primary battlefield — expensive evaluations at
small budgets (10…200·dim) with ``q`` parallel workers — where the
incumbents are Bayesian-optimisation and model-based tools.  Install with
``uv sync --extra baselines-bo`` (CPU-only torch, see ``pyproject.toml``).
Like the cheap-track baselines in :mod:`panobbgo.harness_baselines` they
are :class:`~panobbgo.harness_baselines.AskTellBaselineStrategy` subclasses,
opt-in by name, and run on the virtual clock
(:func:`panobbgo.virtual_clock.run_ask_tell`).

- :class:`BoTorchQLogEIStrategy` (``Baseline_BoTorch_qLogEI``) — BoTorch's
  batch ``qLogExpectedImprovement`` on a standard ``SingleTaskGP``, the
  setup of the BoTorch closed-loop tutorials.
- :class:`BoTorchTuRBOStrategy` (``Baseline_TuRBO1``) — TuRBO-1 as in the
  BoTorch TuRBO tutorial, with the restarts of the original algorithm.
- :class:`SMACBlackBoxStrategy` (``Baseline_SMAC_BB``) — SMAC3's
  ``BlackBoxFacade`` (GP + EI), SMAC's recommendation for low-dimensional
  continuous problems at small budgets.  SMAC has no batch acquisition:
  with ``q > 1`` it is asked once per free worker and may hand out
  near-duplicates.  Its representative result is the ``q = 1`` run; label
  ``q > 1`` rows "SMAC native (no batch acquisition)".
- :class:`PyBOBYQAStrategy` (``Baseline_PyBOBYQA``) — Py-BOBYQA, the
  **local** model-based (quadratic trust-region) reference, not a global
  optimiser.  **Sequential**: it keeps at most one point in flight, so with
  ``q > 1`` workers it uses one.

Not included (``doc/source/guide_benchmarking.rst``, "Expensive-track
baselines"): HEBO (0.3.6 pins ``numpy<1.25`` and ``pymoo==0.6.0``), PDFO
(wheels only up to CPython 3.12; the source build needs a Fortran
toolchain), Ax (BoTorch is used directly with the tutorial setup; Ax adds
plotly, ipywidgets, pymoo and graphviz, and its own default, qLogNEI on the
same ``SingleTaskGP``, differs from qLogEI only in handling noise).

Conventions on top of those of :mod:`panobbgo.harness_baselines`
------------------------------------------------------------------

* **Coordinates.**  Every adapter works in the unit cube mapped affinely
  onto the box (BoTorch, TuRBO; SMAC's ConfigSpace does the same
  internally; Py-BOBYQA with ``scaling_within_bounds=True``).
* **Outputs.**  Each library's own standardisation: ``SingleTaskGP``'s
  default ``Standardize`` transform, TuRBO's explicit standardisation,
  SMAC's ``normalize_y=True``.  No other transformation; a huge finite
  value is modelled as it is.
* **Failed values.**  A NaN value (failed or timed-out call) is still the
  worst value, but a GP cannot take ``+inf``.  BoTorch and TuRBO replace
  every non-finite value by the **worst finite value observed so far**,
  recomputed at every fit.  SMAC stores costs, so a failure is told at
  tell time with the conservative ``worst + (worst - best)`` of the finite
  values known then, which ranks it below every real point known then (a
  later, worse value can still exceed it); a failure before any finite
  value exists waits as a running trial until one does.  Py-BOBYQA gets
  the "moderated extreme barrier" of Powell's solvers in PRIMA / PDFO:
  NaN and ``+inf`` become ``1e30`` (PRIMA's ``FUNCMAX``) and finite values
  are clipped there.
* **Batches and pending points.**  ``q`` is each tool's native batch
  mechanism: BoTorch optimises a joint ``q``-batch and passes asked but
  untold points as ``X_pending``; TuRBO proposes batches of ``q`` by
  Thompson sampling and, being synchronous, waits for a whole batch
  (``ask`` returns ``[]`` meanwhile, like pycma); SMAC is asked once per
  free worker, as its Dask runner does, and keeps asked trials as
  running.  The initial design grows to at least ``q`` points so that the
  first batch fills the workers.
* **Determinism.**  Every random source derives from the run seed: torch
  draws happen in a ``torch.random.fork_rng`` scope seeded from the
  adapter's generator (the global torch state is restored), Sobol engines
  get explicit seeds, SMAC gets ``Scenario(seed=...)``, Py-BOBYQA's thread
  seeds numpy's global RNG (restored by the driver after the run).
* **Wall time.**  GP fits dominate; the per-run cost is in the guide.
  The strategies set ``no_wall_timeout``, so ``benchmark_harness.py``'s
  per-run wall-clock timeout does not cut them.
"""

from __future__ import annotations

import contextlib
import logging
import math
import queue
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from panobbgo.harness_baselines import (
    AskTellAdapter,
    AskTellBaselineStrategy,
    _require,
    _seed32,
)

_logger = logging.getLogger(__name__)

#: The optional extra these baselines need.
BO_EXTRA = "baselines-bo"

#: PRIMA's ``FUNCMAX``: the value Powell's solvers see for a failed call.
FUNCMAX = 1e30

#: Modules the BoTorch adapters import.
_TORCH_MODULES: Tuple[str, ...] = ("torch", "gpytorch", "botorch")


def _req(module: str) -> Any:
    return _require(module, extra=BO_EXTRA)


def impute_worst(fx: np.ndarray) -> np.ndarray:
    """Replace every non-finite entry of ``fx`` (minimisation values) by the worst finite one.

    Raises:
        ValueError: ``fx`` has no finite entry.
    """
    fx = np.asarray(fx, dtype=np.float64)
    finite = np.isfinite(fx)
    if not finite.any():
        raise ValueError("no finite value to impute from")
    out = fx.copy()
    out[~finite] = fx[finite].max()
    return out


def moderated_extreme_barrier(fx: float) -> float:
    """PRIMA's rule for Powell's solvers: NaN → ``FUNCMAX``, values clipped to ``[-FUNCMAX, FUNCMAX]``."""
    fx = float(fx)
    if math.isnan(fx):
        return FUNCMAX
    return max(-FUNCMAX, min(FUNCMAX, fx))


class _UnitBox:
    """Affine map between the unit cube and the problem box."""

    def __init__(self, lo: np.ndarray, hi: np.ndarray) -> None:
        self.lo = np.asarray(lo, dtype=np.float64)
        self.span = np.asarray(hi, dtype=np.float64) - self.lo
        self.dim = int(self.lo.size)

    def to_box(self, u: np.ndarray) -> np.ndarray:
        return self.lo + np.clip(np.asarray(u, dtype=np.float64), 0.0, 1.0) * self.span


@contextlib.contextmanager
def _torch_seeded(torch: Any, rng: np.random.Generator) -> Iterator[None]:
    """Run the block with torch's global CPU RNG seeded from ``rng``; restore it afterwards."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(rng.integers(0, 2**63 - 1)))
        yield


def _sobol(torch: Any, dim: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` scrambled Sobol points in the unit cube (an explicitly seeded engine)."""
    engine = torch.quasirandom.SobolEngine(dimension=dim, scramble=True, seed=int(rng.integers(0, 2**31 - 1)))
    return engine.draw(n, dtype=torch.float64).numpy().copy()


def _fit(botorch: Any, gpytorch: Any, model: Any) -> None:
    """``fit_gpytorch_mll`` on the exact marginal log likelihood.

    If every retry of BoTorch's fitting routine fails (``ModelFittingError``,
    rare: near-duplicate points), the model keeps its prior hyperparameters
    rather than ending the run.
    """
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    try:
        botorch.fit.fit_gpytorch_mll(mll)
    except botorch.exceptions.errors.ModelFittingError as exc:
        _logger.debug("GP fit failed (%s); using the prior hyperparameters", exc)


# -- BoTorch qLogEI --------------------------------------------------------------


class _BoTorchQLogEIAdapter(AskTellAdapter):
    """Batch ``qLogExpectedImprovement`` on a ``SingleTaskGP`` (BoTorch tutorial settings).

    * Initial design: ``max(5, 2·d)`` scrambled Sobol points (Ax's rule for
      the Sobol step), at least ``q``, at most the budget.  As in Ax, the
      model takes over only once ``max(2, ceil(n_init / 2))`` points are
      observed (one of them finite); until then free workers get more
      points of the same Sobol sequence.
    * Model: ``SingleTaskGP`` with its defaults (BoTorch ≥ 0.12: RBF kernel
      with dimension-scaled log-normal length-scale prior, ``Standardize``
      outcome transform) on inputs in the unit cube, fitted by
      ``fit_gpytorch_mll`` before every proposal.  Values are negated
      (BoTorch maximises); failures are imputed with the worst finite
      value.
    * Acquisition: ``qLogExpectedImprovement(best_f = best observed)`` with
      a 256-sample ``SobolQMCNormalSampler`` and the asked but untold
      points as ``X_pending``; ``optimize_acqf(q=n, num_restarts=10,
      raw_samples=512, options={"batch_limit": 5, "maxiter": 200})``.
    """

    requires = _TORCH_MODULES

    def __init__(self, lo: np.ndarray, hi: np.ndarray, seed: int, budget: int, batch_size: int = 1) -> None:
        self._torch = _req("torch")
        self._gpytorch = _req("gpytorch")
        self._botorch = _req("botorch")
        for sub in ("fit", "models", "acquisition.logei", "optim", "sampling.normal", "exceptions.errors"):
            _req(f"botorch.{sub}")
        _req("gpytorch.mlls")
        self._box = _UnitBox(lo, hi)
        self._rng = np.random.default_rng(_seed32(seed))
        d = self._box.dim
        self.n_init = int(min(max(1, budget), max(5, 2 * d, int(batch_size))))
        #: Observations the model needs before it proposes (Ax: half the design).
        self.min_observed = max(2, math.ceil(self.n_init / 2))
        # One scrambled Sobol sequence per run: the initial design, then continuation points.
        self._engine = self._torch.quasirandom.SobolEngine(
            dimension=d, scramble=True, seed=int(self._rng.integers(0, 2**31 - 1))
        )
        self._next_key = 0
        self._pending: Dict[int, np.ndarray] = {}
        self._X: List[np.ndarray] = []
        self._f: List[float] = []

    def _next_design_point(self) -> np.ndarray:
        return self._engine.draw(1, dtype=self._torch.float64).numpy()[0].copy()

    def _hand_out(self, u: np.ndarray, out: List[Tuple[int, np.ndarray]]) -> None:
        key = self._next_key
        self._next_key += 1
        self._pending[key] = np.asarray(u, dtype=np.float64)
        out.append((key, self._box.to_box(u)))

    def _can_model(self) -> bool:
        return len(self._f) >= self.min_observed and bool(np.isfinite(self._f).any())

    def ask(self, n: int) -> List[Tuple[int, np.ndarray]]:
        out: List[Tuple[int, np.ndarray]] = []
        while len(out) < n and self._next_key < self.n_init:
            self._hand_out(self._next_design_point(), out)
        m = n - len(out)
        if m <= 0:
            return out
        if self._can_model():
            for u in self._propose(m):
                self._hand_out(u, out)
        else:
            for _ in range(m):
                self._hand_out(self._next_design_point(), out)
        return out

    def _propose(self, m: int) -> np.ndarray:
        torch, botorch = self._torch, self._botorch
        X = torch.tensor(np.asarray(self._X), dtype=torch.float64)
        Y = torch.tensor(-impute_worst(np.asarray(self._f)), dtype=torch.float64).unsqueeze(-1)
        pending = torch.tensor(np.asarray(list(self._pending.values())), dtype=torch.float64) if self._pending else None
        bounds = torch.stack(
            [torch.zeros(self._box.dim, dtype=torch.float64), torch.ones(self._box.dim, dtype=torch.float64)]
        )
        with _torch_seeded(torch, self._rng):
            model = botorch.models.SingleTaskGP(X, Y)
            _fit(botorch, self._gpytorch, model)
            acq = botorch.acquisition.logei.qLogExpectedImprovement(
                model,
                best_f=Y.max(),
                sampler=botorch.sampling.normal.SobolQMCNormalSampler(sample_shape=torch.Size([256])),
                X_pending=pending,
            )
            cand, _ = botorch.optim.optimize_acqf(
                acq,
                bounds=bounds,
                q=m,
                num_restarts=10,
                raw_samples=512,
                options={"batch_limit": 5, "maxiter": 200},
            )
        return cand.detach().numpy().copy()

    def tell(self, key: int, fx: float) -> None:
        self._X.append(self._pending.pop(key))
        self._f.append(float(fx))


class BoTorchQLogEIStrategy(AskTellBaselineStrategy):
    """BoTorch batch ``qLogEI`` on a standard ``SingleTaskGP`` (see :class:`_BoTorchQLogEIAdapter`)."""

    who: str = "BoTorch_qLogEI"
    requires = _BoTorchQLogEIAdapter.requires
    extra: str = BO_EXTRA
    no_wall_timeout: bool = True

    def make_adapter(self, seed: int, budget: int, batch_size: int) -> AskTellAdapter:
        box = self.problem.box
        return _BoTorchQLogEIAdapter(box[:, 0], box[:, 1], seed, budget, batch_size)


# -- TuRBO-1 ---------------------------------------------------------------------


class _TurboState:
    """The trust-region state of the BoTorch TuRBO-1 tutorial (``TurboState`` / ``update_state``).

    ``success_tolerance = 3`` as in Eriksson et al. (2019) and the
    uber-research reference code (the BoTorch tutorial uses 10, with which
    the region practically never expands at 10…200·d evaluations);
    ``failure_tolerance = ceil(max(4/q, d/q))``; length 0.8 in
    ``[0.5**7, 1.6]``.  ``best_value`` starts at the best value of the run's
    initial design.
    """

    def __init__(self, dim: int, batch_size: int, best_value: float) -> None:
        self.length = 0.8
        self.length_min = 0.5**7
        self.length_max = 1.6
        self.failure_counter = 0
        self.failure_tolerance = int(math.ceil(max(4.0 / batch_size, float(dim) / batch_size)))
        self.success_counter = 0
        self.success_tolerance = 3
        self.best_value = float(best_value)
        self.restart_triggered = False

    def update(self, y_next_max: float) -> None:
        """``update_state`` of the tutorial, for a batch whose best (maximised) value is ``y_next_max``."""
        if y_next_max > self.best_value + 1e-3 * math.fabs(self.best_value):
            self.success_counter += 1
            self.failure_counter = 0
        else:
            self.success_counter = 0
            self.failure_counter += 1
        if self.success_counter == self.success_tolerance:
            self.length = min(2.0 * self.length, self.length_max)
            self.success_counter = 0
        elif self.failure_counter == self.failure_tolerance:
            self.length /= 2.0
            self.failure_counter = 0
        self.best_value = max(self.best_value, y_next_max)
        if self.length < self.length_min:
            self.restart_triggered = True


class _TurboAdapter(AskTellAdapter):
    """TuRBO-1 after the BoTorch tutorial (Eriksson et al. 2019), batch size ``q``.

    * Each trust-region run starts with ``max(2·d, q)`` scrambled Sobol
      points (the tutorial's ``2·d``, at least one batch).  When the
      region collapses (``length < 0.5**7``) the run restarts with a fresh
      design and fresh data, as in the original TuRBO-1.
    * Model per batch: ``SingleTaskGP`` with ``ScaleKernel(MaternKernel(
      nu=2.5, ARD, lengthscale in [0.005, 4]))`` and a Gaussian likelihood
      with noise in ``[1e-8, 1e-3]``, on the run's data standardised,
      fitted by ``fit_gpytorch_mll``; failures are imputed with the run's
      worst finite value.
    * Candidates: Thompson sampling (``MaxPosteriorSampling``, no
      replacement) over ``min(5000, max(2000, 200·d))`` Sobol perturbations
      of the best point inside the length-scale-weighted trust region,
      each coordinate perturbed with probability ``min(20/d, 1)``.
    * Synchronous batches: the next batch is proposed only when the whole
      current one is told; ``ask`` returns ``[]`` meanwhile.
    """

    requires = _TORCH_MODULES

    def __init__(self, lo: np.ndarray, hi: np.ndarray, seed: int, budget: int, batch_size: int = 1) -> None:
        del budget
        self._torch = _req("torch")
        self._gpytorch = _req("gpytorch")
        self._botorch = _req("botorch")
        for sub in ("fit", "models", "generation", "exceptions.errors"):
            _req(f"botorch.{sub}")
        for sub in ("mlls", "kernels", "likelihoods", "constraints", "settings"):
            _req(f"gpytorch.{sub}")
        self._box = _UnitBox(lo, hi)
        self._rng = np.random.default_rng(_seed32(seed))
        self.batch_size = max(1, int(batch_size))
        self.n_init = max(2 * self._box.dim, self.batch_size)
        self.restarts = 0
        self.state: Optional[_TurboState] = None
        self._X: List[np.ndarray] = []  # the current trust-region run's data
        self._f: List[float] = []
        self._next_key = 0
        self._gen: Dict[int, np.ndarray] = {}  # key -> unit x of the current generation
        self._gen_f: Dict[int, float] = {}
        self._undispatched: List[int] = []
        self._gen_is_init = True

    def _new_generation(self) -> None:
        if self.state is None or not self._X:
            points = _sobol(self._torch, self._box.dim, self.n_init, self._rng)
            self._gen_is_init = True
        else:
            points = self._turbo_batch()
            self._gen_is_init = False
        self._gen = {}
        for u in points:
            self._gen[self._next_key] = np.asarray(u, dtype=np.float64)
            self._next_key += 1
        self._gen_f = {}
        self._undispatched = list(self._gen)

    def _turbo_batch(self) -> np.ndarray:
        torch, gpytorch, botorch = self._torch, self._gpytorch, self._botorch
        assert self.state is not None
        dim = self._box.dim
        X = torch.tensor(np.asarray(self._X), dtype=torch.float64)
        y = -impute_worst(np.asarray(self._f))
        std = y.std()
        Y = torch.tensor((y - y.mean()) / (std if std > 0 else 1.0), dtype=torch.float64).unsqueeze(-1)
        with _torch_seeded(torch, self._rng):
            likelihood = gpytorch.likelihoods.GaussianLikelihood(
                noise_constraint=gpytorch.constraints.Interval(1e-8, 1e-3)
            )
            covar = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(
                    nu=2.5, ard_num_dims=dim, lengthscale_constraint=gpytorch.constraints.Interval(0.005, 4.0)
                )
            )
            model = botorch.models.SingleTaskGP(X, Y, covar_module=covar, likelihood=likelihood)
            with gpytorch.settings.max_cholesky_size(float("inf")):
                _fit(botorch, gpytorch, model)
                x_center = X[Y.argmax(), :].clone()
                weights = model.covar_module.base_kernel.lengthscale.detach().reshape(-1)
                weights = weights / weights.mean()
                weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
                tr_lb = torch.clamp(x_center - weights * self.state.length / 2.0, 0.0, 1.0)
                tr_ub = torch.clamp(x_center + weights * self.state.length / 2.0, 0.0, 1.0)
                n_cand = min(5000, max(2000, 200 * dim))
                sobol = torch.quasirandom.SobolEngine(dim, scramble=True, seed=int(self._rng.integers(0, 2**31 - 1)))
                pert = tr_lb + (tr_ub - tr_lb) * sobol.draw(n_cand, dtype=torch.float64)
                prob_perturb = min(20.0 / dim, 1.0)
                mask = torch.rand(n_cand, dim, dtype=torch.float64) <= prob_perturb
                ind = torch.where(mask.sum(dim=1) == 0)[0]
                mask[ind, torch.randint(0, dim, size=(len(ind),))] = True
                X_cand = x_center.expand(n_cand, dim).clone()
                X_cand[mask] = pert[mask]
                sampler = botorch.generation.MaxPosteriorSampling(model=model, replacement=False)
                with torch.no_grad():
                    X_next = sampler(X_cand, num_samples=self.batch_size)
        return X_next.detach().numpy().copy()

    def ask(self, n: int) -> List[Tuple[int, np.ndarray]]:
        out: List[Tuple[int, np.ndarray]] = []
        while len(out) < n:
            if not self._undispatched:
                if self._gen:
                    break  # the current batch still waits on results
                self._new_generation()
            key = self._undispatched.pop(0)
            out.append((key, self._box.to_box(self._gen[key])))
        return out

    def tell(self, key: int, fx: float) -> None:
        if key not in self._gen or key in self._gen_f:
            raise KeyError(f"candidate {key} is not an open point of the current batch")
        self._gen_f[key] = float(fx)
        if len(self._gen_f) < len(self._gen):
            return
        keys = list(self._gen)
        self._X.extend(self._gen[k] for k in keys)
        self._f.extend(self._gen_f[k] for k in keys)
        batch_f = np.asarray([self._gen_f[k] for k in keys])
        self._gen, self._gen_f = {}, {}
        if not np.isfinite(self._f).any():
            self._restart()  # nothing finite yet: another design
            return
        imputed = impute_worst(np.asarray(self._f))
        if self._gen_is_init or self.state is None:
            self.state = _TurboState(self._box.dim, self.batch_size, best_value=float(-imputed.min()))
            return
        worst = imputed.max()
        batch_best = float(-np.nanmin(np.where(np.isfinite(batch_f), batch_f, worst)))
        self.state.update(batch_best)
        if self.state.restart_triggered:
            self._restart()

    def _restart(self) -> None:
        if self.state is not None:
            self.restarts += 1
        self.state = None
        self._X, self._f = [], []


class BoTorchTuRBOStrategy(AskTellBaselineStrategy):
    """TuRBO-1 after the BoTorch tutorial (see :class:`_TurboAdapter`)."""

    who: str = "TuRBO1"
    requires = _TurboAdapter.requires
    extra: str = BO_EXTRA
    no_wall_timeout: bool = True

    def make_adapter(self, seed: int, budget: int, batch_size: int) -> AskTellAdapter:
        box = self.problem.box
        return _TurboAdapter(box[:, 0], box[:, 1], seed, budget, batch_size)


# -- SMAC3 -----------------------------------------------------------------------


_SMAC_QUIETED = False


def _quiet_smac() -> None:
    """SMAC logs INFO lines per trial under the ``smac`` logger; lower it to WARNING once per process."""
    global _SMAC_QUIETED
    if not _SMAC_QUIETED:
        logging.getLogger("smac").setLevel(logging.WARNING)
        _SMAC_QUIETED = True


class _SMACAdapter(AskTellAdapter):
    """SMAC3's ``BlackBoxFacade`` driven through its ask/tell interface.

    * Scenario: one ``Float`` hyperparameter per coordinate (ConfigSpace
      normalises to the unit cube), ``deterministic=True`` (one evaluation
      per configuration; the benchmark objectives are deterministic),
      ``n_trials = budget``, ``seed``.  Everything else is the facade's
      default: Sobol initial design of ``min(8·d, budget/4)`` points, a GP
      with a Matérn-5/2 ARD kernel and ``normalize_y=True``, EI with
      ``xi=0``, local-and-sorted random search over 1000 challengers,
      random interleaving with probability 0.085.
    * ``q > 1``: one ``ask`` per free worker, as SMAC's Dask runner does;
      asked trials are running trials to SMAC, which has no pending-point
      or fantasy handling, so a batch may hold near-duplicates (native
      behaviour; report SMAC at ``q = 1``).
    * Failed values: told as CRASHED with :meth:`failure_cost`.
    * SMAC writes its run files into a temporary directory removed by
      :meth:`close`; ``logging_level=False`` leaves the host's logging
      configuration alone.
    """

    requires = ("smac", "ConfigSpace")

    def __init__(self, lo: np.ndarray, hi: np.ndarray, seed: int, budget: int) -> None:
        smac = _req("smac")
        cs_mod = _req("ConfigSpace")
        self._dataclasses = _req("smac.runhistory.dataclasses")
        self._status = _req("smac.runhistory.enumerations").StatusType
        _quiet_smac()
        self._names = [f"x{j}" for j in range(len(lo))]
        space = cs_mod.ConfigurationSpace(seed=_seed32(seed))
        for name, lo_j, hi_j in zip(self._names, lo, hi):
            space.add(cs_mod.UniformFloatHyperparameter(name, lower=float(lo_j), upper=float(hi_j)))
        self._tmp = tempfile.mkdtemp(prefix="panobbgo-smac-")
        scenario = smac.Scenario(
            space,
            deterministic=True,
            n_trials=int(budget),
            seed=_seed32(seed) % (2**31 - 1),
            output_directory=Path(self._tmp) / "smac",
        )
        self._smac = smac.BlackBoxFacade(scenario, target_function=None, logging_level=False, overwrite=True)
        self._next_key = 0
        self._pending: Dict[int, Any] = {}
        self._deferred: List[Any] = []  # failed trials waiting for a first finite value
        self._worst: Optional[float] = None
        self._best: Optional[float] = None

    def failure_cost(self) -> float:
        """The cost a failed trial is told with: ``worst + (worst - best)`` of the finite values so far."""
        assert self._worst is not None and self._best is not None
        return self._worst + (self._worst - self._best)

    def ask(self, n: int) -> List[Tuple[int, np.ndarray]]:
        out: List[Tuple[int, np.ndarray]] = []
        for _ in range(n):
            info = self._smac.ask()
            key = self._next_key
            self._next_key += 1
            self._pending[key] = info
            out.append((key, np.array([float(info.config[k]) for k in self._names], dtype=np.float64)))
        return out

    def _tell(self, info: Any, cost: float, status: Any) -> None:
        self._smac.tell(info, self._dataclasses.TrialValue(cost=float(cost), status=status), save=False)

    def tell(self, key: int, fx: float) -> None:
        info = self._pending.pop(key)
        fx = float(fx)
        if math.isfinite(fx):
            self._worst = fx if self._worst is None else max(self._worst, fx)
            self._best = fx if self._best is None else min(self._best, fx)
            self._tell(info, fx, self._status.SUCCESS)
            for failed in self._deferred:
                self._tell(failed, self.failure_cost(), self._status.CRASHED)
            self._deferred = []
        elif self._worst is None:
            self._deferred.append(info)
        else:
            self._tell(info, self.failure_cost(), self._status.CRASHED)

    def close(self) -> None:
        shutil.rmtree(self._tmp, ignore_errors=True)


class SMACBlackBoxStrategy(AskTellBaselineStrategy):
    """SMAC3 ``BlackBoxFacade`` (GP-based BO; see :class:`_SMACAdapter`)."""

    who: str = "SMAC_BB"
    requires = _SMACAdapter.requires
    extra: str = BO_EXTRA
    no_wall_timeout: bool = True

    def make_adapter(self, seed: int, budget: int, batch_size: int) -> AskTellAdapter:
        del batch_size
        box = self.problem.box
        return _SMACAdapter(box[:, 0], box[:, 1], seed, budget)


# -- Py-BOBYQA -------------------------------------------------------------------


class _StopBOBYQA(Exception):
    """Raised inside Py-BOBYQA's objective to end its thread (``close``)."""


class _PyBOBYQAAdapter(AskTellAdapter):
    """Py-BOBYQA (Cartis et al.; Powell's BOBYQA) inverted into ask/tell with a worker thread.

    ``pybobyqa.solve`` owns its loop, so it runs in a daemon thread whose
    objective hands each point to :meth:`ask` and blocks until
    :meth:`tell`; the two threads strictly alternate, so a run is
    deterministic.  Settings are Py-BOBYQA's defaults for a noiseless
    bounded problem: ``npt = 2n+1``, ``scaling_within_bounds=True`` (its
    recommendation for finite bounds; ``rhobeg = 0.1`` of the box),
    ``rhoend = 1e-8``, no restarts inside the solver, ``maxfun`` = the
    remaining budget.  When a run terminates before the budget (converged,
    or slow progress), the adapter starts a new one from a fresh uniform
    random point.  Failed values: :func:`moderated_extreme_barrier`.

    **Sequential**: at most one point is in flight; ``ask(n)`` returns one
    point, or ``[]`` while it waits for a result.  A **local** reference:
    Py-BOBYQA's ``seek_global_minimum=True`` mode is not wrapped (TODO).
    """

    requires = ("pybobyqa",)

    def __init__(self, lo: np.ndarray, hi: np.ndarray, seed: int, budget: int) -> None:
        self._pybobyqa = _req("pybobyqa")
        self._lo = np.asarray(lo, dtype=np.float64)
        self._hi = np.asarray(hi, dtype=np.float64)
        if not np.all(self._hi > self._lo):
            # scaling_within_bounds divides by hi - lo: Py-BOBYQA would probe NaN points.
            raise ValueError("Py-BOBYQA needs a box with hi > lo in every coordinate")
        self._rng = np.random.default_rng(_seed32(seed))
        self._budget = int(budget)
        self._asked = 0
        self.runs = 0
        self._to_main: "queue.Queue[Tuple[str, Any]]" = queue.Queue()
        self._to_solver: "queue.Queue[Optional[float]]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._pending_key: Optional[int] = None
        self._run_points = 0  # points the current run has handed out

    def _solver(self, x0: np.ndarray, maxfun: int, np_seed: int) -> None:
        def objfun(x: np.ndarray) -> float:
            self._to_main.put(("x", np.asarray(x, dtype=np.float64).copy()))
            fx = self._to_solver.get()
            if fx is None:
                raise _StopBOBYQA()
            return fx

        try:
            np.random.seed(np_seed)
            self._pybobyqa.solve(
                objfun,
                x0,
                bounds=(self._lo, self._hi),
                maxfun=maxfun,
                scaling_within_bounds=True,
                do_logging=False,
                print_progress=False,
            )
            self._to_main.put(("done", None))
        except _StopBOBYQA:
            pass
        except Exception as exc:  # noqa: BLE001 — handed to the main thread
            self._to_main.put(("error", exc))

    def _start_run(self) -> None:
        x0 = self._rng.uniform(self._lo, self._hi)
        maxfun = max(1, self._budget - self._asked)
        self._thread = threading.Thread(
            target=self._solver,
            args=(x0, maxfun, int(self._rng.integers(0, 2**32))),
            name=f"pybobyqa-run-{self.runs}",
            daemon=True,
        )
        self.runs += 1
        self._run_points = 0
        self._thread.start()

    def ask(self, n: int) -> List[Tuple[int, np.ndarray]]:
        if n <= 0 or self._pending_key is not None:
            return []
        while True:
            if self._thread is None:
                self._start_run()
            kind, payload = self._to_main.get()
            if kind == "x":
                break
            assert self._thread is not None
            self._thread.join()
            self._thread = None
            if kind == "error":
                raise payload
            if self._run_points == 0:
                # A run that ends without evaluating anything (Py-BOBYQA's
                # EXIT_INPUT_ERROR, e.g. a degenerate box lo == hi) would do
                # the same on every restart.
                raise RuntimeError("Py-BOBYQA ended a run without evaluating a point (input error?)")
        key = self._asked
        self._asked += 1
        self._run_points += 1
        self._pending_key = key
        return [(key, np.clip(payload, self._lo, self._hi))]

    def tell(self, key: int, fx: float) -> None:
        if key != self._pending_key:
            raise KeyError(f"candidate {key} is not the pending point")
        self._pending_key = None
        self._to_solver.put(moderated_extreme_barrier(fx))

    def close(self) -> None:
        thread, self._thread = self._thread, None
        if thread is not None and thread.is_alive():
            self._to_solver.put(None)
            thread.join(timeout=10)


class PyBOBYQAStrategy(AskTellBaselineStrategy):
    """Py-BOBYQA, the sequential local model-based reference (see :class:`_PyBOBYQAAdapter`)."""

    who: str = "PyBOBYQA"
    requires = _PyBOBYQAAdapter.requires
    extra: str = BO_EXTRA
    no_wall_timeout: bool = True

    def make_adapter(self, seed: int, budget: int, batch_size: int) -> AskTellAdapter:
        del batch_size
        box = self.problem.box
        return _PyBOBYQAAdapter(box[:, 0], box[:, 1], seed, budget)


#: The expensive-track baseline classes, in registry order.
BO_BASELINE_CLASSES: Tuple[type, ...] = (
    BoTorchQLogEIStrategy,
    BoTorchTuRBOStrategy,
    SMACBlackBoxStrategy,
    PyBOBYQAStrategy,
)


__all__ = [
    "BO_BASELINE_CLASSES",
    "BO_EXTRA",
    "FUNCMAX",
    "BoTorchQLogEIStrategy",
    "BoTorchTuRBOStrategy",
    "PyBOBYQAStrategy",
    "SMACBlackBoxStrategy",
    "impute_worst",
    "moderated_extreme_barrier",
]
