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
CMA-ES Heuristic
================

Covariance Matrix Adaptation Evolution Strategy (CMA-ES).

CMA-ES is the gold-standard algorithm for derivative-free optimization of
continuous, possibly multimodal functions.  It maintains a multivariate
Gaussian search distribution N(m, σ²C) and adapts both the step-size σ
and the full covariance matrix C from the history of successful steps.

Key strengths:
- Invariant under order-preserving transformations of the objective
- Invariant under orthogonal transformations of the search space
- Excellent at following narrow ridges / valleys (e.g. Rosenbrock)
- Self-adaptive: no manual step-size tuning required

This implementation follows the canonical description in:
  N. Hansen (2016). "The CMA Evolution Strategy: A Tutorial."
  arXiv:1604.00772.

The heuristic works asynchronously inside the panobbgo event loop:

1. ``on_start()``  — initialise parameters and emit the first generation of λ candidates.
2. ``on_new_results()``  — collect returned results tagged with our generation ID.
   When at least μ results from the oldest open generation have arrived, perform
   one CMA-ES update step and emit the next generation.

.. codeauthor:: Harald Schilly
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from panobbgo.core import Heuristic
from panobbgo.lib import Point


class CMAES(Heuristic):
    """Covariance Matrix Adaptation Evolution Strategy heuristic.

    Maintains a multivariate Gaussian search distribution N(m, σ²C) and
    adapts it from the history of evaluated points.  Particularly effective
    for functions with elongated or ill-conditioned level sets (e.g. Rosenbrock).

    Args:
        strategy: The optimization strategy instance.
        sigma0 (float, optional): Initial step size as a fraction of the mean
            box half-range.  Defaults to 0.3 (30 % of the search space).
        popsize (int, optional): Override the default population size
            ``λ = 4 + floor(3·ln n)``.  Useful for low-budget runs.
        min_results_fraction (float): Fraction of λ that must arrive before
            a CMA-ES update is triggered.  Default 0.5 (= μ).
    """

    def __init__(
        self,
        strategy,
        sigma0: float = 0.3,
        popsize: Optional[int] = None,
        min_results_fraction: float = 0.5,
    ):
        super().__init__(strategy, name="CMAES")
        self.logger = self.config.get_logger("H:CMA")
        self._sigma0_frac = sigma0
        self._popsize_override = popsize
        self._min_results_fraction = min_results_fraction

        # CMA-ES algorithm state (set in on_start)
        self._m: Optional[np.ndarray] = None
        self._sigma: float = 1.0
        self._C: Optional[np.ndarray] = None
        self._p_c: Optional[np.ndarray] = None
        self._p_sigma: Optional[np.ndarray] = None
        self._B: Optional[np.ndarray] = None
        self._D: Optional[np.ndarray] = None
        self._eigeneval: int = 0
        self._counteval: int = 0

        # Strategy parameters (set in on_start)
        self._lam: int = 0
        self._mu: int = 0
        self._w: Optional[np.ndarray] = None
        self._mu_eff: float = 1.0
        self._c_sigma: float = 0.0
        self._d_sigma: float = 1.0
        self._c_c: float = 0.0
        self._c_1: float = 0.0
        self._c_mu: float = 0.0
        self._chi_n: float = 1.0

        # Generation tracking for async operation
        self._gen: int = 0
        self._pending: Dict[str, dict] = {}
        self._gen_results: Dict[int, List[dict]] = {}

        # Box bounds (set in on_start)
        self._lo: Optional[np.ndarray] = None
        self._hi: Optional[np.ndarray] = None
        self._ranges: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_start(self) -> None:
        """Initialise CMA-ES state and emit the first generation."""
        n = self.problem.dim

        # Population sizes
        lam = self._popsize_override or (4 + int(3 * np.log(max(n, 2))))
        mu = lam // 2

        # Recombination weights (log-linear, positive)
        raw_w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1, dtype=float))
        w = raw_w / raw_w.sum()

        # Effective number of parents
        mu_eff = 1.0 / (w**2).sum()

        # Step-size control
        c_sigma = (mu_eff + 2.0) / (n + mu_eff + 5.0)
        d_sigma = (
            1.0
            + 2.0 * max(0.0, np.sqrt((mu_eff - 1.0) / (n + 1.0)) - 1.0)
            + c_sigma
        )

        # Covariance matrix control
        c_c = (4.0 + mu_eff / n) / (n + 4.0 + 2.0 * mu_eff / n)
        c_1 = 2.0 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(
            1.0 - c_1,
            2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((n + 2.0) ** 2 + mu_eff),
        )

        # Expected norm of N(0,I)
        chi_n = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n**2))

        # Search-space bounds
        box = self.problem.box.box
        lo = box[:, 0]
        hi = box[:, 1]
        ranges = hi - lo

        # Initial mean: box centre
        m = 0.5 * (lo + hi)

        # Initial step size: fraction of mean box half-range
        sigma = self._sigma0_frac * float(np.mean(ranges) / 2.0)
        sigma = max(sigma, 1e-6)

        # Persist
        self._lam = lam
        self._mu = mu
        self._w = w
        self._mu_eff = mu_eff
        self._c_sigma = c_sigma
        self._d_sigma = d_sigma
        self._c_c = c_c
        self._c_1 = c_1
        self._c_mu = c_mu
        self._chi_n = chi_n

        self._lo = lo
        self._hi = hi
        self._ranges = ranges

        self._m = m
        self._sigma = sigma
        self._C = np.eye(n)
        self._p_c = np.zeros(n)
        self._p_sigma = np.zeros(n)
        self._B = np.eye(n)
        self._D = np.ones(n)
        self._eigeneval = 0
        self._counteval = 0

        self._gen = 0
        self._pending = {}
        self._gen_results = {}

        self.logger.info(
            "CMA-ES started: n=%d λ=%d μ=%d σ0=%.4f", n, lam, mu, sigma
        )
        self._emit_generation()

    # ------------------------------------------------------------------
    # Event handler
    # ------------------------------------------------------------------

    def on_new_results(self, results) -> None:
        """Collect results and trigger a CMA-ES update when enough have arrived."""
        for r in results:
            if not r.who.startswith("CMAES:"):
                continue
            if r.fx is None or not np.isfinite(r.fx):
                continue

            info = self._pending.pop(r.who, None)
            if info is None:
                continue

            gen = info["gen"]
            gen_bucket = self._gen_results.get(gen)
            if gen_bucket is None:
                continue

            penalty = self.strategy.constraint_handler.get_penalty_value(r)
            gen_bucket.append(
                {
                    "penalty": penalty,
                    "x": r.x.copy(),
                    "y": info["y"],
                }
            )

        # Check if we can perform an update for the oldest open generation.
        min_needed = max(2, int(self._lam * self._min_results_fraction))
        for gen in sorted(self._gen_results.keys()):
            bucket = self._gen_results[gen]
            if len(bucket) >= min_needed:
                self._update(bucket)
                del self._gen_results[gen]
                self._emit_generation()
                break

    # ------------------------------------------------------------------
    # Core CMA-ES update
    # ------------------------------------------------------------------

    def _emit_generation(self) -> None:
        """Sample λ candidates from N(m, σ²C) and emit them."""
        if self._m is None or self._B is None or self._D is None:
            return

        n = self.problem.dim
        self._gen += 1
        gen = self._gen
        self._gen_results[gen] = []

        for i in range(self._lam):
            z = np.random.randn(n)
            # y = B D z  →  covariance = B D² Bᵀ = C
            y = self._B @ (self._D * z)
            x = self._m + self._sigma * y
            x = self.problem.project(x)

            who = f"CMAES:g{gen}:i{i}"
            self._pending[who] = {"gen": gen, "i": i, "y": y}
            # Put directly to bypass emit()'s ndarray-only check,
            # preserving the custom 'who' tag needed for generation tracking.
            try:
                self._output.put_nowait(Point(x, who))
            except Exception:
                pass

    def _update(self, collected: List[dict]) -> None:
        """Perform one CMA-ES parameter update from a set of evaluated offspring."""
        assert self._m is not None
        assert self._C is not None
        assert self._p_c is not None
        assert self._p_sigma is not None
        assert self._B is not None
        assert self._D is not None
        assert self._w is not None
        assert self._ranges is not None

        # Local non-None aliases (avoid repeated attribute-narrowing loss).
        C = self._C
        p_c = self._p_c
        p_sigma = self._p_sigma
        B = self._B
        D = self._D
        w_full = self._w
        ranges = self._ranges

        n = self.problem.dim

        # Sort by penalty (ascending = minimise)
        collected.sort(key=lambda d: d["penalty"])
        selected = collected[: self._mu]

        # Recombination weights (may use fewer than μ if fewer arrived)
        actual_mu = len(selected)
        if actual_mu < len(w_full):
            # Re-normalise weights for the actual number of survivors
            raw = np.log(actual_mu + 0.5) - np.log(
                np.arange(1, actual_mu + 1, dtype=float)
            )
            w = raw / raw.sum()
            mu_eff = 1.0 / (w**2).sum()
        else:
            w = w_full
            mu_eff = self._mu_eff

        # --- Mean update ---
        self._m = sum(w[i] * d["x"] for i, d in enumerate(selected))  # type: ignore[assignment]

        # Step in normalised y-space (weighted recombination)
        y_w = sum(w[i] * d["y"] for i, d in enumerate(selected))

        # --- Step-size path update ---
        # C^{-1/2} y_w  =  B diag(1/D) Bᵀ y_w
        C_invsqrt_yw = B @ ((1.0 / D) * (B.T @ y_w))

        p_sigma = (1.0 - self._c_sigma) * p_sigma + np.sqrt(
            self._c_sigma * (2.0 - self._c_sigma) * mu_eff
        ) * C_invsqrt_yw

        # Stall indicator: is p_sigma still growing?
        norm_p_sigma = float(np.linalg.norm(p_sigma))
        gen_count = self._counteval / self._lam + 1.0
        expected = np.sqrt(1.0 - (1.0 - self._c_sigma) ** (2.0 * gen_count))
        h_sigma = norm_p_sigma / (expected * self._chi_n) < 1.4 + 2.0 / (n + 1.0)

        # --- Covariance path update ---
        p_c = (1.0 - self._c_c) * p_c + h_sigma * np.sqrt(
            self._c_c * (2.0 - self._c_c) * mu_eff
        ) * y_w

        # --- Covariance matrix update ---
        # Rank-μ term
        Y = np.column_stack([d["y"] for d in selected])  # (n, actual_mu)
        rank_mu = (Y * w[:actual_mu]) @ Y.T  # weighted outer-product sum

        # Correction for h_sigma = 0
        delta_h = (1.0 - h_sigma) * self._c_c * (2.0 - self._c_c)

        C = (
            (1.0 - self._c_1 - self._c_mu) * C
            + self._c_1 * (np.outer(p_c, p_c) + delta_h * C)
            + self._c_mu * rank_mu
        )

        # --- Step-size update (cumulative path length control) ---
        self._sigma *= float(
            np.exp(
                (self._c_sigma / self._d_sigma)
                * (norm_p_sigma / self._chi_n - 1.0)
            )
        )

        # Clamp step size
        max_sigma = float(np.mean(ranges))
        self._sigma = float(np.clip(self._sigma, 1e-12, max_sigma))

        self._counteval += actual_mu

        # Persist path/covariance state back to instance.
        self._p_c = p_c
        self._p_sigma = p_sigma
        self._C = C

        # --- Lazy eigendecomposition ---
        # Update frequency: every lam / (c_1 + c_mu) / n / 10 evaluations
        update_gap = max(1, int(self._lam / (self._c_1 + self._c_mu) / n / 10))
        if self._counteval - self._eigeneval >= update_gap:
            self._eigeneval = self._counteval
            # Enforce symmetry, then decompose
            C_sym = (C + C.T) / 2.0
            try:
                eigvals, B_new = np.linalg.eigh(C_sym)
            except np.linalg.LinAlgError:
                self.logger.warning("CMA-ES: eigendecomposition failed — resetting C")
                self._reset_covariance(n)
                return

            # Clamp eigenvalues (avoid collapse or explosion)
            eigvals = np.maximum(eigvals, 1e-20)
            D_new = np.sqrt(eigvals)
            self._B = B_new
            self._D = D_new
            self._C = C_sym

            # Condition-number guard
            if D_new.max() / D_new.min() > 1e7:
                self.logger.info("CMA-ES: condition number too large — resetting C")
                self._reset_covariance(n)

    def _reset_covariance(self, n: int) -> None:
        """Reset covariance to identity (recover from ill-conditioning)."""
        self._C = np.eye(n)
        self._p_c = np.zeros(n)
        self._p_sigma = np.zeros(n)
        self._B = np.eye(n)
        self._D = np.ones(n)
