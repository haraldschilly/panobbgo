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
CMA-ES Heuristic (with IPOP / BIPOP restart)
=============================================

Covariance Matrix Adaptation Evolution Strategy (CMA-ES) with optional
IPOP (Increasing Population) or BIPOP (Bi-Population) restart support.

CMA-ES is the gold-standard algorithm for derivative-free optimization of
continuous, possibly multimodal functions.  It maintains a multivariate
Gaussian search distribution N(m, σ²C) and adapts both the step-size σ
and the full covariance matrix C from the history of successful steps.

Key strengths:
- Invariant under order-preserving transformations of the objective
- Invariant under orthogonal transformations of the search space
- Excellent at following narrow ridges / valleys (e.g. Rosenbrock)
- Self-adaptive: no manual step-size tuning required

**IPOP restart** (Auger & Hansen, 2005): when the :class:`~panobbgo.analyzers.restart.Restart`
analyzer detects stagnation and fires a ``restart`` event, this heuristic:

1. Resets the search distribution to the new ``center`` provided by the analyzer.
2. Multiplies the population size λ → ``ipop_factor`` · λ (default doubling).
3. Resets σ to its initial fraction of the search-space range.
4. Flushes the pending/stale generation results.

**BIPOP restart** (Hansen, 2009): alternates between two restart regimes,
balancing the cumulative evaluation budget spent in each:

* **Large regime** (IPOP-like): population doubles every time it is selected;
  ``λ_l = 2^k · λ_default`` where ``k`` is the number of times the large
  regime has been selected so far.  σ is reset to its default value.
* **Small regime**: a small population
  ``λ_s = ⌊λ_default · (½ · λ_l/λ_default)^(U[0,1]²)⌋`` is used together
  with a tiny random step size ``σ_s = σ_default · 10^(-2·U[0,1])``.

After each restart, the regime that has consumed the *fewer* total
evaluations is selected next.  This is the strategy that won the BBOB-2009
benchmark and remains the de-facto gold standard for multimodal black-box
optimization.

This implementation follows:
  - N. Hansen (2016). "The CMA Evolution Strategy: A Tutorial." arXiv:1604.00772.
  - A. Auger & N. Hansen (2005). "A restart CMA evolution strategy with increasing
    population size." CEC 2005.
  - N. Hansen (2009). "Benchmarking a BI-Population CMA-ES on the BBOB-2009
    Function Testbed." GECCO Workshop on BBOB.

The heuristic works asynchronously inside the panobbgo event loop:

1. ``on_start()``  — initialise parameters and emit the first generation of λ candidates.
2. ``on_new_results()``  — collect returned results tagged with our generation ID.
   When at least μ results from the oldest open generation have arrived, perform
   one CMA-ES update step and emit the next generation.
3. ``on_restart(center, reason)``  — reset distribution to *center* with the
   restart regime selected by ``restart_mode`` (``"ipop"`` or ``"bipop"``).

.. codeauthor:: Harald Schilly
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from panobbgo.core import Heuristic
from panobbgo.lib import Point


class CMAES(Heuristic):
    """Covariance Matrix Adaptation Evolution Strategy heuristic with IPOP / BIPOP restart.

    Maintains a multivariate Gaussian search distribution N(m, σ²C) and
    adapts it from the history of evaluated points.  Particularly effective
    for functions with elongated or ill-conditioned level sets (e.g. Rosenbrock).

    When paired with the :class:`~panobbgo.analyzers.restart.Restart` analyzer,
    the heuristic implements either:

    * **IPOP-CMA-ES** (``restart_mode="ipop"``, default): each restart multiplies
      the population size by ``ipop_factor`` (default 2.0) and resets the search
      distribution to the new center.  Good for moderately multimodal problems.
    * **BIPOP-CMA-ES** (``restart_mode="bipop"``): alternates between a *large*
      regime (geometric population growth from the base λ) and a *small* regime
      (small λ with a random small σ).  The regime that has consumed *fewer*
      cumulative evaluations is selected next, balancing exploitation and
      exploration.  This is the BBOB-2009 winning strategy and the gold standard
      for highly multimodal problems with limited budget.

    Args:
        strategy: The optimization strategy instance.
        sigma0 (float, optional): Initial step size as a fraction of the mean
            box half-range.  Defaults to 0.3 (30 % of the search space).
        popsize (int, optional): Override the *base* population size
            ``λ = 4 + floor(3·ln n)``.  Useful for low-budget runs.
            On IPOP restarts the population is multiplied relative to the
            *current* λ; in BIPOP mode the large-regime population grows from
            this base value.
        min_results_fraction (float): Fraction of λ that must arrive before
            a CMA-ES update is triggered.  Default 0.5 (= μ).
        ipop_factor (float): Population multiplication factor applied on each
            IPOP restart.  Default 2.0 (standard IPOP doubling).  Ignored in
            BIPOP mode (which always doubles the large regime).
        restart_mode (str): Restart scheme selector — ``"ipop"`` (default) or
            ``"bipop"``.
    """

    def __init__(
        self,
        strategy,
        sigma0: float = 0.3,
        popsize: Optional[int] = None,
        min_results_fraction: float = 0.5,
        ipop_factor: float = 2.0,
        restart_mode: str = "ipop",
    ):
        super().__init__(strategy, name="CMAES")
        self.logger = self.config.get_logger("H:CMA")
        self._sigma0_frac = sigma0
        self._popsize_override = popsize
        self._min_results_fraction = min_results_fraction
        self._ipop_factor = float(ipop_factor)
        if restart_mode not in ("ipop", "bipop"):
            raise ValueError(f"restart_mode must be 'ipop' or 'bipop', got {restart_mode!r}")
        self._restart_mode = restart_mode

        # IPOP restart tracking
        self._restart_count: int = 0
        self._base_lam: int = 0  # λ at first on_start() — base for IPOP/BIPOP

        # BIPOP regime tracking (Hansen 2009)
        # _bipop_large_count: how many times the large regime has been selected
        #   (drives population growth λ_l = 2^k · λ_default)
        # _bipop_evals_large / _bipop_evals_small: cumulative evaluations spent
        #   in each regime — the regime with fewer evals is picked next
        # _bipop_current_regime: "large" or "small" (the regime *currently* running)
        # _bipop_regime_eval_anchor: counteval value when the current regime began,
        #   used to attribute evaluations on regime switch
        self._bipop_large_count: int = 0
        self._bipop_evals_large: int = 0
        self._bipop_evals_small: int = 0
        self._bipop_current_regime: str = "large"  # first run is always "large"
        self._bipop_regime_eval_anchor: int = 0

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
        # How many points of each generation actually made it into the output
        # queue — the update trigger must be based on this, not on λ, or a
        # partially-emitted generation deadlocks the heuristic.
        self._gen_emitted: Dict[int, int] = {}

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
        d_sigma = 1.0 + 2.0 * max(0.0, np.sqrt((mu_eff - 1.0) / (n + 1.0)) - 1.0) + c_sigma

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

        # Remember base population so IPOP doubling scales correctly
        if self._base_lam == 0:
            self._base_lam = lam

        # BIPOP: first run counts as the large regime, anchor at 0 evals
        self._bipop_current_regime = "large"
        self._bipop_regime_eval_anchor = 0

        self.logger.info(
            "CMA-ES started: n=%d λ=%d μ=%d σ0=%.4f mode=%s",
            n,
            lam,
            mu,
            sigma,
            self._restart_mode,
        )
        self._emit_generation()

    # ------------------------------------------------------------------
    # Event handlers
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
        # Base the trigger on the number of points actually emitted for that
        # generation, not on λ — if the output queue clipped the generation,
        # waiting for a λ-based quorum would deadlock.
        for gen in sorted(self._gen_results.keys()):
            bucket = self._gen_results[gen]
            emitted = self._gen_emitted.get(gen, self._lam)
            min_needed = max(2, int(min(self._lam, emitted) * self._min_results_fraction))
            if len(bucket) >= min_needed:
                self._update(bucket)
                del self._gen_results[gen]
                self._gen_emitted.pop(gen, None)
                self._emit_generation()
                break

    def on_restart(self, center: np.ndarray, reason: str = "") -> None:
        """Reset CMA-ES to *center* using the configured restart scheme.

        Called by the :class:`~panobbgo.analyzers.restart.Restart` analyzer
        when stagnation is detected.  Dispatches to the IPOP or BIPOP
        sub-routine according to ``restart_mode``.  Both schemes:

        - Move the search mean to *center* (a fresh region of the search space).
        - Recompute μ and all adaptation parameters for the new λ.
        - Reset evolution paths p_c, p_σ and covariance C to the identity.
        - Flush all stale pending and in-flight generation results.

        The IPOP scheme (Auger & Hansen, 2005) multiplies λ by ``ipop_factor``
        and resets σ to its initial fraction.  The BIPOP scheme (Hansen, 2009)
        alternates between a large regime (geometric λ growth from the base
        population) and a small regime (small λ + random small σ).

        Args:
            center (np.ndarray): New starting mean for the search distribution.
            reason (str): Human-readable reason string from the Restart analyzer.
        """
        if self._lo is None or self._ranges is None:
            # on_start() has not been called yet — ignore
            return

        if self._restart_mode == "bipop":
            self._restart_bipop(center, reason)
        else:
            self._restart_ipop(center, reason)

    def _restart_ipop(self, center: np.ndarray, reason: str) -> None:
        """IPOP restart: multiply λ by ``ipop_factor`` and reset σ."""
        new_lam = int(self._lam * self._ipop_factor)
        new_sigma = self._sigma0_default()
        prev_lam = self._lam
        self._apply_restart(center, new_lam, new_sigma)
        self.logger.info(
            "CMA-ES IPOP restart #%d: λ %d→%d  σ=%.4f  center=%s  (%s)",
            self._restart_count,
            prev_lam,
            new_lam,
            new_sigma,
            np.array2string(self._m, precision=3, suppress_small=True)  # type: ignore[arg-type]
            if self._m is not None
            else "?",
            reason or "stagnation",
        )

    def _restart_bipop(self, center: np.ndarray, reason: str) -> None:
        """BIPOP restart: alternate between large and small regimes (Hansen 2009).

        Attribution rule: evaluations consumed since the previous restart (or
        start) are credited to the regime that was *running*.  The regime that
        has accumulated the *fewer* total evaluations is selected next.  Ties
        are broken by selecting the large regime so that the population grows
        steadily over a long run (matches the BBOB-2009 reference code).
        """
        # 1) Attribute evaluations spent in the regime that is finishing.
        delta = max(0, self._counteval - self._bipop_regime_eval_anchor)
        if self._bipop_current_regime == "large":
            self._bipop_evals_large += delta
        else:
            self._bipop_evals_small += delta

        # 2) Pick next regime.  Ties → large (matches reference behaviour).
        if self._bipop_evals_small < self._bipop_evals_large:
            next_regime = "small"
        else:
            next_regime = "large"

        sigma_default = self._sigma0_default()

        if next_regime == "large":
            # IPOP-style geometric growth: λ_l = 2^k · λ_default
            self._bipop_large_count += 1
            new_lam = int(self._base_lam * (2**self._bipop_large_count))
            new_sigma = sigma_default
        else:
            # Small regime: small population with random small step size.
            # λ_s = floor(λ_default · (½ · λ_l / λ_default)^(U[0,1]²))
            # σ_s = σ_default · 10^(-2·U[0,1])
            cur_large_lam = self._base_lam * (2 ** max(self._bipop_large_count, 1))
            ratio = 0.5 * cur_large_lam / max(self._base_lam, 1)
            u_pop = float(np.random.rand())
            new_lam = int(np.floor(self._base_lam * (ratio ** (u_pop * u_pop))))
            new_lam = max(new_lam, self._base_lam)  # at least base
            u_sig = float(np.random.rand())
            new_sigma = sigma_default * (10.0 ** (-2.0 * u_sig))
            new_sigma = max(new_sigma, 1e-6)

        prev_lam = self._lam
        self._apply_restart(center, new_lam, new_sigma)
        self._bipop_current_regime = next_regime
        self._bipop_regime_eval_anchor = self._counteval  # reset to 0 below

        self.logger.info(
            "CMA-ES BIPOP restart #%d (%s): λ %d→%d  σ=%.4f  large_evals=%d small_evals=%d  large_count=%d  (%s)",
            self._restart_count,
            next_regime,
            prev_lam,
            new_lam,
            new_sigma,
            self._bipop_evals_large,
            self._bipop_evals_small,
            self._bipop_large_count,
            reason or "stagnation",
        )

    def _sigma0_default(self) -> float:
        """Initial step size as a fraction of the mean box half-range."""
        assert self._ranges is not None
        sigma = self._sigma0_frac * float(np.mean(self._ranges) / 2.0)
        return max(sigma, 1e-6)

    def _apply_restart(self, center: np.ndarray, new_lam: int, new_sigma: float) -> None:
        """Reset distribution state for *new_lam* and *new_sigma* at *center*.

        Common bookkeeping shared by both IPOP and BIPOP restart paths.
        Increments :attr:`_restart_count`, recomputes μ and the CMA-ES
        adaptation constants, resets paths/covariance/eigendecomposition,
        flushes stale generation tracking, and emits the first generation.
        """
        self._restart_count += 1

        new_lam = max(new_lam, 4)  # CMA-ES requires λ ≥ 4
        new_mu = max(new_lam // 2, 1)

        # Recombination weights (log-linear, positive)
        raw_w = np.log(new_mu + 0.5) - np.log(np.arange(1, new_mu + 1, dtype=float))
        new_w = raw_w / raw_w.sum()
        new_mu_eff = 1.0 / (new_w**2).sum()

        n = self.problem.dim

        # Recompute adaptation constants for new population size
        new_c_sigma = (new_mu_eff + 2.0) / (n + new_mu_eff + 5.0)
        new_d_sigma = 1.0 + 2.0 * max(0.0, np.sqrt((new_mu_eff - 1.0) / (n + 1.0)) - 1.0) + new_c_sigma
        new_c_c = (4.0 + new_mu_eff / n) / (n + 4.0 + 2.0 * new_mu_eff / n)
        new_c_1 = 2.0 / ((n + 1.3) ** 2 + new_mu_eff)
        new_c_mu = min(
            1.0 - new_c_1,
            2.0 * (new_mu_eff - 2.0 + 1.0 / new_mu_eff) / ((n + 2.0) ** 2 + new_mu_eff),
        )

        new_m = self.problem.project(center)

        self._lam = new_lam
        self._mu = new_mu
        self._w = new_w
        self._mu_eff = new_mu_eff
        self._c_sigma = new_c_sigma
        self._d_sigma = new_d_sigma
        self._c_c = new_c_c
        self._c_1 = new_c_1
        self._c_mu = new_c_mu

        self._m = new_m
        self._sigma = new_sigma
        self._C = np.eye(n)
        self._p_c = np.zeros(n)
        self._p_sigma = np.zeros(n)
        self._B = np.eye(n)
        self._D = np.ones(n)
        self._eigeneval = 0
        self._counteval = 0

        # Flush stale generation tracking and stale queued points — results for
        # pre-restart points are ignored anyway, and clearing frees queue
        # capacity for the (typically larger) new generation.
        self._pending.clear()
        self._gen_results.clear()
        self._gen_emitted.clear()
        self.clear_output()

        # Emit the first generation from the new distribution
        self._emit_generation()

    @property
    def restart_count(self) -> int:
        """Number of restarts triggered so far (IPOP or BIPOP)."""
        return self._restart_count

    @property
    def bipop_regime(self) -> str:
        """Current BIPOP regime (``"large"`` or ``"small"``).

        In IPOP mode this stays at ``"large"``.
        """
        return self._bipop_current_regime

    @property
    def bipop_evals_large(self) -> int:
        """Cumulative evaluations spent in the BIPOP large regime."""
        return self._bipop_evals_large

    @property
    def bipop_evals_small(self) -> int:
        """Cumulative evaluations spent in the BIPOP small regime."""
        return self._bipop_evals_small

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

        # IPOP/BIPOP restarts grow λ beyond the default queue capacity (20);
        # without this, put_nowait drops the overflow and the generation can
        # never collect enough results to trigger the next update (deadlock).
        self.ensure_output_capacity(self._lam)

        emitted = 0
        for i in range(self._lam):
            z = np.random.randn(n)
            # y = B D z  →  covariance = B D² Bᵀ = C
            y = self._B @ (self._D * z)
            x = self._m + self._sigma * y
            x = self.problem.project(x)

            who = f"CMAES:g{gen}:i{i}"
            # Put directly to bypass emit()'s ndarray-only check,
            # preserving the custom 'who' tag needed for generation tracking.
            try:
                self._output.put_nowait(Point(x, who))
            except Exception:
                # Track only points that will actually be evaluated.
                continue
            self._pending[who] = {"gen": gen, "i": i, "y": y}
            emitted += 1

        self._gen_emitted[gen] = emitted
        if emitted < self._lam:
            self.logger.warning(
                "CMA-ES generation %d: only %d/%d points emitted (output queue full)",
                gen,
                emitted,
                self._lam,
            )

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
            raw = np.log(actual_mu + 0.5) - np.log(np.arange(1, actual_mu + 1, dtype=float))
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
        p_c = (1.0 - self._c_c) * p_c + h_sigma * np.sqrt(self._c_c * (2.0 - self._c_c) * mu_eff) * y_w

        # --- Covariance matrix update ---
        # Rank-μ term
        Y = np.column_stack([d["y"] for d in selected])  # (n, actual_mu)
        rank_mu = (Y * w[:actual_mu]) @ Y.T  # weighted outer-product sum

        # Correction for h_sigma = 0
        delta_h = (1.0 - h_sigma) * self._c_c * (2.0 - self._c_c)

        C = (1.0 - self._c_1 - self._c_mu) * C + self._c_1 * (np.outer(p_c, p_c) + delta_h * C) + self._c_mu * rank_mu

        # --- Step-size update (cumulative path length control) ---
        self._sigma *= float(np.exp((self._c_sigma / self._d_sigma) * (norm_p_sigma / self._chi_n - 1.0)))

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
