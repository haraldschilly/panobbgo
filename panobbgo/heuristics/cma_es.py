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

**Self-restart** (the default, ``self_restart=True``): the heuristic also
watches its *own* termination criteria and restarts itself when one fires,
without needing the ``Restart`` analyzer.  Without this a solo CMA-ES has no
termination criterion at all — it runs a single CMA-ES for the whole budget
and, once σ has collapsed, keeps resampling the same point.  Measured on five
MA-BBOB instances at d=5 with a 2500-evaluation budget (2026-09-10): on
average **52 % of the budget was spent after the last improvement of any
size**, 66 % after the last ≥1e-3 relative improvement, and 91 % after 99 %
of the total improvement had been reached; on two of the five instances σ
*diverged* to its clamp and 92 % of the budget bought nothing.

The criteria are the standard ones (Hansen 2016, §4 "Discussion", and the
``cma.CMAOptions`` defaults of pycma); each is a constructor kwarg:

* ``tolx`` (1e-11, σ0-relative): all components of ``σ·sqrt(diag C)`` and
  ``σ·p_c`` below ``tolx · σ0``.
* ``tolfun`` (1e-11): the range of the best objective values of the last
  ``10 + ⌈30n/λ⌉`` generations *and* all values of the current generation is
  below ``tolfun``.
* ``tolfunhist`` (1e-12): the range of the best objective values over that
  same history alone is below ``tolfunhist``.
* ``stagnation`` (window ``20 + ⌈120n/λ⌉`` generations): in *both* the
  best-value and the median-value history, the median of the last 30 % of the
  window is not better than the median of the first 30 %.
* ``conditioncov`` (1e14): the condition number of C exceeds the bound.
* ``noeffectaxis``: adding ``0.1·σ·d_i·b_i`` (one principal axis, cycled per
  generation) does not change the mean m.
* ``noeffectcoord``: adding ``0.2·σ·sqrt(C_ii)`` in any single coordinate does
  not change m.

Two further criteria are *not* in the reference.  They exist because the
reference tolerances are written for runs of many thousands of generations and
almost never fire inside a 500·dim budget (measured 2026-09-10: 5 firings in 10
battery runs, all of them after the search had already reached 1e-10 precision,
which is below the AOCC log floor — hence no measurable gain):

* ``stagnation_frac`` (off by default): restart when the best value seen has not
  improved by more than ``stagnation_rel_tol`` (1e-8, relative) over the last
  ``max(10·λ, stagnation_frac · max_eval)`` evaluations.  This is the only
  criterion that scales with the *budget* rather than with n and λ.  It is off
  because it measured *harmful* at the fractions where it fires often: −0.021
  AOCC at 0.05 and −0.001 at 0.15 on the standard battery, since CMA-ES
  routinely spends 15–30 generations adapting C without improving its best
  value, and interrupting that costs more than the wasted tail is worth.
  Only 0.25 was neutral (+0.002), by which point it has converged onto the
  reference criteria anyway.
* ``sigma_divergence`` (**on** by default): restart when the sampling spread
  ``σ·sqrt(diag C)`` has sat at or above ``sigma_max_frac · range`` in every
  coordinate for ``sigma_divergence_gens`` consecutive generations — the
  panobbgo analogue of pycma's ``tolupsigma``.
  :meth:`CMAES._update` clamps σ at the mean box range, so a diverging run does
  not blow up: it parks at the clamp and samples the whole box (in practice the
  box *boundary*, since everything outside is projected back onto it) for the
  rest of the budget.  Two of the five diagnosed d=5 instances did exactly
  that.  The clamp keeps the run numerically alive at the price of keeping it
  useless; detecting the state and restarting is the reference cure
  (Hansen 2016 stops such a run outright), which is why the clamp itself is
  left alone.  This is the one addition that paid: **+0.0330 AOCC on the
  standard battery, 6 of 6 seeds positive** (+0.024 at d=2, +0.042 at d=5,
  2026-09-10), concentrated entirely in the 11 of 60 cells where it fires.

When a criterion fires, the heuristic restarts through the very same code
path as the ``restart`` event, with a start point chosen by ``restart_from``
(``"random"`` — a uniform draw in the box, the reference IPOP behaviour and
the default; ``"best"`` — the best point this heuristic has evaluated;
``"center"`` — the box centre, i.e. what :meth:`on_start` uses).  Because the
restart path is the IPOP/BIPOP one, ``ipop_factor`` finally has an effect in
solo runs, where it was previously dead code.

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
   one CMA-ES update step, check the termination criteria, and either restart
   (if one fired and ``self_restart`` is on) or emit the next generation.
3. ``on_restart(center, reason)``  — reset distribution to *center* with the
   restart regime selected by ``restart_mode`` (``"ipop"`` or ``"bipop"``).

.. codeauthor:: Harald Schilly
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

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
        self_restart (bool): Watch the internal termination criteria and
            restart without waiting for a ``restart`` event.  Default ``True``.
            ``False`` restores the pre-2026-09 behaviour: one CMA-ES run for
            the whole budget unless the ``Restart`` analyzer intervenes.
        restart_from (str): Where a *self*-restart re-centres the search —
            ``"random"`` (default; uniform draw in the box, the reference IPOP
            behaviour), ``"best"`` (the best point this heuristic evaluated) or
            ``"center"`` (the box centre, as :meth:`on_start` uses).  Restarts
            coming from the ``restart`` event always use the analyzer's centre.
        tolx (float): Termination when all components of ``σ·sqrt(diag C)``
            and ``σ·p_c`` fall below ``tolx · σ0``.  Default 1e-11
            (pycma ``cma.CMAOptions['tolx']``).
        tolfun (float): Termination when the range of the best objective
            values over the last ``10 + ⌈30n/λ⌉`` generations together with all
            values of the current generation is below this.  Default 1e-11.
        tolfunhist (float): Same window, best values only.  Default 1e-12
            (pycma ``tolfunhist``).
        stagnation (int, optional): Window length in generations for the
            median-based stagnation test of Hansen (2016).  ``None`` (default)
            resolves to ``20 + ⌈120n/λ⌉``; ``0`` disables the test.
        conditioncov (float): Termination when ``cond(C)`` exceeds this.
            Default 1e14 (pycma ``tolconditioncov``).
        noeffectaxis (bool): Enable the ``NoEffectAxis`` criterion (default
            ``True``).
        noeffectcoord (bool): Enable the ``NoEffectCoord`` criterion (default
            ``True``).
        stagnation_frac (float, optional): Budget-relative stagnation window as
            a fraction of ``config.max_eval``.  ``None`` (default) disables it
            and leaves only the reference criteria.  The window is
            ``max(10·λ, stagnation_frac · max_eval)`` evaluations; the criterion
            fires when the best value seen has not improved by more than
            ``stagnation_rel_tol`` (relative) over that window.  ``max_eval`` is
            read lazily on each check, never in ``__init__`` — the budget is not
            always known when the heuristic is constructed.
        stagnation_rel_tol (float): Relative improvement that counts as
            progress for ``stagnation_frac``.  Default 1e-8.
        sigma_divergence (bool): Enable the σ-divergence criterion (default
            ``True``): fire when the sampling spread ``σ·sqrt(diag C)`` has
            been at or above ``sigma_max_frac · range`` in *every* coordinate
            for ``sigma_divergence_gens`` consecutive generations.  This is the
            panobbgo analogue of pycma's ``tolupsigma`` "creeping behaviour"
            stop.  A diverged run has no basin worth keeping, so its restart
            always re-centres on the best point seen, whatever ``restart_from``
            says.
        sigma_max_frac (float): Divergence threshold as a fraction of each
            coordinate's box range.  Default 0.3 — a cloud whose per-coordinate
            standard deviation is a third of the box is sampling the box, not a
            neighbourhood.  Measured over 60 standard-battery cells (6 seeds,
            2026-09-10) the criterion fired on 11 of them, every firing gained
            AOCC (mean +0.180, max +0.437), and those cells averaged 0.370
            AOCC without it against 0.625 for the battery as a whole.
        sigma_divergence_gens (int): Consecutive generations above the
            threshold before the criterion fires.  Default 5.
        warm_start (str, optional): Fit the *initial* search distribution to
            the shared archive instead of starting from the box centre
            (``planning/DESIGN_warm_start_2026-09-10.md`` §2).  One of the
            three shared selectors — ``"archive"``, ``"archive_diverse"``,
            ``"archive_leaf"``, see
            :meth:`~panobbgo.core.Heuristic.archive_seed` — or the CMA-ES-only
            ``"archive_cov"``, which additionally seeds ``C`` with the
            covariance of the seed cloud.  ``None`` (default) is the cold
            start, statement for statement the behaviour shipped before.
            Unlike the DE family and PSO, a warm start here saves **no**
            evaluations: CMA-ES has no population to fill, so the whole gain
            is starting in the right basin with the right scale.  At
            ``t = 0`` the archive is empty and the heuristic cold-starts.
    """

    #: Accepted values for the ``restart_from`` constructor argument.
    SUPPORTED_RESTART_FROM = ("random", "best", "center")

    #: Accepted values for ``warm_start``: the shared selectors of
    #: :attr:`~panobbgo.core.Heuristic.WARM_START_MODES` plus the CMA-ES-only
    #: covariance seed.  ``"archive_cov"`` is kept separate from ``"archive"``
    #: on purpose, so the covariance seed can be measured apart from the
    #: mean/σ seed.
    SUPPORTED_WARM_START = Heuristic.WARM_START_MODES + ("archive_cov",)

    def __init__(
        self,
        strategy,
        sigma0: float = 0.3,
        popsize: Optional[int] = None,
        min_results_fraction: float = 0.5,
        ipop_factor: float = 2.0,
        restart_mode: str = "ipop",
        self_restart: bool = True,
        restart_from: str = "random",
        tolx: float = 1e-11,
        tolfun: float = 1e-11,
        tolfunhist: float = 1e-12,
        stagnation: Optional[int] = None,
        conditioncov: float = 1e14,
        noeffectaxis: bool = True,
        noeffectcoord: bool = True,
        stagnation_frac: Optional[float] = None,
        stagnation_rel_tol: float = 1e-8,
        sigma_divergence: bool = True,
        sigma_max_frac: float = 0.3,
        sigma_divergence_gens: int = 5,
        warm_start: Optional[str] = None,
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

        # --- Self-restart: internal termination criteria (Hansen 2016 / pycma) ---
        if restart_from not in self.SUPPORTED_RESTART_FROM:
            raise ValueError(f"restart_from must be one of {self.SUPPORTED_RESTART_FROM!r}, got {restart_from!r}")
        for _n, _v in (("tolx", tolx), ("tolfun", tolfun), ("tolfunhist", tolfunhist)):
            if _v < 0.0:
                raise ValueError(f"{_n} must be >= 0, got {_v!r}")
        if conditioncov <= 1.0:
            raise ValueError(f"conditioncov must be > 1, got {conditioncov!r}")
        if stagnation is not None and int(stagnation) < 0:
            raise ValueError(f"stagnation must be >= 0 or None, got {stagnation!r}")
        self._self_restart = bool(self_restart)
        self._restart_from = restart_from
        self._tolx = float(tolx)
        self._tolfun = float(tolfun)
        self._tolfunhist = float(tolfunhist)
        self._stagnation_cfg = None if stagnation is None else int(stagnation)
        self._conditioncov = float(conditioncov)
        self._noeffectaxis = bool(noeffectaxis)
        self._noeffectcoord = bool(noeffectcoord)

        # --- Budget-relative criteria (panobbgo additions, not in the reference) ---
        if stagnation_frac is not None and not 0.0 < float(stagnation_frac) <= 1.0:
            raise ValueError(f"stagnation_frac must be in (0, 1] or None, got {stagnation_frac!r}")
        if stagnation_rel_tol < 0.0:
            raise ValueError(f"stagnation_rel_tol must be >= 0, got {stagnation_rel_tol!r}")
        if not 0.0 < float(sigma_max_frac) <= 1.0:
            raise ValueError(f"sigma_max_frac must be in (0, 1], got {sigma_max_frac!r}")
        if int(sigma_divergence_gens) < 1:
            raise ValueError(f"sigma_divergence_gens must be >= 1, got {sigma_divergence_gens!r}")
        if warm_start is not None and warm_start not in self.SUPPORTED_WARM_START:
            raise ValueError(
                f"CMAES: warm_start must be None or one of {self.SUPPORTED_WARM_START}, got {warm_start!r}"
            )
        #: Archive selector for :meth:`_warm_start_distribution`, or ``None``
        #: for the cold start.  A *string*, deliberately not a callable: the
        #: trigger is :meth:`warm_start_now`.
        self.warm_start: Optional[str] = warm_start
        #: Region the next warm start is restricted to, or ``None`` for the
        #: whole archive.  Written by
        #: :meth:`panobbgo.strategies.blocks.StrategyBlockBandit._apply_pending_region`
        #: on the main thread just before a block opens, and cleared again
        #: right after — a *one-shot* hand-off from
        #: :class:`~panobbgo.heuristics.meta.MetaAnalyst`
        #: (``planning/DESIGN_meta_level_2026-09-10.md`` §2).  Anything
        #: :meth:`~panobbgo.core.Heuristic.archive_seed` accepts as ``box``:
        #: a :class:`~panobbgo.analyzers.splitter.Splitter.Box` or a
        #: ``(dim, 2)`` bounds array.
        self.warm_start_box: Any = None
        self._stagnation_frac = None if stagnation_frac is None else float(stagnation_frac)
        self._stagnation_rel_tol = float(stagnation_rel_tol)
        self._sigma_divergence = bool(sigma_divergence)
        self._sigma_max_frac = float(sigma_max_frac)
        self._sigma_divergence_gens = int(sigma_divergence_gens)

        # Budget-relative stagnation bookkeeping.  ``_total_evals`` counts the
        # results this heuristic has consumed over the *whole* run (it survives
        # restarts, unlike ``_counteval``); ``_stag_ref_fx`` is the best value
        # that still counts as "the last real improvement".
        self._total_evals: int = 0
        self._stag_ref_fx: float = float("inf")
        self._last_improve_evals: int = 0
        self._sigma_high_gens: int = 0

        # Per-run termination bookkeeping (reset by :meth:`_reset_run_state`)
        self._sigma0: float = 1.0  # σ at the start of the *current* CMA-ES run
        self._fx_hist: List[float] = []  # best penalty per generation
        self._med_hist: List[float] = []  # median penalty per generation
        self._gen_fx: List[float] = []  # all finite penalties of the last generation
        self._hist_len: int = 0  # 10 + ⌈30n/λ⌉ — tolfun/tolfunhist window
        self._stagnation_gens: int = 0  # 20 + ⌈120n/λ⌉ — stagnation window
        self._hist_max: int = 0  # how much history to keep around
        self._cond: float = 1.0  # cond(C) at the last eigendecomposition
        self._self_restart_count: int = 0
        self._last_stop_reason: str = ""

        # Best point this heuristic has evaluated (for ``restart_from="best"``)
        self._best_fx: float = float("inf")
        self._best_x: Optional[np.ndarray] = None

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

    @staticmethod
    def _recombination_weights(mu: int) -> tuple[np.ndarray, float]:
        """The log-linear positive weights for ``mu`` parents, and their ``mu_eff``.

        ``w_i ∝ ln(μ + ½) − ln i``, normalised to sum 1 (Hansen 2016, eq. 49),
        with the effective number of parents ``μ_eff = 1 / Σ w_i²``.  Every
        place that needs weights derives them here — the initial set-up, a
        restart's new λ, a generation that came back short, and the warm
        start — so "fewer parents than μ" is re-normalised the same way
        everywhere.
        """
        raw = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1, dtype=float))
        w = raw / raw.sum()
        return w, float(1.0 / (w**2).sum())

    def on_start(self) -> None:
        """Initialise CMA-ES state and emit the first generation."""
        n = self.problem.dim

        # Population sizes
        lam = self._popsize_override or (4 + int(3 * np.log(max(n, 2))))
        mu = lam // 2

        # Recombination weights (log-linear, positive) and the effective
        # number of parents they imply
        w, mu_eff = self._recombination_weights(mu)

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

        # A warm start replaces m / σ / C — and nothing else.  λ, μ, the
        # adaptation constants and the counters are the cold set-up above, and
        # the generation at the end of this method is emitted exactly as in a
        # cold run.  It runs before ``_reset_run_state`` so that ``_sigma0``
        # (which the ``tolx`` criterion is relative to) is the σ this run
        # actually starts from.
        if self.warm_start:
            self._warm_start_distribution()

        self._reset_run_state()

        self.logger.info(
            "CMA-ES started: n=%d λ=%d μ=%d σ0=%.4f mode=%s self_restart=%s from=%s",
            n,
            lam,
            mu,
            sigma,
            self._restart_mode,
            self._self_restart,
            self._restart_from,
        )
        self._emit_generation()

    # ------------------------------------------------------------------
    # Warm start from the shared archive
    # ------------------------------------------------------------------

    def _warm_start_seeds(self) -> List[np.ndarray]:
        """Positions of the archive points this warm start is fitted to.

        ``k = max(λ, 4 + ⌊3 ln n⌋)`` — a full generation's worth, never fewer
        than the default λ.  ``"archive_cov"`` asks for ``2n`` as well: a
        sample covariance of ``k ≤ n`` points is rank-deficient by
        construction, and would always fall back to the identity.
        """
        n = self.problem.dim
        k = max(self._lam, 4 + int(3 * np.log(max(n, 2))))
        if self.warm_start == "archive_cov":
            k = max(k, 2 * n)
        seeds = self.archive_seed(k, mode=self.warm_start, box=self.warm_start_box)
        return [np.asarray(r.x, dtype=float) for r in seeds]

    def _warm_start_distribution(self) -> bool:
        """Fit ``m`` / ``σ`` / ``C`` to the shared archive.  ``True`` iff seeded.

        The recipe of the design's §2 "Per-arm recipes":

        * **m** — the μ-weighted recombination of the best μ seeds, using the
          very weights :meth:`_update` recombines with (re-normalised when
          fewer than μ seeds exist, exactly as :meth:`_update` does).
        * **σ** — the mean per-coordinate standard deviation of the seed
          cloud, clipped into ``[1e-6·range, σ0_cold]``.  Never *wider* than a
          cold start: a warm start may only narrow the search.
        * **C** — the identity, or (``"archive_cov"``) the seed covariance
          normalised to unit determinant, so it carries the *shape* of the
          cloud while σ alone carries its scale.
        * evolution paths zeroed, ``_counteval`` untouched.

        Returns ``False`` — **without touching any state** — when the archive
        has nothing to give, so a caller can fall back to the cold path (or,
        for :meth:`warm_start_now`, leave a running search alone).
        """
        if not self.warm_start:
            return False
        if self._ranges is None or self._w is None:
            return False  # on_start has not run yet
        seeds = self._warm_start_seeds()
        if not seeds:
            return False

        n = self.problem.dim
        X = np.vstack(seeds)

        # --- mean: μ-weighted recombination of the best seeds ---
        mu = max(1, min(self._mu, len(X)))
        w = self._w if mu == len(self._w) else self._recombination_weights(mu)[0]
        self._m = self.problem.project(w[:mu] @ X[:mu])

        # --- σ: the spread of the seed cloud, never wider than cold ---
        spread = float(np.mean(np.std(X, axis=0))) if len(X) > 1 else 0.0
        floor = 1e-6 * float(np.mean(self._ranges))
        self._sigma = float(np.clip(spread, floor, self._sigma0_default()))

        # --- C, B, D and the evolution paths ---
        self._p_c = np.zeros(n)
        self._p_sigma = np.zeros(n)
        if self.warm_start == "archive_cov" and len(X) >= n + 2:
            self._seed_covariance(X)
        else:
            self._reset_covariance(n)
            self._cond = 1.0
        # B and D were just made consistent with C, so the lazy
        # eigendecomposition schedule restarts from here.  ``_counteval``
        # itself is deliberately left alone.
        self._eigeneval = self._counteval

        self.logger.info(
            "CMA-ES warm start (%s): %d seeds, σ=%.4g, cond(C)=%.3g",
            self.warm_start,
            len(X),
            self._sigma,
            self._cond,
        )
        return True

    def _seed_covariance(self, X: np.ndarray) -> None:
        """Set ``C`` to the seed covariance, normalised to unit determinant.

        Eigendecomposed exactly the way :meth:`_update` does — symmetrise,
        ``eigh``, the same ``1e-20`` eigenvalue floor, the same ``1e7``
        condition guard that falls back to the identity.  The unit-determinant
        normalisation is what keeps the *scale* of the search in σ alone,
        where CMA-ES's step-size control can adapt it; without it the seed
        cloud's scale would be counted twice.
        """
        n = self.problem.dim
        cov = np.atleast_2d(np.asarray(np.cov(X, rowvar=False), dtype=float))
        if cov.shape != (n, n) or not np.all(np.isfinite(cov)):
            self._reset_covariance(n)
            self._cond = 1.0
            return

        C_sym = (cov + cov.T) / 2.0
        try:
            eigvals, B_new = np.linalg.eigh(C_sym)
        except np.linalg.LinAlgError:
            self.logger.warning("CMA-ES warm start: eigendecomposition of the seed covariance failed")
            self._reset_covariance(n)
            self._cond = 1.0
            return

        eigvals = np.maximum(eigvals, 1e-20)
        # det(C) = Π eigvals = 1  ⇔  the eigenvalues have geometric mean 1.
        eigvals = eigvals / float(np.exp(np.mean(np.log(eigvals))))
        D_new = np.sqrt(eigvals)
        self._B = B_new
        self._D = D_new
        self._C = (B_new * eigvals) @ B_new.T
        self._cond = float(D_new.max() / D_new.min()) ** 2

        if D_new.max() / D_new.min() > 1e7:
            # A degenerate seed cloud (duplicates, or fewer independent
            # directions than dimensions) — the identity is the honest prior.
            self.logger.info("CMA-ES warm start: seed covariance too ill-conditioned — using I")
            self._reset_covariance(n)
            self._cond = 1.0

    def warm_start_now(self) -> bool:
        """Re-fit the search distribution to the shared archive, right now.

        The direct-call hook of
        :class:`~panobbgo.strategies.blocks.StrategyBlockBandit`; see
        :meth:`panobbgo.core.Heuristic.warm_start_now`.  Unlike a restart this
        keeps λ, the regime bookkeeping and ``_counteval`` — only the
        distribution moves — but it does drop the stale generation in flight,
        exactly as :meth:`_apply_restart` does, and starts a fresh
        termination-criteria window.

        Nothing is dropped when the archive is empty: the method returns
        ``False`` and the running search is left untouched.
        """
        if self._stopped or not self.warm_start:
            return False
        if self._m is None or self._w is None or self._ranges is None:
            return False  # not started yet — on_start will do the seeding
        if not self._warm_start_distribution():
            return False

        self._pending.clear()
        self._gen_results.clear()
        self._gen_emitted.clear()
        self.clear_output()
        self._reset_run_state()
        self._emit_generation()
        return True

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def on_new_results(self, results) -> None:
        """Collect results and trigger a CMA-ES update when enough have arrived."""
        for r in results:
            if not r.who.startswith("CMAES:"):
                continue

            info = self._pending.pop(r.who, None)
            if info is None:
                continue

            gen = info["gen"]
            gen_bucket = self._gen_results.get(gen)
            if gen_bucket is None:
                continue

            # A non-finite objective is information, not noise: the point is
            # worse than any finite one, so rank it last rather than drop it.
            # Dropping was also unsafe — the point still counts as *emitted*,
            # so enough non-finite results in one generation would leave it
            # permanently short of its quorum and the search would stop
            # emitting for the rest of the run.  No measured case of that is
            # on record; this is a guard, not a fix for an observed failure.
            if r.fx is None or not np.isfinite(r.fx):
                penalty = float("inf")
            else:
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
                # A fired termination criterion replaces the next generation
                # with a restart — ``_apply_restart`` emits from the fresh
                # distribution, so exactly one generation goes out either way.
                reason = self._check_termination() if self._self_restart else None
                if reason is not None:
                    self._self_restart_now(reason)
                else:
                    self._emit_generation()
                break

    def on_restart(self, center: np.ndarray, reason: str = "") -> None:
        """Reset CMA-ES to *center* using the configured restart scheme.

        Called by the :class:`~panobbgo.analyzers.restart.Restart` analyzer
        when stagnation is detected, and by :meth:`_self_restart_now` when one
        of the heuristic's own termination criteria fires (then *center* comes
        from ``restart_from`` instead of the analyzer).  Dispatches to the IPOP
        or BIPOP sub-routine according to ``restart_mode``.  Both schemes:

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
            u_pop = float(self.rng.random())
            new_lam = int(np.floor(self._base_lam * (ratio ** (u_pop * u_pop))))
            new_lam = max(new_lam, self._base_lam)  # at least base
            u_sig = float(self.rng.random())
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
        new_w, new_mu_eff = self._recombination_weights(new_mu)

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

        self._reset_run_state()

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
    # Termination criteria and self-restart
    # ------------------------------------------------------------------

    def _reset_run_state(self) -> None:
        """Reset the per-run bookkeeping the termination criteria read.

        Called once per CMA-ES *run*, i.e. from :meth:`on_start` and from
        :meth:`_apply_restart` after λ and σ of the new run are in place —
        the two window lengths both depend on λ, so they must be resolved
        here rather than in the constructor.
        """
        n = self.problem.dim
        lam = max(self._lam, 1)
        self._sigma0 = self._sigma
        self._fx_hist = []
        self._med_hist = []
        self._gen_fx = []
        self._cond = 1.0
        # Hansen (2016): TolFun looks at the last 10 + ⌈30n/λ⌉ generations,
        # Stagnation at the last 20 + ⌈120n/λ⌉.
        self._hist_len = int(10 + np.ceil(30.0 * n / lam))
        if self._stagnation_cfg is None:
            self._stagnation_gens = int(20 + np.ceil(120.0 * n / lam))
        else:
            self._stagnation_gens = self._stagnation_cfg
        self._hist_max = max(self._hist_len, self._stagnation_gens) + 1
        self._sigma_high_gens = 0
        # A fresh run gets a fresh budget-relative window to prove itself; the
        # *value* it has to beat (``_stag_ref_fx``) stays global, so a restart
        # that only rediscovers the old optimum still counts as stagnation.
        self._last_improve_evals = self._total_evals

    def _record_generation(self, collected: List[dict]) -> None:
        """Fold one finished generation into the termination histories.

        ``collected`` is already sorted ascending by penalty, so its first
        element is the generation's best.  Non-finite penalties (the rank-last
        marker for a failed evaluation) are excluded from the ranges the
        criteria test, but still counted in the median.
        """
        penalties = [float(d["penalty"]) for d in collected]
        best = penalties[0]
        self._fx_hist.append(best)
        self._med_hist.append(float(np.median(penalties)))
        self._gen_fx = [p for p in penalties if np.isfinite(p)]
        if len(self._fx_hist) > self._hist_max:
            del self._fx_hist[: -self._hist_max]
            del self._med_hist[: -self._hist_max]

        if np.isfinite(best) and best < self._best_fx:
            self._best_fx = best
            self._best_x = np.asarray(collected[0]["x"], dtype=float).copy()

        # Budget-relative stagnation clock.  ``_total_evals`` is the run-long
        # evaluation count (``_counteval`` restarts at 0 on every restart).
        self._total_evals += len(penalties)
        if np.isfinite(best) and best < self._stag_ref_fx - self._stagnation_rel_tol * abs(self._stag_ref_fx):
            self._stag_ref_fx = best
            self._last_improve_evals = self._total_evals
        elif not np.isfinite(self._stag_ref_fx) and np.isfinite(best):
            # First finite value: start the clock rather than count it as stagnation.
            self._stag_ref_fx = best
            self._last_improve_evals = self._total_evals

    def _check_termination(self) -> Optional[str]:
        """Return the name of the first termination criterion that fires, else ``None``.

        The criteria and their reference defaults follow Hansen (2016),
        "The CMA Evolution Strategy: A Tutorial", §"Discussion" (termination
        criteria), and the ``cma.CMAOptions`` defaults of pycma.
        """
        m, C, p_c, B, D = self._m, self._C, self._p_c, self._B, self._D
        if m is None or C is None or p_c is None or B is None or D is None:
            return None
        if not self._fx_hist:
            return None

        n = self.problem.dim
        sigma = self._sigma
        sqrt_diag_C = np.sqrt(np.maximum(np.diag(C), 0.0))

        # --- σ divergence: the sampling cloud has grown to box scale ---
        # Counted first so the streak is maintained no matter which criterion
        # ends up firing.  ``_update`` clamps σ at ``mean(ranges)``, so a
        # diverging run parks *at* the clamp instead of terminating — this is
        # the panobbgo stand-in for pycma's ``tolupsigma``.
        #
        # The test is on the *sampling spread* σ·sqrt(diag C), not on σ alone:
        # σ and C share one scale degree of freedom, so a perfectly healthy,
        # converging run can carry a large σ against a small C.  Testing σ on
        # its own produced exactly that false positive twice in 30 battery runs
        # (2026-09-10) and cost 0.04 and 0.20 AOCC on those two cells.  All
        # coordinates must be box-scale — one long axis is a ridge search, not
        # a divergence.
        if self._ranges is not None:
            if np.all(sigma * sqrt_diag_C >= self._sigma_max_frac * self._ranges):
                self._sigma_high_gens += 1
            else:
                self._sigma_high_gens = 0
        if self._sigma_divergence and self._sigma_high_gens >= self._sigma_divergence_gens:
            return "sigma_divergence"

        # --- TolX: the distribution has shrunk below a σ0-relative threshold ---
        tolx = self._tolx * self._sigma0
        if np.all(sigma * sqrt_diag_C < tolx) and np.all(sigma * np.abs(p_c) < tolx):
            return "tolx"

        # --- TolFun / TolFunHist: the objective range has flattened out ---
        if len(self._fx_hist) >= self._hist_len:
            window = [f for f in self._fx_hist[-self._hist_len :] if np.isfinite(f)]
            if window:
                spread = max(window) - min(window)
                combined = window + self._gen_fx
                if max(combined) - min(combined) < self._tolfun:
                    return "tolfun"
                if spread < self._tolfunhist:
                    return "tolfunhist"

        # --- Stagnation: neither the best nor the median history improves ---
        w = self._stagnation_gens
        if w > 0 and len(self._fx_hist) >= w:
            k = max(1, int(0.3 * w))
            improved = False
            for hist in (self._fx_hist[-w:], self._med_hist[-w:]):
                if float(np.median(hist[-k:])) < float(np.median(hist[:k])):
                    improved = True
                    break
            if not improved:
                return "stagnation"

        # --- Budget-relative stagnation: no real progress for a slice of the budget ---
        window = self._stagnation_eval_window()
        if window > 0 and self._total_evals - self._last_improve_evals >= window:
            return "stagnation_evals"

        # --- Condition number of C ---
        if self._cond > self._conditioncov:
            return "conditioncov"

        # --- NoEffectAxis: one principal axis per generation, cycled ---
        if self._noeffectaxis:
            i = (self._gen - 1) % n
            if np.all(m == m + 0.1 * sigma * D[i] * B[:, i]):
                return "noeffectaxis"

        # --- NoEffectCoord: any single coordinate ---
        if self._noeffectcoord and np.any(m == m + 0.2 * sigma * sqrt_diag_C):
            return "noeffectcoord"

        return None

    def _stagnation_eval_window(self) -> int:
        """Budget-relative stagnation window in evaluations (0 = disabled).

        ``config.max_eval`` is read here rather than in ``__init__`` because
        the budget is only guaranteed to be set on the config *after* the
        heuristic is constructed.  The ``10·λ`` floor also makes the criterion
        self-limiting across IPOP restarts: λ doubles each time, so the window
        grows past the remaining budget after a few restarts instead of
        thrashing.
        """
        if self._stagnation_frac is None:
            return 0
        max_eval = getattr(self.config, "max_eval", 0) or 0
        return int(max(10 * max(self._lam, 1), self._stagnation_frac * float(max_eval)))

    def _restart_center(self, mode: Optional[str] = None) -> np.ndarray:
        """Start point for a *self*-restart, per ``restart_from`` or *mode*.

        ``"random"`` draws from :attr:`rng` — the heuristic's own generator —
        so a seeded run stays reproducible.
        """
        mode = mode or self._restart_from
        if mode == "random":
            return self.problem.random_point(rng=self.rng)
        if mode == "best" and self._best_x is not None:
            return self._best_x.copy()
        # ``"center"``, and ``"best"`` before anything has been evaluated
        assert self._lo is not None and self._hi is not None
        return 0.5 * (self._lo + self._hi)

    def _self_restart_now(self, reason: str) -> None:
        """Restart because *our own* criterion ``reason`` fired.

        Goes through :meth:`on_restart`, so the IPOP/BIPOP logic, the
        population growth and the bookkeeping are shared with the
        analyzer-driven path — only the start point differs.  A σ-divergence
        restart overrides ``restart_from``: the distribution has spread over
        the whole box, so there is no basin left to keep and no reason to
        throw the best point seen away.
        """
        self._self_restart_count += 1
        self._last_stop_reason = reason
        mode = "best" if reason == "sigma_divergence" else None
        self.on_restart(self._restart_center(mode), f"self-restart: {reason}")

    @property
    def n_restarts(self) -> int:
        """Total restarts in this run — event-driven plus self-triggered."""
        return self._restart_count

    @property
    def n_self_restarts(self) -> int:
        """Restarts triggered by the internal termination criteria."""
        return self._self_restart_count

    @property
    def last_stop_reason(self) -> str:
        """Name of the termination criterion that fired last (``""`` if none)."""
        return self._last_stop_reason

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
            z = self.rng.standard_normal(n)
            # y = B D z  →  covariance = B D² Bᵀ = C
            y = self._B @ (self._D * z)
            x = self._m + self._sigma * y
            x = self.problem.project(x)

            who = f"CMAES:g{gen}:i{i}"
            # Put directly to bypass emit()'s ndarray-only check,
            # preserving the custom 'who' tag needed for generation tracking.
            self._put(Point(x, who))
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
        self._record_generation(collected)
        selected = collected[: self._mu]

        # Recombination weights (may use fewer than μ if fewer arrived)
        actual_mu = len(selected)
        if actual_mu < len(w_full):
            # Re-normalise weights for the actual number of survivors
            w, mu_eff = self._recombination_weights(actual_mu)
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

            # cond(C) = (max D / min D)², recorded *before* the guard below
            # resets C, so the ``conditioncov`` criterion still sees it.
            self._cond = float(D_new.max() / D_new.min()) ** 2

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
