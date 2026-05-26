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
NL-SHADE-RSP Heuristic
======================

NL-SHADE-RSP (Stanovov, Akhmedova & Semenkin, CEC 2020) — winner of the
CEC-2020 single-objective bound-constrained competition.  NL-SHADE-RSP is
a direct refinement of jSO (:class:`~panobbgo.heuristics.jso.JSO`) and
inherits its weighted ``current-to-pbest-w/1`` mutation, linear
``p_best`` schedule, asymmetric F-cap, success-history adaptation, and
linear population size reduction (LPSR).  Two changes lift it above jSO
on the later CEC test suites:

1. **Rank-based Selective Pressure (RSP)**.  jSO draws the differential
   donor ``r1`` *uniformly* from the live population.  NL-SHADE-RSP
   instead draws it with probability that decays exponentially with
   fitness rank — fitter individuals are picked more often::

       w_i = exp(−kp · i / N)          (i = 0 is the best, N = pool size)
       Pr(pick rank i) = w_i / Σ_j w_j

   With ``kp = 3`` (the paper's value) the best individual is ≈ ``e³``
   times more likely to be chosen than the worst, so the
   ``F · (x_r1 − x_r2)`` term acquires a mild "toward-the-good-basin"
   drift on top of jSO's pbest exploitation.  ``kp → 0`` recovers jSO's
   uniform selection exactly, so the pressure is a continuous dial the
   self-improvement loop can tune.  The ``r2`` donor stays uniform over
   the ``population ∪ archive`` union — the base-class archive stores
   bare position vectors without fitness, so it cannot be ranked without
   a larger refactor.  This is a faithful, well-scoped subset of the
   full NL-SHADE-RSP, which applies RSP to both donors.

2. **Adaptive archive size**.  jSO (and L-SHADE) cap the external
   archive of replaced parents at a *fixed* ``round(archive_factor ·
   NP_current)``.  NL-SHADE-RSP re-draws the archive cap once per
   generation uniformly from ``randint(0, A_max + 1)`` where
   ``A_max = round(archive_factor · NP_current)``.  Randomising how many
   replaced parents the archive holds varies the diversity of the
   ``r2`` donor pool generation-to-generation — sometimes the archive is
   empty (``r2`` comes only from the live population), sometimes it is
   full.  Pair this with the enlarged default ``archive_factor = 2.6``
   (the L-SHADE-RSP / NL-SHADE-RSP setting) so the archive has room to
   grow.

Asynchronous execution
----------------------

NL-SHADE-RSP inherits the entire async pipeline from
:class:`~panobbgo.heuristics.jso.JSO` /
:class:`~panobbgo.heuristics.lshade.LSHADE`: per-slot pending dict,
generation-by-count book-keeping, archive of replaced parents, frozen
anchor memory bin, and warm restart via :meth:`on_restart`.  The only
overridden methods are :meth:`_select_r1` (rank-based donor selection)
and :meth:`_archive_cap` (per-generation adaptive cap); the
trial-generation, memory-update, and LPSR paths are unchanged.

The adaptive archive cap is re-drawn at :meth:`on_start`,
:meth:`on_restart`, and at the end of every generation (after LPSR has
shrunk the population), so the cap always tracks the current population
size.  When the strategy budget is unknown the inherited jSO fall-backs
apply (constant ``p_best``, bypassed F-cap, no LPSR); the rank-based
selection and adaptive archive are unaffected by the budget.

References
----------

* V. Stanovov, S. Akhmedova & E. Semenkin (2021).  "NL-SHADE-RSP
  Algorithm with Adaptive Archive and Selective Pressure for CEC 2021
  Numerical Optimization."  *Proceedings of CEC 2021*.  (The algorithm
  won the CEC-2020 competition under the name NL-SHADE-RSP.)
* J. Brest, M. S. Maučec & B. Bošković (2017).  "Single Objective
  Real-Parameter Optimization: Algorithm jSO."  *Proceedings of CEC
  2017*.  The jSO foundation NL-SHADE-RSP refines.
* R. Tanabe & A. Fukunaga (2014).  "Improving the Search Performance of
  SHADE Using Linear Population Size Reduction."  *Proceedings of CEC
  2014*.  The L-SHADE foundation.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from panobbgo.heuristics.jso import (
    _DEFAULT_H,
    _DEFAULT_NP_INIT,
    _DEFAULT_NP_MIN,
    _DEFAULT_P_BEST_MAX,
    _DEFAULT_P_BEST_MIN,
    JSO,
)


# Default selective-pressure coefficient.  Stanovov et al. use ``kp = 3``;
# larger values concentrate the ``r1`` draw on the leading individuals,
# ``kp = 0`` recovers jSO's uniform selection.
_DEFAULT_KP: float = 3.0

# Enlarged external archive (L-SHADE-RSP / NL-SHADE-RSP), vs jSO's 1.0.
# Gives the per-generation adaptive cap room to vary.
_DEFAULT_ARCHIVE_FACTOR: float = 2.6


class NLSHADE_RSP(JSO):
    """NL-SHADE-RSP: rank-based selective pressure + adaptive archive on jSO.

    Args:
        strategy: The owning :class:`~panobbgo.core.StrategyBase`.
        NP_init: Initial population size.  Default ``30``.
        NP_min: Minimum population size after LPSR shrinking.  Default
            ``4``.
        H: History memory size.  Default ``5`` (inherited jSO value).
        p_best_max: Upper bound on the linear ``p_best`` schedule.
            Default ``0.25``.
        p_best_min: Lower bound on the linear ``p_best`` schedule.
            Default ``0.125``.
        archive_factor: Multiplier for the external archive cap.  Default
            ``2.6`` — the enlarged L-SHADE-RSP / NL-SHADE-RSP archive.
            ``A_max = round(archive_factor · NP_current)``; with
            ``adaptive_archive`` the per-generation cap is drawn from
            ``randint(0, A_max + 1)``.  ``0`` disables the archive.
        kp: Selective-pressure coefficient for the rank-based ``r1``
            draw.  Default ``3.0`` (Stanovov et al.).  Must be a
            non-negative finite float.  ``0`` recovers jSO's uniform
            donor selection.
        adaptive_archive: When ``True`` (default) the archive cap is
            re-drawn uniformly from ``randint(0, A_max + 1)`` once per
            generation.  ``False`` keeps jSO's fixed
            ``round(archive_factor · NP_current)`` cap.
        seed: Optional seed for the per-instance RNG.
        name: Override the heuristic's display name.

    Notes:
        - All numeric arguments are validated; bad values raise
          :class:`ValueError`.
        - State is per-instance, like every Panobbgo heuristic.
        - The rank-based ``r1`` selection biases the differential donor
          toward fitter individuals; the ``r2`` donor stays uniform over
          the ``population ∪ archive`` union (the base archive carries no
          fitness, so it cannot be ranked).
    """

    def __init__(
        self,
        strategy,
        NP_init: int = _DEFAULT_NP_INIT,
        NP_min: int = _DEFAULT_NP_MIN,
        H: int = _DEFAULT_H,
        p_best_max: float = _DEFAULT_P_BEST_MAX,
        p_best_min: float = _DEFAULT_P_BEST_MIN,
        archive_factor: float = _DEFAULT_ARCHIVE_FACTOR,
        kp: float = _DEFAULT_KP,
        adaptive_archive: bool = True,
        seed: Optional[int] = None,
        name: Optional[str] = None,
    ) -> None:
        if not np.isfinite(kp) or kp < 0.0:
            raise ValueError(f"NLSHADE_RSP: kp must be a non-negative finite float, got {kp}")
        if not isinstance(adaptive_archive, bool):
            raise ValueError(f"NLSHADE_RSP: adaptive_archive must be a bool, got {adaptive_archive!r}")

        super().__init__(
            strategy,
            NP_init=NP_init,
            NP_min=NP_min,
            H=H,
            p_best_max=p_best_max,
            p_best_min=p_best_min,
            archive_factor=archive_factor,
            seed=seed,
            name=name or "NLSHADE_RSP",
        )
        self.kp: float = float(kp)
        self.adaptive_archive: bool = adaptive_archive
        # Per-generation adaptive archive cap; refreshed at start / restart
        # and at every end-of-generation.  Initialised here so a stray
        # ``_trim_archive`` before the first refresh is well-defined.
        self._archive_size_target: int = self._max_archive()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _max_archive(self) -> int:
        """``A_max`` — the jSO fixed cap, used as the upper draw bound."""
        return max(int(round(self.archive_factor * self._NP_current)), 0)

    def _refresh_archive_target(self) -> None:
        """Re-draw the per-generation archive cap from ``randint(0, A_max + 1)``."""
        a_max = self._max_archive()
        self._archive_size_target = int(self._rng.integers(0, a_max + 1))

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def _archive_cap(self) -> int:
        """Adaptive per-generation cap (or the fixed jSO cap when disabled)."""
        if not self.adaptive_archive:
            return super()._archive_cap()
        return self._archive_size_target

    def _select_r1(self, live: List[int], target_idx: int, sorted_live: List[int]) -> Optional[int]:
        """Rank-based selective-pressure draw of the ``r1`` donor.

        Sorts the live population (excluding the target) best-first and
        draws an index with probability ``exp(−kp · rank / N)``.  Falls
        back to the inherited uniform draw when ``kp <= 0`` so the
        pressure dial reaches jSO's behaviour at one endpoint.
        """
        if self.kp <= 0.0:
            return super()._select_r1(live, target_idx, sorted_live)
        # ``sorted_live`` is already best-first; drop the target while
        # preserving the fitness ordering so ``rank == position``.
        pool = [i for i in sorted_live if i != target_idx]
        if not pool:
            return None
        n = len(pool)
        ranks = np.arange(n, dtype=float)
        weights = np.exp(-self.kp * ranks / n)
        probs = weights / weights.sum()
        return int(self._rng.choice(np.asarray(pool), p=probs))

    # ------------------------------------------------------------------
    # Heuristic interface
    # ------------------------------------------------------------------

    def on_start(self) -> None:
        """Allocate state and seed the first adaptive archive cap."""
        super().on_start()
        self._refresh_archive_target()

    def on_restart(self, center, reason: str = "") -> None:
        """Reset state around ``center`` and re-seed the adaptive archive cap."""
        super().on_restart(center, reason)
        self._refresh_archive_target()

    def _end_of_generation(self) -> None:
        """Run jSO's end-of-generation updates, then re-draw the archive cap."""
        super()._end_of_generation()
        if self.adaptive_archive:
            self._refresh_archive_target()
