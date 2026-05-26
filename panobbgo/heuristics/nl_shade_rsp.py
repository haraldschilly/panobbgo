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

NL-SHADE-RSP (Stanovov, Akhmedova & Semenkin, CEC 2021) — winner of the
CEC-2021 single-objective bound-constrained competition.  It is a direct
refinement of jSO
(:class:`~panobbgo.heuristics.jso.JSO`) and inherits the entire L-SHADE
/ jSO asynchronous pipeline: per-slot pending dict, generation-by-count
book-keeping, archive of replaced parents, success-history memory with
the frozen jSO anchor bin, the weighted ``current-to-pbest-w/1``
mutation, the linear ``p_best`` schedule, and the asymmetric F-cap.

NL-SHADE-RSP adds three refinements on top of jSO.  This port implements
the three that the asynchronous Panobbgo pipeline can carry cleanly:

1. **Non-Linear Population Size Reduction (NLPSR)**.  jSO (like L-SHADE)
   shrinks the population *linearly* with budget progress.  NL-SHADE-RSP
   uses a non-linear schedule that reduces the population *faster* in
   the early phase::

       NP(r) = round( (NP_min − NP_init) · r^(1 − r) + NP_init )

   where ``r = len(strategy.results) / max_eval`` is budget progress in
   ``[0, 1]``.  The exponent ``1 − r`` makes ``r^(1−r)`` lie *above* the
   linear ``r`` for ``r ∈ (0, 1)`` (e.g. ``0.5^0.5 ≈ 0.707``), so a
   larger fraction of the population is dropped early — concentrating
   the late-search budget on a small, exploitative population sooner.
   At ``r = 0`` the factor is ``0`` (``NP = NP_init``); at ``r = 1`` it
   is ``1`` (``NP = NP_min``).  ``r^(1−r)`` is monotone increasing on
   ``[0, 1]``, so the population is monotone non-increasing.

2. **Rank-based Selective Pressure (RSP)**.  jSO picks the ``r1`` index
   for the differential term uniformly from the live population.
   NL-SHADE-RSP (following LSHADE-RSP, Stanovov et al. CEC 2018) biases
   that draw toward *better* individuals.  After sorting the candidate
   pool by fitness (best first), individual at sorted position ``i``
   (0 = best) of ``n`` gets the rank weight::

       w_i = k_rank · (n − i) / n + 1

   and is drawn with probability ``w_i / Σ_j w_j``.  The best
   individual's weight is ``k_rank + 1``; the worst's is
   ``k_rank / n + 1 ≈ 1``.  ``k_rank = 0`` recovers uniform selection;
   the literature default is ``k_rank = 3``.  Higher pressure pulls the
   differential mutation toward the leading basin.

3. **Adaptive (randomised) archive**.  L-SHADE / jSO cap the external
   archive of replaced parents at a *fixed* ``archive_factor · NP``.
   NL-SHADE-RSP resamples the effective cap *per generation* uniformly
   in ``[0, A_max]`` (``A_max = round(archive_factor · NP_current)``),
   randomising how much historical diversity the differential ``r2``
   draw can reach.  This is the lightweight "randomised archive size"
   variant from ``planning/SELF_IMPROVEMENT_LOOP.md`` §"Next iteration
   ideas".  Set ``adaptive_archive=False`` to recover jSO's fixed cap.

Deviations from the full CEC-2021 paper
---------------------------------------

For transparency (the Panobbgo norm is literature-faithful ports): two
NL-SHADE-RSP mechanisms are intentionally **not** ported here, because
they interact with the synchronous generation model in ways the
asynchronous pipeline does not expose cleanly:

* the *adaptive binomial / exponential crossover* blend (the paper
  adapts the probability of each crossover operator from their relative
  success), and
* the exact *success-ratio archive-probability* adaptation (the paper
  adapts ``pA`` — the probability of drawing ``r2`` from the archive —
  from the relative improvement of archive- vs population-sourced
  trials).  The randomised-cap variant above is a simpler stand-in.

Binomial crossover (inherited from jSO) and the randomised archive cap
are used instead.  These are queued as follow-ups in the planning doc.

Asynchronous execution
----------------------

Identical to jSO / L-SHADE.  The only methods that change are
:meth:`_lpsr_target` (NLPSR), :meth:`_select_r1` (RSP), and the archive
cap pair :meth:`_archive_cap` / :meth:`_end_of_generation` (randomised
archive).  Everything else — the per-slot pending dict, the
generation-by-count cadence, the memory anchor bin, the warm restart —
is inherited unchanged.

References
----------

* V. Stanovov, S. Akhmedova & E. Semenkin (2021). "NL-SHADE-RSP
  Algorithm with Adaptive Archive and Selective Pressure for CEC 2021
  Numerical Optimization." *Proceedings of CEC 2021*.  Winner of the
  CEC-2021 single-objective bound-constrained competition.
* V. Stanovov, S. Akhmedova & E. Semenkin (2018). "LSHADE Algorithm
  with Rank-Based Selective Pressure Strategy for Solving CEC 2017
  Benchmark Problems." *Proceedings of CEC 2018*.  Introduces RSP.
* J. Brest, M. S. Maučec & B. Bošković (2017). "Single Objective
  Real-Parameter Optimization: Algorithm jSO." *Proceedings of CEC
  2017*.  The jSO foundation this refines.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from panobbgo.heuristics.jso import (
    _DEFAULT_ARCHIVE_FACTOR,
    _DEFAULT_H,
    _DEFAULT_NP_INIT,
    _DEFAULT_NP_MIN,
    _DEFAULT_P_BEST_MAX,
    _DEFAULT_P_BEST_MIN,
    JSO,
)

# Default rank-based selective-pressure coefficient (Stanovov et al.).
_DEFAULT_K_RANK: float = 3.0


class NLSHADE_RSP(JSO):
    """NL-SHADE-RSP: non-linear reduction + rank selection + adaptive archive over jSO.

    Args:
        strategy: The owning :class:`~panobbgo.core.StrategyBase`.
        NP_init: Initial population size.  Default ``30``.
        NP_min: Minimum population size after non-linear reduction.
            Default ``4``.
        H: History memory size.  Default ``5`` (inherits the jSO anchor
            bin; must be ``>= 2``).
        p_best_max: Upper bound on the linear ``p_best`` schedule.
            Default ``0.25``.
        p_best_min: Lower bound on the linear ``p_best`` schedule.
            Default ``0.125``.
        archive_factor: Multiplier for the *maximum* external archive
            size.  Default ``1.0``.  With ``adaptive_archive=True`` the
            effective cap is resampled per generation uniformly in
            ``[0, round(archive_factor · NP_current)]``.
        k_rank: Rank-based selective-pressure coefficient.  Default
            ``3.0`` (Stanovov et al.).  Must be a finite float ``>= 0``.
            ``0`` recovers uniform ``r1`` selection (jSO behaviour);
            larger values bias ``r1`` more strongly toward better
            individuals.
        adaptive_archive: When ``True`` (default), resample the archive
            cap per generation (NL-SHADE-RSP).  When ``False``, use the
            fixed jSO / L-SHADE cap.
        seed: Optional seed for the per-instance RNG.
        name: Override the heuristic's display name.

    Notes:
        - All numeric arguments are validated; bad values raise
          :class:`ValueError`.
        - Like every Panobbgo heuristic, all state is per-instance.
        - When the strategy budget is unknown, the non-linear reduction
          (like LPSR) is skipped and the population stays at ``NP_init``;
          the schedules inherited from jSO fall back to their early-phase
          values.  RSP and the randomised archive are budget-independent
          and stay active.
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
        k_rank: float = _DEFAULT_K_RANK,
        adaptive_archive: bool = True,
        seed: Optional[int] = None,
        name: Optional[str] = None,
    ) -> None:
        if not np.isfinite(k_rank) or k_rank < 0.0:
            raise ValueError(f"NLSHADE_RSP: k_rank must be a finite float >= 0, got {k_rank}")
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
        self.k_rank: float = float(k_rank)
        self.adaptive_archive: bool = bool(adaptive_archive)
        # Effective archive cap for the *current* generation; ``None``
        # triggers a lazy first sample.  Reset on start / restart.
        self._rsp_archive_cap: Optional[int] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _archive_max(self) -> int:
        """Upper bound on the archive cap (``round(archive_factor · NP)``)."""
        return max(int(round(self.archive_factor * self._NP_current)), 0)

    def _sample_archive_cap(self, a_max: int) -> int:
        """Draw a per-generation archive cap uniformly in ``[0, a_max]``."""
        if a_max <= 0:
            return 0
        return int(self._rng.integers(0, a_max + 1))

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def _lpsr_target(self, progress: float) -> int:
        """Non-Linear Population Size Reduction (NLPSR).

        ``NP(r) = round((NP_min − NP_init) · r^(1 − r) + NP_init)``.
        Reduces the population faster than the linear L-SHADE schedule
        in the early phase while still reaching ``NP_min`` at ``r = 1``.
        """
        r = float(np.clip(progress, 0.0, 1.0))
        factor = r ** (1.0 - r) if r > 0.0 else 0.0
        return int(round((self.NP_min - self.NP_init) * factor + self.NP_init))

    def _select_r1(self, live: List[int], target_idx: int) -> Optional[int]:
        """Rank-based selective pressure (RSP) draw of the ``r1`` index.

        Sorts the candidate pool (live slots excluding the target) by
        fitness, assigns rank weights ``w_i = k_rank · (n − i) / n + 1``
        (best first), and draws proportionally.  ``k_rank = 0`` degrades
        to a uniform draw — but in that case the rank weights are all
        equal, so the distribution still matches jSO's uniform selection.
        """
        candidates = [i for i in live if i != target_idx]
        if not candidates:
            return None
        ordered = sorted(candidates, key=lambda i: self._fx_of(self._population[i]))  # type: ignore[arg-type]
        n = len(ordered)
        positions = np.arange(n, dtype=float)
        weights = self.k_rank * (n - positions) / n + 1.0
        probs = weights / weights.sum()
        pos = int(self._rng.choice(n, p=probs))
        return ordered[pos]

    def _archive_cap(self) -> int:
        """Per-generation randomised archive cap (NL-SHADE-RSP adaptive archive).

        When ``adaptive_archive`` is off this is the fixed jSO / L-SHADE
        cap.  When on, the cap is resampled uniformly in ``[0, A_max]``
        at generation boundaries (and lazily on first use); intermediate
        :meth:`_trim_archive` calls reuse the current sample, clipped to
        the live ``A_max`` (which may have shrunk under NLPSR).
        """
        a_max = self._archive_max()
        if not self.adaptive_archive:
            return a_max
        if self._rsp_archive_cap is None:
            self._rsp_archive_cap = self._sample_archive_cap(a_max)
        return min(self._rsp_archive_cap, a_max)

    def _end_of_generation(self) -> None:
        """Run the jSO end-of-generation update, then resample the archive cap."""
        super()._end_of_generation()
        if self.adaptive_archive:
            self._rsp_archive_cap = self._sample_archive_cap(self._archive_max())
            self._trim_archive()

    # ------------------------------------------------------------------
    # Heuristic interface
    # ------------------------------------------------------------------

    def on_start(self) -> None:
        """Allocate state and reset the per-generation archive cap."""
        super().on_start()
        self._rsp_archive_cap = None

    def on_restart(self, center, reason: str = "") -> None:
        """Reseed the population and reset the per-generation archive cap."""
        super().on_restart(center, reason)
        self._rsp_archive_cap = None
