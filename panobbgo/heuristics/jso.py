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
jSO Heuristic
=============

jSO (Brest, Maučec & Bošković, CEC 2017) — winner of the
CEC-2017 single-objective bound-constrained competition.  jSO is a
direct refinement of L-SHADE
(:class:`~panobbgo.heuristics.lshade.LSHADE`) and inherits its
success-history adaptation, linear population reduction (LPSR), and
``current-to-pbest/1`` mutation skeleton.  Three changes lift it above
plain L-SHADE on the CEC test suites:

1. **Weighted current-to-pbest mutation** (``current-to-pbest-w/1``).
   The pbest direction is re-weighted by a phase-dependent factor
   ``F_w`` that grows with progress::

       F_w = 0.7 · F   if  progress < 0.2
       F_w = 0.8 · F   if  progress < 0.4
       F_w = 1.2 · F   otherwise

   so the algorithm explores broadly early and exploits aggressively
   late.  The differential ``F · (x_r1 − x_r2)`` term keeps the
   unweighted scaling.

2. **Linear ``p_best`` schedule**.  ``p_best`` decreases linearly from
   ``p_best_max = 0.25`` to ``p_best_min = p_best_max / 2 = 0.125``
   over the budget.  Early in the run, drawing the ``pbest`` from the
   top 25% keeps mutations diverse; once LPSR has shrunk the
   population, the top 12.5% is enough to focus on the leading
   basin.

3. **Asymmetric Cauchy-F clamping**.  Three-phase cap keyed on
   ``progress``::

       F ≤ 0.7   if  progress < 0.6
       F ≤ 0.8   if  progress < 0.9
       F ≤ 1.0   otherwise (unclamped)

   This prevents pathologically large jumps when the population is
   still big while preserving full-range mutation late in the search.
   jSO opts into the L-SHADE :attr:`~panobbgo.heuristics.lshade.LSHADE.F_schedule`
   machinery by construction; the cap is shared infrastructure rather
   than jSO-only code.  Brest et al. (2017, §III-D) document this as
   the asymmetric F-cap; earlier ports of jSO (including the initial
   2026-05-15 Panobbgo ship) implemented only the first phase.  The
   second phase is the literature-faithful completion.

Two architectural tweaks come with the algorithmic changes:

4. **Memory anchoring**.  The last memory bin (``H − 1``) is *frozen*
   at ``M_F = M_CR = 0.9`` and never updated by the SHADE Lehmer-mean
   rule.  ``_update_memory`` advances the pointer through indices
   ``[0, H − 2]``; the anchor bin is still drawn from at sampling time,
   so it stably contributes a "moderately greedy" parameter setting
   regardless of what the live success-history has learned.

5. **Initial memory values**.  ``M_F[0..H − 2]`` start at ``0.3``
   (vs L-SHADE's ``0.5``) and ``M_CR[0..H − 2]`` start at ``0.8``
   (vs ``0.5``).  These are the values Brest et al. measured to
   give good early-run behaviour across the CEC battery.

Asynchronous execution
----------------------

jSO inherits the entire async pipeline from
:class:`~panobbgo.heuristics.lshade.LSHADE`: per-slot pending dict,
generation-by-count book-keeping, archive of replaced parents, and
warm restart via :meth:`on_restart`.  jSO only overrides the small
hooks of L-SHADE's trial and memory templates: :meth:`_current_F_weight`
(``F_w``), :meth:`_current_p_best` (linear ``p_best``),
:meth:`_select_pbest` (the optional shared pbest), :meth:`_init_memory`
and :meth:`_next_mem_ptr` (the anchor bin).  The F-cap is inherited from
:meth:`LSHADE._apply_F_cap` via ``F_schedule="jso"``.

Progress measurement uses ``len(strategy.results) / max_eval`` —
the same idiom L-SHADE uses for LPSR pacing — so the F-clamping and
``F_w`` schedules stay in lock-step with the population shrink.

When the strategy budget is unknown (no ``max_eval``, zero, or
non-numeric), jSO falls back to ``progress = 0.0`` for the ``F_w``
schedule, ``p_best = p_best_max`` for the greediness schedule, and
the F-cap is bypassed entirely.  This matches L-SHADE's "no budget
→ no LPSR" fallback and keeps the heuristic safe in unmeasured
environments.

References
----------

* J. Brest, M. S. Maučec & B. Bošković (2017).  "Single Objective
  Real-Parameter Optimization: Algorithm jSO."  *Proceedings of CEC
  2017*, pp. 1311-1318.  Winner of the CEC-2017 single-objective
  bound-constrained competition.
* R. Tanabe & A. Fukunaga (2014).  "Improving the Search Performance
  of SHADE Using Linear Population Size Reduction."  *Proceedings of
  CEC 2014*.  The L-SHADE foundation jSO refines.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np

from panobbgo.heuristics.lshade import LSHADE
from panobbgo.lib import Result


# Default tuning constants — match Brest et al. (2017, jSO).
_DEFAULT_NP_MIN: int = 4
_DEFAULT_H: int = 5
_DEFAULT_P_BEST_MAX: float = 0.25
_DEFAULT_P_BEST_MIN: float = 0.125
_DEFAULT_ARCHIVE_FACTOR: float = 1.0

# Initial memory values (vs L-SHADE's 0.5 / 0.5).
_INIT_M_F: float = 0.3
_INIT_M_CR: float = 0.8

# Anchor values for the last (H - 1) memory slot.  Frozen at construction
# time and never updated by ``_update_memory``.
_ANCHOR_M_F: float = 0.9
_ANCHOR_M_CR: float = 0.9

# Weighted-mutation schedule for ``F_w``.  Three phases by progress.
_FW_PHASE1_BOUND: float = 0.2
_FW_PHASE2_BOUND: float = 0.4
_FW_PHASE1_FACTOR: float = 0.7
_FW_PHASE2_FACTOR: float = 0.8
_FW_PHASE3_FACTOR: float = 1.2


class JSO(LSHADE):
    """jSO: linear-p_best, weighted-mutation refinement of L-SHADE.

    Args:
        strategy: The owning :class:`~panobbgo.core.StrategyBase`.
        NP_init: Initial population size, or ``"auto"`` for budget-adaptive
            sizing.  Default ``"auto"``; the literature default of ``30``
            is the fallback when the budget is unknown.  See
            :class:`~panobbgo.heuristics.lshade.LSHADE` for the ``"auto"``
            sizing formula and budget notes.
        NP_min: Minimum population size after LPSR shrinking.  Default
            ``4``.
        H: History memory size.  Default ``5`` (Brest et al. report
            ``5`` outperforming the L-SHADE default of ``6`` on the CEC
            battery).  Must be at least ``2`` so the anchor bin
            ``H − 1`` is distinct from the writable bins ``[0, H − 2]``.
        p_best_max: Upper bound on the linear ``p_best`` schedule.
            Default ``0.25``.  Must lie in ``(0, 1]``.
        p_best_min: Lower bound on the linear ``p_best`` schedule.
            Default ``0.125``.  Must satisfy ``0 < p_best_min <= p_best_max``.
        archive_factor: Multiplier for the external archive size.
            Default ``1.0``.  Setting it to ``0`` disables the archive.
        warm_start: Optional archive-seeding mode; see
            :class:`~panobbgo.heuristics.lshade.LSHADE`.  Default ``None``
            (cold start).
        shared_pbest: Widen the ``pbest`` pool with the shared archive
            (``planning/DESIGN_seams_2026-09-11.md`` §2.2): the pool becomes
            the live top-``p_count`` union the best ``p_count`` *foreign*
            results (``who`` not starting with this instance's own tag) from
            the :class:`~panobbgo.analyzers.archive.Archive` analyzer, and
            ``pbest`` is drawn uniformly from that union with the instance's
            own RNG.  Without an ``Archive`` analyzer, or with none of its
            results foreign, the pool and the RNG draw are exactly today's —
            no extra draw is spent probing for foreign points.  Default
            ``False``; like ``inject`` on :class:`~panobbgo.heuristics.cma_es.CMAES`,
            this is inert without a second arm feeding the archive.
        seed: Optional seed for the per-instance RNG.
        name: Override the heuristic's display name.

    Notes:
        - All numeric arguments are validated; bad values raise
          :class:`ValueError`.
        - Memory bins ``[0, H − 2]`` are initialized to
          ``(_INIT_M_F, _INIT_M_CR) = (0.3, 0.8)`` and updated via the
          inherited weighted-Lehmer-mean rule.  Bin ``H − 1`` is frozen
          at ``(_ANCHOR_M_F, _ANCHOR_M_CR) = (0.9, 0.9)`` and never
          touched by ``_update_memory``.
        - The pointer wraps over ``[0, H − 2]`` only — the anchor bin
          is never overwritten.
        - When the strategy budget is unknown, jSO degrades gracefully:
          ``progress = 0.0`` for the F-clamp and ``F_w`` schedule, and
          ``p_best = p_best_max`` for the greediness schedule.  This
          matches L-SHADE's LPSR fallback.
    """

    def __init__(
        self,
        strategy,
        NP_init: Union[int, str] = "auto",
        NP_min: int = _DEFAULT_NP_MIN,
        H: int = _DEFAULT_H,
        p_best_max: float = _DEFAULT_P_BEST_MAX,
        p_best_min: float = _DEFAULT_P_BEST_MIN,
        archive_factor: float = _DEFAULT_ARCHIVE_FACTOR,
        warm_start: Optional[str] = None,
        shared_pbest: bool = False,
        seed: Optional[int] = None,
        name: Optional[str] = None,
    ) -> None:
        if not isinstance(H, int):
            raise ValueError(f"JSO: H must be an integer, got {H!r}")
        if H < 2:
            raise ValueError(f"JSO: H must be >= 2 (anchor bin requires at least one writable bin), got {H}")
        if not np.isfinite(p_best_max) or not (0.0 < p_best_max <= 1.0):
            raise ValueError(f"JSO: p_best_max must be in (0, 1], got {p_best_max}")
        if not np.isfinite(p_best_min) or not (0.0 < p_best_min <= 1.0):
            raise ValueError(f"JSO: p_best_min must be in (0, 1], got {p_best_min}")
        if p_best_min > p_best_max:
            raise ValueError(f"JSO: p_best_min ({p_best_min}) must be <= p_best_max ({p_best_max})")

        super().__init__(
            strategy,
            NP_init=NP_init,
            NP_min=NP_min,
            H=H,
            p_best=p_best_max,  # parent stores fixed greediness; we override per-call
            archive_factor=archive_factor,
            F_schedule="jso",  # jSO opts into the asymmetric F-cap by construction
            warm_start=warm_start,
            seed=seed,
            name=name or "JSO",
        )
        self.p_best_max: float = float(p_best_max)
        self.p_best_min: float = float(p_best_min)
        #: Shared-pbest seam (§2.2): widen the pbest pool with the best
        #: foreign results from the shared archive.
        self.shared_pbest: bool = bool(shared_pbest)
        # ``LSHADE.__init__`` planted the jSO memory through ``_init_memory``.

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_memory(self) -> None:
        """Plant the jSO initial memory and the frozen anchor bin.

        Overrides :meth:`LSHADE._init_memory`, which the base class calls at
        construction, in :meth:`on_start` and in :meth:`on_restart` — so the
        values are in place *before* the first trial is generated, including
        on the ``warm_start`` path, where ``on_start`` seeds the population
        and immediately generates a full generation of trials.
        """
        self._M_F[:] = _INIT_M_F
        self._M_CR[:] = _INIT_M_CR
        # Anchor bin is frozen at sample-time-only values.
        self._M_F[-1] = _ANCHOR_M_F
        self._M_CR[-1] = _ANCHOR_M_CR

    def _current_p_best(self) -> float:
        """Linear ``p_best`` schedule from ``p_best_max`` to ``p_best_min``.

        Overrides :meth:`LSHADE._current_p_best` (which uses the
        ``p_best`` / ``p_best_end`` annealing pair) because jSO names
        its endpoints ``p_best_max`` / ``p_best_min`` and unconditionally
        anneals — there is no constant fall-back regime.  When the
        budget is unknown the schedule falls back to
        ``p_best_max`` (the early-phase value), matching the L-SHADE
        fall-back pattern.
        """
        progress = self._progress()
        if progress is None:
            return self.p_best_max
        return float(self.p_best_max + (self.p_best_min - self.p_best_max) * progress)

    def _current_F_weight(self) -> float:
        """Three-phase factor ``c`` of the weighted pbest term, ``F_w = c · F``.

        Returns ``0.7`` / ``0.8`` / ``1.2`` by progress (Brest et al. 2017);
        :meth:`_generate_trial` multiplies it by the sampled ``F``.  Falls
        back to the early-phase factor when the budget is unknown.
        """
        progress = self._progress()
        if progress is None:
            return _FW_PHASE1_FACTOR
        if progress < _FW_PHASE1_BOUND:
            return _FW_PHASE1_FACTOR
        if progress < _FW_PHASE2_BOUND:
            return _FW_PHASE2_FACTOR
        return _FW_PHASE3_FACTOR

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def _select_pbest(self, sorted_live: List[int], target_idx: int) -> Optional[Tuple[np.ndarray, Optional[int]]]:
        """``pbest`` from the linear-``p_best`` pool, optionally widened (§2.2).

        With ``shared_pbest=True`` the pool is widened with the best foreign
        results from the shared archive.  An empty ``foreign`` (no
        ``shared_pbest``, no ``Archive`` analyzer, or nothing foreign in it
        yet) falls through to exactly the L-SHADE draw — no extra RNG call is
        spent finding that out.
        """
        foreign: List[Result] = []
        if self.shared_pbest:
            archive = self._archive_analyzer()
            if archive is not None:
                foreign = archive.top_k(self._pbest_count(len(sorted_live)), exclude_who=self.name)
        if not foreign:
            return super()._select_pbest(sorted_live, target_idx)

        pbest_pool = sorted_live[: self._pbest_count(len(sorted_live))]
        union: List[Result] = [self._population[i] for i in pbest_pool] + list(foreign)  # type: ignore[misc]
        j = int(self._rng.integers(0, len(union)))
        pbest_slot = union[j]
        if not isinstance(pbest_slot, Result):
            return None
        return np.asarray(pbest_slot.x, dtype=float), (pbest_pool[j] if j < len(pbest_pool) else None)

    def _next_mem_ptr(self, k: int) -> int:
        """Advance the pointer over the writable bins ``[0, H − 2]`` only."""
        return (k + 1) % (self.H - 1)
