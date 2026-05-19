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

3. **Cauchy-F clamping**.  When ``progress < 0.6``, sampled ``F``
   values above ``0.7`` are clamped to ``0.7``.  This prevents
   pathologically large jumps when the population is still big and
   the surrogate landscape has not yet been explored.

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
warm restart via :meth:`on_restart`.  The only methods that change
are :meth:`_sample_F_CR` (Cauchy clamping), :meth:`_generate_trial`
(``F_w`` weighting + linear ``p_best``), and :meth:`_update_memory`
(skip the anchor bin, advance pointer mod ``H − 1``).

Progress measurement uses ``len(strategy.results) / max_eval`` —
the same idiom L-SHADE uses for LPSR pacing — so the F-clamping and
``F_w`` schedules stay in lock-step with the population shrink.

When the strategy budget is unknown (no ``max_eval``, zero, or
non-numeric), jSO falls back to ``progress = 0.0`` for the F-clamp
and ``F_w`` schedule, and ``p_best = p_best_max`` for the greediness
schedule.  This matches L-SHADE's "no budget → no LPSR" fallback and
keeps the heuristic safe in unmeasured environments.

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

from typing import Optional

import numpy as np

from panobbgo.heuristics.lshade import (
    LSHADE,
    _CR_TERMINAL,
    _F_MAX_REDRAWS,
    _PARAM_SCALE,
)
from panobbgo.lib import Result


# Default tuning constants — match Brest et al. (2017, jSO).
_DEFAULT_NP_INIT: int = 30
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

# Cauchy-F clamping schedule: F is clipped at ``_F_CLAMP_VALUE`` while
# ``progress < _F_CLAMP_PROGRESS_BOUND``.
_F_CLAMP_PROGRESS_BOUND: float = 0.6
_F_CLAMP_VALUE: float = 0.7

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
        NP_init: Initial population size.  Default ``30``.  See
            :class:`~panobbgo.heuristics.lshade.LSHADE` for budget
            sizing notes.
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
        NP_init: int = _DEFAULT_NP_INIT,
        NP_min: int = _DEFAULT_NP_MIN,
        H: int = _DEFAULT_H,
        p_best_max: float = _DEFAULT_P_BEST_MAX,
        p_best_min: float = _DEFAULT_P_BEST_MIN,
        archive_factor: float = _DEFAULT_ARCHIVE_FACTOR,
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
            seed=seed,
            name=name or "JSO",
        )
        self.p_best_max: float = float(p_best_max)
        self.p_best_min: float = float(p_best_min)

        # Re-initialize memory bins per jSO defaults (L-SHADE used 0.5).
        self._M_F[:] = _INIT_M_F
        self._M_CR[:] = _INIT_M_CR
        # Anchor bin is frozen at sample-time-only values.
        self._M_F[-1] = _ANCHOR_M_F
        self._M_CR[-1] = _ANCHOR_M_CR

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _progress(self) -> float:
        """Return ``len(strategy.results) / max_eval`` clipped to ``[0, 1]``.

        Falls back to ``0.0`` when the budget is unknown — matching
        L-SHADE's LPSR fallback so an unmeasured environment leaves
        the heuristic in its early-phase regime.
        """
        max_eval = self._max_eval()
        if max_eval is None:
            return 0.0
        try:
            current = float(len(self.strategy.results))
        except Exception:
            return 0.0
        return float(np.clip(current / max_eval, 0.0, 1.0))

    def _current_p_best(self) -> float:
        """Linear ``p_best`` schedule from ``p_best_max`` to ``p_best_min``."""
        return float(self.p_best_max + (self.p_best_min - self.p_best_max) * self._progress())

    def _current_F_weight(self) -> float:
        """Three-phase ``F_w`` factor for the weighted current-to-pbest term."""
        progress = self._progress()
        if progress < _FW_PHASE1_BOUND:
            return _FW_PHASE1_FACTOR
        if progress < _FW_PHASE2_BOUND:
            return _FW_PHASE2_FACTOR
        return _FW_PHASE3_FACTOR

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def _sample_F_CR(self) -> tuple[float, float]:
        """Draw ``(F, CR)`` from a random bin, then clamp F per jSO."""
        # Identical to LSHADE._sample_F_CR up to the F clamp at the end.
        r = int(self._rng.integers(0, self.H))
        m_cr = float(self._M_CR[r])
        if m_cr < 0:
            CR = 0.0
        else:
            CR = float(self._rng.normal(m_cr, _PARAM_SCALE))
            CR = float(np.clip(CR, 0.0, 1.0))

        m_f = float(self._M_F[r])
        F = 0.5
        for _ in range(_F_MAX_REDRAWS):
            f = m_f + _PARAM_SCALE * float(self._rng.standard_cauchy())
            if f > 0.0:
                F = float(min(f, 1.0))
                break

        # jSO Cauchy-F clamping: limit early-phase mutation magnitude.
        if self._progress() < _F_CLAMP_PROGRESS_BOUND and F > _F_CLAMP_VALUE:
            F = _F_CLAMP_VALUE
        return F, CR

    def _generate_trial(self, target_idx: int) -> None:
        """Build and emit one ``current-to-pbest-w/1`` trial vector.

        Differs from L-SHADE's ``current-to-pbest/1`` by:

        * applying the linear ``p_best`` schedule (``_current_p_best``)
          when picking the pbest pool size;
        * weighting the pbest direction by ``F_w`` (``_current_F_weight``)
          while keeping the differential ``F`` unchanged.
        """
        live = self._live_indices()
        if len(live) < 4 or target_idx not in live:
            return
        slot = self._population[target_idx]
        if not isinstance(slot, Result):
            return

        F, CR = self._sample_F_CR()
        F_w = self._current_F_weight()
        x_target = np.asarray(slot.x, dtype=float)

        # pbest: top p% of live population by fitness, with linear p_best schedule.
        sorted_live = sorted(live, key=lambda i: self._fx_of(self._population[i]))  # type: ignore[arg-type]
        p_best = self._current_p_best()
        p_count = max(int(np.ceil(p_best * len(sorted_live))), 1)
        pbest_pool = sorted_live[:p_count]
        pbest_idx = int(self._rng.choice(np.asarray(pbest_pool)))
        pbest_slot = self._population[pbest_idx]
        if not isinstance(pbest_slot, Result):
            return
        x_pbest = np.asarray(pbest_slot.x, dtype=float)

        # r1 from live population, distinct from target.
        r1_pool = [i for i in live if i != target_idx]
        if not r1_pool:
            return
        r1 = int(self._rng.choice(np.asarray(r1_pool)))
        r1_slot = self._population[r1]
        if not isinstance(r1_slot, Result):
            return
        x_r1 = np.asarray(r1_slot.x, dtype=float)

        # r2 from (live ∪ archive) \ {target, r1}.
        union: list[np.ndarray] = []
        for i in live:
            if i == target_idx or i == r1:
                continue
            slot_i = self._population[i]
            if isinstance(slot_i, Result):
                union.append(np.asarray(slot_i.x, dtype=float))
        union.extend(self._archive)
        if not union:
            return
        x_r2 = union[int(self._rng.integers(0, len(union)))]

        # Mutation: current-to-pbest-w/1 — weighted pbest term, unweighted differential.
        v = x_target + F_w * (x_pbest - x_target) + F * (x_r1 - x_r2)
        v = self._reflect_bounds(v, x_target)

        # Binomial crossover with at least one component swapped.
        dim = self.problem.dim
        cross = self._rng.random(dim) < CR
        j_rand = int(self._rng.integers(0, dim))
        cross[j_rand] = True
        u = np.where(cross, v, x_target)

        self._emit_trial(u, target_idx, F, CR)

    def _update_memory(self) -> None:
        """Apply the Lehmer-mean memory update, skipping the anchor bin.

        Identical to :meth:`LSHADE._update_memory` except the writable
        index range is ``[0, H − 2]`` (the anchor bin ``H − 1`` is
        frozen).  The pointer wraps modulo ``H − 1``.
        """
        if not self._success_F:
            return
        if self.H < 2:  # defensive — constructor enforces H >= 2
            return

        F_arr = np.asarray(self._success_F, dtype=float)
        CR_arr = np.asarray(self._success_CR, dtype=float)
        delta_arr = np.asarray(self._success_delta, dtype=float)
        total = float(delta_arr.sum())
        if total > 0.0:
            w = delta_arr / total
        else:
            w = np.full_like(delta_arr, 1.0 / len(delta_arr))

        # Writable pointer must stay strictly below the anchor index.
        write_idx = self._mem_ptr
        if write_idx >= self.H - 1:
            write_idx = 0  # defensive — should be impossible by construction

        # F: weighted Lehmer mean.
        F_num = float(np.sum(w * F_arr * F_arr))
        F_den = float(np.sum(w * F_arr))
        if F_den > 0.0:
            self._M_F[write_idx] = float(np.clip(F_num / F_den, 0.0, 1.0))

        # CR: terminal sentinel rule + weighted Lehmer mean.
        cr_max = float(CR_arr.max())
        if cr_max <= 0.0 or self._M_CR[write_idx] < 0.0:
            self._M_CR[write_idx] = _CR_TERMINAL
        else:
            CR_num = float(np.sum(w * CR_arr * CR_arr))
            CR_den = float(np.sum(w * CR_arr))
            if CR_den > 0.0:
                self._M_CR[write_idx] = float(np.clip(CR_num / CR_den, 0.0, 1.0))

        # Advance pointer over writable range only.
        self._mem_ptr = (write_idx + 1) % (self.H - 1)

    # ------------------------------------------------------------------
    # Heuristic interface
    # ------------------------------------------------------------------

    def on_start(self) -> None:
        """Allocate state, plant the anchor bin, and emit initial trials."""
        super().on_start()
        # ``LSHADE.on_start`` resets memory to ``0.5`` — re-stamp the jSO
        # initial values *and* the frozen anchor bin afterwards.
        self._M_F[:] = _INIT_M_F
        self._M_CR[:] = _INIT_M_CR
        self._M_F[-1] = _ANCHOR_M_F
        self._M_CR[-1] = _ANCHOR_M_CR

    def on_restart(self, center, reason: str = "") -> None:
        """Drop in-flight state and reseed the population around ``center``.

        Mirrors :meth:`LSHADE.on_restart` but re-stamps the jSO initial
        memory values and the frozen anchor bin so the warm-restart
        memory matches construction-time.
        """
        super().on_restart(center, reason)
        # ``LSHADE.on_restart`` resets memory to ``0.5`` — re-stamp jSO.
        self._M_F[:] = _INIT_M_F
        self._M_CR[:] = _INIT_M_CR
        self._M_F[-1] = _ANCHOR_M_F
        self._M_CR[-1] = _ANCHOR_M_CR
