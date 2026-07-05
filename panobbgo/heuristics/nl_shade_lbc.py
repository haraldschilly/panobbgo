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
NL-SHADE-LBC Heuristic
======================

NL-SHADE-LBC (Stanovov, Akhmedova & Semenkin, CEC 2022) — winner of the
CEC-2022 single-objective bound-constrained competition.  It is the
direct successor to
:class:`~panobbgo.heuristics.nl_shade_rsp.NLSHADE_RSP` (CEC 2021) and
inherits its entire pipeline: the jSO-derived asynchronous DE
(per-slot pending dict, generation-by-count book-keeping, archive of
replaced parents, success-history memory with the frozen jSO anchor bin,
weighted ``current-to-pbest-w/1`` mutation, linear ``p_best`` schedule,
asymmetric F-cap, warm restart), plus NL-SHADE-RSP's Non-Linear
Population Size Reduction, Rank-based Selective Pressure on the ``r1``
draw, and randomised adaptive archive cap.

NL-SHADE-LBC adds **one further refinement** on top of NL-SHADE-RSP:
**Linear Bias Change** in the success-history memory update.  The
standard L-SHADE / jSO / NL-SHADE-RSP Lehmer mean uses fixed exponents
(``s^2 / s^1`` — i.e. order ``p = 2`` with spread ``m = 1``).
NL-SHADE-LBC generalises this to::

    L_{p,m}(s, w) = Σ(w_i · s_i^p) / Σ(w_i · s_i^{p − m})

with the *order* ``p`` **changing linearly with budget progress**::

    p_F(r)  = (1 − r) · p_F_init  + r · p_F_final
    p_CR(r) = (1 − r) · p_CR_init + r · p_CR_final

where ``r = len(strategy.results) / max_eval ∈ [0, 1]``.  The spread
``m`` is held constant.  At ``p = 2, m = 1`` the formula recovers the
standard L-SHADE weighted Lehmer mean; with the literature-default
schedule the bias toward larger F values *shrinks* and the bias toward
larger CR values *grows* across the budget.

The defaults follow Stanovov, Akhmedova & Semenkin (2022) — derived
from the MetaBox open-source reference implementation:

* ``p_F_init = 3.5``, ``p_F_final = 1.5`` — F bias starts high
  (concentrating memory on the *largest* successful F's, encouraging
  exploration) and decays to the L-SHADE-style spread by the end
  (letting the small, exploitative F's that survived late dominate).
* ``p_CR_init = 1.0``, ``p_CR_final = 1.5`` — CR bias starts low (the
  weighted mean stays close to the arithmetic mean of successful CR's,
  preserving diversity in the crossover-rate distribution) and grows
  to the same balanced point by the end.
* ``m_lbc = 1.5`` — the spread between numerator and denominator
  exponents.  At ``m = 1.0`` the formula collapses to the standard
  Lehmer mean of order ``p``.

When the strategy budget is unknown (no ``max_eval``), the progress
helper returns ``None`` and the schedule falls back to its **initial**
exponents — i.e. NL-SHADE-LBC behaves as if it were always at the start
of the search.  This is a documented, predictable fallback and matches
the convention used by :class:`~panobbgo.heuristics.lshade.LSHADE.F_schedule`.

Named regimes
-------------

The five LBC schedule kwargs can be set jointly through a named
``lbc_regime`` argument — a single composite knob the self-improvement
loop can flip as one ``categorical_choice`` bandit arm.  See
:data:`_LBC_REGIMES` for the regime-to-tuple mapping.  The four
shipped regimes are:

* ``"cec2022"`` — Stanovov, Akhmedova & Semenkin (2022) literature
  defaults; identical to passing no explicit LBC fields.
* ``"lshade"`` — recovers the standard L-SHADE / jSO / NL-SHADE-RSP
  weighted Lehmer mean at ``p = 2, m = 1`` for both F and CR (no LBC
  schedule).  Useful as a degenerate baseline arm — turns the LBC
  mechanism itself off without dropping the heuristic.
* ``"flat"`` — pure arithmetic mean (``p = 1`` throughout); the
  spread ``m_lbc = 1.5`` is preserved.
* ``"aggressive"`` — strong bias throughout the run (``p_F`` decays
  ``5 → 3``, ``p_CR`` grows ``3 → 5``); the spread ``m_lbc = 1.5``
  is preserved.

The regime is mutually exclusive with explicit per-field LBC kwargs:
passing both raises :class:`ValueError`.  Mirrors
:data:`panobbgo.heuristics.lshade._F_SCHEDULE_REGIMES` shipped
2026-06-23.

Asynchronous execution
----------------------

Identical to NL-SHADE-RSP / jSO / L-SHADE.  The only method that
changes is :meth:`_update_memory` (the Lehmer-mean computation).
Everything else — NLPSR, RSP r1 selection, the randomised adaptive
archive, the jSO frozen anchor bin and pointer-skip rule, warm
restart — is inherited unchanged.

Deviations from the full CEC-2022 paper
---------------------------------------

For transparency (the Panobbgo norm is literature-faithful ports): two
NL-SHADE-LBC mechanisms are intentionally **not** ported here, because
they interact with the synchronous generation model in ways the
asynchronous pipeline does not expose cleanly:

* the *adaptive binomial / exponential crossover blend* (inherited
  from NL-SHADE-RSP — see the same caveat there), and
* the *repetitive generation* bound-constraint handling (Panobbgo's
  asynchronous pipeline runs through ``strategy.constraint_handler`` and
  the L-SHADE midpoint-reflection repair instead).

Both are queued as follow-ups in
``planning/SELF_IMPROVEMENT_LOOP.md``.

CR-zero handling
----------------

The standard L-SHADE / jSO / NL-SHADE-RSP CR=0 terminal sentinel rule
is preserved: if every successful CR in this generation is zero, *or*
the memory bin has already been planted with the sentinel, the bin
remains terminal.  The LBC Lehmer mean is applied only to the strictly
positive subset of the success CR vector, because at ``p_CR < m_lbc``
the denominator exponent goes negative and ``0^{negative} → ∞``.

References
----------

* V. Stanovov, S. Akhmedova & E. Semenkin (2022). "NL-SHADE-LBC
  algorithm with linear parameter adaptation bias change for CEC 2022
  Numerical Optimization." *Proceedings of CEC 2022*.  Winner of the
  CEC-2022 single-objective bound-constrained competition.
* V. Stanovov, S. Akhmedova & E. Semenkin (2021). "NL-SHADE-RSP
  Algorithm with Adaptive Archive and Selective Pressure for CEC 2021
  Numerical Optimization." *Proceedings of CEC 2021*.  The
  NL-SHADE-RSP foundation this refines.
* J. Brest, M. S. Maučec & B. Bošković (2017). "Single Objective
  Real-Parameter Optimization: Algorithm jSO." *Proceedings of CEC
  2017*.  The jSO foundation NL-SHADE-RSP refines.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from panobbgo.heuristics.jso import (
    _DEFAULT_ARCHIVE_FACTOR,
    _DEFAULT_H,
    _DEFAULT_NP_INIT,
    _DEFAULT_NP_MIN,
    _DEFAULT_P_BEST_MAX,
    _DEFAULT_P_BEST_MIN,
)
from panobbgo.heuristics.lshade import _CR_TERMINAL
from panobbgo.heuristics.nl_shade_rsp import NLSHADE_RSP, _DEFAULT_K_RANK

# Defaults from Stanovov, Akhmedova & Semenkin (2022) — also published
# in the MetaBox reference implementation
# (https://github.com/MetaEvo/MetaBox/blob/master/src/baseline/bbo/nlshadelbc.py).
_DEFAULT_P_F_INIT: float = 3.5
_DEFAULT_P_F_FINAL: float = 1.5
_DEFAULT_P_CR_INIT: float = 1.0
_DEFAULT_P_CR_FINAL: float = 1.5
_DEFAULT_M_LBC: float = 1.5

# Named LBC bias-change regimes.  Each value is a 5-tuple
# ``(p_F_init, p_F_final, p_CR_init, p_CR_final, m_lbc)`` consumed by the
# constructor when the matching :func:`_normalize_lbc_regime` key is
# active.  Mirrors :data:`panobbgo.heuristics.lshade._F_SCHEDULE_REGIMES`:
# one well-curated tuple per regime instead of five independent
# float-dial arms.
#
# * ``"cec2022"`` — Stanovov, Akhmedova & Semenkin (2022) literature
#   defaults: F bias starts high and decays toward the L-SHADE balance;
#   CR bias starts low and grows toward the same balance.  Identical to
#   the constructor's per-field defaults — opting into this regime is a
#   no-op on the underlying memory update.
# * ``"lshade"`` — recovers the standard L-SHADE / jSO / NL-SHADE-RSP
#   weighted Lehmer mean (``p = 2, m = 1`` for both F and CR, with no
#   linear change across budget progress).  Useful as a degenerate
#   baseline arm — the bandit can A/B the LBC mechanism itself against
#   its non-LBC predecessor without dropping the heuristic.
# * ``"flat"`` — pure arithmetic mean (``p = 1`` and constant); the
#   spread ``m_lbc = 1.5`` is preserved.  Drops all bias toward larger
#   successful F / CR values; the success-history memory tracks the
#   centre of mass of recent successes.
# * ``"aggressive"`` — strong bias throughout the run (``p_F`` decays
#   from ``5`` to ``3``, ``p_CR`` grows from ``3`` to ``5``); the
#   spread ``m_lbc = 1.5`` is preserved.  Counterpart to ``"flat"`` —
#   sharper concentration on the largest successes than the
#   literature default.
_LBC_REGIMES: Dict[str, Tuple[float, float, float, float, float]] = {
    "cec2022": (_DEFAULT_P_F_INIT, _DEFAULT_P_F_FINAL, _DEFAULT_P_CR_INIT, _DEFAULT_P_CR_FINAL, _DEFAULT_M_LBC),
    "lshade": (2.0, 2.0, 2.0, 2.0, 1.0),
    "flat": (1.0, 1.0, 1.0, 1.0, _DEFAULT_M_LBC),
    "aggressive": (5.0, 3.0, 3.0, 5.0, _DEFAULT_M_LBC),
}


def _normalize_lbc_regime(value: Optional[str]) -> Optional[str]:
    """Validate the constructor's ``lbc_regime`` argument.

    Returns ``None`` for the disabled case (``None``) so the
    constructor can branch on a single ``is None`` check.  Returns a
    key into :data:`_LBC_REGIMES` for the active regimes.  Raises
    :class:`ValueError` for any other input.
    """
    if value is None:
        return None
    if isinstance(value, str):
        if value in _LBC_REGIMES:
            return value
        valid = tuple(sorted(_LBC_REGIMES))
        raise ValueError(f"NLSHADE_LBC: lbc_regime must be one of {valid} (or None), got {value!r}")
    raise ValueError(f"NLSHADE_LBC: lbc_regime must be a string regime name or None, got {value!r}")


# Sentinel for the five LBC float kwargs.  Lets the constructor detect
# which fields the caller explicitly passed so ``lbc_regime`` can raise
# on the ambiguous "regime + field override" case.  Not ``None`` because
# a downstream caller might legitimately pass ``None`` (e.g. via a YAML
# config) and we want that to surface as a normal ``float`` validation
# failure rather than silently activate the regime.
_UNSET: Any = object()


class NLSHADE_LBC(NLSHADE_RSP):
    """NL-SHADE-LBC: linear bias change on the SHADE memory update.

    Args:
        strategy: The owning :class:`~panobbgo.core.StrategyBase`.
        NP_init: Initial population size, or ``"auto"`` for budget-adaptive
            sizing (see :class:`~panobbgo.heuristics.lshade.LSHADE`).
            Default ``30``.
        NP_min: Minimum population size after non-linear reduction.
            Default ``4``.
        H: History memory size.  Default ``5`` (inherits the jSO
            anchor bin; must be ``>= 2``).
        p_best_max: Upper bound on the linear ``p_best`` schedule.
            Default ``0.25``.
        p_best_min: Lower bound on the linear ``p_best`` schedule.
            Default ``0.125``.
        archive_factor: Multiplier for the external archive cap.
            Default ``1.0``.
        k_rank: Rank-based selective-pressure coefficient (NL-SHADE-RSP).
            Default ``3.0``.
        adaptive_archive: When ``True`` (default), resample the archive
            cap per generation (NL-SHADE-RSP).
        p_F_init: Initial exponent of the F Lehmer mean's numerator
            (at progress ``r = 0``).  Default ``3.5``.  Mutually
            exclusive with ``lbc_regime``.
        p_F_final: Final exponent of the F Lehmer mean's numerator
            (at progress ``r = 1``).  Default ``1.5``.  Mutually
            exclusive with ``lbc_regime``.
        p_CR_init: Initial exponent of the CR Lehmer mean's numerator.
            Default ``1.0``.  Mutually exclusive with ``lbc_regime``.
        p_CR_final: Final exponent of the CR Lehmer mean's numerator.
            Default ``1.5``.  Mutually exclusive with ``lbc_regime``.
        m_lbc: Spread between numerator and denominator exponents of
            the Lehmer mean.  The denominator exponent is
            ``p − m_lbc``.  Default ``1.5``.  Must be a finite float
            ``> 0``.  At ``p = 2, m = 1`` the formula recovers the
            standard L-SHADE weighted Lehmer mean.  Mutually exclusive
            with ``lbc_regime``.
        lbc_regime: Optional named LBC regime.  One of ``"cec2022"``
            (Stanovov et al. 2022 defaults — equivalent to passing no
            explicit LBC fields), ``"lshade"`` (recovers the standard
            L-SHADE Lehmer mean at ``p = 2, m = 1``), ``"flat"``
            (pure arithmetic mean — ``p = 1`` throughout, default
            spread), or ``"aggressive"`` (strong bias throughout —
            ``p_F`` decays ``5 → 3``, ``p_CR`` grows ``3 → 5``,
            default spread).  Mutually exclusive with the five
            individual LBC float kwargs above — passing both raises
            :class:`ValueError`.  See
            :data:`_LBC_REGIMES` for the regime-to-tuple mapping.
            Default ``None`` (no preset; the individual float kwargs
            apply, all at their byte-identical CEC 2022 defaults).
        seed: Optional seed for the per-instance RNG.
        name: Override the heuristic's display name.

    Notes:
        - All numeric arguments are validated; bad values raise
          :class:`ValueError`.
        - Like every Panobbgo heuristic, all state is per-instance.
        - When the strategy budget is unknown, the LBC schedule falls
          back to its initial exponents.
        - :attr:`lbc_regime` carries the *normalized* regime name (or
          ``None`` when no preset was active) so the self-improvement
          catalog's ``categorical_choice`` rule can flip the bias
          regime as a single discrete bandit arm.
    """

    def __init__(
        self,
        strategy,
        NP_init: Union[int, str] = _DEFAULT_NP_INIT,
        NP_min: int = _DEFAULT_NP_MIN,
        H: int = _DEFAULT_H,
        p_best_max: float = _DEFAULT_P_BEST_MAX,
        p_best_min: float = _DEFAULT_P_BEST_MIN,
        archive_factor: float = _DEFAULT_ARCHIVE_FACTOR,
        k_rank: float = _DEFAULT_K_RANK,
        adaptive_archive: bool = True,
        p_F_init: float = _UNSET,
        p_F_final: float = _UNSET,
        p_CR_init: float = _UNSET,
        p_CR_final: float = _UNSET,
        m_lbc: float = _UNSET,
        lbc_regime: Optional[str] = None,
        seed: Optional[int] = None,
        name: Optional[str] = None,
    ) -> None:
        regime_key = _normalize_lbc_regime(lbc_regime)
        explicit_lbc_fields = [
            label
            for label, value in (
                ("p_F_init", p_F_init),
                ("p_F_final", p_F_final),
                ("p_CR_init", p_CR_init),
                ("p_CR_final", p_CR_final),
                ("m_lbc", m_lbc),
            )
            if value is not _UNSET
        ]
        if regime_key is not None and explicit_lbc_fields:
            raise ValueError(
                f"NLSHADE_LBC: lbc_regime={lbc_regime!r} is mutually exclusive with "
                f"explicit kwargs {explicit_lbc_fields} — pass the named regime *or* "
                f"the individual fields, not both."
            )
        if regime_key is not None:
            p_F_init, p_F_final, p_CR_init, p_CR_final, m_lbc = _LBC_REGIMES[regime_key]
        else:
            if p_F_init is _UNSET:
                p_F_init = _DEFAULT_P_F_INIT
            if p_F_final is _UNSET:
                p_F_final = _DEFAULT_P_F_FINAL
            if p_CR_init is _UNSET:
                p_CR_init = _DEFAULT_P_CR_INIT
            if p_CR_final is _UNSET:
                p_CR_final = _DEFAULT_P_CR_FINAL
            if m_lbc is _UNSET:
                m_lbc = _DEFAULT_M_LBC

        for label, v in (
            ("p_F_init", p_F_init),
            ("p_F_final", p_F_final),
            ("p_CR_init", p_CR_init),
            ("p_CR_final", p_CR_final),
            ("m_lbc", m_lbc),
        ):
            if not np.isfinite(v):
                raise ValueError(f"NLSHADE_LBC: {label} must be a finite float, got {v!r}")
        if m_lbc <= 0.0:
            raise ValueError(f"NLSHADE_LBC: m_lbc must be > 0, got {m_lbc}")

        super().__init__(
            strategy,
            NP_init=NP_init,
            NP_min=NP_min,
            H=H,
            p_best_max=p_best_max,
            p_best_min=p_best_min,
            archive_factor=archive_factor,
            k_rank=k_rank,
            adaptive_archive=adaptive_archive,
            seed=seed,
            name=name or "NLSHADE_LBC",
        )
        self.p_F_init: float = float(p_F_init)
        self.p_F_final: float = float(p_F_final)
        self.p_CR_init: float = float(p_CR_init)
        self.p_CR_final: float = float(p_CR_final)
        self.m_lbc: float = float(m_lbc)
        self.lbc_regime: Optional[str] = regime_key

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _lbc_exponent(self, p_init: float, p_final: float) -> float:
        """Linear bias change schedule.

        ``p(r) = (1 − r) · p_init + r · p_final``.  When the strategy
        budget is unknown (``_progress() is None``) the schedule falls
        back to ``p_init`` — a documented, predictable fallback.
        """
        progress = self._progress()
        if progress is None:
            return p_init
        r = float(np.clip(progress, 0.0, 1.0))
        return (1.0 - r) * p_init + r * p_final

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def _update_memory(self) -> None:
        """Apply the LBC generalized Lehmer mean for one generation.

        Identical to :meth:`JSO._update_memory` except the fixed
        ``s^2 / s^1`` exponents are replaced by the LBC schedule
        ``s^p(r) / s^(p(r) − m_lbc)``.  The jSO anchor-bin skip and
        pointer-modulo logic (write range ``[0, H − 2]``, advance
        ``% (H − 1)``) is preserved by writing through ``write_idx``
        rather than ``self._mem_ptr`` directly.
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

        # jSO-style anchor-bin skip: only bins [0, H-2] are writable.
        write_idx = self._mem_ptr
        if write_idx >= self.H - 1:
            write_idx = 0  # defensive — should be impossible by construction

        # F memory: LBC Lehmer mean.  F > 0 is guaranteed by the Cauchy
        # redraw logic, so no zero-handling is required.
        p_F = self._lbc_exponent(self.p_F_init, self.p_F_final)
        F_num = float(np.sum(w * F_arr**p_F))
        F_den = float(np.sum(w * F_arr ** (p_F - self.m_lbc)))
        if F_den > 0.0:
            self._M_F[write_idx] = float(np.clip(F_num / F_den, 0.0, 1.0))

        # CR memory: terminal-sentinel rule preserved; LBC Lehmer mean on
        # the strictly-positive subset (so ``CR^{negative}`` is never
        # evaluated at zero).
        cr_max = float(CR_arr.max())
        if cr_max <= 0.0 or self._M_CR[write_idx] < 0.0:
            self._M_CR[write_idx] = _CR_TERMINAL
        else:
            positive = CR_arr > 0.0
            w_pos = w[positive]
            CR_pos = CR_arr[positive]
            w_sum = float(w_pos.sum())
            if w_sum > 0.0:
                w_pos = w_pos / w_sum
                p_CR = self._lbc_exponent(self.p_CR_init, self.p_CR_final)
                CR_num = float(np.sum(w_pos * CR_pos**p_CR))
                CR_den = float(np.sum(w_pos * CR_pos ** (p_CR - self.m_lbc)))
                if CR_den > 0.0:
                    self._M_CR[write_idx] = float(np.clip(CR_num / CR_den, 0.0, 1.0))

        # Advance pointer over writable range only.
        self._mem_ptr = (write_idx + 1) % (self.H - 1)
