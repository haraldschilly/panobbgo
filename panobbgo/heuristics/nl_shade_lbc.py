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
successor of :class:`~panobbgo.heuristics.nl_shade_rsp.NLSHADE_RSP`
(CEC 2021) and shares its asynchronous pipeline, the non-linear
population size reduction (NLPSR), the rank-based selective pressure on
the population draw of ``r2`` and the fitness-sorted ``CR`` hand-out.
Where it differs from NL-SHADE-RSP (paper §III and Algorithm 1):

* **Rank pressure** ``R_i = exp(−4 i / NP)`` (``k_rank = 4``), on ``r2``
  only and only when ``r2`` comes from the population.
* **Archive**: fixed use probability ``p_A = 0.5``; size
  ``N_A = 1.0 · NP``, shrinking with NLPSR.  A full archive takes the new
  parent in place of a randomly probed entry that is *worse* than it —
  up to ``|A|`` probes, then a random entry.
* ``pbest`` among the best ``max(2, ⌊NP (0.2 + 0.1 r)⌋)`` (rising
  ``0.2 → 0.3``).
* **Binomial crossover only**, with the fitness-sorted sampled ``CR``.
* **Bounds**: a trial that leaves the box is generated again — new ``F``,
  ``pbest``, ``r1``, ``r2`` — up to 100 times; what is still outside is
  repaired with the midpoint target ``(bound + x) / 2``.
* **Memory**: ``H = 20 · D`` bins initialised at ``M_F = 0.5``,
  ``M_CR = 0.9``; a generation without successes resets the current bin
  to those values (as in the MetaBox port; the paper is silent).
* No jSO machinery: no F cap, no anchor bin, no ``F_w``, no CR floors,
  no memory averaging.

Its namesake is **Linear Bias Change** in the success-history memory
update.  The standard L-SHADE / jSO / NL-SHADE-RSP Lehmer mean uses fixed exponents
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

Identical to NL-SHADE-RSP / L-SHADE.  NL-SHADE-LBC overrides the hooks
:meth:`_mean_F` / :meth:`_mean_CR` (the LBC Lehmer mean),
:meth:`_crossover` (binomial only), :meth:`_repair_bounds` (midpoint),
:meth:`_archive_insert` (fitness-probed replacement),
:meth:`_update_p_archive` (``p_A`` stays ``0.5``) and
:data:`_TRIAL_ATTEMPTS` (the out-of-bounds regeneration).

Deviations kept on purpose
--------------------------

* ``NP_init="auto"`` with the class coefficient :attr:`AUTO_DIM_COEF`
  ``= 4`` (``≈ 4·dim``, budget-adaptive; measured,
  ``planning/DISCOVERY_2026-09-09.md`` §17/§20/§24) instead of the paper's
  ``23 · D``.
* Asynchronous generations and the order-statistic ``CR`` (see
  :class:`~panobbgo.heuristics.nl_shade_rsp.NLSHADE_RSP`); ranking and
  success weights follow the strategy's constraint handler.
* Algorithm 1 writes the archive-probe stop condition as
  ``f(A_ra) < f(x_i)``; the text ("if the fitness of the new point is
  better than of the selected one, the replacement occurs") is followed.

CR-zero handling
----------------

There is no terminal ``CR`` sentinel (none in the paper).  The LBC
Lehmer mean is applied only to the strictly positive subset of the
success CR vector, because at ``p_CR < m_lbc`` the denominator exponent
goes negative and ``0^{negative} → ∞``; a generation whose successes all
used ``CR = 0`` resets the ``CR`` bin to ``0.9`` (LBC's initial / reset value,
the analogue of NL-SHADE-RSP's ``0.5`` fallback).

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

from panobbgo.heuristics.lshade import LSHADE
from panobbgo.lib import Result
from panobbgo.heuristics.nl_shade_rsp import _DEFAULT_NP_MIN, NLSHADE_RSP

# Algorithm 1 of the CEC-2022 paper.
_DEFAULT_P_BEST: float = 0.2
_DEFAULT_P_BEST_END: float = 0.3
_DEFAULT_ARCHIVE_FACTOR: float = 1.0
_DEFAULT_K_RANK: float = 4.0
_INIT_M_F: float = 0.5
_INIT_M_CR: float = 0.9
#: Out-of-bounds regeneration attempts before the midpoint repair.
_BOUND_RESAMPLES: int = 100

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
            Default ``"auto"``; the literature default of ``30`` is the
            fallback when the budget is unknown.
        NP_min: Minimum population size after non-linear reduction.
            Default ``4``.
        H: History memory size.  Default ``None`` → ``20 · D``.
        p_best: ``pbest`` share at the start of the run.  Default ``0.2``.
        p_best_end: ``pbest`` share at the end of the run.  Default ``0.3``.
        archive_factor: Archive size per individual (``N_A = ⌊factor · NP⌋``,
            at least ``NP_min``).  Default ``1.0``; ``0`` disables it.
        k_rank: Exponent of the rank weights ``exp(−k_rank · i / NP)`` for
            the population draw of ``r2``.  Default ``4``.
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
        warm_start: Optional archive-seeding mode; see
            :class:`~panobbgo.heuristics.lshade.LSHADE`.  Default ``None``
            (cold start).
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

    #: NL-SHADE-LBC wants a bigger swarm than the rest of the L-SHADE
    #: lineage: its measured AOCC optimum is 8 / 20 / 30 at ``d`` = 2 / 5 / 10
    #: against jSO's and L-SHADE's 6 / 15 / 20-30, i.e. ``4·dim`` rather than
    #: ``3·dim`` (``planning/DISCOVERY_2026-09-09.md`` §17, §20, §24).  A
    #: 12-seed paired A/B of ``4·dim`` against the shipped ``3·dim`` ``"auto"``
    #: rule gives +0.0523 AOCC [+0.0098, +0.0948], 10/12 seeds positive.
    AUTO_DIM_COEF: float = 4.0

    INIT_MEMORY: Tuple[float, float] = (_INIT_M_F, _INIT_M_CR)
    NO_SUCCESS_MEMORY: Tuple[float, float] = (_INIT_M_F, _INIT_M_CR)
    _TRIAL_ATTEMPTS: int = _BOUND_RESAMPLES

    def __init__(
        self,
        strategy,
        NP_init: Union[int, str] = "auto",
        NP_min: int = _DEFAULT_NP_MIN,
        H: Optional[int] = None,
        p_best: float = _DEFAULT_P_BEST,
        p_best_end: float = _DEFAULT_P_BEST_END,
        archive_factor: float = _DEFAULT_ARCHIVE_FACTOR,
        k_rank: float = _DEFAULT_K_RANK,
        p_F_init: float = _UNSET,
        p_F_final: float = _UNSET,
        p_CR_init: float = _UNSET,
        p_CR_final: float = _UNSET,
        m_lbc: float = _UNSET,
        lbc_regime: Optional[str] = None,
        warm_start: Optional[str] = None,
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
            p_best=p_best,
            p_best_end=p_best_end,
            archive_factor=archive_factor,
            k_rank=k_rank,
            warm_start=warm_start,
            seed=seed,
            name=name or "NLSHADE_LBC",
        )
        self.p_F_init: float = float(p_F_init)
        self.p_F_final: float = float(p_F_final)
        self.p_CR_init: float = float(p_CR_init)
        self.p_CR_final: float = float(p_CR_final)
        self.m_lbc: float = float(m_lbc)
        self.lbc_regime: Optional[str] = regime_key
        #: ``id(vector) -> (vector, Result)``: the fitness source of archived
        #: vectors, for the fitness-probed replacement.
        self._archive_src: Dict[int, Tuple[np.ndarray, Result]] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _lbc_exponent(self, p_init: float, p_final: float) -> float:
        """Linear bias change schedule.

        ``p(r) = (1 − r) · p_init + r · p_final``.  When the strategy
        budget is unknown (``budget_progress() is None``) the schedule falls
        back to ``p_init`` — a documented, predictable fallback.
        """
        progress = self.budget_progress()
        if progress is None:
            return p_init
        r = float(np.clip(progress, 0.0, 1.0))
        return (1.0 - r) * p_init + r * p_final

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def _mean_F(self, F_arr: np.ndarray, w: np.ndarray) -> Optional[float]:
        """LBC Lehmer mean of the successful ``F``: ``Σ w·F^p / Σ w·F^(p − m)``.

        ``p = p_F(r)`` follows the linear bias change; ``F > 0`` by the
        Cauchy redraw, so no zero handling is needed.
        """
        p_F = self._lbc_exponent(self.p_F_init, self.p_F_final)
        v = self._weighted_lehmer(F_arr, w, p_F, self.m_lbc)
        return self.NO_SUCCESS_MEMORY[0] if v is None else v

    def _mean_CR(self, CR_arr: np.ndarray, w: np.ndarray) -> Optional[float]:
        """LBC Lehmer mean of the successful ``CR`` on the strictly positive subset.

        ``p_CR − m_lbc`` is negative for the default schedule, so
        ``CR^(p − m)`` is never evaluated at zero.  An undefined mean (every
        successful ``CR`` is 0) gives LBC's own reset value ``0.9`` — the
        analogue of NL-SHADE-RSP's ``0.5`` fallback (the MetaBox port returns
        ``0.5`` for ``F`` and ``0.9`` for ``CR`` there).
        """
        p_CR = self._lbc_exponent(self.p_CR_init, self.p_CR_final)
        v = self._weighted_lehmer(CR_arr, w, p_CR, self.m_lbc, positive_only=True)
        return self.NO_SUCCESS_MEMORY[1] if v is None else v

    def _crossover(self, v: np.ndarray, x_target: np.ndarray, CR: float) -> np.ndarray:
        """Binomial crossover only, with the fitness-sorted ``CR`` (no exponential, no ``CR_b``)."""
        return LSHADE._crossover(self, v, x_target, CR)

    def _repair_bounds(self, u: np.ndarray, x_target: np.ndarray) -> np.ndarray:
        """Midpoint target ``(bound + x) / 2`` for what is still outside after the regeneration."""
        return self._reflect_bounds(u, x_target)

    def _update_p_archive(self, n_trials: int) -> None:
        """NL-SHADE-LBC uses a fixed archive probability ``p_A = 0.5``."""
        return

    def _archive_insert(self, parent: Result) -> None:
        """Paper §III: fill up; when full, probe random entries for one worse than ``parent``.

        Up to ``|A|`` random probes; the first entry whose fitness is worse
        than the parent's (by the handler's ranking key) is replaced.  If
        none is found the last probed (random) entry is replaced.  Entries
        of unknown fitness (seeded by a warm start) count as worse.
        """
        cap = self._archive_cap()
        if cap <= 0:
            return
        x = np.array(parent.x, dtype=float, copy=True)
        if len(self._archive) < cap:
            self._archive.append(x)
            self._remember_archived(x, parent)
            return
        parent_key = self._rank_of(parent)
        n = len(self._archive)
        j = 0
        for _ in range(n):
            j = int(self._rng.integers(0, n))
            src = self._archived_result(self._archive[j])
            if src is None or parent_key < self._rank_of(src):
                break
        self._archive[j] = x
        self._remember_archived(x, parent)
        self._trim_archive()

    def _remember_archived(self, x: np.ndarray, parent: Result) -> None:
        """Keep the fitness source of an archived vector (keyed by the array object)."""
        src = self._archive_src
        src[id(x)] = (x, parent)
        if len(src) > 4 * max(len(self._archive), 1) + 16:
            live = {id(a) for a in self._archive}
            for key in [k for k in src if k not in live]:
                del src[key]

    def _archived_result(self, x: np.ndarray) -> Optional[Result]:
        """The :class:`~panobbgo.lib.Result` an archive vector came from, if known."""
        entry = self._archive_src.get(id(x))
        if entry is None or entry[0] is not x:
            return None
        return entry[1]
