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
CEC-2021 single-objective bound-constrained competition.  It is a
success-history adaptive DE in the L-SHADE family and builds on
:class:`~panobbgo.heuristics.lshade.LSHADE`'s asynchronous pipeline
(per-slot pending dict, generation-by-count book-keeping, archive of
replaced parents, warm restart).  It is **not** a jSO: none of jSO's
``F_w`` weighting, F cap, CR floors, anchor bin, memory averaging or
``0.3 / 0.8`` memory initialisation is part of it.

The port follows the authors' reference code (``nlshade-original.cpp``,
mirrored in ``ewarchul/nlshade``) where it and the paper text disagree,
and says so.  Per generation (``NP`` = current population size,
``r = NFE / NFE_max``):

1. **Mutation** ``current-to-pbest/1``,
   ``v = x + F(x_pbest − x) + F(x_r1 − x_r2)``, indices mutually distinct
   and different from the target:

   * ``pbest`` uniform among the best ``max(2, ⌊NP · (0.2 + 0.2 r)⌋)``.
     **Paper vs. code:** the paper writes ``p`` decreasing ``0.4 → 0.2``;
     the code computes ``psize = max(2, NP · (0.2/NFE_max · NFE + 0.2))``,
     *rising* ``0.2 → 0.4``.  The code is followed
     (``p_best = 0.2``, ``p_best_end = 0.4`` on the L-SHADE schedule).
   * ``r1`` uniform from the population.
   * ``r2`` from the archive with probability ``p_A`` (uniform), else from
     the population with **rank-based selective pressure**: the individual
     at rank ``i`` (``0`` = best) is drawn with probability
     ``∝ exp(−k_rank · i / NP)``, ``k_rank = 1``.
2. **Archive probability** ``p_A`` (initially ``0.5``), updated after each
   generation from the mean improvement of the successful trials that
   used the archive (``n_A`` of them) vs. the others::

       p_A = (Δ_A / n_A) / (Δ_A / n_A + Δ_P / (n − n_A)),  clipped to [0.1, 0.9]

   and reset to ``0.5`` when no archive trial succeeded (or ``Δ_A = 0``).
   ``n_A`` counts *successful* archive trials, ``n`` all trials of the
   generation — both as in the code.
3. **Crossover**: once per generation a fair coin picks *binomial* or
   *exponential* for the whole generation.  Each generation samples
   ``NP`` values ``CR ~ N(M_CR, 0.1)`` and hands them out **sorted**, the
   smallest to the best individual.  Exponential crossover copies a
   contiguous run from a random start while ``rand < CR`` (no
   wrap-around).  Binomial crossover ignores that ``CR`` and uses
   ``CR_b = 0`` in the first half of the budget and
   ``CR_b = 2 (r − 0.5)`` after (plus the one forced component).
4. **Bounds**: an out-of-box component is resampled uniformly in the box.
5. **Memory**: ``H = 20 · D`` bins initialised at ``M_F = M_CR = 0.2``,
   updated with the plain weighted Lehmer mean ``(p = 2, m = 1)`` of the
   successful ``F`` / ``CR`` (the *sorted* ``CR`` a trial used); no
   averaging, no anchor bin, no terminal ``CR``; a generation without
   successes resets the current bin to ``0.5 / 0.5``.  ``F`` is Cauchy
   ``(M_F, 0.1)``, redrawn while ``≤ 0``, clipped at 1, with **no F cap**.
6. **NLPSR** (non-linear population size reduction)::

       NP(r) = round((NP_min − NP_init) · r^(1 − r) + NP_init)

   and the archive size ``N_A = max(⌊2.1 · NP(r)⌋, NP_min)`` shrinks with
   it.  A full archive replaces a random entry.

Asynchronous port
-----------------

A "generation" is ``NP_current`` completed trials, as for L-SHADE.  The
sorted-``CR`` hand-out becomes an order statistic: a trial for the
individual at rank ``i`` of ``n`` draws ``n`` values ``CR ~ N(M_CR, 0.1)``
(each from a random bin), sorts them and uses the ``i``-th — the same
marginal distribution the synchronous algorithm gives that rank.  The
crossover coin is drawn lazily at the first trial of each generation.

Deviations kept on purpose
--------------------------

* ``NP_init="auto"`` (``≈ 3·dim``, budget-adaptive; measured,
  ``planning/DISCOVERY_2026-09-09.md`` §17) instead of the paper's
  ``30 · D``.
* The ``p_A`` improvements are the constraint handler's *absolute*
  improvements (``calculate_improvement``), not the code's *relative*
  ``(f_old − f_new) / f_old``, which is undefined for ``f ≤ 0`` and
  meaningless under a penalty / feasibility ordering.
* The success memory stores each trial's own ``F``.  (The reference code
  stores the last ``F`` sampled in the generation for every success — a
  bug that switches the ``F`` adaptation off.)
* A trial that ties its parent keeps the parent (the code takes the trial
  on ``≤``); ranking follows the strategy's constraint handler.

References
----------

* V. Stanovov, S. Akhmedova & E. Semenkin (2021). "NL-SHADE-RSP
  Algorithm with Adaptive Archive and Selective Pressure for CEC 2021
  Numerical Optimization." *Proceedings of CEC 2021*.  Winner of the
  CEC-2021 single-objective bound-constrained competition.
* V. Stanovov, S. Akhmedova & E. Semenkin (2018). "LSHADE Algorithm
  with Rank-Based Selective Pressure Strategy for Solving CEC 2017
  Benchmark Problems." *Proceedings of CEC 2018*.  Introduces RSP.
* R. Tanabe & A. Fukunaga (2014).  "Improving the Search Performance
  of SHADE Using Linear Population Size Reduction."  *Proceedings of
  CEC 2014*.  The L-SHADE foundation.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np

from panobbgo.heuristics.lshade import LSHADE, _F_MAX_REDRAWS, _PARAM_SCALE, _TrialMeta

# Defaults from the reference code (``nlshade-original.cpp``).
_DEFAULT_NP_MIN: int = 4
_DEFAULT_H_PER_DIM: int = 20
_DEFAULT_P_BEST: float = 0.2
_DEFAULT_P_BEST_END: float = 0.4
_DEFAULT_ARCHIVE_FACTOR: float = 2.1
#: Rank-pressure exponent: ``R_i = exp(−k_rank · i / NP)``.
_DEFAULT_K_RANK: float = 1.0
_INIT_M_F: float = 0.2
_INIT_M_CR: float = 0.2
#: Value a bin is reset to after a generation without successes.
_NO_SUCCESS_M: Tuple[float, float] = (0.5, 0.5)
_P_ARCHIVE_INIT: float = 0.5
_P_ARCHIVE_CLIP: Tuple[float, float] = (0.1, 0.9)


def _default_H(strategy) -> int:
    """``H = 20 · D`` (reference code), ``5`` if the dimension is unknown."""
    try:
        dim = int(strategy.problem.dim)
    except Exception:
        dim = 0
    return _DEFAULT_H_PER_DIM * dim if dim > 0 else 5


class NLSHADE_RSP(LSHADE):
    """NL-SHADE-RSP: NLPSR, rank-based ``r2`` pressure, adaptive archive use, mixed crossover.

    Args:
        strategy: The owning :class:`~panobbgo.core.StrategyBase`.
        NP_init: Initial population size, or ``"auto"`` for budget-adaptive
            sizing (see :class:`~panobbgo.heuristics.lshade.LSHADE`).
            Default ``"auto"`` — a measured project default; the paper uses
            ``30 · D``.
        NP_min: Minimum population size after non-linear reduction.
            Default ``4``.
        H: History memory size.  Default ``None`` → ``20 · D``.
        p_best: ``pbest`` share at the start of the run.  Default ``0.2``.
        p_best_end: ``pbest`` share at the end of the run.  Default ``0.4``
            (rising, as in the reference code).
        archive_factor: Archive size per individual, ``N_A = ⌊factor · NP⌋``
            (at least ``NP_min``).  Default ``2.1``; ``0`` disables the
            archive.
        k_rank: Exponent of the rank weights ``exp(−k_rank · i / NP)`` for
            the population draw of ``r2``.  Default ``1``; ``0`` is uniform.
        warm_start: Optional archive-seeding mode; see
            :class:`~panobbgo.heuristics.lshade.LSHADE`.
        seed: Optional seed for the per-instance RNG.
        name: Override the heuristic's display name.

    Notes:
        When the strategy budget is unknown, the population and archive stay
        at their initial sizes, ``p_best`` stays at its start value and the
        binomial crossover uses ``CR_b = 0`` (the first-half value).
    """

    #: Initial ``(M_F, M_CR)``.
    INIT_MEMORY: Tuple[float, float] = (_INIT_M_F, _INIT_M_CR)
    #: ``(M_F, M_CR)`` a bin is reset to after a generation without success.
    NO_SUCCESS_MEMORY: Tuple[float, float] = _NO_SUCCESS_M

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
        warm_start: Optional[str] = None,
        seed: Optional[int] = None,
        name: Optional[str] = None,
    ) -> None:
        if not np.isfinite(k_rank) or k_rank < 0.0:
            raise ValueError(f"{type(self).__name__}: k_rank must be a finite float >= 0, got {k_rank}")
        if H is None:
            H = _default_H(strategy)
        super().__init__(
            strategy,
            NP_init=NP_init,
            NP_min=NP_min,
            H=H,
            p_best=p_best,
            p_best_end=p_best_end,
            archive_factor=archive_factor,
            warm_start=warm_start,
            seed=seed,
            name=name or type(self).__name__,
        )
        self.k_rank: float = float(k_rank)
        self._reset_generation_state()

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def _reset_generation_state(self) -> None:
        """Archive probability, its accumulators and the crossover coin."""
        #: Probability of drawing ``r2`` from the archive.
        self.p_archive: float = _P_ARCHIVE_INIT
        self._arch_delta: float = 0.0
        self._arch_n: int = 0
        self._pop_delta: float = 0.0
        #: This generation's crossover (``None`` until the first trial draws it).
        self._cross_exponential: Optional[bool] = None
        #: Memory bin of the trial being built (for ``_resample_F``).
        self._trial_bin: Optional[int] = None

    def _init_memory(self) -> None:
        """``M_F``, ``M_CR`` start at :attr:`INIT_MEMORY`."""
        self._M_F[:] = self.INIT_MEMORY[0]
        self._M_CR[:] = self.INIT_MEMORY[1]

    # ------------------------------------------------------------------
    # Trial generation hooks
    # ------------------------------------------------------------------

    def _pbest_count(self, n: int) -> int:
        """``max(2, ⌊n · p⌋)`` with ``p`` on the ``p_best → p_best_end`` schedule."""
        return max(int(n * self._current_p_best()), 2)

    def _pbest_pool(self, sorted_live: List[int], target_idx: int) -> List[int]:
        """The top :meth:`_pbest_count` slots, without the target."""
        pool = super()._pbest_pool(sorted_live, target_idx)
        others = [i for i in pool if i != target_idx]
        return others or pool

    def _select_r1(self, live: List[int], target_idx: int, pbest_idx: Optional[int] = None) -> Optional[int]:
        """``r1`` uniform from the population, distinct from target and ``pbest``."""
        pool = [i for i in live if i != target_idx and i != pbest_idx]
        if not pool:
            pool = [i for i in live if i != target_idx]
        if not pool:
            return None
        return int(self._rng.choice(np.asarray(pool)))

    def _rank_weights(self, n: int) -> np.ndarray:
        """``exp(−k_rank · i / n)`` for ranks ``i = 0 … n − 1`` (best first)."""
        return np.exp(-self.k_rank * np.arange(n, dtype=float) / float(n))

    def _select_r2(
        self, live: List[int], sorted_live: List[int], target_idx: int, r1: int, pbest_idx: Optional[int]
    ) -> Optional[Tuple[np.ndarray, bool]]:
        """``r2`` from the archive with probability :attr:`p_archive`, else rank-weighted.

        The population draw weights the individual at rank ``i`` by
        :meth:`_rank_weights` and excludes the target, ``pbest`` and ``r1``
        (the reference code redraws until distinct).
        """
        if self._archive and float(self._rng.random()) < self.p_archive:
            j = int(self._rng.integers(0, len(self._archive)))
            return self._archive[j], True
        weights = self._rank_weights(len(sorted_live))
        excluded = {target_idx, r1, pbest_idx}
        keep = [pos for pos, i in enumerate(sorted_live) if i not in excluded]
        if not keep:
            if not self._archive:
                return None
            j = int(self._rng.integers(0, len(self._archive)))
            return self._archive[j], True
        w = weights[keep]
        pos = keep[int(self._rng.choice(len(keep), p=w / w.sum()))]
        slot = self._population[sorted_live[pos]]
        return np.asarray(slot.x, dtype=float), False  # type: ignore[union-attr]

    def _sample_F(self, r: Optional[int] = None) -> float:
        """``F ~ Cauchy(M_F[r], 0.1)``, redrawn while ``≤ 0``, clipped at 1.

        ``r`` is the individual's memory bin; ``None`` draws a random one.
        """
        if r is None:
            r = int(self._rng.integers(0, self.H))
        m_f = float(self._M_F[r])
        for _ in range(_F_MAX_REDRAWS):
            f = m_f + _PARAM_SCALE * float(self._rng.standard_cauchy())
            if f > 0.0:
                return float(min(f, 1.0))
        return 0.5

    def _sample_CR_for_rank(self, rank: int, n: int) -> Tuple[float, int]:
        """The ``rank``-th smallest of ``n`` draws ``CR ~ N(M_CR[r], 0.1)`` clipped to ``[0, 1]``.

        The asynchronous form of "every individual samples its bin ``r`` and
        a ``CR`` from it, the ``NP`` values are sorted and the smallest goes to
        the best individual".  Returns ``(CR, r)`` with ``r`` the bin of the
        target's *own* draw (the first of the ``n``; all are i.i.d.), which
        its ``F`` is then sampled from — one bin per individual for both, as
        in the reference code.
        """
        n = max(int(n), 1)
        bins = self._rng.integers(0, self.H, size=n)
        crs = np.clip(self._rng.normal(self._M_CR[bins], _PARAM_SCALE), 0.0, 1.0)
        own_bin = int(bins[0])
        crs.sort()
        return float(crs[min(max(rank, 0), n - 1)]), own_bin

    def _trial_F_CR(self, target_idx: int, sorted_live: List[int]) -> Tuple[float, float]:
        """``CR`` the order statistic for the target's rank; ``F`` from the target's own bin."""
        CR, r = self._sample_CR_for_rank(sorted_live.index(target_idx), len(sorted_live))
        self._trial_bin = r
        return self._sample_F(r), CR

    def _resample_F(self) -> float:
        """A new ``F`` from the same individual's bin (NL-SHADE-LBC's regeneration)."""
        return self._sample_F(self._trial_bin)

    def _binomial_CR(self) -> float:
        """``CR_b``: ``0`` in the first half of the budget, ``2 (r − 0.5)`` after."""
        progress = self._progress()
        if progress is None or progress <= 0.5:
            return 0.0
        return 2.0 * (progress - 0.5)

    def _exponential_crossover(self, v: np.ndarray, x_target: np.ndarray, CR: float) -> np.ndarray:
        """Contiguous run from a random start, extended while ``rand < CR`` (no wrap-around)."""
        dim = self.problem.dim
        start = int(self._rng.integers(0, dim))
        end = start + 1
        while end < dim and float(self._rng.random()) < CR:
            end += 1
        u = np.array(x_target, dtype=float, copy=True)
        u[start:end] = v[start:end]
        return u

    def _crossover(self, v: np.ndarray, x_target: np.ndarray, CR: float) -> np.ndarray:
        """This generation's crossover: exponential with ``CR``, or binomial with ``CR_b``."""
        if self._cross_exponential is None:
            self._cross_exponential = bool(self._rng.random() < 0.5)
        if self._cross_exponential:
            return self._exponential_crossover(v, x_target, CR)
        return super()._crossover(v, x_target, self._binomial_CR())

    def _repair_bounds(self, u: np.ndarray, x_target: np.ndarray) -> np.ndarray:
        """Out-of-box components are resampled uniformly in the box."""
        lb = self.problem.box[:, 0]
        ub = self.problem.box[:, 1]
        bad = (u < lb) | (u > ub)
        if not np.any(bad):
            return u
        out = np.array(u, dtype=float, copy=True)
        out[bad] = self._rng.uniform(lb[bad], ub[bad])
        return out

    # ------------------------------------------------------------------
    # Archive
    # ------------------------------------------------------------------

    def _archive_cap(self) -> int:
        """``N_A = max(⌊archive_factor · NP_current⌋, NP_min)``; ``0`` disables it."""
        if self.archive_factor <= 0.0:
            return 0
        return max(int(self.archive_factor * self._NP_current), self.NP_min)

    def _archive_insert(self, parent) -> None:
        """Append while there is room, else overwrite a random entry."""
        cap = self._archive_cap()
        if cap <= 0:
            return
        x = np.array(parent.x, dtype=float, copy=True)
        if len(self._archive) < cap:
            self._archive.append(x)
        else:
            self._archive[int(self._rng.integers(0, len(self._archive)))] = x
            self._trim_archive()

    # ------------------------------------------------------------------
    # Memory and generation bookkeeping
    # ------------------------------------------------------------------

    def _cr_terminal(self, CR_arr: np.ndarray, k: int) -> bool:
        """No terminal ``CR`` in NL-SHADE-RSP."""
        return False

    def _mean_F(self, F_arr: np.ndarray, w: np.ndarray) -> Optional[float]:
        v = super()._mean_F(F_arr, w)
        return 0.5 if v is None else v

    def _mean_CR(self, CR_arr: np.ndarray, w: np.ndarray) -> Optional[float]:
        v = super()._mean_CR(CR_arr, w)
        return 0.5 if v is None else v

    def _memory_no_success(self, k: int) -> None:
        """Reset bin ``k`` to :attr:`NO_SUCCESS_MEMORY` (the pointer does not move)."""
        self._M_F[k] = self.NO_SUCCESS_MEMORY[0]
        self._M_CR[k] = self.NO_SUCCESS_MEMORY[1]

    def _record_success(self, meta: _TrialMeta, delta: float) -> None:
        """Accumulate the improvement per ``r2`` source for the ``p_A`` update."""
        super()._record_success(meta, delta)
        if getattr(meta, "from_archive", False):
            self._arch_delta += float(delta)
            self._arch_n += 1
        else:
            self._pop_delta += float(delta)

    def _update_p_archive(self, n_trials: int) -> None:
        """Adapt :attr:`p_archive` from this generation's archive vs. population successes."""
        if self._arch_n > 0:
            mean_a = self._arch_delta / self._arch_n
            rest = n_trials - self._arch_n
            mean_p = self._pop_delta / rest if rest > 0 else 0.0
            if mean_a <= 0.0 or mean_a + mean_p <= 0.0:
                self.p_archive = _P_ARCHIVE_INIT
            else:
                lo, hi = _P_ARCHIVE_CLIP
                self.p_archive = float(np.clip(mean_a / (mean_a + mean_p), lo, hi))
        else:
            self.p_archive = _P_ARCHIVE_INIT

    def _end_of_generation(self) -> None:
        """``p_A`` update, then memory + NLPSR, then a fresh crossover coin."""
        self._update_p_archive(self._gen_completed)
        self._arch_delta = 0.0
        self._arch_n = 0
        self._pop_delta = 0.0
        super()._end_of_generation()
        self._cross_exponential = None

    def _lpsr_target(self, progress: float) -> int:
        """Non-Linear Population Size Reduction (NLPSR).

        ``NP(r) = round((NP_min − NP_init) · r^(1 − r) + NP_init)``.
        Reduces the population faster than the linear L-SHADE schedule
        in the early phase while still reaching ``NP_min`` at ``r = 1``.
        """
        r = float(np.clip(progress, 0.0, 1.0))
        factor = r ** (1.0 - r) if r > 0.0 else 0.0
        return int(round((self.NP_min - self.NP_init) * factor + self.NP_init))

    # ------------------------------------------------------------------
    # Heuristic interface
    # ------------------------------------------------------------------

    def on_start(self) -> None:
        """Reset ``p_A`` and the crossover coin, then start as L-SHADE."""
        self._reset_generation_state()
        super().on_start()

    def on_restart(self, center, reason: str = "") -> None:
        """Reset ``p_A`` and the crossover coin, then restart as L-SHADE."""
        self._reset_generation_state()
        super().on_restart(center, reason)
