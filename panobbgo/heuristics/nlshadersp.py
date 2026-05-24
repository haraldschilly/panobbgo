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

NL-SHADE-RSP (Stanovov, Akhmedova & Semenkin, CEC 2021) — one of the
winners of the CEC-2021 single-objective bound-constrained competition
and a direct descendant of jSO
(:class:`~panobbgo.heuristics.jso.JSO`) / L-SHADE
(:class:`~panobbgo.heuristics.lshade.LSHADE`).  It keeps jSO's
success-history adaptation, weighted ``current-to-pbest-w/1`` mutation,
linear ``p_best`` schedule, asymmetric F-cap, and frozen-anchor memory
bin, and layers on the two refinements the name encodes:

1. **Rank-based Selective Pressure (RSP)**.  jSO draws the differential
   donors ``r1`` and ``r2`` *uniformly* from the population (and the
   archive, for ``r2``).  NL-SHADE-RSP instead samples them with a
   rank-based probability so that fitter individuals are preferred —
   increasing selective pressure on the leading basin.  Each individual
   is sorted by fitness (best first) and assigned a rank::

       Rank_i = k · (NP − i) + 1            (i = 1 best … NP worst)

   and a selection probability ``pr_i = Rank_i / Σ_j Rank_j``.  The
   greediness factor ``k`` (``rank_greediness``, default ``3.0`` per
   Stanovov et al. 2018 / 2021) controls how strongly the selection
   favours the best individuals — ``k = 0`` recovers jSO's uniform
   donor selection exactly.  The ``r2`` index may also come from the
   external archive of replaced parents; archive members carry the
   minimum rank weight (they are discarded — i.e. *worse* — solutions)
   so the rank-based pressure extends across the population ∪ archive
   union in a single weighted draw.

2. **Non-Linear Population Size Reduction (NLPSR)**.  L-SHADE / jSO
   shrink the population *linearly* from ``NP_init`` to ``NP_min`` over
   the budget.  NL-SHADE-RSP uses a non-linear law::

       NP(progress) = round(NP_init + (NP_min − NP_init) · progress^(1 − progress))

   keyed on ``progress = len(strategy.results) / max_eval``.  The
   ``progress^(1 − progress)`` factor equals ``0`` at ``progress = 0``
   and ``1`` at ``progress = 1`` (same endpoints as the linear law) but
   is *concave* in between, so the population shrinks faster than linear
   early in the run and slower late — concentrating evaluations on the
   leading basin sooner.

Both refinements are confined to two small overrides
(:meth:`_generate_trial` for RSP, :meth:`_lpsr_target` for NLPSR); the
entire asynchronous pipeline — per-slot pending dict,
generation-by-count book-keeping, archive of replaced parents, warm
restart, and the jSO memory anchoring — is inherited unchanged from
:class:`~panobbgo.heuristics.jso.JSO`.

Scope note
----------

The full NL-SHADE-RSP additionally adapts the *probability* of drawing
``r2`` from the archive (rather than the population) online, and adapts
the crossover-rate update rule.  Those refinements are deferred; this
ship covers the two refinements the acronym names (NL + RSP).  See the
"NL-SHADE-RSP follow-ups" note in ``planning/SELF_IMPROVEMENT_LOOP.md``.

Asynchronous execution
----------------------

Identical to jSO / L-SHADE.  The only async-relevant additions are the
rank-weight computation (read-only over the live population, recomputed
per trial like jSO's pbest sort) and the non-linear ``_lpsr_target``
(a pure function of ``progress``).  When the strategy budget is unknown
the non-linear reduction is a no-op (population stays at ``NP_init``),
matching the L-SHADE / jSO "no budget → no LPSR" fallback.

References
----------

* V. Stanovov, S. Akhmedova & E. Semenkin (2021).  "NL-SHADE-RSP
  Algorithm with Adaptive Archive and Selective Pressure for CEC 2021
  Numerical Optimization."  *Proceedings of CEC 2021*, pp. 809-816.
* V. Stanovov, S. Akhmedova & E. Semenkin (2018).  "LSHADE Algorithm
  with Rank-Based Selective Pressure Strategy for Solving CEC 2017
  Benchmark Problems."  *Proceedings of CEC 2018*.  Introduces the
  ``Rank_i = k · (NP − i) + 1`` selective-pressure scheme.
* J. Brest, M. S. Maučec & B. Bošković (2017).  "Single Objective
  Real-Parameter Optimization: Algorithm jSO."  *Proceedings of CEC
  2017*.  The jSO foundation NL-SHADE-RSP refines.
"""

from __future__ import annotations

from typing import Dict, List, Optional

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
from panobbgo.lib import Result


# Rank greediness factor ``k`` in ``Rank_i = k · (NP − i) + 1``.  Stanovov
# et al. (2018, 2021) use ``k = 3``.  ``k = 0`` collapses to uniform donor
# selection (i.e. byte-equivalent to jSO's draw).
_DEFAULT_RANK_GREEDINESS: float = 3.0

# Rank weight assigned to external-archive members when drawing ``r2``.
# Archive entries are discarded (beaten) parents, so they sit at the
# bottom of the selective-pressure ladder — the minimum rank weight an
# in-population individual could carry (``Rank = k · 0 + 1 = 1``).
_ARCHIVE_RANK_WEIGHT: float = 1.0


class NLSHADERSP(JSO):
    """NL-SHADE-RSP: rank-based selective pressure + non-linear LPSR over jSO.

    Args:
        strategy: The owning :class:`~panobbgo.core.StrategyBase`.
        NP_init: Initial population size.  Default ``30``.
        NP_min: Minimum population size after non-linear shrinking.
            Default ``4``.
        H: History memory size.  Default ``5`` (inherited jSO value);
            must be at least ``2`` for the anchor-bin separation.
        p_best_max: Upper bound on the linear ``p_best`` schedule.
            Default ``0.25``.
        p_best_min: Lower bound on the linear ``p_best`` schedule.
            Default ``0.125``.
        archive_factor: Multiplier for the external archive size.
            Default ``1.0``.
        rank_greediness: Selective-pressure factor ``k`` in
            ``Rank_i = k · (NP − i) + 1``.  Default ``3.0`` (Stanovov et
            al.).  Must be a non-negative finite float.  ``0`` recovers
            jSO's uniform donor selection.
        seed: Optional seed for the per-instance RNG.
        name: Override the heuristic's display name.

    Notes:
        - All numeric arguments are validated; bad values raise
          :class:`ValueError`.
        - The two NL-SHADE-RSP refinements live entirely in
          :meth:`_generate_trial` (RSP) and :meth:`_lpsr_target` (NLPSR);
          all other state and behaviour is inherited from jSO.
        - When the strategy budget is unknown the non-linear reduction is
          a no-op (population pinned at ``NP_init``), matching the jSO /
          L-SHADE LPSR fallback.
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
        rank_greediness: float = _DEFAULT_RANK_GREEDINESS,
        seed: Optional[int] = None,
        name: Optional[str] = None,
    ) -> None:
        if not np.isfinite(rank_greediness) or rank_greediness < 0.0:
            raise ValueError(f"NLSHADERSP: rank_greediness must be a non-negative finite float, got {rank_greediness}")

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
        self.rank_greediness: float = float(rank_greediness)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rank_weights(self, sorted_live: List[int]) -> Dict[int, float]:
        """Rank-based selective-pressure weights keyed by population index.

        ``sorted_live`` is the live population sorted *ascending by
        fitness* (best first).  The best individual (position ``0``) gets
        the largest weight ``k · (NP − 1) + 1`` and the worst (last
        position) gets ``1``, per ``Rank_i = k · (NP − i) + 1``.
        """
        n = len(sorted_live)
        return {idx: self.rank_greediness * (n - 1 - j) + 1.0 for j, idx in enumerate(sorted_live)}

    def _rank_choice(self, pool: List[int], weights: Dict[int, float]) -> int:
        """Sample one index from ``pool`` proportional to its rank weight."""
        w = np.asarray([weights[i] for i in pool], dtype=float)
        total = float(w.sum())
        if total <= 0.0:  # defensive — weights are >= 1 by construction
            return int(self._rng.choice(np.asarray(pool)))
        return int(self._rng.choice(np.asarray(pool), p=w / total))

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def _lpsr_target(self, progress: float) -> int:
        """Non-linear population-size target (Stanovov et al. 2021).

        ``NP = round(NP_init + (NP_min − NP_init) · progress^(1 − progress))``.
        The ``progress^(1 − progress)`` factor matches the linear law's
        endpoints (``0`` at ``progress = 0``, ``1`` at ``progress = 1``)
        but is concave in between, so the population shrinks faster than
        linear early and slower late.
        """
        frac = float(progress) ** (1.0 - float(progress))
        return int(round(self.NP_init + (self.NP_min - self.NP_init) * frac))

    def _generate_trial(self, target_idx: int) -> None:
        """Build and emit one rank-pressured ``current-to-pbest-w/1`` trial.

        Differs from jSO's :meth:`~panobbgo.heuristics.jso.JSO._generate_trial`
        only in how the differential donors ``r1`` / ``r2`` are picked:
        rank-based selective pressure (best individuals preferred) instead
        of uniform.  The weighted pbest term and the linear ``p_best``
        schedule are inherited from jSO unchanged.
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

        sorted_live = sorted(live, key=lambda i: self._fx_of(self._population[i]))  # type: ignore[arg-type]
        weights = self._rank_weights(sorted_live)

        # pbest: top p% of live population by fitness (linear p_best schedule).
        p_best = self._current_p_best()
        p_count = max(int(np.ceil(p_best * len(sorted_live))), 1)
        pbest_pool = sorted_live[:p_count]
        pbest_idx = int(self._rng.choice(np.asarray(pbest_pool)))
        pbest_slot = self._population[pbest_idx]
        if not isinstance(pbest_slot, Result):
            return
        x_pbest = np.asarray(pbest_slot.x, dtype=float)

        # r1: rank-weighted selection from live population, distinct from target.
        r1_pool = [i for i in sorted_live if i != target_idx]
        if not r1_pool:
            return
        r1 = self._rank_choice(r1_pool, weights)
        r1_slot = self._population[r1]
        if not isinstance(r1_slot, Result):
            return
        x_r1 = np.asarray(r1_slot.x, dtype=float)

        # r2: rank-weighted draw over (live \ {target, r1}) ∪ archive.  Live
        # members keep their rank weight; archive members sit at the minimum
        # rank weight (they are discarded parents — the worst-rank tier).
        union_vectors: List[np.ndarray] = []
        union_weights: List[float] = []
        for i in sorted_live:
            if i == target_idx or i == r1:
                continue
            slot_i = self._population[i]
            if isinstance(slot_i, Result):
                union_vectors.append(np.asarray(slot_i.x, dtype=float))
                union_weights.append(weights[i])
        for vec in self._archive:
            union_vectors.append(vec)
            union_weights.append(_ARCHIVE_RANK_WEIGHT)
        if not union_vectors:
            return
        probs = np.asarray(union_weights, dtype=float)
        probs = probs / probs.sum()
        x_r2 = union_vectors[int(self._rng.choice(len(union_vectors), p=probs))]

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
