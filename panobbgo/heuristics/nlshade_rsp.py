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
CEC-2021 single-objective bound-constrained competition.  NL-SHADE-RSP
is a direct refinement of jSO
(:class:`~panobbgo.heuristics.jso.JSO`) and inherits its entire async
pipeline: weighted ``current-to-pbest-w/1`` mutation, linear ``p_best``
schedule (``0.25 → 0.125``), three-phase asymmetric F-cap, success-history
``(F, CR)`` adaptation with frozen anchor bin, and Linear Population Size
Reduction (LPSR).  Two refinements lift it above jSO on the CEC test
suites:

1. **Rank-based Selective Pressure (RSP)**.  Instead of sampling ``r1``
   uniformly from the live population for the differential
   ``F · (x_r1 − x_r2)`` term, NL-SHADE-RSP samples with probability
   proportional to ``(NP − rank)`` so better individuals are picked more
   often.  Rank ``0`` (the best) gets weight ``NP``; rank ``NP − 1`` (the
   worst) gets weight ``1``.  This concentrates the differential search
   direction on the leading basin while still keeping every individual
   reachable.

2. **Adaptive archive size**.  Instead of holding the archive at a fixed
   ``A_max = ceil(archive_factor · NP_current)`` cap, NL-SHADE-RSP samples
   a fresh ``arc_size_t = randint(0, A_max + 1)`` at the start of each
   asynchronous generation and trims the archive down to that size.
   The effective archive contribution to ``r2`` sampling therefore varies
   per generation — a "diversity-of-replaced-parents" knob the bandit can
   tune by retuning ``archive_factor``.

Both refinements are implemented as overrides of the trial-generation
machinery (``_generate_trial``) and the archive-trimming routine
(``_trim_archive``), leaving the rest of the L-SHADE / jSO pipeline
unchanged.  In particular, NL-SHADE-RSP keeps:

* The jSO three-phase asymmetric F-cap (``F_schedule=True``).
* The jSO weighted-mutation ``F_w`` schedule (``0.7 / 0.8 / 1.2``).
* The jSO linear ``p_best`` schedule (``p_best_max → p_best_min``).
* The jSO frozen-anchor memory bin (``(0.9, 0.9)`` at index ``H − 1``).
* The jSO initial memory values (``M_F = 0.3``, ``M_CR = 0.8``).
* L-SHADE's LPSR pacing (``len(strategy.results) / max_eval``).

Asynchronous execution
----------------------

NL-SHADE-RSP inherits the entire async pipeline from L-SHADE / jSO: the
per-slot pending dict, generation-by-count book-keeping, archive of
replaced parents, and warm restart via :meth:`on_restart`.  The only
async-relevant change is the adaptive archive cap that refreshes at the
start of each async generation (``_end_of_generation`` calls
``_refresh_arc_size_t`` before delegating to the base implementation).
When the strategy budget is unknown, the heuristic degrades to constant
``A_max`` archive sizing — matching L-SHADE's "no budget → no LPSR"
fallback for the population shrinker.

References
----------

* V. Stanovov, S. Akhmedova & E. Semenkin (2021).  "NL-SHADE-RSP
  Algorithm with Adaptive Archive and Selective Pressure for CEC 2021
  Numerical Optimization."  *2021 IEEE Congress on Evolutionary
  Computation (CEC)*, pp. 809-816.  Winner of the CEC-2021
  single-objective bound-constrained competition.
* J. Brest, M. S. Maučec & B. Bošković (2017).  "Single Objective
  Real-Parameter Optimization: Algorithm jSO."  *Proceedings of CEC
  2017*.  Winner of the CEC-2017 single-objective bound-constrained
  competition.  The jSO foundation NL-SHADE-RSP refines.
* R. Tanabe & A. Fukunaga (2014).  "Improving the Search Performance
  of SHADE Using Linear Population Size Reduction."  *Proceedings of
  CEC 2014*.  The L-SHADE foundation jSO refines.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from panobbgo.heuristics.jso import (
    JSO,
    _ANCHOR_M_CR,
    _ANCHOR_M_F,
    _DEFAULT_ARCHIVE_FACTOR,
    _DEFAULT_H,
    _DEFAULT_NP_INIT,
    _DEFAULT_NP_MIN,
    _DEFAULT_P_BEST_MAX,
    _DEFAULT_P_BEST_MIN,
    _INIT_M_CR,
    _INIT_M_F,
)
from panobbgo.lib import Result


class NLSHADERSP(JSO):
    """NL-SHADE-RSP: rank-based selection pressure + adaptive archive on top of jSO.

    Args:
        strategy: The owning :class:`~panobbgo.core.StrategyBase`.
        NP_init: Initial population size.  Default ``30`` — inherited
            from :class:`~panobbgo.heuristics.jso.JSO`.
        NP_min: Minimum population size after LPSR shrinking.  Default
            ``4``.
        H: History memory size.  Default ``5`` (jSO's value — one more
            writable bin than L-SHADE's ``6`` minus the anchor bin).
            Must be at least ``2``.
        p_best_max: Upper bound on the linear ``p_best`` schedule.
            Default ``0.25``.
        p_best_min: Lower bound on the linear ``p_best`` schedule.
            Default ``0.125``.
        archive_factor: Multiplier for the *maximum* external archive
            size; each generation a fresh ``arc_size_t`` is sampled
            uniformly from ``[0, ceil(archive_factor · NP_current)]``.
            Default ``1.0``.  Setting it to ``0`` collapses the
            adaptive cap to ``arc_size_t ≡ 0`` (no archive, like
            disabling it on L-SHADE).
        seed: Optional seed for the per-instance RNG.
        name: Override the heuristic's display name.

    Notes:
        - Like every Panobbgo heuristic, all state is per-instance.
        - The rank-based weights are linear: rank ``0`` (best) gets
          weight ``NP``; rank ``NP − 1`` (worst) gets weight ``1``.
          Sampling probability is ``w_i / Σ w_j`` over the candidates
          (live population minus the target slot).
        - The adaptive archive cap is refreshed in
          :meth:`_end_of_generation`, which fires every ``NP_current``
          completed evolutionary trials.  Between refreshes the cap is
          constant, so successful trials that fall inside the same
          generation share one cap value.
        - When LPSR has shrunk the population to ``NP_min`` and the
          adaptive cap is small, the archive may be empty for many
          generations in a row.  The trial-generation path falls back
          to ``r2`` from the live population alone in that case
          (inherited from :meth:`JSO._generate_trial`).
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
        super().__init__(
            strategy,
            NP_init=NP_init,
            NP_min=NP_min,
            H=H,
            p_best_max=p_best_max,
            p_best_min=p_best_min,
            archive_factor=archive_factor,
            seed=seed,
            name=name or "NLSHADERSP",
        )
        # Adaptive archive cap.  Initialised to the full ``A_max`` so the
        # first generation behaves like jSO until ``_end_of_generation``
        # rolls a fresh value.
        self._arc_size_t: int = max(int(round(self.archive_factor * self._NP_current)), 0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _refresh_arc_size_t(self) -> None:
        """Sample the next generation's archive cap uniformly from ``[0, A_max]``.

        ``A_max = ceil(archive_factor · NP_current)`` — the same upper
        bound L-SHADE / jSO use as their *fixed* cap.  Drawing
        ``arc_size_t`` afresh each generation randomises the effective
        diversity of replaced parents the archive holds.  When
        ``archive_factor == 0`` the upper bound collapses to ``0`` and
        ``arc_size_t`` is deterministically ``0`` (no archive).
        """
        A_max = max(int(round(self.archive_factor * self._NP_current)), 0)
        self._arc_size_t = int(self._rng.integers(0, A_max + 1))

    def _rank_weights(self, sorted_live, target_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Build rank-based selection weights for ``r1`` candidates.

        Returns a pair ``(candidates, probs)`` where ``candidates`` is the
        array of live indices excluding ``target_idx`` (in best-first
        rank order) and ``probs`` is the normalised weight vector with
        ``probs[i] ∝ (NP − rank_of_candidates[i])``.
        """
        NP = len(sorted_live)
        # Pairs of (rank, idx) keeping best-first order; skip the target.
        candidate_ranks = [(rank, idx) for rank, idx in enumerate(sorted_live) if idx != target_idx]
        if not candidate_ranks:
            return np.empty(0, dtype=int), np.empty(0, dtype=float)
        candidates = np.fromiter((idx for _, idx in candidate_ranks), dtype=int, count=len(candidate_ranks))
        weights = np.fromiter(
            (float(NP - rank) for rank, _ in candidate_ranks),
            dtype=float,
            count=len(candidate_ranks),
        )
        total = float(weights.sum())
        if total <= 0.0:
            probs = np.full_like(weights, 1.0 / len(weights))
        else:
            probs = weights / total
        return candidates, probs

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def _trim_archive(self) -> None:
        """Trim the archive down to ``self._arc_size_t`` (drop random entries)."""
        cap = max(self._arc_size_t, 0)
        while len(self._archive) > cap:
            j = int(self._rng.integers(0, len(self._archive)))
            self._archive.pop(j)

    def _end_of_generation(self) -> None:
        """Run the base end-of-generation, then refresh the adaptive archive cap.

        The refresh happens *after* :meth:`LSHADE._end_of_generation` runs
        the memory update and LPSR shrink so the new cap scales with
        the *post-shrink* ``NP_current``.  This keeps the archive
        proportional to the current (possibly smaller) population — a
        cap drawn against the pre-shrink size could leave the archive
        larger than the live population it is supposed to diversify.

        After the refresh we also call :meth:`_trim_archive` once so
        the new cap is enforced immediately, before any new successful
        trial appends to the archive.
        """
        super()._end_of_generation()
        self._refresh_arc_size_t()
        self._trim_archive()

    def _generate_trial(self, target_idx: int) -> None:
        """Build and emit one ``current-to-pbest-w/1`` trial with rank-based ``r1``.

        Differs from :meth:`JSO._generate_trial` in exactly one place:
        the differential ``r1`` is drawn with rank-based selection
        pressure instead of a uniform sample.  Everything else — the
        jSO weighted mutation, linear ``p_best`` schedule, asymmetric
        F-cap, binomial crossover, and bounds reflection — is identical.
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

        # pbest: top p% of live population by fitness, with jSO's linear schedule.
        sorted_live = sorted(live, key=lambda i: self._fx_of(self._population[i]))  # type: ignore[arg-type]
        p_best = self._current_p_best()
        p_count = max(int(np.ceil(p_best * len(sorted_live))), 1)
        pbest_pool = sorted_live[:p_count]
        pbest_idx = int(self._rng.choice(np.asarray(pbest_pool)))
        pbest_slot = self._population[pbest_idx]
        if not isinstance(pbest_slot, Result):
            return
        x_pbest = np.asarray(pbest_slot.x, dtype=float)

        # r1 with rank-based selection pressure (NL-SHADE-RSP refinement).
        r1_candidates, probs = self._rank_weights(sorted_live, target_idx)
        if r1_candidates.size == 0:
            return
        r1 = int(self._rng.choice(r1_candidates, p=probs))
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

        # Mutation: jSO weighted current-to-pbest-w/1.
        v = x_target + F_w * (x_pbest - x_target) + F * (x_r1 - x_r2)
        v = self._reflect_bounds(v, x_target)

        # Binomial crossover with at least one component swapped.
        dim = self.problem.dim
        cross = self._rng.random(dim) < CR
        j_rand = int(self._rng.integers(0, dim))
        cross[j_rand] = True
        u = np.where(cross, v, x_target)

        self._emit_trial(u, target_idx, F, CR)

    # ------------------------------------------------------------------
    # Heuristic interface
    # ------------------------------------------------------------------

    def on_start(self) -> None:
        """Allocate state, plant the anchor bin, refresh the archive cap."""
        super().on_start()
        # ``NP_current`` was just restored to ``NP_init``; pick an initial
        # archive cap deterministically (not yet sampled — first refresh
        # happens at the first end-of-generation) so the early-fill phase
        # has a well-defined cap.
        self._arc_size_t = max(int(round(self.archive_factor * self._NP_current)), 0)

    def on_restart(self, center, reason: str = "") -> None:
        """Drop in-flight state, reseed, and reset the archive cap."""
        super().on_restart(center, reason)
        self._arc_size_t = max(int(round(self.archive_factor * self._NP_current)), 0)


# Re-export the jSO memory anchors so callers can introspect them
# without reaching into the parent module.  Kept as module-level
# attributes for symmetry with :mod:`panobbgo.heuristics.jso`.
__all__ = [
    "NLSHADERSP",
    "_ANCHOR_M_CR",
    "_ANCHOR_M_F",
    "_INIT_M_CR",
    "_INIT_M_F",
]
