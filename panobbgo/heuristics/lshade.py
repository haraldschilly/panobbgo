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
L-SHADE Heuristic
=================

Linear Population Size Reduction Success-History Adaptive Differential
Evolution (L-SHADE), Tanabe & Fukunaga (CEC 2014).

The basic Differential Evolution heuristic shipped in
:mod:`panobbgo.heuristics.differential_evolution` uses fixed
``F = 0.8`` / ``CR = 0.9`` and a fixed population size — a simple,
robust baseline.  L-SHADE adds two literature-best tricks that won
CEC-2014 and have been the high-water mark for single-population
black-box solvers ever since:

1. **Success-History parameter Adaptation (SHADE / Zhang-Sanderson 2009)**.
   Instead of fixed scalars, each trial draws its own ``(F_i, CR_i)``
   from per-bin Cauchy / Normal distributions.  After every "generation"
   of ``NP_current`` trials, the bins that produced *successful*
   replacements update their centres via the **weighted Lehmer mean**
   of the F/CR values that worked, weighted by the magnitude of the
   improvement they produced.  The memory bins rotate cyclically so
   recent successes dominate.
2. **Linear Population Size Reduction (LPSR)**.  The population shrinks
   linearly from ``NP_init`` (typically large, 18·d in the paper) down
   to ``NP_min = 4`` over the strategy's evaluation budget.  Larger
   populations explore broadly early; smaller ones exploit the leading
   basin late.

Mutation uses ``current-to-pbest/1`` (Zhang-Sanderson 2009)::

    v_i = x_i + F_i · (x_pbest − x_i) + F_i · (x_r1 − x_r2)

where ``x_pbest`` is drawn from the top ``p_best · |population|`` by
fitness, ``x_r1`` from the population, and ``x_r2`` from the union of
the population and an *external archive* of recently-replaced parents
(at most ``archive_factor · NP_current`` entries).  The crossover is
binomial with rate ``CR_i``.  Out-of-bounds components are repaired by
midpoint reflection (Tanabe-Fukunaga §III-A): ``v[j] = (lb[j] + x_i[j]) / 2``
when ``v[j] < lb[j]``, symmetric for ``ub[j]``.

Optionally pass ``p_best_end`` to enable the iLSHADE / jSO adaptive
``p_best`` schedule (Brest et al. 2016 / 2017).  The effective
greediness at evaluation count ``e`` (out of
``E = strategy.config.max_eval``) becomes
``p_eff(e) = p_best − (p_best − p_best_end) · min(e/E, 1)``,
shrinking the pool of ``pbest`` candidates as the population shrinks
under LPSR.  The canonical jSO setting is
``p_best = 0.25, p_best_end = 0.125``.  ``p_best_end=None`` (the
default) preserves the constant-``p_best`` L-SHADE behaviour.

Optionally pass ``F_schedule=True`` to enable the jSO asymmetric
F-cap (Brest et al. 2017).  The cap is keyed on progress
``e / E``: ``F ≤ 0.7`` while ``progress < 0.6``, ``F ≤ 0.8`` while
``progress < 0.9``, and unclamped in the final 10%.  This prevents
pathologically large jumps when the population is still big while
preserving full-range mutation late in the search.  ``F_schedule=None``
(default) keeps the unclamped Tanabe-Fukunaga behaviour.  jSO sets
``F_schedule=True`` by construction.

Asynchronous execution
----------------------

Like Panobbgo's other population heuristics, L-SHADE here runs
asynchronously inside the event loop:

1. ``on_start()`` emits ``NP_init`` random initial positions.
2. ``on_new_results()`` matches incoming results back to their slot via
   the ``who`` tag, fills the slot on the first arrival, and on later
   arrivals competes the trial against the slot's incumbent — the loser
   is discarded, the winner stays in the population, and the loser (if
   it was the parent) is pushed onto the archive.  After every
   ``NP_current`` evolutionary trials complete, the heuristic updates
   the memory bins with the successful F/CR triples and applies LPSR.
3. ``on_restart(center, reason)`` drops in-flight trials, clears the
   archive, resets the memory bins, and reseeds the population around
   the suggested center — matching the warm-restart behaviour of
   :class:`~panobbgo.heuristics.pso.PSO` and
   :class:`~panobbgo.heuristics.cma_es.CMAES`.

Notes on the async / sync gap
------------------------------

Synchronous L-SHADE applies parameter adaptation only at the *end* of
each generation, after every individual has been re-evaluated.  In the
async port we batch by *count*: every ``NP_current`` completed
evolutionary trials forms one "generation".  This keeps the same total
update cadence; the only difference is that within one async generation
a slot may have been touched twice (or zero times) rather than exactly
once.  In practice the SHADE memory adaptation is robust to this
because the weighted Lehmer mean is invariant under the order of its
contributing samples.

Constraint handling delegates to ``strategy.constraint_handler`` exactly
like :class:`~panobbgo.heuristics.differential_evolution.DifferentialEvolution`:
``is_better`` for trial-vs-target competition, ``get_penalty_value`` for
the scalar fitness used to rank pbest candidates and to weight memory
updates by improvement magnitude.

References
----------

* J. Zhang & A. Sanderson (2009). "JADE: Adaptive Differential Evolution
  with Optional External Archive." *IEEE Transactions on Evolutionary
  Computation*, 13(5):945-958.
* R. Tanabe & A. Fukunaga (2013). "Success-History Based Parameter
  Adaptation for Differential Evolution." *Proceedings of CEC 2013*.
* R. Tanabe & A. Fukunaga (2014). "Improving the Search Performance of
  SHADE Using Linear Population Size Reduction." *Proceedings of
  CEC 2014*.  Winner of the CEC-2014 single-objective competition.
* J. Brest, M. S. Maučec & B. Bošković (2016). "iL-SHADE: Improved
  L-SHADE Algorithm for Single Objective Real-Parameter Optimization."
  *Proceedings of CEC 2016*.  Introduces the linearly-decreasing
  ``p_best`` schedule.
* J. Brest, M. S. Maučec & B. Bošković (2017). "Single Objective
  Real-Parameter Optimization: Algorithm jSO." *Proceedings of CEC
  2017*.  Winner of the CEC-2017 single-objective competition.
"""

from __future__ import annotations

import uuid
from typing import Dict, List, Optional

import numpy as np

from panobbgo.core import Heuristic
from panobbgo.lib import Point, Result


# Default tuning constants — match the values from Tanabe & Fukunaga
# (2014, Algorithm 1).
_DEFAULT_NP_INIT: int = 30
_DEFAULT_NP_MIN: int = 4
_DEFAULT_H: int = 6
_DEFAULT_P_BEST: float = 0.11
_DEFAULT_ARCHIVE_FACTOR: float = 1.0
# Cauchy/Normal scale used by SHADE for F/CR sampling — fixed to 0.1 in
# all published variants of the algorithm.
_PARAM_SCALE: float = 0.1
# Maximum number of Cauchy redraws when sampling F.  A failed F sample
# is one with F <= 0; the redraw distribution is heavy-tailed so 100
# attempts give an effectively zero failure probability.
_F_MAX_REDRAWS: int = 100
# Sentinel emitted into M_CR when an entire generation produced only
# CR = 0 successes; subsequent draws then deterministically return 0
# (per Tanabe-Fukunaga §III-B).
_CR_TERMINAL: float = -1.0

# Asymmetric F-cap schedule (Brest et al. 2017, jSO).  Three phases keyed
# on ``progress = len(results) / max_eval``:
#
#   progress ∈ [0, _F_SCHEDULE_PHASE1_BOUND)   → F clamped at _F_SCHEDULE_PHASE1_CAP
#   progress ∈ [_F_SCHEDULE_PHASE1_BOUND,
#               _F_SCHEDULE_PHASE2_BOUND)      → F clamped at _F_SCHEDULE_PHASE2_CAP
#   progress ∈ [_F_SCHEDULE_PHASE2_BOUND, 1]   → F unclamped (just F ≤ 1)
#
# The 60% / 90% breakpoints and 0.7 / 0.8 caps match the canonical jSO
# spec (Brest, Maučec & Bošković 2017, §III-D).  When ``F_schedule`` is
# off the cap is bypassed and the heuristic reproduces the byte-identical
# L-SHADE behaviour shipped 2026-05-10.
_F_SCHEDULE_PHASE1_BOUND: float = 0.6
_F_SCHEDULE_PHASE2_BOUND: float = 0.9
_F_SCHEDULE_PHASE1_CAP: float = 0.7
_F_SCHEDULE_PHASE2_CAP: float = 0.8


class _Dropped:
    """Sentinel used to mark population slots removed by LPSR."""


_DROPPED = _Dropped()


class LSHADE(Heuristic):
    """L-SHADE: linear-population-reduction SHADE adaptive DE.

    Args:
        strategy: The owning :class:`~panobbgo.core.StrategyBase`.
        NP_init: Initial population size.  Default ``30`` — the standard
            literature setting.  The CEC-2014 paper used ``18 · d``,
            which is a heavier swarm than Panobbgo's typical budget can
            support; ``30`` is a good middle ground for the 2-10 D
            problems in our benchmark battery.
        NP_min: Minimum population size after LPSR shrinking.  Default
            ``4`` — required by ``current-to-pbest/1`` (mutation needs
            at least four distinct individuals).  Must satisfy
            ``NP_min <= NP_init``.
        H: History memory size — number of (M_F, M_CR) bins.  Default
            ``6`` — the value used by both SHADE and L-SHADE.  Larger
            values smooth memory updates but slow adaptation; smaller
            values track recent successes more tightly at the cost of
            more variance.
        p_best: Greediness factor for ``current-to-pbest/1``.  Each
            trial picks its ``pbest`` uniformly from the top
            ``ceil(p_best · |population|)`` by fitness.  Default
            ``0.11`` per Tanabe-Fukunaga §III-A.  Must lie in ``(0, 1]``.
            When ``p_best_end`` is set, this is the *initial* value of
            a linearly-annealed schedule (iLSHADE / jSO).
        p_best_end: Optional terminal greediness for the iLSHADE
            (Brest et al. 2016) / jSO (Brest et al. 2017) adaptive
            schedule.  When set, the effective ``p_best`` at evaluation
            count ``e`` (out of ``E = strategy.config.max_eval``) is
            ``p_eff(e) = p_best − (p_best − p_best_end) · min(e/E, 1)``.
            The canonical jSO setting is
            ``p_best = 0.25, p_best_end = 0.125`` — greediness halves
            as the population shrinks under LPSR so the late-search
            mutation pulls toward a narrower, more tightly-chosen
            ``pbest`` slice.  Must lie in ``(0, 1]`` when set;
            ``None`` (the default) keeps ``p_best`` constant for
            byte-identical L-SHADE behaviour.  Falls back to constant
            ``p_best`` whenever the strategy budget is unknown.
        archive_factor: Multiplier for the external archive size; the
            archive is capped at ``ceil(archive_factor · NP_current)``.
            Default ``1.0``.  Setting it to ``0`` disables the archive
            (``r2`` is then drawn only from the live population).
        F_schedule: Optional opt-in for the jSO asymmetric F-cap (Brest
            et al. 2017).  When ``True``, sampled ``F`` is clamped to
            ``0.7`` while ``progress < 0.6``, to ``0.8`` while
            ``progress < 0.9``, and left unclamped in the final 10% of
            the budget.  ``None`` (default) keeps the unclamped
            Tanabe-Fukunaga behaviour.  jSO opts in by construction.
            Falls back to the unclamped behaviour when the strategy
            budget is unknown.
        seed: Optional seed for the per-instance RNG.  ``None`` (default)
            seeds from ``np.random.default_rng()``.
        name: Override the heuristic's display name.

    Notes:
        - All numeric arguments are validated; bad values raise
          :class:`ValueError`.
        - Like every Panobbgo heuristic, all state is per-instance.
          Multiple ``LSHADE`` heuristics in one strategy run
          independently and never share memory bins.
        - LPSR scales the population by progress
          ``len(strategy.results) / strategy.config.max_eval``.  When
          the budget is unknown (no ``max_eval``, zero, or non-numeric)
          the heuristic falls back to a *constant* population at
          ``NP_init``.
    """

    def __init__(
        self,
        strategy,
        NP_init: int = _DEFAULT_NP_INIT,
        NP_min: int = _DEFAULT_NP_MIN,
        H: int = _DEFAULT_H,
        p_best: float = _DEFAULT_P_BEST,
        p_best_end: Optional[float] = None,
        archive_factor: float = _DEFAULT_ARCHIVE_FACTOR,
        F_schedule: Optional[bool] = None,
        seed: Optional[int] = None,
        name: Optional[str] = None,
    ) -> None:
        if not isinstance(NP_init, int):
            raise ValueError(f"LSHADE: NP_init must be an integer, got {NP_init!r}")
        if NP_init < 4:
            raise ValueError(f"LSHADE: NP_init must be >= 4, got {NP_init}")
        if not isinstance(NP_min, int):
            raise ValueError(f"LSHADE: NP_min must be an integer, got {NP_min!r}")
        if NP_min < 4:
            raise ValueError(f"LSHADE: NP_min must be >= 4, got {NP_min}")
        if NP_min > NP_init:
            raise ValueError(f"LSHADE: NP_min ({NP_min}) must be <= NP_init ({NP_init})")
        if not isinstance(H, int):
            raise ValueError(f"LSHADE: H must be an integer, got {H!r}")
        if H < 1:
            raise ValueError(f"LSHADE: H must be >= 1, got {H}")
        if not np.isfinite(p_best) or not (0.0 < p_best <= 1.0):
            raise ValueError(f"LSHADE: p_best must be in (0, 1], got {p_best}")
        if p_best_end is not None and (not np.isfinite(p_best_end) or not (0.0 < p_best_end <= 1.0)):
            raise ValueError(f"LSHADE: p_best_end must be in (0, 1] when set, got {p_best_end}")
        if not np.isfinite(archive_factor) or archive_factor < 0.0:
            raise ValueError(f"LSHADE: archive_factor must be a non-negative finite float, got {archive_factor}")
        if F_schedule is not None and not isinstance(F_schedule, bool):
            raise ValueError(f"LSHADE: F_schedule must be a bool or None, got {F_schedule!r}")

        super().__init__(strategy, name=name or "LSHADE")
        self.NP_init: int = NP_init
        self.NP_min: int = NP_min
        self.H: int = H
        self.p_best: float = float(p_best)
        self.p_best_end: Optional[float] = None if p_best_end is None else float(p_best_end)
        self.archive_factor: float = float(archive_factor)
        self.F_schedule: Optional[bool] = F_schedule
        self._rng: np.random.Generator = np.random.default_rng(seed)

        # Success-history memory.  Initial value 0.5 per the SHADE paper.
        self._M_F: np.ndarray = np.full(H, 0.5, dtype=float)
        self._M_CR: np.ndarray = np.full(H, 0.5, dtype=float)
        self._mem_ptr: int = 0

        # Population bookkeeping.  ``_population[i]`` is one of:
        #   * ``None``                — slot still pending its initial fill
        #   * a :class:`Result`       — currently-occupied slot
        #   * :data:`_DROPPED`        — slot removed by LPSR; future returns
        #                               for this slot are silently dropped.
        self._population: List[Optional[Result]] = []
        self._NP_current: int = NP_init

        # External archive of replaced parents.  Stored as raw position
        # vectors so we don't carry full :class:`Result` overhead.
        self._archive: List[np.ndarray] = []

        # Pending trials: req_id -> (slot_idx, F, CR).  Initial random
        # trials use F=NaN, CR=NaN so they don't contribute to the
        # success memory.
        self._pending: Dict[str, "_TrialMeta"] = {}

        # Current-generation success buffer.  At end of generation we
        # update memory and clear.
        self._gen_completed: int = 0  # evolutionary trials finished this gen
        self._success_F: List[float] = []
        self._success_CR: List[float] = []
        self._success_delta: List[float] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _max_eval(self) -> Optional[float]:
        """Return the strategy's evaluation budget, or ``None`` if unknown."""
        try:
            v = float(self.strategy.config.max_eval)  # type: ignore[union-attr]
        except Exception:
            return None
        if not np.isfinite(v) or v <= 0.0:
            return None
        return v

    def _progress(self) -> Optional[float]:
        """Return ``len(strategy.results) / max_eval`` clipped to ``[0, 1]``.

        Returns ``None`` when the budget is unknown so callers can
        distinguish "early phase" (progress = 0.0) from "no budget at
        all" and pick a safe fallback for each schedule.
        """
        max_eval = self._max_eval()
        if max_eval is None:
            return None
        try:
            current = float(len(self.strategy.results))
        except Exception:
            return None
        return float(np.clip(current / max_eval, 0.0, 1.0))

    def _current_p_best(self) -> float:
        """Return the ``p_best`` value to use for the next trial.

        When ``p_best_end`` is ``None`` this is the constant ``self.p_best``,
        i.e. the byte-identical L-SHADE behaviour shipped 2026-05-10.
        Otherwise it is a linearly-annealed schedule paced by
        ``len(strategy.results) / strategy.config.max_eval`` — the
        iLSHADE (Brest et al. 2016) / jSO (Brest et al. 2017) move that
        shrinks greediness as the population shrinks under LPSR.  When
        the budget is unknown (no ``max_eval``, zero, or non-numeric)
        the heuristic falls back to constant ``self.p_best`` rather than
        guessing a horizon.
        """
        if self.p_best_end is None:
            return self.p_best
        progress = self._progress()
        if progress is None:
            return self.p_best
        return self.p_best - (self.p_best - self.p_best_end) * progress

    def _apply_F_cap(self, F: float) -> float:
        """Apply the jSO asymmetric F-cap (Brest et al. 2017) when opted in.

        ``F_schedule=None`` (default) leaves ``F`` unchanged — the
        byte-identical L-SHADE behaviour.  ``F_schedule=True`` applies
        the three-phase cap keyed on
        ``progress = len(strategy.results) / strategy.config.max_eval``:

        * ``progress < 0.6``: ``F`` clamped at ``0.7``.
        * ``0.6 <= progress < 0.9``: ``F`` clamped at ``0.8``.
        * ``progress >= 0.9``: ``F`` left unclamped (≤ 1.0 by sampler).

        When the strategy budget is unknown the cap is bypassed — the
        same safe fallback used by :meth:`_current_p_best` and
        :meth:`_apply_lpsr`.
        """
        if not self.F_schedule:
            return F
        progress = self._progress()
        if progress is None:
            return F
        if progress < _F_SCHEDULE_PHASE1_BOUND:
            return min(F, _F_SCHEDULE_PHASE1_CAP)
        if progress < _F_SCHEDULE_PHASE2_BOUND:
            return min(F, _F_SCHEDULE_PHASE2_CAP)
        return F

    def _emit_trial(self, x: np.ndarray, slot_idx: int, F: float, CR: float) -> bool:
        """Project, queue, and book-keep one candidate point."""
        if self._stopped:
            return False
        try:
            x_proj = self.problem.project(x)
        except Exception as exc:
            self.logger.debug(f"LSHADE: projection failed: {exc}")
            return False

        req_id = uuid.uuid4().hex
        who = f"{self.name}:{req_id}"
        try:
            self._output.put_nowait(Point(x_proj, who))
        except Exception as exc:  # queue full or shutdown
            self.logger.debug(f"LSHADE: emit failed: {exc}")
            return False
        self._pending[req_id] = _TrialMeta(slot_idx=slot_idx, F=F, CR=CR)
        return True

    def _live_indices(self) -> List[int]:
        """Indices of currently-filled, non-dropped population slots."""
        out: List[int] = []
        for i, slot in enumerate(self._population):
            if isinstance(slot, Result):
                out.append(i)
        return out

    def _fx_of(self, r: Result) -> float:
        """Scalar fitness for ranking — falls back to ``r.fx`` if no handler."""
        handler = getattr(self.strategy, "constraint_handler", None)
        if handler is None:
            return float(r.fx) if r.fx is not None else float("inf")
        return handler.get_penalty_value(r)

    def _archive_cap(self) -> int:
        """Maximum number of replaced parents the external archive retains.

        Default is the fixed ``archive_factor · NP_current`` cap from
        Tanabe-Fukunaga (2014).  Subclasses (e.g.
        :class:`~panobbgo.heuristics.nl_shade_rsp.NLSHADE_RSP`) override
        this to randomise the cap per generation.
        """
        return max(int(round(self.archive_factor * self._NP_current)), 0)

    def _trim_archive(self) -> None:
        """Cap the archive at :meth:`_archive_cap` (drop random entries)."""
        cap = self._archive_cap()
        while len(self._archive) > cap:
            j = int(self._rng.integers(0, len(self._archive)))
            self._archive.pop(j)

    def _select_r1(self, live: List[int], target_idx: int) -> Optional[int]:
        """Pick the index ``r1`` for the differential ``F · (x_r1 − x_r2)`` term.

        Default: uniform over live slots excluding the target — the
        Tanabe-Fukunaga / jSO behaviour.  Subclasses (e.g.
        :class:`~panobbgo.heuristics.nl_shade_rsp.NLSHADE_RSP`) override
        this to bias the choice toward higher-ranked individuals
        (rank-based selective pressure).  Returns ``None`` when no
        candidate is available so the caller can abort the trial.
        """
        r1_pool = [i for i in live if i != target_idx]
        if not r1_pool:
            return None
        return int(self._rng.choice(np.asarray(r1_pool)))

    def _sample_F_CR(self) -> tuple[float, float]:
        """Draw one ``(F, CR)`` pair from a random history bin."""
        r = int(self._rng.integers(0, self.H))
        # CR sampling: Normal(M_CR[r], 0.1), clamped to [0, 1].  The
        # M_CR = -1 sentinel collapses to deterministic CR = 0.
        m_cr = float(self._M_CR[r])
        if m_cr < 0:
            CR = 0.0
        else:
            CR = float(self._rng.normal(m_cr, _PARAM_SCALE))
            CR = float(np.clip(CR, 0.0, 1.0))

        # F sampling: Cauchy(M_F[r], 0.1), regenerate while F <= 0,
        # clip at 1.  Bounded redraws to prevent worst-case loops.
        m_f = float(self._M_F[r])
        F = 0.5
        for _ in range(_F_MAX_REDRAWS):
            f = m_f + _PARAM_SCALE * float(self._rng.standard_cauchy())
            if f > 0.0:
                F = float(min(f, 1.0))
                break
        return self._apply_F_cap(F), CR

    def _reflect_bounds(self, v: np.ndarray, x_target: np.ndarray) -> np.ndarray:
        """Midpoint reflection bounds repair (Tanabe-Fukunaga §III-A)."""
        lb = self.problem.box[:, 0]
        ub = self.problem.box[:, 1]
        out = v.copy()
        below = out < lb
        if np.any(below):
            out[below] = (lb[below] + x_target[below]) / 2.0
        above = out > ub
        if np.any(above):
            out[above] = (ub[above] + x_target[above]) / 2.0
        return out

    def _generate_trial(self, target_idx: int) -> None:
        """Build and emit one ``current-to-pbest/1`` trial vector."""
        live = self._live_indices()
        if len(live) < 4 or target_idx not in live:
            return
        slot = self._population[target_idx]
        if not isinstance(slot, Result):
            return

        F, CR = self._sample_F_CR()
        x_target = np.asarray(slot.x, dtype=float)

        # pbest: top p% of live population by fitness (ascending — best first).
        # ``_current_p_best`` honours the optional iLSHADE / jSO linearly-
        # decreasing schedule when ``p_best_end`` is set; otherwise it is
        # the constant ``self.p_best``.
        sorted_live = sorted(live, key=lambda i: self._fx_of(self._population[i]))  # type: ignore[arg-type]
        p_eff = self._current_p_best()
        p_count = max(int(np.ceil(p_eff * len(sorted_live))), 1)
        pbest_pool = sorted_live[:p_count]
        pbest_idx = int(self._rng.choice(np.asarray(pbest_pool)))
        pbest_slot = self._population[pbest_idx]
        if not isinstance(pbest_slot, Result):
            return
        x_pbest = np.asarray(pbest_slot.x, dtype=float)

        # r1 from live population, distinct from target.
        r1 = self._select_r1(live, target_idx)
        if r1 is None:
            return
        r1_slot = self._population[r1]
        if not isinstance(r1_slot, Result):
            return
        x_r1 = np.asarray(r1_slot.x, dtype=float)

        # r2 from (live ∪ archive) \ {target, r1}.
        union: List[np.ndarray] = []
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

        # Mutation: current-to-pbest/1.
        v = x_target + F * (x_pbest - x_target) + F * (x_r1 - x_r2)
        v = self._reflect_bounds(v, x_target)

        # Binomial crossover with at least one component swapped.
        dim = self.problem.dim
        cross = self._rng.random(dim) < CR
        j_rand = int(self._rng.integers(0, dim))
        cross[j_rand] = True
        u = np.where(cross, v, x_target)

        self._emit_trial(u, target_idx, F, CR)

    def _update_memory(self) -> None:
        """Apply the weighted Lehmer-mean memory update for one generation."""
        if not self._success_F:
            return  # no successes — leave memory untouched
        F_arr = np.asarray(self._success_F, dtype=float)
        CR_arr = np.asarray(self._success_CR, dtype=float)
        delta_arr = np.asarray(self._success_delta, dtype=float)
        total = float(delta_arr.sum())
        if total > 0.0:
            w = delta_arr / total
        else:
            w = np.full_like(delta_arr, 1.0 / len(delta_arr))

        # F: weighted Lehmer mean; F is always > 0 by construction.
        F_num = float(np.sum(w * F_arr * F_arr))
        F_den = float(np.sum(w * F_arr))
        if F_den > 0.0:
            self._M_F[self._mem_ptr] = float(np.clip(F_num / F_den, 0.0, 1.0))

        # CR: if all successes had CR = 0 OR the bin is already terminal,
        # plant the terminal sentinel (-1).  Otherwise weighted Lehmer mean.
        cr_max = float(CR_arr.max())
        if cr_max <= 0.0 or self._M_CR[self._mem_ptr] < 0.0:
            self._M_CR[self._mem_ptr] = _CR_TERMINAL
        else:
            CR_num = float(np.sum(w * CR_arr * CR_arr))
            CR_den = float(np.sum(w * CR_arr))
            if CR_den > 0.0:
                self._M_CR[self._mem_ptr] = float(np.clip(CR_num / CR_den, 0.0, 1.0))

        self._mem_ptr = (self._mem_ptr + 1) % self.H

    def _lpsr_target(self, progress: float) -> int:
        """Target population size at ``progress`` (Tanabe-Fukunaga 2014, linear).

        Linear interpolation from ``NP_init`` (progress 0) down to
        ``NP_min`` (progress 1).  Subclasses (e.g.
        :class:`~panobbgo.heuristics.nl_shade_rsp.NLSHADE_RSP`) override
        this with a non-linear schedule.
        """
        return int(round(self.NP_init - (self.NP_init - self.NP_min) * progress))

    def _apply_lpsr(self) -> None:
        """Shrink the population to ``NP_target`` based on budget progress."""
        progress = self._progress()
        if progress is None:
            return
        target = self._lpsr_target(progress)
        target = max(target, self.NP_min)
        target = min(target, self._NP_current)
        if target >= self._NP_current:
            return

        # Drop the worst (NP_current - target) live slots by fitness.
        live = self._live_indices()
        if len(live) <= target:
            self._NP_current = target
            self._trim_archive()
            return
        sorted_live = sorted(live, key=lambda i: self._fx_of(self._population[i]))  # type: ignore[arg-type]
        n_drop = self._NP_current - target
        for j in sorted_live[-n_drop:]:  # worst n_drop
            self._population[j] = _DROPPED  # type: ignore[assignment]
        self._NP_current = target
        self._trim_archive()

    def _end_of_generation(self) -> None:
        """Run memory + LPSR updates and reset the success buffer."""
        self._update_memory()
        self._apply_lpsr()
        self._gen_completed = 0
        self._success_F.clear()
        self._success_CR.clear()
        self._success_delta.clear()

    def _wake_idle_slots(self) -> None:
        """Make sure every live slot has at most one pending trial.

        Slots that finished their initial random fill before the
        population reached the four-individual threshold for
        ``current-to-pbest/1`` end up *idle* — filled, but with no
        in-flight trial.  Once the threshold is met we kick them off
        with their first evolutionary trial so the swarm gets back to
        full async throughput.
        """
        live = set(self._live_indices())
        if len(live) < 4:
            return
        active = {meta.slot_idx for meta in self._pending.values()}
        for slot_idx in sorted(live - active):
            self._generate_trial(slot_idx)

    # ------------------------------------------------------------------
    # Heuristic interface
    # ------------------------------------------------------------------

    def on_start(self) -> None:
        """Allocate state and emit ``NP_init`` random initial trials."""
        self._population = [None] * self.NP_init
        self._NP_current = self.NP_init
        self._archive.clear()
        self._pending.clear()
        self._M_F[:] = 0.5
        self._M_CR[:] = 0.5
        self._mem_ptr = 0
        self._gen_completed = 0
        self._success_F.clear()
        self._success_CR.clear()
        self._success_delta.clear()

        for i in range(self.NP_init):
            x = self.problem.random_point()
            self._emit_trial(x, i, F=float("nan"), CR=float("nan"))

    def on_new_results(self, results) -> None:
        """Process incoming evaluations and dispatch follow-up trials."""
        if not self._population:
            return  # not started yet

        prefix = f"{self.name}:"
        handler = self.strategy.constraint_handler
        for r in results:
            who: str = getattr(r, "who", "") or ""
            if not who.startswith(prefix):
                continue
            req_id = who[len(prefix) :]
            meta = self._pending.pop(req_id, None)
            if meta is None:
                continue  # stale or unknown trial id
            slot_idx = meta.slot_idx

            # Slot dropped by LPSR after we issued this trial — discard.
            if slot_idx >= len(self._population):
                continue
            slot = self._population[slot_idx]
            if isinstance(slot, _Dropped):
                continue

            if slot is None:
                # Initial random fill — just store.  No success counted.
                self._population[slot_idx] = r
            else:
                # Competitive trial.  Compete; loser may go to archive.
                target = slot
                if handler.is_better(target, r):
                    delta = abs(self._fx_of(target) - self._fx_of(r))
                    self._archive.append(np.asarray(target.x, dtype=float))
                    self._trim_archive()
                    self._population[slot_idx] = r
                    if not np.isnan(meta.F) and not np.isnan(meta.CR):
                        self._success_F.append(meta.F)
                        self._success_CR.append(meta.CR)
                        # Floor delta so an unweighted-but-real success
                        # still influences the Lehmer mean.
                        self._success_delta.append(max(float(delta), 1e-30))
                self._gen_completed += 1

                if self._gen_completed >= max(self._NP_current, 1):
                    self._end_of_generation()

            # Emit a follow-up trial for this slot if it survived.
            if slot_idx < len(self._population) and isinstance(self._population[slot_idx], Result):
                self._generate_trial(slot_idx)

            # Wake up any idle live slots (filled, but no pending trial).
            self._wake_idle_slots()

    def on_restart(self, center, reason: str = "") -> None:
        """Drop in-flight state and reseed the population around ``center``.

        Mirrors the warm-restart pattern used by
        :class:`~panobbgo.heuristics.pso.PSO` and the IPOP/BIPOP CMA-ES
        variants: archive cleared, memory bins reset to 0.5, slots
        re-randomised in a small ball around ``center`` (or random in
        the box if ``center`` is ``None``), and a fresh round of
        initial-random trials emitted.
        """
        if self._stopped:
            return
        self.clear_output()
        self._pending.clear()
        if not self._population:
            return  # not started yet — nothing to reset

        self._archive.clear()
        self._M_F[:] = 0.5
        self._M_CR[:] = 0.5
        self._mem_ptr = 0
        self._gen_completed = 0
        self._success_F.clear()
        self._success_CR.clear()
        self._success_delta.clear()
        # Restore full-size population; LPSR will shrink it again from scratch.
        self._population = [None] * self.NP_init
        self._NP_current = self.NP_init

        ranges = self.problem.box[:, 1] - self.problem.box[:, 0]
        ball = 0.1 * ranges  # small reseed ball; conservative
        if center is None:
            base = None
        else:
            base = np.asarray(center, dtype=float)
        for i in range(self.NP_init):
            if base is None:
                x = self.problem.random_point()
            else:
                offset = self._rng.uniform(-ball, ball)
                x = self.problem.project(base + offset)
            self._emit_trial(x, i, F=float("nan"), CR=float("nan"))


class _TrialMeta:
    """Per-trial bookkeeping used to identify which slot/F/CR a result came from."""

    __slots__ = ("slot_idx", "F", "CR")

    def __init__(self, slot_idx: int, F: float, CR: float) -> None:
        self.slot_idx = slot_idx
        self.F = F
        self.CR = CR
