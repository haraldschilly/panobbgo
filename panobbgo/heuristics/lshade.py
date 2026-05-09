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
L-SHADE (Linear Population Size Reduction Success-History Adaptive DE)
======================================================================

Asynchronous adaptation of L-SHADE (Tanabe & Fukunaga, 2014), the
CEC-2014 winner and one of the strongest single-population black-box
optimisers in the literature.  L-SHADE generalises basic Differential
Evolution along three axes that consistently win against fixed-parameter
DE in benchmark studies:

1. **DE/current-to-pbest/1 mutation with archive** (JADE; Zhang &
   Sanderson, 2009).  The mutant for individual ``x_i`` is::

       v_i = x_i + F · (x_pbest − x_i) + F · (x_r1 − x_r2)

   where ``x_pbest`` is sampled uniformly from the top
   ``p_best_rate · NP`` individuals, ``x_r1`` is sampled from the
   current population, and ``x_r2`` is sampled from the union of the
   population and an *external archive* of recently-replaced parents.
   The archive injects diversity; the pbest term injects greediness.
2. **Per-individual F and CR sampled from a success-history memory**.
   ``M_F`` and ``M_CR`` hold ``H`` recently-successful parameter
   values.  Each trial samples its own ``(F, CR)`` from a randomly
   selected memory slot — ``F ∼ Cauchy(M_F[r], 0.1)`` clipped to
   ``(0, 1]``, ``CR ∼ Normal(M_CR[r], 0.1)`` clipped to ``[0, 1]``.
   After every "generation" (NP successful or attempted trials) the
   memory is updated with the *weighted Lehmer mean* of the successful
   ``F`` and ``CR`` values, weighted by the fitness improvement of
   each successful trial.  This makes the parameters self-adapt to
   the landscape.
3. **Linear Population Size Reduction (LPSR)**.  Population size
   shrinks linearly from ``NP_init`` to ``NP_min`` as evaluations
   accumulate::

       NP(t) = round(NP_init − (NP_init − NP_min) · t / max_eval)

   The largest, weakest individuals are dropped first — concentrating
   evaluations on the more promising members late in the budget.

Asynchronous adaptation
~~~~~~~~~~~~~~~~~~~~~~~

The reference L-SHADE is a generation-based loop: produce ``NP``
trials, evaluate all of them, then update memories and shrink.
Panobbgo's heuristics emit and consume points asynchronously through
the event loop, so we adapt the algorithm in three small ways:

* Each trial carries its own pre-sampled ``(F, CR)`` tagged into the
  trial id.  When the result arrives, the heuristic knows exactly
  which parameter pair produced it.
* "Generations" become *batches of NP trial outcomes*.  Successes
  inside the batch contribute ``(F, CR, Δfx)`` records; at batch
  completion the memory entries are updated and the success buffer
  is cleared.
* LPSR fires on the same batch boundary by checking the current
  evaluation budget consumed (``len(strategy.results)``) against
  ``strategy.config.max_eval``.  When ``max_eval`` is unknown the
  population size stays constant at ``NP_init`` — the original DE
  behaviour, byte-identical to a non-LPSR run.

Why ship this alongside the existing :class:`DifferentialEvolution`?

Panobbgo already has a basic ``DE/rand/1/bin`` heuristic, but L-SHADE
consistently outperforms it on multimodal benchmarks where
fixed-parameter DE either over-explores (large ``F``) or
over-exploits (small ``F``).  The two heuristics are complementary —
``DE/rand/1`` excels with infinite budgets where the population can
cover the space, L-SHADE wins under the kind of budget-constrained,
expensive-evaluation regime Panobbgo targets.  Adding L-SHADE as a
separate heuristic (rather than replacing the existing DE) lets the
self-improvement loop and the structural mutation catalog choose
whichever variant wins on a given problem family.

References
----------

* R. Tanabe & A. Fukunaga (2013). "Success-History Based Parameter
  Adaptation for Differential Evolution."  *IEEE Congress on
  Evolutionary Computation*, 71–78.  (SHADE — predecessor.)
* R. Tanabe & A. Fukunaga (2014). "Improving the Search Performance
  of SHADE Using Linear Population Size Reduction."  *IEEE Congress
  on Evolutionary Computation*, 1658–1665.  (L-SHADE — original.)
* J. Zhang & A. C. Sanderson (2009). "JADE: Adaptive Differential
  Evolution With Optional External Archive."  *IEEE Transactions on
  Evolutionary Computation*, 13(5):945–958.  (current-to-pbest +
  archive design.)
"""

from __future__ import annotations

import uuid
from typing import Dict, List, Optional, Tuple

import numpy as np

from panobbgo.core import Heuristic
from panobbgo.lib import Point, Result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _weighted_lehmer_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """Return the weighted Lehmer mean ``Σwv² / Σwv`` (Tanabe-Fukunaga 2014).

    Falls back to the unweighted Lehmer mean when all weights are zero
    (e.g. every "successful" trial happened to land at the parent's
    fitness, which is rare but possible under bit-for-bit tied
    objectives).  Returns ``float('nan')`` for empty inputs so callers
    can decide to keep the previous memory value.
    """
    if values.size == 0:
        return float("nan")
    w_sum = float(np.sum(weights))
    if w_sum <= 0.0:
        # Degenerate: all weights zero.  Use uniform weights instead.
        weights = np.ones_like(values)
        w_sum = float(np.sum(weights))
    num = float(np.sum(weights * values * values))
    den = float(np.sum(weights * values))
    if den <= 0.0:
        # All values are zero — nothing to learn from.
        return float("nan")
    return num / den


def _sample_cauchy_F(rng: np.random.Generator, mu: float, sigma: float = 0.1) -> float:
    """Sample ``F ∼ Cauchy(mu, sigma)`` clipped per Tanabe-Fukunaga 2014.

    ``F`` is **resampled** as long as it comes out non-positive (the
    Cauchy distribution has a non-trivial mass below zero) and is
    **clipped** to ``1.0`` when it exceeds it.  At most ``32`` resample
    attempts are made before falling back to ``mu`` clipped to the
    valid range — an extremely unlikely path that exists only to
    keep the heuristic robust against pathological RNG draws.
    """
    for _ in range(32):
        f = mu + sigma * rng.standard_cauchy()
        if f >= 1.0:
            return 1.0
        if f > 0.0:
            return float(f)
    # Pathological: fall back to the centre, clipped to the valid range.
    return float(min(max(mu, 1e-6), 1.0))


def _sample_normal_CR(rng: np.random.Generator, mu: float, sigma: float = 0.1) -> float:
    """Sample ``CR ∼ Normal(mu, sigma)`` clipped to ``[0, 1]``."""
    cr = float(mu + sigma * rng.standard_normal())
    return float(min(max(cr, 0.0), 1.0))


# ---------------------------------------------------------------------------
# L-SHADE heuristic
# ---------------------------------------------------------------------------


class LSHADE(Heuristic):
    """Asynchronous L-SHADE (Tanabe-Fukunaga 2014) heuristic.

    Args:
        strategy: The owning :class:`~panobbgo.core.StrategyBase`.
        NP_init: Initial population size.  Default ``18 · dim`` is the
            Tanabe-Fukunaga recommendation but this is *clipped* at
            instantiation against the configured ``cap`` so a tiny
            problem does not explode the swarm.  Set to a positive
            integer to override the dim-scaled default.  ``None`` means
            "use ``18 · dim``".
        NP_min: Minimum population size that LPSR shrinks toward.
            Default ``4`` is the smallest size that supports
            ``DE/current-to-pbest/1`` with a non-empty archive
            (need ``r1, r2, pbest`` distinct from ``i``).
        H: Memory size for ``M_F`` and ``M_CR``.  Default ``6`` from
            the L-SHADE paper.
        p_best_rate: Top fraction of the population eligible for the
            ``pbest`` term in the mutation.  Default ``0.11`` from the
            L-SHADE paper.  Bounded to ``(0, 1]``; the implementation
            also enforces a floor of two individuals so the pbest pool
            is never empty in tiny populations.
        archive_factor: Maximum archive size as a multiple of the
            current population size.  Default ``2.6`` from the original
            L-SHADE paper; values closer to ``1.0`` are common in
            simpler implementations and reduce memory.  When the
            archive is full, the oldest entry is replaced first.
        seed: Optional seed for the per-instance RNG.  ``None`` (default)
            seeds from ``np.random.default_rng()``.
        name: Override the heuristic's display name.

    Notes:
        * Constructor validates all numeric arguments and raises
          :class:`ValueError` on bad inputs.
        * The heuristic respects constraints via
          ``self.strategy.constraint_handler.is_better`` and
          ``get_penalty_value`` exactly like
          :class:`~panobbgo.heuristics.differential_evolution.DifferentialEvolution`.
        * When ``strategy.config.max_eval`` is unknown, LPSR is
          disabled and the population stays at ``NP_init`` (matching the
          basic DE behaviour).  This keeps the heuristic robust on
          ad-hoc strategies and tests that do not set a budget.
        * State is fully owned by the instance — multiple L-SHADE
          heuristics in one strategy do not interfere.
    """

    def __init__(
        self,
        strategy,
        NP_init: Optional[int] = None,
        NP_min: int = 4,
        H: int = 6,
        p_best_rate: float = 0.11,
        archive_factor: float = 2.6,
        seed: Optional[int] = None,
        name: Optional[str] = None,
    ) -> None:
        if NP_init is not None:
            if not isinstance(NP_init, int):
                raise ValueError(f"LSHADE: NP_init must be an integer or None, got {NP_init!r}")
            if NP_init < 4:
                raise ValueError(f"LSHADE: NP_init must be >= 4, got {NP_init}")
        if not isinstance(NP_min, int):
            raise ValueError(f"LSHADE: NP_min must be an integer, got {NP_min!r}")
        if NP_min < 4:
            raise ValueError(f"LSHADE: NP_min must be >= 4, got {NP_min}")
        if NP_init is not None and NP_init < NP_min:
            raise ValueError(f"LSHADE: NP_init ({NP_init}) must be >= NP_min ({NP_min})")
        if not isinstance(H, int) or H < 1:
            raise ValueError(f"LSHADE: H must be a positive integer, got {H!r}")
        if not np.isfinite(p_best_rate) or not (0.0 < p_best_rate <= 1.0):
            raise ValueError(f"LSHADE: p_best_rate must be in (0, 1], got {p_best_rate}")
        if not np.isfinite(archive_factor) or archive_factor < 0.0:
            raise ValueError(f"LSHADE: archive_factor must be a non-negative finite float, got {archive_factor}")

        super().__init__(strategy, name=name or "LSHADE")
        self.NP_init_user: Optional[int] = NP_init
        self.NP_min: int = int(NP_min)
        self.H: int = int(H)
        self.p_best_rate: float = float(p_best_rate)
        self.archive_factor: float = float(archive_factor)
        self._rng: np.random.Generator = np.random.default_rng(seed)

        # Allocated on on_start() once we know problem.dim.
        self.NP_init: int = 0  # actual initial population after dim scaling
        self.NP_current: int = 0
        self.population: List[Optional[Result]] = []  # length NP_current
        self.archive: List[np.ndarray] = []  # bounded by archive_factor * NP_current
        self.M_F: np.ndarray = np.array([])  # length H, in (0, 1]
        self.M_CR: np.ndarray = np.array([])  # length H, in [0, 1]
        self._mem_pos: int = 0  # round-robin write index into M_F / M_CR

        # Pending trials: req_id -> (target_idx, F, CR, parent_penalty)
        # parent_penalty is the constraint-aware scalar at trial
        # creation, used to weight successful (F, CR) records by
        # improvement magnitude.
        self._pending: Dict[str, Tuple[int, float, float, float]] = {}

        # Successful (F, CR, Δpenalty) records since the last memory
        # update.  Cleared when `_update_memories` fires.
        self._success_F: List[float] = []
        self._success_CR: List[float] = []
        self._success_dF: List[float] = []

        # Counter of trial outcomes since the last memory update; the
        # memory update fires every NP_current outcomes (a "generation").
        self._outcomes_since_update: int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _default_NP_init(self, dim: int) -> int:
        """Return the Tanabe-Fukunaga default ``18 · dim`` clipped to ``cap``.

        Clip both upward (``cap`` from the strategy's queue capacity, so
        the heuristic does not exceed the queue's tolerance) and
        downward (``NP_min``).  If the user passed an explicit
        ``NP_init`` it overrides the dim-scaled default.
        """
        if self.NP_init_user is not None:
            return max(self.NP_init_user, self.NP_min)
        # The L-SHADE paper recommends 18*dim.  Clip at a generous upper
        # bound so a 50-D problem doesn't allocate hundreds of slots in
        # a budget-constrained Panobbgo run.  ``cap`` is the heuristic's
        # output-queue capacity, a natural ceiling.
        return max(min(18 * dim, max(self.cap // 2, self.NP_min)), self.NP_min)

    def _emit_trial(
        self,
        x: np.ndarray,
        target_idx: int,
        F: float,
        CR: float,
        parent_penalty: float,
    ) -> bool:
        """Emit a candidate point tagged with a fresh trial id.

        Returns True if the point was queued, False otherwise.
        """
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
        self._pending[req_id] = (target_idx, float(F), float(CR), float(parent_penalty))
        return True

    def _penalty(self, r: Optional[Result]) -> float:
        """Constraint-aware scalar fitness for ``r`` (lower is better).

        ``None`` results map to ``+inf`` so any populated slot strictly
        dominates an empty one for sorting purposes.
        """
        if r is None:
            return float("inf")
        return float(self.strategy.constraint_handler.get_penalty_value(r))

    def _filled_indices(self) -> List[int]:
        """Indices of population slots that hold a Result (i.e. are filled)."""
        return [i for i, r in enumerate(self.population) if r is not None]

    def _sorted_indices(self) -> List[int]:
        """Population indices sorted by penalty (best first).

        Empty slots are excluded.  Ties are broken stably by index so
        the order is deterministic for a given (population, handler)
        pair.
        """
        filled = self._filled_indices()
        return sorted(filled, key=lambda i: self._penalty(self.population[i]))

    def _select_pbest(self, exclude_idx: int) -> Optional[int]:
        """Sample uniformly from the top ``p_best_rate · NP`` (excluding self).

        Returns ``None`` when the pool is empty (e.g. only one filled
        slot and it is ``exclude_idx``).
        """
        sorted_idx = self._sorted_indices()
        if not sorted_idx:
            return None
        # At least 2 candidates so the pool is never empty after excluding self.
        p_count = max(2, int(np.ceil(self.p_best_rate * len(sorted_idx))))
        p_count = min(p_count, len(sorted_idx))
        candidates = [j for j in sorted_idx[:p_count] if j != exclude_idx]
        if not candidates:
            # Fallback: the entire top-p pool was just `exclude_idx`.
            # Pick the next best filled index instead.
            candidates = [j for j in sorted_idx if j != exclude_idx]
            if not candidates:
                return None
        return int(self._rng.choice(candidates))

    def _select_r1_r2(self, exclude_idxs: List[int]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Pick ``x_r1`` from the population and ``x_r2`` from population ∪ archive.

        Returns ``None`` if there aren't enough distinct individuals.
        ``r1`` and ``r2`` must be distinct from each other and from
        ``exclude_idxs`` (the target index and pbest index).  ``r2`` may
        come from the archive — that is the key JADE/L-SHADE diversity
        injection.
        """
        filled = [i for i in self._filled_indices() if i not in exclude_idxs]
        if not filled:
            return None
        i_r1 = int(self._rng.choice(filled))
        x_r1 = self.population[i_r1].x  # type: ignore[union-attr]

        # r2 sampled from population ∪ archive, distinct from r1 and the
        # excluded indices.
        union_pop_idxs = [i for i in filled if i != i_r1]
        archive_count = len(self.archive)
        total = len(union_pop_idxs) + archive_count
        if total == 0:
            return None
        pick = int(self._rng.integers(0, total))
        if pick < len(union_pop_idxs):
            x_r2 = self.population[union_pop_idxs[pick]].x  # type: ignore[union-attr]
        else:
            x_r2 = self.archive[pick - len(union_pop_idxs)]
        return np.asarray(x_r1, dtype=float), np.asarray(x_r2, dtype=float)

    def _generate_trial(self, target_idx: int) -> bool:
        """Create one trial vector for slot ``target_idx`` and emit it.

        Implements DE/current-to-pbest/1/bin with success-history-based
        adaptation of F and CR.  Returns True on successful emit.
        """
        if self.population[target_idx] is None:
            return False
        if not (0 <= self._mem_pos < self.H):  # paranoia
            self._mem_pos = 0

        # Sample (F, CR) from a randomly chosen memory slot.
        r = int(self._rng.integers(0, self.H))
        F = _sample_cauchy_F(self._rng, float(self.M_F[r]))
        CR = _sample_normal_CR(self._rng, float(self.M_CR[r]))

        # Pick pbest, r1, r2.
        pbest_idx = self._select_pbest(exclude_idx=target_idx)
        if pbest_idx is None:
            return False
        r1r2 = self._select_r1_r2(exclude_idxs=[target_idx, pbest_idx])
        if r1r2 is None:
            return False
        x_r1, x_r2 = r1r2

        x_target = np.asarray(self.population[target_idx].x, dtype=float)  # type: ignore[union-attr]
        x_pbest = np.asarray(self.population[pbest_idx].x, dtype=float)  # type: ignore[union-attr]

        # Mutation: current-to-pbest/1 with archive-aware r2.
        v = x_target + F * (x_pbest - x_target) + F * (x_r1 - x_r2)

        # Binomial crossover.
        dim = self.problem.dim
        cross_mask = self._rng.random(dim) < CR
        if not np.any(cross_mask):
            cross_mask[int(self._rng.integers(0, dim))] = True
        u = np.where(cross_mask, v, x_target)

        return self._emit_trial(
            u,
            target_idx,
            F=F,
            CR=CR,
            parent_penalty=self._penalty(self.population[target_idx]),
        )

    def _archive_add(self, x: np.ndarray) -> None:
        """Push ``x`` into the bounded external archive (oldest-out)."""
        cap = max(int(np.ceil(self.archive_factor * max(self.NP_current, 1))), 0)
        if cap == 0:
            return
        if len(self.archive) < cap:
            self.archive.append(np.array(x, dtype=float, copy=True))
        else:
            # Replace a uniformly random entry — the canonical L-SHADE
            # eviction rule.  Deterministic FIFO would bias the archive
            # toward recent generations only.
            i = int(self._rng.integers(0, cap))
            self.archive[i] = np.array(x, dtype=float, copy=True)

    def _update_memories(self) -> None:
        """Refresh M_F[mem_pos] and M_CR[mem_pos] from the success buffer."""
        if not self._success_F:
            return
        F_arr = np.asarray(self._success_F, dtype=float)
        CR_arr = np.asarray(self._success_CR, dtype=float)
        w_arr = np.asarray(self._success_dF, dtype=float)
        # Improvement-weighted Lehmer means.
        new_F = _weighted_lehmer_mean(F_arr, w_arr)
        new_CR = _weighted_lehmer_mean(CR_arr, w_arr)
        if np.isfinite(new_F) and 0.0 < new_F <= 1.0:
            self.M_F[self._mem_pos] = new_F
        if np.isfinite(new_CR) and 0.0 <= new_CR <= 1.0:
            self.M_CR[self._mem_pos] = new_CR
        self._mem_pos = (self._mem_pos + 1) % self.H
        self._success_F.clear()
        self._success_CR.clear()
        self._success_dF.clear()

    def _maybe_shrink(self) -> None:
        """Apply Linear Population Size Reduction if the budget is known.

        Drops the *worst* (highest-penalty) filled slots until the
        population matches the LPSR-prescribed size.  Empty slots are
        dropped first so the heuristic does not stall waiting for
        initial trials before it can shrink.  Pending trials whose
        target was dropped become orphaned and are silently discarded
        on arrival.
        """
        try:
            max_eval = float(self.strategy.config.max_eval)  # type: ignore[union-attr]
            current_eval = float(len(self.strategy.results))
        except Exception:
            return
        if not np.isfinite(max_eval) or max_eval <= 0.0:
            return

        progress = min(max(current_eval / max_eval, 0.0), 1.0)
        target_NP = int(round(self.NP_init - (self.NP_init - self.NP_min) * progress))
        target_NP = max(target_NP, self.NP_min)
        if target_NP >= self.NP_current:
            return  # nothing to do

        # Drop empty slots first (cheap), then worst-penalty filled ones.
        # We rebuild the population with `target_NP` survivors so indices
        # remain dense in [0, target_NP) — the bookkeeping for
        # `_select_pbest` / `_select_r1_r2` assumes that.
        empty = [i for i, r in enumerate(self.population) if r is None]
        if len(self.population) - len(empty) <= target_NP:
            # Filled count fits — keep all filled slots, drop empties first.
            survivors_idxs = self._filled_indices()[:target_NP]
        else:
            # Need to drop filled slots too: keep the best target_NP.
            survivors_idxs = self._sorted_indices()[:target_NP]

        survivors = [self.population[i] for i in survivors_idxs]
        # Pad with None if we somehow ended up short (won't happen for a
        # healthy population but keeps the invariant explicit).
        while len(survivors) < target_NP:
            survivors.append(None)
        # Remap pending trials: orphans (target dropped) are dropped on
        # arrival; survivors get a fresh index in the new compact layout.
        old_to_new: Dict[int, int] = {old: new for new, old in enumerate(survivors_idxs)}
        new_pending: Dict[str, Tuple[int, float, float, float]] = {}
        for req_id, (old_idx, F, CR, pp) in self._pending.items():
            if old_idx in old_to_new:
                new_pending[req_id] = (old_to_new[old_idx], F, CR, pp)
        self._pending = new_pending

        self.population = survivors
        self.NP_current = target_NP
        # Trim archive to the new cap.
        cap = max(int(np.ceil(self.archive_factor * self.NP_current)), 0)
        if len(self.archive) > cap:
            self.archive = self.archive[len(self.archive) - cap :]

    # ------------------------------------------------------------------
    # Heuristic interface
    # ------------------------------------------------------------------

    def on_start(self) -> None:
        """Allocate state and emit the initial population of random points."""
        dim = self.problem.dim
        self.NP_init = self._default_NP_init(dim)
        self.NP_current = self.NP_init
        self.population = [None] * self.NP_init
        self.archive = []
        # Initialise memories at the JADE/L-SHADE neutral defaults
        # (F = CR = 0.5).  The first generation immediately starts to
        # adapt them.
        self.M_F = np.full(self.H, 0.5, dtype=float)
        self.M_CR = np.full(self.H, 0.5, dtype=float)
        self._mem_pos = 0
        self._pending = {}
        self._success_F.clear()
        self._success_CR.clear()
        self._success_dF.clear()
        self._outcomes_since_update = 0

        # Emit the initial population as random points.  These come back
        # through on_new_results and fill `population` slots; only then
        # do we start producing real DE trials (which need at least one
        # filled slot per `r1, r2, pbest` term).
        for i in range(self.NP_init):
            x = self.problem.random_point()
            # F / CR are unused for initial trials but stored for
            # uniformity with the pending dict layout.
            self._emit_trial(x, i, F=0.5, CR=0.5, parent_penalty=float("inf"))

    def on_new_results(self, results) -> None:
        """Process incoming results and emit follow-up trials."""
        if not self.population:
            return  # not started yet

        prefix = f"{self.name}:"
        handler = self.strategy.constraint_handler
        for r in results:
            who: str = getattr(r, "who", "") or ""
            if not who.startswith(prefix):
                continue
            req_id = who[len(prefix) :]
            payload = self._pending.pop(req_id, None)
            if payload is None:
                continue  # stale or unknown trial id
            target_idx, F, CR, parent_penalty = payload

            # Slot may have been dropped by LPSR between emit and arrival.
            if target_idx >= self.NP_current:
                continue

            slot = self.population[target_idx]
            if slot is None:
                # Initial-population fill.  No archive update, no success.
                self.population[target_idx] = r
            else:
                if handler.is_better(slot, r):
                    # Trial wins.  Old parent goes to the archive,
                    # success record buffers (F, CR, improvement).
                    self._archive_add(np.asarray(slot.x, dtype=float))
                    new_penalty = self._penalty(r)
                    delta = parent_penalty - new_penalty
                    if not np.isfinite(delta):
                        delta = 0.0
                    if delta > 0.0:
                        self._success_F.append(F)
                        self._success_CR.append(CR)
                        self._success_dF.append(float(delta))
                    self.population[target_idx] = r
                # else: parent wins, no archive / success update.

            self._outcomes_since_update += 1
            # Memory update + LPSR fire on a "generation" boundary
            # defined as NP_current outcomes.
            if self._outcomes_since_update >= self.NP_current:
                self._update_memories()
                self._maybe_shrink()
                self._outcomes_since_update = 0

            # Emit the next trial for this slot, *if* the slot survived
            # the LPSR shrink and the population is healthy enough to
            # construct DE/current-to-pbest/1.
            if target_idx < self.NP_current and self.population[target_idx] is not None:
                if len(self._filled_indices()) >= 4:
                    self._generate_trial(target_idx)
                else:
                    # Population not ready yet — emit a fresh random
                    # point so the slot keeps producing.  This also
                    # bootstraps when initial trials trickle in
                    # one-by-one rather than as a batch.  ``parent_penalty``
                    # is the *current* slot's penalty (not ``inf``) so a
                    # successful improvement records a finite delta in the
                    # success buffer.
                    x = self.problem.random_point()
                    self._emit_trial(
                        target_idx=target_idx,
                        x=x,
                        F=0.5,
                        CR=0.5,
                        parent_penalty=self._penalty(self.population[target_idx]),
                    )

    def on_restart(self, center, reason: str = "") -> None:
        """Drop in-flight state and re-seed the population around ``center``.

        Mirrors the IPOP-style warm restart used by
        :class:`~panobbgo.heuristics.cma_es.CMAES` and
        :class:`~panobbgo.heuristics.pso.PSO`: clear pending trials,
        wipe the archive, reset memories to their neutral defaults, and
        scatter ``NP_init`` fresh random points around ``center``.
        """
        if self._stopped:
            return
        self.clear_output()
        self._pending.clear()
        if not self.population:
            return  # not started yet — nothing to reset

        dim = self.problem.dim
        self.NP_current = self.NP_init
        self.population = [None] * self.NP_init
        self.archive = []
        self.M_F = np.full(self.H, 0.5, dtype=float)
        self.M_CR = np.full(self.H, 0.5, dtype=float)
        self._mem_pos = 0
        self._success_F.clear()
        self._success_CR.clear()
        self._success_dF.clear()
        self._outcomes_since_update = 0

        if center is None:
            base = self.problem.random_point()
        else:
            base = np.asarray(center, dtype=float)
        # Scatter inside a small ball around `center` proportional to
        # the box ranges — the same heuristic the PSO restart uses.
        ranges = self.problem.box[:, 1] - self.problem.box[:, 0]
        for i in range(self.NP_init):
            offset = self._rng.uniform(-0.1, 0.1, size=dim) * ranges
            x = self.problem.project(base + offset)
            self._emit_trial(x, i, F=0.5, CR=0.5, parent_penalty=float("inf"))
