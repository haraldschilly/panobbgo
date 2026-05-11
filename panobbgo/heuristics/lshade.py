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
L-SHADE Heuristic (Linear Population Size Reduction Success-History DE)
========================================================================

L-SHADE (Tanabe & Fukunaga, 2014) is the CEC-2014 single-objective
competition winner and remains one of the strongest single-population
black-box optimizers in the literature.  It combines three ideas on top
of the classical :class:`~panobbgo.heuristics.differential_evolution.DifferentialEvolution`
(``DE/rand/1/bin``) implementation:

1. **Success-history adaptation of F and CR** (SHADE, Tanabe-Fukunaga
   2013).  Instead of fixed control parameters, the heuristic keeps a
   short history of recently successful ``F`` and ``CR`` values in two
   memory arrays ``M_F`` and ``M_CR`` of length ``H``.  Each trial
   samples its parameters from a Cauchy / Normal distribution centred
   on a random history slot, so good values are reused with
   exploration noise.  After every wave of trials, the memory slot
   ``M_F[k], M_CR[k]`` is overwritten with the Lehmer / arithmetic mean
   of *successful* values, weighted by their improvement size, and
   ``k`` rotates through the memory.

2. **"current-to-pbest/1" mutation with an external archive** (JADE,
   Zhang-Sanderson 2009).  The trial vector is

   .. math::

      v_i = x_i + F_i\\,(x_{\\mathrm{pbest}} - x_i)
                + F_i\\,(x_{r_1} - x_{r_2})

   where ``x_pbest`` is uniformly drawn from the top ``p · NP``
   individuals (``p ∈ [pbest_min, pbest_max]``, default
   ``[0.11, 0.20]`` — Tanabe-Fukunaga 2014), ``x_r1`` is drawn from
   the current population and ``x_r2`` is drawn from the *union* of
   the population and the external archive of recent losers.  The
   archive of size ``≤ |P|`` retains diversity that pure population
   replacement would otherwise lose.

3. **Linear Population Size Reduction (LPSR)**.  The population size
   shrinks linearly from ``N_init`` to ``N_min`` (typically ``4``) as
   the evaluation budget is spent:

   .. math::

      \\mathrm{NP}(g) = \\lfloor N_\\mathrm{init}
                  - (N_\\mathrm{init} - N_\\mathrm{min})\\,
                    \\mathrm{NFE} / \\mathrm{NFE}_\\mathrm{max} \\rceil

   Worst individuals are pruned to match the new ``NP``.  This
   concentrates the remaining budget on exploitation around the best
   discovered region without sacrificing the early-exploration value
   of a large initial population.

The defaults follow the published L-SHADE (Tanabe-Fukunaga 2014):
``N_init = 18·dim`` (capped at ``cap`` for sanity), ``N_min = 4``,
``H = 6`` memory slots, ``p_best ∈ [0.11, 0.20]``.

Asynchronous Adaptation
~~~~~~~~~~~~~~~~~~~~~~~

The literature L-SHADE iterates in synchronous generations: emit ``NP``
trials, evaluate all of them, then update the population and memory
in lockstep.  Panobbgo's event-driven loop delivers results one (or a
few) at a time, so this heuristic adapts as follows without losing the
algorithm's properties:

* Every trial is sampled from the *current* memory and *current*
  population state — i.e. the heuristic always uses the freshest
  information rather than waiting for a full generation to complete.
  The original L-SHADE is also asynchronous within a generation in
  that the memory is updated at most once per generation; we update
  per *wave* of ``NP_eff`` successful selections, which collapses to
  the literature schedule when results return in order.
* Comparison is done via the strategy's
  :attr:`~panobbgo.lib.constraints.ConstraintHandler.is_better`, so
  L-SHADE works on constrained problems out-of-the-box.
* LPSR is paced by ``len(strategy.results) / strategy.config.max_eval``.
  If the budget is unknown the heuristic falls back to a constant
  population at ``N_init`` (no LPSR), which preserves the SHADE
  behaviour.

The implementation follows:

* R. Tanabe & A. Fukunaga (2013). "Success-history based parameter
  adaptation for differential evolution." *CEC 2013.*
* R. Tanabe & A. Fukunaga (2014). "Improving the search performance
  of SHADE using linear population size reduction." *CEC 2014* —
  competition winner.
* J. Zhang & A. C. Sanderson (2009). "JADE: Adaptive differential
  evolution with optional external archive." *IEEE TEvC* 13(5):945–958.

.. codeauthor:: Harald Schilly
"""

from __future__ import annotations

import uuid
from typing import Dict, List, Optional

import numpy as np

from panobbgo.core import Heuristic
from panobbgo.lib import Point, Result


class LSHADE(Heuristic):
    """Linear Population Size Reduction Success-History Adaptive DE.

    See the module docstring for the algorithm reference and how the
    asynchronous adaptation differs from the literature schedule.

    Args:
        strategy: The owning :class:`~panobbgo.core.StrategyBase`.
        NP_init: Initial population size.  Default ``None`` defers to the
            problem dimension at :meth:`on_start` time and uses
            ``min(18·dim, NP_init_cap)``, the canonical L-SHADE setting
            (Tanabe-Fukunaga 2014).  Pass an integer to override.  Must
            be ``>= NP_min``.
        NP_init_cap: Upper bound on the auto-derived ``NP_init`` when
            ``NP_init is None``.  Default ``50`` so the heuristic is
            usable inside Panobbgo's typical 75–500 evaluation budgets.
        NP_min: Lower bound on the linearly-shrinking population.
            Default ``4`` — the literature value (you need at least
            four distinct candidates per mutation).  Must be ``>= 4``.
        H: Number of history slots in the success-history memory.
            Default ``6`` (Tanabe-Fukunaga 2014).  Larger ``H`` smooths
            adaptation but slows reaction to changing landscapes.
        pbest_min: Lower bound on the ``current-to-pbest/1`` ``p``
            fraction.  Default ``0.11`` (Tanabe-Fukunaga 2014).
        pbest_max: Upper bound on the ``p`` fraction.  Default ``0.20``.
            For every trial, ``p`` is drawn uniformly from
            ``[pbest_min, pbest_max]``.
        archive_factor: External archive size as a fraction of the
            current population size.  Default ``1.0`` (archive holds up
            to ``|P|`` losers); ``0.0`` disables the archive and
            reverts to pure population sampling.
        seed: Optional seed for the per-instance RNG.  ``None`` (default)
            seeds from ``np.random.default_rng()``.
        name: Override the heuristic's display name.

    Notes:
        - All numeric arguments are validated; :class:`ValueError` is
          raised on bad inputs.
        - The constraint handler determines selection — L-SHADE inherits
          constrained-optimisation support from
          :class:`DifferentialEvolution`.
        - LPSR is paced by the strategy's ``max_eval``; when that is not
          configured the heuristic runs SHADE (no shrinkage) at constant
          ``NP_init``.
    """

    def __init__(
        self,
        strategy,
        NP_init: Optional[int] = None,
        NP_init_cap: int = 50,
        NP_min: int = 4,
        H: int = 6,
        pbest_min: float = 0.11,
        pbest_max: float = 0.20,
        archive_factor: float = 1.0,
        seed: Optional[int] = None,
        name: Optional[str] = None,
    ) -> None:
        if NP_init is not None:
            if not isinstance(NP_init, int):
                raise ValueError(f"LSHADE: NP_init must be an integer or None, got {NP_init!r}")
            if NP_init < 4:
                raise ValueError(f"LSHADE: NP_init must be >= 4, got {NP_init}")
        if not isinstance(NP_init_cap, int) or NP_init_cap < 4:
            raise ValueError(f"LSHADE: NP_init_cap must be an integer >= 4, got {NP_init_cap!r}")
        if not isinstance(NP_min, int) or NP_min < 4:
            raise ValueError(f"LSHADE: NP_min must be an integer >= 4, got {NP_min!r}")
        if NP_init is not None and NP_init < NP_min:
            raise ValueError(f"LSHADE: NP_init ({NP_init}) must be >= NP_min ({NP_min})")
        if not isinstance(H, int) or H < 1:
            raise ValueError(f"LSHADE: H must be a positive integer, got {H!r}")
        if not (0.0 < pbest_min <= pbest_max <= 1.0):
            raise ValueError(f"LSHADE: require 0 < pbest_min ({pbest_min}) <= pbest_max ({pbest_max}) <= 1")
        if not np.isfinite(archive_factor) or archive_factor < 0.0:
            raise ValueError(f"LSHADE: archive_factor must be >= 0, got {archive_factor!r}")

        super().__init__(strategy, name=name or "LSHADE")
        self.NP_init_param: Optional[int] = NP_init
        self.NP_init_cap: int = int(NP_init_cap)
        self.NP_min: int = int(NP_min)
        self.H: int = int(H)
        self.pbest_min: float = float(pbest_min)
        self.pbest_max: float = float(pbest_max)
        self.archive_factor: float = float(archive_factor)
        self._rng: np.random.Generator = np.random.default_rng(seed)

        # Effective NP_init: resolved on on_start when problem.dim is known.
        self.NP_init: int = 0
        # Current population size (shrinks under LPSR).
        self.NP: int = 0

        # Population stored as parallel lists indexed by population slot.
        # ``_population[i]`` is the Result object currently occupying slot i
        # (or None during initial fill).  Best individual is identified by
        # the smallest penalty value via the strategy's constraint handler.
        self._population: List[Optional[Result]] = []
        # Number of initial population slots whose results have arrived.
        self._n_filled: int = 0

        # SHADE success-history memory.  Tanabe-Fukunaga 2014 initialise
        # both arrays to 0.5 — neutral starting point.
        self._M_F: np.ndarray = np.full(self.H, 0.5, dtype=float)
        self._M_CR: np.ndarray = np.full(self.H, 0.5, dtype=float)
        # Next memory slot to overwrite (rotates 0..H-1).
        self._k_mem: int = 0

        # External archive — list of Result objects of recent losers.
        self._archive: List[Result] = []

        # Pending trials: req_id -> dict with target slot index, the F
        # value used, and the CR value used.  Required for memory
        # adaptation when a trial succeeds.
        self._pending: Dict[str, Dict] = {}

        # Success buffer for the current "wave": parallel lists of
        # (F_used, CR_used, |f_target - f_trial|).  Flushed into the
        # success-history memory once we have NP successes (matching the
        # literature's "one update per generation" cadence in the limit).
        self._S_F: List[float] = []
        self._S_CR: List[float] = []
        self._S_delta: List[float] = []
        self._wave_outcomes: int = 0  # both successes and failures in current wave

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _penalty(self, r: Result) -> float:
        """Scalar penalty value used for ranking / improvement amount."""
        return float(self.strategy.constraint_handler.get_penalty_value(r))

    def _emit_trial(
        self,
        x: np.ndarray,
        target_idx: int,
        F_used: float,
        CR_used: float,
    ) -> bool:
        """Emit a trial vector and record its bookkeeping."""
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
        except Exception as exc:
            self.logger.debug(f"LSHADE: emit failed: {exc}")
            return False
        self._pending[req_id] = {
            "target": target_idx,
            "F": float(F_used),
            "CR": float(CR_used),
            "initial": False,
        }
        return True

    def _emit_initial(self, x: np.ndarray, target_idx: int) -> bool:
        """Emit an initial-population candidate (no F/CR adaptation)."""
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
        except Exception as exc:
            self.logger.debug(f"LSHADE: emit failed: {exc}")
            return False
        self._pending[req_id] = {
            "target": target_idx,
            "F": 0.5,
            "CR": 0.5,
            "initial": True,
        }
        return True

    def _sample_params(self) -> tuple:
        """Sample (F, CR) for one trial from the success-history memory.

        Follows Tanabe-Fukunaga 2013/2014:

        * Pick a memory slot ``r ∈ [0, H)`` uniformly.
        * ``CR ∼ N(M_CR[r], 0.1)``, clipped to ``[0, 1]``.
        * ``F ∼ Cauchy(M_F[r], 0.1)``, clipped at ``(0, 1]`` and resampled
          when negative.
        """
        r = int(self._rng.integers(0, self.H))
        mu_cr = float(self._M_CR[r])
        mu_f = float(self._M_F[r])

        if mu_cr < 0.0:
            CR = 0.0
        else:
            CR = float(self._rng.normal(mu_cr, 0.1))
            CR = min(1.0, max(0.0, CR))

        # Cauchy sample with up to a few resamples if non-positive.
        # The literature retries until F > 0, which terminates a.s.;
        # cap retries to keep this loop bounded.
        for _ in range(32):
            F = float(mu_f + 0.1 * self._rng.standard_cauchy())
            if F > 0.0:
                break
        else:
            F = 0.5  # extremely unlikely fall-back
        F = min(1.0, F)
        return F, CR

    def _current_target_NP(self) -> int:
        """LPSR: target population size at the current NFE.

        Linearly interpolates ``NP_init → NP_min`` over ``[0, max_eval]``.
        Returns ``NP_init`` (no shrinkage) when the budget is unknown.
        """
        try:
            max_eval = float(self.strategy.config.max_eval)
            current = float(len(self.strategy.results))
        except Exception:
            return self.NP_init
        if not np.isfinite(max_eval) or max_eval <= 0.0:
            return self.NP_init
        progress = min(max(current / max_eval, 0.0), 1.0)
        target = self.NP_init - (self.NP_init - self.NP_min) * progress
        return max(self.NP_min, int(round(target)))

    def _shrink_population_if_needed(self) -> None:
        """Drop worst individuals to match the LPSR-scheduled NP.

        Idempotent: when no shrinkage is needed (or when not enough
        slots are filled to compute penalties) this is a no-op.  The
        archive shrinks alongside the population — the archive cap is
        ``archive_factor · NP``.
        """
        target = self._current_target_NP()
        if target >= self.NP:
            return  # never grow back

        # We only prune individuals whose results have arrived.  If
        # init is still in flight we leave the slots as-is; this
        # function will be called again when more results arrive.
        filled = [(i, r) for i, r in enumerate(self._population) if r is not None]
        if len(filled) <= target:
            self.NP = target
            self._cap_archive()
            return

        # Rank filled slots best-first via the constraint handler.
        filled.sort(key=lambda ir: self._penalty(ir[1]))
        keep_indices = {i for i, _ in filled[:target]}

        # Compact the population list to length ``target`` so further
        # indexing stays simple.  Drop slots whose result hasn't
        # arrived yet (they would have been pruned anyway in the
        # synchronous schedule).
        new_population: List[Optional[Result]] = []
        for i, r in enumerate(self._population):
            if i in keep_indices:
                new_population.append(r)
        self._population = new_population
        self._n_filled = len(self._population)
        self.NP = target

        # Pending trials whose target slot no longer exists become
        # detached.  We keep their bookkeeping so memory updates still
        # happen, but on result-arrival the target slot will simply not
        # be there to update — we treat it as a no-op replacement.
        self._cap_archive()

    def _cap_archive(self) -> None:
        cap = int(round(self.archive_factor * self.NP))
        if cap <= 0:
            self._archive = []
            return
        while len(self._archive) > cap:
            # Drop the oldest archive member; Tanabe-Fukunaga 2014 use a
            # random eviction, but FIFO is simpler and empirically
            # comparable for the archive sizes used in practice.
            self._archive.pop(0)

    def _select_pbest_idx(self) -> Optional[int]:
        """Index of a random member of the top ``⌈p · NP⌉`` individuals."""
        filled = [(i, r) for i, r in enumerate(self._population) if r is not None]
        if not filled:
            return None
        filled.sort(key=lambda ir: self._penalty(ir[1]))
        p = float(self._rng.uniform(self.pbest_min, self.pbest_max))
        n_top = max(2, int(np.ceil(p * len(filled))))
        n_top = min(n_top, len(filled))
        choice = int(self._rng.integers(0, n_top))
        return filled[choice][0]

    def _generate_trial(self, target_idx: int) -> None:
        """Produce a current-to-pbest/1 trial for the target slot.

        Requires the population to have at least 4 filled slots so the
        three additional members can be drawn distinctly.
        """
        filled_idx = [i for i, r in enumerate(self._population) if r is not None]
        if len(filled_idx) < 4:
            return

        target = self._population[target_idx]
        if target is None:
            # Target slot was pruned (or never filled).  Re-emit a random
            # point so the slot can recover.
            x = self.problem.random_point()
            self._emit_initial(x, target_idx)
            return

        F, CR = self._sample_params()
        x_target = np.asarray(target.x, dtype=float)

        pbest_idx = self._select_pbest_idx()
        if pbest_idx is None:
            return
        pbest_member = self._population[pbest_idx]
        if pbest_member is None:
            return
        x_pbest = np.asarray(pbest_member.x, dtype=float)

        # r1 from population \ {target}
        candidates_r1 = [i for i in filled_idx if i != target_idx]
        r1_idx = int(self._rng.choice(candidates_r1))
        r1_member = self._population[r1_idx]
        if r1_member is None:
            return
        x_r1 = np.asarray(r1_member.x, dtype=float)

        # r2 from population ∪ archive, distinct from target and r1.
        pop_union: List[np.ndarray] = []
        for j in filled_idx:
            if j == target_idx or j == r1_idx:
                continue
            member = self._population[j]
            if member is None:
                continue
            pop_union.append(np.asarray(member.x, dtype=float))
        for ar in self._archive:
            pop_union.append(np.asarray(ar.x, dtype=float))
        if not pop_union:
            return
        r2_pick = int(self._rng.integers(0, len(pop_union)))
        x_r2 = pop_union[r2_pick]

        # current-to-pbest/1 mutation
        v = x_target + F * (x_pbest - x_target) + F * (x_r1 - x_r2)

        # Binomial crossover with a guaranteed cross point.
        dim = self.problem.dim
        cross_mask = self._rng.random(dim) < CR
        j_rand = int(self._rng.integers(0, dim))
        cross_mask[j_rand] = True
        u = np.where(cross_mask, v, x_target)

        # Bound handling: midpoint reflection on violation (Tanabe-Fukunaga
        # 2014 §III-B).  Where ``u`` lies outside the box, set
        # ``u_j = (box_lo + x_target_j) / 2`` for the low side and
        # symmetrically for the high side.  This keeps the trial inside
        # the box without collapsing to the boundary.
        try:
            box = self.problem.box  # shape (dim, 2)
            low = box[:, 0]
            high = box[:, 1]
            below = u < low
            above = u > high
            if np.any(below):
                u = np.where(below, (low + x_target) * 0.5, u)
            if np.any(above):
                u = np.where(above, (high + x_target) * 0.5, u)
        except Exception:
            # If the problem doesn't expose box (unlikely in practice)
            # fall back to the standard project() in _emit_trial.
            pass

        self._emit_trial(u, target_idx, F, CR)

    def _update_memory(self) -> None:
        """Overwrite ``M_F[k_mem], M_CR[k_mem]`` from the success buffer.

        Tanabe-Fukunaga 2014 use the *weighted Lehmer mean* of successful
        F values and the *weighted arithmetic mean* of successful CR
        values, where the weight is ``Δf_i / Σ_j Δf_j``.
        """
        if not self._S_F:
            return  # no successes — memory stays put
        S_F = np.asarray(self._S_F, dtype=float)
        S_CR = np.asarray(self._S_CR, dtype=float)
        S_delta = np.asarray(self._S_delta, dtype=float)

        total = float(S_delta.sum())
        if total <= 0.0:
            weights = np.full_like(S_delta, 1.0 / len(S_delta))
        else:
            weights = S_delta / total

        # Lehmer mean of F (numerator: Σ w·F² ; denominator: Σ w·F).
        denom_F = float((weights * S_F).sum())
        if denom_F > 0.0:
            mean_F = float((weights * S_F * S_F).sum() / denom_F)
        else:
            mean_F = float(self._M_F[self._k_mem])
        mean_F = max(0.0, min(1.0, mean_F))

        # Arithmetic mean of CR; if every success had CR ≈ 0 the slot
        # is "frozen" to −1 so future samples set CR = 0 directly
        # (Tanabe-Fukunaga 2014 §III-C: "M_CR,k,G+1 = ⊥").
        mean_CR = float((weights * S_CR).sum())
        if mean_CR < 1e-12 and float(S_CR.max(initial=0.0)) < 1e-12:
            mean_CR = -1.0
        else:
            mean_CR = max(0.0, min(1.0, mean_CR))

        self._M_F[self._k_mem] = mean_F
        self._M_CR[self._k_mem] = mean_CR
        self._k_mem = (self._k_mem + 1) % self.H

        # Clear the success buffer; the next wave starts fresh.
        self._S_F.clear()
        self._S_CR.clear()
        self._S_delta.clear()
        self._wave_outcomes = 0

    # ------------------------------------------------------------------
    # Heuristic interface
    # ------------------------------------------------------------------

    def on_start(self) -> None:
        """Resolve ``NP_init`` from the problem dim and emit the seed population."""
        dim = self.problem.dim
        if self.NP_init_param is not None:
            self.NP_init = int(self.NP_init_param)
        else:
            self.NP_init = max(self.NP_min, min(self.NP_init_cap, 18 * int(dim)))
        self.NP = self.NP_init

        self._population = [None] * self.NP
        self._n_filled = 0
        self._M_F = np.full(self.H, 0.5, dtype=float)
        self._M_CR = np.full(self.H, 0.5, dtype=float)
        self._k_mem = 0
        self._archive = []
        self._pending = {}
        self._S_F.clear()
        self._S_CR.clear()
        self._S_delta.clear()
        self._wave_outcomes = 0

        for i in range(self.NP):
            x = self.problem.random_point()
            self._emit_initial(x, i)

    def on_new_results(self, results) -> None:
        """Process incoming results, update population/memory, emit follow-ups."""
        if not self._population:
            return  # not started yet

        prefix = f"{self.name}:"
        handler = self.strategy.constraint_handler

        # We may receive many results in one batch.  Each generates at
        # most one follow-up trial (the standard async-DE pattern); but
        # we may emit *extra* follow-ups during the initial fill so
        # idle slots get reactivated once the population is large
        # enough.
        for r in results:
            who: str = getattr(r, "who", "") or ""
            if not who.startswith(prefix):
                continue
            req_id = who[len(prefix) :]
            meta = self._pending.pop(req_id, None)
            if meta is None:
                continue  # stale id (e.g. left over after on_restart)
            target_idx = int(meta["target"])

            # Slot may have been pruned by LPSR; in that case ignore the
            # result (it doesn't reduce the population) but still count
            # it toward the wave so memory updates eventually fire.
            slot_alive = 0 <= target_idx < len(self._population)

            if meta["initial"]:
                # Initial-population fill: just store the result.
                if slot_alive and self._population[target_idx] is None:
                    self._population[target_idx] = r
                    self._n_filled += 1
            else:
                # Trial competing with target slot.
                if slot_alive:
                    target = self._population[target_idx]
                    if target is None:
                        # Empty slot — accept unconditionally.
                        self._population[target_idx] = r
                    else:
                        # Selection.
                        if handler.is_better(target, r):
                            # Improved!  Archive the old target, replace
                            # in the population, log success for memory.
                            f_old = self._penalty(target)
                            f_new = self._penalty(r)
                            self._archive.append(target)
                            self._cap_archive()
                            self._population[target_idx] = r
                            self._S_F.append(float(meta["F"]))
                            self._S_CR.append(float(meta["CR"]))
                            self._S_delta.append(max(0.0, f_old - f_new))
                self._wave_outcomes += 1

            # If the wave has produced NP outcomes, refresh memory.
            if self._wave_outcomes >= self.NP:
                self._update_memory()

            # LPSR check — shrink the population if the schedule says
            # we should.  Done after every result so the schedule is
            # smooth rather than stepwise.  Note this can invalidate
            # the ``target_idx`` for this trial (the slot was pruned),
            # so we recompute slot_alive after the shrink.
            self._shrink_population_if_needed()
            slot_alive = 0 <= target_idx < len(self._population)

            # Decide what to emit next.
            if self._n_filled < len(self._population):
                # Still in initial fill: do nothing (the initial wave
                # already queued NP points; we just wait for them).  But
                # if results arrived after a partial fill and we already
                # have enough to evolve, kick off DE for filled slots.
                if self._n_filled >= 4:
                    # Generate a follow-up for this result's slot.
                    if slot_alive and self._population[target_idx] is not None:
                        self._generate_trial(target_idx)
            else:
                # Population fully filled — generate a successor trial
                # for the slot that just received a result.
                if slot_alive and self._population[target_idx] is not None:
                    self._generate_trial(target_idx)

    def on_restart(self, center, reason: str = "") -> None:
        """Reset population around ``center`` and flush state.

        Mirrors the IPOP-style restart used by
        :class:`~panobbgo.heuristics.cma_es.CMAES` and
        :class:`~panobbgo.heuristics.pso.PSO`: pending trials are
        dropped, the population is re-initialised in a small
        randomised ball around the suggested center, the
        success-history memory is reset to neutral values, and a fresh
        initial wave is emitted.
        """
        if self._stopped:
            return
        self.clear_output()
        self._pending.clear()

        # Resize back to NP_init for the fresh restart so LPSR has room
        # to operate again.  This mirrors L-SHADE's behaviour when used
        # under IPOP-style restart in the published literature.
        self.NP = self.NP_init
        self._population = [None] * self.NP
        self._n_filled = 0
        self._archive.clear()
        self._S_F.clear()
        self._S_CR.clear()
        self._S_delta.clear()
        self._wave_outcomes = 0
        self._M_F = np.full(self.H, 0.5, dtype=float)
        self._M_CR = np.full(self.H, 0.5, dtype=float)
        self._k_mem = 0

        if center is None:
            for i in range(self.NP):
                self._emit_initial(self.problem.random_point(), i)
            return

        # Scatter around the new center with a small radius derived
        # from the box ranges so the restart is meaningful even when
        # the search has converged tightly.
        try:
            box = self.problem.box
            ranges = box[:, 1] - box[:, 0]
        except Exception:
            ranges = np.ones(self.problem.dim, dtype=float)
        radius = 0.1 * ranges
        base = np.asarray(center, dtype=float)
        for i in range(self.NP):
            offset = self._rng.uniform(-radius, radius)
            self._emit_initial(base + offset, i)
