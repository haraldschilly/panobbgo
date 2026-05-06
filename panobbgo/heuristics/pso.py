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
Particle Swarm Optimization (PSO) Heuristic
============================================

Asynchronous Particle Swarm Optimization with the canonical
Clerc-Kennedy (2002) constriction-coefficient parameters.

PSO maintains a population of *particles*, each with a position ``x_i``,
a velocity ``v_i``, and a memory of its best-so-far position ``pbest_i``.
On every step a particle is pulled toward its personal best and toward
its *neighbourhood's* best position ``nbest_i``::

    v_i ← w · v_i + c1·r1·(pbest_i − x_i) + c2·r2·(nbest_i − x_i)
    x_i ← x_i + v_i

with ``r1, r2 ∼ U(0, 1)^d`` independent per-component random vectors.
``nbest_i`` is the best of the personal bests inside the *topology*
neighbourhood of particle ``i``:

* ``topology="gbest"`` (default) — every particle's neighbourhood is the
  whole swarm, so ``nbest_i = gbest`` for all ``i``.  Fastest contraction
  but most prone to premature convergence on multimodal problems.
* ``topology="lbest"`` — ring topology, particle ``i`` sees the ``lbest_k``
  particles on each side along the ring (default ``k=2``, so a
  five-particle window centred on ``i``).  Slower contraction but better
  multimodal exploration; information about a new basin propagates
  around the ring rather than instantly to all particles.

Default constriction parameters use the canonical values
``φ = c1 + c2 = 4.1`` and ``χ = 2 / (φ − 2 + √(φ² − 4·φ)) ≈ 0.7298``,
which together with ``c1 = c2 = 2.05`` yield effective coefficients
``χ·c1 = χ·c2 ≈ 1.49618`` and an inertia weight ``w = χ ≈ 0.7298``.

Optional **adaptive inertia** (Shi & Eberhart, 1998): pass ``w_end`` to
linearly decrease the inertia weight from ``w`` (the *initial* value) to
``w_end`` over the strategy's evaluation budget — the swarm explores
broadly with high inertia early and refines with low inertia late.  The
schedule is paced by ``len(strategy.results) / strategy.config.max_eval``;
when the budget is unknown the heuristic falls back to constant ``w``.

Key differences from existing population heuristics:

* :class:`~panobbgo.heuristics.cma_es.CMAES` adapts a *covariance matrix*
  and re-samples from it; it has no per-individual memory.
* :class:`~panobbgo.heuristics.differential_evolution.DifferentialEvolution`
  generates trial vectors via *recombination* of three randomly-chosen
  population members (``DE/rand/1``); it has no momentum.
* PSO carries a *velocity* (momentum) per particle and uses *social*
  attraction toward its neighbourhood's best; this gives it markedly
  different exploration dynamics — fast contraction once a basin is
  found while retaining inertia from the prior search direction.

The implementation runs **asynchronously** inside the panobbgo event loop,
following the same pattern as :class:`DifferentialEvolution`:

1. ``on_start()`` emits ``NP`` random initial positions, one per particle.
2. ``on_new_results()`` matches incoming results back to their particle
   index (via the ``who`` tag), updates ``pbest_i`` if the result improves,
   refreshes the cached neighbourhood bests, and emits the particle's next
   position generated from the velocity update above.
3. ``on_restart(center, reason)`` resets all particles to a randomized
   ball around the new center, drops in-flight trials, and starts a fresh
   swarm (matching the "warm restart" behaviour of CMA-ES IPOP).

References
----------

* J. Kennedy & R. Eberhart (1995). "Particle Swarm Optimization."
  *Proceedings of ICNN'95.*
* Y. Shi & R. Eberhart (1998). "A Modified Particle Swarm Optimizer."
  *Proceedings of the IEEE International Conference on Evolutionary
  Computation*, pages 69–73 — linearly decreasing inertia weight.
* M. Clerc & J. Kennedy (2002). "The Particle Swarm — Explosion, Stability,
  and Convergence in a Multidimensional Complex Space."
  *IEEE Transactions on Evolutionary Computation*, 6(1):58–73.
* R. Mendes, J. Kennedy, J. Neves (2004). "The Fully Informed Particle
  Swarm: Simpler, Maybe Better." *IEEE Transactions on Evolutionary
  Computation*, 8(3):204–210 — neighbourhood topologies.
* R. Poli, J. Kennedy, T. Blackwell (2007). "Particle Swarm Optimization:
  An Overview." *Swarm Intelligence*, 1(1):33–57.
"""

from __future__ import annotations

import uuid
from typing import Dict, List, Optional

import numpy as np

from panobbgo.core import Heuristic
from panobbgo.lib import Point, Result


# Canonical Clerc-Kennedy (2002) constriction-coefficient parameters.
# χ = 2 / (φ − 2 + √(φ² − 4·φ))  with φ = c1 + c2 = 4.1
_DEFAULT_W: float = 0.7298437881283576  # χ
_DEFAULT_C1: float = 1.49618  # χ · 2.05
_DEFAULT_C2: float = 1.49618  # χ · 2.05


_VALID_TOPOLOGIES: tuple = ("gbest", "lbest")


class PSO(Heuristic):
    """Asynchronous Particle Swarm Optimization heuristic.

    Args:
        strategy: The owning :class:`~panobbgo.core.StrategyBase`.
        NP: Swarm size (number of particles).  Default ``20`` — large
            enough to avoid premature convergence on multimodal problems
            yet cheap enough that the per-iteration overhead is tiny next
            to a typical black-box evaluation cost.
        w: Inertia weight (``χ`` in the constriction formulation).
            Default ``0.7298`` — the canonical Clerc-Kennedy value that
            provably guarantees convergence with ``c1 = c2 = 1.49618``.
            Also acts as the *initial* value of the linearly-decreasing
            inertia schedule when ``w_end`` is provided.
        c1: Cognitive (personal-best attraction) coefficient.
            Default ``1.49618`` — Clerc-Kennedy.
        c2: Social (neighbourhood-best attraction) coefficient.
            Default ``1.49618`` — Clerc-Kennedy.
        v_max_frac: Maximum velocity per dimension as a fraction of the
            corresponding box range.  Velocities are clamped to
            ``[-v_max_frac · range, +v_max_frac · range]`` to prevent the
            swarm from exploding outside the search box.  Default
            ``0.5`` — a common conservative choice.
        topology: Neighbourhood structure.  ``"gbest"`` (default) — every
            particle pulls toward the swarm-wide global best.
            ``"lbest"`` — ring topology, each particle sees only the
            ``lbest_k`` neighbours on each side along the ring (so a
            ``2*lbest_k + 1`` window centred on itself, including itself).
            ``"lbest"`` slows premature convergence on multimodal
            problems by letting better information about a new basin
            propagate gradually around the ring instead of pulling the
            entire swarm immediately.
        lbest_k: Half-width of the ring neighbourhood when
            ``topology="lbest"``.  Each particle sees ``lbest_k``
            neighbours on its left, ``lbest_k`` on its right, plus
            itself.  Must be ``>= 1`` and ``< NP / 2``; otherwise the
            ring degenerates and the topology is effectively ``gbest``.
            Default ``2``.  Ignored when ``topology="gbest"``.
        w_end: Final inertia weight for the linearly-decreasing
            (Shi-Eberhart 1998) schedule.  When set, the inertia at
            evaluation count ``e`` (out of ``E = strategy.config.max_eval``)
            is ``w_eff(e) = w − (w − w_end) · min(e / E, 1)``.  When
            ``None`` (default) inertia is constant at ``w`` — the original
            Clerc-Kennedy behaviour.  Common choice: ``w = 0.9``,
            ``w_end = 0.4``.  Ignored when the strategy budget is unknown
            (no ``max_eval`` configured), in which case the heuristic
            falls back to constant ``w``.
        seed: Optional seed for the per-instance RNG.  ``None`` (default)
            seeds from ``np.random.default_rng()``.
        name: Override the heuristic's display name.

    Notes:
        - The constructor validates all numeric arguments and raises
          :class:`ValueError` on bad inputs.
        - All particle bookkeeping (positions, velocities, personal bests,
          neighbourhood bests) lives in the heuristic instance — no
          shared global state, so multiple PSO heuristics in one strategy
          are independent.
        - The heuristic respects constraints via
          ``self.strategy.constraint_handler.is_better`` and
          ``get_penalty_value`` exactly like
          :class:`DifferentialEvolution`.
    """

    def __init__(
        self,
        strategy,
        NP: int = 20,
        w: float = _DEFAULT_W,
        c1: float = _DEFAULT_C1,
        c2: float = _DEFAULT_C2,
        v_max_frac: float = 0.5,
        topology: str = "gbest",
        lbest_k: int = 2,
        w_end: Optional[float] = None,
        seed: Optional[int] = None,
        name: Optional[str] = None,
    ) -> None:
        if not isinstance(NP, int):
            raise ValueError(f"PSO: NP must be an integer, got {NP!r}")
        if NP < 2:
            raise ValueError(f"PSO: NP must be >= 2, got {NP}")
        if not np.isfinite(w):
            raise ValueError(f"PSO: w must be finite, got {w}")
        if not np.isfinite(c1) or c1 < 0.0:
            raise ValueError(f"PSO: c1 must be a non-negative finite float, got {c1}")
        if not np.isfinite(c2) or c2 < 0.0:
            raise ValueError(f"PSO: c2 must be a non-negative finite float, got {c2}")
        if not np.isfinite(v_max_frac) or v_max_frac <= 0.0:
            raise ValueError(f"PSO: v_max_frac must be a positive finite float, got {v_max_frac}")
        if topology not in _VALID_TOPOLOGIES:
            raise ValueError(f"PSO: topology must be one of {_VALID_TOPOLOGIES}, got {topology!r}")
        if not isinstance(lbest_k, int):
            raise ValueError(f"PSO: lbest_k must be an integer, got {lbest_k!r}")
        if lbest_k < 1:
            raise ValueError(f"PSO: lbest_k must be >= 1, got {lbest_k}")
        if w_end is not None:
            if not np.isfinite(w_end):
                raise ValueError(f"PSO: w_end must be finite when set, got {w_end}")

        super().__init__(strategy, name=name or "PSO")
        self.NP: int = NP
        self.w: float = float(w)
        self.c1: float = float(c1)
        self.c2: float = float(c2)
        self.v_max_frac: float = float(v_max_frac)
        self.topology: str = topology
        self.lbest_k: int = lbest_k
        self.w_end: Optional[float] = None if w_end is None else float(w_end)
        self._rng: np.random.Generator = np.random.default_rng(seed)

        # Per-particle state.  Sized once on_start() runs (we need
        # ``problem.dim`` to allocate velocity arrays).
        self._positions: Optional[np.ndarray] = None  # (NP, dim)
        self._velocities: Optional[np.ndarray] = None  # (NP, dim)
        self._pbest_x: Optional[np.ndarray] = None  # (NP, dim)
        self._pbest_result: List[Optional[Result]] = []  # length NP
        self._gbest_idx: Optional[int] = None  # index into _pbest_result

        # Pending trials: req_id -> particle index.  When a result with
        # this id arrives we know which particle slot to update.  A
        # particle has at most one trial pending at any time.
        self._pending: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _v_max(self) -> np.ndarray:
        """Per-dimension velocity clamp from the problem box ranges."""
        ranges = self.problem.box[:, 1] - self.problem.box[:, 0]
        return self.v_max_frac * ranges

    def _emit_trial(self, x: np.ndarray, particle_idx: int) -> bool:
        """Emit a candidate point tagged with a fresh trial id.

        Returns True if the point was queued, False otherwise.  The
        heuristic stops emitting once :attr:`_stopped` is set.
        """
        if self._stopped:
            return False
        try:
            x_proj = self.problem.project(x)
        except Exception as exc:
            self.logger.debug(f"PSO: projection failed: {exc}")
            return False

        req_id = uuid.uuid4().hex
        who = f"{self.name}:{req_id}"
        try:
            self._output.put_nowait(Point(x_proj, who))
        except Exception as exc:  # queue full or shutdown
            self.logger.debug(f"PSO: emit failed: {exc}")
            return False
        self._pending[req_id] = particle_idx
        # Remember the actually-evaluated position so the velocity
        # update next time uses the projected coordinates (not the
        # pre-projection ones, which may be outside the box).
        if self._positions is not None:
            self._positions[particle_idx] = x_proj
        return True

    def _update_global_best(self) -> None:
        """Recompute ``_gbest_idx`` from the current per-particle bests."""
        best_idx: Optional[int] = None
        best_result: Optional[Result] = None
        handler = self.strategy.constraint_handler
        for i, pb in enumerate(self._pbest_result):
            if pb is None:
                continue
            if best_result is None or handler.is_better(best_result, pb):
                best_result = pb
                best_idx = i
        self._gbest_idx = best_idx

    def _neighbourhood_best_idx(self, particle_idx: int) -> Optional[int]:
        """Return the index of the best ``pbest`` in ``particle_idx``'s neighbourhood.

        For ``topology="gbest"`` the neighbourhood is the whole swarm and
        this collapses to ``self._gbest_idx``.  For ``topology="lbest"``
        only the ring neighbours within ``lbest_k`` of ``particle_idx``
        plus the particle itself contribute, so each particle moves
        toward a *local* (not global) best — the classic Kennedy 1999
        ring topology.

        Returns ``None`` when no particle in the neighbourhood has a
        scored personal best yet.
        """
        if self.topology == "gbest":
            return self._gbest_idx

        # lbest: ring window of half-width lbest_k centred on particle_idx.
        # Cap k at NP // 2 so the window never exceeds the swarm — at the
        # cap the lbest topology naturally degenerates to gbest, which is
        # the documented behaviour.
        k = min(self.lbest_k, self.NP // 2)
        handler = self.strategy.constraint_handler
        best_idx: Optional[int] = None
        best_result: Optional[Result] = None
        for offset in range(-k, k + 1):
            j = (particle_idx + offset) % self.NP
            pb = self._pbest_result[j]
            if pb is None:
                continue
            if best_result is None or handler.is_better(best_result, pb):
                best_result = pb
                best_idx = j
        return best_idx

    def _current_inertia(self) -> float:
        """Return the inertia weight to use for the next velocity update.

        When ``w_end`` is ``None`` this is the constant ``self.w``.
        Otherwise it is a linearly-decreasing schedule paced by
        ``len(strategy.results) / strategy.config.max_eval`` (Shi &
        Eberhart, 1998).  When the budget is unknown (no ``max_eval``,
        zero, or non-numeric) the heuristic falls back to constant
        ``self.w`` rather than guessing a horizon.
        """
        if self.w_end is None:
            return self.w
        try:
            max_eval = float(self.strategy.config.max_eval)  # type: ignore[union-attr]
            current = float(len(self.strategy.results))
        except Exception:
            return self.w
        if not np.isfinite(max_eval) or max_eval <= 0.0:
            return self.w
        progress = min(max(current / max_eval, 0.0), 1.0)
        return self.w - (self.w - self.w_end) * progress

    def _generate_next(self, particle_idx: int) -> None:
        """Produce the next candidate position for ``particle_idx``.

        Falls back to a fresh random point if we don't yet have a
        neighbourhood best (e.g. all initial trials still pending).
        """
        if self._positions is None or self._velocities is None or self._pbest_x is None:
            return

        nbest_idx = self._neighbourhood_best_idx(particle_idx)
        if nbest_idx is None or self._pbest_x[particle_idx] is None:
            # No memory to pull from yet — emit a fresh random point so
            # the particle stays active.
            x = self.problem.random_point()
            self._emit_trial(x, particle_idx)
            return

        dim = self.problem.dim
        x_i = self._positions[particle_idx]
        v_i = self._velocities[particle_idx]
        p_i = self._pbest_x[particle_idx]
        n = self._pbest_x[nbest_idx]

        w = self._current_inertia()
        r1 = self._rng.random(dim)
        r2 = self._rng.random(dim)
        new_v = w * v_i + self.c1 * r1 * (p_i - x_i) + self.c2 * r2 * (n - x_i)

        # Velocity clamp.
        v_max = self._v_max()
        np.clip(new_v, -v_max, v_max, out=new_v)
        self._velocities[particle_idx] = new_v

        new_x = x_i + new_v
        self._emit_trial(new_x, particle_idx)

    # ------------------------------------------------------------------
    # Heuristic interface
    # ------------------------------------------------------------------

    def on_start(self) -> None:
        """Allocate state and emit the initial swarm of random positions."""
        dim = self.problem.dim
        self._positions = np.zeros((self.NP, dim), dtype=float)
        self._velocities = np.zeros((self.NP, dim), dtype=float)
        self._pbest_x = np.zeros((self.NP, dim), dtype=float)
        self._pbest_result = [None] * self.NP
        self._gbest_idx = None
        self._pending = {}

        # Initial velocities are drawn uniformly inside the velocity
        # clamp.  This avoids the "stalled at zero velocity for the
        # first iteration" pathology of pure-zero-init PSO.
        v_max = self._v_max()
        self._velocities = self._rng.uniform(-v_max, v_max, size=(self.NP, dim))

        for i in range(self.NP):
            x = self.problem.random_point()
            self._positions[i] = x
            self._emit_trial(x, i)

    def on_new_results(self, results) -> None:
        """Process incoming evaluation results and emit follow-up trials."""
        if self._positions is None or self._pbest_x is None:
            return  # not started yet

        prefix = f"{self.name}:"
        handler = self.strategy.constraint_handler
        for r in results:
            who: str = getattr(r, "who", "") or ""
            if not who.startswith(prefix):
                continue
            req_id = who[len(prefix) :]
            particle_idx = self._pending.pop(req_id, None)
            if particle_idx is None:
                continue  # stale or unknown trial id

            # Update personal best.
            current = self._pbest_result[particle_idx]
            if current is None or handler.is_better(current, r):
                self._pbest_result[particle_idx] = r
                self._pbest_x[particle_idx] = np.asarray(r.x, dtype=float)

            # Refresh global best after every update so the next
            # particle to move can immediately benefit from the new
            # information.  Cheap: O(NP) per result.
            self._update_global_best()

            # Emit the next trial for this particle.
            self._generate_next(particle_idx)

    def on_restart(self, center, reason: str = "") -> None:
        """Drop in-flight trials and reseed the swarm around ``center``.

        The behaviour mirrors the IPOP-style warm restart used by
        :class:`~panobbgo.heuristics.cma_es.CMAES`: the global memory is
        wiped, particles are scattered around the suggested center, and
        the next evaluation cycle behaves as if the heuristic had just
        started — except the strategy keeps its accumulated history.
        """
        if self._stopped:
            return
        self.clear_output()
        self._pending.clear()
        if self._positions is None or self._velocities is None or self._pbest_x is None:
            return  # not started yet — nothing to reset

        dim = self.problem.dim
        v_max = self._v_max()
        # Scatter particles in a small ball around the new center, with
        # radius equal to v_max.  A ball, not a point: identical
        # positions would collapse the swarm immediately.
        if center is None:
            base = self.problem.random_point()
        else:
            base = np.asarray(center, dtype=float)
        for i in range(self.NP):
            offset = self._rng.uniform(-v_max, v_max, size=dim)
            x = base + offset
            x = self.problem.project(x)
            self._positions[i] = x
            self._velocities[i] = self._rng.uniform(-v_max, v_max, size=dim)
            self._pbest_x[i] = x
            self._pbest_result[i] = None
            self._emit_trial(x, i)
        self._gbest_idx = None
