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
the *swarm best the particle is allowed to see* — call it ``sbest_i``::

    v_i ← χ · (v_i + c1·r1·(pbest_i − x_i) + c2·r2·(sbest_i − x_i))
    x_i ← x_i + v_i

with ``r1, r2 ∼ U(0, 1)^d`` independent per-component random vectors and
``χ`` the constriction coefficient that guarantees convergence (Clerc &
Kennedy, 2002).  Default parameters use the canonical values
``φ = c1 + c2 = 4.1`` and ``χ = 2 / (φ − 2 + √(φ² − 4·φ)) ≈ 0.7298``,
which together with ``c1 = c2 = 2.05`` yield effective coefficients
``χ·c1 = χ·c2 ≈ 1.49618`` and an inertia weight ``w = χ ≈ 0.7298``.

Topology
~~~~~~~~

How ``sbest_i`` is determined depends on the *swarm topology*:

* ``"gbest"`` (default, Kennedy-Eberhart 1995) — the social neighbourhood
  is the entire swarm: every particle pulls toward the single global
  best.  Fast contraction once a basin is found, but premature
  convergence on multimodal problems where the first good basin is
  rarely the global one.
* ``"lbest"`` — *ring* topology: each particle ``i`` is connected only
  to its ``k_neighbors`` closest indices on a wrap-around ring, so it
  pulls toward the best ``pbest`` among ``{(i ± j) mod NP : j ≤ k}``.
  Information about a new best diffuses through the swarm only one hop
  per iteration, which is *slower* but lets multiple sub-swarms explore
  different basins in parallel.  Empirically beats ``gbest`` on
  multimodal problems (Kennedy & Mendes, 2002).

The two topologies are *complementary*: ``gbest`` excels on unimodal /
weakly-multimodal problems where rapid exploitation pays off, ``lbest``
on highly-multimodal landscapes where a swarm-wide attractor would lock
all particles into the first decent basin.  Panobbgo's structural
mutation catalog ships *both* variants so the self-improvement loop
can pick whichever helps on a given problem family.

Key differences from existing population heuristics:

* :class:`~panobbgo.heuristics.cma_es.CMAES` adapts a *covariance matrix*
  and re-samples from it; it has no per-individual memory.
* :class:`~panobbgo.heuristics.differential_evolution.DifferentialEvolution`
  generates trial vectors via *recombination* of three randomly-chosen
  population members (``DE/rand/1``); it has no momentum.
* PSO carries a *velocity* (momentum) per particle and uses *social*
  attraction toward the swarm or neighbourhood best; this gives it
  markedly different exploration dynamics — fast contraction once a
  basin is found while retaining inertia from the prior search
  direction.

The implementation runs **asynchronously** inside the panobbgo event loop,
following the same pattern as :class:`DifferentialEvolution`:

1. ``on_start()`` emits ``NP`` random initial positions, one per particle.
2. ``on_new_results()`` matches incoming results back to their particle
   index (via the ``who`` tag), updates ``pbest_i`` if the result improves,
   refreshes the global best (for reporting) plus the per-particle
   neighbourhood best (for ``lbest`` topology), and emits the particle's
   next position generated from the velocity update above.
3. ``on_restart(center, reason)`` resets all particles to a randomized
   ball around the new center, drops in-flight trials, and starts a fresh
   swarm (matching the "warm restart" behaviour of CMA-ES IPOP).

References
----------

* J. Kennedy & R. Eberhart (1995). "Particle Swarm Optimization."
  *Proceedings of ICNN'95.*
* M. Clerc & J. Kennedy (2002). "The Particle Swarm — Explosion, Stability,
  and Convergence in a Multidimensional Complex Space."
  *IEEE Transactions on Evolutionary Computation*, 6(1):58–73.
* J. Kennedy & R. Mendes (2002). "Population Structure and Particle Swarm
  Performance." *Proceedings of CEC 2002*, 1671–1676.
  Empirical study showing ``lbest`` beats ``gbest`` on multimodal
  benchmarks.
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

# Topologies controlling the *social* attraction term in the velocity
# update.  Kept as a tuple for runtime introspection (``topology in
# _TOPOLOGIES``).
_TOPOLOGIES: tuple = ("gbest", "lbest")


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
        c1: Cognitive (personal-best attraction) coefficient.
            Default ``1.49618`` — Clerc-Kennedy.
        c2: Social (global-best attraction) coefficient.
            Default ``1.49618`` — Clerc-Kennedy.
        v_max_frac: Maximum velocity per dimension as a fraction of the
            corresponding box range.  Velocities are clamped to
            ``[-v_max_frac · range, +v_max_frac · range]`` to prevent the
            swarm from exploding outside the search box.  Default
            ``0.5`` — a common conservative choice.
        topology: Social-attraction topology (see module docstring).
            ``"gbest"`` (default) pulls every particle toward the single
            global best.  ``"lbest"`` switches to a wrap-around *ring*
            of width ``2 · k_neighbors + 1`` and pulls toward the best
            ``pbest`` in that ring; this slows information diffusion and
            is empirically stronger on multimodal problems
            (Kennedy & Mendes, 2002).
        k_neighbors: Half-width of the ``"lbest"`` neighbourhood.
            Default ``2`` (neighbourhood size 5, including self) which
            matches the recommendation in Kennedy & Mendes 2002.  Must
            be a positive integer; ``k_neighbors >= NP // 2`` makes the
            neighbourhood span the whole swarm and degenerates to
            ``gbest``.  Ignored when ``topology == "gbest"``.
        seed: Optional seed for the per-instance RNG.  ``None`` (default)
            seeds from ``np.random.default_rng()``.
        name: Override the heuristic's display name.

    Notes:
        - The constructor validates all numeric arguments and raises
          :class:`ValueError` on bad inputs.
        - All particle bookkeeping (positions, velocities, personal bests,
          global best) lives in the heuristic instance — no shared global
          state, so multiple PSO heuristics in one strategy are
          independent.
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
        k_neighbors: int = 2,
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
        if topology not in _TOPOLOGIES:
            raise ValueError(f"PSO: topology must be one of {_TOPOLOGIES}, got {topology!r}")
        if not isinstance(k_neighbors, int):
            raise ValueError(f"PSO: k_neighbors must be an integer, got {k_neighbors!r}")
        if k_neighbors < 1:
            raise ValueError(f"PSO: k_neighbors must be >= 1, got {k_neighbors}")

        super().__init__(strategy, name=name or "PSO")
        self.NP: int = NP
        self.w: float = float(w)
        self.c1: float = float(c1)
        self.c2: float = float(c2)
        self.v_max_frac: float = float(v_max_frac)
        self.topology: str = topology
        self.k_neighbors: int = int(k_neighbors)
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

    def _ring_neighbors(self, particle_idx: int) -> List[int]:
        """Indices in the ring neighbourhood of ``particle_idx``.

        The wrap-around ring includes the particle itself plus
        :attr:`k_neighbors` indices on each side.  Ordered so the centre
        particle's index appears in the middle, but the order is not
        observed by callers — only the set matters.
        """
        return [(particle_idx + j) % self.NP for j in range(-self.k_neighbors, self.k_neighbors + 1)]

    def _social_best_idx(self, particle_idx: int) -> Optional[int]:
        """Return the index of the social attractor for ``particle_idx``.

        For ``topology == "gbest"`` this is just :attr:`_gbest_idx`.
        For ``topology == "lbest"`` it is the index of the best
        ``pbest`` among the wrap-around ring of width
        ``2·k_neighbors + 1`` centred on ``particle_idx`` — slower
        information diffusion than the gbest variant but stronger on
        multimodal problems (Kennedy & Mendes, 2002).

        Returns ``None`` if no neighbour has accumulated a personal
        best yet, in which case :meth:`_generate_next` falls back to a
        random point.
        """
        if self.topology == "gbest":
            return self._gbest_idx

        # lbest: scan the ring neighbourhood.
        handler = self.strategy.constraint_handler
        best_idx: Optional[int] = None
        best_result: Optional[Result] = None
        for j in self._ring_neighbors(particle_idx):
            pb = self._pbest_result[j]
            if pb is None:
                continue
            if best_result is None or handler.is_better(best_result, pb):
                best_result = pb
                best_idx = j
        return best_idx

    def _generate_next(self, particle_idx: int) -> None:
        """Produce the next candidate position for ``particle_idx``.

        Falls back to a fresh random point if we don't yet have a
        social attractor (e.g. all initial trials still pending, or
        ``lbest`` mode with the entire local neighbourhood empty).
        """
        if self._positions is None or self._velocities is None or self._pbest_x is None:
            return

        social_idx = self._social_best_idx(particle_idx)
        if social_idx is None or self._pbest_x[particle_idx] is None:
            # No memory to pull from yet — emit a fresh random point so
            # the particle stays active.
            x = self.problem.random_point()
            self._emit_trial(x, particle_idx)
            return

        dim = self.problem.dim
        x_i = self._positions[particle_idx]
        v_i = self._velocities[particle_idx]
        p_i = self._pbest_x[particle_idx]
        g = self._pbest_x[social_idx]

        r1 = self._rng.random(dim)
        r2 = self._rng.random(dim)
        new_v = self.w * v_i + self.c1 * r1 * (p_i - x_i) + self.c2 * r2 * (g - x_i)

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
