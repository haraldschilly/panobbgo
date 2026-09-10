# -*- coding: utf8 -*-
# Copyright 2026 Harald Schilly <harald.schilly@gmail.com>
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

r"""
Block bandit
============

A *block scheduler* that owns the three things the existing bandit
strategies get wrong (``planning/DESIGN_block_bandit_2026-09-10.md`` §0):
the size of a pull, the reward, and the denominator the reward is divided
by.  The bandit rule itself is pluggable and deliberately boring.

``StrategyUCB`` and ``StrategyThompsonSampling`` treat *one point* as one
pull and only pay out ``on_new_best``.  A population heuristic that emits
λ points per generation can win at most one of those λ, so its estimated
value is bounded by ``1/λ`` by construction — a measurement problem, not
a tuning problem.  Here a pull is a **block** of roughly
``max_eval / n_blocks`` evaluations handed to a single arm, and the reward
is the AOCC area that arm bought with them.

Blocks
------

    block_evals = max(round(max_eval / n_blocks), 2 * dim)

A block closes when it has spent ``block_evals`` evaluations **and** the
owner's output queue ran empty on the last draw (``has_points`` is
``False``), with a hard cap of ``2 * block_evals``.  That single rule is
what keeps a generation from ever being cut in half — the defect
``StrategyPhased`` has at its phase edges (``phased.py:556``).  ``n_blocks
= 50`` keeps the number of *decisions* constant across dimension and
budget and coincides with the synchronous main-loop batch
``max_eval / 50`` (``core.py:1616``).

Reward: AOCC area per evaluation, in decades
--------------------------------------------

AOCC decomposes exactly over blocks, so maximising the per-block mean
drop of the log-precision *is* maximising AOCC.  The optimizer does not
know ``f_opt``, so we use a run-local anchor ``a``::

    r_b = clip( mean_{t in block} log10( (best(t0) - a) / (best(t) - a) ) / D, 0, 1 )

with ``t0`` the first evaluation of the block and ``D = decades``.  This
reward is

* **anytime** — the mean over the block, not the endpoint, so a block that
  drops early and then flatlines beats one that drops the same amount on
  its very last evaluation;
* **scale-free** — invariant under ``f -> c*f + k`` for ``c > 0``, because
  the anchor is built from the same observations (``reward="endpoint"``
  keeps the invariance but drops the anytime property; it exists as an
  ablation);
* **per evaluation**, so a 90-point generation is judged on progress per
  evaluation rather than on hit rate.

The anchor is set **when the first block closes**, as ``best - (f_med -
best)`` over every penalty value observed so far, and re-anchored with the
same spread rule whenever the running best reaches it.  The first block
therefore also gets an area reward — computed at its close, with the
anchor its own data defined.  (The alternative, giving the prologue blocks
``reward="endpoint"`` until an anchor exists, needs a second reward scale
and buys nothing: the block is scored once either way.)

Policy
------

``policy="ducb"`` (default) is discounted UCB — all-time averages cannot
forget, and these arms are strongly non-stationary::

    close block for owner a:  n_a <- g*n_a + 1;  S_a <- g*S_a + r_b;  N <- g*N + 1
    select:                   argmax_a  S_a/n_a + c*sqrt(log N / n_a)

A **prologue** of one half-length block per arm, in registration order,
gives every arm one measurement.  In the **tail** (``evals_remaining <
tail_frac * max_eval``) the exploration term is switched off.  A
**hysteresis** factor keeps the incumbent unless a challenger beats it by
``hysteresis`` times — every switch costs a transient.

``policy="uniform"`` is round-robin over blocks and learns nothing; it
isolates the effect of *blocking* from the effect of *learning*.

Pause and resume
----------------

A paused arm is simply one whose ``get_points`` is not called: its state
is instance state, its refill is event-driven, and ``active`` keeps it
registered.  No pause/resume API is needed.  Paused arms keep receiving
``on_new_results`` — population arms filter on their own ``who`` prefix,
reactive ones just top their queue up.  Their RNG streams advance as a
function of the schedule, which under ``sync_evaluation`` is still a pure
function of the seed.

Optionally (``warm_start_on_resume=True``, **off by default**) an arm that
is re-acquired with an *empty* queue is offered the best results from the
shared archive through a ``warm_start(results)`` hook, if it implements
one.  A foreign warm start can hurt, so this is a per-arm opt-in decided
by its own A/B, never a default.

.. codeauthor:: Harald Schilly <harald.schilly@gmail.com>
"""

from __future__ import annotations

import heapq
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from panobbgo.core import Heuristic, StrategyBase
from panobbgo.lib import Result


class StrategyBlockBandit(StrategyBase):
    r"""
    Hand a whole *block* of evaluations to one heuristic, then score that
    block by the AOCC area it bought and let a discounted UCB rule pick
    the owner of the next one.

    :param n_blocks: number of blocks the budget is cut into (the number of
        *decisions*, held constant across dimension and budget).
    :param block_evals: explicit block size; overrides ``n_blocks``.
    :param size: points requested per main-loop pass, exactly
        :class:`~panobbgo.strategies.round_robin.StrategyRoundRobin`'s
        ``size``.  With a single arm this makes the two strategies emit the
        identical evaluation sequence.
    :param policy: ``"ducb"`` (discounted UCB) or ``"uniform"``
        (round-robin over blocks, no learning).
    :param gamma: discount of the per-arm statistics at each block close.
    :param ucb_c: exploration weight; ``0`` in the tail.
    :param tail_frac: fraction of the budget that is exploit-only.
    :param decades: log-precision decades that earn the full reward.
    :param reward: ``"area"`` (anytime, default) or ``"endpoint"``.
    :param prior: ``"none"``; ``"dim"`` (per-dimension priors from the
        evaluation battery) is not implemented yet.
    :param warm_start_on_resume: offer re-acquired, empty arms the best
        results so far via their optional ``warm_start`` hook.
    :param warm_start_k: how many results such an offer carries.
    :param hysteresis: a challenger must beat the incumbent by this factor.
    """

    def __init__(
        self,
        problem,
        *,
        n_blocks: int = 50,
        block_evals: Optional[int] = None,
        size: int = 10,
        policy: str = "ducb",
        gamma: float = 0.9,
        ucb_c: float = 0.5,
        tail_frac: float = 0.25,
        decades: float = 2.0,
        reward: str = "area",
        prior: str = "none",
        warm_start_on_resume: bool = False,
        warm_start_k: int = 10,
        hysteresis: float = 1.2,
        **kwargs: Any,
    ) -> None:
        if policy not in ("ducb", "uniform"):
            raise ValueError("policy must be 'ducb' or 'uniform', got %r" % policy)
        if reward not in ("area", "endpoint"):
            raise ValueError("reward must be 'area' or 'endpoint', got %r" % reward)
        if prior == "dim":
            # §3: seeded from the per-dimension means of the evaluation
            # battery.  It is an ablation that has to be held out at
            # d = 10/20 before it can be trusted, so it stays unbuilt.
            raise NotImplementedError("prior='dim' is not implemented yet (design §3)")
        if prior != "none":
            raise ValueError("prior must be 'none' or 'dim', got %r" % prior)

        self.n_blocks: int = int(n_blocks)
        self.size: int = int(size)
        self.policy: str = str(policy)
        self.gamma: float = float(gamma)
        self.ucb_c: float = float(ucb_c)
        self.tail_frac: float = float(tail_frac)
        self.decades: float = float(decades)
        self.reward_kind: str = str(reward)
        self.prior: str = str(prior)
        self.warm_start_on_resume: bool = bool(warm_start_on_resume)
        self.warm_start_k: int = int(warm_start_k)
        self.hysteresis: float = float(hysteresis)

        #: per-arm discounted statistics, keyed by heuristic name
        self._S: Dict[str, float] = {}
        self._n: Dict[str, float] = {}
        self._N: float = 0.0

        #: block state
        self._block_evals: Optional[int] = int(block_evals) if block_evals else None
        self._owner: Optional[str] = None
        self._last_owner: Optional[str] = None
        self._block_size: int = 0
        self._block_n: int = 0
        self._block_drained: bool = False
        self._block_is_prologue: bool = False
        self._phi0: Optional[float] = None
        self._trace: List[float] = []
        self._blocks: List[Dict[str, Any]] = []
        self._blocks_closed: int = 0
        self._prologue: Optional[List[str]] = None
        self._prologue_pick: bool = False

        #: run-local objective bookkeeping (penalty values, "phi")
        self._anchor: Optional[float] = None
        self._best_phi: float = float("inf")
        self._phis: List[float] = []
        self._top: List[Tuple[float, int, Result]] = []
        self._top_seq: int = 0

        StrategyBase.__init__(self, problem, **kwargs)

    # -- configuration -----------------------------------------------------

    @property
    def block_evals(self) -> int:
        """Evaluations per block, resolved lazily against ``config.max_eval``.

        The budget is regularly assigned *after* construction
        (``s.config.max_eval = ...``), so this cannot be computed in
        :meth:`__init__`.
        """
        if self._block_evals is None:
            self._block_evals = max(int(round(self._max_eval() / max(1, self.n_blocks))), 2 * self.problem.dim)
        return self._block_evals

    def _max_eval(self) -> int:
        try:
            return int(self.config.max_eval)
        except (TypeError, ValueError):
            return 1000

    def add_heuristic(self, h: Heuristic) -> None:
        StrategyBase.add_heuristic(self, h)
        self._S.setdefault(h.name, 0.0)
        self._n.setdefault(h.name, 0.0)

    # -- accounting: every result feeds the block's log-precision trace ----

    def on_new_results(self, results: List[Result]) -> None:
        """Accumulate the running best of the *open* block, one entry per result.

        Results arrive in batches and, under ``sync_evaluation``, a batch is
        fully handled before the next :meth:`execute`, so the trace is a
        deterministic function of the schedule.  The log terms themselves
        are only formed at :meth:`_close_block`: the anchor they need does
        not exist until the first block ends.
        """
        for r in results:
            phi = self._penalty(r)
            if phi is None:
                continue  # inf / nan / failed evaluation: no information
            if self._owner is not None and self._phi0 is None:
                # first evaluation of the very first block: nothing was
                # known before it, so it *is* best(t0).
                self._phi0 = phi
            self._phis.append(phi)
            if phi < self._best_phi:
                self._best_phi = phi
            self._remember_top(phi, r)
            self._reanchor_if_needed()
            if self._owner is not None:
                self._trace.append(self._best_phi)

    def _penalty(self, r: Result) -> Optional[float]:
        """The scalar the constraint handler minimises, or ``None`` if unusable."""
        try:
            phi = float(self.constraint_handler.get_penalty_value(r))
        except (TypeError, ValueError):
            return None
        return phi if np.isfinite(phi) else None

    def _spread_anchor(self) -> float:
        """``best - (f_med - best)``: one median-spread below the incumbent.

        Affine-equivariant in the objective (``f -> c*f + k`` moves best and
        median the same way), which is what makes the reward scale-free.
        The fallbacks only fire on degenerate data, where the reward is 0
        regardless of the anchor.
        """
        best = self._best_phi
        spread = float(np.median(self._phis)) - best
        if spread <= 0.0:
            spread = max(self._phis) - best
        if spread <= 0.0:
            spread = 1.0
        return best - spread

    def _reanchor_if_needed(self) -> None:
        if self._anchor is not None and self._best_phi <= self._anchor:
            self._anchor = self._spread_anchor()

    def _remember_top(self, phi: float, r: Result) -> None:
        """Keep the ``warm_start_k`` best results of the shared archive."""
        if not self.warm_start_on_resume or self.warm_start_k <= 0:
            return
        self._top_seq += 1
        item = (-phi, self._top_seq, r)
        if len(self._top) < self.warm_start_k:
            heapq.heappush(self._top, item)
        elif item > self._top[0]:
            heapq.heapreplace(self._top, item)

    def _top_results(self) -> List[Result]:
        return [r for _, _, r in sorted(self._top, key=lambda t: -t[0])]

    # -- reward ------------------------------------------------------------

    def _block_reward(self) -> float:
        """Area under the log-precision curve of the open block, in ``[0, 1]``."""
        a, phi0 = self._anchor, self._phi0
        if a is None or phi0 is None or not self._trace:
            return 0.0
        d0 = phi0 - a
        if not np.isfinite(d0) or d0 <= 0.0:
            return 0.0
        tiny = float(np.finfo(float).tiny)
        trace = self._trace[-1:] if self.reward_kind == "endpoint" else self._trace
        gains = [np.log10(d0 / max(phi - a, tiny)) for phi in trace]
        return float(np.clip(float(np.mean(gains)) / self.decades, 0.0, 1.0))

    # -- the block life cycle ----------------------------------------------

    def _close_block(self) -> None:
        owner = self._owner
        if owner is None:
            return
        if self._anchor is None and self._phis:
            # The anchor is defined by the first block's own data, so that
            # block is scored on the same scale as every later one.
            self._anchor = self._spread_anchor()
        reward = self._block_reward()

        g = self.gamma
        self._n[owner] = g * self._n.get(owner, 0.0) + 1.0
        self._S[owner] = g * self._S.get(owner, 0.0) + reward
        self._N = g * self._N + 1.0

        self._blocks.append(
            {
                "owner": owner,
                "evals": self._block_n,
                "size": self._block_size,
                "drained": self._block_drained,
                "prologue": self._block_is_prologue,
                "reward": reward,
            }
        )
        self._blocks_closed += 1
        self._last_owner = owner
        self._owner = None
        self.logger.debug(
            "block %d closed: %s spent %d/%d evals, r=%.4f (Q=%.4f)"
            % (self._blocks_closed, owner, self._block_n, self._block_size, reward, self._q(owner))
        )

    def _open_block(self, h: Heuristic) -> None:
        self._owner = h.name
        self._block_n = 0
        self._block_drained = False
        self._block_is_prologue = self._prologue_pick
        self._block_size = max(1, self.block_evals // 2) if self._prologue_pick else self.block_evals
        self._trace = []
        self._phi0 = self._best_phi if np.isfinite(self._best_phi) else None
        if self._should_warm_start(h):
            # contract: once per (re-)acquisition, before the first
            # get_points of the block, and only on an empty queue.
            self.logger.debug("warm start of %s with %d results" % (h.name, len(self._top)))
            h.warm_start(self._top_results())  # pyright: ignore[reportAttributeAccessIssue]

    def _block_over(self, owner: Heuristic) -> bool:
        """Is the open block finished?

        ``_block_drained`` is the queue state *at the moment of the last
        draw*, not now: a reactive arm tops its queue up on every result
        batch, so its live ``has_points`` is almost always ``True`` and
        would pin every block to the hard cap.  What the rule has to
        protect is that we never walk away from points the arm had already
        queued — exactly what the recorded flag says.
        """
        if not owner.has_points and not self._can_still_produce():
            return True  # starved arm, and nothing in flight will wake it
        if self._block_n >= 2 * self._block_size:
            return True  # hard cap
        return self._block_n >= self._block_size and self._block_drained

    def _can_warm_start(self, h: Heuristic) -> bool:
        return self.warm_start_on_resume and callable(getattr(h, "warm_start", None))

    def _should_warm_start(self, h: Heuristic) -> bool:
        return self._can_warm_start(h) and not h.has_points

    # -- selection ---------------------------------------------------------

    def _q(self, name: str) -> float:
        n = self._n.get(name, 0.0)
        return self._S.get(name, 0.0) / n if n > 0.0 else 0.0

    def _exploration_c(self) -> float:
        """``ucb_c``, or 0 once the tail of the budget is reached (design §3)."""
        max_eval = self._max_eval()
        remaining = max_eval - len(self.results)
        return 0.0 if remaining < self.tail_frac * max_eval else self.ucb_c

    def _score(self, name: str, c: float) -> float:
        n = self._n.get(name, 0.0)
        if n <= 0.0:
            return float("inf")  # never measured: take it before anything else
        q = self._S.get(name, 0.0) / n
        if c <= 0.0:
            return q
        return q + c * float(np.sqrt(np.log(max(self._N, 1.0)) / n))

    def _arm_prior(self, name: str) -> float:
        """Optimistic prior of an arm that has not been measured yet."""
        return 1.0  # prior="none"; prior="dim" is refused in __init__

    def _maybe_end_prologue(self) -> None:
        """Early exit: the leader's LCB already beats every untried arm's prior."""
        if not self._prologue:
            return
        c = self.ucb_c
        lcb = -float("inf")
        for name, n in self._n.items():
            if n > 0.0:
                lcb = max(lcb, self._S[name] / n - c * float(np.sqrt(np.log(max(self._N, 1.0)) / n)))
        if lcb > max(self._arm_prior(name) for name in self._prologue):
            self.logger.debug("prologue cut short, leader LCB %.4f" % lcb)
            self._prologue = []

    def _select(self) -> Optional[Heuristic]:
        """Owner of the next block, or ``None`` if no arm can produce."""
        self._prologue_pick = False
        # An arm can only own a block if it can hand out points -- or if a
        # warm start is what it is waiting for.  (Design §5 lets a scoreless
        # arm be picked and returns ``[]``; under sync evaluation that stops
        # the result flow the arm needs to refill, so the queue gate stays.)
        ready = [h for h in self.heuristics if h.has_points or self._can_warm_start(h)]
        if not ready:
            return None

        self._maybe_end_prologue()
        if self._prologue:
            for name in list(self._prologue):
                for h in ready:
                    if h.name == name:
                        self._prologue.remove(name)
                        self._prologue_pick = True
                        return h

        if self.policy == "uniform":
            names = [h.name for h in ready]
            i = (names.index(self._last_owner) + 1) % len(names) if self._last_owner in names else 0
            return ready[i]

        c = self._exploration_c()
        scores = {h.name: self._score(h.name, c) for h in ready}
        leader = max(ready, key=lambda h: scores[h.name])
        incumbent = next((h for h in ready if h.name == self._last_owner), None)
        if incumbent is not None and incumbent is not leader:
            # every switch costs a transient (design §12: up to -0.13)
            if scores[leader.name] < self.hysteresis * scores[incumbent.name]:
                return incumbent
        return leader

    # -- main loop ---------------------------------------------------------

    def execute(self) -> List[Any]:
        if self._prologue is None:
            self._prologue = [h.name for h in self.heuristics]

        owner = self._heuristics.get(self._owner) if self._owner is not None else None
        if owner is not None and not owner.active:
            owner = None
        if owner is None or self._block_over(owner):
            self._close_block()
            owner = self._select()
            if owner is None:
                return []  # nothing can produce; core.py's stall guard is the backstop
            self._open_block(owner)

        # A fixed request size, exactly StrategyRoundRobin's: the block
        # length is enforced by *when* the owner changes, not by truncating
        # a draw.  Capping the last draw of a block at ``block_evals -
        # spent`` would both cut generations and make the single-arm case
        # differ from StrategyRoundRobin for no gain -- the overshoot is at
        # most ``size - 1`` evaluations per block.
        points = owner.get_points(self.size)
        self._block_n += len(points)
        self._block_drained = not owner.has_points
        return points

    def _get_status_info(self) -> Dict[str, str]:
        """Return strategy-specific status info."""
        info: Dict[str, str] = {"blocks": "%d x %d" % (self._blocks_closed, self._block_size)}
        if self._owner is not None:
            info["arm"] = "%s (%d/%d)" % (self._owner, self._block_n, self._block_size)
        if self._prologue:
            info["prologue"] = ",".join(self._prologue)
        measured = [name for name, n in self._n.items() if n > 0.0]
        if measured:
            best = max(measured, key=self._q)
            info["best_arm"] = "%s (Q=%.3f)" % (best, self._q(best))
        return info
