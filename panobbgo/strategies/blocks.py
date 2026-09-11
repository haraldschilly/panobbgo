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

    block_evals = max(round(max_eval / n_blocks), 2 * dim)        # n_blocks=…
    block_evals = max(2 * dim, 4 * lambda_ref)                    # block_evals="auto"

A block closes when it has spent ``block_evals`` evaluations **and** the
owner's output queue ran empty on the last draw (``has_points`` is
``False``), with a hard cap of ``2 * block_evals``.  That single rule is
what keeps a generation from ever being cut in half — the defect
``StrategyPhased`` has at its phase edges (``phased.py:556``).  ``n_blocks
= 50`` keeps the number of *decisions* constant across dimension and
budget and coincides with the synchronous main-loop batch
``max_eval / 50`` (``core.py:1616``).

The budget form is not the *right* parametrisation, though.  The
block-length screen (``planning/DISCOVERY_2026-09-09.md`` §26) found an
interior optimum at an **absolute ~20–25 evaluations**, the same figure at
every dimension on the 500·dim battery — i.e. a *generation count*, not a
budget fraction.  The reason is the warm start: ``warm_start_now`` clears
the arm's output queue and its in-flight trials, so every block boundary
throws away up to one generation of work.  Blocks shorter than a few
generations pay that toll too often; much longer ones stop being decisions.
``block_evals="auto"`` therefore sizes a block as **four reference
generations**, ``max(2*dim, 4*lambda_ref)``, where ``lambda_ref`` is the
largest generation size the arms report (``_lam`` on CMA-ES, ``NP_init`` /
the current NP on the DE family, ``NP`` on PSO) and 20 when none of them
does.  ``n_blocks`` remains the default until the screen says otherwise.

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
is **re-acquired after a gap** is warm-started from the shared archive.
Two hooks are honoured: an arm that implements ``warm_start(results)`` as a
method is handed this strategy's own top-k, while an arm that opted in
through its constructor (``warm_start="archive"`` on the L-SHADE family and
PSO) is triggered via :meth:`~panobbgo.core.Heuristic.warm_start_now` and
fetches its own, richer selection from the
:class:`~panobbgo.analyzers.archive.Archive` analyzer.  A foreign warm
start can hurt, so this is a per-arm opt-in decided by its own A/B, never a
default.

The trigger is the *gap*, not an empty queue.  A queued generation is stale
by construction on re-acquisition: under ``sync_evaluation`` the last batch
of a closing block is harvested during the *next* arm's block, so the
paused arm completes its generation and emits the following one while
paused — from its own old population, while the other arm went on
improving the archive.  Gating the hook on ``not has_points`` (as this
strategy did until 2026-09-11) therefore excluded exactly the
self-refilling population arms the hook exists for: a d=5 probe with
CMA-ES + L-SHADE saw ``has_points`` true at all 46 block opens and fired
the hook zero times.  So the queue is *cleared* before the hook runs and
the arm rebuilds it from the shared archive.

A warm start is never free — ``warm_start_now`` drops the arm's in-flight
generation — so two guards decide whether it is worth paying, and they
compose (both must pass):

``warm_start_only_if_better`` (**off** by default; available, opt-in)
    Fire only if the archive's best penalty value is *strictly better* than
    the best this arm has produced itself.  If the arm is the one that made
    the latest progress there is nothing to import: re-seeding it can only
    destroy its adaptation state.  It is the sharper of the two on paper —
    it catches the case where a foreign point exists but is worse — but it
    measured **slightly negative** on the 12-seed roster (−0.007,
    ``planning/DISCOVERY_2026-09-09.md`` §30), so it is off by default:
    apparently the cases it suppresses are worth re-seeding anyway.  Prefer
    ``warm_start_only_if_foreign`` as the shipped guard.

``warm_start_only_if_foreign`` (on by default, the shipped guard)
    Fire only if at least one of the top-k results was produced by another
    arm.  Cheaper and coarser; kept because it is the criterion §5 of the
    design names, and because it still guards the callable
    ``warm_start(results)`` path, which is handed exactly those top-k.

Set either to ``False`` to measure the unguarded behaviour.

Region hand-offs
----------------

:class:`~panobbgo.heuristics.meta.MetaAnalyst` may publish
``meta_region(arm=…, box=…)`` — "re-seed this arm from *that* part of the
box".  :meth:`~StrategyBlockBandit.on_meta_region` only *records* the
request; it is applied at the next :meth:`~StrategyBlockBandit._open_block`
for that arm, which sets the arm's ``warm_start_box`` and forces the warm
start.  Nothing is mutated from the bus thread: ``warm_start_now`` clears
the arm's output queue, which ``execute()`` may be draining at that moment
(``planning/DESIGN_meta_level_2026-09-10.md`` §2).

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
    :param block_evals: explicit block size, or ``"auto"`` for
        ``max(2*dim, 4*lambda_ref)`` — four reference generations, the
        parametrisation ``planning/DISCOVERY_2026-09-09.md`` §26 points at.
        Either form overrides ``n_blocks``.
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
    :param warm_start_on_resume: warm-start an arm that is re-acquired after
        a gap — via its ``warm_start(results)`` method if it has one, else
        via :meth:`~panobbgo.core.Heuristic.warm_start_now`.  Its stale
        queue is cleared first.
    :param warm_start_k: how many results a ``warm_start(results)`` offer
        carries, and the depth of the ``warm_start_only_if_foreign`` test
        (``warm_start_now`` arms query the ``Archive`` themselves).
    :param warm_start_only_if_foreign: skip the warm start when every one of
        the top-k results is the arm's own — re-seeding an arm from itself
        buys nothing and costs its in-flight generation.
    :param warm_start_only_if_better: skip the warm start unless the
        archive's best penalty value beats the arm's own best.  The sharper
        of the two guards on paper, but it measured slightly negative on the
        12-seed roster (-0.007, §30), so it is **off by default**; available
        as an opt-in, and it composes with the foreign guard.
    :param hysteresis: a challenger must beat the incumbent by this factor.
    """

    def __init__(
        self,
        problem,
        *,
        n_blocks: int = 50,
        block_evals: Optional[int | str] = None,
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
        warm_start_only_if_foreign: bool = True,
        warm_start_only_if_better: bool = False,
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
        self.warm_start_only_if_foreign: bool = bool(warm_start_only_if_foreign)
        self.warm_start_only_if_better: bool = bool(warm_start_only_if_better)
        self.hysteresis: float = float(hysteresis)

        #: per-arm discounted statistics, keyed by heuristic name
        self._S: Dict[str, float] = {}
        self._n: Dict[str, float] = {}
        self._N: float = 0.0

        #: block state
        if isinstance(block_evals, str) and block_evals != "auto":
            raise ValueError("block_evals must be an int, None or 'auto', got %r" % block_evals)
        #: ``"auto"`` until the arms exist; then an int (see :attr:`block_evals`)
        self._block_evals: Optional[int | str] = block_evals if block_evals else None
        if isinstance(self._block_evals, int):
            self._block_evals = int(self._block_evals)
        self._owner: Optional[str] = None
        self._last_owner: Optional[str] = None
        self._block_size: int = 0
        self._block_n: int = 0
        self._block_drained: bool = False
        self._block_is_prologue: bool = False
        self._block_warm_started: bool = False
        self._block_region: bool = False
        self._phi0: Optional[float] = None
        self._trace: List[float] = []
        self._blocks: List[Dict[str, Any]] = []
        self._blocks_closed: int = 0
        self._prologue: Optional[List[str]] = None
        self._prologue_pick: bool = False

        #: warm-start bookkeeping: arms that have owned a block at least
        #: once (only those can be *re*-acquired), and per-arm counters of
        #: how often the hook fired and how often it reported a seed used.
        self._acquired: set[str] = set()
        self._warm_started: Dict[str, int] = {}
        self._warm_used: Dict[str, int] = {}
        #: best penalty value each arm has produced *itself*, by ``who`` prefix
        self._arm_best: Dict[str, float] = {}

        #: pending region hand-offs, ``arm name -> box`` (see
        #: :meth:`on_meta_region`).  Recorded on the bus thread, *applied* on
        #: the main thread at the next :meth:`_open_block` for that arm.
        self._pending_region: Dict[str, Any] = {}
        #: the arm whose block open is currently carrying a region request
        self._region_forced: Optional[str] = None
        #: per-arm count of region hand-offs actually applied
        self._regions_applied: Dict[str, int] = {}

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
        """Evaluations per block, resolved lazily on first use.

        Both forms need state that does not exist at construction time: the
        budget is regularly assigned afterwards (``s.config.max_eval = ...``)
        and ``"auto"`` needs the arms, whose generation sizes are only known
        once they have handled ``on_start``.  First access is from
        :meth:`execute`, by which time both hold.
        """
        if self._block_evals == "auto":
            self._block_evals = max(2 * self.problem.dim, 4 * self._lambda_ref())
        elif self._block_evals is None:
            self._block_evals = max(int(round(self._max_eval() / max(1, self.n_blocks))), 2 * self.problem.dim)
        return int(self._block_evals)

    #: Attributes an arm may expose to report its generation size, best
    #: first.  CMA-ES sets ``_lam`` in ``on_start`` (and again after every
    #: IPOP restart); the L-SHADE family carries the LPSR-shrunk
    #: ``_NP_current`` alongside the constructor's ``NP_init``; PSO and the
    #: plain DE use ``NP``.
    GENERATION_ATTRS: Tuple[str, ...] = ("_lam", "_NP_current", "NP_init", "NP")

    #: Fallback generation size when no arm reports one — the ~20-evaluation
    #: block ``planning/DISCOVERY_2026-09-09.md`` §26 measured, over 4.
    LAMBDA_REF_DEFAULT: int = 5

    def _lambda_ref(self) -> int:
        """Largest generation size the arms report, else :attr:`LAMBDA_REF_DEFAULT`."""
        best = 0
        for h in self._heuristics.values():
            for attr in self.GENERATION_ATTRS:
                value = getattr(h, attr, None)
                if isinstance(value, (int, np.integer)) and not isinstance(value, bool) and value > 0:
                    best = max(best, int(value))
                    break  # first attribute that answers wins, per arm
        return best if best > 0 else self.LAMBDA_REF_DEFAULT

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
            self._credit_arm_best(phi, r)
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

    def _credit_arm_best(self, phi: float, r: Result) -> None:
        """Track the best penalty value each arm produced *itself*.

        Keyed by the arm name, which is the ``who`` prefix before the first
        ``":"`` — population arms tag their points ``"CMAES:g3:i0"``.  A
        ``who`` that matches no registered arm is ignored rather than
        creating a phantom entry.
        """
        who = str(getattr(r, "who", "") or "")
        name = who.split(":")[0]
        if name not in self._heuristics:
            return
        if phi < self._arm_best.get(name, float("inf")):
            self._arm_best[name] = phi

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
                "warm_start": self._block_warm_started,
                "region": self._block_region,
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
        self._block_region = self._apply_pending_region(h)
        self._block_warm_started = self._warm_start(h)
        # The box is a *one-shot* hand-off: whatever the warm start made of
        # it, the arm goes back to querying the whole archive afterwards.
        setattr(h, "warm_start_box", None)
        self._region_forced = None
        self._acquired.add(h.name)

    def _block_over(self, owner: Heuristic) -> bool:
        """Is the open block finished?

        ``_block_drained`` is the queue state *at the moment of the last
        draw*, not now: a reactive arm tops its queue up on every result
        batch, so its live ``has_points`` is almost always ``True`` and
        would pin every block to the hard cap.  What the rule has to
        protect is that we never walk away from points the arm had already
        queued — exactly what the recorded flag says.
        """
        if not owner.can_produce and not self._can_still_produce():
            return True  # starved arm, and nothing in flight will wake it
        if self._block_n >= 2 * self._block_size:
            return True  # hard cap
        return self._block_n >= self._block_size and self._block_drained

    # -- the meta level's region hand-off ----------------------------------

    def on_meta_region(self, arm: Any = None, box: Any = None, **_: Any) -> None:
        """Record a region request from
        :class:`~panobbgo.heuristics.meta.MetaAnalyst`; do **not** act on it.

        ``planning/DESIGN_meta_level_2026-09-10.md`` §2: this runs on the
        event-bus dispatcher thread while :meth:`execute` runs on the main
        loop's.  ``warm_start_now`` clears the arm's output queue, which
        ``execute`` may be draining at that instant, so the mutation is
        deferred to :meth:`_open_block` — where it also reuses the
        scheduler's existing, tested warm-start path verbatim.

        A second request for the same arm before the first was applied
        simply replaces it: the newer analysis is the better one.
        """
        if box is None or arm is None:
            return
        name = str(arm)
        if name not in self._heuristics:
            self.logger.debug("meta_region for unknown arm %r ignored" % name)
            return
        self._pending_region[name] = box

    def _apply_pending_region(self, h: Heuristic) -> bool:
        """Hand ``h`` the box it was assigned, if any.  Main thread only.

        Sets the arm's ``warm_start_box`` — consulted by ``CMAES``'s and the
        L-SHADE family's ``archive_seed`` calls — and forces
        :meth:`_should_warm_start`, so the arm is actually re-seeded rather
        than left to the gap rule.  The two ``warm_start_only_if_*`` guards
        still apply: a hand-off is a *suggestion*, not an override of the
        policy the spec chose.
        """
        box = self._pending_region.pop(h.name, None)
        if box is None:
            return False
        if not self._can_warm_start(h):
            self.logger.debug("meta_region for %s dropped: the arm takes no warm start" % h.name)
            return False
        setattr(h, "warm_start_box", box)
        self._region_forced = h.name
        self._regions_applied[h.name] = self._regions_applied.get(h.name, 0) + 1
        self.logger.debug("meta_region applied to %s" % h.name)
        return True

    def _can_warm_start(self, h: Heuristic) -> bool:
        """Does this arm accept a warm start?

        Two hooks are recognised.  An arm may implement ``warm_start(results)``
        as a *method*, in which case it is handed this strategy's own top-k.
        Or — the route the DE family and PSO take — it exposes ``warm_start``
        as a *mode string* set in its constructor and overrides
        :meth:`~panobbgo.core.Heuristic.warm_start_now`, which pulls its own
        seeds from the shared :class:`~panobbgo.analyzers.archive.Archive`.
        The base-class ``warm_start_now`` returns ``False`` for everything
        else, so the override check is what keeps an un-opted-in arm out of
        the ``ready`` list.
        """
        if not self.warm_start_on_resume:
            return False
        if callable(getattr(h, "warm_start", None)):
            return True
        return bool(getattr(h, "warm_start", None)) and type(h).warm_start_now is not Heuristic.warm_start_now

    def _should_warm_start(self, h: Heuristic) -> bool:
        """Once per block open, and only on a **re-acquisition after a gap**.

        Deliberately *not* gated on ``not h.has_points``.  A paused arm keeps
        receiving ``on_new_results``, so a population arm completes the
        generation whose results are harvested during the next arm's block
        and immediately emits the following one — its queue is full again by
        the time it gets a block back, and that queue was built from its own
        pre-gap population.  Stale by construction, so it is replaced, not
        deferred to.  (The 2026-09-11 screening run measured 0 hook calls in
        46 block opens with the ``has_points`` gate in place.)
        """
        if not self._can_warm_start(h):
            return False
        if self._region_forced == h.name:
            return True  # a region hand-off is exactly a reason to re-seed
        if h.name not in self._acquired:
            return False  # first acquisition: it has just cold-started
        if h.name != self._last_owner:
            return True  # re-acquisition after a gap: the queue is stale
        # Consecutive blocks are no gap and need no re-seed -- unless the arm
        # has nothing to give, in which case the hook is the only thing that
        # can refill it and withholding it spins the main loop into the
        # stall guard.
        return not h.has_points

    def _archive_beats(self, h: Heuristic) -> bool:
        """Does the shared archive hold a point better than ``h``'s own best?

        ``self._best_phi`` is the run-wide minimum, i.e. exactly the best the
        archive can offer; ``self._arm_best[h.name]`` is what this arm paid
        for itself.  An arm that has produced nothing yet has everything to
        gain, so the answer is ``True``.
        """
        own = self._arm_best.get(h.name)
        if own is None:
            return True
        return self._best_phi < own

    @staticmethod
    def _is_own(h: Heuristic, r: Result) -> bool:
        """Was ``r`` produced by ``h``?  ``who`` may carry a ``name:...`` suffix."""
        who = str(getattr(r, "who", "") or "")
        return who == h.name or who.startswith(h.name + ":")

    def _warm_start(self, h: Heuristic) -> bool:
        """Warm-start ``h`` for the block about to open.  ``True`` iff it fired.

        The stale queue is dropped **before** either hook runs: the
        ``warm_start(results)`` contract used to rely on an empty queue as a
        precondition, and the mode-string arms clear their own output anyway
        (:meth:`panobbgo.heuristics.lshade.LSHADE.warm_start_now`), so
        clearing here is idempotent and makes both paths behave alike.  It
        only happens once we know there is something to re-seed *from* — a
        cleared queue with nothing to replace it would throw a generation
        away for nothing.
        """
        if not self._should_warm_start(h):
            return False
        seeds = self._top_results()
        if not seeds:
            return False  # nothing in the archive yet: leave the arm alone
        if self.warm_start_only_if_foreign and all(self._is_own(h, r) for r in seeds):
            # The arm is the one making the progress; re-seeding it from its
            # own points is a no-op that still discards its live generation.
            self.logger.debug("warm start of %s skipped: top-%d are all its own" % (h.name, len(seeds)))
            return False
        if self.warm_start_only_if_better and not self._archive_beats(h):
            # Sharper than the foreignness test: a foreign point that is
            # *worse* than what this arm already has is not worth a
            # generation either (§26 — every warm start costs one).
            self.logger.debug("warm start of %s skipped: it already holds the best point" % h.name)
            return False

        h.clear_output()
        hook = getattr(h, "warm_start", None)
        if callable(hook):
            self.logger.debug("warm start of %s with %d results" % (h.name, len(seeds)))
            hook(seeds)
            used = True
        else:
            # The arm sources its own seeds from the shared ``Archive``
            # analyzer (which is bounded at K = 256 and supports box /
            # diversity / per-leaf selectors), so it gets a better pool
            # than this strategy's ``warm_start_k`` incumbents — see
            # :meth:`panobbgo.core.Heuristic.archive_seed`.
            used = bool(h.warm_start_now())
            self.logger.debug("warm start of %s from its own archive: %s" % (h.name, used))
        self._warm_started[h.name] = self._warm_started.get(h.name, 0) + 1
        if used:
            self._warm_used[h.name] = self._warm_used.get(h.name, 0) + 1
        return used

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
        # ``can_produce``, not ``has_points``: an on-demand arm (a solver
        # bridge) has an empty queue between round trips, so gating on the
        # queue alone makes it permanently unselectable.
        ready = [h for h in self.heuristics if h.can_produce or self._can_warm_start(h)]
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
        points = owner.produce(self.size)
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
