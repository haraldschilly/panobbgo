# Design: block allocation with an AOCC-shaped reward (2026-09-10)

Read-only analysis of the selection strategies, then a design for
`StrategyBlockBandit`.  Produced by an Opus subagent; citations verified
by the orchestrator (`ucb.py:126`, `phased.py:556`, `cma_es.py:300,335-343`,
`core.py:1616`).  Status: **design accepted, implementation in progress.**

## 0. The finding

`StrategyUCB` and `StrategyThompsonSampling` have textbook-correct score
arithmetic.  Everything *around* it measures the wrong thing:

1. **A pull is one point** — `ucb.py:126`, `thompson.py:165`,
   `contextual.py:224`.
2. **Reward fires only on a new global best**, in `on_new_best` —
   `ucb.py:56-79`, `thompson.py:79-99`.  Everything else scores 0.
3. **Counts are denominated in points** — `ucb.py:130`, `thompson.py:169`.
   A population arm emits λ points per generation of which at most one
   can be the new best, so its estimated value is **≤ 1/λ by
   construction**.

2 + 3 is the §6 `Center` pathology in UCB/TS form: a one-point arm that
is lucky once holds Q = 1.0 with n = 1 forever; CMA-ES with λ = 10 is
capped at Q ≤ 0.1.  Not a tuning problem — a measurement problem.  So:
**do not write a new bandit.  Write a block scheduler that owns pull
size, reward and denominator, with the bandit rule pluggable.**

Other defects found on the way:

* `StrategyRewarding credit="ema"` is not a bandit pull at all: every
  ready arm gets `max(1, round(target·p))` points on every pass
  (`rewarding.py:270-285`).  Probability matching sets proportions of an
  interleave; §9's starvation is built in.
* The reward `1 − exp(−Δ)` uses the **raw** objective delta.  It
  saturates at Δ ≳ 5 and vanishes at Δ ≲ 1e-3 — informative only in a
  narrow band, and dead exactly where AOCC accumulates its area (AOCC is
  a log-precision integral, `ioh_runner.py:86-90`).
* `last_best` is mutated inside the batch loop (`rewarding.py:184,188`)
  so the arm listed first collects the improvement.
* UCB/TS use all-time averages against strongly non-stationary arms and
  select "first ranked arm with a non-empty queue" (no `has_points`
  gate) — queue occupancy, not score, often decides.
* TS: β grows by λ per generation, α by ≤ 1 — a population arm's
  posterior collapses deterministically.
* `StrategyPhased` (`phased.py:138-145,177-228,556`): fixed percentage
  cutoffs, bandit stats **reset** at every boundary, and
  `points[:remaining]` **cuts a generation** at the phase edge.  CMA-ES
  ignores foreign results (`cma_es.py:300`) and its queue still holds
  the generation it emitted at `on_start`, so what §12 measured as a
  "warm start" was a *delayed cold start* from a stale generation.

## 1. Block sizing

    block_evals = max(round(max_eval / n_blocks), 2·dim)   # n_blocks = 50

**A block does not close until the owner's output queue is empty**
(`has_points is False`), hard cap 2·block_evals.  This is the
load-bearing rule: no generation is ever cut, expressed entirely through
the existing `has_points` (`core.py:756-763`).  `n_blocks = 50` keeps
the number of *decisions* constant across dimension and budget, and
coincides with the sync-mode loop batch `max_eval/50` (`core.py:1616`),
so block boundaries align with loop passes for free.  Prefer the budget
form over `c·dim`: on the standard battery `budget = 500·dim`, so the
two are the same curve.

## 2. Reward: AOCC area gained per evaluation, in decades

AOCC decomposes exactly over blocks, so maximising the per-block mean
drop in φ *is* maximising AOCC.  The optimizer does not know `f_opt`;
use the run-local anchored form

    r_b = clip( mean_{t∈block} log10( (best(t0) − a) / (best(t) − a) ) / D , 0, 1 )

`t0` = first evaluation of the block, `a` = run-local anchor set after
the first block as `best − (f_med − best)` and re-anchored downward if
`best ≤ a`, `D = 2.0` decades → full reward.

* **AOCC-shaped**: mean over the block, not endpoint — a block that
  drops early and flatlines beats one that drops the same amount at its
  last evaluation.  The anytime property none of the current rewards
  has.
* **Scale-free**: invariant under `f → c·f + k`.  The current reward is
  invariant under neither.
* **Per evaluation**, so a 90-point generation is judged on progress per
  evaluation, not hit rate.
* Ablations: `reward="endpoint"` (log-relative endpoint improvement,
  blind to *when*), `reward="rank"` (robust, discards magnitude).

## 3. Policy: discounted UCB

    close block for owner a:  n_a ← γ·n_a + 1;  S_a ← γ·S_a + r_b;  N ← γ·N + 1
    select:                   argmax_a  S_a/n_a + c·sqrt(log N / n_a)
    γ = 0.9 (half-life ≈ 7 blocks), c = 0.5

* Not plain UCB1: all-time averages cannot forget; arms here are
  non-stationary (a fresh restart is good again, a converged arm is 0).
* Not Thompson: rewards are continuous; a Beta-Bernoulli posterior has
  the wrong width (exactly `thompson.py:148-150`).  Gaussian TS is the
  second choice, worth an ablation.
* Not EMA probability matching: allocates in proportion, keeps a floor
  of `explore/n` on every arm forever; with ~50 decisions and a 0.58 vs
  0.42 gap, argmax-with-confidence is the right shape.
* Not successive halving as the whole policy: irreversible on one noisy
  block.  Right shape for the prologue only.
* `c = 0.5`, not √2: narrow reward band, ~50 decisions.

**Prologue**: one half-length block per arm, deterministic order (~5 %
of budget with 5 arms; §12 says 10 % on a design costs −0.38).  Early
exit when the leader's LCB exceeds every untried arm's optimistic prior.
**Dimension prior** (`prior="dim"`) seeded from §9/§14 per-dim means is
an *ablation*, not the default — it was fitted on the evaluation battery
and must be held out at d = 10/20.

**Tail**: `evals_remaining < 0.25·max_eval` → `c = 0`, argmax only.
**Hysteresis** throughout: switch from the incumbent only if the
challenger's Q̂ exceeds it by ×1.2 — every switch costs a transient (§12:
up to −0.13).

## 4. Pause, resume, warm start

A paused arm is one whose `get_points` is not called — state is instance
state, refill is event-driven, `active` keeps it registered.  **No
pause/resume API is needed.**  Constraints, all satisfied by the block
rule: never leave a partially drained generation (`has_points` gate);
paused arms keep receiving `on_new_results` and population arms filter
on their own `who` prefix (`cma_es.py:300`, `pso.py:698-707`), reactive
arms just top up their queue (RNG stream advances as a function of the
schedule — still a pure function of the seed under `sync_evaluation`,
document it).

Heuristic contract additions: an *optional* `warm_start(points)`, called
**once on re-acquisition after a gap, before the first `get_points` of
the block, only when the queue is empty**, with the top-k results by
penalty value from the *shared* store (all arms' evaluations — the
point of the shared archive).  **Default OFF**: §12 measured
NLSHADE_LBC → CMA-ES at −0.132, a foreign warm start can hurt.  Per-arm
opt-in decided by its own A/B.  (A separate design covers per-arm
warm-start semantics.)

## 5. Implementation: a new class

`StrategyBlockBandit` in `panobbgo/strategies/blocks.py`, ~150 lines,
exported from `panobbgo/strategies/__init__.py`.  Not a mode of
`StrategyPhased` (fixed cutoffs, per-phase arm sets, stat resets, five
`issubclass` re-implementations, the truncation bug — it would reuse
nothing and inherit the bug surface).  Not a `credit=` of
`StrategyRewarding` (would replace 100 % of `execute` and make the
existing credit A/B compare incomparable things).

    class StrategyBlockBandit(StrategyBase):
        def __init__(self, problem, *, n_blocks=50, block_evals=None,
                     policy="ducb", gamma=0.9, ucb_c=0.5, tail_frac=0.25,
                     decades=2.0, reward="area", prior="none",
                     warm_start_on_resume=False, hysteresis=1.2, **kw)
        # state: _owner, _last_owner, _block_t0, _phi0, _phi_sum, _phi_n,
        #        _anchor, _S, _n, _N, _prologue, _blocks_closed
        # add_heuristic, on_new_results (accumulate φ; anchor),
        # _block_reward, _close_block, _select, _open_block, execute,
        # _get_status_info

`execute()`: (1) if no owner, or consumed ≥ block_evals **and** not
`owner.has_points`, or consumed ≥ 2·block_evals → close, select, open;
(2) `owner.get_points(min(block_evals − consumed, target))`; (3) never
truncate; (4) if the chosen arm has no points, return `[]` — it refills
on the next result batch, `core.py:1637-1651`'s stall guard is the
backstop.  `policy="uniform"` (round-robin over blocks) must exist: it
isolates *blocking* from *learning*.

Tests (`sync_evaluation=True`, fixed seed, `testing_mode=True`; patterns
in `tests/test_strategy_rewarding_credit.py:33-50` and
`tests/test_reproducibility.py:19-49`):

1. blocks are contiguous (`who` constant in runs of ≥ block_evals)
2. a generation is never cut (`owner.has_points is False` at every
   ownership change, with a stub emitting batches of 7)
3. **reward is scale-free**: identical rewards for `fx` and `1000·fx+5`
4. reward is anytime: same total drop, early beats late
5. prologue covers every arm; tail is exploit-only
6. same seed → same trajectory
7. `warm_start` called once per re-acquisition, never with a non-empty
   queue, never when the flag is off
8. **single arm matches `StrategyRoundRobin`** exactly — pins that the
   scheduler adds nothing in the single-arm case, protecting
   `RoundRobin_CMAES`

## 6. Experiment plan

Step 0 — Phase A defaults land first (tuned arms, or every number
re-measures §9).  Step 1 — **oracle recompute on tuned arms** gates the
arm set: an arm winning 0 cells after tuning is dropped.  Step 2 —
screen on 3 seeds: `CMAES_tuned_alone` (the bar), `Blocks_uniform`
(blocking without learning — §12 done right), `Blocks_ducb`,
`Blocks_ducb_prior`, `Rewarding_ema` on tuned arms, block-size
sensitivity `n_blocks ∈ {25, 100}`.  No Random/Center/Nearby arms.
Gates: `Blocks_uniform` within −0.02 of the bar, else one strong arm is
the answer; `Blocks_ducb ≥ Blocks_uniform + 0.005`, else the reward
carries no signal.  Step 3 — 12-seed roster: paired CI excluding zero
on the positive side, ≥ 9/12 seeds, no dimension negative (else ship
dimension-gated).  **If the recomputed headroom is below ~+0.03, stop**:
a policy paying 5–10 % on a prologue cannot chase it.  Step 4 — d = 10
and 20 hold-out.  Step 5 — warm start as its own factor, off by default.
