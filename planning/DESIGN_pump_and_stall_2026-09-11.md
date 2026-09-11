# Design: the pump-thread arms and the wall-clock stall guard (2026-09-11)

Read-only analysis of the two scheduling defects the invariant sweep
found — F3 and F4 of `planning/results/2026-09-10/invariants_findings.md`
— then a design for both.  Citations verified against the working tree at
`99c1a0b` (`core.py:1691-1776`, `lbfgsb.py:403-488`, `cobyqa.py:314-354`,
`round_robin.py:33-47`, `blocks.py:791-814`).  Status: **design proposed,
not implemented.**

## 0. The finding

Both defects are the same category as §18: *the harness has been
measuring something other than what the spec says.*

**F3.** `LBFGSB`, `COBYQA` and `LocalPenaltySearch` produce points from a
daemon *pump thread* (`lbfgsb.py:417`, `cobyqa.py:323`,
`local_penalty_search.py:189`) that relays a subprocess's `f(x)` requests
into the output queue.  Their queue is therefore empty *most of the
time*.  `StrategyRoundRobin.execute` polls one arm per attempt and takes
the first non-empty answer (`round_robin.py:39-46`), so any competitor
with a stocked queue satisfies the pass before the pump is ever given a
turn.  Re-run today, verbatim from the findings:

    Random + LBFGSB, DeJong d=3, 150 evals, seed 42, sync
    -> Counter({'Random': 150})      LBFGSB: 0      15 loops, 0.2 s wall

0.2 s is the whole run.  The `spawn`-context subprocess
(`lbfgsb.py:300-325`) has not finished importing `scipy` by then, so the
arm is not merely starved — it never emitted a first point at all.  Every
portfolio number that carries an L-BFGS-B or COBYQA arm measured a
portfolio *without* it.

**F4.** The main loop's progress guard is denominated in **seconds**
(`core.py:1736-1759`), and so is the synchronous settle point
(`core.py:1722`: `wait_idle(timeout=self._max_stall_seconds)`).  A slow
*handler* — `GaussianProcessHeuristic.on_new_results` at ~0.4 s — makes
`wait_idle` time out; the next pass then sees every queue empty (the bus
has not delivered the refills yet), and two such passes end the run.
GP+Random at 300 evals stops between 70 and 240 evaluations depending on
machine load; COBYQA alone produces 0.  A seeded `sync_evaluation=True`
run is thus a function of machine speed, which contradicts both
`tests/test_reproducibility.py` and AGENTS.md "Local runs" (*"Measure
progress in evaluations, not wall time"*, `AGENTS.md:88-90`).

The two are one design error seen twice: **the scheduler treats "no
points right now" as "no points ever", and uses the clock to decide how
long "right now" lasts.**

---

## 1. F3 — the pump bridge

### 1.1 What the protocol actually is

Read end to end, the bridge is a *strict request/response* protocol with
exactly one outstanding evaluation, and `cap=1` says so
(`lbfgsb.py:258`, `cobyqa.py:223`):

| step | thread | code |
|---|---|---|
| worker calls `f(x)`, blocks on `recv` | subprocess | `lbfgsb.py:143-151` |
| pump wakes on `p1.poll(0.1)`, `emit(x)` | pump daemon | `lbfgsb.py:427-434` |
| strategy drains the queue | main loop | `round_robin.py:42` |
| strategy evaluates, publishes `new_results` | main loop | `core.py:1893-1907` |
| `on_new_results` sends the penalty back | **event-bus** | `lbfgsb.py:483-488` |
| worker resumes, next `f(x)` | subprocess | `lbfgsb.py:391` |

Four threads (main, bus dispatcher, pump, subprocess) take turns on a
round trip in which **only one of them is ever runnable**.  The pump adds
no concurrency whatsoever; it exists only because `on_start` may not
block the serial dispatcher (`core.py:965-976`).  That is the whole
defect: a thread was introduced to avoid blocking the bus, and it turned
a synchronous protocol into an asynchronous one that the scheduler
cannot see.

Three second-order consequences of the pump, all currently live:

* `Heuristic.active` returns `True` while *any* owned thread is alive
  (`core.py:885-886`).  The pump loops forever on `while not
  self._stopped`, so a bridge arm reads "active" even when its subprocess
  has died.  `_can_still_produce` (`core.py:2032-2034`) inherits this.
* Under `sync_evaluation` the strategy never populates `self.pending`
  (`core.py:1893-1907` writes only `finished` / `tasks_walltimes`), so
  `_can_still_produce` reduces to *exactly* that always-true pump check.
  The one liveness signal the core has is, under sync, a constant.
* The pipe's parent end is touched from two threads — `recv` on the pump
  (`lbfgsb.py:429`), `send` on the bus (`lbfgsb.py:488`).  It works only
  because the protocol never lets both happen at once.

### 1.2 The options

**(a) Bounded blocking `get_points` on a condition variable.**  Correct
in principle: the strategy waits for the arm it selected instead of
skipping it, on a CV signalled by the pump, never on the clock.  But it
keeps the pump, and therefore keeps the two-thread pipe, the
always-alive `active`, and a genuine deadlock edge — a strategy that
waits on an arm whose worker is waiting for the `fx` of a point the
strategy is still holding in its unreturned `points` list.  Guarding that
needs an explicit "is an evaluation of mine outstanding?" flag, which is
most of the work of (d) with the thread still there.

**(b) Pre-request / keep one point queued ahead.**  **Not viable, and it
should be said plainly.**  L-BFGS-B's next iterate is a function of
`f` at the current one; COBYQA's next interpolation point is a function
of the trust-region update.  A "speculative next point" would be a
fabrication, not a proposal, and feeding its evaluation back would
corrupt the solver's model.  The *one* legitimate batch inside this
family is the finite-difference gradient: `approx_grad=True`
(`lbfgsb.py:366`) makes scipy call `f` at `x + εe_i` for `i = 1..n`
sequentially, and those `n` points are independent.  Supplying `fprime`
that requests all `n` at once would cut a d-dimensional descent from
`n+1` round trips to 2 — a real throughput win, worth its own ticket,
**orthogonal to F3** (it raises the arm's points per round from 1 to
`n`; it does not make the scheduler see the arm at all).

**(c) Policy: bridge arms only inside `StrategyBlockBandit`, excluded
from RoundRobin.**  Rejected.  It is a documentation change dressed as a
fix: `StrategyRoundRobin` is what every *solo* spec in the harness runs
under (`RoundRobin_CMAES` etc.), and `blocks.py` is not immune either —
its readiness gate is `h.has_points or self._can_warm_start(h)`
(`blocks.py:761`) and a bridge arm takes no warm start, so it is
unselectable there too, and `_block_over` would close its block on the
first empty draw (`blocks.py:552`).  The rule would also have to be
enforced somewhere, and nothing enforces it today.

**(d) Drive the solver from the main thread.**  `scipy.optimize.minimize`
cannot yield — no callback hook can suspend a C/Fortran driver mid-step —
so the solver must stay in its own subprocess (or `greenlet`, a new
dependency for no gain, or a hand-written resumable trust-region, which
is a rewrite of COBYQA).  What *can* move to the main thread is the
**pipe**.  The subprocess stays exactly as it is; the pump thread goes
away; the strategy pulls.

### 1.3 Recommendation: (d) as a *pull* bridge

> A pump-thread heuristic becomes a **pull-based** heuristic: it produces
> its next point when, and only when, the strategy asks for it, on the
> strategy's own thread.

The state machine, all of it on the main loop's thread:

```
next_point(timeout):
    if self._fx_pending is not None:          # answer the outstanding f(x)
        self.p1.send(self._fx_pending); self._fx_pending = None
        self._outstanding = False
    if self._outstanding:
        return []            # we owe the worker a result we don't have yet
    while self.p1.poll(timeout):              # timeout = deadlock backstop only
        msg = self.p1.recv()
        if msg is _X0_REQUEST:                # lbfgsb warm-start sentinel
            self.p1.send(self._warm_start_x0()); continue
        self.emit(msg)                        # project + tag, core.py:823-845
        self._outstanding = True
        return self.get_points(1)
    return []                                 # worker gone or wedged
```

and `on_new_results` shrinks to a *store*, never a send:

```
def on_new_results(self, results):
    for r in results:
        if r.who == self.name:
            self._fx_pending = self.strategy.constraint_handler.get_penalty_value(r)
```

What this buys, against (a):

* **One thread owns the pipe.**  The `send`/`recv` split across the bus
  and pump threads (`lbfgsb.py:488` vs `429`) disappears; so does the
  undocumented thread-safety assumption on `multiprocessing.Connection`.
* **No deadlock edge.**  The `_outstanding` guard is *structural*: the
  arm returns `[]` — instantly, no wait — whenever it owes the worker a
  value.  The strategy can never block on an arm that is blocked on the
  strategy.
* **`active` becomes honest**: `self.lbfgsb is not None and
  self.lbfgsb.is_alive()` instead of "a daemon thread is looping"
  (`core.py:885`).
* **Determinism** — see §1.5.
* It deletes code: `_pump` (`lbfgsb.py:419-439`, `cobyqa.py:325-339`,
  `local_penalty_search.py:191-223`), `emit_reliable`
  (`local_penalty_search.py:157-175`), and the "excluded from the
  reproducible synchronous mode's guarantees" caveat in all three
  docstrings (`lbfgsb.py:412-415`, `cobyqa.py:319-321`,
  `local_penalty_search.py:182-185`).

The `timeout` is **not** a scheduling parameter.  A healthy worker
replies in microseconds once it has its `fx`; the only way `poll` expires
is a dead or wedged subprocess, which is an error, logged as one, after
which the arm marks itself stopped.  Default: `bridge_timeout = 60.0`.
The one place it is genuinely consumed is the *first* point of the run,
where a `spawn` subprocess has to import scipy (~0.5–1 s) — the case the
0.2 s probe above never reached.

### 1.4 The interface change on `Heuristic`

Three additions to `panobbgo/core.py`, all with defaults that leave every
existing heuristic byte-identical:

```python
class Heuristic(Module):
    #: True iff this heuristic produces on demand (a solver bridge)
    #: rather than reactively topping up its queue from event handlers.
    on_demand: bool = False

    def produce(self, limit=None, timeout=None) -> List[Point]:
        """The scheduler's point-acquisition call.  Default: drain."""
        return self.get_points(limit)

    @property
    def can_produce(self) -> bool:
        """``has_points``, or: an on-demand arm that is ready to be asked."""
        return self.has_points
```

Bridge classes override `produce` with §1.3's state machine and
`can_produce` with `self.has_points or (self._alive() and not
self._outstanding) or self._fx_pending is not None`.

Call-site changes — mechanical, one line each:

| file:line | now | becomes |
|---|---|---|
| `round_robin.py:42` | `hs[i].get_points(self.size)` | `hs[i].produce(self.size)` |
| `blocks.py:811` | `owner.get_points(self.size)` | `owner.produce(self.size)` |
| `blocks.py:761` | `h.has_points or …` | `h.can_produce or …` |
| `rewarding.py:272,284` | `h.has_points` / `get_points` | `can_produce` / `produce` |
| `ucb.py:126`, `thompson.py:165`, `contextual.py:224`, `phased.py:354,385,427,459,506` | `h.get_points(1)` | `h.produce(1)` |

`blocks.py:813` (`self._block_drained = not owner.has_points`) stays on
`has_points` deliberately: an on-demand arm is drained after every single
point, which is the truth — a bridge arm's "generation" is one point, so
it closes its block at `block_evals` and never at the hard cap.

`round_robin.py:33-47` additionally loses its `import time` and
`time.sleep(1e-3)` (`round_robin.py:34,46`): with `produce`, the retry
loop's only purpose — giving the pump its 0.1 s — is gone.  What remains
is the `max_attempts` walk over the arms, and the F1 fix (`len(hs) == 0`)
must land in the same edit.

### 1.5 Determinism

Under `sync_evaluation=True` the claim becomes provable rather than
disclaimed:

1. The strategy asks arm *A* for points on the main thread.
2. If *A* is a bridge, the answer is `[]` (an `fx` is owed) or exactly
   one point, which is a pure function of the `fx` values already sent
   and the worker seed (`lbfgsb.py:277`, `_worker_seed`).
3. The evaluation order is submission order (`core.py:1893-1901`).
4. `wait_idle()` (§2) drains the bus before the next pass.

No step consults the clock; no step depends on which thread ran first.
The sequence of points is a function of (seed, arm order, budget) alone —
the same statement `tests/test_reproducibility.py:42-49` makes for the
reactive arms, now extended to the bridges.  The residual wall-clock
dependence, `p1.poll(timeout)`, only decides whether the run *fails*, not
which points it evaluates.

Note what does **not** change: a bridge arm still contributes ~1 point
per round it is polled, against 10 for a queued competitor
(`round_robin.py:28`, `size=10`).  That is correct — the solver is
sequential and cannot spend a 10-point budget — and the fix's contract is
"contributes every round it owns", not "contributes half the budget".
The `size=1` configuration gives the 50/50 interleave when that is
wanted.

---

## 2. F4 — liveness, not seconds

### 2.1 The predicate

Replace "no progress for N seconds" with "**nothing can produce**".  The
whole state is already there:

```python
def _alive(self) -> bool:
    """Can this run still produce a point without outside help?"""
    if self.pending:                              # evaluations in flight (async)
        return True
    if self.eventbus.inflight > 0:                # a handler is queued or running
        return True
    return any(h.can_produce for h in self.heuristics)
```

* `self.pending` — `core.py:1916,1838`; empty by construction under sync
  (`core.py:1893-1907`), which is why it may not be the *only* term.
* `eventbus.inflight` — a new public read of `EventBus._inflight`
  (`core.py:989,1088,1103`) under `_cv`.  This is the term that fixes the
  GP case: a queued `on_new_results` *is* a heuristic about to refill.
* `h.can_produce` — §1.4; queued points, or an on-demand arm ready to be
  asked.  `self.heuristics` already filters on `h.active`
  (`core.py:1380-1382`).

`_alive()` is monotone in the right direction: it is `False` only when
the bus is drained, nothing is in flight, and every arm's queue is empty —
in which state nothing in the process can change any of those three.
Stopping is then not a heuristic judgement but a fact.

`_can_still_produce` (`core.py:2024-2034`) collapses into `_alive`; its
sole caller `_collect_points_safely` (`core.py:2066`) gets the better
answer for free, and its 20-attempt / `sleep(0.01)` retry loop
(`core.py:2043,2069-2072`) can be deleted outright — with `produce` there
is nothing to wait *for*.

### 2.2 The loop

```python
if sync:
    self.eventbus.wait_idle()                   # no timeout: evaluations, not seconds
    self.jobs_per_client = max(1, int(self.config.max_eval / 50.0))
...
if not self._alive():
    self.logger.info("no heuristic can produce a point; ending run at %d/%d evaluations",
                     len(self.results), self.config.max_eval)
    break
```

* `core.py:1722` loses its timeout.  A 0.4 s GP handler simply takes
  0.4 s; the run does not shorten by one evaluation.  This *is* the F4
  fix — the stall guard was the second-order symptom, the `wait_idle`
  timeout the first.
* Under sync, one dead pass is conclusive, so the break is immediate and
  is logged at INFO: an exhausted portfolio ending cleanly is the normal
  end of a `Center`-only run, not an incident (this is also the shape F1
  needs).
* Under async, require `k = 3` consecutive dead passes before breaking,
  to absorb the submit/complete window in `_run_threaded_evaluation`
  (`core.py:1910-1951`) where `pending` can be momentarily empty.
* `_loops_without_progress` / `_max_loops_without_progress`
  (`core.py:1683-1684`) and `_stall_started` (`core.py:1685`) are
  deleted.  `_max_total_loops` (`core.py:1689`) stays — it is a
  loop-count backstop, not a clock.

### 2.3 What stays on the wall clock

Exactly one thing, and it is an error path:

```python
self._deadlock_seconds = float(getattr(self.config, "deadlock_seconds", 600.0))
# ... in the loop, only while _alive() is True and len(self.results) has not moved:
self.logger.error(
    "deadlock backstop: %.0fs without a new result while the run still claims to be "
    "alive (pending=%d, bus=%d, ready=%s). This is a bug; report it.",
    ..., [h.name for h in self.heuristics if h.can_produce])
break
```

Rationale: a *correct* run can never trip it, because `_alive()` False is
what ends a finished run and `_alive()` True means something is running.
It fires only on a genuine wedge — a bridge subprocess killed by the OOM
killer with its pipe still open, a handler in an infinite loop.  600 s,
not 30: the threshold must be far outside any legitimate slow-arm regime,
or it reintroduces F4 at a larger constant.

`config.max_stall_seconds` (`config.py:285`) is renamed to
`deadlock_seconds` with the default 30.0 → 600.0 and a docstring that
says "backstop for a *bug*, not a scheduling parameter".  Keep
`max_stall_seconds` as a deprecated alias for one release — three
existing call sites set it (`tests/test_core.py:253`,
`tests/test_invariants.py:293`, the findings' probe helper) and external
YAML may too.

### 2.4 Existing tests that must change

| test | now | after |
|---|---|---|
| `tests/test_core.py:236-260` `test_stall_guard_aborts_starved_run` | sets `max_stall_seconds = 1.0`, asserts `elapsed < 20` | drop the config line; `Silent` emits once, then `_alive()` is False on the next pass — assert the run ends with `< 50` results **and** that it took ≤ a handful of loops (`strategy.loops`), which is the evaluation-denominated version of the same claim. Rename to `test_starved_run_ends_when_nothing_can_produce`. |
| `tests/test_invariants.py:248-293` (`STALL`, `cfg.max_stall_seconds`) | 3 s stall cap on every probe | delete the knob; the probes get *faster* (no arm is now waiting out a timeout) |
| `tests/test_invariants.py:796-799` `test_subprocess_bridge_contributes_next_to_a_competitor` | `xfail(strict=True)` | un-xfail; strengthen per §4.1 |
| `tests/test_invariants.py:802-815` `test_slow_heuristic_still_spends_its_budget` | `xfail(strict=True)`, COBYQA | un-xfail. `TOO_SLOW` (`tests/test_invariants.py:105-108`) loses its COBYQA entry; GP stays there for runtime, not correctness |
| `tests/test_termination.py` | untouched | untouched — it drives handlers directly and never enters `_run`; it is not a stall-guard test despite the filename. The new liveness tests (§4.3) belong here |

Strict xfails mean these flip in the *same* commit as the fix.

---

## 3. Interaction: yes, a slow arm now slows the run

With §1 and §2 in place, a `GaussianProcessHeuristic` that spends 0.4 s
per result batch makes a 300-evaluation run take ~2 minutes of wall time,
and **nothing truncates it**.  That is the correct behaviour and should
be stated as a contract, not apologised for:

* The unit of budget is the evaluation (`AGENTS.md:88-90`; every AOCC in
  the harness is an integral over evaluations, `ioh_runner.py:86-90`).
  A run that spends 240 of 300 evaluations has not "run faster" — it has
  produced an incomparable number.  Worse, it produces one *silently*:
  `aocc` pads a short trajectory to `budget` with its last value
  (`ioh_runner.py:83-85`), so a truncated run scores exactly as if the
  arm had flatlined for the missing 60 evaluations.  Nothing in the
  metric reports the truncation.
* The cost of the fix is wall time only, and it is paid by exactly the
  configurations that were previously producing wrong numbers.
* The benefit is that `same seed → same trajectory` becomes true across
  machines, which is what makes the 12-seed roster comparable to
  yesterday's.

**The one knob that bounds it — a warning, never a truncation:**

```python
config.max_seconds_per_point = 0.0      # 0 disables; a diagnostic threshold
```

Accounted per arm from `tasks_walltimes` (`core.py:1904`) plus the bus
time attributable to its handlers.  When the running mean for an arm
exceeds the threshold, log **once** per arm at WARNING —
`"GaussianProcessHeuristic: 0.42 s/evaluation; a 300-evaluation run will
take ~2 min"` — and surface it in `_get_status_info`
(`core.py:2010-2011`).  It never removes the arm, never shortens the run
and never enters any decision: a threshold that changes behaviour is F4
again with a nicer name.  The user's remedies are the honest ones: drop
the arm, lower `max_eval`, or accept the wall time.

---

## 4. Tests that pin the contract

### 4.1 A pump arm beside a competitor contributes every round it owns

`tests/test_invariants.py`, replacing the xfail:

```python
def test_on_demand_arm_contributes_next_to_a_competitor():
    _, fx, who = run([_make("Random"), _make("LBFGSB")], probe_problem(), max_eval=150)
    n = sum(1 for w in who if w.startswith("LBFGSB"))
    assert n >= 150 // (10 + 1) - 1          # >= one point per full round robin
```

plus the sharper `size=1` variant, where the contract is an exact
alternation and the count is `75 ± 1`.  Parametrise over `LBFGSB`,
`COBYQA` and `LocalPenaltySearch`, and add the same assertion under
`StrategyBlockBandit` with `policy="uniform"` (the `blocks.py:761`
readiness gate is a second, independent way to starve the arm).

### 4.2 The reproducibility test that cannot exist today

```python
class _Slow(Random):                      # any reactive arm will do
    def on_new_results(self, results):
        time.sleep(self.delay)            # burns bus-dispatcher time, nothing else
        super().on_new_results(results)

def test_trajectory_is_independent_of_handler_latency():
    fast = _run(seed=1234, delay=0.0)
    slow = _run(seed=1234, delay=0.05)    # ~15 s over 300 evals; use 60 evals
    np.testing.assert_array_equal(fast.x,  slow.x)
    np.testing.assert_array_equal(fast.fx, slow.fx)
    assert list(fast.who) == list(slow.who)
    assert len(fast.fx) == len(slow.fx) == 60
```

This is the test F4 makes possible: today the `delay=0.05` run stops
early and the lengths differ.  It belongs next to
`test_same_seed_same_trajectory` (`tests/test_reproducibility.py:42-49`)
and is the *machine-independence* half of the contract — the half the
existing back-to-back comparison structurally cannot see (findings F4,
consequence 2).  Keep it at 60 evaluations so it costs ~3 s.

### 4.3 The liveness predicate itself

Unit tests in `tests/test_termination.py`, which already has the
`MockStrategy` scaffold (`tests/test_termination.py:45-74`):

1. `_alive()` is `True` while a bridge arm owes nothing and its
   subprocess lives — with a stub `on_demand` heuristic whose
   `can_produce` is `True` and whose queue is empty.  (The *point* of the
   predicate: "empty queue" and "cannot produce" are different facts.)
2. `_alive()` is `True` while `eventbus.inflight > 0`, with a handler
   parked on a `threading.Event`.
3. `_alive()` is `False` exactly when the bus is idle, `pending` is
   empty and every arm's `can_produce` is `False` — and the loop breaks
   on the first such pass under sync.
4. `_alive()` stays `True` for a bridge arm that owes an `fx`, and its
   `produce()` returns `[]` immediately rather than blocking — the
   deadlock guard of §1.3, asserted with a wall-clock *upper* bound
   (a blocking bug shows up as a hang, so the assertion is `< 0.5 s`).

### 4.4 Regression

`test_spends_its_budget` (`tests/test_invariants.py`, 18 arms) gains
`LBFGSB`, `COBYQA`, `LocalPenaltySearch` and `GaussianProcessHeuristic`
— none of which can currently be in it.  And the single-arm equality
`StrategyBlockBandit ≡ StrategyRoundRobin` (design §5, test 8 of
`DESIGN_block_bandit_2026-09-10.md`) must be re-asserted with a bridge
arm as the single arm.

---

## 5. Effort and risk

| step | effort | risk |
|---|---|---|
| 1. `Heuristic.on_demand` / `produce` / `can_produce` + call sites | ~40 lines, 8 files | **low** — defaults are identities; every existing arm keeps `get_points` semantics |
| 2. `lbfgsb.py` pull bridge | ~60 lines changed, `_pump` deleted | **medium** — the `_X0_REQUEST` interleave (`lbfgsb.py:372-379,427-432`) and `on_restart`'s pipe re-creation (`lbfgsb.py:490-522`) both move onto the main thread; `_fx_pending` must be cleared on restart or the fresh worker is answered with the dead one's value |
| 3. `cobyqa.py` pull bridge | ~40 lines | **low** — strictly simpler than L-BFGS-B (no sentinel) |
| 4. `local_penalty_search.py` pull bridge | ~60 lines | **medium** — richer protocol (`eval`/`done`/`error`, `local_penalty_search.py:196-219`) and a `_waiting_for_eval` flag that becomes `_outstanding` |
| 5. `_alive()` + loop rewrite + `deadlock_seconds` | ~50 lines in `core.py`, 1 in `config.py` | **medium** — this is the file everything runs through |
| 6. tests §4 + the six existing tests of §2.4 | ~150 lines | low |

**Deadlock potential.** The one real hazard, and the reason to prefer
(d) over (a): a strategy must never wait on an arm that is waiting on the
strategy.  In (d) this is structural — `produce` returns `[]` the instant
`_outstanding` is set, with no wait primitive anywhere in the path.  The
only blocking call left is `p1.poll(timeout)`, and it is reached *only*
after the worker has been given everything it asked for, i.e. only when
the worker is runnable.  Test 4.3(4) pins it.

**The event-bus thread.** `on_new_results` becomes a pure assignment
(`self._fx_pending = val`) — no pipe I/O, no blocking — which strictly
*improves* the bus contract at `core.py:976` ("handlers must return
promptly").  One subtlety: `_fx_pending` is written on the bus thread and
read on the main thread.  Under sync, `wait_idle()` orders them (§2.2).
Under async it is a benign single-writer/single-reader hand-off of a
float; make it explicit with a one-element `queue.Queue` rather than
relying on the GIL, and the async case is safe too.

**Dask.** `config.evaluation_method == "dask"` (`core.py:1708-1711`,
`dask_evaluation.DaskEvaluators`) is untouched by §1 — `produce` is a
scheduler-side call and a bridge subprocess lives in the *client*
process, so the bridge and Dask are independent.  §2 needs one check:
`self.pending` must be populated by the Dask path for `_alive()` to be
correct there (`_DirectEvaluators.outstanding`, `core.py:1168-1171`,
reads it, so it should be — verify before landing, and add the async
`k = 3` margin regardless).  Bridge arms under Dask remain a bad idea for
a different reason (one sequential solver rate-limits a distributed pool),
but that is a throughput remark, not a correctness one.

**What this design does *not* claim.** It does not make bridge arms
*competitive* — an arm producing 1 point per round against a population
arm's 10 will still look weak under RoundRobin, and F6-style oracle
recomputes must be redone once the arms actually participate.  It makes
them *measured*.  Every portfolio number involving `LBFGSB`, `COBYQA` or
`LocalPenaltySearch` — and every GP or slow-arm number truncated by F4 —
has to be recomputed after this lands; that re-measurement, not the
diff, is the expensive part.

## 6. Order of work

1. §1.4 interface (no behaviour change) — lands green on its own.
2. §2 liveness + `deadlock_seconds`, with §2.4's test edits.  F4 flips
   here; `test_slow_heuristic_still_spends_its_budget` un-xfails.
3. `cobyqa.py` pull bridge (the simplest of the three) + §4.1 for it.
4. `lbfgsb.py`, then `local_penalty_search.py`.  F3 flips at step 4.
5. §4.2's machine-independence test, §4.4's regression additions.
6. Re-run the affected harness cells.  Nothing in `planning/results/`
   that carries a bridge arm or a slow arm survives this change.
