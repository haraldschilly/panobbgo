# -*- coding: utf8 -*-
"""The pull bridge, the liveness predicate, and what they make provable.

Pins the two contracts of ``planning/DESIGN_pump_and_stall_2026-09-11.md``:

**F3** — a solver bridge (:class:`~panobbgo.heuristics.lbfgsb.LBFGSB`,
:class:`~panobbgo.heuristics.cobyqa.COBYQA`) runs a sequential SciPy optimizer
in a subprocess, one evaluation at a time.  Until 2026-09 it handed its points
over from a daemon pump thread, so its output queue was empty whenever a
scheduler looked: beside any competitor with a stocked queue it contributed
**zero** points, and every portfolio measurement that carried such an arm had
measured a portfolio without it.

**F4** — the main loop's progress guard was denominated in *seconds*, so a slow
handler (a GP fit at ~0.4 s) truncated the run at whatever evaluation count the
machine's speed happened to produce.  Progress is counted in evaluations
(``AGENTS.md`` "Local runs"), so the trajectory of a seeded run must not depend
on how long anything takes.
"""

from __future__ import annotations

import collections
import time

import numpy as np
import pytest

from panobbgo.heuristics import COBYQA, LBFGSB, Nearby, Random
from panobbgo.heuristics.local_penalty_search import LocalPenaltySearch
from panobbgo.lib.classic import DeJong, Rosenbrock, RosenbrockConstraint
from panobbgo.strategies import StrategyRoundRobin

#: One :class:`~panobbgo.strategies.round_robin.StrategyRoundRobin` request.
SIZE = 10


def run(
    factories,
    problem=None,
    seed=42,
    max_eval=150,
    size=SIZE,
    strategy_cls=StrategyRoundRobin,
    stall=None,
):
    """A seeded, synchronous run; returns ``(fx, who, strategy)``."""
    s = strategy_cls(problem or DeJong(dims=3), parse_args=False, testing_mode=True, seed=seed, size=size)
    s.config.max_eval = max_eval
    s.config.sync_evaluation = True
    s.config.stop_on_convergence = False
    s.config.ui_show = False
    s.config.evaluation_method = "threaded"
    if stall is not None:
        # The deprecated wall-clock knob.  Setting it must no longer change
        # anything: that is half of what these tests assert.
        s.config.max_stall_seconds = stall
    for f in factories:
        s.add_heuristic(f(s))
    s.start()
    df = s.results.results
    if df is None or len(df) == 0:
        return np.zeros(0), (), s
    return (
        df["fx"].to_numpy(dtype=float).ravel(),
        tuple(str(w) for w in df["who"].to_numpy().ravel()),
        s,
    )


def counts(who):
    return collections.Counter(w.split(":")[0] for w in who)


# ---------------------------------------------------------------------------
# F3: the bridge arms contribute
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [LBFGSB, COBYQA])
def test_bridge_arm_contributes_next_to_a_competitor(cls):
    """Every round the arm owns yields exactly one point — not zero.

    A sequential solver cannot fill a 10-point request, so the contract is
    "one point per poll", not "half the budget".  Before the pull bridge this
    count was **0**: ``Random`` always had 20 points queued and satisfied the
    pass before the pump thread was ever given its 0.1 s.
    """
    _, who, s = run([lambda st: Random(st), lambda st: cls(st)], max_eval=150)
    c = counts(who)
    assert c[cls.__name__] > 0, f"{cls.__name__} was starved again: {c}"
    # Round robin alternates, so the arm owns about half the passes and each
    # ownership is worth one point.
    assert c[cls.__name__] >= s.loops // 2 - 1, (c, s.loops)


def test_bridge_arm_and_random_alternate_at_size_one():
    """With ``size=1`` the interleave is exact: one each, round after round."""
    _, who, _ = run([lambda st: Random(st), lambda st: LBFGSB(st)], max_eval=60, size=1)
    c = counts(who)
    assert c["LBFGSB"] == pytest.approx(30, abs=2), c
    assert c["Random"] == pytest.approx(30, abs=2), c


def test_local_penalty_search_contributes_on_a_constrained_problem():
    """The last pump-thread arm, on the problem class it exists for.

    ``LocalPenaltySearch`` descends on the *penalized* objective, so a
    constrained problem is where it has something to do — and where F7
    measured it emitting 0 of 150 points beside ``Random``.  Its worker is a
    server that idles between descents, so this also exercises
    ``_bridge_pending_request``: ``produce`` must return empty-handed rather
    than sit on the pipe whenever no descent is running.
    """
    _, who, s = run(
        [lambda st: Random(st), lambda st: LocalPenaltySearch(st)],
        problem=RosenbrockConstraint(3),
        max_eval=150,
    )
    c = counts(who)
    assert c["LocalPenaltySearch"] > 0, f"the constrained arm was starved again: {c}"
    assert c["LocalPenaltySearch"] >= s.loops // 2 - 1, (c, s.loops)


def test_local_penalty_search_spends_the_budget_alone():
    """Solo it drives the whole run: one evaluation per descent step."""
    fx, who, _ = run(
        [lambda st: LocalPenaltySearch(st)],
        problem=RosenbrockConstraint(3),
        max_eval=120,
    )
    assert len(fx) >= 120
    assert set(counts(who)) == {"LocalPenaltySearch"}


def test_local_penalty_search_run_is_reproducible():
    fa, wa, _ = run(
        [lambda st: Random(st), lambda st: LocalPenaltySearch(st)],
        problem=RosenbrockConstraint(3),
        max_eval=90,
    )
    fb, wb, _ = run(
        [lambda st: Random(st), lambda st: LocalPenaltySearch(st)],
        problem=RosenbrockConstraint(3),
        max_eval=90,
    )
    np.testing.assert_array_equal(fa, fb)
    assert list(wa) == list(wb)


def test_bridge_arm_contributes_under_the_block_scheduler():
    """``StrategyBlockBandit`` gated readiness on ``has_points`` too.

    An on-demand arm's queue is empty between round trips, so the old gate
    (``blocks.py``'s ``ready`` list) made it permanently unselectable — a
    second, independent way to starve it.
    """
    from panobbgo.strategies import StrategyBlockBandit

    _, who, _ = run(
        [lambda st: Random(st), lambda st: LBFGSB(st)],
        max_eval=90,
        strategy_cls=StrategyBlockBandit,
    )
    c = counts(who)
    assert c["LBFGSB"] > 0, f"the block scheduler starved the bridge arm: {c}"


def test_bridge_arm_spends_the_whole_budget_alone():
    """L-BFGS-B multi-starts until the budget is gone; nothing truncates it."""
    fx, who, _ = run([lambda st: LBFGSB(st)], max_eval=120)
    assert len(fx) >= 120
    assert set(counts(who)) == {"LBFGSB"}


def test_bridge_run_is_reproducible_from_its_seed():
    """The claim the pump thread made impossible.

    A bridge arm's point is a function of the values it was given and its
    worker's seed, and every pipe operation now happens on the main loop's
    thread — so two runs of the same seed agree exactly, bridge arm included.
    """
    fa, wa, _ = run([lambda st: Random(st), lambda st: LBFGSB(st)], max_eval=90)
    fb, wb, _ = run([lambda st: Random(st), lambda st: LBFGSB(st)], max_eval=90)
    assert len(fa) == len(fb)
    np.testing.assert_array_equal(fa, fb)
    assert list(wa) == list(wb)


def test_cobyqa_ends_the_run_cleanly_when_it_converges():
    """COBYQA does not multi-start: when the descent ends, the arm goes away.

    The worker exits, ``produce`` sees a dead subprocess with an empty pipe,
    the heuristic goes inactive, and :meth:`StrategyBase._alive` ends the run.
    No exception, no wall-clock guard, and — the F1 shape — no leaked threads.
    """
    fx, who, s = run([lambda st: COBYQA(st)], max_eval=300)
    assert 0 < len(fx) < 300, "expected a converged, budget-short run"
    assert set(counts(who)) == {"COBYQA"}
    h = s._heuristics["COBYQA"]
    assert not h.can_produce
    assert not h.active
    assert h.cobyqa is None or not h.cobyqa.is_alive()


# ---------------------------------------------------------------------------
# The liveness predicate
# ---------------------------------------------------------------------------


def test_no_pump_threads_are_left_anywhere():
    """Every solver bridge is now pull-based; nothing emits from a thread.

    ``StrategyBase._can_still_produce`` used to count live module threads, so
    any strategy carrying one of these arms got an unconditional "yes" — the
    blind spot that made the liveness predicate a constant.
    """
    for cls in (LBFGSB, COBYQA, LocalPenaltySearch):
        assert cls.on_demand is True, cls.__name__
        assert not hasattr(cls, "_pump"), f"{cls.__name__} still has a pump thread"


def test_liveness_is_true_while_a_bridge_arm_owes_us_nothing():
    """A bridge arm with an empty queue is *not* a dead arm.

    This is the distinction the old guard could not draw: it read "every
    queue is empty" and concluded "nothing can produce".
    """
    s = StrategyRoundRobin(DeJong(dims=3), parse_args=False, testing_mode=True, seed=42)
    s.config.max_eval = 20
    s.config.sync_evaluation = True
    s.config.stop_on_convergence = False
    s.config.ui_show = False
    s.config.evaluation_method = "threaded"
    s.add_heuristic(LBFGSB(s))
    s.initialize()
    try:
        h = s._heuristics["LBFGSB"]
        assert not h.has_points  # the queue is empty by construction
        assert h.can_produce  # ... and the arm is nonetheless ready
        assert s._alive()

        # Take its point: now it owes us an evaluation, so it cannot produce
        # and — critically — asking again must not block.
        pts = h.produce(1)
        assert len(pts) == 1
        assert h._outstanding
        assert not h.can_produce
        t0 = time.time()
        assert h.produce(1) == []
        assert time.time() - t0 < 0.5, "produce() blocked on an arm that is blocked on us"
    finally:
        s._cleanup()


def test_bridge_arm_is_stopped_and_joined_by_cleanup():
    """F1's leak: ``_cleanup`` must reach heuristics that already went inactive."""
    _, _, s = run([lambda st: COBYQA(st)], max_eval=300)
    h = s._heuristics["COBYQA"]
    assert h._stopped
    assert h.cobyqa is None or not h.cobyqa.is_alive()


# ---------------------------------------------------------------------------
# F4: the trajectory does not depend on how long anything takes
# ---------------------------------------------------------------------------


class _SlowNearby(Nearby):
    """A reactive arm whose result handler is artificially slow.

    The delay burns time on the event-bus dispatcher thread, exactly where
    ``GaussianProcessHeuristic`` spends its ~0.4 s per batch.  Under the old
    guard this shortened the run: ``wait_idle`` timed out, the next pass found
    every queue empty because the refills were still undelivered, and two such
    passes ended the optimization.
    """

    delay = 0.0

    def on_new_results(self, results):
        if self.delay:
            time.sleep(self.delay)
        return super().on_new_results(results)


def _slow_run(delay, max_eval=60, seed=1234, stall=None):
    def make(st):
        h = _SlowNearby(st, radius=0.1, axes="all", new=3)
        h.delay = delay
        return h

    return run(
        [lambda st: Random(st), make],
        problem=Rosenbrock(dim=2),
        seed=seed,
        max_eval=max_eval,
        stall=stall,
    )


def test_trajectory_is_independent_of_handler_latency():
    """Identical trajectory with and without an artificial per-batch delay.

    The machine-independence half of the reproducibility contract, which
    ``tests/test_reproducibility.py`` structurally cannot see: it compares two
    runs back to back on one machine at one load level.  A seeded run must
    evaluate the same points, in the same order, and *the same number of
    them*, however slow the machine is.
    """
    fast_fx, fast_who, _ = _slow_run(0.0)
    slow_fx, slow_who, _ = _slow_run(0.05)

    assert len(fast_fx) == len(slow_fx) >= 60, (len(fast_fx), len(slow_fx))
    np.testing.assert_array_equal(fast_fx, slow_fx)
    assert list(fast_who) == list(slow_who)


def test_the_wall_clock_knob_can_no_longer_truncate_a_run():
    """``max_stall_seconds`` is inert — the sharpest form of the F4 fix.

    A per-batch handler delay an order of magnitude *above* the configured
    stall threshold used to end the run early (300 evaluations became 70–240,
    depending on machine load).  It must now change nothing at all: neither
    the number of evaluations nor which points they were.
    """
    ref_fx, ref_who, _ = _slow_run(0.05)
    fx, who, _ = _slow_run(0.05, stall=0.001)

    assert len(fx) >= 60, f"a 1 ms stall threshold truncated the run to {len(fx)}"
    np.testing.assert_array_equal(ref_fx, fx)
    assert list(ref_who) == list(who)


# ---------------------------------------------------------------------------
# F1: an exhausted portfolio ends cleanly
# ---------------------------------------------------------------------------


def test_exhausted_round_robin_ends_cleanly():
    """Every arm inactive must end the run, not divide by zero.

    ``StrategyRoundRobin.execute`` computed ``% len(self.heuristics)`` over the
    *active* arms; a one-shot arm that unsubscribes empties that list.  The
    ``ZeroDivisionError`` escaped ``_run`` and ``start`` — which caught only
    ``KeyboardInterrupt`` — so ``_cleanup`` never ran and the event-bus
    dispatcher, the evaluator pool and the results store were all leaked.
    """
    from panobbgo.heuristics import Center

    fx, who, s = run([lambda st: Center(st)], max_eval=300)
    assert 0 < len(fx) < 300
    assert set(counts(who)) == {"Center"}
    assert s.heuristics == []  # nothing active is left
    assert not s.eventbus._thread.is_alive()  # _cleanup ran: the bus is down


def test_cleanup_runs_when_execute_raises():
    """Any exception out of the main loop still tears the strategy down.

    ``start`` used to catch only ``KeyboardInterrupt``, so any other exception
    leaked the event-bus dispatcher thread and the evaluator pool.
    """
    captured = {}

    class Exploding(StrategyRoundRobin):
        def execute(self):
            captured["strategy"] = self
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        run([lambda st: Random(st)], max_eval=50, strategy_cls=Exploding)

    s = captured["strategy"]
    assert not s.eventbus._thread.is_alive(), "the event-bus thread was leaked"
