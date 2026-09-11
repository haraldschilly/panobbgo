# -*- coding: utf8 -*-
from __future__ import unicode_literals

import numpy as np
import time
import pytest
from panobbgo.utils import PanobbgoTestCase
from panobbgo.lib import Point, Result, BoundingBox, Problem


class MockConstraintHandler:
    def get_penalty_value(self, result):
        if result is None or result.fx is None:
            return float("inf")
        cv = result.cv if result.cv is not None else 0.0
        return result.fx + 100.0 * cv


class LocalPenaltySearchTest(PanobbgoTestCase):
    def setUp(self):
        super().setUp()
        self.strategy.constraint_handler = MockConstraintHandler()
        self.strategy.name = "MockStrategy"
        # Increase queue capacity for testing
        self.strategy.config.capacity = 100

    def test_initialization(self):
        from panobbgo.heuristics.local_penalty_search import LocalPenaltySearch

        h = LocalPenaltySearch(self.strategy)
        assert h is not None
        assert h.name == "LocalPenaltySearch"

    def test_start_stop(self):
        from panobbgo.heuristics.local_penalty_search import LocalPenaltySearch

        h = LocalPenaltySearch(self.strategy)
        h.__start__()
        time.sleep(0.5)
        # Process should be alive
        assert h.process is not None
        assert h.process.is_alive()

        h.__stop__()
        time.sleep(0.5)
        assert not h.process.is_alive()

    def test_optimization_flow(self):
        """One full round trip against the real subprocess, pull-style.

        ``produce`` drives the whole protocol on this thread: it sends the
        queued ``start``, receives the first ``eval`` request and emits the
        point.  ``on_new_results`` stores the value; the next ``produce`` hands
        it over and receives the next request.
        """
        from panobbgo.heuristics.local_penalty_search import LocalPenaltySearch

        h = LocalPenaltySearch(self.strategy)
        h.__start__()
        try:
            h.on_start()  # queues the first descent; sends nothing yet
            assert h._pending_start is not None

            points = h.produce(1, timeout=20.0)
            assert len(points) > 0, "Heuristic did not emit initial point"
            p1 = points[0]
            assert h._outstanding

            # Simple objective: x^2
            r1 = Result(p1, float(np.sum(p1.x**2)), cv_vec=None)
            h.on_new_results([r1])
            assert not h._fx_inbox.empty(), "the penalty value was not stored"

            points2 = h.produce(1, timeout=20.0)
            assert len(points2) > 0, "Heuristic did not emit next point after result"
        finally:
            h.__stop__()


# ---------------------------------------------------------------------------
# In-process tests for the subprocess worker function.
#
# The worker normally runs in a *spawned* process (invisible to coverage and
# hard to assert against), so these tests drive `_local_penalty_search_worker`
# directly in the main process through a scripted pipe object.
# ---------------------------------------------------------------------------


class ScriptedPipe:
    """Mimics one end of a multiprocessing Pipe with a scripted conversation.

    Messages the worker `recv()`s come from `inbox`; everything the worker
    `send()`s is recorded in `sent`. Eval requests are answered according to
    `eval_reply` ("result" computes a sphere value; "abort"/"stop" inject the
    respective control message; a non-float value provokes an optimizer error).
    """

    def __init__(self, inbox=None, eval_reply="result"):
        self.inbox = list(inbox or [])
        self.sent = []
        self.eval_reply = eval_reply

    def poll(self, timeout=None):
        if not self.inbox:
            # Script exhausted — behave like a closed pipe so every code
            # path terminates instead of polling an empty inbox forever.
            raise EOFError
        return True

    def recv(self):
        return self.inbox.pop(0)

    def send(self, msg):
        self.sent.append(msg)
        if msg["type"] == "eval":
            if self.eval_reply == "result":
                val = float(np.sum(np.asarray(msg["x"], dtype=float) ** 2))
                self.inbox.append({"type": "result", "value": val})
            elif self.eval_reply == "garbage":
                self.inbox.append({"type": "result", "value": "not-a-number"})
            else:  # "abort" or "stop"
                self.inbox.append({"type": self.eval_reply})
        elif msg["type"] in ("done", "error"):
            # End the worker loop after the optimization concludes.
            self.inbox.append({"type": "stop"})


def _run_worker(pipe, dim=1):
    from panobbgo.heuristics.local_penalty_search import _local_penalty_search_worker

    bounds = [(-5.0, 5.0)] * dim
    _local_penalty_search_worker(pipe, "Nelder-Mead", dim, bounds, max_iter=5)


def test_worker_stop_message_exits():
    pipe = ScriptedPipe(inbox=[{"type": "stop"}])
    _run_worker(pipe)
    assert pipe.sent == []


def test_worker_full_optimization_run():
    pipe = ScriptedPipe(inbox=[{"type": "start", "x0": np.array([2.0])}])
    _run_worker(pipe)

    evals = [m for m in pipe.sent if m["type"] == "eval"]
    dones = [m for m in pipe.sent if m["type"] == "done"]
    assert len(evals) > 0, "worker never requested an evaluation"
    assert len(dones) == 1, "worker did not report completion"


def test_worker_abort_during_eval():
    pipe = ScriptedPipe(inbox=[{"type": "start", "x0": np.array([2.0])}], eval_reply="abort")
    _run_worker(pipe)

    # Aborted optimizations end silently — no done/error message.
    assert not [m for m in pipe.sent if m["type"] in ("done", "error")]


def test_worker_stop_during_eval():
    pipe = ScriptedPipe(inbox=[{"type": "start", "x0": np.array([2.0])}], eval_reply="stop")
    _run_worker(pipe)
    assert not [m for m in pipe.sent if m["type"] in ("done", "error")]


def test_worker_optimizer_error_reported():
    # A non-numeric objective value makes scipy blow up -> worker sends "error".
    pipe = ScriptedPipe(inbox=[{"type": "start", "x0": np.array([2.0])}], eval_reply="garbage")
    _run_worker(pipe)
    assert [m for m in pipe.sent if m["type"] == "error"], "worker did not report the optimizer error"


def test_worker_pipe_eof_exits():
    class EOFPipe:
        def poll(self, timeout=None):
            raise EOFError

    _run_worker(EOFPipe())  # must simply return, not raise


# ---------------------------------------------------------------------------
# Branch tests for the heuristic side (no subprocess spawned).
# ---------------------------------------------------------------------------


class LocalPenaltySearchBranchTest(PanobbgoTestCase):
    def setUp(self):
        super().setUp()
        from unittest import mock

        self.strategy.constraint_handler = MockConstraintHandler()
        self.mock = mock

    def make_heuristic(self):
        from panobbgo.heuristics.local_penalty_search import LocalPenaltySearch

        h = LocalPenaltySearch(self.strategy)
        # No __start__() — tests drive the pipe directly, no subprocess.
        h.parent_conn = self.mock.MagicMock()
        return h

    def make_running(self):
        """A heuristic with a live worker and an active descent."""
        h = self.make_heuristic()
        h.process = self.mock.MagicMock()
        h.process.is_alive.return_value = True
        h._optimization_active = True
        return h

    def test_start_optimization_send_failure_logged(self):
        h = self.make_heuristic()
        h.parent_conn.send.side_effect = OSError("pipe broke")
        h._start_optimization(np.array([0.0, 0.0]))
        assert h._optimization_active is False

    # -- deferred control --------------------------------------------------
    #
    # ``on_restart`` / ``on_new_best`` / ``on_start`` run on the event-bus
    # dispatcher thread, which may not write to the pipe (the main loop's
    # thread does the ``send``/``recv`` of the round trip) nor touch the
    # output queue (the strategy may be draining it).  They record; ``produce``
    # applies.  See panobbgo/heuristics/local_penalty_search.py.

    def test_on_restart_defers_the_abort_and_restart(self):
        h = self.make_running()
        h.on_restart(np.array([0.1, 0.2]), "test restart")

        # Nothing has reached the pipe yet.
        h.parent_conn.send.assert_not_called()
        assert h._pending_abort and h._pending_clear
        assert h._pending_start is not None

        h._apply_pending_control()
        types = [c.args[0]["type"] for c in h.parent_conn.send.call_args_list]
        assert types == ["abort", "start"]
        assert h._optimization_active is True

    def test_on_restart_survives_abort_send_failure(self):
        h = self.make_running()
        h.parent_conn.send.side_effect = [OSError("gone"), None]
        h.on_restart(np.array([0.1, 0.2]), "test restart")
        h._apply_pending_control()
        assert h._optimization_active is True

    def test_on_new_best_restarts_when_idle(self):
        h = self.make_heuristic()
        best = Result(Point(np.array([0.5, 0.5]), "test"), 1.0)
        h.on_new_best(best)
        assert h._pending_start is not None
        h.parent_conn.send.assert_not_called()

        h._apply_pending_control()
        assert h._optimization_active is True
        assert h.parent_conn.send.call_args[0][0]["type"] == "start"

    def test_on_new_best_leaves_a_running_search_alone(self):
        h = self.make_running()
        h.on_new_best(Result(Point(np.array([0.5, 0.5]), "test"), 1.0))
        assert h._pending_start is None

    def test_on_start_queues_the_first_search(self):
        h = self.make_heuristic()
        h.on_start()
        assert h._pending_start is not None
        h.parent_conn.send.assert_not_called()

    # -- the f(x) hand-off -------------------------------------------------

    def test_on_new_results_ignored_when_not_waiting(self):
        h = self.make_heuristic()
        r = Result(Point(np.array([0.5, 0.5]), h.name), 1.0)
        h.on_new_results([r])
        assert h._fx_inbox.empty()

    def test_on_new_results_ignores_foreign_results(self):
        h = self.make_heuristic()
        h._outstanding = True
        h._waiting_for_eval = True
        r = Result(Point(np.array([0.5, 0.5]), "SomeoneElse"), 1.0)
        h.on_new_results([r])
        assert h._fx_inbox.empty()
        assert h._waiting_for_eval is True

    def test_on_new_results_stores_the_penalty_value(self):
        """Stored, not sent: the bus thread must not do I/O."""
        h = self.make_heuristic()
        h._outstanding = True
        h._waiting_for_eval = True
        r = Result(Point(np.array([0.5, 0.5]), h.name), 2.5)
        h.on_new_results([r])

        h.parent_conn.send.assert_not_called()
        assert h._fx_inbox.get_nowait() == 2.5
        assert h._waiting_for_eval is False

    def test_produce_sends_the_stored_value_and_takes_the_next_point(self):
        h = self.make_running()
        h._outstanding = True
        h._fx_inbox.put(2.5)
        h.parent_conn.poll.return_value = True
        h.parent_conn.recv.return_value = {"type": "eval", "x": np.array([0.3, 0.3])}

        points = h.produce(1)
        assert h.parent_conn.send.call_args_list[0].args[0] == {"type": "result", "value": 2.5}
        assert len(points) == 1
        np.testing.assert_allclose(points[0].x, [0.3, 0.3])
        assert h._outstanding and h._waiting_for_eval

    def test_produce_returns_nothing_while_the_worker_idles(self):
        """The worker is a server: between descents it sends nothing at all.

        ``_bridge_pending_request`` says so, which is what keeps ``produce``
        from sitting on the pipe until the deadlock backstop.
        """
        h = self.make_heuristic()  # no descent, no pending start
        h.process = self.mock.MagicMock()
        h.process.is_alive.return_value = True
        assert h.can_produce is False
        assert h.produce(1) == []
        h.parent_conn.poll.assert_not_called()

    def test_produce_handles_done_and_error(self):
        for kind in ("done", "error"):
            h = self.make_running()
            h.parent_conn.poll.return_value = True
            h.parent_conn.recv.return_value = {"type": kind, "message": "whatever"}
            assert h.produce(1) == []
            assert h._optimization_active is False
            assert h._waiting_for_eval is False
            assert h._outstanding is False

    def test_produce_emits_the_point_of_an_eval_request(self):
        h = self.make_running()
        h.parent_conn.poll.return_value = True
        h.parent_conn.recv.return_value = {"type": "eval", "x": np.array([0.3, 0.3])}
        points = h.produce(1)
        assert len(points) == 1
        assert h._optimization_active is True


def test_worker_eval_send_failure_aborts_optimization():
    """pipe.send failing inside the objective → StopIteration → silent end."""

    class SendFailsOnEval(ScriptedPipe):
        def send(self, msg):
            if msg["type"] == "eval":
                raise OSError("pipe broke")
            super().send(msg)

    pipe = SendFailsOnEval(inbox=[{"type": "start", "x0": np.array([2.0])}])
    _run_worker(pipe)
    assert not [m for m in pipe.sent if m["type"] in ("done", "error")]


def test_worker_unexpected_recv_error_exits():
    """A non-EOF error in the outer loop must break it, not crash."""

    class RecvBlowsUp(ScriptedPipe):
        def recv(self):
            raise ValueError("corrupted message")

    pipe = RecvBlowsUp(inbox=[{"type": "start", "x0": np.array([2.0])}])
    _run_worker(pipe)  # must return, not raise
    assert pipe.sent == []
