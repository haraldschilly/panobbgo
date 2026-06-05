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
        from panobbgo.heuristics.local_penalty_search import LocalPenaltySearch
        import threading

        h = LocalPenaltySearch(self.strategy)
        h.__start__()

        # Start the on_start loop in a thread
        t = threading.Thread(target=h.on_start)
        t.daemon = True
        t.start()

        # Wait for initial point
        start_time = time.time()
        points = []
        while time.time() - start_time < 10.0:
            new_pts = h.get_points()
            if new_pts:
                points.extend(new_pts)
                break
            time.sleep(0.1)

        assert len(points) > 0, "Heuristic did not emit initial point"
        p1 = points[0]

        # Provide result for p1
        # Simple objective: x^2
        fx1 = np.sum(p1.x**2)
        r1 = Result(p1, fx1, cv_vec=None)

        # Feed result back. This should run in a thread or be called directly.
        # Since on_new_results uses a lock and writes to pipe, it should be safe to call directly
        # from main thread while on_start is blocked/polling in another thread.
        h.on_new_results([r1])

        # Expect another point (optimization step)
        start_time = time.time()
        points2 = []
        while time.time() - start_time < 10.0:
            new_pts = h.get_points()
            if new_pts:
                points2.extend(new_pts)
                break
            time.sleep(0.1)

        assert len(points2) > 0, "Heuristic did not emit next point after result"

        # Cleanup
        h._stopped = True
        h.__stop__()
        t.join(timeout=2.0)


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

    def test_emit_reliable_returns_false_when_stopped(self):
        h = self.make_heuristic()
        h._stopped = True
        assert h.emit_reliable(np.array([0.0, 0.0])) is False

    def test_emit_reliable_retries_on_full_queue(self):
        import queue
        import threading

        h = self.make_heuristic()
        h._output = queue.Queue(1)
        h._output.put("blocker")

        # Free the queue only after emit_reliable's first 1.0s put() timed
        # out, so the `except Full: continue` retry path actually executes.
        threading.Timer(1.2, lambda: h._output.get()).start()
        assert h.emit_reliable(np.array([0.0, 0.0])) is True

    def test_start_optimization_send_failure_logged(self):
        h = self.make_heuristic()
        h.parent_conn.send.side_effect = OSError("pipe broke")
        h._start_optimization(np.array([0.0, 0.0]))
        assert h._optimization_active is False

    def test_on_restart_aborts_and_restarts(self):
        h = self.make_heuristic()
        h._optimization_active = True
        h.on_restart(np.array([0.1, 0.2]), "test restart")

        types = [c.args[0]["type"] for c in h.parent_conn.send.call_args_list]
        assert types == ["abort", "start"]
        assert h._optimization_active is True

    def test_on_restart_survives_abort_send_failure(self):
        h = self.make_heuristic()
        h.parent_conn.send.side_effect = [OSError("gone"), None]
        h.on_restart(np.array([0.1, 0.2]), "test restart")
        assert h._optimization_active is True

    def test_on_new_best_restarts_when_idle(self):
        h = self.make_heuristic()
        best = Result(Point(np.array([0.5, 0.5]), "test"), 1.0)
        h.on_new_best(best)
        assert h._optimization_active is True
        assert h.parent_conn.send.call_args[0][0]["type"] == "start"

    def test_on_new_results_ignored_when_not_waiting(self):
        h = self.make_heuristic()
        r = Result(Point(np.array([0.5, 0.5]), h.name), 1.0)
        h.on_new_results([r])
        h.parent_conn.send.assert_not_called()

    def test_on_new_results_ignores_foreign_results(self):
        h = self.make_heuristic()
        h._waiting_for_eval = True
        r = Result(Point(np.array([0.5, 0.5]), "SomeoneElse"), 1.0)
        h.on_new_results([r])
        h.parent_conn.send.assert_not_called()
        assert h._waiting_for_eval is True

    def test_on_new_results_sends_penalty_value(self):
        h = self.make_heuristic()
        h._waiting_for_eval = True
        r = Result(Point(np.array([0.5, 0.5]), h.name), 2.5)
        h.on_new_results([r])

        msg = h.parent_conn.send.call_args[0][0]
        assert msg["type"] == "result"
        assert msg["value"] == 2.5
        assert h._waiting_for_eval is False

    def test_on_new_results_send_failure_logged(self):
        h = self.make_heuristic()
        h._waiting_for_eval = True
        h.parent_conn.send.side_effect = OSError("pipe broke")
        r = Result(Point(np.array([0.5, 0.5]), h.name), 2.5)
        h.on_new_results([r])  # must not raise

    def test_on_start_loop_handles_messages(self):
        # Script: eval -> done -> error -> EOF (breaks the loop).
        h = self.make_heuristic()
        msgs = [
            {"type": "eval", "x": np.array([0.3, 0.3])},
            {"type": "done", "message": "converged"},
            {"type": "error", "message": "boom"},
        ]

        def poll(timeout=None):
            if msgs:
                return True
            raise EOFError

        h.parent_conn.poll.side_effect = poll
        h.parent_conn.recv.side_effect = lambda: msgs.pop(0)

        h.on_start()  # returns via the EOF branch

        # The eval message must have produced an emitted point.
        assert len(h.get_points()) == 1
        assert h._optimization_active is False


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
