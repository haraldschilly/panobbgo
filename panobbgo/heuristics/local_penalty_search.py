# -*- coding: utf8 -*-
# Copyright 2025 Panobbgo Contributors
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

from panobbgo.core import PipeBridgeHeuristic
import multiprocessing
import time
from typing import Any, Optional


def _local_penalty_search_worker(pipe, method, dim, bounds, max_iter=50):
    """
    Worker function that runs optimization in a separate process.

    Args:
        pipe: Multiprocessing connection for communication with parent.
        method: Optimization method (str).
        dim: Problem dimension.
        bounds: List of (min, max) tuples.
        max_iter: Maximum iterations for the optimizer.
    """
    try:
        from scipy.optimize import minimize
    except ImportError:
        return

    def objective_function(x):
        # 1. Send candidate point to parent for evaluation
        try:
            pipe.send({"type": "eval", "x": x})
        except Exception:
            raise StopIteration("Pipe closed")

        # 2. Wait for penalized objective value
        # Use a loop with poll/timeout to detect aborts or parent death
        start_wait = time.time()
        while True:
            if time.time() - start_wait > 600.0:  # 10 minute timeout per evaluation
                raise StopIteration("Timeout waiting for evaluation result")

            try:
                if pipe.poll(0.1):
                    msg = pipe.recv()
                    if msg["type"] == "result":
                        return msg["value"]
                    elif msg["type"] == "stop":
                        raise StopIteration("Stop requested")
                    elif msg["type"] == "abort":
                        raise StopIteration("Optimization aborted")
            except (EOFError, OSError):
                raise StopIteration("Pipe closed")

    while True:
        try:
            # Wait for start command
            if pipe.poll(1.0):
                msg = pipe.recv()
                if msg["type"] == "stop":
                    break

                if msg["type"] == "start":
                    x0 = msg["x0"]
                    # Run optimization
                    try:
                        options = {"maxiter": max_iter}
                        res = minimize(objective_function, x0, method=method, bounds=bounds, options=options)
                        # Notify parent we are done
                        pipe.send({"type": "done", "success": res.success, "message": res.message})
                    except StopIteration:
                        # Optimization interrupted (e.g. by stop/abort or pipe close)
                        pass
                    except Exception as e:
                        # Optimization failed for other reasons
                        try:
                            pipe.send({"type": "error", "message": str(e)})
                        except Exception:
                            pass
        except (EOFError, OSError):
            break
        except Exception:
            # unexpected error in worker loop
            break


class LocalPenaltySearch(PipeBridgeHeuristic):
    """
    Heuristic that runs a local search using Scipy's optimizers (e.g. L-BFGS-B, Nelder-Mead)
    on the penalized objective function provided by the strategy's constraint handler.

    It runs in a separate process and communicates via Pipe, as a **pull
    bridge** (:class:`~panobbgo.core.PipeBridgeHeuristic`): the strategy asks
    for the next point with :meth:`~panobbgo.core.PipeBridgeHeuristic.produce`
    and the whole round trip happens on the caller's thread.

    1. ``produce`` receives an ``"eval"`` request from the worker and emits
       the point it carries.
    2. ``on_new_results`` *stores* the penalty value (it runs on the event-bus
       dispatcher thread, which must neither block nor do I/O).
    3. The next ``produce`` sends that value back and receives the next
       request.

    Until 2026-09 steps 1 and 2 ran on a daemon "pump" thread and the event-bus
    thread respectively, so the output queue was empty whenever a scheduler
    looked at it and any competitor with a stocked queue starved this arm
    completely (F3 of ``planning/results/2026-09-10/invariants_findings.md``).

    Unlike L-BFGS-B and COBYQA, this worker is a **server**: it idles between
    searches, waiting for a ``"start"`` command, so
    :meth:`_bridge_pending_request` reports whether a descent is actually
    running.  The commands that start one arrive on event handlers
    (:meth:`on_start`, :meth:`on_new_best`, :meth:`on_restart`) and are
    therefore *recorded* there and sent from :meth:`produce` — only one thread
    may write to the pipe.
    """

    def __init__(self, strategy, method="L-BFGS-B", max_iter=50):
        super().__init__(strategy, name="LocalPenaltySearch")
        self.method = method
        self.max_iter = max_iter
        self.ctx = multiprocessing.get_context("spawn")
        self.parent_conn, self.child_conn = self.ctx.Pipe()
        self.process = None

        # State tracking
        self._optimization_active = False
        self._waiting_for_eval = False  # True if worker is waiting for us to send a result
        self._pending_x = None  # Last point emitted, waiting for result
        # Deferred control commands, recorded by event handlers and sent by
        # ``produce`` on the main loop's thread.
        self._pending_start: Optional[Any] = None
        self._pending_abort: bool = False
        self._pending_clear: bool = False

    # ------------------------------------------------------------------
    # The pull bridge (see panobbgo.core.PipeBridgeHeuristic)
    # ------------------------------------------------------------------

    @property
    def p1(self) -> Any:
        """The base class's name for :attr:`parent_conn`."""
        return self.parent_conn

    def _bridge_process(self) -> Any:
        return self.process

    def _bridge_pending_request(self) -> bool:
        """A descent is running, or one is queued up to be started."""
        return self._optimization_active or self._pending_start is not None

    def _bridge_point(self, msg: Any) -> Any:
        """Unwrap the ``{"type": "eval", "x": ...}`` request."""
        return msg["x"]

    def _bridge_send_fx(self, fx: float) -> None:
        """This worker's replies are tagged, not bare floats."""
        self.parent_conn.send({"type": "result", "value": fx})

    def _bridge_control(self, msg: Any) -> bool:
        """Consume everything that is not an evaluation request."""
        kind = msg.get("type") if isinstance(msg, dict) else None
        if kind == "eval":
            self._pending_x = msg["x"]
            self._waiting_for_eval = True
            return False
        if kind == "done":
            self._end_optimization()
            self.logger.debug(f"Local search finished: {msg.get('message', 'unknown')}")
            return True
        if kind == "error":
            self.logger.warning(f"Local search error: {msg.get('message')}")
            self._end_optimization()
            return True
        self.logger.warning(f"LocalPenaltySearch: unexpected message {kind!r}")
        return True

    def _end_optimization(self) -> None:
        """The descent ended; the worker is idle until the next ``start``."""
        self._optimization_active = False
        self._waiting_for_eval = False
        self._pending_x = None
        # The worker never got the value it last asked for, or asked for
        # nothing more: forget the round trip so the next descent starts clean.
        self._bridge_reset()

    def produce(self, limit=None, timeout=None):
        """Flush any deferred control command, then do the round trip."""
        self._apply_pending_control()
        return super().produce(limit, timeout)

    def _apply_pending_control(self) -> None:
        """Send the commands the event handlers recorded.  Main thread only.

        ``on_new_best`` / ``on_restart`` / ``on_start`` run on the event-bus
        dispatcher thread.  Writing to the pipe there would race the
        ``send``/``recv`` this class does from the main loop's thread — two
        threads writing pickled frames into one connection is corruption — so
        they only record, and this applies.
        """
        if self._pending_clear:
            self._pending_clear = False
            self.clear_output()
        if self._pending_abort:
            self._pending_abort = False
            try:
                self.parent_conn.send({"type": "abort"})
            except Exception:
                pass
            self._optimization_active = False
            self._waiting_for_eval = False
            self._bridge_reset()
        if self._pending_start is not None and not self._optimization_active:
            x0 = self._pending_start
            self._pending_start = None
            self._start_optimization(x0)

    def __start__(self):
        # Convert bounds to list of tuples for pickling and scipy compatibility
        bounds = [tuple(row) for row in self.problem.box.box]

        self.process = self.ctx.Process(
            target=_local_penalty_search_worker,
            args=(self.child_conn, self.method, self.problem.dim, bounds, self.max_iter),
            name=f"{self.name}-Worker",
        )
        self.process.daemon = True
        self.process.start()

    def __stop__(self):
        super().__stop__()
        # Send stop message
        try:
            self.parent_conn.send({"type": "stop"})
        except Exception:
            pass

        if self.process and self.process.is_alive():
            self.process.join(timeout=1.0)
            if self.process.is_alive():
                self.logger.warning("LocalPenaltySearch process did not exit gracefully, terminating...")
                self.process.terminate()
                self.process.join(timeout=0.1)
                if self.process.is_alive():
                    self.process.kill()

        # Close pipes
        try:
            self.parent_conn.close()
            self.child_conn.close()
        except Exception:
            pass

    def on_start(self):
        """Queue the first local search.

        The starting point is drawn *here*, so this heuristic consumes exactly
        the same number of draws from its module RNG, at the same point in the
        stream, as it always did.  Only the ``send`` is deferred to
        :meth:`produce` — see :meth:`_apply_pending_control`.
        """
        self._pending_start = self.problem.random_point(rng=self.rng)

    def _start_optimization(self, x0):
        if not self._optimization_active:
            try:
                self.parent_conn.send({"type": "start", "x0": x0})
                self._optimization_active = True
                self._waiting_for_eval = False
                self.logger.debug("Started local search optimization")
            except Exception as e:
                self.logger.error(f"Failed to start optimization: {e}")

    def on_restart(self, center, reason):
        """Reset the search around the new center.

        Records the intent; :meth:`_apply_pending_control` performs it on the
        main loop's thread (this handler runs on the event bus, which may not
        touch the pipe or the output queue — the strategy may be draining it
        at that instant).
        """
        self._pending_clear = True
        if center is not None:
            self._pending_abort = True
            self._pending_start = center

    def on_new_best(self, best):
        # If idle, restart the search from the new best.
        if not self._optimization_active and self._pending_start is None:
            self._pending_start = best.x

    def on_new_results(self, results):
        """Store the penalty value of our outstanding point.

        Delegates to :class:`~panobbgo.core.PipeBridgeHeuristic`, which keeps
        the value in a queue until :meth:`produce` can send it from the main
        thread.
        """
        super().on_new_results(results)
        if not self._fx_inbox.empty():
            self._waiting_for_eval = False
            self._pending_x = None
