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

from panobbgo.core import Heuristic
from panobbgo.lib import Point
import multiprocessing
import time
from queue import Full


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
                        options = {'maxiter': max_iter}
                        res = minimize(
                            objective_function,
                            x0,
                            method=method,
                            bounds=bounds,
                            options=options
                        )
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


class LocalPenaltySearch(Heuristic):
    """
    Heuristic that runs a local search using Scipy's optimizers (e.g. L-BFGS-B, Nelder-Mead)
    on the penalized objective function provided by the strategy's constraint handler.

    It runs in a separate process and communicates via Pipe.
    The heuristic thread acts as a bridge:
    1. Receives "eval" requests from worker -> Emits point to strategy.
    2. Receives results from strategy -> Sends penalty value back to worker.
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

    def emit_reliable(self, point):
        """
        Emits a single point reliably, blocking if necessary until the queue accepts it.
        """
        if self._stopped:
            return False

        x = self.problem.project(point)
        p = Point(x, self.name)

        while not self._stopped:
            try:
                self._output.put(p, timeout=1.0)
                return True
            except Full:
                continue
            except Exception as e:
                self.logger.error(f"Error emitting point: {e}")
                return False
        return False

    def on_start(self):
        # Initial start
        x0 = self.problem.random_point()
        self._start_optimization(x0)

        while not self._stopped:
            try:
                # Check for messages from worker
                if self.parent_conn.poll(0.1):
                    msg = self.parent_conn.recv()

                    if msg["type"] == "eval":
                        x = msg["x"]
                        self._pending_x = x
                        self._waiting_for_eval = True
                        if not self.emit_reliable(x):
                            self.logger.warning("Failed to emit reliable point, stopping local search.")
                            self._optimization_active = False

                    elif msg["type"] == "done":
                        self._optimization_active = False
                        self._waiting_for_eval = False
                        self._pending_x = None
                        self.logger.debug(f"Local search finished: {msg.get('message', 'unknown')}")

                    elif msg["type"] == "error":
                        self.logger.warning(f"Local search error: {msg.get('message')}")
                        self._optimization_active = False
                        self._waiting_for_eval = False

            except (EOFError, OSError):
                self.logger.debug("Pipe closed, stopping heuristic loop")
                break
            except Exception as e:
                self.logger.error(f"Error in LocalPenaltySearch loop: {e}")
                # Don't break, try to recover
                pass

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
        """
        Respond to a restart event by resetting the search around the new center.
        """
        self.clear_output()
        if center is not None:
            # Send abort message to stop any ongoing search
            try:
                self.parent_conn.send({"type": "abort"})
            except Exception:
                pass
            self._optimization_active = False
            self._waiting_for_eval = False
            self._start_optimization(center)

    def on_new_best(self, best):
        # If idle, restart search from new best
        if not self._optimization_active:
            x0 = best.x
            self._start_optimization(x0)

    def on_new_results(self, results):
        # We need to find the result corresponding to our pending evaluation
        if not self._waiting_for_eval:
            return

        my_results = [r for r in results if r.who == self.name]
        if not my_results:
            return

        # Take the latest one, assuming sequential processing
        # (the worker blocks waiting for the result)
        result = my_results[-1]

        # Calculate penalized value
        val = self.strategy.constraint_handler.get_penalty_value(result)

        try:
            self.parent_conn.send({"type": "result", "value": val})
            self._waiting_for_eval = False
            self._pending_x = None
        except Exception as e:
            self.logger.error(f"Failed to send result to worker: {e}")
