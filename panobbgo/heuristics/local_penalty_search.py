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
import multiprocessing
import threading


class LocalPenaltySearch(Heuristic):
    """
    Heuristic that runs a local search using Scipy's optimizers (e.g. L-BFGS-B, Nelder-Mead)
    on the penalized objective function provided by the strategy's constraint handler.

    It runs in a separate process and communicates via Pipe.
    """

    def __init__(self, strategy, method="L-BFGS-B"):
        super().__init__(strategy, name="LocalPenaltySearch")
        self.method = method
        self.ctx = multiprocessing.get_context("spawn")
        self.parent_conn, self.child_conn = self.ctx.Pipe()
        self.process = None
        self._waiting_for_result = False
        self._lock = threading.Lock()

        # State tracking
        self._optimization_active = False

    def __start__(self):
        # Convert bounds to list of tuples for pickling and scipy compatibility
        bounds = [tuple(row) for row in self.problem.box.box]

        self.process = self.ctx.Process(
            target=self._worker,
            args=(self.child_conn, self.method, self.problem.dim, bounds),
            name=f"{self.name}-Worker",
        )
        self.process.daemon = True
        self.process.start()

    def __stop__(self):
        super().__stop__()
        # Send stop message
        try:
            with self._lock:
                self.parent_conn.send({"type": "stop"})
        except Exception:
            pass

        if self.process and self.process.is_alive():
            self.process.join(timeout=1.0)
            if self.process.is_alive():
                self.process.terminate()

    @staticmethod
    def _worker(pipe, method, dim, bounds):
        # Worker loop
        # Delayed import
        try:
            from scipy.optimize import minimize
        except ImportError:
            return

        # Callback for minimize to evaluate function
        def func(x):
            # Send x to parent
            pipe.send({"type": "eval", "x": x})
            # Wait for response
            try:
                response = pipe.recv()
                if response["type"] == "result":
                    return response["value"]
                elif response["type"] == "abort":
                    raise StopIteration("Optimization aborted by parent")
                elif response["type"] == "stop":
                    raise StopIteration("Stop heuristic")
                else:
                    raise RuntimeError(f"Unknown response: {response}")
            except (EOFError, StopIteration):
                raise StopIteration

        while True:
            # Wait for start command
            try:
                if pipe.poll(1.0):
                    msg = pipe.recv()
                    if msg["type"] == "stop":
                        break
                    if msg["type"] == "start":
                        x0 = msg["x0"]
                        # Run optimization
                        try:
                            minimize(func, x0, method=method, bounds=bounds)
                            # Notify parent we are done with this run
                            pipe.send({"type": "done"})
                        except StopIteration:
                            pass  # Aborted
                        except Exception:
                            # Log error or ignore
                            pass
            except (EOFError, BrokenPipeError):
                break
            except Exception:
                pass

    def on_start(self):
        # Start initial optimization
        x0 = self.problem.random_point()
        self._start_optimization(x0)

        while not self._stopped:
            if self.parent_conn.poll(0.5):
                try:
                    msg = self.parent_conn.recv()
                    if msg["type"] == "eval":
                        x = msg["x"]
                        with self._lock:
                            self._waiting_for_result = True
                        self.emit(x)
                    elif msg["type"] == "done":
                        with self._lock:
                            self._waiting_for_result = False
                            self._optimization_active = False
                            # Optimization finished naturally
                            # We could restart automatically or wait for on_new_best
                except EOFError:
                    break
            else:
                pass

    def _start_optimization(self, x0):
        with self._lock:
            # Only start if not active
            if not self._optimization_active:
                try:
                    self.parent_conn.send({"type": "start", "x0": x0})
                    self._optimization_active = True
                    self._waiting_for_result = False
                except Exception:
                    pass

    def on_new_best(self, best):
        # Restart search from new best if idle
        with self._lock:
            if not self._optimization_active:
                try:
                    self.parent_conn.send({"type": "start", "x0": best.x})
                    self._optimization_active = True
                    self._waiting_for_result = False
                except Exception:
                    pass

    def on_new_results(self, results):
        # Filter for my results
        my_results = [r for r in results if r.who == self.name]
        if not my_results:
            return

        # Assuming sequential execution, the last result corresponds to current wait
        result = my_results[-1]

        with self._lock:
            if self._waiting_for_result:
                val = self.strategy.constraint_handler.get_penalty_value(result)
                try:
                    self.parent_conn.send({"type": "result", "value": val})
                except Exception:
                    pass
                self._waiting_for_result = False
