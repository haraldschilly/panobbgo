# -*- coding: utf8 -*-
# Copyright 2012-2026 Harald Schilly <harald.schilly@gmail.com>
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

from panobbgo.core import HeuristicSubprocess
import numpy as np
from functools import reduce
import operator
import threading
import time
from typing import Any, List, Optional


def rank_weights(distances):
    """WLS weights ``1 / (1 + rank)``: the nearest point gets 1, the next 1/2, ...

    ``rank`` is each point's position when sorted by distance (ties keep
    input order).  ``np.argsort`` alone gives the *indices* in sorted order,
    not ranks, and used as ranks it weights the points arbitrarily.
    """
    ranks = np.argsort(np.argsort(distances, kind="stable"), kind="stable")
    return 1.0 / (1 + ranks)


class QuadraticWlsModel(HeuristicSubprocess):
    """Fit a weighted quadratic model to the best box and propose its minimiser.

    For each new best box the results in it are fitted by weighted least
    squares (weights :func:`rank_weights` of the distance to the box's best
    point), and the minimiser of the fitted quadratic within the problem box
    becomes the next search point.

    The fit runs in a worker subprocess and is **pulled** from the main
    loop's thread, like the solver bridges
    (:class:`~panobbgo.core.PipeBridgeHeuristic`):

    * :meth:`on_new_best_box` runs on the event-bus thread and only
      *records* a snapshot of the latest best box.  It never touches the
      pipe, so a slow fit cannot stall the other handlers.  Boxes that
      arrive before the next :meth:`produce` coalesce: the latest one wins.
    * :meth:`produce` (main thread) sends the recorded box to the worker and
      collects its answer.  Under ``sync_evaluation`` it waits for the fit —
      :attr:`fit_timeout` is only a deadlock backstop for a wedged worker —
      so the emitted point is a function of the event sequence, not of
      machine speed.  Otherwise it takes an answer only when one is ready
      and returns ``[]`` in the meantime.

    At most one fit is outstanding.  Every request carries an id and every
    reply echoes it, so a reply to an abandoned request (after the backstop
    fired) is recognised and dropped.
    """

    #: Reads Splitter boxes / subscribes to its events (installed on demand).
    requires_analyzers = ("Splitter",)

    #: Points are produced on the caller's thread (see :meth:`produce`).
    on_demand = True

    #: Deadlock backstop for the synchronous wait in :meth:`produce`, in
    #: seconds.  Not a scheduling parameter: a live worker is waited for as
    #: long as its fit takes, and this only expires when it is wedged.
    fit_timeout: float = 60.0

    #: Granularity of that wait: how quickly a *dead* worker is noticed.
    _poll_slice: float = 0.05

    def __init__(self, strategy):
        HeuristicSubprocess.__init__(self, strategy)
        self.logger = self.config.get_logger("H:WLS")
        #: Id of the last request sent; every reply echoes the id it answers,
        #: so a late reply to an abandoned request is recognised and dropped.
        self._request_id = 0
        #: Id of the request whose reply we are waiting for, or ``None``.
        #: Main thread only.
        self._inflight_id: Optional[int] = None
        #: ``(points, fx_vals, best_x)`` of the latest best box, recorded on
        #: the event-bus thread and consumed by :meth:`produce`.
        self._pending_fit: Optional[tuple] = None
        self._pending_lock = threading.Lock()

    @staticmethod
    def subprocess(pipe):
        import numpy as np
        from pandas import DataFrame
        import statsmodels.api as sm
        from scipy.optimize import fmin_l_bfgs_b
        import traceback

        # Protocol: request ``(req_id, points, bounds, best_point, fx_vals)``,
        # reply ``(req_id, solution_or_None)``.  ``req_id`` is ``None`` in a
        # failure reply when the request itself could not be read.
        while True:
            req_id = None
            try:
                # Wait for input
                if not pipe.poll(1.0):
                    continue

                payload = pipe.recv()
                req_id, points, bounds, best_point, fx_vals = payload
                dim = points.shape[1]

                # Build data dictionary in the correct order for statsmodels (Intercept first)
                # Use vectorized slicing instead of list comprehensions for performance
                data = {"Intercept": np.ones(len(points))}
                for i in range(dim):
                    data["x%s" % i] = points[:, i]
                for i in range(dim):
                    for j in range(i + 1, dim):
                        data["x%s:x%s" % (i, j)] = points[:, i] * points[:, j]
                for i in range(dim):
                    data["x%s^2" % i] = points[:, i] ** 2

                # DataFrame creation preserves insertion order in Python 3.7+
                X = DataFrame(data)

                # Define columns explicitly for clarity, though it now matches X's order
                cols = ["Intercept"] + ["x%i" % i for i in range(dim)]
                mixedterms = reduce(
                    operator.add,
                    [["x%s:x%s" % (i, j) for j in range(i + 1, dim)] for i in range(dim)],
                )
                cols.extend(mixedterms)
                cols.extend(["x%s^2" % i for i in range(dim)])
                X.columns = cols

                y = DataFrame({"y": fx_vals})

                # Optimized distance calculation using axis parameter instead of apply_along_axis
                distances = np.linalg.norm(points - best_point, axis=1)
                weights = rank_weights(distances)

                model = sm.WLS(y, X, weights=weights)  # type: ignore
                result = model.fit()

                def predict(xx):
                    """
                    helper for the while loop:
                    calculates the prediction based on the model result
                    """
                    # dim is from outer scope (loop variable)
                    # result is from outer scope (loop variable)

                    # Use outer product for mixed terms to avoid nested loops
                    outer = np.outer(xx, xx)
                    mixed = outer[np.triu_indices(dim, k=1)]
                    # Construct feature vector: [1, x0..xn, mixed, squared]
                    res = np.concatenate(([1], xx, mixed, xx**2))
                    return result.predict(res)

                # optimize predict with x \in bounds

                sol, _, _ = fmin_l_bfgs_b(predict, np.zeros(dim), bounds=bounds, approx_grad=True)

                pipe.send((req_id, sol))
            except EOFError:
                break
            except Exception:
                # Log the error to stderr so it's visible in tests/logs
                traceback.print_exc()
                # Send None to indicate failure and prevent parent from hanging indefinitely
                try:
                    pipe.send((req_id, None))
                except Exception:
                    pass

    def on_new_best_box(self, best_box):
        """Record a snapshot of ``best_box`` for the next :meth:`produce`.

        Runs on the event-bus thread: cheap, no pipe I/O.  The arrays are
        copied here because the box keeps changing as results arrive.
        """
        get_val = self.strategy.constraint_handler.get_penalty_value
        # Only finite penalty values reach the least-squares fit (an infinite
        # one, e.g. from a NaN constraint, would make it nan).
        usable = [(r, v) for r in best_box.results for v in (get_val(r),) if np.isfinite(v)]
        if not usable:
            return
        pointarray = np.r_[[r.x for r, _ in usable]]
        fx_vals = np.array([v for _, v in usable], dtype=float)
        best_x = np.array(best_box.best.x, dtype=float)
        with self._pending_lock:
            self._pending_fit = (pointarray, fx_vals, best_x)

    # -- the pull side (main thread) ---------------------------------------

    @property
    def can_produce(self) -> bool:
        """A queued point, a recorded box to fit, or a fit on its way."""
        if self.has_points:
            return True
        if self._stopped or not self._worker_alive():
            return False
        return self._pending_fit is not None or self._inflight_id is not None

    def produce(self, limit: Optional[int] = None, timeout: Optional[float] = None) -> List[Any]:
        """Send the latest recorded box to the worker and hand out its answer.

        Under ``sync_evaluation`` this waits for the fit (``timeout``, default
        :attr:`fit_timeout`, is a deadlock backstop only); otherwise it never
        blocks and returns ``[]`` while the fit is still running.
        """
        if self._stopped:
            return []
        try:
            self._collect_reply(wait=0.0)
            if self._inflight_id is None:
                self._send_pending()
            if self._inflight_id is not None and self._sync():
                self._collect_reply(wait=self.fit_timeout if timeout is None else timeout)
        except (EOFError, OSError) as e:
            self.logger.warning("QuadraticWlsModel: worker pipe closed (%s)." % e)
            self._inflight_id = None
        except Exception as e:
            self.logger.error(f"Error communicating with QuadraticWlsModel subprocess: {e}")
            self._inflight_id = None
        return self.get_points(limit)

    def _sync(self) -> bool:
        return bool(getattr(self.config, "sync_evaluation", False))

    def _worker_alive(self) -> bool:
        proc = getattr(self, "_HeuristicSubprocess__subprocess", None)
        return proc is not None and proc.is_alive()

    def _send_pending(self) -> None:
        """Send the recorded box, if any, as a new request."""
        with self._pending_lock:
            pending, self._pending_fit = self._pending_fit, None
        if pending is None:
            return
        pointarray, fx_vals, best_x = pending
        # Bounds as a list of tuples for scipy compatibility.
        bounds = [tuple(row) for row in self.problem.box.box]
        self._request_id += 1
        self.pipe.send((self._request_id, pointarray, bounds, best_x, fx_vals))
        self._inflight_id = self._request_id

    def _collect_reply(self, wait: float) -> None:
        """Read replies for up to ``wait`` seconds until the in-flight one arrives; emit it.

        Replies to earlier, abandoned requests are read and dropped: without
        the id, every later emission would be the answer to the box before.
        A failure reply whose request could not be read (id ``None``) counts
        as the answer to the in-flight request, the only one outstanding.
        With ``wait > 0`` an expired deadline abandons the request — a
        wedged worker, logged as an error.
        """
        if self._inflight_id is None:
            return
        deadline = time.monotonic() + wait
        while True:
            remaining = deadline - time.monotonic()
            if not self.pipe.poll(max(0.0, min(self._poll_slice, remaining)) if wait > 0 else 0):
                if wait <= 0:
                    return  # async: not ready yet, try again on the next pull
                if not self._worker_alive() and not self.pipe.poll(0):
                    self.logger.warning("QuadraticWlsModel: worker process is not running.")
                    self._inflight_id = None
                    return
                if remaining <= 0:
                    self.logger.error(
                        "QuadraticWlsModel: no reply from the worker for %.0fs; abandoning request %s. "
                        "This is a bug, not a slow fit." % (wait, self._inflight_id)
                    )
                    self._inflight_id = None
                    return
                continue
            rid, sol = self.pipe.recv()
            if rid == self._inflight_id or rid is None:
                self._inflight_id = None
                if sol is not None:
                    self.emit(sol)
                else:
                    self.logger.warning("QuadraticWlsModel subprocess returned None (error occurred).")
                return
            self.logger.debug("QuadraticWlsModel: dropping a stale reply to request %s." % rid)
