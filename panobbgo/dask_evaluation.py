# -*- coding: utf8 -*-
# Copyright 2026 Panobbgo Contributors
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

"""
Dask-based distributed evaluation backend
=========================================

Everything Dask-specific that used to live inline in :mod:`panobbgo.core`.
``dask`` is imported lazily — only when a strategy actually runs with
``evaluation_method = "dask"`` — so the rest of the framework carries no
Dask code paths or import-time dependency.

The functions operate on a :class:`~panobbgo.core.StrategyBase` instance and
use the strategy's ``_client`` / ``_cluster`` / ``_problem_future``
attributes, which tests and external code set/inspect directly.
"""

from __future__ import annotations

import time as time_module
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

if TYPE_CHECKING:
    from .core import StrategyBase


def setup_cluster(strategy: "StrategyBase", problem: Any) -> None:
    """
    Set up a Dask cluster based on configuration.
    Supports both local and remote clusters.
    """
    from dask.distributed import Client, LocalCluster

    config = strategy.config
    if config.dask_cluster_type == "local":
        # Create a local cluster
        strategy.logger.info("Setting up local Dask cluster with %d workers" % config.dask_n_workers)
        strategy._cluster = LocalCluster(
            n_workers=int(config.dask_n_workers),
            threads_per_worker=int(config.dask_threads_per_worker),
            memory_limit=str(config.dask_memory_limit),
            dashboard_address=str(config.dask_dashboard_address),
            silence_logs=False,
        )
        strategy._client = Client(strategy._cluster)
    else:
        # Connect to remote cluster
        strategy.logger.info("Connecting to remote Dask cluster at %s" % config.dask_scheduler_address)
        strategy._client = Client(config.dask_scheduler_address)

    # Scatter the problem to all workers
    strategy._problem_future = strategy._client.scatter(problem, broadcast=True)

    strategy.logger.info("Dask cluster ready with %d workers" % len(strategy._client.scheduler_info()["workers"]))
    if config.dask_cluster_type == "local":
        strategy.logger.info("Dashboard available at: http://localhost%s" % config.dask_dashboard_address)


class DaskEvaluators:
    """
    Evaluators view for the Dask backend (compatibility object for strategies
    that reference ``strategy.evaluators``).
    """

    def __init__(self, strategy: "StrategyBase") -> None:
        self.strategy = strategy

    @property
    def outstanding(self) -> List[str]:
        # Return list of pending task keys
        return list(self.strategy.pending.keys())

    def __len__(self) -> int:
        # Return number of Dask workers
        if hasattr(self.strategy, "_client"):
            return len(self.strategy._client.scheduler_info()["workers"])
        return 0


def evaluate_point(problem: Any, point: Any, timeout: Optional[float] = None) -> Tuple[Any, float, Any, bool]:
    """The dask task: evaluate one point; returns ``(result, walltime in seconds, error, timed_out)``.

    A raising objective is reported as ``(None, walltime, repr(exc), False)``
    rather than raised, so its walltime is booked like the local pool books a
    failed task's.

    With ``timeout`` (``evaluation.timeout``) the limit is enforced **here, on
    the worker, per call**: the objective runs in a child process
    (:func:`panobbgo.timeout_call.call_with_timeout`) whose clock starts when
    the call starts — queue time on the cluster never counts — and which is
    killed on expiry, so a hung objective never holds its dask worker.  The
    task then returns ``(None, walltime, None, True)`` and the client books a
    ``NaN`` placeholder.  Without ``timeout`` the objective runs in the
    worker itself, as before (no subprocess overhead).
    """
    import time

    if timeout is None:
        t0 = time.perf_counter()
        try:
            result = problem(point)
        except Exception as exc:
            return None, time.perf_counter() - t0, repr(exc), False
        return result, time.perf_counter() - t0, None, False

    from panobbgo.timeout_call import call_with_timeout

    call = call_with_timeout(problem, point, timeout)
    return call.result, call.seconds, call.error, call.timed_out


def run_evaluation(strategy: "StrategyBase", points: List[Any]) -> List[Any]:
    """
    Run evaluation using Dask distributed computing.

    Submits each point as a Dask future, updates the strategy's task
    accounting, and returns the newly finished results.

    ``evaluation.timeout`` is enforced on the worker (:func:`evaluate_point`);
    the client does not expire futures itself.  A worker that dies loses its
    task to dask's own recovery (it reruns elsewhere, or the future fails and
    the evaluation is booked as failed), and a task that cannot run because
    the cluster has no workers is reported by the strategy's waiting WARNING
    — neither needs a client-side clock, which would charge queue time.
    """
    _warn_ignored_options(strategy)
    t = getattr(strategy.config, "evaluation_timeout", None)
    timeout = float(t) if t else None

    # distribute work using Dask futures
    # Submit each point as a separate task; remember its point so a failure
    # can be reported (``failed_evaluations``).
    points_by_key = strategy.__dict__.setdefault("_dask_points", {})
    submitted_at = strategy.__dict__.setdefault("_dask_submitted", {})
    new_futures = []
    for point in points:
        future = strategy._client.submit(
            evaluate_point,
            strategy._problem_future,
            point,
            timeout,
            pure=False,  # Function may have side effects
        )
        points_by_key[future.key] = point
        submitted_at[future.key] = time_module.time()
        new_futures.append(future)

    # and don't forget, this updates the statistics
    _add_tasks(strategy, new_futures)

    # collect new results for each finished task, hand them over to result DB
    new_results = []
    failed = []
    for future_id in strategy.new_finished:
        future = strategy.pending.pop(future_id, None)
        point = points_by_key.pop(future_id, None)
        submitted_at.pop(future_id, None)
        if future is not None:
            try:
                result, walltime, error, timed_out = future.result()
            except Exception as e:
                # Lost worker, cancellation, ...: no timing to book.
                strategy.logger.error("Task failed with error: %s" % e)
                if point is not None:
                    failed.append(point)
                continue
            strategy.record_walltime(walltime)
            if timed_out:
                if point is not None:
                    new_results.append(strategy._timed_out_result(point, walltime))
            elif error is not None:
                strategy.logger.error("Evaluation failed: %s" % error)
                if point is not None:
                    failed.append(point)
            elif isinstance(result, list):
                new_results.extend(result)
            else:
                new_results.append(result)

    strategy._publish_failures(failed)
    return new_results


def _warn_ignored_options(strategy: "StrategyBase") -> None:
    """Warn once that ``evaluation.sync`` does not apply to dask.

    Results arrive in completion order; the option used to be dropped
    silently.  (``evaluation.timeout`` does apply: see :func:`evaluate_point`.)
    """
    if getattr(strategy, "_warned_dask_options", False):
        return
    if getattr(strategy.config, "sync_evaluation", False):
        setattr(strategy, "_warned_dask_options", True)
        strategy.logger.warning(
            "evaluation.sync ignored for evaluation.method 'dask': results arrive in completion order. "
            "Use 'threaded' or 'processes' for reproducible runs."
        )


def _add_tasks(strategy: "StrategyBase", new_futures: List[Any]) -> None:
    """
    Accounting routine for the parallel Dask tasks
    (moved from ``StrategyBase._add_tasks``).
    """
    if new_futures is not None:
        for future in new_futures:
            # Use future.key as identifier
            strategy.pending[future.key] = future

    # Find completed futures
    strategy.new_finished = []
    completed_keys = []

    for future_id, future in list(strategy.pending.items()):
        if future.done():
            strategy.new_finished.append(future_id)
            completed_keys.append(future_id)
            strategy.n_finished += 1

    if time_module.time() - strategy.show_last > float(strategy.config.show_interval):
        strategy.info()
        strategy.show_last = time_module.time()


def close(strategy: "StrategyBase") -> None:
    """
    Cancel any outstanding Dask futures and close client + cluster.
    """
    for future in list(strategy.pending.values()):
        try:
            future.cancel()
        except Exception:
            pass

    # Close Dask client and cluster
    if hasattr(strategy, "_client"):
        strategy._client.close()
    if hasattr(strategy, "_cluster"):
        strategy._cluster.close()
