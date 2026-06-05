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
from typing import TYPE_CHECKING, Any, List

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


def run_evaluation(strategy: "StrategyBase", points: List[Any]) -> List[Any]:
    """
    Run evaluation using Dask distributed computing.

    Submits each point as a Dask future, updates the strategy's task
    accounting, and returns the newly finished results.
    """

    # Helper function to evaluate a point using the problem
    def evaluate_point(problem, point):
        """Evaluate a single point using the problem instance"""
        return problem(point)

    # distribute work using Dask futures
    # Submit each point as a separate task
    new_futures = []
    for point in points:
        future = strategy._client.submit(
            evaluate_point,
            strategy._problem_future,
            point,
            pure=False,  # Function may have side effects
        )
        new_futures.append(future)

    # and don't forget, this updates the statistics
    _add_tasks(strategy, new_futures)

    # collect new results for each finished task, hand them over to result DB
    new_results = []
    for future_id in strategy.new_finished:
        future = strategy.pending.pop(future_id, None)
        if future is not None:
            try:
                result = future.result()
                if isinstance(result, list):
                    new_results.extend(result)
                else:
                    new_results.append(result)
            except Exception as e:
                strategy.logger.error("Task failed with error: %s" % e)

    return new_results


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
            strategy.finished.append(future_id)

            # Calculate elapsed time if available
            if hasattr(future, "done") and future.done():
                try:
                    # Get task duration from Dask
                    # Note: This is approximate, based on completion time
                    strategy.tasks_walltimes[future_id] = 0.1  # placeholder
                except Exception:
                    pass

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
