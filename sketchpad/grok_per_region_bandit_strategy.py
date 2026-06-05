# -*- coding: utf8 -*-
# Copyright 2012-2026 Panobbgo Contributors
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
Per-Region Bandit Strategy
==========================

A spatially-aware bandit strategy that maintains independent multi-armed
bandits (Thompson Sampling style) for each leaf region identified by the
Splitter analyzer.

**Rationale**

- The global bandit (StrategyUCB / StrategyThompsonSampling / StrategyRewarding)
  uses a single set of performance statistics for the entire search space.
- Different regions of the landscape often require different heuristics:
  - Smooth valley → local optimizer (NelderMead, LBFGSB, COBYQA)
  - Multimodal basin → global DE-family or Sobol exploration
  - High-density cluster → Nearby perturbation or Restart

By scoping a bandit per Splitter.Box leaf, the strategy learns *locally*
optimal heuristic allocation while sharing the global heuristic pool and
the Splitter's hierarchical decomposition.

**Implementation notes (prototype)**

- Reuses the reward/improvement logic from StrategyThompsonSampling.
- Each leaf maintains its own ``ts_total_reward`` / ``ts_counts`` per heuristic.
- On ``execute()`` we lookup the current "biggest_leaf" (or best_box) and
  delegate selection to that leaf's bandit.
- Falls back to global Thompson if no Splitter leaf is available (e.g. early
  in the run).
- Respects pending evaluation budget accounting via the inherited
  ``_collect_points_safely`` and ``evaluators.outstanding`` logic.
- No changes to Splitter.Box or existing heuristics — minimal surface area.

This is the first step toward a full per-region adaptive portfolio. Future
work can add per-leaf heuristic subsets, region-specific config, or
hierarchical borrowing from parent boxes.

See AGENTS.md, benchmark_harness.py, and planning/SELF_IMPROVEMENT_LOOP.md
for measurement requirements.
"""

from __future__ import annotations

import numpy as np
from collections import defaultdict
from typing import Any, Dict, List, Optional

from panobbgo.core import StrategyBase, Result


class StrategyPerRegionBandit(StrategyBase):
    """
    Per-region Thompson Sampling bandit.

    Maintains an independent Beta-Bernoulli bandit (alpha/beta derived from
    accumulated reward and selection count) for each leaf box returned by
    the Splitter analyzer's ``get_leaf()`` / ``biggest_leaf``.

    Selection always queries the bandit belonging to the *current biggest
    leaf* (the one with largest log-volume, i.e. the least-explored region).
    This biases the search toward under-explored spatial regions while
    letting the per-region bandit decide *which heuristic* to use there.

    Inherits all budget accounting, Dask/threaded evaluation, constraint
    handling, and event-bus integration from StrategyBase.
    """

    def __init__(self, problem, **kwargs):
        """
        Initialize the per-region bandit strategy.

        Args:
            problem: The optimization problem instance.
            **kwargs: Passed to StrategyBase (max_eval, config overrides, etc.).
        """
        self.last_best: Optional[Result] = None
        self.total_selections: int = 0

        # Per-leaf bandit state: leaf_id -> heuristic_name -> (total_reward, count)
        # We key by leaf.id (unique per Splitter.Box) so splits create fresh
        # bandits for child regions.
        self._leaf_bandits: Dict[int, Dict[str, Dict[str, float]]] = defaultdict(
            lambda: defaultdict(lambda: {"total_reward": 0.0, "count": 0})
        )

        # Cache of leaf objects for quick lookup
        self._leaf_cache: Dict[int, Any] = {}

        StrategyBase.__init__(self, problem, **kwargs)

    def add_heuristic(self, h):
        """Register heuristic and ensure it has bandit stats (global fallback)."""
        StrategyBase.add_heuristic(self, h)
        # Global fallback stats (used if no Splitter leaf yet)
        if not hasattr(h, "ts_total_reward"):
            h.ts_total_reward = 0.0
            h.ts_counts = 0

    def _get_reward(self, best: Result) -> float:
        """
        Compute bounded reward [0,1] from improvement over current best.

        Matches the reward function used in StrategyThompsonSampling and
        StrategyUCB for consistency with the self-improvement ledger.
        """
        if self.last_best is None:
            return 1.0

        improvement = self.constraint_handler.calculate_improvement(self.last_best, best)
        # Saturating transform used across bandit strategies
        reward = 1.0 - np.exp(-1.0 * improvement)
        return max(0.0, reward)

    def on_new_best(self, best: Result) -> None:
        """
        Update the bandit statistics for the heuristic that produced the new best.

        The leaf is determined from the Splitter using the result's location.
        If no Splitter is present (or no leaf yet), falls back to global stats.
        """
        reward = self._get_reward(best)
        self.last_best = best

        try:
            splitter = self.analyzer("Splitter")
            leaf = splitter.get_leaf(best)
            leaf_id = getattr(leaf, "id", 0)

            # Update per-leaf bandit for the heuristic that produced this best
            who = best.who.split(":")[0]  # strip any :g3:i0 suffix
            bandit = self._leaf_bandits[leaf_id][who]
            bandit["total_reward"] += reward
            bandit["count"] += 1  # count the successful pull

            self._leaf_cache[leaf_id] = leaf

            self.logger.info(
                f"PerRegionBandit: leaf={leaf_id} (depth={getattr(leaf, 'depth', 0)}) "
                f"updated {who}: reward={reward:.4f} → total={bandit['total_reward']:.3f} "
                f"count={bandit['count']}"
            )
        except (KeyError, AttributeError, Exception):
            # Fallback: update global heuristic stats (early-run or no Splitter)
            try:
                h = self.heuristic(best.who)
                if hasattr(h, "ts_total_reward"):
                    h.ts_total_reward = getattr(h, "ts_total_reward", 0.0) + reward
                    h.ts_counts = getattr(h, "ts_counts", 0) + 1
            except Exception:
                pass

        self.logger.info("\u2318 %s | Δ %.7f %s (PerRegionBandit)" % (best, reward, best.who))

    def _get_status_info(self) -> Dict[str, str]:
        """Return strategy-specific status for logging (active leaves, etc.)."""
        info = {}
        active_leaves = len(self._leaf_bandits)
        if active_leaves > 0:
            info["regions"] = f"{active_leaves} leaves"
            # Report the biggest leaf's bandit if available
            try:
                splitter = self.analyzer("Splitter")
                biggest = getattr(splitter, "biggest_leaf", None)
                if biggest is not None:
                    leaf_id = getattr(biggest, "id", 0)
                    info["biggest_leaf"] = f"id={leaf_id} depth={getattr(biggest, 'depth', 0)}"
            except Exception:
                pass
        return info

    def execute(self):
        """
        Main point-generation loop.

        Queries the current biggest leaf (via Splitter.biggest_leaf or best_box)
        and uses that leaf's per-region bandit to select which heuristic to
        pull from. Falls back to global Thompson sampling if no leaf info yet.

        Respects the pending-evaluation budget via the inherited
        ``_collect_points_safely`` mechanism.
        """
        points = []
        target = self.jobs_per_client * len(self.evaluators)

        if len(self.evaluators.outstanding) < target:

            def until(points, target):
                return len(self.evaluators.outstanding) + len(points) >= target

            def selector():
                heurs = self.heuristics
                if not heurs:
                    return None

                # Try to get current region (prefer biggest_leaf for exploration bias)
                leaf_id = 0
                try:
                    splitter = self.analyzer("Splitter")
                    # biggest_leaf is the least-explored (largest volume) region
                    current_leaf = getattr(splitter, "biggest_leaf", None)
                    if current_leaf is None:
                        current_leaf = getattr(splitter, "best_box", None)
                    if current_leaf is not None:
                        leaf_id = getattr(current_leaf, "id", 0)
                        self._leaf_cache[leaf_id] = current_leaf
                except Exception:
                    # No Splitter yet or error — use global fallback (leaf_id=0)
                    pass

                # Build per-heuristic scores from the selected leaf's bandit
                samples = []
                for h in heurs:
                    hname = h.name
                    bandit = self._leaf_bandits[leaf_id][hname]

                    alpha = 1.0 + bandit["total_reward"]
                    beta = 1.0 + max(0.0, bandit["count"] - bandit["total_reward"])

                    # Thompson sample
                    theta = np.random.beta(alpha, beta)
                    samples.append((theta, h, alpha, beta))

                # Pick highest sampled heuristic
                samples.sort(key=lambda x: x[0], reverse=True)

                for _, h, alpha, beta in samples:
                    new_points = h.get_points(1)
                    if new_points:
                        count = len(new_points)
                        # Update selection count immediately (increases beta)
                        self._leaf_bandits[leaf_id][h.name]["count"] += count
                        self.total_selections += count

                        self.logger.debug(
                            f"PerRegionBandit: leaf={leaf_id} selected {h.name} "
                            f"(θ≈{samples[0][0]:.3f}, α={alpha:.1f}, β={beta:.1f})"
                        )
                        return new_points

                return []

            points = self._collect_points_safely(target, selector, until=until)

        return points

    def on_new_results(self, results: List[Result]) -> None:
        """
        Forward to base class (if it implements the method) and optionally
        update per-leaf stats for non-best results (future extension).
        Prototype keeps updates in on_new_best only for simplicity.
        """
        # StrategyBase does not declare on_new_results in its stub;
        # we forward only if the concrete parent implements it.
        if hasattr(StrategyBase, "on_new_results") or hasattr(self, "_on_new_results"):
            # Use direct call to avoid super() attribute warning in static check
            StrategyBase.on_new_results(self, results) if hasattr(StrategyBase, "on_new_results") else None

        # Optional: could extend with near-best spatial rewards per leaf,
        # but prototype keeps it simple — only on_new_best updates bandits.
        pass
