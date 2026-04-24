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
Phased Strategy
===============

A meta-strategy that divides the evaluation budget into disjoint phases,
each using a different sub-strategy and set of heuristics.

This enables exploration-first approaches (e.g., Random sampling in early
phases) transitioning to model-based exploitation (e.g., Gaussian Process,
L-BFGS-B) in later phases.

All heuristics are registered on the shared event bus from the start,
so model-building heuristics accumulate data even before their phase
is active. Only the selection logic and active heuristic set change
at phase boundaries.
"""

from __future__ import annotations

import time as time_module

import numpy as np

from panobbgo.core import StrategyBase


class StrategyPhased(StrategyBase):
    """
    Phased meta-strategy that composes existing strategies across budget phases.

    Each phase specifies:
    - A fraction of the total budget (``pct``, optional on last phase)
    - A sub-strategy class and its kwargs for selection logic
    - A list of heuristic classes and their kwargs

    Example::

        StrategyPhased(problem, phases=[
            {
                "pct": 25,
                "strategy": (StrategyRoundRobin, {"size": 10}),
                "heuristics": [(Random, {}), (LatinHypercube, {})],
            },
            {
                "strategy": (StrategyRewarding, {}),
                "heuristics": [(GaussianProcessHeuristic, {}), (LBFGSB, {})],
            },
        ])

    At each phase boundary, bandit/selection statistics are reset for the
    new phase's heuristics, but domain state (GP models, accumulated data,
    etc.) is preserved.
    """

    def __init__(self, problem, phases, **kwargs):
        # Validate and store phase config before base init
        self._phase_configs = self._validate_phases(phases)
        self._current_phase = 0
        self._phase_cutoffs: list[int] = []  # computed in start()
        self._phase_heuristic_names: list[set[str]] = []
        self._phase_strategy_info: list[dict] = []

        # State for delegated strategy logic
        self.last_best = None
        self.total_selections = 0

        StrategyBase.__init__(self, problem, **kwargs)

    @staticmethod
    def _validate_phases(phases):
        """Validate the phases configuration list."""
        if not phases or not isinstance(phases, list):
            raise ValueError("phases must be a non-empty list")

        if len(phases) < 2:
            raise ValueError("StrategyPhased requires at least 2 phases")

        total_pct = 0
        configs = []
        for i, phase in enumerate(phases):
            if not isinstance(phase, dict):
                raise ValueError(f"Phase {i} must be a dict, got {type(phase)}")

            # Validate strategy
            if "strategy" not in phase:
                raise ValueError(f"Phase {i} missing 'strategy' key")
            strat = phase["strategy"]
            if not isinstance(strat, tuple) or len(strat) != 2:
                raise ValueError(f"Phase {i} 'strategy' must be a (StrategyClass, kwargs_dict) tuple")

            # Validate heuristics
            if "heuristics" not in phase:
                raise ValueError(f"Phase {i} missing 'heuristics' key")
            heurs = phase["heuristics"]
            if not isinstance(heurs, list) or len(heurs) == 0:
                raise ValueError(f"Phase {i} 'heuristics' must be a non-empty list")
            for j, h in enumerate(heurs):
                if not isinstance(h, tuple) or len(h) != 2:
                    raise ValueError(f"Phase {i} heuristic {j} must be a (HeuristicClass, kwargs_dict) tuple")

            # Validate pct
            is_last = i == len(phases) - 1
            pct = phase.get("pct")
            if pct is not None:
                pct = float(pct)
                if pct <= 0 or pct > 100:
                    raise ValueError(f"Phase {i} pct must be in (0, 100], got {pct}")
                total_pct += pct
            elif not is_last:
                raise ValueError(f"Phase {i} must have 'pct' (only the last phase may omit it)")

            configs.append({"pct": pct, "strategy": strat, "heuristics": heurs})

        if total_pct >= 100:
            raise ValueError(f"Sum of explicit pct values must be < 100, got {total_pct}")

        # Fill in last phase pct if missing
        if configs[-1]["pct"] is None:
            configs[-1]["pct"] = 100.0 - total_pct

        return configs

    def start(self):
        # Compute phase cutoffs from max_eval
        max_eval = int(self.config.max_eval) if self.config.max_eval else 1000
        cumulative = 0.0
        for cfg in self._phase_configs:
            cumulative += cfg["pct"]
            cutoff = int(max_eval * cumulative / 100.0)
            self._phase_cutoffs.append(cutoff)

        # Instantiate all heuristics for all phases, tracking which belong where
        for cfg in self._phase_configs:
            phase_names: set[str] = set()
            strat_cls, strat_kwargs = cfg["strategy"]

            for heur_cls, heur_kwargs in cfg["heuristics"]:
                h = heur_cls(self, **heur_kwargs)
                # Deduplicate: if the same heuristic name already exists
                # (e.g., same class used across phases), add to this phase's set
                if h.name not in self._heuristics:
                    self._hs.append(h)
                phase_names.add(h.name)

            self._phase_heuristic_names.append(phase_names)
            self._phase_strategy_info.append(
                {
                    "cls": strat_cls,
                    "kwargs": strat_kwargs,
                    "state": {},
                }
            )

        # Now call the base start() which registers heuristics, analyzers, etc.
        StrategyBase.start(self)

    def _phase_heuristics(self):
        """Return only the heuristics active in the current phase."""
        names = self._phase_heuristic_names[self._current_phase]
        return [h for h in self.heuristics if h.name in names]

    def _check_phase_transition(self):
        """Check if we should advance to the next phase."""
        n_results = len(self.results)
        while (
            self._current_phase < len(self._phase_cutoffs) - 1 and n_results >= self._phase_cutoffs[self._current_phase]
        ):
            old_phase = self._current_phase
            self._current_phase += 1
            new_info = self._phase_strategy_info[self._current_phase]
            self.logger.info(
                f"Phase transition: {old_phase} -> {self._current_phase} "
                f"(at {n_results} results, strategy: {new_info['cls'].__name__})"
            )
            self._init_phase_stats(self._current_phase)

    def _init_phase_stats(self, phase_idx):
        """Initialize/reset selection statistics for a phase's heuristics."""
        info = self._phase_strategy_info[phase_idx]
        strat_cls = info["cls"]
        names = self._phase_heuristic_names[phase_idx]
        self.total_selections = 0

        # Import strategy classes for isinstance checks
        from panobbgo.strategies.contextual import StrategyLinUCB
        from panobbgo.strategies.rewarding import StrategyRewarding
        from panobbgo.strategies.thompson import StrategyThompsonSampling
        from panobbgo.strategies.ucb import StrategyUCB

        for h in self.heuristics:
            if h.name not in names:
                continue

            if issubclass(strat_cls, StrategyRewarding):
                h.performance = 1.0
            elif issubclass(strat_cls, StrategyUCB):
                h.ucb_count = 0
                h.ucb_total_reward = 0.0
            elif issubclass(strat_cls, StrategyThompsonSampling):
                h.ts_counts = 0
                h.ts_total_reward = 0.0
                h.ts_alpha = 1.0
                h.ts_beta = 1.0
            elif issubclass(strat_cls, StrategyLinUCB):
                d = 3  # context dimension
                h.linucb_A = np.eye(d)
                h.linucb_b = np.zeros(d)
                h.linucb_A_inv = np.eye(d)

        # Reset phase-level state
        info["state"] = {}
        if issubclass(strat_cls, StrategyLinUCB):
            info["state"]["recent_rewards"] = []

    def add_heuristic(self, h):
        """Override to initialize stats for the first phase's strategy."""
        StrategyBase.add_heuristic(self, h)
        # Initialize stats for whatever phase this heuristic first appears in
        for i, names in enumerate(self._phase_heuristic_names):
            if h.name in names:
                self._init_phase_heuristic(h, i)
                break

    def _init_phase_heuristic(self, h, phase_idx):
        """Initialize a single heuristic's selection stats for a given phase."""
        info = self._phase_strategy_info[phase_idx]
        strat_cls = info["cls"]

        from panobbgo.strategies.contextual import StrategyLinUCB
        from panobbgo.strategies.rewarding import StrategyRewarding
        from panobbgo.strategies.thompson import StrategyThompsonSampling
        from panobbgo.strategies.ucb import StrategyUCB

        if issubclass(strat_cls, StrategyRewarding):
            h.performance = 1.0
        elif issubclass(strat_cls, StrategyUCB):
            h.ucb_count = 0
            h.ucb_total_reward = 0.0
        elif issubclass(strat_cls, StrategyThompsonSampling):
            h.ts_counts = 0
            h.ts_total_reward = 0.0
            h.ts_alpha = 1.0
            h.ts_beta = 1.0
        elif issubclass(strat_cls, StrategyLinUCB):
            d = 3
            h.linucb_A = np.eye(d)
            h.linucb_b = np.zeros(d)
            h.linucb_A_inv = np.eye(d)

    # ── Event handlers (delegated to current phase's strategy logic) ──

    def on_new_best(self, best):
        """Delegate reward logic to the current phase's strategy."""
        info = self._phase_strategy_info[self._current_phase]
        strat_cls = info["cls"]

        from panobbgo.strategies.rewarding import StrategyRewarding
        from panobbgo.strategies.thompson import StrategyThompsonSampling
        from panobbgo.strategies.ucb import StrategyUCB

        if self.last_best is None:
            reward = 1.0
        else:
            improvement = self.constraint_handler.calculate_improvement(self.last_best, best)
            reward = 1.0 - np.exp(-1.0 * improvement)

        try:
            h = self.heuristic(best.who)
        except KeyError:
            self.last_best = best
            return

        if issubclass(strat_cls, StrategyRewarding):
            h.performance += reward
        elif issubclass(strat_cls, StrategyUCB):
            if hasattr(h, "ucb_total_reward"):
                h.ucb_total_reward += reward
        elif issubclass(strat_cls, StrategyThompsonSampling):
            if hasattr(h, "ts_total_reward"):
                h.ts_total_reward += reward

        self.logger.info("\u2318 %s | \u0394 %.7f %s (Phased/%s)" % (best, reward, best.who, strat_cls.__name__))
        self.last_best = best

    def on_new_results(self, results):
        """Delegate result processing to the current phase's strategy."""
        if not results:
            return

        info = self._phase_strategy_info[self._current_phase]
        strat_cls = info["cls"]

        from panobbgo.strategies.rewarding import StrategyRewarding

        # Rewarding strategy gives small rewards for points near best
        if issubclass(strat_cls, StrategyRewarding) and self.last_best is not None:
            ranges = self.problem.ranges
            safe_ranges = np.where(ranges == 0, 1.0, ranges)

            for r in results:
                if self.constraint_handler.is_better(self.last_best, r):
                    continue
                if self.last_best.cv == 0 and r.cv == 0:
                    self._reward_near_best(r, self.last_best, safe_ranges)

    def _reward_near_best(self, r, last_best, ranges):
        """Small reward for points near the best (Rewarding strategy logic)."""
        diff = abs(r.fx - last_best.fx)
        denom = abs(last_best.fx) if abs(last_best.fx) > 1e-9 else 1.0
        rel_diff = diff / denom
        if rel_diff > 0.1:
            return
        value_score = 1.0 / (1.0 + 10.0 * rel_diff)
        dist = np.linalg.norm((r.x - last_best.x) / ranges)
        spatial_factor = 1.0 - np.exp(-10.0 * dist)
        reward = value_score * spatial_factor * 0.1
        if reward > 0.001:
            try:
                self.heuristic(r.who).performance += reward
            except KeyError:
                pass

    # ── Selection logic per strategy type ──

    def _execute_round_robin(self, phase_heurs, strat_kwargs):
        """Round-robin selection logic."""
        size = strat_kwargs.get("size", 10)
        info = self._phase_strategy_info[self._current_phase]
        state = info["state"]
        current = state.get("rr_current", 0)

        points = []
        attempts = 0
        max_attempts = 10
        while len(points) == 0 and attempts < max_attempts:
            if not phase_heurs:
                break
            current = current % len(phase_heurs)
            new_points = phase_heurs[current].get_points(size)
            points.extend(new_points)
            current = (current + 1) % len(phase_heurs)
            attempts += 1
            if len(new_points) == 0:
                time_module.sleep(1e-3)

        state["rr_current"] = current
        return points

    def _execute_rewarding(self, phase_heurs, strat_kwargs):
        """Rewarding (probability-based) selection logic."""
        target = self.jobs_per_client * len(self.evaluators)
        if len(self.evaluators.outstanding) >= target:
            return []

        try:
            s = float(self.config.smooth)
        except (ValueError, TypeError):
            s = 0.5

        discount_val = strat_kwargs.get("discount", None)

        def selector():
            if not phase_heurs:
                return None
            batch = []
            perf_sum = sum(h.performance for h in phase_heurs)
            for h in phase_heurs:
                prob = (h.performance + s) / (perf_sum + s * len(phase_heurs))
                nb_h = max(1, round(target * prob))
                h_pts = h.get_points(nb_h)
                if h_pts:
                    # Discount
                    val = discount_val if discount_val is not None else self.config.discount
                    try:
                        d = float(val)
                    except (ValueError, TypeError):
                        d = 0.95
                    h.performance *= d ** len(h_pts)
                    batch.extend(h_pts)
            return batch

        return self._collect_points_safely(target, selector)

    def _execute_ucb(self, phase_heurs, strat_kwargs):
        """UCB1 selection logic."""
        target = self.jobs_per_client * len(self.evaluators)
        if len(self.evaluators.outstanding) >= target:
            return []

        c = float(strat_kwargs.get("ucb_c", 1.414))

        def until(points, target):
            return len(self.evaluators.outstanding) + len(points) >= target

        def selector():
            if not phase_heurs:
                return None
            scores = []
            for h in phase_heurs:
                if not hasattr(h, "ucb_count"):
                    h.ucb_count = 0
                    h.ucb_total_reward = 0.0
                if h.ucb_count == 0:
                    score = float("inf")
                else:
                    avg = h.ucb_total_reward / h.ucb_count
                    exploration = c * np.sqrt(np.log(max(1, self.total_selections)) / h.ucb_count)
                    score = avg + exploration
                scores.append((score, h))
            scores.sort(key=lambda x: x[0], reverse=True)
            for _score, h in scores:
                new_points = h.get_points(1)
                if new_points:
                    h.ucb_count += len(new_points)
                    self.total_selections += len(new_points)
                    return new_points
            return []

        return self._collect_points_safely(target, selector, until=until)

    def _execute_thompson(self, phase_heurs, strat_kwargs):
        """Thompson Sampling selection logic."""
        target = self.jobs_per_client * len(self.evaluators)
        if len(self.evaluators.outstanding) >= target:
            return []

        def until(points, target):
            return len(self.evaluators.outstanding) + len(points) >= target

        def selector():
            if not phase_heurs:
                return None
            samples = []
            for h in phase_heurs:
                if not hasattr(h, "ts_counts"):
                    h.ts_counts = 0
                    h.ts_total_reward = 0.0
                alpha = 1.0 + h.ts_total_reward
                beta = 1.0 + max(0.0, h.ts_counts - h.ts_total_reward)
                theta = np.random.beta(alpha, beta)
                samples.append((theta, h))
            samples.sort(key=lambda x: x[0], reverse=True)
            for _theta, h in samples:
                new_points = h.get_points(1)
                if new_points:
                    h.ts_counts += len(new_points)
                    self.total_selections += len(new_points)
                    return new_points
            return []

        return self._collect_points_safely(target, selector, until=until)

    def _execute_linucb(self, phase_heurs, strat_kwargs):
        """LinUCB selection logic."""
        target = self.jobs_per_client * len(self.evaluators)
        if len(self.evaluators.outstanding) >= target:
            return []

        alpha = float(strat_kwargs.get("linucb_alpha", 0.5))
        context_dim = 3

        # Build context vector
        n_evals = len(self.results)
        max_evals = float(self.config.max_eval) if self.config.max_eval else 1000.0
        feat_progress = min(1.0, n_evals / max_evals)

        info = self._phase_strategy_info[self._current_phase]
        recent = info["state"].get("recent_rewards", [])
        feat_success = sum(1 for r in recent if r > 0) / len(recent) if recent else 0.0
        context = np.array([1.0, feat_progress, feat_success])

        def until(points, target):
            return len(self.evaluators.outstanding) + len(points) >= target

        def selector():
            if not phase_heurs:
                return None
            scores = []
            for h in phase_heurs:
                if not hasattr(h, "linucb_A"):
                    h.linucb_A = np.eye(context_dim)
                    h.linucb_b = np.zeros(context_dim)
                    h.linucb_A_inv = np.eye(context_dim)
                theta = h.linucb_A_inv @ h.linucb_b
                mean = context @ theta
                variance = context @ h.linucb_A_inv @ context
                score = mean + alpha * np.sqrt(variance)
                scores.append((score, h))
            scores.sort(key=lambda x: x[0], reverse=True)
            for _score, h in scores:
                new_points = h.get_points(1)
                if new_points:
                    for p in new_points:
                        p.context_vector = context
                    return new_points
            return []

        return self._collect_points_safely(target, selector, until=until)

    # ── Main execute dispatch ──

    def execute(self):
        self._check_phase_transition()

        # Cap point generation to not exceed the current phase's budget.
        # Without this, a fast-generating strategy (e.g., RoundRobin) can
        # submit so many in-flight evaluations that when they complete,
        # results jump past the next phase's cutoff — skipping it entirely.
        phase_cutoff = self._phase_cutoffs[self._current_phase]
        remaining = phase_cutoff - len(self.results) - len(self.pending)
        if remaining <= 0:
            return []

        info = self._phase_strategy_info[self._current_phase]
        strat_cls = info["cls"]
        strat_kwargs = info["kwargs"]
        phase_heurs = self._phase_heuristics()

        if not phase_heurs:
            return []

        from panobbgo.strategies.contextual import StrategyLinUCB
        from panobbgo.strategies.round_robin import StrategyRoundRobin
        from panobbgo.strategies.rewarding import StrategyRewarding
        from panobbgo.strategies.thompson import StrategyThompsonSampling
        from panobbgo.strategies.ucb import StrategyUCB

        if issubclass(strat_cls, StrategyRoundRobin):
            points = self._execute_round_robin(phase_heurs, strat_kwargs)
        elif issubclass(strat_cls, StrategyLinUCB):
            points = self._execute_linucb(phase_heurs, strat_kwargs)
        elif issubclass(strat_cls, StrategyThompsonSampling):
            points = self._execute_thompson(phase_heurs, strat_kwargs)
        elif issubclass(strat_cls, StrategyUCB):
            points = self._execute_ucb(phase_heurs, strat_kwargs)
        elif issubclass(strat_cls, StrategyRewarding):
            points = self._execute_rewarding(phase_heurs, strat_kwargs)
        else:
            raise ValueError(f"Unknown strategy class: {strat_cls}")

        return points[:remaining]

    def _get_status_info(self):
        """Return strategy-specific status info."""
        info = {}
        phase = self._current_phase
        total_phases = len(self._phase_configs)
        strat_name = self._phase_strategy_info[phase]["cls"].__name__
        info["phase"] = f"{phase + 1}/{total_phases} ({strat_name})"
        if phase < len(self._phase_cutoffs):
            info["phase_budget"] = f"until {self._phase_cutoffs[phase]} evals"
        return info
