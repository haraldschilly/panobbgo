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

"""
Bandit selection policies shared by the strategies
==================================================

The per-heuristic statistics, the one-pull selectors and the reward rules of
:class:`~.ucb.StrategyUCB`, :class:`~.thompson.StrategyThompsonSampling`,
:class:`~.contextual.StrategyLinUCB` and :class:`~.rewarding.StrategyRewarding`.
:class:`~.phased.StrategyPhased` runs the same policies inside a phase, so both
call these functions instead of keeping two copies.

A *selector* returns ``None`` when there is no heuristic to ask (the stop signal
of :meth:`~panobbgo.core.StrategyBase._collect_points_safely`), else the points
of one pull (possibly empty).  Selectors that count pulls update
``owner.total_selections``.
"""

from __future__ import annotations

import numpy as np

from panobbgo.core import known_budget

#: Context dimension of LinUCB: (bias, budget progress, recent success rate).
LINUCB_DIM = 3

#: Default LinUCB exploration weight ``α``, shared by
#: :class:`~.contextual.StrategyLinUCB` and a LinUCB phase of
#: :class:`~.phased.StrategyPhased`.
LINUCB_ALPHA = 2.0

#: Length of the recent-reward window behind the LinUCB success-rate feature.
LINUCB_RECENT = 100


# ── Per-heuristic statistics ──


def init_ucb(h) -> None:
    """Fresh UCB1 statistics: points generated and accumulated reward."""
    h.ucb_count = 0
    h.ucb_total_reward = 0.0


def init_thompson(h) -> None:
    """Fresh Beta(1, 1) Thompson statistics."""
    h.ts_counts = 0
    h.ts_total_reward = 0.0
    h.ts_alpha = 1.0
    h.ts_beta = 1.0


def init_linucb(h, d: int = LINUCB_DIM) -> None:
    """Fresh disjoint-LinUCB model: ``A = I``, ``b = 0``."""
    h.linucb_A = np.eye(d)
    h.linucb_b = np.zeros(d)
    h.linucb_A_inv = np.eye(d)


def init_rewarding(h) -> None:
    """Fresh Rewarding statistics: ``performance = 1``, no evaluations credited."""
    h.performance = 1.0
    h.n_evals = 0


# ── Rewards ──


def improvement_reward(constraint_handler, last_best, best) -> float:
    """``1 - exp(-improvement)`` of *best* over *last_best*; 1.0 for the first best."""
    if last_best is None:
        return 1.0
    improvement = constraint_handler.calculate_improvement(last_best, best)
    return 1.0 - np.exp(-1.0 * improvement)


def linucb_reward(improvement) -> float:
    """LinUCB reward of an improvement: 0 if ``improvement <= 0``, else ``1 - exp(-improvement)``."""
    if improvement <= 0:
        return 0.0
    return 1.0 - np.exp(-1.0 * improvement)


def linucb_observe(constraint_handler, local_best, result):
    """Reward of *result* against the running *local_best*; returns ``(reward, new local_best)``.

    With no *local_best* yet the result is the first point and counts as a
    success (improvement 1.0).
    """
    if local_best is None:
        return linucb_reward(1.0), result
    improvement = constraint_handler.calculate_improvement(local_best, result)
    if constraint_handler.is_better(local_best, result):
        local_best = result
    return linucb_reward(improvement), local_best


def linucb_update(h, x, r: float) -> None:
    """Disjoint-LinUCB update of heuristic *h*: ``A += x xᵀ``, ``b += r·x``, refresh ``A⁻¹``.

    Also counts the update in ``h.linucb_count`` / ``h.linucb_reward``.
    """
    h.linucb_A += np.outer(x, x)
    h.linucb_b += r * x
    # d = 3: a full inverse is as cheap as Sherman-Morrison
    h.linucb_A_inv = np.linalg.inv(h.linucb_A)
    h.linucb_count = getattr(h, "linucb_count", 0) + 1
    h.linucb_reward = getattr(h, "linucb_reward", 0.0) + r


def push_recent(buf: list, r: float, cap: int = LINUCB_RECENT) -> None:
    """Append *r* to *buf*, dropping the oldest entry beyond *cap*."""
    buf.append(r)
    if len(buf) > cap:
        buf.pop(0)


def discount_factor(val) -> float:
    """``float(val)``, or 0.95 when *val* is ``None`` or not a number."""
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.95


def ema_discount(val) -> float:
    """The discount behind the EMA Rewarding credit: ``float(val)`` in (0, 1), else 0.95."""
    try:
        d = float(val)
    except (ValueError, TypeError):
        d = 0.95
    return d if 0.0 < d < 1.0 else 0.95


def ema_credit(constraint_handler, lookup, last_best, results, alpha: float):
    """EMA Rewarding credit: update each emitter's reward per evaluation; returns the new best.

    ``performance ← (1 - alpha)·performance + alpha·reward`` where the reward
    is ``1 - exp(-improvement)`` over *last_best* when the result improves
    it (1.0 for the first result), else 0.  ``lookup(who)`` finds the
    emitter; a result whose emitter it does not know (``KeyError``) is
    skipped entirely.
    """
    for r in results:
        try:
            h = lookup(r.who)
        except KeyError:
            continue
        if last_best is None:
            reward = 1.0
            last_best = r
        elif constraint_handler.is_better(last_best, r):
            improvement = constraint_handler.calculate_improvement(last_best, r)
            reward = max(0.0, 1.0 - float(np.exp(-improvement)))
            last_best = r
        else:
            reward = 0.0
        h.n_evals = getattr(h, "n_evals", 0) + 1
        h.performance = (1.0 - alpha) * h.performance + alpha * reward
    return last_best


def near_best_rewards(constraint_handler, problem, last_best, results):
    """Yield ``(result, reward)`` for feasible results close in value to *last_best*.

    The legacy Rewarding credit: a result within 10 % of the best value earns
    up to 0.1, more the farther it lies from the best point (normalised by the
    box ranges).  Results better than *last_best* and rewards ≤ 0.001 are
    skipped.
    """
    ranges = problem.ranges
    safe_ranges = np.where(ranges == 0, 1.0, ranges)
    for r in results:
        if constraint_handler.is_better(last_best, r):
            continue  # a new best: credited by on_new_best
        if not (last_best.cv == 0 and r.cv == 0):
            continue
        diff = abs(r.fx - last_best.fx)
        denom = abs(last_best.fx) if abs(last_best.fx) > 1e-9 else 1.0
        rel_diff = diff / denom
        if rel_diff > 0.1:
            continue
        # 1 at equal value, 0.5 at 10 % off
        value_score = 1.0 / (1.0 + 10.0 * rel_diff)
        # Reward distance from the best: other optima of similar value
        dist = np.linalg.norm((r.x - last_best.x) / safe_ranges)
        spatial_factor = 1.0 - np.exp(-10.0 * dist)
        reward = value_score * spatial_factor * 0.1
        if reward > 0.001:
            yield r, reward


# ── Selectors ──


def collect_pulls(strategy, selector, count_outstanding: bool = True) -> list:
    """Fill *strategy*'s evaluator queue by repeated calls of ``selector(target)``.

    ``target = jobs_per_client × evaluators``; nothing is pulled while that
    many evaluations are outstanding.  With *count_outstanding* the
    collection stops once outstanding plus collected points reach the target
    (one-point pulls), otherwise once the collected points alone do.
    """
    target = strategy.jobs_per_client * len(strategy.evaluators)
    if len(strategy.evaluators.outstanding) >= target:
        return []

    def until(points, target):
        return len(strategy.evaluators.outstanding) + len(points) >= target

    return strategy._collect_points_safely(target, lambda: selector(target), until=until if count_outstanding else None)


def ucb_select(owner, heurs, c: float):
    """One UCB1 pull: ask heuristics by ``Q + c·sqrt(ln N / n)`` (unpulled first)."""
    if not heurs:
        return None
    scores = []
    for h in heurs:
        if not hasattr(h, "ucb_count"):
            init_ucb(h)
        if h.ucb_count == 0:
            score = float("inf")  # force exploration of unselected arms
        else:
            average_reward = h.ucb_total_reward / h.ucb_count
            exploration_term = c * np.sqrt(np.log(max(1, owner.total_selections)) / h.ucb_count)
            score = average_reward + exploration_term
        scores.append((score, h))
    scores.sort(key=lambda x: x[0], reverse=True)
    for _score, h in scores:
        new_points = h.produce(1)
        if new_points:
            # Counted now: a pull that never becomes a best lowers Q.
            h.ucb_count += len(new_points)
            owner.total_selections += len(new_points)
            return new_points
    return []


def thompson_select(owner, heurs, rng):
    """One Thompson pull: sample ``θ ~ Beta(1 + reward, 1 + failures)`` per heuristic."""
    if not heurs:
        return None
    samples = []
    for h in heurs:
        if not hasattr(h, "ts_counts"):
            h.ts_counts = 0
            h.ts_total_reward = 0.0
        alpha = 1.0 + h.ts_total_reward
        beta = 1.0 + max(0.0, h.ts_counts - h.ts_total_reward)
        h.ts_alpha = alpha
        h.ts_beta = beta
        samples.append((rng.beta(alpha, beta), h))
    samples.sort(key=lambda x: x[0], reverse=True)
    for _theta, h in samples:
        new_points = h.produce(1)
        if new_points:
            # Attempts count at once: beta grows until the reward comes back,
            # which balances exploration in the asynchronous setting.
            h.ts_counts += len(new_points)
            owner.total_selections += len(new_points)
            return new_points
    return []


def linucb_context(n_results: int, max_eval, recent_rewards) -> np.ndarray:
    """The LinUCB context ``[1, budget progress, recent success rate]``."""
    max_evals = known_budget(max_eval) or 1000.0
    feat_progress = min(1.0, n_results / max_evals)
    if recent_rewards:
        feat_success = sum(1 for r in recent_rewards if r > 0) / len(recent_rewards)
    else:
        feat_success = 0.0
    return np.array([1.0, feat_progress, feat_success])


def linucb_select(heurs, context: np.ndarray, alpha: float, d: int = LINUCB_DIM):
    """One LinUCB pull by ``xᵀθ + α·sqrt(xᵀA⁻¹x)``; the points carry *context*."""
    if not heurs:
        return None
    scores = []
    for h in heurs:
        if not hasattr(h, "linucb_A"):
            init_linucb(h, d)
        theta = h.linucb_A_inv @ h.linucb_b
        mean = context @ theta
        variance = context @ h.linucb_A_inv @ context
        scores.append((mean + alpha * np.sqrt(variance), h))
    scores.sort(key=lambda x: x[0], reverse=True)
    for _score, h in scores:
        new_points = h.produce(1)
        if new_points:
            for p in new_points:
                p.context_vector = context
            return new_points
    return []


def ema_select(heurs, target: int, explore: float) -> list:
    """One EMA Rewarding round: probability matching with an exploration floor.

    Over the heuristics that can produce, each emits ``≈ target · p`` points
    (at least one) with ``p = (1 - explore)·perf/Σperf + explore/n``, or
    uniform when no performance is positive.
    """
    ready = [h for h in heurs if h.can_produce]
    if not ready:
        return []
    perf = np.array([max(0.0, float(h.performance)) for h in ready])
    n = len(ready)
    if perf.sum() <= 0.0:
        probs = np.full(n, 1.0 / n)
    else:
        probs = (1.0 - explore) * perf / perf.sum() + explore / n
    batch = []
    for h, p in zip(ready, probs):
        nb_h = max(1, int(round(target * p)))
        batch.extend(h.produce(nb_h))
    return batch


def rewarding_select(heurs, target: int, smooth: float, discount):
    """One legacy Rewarding round: each heuristic emits ∝ ``performance + smooth``.

    Every emitted point multiplies the emitter's ``performance`` by
    :func:`discount_factor` of *discount*.
    """
    if not heurs:
        return None
    batch = []
    perf_sum = sum(h.performance for h in heurs)
    for h in heurs:
        prob = (h.performance + smooth) / (perf_sum + smooth * len(heurs))
        nb_h = max(1, round(target * prob))
        h_pts = h.produce(nb_h)
        if h_pts:
            h.performance *= discount_factor(discount) ** len(h_pts)
            batch.extend(h_pts)
    return batch
