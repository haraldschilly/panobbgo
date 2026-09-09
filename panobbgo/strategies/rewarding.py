# -*- coding: utf8 -*-
from __future__ import division
from __future__ import unicode_literals

# Copyright 2012 Harald Schilly <harald.schilly@gmail.com>
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

from panobbgo.core import StrategyBase
import numpy as np


class StrategyRewarding(StrategyBase):
    """
    This strategy rewards given :mod:`.heuristics` by selecting
    those more often, which produce better search points.

    Two credit-assignment modes (``credit=`` kwarg or ``config.rewarding_credit``):

    ``"ema"`` (default)
        ``performance`` is an exponential moving average of the reward
        *per evaluation spent* by that heuristic (reward ``1 - exp(-improvement)``
        when the evaluation improves the best, else 0; smoothing
        ``1 - config.discount``).  Selection is probability matching over the
        heuristics that currently have points, with an exploration floor of
        ``explore`` (``config.rewarding_explore``, default 0.2) spread evenly.

    ``"legacy"``
        Every point drawn from a heuristic multiplies its ``performance`` by
        ``config.discount`` (0.95); a new best adds ``1 - exp(-improvement)``
        to the emitter's performance.  Selection is proportional to
        ``performance + config.smooth``.  This credits the *emitter of the
        best point* rather than improvement per evaluation spent, so a
        heuristic that emits a generation of 90 points is discounted by
        ``0.95**90`` at once while a one-shot heuristic keeps its weight
        for the whole run — the strategy degenerates to a biased
        round-robin (planning/DISCOVERY_2026-09-09.md §6).

    Measured on the standard IOH battery (mean AOCC, 12-seed decision
    roster, paired per seed, same arm set): ``"ema"`` with ``explore=0.2``
    scores **+0.0135 [+0.0032, +0.0238]** over ``"legacy"`` (9/12 seeds),
    the gain sitting at 2-D (**+0.0248 [+0.0049, +0.0447]**) and flat at
    5-D (+0.0022 [-0.0030, +0.0074]).
    """

    def __init__(self, problem, credit=None, explore=None, **kwargs):
        self.last_best = None
        StrategyBase.__init__(self, problem, **kwargs)
        self.credit = str(credit if credit is not None else getattr(self.config, "rewarding_credit", "legacy"))
        if self.credit not in ("legacy", "ema"):
            raise ValueError("credit must be 'legacy' or 'ema', got %r" % self.credit)
        self.explore = float(explore if explore is not None else getattr(self.config, "rewarding_explore", 0.1))
        self._ema_alpha = 1.0 - self._discount_factor()

    def _discount_factor(self):
        try:
            d = float(self.config.discount)
        except ValueError, TypeError:
            d = 0.95
        return d if 0.0 < d < 1.0 else 0.95

    def add_heuristic(self, h):
        StrategyBase.add_heuristic(self, h)
        h.performance = 1.0
        h.n_evals = 0

    def discount(self, heur, discount=None, times=1):
        """
        Discount the given heuristic after emitting a point.

        Args:

        - ``discount``: positive float, default ``config.default``
        - ``times``: how often
        """
        # Ensure discount is a float. If None, fetch from config
        # which might return a string/float.
        val = discount if discount is not None else self.config.discount
        if val is None:
            # Fallback if config returns None, though typically it has
            # a default
            d = 0.95
        else:
            try:
                d = float(val)
            except ValueError, TypeError:
                # Handle cases where val cannot be converted to float
                d = 0.95

        d = d**times
        heur.performance *= d

    def reward(self, best):
        """
        Give this heuristic a reward (e.g. when it finds a new point)

        Args:

        - ``best``: new (best) result
        """
        if self.last_best is None:
            return 1.0

        # currently, only reward if better point found.
        # (near best values are rewarded in on_new_results)
        fx_delta, reward = 0.0, 0.0
        # fx_delta = np.log1p(self.best.fx - r.fx) # log1p ok?

        improvement = self.constraint_handler.calculate_improvement(self.last_best, best)
        fx_delta = 1.0 - np.exp(-1.0 * improvement)  # saturates to 1
        fx_delta = 0.0 if fx_delta <= 0 else fx_delta
        # if self.fx_delta_last == None: self.fx_delta_last = fx_delta
        reward = fx_delta  # / self.fx_delta_last
        self.heuristic(best.who).performance += reward
        # self.fx_delta_last = fx_delta
        return reward

    def on_new_best(self, best):
        if self.credit == "ema":
            self.logger.info("\u2318 %s by %s" % (best, best.who))
            return
        reward = self.reward(best)
        self.logger.info("\u2318 %s | \u0394 %.7f %s" % (best, reward, best.who))
        self.last_best = best

    def _get_status_info(self):
        """Return strategy-specific status info."""
        info = {}
        # Report best performance
        max_perf = 0
        best_h_name = ""
        for h in self.heuristics:
            if hasattr(h, "performance") and h.performance > max_perf:
                max_perf = h.performance
                best_h_name = h.name

        if best_h_name:
            info["best_heuristic"] = f"{best_h_name} (perf={max_perf:.2f})"
        return info

    def on_new_results(self, results):
        if self.credit == "ema":
            self._credit_ema(results)
            return
        if self.last_best is None:
            return

        # Use problem ranges for normalization
        ranges = self.problem.ranges
        # Avoid division by zero
        safe_ranges = np.where(ranges == 0, 1.0, ranges)

        for r in results:
            # Skip if r is better than last_best (handled by on_new_best)
            if self.constraint_handler.is_better(self.last_best, r):
                continue

            # Check feasibility and reward near best
            # We only reward if both are feasible for now to be safe
            # and consistent with "value"
            if self.last_best.cv == 0 and r.cv == 0:
                self._reward_near_best(r, self.last_best, safe_ranges)

    def _credit_ema(self, results):
        """Update each emitter's EMA reward-per-evaluation with this batch."""
        a = self._ema_alpha
        for r in results:
            try:
                h = self.heuristic(r.who)
            except KeyError:
                continue
            if self.last_best is None:
                reward = 1.0
                self.last_best = r
            elif self.constraint_handler.is_better(self.last_best, r):
                improvement = self.constraint_handler.calculate_improvement(self.last_best, r)
                reward = max(0.0, 1.0 - float(np.exp(-improvement)))
                self.last_best = r
            else:
                reward = 0.0
            h.n_evals += 1
            h.performance = (1.0 - a) * h.performance + a * reward

    def _reward_near_best(self, r, last_best, ranges):
        # Value closeness
        # We want to reward if r.fx is close to last_best.fx
        # Note: last_best is better, so r.fx >= last_best.fx
        diff = abs(r.fx - last_best.fx)
        denom = abs(last_best.fx) if abs(last_best.fx) > 1e-9 else 1.0
        rel_diff = diff / denom

        # If relative difference is too big (>10%), no reward.
        if rel_diff > 0.1:
            return

        # Score decreases as difference increases
        # 1.0 / (1.0 + 10 * rel_diff) maps 0 -> 1, 0.1 -> 0.5
        value_score = 1.0 / (1.0 + 10.0 * rel_diff)

        # Spatial distance
        # We want to reward points that are "far" from the best,
        # encouraging exploration of other optima with similar values.
        dist = np.linalg.norm((r.x - last_best.x) / ranges)

        # Penalty for being close. Maps 0 -> 0. As dist increases,
        # approaches 1.
        # Scale: if dist is 10% of range (0.1), we want reasonable penalty?
        # If dist=0.1, exp(-10*0.1) = exp(-1) = 0.36. Factor = 0.64.
        # If dist=0.01, exp(-0.1) = 0.9. Factor = 0.1.
        spatial_factor = 1.0 - np.exp(-10.0 * dist)

        # Combine and scale
        # We use a smaller scale than the main reward (which is ~1.0)
        # e.g. 0.1 max reward for near best.
        reward = value_score * spatial_factor * 0.1

        if reward > 0.001:
            self.heuristic(r.who).performance += reward

    def execute(self):
        points = []
        target = self.jobs_per_client * len(self.evaluators)
        # self.logger.debug(
        #     "per_client = %s | target = %s" % (self.jobs_per_client, target)
        # )
        if len(self.evaluators.outstanding) < target:
            try:
                s = float(self.config.smooth)
            except ValueError, TypeError:
                s = 0.5

            def selector():
                heurs = self.heuristics
                if not heurs:
                    return None
                if self.credit == "ema":
                    return self._select_ema(heurs, target)

                batch = []
                perf_sum = sum(h.performance for h in heurs)
                for h in heurs:
                    # calc probability based on performance with additive
                    # smoothing
                    prob = (h.performance + s) / (perf_sum + s * len(heurs))
                    nb_h = max(1, round(target * prob))
                    h_pts = h.get_points(nb_h)

                    if h_pts:
                        # Discount the heuristic performance for emitting points
                        self.discount(h, times=len(h_pts))
                        batch.extend(h_pts)

                    # print "  %16s -> %s" % (h, nb_h)
                return batch

            points = self._collect_points_safely(target, selector)

        return points

    def _select_ema(self, heurs, target):
        """Probability matching with an exploration floor over heuristics that have points."""
        ready = [h for h in heurs if h.has_points]
        if not ready:
            return []
        perf = np.array([max(0.0, float(h.performance)) for h in ready])
        n = len(ready)
        if perf.sum() <= 0.0:
            probs = np.full(n, 1.0 / n)
        else:
            probs = (1.0 - self.explore) * perf / perf.sum() + self.explore / n
        batch = []
        for h, p in zip(ready, probs):
            nb_h = max(1, int(round(target * p)))
            batch.extend(h.get_points(nb_h))
        return batch
