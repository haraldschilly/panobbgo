# -*- coding: utf8 -*-
# Copyright 2024 Panobbgo Contributors
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
import numpy as np


class FeasibleSearch(Heuristic):
    """
    Constraint-specific heuristic that targets feasible regions.

    Behavior:
    - Listens for new best results.
    - If the current best result is infeasible (cv > 0), it generates candidate points
      in an attempt to reduce constraint violation.
    - It uses a simple randomized local search around the current infeasible best,
      with adaptive step sizes (radius).
    - If the current best is feasible, this heuristic reduces its activity (emits fewer points
      or stops), as its primary goal is finding feasible regions.
    """

    def __init__(self, strategy, radius=0.1, decay=0.95, samples=5):
        """
        Args:
            strategy: Reference to the strategy.
            radius (float): Initial search radius (relative to box size).
            decay (float): Decay factor for radius when improvement is found.
            samples (int): Number of points to generate per event.
        """
        super().__init__(strategy, name="FeasibleSearch")
        self.radius = radius
        self.decay = decay
        self.samples = samples
        self.current_best = None
        self.best_feasible = None

    def on_new_best(self, best):
        """
        React to a new best point found by the strategy.
        If it's infeasible, we try to improve it (reduce CV).
        """
        self.current_best = best

        if best.cv == 0:
            self.best_feasible = best

        # If we are already feasible, we might not need to do much.
        # But let's check if we want to support finding *better* feasible points too?
        # The docstring says "targets feasible regions".
        # So primarily for when cv > 0.

        if best.cv > 0:
            self._generate_repair_points(best)
        else:
            # Maybe generate a few points near the boundary?
            # For now, let's just stay idle or generate very few.
            pass

    def on_new_results(self, results):
        """
        Listen to all results to find any feasible point, not just the best.
        """
        for r in results:
            if r.cv == 0:
                if self.best_feasible is None or r.fx < self.best_feasible.fx:
                    self.best_feasible = r

    def _generate_repair_points(self, center_result):
        """
        Generates points around the center_result to reduce CV.
        """
        x = center_result.x
        if x is None:
            return

        points = []

        # If we have a feasible point and an infeasible point,
        # search on the line between them!
        if self.best_feasible is not None:
             xf = self.best_feasible.x
             diff = xf - x

             # Generate points on the segment (biased towards feasible side?)
             # x_new = x + alpha * (xf - x)
             # alpha in (0, 1]

             for _ in range(self.samples):
                 alpha = np.random.random()
                 candidate_x = x + alpha * diff
                 points.append(candidate_x)

        else:
            # Fallback to random search if no feasible point known
            for _ in range(self.samples):
                # Generate random direction
                direction = np.random.randn(self.problem.dim)
                norm = np.linalg.norm(direction)
                if norm > 1e-9:
                    direction /= norm

                # Scale by radius and problem range
                step = direction * self.radius * self.problem.ranges

                # Create candidate
                candidate_x = x + step

                # Project to box (constraints might be outside box? No, box constraints are hard)
                candidate_x = self.problem.project(candidate_x)

                points.append(candidate_x)

        self.emit(points)
