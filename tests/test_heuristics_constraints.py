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

from __future__ import unicode_literals

import numpy as np
from panobbgo.utils import PanobbgoTestCase
from panobbgo.lib import Point, Result
from panobbgo.lib.constraints import DefaultConstraintHandler

class HeuristicConstraintsTests(PanobbgoTestCase):

    def setUp(self):
        super().setUp()
        # Ensure strategy has a constraint handler
        # DefaultConstraintHandler with rho=100.0
        self.strategy.constraint_handler = DefaultConstraintHandler(self.strategy, rho=100.0)

        # Override results to be a list for simpler testing if needed,
        # but heuristics might expect Results object.
        # Heuristics often access results from Splitter boxes or strategy.results

    def test_nelder_mead_with_constraints(self):
        from panobbgo.heuristics.nelder_mead import NelderMead

        nm = NelderMead(self.strategy)
        dim = 2

        # Create results
        # R1: Infeasible but low FX.
        # fx = 0.0, cv = 1.0. Penalty ~ 100.0 (using DefaultConstraintHandler logic approx)
        # DefaultConstraintHandler logic for infeasible vs infeasible:
        # improvement = rho * delta_cv.
        # For penalty calculation: get_penalty_value = fx + rho * cv
        # R1 penalty = 0.0 + 100.0 * 1.0 = 100.0

        x1 = np.array([0.0, 0.0])
        r1 = Result(Point(x1, "test"), 0.0, cv_vec=np.array([1.0, 0.0]))

        # R2: Feasible but higher FX.
        # fx = 10.0, cv = 0.0. Penalty = 10.0
        x2 = np.array([1.0, 1.0])
        r2 = Result(Point(x2, "test"), 10.0, cv_vec=np.array([0.0, 0.0]))

        # R3: Feasible, FX=20.0. Penalty = 20.0
        x3 = np.array([2.0, 2.0])
        r3 = Result(Point(x3, "test"), 20.0, cv_vec=np.array([0.0, 0.0]))

        # We need dim+1 = 3 points for simplex.
        # If we just give these 3 points to gram_schmidt

        results = [r1, r2, r3]

        # gram_schmidt sorts results.
        # Currently it sorts by fx.
        # Sorted by fx: r1 (0.0), r2 (10.0), r3 (20.0)
        # Sorted by penalty: r2 (10.0), r3 (20.0), r1 (100.0)

        # Currently implemented behavior (EXPECTED TO FAIL if we assert correctness)
        # But we want to confirm it fails, so we assert the CORRECT behavior and expect failure.

        base = nm.gram_schmidt(dim, results)

        # If base is None, it means gram_schmidt failed (maybe collinearity?)
        # [0,0], [1,1], [2,2] are collinear. gram_schmidt handles this by skipping.
        # So we need non-collinear points.

        x3 = np.array([0.0, 1.0])
        r3 = Result(Point(x3, "test"), 20.0, cv_vec=np.array([0.0, 0.0]))
        results = [r1, r2, r3]

        base = nm.gram_schmidt(dim, results)

        # Check first point in returned base (should be best point)
        # base returns list of Result objects? No, it returns list of Results in 'ret' variable
        # gram_schmidt returns 'ret' which is list of Results.

        best_res = base[0]

        # We want best_res to be r2 (Feasible, lowest fx among feasible)
        # r1 is infeasible with high penalty.

        # With current implementation (sort by fx), it will pick r1 (fx=0.0).
        # With fix, it should pick r2.

        assert best_res.x[0] == 1.0 and best_res.x[1] == 1.0, \
            f"Expected best result to be r2 (feasible), but got {best_res.x} with fx={best_res.fx}, cv={best_res.cv}"

    def test_weighted_average_constraints(self):
        from panobbgo.heuristics.weighted_average import WeightedAverage
        from panobbgo.analyzers.splitter import Splitter
        from panobbgo.lib.classic import Rosenbrock

        # Override problem with custom box to allow points at [10, 10]
        self.problem = Rosenbrock(dims=2, box=[(-100, 100), (-100, 100)])
        self.strategy.problem = self.problem

        wa = WeightedAverage(self.strategy)
        wa.__start__()

        # Mock Splitter box
        class MockBox:
            def __init__(self, results):
                self.results = results

        # R1: Infeasible, fx=0.0, cv=1.0 -> Penalty=100
        r1 = Result(Point(np.array([0.0, 0.0]), "test"), 0.0, cv_vec=np.array([1.0]))
        # R2: Feasible, fx=10.0, cv=0.0 -> Penalty=10
        r2 = Result(Point(np.array([10.0, 10.0]), "test"), 10.0, cv_vec=np.array([0.0]))
        # R3: Feasible, fx=10.0, cv=0.0 -> Penalty=10
        r3 = Result(Point(np.array([10.0, 10.0]), "test"), 10.0, cv_vec=np.array([0.0]))

        box = MockBox([r1, r2, r3])

        # Mock Splitter behavior
        self.strategy._analyzers = {'Splitter': None} # Just to ensure we can access it
        self.strategy.analyzer = lambda name: type('MockSplitter', (), {'get_leaf': lambda self, x: box})()

        # If we trigger on_new_best with r2 (feasible)
        # wa calculates weights based on (result.fx - best.fx)
        # weights = log1p(yy - best.fx)

        # If yy uses fx:
        # r1.fx = 0.0. best.fx = 10.0.
        # yy - best.fx = -10.0.
        # log1p(-10) -> NaN or Error.
        # Actually WeightedAverage does: weights = -weights + max...

        # If we use penalty:
        # r1 penalty = 100. best penalty = 10.
        # yy - best = 90.
        # log1p(90) = 4.5
        # This gives lower weight to r1 (because of minus sign later? check implementation)

        # Current implementation:
        # weights = np.log1p(yy - best.fx)
        # weights = -weights + (1 + self.k) * weights.max()
        # Lower yy (better fx) -> lower log1p -> higher weight (due to -weights).

        # So r1 (fx=0) would get HIGHER weight than r2 (fx=10) if using fx.
        # We want r1 to get LOWER weight because it is infeasible (high penalty).

        # Let's run it and see what happens.
        # Note: on_new_best clears output and emits points.
        # We need to capture emitted points.

        wa.on_new_best(r2)

        points = wa.get_points(10)

        # If r1 (0,0) has high weight, average will be pulled towards 0.
        # If r2 (10,10) has high weight, average will be near 10.

        avg_point = np.mean([p.x for p in points], axis=0)

        # With fx: r1 (0.0) is much better than r2 (10.0). r1 dominates. Avg near 0.
        # With penalty: r2 (10.0) is much better than r1 (100.0). r2 dominates. Avg near 10.

        assert avg_point[0] > 5.0, f"Weighted Average pulled towards infeasible point (0,0). Avg: {avg_point}"
