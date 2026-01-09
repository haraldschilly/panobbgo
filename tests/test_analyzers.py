# -*- coding: utf8 -*-
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
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as rnd

from panobbgo.utils import PanobbgoTestCase
from panobbgo.lib.lib import Point, Result


class AnalyzersUtils(PanobbgoTestCase):
    def setUp(self):
        from panobbgo.lib.classic import RosenbrockConstraint

        self.problem = RosenbrockConstraint(2)
        self.strategy = self.init_strategy()

    def test_best(self):
        from panobbgo.analyzers.best import Best

        best = Best(self.strategy)
        assert best is not None
        results = self.random_results(2, 10, pcv=0.99)
        N = 10
        for i in range(N):
            p = Point(rnd.random(self.problem.dim), "test")
            cv_vec = np.zeros(self.problem.dim)
            cv_vec[0] = N - (N / (i + 1))
            r = Result(p, 1.0 / (i + 1), cv_vec=cv_vec)
            results.append(r)
        # import random
        # random.shuffle(results)
        best.on_new_results(results)
        best._check_pareto_front()
        print("Pareto Front:")
        for r in best.pareto_front:
            print(r)

    def test_best_properties(self):
        """Test Best analyzer property methods."""
        from panobbgo.analyzers.best import Best

        best_analyzer = Best(self.strategy)

        # Initially should be None
        assert best_analyzer.best is None
        assert best_analyzer.cv is None
        assert best_analyzer.pareto is None
        assert best_analyzer.min is None
        assert len(best_analyzer.pareto_front) == 0

    def test_best_on_new_results_feasible(self):
        """Test Best analyzer with feasible results."""
        from panobbgo.analyzers.best import Best

        best_analyzer = Best(self.strategy)

        # Create some test results with different objective values
        results = []
        for i in range(5):
            x = np.array([float(i), float(i+1)])
            point = Point(x, f"test_{i}")
            # Make objective values decrease
            fx = 10.0 - i
            result = Result(point, fx, cv_vec=None)
            results.append(result)

        # Process results
        best_analyzer.on_new_results(results)

        # Check that best result is found
        assert best_analyzer.best is not None
        assert best_analyzer.best.fx == 6.0  # Best (lowest) objective value
        assert best_analyzer.min is not None
        assert best_analyzer.min.fx == 6.0
        assert len(best_analyzer.pareto_front) > 0

    def test_best_on_new_results_constrained(self):
        """Test Best analyzer with constrained results."""
        from panobbgo.analyzers.best import Best

        best_analyzer = Best(self.strategy)

        results = []
        for i in range(3):
            x = np.array([float(i), float(i+1)])
            point = Point(x, f"test_{i}")

            # First result is feasible (cv=0), others have constraint violations
            if i == 0:
                cv_vec = np.array([0.0, 0.0])
                fx = 5.0
            else:
                cv_vec = np.array([float(i), 0.0])  # Constraint violation
                fx = 1.0  # Better objective but infeasible

            result = Result(point, fx, cv_vec=cv_vec)
            results.append(result)

        best_analyzer.on_new_results(results)

        # Best should be the feasible one
        assert best_analyzer.best is not None
        assert best_analyzer.best.fx == 5.0
        assert best_analyzer.cv is not None
        # cv should be the result with minimal constraint violation (the feasible one)
        assert best_analyzer.cv.fx == 5.0

    def test_best_pareto_front(self):
        """Test pareto front management."""
        from panobbgo.analyzers.best import Best

        best_analyzer = Best(self.strategy)

        # Create results that form a Pareto front
        results = [
            Result(Point(np.array([0.0, 0.0]), "p1"), 5.0, cv_vec=np.array([0.0, 0.0])),  # Feasible
            Result(Point(np.array([1.0, 1.0]), "p2"), 3.0, cv_vec=np.array([2.0, 0.0])),  # Better obj, worse cv
            Result(Point(np.array([2.0, 2.0]), "p3"), 4.0, cv_vec=np.array([1.0, 0.0])),  # Middle
        ]

        best_analyzer.on_new_results(results)

        pareto_front = best_analyzer.pareto_front
        assert len(pareto_front) > 0

        # Check pareto front contains expected points
        fx_values = [r.fx for r in pareto_front]
        cv_values = [np.sum(r.cv_vec) if r.cv_vec is not None else 0 for r in pareto_front]

        # Should have non-dominated solutions
        assert len(fx_values) >= 1

if __name__ == "__main__":
    import unittest

    unittest.main()
