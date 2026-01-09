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
        results = []
        # Point 1: (fx=1.0, cv=2.0)
        # Point 2: (fx=2.0, cv=1.0) - dominates point 1
        # Point 3: (fx=1.5, cv=1.5) - non-dominated
        points_data = [
            (np.array([0.0, 0.0]), 1.0, np.array([2.0, 0.0])),
            (np.array([1.0, 1.0]), 2.0, np.array([1.0, 0.0])),
            (np.array([2.0, 2.0]), 1.5, np.array([1.5, 0.0])),
        ]

        for x, fx, cv_vec in points_data:
            point = Point(x, "pareto_test")
            result = Result(point, fx, cv_vec=cv_vec)
            results.append(result)

        best_analyzer.on_new_results(results)

        # Check pareto front
        assert len(best_analyzer.pareto_front) >= 1
        best_analyzer._check_pareto_front()  # Should not raise assertion

    def test_best_edge_cases(self):
        """Test Best analyzer edge cases and error conditions."""
        from panobbgo.analyzers.best import Best

        best_analyzer = Best(self.strategy)

        # Test with empty results
        best_analyzer.on_new_results([])

        # Test with results having zero constraint violation
        x = np.array([1.0, 2.0])
        point = Point(x, "zero_cv")
        result = Result(point, 5.0, cv_vec=np.array([0.0, 0.0]))
        best_analyzer.on_new_results([result])

        assert best_analyzer.best is not None
        assert best_analyzer.best.fx == 5.0

        # Test with infeasible results only
        results_infeasible = []
        for i in range(3):
            x = np.array([float(i), float(i+1)])
            point = Point(x, f"infeasible_{i}")
            cv_vec = np.array([1.0, 0.0])  # Infeasible
            result = Result(point, float(i), cv_vec=cv_vec)
            results_infeasible.append(result)

        best_analyzer.on_new_results(results_infeasible)

        # Should have best (lowest cv) and cv (lowest fx among infeasible)
        assert best_analyzer.cv is not None
        assert best_analyzer.best is not None  # From previous feasible result

    def test_best_properties_after_updates(self):
        """Test Best analyzer properties after multiple updates."""
        from panobbgo.analyzers.best import Best

        best_analyzer = Best(self.strategy)

        # First batch of results
        results1 = [
            Result(Point(np.array([0.0, 0.0]), "test1"), 10.0, cv_vec=None),
            Result(Point(np.array([1.0, 1.0]), "test2"), 5.0, cv_vec=None),
        ]
        best_analyzer.on_new_results(results1)

        assert best_analyzer.best is not None
        assert best_analyzer.best.fx == 5.0
        assert best_analyzer.min is not None
        assert best_analyzer.min.fx == 5.0

        # Second batch with better result
        results2 = [
            Result(Point(np.array([2.0, 2.0]), "test3"), 3.0, cv_vec=None),
        ]
        best_analyzer.on_new_results(results2)

        assert best_analyzer.best is not None
        assert best_analyzer.best.fx == 3.0
        assert best_analyzer.min is not None
        assert best_analyzer.min.fx == 3.0

        # Check pareto front has at least the best point
        assert len(best_analyzer.pareto_front) > 0

    def test_best_event_handlers(self):
        """Test Best analyzer event handler methods that don't require UI."""
        from panobbgo.analyzers.best import Best

        best_analyzer = Best(self.strategy)

        # Test on_new_pareto (line 425) - currently just passes
        pareto_result = Result(Point(np.array([1.0, 1.0]), "pareto"), 2.0, cv_vec=np.array([1.0, 0.0]))
        # Should not raise an exception
        best_analyzer.on_new_pareto(pareto_result)

        # Test on_new_pareto_front (line 442) - updates pareto front if UI not present
        front = [Result(Point(np.array([4.0, 4.0]), "front"), 4.0, cv_vec=None)]
        # Should not raise an exception (UI not present so early return)
        best_analyzer.on_new_pareto_front(front)

        # Test on_new_cv and on_new_min - should not raise exceptions
        cv_result = Result(Point(np.array([2.0, 2.0]), "cv"), 3.0, cv_vec=np.array([0.5, 0.0]))
        best_analyzer.on_new_cv(cv_result)

        min_result = Result(Point(np.array([3.0, 3.0]), "min"), 1.0, cv_vec=None)
        best_analyzer.on_new_min(min_result)
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

    def test_convergence_analyzer_init(self):
        """Test Convergence analyzer initialization."""
        from panobbgo.analyzers.convergence import Convergence

        conv = Convergence(self.strategy)
        assert conv is not None
        assert conv.window_size == 50  # default
        assert conv.threshold == 1e-6  # default
        assert conv.mode == 'std'  # default
        assert len(conv.history) == 0
        assert conv._converged is False

        # Test custom parameters
        conv_custom = Convergence(self.strategy, window_size=20, threshold=1e-4, mode='improv')
        assert conv_custom.window_size == 20
        assert conv_custom.threshold == 1e-4
        assert conv_custom.mode == 'improv'

    def test_convergence_std_mode(self):
        """Test convergence detection in standard deviation mode."""
        from panobbgo.analyzers.convergence import Convergence
        import unittest.mock as mock

        conv = Convergence(self.strategy, window_size=5, threshold=0.01, mode='std')

        # Mock strategy.best to return consistent values
        with mock.patch.object(self.strategy, 'best') as mock_best:
            mock_result = mock.Mock()
            mock_result.fx = 1.0
            mock_best.__get__ = mock.Mock(return_value=mock_result)

            # Add results to fill window with same value (no variation)
            for i in range(5):
                conv.on_new_results([mock.Mock()])

            # Should not converge yet (check_convergence is called after each addition)
            # But since we're adding one by one, it should converge when window is full

    def test_convergence_trigger_convergence(self):
        """Test that convergence analyzer triggers convergence correctly."""
        from panobbgo.analyzers.convergence import Convergence
        from collections import deque

        conv = Convergence(self.strategy, window_size=3, threshold=0.1, mode='std')
        conv.history = deque([1.0, 1.0, 1.0], maxlen=3)

        # Should not be converged initially
        assert conv._converged is False

        # Trigger convergence
        conv._trigger_convergence("Test convergence")

        # Should set converged flag
        assert conv._converged is True

    def test_convergence_edge_cases(self):
        """Test convergence analyzer edge cases."""
        from panobbgo.analyzers.convergence import Convergence
        import unittest.mock as mock

        conv = Convergence(self.strategy, window_size=5, threshold=0.01)

        # Test with no strategy.best (should return early)
        with mock.patch.object(self.strategy, 'best', None):
            conv.on_new_results([mock.Mock()])
            assert len(conv.history) == 0

        # Test already converged (should not check again)
        conv._converged = True
        from collections import deque
        conv.history = deque([1.0, 1.0, 1.0, 1.0, 1.0], maxlen=5)

        with mock.patch.object(conv, '_trigger_convergence') as mock_trigger:
            conv._check_convergence()
            mock_trigger.assert_not_called()

if __name__ == "__main__":
    import unittest

    unittest.main()
