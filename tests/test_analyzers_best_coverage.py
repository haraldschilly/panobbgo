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

import unittest
import unittest.mock as mock
import numpy as np
from panobbgo.utils import PanobbgoTestCase
from panobbgo.lib import Point, Result
from panobbgo.lib.constraints import DefaultConstraintHandler

class TestAnalyzersBestCoverage(PanobbgoTestCase):
    def setUp(self):
        from panobbgo.lib.classic import RosenbrockConstraint

        self.problem = RosenbrockConstraint(2)
        self.strategy = self.init_strategy()
        # Ensure strategy has a constraint handler
        self.strategy.constraint_handler = DefaultConstraintHandler(self.strategy)

    def test_on_refresh_best_updates(self):
        """Test on_refresh_best updates best point when better."""
        from panobbgo.analyzers.best import Best

        best_analyzer = Best(self.strategy)

        # Initial best point
        p1 = Point(np.array([1.0, 1.0]), "init")
        r1 = Result(p1, 10.0, cv_vec=np.array([1.0]))
        best_analyzer.on_new_results([r1])

        # Verify initial state
        self.assertIsNotNone(best_analyzer.best)
        self.assertEqual(best_analyzer.best.fx, 10.0)

        # New candidate that is better (e.g. better penalty/Lagrangian value)
        p2 = Point(np.array([0.5, 0.5]), "candidate")
        r2 = Result(p2, 5.0, cv_vec=np.array([1.0])) # Same CV, better FX

        # Track events
        events_published = []
        def side_effect(key, **kwargs):
            events_published.append(key)

        with mock.patch.object(self.strategy.eventbus, 'publish', side_effect=side_effect):
            # Call on_refresh_best
            best_analyzer.on_refresh_best([r2])

        # Verify update
        self.assertEqual(best_analyzer.best.fx, 5.0)
        self.assertEqual(best_analyzer.best.who, "candidate")

        # Verify events
        self.assertIn("new_best", events_published)
        self.assertIn("new_pareto", events_published)

    def test_on_refresh_best_no_update(self):
        """Test on_refresh_best ignores worse points."""
        from panobbgo.analyzers.best import Best

        best_analyzer = Best(self.strategy)

        # Initial best point
        p1 = Point(np.array([1.0, 1.0]), "init")
        r1 = Result(p1, 10.0, cv_vec=np.array([1.0]))
        best_analyzer.on_new_results([r1])

        # Candidate that is worse
        p2 = Point(np.array([2.0, 2.0]), "worse")
        r2 = Result(p2, 20.0, cv_vec=np.array([2.0]))

        # Track events
        events_published = []
        def side_effect(key, **kwargs):
            events_published.append(key)

        with mock.patch.object(self.strategy.eventbus, 'publish', side_effect=side_effect):
            best_analyzer.on_refresh_best([r2])

        # Verify NO update
        self.assertEqual(best_analyzer.best.fx, 10.0)

        # Verify NO events
        self.assertNotIn("new_best", events_published)
        self.assertNotIn("new_pareto", events_published)

    def test_on_refresh_best_custom_handler(self):
        """Test on_refresh_best uses strategy.constraint_handler.is_better."""
        from panobbgo.analyzers.best import Best

        best_analyzer = Best(self.strategy)

        # Initial point
        r1 = Result(Point(np.array([1.0, 1.0]), "init"), 10.0)
        best_analyzer.on_new_results([r1])

        # Candidate
        r2 = Result(Point(np.array([0.0, 0.0]), "cand"), 5.0)

        # Mock constraint handler
        with mock.patch.object(self.strategy.constraint_handler, 'is_better') as mock_is_better:
            mock_is_better.return_value = True # Force it to be better

            best_analyzer.on_refresh_best([r2])

            mock_is_better.assert_called_with(r1, r2)
            self.assertEqual(best_analyzer.best, r2)

    def test_progress_reporting(self):
        """Test that significant events trigger progress reporting."""
        from panobbgo.analyzers.best import Best

        best_analyzer = Best(self.strategy)

        # Mock _update_progress_status on strategy
        # StrategyBase might not have this method mocked easily if it's dynamic,
        # but we can check if it calls the method if it exists.

        # We need to ensure strategy has panobbgo_logger and it has progress_reporter
        # StrategyBase init creates them.

        # Mock the strategy method
        self.strategy._update_progress_status = mock.Mock()

        # 1. New Min
        r_min = Result(Point(np.array([0.0, 0.0]), "min"), 1.0)
        best_analyzer.on_new_min(r_min)
        self.strategy._update_progress_status.assert_called()
        self.strategy._update_progress_status.reset_mock()

        # 2. New CV
        r_cv = Result(Point(np.array([0.0, 0.0]), "cv"), 10.0, cv_vec=np.array([0.0]))
        best_analyzer.on_new_cv(r_cv)
        self.strategy._update_progress_status.assert_called()
        self.strategy._update_progress_status.reset_mock()

        # 3. New Pareto Front
        best_analyzer.on_new_pareto_front([])
        self.strategy._update_progress_status.assert_called()

    def test_pareto_update_complex(self):
        """Test complex Pareto front updates with dominated points."""
        from panobbgo.analyzers.best import Best

        best_analyzer = Best(self.strategy)

        # Create a sequence of points
        # Format: (fx, cv)
        points_data = [
            (10.0, 5.0),  # P1: Initial baseline
            (8.0, 5.0),   # P2: Dominates P1 (better fx, same cv) -> P1 removed
            (12.0, 2.0),  # P3: Non-dominated (worse fx, better cv) -> Added
            (9.0, 3.0),   # P4: Non-dominated vs P2 and P3 -> Added
            (13.0, 1.0),  # P5: Better CV than all, worse FX -> Added
            (11.0, 3.0),  # P6: Dominated by P4 (worse fx, same cv) -> Rejected
            (9.0, 4.0),   # P7: Dominated by P4 (same fx, worse cv) -> Rejected
        ]

        results = []
        for i, (fx, cv_val) in enumerate(points_data):
            p = Point(np.array([float(i), float(i)]), f"P{i+1}")
            cv_vec = np.array([cv_val])
            r = Result(p, fx, cv_vec=cv_vec)
            results.append(r)

        # Feed one by one to simulate optimization process
        for r in results:
            best_analyzer.on_new_results([r])

        pf = best_analyzer.pareto_front
        pf_values = [(r.fx, r.cv) for r in pf]

        # Expected Front:
        # P2: (8.0, 5.0)
        # P4: (9.0, 3.0)
        # P3: (12.0, 2.0)
        # P5: (13.0, 1.0)

        # Sorted by fx (increasing)
        expected = [
            (8.0, 5.0),
            (9.0, 3.0),
            (12.0, 2.0),
            (13.0, 1.0)
        ]

        # Extract values and compare
        pf_simple = [(r.fx, r.cv) for r in pf]

        self.assertEqual(pf_simple, expected)

        # Verify P1, P6, P7 are NOT in front
        # P1 (10, 5) dominated by P2 (8, 5)
        self.assertNotIn((10.0, 5.0), pf_simple)
        # P6 (11, 3) dominated by P4 (9, 3)
        self.assertNotIn((11.0, 3.0), pf_simple)
        # P7 (9, 4) dominated by P4 (9, 3)
        self.assertNotIn((9.0, 4.0), pf_simple)

    def test_pareto_sorting(self):
        """Test that Pareto front is always sorted by fx."""
        from panobbgo.analyzers.best import Best

        best_analyzer = Best(self.strategy)

        # Add points in random order
        data = [
            (10.0, 1.0),
            (5.0, 2.0),
            (15.0, 0.5),
            (2.0, 3.0)
        ]
        # None dominate others here (fx decreases as cv increases)

        results = []
        for fx, cv_val in data:
            r = Result(Point(np.zeros(2), "t"), fx, cv_vec=np.array([cv_val]))
            results.append(r)

        best_analyzer.on_new_results(results)

        pf = best_analyzer.pareto_front
        fx_values = [r.fx for r in pf]

        # Check sorted
        self.assertEqual(fx_values, sorted(fx_values))

        # Should be: [2.0, 5.0, 10.0, 15.0]
        self.assertEqual(fx_values, [2.0, 5.0, 10.0, 15.0])

if __name__ == "__main__":
    unittest.main()
