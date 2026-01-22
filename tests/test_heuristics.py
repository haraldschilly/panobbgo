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

import numpy as np

from panobbgo.utils import PanobbgoTestCase
from panobbgo.lib import Point, Result


class HeuristicTests(PanobbgoTestCase):


    def test_random(self):
        from panobbgo.heuristics.random import Random

        rnd = Random(self.strategy)
        assert rnd is not None

    def test_latin_hypercube(self):
        from panobbgo.heuristics.latin_hypercube import LatinHypercube

        lhyp = LatinHypercube(self.strategy, 3)
        assert lhyp is not None

    def test_nelder_mead(self):
        from panobbgo.heuristics.nelder_mead import NelderMead

        nm = NelderMead(self.strategy)
        assert nm is not None
        dim = 5
        pts = self.random_results(dim, 10)
        # make it ill conditioned
        pts.insert(0, pts[0])
        base = nm.gram_schmidt(dim, pts)
        M = np.array([_.x for _ in base])
        assert np.linalg.matrix_rank(M) == dim

    def test_center(self):
        from panobbgo.heuristics.center import Center

        cntr = Center(self.strategy)
        assert cntr is not None
        # box = cntr.on_start()
        # assert np.allclose(box, [1, 0.])

    def test_extremal(self):
        from panobbgo.heuristics.extremal import Extremal

        extr = Extremal(self.strategy, prob=range(10))
        assert isinstance(extr.probabilities, np.ndarray)
        assert np.isclose(sum(np.diff(extr.probabilities)), 1.0)
        extr = Extremal(self.strategy)
        extr.__start__()

        # min, max, center, zero
        box = self.problem.box
        assert np.allclose(extr.vals[0], box[:, 0])
        assert np.allclose(extr.vals[1], np.zeros_like(extr.vals[1]))
        assert np.allclose(extr.vals[2], self.problem.center)
        assert np.allclose(extr.vals[3], box[:, 1])

        # simulate on_start, produces one point in testing mode
        extr.on_start()
        from queue import Queue

        assert isinstance(extr._output, Queue)
        p = extr._output.get()
        from panobbgo.lib import Point

        assert isinstance(p, Point)
        assert p in self.problem.box

    def test_gaussian_process(self):
        from panobbgo.heuristics.gaussian_process import GaussianProcessHeuristic

        gp = GaussianProcessHeuristic(self.strategy)
        assert gp is not None
        assert gp.acquisition_func is not None
        assert gp.kappa == 1.96
        assert gp.xi == 0.01

    def test_nearby(self):
        from panobbgo.heuristics.nearby import Nearby

        nearby = Nearby(self.strategy)
        assert nearby is not None

    def test_heuristic_point_generation_tdd(self):
        """
        TDD Test: Heuristics should not crash and should return proper data structures.

        This test defines the minimal expected behavior: heuristics should not hang
        and should return lists (even if empty).
        """
        from panobbgo.heuristics import Random, Nearby

        # Test Random heuristic basic functionality
        random_h = Random(self.strategy)
        random_h.__start__()

        # Should be able to call get_points without crashing
        points = random_h.get_points(5)

        # Should return a list (may be empty due to known issues)
        assert isinstance(points, list), "Random get_points should return a list"

        # Test Nearby heuristic basic functionality
        nearby_h = Nearby(self.strategy)
        nearby_h.__start__()

        # Should be able to call get_points without crashing
        nearby_points = nearby_h.get_points(3)

        # Should return a list (may be empty)
        assert isinstance(nearby_points, list), "Nearby get_points should return a list"

    def test_nearby_direct_functionality(self):
        """
        Test Nearby heuristic direct functionality without event system.
        """
        from panobbgo.heuristics.nearby import Nearby
        from panobbgo.lib import Point

        # Create heuristic
        nearby_h = Nearby(self.strategy, new=2, radius=0.1, axes="one")
        nearby_h.__start__()

        # Create a test best point
        best_point = Point(np.array([0.5, 0.5]), "test")

        # Directly call on_new_best to test point generation
        nearby_h.on_new_best(best_point)

        # Should have generated points
        points = nearby_h.get_points(10)

        assert isinstance(points, list), "Should return a list"
        assert len(points) >= 2, f"Should generate at least 2 points, got {len(points)}"

        # Check that points are within bounds and close to original
        for point in points:
            assert point in self.problem.box, f"Point {point.x} out of bounds"

            # Should be close to the best point (within radius * range)
            distance = np.max(np.abs(point.x - best_point.x))
            max_allowed_distance = 0.1 * np.max(self.problem.ranges)
            assert distance <= max_allowed_distance, f"Point too far: {distance} > {max_allowed_distance}"

        print(f"✅ Nearby heuristic generated {len(points)} valid points")

    def test_lbfgsb_error_handling(self):
        """Test LBFGSB heuristic error handling."""
        from panobbgo.heuristics.lbfgsb import LBFGSB

        # Should initialize without errors in test environment
        lbfgsb = LBFGSB(self.strategy)
        assert lbfgsb is not None

        # Test that __start__ method has error handling
        # (We can't fully test multiprocessing in unit tests)

    def test_quadratic_wls_model(self):
        """Test QuadraticWlsModel heuristic initialization."""
        from panobbgo.heuristics.quadratic_wls import QuadraticWlsModel

        # Should initialize without errors in test environment
        wls = QuadraticWlsModel(self.strategy)
        assert wls is not None

        # Test that __start__ method works (subprocess initialization)
        # We can't fully test the subprocess functionality in unit tests
        # but can verify it doesn't crash on init

    def test_gaussian_process_heuristic(self):
        """Test GaussianProcessHeuristic initialization and basic functionality."""
        from panobbgo.heuristics.gaussian_process import GaussianProcessHeuristic, AcquisitionFunction
        from panobbgo.lib import Point, Result
        import unittest.mock as mock

        # Test initialization with different acquisition functions
        for acq_func in [AcquisitionFunction.EI, AcquisitionFunction.UCB, AcquisitionFunction.PI]:
            gp = GaussianProcessHeuristic(self.strategy, acquisition_func=acq_func)
            assert gp is not None
            assert gp.acquisition_func == acq_func

        gp = GaussianProcessHeuristic(self.strategy)
        assert gp is not None
        gp.__start__()

        # Mock GP model for testing acquisition functions
        with mock.patch('sklearn.gaussian_process.GaussianProcessRegressor') as mock_gp:
            mock_gp_instance = mock.Mock()
            mock_gp_instance.predict.return_value = (np.array([1.0]), np.array([0.5]))
            mock_gp.return_value = mock_gp_instance

            gp.gp_model = mock_gp_instance
            gp.best_y = 0.5

            # Test acquisition function evaluation
            X_test = np.array([[0.5, 0.5]])
            acq_values = gp._evaluate_acquisition(X_test)
            assert len(acq_values) == 1

        # Test with insufficient data (should not crash)
        gp.on_new_results([])

        # Test with mock results but insufficient for GP
        mock_results = [Result(Point([0.1, 0.1], "test"), 1.0, 0.0)]
        gp.on_new_results(mock_results)

    def test_weighted_average(self):
        """Test WeightedAverage heuristic."""
        from panobbgo.heuristics.weighted_average import WeightedAverage
        from panobbgo.analyzers.splitter import Splitter
        from panobbgo.lib import Point, Result

        # Create weighted average heuristic
        wa = WeightedAverage(self.strategy, k=0.1)
        wa.__start__()

        # Create a splitter with some results
        splitter = Splitter(self.strategy)
        splitter.__start__()

        # Add some results to the splitter
        results = []
        for i in range(5):
            x = np.array([0.5, 0.5])  # Same point for all to create a box
            point = Point(x, f"wa_test_{i}")
            result = Result(point, 1.0 + i * 0.1, cv_vec=None)  # Slightly different fx
            results.append(result)

        splitter.on_new_results(results)

        # Create a "best" result
        best_point = Point(np.array([0.5, 0.5]), "best")
        best_result = Result(best_point, 1.0, cv_vec=None)

        # Should not crash when processing best result
        wa.on_new_best(best_result)

        # Should work without errors
        assert wa is not None

    def test_nelder_mead_split(self):
        """Test the split of Nelder-Mead into init and sampling phases."""
        from panobbgo.heuristics.nelder_mead import NelderMead

        nm = NelderMead(self.strategy)
        assert nm is not None
        dim = 5
        pts = self.random_results(dim, 10)
        # make it ill conditioned
        pts.insert(0, pts[0])
        base = nm.gram_schmidt(dim, pts)

        # Test init phase
        worst, centroid = nm.nelder_mead_init(base)
        assert worst is not None
        assert centroid is not None
        assert centroid.shape == (dim,)

        # Verify worst point is actually the one with highest fx
        worst_idx, expected_worst = max(enumerate(base), key=lambda _: _[1].fx)
        assert np.allclose(worst.x, expected_worst.x)
        assert worst.fx == expected_worst.fx

        # Verify centroid calculation
        others_x = [p.x for i, p in enumerate(base) if i != worst_idx]
        expected_centroid = np.average(others_x, axis=0)
        assert np.allclose(centroid, expected_centroid)

        # Test sampling phase
        new_point_x = nm.nelder_mead_sample(worst, centroid)
        assert new_point_x is not None
        assert new_point_x.shape == (dim,)

        # Test legacy wrapper
        new_point_x_legacy = nm.nelder_mead(base)
        assert new_point_x_legacy is not None
        assert new_point_x_legacy.shape == (dim,)
    def test_feasible_search(self):
        """Test FeasibleSearch heuristic functionality."""
        from panobbgo.heuristics.feasible_search import FeasibleSearch
        from panobbgo.lib import Point, Result

        # Initialize heuristic
        fs = FeasibleSearch(self.strategy)
        fs.__start__()

        # Case 1: No feasible point known, best is infeasible
        # Current best is infeasible
        best_infeasible = Result(Point(np.array([0.0, 0.0]), "test"), 0.0, cv_vec=np.array([1.0]))
        fs.on_new_best(best_infeasible)

        # Should generate random points around the best infeasible point
        points = fs.get_points(10)
        assert len(points) > 0
        for p in points:
             # Just check they are generated, simple random perturbation
             assert p in self.problem.box

        # Case 2: Feasible point found via on_new_results
        feasible_point = Result(Point(np.array([1.0, 1.0]), "test"), 0.0, cv_vec=np.array([0.0]))
        fs.on_new_results([feasible_point])
        assert fs.best_feasible is feasible_point

        # Case 3: Best is still infeasible, but we know a feasible point
        # Should generate points on the line between them
        fs.on_new_best(best_infeasible)
        points_line = fs.get_points(10)
        assert len(points_line) > 0
        # Points should be on line segment [0,0] to [1,1]
        # x = alpha * [1,1], alpha in [0,1]
        for p in points_line:
            # Check collinearity: x[0] should equal x[1] approx
            assert np.isclose(p.x[0], p.x[1])
            # Check bounds
            assert 0.0 <= p.x[0] <= 1.0

        # Case 4: Best becomes feasible
        fs.on_new_best(feasible_point)
        # Should NOT generate points (or few)
        # implementation says pass if best.cv == 0
        points_empty = fs.get_points(10)
        assert len(points_empty) == 0
    def test_constraint_gradient(self):
        """Test ConstraintGradient heuristic."""
        from panobbgo.heuristics.constraint_gradient import ConstraintGradient
        from panobbgo.lib import Point, Result

        cg = ConstraintGradient(self.strategy)
        cg.__start__()

        # Mock results for gradient estimation
        # We need dim+1 points. Let's do 2D.
        # Function: cv = x + y (gradient [1, 1])
        # x0 = [0, 0], cv=0
        # x1 = [1, 0], cv=1
        # x2 = [0, 1], cv=1

        # Best is x3 = [2, 2], cv=4 (infeasible)
        # We want to move towards [0,0] (descent)

        # Override problem with a 2D one
        from panobbgo.lib.classic import Rosenbrock
        self.strategy.problem = Rosenbrock(dims=2)
        # Re-initialize heuristic to pick up new problem from strategy
        cg = ConstraintGradient(self.strategy)
        cg.__start__()

        r1 = Result(Point(np.array([0.0, 0.0]), "t1"), 0.0, cv_vec=np.array([0.0])) # cv=0
        r2 = Result(Point(np.array([1.0, 0.0]), "t2"), 0.0, cv_vec=np.array([1.0])) # cv=1
        r3 = Result(Point(np.array([0.0, 1.0]), "t3"), 0.0, cv_vec=np.array([1.0])) # cv=1

        self.strategy.results = [r1, r2, r3]

        best = Result(Point(np.array([0.5, 0.5]), "best"), 0.0, cv_vec=np.array([1.0])) # cv=1.0 (Wait, 0.5+0.5=1)

        # Trigger
        cg.on_new_best(best)

        points = cg.get_points(10)
        assert len(points) == 1

        p = points[0]
        # Gradient of x+y is [1, 1]. Normalized: [0.7, 0.7]
        # Descent direction: [-0.7, -0.7]
        # New point should be best.x - step
        # best.x = [0.5, 0.5]
        # new_x < 0.5

        assert p.x[0] < 0.5
        assert p.x[1] < 0.5
        assert p in self.problem.box

    def test_gaussian_process_norm_cdf(self):
        """Test the _norm_cdf static method in GaussianProcessHeuristic."""
        from panobbgo.heuristics.gaussian_process import GaussianProcessHeuristic
        import numpy as np

        # Test scalar values
        assert np.isclose(GaussianProcessHeuristic._norm_cdf(0), 0.5)
        assert np.isclose(GaussianProcessHeuristic._norm_cdf(1.96), 0.9750021048517795)
        assert np.isclose(GaussianProcessHeuristic._norm_cdf(-1.96), 0.024997895148220428)

        # Test with a numpy array
        x_values = np.array([-1.96, 0, 1.96])
        cdf_values = GaussianProcessHeuristic._norm_cdf(x_values)
        expected_values = np.array([0.024997895148220428, 0.5, 0.9750021048517795])
        assert np.allclose(cdf_values, expected_values)
