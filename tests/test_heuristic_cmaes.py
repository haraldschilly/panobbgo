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
"""Tests for the CMA-ES heuristic."""

from __future__ import annotations

import numpy as np
import pytest

from panobbgo.utils import PanobbgoTestCase
from panobbgo.lib import Point, Result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_result(x: np.ndarray, fx: float, who: str = "test") -> Result:
    """Create a Result for testing."""
    return Result(Point(x, who), fx)


# ---------------------------------------------------------------------------
# Test class using the PanobbgoTestCase infrastructure
# ---------------------------------------------------------------------------


class TestCMAES(PanobbgoTestCase):
    def setUp(self):
        super().setUp()
        from panobbgo.lib.constraints import DefaultConstraintHandler

        self.strategy.constraint_handler = DefaultConstraintHandler(self.strategy)

    # --- Instantiation and initialisation ---

    def test_import(self):
        from panobbgo.heuristics import CMAES

        cma = CMAES(self.strategy)
        assert cma is not None

    def test_on_start_sets_state(self):
        from panobbgo.heuristics import CMAES

        cma = CMAES(self.strategy)
        cma.on_start()

        n = self.problem.dim  # Rosenbrock 2D
        assert cma._m is not None
        assert cma._m.shape == (n,)
        assert cma._C is not None
        assert cma._C.shape == (n, n)
        assert np.allclose(cma._C, np.eye(n))
        assert cma._sigma > 0
        assert cma._lam >= 4
        assert cma._mu == cma._lam // 2
        assert cma._w is not None
        assert len(cma._w) == cma._mu
        assert np.isclose(cma._w.sum(), 1.0)

    def test_default_popsize(self):
        from panobbgo.heuristics import CMAES

        for dim in [2, 5, 10]:
            from panobbgo.lib.classic import Rosenbrock

            self.problem = Rosenbrock(dim)
            self.strategy = self.init_strategy()
            from panobbgo.lib.constraints import DefaultConstraintHandler

            self.strategy.constraint_handler = DefaultConstraintHandler(self.strategy)
            cma = CMAES(self.strategy)
            cma.on_start()
            expected_lam = 4 + int(3 * np.log(max(dim, 2)))
            assert cma._lam == expected_lam

    def test_custom_popsize(self):
        from panobbgo.heuristics import CMAES

        cma = CMAES(self.strategy, popsize=10)
        cma.on_start()
        assert cma._lam == 10
        assert cma._mu == 5

    def test_mean_initialised_at_box_centre(self):
        from panobbgo.heuristics import CMAES

        cma = CMAES(self.strategy)
        cma.on_start()
        box = self.problem.box.box
        expected_centre = 0.5 * (box[:, 0] + box[:, 1])
        assert np.allclose(cma._m, expected_centre)

    # --- Point emission ---

    def test_emits_lam_points_on_start(self):
        from panobbgo.heuristics import CMAES

        cma = CMAES(self.strategy)
        cma.on_start()
        points = cma.get_points(100)
        assert len(points) == cma._lam
        assert all(isinstance(p, Point) for p in points)

    def test_emitted_points_within_box(self):
        from panobbgo.heuristics import CMAES

        cma = CMAES(self.strategy)
        cma.on_start()
        points = cma.get_points(100)
        box = self.problem.box.box
        for p in points:
            in_box = np.all(p.x >= box[:, 0]) and np.all(p.x <= box[:, 1])
            assert in_box, f"Point {p.x} outside box {box}"

    def test_who_tags_contain_generation_info(self):
        from panobbgo.heuristics import CMAES

        cma = CMAES(self.strategy)
        cma.on_start()
        points = cma.get_points(100)
        for p in points:
            assert p.who.startswith("CMAES:g"), f"Unexpected who: {p.who}"

    # --- Result collection and update ---

    def _run_one_generation(self, cma, fx_offset: float = 0.0):
        """Drain the output queue, fake all results, and feed them back."""
        points = cma.get_points(100)
        results = []
        for p in points:
            fx = np.sum(p.x**2) + fx_offset
            r = Result(p, fx)
            results.append(r)
        cma.on_new_results(results)
        return results

    def test_update_triggers_new_emission(self):
        from panobbgo.heuristics import CMAES

        cma = CMAES(self.strategy)
        cma.on_start()
        # Drain initial gen-1 points
        pts_gen1 = cma.get_points(100)
        assert len(pts_gen1) == cma._lam

        # Feed results for gen 1 → triggers update + gen-2 emission
        results = [Result(p, float(np.sum(p.x**2))) for p in pts_gen1]
        cma.on_new_results(results)

        # A new generation should have been emitted
        pts_gen2 = cma.get_points(100)
        assert len(pts_gen2) == cma._lam

    def test_mean_shifts_toward_optimum(self):
        """After several updates with a shifted sphere, mean should move closer to optimum."""
        from panobbgo.heuristics import CMAES

        np.random.seed(42)
        cma = CMAES(self.strategy, sigma0=0.5, popsize=10)
        cma.on_start()

        # Sphere optimum at (1, 1) — not at box centre (0, 0)
        optimum = np.ones(self.problem.dim)
        initial_dist = np.linalg.norm(cma._m - optimum)

        # Run 15 generations with shifted sphere
        for _ in range(15):
            points = cma.get_points(100)
            results = [Result(p, float(np.sum((p.x - 1.0)**2))) for p in points]
            cma.on_new_results(results)

        final_dist = np.linalg.norm(cma._m - optimum)
        assert final_dist < initial_dist, (
            f"Mean did not converge: {initial_dist:.4f} -> {final_dist:.4f}"
        )

    def test_sigma_decreases_near_optimum(self):
        """Step size should decrease after CMA-ES has converged close to optimum."""
        from panobbgo.heuristics import CMAES

        np.random.seed(7)
        cma = CMAES(self.strategy, sigma0=0.3, popsize=8)
        cma.on_start()

        # Manually place mean at the optimum so CMA-ES immediately refines
        cma._m = np.ones(self.problem.dim)  # Rosenbrock optimum
        initial_sigma = cma._sigma

        # Run many generations with a tight shifted sphere centred at optimum
        for _ in range(30):
            points = cma.get_points(100)
            results = [Result(p, float(np.sum((p.x - 1.0)**2))) for p in points]
            cma.on_new_results(results)

        assert cma._sigma < initial_sigma * 0.95, (
            f"sigma did not shrink near optimum: {initial_sigma:.4f} -> {cma._sigma:.4f}"
        )

    def test_ignores_foreign_results(self):
        """Results from other heuristics should not affect CMA-ES state."""
        from panobbgo.heuristics import CMAES

        cma = CMAES(self.strategy)
        cma.on_start()
        _ = cma.get_points(100)  # drain initial gen
        m_before = cma._m.copy()

        # Feed results from a different heuristic
        foreign = [
            Result(Point(np.zeros(self.problem.dim), "RandomHeuristic"), 0.5),
            Result(Point(np.ones(self.problem.dim), "NelderMead"), 1.5),
        ]
        cma.on_new_results(foreign)

        # Mean should be unchanged
        assert np.allclose(cma._m, m_before)

    def test_handles_partial_results(self):
        """Partial generation (min_results_fraction=0.5) should still trigger update."""
        from panobbgo.heuristics import CMAES

        cma = CMAES(self.strategy, popsize=8, min_results_fraction=0.5)
        cma.on_start()
        points = cma.get_points(100)
        assert len(points) == 8

        # Feed only half the results (4 out of 8)
        partial = [Result(p, float(np.sum(p.x**2))) for p in points[:4]]
        cma.on_new_results(partial)

        # Should have emitted next generation
        next_pts = cma.get_points(100)
        assert len(next_pts) == 8

    def test_handles_inf_fx_gracefully(self):
        """Infinite fx values should be ignored."""
        from panobbgo.heuristics import CMAES

        cma = CMAES(self.strategy, popsize=6)
        cma.on_start()
        points = cma.get_points(100)

        # Mix valid and infinite results
        results = []
        for i, p in enumerate(points):
            fx = float("inf") if i % 2 == 0 else np.sum(p.x**2)
            results.append(Result(p, fx))
        # Should not raise
        cma.on_new_results(results)

    def test_covariance_stays_positive_definite(self):
        """Covariance matrix should remain positive (semi-)definite."""
        from panobbgo.heuristics import CMAES

        np.random.seed(0)
        cma = CMAES(self.strategy, sigma0=0.4, popsize=8)
        cma.on_start()

        for _ in range(15):
            points = cma.get_points(100)
            results = [Result(p, float(np.sum(p.x**2))) for p in points]
            cma.on_new_results(results)

        eigvals = np.linalg.eigvalsh(cma._C)
        assert np.all(eigvals > -1e-10), f"Negative eigenvalues: {eigvals}"

    def test_weights_sum_to_one(self):
        from panobbgo.heuristics import CMAES

        for lam in [4, 6, 10, 20]:
            cma = CMAES(self.strategy, popsize=lam)
            cma.on_start()
            assert np.isclose(cma._w.sum(), 1.0), f"Weights don't sum to 1 (lam={lam})"

    def test_chi_n_reasonable(self):
        from panobbgo.heuristics import CMAES

        cma = CMAES(self.strategy)
        cma.on_start()
        n = self.problem.dim
        # chi_n ≈ sqrt(n) for large n
        assert abs(cma._chi_n - np.sqrt(n)) < 0.5, f"chi_n={cma._chi_n}"

    def test_sigma0_fraction(self):
        """Larger sigma0 fraction should give larger initial step size."""
        from panobbgo.heuristics import CMAES

        cma1 = CMAES(self.strategy, sigma0=0.1)
        cma2 = CMAES(self.strategy, sigma0=0.8)
        cma1.on_start()
        cma2.on_start()
        assert cma2._sigma > cma1._sigma, (
            f"sigma with sigma0=0.8 ({cma2._sigma:.4f}) should exceed "
            f"sigma with sigma0=0.1 ({cma1._sigma:.4f})"
        )

    def test_exportable_from_package(self):
        """CMAES must be importable from the top-level heuristics package."""
        from panobbgo.heuristics import CMAES

        assert CMAES is not None


# ---------------------------------------------------------------------------
# Functional test: CMA-ES on Rosenbrock 2D within a real strategy
# ---------------------------------------------------------------------------


@pytest.mark.flaky(retries=3)
def test_cmaes_integration_rosenbrock():
    """End-to-end test: CMA-ES should improve upon random search on Rosenbrock 2D."""
    from panobbgo.lib.classic import Rosenbrock
    from panobbgo.strategies import StrategyRewarding
    from panobbgo.heuristics import CMAES, Random, NelderMead

    problem = Rosenbrock(2)
    np.random.seed(12345)

    with StrategyRewarding(
        problem,
        max_evaluations=150,
        evaluation_method="threaded",
    ) as strategy:
        strategy.add(Random)
        strategy.add(NelderMead)
        strategy.add(CMAES, sigma0=0.4)
        strategy.start()
        best_fx = strategy.best.fx if strategy.best else float("inf")

    # On Rosenbrock 2D the global minimum is 0 at (1,1).
    # 150 evaluations should reliably reach fx < 5 with CMA-ES in the mix.
    assert best_fx < 5.0, f"CMA-ES + NelderMead should find near-optimum; got {best_fx:.4f}"
