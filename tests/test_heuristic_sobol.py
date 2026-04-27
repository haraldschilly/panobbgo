# -*- coding: utf8 -*-
# Copyright 2012 -- 2026 Harald Schilly <harald.schilly@gmail.com>
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

"""Tests for the Sobol' quasi-random heuristic."""

from __future__ import annotations

import numpy as np
import pytest

from panobbgo.utils import PanobbgoTestCase


class SobolHeuristicTests(PanobbgoTestCase):
    """Construction-time validation, sampling primitives, and emit behaviour.

    Inherits the :class:`~panobbgo.utils.PanobbgoTestCase` mock-strategy
    + ``Rosenbrock(2)`` problem fixture used by every other heuristic test.
    """

    def setUp(self):
        super().setUp()
        from panobbgo.lib.constraints import DefaultConstraintHandler

        self.strategy.constraint_handler = DefaultConstraintHandler(self.strategy)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def test_default_construction(self):
        from panobbgo.heuristics.sobol import Sobol

        s = Sobol(self.strategy)
        assert s.n == 16
        assert s.scramble is True
        assert s.seed is None
        assert s.name == "Sobol"

    def test_custom_construction(self):
        from panobbgo.heuristics.sobol import Sobol

        s = Sobol(self.strategy, n=32, scramble=False, seed=123, name="Sobol_DOE")
        assert s.n == 32
        assert s.scramble is False
        assert s.seed == 123
        assert s.name == "Sobol_DOE"

    def test_invalid_n_type(self):
        from panobbgo.heuristics.sobol import Sobol

        with pytest.raises(ValueError, match="must be an integer"):
            Sobol(self.strategy, n=8.0)  # type: ignore[arg-type]

    def test_invalid_n_value(self):
        from panobbgo.heuristics.sobol import Sobol

        with pytest.raises(ValueError, match="must be positive"):
            Sobol(self.strategy, n=0)
        with pytest.raises(ValueError, match="must be positive"):
            Sobol(self.strategy, n=-4)

    def test_invalid_scramble_type(self):
        from panobbgo.heuristics.sobol import Sobol

        with pytest.raises(ValueError, match="scramble must be a bool"):
            Sobol(self.strategy, scramble=1)  # type: ignore[arg-type]

    def test_non_power_of_two_warns_only(self):
        """``n=10`` is permitted but emits a debug log — no error."""
        from panobbgo.heuristics.sobol import Sobol

        s = Sobol(self.strategy, n=10)
        pts = s._generate()
        assert pts.shape == (10, self.problem.dim)

    # ------------------------------------------------------------------
    # Sampling primitive — scaling and uniformity
    # ------------------------------------------------------------------

    def test_generate_shape(self):
        from panobbgo.heuristics.sobol import Sobol

        s = Sobol(self.strategy, n=16, seed=0)
        pts = s._generate()
        assert pts.shape == (16, self.problem.dim)

    def test_points_inside_box(self):
        """Every sampled point must lie inside ``problem.box``."""
        from panobbgo.heuristics.sobol import Sobol

        s = Sobol(self.strategy, n=64, seed=0)
        pts = s._generate()
        lower = self.problem.box[:, 0]
        upper = self.problem.box[:, 1]
        assert np.all(pts >= lower - 1e-9)
        assert np.all(pts <= upper + 1e-9)

    def test_unscrambled_deterministic(self):
        """Two ``scramble=False`` instances produce identical sequences."""
        from panobbgo.heuristics.sobol import Sobol

        s1 = Sobol(self.strategy, n=16, scramble=False)
        s2 = Sobol(self.strategy, n=16, scramble=False)
        assert np.allclose(s1._generate(), s2._generate())

    def test_seeded_scrambled_reproducible(self):
        """Same scramble + same seed → identical points."""
        from panobbgo.heuristics.sobol import Sobol

        s1 = Sobol(self.strategy, n=16, scramble=True, seed=42)
        s2 = Sobol(self.strategy, n=16, scramble=True, seed=42)
        assert np.allclose(s1._generate(), s2._generate())

    def test_different_seeds_differ(self):
        """Scrambled samples with different seeds disagree."""
        from panobbgo.heuristics.sobol import Sobol

        s1 = Sobol(self.strategy, n=16, scramble=True, seed=1)
        s2 = Sobol(self.strategy, n=16, scramble=True, seed=2)
        assert not np.allclose(s1._generate(), s2._generate())

    def test_low_discrepancy_vs_uniform(self):
        """Sobol' star discrepancy should beat plain uniform sampling.

        We use a coarse proxy: divide each dimension into 4 equal bins and
        check that Sobol' fills bins more evenly than uniform random.  With
        ``n = 64`` Sobol' should hit each (bin per axis) far closer to its
        expected count of ``n / 4 = 16`` than a uniform sample.
        """
        from panobbgo.heuristics.sobol import Sobol

        n = 64
        rng = np.random.default_rng(seed=0)

        # Uniform baseline.
        uniform_pts = rng.uniform(
            self.problem.box[:, 0],
            self.problem.box[:, 1],
            size=(n, self.problem.dim),
        )
        s = Sobol(self.strategy, n=n, scramble=True, seed=0)
        sobol_pts = s._generate()

        def _max_bin_imbalance(pts: np.ndarray, n_bins: int = 4) -> float:
            ranges = self.problem.box[:, 1] - self.problem.box[:, 0]
            normalized = (pts - self.problem.box[:, 0]) / ranges
            normalized = np.clip(normalized, 0.0, 1.0 - 1e-12)
            bins = np.floor(normalized * n_bins).astype(int)
            expected = pts.shape[0] / n_bins
            worst = 0.0
            for axis in range(pts.shape[1]):
                counts = np.bincount(bins[:, axis], minlength=n_bins)
                worst = max(worst, float(np.max(np.abs(counts - expected))))
            return worst

        sobol_imbalance = _max_bin_imbalance(sobol_pts)
        uniform_imbalance = _max_bin_imbalance(uniform_pts)
        assert sobol_imbalance <= uniform_imbalance, (
            f"Sobol imbalance {sobol_imbalance} should be <= uniform imbalance {uniform_imbalance}"
        )

    # ------------------------------------------------------------------
    # on_start emits to the output queue
    # ------------------------------------------------------------------

    def test_on_start_emits_n_points(self):
        from panobbgo.heuristics.sobol import Sobol

        n = 8
        s = Sobol(self.strategy, n=n, scramble=False)
        s.on_start()

        emitted = s.get_points()
        assert len(emitted) == n
        for pt in emitted:
            # Each emitted Point's x array must be inside the box.
            assert np.all(pt.x >= self.problem.box[:, 0] - 1e-9)
            assert np.all(pt.x <= self.problem.box[:, 1] + 1e-9)
            # The Point should be tagged with the heuristic's name.
            assert pt.who == "Sobol"

    def test_on_start_handles_arbitrary_n(self):
        """Non-power-of-two ``n`` works through ``random()`` not ``random_base2``."""
        from panobbgo.heuristics.sobol import Sobol

        s = Sobol(self.strategy, n=11, scramble=False)
        pts = s._generate()
        assert pts.shape == (11, self.problem.dim)
        # Still inside the box.
        assert np.all(pts >= self.problem.box[:, 0] - 1e-9)
        assert np.all(pts <= self.problem.box[:, 1] + 1e-9)

    # ------------------------------------------------------------------
    # Higher-dimensional sampling
    # ------------------------------------------------------------------

    def test_higher_dim_problem(self):
        """Make sure the heuristic transparently scales with ``problem.dim``."""
        from panobbgo.heuristics.sobol import Sobol
        from panobbgo.lib.classic import Rosenbrock

        higher = Rosenbrock(5)
        from panobbgo.lib.constraints import DefaultConstraintHandler

        # Reuse mock strategy but swap the problem.
        self.strategy.problem = higher
        self.strategy.constraint_handler = DefaultConstraintHandler(self.strategy)

        s = Sobol(self.strategy, n=32, seed=0)
        pts = s._generate()
        assert pts.shape == (32, 5)
        assert np.all(pts >= higher.box[:, 0] - 1e-9)
        assert np.all(pts <= higher.box[:, 1] + 1e-9)


# ----------------------------------------------------------------------
# Module-level registration
# ----------------------------------------------------------------------


def test_sobol_in_heuristics_init():
    """Top-level ``panobbgo.heuristics`` package exports ``Sobol``."""
    from panobbgo import heuristics

    assert hasattr(heuristics, "Sobol")
    assert "Sobol" in heuristics.__all__
