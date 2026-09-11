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
"""``CMAES(inject=)`` — the injection seam of ``planning/DESIGN_seams_2026-09-11.md`` §2.1.

Hansen (2011), arXiv:1110.4181: a foreign result (one this instance did not
emit) is ranked alongside the current generation's own offspring, its
Mahalanobis-normalised step clipped to ``c_y`` so it cannot drag the mean,
the step-size path or the covariance update further than one clipped step
would.  The invariants pinned here match the design's list exactly:

* ``inject=True`` alone is byte-identical to ``inject=False`` — no foreign
  points ever arrive in a solo run, so the seam is inert without a second
  arm (also pinned as a dead-parameter allowlist entry in
  ``tests/test_invariants.py``).
* the clipping helper rescales a far-outside point to norm ``c_y`` exactly.
* a foreign point better than every offspring is selected (rank 0) and
  pulls the mean toward it.
* the per-generation cap keeps exactly ``inject_max`` points, the best ones.
"""

from __future__ import annotations

import numpy as np
import pytest

from panobbgo.lib import Point, Result
from panobbgo.lib.constraints import DefaultConstraintHandler
from panobbgo.utils import PanobbgoTestCase


def _strategy(seed=1234, max_eval=80, popsize=8):
    from panobbgo.heuristics import CMAES
    from panobbgo.lib.classic import Rosenbrock
    from panobbgo.strategies import StrategyRoundRobin

    s = StrategyRoundRobin(Rosenbrock(dim=2), parse_args=False, testing_mode=True, seed=seed)
    s.config.max_eval = max_eval
    s.config.sync_evaluation = True
    s.config.stop_on_convergence = False
    s.add_heuristic(CMAES(s, popsize=popsize, inject=False))
    return s


def _run(seed=1234, max_eval=80, popsize=8, inject=False):
    from panobbgo.heuristics import CMAES
    from panobbgo.lib.classic import Rosenbrock
    from panobbgo.strategies import StrategyRoundRobin

    s = StrategyRoundRobin(Rosenbrock(dim=2), parse_args=False, testing_mode=True, seed=seed)
    s.config.max_eval = max_eval
    s.config.sync_evaluation = True
    s.config.stop_on_convergence = False
    s.add_heuristic(CMAES(s, popsize=popsize, inject=inject))
    s.start()
    df = s.results.results
    assert df is not None and len(df) >= max_eval
    return (
        df["x"].to_numpy(dtype=float),
        df["fx"].to_numpy(dtype=float).ravel(),
        df["who"].to_numpy().ravel().astype(str),
    )


def _assert_same(a, b):
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    assert list(a[2]) == list(b[2])


# ----------------------------------------------------------------------
# (1) identity: inert alone
# ----------------------------------------------------------------------


def test_inject_alone_is_byte_identical_to_no_inject():
    """No foreign points ever arrive in a solo run -- the seam must be inert."""
    _assert_same(_run(inject=False), _run(inject=True))


def test_inject_alone_is_byte_identical_across_seeds_too():
    """The identity must hold beyond one lucky seed."""
    for seed in (1, 2, 99):
        _assert_same(_run(seed=seed, inject=False), _run(seed=seed, inject=True))


# ----------------------------------------------------------------------
# (2) the clipping helper
# ----------------------------------------------------------------------


def test_clip_injected_leaves_a_short_step_alone():
    from panobbgo.heuristics import CMAES

    n = 3
    B = np.eye(n)
    D = np.ones(n)
    c_y = float(np.sqrt(n) + 2.0 * n / (n + 2.0))
    y = np.array([0.1, -0.2, 0.05])
    out = CMAES._clip_injected(y, B, D, c_y)
    np.testing.assert_array_equal(out, y)


def test_clip_injected_rescales_a_far_step_to_norm_c_y():
    from panobbgo.heuristics import CMAES

    n = 4
    rng = np.random.default_rng(0)
    # A non-trivial orthonormal basis and positive scales, so the Mahalanobis
    # norm is not just the Euclidean one.
    B, _ = np.linalg.qr(rng.standard_normal((n, n)))
    D = np.array([0.5, 1.0, 2.0, 3.0])
    c_y = float(np.sqrt(n) + 2.0 * n / (n + 2.0))

    y = np.array([50.0, -30.0, 10.0, -5.0])  # far outside any sane support
    clipped = CMAES._clip_injected(y, B, D, c_y)

    z = (1.0 / D) * (B.T @ clipped)
    assert np.linalg.norm(z) == pytest.approx(c_y, rel=1e-9)
    # direction preserved
    z_orig = (1.0 / D) * (B.T @ y)
    np.testing.assert_allclose(z / np.linalg.norm(z), z_orig / np.linalg.norm(z_orig))


# ----------------------------------------------------------------------
# (3) + (4): direct on the heuristic (PanobbgoTestCase infrastructure)
# ----------------------------------------------------------------------


class TestCMAESInject(PanobbgoTestCase):
    def setUp(self):
        super().setUp()
        self.strategy.constraint_handler = DefaultConstraintHandler(self.strategy)

    def _make(self, **kw):
        from panobbgo.heuristics import CMAES

        return CMAES(self.strategy, inject=True, **kw)

    def test_far_foreign_point_is_clipped_on_arrival(self):
        cma = self._make(popsize=8)
        cma.on_start()
        cma.get_points()

        n = self.strategy.problem.dim
        c_y = float(np.sqrt(n) + 2.0 * n / (n + 2.0))

        far = Result(Point(np.array([1000.0, -1000.0]), "OTHER:x"), -5.0)
        cma.on_new_results([far])

        gen = min(cma._gen_results.keys())
        entry = cma._injected[gen][0]
        z = (1.0 / cma._D) * (cma._B.T @ entry["y"])
        assert np.linalg.norm(z) == pytest.approx(c_y, rel=1e-6)
        # the stored position is recomputed from the clipped y, not x_f
        np.testing.assert_allclose(entry["x"], cma._m + cma._sigma * entry["y"])

    def test_better_foreign_point_is_selected_and_pulls_the_mean(self):
        cma = self._make(popsize=6, min_results_fraction=0.5)
        cma.on_start()
        pts = cma.get_points()

        m0 = cma._m.copy()
        target = cma.problem.project(m0 + 0.5 * np.ones(cma.problem.dim))
        foreign = Result(Point(target, "OTHER:x"), -1e6)
        cma.on_new_results([foreign])

        # mediocre own offspring -- worse than the injected point
        for i, p in enumerate(pts):
            cma.on_new_results([Result(p, 100.0 + i)])
            if not cma._gen_results:
                break

        assert cma._best_fx == pytest.approx(-1e6)
        np.testing.assert_allclose(cma._best_x, target)
        d0 = float(np.linalg.norm(m0 - target))
        d1 = float(np.linalg.norm(cma._m - target))
        assert d1 < d0, "the mean must move toward the injected point"

    def test_non_finite_and_infinite_penalty_foreign_points_are_ignored(self):
        cma = self._make(popsize=8)
        cma.on_start()
        cma.get_points()

        bad_fx = Result(Point(np.zeros(2), "OTHER:x"), float("nan"))
        cma.on_new_results([bad_fx])
        gen = min(cma._gen_results.keys())
        assert cma._injected.get(gen, []) == []

    def test_injection_cap_keeps_exactly_inject_max_best_points(self):
        cma = self._make(popsize=8)  # lam=8 -> inject_max = max(1, 8 // 4) = 2
        cma.on_start()
        cma.get_points()
        inject_max = max(1, cma._lam // 4)
        assert inject_max == 2

        # 3 + inject_max foreign points arrive; penalties increasingly better.
        n_foreign = inject_max + 3
        for i in range(n_foreign):
            r = Result(Point(np.array([float(i) * 1e-4, 0.0]), "OTHER:x%d" % i), float(-i))
            cma.on_new_results([r])

        gen = min(cma._gen_results.keys())
        bucket = cma._injected[gen]
        assert len(bucket) == inject_max
        # the best `inject_max` by penalty are the two most negative: the
        # last two points offered (penalties -(n_foreign-1), -(n_foreign-2)).
        kept = sorted(d["penalty"] for d in bucket)
        expected = sorted([float(-(n_foreign - 1)), float(-(n_foreign - 2))])
        assert kept == expected

    def test_injected_list_is_cleared_on_warm_start_now(self):
        from panobbgo.analyzers import Archive

        archive = Archive(self.strategy)
        self.strategy.analyzer.side_effect = lambda name: archive if name == "Archive" else None

        cma = self._make(popsize=8, warm_start="archive")
        cma.on_start()
        cma.get_points()

        foreign = Result(Point(np.array([0.1, 0.1]), "OTHER:x"), -3.0)
        cma.on_new_results([foreign])
        gen = min(cma._gen_results.keys())
        assert cma._injected.get(gen)

        # seed the archive so the warm start actually re-fits (design §4 risk 1)
        seeds = [Result(Point(np.array([0.2, 0.2]), "SEED:%d" % i), float(i)) for i in range(6)]
        archive.on_new_results(seeds)

        assert cma.warm_start_now() is True
        assert cma._injected == {}

    def test_injected_list_is_cleared_on_restart(self):
        cma = self._make(popsize=8)
        cma.on_start()
        cma.get_points()

        foreign = Result(Point(np.array([0.1, 0.1]), "OTHER:x"), -3.0)
        cma.on_new_results([foreign])
        gen = min(cma._gen_results.keys())
        assert cma._injected.get(gen)

        cma.on_restart(np.zeros(2), reason="test")
        assert cma._injected == {}

    def test_inject_true_is_a_no_op_when_no_generation_is_open(self):
        cma = self._make(popsize=8)
        # on_start() not called yet -- no open generation
        foreign = Result(Point(np.zeros(2), "OTHER:x"), -3.0)
        cma._maybe_inject(foreign)  # must not raise
        assert cma._injected == {}


# ----------------------------------------------------------------------
# (5) instance-specific `who` tag (design §2.1 / §4 risk 3)
# ----------------------------------------------------------------------


def test_who_prefix_is_instance_specific():
    from panobbgo.heuristics import CMAES
    from panobbgo.lib.classic import Rosenbrock
    from panobbgo.strategies import StrategyRoundRobin

    s = StrategyRoundRobin(Rosenbrock(dim=2), parse_args=False, testing_mode=True, seed=1)
    s.config.sync_evaluation = True
    a = CMAES(s, popsize=6, name="CMAES_A")
    b = CMAES(s, popsize=6, name="CMAES_B")
    assert a._who_prefix == "CMAES_A:"
    assert b._who_prefix == "CMAES_B:"

    a.on_start()
    pts = a.get_points()
    assert all(p.who.startswith("CMAES_A:g") for p in pts)
    # the default (no name given) keeps the pre-existing tag
    plain = CMAES(s, popsize=6)
    assert plain._who_prefix == "CMAES:"
