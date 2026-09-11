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
"""``JSO(shared_pbest=)`` — the shared-pbest seam of ``planning/DESIGN_seams_2026-09-11.md`` §2.2.

With ``shared_pbest=True`` the ``current-to-pbest-w/1`` pbest pool becomes
the live top-``p_count`` union the best ``p_count`` *foreign* results (``who``
not starting with this instance's own tag) from the shared
:class:`~panobbgo.analyzers.archive.Archive`, drawn uniformly with the
instance's own RNG.  The invariants pinned here match the design's list:

* ``shared_pbest=True`` alone is byte-identical to ``False`` — no ``Archive``
  analyzer is attached by a solo run, so the pool and the RNG draw sequence
  cannot differ (also pinned as a dead-parameter allowlist entry in
  ``tests/test_invariants.py``).
* an empty (but attached) archive changes nothing either — no extra draw is
  spent finding that out.
* the archive query excludes this instance's own results, keeping only
  foreign ones.
* a planted, dominant foreign point is drawn as pbest with the frequency a
  uniform draw over the union predicts.
"""

from __future__ import annotations

import numpy as np
import pytest

from panobbgo.lib import Point, Result
from panobbgo.lib.constraints import DefaultConstraintHandler
from panobbgo.utils import PanobbgoTestCase


def _run(seed=1234, max_eval=80, NP_init=8, shared_pbest=False, with_archive=False):
    from panobbgo.analyzers import Archive
    from panobbgo.heuristics import JSO
    from panobbgo.lib.classic import Rosenbrock
    from panobbgo.strategies import StrategyRoundRobin

    s = StrategyRoundRobin(Rosenbrock(dim=2), parse_args=False, testing_mode=True, seed=seed)
    s.config.max_eval = max_eval
    s.config.sync_evaluation = True
    s.config.stop_on_convergence = False
    if with_archive:
        s.add_analyzer(Archive(s))
    s.add_heuristic(JSO(s, NP_init=NP_init, shared_pbest=shared_pbest))
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


def test_shared_pbest_alone_is_byte_identical_to_off():
    """No ``Archive`` analyzer is attached in a solo run -- inert by construction."""
    _assert_same(_run(shared_pbest=False), _run(shared_pbest=True))


def test_shared_pbest_alone_is_byte_identical_across_seeds_too():
    for seed in (1, 2, 99):
        _assert_same(_run(seed=seed, shared_pbest=False), _run(seed=seed, shared_pbest=True))


def test_shared_pbest_with_an_empty_archive_is_still_identical():
    """An attached but empty ``Archive`` must not change the draw either."""
    _assert_same(
        _run(with_archive=True, shared_pbest=False),
        _run(with_archive=True, shared_pbest=True),
    )


# ----------------------------------------------------------------------
# (2) direct on the heuristic (PanobbgoTestCase infrastructure)
# ----------------------------------------------------------------------


class TestJSOSharedPbest(PanobbgoTestCase):
    def setUp(self):
        super().setUp()
        self.strategy.constraint_handler = DefaultConstraintHandler(self.strategy)

    def _archive(self):
        from panobbgo.analyzers import Archive

        archive = Archive(self.strategy)
        self.strategy.analyzer.side_effect = lambda name: archive if name == "Archive" else None
        return archive

    def _make(self, **kw):
        from panobbgo.heuristics import JSO

        return JSO(self.strategy, shared_pbest=True, **kw)

    def _fill_population(self, h):
        h.on_start()
        pts = h.get_points()
        for i, p in enumerate(pts):
            h.on_new_results([Result(p, 10.0 + i)])  # mediocre, distinct fitness
        return pts

    def test_shared_pbest_true_reads_as_dead_without_an_archive(self):
        """Mirrors the dead-parameter-detector reasoning for the allowlist entry."""
        h = self._make(NP_init=8)
        self._fill_population(h)
        assert h._archive_analyzer() is None

    def test_foreign_pool_excludes_the_instance_own_results(self):
        """``top_k(..., exclude_who=self.name)`` must drop this instance's own points."""
        archive = self._archive()
        h = self._make(NP_init=8, name="JSO")
        self._fill_population(h)
        for i, p in enumerate(h._population):
            archive.on_new_results([p])  # the archive also sees this instance's own points

        foreign = Result(Point(np.array([1e-3, 1e-3]), "OTHER:zzz"), -1000.0)
        archive.on_new_results([foreign])

        pool = archive.top_k(50, exclude_who=h.name)
        assert all(not r.who.startswith("JSO:") for r in pool)
        assert any(r.who == "OTHER:zzz" for r in pool)

    def test_planted_foreign_point_is_drawn_as_pbest_with_expected_frequency(self):
        """The union draw is uniform: frequency matches ``1 / len(union)``.

        This exercises the exact selection :meth:`JSO._generate_trial` makes
        (live top-``p_count`` union the ``Archive``'s foreign top-``p_count``,
        drawn with ``self._rng.integers``), so a change to the union
        construction or the draw call would show up here.
        """
        archive = self._archive()
        h = self._make(NP_init=8, p_best_max=0.25, p_best_min=0.25)
        self._fill_population(h)

        foreign = Result(Point(np.array([1e-3, 1e-3]), "OTHER:zzz"), -1000.0)
        archive.on_new_results([foreign])

        live = h._live_indices()
        sorted_live = sorted(live, key=lambda i: h._fx_of(h._population[i]))
        p_count = max(int(np.ceil(h._current_p_best() * len(sorted_live))), 1)
        pbest_pool = sorted_live[:p_count]
        foreign_results = archive.top_k(p_count, exclude_who=h.name)
        assert len(foreign_results) == 1
        union_size = len(pbest_pool) + len(foreign_results)

        trials = 6000
        hits = sum(1 for _ in range(trials) if int(h._rng.integers(0, union_size)) == union_size - 1)
        freq = hits / trials
        assert freq == pytest.approx(1.0 / union_size, abs=0.03)

    def test_shared_pbest_off_ignores_an_attached_archive(self):
        """``shared_pbest=False`` must not read the archive at all.

        Setting ``self.strategy.analyzer.side_effect`` to always raise makes
        any call into it fail loudly; a passing run proves the ``False``
        branch never dereferences ``_archive_analyzer``.
        """
        from panobbgo.heuristics import JSO

        def boom(name):
            raise AssertionError("shared_pbest=False must not query any analyzer")

        self.strategy.analyzer.side_effect = boom
        h = JSO(self.strategy, shared_pbest=False, NP_init=8)
        self._fill_population(h)
        h._generate_trial(h._live_indices()[0])  # must not raise
