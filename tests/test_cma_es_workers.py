# -*- coding: utf8 -*-
# Copyright 2012-2026 Panobbgo Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""CMA-ES's worker floor on λ (``popsize_min_workers``, DISCOVERY §63).

λ is raised to the number of parallel workers so one generation fills them;
the warm start's archive fit keeps the size it has without the floor.
"""

from __future__ import annotations

from typing import Any, List

import numpy as np
import pytest

from panobbgo.core import Analyzer
from panobbgo.lib.classic import Rosenbrock
from tests.support import PanobbgoTestCase


class TestWorkerFloor(PanobbgoTestCase):
    """Unit level: a stub strategy with ``len(evaluators) == q`` (Rosenbrock(2): default λ = 6)."""

    def setUp(self) -> None:
        super().setUp()
        from panobbgo.lib.constraints import DefaultConstraintHandler

        self.strategy.constraint_handler = DefaultConstraintHandler(self.strategy)

    def _cma(self, q: int, **kw: Any) -> Any:
        from panobbgo.heuristics import CMAES

        self.strategy.evaluators = [None] * q
        cma = CMAES(self.strategy, **kw)
        cma.on_start()
        return cma

    def test_lambda_is_raised_to_the_workers(self) -> None:
        cma = self._cma(16)
        assert (cma._lam, cma._mu, cma._base_lam) == (16, 8, 16)
        assert cma._serial_lam() == 6

    def test_floor_does_not_bind_below_the_default(self) -> None:
        cma = self._cma(4)
        assert cma._lam == cma._base_lam == cma._serial_lam() == 6

    def test_explicit_popsize_and_opt_out_are_not_raised(self) -> None:
        assert self._cma(16, popsize=5)._lam == 5
        assert self._cma(16, popsize_min_workers=False)._lam == 6

    def test_ipop_grows_from_the_raised_base_and_serial_lambda_follows(self) -> None:
        cma = self._cma(16)
        cma.on_restart(self.problem.random_point(), "test")
        assert cma._lam == 32
        assert cma._serial_lam() == 12

    def test_warm_start_fits_the_serial_top_k(self) -> None:
        """The hand-off fits the archive top-6 (serial λ), not the top-16, and μ = 3 of them."""
        from panobbgo.lib import Point, Result

        cma = self._cma(16, warm_start="archive")
        rng = np.random.default_rng(0)
        pool = [Result(Point(rng.uniform(-1, 1, 2), "x"), float(i)) for i in range(40)]
        asked: List[int] = []

        def archive_seed(k: int, mode: Any = None, box: Any = None) -> List[Any]:
            asked.append(k)
            return pool[:k]

        cma.archive_seed = archive_seed  # type: ignore[method-assign]
        assert cma._warm_start_distribution()
        assert asked == [6]
        # μ = 3: the mean is the recombination of the best three seeds only.
        w = cma._recombination_weights(3)[0]
        X = np.vstack([np.asarray(r.x, dtype=float) for r in pool[:3]])
        np.testing.assert_allclose(cma._m, self.problem.project(w @ X))


class _Recorder(Analyzer):
    """Keeps every delivered result, in delivery order."""

    def __init__(self, strategy: Any) -> None:
        Analyzer.__init__(self, strategy, name="Recorder")
        self.seen: List[Any] = []

    def on_new_results(self, results: List[Any]) -> None:
        self.seen.extend(results)


def _run(q: int, *, floor: bool, max_eval: int = 120, seed: int = 4) -> List[Any]:
    """Every result of a RoundRobin CMA-ES run on ``q`` virtual workers, in delivery order."""
    from panobbgo.heuristics import CMAES
    from panobbgo.strategies import StrategyRoundRobin
    from panobbgo.virtual_clock import VirtualSpec

    s = StrategyRoundRobin(Rosenbrock(dim=2), parse_args=False, testing_mode=True, seed=seed, size=10)
    s.config.max_eval = max_eval
    s.config.stop_on_convergence = False
    VirtualSpec(workers=q, duration="lognormal").apply(s)
    s.add(CMAES, popsize_min_workers=floor)
    rec = _Recorder(s)
    s.add_analyzer(rec)
    s.start()
    return rec.seen


def _max_busy(results: List[Any]) -> int:
    return max(sum(1 for r in results if r.t_dispatch <= t < r.t_complete) for t in {r.t_dispatch for r in results})


@pytest.mark.parametrize("q", [16])
def test_one_generation_fills_the_virtual_workers(q: int) -> None:
    """On the virtual clock the floor keeps all q workers busy; without it at most ~1.5·λ = 9 are."""
    on, off = _run(q, floor=True), _run(q, floor=False)
    assert len(on) == len(off) == 120
    assert _max_busy(on) == q
    assert _max_busy(off) < q
    # The same budget finishes sooner in virtual time.
    assert max(r.t_complete for r in on) < max(r.t_complete for r in off)


def test_floor_that_does_not_bind_changes_nothing() -> None:
    """q = 4 < λ = 6: the run is bit-identical with and without the floor."""
    on, off = _run(4, floor=True), _run(4, floor=False)
    assert len(on) == len(off) == 120
    assert [tuple(r.x) for r in on] == [tuple(r.x) for r in off]
    assert [r.fx for r in on] == [r.fx for r in off]
