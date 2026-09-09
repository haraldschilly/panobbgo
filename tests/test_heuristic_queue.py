# -*- coding: utf8 -*-
"""The heuristic output queue never loses points (planning/DISCOVERY_2026-09-09.md §5)."""

import numpy as np
import pytest

from panobbgo.lib.classic import Rosenbrock


def _strategy(cap=20, seed=0):
    from panobbgo.strategies import StrategyRoundRobin

    s = StrategyRoundRobin(Rosenbrock(dim=5), parse_args=False, testing_mode=True, seed=seed)
    s.config.capacity = cap
    return s


def test_emit_beyond_capacity_keeps_every_point():
    from panobbgo.heuristics import Random

    s = _strategy(cap=4)
    h = Random(s)
    pts = [np.full(5, i / 10.0) for i in range(10)]
    h.emit(pts)
    got = h.get_points()
    assert len(got) == 10
    assert [p.x[0] for p in got] == [i / 10.0 for i in range(10)]


def test_emit_rejects_non_arrays():
    from panobbgo.heuristics import Random

    h = Random(_strategy())
    with pytest.raises(TypeError):
        h.emit([[0.0] * 5])


def test_emit_none_stops_the_heuristic():
    from panobbgo.heuristics import Random

    h = Random(_strategy())
    h.emit(None)
    assert h._stopped


def test_lshade_initial_population_is_not_truncated():
    from panobbgo.heuristics import JSO

    s = _strategy(cap=20)
    h = JSO(s, NP_init=90)
    h.on_start()
    assert h._output.qsize() == 90
    assert len(h._pending) == 90
    assert len(h.get_points()) == 90


def test_pso_initial_swarm_is_not_truncated():
    from panobbgo.heuristics import PSO

    s = _strategy(cap=20)
    h = PSO(s, NP=60)
    h.on_start()
    assert h._output.qsize() == 60
