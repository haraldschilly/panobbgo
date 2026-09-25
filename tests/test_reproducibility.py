# -*- coding: utf8 -*-
"""A seeded strategy run must be a pure function of its seed.

Two independent runs with the same ``seed`` and ``sync_evaluation=True``
have to produce the identical sequence of evaluated points and values.
This is the property every paired benchmark comparison in the harness
relies on; see ``planning/DISCOVERY_2026-09-09.md`` §2 for the
measurements that motivated it.
"""

from __future__ import annotations

import numpy as np

from panobbgo.lib.classic import Rosenbrock


def _run(seed, max_eval=60, heuristics=None, strategy_cls=None):
    from panobbgo.heuristics import JSO, Nearby, Random
    from panobbgo.strategies import StrategyRewarding

    strategy_cls = strategy_cls or StrategyRewarding
    problem = Rosenbrock(dim=2)
    s = strategy_cls(problem, parse_args=False, seed=seed)
    s.config.max_eval = max_eval
    s.config.sync_evaluation = True
    s.config.stop_on_convergence = False
    for factory in heuristics or (
        lambda st: Random(st),
        lambda st: Nearby(st, radius=0.1, axes="all", new=3),
        lambda st: JSO(st, NP_init=8),
    ):
        s.add_heuristic(factory(s))
    s.start()
    df = s.results.results  # the public property materialises the frame
    assert df is not None and len(df) >= max_eval
    x = df["x"].to_numpy(dtype=float)
    fx = df["fx"].to_numpy(dtype=float).ravel()
    who = df["who"].to_numpy().ravel().astype(str)
    return x, fx, who, s.seed


def test_same_seed_same_trajectory():
    xa, fa, wa, seed_a = _run(seed=1234)
    xb, fb, wb, seed_b = _run(seed=1234)
    assert seed_a == seed_b == 1234
    assert len(fa) == len(fb)
    np.testing.assert_array_equal(xa, xb)
    np.testing.assert_array_equal(fa, fb)
    assert list(wa) == list(wb)


def test_different_seed_different_trajectory():
    xa, fa, _, _ = _run(seed=1)
    xb, fb, _, _ = _run(seed=2)
    assert not (len(fa) == len(fb) and np.array_equal(fa, fb))


def test_seed_precedence():
    from panobbgo.strategies import StrategyRoundRobin

    problem = Rosenbrock(dim=2)
    s = StrategyRoundRobin(problem, parse_args=False, seed=99)
    assert s.seed == 99

    # 0 is a seed, not "unset".
    assert StrategyRoundRobin(problem, parse_args=False, seed=0).seed == 0

    # config.seed is the fallback when no kwarg is given.
    s = StrategyRoundRobin(problem, parse_args=False)
    s.config.seed = 0
    assert StrategyRoundRobin(problem, parse_args=False, seed=None).seed != 0  # own config

    # Without an explicit seed, numpy's global state decides — so a
    # preceding np.random.seed still pins the run.
    np.random.seed(5)
    s1 = StrategyRoundRobin(problem, parse_args=False)
    np.random.seed(5)
    s2 = StrategyRoundRobin(problem, parse_args=False)
    assert s1.seed == s2.seed


def test_spawn_rng_streams_are_deterministic_and_distinct():
    from panobbgo.strategies import StrategyRoundRobin

    problem = Rosenbrock(dim=2)
    a = StrategyRoundRobin(problem, parse_args=False, seed=3)
    b = StrategyRoundRobin(problem, parse_args=False, seed=3)
    ra1, ra2 = a.spawn_rng(), a.spawn_rng()
    rb1, rb2 = b.spawn_rng(), b.spawn_rng()
    assert ra1.random() == rb1.random()
    assert ra2.random() == rb2.random()
    assert ra1.random() != ra2.random()


def test_modules_get_seeded_rng():
    from panobbgo.heuristics import Random
    from panobbgo.strategies import StrategyRoundRobin

    problem = Rosenbrock(dim=2)
    s = StrategyRoundRobin(problem, parse_args=False, seed=11)
    h = Random(s)
    assert isinstance(h.rng, np.random.Generator)
    t = StrategyRoundRobin(problem, parse_args=False, seed=11)
    g = Random(t)
    assert h.rng.random() == g.rng.random()


# --------------------------------------------------------------------------
# Keyed RNG streams (2026-09-25): a module's stream depends on the master
# seed and its own name only, not on which other modules exist.
# --------------------------------------------------------------------------


def _strategy(seed=5):
    from panobbgo.strategies import StrategyRoundRobin

    return StrategyRoundRobin(Rosenbrock(dim=2), parse_args=False, testing_mode=True, seed=seed)


def test_unrelated_module_does_not_change_a_module_stream():
    from panobbgo.heuristics import Nearby, Random
    from panobbgo.analyzers import Archive

    alone = _strategy()
    crowded = _strategy()
    Random(crowded)  # built before ...
    Archive(crowded)
    ref = Nearby(alone).rng.random(8)
    got = Nearby(crowded).rng.random(8)
    Random(crowded)  # ... and after: neither shifts Nearby's stream
    np.testing.assert_array_equal(ref, got)


def test_strategy_stream_is_independent_of_the_modules():
    from panobbgo.heuristics import Nearby, Random

    bare, busy = _strategy(), _strategy()
    Random(busy)
    Nearby(busy)
    np.testing.assert_array_equal(bare.rng.random(8), busy.rng.random(8))


def test_two_instances_of_one_class_get_distinct_streams():
    from panobbgo.heuristics import Random

    s = _strategy()
    a, b = Random(s), Random(s)
    ra, rb = a.rng.random(8), b.rng.random(8)
    assert not np.array_equal(ra, rb)
    # ... reproducibly: the n-th instance of a name always gets the same stream.
    t = _strategy()
    np.testing.assert_array_equal(Random(t).rng.random(8), ra)
    np.testing.assert_array_equal(Random(t).rng.random(8), rb)
    # A custom name is its own key.
    u = _strategy()
    assert not np.array_equal(Random(u, name="Other").rng.random(8), ra)


def test_default_analyzers_do_not_shift_module_streams():
    """Whether the on-demand Splitter is built changes no other stream."""
    from panobbgo.heuristics import Random, WeightedAverage

    streams = []
    for extra in (False, True):
        s = _strategy()
        s.config.max_eval = 5
        h = Random(s)
        s.add_heuristic(h)
        if extra:
            s.add_heuristic(WeightedAverage(s))  # needs the Splitter
        try:
            s.initialize()
            best = s.analyzer("Best")
            streams.append((h.rng.random(4), best.rng.random(4), s.rng.random(4)))
        finally:
            s._cleanup()
    for a, b in zip(*streams):
        np.testing.assert_array_equal(a, b)
