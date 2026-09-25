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

"""Shared test helpers (moved out of the library's ``panobbgo.utils``)."""

import functools
import unittest
from unittest import mock

import numpy as np


def rng_spawner(seed=0):
    """A ``spawn_rng`` callable: independent generators derived from ``seed``.

    The *k*-th call always returns the same stream, like
    :meth:`panobbgo.core.StrategyBase.spawn_rng`, so modules built against a
    test double are reproducible.
    """
    seq = np.random.SeedSequence(seed)
    return lambda: np.random.default_rng(seq.spawn(1)[0])


def attach_spawn_rng(strategy, seed=0):
    """Give a strategy stand-in (``SimpleNamespace``, ``Mock``, ...) a ``spawn_rng``; returns it."""
    strategy.seed = seed
    strategy.spawn_rng = rng_spawner(seed)
    return strategy


class StrategyDouble:
    """Base class for hand-rolled strategy stand-ins.

    :class:`~panobbgo.core.Module` draws its generator from
    ``strategy.spawn_rng()``; subclasses get a seeded one (class attribute
    :attr:`seed`, or set ``self.seed`` before the first module is built).
    """

    seed = 0

    def spawn_rng(self):
        spawn = self.__dict__.get("_spawn_rng")
        if spawn is None:
            spawn = self.__dict__["_spawn_rng"] = rng_spawner(self.seed)
        return spawn()


def expected_failure(exptn, msg=None):
    """
    Wrapper for a test function, which expects a certain Exception.

    Example::

        @expected_failure(ValueError, "point must be an instance of lib.Point")
        def test_result_error(self):
            Result([1., 1.], 1.1)

    @param Exception exptn: exception class
    @param str msg: expected message
    """

    def wrapper(testfn):
        @functools.wraps(testfn)
        def inner(*args, **kwargs):
            try:
                testfn(*args, **kwargs)
            except exptn as ex:
                if msg is not None:
                    assert str(ex) == msg, "message: '%s'" % str(ex)
            else:
                raise AssertionError("No Exception '%s' raised in '%s'" % (exptn.__name__, testfn.__name__))

        return inner

    return wrapper


class PanobbgoTestCase(unittest.TestCase):
    def __init__(self, name):
        unittest.TestCase.__init__(self, name)
        from panobbgo.config import Config

        self.config = Config(parse_args=False, testing_mode=True)

    def setUp(self):
        from panobbgo.lib.classic import Rosenbrock

        self.problem = Rosenbrock(2)
        self.strategy = self.init_strategy()

    def random_results(self, dim, N, pcv=0.0, rng=None):
        """Build ``N`` synthetic :class:`~panobbgo.lib.Result` objects (test helper).

        :param rng: optional :class:`numpy.random.Generator`; ``None`` falls back
            to numpy's global state so ``np.random.seed`` still pins the data.
        """
        from panobbgo.lib import Result, Point

        r_ = rng if rng is not None else np.random
        results = []
        for _ in range(N):
            p = Point(r_.random(dim), "test")
            cv_vec = np.zeros(dim)
            if pcv > 0.0:
                for cvidx in range(dim):
                    if r_.random() < pcv:
                        cv_vec[cvidx] = r_.standard_normal()
            r = Result(p, r_.random(), cv_vec=cv_vec)
            results.append(r)
        return results

    @mock.patch("panobbgo.core.StrategyBase")
    def init_strategy(self, StrategyBaseMock):
        strategy = StrategyBaseMock()
        strategy.problem = self.problem
        strategy.config = self.config
        # Modules derive their RNG from the strategy, so the stand-in has to
        # honour that contract — with a fixed seed, so module tests are
        # reproducible, and with *independent* streams per module, as
        # StrategyBase.spawn_rng hands out (two modules built here used to
        # draw identical numbers).
        strategy.seed = 0
        strategy.rng = np.random.default_rng(0)
        strategy.spawn_rng.side_effect = rng_spawner(0)
        return strategy
