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

"""``evaluation_method="processes"``: a spawn-context process pool.

The old implementation launched ``python3 -c`` per point with ``panobbgo/``
itself first on ``sys.path`` (so ``panobbgo/logging`` shadowed the stdlib and
every evaluation failed), never removed failed tasks from ``pending`` and
killed every evaluation after a hard-coded 30 s.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from panobbgo.lib import Problem
from panobbgo.lib.classic import Rosenbrock


class _Failing(Problem):
    """Every evaluation raises (module level, so it pickles for spawn)."""

    def __init__(self):
        super().__init__([(-1, 1), (-1, 1)])

    def eval(self, x):
        raise RuntimeError("boom")


class _Slow(Problem):
    def __init__(self, delay):
        self.delay = delay
        super().__init__([(-1, 1), (-1, 1)])

    def eval(self, x):
        time.sleep(self.delay)
        return float(np.sum(x**2))


def _fx(s):
    return s.results.results["fx"].to_numpy(dtype=float).ravel()


def _run(problem, max_eval, sync, timeout=None, method="processes"):
    from panobbgo.heuristics import Random
    from panobbgo.strategies import StrategyRoundRobin

    s = StrategyRoundRobin(problem, parse_args=False, testing_mode=True, seed=3)
    s.config.evaluation_method = method
    s.config.dask_n_workers = 2
    s.config.max_eval = max_eval
    s.config.sync_evaluation = sync
    s.config.evaluation_timeout = timeout
    s.config.stop_on_convergence = False
    s.add(Random)
    s.start()
    return s


@pytest.mark.parametrize("sync", [True, False], ids=["sync", "async"])
def test_processes_mode_evaluates_the_budget(sync):
    s = _run(Rosenbrock(dim=2), 12, sync)
    assert len(s.results) == 12
    assert np.all(np.isfinite(_fx(s)))


def test_processes_mode_matches_threaded_under_sync():
    a = _run(Rosenbrock(dim=2), 10, True)
    b = _run(Rosenbrock(dim=2), 10, True, method="threaded")
    assert np.array_equal(_fx(a), _fx(b))


def test_failed_evaluations_leave_pending():
    s = _run(_Failing(), 6, False)
    assert len(s.results) == 0
    assert not s.pending
    assert s._dispatched == 6


def test_unpicklable_problem_is_a_clear_error():
    p = Rosenbrock(dim=2)
    p.hook = lambda: None  # type: ignore[attr-defined]
    with pytest.raises(TypeError, match="picklable"):
        _run(p, 4, True)


@pytest.mark.parametrize("sync", [True, False], ids=["sync", "async"])
def test_timeout_is_configurable_and_abandons_the_evaluation(sync):
    t0 = time.time()
    s = _run(_Slow(1.5), 2, sync, timeout=0.3)
    assert len(s.results) == 0
    assert not s.pending
    assert time.time() - t0 < 15
