# -*- coding: utf8 -*-
"""The block scheduler of :class:`~panobbgo.strategies.StrategyBlockBandit`.

The eight properties the design (``planning/DESIGN_block_bandit_2026-09-10.md``
§5) asks to be pinned: contiguous blocks, generations that are never cut,
a reward that is scale-free and anytime, a prologue that covers every arm,
an exploit-only tail, reproducibility, the ``warm_start`` contract, and —
the one that protects ``RoundRobin_CMAES`` — a single-arm run that is
indistinguishable from :class:`~panobbgo.strategies.StrategyRoundRobin`.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from panobbgo.core import Heuristic
from panobbgo.lib import Point, Result
from panobbgo.lib.classic import Rosenbrock
from panobbgo.strategies import StrategyBlockBandit, StrategyRoundRobin

BATCH = 7


class Generational(Heuristic):
    """Population-shaped stub: emits a generation of ``batch`` points and
    only produces the next one once the previous has been fully drawn."""

    def __init__(self, strategy, name="Gen", batch=BATCH):
        self.batch = int(batch)
        Heuristic.__init__(self, strategy, name=name, cap=batch)

    def _generation(self):
        self.emit([self.problem.random_point(rng=self.rng) for _ in range(self.batch)])

    def on_start(self):
        self._generation()

    def on_new_results(self, results):
        if not self.has_points:
            self._generation()


class Restarting(Heuristic):
    """Emits one generation on ``start``; afterwards only a *warm start*
    refills it.  Records every ``warm_start`` call it receives."""

    def __init__(self, strategy, name="Restart", batch=BATCH):
        self.batch = int(batch)
        self.warm_calls: list[tuple[int, int]] = []
        Heuristic.__init__(self, strategy, name=name, cap=batch)

    def _generation(self):
        self.emit([self.problem.random_point(rng=self.rng) for _ in range(self.batch)])

    def on_start(self):
        self._generation()

    def on_new_results(self, results):
        """Never refills — but keeps the heuristic subscribed, hence ``active``."""

    def warm_start(self, results):
        self.warm_calls.append((len(results), self._output.qsize()))
        self._generation()


def _strategy(cls=StrategyBlockBandit, seed=0, max_eval=420, arms=(), **kw):
    s = cls(Rosenbrock(dim=2), parse_args=False, testing_mode=True, seed=seed, **kw)
    s.config.max_eval = max_eval
    s.config.sync_evaluation = True
    s.config.stop_on_convergence = False
    s.config.evaluation_method = "threaded"
    for factory in arms:
        s.add_heuristic(factory(s))
    return s


def _run(**kw):
    s = _strategy(**kw)
    s.start()
    return s


def _trajectory(s):
    df = s.results.results
    assert df is not None
    x = df["x"].to_numpy(dtype=float)
    fx = df[("fx", 0)].to_numpy(dtype=float).ravel()
    who = df[("who", 0)].to_numpy().ravel().astype(str)
    return x, fx, who


def _runs(who):
    """Maximal runs of a constant ``who`` as ``(name, length)`` pairs."""
    return [(name, len(list(grp))) for name, grp in itertools.groupby(who)]


def _result(fx, who="A"):
    return Result(Point(np.zeros(2), who), fx, cv_vec=np.zeros(2))


def _score_block(s, values, anchor=None, name="A"):
    """Open a block, feed it ``values``, close it, return its reward."""
    if anchor is not None:
        s._anchor = anchor
    s._open_block(s._heuristics[name])
    s.on_new_results([_result(v, name) for v in values])
    s._close_block()
    return s._blocks[-1]["reward"]


def _reward_probe(**kw):
    """A strategy with one registered arm, driven block by block by hand."""
    return _strategy(arms=(lambda st: Generational(st, name="A"),), **kw)


# 1 --------------------------------------------------------------------


def test_blocks_are_contiguous():
    s = _run(
        max_eval=420,
        n_blocks=10,
        arms=(
            lambda st: Generational(st, name="A"),
            lambda st: Generational(st, name="B"),
            lambda st: Generational(st, name="C"),
        ),
    )
    assert s.block_evals == 42
    _, _, who = _trajectory(s)
    runs = _runs(who)
    assert len(runs) > 1, "more than one arm must have owned a block"
    # every run but the last (which the budget truncates) is at least one
    # prologue block long -- ownership does not alternate point by point
    for _, length in runs[:-1]:
        assert length >= s.block_evals // 2
    # and every closed block spent at least its own length
    for b in s._blocks[:-1]:
        assert b["evals"] >= b["size"]


# 2 --------------------------------------------------------------------


def test_a_generation_is_never_cut():
    s = _run(
        max_eval=420,
        n_blocks=10,
        arms=(
            lambda st: Generational(st, name="A"),
            lambda st: Generational(st, name="B"),
        ),
    )
    assert s._blocks, "the run must have closed at least one block"
    for b in s._blocks:
        # the owner's queue had run empty on the last draw of the block --
        # this is the ``has_points is False`` gate at the ownership change
        assert b["drained"] is True, b
        # ... which for a generational arm means whole generations only
        assert b["evals"] % BATCH == 0, b
    _, _, who = _trajectory(s)
    for _, length in _runs(who)[:-1]:
        assert length % BATCH == 0


# 3 --------------------------------------------------------------------


def test_reward_is_scale_free():
    """``f`` and ``1000*f + 5`` have to earn the identical reward."""
    raw = [[100.0, 60.0, 55.0, 20.0], [18.0, 17.0, 4.0, 3.5]]

    plain = _reward_probe()
    shifted = _reward_probe()
    got_plain = [_score_block(plain, block) for block in raw]
    got_shifted = [_score_block(shifted, [1000.0 * v + 5.0 for v in block]) for block in raw]

    assert got_plain[0] > 0.0, "the reward must not be identically zero"
    for a, b in zip(got_plain, got_shifted):
        assert a == pytest.approx(b, abs=1e-12)
    # the anchor itself moves affinely with the objective
    assert shifted._anchor == pytest.approx(1000.0 * plain._anchor + 5.0)


# 4 --------------------------------------------------------------------


def test_reward_is_anytime():
    """Same total drop: dropping early beats dropping at the last evaluation."""
    early = [100.0, 10.0, 10.0, 10.0]
    late = [100.0, 100.0, 100.0, 10.0]

    r_early = _score_block(_reward_probe(), early, anchor=0.0)
    r_late = _score_block(_reward_probe(), late, anchor=0.0)
    assert r_early == pytest.approx(0.375)
    assert r_late == pytest.approx(0.125)
    assert r_early > r_late

    # the endpoint ablation is blind to *when* the drop happened
    e_early = _score_block(_reward_probe(reward="endpoint"), early, anchor=0.0)
    e_late = _score_block(_reward_probe(reward="endpoint"), late, anchor=0.0)
    assert e_early == pytest.approx(0.5)
    assert e_late == pytest.approx(0.5)


# 5 --------------------------------------------------------------------


def test_prologue_covers_every_arm():
    names = ["A", "B", "C"]
    s = _run(
        max_eval=420,
        n_blocks=10,
        arms=tuple((lambda n: lambda st: Generational(st, name=n))(n) for n in names),
    )
    prologue = [b for b in s._blocks if b["prologue"]]
    assert len(prologue) == len(names)
    assert sorted(b["owner"] for b in prologue) == names
    assert [b["owner"] for b in s._blocks[: len(names)]] == names  # registration order
    for b in prologue:
        assert b["size"] == s.block_evals // 2  # half length
    assert not any(b["prologue"] for b in s._blocks[len(names) :])


def test_tail_is_exploit_only():
    s = _strategy(max_eval=100, tail_frac=0.25, ucb_c=0.5, arms=())
    assert s._exploration_c() == pytest.approx(0.5)
    s.results.add_results([_result(1.0) for _ in range(80)])
    assert s._exploration_c() == 0.0

    # with c = 0 the barely-tried arm no longer outranks the better one
    a, b = Generational(s, name="A"), Generational(s, name="B")
    s.add_heuristic(a)
    s.add_heuristic(b)
    a.emit([np.zeros(2)])
    b.emit([np.zeros(2)])
    s._prologue = []
    s._N, s._n["A"], s._S["A"], s._n["B"], s._S["B"] = 11.0, 10.0, 5.0, 1.0, 0.1
    assert s._score("B", 0.5) > s._score("A", 0.5)  # exploration would pick B
    assert s._select().name == "A"  # the tail does not


# 6 --------------------------------------------------------------------


def test_same_seed_same_trajectory():
    arms = (
        lambda st: Generational(st, name="A"),
        lambda st: Generational(st, name="B"),
        lambda st: Generational(st, name="C"),
    )
    xa, fa, wa = _trajectory(_run(seed=1234, max_eval=210, n_blocks=10, arms=arms))
    xb, fb, wb = _trajectory(_run(seed=1234, max_eval=210, n_blocks=10, arms=arms))
    assert len(fa) == len(fb) >= 210
    np.testing.assert_array_equal(xa, xb)
    np.testing.assert_array_equal(fa, fb)
    assert list(wa) == list(wb)


# 7 --------------------------------------------------------------------


def _warm_start_run(flag):
    s = _run(
        max_eval=210,
        n_blocks=10,
        # uniform: the arms alternate by construction, so the test pins the
        # warm-start contract and not the bandit's choices
        policy="uniform",
        warm_start_on_resume=flag,
        arms=(
            lambda st: Generational(st, name="G"),
            lambda st: Restarting(st, name="R"),
        ),
    )
    return s, s._heuristics["R"]


def test_warm_start_is_off_by_default():
    s = _strategy(arms=())
    assert s.warm_start_on_resume is False
    s, r = _warm_start_run(False)
    assert r.warm_calls == []
    # without a warm start the emptied arm can never be re-acquired
    assert sum(1 for b in s._blocks if b["owner"] == "R") == 1


def test_warm_start_once_per_reacquisition_and_never_on_a_full_queue():
    s, r = _warm_start_run(True)
    owned = sum(1 for b in s._blocks if b["owner"] == "R")
    assert owned >= 2, "the arm must have been re-acquired at least once"
    # the first acquisition still holds the generation from ``on_start``;
    # every later one arrives empty -> exactly one call per re-acquisition
    assert len(r.warm_calls) == owned - 1
    for n_results, qsize in r.warm_calls:
        assert qsize == 0  # never with a non-empty queue
        assert 0 < n_results <= s.warm_start_k


# 8 --------------------------------------------------------------------


def test_single_arm_matches_round_robin():
    """One arm: the scheduler must add nothing at all (protects RoundRobin_CMAES)."""
    from panobbgo.heuristics import Random

    def run(cls):
        return _trajectory(_run(cls=cls, seed=7, max_eval=80, arms=(lambda st: Random(st),)))

    xa, fa, wa = run(StrategyBlockBandit)
    xb, fb, wb = run(StrategyRoundRobin)
    assert len(fa) == len(fb)
    np.testing.assert_array_equal(xa, xb)
    np.testing.assert_array_equal(fa, fb)
    assert list(wa) == list(wb)


# extras ---------------------------------------------------------------


def test_uniform_policy_is_round_robin_over_blocks():
    names = ["A", "B", "C"]
    s = _run(
        max_eval=420,
        n_blocks=10,
        policy="uniform",
        arms=tuple((lambda n: lambda st: Generational(st, name=n))(n) for n in names),
    )
    owners = [b["owner"] for b in s._blocks]
    assert owners[:3] == names  # prologue
    for prev, nxt in zip(owners[3:], owners[4:]):
        assert nxt == names[(names.index(prev) + 1) % len(names)]


def test_argument_validation():
    with pytest.raises(ValueError, match="policy"):
        _strategy(policy="bogus")
    with pytest.raises(ValueError, match="reward"):
        _strategy(reward="bogus")
    with pytest.raises(NotImplementedError):
        _strategy(prior="dim")
    with pytest.raises(ValueError, match="prior"):
        _strategy(prior="bogus")
