# -*- coding: utf8 -*-
"""The regime gate of :class:`~panobbgo.strategies.StrategyBlockBandit`.

``planning/DESIGN_regime_gating_2026-09-11.md`` §2 and §6 step 1: the
table, its first-match order, the constructor contract, the arm mask —
honoured in ``_select`` and in the prologue and nowhere else — and the two
byte-identity claims the design rests on (``regime_gate=None`` is today's
run; a gate that enables every arm is ``None``).
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from panobbgo.core import Heuristic
from panobbgo.lib.classic import Rosenbrock
from panobbgo.strategies import StrategyBlockBandit, StrategyRoundRobin
from panobbgo.strategies.blocks import (
    NOISE_CLASSES,
    REGIME_TABLE_V1,
    lookup_regime_table,
    regime_arms_for_roles,
    regime_predicate_holds,
)

BATCH = 7


class Generational(Heuristic):
    """Population-shaped stub (as in ``test_strategy_blocks.py``)."""

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


class Constrained(Rosenbrock):
    """Rosenbrock with one (always satisfied) constraint, so ``eval_constraints`` returns a vector."""

    def eval_constraints(self, x):
        return np.array([-1.0])


def _stub(name):
    return lambda st: Generational(st, name=name)


def _strategy(cls=StrategyBlockBandit, seed=0, max_eval=200, dim=2, arms=(), problem=None, **kw):
    s = cls(problem or Rosenbrock(dim=dim), parse_args=False, testing_mode=True, seed=seed, **kw)
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


def _assert_same(a, b):
    xa, fa, wa = a
    xb, fb, wb = b
    assert len(fa) == len(fb)
    np.testing.assert_array_equal(xa, xb)
    np.testing.assert_array_equal(fa, fb)
    assert list(wa) == list(wb)


# -- the table ----------------------------------------------------------------


@pytest.mark.parametrize(
    "noise, dim, bpd, constrained, row, roles",
    [
        # every row, hit on the nose (DISCOVERY §42 / §44)
        ("outlier", 2, 500, False, ("outlier", None, None, None), ("CMAES",)),
        ("outlier", 10, 2000, False, ("outlier", None, None, None), ("CMAES",)),
        ("clean", 5, 500, True, (None, None, None, True), ("JSO",)),
        ("bounded", 2, 500, False, ("bounded", "dim <= 5", None, None), ("CMAES", "JSO")),
        ("bounded", 5, 500, False, ("bounded", "dim <= 5", None, None), ("CMAES", "JSO")),
        ("clean", 10, 500, False, (None, "dim >= 10", "bpd <= 500", None), ("CMAES",)),
        ("clean", 20, 100, False, (None, "dim >= 10", "bpd <= 500", None), ("CMAES",)),
        ("clean", 2, 200, False, (None, "dim <= 5", "bpd <= 200", None), ("CMAES", "JSO")),
        ("clean", 5, 100, False, (None, "dim <= 5", "bpd <= 200", None), ("CMAES", "JSO")),
        # the default: noiseless 500·dim and 2000·dim at d <= 5, d = 10 above 500·dim, d = 7
        ("clean", 2, 500, False, (None, None, None, None), ("CMAES",)),
        ("clean", 5, 2000, False, (None, None, None, None), ("CMAES",)),
        ("clean", 10, 2000, False, (None, None, None, None), ("CMAES",)),
        ("clean", 7, 200, False, (None, None, None, None), ("CMAES",)),
        ("bounded", 10, 2000, False, (None, None, None, None), ("CMAES",)),
    ],
)
def test_table_lookup(noise, dim, bpd, constrained, row, roles):
    assert lookup_regime_table(noise, dim, bpd, constrained) == (row, roles)


def test_first_match_order():
    # outlier beats everything, constrained included
    assert lookup_regime_table("outlier", 2, 100, True)[1] == ("CMAES",)
    # constrained beats the "share" rows: bounded noise at d <= 5, and 200·dim
    assert lookup_regime_table("bounded", 5, 500, True)[0] == (None, None, None, True)
    assert lookup_regime_table("clean", 2, 200, True)[0] == (None, None, None, True)
    # ... and, by the same order, the d >= 10 row (unmeasured, documented)
    assert lookup_regime_table("clean", 10, 500, True)[1] == ("JSO",)
    # bounded noise at d = 10 is not the d <= 5 row: it falls to d >= 10 / default
    assert lookup_regime_table("bounded", 10, 500, False)[0] == (None, "dim >= 10", "bpd <= 500", None)
    # the table ends in a catch-all, and it is the last row
    assert list(REGIME_TABLE_V1)[-1] == (None, None, None, None)
    for key in REGIME_TABLE_V1:
        assert len(key) == 4


def test_table_is_exhaustive_over_the_grid():
    """No (class, dim, bpd, constrained) combination falls through."""
    for noise, dim, bpd, con in itertools.product(NOISE_CLASSES, (1, 2, 5, 7, 10, 20), (50, 200, 500, 2000), (0, 1)):
        _, roles = lookup_regime_table(noise, dim, bpd, bool(con))
        assert roles and set(roles) <= {"CMAES", "JSO"}


def test_predicates():
    assert regime_predicate_holds(None, "dim", 3)
    assert regime_predicate_holds("dim <= 5", "dim", 5)
    assert not regime_predicate_holds("dim < 5", "dim", 5)
    assert regime_predicate_holds("bpd >= 500", "bpd", 500.0)
    assert regime_predicate_holds("bpd == 200", "bpd", 200)
    with pytest.raises(ValueError, match="malformed"):
        regime_predicate_holds("dim ~ 5", "dim", 5)
    with pytest.raises(ValueError, match="slot"):
        regime_predicate_holds("dim <= 5", "bpd", 5)
    with pytest.raises(ValueError, match="noise class"):
        lookup_regime_table("gauss", 2, 500, False)
    with pytest.raises(LookupError):
        lookup_regime_table("clean", 2, 500, False, table={("outlier", None, None, None): ("CMAES",)})


def test_roles_map_to_arms_by_exact_class_or_name():
    from panobbgo.heuristics import CMAES, JSO, NLSHADE_LBC, Random

    s = _strategy(max_eval=100)
    s.add_heuristic(CMAES(s))
    s.add_heuristic(JSO(s, NP_init=8))
    s.add_heuristic(NLSHADE_LBC(s, NP_init=8))  # a JSO *subclass*: not the arm §42 measured
    s.add_heuristic(Random(s, name="CMAES2"))
    s.add_heuristic(Generational(s, name="JSO_stub"))
    hs = list(s._heuristics.values())
    assert regime_arms_for_roles(hs, ("CMAES",)) == ["CMAES"]
    assert regime_arms_for_roles(hs, ("JSO",)) == ["JSO"]
    assert regime_arms_for_roles(hs, ("CMAES", "JSO")) == ["CMAES", "JSO"]
    assert regime_arms_for_roles(hs, ("CMAES2",)) == ["CMAES2"]  # by name
    assert regime_arms_for_roles(hs, ("PSO",)) == []


# -- the constructor ----------------------------------------------------------


def test_constructor_validation():
    assert _strategy(regime_gate=None).regime_gate is None
    for cls in NOISE_CLASSES:
        assert _strategy(regime_gate="oracle:%s" % cls).regime_gate == "oracle:%s" % cls
    with pytest.raises(NotImplementedError, match="table-v1"):
        _strategy(regime_gate="table-v1")
    with pytest.raises(ValueError, match="names no class"):
        _strategy(regime_gate="oracle")
    with pytest.raises(ValueError, match="regime_gate"):
        _strategy(regime_gate="oracle:gauss")
    with pytest.raises(ValueError, match="regime_gate"):
        _strategy(regime_gate="bogus")
    with pytest.raises(ValueError, match="regime_gate"):
        _strategy(regime_gate=3)


# -- the mask -----------------------------------------------------------------


def test_mask_honoured_in_select_and_prologue():
    s = _run(regime_gate="oracle:outlier", n_blocks=10, arms=(_stub("CMAES"), _stub("JSO")))
    assert s.regime_decision is not None
    assert s.regime_decision["row"] == ("outlier", None, None, None)
    assert s.regime_decision["enabled"] == ["CMAES"]
    assert s.regime_decision["disabled"] == ["JSO"]
    assert s._enabled == {"CMAES": True, "JSO": False}
    owners = [b["owner"] for b in s._blocks]
    assert owners and set(owners) == {"CMAES"}
    assert [b["owner"] for b in s._blocks if b["prologue"]] == ["CMAES"]  # no prologue block for JSO
    _, _, who = _trajectory(s)
    assert set(who) == {"CMAES"}
    # the disabled arm was constructed and registered all the same
    assert "JSO" in s._heuristics and "JSO" in s._n


def test_select_reads_the_mask():
    """``_select`` is the one place the mask is read: flip bits, watch it follow."""
    s = _strategy(max_eval=100, arms=())
    a, b = Generational(s, name="A"), Generational(s, name="B")
    s.add_heuristic(a)
    s.add_heuristic(b)
    a.emit([np.zeros(2)])
    b.emit([np.zeros(2)])
    s._prologue = []
    assert s._enabled == {} and s._select().name in ("A", "B")
    s._enabled = {"A": True, "B": False}
    assert s._select().name == "A"
    s._enabled = {"A": False, "B": True}
    assert s._select().name == "B"
    s._enabled = {"A": False, "B": False}
    assert s._select() is None
    # the prologue honours it too: a disabled arm listed there is skipped
    s._enabled = {"A": False, "B": True}
    s._prologue = ["A", "B"]
    assert s._select().name == "B" and s._prologue_pick


def test_gate_reads_dim_budget_and_constraints_once():
    # bpd = 200/2 = 100 -> the 200·dim row: both arms
    s = _run(regime_gate="oracle:clean", max_eval=200, arms=(_stub("CMAES"), _stub("JSO")))
    d = s.regime_decision
    assert (d["dim"], d["max_eval"], d["bpd"], d["constrained"]) == (2, 200, 100.0, False)
    assert d["row"] == (None, "dim <= 5", "bpd <= 200", None)
    assert d["enabled"] == ["CMAES", "JSO"]
    assert {b["owner"] for b in s._blocks} == {"CMAES", "JSO"}

    # bpd = 1200/2 = 600 -> default row: CMA-ES alone
    s = _run(regime_gate="oracle:clean", max_eval=1200, arms=(_stub("CMAES"), _stub("JSO")))
    assert s.regime_decision["row"] == (None, None, None, None)
    assert {b["owner"] for b in s._blocks} == {"CMAES"}

    # constrained -> jSO alone, even at 200·dim
    s = _run(regime_gate="oracle:clean", max_eval=200, problem=Constrained(dim=2), arms=(_stub("CMAES"), _stub("JSO")))
    assert s.regime_decision["constrained"] is True
    assert s.regime_decision["row"] == (None, None, None, True)
    assert {b["owner"] for b in s._blocks} == {"JSO"}

    # no gate: nothing is decided, nothing is masked
    s = _run(max_eval=200, arms=(_stub("CMAES"), _stub("JSO")))
    assert s.regime_decision is None and s._enabled == {}


def test_unknown_roles(caplog):
    # partial match: the row names CMAES + JSO, the strategy has CMAES + X -> X is disabled
    s = _run(regime_gate="oracle:bounded", max_eval=200, arms=(_stub("CMAES"), _stub("X")))
    assert s.regime_decision["enabled"] == ["CMAES"]
    assert {b["owner"] for b in s._blocks} == {"CMAES"}

    # nothing matches: every arm stays enabled and a warning is logged
    with caplog.at_level("WARNING"):
        s = _run(regime_gate="oracle:outlier", max_eval=200, arms=(_stub("A"), _stub("B")))
    assert s.regime_decision["enabled"] == ["A", "B"]
    assert s.regime_decision["disabled"] == []
    assert {b["owner"] for b in s._blocks} == {"A", "B"}
    assert any("leaving every arm enabled" in r.getMessage() for r in caplog.records)


def test_one_arm_mask_matches_round_robin():
    """A one-arm mask adds nothing (test #8 of ``test_strategy_blocks.py``, behind the gate).

    The masked arm is registered *after* the enabled one so the enabled
    arm's RNG stream is the first spawn in both runs.
    """
    gated = _run(regime_gate="oracle:outlier", seed=7, max_eval=80, arms=(_stub("CMAES"), _stub("JSO")))
    alone = _run(cls=StrategyRoundRobin, seed=7, max_eval=80, arms=(_stub("CMAES"),))
    _assert_same(_trajectory(gated), _trajectory(alone))


# -- byte identity on the real arms (design §2.2) -----------------------------


def _real(gate, seed=11, max_eval=160, dim=2):
    from panobbgo.analyzers import Archive
    from panobbgo.heuristics import CMAES, JSO

    s = _strategy(
        seed=seed,
        max_eval=max_eval,
        dim=dim,
        policy="uniform",
        warm_start_on_resume=True,
        warm_start_only_if_foreign=False,
        regime_gate=gate,
    )
    s.add(CMAES, warm_start="archive")
    s.add(JSO, NP_init="auto", warm_start="archive")
    s.add_analyzer(Archive(s))
    s.start()
    return s


@pytest.fixture(scope="module")
def ungated():
    s = _real(None)
    return _trajectory(s)


def test_none_is_reproducible(ungated):
    """``regime_gate=None`` is a pure function of the seed (the pre-gate contract)."""
    _assert_same(_trajectory(_real(None)), ungated)


def test_gate_that_enables_every_arm_is_byte_identical_to_none(ungated):
    """The arms are always all constructed, so the mask is the *only* difference."""
    s = _real("oracle:bounded")  # d = 2 -> ("CMAES", "JSO")
    assert s.regime_decision["enabled"] == ["CMAES", "JSO"]
    _assert_same(_trajectory(s), ungated)
    s = _real("oracle:clean")  # 160 / 2 = 80·dim -> ("CMAES", "JSO")
    assert s.regime_decision["row"] == (None, "dim <= 5", "bpd <= 200", None)
    _assert_same(_trajectory(s), ungated)


def test_gate_that_disables_an_arm_changes_the_run(ungated):
    s = _real("oracle:outlier")
    assert s.regime_decision["enabled"] == ["CMAES"]
    _, fx, who = _trajectory(s)
    assert {w.split(":")[0] for w in who} == {"CMAES"}
    assert "JSO" in {w.split(":")[0] for w in ungated[2]}
    assert len(fx) >= 160
    # deterministic in its own right
    _assert_same(_trajectory(_real("oracle:outlier")), (_, fx, who))


# -- the harness hook (design §4: the oracle is the harness handing the class in) --


def test_problem_is_constrained():
    assert Rosenbrock(dim=2).is_constrained() is False
    assert Constrained(dim=2).is_constrained() is True
    from panobbgo.lib.families import Family

    assert Family("sphere", dim=2, seed=0).is_constrained() is False
    assert Family("sphere", dim=2, seed=0, n_constraints=1).is_constrained() is True


def test_noise_class_of():
    from panobbgo.harness_ioh import noise_class_of

    assert noise_class_of("MA-BBOB") == "clean"
    assert noise_class_of("BBOB") == "clean"
    assert noise_class_of("MA-BBOB-noisy-gauss") == "bounded"
    assert noise_class_of("MA-BBOB-noisy-unif") == "bounded"
    assert noise_class_of("MA-BBOB-noisy-cauchy") == "outlier"


def test_spec_with_regime_class():
    from panobbgo.benchmark import StrategySpec
    from panobbgo.heuristics import Random

    plain = StrategySpec(name="p", strategy_class=StrategyRoundRobin, heuristics=[(Random, {})])
    assert plain.with_regime_class("outlier") is plain
    pinned = StrategySpec(
        name="q", strategy_class=StrategyBlockBandit, heuristics=[], config_overrides={"regime_gate": "oracle:clean"}
    )
    assert pinned.with_regime_class("outlier") is pinned  # an explicit class is not overridden
    bare = StrategySpec(
        name="r",
        strategy_class=StrategyBlockBandit,
        heuristics=[],
        config_overrides={"regime_gate": "oracle", "policy": "uniform"},
        seed_name="shared",
    )
    resolved = bare.with_regime_class("bounded")
    assert resolved.config_overrides == {"regime_gate": "oracle:bounded", "policy": "uniform"}
    assert resolved.seed_name == "shared" and resolved.name == "r"
    assert bare.config_overrides["regime_gate"] == "oracle"  # the original is untouched


def test_harness_spec_is_registered_and_buildable():
    from panobbgo.harness_ioh import make_ioh_strategies

    spec = {s.name: s for s in make_ioh_strategies()}["RegimeGate_oracle"]
    portfolio = {s.name: s for s in make_ioh_strategies()}["Blocks_warm_CMAES_JSO"]
    assert spec.config_overrides["regime_gate"] == "oracle"
    assert spec.rng_identity == portfolio.rng_identity  # one RNG stream: the delta is the gate alone
    assert spec.heuristics == portfolio.heuristics
    assert {k: v for k, v in spec.config_overrides.items() if k != "regime_gate"} == portfolio.config_overrides
    # unresolved, the strategy refuses to guess the class
    with pytest.raises(ValueError, match="names no class"):
        spec.create_strategy(Rosenbrock(dim=2), seed=1, max_eval=100)
    s = spec.with_regime_class("outlier").create_strategy(Rosenbrock(dim=2), seed=1, max_eval=100)
    assert isinstance(s, StrategyBlockBandit)
    assert s.regime_gate == "oracle:outlier" and s.config.max_eval == 100
    s._cleanup()
