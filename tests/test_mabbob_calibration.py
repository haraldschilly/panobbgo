"""The calibration: a published MA-BBOB entry's code under the harness.

The tests named ``test_entry_runs_*`` execute the published code; the others do not.
"""

import hashlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest

import scripts.rebaseline as rb
from panobbgo.harness_baselines import NeighborhoodAdaptiveDEStrategy, make_baseline_strategies
from panobbgo.lib.classic import Rosenbrock

VENDORED = Path(__file__).parents[1] / "panobbgo" / "third_party" / "neighborhood_adaptive_de.py"
ZENODO_MD5 = "8802791b458e1f6931bf87b70bd580b1"
MODULE = "panobbgo.third_party.neighborhood_adaptive_de"


def test_vendored_code_differs_from_the_published_file_only_by_np_inf():
    header, body = VENDORED.read_bytes().split(b"# --- end of header ---\n", 1)
    assert ZENODO_MD5.encode() in header and b"CC BY 4.0" in header
    assert body.count(b"np.inf") == 1
    assert hashlib.md5(body.replace(b"np.inf", b"np.Inf")).hexdigest() == ZENODO_MD5


def test_calibration_suite_is_opt_in_and_runs_only_the_entry_on_tune_rows():
    suite = rb.SUITES["ioh-mabbob-official-calibrate"]
    assert suite.opt_in and suite.battery == "mabbob-official" and suite.ref_key == "mabbob-official_calibrate"
    assert suite.name not in {s.name for s in rb.resolve_suites("all")}
    assert suite.name not in rb.HELDOUT_SUITES
    assert rb.STRATEGY_SETS["calibrate"](suite) == ["Baseline_NeighborhoodAdaptiveDE"]
    with pytest.raises(ValueError, match="release=none"):
        rb.check_dispatch([suite], "auto")


def test_entry_is_a_selectable_baseline_only_by_name():
    pick = make_baseline_strategies(["Baseline_NeighborhoodAdaptiveDE"])
    assert [s.strategy_class for s in pick][3:] == [NeighborhoodAdaptiveDEStrategy]
    assert not any(s.strategy_class is NeighborhoodAdaptiveDEStrategy for s in make_baseline_strategies())


def test_adapter_hands_the_entry_the_problem_seed_and_budget(monkeypatch):
    """A stand-in for the entry, so this test runs none of the published code."""
    seen = {}

    class Entry:
        def __init__(self, budget_factor):
            seen["factor"] = budget_factor

        def __call__(self, func, seed):
            seen.update(seed=seed, dim=func.meta_data.n_variables, lb=func.bounds.lb, ub=func.bounds.ub)
            seen["f"] = func(np.array([1.0, 1.0]))

    monkeypatch.setitem(sys.modules, MODULE, types.SimpleNamespace(NeighborhoodAdaptiveDE=Entry))
    problem = Rosenbrock(dims=2)
    strategy = NeighborhoodAdaptiveDEStrategy(problem, seed=2**32 + 7)
    strategy.config.max_eval = 4000
    strategy.start()
    assert seen["factor"] == 2000 and seen["seed"] == 7 and seen["dim"] == 2
    assert np.array_equal(seen["lb"], problem.box[:, 0]) and np.array_equal(seen["ub"], problem.box[:, 1])
    assert seen["f"] == pytest.approx(0.0)
    assert strategy.best.fx == pytest.approx(0.0)


def _run_entry(seed, budget=300):
    strategy = NeighborhoodAdaptiveDEStrategy(Rosenbrock(dims=2), seed=seed)
    strategy.config.max_eval = budget
    strategy.start()
    return strategy


def test_entry_runs_to_exactly_the_budget_inside_the_box():
    strategy = _run_entry(1, budget=317)  # 317: past the initial population, mid-generation
    frame = strategy.results.results
    assert len(frame) == 317
    box = strategy.problem.box
    xs = frame["x"].to_numpy()
    assert (xs >= box[:, 0]).all() and (xs <= box[:, 1]).all()


def test_entry_runs_seeded_are_reproducible_and_seeds_differ():
    first, again, other = _run_entry(1), _run_entry(1), _run_entry(2)
    assert np.array_equal(first.results.results["fx"], again.results.results["fx"])
    assert not np.array_equal(first.results.results["fx"], other.results.results["fx"])
