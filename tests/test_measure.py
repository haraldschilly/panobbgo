# Copyright 2012-2026 Harald Schilly <harald.schilly@gmail.com>
"""``scripts/measure.py``: units, coverage, the shard plan, a tiny run and the aggregation."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import pytest

from panobbgo.harness_baselines import BO_BASELINE_NAMES, EXTERNAL_BASELINE_NAMES
from panobbgo.harness_ioh import IOHHarnessResult, IOHRunRecord, make_ioh_strategies

_PATH = Path(__file__).resolve().parent.parent / "scripts" / "measure.py"
_spec = importlib.util.spec_from_file_location("measure_cli", _PATH)
assert _spec is not None and _spec.loader is not None
ms = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = ms  # dataclasses look their module up there
_spec.loader.exec_module(ms)


def test_group_names_are_registered_baselines():
    assert set(ms.CHEAP_EXTERNALS) <= set(EXTERNAL_BASELINE_NAMES)
    assert ms.LOCAL_REFERENCE in BO_BASELINE_NAMES
    for g in ms.BO_GROUPS:
        assert set(ms.GROUPS[g]) <= set(BO_BASELINE_NAMES)
    # Only external baselines are "Baseline_*"; the panobbgo specs are not.
    assert not any(ms.is_external(s.name) for s in make_ioh_strategies())
    assert all(ms.is_external(n) for g in ms.GROUPS.values() for n in g)
    assert ms.HEADLINE_SPEC in {s.name for s in make_ioh_strategies()}
    assert ms.group_of("Baseline_SMAC_BB") == "SMAC" and ms.group_of(ms.HEADLINE_SPEC) == "core"


def test_unit_id_round_trip():
    u = ms.Unit("core", "free", 20, 4, 10, 42)
    assert u.id == "core.free.b20.q4.d10.s42"
    assert ms.Unit.parse(u.id) == u
    assert u.n_runs == 15
    assert ms.Unit("SMAC", "failure", 100, 1, 5, 7).n_runs == 12
    one = ms.Unit.parse("qLogEI.free.b100.q64.d5.s42.i0")
    assert one.inst == 0 and one.n_runs == 5 and one.id == "qLogEI.free.b100.q64.d5.s42.i0"
    assert [x.inst for x in u.split()] == [0, 1, 2] and one.split() == [one]
    # Family splits: all instances of the k-th family, or one run with both.
    fam = ms.Unit.parse("SMAC.free.b20.q1.d10.s42.f4")
    assert fam.fam == 4 and fam.inst == -1 and fam.n_runs == 3 and fam.id == "SMAC.free.b20.q1.d10.s42.f4"
    both = ms.Unit.parse("SMAC.free.b20.q1.d10.s42.i2.f4")
    assert (both.inst, both.fam, both.n_runs) == (2, 4, 1) and ms.Unit.parse(both.id) == both
    assert [x.fam for x in u.split_families()] == [0, 1, 2, 3, 4] and fam.split_families() == [fam]
    assert [x.id for x in fam.split()] == [f"SMAC.free.b20.q1.d10.s42.i{j}.f4" for j in range(3)]
    assert len(ms.runs_of(u)) == u.n_runs and ms.runs_of(both) == {both}
    for bad in (
        "core.free.b20.q4.d10",
        "core.free.20.q4.d10.s42",
        "nope.free.b20.q4.d10.s42",
        "core.x.b2.q1.d2.s1",
        "core.free.b20.q4.d10.s42.x0",
        "core.free.b20.q4.d10.s42.i3",
        "core.free.b20.q4.d10.s42.f5",  # the free preset has 5 families
        "core.failure.b20.q4.d5.s42.f4",  # the failure preset 4
        "core.free.b20.q4.d10.s42.f0.i0",  # instance first
        "core.free.b20.q4.d10.s42.i0.f0.x",
        # Not canonical: one spelling per unit (ids are matched as strings).
        "core.free.b20.q4.d10.s42.i-1",
        "core.free.b20.q4.d10.s42.f-1",
        "core.free.b20.q4.d10.s42.i01",
        "core.free.b20.q4.d10.s42.i0.f00",
        "core.free.b020.q4.d10.s42",
        "core.free.b20.q4.d10.s042",
    ):
        with pytest.raises(ValueError):
            ms.Unit.parse(bad)


def test_units_respect_q_rule_and_coverage():
    units = ms.make_units([42, 7], ["free", "failure"], [20, 100], None, [1, 4, 16, 64], list(ms.GROUPS))
    assert len(set(units)) == len(units)
    # q <= bm: q = 64 only at 100*d.
    assert all(u.q <= u.bm for u in units)
    assert {u.q for u in units if u.bm == 20} == {1, 4, 16}
    assert {u.q for u in units if u.bm == 100} == {1, 4, 16, 64}
    # SMAC: q = 1 only; the failure preset has no d = 10.
    assert {u.q for u in units if u.group == "SMAC"} == {1}
    assert not any(u.preset == "failure" and u.dim == 10 for u in units)
    # The core group covers the whole grid, the GP groups what their coverage admits.
    core = {(u.preset, u.bm, u.q, u.dim, u.seed) for u in units if u.group == "core"}
    for u in units:
        assert (u.preset, u.bm, u.q, u.dim, u.seed) in core
        assert ms.covered(u.group, u.bm, u.dim, u.q)
    # Coverage does not depend on q (SMAC aside): the pool stays fixed across q.
    for g in ("qLogEI", "TuRBO1"):
        for bm in (20, 100):
            for d in (2, 5, 10):
                assert len({ms.covered(g, bm, d, q) for q in (1, 4, 16, 64)}) == 1
    # Hours per run: no qLogEI or SMAC at d = 10, 100*d; TuRBO (one proposal per batch) runs there.
    assert ms.covered("qLogEI", 100, 5, 1) and not ms.covered("qLogEI", 100, 10, 1)
    assert not ms.covered("SMAC", 100, 10, 1) and not ms.covered("SMAC", 20, 2, 4)
    assert not ms.covered("SMAC", 100, 5, 1)
    assert ms.covered("TuRBO1", 100, 10, 1)
    assert not any(u.group == "qLogEI" and u.bm * u.dim > 500 for u in units)
    # The estimate grows with the budget; with q it shrinks for TuRBO and grows for qLogEI.
    assert ms.run_seconds("qLogEI", 10, 1000, 1) > ms.MAX_RUN_SECONDS > ms.run_seconds("qLogEI", 5, 500, 1)
    assert ms.run_seconds("SMAC", 5, 500, 1) > ms.MAX_RUN_SECONDS > ms.run_seconds("SMAC", 10, 200, 1)
    assert ms.run_seconds("TuRBO1", 5, 500, 16) < ms.run_seconds("TuRBO1", 5, 500, 1)
    assert ms.run_seconds("qLogEI", 5, 500, 64) > ms.run_seconds("qLogEI", 5, 500, 16)
    assert ms.run_seconds("qLogEI", 5, 500, 16) > ms.run_seconds("qLogEI", 5, 500, 1)
    # Outside the table: the same dimension's nearest budget (and q), scaled by the budget.
    assert ms.run_seconds("SMAC", 2, 40, 4) == ms.run_seconds("SMAC", 2, 40, 1)
    assert ms.run_seconds("TuRBO1", 2, 100, 1) > ms.run_seconds("TuRBO1", 2, 40, 1)
    assert ms.run_seconds("qLogEI", 5, 500, 32) == ms.run_seconds("qLogEI", 5, 500, 64)  # a log tie: the dearer
    # dims restricts; q == bm is admitted.
    assert {u.dim for u in ms.make_units([42], ["free"], [20], [2], [1], ["core"])} == {2}
    assert {u.q for u in ms.make_units([42], ["free"], [20], [2], [16, 20, 64], ["core"])} == {16, 20}
    with pytest.raises(ValueError):
        ms.make_units([42], ["nope"], [20], None, [1], ["core"])
    with pytest.raises(ValueError):
        ms.make_units([42], ["free"], [20], None, [1], ["nope"])


def test_plan_packs_every_run_once_by_group_and_splits_long_units():
    units = ms.make_units([42, 7, 1234], ["free"], [20, 100], None, [1, 4, 16, 64], list(ms.GROUPS))
    extra = [ms.Unit.parse("qLogEI.free.b100.q64.d5.s99.i0"), units[0]]  # the second is in the grid already
    entries = ms.plan(units, jobs=4, target_minutes=ms.TARGET_MINUTES, extra=extra)
    planned = [ms.Unit.parse(u) for e in entries for u in e["units"].split(";")]
    # Every single run of the grid exactly once, split or not; the extra unit's once more.
    runs = sorted((r for u in planned for r in ms.runs_of(u)), key=lambda u: u.id)
    expected = [r for u in units + [extra[0]] for r in ms.runs_of(u)]
    assert runs == sorted(expected, key=lambda u: u.id)
    by_id = {u.id: u for u in planned}
    # SMAC at d = 10, 20*d (80 min a unit on the runners, SMAC-01 of run 36274781342) is split by instance;
    # qLogEI at d = 5, 100*d, q = 64 (about 100 min an instance) by family, one round of 3 runs.
    assert "SMAC.free.b20.q1.d10.s42.i0" in by_id and "SMAC.free.b20.q1.d10.s42" not in by_id
    assert "qLogEI.free.b100.q64.d5.s42.f0" in by_id and "qLogEI.free.b100.q64.d5.s42.i0" not in by_id
    assert "qLogEI.free.b100.q1.d5.s42.i0" in by_id  # an instance fits: instance first
    assert "qLogEI.free.b20.q1.d2.s42" in by_id  # a unit that fits stays whole
    for e in entries:
        groups = {ms.Unit.parse(u).group for u in e["units"].split(";")}
        assert groups == {e["group"]}
        assert e["n_units"] == len(e["units"].split(";"))
        assert e["est_min"] <= ms.TARGET_MINUTES
    assert not ms.plan_problems(entries, ms.TARGET_MINUTES)
    assert entries[0]["group"] == "core"
    # The extra q = 64 instance does not fit either: split by family, one run each (~73 min), so even
    # packed every part needs a shard of its own.
    cal = [e for e in entries if e["calibration"]]
    assert [e["shard"] for e in cal] == [f"extra-{k:02d}" for k in range(1, 6)]
    assert [e["units"] for e in cal] == [f"qLogEI.free.b100.q64.d5.s99.i0.f{k}" for k in range(5)]
    assert len({e["shard"] for e in entries}) == len(entries)


def test_extra_units_are_deduplicated_against_the_grid():
    units = ms.make_units([42], ["free"], [100], [5], [16], ["qLogEI"])  # the whole unit ...q16.d5.s42
    assert [u.id for u in units] == ["qLogEI.free.b100.q16.d5.s42"]
    extra = [
        ms.Unit.parse("qLogEI.free.b100.q16.d5.s42.i0"),  # covered by the whole grid unit
        ms.Unit.parse("qLogEI.free.b100.q64.d5.s42.i0"),  # new
        ms.Unit.parse("qLogEI.free.b100.q64.d5.s42.i0"),  # a duplicate extra
        ms.Unit.parse("TuRBO1.free.b20.q4.d2.s7"),  # new, whole
    ]
    assert [u.id for u in ms.uncovered(extra, units)] == ["qLogEI.free.b100.q64.d5.s42.i0", "TuRBO1.free.b20.q4.d2.s7"]
    # Partly covered: only the uncovered instances remain.
    grid = [ms.Unit.parse("TuRBO1.free.b20.q4.d2.s7.i1")]
    left = ms.uncovered([ms.Unit.parse("TuRBO1.free.b20.q4.d2.s7")], grid)
    assert [u.id for u in left] == ["TuRBO1.free.b20.q4.d2.s7.i0", "TuRBO1.free.b20.q4.d2.s7.i2"]
    entries = ms.plan(units, 4, 180, extra)  # at 180 min the q = 64 instance is not split
    cal = [e for e in entries if e["calibration"]]
    assert [e["units"] for e in cal] == ["qLogEI.free.b100.q64.d5.s42.i0", "TuRBO1.free.b20.q4.d2.s7"]
    assert not any(e["calibration"] for e in entries if not e["shard"].startswith("extra-"))
    # Extras are packed like the grid (per group, up to the target), not one shard each.
    small = [ms.Unit.parse(f"TuRBO1.free.b20.q4.d2.s{s}") for s in (7, 8, 9)] + [
        ms.Unit.parse("SMAC.free.b20.q1.d2.s7")
    ]
    cal = [e for e in ms.plan([], 4, ms.TARGET_MINUTES, small) if e["calibration"]]
    assert [(e["shard"], e["group"], e["n_units"]) for e in cal] == [("extra-01", "TuRBO1", 3), ("extra-02", "SMAC", 1)]
    # The documented smoke example with the default seeds: every run once.
    grid5 = ms.make_units([42, 7, 1234, 2025, 3], ["free"], [20, 100], None, [1, 4, 16, 64], list(ms.GROUPS))
    smoke = [ms.Unit.parse(u) for u in ("qLogEI.free.b100.q16.d5.s42.i0", "qLogEI.free.b100.q64.d5.s42.i0")]
    runs = [
        r for e in ms.plan(grid5, 4, 180, smoke) for u in e["units"].split(";") for r in ms.runs_of(ms.Unit.parse(u))
    ]
    assert len(runs) == len(set(runs))


def test_split_to_fit_goes_by_instance_then_family_then_single_runs():
    u = ms.Unit.parse("qLogEI.free.b100.q64.d5.s42")
    assert ms.split_to_fit(u, 4, 1000) == [u]
    assert ms.split_to_fit(u, 4, 150) == u.split()  # 5 runs, 2 rounds: ~145 min
    assert ms.split_to_fit(u, 4, 90) == u.split_families()  # 3 runs, 1 round: ~73 min
    atoms = ms.split_to_fit(u, 4, 10)  # nothing fits: single runs, which plan_problems refuses
    assert len(atoms) == u.n_runs and all(a.n_runs == 1 for a in atoms)
    # The failure preset: an instance's 4 families are one round, as a family's 3 instances.
    f = ms.Unit.parse("qLogEI.failure.b100.q64.d5.s42")
    assert ms.split_to_fit(f, 4, 90) == f.split()


def test_plan_refuses_a_shard_above_the_target(capsys):
    """Exit 2 and no matrix when a shard is estimated above the target (it used to warn only)."""
    grid = ["plan", "--seeds", "1", "--dims", "5", "--budgets", "100", "--qs", "64", "--groups", "qLogEI"]
    assert ms.main(grid) == 0
    out = json.loads(capsys.readouterr().out)["include"]
    assert out and max(e["est_min"] for e in out) <= ms.TARGET_MINUTES
    # One run of qLogEI at d = 5, 500 evaluations, q = 64 takes over an hour: no split fits 30 minutes.
    assert ms.main(grid + ["--target-minutes", "30"]) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and "above the target 30" in captured.err
    # One line per refused unit of the grid, not one per single run it was split into.
    [line] = captured.err.strip().splitlines()
    assert (
        line.startswith("error: qLogEI.free.b100.q64.d5.s") and "15 shard(s)" in line and "the longest 73 min" in line
    )
    # An extra (calibration) unit is held to the same target.
    extra = ["plan", "--seeds", "1", "--dims", "2", "--qs", "1", "--groups", "core"]
    assert ms.main(extra + ["--extra-units", "SMAC.free.b20.q1.d10.s42.i0.f0", "--target-minutes", "20"]) == 2
    assert "extra-01" in capsys.readouterr().err
    # A target above the step limit is refused too.
    assert ms.main(extra + ["--target-minutes", str(ms.STEP_LIMIT_MINUTES + 1)]) == 2
    assert "step limit" in capsys.readouterr().err


def test_the_cost_model_covers_the_measured_units():
    """The worst unit walls of run 36274781342 (4-core runners, minutes) are within the estimate."""
    measured = {
        "SMAC.free.b20.q1.d10.s42": 80.1,  # SMAC-01 held four of these: 5.5 h, cut at the step limit
        "SMAC.free.b100.q1.d2.s42": 10.8,
        "qLogEI.free.b100.q1.d5.s42": 127.3,
        "qLogEI.free.b100.q16.d5.s42": 139.9,
        "qLogEI.free.b100.q64.d5.s42.i0": 106.4,
        "qLogEI.free.b100.q64.d2.s42": 54.7,
        "qLogEI.free.b20.q16.d10.s42": 49.1,
        "TuRBO1.free.b100.q1.d10.s42": 40.9,
        "TuRBO1.free.b100.q4.d10.s42": 13.9,
        "TuRBO1.free.b20.q1.d2.s42": 1.2,
        "core.free.b100.q64.d10.s42": 3.1,
        "core.free.b20.q1.d2.s42": 0.2,
    }
    for unit, minutes in measured.items():
        assert ms.unit_minutes(ms.Unit.parse(unit), 4) >= minutes, unit
    # The default grid: every shard within the target, well below the step limit.
    grid = ms.make_units(ms.resolve_seeds("5"), ["free", "failure"], [20, 100], None, [1, 4, 16, 64], list(ms.GROUPS))
    entries = ms.plan(grid, 4, ms.TARGET_MINUTES)
    assert not ms.plan_problems(entries, ms.TARGET_MINUTES) and len(entries) <= 256
    assert ms.TARGET_MINUTES < ms.STEP_LIMIT_MINUTES


def test_instances_select_the_family_and_the_instance():
    """``.f<k>`` is the k-th family in battery order (FREE_FAMILIES / FAILURE_FAMILIES)."""
    from panobbgo.harness_families import FAILURE_FAMILIES, FREE_FAMILIES

    for preset, fams in (("free", FREE_FAMILIES), ("failure", FAILURE_FAMILIES)):
        assert ms.PRESET_FAMILIES[preset] == len(fams)
        for k, cfg in enumerate(fams):
            sel = ms._instances(preset, 2, fam=k)
            assert [int(p.instance) for _, p in sel] == list(range(ms.N_INSTANCES))
            assert {str(p.family) for _, p in sel} == {cfg.name()}
    [(_, one)] = ms._instances("free", 2, inst=2, fam=1)
    assert (str(one.family), int(one.instance)) == (FREE_FAMILIES[1].name(), 2)
    assert len(ms._instances("free", 2)) == 15 and len(ms._instances("free", 2, inst=0)) == 5


def test_plan_cli_is_stdlib_only():
    code = (
        "import runpy, sys; sys.argv = ['measure.py', 'plan', '--seeds', '1', '--dims', '2', '--qs', '1,4',"
        " '--extra-units', 'qLogEI.free.b100.q16.d5.s42.i0;qLogEI.free.b100.q64.d5.s42.i0'];"
        f"\ntry:\n    runpy.run_path({str(_PATH)!r}, run_name='__main__')\nexcept SystemExit as e:\n    assert not e.code"
        "\nassert 'numpy' not in sys.modules and 'panobbgo' not in sys.modules, 'plan imported heavy modules'"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    matrix = json.loads(out.strip().splitlines()[0])["include"]
    ids = [u for e in matrix for u in e["units"].split(";")]
    assert "core.free.b20.q4.d2.s42" in ids and "SMAC.free.b20.q1.d2.s42" in ids
    assert not any(".q4." in u for u in ids if u.startswith("SMAC"))
    # The q = 64 instance (~145 min) does not fit the 90-minute target: one shard per family.
    assert [u for u in ids if ".q64." in u] == [f"qLogEI.free.b100.q64.d5.s42.i0.f{k}" for k in range(5)]


# ---------------------------------------------------------------------------
# aggregation on synthetic results
# ---------------------------------------------------------------------------


def _payload(
    unit: str,
    shard: str,
    scores: Mapping[str, Sequence[Optional[float]]],
    cpu: str = "cpu A",
    errors: Optional[Dict[int, str]] = None,
    calibration: bool = False,
) -> dict:
    """A unit result file: ``scores[strategy]`` = the AOCC per instance (aocc_time = AOCC / 2; None = crashed).

    Instances 0, 1 are the ellipsoid's 0, 1; instances 2, 3 rastrigin's 0, 1 (a ``.i<j>`` unit: the two
    families' instance ``j``; a ``.f<k>`` unit: family k's instances; both: that one run).  ``errors``: an error string per run index (e.g. ``EndedEarly``).
    """
    u = ms.Unit.parse(unit)

    def kind(i: int) -> str:
        if u.fam >= 0:  # a .f<k> unit: family k (0 = ellipsoid, 1 = rastrigin), its instances in order
            return ("ellipsoid", "rastrigin")[u.fam]
        return "ellipsoid" if (i < 2 if u.inst < 0 else i == 0) else "rastrigin"

    def inst(i: int) -> int:
        if u.inst >= 0:
            return u.inst
        return i if u.fam >= 0 else i % 2

    runs = [
        IOHRunRecord(
            problem_kind=kind(i),
            dim=u.dim,
            instance=inst(i),
            strategy_name=name,
            rep=0,
            budget=u.bm * u.dim,
            n_evals=u.bm * u.dim,
            best_fx=1.0,
            f_opt=0.0,
            aocc=0.0 if a is None else a,
            elapsed_s=1.0,
            seed=0,
            aocc_time=None if a is None else a / 2,
            error="RuntimeError: boom" if a is None else (errors or {}).get(i),
        )
        for name, vals in scores.items()
        for i, a in enumerate(vals)
    ]
    res = IOHHarnessResult("measure-free-b20", "families", -8.0, 2.0, runs, sync_eval=True, virtual={"workers": u.q})
    return {
        **asdict(u),
        "unit": unit,
        "shard": shard,
        "host": {"cpu_model": cpu, "avx512f": False},
        "git_sha": "abc1234",
        "elapsed_s": 1.0,
        "calibration": calibration,
        "result": res.to_dict(),
    }


def _write(tmp: Path, payloads: List[dict], meta: Optional[dict] = None) -> None:
    for p in payloads:
        d = tmp / p["shard"]
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{p['unit']}.json").write_text(json.dumps(p))
    if meta:
        (tmp / f"meta_{meta['shard']}.json").write_text(json.dumps(meta))


SEEDS = (42, 7, 1234)


def _grid(tmp: Path, qs=(1, 4), gp_seeds=SEEDS, crash_ngopt_at=None, bm=20) -> List[str]:
    """A synthetic run with seed-dependent scores: TuRBO in every q cell, SMAC at q = 1 only; returns the plan."""
    payloads, planned = [], []
    for q in qs:
        for j, seed in enumerate(SEEDS):
            b = 0.02 * j  # a seed effect every strategy shares
            ngopt: List[Optional[float]] = [0.40 + b, 0.50 + b, 0.45 + b, 0.40 + b]
            if crash_ngopt_at == (q, seed):
                ngopt[0] = None
            core: Dict[str, List[Optional[float]]] = {
                "Blocks_warm_CMAES_JSO": [0.50 + b + 0.01 * j, 0.60 + b, 0.25 + b, 0.55 + b],
                "RoundRobin_CMAES": [0.45 + b, 0.55 + b, 0.50 + b, 0.50 + b],
                "Baseline_NGOpt": ngopt,
                "Baseline_pycma_IPOP": [0.30 + b] * 4,
            }
            unit = f"core.free.b{bm}.q{q}.d2.s{seed}"
            planned.append(unit)
            payloads.append(_payload(unit, "core-01", core))
            turbo = f"TuRBO1.free.b{bm}.q{q}.d2.s{seed}"
            planned.append(turbo)
            if seed in gp_seeds:
                payloads.append(_payload(turbo, "TuRBO1-01", {"Baseline_TuRBO1": [0.42 + b] * 4}, cpu="cpu B"))
            if q == 1:
                smac = f"SMAC.free.b{bm}.q1.d2.s{seed}"
                planned.append(smac)
                payloads.append(_payload(smac, "SMAC-01", {"Baseline_SMAC_BB": [0.9 + b] * 4}, cpu="cpu B"))
    _write(tmp, payloads, meta={"shard": "core-01", "failed": [], "github_run_id": "123"})
    return planned


def test_aggregate_fixed_pool_best_of_and_headline(tmp_path):
    planned = _grid(tmp_path)
    missing = ms.aggregate(tmp_path, planned + ["core.free.b20.q4.d2.s99"])
    assert missing["missing_units"] == ["core.free.b20.q4.d2.s99"]
    # A missing unit keeps the pool; the q = 4 cell is flagged below the plan (12 of 16 runs).
    q4m = missing["cells"]["free/d2/b20/q4"]
    assert q4m["pool"] == ["Baseline_NGOpt", "Baseline_TuRBO1", "Baseline_pycma_IPOP"]
    assert (q4m["n_common"], q4m["planned_runs"]) == (12, 16)
    assert any("below the plan" in f for f in q4m["flags"])
    md = ms.summary_markdown(missing)
    assert "missing: core.free.b20.q4.d2.s99" in md and "12/16!" in md
    summary = ms.aggregate(tmp_path, planned)
    assert summary["github_run_id"] == ["123"]
    assert set(summary["fp_classes"]) == {"cpu A", "cpu B"}
    q1, q4 = summary["cells"]["free/d2/b20/q1"], summary["cells"]["free/d2/b20/q4"]
    # The pool is fixed across q: SMAC (q = 1 only) is a reference row, never the pool's best.
    assert q1["pool"] == q4["pool"] == ["Baseline_NGOpt", "Baseline_TuRBO1", "Baseline_pycma_IPOP"]
    assert q1["pool_best"]["aocc"] == "Baseline_NGOpt"
    assert not q1["strategies"]["Baseline_SMAC_BB"]["in_pool"]
    assert q1["headline_metric"] == "aocc" and q4["headline_metric"] == "aocc_time"
    assert "Baseline_SMAC_BB" in q1["planned"] and "Baseline_SMAC_BB" not in q4["planned"]
    h = q1["strategies"]["Blocks_warm_CMAES_JSO"]
    # Per seed j: mean(0.50+b+0.01j, 0.60+b, 0.25+b, 0.55+b) - mean(0.40+b, 0.50+b, 0.45+b, 0.40+b) = 0.0375 + 0.0025 j.
    st = h["vs_pool_best"]["aocc"]
    assert st["delta"] == pytest.approx(0.04)
    assert st["wins"] == 3 and st["n_seeds"] == 3 and st["ci_low"] < 0.04 < st["ci_high"] and 0 < st["p"] < 0.05
    # Against every external, SMAC included; a cross-job pair is only metadata.
    assert set(h["vs"]) == {"Baseline_NGOpt", "Baseline_TuRBO1", "Baseline_pycma_IPOP", "Baseline_SMAC_BB"}
    assert h["vs"]["Baseline_SMAC_BB"]["aocc"]["delta"] < 0 and h["vs"]["Baseline_TuRBO1"]["aocc"]["cross_fp"]
    # Per family: worse on rastrigin than the pool's best, better on the ellipsoid.
    pf = h["per_family_vs_pool_best"]
    assert pf["rastrigin"]["delta"] < 0 < pf["ellipsoid"]["delta"]
    assert set(h["per_family"]) == {"ellipsoid", "rastrigin"}
    # Holm over the headline cells.
    assert q1["headline"]["p_holm"] >= q1["headline"]["p"]
    assert q1["headline"]["vs"] == "Baseline_NGOpt"
    md = ms.summary_markdown(summary)
    assert "≈" not in md and "Holm" in md and "SMAC_BB (q=1 only)" in md
    assert "cannot be significant" in md and "conditional on the fixed instances" in md
    assert "(reference)" in md and "(pool)" in md


def test_a_missing_unit_keeps_the_pool_and_shrinks_n(tmp_path):
    # TuRBO misses seed 1234 everywhere (a cut shard): it stays in the pool; every comparison runs on the
    # runs present for the headline spec and every pool member, i.e. seeds 42 and 7.
    planned = _grid(tmp_path, gp_seeds=(42, 7))
    summary = ms.aggregate(tmp_path, planned)
    for c in ("free/d2/b20/q1", "free/d2/b20/q4"):
        cell = summary["cells"][c]
        assert cell["pool"] == ["Baseline_NGOpt", "Baseline_TuRBO1", "Baseline_pycma_IPOP"]
        turbo = cell["strategies"]["Baseline_TuRBO1"]
        assert (turbo["n_seeds"], turbo["planned_seeds"], turbo["complete"]) == (2, 3, False)
        assert (cell["n_common"], cell["n_common_seeds"], cell["planned_runs"]) == (8, 2, 12)
        assert any("below the plan" in f for f in cell["flags"])
        head = cell["headline"]
        assert head["n_seeds"] == 2 and head["n_pairs"] == 8
        # Every Δ of the cell on the same common runs, the headline's Holm set included.
        h = cell["strategies"][ms.HEADLINE_SPEC]
        assert all(st["aocc"]["n_seeds"] == 2 for st in h["vs"].values() if st["aocc"]["n_seeds"])
        assert cell["strategies"]["Baseline_NGOpt"]["n_common"] == 8
    # Means on the common runs: seeds 42 (b = 0) and 7 (b = 0.02) only.
    ngopt = summary["cells"]["free/d2/b20/q1"]["strategies"]["Baseline_NGOpt"]
    assert ngopt["aocc"] == pytest.approx(0.4375 + 0.01)
    md = ms.summary_markdown(summary)
    assert "2/3!" in md and "8/12!" in md


def test_pool_is_fixed_when_a_baseline_misses_a_q_cell(tmp_path):
    planned = _grid(tmp_path)
    for f in (tmp_path / "TuRBO1-01").glob("TuRBO1.free.b20.q4.*.json"):
        f.unlink()  # TuRBO present at q = 1 only
    summary = ms.aggregate(tmp_path)  # no plan: expected seeds are the cell's
    assert summary["cells"]["free/d2/b20/q1"]["pool"] == ["Baseline_NGOpt", "Baseline_pycma_IPOP"]
    assert summary["cells"]["free/d2/b20/q4"]["pool"] == ["Baseline_NGOpt", "Baseline_pycma_IPOP"]
    with_plan = ms.aggregate(tmp_path, planned)
    assert "Baseline_TuRBO1" in with_plan["cells"]["free/d2/b20/q4"]["missing"]


def test_errored_runs_score_zero_time_and_leave_the_pool(tmp_path):
    planned = _grid(tmp_path, crash_ngopt_at=(4, 7))
    cells = ms.collect(ms.load_units(tmp_path)[0])
    crashed = cells[("free", 2, 20, 4)]["Baseline_NGOpt"][(7, "ellipsoid", 0)]
    assert crashed.hard_error and crashed.aocc == 0.0 and crashed.aocc_time == 0.0
    summary = ms.aggregate(tmp_path, planned)
    q1, q4 = summary["cells"]["free/d2/b20/q1"], summary["cells"]["free/d2/b20/q4"]
    assert q4["strategies"]["Baseline_NGOpt"]["errors"] == 1
    # NGOpt errored in one q cell: out of the pool in every q cell of the (preset, dim, bm), flagged where it leads.
    assert "Baseline_NGOpt" not in q1["pool"] and "Baseline_NGOpt" not in q4["pool"]
    assert q1["pool_best"]["aocc"] == "Baseline_TuRBO1"
    assert q1["pool_excluded"]["Baseline_NGOpt"] == "errors at q=4"
    assert any("NGOpt" in f and "errors at q=4" in f for f in q1["flags"])
    assert "0/1" in ms.summary_markdown(summary)  # errors panobbgo/external in the headline table


def test_ended_early_is_not_an_error(tmp_path):
    payloads = [
        _payload(
            f"core.free.b20.q1.d2.s{seed}",
            "core-01",
            {ms.HEADLINE_SPEC: [0.5, 0.5, 0.5, 0.5], "Baseline_NGOpt": [0.4, 0.4, 0.4, 0.4]},
            errors={0: "EndedEarly: stopped at 30/40 evaluations"},
        )
        for seed in SEEDS
    ]
    _write(tmp_path, payloads)
    cell = ms.aggregate(tmp_path)["cells"]["free/d2/b20/q1"]
    ng = cell["strategies"]["Baseline_NGOpt"]
    assert ng["errors"] == 0 and ng["ended_early"] == 3 and ng["in_pool"]
    assert ng["aocc"] == pytest.approx(0.4) and ng["aocc_time"] == pytest.approx(0.2)


def test_aggregate_over_split_units_equals_the_whole(tmp_path):
    scores = {ms.HEADLINE_SPEC: [0.5, 0.6, 0.3, 0.2], "Baseline_NGOpt": [0.4, 0.5, 0.45, 0.4]}
    whole, split = tmp_path / "whole", tmp_path / "split"
    _write(whole, [_payload(f"core.free.b20.q1.d2.s{s}", "core-01", scores) for s in SEEDS])
    # The same runs as .i0 / .i1 units: (ellipsoid, rastrigin) instance j each.
    parts = []
    for s in SEEDS:
        for j in (0, 1):
            part = {n: [v[j], v[2 + j]] for n, v in scores.items()}
            parts.append(_payload(f"core.free.b20.q1.d2.s{s}.i{j}", "core-01", part))
    _write(split, parts)
    a = ms.aggregate(whole)["cells"]["free/d2/b20/q1"]
    b = ms.aggregate(split, [p["unit"] for p in parts])["cells"]["free/d2/b20/q1"]
    for key in ("pool", "pool_best", "n_common", "headline"):
        assert a[key] == b[key], key
    assert a["strategies"][ms.HEADLINE_SPEC]["aocc"] == b["strategies"][ms.HEADLINE_SPEC]["aocc"]


def test_aggregate_over_family_split_units_equals_the_whole(tmp_path):
    """``.f<k>`` and ``.i<j>.f<k>`` units aggregate to the whole; a missing ``.f`` unit is reported by its id."""
    scores = {ms.HEADLINE_SPEC: [0.5, 0.6, 0.3, 0.2], "Baseline_NGOpt": [0.4, 0.5, 0.45, 0.4]}
    whole, fams, atoms = tmp_path / "whole", tmp_path / "fams", tmp_path / "atoms"
    _write(whole, [_payload(f"core.free.b20.q1.d2.s{s}", "core-01", scores) for s in SEEDS])
    # Family k (0 = ellipsoid, 1 = rastrigin in _payload) holds runs 2k, 2k + 1 of the whole: instances 0, 1.
    by_fam = [
        _payload(f"core.free.b20.q1.d2.s{s}.f{k}", "core-01", {n: v[2 * k : 2 * k + 2] for n, v in scores.items()})
        for s in SEEDS
        for k in (0, 1)
    ]
    by_run = [
        _payload(f"core.free.b20.q1.d2.s{s}.i{j}.f{k}", "core-01", {n: [v[2 * k + j]] for n, v in scores.items()})
        for s in SEEDS
        for k in (0, 1)
        for j in (0, 1)
    ]
    _write(fams, by_fam)
    _write(atoms, by_run)
    a = ms.aggregate(whole)["cells"]["free/d2/b20/q1"]
    for src, parts in ((fams, by_fam), (atoms, by_run)):
        ids = [p["unit"] for p in parts]
        summary = ms.aggregate(src, ids)
        assert summary["missing_units"] == []
        b = summary["cells"]["free/d2/b20/q1"]
        for key in ("pool", "pool_best", "n_common", "headline"):
            assert a[key] == b[key], (src.name, key)
        assert a["strategies"][ms.HEADLINE_SPEC]["aocc"] == b["strategies"][ms.HEADLINE_SPEC]["aocc"]
        assert a["strategies"][ms.HEADLINE_SPEC]["per_family"] == b["strategies"][ms.HEADLINE_SPEC]["per_family"]
    # A planned .f unit that left no file (a cut shard) is missing by its id; the rest still aggregates.
    (fams / "core-01" / "core.free.b20.q1.d2.s7.f1.json").unlink()
    cut = ms.aggregate(fams, [p["unit"] for p in by_fam])
    assert cut["missing_units"] == ["core.free.b20.q1.d2.s7.f1"]
    assert cut["cells"]["free/d2/b20/q1"]["n_common"] < a["n_common"]


def test_calibration_units_stay_out_of_the_analysis(tmp_path):
    planned = _grid(tmp_path)
    # A calibration unit in a q cell the grid does not have (q = 16), for one GP baseline.
    cal = _payload(
        "TuRBO1.free.b20.q16.d2.s42.i0", "extra-01", {"Baseline_TuRBO1": [0.9, 0.9]}, cpu="cpu B", calibration=True
    )
    _write(tmp_path, [cal])
    summary = ms.aggregate(tmp_path, planned)
    assert "free/d2/b20/q16" not in summary["cells"]
    assert summary["cells"]["free/d2/b20/q1"]["pool"] == ["Baseline_NGOpt", "Baseline_TuRBO1", "Baseline_pycma_IPOP"]
    [row] = summary["calibration"]
    assert row["unit"] == "TuRBO1.free.b20.q16.d2.s42.i0" and row["runs"] == 2 and row["aocc"] == pytest.approx(0.9)
    assert "## Calibration" in ms.summary_markdown(summary)
    # A calibration unit with a grid unit's id never trips the duplicate check.
    dup = _payload("core.free.b20.q1.d2.s42", "extra-02", {"RoundRobin_CMAES": [0.5] * 4}, calibration=True)
    _write(tmp_path, [dup])
    ms.aggregate(tmp_path, planned)


def test_q_equals_bm_cell(tmp_path):
    planned = _grid(tmp_path, qs=(1, 20), bm=20)
    summary = ms.aggregate(tmp_path, planned)
    cell = summary["cells"]["free/d2/b20/q20"]
    assert cell["headline_metric"] == "aocc_time"
    assert cell["headline"]["delta"] == pytest.approx(
        cell["strategies"][ms.HEADLINE_SPEC]["vs"][cell["headline"]["vs"]]["aocc_time"]["delta"]
    )


def test_holm():
    adj = ms.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": float("nan")})
    assert adj["a"] == pytest.approx(0.03) and adj["c"] == pytest.approx(0.06) and adj["b"] == pytest.approx(0.06)
    assert adj["d"] != adj["d"]


def test_aggregate_refuses_a_duplicate_unit(tmp_path):
    p = _payload("core.free.b20.q1.d2.s42", "core-01", {"RoundRobin_CMAES": [0.5]})
    q = dict(p, shard="core-02")
    _write(tmp_path, [p, q])
    with pytest.raises(ValueError, match="twice"):
        ms.aggregate(tmp_path)


def test_load_units_skips_unreadable_files(tmp_path):
    _grid(tmp_path)
    (tmp_path / "core-01" / "core.free.b20.q1.d2.s5.json").write_text("{not json")
    (tmp_path / "core-01" / "stray.json").write_text("[1, 2]")
    payloads, bad = ms.load_units(tmp_path)
    assert len(bad) == 2 and payloads
    summary = ms.aggregate(tmp_path)
    assert len(summary["unreadable_files"]) == 2
    assert "unreadable" in ms.summary_markdown(summary)


def test_write_atomic(tmp_path):
    path = tmp_path / "sub" / "x.json"
    ms.write_atomic(path, "one")
    ms.write_atomic(path, "two")
    assert path.read_text() == "two"
    assert [p.name for p in path.parent.iterdir()] == ["x.json"]


# ---------------------------------------------------------------------------
# a tiny real run
# ---------------------------------------------------------------------------


def test_run_and_aggregate_end_to_end(tmp_path, monkeypatch):
    """The panobbgo specs only (no optional extra), instance 0 of the first family at d = 2, 20*d evaluations, q = 4."""
    monkeypatch.setitem(ms.GROUPS, "core", ())
    real_instances = ms._instances
    monkeypatch.setattr(
        ms, "_instances", lambda preset, dim, inst=-1, fam=-1: real_instances(preset, dim, inst, fam)[:1]
    )
    out = tmp_path / "raw" / "core-01"
    unit = "core.free.b20.q4.d2.s42.i0"
    rc = ms.main(["run", "--units", unit, "--shard", "core-01", "--out-dir", str(out), "--jobs", "1", "--no-nice"])
    assert rc == 0
    meta = json.loads((out / "meta_core-01.json").read_text())
    assert meta["done"] == [unit] and meta["status"] == 0
    assert meta["host"]["cpu_count"]
    payload = json.loads((out / f"{unit}.json").read_text())
    res = IOHHarnessResult.from_dict(payload["result"])
    assert res.virtual is not None and res.virtual["workers"] == 4 and res.virtual["model"] == "lognormal"
    assert res.virtual["durations"] == "crn"
    assert {r.strategy_name for r in res.runs} == {s.name for s in make_ioh_strategies()}
    assert all(r.budget == 40 and r.aocc_time is not None and r.instance == 0 for r in res.runs)
    summary = ms.aggregate(tmp_path / "raw")
    cell = summary["cells"]["free/d2/b20/q4"]
    assert cell["pool"] == [] and cell["pool_best"]["aocc"] is None  # no baseline ran
    assert set(cell["strategies"]) == {s.name for s in make_ioh_strategies()}
    ms.summary_markdown(summary)


def test_run_unit_honours_the_instance_index(monkeypatch):
    """No ``_instances`` patch: a ``.i<j>`` unit runs instance j of every family, nothing else."""
    only = [s for s in make_ioh_strategies() if s.name == "RoundRobin_Random"]
    monkeypatch.setattr(ms, "_strategies", lambda group: only)
    payload = ms.run_unit(ms.Unit.parse("core.free.b20.q4.d2.s42.i1"), jobs=1)
    runs = payload["result"]["runs"]
    assert payload["inst"] == 1
    assert {r["instance"] for r in runs} == {1}
    assert len(runs) == ms.PRESET_FAMILIES["free"] == len({r["problem_kind"] for r in runs})


def test_the_workflow_matches_the_script():
    import yaml

    workflow = yaml.safe_load((ms.REPO_ROOT / ".github" / "workflows" / "measure.yml").read_text())
    inputs = workflow[True]["workflow_dispatch"]["inputs"]  # YAML 1.1: on -> True
    for group in ms.GROUPS:
        assert group in inputs["groups"]["description"], f"{group} missing from the groups input"
    for preset in ms.PRESET_DIMS:
        assert preset in inputs["presets"]["description"]
    defaults = vars(ms.build_parser().parse_args(["plan"]))
    for key in ("seeds", "presets", "budgets", "qs", "groups"):
        assert str(inputs[key]["default"]) == str(defaults[key]), key
    assert "extra_units" in inputs
    steps = workflow["jobs"]["measure"]["steps"]
    [install] = [s for s in steps if s.get("name") == "Install dependencies"]
    for extra in ("dev", "baselines", "baselines-bo"):
        assert f"--extra {extra}" in install["run"]
    [measure] = [s for s in steps if s.get("name") == "Measure"]
    # The step's own limit sits below the job's, so the upload still runs.
    assert measure["timeout-minutes"] == ms.STEP_LIMIT_MINUTES
    assert ms.TARGET_MINUTES < measure["timeout-minutes"] < workflow["jobs"]["measure"]["timeout-minutes"] <= 150
    assert "--calibration" in measure["run"] and measure["env"]["CALIBRATION"] == "${{ matrix.calibration }}"
    [upload] = [s for s in steps if s.get("name") == "Upload shard results"]
    assert upload["if"] == "always()" and upload["with"]["overwrite"] is True
    [summary_upload] = [s for s in workflow["jobs"]["aggregate"]["steps"] if s.get("name") == "Upload the summary"]
    assert summary_upload["with"]["overwrite"] is True
    # Artifacts only: nothing in this workflow may write to the repository.
    assert workflow["permissions"] == {"contents": "read"}
    assert all("permissions" not in job for job in workflow["jobs"].values())


# ---------------------------------------------------------------------------
# the opt-in trq group (the opt-in candidates: DISCOVERY §66, §69.5, §72) and the ex-ellipsoid view
# ---------------------------------------------------------------------------


def _plan_ids(*argv: str) -> List[str]:
    """The unit ids of ``measure.py plan --seeds 2 *argv``, run as the workflow runs it."""
    code = (
        "import runpy, sys; sys.argv = ['measure.py', 'plan', '--seeds', '2'"
        + "".join(f", {a!r}" for a in argv)
        + f"]\nrunpy.run_path({str(_PATH)!r}, run_name='__main__')"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    matrix = json.loads(proc.stdout.strip().splitlines()[0])["include"]
    return [u for e in matrix for u in e["units"].split(";")]


def test_trq_group_is_opt_in_and_runs_the_core_cells():
    from panobbgo.harness_ioh import (
        BLOCKS_VARIANT_NAMES,
        TRUST_REGION_NAMES,
        make_blocks_variant_strategies,
        make_trust_region_strategies,
    )

    # The plan's copy of the names (plan imports no panobbgo) is the registry's: the opt-in candidates are
    # the trust-region specs (§66, §69.5's RoundRobin_TRQ_r05) and the Blocks variants (§72's gate).
    assert ms.TRUST_REGION_SPECS == TRUST_REGION_NAMES and "RoundRobin_TRQ_r05" in ms.TRUST_REGION_SPECS
    assert {"RoundRobin_TRQ_fa", "RoundRobin_TRQ_aware"} <= set(ms.TRUST_REGION_SPECS)  # §71's candidates
    assert ms.BLOCKS_VARIANT_SPECS == BLOCKS_VARIANT_NAMES == ("Blocks_warm_CMAES_JSO_dimbudget",)
    assert ms.TRQ_SPECS == TRUST_REGION_NAMES + BLOCKS_VARIANT_NAMES
    assert [s.name for s in make_trust_region_strategies(ms.TRUST_REGION_SPECS)] == list(ms.TRUST_REGION_SPECS)
    assert [s.name for s in make_blocks_variant_strategies(ms.BLOCKS_VARIANT_SPECS)] == list(ms.BLOCKS_VARIANT_SPECS)
    assert not any(ms.is_external(n) for n in ms.TRQ_SPECS) and ms.GROUPS["trq"] == ()
    assert all(ms.group_of(n) == "trq" for n in ms.TRQ_SPECS) and ms.group_of("RoundRobin_CMAES") == "core"
    specs = ms._strategies("trq")
    assert [s.name for s in specs] == list(ms.TRQ_SPECS)
    # Every candidate draws the RNG streams of the spec it is compared with (paired deltas carry the option).
    rng = {s.name: s.rng_identity for s in specs}
    assert rng["RoundRobin_TRQ_r05"] == rng["RoundRobin_TRQ"] == "RoundRobin_TRQ"
    assert rng["RoundRobin_TRQ_fa"] == rng["RoundRobin_TRQ_aware"] == "RoundRobin_TRQ"
    assert rng["Blocks_warm_CMAES_JSO_dimbudget"] == rng["Blocks_warm_CMAES_JSO_TRQ"] == ms.HEADLINE_SPEC
    # 'all' is the measurement of record: core and the GP groups, not trq.
    assert "trq" not in ms.DEFAULT_GROUPS and set(ms.DEFAULT_GROUPS) | {"trq"} == set(ms.GROUPS)
    default = _plan_ids()
    assert default and not any(u.startswith("trq.") for u in default)
    assert _plan_ids("--groups", "all") == default
    assert {ms.Unit.parse(u).group for u in default} == set(ms.DEFAULT_GROUPS)
    # Named: the same cells and seeds as core (every cell, q <= bm), nothing else.
    trq = _plan_ids("--groups", "trq")
    core = [u for u in default if u.startswith("core.")]
    assert sorted(u.removeprefix("trq.") for u in trq) == sorted(u.removeprefix("core.") for u in core)
    both = _plan_ids("--groups", "core,qLogEI,TuRBO1,SMAC,trq")
    assert sorted(both) == sorted(default + trq)
    # Packed like core: shards of at most CORE_TARGET_MINUTES, trq units only, after the other groups.
    units = ms.make_units(ms.resolve_seeds("5"), ["free", "failure"], [20, 100], None, [1, 4, 16, 64], ["trq"])
    entries = ms.plan(units, 4, ms.TARGET_MINUTES)
    assert entries and all(e["group"] == "trq" and e["shard"].startswith("trq-") for e in entries)
    assert all(e["est_min"] <= ms.CORE_TARGET_MINUTES for e in entries)
    assert not ms.plan_problems(entries, ms.TARGET_MINUTES)
    # The (estimated) cost table has every cell of the free grid.
    for u in units:
        assert (u.dim, u.bm * u.dim, u.q) in ms.RUNNER_SECONDS["trq"]
    # The estimate covers all seven specs: the first three's laptop times plus the later candidates'.
    tables = (ms.TRQ_LAPTOP_SECONDS, ms.CANDIDATE_LAPTOP_SECONDS, ms.FAILURE_CANDIDATE_LAPTOP_SECONDS)
    assert all(set(t) == set(ms.RUNNER_SECONDS["trq"]) for t in tables)
    for k in ms.TRQ_LAPTOP_SECONDS:
        want = sum(t[k] for t in tables) * ms.TRQ_RUNNER_FACTOR
        assert want <= ms.RUNNER_SECONDS["trq"][k] < want + 1 and all(t[k] > 0 for t in tables)


def _with_trq(tmp: Path, planned: List[str], qs=(1, 4), bm=20) -> List[str]:
    """Add trq units to a :func:`_grid` run: RoundRobin_TRQ great on the ellipsoid only, the others middling.

    The §69.5 / §72 candidates: ``RoundRobin_TRQ_r05`` 0.10 below RoundRobin_TRQ on the ellipsoid and 0.05
    above on rastrigin, ``Blocks_warm_CMAES_JSO_dimbudget`` the headline spec's scores exactly (the gate does not
    bind) except +0.1 on run 2 (rastrigin instance 0) of seed 7 at q = 4.  §71's candidates:
    ``RoundRobin_TRQ_aware`` RoundRobin_TRQ's scores exactly (no failures), ``RoundRobin_TRQ_fa`` too except
    +0.04 on rastrigin at q = 1.
    """
    payloads = []
    for q in qs:
        for j, seed in enumerate(SEEDS):
            b = 0.02 * j
            unit = f"trq.free.b{bm}.q{q}.d2.s{seed}"
            planned = planned + [unit]
            scores: Dict[str, List[Optional[float]]] = {
                "RoundRobin_TRQ": [0.95 + b, 0.95 + b, 0.30 + b, 0.30 + b],
                "Blocks_warm_CMAES_JSO_TRQ": [0.70 + b, 0.70 + b, 0.35 + b, 0.35 + b],
                "RoundRobin_COBYQA": [0.20 + b] * 4,
            }
            scores["RoundRobin_TRQ_r05"] = [0.85 + b, 0.85 + b, 0.35 + b, 0.35 + b]
            # The same expressions as _grid's headline row, so the values are bit-equal.
            gated: List[Optional[float]] = [0.50 + b + 0.01 * j, 0.60 + b, 0.25 + b, 0.55 + b]
            if (q, seed) == (4, 7):
                gated[2] = 0.35 + b
            scores["Blocks_warm_CMAES_JSO_dimbudget"] = gated
            # The same expressions as RoundRobin_TRQ's row, so the values are bit-equal.
            scores["RoundRobin_TRQ_aware"] = [0.95 + b, 0.95 + b, 0.30 + b, 0.30 + b]
            fa = 0.04 if q == 1 else 0.0
            scores["RoundRobin_TRQ_fa"] = [0.95 + b, 0.95 + b, 0.30 + b + fa, 0.30 + b + fa]
            payloads.append(_payload(unit, "trq-01", scores))
    _write(tmp, payloads)
    return planned


def test_aggregate_treats_trq_specs_as_secondary(tmp_path):
    planned = _grid(tmp_path)
    before = ms.aggregate(tmp_path, planned)
    planned = _with_trq(tmp_path, planned)
    summary = ms.aggregate(tmp_path, planned)
    assert summary["headline_spec"] == ms.HEADLINE_SPEC == "Blocks_warm_CMAES_JSO"
    for c in ("free/d2/b20/q1", "free/d2/b20/q4"):
        cell, old = summary["cells"][c], before["cells"][c]
        # Not in the pool, not the headline: pool, best-of and the headline (Holm family) are unchanged.
        assert cell["pool"] == old["pool"] and cell["pool_best"] == old["pool_best"]
        assert cell["headline"] == old["headline"]
        assert cell["n_common"] == old["n_common"] and not set(ms.TRQ_SPECS) & set(cell["missing"])
        for n in ms.TRQ_SPECS:
            assert n in cell["planned"] and n in cell["present"]
            r = cell["strategies"][n]
            assert (r["external"], r["in_pool"], r["group"]) == (False, False, "trq")
            assert r["vs_pool_best"]["aocc"]["n_seeds"] == 3 and r["complete"]
    md = ms.summary_markdown(summary)
    headline = md.split("## Headline\n", 1)[1].split("\n## ", 1)[0]
    # The best other panobbgo spec is a trq spec: RoundRobin_TRQ_fa (0.645 + b at q = 1; at q = 4 it ties RoundRobin_TRQ
    # and _aware at 0.625 + b, and the tie goes to the larger name) against RoundRobin_CMAES' 0.5 + b.
    rows = [line for line in headline.splitlines() if line.startswith("| free/")]
    assert len(rows) == 2 and all("| RoundRobin_TRQ_fa 0." in line for line in rows)
    assert "| RoundRobin_COBYQA |" in md and "| Blocks_warm_CMAES_JSO_TRQ |" in md  # the per-cell tables


def test_trq_at_a_q_the_grid_lacks_leaves_the_pool_and_headline(tmp_path):
    """A q cell only trq ran (no external) must not empty the pool of its (preset, dim, bm)."""
    planned = _grid(tmp_path)
    before = ms.aggregate(tmp_path, planned)
    planned = _with_trq(tmp_path, planned, qs=(1, 4, 16))
    summary = ms.aggregate(tmp_path, planned)
    for c in ("free/d2/b20/q1", "free/d2/b20/q4"):
        cell, old = summary["cells"][c], before["cells"][c]
        assert cell["pool"] == old["pool"] == ["Baseline_NGOpt", "Baseline_TuRBO1", "Baseline_pycma_IPOP"]
        assert cell["pool_excluded"] == old["pool_excluded"] and cell["pool_best"] == old["pool_best"]
        assert cell["headline"] == old["headline"] is not None
    only_trq = summary["cells"]["free/d2/b20/q16"]
    assert only_trq["pool"] == [] and only_trq["headline"] is None
    assert any("panobbgo specs only" in f for f in only_trq["flags"])
    ms.summary_markdown(summary)


def test_a_missing_trq_unit_leaves_n_pool_and_headline(tmp_path):
    """A planned trq unit without a result (a cut trq shard) is reported missing and changes nothing else."""
    planned = _grid(tmp_path)
    before = ms.aggregate(tmp_path, planned)
    planned = _with_trq(tmp_path, planned)
    (tmp_path / "trq-01" / "trq.free.b20.q1.d2.s7.json").unlink()
    summary = ms.aggregate(tmp_path, planned)
    assert summary["missing_units"] == ["trq.free.b20.q1.d2.s7"]
    for c in ("free/d2/b20/q1", "free/d2/b20/q4"):
        cell, old = summary["cells"][c], before["cells"][c]
        assert (cell["n_common"], cell["planned_runs"]) == (old["n_common"], old["planned_runs"])
        assert cell["pool"] == old["pool"] and cell["pool_best"] == old["pool_best"]
        assert cell["headline"] == old["headline"]
    trq = summary["cells"]["free/d2/b20/q1"]["strategies"]["RoundRobin_TRQ"]
    assert (trq["n_seeds"], trq["planned_seeds"], trq["complete"]) == (2, 3, False)
    assert "missing: trq.free.b20.q1.d2.s7" in ms.summary_markdown(summary)


def test_ex_ellipsoid_view_is_descriptive_and_reselects_the_pool_best(tmp_path):
    planned = _with_trq(tmp_path, _grid(tmp_path))
    summary = ms.aggregate(tmp_path, planned)
    q1 = summary["cells"]["free/d2/b20/q1"]
    ex = q1["ex_ellipsoid"]
    assert ex["excluded"] == ["ellipsoid"] and ex["n_common"] == q1["n_common"] // 2 == 6
    # Rastrigin only: NGOpt 0.425 + b, TuRBO 0.42 + b, IPOP 0.30 + b (SMAC is outside the pool).
    assert ex["pool_best"]["aocc"] == "Baseline_NGOpt"
    assert ex["strategies"]["Baseline_NGOpt"]["aocc"] == pytest.approx(0.425 + 0.02)
    # Blocks on rastrigin 0.40 + b: -0.025 against NGOpt, where it is +0.04 on all families.
    h = ex["strategies"][ms.HEADLINE_SPEC]["vs_pool_best"]["aocc"]
    assert h["delta"] == pytest.approx(-0.025) and h["wins"] == 0 and h["n_seeds"] == 3
    assert q1["headline"]["delta"] == pytest.approx(0.04)  # the Holm family stays over all families
    # RoundRobin_TRQ leads on all families only through the ellipsoid.
    assert q1["strategies"]["RoundRobin_TRQ"]["vs_pool_best"]["aocc"]["delta"] > 0
    assert ex["strategies"]["RoundRobin_TRQ"]["vs_pool_best"]["aocc"]["delta"] == pytest.approx(0.30 - 0.425)
    assert "vs_pool_best" not in ex["strategies"]["Baseline_NGOpt"]
    md = ms.summary_markdown(summary)
    table = md.split("## Without the ellipsoid family (descriptive)\n", 1)[1].split("\n## ", 1)[0]
    assert "**Descriptive**" in table and "not part of the Holm family" in table
    rows = [line for line in table.splitlines() if line.startswith("| free/")]
    assert len(rows) == 2
    # Without the ellipsoid the best other panobbgo spec is RoundRobin_CMAES (0.50 + b), not RoundRobin_TRQ.
    assert all("| RoundRobin_CMAES 0." in line and "NGOpt" in line for line in rows)
    # A cell without the family (or with nothing else) has no such view.
    cells = ms.collect(ms.load_units(tmp_path)[0])
    strats = cells[("free", 2, 20, 1)]
    only_rastrigin = {k for k in strats[ms.HEADLINE_SPEC] if k[1] == "rastrigin"}
    only_ellipsoid = {k for k in strats[ms.HEADLINE_SPEC] if k[1] == "ellipsoid"}
    assert ms.ex_ellipsoid(strats, ["Baseline_NGOpt"], only_rastrigin) is None
    assert ms.ex_ellipsoid(strats, ["Baseline_NGOpt"], only_ellipsoid) is None
    assert ms.is_ex_family("ellipsoid_fhs_crash") and not ms.is_ex_family("rastrigin")


def test_secondary_specs_get_a_paired_delta_against_the_headline(tmp_path):
    """Every secondary panobbgo spec, the trq group's across jobs included, is paired with the headline spec."""
    planned = _grid(tmp_path)
    before = ms.aggregate(tmp_path, planned)
    planned = _with_trq(tmp_path, planned)
    summary = ms.aggregate(tmp_path, planned)
    for c in ("free/d2/b20/q1", "free/d2/b20/q4"):
        cell, old = summary["cells"][c], before["cells"][c]
        # Descriptive: the pool, its best, n and the headline (the Holm family) do not move.
        assert cell["headline"] == old["headline"] and cell["pool_best"] == old["pool_best"]
        assert cell["n_common"] == old["n_common"] == 12
        rows = cell["strategies"]
        assert rows[ms.HEADLINE_SPEC]["vs_headline"] is None
        assert all("vs_headline" not in r for r in rows.values() if r["external"])
        secondary = {n for n, r in rows.items() if r.get("vs_headline")}
        assert secondary == {"RoundRobin_CMAES", *ms.TRQ_SPECS}
        for n in secondary:
            st = rows[n]["vs_headline"]
            assert set(st) == set(ms.METRICS)
            assert all(st[m]["n_seeds"] == 3 and st[m]["n_pairs"] == 12 for m in ms.METRICS)
        # The trq units ran in another job than the headline spec's: the pairs cross jobs (metadata only).
        assert rows["RoundRobin_TRQ"]["vs_headline"]["aocc"]["cross_job"]
        assert not rows["RoundRobin_CMAES"]["vs_headline"]["aocc"]["cross_job"]
    q1, q4 = (summary["cells"][f"free/d2/b20/q{q}"]["strategies"] for q in (1, 4))
    # Per seed j: RoundRobin_TRQ 0.625 + b against the headline's 0.475 + b + 0.0025 j.
    st = q1["RoundRobin_TRQ"]["vs_headline"]["aocc"]
    assert st["delta"] == pytest.approx(0.1475) and st["wins"] == 3 and st["n_equal"] == 0
    assert q4["RoundRobin_TRQ"]["vs_headline"]["aocc_time"]["delta"] == pytest.approx(0.1475 / 2)
    assert q1["RoundRobin_TRQ_r05"]["vs_headline"]["aocc"]["delta"] == pytest.approx(0.1475 - 0.025)
    # The gated variant where the gate does not bind: identical to the headline spec, Δ exactly 0.
    gate1 = q1["Blocks_warm_CMAES_JSO_dimbudget"]["vs_headline"]
    for m in ms.METRICS:
        assert gate1[m]["delta"] == 0.0 and gate1[m]["wins"] == 0 and gate1[m]["n_equal"] == 12
    # One run differs (seed 7 at q = 4, +0.1 AOCC on 1 of its 4 runs).
    gate4 = q4["Blocks_warm_CMAES_JSO_dimbudget"]["vs_headline"]
    assert gate4["aocc"]["delta"] == pytest.approx(0.1 / 4 / 3) and gate4["aocc"]["wins"] == 1
    assert gate4["aocc_time"]["n_equal"] == 11 and gate4["aocc_time"]["delta"] == pytest.approx(0.1 / 8 / 3)
    md = ms.summary_markdown(summary)
    table = md.split(f"## Secondary panobbgo specs − {ms.HEADLINE_SPEC} (descriptive)\n", 1)[1].split("\n## ", 1)[0]
    assert "**Descriptive**" in table and "not part of the Holm family" in table
    rows_md = [line for line in table.splitlines() if line.startswith("| free/")]
    assert len(rows_md) == 2 * 8
    # core rows first, then trq; the metric column is the cell's headline metric.
    q1_rows = [line for line in rows_md if line.startswith("| free/d2/b20/q1 |")]
    assert q1_rows[0].startswith("| free/d2/b20/q1 | RoundRobin_CMAES | core | 12 | aocc |")
    [gated] = [line for line in q1_rows if "| Blocks_warm_CMAES_JSO_dimbudget | trq |" in line]
    assert gated.endswith("| 12/12 |") and "+0.000" in gated
    [gated4] = [line for line in rows_md if line.startswith("| free/d2/b20/q4 | Blocks_warm_CMAES_JSO_dimbudget |")]
    assert "| aocc_time |" in gated4 and gated4.endswith("| 11/12 |") and " 1/3 |" in gated4
    # Ties at 0 are shown apart; pairs across FP classes are marked.
    q4c = summary["cells"]["free/d2/b20/q4"]
    st = q4c["strategies"]["Blocks_warm_CMAES_JSO_dimbudget"]["vs_headline"]
    st["aocc_time"].update(n_equal=10, n_zero_ties=1, cross_fp=True)
    [marked] = [
        line
        for line in ms.summary_markdown(summary).splitlines()
        if line.startswith("| free/d2/b20/q4 | Blocks_warm_CMAES_JSO_dimbudget |")
    ]
    assert marked.endswith("| 10/12 (+1 at 0) (FP) |")
    # Without the trq group the core secondaries still get the table.
    assert "## Secondary panobbgo specs" in ms.summary_markdown(before)


def test_vs_headline_counts_equal_pairs_on_the_given_keys():
    obs = lambda v: ms.Obs(aocc=v, aocc_time=v / 2, shard="s", fp="f", error=None, elapsed_s=1.0)  # noqa: E731
    head = {(1, "a", 0): obs(0.5), (1, "a", 1): obs(0.4), (2, "a", 0): obs(0.3)}
    other = {(1, "a", 0): obs(0.5), (1, "a", 1): obs(0.6), (2, "a", 0): obs(0.3), (3, "a", 0): obs(0.9)}
    st = ms.vs_headline(other, head, [(1, "a", 0), (1, "a", 1), (2, "a", 0), (3, "a", 0)])
    assert st["aocc"]["n_pairs"] == 3 and st["aocc"]["n_equal"] == 2 and st["aocc"]["n_seeds"] == 2
    assert st["aocc"]["delta"] == pytest.approx((0.1 + 0.0) / 2)
    st = ms.vs_headline(other, head, [(2, "a", 0)])
    assert st["aocc_time"]["n_equal"] == 1 and st["aocc_time"]["delta"] == 0.0
    # Ties at the score floor are not "equal"; a missing time score (None) and a key the spec lacks drop out.
    head[(4, "a", 0)], other[(4, "a", 0)] = obs(0.0), obs(0.0)
    head[(5, "a", 0)] = obs(0.7)
    other[(5, "a", 0)] = ms.Obs(aocc=0.7, aocc_time=None, shard="s", fp="f", error=None, elapsed_s=1.0)
    head[(6, "a", 0)] = obs(0.2)  # the spec has no such run
    keys = [(4, "a", 0), (5, "a", 0), (6, "a", 0), (2, "a", 0)]
    st = ms.vs_headline(other, head, keys)
    assert (st["aocc"]["n_pairs"], st["aocc"]["n_equal"], st["aocc"]["n_zero_ties"]) == (3, 2, 1)
    assert (st["aocc_time"]["n_pairs"], st["aocc_time"]["n_equal"], st["aocc_time"]["n_zero_ties"]) == (2, 1, 1)


def test_variants_get_a_paired_delta_against_their_base_spec(tmp_path):
    """A spec on another spec's RNG streams (``seed_name``) is paired with that spec: r05 / fa / aware − TRQ."""
    planned = _with_trq(tmp_path, _grid(tmp_path))
    summary = ms.aggregate(tmp_path, planned)
    q1, q4 = (summary["cells"][f"free/d2/b20/q{q}"]["strategies"] for q in (1, 4))
    for rows in (q1, q4):
        # The synthetic payloads record no rng_identity: the registry's seed_name decides the base.
        for n in ("RoundRobin_TRQ_r05", "RoundRobin_TRQ_fa", "RoundRobin_TRQ_aware"):
            assert rows[n]["base_spec"] == "RoundRobin_TRQ" and set(rows[n]["vs_base"]) == set(ms.METRICS)
        for n in ("Blocks_warm_CMAES_JSO_dimbudget", "Blocks_warm_CMAES_JSO_TRQ"):
            assert rows[n]["base_spec"] == ms.HEADLINE_SPEC
            assert rows[n]["vs_base"] == rows[n]["vs_headline"]  # the same pairs, the same numbers
        # Specs on their own streams have no base; externals get no row at all.
        for n in ("RoundRobin_TRQ", "RoundRobin_CMAES", "RoundRobin_COBYQA", ms.HEADLINE_SPEC):
            assert rows[n]["base_spec"] is None and rows[n]["vs_base"] is None
        assert all("vs_base" not in r for r in rows.values() if r["external"])
    # r05: −0.10 on the ellipsoid, +0.05 on rastrigin, per seed −0.025.
    st = q1["RoundRobin_TRQ_r05"]["vs_base"]["aocc"]
    assert st["delta"] == pytest.approx(-0.025) and st["wins"] == 0 and st["n_pairs"] == 12
    # aware: bit-identical everywhere; fa: +0.04 on rastrigin at q = 1 (+0.02 a seed), identical at q = 4.
    for rows in (q1, q4):
        aware = rows["RoundRobin_TRQ_aware"]["vs_base"]
        assert all(aware[m]["delta"] == 0.0 and aware[m]["n_equal"] == 12 for m in ms.METRICS)
    fa1, fa4 = q1["RoundRobin_TRQ_fa"]["vs_base"], q4["RoundRobin_TRQ_fa"]["vs_base"]
    assert fa1["aocc"]["delta"] == pytest.approx(0.02) and fa1["aocc"]["wins"] == 3 and fa1["aocc"]["n_equal"] == 6
    assert fa4["aocc_time"]["delta"] == 0.0 and fa4["aocc_time"]["n_equal"] == 12
    # The Holm family and the pool do not move.
    before = ms.aggregate(tmp_path, planned)
    assert summary["cells"]["free/d2/b20/q1"]["headline"] == before["cells"]["free/d2/b20/q1"]["headline"]
    md = ms.summary_markdown(summary)
    table = md.split("## Variants − their base spec (same RNG streams, descriptive)\n", 1)[1].split("\n## ", 1)[0]
    assert "**Descriptive**" in table and "not part of the Holm family" in table
    rows_md = [line for line in table.splitlines() if line.startswith("| free/")]
    # The three TRQ variants per cell; the headline's variants are in the vs-headline table only.
    assert len(rows_md) == 2 * 3 and all("| RoundRobin_TRQ |" in line for line in rows_md)
    [fa_q1] = [line for line in rows_md if line.startswith("| free/d2/b20/q1 | RoundRobin_TRQ_fa |")]
    assert "| aocc | +0.020 " in fa_q1 and fa_q1.endswith("| 6/12 |")
    [aware_q4] = [line for line in rows_md if line.startswith("| free/d2/b20/q4 | RoundRobin_TRQ_aware |")]
    assert "| aocc_time | +0.000 " in aware_q4 and aware_q4.endswith("| 12/12 |")


def test_base_specs_reads_the_recorded_rng_identity_and_falls_back_to_the_registry():
    def payload(unit: str, ids: Optional[Dict[str, str]], names: Sequence[str]) -> dict:
        d = _payload(unit, "s", {n: [0.5] * 4 for n in names})
        if ids is not None:
            d["rng_identity"] = ids
        return d

    recorded = payload(
        "trq.free.b20.q1.d2.s42",
        {"RoundRobin_TRQ": "RoundRobin_TRQ", "X_variant": "X", "RoundRobin_TRQ_fa": "RoundRobin_TRQ"},
        ["RoundRobin_TRQ", "X_variant", "RoundRobin_TRQ_fa"],
    )
    old = payload("trq.free.b20.q4.d2.s42", None, ["RoundRobin_TRQ_r05", "Unknown_spec"])
    bases = ms.base_specs([recorded, old])
    # Recorded identities win; an old payload falls back to the registry; an unknown name has no base.
    assert bases == {"X_variant": "X", "RoundRobin_TRQ_fa": "RoundRobin_TRQ", "RoundRobin_TRQ_r05": "RoundRobin_TRQ"}
    # Two payloads that disagree on a spec's streams cannot be paired.
    clash = payload("trq.free.b20.q4.d2.s7", {"X_variant": "Y"}, ["X_variant"])
    with pytest.raises(ValueError, match="RNG identity"):
        ms.base_specs([recorded, clash])


def test_run_unit_records_each_specs_rng_identity(monkeypatch):
    from panobbgo.harness_ioh import make_trust_region_strategies

    specs = make_trust_region_strategies(["RoundRobin_TRQ", "RoundRobin_TRQ_aware"])
    monkeypatch.setattr(ms, "_strategies", lambda group: specs)
    payload = ms.run_unit(ms.Unit.parse("trq.free.b20.q4.d2.s42.i0.f0"), jobs=1)
    assert payload["rng_identity"] == {"RoundRobin_TRQ": "RoundRobin_TRQ", "RoundRobin_TRQ_aware": "RoundRobin_TRQ"}
    assert ms.base_specs([payload]) == {"RoundRobin_TRQ_aware": "RoundRobin_TRQ"}


# ---------------------------------------------------------------------------
# the opt-in wide preset (DISCOVERY §68) and the cost estimate
# ---------------------------------------------------------------------------


def test_wide_preset_is_opt_in_and_wired():
    from panobbgo.harness_families import WIDE_FAMILIES

    assert ms.PRESET_FAMILIES["wide"] == len(WIDE_FAMILIES) == 15
    assert ms.PRESET_DIMS["wide"] == (2, 5, 10)
    assert vars(ms.build_parser().parse_args(["plan"]))["presets"] == "free"
    for k, cfg in enumerate(WIDE_FAMILIES):
        sel = ms._instances("wide", 2, fam=k)
        assert [int(p.instance) for _, p in sel] == list(range(ms.N_INSTANCES))
        assert {str(p.family) for _, p in sel} == {cfg.name()}
    assert len(ms._instances("wide", 5)) == 45
    unit = ms.Unit.parse("core.wide.b100.q4.d5.s42.f14")
    assert unit.n_runs == 3 and ms.Unit.parse("core.wide.b100.q4.d5.s42").n_runs == 45
    with pytest.raises(ValueError):
        ms.Unit.parse("core.wide.b100.q4.d5.s42.f15")


def test_cost_sums_the_plan(capsys):
    units = ms.make_units([42, 7], ["wide"], [20, 100], None, [1, 4], ["core", "TuRBO1"])
    entries = ms.plan(units, 4, ms.TARGET_MINUTES)
    rows = ms.plan_cost(entries)
    assert set(rows) == {"core", "TuRBO1"}
    for group, row in rows.items():
        mine = [e for e in entries if e["group"] == group]
        assert row["shards"] == len(mine)
        assert row["runner_min"] == sum(e["est_min"] for e in mine)
        assert row["longest_min"] == max(e["est_min"] for e in mine)
        assert row["runs"] == sum(u.n_runs for u in units if u.group == group)
    # The wide preset is 3x the free preset's runs.
    free = ms.make_units([42, 7], ["free"], [20, 100], None, [1, 4], ["core"])
    assert rows["core"]["runs"] == 3 * sum(u.n_runs for u in free)
    assert ms.main(["cost", "--presets", "wide", "--seeds", "2", "--groups", "core", "--qs", "1,4"]) == 0
    out = capsys.readouterr().out
    assert out.splitlines()[0].startswith("| group |") and "| core |" in out and "refused" not in out
    # The full wide grid with the GP groups is refused by plan (more than 256 shards): cost says so.
    assert ms.main(["cost", "--presets", "wide"]) == 0
    assert "256" in capsys.readouterr().out
