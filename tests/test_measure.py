# Copyright 2012-2026 Harald Schilly <harald.schilly@gmail.com>
"""``scripts/measure.py``: units, coverage, the shard plan, a tiny run and the aggregation."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

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
    for bad in (
        "core.free.b20.q4.d10",
        "core.free.20.q4.d10.s42",
        "nope.free.b20.q4.d10.s42",
        "core.x.b2.q1.d2.s1",
        "core.free.b20.q4.d10.s42.x0",
        "core.free.b20.q4.d10.s42.i3",
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
    assert ms.covered("TuRBO1", 100, 10, 1)
    assert not any(u.group == "qLogEI" and u.bm * u.dim > 500 for u in units)
    # The estimate grows with the budget; with q it shrinks for TuRBO and grows for qLogEI.
    assert ms.laptop_seconds("qLogEI", 10, 1000) > ms.MAX_RUN_SECONDS > ms.laptop_seconds("qLogEI", 5, 500)
    assert ms.laptop_seconds("TuRBO1", 5, 500, q=16) < ms.laptop_seconds("TuRBO1", 5, 500, q=1)
    assert ms.laptop_seconds("qLogEI", 5, 500, q=64) > ms.laptop_seconds("qLogEI", 5, 500, q=16)
    assert ms.laptop_seconds("qLogEI", 5, 500, q=16) > ms.laptop_seconds("qLogEI", 5, 500, q=1)
    assert ms.laptop_seconds("SMAC", 2, 40, q=4) == ms.laptop_seconds("SMAC", 2, 40, q=1)
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
    entries = ms.plan(units, jobs=4, target_minutes=180, extra=extra)
    planned = [ms.Unit.parse(u) for e in entries for u in e["units"].split(";")]
    # Every (unit, instance) of the grid exactly once, split or not; the extra unit once more.
    covered = sorted((replace_inst(u, j) for u in planned for j in insts(u)), key=lambda u: u.id)
    expected = sorted((replace_inst(u, j) for u in units for j in range(ms.N_INSTANCES)), key=lambda u: u.id)
    assert covered == sorted(expected + [extra[0]], key=lambda u: u.id)
    assert any(u.inst >= 0 for u in planned), "the long qLogEI units at d = 5, 100*d, q = 64 are split"
    for e in entries:
        groups = {ms.Unit.parse(u).group for u in e["units"].split(";")}
        assert groups == {e["group"]}
        assert e["n_units"] == len(e["units"].split(";"))
        # A shard exceeds the target only when a single unit does.
        assert e["est_min"] <= 180 + 1 or e["n_units"] == 1
        assert e["est_min"] <= ms.STEP_LIMIT_MINUTES
    assert entries[0]["group"] == "core"
    assert [e["shard"] for e in entries if e["shard"].startswith("extra-")] == ["extra-01"]
    assert len({e["shard"] for e in entries}) == len(entries)


def replace_inst(u, j):
    """``u`` for instance ``j`` alone."""
    return ms.Unit(u.group, u.preset, u.bm, u.q, u.dim, u.seed, j)


def insts(u):
    return range(ms.N_INSTANCES) if u.inst < 0 else [u.inst]


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
    assert "qLogEI.free.b100.q64.d5.s42.i0" in ids


# ---------------------------------------------------------------------------
# aggregation on synthetic results
# ---------------------------------------------------------------------------


def _payload(unit: str, shard: str, scores: Dict[str, List[Optional[float]]], cpu: str = "cpu A") -> dict:
    """A unit result file: ``scores[strategy]`` = the AOCC per instance (aocc_time = AOCC / 2; None = crashed).

    Instances 0, 1 are the ellipsoid's 0, 1; instances 2, 3 rastrigin's 0, 1.
    """
    u = ms.Unit.parse(unit)
    runs = [
        IOHRunRecord(
            problem_kind="ellipsoid" if i < 2 else "rastrigin",
            dim=u.dim,
            instance=i % 2,
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
            error="RuntimeError: boom" if a is None else None,
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
    # A missing core seed makes every core external incomplete at q = 4: out of the pool at every q.
    assert missing["cells"]["free/d2/b20/q1"]["pool"] == ["Baseline_TuRBO1"]
    assert "incomplete at q=4" in missing["cells"]["free/d2/b20/q1"]["pool_excluded"]["Baseline_NGOpt"]
    assert "missing: core.free.b20.q4.d2.s99" in ms.summary_markdown(missing)
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


def test_pool_is_fixed_when_a_baseline_misses_seeds(tmp_path):
    # TuRBO misses seed 1234 everywhere: incomplete -> out of the pool in every q cell.
    planned = _grid(tmp_path, gp_seeds=(42, 7))
    summary = ms.aggregate(tmp_path, planned)
    for c in ("free/d2/b20/q1", "free/d2/b20/q4"):
        cell = summary["cells"][c]
        assert cell["pool"] == ["Baseline_NGOpt", "Baseline_pycma_IPOP"]
        turbo = cell["strategies"]["Baseline_TuRBO1"]
        assert (turbo["n_seeds"], turbo["planned_seeds"], turbo["complete"]) == (2, 3, False)
    assert "2/3!" in ms.summary_markdown(summary)


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
    monkeypatch.setattr(ms, "_instances", lambda preset, dim, inst=-1: real_instances(preset, dim, inst)[:1])
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
    assert measure["timeout-minutes"] < workflow["jobs"]["measure"]["timeout-minutes"] <= 350
    [upload] = [s for s in steps if s.get("name") == "Upload shard results"]
    assert upload["if"] == "always()" and upload["with"]["overwrite"] is True
    [summary_upload] = [s for s in workflow["jobs"]["aggregate"]["steps"] if s.get("name") == "Upload the summary"]
    assert summary_upload["with"]["overwrite"] is True
    # Artifacts only: nothing in this workflow may write to the repository.
    assert workflow["permissions"] == {"contents": "read"}
    assert all("permissions" not in job for job in workflow["jobs"].values())
