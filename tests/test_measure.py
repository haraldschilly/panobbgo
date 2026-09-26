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


def test_unit_id_round_trip():
    u = ms.Unit("core", "free", 20, 4, 10, 42)
    assert u.id == "core.free.b20.q4.d10.s42"
    assert ms.Unit.parse(u.id) == u
    assert u.n_runs == 15
    assert ms.Unit("SMAC", "failure", 100, 1, 5, 7).n_runs == 12
    for bad in ("core.free.b20.q4.d10", "core.free.20.q4.d10.s42", "nope.free.b20.q4.d10.s42", "core.x.b2.q1.d2.s1"):
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
    # Hours per run: no qLogEI or SMAC at d = 10, 100*d; TuRBO (one proposal per batch) runs there.
    assert ms.covered("qLogEI", 100, 5, 1) and not ms.covered("qLogEI", 100, 10, 1)
    assert not ms.covered("SMAC", 100, 10, 1) and not ms.covered("SMAC", 20, 2, 4)
    assert ms.covered("TuRBO1", 100, 10, 1)
    assert not any(u.group == "qLogEI" and u.bm * u.dim > 500 for u in units)
    # The estimate grows with the budget and shrinks with q for TuRBO only.
    assert ms.laptop_seconds("qLogEI", 10, 1000) > ms.MAX_RUN_SECONDS > ms.laptop_seconds("qLogEI", 5, 500)
    assert ms.laptop_seconds("TuRBO1", 5, 500, q=16) < ms.laptop_seconds("TuRBO1", 5, 500, q=1)
    assert ms.laptop_seconds("qLogEI", 5, 500, q=16) == ms.laptop_seconds("qLogEI", 5, 500, q=1)
    # dims restricts.
    assert {u.dim for u in ms.make_units([42], ["free"], [20], [2], [1], ["core"])} == {2}
    with pytest.raises(ValueError):
        ms.make_units([42], ["nope"], [20], None, [1], ["core"])
    with pytest.raises(ValueError):
        ms.make_units([42], ["free"], [20], None, [1], ["nope"])


def test_plan_packs_every_unit_once_by_group():
    units = ms.make_units([42, 7, 1234], ["free"], [20, 100], None, [1, 4, 16, 64], list(ms.GROUPS))
    entries = ms.plan(units, jobs=4, target_minutes=180)
    planned = [u for e in entries for u in e["units"].split(";")]
    assert sorted(planned) == sorted(u.id for u in units)
    for e in entries:
        groups = {ms.Unit.parse(u).group for u in e["units"].split(";")}
        assert groups == {e["group"]}
        assert e["shard"].startswith(e["group"] + "-")
        assert e["n_units"] == len(e["units"].split(";"))
        # A shard exceeds the target only when a single unit does.
        assert e["est_min"] <= 180 + 1 or e["n_units"] == 1
    assert entries[0]["group"] == "core"
    assert len({e["shard"] for e in entries}) == len(entries)


def test_plan_cli_is_stdlib_only():
    code = (
        "import runpy, sys; sys.argv = ['measure.py', 'plan', '--seeds', '1', '--dims', '2', '--qs', '1,4'];"
        f"\ntry:\n    runpy.run_path({str(_PATH)!r}, run_name='__main__')\nexcept SystemExit as e:\n    assert not e.code"
        "\nassert 'numpy' not in sys.modules and 'panobbgo' not in sys.modules, 'plan imported heavy modules'"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    matrix = json.loads(out.strip().splitlines()[0])["include"]
    ids = [u for e in matrix for u in e["units"].split(";")]
    assert "core.free.b20.q4.d2.s42" in ids and "SMAC.free.b20.q1.d2.s42" in ids
    assert not any(".q4." in u for u in ids if u.startswith("SMAC"))


# ---------------------------------------------------------------------------
# aggregation on synthetic results
# ---------------------------------------------------------------------------


def _payload(unit: str, shard: str, scores: Dict[str, List[float]], cpu: str = "cpu A") -> dict:
    """A unit result file: ``scores[strategy]`` = the AOCC per instance (aocc_time = AOCC / 2)."""
    u = ms.Unit.parse(unit)
    runs = [
        IOHRunRecord(
            problem_kind="ellipsoid",
            dim=u.dim,
            instance=i,
            strategy_name=name,
            rep=0,
            budget=u.bm * u.dim,
            n_evals=u.bm * u.dim,
            best_fx=1.0,
            f_opt=0.0,
            aocc=a,
            elapsed_s=1.0,
            seed=0,
            aocc_time=a / 2,
        )
        for name, vals in scores.items()
        for i, a in enumerate(vals)
    ]
    res = IOHHarnessResult("measure-free-b20", "families", -8.0, 2.0, runs, sync_eval=True, virtual={"workers": u.q})
    return {
        **{k: v for k, v in asdict(u).items()},
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


def test_aggregate_pairs_against_the_best_external(tmp_path):
    payloads = []
    for seed, bump in ((42, 0.0), (7, 0.02), (1234, 0.04)):
        payloads.append(
            _payload(
                f"core.free.b20.q4.d2.s{seed}",
                "core-01",
                {
                    "RoundRobin_CMAES": [0.5 + bump, 0.6 + bump],
                    "Baseline_NGOpt": [0.4, 0.5],
                    "Baseline_pycma_IPOP": [0.3, 0.3],
                },
            )
        )
        payloads.append(
            _payload(f"TuRBO1.free.b20.q4.d2.s{seed}", "TuRBO1-01", {"Baseline_TuRBO1": [0.7, 0.7]}, cpu="cpu B")
        )
    _write(tmp_path, payloads, meta={"shard": "core-01", "failed": [], "github_run_id": "123"})
    planned = [p["unit"] for p in payloads] + ["core.free.b20.q4.d2.s99"]
    summary = ms.aggregate(tmp_path, planned)
    assert summary["missing_units"] == ["core.free.b20.q4.d2.s99"]
    assert summary["github_run_id"] == ["123"]
    assert set(summary["fp_classes"]) == {"cpu A", "cpu B"}
    cell = summary["cells"]["free/d2/b20/q4"]
    assert cell["best_external"] == {"aocc": "Baseline_TuRBO1", "aocc_time": "Baseline_TuRBO1"}
    assert cell["best_core_external"]["aocc"] == "Baseline_NGOpt"
    row = cell["strategies"]["RoundRobin_CMAES"]
    assert row["aocc"] == pytest.approx(0.57)
    assert row["aocc_time"] == pytest.approx(0.285)
    vs = row["vs_best_aocc"]
    # Per seed: mean(0.5+b, 0.6+b) - 0.7 = -0.15 + b.
    assert vs["delta"] == pytest.approx(-0.13)
    assert vs["n_seeds"] == 3 and vs["n_pairs"] == 6 and vs["wins"] == 0
    assert vs["cross_job"] and vs["cross_fp"]
    assert vs["ci_low"] < vs["delta"] < vs["ci_high"]
    core = row["vs_core_aocc"]
    assert core["delta"] == pytest.approx(0.12)
    assert core["wins"] == 3 and not core["cross_job"] and not core["cross_fp"]
    assert "vs_best_aocc" not in cell["strategies"]["Baseline_NGOpt"]
    md = ms.summary_markdown(summary)
    assert "free/d2/b20/q4" in md and "≈" in md and "missing: core.free.b20.q4.d2.s99" in md


def test_aggregate_refuses_a_duplicate_unit(tmp_path):
    p = _payload("core.free.b20.q1.d2.s42", "core-01", {"RoundRobin_CMAES": [0.5]})
    q = dict(p, shard="core-02")
    _write(tmp_path, [p, q])
    with pytest.raises(ValueError, match="twice"):
        ms.aggregate(tmp_path)


# ---------------------------------------------------------------------------
# a tiny real run
# ---------------------------------------------------------------------------


def test_run_and_aggregate_end_to_end(tmp_path, monkeypatch):
    """The panobbgo specs only (no optional extra), one instance at d = 2, 20*d evaluations, q = 4."""
    monkeypatch.setitem(ms.GROUPS, "core", ())
    real_instances = ms._instances
    monkeypatch.setattr(ms, "_instances", lambda preset, dim: real_instances(preset, dim)[:1])
    out = tmp_path / "raw" / "core-01"
    rc = ms.main(
        ["run", "--units", "core.free.b20.q4.d2.s42", "--shard", "core-01", "--out-dir", str(out), "--jobs", "1"]
    )
    assert rc == 0
    meta = json.loads((out / "meta_core-01.json").read_text())
    assert meta["done"] == ["core.free.b20.q4.d2.s42"] and meta["status"] == 0
    assert meta["host"]["cpu_count"]
    payload = json.loads((out / "core.free.b20.q4.d2.s42.json").read_text())
    res = IOHHarnessResult.from_dict(payload["result"])
    assert res.virtual is not None and res.virtual["workers"] == 4 and res.virtual["model"] == "lognormal"
    assert {r.strategy_name for r in res.runs} == {s.name for s in make_ioh_strategies()}
    assert all(r.budget == 40 and r.aocc_time is not None for r in res.runs)
    summary = ms.aggregate(tmp_path / "raw")
    cell = summary["cells"]["free/d2/b20/q4"]
    assert cell["best_external"]["aocc"] is None  # no baseline ran
    assert set(cell["strategies"]) == {s.name for s in make_ioh_strategies()}


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
    steps = workflow["jobs"]["measure"]["steps"]
    [install] = [s for s in steps if s.get("name") == "Install dependencies"]
    for extra in ("dev", "baselines", "baselines-bo"):
        assert f"--extra {extra}" in install["run"]
    [measure] = [s for s in steps if s.get("name") == "Measure"]
    # The step's own limit sits below the job's, so the upload still runs.
    assert measure["timeout-minutes"] < workflow["jobs"]["measure"]["timeout-minutes"] <= 350
    [upload] = [s for s in steps if s.get("name") == "Upload shard results"]
    assert upload["if"] == "always()"
    # Artifacts only: nothing in this workflow may write to the repository.
    assert workflow["permissions"] == {"contents": "read"}
    assert all("permissions" not in job for job in workflow["jobs"].values())
