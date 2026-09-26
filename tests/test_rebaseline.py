# Copyright 2012-2026 Harald Schilly <harald.schilly@gmail.com>
"""``scripts/rebaseline.py``: the matrix, the shard commands, the aggregation, the release helpers."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Optional

import pytest

from panobbgo.harness import HarnessConfig, HarnessResult
from panobbgo.harness_ioh import DEFAULT_DECISION_SEEDS, IOHHarnessResult, IOHMultiSeedResult, IOHRunRecord

_PATH = Path(__file__).resolve().parent.parent / "scripts" / "rebaseline.py"
_spec = importlib.util.spec_from_file_location("rebaseline_cli", _PATH)
assert _spec is not None and _spec.loader is not None
rb = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = rb  # dataclasses look their module up there
_spec.loader.exec_module(rb)


def test_roster_is_the_decision_roster():
    assert rb.ROSTER == tuple(DEFAULT_DECISION_SEEDS)


def test_resolve_seeds():
    assert rb.resolve_seeds("3") == [42, 7, 1234]
    assert rb.resolve_seeds("42, 99") == [42, 99]
    assert rb.resolve_seeds("42,") == [42]
    # A shard's matrix entry is always a seed list, even with one seed.
    assert rb.seed_list("42") == [42]
    assert rb.seed_list("2025,3,11") == [2025, 3, 11]
    with pytest.raises(ValueError):
        rb.resolve_seeds("13")
    with pytest.raises(ValueError):
        rb.resolve_seeds("42,42")


def test_full_plan_covers_every_seed_once_per_suite():
    entries = rb.plan(rb.resolve_suites("all"), rb.resolve_seeds("12"))
    assert 20 <= len(entries) <= 40
    # 'all' is every reference suite and no opt-in A/B suite.
    assert {e["suite"] for e in entries} == {n for n, s in rb.SUITES.items() if not s.opt_in}
    for suite in (n for n, s in rb.SUITES.items() if not s.opt_in):
        seeds = [int(s) for e in entries if e["suite"] == suite for s in e["seeds"].split(",")]
        assert seeds == list(rb.ROSTER)
    assert len({(e["suite"], e["shard"]) for e in entries}) == len(entries)
    assert rb.plan(rb.resolve_suites("ioh-quick"), [42]) == [
        {"suite": "ioh-quick", "shard": "01", "seeds": "42", "extras": ""}
    ]
    assert {e["extras"] for e in entries if e["suite"] == "ioh-external"} == {"baselines"}


def test_shard_commands_are_sync_and_have_no_wall_clock_limit(tmp_path):
    for suite in rb.SUITES.values():
        for argv, out in rb.shard_commands(suite, [42, 7], "01", tmp_path, jobs=4, python="py"):
            assert out.parent == tmp_path
            assert "--timeout" not in argv and not any(a.startswith("timeout=") for a in argv)
            if suite.kind == "composite":
                assert "--no-timeout" in argv and "--sync-eval" in argv
            if suite.kind == "ioh":
                assert "--sync-eval" in argv and "--baselines" in argv


def test_harness_cli_no_timeout():
    from benchmark_harness import build_parser

    assert build_parser().parse_args(["run", "--quick"]).timeout == 120.0
    assert build_parser().parse_args(["run", "--quick", "--no-timeout"]).timeout is None


#: The pycma / Nevergrad / Optuna baselines: the ``baselines`` extra's share of
#: ``EXTERNAL_BASELINE_NAMES`` (other extras may append names there).
CHEAP_EXTERNAL = (
    "Baseline_pycma_IPOP",
    "Baseline_pycma_BIPOP",
    "Baseline_NGOpt",
    "Baseline_NG_CMA",
    "Baseline_NG_TwoPointsDE",
    "Baseline_Optuna_CmaEs",
    "Baseline_Optuna_TPE",
)


def test_external_suite_names_the_baselines_of_its_extra(tmp_path):
    from panobbgo.harness_baselines import DEFAULT_BASELINE_NAMES
    from panobbgo.harness_ioh import make_ioh_strategies

    suite = rb.SUITES["ioh-external"]
    assert suite.kind == "ioh" and suite.battery == "standard" and suite.extras == ("baselines",)
    [(argv, _)] = rb.shard_commands(suite, [42], "01", tmp_path, jobs=2, python="py")
    names = argv[argv.index("--strategies") + 1 : argv.index("--seeds")]
    expected = [s.name for s in make_ioh_strategies()] + list(DEFAULT_BASELINE_NAMES) + list(CHEAP_EXTERNAL)
    assert names == expected
    # The plain suites keep the harness' default strategy set.
    [(plain, _)] = rb.shard_commands(rb.SUITES["ioh-standard"], [42], "01", tmp_path, jobs=2, python="py")
    assert "--strategies" not in plain


def test_a_baseline_of_another_extra_stays_out_of_the_external_suite(monkeypatch):
    # An expensive-track baseline appended to the registry (its class names
    # extra "baselines-bo") must not join a suite that installs only "baselines".
    import panobbgo.harness_baselines as hb

    class FakeBO:
        extra = "baselines-bo"

    real = hb.make_external_baseline_strategies

    def with_bo():
        return real() + [hb.StrategySpec(name="Baseline_FakeBO", strategy_class=FakeBO, heuristics=[])]

    monkeypatch.setattr(hb, "make_external_baseline_strategies", with_bo)
    names = rb.STRATEGY_SETS["external"](rb.SUITES["ioh-external"])
    assert "Baseline_FakeBO" not in names and set(CHEAP_EXTERNAL) <= set(names)
    assert rb.external_baselines_for(("baselines", "baselines-bo"))[-1] == "Baseline_FakeBO"


def test_every_strategy_set_is_satisfiable_by_its_suites_extras():
    import panobbgo.harness_baselines as hb

    classes = {s.name: s.strategy_class for s in hb.make_external_baseline_strategies()}
    for suite in rb.SUITES.values():
        if not suite.variant:
            continue
        for name in rb.STRATEGY_SETS[suite.variant](suite):
            if name in classes:
                assert rb.baseline_extra(classes[name]) in suite.extras, (suite.name, name)


def test_ioh_cli_accepts_the_external_strategy_set():
    # The names survive ioh_benchmark.py's filter: no "unknown strategy" is dropped.
    spec = importlib.util.spec_from_file_location("ioh_benchmark_cli", rb.REPO_ROOT / "scripts" / "ioh_benchmark.py")
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    for module in ("cma", "nevergrad", "optuna", "cmaes"):
        pytest.importorskip(module)
    names = rb.STRATEGY_SETS["external"](rb.SUITES["ioh-external"])
    args = argparse.Namespace(legacy=False, standard=True, full=False, baselines=True, strategies=names)
    assert [s.name for s in cli._resolve_strategies(args)] == names


def test_the_cma_ab_suites(tmp_path):
    """The §57 A/B suites: opt-in, one strategy set, the BBOB battery arguments on the command line."""
    from panobbgo.harness_ioh import CMAES_VARIANT_NAMES

    names = ["ioh-cma-ab", "ioh-cma-ab-bbob-b200", "ioh-cma-ab-bbob-b500"]
    assert all(rb.SUITES[n].opt_in for n in names)
    assert not any(n in {s.name for s in rb.resolve_suites("all")} for n in names)
    assert [s.name for s in rb.resolve_suites(",".join(names))] == names
    for n in names:
        suite = rb.SUITES[n]
        assert rb.STRATEGY_SETS[suite.variant](suite) == [
            "RoundRobin_CMAES",
            *CMAES_VARIANT_NAMES,
            "Baseline_Optuna_CmaEs",
            "Baseline_pycma_BIPOP",
        ]
        [(argv, _)] = rb.shard_commands(suite, [42, 7], "01", tmp_path, jobs=2, python="py")
        assert "--sync-eval" in argv and "--baselines" in argv and "--timeout" not in argv
    [(argv, _)] = rb.shard_commands(rb.SUITES["ioh-cma-ab-bbob-b500"], [42], "01", tmp_path, jobs=2, python="py")
    cmd = " ".join(argv)
    assert "--bbob --baselines" in cmd and "--bbob-dims 5 10 --bbob-instances 0 1 --budget-multiplier 500" in cmd
    [(argv, _)] = rb.shard_commands(rb.SUITES["ioh-cma-ab"], [42], "01", tmp_path, jobs=2, python="py")
    assert "--standard" in argv and "--bbob" not in argv
    assert rb.SUITES["ioh-cma-ab-bbob-b200"].ref_key == "bbob_cma_ab_b200"


def test_ioh_cli_accepts_the_cma_ab_strategy_set():
    spec = importlib.util.spec_from_file_location("ioh_benchmark_cli2", rb.REPO_ROOT / "scripts" / "ioh_benchmark.py")
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    for module in ("cma", "optuna", "cmaes"):
        pytest.importorskip(module)
    names = rb.STRATEGY_SETS["cma_ab"](rb.SUITES["ioh-cma-ab"])
    args = argparse.Namespace(legacy=False, standard=True, full=False, baselines=True, strategies=names)
    specs = {s.name: s for s in cli._resolve_strategies(args)}
    assert set(specs) == set(names)
    # Variants of one arm share the flagship's RNG identity (DISCOVERY §18).
    assert {specs[n].rng_identity for n in names if n.startswith("RoundRobin_CMAES")} == {"RoundRobin_CMAES"}


def _ioh_shard(
    path: Path,
    seeds,
    battery: str = "ioh-quick",
    strategy: str = "A",
    aocc: Optional[float] = None,
    blas_threads: Optional[int] = None,
    fp_env_id: Optional[str] = None,
):
    results = []
    for seed in seeds:
        run = IOHRunRecord(
            problem_kind="MA-BBOB",
            dim=2,
            instance=0,
            strategy_name=strategy,
            rep=0,
            budget=10,
            n_evals=10,
            best_fx=1.0,
            f_opt=0.0,
            aocc=seed / 10000 if aocc is None else aocc,
            elapsed_s=0.1,
            seed=seed,
        )
        results.append(IOHHarnessResult(battery, "MA-BBOB", -8, 2, [run], sync_eval=True))
    multi = IOHMultiSeedResult(
        battery,
        "MA-BBOB",
        -8,
        2,
        list(seeds),
        results,
        sync_eval=True,
        blas_threads=blas_threads,
        fp_env={"id": fp_env_id} if fp_env_id else None,
        fp_env_id=fp_env_id,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(multi.to_json())


def test_aggregate(tmp_path):
    src = tmp_path / "raw"
    # Artifacts land in one directory per artifact name (gh run download).
    for seed in (7, 42):
        d = src / f"rebaseline-composite-quick-{seed}" / "composite-quick"
        d.mkdir(parents=True)
        HarnessResult(HarnessConfig(seed=seed), "", 0, 0.0, [], 0.5).save(str(d / f"composite-quick_s{seed}.json"))
        (d / f"meta_{seed}.json").write_text(json.dumps({"suite": "composite-quick", "shard": str(seed), "status": 0}))
    _ioh_shard(src / "a" / "ioh-quick" / "ioh-quick_shard02.json", [7])
    _ioh_shard(src / "b" / "ioh-quick" / "ioh-quick_shard01.json", [42])
    fam = src / "c" / "families-free"
    fam.mkdir(parents=True)
    (fam / "families-free_shard02.json").write_text(json.dumps([{"seed": 7, "s": "x", "aocc": 0.1}]))
    (fam / "families-free_shard01.json").write_text(json.dumps([{"seed": 42, "s": "x", "aocc": 0.2}]))

    out = tmp_path / "ref"
    manifest = rb.aggregate(src, out)

    assert manifest["suites"]["composite-quick"]["seeds"] == [42, 7]
    assert HarnessResult.load(str(out / "ref_composite_quick_s7.json")).config.seed == 7
    combined = IOHMultiSeedResult.from_dict(json.loads((out / "ref_ioh_quick.json").read_text()))
    assert combined.base_seeds == [42, 7] and combined.sync_eval
    assert combined.per_strategy_seed_aocc()["A"] == [0.0042, 0.0007]
    single = IOHHarnessResult.from_dict(json.loads((out / "ref_ioh_quick_s7.json").read_text()))
    assert single.runs[0].seed == 7
    rows = json.loads((out / "ref_family_screen_free.json").read_text())
    assert [r["seed"] for r in rows] == [42, 7]
    assert json.loads((out / "ref_MANIFEST.json").read_text())["failed_shards"] == []


def _fake_shard(suite, seeds, shard: str, root: Path) -> None:
    """The result file(s) and meta a shard of ``suite`` leaves, as ``gh run download`` lays them out."""
    d = root / f"shard-{suite.name}-{shard}" / suite.name
    d.mkdir(parents=True)
    # Every suite has its own number and spec names, so a summary that reads
    # another suite's file (e.g. one keyed by battery, not by ref_key) shows.
    value = _suite_value(suite.name)
    if suite.kind == "composite":
        for seed in seeds:
            HarnessResult(HarnessConfig(seed=seed), "", 0, 0.0, [], value).save(str(d / f"{suite.name}_s{seed}.json"))
    elif suite.kind == "ioh":
        path = d / f"{suite.name}_shard{shard}.json"
        _ioh_shard(path, seeds, battery=f"ioh-{suite.battery}", strategy=f"{suite.name}/A", aocc=value)
    else:
        specs = (f"{suite.name}/A", f"{suite.name}/B")
        rows = [{"seed": s, "s": spec, "aocc": value, "err": None} for s in seeds for spec in specs]
        (d / f"{suite.name}_shard{shard}.json").write_text(json.dumps(rows))
    meta = {"suite": suite.name, "shard": shard, "seeds": list(seeds), "status": 0, "github_run_id": "5"}
    (d / f"meta_{shard}.json").write_text(json.dumps(meta))


def _suite_value(name: str) -> float:
    return (list(rb.SUITES).index(name) + 1) / 100


def test_every_suite_goes_through_plan_aggregate_and_summary(tmp_path):
    # The registry check: a suite added to SUITES must come out of every code path.
    seeds = rb.resolve_seeds("12")
    # Every suite, the opt-in A/B suites included (named, as a dispatch would).
    planned = rb.plan(rb.resolve_suites(",".join(rb.SUITES)), seeds)
    assert {e["suite"] for e in planned} == set(rb.SUITES)
    for e in planned:
        _fake_shard(rb.SUITES[e["suite"]], rb.seed_list(e["seeds"]), e["shard"], tmp_path / "raw")
    out = tmp_path / "ref"
    manifest = rb.aggregate(tmp_path / "raw", out, planned)
    assert manifest["failed_shards"] == [] and list(manifest["suites"]) == list(rb.SUITES)

    files = [f for info in manifest["suites"].values() for f in info["files"]]
    assert len(files) == len(set(files)), "two suites write the same reference file"
    summary = json.loads((out / "SUMMARY.json").read_text())
    assert list(summary["suites"]) == list(rb.SUITES)
    for name, suite in rb.SUITES.items():
        entry = summary["suites"][name]
        assert entry["seeds"] == list(rb.ROSTER), name
        assert all((out / f).exists() for f in entry["files"]), name
        value = _suite_value(name)
        if suite.kind == "composite":
            comp = entry["composite_score"]
            assert comp["mean"] == pytest.approx(value) and comp["per_seed"] == {str(s): value for s in rb.ROSTER}
        elif suite.kind == "ioh":
            assert entry["files"][0] == f"ref_ioh_{suite.ref_key}.json" and len(entry["files"]) == 13
            assert entry["mean_aocc"] == pytest.approx(value), name
            assert entry["per_spec_aocc"] == {f"{name}/A": pytest.approx(value)}, name
        else:
            assert entry["files"] == [f"ref_family_screen_{suite.ref_key}.json"]
            assert entry["rows"] == 24 and entry["failed_rows"] == 0
            assert entry["mean_aocc"] == pytest.approx(value), name
            assert entry["per_spec_aocc"] == {f"{name}/A": pytest.approx(value), f"{name}/B": pytest.approx(value)}
    md = rb.summary_markdown(summary)
    assert all(f"| {name} | 12 |" in md for name in rb.SUITES)


def _family_screen_presets() -> set:
    """The keys of ``PRESETS`` in ``benchmarks/family_screen.py`` (a local of its ``main``)."""
    import ast

    tree = ast.parse((rb.REPO_ROOT / "benchmarks" / "family_screen.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "PRESETS" for t in node.targets):
            assert isinstance(node.value, ast.Dict)
            return {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
    raise AssertionError("no PRESETS in family_screen.py")


def test_suite_registry_is_consistent():
    import tomllib

    extras = tomllib.loads((rb.REPO_ROOT / "pyproject.toml").read_text())["project"]["optional-dependencies"]
    presets = _family_screen_presets()
    assert {"free", "constrained", "shapes", "failure"} <= presets
    for suite in rb.SUITES.values():
        assert suite.kind in ("composite", "ioh", "families"), suite.name
        assert set(suite.extras) <= set(extras), suite.name
        if suite.kind == "families":
            assert suite.battery in presets, suite.name
        if suite.kind == "ioh":
            assert suite.name.startswith("ioh-"), "the workflow syncs the IOH worker venv for ioh-* suites"
        if suite.variant:
            # A variant picks an IOH strategy set; "s<digits>" would read as a seed in ref_*_s<seed>.json.
            assert suite.kind == "ioh" and suite.variant in rb.STRATEGY_SETS, suite.name
            assert not re.fullmatch(r"s\d+", suite.variant), suite.name
    for variant in rb.STRATEGY_SETS:
        assert any(s.variant == variant for s in rb.SUITES.values()), f"strategy set {variant} is used by no suite"


def test_the_workflow_names_every_suite_and_installs_its_extras():
    import yaml

    text = (rb.REPO_ROOT / ".github" / "workflows" / "rebaseline.yml").read_text()
    workflow = yaml.safe_load(text)
    suites_input = workflow[True]["workflow_dispatch"]["inputs"]["suites"]["description"]  # YAML 1.1: on -> True
    for name in rb.SUITES:
        assert name in suites_input, f"{name} missing from the workflow's suites input"
    [install] = [s for s in workflow["jobs"]["measure"]["steps"] if s.get("name") == "Install dependencies"]
    # Always run: a cache hit does not say which extras the cached .venv has.
    assert "if" not in install
    assert install["env"]["EXTRAS"] == "${{ matrix.extras }}" and "$EXTRAS" in install["run"]
    assert "--extra dev" in install["run"] and "uv sync" in install["run"]


def test_aggregate_keeps_blas_threads_and_refuses_a_mix(tmp_path):
    _ioh_shard(tmp_path / "a" / "ioh-quick" / "ioh-quick_shard01.json", [42], blas_threads=1)
    _ioh_shard(tmp_path / "b" / "ioh-quick" / "ioh-quick_shard02.json", [7], blas_threads=1)
    rb.aggregate(tmp_path, tmp_path / "ref")
    merged = IOHMultiSeedResult.from_dict(json.loads((tmp_path / "ref" / "ref_ioh_quick.json").read_text()))
    assert merged.blas_threads == 1
    _ioh_shard(tmp_path / "b" / "ioh-quick" / "ioh-quick_shard02.json", [7], blas_threads=None)
    with pytest.raises(ValueError, match="BLAS thread"):
        rb.aggregate(tmp_path, tmp_path / "ref2")


def test_aggregate_refuses_mixed_fp_environments_unless_allowed(tmp_path):
    _ioh_shard(tmp_path / "a" / "ioh-quick" / "ioh-quick_shard01.json", [42], fp_env_id="aaaa")
    _ioh_shard(tmp_path / "b" / "ioh-quick" / "ioh-quick_shard02.json", [7], fp_env_id="aaaa")
    fam = tmp_path / "c" / "families-free"
    fam.mkdir(parents=True)
    (fam / "families-free_shard01.json").write_text(
        json.dumps([{"seed": 42, "s": "x", "aocc": 0.2, "fp_env_id": "aaaa"}])
    )
    manifest = rb.aggregate(tmp_path, tmp_path / "ref")
    assert manifest["fp_env_ids"] == ["aaaa"] and not manifest["mixed_fp"]
    assert manifest["suites"]["ioh-quick"]["fp_env_ids"] == ["aaaa"]
    assert manifest["suites"]["families-free"]["fp_env_ids"] == ["aaaa"]
    assert manifest["fp_env"] == {"aaaa": {"id": "aaaa"}}
    merged = IOHMultiSeedResult.from_dict(json.loads((tmp_path / "ref" / "ref_ioh_quick.json").read_text()))
    assert merged.fp_env_id == "aaaa"
    assert json.loads((tmp_path / "ref" / rb.SUMMARY).read_text())["fp_env_ids"] == ["aaaa"]

    # One shard from another FP environment: refused ...
    _ioh_shard(tmp_path / "b" / "ioh-quick" / "ioh-quick_shard02.json", [7], fp_env_id="bbbb")
    with pytest.raises(ValueError, match="different FP environments"):
        rb.aggregate(tmp_path, tmp_path / "ref2")
    # ... a shard without the record too (an older file) ...
    _ioh_shard(tmp_path / "b" / "ioh-quick" / "ioh-quick_shard02.json", [7], fp_env_id=None)
    with pytest.raises(ValueError, match="different FP environments"):
        rb.aggregate(tmp_path, tmp_path / "ref2")
    # ... and merged under --allow-mixed-fp, with an id no compare mistakes for a real one.
    _ioh_shard(tmp_path / "b" / "ioh-quick" / "ioh-quick_shard02.json", [7], fp_env_id="bbbb")
    rc = rb.main(["aggregate", str(tmp_path), "--out-dir", str(tmp_path / "ref3"), "--allow-mixed-fp"])
    assert rc == 0
    manifest = json.loads((tmp_path / "ref3" / rb.MANIFEST).read_text())
    assert manifest["mixed_fp"] and manifest["suites"]["ioh-quick"]["fp_env_ids"] == ["aaaa", "bbbb"]
    assert manifest["fp_env_ids"] == ["aaaa", "bbbb"]
    merged = IOHMultiSeedResult.from_dict(json.loads((tmp_path / "ref3" / "ref_ioh_quick.json").read_text()))
    assert merged.fp_env_id == "mixed:aaaa,bbbb" and merged.fp_env is None


def test_aggregate_refuses_mixed_fp_in_composite_and_families(tmp_path):
    for seed, fp in ((7, "aaaa"), (42, "bbbb")):
        d = tmp_path / f"c{seed}" / "composite-quick"
        d.mkdir(parents=True)
        HarnessResult(HarnessConfig(seed=seed, sync_eval=True), "", 0, 0.0, [], 0.5, fp_env_id=fp).save(
            str(d / f"composite-quick_s{seed}.json")
        )
    with pytest.raises(ValueError, match="composite-quick: shards measured in different FP"):
        rb.aggregate(tmp_path, tmp_path / "ref")
    shutil.rmtree(tmp_path / "c7")
    shutil.rmtree(tmp_path / "c42")
    fam = tmp_path / "f" / "families-free"
    fam.mkdir(parents=True)
    (fam / "families-free_shard01.json").write_text(json.dumps([{"seed": 42, "s": "x", "aocc": 0.2, "fp_env_id": "a"}]))
    (fam / "families-free_shard02.json").write_text(json.dumps([{"seed": 7, "s": "x", "aocc": 0.2, "fp_env_id": "b"}]))
    with pytest.raises(ValueError, match="families-free: shards measured in different FP"):
        rb.aggregate(tmp_path, tmp_path / "ref")


def test_the_workflows_pin_the_fp_environment():
    import yaml

    from panobbgo.fp_env import PIN_ENV

    wf = rb.REPO_ROOT / ".github" / "workflows"
    measure = yaml.safe_load((wf / "rebaseline.yml").read_text())["jobs"]["measure"]
    assert measure["env"] == PIN_ENV
    test_job = yaml.safe_load((wf / "tests.yml").read_text())["jobs"]["test"]
    assert {k: test_job["env"][k] for k in PIN_ENV} == PIN_ENV
    check = yaml.safe_load((wf / "fp-check.yml").read_text())
    assert check["jobs"]["screen"]["strategy"]["matrix"]["job"] == list(range(1, 9))


def test_aggregate_rejects_a_seed_measured_twice(tmp_path):
    _ioh_shard(tmp_path / "a" / "ioh-quick" / "ioh-quick_shard01.json", [42])
    _ioh_shard(tmp_path / "b" / "ioh-quick" / "ioh-quick_shard02.json", [42])
    with pytest.raises(ValueError, match="more than one shard"):
        rb.aggregate(tmp_path, tmp_path / "ref")


# ---------------------------------------------------------------------------
# summary, publish and fetch (the gh CLI is faked: no network)
# ---------------------------------------------------------------------------


def _aggregated(tmp_path: Path) -> Path:
    """A small aggregated directory: composite quick (2 seeds) and families-free."""
    src = tmp_path / "raw"
    # The run started before midnight; aggregation happens "now".
    for seed, score, started in ((42, 0.5, "2026-01-02T00:10:00+00:00"), (7, 0.25, "2026-01-01T23:50:00+00:00")):
        d = src / f"shard-{seed}" / "composite-quick"
        d.mkdir(parents=True)
        HarnessResult(HarnessConfig(seed=seed), "", 0, 0.0, [], score).save(str(d / f"composite-quick_s{seed}.json"))
        meta = {
            "suite": "composite-quick",
            "shard": str(seed),
            "status": 0,
            "git_sha": "abc1234",
            "github_run_id": "99",
            "started": started,
        }
        (d / f"meta_{seed}.json").write_text(json.dumps(meta))
    fam = src / "f" / "families-free"
    fam.mkdir(parents=True)
    rows = [
        {"seed": 42, "s": "A", "aocc": 0.4, "err": None},
        {"seed": 42, "s": "B", "aocc": 0.2, "err": "Crash"},
    ]
    (fam / "families-free_shard01.json").write_text(json.dumps(rows))
    out = tmp_path / "ref"
    rb.aggregate(src, out)
    return out


def test_aggregate_writes_the_summary(tmp_path):
    out = _aggregated(tmp_path)
    summary = json.loads((out / "SUMMARY.json").read_text())
    assert summary["release"] is None and summary["github_run_id"] == ["99"]
    comp = summary["suites"]["composite-quick"]["composite_score"]
    assert comp["mean"] == pytest.approx(0.375) and comp["min"] == 0.25 and comp["max"] == 0.5
    assert comp["per_seed"] == {"42": 0.5, "7": 0.25}
    fam = summary["suites"]["families-free"]
    assert fam["rows"] == 2 and fam["failed_rows"] == 1
    assert fam["per_spec_aocc"] == {"A": 0.4, "B": 0.2}
    md = rb.summary_markdown(summary)
    assert "composite mean 0.3750" in md and "<!-- rebaseline-run: 99 -->" in md


class FakeGh:
    """Records ``gh`` calls against a fake repository ``o/r``.

    ``releases``: tag -> ``(notes, is_draft)``; ``tags``: git tag -> commit,
    or ``("annotated", commit)`` for an annotated tag (a published release's
    tag exists implicitly, pointing at ``abc1234``).  ``summary`` /
    ``manifest``: the text of the release's ``SUMMARY.json`` / manifest
    assets for ``release download``.
    """

    def __init__(self, releases=None, tags=None, tarball=None, summary=None, manifest=None):
        self.calls = []
        self.releases = dict(releases or {})
        self.tags = dict(tags or {})
        for tag, (_, draft) in self.releases.items():
            if not draft:
                self.tags.setdefault(tag, "abc1234")
        self.tarball = tarball
        self.summary = summary
        self.manifest = manifest
        self.uploaded = {}

    def writes(self):
        return [c for c in self.calls if c[0] == "release" and c[1] in ("create", "upload", "edit")]

    def _list(self, jq):
        # GitHub API objects, projected by the object constructor of the
        # caller's --jq: a field it does not select is missing from the output.
        fields = dict((k, v) for k, v in re.findall(r"(\w+):\s*\.(\w+)", jq))
        lines = []
        for tag, (body, draft) in self.releases.items():
            api = {"tag_name": tag, "draft": draft, "body": body, "name": tag}
            lines.append(json.dumps({k: api[v] for k, v in fields.items()}))
        return "".join(line + "\n" for line in lines)

    def __call__(self, args, check=True):
        args = list(args)
        self.calls.append(args)
        rc, out, err = 0, "", ""
        if args[:2] == ["api", "--paginate"]:
            out = self._list(args[args.index("--jq") + 1])
        elif args[0] == "api" and "/git/tags/" in args[1]:
            tag = args[1].rsplit("/git/tags/tagobj-", 1)[1]
            out = json.dumps({"object": {"type": "commit", "sha": self.tags[tag][1]}})
        elif args[0] == "api":
            tag = args[1].rsplit("/git/ref/tags/", 1)[1]
            if tag not in self.tags:
                rc, err = 1, "gh: Not Found (HTTP 404)"
            elif isinstance(self.tags[tag], tuple):
                out = json.dumps({"object": {"type": "tag", "sha": f"tagobj-{tag}"}})
            else:
                out = json.dumps({"object": {"type": "commit", "sha": self.tags[tag]}})
        elif args[:2] == ["release", "upload"]:
            # The tarball lives in a temporary directory: read it now.
            for a in args[3:]:
                if a.endswith(".tar.gz"):
                    with tarfile.open(a) as tar:
                        self.uploaded[Path(a).name] = sorted(tar.getnames())
                elif a.endswith(".json"):
                    self.uploaded[Path(a).name] = json.loads(Path(a).read_text())
        elif args[:2] == ["release", "download"]:
            dest = Path(args[args.index("--dir") + 1])
            patterns = [args[i + 1] for i, a in enumerate(args) if a == "--pattern"]
            if self.tarball is not None and self.tarball.name in patterns:
                shutil.copyfile(self.tarball, dest / self.tarball.name)
            # Default assets of an existing release: a summary naming it.
            summary = self.summary
            if summary is None and args[2] in self.releases:
                summary = json.dumps({"release": {"tag": args[2]}, "from": "release"})
            if summary is not None and "SUMMARY.json" in patterns:
                (dest / "SUMMARY.json").write_text(summary)
            if self.manifest is not None and "ref_MANIFEST.json" in patterns:
                (dest / "ref_MANIFEST.json").write_text(self.manifest)
        if check and rc:
            raise subprocess.CalledProcessError(rc, ["gh", *args], out, err)
        return subprocess.CompletedProcess(["gh", *args], rc, out, err)


def test_publish_creates_a_prerelease_and_records_it(tmp_path, monkeypatch):
    out = _aggregated(tmp_path)
    gh = FakeGh()
    monkeypatch.setattr(rb, "_gh", gh)
    assert json.loads((out / "SUMMARY.json").read_text())["release"] is None
    summary = rb.publish(out, None, "o/r", "abc1234")

    # The date of the earliest shard start, not of the aggregation.
    tag = "rebaseline-2026-01-01"
    # Immutable releases: create a draft, attach the assets, then publish.
    assert [c[:2] for c in gh.calls[-3:]] == [["release", "create"], ["release", "upload"], ["release", "edit"]]
    create, _, finish = gh.calls[-3:]
    assert create[2] == tag and "--draft" in create and "--prerelease" in create and "--latest=false" in create
    assert create[create.index("--target") + 1] == "abc1234"
    assert finish[2] == tag and "--draft=false" in finish and "--latest=false" in finish
    assert gh.uploaded[f"{tag}.tar.gz"] == sorted(p.name for p in out.glob("ref_*.json"))
    assert "ref_MANIFEST.json" in gh.uploaded[f"{tag}.tar.gz"]
    release = {"tag": tag, "url": f"https://github.com/o/r/releases/tag/{tag}", "asset": f"{tag}.tar.gz"}
    assert gh.uploaded["SUMMARY.json"]["release"] == release
    assert summary["release"] == release
    assert json.loads((out / "SUMMARY.json").read_text())["release"] == release
    assert gh.uploaded["SUMMARY.json"] == summary  # the committed copy equals the asset
    # Nothing but the summary is left behind next to the results.
    assert not list(tmp_path.glob("*.tar.gz")) and not list(out.glob("*.tar.gz"))
    # Re-summarizing keeps the recorded release.
    assert rb.write_summary(out)["release"] == release


OTHER = "<!-- rebaseline-run: 12 -->"
MINE = "<!-- rebaseline-run: 99 -->"


def _publish(out, monkeypatch, tag: Optional[str] = "rebaseline-x", target=None, **fake):
    gh = FakeGh(**fake)
    monkeypatch.setattr(rb, "_gh", gh)
    return rb.publish(out, tag, "o/r", target)["release"], gh


def test_publish_never_touches_another_runs_release(tmp_path, monkeypatch):
    out = _aggregated(tmp_path)
    release, gh = _publish(out, monkeypatch, releases={"rebaseline-x": (OTHER, False)})
    assert release["tag"] == "rebaseline-x-run99"
    assert [c[:3] for c in gh.writes()][0] == ["release", "create", "rebaseline-x-run99"]
    assert all(c[2] == "rebaseline-x-run99" for c in gh.writes())
    # Another run's *draft* is not adopted either.
    release, gh = _publish(out, monkeypatch, releases={"rebaseline-x": (OTHER, True)})
    assert release["tag"] == "rebaseline-x-run99"


def test_publish_treats_a_bare_tag_as_taken(tmp_path, monkeypatch):
    # e.g. a tag burned by a deleted immutable release: no release, but the tag exists.
    out = _aggregated(tmp_path)
    release, gh = _publish(out, monkeypatch, tag=None, tags={"rebaseline-2026-01-01": "abc1234"})
    assert release["tag"] == "rebaseline-2026-01-01-run99"
    assert gh.writes()[0][:3] == ["release", "create", "rebaseline-2026-01-01-run99"]


def test_publish_resumes_its_draft_and_leaves_its_published_release(tmp_path, monkeypatch):
    out = _aggregated(tmp_path)
    # An interrupted attempt of this run left a draft: fill and publish it.
    release, gh = _publish(out, monkeypatch, releases={"rebaseline-x": (MINE, True)})
    assert release["tag"] == "rebaseline-x"
    assert [c[:3] for c in gh.writes()] == [["release", "upload", "rebaseline-x"], ["release", "edit", "rebaseline-x"]]

    # Already published: immutable, nothing written (a re-run of the job does not fail).
    release, gh = _publish(out, monkeypatch, releases={"rebaseline-x": (MINE, False)})
    assert release["tag"] == "rebaseline-x" and gh.writes() == []


def test_publish_finds_its_fallback_release(tmp_path, monkeypatch):
    # Another run owns rebaseline-x, this run already owns rebaseline-x-run99.
    out = _aggregated(tmp_path)
    releases = {"rebaseline-x": (OTHER, False), "rebaseline-x-run99": (MINE, True)}
    release, gh = _publish(out, monkeypatch, releases=releases)
    assert release["tag"] == "rebaseline-x-run99"
    assert [c[:3] for c in gh.writes()] == [
        ["release", "upload", "rebaseline-x-run99"],
        ["release", "edit", "rebaseline-x-run99"],
    ]
    releases["rebaseline-x-run99"] = (MINE, False)
    release, gh = _publish(out, monkeypatch, releases=releases)
    assert release["tag"] == "rebaseline-x-run99" and gh.writes() == []


def test_publish_reuses_its_release_under_any_tag(tmp_path, monkeypatch):
    # A re-run after midnight must not publish a duplicate under a new date.
    out = _aggregated(tmp_path)
    release, gh = _publish(out, monkeypatch, tag=None, releases={"rebaseline-2025-12-31": (MINE, False)})
    assert release["tag"] == "rebaseline-2025-12-31" and gh.writes() == []


def test_publish_refuses(tmp_path, monkeypatch):
    out = _aggregated(tmp_path)
    # A resumed draft whose (existing) tag points elsewhere than the measured commit.
    with pytest.raises(ValueError, match="points at"):
        _publish(
            out,
            monkeypatch,
            target="def5678",
            releases={"rebaseline-x": (MINE, True)},
            tags={"rebaseline-x": "abc1234"},
        )
    # The fallback is taken too.
    with pytest.raises(ValueError, match="taken too"):
        _publish(out, monkeypatch, tags={"rebaseline-x": "a", "rebaseline-x-run99": "b"})
    # Any gh failure but a 404 is an error, not "absent".
    monkeypatch.setattr(rb, "_gh", lambda args, check=True: subprocess.CompletedProcess(args, 1, "", "HTTP 502"))
    with pytest.raises(RuntimeError, match="502"):
        rb.tag_commit("rebaseline-x", "o/r")


def test_publish_without_a_run_id_needs_a_tag(tmp_path, monkeypatch):
    out = _aggregated(tmp_path)
    manifest = json.loads((out / "ref_MANIFEST.json").read_text())
    manifest["github_run_id"] = []
    (out / "ref_MANIFEST.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="--tag"):
        _publish(out, monkeypatch, tag=None)
    # An explicit tag of a release from some run is never adopted, and there is no fallback.
    with pytest.raises(ValueError, match="taken"):
        _publish(out, monkeypatch, releases={"rebaseline-x": (MINE, True)})
    release, gh = _publish(out, monkeypatch, releases={"rebaseline-x": ("<!-- rebaseline-run: local -->", True)})
    assert release["tag"] == "rebaseline-x" and gh.writes()[0][1] == "upload"


def test_publish_refuses_failed_shards_under_the_default_tag(tmp_path, monkeypatch):
    out = _aggregated(tmp_path)
    manifest = json.loads((out / "ref_MANIFEST.json").read_text())
    manifest["failed_shards"] = ["ioh-standard/03"]
    (out / "ref_MANIFEST.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="failed shards"):
        _publish(out, monkeypatch, tag=None)
    release, _ = _publish(out, monkeypatch, tag="rebaseline-partial")
    assert release["tag"] == "rebaseline-partial"


def test_publish_failure_leaves_the_local_summary_unpublished(tmp_path, monkeypatch):
    out = _aggregated(tmp_path)
    gh = FakeGh()

    def failing(args, check=True):
        if args[:2] == ["release", "edit"]:
            raise subprocess.CalledProcessError(1, args)
        return gh(args, check)

    monkeypatch.setattr(rb, "_gh", failing)
    with pytest.raises(subprocess.CalledProcessError):
        rb.publish(out, "rebaseline-x", "o/r", None)
    assert json.loads((out / "SUMMARY.json").read_text())["release"] is None


def test_title_of_a_tag_without_the_prefix(tmp_path, monkeypatch):
    out = _aggregated(tmp_path)
    _, gh = _publish(out, monkeypatch, tag="smoke")
    create = gh.writes()[0]
    assert create[create.index("--title") + 1] == "Re-baseline smoke (reference data)"


def test_fetch_unpacks_the_reference_files(tmp_path, monkeypatch):
    out = _aggregated(tmp_path)
    tarball = rb.make_tarball(out, "rebaseline-2026-01-02", tmp_path)
    committed = (out / "SUMMARY.json").read_text()
    gh = FakeGh(tarball=tarball, summary=committed)
    monkeypatch.setattr(rb, "_gh", gh)

    dest = tmp_path / "fetched"
    dest.mkdir()
    (dest / "other.tar.gz").write_text("not ours")
    files = rb.fetch("rebaseline-2026-01-02", dest, repo="o/r")
    assert [f.name for f in files] == sorted(p.name for p in out.glob("ref_*.json"))
    assert HarnessResult.load(str(dest / "ref_composite_quick_s7.json")).config.seed == 7
    assert (dest / "SUMMARY.json").read_text() == committed
    assert (dest / "other.tar.gz").read_text() == "not ours"  # only <tag>.tar.gz is handled
    assert not (dest / "rebaseline-2026-01-02.tar.gz").exists()
    (download,) = gh.calls
    assert download[:3] == ["release", "download", "rebaseline-2026-01-02"] and download[-2:] == ["--repo", "o/r"]
    assert "rebaseline-2026-01-02.tar.gz" in download and "*.tar.gz" not in download


def test_fetch_keeps_differing_committed_files(tmp_path, monkeypatch, capsys):
    out = _aggregated(tmp_path)
    tarball = rb.make_tarball(out, "rebaseline-2026-01-02", tmp_path)
    monkeypatch.setattr(rb, "_gh", FakeGh(tarball=tarball, summary='{"release": "from the release"}'))
    dest = tmp_path / "fetched"
    dest.mkdir()
    (dest / "SUMMARY.json").write_text("local summary")
    (dest / "ref_MANIFEST.json").write_text("local manifest")
    rb.fetch("rebaseline-2026-01-02", dest, repo="o/r")
    assert (dest / "SUMMARY.json").read_text() == "local summary"
    assert (dest / "ref_MANIFEST.json").read_text() == "local manifest"
    err = capsys.readouterr().err
    assert "SUMMARY.json differs" in err and "ref_MANIFEST.json differs" in err


def test_default_fetch_dir():
    assert rb.default_fetch_dir("rebaseline-2026-09-26") == rb.REPO_ROOT / "planning" / "results" / "2026-09-26"
    assert rb.default_fetch_dir("rebaseline-2026-09-26-run5").name == "2026-09-26"
    assert rb.default_fetch_dir("other").name == "other"


def test_a_planned_shard_without_a_meta_file_is_failed(tmp_path, monkeypatch):
    # A shard that died before writing its meta (timeout, lost runner) is invisible without the plan.
    out = _aggregated(tmp_path)  # metas: composite-quick shards "42" and "7"
    planned = [
        {"suite": "composite-quick", "shard": "42", "seeds": "42"},
        {"suite": "composite-quick", "shard": "7", "seeds": "7"},
        {"suite": "ioh-standard", "shard": "03", "seeds": "2025"},
    ]
    manifest = rb.aggregate(tmp_path / "raw", out, planned)
    assert manifest["missing_shards"] == ["ioh-standard/03"]
    assert manifest["failed_shards"] == ["ioh-standard/03"] and manifest["planned_shards"] == 3
    with pytest.raises(ValueError, match="failed shards"):
        _publish(out, monkeypatch, tag=None)
    # Complete against the plan: nothing missing.
    assert rb.aggregate(tmp_path / "raw", out, planned[:2])["failed_shards"] == []
    # The CLI reads the matrix JSON of `plan`.
    (tmp_path / "plan.json").write_text(json.dumps({"include": planned}))
    rb.main(["aggregate", str(tmp_path / "raw"), "--out-dir", str(out), "--plan", str(tmp_path / "plan.json")])
    assert json.loads((out / "ref_MANIFEST.json").read_text())["missing_shards"] == ["ioh-standard/03"]


def test_publish_refuses_the_default_tag_unless_measure_succeeded(tmp_path, monkeypatch):
    out = _aggregated(tmp_path)
    for result in ("failure", "cancelled", "skipped"):
        with pytest.raises(ValueError, match=f"ended '{result}'"):
            gh = FakeGh()
            monkeypatch.setattr(rb, "_gh", gh)
            rb.publish(out, None, "o/r", None, measure_result=result)
        assert gh.writes() == []
    gh = FakeGh()
    monkeypatch.setattr(rb, "_gh", gh)
    assert rb.publish(out, None, "o/r", None, measure_result="success")["release"]["tag"] == "rebaseline-2026-01-01"
    # An explicit tag publishes an incomplete run anyway.
    gh = FakeGh()
    monkeypatch.setattr(rb, "_gh", gh)
    assert rb.publish(out, "rebaseline-partial", "o/r", None, measure_result="failure")["release"]["tag"] == (
        "rebaseline-partial"
    )


def test_an_already_published_release_is_downloaded_not_rewritten(tmp_path, monkeypatch):
    out = _aggregated(tmp_path)
    release_summary = json.dumps({"release": {"tag": "rebaseline-x"}, "from": "release"})
    release, gh = _publish(
        out, monkeypatch, releases={"rebaseline-x": (MINE, False)}, summary=release_summary, manifest='{"m": 1}'
    )
    assert release == {"tag": "rebaseline-x"}  # the release's own summary, returned as is
    assert gh.writes() == []
    assert (out / "SUMMARY.json").read_text() == release_summary
    assert (out / "ref_MANIFEST.json").read_text() == '{"m": 1}'
    (download,) = [c for c in gh.calls if c[:2] == ["release", "download"]]
    assert download[2] == "rebaseline-x" and str(out) in download


def test_annotated_tags_are_dereferenced(tmp_path, monkeypatch):
    out = _aggregated(tmp_path)
    gh = FakeGh(tags={"rebaseline-x": ("annotated", "abc1234")})
    monkeypatch.setattr(rb, "_gh", gh)
    assert rb.tag_commit("rebaseline-x", "o/r") == "abc1234"
    # A resumed draft on an annotated tag: accepted on the measured commit, refused elsewhere.
    draft = {"rebaseline-x": (MINE, True)}
    release, _ = _publish(
        out, monkeypatch, target="abc1234", releases=draft, tags={"rebaseline-x": ("annotated", "abc1234")}
    )
    assert release["tag"] == "rebaseline-x"
    with pytest.raises(ValueError, match="points at abc1234"):
        _publish(out, monkeypatch, target="def5678", releases=draft, tags={"rebaseline-x": ("annotated", "abc1234")})
