# Copyright 2012-2026 Harald Schilly <harald.schilly@gmail.com>
"""``scripts/rebaseline.py``: the matrix, the shard commands and the aggregation."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

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
    for suite in rb.SUITES:
        seeds = [int(s) for e in entries if e["suite"] == suite for s in e["seeds"].split(",")]
        assert seeds == list(rb.ROSTER)
    assert len({(e["suite"], e["shard"]) for e in entries}) == len(entries)
    assert rb.plan(rb.resolve_suites("ioh-quick"), [42]) == [{"suite": "ioh-quick", "shard": "01", "seeds": "42"}]


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


def _ioh_shard(path: Path, seeds):
    results = []
    for seed in seeds:
        run = IOHRunRecord(
            problem_kind="MA-BBOB",
            dim=2,
            instance=0,
            strategy_name="A",
            rep=0,
            budget=10,
            n_evals=10,
            best_fx=1.0,
            f_opt=0.0,
            aocc=seed / 10000,
            elapsed_s=0.1,
            seed=seed,
        )
        results.append(IOHHarnessResult("ioh-quick", "MA-BBOB", -8, 2, [run], sync_eval=True))
    multi = IOHMultiSeedResult("ioh-quick", "MA-BBOB", -8, 2, list(seeds), results, sync_eval=True)
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


def test_aggregate_rejects_a_seed_measured_twice(tmp_path):
    _ioh_shard(tmp_path / "a" / "ioh-quick" / "ioh-quick_shard01.json", [42])
    _ioh_shard(tmp_path / "b" / "ioh-quick" / "ioh-quick_shard02.json", [42])
    with pytest.raises(ValueError, match="more than one shard"):
        rb.aggregate(tmp_path, tmp_path / "ref")
