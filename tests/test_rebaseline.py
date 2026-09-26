# Copyright 2012-2026 Harald Schilly <harald.schilly@gmail.com>
"""``scripts/rebaseline.py``: the matrix, the shard commands, the aggregation, the release helpers."""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import tarfile
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


# ---------------------------------------------------------------------------
# summary, publish and fetch (the gh CLI is faked: no network)
# ---------------------------------------------------------------------------


def _aggregated(tmp_path: Path) -> Path:
    """A small aggregated directory: composite quick (2 seeds) and families-free."""
    src = tmp_path / "raw"
    for seed, score in ((42, 0.5), (7, 0.25)):
        d = src / f"shard-{seed}" / "composite-quick"
        d.mkdir(parents=True)
        HarnessResult(HarnessConfig(seed=seed), "", 0, 0.0, [], score).save(str(d / f"composite-quick_s{seed}.json"))
        meta = {
            "suite": "composite-quick",
            "shard": str(seed),
            "status": 0,
            "git_sha": "abc1234",
            "github_run_id": "99",
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
    """Records ``gh`` calls; ``releases`` maps tag -> ``(notes, is_draft)`` of the releases that exist."""

    def __init__(self, releases=None, tarball=None):
        self.calls = []
        self.releases = dict(releases or {})
        self.tarball = tarball
        self.uploaded = {}

    def __call__(self, args, check=True):
        args = list(args)
        self.calls.append(args)
        rc, out = 0, ""
        if args[:2] == ["release", "view"]:
            if args[2] in self.releases:
                body, draft = self.releases[args[2]]
                out = json.dumps({"body": body, "isDraft": draft})
            else:
                rc = 1
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
            assert self.tarball is not None
            shutil.copyfile(self.tarball, dest / self.tarball.name)
            (dest / "SUMMARY.json").write_text("{}")
        if check and rc:
            raise subprocess.CalledProcessError(rc, ["gh", *args])
        return subprocess.CompletedProcess(["gh", *args], rc, out, "")


def test_publish_creates_a_prerelease_and_records_it(tmp_path, monkeypatch):
    out = _aggregated(tmp_path)
    gh = FakeGh()
    monkeypatch.setattr(rb, "_gh", gh)
    summary = rb.publish(out, None, "o/r", "abc1234")

    tag = rb.default_tag(summary)
    assert tag.startswith("rebaseline-20")
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
    assert json.loads((out / "SUMMARY.json").read_text())["release"] == release
    # Nothing but the summary is left behind next to the results.
    assert not list(tmp_path.glob("*.tar.gz")) and not list(out.glob("*.tar.gz"))
    # Re-summarizing keeps the recorded release.
    assert rb.write_summary(out)["release"] == release


def test_publish_never_overwrites_another_runs_release(tmp_path, monkeypatch):
    out = _aggregated(tmp_path)
    gh = FakeGh(releases={"rebaseline-x": ("<!-- rebaseline-run: 12 -->", False)})
    monkeypatch.setattr(rb, "_gh", gh)
    assert rb.publish(out, "rebaseline-x", "o/r", None)["release"]["tag"] == "rebaseline-x-run99"
    assert any(c[:3] == ["release", "create", "rebaseline-x-run99"] for c in gh.calls)
    assert not any(c[2] == "rebaseline-x" for c in gh.calls if c[1] in ("upload", "edit"))


def test_publish_resumes_its_draft_and_leaves_its_published_release(tmp_path, monkeypatch):
    out = _aggregated(tmp_path)
    # An interrupted attempt of this run left a draft: fill and publish it.
    gh = FakeGh(releases={"rebaseline-x": ("<!-- rebaseline-run: 99 -->", True)})
    monkeypatch.setattr(rb, "_gh", gh)
    assert rb.publish(out, "rebaseline-x", "o/r", None)["release"]["tag"] == "rebaseline-x"
    assert [c[:3] for c in gh.calls[1:]] == [["release", "upload", "rebaseline-x"], ["release", "edit", "rebaseline-x"]]

    # Already published: immutable, nothing to do (a re-run of the job does not fail).
    gh = FakeGh(releases={"rebaseline-x": ("<!-- rebaseline-run: 99 -->", False)})
    monkeypatch.setattr(rb, "_gh", gh)
    assert rb.publish(out, "rebaseline-x", "o/r", None)["release"]["tag"] == "rebaseline-x"
    assert [c[:2] for c in gh.calls] == [["release", "view"]]


def test_fetch_unpacks_the_reference_files(tmp_path, monkeypatch):
    out = _aggregated(tmp_path)
    tarball = rb.make_tarball(out, "rebaseline-2026-01-02", tmp_path)
    gh = FakeGh(tarball=tarball)
    monkeypatch.setattr(rb, "_gh", gh)

    dest = tmp_path / "fetched"
    files = rb.fetch("rebaseline-2026-01-02", dest, repo="o/r")
    assert [f.name for f in files] == sorted(p.name for p in out.glob("ref_*.json"))
    assert HarnessResult.load(str(dest / "ref_composite_quick_s7.json")).config.seed == 7
    assert not list(dest.glob("*.tar.gz")) and (dest / "SUMMARY.json").exists()
    (download,) = gh.calls
    assert download[:3] == ["release", "download", "rebaseline-2026-01-02"] and download[-2:] == ["--repo", "o/r"]


def test_default_fetch_dir():
    assert rb.default_fetch_dir("rebaseline-2026-09-26") == rb.REPO_ROOT / "planning" / "results" / "2026-09-26"
    assert rb.default_fetch_dir("rebaseline-2026-09-26-run5").name == "2026-09-26"
    assert rb.default_fetch_dir("other").name == "other"
