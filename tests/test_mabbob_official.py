# Copyright 2012-2026 Harald Schilly <harald.schilly@gmail.com>
"""The official MA-BBOB rows: split, data, the competition's AOCC, the worker kind and the comparison tool."""

from __future__ import annotations

import dataclasses
import hashlib
import http.client
import importlib.util
import io
import json
import os
import sys
import urllib.error
import zipfile
import zlib
from pathlib import Path
from typing import List

import numpy as np
import pytest

import panobbgo.harness_ioh as hi
import panobbgo.mabbob_official as mo
from panobbgo.features import FeatureLogSpec
from panobbgo.harness_ioh import IOHBatterySpec, IOHHarnessResult, IOHRunRecord
from panobbgo.ioh_runner import aocc
from panobbgo.lib.ioh_wrapper import worker_available

_ROOT = Path(__file__).resolve().parent.parent
_FIXTURE = json.loads((_ROOT / "tests" / "data" / "mabbob_official_fixture.json").read_text())


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", _ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cmp = _load_script("mabbob_compare")
cli = _load_script("ioh_benchmark")

requires_worker = pytest.mark.skipif(not worker_available(), reason="IOH worker venv not set up")


def _official_data() -> None:
    """The instance files, over the network or from the cache; only CI insists (a skip there would hide a failure)."""
    try:
        mo.ensure_data()
    except RuntimeError as exc:
        if os.environ.get("CI"):
            raise
        pytest.skip(f"official instance files unavailable: {exc}")


TUNE, HELD = mo.split_rows()


# ---------------------------------------------------------------------------
# The split
# ---------------------------------------------------------------------------


def test_split_is_exact_disjoint_and_complete():
    assert len(TUNE) == len(HELD) == 500
    assert not set(TUNE) & set(HELD)
    assert sorted(TUNE + HELD) == list(range(1000))
    assert TUNE == tuple(sorted(TUNE)) and HELD == tuple(sorted(HELD))


def test_split_is_pinned():
    # A changed salt or rule is a new split: every number on the old held-out half is then development data.
    assert mo.SPLIT_SALT == "panobbgo-mabbob-official-split-2026-11-04"
    assert TUNE[:8] == (0, 2, 3, 6, 7, 10, 12, 14) and HELD[:8] == (1, 4, 5, 8, 9, 11, 13, 17)
    digest = hashlib.sha256(",".join(map(str, TUNE)).encode()).hexdigest()
    assert digest == "b5f74025f2f8675a0f1e474db1c97b5d170dfe81d94b41d51cbb134d18a8d983"


def test_split_of():
    assert {mo.split_of(r) for r in TUNE} == {"tune"} and {mo.split_of(r) for r in HELD} == {"heldout"}
    for bad in (-1, 1000):
        with pytest.raises(ValueError):
            mo.split_of(bad)


def test_chunks_partition_the_rows():
    chunks = [mo.chunk_rows(TUNE, k, 4) for k in range(1, 5)]
    assert sorted(r for c in chunks for r in c) == list(TUNE)
    assert {len(c) for c in chunks} == {125}
    for bad in (0, 5):
        with pytest.raises(ValueError):
            mo.chunk_rows(TUNE, bad, 4)


# ---------------------------------------------------------------------------
# Held-out needs an explicit flag
# ---------------------------------------------------------------------------


def _spec(**kw) -> IOHBatterySpec:
    base = dict(name="x", problem_kind=mo.OFFICIAL_KIND, dims=(2, 5), instances=TUNE[:2], budget_multiplier=2000)
    return IOHBatterySpec(**{**base, **kw})


def test_default_battery_is_tune_and_heldout_battery_is_flagged():
    tune = hi.make_mabbob_official_battery()
    held = hi.make_mabbob_official_heldout_battery()
    assert tune.instances == TUNE and not tune.heldout and tune.problem_kind == mo.OFFICIAL_KIND
    assert held.instances == HELD and held.heldout
    for b in (tune, held):
        assert b.dims == (2, 5) and b.budget_for(2) == 4000 and b.budget_for(5) == 10000 and b.reps == 1
    assert hi.make_mabbob_official_battery(chunk=(2, 100)).instances == TUNE[1::100]
    assert hi.make_mabbob_official_heldout_battery(chunk=(3, 4)).instances == HELD[2::4]
    assert hi.make_mabbob_official_battery(chunk=(2, 100)).name == tune.name  # shards merge


def test_heldout_rows_are_refused_without_the_flag():
    with pytest.raises(ValueError, match="held-out"):
        _spec(instances=(TUNE[0], HELD[0]))
    with pytest.raises(ValueError, match="not official"):
        _spec(instances=(1000,))
    with pytest.raises(ValueError, match="tune|held-out"):
        _spec(instances=TUNE[:1], heldout=True)
    with pytest.raises(ValueError, match="official"):
        IOHBatterySpec(name="x", problem_kind="MA-BBOB", dims=(2,), instances=(0,), heldout=True)


@pytest.mark.parametrize(
    "change",
    [
        dict(dims=(2,)),
        dict(reps=2),
        dict(budget_multiplier=500),
        dict(log_hi=6.0),
        dict(extra_builder_kwargs=(("a", 1),)),
    ],
)
def test_a_heldout_battery_has_no_knobs(change):
    with pytest.raises(ValueError, match="fixed"):
        dataclasses.replace(hi.make_mabbob_official_heldout_battery(), **change)


def test_the_problem_refuses_a_heldout_row_before_any_worker_starts(monkeypatch):
    from panobbgo.lib import ioh_wrapper

    monkeypatch.setattr(ioh_wrapper.IOHProblem, "_start_fresh_worker", lambda self: pytest.fail("no worker"))
    with pytest.raises(ValueError, match="held-out"):
        ioh_wrapper.IOHProblem(mo.OFFICIAL_KIND, HELD[0], 2)
    with pytest.raises(ValueError, match="heldout=True"):
        ioh_wrapper.IOHProblem("MA-BBOB", 0, 2, heldout=True)


def test_a_chunk_past_the_rows_is_refused():
    with pytest.raises(ValueError, match="empty"):
        mo.chunk_rows(TUNE, 600, 1000)


def test_run_one_refuses_a_heldout_row_without_the_flag(monkeypatch):
    kw = dict(
        strategy_spec=hi.make_ioh_strategies()[0],
        problem_kind=mo.OFFICIAL_KIND,
        dim=2,
        rep=0,
        budget=10,
        seed=1,
        builder_kwargs={},
        log_lo=-8.0,
        log_hi=2.0,
    )
    with pytest.raises(ValueError, match="held-out"):
        hi._run_one(instance=HELD[0], **kw)
    with pytest.raises(ValueError, match="tune|held-out"):
        hi._run_one(instance=TUNE[0], heldout=True, **kw)
    with pytest.raises(ValueError, match="never train"):
        hi._run_one(instance=HELD[0], heldout=True, log_features=FeatureLogSpec.parse("0.5"), **kw)


def _stub_run_one(calls: List[dict]):
    def run_one(**task):
        calls.append(task)
        return IOHRunRecord(
            problem_kind=task["problem_kind"],
            dim=task["dim"],
            instance=task["instance"],
            strategy_name=task["strategy_spec"].name,
            rep=task["rep"],
            budget=task["budget"],
            n_evals=task["budget"],
            best_fx=0.0,
            f_opt=0.0,
            aocc=0.5,
            elapsed_s=0.0,
            seed=task["seed"],
        )

    return run_one


def test_harness_prints_the_banner_and_passes_the_flag(monkeypatch, capsys):
    calls: List[dict] = []
    monkeypatch.setattr(hi, "_run_one", _stub_run_one(calls))
    monkeypatch.setattr(hi, "ensure_data", lambda: None)
    spec = hi.make_ioh_strategies()[:1]
    hi.run_ioh_harness(spec, hi.make_mabbob_official_heldout_battery(chunk=(1, 250)), progress=False)
    assert "HELD-OUT" in capsys.readouterr().err
    assert calls and all(c["heldout"] is True for c in calls)
    with pytest.raises(ValueError, match="never train"):
        hi.run_ioh_harness(
            spec,
            hi.make_mabbob_official_heldout_battery(chunk=(1, 250)),
            progress=False,
            log_features=FeatureLogSpec.parse("0.5"),
        )
    calls.clear()
    hi.run_ioh_harness(spec, hi.make_mabbob_official_battery(chunk=(1, 250)), progress=False)
    assert "HELD-OUT" not in capsys.readouterr().err
    assert all(c["heldout"] is False for c in calls)


@pytest.mark.parametrize(
    "argv",
    [
        ["--mabbob-official-heldout", "--reps", "2"],
        ["--mabbob-official-heldout", "--budget-multiplier", "100"],
        ["--mabbob-official-heldout", "--log-features", "0.5"],
        ["--mabbob-official-heldout", "--virtual-workers", "4"],
        ["--mabbob-official-heldout", "--legacy"],
        ["--quick", "--official-chunk", "1/4"],
        ["--mabbob-official-heldout", "--decision-seeds"],
        ["--mabbob-official-heldout", "--seeds", "1", "2"],
        ["--mabbob-official", "--official-chunk", "600/1000"],
    ],
)
def test_cli_refuses(argv, monkeypatch):
    # A broken guard must fail here, not start a real run (and never a real held-out one).
    for name in ("run_ioh_harness", "run_ioh_harness_multi_seed"):
        monkeypatch.setattr(cli, name, lambda *a, **k: pytest.fail("the run was not refused"))
    with pytest.raises(SystemExit) as exc:
        cli.main(["run", *argv])
    assert exc.value.code not in (0, None)


def test_cli_flags_select_the_batteries(monkeypatch):
    seen = []

    def fake_run(strategies, battery, **kw):
        seen.append(battery)
        return IOHHarnessResult(battery.name, battery.problem_kind, -8.0, 2.0, [])

    monkeypatch.setattr(cli, "run_ioh_harness", fake_run)
    assert cli.main(["run", "--mabbob-official", "--official-chunk", "1/100", "--quiet"]) == 0
    assert seen[-1].instances == TUNE[0::100] and not seen[-1].heldout
    assert cli.main(["run", "--mabbob-official-heldout", "--official-chunk", "1/100", "--quiet"]) == 0
    assert seen[-1].instances == HELD[0::100] and seen[-1].heldout


def test_the_jso_spec_joins_only_when_named():
    assert hi.JSO_ONLY_NAME not in {s.name for s in hi.make_ioh_strategies()}
    assert hi.make_jso_strategies(["RoundRobin_CMAES"]) == []
    [spec] = hi.make_jso_strategies([hi.JSO_ONLY_NAME])
    assert spec.name == "RoundRobin_JSO" and spec.heuristics[0][1] == {"NP_init": "auto"}


# ---------------------------------------------------------------------------
# Existing batteries are untouched
# ---------------------------------------------------------------------------


def test_existing_presets_are_unchanged():
    assert hi.make_quick_battery().instances == (0, 1, 2) and hi.make_quick_battery().problem_kind == "MA-BBOB"
    assert hi.make_full_battery().instances == tuple(range(10)) and hi.make_full_battery().budget_multiplier == 2000
    assert hi.make_full_battery().heldout is False
    # The seed payload of an existing kind (the whole reproducibility contract of a battery).
    assert hi._derive_seed(42, "MA-BBOB", 2, 0, "RoundRobin_CMAES", 0) == int.from_bytes(
        hashlib.sha256(b"42|MA-BBOB|2|0|RoundRobin_CMAES|0").digest()[:4], "little"
    )
    # A row of one instance id means something else in each kind: different RNG stream.
    assert hi._derive_seed(42, mo.OFFICIAL_KIND, 2, 0, "S", 0) != hi._derive_seed(42, "MA-BBOB", 2, 0, "S", 0)


@requires_worker
def test_existing_worker_kinds_build_the_same_problems():
    # Literal values from the worker before the official kind existed (master ca3a85b).
    from panobbgo.lib.ioh_wrapper import IOHProblem

    x = np.array([1.0, -2.0, 3.0, -4.0, 0.5])
    for kind, kw, fx, opt, pid, name in (
        ("MA-BBOB", {}, 4.658774863566108, 3.5082494139920753e-12, 0, "ManyAffine"),
        ("BBOB", {"fid": 8}, 7328.403635503238, 98.62, 8, "Rosenbrock"),
    ):
        p = IOHProblem(kind, 3, 5, **kw)
        try:
            assert float(p.eval(x)) == pytest.approx(fx, rel=1e-12) and p.optimum_y == pytest.approx(opt, rel=1e-9)
            assert (p.ioh_problem_id, p.ioh_instance, p.ioh_name) == (pid, 3, name)
        finally:
            p.close()


# ---------------------------------------------------------------------------
# The data: pinned URL, verified content, no vendored files
# ---------------------------------------------------------------------------


def _fake_csvs():
    header24 = "," + ",".join(map(str, range(24)))
    header20 = "," + ",".join(map(str, range(20)))
    iids = "\n".join([header24] + [f"{i}," + ",".join(["1"] * 24) for i in range(mo.N_ROWS)]) + "\n"
    weights = "\n".join([header24] + [f"{i}," + ",".join(["0.5"] * 24) for i in range(mo.N_ROWS)]) + "\n"
    xopt = "\n".join([header20] + [f"{i}," + ",".join(["0.25"] * 20) for i in range(mo.N_ROWS)]) + "\n"
    return {"iids.csv": iids.encode(), "weights.csv": weights.encode(), "opt_locs.csv": xopt.encode()}


def _zip_of(files) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)
        z.writestr("not_read.py", "raise SystemExit('never run')")
    return buf.getvalue()


def test_ensure_data_fetches_once_and_verifies(monkeypatch, tmp_path):
    files = _fake_csvs()
    monkeypatch.setattr(mo, "CSV_SHA256", {n: hashlib.sha256(d).hexdigest() for n, d in files.items()})
    fetched = []
    monkeypatch.setattr(mo, "_download", lambda url: fetched.append(url) or _zip_of(files))
    target = tmp_path / "cache"
    assert mo.ensure_data(target) == target
    assert sorted(p.name for p in target.iterdir()) == sorted(files) and len(fetched) == 1
    mo.ensure_data(target)
    assert len(fetched) == 1, "verified files are not fetched again"
    table = mo._load_table(str(target))
    assert table.xopt[7][:2] == (0.25, 0.25) and table.iids[999][23] == 1 and table.weights[0][0] == 0.5


@pytest.mark.parametrize(
    "failure",
    [
        http.client.IncompleteRead(b"x"),
        urllib.error.URLError("down"),
        b"<html>a proxy's error page</html>",
        b"no members",
        zlib.error("corrupt deflate"),
    ],
    ids=["cut", "url", "html", "members", "deflate"],
)
def test_a_failed_fetch_is_retried_then_names_the_offline_way_out(monkeypatch, tmp_path, failure):
    monkeypatch.setattr(mo.time, "sleep", lambda _s: None)
    files = _fake_csvs()
    monkeypatch.setattr(mo, "CSV_SHA256", {n: hashlib.sha256(d).hexdigest() for n, d in files.items()})
    good = _zip_of(files)
    calls: List[int] = []

    def flaky(_url):
        calls.append(1)
        if len(calls) > 1:
            return good
        if isinstance(failure, Exception):
            raise failure
        return _zip_of({"other.txt": b"x"}) if failure == b"no members" else failure

    monkeypatch.setattr(mo, "_download", flaky)
    assert mo.ensure_data(tmp_path / "ok") == tmp_path / "ok" and len(calls) == 2

    def down(_url):
        raise urllib.error.URLError("down")

    monkeypatch.setattr(mo, "_download", down)
    with pytest.raises(RuntimeError, match=mo.DATA_DIR_ENV):
        mo.ensure_data(tmp_path / "never")


def test_cached_files_are_readable_by_others(monkeypatch, tmp_path):
    files = _fake_csvs()
    monkeypatch.setattr(mo, "CSV_SHA256", {n: hashlib.sha256(d).hexdigest() for n, d in files.items()})
    monkeypatch.setattr(mo, "_download", lambda _url: _zip_of(files))
    target = mo.ensure_data(tmp_path / "shared")
    assert all((target / n).stat().st_mode & 0o777 == 0o644 for n in mo.CSV_SHA256)


def test_ensure_data_refuses_a_wrong_download_and_writes_nothing(monkeypatch, tmp_path):
    files = _fake_csvs()
    monkeypatch.setattr(mo, "CSV_SHA256", {n: hashlib.sha256(d).hexdigest() for n, d in files.items()})
    tampered = {**files, "weights.csv": files["weights.csv"] + b"\n"}
    monkeypatch.setattr(mo, "_download", lambda url: _zip_of(tampered))
    target = tmp_path / "cache"
    with pytest.raises(ValueError, match="SHA-256"):
        mo.ensure_data(target)
    assert list(target.iterdir()) == []


def test_ensure_data_does_not_overwrite_a_named_directory(monkeypatch, tmp_path):
    monkeypatch.setattr(mo, "_download", lambda url: pytest.fail("must not download"))
    (tmp_path / "weights.csv").write_text("someone's work")
    with pytest.raises(ValueError, match="pinned"):
        mo.ensure_data(tmp_path)
    assert (tmp_path / "weights.csv").read_text() == "someone's work"


def test_the_url_is_pinned_to_a_commit_and_no_csv_is_vendored():
    assert "a3ecaac6c06aa97190249c0c2c74b6a3aac14304" in mo.DATA_URL
    tracked = _ROOT / "panobbgo"
    assert not list(tracked.rglob("opt_locs.csv")) and not list((_ROOT / "tools").glob("**/src/**/opt_locs.csv"))


# ---------------------------------------------------------------------------
# The scorer, pinned to iohinspector with literal values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("run", _FIXTURE["runs"], ids=lambda r: f"{r['entry']}-d{r['dim']}-r{r['row']}")
def test_official_aocc_matches_iohinspector(run):
    points = [(int(e), float(v)) for e, v in run["points"]]
    score = mo.official_aocc_of_points(points, run["budget"])
    if run["iohinspector"] is not None:
        assert score == pytest.approx(run["iohinspector"], abs=1e-12)
    else:
        # ERADS row 452 at d = 2: below the lower target from the first evaluation.
        assert (run["entry"], run["row"]) == ("ERADS", 452) and score == pytest.approx(1.0, abs=1e-12)


def test_official_and_repo_aocc_differ_by_the_stated_term():
    # q(1) = 1 (gap 1e-8), then 0 (gap 100)?  A best-so-far never rises; take a falling trace.
    gaps = [100.0, 1.0, 1e-8, 1e-8]
    repo = aocc(gaps, f_opt=0.0, budget=4)
    assert repo == pytest.approx((0.0 + 0.2 + 1.0 + 1.0) / 4)
    # The official integral: q(1) over [0, 1), q(1) over [1, 2), q(2) over [2, 3), q(3) over [3, 4].
    assert mo.official_aocc(gaps, 4) == pytest.approx((0.0 + 0.0 + 0.2 + 1.0) / 4)
    assert mo.official_aocc(gaps, 4) == pytest.approx(repo + (0.0 - 1.0) / 4)


def test_conversion_is_exact_on_random_traces():
    rng = np.random.default_rng(3)
    for budget in (50, 4000):
        trace = np.minimum.accumulate(10.0 ** rng.uniform(-10, 3, size=budget))
        repo = aocc(trace.tolist(), f_opt=0.0, budget=budget)
        conv = mo.official_from_repo_aocc(repo, float(trace[0]), float(trace[-1]), budget)
        # By the definition: left endpoints, the first value counted twice.
        q = mo._quality(trace, -8.0, 2.0)
        assert conv == pytest.approx((q[0] + q[:-1].sum()) / budget, abs=1e-12)
        assert abs(conv - repo) <= 1.0 / budget


def test_budget_truncation_and_hold():
    # Past the budget nothing counts; a run that stops early holds its best to the end.
    over = [(1, 100.0), (2, 1.0), (5, 1e-3), (9, 1e-9)]
    assert mo.expand_improvements(over, 6).tolist() == [100.0, 1.0, 1.0, 1.0, 1e-3, 1e-3]
    assert mo.official_aocc_of_points(over, 6) == pytest.approx(mo.official_aocc([100.0, 1.0, 1.0, 1.0, 1e-3, 1e-3], 6))
    assert mo.expand_improvements([(1, 5.0), (3, 2.0)], 5).tolist() == [5.0, 5.0, 2.0, 2.0, 2.0]
    # A value logged at exactly the budget counts.
    assert mo.expand_improvements([(1, 5.0), (4, 2.0)], 4).tolist() == [5.0, 5.0, 5.0, 2.0]
    # Evaluations before the first logged line are the worst score.
    assert mo.expand_improvements([(3, 5.0)], 4).tolist() == [np.inf, np.inf, 5.0, 5.0]
    assert mo.official_aocc([], 4) == 0.0
    long = [1.0] * 10
    assert mo.official_aocc(long, 4) == mo.official_aocc(long[:4], 4)


def test_short_trace_is_scored_like_its_hold():
    assert mo.official_aocc([100.0, 1e-8], 5) == pytest.approx(mo.official_aocc([100.0, 1e-8, 1e-8, 1e-8, 1e-8], 5))


def test_parse_ioh_dat():
    text = ["evaluations raw_y\n", "1 3.5\n", "7 1.25\n", "evaluations raw_y x0 x1\n", "1 2.0 -4.0 -4.0\n"]
    assert mo.parse_ioh_dat(text) == [[(1, 3.5), (7, 1.25)], [(1, 2.0)]]
    with pytest.raises(ValueError):
        mo.parse_ioh_dat(["1 2.0\n"])


def test_load_logged_entry_checks_the_rows(tmp_path):
    dat = tmp_path / "IOHprofiler_f100_DIM2.dat"
    dat.write_text("".join(f"evaluations raw_y\n1 {i + 1}.0\n" for i in range(mo.N_ROWS)))
    runs = [{"instance": i, "evals": 1, "best": {"evals": 1, "y": i + 1.0, "x": [0.0, 0.0]}} for i in range(mo.N_ROWS)]
    meta = {
        "algorithm": {"name": "A"},
        "scenarios": [{"dimension": 2, "path": "IOHprofiler_f100_DIM2.dat", "runs": runs}],
    }
    (tmp_path / "IOHprofiler_f100_ManyAffine.json").write_text(json.dumps(meta))
    entry = mo.load_logged_entry(tmp_path, 2)
    assert entry.name == "A" and entry.rows[3] == 3 and entry.runs[3] == ((1, 4.0),)
    with pytest.raises(ValueError, match="scenario"):
        mo.load_logged_entry(tmp_path, 5)
    runs[1]["instance"] = 0
    (tmp_path / "IOHprofiler_f100_ManyAffine.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="one run per"):
        mo.load_logged_entry(tmp_path, 2)


# ---------------------------------------------------------------------------
# The stored reference
# ---------------------------------------------------------------------------

#: iohinspector 0.0.8, transform_fval(df, 1e-8, 1e2) + get_aocc, on the entries' logged data.
IOHINSPECTOR_MEANS = {
    ("NeighborhoodAdaptiveDE", 2): 0.8668391753836635,
    ("NeighborhoodAdaptiveDE", 5): 0.8077707676737237,
    ("ERADS", 2): 0.8369573536583951,
    ("ERADS", 5): 0.7510839194017003,
    ("PSS", 2): 0.601939713837148,
    ("PSS", 5): 0.5116048148989223,
    ("ACA", 2): 0.4821759440421571,
    ("ACA", 5): 0.32782890365342277,
}


def test_reference_agrees_with_iohinspector():
    doc = cmp.load_reference()
    assert list(doc["entries"]) == ["NeighborhoodAdaptiveDE", "ERADS", "PSS", "ACA"]
    assert doc["instances"]["sha256"] == mo.CSV_SHA256
    for (name, dim), official in IOHINSPECTOR_MEANS.items():
        scores = np.asarray(doc["entries"][name]["aocc"][str(dim)])
        assert doc["entries"][name]["mean"][str(dim)] == pytest.approx(scores.mean(), abs=1e-6)
        if (name, dim) == ("ERADS", 2):
            # iohinspector 0.0.8 loses this one run; its mean is over the other 999.
            assert scores[452] == 1.0
            assert np.delete(scores, 452).mean() == pytest.approx(official, abs=1e-6)
            assert scores.mean() - official == pytest.approx(1.63e-4, abs=1e-6)
        else:
            assert scores.mean() == pytest.approx(official, abs=1e-6)


def test_reference_records_source_and_attribution():
    doc = cmp.load_reference()
    for entry in doc["entries"].values():
        assert entry["source"] and entry["licence"] and entry["attribution"] and entry["algorithm"]
    assert "CC BY 4.0" == doc["entries"]["NeighborhoodAdaptiveDE"]["licence"]
    assert "10.5281/zenodo.15074959" in doc["entries"]["NeighborhoodAdaptiveDE"]["attribution"]
    assert doc["entries"]["ERADS"]["note"] == cmp.ENTRY_SOURCES["ERADS"]["note"] == cmp.ERADS_NOTE
    for name, sources in cmp.ENTRY_SOURCES.items():
        assert {k: doc["entries"][name][k] for k in sources} == sources


def test_a_merged_file_from_several_environments_warns():
    report = {"seeds": [42], "fp_env_ids": ["mixed:a,b"], "problems": {}, "splits": {}, "breakdown_entry": "E"}
    out = io.StringIO()
    cmp.print_report(report, out)
    assert "several floating-point environments" in out.getvalue()


# ---------------------------------------------------------------------------
# The comparison tool, on a small fixture
# ---------------------------------------------------------------------------


def _record(strategy: str, dim: int, row: int, score: float, *, seed: int = 1, first: float = 1.0, last: float = 1.0):
    return IOHRunRecord(
        problem_kind=mo.OFFICIAL_KIND,
        dim=dim,
        instance=row,
        strategy_name=strategy,
        rep=0,
        budget=2000 * dim,
        n_evals=2000 * dim,
        best_fx=last,
        f_opt=0.0,
        aocc=score,
        elapsed_s=0.0,
        seed=seed,
        trace_evals=[1, 2000 * dim],
        trace_fx=[first, last],
    )


def _result(runs: List[IOHRunRecord], **kw) -> IOHHarnessResult:
    base = dict(
        battery_name="mabbob-official-tune",
        problem_kind=mo.OFFICIAL_KIND,
        log_lo=-8.0,
        log_hi=2.0,
        runs=runs,
        sync_eval=True,
    )
    return IOHHarnessResult(**{**base, **kw})


def _reference(value: float = 0.5, other: float = 0.9) -> dict:
    return {
        "entries": {
            "E1": {"aocc": {"2": [value] * mo.N_ROWS, "5": [value] * mo.N_ROWS}},
            "E2": {"aocc": {"2": [other] * mo.N_ROWS, "5": [other] * mo.N_ROWS}},
        }
    }


def _weights() -> np.ndarray:
    # Each row dominated by one function, cycling through the 24.
    w = np.zeros((mo.N_ROWS, 24))
    for row in range(mo.N_ROWS):
        w[row, row % 24] = 0.75
        w[row, (row + 1) % 24] = 0.25
    return w


def _scores(results) -> "cmp.OurScores":
    out = cmp.OurScores()
    for seed, res in results:
        out.add("mem", seed, res)
    return out


def test_compare_paired_delta_wins_and_bootstrap():
    rows = TUNE[:40]
    # Scores above the entry on 30 rows, below on 10.
    runs = [_record("S", d, r, 0.7 if i < 30 else 0.4) for d in (2, 5) for i, r in enumerate(rows)]
    report = cmp.compare(_scores([(1, _result(runs))]), _reference(), _weights(), n_boot=2000)
    assert list(report["splits"]) == ["tune"]
    section = report["splits"]["tune"]
    assert section["coverage"]["S"] == {2: 40, 5: 40} and section["rows"] == 500
    cell = section["strategies"]["S"]["E1"]["d=2"]
    assert (cell["n"], cell["wins"], cell["losses"], cell["ties"]) == (40, 30, 10, 0)
    assert cell["ours"] == pytest.approx((30 * 0.7 + 10 * 0.4) / 40) and cell["entry"] == pytest.approx(0.5)
    assert cell["delta"] == pytest.approx(0.125) and cell["ci_lo"] < 0.125 < cell["ci_hi"]
    # The mean of the two dimensions is the per-row mean, so here the same.
    assert section["strategies"]["S"]["E1"]["mean"]["delta"] == pytest.approx(0.125)
    assert section["strategies"]["S"]["E2"]["d=5"]["delta"] == pytest.approx(0.125 - 0.4)
    # Deterministic.
    again = cmp.compare(_scores([(1, _result(runs))]), _reference(), _weights(), n_boot=2000)
    assert again["splits"]["tune"]["strategies"]["S"]["E1"]["d=2"] == cell


def test_mean_of_dims_needs_both_dims_on_a_row():
    runs = [_record("S", 2, r, 0.6) for r in TUNE[:4]] + [_record("S", 5, r, 0.8) for r in TUNE[:2]]
    section = cmp.compare(_scores([(1, _result(runs))]), _reference(), _weights(), n_boot=200)["splits"]["tune"]
    cells = section["strategies"]["S"]["E1"]
    assert (cells["d=2"]["n"], cells["d=5"]["n"], cells["mean"]["n"]) == (4, 2, 2)
    assert cells["mean"]["ours"] == pytest.approx(0.7)


def test_splits_are_reported_apart():
    runs = [_record("S", 2, TUNE[0], 0.6), _record("S", 2, HELD[0], 0.8)]
    report = cmp.compare(_scores([(1, _result(runs))]), _reference(), _weights(), n_boot=200)
    assert report["splits"]["tune"]["strategies"]["S"]["E1"]["d=2"]["ours"] == pytest.approx(0.6)
    assert report["splits"]["heldout"]["strategies"]["S"]["E1"]["d=2"]["ours"] == pytest.approx(0.8)


def test_only_a_tune_run_has_no_heldout_section():
    runs = [_record("S", 2, TUNE[0], 0.6)]
    assert list(cmp.compare(_scores([(1, _result(runs))]), _reference(), _weights(), n_boot=50)["splits"]) == ["tune"]


def test_scores_are_converted_to_the_official_metric():
    # first = 1e-8 (q = 1), last = 100 (q = 0) is impossible for a best-so-far, but makes the term plain.
    rec = _record("S", 2, TUNE[0], 0.5, first=100.0, last=1e-8)
    assert cmp.official_score_of_record(rec) == pytest.approx(0.5 - 1.0 / 4000)
    # A crash has no trace and keeps its 0.
    crash = dataclasses.replace(rec, aocc=0.0, trace_evals=[], trace_fx=[], error="RuntimeError: x")
    assert cmp.official_score_of_record(crash) == 0.0
    with pytest.raises(ValueError, match="evaluation 1"):
        cmp.official_score_of_record(dataclasses.replace(rec, trace_evals=[2, 4000]))


def test_seeds_are_averaged_per_row():
    a = _result([_record("S", 2, TUNE[0], 0.6, seed=1)])
    b = _result([_record("S", 2, TUNE[0], 0.8, seed=2)])
    report = cmp.compare(_scores([(1, a), (2, b)]), _reference(), _weights(), n_boot=50)
    assert report["seeds"] == [1, 2]
    assert report["splits"]["tune"]["strategies"]["S"]["E1"]["d=2"]["ours"] == pytest.approx(0.7)
    with pytest.raises(ValueError, match="duplicate"):
        _scores([(1, a), (1, a)])


@pytest.mark.parametrize(
    "change,match",
    [
        (dict(problem_kind="MA-BBOB"), "problem kind"),
        (dict(sync_eval=False), "synchronous"),
        (dict(virtual={"workers": 4}), "synchronous"),
        (dict(log_hi=6.0), "bounds"),
    ],
)
def test_runs_outside_the_rules_are_refused(change, match):
    with pytest.raises(ValueError, match=match):
        _scores([(1, _result([_record("S", 2, TUNE[0], 0.6)], **change))])


def test_a_run_at_another_budget_is_refused():
    rec = dataclasses.replace(_record("S", 2, TUNE[0], 0.6), budget=1000)
    with pytest.raises(ValueError, match="2000"):
        _scores([(1, _result([rec]))])


def test_breakdown_by_dominant_function_and_class():
    w = _weights()
    dominant = cmp.dominant_fids(w)
    assert dominant[0] == 1 and dominant[25] == 2
    # Lose on rows dominated by f1, win on the others.
    runs = [_record("S", d, r, 0.1 if dominant[r] == 1 else 0.6) for d in (2, 5) for r in TUNE]
    section = cmp.compare(_scores([(1, _result(runs))]), _reference(), w, n_boot=50)["splits"]["tune"]
    tables = section["breakdown"]["S"]
    f1 = tables["by_dominant_fid"]["1"]
    assert f1["n"] == sum(1 for r in TUNE if dominant[r] == 1)
    assert f1["d=2"] == pytest.approx(-0.4) and f1["mean"] == pytest.approx(-0.4)
    assert tables["by_dominant_fid"]["7"]["d=5"] == pytest.approx(0.1)
    # f1 takes weight 0.75 on its own rows and 0.25 on f24's rows, which it does not lose.
    assert -0.4 < f1["weighted"] < 0.1
    classes = tables["by_dominant_class"]
    assert sum(c["n"] for c in classes.values()) == 500
    assert classes["low-cond"]["mean"] == pytest.approx(0.1) and classes["separable"]["mean"] < 0.1


def test_breakdown_entry_is_selectable():
    runs = [_record("S", 2, TUNE[0], 0.6)]
    report = cmp.compare(_scores([(1, _result(runs))]), _reference(), _weights(), breakdown_entry="E2", n_boot=50)
    assert report["breakdown_entry"] == "E2"
    with pytest.raises(ValueError, match="unknown entry"):
        cmp.compare(_scores([(1, _result(runs))]), _reference(), _weights(), breakdown_entry="X", n_boot=50)


def test_the_report_marks_partial_and_failed_runs():
    runs = [_record("S", 2, TUNE[0], 0.6), dataclasses.replace(_record("T", 2, TUNE[0], 0.0), error="RuntimeError: x")]
    report = cmp.compare(_scores([(1, _result(runs))]), _reference(), _weights(), n_boot=50)
    buf = io.StringIO()
    cmp.print_report(report, buf)
    text = buf.getvalue()
    assert "PARTIAL: S" in text and "not a result" in text
    assert "WARNING: T" in text and "'crashed': 1" in text
    assert "split: tune" in text and "COCO class" in text


def test_compare_cli_end_to_end(monkeypatch, tmp_path, capsys):
    multi = hi.IOHMultiSeedResult(
        battery_name="mabbob-official-tune",
        problem_kind=mo.OFFICIAL_KIND,
        log_lo=-8.0,
        log_hi=2.0,
        base_seeds=[42],
        results=[_result([_record("S", d, r, 0.6, seed=42) for d in (2, 5) for r in TUNE[:5]])],
        sync_eval=True,
    )
    res_path = tmp_path / "res.json"
    res_path.write_text(multi.to_json())
    ref_path = tmp_path / "ref.json"
    ref_path.write_text(json.dumps(_reference()))
    out_path = tmp_path / "cmp.json"
    monkeypatch.setattr(cmp, "load_table", lambda: mo.OfficialTable((), tuple(map(tuple, _weights())), ()))
    argv = ["compare", str(res_path), "--reference", str(ref_path), "--bootstrap", "50", "--json-out", str(out_path)]
    with pytest.raises(ValueError, match="scores"):  # the small reference covers every row or is refused
        bad = json.loads(ref_path.read_text())
        bad["entries"]["E1"]["aocc"]["2"] = [0.5]
        (tmp_path / "bad.json").write_text(json.dumps(bad))
        cmp.main(["compare", str(res_path), "--reference", str(tmp_path / "bad.json")])
    assert cmp.main(argv) == 0
    assert "E1" in capsys.readouterr().out
    saved = json.loads(out_path.read_text())
    assert saved["splits"]["tune"]["strategies"]["S"]["E1"]["d=5"]["delta"] == pytest.approx(0.1)


def test_compare_scores_the_official_aocc_not_the_stored_one():
    """The stored AOCC is the repo's; comparing it unconverted would flatter us by up to 1/B."""
    rec = _record("S", 2, TUNE[0], 0.5, first=100.0, last=1e-8)
    scores = cmp.OurScores()
    scores.add("res.json", 42, _result([rec]))
    expected = mo.official_from_repo_aocc(0.5, 100.0, 1e-8, rec.budget, -8.0, 2.0)
    assert scores.cells[("S", 2, TUNE[0])] == [pytest.approx(expected, abs=1e-15)]
    assert abs(expected - 0.5) > 1e-5


@pytest.mark.parametrize(("split", "says"), [("heldout", "Conservative for us"), ("tune", "Not conservative")])
def test_the_report_calls_only_a_heldout_comparison_conservative(split, says):
    section = {"rows": 1, "coverage": {}, "strategies": {}, "breakdown": {}}
    report = {"seeds": [42], "fp_env_ids": ["x"], "problems": {}, "splits": {split: section}, "breakdown_entry": "E"}
    out = io.StringIO()
    cmp.print_report(report, out)
    text = out.getvalue()
    assert says in text and "not a competition result" in text and "row 452" in text
    assert ("Not conservative" if split == "heldout" else "Conservative for us") not in text


def test_build_reference_from_logged_data(tmp_path):
    root = tmp_path / "entry"
    root.mkdir()
    scenarios = []
    for dim in (2, 5):
        dat = f"IOHprofiler_f100_DIM{dim}.dat"
        (root / dat).write_text("".join(f"evaluations raw_y\n1 100.0\n{dim} 1e-9\n" for _ in range(mo.N_ROWS)))
        best = {"evals": dim, "y": 1e-9, "x": [0.0] * dim}
        runs = [{"instance": i, "evals": dim, "best": best} for i in range(mo.N_ROWS)]
        scenarios.append({"dimension": dim, "path": dat, "runs": runs})
    (root / "IOHprofiler_f100_ManyAffine.json").write_text(
        json.dumps({"algorithm": {"name": "Alg"}, "scenarios": scenarios})
    )
    doc = cmp.build_reference({"X": root})
    expected = mo.official_aocc_of_points([(1, 100.0), (2, 1e-9)], 4000)
    assert doc["entries"]["X"]["algorithm"] == "Alg"
    assert doc["entries"]["X"]["aocc"]["2"][7] == round(expected, cmp.REFERENCE_DECIMALS)
    # The dumped file reads back the same, with the lists on one line.
    text = cmp.dump_reference(doc)
    assert json.loads(text) == doc and len(text.splitlines()) < 60


# ---------------------------------------------------------------------------
# The worker kind
# ---------------------------------------------------------------------------


@requires_worker
@pytest.mark.parametrize("case", _FIXTURE["replay"], ids=lambda c: f"{c['entry']}-d{c['dim']}-r{c['row']}")
def test_worker_kind_reproduces_logged_best_values(case):
    _official_data()
    from panobbgo.lib.ioh_wrapper import IOHProblem

    p = IOHProblem(mo.OFFICIAL_KIND, case["row"], case["dim"], heldout=mo.split_of(case["row"]) == "heldout")
    try:
        assert p.ioh_instance == case["row"] and p.ioh_problem_id == mo.OFFICIAL_PROBLEM_ID
        assert p.dim == case["dim"]
        value = float(p.eval(np.asarray(case["x"])))  # the logger writes f(x), not f(x) - optimum
        # The entries logged with ioh 0.3.14 / 0.3.17; the worker runs 0.3.22.
        assert value == pytest.approx(case["y"], rel=1e-6)
    finally:
        p.close()


@requires_worker
def test_official_rows_are_not_the_ids_of_the_plain_kind():
    _official_data()
    from panobbgo.lib.ioh_wrapper import IOHProblem

    x = np.array([1.0, -2.0])
    official = IOHProblem(mo.OFFICIAL_KIND, 3, 2)
    plain = IOHProblem("MA-BBOB", 3, 2)
    try:
        assert float(official.eval(x)) != float(plain.eval(x))
    finally:
        official.close()
        plain.close()


@requires_worker
def test_harness_end_to_end_on_official_rows():
    _official_data()
    battery = dataclasses.replace(hi.make_mabbob_official_battery(chunk=(1, 250)), dims=(2,), budget_multiplier=2000)
    spec = [s for s in hi.make_ioh_strategies() if s.name == "RoundRobin_Random"]
    result = hi.run_ioh_harness(spec, battery, base_seed=42, progress=False)
    assert [r.instance for r in result.runs] == list(TUNE[0::250])
    for rec in result.runs:
        assert rec.error is None and rec.n_evals == 4000 and rec.trace_evals[0] == 1
        assert rec.problem_kind == mo.OFFICIAL_KIND and 0.0 < rec.aocc < 1.0
        assert rec.f_opt == 0.0  # the competition scores f(x), not f(x) - optimum
        assert abs(cmp.official_score_of_record(rec) - rec.aocc) <= 1.0 / 4000
    scores = _scores([(42, result)])
    assert scores.strategies == ["RoundRobin_Random"]
    report = cmp.compare(scores, cmp.load_reference(), mo.load_table().weights, n_boot=50)
    assert report["splits"]["tune"]["coverage"]["RoundRobin_Random"][2] == 2
    classes = report["splits"]["tune"]["breakdown"]["RoundRobin_Random"]["by_dominant_class"]
    assert sum(c["n"] for c in classes.values()) == 2


@requires_worker
def test_a_heldout_battery_runs_end_to_end():
    """The flag reaches the problem: without it every run would be refused and recorded as an error."""
    _official_data()
    spec = [s for s in hi.make_ioh_strategies() if s.name == "RoundRobin_Random"]
    result = hi.run_ioh_harness(
        spec, hi.make_mabbob_official_heldout_battery(chunk=(1, 250)), base_seed=42, progress=False
    )
    assert sorted({r.instance for r in result.runs}) == sorted(HELD[0::250])
    assert all(r.error is None and r.n_evals == 2000 * r.dim and r.f_opt == 0.0 for r in result.runs)


@requires_worker
def test_worker_refuses_a_create_without_the_row():
    from panobbgo.lib import ioh_wrapper

    # The wrapper always sends the row; a direct request without it is an error, and the worker stays alive.
    _official_data()
    p = ioh_wrapper.IOHProblem(mo.OFFICIAL_KIND, TUNE[0], 2)
    try:
        with pytest.raises(RuntimeError, match="requires"):
            p._call("create", kind=mo.OFFICIAL_KIND, instance=0, dim=2)
        with pytest.raises(RuntimeError, match="coordinates"):
            p._call("create", kind=mo.OFFICIAL_KIND, instance=0, dim=5, xopt=[0.0], weights=[1.0] * 24, iids=[1] * 24)
        assert p._call("create", kind="MA-BBOB", instance=0, dim=2)["ok"]
    finally:
        p.close()
