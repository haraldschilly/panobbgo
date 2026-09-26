# Copyright 2012-2026 Harald Schilly <harald.schilly@gmail.com>
"""The FP environment: the pin (``panobbgo.fp_env`` / ``fp_pin``), the record, and the compare warnings."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from panobbgo import fp_env
from panobbgo.harness import HarnessConfig, HarnessResult
from panobbgo.harness_ioh import IOHHarnessResult, run_ioh_harness, make_quick_battery

ROOT = Path(__file__).resolve().parent.parent
X86 = fp_env.pin_supported()


def _load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _clean_env(**extra: str) -> dict:
    """``os.environ`` without the pin variables (CI sets them job-wide), plus ``extra``."""
    env = {k: v for k, v in os.environ.items() if k not in (*fp_env.PIN_ENV, fp_env.OPT_OUT_VAR)}
    env.update(extra)
    return env


def _python(code: str, env: dict, *flags: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, *flags, "-c", code], env=env, cwd=ROOT, capture_output=True, text=True, timeout=120
    )


@pytest.mark.skipif(not X86, reason="the numpy feature names are x86-64 dispatch targets")
def test_disabled_numpy_features_are_dispatch_targets_of_this_numpy():
    from numpy._core import _multiarray_umath as mu

    assert set(fp_env.NUMPY_DISABLED_FEATURES.split()) <= set(mu.__cpu_dispatch__)


@pytest.mark.skipif(not X86, reason="the pin applies on x86-64 with AVX2 only")
def test_pinned_numpy_imports_cleanly_under_W_error():
    """An unknown feature name makes numpy warn at import; ``-W error`` turns that into a failure."""
    r = _python("import numpy, scipy.linalg", _clean_env(**fp_env.PIN_ENV), "-W", "error")
    assert r.returncode == 0, r.stderr


@pytest.mark.skipif(not X86, reason="the pin applies on x86-64 with AVX2 only")
def test_fp_pin_sets_the_env_before_numpy_and_honours_the_opt_out():
    code = (
        "import sys, os, panobbgo.fp_pin as p; import numpy; from panobbgo import fp_env; e = fp_env.collect(); "
        "print(p.PINNED, os.environ.get('OPENBLAS_CORETYPE'), os.environ.get('NPY_DISABLE_CPU_FEATURES'), "
        "sorted({b['architecture'] for b in e['blas']}), e['numpy_simd'])"
    )
    r = _python(code, _clean_env())
    assert r.returncode == 0, r.stderr
    assert r.stdout.startswith(f"True Haswell {fp_env.NUMPY_DISABLED_FEATURES}")
    assert r.stdout.rstrip().endswith("['Haswell'] ['X86_V3']")  # the AVX2 kernels, no X86_V4 loops
    r = _python(code, _clean_env(PANOBBGO_FP_PIN="0"))
    assert r.returncode == 0, r.stderr
    assert r.stdout.startswith("False None None")


def test_pin_leaves_the_env_alone_once_numpy_is_loaded(monkeypatch):
    import numpy  # noqa: F401  # pyright: ignore[reportUnusedImport]

    for var in fp_env.PIN_ENV:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv(fp_env.OPT_OUT_VAR, raising=False)
    assert fp_env.pin_fp_env() is False
    assert all(var not in os.environ for var in fp_env.PIN_ENV)


@pytest.mark.parametrize("value,requested", [(None, True), ("1", True), ("0", False), ("off", False), ("No", False)])
def test_pin_requested(value, requested):
    env = {} if value is None else {fp_env.OPT_OUT_VAR: value}
    assert fp_env.pin_requested(env) is requested


def test_child_env_carries_the_pin_unless_opted_out():
    base = {"PATH": "/bin"}
    if X86:
        assert fp_env.child_env(base) == {**base, **fp_env.PIN_ENV}
    else:
        assert fp_env.child_env(base) == base
    off = {**base, fp_env.OPT_OUT_VAR: "0"}
    assert fp_env.child_env(off) == off


def test_collect_and_id():
    env = fp_env.collect()
    for key in ("cpu", "machine", "isa", "blas", "numpy", "scipy", "numpy_simd", "libc", "pin", "id"):
        assert key in env
    assert env["blas"] and all({"internal_api", "version", "architecture"} <= set(b) for b in env["blas"])
    assert env["id"] == fp_env.fp_env_id(env) and len(env["id"]) == 12
    json.dumps(env)  # JSON-clean
    # The CPU model and ISA flags do not enter the id; the kernels do.
    assert fp_env.fp_env_id({**env, "cpu": "other", "isa": {"avx512f": True}}) == env["id"]
    other_blas = [{**b, "architecture": "SkylakeX"} for b in env["blas"]]
    assert fp_env.fp_env_id({**env, "blas": other_blas}) != env["id"]
    assert fp_env.fp_env_id({**env, "numpy_simd": ["X86_V3", "X86_V4"]}) != env["id"]
    assert fp_env.fp_env_id(None) is None


def test_mismatch_texts():
    assert fp_env.mismatch("a", "a", "x", "y") is None
    assert "FP environment mismatch" in (fp_env.mismatch("a", "b", "x", "y") or "")
    assert "unknown on one side" in (fp_env.mismatch(None, "b", "x", "y") or "")
    assert fp_env.known_mismatch("a", "b") and not fp_env.known_mismatch(None, "b")


def test_results_record_the_fp_env_and_round_trip():
    res = run_ioh_harness([], make_quick_battery(), progress=False)
    assert res.fp_env is not None and res.fp_env_id == res.fp_env["id"]
    back = IOHHarnessResult.from_dict(json.loads(res.to_json()))
    assert back.fp_env_id == res.fp_env_id and back.fp_env == res.fp_env
    comp = HarnessResult(HarnessConfig(), "", 0, 0.0, [], 0.5, **fp_env.current())
    loaded = HarnessResult._from_dict(json.loads(json.dumps(comp.to_dict())))
    assert loaded.fp_env_id == comp.fp_env_id and loaded.fp_env == comp.fp_env
    old = {k: v for k, v in comp.to_dict().items() if not k.startswith("fp_env")}
    assert HarnessResult._from_dict(old).fp_env_id is None


def test_ioh_compare_warns_and_refuses_to_gate_on_fp_mismatch(tmp_path, capsys):
    cli = _load_script(ROOT / "scripts" / "ioh_benchmark.py", "ioh_benchmark_fp_under_test")
    a = run_ioh_harness([], make_quick_battery(), progress=False)
    b = dataclasses.replace(a, fp_env_id="other")
    pa, pb = tmp_path / "a.json", tmp_path / "b.json"
    pa.write_text(a.to_json())
    pb.write_text(b.to_json())
    assert cli.main(["compare", str(pa), str(pa)]) == 0
    assert "FP environment" not in capsys.readouterr().err
    cli.main(["compare", str(pa), str(pb)])
    assert "FP environment mismatch" in capsys.readouterr().err
    assert cli.main(["compare", str(pa), str(pb), "--fail-on-regression"]) == 2
    assert "refuses to gate" in capsys.readouterr().err
    # An older file without the record: a warning, not a refusal.
    pb.write_text(dataclasses.replace(a, fp_env_id=None).to_json())
    cli.main(["compare", str(pa), str(pb), "--fail-on-regression"])
    err = capsys.readouterr().err
    assert "unknown on one side" in err and "refuses to gate" not in err


def test_composite_compare_warns_and_refuses_to_gate_on_fp_mismatch(tmp_path, capsys):
    import benchmark_harness

    a = HarnessResult(HarnessConfig(seed=1), "", 0, 0.0, [], 0.5, fp_env_id="aaaa")
    b = dataclasses.replace(a, fp_env_id="bbbb")
    pa, pb = tmp_path / "a.json", tmp_path / "b.json"
    a.save(str(pa))
    b.save(str(pb))
    assert benchmark_harness.main(["compare", str(pa), str(pb)]) == 0
    assert "FP environment mismatch" in capsys.readouterr().err
    assert benchmark_harness.main(["compare", str(pa), str(pb), "--fail-on-regression"]) == 2
    assert "refuses to gate" in capsys.readouterr().err


def test_screen_rows_warn_on_mixed_fp(tmp_path, capsys):
    sys.path.insert(0, str(ROOT / "benchmarks"))
    try:
        import _screen
    finally:
        sys.path.pop(0)
    rows = [{"seed": 42, "s": "x", "fp_env_id": "a"}, {"seed": 7, "s": "x", "fp_env_id": "a"}]
    assert _screen.warn_mixed_fp(rows, "f.json") == ["a"]
    assert capsys.readouterr().err == ""
    assert _screen.warn_mixed_fp(rows + [{"seed": 3, "s": "x"}], "f.json") == [None, "a"]
    assert "mixes FP environments" in capsys.readouterr().err


def test_fp_check_script(tmp_path, capsys):
    check = _load_script(ROOT / "scripts" / "fp_check.py", "fp_check_under_test")
    rows = [{"seed": 42, "s": "x", "aocc": 0.5, "fp_env_id": "a"}]
    for job, fp in (("job1", "a"), ("job2", "b")):
        d = tmp_path / f"fp-check-{job}" / job
        d.mkdir(parents=True)
        (d / "rows.json").write_text(json.dumps([{**r, "fp_env_id": fp} for r in rows]))
        (d / "fp_env.json").write_text(json.dumps({"cpu": job, "id": fp, "blas": [], "isa": {}}))
    assert check.main([str(tmp_path)]) == 0  # fp_env_id is not part of the numbers
    out = capsys.readouterr().out
    assert "OK: all 2 jobs" in out and "job2" in out
    (tmp_path / "fp-check-job2" / "job2" / "rows.json").write_text(json.dumps([{**rows[0], "aocc": 0.6}]))
    assert check.main([str(tmp_path)]) == 1
    assert "differing_rows=1" in capsys.readouterr().out


def test_pool_workers_share_the_parents_fp_env():
    """Spawned ``TaskPool`` workers inherit the environment: the same kernels as this process."""
    from panobbgo.local_run import TaskPool

    with TaskPool(2, niceness=None, min_free_gb=0.0) as pool:
        ids = pool.map(fp_env.current_id, [{}, {}])
    assert set(ids) == {fp_env.collect()["id"]}
