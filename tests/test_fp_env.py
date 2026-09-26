# Copyright 2012-2026 Harald Schilly <harald.schilly@gmail.com>
"""The FP environment: the pin (``panobbgo.fp_env`` / ``fp_pin``), the record, and the compare warnings."""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from panobbgo import fp_env
from panobbgo.harness import HarnessConfig, HarnessResult, ProblemStrategyResult
from panobbgo.harness_ioh import IOHHarnessResult, IOHRunRecord, make_quick_battery, run_ioh_harness

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

    disabled = set(fp_env.NUMPY_DISABLED_FEATURES.split())
    dispatch = set(mu.__cpu_dispatch__)
    assert disabled <= dispatch  # an unknown name would warn (fail under -W error) at import
    # Every target above AVX2 is disabled: a new AVX-512 / AVX10 target in a numpy bump fails here.
    assert dispatch - {"X86_V3"} <= disabled, dispatch - {"X86_V3"} - disabled


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
    for key in ("cpu", "machine", "isa", "blas", "numpy", "scipy", "numpy_simd", "libc", "python", "pin", "id"):
        assert key in env
    assert env["blas"] and all({"internal_api", "version", "architecture"} <= set(b) for b in env["blas"])
    assert env["id"] == fp_env.fp_env_id(env) and len(env["id"]) == 12
    json.dumps(env)  # JSON-clean
    # The CPU model and ISA flags do not enter the id; the kernels do.
    assert fp_env.fp_env_id({**env, "cpu": "other", "isa": {"avx512f": True}}) == env["id"]
    other_blas = [{**b, "architecture": "SkylakeX"} for b in env["blas"]]
    assert fp_env.fp_env_id({**env, "blas": other_blas}) != env["id"]
    assert fp_env.fp_env_id({**env, "numpy_simd": ["X86_V3", "X86_V4"]}) != env["id"]
    # libc and the Python minor version are in (the laptop and the runners differ there).
    assert fp_env.fp_env_id({**env, "libc": "glibc 2.39"}) != env["id"] or env["libc"] == "glibc 2.39"
    assert fp_env.fp_env_id({**env, "python": "3.13"}) != env["id"] or env["python"] == "3.13"
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


def _ioh_result(fp_env_id):
    """A one-run IOH result that compares clean against itself (exit 0 under --fail-on-regression)."""
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
        aocc=0.5,
        elapsed_s=0.1,
        seed=42,
    )
    return IOHHarnessResult("quick", "MA-BBOB", -8, 2, [run], sync_eval=True, blas_threads=1, fp_env_id=fp_env_id)


def _composite_result(fp_env_id):
    """A one-pair composite result that compares clean against itself (exit 0 under --fail-on-regression)."""
    psr = ProblemStrategyResult("P", 2, "S", 0.0, 1e-3, 10, [], score=0.5)
    return HarnessResult(HarnessConfig(seed=1, sync_eval=True), "", 0, 0.0, [psr], 0.5, fp_env_id=fp_env_id)


def test_ioh_compare_refuses_to_gate_on_fp_mismatch_only(tmp_path, capsys):
    cli = _load_script(ROOT / "scripts" / "ioh_benchmark.py", "ioh_benchmark_fp_under_test")
    paths = {}
    for key, fp in (("a", "aaaa"), ("a2", "aaaa"), ("b", "bbbb"), ("none", None)):
        paths[key] = tmp_path / f"{key}.json"
        paths[key].write_text(_ioh_result(fp).to_json())
    # Equal ids: the fixture passes the gate ...
    assert cli.main(["compare", str(paths["a"]), str(paths["a2"]), "--fail-on-regression"]) == 0
    assert "FP environment" not in capsys.readouterr().err
    # ... different ids: a warning, and exit 2 only when gating.
    assert cli.main(["compare", str(paths["a"]), str(paths["b"])]) == 0
    assert "FP environment mismatch" in capsys.readouterr().err
    assert cli.main(["compare", str(paths["a"]), str(paths["b"]), "--fail-on-regression"]) == 2
    assert "refuses to gate" in capsys.readouterr().err
    # An older file without the record: a warning only, until the pinned references exist.
    assert cli.main(["compare", str(paths["a"]), str(paths["none"]), "--fail-on-regression"]) == 0
    err = capsys.readouterr().err
    assert "unknown on one side" in err and "refuses to gate" not in err


def test_composite_compare_refuses_to_gate_on_fp_mismatch_only(tmp_path, capsys):
    import benchmark_harness

    paths = {}
    for key, fp in (("a", "aaaa"), ("a2", "aaaa"), ("b", "bbbb"), ("none", None)):
        paths[key] = tmp_path / f"{key}.json"
        _composite_result(fp).save(str(paths[key]))
    assert benchmark_harness.main(["compare", str(paths["a"]), str(paths["a2"]), "--fail-on-regression"]) == 0
    assert "FP environment" not in capsys.readouterr().err
    assert benchmark_harness.main(["compare", str(paths["a"]), str(paths["b"])]) == 0
    assert "FP environment mismatch" in capsys.readouterr().err
    assert benchmark_harness.main(["compare", str(paths["a"]), str(paths["b"]), "--fail-on-regression"]) == 2
    assert "refuses to gate" in capsys.readouterr().err
    assert benchmark_harness.main(["compare", str(paths["a"]), str(paths["none"]), "--fail-on-regression"]) == 0
    err = capsys.readouterr().err
    assert "unknown on one side" in err and "refuses to gate" not in err


def test_current_never_raises(monkeypatch):
    def boom():
        raise RuntimeError("no threadpoolctl")

    monkeypatch.setattr(fp_env, "collect", boom)
    with pytest.warns(RuntimeWarning, match="could not record the FP environment"):
        assert fp_env.current() == {"fp_env": None, "fp_env_id": None}


def test_family_harness_records_the_fp_env():
    from panobbgo.harness_families import make_families_battery, run_family_harness
    from panobbgo.harness_ioh import make_ioh_strategies

    spec = [s for s in make_ioh_strategies() if s.name == "RoundRobin_Random"]
    res = run_family_harness(
        spec, make_families_battery(dims=(2,), n_instances=1)[:1], budget_multiplier=3, progress=False
    )
    assert res.fp_env is not None and res.fp_env_id == fp_env.current_id()


def test_family_screen_rows_carry_the_fp_env_id(tmp_path):
    out = tmp_path / "rows.json"
    r = subprocess.run(
        [
            sys.executable,
            str(ROOT / "benchmarks" / "family_screen.py"),
            str(out),
            "42",
            "preset=free",
            "dims=2",
            "ninst=1",
            "bm=3",
            "specs=CMAES_alone",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert r.returncode == 0, r.stderr[-2000:]
    rows = json.loads(out.read_text())
    ids = {row.get("fp_env_id") for row in rows}
    assert rows and len(ids) == 1 and None not in ids
    # The screen ran in a fresh process that pinned itself: its id is the pinned one.
    if X86:
        env = _clean_env(**fp_env.PIN_ENV)
        check = _python("from panobbgo import fp_env; print(fp_env.current_id())", env)
        assert check.returncode == 0, check.stderr
        assert ids == {check.stdout.strip()} or os.environ.get(fp_env.OPT_OUT_VAR) == "0"


#: Entry points: a pinned import must come before any non-stdlib import (numpy loads there).
ENTRY_POINTS = [
    "benchmark_harness.py",
    "scripts/ioh_benchmark.py",
    "scripts/ioh_smoke.py",
    "benchmarks/family_screen.py",
    "benchmarks/portfolio_screen.py",
    "benchmarks/oracle.py",
    "benchmarks/meta_screen.py",
    "benchmarks/arm_sweep.py",
    "benchmarks/np_accept.py",
    "benchmarks/coco_benchmark.py",
]


def _top_level_imports(path: Path):
    """``(line, module)`` of every module-level import statement, in order (``if`` blocks included)."""
    tree = ast.parse(path.read_text())
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.extend((node.lineno, a.name) for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            out.append((node.lineno, node.module))
    # Only module level and module-level ``if`` blocks, not function bodies.
    funcs = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    inside = {ln for f in funcs for ln in range(f.lineno, (f.end_lineno or f.lineno) + 1)}
    return sorted((ln, m) for ln, m in out if ln not in inside)


@pytest.mark.parametrize("rel", ENTRY_POINTS)
def test_entry_points_import_fp_pin_before_anything_that_loads_numpy(rel):
    imports = _top_level_imports(ROOT / rel)
    pin = [ln for ln, m in imports if m == "panobbgo.fp_pin"]
    assert pin, f"{rel} does not import panobbgo.fp_pin"
    stdlib = set(sys.stdlib_module_names) | {"__future__"}
    first_other = min(
        (ln for ln, m in imports if m.split(".")[0] not in stdlib and m != "panobbgo.fp_pin"), default=None
    )
    assert first_other is None or pin[0] < first_other, (
        f"{rel}: panobbgo.fp_pin (line {pin[0]}) after line {first_other}"
    )


def test_rebaseline_loads_nothing_but_stdlib_at_module_level():
    """``rebaseline.py`` pins in ``main()``; nothing above it may load numpy first."""
    stdlib = set(sys.stdlib_module_names) | {"__future__"}
    imports = _top_level_imports(ROOT / "scripts" / "rebaseline.py")
    assert all(m.split(".")[0] in stdlib for _, m in imports), imports


@pytest.mark.skipif(not X86, reason="the pin applies on x86-64 with AVX2 only")
def test_import_panobbgo_pins_every_variable():
    """The package's import-time side effect: the whole pin, torch / MKL / oneDNN included."""
    code = "import os, panobbgo, json; print(json.dumps({k: os.environ.get(k) for k in panobbgo.fp_env.PIN_ENV}))"
    r = _python(code, _clean_env())
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout) == fp_env.PIN_ENV
    r = _python(code, _clean_env(PANOBBGO_FP_PIN="0"))
    assert set(json.loads(r.stdout).values()) == {None}


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
